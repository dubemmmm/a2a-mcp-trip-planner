"""Paid delegation end to end: Concierge -> A2A -> priced specialist -> ledger.

A real priced specialist (FakeModel instead of OpenAI) runs under uvicorn;
the ledger runs in-process behind its HTTP service; the Concierge side is
the real `_delegate` with a `Payer`. Every test checks where the credits
ended up.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass

import httpx
import pytest
import pytest_asyncio
from a2a.types import AgentSkill
from agents import Agent
from agents.usage import Usage

from _helpers.fake_model import FakeModel
from a2a_servers.scaffolding import build_a2a_app, build_agent_card
from agents_concierge.delegation import DelegationError, _delegate
from economy import Ledger, ReservationState
from economy.card import pricing_extension
from economy.client import LedgerClient
from economy.payee import Payee
from economy.payer import Payer
from economy.policy import SpendPolicy
from economy.pricing import TRAVEL_PRICE_SHEET, PricingPolicy
from economy.service import Credentials, create_app
from shared.a2a_client import AgentDiscovery
from shared.agent_events import AgentEvent, AgentEventQueue

from ..a2a_servers.test_live_a2a import _serve
from ..economy._fixtures import ALLOWANCE_MC, seed

PORT = 54332
LEDGER_URL = "http://ledger"
KEYS = {"concierge": "kc", "travel": "kt", "local": "kl", "transit": "kx"}
SKILL = "search_flights"
# 20k in + 2k out = 700 mc LLM; base 400 -> charge 1100 against a 1500 quote.
USAGE = Usage(requests=1, input_tokens=20_000, output_tokens=2_000, total_tokens=22_000)


@dataclass
class World:
    ledger: Ledger
    client_for: Callable[[str], LedgerClient]
    fake: FakeModel
    url: str


StartWorld = Callable[..., AbstractAsyncContextManager[World]]


@pytest_asyncio.fixture
async def ledger() -> AsyncIterator[Ledger]:
    led = await Ledger.open(":memory:")
    try:
        await seed(led)
        await led.ensure_wallet("concierge")  # no daily cap
        yield led
    finally:
        await led.close()


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch, ledger: Ledger) -> StartWorld:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ENVIRONMENT", "test")

    @contextlib.asynccontextmanager
    async def start(
        *, responses: tuple[str, ...] = ("Flights: ATL->LIS $612",)
    ) -> AsyncIterator[World]:
        ledger_app = create_app(ledger=ledger, credentials=Credentials(agent_keys=KEYS))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=ledger_app), base_url=LEDGER_URL
        ) as http:

            def client_for(agent: str) -> LedgerClient:
                return LedgerClient(LEDGER_URL, agent_id=agent, agent_key=KEYS[agent], http=http)

            fake = FakeModel(responses=list(responses), usage=USAGE)
            agent = Agent[None](name="travel", instructions="find flights", model=fake)
            card = build_agent_card(
                name="travel",
                description="priced test specialist",
                url=f"http://localhost:{PORT}",
                skills=[AgentSkill(id=SKILL, name=SKILL, description=SKILL, tags=["t"])],
                extensions=[pricing_extension(TRAVEL_PRICE_SHEET, ledger_url=LEDGER_URL)],
            )
            pricing = PricingPolicy(TRAVEL_PRICE_SHEET)
            app = build_a2a_app(
                agent, card, pricing=pricing, payee=Payee(client_for("travel"), pricing)
            )
            async with _serve(app, PORT) as url:
                yield World(ledger=ledger, client_for=client_for, fake=fake, url=url)

    return start


def _payer(w: World, *, budget: int = 8_000, call_cap: int = 3_000) -> Payer:
    return Payer(
        w.client_for("concierge"),
        SpendPolicy(max_call_mc=call_cap),
        trip_id="trip-e2e",
        trip_budget_mc=budget,
    )


async def _call(w: World, payer: Payer | None, queue: AgentEventQueue | None = None) -> str:
    async with AgentDiscovery() as discovery:
        return await _delegate(
            request="find flights ATL to LIS",
            agent_url=w.url,
            skill_id=SKILL,
            discovery=discovery,
            client_factory=None,
            timeout_seconds=10.0,
            event_queue=queue,
            event_agent_name="travel",
            payer=payer,
        )


def _drain(queue: AgentEventQueue) -> list[AgentEvent]:
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


async def _balances(w: World) -> tuple[int, int]:
    concierge = await w.ledger.get_wallet("concierge")
    travel = await w.ledger.get_wallet("travel")
    assert concierge.reserved_mc == 0, "no hold may outlive the call"
    return concierge.available_mc, travel.available_mc


async def test_paid_call_moves_credits_and_returns_a_receipt(world: StartWorld) -> None:
    queue: AgentEventQueue = asyncio.Queue()
    async with world() as w:
        payer = _payer(w)
        text = await _call(w, payer, queue)
        balances = await _balances(w)
        (transfer, *_) = [
            t for t in await w.ledger.journal(trip_id="trip-e2e") if t.kind.value == "capture"
        ]

    assert "ATL->LIS $612" in text
    assert text.endswith("[charged 1.10 credits · trip remaining 6.90 credits]")
    assert balances == (ALLOWANCE_MC - 1_100, 1_100)
    reservation = await w.ledger.get_reservation(transfer.reservation_id or "")
    assert reservation.state is ReservationState.CAPTURED
    assert reservation.a2a_task_id  # bound to the task that did the work
    kinds = [e.kind for e in _drain(queue)]
    for expected in ("payment.quoted", "payment.reserved", "payment.captured"):
        assert expected in kinds
    assert kinds.index("payment.reserved") < kinds.index("payment.captured")


async def test_counter_offer_caps_the_charge(world: StartWorld) -> None:
    async with world() as w:
        text = await _call(w, _payer(w, call_cap=1_000))
        balances = await _balances(w)
    # Quote 1500 -> counter at 1000 -> metered 1100 capped at 1000.
    assert "[charged 1.00 credits" in text
    assert balances == (ALLOWANCE_MC - 1_000, 1_000)


async def test_unaffordable_call_is_not_made(world: StartWorld) -> None:
    queue: AgentEventQueue = asyncio.Queue()
    async with world() as w:
        text = await _call(w, _payer(w, budget=300), queue)
        balances = await _balances(w)

    assert text.startswith("NOT CALLED: the travel specialist was not called because")
    assert "trip's remaining budget" in text
    assert w.fake.calls == []  # the specialist never ran
    assert balances == (ALLOWANCE_MC, 0)
    rejected = [e for e in _drain(queue) if e.kind == "payment.rejected"]
    assert rejected and rejected[0].data["reason"] == "trip_budget_exhausted"


async def test_priced_specialist_without_a_wallet_is_not_called(world: StartWorld) -> None:
    async with world() as w:
        text = await _call(w, payer=None)
        balances = await _balances(w)
    assert "no wallet configured" in text
    assert w.fake.calls == []
    assert balances == (ALLOWANCE_MC, 0)


async def test_failed_work_returns_the_hold(world: StartWorld) -> None:
    queue: AgentEventQueue = asyncio.Queue()
    async with world(responses=()) as w:  # FakeModel has nothing to say -> run fails
        payer = _payer(w)
        with pytest.raises(DelegationError):
            await _call(w, payer, queue)
        balances = await _balances(w)

    assert balances == (ALLOWANCE_MC, 0)
    assert payer.committed_mc == 0
    assert "payment.released" in [e.kind for e in _drain(queue)]


async def test_task_without_payment_is_refused(world: StartWorld) -> None:
    from a2a.client import ClientFactory
    from a2a.client.client import ClientConfig

    from agents_concierge.delegation import _submit_and_await

    async with world() as w:
        async with AgentDiscovery() as discovery, httpx.AsyncClient(timeout=10) as http:
            card = await discovery.discover(w.url)
            client = ClientFactory(ClientConfig(httpx_client=http)).create(card)
            with pytest.raises(DelegationError, match="reported failure"):
                await _submit_and_await(
                    client,
                    card,
                    request="free flights please",
                    skill_id=SKILL,
                    display_name="travel",
                    timeout_seconds=10.0,
                    event_queue=None,
                    metadata=None,
                )
        balances = await _balances(w)
    assert w.fake.calls == []
    assert balances == (ALLOWANCE_MC, 0)


async def test_cancelling_a_paid_task_returns_its_hold(world: StartWorld) -> None:
    from unittest.mock import AsyncMock, MagicMock

    from a2a_servers.scaffolding import OpenAIAgentExecutor
    from economy.protocol import QuoteRequest

    async with world() as w:
        pricing = PricingPolicy(TRAVEL_PRICE_SHEET)
        payee = Payee(w.client_for("travel"), pricing)
        quote = await payee.make_quote(QuoteRequest(payer="concierge"), skill_id=SKILL)
        reservation = await w.client_for("concierge").reserve(quote.quote_id)  # type: ignore[union-attr]

        executor = OpenAIAgentExecutor(Agent[None](name="x", model=w.fake), payee=payee)
        executor._holds["task-9"] = reservation.reservation_id  # as if mid-run
        context = MagicMock(task_id="task-9", context_id="ctx")
        await executor.cancel(context, AsyncMock())

        settled = await w.ledger.get_reservation(reservation.reservation_id)
        balances = await _balances(w)
    assert settled.state is ReservationState.RELEASED
    assert balances == (ALLOWANCE_MC, 0)
    assert executor._holds == {}
