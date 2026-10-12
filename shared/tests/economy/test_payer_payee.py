"""Payer and Payee against a real ledger served in-process over HTTP.

No A2A here: the payee's `make_quote` stands in for the specialist's quote
reply, so these tests pin the money flow on its own.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterator, Callable

import httpx
import pytest
import pytest_asyncio

from economy import Ledger, ReservationNotHeld, ReservationState
from economy.client import LedgerClient
from economy.payee import Payee, PaymentRefused
from economy.payer import Authorization, Payer, Rejection
from economy.policy import RejectReason, SpendPolicy
from economy.pricing import TRAVEL_PRICE_SHEET, PricingPolicy
from economy.protocol import QuoteDeclined, QuoteOffer, QuoteRequest
from economy.service import Credentials, create_app
from shared.agent_runner import RunUsage

from ._fixtures import ALLOWANCE_MC, FakeClock, seed

KEYS = {"concierge": "kc", "travel": "kt", "local": "kl", "transit": "kx"}
BASE_URL = "http://ledger"
# 20k in + 2k out tokens = 700 mc of LLM; search_flights = 200 mc of API.
USAGE = RunUsage(
    requests=1, input_tokens=20_000, output_tokens=2_000, tool_calls={"search_flights": 1}
)

ClientFor = Callable[[str], LedgerClient]


@pytest_asyncio.fixture
async def ledger() -> AsyncIterator[Ledger]:
    led = await Ledger.open(":memory:", clock=FakeClock())
    try:
        await seed(led)
        await led.ensure_wallet("concierge")  # no daily cap unless a test sets one
        yield led
    finally:
        await led.close()


@pytest_asyncio.fixture
async def client_for(ledger: Ledger) -> AsyncIterator[ClientFor]:
    app = create_app(ledger=ledger, credentials=Credentials(agent_keys=KEYS))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE_URL) as http:
        yield lambda agent: LedgerClient(BASE_URL, agent_id=agent, agent_key=KEYS[agent], http=http)


@pytest.fixture
def payee(client_for: ClientFor) -> Payee:
    return Payee(client_for("travel"), PricingPolicy(TRAVEL_PRICE_SHEET))


def _payer(client_for: ClientFor, *, budget: int = 8_000, call_cap: int = 3_000) -> Payer:
    return Payer(
        client_for("concierge"),
        SpendPolicy(max_call_mc=call_cap),
        trip_id="trip-1",
        trip_budget_mc=budget,
    )


def _quoter(payee: Payee, calls: list[int | None] | None = None):  # type: ignore[no-untyped-def]
    """request_quote backed by the real payee, recording each counter limit."""

    async def request_quote(max_mc: int | None) -> QuoteOffer | QuoteDeclined:
        if calls is not None:
            calls.append(max_mc)
        reply = await payee.make_quote(
            QuoteRequest(payer="concierge", max_mc=max_mc), skill_id="search_flights"
        )
        if isinstance(reply, QuoteDeclined):
            return reply
        return QuoteOffer(reply.quote_id, reply.skill_id, reply.amount_mc, reply.payee)

    return request_quote


# ---------- payee ----------


async def test_payee_quotes_its_list_ceiling(payee: Payee) -> None:
    quote = await payee.make_quote(QuoteRequest(payer="concierge"), skill_id="search_flights")
    assert not isinstance(quote, QuoteDeclined)
    assert (quote.payer, quote.payee, quote.amount_mc) == ("concierge", "travel", 1_500)


async def test_payee_accepts_a_counter_down_to_its_base(payee: Payee) -> None:
    quote = await payee.make_quote(
        QuoteRequest(payer="concierge", max_mc=1_000), skill_id="search_flights"
    )
    assert not isinstance(quote, QuoteDeclined)
    assert quote.amount_mc == 1_000
    declined = await payee.make_quote(
        QuoteRequest(payer="concierge", max_mc=399), skill_id="search_flights"
    )
    assert isinstance(declined, QuoteDeclined)
    assert declined.reason == "below_base"


async def test_payee_declines_when_the_ledger_refuses(payee: Payee) -> None:
    declined = await payee.make_quote(QuoteRequest(payer="nobody"), skill_id="search_flights")
    assert isinstance(declined, QuoteDeclined)
    assert declined.reason == "unknown_wallet"


async def test_payee_must_act_as_its_sheet_wallet(client_for: ClientFor) -> None:
    with pytest.raises(ValueError):
        Payee(client_for("local"), PricingPolicy(TRAVEL_PRICE_SHEET))


async def test_payee_accept_binds_and_capture_charges_metered_cost(
    payee: Payee, client_for: ClientFor
) -> None:
    payer = _payer(client_for)
    auth = await payer.authorize(base_mc=400, request_quote=_quoter(payee))
    assert isinstance(auth, Authorization)

    bound = await payee.accept(
        auth.reservation.reservation_id, skill_id="search_flights", task_id="task-1"
    )
    assert bound.a2a_task_id == "task-1"
    charge = await payee.capture(bound, USAGE)
    assert (charge.base_mc, charge.llm_mc, charge.api_mc, charge.charged_mc) == (
        400,
        700,
        200,
        1_300,
    )
    assert (await client_for("travel").get_wallet()).available_mc == 1_300


async def test_payee_refuses_bad_payment(
    payee: Payee, client_for: ClientFor, ledger: Ledger
) -> None:
    payer = _payer(client_for)
    auth = await payer.authorize(base_mc=400, request_quote=_quoter(payee))
    assert isinstance(auth, Authorization)
    rid = auth.reservation.reservation_id

    with pytest.raises(PaymentRefused, match="pays for search_flights"):
        await payee.accept(rid, skill_id="search_hotels", task_id="t")
    with pytest.raises(PaymentRefused, match="could not be verified"):
        await payee.accept("r_missing", skill_id=None, task_id="t")

    local_sheet = dataclasses.replace(TRAVEL_PRICE_SHEET, payee_wallet="local")
    local_payee = Payee(client_for("local"), PricingPolicy(local_sheet))
    with pytest.raises(PaymentRefused, match="could not be verified|does not pay"):
        await local_payee.accept(rid, skill_id=None, task_id="t")

    await payee.accept(rid, skill_id=None, task_id="t1")
    with pytest.raises(PaymentRefused):  # one reservation pays for one task
        await payee.accept(rid, skill_id=None, task_id="t2")

    await ledger.release(rid)
    with pytest.raises(PaymentRefused, match="released"):
        await payee.accept(rid, skill_id=None, task_id="t1")


async def test_payee_release_never_raises(payee: Payee) -> None:
    await payee.release("r_missing")  # logged, not raised


# ---------- payer ----------


async def test_payer_reserves_and_settles_a_captured_call(
    payee: Payee, client_for: ClientFor
) -> None:
    payer = _payer(client_for)
    auth = await payer.authorize(base_mc=400, request_quote=_quoter(payee))
    assert isinstance(auth, Authorization)
    assert auth.reservation.trip_id == "trip-1"
    assert payer.committed_mc == 1_500

    bound = await payee.accept(auth.reservation.reservation_id, skill_id=None, task_id="t")
    await payee.capture(bound, USAGE)
    settlement = await payer.settle(auth)

    assert (settlement.state, settlement.charged_mc) == (ReservationState.CAPTURED, 1_300)
    assert payer.committed_mc == 1_300
    assert payer.trip_remaining_mc == 8_000 - 1_300
    assert (await client_for("concierge").get_wallet()).available_mc == ALLOWANCE_MC - 1_300


async def test_payer_releases_a_hold_nobody_settled(payee: Payee, client_for: ClientFor) -> None:
    payer = _payer(client_for)
    auth = await payer.authorize(base_mc=400, request_quote=_quoter(payee))
    assert isinstance(auth, Authorization)

    settlement = await payer.settle(auth)
    assert (settlement.state, settlement.charged_mc) == (ReservationState.RELEASED, 0)
    assert payer.committed_mc == 0
    # The specialist finishing late cannot capture any more.
    bound = await client_for("travel").get_reservation(auth.reservation.reservation_id)
    with pytest.raises(ReservationNotHeld):
        await payee.capture(bound, USAGE)


async def test_payer_counters_once_at_the_call_cap(payee: Payee, client_for: ClientFor) -> None:
    calls: list[int | None] = []
    payer = _payer(client_for, call_cap=1_000)
    auth = await payer.authorize(base_mc=400, request_quote=_quoter(payee, calls))
    assert isinstance(auth, Authorization)
    assert calls == [None, 1_000]
    assert auth.counter_offered
    assert auth.reservation.amount_mc == 1_000


async def test_payer_rejects_without_reserving_when_the_trip_is_spent(
    payee: Payee, client_for: ClientFor
) -> None:
    payer = _payer(client_for, budget=300)
    rejection = await payer.authorize(base_mc=400, request_quote=_quoter(payee))
    assert isinstance(rejection, Rejection)
    assert rejection.reason is RejectReason.TRIP_BUDGET_EXHAUSTED
    assert rejection.quote_mc == 1_500
    wallet = await client_for("concierge").get_wallet()
    assert (wallet.available_mc, wallet.reserved_mc) == (ALLOWANCE_MC, 0)


async def test_payer_rejects_when_the_specialist_declines(
    payee: Payee, client_for: ClientFor
) -> None:
    # Call cap below the specialist's base: the counter is never offered.
    payer = _payer(client_for, call_cap=100)
    rejection = await payer.authorize(base_mc=400, request_quote=_quoter(payee))
    assert isinstance(rejection, Rejection)
    assert rejection.reason is RejectReason.OVER_CALL_CAP

    async def declines(max_mc: int | None) -> QuoteDeclined:
        return QuoteDeclined("busy", "try later")

    rejection = await _payer(client_for).authorize(base_mc=400, request_quote=declines)
    assert isinstance(rejection, Rejection)
    assert rejection.reason is RejectReason.DECLINED


async def test_payer_respects_the_daily_cap(
    payee: Payee, client_for: ClientFor, ledger: Ledger
) -> None:
    await ledger.ensure_wallet("concierge", daily_cap_mc=300)
    rejection = await _payer(client_for).authorize(base_mc=400, request_quote=_quoter(payee))
    assert isinstance(rejection, Rejection)
    assert rejection.reason is RejectReason.DAILY_CAP


async def test_parallel_calls_never_overspend_the_trip(payee: Payee, client_for: ClientFor) -> None:
    payer = _payer(client_for, budget=3_000)
    results = await asyncio.gather(
        *(payer.authorize(base_mc=400, request_quote=_quoter(payee)) for _ in range(3))
    )
    authorized = [r for r in results if isinstance(r, Authorization)]
    assert len(authorized) == 2  # 2 x 1500 fills the 3000 budget exactly
    assert payer.committed_mc == 3_000
    assert payer.trip_remaining_mc == 0


async def test_unreachable_ledger_fails_closed() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(refuse)) as http:
        ledger = LedgerClient(BASE_URL, agent_id="concierge", agent_key="kc", http=http)
        payer = Payer(ledger, SpendPolicy(max_call_mc=3_000), trip_id="t", trip_budget_mc=8_000)

        async def offer(max_mc: int | None) -> QuoteOffer:
            return QuoteOffer("q_1", "search_flights", 1_000, "travel")

        rejection = await payer.authorize(base_mc=400, request_quote=offer)
        assert isinstance(rejection, Rejection)
        assert rejection.reason is RejectReason.BILLING_UNAVAILABLE


async def test_unconfirmed_settlement_keeps_counting_the_hold(
    payee: Payee, client_for: ClientFor
) -> None:
    payer = _payer(client_for)
    auth = await payer.authorize(base_mc=400, request_quote=_quoter(payee))
    assert isinstance(auth, Authorization)

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(refuse)) as http:
        payer.ledger = LedgerClient(BASE_URL, agent_id="concierge", agent_key="kc", http=http)
        settlement = await payer.settle(auth)
    assert settlement.state is None
    assert payer.committed_mc == 1_500


async def test_payer_validation(client_for: ClientFor) -> None:
    with pytest.raises(ValueError):
        _payer(client_for, budget=-1)
