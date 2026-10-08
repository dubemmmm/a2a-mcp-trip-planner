"""The ledger HTTP service and LedgerClient, driven over httpx.ASGITransport."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from economy import (
    InsufficientFunds,
    InvalidRequest,
    Ledger,
    LedgerAuthError,
    LedgerError,
    LedgerForbidden,
    LedgerUnavailable,
    QuoteExpired,
    ReservationState,
    TransferKind,
    UnknownReservation,
)
from economy.client import LedgerClient
from economy.service import Credentials, _sweep_forever, bootstrap_ledger, create_app
from shared.config import get_settings

from ._fixtures import ALLOWANCE_MC, DAILY_CAP_MC, TODAY, FakeClock, seed

KEYS = {
    "concierge": "key-concierge",
    "travel": "key-travel",
    "local": "key-local",
    "transit": "key-transit",
}
ADMIN_KEY = "key-admin"
BASE_URL = "http://ledger"

ClientFor = Callable[[str], LedgerClient]


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest_asyncio.fixture
async def ledger(clock: FakeClock) -> AsyncIterator[Ledger]:
    led = await Ledger.open(":memory:", clock=clock)
    try:
        await seed(led)
        yield led
    finally:
        await led.close()


@pytest_asyncio.fixture
async def http(ledger: Ledger) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(ledger=ledger, credentials=Credentials(agent_keys=KEYS, admin_key=ADMIN_KEY))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=BASE_URL
    ) as client:
        yield client


@pytest.fixture
def client_for(http: httpx.AsyncClient) -> ClientFor:
    def make(agent_id: str) -> LedgerClient:
        key = ADMIN_KEY if agent_id == "admin" else KEYS[agent_id]
        return LedgerClient(BASE_URL, agent_id=agent_id, agent_key=key, http=http)

    return make


async def _quote(travel: LedgerClient, amount_mc: int = 2_000) -> str:
    q = await travel.create_quote(
        payer="concierge", skill_id="search_flights", amount_mc=amount_mc, ttl_seconds=60
    )
    return q.quote_id


# ---------- the happy path ----------


async def test_full_paid_delegation_over_http(client_for: ClientFor) -> None:
    concierge, travel, admin = client_for("concierge"), client_for("travel"), client_for("admin")

    quote_id = await _quote(travel, 2_000)
    held = await concierge.reserve(quote_id, trip_id="trip-9")
    assert held.state is ReservationState.HELD
    await travel.bind_task(held.reservation_id, "task-42")
    done = await travel.capture(held.reservation_id, 1_600)

    assert (done.state, done.captured_mc, done.a2a_task_id) == (
        ReservationState.CAPTURED,
        1_600,
        "task-42",
    )
    payer = await concierge.get_wallet()
    assert (payer.available_mc, payer.reserved_mc, payer.spent_today_mc) == (
        ALLOWANCE_MC - 1_600,
        0,
        1_600,
    )
    assert (await travel.get_wallet()).available_mc == 1_600
    kinds = [t.kind for t in await admin.journal(trip_id="trip-9")]
    assert kinds == [TransferKind.RESERVE, TransferKind.CAPTURE, TransferKind.REFUND]


async def test_release_by_either_party(client_for: ClientFor) -> None:
    concierge, travel = client_for("concierge"), client_for("travel")
    first = await concierge.reserve(await _quote(travel))
    second = await concierge.reserve(await _quote(travel))
    assert (await concierge.release(first.reservation_id)).state is ReservationState.RELEASED
    assert (await travel.release(second.reservation_id)).state is ReservationState.RELEASED
    assert (await concierge.get_wallet()).available_mc == ALLOWANCE_MC


async def test_agents_read_their_own_journal(client_for: ClientFor) -> None:
    concierge, travel = client_for("concierge"), client_for("travel")
    r = await concierge.reserve(await _quote(travel))
    await travel.capture(r.reservation_id, 500)
    own = await travel.journal(wallet_id="travel")
    assert [t.kind for t in own] == [TransferKind.CAPTURE]


# ---------- auth ----------


async def test_health_needs_no_auth(http: httpx.AsyncClient) -> None:
    assert (await http.get("/health")).json() == {"status": "ok"}


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"X-Agent-Id": "travel"},
        {"Authorization": "Bearer key-travel"},
        {"Authorization": "Bearer wrong", "X-Agent-Id": "travel"},
        {"Authorization": "Basic key-travel", "X-Agent-Id": "travel"},
        # A valid key presented under another agent's name.
        {"Authorization": "Bearer key-travel", "X-Agent-Id": "concierge"},
        {"Authorization": "Bearer key-travel", "X-Agent-Id": "admin"},
        {"Authorization": "Bearer key-x", "X-Agent-Id": "stranger"},
    ],
)
async def test_bad_credentials_are_401(http: httpx.AsyncClient, headers: dict[str, str]) -> None:
    response = await http.get("/wallets/travel", headers=headers)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


async def test_admin_disabled_without_admin_key(ledger: Ledger) -> None:
    app = create_app(ledger=ledger, credentials=Credentials(agent_keys=KEYS))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE_URL) as http:
        admin = LedgerClient(BASE_URL, agent_id="admin", agent_key="anything", http=http)
        with pytest.raises(LedgerAuthError):
            await admin.get_wallet("concierge")


def test_credentials_from_settings(monkeypatch: pytest.MonkeyPatch, minimal_env: None) -> None:
    monkeypatch.setenv("AGENT_KEY_TRAVEL", "t")
    monkeypatch.setenv("ECONOMY_ADMIN_KEY", "a")
    creds = Credentials.from_settings(get_settings())
    assert creds.agent_keys == {"travel": "t"}
    assert creds.admin_key == "a"


# ---------- permissions ----------


async def test_only_the_payee_binds_and_captures(client_for: ClientFor) -> None:
    concierge, travel = client_for("concierge"), client_for("travel")
    r = await concierge.reserve(await _quote(travel))
    with pytest.raises(LedgerForbidden):
        await concierge.capture(r.reservation_id, 1)
    with pytest.raises(LedgerForbidden):
        await concierge.bind_task(r.reservation_id, "t")


async def test_only_the_quoted_payer_reserves(client_for: ClientFor) -> None:
    quote_id = await _quote(client_for("travel"))
    with pytest.raises(LedgerForbidden):
        await client_for("local").reserve(quote_id)
    with pytest.raises(LedgerForbidden):
        await client_for("travel").reserve(quote_id)


async def test_outsiders_cannot_see_or_touch_a_reservation(client_for: ClientFor) -> None:
    concierge, travel, local = client_for("concierge"), client_for("travel"), client_for("local")
    r = await concierge.reserve(await _quote(travel))
    with pytest.raises(LedgerForbidden):
        await local.get_reservation(r.reservation_id)
    with pytest.raises(LedgerForbidden):
        await local.release(r.reservation_id)
    with pytest.raises(LedgerForbidden):
        await local.capture(r.reservation_id, 1)
    seen = await client_for("admin").get_reservation(r.reservation_id)
    assert seen.reservation_id == r.reservation_id


async def test_admin_cannot_release_for_a_party(client_for: ClientFor) -> None:
    r = await client_for("concierge").reserve(await _quote(client_for("travel")))
    with pytest.raises(LedgerForbidden):
        await client_for("admin").release(r.reservation_id)


async def test_wallets_and_journals_are_private(client_for: ClientFor) -> None:
    local = client_for("local")
    with pytest.raises(LedgerForbidden):
        await local.get_wallet("concierge")
    with pytest.raises(LedgerForbidden):
        await local.journal()
    with pytest.raises(LedgerForbidden):
        await local.journal(wallet_id="concierge")
    assert (await client_for("admin").get_wallet("concierge")).available_mc == ALLOWANCE_MC


async def test_admin_cannot_quote_and_agents_cannot_admin(
    http: httpx.AsyncClient, client_for: ClientFor
) -> None:
    with pytest.raises(LedgerForbidden):
        await client_for("admin").create_quote(
            payer="concierge", skill_id="s", amount_mc=1, ttl_seconds=1
        )
    agent = {"Authorization": "Bearer key-concierge", "X-Agent-Id": "concierge"}
    body = {"wallet_id": "concierge", "amount_mc": 1, "day": "2030-01-01"}
    assert (await http.post("/admin/allowances", json=body, headers=agent)).status_code == 403
    assert (await http.post("/admin/expire", headers=agent)).status_code == 403


# ---------- admin endpoints ----------


async def test_admin_grants_allowance_and_expires_holds(
    http: httpx.AsyncClient, client_for: ClientFor, clock: FakeClock
) -> None:
    admin = {"Authorization": f"Bearer {ADMIN_KEY}", "X-Agent-Id": "admin"}
    tomorrow = (TODAY + timedelta(days=1)).isoformat()
    body = {"wallet_id": "concierge", "amount_mc": 1_000, "day": tomorrow}
    response = await http.post("/admin/allowances", json=body, headers=admin)
    assert response.status_code == 201
    assert response.json()["kind"] == "allowance"

    r = await client_for("concierge").reserve(await _quote(client_for("travel")))
    clock.advance(10_000)
    expired = (await http.post("/admin/expire", headers=admin)).json()["expired"]
    assert expired == [r.reservation_id]


# ---------- error mapping ----------


async def test_ledger_errors_arrive_as_typed_exceptions(client_for: ClientFor) -> None:
    concierge, travel = client_for("concierge"), client_for("travel")
    big = await _quote(travel, DAILY_CAP_MC)
    await concierge.reserve(big)
    with pytest.raises(LedgerError) as excinfo:
        await concierge.reserve(await _quote(travel, 1))
    assert excinfo.value.code == "daily_cap_exceeded"
    with pytest.raises(UnknownReservation):
        await concierge.release("r_missing")


async def test_insufficient_funds_is_402(
    http: httpx.AsyncClient, ledger: Ledger, client_for: ClientFor
) -> None:
    await ledger.ensure_wallet("concierge")  # lift the cap
    quote_id = await _quote(client_for("travel"), ALLOWANCE_MC + 1)
    response = await http.post(
        "/reservations",
        json={"quote_id": quote_id},
        headers={"Authorization": "Bearer key-concierge", "X-Agent-Id": "concierge"},
    )
    assert response.status_code == 402
    with pytest.raises(InsufficientFunds):
        await client_for("concierge").reserve(quote_id)


async def test_expired_quote_is_410(client_for: ClientFor, clock: FakeClock) -> None:
    quote_id = await _quote(client_for("travel"))
    clock.advance(61)
    with pytest.raises(QuoteExpired):
        await client_for("concierge").reserve(quote_id)


@pytest.mark.parametrize(
    "body",
    [
        "not json",
        {"payer": "concierge"},
        {"payer": "concierge", "skill_id": "s", "amount_mc": 0, "ttl_seconds": 1},
        {"payer": "concierge", "skill_id": "s", "amount_mc": 1, "ttl_seconds": 1, "x": 1},
        # Passes the schema, breaks a ledger rule (payer == payee).
        {"payer": "travel", "skill_id": "s", "amount_mc": 1, "ttl_seconds": 1},
    ],
)
async def test_invalid_requests_are_422(http: httpx.AsyncClient, body: object) -> None:
    headers = {"Authorization": "Bearer key-travel", "X-Agent-Id": "travel"}
    if isinstance(body, str):
        response = await http.post("/quotes", content=body, headers=headers)
    else:
        response = await http.post("/quotes", json=body, headers=headers)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == InvalidRequest.code


# ---------- client transport behaviour ----------


def _failing_client(calls: list[str], response: httpx.Response | None = None) -> LedgerClient:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if response is None:
            raise httpx.ConnectError("refused", request=request)
        return response

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return LedgerClient(BASE_URL, agent_id="concierge", agent_key="k", http=http)


async def test_unreachable_ledger_is_retried_once_then_unavailable() -> None:
    calls: list[str] = []
    client = _failing_client(calls)
    with pytest.raises(LedgerUnavailable):
        await client.reserve("q_1")
    assert calls == ["/reservations", "/reservations"]


async def test_quote_creation_is_not_retried() -> None:
    calls: list[str] = []
    client = _failing_client(calls)
    with pytest.raises(LedgerUnavailable):
        await client.create_quote(payer="x", skill_id="s", amount_mc=1, ttl_seconds=1)
    assert calls == ["/quotes"]


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (httpx.Response(500, text="boom"), LedgerUnavailable),
        (httpx.Response(418, text="teapot"), LedgerError),
        (httpx.Response(409, json={"error": {"code": "new_code", "message": "m"}}), LedgerError),
    ],
)
async def test_unexpected_responses(response: httpx.Response, expected: type[Exception]) -> None:
    client = _failing_client([], response)
    with pytest.raises(expected):
        await client.release("r_1")


async def test_client_owns_and_closes_its_http_client() -> None:
    async with LedgerClient(BASE_URL, agent_id="travel", agent_key="k") as client:
        http = client._http
    assert http.is_closed


def test_client_requires_credentials() -> None:
    with pytest.raises(ValueError):
        LedgerClient(BASE_URL, agent_id="", agent_key="k")


# ---------- bootstrap / lifespan / sweeper ----------


async def test_bootstrap_is_idempotent_per_day(clock: FakeClock) -> None:
    async with await Ledger.open(":memory:", clock=clock) as led:
        for _ in range(2):
            await bootstrap_ledger(led, daily_allowance_mc=7_000, daily_cap_mc=3_000, today=TODAY)
        concierge = await led.get_wallet("concierge")
        assert (concierge.available_mc, concierge.daily_cap_mc) == (7_000, 3_000)
        assert (await led.get_wallet("transit")).available_mc == 0
        await bootstrap_ledger(
            led, daily_allowance_mc=7_000, daily_cap_mc=3_000, today=TODAY + timedelta(days=1)
        )
        assert (await led.get_wallet("concierge")).available_mc == 14_000


async def test_lifespan_opens_and_bootstraps_the_configured_ledger(
    monkeypatch: pytest.MonkeyPatch, minimal_env: None, tmp_path: Path
) -> None:
    monkeypatch.setenv("LEDGER_DB_PATH", str(tmp_path / "ledger.db"))
    monkeypatch.setenv("ECONOMY_DAILY_ALLOWANCE_MC", "9000")
    app = create_app(credentials=Credentials(agent_keys=KEYS, admin_key=ADMIN_KEY))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url=BASE_URL
        ) as http:
            admin = LedgerClient(BASE_URL, agent_id="admin", agent_key=ADMIN_KEY, http=http)
            assert (await admin.get_wallet("concierge")).available_mc == 9_000
    assert (tmp_path / "ledger.db").exists()


async def test_sweeper_expires_stale_holds(
    monkeypatch: pytest.MonkeyPatch, minimal_env: None
) -> None:
    monkeypatch.setenv("ECONOMY_SWEEP_INTERVAL_SECONDS", "0.01")
    async with await Ledger.open(":memory:") as led:
        await seed(led, day=datetime.now(timezone.utc).date())
        q = await led.create_quote(
            payer="concierge", payee="travel", skill_id="s", amount_mc=10, ttl_seconds=60
        )
        r = await led.reserve(q.quote_id, hold_seconds=0.01)
        sweeper = asyncio.create_task(_sweep_forever(led, get_settings()))
        try:
            for _ in range(100):
                await asyncio.sleep(0.01)
                if (await led.get_reservation(r.reservation_id)).state is not ReservationState.HELD:
                    break
        finally:
            sweeper.cancel()
            with pytest.raises(asyncio.CancelledError):
                await sweeper
        assert (await led.get_reservation(r.reservation_id)).state is ReservationState.EXPIRED
