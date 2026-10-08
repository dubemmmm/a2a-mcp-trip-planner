"""HTTP service exposing the ledger to other processes.

Run it with:

    uvicorn economy.service:_uvicorn_app --factory --env-file .env --port 8005

Every request except `/health` authenticates with two headers:

    Authorization: Bearer <AGENT_KEY_<NAME>>    (or ECONOMY_ADMIN_KEY)
    X-Agent-Id:    <name>                       (or "admin")

Permissions follow the design: the payee quotes, binds and captures; the
payer reserves; either side may release; balances and journals are visible
to their owner and to the admin.

The ledger is injected (`create_app(ledger=...)`) so tests drive the service
over `httpx.ASGITransport` without a lifespan; otherwise the lifespan opens
`LEDGER_DB_PATH`, bootstraps the wallets and runs the expiry sweeper.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, TypeVar

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from shared.config import Settings, get_settings

from .errors import InvalidRequest, LedgerAuthError, LedgerError, LedgerForbidden
from .ledger import Ledger
from .models import Reservation, WalletStatus

_LOGGER = structlog.get_logger(__name__)

ADMIN_ID = "admin"

# Wallets the ledger creates at startup. Payers get the daily allowance and
# the daily cap; payees start empty and are only ever paid.
PAYER_WALLETS: tuple[str, ...] = ("concierge",)
PAYEE_WALLETS: tuple[str, ...] = ("travel", "local", "transit")

_ModelT = TypeVar("_ModelT", bound=BaseModel)


# ------------------------------------------------------------------ requests


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QuoteBody(_Body):
    payer: str = Field(..., min_length=1)
    skill_id: str = Field(..., min_length=1)
    amount_mc: int = Field(..., gt=0)
    ttl_seconds: float = Field(..., gt=0, le=3600)


class ReserveBody(_Body):
    quote_id: str = Field(..., min_length=1)
    trip_id: str | None = None
    hold_seconds: float | None = Field(default=None, gt=0, le=3600)


class BindBody(_Body):
    a2a_task_id: str = Field(..., min_length=1)


class CaptureBody(_Body):
    amount_mc: int = Field(..., ge=0)


class AllowanceBody(_Body):
    wallet_id: str = Field(..., min_length=1)
    amount_mc: int = Field(..., gt=0)
    day: date


# ---------------------------------------------------------------------- auth


@dataclass(frozen=True)
class Caller:
    agent_id: str

    @property
    def is_admin(self) -> bool:
        return self.agent_id == ADMIN_ID


@dataclass(frozen=True)
class Credentials:
    """Agent name -> key, plus the admin key. Compared in constant time."""

    agent_keys: Mapping[str, str]
    admin_key: str | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "Credentials":
        secrets = {
            "concierge": settings.agent_key_concierge,
            "travel": settings.agent_key_travel,
            "local": settings.agent_key_local,
            "transit": settings.agent_key_transit,
        }
        keys = {name: s.get_secret_value() for name, s in secrets.items() if s is not None}
        admin = settings.economy_admin_key
        return cls(agent_keys=keys, admin_key=admin.get_secret_value() if admin else None)

    def authenticate(self, request: Request) -> Caller:
        agent_id = request.headers.get("x-agent-id", "")
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not token or not agent_id:
            raise LedgerAuthError("missing bearer token or X-Agent-Id header")
        expected = self.admin_key if agent_id == ADMIN_ID else self.agent_keys.get(agent_id)
        if expected is None or not hmac.compare_digest(token.encode(), expected.encode()):
            raise LedgerAuthError("invalid credentials")
        return Caller(agent_id)


def _require(allowed: bool, message: str) -> None:
    if not allowed:
        raise LedgerForbidden(message)


# ------------------------------------------------------------------- helpers


def _ledger(request: Request) -> Ledger:
    ledger: Ledger = request.app.state.ledger
    return ledger


def _caller(request: Request) -> Caller:
    credentials: Credentials = request.app.state.credentials
    return credentials.authenticate(request)


async def _body(request: Request, model: type[_ModelT]) -> _ModelT:
    try:
        raw = await request.json()
    except json.JSONDecodeError as e:
        raise InvalidRequest("request body must be valid JSON") from e
    try:
        return model.model_validate(raw)
    except ValidationError as e:
        raise InvalidRequest(str(e)) from e


def _json(model: BaseModel, status_code: int = 200) -> JSONResponse:
    return JSONResponse(model.model_dump(mode="json"), status_code=status_code)


async def _party_reservation(request: Request, caller: Caller) -> Reservation:
    reservation = await _ledger(request).get_reservation(request.path_params["reservation_id"])
    _require(
        caller.is_admin or caller.agent_id in (reservation.payer, reservation.payee),
        "only the payer, the payee or the admin may see this reservation",
    )
    return reservation


# -------------------------------------------------------------------- routes


async def _health(_request: Request) -> Response:
    return JSONResponse({"status": "ok"})


async def _create_quote(request: Request) -> Response:
    caller = _caller(request)
    _require(not caller.is_admin, "quotes are issued by the payee agent")
    body = await _body(request, QuoteBody)
    quote = await _ledger(request).create_quote(
        payer=body.payer,
        payee=caller.agent_id,
        skill_id=body.skill_id,
        amount_mc=body.amount_mc,
        ttl_seconds=body.ttl_seconds,
    )
    return _json(quote, status_code=201)


async def _reserve(request: Request) -> Response:
    caller = _caller(request)
    body = await _body(request, ReserveBody)
    ledger = _ledger(request)
    quote = await ledger.get_quote(body.quote_id)
    _require(caller.agent_id == quote.payer, "only the quoted payer may reserve")
    reservation = await ledger.reserve(
        body.quote_id, trip_id=body.trip_id, hold_seconds=body.hold_seconds
    )
    return _json(reservation, status_code=201)


async def _get_reservation(request: Request) -> Response:
    return _json(await _party_reservation(request, _caller(request)))


async def _bind(request: Request) -> Response:
    caller = _caller(request)
    reservation = await _party_reservation(request, caller)
    _require(caller.agent_id == reservation.payee, "only the payee may bind a task")
    body = await _body(request, BindBody)
    return _json(await _ledger(request).bind_task(reservation.reservation_id, body.a2a_task_id))


async def _capture(request: Request) -> Response:
    caller = _caller(request)
    reservation = await _party_reservation(request, caller)
    _require(caller.agent_id == reservation.payee, "only the payee may capture")
    body = await _body(request, CaptureBody)
    return _json(await _ledger(request).capture(reservation.reservation_id, body.amount_mc))


async def _release(request: Request) -> Response:
    caller = _caller(request)
    reservation = await _party_reservation(request, caller)
    _require(
        caller.agent_id in (reservation.payer, reservation.payee),
        "only the payer or the payee may release",
    )
    return _json(await _ledger(request).release(reservation.reservation_id))


async def _get_wallet(request: Request) -> Response:
    caller = _caller(request)
    wallet_id = request.path_params["wallet_id"]
    _require(caller.is_admin or caller.agent_id == wallet_id, "not your wallet")
    ledger = _ledger(request)
    wallet = await ledger.get_wallet(wallet_id)
    status = WalletStatus(**wallet.model_dump(), spent_today_mc=await ledger.daily_spend(wallet_id))
    return _json(status)


async def _journal(request: Request) -> Response:
    caller = _caller(request)
    trip_id = request.query_params.get("trip_id")
    wallet_id = request.query_params.get("wallet_id")
    if not caller.is_admin:
        _require(wallet_id == caller.agent_id, "agents may only read their own journal")
    transfers = await _ledger(request).journal(trip_id=trip_id, wallet_id=wallet_id)
    return JSONResponse({"transfers": [t.model_dump(mode="json") for t in transfers]})


async def _grant_allowance(request: Request) -> Response:
    _require(_caller(request).is_admin, "admin only")
    body = await _body(request, AllowanceBody)
    transfer = await _ledger(request).grant_allowance(body.wallet_id, body.amount_mc, day=body.day)
    return _json(transfer, status_code=201)


async def _expire(request: Request) -> Response:
    _require(_caller(request).is_admin, "admin only")
    expired = await _ledger(request).expire_stale()
    return JSONResponse({"expired": [r.reservation_id for r in expired]})


async def _ledger_error(_request: Request, exc: Exception) -> Response:
    assert isinstance(exc, LedgerError)
    return JSONResponse(
        {"error": {"code": exc.code, "message": str(exc)}}, status_code=exc.http_status
    )


async def _value_error(_request: Request, exc: Exception) -> Response:
    # The ledger raises ValueError for arguments that pass the body schema
    # but break a rule (payer == payee, a blank id after stripping, ...).
    return JSONResponse(
        {"error": {"code": InvalidRequest.code, "message": str(exc)}},
        status_code=InvalidRequest.http_status,
    )


# ----------------------------------------------------------------- lifecycle


async def bootstrap_ledger(
    ledger: Ledger,
    *,
    daily_allowance_mc: int,
    daily_cap_mc: int,
    today: date,
) -> None:
    """Create every agent wallet and grant payers today's allowance.

    Safe to call repeatedly: wallet creation upserts and the allowance is
    keyed by day, so it is granted at most once per wallet per day.
    """
    for wallet_id in PAYER_WALLETS:
        await ledger.ensure_wallet(wallet_id, daily_cap_mc=daily_cap_mc)
        if daily_allowance_mc > 0:
            await ledger.grant_allowance(wallet_id, daily_allowance_mc, day=today)
    for wallet_id in PAYEE_WALLETS:
        await ledger.ensure_wallet(wallet_id)


async def _sweep_forever(ledger: Ledger, settings: Settings) -> None:
    """Expire stale holds and grant each new day's allowance."""
    while True:
        await asyncio.sleep(settings.economy_sweep_interval_seconds)
        try:
            await ledger.expire_stale()
            await bootstrap_ledger(
                ledger,
                daily_allowance_mc=settings.economy_daily_allowance_mc,
                daily_cap_mc=settings.economy_daily_cap_mc,
                today=datetime.now(timezone.utc).date(),
            )
        except Exception:  # keep sweeping; one bad pass must not stop expiry
            _LOGGER.exception("ledger_sweep_failed")


def create_app(
    *,
    ledger: Ledger | None = None,
    credentials: Credentials | None = None,
) -> Starlette:
    """Build the ledger service.

    Args:
        ledger: An open ledger to serve. When None, the lifespan opens
            `LEDGER_DB_PATH`, bootstraps wallets and runs the sweeper.
        credentials: Agent and admin keys. Defaults to the ones in Settings.
    """

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        if ledger is not None:
            yield
            return
        settings = get_settings()
        opened = await Ledger.open(
            settings.ledger_db_path, default_hold_seconds=settings.economy_hold_seconds
        )
        app.state.ledger = opened
        await bootstrap_ledger(
            opened,
            daily_allowance_mc=settings.economy_daily_allowance_mc,
            daily_cap_mc=settings.economy_daily_cap_mc,
            today=datetime.now(timezone.utc).date(),
        )
        sweeper = asyncio.create_task(_sweep_forever(opened, settings))
        try:
            yield
        finally:
            sweeper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sweeper
            await opened.close()

    routes = [
        Route("/health", _health),
        Route("/quotes", _create_quote, methods=["POST"]),
        Route("/reservations", _reserve, methods=["POST"]),
        Route("/reservations/{reservation_id}", _get_reservation),
        Route("/reservations/{reservation_id}/bind", _bind, methods=["POST"]),
        Route("/reservations/{reservation_id}/capture", _capture, methods=["POST"]),
        Route("/reservations/{reservation_id}/release", _release, methods=["POST"]),
        Route("/wallets/{wallet_id}", _get_wallet),
        Route("/journal", _journal),
        Route("/admin/allowances", _grant_allowance, methods=["POST"]),
        Route("/admin/expire", _expire, methods=["POST"]),
    ]
    exception_handlers: dict[Any, Any] = {LedgerError: _ledger_error, ValueError: _value_error}
    app = Starlette(routes=routes, lifespan=lifespan, exception_handlers=exception_handlers)
    app.state.ledger = ledger
    app.state.credentials = credentials or Credentials.from_settings(get_settings())
    return app


def _uvicorn_app() -> Starlette:
    """Factory for `uvicorn economy.service:_uvicorn_app --factory`."""
    return create_app()
