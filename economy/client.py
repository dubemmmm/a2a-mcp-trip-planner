"""Async HTTP client for the ledger service.

Agents use this instead of touching the ledger database. Errors come back as
the same typed exceptions the in-process `Ledger` raises, so callers handle
one set of failures. Network failures surface as `LedgerUnavailable` — the
caller must fail closed (never run paid work for free).

    async with LedgerClient(settings.ledger_url, agent_id="travel", agent_key=key) as ledger:
        quote = await ledger.create_quote(
            payer="concierge", skill_id="search_flights", amount_mc=1_200, ttl_seconds=60
        )

Reserve, bind, capture and release are idempotent on the ledger side, so the
client retries them once on a transport error. Creating a quote is not
retried: a lost response would only orphan an unused quote, but a retry would
hand the caller a different quote id than the one the payee recorded first.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any, Self

import httpx

from shared.config import Settings

from .errors import ERRORS_BY_CODE, LedgerError, LedgerUnavailable
from .models import Quote, Reservation, Transfer, WalletStatus

DEFAULT_TIMEOUT_SECONDS = 10.0


class LedgerClient:
    """Typed client for one agent's view of the ledger."""

    def __init__(
        self,
        base_url: str,
        *,
        agent_id: str,
        agent_key: str,
        http: httpx.AsyncClient | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if not agent_id or not agent_key:
            raise ValueError("agent_id and agent_key are required")
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {agent_key}", "X-Agent-Id": agent_id}
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient(timeout=timeout_seconds)
        self.agent_id = agent_id

    @classmethod
    def from_settings(cls, agent_id: str, settings: Settings) -> "LedgerClient | None":
        """A client acting as `agent_id`, or None if that agent has no key.

        No key means the agent is not part of the economy: specialists then
        run free (no pricing on their card), the Concierge pays no one.
        """
        secret = getattr(settings, f"agent_key_{agent_id}", None)
        if secret is None:
            return None
        return cls(settings.ledger_url, agent_id=agent_id, agent_key=secret.get_secret_value())

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    # ---------------------------------------------------------------- payee

    async def create_quote(
        self, *, payer: str, skill_id: str, amount_mc: int, ttl_seconds: float
    ) -> Quote:
        data = await self._request(
            "POST",
            "/quotes",
            json={
                "payer": payer,
                "skill_id": skill_id,
                "amount_mc": amount_mc,
                "ttl_seconds": ttl_seconds,
            },
            retry=False,
        )
        return Quote.model_validate(data)

    async def bind_task(self, reservation_id: str, a2a_task_id: str) -> Reservation:
        data = await self._request(
            "POST", f"/reservations/{reservation_id}/bind", json={"a2a_task_id": a2a_task_id}
        )
        return Reservation.model_validate(data)

    async def capture(self, reservation_id: str, amount_mc: int) -> Reservation:
        data = await self._request(
            "POST", f"/reservations/{reservation_id}/capture", json={"amount_mc": amount_mc}
        )
        return Reservation.model_validate(data)

    # ---------------------------------------------------------------- payer

    async def reserve(
        self,
        quote_id: str,
        *,
        trip_id: str | None = None,
        hold_seconds: float | None = None,
    ) -> Reservation:
        body: dict[str, Any] = {"quote_id": quote_id, "trip_id": trip_id}
        if hold_seconds is not None:
            body["hold_seconds"] = hold_seconds
        data = await self._request("POST", "/reservations", json=body)
        return Reservation.model_validate(data)

    # ----------------------------------------------------------------- both

    async def release(self, reservation_id: str) -> Reservation:
        data = await self._request("POST", f"/reservations/{reservation_id}/release")
        return Reservation.model_validate(data)

    async def get_reservation(self, reservation_id: str) -> Reservation:
        data = await self._request("GET", f"/reservations/{reservation_id}")
        return Reservation.model_validate(data)

    async def get_wallet(self, wallet_id: str | None = None) -> WalletStatus:
        data = await self._request("GET", f"/wallets/{wallet_id or self.agent_id}")
        return WalletStatus.model_validate(data)

    async def journal(
        self, *, trip_id: str | None = None, wallet_id: str | None = None
    ) -> tuple[Transfer, ...]:
        params = {k: v for k, v in {"trip_id": trip_id, "wallet_id": wallet_id}.items() if v}
        data = await self._request("GET", "/journal", params=params)
        return tuple(Transfer.model_validate(t) for t in data["transfers"])

    # ------------------------------------------------------------- plumbing

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        retry: bool = True,
    ) -> Any:
        attempts = 2 if retry else 1
        for attempt in range(attempts):
            try:
                response = await self._http.request(
                    method,
                    f"{self._base_url}{path}",
                    json=json,
                    params=params,
                    headers=self._headers,
                )
                break
            except httpx.TransportError as e:
                if attempt + 1 == attempts:
                    raise LedgerUnavailable(f"ledger at {self._base_url} unreachable: {e}") from e
        if response.is_success:
            return response.json()
        raise _error_from_response(response)


def _error_from_response(response: httpx.Response) -> LedgerError:
    try:
        error = response.json()["error"]
        code, message = str(error["code"]), str(error["message"])
    except (ValueError, KeyError, TypeError):
        if response.status_code >= 500:
            return LedgerUnavailable(f"ledger returned HTTP {response.status_code}")
        return LedgerError(f"ledger returned HTTP {response.status_code}: {response.text[:200]}")
    return ERRORS_BY_CODE.get(code, LedgerError)(message)
