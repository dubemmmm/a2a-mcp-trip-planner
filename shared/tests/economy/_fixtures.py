"""Shared helpers for the economy tests: a controllable clock and a seeded ledger."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from economy import Ledger, Quote, Reservation

START = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
TODAY = START.date()

ALLOWANCE_MC = 50_000
DAILY_CAP_MC = 20_000


class FakeClock:
    """A clock tests move by hand."""

    def __init__(self, now: datetime = START) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


async def seed(ledger: Ledger, *, day: date = TODAY) -> None:
    """Concierge pays (allowance + cap); the specialists start empty."""
    await ledger.ensure_wallet("concierge", daily_cap_mc=DAILY_CAP_MC)
    await ledger.grant_allowance("concierge", ALLOWANCE_MC, day=day)
    for name in ("travel", "local", "transit"):
        await ledger.ensure_wallet(name)


async def quote(
    ledger: Ledger,
    amount_mc: int = 1_000,
    *,
    payee: str = "travel",
    ttl_seconds: float = 60,
) -> Quote:
    return await ledger.create_quote(
        payer="concierge",
        payee=payee,
        skill_id="search_flights",
        amount_mc=amount_mc,
        ttl_seconds=ttl_seconds,
    )


async def hold(
    ledger: Ledger,
    amount_mc: int = 1_000,
    *,
    payee: str = "travel",
    trip_id: str | None = "trip-1",
) -> Reservation:
    q = await quote(ledger, amount_mc, payee=payee)
    return await ledger.reserve(q.quote_id, trip_id=trip_id)
