"""Tests for the in-process Ledger: every operation, error and idempotent repeat."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta, timezone, datetime
from pathlib import Path

import pytest
import pytest_asyncio

from economy import (
    CaptureExceedsHold,
    DailyCapExceeded,
    IdempotencyConflict,
    InsufficientFunds,
    Ledger,
    QuoteAlreadyReserved,
    QuoteExpired,
    ReservationNotHeld,
    ReservationState,
    TaskAlreadyBound,
    TransferKind,
    UnknownQuote,
    UnknownReservation,
    UnknownWallet,
)
from economy.ledger import _ts

from ._fixtures import ALLOWANCE_MC, DAILY_CAP_MC, TODAY, FakeClock, hold, quote, seed


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest_asyncio.fixture
async def ledger(clock: FakeClock) -> AsyncIterator[Ledger]:
    led = await Ledger.open(":memory:", clock=clock, default_hold_seconds=180)
    try:
        await seed(led)
        yield led
    finally:
        await led.close()


# ---------- wallets ----------


async def test_ensure_wallet_creates_empty_wallet(ledger: Ledger) -> None:
    wallet = await ledger.get_wallet("travel")
    assert (wallet.available_mc, wallet.reserved_mc, wallet.daily_cap_mc) == (0, 0, None)


async def test_ensure_wallet_is_an_upsert_that_updates_the_cap(ledger: Ledger) -> None:
    await ledger.ensure_wallet("concierge", daily_cap_mc=5)
    wallet = await ledger.get_wallet("concierge")
    assert wallet.daily_cap_mc == 5
    assert wallet.available_mc == ALLOWANCE_MC  # balance untouched


@pytest.mark.parametrize("bad", ["", "   ", "treasury"])
async def test_ensure_wallet_rejects_bad_names(ledger: Ledger, bad: str) -> None:
    with pytest.raises(ValueError):
        await ledger.ensure_wallet(bad)


async def test_unknown_wallet(ledger: Ledger) -> None:
    with pytest.raises(UnknownWallet):
        await ledger.get_wallet("nobody")


# ---------- allowance ----------


async def test_allowance_is_minted_from_the_treasury(ledger: Ledger) -> None:
    (entry,) = await ledger.journal(wallet_id="concierge")
    assert entry.kind is TransferKind.ALLOWANCE
    assert (entry.from_wallet, entry.to_wallet, entry.amount_mc) == (
        "treasury",
        "concierge",
        ALLOWANCE_MC,
    )


async def test_allowance_is_granted_once_per_day(ledger: Ledger) -> None:
    first = (await ledger.journal(wallet_id="concierge"))[0]
    again = await ledger.grant_allowance("concierge", ALLOWANCE_MC, day=TODAY)
    assert again == first
    assert (await ledger.get_wallet("concierge")).available_mc == ALLOWANCE_MC


async def test_allowance_repeat_with_other_amount_conflicts(ledger: Ledger) -> None:
    with pytest.raises(IdempotencyConflict):
        await ledger.grant_allowance("concierge", 1, day=TODAY)


async def test_allowance_next_day_adds_up(ledger: Ledger) -> None:
    await ledger.grant_allowance("concierge", ALLOWANCE_MC, day=TODAY + timedelta(days=1))
    assert (await ledger.get_wallet("concierge")).available_mc == 2 * ALLOWANCE_MC


async def test_allowance_rejects_unknown_wallet_and_bad_amount(ledger: Ledger) -> None:
    with pytest.raises(UnknownWallet):
        await ledger.grant_allowance("nobody", 10, day=TODAY)
    with pytest.raises(ValueError):
        await ledger.grant_allowance("concierge", 0, day=TODAY)


# ---------- quotes ----------


async def test_quote_records_offer_and_expiry(ledger: Ledger, clock: FakeClock) -> None:
    q = await quote(ledger, 1_234, ttl_seconds=60)
    assert q.quote_id.startswith("q_")
    assert q.expires_at == clock.now + timedelta(seconds=60)
    assert await ledger.get_quote(q.quote_id) == q


async def test_quote_validation(ledger: Ledger) -> None:
    with pytest.raises(ValueError):
        await ledger.create_quote(
            payer="travel", payee="travel", skill_id="x", amount_mc=1, ttl_seconds=1
        )
    with pytest.raises(ValueError):
        await quote(ledger, 0)
    with pytest.raises(ValueError):
        await quote(ledger, 10, ttl_seconds=0)
    with pytest.raises(UnknownWallet):
        await quote(ledger, 10, payee="nobody")
    with pytest.raises(UnknownQuote):
        await ledger.get_quote("q_missing")


# ---------- reserve ----------


async def test_reserve_moves_available_to_reserved(ledger: Ledger, clock: FakeClock) -> None:
    r = await hold(ledger, 1_500)
    assert r.state is ReservationState.HELD
    assert r.expires_at == clock.now + timedelta(seconds=180)
    wallet = await ledger.get_wallet("concierge")
    assert (wallet.available_mc, wallet.reserved_mc) == (ALLOWANCE_MC - 1_500, 1_500)
    entry = (await ledger.journal(trip_id="trip-1"))[-1]
    assert (entry.kind, entry.from_wallet, entry.to_wallet) == (
        TransferKind.RESERVE,
        "concierge",
        None,
    )


async def test_reserve_repeat_returns_the_same_reservation(ledger: Ledger) -> None:
    q = await quote(ledger, 1_000)
    first = await ledger.reserve(q.quote_id, trip_id="t")
    again = await ledger.reserve(q.quote_id, trip_id="t")
    assert again == first
    assert (await ledger.get_wallet("concierge")).reserved_mc == 1_000


async def test_reserve_repeat_after_settlement_still_returns_it(ledger: Ledger) -> None:
    q = await quote(ledger, 1_000)
    r = await ledger.reserve(q.quote_id)
    await ledger.release(r.reservation_id)
    again = await ledger.reserve(q.quote_id)
    assert again.state is ReservationState.RELEASED
    assert (await ledger.get_wallet("concierge")).available_mc == ALLOWANCE_MC


async def test_reserve_same_quote_for_another_trip_is_refused(ledger: Ledger) -> None:
    q = await quote(ledger, 1_000)
    await ledger.reserve(q.quote_id, trip_id="a")
    with pytest.raises(QuoteAlreadyReserved):
        await ledger.reserve(q.quote_id, trip_id="b")


async def test_reserve_expired_quote(ledger: Ledger, clock: FakeClock) -> None:
    q = await quote(ledger, 1_000, ttl_seconds=60)
    clock.advance(60)
    with pytest.raises(QuoteExpired):
        await ledger.reserve(q.quote_id)


async def test_reserve_unknown_quote(ledger: Ledger) -> None:
    with pytest.raises(UnknownQuote):
        await ledger.reserve("q_missing")


async def test_reserve_insufficient_funds_changes_nothing(ledger: Ledger) -> None:
    await ledger.ensure_wallet("concierge")  # lift the cap
    q = await quote(ledger, ALLOWANCE_MC + 1)
    with pytest.raises(InsufficientFunds):
        await ledger.reserve(q.quote_id)
    wallet = await ledger.get_wallet("concierge")
    assert (wallet.available_mc, wallet.reserved_mc) == (ALLOWANCE_MC, 0)
    assert len(await ledger.journal()) == 1  # just the allowance


async def test_reserve_daily_cap(ledger: Ledger) -> None:
    await hold(ledger, DAILY_CAP_MC - 100)
    with pytest.raises(DailyCapExceeded):
        await hold(ledger, 101)
    await hold(ledger, 100)  # exactly at the cap is fine
    assert await ledger.daily_spend("concierge") == DAILY_CAP_MC


async def test_daily_cap_counts_captured_not_held_amount(ledger: Ledger) -> None:
    r = await hold(ledger, DAILY_CAP_MC)
    await ledger.capture(r.reservation_id, 4_000)
    assert await ledger.daily_spend("concierge") == 4_000
    await hold(ledger, DAILY_CAP_MC - 4_000)


async def test_daily_cap_ignores_released_holds_and_resets_next_day(
    ledger: Ledger, clock: FakeClock
) -> None:
    r = await hold(ledger, DAILY_CAP_MC)
    await ledger.release(r.reservation_id)
    assert await ledger.daily_spend("concierge") == 0
    r = await hold(ledger, DAILY_CAP_MC)
    await ledger.capture(r.reservation_id, DAILY_CAP_MC)
    clock.advance(24 * 3600)
    await hold(ledger, DAILY_CAP_MC)
    assert await ledger.daily_spend("concierge", TODAY) == DAILY_CAP_MC


async def test_reserve_hold_seconds_override_and_validation(
    ledger: Ledger, clock: FakeClock
) -> None:
    q = await quote(ledger)
    r = await ledger.reserve(q.quote_id, hold_seconds=5)
    assert r.expires_at == clock.now + timedelta(seconds=5)
    with pytest.raises(ValueError):
        await ledger.reserve((await quote(ledger)).quote_id, hold_seconds=0)


# ---------- bind ----------


async def test_bind_task(ledger: Ledger) -> None:
    r = await hold(ledger)
    bound = await ledger.bind_task(r.reservation_id, "task-1")
    assert bound.a2a_task_id == "task-1"
    assert await ledger.bind_task(r.reservation_id, "task-1") == bound
    with pytest.raises(TaskAlreadyBound):
        await ledger.bind_task(r.reservation_id, "task-2")


async def test_bind_task_refuses_settled_and_unknown(ledger: Ledger) -> None:
    r = await hold(ledger)
    await ledger.release(r.reservation_id)
    with pytest.raises(ReservationNotHeld):
        await ledger.bind_task(r.reservation_id, "task-1")
    with pytest.raises(UnknownReservation):
        await ledger.bind_task("r_missing", "task-1")
    with pytest.raises(ValueError):
        await ledger.bind_task(r.reservation_id, " ")


async def test_bind_after_hold_expiry_expires_it(ledger: Ledger, clock: FakeClock) -> None:
    r = await hold(ledger, 1_000)
    clock.advance(180)
    with pytest.raises(ReservationNotHeld):
        await ledger.bind_task(r.reservation_id, "task-1")
    assert (await ledger.get_reservation(r.reservation_id)).state is ReservationState.EXPIRED
    assert (await ledger.get_wallet("concierge")).available_mc == ALLOWANCE_MC


# ---------- capture ----------


async def test_capture_full_amount(ledger: Ledger) -> None:
    r = await hold(ledger, 2_000)
    done = await ledger.capture(r.reservation_id, 2_000)
    assert (done.state, done.captured_mc) == (ReservationState.CAPTURED, 2_000)
    assert done.settled_at is not None
    payer, payee = await ledger.get_wallet("concierge"), await ledger.get_wallet("travel")
    assert (payer.available_mc, payer.reserved_mc) == (ALLOWANCE_MC - 2_000, 0)
    assert payee.available_mc == 2_000
    kinds = [t.kind for t in await ledger.journal(trip_id="trip-1")]
    assert kinds == [TransferKind.RESERVE, TransferKind.CAPTURE]  # no zero refund row


async def test_capture_partial_refunds_the_rest(ledger: Ledger) -> None:
    r = await hold(ledger, 2_000)
    await ledger.capture(r.reservation_id, 1_250)
    payer, payee = await ledger.get_wallet("concierge"), await ledger.get_wallet("travel")
    assert (payer.available_mc, payer.reserved_mc) == (ALLOWANCE_MC - 1_250, 0)
    assert payee.available_mc == 1_250
    capture, refund = (await ledger.journal(trip_id="trip-1"))[1:]
    assert (capture.kind, capture.from_wallet, capture.to_wallet, capture.amount_mc) == (
        TransferKind.CAPTURE,
        "concierge",
        "travel",
        1_250,
    )
    assert (refund.kind, refund.to_wallet, refund.amount_mc) == (
        TransferKind.REFUND,
        "concierge",
        750,
    )


async def test_capture_zero(ledger: Ledger) -> None:
    r = await hold(ledger, 2_000)
    done = await ledger.capture(r.reservation_id, 0)
    assert done.captured_mc == 0
    assert (await ledger.get_wallet("concierge")).available_mc == ALLOWANCE_MC


async def test_capture_is_idempotent(ledger: Ledger) -> None:
    r = await hold(ledger, 2_000)
    first = await ledger.capture(r.reservation_id, 1_000)
    assert await ledger.capture(r.reservation_id, 1_000) == first
    assert (await ledger.get_wallet("travel")).available_mc == 1_000
    with pytest.raises(IdempotencyConflict):
        await ledger.capture(r.reservation_id, 1_500)


async def test_capture_refusals(ledger: Ledger) -> None:
    r = await hold(ledger, 2_000)
    with pytest.raises(CaptureExceedsHold):
        await ledger.capture(r.reservation_id, 2_001)
    with pytest.raises(ValueError):
        await ledger.capture(r.reservation_id, -1)
    await ledger.release(r.reservation_id)
    with pytest.raises(ReservationNotHeld):
        await ledger.capture(r.reservation_id, 1)
    with pytest.raises(UnknownReservation):
        await ledger.capture("r_missing", 1)


async def test_late_capture_expires_the_hold_instead(ledger: Ledger, clock: FakeClock) -> None:
    r = await hold(ledger, 2_000)
    clock.advance(181)
    with pytest.raises(ReservationNotHeld):
        await ledger.capture(r.reservation_id, 2_000)
    assert (await ledger.get_reservation(r.reservation_id)).state is ReservationState.EXPIRED
    assert (await ledger.get_wallet("concierge")).available_mc == ALLOWANCE_MC
    assert (await ledger.get_wallet("travel")).available_mc == 0


# ---------- release / expire ----------


async def test_release_returns_everything_and_repeats_safely(ledger: Ledger) -> None:
    r = await hold(ledger, 2_000)
    released = await ledger.release(r.reservation_id)
    assert released.state is ReservationState.RELEASED
    assert await ledger.release(r.reservation_id) == released
    wallet = await ledger.get_wallet("concierge")
    assert (wallet.available_mc, wallet.reserved_mc) == (ALLOWANCE_MC, 0)


async def test_release_after_capture_is_refused(ledger: Ledger) -> None:
    r = await hold(ledger, 2_000)
    await ledger.capture(r.reservation_id, 2_000)
    with pytest.raises(ReservationNotHeld):
        await ledger.release(r.reservation_id)


async def test_expire_stale_only_touches_overdue_holds(ledger: Ledger, clock: FakeClock) -> None:
    old = await hold(ledger, 1_000)
    clock.advance(100)
    young = await hold(ledger, 1_000)
    settled = await hold(ledger, 1_000)
    await ledger.capture(settled.reservation_id, 1_000)
    clock.advance(90)  # old is 190s, young 90s

    expired = await ledger.expire_stale()
    assert [r.reservation_id for r in expired] == [old.reservation_id]
    assert expired[0].state is ReservationState.EXPIRED
    assert (await ledger.get_reservation(young.reservation_id)).state is ReservationState.HELD
    assert await ledger.expire_stale() == ()
    last = (await ledger.journal(wallet_id="concierge"))[-1]
    assert (last.kind, last.reservation_id) == (TransferKind.EXPIRE, old.reservation_id)


# ---------- reads / persistence ----------


async def test_journal_filters(ledger: Ledger) -> None:
    await hold(ledger, 1_000, trip_id="a")
    await hold(ledger, 1_000, payee="local", trip_id="b")
    assert {t.trip_id for t in await ledger.journal(trip_id="a")} == {"a"}
    assert len(await ledger.journal(wallet_id="local")) == 0  # holds name only the payer
    assert len(await ledger.journal(wallet_id="concierge", trip_id="b")) == 1


async def test_daily_spend_unknown_wallet(ledger: Ledger) -> None:
    with pytest.raises(UnknownWallet):
        await ledger.daily_spend("nobody")


async def test_state_survives_reopen(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "ledger.db"
    async with await Ledger.open(path, clock=clock) as led:
        await seed(led)
        r = await hold(led, 3_000)
    async with await Ledger.open(path, clock=clock) as led:
        assert (await led.get_wallet("concierge")).reserved_mc == 3_000
        await led.capture(r.reservation_id, 3_000)
        assert (await led.get_wallet("travel")).available_mc == 3_000


def test_timestamps_are_fixed_width_utc() -> None:
    whole = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert _ts(whole) < _ts(whole + timedelta(microseconds=1))
    assert len(_ts(whole)) == len(_ts(whole + timedelta(microseconds=1)))
    with pytest.raises(ValueError):
        _ts(datetime(2026, 1, 1))


def test_hold_seconds_must_be_positive() -> None:
    with pytest.raises(ValueError):
        Ledger(conn=None, default_hold_seconds=0)  # type: ignore[arg-type]
