"""Ledger invariants under random operation sequences, and under concurrency.

The random tests are a hand-rolled, seeded state machine: each seed drives a
few hundred random operations (valid and invalid) and every invariant is
checked after every step. A failure prints the seed and step, so it replays
exactly.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from economy import (
    InsufficientFunds,
    Ledger,
    LedgerError,
    Reservation,
    ReservationState,
    TransferKind,
)

from ._fixtures import FakeClock, quote

WALLETS = ("concierge", "travel", "local", "transit")
PAYERS = ("concierge", "travel")
CAPS = {"concierge": 30_000, "travel": None, "local": None, "transit": None}
HOLD_SECONDS = 120
STEPS = 200


class Model:
    """What the test remembers between steps, to check invariants against."""

    def __init__(self) -> None:
        self.quote_ids: list[str] = []
        self.reservation_ids: list[str] = []
        self.terminal: dict[str, Reservation] = {}


async def _step(ledger: Ledger, clock: FakeClock, rng: random.Random, m: Model) -> None:
    op = rng.choice(
        [
            "allowance",
            "quote",
            "quote",
            "reserve",
            "reserve",
            "bind",
            "capture",
            "capture",
            "release",
            "expire",
            "tick",
        ]
    )
    try:
        if op == "allowance":
            wallet = rng.choice(PAYERS)
            day = clock.now.date()
            # Same wallet + day may repeat (idempotent) or conflict on amount.
            await ledger.grant_allowance(wallet, rng.choice([5_000, 10_000]), day=day)
        elif op == "quote":
            payer = rng.choice(PAYERS)
            payee = rng.choice([w for w in WALLETS if w != payer])
            q = await ledger.create_quote(
                payer=payer,
                payee=payee,
                skill_id="s",
                amount_mc=rng.randint(1, 12_000),
                ttl_seconds=rng.choice([1, 30, 90]),
            )
            m.quote_ids.append(q.quote_id)
        elif op == "reserve" and m.quote_ids:
            r = await ledger.reserve(
                rng.choice(m.quote_ids),
                trip_id=rng.choice([None, "t1", "t2"]),
                hold_seconds=rng.choice([None, 1, 60]),
            )
            if r.reservation_id not in m.reservation_ids:
                m.reservation_ids.append(r.reservation_id)
        elif op == "bind" and m.reservation_ids:
            await ledger.bind_task(rng.choice(m.reservation_ids), rng.choice(["a", "b"]))
        elif op == "capture" and m.reservation_ids:
            rid = rng.choice(m.reservation_ids)
            held = await ledger.get_reservation(rid)
            await ledger.capture(rid, rng.randint(0, held.amount_mc + 500))
        elif op == "release" and m.reservation_ids:
            await ledger.release(rng.choice(m.reservation_ids))
        elif op == "expire":
            await ledger.expire_stale()
        elif op == "tick":
            clock.advance(rng.choice([1, 20, 70, 3_600, 12 * 3_600]))
    except LedgerError:
        pass  # refusals are expected; the invariants must hold regardless


async def _check(ledger: Ledger, m: Model, where: str) -> None:
    wallets = {w: await ledger.get_wallet(w) for w in WALLETS}
    journal = await ledger.journal()
    reservations = [await ledger.get_reservation(rid) for rid in m.reservation_ids]

    # 1. Conservation: money only enters through allowances.
    minted = sum(t.amount_mc for t in journal if t.kind is TransferKind.ALLOWANCE)
    held_total = sum(w.available_mc + w.reserved_mc for w in wallets.values())
    assert held_total == minted, f"{where}: conservation broken"

    # 2. No negatives (also enforced by the models and CHECK constraints).
    for w in wallets.values():
        assert w.available_mc >= 0 and w.reserved_mc >= 0, f"{where}: negative {w}"

    # 3. Reserved balance == sum of live holds, per payer.
    for name, w in wallets.items():
        live = sum(
            r.amount_mc
            for r in reservations
            if r.payer == name and r.state is ReservationState.HELD
        )
        assert w.reserved_mc == live, f"{where}: {name} reserved {w.reserved_mc} != {live}"

    # 4. The journal replays to every available balance.
    for name, w in wallets.items():
        replay = 0
        for t in journal:
            if t.to_wallet == name:
                replay += t.amount_mc
            if t.from_wallet == name and t.kind is TransferKind.RESERVE:
                replay -= t.amount_mc
        assert w.available_mc == replay, f"{where}: {name} journal replays to {replay}"

    # 5. Captures never exceed holds; terminal states never change.
    for r in reservations:
        if r.captured_mc is not None:
            assert r.captured_mc <= r.amount_mc, f"{where}: over-capture {r}"
        if r.reservation_id in m.terminal:
            assert r == m.terminal[r.reservation_id], f"{where}: terminal state changed"
        elif r.state is not ReservationState.HELD:
            m.terminal[r.reservation_id] = r

    # 6. Capped payers never commit more than their cap in a day.
    for name, cap in CAPS.items():
        if cap is not None:
            days = {r.created_at.date() for r in reservations if r.payer == name}
            for day in days:
                spent = await ledger.daily_spend(name, day)
                assert spent <= cap, f"{where}: {name} spent {spent} on {day}"


SEEDS = range(20)


async def _run(seed: int, *, check: bool) -> Model:
    rng = random.Random(seed)
    clock = FakeClock()
    async with await Ledger.open(
        ":memory:", clock=clock, default_hold_seconds=HOLD_SECONDS
    ) as ledger:
        for name in WALLETS:
            await ledger.ensure_wallet(name, daily_cap_mc=CAPS[name])
        await ledger.grant_allowance("concierge", 40_000, day=clock.now.date())
        model = Model()
        for step in range(STEPS):
            await _step(ledger, clock, rng, model)
            if check:
                await _check(ledger, model, f"seed={seed} step={step}")
            elif step == STEPS - 1:
                await _check(ledger, model, f"seed={seed} final")
    return model


@pytest.mark.parametrize("seed", SEEDS)
async def test_invariants_hold_under_random_operations(seed: int) -> None:
    await _run(seed, check=True)


async def test_random_runs_reach_every_terminal_state() -> None:
    """Guard against a generator that only ever produces refusals."""
    reached: set[ReservationState] = set()
    for seed in SEEDS:
        model = await _run(seed, check=False)
        reached |= {r.state for r in model.terminal.values()}
    assert reached == {
        ReservationState.CAPTURED,
        ReservationState.RELEASED,
        ReservationState.EXPIRED,
    }


# ---------- concurrency ----------


@pytest_asyncio.fixture
async def funded(tmp_path: Path) -> AsyncIterator[Path]:
    """A ledger file where the concierge can afford exactly five 1000 mc holds."""
    path = tmp_path / "ledger.db"
    async with await Ledger.open(path) as led:
        await led.ensure_wallet("concierge")
        await led.ensure_wallet("travel")
        await led.grant_allowance("concierge", 5_000, day=FakeClock().now.date())
    yield path


async def _race(ledgers: list[Ledger], attempts: int) -> list[object]:
    quotes = [await quote(ledgers[0], 1_000) for _ in range(attempts)]
    return await asyncio.gather(
        *(ledgers[i % len(ledgers)].reserve(q.quote_id) for i, q in enumerate(quotes)),
        return_exceptions=True,
    )


async def test_parallel_reserves_on_one_connection_never_overspend(funded: Path) -> None:
    async with await Ledger.open(funded) as led:
        results = await _race([led], attempts=20)
        wallet = await led.get_wallet("concierge")
    assert sum(isinstance(r, Reservation) for r in results) == 5
    assert all(isinstance(r, (Reservation, InsufficientFunds)) for r in results)
    assert (wallet.available_mc, wallet.reserved_mc) == (0, 5_000)


async def test_parallel_reserves_across_connections_never_overspend(funded: Path) -> None:
    """Two ledgers on one file: only BEGIN IMMEDIATE stands between them."""
    async with await Ledger.open(funded) as a, await Ledger.open(funded) as b:
        results = await _race([a, b], attempts=20)
        wallet = await a.get_wallet("concierge")
    assert sum(isinstance(r, Reservation) for r in results) == 5
    assert all(isinstance(r, (Reservation, InsufficientFunds)) for r in results)
    assert (wallet.available_mc, wallet.reserved_mc) == (0, 5_000)


async def test_parallel_duplicate_captures_pay_once(funded: Path) -> None:
    async with await Ledger.open(funded) as a, await Ledger.open(funded) as b:
        q = await quote(a, 1_000)
        r = await a.reserve(q.quote_id)
        results = await asyncio.gather(
            *(led.capture(r.reservation_id, 800) for led in (a, b, a, b)),
            return_exceptions=True,
        )
        assert all(isinstance(x, Reservation) for x in results), results
        assert (await a.get_wallet("travel")).available_mc == 800
        assert (await a.get_wallet("concierge")).available_mc == 4_200
