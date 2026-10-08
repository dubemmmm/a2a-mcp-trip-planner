"""Async SQLite ledger: wallets, quotes, escrowed reservations, and a journal.

The ledger is the only component allowed to change balances. Its contract:

  * Conservation — every transfer debits exactly what it credits; the total
    across wallets changes only when the treasury mints an allowance.
  * No negative balances — enforced by CHECK constraints and conditional
    UPDATEs (`... WHERE available_mc >= ?`).
  * A reservation changes state at most once, from HELD to a terminal state.
  * Captured amount <= reserved amount.
  * A payer's daily spend (captures + live holds since 00:00 UTC) stays under
    its `daily_cap_mc`.

Atomicity: every write runs inside `BEGIN IMMEDIATE`, which takes SQLite's
write lock up front, so two processes cannot both spend the last credits. An
asyncio lock additionally serializes writers sharing this one connection.

Idempotency: repeating an operation returns the first result instead of
applying it again. Journal keys are derived (`reserve:{quote_id}`,
`capture:{reservation_id}`, ...) and UNIQUE, so the database refuses a double
apply even if the code above it were wrong.

    async with await Ledger.open("./ledger.db") as ledger:
        await ledger.ensure_wallet("concierge", daily_cap_mc=50_000)
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, Self
from uuid import uuid4

import aiosqlite
import structlog

from .errors import (
    CaptureExceedsHold,
    DailyCapExceeded,
    IdempotencyConflict,
    InsufficientFunds,
    QuoteAlreadyReserved,
    QuoteExpired,
    ReservationNotHeld,
    TaskAlreadyBound,
    UnknownQuote,
    UnknownReservation,
    UnknownWallet,
)
from .models import (
    TREASURY,
    Quote,
    Reservation,
    ReservationState,
    Transfer,
    TransferKind,
    Wallet,
)

_LOGGER = structlog.get_logger(__name__)

DEFAULT_HOLD_SECONDS = 180.0

Clock = Callable[[], datetime]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS wallets (
    wallet_id     TEXT    PRIMARY KEY,
    available_mc  INTEGER NOT NULL DEFAULT 0 CHECK (available_mc >= 0),
    reserved_mc   INTEGER NOT NULL DEFAULT 0 CHECK (reserved_mc >= 0),
    daily_cap_mc  INTEGER CHECK (daily_cap_mc IS NULL OR daily_cap_mc >= 0),
    created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS quotes (
    quote_id    TEXT    PRIMARY KEY,
    payer       TEXT    NOT NULL REFERENCES wallets(wallet_id),
    payee       TEXT    NOT NULL REFERENCES wallets(wallet_id),
    skill_id    TEXT    NOT NULL,
    amount_mc   INTEGER NOT NULL CHECK (amount_mc > 0),
    expires_at  TEXT    NOT NULL,
    created_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS reservations (
    reservation_id  TEXT    PRIMARY KEY,
    quote_id        TEXT    NOT NULL UNIQUE REFERENCES quotes(quote_id),
    payer           TEXT    NOT NULL REFERENCES wallets(wallet_id),
    payee           TEXT    NOT NULL REFERENCES wallets(wallet_id),
    skill_id        TEXT    NOT NULL,
    amount_mc       INTEGER NOT NULL CHECK (amount_mc > 0),
    captured_mc     INTEGER CHECK (captured_mc IS NULL OR
                                   (captured_mc >= 0 AND captured_mc <= amount_mc)),
    trip_id         TEXT,
    a2a_task_id     TEXT,
    state           TEXT    NOT NULL
                    CHECK (state IN ('held', 'captured', 'released', 'expired')),
    expires_at      TEXT    NOT NULL,
    created_at      TEXT    NOT NULL,
    settled_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_reservations_held
    ON reservations(state, expires_at);
CREATE INDEX IF NOT EXISTS idx_reservations_payer_day
    ON reservations(payer, created_at);

CREATE TABLE IF NOT EXISTS transfers (
    transfer_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    idempotency_key  TEXT    NOT NULL UNIQUE,
    kind             TEXT    NOT NULL,
    from_wallet      TEXT,
    to_wallet        TEXT,
    amount_mc        INTEGER NOT NULL CHECK (amount_mc >= 0),
    reservation_id   TEXT    REFERENCES reservations(reservation_id),
    trip_id          TEXT,
    created_at       TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_transfers_trip ON transfers(trip_id);
CREATE INDEX IF NOT EXISTS idx_transfers_reservation ON transfers(reservation_id);
"""

# Fixed-width UTC timestamps so string comparison in SQL is chronological.
# (`datetime.isoformat()` drops the microseconds when they are zero.)
_TS_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ts(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("ledger timestamps must be timezone-aware")
    return dt.astimezone(timezone.utc).strftime(_TS_FORMAT)


def _parse_ts(value: str) -> datetime:
    return datetime.strptime(value, _TS_FORMAT).replace(tzinfo=timezone.utc)


def _day_bounds(day: date) -> tuple[str, str]:
    start = datetime.combine(day, time.min, tzinfo=timezone.utc)
    return _ts(start), _ts(start + timedelta(days=1))


class Ledger:
    """Async SQLite ledger. One instance owns one connection."""

    def __init__(
        self,
        conn: aiosqlite.Connection,
        *,
        clock: Clock = _utcnow,
        default_hold_seconds: float = DEFAULT_HOLD_SECONDS,
    ) -> None:
        if default_hold_seconds <= 0:
            raise ValueError("default_hold_seconds must be positive")
        self._conn = conn
        self._clock = clock
        self._default_hold_seconds = default_hold_seconds
        self._write_lock = asyncio.Lock()

    @classmethod
    async def open(
        cls,
        db_path: str | Path,
        *,
        clock: Clock = _utcnow,
        default_hold_seconds: float = DEFAULT_HOLD_SECONDS,
    ) -> "Ledger":
        """Open the ledger at `db_path` (creating it if absent).

        Pass ":memory:" for a throwaway ledger in tests.
        """
        path = str(db_path)
        # isolation_level=None: no implicit transactions; every write opens an
        # explicit BEGIN IMMEDIATE (see `_write`).
        conn = await aiosqlite.connect(path, isolation_level=None)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA foreign_keys = ON")
        await conn.execute("PRAGMA busy_timeout = 5000")
        if path != ":memory:":
            await conn.execute("PRAGMA journal_mode = WAL")
        await conn.executescript(_SCHEMA)
        return cls(conn, clock=clock, default_hold_seconds=default_hold_seconds)

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
        await self._conn.close()

    @asynccontextmanager
    async def _write(self) -> AsyncIterator[aiosqlite.Connection]:
        """One serialized, all-or-nothing write transaction."""
        async with self._write_lock:
            await self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                await self._conn.execute("ROLLBACK")
                raise
            else:
                await self._conn.execute("COMMIT")

    # ---------------------------------------------------------------- wallets

    async def ensure_wallet(self, wallet_id: str, *, daily_cap_mc: int | None = None) -> Wallet:
        """Create the wallet if absent; always (re)apply its daily cap."""
        if not wallet_id.strip():
            raise ValueError("wallet_id must be non-empty")
        if wallet_id == TREASURY:
            raise ValueError(f"'{TREASURY}' is reserved for the allowance issuer")
        if daily_cap_mc is not None and daily_cap_mc < 0:
            raise ValueError("daily_cap_mc must be >= 0")

        async with self._write() as conn:
            await conn.execute(
                """
                INSERT INTO wallets (wallet_id, daily_cap_mc, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(wallet_id) DO UPDATE SET daily_cap_mc = excluded.daily_cap_mc
                """,
                (wallet_id, daily_cap_mc, _ts(self._clock())),
            )
            return await self._wallet(conn, wallet_id)

    async def get_wallet(self, wallet_id: str) -> Wallet:
        return await self._wallet(self._conn, wallet_id)

    async def grant_allowance(self, wallet_id: str, amount_mc: int, *, day: date) -> Transfer:
        """Mint `amount_mc` into a wallet, at most once per wallet per day."""
        if amount_mc <= 0:
            raise ValueError("amount_mc must be positive")
        key = f"allowance:{wallet_id}:{day.isoformat()}"

        async with self._write() as conn:
            existing = await self._transfer_by_key(conn, key)
            if existing is not None:
                if existing.amount_mc != amount_mc:
                    raise IdempotencyConflict(
                        f"allowance for {wallet_id} on {day} was already "
                        f"granted as {existing.amount_mc} mc"
                    )
                return existing

            await self._wallet(conn, wallet_id)  # raises UnknownWallet
            await conn.execute(
                "UPDATE wallets SET available_mc = available_mc + ? WHERE wallet_id = ?",
                (amount_mc, wallet_id),
            )
            transfer = await self._journal(
                conn,
                key=key,
                kind=TransferKind.ALLOWANCE,
                from_wallet=TREASURY,
                to_wallet=wallet_id,
                amount_mc=amount_mc,
            )
        _LOGGER.info("ledger_allowance", wallet=wallet_id, amount_mc=amount_mc, day=str(day))
        return transfer

    # ----------------------------------------------------------------- quotes

    async def create_quote(
        self,
        *,
        payer: str,
        payee: str,
        skill_id: str,
        amount_mc: int,
        ttl_seconds: float,
    ) -> Quote:
        """Record an offer from `payee` to do `skill_id` for `payer`."""
        if payer == payee:
            raise ValueError("payer and payee must differ")
        if not skill_id.strip():
            raise ValueError("skill_id must be non-empty")
        if amount_mc <= 0:
            raise ValueError("amount_mc must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")

        now = self._clock()
        quote = Quote(
            quote_id=f"q_{uuid4().hex}",
            payer=payer,
            payee=payee,
            skill_id=skill_id,
            amount_mc=amount_mc,
            expires_at=now + timedelta(seconds=ttl_seconds),
            created_at=now,
        )
        async with self._write() as conn:
            await self._wallet(conn, payer)
            await self._wallet(conn, payee)
            await conn.execute(
                """
                INSERT INTO quotes
                    (quote_id, payer, payee, skill_id, amount_mc, expires_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    quote.quote_id,
                    payer,
                    payee,
                    skill_id,
                    amount_mc,
                    _ts(quote.expires_at),
                    _ts(quote.created_at),
                ),
            )
        _LOGGER.info(
            "ledger_quote",
            quote_id=quote.quote_id,
            payer=payer,
            payee=payee,
            skill_id=skill_id,
            amount_mc=amount_mc,
        )
        return quote

    async def get_quote(self, quote_id: str) -> Quote:
        return await self._quote(self._conn, quote_id)

    # ----------------------------------------------------------- reservations

    async def reserve(
        self,
        quote_id: str,
        *,
        trip_id: str | None = None,
        hold_seconds: float | None = None,
    ) -> Reservation:
        """Hold the quoted amount from the payer's available balance.

        Repeating the call for the same quote and trip returns the original
        reservation, whatever state it has reached since.
        """
        hold = self._default_hold_seconds if hold_seconds is None else hold_seconds
        if hold <= 0:
            raise ValueError("hold_seconds must be positive")

        async with self._write() as conn:
            quote = await self._quote(conn, quote_id)
            existing = await self._reservation_for_quote(conn, quote_id)
            if existing is not None:
                if existing.trip_id != trip_id:
                    raise QuoteAlreadyReserved(
                        f"quote {quote_id} is already reserved by {existing.reservation_id}"
                    )
                return existing

            now = self._clock()
            if now >= quote.expires_at:
                raise QuoteExpired(f"quote {quote_id} expired at {quote.expires_at}")

            payer = await self._wallet(conn, quote.payer)
            if payer.daily_cap_mc is not None:
                spent = await self._daily_spend(conn, payer.wallet_id, now.date())
                if spent + quote.amount_mc > payer.daily_cap_mc:
                    raise DailyCapExceeded(
                        f"{payer.wallet_id} has spent {spent} of its {payer.daily_cap_mc} mc "
                        f"daily cap; {quote.amount_mc} mc more would exceed it"
                    )

            cursor = await conn.execute(
                """
                UPDATE wallets
                   SET available_mc = available_mc - ?, reserved_mc = reserved_mc + ?
                 WHERE wallet_id = ? AND available_mc >= ?
                """,
                (quote.amount_mc, quote.amount_mc, payer.wallet_id, quote.amount_mc),
            )
            if cursor.rowcount == 0:
                raise InsufficientFunds(
                    f"{payer.wallet_id} has {payer.available_mc} mc available, "
                    f"needs {quote.amount_mc} mc"
                )

            reservation_id = f"r_{uuid4().hex}"
            await conn.execute(
                """
                INSERT INTO reservations
                    (reservation_id, quote_id, payer, payee, skill_id, amount_mc,
                     trip_id, state, expires_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reservation_id,
                    quote_id,
                    quote.payer,
                    quote.payee,
                    quote.skill_id,
                    quote.amount_mc,
                    trip_id,
                    ReservationState.HELD.value,
                    _ts(now + timedelta(seconds=hold)),
                    _ts(now),
                ),
            )
            await self._journal(
                conn,
                key=f"reserve:{quote_id}",
                kind=TransferKind.RESERVE,
                from_wallet=quote.payer,
                to_wallet=None,
                amount_mc=quote.amount_mc,
                reservation_id=reservation_id,
                trip_id=trip_id,
            )
            reservation = await self._reservation(conn, reservation_id)
        _LOGGER.info(
            "ledger_reserve",
            reservation_id=reservation.reservation_id,
            quote_id=quote_id,
            payer=reservation.payer,
            payee=reservation.payee,
            amount_mc=reservation.amount_mc,
            trip_id=trip_id,
        )
        return reservation

    async def get_reservation(self, reservation_id: str) -> Reservation:
        return await self._reservation(self._conn, reservation_id)

    async def bind_task(self, reservation_id: str, a2a_task_id: str) -> Reservation:
        """Record which A2A task this hold pays for. Each hold pays for one task."""
        if not a2a_task_id.strip():
            raise ValueError("a2a_task_id must be non-empty")

        expired = False
        async with self._write() as conn:
            reservation = await self._reservation(conn, reservation_id)
            if reservation.a2a_task_id == a2a_task_id:
                return reservation
            if reservation.a2a_task_id is not None:
                raise TaskAlreadyBound(
                    f"{reservation_id} already pays for task {reservation.a2a_task_id}"
                )
            if reservation.state is not ReservationState.HELD:
                raise ReservationNotHeld(f"{reservation_id} is {reservation.state.value}")
            if self._clock() >= reservation.expires_at:
                reservation = await self._return_hold(conn, reservation, ReservationState.EXPIRED)
                expired = True
            else:
                await conn.execute(
                    "UPDATE reservations SET a2a_task_id = ? WHERE reservation_id = ?",
                    (a2a_task_id, reservation_id),
                )
                reservation = await self._reservation(conn, reservation_id)
        if expired:
            raise ReservationNotHeld(f"{reservation_id} expired before the task was bound")
        return reservation

    async def capture(self, reservation_id: str, amount_mc: int) -> Reservation:
        """Pay the payee `amount_mc` from the hold and return the rest to the payer.

        A capture after the hold's expiry expires it instead (the payer gets
        everything back) and raises `ReservationNotHeld`.
        """
        if amount_mc < 0:
            raise ValueError("amount_mc must be >= 0")

        expired = False
        async with self._write() as conn:
            reservation = await self._reservation(conn, reservation_id)
            if reservation.state is ReservationState.CAPTURED:
                if reservation.captured_mc != amount_mc:
                    raise IdempotencyConflict(
                        f"{reservation_id} was already captured for {reservation.captured_mc} mc"
                    )
                return reservation
            if reservation.state is not ReservationState.HELD:
                raise ReservationNotHeld(f"{reservation_id} is {reservation.state.value}")
            if amount_mc > reservation.amount_mc:
                raise CaptureExceedsHold(
                    f"cannot capture {amount_mc} mc from a {reservation.amount_mc} mc hold"
                )

            now = self._clock()
            if now >= reservation.expires_at:
                reservation = await self._return_hold(conn, reservation, ReservationState.EXPIRED)
                expired = True
            else:
                refund_mc = reservation.amount_mc - amount_mc
                await conn.execute(
                    """
                    UPDATE wallets
                       SET reserved_mc = reserved_mc - ?, available_mc = available_mc + ?
                     WHERE wallet_id = ?
                    """,
                    (reservation.amount_mc, refund_mc, reservation.payer),
                )
                await conn.execute(
                    "UPDATE wallets SET available_mc = available_mc + ? WHERE wallet_id = ?",
                    (amount_mc, reservation.payee),
                )
                await conn.execute(
                    """
                    UPDATE reservations
                       SET state = ?, captured_mc = ?, settled_at = ?
                     WHERE reservation_id = ?
                    """,
                    (ReservationState.CAPTURED.value, amount_mc, _ts(now), reservation_id),
                )
                await self._journal(
                    conn,
                    key=f"capture:{reservation_id}",
                    kind=TransferKind.CAPTURE,
                    from_wallet=reservation.payer,
                    to_wallet=reservation.payee,
                    amount_mc=amount_mc,
                    reservation_id=reservation_id,
                    trip_id=reservation.trip_id,
                )
                if refund_mc > 0:
                    await self._journal(
                        conn,
                        key=f"refund:{reservation_id}",
                        kind=TransferKind.REFUND,
                        from_wallet=None,
                        to_wallet=reservation.payer,
                        amount_mc=refund_mc,
                        reservation_id=reservation_id,
                        trip_id=reservation.trip_id,
                    )
                reservation = await self._reservation(conn, reservation_id)

        if expired:
            raise ReservationNotHeld(f"{reservation_id} expired before it was captured")
        _LOGGER.info(
            "ledger_capture",
            reservation_id=reservation_id,
            payer=reservation.payer,
            payee=reservation.payee,
            captured_mc=amount_mc,
            refunded_mc=reservation.amount_mc - amount_mc,
            a2a_task_id=reservation.a2a_task_id,
            trip_id=reservation.trip_id,
        )
        return reservation

    async def release(self, reservation_id: str) -> Reservation:
        """Return the whole hold to the payer. Repeating it is a no-op."""
        async with self._write() as conn:
            reservation = await self._reservation(conn, reservation_id)
            if reservation.state is ReservationState.RELEASED:
                return reservation
            if reservation.state is not ReservationState.HELD:
                raise ReservationNotHeld(f"{reservation_id} is {reservation.state.value}")
            reservation = await self._return_hold(conn, reservation, ReservationState.RELEASED)
        _LOGGER.info(
            "ledger_release",
            reservation_id=reservation_id,
            payer=reservation.payer,
            amount_mc=reservation.amount_mc,
            trip_id=reservation.trip_id,
        )
        return reservation

    async def expire_stale(self) -> tuple[Reservation, ...]:
        """Expire every hold past its `expires_at`. Returns what it expired."""
        expired: list[Reservation] = []
        async with self._write() as conn:
            async with conn.execute(
                """
                SELECT * FROM reservations
                 WHERE state = ? AND expires_at <= ?
                 ORDER BY expires_at
                """,
                (ReservationState.HELD.value, _ts(self._clock())),
            ) as cursor:
                rows = await cursor.fetchall()
            for row in rows:
                stale = _reservation_from_row(row)
                expired.append(await self._return_hold(conn, stale, ReservationState.EXPIRED))
        if expired:
            _LOGGER.info("ledger_expired", count=len(expired))
        return tuple(expired)

    # ------------------------------------------------------------------ reads

    async def daily_spend(self, wallet_id: str, day: date | None = None) -> int:
        """Captures plus live holds for `wallet_id` on `day` (UTC; default today)."""
        await self._wallet(self._conn, wallet_id)
        return await self._daily_spend(self._conn, wallet_id, day or self._clock().date())

    async def journal(
        self,
        *,
        trip_id: str | None = None,
        wallet_id: str | None = None,
    ) -> tuple[Transfer, ...]:
        """Transfers in the order they happened, optionally filtered."""
        clauses: list[str] = []
        params: list[Any] = []
        if trip_id is not None:
            clauses.append("trip_id = ?")
            params.append(trip_id)
        if wallet_id is not None:
            clauses.append("(from_wallet = ? OR to_wallet = ?)")
            params.extend((wallet_id, wallet_id))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        async with self._conn.execute(
            f"SELECT * FROM transfers {where} ORDER BY transfer_id", params
        ) as cursor:
            rows = await cursor.fetchall()
        return tuple(_transfer_from_row(row) for row in rows)

    # ---------------------------------------------------------------- helpers

    async def _return_hold(
        self,
        conn: aiosqlite.Connection,
        reservation: Reservation,
        state: ReservationState,
    ) -> Reservation:
        """Move a HELD reservation's full amount back to the payer."""
        await conn.execute(
            """
            UPDATE wallets
               SET reserved_mc = reserved_mc - ?, available_mc = available_mc + ?
             WHERE wallet_id = ?
            """,
            (reservation.amount_mc, reservation.amount_mc, reservation.payer),
        )
        await conn.execute(
            "UPDATE reservations SET state = ?, settled_at = ? WHERE reservation_id = ?",
            (state.value, _ts(self._clock()), reservation.reservation_id),
        )
        kind = TransferKind.EXPIRE if state is ReservationState.EXPIRED else TransferKind.RELEASE
        await self._journal(
            conn,
            key=f"{kind.value}:{reservation.reservation_id}",
            kind=kind,
            from_wallet=None,
            to_wallet=reservation.payer,
            amount_mc=reservation.amount_mc,
            reservation_id=reservation.reservation_id,
            trip_id=reservation.trip_id,
        )
        return await self._reservation(conn, reservation.reservation_id)

    async def _journal(
        self,
        conn: aiosqlite.Connection,
        *,
        key: str,
        kind: TransferKind,
        from_wallet: str | None,
        to_wallet: str | None,
        amount_mc: int,
        reservation_id: str | None = None,
        trip_id: str | None = None,
    ) -> Transfer:
        try:
            cursor = await conn.execute(
                """
                INSERT INTO transfers
                    (idempotency_key, kind, from_wallet, to_wallet, amount_mc,
                     reservation_id, trip_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    kind.value,
                    from_wallet,
                    to_wallet,
                    amount_mc,
                    reservation_id,
                    trip_id,
                    _ts(self._clock()),
                ),
            )
        except sqlite3.IntegrityError as e:
            # The state checks above should make this unreachable; the UNIQUE
            # key is the backstop that refuses a double apply regardless.
            raise IdempotencyConflict(f"transfer {key} was already recorded") from e
        transfer = await self._transfer_by_key(conn, key)
        if transfer is None or cursor.lastrowid != transfer.transfer_id:
            raise RuntimeError(f"transfer {key} was not recorded")
        return transfer

    async def _daily_spend(self, conn: aiosqlite.Connection, wallet_id: str, day: date) -> int:
        start, end = _day_bounds(day)
        async with conn.execute(
            """
            SELECT COALESCE(SUM(CASE state
                                    WHEN 'held'     THEN amount_mc
                                    WHEN 'captured' THEN captured_mc
                                    ELSE 0 END), 0) AS spent
              FROM reservations
             WHERE payer = ? AND created_at >= ? AND created_at < ?
            """,
            (wallet_id, start, end),
        ) as cursor:
            row = await cursor.fetchone()
        return int(row["spent"]) if row is not None else 0

    async def _wallet(self, conn: aiosqlite.Connection, wallet_id: str) -> Wallet:
        async with conn.execute("SELECT * FROM wallets WHERE wallet_id = ?", (wallet_id,)) as cur:
            row = await cur.fetchone()
        if row is None:
            raise UnknownWallet(f"no wallet named {wallet_id!r}")
        return Wallet(
            wallet_id=row["wallet_id"],
            available_mc=row["available_mc"],
            reserved_mc=row["reserved_mc"],
            daily_cap_mc=row["daily_cap_mc"],
        )

    async def _quote(self, conn: aiosqlite.Connection, quote_id: str) -> Quote:
        async with conn.execute("SELECT * FROM quotes WHERE quote_id = ?", (quote_id,)) as cur:
            row = await cur.fetchone()
        if row is None:
            raise UnknownQuote(f"no quote {quote_id!r}")
        return Quote(
            quote_id=row["quote_id"],
            payer=row["payer"],
            payee=row["payee"],
            skill_id=row["skill_id"],
            amount_mc=row["amount_mc"],
            expires_at=_parse_ts(row["expires_at"]),
            created_at=_parse_ts(row["created_at"]),
        )

    async def _reservation(self, conn: aiosqlite.Connection, reservation_id: str) -> Reservation:
        async with conn.execute(
            "SELECT * FROM reservations WHERE reservation_id = ?", (reservation_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise UnknownReservation(f"no reservation {reservation_id!r}")
        return _reservation_from_row(row)

    async def _reservation_for_quote(
        self, conn: aiosqlite.Connection, quote_id: str
    ) -> Reservation | None:
        async with conn.execute(
            "SELECT * FROM reservations WHERE quote_id = ?", (quote_id,)
        ) as cur:
            row = await cur.fetchone()
        return None if row is None else _reservation_from_row(row)

    async def _transfer_by_key(self, conn: aiosqlite.Connection, key: str) -> Transfer | None:
        async with conn.execute("SELECT * FROM transfers WHERE idempotency_key = ?", (key,)) as cur:
            row = await cur.fetchone()
        return None if row is None else _transfer_from_row(row)


def _reservation_from_row(row: aiosqlite.Row) -> Reservation:
    return Reservation(
        reservation_id=row["reservation_id"],
        quote_id=row["quote_id"],
        payer=row["payer"],
        payee=row["payee"],
        skill_id=row["skill_id"],
        amount_mc=row["amount_mc"],
        captured_mc=row["captured_mc"],
        trip_id=row["trip_id"],
        a2a_task_id=row["a2a_task_id"],
        state=ReservationState(row["state"]),
        expires_at=_parse_ts(row["expires_at"]),
        created_at=_parse_ts(row["created_at"]),
        settled_at=_parse_ts(row["settled_at"]) if row["settled_at"] else None,
    )


def _transfer_from_row(row: aiosqlite.Row) -> Transfer:
    return Transfer(
        transfer_id=row["transfer_id"],
        idempotency_key=row["idempotency_key"],
        kind=TransferKind(row["kind"]),
        from_wallet=row["from_wallet"],
        to_wallet=row["to_wallet"],
        amount_mc=row["amount_mc"],
        reservation_id=row["reservation_id"],
        trip_id=row["trip_id"],
        created_at=_parse_ts(row["created_at"]),
    )
