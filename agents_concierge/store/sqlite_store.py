"""Async SQLite-backed persistence for the Concierge agent.

Two tables:
  * `user_preferences` — one row per user, JSON blob of typed fields.
  * `itineraries`      — append-only log of trips the user has saved.

Why a single-blob `data` column instead of normalized fields: the data is read
together, written together, never queried by sub-field, and Pydantic already
gives us schema validation on read. Trying to normalize Interest tuples into
junction tables would just be ceremony.

A `ConciergeStore` owns one aiosqlite connection. Use it as an async context
manager so cleanup is automatic:

    async with ConciergeStore.open("./concierge.db") as store:
        prefs = await store.get_user_preferences("alice")
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Self

import aiosqlite

from shared.models import Itinerary

from ..models import StoredItinerary, UserPreferences

_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id TEXT PRIMARY KEY,
    data    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS itineraries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT    NOT NULL,
    data        TEXT    NOT NULL,
    created_at  TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_itineraries_user_id
    ON itineraries(user_id);
"""


class ConciergeStore:
    """Async SQLite store for Concierge user data."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    @classmethod
    async def open(cls, db_path: str | Path) -> "ConciergeStore":
        """Open the DB at `db_path` (creating it if absent) and run the schema."""
        path = str(db_path)
        conn = await aiosqlite.connect(path)
        # Set row factory so cursor results behave dict-like.
        conn.row_factory = aiosqlite.Row
        await conn.executescript(_SCHEMA)
        await conn.commit()
        return cls(conn)

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

    # ------------------------------------------------------------------ prefs

    async def get_user_preferences(self, user_id: str) -> UserPreferences:
        """Return stored preferences, or a permissive default for unknown users.

        Defaults-not-errors is the contract per the build plan: the Concierge
        will call this on first-time users, and we don't want them to need
        a separate "does this user exist?" check.
        """
        if not user_id.strip():
            raise ValueError("user_id must be non-empty")

        async with self._conn.execute(
            "SELECT data FROM user_preferences WHERE user_id = ?", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return UserPreferences.default_for(user_id)
        return UserPreferences.model_validate_json(row["data"])

    async def upsert_user_preferences(self, prefs: UserPreferences) -> None:
        """Insert or replace a user's preferences."""
        payload = prefs.model_dump_json()
        await self._conn.execute(
            """
            INSERT INTO user_preferences (user_id, data)
            VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET data = excluded.data
            """,
            (prefs.user_id, payload),
        )
        await self._conn.commit()

    # ------------------------------------------------------------- itineraries

    async def save_itinerary(self, user_id: str, itinerary: Itinerary) -> StoredItinerary:
        """Append an itinerary for `user_id`. Returns the persisted record."""
        if not user_id.strip():
            raise ValueError("user_id must be non-empty")

        created_at = datetime.now(timezone.utc)
        cursor = await self._conn.execute(
            """
            INSERT INTO itineraries (user_id, data, created_at)
            VALUES (?, ?, ?)
            """,
            (user_id, itinerary.model_dump_json(), created_at.isoformat()),
        )
        await self._conn.commit()
        row_id = cursor.lastrowid
        if row_id is None:
            # aiosqlite always populates lastrowid on AUTOINCREMENT inserts;
            # this branch is defensive against driver pathologies.
            raise RuntimeError("INSERT did not return a row id")
        return StoredItinerary(
            id=row_id, user_id=user_id, itinerary=itinerary, created_at=created_at
        )

    async def list_itineraries(self, user_id: str) -> tuple[StoredItinerary, ...]:
        """List a user's saved itineraries, newest first.

        Unknown users return an empty tuple — consistent with the
        defaults-not-errors policy on `get_user_preferences`.
        """
        if not user_id.strip():
            raise ValueError("user_id must be non-empty")

        async with self._conn.execute(
            """
            SELECT id, user_id, data, created_at
              FROM itineraries
             WHERE user_id = ?
             ORDER BY id DESC
            """,
            (user_id,),
        ) as cursor:
            rows = await cursor.fetchall()

        return tuple(
            StoredItinerary(
                id=int(row["id"]),
                user_id=row["user_id"],
                itinerary=Itinerary.model_validate_json(row["data"]),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        )