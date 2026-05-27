"""Tests for the Concierge SQLite store."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pytest
import pytest_asyncio

from agents_concierge.models import UserPreferences
from agents_concierge.store.sqlite_store import ConciergeStore
from shared.models import (
    DailyPlan,
    FlightOption,
    HotelOption,
    Interest,
    Itinerary,
    PriceLevel,
    TripRequest,
)


# ---------- fixtures ----------

@pytest_asyncio.fixture
async def store(tmp_path: Path) -> AsyncIterator[ConciergeStore]:
    db_path = tmp_path / "concierge.db"
    s = await ConciergeStore.open(db_path)
    try:
        yield s
    finally:
        await s.close()


def _trip() -> TripRequest:
    return TripRequest(
        user_id="alice",
        destination="Lisbon",
        origin="ATL",
        start_date=date(2026, 5, 10),
        end_date=date(2026, 5, 14),
        budget_usd=Decimal("1500"),
        interests=(Interest.FOOD, Interest.HISTORY),
    )


def _itinerary() -> Itinerary:
    trip = _trip()
    flight = FlightOption(
        airline="TAP",
        flight_number="TP202",
        origin="ATL",
        destination="LIS",
        depart_at=datetime(2026, 5, 10, 18, 0),
        arrive_at=datetime(2026, 5, 11, 7, 30),
        price_usd=Decimal("612"),
    )
    hotel = HotelOption(
        name="Alfama Stay",
        city="Lisbon",
        check_in=date(2026, 5, 10),
        check_out=date(2026, 5, 14),
        price_per_night_usd=Decimal("120"),
    )
    return Itinerary(
        trip_request=trip,
        selected_flight=flight,
        selected_hotel=hotel,
        daily_plans=(DailyPlan(day=date(2026, 5, 11)),),
        estimated_total_usd=Decimal("1092"),
        summary="4 nights in Lisbon",
    )


# ---------- open / lifecycle ----------

class TestOpen:
    async def test_creates_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "fresh.db"
        assert not db_path.exists()
        s = await ConciergeStore.open(db_path)
        try:
            assert db_path.exists()
        finally:
            await s.close()

    async def test_idempotent_schema(self, tmp_path: Path) -> None:
        """Opening the same file twice must not re-create or fail."""
        db_path = tmp_path / "shared.db"
        s1 = await ConciergeStore.open(db_path)
        await s1.upsert_user_preferences(UserPreferences(user_id="a", home_airport="ATL"))
        await s1.close()

        s2 = await ConciergeStore.open(db_path)
        try:
            prefs = await s2.get_user_preferences("a")
            assert prefs.home_airport == "ATL"
        finally:
            await s2.close()

    async def test_context_manager(self, tmp_path: Path) -> None:
        async with await ConciergeStore.open(tmp_path / "cm.db") as s:
            await s.upsert_user_preferences(UserPreferences(user_id="b"))
            assert (await s.get_user_preferences("b")).user_id == "b"


# ---------- user preferences ----------

class TestUserPreferences:
    async def test_unknown_user_returns_defaults(self, store: ConciergeStore) -> None:
        prefs = await store.get_user_preferences("nobody")
        assert prefs.user_id == "nobody"
        assert prefs.home_airport is None
        assert prefs.preferred_interests == ()
        assert prefs.preferred_price_level is None

    async def test_upsert_then_get(self, store: ConciergeStore) -> None:
        prefs = UserPreferences(
            user_id="alice",
            home_airport="ATL",
            preferred_interests=(Interest.FOOD, Interest.NATURE),
            preferred_price_level=PriceLevel.MODERATE,
            dietary_restrictions=("vegetarian",),
        )
        await store.upsert_user_preferences(prefs)
        rebuilt = await store.get_user_preferences("alice")
        assert rebuilt == prefs

    async def test_upsert_overwrites_existing(self, store: ConciergeStore) -> None:
        await store.upsert_user_preferences(
            UserPreferences(user_id="alice", home_airport="ATL")
        )
        await store.upsert_user_preferences(
            UserPreferences(user_id="alice", home_airport="JFK", notes="moved")
        )
        rebuilt = await store.get_user_preferences("alice")
        assert rebuilt.home_airport == "JFK"
        assert rebuilt.notes == "moved"

    async def test_multiple_users_isolated(self, store: ConciergeStore) -> None:
        await store.upsert_user_preferences(
            UserPreferences(user_id="alice", home_airport="ATL")
        )
        await store.upsert_user_preferences(
            UserPreferences(user_id="bob", home_airport="JFK")
        )
        assert (await store.get_user_preferences("alice")).home_airport == "ATL"
        assert (await store.get_user_preferences("bob")).home_airport == "JFK"

    async def test_empty_user_id_rejected(self, store: ConciergeStore) -> None:
        with pytest.raises(ValueError):
            await store.get_user_preferences("")
        with pytest.raises(ValueError):
            await store.get_user_preferences("   ")


# ---------- itineraries ----------

class TestItineraries:
    async def test_save_returns_record_with_id(self, store: ConciergeStore) -> None:
        record = await store.save_itinerary("alice", _itinerary())
        assert record.id >= 1
        assert record.user_id == "alice"
        assert record.itinerary == _itinerary()

    async def test_ids_increment(self, store: ConciergeStore) -> None:
        a = await store.save_itinerary("alice", _itinerary())
        b = await store.save_itinerary("alice", _itinerary())
        assert b.id == a.id + 1

    async def test_list_newest_first(self, store: ConciergeStore) -> None:
        a = await store.save_itinerary("alice", _itinerary())
        b = await store.save_itinerary("alice", _itinerary())
        listed = await store.list_itineraries("alice")
        assert tuple(r.id for r in listed) == (b.id, a.id)

    async def test_list_unknown_user_returns_empty(
        self, store: ConciergeStore
    ) -> None:
        # Defaults-not-errors contract.
        assert await store.list_itineraries("nobody") == ()

    async def test_list_per_user(self, store: ConciergeStore) -> None:
        await store.save_itinerary("alice", _itinerary())
        await store.save_itinerary("bob", _itinerary())
        await store.save_itinerary("alice", _itinerary())
        assert len(await store.list_itineraries("alice")) == 2
        assert len(await store.list_itineraries("bob")) == 1

    async def test_itinerary_round_trip(self, store: ConciergeStore) -> None:
        saved = await store.save_itinerary("alice", _itinerary())
        listed = await store.list_itineraries("alice")
        assert listed[0].itinerary == saved.itinerary

    async def test_empty_user_id_rejected(self, store: ConciergeStore) -> None:
        with pytest.raises(ValueError):
            await store.save_itinerary("", _itinerary())
        with pytest.raises(ValueError):
            await store.list_itineraries("")
