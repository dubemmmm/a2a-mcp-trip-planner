"""Tests for the user_data tool layer.

The tools are thin façades — these tests verify they delegate correctly and
preserve types. Heavy correctness tests live in test_sqlite_store.py.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pytest_asyncio

from agents_concierge.models import UserPreferences
from agents_concierge.store.sqlite_store import ConciergeStore
from agents_concierge.tools.user_data import (
    get_user_preferences,
    list_itineraries,
    save_itinerary,
    save_user_preferences,
)
from shared.models import (
    FlightOption,
    HotelOption,
    Interest,
    Itinerary,
    TripRequest,
)


@pytest_asyncio.fixture
async def store(tmp_path: Path) -> AsyncIterator[ConciergeStore]:
    async with await ConciergeStore.open(tmp_path / "tools.db") as s:
        yield s


def _itinerary(user_id: str = "alice") -> Itinerary:
    return Itinerary(
        trip_request=TripRequest(
            user_id=user_id,
            destination="Lisbon",
            origin="ATL",
            start_date=date(2026, 5, 10),
            end_date=date(2026, 5, 14),
            budget_usd=Decimal("1500"),
        ),
        selected_flight=FlightOption(
            airline="TAP",
            flight_number="TP202",
            origin="ATL",
            destination="LIS",
            depart_at=datetime(2026, 5, 10, 18, 0),
            arrive_at=datetime(2026, 5, 11, 7, 30),
            price_usd=Decimal("612"),
        ),
        selected_hotel=HotelOption(
            name="Alfama Stay",
            city="Lisbon",
            check_in=date(2026, 5, 10),
            check_out=date(2026, 5, 14),
            price_per_night_usd=Decimal("120"),
        ),
        estimated_total_usd=Decimal("1092"),
        summary="4 nights in Lisbon",
    )


class TestGetUserPreferences:
    async def test_unknown_returns_default(self, store: ConciergeStore) -> None:
        prefs = await get_user_preferences(store, "noone")
        assert prefs == UserPreferences.default_for("noone")

    async def test_known_returns_stored(self, store: ConciergeStore) -> None:
        original = UserPreferences(user_id="alice", home_airport="ATL")
        await save_user_preferences(store, original)
        assert await get_user_preferences(store, "alice") == original


class TestSaveUserPreferences:
    async def test_persists_and_overwrites(self, store: ConciergeStore) -> None:
        await save_user_preferences(store, UserPreferences(user_id="alice", home_airport="ATL"))
        await save_user_preferences(store, UserPreferences(user_id="alice", home_airport="JFK"))
        assert (await get_user_preferences(store, "alice")).home_airport == "JFK"


class TestSaveItinerary:
    async def test_returns_stored_record(self, store: ConciergeStore) -> None:
        record = await save_itinerary(store, "alice", _itinerary())
        assert record.id >= 1
        assert record.user_id == "alice"

    async def test_appears_in_list(self, store: ConciergeStore) -> None:
        await save_itinerary(store, "alice", _itinerary())
        listed = await list_itineraries(store, "alice")
        assert len(listed) == 1


class TestListItineraries:
    async def test_empty_for_unknown_user(self, store: ConciergeStore) -> None:
        assert await list_itineraries(store, "nobody") == ()

    async def test_orders_newest_first(self, store: ConciergeStore) -> None:
        a = await save_itinerary(store, "alice", _itinerary())
        b = await save_itinerary(store, "alice", _itinerary())
        listed = await list_itineraries(store, "alice")
        assert listed[0].id == b.id
        assert listed[1].id == a.id

    async def test_isolated_per_user(self, store: ConciergeStore) -> None:
        await save_itinerary(store, "alice", _itinerary("alice"))
        await save_itinerary(store, "bob", _itinerary("bob"))
        alice_only = await list_itineraries(store, "alice")
        assert len(alice_only) == 1
        assert alice_only[0].user_id == "alice"