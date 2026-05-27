"""End-to-end MCP tests for the Concierge server."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pytest_asyncio
from mcp.shared.memory import create_connected_server_and_client_session

from agents_concierge.models import UserPreferences
from agents_concierge.store.sqlite_store import ConciergeStore
from mcp_servers.concierge import build_server
from shared.models import (
    FlightOption,
    HotelOption,
    Interest,
    Itinerary,
    TripRequest,
)


@pytest_asyncio.fixture
async def store(tmp_path: Path) -> AsyncIterator[ConciergeStore]:
    async with await ConciergeStore.open(tmp_path / "mcp.db") as s:
        yield s


def _itinerary() -> Itinerary:
    return Itinerary(
        trip_request=TripRequest(
            user_id="alice",
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


class TestListTools:
    async def test_lists_four_tools(self, store: ConciergeStore) -> None:
        server = build_server(store)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.list_tools()
        names = {t.name for t in result.tools}
        assert names == {
            "get_user_preferences",
            "save_user_preferences",
            "save_itinerary",
            "list_itineraries",
        }


class TestGetUserPreferencesTool:
    async def test_unknown_user_returns_defaults(self, store: ConciergeStore) -> None:
        server = build_server(store)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(
                "get_user_preferences", {"user_id": "nobody"}
            )
        payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
        assert payload["user_id"] == "nobody"
        assert payload["home_airport"] is None
        assert payload["preferred_interests"] == []


class TestSaveUserPreferencesTool:
    async def test_persists_via_mcp(self, store: ConciergeStore) -> None:
        prefs = UserPreferences(
            user_id="alice",
            home_airport="ATL",
            preferred_interests=(Interest.FOOD,),
        )
        server = build_server(store)
        async with create_connected_server_and_client_session(server) as session:
            saved = await session.call_tool(
                "save_user_preferences",
                {"prefs": json.loads(prefs.model_dump_json())},
            )
            got = await session.call_tool(
                "get_user_preferences", {"user_id": "alice"}
            )
        assert not saved.isError
        payload = json.loads(got.content[0].text)  # type: ignore[union-attr]
        assert payload["home_airport"] == "ATL"
        assert payload["preferred_interests"] == ["food"]


class TestSaveItineraryTool:
    async def test_saves_and_returns_record(self, store: ConciergeStore) -> None:
        server = build_server(store)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(
                "save_itinerary",
                {
                    "user_id": "alice",
                    "itinerary": json.loads(_itinerary().model_dump_json()),
                },
            )
        payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
        assert payload["id"] >= 1
        assert payload["user_id"] == "alice"
        assert payload["itinerary"]["summary"] == "4 nights in Lisbon"


class TestListItinerariesTool:
    async def test_empty_for_unknown(self, store: ConciergeStore) -> None:
        server = build_server(store)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(
                "list_itineraries", {"user_id": "noone"}
            )
        payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
        assert payload == []

    async def test_returns_saved(self, store: ConciergeStore) -> None:
        server = build_server(store)
        async with create_connected_server_and_client_session(server) as session:
            await session.call_tool(
                "save_itinerary",
                {
                    "user_id": "alice",
                    "itinerary": json.loads(_itinerary().model_dump_json()),
                },
            )
            await session.call_tool(
                "save_itinerary",
                {
                    "user_id": "alice",
                    "itinerary": json.loads(_itinerary().model_dump_json()),
                },
            )
            result = await session.call_tool(
                "list_itineraries", {"user_id": "alice"}
            )
        payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
        assert len(payload) == 2
        # Newest first → id descending.
        assert payload[0]["id"] > payload[1]["id"]