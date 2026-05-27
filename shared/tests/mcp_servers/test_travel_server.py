"""End-to-end MCP tests for the Travel server."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import datetime
from decimal import Decimal
from typing import Any

import httpx
import pytest_asyncio
import respx
from mcp.shared.memory import create_connected_server_and_client_session

from agents_travel.clients.serpapi import _SEARCH_URL, SerpAPIClient
from mcp_servers.travel import build_server
from shared.models import FlightOption, HotelOption


def _flights_body(price: float = 612.0) -> dict[str, Any]:
    return {
        "best_flights": [
            {
                "flights": [
                    {
                        "departure_airport": {"id": "ATL", "time": "2026-05-10 18:00"},
                        "arrival_airport": {"id": "LIS", "time": "2026-05-11 07:30"},
                        "airline": "TAP",
                        "flight_number": "TP202",
                    }
                ],
                "price": price,
            }
        ],
        "other_flights": [],
    }


def _hotels_body() -> dict[str, Any]:
    return {
        "properties": [
            {
                "name": "Alfama Stay",
                "type": "hotel",
                "rate_per_night": {"extracted_lowest": 120.0},
                "overall_rating": 4.5,
                "address": "Lisbon",
            }
        ]
    }


@pytest_asyncio.fixture
async def serpapi_client() -> AsyncIterator[SerpAPIClient]:
    client = SerpAPIClient("test-serp")
    try:
        yield client
    finally:
        await client.aclose()


class TestListTools:
    async def test_lists_three_tools(self, serpapi_client: SerpAPIClient) -> None:
        server = build_server(serpapi_client)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.list_tools()
        names = {t.name for t in result.tools}
        assert names == {"search_flights", "search_hotels", "estimate_total_cost"}


class TestSearchFlightsTool:
    @respx.mock
    async def test_one_way(self, serpapi_client: SerpAPIClient) -> None:
        respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_flights_body())
        )
        server = build_server(serpapi_client)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(
                "search_flights",
                {
                    "origin": "ATL",
                    "destination": "LIS",
                    "depart_date": "2026-05-10",
                },
            )
        assert not result.isError
        payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
        assert len(payload) == 1
        assert payload[0]["airline"] == "TAP"
        assert payload[0]["origin"] == "ATL"

    @respx.mock
    async def test_max_price_filter(self, serpapi_client: SerpAPIClient) -> None:
        respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_flights_body(price=900.0))
        )
        server = build_server(serpapi_client)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(
                "search_flights",
                {
                    "origin": "ATL",
                    "destination": "LIS",
                    "depart_date": "2026-05-10",
                    "max_price_usd": 800,
                },
            )
        payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
        assert payload == []  # filtered out


class TestSearchHotelsTool:
    @respx.mock
    async def test_returns_hotels(self, serpapi_client: SerpAPIClient) -> None:
        respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_hotels_body())
        )
        server = build_server(serpapi_client)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(
                "search_hotels",
                {
                    "city": "Lisbon",
                    "check_in": "2026-05-10",
                    "check_out": "2026-05-14",
                },
            )
        payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
        assert len(payload) == 1
        assert payload[0]["name"] == "Alfama Stay"


class TestEstimateTotalCostTool:
    async def test_pure_computation(self, serpapi_client: SerpAPIClient) -> None:
        flight = FlightOption(
            airline="TAP",
            flight_number="TP202",
            origin="ATL",
            destination="LIS",
            depart_at=datetime(2026, 5, 10, 18, 0),
            arrive_at=datetime(2026, 5, 11, 7, 30),
            price_usd=Decimal("600"),
        )
        from datetime import date

        hotel = HotelOption(
            name="Alfama Stay",
            city="Lisbon",
            check_in=date(2026, 5, 10),
            check_out=date(2026, 5, 14),  # 4 nights @ 120 = 480
            price_per_night_usd=Decimal("120"),
        )
        server = build_server(serpapi_client)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(
                "estimate_total_cost",
                {
                    "flight": json.loads(flight.model_dump_json()),
                    "hotel": json.loads(hotel.model_dump_json()),
                    "extras_usd": 50,
                },
            )
        assert not result.isError
        payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
        # 600 + 480 + 50 = 1130
        assert Decimal(payload["total_usd"]) == Decimal("1130")