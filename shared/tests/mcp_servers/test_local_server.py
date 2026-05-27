"""End-to-end MCP tests for the Local Experience server.

We use `create_connected_server_and_client_session` to run a real
`ClientSession` against our `Server` in the same process. This exercises:
  * tool discovery (list_tools)
  * tool invocation (call_tool)
  * JSON schema declared in list_tools
  * our `to_json_text` encoding
  * the real Pydantic round-trip on the way out

External APIs are mocked at the HTTP layer with respx, exactly as in the
unit tests for the underlying tools.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import pytest_asyncio
import respx
from mcp.shared.memory import create_connected_server_and_client_session

from agents_local.clients.google_places import _PLACES_SEARCH_URL, GooglePlacesClient
from agents_local.clients.openweather import (
    _FORECAST_URL,
    _GEO_URL,
    OpenWeatherClient,
)
from mcp_servers.local import build_server


# ---------- helpers ----------

def _place(name: str) -> dict[str, Any]:
    return {
        "id": f"id-{name}",
        "displayName": {"text": name},
        "formattedAddress": "1 Main St",
        "rating": 4.5,
        "priceLevel": "PRICE_LEVEL_MODERATE",
        "editorialSummary": {"text": f"{name} is nice."},
    }


def _places_body(*names: str) -> dict[str, Any]:
    return {"places": [_place(n) for n in names]}


@pytest_asyncio.fixture
async def clients() -> AsyncIterator[tuple[GooglePlacesClient, OpenWeatherClient]]:
    """Real client objects; respx intercepts HTTP in each test."""
    places = GooglePlacesClient("test-google-key")
    weather = OpenWeatherClient("test-weather-key")
    try:
        yield places, weather
    finally:
        await places.aclose()
        await weather.aclose()


# ---------- tool discovery ----------

class TestListTools:
    async def test_lists_all_three_tools(
        self,
        clients: tuple[GooglePlacesClient, OpenWeatherClient],
    ) -> None:
        server = build_server(*clients)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.list_tools()
        names = {tool.name for tool in result.tools}
        assert names == {
            "find_attractions",
            "recommend_restaurants",
            "get_weather_forecast",
        }

    async def test_schemas_declare_required_fields(
        self,
        clients: tuple[GooglePlacesClient, OpenWeatherClient],
    ) -> None:
        server = build_server(*clients)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.list_tools()
        by_name = {t.name: t for t in result.tools}
        assert "city" in by_name["find_attractions"].inputSchema["required"]
        assert "interests" in by_name["find_attractions"].inputSchema["required"]
        assert set(by_name["get_weather_forecast"].inputSchema["required"]) == {
            "city", "start_date", "end_date"
        }


# ---------- call_tool: find_attractions ----------

class TestFindAttractionsTool:
    @respx.mock
    async def test_happy_path_returns_json_attractions(
        self,
        clients: tuple[GooglePlacesClient, OpenWeatherClient],
    ) -> None:
        respx.post(_PLACES_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_places_body("Cafe A", "Cafe B"))
        )
        server = build_server(*clients)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(
                "find_attractions",
                {"city": "Lisbon", "interests": ["food"]},
            )
        assert not result.isError
        assert len(result.content) == 1
        payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
        assert isinstance(payload, list)
        assert len(payload) == 2
        assert payload[0]["name"] == "Cafe A"
        assert payload[0]["category"] == "food"

    @respx.mock
    async def test_multiple_interests_dedupe_across_categories(
        self,
        clients: tuple[GooglePlacesClient, OpenWeatherClient],
    ) -> None:
        # Same name across both interest queries → returned only once.
        respx.post(_PLACES_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_places_body("Same Spot"))
        )
        server = build_server(*clients)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(
                "find_attractions",
                {"city": "Lisbon", "interests": ["food", "history"]},
            )
        payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
        assert len(payload) == 1


# ---------- call_tool: recommend_restaurants ----------

class TestRecommendRestaurantsTool:
    @respx.mock
    async def test_default_returns_all(
        self,
        clients: tuple[GooglePlacesClient, OpenWeatherClient],
    ) -> None:
        respx.post(_PLACES_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_places_body("R1", "R2"))
        )
        server = build_server(*clients)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(
                "recommend_restaurants", {"city": "Lisbon"}
            )
        payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
        assert len(payload) == 2

    @respx.mock
    async def test_price_level_filter_applied(
        self,
        clients: tuple[GooglePlacesClient, OpenWeatherClient],
    ) -> None:
        body = {
            "places": [
                {
                    "id": "1",
                    "displayName": {"text": "Cheap Eats"},
                    "priceLevel": "PRICE_LEVEL_INEXPENSIVE",
                },
                {
                    "id": "2",
                    "displayName": {"text": "Mid Bistro"},
                    "priceLevel": "PRICE_LEVEL_MODERATE",
                },
            ]
        }
        respx.post(_PLACES_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=body)
        )
        server = build_server(*clients)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(
                "recommend_restaurants",
                {"city": "Lisbon", "price_level": "MODERATE"},
            )
        payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
        assert len(payload) == 1
        assert payload[0]["name"] == "Mid Bistro"


# ---------- call_tool: get_weather_forecast ----------

class TestWeatherTool:
    @respx.mock
    async def test_returns_forecast_with_days(
        self,
        clients: tuple[GooglePlacesClient, OpenWeatherClient],
    ) -> None:
        from datetime import datetime, timezone

        def _interval(day: int) -> dict[str, Any]:
            ts = int(
                datetime(2026, 5, day, 12, tzinfo=timezone.utc).timestamp()
            )
            return {
                "dt": ts,
                "main": {"temp": 20.0, "temp_min": 15.0, "temp_max": 25.0},
                "weather": [{"description": "sunny"}],
                "pop": 0.2,
            }

        respx.get(_GEO_URL).mock(
            return_value=httpx.Response(
                200, json=[{"name": "Lisbon", "lat": 38.7, "lon": -9.1}]
            )
        )
        respx.get(_FORECAST_URL).mock(
            return_value=httpx.Response(
                200, json={"list": [_interval(11), _interval(12)]}
            )
        )
        server = build_server(*clients)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(
                "get_weather_forecast",
                {
                    "city": "Lisbon",
                    "start_date": "2026-05-10",
                    "end_date": "2026-05-14",
                },
            )
        payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
        assert payload["city"] == "Lisbon"
        assert len(payload["days"]) == 2


# ---------- call_tool: error paths ----------

class TestErrorPaths:
    async def test_unknown_tool_returns_error(
        self,
        clients: tuple[GooglePlacesClient, OpenWeatherClient],
    ) -> None:
        server = build_server(*clients)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool("nope", {})
        assert result.isError

    async def test_missing_required_arg_returns_error(
        self,
        clients: tuple[GooglePlacesClient, OpenWeatherClient],
    ) -> None:
        server = build_server(*clients)
        async with create_connected_server_and_client_session(server) as session:
            # `interests` is required by both our handler and the schema.
            result = await session.call_tool(
                "find_attractions", {"city": "Lisbon"}
            )
        assert result.isError