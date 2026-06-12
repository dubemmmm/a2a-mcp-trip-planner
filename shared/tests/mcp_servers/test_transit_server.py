"""End-to-end MCP tests for the Ground Transport server.

Runs a real `ClientSession` against our `Server` in the same process via
`create_connected_server_and_client_session`, exercising tool discovery, tool
invocation, the declared JSON schema, and the `to_json_text` encoding. The
Google Routes API is mocked at the HTTP layer with respx.
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

from agents_transit.clients.google_routes import _ROUTES_URL, GoogleRoutesClient
from mcp_servers.transit import build_server


# ---------- helpers ----------

def _route(
    *,
    distance_meters: int = 5000,
    duration: str = "1200s",
    description: str | None = "Main route",
) -> dict[str, Any]:
    body: dict[str, Any] = {"distanceMeters": distance_meters, "duration": duration}
    if description is not None:
        body["description"] = description
    return body


def _response(*routes: dict[str, Any]) -> dict[str, Any]:
    return {"routes": list(routes)}


@pytest_asyncio.fixture
async def client() -> AsyncIterator[GoogleRoutesClient]:
    """Real client object; respx intercepts HTTP in each test."""
    routes = GoogleRoutesClient("test-routes-key")
    try:
        yield routes
    finally:
        await routes.aclose()


# ---------- tool discovery ----------

class TestListTools:
    async def test_lists_all_three_tools(self, client: GoogleRoutesClient) -> None:
        server = build_server(client)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.list_tools()
        names = {tool.name for tool in result.tools}
        assert names == {"get_route", "compare_modes", "total_travel_minutes"}

    async def test_schemas_declare_required_fields(
        self, client: GoogleRoutesClient
    ) -> None:
        server = build_server(client)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.list_tools()
        by_name = {t.name: t for t in result.tools}
        assert set(by_name["get_route"].inputSchema["required"]) == {
            "origin", "destination"
        }
        assert by_name["total_travel_minutes"].inputSchema["required"] == ["routes"]


# ---------- call_tool: get_route ----------

class TestGetRouteTool:
    @respx.mock
    async def test_happy_path_returns_json_option(
        self, client: GoogleRoutesClient
    ) -> None:
        respx.post(_ROUTES_URL).mock(
            return_value=httpx.Response(
                200,
                json=_response(
                    _route(distance_meters=5000, duration="1200s", description="Metro")
                ),
            )
        )
        server = build_server(client)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(
                "get_route", {"origin": "Airport", "destination": "Hotel"}
            )
        assert not result.isError
        payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
        assert payload["origin"] == "Airport"
        assert payload["destination"] == "Hotel"
        assert payload["mode"] == "transit"
        assert payload["duration_minutes"] == 20

    @respx.mock
    async def test_mode_is_forwarded_to_api(self, client: GoogleRoutesClient) -> None:
        route = respx.post(_ROUTES_URL).mock(
            return_value=httpx.Response(200, json=_response(_route()))
        )
        server = build_server(client)
        async with create_connected_server_and_client_session(server) as session:
            await session.call_tool(
                "get_route",
                {"origin": "A", "destination": "B", "mode": "walk"},
            )
        body = json.loads(route.calls.last.request.read().decode())
        assert body["travelMode"] == "WALK"

    @respx.mock
    async def test_no_route_returns_null(self, client: GoogleRoutesClient) -> None:
        respx.post(_ROUTES_URL).mock(return_value=httpx.Response(200, json={}))
        server = build_server(client)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(
                "get_route", {"origin": "A", "destination": "B"}
            )
        assert result.content[0].text == "null"  # type: ignore[union-attr]


# ---------- call_tool: compare_modes ----------

class TestCompareModesTool:
    @respx.mock
    async def test_returns_modes_sorted_fastest_first(
        self, client: GoogleRoutesClient
    ) -> None:
        respx.post(_ROUTES_URL).mock(
            side_effect=[
                httpx.Response(200, json=_response(_route(duration="1800s"))),  # transit 30m
                httpx.Response(200, json=_response(_route(duration="600s"))),   # drive 10m
                httpx.Response(200, json=_response(_route(duration="3600s"))),  # walk 60m
            ]
        )
        server = build_server(client)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(
                "compare_modes", {"origin": "A", "destination": "B"}
            )
        payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
        assert [o["mode"] for o in payload] == ["drive", "transit", "walk"]

    @respx.mock
    async def test_respects_explicit_modes(self, client: GoogleRoutesClient) -> None:
        route = respx.post(_ROUTES_URL).mock(
            return_value=httpx.Response(200, json=_response(_route()))
        )
        server = build_server(client)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(
                "compare_modes",
                {"origin": "A", "destination": "B", "modes": ["bicycle"]},
            )
        payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
        assert route.call_count == 1
        assert len(payload) == 1
        assert payload[0]["mode"] == "bicycle"


# ---------- call_tool: total_travel_minutes ----------

class TestTotalTravelMinutesTool:
    async def test_sums_legs(self, client: GoogleRoutesClient) -> None:
        legs = [
            {
                "origin": "A",
                "destination": "B",
                "mode": "walk",
                "distance_meters": 400,
                "duration_minutes": 5,
                "summary": "walk: 0.4 km, ~5 min",
            },
            {
                "origin": "B",
                "destination": "C",
                "mode": "transit",
                "distance_meters": 6000,
                "duration_minutes": 15,
                "summary": "transit: 6.0 km, ~15 min",
            },
        ]
        server = build_server(client)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(
                "total_travel_minutes", {"routes": legs}
            )
        payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
        assert payload["total_minutes"] == 20


# ---------- call_tool: error paths ----------

class TestErrorPaths:
    async def test_unknown_tool_returns_error(self, client: GoogleRoutesClient) -> None:
        server = build_server(client)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool("nope", {})
        assert result.isError

    async def test_missing_required_arg_returns_error(
        self, client: GoogleRoutesClient
    ) -> None:
        server = build_server(client)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool("get_route", {"origin": "A"})
        assert result.isError
