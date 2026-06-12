"""Tests for the Google Routes HTTP client (network layer only)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from agents_transit.clients.google_routes import (
    _ROUTES_URL,
    GoogleRoutesClient,
    RouteResult,
    RoutesAPIError,
)


# ---------- helpers ----------

def _route(
    *,
    distance_meters: int | None = 5000,
    duration: str | None = "1200s",
    description: str | None = "Main route",
) -> dict[str, Any]:
    body: dict[str, Any] = {}
    if distance_meters is not None:
        body["distanceMeters"] = distance_meters
    if duration is not None:
        body["duration"] = duration
    if description is not None:
        body["description"] = description
    return body


def _response(*routes: dict[str, Any]) -> dict[str, Any]:
    return {"routes": list(routes)}


# ---------- construction ----------

class TestConstruction:
    def test_empty_api_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            GoogleRoutesClient("")

    def test_none_api_key_allowed(self) -> None:
        # A server can boot without a key; the call fails later, not here.
        GoogleRoutesClient(None)


# ---------- compute_routes ----------

class TestComputeRoutes:
    @respx.mock
    async def test_parses_route_fields(self) -> None:
        respx.post(_ROUTES_URL).mock(
            return_value=httpx.Response(
                200, json=_response(_route(distance_meters=5000, duration="1200s"))
            )
        )
        async with GoogleRoutesClient("k") as client:
            results = await client.compute_routes("A", "B", travel_mode="TRANSIT")
        assert len(results) == 1
        assert isinstance(results[0], RouteResult)
        assert results[0].distance_meters == 5000
        assert results[0].duration == "1200s"

    @respx.mock
    async def test_no_routes_field_returns_empty(self) -> None:
        # The API omits "routes" entirely when nothing is found.
        respx.post(_ROUTES_URL).mock(return_value=httpx.Response(200, json={}))
        async with GoogleRoutesClient("k") as client:
            results = await client.compute_routes("A", "B", travel_mode="WALK")
        assert results == ()

    @respx.mock
    async def test_drive_mode_sets_routing_preference(self) -> None:
        route = respx.post(_ROUTES_URL).mock(
            return_value=httpx.Response(200, json=_response(_route()))
        )
        async with GoogleRoutesClient("k") as client:
            await client.compute_routes("A", "B", travel_mode="DRIVE")
        body = json.loads(route.calls.last.request.read().decode())
        assert body["travelMode"] == "DRIVE"
        assert body["routingPreference"] == "TRAFFIC_AWARE"

    @respx.mock
    async def test_non_drive_mode_omits_routing_preference(self) -> None:
        route = respx.post(_ROUTES_URL).mock(
            return_value=httpx.Response(200, json=_response(_route()))
        )
        async with GoogleRoutesClient("k") as client:
            await client.compute_routes("A", "B", travel_mode="TRANSIT")
        body = json.loads(route.calls.last.request.read().decode())
        assert "routingPreference" not in body

    @respx.mock
    async def test_request_carries_origin_and_destination(self) -> None:
        route = respx.post(_ROUTES_URL).mock(
            return_value=httpx.Response(200, json=_response(_route()))
        )
        async with GoogleRoutesClient("k") as client:
            await client.compute_routes("Lisbon Airport", "Belem Tower", travel_mode="WALK")
        body = json.loads(route.calls.last.request.read().decode())
        assert body["origin"]["address"] == "Lisbon Airport"
        assert body["destination"]["address"] == "Belem Tower"

    async def test_empty_origin_rejected(self) -> None:
        async with GoogleRoutesClient("k") as client:
            with pytest.raises(ValueError, match="origin"):
                await client.compute_routes("", "B", travel_mode="WALK")

    async def test_empty_destination_rejected(self) -> None:
        async with GoogleRoutesClient("k") as client:
            with pytest.raises(ValueError, match="destination"):
                await client.compute_routes("A", "", travel_mode="WALK")

    async def test_invalid_travel_mode_rejected(self) -> None:
        async with GoogleRoutesClient("k") as client:
            with pytest.raises(ValueError, match="travel_mode"):
                await client.compute_routes("A", "B", travel_mode="FLY")

    async def test_missing_api_key_raises(self) -> None:
        async with GoogleRoutesClient(None) as client:
            with pytest.raises(RoutesAPIError, match="not configured"):
                await client.compute_routes("A", "B", travel_mode="WALK")

    @respx.mock
    async def test_http_error_status_raises(self) -> None:
        respx.post(_ROUTES_URL).mock(
            return_value=httpx.Response(403, text="forbidden")
        )
        async with GoogleRoutesClient("k") as client:
            with pytest.raises(RoutesAPIError) as exc:
                await client.compute_routes("A", "B", travel_mode="WALK")
        assert exc.value.status_code == 403

    @respx.mock
    async def test_network_error_raises(self) -> None:
        respx.post(_ROUTES_URL).mock(side_effect=httpx.ConnectError("boom"))
        async with GoogleRoutesClient("k") as client:
            with pytest.raises(RoutesAPIError, match="network error"):
                await client.compute_routes("A", "B", travel_mode="WALK")

    @respx.mock
    async def test_non_json_response_raises(self) -> None:
        respx.post(_ROUTES_URL).mock(
            return_value=httpx.Response(200, text="not json")
        )
        async with GoogleRoutesClient("k") as client:
            with pytest.raises(RoutesAPIError, match="non-JSON"):
                await client.compute_routes("A", "B", travel_mode="WALK")

    @respx.mock
    async def test_unexpected_routes_type_raises(self) -> None:
        respx.post(_ROUTES_URL).mock(
            return_value=httpx.Response(200, json={"routes": "nope"})
        )
        async with GoogleRoutesClient("k") as client:
            with pytest.raises(RoutesAPIError, match="unexpected 'routes'"):
                await client.compute_routes("A", "B", travel_mode="WALK")
