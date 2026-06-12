"""Tests for the routing tool layer (Routes client → RouteOption model)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from agents_transit.clients.google_routes import _ROUTES_URL, GoogleRoutesClient
from agents_transit.models import RouteOption, TravelMode
from agents_transit.tools.routing import (
    _parse_duration_seconds,
    compare_modes,
    get_route,
    total_travel_minutes,
)


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


def _option(mode: TravelMode, duration_minutes: int) -> RouteOption:
    return RouteOption(
        origin="A",
        destination="B",
        mode=mode,
        distance_meters=1000,
        duration_minutes=duration_minutes,
        summary="x",
    )


# ---------- pure helpers ----------

class TestParseDurationSeconds:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1200s", 1200),
            ("600s", 600),
            ("0s", 0),
            ("90.4s", 90),  # fractional seconds round
        ],
    )
    def test_valid(self, raw: str, expected: int) -> None:
        assert _parse_duration_seconds(raw) == expected

    @pytest.mark.parametrize("raw", ["1200", "abcs", "", "s12"])
    def test_malformed_raises(self, raw: str) -> None:
        with pytest.raises(ValueError, match="unexpected duration"):
            _parse_duration_seconds(raw)


class TestTotalTravelMinutes:
    def test_sums_legs(self) -> None:
        legs = [_option(TravelMode.WALK, 5), _option(TravelMode.TRANSIT, 15)]
        assert total_travel_minutes(legs) == 20

    def test_empty_is_zero(self) -> None:
        assert total_travel_minutes([]) == 0


# ---------- get_route ----------

class TestGetRoute:
    @respx.mock
    async def test_maps_result_to_option(self) -> None:
        respx.post(_ROUTES_URL).mock(
            return_value=httpx.Response(
                200,
                json=_response(
                    _route(distance_meters=5000, duration="1200s", description="Metro Blue")
                ),
            )
        )
        async with GoogleRoutesClient("k") as client:
            opt = await get_route(client, "Airport", "Hotel", mode=TravelMode.TRANSIT)
        assert opt is not None
        assert isinstance(opt, RouteOption)
        assert opt.origin == "Airport"
        assert opt.destination == "Hotel"
        assert opt.mode is TravelMode.TRANSIT
        assert opt.distance_meters == 5000
        assert opt.duration_minutes == 20  # 1200s
        assert "Metro Blue" in opt.summary

    @respx.mock
    async def test_no_route_returns_none(self) -> None:
        respx.post(_ROUTES_URL).mock(return_value=httpx.Response(200, json={}))
        async with GoogleRoutesClient("k") as client:
            opt = await get_route(client, "A", "B", mode=TravelMode.TRANSIT)
        assert opt is None

    @respx.mock
    async def test_mode_is_sent_to_api(self) -> None:
        route = respx.post(_ROUTES_URL).mock(
            return_value=httpx.Response(200, json=_response(_route()))
        )
        async with GoogleRoutesClient("k") as client:
            await get_route(client, "A", "B", mode=TravelMode.WALK)
        body = json.loads(route.calls.last.request.read().decode())
        assert body["travelMode"] == "WALK"

    @respx.mock
    async def test_sub_minute_route_reports_one_minute(self) -> None:
        respx.post(_ROUTES_URL).mock(
            return_value=httpx.Response(
                200, json=_response(_route(distance_meters=50, duration="20s"))
            )
        )
        async with GoogleRoutesClient("k") as client:
            opt = await get_route(client, "A", "B", mode=TravelMode.WALK)
        assert opt is not None
        assert opt.duration_minutes == 1

    @respx.mock
    async def test_missing_description_still_summarizes(self) -> None:
        respx.post(_ROUTES_URL).mock(
            return_value=httpx.Response(
                200, json=_response(_route(description=None))
            )
        )
        async with GoogleRoutesClient("k") as client:
            opt = await get_route(client, "A", "B", mode=TravelMode.DRIVE)
        assert opt is not None
        assert "drive" in opt.summary

    async def test_empty_origin_rejected(self) -> None:
        async with GoogleRoutesClient("k") as client:
            with pytest.raises(ValueError, match="origin"):
                await get_route(client, "", "B")

    async def test_empty_destination_rejected(self) -> None:
        async with GoogleRoutesClient("k") as client:
            with pytest.raises(ValueError, match="destination"):
                await get_route(client, "A", "")


# ---------- compare_modes ----------

class TestCompareModes:
    @respx.mock
    async def test_returns_modes_sorted_fastest_first(self) -> None:
        # Default order is (TRANSIT, DRIVE, WALK); responses returned in that
        # call order. DRIVE is fastest, so it should sort first.
        respx.post(_ROUTES_URL).mock(
            side_effect=[
                httpx.Response(200, json=_response(_route(duration="1800s"))),  # transit 30m
                httpx.Response(200, json=_response(_route(duration="600s"))),   # drive 10m
                httpx.Response(200, json=_response(_route(duration="3600s"))),  # walk 60m
            ]
        )
        async with GoogleRoutesClient("k") as client:
            opts = await compare_modes(client, "A", "B")
        assert [o.mode for o in opts] == [
            TravelMode.DRIVE,
            TravelMode.TRANSIT,
            TravelMode.WALK,
        ]

    @respx.mock
    async def test_drops_modes_without_a_route(self) -> None:
        respx.post(_ROUTES_URL).mock(
            side_effect=[
                httpx.Response(200, json=_response(_route(duration="1800s"))),  # transit
                httpx.Response(200, json=_response(_route(duration="600s"))),   # drive
                httpx.Response(200, json={}),                                   # walk: none
            ]
        )
        async with GoogleRoutesClient("k") as client:
            opts = await compare_modes(client, "A", "B")
        assert len(opts) == 2
        assert all(o.mode is not TravelMode.WALK for o in opts)

    @respx.mock
    async def test_respects_explicit_modes(self) -> None:
        route = respx.post(_ROUTES_URL).mock(
            return_value=httpx.Response(200, json=_response(_route()))
        )
        async with GoogleRoutesClient("k") as client:
            opts = await compare_modes(
                client, "A", "B", modes=[TravelMode.BICYCLE]
            )
        assert route.call_count == 1
        assert len(opts) == 1
        assert opts[0].mode is TravelMode.BICYCLE

    async def test_empty_modes_rejected(self) -> None:
        async with GoogleRoutesClient("k") as client:
            with pytest.raises(ValueError, match="at least one mode"):
                await compare_modes(client, "A", "B", modes=[])
