"""Tests for the flights tool."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

import httpx
import pytest
import respx

from agents_travel.clients.serpapi import _SEARCH_URL, SerpAPIClient
from agents_travel.tools.flights import _parse_serpapi_time, search_flights
from shared.models import FlightOption


def _flight(price: float, *, stops: int = 0, airline: str = "TAP") -> dict[str, Any]:
    segments = []
    for i in range(stops + 1):
        segments.append(
            {
                "departure_airport": {
                    "id": "ATL" if i == 0 else f"X{i}",
                    "time": f"2026-05-10 {18 + i:02d}:00",
                },
                "arrival_airport": {
                    "id": f"X{i + 1}" if i < stops else "LIS",
                    "time": f"2026-05-11 {7 + i:02d}:30",
                },
                "airline": airline,
                "flight_number": f"TP {200 + i}",
            }
        )
    return {"flights": segments, "price": price}


def _flights_body(*prices: float) -> dict[str, Any]:
    return {
        "best_flights": [_flight(p) for p in prices],
        "other_flights": [],
    }


class TestParseSerpAPITime:
    def test_space_separator(self) -> None:
        assert _parse_serpapi_time("2026-05-10 18:00") == datetime(2026, 5, 10, 18, 0)

    def test_T_separator(self) -> None:
        assert _parse_serpapi_time("2026-05-10T18:00") == datetime(2026, 5, 10, 18, 0)

    def test_with_seconds(self) -> None:
        assert _parse_serpapi_time("2026-05-10 18:00:30") == datetime(
            2026, 5, 10, 18, 0, 30
        )


class TestSearchFlights:
    @respx.mock
    async def test_returns_flight_options(self) -> None:
        respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_flights_body(612.0, 800.0))
        )
        async with SerpAPIClient("k") as client:
            results = await search_flights(
                client,
                origin="ATL",
                destination="LIS",
                depart_date=date(2026, 5, 10),
                return_date=date(2026, 5, 14),
            )
        assert len(results) == 2
        assert all(isinstance(r, FlightOption) for r in results)

    @respx.mock
    async def test_sorted_by_price_ascending(self) -> None:
        respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_flights_body(900.0, 600.0, 750.0))
        )
        async with SerpAPIClient("k") as client:
            results = await search_flights(
                client,
                origin="ATL",
                destination="LIS",
                depart_date=date(2026, 5, 10),
            )
        prices = [r.price_usd for r in results]
        assert prices == sorted(prices)
        assert prices[0] == Decimal("600.0")

    @respx.mock
    async def test_max_price_filter(self) -> None:
        respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_flights_body(500.0, 700.0, 900.0))
        )
        async with SerpAPIClient("k") as client:
            results = await search_flights(
                client,
                origin="ATL",
                destination="LIS",
                depart_date=date(2026, 5, 10),
                max_price_usd=Decimal("750"),
            )
        assert {r.price_usd for r in results} == {Decimal("500.0"), Decimal("700.0")}

    @respx.mock
    async def test_maps_origin_destination_from_segments(self) -> None:
        # Use a 1-stop flight to verify we pull origin from first segment,
        # destination from the *last* segment.
        body = {
            "best_flights": [_flight(700.0, stops=1)],
            "other_flights": [],
        }
        respx.get(_SEARCH_URL).mock(return_value=httpx.Response(200, json=body))
        async with SerpAPIClient("k") as client:
            results = await search_flights(
                client, origin="ATL", destination="LIS", depart_date=date(2026, 5, 10)
            )
        assert results[0].origin == "ATL"
        assert results[0].destination == "LIS"
        assert results[0].stops == 1

    @respx.mock
    async def test_skips_malformed_entries(self) -> None:
        """A flight result without segments must be skipped, not crash the batch."""
        body = {
            "best_flights": [
                {"flights": [], "price": 100.0},  # malformed: no segments
                _flight(200.0),
            ],
            "other_flights": [],
        }
        respx.get(_SEARCH_URL).mock(return_value=httpx.Response(200, json=body))
        async with SerpAPIClient("k") as client:
            results = await search_flights(
                client, origin="ATL", destination="LIS", depart_date=date(2026, 5, 10)
            )
        assert len(results) == 1
        assert results[0].price_usd == Decimal("200.0")

    @respx.mock
    async def test_empty_results(self) -> None:
        respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(
                200, json={"best_flights": [], "other_flights": []}
            )
        )
        async with SerpAPIClient("k") as client:
            results = await search_flights(
                client, origin="ATL", destination="LIS", depart_date=date(2026, 5, 10)
            )
        assert results == ()