"""Tests for the SerpApi async client."""

from __future__ import annotations

from datetime import date
from typing import Any

import httpx
import pytest
import respx

from agents_travel.clients.serpapi import (
    _SEARCH_URL,
    SerpAPIClient,
    SerpAPIError,
)


def _flight_response(*, n_best: int = 1, n_other: int = 1) -> dict[str, Any]:
    def _one(price: float, stops: int = 0) -> dict[str, Any]:
        segments = []
        for i in range(stops + 1):
            segments.append(
                {
                    "departure_airport": {
                        "id": "ATL" if i == 0 else f"X{i}",
                        "name": "x",
                        "time": f"2026-05-10 {18 + i:02d}:00",
                    },
                    "arrival_airport": {
                        "id": f"X{i + 1}" if i < stops else "LIS",
                        "name": "y",
                        "time": f"2026-05-11 {7 + i:02d}:30",
                    },
                    "airline": "TAP",
                    "flight_number": f"TP {200 + i}",
                    "duration": 510,
                }
            )
        return {
            "flights": segments,
            "price": price,
            "total_duration": 510,
            "type": "Round trip",
        }

    return {
        "best_flights": [_one(612.0 + i) for i in range(n_best)],
        "other_flights": [_one(800.0 + i, stops=1) for i in range(n_other)],
    }


def _hotel_response(*names: str) -> dict[str, Any]:
    return {
        "properties": [
            {
                "name": n,
                "type": "hotel",
                "rate_per_night": {"extracted_lowest": 120.0 + i * 10},
                "total_rate": {"extracted_lowest": 480.0 + i * 40},
                "overall_rating": 4.5,
                "address": f"{i} Main St, Lisbon",
                "link": f"https://google.com/hotels/{i}",
            }
            for i, n in enumerate(names)
        ]
    }


class TestConstruction:
    async def test_empty_api_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            SerpAPIClient("")


class TestSearchFlights:
    @respx.mock
    async def test_returns_combined_results(self) -> None:
        respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_flight_response(n_best=2, n_other=1))
        )
        async with SerpAPIClient("k") as client:
            results = await client.search_flights(
                origin="ATL",
                destination="LIS",
                outbound_date=date(2026, 5, 10),
                return_date=date(2026, 5, 14),
            )
        assert len(results) == 3
        # First two are non-stop best flights, third has 1 stop
        assert len(results[0].flights) == 1
        assert len(results[2].flights) == 2

    @respx.mock
    async def test_one_way_sends_type_2_and_no_return_date(self) -> None:
        route = respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json={"best_flights": [], "other_flights": []})
        )
        async with SerpAPIClient("k") as client:
            await client.search_flights(
                origin="atl",  # lowercase to verify uppercasing
                destination="lis",
                outbound_date=date(2026, 5, 10),
            )
        params = route.calls.last.request.url.params
        assert params["type"] == "2"
        assert params["departure_id"] == "ATL"
        assert params["arrival_id"] == "LIS"
        assert "return_date" not in params

    @respx.mock
    async def test_round_trip_sends_type_1_and_return_date(self) -> None:
        route = respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json={"best_flights": [], "other_flights": []})
        )
        async with SerpAPIClient("k") as client:
            await client.search_flights(
                origin="ATL",
                destination="LIS",
                outbound_date=date(2026, 5, 10),
                return_date=date(2026, 5, 14),
            )
        params = route.calls.last.request.url.params
        assert params["type"] == "1"
        assert params["return_date"] == "2026-05-14"
        assert params["outbound_date"] == "2026-05-10"

    @respx.mock
    async def test_api_key_in_params(self) -> None:
        route = respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json={"best_flights": [], "other_flights": []})
        )
        async with SerpAPIClient("serp-abc") as client:
            await client.search_flights(
                origin="ATL", destination="LIS", outbound_date=date(2026, 5, 10)
            )
        assert route.calls.last.request.url.params["api_key"] == "serp-abc"

    async def test_empty_origin_rejected(self) -> None:
        async with SerpAPIClient("k") as client:
            with pytest.raises(ValueError):
                await client.search_flights(
                    origin="", destination="LIS", outbound_date=date(2026, 5, 10)
                )

    async def test_return_before_outbound_rejected(self) -> None:
        async with SerpAPIClient("k") as client:
            with pytest.raises(ValueError, match="after"):
                await client.search_flights(
                    origin="ATL",
                    destination="LIS",
                    outbound_date=date(2026, 5, 14),
                    return_date=date(2026, 5, 10),
                )

    async def test_zero_adults_rejected(self) -> None:
        async with SerpAPIClient("k") as client:
            with pytest.raises(ValueError):
                await client.search_flights(
                    origin="ATL",
                    destination="LIS",
                    outbound_date=date(2026, 5, 10),
                    adults=0,
                )

    @respx.mock
    async def test_query_error_in_body(self) -> None:
        respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json={"error": "Invalid airport code: ZZZ"})
        )
        async with SerpAPIClient("k") as client:
            with pytest.raises(SerpAPIError, match="Invalid airport code"):
                await client.search_flights(
                    origin="ZZZ",
                    destination="LIS",
                    outbound_date=date(2026, 5, 10),
                )

    @respx.mock
    async def test_http_error(self) -> None:
        respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(401, text="bad key")
        )
        async with SerpAPIClient("k") as client:
            with pytest.raises(SerpAPIError) as exc:
                await client.search_flights(
                    origin="ATL", destination="LIS", outbound_date=date(2026, 5, 10)
                )
        assert exc.value.status_code == 401

    @respx.mock
    async def test_network_error(self) -> None:
        respx.get(_SEARCH_URL).mock(side_effect=httpx.ConnectError("nope"))
        async with SerpAPIClient("k") as client:
            with pytest.raises(SerpAPIError, match="network error"):
                await client.search_flights(
                    origin="ATL", destination="LIS", outbound_date=date(2026, 5, 10)
                )

    @respx.mock
    async def test_unexpected_best_flights_type(self) -> None:
        respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(
                200, json={"best_flights": "nope", "other_flights": []}
            )
        )
        async with SerpAPIClient("k") as client:
            with pytest.raises(SerpAPIError, match="unexpected"):
                await client.search_flights(
                    origin="ATL", destination="LIS", outbound_date=date(2026, 5, 10)
                )


class TestSearchHotels:
    @respx.mock
    async def test_returns_parsed_properties(self) -> None:
        respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_hotel_response("H1", "H2"))
        )
        async with SerpAPIClient("k") as client:
            results = await client.search_hotels(
                query="Lisbon", check_in=date(2026, 5, 10), check_out=date(2026, 5, 14)
            )
        assert len(results) == 2
        assert results[0].name == "H1"
        assert results[0].rate_per_night is not None
        assert results[0].rate_per_night.extracted_lowest == pytest.approx(120.0)

    @respx.mock
    async def test_sends_required_params(self) -> None:
        route = respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json={"properties": []})
        )
        async with SerpAPIClient("k") as client:
            await client.search_hotels(
                query="Lisbon",
                check_in=date(2026, 5, 10),
                check_out=date(2026, 5, 14),
                adults=2,
            )
        params = route.calls.last.request.url.params
        assert params["engine"] == "google_hotels"
        assert params["q"] == "Lisbon"
        assert params["check_in_date"] == "2026-05-10"
        assert params["check_out_date"] == "2026-05-14"
        assert params["adults"] == "2"
        assert params["currency"] == "USD"

    async def test_empty_query_rejected(self) -> None:
        async with SerpAPIClient("k") as client:
            with pytest.raises(ValueError):
                await client.search_hotels(
                    query="", check_in=date(2026, 5, 10), check_out=date(2026, 5, 14)
                )

    async def test_checkout_before_checkin_rejected(self) -> None:
        async with SerpAPIClient("k") as client:
            with pytest.raises(ValueError, match="after"):
                await client.search_hotels(
                    query="Lisbon",
                    check_in=date(2026, 5, 14),
                    check_out=date(2026, 5, 10),
                )

    @respx.mock
    async def test_empty_properties(self) -> None:
        respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json={"properties": []})
        )
        async with SerpAPIClient("k") as client:
            results = await client.search_hotels(
                query="Nowhere", check_in=date(2026, 5, 10), check_out=date(2026, 5, 14)
            )
        assert results == ()

    @respx.mock
    async def test_missing_properties_key_returns_empty(self) -> None:
        respx.get(_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        async with SerpAPIClient("k") as client:
            results = await client.search_hotels(
                query="Lisbon", check_in=date(2026, 5, 10), check_out=date(2026, 5, 14)
            )
        assert results == ()