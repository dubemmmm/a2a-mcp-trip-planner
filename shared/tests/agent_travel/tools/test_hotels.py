"""Tests for the hotels tool."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

import httpx
import pytest
import respx

from agents_travel.clients.serpapi import _SEARCH_URL, SerpAPIClient
from agents_travel.tools.hotels import search_hotels
from shared.models import HotelOption


def _hotel(
    name: str,
    *,
    rate: float | None = 120.0,
    rating: float | None = 4.5,
) -> dict[str, Any]:
    body: dict[str, Any] = {"name": name, "type": "hotel"}
    if rate is not None:
        body["rate_per_night"] = {"extracted_lowest": rate}
    if rating is not None:
        body["overall_rating"] = rating
    body["address"] = f"address of {name}"
    body["link"] = f"https://example.com/{name}"
    return body


def _hotels_body(*hotels: dict[str, Any]) -> dict[str, Any]:
    return {"properties": list(hotels)}


class TestSearchHotels:
    @respx.mock
    async def test_returns_hotel_options(self) -> None:
        body = _hotels_body(_hotel("Alfama Stay", rate=120), _hotel("Baixa Inn", rate=80))
        respx.get(_SEARCH_URL).mock(return_value=httpx.Response(200, json=body))
        async with SerpAPIClient("k") as client:
            results = await search_hotels(
                client,
                city="Lisbon",
                check_in=date(2026, 5, 10),
                check_out=date(2026, 5, 14),
            )
        assert len(results) == 2
        assert all(isinstance(r, HotelOption) for r in results)
        # City and dates stamped from inputs.
        assert all(r.city == "Lisbon" for r in results)
        assert all(r.check_in == date(2026, 5, 10) for r in results)

    @respx.mock
    async def test_sorted_by_price(self) -> None:
        body = _hotels_body(
            _hotel("Expensive", rate=300),
            _hotel("Cheap", rate=80),
            _hotel("Mid", rate=150),
        )
        respx.get(_SEARCH_URL).mock(return_value=httpx.Response(200, json=body))
        async with SerpAPIClient("k") as client:
            results = await search_hotels(
                client,
                city="Lisbon",
                check_in=date(2026, 5, 10),
                check_out=date(2026, 5, 14),
            )
        prices = [r.price_per_night_usd for r in results]
        assert prices == sorted(prices)
        assert results[0].name == "Cheap"

    @respx.mock
    async def test_max_price_per_night_filter(self) -> None:
        body = _hotels_body(
            _hotel("A", rate=80),
            _hotel("B", rate=150),
            _hotel("C", rate=300),
        )
        respx.get(_SEARCH_URL).mock(return_value=httpx.Response(200, json=body))
        async with SerpAPIClient("k") as client:
            results = await search_hotels(
                client,
                city="Lisbon",
                check_in=date(2026, 5, 10),
                check_out=date(2026, 5, 14),
                max_price_per_night_usd=Decimal("200"),
            )
        assert {r.name for r in results} == {"A", "B"}

    @respx.mock
    async def test_skips_properties_without_rate(self) -> None:
        body = _hotels_body(
            _hotel("Has Price", rate=120),
            _hotel("No Price", rate=None),
        )
        respx.get(_SEARCH_URL).mock(return_value=httpx.Response(200, json=body))
        async with SerpAPIClient("k") as client:
            results = await search_hotels(
                client,
                city="Lisbon",
                check_in=date(2026, 5, 10),
                check_out=date(2026, 5, 14),
            )
        assert len(results) == 1
        assert results[0].name == "Has Price"

    @respx.mock
    async def test_carries_address_and_link(self) -> None:
        body = _hotels_body(_hotel("H1", rate=100))
        respx.get(_SEARCH_URL).mock(return_value=httpx.Response(200, json=body))
        async with SerpAPIClient("k") as client:
            results = await search_hotels(
                client,
                city="Lisbon",
                check_in=date(2026, 5, 10),
                check_out=date(2026, 5, 14),
            )
        assert results[0].address == "address of H1"
        assert results[0].booking_url == "https://example.com/H1"

    @respx.mock
    async def test_empty_returns_empty(self) -> None:
        respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json={"properties": []})
        )
        async with SerpAPIClient("k") as client:
            results = await search_hotels(
                client,
                city="Lisbon",
                check_in=date(2026, 5, 10),
                check_out=date(2026, 5, 14),
            )
        assert results == ()