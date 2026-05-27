"""Tests for the attractions tool layer (Places client → Attraction model)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from agents_local.clients.google_places import _PLACES_SEARCH_URL, GooglePlacesClient
from agents_local.tools.attractions import (
    _build_query,
    _map_price_level,
    find_attractions,
    recommend_restaurants,
)
from shared.models import Attraction, Interest, PriceLevel


# ---------- helpers ----------

def _place(
    name: str,
    *,
    rating: float | None = 4.5,
    price_level: str | None = "PRICE_LEVEL_MODERATE",
    address: str | None = "1 Main St",
    summary: str | None = "A nice place.",
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": f"id-{name.lower().replace(' ', '-')}",
        "displayName": {"text": name},
    }
    if address is not None:
        body["formattedAddress"] = address
    if rating is not None:
        body["rating"] = rating
    if price_level is not None:
        body["priceLevel"] = price_level
    if summary is not None:
        body["editorialSummary"] = {"text": summary}
    return body


def _response(*names: str) -> dict[str, Any]:
    return {"places": [_place(n) for n in names]}


# ---------- pure helpers ----------

class TestBuildQuery:
    def test_food_query_includes_city_and_category(self) -> None:
        assert "Lisbon" in _build_query("Lisbon", Interest.FOOD)
        assert "restaurants" in _build_query("Lisbon", Interest.FOOD).lower()

    def test_history_query(self) -> None:
        q = _build_query("Porto", Interest.HISTORY)
        assert "Porto" in q
        assert "historical" in q.lower()


class TestMapPriceLevel:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("PRICE_LEVEL_FREE", PriceLevel.BUDGET),
            ("PRICE_LEVEL_INEXPENSIVE", PriceLevel.BUDGET),
            ("PRICE_LEVEL_MODERATE", PriceLevel.MODERATE),
            ("PRICE_LEVEL_EXPENSIVE", PriceLevel.UPSCALE),
            ("PRICE_LEVEL_VERY_EXPENSIVE", PriceLevel.LUXURY),
            (None, None),
            ("PRICE_LEVEL_UNKNOWN", None),  # unknown variants → None, no crash
        ],
    )
    def test_mapping(self, raw: str | None, expected: PriceLevel | None) -> None:
        assert _map_price_level(raw) == expected


# ---------- find_attractions ----------

class TestFindAttractions:
    @respx.mock
    async def test_one_interest_returns_attractions(self) -> None:
        respx.post(_PLACES_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_response("Cafe A", "Cafe B"))
        )
        async with GooglePlacesClient("k") as client:
            results = await find_attractions(client, "Lisbon", [Interest.FOOD])
        assert len(results) == 2
        assert all(isinstance(r, Attraction) for r in results)
        assert all(r.category is Interest.FOOD for r in results)
        assert results[0].name == "Cafe A"

    @respx.mock
    async def test_multiple_interests_each_get_called(self) -> None:
        route = respx.post(_PLACES_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_response("Place X"))
        )
        async with GooglePlacesClient("k") as client:
            await find_attractions(
                client, "Lisbon", [Interest.FOOD, Interest.HISTORY, Interest.ART]
            )
        assert route.call_count == 3

    @respx.mock
    async def test_dedupes_by_name_across_interests(self) -> None:
        # Both interests return the same place; tool should keep only one.
        respx.post(_PLACES_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_response("Same Place"))
        )
        async with GooglePlacesClient("k") as client:
            results = await find_attractions(
                client, "Lisbon", [Interest.FOOD, Interest.HISTORY]
            )
        assert len(results) == 1

    @respx.mock
    async def test_uses_editorial_summary_when_present(self) -> None:
        body = {"places": [_place("Spot", summary="Wonderful 17th-century cafe.")]}
        respx.post(_PLACES_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=body)
        )
        async with GooglePlacesClient("k") as client:
            results = await find_attractions(client, "Lisbon", [Interest.FOOD])
        assert results[0].description == "Wonderful 17th-century cafe."

    @respx.mock
    async def test_fallback_description_when_no_summary(self) -> None:
        body = {"places": [_place("Spot", summary=None)]}
        respx.post(_PLACES_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=body)
        )
        async with GooglePlacesClient("k") as client:
            results = await find_attractions(client, "Lisbon", [Interest.HISTORY])
        assert results[0].description  # non-empty fallback
        assert "history" in results[0].description.lower()

    async def test_empty_city_rejected(self) -> None:
        async with GooglePlacesClient("k") as client:
            with pytest.raises(ValueError, match="non-empty"):
                await find_attractions(client, "", [Interest.FOOD])

    async def test_no_interests_rejected(self) -> None:
        async with GooglePlacesClient("k") as client:
            with pytest.raises(ValueError, match="interest"):
                await find_attractions(client, "Lisbon", [])

    @respx.mock
    async def test_empty_results_returns_empty(self) -> None:
        respx.post(_PLACES_SEARCH_URL).mock(
            return_value=httpx.Response(200, json={"places": []})
        )
        async with GooglePlacesClient("k") as client:
            results = await find_attractions(client, "Nowhere", [Interest.FOOD])
        assert results == ()


# ---------- recommend_restaurants ----------

class TestRecommendRestaurants:
    @respx.mock
    async def test_basic_request_returns_food_attractions(self) -> None:
        respx.post(_PLACES_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_response("R1", "R2"))
        )
        async with GooglePlacesClient("k") as client:
            results = await recommend_restaurants(client, "Lisbon")
        assert len(results) == 2
        assert all(r.category is Interest.FOOD for r in results)

    @respx.mock
    async def test_cuisine_folded_into_query(self) -> None:
        route = respx.post(_PLACES_SEARCH_URL).mock(
            return_value=httpx.Response(200, json={"places": []})
        )
        async with GooglePlacesClient("k") as client:
            await recommend_restaurants(client, "Lisbon", cuisine="Portuguese")
        body = route.calls.last.request.read().decode()
        assert "Portuguese" in body
        assert "Lisbon" in body

    @respx.mock
    async def test_price_level_filter(self) -> None:
        body = {
            "places": [
                _place("Cheap Eats", price_level="PRICE_LEVEL_INEXPENSIVE"),
                _place("Mid Bistro", price_level="PRICE_LEVEL_MODERATE"),
                _place("Fancy Place", price_level="PRICE_LEVEL_EXPENSIVE"),
            ]
        }
        respx.post(_PLACES_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=body)
        )
        async with GooglePlacesClient("k") as client:
            results = await recommend_restaurants(
                client, "Lisbon", price_level=PriceLevel.MODERATE
            )
        assert len(results) == 1
        assert results[0].name == "Mid Bistro"

    @respx.mock
    async def test_no_price_filter_returns_all(self) -> None:
        body = {
            "places": [
                _place("A", price_level="PRICE_LEVEL_INEXPENSIVE"),
                _place("B", price_level="PRICE_LEVEL_EXPENSIVE"),
            ]
        }
        respx.post(_PLACES_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=body)
        )
        async with GooglePlacesClient("k") as client:
            results = await recommend_restaurants(client, "Lisbon")
        assert len(results) == 2

    async def test_empty_city_rejected(self) -> None:
        async with GooglePlacesClient("k") as client:
            with pytest.raises(ValueError):
                await recommend_restaurants(client, "")