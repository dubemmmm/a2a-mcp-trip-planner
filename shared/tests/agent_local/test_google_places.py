"""Tests for the Google Places async client.

We mock at the HTTP layer (respx) so the tests exercise the real request
construction, the real response parsing, and our error paths — without ever
hitting Google.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from agents_local.clients.google_places import (
    _PLACES_SEARCH_URL,
    GooglePlacesClient,
    PlacesAPIError,
)


def _sample_response(n: int = 2) -> dict[str, Any]:
    return {
        "places": [
            {
                "id": f"place-{i}",
                "displayName": {"text": f"Place {i}", "languageCode": "en"},
                "formattedAddress": f"{i} Main St, Lisbon",
                "rating": 4.0 + i * 0.1,
                "priceLevel": "PRICE_LEVEL_MODERATE",
                "types": ["restaurant", "food"],
                "editorialSummary": {"text": f"Nice place {i}"},
            }
            for i in range(n)
        ]
    }


class TestClientConstruction:
    async def test_empty_api_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            GooglePlacesClient("")

    async def test_owns_client_by_default_and_closes_it(self) -> None:
        client = GooglePlacesClient("k")
        assert client._owns_client is True
        await client.aclose()
        # No assertion possible on httpx.AsyncClient post-close state other than
        # that aclose() itself was idempotent (a second call must not raise).
        await client.aclose()

    async def test_external_http_client_not_closed(self) -> None:
        async with httpx.AsyncClient() as http:
            client = GooglePlacesClient("k", http_client=http)
            assert client._owns_client is False
            await client.aclose()
            # The externally owned client must remain usable.
            assert not http.is_closed


class TestSearchTextSuccess:
    @respx.mock
    async def test_returns_parsed_results(self) -> None:
        respx.post(_PLACES_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_sample_response(2))
        )
        async with GooglePlacesClient("test-key") as client:
            results = await client.search_text("restaurants in Lisbon")
        assert len(results) == 2
        assert results[0].id == "place-0"
        assert results[0].display_name.text == "Place 0"
        assert results[0].price_level == "PRICE_LEVEL_MODERATE"
        assert results[1].rating == pytest.approx(4.1)

    @respx.mock
    async def test_sends_required_headers(self) -> None:
        route = respx.post(_PLACES_SEARCH_URL).mock(
            return_value=httpx.Response(200, json={"places": []})
        )
        async with GooglePlacesClient("secret-abc") as client:
            await client.search_text("anything")

        request = route.calls.last.request
        assert request.headers["X-Goog-Api-Key"] == "secret-abc"
        assert "X-Goog-FieldMask" in request.headers
        assert request.headers["Content-Type"] == "application/json"

    @respx.mock
    async def test_sends_query_and_page_size(self) -> None:
        route = respx.post(_PLACES_SEARCH_URL).mock(
            return_value=httpx.Response(200, json={"places": []})
        )
        async with GooglePlacesClient("k") as client:
            await client.search_text("museums in Porto", max_results=7)

        body = route.calls.last.request.read().decode()
        assert "museums in Porto" in body
        assert '"pageSize": 7' in body or '"pageSize":7' in body

    @respx.mock
    async def test_empty_places_returns_empty_tuple(self) -> None:
        respx.post(_PLACES_SEARCH_URL).mock(
            return_value=httpx.Response(200, json={"places": []})
        )
        async with GooglePlacesClient("k") as client:
            results = await client.search_text("nowhereville restaurants")
        assert results == ()

    @respx.mock
    async def test_missing_places_field_returns_empty_tuple(self) -> None:
        """`places` may simply be absent when there are no results."""
        respx.post(_PLACES_SEARCH_URL).mock(
            return_value=httpx.Response(200, json={})
        )
        async with GooglePlacesClient("k") as client:
            results = await client.search_text("xyz")
        assert results == ()

    @respx.mock
    async def test_missing_optional_fields(self) -> None:
        """Place results with only required fields must still parse."""
        body = {
            "places": [
                {
                    "id": "minimal",
                    "displayName": {"text": "Minimal Place"},
                }
            ]
        }
        respx.post(_PLACES_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=body)
        )
        async with GooglePlacesClient("k") as client:
            results = await client.search_text("anything")
        assert len(results) == 1
        assert results[0].rating is None
        assert results[0].price_level is None
        assert results[0].formatted_address is None


class TestSearchTextValidation:
    async def test_empty_query_rejected(self) -> None:
        async with GooglePlacesClient("k") as client:
            with pytest.raises(ValueError, match="non-empty"):
                await client.search_text("")

    async def test_whitespace_query_rejected(self) -> None:
        async with GooglePlacesClient("k") as client:
            with pytest.raises(ValueError, match="non-empty"):
                await client.search_text("   ")

    async def test_max_results_out_of_range(self) -> None:
        async with GooglePlacesClient("k") as client:
            with pytest.raises(ValueError, match="between 1 and 20"):
                await client.search_text("q", max_results=0)
            with pytest.raises(ValueError, match="between 1 and 20"):
                await client.search_text("q", max_results=21)


class TestSearchTextErrors:
    @respx.mock
    async def test_http_error_status(self) -> None:
        respx.post(_PLACES_SEARCH_URL).mock(
            return_value=httpx.Response(403, text="quota exceeded")
        )
        async with GooglePlacesClient("k") as client:
            with pytest.raises(PlacesAPIError) as exc:
                await client.search_text("q")
        assert exc.value.status_code == 403
        assert "403" in str(exc.value)

    @respx.mock
    async def test_network_error(self) -> None:
        respx.post(_PLACES_SEARCH_URL).mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        async with GooglePlacesClient("k") as client:
            with pytest.raises(PlacesAPIError, match="network error"):
                await client.search_text("q")

    @respx.mock
    async def test_non_json_body(self) -> None:
        respx.post(_PLACES_SEARCH_URL).mock(
            return_value=httpx.Response(
                200, text="<html>error</html>", headers={"content-type": "text/html"}
            )
        )
        async with GooglePlacesClient("k") as client:
            with pytest.raises(PlacesAPIError, match="non-JSON"):
                await client.search_text("q")

    @respx.mock
    async def test_unexpected_places_type(self) -> None:
        respx.post(_PLACES_SEARCH_URL).mock(
            return_value=httpx.Response(200, json={"places": "not-a-list"})
        )
        async with GooglePlacesClient("k") as client:
            with pytest.raises(PlacesAPIError, match="unexpected"):
                await client.search_text("q")

    @respx.mock
    async def test_malformed_place_entry(self) -> None:
        """A `places[]` entry missing the required `id` must error clearly."""
        respx.post(_PLACES_SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json={"places": [{"displayName": {"text": "no-id"}}]},
            )
        )
        async with GooglePlacesClient("k") as client:
            with pytest.raises(PlacesAPIError, match="failed to parse"):
                await client.search_text("q")