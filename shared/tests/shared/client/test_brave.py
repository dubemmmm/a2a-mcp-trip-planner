"""Tests for the Brave Search async client."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from shared.clients.brave import (
    _SEARCH_URL,
    BraveAPIError,
    BraveSearchClient,
)


def _body(n: int = 2) -> dict[str, Any]:
    return {
        "web": {
            "results": [
                {
                    "title": f"Result {i}",
                    "url": f"https://example.com/{i}",
                    "description": f"Snippet {i}",
                    "age": "1 day ago" if i % 2 == 0 else None,
                }
                for i in range(n)
            ]
        }
    }


class TestConstruction:
    async def test_empty_api_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            BraveSearchClient("")

    async def test_external_client_not_closed(self) -> None:
        async with httpx.AsyncClient() as http:
            client = BraveSearchClient("k", http_client=http)
            await client.aclose()
            assert not http.is_closed


class TestSearchSuccess:
    @respx.mock
    async def test_returns_parsed_hits(self) -> None:
        respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_body(3))
        )
        async with BraveSearchClient("k") as client:
            results = await client.search("lisbon weather may")
        assert len(results) == 3
        assert results[0].title == "Result 0"
        assert results[0].url == "https://example.com/0"
        assert results[1].age is None  # odd index

    @respx.mock
    async def test_sends_required_header_and_params(self) -> None:
        route = respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_body(0))
        )
        async with BraveSearchClient("brave-secret") as client:
            await client.search("portugal visa", count=7, country="US")

        request = route.calls.last.request
        assert request.headers["X-Subscription-Token"] == "brave-secret"
        assert request.headers["Accept"] == "application/json"
        params = request.url.params
        assert params["q"] == "portugal visa"
        assert params["count"] == "7"
        assert params["country"] == "US"
        assert params["safesearch"] == "moderate"

    @respx.mock
    async def test_missing_web_section_returns_empty(self) -> None:
        respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json={"news": {"results": []}})
        )
        async with BraveSearchClient("k") as client:
            results = await client.search("q")
        assert results == ()

    @respx.mock
    async def test_empty_results_returns_empty(self) -> None:
        respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json={"web": {"results": []}})
        )
        async with BraveSearchClient("k") as client:
            results = await client.search("q")
        assert results == ()

    @respx.mock
    async def test_missing_description_defaults_to_empty(self) -> None:
        body = {"web": {"results": [{"title": "T", "url": "https://x"}]}}
        respx.get(_SEARCH_URL).mock(return_value=httpx.Response(200, json=body))
        async with BraveSearchClient("k") as client:
            results = await client.search("q")
        assert results[0].description == ""


class TestSearchValidation:
    async def test_empty_query_rejected(self) -> None:
        async with BraveSearchClient("k") as client:
            with pytest.raises(ValueError, match="non-empty"):
                await client.search("   ")

    async def test_count_out_of_range(self) -> None:
        async with BraveSearchClient("k") as client:
            with pytest.raises(ValueError, match="between 1 and 20"):
                await client.search("q", count=0)
            with pytest.raises(ValueError, match="between 1 and 20"):
                await client.search("q", count=21)

    async def test_invalid_safesearch_rejected(self) -> None:
        async with BraveSearchClient("k") as client:
            with pytest.raises(ValueError, match="safesearch"):
                await client.search("q", safesearch="loud")  # type: ignore[arg-type]


class TestSearchErrors:
    @respx.mock
    async def test_http_error(self) -> None:
        respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(429, text="rate limited")
        )
        async with BraveSearchClient("k") as client:
            with pytest.raises(BraveAPIError) as exc:
                await client.search("q")
        assert exc.value.status_code == 429

    @respx.mock
    async def test_network_error(self) -> None:
        respx.get(_SEARCH_URL).mock(side_effect=httpx.ConnectError("nope"))
        async with BraveSearchClient("k") as client:
            with pytest.raises(BraveAPIError, match="network error"):
                await client.search("q")

    @respx.mock
    async def test_non_json_body(self) -> None:
        respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(200, text="<html/>", headers={"content-type": "text/html"})
        )
        async with BraveSearchClient("k") as client:
            with pytest.raises(BraveAPIError, match="non-JSON"):
                await client.search("q")

    @respx.mock
    async def test_unexpected_web_type(self) -> None:
        respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json={"web": "not-a-dict"})
        )
        async with BraveSearchClient("k") as client:
            with pytest.raises(BraveAPIError, match="unexpected"):
                await client.search("q")