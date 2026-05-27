"""Tests for shared.tools.web_search."""

from __future__ import annotations

import httpx
import pytest
import respx

from shared.clients.brave import _SEARCH_URL, BraveSearchClient
from shared.tools.web_search import SearchHit, web_search


class TestWebSearch:
    @respx.mock
    async def test_returns_provider_agnostic_hits(self) -> None:
        body = {
            "web": {
                "results": [
                    {
                        "title": "Lisbon visa info",
                        "url": "https://example.com/lisbon-visa",
                        "description": "How to get a Portugal visa.",
                        "age": "2 days ago",
                    }
                ]
            }
        }
        respx.get(_SEARCH_URL).mock(return_value=httpx.Response(200, json=body))
        async with BraveSearchClient("k") as client:
            results = await web_search(client, "lisbon visa")
        assert len(results) == 1
        assert isinstance(results[0], SearchHit)
        # SearchHit uses `snippet`, not `description` — verify the mapping.
        assert results[0].snippet == "How to get a Portugal visa."
        assert results[0].title == "Lisbon visa info"

    @respx.mock
    async def test_passes_count_through(self) -> None:
        route = respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json={"web": {"results": []}})
        )
        async with BraveSearchClient("k") as client:
            await web_search(client, "q", count=10)
        assert route.calls.last.request.url.params["count"] == "10"

    @respx.mock
    async def test_passes_country_through(self) -> None:
        route = respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json={"web": {"results": []}})
        )
        async with BraveSearchClient("k") as client:
            await web_search(client, "q", country="PT")
        assert route.calls.last.request.url.params["country"] == "PT"

    @respx.mock
    async def test_empty_results_returns_empty(self) -> None:
        respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json={"web": {"results": []}})
        )
        async with BraveSearchClient("k") as client:
            results = await web_search(client, "asdfqwerty")
        assert results == ()

    def test_search_hit_is_frozen(self) -> None:
        hit = SearchHit(title="t", url="u")
        with pytest.raises(Exception):  # pydantic ValidationError on mutation
            hit.title = "new"  # type: ignore[misc]