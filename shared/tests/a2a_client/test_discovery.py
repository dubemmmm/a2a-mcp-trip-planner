"""Tests for `shared.a2a_client.discovery`."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import pytest_asyncio
import respx

from shared.a2a_client.discovery import (
    AgentCardInvalidError,
    AgentDiscovery,
    AgentDiscoveryTimeout,
    AgentNotFoundError,
)

_CARD_PATH = "/.well-known/agent-card.json"


def _card_body(name: str = "travel") -> dict[str, Any]:
    """Minimal Agent Card JSON the resolver will accept."""
    return {
        "name": name,
        "description": "test agent",
        "version": "0.1.0",
        "supported_interfaces": [
            {
                "url": "http://localhost:8002",
                "protocol_binding": "JSONRPC",
                "protocol_version": "0.4.0",
            }
        ],
        "capabilities": {"streaming": True},
        "default_input_modes": ["text/plain"],
        "default_output_modes": ["text/plain"],
        "skills": [
            {
                "id": "search_flights",
                "name": "Search flights",
                "description": "find flights",
                "tags": ["flights"],
            }
        ],
    }


@pytest_asyncio.fixture
async def discovery() -> AsyncIterator[AgentDiscovery]:
    async with AgentDiscovery(cache_ttl_seconds=60.0) as d:
        yield d


# ---------- construction & validation ----------

class TestConstruction:
    async def test_invalid_ttl(self) -> None:
        with pytest.raises(ValueError):
            AgentDiscovery(cache_ttl_seconds=-1)

    async def test_invalid_timeout(self) -> None:
        with pytest.raises(ValueError):
            AgentDiscovery(timeout_seconds=0)

    async def test_empty_url_rejected(self, discovery: AgentDiscovery) -> None:
        with pytest.raises(ValueError):
            await discovery.discover("")

    async def test_external_client_not_closed(self) -> None:
        async with httpx.AsyncClient() as http:
            d = AgentDiscovery(http_client=http)
            await d.aclose()
            assert not http.is_closed


# ---------- happy path ----------

class TestDiscoverSuccess:
    @respx.mock
    async def test_returns_card(self, discovery: AgentDiscovery) -> None:
        respx.get(f"http://localhost:8002{_CARD_PATH}").mock(
            return_value=httpx.Response(200, json=_card_body("travel"))
        )
        card = await discovery.discover("http://localhost:8002")
        assert card.name == "travel"
        assert len(card.skills) == 1
        assert card.skills[0].id == "search_flights"

    @respx.mock
    async def test_trailing_slash_normalized(
        self, discovery: AgentDiscovery
    ) -> None:
        respx.get(f"http://localhost:8002{_CARD_PATH}").mock(
            return_value=httpx.Response(200, json=_card_body())
        )
        # Both should resolve to the same cache key.
        await discovery.discover("http://localhost:8002/")
        assert discovery.cached_urls() == ("http://localhost:8002",)


# ---------- caching ----------

class TestCaching:
    @respx.mock
    async def test_second_call_is_cached(
        self, discovery: AgentDiscovery
    ) -> None:
        route = respx.get(f"http://localhost:8002{_CARD_PATH}").mock(
            return_value=httpx.Response(200, json=_card_body())
        )
        await discovery.discover("http://localhost:8002")
        await discovery.discover("http://localhost:8002")
        assert route.call_count == 1

    @respx.mock
    async def test_force_refresh_bypasses_cache(
        self, discovery: AgentDiscovery
    ) -> None:
        route = respx.get(f"http://localhost:8002{_CARD_PATH}").mock(
            return_value=httpx.Response(200, json=_card_body())
        )
        await discovery.discover("http://localhost:8002")
        await discovery.discover("http://localhost:8002", force_refresh=True)
        assert route.call_count == 2

    @respx.mock
    async def test_invalidate_drops_entry(
        self, discovery: AgentDiscovery
    ) -> None:
        route = respx.get(f"http://localhost:8002{_CARD_PATH}").mock(
            return_value=httpx.Response(200, json=_card_body())
        )
        await discovery.discover("http://localhost:8002")
        discovery.invalidate("http://localhost:8002")
        await discovery.discover("http://localhost:8002")
        assert route.call_count == 2

    @respx.mock
    async def test_ttl_zero_means_no_cache(self) -> None:
        async with AgentDiscovery(cache_ttl_seconds=0) as d:
            route = respx.get(f"http://localhost:8002{_CARD_PATH}").mock(
                return_value=httpx.Response(200, json=_card_body())
            )
            await d.discover("http://localhost:8002")
            await d.discover("http://localhost:8002")
            assert route.call_count == 2

    @respx.mock
    async def test_concurrent_discoveries_collapse(
        self, discovery: AgentDiscovery
    ) -> None:
        """Two concurrent first-time lookups of the same URL must hit the
        network exactly once."""
        route = respx.get(f"http://localhost:8002{_CARD_PATH}").mock(
            return_value=httpx.Response(200, json=_card_body())
        )
        results = await asyncio.gather(
            discovery.discover("http://localhost:8002"),
            discovery.discover("http://localhost:8002"),
        )
        assert route.call_count == 1
        assert all(r.name == "travel" for r in results)


# ---------- error paths ----------

class TestNotFound:
    @respx.mock
    async def test_404(self, discovery: AgentDiscovery) -> None:
        respx.get(f"http://localhost:8002{_CARD_PATH}").mock(
            return_value=httpx.Response(404, text="not found")
        )
        with pytest.raises(AgentNotFoundError):
            await discovery.discover("http://localhost:8002")

    @respx.mock
    async def test_connection_refused(
        self, discovery: AgentDiscovery
    ) -> None:
        respx.get(f"http://localhost:8002{_CARD_PATH}").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        with pytest.raises(AgentNotFoundError):
            await discovery.discover("http://localhost:8002")

    @respx.mock
    async def test_failed_calls_not_cached(
        self, discovery: AgentDiscovery
    ) -> None:
        route = respx.get(f"http://localhost:8002{_CARD_PATH}").mock(
            return_value=httpx.Response(404)
        )
        with pytest.raises(AgentNotFoundError):
            await discovery.discover("http://localhost:8002")
        with pytest.raises(AgentNotFoundError):
            await discovery.discover("http://localhost:8002")
        # Both attempts hit the network — failure isn't cached.
        assert route.call_count == 2


class TestInvalidCard:
    @respx.mock
    async def test_invalid_json(self, discovery: AgentDiscovery) -> None:
        respx.get(f"http://localhost:8002{_CARD_PATH}").mock(
            return_value=httpx.Response(
                200, text="not json", headers={"content-type": "application/json"}
            )
        )
        with pytest.raises(AgentCardInvalidError):
            await discovery.discover("http://localhost:8002")

    @respx.mock
    async def test_schema_violation(self, discovery: AgentDiscovery) -> None:
        """JSON parses, but the body doesn't match the AgentCard schema."""
        respx.get(f"http://localhost:8002{_CARD_PATH}").mock(
            return_value=httpx.Response(200, json={"this": "is not a card"})
        )
        with pytest.raises(AgentCardInvalidError):
            await discovery.discover("http://localhost:8002")


class TestTimeout:
    async def test_timeout(self) -> None:
        """Force a timeout by making the timeout very short and the response slow."""
        async with AgentDiscovery(timeout_seconds=0.05) as d:
            with respx.mock:
                async def _slow(request: httpx.Request) -> httpx.Response:
                    await asyncio.sleep(0.5)
                    return httpx.Response(200, json=_card_body())

                respx.get(f"http://localhost:8002{_CARD_PATH}").mock(
                    side_effect=_slow
                )
                with pytest.raises(AgentDiscoveryTimeout):
                    await d.discover("http://localhost:8002")