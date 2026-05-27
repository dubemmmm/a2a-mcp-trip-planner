"""Agent discovery — fetch and cache `AgentCard`s by URL.

Wraps a2a-sdk's `A2ACardResolver` so the caller doesn't need to manage an
`httpx.AsyncClient`, and adds:
  * a per-URL TTL cache so back-to-back calls don't re-fetch
  * a typed error taxonomy that distinguishes "no such agent" from "found
    but malformed" from "network issue"

The cache is intentionally per-process and in-memory. Layer 6's orchestrator
will discover Travel and Local Experience once at startup and reuse the
cached cards across requests; we don't need persistence yet.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import httpx
import structlog
from a2a.client import A2ACardResolver, AgentCardResolutionError
from a2a.types import AgentCard

_LOGGER = structlog.get_logger(__name__)

# Default TTL: 15 minutes. Cards almost never change; we re-fetch
# occasionally to catch agent restarts and skill updates.
DEFAULT_CACHE_TTL_SECONDS = 15 * 60
DEFAULT_TIMEOUT_SECONDS = 10.0


class AgentDiscoveryError(RuntimeError):
    """Base class for discovery failures."""


class AgentNotFoundError(AgentDiscoveryError):
    """The agent card endpoint returned 404 or the URL is unreachable."""


class AgentCardInvalidError(AgentDiscoveryError):
    """The agent card endpoint responded, but the payload is unparseable."""


class AgentDiscoveryTimeout(AgentDiscoveryError):
    """The discovery request did not complete within the timeout."""


@dataclass(frozen=True)
class _CachedCard:
    card: AgentCard
    fetched_at: float


class AgentDiscovery:
    """Per-process cache of discovered Agent Cards.

    Instantiate once and reuse across the lifetime of the orchestrator.
    Safe for concurrent use; the cache is guarded by an asyncio lock so two
    concurrent discoveries of the same URL only hit the network once.
    """

    def __init__(
        self,
        *,
        cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be >= 0")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        self._cache_ttl = cache_ttl_seconds
        self._timeout = timeout_seconds
        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds, connect=5.0)
        )
        self._cache: dict[str, _CachedCard] = {}
        self._inflight: dict[str, asyncio.Future[AgentCard]] = {}
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    async def __aenter__(self) -> "AgentDiscovery":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def discover(self, url: str, *, force_refresh: bool = False) -> AgentCard:
        """Fetch the Agent Card at `url`, with caching.

        `url` should be the agent's base URL (e.g. ``http://localhost:8001``),
        not the full card path. We append ``/.well-known/agent-card.json``
        ourselves through the resolver.

        Args:
            url: Agent base URL.
            force_refresh: Skip the cache and re-fetch unconditionally.

        Raises:
            AgentNotFoundError: 404, connection refused, DNS failure.
            AgentCardInvalidError: Non-JSON or schema-invalid card body.
            AgentDiscoveryTimeout: Request exceeded the configured timeout.
            ValueError: For empty `url`.
        """
        if not url.strip():
            raise ValueError("url must be non-empty")
        normalized = url.rstrip("/")

        log = _LOGGER.bind(url=normalized)

        async with self._lock:
            if not force_refresh:
                cached = self._cache.get(normalized)
                if cached is not None:
                    age = time.monotonic() - cached.fetched_at
                    if age < self._cache_ttl:
                        log.debug("agent_card_cache_hit", age_seconds=age)
                        return cached.card
                    log.debug("agent_card_cache_expired", age_seconds=age)

                # If another coroutine is already fetching this URL, wait for
                # its result instead of making a second network call.
                if normalized in self._inflight:
                    fut = self._inflight[normalized]
                    is_fetcher = False
                else:
                    fut = asyncio.get_running_loop().create_future()
                    # Silences "Future exception was never retrieved" when the
                    # fetch fails and no other coroutine is awaiting this fut.
                    fut.add_done_callback(
                        lambda f: f.exception() if not f.cancelled() else None
                    )
                    self._inflight[normalized] = fut
                    is_fetcher = True
            else:
                # force_refresh bypasses both the cache and any in-flight
                # coalescing — the caller explicitly wants a fresh fetch.
                fut = asyncio.get_running_loop().create_future()
                is_fetcher = True

        if not is_fetcher:
            return await fut

        try:
            card = await self._fetch(normalized, log)
        except Exception as exc:
            async with self._lock:
                self._inflight.pop(normalized, None)
            if not fut.done():
                fut.set_exception(exc)
            raise

        async with self._lock:
            self._cache[normalized] = _CachedCard(card=card, fetched_at=time.monotonic())
            self._inflight.pop(normalized, None)
        fut.set_result(card)
        return card

    async def _fetch(self, normalized: str, log: structlog.stdlib.BoundLogger) -> AgentCard:
        """Make the network call and return a validated AgentCard."""
        resolver = A2ACardResolver(httpx_client=self._http, base_url=normalized)
        try:
            card = await asyncio.wait_for(
                resolver.get_agent_card(), timeout=self._timeout
            )
        except asyncio.TimeoutError as e:
            log.warning("agent_card_discovery_timeout")
            raise AgentDiscoveryTimeout(
                f"timed out discovering agent at {normalized}"
            ) from e
        except AgentCardResolutionError as e:
            # The resolver wraps a few different underlying problems behind
            # this exception. We inspect the message to pick the right
            # subclass; if we can't tell, default to NotFound (the more
            # common case).
            message = str(e)
            cause = e.__cause__
            is_invalid_payload = (
                "JSONDecodeError" in repr(cause)
                or "ValidationError" in repr(cause)
                or "invalid" in message.lower()
                or "malformed" in message.lower()
            )
            if is_invalid_payload:
                log.warning("agent_card_invalid", reason=message)
                raise AgentCardInvalidError(
                    f"agent card at {normalized} is malformed: {message}"
                ) from e
            log.warning("agent_card_not_found", reason=message)
            raise AgentNotFoundError(
                f"could not reach agent at {normalized}: {message}"
            ) from e
        except httpx.HTTPError as e:
            log.warning("agent_card_network_error", error=str(e))
            raise AgentNotFoundError(
                f"network error reaching {normalized}: {e}"
            ) from e

        # Defensive validation. Protobuf's JSON parsing is permissive — it
        # accepts any object and silently fills missing fields with defaults.
        # An empty-name card almost certainly means the endpoint responded
        # but with the wrong shape (misconfiguration, wrong port, etc.).
        if not card.name:
            log.warning("agent_card_missing_name")
            raise AgentCardInvalidError(
                f"agent card at {normalized} has no name; "
                f"likely wrong endpoint or misconfigured server"
            )

        log.info("agent_card_discovered", name=card.name)
        return card

    def invalidate(self, url: str) -> None:
        """Drop the cached entry for `url`. Safe to call if the URL isn't cached."""
        self._cache.pop(url.rstrip("/"), None)

    def cached_urls(self) -> tuple[str, ...]:
        """Return the set of URLs currently in the cache (for tests/observability)."""
        return tuple(self._cache.keys())