"""Async HTTP client for the Brave Search API (Web Search endpoint).

Single responsibility: speak HTTP to Brave. Auth is via the
`X-Subscription-Token` header. The tool layer (`shared.tools.web_search`)
massages results into a clean shape for agents to consume.

Docs: https://api.search.brave.com/app/documentation/web-search/get-started
"""

from __future__ import annotations

from types import TracebackType
from typing import Self

import httpx
from pydantic import BaseModel, ConfigDict, Field

_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
_DEFAULT_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


class BraveAPIError(RuntimeError):
    """Raised on HTTP error or unparseable response from Brave Search."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class WebResult(BaseModel):
    """A single web result from the Brave Search response."""

    model_config = ConfigDict(extra="ignore")

    title: str
    url: str
    description: str = ""
    age: str | None = None


class _WebSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    results: tuple[WebResult, ...] = Field(default_factory=tuple)


class BraveSearchClient:
    """Async client for Brave's web search endpoint."""

    def __init__(
        self,
        api_key: str,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be a non-empty string")
        self._api_key = api_key
        self._owns_client = http_client is None
        self._http: httpx.AsyncClient = http_client or httpx.AsyncClient(
            timeout=_DEFAULT_TIMEOUT
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    async def search(
        self,
        query: str,
        *,
        count: int = 5,
        country: str | None = None,
        safesearch: str = "moderate",
    ) -> tuple[WebResult, ...]:
        """Run a web search.

        Args:
            query: Free-text search query.
            count: 1-20, number of results to request.
            country: Two-letter country code biasing results, e.g. "US".
            safesearch: "off" | "moderate" | "strict".

        Raises:
            ValueError: On invalid inputs.
            BraveAPIError: On network/HTTP/parse failure.
        """
        if not query.strip():
            raise ValueError("query must be non-empty")
        if not 1 <= count <= 20:
            raise ValueError("count must be between 1 and 20")
        if safesearch not in {"off", "moderate", "strict"}:
            raise ValueError("safesearch must be 'off', 'moderate', or 'strict'")

        params: dict[str, str | int] = {
            "q": query,
            "count": count,
            "safesearch": safesearch,
        }
        if country is not None:
            params["country"] = country

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self._api_key,
        }

        try:
            response = await self._http.get(
                _SEARCH_URL, params=params, headers=headers
            )
        except httpx.HTTPError as e:
            raise BraveAPIError(f"network error talking to Brave: {e}") from e

        if response.status_code >= 400:
            raise BraveAPIError(
                f"Brave returned {response.status_code}: {response.text[:200]}",
                status_code=response.status_code,
            )

        try:
            body = response.json()
        except ValueError as e:
            raise BraveAPIError(f"non-JSON response from Brave: {e}") from e

        web_section = body.get("web")
        if web_section is None:
            return ()
        if not isinstance(web_section, dict):
            raise BraveAPIError(
                f"unexpected 'web' field type: {type(web_section).__name__}"
            )

        try:
            parsed = _WebSection.model_validate(web_section)
        except Exception as e:
            raise BraveAPIError(f"failed to parse web section: {e}") from e
        return parsed.results