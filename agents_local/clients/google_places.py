"""Async HTTP client for the Google Places API (New).

Single responsibility: speak HTTP to Google. No domain mapping happens here —
the tool layer turns `PlaceResult` into the shared `Attraction` model. This
split makes the client trivial to unit test (with respx) and keeps domain
knowledge out of network code.

Auth: the API key is sent in the `X-Goog-Api-Key` header. A field mask
restricts response bytes to what we actually need.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any, Self

import httpx
from pydantic import BaseModel, ConfigDict, Field

_PLACES_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
_DEFAULT_FIELD_MASK = ",".join(
    [
        "places.id",
        "places.displayName",
        "places.formattedAddress",
        "places.rating",
        "places.priceLevel",
        "places.types",
        "places.editorialSummary",
    ]
)
_DEFAULT_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


class PlacesAPIError(RuntimeError):
    """Raised when the Google Places API returns an error or unparseable response."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class _DisplayName(BaseModel):
    model_config = ConfigDict(extra="ignore")
    text: str
    languageCode: str | None = None


class _EditorialSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")
    text: str | None = None


class PlaceResult(BaseModel):
    """Lightly typed projection of a single place from the API response."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    display_name: _DisplayName = Field(alias="displayName")
    formatted_address: str | None = Field(default=None, alias="formattedAddress")
    rating: float | None = None
    price_level: str | None = Field(default=None, alias="priceLevel")
    types: tuple[str, ...] = Field(default_factory=tuple)
    editorial_summary: _EditorialSummary | None = Field(
        default=None, alias="editorialSummary"
    )


class GooglePlacesClient:
    """Async client for `places:searchText`."""

    def __init__(
        self,
        api_key: str | None,
        *,
        http_client: httpx.AsyncClient | None = None,
        field_mask: str = _DEFAULT_FIELD_MASK,
    ) -> None:
        if api_key is not None and not api_key:
            raise ValueError("api_key must be non-empty")
        self._api_key = api_key
        self._field_mask = field_mask
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
        """Close the underlying HTTP client (only if we own it)."""
        if self._owns_client:
            await self._http.aclose()

    async def search_text(
        self,
        query: str,
        *,
        max_results: int = 10,
    ) -> tuple[PlaceResult, ...]:
        """Run a `places:searchText` query and return parsed results.

        Args:
            query: Free-text query like "historical sites in Lisbon".
            max_results: 1-20, clamped server-side.

        Raises:
            PlacesAPIError: On HTTP error or malformed response.
        """
        if not query.strip():
            raise ValueError("query must be non-empty")
        if not 1 <= max_results <= 20:
            raise ValueError("max_results must be between 1 and 20")
        if not self._api_key:
            raise PlacesAPIError(
                "Google Places API key is not configured. "
                "Set GOOGLE_PLACES_API_KEY in your environment."
            )
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self._api_key,
            "X-Goog-FieldMask": self._field_mask,
        }
        payload: dict[str, Any] = {"textQuery": query, "pageSize": max_results}

        try:
            response = await self._http.post(
                _PLACES_SEARCH_URL, headers=headers, json=payload
            )
        except httpx.HTTPError as e:
            raise PlacesAPIError(f"network error talking to Google Places: {e}") from e

        if response.status_code >= 400:
            raise PlacesAPIError(
                f"Google Places returned {response.status_code}: {response.text[:200]}",
                status_code=response.status_code,
            )

        try:
            body = response.json()
        except ValueError as e:
            raise PlacesAPIError(f"non-JSON response from Google Places: {e}") from e

        raw_places = body.get("places", [])
        if not isinstance(raw_places, list):
            raise PlacesAPIError(
                f"unexpected 'places' field type: {type(raw_places).__name__}"
            )

        results: list[PlaceResult] = []
        for raw in raw_places:
            try:
                results.append(PlaceResult.model_validate(raw))
            except Exception as e:  # pydantic ValidationError
                raise PlacesAPIError(f"failed to parse place result: {e}") from e
        return tuple(results)