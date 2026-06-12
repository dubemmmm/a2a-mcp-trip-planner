"""Async HTTP client for the Google Routes API (`directions/v2:computeRoutes`).

Single responsibility: speak HTTP to Google. No domain mapping happens here —
the tool layer turns `RouteResult` into the local `RouteOption` model. This
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

_ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
_DEFAULT_FIELD_MASK = ",".join(
    [
        "routes.duration",
        "routes.distanceMeters",
        "routes.description",
    ]
)
_DEFAULT_TIMEOUT = httpx.Timeout(10.0, connect=5.0)

# Travel modes the Routes API accepts. The tool layer maps our `TravelMode`
# enum onto these; the client just guards against anything else.
_VALID_TRAVEL_MODES = frozenset({"DRIVE", "TRANSIT", "WALK", "BICYCLE"})


class RoutesAPIError(RuntimeError):
    """Raised when the Google Routes API returns an error or unparseable response."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RouteResult(BaseModel):
    """Lightly typed projection of a single route from the API response.

    `duration` arrives as a protobuf duration string like "1200s"; parsing it
    into minutes is a domain concern and lives in the tool layer.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    distance_meters: int = Field(default=0, alias="distanceMeters", ge=0)
    duration: str
    description: str | None = None


class GoogleRoutesClient:
    """Async client for `directions/v2:computeRoutes`."""

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

    async def compute_routes(
        self,
        origin: str,
        destination: str,
        *,
        travel_mode: str,
    ) -> tuple[RouteResult, ...]:
        """Compute routes between `origin` and `destination` for one travel mode.

        Args:
            origin: Free-text origin (e.g. "Lisbon Airport").
            destination: Free-text destination (e.g. "Belem Tower").
            travel_mode: One of DRIVE / TRANSIT / WALK / BICYCLE.

        Returns:
            The routes the API found, in its preferred order. Empty when no
            route exists (the API omits the `routes` field in that case).

        Raises:
            ValueError: On invalid inputs.
            RoutesAPIError: On missing key, HTTP error, or malformed response.
        """
        if not origin.strip():
            raise ValueError("origin must be non-empty")
        if not destination.strip():
            raise ValueError("destination must be non-empty")
        if travel_mode not in _VALID_TRAVEL_MODES:
            raise ValueError(
                f"travel_mode must be one of {sorted(_VALID_TRAVEL_MODES)}, "
                f"got {travel_mode!r}"
            )
        if not self._api_key:
            raise RoutesAPIError(
                "Google Routes API key is not configured. "
                "Set GOOGLE_ROUTES_API_KEY in your environment."
            )

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self._api_key,
            "X-Goog-FieldMask": self._field_mask,
        }
        payload: dict[str, Any] = {
            "origin": {"address": origin},
            "destination": {"address": destination},
            "travelMode": travel_mode,
        }
        # routingPreference is only valid for driving; sending it for other
        # modes is a 400 from the API.
        if travel_mode == "DRIVE":
            payload["routingPreference"] = "TRAFFIC_AWARE"

        try:
            response = await self._http.post(
                _ROUTES_URL, headers=headers, json=payload
            )
        except httpx.HTTPError as e:
            raise RoutesAPIError(f"network error talking to Google Routes: {e}") from e

        if response.status_code >= 400:
            raise RoutesAPIError(
                f"Google Routes returned {response.status_code}: {response.text[:200]}",
                status_code=response.status_code,
            )

        try:
            body = response.json()
        except ValueError as e:
            raise RoutesAPIError(f"non-JSON response from Google Routes: {e}") from e

        raw_routes = body.get("routes", [])
        if not isinstance(raw_routes, list):
            raise RoutesAPIError(
                f"unexpected 'routes' field type: {type(raw_routes).__name__}"
            )

        results: list[RouteResult] = []
        for raw in raw_routes:
            try:
                results.append(RouteResult.model_validate(raw))
            except Exception as e:  # pydantic ValidationError
                raise RoutesAPIError(f"failed to parse route result: {e}") from e
        return tuple(results)
