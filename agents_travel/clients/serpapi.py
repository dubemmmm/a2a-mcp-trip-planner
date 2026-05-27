"""Async HTTP client for SerpApi (Google Flights + Google Hotels engines).

Single responsibility: HTTP only. The tool layer maps the lightly-typed
projections here into shared `FlightOption` / `HotelOption` models.

SerpApi's error model: HTTP 4xx for auth/quota issues, HTTP 200 with an
`error` field in the JSON for query-level problems. Both surface as
`SerpAPIError`.
"""

from __future__ import annotations

from datetime import date
from types import TracebackType
from typing import Any, Self

import httpx
from pydantic import BaseModel, ConfigDict, Field

_SEARCH_URL = "https://serpapi.com/search.json"
_FLIGHTS_ENGINE = "google_flights"
_HOTELS_ENGINE = "google_hotels"
_DEFAULT_TIMEOUT = httpx.Timeout(15.0, connect=5.0)


class SerpAPIError(RuntimeError):
    """Raised on HTTP/parse/query errors from SerpApi."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


# ----------------------- flight response projections -----------------------

class _Airport(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str  # IATA code
    name: str | None = None
    time: str  # "YYYY-MM-DD HH:MM" local


class FlightSegment(BaseModel):
    """One leg of a journey (a single flight between two airports)."""

    model_config = ConfigDict(extra="ignore")

    departure_airport: _Airport
    arrival_airport: _Airport
    airline: str
    flight_number: str
    duration: int | None = None  # minutes


class FlightResult(BaseModel):
    """A complete itinerary option from SerpApi (one or more segments)."""

    model_config = ConfigDict(extra="ignore")

    flights: tuple[FlightSegment, ...]
    price: float
    total_duration: int | None = None
    type: str | None = None
    booking_token: str | None = None


# ----------------------- hotel response projections ------------------------

class _RateInfo(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    extracted_lowest: float | None = None


class HotelProperty(BaseModel):
    """A single hotel property from SerpApi."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    name: str
    type: str | None = None
    rate_per_night: _RateInfo | None = None
    total_rate: _RateInfo | None = None
    overall_rating: float | None = None
    address: str | None = None
    link: str | None = None


# -------------------------------- client -----------------------------------

class SerpAPIClient:
    """Async client for `google_flights` and `google_hotels` engines."""

    def __init__(
        self,
        api_key: str | None,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if isinstance(api_key, str) and not api_key.strip():
            raise ValueError("api_key must be non-empty")
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

    async def _get(self, params: dict[str, str | int | float]) -> dict[str, Any]:
        """Internal: issue a SerpApi GET and return parsed JSON.

        Translates HTTP/JSON/error-field failures into SerpAPIError.
        """
        if not self._api_key:
            raise SerpAPIError("SERPAPI_API_KEY is not configured")
        params["api_key"] = self._api_key
        try:
            response = await self._http.get(_SEARCH_URL, params=params)
        except httpx.HTTPError as e:
            raise SerpAPIError(f"network error talking to SerpApi: {e}") from e

        if response.status_code >= 400:
            raise SerpAPIError(
                f"SerpApi returned {response.status_code}: {response.text[:200]}",
                status_code=response.status_code,
            )
        try:
            body = response.json()
        except ValueError as e:
            raise SerpAPIError(f"non-JSON response from SerpApi: {e}") from e

        if not isinstance(body, dict):
            raise SerpAPIError(
                f"unexpected SerpApi body type: {type(body).__name__}"
            )

        # Query-level error: HTTP 200 with an `error` field.
        if "error" in body:
            raise SerpAPIError(f"SerpApi query error: {body['error']}")

        return body

    async def search_flights(
        self,
        *,
        origin: str,
        destination: str,
        outbound_date: date,
        return_date: date | None = None,
        adults: int = 1,
        currency: str = "USD",
    ) -> tuple[FlightResult, ...]:
        """Search Google Flights for the given route and dates.

        If `return_date` is None, performs a one-way search; otherwise round trip.
        Returns `best_flights` + `other_flights` concatenated.

        Raises:
            ValueError: On invalid inputs.
            SerpAPIError: On any SerpApi failure.
        """
        if not origin.strip() or not destination.strip():
            raise ValueError("origin and destination must be non-empty")
        if adults < 1:
            raise ValueError("adults must be >= 1")
        if return_date is not None and return_date <= outbound_date:
            raise ValueError("return_date must be after outbound_date")

        params: dict[str, str | int | float] = {
            "engine": _FLIGHTS_ENGINE,
            "departure_id": origin.upper(),
            "arrival_id": destination.upper(),
            "outbound_date": outbound_date.isoformat(),
            "adults": adults,
            "currency": currency,
            "type": 1 if return_date else 2,  # 1 = round trip, 2 = one way
        }
        if return_date is not None:
            params["return_date"] = return_date.isoformat()

        body = await self._get(params)

        combined: list[FlightResult] = []
        for key in ("best_flights", "other_flights"):
            raw_list = body.get(key, [])
            if not isinstance(raw_list, list):
                raise SerpAPIError(
                    f"unexpected '{key}' field type: {type(raw_list).__name__}"
                )
            for raw in raw_list:
                try:
                    combined.append(FlightResult.model_validate(raw))
                except Exception as e:
                    raise SerpAPIError(f"failed to parse flight result: {e}") from e
        return tuple(combined)

    async def search_hotels(
        self,
        *,
        query: str,
        check_in: date,
        check_out: date,
        adults: int = 1,
        currency: str = "USD",
    ) -> tuple[HotelProperty, ...]:
        """Search Google Hotels for `query` within the date range.

        Raises:
            ValueError: On invalid inputs.
            SerpAPIError: On any SerpApi failure.
        """
        if not query.strip():
            raise ValueError("query must be non-empty")
        if check_out <= check_in:
            raise ValueError("check_out must be after check_in")
        if adults < 1:
            raise ValueError("adults must be >= 1")

        params: dict[str, str | int | float] = {
            "engine": _HOTELS_ENGINE,
            "q": query,
            "check_in_date": check_in.isoformat(),
            "check_out_date": check_out.isoformat(),
            "adults": adults,
            "currency": currency,
        }
        body = await self._get(params)

        raw_props = body.get("properties", [])
        if not isinstance(raw_props, list):
            raise SerpAPIError(
                f"unexpected 'properties' field type: {type(raw_props).__name__}"
            )
        results: list[HotelProperty] = []
        for raw in raw_props:
            try:
                results.append(HotelProperty.model_validate(raw))
            except Exception as e:
                raise SerpAPIError(f"failed to parse hotel result: {e}") from e
        return tuple(results)