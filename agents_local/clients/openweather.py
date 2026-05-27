"""Async HTTP client for OpenWeatherMap.

Uses two endpoints from the free tier:
  * Geocoding (`/geo/1.0/direct`) — city name → lat/lon
  * 5-day / 3-hour forecast (`/data/2.5/forecast`) — coarse hourly forecast

The free tier does not offer a true "daily" endpoint, so the tool layer
aggregates 3-hour intervals into per-day summaries. This client just speaks
HTTP; aggregation lives in `tools.weather`.
"""

from __future__ import annotations

from types import TracebackType
from typing import Self

import httpx
from pydantic import BaseModel, ConfigDict, Field

_GEO_URL = "https://api.openweathermap.org/geo/1.0/direct"
_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
_DEFAULT_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


class WeatherAPIError(RuntimeError):
    """Raised on HTTP or parse errors from OpenWeatherMap."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class GeocodeResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    lat: float
    lon: float
    country: str | None = None


class _ForecastMain(BaseModel):
    model_config = ConfigDict(extra="ignore")
    temp: float
    temp_min: float
    temp_max: float


class _ForecastWeather(BaseModel):
    model_config = ConfigDict(extra="ignore")
    description: str


class ForecastInterval(BaseModel):
    """A single 3-hour forecast slice."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    dt: int  # unix timestamp UTC
    main: _ForecastMain
    weather: tuple[_ForecastWeather, ...]
    pop: float = Field(default=0.0, ge=0.0, le=1.0)  # probability of precipitation


class OpenWeatherClient:
    def __init__(
        self,
        api_key: str | None,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if api_key is not None and not api_key:
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

    async def geocode(self, city: str) -> GeocodeResult:
        """Resolve a city name to its first geocoding hit.

        Raises:
            WeatherAPIError: On network/HTTP error, or when no results.
        """
        if not city.strip():
            raise ValueError("city must be non-empty")
        
        if not self._api_key:
            ValueError("api_key must be a non-empty string")

        params: dict[str, str | int | float] = {
            "q": city,
            "limit": 1,
            "appid": self._api_key,
        }
        try:
            response = await self._http.get(_GEO_URL, params=params)
        except httpx.HTTPError as e:
            raise WeatherAPIError(f"network error during geocode: {e}") from e

        if response.status_code >= 400:
            raise WeatherAPIError(
                f"geocode returned {response.status_code}: {response.text[:200]}",
                status_code=response.status_code,
            )

        try:
            body = response.json()
        except ValueError as e:
            raise WeatherAPIError(f"non-JSON geocode response: {e}") from e

        if not isinstance(body, list) or not body:
            raise WeatherAPIError(f"no geocoding results for city: {city!r}")

        try:
            return GeocodeResult.model_validate(body[0])
        except Exception as e:
            raise WeatherAPIError(f"failed to parse geocode result: {e}") from e

    async def forecast(self, lat: float, lon: float) -> tuple[ForecastInterval, ...]:
        """Return 3-hour-interval forecast (~5 days)."""
        params: dict[str, str | int | float] = {
            "lat": lat,
            "lon": lon,
            "appid": self._api_key,
            "units": "metric",
        }
        if not self._api_key:
            ValueError("api_key must be a non-empty string")
        try:
            response = await self._http.get(_FORECAST_URL, params=params)
        except httpx.HTTPError as e:
            raise WeatherAPIError(f"network error during forecast: {e}") from e

        if response.status_code >= 400:
            raise WeatherAPIError(
                f"forecast returned {response.status_code}: {response.text[:200]}",
                status_code=response.status_code,
            )

        try:
            body = response.json()
        except ValueError as e:
            raise WeatherAPIError(f"non-JSON forecast response: {e}") from e

        raw_list = body.get("list", [])
        if not isinstance(raw_list, list):
            raise WeatherAPIError(
                f"unexpected 'list' field type: {type(raw_list).__name__}"
            )

        intervals: list[ForecastInterval] = []
        for raw in raw_list:
            try:
                intervals.append(ForecastInterval.model_validate(raw))
            except Exception as e:
                raise WeatherAPIError(f"failed to parse forecast interval: {e}") from e
        return tuple(intervals)