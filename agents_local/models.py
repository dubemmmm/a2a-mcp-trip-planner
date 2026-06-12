"""Types local to the Local Experience Agent.

These do NOT cross agent boundaries (yet). Anything that needs to travel over
A2A goes in `shared.models`. Promoting a type from here to `shared` is a
deliberate design action — do it when another agent legitimately needs to
consume it.
"""

from __future__ import annotations

from datetime import date as date_type
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

_FROZEN = ConfigDict(frozen=True)


class DailyWeather(BaseModel):
    """Aggregated single-day forecast (built from 3-hour OpenWeather intervals)."""

    model_config = _FROZEN

    date: date_type
    high_celsius: float
    low_celsius: float
    description: str = Field(..., min_length=1)
    chance_of_rain: float = Field(..., ge=0.0, le=1.0)


class ForecastCoverageStatus(str, Enum):
    """How much of the requested date range the provider returned."""

    FULL = "full"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class ForecastCoverage(BaseModel):
    """Explicit provider coverage metadata for a forecast request."""

    model_config = _FROZEN

    requested_start_date: date_type
    requested_end_date_exclusive: date_type
    available_start_date: date_type | None = None
    available_end_date_inclusive: date_type | None = None
    status: ForecastCoverageStatus


class WeatherForecast(BaseModel):
    model_config = _FROZEN

    city: str = Field(..., min_length=1)
    coverage: ForecastCoverage
    days: tuple[DailyWeather, ...] = Field(default_factory=tuple)
