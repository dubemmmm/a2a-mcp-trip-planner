"""Types local to the Local Experience Agent.

These do NOT cross agent boundaries (yet). Anything that needs to travel over
A2A goes in `shared.models`. Promoting a type from here to `shared` is a
deliberate design action — do it when another agent legitimately needs to
consume it.
"""

from __future__ import annotations

from datetime import date as date_type

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


class WeatherForecast(BaseModel):
    model_config = _FROZEN

    city: str = Field(..., min_length=1)
    days: tuple[DailyWeather, ...] = Field(default_factory=tuple)