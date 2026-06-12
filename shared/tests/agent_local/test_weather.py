"""Tests for the weather tool."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import httpx
import pytest
import respx

from agents_local.clients.openweather import (
    _FORECAST_URL,
    _GEO_URL,
    OpenWeatherClient,
)
from agents_local.tools.weather import get_weather_forecast


def _geo_body(name: str = "Lisbon", lat: float = 38.7, lon: float = -9.1) -> list[dict[str, Any]]:
    return [{"name": name, "lat": lat, "lon": lon, "country": "PT"}]


def _interval(
    day: date,
    hour: int,
    *,
    temp: float = 18.0,
    temp_min: float = 15.0,
    temp_max: float = 22.0,
    description: str = "clear sky",
    pop: float = 0.0,
) -> dict[str, Any]:
    dt = datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc)
    return {
        "dt": int(dt.timestamp()),
        "main": {"temp": temp, "temp_min": temp_min, "temp_max": temp_max},
        "weather": [{"description": description}],
        "pop": pop,
    }


class TestGetWeatherForecast:
    @respx.mock
    async def test_aggregates_intervals_into_days(self) -> None:
        # 2 days of forecast, 4 intervals each.
        d1 = date(2026, 5, 11)
        d2 = date(2026, 5, 12)
        intervals = [
            _interval(d1, h, temp_max=20.0 + h / 24, temp_min=12.0)
            for h in (0, 6, 12, 18)
        ] + [
            _interval(d2, h, temp_max=25.0, temp_min=15.0, pop=0.3)
            for h in (0, 6, 12, 18)
        ]

        respx.get(_GEO_URL).mock(return_value=httpx.Response(200, json=_geo_body()))
        respx.get(_FORECAST_URL).mock(
            return_value=httpx.Response(200, json={"list": intervals})
        )

        async with OpenWeatherClient("k") as client:
            forecast = await get_weather_forecast(
                client, "Lisbon", date(2026, 5, 10), date(2026, 5, 14)
            )

        assert forecast.city == "Lisbon"
        assert forecast.coverage.requested_start_date == date(2026, 5, 10)
        assert forecast.coverage.requested_end_date_exclusive == date(2026, 5, 14)
        assert forecast.coverage.available_start_date == d1
        assert forecast.coverage.available_end_date_inclusive == d2
        assert forecast.coverage.status.value == "partial"
        assert len(forecast.days) == 2
        # Highs and lows aggregated correctly
        assert forecast.days[0].date == d1
        assert forecast.days[0].low_celsius == pytest.approx(12.0)
        assert forecast.days[1].chance_of_rain == pytest.approx(0.3)

    @respx.mock
    async def test_clips_outside_date_range(self) -> None:
        d_inside = date(2026, 5, 11)
        d_outside = date(2026, 5, 20)
        intervals = [
            _interval(d_inside, 12),
            _interval(d_outside, 12),
        ]
        respx.get(_GEO_URL).mock(return_value=httpx.Response(200, json=_geo_body()))
        respx.get(_FORECAST_URL).mock(
            return_value=httpx.Response(200, json={"list": intervals})
        )

        async with OpenWeatherClient("k") as client:
            forecast = await get_weather_forecast(
                client, "Lisbon", date(2026, 5, 10), date(2026, 5, 14)
            )

        # Only d_inside falls in the requested window.
        assert len(forecast.days) == 1
        assert forecast.days[0].date == d_inside
        assert forecast.coverage.status.value == "partial"
        assert forecast.coverage.available_start_date == d_inside
        assert forecast.coverage.available_end_date_inclusive == d_outside

    @respx.mock
    async def test_end_date_is_exclusive(self) -> None:
        d = date(2026, 5, 14)
        intervals = [_interval(d, 12)]
        respx.get(_GEO_URL).mock(return_value=httpx.Response(200, json=_geo_body()))
        respx.get(_FORECAST_URL).mock(
            return_value=httpx.Response(200, json={"list": intervals})
        )

        async with OpenWeatherClient("k") as client:
            forecast = await get_weather_forecast(
                client, "Lisbon", date(2026, 5, 10), date(2026, 5, 14)
            )
        assert forecast.days == ()
        assert forecast.coverage.status.value == "unavailable"
        assert forecast.coverage.available_start_date == d
        assert forecast.coverage.available_end_date_inclusive == d

    @respx.mock
    async def test_empty_forecast_returns_empty_days(self) -> None:
        respx.get(_GEO_URL).mock(return_value=httpx.Response(200, json=_geo_body()))
        respx.get(_FORECAST_URL).mock(
            return_value=httpx.Response(200, json={"list": []})
        )

        async with OpenWeatherClient("k") as client:
            forecast = await get_weather_forecast(
                client, "Lisbon", date(2026, 5, 10), date(2026, 5, 14)
            )
        assert forecast.days == ()
        assert forecast.city == "Lisbon"
        assert forecast.coverage.status.value == "unavailable"
        assert forecast.coverage.available_start_date is None
        assert forecast.coverage.available_end_date_inclusive is None

    @respx.mock
    async def test_full_coverage_is_reported_explicitly(self) -> None:
        d1 = date(2026, 5, 11)
        d2 = date(2026, 5, 12)
        intervals = [_interval(d1, 12), _interval(d2, 12)]
        respx.get(_GEO_URL).mock(return_value=httpx.Response(200, json=_geo_body()))
        respx.get(_FORECAST_URL).mock(
            return_value=httpx.Response(200, json={"list": intervals})
        )

        async with OpenWeatherClient("k") as client:
            forecast = await get_weather_forecast(
                client, "Lisbon", d1, date(2026, 5, 13)
            )

        assert forecast.coverage.status.value == "full"

    async def test_empty_city_rejected(self) -> None:
        async with OpenWeatherClient("k") as client:
            with pytest.raises(ValueError):
                await get_weather_forecast(
                    client, "", date(2026, 5, 10), date(2026, 5, 14)
                )

    async def test_bad_date_range_rejected(self) -> None:
        async with OpenWeatherClient("k") as client:
            with pytest.raises(ValueError, match="after"):
                await get_weather_forecast(
                    client, "Lisbon", date(2026, 5, 14), date(2026, 5, 10)
                )
