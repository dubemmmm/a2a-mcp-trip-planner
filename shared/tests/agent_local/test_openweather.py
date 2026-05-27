"""Tests for the OpenWeatherMap async client."""

from __future__ import annotations

import httpx
import pytest
import respx

from agents_local.clients.openweather import (
    _FORECAST_URL,
    _GEO_URL,
    OpenWeatherClient,
    WeatherAPIError,
)


def _geocode_body() -> list[dict[str, object]]:
    return [{"name": "Lisbon", "lat": 38.7223, "lon": -9.1393, "country": "PT"}]


def _forecast_body(n_intervals: int = 8) -> dict[str, object]:
    return {
        "list": [
            {
                "dt": 1_700_000_000 + i * 10_800,  # 3-hour steps
                "main": {
                    "temp": 18.0 + i,
                    "temp_min": 15.0 + i,
                    "temp_max": 22.0 + i,
                },
                "weather": [{"description": "scattered clouds"}],
                "pop": 0.1 * (i % 5),
            }
            for i in range(n_intervals)
        ]
    }


class TestClientConstruction:
    async def test_empty_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            OpenWeatherClient("")


class TestGeocode:
    @respx.mock
    async def test_returns_first_result(self) -> None:
        respx.get(_GEO_URL).mock(
            return_value=httpx.Response(200, json=_geocode_body())
        )
        async with OpenWeatherClient("k") as client:
            result = await client.geocode("Lisbon")
        assert result.name == "Lisbon"
        assert result.lat == pytest.approx(38.7223)
        assert result.country == "PT"

    @respx.mock
    async def test_passes_required_params(self) -> None:
        route = respx.get(_GEO_URL).mock(
            return_value=httpx.Response(200, json=_geocode_body())
        )
        async with OpenWeatherClient("ow-key-123") as client:
            await client.geocode("Lisbon")
        params = route.calls.last.request.url.params
        assert params["q"] == "Lisbon"
        assert params["appid"] == "ow-key-123"
        assert params["limit"] == "1"

    async def test_empty_city_rejected(self) -> None:
        async with OpenWeatherClient("k") as client:
            with pytest.raises(ValueError):
                await client.geocode("")

    @respx.mock
    async def test_no_results_raises(self) -> None:
        respx.get(_GEO_URL).mock(return_value=httpx.Response(200, json=[]))
        async with OpenWeatherClient("k") as client:
            with pytest.raises(WeatherAPIError, match="no geocoding results"):
                await client.geocode("Zzzzz")

    @respx.mock
    async def test_http_error_status(self) -> None:
        respx.get(_GEO_URL).mock(
            return_value=httpx.Response(401, text="invalid api key")
        )
        async with OpenWeatherClient("k") as client:
            with pytest.raises(WeatherAPIError) as exc:
                await client.geocode("Lisbon")
        assert exc.value.status_code == 401


class TestForecast:
    @respx.mock
    async def test_returns_intervals(self) -> None:
        respx.get(_FORECAST_URL).mock(
            return_value=httpx.Response(200, json=_forecast_body(4))
        )
        async with OpenWeatherClient("k") as client:
            intervals = await client.forecast(38.7, -9.1)
        assert len(intervals) == 4
        assert intervals[0].main.temp == pytest.approx(18.0)
        assert intervals[0].weather[0].description == "scattered clouds"

    @respx.mock
    async def test_passes_metric_units_and_coords(self) -> None:
        route = respx.get(_FORECAST_URL).mock(
            return_value=httpx.Response(200, json={"list": []})
        )
        async with OpenWeatherClient("k") as client:
            await client.forecast(40.0, -73.0)
        params = route.calls.last.request.url.params
        assert params["lat"] == "40.0"
        assert params["lon"] == "-73.0"
        assert params["units"] == "metric"

    @respx.mock
    async def test_missing_list_field_returns_empty(self) -> None:
        respx.get(_FORECAST_URL).mock(return_value=httpx.Response(200, json={}))
        async with OpenWeatherClient("k") as client:
            intervals = await client.forecast(0, 0)
        assert intervals == ()

    @respx.mock
    async def test_network_error(self) -> None:
        respx.get(_FORECAST_URL).mock(side_effect=httpx.ReadTimeout("slow"))
        async with OpenWeatherClient("k") as client:
            with pytest.raises(WeatherAPIError, match="network error"):
                await client.forecast(0, 0)

    @respx.mock
    async def test_malformed_interval(self) -> None:
        respx.get(_FORECAST_URL).mock(
            return_value=httpx.Response(
                200, json={"list": [{"dt": 1, "main": {}, "weather": []}]}
            )
        )
        async with OpenWeatherClient("k") as client:
            with pytest.raises(WeatherAPIError, match="failed to parse"):
                await client.forecast(0, 0)