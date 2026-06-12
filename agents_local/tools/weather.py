"""Weather forecast tool.

OpenWeather's free tier gives 3-hour intervals over ~5 days. The agent wants
*daily* summaries, so this tool aggregates intervals into per-day buckets
clipped to the user-requested date range.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timezone

from ..clients.openweather import (
    ForecastInterval,
    OpenWeatherClient,
)
from ..models import (
    DailyWeather,
    ForecastCoverage,
    ForecastCoverageStatus,
    WeatherForecast,
)


def _interval_date(interval: ForecastInterval) -> date:
    return datetime.fromtimestamp(interval.dt, tz=timezone.utc).date()


def _aggregate_day(day: date, intervals: list[ForecastInterval]) -> DailyWeather:
    highs = [i.main.temp_max for i in intervals]
    lows = [i.main.temp_min for i in intervals]
    rain_chances = [i.pop for i in intervals]

    # The description from the interval closest to local midday is most
    # representative; here we approximate with the median interval.
    middle = intervals[len(intervals) // 2]
    description = (
        middle.weather[0].description if middle.weather else "no description available"
    )

    return DailyWeather(
        date=day,
        high_celsius=max(highs),
        low_celsius=min(lows),
        description=description,
        chance_of_rain=max(rain_chances) if rain_chances else 0.0,
    )


def _requested_dates(start_date: date, end_date: date) -> set[date]:
    return {
        date.fromordinal(ordinal)
        for ordinal in range(start_date.toordinal(), end_date.toordinal())
    }


async def get_weather_forecast(
    client: OpenWeatherClient,
    city: str,
    start_date: date,
    end_date: date,
) -> WeatherForecast:
    """Return a daily forecast for `city` between `start_date` and `end_date`.

    Args:
        client: Open `OpenWeatherClient`.
        city: City name (geocoded internally).
        start_date: First day (inclusive).
        end_date: Last day (exclusive — matches our date-range convention).

    The response includes coverage metadata because the free-tier provider only
    returns roughly five days. Callers can distinguish a complete forecast from
    a partial response or a request entirely outside the available window.
    """
    if not city.strip():
        raise ValueError("city must be non-empty")
    if end_date <= start_date:
        raise ValueError("end_date must be after start_date")

    geocode = await client.geocode(city)
    intervals = await client.forecast(geocode.lat, geocode.lon)

    all_available_dates = {_interval_date(interval) for interval in intervals}
    requested_dates = _requested_dates(start_date, end_date)
    covered_requested_dates = requested_dates & all_available_dates

    if requested_dates <= all_available_dates:
        coverage_status = ForecastCoverageStatus.FULL
    elif covered_requested_dates:
        coverage_status = ForecastCoverageStatus.PARTIAL
    else:
        coverage_status = ForecastCoverageStatus.UNAVAILABLE

    coverage = ForecastCoverage(
        requested_start_date=start_date,
        requested_end_date_exclusive=end_date,
        available_start_date=min(all_available_dates) if all_available_dates else None,
        available_end_date_inclusive=max(all_available_dates) if all_available_dates else None,
        status=coverage_status,
    )

    by_day: dict[date, list[ForecastInterval]] = defaultdict(list)
    for interval in intervals:
        d = _interval_date(interval)
        if start_date <= d < end_date:
            by_day[d].append(interval)

    daily = tuple(
        _aggregate_day(day, by_day[day]) for day in sorted(by_day) if by_day[day]
    )
    return WeatherForecast(city=geocode.name, coverage=coverage, days=daily)
