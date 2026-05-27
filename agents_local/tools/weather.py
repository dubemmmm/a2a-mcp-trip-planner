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
from ..models import DailyWeather, WeatherForecast


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

    Notes:
        Free-tier forecast only covers ~5 days from now. Days outside that
        window are silently omitted; callers can detect this by comparing the
        number of returned days to the requested span.
    """
    if not city.strip():
        raise ValueError("city must be non-empty")
    if end_date <= start_date:
        raise ValueError("end_date must be after start_date")

    geocode = await client.geocode(city)
    intervals = await client.forecast(geocode.lat, geocode.lon)

    by_day: dict[date, list[ForecastInterval]] = defaultdict(list)
    for interval in intervals:
        d = _interval_date(interval)
        if start_date <= d < end_date:
            by_day[d].append(interval)

    daily = tuple(
        _aggregate_day(day, by_day[day]) for day in sorted(by_day) if by_day[day]
    )
    return WeatherForecast(city=geocode.name, days=daily)