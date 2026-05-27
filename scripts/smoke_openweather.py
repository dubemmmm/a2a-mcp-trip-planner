# scripts/smoke_openweather.py
"""Test that OpenWeather works. Free API, no cost."""
import asyncio
from datetime import date, timedelta
from agents_local.clients.openweather import OpenWeatherClient
from agents_local.tools.weather import get_weather_forecast
from shared.config import get_api_key


async def main() -> None:
    today = date.today()
    async with OpenWeatherClient(get_api_key("openweather")) as client:
        forecast = await get_weather_forecast(
            client, "Lisbon", today, today + timedelta(days=3)
        )
    print(f"City: {forecast.city}")
    for day in forecast.days:
        print(f"  {day.date}: {day.low_celsius:.1f}-{day.high_celsius:.1f}°C, "
              f"{day.description}, {day.chance_of_rain*100:.0f}% rain")


if __name__ == "__main__":
    asyncio.run(main())