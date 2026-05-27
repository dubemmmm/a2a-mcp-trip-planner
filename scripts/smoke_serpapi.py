# scripts/smoke_serpapi.py
"""Test that SerpApi flight search works. Uses 1 of your 100 monthly searches."""
import asyncio
from datetime import date, timedelta
from agents_travel.clients.serpapi import SerpAPIClient
from agents_travel.tools.flights import search_flights
from shared.config import get_api_key


async def main() -> None:
    depart = date.today() + timedelta(days=30)
    async with SerpAPIClient(get_api_key("serpapi")) as client:
        results = await search_flights(
            client,
            origin="ATL",
            destination="LIS",
            depart_date=depart,
        )
    print(f"Found {len(results)} flight options:")
    for flight in results[:3]:
        print(f"  {flight.airline} {flight.flight_number}: "
              f"${flight.price_usd}, {flight.stops} stops")


if __name__ == "__main__":
    asyncio.run(main())