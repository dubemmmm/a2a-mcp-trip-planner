# scripts/smoke_places.py
"""Test that Google Places works."""
import asyncio
from agents_local.clients.google_places import GooglePlacesClient
from agents_local.tools.attractions import find_attractions
from shared.models import Interest
from shared.config import get_api_key


async def main() -> None:
    async with GooglePlacesClient(get_api_key("google_places")) as client:
        results = await find_attractions(
            client, "Lisbon", [Interest.FOOD, Interest.HISTORY],
            max_results_per_interest=2,
        )
    for attraction in results:
        print(f"- {attraction.name} ({attraction.category.value})")
        if attraction.rating:
            print(f"  Rating: {attraction.rating}")
        print(f"  {attraction.description[:80]}")
        print()


if __name__ == "__main__":
    asyncio.run(main())