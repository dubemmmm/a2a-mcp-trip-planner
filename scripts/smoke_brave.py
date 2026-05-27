# scripts/smoke_brave.py
"""Test that Brave works. Uses 1 query from your 2k/month free tier."""
import asyncio
from shared.clients.brave import BraveSearchClient
from shared.tools.web_search import web_search
from shared.config import get_api_key


async def main() -> None:
    async with BraveSearchClient(get_api_key("brave")) as client:
        results = await web_search(client, "Portugal travel visa requirements", count=3)
    for hit in results:
        print(f"- {hit.title}")
        print(f"  {hit.url}")
        print(f"  {hit.snippet[:100]}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())