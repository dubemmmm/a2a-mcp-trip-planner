"""Smoke test: prove the Local MCP server boots and exposes its tools.

This bypasses the production entrypoint (which needs all API keys) and uses
the same in-process pattern the tests use. No external accounts needed.

Run:  python scripts/smoke_mcp.py
"""
import asyncio

from mcp.shared.memory import create_connected_server_and_client_session

from agents_local.clients.google_places import GooglePlacesClient
from agents_local.clients.openweather import OpenWeatherClient
from mcp_servers.local import build_server


async def main() -> None:
    # Dummy keys are fine here — we're listing tools, not calling them.
    # The clients only make network requests when a tool actually fires.
    places = GooglePlacesClient("dummy-key")
    weather = OpenWeatherClient("dummy-key")
    try:
        server = build_server(places, weather)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.list_tools()
            print(f"Local MCP server exposes {len(result.tools)} tools:")
            for tool in result.tools:
                desc = tool.description or ""
                print(f"  - {tool.name}: {desc[:70]}")
    finally:
        await places.aclose()
        await weather.aclose()


if __name__ == "__main__":
    asyncio.run(main())