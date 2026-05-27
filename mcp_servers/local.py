"""MCP server for the Local Experience Agent.

Exposes three tools:
  * find_attractions(city, interests, max_results_per_interest)
  * recommend_restaurants(city, cuisine, price_level, max_results)
  * get_weather_forecast(city, start_date, end_date)

The server is built via `build_server(places_client, weather_client)` so tests
can inject pre-configured clients with respx mocks. The `__main__` block reads
real API keys from config and runs over stdio.
"""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

import mcp.types as types
from mcp.server import Server

from agents_local.clients.google_places import GooglePlacesClient
from agents_local.clients.openweather import OpenWeatherClient
from agents_local.tools.attractions import find_attractions, recommend_restaurants
from agents_local.tools.weather import get_weather_forecast
from shared.models import Interest, PriceLevel

from .scaffolding import serve_stdio, to_json_text


def build_server(
    places_client: GooglePlacesClient,
    weather_client: OpenWeatherClient,
) -> Server[Any]:
    """Construct the MCP server with the given clients bound to its tools."""
    server: Server[Any] = Server("local-experience")

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="find_attractions",
                description="Find attractions in a city across one or more interest categories.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "minLength": 1},
                        "interests": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [i.value for i in Interest],
                            },
                            "minItems": 1,
                        },
                        "max_results_per_interest": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 20,
                            "default": 5,
                        },
                    },
                    "required": ["city", "interests"],
                    "additionalProperties": False,
                },
            ),
            types.Tool(
                name="recommend_restaurants",
                description="Recommend restaurants in a city, optionally filtered by cuisine and price.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "minLength": 1},
                        "cuisine": {"type": "string"},
                        "price_level": {
                            "type": "string",
                            "enum": [p.name for p in PriceLevel],
                        },
                        "max_results": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 20,
                            "default": 10,
                        },
                    },
                    "required": ["city"],
                    "additionalProperties": False,
                },
            ),
            types.Tool(
                name="get_weather_forecast",
                description="Get daily weather forecast for a city between two dates (end exclusive).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "minLength": 1},
                        "start_date": {"type": "string", "format": "date"},
                        "end_date": {"type": "string", "format": "date"},
                    },
                    "required": ["city", "start_date", "end_date"],
                    "additionalProperties": False,
                },
            ),
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        if name == "find_attractions":
            interests = tuple(Interest(v) for v in arguments["interests"])
            results = await find_attractions(
                places_client,
                arguments["city"],
                interests,
                max_results_per_interest=arguments.get("max_results_per_interest", 5),
            )
            return to_json_text(results)

        if name == "recommend_restaurants":
            price_level = (
                PriceLevel[arguments["price_level"]]
                if "price_level" in arguments
                else None
            )
            results = await recommend_restaurants(
                places_client,
                arguments["city"],
                cuisine=arguments.get("cuisine"),
                price_level=price_level,
                max_results=arguments.get("max_results", 10),
            )
            return to_json_text(results)

        if name == "get_weather_forecast":
            result = await get_weather_forecast(
                weather_client,
                arguments["city"],
                date.fromisoformat(arguments["start_date"]),
                date.fromisoformat(arguments["end_date"]),
            )
            return to_json_text(result)

        raise ValueError(f"unknown tool: {name}")

    return server


async def _main() -> None:
    """Production entrypoint. Resolves keys lazily — server boots even if
    keys are missing; individual tools will report the missing key when called."""
    from shared.config import MissingAPIKeyError, get_api_key

    def _key_or_none(service: str) -> str | None:
        try:
            return get_api_key(service)
        except MissingAPIKeyError:
            return None

    places = GooglePlacesClient(_key_or_none("google_places"))
    weather = OpenWeatherClient(_key_or_none("openweather"))
    try:
        await serve_stdio(build_server(places, weather))
    finally:
        await places.aclose()
        await weather.aclose()


if __name__ == "__main__":
    asyncio.run(_main())