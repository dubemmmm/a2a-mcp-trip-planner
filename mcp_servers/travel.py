"""MCP server for the Travel Agent.

Exposes:
  * search_flights(origin, destination, depart_date, return_date?, max_price_usd?, adults?)
  * search_hotels(city, check_in, check_out, max_price_per_night_usd?, adults?)
  * estimate_total_cost(flight, hotel, extras_usd?)
"""

from __future__ import annotations

import asyncio
from datetime import date
from decimal import Decimal
from typing import Any

import mcp.types as types
from mcp.server import Server

from agents_travel.clients.serpapi import SerpAPIClient
from agents_travel.tools.flights import search_flights
from agents_travel.tools.hotels import search_hotels
from agents_travel.tools.pricing import estimate_total_cost
from shared.models import FlightOption, HotelOption

from .scaffolding import serve_stdio, to_json_text


def build_server(serpapi_client: SerpAPIClient) -> Server[Any]:
    server: Server[Any] = Server("travel")

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="search_flights",
                description="Search flights between two airports.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string", "minLength": 3},
                        "destination": {"type": "string", "minLength": 3},
                        "depart_date": {"type": "string", "format": "date"},
                        "return_date": {"type": "string", "format": "date"},
                        "max_price_usd": {"type": "number", "exclusiveMinimum": 0},
                        "adults": {"type": "integer", "minimum": 1, "default": 1},
                    },
                    "required": ["origin", "destination", "depart_date"],
                    "additionalProperties": False,
                },
            ),
            types.Tool(
                name="search_hotels",
                description="Search hotels in a city between two dates.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "minLength": 1},
                        "check_in": {"type": "string", "format": "date"},
                        "check_out": {"type": "string", "format": "date"},
                        "max_price_per_night_usd": {"type": "number", "exclusiveMinimum": 0},
                        "adults": {"type": "integer", "minimum": 1, "default": 1},
                    },
                    "required": ["city", "check_in", "check_out"],
                    "additionalProperties": False,
                },
            ),
            types.Tool(
                name="estimate_total_cost",
                description="Sum flight, hotel, and extras into a single total. Pure function.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "flight": {"type": "object"},
                        "hotel": {"type": "object"},
                        "extras_usd": {"type": "number", "minimum": 0, "default": 0},
                    },
                    "required": ["flight", "hotel"],
                    "additionalProperties": False,
                },
            ),
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        if name == "search_flights":
            flight_max_price = (
                Decimal(str(arguments["max_price_usd"]))
                if "max_price_usd" in arguments
                else None
            )
            return_date = (
                date.fromisoformat(arguments["return_date"])
                if "return_date" in arguments
                else None
            )
            flights = await search_flights(
                serpapi_client,
                origin=arguments["origin"],
                destination=arguments["destination"],
                depart_date=date.fromisoformat(arguments["depart_date"]),
                return_date=return_date,
                max_price_usd=flight_max_price,
                adults=arguments.get("adults", 1),
            )
            return to_json_text(flights)

        if name == "search_hotels":
            hotel_max_price = (
                Decimal(str(arguments["max_price_per_night_usd"]))
                if "max_price_per_night_usd" in arguments
                else None
            )
            hotels = await search_hotels(
                serpapi_client,
                city=arguments["city"],
                check_in=date.fromisoformat(arguments["check_in"]),
                check_out=date.fromisoformat(arguments["check_out"]),
                max_price_per_night_usd=hotel_max_price,
                adults=arguments.get("adults", 1),
            )
            return to_json_text(hotels)

        if name == "estimate_total_cost":
            flight = FlightOption.model_validate(arguments["flight"])
            hotel = HotelOption.model_validate(arguments["hotel"])
            extras = Decimal(str(arguments.get("extras_usd", 0)))
            total = estimate_total_cost(flight, hotel, extras_usd=extras)
            return to_json_text({"total_usd": str(total)})

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

    client = SerpAPIClient(_key_or_none("serpapi"))
    print('hereeeeeee!!!')
    try:
        await serve_stdio(build_server(client))
    finally:
        
        await client.aclose()

if __name__ == "__main__":
    asyncio.run(_main())