"""MCP server for the Ground Transport ("Getting Around") Agent.

Exposes:
  * get_route(origin, destination, mode?)
  * compare_modes(origin, destination, modes?)
  * total_travel_minutes(routes)   — pure; sums a list of RouteOption legs

The server is built via `build_server(routes_client)` so tests can inject a
client with respx mocks. The `__main__` block reads the real Google Routes
key from config and runs over stdio.
"""

from __future__ import annotations

import asyncio
from typing import Any

import mcp.types as types
from mcp.server import Server

from agents_transit.clients.google_routes import GoogleRoutesClient
from agents_transit.models import RouteOption, TravelMode
from agents_transit.tools.routing import (
    compare_modes,
    get_route,
    total_travel_minutes,
)

from .scaffolding import serve_stdio, to_json_text

_MODE_VALUES = [m.value for m in TravelMode]


def build_server(routes_client: GoogleRoutesClient) -> Server[Any]:
    """Construct the MCP server with `routes_client` bound to its tools."""
    server: Server[Any] = Server("ground-transport")

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="get_route",
                description=(
                    "Find the best route between two points for one travel "
                    "mode. Returns null when no route exists for that mode."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string", "minLength": 1},
                        "destination": {"type": "string", "minLength": 1},
                        "mode": {
                            "type": "string",
                            "enum": _MODE_VALUES,
                            "default": TravelMode.TRANSIT.value,
                        },
                    },
                    "required": ["origin", "destination"],
                    "additionalProperties": False,
                },
            ),
            types.Tool(
                name="compare_modes",
                description=(
                    "Get one route per travel mode between two points, sorted "
                    "fastest-first. Modes with no available route are dropped. "
                    "Defaults to transit, drive, and walk."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string", "minLength": 1},
                        "destination": {"type": "string", "minLength": 1},
                        "modes": {
                            "type": "array",
                            "items": {"type": "string", "enum": _MODE_VALUES},
                            "minItems": 1,
                        },
                    },
                    "required": ["origin", "destination"],
                    "additionalProperties": False,
                },
            ),
            types.Tool(
                name="total_travel_minutes",
                description=(
                    "Sum the duration of a sequence of route legs into one "
                    "total in minutes. Pure computation; no API call. Pass "
                    "the legs as JSON objects matching the RouteOption schema."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "routes": {
                            "type": "array",
                            "items": {"type": "object"},
                        },
                    },
                    "required": ["routes"],
                    "additionalProperties": False,
                },
            ),
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        if name == "get_route":
            mode = TravelMode(arguments.get("mode", TravelMode.TRANSIT.value))
            result = await get_route(
                routes_client,
                arguments["origin"],
                arguments["destination"],
                mode=mode,
            )
            return to_json_text(result)

        if name == "compare_modes":
            kwargs: dict[str, Any] = {}
            if "modes" in arguments:
                kwargs["modes"] = tuple(TravelMode(v) for v in arguments["modes"])
            results = await compare_modes(
                routes_client,
                arguments["origin"],
                arguments["destination"],
                **kwargs,
            )
            return to_json_text(results)

        if name == "total_travel_minutes":
            legs = tuple(
                RouteOption.model_validate(r) for r in arguments["routes"]
            )
            total = total_travel_minutes(legs)
            return to_json_text({"total_minutes": total})

        raise ValueError(f"unknown tool: {name}")

    return server


async def _main() -> None:
    """Production entrypoint. Resolves the key lazily — server boots even if
    the key is missing; the tools report the missing key when called."""
    from shared.config import MissingAPIKeyError, get_api_key

    def _key_or_none(service: str) -> str | None:
        try:
            return get_api_key(service)
        except MissingAPIKeyError:
            return None

    client = GoogleRoutesClient(_key_or_none("google_routes"))
    try:
        await serve_stdio(build_server(client))
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(_main())
