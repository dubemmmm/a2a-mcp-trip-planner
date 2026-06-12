"""Travel Agent — flight + hotel search and cost estimation via MCP."""

from __future__ import annotations

import os
import sys

from agents import Agent
from agents.mcp import MCPServer, MCPServerStdio, MCPServerStreamableHttp

from shared.config import get_settings

TRAVEL_AGENT_NAME = "travel"

TRAVEL_AGENT_INSTRUCTIONS = """You are the Travel agent in a trip-planning system.

Your job is to find flights and accommodations and estimate total cost:
  * Use `search_flights` for routes. Always pass IATA codes. For round trips
    pass both `depart_date` and `return_date`; omit `return_date` for one-way.
  * Use `search_hotels` for lodging in the destination city.
  * Use `estimate_total_cost` to sum a chosen flight + hotel + optional extras.
    Pass the flight and hotel as JSON objects matching the FlightOption and
    HotelOption schemas.

Guidelines:
  * Stay within any budget the user mentions. Prefer the cheapest viable
    option unless the user explicitly asks for premium.
  * Surface 2-3 flight options and 2-3 hotel options; don't dump every result.
  * When results are empty, say so plainly and suggest a fix (different dates,
    nearby airport) rather than fabricating data.
  * Cite the airline, flight number, hotel name, and prices exactly as the
    tools returned them.
""".strip()


def build_travel_mcp_server(*, mcp_url: str | None = None) -> MCPServer:
    if mcp_url is not None:
        return MCPServerStreamableHttp(
            params={"url": mcp_url},
            name=f"{TRAVEL_AGENT_NAME}-mcp",
        )
    return MCPServerStdio(
        # Pass the parent environment so the subprocess inherits secrets
        # (e.g. OPENAI_API_KEY). MCPServerStdio otherwise uses a filtered
        # default env that strips everything but PATH/HOME/etc.
        params={
            "command": sys.executable,
            "args": ["-m", "mcp_servers.travel"],
            "env": {**os.environ},
        },
        name=f"{TRAVEL_AGENT_NAME}-mcp",
        cache_tools_list=True,
    )


def build_travel_agent(
    *,
    mcp_server: MCPServer | None = None,
    mcp_url: str | None = None,
    model: str | None = None,
) -> Agent[None]:
    """Construct the Travel Agent."""
    server = mcp_server if mcp_server is not None else build_travel_mcp_server(mcp_url=mcp_url)
    return Agent[None](
        name=TRAVEL_AGENT_NAME,
        instructions=TRAVEL_AGENT_INSTRUCTIONS,
        mcp_servers=[server],
        model=model or get_settings().openai_default_model,
    )