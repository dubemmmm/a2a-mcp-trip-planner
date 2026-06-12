"""Ground Transport ("Getting Around") Agent — routing and mode comparison via MCP.

Mirrors the Travel and Local Experience agents: the agent reaches its tools
through an MCP server, connected either over stdio (in-process dev, the
default) or Streamable HTTP (cross-process deployment via `mcp_url=...`).

`build_transit_agent` returns the Agent; the MCP server object it depends on
is returned separately by `build_transit_mcp_server` so the caller owns the
server's lifecycle (entering it as an async context manager).
"""

from __future__ import annotations

import os
import sys

from agents import Agent
from agents.mcp import MCPServer, MCPServerStdio, MCPServerStreamableHttp

from shared.config import get_settings

TRANSIT_AGENT_NAME = "ground-transport"

TRANSIT_AGENT_INSTRUCTIONS = """You are the Ground Transport agent in a trip-planning system.

Your job is to help travelers get around once they've arrived at a destination:
  * Use `get_route` to find the best route between two points for a single
    travel mode (transit, drive, walk, or bicycle). Transit is the default.
  * Use `compare_modes` when the user wants to weigh options — it returns one
    route per mode, sorted fastest-first, dropping modes with no route.
  * Use `total_travel_minutes` to sum a sequence of legs (e.g. a day's hops)
    into a single travel-time figure. Pass the legs as JSON objects matching
    the RouteOption schema.

Guidelines:
  * Lead with the fastest sensible option, but mention trade-offs (a walk that
    saves a transfer, a drive that beats transit by a wide margin).
  * Report distances and durations exactly as the tools return them; don't
    round away the numbers the traveler needs.
  * When no route exists for a mode (e.g. no transit coverage), say so plainly
    rather than inventing one.
  * Origins and destinations are free text (e.g. "Lisbon Airport",
    "Belem Tower"); pass them through as the user phrased them.
""".strip()


def build_transit_mcp_server(*, mcp_url: str | None = None) -> MCPServer:
    """Construct the MCP server connector for the Ground Transport agent.

    Defaults to stdio (spawns `python -m mcp_servers.transit`). Pass `mcp_url`
    to use the Streamable HTTP transport instead.
    """
    if mcp_url is not None:
        return MCPServerStreamableHttp(
            params={"url": mcp_url},
            name=f"{TRANSIT_AGENT_NAME}-mcp",
        )
    return MCPServerStdio(
        # Pass the parent environment so the subprocess inherits secrets
        # (e.g. OPENAI_API_KEY). MCPServerStdio otherwise uses a filtered
        # default env that strips everything but PATH/HOME/etc.
        params={
            "command": sys.executable,
            "args": ["-m", "mcp_servers.transit"],
            "env": {**os.environ},
        },
        name=f"{TRANSIT_AGENT_NAME}-mcp",
        cache_tools_list=True,
    )


def build_transit_agent(
    *,
    mcp_server: MCPServer | None = None,
    mcp_url: str | None = None,
    model: str | None = None,
) -> Agent[None]:
    """Construct the Ground Transport Agent.

    Args:
        mcp_server: Pre-built MCP server. Tests pass this to inject a
            mock-or-test server. If None, builds one from `mcp_url` or stdio.
        mcp_url: HTTP URL for the MCP server (ignored if `mcp_server` given).
        model: Optional model override; defaults to `OPENAI_DEFAULT_MODEL`.
    """
    server = (
        mcp_server if mcp_server is not None
        else build_transit_mcp_server(mcp_url=mcp_url)
    )
    return Agent[None](
        name=TRANSIT_AGENT_NAME,
        instructions=TRANSIT_AGENT_INSTRUCTIONS,
        mcp_servers=[server],
        model=model or get_settings().openai_default_model,
    )
