"""Local Experience Agent — builds the SDK Agent with its MCP tools wired in.

The agent reaches its MCP server one of two ways:
  * stdio   — for in-process dev (default; spawns `python -m mcp_servers.local`)
  * HTTP    — for cross-process deployment (pass `mcp_url=...`)

The MCP server isn't started by `build_local_agent`; it returns the server
*object*, which the caller enters as an async context manager. This keeps
lifecycle explicit and lets multiple call sites share one server connection.
"""

from __future__ import annotations

import os
import sys

from agents import Agent
from agents.mcp import MCPServer, MCPServerStdio, MCPServerStreamableHttp

from shared.config import get_settings

LOCAL_AGENT_NAME = "local-experience"

LOCAL_AGENT_INSTRUCTIONS = """You are the Local Experience agent in a trip-planning system.

Your job is to help travelers discover what to do at a destination:
  * Use `find_attractions` to surface things to do across interest categories.
  * Use `recommend_restaurants` for dining picks, optionally by cuisine and price.
  * Use `get_weather_forecast` when the user asks about conditions or when
    weather would change your recommendations (e.g. outdoor activities).

Guidelines:
  * Prefer 3-5 well-chosen recommendations over long lists.
  * If asked for an itinerary, group attractions by day and respect opening
    hours implied by typical operating patterns.
  * Cite the city you searched in your response so other agents and the user
    can verify the geography matches.
  * When tools return no results, say so plainly and suggest broadening the
    search rather than inventing places.
  * Check the weather tool's `coverage.status`. If it is `partial` or
    `unavailable`, state the requested and available date ranges explicitly;
    never present seasonal expectations as a live forecast.
""".strip()


def build_local_mcp_server(*, mcp_url: str | None = None) -> MCPServer:
    """Construct the MCP server connector for the Local Experience agent.

    Defaults to stdio (spawns `python -m mcp_servers.local`). Pass `mcp_url`
    to use the Streamable HTTP transport instead.
    """
    if mcp_url is not None:
        return MCPServerStreamableHttp(
            params={"url": mcp_url},
            name=f"{LOCAL_AGENT_NAME}-mcp",
        )
    return MCPServerStdio(
        # Pass the parent environment so the subprocess inherits secrets
        # (e.g. OPENAI_API_KEY). MCPServerStdio otherwise uses a filtered
        # default env that strips everything but PATH/HOME/etc.
        params={
            "command": sys.executable,
            "args": ["-m", "mcp_servers.local"],
            "env": {**os.environ},
        },
        name=f"{LOCAL_AGENT_NAME}-mcp",
        cache_tools_list=True,
    )


def build_local_agent(
    *,
    mcp_server: MCPServer | None = None,
    mcp_url: str | None = None,
    model: str | None = None,
) -> Agent[None]:
    """Construct the Local Experience Agent.

    Args:
        mcp_server: Pre-built MCP server. Tests pass this to inject a
            mock-or-test server. If None, builds one from `mcp_url` or stdio.
        mcp_url: HTTP URL for the MCP server (ignored if `mcp_server` given).
        model: Optional model override; defaults to `OPENAI_DEFAULT_MODEL`.
    """
    server = mcp_server if mcp_server is not None else build_local_mcp_server(mcp_url=mcp_url)
    return Agent[None](
        name=LOCAL_AGENT_NAME,
        instructions=LOCAL_AGENT_INSTRUCTIONS,
        mcp_servers=[server],
        model=model or get_settings().openai_default_model,
    )
