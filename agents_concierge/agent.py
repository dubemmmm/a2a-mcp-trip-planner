"""Concierge Agent — orchestrates trip planning, stores user data via MCP.

The Concierge uses the heavier orchestrator model (`OPENAI_ORCHESTRATOR_MODEL`)
because it does the planning and synthesis work that needs stronger reasoning.

Two flavors of construction:
  * `build_concierge_agent(...)` — Layer 4-style. MCP tools only. Used in
    isolation tests and by the A2A server wrapper.
  * `build_orchestrating_concierge(...)` — Layer 6. MCP tools PLUS delegation
    tools that call Travel and Local Experience over A2A. This is the
    agent users actually talk to in the full system.
"""

from __future__ import annotations

import os
import sys

from agents import Agent, FunctionTool
from agents.mcp import MCPServer, MCPServerStdio, MCPServerStreamableHttp

from shared.config import get_settings

CONCIERGE_AGENT_NAME = "concierge"

# Instructions for the standalone (no-A2A) Concierge.
CONCIERGE_AGENT_INSTRUCTIONS = """You are the Concierge agent in a trip-planning system.

You coordinate a trip end to end. Your tools so far are user-data tools:
  * `get_user_preferences(user_id)` — read a user's stored preferences before
    planning. Unknown users return safe defaults; that's fine.
  * `save_user_preferences(prefs)` — persist updates when the user explicitly
    shares a new preference (home airport, interests, dietary restrictions).
  * `save_itinerary(user_id, itinerary)` — persist a finalized itinerary.
  * `list_itineraries(user_id)` — recall a user's past trips when context
    would help.

Guidelines:
  * Always start by fetching preferences if you have a user_id; don't ask
    the user for information that's already stored.
  * Be conservative about writing: only `save_user_preferences` for facts the
    user volunteered, not anything you inferred.
  * Itineraries are saved only when the user has explicitly approved them.
  * Keep responses concise; users are deciding, not browsing.
""".strip()


# Extra instructions appended when delegation tools are attached.
ORCHESTRATING_CONCIERGE_INSTRUCTIONS = """You are the Concierge agent in a trip-planning system.

You coordinate a trip end to end by combining your own user-data tools with
three specialist agents you can call:

User-data tools (your own MCP):
  * `get_user_preferences(user_id)` — read stored preferences. Unknown users
    return safe defaults.
  * `save_user_preferences(prefs)` — persist updates the user explicitly shares.
  * `save_itinerary(user_id, itinerary)` — persist a finalized itinerary.
  * `list_itineraries(user_id)` — recall a user's past trips.

Specialist agents (call via A2A):
  * `call_travel_agent(request)` — flights, hotels, total cost. The Travel
    agent has access to real flight and hotel search; ask in natural language.
  * `call_local_experience_agent(request)` — attractions, restaurants, weather.
    Use for anything about what to do at the destination or local conditions.
  * `call_transit_agent(request)` — local routes, directions, and travel time
    between places at the destination (transit, driving, walking, cycling).
    Use for getting around once the traveler has arrived.

How to plan a trip:
  1. If you have a user_id, fetch preferences first.
  2. Decompose the user's request. Most full trips need several specialists:
     travel/lodging, things to do and weather, and getting around.
  3. Send each specialist a focused request with the destination, dates, and
     constraints. Run them in whatever order makes sense — they are
     independent, so you can call both in the same turn.
  4. Synthesize a coherent itinerary from the responses. Don't re-call a
     specialist for information you already have.
  5. Don't ask the user for information that's in their preferences or in a
     specialist's response.
  6. Itineraries are saved only when the user has explicitly approved them.

If a specialist fails (timeout, no results), say so plainly and either retry
with a clearer request or proceed with what you have.

Keep responses concise; users are deciding, not browsing.
""".strip()


def build_concierge_mcp_server(*, mcp_url: str | None = None) -> MCPServer:
    if mcp_url is not None:
        return MCPServerStreamableHttp(
            params={"url": mcp_url},
            name=f"{CONCIERGE_AGENT_NAME}-mcp",
        )
    return MCPServerStdio(
        # Pass the parent environment so the subprocess inherits secrets
        # (e.g. OPENAI_API_KEY). MCPServerStdio otherwise uses a filtered
        # default env that strips everything but PATH/HOME/etc.
        params={
            "command": sys.executable,
            "args": ["-m", "mcp_servers.concierge"],
            "env": {**os.environ},
        },
        name=f"{CONCIERGE_AGENT_NAME}-mcp",
        cache_tools_list=True,
    )


def build_concierge_agent(
    *,
    mcp_server: MCPServer | None = None,
    mcp_url: str | None = None,
    model: str | None = None,
) -> Agent[None]:
    """Construct the standalone Concierge Agent (Layer 4-style, no delegation).

    Uses the orchestrator model by default; pass `model` to override.
    """
    server = (
        mcp_server if mcp_server is not None
        else build_concierge_mcp_server(mcp_url=mcp_url)
    )
    return Agent[None](
        name=CONCIERGE_AGENT_NAME,
        instructions=CONCIERGE_AGENT_INSTRUCTIONS,
        mcp_servers=[server],
        model=model or get_settings().openai_orchestrator_model,
    )


def build_orchestrating_concierge(
    delegation_tools: list[FunctionTool],
    *,
    mcp_server: MCPServer | None = None,
    mcp_url: str | None = None,
    model: str | None = None,
) -> Agent[None]:
    """Construct the Concierge with delegation tools attached.

    This is the Layer 6 entry point. The agent has both its own MCP tools
    (user data) and the function tools that call Travel and Local Experience
    over A2A. The LLM decides which to invoke and in what order.

    Args:
        delegation_tools: Output of `build_delegation_tools(...)` —
            typically two tools, one per specialist.
        mcp_server: Pre-built Concierge MCP server. If None, builds one.
        mcp_url: HTTP URL for the MCP server (ignored if `mcp_server` given).
        model: Model override; defaults to `OPENAI_ORCHESTRATOR_MODEL`.
    """
    server = (
        mcp_server if mcp_server is not None
        else build_concierge_mcp_server(mcp_url=mcp_url)
    )
    return Agent[None](
        name=CONCIERGE_AGENT_NAME,
        instructions=ORCHESTRATING_CONCIERGE_INSTRUCTIONS,
        mcp_servers=[server],
        tools=list(delegation_tools),
        model=model or get_settings().openai_orchestrator_model,
    )
