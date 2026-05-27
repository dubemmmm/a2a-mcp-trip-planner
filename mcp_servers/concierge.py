"""MCP server for the Concierge Agent.

Exposes user-data tools backed by the SQLite store:
  * get_user_preferences(user_id)
  * save_user_preferences(prefs)
  * save_itinerary(user_id, itinerary)
  * list_itineraries(user_id)
"""

from __future__ import annotations

import asyncio
from typing import Any

import mcp.types as types
from mcp.server import Server

from agents_concierge.models import UserPreferences
from agents_concierge.store.sqlite_store import ConciergeStore
from agents_concierge.tools.user_data import (
    get_user_preferences,
    list_itineraries,
    save_itinerary,
    save_user_preferences,
)
from shared.models import Itinerary

from .scaffolding import serve_stdio, to_json_text


def build_server(store: ConciergeStore) -> Server[Any]:
    server: Server[Any] = Server("concierge")

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="get_user_preferences",
                description="Get a user's stored trip preferences. Unknown users get sensible defaults.",
                inputSchema={
                    "type": "object",
                    "properties": {"user_id": {"type": "string", "minLength": 1}},
                    "required": ["user_id"],
                    "additionalProperties": False,
                },
            ),
            types.Tool(
                name="save_user_preferences",
                description="Insert or update a user's stored preferences.",
                inputSchema={
                    "type": "object",
                    "properties": {"prefs": {"type": "object"}},
                    "required": ["prefs"],
                    "additionalProperties": False,
                },
            ),
            types.Tool(
                name="save_itinerary",
                description="Persist a planned itinerary for a user. Returns the stored record.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "minLength": 1},
                        "itinerary": {"type": "object"},
                    },
                    "required": ["user_id", "itinerary"],
                    "additionalProperties": False,
                },
            ),
            types.Tool(
                name="list_itineraries",
                description="List a user's saved itineraries, newest first.",
                inputSchema={
                    "type": "object",
                    "properties": {"user_id": {"type": "string", "minLength": 1}},
                    "required": ["user_id"],
                    "additionalProperties": False,
                },
            ),
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        if name == "get_user_preferences":
            prefs = await get_user_preferences(store, arguments["user_id"])
            return to_json_text(prefs)

        if name == "save_user_preferences":
            prefs = UserPreferences.model_validate(arguments["prefs"])
            await save_user_preferences(store, prefs)
            return to_json_text({"ok": True, "user_id": prefs.user_id})

        if name == "save_itinerary":
            itinerary = Itinerary.model_validate(arguments["itinerary"])
            record = await save_itinerary(store, arguments["user_id"], itinerary)
            return to_json_text(record)

        if name == "list_itineraries":
            records = await list_itineraries(store, arguments["user_id"])
            return to_json_text(records)

        raise ValueError(f"unknown tool: {name}")

    return server


async def _main() -> None:
    from shared.config import get_settings

    # Production DB lives next to the config file; override via CONCIERGE_DB_PATH
    # once we have a deployment story.
    settings = get_settings()
    db_path = "./concierge.db"
    _ = settings  # reserved for future use (e.g. settings.concierge_db_path)
    async with await ConciergeStore.open(db_path) as store:
        await serve_stdio(build_server(store))


if __name__ == "__main__":
    asyncio.run(_main())