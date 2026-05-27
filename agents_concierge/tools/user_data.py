"""User-data tools the Concierge agent's LLM calls.

These wrap `ConciergeStore` so the MCP layer doesn't import store internals
directly — and so we can swap SQLite for something else later without
touching the agent or MCP wiring.
"""

from __future__ import annotations

from shared.models import Itinerary

from ..models import StoredItinerary, UserPreferences
from ..store.sqlite_store import ConciergeStore


async def get_user_preferences(
    store: ConciergeStore, user_id: str
) -> UserPreferences:
    """Return preferences for `user_id`, or sensible defaults for unknown users."""
    return await store.get_user_preferences(user_id)


async def save_user_preferences(
    store: ConciergeStore, prefs: UserPreferences
) -> None:
    """Upsert a user's preferences."""
    await store.upsert_user_preferences(prefs)


async def save_itinerary(
    store: ConciergeStore, user_id: str, itinerary: Itinerary
) -> StoredItinerary:
    """Persist `itinerary` under `user_id`. Returns the stored record."""
    return await store.save_itinerary(user_id, itinerary)


async def list_itineraries(
    store: ConciergeStore, user_id: str
) -> tuple[StoredItinerary, ...]:
    """List `user_id`'s saved itineraries, newest first."""
    return await store.list_itineraries(user_id)