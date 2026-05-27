"""Concierge-local data types.

`UserPreferences` and `StoredItinerary` live here rather than in `shared/`
because no other agent needs them. They become candidates for promotion to
`shared/` when (and only when) a second agent legitimately reads or writes
them. Until then, they stay private to the Concierge.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from shared.models import Interest, Itinerary, PriceLevel

_FROZEN = ConfigDict(frozen=True)


class UserPreferences(BaseModel):
    """A user's stored trip-planning preferences.

    Defaults are intentionally permissive: an unknown user gets sensible
    behavior without the tool crashing, which is the behavior we want for
    the Concierge agent's first-time flow.
    """

    model_config = _FROZEN

    user_id: str = Field(..., min_length=1)
    home_airport: str | None = None
    preferred_interests: tuple[Interest, ...] = Field(default_factory=tuple)
    preferred_price_level: PriceLevel | None = None
    dietary_restrictions: tuple[str, ...] = Field(default_factory=tuple)
    notes: str | None = None

    @classmethod
    def default_for(cls, user_id: str) -> "UserPreferences":
        """Return a permissive default record for an unknown user."""
        return cls(user_id=user_id)


class StoredItinerary(BaseModel):
    """An itinerary plus its persistence metadata."""

    model_config = _FROZEN

    id: int
    user_id: str = Field(..., min_length=1)
    itinerary: Itinerary
    created_at: datetime