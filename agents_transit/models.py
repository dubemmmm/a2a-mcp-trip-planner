"""Types local to the Ground Transport ("Getting Around") Agent.

These do NOT cross agent boundaries (yet). Anything that needs to travel over
A2A goes in `shared.models`. Promoting a type from here to `shared` is a
deliberate design action — do it when another agent legitimately needs to
consume it.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

_FROZEN = ConfigDict(frozen=True)


class TravelMode(str, Enum):
    """How you move between two points once you've arrived."""

    DRIVE = "drive"
    TRANSIT = "transit"
    WALK = "walk"
    BICYCLE = "bicycle"


class RouteOption(BaseModel):
    """A single way to get from one place to another."""

    model_config = _FROZEN

    origin: str = Field(..., min_length=1)
    destination: str = Field(..., min_length=1)
    mode: TravelMode
    distance_meters: int = Field(..., ge=0)
    duration_minutes: int = Field(..., ge=0)
    summary: str = Field(..., min_length=1)

    @property
    def distance_km(self) -> float:
        return self.distance_meters / 1000
