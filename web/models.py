"""Validated request and response types for the public trip-planning form."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agents_concierge.models import TripPlanResponse
from shared.models import Interest

LodgingStyle = Literal["budget", "midrange", "upscale", "luxury"]
TripPace = Literal["relaxed", "balanced", "active"]
TransportMode = Literal["transit", "drive", "walk", "bicycle"]
TRIP_PLAN_SCHEMA_VERSION: Literal["trip-plan.v1"] = "trip-plan.v1"


class WebTripRequest(BaseModel):
    """Structured trip preferences submitted by the public form."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    origin: str = Field(..., min_length=2, max_length=100)
    destination: str = Field(..., min_length=2, max_length=100)
    start_date: date
    end_date: date
    travelers: int = Field(default=1, ge=1, le=20)
    budget_usd: Decimal = Field(..., ge=100, le=100_000)
    interests: tuple[Interest, ...] = Field(..., min_length=1, max_length=8)
    lodging_style: LodgingStyle = "midrange"
    pace: TripPace = "balanced"
    transport_modes: tuple[TransportMode, ...] = Field(
        default=("transit", "walk"), min_length=1, max_length=4
    )
    notes: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _validate_dates_and_locations(self) -> "WebTripRequest":
        if self.start_date < date.today():
            raise ValueError("start_date cannot be in the past")
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        if self.origin.casefold() == self.destination.casefold():
            raise ValueError("origin and destination must be different")
        return self

    @property
    def nights(self) -> int:
        return (self.end_date - self.start_date).days

    def to_agent_prompt(self) -> str:
        """Turn validated form fields into an explicit orchestration request."""
        interests = ", ".join(interest.value for interest in self.interests)
        modes = ", ".join(self.transport_modes)
        notes = self.notes or "None"
        return (
            "Plan this trip end to end using the specialist agents as tools, then "
            "aggregate their results into one practical itinerary.\n\n"
            f"Origin: {self.origin}\n"
            f"Destination: {self.destination}\n"
            f"Dates: {self.start_date.isoformat()} through {self.end_date.isoformat()} "
            f"({self.nights} nights)\n"
            f"Travelers: {self.travelers}\n"
            f"Total budget: ${self.budget_usd}\n"
            f"Interests: {interests}\n"
            f"Lodging style: {self.lodging_style}\n"
            f"Itinerary pace: {self.pace}\n"
            f"Preferred local transport: {modes}\n"
            f"Additional notes: {notes}\n\n"
            "Ask the Travel agent for flights and lodging, the Local Experience "
            "agent for attractions, restaurants, and weather, and the Ground "
            "Transport agent for useful routes between the airport, hotel area, "
            "and recommended activities. Clearly distinguish live weather from "
            "requests outside the provider's forecast coverage. Respect the total "
            "budget and do not invent unavailable prices or routes. Populate the "
            "required structured response faithfully. When a specialist fails, "
            "mark that section partial or unavailable and add a clear alert instead "
            "of fabricating data. Keep summaries concise and put route details on "
            "the relevant activities."
        )


class PlanResult(BaseModel):
    """Successful result returned to the browser."""

    schema_version: Literal["trip-plan.v1"] = TRIP_PLAN_SCHEMA_VERSION
    plan: TripPlanResponse
    duration_seconds: float = Field(..., ge=0)
    validation_retried: bool = False
