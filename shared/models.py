"""Shared data contracts.

These are the wire-level types passed between agents (over A2A) and returned
from MCP tools. Anything that crosses an agent boundary should be defined here
so changes are reviewed in one place. Models are frozen by default — agents
should produce new instances rather than mutating in place.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Interest(str, Enum):
    FOOD = "food"
    HISTORY = "history"
    NATURE = "nature"
    ART = "art"
    NIGHTLIFE = "nightlife"
    SHOPPING = "shopping"
    ARCHITECTURE = "architecture"
    BEACHES = "beaches"


class PriceLevel(int, Enum):
    BUDGET = 1
    MODERATE = 2
    UPSCALE = 3
    LUXURY = 4


_FROZEN = ConfigDict(frozen=True)


class TripRequest(BaseModel):
    """User-facing trip request that enters the system at the Concierge."""

    model_config = _FROZEN

    user_id: str = Field(..., min_length=1)
    destination: str = Field(..., min_length=1, description="City or region")
    origin: str = Field(..., min_length=1, description="Departure city or airport code")
    start_date: date
    end_date: date
    budget_usd: Decimal = Field(..., gt=0)
    interests: tuple[Interest, ...] = Field(default_factory=tuple)
    travelers: int = Field(default=1, ge=1, le=20)
    notes: str | None = None

    @model_validator(mode="after")
    def _end_after_start(self) -> "TripRequest":
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be strictly after start_date")
        return self

    @property
    def nights(self) -> int:
        return (self.end_date - self.start_date).days


class FlightOption(BaseModel):
    model_config = _FROZEN

    airline: str = Field(..., min_length=1)
    flight_number: str = Field(..., min_length=1)
    origin: str = Field(..., min_length=1)
    destination: str = Field(..., min_length=1)
    depart_at: datetime
    arrive_at: datetime
    price_usd: Decimal = Field(..., gt=0)
    stops: int = Field(default=0, ge=0)
    booking_url: str | None = None

    @model_validator(mode="after")
    def _arrive_after_depart(self) -> "FlightOption":
        if self.arrive_at <= self.depart_at:
            raise ValueError("arrive_at must be after depart_at")
        return self


class HotelOption(BaseModel):
    model_config = _FROZEN

    name: str = Field(..., min_length=1)
    city: str = Field(..., min_length=1)
    check_in: date
    check_out: date
    price_per_night_usd: Decimal = Field(..., gt=0)
    rating: float | None = Field(default=None, ge=0, le=5)
    address: str | None = None
    booking_url: str | None = None

    @model_validator(mode="after")
    def _checkout_after_checkin(self) -> "HotelOption":
        if self.check_out <= self.check_in:
            raise ValueError("check_out must be after check_in")
        return self

    @property
    def total_price_usd(self) -> Decimal:
        nights = (self.check_out - self.check_in).days
        return self.price_per_night_usd * nights


class Attraction(BaseModel):
    model_config = _FROZEN

    name: str = Field(..., min_length=1)
    category: Interest
    description: str = Field(..., min_length=1)
    address: str | None = None
    price_level: PriceLevel | None = None
    rating: float | None = Field(default=None, ge=0, le=5)
    duration_minutes: int | None = Field(default=None, ge=0)


class DailyPlan(BaseModel):
    model_config = _FROZEN

    day: date
    activities: tuple[Attraction, ...] = Field(default_factory=tuple)
    notes: str | None = None


class Itinerary(BaseModel):
    """Final output the Concierge synthesizes for the user."""

    model_config = _FROZEN

    trip_request: TripRequest
    selected_flight: FlightOption | None = None
    selected_hotel: HotelOption | None = None
    daily_plans: tuple[DailyPlan, ...] = Field(default_factory=tuple)
    estimated_total_usd: Decimal = Field(..., ge=0)
    summary: str = Field(..., min_length=1)