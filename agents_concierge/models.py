"""Concierge-local data types.

`UserPreferences` and `StoredItinerary` live here rather than in `shared/`
because no other agent needs them. They become candidates for promotion to
`shared/` when (and only when) a second agent legitimately reads or writes
them. Until then, they stay private to the Concierge.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class AvailabilityStatus(str, Enum):
    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class BudgetStatus(str, Enum):
    WITHIN_BUDGET = "within_budget"
    TIGHT = "tight"
    OVER_BUDGET = "over_budget"
    UNKNOWN = "unknown"


class PlanningAlert(BaseModel):
    """A provider failure, limitation, or important caveat for the traveler."""

    model_config = _FROZEN

    severity: AlertSeverity
    title: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    source: str | None = None


class BudgetSummary(BaseModel):
    """Known and estimated costs without pretending unavailable prices exist."""

    model_config = _FROZEN

    total_budget_usd: float = Field(..., ge=0)
    flight_estimate_usd: float | None = Field(default=None, ge=0)
    lodging_estimate_usd: float | None = Field(default=None, ge=0)
    estimated_trip_total_usd: float | None = Field(default=None, ge=0)
    remaining_usd: float | None = None
    status: BudgetStatus
    notes: tuple[str, ...] = Field(default_factory=tuple)


class FlightSummary(BaseModel):
    """One flight option suitable for a compact comparison card."""

    model_config = _FROZEN

    airline: str = Field(..., min_length=1)
    flight_number: str | None = None
    departure: str = Field(..., min_length=1)
    arrival: str = Field(..., min_length=1)
    price_usd: float | None = Field(default=None, ge=0)
    stops: int | None = Field(default=None, ge=0)
    note: str | None = None


class FlightSection(BaseModel):
    model_config = _FROZEN

    status: AvailabilityStatus
    summary: str = Field(..., min_length=1)
    recommended_option_index: int | None = Field(default=None, ge=0)
    options: tuple[FlightSummary, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _recommended_flight_exists(self) -> "FlightSection":
        if self.recommended_option_index is not None and self.recommended_option_index >= len(
            self.options
        ):
            raise ValueError("recommended flight index is outside options")
        return self


class HotelSummary(BaseModel):
    """One lodging option suitable for a compact comparison card."""

    model_config = _FROZEN

    name: str = Field(..., min_length=1)
    neighborhood: str | None = None
    price_per_night_usd: float | None = Field(default=None, ge=0)
    total_price_usd: float | None = Field(default=None, ge=0)
    rating: float | None = Field(default=None, ge=0, le=5)
    why_it_fits: str = Field(..., min_length=1)
    caveat: str | None = None


class HotelSection(BaseModel):
    model_config = _FROZEN

    status: AvailabilityStatus
    summary: str = Field(..., min_length=1)
    recommended_option_index: int | None = Field(default=None, ge=0)
    options: tuple[HotelSummary, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _recommended_hotel_exists(self) -> "HotelSection":
        if self.recommended_option_index is not None and self.recommended_option_index >= len(
            self.options
        ):
            raise ValueError("recommended hotel index is outside options")
        return self


class WeatherDaySummary(BaseModel):
    model_config = _FROZEN

    day: date
    high_celsius: float | None = None
    low_celsius: float | None = None
    description: str = Field(..., min_length=1)
    chance_of_rain: float | None = Field(default=None, ge=0, le=1)


class WeatherSection(BaseModel):
    model_config = _FROZEN

    status: AvailabilityStatus
    requested_start_date: date
    requested_end_date: date
    available_start_date: date | None = None
    available_end_date: date | None = None
    summary: str = Field(..., min_length=1)
    packing_advice: tuple[str, ...] = Field(default_factory=tuple)
    days: tuple[WeatherDaySummary, ...] = Field(default_factory=tuple)


class TransportOptionSummary(BaseModel):
    model_config = _FROZEN

    mode: str = Field(..., min_length=1)
    duration_minutes: int | None = Field(default=None, ge=0)
    distance_km: float | None = Field(default=None, ge=0)
    note: str | None = None


class ActivitySummary(BaseModel):
    """A single itinerary stop with optional local transport details."""

    model_config = _FROZEN

    period: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    location: str | None = None
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    transport_options: tuple[TransportOptionSummary, ...] = Field(default_factory=tuple)


class ItineraryDaySummary(BaseModel):
    model_config = _FROZEN

    day: date
    title: str = Field(..., min_length=1)
    weather_note: str | None = None
    activities: tuple[ActivitySummary, ...] = Field(..., min_length=1)


class TripPlanResponse(BaseModel):
    """Structured Concierge result rendered by the public web application."""

    model_config = _FROZEN

    title: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    origin: str = Field(..., min_length=1)
    destination: str = Field(..., min_length=1)
    start_date: date
    end_date: date
    travelers: int = Field(..., ge=1, le=20)
    interests: tuple[Interest, ...] = Field(default_factory=tuple)
    pace: str = Field(..., min_length=1)
    alerts: tuple[PlanningAlert, ...] = Field(default_factory=tuple)
    budget: BudgetSummary
    flights: FlightSection
    hotels: HotelSection
    weather: WeatherSection
    days: tuple[ItineraryDaySummary, ...] = Field(..., min_length=1)
    transport_strategy: tuple[str, ...] = Field(default_factory=tuple)
    next_steps: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _trip_dates_are_consistent(self) -> "TripPlanResponse":
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        for itinerary_day in self.days:
            if not self.start_date <= itinerary_day.day <= self.end_date:
                raise ValueError("itinerary day is outside the trip dates")
        return self
