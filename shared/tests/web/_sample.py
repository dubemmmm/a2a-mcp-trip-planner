"""Representative structured trip plan used by web-layer tests."""

from __future__ import annotations

from datetime import date, timedelta

from agents_concierge.models import (
    ActivitySummary,
    AlertSeverity,
    AvailabilityStatus,
    BudgetStatus,
    BudgetSummary,
    FlightSection,
    HotelSection,
    HotelSummary,
    ItineraryDaySummary,
    PlanningAlert,
    TransportOptionSummary,
    TripPlanResponse,
    WeatherDaySummary,
    WeatherSection,
)
from shared.models import Interest


def sample_trip_plan() -> TripPlanResponse:
    start = date.today() + timedelta(days=10)
    end = start + timedelta(days=4)
    return TripPlanResponse(
        title="Lisbon food and history escape",
        summary="A central four-night stay with food, history, and simple transit.",
        origin="ATL",
        destination="Lisbon",
        start_date=start,
        end_date=end,
        travelers=1,
        interests=(Interest.FOOD, Interest.HISTORY),
        pace="balanced",
        alerts=(
            PlanningAlert(
                severity=AlertSeverity.WARNING,
                title="Flight prices unavailable",
                message="The flight search timed out; retry before booking.",
                source="Travel agent",
            ),
        ),
        budget=BudgetSummary(
            total_budget_usd=1800,
            flight_estimate_usd=None,
            lodging_estimate_usd=620,
            estimated_trip_total_usd=None,
            remaining_usd=1180,
            status=BudgetStatus.UNKNOWN,
            notes=("Flight costs are not included yet.",),
        ),
        flights=FlightSection(
            status=AvailabilityStatus.UNAVAILABLE,
            summary="No verified flight options were returned.",
            recommended_option_index=None,
            options=(),
        ),
        hotels=HotelSection(
            status=AvailabilityStatus.AVAILABLE,
            summary="Two central options fit the lodging target.",
            recommended_option_index=0,
            options=(
                HotelSummary(
                    name="Central Lisbon Hotel",
                    neighborhood="Baixa",
                    price_per_night_usd=155,
                    total_price_usd=620,
                    rating=4.5,
                    why_it_fits="Walkable to food and historic sights.",
                    caveat=None,
                ),
            ),
        ),
        weather=WeatherSection(
            status=AvailabilityStatus.AVAILABLE,
            requested_start_date=start,
            requested_end_date=end,
            available_start_date=start,
            available_end_date=end,
            summary="Live coverage is available for the full trip.",
            packing_advice=("Bring comfortable walking shoes.",),
            days=(
                WeatherDaySummary(
                    day=start,
                    high_celsius=25,
                    low_celsius=17,
                    description="Clear sky",
                    chance_of_rain=0.1,
                ),
            ),
        ),
        days=(
            ItineraryDaySummary(
                day=start,
                title="Arrival and Baixa",
                weather_note="Clear and comfortable for walking.",
                activities=(
                    ActivitySummary(
                        period="Evening",
                        title="Explore Baixa",
                        description="Check in and take an easy orientation walk.",
                        location="Baixa",
                        estimated_cost_usd=25,
                        transport_options=(
                            TransportOptionSummary(
                                mode="transit",
                                duration_minutes=35,
                                distance_km=None,
                                note="Airport to the hotel area.",
                            ),
                        ),
                    ),
                ),
            ),
        ),
        transport_strategy=("Use transit for airport transfers.",),
        next_steps=("Retry the flight search.", "Reserve refundable lodging."),
    )
