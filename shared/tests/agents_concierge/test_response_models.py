"""Validation tests for the structured Concierge response contract."""

from __future__ import annotations

import pytest
from agents.agent_output import AgentOutputSchema
from pydantic import ValidationError

from agents_concierge.models import FlightSection, HotelSection, TripPlanResponse
from shared.tests.web._sample import sample_trip_plan


def test_structured_plan_round_trips_json() -> None:
    plan = sample_trip_plan()

    restored = type(plan).model_validate_json(plan.model_dump_json())

    assert restored == plan
    assert restored.alerts[0].title == "Flight prices unavailable"
    assert restored.hotels.options[0].total_price_usd == 620


def test_agents_sdk_builds_strict_schema() -> None:
    schema = AgentOutputSchema(TripPlanResponse).json_schema()

    assert schema["additionalProperties"] is False
    assert "days" in schema["required"]
    assert "budget" in schema["required"]
    assert "weather" in schema["required"]


def test_unavailable_provider_section_can_have_no_options() -> None:
    plan = sample_trip_plan()

    assert plan.flights.options == ()
    assert plan.flights.recommended_option_index is None


def test_recommended_flight_index_must_exist() -> None:
    with pytest.raises(ValidationError, match="outside options"):
        FlightSection(
            status="available",
            summary="A recommendation was selected.",
            recommended_option_index=0,
            options=(),
        )


def test_recommended_hotel_index_must_exist() -> None:
    with pytest.raises(ValidationError, match="outside options"):
        HotelSection(
            status="available",
            summary="A recommendation was selected.",
            recommended_option_index=2,
            options=(),
        )


def test_itinerary_days_must_fall_inside_trip() -> None:
    payload = sample_trip_plan().model_dump(mode="json")
    payload["days"][0]["day"] = "2040-01-01"

    with pytest.raises(ValidationError, match="outside the trip dates"):
        type(sample_trip_plan()).model_validate(payload)
