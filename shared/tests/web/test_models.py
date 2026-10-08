"""Tests for structured public trip requests."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from web.models import WebTripRequest


def _payload() -> dict[str, object]:
    start = date.today() + timedelta(days=10)
    return {
        "origin": "ATL",
        "destination": "Lisbon",
        "start_date": start.isoformat(),
        "end_date": (start + timedelta(days=4)).isoformat(),
        "travelers": 2,
        "budget_usd": 2500,
        "interests": ["food", "history"],
        "lodging_style": "midrange",
        "pace": "balanced",
        "transport_modes": ["transit", "walk"],
        "notes": "One traveler is vegetarian.",
    }


def test_valid_request_builds_explicit_agent_prompt() -> None:
    request = WebTripRequest.model_validate(_payload())

    prompt = request.to_agent_prompt()

    assert "Origin: ATL" in prompt
    assert "Destination: Lisbon" in prompt
    assert "Interests: food, history" in prompt
    assert "Travel agent" in prompt
    assert "Local Experience" in prompt
    assert "Ground Transport" in prompt
    assert "structured response" in prompt
    assert "mark that section partial or unavailable" in prompt


def test_rejects_past_departure() -> None:
    payload = _payload()
    payload["start_date"] = (date.today() - timedelta(days=1)).isoformat()

    with pytest.raises(ValidationError, match="past"):
        WebTripRequest.model_validate(payload)


def test_rejects_empty_interests() -> None:
    payload = _payload()
    payload["interests"] = []

    with pytest.raises(ValidationError):
        WebTripRequest.model_validate(payload)


def test_rejects_matching_origin_and_destination() -> None:
    payload = _payload()
    payload["destination"] = "atl"

    with pytest.raises(ValidationError, match="different"):
        WebTripRequest.model_validate(payload)
