"""ASGI-level tests for the public trip-planning application."""

from __future__ import annotations

from datetime import date, timedelta

import httpx

from shared.agent_events import AgentEventQueue, emit_agent_event
from web.app import create_app
from web.models import PlanResult, WebTripRequest

from ._sample import sample_trip_plan


def _payload() -> dict[str, object]:
    start = date.today() + timedelta(days=10)
    return {
        "origin": "ATL",
        "destination": "Lisbon",
        "start_date": start.isoformat(),
        "end_date": (start + timedelta(days=4)).isoformat(),
        "travelers": 1,
        "budget_usd": 1800,
        "interests": ["food", "history"],
        "lodging_style": "midrange",
        "pace": "balanced",
        "transport_modes": ["transit", "walk"],
        "notes": None,
    }


async def _fake_plan(trip: WebTripRequest) -> PlanResult:
    return PlanResult(
        plan=sample_trip_plan(),
        duration_seconds=1.25,
    )


async def test_homepage_serves_structured_form() -> None:
    app = create_app(plan_runner=_fake_plan)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/")

    assert response.status_code == 200
    assert "Where are you headed?" in response.text
    assert 'name="interests"' in response.text
    assert 'name="transport_modes"' in response.text
    assert "/static/itinerary.js?v=trip-plan.v1" in response.text
    assert 'class="itinerary-content"' in response.text
    assert response.headers["cache-control"] == "no-store, max-age=0"
    assert response.headers["x-trip-planner-schema"] == "trip-plan.v1"


async def test_health_endpoint() -> None:
    app = create_app(plan_runner=_fake_plan)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "trip-planner-web",
        "response_schema": "trip-plan.v1",
    }


async def test_plan_endpoint_returns_aggregated_result() -> None:
    app = create_app(plan_runner=_fake_plan)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/api/plan", json=_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "trip-plan.v1"
    assert payload["plan"]["title"] == "Lisbon food and history escape"
    assert payload["plan"]["flights"]["status"] == "unavailable"
    assert payload["plan"]["hotels"]["recommended_option_index"] == 0
    assert payload["duration_seconds"] == 1.25
    assert payload["validation_retried"] is False
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-trip-planner-schema"] == "trip-plan.v1"


async def test_plan_endpoint_returns_json_safe_validation_error() -> None:
    payload = _payload()
    payload["interests"] = []
    app = create_app(plan_runner=_fake_plan)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/api/plan", json=payload)

    assert response.status_code == 422
    assert response.json()["error"] == "Please correct the highlighted trip details."
    assert response.json()["details"]


async def test_plan_endpoint_rejects_non_json_body() -> None:
    app = create_app(plan_runner=_fake_plan)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/plan", content="not json", headers={"content-type": "application/json"}
        )

    assert response.status_code == 400


async def test_live_plan_stream_returns_events_and_final_result() -> None:
    app = create_app(plan_runner=_fake_plan)

    async def live_plan(trip: WebTripRequest, events: AgentEventQueue) -> PlanResult:
        for status in ("submitted", "working", "completed"):
            emit_agent_event(
                events,
                kind="agent.status",
                agent="travel",
                status=status,
                message=f"Travel is {status}",
            )
        return await _fake_plan(trip)

    app.state.streaming_plan_runner = live_plan
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        accepted = await client.post(
            "/api/plan",
            json=_payload(),
            headers={"Prefer": "respond-async"},
        )
        stream = await client.get(accepted.json()["stream_url"])

    assert accepted.status_code == 202
    assert accepted.json()["plan_id"]
    assert stream.status_code == 200
    assert stream.headers["content-type"].startswith("text/event-stream")
    assert '"status":"submitted"' in stream.text
    assert '"status":"working"' in stream.text
    assert '"status":"completed"' in stream.text
    assert '"kind":"plan.completed"' in stream.text
    assert '"schema_version":"trip-plan.v1"' in stream.text


async def test_live_plan_stream_requires_known_plan_id() -> None:
    app = create_app(plan_runner=_fake_plan)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        missing = await client.get("/api/plan/stream")
        unknown = await client.get("/api/plan/stream?plan_id=missing")

    assert missing.status_code == 400
    assert unknown.status_code == 404
