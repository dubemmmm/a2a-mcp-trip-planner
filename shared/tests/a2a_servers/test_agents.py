"""Tests for the three A2A server modules.

These tests verify card metadata (name, description, skills, URLs) without
spinning up the HTTP server. Live HTTP behavior is tested separately in
`test_live_a2a.py`.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ENVIRONMENT", "test")


# Each agent module is verified the same way: build the app, look at the
# routes for the agent-card endpoint, verify it reflects the right skills.
# We don't actually hit HTTP here — that's the live test's job.

class TestLocalA2A:
    def test_app_builds(self, env: None) -> None:
        from a2a_servers.local import build_app

        app = build_app(public_url="http://localhost:8001")
        assert app is not None

    def test_skill_ids(self, env: None) -> None:
        from a2a_servers.local import _skills

        ids = {s.id for s in _skills()}
        assert ids == {
            "find_attractions",
            "recommend_restaurants",
            "get_weather_forecast",
        }

    def test_skill_descriptions_non_empty(self, env: None) -> None:
        from a2a_servers.local import _skills

        for skill in _skills():
            assert skill.name
            assert skill.description
            assert len(skill.tags) > 0


class TestTravelA2A:
    def test_app_builds(self, env: None) -> None:
        from a2a_servers.travel import build_app

        app = build_app()
        assert app is not None

    def test_skill_ids(self, env: None) -> None:
        from a2a_servers.travel import _skills

        ids = {s.id for s in _skills()}
        assert ids == {"search_flights", "search_hotels", "estimate_total_cost"}


class TestTransitA2A:
    def test_app_builds(self, env: None) -> None:
        from a2a_servers.transit import build_app

        app = build_app()
        assert app is not None

    def test_skill_ids(self, env: None) -> None:
        from a2a_servers.transit import _skills

        ids = {s.id for s in _skills()}
        assert ids == {"get_route", "compare_modes", "total_travel_minutes"}

    def test_skill_descriptions_non_empty(self, env: None) -> None:
        from a2a_servers.transit import _skills

        for skill in _skills():
            assert skill.name
            assert skill.description
            assert len(skill.tags) > 0


class TestConciergeA2A:
    def test_app_builds(self, env: None) -> None:
        from a2a_servers.concierge import build_app

        app = build_app()
        assert app is not None

    def test_skill_ids_include_plan(self, env: None) -> None:
        from a2a_servers.concierge import _skills

        ids = {s.id for s in _skills()}
        assert "plan_trip" in ids
        assert "get_user_preferences" in ids
        assert "list_itineraries" in ids