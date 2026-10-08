"""Shared pytest fixtures.

Settings are cached via `lru_cache`, so any test that mutates env vars must
drop the cache afterward, or it leaks into other tests. The `clean_env`
fixture (autouse) handles that.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from shared import config as config_module


# Every env var the application reads. Listed explicitly so a typo in
# a real env var name is caught here rather than silently leaking.
_APP_ENV_VARS = (
    "ENVIRONMENT",
    "OPENAI_API_KEY",
    "OPENAI_DEFAULT_MODEL",
    "OPENAI_ORCHESTRATOR_MODEL",
    "SERPAPI_API_KEY",
    "GOOGLE_PLACES_API_KEY",
    "GOOGLE_ROUTES_API_KEY",
    "OPENWEATHER_API_KEY",
    "TRAVEL_A2A_URL",
    "LOCAL_A2A_URL",
    "TRANSIT_A2A_URL",
    "LEDGER_URL",
    "LEDGER_DB_PATH",
    "ECONOMY_ADMIN_KEY",
    "AGENT_KEY_CONCIERGE",
    "AGENT_KEY_TRAVEL",
    "AGENT_KEY_LOCAL",
    "AGENT_KEY_TRANSIT",
    "ECONOMY_DAILY_ALLOWANCE_MC",
    "ECONOMY_DAILY_CAP_MC",
    "ECONOMY_HOLD_SECONDS",
    "ECONOMY_SWEEP_INTERVAL_SECONDS",
    "WEB_PLAN_TIMEOUT_SECONDS",
    "WEB_MAX_TURNS",
    "WEB_MAX_CONCURRENT_PLANS",
    "LOG_LEVEL",
    "LOG_FORMAT",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[None]:
    """Wipe app env vars and the settings cache around every test.

    Also chdirs into a tmp dir so an accidental `.env` in the repo root
    can't leak into the test.
    """
    for var in _APP_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    config_module.reset_settings_cache()
    yield
    config_module.reset_settings_cache()


@pytest.fixture
def minimal_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The smallest valid environment: just OPENAI_API_KEY set."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-minimal")
    monkeypatch.setenv("ENVIRONMENT", "test")
