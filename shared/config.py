"""Centralized, typed configuration.

Settings are loaded from environment variables (and a `.env` file if present)
and validated by Pydantic. `get_settings()` is cached so the validation cost is
paid exactly once per process.

Use `get_api_key(service)` instead of reading the secret fields directly — it
gives clear errors when an optional key is missing at the moment you need it.

"""

from __future__ import annotations
from enum import Enum
from functools import lru_cache

from pydantic import Field, SecretStr

from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    DEV = "dev"
    PROD = "prod"
    TEST = "test"


class LogFormat(str, Enum):
    JSON = "json"
    PRETTY = "pretty"


class Settings(BaseSettings):
    """All runtime configurations, validated at load time"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -- Environment ---

    environment: Environment = Environment.DEV

    # -- OpenAI (required) --
    openai_api_key: SecretStr = Field(..., description="OpenAI API Key")
    openai_default_model: str = "gpt-5.4-mini"
    openai_orchestrator_model: str = "gpt-5.5"

    # --- External APIs (optional at Layer 1) ---
    serpapi_api_key: SecretStr | None = None
    google_places_api_key: SecretStr | None = None
    google_routes_api_key: SecretStr | None = None
    openweather_api_key: SecretStr | None = None
    brave_api_key: SecretStr | None = None

    # -- Agent network --
    travel_a2a_url: str = "http://127.0.0.1:8002"
    local_a2a_url: str = "http://127.0.0.1:8001"
    transit_a2a_url: str = "http://127.0.0.1:8004"

    # -- Agent economy --
    # One shared-secret key per agent; the ledger holds them all, each agent
    # only its own. Missing keys mean that agent cannot use the ledger.
    ledger_url: str = "http://127.0.0.1:8005"
    ledger_db_path: str = "ledger.db"
    economy_admin_key: SecretStr | None = None
    agent_key_concierge: SecretStr | None = None
    agent_key_travel: SecretStr | None = None
    agent_key_local: SecretStr | None = None
    agent_key_transit: SecretStr | None = None
    economy_daily_allowance_mc: int = Field(default=50_000, ge=0)
    economy_daily_cap_mc: int = Field(default=50_000, ge=0)
    economy_hold_seconds: float = Field(default=180.0, gt=0)
    economy_sweep_interval_seconds: float = Field(default=30.0, gt=0)
    # Concierge spend policy: caps on a single delegation and on one trip.
    economy_max_call_mc: int = Field(default=3_000, gt=0)
    economy_max_trip_mc: int = Field(default=8_000, ge=0)

    # -- Public web app --
    web_plan_timeout_seconds: float = Field(default=360.0, gt=0)
    web_max_turns: int = Field(default=18, ge=1, le=50)
    web_max_concurrent_plans: int = Field(default=2, ge=1, le=20)

    # -- Logging --
    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.PRETTY


class MissingAPIKeyError(RuntimeError):
    """Raised when code requests an API key that was not configured."""


# Services whose keys are tracked in Settings. Maps service name → field name.
_KEY_FIELDS: dict[str, str] = {
    "openai": "openai_api_key",
    "serpapi": "serpapi_api_key",
    "google_places": "google_places_api_key",
    "google_routes": "google_routes_api_key",
    "openweather": "openweather_api_key",
    "brave": "brave_api_key",
}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings instance (loaded and validated once)."""
    return Settings()


def get_api_key(service: str) -> str:
    """Return the API key for `service`, raising if unknown or unconfigured.

    Args:
        service: Lowercase service identifier, e.g. "openai", "serpapi".

    Raises:
        MissingAPIKeyError: If the service is unknown, or its key is not set.
    """
    key_field = _KEY_FIELDS.get(service.lower())
    if key_field is None:
        known = ", ".join(sorted(_KEY_FIELDS))
        raise MissingAPIKeyError(f"Unknown service '{service}'. Known services: {known}")

    settings = get_settings()
    secret: SecretStr | None = getattr(settings, key_field)
    if secret is None:
        env_var = key_field.upper()
        raise MissingAPIKeyError(f"API key for '{service}' is not configured. Set {env_var}.")
    return secret.get_secret_value()


def reset_settings_cache() -> None:
    """Drop the cached Settings. Useful in tests after mutating the environment."""
    get_settings.cache_clear()
