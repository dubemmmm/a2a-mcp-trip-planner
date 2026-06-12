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
    
    environment : Environment = Environment.DEV
    
    # -- OpenAI (required) --
    openai_api_key : SecretStr = Field(..., description="OpenAI API Key")
    openai_default_model: str = "gpt-5.4-mini"
    openai_orchestrator_model : str = "gpt-5.5"
    
    # --- External APIs (optional at Layer 1) ---
    serpapi_api_key: SecretStr | None = None
    google_places_api_key: SecretStr | None = None
    google_routes_api_key: SecretStr | None = None
    openweather_api_key: SecretStr | None = None
    brave_api_key: SecretStr | None = None
    
    # -- Logging --
    log_level : str = "INFO"
    log_format : LogFormat = LogFormat.PRETTY
    
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
        raise MissingAPIKeyError(
            f"Unknown service '{service}'. Known services: {known}"
        )

    settings = get_settings()
    secret: SecretStr | None = getattr(settings, key_field)
    if secret is None:
        env_var = key_field.upper()
        raise MissingAPIKeyError(
            f"API key for '{service}' is not configured. Set {env_var}."
        )
    return secret.get_secret_value()


def reset_settings_cache() -> None:
    """Drop the cached Settings. Useful in tests after mutating the environment."""
    get_settings.cache_clear()

    