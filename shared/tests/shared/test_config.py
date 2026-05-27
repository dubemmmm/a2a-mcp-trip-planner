"""Tests for shared.config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.config import (
    Environment,
    LogFormat,
    MissingAPIKeyError,
    Settings,
    get_api_key,
    get_settings,
    reset_settings_cache,
)


class TestSettings:
    def test_missing_required_key_raises(self) -> None:
        """Without OPENAI_API_KEY, Settings() must fail validation."""
        with pytest.raises(ValidationError) as exc:
            Settings()  # type: ignore[call-arg]
        # The error mentions the missing field, not the secret value.
        assert "openai_api_key" in str(exc.value).lower()

    def test_minimal_valid_env(self, minimal_env: None) -> None:
        s = Settings()  # type: ignore[call-arg]
        assert s.openai_api_key.get_secret_value() == "sk-test-minimal"
        assert s.environment is Environment.TEST
        # Defaults applied:
        assert s.openai_default_model == "gpt-5.4-mini"
        assert s.openai_orchestrator_model == "gpt-5.5"
        assert s.log_format is LogFormat.PRETTY

    def test_secret_not_in_repr(self, minimal_env: None) -> None:
        """SecretStr must hide the value in repr/str (no accidental log leaks)."""
        s = Settings()  # type: ignore[call-arg]
        assert "sk-test-minimal" not in repr(s)
        assert "sk-test-minimal" not in str(s)

    def test_invalid_environment_value_rejected(
        self, minimal_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ENVIRONMENT", "staging")  # not in enum
        with pytest.raises(ValidationError):
            Settings()  # type: ignore[call-arg]

    def test_invalid_log_format_rejected(
        self, minimal_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOG_FORMAT", "xml")
        with pytest.raises(ValidationError):
            Settings()  # type: ignore[call-arg]

    def test_extra_env_vars_ignored(
        self, minimal_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unknown env vars must not crash loading."""
        monkeypatch.setenv("SOME_UNRELATED_VAR", "value")
        s = Settings()  # type: ignore[call-arg]
        assert s.openai_api_key.get_secret_value() == "sk-test-minimal"

    def test_case_insensitive_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("openai_api_key", "sk-lower")  # lowercase
        s = Settings()  # type: ignore[call-arg]
        assert s.openai_api_key.get_secret_value() == "sk-lower"


class TestGetSettingsCache:
    def test_returns_same_instance(self, minimal_env: None) -> None:
        a = get_settings()
        b = get_settings()
        assert a is b

    def test_reset_clears_cache(
        self, minimal_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = get_settings()
        monkeypatch.setenv("OPENAI_API_KEY", "sk-rotated")
        reset_settings_cache()
        second = get_settings()
        assert first is not second
        assert second.openai_api_key.get_secret_value() == "sk-rotated"


class TestGetAPIKey:
    def test_returns_configured_key(self, minimal_env: None) -> None:
        assert get_api_key("openai") == "sk-test-minimal"

    def test_unknown_service_raises(self, minimal_env: None) -> None:
        with pytest.raises(MissingAPIKeyError) as exc:
            get_api_key("twilio")
        assert "twilio" in str(exc.value)
        assert "Known services" in str(exc.value)

    def test_unconfigured_optional_key_raises(self, minimal_env: None) -> None:
        with pytest.raises(MissingAPIKeyError) as exc:
            get_api_key("serpapi")
        assert "SERPAPI_API_KEY" in str(exc.value)

    def test_case_insensitive_service_name(self, minimal_env: None) -> None:
        assert get_api_key("OpenAI") == "sk-test-minimal"
        assert get_api_key("OPENAI") == "sk-test-minimal"

    def test_returns_optional_key_when_set(
        self, minimal_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SERPAPI_API_KEY", "serp-123")
        reset_settings_cache()
        assert get_api_key("serpapi") == "serp-123"