"""Tests for shared.tracing."""

from __future__ import annotations

import pytest


def _make_env(monkeypatch: pytest.MonkeyPatch, env: str) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ENVIRONMENT", env)


class TestSetupTracing:
    def test_disabled_in_test_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _make_env(monkeypatch, "test")
        from shared.tracing import setup_tracing
        enabled = setup_tracing()
        assert enabled is False

    def test_enabled_in_dev_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _make_env(monkeypatch, "dev")
        from shared.tracing import setup_tracing
        enabled = setup_tracing()
        assert enabled is True

    def test_enabled_in_prod_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _make_env(monkeypatch, "prod")
        from shared.tracing import setup_tracing
        enabled = setup_tracing()
        assert enabled is True

    def test_calls_sdk_with_correct_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify we actually flip the SDK's tracing switch."""
        _make_env(monkeypatch, "test")
        recorded: list[bool] = []

        def fake_set_tracing_disabled(disabled: bool) -> None:
            recorded.append(disabled)

        # Patch the symbol where shared.tracing imported it.
        import shared.tracing as tracing_module
        monkeypatch.setattr(
            tracing_module, "set_tracing_disabled", fake_set_tracing_disabled
        )

        tracing_module.setup_tracing()
        assert recorded == [True]