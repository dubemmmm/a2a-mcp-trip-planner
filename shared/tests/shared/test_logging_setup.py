"""Tests for shared.logging_setup."""

from __future__ import annotations

import io
import json
import logging

import pytest
import structlog

from shared.logging_setup import setup_logger


@pytest.fixture
def env_with_log_format(monkeypatch: pytest.MonkeyPatch):
    """Factory: build the minimal env, allowing override of LOG_FORMAT/LOG_LEVEL."""
    def _setup(fmt: str = "json", level: str = "INFO") -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("ENVIRONMENT", "test")
        monkeypatch.setenv("LOG_FORMAT", fmt)
        monkeypatch.setenv("LOG_LEVEL", level)
    return _setup


def _capture_log(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """Redirect structlog's PrintLogger output (it writes to stderr by default)."""
    buf = io.StringIO()
    # The PrintLoggerFactory captures sys.stderr at configure time, so we
    # patch sys.stderr before setup_logger is called.
    import sys
    monkeypatch.setattr(sys, "stderr", buf)
    return buf


class TestSetupLogger:
    def test_returns_bound_logger(self, env_with_log_format) -> None:
        env_with_log_format()
        log = setup_logger("travel_agent")
        # structlog's bound logger isn't a stdlib Logger; just check it's a logger.
        assert hasattr(log, "info")
        assert hasattr(log, "bind")

    def test_agent_name_in_output(
        self, env_with_log_format, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_with_log_format(fmt="json")
        buf = _capture_log(monkeypatch)
        log = setup_logger("travel_agent")
        log.info("hello")
        output = buf.getvalue()
        assert output, "expected log output"
        record = json.loads(output.strip().splitlines()[-1])
        assert record["agent"] == "travel_agent"
        assert record["event"] == "hello"

    def test_json_format_is_parseable(
        self, env_with_log_format, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_with_log_format(fmt="json")
        buf = _capture_log(monkeypatch)
        log = setup_logger("concierge")
        log.info("event_x", count=3, ok=True)
        line = buf.getvalue().strip().splitlines()[-1]
        record = json.loads(line)
        assert record["event"] == "event_x"
        assert record["count"] == 3
        assert record["ok"] is True
        assert record["level"] == "info"
        assert "timestamp" in record

    def test_pretty_format_is_not_json(
        self, env_with_log_format, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_with_log_format(fmt="pretty")
        buf = _capture_log(monkeypatch)
        log = setup_logger("local_experience")
        log.info("readable_event")
        line = buf.getvalue().strip().splitlines()[-1]
        with pytest.raises(json.JSONDecodeError):
            json.loads(line)
        assert "readable_event" in line
        assert "local_experience" in line

    def test_log_level_filters(
        self, env_with_log_format, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_with_log_format(fmt="json", level="WARNING")
        buf = _capture_log(monkeypatch)
        log = setup_logger("travel_agent")
        log.debug("debug_event")
        log.info("info_event")
        log.warning("warn_event")
        output = buf.getvalue()
        assert "debug_event" not in output
        assert "info_event" not in output
        assert "warn_event" in output

    def test_invalid_log_level_raises(
        self, env_with_log_format, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_with_log_format(level="LOUD")  # not a real level
        with pytest.raises(ValueError):
            setup_logger("travel_agent")

    def test_bind_adds_extra_context(
        self, env_with_log_format, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_with_log_format(fmt="json")
        buf = _capture_log(monkeypatch)
        log = setup_logger("travel_agent").bind(task_id="t-1")
        log.info("doing_work")
        record = json.loads(buf.getvalue().strip().splitlines()[-1])
        assert record["agent"] == "travel_agent"
        assert record["task_id"] == "t-1"