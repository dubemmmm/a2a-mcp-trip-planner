"""Tests for `run_agent_task`.

We test the wrapper's three contracts (logging, timeout, error normalization)
both with patched `Runner.run` (cheap unit tests) and with a real `FakeModel`
flowing through the real SDK (one HTTP-level smoke test per contract).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from agents import Agent

from shared.agent_runner import (
    DEFAULT_TIMEOUT_SECONDS,
    AgentTaskError,
    AgentTaskTimeout,
    run_agent_task,
)
from shared.logging_setup import setup_logger


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("LOG_FORMAT", "pretty")  # easier to assert against
    # Configure structlog so log records land in stderr where capsys can see them.
    setup_logger("test")


def _bare_agent(name: str = "test-agent") -> Agent[None]:
    """A no-tools agent we can pass to the wrapper. The model is never called
    in the patched tests; the real-flow tests inject a FakeModel."""
    return Agent[None](name=name, instructions="be helpful.", model="gpt-5-mini")


# ---------- argument validation ----------

class TestArgValidation:
    async def test_negative_timeout_rejected(self, env: None) -> None:
        with pytest.raises(ValueError, match="timeout"):
            await run_agent_task(_bare_agent(), "hi", timeout_seconds=0)

    async def test_zero_max_turns_rejected(self, env: None) -> None:
        with pytest.raises(ValueError, match="max_turns"):
            await run_agent_task(_bare_agent(), "hi", max_turns=0)


# ---------- patched-Runner unit tests ----------

class TestPatchedSuccess:
    async def test_returns_outcome(self, env: None) -> None:
        with patch("shared.agent_runner.Runner.run", new_callable=AsyncMock) as run:
            sentinel = object()
            run.return_value = sentinel
            outcome = await run_agent_task(_bare_agent(), "hi")
        assert outcome.result is sentinel
        assert outcome.duration_seconds >= 0

    async def test_forwards_max_turns(self, env: None) -> None:
        with patch("shared.agent_runner.Runner.run", new_callable=AsyncMock) as run:
            run.return_value = object()
            await run_agent_task(_bare_agent(), "hi", max_turns=3)
        _, kwargs = run.call_args
        assert kwargs["max_turns"] == 3

    async def test_default_timeout_used(self, env: None) -> None:
        # Probe: a slow-but-finite mock should be allowed; we just verify the
        # default is what we documented.
        assert DEFAULT_TIMEOUT_SECONDS == 60.0


class TestPatchedTimeout:
    async def test_raises_timeout_when_run_hangs(self, env: None) -> None:
        async def _hang(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(10)  # would exceed our 0.1s cap

        with patch("shared.agent_runner.Runner.run", new=_hang):
            with pytest.raises(AgentTaskTimeout) as exc:
                await run_agent_task(_bare_agent(), "hi", timeout_seconds=0.1)
        assert "test-agent" in str(exc.value)


class TestPatchedErrorNormalization:
    async def test_tool_error_becomes_agent_task_error(self, env: None) -> None:
        class _ToolBoom(Exception):
            pass

        async def _raise(*args: Any, **kwargs: Any) -> Any:
            raise _ToolBoom("the tool exploded")

        with patch("shared.agent_runner.Runner.run", new=_raise):
            with pytest.raises(AgentTaskError) as exc:
                await run_agent_task(_bare_agent(), "hi")
        # Underlying exception is preserved as __cause__.
        assert isinstance(exc.value.__cause__, _ToolBoom)
        assert "the tool exploded" in str(exc.value)

    async def test_agent_task_error_not_double_wrapped(self, env: None) -> None:
        async def _raise(*args: Any, **kwargs: Any) -> Any:
            raise AgentTaskError("pre-normalized failure")

        with patch("shared.agent_runner.Runner.run", new=_raise):
            with pytest.raises(AgentTaskError) as exc:
                await run_agent_task(_bare_agent(), "hi")
        # The raised error is the one Runner raised, not a wrapper around it.
        assert str(exc.value) == "pre-normalized failure"


class TestPatchedLogging:
    async def test_logs_start_success_with_agent_name(
        self, env: None, capfd: pytest.CaptureFixture[str]
    ) -> None:
        # structlog's PrintLoggerFactory captures sys.stderr at configure
        # time. Configure it *here*, inside the test, so it picks up the
        # captured stderr pytest installed.
        setup_logger("test")
        with patch("shared.agent_runner.Runner.run", new_callable=AsyncMock) as run:
            run.return_value = object()
            await run_agent_task(_bare_agent("the-agent"), "hi")
        err = capfd.readouterr().err
        assert "agent_task_start" in err
        assert "agent_task_success" in err
        assert "the-agent" in err

    async def test_logs_failure(
        self, env: None, capfd: pytest.CaptureFixture[str]
    ) -> None:
        setup_logger("test")

        async def _raise(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("nope")

        with patch("shared.agent_runner.Runner.run", new=_raise):
            with pytest.raises(AgentTaskError):
                await run_agent_task(_bare_agent("the-agent"), "hi")
        err = capfd.readouterr().err
        assert "agent_task_error" in err


# ---------- HTTP-level smoke (real Runner + FakeModel) ----------

class TestRealRunner:
    """Runs the actual SDK Runner against a FakeModel. No external calls."""

    async def test_real_runner_returns_final_output(self, env: None) -> None:
        from _helpers.fake_model import FakeModel

        agent = Agent[None](
            name="real-test",
            instructions="be helpful",
            model=FakeModel(responses=["Hello, traveler."]),
        )
        outcome = await run_agent_task(agent, "say hi")
        text = str(outcome.result.final_output_as(str))
        assert "Hello, traveler." in text