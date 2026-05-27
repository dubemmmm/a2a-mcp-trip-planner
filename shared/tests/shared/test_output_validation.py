"""Tests for `validate_agent_output` (output_type-first design)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from agents import Agent
from agents.exceptions import ModelBehaviorError
from pydantic import BaseModel

from shared.agent_runner import AgentTaskError, AgentTaskOutcome
from shared.output_validation import (
    OutputValidationError,
    validate_agent_output,
)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ENVIRONMENT", "test")


class TripSummary(BaseModel):
    destination: str
    nights: int
    total_usd: float


def _bare_agent(name: str = "validate-test") -> Agent[None]:
    return Agent[None](name=name, instructions="be helpful.", model="gpt-5-mini")


def _make_outcome(final_output: Any) -> AgentTaskOutcome:
    """Build an AgentTaskOutcome whose result.final_output is final_output."""

    class _StubResult:
        def __init__(self) -> None:
            self.final_output = final_output

    return AgentTaskOutcome(result=_StubResult(), duration_seconds=0.01)  # type: ignore[arg-type]


def _validation_error(message: str) -> AgentTaskError:
    """Build the exact wrapper error our agent_runner would raise on SDK validation failure."""
    err = AgentTaskError(f"agent failed: ModelBehaviorError: {message}")
    err.__cause__ = ModelBehaviorError(message)
    return err


# ---------- happy path: SDK returns a typed instance ----------

class TestFirstAttemptSuccess:
    async def test_typed_instance_passes_through(self, env: None) -> None:
        valid = TripSummary(destination="Lisbon", nights=4, total_usd=1280.0)
        with patch(
            "shared.output_validation.run_agent_task",
            new_callable=AsyncMock,
            return_value=_make_outcome(valid),
        ):
            v = await validate_agent_output(_bare_agent(), "plan", TripSummary)
        assert not v.retried
        assert v.value is valid

    async def test_dict_coerced_to_schema(self, env: None) -> None:
        """Some custom output schemas could return a dict; we coerce it."""
        payload = {"destination": "Lisbon", "nights": 4, "total_usd": 1280.0}
        with patch(
            "shared.output_validation.run_agent_task",
            new_callable=AsyncMock,
            return_value=_make_outcome(payload),
        ):
            v = await validate_agent_output(_bare_agent(), "plan", TripSummary)
        assert v.value.destination == "Lisbon"

    async def test_json_string_coerced_to_schema(self, env: None) -> None:
        """Non-strict providers may hand us a JSON string."""
        raw = '{"destination": "Porto", "nights": 3, "total_usd": 900}'
        with patch(
            "shared.output_validation.run_agent_task",
            new_callable=AsyncMock,
            return_value=_make_outcome(raw),
        ):
            v = await validate_agent_output(_bare_agent(), "plan", TripSummary)
        assert v.value.nights == 3

    async def test_unexpected_final_output_type_raises(self, env: None) -> None:
        with patch(
            "shared.output_validation.run_agent_task",
            new_callable=AsyncMock,
            return_value=_make_outcome(12345),  # int — nonsense
        ):
            with pytest.raises(TypeError, match="unexpected final_output"):
                await validate_agent_output(_bare_agent(), "plan", TripSummary)


# ---------- agent already has output_type set ----------

class TestExistingOutputType:
    async def test_same_schema_is_fine(self, env: None) -> None:
        agent = _bare_agent().clone(output_type=TripSummary)
        valid = TripSummary(destination="Lisbon", nights=4, total_usd=1280.0)
        with patch(
            "shared.output_validation.run_agent_task",
            new_callable=AsyncMock,
            return_value=_make_outcome(valid),
        ):
            v = await validate_agent_output(agent, "plan", TripSummary)
        assert v.value is valid

    async def test_conflicting_schema_rejected(self, env: None) -> None:
        class OtherSchema(BaseModel):
            x: int

        agent = _bare_agent().clone(output_type=OtherSchema)
        with pytest.raises(ValueError, match="conflicts"):
            await validate_agent_output(agent, "plan", TripSummary)


# ---------- retry path ----------

class TestRetrySucceeds:
    async def test_invalid_then_valid(self, env: None) -> None:
        valid = TripSummary(destination="Lisbon", nights=4, total_usd=1280.0)
        # First call raises a validation error; second call returns valid output.
        call_results: list[Any] = [
            _validation_error("missing field 'nights'"),
            _make_outcome(valid),
        ]

        async def _runner(*args: Any, **kwargs: Any) -> Any:
            r = call_results.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        with patch("shared.output_validation.run_agent_task", new=_runner):
            v = await validate_agent_output(_bare_agent(), "plan", TripSummary)
        assert v.retried is True
        assert v.value is valid

    async def test_retry_prompt_carries_validation_error(self, env: None) -> None:
        valid = TripSummary(destination="L", nights=1, total_usd=10.0)
        captured: list[str] = []

        async def _runner(agent: Any, prompt: str, **kwargs: Any) -> Any:
            captured.append(prompt)
            if len(captured) == 1:
                raise _validation_error("nights must be > 0")
            return _make_outcome(valid)

        with patch("shared.output_validation.run_agent_task", new=_runner):
            await validate_agent_output(_bare_agent(), "original request", TripSummary)
        assert len(captured) == 2
        assert captured[0] == "original request"
        # Retry prompt must include the error message and the original task.
        assert "nights must be > 0" in captured[1]
        assert "original request" in captured[1]


class TestRetryFails:
    async def test_two_validation_failures_raise(self, env: None) -> None:
        async def _runner(*args: Any, **kwargs: Any) -> Any:
            raise _validation_error("schema mismatch")

        with patch("shared.output_validation.run_agent_task", new=_runner):
            with pytest.raises(OutputValidationError) as exc:
                await validate_agent_output(_bare_agent(), "plan", TripSummary)
        assert exc.value.first_error is not None
        assert "schema mismatch" in exc.value.first_error
        assert exc.value.retry_error is not None
        assert "schema mismatch" in exc.value.retry_error

    async def test_retry_runtime_failure_propagates(self, env: None) -> None:
        results: list[Any] = [
            _validation_error("bad first"),
            AgentTaskError("tool blew up on retry"),  # not a validation error
        ]

        async def _runner(*args: Any, **kwargs: Any) -> Any:
            r = results.pop(0)
            raise r

        with patch("shared.output_validation.run_agent_task", new=_runner):
            with pytest.raises(OutputValidationError) as exc:
                await validate_agent_output(_bare_agent(), "plan", TripSummary)
        assert "bad first" in exc.value.first_error
        assert exc.value.retry_error is None


# ---------- non-validation errors bubble unchanged ----------

class TestNonValidationErrors:
    async def test_first_call_non_validation_error_bubbles(self, env: None) -> None:
        """A tool failure or timeout on attempt 1 must NOT be eaten."""
        non_validation = AgentTaskError("tool failed")  # no __cause__ from SDK

        async def _runner(*args: Any, **kwargs: Any) -> Any:
            raise non_validation

        with patch("shared.output_validation.run_agent_task", new=_runner):
            with pytest.raises(AgentTaskError, match="tool failed"):
                await validate_agent_output(_bare_agent(), "plan", TripSummary)


# ---------- agent.clone is used (caller's agent not mutated) ----------

class TestNoMutation:
    async def test_callers_agent_output_type_unchanged(self, env: None) -> None:
        agent = _bare_agent()
        assert agent.output_type is None  # precondition
        valid = TripSummary(destination="L", nights=1, total_usd=10.0)
        with patch(
            "shared.output_validation.run_agent_task",
            new_callable=AsyncMock,
            return_value=_make_outcome(valid),
        ):
            await validate_agent_output(agent, "plan", TripSummary)
        # The function must clone — the caller's agent stays unmodified.
        assert agent.output_type is None


# ---------- real SDK end-to-end (FakeModel, real Runner) ----------

class TestRealSDKHappyPath:
    async def test_real_runner_with_valid_json(self, env: None) -> None:
        """Exercise the actual SDK code path: clone → output_type → Runner →
        Pydantic validation → typed final_output. No patches."""
        from _helpers.fake_model import FakeModel

        agent = Agent[None](
            name="real-validate",
            instructions="output JSON only",
            model=FakeModel(
                responses=['{"destination": "Lisbon", "nights": 4, "total_usd": 1280.0}']
            ),
        )
        v = await validate_agent_output(agent, "plan", TripSummary)
        assert not v.retried
        assert v.value == TripSummary(
            destination="Lisbon", nights=4, total_usd=1280.0
        )

    async def test_real_runner_retries_on_sdk_validation_failure(
        self, env: None
    ) -> None:
        """Bad JSON first, valid JSON second — proves the SDK's
        ModelBehaviorError flows through our retry path."""
        from _helpers.fake_model import FakeModel

        agent = Agent[None](
            name="real-validate-retry",
            instructions="output JSON only",
            model=FakeModel(
                responses=[
                    '{"destination": "Lisbon"}',  # missing nights, total_usd
                    '{"destination": "Lisbon", "nights": 4, "total_usd": 1280.0}',
                ]
            ),
        )
        v = await validate_agent_output(agent, "plan", TripSummary)
        assert v.retried is True
        assert v.value.nights == 4