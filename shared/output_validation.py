"""Run an agent with a Pydantic output schema and one retry on validation failure.

The primary mechanism is the Agents SDK's own `output_type`: when set, the SDK
sends the schema to the model as a strict JSON schema, the model is constrained
at the token level, and `result.final_output` comes back as a typed instance.
When the SDK validates that output and it fails (which can happen with
non-strict schemas, non-OpenAI providers, or rare strict-mode edge cases), it
raises `ModelBehaviorError`. We catch that, re-prompt the agent once with the
validation error attached, and retry. Two failures → `OutputValidationError`.

Why one retry, not unlimited: if the model can't produce valid JSON twice in
a row, something more serious is wrong than "needs more tries" — typically a
schema mismatch in the prompt or a hallucinated field. Looping burns tokens
without fixing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import structlog
from agents import Agent
from agents.exceptions import ModelBehaviorError
from pydantic import BaseModel

from .agent_runner import AgentTaskError, run_agent_task

_LOGGER = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class OutputValidationError(RuntimeError):
    """Raised when an agent's output cannot be coerced into the expected schema.

    Carries both attempts (raw text and the SDK's error messages) so callers
    can log, surface, or fall back as appropriate.
    """

    def __init__(
        self,
        message: str,
        *,
        first_error: str,
        retry_error: str | None = None,
    ) -> None:
        super().__init__(message)
        self.first_error = first_error
        self.retry_error = retry_error


@dataclass(frozen=True)
class ValidatedOutput(Generic[T]):
    """A successful coercion plus whether a retry was needed to get there."""

    value: T
    retried: bool


def _build_retry_prompt(original_input: str, error_message: str) -> str:
    """Compose a focused retry prompt that includes the validation error."""
    return (
        f"Your previous response did not match the required output schema.\n"
        f"Validation error: {error_message}\n\n"
        f"Please answer the original request again, returning a response that "
        f"strictly matches the schema. Original request:\n{original_input}"
    )


async def validate_agent_output(
    agent: Agent[Any],
    task_input: str,
    schema: type[T],
    *,
    timeout_seconds: float = 60.0,
    max_turns: int = 10,
    logger: structlog.stdlib.BoundLogger | None = None,
) -> ValidatedOutput[T]:
    """Run `agent` with `schema` enforced, retrying once if validation fails.

    The function clones the agent with `output_type=schema` so the caller's
    agent object is not mutated. If the schema is already set on the agent,
    it must equal `schema` — otherwise we'd silently override the caller's
    intent.

    Raises:
        OutputValidationError: After two attempts fail validation.
        AgentTaskError / AgentTaskTimeout: Propagated from `run_agent_task`.
        ValueError: If the agent has an incompatible output_type already set.
    """
    if agent.output_type is not None and agent.output_type is not schema:
        raise ValueError(
            f"agent {agent.name!r} already has output_type "
            f"{agent.output_type!r}, which conflicts with {schema!r}"
        )

    typed_agent = agent.clone(output_type=schema)
    log = (logger or _LOGGER).bind(agent_name=agent.name, schema=schema.__name__)

    # ----- attempt 1 -----
    first_error: str | None = None
    try:
        outcome = await run_agent_task(
            typed_agent,
            task_input,
            timeout_seconds=timeout_seconds,
            max_turns=max_turns,
            logger=logger,
        )
    except AgentTaskError as e:
        # The SDK raises ModelBehaviorError from inside the run; our wrapper
        # normalizes it to AgentTaskError. Distinguish schema-validation
        # failures from other run failures by inspecting the cause.
        if not isinstance(e.__cause__, ModelBehaviorError):
            raise
        first_error = str(e.__cause__)
        log.warning("output_validation_failed", error=first_error)
    else:
        value = _ensure_schema_type(outcome.result.final_output, schema)
        return ValidatedOutput(value=value, retried=False)

    assert first_error is not None  # narrowing

    # ----- attempt 2 -----
    retry_prompt = _build_retry_prompt(task_input, first_error)
    try:
        retry_outcome = await run_agent_task(
            typed_agent,
            retry_prompt,
            timeout_seconds=timeout_seconds,
            max_turns=max_turns,
            logger=logger,
        )
    except AgentTaskError as e:
        if isinstance(e.__cause__, ModelBehaviorError):
            log.error("output_validation_failed_after_retry", error=str(e.__cause__))
            raise OutputValidationError(
                "agent output failed schema validation twice",
                first_error=first_error,
                retry_error=str(e.__cause__),
            )
        # Retry failed for a non-validation reason (timeout, tool error).
        # Surface as a validation failure with the first error for context.
        raise OutputValidationError(
            "retry attempt failed before producing output",
            first_error=first_error,
        )

    value = _ensure_schema_type(retry_outcome.result.final_output, schema)
    log.info("output_validation_retry_succeeded")
    return ValidatedOutput(value=value, retried=True)


def _ensure_schema_type(final_output: Any, schema: type[T]) -> T:
    """Guarantee `final_output` is an instance of `schema`.

    With `output_type` set, the SDK normally returns a typed instance — but
    we re-validate defensively in case a future SDK version, a non-OpenAI
    provider, or a custom `AgentOutputSchemaBase` subclass returns a dict or
    string instead.
    """
    if isinstance(final_output, schema):
        return final_output
    if isinstance(final_output, dict):
        return schema.model_validate(final_output)
    if isinstance(final_output, str):
        return schema.model_validate_json(final_output)
    raise TypeError(
        f"unexpected final_output type {type(final_output).__name__}; "
        f"expected {schema.__name__}, dict, or str"
    )