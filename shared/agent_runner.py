"""Agent task execution wrapper.

`run_agent_task` is the only entry point agents use to invoke `Runner.run`.
It wraps the SDK with three guarantees the agent shouldn't have to provide:

  * **Logging**: a structured log line on start, success, and failure with
    the agent name, the input, the duration, and the failure mode.
  * **Timeout**: a hard cap on wall time. Tool calls that hang forever (a
    flaky external API, a runaway loop) surface as `AgentTaskTimeout`.
  * **Error normalization**: SDK exceptions are caught and re-raised as
    `AgentTaskError` so callers have one type to catch.

The wrapper is intentionally thin — no retries, no fallbacks. Those belong
in higher layers (Layer 7 reliability work), and forcing them through here
would couple every agent to a single retry policy.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import structlog
from agents import Agent, RunResult, RunResultStreaming, Runner
from agents.items import ToolCallItem, ToolCallOutputItem

from .agent_events import AgentEventQueue, emit_agent_event

_LOGGER = structlog.get_logger(__name__)

# A run that takes longer than this almost always indicates a stuck tool call
# or runaway agent loop, both of which we want to surface as a failure.
DEFAULT_TIMEOUT_SECONDS = 60.0


class AgentTaskError(RuntimeError):
    """Normalized failure from `run_agent_task`.

    Wraps the underlying exception so callers only catch this one type.
    """

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.__cause__ = cause


class AgentTaskTimeout(AgentTaskError):
    """Raised when a run exceeds its time budget."""


@dataclass(frozen=True)
class RunUsage:
    """What one run consumed: model tokens and tool calls by tool name.

    This is the meter the agent economy prices from. Tool calls are counted
    here, in the agent's process, rather than inside the MCP servers: those
    run as stdio subprocesses, so a counter there would never reach the code
    that charges for the task.
    """

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: Mapping[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tool_calls": dict(self.tool_calls),
        }


@dataclass(frozen=True)
class AgentTaskOutcome:
    """A successful run plus the metadata we logged about it."""

    result: RunResult | RunResultStreaming
    duration_seconds: float
    usage: RunUsage = field(default_factory=RunUsage)


def usage_of(result: object) -> RunUsage:
    """Read token usage and tool calls off an SDK run result.

    Tolerates results without usage (test doubles, partial results) by
    reporting zero rather than failing the run it describes.
    """
    wrapper = getattr(result, "context_wrapper", None)
    sdk_usage = getattr(wrapper, "usage", None)
    tools: Counter[str] = Counter(
        item.tool_name or "tool"
        for item in getattr(result, "new_items", ())
        if isinstance(item, ToolCallItem)
    )
    return RunUsage(
        requests=int(getattr(sdk_usage, "requests", 0) or 0),
        input_tokens=int(getattr(sdk_usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(sdk_usage, "output_tokens", 0) or 0),
        tool_calls=dict(tools),
    )


async def run_agent_task(
    agent: Agent[Any],
    task_input: str,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_turns: int = 10,
    logger: structlog.stdlib.BoundLogger | None = None,
    event_queue: AgentEventQueue | None = None,
) -> AgentTaskOutcome:
    """Run `agent` against `task_input` with logging, timeout, and error normalization.

    Args:
        agent: The Agent to run.
        task_input: The user-facing prompt.
        timeout_seconds: Hard wall-time cap. Defaults to 60s.
        max_turns: Cap on agent loop iterations (forwarded to Runner.run).
        logger: Optional pre-bound logger; falls back to module logger.
        event_queue: Optional in-process queue for live execution events. When
            supplied, the SDK streaming runner is used so tool calls and tool
            outputs can be observed as they happen.

    Returns:
        AgentTaskOutcome containing the SDK's RunResult and duration.

    Raises:
        AgentTaskTimeout: If the run exceeds `timeout_seconds`.
        AgentTaskError: For any other SDK or tool-layer failure.
        ValueError: For invalid arguments.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if max_turns < 1:
        raise ValueError("max_turns must be >= 1")

    log = (logger or _LOGGER).bind(
        agent_name=agent.name,
        timeout_seconds=timeout_seconds,
        max_turns=max_turns,
    )
    log.info("agent_task_start", input_length=len(task_input))
    emit_agent_event(
        event_queue,
        kind="agent.status",
        agent=agent.name,
        status="working",
        message=f"{agent.name.title()} started working",
    )
    started_at = time.monotonic()

    try:
        if event_queue is None:
            result: RunResult | RunResultStreaming = await asyncio.wait_for(
                Runner.run(agent, task_input, max_turns=max_turns),
                timeout=timeout_seconds,
            )
        else:
            result = await asyncio.wait_for(
                _run_streamed(
                    agent,
                    task_input,
                    max_turns=max_turns,
                    event_queue=event_queue,
                ),
                timeout=timeout_seconds,
            )
    except asyncio.TimeoutError as e:
        duration = time.monotonic() - started_at
        log.warning("agent_task_timeout", duration_seconds=duration)
        emit_agent_event(
            event_queue,
            kind="agent.status",
            agent=agent.name,
            status="failed",
            message=f"{agent.name.title()} timed out",
        )
        raise AgentTaskTimeout(f"agent {agent.name!r} exceeded {timeout_seconds}s") from e
    except AgentTaskError:
        # Already normalized (e.g. by a nested run); don't double-wrap.
        raise
    except Exception as e:
        duration = time.monotonic() - started_at
        log.exception("agent_task_error", duration_seconds=duration)
        emit_agent_event(
            event_queue,
            kind="agent.status",
            agent=agent.name,
            status="failed",
            message=f"{agent.name.title()} could not complete the task",
            data={"error": str(e)},
        )
        raise AgentTaskError(
            f"agent {agent.name!r} failed: {type(e).__name__}: {e}", cause=e
        ) from e

    duration = time.monotonic() - started_at
    usage = usage_of(result)
    log.info(
        "agent_task_success",
        duration_seconds=duration,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        tool_calls=dict(usage.tool_calls),
    )
    emit_agent_event(
        event_queue,
        kind="agent.status",
        agent=agent.name,
        status="completed",
        message=f"{agent.name.title()} finished the plan",
        data={"duration_seconds": duration},
    )
    return AgentTaskOutcome(result=result, duration_seconds=duration, usage=usage)


async def _run_streamed(
    agent: Agent[Any],
    task_input: str,
    *,
    max_turns: int,
    event_queue: AgentEventQueue,
) -> RunResultStreaming:
    """Drain semantic SDK events while retaining the final streaming result."""
    result = Runner.run_streamed(agent, task_input, max_turns=max_turns)
    tool_names: dict[str, str] = {}

    async for event in result.stream_events():
        if event.type != "run_item_stream_event":
            continue
        if event.name == "tool_called" and isinstance(event.item, ToolCallItem):
            tool_name = event.item.tool_name or "tool"
            if event.item.call_id:
                tool_names[event.item.call_id] = tool_name
            emit_agent_event(
                event_queue,
                kind="tool.called",
                agent=agent.name,
                status="working",
                message=f"Calling {tool_name.replace('_', ' ')}",
                data={"tool_name": tool_name, "call_id": event.item.call_id},
            )
        elif event.name == "tool_output" and isinstance(event.item, ToolCallOutputItem):
            call_id = event.item.call_id
            tool_name = tool_names.get(call_id or "", "tool")
            output = str(event.item.output)
            emit_agent_event(
                event_queue,
                kind="tool.completed",
                agent=agent.name,
                status="working",
                message=f"{tool_name.replace('_', ' ').title()} returned",
                data={
                    "tool_name": tool_name,
                    "call_id": call_id,
                    "preview": output[:240],
                },
            )

    return result
