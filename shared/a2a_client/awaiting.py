"""Wait for an A2A task to complete in either stream or poll mode.

Two modes:

  * **stream** (default): consume the open `StreamResponse` iterator from the
    `TaskHandle` directly. The remote agent pushes status and artifact events;
    we accumulate text and stop on the first terminal task state. This is the
    low-latency path the Concierge will use in Layer 6.

  * **poll**: use only for clients that can't hold a long-lived HTTP stream
    open (some serverless platforms, browsers behind proxies). The caller
    drains the events stream to exhaustion to grab the task id, then polls
    `get_task` until terminal.

Both modes return the same `TaskResult` shape so the orchestrator doesn't
care which transport ran underneath.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Literal

import structlog
from a2a.client import Client
from a2a.types import (
    GetTaskRequest,
    StreamResponse,
    TaskState,
)

from shared.agent_events import AgentEventQueue, emit_agent_event

from .submission import TaskHandle

_LOGGER = structlog.get_logger(__name__)

DEFAULT_AWAIT_TIMEOUT_SECONDS = 120.0
DEFAULT_POLL_INTERVAL_SECONDS = 1.0

WaitMode = Literal["stream", "poll"]

_TERMINAL_STATES = {
    TaskState.TASK_STATE_COMPLETED,
    TaskState.TASK_STATE_FAILED,
    TaskState.TASK_STATE_CANCELED,
    TaskState.TASK_STATE_REJECTED,
}


class TaskWaitError(RuntimeError):
    """Base class for failures while awaiting a task."""


class TaskWaitTimeout(TaskWaitError):
    """The task did not reach a terminal state within the timeout."""


class TaskFailedError(TaskWaitError):
    """The task reached FAILED, CANCELED, or REJECTED."""

    def __init__(self, message: str, *, state: int) -> None:
        super().__init__(message)
        self.state = state


@dataclass(frozen=True)
class TaskResult:
    """Unified result type for both stream and poll modes."""

    task_id: str
    state: int  # TaskState enum value
    artifacts_text: tuple[str, ...] = field(default_factory=tuple)
    duration_seconds: float = 0.0

    @property
    def is_success(self) -> bool:
        return self.state == TaskState.TASK_STATE_COMPLETED

    @property
    def combined_text(self) -> str:
        """All artifact text concatenated with newlines (convenience accessor)."""
        return "\n".join(self.artifacts_text)


def _payload_field(event: StreamResponse) -> str | None:
    """Return the name of the StreamResponse.payload oneof field that is set."""
    result: str | None = event.WhichOneof("payload")
    return result


def _extract_state(event: StreamResponse) -> int | None:
    """Return the task state from the actually-set payload field, or None."""
    which = _payload_field(event)
    if which == "task":
        return int(event.task.status.state)
    if which == "status_update":
        return int(event.status_update.status.state)
    return None


def _extract_task_id(event: StreamResponse) -> str | None:
    """Return a task id from the actually-set payload field, or None."""
    which = _payload_field(event)
    if which == "task":
        return event.task.id or None
    if which == "status_update":
        return event.status_update.task_id or None
    if which == "artifact_update":
        return event.artifact_update.task_id or None
    return None


def _extract_artifact_text(event: StreamResponse) -> tuple[str, ...]:
    """Extract any text Parts from artifact-bearing events."""
    out: list[str] = []
    which = _payload_field(event)
    if which == "task":
        for artifact in event.task.artifacts:
            for part in artifact.parts:
                if part.text:
                    out.append(part.text)
    elif which == "artifact_update":
        for part in event.artifact_update.artifact.parts:
            if part.text:
                out.append(part.text)
    return tuple(out)


async def _await_stream(
    handle: TaskHandle,
    *,
    timeout_seconds: float,
    event_queue: AgentEventQueue | None,
    event_agent_name: str | None,
) -> TaskResult:
    """Stream-mode wait: consume events until terminal state."""
    started_at = time.monotonic()
    log = _LOGGER.bind(
        agent=handle.agent_card.name,
        skill=handle.skill_id,
        message_id=handle.message_id,
        mode="stream",
    )

    collected_text: list[str] = []
    last_state: int | None = None
    task_id: str | None = None

    async def _drain() -> None:
        nonlocal last_state, task_id
        async for event in handle.events:
            tid = _extract_task_id(event)
            if tid is not None:
                task_id = tid
            state = _extract_state(event)
            if state is not None:
                last_state = state
                status = TaskState.Name(state).removeprefix("TASK_STATE_").lower()
                emit_agent_event(
                    event_queue,
                    kind="agent.status",
                    agent=event_agent_name or handle.agent_card.name,
                    status=status,
                    message=(
                        f"{(event_agent_name or handle.agent_card.name).title()} "
                        f"is {status.replace('_', ' ')}"
                    ),
                    data={"task_id": task_id, "skill_id": handle.skill_id},
                )
            artifact_text = _extract_artifact_text(event)
            for text in artifact_text:
                collected_text.append(text)
            if artifact_text:
                emit_agent_event(
                    event_queue,
                    kind="agent.artifact",
                    agent=event_agent_name or handle.agent_card.name,
                    status="working",
                    message=(
                        f"{(event_agent_name or handle.agent_card.name).title()} returned research"
                    ),
                    data={"task_id": task_id, "preview": "\n".join(artifact_text)[:240]},
                )
            if last_state in _TERMINAL_STATES:
                return

    try:
        await asyncio.wait_for(_drain(), timeout=timeout_seconds)
    except asyncio.TimeoutError as e:
        log.warning("a2a_task_wait_timeout", duration=time.monotonic() - started_at)
        raise TaskWaitTimeout(
            f"task {task_id or '<unknown>'} did not terminate within {timeout_seconds}s"
        ) from e

    duration = time.monotonic() - started_at
    if last_state is None or task_id is None:
        raise TaskWaitError("task stream closed without reporting a terminal state")

    if last_state not in _TERMINAL_STATES:
        # The stream ended but we never saw a terminal event. This is the
        # remote side hanging up mid-task — surface it as a wait error so
        # callers can decide whether to reconnect/retry (vs. an explicit
        # failure that they should not retry).
        raise TaskWaitError(
            f"task {task_id} stream closed before reaching terminal state "
            f"(last state: {last_state})"
        )

    if last_state != TaskState.TASK_STATE_COMPLETED:
        log.warning("a2a_task_failed", state=last_state, duration=duration)
        raise TaskFailedError(
            f"task {task_id} terminated with state {last_state}", state=last_state
        )

    log.info(
        "a2a_task_completed",
        duration=duration,
        artifact_count=len(collected_text),
    )
    return TaskResult(
        task_id=task_id,
        state=last_state,
        artifacts_text=tuple(collected_text),
        duration_seconds=duration,
    )


async def _await_poll(
    handle: TaskHandle,
    client: Client,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> TaskResult:
    """Poll-mode wait: drain the event stream to find the task id,
    then poll `get_task` until terminal."""
    started_at = time.monotonic()
    log = _LOGGER.bind(
        agent=handle.agent_card.name,
        skill=handle.skill_id,
        message_id=handle.message_id,
        mode="poll",
    )

    task_id: str | None = None
    # The send_message call has already opened the stream; drain it just to
    # discover the task id, then close it. In poll mode we don't rely on the
    # stream for completion, but we do need the task id from it.
    try:
        async with asyncio.timeout(timeout_seconds):
            async for event in handle.events:
                tid = _extract_task_id(event)
                if tid is not None:
                    task_id = tid
                    break
    except asyncio.TimeoutError as e:
        raise TaskWaitTimeout(f"never received a task id within {timeout_seconds}s") from e

    if task_id is None:
        raise TaskWaitError("event stream closed before yielding a task id")

    # Now poll. We don't re-time-out from scratch — the same budget covers
    # discovery + polling.
    deadline = started_at + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            log.warning("a2a_task_wait_timeout", duration=time.monotonic() - started_at)
            raise TaskWaitTimeout(f"task {task_id} did not terminate within {timeout_seconds}s")
        try:
            task = await asyncio.wait_for(
                client.get_task(GetTaskRequest(id=task_id)),
                timeout=min(remaining, 10.0),
            )
        except asyncio.TimeoutError:
            # Treat as a transient hiccup — fall through to retry on the next loop.
            log.debug("get_task_timed_out_polling")
            await asyncio.sleep(min(poll_interval_seconds, remaining))
            continue

        state = int(task.status.state)
        if state in _TERMINAL_STATES:
            duration = time.monotonic() - started_at
            artifact_text = tuple(p.text for a in task.artifacts for p in a.parts if p.text)
            if state != TaskState.TASK_STATE_COMPLETED:
                raise TaskFailedError(f"task {task_id} terminated with state {state}", state=state)
            log.info(
                "a2a_task_completed",
                duration=duration,
                artifact_count=len(artifact_text),
            )
            return TaskResult(
                task_id=task_id,
                state=state,
                artifacts_text=artifact_text,
                duration_seconds=duration,
            )

        await asyncio.sleep(min(poll_interval_seconds, max(remaining - 0.01, 0.01)))


async def await_task(
    handle: TaskHandle,
    *,
    mode: WaitMode = "stream",
    timeout_seconds: float = DEFAULT_AWAIT_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    client: Client | None = None,
    event_queue: AgentEventQueue | None = None,
    event_agent_name: str | None = None,
) -> TaskResult:
    """Wait for the task referenced by `handle` to reach a terminal state.

    Args:
        handle: The `TaskHandle` returned by `submit_task`.
        mode: "stream" (default) or "poll". Poll mode requires `client`.
        timeout_seconds: Hard wall-time cap.
        poll_interval_seconds: Delay between `get_task` calls in poll mode.
        client: The same `Client` used to submit, required for poll mode.

    Raises:
        TaskWaitTimeout: Wall-time budget exceeded.
        TaskFailedError: Task reached FAILED/CANCELED/REJECTED.
        TaskWaitError: Stream closed without a terminal state.
        ValueError: For invalid arguments.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")
    if mode == "stream":
        return await _await_stream(
            handle,
            timeout_seconds=timeout_seconds,
            event_queue=event_queue,
            event_agent_name=event_agent_name,
        )
    if mode == "poll":
        if client is None:
            raise ValueError("poll mode requires `client`")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be > 0")
        return await _await_poll(
            handle,
            client,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
    raise ValueError(f"unknown mode {mode!r}; expected 'stream' or 'poll'")
