"""Tests for `shared.a2a_client.awaiting`."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from a2a.types import (
    AgentCard,
    AgentSkill,
    Artifact,
    GetTaskRequest,
    Message,
    Part,
    Role,
    StreamResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatus,
    TaskStatusUpdateEvent,
    TaskState,
)

from shared.a2a_client.awaiting import (
    TaskFailedError,
    TaskWaitError,
    TaskWaitTimeout,
    await_task,
)
from shared.a2a_client.submission import TaskHandle
from shared.agent_events import AgentEventQueue


def _card() -> AgentCard:
    return AgentCard(
        name="test-agent",
        description="x",
        version="0.1.0",
        skills=[AgentSkill(id="search_flights", name="x", description="x", tags=[])],
    )


def _stream(events: list[StreamResponse]) -> AsyncIterator[StreamResponse]:
    async def _gen() -> AsyncIterator[StreamResponse]:
        for e in events:
            yield e

    return _gen()


def _task_event(task_id: str, state: int) -> StreamResponse:
    return StreamResponse(
        task=Task(
            id=task_id,
            context_id="ctx",
            status=TaskStatus(state=state),
        )
    )


def _status_update(task_id: str, state: int) -> StreamResponse:
    return StreamResponse(
        status_update=TaskStatusUpdateEvent(
            task_id=task_id,
            context_id="ctx",
            status=TaskStatus(state=state),
        )
    )


def _artifact_update(task_id: str, text: str) -> StreamResponse:
    return StreamResponse(
        artifact_update=TaskArtifactUpdateEvent(
            task_id=task_id,
            context_id="ctx",
            artifact=Artifact(
                artifact_id="art1",
                parts=[Part(text=text)],
            ),
        )
    )


def _handle(events: list[StreamResponse]) -> TaskHandle:
    return TaskHandle(
        agent_card=_card(),
        skill_id="search_flights",
        message_id="msg-1",
        events=_stream(events),
    )


# ---------- stream mode happy path ----------


class TestStreamSuccess:
    async def test_completes_with_artifact(self) -> None:
        events = [
            _task_event("t1", TaskState.TASK_STATE_SUBMITTED),
            _status_update("t1", TaskState.TASK_STATE_WORKING),
            _artifact_update("t1", "Hello from agent."),
            _status_update("t1", TaskState.TASK_STATE_COMPLETED),
        ]
        result = await await_task(_handle(events), timeout_seconds=2.0)
        assert result.is_success
        assert result.task_id == "t1"
        assert result.state == TaskState.TASK_STATE_COMPLETED
        assert "Hello from agent." in result.combined_text
        assert result.duration_seconds >= 0

    async def test_terminal_in_task_event(self) -> None:
        """Sometimes the whole flow arrives as one Task event with terminal status."""
        task = Task(
            id="t1",
            context_id="ctx",
            status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
            artifacts=[Artifact(artifact_id="a1", parts=[Part(text="done")])],
        )
        events = [StreamResponse(task=task)]
        result = await await_task(_handle(events), timeout_seconds=2.0)
        assert result.is_success
        assert "done" in result.combined_text

    async def test_multiple_artifacts_concatenated(self) -> None:
        events = [
            _task_event("t1", TaskState.TASK_STATE_SUBMITTED),
            _artifact_update("t1", "part one"),
            _artifact_update("t1", "part two"),
            _status_update("t1", TaskState.TASK_STATE_COMPLETED),
        ]
        result = await await_task(_handle(events), timeout_seconds=2.0)
        assert result.artifacts_text == ("part one", "part two")
        assert "part one\npart two" in result.combined_text

    async def test_forwards_live_status_and_artifact_events(self) -> None:
        events = [
            _task_event("t1", TaskState.TASK_STATE_SUBMITTED),
            _status_update("t1", TaskState.TASK_STATE_WORKING),
            _artifact_update("t1", "Live flight research"),
            _status_update("t1", TaskState.TASK_STATE_COMPLETED),
        ]
        queue: AgentEventQueue = asyncio.Queue()

        await await_task(
            _handle(events),
            timeout_seconds=2.0,
            event_queue=queue,
            event_agent_name="travel",
        )

        published = []
        while not queue.empty():
            published.append(queue.get_nowait())
        statuses = [event.status for event in published if event.kind == "agent.status"]
        assert statuses == ["submitted", "working", "completed"]
        artifact = next(event for event in published if event.kind == "agent.artifact")
        assert artifact.agent == "travel"
        assert artifact.data["preview"] == "Live flight research"


# ---------- stream mode failure paths ----------


class TestStreamFailure:
    async def test_failed_state_raises(self) -> None:
        events = [
            _task_event("t1", TaskState.TASK_STATE_SUBMITTED),
            _status_update("t1", TaskState.TASK_STATE_WORKING),
            _status_update("t1", TaskState.TASK_STATE_FAILED),
        ]
        with pytest.raises(TaskFailedError) as exc:
            await await_task(_handle(events), timeout_seconds=2.0)
        assert exc.value.state == TaskState.TASK_STATE_FAILED

    async def test_canceled_state_raises(self) -> None:
        events = [
            _task_event("t1", TaskState.TASK_STATE_SUBMITTED),
            _status_update("t1", TaskState.TASK_STATE_CANCELED),
        ]
        with pytest.raises(TaskFailedError) as exc:
            await await_task(_handle(events), timeout_seconds=2.0)
        assert exc.value.state == TaskState.TASK_STATE_CANCELED

    async def test_stream_closes_without_terminal(self) -> None:
        events = [
            _task_event("t1", TaskState.TASK_STATE_SUBMITTED),
            _status_update("t1", TaskState.TASK_STATE_WORKING),
            # No terminal event — stream ends here.
        ]
        with pytest.raises(TaskWaitError, match="stream closed"):
            await await_task(_handle(events), timeout_seconds=2.0)

    async def test_timeout(self) -> None:
        """Use an explicitly-slow async generator to force timeout."""

        async def _slow_stream() -> AsyncIterator[StreamResponse]:
            yield _task_event("t1", TaskState.TASK_STATE_SUBMITTED)
            await asyncio.sleep(5)  # exceeds our 0.05s budget
            yield _status_update("t1", TaskState.TASK_STATE_COMPLETED)

        handle = TaskHandle(
            agent_card=_card(),
            skill_id="search_flights",
            message_id="m",
            events=_slow_stream(),
        )
        with pytest.raises(TaskWaitTimeout):
            await await_task(handle, timeout_seconds=0.05)


# ---------- argument validation ----------


class TestArgValidation:
    async def test_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match="unknown mode"):
            await await_task(_handle([]), mode="hybrid")  # type: ignore[arg-type]

    async def test_poll_mode_requires_client(self) -> None:
        with pytest.raises(ValueError, match="poll mode requires"):
            await await_task(_handle([]), mode="poll")

    async def test_negative_timeout(self) -> None:
        with pytest.raises(ValueError):
            await await_task(_handle([]), timeout_seconds=0)


# ---------- poll mode ----------


class _PollClient:
    """Fake a2a-sdk Client that returns a scripted sequence of Task states."""

    def __init__(self, states: list[int], task_id: str = "t1") -> None:
        self._states = list(states)
        self._task_id = task_id
        self.get_task_calls = 0

    async def get_task(self, request: GetTaskRequest, *, context: Any = None) -> Task:
        self.get_task_calls += 1
        state = self._states.pop(0) if self._states else TaskState.TASK_STATE_COMPLETED
        return Task(
            id=self._task_id,
            context_id="ctx",
            status=TaskStatus(state=state),
            artifacts=[Artifact(artifact_id="a1", parts=[Part(text="poll result")])]
            if state == TaskState.TASK_STATE_COMPLETED
            else [],
        )


class TestPollMode:
    async def test_polls_until_completed(self) -> None:
        # Initial stream yields just the Task event so we can extract the id.
        handle = _handle([_task_event("t1", TaskState.TASK_STATE_WORKING)])
        client = _PollClient(
            states=[
                TaskState.TASK_STATE_WORKING,
                TaskState.TASK_STATE_WORKING,
                TaskState.TASK_STATE_COMPLETED,
            ]
        )
        result = await await_task(
            handle,
            mode="poll",
            client=client,  # type: ignore[arg-type]
            timeout_seconds=5.0,
            poll_interval_seconds=0.01,
        )
        assert result.is_success
        assert result.task_id == "t1"
        assert "poll result" in result.combined_text
        assert client.get_task_calls == 3

    async def test_polls_until_failed(self) -> None:
        handle = _handle([_task_event("t1", TaskState.TASK_STATE_SUBMITTED)])
        client = _PollClient(
            states=[
                TaskState.TASK_STATE_WORKING,
                TaskState.TASK_STATE_FAILED,
            ]
        )
        with pytest.raises(TaskFailedError) as exc:
            await await_task(
                handle,
                mode="poll",
                client=client,  # type: ignore[arg-type]
                timeout_seconds=5.0,
                poll_interval_seconds=0.01,
            )
        assert exc.value.state == TaskState.TASK_STATE_FAILED

    async def test_poll_timeout(self) -> None:
        handle = _handle([_task_event("t1", TaskState.TASK_STATE_WORKING)])
        # Keep returning WORKING forever.
        client = _PollClient(states=[TaskState.TASK_STATE_WORKING] * 1000)
        with pytest.raises(TaskWaitTimeout):
            await await_task(
                handle,
                mode="poll",
                client=client,  # type: ignore[arg-type]
                timeout_seconds=0.1,
                poll_interval_seconds=0.02,
            )

    async def test_poll_no_task_id_in_stream(self) -> None:
        """If the stream closes without ever yielding a task id, surface clearly."""
        handle = _handle([])  # empty stream
        client = _PollClient(states=[])
        with pytest.raises(TaskWaitError, match="task id"):
            await await_task(
                handle,
                mode="poll",
                client=client,  # type: ignore[arg-type]
                timeout_seconds=1.0,
                poll_interval_seconds=0.01,
            )
