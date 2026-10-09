"""A deterministic in-process `Model` for tests.

Use this when you want to exercise real `Runner.run` code paths (system prompt
plumbing, tool discovery, output assembly) without making OpenAI calls.

Usage:
    fake = FakeModel(responses=["Hi from the agent"])
    agent = build_local_agent(mcp_server=some_server, model=fake)
    outcome = await run_agent_task(agent, "test")
    assert "Hi" in outcome.result.final_output_as(str)

Each call to `get_response` pops one response from the FIFO queue. If the
queue is empty, it raises — that fail-loud behavior prevents accidental
re-use of the last response and forces tests to declare every model turn.
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator
from typing import Any

from agents import Model
from agents.items import ModelResponse
from agents.usage import Usage
from openai.types.responses import ResponseOutputMessage, ResponseOutputText


class FakeModelExhausted(RuntimeError):
    """Raised when the fake model is asked for a response it wasn't given."""


class FakeModel(Model):
    """Returns canned text responses one turn at a time."""

    def __init__(self, responses: list[str], *, usage: Usage | None = None) -> None:
        self._queue: deque[str] = deque(responses)
        # Token usage reported on every response; the SDK sums it per run.
        self._usage = usage
        self.calls: list[dict[str, Any]] = []  # introspection for tests

    async def get_response(
        self,
        system_instructions: str | None,
        input: Any,
        model_settings: Any,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Any],
        tracing: Any,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: Any = None,
    ) -> ModelResponse:
        if not self._queue:
            raise FakeModelExhausted(
                "FakeModel queue exhausted; declare more responses in the test."
            )
        text = self._queue.popleft()
        self.calls.append(
            {
                "system_instructions": system_instructions,
                "input": input,
                "tool_names": [getattr(t, "name", None) for t in tools],
                "output_schema": output_schema,
            }
        )
        message = ResponseOutputMessage(
            id=f"msg_{len(self.calls)}",
            type="message",
            role="assistant",
            status="completed",
            content=[
                ResponseOutputText(type="output_text", text=text, annotations=[])
            ],
        )
        return ModelResponse(
            output=[message],
            usage=self._usage or Usage(),
            response_id=f"resp_{len(self.calls)}",
        )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: Any,
        model_settings: Any,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Any],
        tracing: Any,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: Any = None,
    ) -> AsyncIterator[Any]:
        # Streaming isn't exercised by run_agent_task (it uses Runner.run, not
        # Runner.run_streamed), so we just make the method satisfy the ABC.
        raise NotImplementedError("FakeModel does not support streaming.")
        yield  # pragma: no cover  # makes this an async generator