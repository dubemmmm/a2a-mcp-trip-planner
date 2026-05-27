"""Live A2A round-trip test.

Spins up a real Starlette + uvicorn server hosting our A2A app, then drives
it with the official `a2a.client.create_client`. This is the "test that
catches schema bugs" for Layer 4: it exercises the real wire format, real
JSON-RPC dispatch, real Agent Card discovery, real streaming, and our
executor adapter — without making OpenAI calls (FakeModel substitutes the
LLM, no MCP servers are attached).

If this test passes, deploying under uvicorn just works.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
import pytest_asyncio
from a2a.client import ClientConfig, create_client
from a2a.types import Message, Part, Role, SendMessageRequest, TaskState
from agents import Agent

from a2a_servers.scaffolding import build_a2a_app, build_agent_card
from _helpers.fake_model import FakeModel


PORT = 54330


def _new_skill():
    from a2a.types import AgentSkill

    return AgentSkill(
        id="echo",
        name="echo",
        description="echo back the user input",
        tags=["test"],
    )


@asynccontextmanager
async def _serve(app: Any, port: int) -> AsyncIterator[str]:
    """Run the ASGI app on `port`; yield the base URL."""
    import uvicorn

    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="error", lifespan="on"
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
    task = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.02)
    else:
        task.cancel()
        raise RuntimeError("uvicorn did not start in time")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


@pytest_asyncio.fixture
async def live_agent_url(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[str]:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ENVIRONMENT", "test")

    fake = FakeModel(responses=["Hello from the A2A agent."])
    agent = Agent[None](name="echo", instructions="echo back", model=fake)
    base_url = f"http://localhost:{PORT}"
    app = build_a2a_app(
        agent,
        build_agent_card(
            name="echo-test",
            description="test agent",
            url=base_url,
            skills=[_new_skill()],
        ),
    )
    async with _serve(app, PORT) as url:
        yield url


# ---------- Agent Card discovery ----------

class TestAgentCardDiscovery:
    async def test_card_served_at_well_known(self, live_agent_url: str) -> None:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{live_agent_url}/.well-known/agent-card.json", timeout=5.0
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "echo-test"
        assert body["description"] == "test agent"
        assert any(s["id"] == "echo" for s in body["skills"])

    async def test_card_advertises_streaming(self, live_agent_url: str) -> None:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{live_agent_url}/.well-known/agent-card.json", timeout=5.0
            )
        body = resp.json()
        assert body["capabilities"]["streaming"] is True


# ---------- Send message + streaming ----------

class TestSendMessageStreaming:
    async def test_streams_to_completion(self, live_agent_url: str) -> None:
        """End-to-end: send_message → stream events → final task is COMPLETED."""
        client = await create_client(live_agent_url)
        try:
            req = SendMessageRequest(
                message=Message(
                    message_id="test-msg-1",
                    role=Role.ROLE_USER,
                    parts=[Part(text="say hello")],
                )
            )
            events: list[Any] = []
            async for event in client.send_message(req):
                events.append(event)
        finally:
            await client.close()

        # At minimum we should see a terminal event with COMPLETED.
        assert len(events) > 0, "expected at least one streamed event"
        # Find the final task state across events.
        seen_states: list[int] = []
        for e in events:
            for attr in ("task", "status_update", "artifact_update"):
                obj = getattr(e, attr, None)
                if obj is not None and hasattr(obj, "status"):
                    seen_states.append(obj.status.state)
                elif obj is not None and hasattr(obj, "state"):
                    seen_states.append(obj.state)
        assert TaskState.TASK_STATE_COMPLETED in seen_states or any(
            getattr(getattr(e, "task", None), "status", None)
            and e.task.status.state == TaskState.TASK_STATE_COMPLETED
            for e in events
        )

    async def test_response_text_in_final_artifact(
        self, live_agent_url: str
    ) -> None:
        """The agent's text response must reach the client as an artifact."""
        client = await create_client(live_agent_url)
        try:
            req = SendMessageRequest(
                message=Message(
                    message_id="test-msg-2",
                    role=Role.ROLE_USER,
                    parts=[Part(text="say hello")],
                )
            )
            collected_text: list[str] = []
            async for event in client.send_message(req):
                # Walk every Part on every artifact we encounter.
                task = getattr(event, "task", None)
                if task is not None:
                    for artifact in task.artifacts:
                        for part in artifact.parts:
                            if part.text:
                                collected_text.append(part.text)
                au = getattr(event, "artifact_update", None)
                if au is not None:
                    for part in au.artifact.parts:
                        if part.text:
                            collected_text.append(part.text)
        finally:
            await client.close()

        joined = " ".join(collected_text)
        assert "A2A agent" in joined or "Hello" in joined, (
            f"expected fake model response in artifacts; got: {collected_text!r}"
        )