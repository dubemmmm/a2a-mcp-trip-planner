"""End-to-end test for the A2A client modules.

Spins up a real uvicorn-hosted A2A server (using the Layer 4 scaffolding +
a FakeModel agent), then drives it through the full client surface:
discover → submit → await. If this passes, the Concierge in Layer 6 can
reliably call Travel and Local Experience over the wire.

No OpenAI calls; no real external APIs. Pure protocol-level verification.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
import pytest_asyncio
from a2a.client import ClientFactory
from a2a.types import AgentSkill
from agents import Agent

from a2a_servers.scaffolding import build_a2a_app, build_agent_card
from shared.a2a_client.discovery import (
    AgentDiscovery,
    
)
from shared.a2a_client.awaiting import await_task
from shared.a2a_client.submission import submit_task
from _helpers.fake_model import FakeModel

PORT = 54340


@asynccontextmanager
async def _serve(app: Any, port: int) -> AsyncIterator[str]:
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
async def server_url(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[str]:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ENVIRONMENT", "test")

    base_url = f"http://localhost:{PORT}"
    agent = Agent[None](
        name="echo-server",
        instructions="echo back",
        model=FakeModel(responses=["Live agent says hello."]),
    )
    app = build_a2a_app(
        agent,
        build_agent_card(
            name="echo-server",
            description="end-to-end test agent",
            url=base_url,
            skills=[
                AgentSkill(
                    id="echo",
                    name="echo",
                    description="echo input",
                    tags=["test"],
                ),
            ],
        ),
    )
    async with _serve(app, PORT) as url:
        yield url


class TestEndToEnd:
    async def test_discover_submit_await(self, server_url: str) -> None:
        async with AgentDiscovery() as discovery:
            card = await discovery.discover(server_url)
            assert card.name == "echo-server"

            client = ClientFactory().create(card)
            try:
                handle = await submit_task(
                    client,
                    card,
                    skill_id="echo",
                    payload="hello live agent",
                )
                result = await await_task(handle, timeout_seconds=10.0)
            finally:
                await client.close()

        assert result.is_success
        assert "Live agent says hello." in result.combined_text

    async def test_unknown_skill_caught_before_send(self, server_url: str) -> None:
        """The pre-flight check should fire before we ever hit the network."""
        from shared.a2a_client.submission import UnknownSkillError

        async with AgentDiscovery() as discovery:
            card = await discovery.discover(server_url)
            client = ClientFactory().create(card)
            try:
                with pytest.raises(UnknownSkillError):
                    await submit_task(
                        client,
                        card,
                        skill_id="nonexistent_skill",
                        payload="x",
                    )
            finally:
                await client.close()