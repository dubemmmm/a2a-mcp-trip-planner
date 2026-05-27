"""End-to-end Layer 6 test: real OpenAI + real A2A.

Spins up two A2A servers (Travel + Local Experience), wires them into a
real Concierge agent, and runs a trip-planning request through real OpenAI.
The specialists use FakeModel responses (deterministic, no API costs); the
Concierge's LLM is the real model — we want to verify it actually:

  1. Decides to call both specialists (not just one)
  2. Composes reasonable sub-prompts
  3. Synthesizes a coherent response from their outputs

This test is **gated on a real OPENAI_API_KEY**. Without one, it skips with
a clear reason. We don't mock OpenAI here — the whole point is to verify
the full agent loop works against the live model.

Cost: ~$0.005-$0.02 per run depending on the model.

Run:
    pytest tests/agents_concierge/test_layer6_live.py -m live
or skip cleanly with:
    pytest -m 'not live'
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
import pytest_asyncio
from a2a.types import AgentSkill
from agents import Agent

from a2a_servers.scaffolding import build_a2a_app, build_agent_card
from agents_concierge.agent import build_orchestrating_concierge
from agents_concierge.delegation import build_delegation_tools
from agents_concierge.store.sqlite_store import ConciergeStore
from mcp_servers.concierge import build_server as build_concierge_mcp_server
from mcp_servers.scaffolding import serve_stdio  # noqa: F401  (imported for typing)
from shared.a2a_client import AgentDiscovery
from shared.agent_runner import run_agent_task
from _helpers.fake_model import FakeModel


# Live tests are opt-in via the `live` marker AND a real key.
pytestmark = pytest.mark.live


_TRAVEL_PORT = 54350
_LOCAL_PORT = 54351


_TRAVEL_FAKE_RESPONSE = (
    "Found 2 flight options for ATL → LIS departing 2026-06-15:\n"
    "1. TAP Portugal TP202 — $612, 1 stop, departs 18:00 arrives 07:30+1\n"
    "2. Delta DL128 — $748, nonstop, departs 21:30 arrives 11:00+1\n"
    "\n"
    "Found 2 hotel options for Lisbon 2026-06-15 to 2026-06-19:\n"
    "- Hotel Alfama Stay — $120/night, 4.5 stars\n"
    "- Avenida Liberdade Hotel — $145/night, 4.6 stars\n"
)

_LOCAL_FAKE_RESPONSE = (
    "Top picks for Lisbon (food + history):\n"
    "1. Time Out Market — iconic food hall with top chefs (food)\n"
    "2. Jerónimos Monastery — 16th-century Manueline landmark (history)\n"
    "3. Castelo de São Jorge — hilltop Moorish castle with panoramic views (history)\n"
    "\n"
    "Weather forecast for Lisbon 2026-06-15 to 2026-06-19:\n"
    "- 2026-06-15: 24°C high, clear sky, 0% rain\n"
    "- 2026-06-16: 26°C high, few clouds, 0% rain\n"
    "- 2026-06-17: 25°C high, clear sky, 0% rain\n"
    "- 2026-06-18: 27°C high, clear sky, 0% rain\n"
)


@asynccontextmanager
async def _serve(app: Any, port: int) -> AsyncIterator[str]:
    """Run an ASGI app on `port`; yield the base URL."""
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


def _fake_specialist_app(
    name: str, port: int, skill_id: str, response: str
) -> Any:
    """Build an A2A app fronted by a FakeModel specialist."""
    agent = Agent[None](
        name=name,
        instructions="Reply with helpful trip-planning information.",
        # FakeModel returns the canned response regardless of how many times
        # the agent is invoked (we provide a generous queue).
        model=FakeModel(responses=[response] * 5),
    )
    card = build_agent_card(
        name=name,
        description=f"test {name} specialist",
        url=f"http://127.0.0.1:{port}",
        skills=[
            AgentSkill(
                id=skill_id, name=skill_id, description=skill_id, tags=["test"]
            )
        ],
    )
    return build_a2a_app(agent, card)


@pytest_asyncio.fixture
async def specialists() -> AsyncIterator[tuple[str, str]]:
    """Run Travel + Local Experience A2A servers and yield their base URLs."""
    travel_app = _fake_specialist_app(
        "travel", _TRAVEL_PORT, "search_flights", _TRAVEL_FAKE_RESPONSE
    )
    local_app = _fake_specialist_app(
        "local-experience", _LOCAL_PORT, "find_attractions", _LOCAL_FAKE_RESPONSE
    )
    async with _serve(travel_app, _TRAVEL_PORT) as travel_url:
        async with _serve(local_app, _LOCAL_PORT) as local_url:
            yield travel_url, local_url


def _real_openai_key_present() -> bool:
    """True if a real-looking OpenAI key is in the environment."""
    key = os.environ.get("OPENAI_API_KEY", "")
    # Treat obvious test sentinels as 'not real'.
    return bool(key) and not key.startswith("sk-test")


@pytest.mark.skipif(
    not _real_openai_key_present(),
    reason="No real OPENAI_API_KEY in env; live test skipped.",
)
class TestLayer6LiveEndToEnd:
    async def test_concierge_delegates_to_both_specialists(
        self, specialists: tuple[str, str], tmp_path: Any
    ) -> None:
        travel_url, local_url = specialists

        # Real Concierge: real MCP server (SQLite), real delegation tools,
        # real OpenAI model.
        store = await ConciergeStore.open(tmp_path / "concierge.db")
        mcp_server = build_concierge_mcp_server(store)
        try:
            discovery = AgentDiscovery()
            try:
                tools = build_delegation_tools(
                    travel_url=travel_url,
                    local_url=local_url,
                    discovery=discovery,
                    timeout_seconds=60.0,
                )

                # We need to attach the MCP server in-process. Use the same
                # build_orchestrating_concierge pathway but with the test
                # store directly. The MCP layer normally spawns a subprocess;
                # in this test we'd rather inject the store directly. But
                # since the SDK requires an MCPServer object, we use the
                # in-process server build path:
                from agents.mcp import MCPServer  # noqa: F401

                # The standard Concierge MCP server in mcp_servers/concierge.py
                # spawns a subprocess that needs OPENAI_API_KEY etc. For this
                # live test we use the stdio MCP server path via the agent
                # builder, which the SDK manages.
                agent = build_orchestrating_concierge(tools)

                async with agent.mcp_servers[0]:
                    outcome = await run_agent_task(
                        agent,
                        "I want to visit Lisbon from June 15-19, 2026, departing "
                        "from ATL. Budget around $1500. I like food and history. "
                        "Plan the whole trip — flights, hotel, things to do, "
                        "and tell me the weather.",
                        timeout_seconds=180.0,
                        max_turns=10,
                    )
            finally:
                await discovery.aclose()
        finally:
            await store.close()

        text = str(outcome.result.final_output_as(str))
        text_lower = text.lower()

        # The Concierge must have synthesized info from BOTH specialists.
        # We check for distinctive content from each fake response.
        # (We don't lock to exact phrasing — the LLM rewrites freely — but
        # specific names/numbers should survive synthesis.)
        travel_signals = ("TAP" in text or "612" in text or "Alfama" in text)
        local_signals = (
            "Jerónimos" in text
            or "Time Out" in text
            or "Castelo" in text
            or "monastery" in text_lower
        )
        assert travel_signals, (
            f"expected travel signals (TAP/612/Alfama) in Concierge response; "
            f"got:\n{text}"
        )
        assert local_signals, (
            f"expected local signals (Jerónimos/Time Out/Castelo) in "
            f"Concierge response; got:\n{text}"
        )