"""Tests for a2a_servers.scaffolding."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from a2a.types import (
    AgentCapabilities,
    AgentSkill,
    Part,
    Role,
    TaskState,
)
from agents import Agent

from a2a_servers.scaffolding import (
    DEFAULT_PROTOCOL_VERSION,
    OpenAIAgentExecutor,
    build_a2a_app,
    build_agent_card,
    task_is_terminal,
    text_part,
)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ENVIRONMENT", "test")


def _skill(id_: str = "thing") -> AgentSkill:
    return AgentSkill(
        id=id_, name=f"do {id_}", description=f"does {id_}", tags=[id_]
    )


# ---------- build_agent_card ----------

class TestBuildAgentCard:
    def test_basic_card(self, env: None) -> None:
        card = build_agent_card(
            name="travel",
            description="finds flights",
            url="http://localhost:8002",
            skills=[_skill("search_flights")],
        )
        assert card.name == "travel"
        assert card.description == "finds flights"
        assert card.version == "0.1.0"
        assert len(card.skills) == 1
        assert card.skills[0].id == "search_flights"

    def test_streaming_capability_default_true(self, env: None) -> None:
        card = build_agent_card(
            name="x", description="x", url="http://x", skills=[_skill()]
        )
        assert card.capabilities.streaming is True

    def test_streaming_can_be_disabled(self, env: None) -> None:
        card = build_agent_card(
            name="x",
            description="x",
            url="http://x",
            skills=[_skill()],
            streaming=False,
        )
        assert card.capabilities.streaming is False

    def test_interface_lists_jsonrpc(self, env: None) -> None:
        card = build_agent_card(
            name="x",
            description="x",
            url="http://localhost:9999",
            skills=[_skill()],
        )
        assert len(card.supported_interfaces) == 1
        iface = card.supported_interfaces[0]
        assert iface.url == "http://localhost:9999"
        assert iface.protocol_binding == "JSONRPC"
        assert iface.protocol_version == DEFAULT_PROTOCOL_VERSION

    def test_default_modes_are_text(self, env: None) -> None:
        card = build_agent_card(
            name="x", description="x", url="http://x", skills=[_skill()]
        )
        assert "text/plain" in card.default_input_modes
        assert "text/plain" in card.default_output_modes


# ---------- helpers ----------

class TestTaskIsTerminal:
    @pytest.mark.parametrize(
        "state,terminal",
        [
            (TaskState.TASK_STATE_COMPLETED, True),
            (TaskState.TASK_STATE_FAILED, True),
            (TaskState.TASK_STATE_CANCELED, True),
            (TaskState.TASK_STATE_REJECTED, True),
            (TaskState.TASK_STATE_SUBMITTED, False),
            (TaskState.TASK_STATE_WORKING, False),
            (TaskState.TASK_STATE_INPUT_REQUIRED, False),
            (TaskState.TASK_STATE_AUTH_REQUIRED, False),
        ],
    )
    def test_terminal_states(
        self, state: int, terminal: bool
    ) -> None:
        assert task_is_terminal(state) is terminal


class TestTextPart:
    def test_creates_text_part(self) -> None:
        p = text_part("hello")
        assert p.text == "hello"


# ---------- build_a2a_app ----------

class TestBuildA2AApp:
    def test_returns_starlette_app(self, env: None) -> None:
        from starlette.applications import Starlette

        agent = Agent[None](name="x", instructions="be helpful", model="gpt-5-mini")
        card = build_agent_card(
            name="x", description="x", url="http://x", skills=[_skill()]
        )
        app = build_a2a_app(agent, card)
        assert isinstance(app, Starlette)

    def test_routes_include_agent_card_and_jsonrpc(self, env: None) -> None:
        agent = Agent[None](name="x", instructions="be helpful", model="gpt-5-mini")
        card = build_agent_card(
            name="x", description="x", url="http://x", skills=[_skill()]
        )
        app = build_a2a_app(agent, card)
        paths = {getattr(r, "path", None) for r in app.routes}
        assert "/.well-known/agent-card.json" in paths
        # The JSON-RPC dispatcher is mounted at "/"
        assert "/" in paths