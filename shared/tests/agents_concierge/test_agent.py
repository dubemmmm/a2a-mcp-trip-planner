"""Tests for the Concierge agent builder."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from agents.mcp import MCPServerStdio, MCPServerStreamableHttp

from agents_concierge.agent import (
    CONCIERGE_AGENT_INSTRUCTIONS,
    CONCIERGE_AGENT_NAME,
    ORCHESTRATING_CONCIERGE_INSTRUCTIONS,
    build_concierge_agent,
    build_concierge_mcp_server,
    build_orchestrating_concierge,
)
from agents_concierge.delegation import build_delegation_tools
from shared.a2a_client import AgentDiscovery


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ENVIRONMENT", "test")


class TestBuildConciergeMcpServer:
    def test_stdio_default(self, env: None) -> None:
        server = build_concierge_mcp_server()
        assert isinstance(server, MCPServerStdio)

    def test_http_when_url_given(self, env: None) -> None:
        server = build_concierge_mcp_server(mcp_url="http://localhost:8003/mcp")
        assert isinstance(server, MCPServerStreamableHttp)


class TestBuildConciergeAgent:
    def test_uses_orchestrator_model_by_default(self, env: None) -> None:
        agent = build_concierge_agent()
        assert agent.name == CONCIERGE_AGENT_NAME
        # Concierge uses the heavier model — gpt-5 per config defaults.
        assert agent.model == "gpt-5.5"

    def test_model_override(self, env: None) -> None:
        agent = build_concierge_agent(model="gpt-5.4-mini")
        assert agent.model == "gpt-5.4-mini"

    def test_attaches_mcp_server(self, env: None) -> None:
        agent = build_concierge_agent()
        assert len(agent.mcp_servers) == 1

    def test_no_function_tools_attached(self, env: None) -> None:
        """The standalone variant has only MCP tools, no function tools."""
        agent = build_concierge_agent()
        assert agent.tools == []


class TestBuildOrchestratingConcierge:
    def _tools(self) -> list:
        discovery = MagicMock(spec=AgentDiscovery)
        return build_delegation_tools(
            travel_url="http://travel:8002",
            local_url="http://local:8001",
            discovery=discovery,
        )

    def test_uses_orchestrator_model_by_default(self, env: None) -> None:
        agent = build_orchestrating_concierge(self._tools())
        assert agent.name == CONCIERGE_AGENT_NAME
        assert agent.model == "gpt-5.5"

    def test_attaches_both_mcp_and_function_tools(self, env: None) -> None:
        tools = self._tools()
        agent = build_orchestrating_concierge(tools)
        assert len(agent.mcp_servers) == 1
        # Both delegation tools must be present.
        assert len(agent.tools) == 2
        tool_names = {t.name for t in agent.tools}
        assert tool_names == {"call_travel_agent", "call_local_experience_agent"}

    def test_uses_orchestrating_instructions(self, env: None) -> None:
        agent = build_orchestrating_concierge(self._tools())
        assert agent.instructions == ORCHESTRATING_CONCIERGE_INSTRUCTIONS

    def test_model_override(self, env: None) -> None:
        agent = build_orchestrating_concierge(self._tools(), model="gpt-5-mini")
        assert agent.model == "gpt-5-mini"


class TestSystemPromptContent:
    @pytest.mark.parametrize(
        "tool_name",
        [
            "get_user_preferences",
            "save_user_preferences",
            "save_itinerary",
            "list_itineraries",
        ],
    )
    def test_standalone_mentions_each_tool(self, tool_name: str) -> None:
        assert tool_name in CONCIERGE_AGENT_INSTRUCTIONS

    def test_standalone_mentions_default_user_handling(self) -> None:
        assert "default" in CONCIERGE_AGENT_INSTRUCTIONS.lower()


class TestOrchestratingPromptContent:
    @pytest.mark.parametrize(
        "tool_name",
        [
            # MCP tools
            "get_user_preferences",
            "save_itinerary",
            # Delegation tools
            "call_travel_agent",
            "call_local_experience_agent",
        ],
    )
    def test_mentions_each_tool(self, tool_name: str) -> None:
        assert tool_name in ORCHESTRATING_CONCIERGE_INSTRUCTIONS

    def test_mentions_parallel_delegation(self) -> None:
        """The prompt should encourage calling both specialists in one turn."""
        # We don't lock to exact phrasing, just check key concepts.
        text = ORCHESTRATING_CONCIERGE_INSTRUCTIONS.lower()
        assert "specialist" in text
        assert "both" in text or "parallel" in text or "same turn" in text

    def test_mentions_failure_handling(self) -> None:
        """The prompt must tell the LLM what to do when a specialist fails."""
        assert "fail" in ORCHESTRATING_CONCIERGE_INSTRUCTIONS.lower()