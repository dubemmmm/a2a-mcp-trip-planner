"""Tests for the Travel agent builder."""

from __future__ import annotations

import pytest
from agents.mcp import MCPServerStdio, MCPServerStreamableHttp

from agents_travel.agent import (
    TRAVEL_AGENT_INSTRUCTIONS,
    TRAVEL_AGENT_NAME,
    build_travel_agent,
    build_travel_mcp_server,
)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ENVIRONMENT", "test")


class TestBuildTravelMcpServer:
    def test_stdio_default(self, env: None) -> None:
        server = build_travel_mcp_server()
        assert isinstance(server, MCPServerStdio)

    def test_http_when_url_given(self, env: None) -> None:
        server = build_travel_mcp_server(mcp_url="http://localhost:8002/mcp")
        assert isinstance(server, MCPServerStreamableHttp)


class TestBuildTravelAgent:
    def test_name_and_default_model(self, env: None) -> None:
        agent = build_travel_agent()
        assert agent.name == TRAVEL_AGENT_NAME
        assert agent.model == "gpt-5.4-mini"

    def test_attaches_mcp_server(self, env: None) -> None:
        agent = build_travel_agent()
        assert len(agent.mcp_servers) == 1


class TestSystemPromptContent:
    @pytest.mark.parametrize(
        "tool_name",
        ["search_flights", "search_hotels", "estimate_total_cost"],
    )
    def test_mentions_each_tool(self, tool_name: str) -> None:
        assert tool_name in TRAVEL_AGENT_INSTRUCTIONS

    def test_mentions_iata_convention(self) -> None:
        # The flight-search tool requires IATA codes — the prompt must say so
        # to avoid hallucinated city names.
        assert "IATA" in TRAVEL_AGENT_INSTRUCTIONS