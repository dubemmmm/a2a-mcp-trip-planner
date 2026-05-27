"""Tests for the Local Experience agent builder."""

from __future__ import annotations

import pytest
from agents.mcp import MCPServerStdio

from agents_local.agent import (
    LOCAL_AGENT_INSTRUCTIONS,
    LOCAL_AGENT_NAME,
    build_local_agent,
    build_local_mcp_server,
)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ENVIRONMENT", "test")


class TestBuildLocalMcpServer:
    def test_stdio_default(self, env: None) -> None:
        server = build_local_mcp_server()
        assert isinstance(server, MCPServerStdio)
        assert server.name == f"{LOCAL_AGENT_NAME}-mcp"

    def test_http_when_url_given(self, env: None) -> None:
        from agents.mcp import MCPServerStreamableHttp

        server = build_local_mcp_server(mcp_url="http://localhost:8001/mcp")
        assert isinstance(server, MCPServerStreamableHttp)


class TestBuildLocalAgent:
    def test_name_and_model_default(self, env: None) -> None:
        agent = build_local_agent()
        assert agent.name == LOCAL_AGENT_NAME
        # Defaults from Settings — gpt-5-mini is the configured default.
        assert agent.model == "gpt-5.4-mini"

    def test_model_override(self, env: None) -> None:
        agent = build_local_agent(model="custom-model")
        assert agent.model == "custom-model"

    def test_attaches_one_mcp_server(self, env: None) -> None:
        agent = build_local_agent()
        assert len(agent.mcp_servers) == 1
        assert agent.mcp_servers[0].name == f"{LOCAL_AGENT_NAME}-mcp"

    def test_uses_injected_server(self, env: None) -> None:
        injected = build_local_mcp_server(mcp_url="http://localhost:9999/mcp")
        agent = build_local_agent(mcp_server=injected)
        assert agent.mcp_servers[0] is injected


class TestSystemPromptContent:
    """The prompt must mention the three tools so the model knows it has them."""

    @pytest.mark.parametrize(
        "tool_name",
        ["find_attractions", "recommend_restaurants", "get_weather_forecast"],
    )
    def test_mentions_each_tool(self, tool_name: str) -> None:
        assert tool_name in LOCAL_AGENT_INSTRUCTIONS

    def test_identifies_role(self) -> None:
        # Loosely check the agent is told what kind of system it's in.
        assert "trip-planning" in LOCAL_AGENT_INSTRUCTIONS.lower()
        assert "local experience" in LOCAL_AGENT_INSTRUCTIONS.lower()