"""Shared A2A server scaffolding.

Each agent's A2A module defines its Agent Card and skills, then calls
`build_a2a_app(agent, agent_card)` to get a uvicorn-ready Starlette app.

The bridge between A2A and our existing agent code is `OpenAIAgentExecutor`,
which:

  * extracts the user's text from the incoming A2A message,
  * runs it through `run_agent_task` (Layer 3.2),
  * publishes status updates (submitted → working → completed/failed) via
    `TaskUpdater`, so streaming clients see progress in real time,
  * publishes the final response as an artifact.

Task state is held in `InMemoryTaskStore` — sufficient for Layer 4. When we
add input-required in a later layer (or do Layer 7 reliability work), this
becomes the swap point for `DatabaseTaskStore`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
)
from a2a.utils.constants import DEFAULT_RPC_URL
from agents import Agent
from starlette.applications import Starlette

from shared.agent_runner import (
    AgentTaskError,
    AgentTaskTimeout,
    run_agent_task,
)

DEFAULT_PROTOCOL_VERSION = "0.4.0"
DEFAULT_INPUT_MODES = ("text/plain",)
DEFAULT_OUTPUT_MODES = ("text/plain",)


def build_agent_card(
    *,
    name: str,
    description: str,
    url: str,
    skills: Iterable[AgentSkill],
    version: str = "0.1.0",
    streaming: bool = True,
) -> AgentCard:
    """Construct an `AgentCard` proto with sensible defaults.

    `url` is the public base URL the card advertises (e.g. http://localhost:8001).
    The JSON-RPC endpoint is mounted at "/" by default, which matches the
    `DEFAULT_RPC_URL` constant the SDK uses everywhere.
    """
    return AgentCard(
        name=name,
        description=description,
        version=version,
        supported_interfaces=[
            AgentInterface(
                url=url,
                protocol_binding="JSONRPC",
                protocol_version=DEFAULT_PROTOCOL_VERSION,
            )
        ],
        capabilities=AgentCapabilities(streaming=streaming),
        default_input_modes=list(DEFAULT_INPUT_MODES),
        default_output_modes=list(DEFAULT_OUTPUT_MODES),
        skills=list(skills),
    )


class OpenAIAgentExecutor(AgentExecutor):
    """Run our existing Agents-SDK agent in response to A2A task requests.

    The executor pattern is the SDK's seam between protocol and agent logic.
    We extract the user's text, run the agent, publish updates to the event
    queue, and let the framework handle JSON-RPC plumbing and persistence.
    """

    def __init__(
        self,
        agent: Agent[Any],
        *,
        timeout_seconds: float = 120.0,
        max_turns: int = 10,
    ) -> None:
        self._agent = agent
        self._timeout_seconds = timeout_seconds
        self._max_turns = max_turns

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        # The framework populates task_id/context_id before calling execute().
        task_id = context.task_id or ""
        context_id = context.context_id or ""

        # Enqueue the initial Task object so subsequent TaskStatusUpdateEvents
        # are accepted by the framework. Without this, the active-task
        # validator complains "Agent should enqueue Task before
        # TaskStatusUpdateEvent event".
        await event_queue.enqueue_event(
            Task(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
            )
        )

        updater = TaskUpdater(
            event_queue, task_id=task_id, context_id=context_id
        )

        # Start work — streaming clients see this immediately.
        await updater.start_work()

        user_input = context.get_user_input()
        if not user_input.strip():
            await updater.failed(
                message=_agent_text_message("empty input message")
            )
            return

        try:
            async with _open_mcp_servers(self._agent):
                outcome = await run_agent_task(
                    self._agent,
                    user_input,
                    timeout_seconds=self._timeout_seconds,
                    max_turns=self._max_turns,
                )
        except AgentTaskTimeout as e:
            await updater.failed(
                message=_agent_text_message(f"agent timed out: {e}")
            )
            return
        except AgentTaskError as e:
            await updater.failed(message=_agent_text_message(str(e)))
            return

        final_text = str(outcome.result.final_output_as(str))
        # Publish the response as both an artifact (so it persists on the
        # task record) and a final message (so streaming clients see it).
        await updater.add_artifact(
            parts=[Part(text=final_text)],
            name=f"{self._agent.name}-response",
        )
        await updater.complete()

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        # We don't have a meaningful way to cancel mid-LLM-call yet; mark the
        # task canceled at the protocol level and let the agent task finish in
        # the background. Layer 7 work will introduce real cancellation.
        updater = TaskUpdater(
            event_queue,
            task_id=context.task_id or "",
            context_id=context.context_id or "",
        )
        await updater.cancel()


def _open_mcp_servers(agent: Agent[Any]) -> "_MCPServersContext":
    """Return an async context manager that opens all of `agent.mcp_servers`.

    Each MCP server is connected on entry and closed on exit.
    """
    return _MCPServersContext(list(agent.mcp_servers))


class _MCPServersContext:
    """Async context manager that opens/closes all of an agent's MCP servers."""

    def __init__(self, servers: list[Any]) -> None:
        self._servers = servers
        self._entered: list[Any] = []

    async def __aenter__(self) -> "_MCPServersContext":
        for server in self._servers:
            await server.__aenter__()
            self._entered.append(server)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        # Close in reverse order, swallowing exit errors so one bad close
        # doesn't mask the others.
        last_exc: BaseException | None = None
        for server in reversed(self._entered):
            try:
                await server.__aexit__(exc_type, exc, tb)
            except Exception as e:
                last_exc = e
        self._entered.clear()
        if last_exc is not None and exc is None:
            raise last_exc


def _agent_text_message(text: str) -> Message:
    """Build an agent-role Message containing a single text Part."""
    import uuid

    return Message(
        message_id=str(uuid.uuid4()),
        role=Role.ROLE_AGENT,
        parts=[Part(text=text)],
    )


def build_a2a_app(
    agent: Agent[Any],
    agent_card: AgentCard,
    *,
    timeout_seconds: float = 120.0,
    max_turns: int = 10,
) -> Starlette:
    """Construct the Starlette app exposing `agent` over A2A.

    Two route groups are mounted:
      * `GET /.well-known/agent-card.json` — Agent Card discovery.
      * `POST /` — JSON-RPC dispatcher (send_message, get_task, subscribe, …).
    """
    task_store = InMemoryTaskStore()
    executor = OpenAIAgentExecutor(
        agent, timeout_seconds=timeout_seconds, max_turns=max_turns
    )
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
        agent_card=agent_card,
    )

    routes = [
        *create_agent_card_routes(agent_card=agent_card),
        *create_jsonrpc_routes(
            request_handler, rpc_url=DEFAULT_RPC_URL, enable_v0_3_compat=True
        ),
    ]
    return Starlette(routes=routes)


def text_part(text: str) -> Part:
    """Construct a text-content Part. Convenience for callers."""
    return Part(text=text)


def task_is_terminal(state: int) -> bool:
    """Return True if `state` is a terminal task state."""
    return state in {
        TaskState.TASK_STATE_COMPLETED,
        TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_CANCELED,
        TaskState.TASK_STATE_REJECTED,
    }