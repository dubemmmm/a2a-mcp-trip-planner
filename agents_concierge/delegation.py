"""Delegation tools for the Concierge Agent.

Each tool wraps the full A2A client flow (discover → submit → await) into a
single function the Concierge's LLM can call like any other tool.

The Concierge can decide *whether* to call a specialist (it might not need
flights for a question like "what's the weather in Lisbon next week"), and
*what to say* to that specialist (it composes a natural-language sub-prompt
from the user's request). The specialist is itself a full agent with its
own MCP tools, so the Concierge gets a rich, synthesized response back —
not raw tool output.

Tools are built via `build_delegation_tools(...)` which takes a discovery
instance and the public URLs of the specialists. URLs are closed over so the
LLM doesn't see them (they're not parameters of the tool).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
import structlog
from a2a.client import Client, ClientFactory
from a2a.client.client import ClientConfig
from a2a.types import AgentCard
from agents import FunctionTool, function_tool

from shared.agent_events import AgentEventQueue, emit_agent_event
from shared.a2a_client import (
    AgentDiscovery,
    AgentDiscoveryError,
    SubmissionError,
    TaskFailedError,
    TaskWaitError,
    TaskWaitTimeout,
    await_task,
    submit_task,
)

_LOGGER = structlog.get_logger(__name__)

# Skill ids match those declared in a2a_servers.travel/local/transit.
TRAVEL_SKILL_ID = "search_flights"  # generic "do travel work" entry point
LOCAL_SKILL_ID = "find_attractions"  # generic "do local research" entry point
TRANSIT_SKILL_ID = "get_route"  # generic "get around the destination" entry point

DEFAULT_DELEGATION_TIMEOUT_SECONDS = 120.0

# httpx's default read timeout is 5s — far too short for a streaming response
# from a specialist agent that must run LLM inference + external API calls.
# Set read=None so the transport never cuts the stream early; the application-
# level timeout_seconds budget enforces the wall-time cap instead.
_HTTPX_TIMEOUT = httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0)


class DelegationError(RuntimeError):
    """Surfaced when a delegation call cannot return useful data.

    The Concierge's LLM sees this as the tool's failure result and can decide
    whether to apologize to the user, retry with different wording, or
    proceed without the specialist's input.
    """


@asynccontextmanager
async def _client_for_card(
    card: AgentCard,
    factory: ClientFactory | None,
) -> AsyncIterator[Client]:
    """Yield an A2A Client and guarantee cleanup on exit.

    Two paths:
      * factory is None (production): create a per-call httpx.AsyncClient so
        parallel delegations each own their connection pool and closing one
        cannot affect another.
      * factory is provided (tests): create the client from the injected
        factory and close it on exit, preserving the test contract that
        client.close() is awaited exactly once.
    """
    if factory is not None:
        client = factory.create(card)
        try:
            yield client
        finally:
            await client.close()
    else:
        async with httpx.AsyncClient(timeout=_HTTPX_TIMEOUT) as http:
            per_call_factory = ClientFactory(ClientConfig(httpx_client=http))
            yield per_call_factory.create(card)


async def _delegate(
    *,
    request: str,
    agent_url: str,
    skill_id: str,
    discovery: AgentDiscovery,
    client_factory: ClientFactory | None,
    timeout_seconds: float,
    event_queue: AgentEventQueue | None = None,
    event_agent_name: str | None = None,
) -> str:
    """Run one full delegation: discover, submit, await, return text.

    Returns the combined artifact text from the remote agent. Raises
    `DelegationError` on any failure path so the SDK reports a single
    consistent shape to the LLM.
    """
    display_name = event_agent_name or skill_id
    log = _LOGGER.bind(agent_url=agent_url, skill_id=skill_id)
    emit_agent_event(
        event_queue,
        kind="agent.status",
        agent=display_name,
        status="requested",
        message=f"Concierge requested the {display_name.title()} specialist",
        data={"skill_id": skill_id},
    )
    try:
        card: AgentCard = await discovery.discover(agent_url)
    except AgentDiscoveryError as e:
        log.warning("delegation_discovery_failed", error=str(e))
        emit_agent_event(
            event_queue,
            kind="agent.status",
            agent=display_name,
            status="failed",
            message=f"{display_name.title()} specialist could not be reached",
            data={"error": str(e)},
        )
        raise DelegationError(f"could not reach agent at {agent_url}: {e}") from e

    async with _client_for_card(card, client_factory) as client:
        try:
            handle = await submit_task(client, card, skill_id=skill_id, payload=request)
        except SubmissionError as e:
            log.warning("delegation_submit_failed", error=str(e))
            emit_agent_event(
                event_queue,
                kind="agent.status",
                agent=display_name,
                status="failed",
                message=f"Request to {display_name.title()} could not be submitted",
                data={"error": str(e)},
            )
            raise DelegationError(f"could not send to {card.name}: {e}") from e

        try:
            result = await await_task(
                handle,
                timeout_seconds=timeout_seconds,
                event_queue=event_queue,
                event_agent_name=display_name,
            )
        except TaskWaitTimeout as e:
            log.warning("delegation_timeout", timeout=timeout_seconds)
            emit_agent_event(
                event_queue,
                kind="agent.status",
                agent=display_name,
                status="failed",
                message=f"{display_name.title()} specialist timed out",
            )
            raise DelegationError(f"{card.name} did not respond within {timeout_seconds}s") from e
        except TaskFailedError as e:
            log.warning("delegation_remote_failed", state=e.state)
            emit_agent_event(
                event_queue,
                kind="agent.status",
                agent=display_name,
                status="failed",
                message=f"{display_name.title()} specialist reported a failure",
                data={"state": e.state},
            )
            raise DelegationError(f"{card.name} reported failure: {e}") from e
        except TaskWaitError as e:
            log.warning("delegation_wait_error", error=str(e))
            emit_agent_event(
                event_queue,
                kind="agent.status",
                agent=display_name,
                status="failed",
                message=f"{display_name.title()} specialist response was incomplete",
                data={"error": str(e)},
            )
            raise DelegationError(f"{card.name} responded incompletely: {e}") from e

    log.info(
        "delegation_succeeded",
        agent=card.name,
        duration=result.duration_seconds,
        artifact_chars=len(result.combined_text),
    )
    if not result.combined_text.strip():
        emit_agent_event(
            event_queue,
            kind="agent.status",
            agent=display_name,
            status="failed",
            message=f"{display_name.title()} specialist returned no research",
        )
        raise DelegationError(f"{card.name} returned no text")
    return result.combined_text


def build_delegation_tools(
    *,
    travel_url: str,
    local_url: str,
    transit_url: str | None = None,
    discovery: AgentDiscovery,
    client_factory: ClientFactory | None = None,
    timeout_seconds: float = DEFAULT_DELEGATION_TIMEOUT_SECONDS,
    event_queue: AgentEventQueue | None = None,
) -> list[FunctionTool]:
    """Build the delegation tools as a closure over the given URLs/clients.

    Args:
        travel_url: Public base URL of the Travel A2A server.
        local_url: Public base URL of the Local Experience A2A server.
        transit_url: Public base URL of the Ground Transport A2A server. When
            None, no `call_transit_agent` tool is built (lets callers run a
            subset of specialists).
        discovery: Shared discovery cache. One per Concierge process is fine.
        client_factory: Optional override for testing. In production leave as
            None — each delegation call will create its own httpx client so
            parallel calls cannot close each other's connection pool.
        timeout_seconds: Per-delegation wall-time cap.
        event_queue: Optional queue receiving live specialist lifecycle events.

    Returns:
        A list of `FunctionTool`s ready to attach to the Concierge agent —
        one per specialist whose URL was provided.
    """

    @function_tool(
        name_override="call_travel_agent",
        description_override=(
            "Ask the Travel specialist agent to search for flights, hotels, "
            "or estimate trip cost. Pass a clear natural-language request "
            "(e.g. 'find round-trip flights ATL to LIS for May 10-14 under "
            "$800, plus hotels under $150/night'). The specialist will return "
            "a synthesized text response. Use this when the user's question "
            "involves transport or lodging."
        ),
    )
    async def call_travel_agent(request: str) -> str:
        return await _delegate(
            request=request,
            agent_url=travel_url,
            skill_id=TRAVEL_SKILL_ID,
            discovery=discovery,
            client_factory=client_factory,
            timeout_seconds=timeout_seconds,
            event_queue=event_queue,
            event_agent_name="travel",
        )

    @function_tool(
        name_override="call_local_experience_agent",
        description_override=(
            "Ask the Local Experience specialist agent for attractions, "
            "restaurants, or weather at a destination. Pass a clear "
            "natural-language request (e.g. 'recommend 3 history sites and "
            "the weather for Lisbon May 10-14'). The specialist will return "
            "a synthesized text response. Use this when the user's question "
            "involves things to do, dining, or local conditions."
        ),
    )
    async def call_local_experience_agent(request: str) -> str:
        return await _delegate(
            request=request,
            agent_url=local_url,
            skill_id=LOCAL_SKILL_ID,
            discovery=discovery,
            client_factory=client_factory,
            timeout_seconds=timeout_seconds,
            event_queue=event_queue,
            event_agent_name="local",
        )

    tools: list[FunctionTool] = [call_travel_agent, call_local_experience_agent]

    if transit_url is not None:

        @function_tool(
            name_override="call_transit_agent",
            description_override=(
                "Ask the Ground Transport specialist agent how to get around "
                "a destination. Pass a clear natural-language request (e.g. "
                "'fastest way from Lisbon Airport to Belem Tower, compare "
                "transit, driving, and walking'). The specialist will return "
                "a synthesized text response. Use this when the user's "
                "question involves local routes, directions, or travel time "
                "between places at the destination."
            ),
        )
        async def call_transit_agent(request: str) -> str:
            return await _delegate(
                request=request,
                agent_url=transit_url,
                skill_id=TRANSIT_SKILL_ID,
                discovery=discovery,
                client_factory=client_factory,
                timeout_seconds=timeout_seconds,
                event_queue=event_queue,
                event_agent_name="transit",
            )

        tools.append(call_transit_agent)

    return tools
