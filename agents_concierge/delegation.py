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

Paid specialists (their Agent Card carries the pricing extension) are called
through a `Payer`: ask for a quote, let `SpendPolicy` decide, reserve the
credits, send the task with the reservation id, then settle. A call the
budget does not allow is not an error — the tool returns a "NOT CALLED"
result naming the reason, so the LLM can work around it.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
import structlog
from a2a.client import Client, ClientFactory
from a2a.client.client import ClientConfig
from a2a.types import AgentCard, Message, Part, Role, SendMessageRequest
from agents import FunctionTool, function_tool
from google.protobuf.json_format import MessageToDict

from economy.card import InvalidPricingExtension, read_pricing
from economy.models import ReservationState
from economy.payer import Authorization, Payer, Rejection, Settlement
from economy.policy import RejectReason, credits
from economy.protocol import (
    ProtocolError,
    QuoteDeclined,
    QuoteOffer,
    QuoteRequest,
    paid_task_metadata,
    parse_quote_reply,
    quote_request_metadata,
)

from shared.agent_events import AgentEventQueue, emit_agent_event
from shared.a2a_client import (
    AgentDiscovery,
    AgentDiscoveryError,
    SubmissionError,
    TaskFailedError,
    TaskResult,
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
    payer: Payer | None = None,
) -> str:
    """Run one full delegation: discover, (pay,) submit, await, return text.

    Returns the combined artifact text from the remote agent — with a receipt
    line for a paid call, or a "NOT CALLED" explanation when the call was not
    affordable. Raises `DelegationError` on any failure path so the SDK
    reports a single consistent shape to the LLM.
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

    try:
        pricing = read_pricing(card)
    except InvalidPricingExtension as e:
        raise DelegationError(f"{card.name} publishes malformed pricing: {e}") from e
    if pricing is not None and payer is None:
        return _not_called(
            display_name,
            "it charges credits and this Concierge has no wallet configured",
            event_queue,
        )

    authorization: Authorization | None = None
    settlement: Settlement | None = None
    async with _client_for_card(card, client_factory) as client:
        if pricing is not None and payer is not None:
            listed = pricing.price(skill_id)

            async def request_quote(max_mc: int | None) -> QuoteOffer | QuoteDeclined:
                reply = await _request_quote(
                    client,
                    skill_id=skill_id,
                    request=request,
                    quote_request=QuoteRequest(payer=payer.wallet, max_mc=max_mc),
                    agent_name=card.name,
                )
                if isinstance(reply, QuoteOffer):
                    emit_agent_event(
                        event_queue,
                        kind="payment.quoted",
                        agent=display_name,
                        message=f"{display_name.title()} quoted {credits(reply.amount_mc)}",
                        data={
                            "skill_id": skill_id,
                            "quote_id": reply.quote_id,
                            "ceiling_mc": reply.amount_mc,
                            "counter": max_mc is not None,
                        },
                    )
                return reply

            decision = await payer.authorize(
                base_mc=listed.base_mc if listed else 0, request_quote=request_quote
            )
            if isinstance(decision, Rejection):
                log.info("delegation_not_paid", reason=str(decision.reason))
                return _not_called(display_name, decision.message, event_queue, decision)
            authorization = decision
            emit_agent_event(
                event_queue,
                kind="payment.reserved",
                agent=display_name,
                message=(
                    f"Held {credits(authorization.reservation.amount_mc)} "
                    f"for {display_name.title()}"
                ),
                data={
                    "reservation_id": authorization.reservation.reservation_id,
                    "amount_mc": authorization.reservation.amount_mc,
                    "trip_remaining_mc": payer.trip_remaining_mc,
                },
            )

        try:
            result = await _submit_and_await(
                client,
                card,
                request=request,
                skill_id=skill_id,
                display_name=display_name,
                timeout_seconds=timeout_seconds,
                event_queue=event_queue,
                metadata=(
                    paid_task_metadata(authorization.reservation.reservation_id)
                    if authorization is not None
                    else None
                ),
            )
        finally:
            if authorization is not None and payer is not None:
                settlement = await payer.settle(authorization)
                _emit_settlement(event_queue, display_name, settlement, payer)

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
    if settlement is not None and payer is not None:
        return f"{result.combined_text}\n\n{_receipt(settlement, payer)}"
    return result.combined_text


async def _submit_and_await(
    client: Client,
    card: AgentCard,
    *,
    request: str,
    skill_id: str,
    display_name: str,
    timeout_seconds: float,
    event_queue: AgentEventQueue | None,
    metadata: dict[str, object] | None,
) -> TaskResult:
    """Submit the task and wait for it, mapping every failure to DelegationError."""
    log = _LOGGER.bind(agent=card.name, skill_id=skill_id)
    try:
        handle = await submit_task(
            client, card, skill_id=skill_id, payload=request, metadata=metadata
        )
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
    return result


async def _request_quote(
    client: Client,
    *,
    skill_id: str,
    request: str,
    quote_request: QuoteRequest,
    agent_name: str,
) -> QuoteOffer | QuoteDeclined:
    """Ask a priced specialist for a quote: one message out, one message back."""
    message = Message(message_id=str(uuid.uuid4()), role=Role.ROLE_USER, parts=[Part(text=request)])
    message.metadata.update(quote_request_metadata(skill_id, quote_request))
    payloads: list[dict[str, object]] = []
    try:
        # Drain the whole (one-message) stream rather than returning from
        # inside it, so the client's generator always closes cleanly.
        async for event in client.send_message(SendMessageRequest(message=message)):
            if event.WhichOneof("payload") == "message":
                payloads.extend(
                    MessageToDict(part.data) for part in event.message.parts if part.HasField("data")
                )
    except Exception as e:
        raise DelegationError(f"could not get a quote from {agent_name}: {e}") from e
    for payload in payloads:
        if isinstance(payload, dict):
            try:
                return parse_quote_reply(payload)
            except ProtocolError as e:
                raise DelegationError(f"{agent_name} sent a malformed quote: {e}") from e
    raise DelegationError(f"{agent_name} did not answer the quote request with a quote")


def _not_called(
    display_name: str,
    reason: str,
    event_queue: AgentEventQueue | None,
    rejection: Rejection | None = None,
) -> str:
    emit_agent_event(
        event_queue,
        kind="payment.rejected",
        agent=display_name,
        status="skipped",
        message=f"{display_name.title()} specialist not called: {reason}",
        data={
            "reason": _reason_code(rejection),
            "quote_mc": rejection.quote_mc if rejection else None,
        },
    )
    return (
        f"NOT CALLED: the {display_name} specialist was not called because {reason}. "
        "Do not retry it for this trip; answer without it or tell the user."
    )


def _reason_code(rejection: Rejection | None) -> str:
    if rejection is None:
        return "no_wallet"
    reason = rejection.reason
    return reason.value if isinstance(reason, RejectReason) else reason


def _emit_settlement(
    event_queue: AgentEventQueue | None,
    display_name: str,
    settlement: Settlement,
    payer: Payer,
) -> None:
    captured = settlement.state is ReservationState.CAPTURED
    emit_agent_event(
        event_queue,
        kind="payment.captured" if captured else "payment.released",
        agent=display_name,
        message=(
            f"Paid {display_name.title()} {credits(settlement.charged_mc)}"
            if captured
            else f"Hold for {display_name.title()} returned"
        ),
        data={
            "reservation_id": settlement.reservation_id,
            "charged_mc": settlement.charged_mc,
            "state": settlement.state.value if settlement.state is not None else "unconfirmed",
            "trip_remaining_mc": payer.trip_remaining_mc,
        },
    )


def _receipt(settlement: Settlement, payer: Payer) -> str:
    return (
        f"[charged {credits(settlement.charged_mc)} · "
        f"trip remaining {credits(payer.trip_remaining_mc)}]"
    )


def build_delegation_tools(
    *,
    travel_url: str,
    local_url: str,
    transit_url: str | None = None,
    discovery: AgentDiscovery,
    client_factory: ClientFactory | None = None,
    timeout_seconds: float = DEFAULT_DELEGATION_TIMEOUT_SECONDS,
    event_queue: AgentEventQueue | None = None,
    payer: Payer | None = None,
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
        payer: Pays priced specialists from the Concierge's wallet for this
            trip. Without one, priced specialists are reported as not called;
            free specialists work either way.

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
            payer=payer,
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
            payer=payer,
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
                payer=payer,
            )

        tools.append(call_transit_agent)

    return tools
