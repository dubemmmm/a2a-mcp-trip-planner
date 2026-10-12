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

Economy, in two strengths:

  * priced (`pricing=` a `PricingPolicy`): each completed run is metered —
    tokens and tool calls from `RunUsage` — and the would-be charge is
    attached to the artifact's metadata under `"economy"` (status "metered").
  * paid (`payee=` a `Payee`): the agent answers quote requests and only runs
    tasks that carry a held ledger reservation; the metered charge is
    captured on success (status "captured") and the hold released on failure
    or cancellation. See `economy.protocol` for the message shapes.

Task state is held in `InMemoryTaskStore` — sufficient for Layer 4. When we
add input-required in a later layer (or do Layer 7 reliability work), this
becomes the swap point for `DatabaseTaskStore`.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Iterable
from typing import Any

import structlog

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
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
from google.protobuf.json_format import MessageToDict
from starlette.applications import Starlette

from economy.errors import LedgerError
from economy.models import Reservation
from economy.payee import Payee, PaymentRefused
from economy.pricing import PricingPolicy
from economy.protocol import (
    ProtocolError,
    QuoteDeclined,
    QuoteOffer,
    QuoteRequest,
    declined_payload,
    offer_payload,
    parse_quote_request,
    parse_reservation_id,
)
from shared.agent_runner import (
    AgentTaskError,
    AgentTaskOutcome,
    AgentTaskTimeout,
    run_agent_task,
)

_LOGGER = structlog.get_logger(__name__)

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
    extensions: Iterable[AgentExtension] = (),
) -> AgentCard:
    """Construct an `AgentCard` proto with sensible defaults.

    `url` is the public base URL the card advertises (e.g. http://localhost:8001).
    The JSON-RPC endpoint is mounted at "/" by default, which matches the
    `DEFAULT_RPC_URL` constant the SDK uses everywhere. `extensions` are
    declared under `capabilities` (e.g. the pricing extension).
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
        capabilities=AgentCapabilities(streaming=streaming, extensions=list(extensions)),
        default_input_modes=list(DEFAULT_INPUT_MODES),
        default_output_modes=list(DEFAULT_OUTPUT_MODES),
        skills=list(skills),
    )


class OpenAIAgentExecutor(AgentExecutor):
    """Run our existing Agents-SDK agent in response to A2A task requests.

    The executor pattern is the SDK's seam between protocol and agent logic.
    We extract the user's text, run the agent, publish updates to the event
    queue, and let the framework handle JSON-RPC plumbing and persistence.

    With a `payee`, the agent only works when paid: a quote request gets a
    quote back as a plain Message (no task), and a task must carry a held
    reservation naming this agent, which is captured on success and released
    on failure or cancellation.
    """

    def __init__(
        self,
        agent: Agent[Any],
        *,
        timeout_seconds: float = 120.0,
        max_turns: int = 10,
        pricing: PricingPolicy | None = None,
        payee: Payee | None = None,
    ) -> None:
        self._agent = agent
        self._timeout_seconds = timeout_seconds
        self._max_turns = max_turns
        self._payee = payee
        self._pricing = payee.pricing if payee is not None else pricing
        # task_id -> reservation_id for paid tasks still running, so a
        # cancellation can return the hold.
        self._holds: dict[str, str] = {}

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        # The framework populates task_id/context_id before calling execute().
        task_id = context.task_id or ""
        context_id = context.context_id or ""
        metadata = _message_metadata(context)

        # A quote request is answered with a plain Message — no task is
        # created, nothing runs, nothing is charged.
        if self._payee is not None:
            try:
                quote_request = parse_quote_request(metadata)
            except ProtocolError as e:
                await event_queue.enqueue_event(
                    _data_message(context_id, declined_payload(QuoteDeclined("bad_request", str(e))))
                )
                return
            if quote_request is not None:
                await self._reply_quote(context_id, metadata, quote_request, event_queue)
                return

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

        reservation: Reservation | None = None
        if self._payee is not None:
            reservation_id = parse_reservation_id(metadata)
            if reservation_id is None:
                await updater.failed(
                    message=_agent_text_message(
                        "payment required: request a quote, reserve it at the "
                        "ledger, and send the reservation_id with the task"
                    )
                )
                return
            try:
                reservation = await self._payee.accept(
                    reservation_id, skill_id=_skill_id(metadata), task_id=task_id
                )
            except PaymentRefused as e:
                await updater.failed(message=_agent_text_message(f"payment refused: {e}"))
                return
            self._holds[task_id] = reservation.reservation_id

        try:
            try:
                async with _open_mcp_servers(self._agent):
                    outcome = await run_agent_task(
                        self._agent,
                        user_input,
                        timeout_seconds=self._timeout_seconds,
                        max_turns=self._max_turns,
                    )
            except AgentTaskError as e:
                if reservation is not None and self._payee is not None:
                    await self._payee.release(reservation.reservation_id)
                prefix = "agent timed out: " if isinstance(e, AgentTaskTimeout) else ""
                await updater.failed(message=_agent_text_message(f"{prefix}{e}"))
                return

            final_text = str(outcome.result.final_output_as(str))
            receipt = await self._settle(context, metadata, outcome, reservation)
            # Publish the response as both an artifact (so it persists on the
            # task record) and a final message (so streaming clients see it).
            await updater.add_artifact(
                parts=[Part(text=final_text)],
                name=f"{self._agent.name}-response",
                metadata=receipt,
            )
            await updater.complete()
        finally:
            self._holds.pop(task_id, None)

    async def _reply_quote(
        self,
        context_id: str,
        metadata: dict[str, Any],
        request: QuoteRequest,
        event_queue: EventQueue,
    ) -> None:
        assert self._payee is not None
        skill_id = _priced_skill(metadata, self._payee.pricing)
        reply = await self._payee.make_quote(request, skill_id=skill_id)
        if isinstance(reply, QuoteDeclined):
            payload = declined_payload(reply)
        else:
            offer = QuoteOffer(
                quote_id=reply.quote_id,
                skill_id=reply.skill_id,
                amount_mc=reply.amount_mc,
                payee=reply.payee,
            )
            payload = offer_payload(offer, expires_at=reply.expires_at.isoformat())
        _LOGGER.info("quote_replied", agent=self._agent.name, skill_id=skill_id, **payload)
        await event_queue.enqueue_event(_data_message(context_id, payload))

    async def _settle(
        self,
        context: RequestContext,
        metadata: dict[str, Any],
        outcome: AgentTaskOutcome,
        reservation: Reservation | None,
    ) -> dict[str, Any] | None:
        """Charge (paid) or just meter (priced) the finished run.

        Returns the artifact's `"economy"` metadata, or None for a free agent.
        """
        if self._pricing is None:
            return None
        usage = outcome.usage.to_dict()

        if reservation is not None and self._payee is not None:
            try:
                charge = await self._payee.capture(reservation, outcome.usage)
                status = "captured"
            except LedgerError as e:
                # Usually the payer already released (it gave up waiting) or
                # the hold expired. The work is delivered; it just isn't paid.
                charge = self._pricing.charge(
                    reservation.skill_id, outcome.usage, ceiling_mc=reservation.amount_mc
                )
                status = "unpaid"
                _LOGGER.warning(
                    "task_capture_failed",
                    task_id=context.task_id,
                    reservation_id=reservation.reservation_id,
                    error=str(e),
                )
            _LOGGER.info(
                "task_settled",
                agent=self._agent.name,
                task_id=context.task_id,
                reservation_id=reservation.reservation_id,
                status=status,
                charged_mc=charge.charged_mc,
                **usage,
            )
            return {
                "economy": {
                    "status": status,
                    "reservation_id": reservation.reservation_id,
                    "charge": charge.model_dump(),
                    "usage": usage,
                }
            }

        skill_id = _priced_skill(metadata, self._pricing)
        charge = self._pricing.charge(
            skill_id, outcome.usage, ceiling_mc=self._pricing.quote(skill_id)
        )
        self._pricing.record(skill_id, outcome.usage)
        _LOGGER.info(
            "task_metered",
            agent=self._agent.name,
            task_id=context.task_id,
            skill_id=skill_id,
            charged_mc=charge.charged_mc,
            uncapped_mc=charge.uncapped_mc,
            ceiling_mc=charge.ceiling_mc,
            **usage,
        )
        return {"economy": {"status": "metered", "charge": charge.model_dump(), "usage": usage}}

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        # We don't have a meaningful way to cancel mid-LLM-call yet; mark the
        # task canceled at the protocol level and let the agent task finish in
        # the background. Layer 7 work will introduce real cancellation.
        # A paid task's hold is returned now, so the payer is never charged
        # for cancelled work: the late capture is refused by the ledger.
        reservation_id = self._holds.pop(context.task_id or "", None)
        if reservation_id is not None and self._payee is not None:
            await self._payee.release(reservation_id)
        updater = TaskUpdater(
            event_queue,
            task_id=context.task_id or "",
            context_id=context.context_id or "",
        )
        await updater.cancel()


def _message_metadata(context: RequestContext) -> dict[str, Any]:
    message = context.message
    return MessageToDict(message.metadata) if message else {}


def _skill_id(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("skill_id")
    return value if isinstance(value, str) and value else None


def _priced_skill(metadata: dict[str, Any], pricing: PricingPolicy) -> str:
    """The priced skill the caller asked for, else the sheet's default.

    `submit_task` puts the skill id in the message metadata; the A2A spec
    leaves skill routing to the agent, so an absent or unknown id is priced
    as the agent's default skill rather than rejected.
    """
    requested = _skill_id(metadata)
    if requested is not None and requested in pricing.sheet.skills:
        return requested
    return pricing.sheet.default_skill


def _data_message(context_id: str, payload: dict[str, Any]) -> Message:
    """An agent-role Message carrying one structured data part."""
    import uuid

    part = Part()
    part.data.struct_value.update(payload)
    return Message(
        message_id=str(uuid.uuid4()),
        context_id=context_id,
        role=Role.ROLE_AGENT,
        parts=[part],
    )


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
    pricing: PricingPolicy | None = None,
    payee: Payee | None = None,
) -> Starlette:
    """Construct the Starlette app exposing `agent` over A2A.

    Two route groups are mounted:
      * `GET /.well-known/agent-card.json` — Agent Card discovery.
      * `POST /` — JSON-RPC dispatcher (send_message, get_task, subscribe, …).
    """
    task_store = InMemoryTaskStore()
    executor = OpenAIAgentExecutor(
        agent,
        timeout_seconds=timeout_seconds,
        max_turns=max_turns,
        pricing=pricing,
        payee=payee,
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

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if payee is not None:
                await payee.ledger.close()

    return Starlette(routes=routes, lifespan=lifespan)


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