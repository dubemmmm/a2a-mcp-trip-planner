"""Starlette application exposing the trip planner through a structured form."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from agents_concierge.agent import build_orchestrating_concierge
from agents_concierge.delegation import build_delegation_tools
from agents_concierge.models import TripPlanResponse
from economy.client import LedgerClient
from economy.payer import Payer
from economy.policy import SpendPolicy
from shared.a2a_client import AgentDiscovery
from shared.agent_events import AgentEventQueue, emit_agent_event
from shared.agent_runner import AgentTaskError, AgentTaskTimeout
from shared.config import get_settings
from shared.output_validation import OutputValidationError, validate_agent_output

from .models import TRIP_PLAN_SCHEMA_VERSION, PlanResult, WebTripRequest

_ROOT = Path(__file__).resolve().parent
_STATIC = _ROOT / "static"
PlanRunner = Callable[[WebTripRequest], Awaitable[PlanResult]]
StreamingPlanRunner = Callable[[WebTripRequest, AgentEventQueue], Awaitable[PlanResult]]

_MAX_LIVE_SESSIONS = 100
_SESSION_RETENTION_SECONDS = 600.0
_TERMINAL_EVENT_KINDS = {"plan.completed", "plan.failed"}


@dataclass
class PlanSession:
    """One background planning run and its single-consumer event queue."""

    session_id: str
    events: AgentEventQueue
    task: asyncio.Task[None] | None = None
    cleanup_task: asyncio.Task[None] | None = None


async def _run_agent_plan(
    trip: WebTripRequest,
    *,
    discovery: AgentDiscovery,
    event_queue: AgentEventQueue | None = None,
) -> PlanResult:
    settings = get_settings()
    # With a Concierge key, each plan is one trip with its own credit budget;
    # without one, paid specialists are simply reported as not called.
    ledger = LedgerClient.from_settings("concierge", settings)
    payer = (
        Payer(
            ledger,
            SpendPolicy(max_call_mc=settings.economy_max_call_mc),
            trip_id=f"web-{uuid4().hex[:12]}",
            trip_budget_mc=settings.economy_max_trip_mc,
        )
        if ledger is not None
        else None
    )
    tools = build_delegation_tools(
        travel_url=settings.travel_a2a_url,
        local_url=settings.local_a2a_url,
        transit_url=settings.transit_a2a_url,
        discovery=discovery,
        timeout_seconds=min(settings.web_plan_timeout_seconds, 180.0),
        event_queue=event_queue,
        payer=payer,
    )
    agent = build_orchestrating_concierge(tools)
    mcp_server: Any = agent.mcp_servers[0]
    started_at = time.monotonic()
    try:
        async with mcp_server:
            validated = await validate_agent_output(
                agent,
                trip.to_agent_prompt(),
                TripPlanResponse,
                timeout_seconds=settings.web_plan_timeout_seconds,
                max_turns=settings.web_max_turns,
                event_queue=event_queue,
            )
    finally:
        if ledger is not None:
            await ledger.close()
    return PlanResult(
        plan=validated.value,
        duration_seconds=time.monotonic() - started_at,
        validation_retried=validated.retried,
    )


async def _homepage(_request: Request) -> Response:
    response = FileResponse(_STATIC / "index.html")
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["X-Trip-Planner-Schema"] = TRIP_PLAN_SCHEMA_VERSION
    return response


async def _health(_request: Request) -> Response:
    return JSONResponse(
        {
            "status": "ok",
            "service": "trip-planner-web",
            "response_schema": TRIP_PLAN_SCHEMA_VERSION,
        }
    )


async def _plan(request: Request) -> Response:
    try:
        body: Any = await request.json()
    except Exception:
        return JSONResponse({"error": "Request body must be valid JSON."}, status_code=400)

    try:
        trip = WebTripRequest.model_validate(body)
    except ValidationError as exc:
        return JSONResponse(
            {
                "error": "Please correct the highlighted trip details.",
                "details": exc.errors(include_context=False),
            },
            status_code=422,
        )

    if "respond-async" in request.headers.get("prefer", "").lower():
        return _start_plan_session(request, trip)

    runner: PlanRunner | None = getattr(request.app.state, "plan_runner", None)
    semaphore: asyncio.Semaphore | None = getattr(request.app.state, "plan_semaphore", None)
    if runner is None or semaphore is None:
        return JSONResponse(
            {"error": "The planning service is still starting. Try again shortly."},
            status_code=503,
        )

    try:
        async with semaphore:
            result = await runner(trip)
    except AgentTaskTimeout:
        return JSONResponse(
            {"error": "Planning took too long. Try a narrower trip request."},
            status_code=504,
        )
    except AgentTaskError:
        return JSONResponse(
            {"error": "The agent network could not complete this trip right now."},
            status_code=502,
        )
    except OutputValidationError:
        return JSONResponse(
            {"error": "The trip was planned, but its result could not be structured."},
            status_code=502,
        )

    response = JSONResponse(result.model_dump(mode="json"))
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Trip-Planner-Schema"] = TRIP_PLAN_SCHEMA_VERSION
    return response


def _start_plan_session(request: Request, trip: WebTripRequest) -> Response:
    runner: StreamingPlanRunner | None = getattr(request.app.state, "streaming_plan_runner", None)
    semaphore: asyncio.Semaphore | None = getattr(request.app.state, "plan_semaphore", None)
    sessions: dict[str, PlanSession] | None = getattr(request.app.state, "plan_sessions", None)
    if runner is None or semaphore is None or sessions is None:
        return JSONResponse(
            {"error": "The planning service is still starting. Try again shortly."},
            status_code=503,
        )
    if len(sessions) >= _MAX_LIVE_SESSIONS:
        return JSONResponse(
            {"error": "The live planner is at capacity. Try again shortly."},
            status_code=503,
        )

    session_id = uuid4().hex
    session = PlanSession(session_id=session_id, events=asyncio.Queue())
    sessions[session_id] = session
    session.task = asyncio.create_task(
        _execute_plan_session(
            session,
            trip=trip,
            runner=runner,
            semaphore=semaphore,
            sessions=sessions,
        ),
        name=f"web-plan-{session_id}",
    )
    response = JSONResponse(
        {
            "plan_id": session_id,
            "stream_url": f"/api/plan/stream?plan_id={session_id}",
        },
        status_code=202,
    )
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Trip-Planner-Schema"] = TRIP_PLAN_SCHEMA_VERSION
    return response


async def _execute_plan_session(
    session: PlanSession,
    *,
    trip: WebTripRequest,
    runner: StreamingPlanRunner,
    semaphore: asyncio.Semaphore,
    sessions: dict[str, PlanSession],
) -> None:
    emit_agent_event(
        session.events,
        kind="plan.queued",
        agent="concierge",
        status="queued",
        message="Trip brief received; waiting for the Concierge",
    )
    try:
        async with semaphore:
            result = await runner(trip, session.events)
    except AgentTaskTimeout:
        _emit_plan_failure(
            session.events,
            "Planning took too long. Try a narrower trip request.",
        )
    except AgentTaskError:
        _emit_plan_failure(
            session.events,
            "The agent network could not complete this trip right now.",
        )
    except OutputValidationError:
        _emit_plan_failure(
            session.events,
            "The trip was planned, but its result could not be structured.",
        )
    except Exception:
        _emit_plan_failure(
            session.events,
            "The planning service encountered an unexpected error.",
        )
    else:
        emit_agent_event(
            session.events,
            kind="plan.completed",
            agent="concierge",
            status="completed",
            message="The itinerary is ready",
            data={"result": result.model_dump(mode="json")},
        )
    if session.session_id in sessions:
        session.cleanup_task = asyncio.create_task(
            _expire_plan_session(session.session_id, sessions),
            name=f"web-plan-expiry-{session.session_id}",
        )


async def _expire_plan_session(session_id: str, sessions: dict[str, PlanSession]) -> None:
    await asyncio.sleep(_SESSION_RETENTION_SECONDS)
    sessions.pop(session_id, None)


def _emit_plan_failure(event_queue: AgentEventQueue, message: str) -> None:
    emit_agent_event(
        event_queue,
        kind="plan.failed",
        agent="concierge",
        status="failed",
        message=message,
    )


async def _plan_stream(request: Request) -> Response:
    session_id = request.query_params.get("plan_id", "").strip()
    if not session_id:
        return JSONResponse({"error": "plan_id is required."}, status_code=400)

    sessions: dict[str, PlanSession] | None = getattr(request.app.state, "plan_sessions", None)
    session = sessions.get(session_id) if sessions is not None else None
    if session is None:
        return JSONResponse({"error": "Planning session not found."}, status_code=404)

    async def stream() -> AsyncIterator[str]:
        sequence = 0
        terminal = False
        try:
            while not terminal:
                try:
                    event = await asyncio.wait_for(session.events.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue

                sequence += 1
                terminal = event.kind in _TERMINAL_EVENT_KINDS
                payload = json.dumps(event.to_dict(), separators=(",", ":"))
                yield f"id: {sequence}\nevent: planner\ndata: {payload}\n\n"
        finally:
            if terminal and sessions is not None:
                sessions.pop(session_id, None)
                if session.cleanup_task is not None:
                    session.cleanup_task.cancel()
                    await asyncio.gather(session.cleanup_task, return_exceptions=True)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def create_app(*, plan_runner: PlanRunner | None = None) -> Starlette:
    """Build the public web app, optionally injecting a planner for tests."""

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        settings = get_settings()
        app.state.plan_semaphore = asyncio.Semaphore(settings.web_max_concurrent_plans)
        app.state.plan_sessions = {}
        if plan_runner is not None:
            app.state.plan_runner = plan_runner

            async def injected_streaming_runner(
                trip: WebTripRequest, _events: AgentEventQueue
            ) -> PlanResult:
                return await plan_runner(trip)

            app.state.streaming_plan_runner = injected_streaming_runner
            try:
                yield
            finally:
                await _cancel_plan_sessions(app)
            return

        async with AgentDiscovery() as discovery:

            async def run(trip: WebTripRequest) -> PlanResult:
                return await _run_agent_plan(trip, discovery=discovery)

            async def run_streaming(trip: WebTripRequest, events: AgentEventQueue) -> PlanResult:
                return await _run_agent_plan(
                    trip,
                    discovery=discovery,
                    event_queue=events,
                )

            app.state.plan_runner = run
            app.state.streaming_plan_runner = run_streaming
            try:
                yield
            finally:
                await _cancel_plan_sessions(app)

    routes = [
        Route("/", _homepage),
        Route("/health", _health),
        Route("/api/plan", _plan, methods=["POST"]),
        Route("/api/plan/stream", _plan_stream, methods=["GET"]),
        Mount("/static", app=StaticFiles(directory=_STATIC), name="static"),
    ]
    app = Starlette(debug=False, routes=routes, lifespan=lifespan)
    if plan_runner is not None:
        app.state.plan_runner = plan_runner
        app.state.plan_semaphore = asyncio.Semaphore(2)
        app.state.plan_sessions = {}

        async def injected_streaming_runner(
            trip: WebTripRequest, _events: AgentEventQueue
        ) -> PlanResult:
            return await plan_runner(trip)

        app.state.streaming_plan_runner = injected_streaming_runner
    return app


async def _cancel_plan_sessions(app: Starlette) -> None:
    sessions: dict[str, PlanSession] = getattr(app.state, "plan_sessions", {})
    tasks = [
        task
        for session in sessions.values()
        for task in (session.task, session.cleanup_task)
        if task is not None
    ]
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    sessions.clear()


app = create_app()
