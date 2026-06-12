"""End-to-end smoke: Concierge orchestrating all specialist agents.

This runs the FULL system against real services:
  * Real Travel A2A server (gets real flight + hotel data from SerpApi)
  * Real Local Experience A2A server (real Google Places + OpenWeather)
  * Real Ground Transport A2A server (real routes from Google Routes)
  * Real Concierge with real OpenAI

Requires:
  OPENAI_API_KEY            (required)
  SERPAPI_API_KEY           (required — Travel)
  GOOGLE_PLACES_API_KEY     (required — Local Experience)
  OPENWEATHER_API_KEY       (required — Local Experience)
  GOOGLE_ROUTES_API_KEY     (required — Ground Transport)

This will use:
  * 1-2 SerpApi searches (flights + hotels)
  * 1-3 Google Places searches
  * 1-2 OpenWeather queries
  * 1-3 Google Routes lookups
  * ~$0.01-$0.05 of OpenAI tokens

Run:
  python scripts/smoke_end_to_end.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from dotenv import load_dotenv

from a2a_servers.local import build_app as build_local_a2a_app
from a2a_servers.transit import build_app as build_transit_a2a_app
from a2a_servers.travel import build_app as build_travel_a2a_app
from agents_concierge.agent import build_orchestrating_concierge
from agents_concierge.delegation import build_delegation_tools
from shared.a2a_client import AgentDiscovery
from shared.agent_runner import run_agent_task

load_dotenv()


REQUIRED_KEYS = (
    "OPENAI_API_KEY",
    "SERPAPI_API_KEY",
    "GOOGLE_PLACES_API_KEY",
    "OPENWEATHER_API_KEY",
    "GOOGLE_ROUTES_API_KEY",
)

LOCAL_PORT = 18001
TRAVEL_PORT = 18002
TRANSIT_PORT = 18004


def _check_keys() -> None:
    missing = [k for k in REQUIRED_KEYS if not os.environ.get(k)]
    if missing:
        print("Missing required env vars:")
        for k in missing:
            print(f"  - {k}")
        print("\nSet them in your .env or shell and re-run.")
        sys.exit(1)


@asynccontextmanager
async def _serve(app: Any, port: int, name: str):
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning", lifespan="on"
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
    task = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.05)
    else:
        task.cancel()
        raise RuntimeError(f"{name} uvicorn did not start")
    print(f"  {name} running on http://127.0.0.1:{port}")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except asyncio.TimeoutError:
            task.cancel()


async def main() -> None:
    _check_keys()

    print("Starting specialist agents...")
    local_app = build_local_a2a_app(public_url=f"http://127.0.0.1:{LOCAL_PORT}")
    travel_app = build_travel_a2a_app(public_url=f"http://127.0.0.1:{TRAVEL_PORT}")
    transit_app = build_transit_a2a_app(
        public_url=f"http://127.0.0.1:{TRANSIT_PORT}"
    )

    async with _serve(local_app, LOCAL_PORT, "local-experience") as local_url:
        async with _serve(travel_app, TRAVEL_PORT, "travel") as travel_url:
            async with _serve(
                transit_app, TRANSIT_PORT, "ground-transport"
            ) as transit_url:
                discovery = AgentDiscovery()
                try:
                    tools = build_delegation_tools(
                        travel_url=travel_url,
                        local_url=local_url,
                        transit_url=transit_url,
                        discovery=discovery,
                        timeout_seconds=180.0,
                    )
                    agent = build_orchestrating_concierge(tools)

                    prompt = (
                        "I'd like to visit Lisbon for 4 days, departing from ATL "
                        "on June 11 2026 and returning June 15. Budget around $1500. "
                        "I'm interested in shopping, history and beaches. Plan the whole trip: "
                        "find flights and a hotel, recommend things to do, check "
                        "the weather, and explain how to get from Lisbon Airport "
                        "to the hotel area and between the main attractions. "
                        "Compare transit, driving, and walking where useful."
                    )

                    print("\nPrompt:")
                    print(f"  {prompt}\n")
                    print("Running Concierge orchestration...")
                    print("=" * 70)

                    async with agent.mcp_servers[0]:
                        outcome = await run_agent_task(
                            agent, prompt, timeout_seconds=360.0, max_turns=18
                        )

                    print(str(outcome.result.final_output_as(str)))
                    print("=" * 70)
                    print(f"\nDone in {outcome.duration_seconds:.1f}s.")
                finally:
                    await discovery.aclose()


if __name__ == "__main__":
    asyncio.run(main())
