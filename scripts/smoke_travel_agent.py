"""Smoke test: Travel Agent running on its own against real services.

Exercises the full Travel Agent loop:
  * Spawns the Travel MCP server via stdio
  * Calls real SerpApi for flights + hotels
  * Uses real OpenAI to drive the agent

Requires:
  OPENAI_API_KEY   (required)
  SERPAPI_API_KEY  (required — flights + hotels)

Cost: ~1-2 SerpApi searches + ~$0.01 of OpenAI tokens

Run:
  python scripts/smoke_travel_agent.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()


REQUIRED_KEYS = ("OPENAI_API_KEY", "SERPAPI_API_KEY")


def _check_keys() -> None:
    missing = [k for k in REQUIRED_KEYS if not os.environ.get(k)]
    if missing:
        print("Missing required env vars:")
        for k in missing:
            print(f"  - {k}")
        print("\nSet them in your .env or shell and re-run.")
        sys.exit(1)


async def main() -> None:
    _check_keys()

    from agents_travel.agent import build_travel_agent, build_travel_mcp_server
    from shared.agent_runner import run_agent_task

    mcp_server = build_travel_mcp_server()
    agent = build_travel_agent(mcp_server=mcp_server)

    prompt = (
        "Search for round-trip flights from JFK to LHR departing 2026-08-01 "
        "and returning 2026-08-08. Also find hotels in London for those dates. "
        "Pick the cheapest flight and hotel you find and estimate the total cost."
    )

    print("Travel Agent smoke test")
    print("=" * 60)
    print(f"Prompt: {prompt}\n")

    async with mcp_server:
        outcome = await run_agent_task(agent, prompt, timeout_seconds=120.0, max_turns=8)

    print(outcome.result.final_output_as(str))
    print("=" * 60)
    print(f"Done in {outcome.duration_seconds:.1f}s.")


if __name__ == "__main__":
    asyncio.run(main())
