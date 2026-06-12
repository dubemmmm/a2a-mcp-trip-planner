"""Smoke test: Ground Transport Agent running on its own against real services.

Exercises the full Ground Transport Agent loop:
  * Spawns the Transit MCP server via stdio
  * Calls the real Google Routes API for routes + mode comparison
  * Uses real OpenAI to drive the agent

Requires:
  OPENAI_API_KEY          (required)
  GOOGLE_ROUTES_API_KEY   (required — routing)

Cost: ~3-4 Google Routes lookups + ~$0.01 of OpenAI tokens

Run:
  python scripts/smoke_transit_agent.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()


REQUIRED_KEYS = ("OPENAI_API_KEY", "GOOGLE_ROUTES_API_KEY")


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

    from agents_transit.agent import build_transit_agent, build_transit_mcp_server
    from shared.agent_runner import run_agent_task

    mcp_server = build_transit_mcp_server()
    agent = build_transit_agent(mcp_server=mcp_server)

    prompt = (
        "I'm staying near Rossio Square in Lisbon and want to visit Belem "
        "Tower. What's the fastest way to get there — compare public transit, "
        "driving, and walking — and roughly how long does each take?"
    )

    print("Ground Transport Agent smoke test")
    print("=" * 60)
    print(f"Prompt: {prompt}\n")

    async with mcp_server:
        outcome = await run_agent_task(agent, prompt, timeout_seconds=120.0, max_turns=8)

    print(outcome.result.final_output_as(str))
    print("=" * 60)
    print(f"Done in {outcome.duration_seconds:.1f}s.")


if __name__ == "__main__":
    asyncio.run(main())
