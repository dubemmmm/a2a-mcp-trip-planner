"""Smoke test: Local Experience Agent running on its own against real services.

Exercises the full Local Experience Agent loop:
  * Spawns the Local MCP server via stdio
  * Calls real Google Places for attractions + restaurants
  * Calls real OpenWeather for the forecast
  * Uses real OpenAI to drive the agent

Requires:
  OPENAI_API_KEY          (required)
  GOOGLE_PLACES_API_KEY   (required — attractions + restaurants)
  OPENWEATHER_API_KEY     (required — weather forecast)

Cost: ~2-3 Google Places lookups + 1 OpenWeather query + ~$0.01 of OpenAI tokens

Run:
  python scripts/smoke_local_agent.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()


REQUIRED_KEYS = ("OPENAI_API_KEY", "GOOGLE_PLACES_API_KEY", "OPENWEATHER_API_KEY")


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

    from agents_local.agent import build_local_agent, build_local_mcp_server
    from shared.agent_runner import run_agent_task

    mcp_server = build_local_mcp_server()
    agent = build_local_agent(mcp_server=mcp_server)

    prompt = (
        "I'll be in Lisbon from June 15 to June 19 2026. "
        "I'm interested in history and food. "
        "Recommend a few must-see attractions, a couple of restaurants, "
        "and tell me what the weather will be like during my stay."
    )

    print("Local Experience Agent smoke test")
    print("=" * 60)
    print(f"Prompt: {prompt}\n")

    async with mcp_server:
        outcome = await run_agent_task(agent, prompt, timeout_seconds=120.0, max_turns=8)

    print(outcome.result.final_output_as(str))
    print("=" * 60)
    print(f"Done in {outcome.duration_seconds:.1f}s.")


if __name__ == "__main__":
    asyncio.run(main())
