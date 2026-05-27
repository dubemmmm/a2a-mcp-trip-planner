"""Smoke test using FakeModel — no API keys needed.

Run with:  python scripts/smoke_fake.py
"""
import asyncio
import sys
sys.path.insert(0, "tests")  # so we can import the FakeModel helper

from agents import Agent
from shared.tests._helpers.fake_model import FakeModel
from shared.agent_runner import run_agent_task


async def main() -> None:
    # Build a tiny agent with no tools and a FakeModel that returns canned text.
    agent = Agent[None](
        name="smoke-test",
        instructions="be helpful",
        model=FakeModel(responses=["Hello from a fake model — wiring works."]),
    )
    outcome = await run_agent_task(agent, "say hi")
    print("Agent said:", outcome.result.final_output_as(str))
    print("Duration:", outcome.duration_seconds, "seconds")


if __name__ == "__main__":
    asyncio.run(main())