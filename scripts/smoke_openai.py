# scripts/smoke_openai.py
"""Test that the OpenAI key works. Uses the real model, makes one short call.
Costs roughly $0.0001."""
import asyncio
from agents import Agent
from shared.agent_runner import run_agent_task


async def main() -> None:
    agent = Agent[None](
        name="openai-smoke",
        instructions="be very brief",
        model="gpt-5-mini",
    )
    outcome = await run_agent_task(agent, "say 'hello' and nothing else")
    print("OpenAI said:", outcome.result.final_output_as(str))


if __name__ == "__main__":
    asyncio.run(main())