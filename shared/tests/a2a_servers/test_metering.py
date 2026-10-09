"""The executor meters priced runs and attaches the charge to the artifact.

Same harness as `test_live_a2a.py`: a real uvicorn server, the official A2A
client, and a FakeModel instead of OpenAI — here one that reports token
usage, so the charge is predictable.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

import pytest
from a2a.client import create_client
from a2a.types import AgentSkill, Message, Part, Role, SendMessageRequest
from agents import Agent
from agents.usage import Usage
from google.protobuf.json_format import MessageToDict

from _helpers.fake_model import FakeModel
from a2a_servers.scaffolding import build_a2a_app, build_agent_card
from economy.pricing import PriceSheet, PricingPolicy, SkillPrice, TokenRate

from .test_live_a2a import _serve

PORT = 54331

SHEET = PriceSheet(
    payee_wallet="echo",
    skills={
        "echo": SkillPrice(base_mc=400, max_mc=2_500, estimate_mc=1_100),
        "shout": SkillPrice(base_mc=100, max_mc=300, estimate_mc=100),
    },
    token_rate=TokenRate(input_mc_per_mtok=25_000, output_mc_per_mtok=100_000),
)
# 20k input + 2k output tokens = 500 + 200 = 700 mc of LLM cost.
RUN_USAGE = Usage(requests=1, input_tokens=20_000, output_tokens=2_000, total_tokens=22_000)

Serve = Callable[[PricingPolicy | None], AbstractAsyncContextManager[str]]


@pytest.fixture
def serve(monkeypatch: pytest.MonkeyPatch) -> Serve:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ENVIRONMENT", "test")

    def start(pricing: PricingPolicy | None) -> AbstractAsyncContextManager[str]:
        fake = FakeModel(responses=["echoed"], usage=RUN_USAGE)
        agent = Agent[None](name="echo", instructions="echo back", model=fake)
        card = build_agent_card(
            name="echo-test",
            description="test agent",
            url=f"http://localhost:{PORT}",
            skills=[
                AgentSkill(id="echo", name="echo", description="echo", tags=["t"]),
                AgentSkill(id="shout", name="shout", description="shout", tags=["t"]),
            ],
        )
        return _serve(build_a2a_app(agent, card, pricing=pricing), PORT)

    return start


async def _artifact_metadata(url: str, skill_id: str | None) -> list[dict[str, Any]]:
    message = Message(message_id="m-1", role=Role.ROLE_USER, parts=[Part(text="hello")])
    if skill_id is not None:
        message.metadata.update({"skill_id": skill_id})
    client = await create_client(url)
    found: list[dict[str, Any]] = []
    try:
        async for event in client.send_message(SendMessageRequest(message=message)):
            update = getattr(event, "artifact_update", None)
            if update is not None and update.artifact.parts:
                found.append(MessageToDict(update.artifact.metadata))
            task = getattr(event, "task", None)
            if task is not None:
                found.extend(MessageToDict(a.metadata) for a in task.artifacts)
    finally:
        await client.close()
    return found


async def _economy(url: str, skill_id: str | None) -> dict[str, Any]:
    metadata = await _artifact_metadata(url, skill_id)
    assert metadata, "expected the response artifact"
    return metadata[-1]["economy"]  # type: ignore[no-any-return]


async def test_priced_run_attaches_metered_charge(serve: Serve) -> None:
    async with serve(PricingPolicy(SHEET)) as url:
        economy = await _economy(url, "echo")

    assert economy["status"] == "metered"
    assert economy["usage"]["input_tokens"] == 20_000
    charge = economy["charge"]
    assert charge["skill_id"] == "echo"
    assert (charge["base_mc"], charge["llm_mc"], charge["api_mc"]) == (400, 700, 0)
    assert charge["ceiling_mc"] == 1_500  # base 400 + estimate 1100
    assert charge["charged_mc"] == 1_100


async def test_charge_is_capped_at_the_skill_ceiling(serve: Serve) -> None:
    async with serve(PricingPolicy(SHEET)) as url:
        charge = (await _economy(url, "shout"))["charge"]
    assert charge["ceiling_mc"] == 200
    assert charge["charged_mc"] == 200  # 100 base + 700 metered, capped


@pytest.mark.parametrize("skill_id", [None, "not-a-priced-skill"])
async def test_unknown_or_missing_skill_is_priced_as_the_default(
    serve: Serve, skill_id: str | None
) -> None:
    async with serve(PricingPolicy(SHEET)) as url:
        charge = (await _economy(url, skill_id))["charge"]
    assert charge["skill_id"] == "echo"


async def test_each_run_updates_the_estimate(serve: Serve) -> None:
    policy = PricingPolicy(SHEET)
    async with serve(policy) as url:
        await _economy(url, "echo")
    # One observed run of 700 mc pulls the 1100 mc estimate down.
    assert policy.estimate_mc("echo") < 1_100


async def test_unpriced_agent_attaches_no_economy_metadata(serve: Serve) -> None:
    async with serve(None) as url:
        metadata = await _artifact_metadata(url, "echo")
    assert metadata
    assert all("economy" not in m for m in metadata)
