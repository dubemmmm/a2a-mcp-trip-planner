"""PricingPolicy: quotes, metered charges and learned estimates."""

from __future__ import annotations

import pytest

from economy.pricing import (
    LOCAL_PRICE_SHEET,
    TRANSIT_PRICE_SHEET,
    TRAVEL_PRICE_SHEET,
    PriceSheet,
    PricingPolicy,
    SkillPrice,
    TokenRate,
    UnknownSkillPrice,
)
from shared.agent_runner import RunUsage

SHEET = PriceSheet(
    payee_wallet="travel",
    skills={
        "flights": SkillPrice(base_mc=400, max_mc=2_500, estimate_mc=1_100),
        "math": SkillPrice(base_mc=0, max_mc=100, estimate_mc=0),
    },
    token_rate=TokenRate(input_mc_per_mtok=25_000, output_mc_per_mtok=100_000),
    tool_call_mc={"search_flights": 200, "estimate_total_cost": 0},
    default_tool_call_mc=50,
)


def _usage(input_tokens: int = 0, output_tokens: int = 0, **tools: int) -> RunUsage:
    return RunUsage(
        requests=1, input_tokens=input_tokens, output_tokens=output_tokens, tool_calls=tools
    )


@pytest.fixture
def policy() -> PricingPolicy:
    return PricingPolicy(SHEET)


# ---------- quotes ----------


def test_quote_is_base_plus_estimate(policy: PricingPolicy) -> None:
    assert policy.quote("flights") == 1_500


def test_quote_is_clamped_to_max_and_base(policy: PricingPolicy) -> None:
    assert policy.quote("flights", load_multiplier=10) == 2_500
    assert policy.quote("flights", load_multiplier=0.01) == 400


def test_quote_is_never_zero(policy: PricingPolicy) -> None:
    # A ledger quote must be positive, even for a free-to-run skill.
    assert policy.quote("math") == 1


def test_quote_rounds_up(policy: PricingPolicy) -> None:
    assert policy.quote("flights", load_multiplier=1.0001) == 1_501


def test_quote_validation(policy: PricingPolicy) -> None:
    with pytest.raises(ValueError):
        policy.quote("flights", load_multiplier=0)
    with pytest.raises(UnknownSkillPrice):
        policy.quote("nope")


# ---------- metering ----------


def test_llm_cost_rounds_up_per_run(policy: PricingPolicy) -> None:
    assert policy.llm_cost_mc(_usage(input_tokens=20_000, output_tokens=2_000)) == 700
    assert policy.llm_cost_mc(_usage(input_tokens=1)) == 1
    assert policy.llm_cost_mc(_usage()) == 0


def test_api_cost_per_tool_with_default(policy: PricingPolicy) -> None:
    usage = _usage(search_flights=2, estimate_total_cost=3, mystery_tool=1)
    assert policy.api_cost_mc(usage) == 2 * 200 + 0 + 50


def test_charge_is_metered_cost_plus_base(policy: PricingPolicy) -> None:
    usage = _usage(input_tokens=20_000, output_tokens=2_000, search_flights=1)
    charge = policy.charge("flights", usage, ceiling_mc=2_000)
    assert (charge.base_mc, charge.llm_mc, charge.api_mc) == (400, 700, 200)
    assert charge.uncapped_mc == charge.charged_mc == 1_300


def test_charge_never_exceeds_the_ceiling(policy: PricingPolicy) -> None:
    usage = _usage(input_tokens=200_000, search_flights=5)
    charge = policy.charge("flights", usage, ceiling_mc=1_500)
    assert charge.uncapped_mc > 1_500
    assert charge.charged_mc == 1_500


def test_charge_validation(policy: PricingPolicy) -> None:
    with pytest.raises(ValueError):
        policy.charge("flights", _usage(), ceiling_mc=0)
    with pytest.raises(UnknownSkillPrice):
        policy.charge("nope", _usage(), ceiling_mc=10)


# ---------- learning ----------


def test_record_moves_the_estimate_toward_observed_cost() -> None:
    policy = PricingPolicy(SHEET, smoothing=0.5)
    observed = _usage(input_tokens=40_000)  # 1000 mc of LLM, no tools
    policy.record("flights", observed)
    assert policy.estimate_mc("flights") == 1_050  # halfway from 1100 to 1000
    for _ in range(30):
        policy.record("flights", observed)
    assert policy.estimate_mc("flights") == 1_000
    assert policy.quote("flights") == 1_400


def test_record_only_touches_its_skill(policy: PricingPolicy) -> None:
    policy.record("flights", _usage(input_tokens=1_000_000))
    assert policy.estimate_mc("math") == 0


def test_estimates_are_per_policy_instance() -> None:
    first, second = PricingPolicy(SHEET), PricingPolicy(SHEET)
    first.record("flights", _usage())
    assert second.estimate_mc("flights") == 1_100


def test_smoothing_validation() -> None:
    for bad in (0, -0.1, 1.5):
        with pytest.raises(ValueError):
            PricingPolicy(SHEET, smoothing=bad)


# ---------- sheets ----------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_mc": -1, "max_mc": 10, "estimate_mc": 0},
        {"base_mc": 0, "max_mc": 0, "estimate_mc": 0},
        {"base_mc": 20, "max_mc": 10, "estimate_mc": 0},
        {"base_mc": 0, "max_mc": 10, "estimate_mc": -1},
        {"base_mc": 0, "max_mc": 10, "estimate_mc": 0, "quote_ttl_seconds": 0},
    ],
)
def test_skill_price_validation(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        SkillPrice(**kwargs)  # type: ignore[arg-type]


def test_sheet_needs_skills_and_has_a_default() -> None:
    assert SHEET.default_skill == "flights"
    with pytest.raises(ValueError):
        PriceSheet(payee_wallet="x", skills={}, token_rate=SHEET.token_rate)


@pytest.mark.parametrize("sheet", [TRAVEL_PRICE_SHEET, LOCAL_PRICE_SHEET, TRANSIT_PRICE_SHEET])
def test_shipped_sheets_quote_inside_their_bounds(sheet: PriceSheet) -> None:
    policy = PricingPolicy(sheet)
    for skill_id, price in sheet.skills.items():
        assert price.base_mc <= policy.quote(skill_id) <= price.max_mc
        # Every priced tool is one the sheet knows the cost of.
        assert skill_id in sheet.tool_call_mc
