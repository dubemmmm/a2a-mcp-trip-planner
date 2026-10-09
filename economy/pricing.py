"""Cost-plus pricing for specialist agents.

A specialist charges roughly what a call costs it, plus a base fee:

    quote  = clamp(ceil((base + estimated_variable) * load), base, max)
    charge = min(quote, base + metered_llm + metered_api)

The quote is a ceiling the payer agrees to up front; the charge is settled
after the run from what was actually metered (`RunUsage`), so the payer never
pays more than it saw and usually pays less.

`estimated_variable` starts at the sheet's `estimate_mc` and then tracks an
exponential moving average of metered variable cost per skill, so quotes
follow what the agent really spends.

Everything is integer millicredits. The rates in the sheets are simulated
credits, not real prices: they are placeholders to tune once metered runs
show what a typical call consumes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from shared.agent_runner import RunUsage

_PER_MILLION = 1_000_000


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


@dataclass(frozen=True)
class SkillPrice:
    """The published price terms of one skill."""

    base_mc: int
    max_mc: int
    estimate_mc: int  # starting guess of variable (LLM + API) cost per call
    quote_ttl_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.base_mc < 0 or self.estimate_mc < 0:
            raise ValueError("base_mc and estimate_mc must be >= 0")
        if self.max_mc <= 0 or self.max_mc < self.base_mc:
            raise ValueError("max_mc must be positive and >= base_mc")
        if self.quote_ttl_seconds <= 0:
            raise ValueError("quote_ttl_seconds must be positive")


@dataclass(frozen=True)
class TokenRate:
    """Millicredits per million model tokens."""

    input_mc_per_mtok: int
    output_mc_per_mtok: int


@dataclass(frozen=True)
class PriceSheet:
    """Everything one specialist needs to quote and charge."""

    payee_wallet: str
    skills: Mapping[str, SkillPrice]
    token_rate: TokenRate
    # External-API cost per MCP tool call, by tool name. Tools that are pure
    # computation are listed at 0; unlisted tools cost `default_tool_call_mc`.
    tool_call_mc: Mapping[str, int] = field(default_factory=dict)
    default_tool_call_mc: int = 0

    def __post_init__(self) -> None:
        if not self.skills:
            raise ValueError("a price sheet needs at least one skill")

    @property
    def default_skill(self) -> str:
        """The skill to price by when a request names none."""
        return next(iter(self.skills))


class UnknownSkillPrice(KeyError):
    """The skill has no price on this sheet."""


class Charge(BaseModel):
    """What one completed task costs, and why."""

    model_config = ConfigDict(frozen=True)

    skill_id: str
    base_mc: int = Field(..., ge=0)
    llm_mc: int = Field(..., ge=0)
    api_mc: int = Field(..., ge=0)
    ceiling_mc: int = Field(..., gt=0)
    charged_mc: int = Field(..., ge=0)

    @property
    def uncapped_mc(self) -> int:
        return self.base_mc + self.llm_mc + self.api_mc


class PricingPolicy:
    """Quotes ceilings, computes charges, and learns per-skill cost estimates.

    Holds in-memory state (the moving averages), so keep one instance per
    agent process.
    """

    def __init__(self, sheet: PriceSheet, *, smoothing: float = 0.3) -> None:
        if not 0 < smoothing <= 1:
            raise ValueError("smoothing must be in (0, 1]")
        self.sheet = sheet
        self._smoothing = smoothing
        self._estimates: dict[str, float] = {
            skill_id: float(price.estimate_mc) for skill_id, price in sheet.skills.items()
        }

    def price(self, skill_id: str) -> SkillPrice:
        try:
            return self.sheet.skills[skill_id]
        except KeyError:
            raise UnknownSkillPrice(skill_id) from None

    def estimate_mc(self, skill_id: str) -> int:
        self.price(skill_id)
        return round(self._estimates[skill_id])

    def quote(self, skill_id: str, *, load_multiplier: float = 1.0) -> int:
        """The ceiling to offer for one call of `skill_id`."""
        if load_multiplier <= 0:
            raise ValueError("load_multiplier must be positive")
        price = self.price(skill_id)
        raw = (price.base_mc + self.estimate_mc(skill_id)) * load_multiplier
        return max(price.base_mc, min(price.max_mc, math.ceil(raw)), 1)

    def llm_cost_mc(self, usage: RunUsage) -> int:
        rate = self.sheet.token_rate
        return _ceil_div(
            usage.input_tokens * rate.input_mc_per_mtok
            + usage.output_tokens * rate.output_mc_per_mtok,
            _PER_MILLION,
        )

    def api_cost_mc(self, usage: RunUsage) -> int:
        return sum(
            calls * self.sheet.tool_call_mc.get(tool, self.sheet.default_tool_call_mc)
            for tool, calls in usage.tool_calls.items()
        )

    def charge(self, skill_id: str, usage: RunUsage, *, ceiling_mc: int) -> Charge:
        """Settle one task: metered cost plus base, capped at the agreed ceiling."""
        if ceiling_mc <= 0:
            raise ValueError("ceiling_mc must be positive")
        price = self.price(skill_id)
        llm_mc, api_mc = self.llm_cost_mc(usage), self.api_cost_mc(usage)
        return Charge(
            skill_id=skill_id,
            base_mc=price.base_mc,
            llm_mc=llm_mc,
            api_mc=api_mc,
            ceiling_mc=ceiling_mc,
            charged_mc=min(ceiling_mc, price.base_mc + llm_mc + api_mc),
        )

    def record(self, skill_id: str, usage: RunUsage) -> None:
        """Fold one run's metered variable cost into the skill's estimate."""
        self.price(skill_id)
        observed = self.llm_cost_mc(usage) + self.api_cost_mc(usage)
        previous = self._estimates[skill_id]
        self._estimates[skill_id] = previous + self._smoothing * (observed - previous)


# ------------------------------------------------------------- price sheets
#
# Simulated rates. Token rates assume the specialists' default model; tool
# costs approximate one external API request per call (compare_modes makes
# one request per travel mode).

SPECIALIST_TOKEN_RATE = TokenRate(input_mc_per_mtok=25_000, output_mc_per_mtok=100_000)

TRAVEL_PRICE_SHEET = PriceSheet(
    payee_wallet="travel",
    skills={
        "search_flights": SkillPrice(base_mc=400, max_mc=2_500, estimate_mc=1_100),
        "search_hotels": SkillPrice(base_mc=300, max_mc=2_000, estimate_mc=1_000),
        "estimate_total_cost": SkillPrice(base_mc=100, max_mc=500, estimate_mc=200),
    },
    token_rate=SPECIALIST_TOKEN_RATE,
    tool_call_mc={"search_flights": 200, "search_hotels": 200, "estimate_total_cost": 0},
)

LOCAL_PRICE_SHEET = PriceSheet(
    payee_wallet="local",
    skills={
        "find_attractions": SkillPrice(base_mc=300, max_mc=2_000, estimate_mc=900),
        "recommend_restaurants": SkillPrice(base_mc=300, max_mc=2_000, estimate_mc=900),
        "get_weather_forecast": SkillPrice(base_mc=100, max_mc=800, estimate_mc=300),
    },
    token_rate=SPECIALIST_TOKEN_RATE,
    tool_call_mc={
        "find_attractions": 100,
        "recommend_restaurants": 100,
        "get_weather_forecast": 20,
    },
)

TRANSIT_PRICE_SHEET = PriceSheet(
    payee_wallet="transit",
    skills={
        "get_route": SkillPrice(base_mc=200, max_mc=1_500, estimate_mc=600),
        "compare_modes": SkillPrice(base_mc=300, max_mc=2_000, estimate_mc=900),
        "total_travel_minutes": SkillPrice(base_mc=100, max_mc=500, estimate_mc=150),
    },
    token_rate=SPECIALIST_TOKEN_RATE,
    tool_call_mc={"get_route": 50, "compare_modes": 150, "total_travel_minutes": 0},
)
