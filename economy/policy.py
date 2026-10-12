"""SpendPolicy: should the payer accept a quote?

A pure function of the quote and the payer's limits. Code enforces the caps;
the LLM only ever sees the outcome.

    decision = policy.decide(quote_mc=2_100, base_mc=400, limits=limits)

Limits that bind a single call, tightest first:

  * per call  — `max_call_mc`
  * per trip  — what is left of this planning session's budget
  * per day   — what is left of the wallet's daily cap
  * balance   — what the wallet actually holds

If the quote is over the tightest limit but the specialist's base price is
not, the policy counters once at that limit (the specialist may accept a
lower ceiling). Otherwise it rejects, naming the limit that bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RejectReason(str, Enum):
    OVER_CALL_CAP = "over_call_cap"
    TRIP_BUDGET_EXHAUSTED = "trip_budget_exhausted"
    DAILY_CAP = "daily_cap"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    DECLINED = "declined"  # the specialist would not quote under our limit
    BILLING_UNAVAILABLE = "billing_unavailable"


@dataclass(frozen=True)
class SpendLimits:
    """The payer's headroom at the moment of deciding, all in millicredits."""

    call_cap_mc: int
    trip_remaining_mc: int
    day_remaining_mc: int | None  # None = no daily cap
    available_mc: int

    def binding(self) -> tuple[int, RejectReason]:
        """The tightest limit and the reason it would reject with."""
        candidates = [
            (self.call_cap_mc, RejectReason.OVER_CALL_CAP),
            (self.trip_remaining_mc, RejectReason.TRIP_BUDGET_EXHAUSTED),
            (self.available_mc, RejectReason.INSUFFICIENT_FUNDS),
        ]
        if self.day_remaining_mc is not None:
            candidates.append((self.day_remaining_mc, RejectReason.DAILY_CAP))
        # min() keeps the first of equal limits, so ties report the order above.
        return min(candidates, key=lambda c: c[0])


@dataclass(frozen=True)
class Accept:
    quote_mc: int


@dataclass(frozen=True)
class Counter:
    max_mc: int


@dataclass(frozen=True)
class Reject:
    reason: RejectReason
    message: str


Decision = Accept | Counter | Reject


@dataclass(frozen=True)
class SpendPolicy:
    max_call_mc: int

    def __post_init__(self) -> None:
        if self.max_call_mc <= 0:
            raise ValueError("max_call_mc must be positive")

    def decide(
        self,
        *,
        quote_mc: int,
        base_mc: int,
        limits: SpendLimits,
        allow_counter: bool = True,
    ) -> Decision:
        if quote_mc <= 0:
            raise ValueError("quote_mc must be positive")
        limit_mc, reason = limits.binding()
        if quote_mc <= limit_mc:
            return Accept(quote_mc)
        if allow_counter and base_mc <= limit_mc and limit_mc > 0:
            return Counter(limit_mc)
        return Reject(
            reason,
            f"quote {credits(quote_mc)} exceeds {_LIMIT_NAMES[reason]} {credits(limit_mc)}",
        )


_LIMIT_NAMES = {
    RejectReason.OVER_CALL_CAP: "the per-call cap of",
    RejectReason.TRIP_BUDGET_EXHAUSTED: "the trip's remaining budget of",
    RejectReason.DAILY_CAP: "today's remaining allowance of",
    RejectReason.INSUFFICIENT_FUNDS: "the wallet balance of",
}


def credits(mc: int) -> str:
    """Render millicredits for people and prompts: 1840 -> '1.84 credits'."""
    return f"{mc / 1000:.2f} credits"
