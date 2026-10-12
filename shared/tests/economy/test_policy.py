"""SpendPolicy: accept, counter once, or reject naming the binding limit."""

from __future__ import annotations

import pytest

from economy.policy import (
    Accept,
    Counter,
    Reject,
    RejectReason,
    SpendLimits,
    SpendPolicy,
    credits,
)

POLICY = SpendPolicy(max_call_mc=3_000)


def _limits(
    *,
    trip: int = 8_000,
    day: int | None = 50_000,
    available: int = 50_000,
    call: int = 3_000,
) -> SpendLimits:
    return SpendLimits(
        call_cap_mc=call, trip_remaining_mc=trip, day_remaining_mc=day, available_mc=available
    )


def test_accepts_a_quote_within_every_limit() -> None:
    assert POLICY.decide(quote_mc=1_500, base_mc=400, limits=_limits()) == Accept(1_500)


def test_accepts_a_quote_exactly_at_the_limit() -> None:
    assert POLICY.decide(quote_mc=3_000, base_mc=400, limits=_limits()) == Accept(3_000)


@pytest.mark.parametrize(
    ("limits", "counter_at"),
    [
        (_limits(), 3_000),  # per-call cap binds
        (_limits(trip=2_000), 2_000),
        (_limits(day=1_000), 1_000),
        (_limits(available=900), 900),
    ],
)
def test_counters_at_the_tightest_limit(limits: SpendLimits, counter_at: int) -> None:
    assert POLICY.decide(quote_mc=4_000, base_mc=400, limits=limits) == Counter(counter_at)


@pytest.mark.parametrize(
    ("limits", "reason"),
    [
        (_limits(call=300), RejectReason.OVER_CALL_CAP),
        (_limits(trip=300), RejectReason.TRIP_BUDGET_EXHAUSTED),
        (_limits(day=300), RejectReason.DAILY_CAP),
        (_limits(available=300), RejectReason.INSUFFICIENT_FUNDS),
        (_limits(trip=0), RejectReason.TRIP_BUDGET_EXHAUSTED),
    ],
)
def test_rejects_when_even_the_base_price_does_not_fit(
    limits: SpendLimits, reason: RejectReason
) -> None:
    decision = POLICY.decide(quote_mc=1_500, base_mc=400, limits=limits)
    assert isinstance(decision, Reject)
    assert decision.reason is reason
    assert "1.50 credits" in decision.message


def test_counter_is_offered_only_once() -> None:
    decision = POLICY.decide(
        quote_mc=2_500, base_mc=400, limits=_limits(trip=2_000), allow_counter=False
    )
    assert isinstance(decision, Reject)
    assert decision.reason is RejectReason.TRIP_BUDGET_EXHAUSTED


def test_no_daily_cap_means_day_never_binds() -> None:
    limits = _limits(day=None)
    assert limits.binding() == (3_000, RejectReason.OVER_CALL_CAP)


def test_ties_report_the_most_specific_cause_first() -> None:
    # call cap and trip budget both 3000 -> the call cap is named.
    assert _limits(trip=3_000).binding()[1] is RejectReason.OVER_CALL_CAP


def test_validation() -> None:
    with pytest.raises(ValueError):
        SpendPolicy(max_call_mc=0)
    with pytest.raises(ValueError):
        POLICY.decide(quote_mc=0, base_mc=0, limits=_limits())


@pytest.mark.parametrize(("mc", "text"), [(0, "0.00"), (1_840, "1.84"), (50_000, "50.00")])
def test_credits_rendering(mc: int, text: str) -> None:
    assert credits(mc) == f"{text} credits"
