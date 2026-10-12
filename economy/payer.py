"""The Concierge's side of a paid call: quote, decide, reserve, settle.

One `Payer` per trip (one planning session). It tracks what the trip has
committed — live holds plus captured charges — so the per-trip budget holds
even when the Concierge delegates to several specialists at once:
`authorize` runs under a lock, so two parallel calls cannot both claim the
last of the budget.

Nothing here knows about A2A. The caller supplies `request_quote`, a
function that asks the specialist for a quote (with an optional counter
limit) and returns its reply.

Every failure to pay comes back as a `Rejection` rather than an exception,
so the delegation tool can tell the LLM "not called, and why" instead of
failing the whole plan.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import structlog

from .client import LedgerClient
from .errors import DailyCapExceeded, InsufficientFunds, LedgerError, LedgerUnavailable
from .models import Reservation, ReservationState
from .policy import Accept, Counter, Reject, RejectReason, SpendLimits, SpendPolicy, credits
from .protocol import QuoteDeclined, QuoteOffer

_LOGGER = structlog.get_logger(__name__)

RequestQuote = Callable[[int | None], Awaitable[QuoteOffer | QuoteDeclined]]


@dataclass(frozen=True)
class Authorization:
    """Credits are held; the task may be submitted with this reservation."""

    reservation: Reservation
    counter_offered: bool


@dataclass(frozen=True)
class Rejection:
    """The call was not made. `message` is safe to show the LLM."""

    reason: RejectReason | str
    message: str
    quote_mc: int | None = None


@dataclass(frozen=True)
class Settlement:
    """How a reservation ended, as far as the payer could confirm."""

    reservation_id: str
    state: ReservationState | None  # None: the ledger could not be reached
    charged_mc: int


class Payer:
    def __init__(
        self,
        ledger: LedgerClient,
        policy: SpendPolicy,
        *,
        trip_id: str,
        trip_budget_mc: int,
    ) -> None:
        if trip_budget_mc < 0:
            raise ValueError("trip_budget_mc must be >= 0")
        self.ledger = ledger
        self.policy = policy
        self.trip_id = trip_id
        self.trip_budget_mc = trip_budget_mc
        self._committed_mc = 0
        self._lock = asyncio.Lock()

    @property
    def wallet(self) -> str:
        return self.ledger.agent_id

    @property
    def committed_mc(self) -> int:
        """Live holds plus captured charges for this trip."""
        return self._committed_mc

    @property
    def trip_remaining_mc(self) -> int:
        return max(0, self.trip_budget_mc - self._committed_mc)

    async def authorize(
        self, *, base_mc: int, request_quote: RequestQuote
    ) -> Authorization | Rejection:
        """Quote, decide (countering at most once) and reserve."""
        async with self._lock:
            try:
                return await self._authorize(base_mc=base_mc, request_quote=request_quote)
            except LedgerUnavailable as e:
                return Rejection(RejectReason.BILLING_UNAVAILABLE, f"billing unavailable: {e}")

    async def _authorize(
        self, *, base_mc: int, request_quote: RequestQuote
    ) -> Authorization | Rejection:
        reply = await request_quote(None)
        if isinstance(reply, QuoteDeclined):
            return Rejection(
                RejectReason.DECLINED, f"specialist declined to quote: {reply.message}"
            )

        limits = await self._limits()
        decision = self.policy.decide(quote_mc=reply.amount_mc, base_mc=base_mc, limits=limits)
        countered = False
        if isinstance(decision, Counter):
            countered = True
            reply = await request_quote(decision.max_mc)
            if isinstance(reply, QuoteDeclined):
                return Rejection(
                    RejectReason.DECLINED,
                    f"specialist would not work for {credits(decision.max_mc)}: {reply.message}",
                )
            decision = self.policy.decide(
                quote_mc=reply.amount_mc, base_mc=base_mc, limits=limits, allow_counter=False
            )
        if isinstance(decision, Reject):
            return Rejection(decision.reason, decision.message, quote_mc=reply.amount_mc)
        assert isinstance(decision, Accept)

        try:
            reservation = await self.ledger.reserve(reply.quote_id, trip_id=self.trip_id)
        except InsufficientFunds as e:
            return Rejection(RejectReason.INSUFFICIENT_FUNDS, str(e), quote_mc=reply.amount_mc)
        except DailyCapExceeded as e:
            return Rejection(RejectReason.DAILY_CAP, str(e), quote_mc=reply.amount_mc)
        except LedgerUnavailable:
            raise
        except LedgerError as e:
            return Rejection(e.code, f"could not reserve: {e}", quote_mc=reply.amount_mc)

        self._committed_mc += reservation.amount_mc
        _LOGGER.info(
            "payer_reserved",
            trip_id=self.trip_id,
            reservation_id=reservation.reservation_id,
            payee=reservation.payee,
            amount_mc=reservation.amount_mc,
            countered=countered,
            trip_committed_mc=self._committed_mc,
        )
        return Authorization(reservation=reservation, counter_offered=countered)

    async def _limits(self) -> SpendLimits:
        wallet = await self.ledger.get_wallet()
        day_remaining = (
            None
            if wallet.daily_cap_mc is None
            else max(0, wallet.daily_cap_mc - wallet.spent_today_mc)
        )
        return SpendLimits(
            call_cap_mc=self.policy.max_call_mc,
            trip_remaining_mc=self.trip_remaining_mc,
            day_remaining_mc=day_remaining,
            available_mc=wallet.available_mc,
        )

    async def settle(self, authorization: Authorization) -> Settlement:
        """Finish a call: release the hold if the specialist never settled it.

        Called after every paid call, success or not. On success the
        specialist has already captured; otherwise (failure, timeout) the
        payer releases, and a late capture from the specialist is refused.
        """
        held = authorization.reservation
        try:
            current = await self.ledger.get_reservation(held.reservation_id)
            if current.state is ReservationState.HELD:
                current = await self.ledger.release(held.reservation_id)
        except LedgerError as e:
            # Unconfirmed: keep counting the full hold against the trip, and
            # let the ledger's expiry return it if nobody settles.
            _LOGGER.warning(
                "payer_settle_unconfirmed", reservation_id=held.reservation_id, error=str(e)
            )
            return Settlement(held.reservation_id, None, 0)

        charged = 0
        if current.state is ReservationState.CAPTURED and current.captured_mc is not None:
            charged = current.captured_mc
        async with self._lock:
            self._committed_mc += charged - held.amount_mc
        _LOGGER.info(
            "payer_settled",
            trip_id=self.trip_id,
            reservation_id=held.reservation_id,
            state=current.state.value,
            charged_mc=charged,
            trip_committed_mc=self._committed_mc,
        )
        return Settlement(held.reservation_id, current.state, charged)
