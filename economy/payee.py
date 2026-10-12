"""The specialist's side of a paid call: quote, check payment, settle.

The A2A executor drives this; nothing here knows about A2A.

Trust model: the specialist does not authenticate its callers. It does not
need to — the ledger already did. Only the real payer can reserve against a
quote (the ledger checks the payer's key), so a held reservation naming this
agent as payee *is* the proof of payment. A caller who lies about who it is
can only produce a quote nobody can reserve.

Every reservation pays for exactly one task (`bind_task` is single-use), so
replaying a reservation id gets refused rather than paid twice.
"""

from __future__ import annotations

import structlog

from shared.agent_runner import RunUsage

from .client import LedgerClient
from .errors import LedgerError, LedgerUnavailable
from .models import Quote, Reservation, ReservationState
from .pricing import Charge, PricingPolicy
from .protocol import QuoteDeclined, QuoteRequest

_LOGGER = structlog.get_logger(__name__)


class PaymentRefused(Exception):
    """The task's payment is missing or does not cover it. Do not run it."""


class Payee:
    """Quotes and settles for one priced agent."""

    def __init__(self, ledger: LedgerClient, pricing: PricingPolicy) -> None:
        if ledger.agent_id != pricing.sheet.payee_wallet:
            raise ValueError(
                f"ledger client acts as {ledger.agent_id!r} but the price sheet "
                f"pays {pricing.sheet.payee_wallet!r}"
            )
        self.ledger = ledger
        self.pricing = pricing

    @property
    def wallet(self) -> str:
        return self.pricing.sheet.payee_wallet

    async def make_quote(self, request: QuoteRequest, *, skill_id: str) -> Quote | QuoteDeclined:
        """Offer a ceiling for `skill_id`, honouring a counter-offer if we can."""
        price = self.pricing.price(skill_id)
        amount_mc = self.pricing.quote(skill_id)
        if request.max_mc is not None and amount_mc > request.max_mc:
            if request.max_mc < max(price.base_mc, 1):
                return QuoteDeclined(
                    "below_base",
                    f"{skill_id} cannot be done for less than {price.base_mc} mc",
                )
            amount_mc = request.max_mc
        try:
            return await self.ledger.create_quote(
                payer=request.payer,
                skill_id=skill_id,
                amount_mc=amount_mc,
                ttl_seconds=price.quote_ttl_seconds,
            )
        except LedgerUnavailable as e:
            return QuoteDeclined("billing_unavailable", str(e))
        except LedgerError as e:
            return QuoteDeclined(e.code, str(e))

    async def accept(
        self, reservation_id: str, *, skill_id: str | None, task_id: str
    ) -> Reservation:
        """Verify the payment for a task and bind it, or raise PaymentRefused."""
        try:
            reservation = await self.ledger.get_reservation(reservation_id)
            if reservation.payee != self.wallet:
                raise PaymentRefused(f"reservation {reservation_id} does not pay {self.wallet}")
            if reservation.state is not ReservationState.HELD:
                raise PaymentRefused(f"reservation {reservation_id} is {reservation.state.value}")
            if skill_id is not None and skill_id != reservation.skill_id:
                raise PaymentRefused(
                    f"reservation {reservation_id} pays for {reservation.skill_id}, not {skill_id}"
                )
            return await self.ledger.bind_task(reservation_id, task_id)
        except LedgerError as e:
            raise PaymentRefused(f"payment could not be verified: {e}") from e

    async def capture(self, reservation: Reservation, usage: RunUsage) -> Charge:
        """Charge the metered cost, capped at the hold. Raises LedgerError."""
        charge = self.pricing.charge(reservation.skill_id, usage, ceiling_mc=reservation.amount_mc)
        await self.ledger.capture(reservation.reservation_id, charge.charged_mc)
        self.pricing.record(reservation.skill_id, usage)
        return charge

    async def release(self, reservation_id: str) -> None:
        """Return the hold after failed work. Never raises: the hold also expires."""
        try:
            await self.ledger.release(reservation_id)
        except LedgerError as e:
            _LOGGER.warning("payee_release_failed", reservation_id=reservation_id, error=str(e))
