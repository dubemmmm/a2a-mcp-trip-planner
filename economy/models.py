"""Typed records the ledger reads and writes.

These are the contract between the ledger, its HTTP service, and the client:
the service serializes them with `model_dump(mode="json")` and the client
rebuilds them with `model_validate`.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

MC_PER_CREDIT = 1000

# The issuer of daily allowances. Not a wallet row: allowances are minted, and
# their journal entries name it as `from_wallet`. A `None` wallet on a journal
# entry always means "the payer's hold".
TREASURY = "treasury"


class ReservationState(str, Enum):
    """A reservation changes state at most once, from HELD to a terminal state."""

    HELD = "held"
    CAPTURED = "captured"
    RELEASED = "released"
    EXPIRED = "expired"


class TransferKind(str, Enum):
    ALLOWANCE = "allowance"  # minted into a wallet
    RESERVE = "reserve"  # payer available -> payer reserved
    CAPTURE = "capture"  # payer reserved -> payee available
    REFUND = "refund"  # unused part of a capture, back to payer available
    RELEASE = "release"  # whole hold back to payer available
    EXPIRE = "expire"  # whole hold back to payer available, by the sweeper


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True)


class Wallet(_Record):
    wallet_id: str = Field(..., min_length=1)
    available_mc: int = Field(..., ge=0)
    reserved_mc: int = Field(..., ge=0)
    daily_cap_mc: int | None = Field(default=None, ge=0)


class Quote(_Record):
    """A time-limited offer, issued by the ledger at the payee's request."""

    quote_id: str
    payer: str
    payee: str
    skill_id: str
    amount_mc: int = Field(..., gt=0)
    expires_at: datetime
    created_at: datetime


class Reservation(_Record):
    """Credits held against one quote until captured, released or expired."""

    reservation_id: str
    quote_id: str
    payer: str
    payee: str
    skill_id: str
    amount_mc: int = Field(..., gt=0)
    captured_mc: int | None = Field(default=None, ge=0)
    trip_id: str | None = None
    a2a_task_id: str | None = None
    state: ReservationState
    expires_at: datetime
    created_at: datetime
    settled_at: datetime | None = None


class WalletStatus(Wallet):
    """A wallet plus what it has committed today (captures + live holds)."""

    spent_today_mc: int = Field(..., ge=0)


class Transfer(_Record):
    """One row of the append-only journal."""

    transfer_id: int
    idempotency_key: str
    kind: TransferKind
    from_wallet: str | None
    to_wallet: str | None
    amount_mc: int = Field(..., ge=0)
    reservation_id: str | None
    trip_id: str | None
    created_at: datetime
