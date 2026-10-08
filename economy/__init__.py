"""Agent economy: wallets, quotes, escrowed reservations, and the ledger.

Every A2A call between agents can cost credits. The ledger is the only
component allowed to move them; agents hold keys, not money.

Amounts are integer millicredits (1 credit = 1000 mc) so there is never any
float rounding.
"""

from .errors import (
    CaptureExceedsHold,
    DailyCapExceeded,
    IdempotencyConflict,
    InvalidRequest,
    InsufficientFunds,
    LedgerAuthError,
    LedgerError,
    LedgerForbidden,
    LedgerUnavailable,
    QuoteAlreadyReserved,
    QuoteExpired,
    ReservationNotHeld,
    TaskAlreadyBound,
    UnknownQuote,
    UnknownReservation,
    UnknownWallet,
)
from .ledger import Ledger
from .models import (
    MC_PER_CREDIT,
    Quote,
    Reservation,
    ReservationState,
    Transfer,
    TransferKind,
    Wallet,
    WalletStatus,
)

__all__ = [
    "MC_PER_CREDIT",
    "CaptureExceedsHold",
    "DailyCapExceeded",
    "IdempotencyConflict",
    "InvalidRequest",
    "InsufficientFunds",
    "Ledger",
    "LedgerAuthError",
    "LedgerError",
    "LedgerForbidden",
    "LedgerUnavailable",
    "Quote",
    "QuoteAlreadyReserved",
    "QuoteExpired",
    "Reservation",
    "ReservationNotHeld",
    "ReservationState",
    "TaskAlreadyBound",
    "Transfer",
    "TransferKind",
    "UnknownQuote",
    "UnknownReservation",
    "UnknownWallet",
    "Wallet",
    "WalletStatus",
]
