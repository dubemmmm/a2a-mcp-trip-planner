"""Typed ledger errors.

Each error carries a stable `code` and the HTTP status the service maps it
to. The client maps the code back to the same class, so callers catch the
same exceptions whether they use the `Ledger` in-process or over HTTP.
"""

from __future__ import annotations


class LedgerError(Exception):
    """Base class for every ledger failure."""

    code = "ledger_error"
    http_status = 500


class InvalidRequest(LedgerError):
    """The request body or parameters were malformed."""

    code = "invalid_request"
    http_status = 422


class LedgerAuthError(LedgerError):
    """Missing or invalid agent credentials."""

    code = "unauthorized"
    http_status = 401


class LedgerForbidden(LedgerError):
    """Authenticated, but not allowed to act on this wallet or reservation."""

    code = "forbidden"
    http_status = 403


class LedgerUnavailable(LedgerError):
    """The ledger could not be reached. Callers must fail closed."""

    code = "ledger_unavailable"
    http_status = 503


class UnknownWallet(LedgerError):
    code = "unknown_wallet"
    http_status = 404


class UnknownQuote(LedgerError):
    code = "unknown_quote"
    http_status = 404


class UnknownReservation(LedgerError):
    code = "unknown_reservation"
    http_status = 404


class InsufficientFunds(LedgerError):
    code = "insufficient_funds"
    http_status = 402


class DailyCapExceeded(LedgerError):
    code = "daily_cap_exceeded"
    http_status = 402


class QuoteExpired(LedgerError):
    code = "quote_expired"
    http_status = 410


class QuoteAlreadyReserved(LedgerError):
    """The quote was reserved by an earlier call with different parameters."""

    code = "quote_already_reserved"
    http_status = 409


class ReservationNotHeld(LedgerError):
    """The reservation already reached a terminal state (or just expired)."""

    code = "reservation_not_held"
    http_status = 409


class CaptureExceedsHold(LedgerError):
    code = "capture_exceeds_hold"
    http_status = 409


class TaskAlreadyBound(LedgerError):
    """The reservation already pays for a different A2A task."""

    code = "task_already_bound"
    http_status = 409


class IdempotencyConflict(LedgerError):
    """A repeated operation disagreed with the one already recorded."""

    code = "idempotency_conflict"
    http_status = 409


ERRORS_BY_CODE: dict[str, type[LedgerError]] = {
    cls.code: cls
    for cls in (
        LedgerError,
        InvalidRequest,
        LedgerAuthError,
        LedgerForbidden,
        LedgerUnavailable,
        UnknownWallet,
        UnknownQuote,
        UnknownReservation,
        InsufficientFunds,
        DailyCapExceeded,
        QuoteExpired,
        QuoteAlreadyReserved,
        ReservationNotHeld,
        CaptureExceedsHold,
        TaskAlreadyBound,
        IdempotencyConflict,
    )
}
