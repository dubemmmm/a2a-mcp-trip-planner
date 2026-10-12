"""The paid-delegation handshake, as plain dicts carried in A2A metadata.

No new JSON-RPC methods: everything rides on standard `message/send`.

1. Quote request — a message whose metadata carries

       {"skill_id": "...", "pricing": {"phase": "quote", "payer": "concierge", "max_mc": 2000}}

   (`max_mc` only on a counter-offer). The specialist answers with a plain
   Message (not a Task) holding one data part:

       {"quote": {"quote_id": "q_…", "skill_id": "...", "amount_mc": 1500,
                  "payee": "travel", "expires_at": "…"}}
   or
       {"declined": {"reason": "below_base", "message": "..."}}

2. Paid task — the normal task message, with metadata

       {"skill_id": "...", "pricing": {"reservation_id": "r_…"}}

3. Receipt — the response artifact's metadata carries
   `{"economy": {"status": "captured" | "unpaid" | "metered", ...}}`.

These helpers only build and parse dicts, so both sides share one source of
truth and neither needs to know about protobuf.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PRICING_KEY = "pricing"
PHASE_QUOTE = "quote"


class ProtocolError(ValueError):
    """A pricing message did not have the expected shape."""


@dataclass(frozen=True)
class QuoteRequest:
    payer: str
    max_mc: int | None = None


@dataclass(frozen=True)
class QuoteOffer:
    quote_id: str
    skill_id: str
    amount_mc: int
    payee: str


@dataclass(frozen=True)
class QuoteDeclined:
    reason: str
    message: str


# ------------------------------------------------------------------ requests


def quote_request_metadata(skill_id: str, request: QuoteRequest) -> dict[str, Any]:
    pricing: dict[str, Any] = {"phase": PHASE_QUOTE, "payer": request.payer}
    if request.max_mc is not None:
        pricing["max_mc"] = request.max_mc
    return {"skill_id": skill_id, PRICING_KEY: pricing}


def paid_task_metadata(reservation_id: str) -> dict[str, Any]:
    return {PRICING_KEY: {"reservation_id": reservation_id}}


def pricing_section(metadata: dict[str, Any]) -> dict[str, Any]:
    section = metadata.get(PRICING_KEY)
    return section if isinstance(section, dict) else {}


def parse_quote_request(metadata: dict[str, Any]) -> QuoteRequest | None:
    """The quote request in a message's metadata, or None if it is not one."""
    section = pricing_section(metadata)
    if section.get("phase") != PHASE_QUOTE:
        return None
    payer = section.get("payer")
    if not isinstance(payer, str) or not payer:
        raise ProtocolError("a quote request must name its payer")
    max_mc = section.get("max_mc")
    return QuoteRequest(payer=payer, max_mc=None if max_mc is None else _whole(max_mc))


def parse_reservation_id(metadata: dict[str, Any]) -> str | None:
    value = pricing_section(metadata).get("reservation_id")
    return value if isinstance(value, str) and value else None


# ------------------------------------------------------------------- replies


def offer_payload(offer: QuoteOffer, *, expires_at: str) -> dict[str, Any]:
    return {
        "quote": {
            "quote_id": offer.quote_id,
            "skill_id": offer.skill_id,
            "amount_mc": offer.amount_mc,
            "payee": offer.payee,
            "expires_at": expires_at,
        }
    }


def declined_payload(declined: QuoteDeclined) -> dict[str, Any]:
    return {"declined": {"reason": declined.reason, "message": declined.message}}


def parse_quote_reply(payload: dict[str, Any]) -> QuoteOffer | QuoteDeclined:
    try:
        if "quote" in payload:
            quote = payload["quote"]
            return QuoteOffer(
                quote_id=str(quote["quote_id"]),
                skill_id=str(quote["skill_id"]),
                amount_mc=_whole(quote["amount_mc"]),
                payee=str(quote["payee"]),
            )
        if "declined" in payload:
            declined = payload["declined"]
            return QuoteDeclined(reason=str(declined["reason"]), message=str(declined["message"]))
    except (KeyError, TypeError) as e:
        raise ProtocolError(f"malformed quote reply: {e!r}") from e
    raise ProtocolError("quote reply has neither 'quote' nor 'declined'")


def _whole(value: Any) -> int:
    # protobuf Struct carries numbers as floats.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"expected a millicredit amount, got {value!r}")
    if float(value) != int(value) or value <= 0:
        raise ProtocolError(f"expected a positive whole millicredit amount, got {value!r}")
    return int(value)
