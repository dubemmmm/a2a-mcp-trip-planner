"""The handshake's metadata shapes, built and parsed."""

from __future__ import annotations

import pytest

from economy.protocol import (
    ProtocolError,
    QuoteDeclined,
    QuoteOffer,
    QuoteRequest,
    declined_payload,
    offer_payload,
    paid_task_metadata,
    parse_quote_reply,
    parse_quote_request,
    parse_reservation_id,
    quote_request_metadata,
)


def test_quote_request_round_trip() -> None:
    meta = quote_request_metadata("search_flights", QuoteRequest(payer="concierge", max_mc=1_000))
    assert meta["skill_id"] == "search_flights"
    assert parse_quote_request(meta) == QuoteRequest(payer="concierge", max_mc=1_000)
    plain = quote_request_metadata("s", QuoteRequest(payer="concierge"))
    assert "max_mc" not in plain["pricing"]
    assert parse_quote_request(plain) == QuoteRequest(payer="concierge")


def test_struct_floats_are_accepted_as_whole_amounts() -> None:
    meta = {"pricing": {"phase": "quote", "payer": "c", "max_mc": 1000.0}}
    assert parse_quote_request(meta) == QuoteRequest(payer="c", max_mc=1_000)


@pytest.mark.parametrize(
    "meta", [{}, {"skill_id": "s"}, {"pricing": "x"}, {"pricing": {"phase": "work"}}]
)
def test_non_quote_messages_are_not_quote_requests(meta: dict[str, object]) -> None:
    assert parse_quote_request(meta) is None


@pytest.mark.parametrize(
    "pricing",
    [
        {"phase": "quote"},
        {"phase": "quote", "payer": ""},
        {"phase": "quote", "payer": "c", "max_mc": 1.5},
        {"phase": "quote", "payer": "c", "max_mc": -1},
        {"phase": "quote", "payer": "c", "max_mc": "lots"},
        {"phase": "quote", "payer": "c", "max_mc": True},
    ],
)
def test_malformed_quote_requests(pricing: dict[str, object]) -> None:
    with pytest.raises(ProtocolError):
        parse_quote_request({"pricing": pricing})


def test_reservation_id_round_trip() -> None:
    assert parse_reservation_id(paid_task_metadata("r_1")) == "r_1"
    assert parse_reservation_id({}) is None
    assert parse_reservation_id({"pricing": {"reservation_id": ""}}) is None


def test_reply_round_trips() -> None:
    offer = QuoteOffer("q_1", "search_flights", 1_500, "travel")
    assert parse_quote_reply(offer_payload(offer, expires_at="2026-10-09T00:00:00Z")) == offer
    declined = QuoteDeclined("below_base", "too low")
    assert parse_quote_reply(declined_payload(declined)) == declined


@pytest.mark.parametrize(
    "payload",
    [{}, {"quote": {"quote_id": "q"}}, {"quote": "x"}, {"declined": {"reason": "r"}}],
)
def test_malformed_replies(payload: dict[str, object]) -> None:
    with pytest.raises(ProtocolError):
        parse_quote_reply(payload)
