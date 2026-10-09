"""The pricing extension on A2A Agent Cards.

A priced agent declares one extension in `capabilities.extensions`:

    {
      "uri": "urn:trip-planner:ext:pricing:v1",
      "required": true,
      "params": {
        "currency": "millicredit",
        "ledger_url": "http://127.0.0.1:8005",
        "payee_wallet": "travel",
        "skills": {
          "search_flights": {"base_mc": 400, "max_mc": 2500, "quote_ttl_s": 60}
        }
      }
    }

`required: true` tells a client that does not understand pricing that it
cannot use this agent. Callers learn prices the same way they learn skills:
from the card, through the existing discovery cache.

Only the list terms are published. Token rates, tool costs and the running
estimates stay inside the agent; a caller gets the actual ceiling by asking
for a quote.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from a2a.types import AgentCard, AgentExtension
from google.protobuf.json_format import MessageToDict

from .pricing import PriceSheet

PRICING_EXTENSION_URI = "urn:trip-planner:ext:pricing:v1"
CURRENCY = "millicredit"


class InvalidPricingExtension(ValueError):
    """The card declares the pricing extension but its params are malformed."""


@dataclass(frozen=True)
class ListedPrice:
    """One skill's published terms."""

    base_mc: int
    max_mc: int
    quote_ttl_seconds: float


@dataclass(frozen=True)
class CardPricing:
    """The pricing terms read off a discovered card."""

    ledger_url: str
    payee_wallet: str
    skills: Mapping[str, ListedPrice]

    def price(self, skill_id: str) -> ListedPrice | None:
        return self.skills.get(skill_id)


def pricing_extension(sheet: PriceSheet, *, ledger_url: str) -> AgentExtension:
    """Build the card extension that publishes `sheet`'s list prices."""
    extension = AgentExtension(
        uri=PRICING_EXTENSION_URI,
        description="Calls cost credits: quote, reserve, then settle on completion.",
        required=True,
    )
    extension.params.update(
        {
            "currency": CURRENCY,
            "ledger_url": ledger_url,
            "payee_wallet": sheet.payee_wallet,
            "skills": {
                skill_id: {
                    "base_mc": price.base_mc,
                    "max_mc": price.max_mc,
                    "quote_ttl_s": price.quote_ttl_seconds,
                }
                for skill_id, price in sheet.skills.items()
            },
        }
    )
    return extension


def read_pricing(card: AgentCard) -> CardPricing | None:
    """Return the card's pricing terms, or None if the agent is free.

    Raises:
        InvalidPricingExtension: the extension is present but malformed.
    """
    for extension in card.capabilities.extensions:
        if extension.uri == PRICING_EXTENSION_URI:
            return _parse(MessageToDict(extension.params))
    return None


def _parse(params: dict[str, Any]) -> CardPricing:
    try:
        if params["currency"] != CURRENCY:
            raise InvalidPricingExtension(f"unsupported currency {params['currency']!r}")
        skills = {
            str(skill_id): ListedPrice(
                # protobuf Struct carries every number as a float.
                base_mc=_whole(terms["base_mc"]),
                max_mc=_whole(terms["max_mc"]),
                quote_ttl_seconds=float(terms["quote_ttl_s"]),
            )
            for skill_id, terms in params["skills"].items()
        }
        return CardPricing(
            ledger_url=str(params["ledger_url"]),
            payee_wallet=str(params["payee_wallet"]),
            skills=skills,
        )
    except InvalidPricingExtension:
        raise
    except (KeyError, TypeError, AttributeError, ValueError) as e:
        raise InvalidPricingExtension(f"malformed pricing extension: {e!r}") from e


def _whole(value: Any) -> int:
    number = float(value)
    if not number.is_integer() or number < 0:
        raise InvalidPricingExtension(f"expected a whole millicredit amount, got {value!r}")
    return int(number)
