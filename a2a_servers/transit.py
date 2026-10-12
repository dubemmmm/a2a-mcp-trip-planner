"""A2A server for the Ground Transport ("Getting Around") Agent.

Mount with:
    uvicorn a2a_servers.transit:_uvicorn_app --factory --host 0.0.0.0 --port 8004
"""

from __future__ import annotations

from a2a.types import AgentSkill

from agents_transit.agent import TRANSIT_AGENT_NAME, build_transit_agent

from economy.card import pricing_extension
from economy.client import LedgerClient
from economy.payee import Payee
from economy.pricing import TRANSIT_PRICE_SHEET, PriceSheet, PricingPolicy
from shared.config import get_settings

from .scaffolding import build_a2a_app, build_agent_card

TRANSIT_AGENT_DESCRIPTION = (
    "Plans how to get around a destination — transit, driving, walking, and "
    "cycling routes, with mode comparison. Backed by the Google Routes API."
)


def _skills() -> list[AgentSkill]:
    return [
        AgentSkill(
            id="get_route",
            name="Get a route",
            description=(
                "Find the best route between two points for one travel mode "
                "(transit, drive, walk, or bicycle), with distance and "
                "duration."
            ),
            tags=["routing", "directions", "transport"],
            examples=[
                "How do I get from Lisbon Airport to Belem Tower by transit?",
                "Driving route from my hotel to the Oceanarium.",
            ],
        ),
        AgentSkill(
            id="compare_modes",
            name="Compare travel modes",
            description=(
                "Compare how long it takes to get between two points by "
                "transit, driving, and walking, sorted fastest-first."
            ),
            tags=["routing", "comparison", "transport"],
            examples=[
                "What's the fastest way from Rossio to Belem — transit, "
                "driving, or walking?",
            ],
        ),
        AgentSkill(
            id="total_travel_minutes",
            name="Total travel time",
            description=(
                "Sum a sequence of route legs into a single travel-time "
                "figure. Pure computation; no API call."
            ),
            tags=["routing", "budget", "time"],
            examples=[
                "How much total travel time across these three legs of my day?"
            ],
        ),
    ]


def build_app(
    *,
    public_url: str = "http://localhost:8004",
    price_sheet: PriceSheet = TRANSIT_PRICE_SHEET,
    ledger: LedgerClient | None = None,
) -> object:
    """Build the Starlette ASGI app for this server.

    `public_url` is the externally-reachable base URL advertised in the
    Agent Card. For local dev that's http://localhost:8004.
    """
    agent = build_transit_agent()
    card = build_agent_card(
        name=TRANSIT_AGENT_NAME,
        description=TRANSIT_AGENT_DESCRIPTION,
        url=public_url,
        skills=_skills(),
        # Only an agent that can take payment advertises a price.
        extensions=(
            [pricing_extension(price_sheet, ledger_url=get_settings().ledger_url)]
            if ledger is not None
            else []
        ),
    )
    pricing = PricingPolicy(price_sheet)
    payee = Payee(ledger, pricing) if ledger is not None else None
    return build_a2a_app(agent, card, pricing=pricing, payee=payee)


def _uvicorn_app() -> object:
    """Factory used by uvicorn (`a2a_servers.transit:_uvicorn_app`) via --factory."""
    return build_app(ledger=LedgerClient.from_settings("transit", get_settings()))
