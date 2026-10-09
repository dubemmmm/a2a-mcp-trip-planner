"""A2A server for the Travel Agent.

Mount with:
    uvicorn a2a_servers.travel:app --host 0.0.0.0 --port 8002
"""

from __future__ import annotations

from a2a.types import AgentSkill

from agents_travel.agent import TRAVEL_AGENT_NAME, build_travel_agent

from economy.card import pricing_extension
from economy.pricing import TRAVEL_PRICE_SHEET, PriceSheet, PricingPolicy
from shared.config import get_settings

from .scaffolding import build_a2a_app, build_agent_card

TRAVEL_AGENT_DESCRIPTION = (
    "Searches flights and hotels and estimates trip cost. "
    "Backed by SerpApi (Google Flights + Google Hotels)."
)


def _skills() -> list[AgentSkill]:
    return [
        AgentSkill(
            id="search_flights",
            name="Search flights",
            description=(
                "Find flights between two airports for a given date "
                "(one-way) or date pair (round-trip). Pass IATA codes."
            ),
            tags=["flights", "travel", "transport"],
            examples=[
                "Find round-trip flights ATL to LIS, May 10–14, under $800.",
            ],
        ),
        AgentSkill(
            id="search_hotels",
            name="Search hotels",
            description=(
                "Find hotels in a city between two dates, optionally "
                "filtered by maximum nightly price."
            ),
            tags=["hotels", "lodging", "accommodation"],
            examples=[
                "Find hotels in Lisbon May 10–14 under $150 per night.",
            ],
        ),
        AgentSkill(
            id="estimate_total_cost",
            name="Estimate total trip cost",
            description=(
                "Sum a selected flight + hotel + extras into a single total. "
                "Pure computation; no API call."
            ),
            tags=["pricing", "budget"],
            examples=[
                "Total cost: $600 flight + 4 nights at $120 + $50 extras."
            ],
        ),
    ]


def build_app(
    *, public_url: str = "http://localhost:8002", price_sheet: PriceSheet = TRAVEL_PRICE_SHEET
) -> object:
    agent = build_travel_agent()
    card = build_agent_card(
        name=TRAVEL_AGENT_NAME,
        description=TRAVEL_AGENT_DESCRIPTION,
        url=public_url,
        skills=_skills(),
        extensions=[pricing_extension(price_sheet, ledger_url=get_settings().ledger_url)],
    )
    return build_a2a_app(agent, card, pricing=PricingPolicy(price_sheet))


def _uvicorn_app() -> object:
    return build_app()
