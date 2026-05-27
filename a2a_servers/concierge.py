"""A2A server for the Concierge Agent.

Mount with:
    uvicorn a2a_servers.concierge:app --host 0.0.0.0 --port 8003

Note: the Concierge in Layer 4 still only has its own MCP tools (user data).
Its A2A *client* side — calling Travel and Local Experience — is built in
Layer 5; orchestration logic is Layer 6.
"""

from __future__ import annotations

from a2a.types import AgentSkill

from agents_concierge.agent import CONCIERGE_AGENT_NAME, build_concierge_agent

from .scaffolding import build_a2a_app, build_agent_card

CONCIERGE_AGENT_DESCRIPTION = (
    "Plans and remembers a traveler's trips. Stores preferences and "
    "itineraries in a local SQLite database; in later layers, orchestrates "
    "Travel and Local Experience agents."
)


def _skills() -> list[AgentSkill]:
    return [
        AgentSkill(
            id="plan_trip",
            name="Plan a trip",
            description=(
                "Plan a trip end-to-end given a user request (destination, "
                "dates, budget, interests). Reads any stored preferences and "
                "(in later layers) coordinates with specialist agents."
            ),
            tags=["planning", "itinerary", "orchestration"],
            examples=[
                "Plan a 4-day trip to Lisbon for under $1500 emphasizing "
                "food and history.",
            ],
        ),
        AgentSkill(
            id="get_user_preferences",
            name="Recall user preferences",
            description="Fetch a user's stored trip preferences.",
            tags=["user-data", "preferences"],
            examples=["What does user alice prefer for trip planning?"],
        ),
        AgentSkill(
            id="list_itineraries",
            name="List past itineraries",
            description=(
                "List a user's previously saved itineraries, newest first."
            ),
            tags=["user-data", "history"],
            examples=["Show me alice's past trips."],
        ),
    ]


def build_app(*, public_url: str = "http://localhost:8003") -> object:
    agent = build_concierge_agent()
    card = build_agent_card(
        name=CONCIERGE_AGENT_NAME,
        description=CONCIERGE_AGENT_DESCRIPTION,
        url=public_url,
        skills=_skills(),
    )
    return build_a2a_app(agent, card)


def _uvicorn_app() -> object:
    return build_app()