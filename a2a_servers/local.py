"""A2A server for the Local Experience Agent.

Mount with:
    uvicorn a2a_servers.local:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

from a2a.types import AgentSkill

from agents_local.agent import LOCAL_AGENT_NAME, build_local_agent

from .scaffolding import build_a2a_app, build_agent_card

LOCAL_AGENT_DESCRIPTION = (
    "Discovers attractions, restaurants, and local weather at a destination. "
    "Backed by Google Places and OpenWeatherMap."
)


def _skills() -> list[AgentSkill]:
    return [
        AgentSkill(
            id="find_attractions",
            name="Find attractions",
            description=(
                "Suggest things to do in a city across one or more interest "
                "categories (food, history, nature, art, nightlife, shopping, "
                "architecture, beaches)."
            ),
            tags=["attractions", "things-to-do", "tourism"],
            examples=[
                "What should I do in Lisbon if I like food and history?",
                "Recommend nature activities near Porto.",
            ],
        ),
        AgentSkill(
            id="recommend_restaurants",
            name="Recommend restaurants",
            description=(
                "Recommend restaurants in a city, optionally filtered by "
                "cuisine and price level."
            ),
            tags=["restaurants", "dining", "food"],
            examples=[
                "Find moderate-priced Portuguese restaurants in Lisbon.",
            ],
        ),
        AgentSkill(
            id="get_weather_forecast",
            name="Get weather forecast",
            description=(
                "Return a daily weather forecast for a city across a date "
                "range (free tier covers ~5 days from today)."
            ),
            tags=["weather", "forecast"],
            examples=["What's the weather in Lisbon for the next 4 days?"],
        ),
    ]


def build_app(*, public_url: str = "http://localhost:8001") -> object:
    """Build the Starlette ASGI app for this server.

    `public_url` is the externally-reachable base URL advertised in the
    Agent Card. For local dev that's http://localhost:8001.
    """
    agent = build_local_agent()
    card = build_agent_card(
        name=LOCAL_AGENT_NAME,
        description=LOCAL_AGENT_DESCRIPTION,
        url=public_url,
        skills=_skills(),
    )
    return build_a2a_app(agent, card)


def _uvicorn_app() -> object:
    """Factory used by uvicorn (`a2a_servers.local:_uvicorn_app`) via --factory."""
    return build_app()