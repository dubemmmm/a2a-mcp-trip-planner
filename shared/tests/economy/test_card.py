"""The pricing extension: built into cards, served by the specialists, read back."""

from __future__ import annotations

import httpx
import pytest
from a2a.types import AgentCapabilities, AgentCard, AgentExtension
from google.protobuf.json_format import ParseDict

from a2a_servers import local, transit, travel
from a2a_servers.scaffolding import build_agent_card
from economy.card import (
    PRICING_EXTENSION_URI,
    InvalidPricingExtension,
    ListedPrice,
    pricing_extension,
    read_pricing,
)
from economy.client import LedgerClient
from economy.pricing import TRAVEL_PRICE_SHEET, PriceSheet

LEDGER = "http://ledger:8005"


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("LEDGER_URL", LEDGER)


def _card(*extensions: AgentExtension) -> AgentCard:
    return build_agent_card(
        name="t", description="d", url="http://t", skills=[], extensions=extensions
    )


def test_round_trip_through_the_card() -> None:
    card = _card(pricing_extension(TRAVEL_PRICE_SHEET, ledger_url=LEDGER))
    pricing = read_pricing(card)
    assert pricing is not None
    assert (pricing.ledger_url, pricing.payee_wallet) == (LEDGER, "travel")
    assert pricing.price("search_flights") == ListedPrice(
        base_mc=400, max_mc=2_500, quote_ttl_seconds=60.0
    )
    assert pricing.price("nope") is None
    assert set(pricing.skills) == set(TRAVEL_PRICE_SHEET.skills)


def test_extension_is_required_and_hides_internal_rates() -> None:
    ext = pricing_extension(TRAVEL_PRICE_SHEET, ledger_url=LEDGER)
    assert ext.uri == PRICING_EXTENSION_URI
    assert ext.required is True
    published = set(ext.params["skills"]["search_flights"].keys())
    assert published == {"base_mc", "max_mc", "quote_ttl_s"}


def test_free_agent_has_no_pricing() -> None:
    other = AgentExtension(uri="urn:something:else")
    assert read_pricing(_card()) is None
    assert read_pricing(_card(other)) is None


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"currency": "usd", "ledger_url": "x", "payee_wallet": "x", "skills": {}},
        {"currency": "millicredit", "ledger_url": "x", "payee_wallet": "x"},
        {
            "currency": "millicredit",
            "ledger_url": "x",
            "payee_wallet": "x",
            "skills": {"s": {"base_mc": 1.5, "max_mc": 2, "quote_ttl_s": 1}},
        },
        {
            "currency": "millicredit",
            "ledger_url": "x",
            "payee_wallet": "x",
            "skills": {"s": {"base_mc": "lots", "max_mc": 2, "quote_ttl_s": 1}},
        },
        {
            "currency": "millicredit",
            "ledger_url": "x",
            "payee_wallet": "x",
            "skills": {"s": {"max_mc": 2, "quote_ttl_s": 1}},
        },
    ],
)
def test_malformed_extension_is_rejected(params: dict[str, object]) -> None:
    ext = AgentExtension(uri=PRICING_EXTENSION_URI)
    ext.params.update(params)
    with pytest.raises(InvalidPricingExtension):
        read_pricing(_card(ext))


def test_reads_a_card_parsed_from_json() -> None:
    """Discovery gets the card as JSON, where every number is a float."""
    card = _card(pricing_extension(TRAVEL_PRICE_SHEET, ledger_url=LEDGER))
    from google.protobuf.json_format import MessageToDict

    parsed = ParseDict(MessageToDict(card), AgentCard())
    pricing = read_pricing(parsed)
    assert pricing is not None
    assert pricing.price("search_hotels") == ListedPrice(300, 2_000, 60.0)


def test_build_agent_card_defaults_to_no_extensions() -> None:
    card = build_agent_card(name="t", description="d", url="http://t", skills=[])
    assert list(card.capabilities.extensions) == []
    assert card.capabilities == AgentCapabilities(streaming=True)


async def _served_card(app: object) -> AgentCard:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://agent",  # type: ignore[arg-type]
    ) as client:
        body = (await client.get("/.well-known/agent-card.json")).json()
    return ParseDict(body, AgentCard(), ignore_unknown_fields=True)


@pytest.mark.parametrize(
    ("module", "wallet"),
    [(travel, "travel"), (local, "local"), (transit, "transit")],
)
async def test_specialists_with_a_ledger_serve_prices_for_every_skill(
    env: None, module: object, wallet: str
) -> None:
    async with httpx.AsyncClient() as http:
        ledger = LedgerClient(LEDGER, agent_id=wallet, agent_key="k", http=http)
        card = await _served_card(module.build_app(ledger=ledger))  # type: ignore[attr-defined]

    pricing = read_pricing(card)
    assert pricing is not None
    assert (pricing.payee_wallet, pricing.ledger_url) == (wallet, LEDGER)
    assert set(pricing.skills) == {s.id for s in card.skills}


@pytest.mark.parametrize("module", [travel, local, transit])
async def test_specialists_without_a_ledger_are_free(env: None, module: object) -> None:
    """A card never advertises a price the agent cannot actually collect."""
    card = await _served_card(module.build_app())  # type: ignore[attr-defined]
    assert read_pricing(card) is None


async def test_ledger_client_from_settings(env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from shared.config import get_settings

    assert LedgerClient.from_settings("travel", get_settings()) is None
    monkeypatch.setenv("AGENT_KEY_TRAVEL", "secret")
    get_settings.cache_clear()
    client = LedgerClient.from_settings("travel", get_settings())
    assert client is not None
    assert client.agent_id == "travel"
    await client.close()


def test_custom_price_sheet_is_served(env: None) -> None:
    sheet = PriceSheet(
        payee_wallet="travel",
        skills={
            skill_id: price
            for skill_id, price in TRAVEL_PRICE_SHEET.skills.items()
            if skill_id == "search_flights"
        },
        token_rate=TRAVEL_PRICE_SHEET.token_rate,
    )
    app = travel.build_app(price_sheet=sheet)
    assert app is not None
