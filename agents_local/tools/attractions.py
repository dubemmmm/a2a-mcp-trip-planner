"""Tools that turn Google Places results into domain `Attraction` objects.

The agent's LLM calls these. They are pure-ish: they take typed inputs, call
the client, map results into shared models, and return them. No retry logic,
no caching — those belong in higher layers.
"""

from __future__ import annotations

from collections.abc import Iterable

from shared.models import Attraction, Interest, PriceLevel

from ..clients.google_places import GooglePlacesClient, PlaceResult

# Map our interest enum to a natural-language query Google Places handles well.
_INTEREST_QUERY: dict[Interest, str] = {
    Interest.FOOD: "popular restaurants and cafes",
    Interest.HISTORY: "historical sites and landmarks",
    Interest.NATURE: "parks and gardens",
    Interest.ART: "art museums and galleries",
    Interest.NIGHTLIFE: "bars and nightlife",
    Interest.SHOPPING: "shopping districts and markets",
    Interest.ARCHITECTURE: "architectural landmarks",
    Interest.BEACHES: "beaches and waterfronts",
}

# Map Google's PRICE_LEVEL_* enum strings to our PriceLevel.
_PRICE_LEVEL_MAP: dict[str, PriceLevel] = {
    "PRICE_LEVEL_FREE": PriceLevel.BUDGET,
    "PRICE_LEVEL_INEXPENSIVE": PriceLevel.BUDGET,
    "PRICE_LEVEL_MODERATE": PriceLevel.MODERATE,
    "PRICE_LEVEL_EXPENSIVE": PriceLevel.UPSCALE,
    "PRICE_LEVEL_VERY_EXPENSIVE": PriceLevel.LUXURY,
}


def _build_query(city: str, interest: Interest) -> str:
    base = _INTEREST_QUERY[interest]
    return f"{base} in {city}"


def _map_price_level(raw: str | None) -> PriceLevel | None:
    if raw is None:
        return None
    return _PRICE_LEVEL_MAP.get(raw)


def _place_to_attraction(place: PlaceResult, category: Interest) -> Attraction:
    description = (
        place.editorial_summary.text
        if place.editorial_summary and place.editorial_summary.text
        else f"Popular {category.value} spot."
    )
    return Attraction(
        name=place.display_name.text,
        category=category,
        description=description,
        address=place.formatted_address,
        price_level=_map_price_level(place.price_level),
        rating=place.rating,
    )


async def find_attractions(
    client: GooglePlacesClient,
    city: str,
    interests: Iterable[Interest],
    *,
    max_results_per_interest: int = 5,
) -> tuple[Attraction, ...]:
    """Search Google Places for attractions matching each interest in `city`.

    Args:
        client: An open `GooglePlacesClient`.
        city: Destination city (e.g. "Lisbon").
        interests: Which interest categories to search.
        max_results_per_interest: 1-20, cap per category.

    Returns:
        Combined, de-duplicated attractions across all interests.

    Raises:
        ValueError: On invalid inputs.
        PlacesAPIError: Bubbled from the client.
    """
    if not city.strip():
        raise ValueError("city must be non-empty")
    interests_list = list(interests)
    if not interests_list:
        raise ValueError("at least one interest is required")

    collected: list[Attraction] = []
    seen_names: set[str] = set()
    for interest in interests_list:
        places = await client.search_text(
            _build_query(city, interest), max_results=max_results_per_interest
        )
        for place in places:
            attraction = _place_to_attraction(place, interest)
            # Cheap de-dup on name: Google sometimes returns the same place
            # under different category queries.
            if attraction.name in seen_names:
                continue
            seen_names.add(attraction.name)
            collected.append(attraction)
    return tuple(collected)


async def recommend_restaurants(
    client: GooglePlacesClient,
    city: str,
    *,
    cuisine: str | None = None,
    price_level: PriceLevel | None = None,
    max_results: int = 10,
) -> tuple[Attraction, ...]:
    """Search restaurants in `city`, optionally filtered by cuisine / price.

    Cuisine is folded into the search query; price level is applied as a
    post-filter on results (Places' text-search doesn't accept it directly).
    """
    if not city.strip():
        raise ValueError("city must be non-empty")

    parts = ["restaurants"]
    if cuisine:
        parts.insert(0, cuisine)
    query = f"{' '.join(parts)} in {city}"

    places = await client.search_text(query, max_results=max_results)
    attractions = tuple(_place_to_attraction(p, Interest.FOOD) for p in places)

    if price_level is None:
        return attractions
    return tuple(a for a in attractions if a.price_level == price_level)