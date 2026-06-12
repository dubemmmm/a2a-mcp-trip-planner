"""Tools that turn Google Routes results into domain `RouteOption` objects.

The agent's LLM calls these. They are pure-ish: they take typed inputs, call
the client, map results into local models, and return them. No retry logic,
no caching — those belong in higher layers.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..clients.google_routes import GoogleRoutesClient, RouteResult
from ..models import RouteOption, TravelMode

# Map our TravelMode enum to the string the Routes API expects.
_MODE_TO_API: dict[TravelMode, str] = {
    TravelMode.DRIVE: "DRIVE",
    TravelMode.TRANSIT: "TRANSIT",
    TravelMode.WALK: "WALK",
    TravelMode.BICYCLE: "BICYCLE",
}

# The modes `compare_modes` checks when the caller doesn't specify. Biking is
# left out by default — it's the least universally useful for a visitor.
_DEFAULT_COMPARE_MODES = (TravelMode.TRANSIT, TravelMode.DRIVE, TravelMode.WALK)


def _parse_duration_seconds(raw: str) -> int:
    """Parse a Routes API duration string like "1200s" into integer seconds."""
    text = raw.strip()
    if not text.endswith("s"):
        raise ValueError(f"unexpected duration format: {raw!r}")
    try:
        return round(float(text[:-1]))
    except ValueError as e:
        raise ValueError(f"unexpected duration format: {raw!r}") from e


def _route_to_option(
    result: RouteResult,
    origin: str,
    destination: str,
    mode: TravelMode,
) -> RouteOption:
    seconds = _parse_duration_seconds(result.duration)
    # A real route is never 0 minutes; round up so a 30-second hop reads as 1.
    minutes = max(1, round(seconds / 60))
    km = result.distance_meters / 1000
    label = f"{mode.value} via {result.description}" if result.description else mode.value
    summary = f"{label}: {km:.1f} km, ~{minutes} min"
    return RouteOption(
        origin=origin,
        destination=destination,
        mode=mode,
        distance_meters=result.distance_meters,
        duration_minutes=minutes,
        summary=summary,
    )


async def get_route(
    client: GoogleRoutesClient,
    origin: str,
    destination: str,
    *,
    mode: TravelMode = TravelMode.TRANSIT,
) -> RouteOption | None:
    """Find the best route from `origin` to `destination` for one travel mode.

    Args:
        client: An open `GoogleRoutesClient`.
        origin: Where you're starting (free text, e.g. "Lisbon Airport").
        destination: Where you're going (free text, e.g. "Belem Tower").
        mode: How you want to travel. Defaults to public transit.

    Returns:
        The API's preferred route mapped to a `RouteOption`, or `None` when no
        route exists for that mode (e.g. no transit coverage).

    Raises:
        ValueError: On invalid inputs.
        RoutesAPIError: Bubbled from the client.
    """
    if not origin.strip():
        raise ValueError("origin must be non-empty")
    if not destination.strip():
        raise ValueError("destination must be non-empty")

    results = await client.compute_routes(
        origin, destination, travel_mode=_MODE_TO_API[mode]
    )
    if not results:
        return None
    return _route_to_option(results[0], origin, destination, mode)


async def compare_modes(
    client: GoogleRoutesClient,
    origin: str,
    destination: str,
    *,
    modes: Iterable[TravelMode] = _DEFAULT_COMPARE_MODES,
) -> tuple[RouteOption, ...]:
    """Get a route for each mode and return them sorted fastest-first.

    Modes with no available route are silently dropped, so a caller can ask
    for everything and trust that whatever comes back is real.

    Raises:
        ValueError: If `modes` is empty.
        RoutesAPIError: Bubbled from the client.
    """
    modes_list = list(modes)
    if not modes_list:
        raise ValueError("at least one mode is required")

    options: list[RouteOption] = []
    for mode in modes_list:
        option = await get_route(client, origin, destination, mode=mode)
        if option is not None:
            options.append(option)
    return tuple(sorted(options, key=lambda o: o.duration_minutes))


def total_travel_minutes(routes: Iterable[RouteOption]) -> int:
    """Sum the duration of a sequence of legs (e.g. a day's hops). Pure; no I/O."""
    return sum(route.duration_minutes for route in routes)
