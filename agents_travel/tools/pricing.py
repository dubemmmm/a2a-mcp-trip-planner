"""Pure trip cost estimation. No I/O."""

from __future__ import annotations

from decimal import Decimal

from shared.models import FlightOption, HotelOption


def estimate_total_cost(
    flight: FlightOption,
    hotel: HotelOption,
    *,
    extras_usd: Decimal = Decimal("0"),
) -> Decimal:
    """Sum the trip cost: flight + hotel total + optional extras.

    SerpApi already returns prices for the configured passenger count and the
    full date range, so no per-traveler multiplier is applied here. `extras_usd`
    is a catch-all for fees, transport, activities, taxes, etc.

    Raises:
        ValueError: If `extras_usd` is negative.
    """
    if extras_usd < 0:
        raise ValueError("extras_usd must be non-negative")
    return flight.price_usd + hotel.total_price_usd + extras_usd