"""Hotel search tool — SerpApi results to `HotelOption` domain models."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from shared.models import HotelOption

from ..clients.serpapi import HotelProperty, SerpAPIClient


def _to_hotel_option(
    prop: HotelProperty, *, city: str, check_in: date, check_out: date
) -> HotelOption | None:
    """Map SerpApi property to a HotelOption.

    Returns None when the property has no usable nightly rate — SerpApi
    occasionally returns vacation rentals or listings without prices.
    """
    if prop.rate_per_night is None or prop.rate_per_night.extracted_lowest is None:
        return None
    return HotelOption(
        name=prop.name,
        city=city,
        check_in=check_in,
        check_out=check_out,
        price_per_night_usd=Decimal(str(prop.rate_per_night.extracted_lowest)),
        rating=prop.overall_rating,
        address=prop.address,
        booking_url=prop.link,
    )


async def search_hotels(
    client: SerpAPIClient,
    *,
    city: str,
    check_in: date,
    check_out: date,
    max_price_per_night_usd: Decimal | None = None,
    adults: int = 1,
) -> tuple[HotelOption, ...]:
    """Search hotels in `city` between the dates, sorted by nightly price.

    Args:
        client: An open `SerpAPIClient`.
        city: City name (used as the SerpApi query and stamped onto results).
        check_in: Check-in date.
        check_out: Check-out date (exclusive — Google's convention too).
        max_price_per_night_usd: Optional cap, applied client-side.
        adults: Occupants.
    """
    props = await client.search_hotels(
        query=city, check_in=check_in, check_out=check_out, adults=adults
    )
    options: list[HotelOption] = []
    for prop in props:
        option = _to_hotel_option(
            prop, city=city, check_in=check_in, check_out=check_out
        )
        if option is None:
            continue
        if (
            max_price_per_night_usd is not None
            and option.price_per_night_usd > max_price_per_night_usd
        ):
            continue
        options.append(option)
    options.sort(key=lambda o: o.price_per_night_usd)
    return tuple(options)