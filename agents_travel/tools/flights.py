"""Flight search tool — SerpApi results to `FlightOption` domain models."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from shared.models import FlightOption

from ..clients.serpapi import FlightResult, SerpAPIClient


def _parse_serpapi_time(s: str) -> datetime:
    """Parse SerpApi's local-time strings like '2026-05-10 18:00'.

    Tolerates both space- and 'T'-separated forms; ISO seconds and timezone
    suffixes pass through unchanged.
    """
    return datetime.fromisoformat(s.replace(" ", "T"))


def _to_flight_option(result: FlightResult) -> FlightOption:
    """Map an aggregate SerpApi result to a single `FlightOption`.

    For multi-leg journeys we expose the outermost departure/arrival pair and
    record the stop count. Airline/flight number come from the first segment
    (the operating carrier most users recognize).
    """
    if not result.flights:
        raise ValueError("FlightResult has no segments")

    first = result.flights[0]
    last = result.flights[-1]
    depart_at = _parse_serpapi_time(first.departure_airport.time)
    arrive_at = _parse_serpapi_time(last.arrival_airport.time)

    return FlightOption(
        airline=first.airline,
        flight_number=first.flight_number,
        origin=first.departure_airport.id,
        destination=last.arrival_airport.id,
        depart_at=depart_at,
        arrive_at=arrive_at,
        price_usd=Decimal(str(result.price)),
        stops=max(len(result.flights) - 1, 0),
        booking_url=None,
    )


async def search_flights(
    client: SerpAPIClient,
    *,
    origin: str,
    destination: str,
    depart_date: date,
    return_date: date | None = None,
    max_price_usd: Decimal | None = None,
    adults: int = 1,
) -> tuple[FlightOption, ...]:
    """Search flights and return options sorted by price ascending.

    Args:
        client: An open `SerpAPIClient`.
        origin: IATA airport code (e.g. "ATL").
        destination: IATA airport code (e.g. "LIS").
        depart_date: Outbound date.
        return_date: If set, round-trip pricing; if None, one-way.
        max_price_usd: Optional cap, applied client-side.
        adults: Passenger count; SerpApi multiplies pricing accordingly.
    """
    results = await client.search_flights(
        origin=origin,
        destination=destination,
        outbound_date=depart_date,
        return_date=return_date,
        adults=adults,
    )
    options: list[FlightOption] = []
    for r in results:
        try:
            option = _to_flight_option(r)
        except ValueError:
            continue  # skip malformed entries silently rather than fail the batch
        if max_price_usd is not None and option.price_usd > max_price_usd:
            continue
        options.append(option)
    options.sort(key=lambda o: o.price_usd)
    return tuple(options)