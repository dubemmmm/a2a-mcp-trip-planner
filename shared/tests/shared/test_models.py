"""Tests for shared.models — the data contracts between agents."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from shared.models import (
    Attraction,
    DailyPlan,
    FlightOption,
    HotelOption,
    Interest,
    Itinerary,
    PriceLevel,
    TripRequest,
)


# ---------- helpers ----------

def _valid_trip(**overrides) -> TripRequest:
    defaults = dict(
        user_id="u-1",
        destination="Lisbon",
        origin="ATL",
        start_date=date(2026, 5, 10),
        end_date=date(2026, 5, 14),
        budget_usd=Decimal("1500.00"),
        interests=(Interest.FOOD, Interest.HISTORY),
        travelers=1,
    )
    defaults.update(overrides)
    return TripRequest(**defaults)


def _valid_flight(**overrides) -> FlightOption:
    defaults = dict(
        airline="TAP",
        flight_number="TP202",
        origin="ATL",
        destination="LIS",
        depart_at=datetime(2026, 5, 10, 18, 0),
        arrive_at=datetime(2026, 5, 11, 7, 30),
        price_usd=Decimal("612.40"),
        stops=0,
    )
    defaults.update(overrides)
    return FlightOption(**defaults)


def _valid_hotel(**overrides) -> HotelOption:
    defaults = dict(
        name="Hotel Alfama",
        city="Lisbon",
        check_in=date(2026, 5, 10),
        check_out=date(2026, 5, 14),
        price_per_night_usd=Decimal("120.00"),
        rating=4.5,
    )
    defaults.update(overrides)
    return HotelOption(**defaults)


# ---------- TripRequest ----------

class TestTripRequest:
    def test_valid_request_constructs(self) -> None:
        t = _valid_trip()
        assert t.nights == 4
        assert t.travelers == 1

    def test_end_before_start_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc:
            _valid_trip(start_date=date(2026, 5, 14), end_date=date(2026, 5, 10))
        assert "end_date" in str(exc.value)

    def test_same_day_rejected(self) -> None:
        d = date(2026, 5, 10)
        with pytest.raises(ValidationError):
            _valid_trip(start_date=d, end_date=d)

    def test_zero_budget_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_trip(budget_usd=Decimal("0"))

    def test_negative_budget_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_trip(budget_usd=Decimal("-100"))

    def test_empty_user_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_trip(user_id="")

    def test_too_many_travelers_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_trip(travelers=21)

    def test_zero_travelers_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_trip(travelers=0)

    def test_frozen_cannot_mutate(self) -> None:
        t = _valid_trip()
        with pytest.raises(ValidationError):
            t.destination = "Madrid"  # type: ignore[misc]

    def test_round_trip_json(self) -> None:
        t = _valid_trip()
        rebuilt = TripRequest.model_validate_json(t.model_dump_json())
        assert rebuilt == t

    def test_nights_property(self) -> None:
        t = _valid_trip(
            start_date=date(2026, 6, 1), end_date=date(2026, 6, 8)
        )
        assert t.nights == 7


# ---------- FlightOption ----------

class TestFlightOption:
    def test_valid_flight_constructs(self) -> None:
        f = _valid_flight()
        assert f.stops == 0

    def test_arrive_before_depart_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_flight(
                depart_at=datetime(2026, 5, 11, 7, 30),
                arrive_at=datetime(2026, 5, 10, 18, 0),
            )

    def test_negative_stops_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_flight(stops=-1)

    def test_zero_price_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_flight(price_usd=Decimal("0"))


# ---------- HotelOption ----------

class TestHotelOption:
    def test_valid_hotel_constructs(self) -> None:
        h = _valid_hotel()
        assert h.rating == 4.5

    def test_total_price_math(self) -> None:
        h = _valid_hotel(
            check_in=date(2026, 5, 10),
            check_out=date(2026, 5, 14),
            price_per_night_usd=Decimal("120.00"),
        )
        assert h.total_price_usd == Decimal("480.00")

    def test_checkout_before_checkin_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_hotel(
                check_in=date(2026, 5, 14), check_out=date(2026, 5, 10)
            )

    def test_rating_above_5_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_hotel(rating=5.5)

    def test_rating_below_0_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_hotel(rating=-0.1)


# ---------- Attraction, DailyPlan, Itinerary ----------

class TestAttraction:
    def test_valid_attraction(self) -> None:
        a = Attraction(
            name="Jeronimos Monastery",
            category=Interest.HISTORY,
            description="16th-century monastery in Belem.",
            price_level=PriceLevel.MODERATE,
            rating=4.7,
            duration_minutes=90,
        )
        assert a.category is Interest.HISTORY
        assert a.price_level is PriceLevel.MODERATE


class TestItinerary:
    def test_round_trip_with_nested_models(self) -> None:
        trip = _valid_trip()
        flight = _valid_flight()
        hotel = _valid_hotel()
        plan = DailyPlan(
            day=date(2026, 5, 11),
            activities=(
                Attraction(
                    name="Time Out Market",
                    category=Interest.FOOD,
                    description="Food hall with top Lisbon chefs.",
                    duration_minutes=120,
                ),
            ),
        )
        itinerary = Itinerary(
            trip_request=trip,
            selected_flight=flight,
            selected_hotel=hotel,
            daily_plans=(plan,),
            estimated_total_usd=Decimal("1280.40"),
            summary="4 nights in Lisbon emphasizing food and history.",
        )
        rebuilt = Itinerary.model_validate_json(itinerary.model_dump_json())
        assert rebuilt == itinerary

    def test_empty_summary_rejected(self) -> None:
        trip = _valid_trip()
        with pytest.raises(ValidationError):
            Itinerary(
                trip_request=trip,
                estimated_total_usd=Decimal("0"),
                summary="",
            )