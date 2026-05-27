"""Tests for the pricing tool — pure function, no I/O."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import pytest

from agents_travel.tools.pricing import estimate_total_cost
from shared.models import FlightOption, HotelOption


def _flight(price: float) -> FlightOption:
    return FlightOption(
        airline="TAP",
        flight_number="TP202",
        origin="ATL",
        destination="LIS",
        depart_at=datetime(2026, 5, 10, 18, 0),
        arrive_at=datetime(2026, 5, 11, 7, 30),
        price_usd=Decimal(str(price)),
    )


def _hotel(per_night: float, nights: int = 4) -> HotelOption:
    return HotelOption(
        name="Alfama Stay",
        city="Lisbon",
        check_in=date(2026, 5, 10),
        check_out=date(2026, 5, 10 + nights),
        price_per_night_usd=Decimal(str(per_night)),
    )


class TestEstimateTotalCost:
    def test_basic_sum(self) -> None:
        total = estimate_total_cost(_flight(600), _hotel(120, nights=4))
        # 600 + 480 = 1080
        assert total == Decimal("1080")

    def test_with_extras(self) -> None:
        total = estimate_total_cost(
            _flight(600), _hotel(120, nights=4), extras_usd=Decimal("75.50")
        )
        assert total == Decimal("1155.50")

    def test_zero_extras_default(self) -> None:
        total = estimate_total_cost(_flight(500), _hotel(100, nights=3))
        assert total == Decimal("800")

    def test_negative_extras_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            estimate_total_cost(
                _flight(100), _hotel(50, nights=1), extras_usd=Decimal("-1")
            )

    def test_one_night_stay(self) -> None:
        # Verifies hotel.total_price_usd computes nights correctly.
        total = estimate_total_cost(_flight(400), _hotel(200, nights=1))
        assert total == Decimal("600")

    def test_seven_night_stay(self) -> None:
        total = estimate_total_cost(_flight(800), _hotel(150, nights=7))
        assert total == Decimal("1850")

    def test_returns_decimal(self) -> None:
        # Type contract: no float drift.
        result = estimate_total_cost(_flight(100.10), _hotel(50.05, nights=2))
        assert isinstance(result, Decimal)