"""Tests for ALE v2 research-backed features."""
import pytest
from backtest.learning_engine import (
    classify_insider_pattern,
    compute_52wk_proximity,
)


class TestOpportunisticVsRoutine:
    def test_routine_same_month_3_years(self):
        """Insider buying in June for 3+ consecutive years = routine."""
        history = [
            {'date': '2023-06-15'}, {'date': '2024-06-20'}, {'date': '2025-06-10'},
        ]
        assert classify_insider_pattern(history) == 'routine'

    def test_opportunistic_irregular(self):
        """Irregular timing = opportunistic."""
        history = [
            {'date': '2023-03-15'}, {'date': '2024-09-20'}, {'date': '2025-06-10'},
        ]
        assert classify_insider_pattern(history) == 'opportunistic'

    def test_insufficient_history(self):
        """<3 data points = insufficient_history."""
        history = [{'date': '2025-06-15'}]
        assert classify_insider_pattern(history) == 'insufficient_history'

    def test_empty(self):
        assert classify_insider_pattern([]) == 'insufficient_history'

    def test_routine_with_extra_trades(self):
        """Routine pattern still detected with additional non-routine trades."""
        history = [
            {'date': '2022-06-01'}, {'date': '2023-06-15'},
            {'date': '2024-06-20'}, {'date': '2024-11-05'},
        ]
        assert classify_insider_pattern(history) == 'routine'


class Test52WeekProximity:
    def test_at_low(self):
        assert compute_52wk_proximity(price=50, high_52wk=100, low_52wk=50) == 0.0

    def test_at_high(self):
        assert compute_52wk_proximity(price=100, high_52wk=100, low_52wk=50) == 1.0

    def test_midpoint(self):
        assert compute_52wk_proximity(price=75, high_52wk=100, low_52wk=50) == 0.5

    def test_no_range(self):
        assert compute_52wk_proximity(price=50, high_52wk=50, low_52wk=50) is None

    def test_none_price(self):
        assert compute_52wk_proximity(price=None, high_52wk=100, low_52wk=50) is None

    def test_none_high(self):
        assert compute_52wk_proximity(price=50, high_52wk=None, low_52wk=50) is None

    def test_clamped_above(self):
        """Price above 52wk high should clamp to 1.0."""
        assert compute_52wk_proximity(price=120, high_52wk=100, low_52wk=50) == 1.0

    def test_quarter(self):
        assert compute_52wk_proximity(price=62.5, high_52wk=100, low_52wk=50) == 0.25
