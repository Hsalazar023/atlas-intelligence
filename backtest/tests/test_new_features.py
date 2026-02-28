"""Tests for ALE v2/v3 research-backed features."""
import math
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from backtest.learning_engine import (
    classify_insider_pattern,
    compute_52wk_proximity,
    init_db, insert_signal,
    _compute_momentum_features, _compute_volume_spike,
    _compute_insider_buy_ratio, _compute_sector_avg_car,
    _compute_vix_regime_interaction,
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


# ── v3 Alpha Features ────────────────────────────────────────────────────────

class TestMomentumFeatures:
    """Tests for _compute_momentum_features()."""

    def test_computes_positive_momentum(self):
        """Momentum should be positive when price has risen."""
        price_index = {}
        for i in range(200):
            d = (datetime(2025, 1, 1) + timedelta(days=i)).strftime('%Y-%m-%d')
            price_index[d] = 100.0 + i * 0.25
        result = _compute_momentum_features(price_index, '2025-06-15')
        assert result['momentum_1m'] is not None
        assert result['momentum_1m'] > 0
        assert result['momentum_3m'] is not None
        assert result['momentum_3m'] > 0

    def test_computes_negative_momentum(self):
        """Momentum should be negative when price has fallen."""
        price_index = {}
        for i in range(200):
            d = (datetime(2025, 1, 1) + timedelta(days=i)).strftime('%Y-%m-%d')
            price_index[d] = 200.0 - i * 0.5
        result = _compute_momentum_features(price_index, '2025-06-15')
        assert result['momentum_1m'] < 0

    def test_returns_none_for_empty_index(self):
        result = _compute_momentum_features({}, '2025-06-15')
        assert result['momentum_1m'] is None
        assert result['momentum_3m'] is None
        assert result['momentum_6m'] is None

    def test_returns_none_for_insufficient_data(self):
        """6m momentum should be None if not enough lookback data."""
        price_index = {'2025-06-15': 100.0, '2025-06-14': 99.0}
        result = _compute_momentum_features(price_index, '2025-06-15')
        assert result['momentum_6m'] is None


class TestVolumeSpikeFeature:
    """Tests for _compute_volume_spike()."""

    def test_spike_detection(self):
        """Volume 3x above average should return ~3.0."""
        candles = {}
        for i in range(25):
            d = (datetime(2025, 6, 1) + timedelta(days=i)).strftime('%Y-%m-%d')
            vol = 1_000_000 if i < 22 else 3_000_000
            candles[d] = {'o': 100, 'h': 101, 'l': 99, 'c': 100, 'v': vol}
        result = _compute_volume_spike(candles, '2025-06-23')
        assert result is not None
        assert result > 2.5

    def test_normal_volume(self):
        candles = {}
        for i in range(25):
            d = (datetime(2025, 6, 1) + timedelta(days=i)).strftime('%Y-%m-%d')
            candles[d] = {'o': 100, 'h': 101, 'l': 99, 'c': 100, 'v': 1_000_000}
        result = _compute_volume_spike(candles, '2025-06-23')
        assert result is not None
        assert 0.8 <= result <= 1.2

    def test_empty_candles(self):
        assert _compute_volume_spike({}, '2025-06-15') is None


class TestInsiderBuyRatio:
    """Tests for _compute_insider_buy_ratio()."""

    def test_no_prior_signals(self):
        conn = init_db(db_path=Path(tempfile.mktemp(suffix='.db')))
        insert_signal(conn, {
            'ticker': 'AAPL', 'signal_date': '2025-06-15',
            'source': 'edgar', 'insider_name': 'John', 'representative': None,
        })
        sig = conn.execute("SELECT id FROM signals").fetchone()
        ratio = _compute_insider_buy_ratio(conn, 'AAPL', '2025-06-15', sig['id'])
        assert ratio == 0.0
        conn.close()

    def test_multiple_prior_signals(self):
        conn = init_db(db_path=Path(tempfile.mktemp(suffix='.db')))
        for i in range(3):
            insert_signal(conn, {
                'ticker': 'AAPL', 'signal_date': f'2025-05-{10+i:02d}',
                'source': 'edgar', 'insider_name': f'Person{i}', 'representative': None,
            })
        insert_signal(conn, {
            'ticker': 'AAPL', 'signal_date': '2025-06-15',
            'source': 'edgar', 'insider_name': 'Test', 'representative': None,
        })
        sig = conn.execute(
            "SELECT id FROM signals WHERE signal_date='2025-06-15'"
        ).fetchone()
        ratio = _compute_insider_buy_ratio(conn, 'AAPL', '2025-06-15', sig['id'])
        assert ratio == round(math.log1p(3), 4)
        conn.close()


class TestSectorAvgCar:
    def test_no_sector(self):
        conn = init_db(db_path=Path(tempfile.mktemp(suffix='.db')))
        assert _compute_sector_avg_car(conn, None) is None
        conn.close()

    def test_with_data(self):
        conn = init_db(db_path=Path(tempfile.mktemp(suffix='.db')))
        for i, car in enumerate([0.05, 0.10, -0.03]):
            conn.execute(
                """INSERT INTO signals (ticker, signal_date, source, representative,
                   insider_name, sector, outcome_30d_filled, car_30d)
                   VALUES (?, ?, 'edgar', '', ?, 'Technology', 1, ?)""",
                (f'T{i}', f'2025-06-{10+i:02d}', f'Ins{i}', car)
            )
        conn.commit()
        result = _compute_sector_avg_car(conn, 'Technology')
        assert result is not None
        expected = round((0.05 + 0.10 - 0.03) / 3, 6)
        assert abs(result - expected) < 0.001
        conn.close()


class TestVixRegimeInteraction:
    def test_no_convergence(self):
        assert _compute_vix_regime_interaction(25.0, 0) == 25.0

    def test_with_convergence(self):
        assert _compute_vix_regime_interaction(25.0, 1) == 50.0

    def test_none_vix(self):
        assert _compute_vix_regime_interaction(None, 1) is None
