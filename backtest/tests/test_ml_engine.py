"""Tests for the walk-forward ML engine."""
import pytest
import sqlite3
import tempfile
from pathlib import Path
from backtest.ml_engine import (
    prepare_features,
    walk_forward_train,
    walk_forward_regression,
    compute_information_coefficient,
    WalkForwardResult,
    RegressionResult,
    CAR_ABSOLUTE_MIN,
    CAR_ABSOLUTE_MAX,
)
from backtest.learning_engine import init_db


@pytest.fixture
def db_with_outcomes():
    """DB with enough signals+outcomes for ML training."""
    conn = init_db(db_path=Path(tempfile.mktemp(suffix='.db')))
    import random
    random.seed(42)
    for i in range(200):
        month = (i % 18) + 1
        year = 2024 + (month - 1) // 12
        month_actual = ((month - 1) % 12) + 1
        day = (i % 28) + 1
        date = f'{year}-{month_actual:02d}-{day:02d}'
        car = random.gauss(0.005, 0.05)
        # Add some extreme outliers to test winsorization
        if i == 0:
            car = 17.32  # +1732% — should be clipped
        if i == 1:
            car = -1.5   # -150% — should be clipped
        conn.execute(
            """INSERT INTO signals (ticker, signal_date, source, representative,
               insider_name, insider_role, trade_size_points, person_trade_count,
               person_hit_rate_30d, same_ticker_signals_7d, same_ticker_signals_30d,
               has_convergence, outcome_30d_filled, car_30d, sector,
               disclosure_delay, vix_at_signal)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)""",
            (f'T{i%50}', date, 'edgar' if i % 3 else 'congress',
             f'Rep{i}' if i % 3 == 0 else '',
             f'Ins{i}' if i % 3 else '',
             random.choice(['CEO', 'CFO', 'Director', 'VP', '']),
             random.choice([3, 5, 8, 10, 15]),
             random.randint(0, 20),
             round(random.random(), 2) if i > 5 else None,
             random.randint(1, 5), random.randint(1, 10),
             1 if i % 30 == 0 else 0, round(car, 6),
             random.choice(['Technology', 'Healthcare', 'Industrials', 'Financials']),
             random.choice([1, 5, 10, 30, None]),
             round(random.uniform(12, 35), 2) if i % 2 == 0 else None)
        )
    conn.commit()
    yield conn
    conn.close()


class TestPrepareFeatures:
    def test_returns_dataframe(self, db_with_outcomes):
        X, y, ids, dates, car_w = prepare_features(db_with_outcomes)
        assert len(X) == len(y) == len(ids) == len(dates) == len(car_w)
        assert len(X) > 0

    def test_no_leakage(self, db_with_outcomes):
        """Feature matrix should not contain outcome columns."""
        X, y, ids, dates, car_w = prepare_features(db_with_outcomes)
        for col in X.columns:
            assert 'car_' not in col
            assert 'return_' not in col

    def test_winsorization_clips_extreme_cars(self, db_with_outcomes):
        """Extreme CARs should be clipped to hard bounds."""
        X, y, ids, dates, car_w = prepare_features(db_with_outcomes)
        assert car_w.min() >= CAR_ABSOLUTE_MIN, f"Min CAR {car_w.min()} below {CAR_ABSOLUTE_MIN}"
        assert car_w.max() <= CAR_ABSOLUTE_MAX, f"Max CAR {car_w.max()} above {CAR_ABSOLUTE_MAX}"

    def test_feature_columns_present(self, db_with_outcomes):
        """All expected feature columns should be in the feature matrix."""
        X, y, ids, dates, car_w = prepare_features(db_with_outcomes)
        assert 'vix_at_signal' in X.columns
        assert 'insider_role' in X.columns
        # urgent_filing removed (0% importance)
        assert 'urgent_filing' not in X.columns


class TestWalkForward:
    def test_produces_results(self, db_with_outcomes):
        result = walk_forward_train(db_with_outcomes, min_train_months=3, test_months=1,
                                    min_train_samples=30, min_test_samples=5)
        assert isinstance(result, WalkForwardResult)
        assert result.n_folds > 0
        assert result.oos_ic is not None
        assert result.oos_hit_rate is not None

    def test_no_lookahead(self, db_with_outcomes):
        """Train dates must be strictly before test dates in every fold."""
        result = walk_forward_train(db_with_outcomes, min_train_months=3, test_months=1,
                                    min_train_samples=30, min_test_samples=5)
        for fold in result.folds:
            assert fold['train_end'] <= fold['test_start']


class TestWalkForwardRegression:
    def test_produces_results(self, db_with_outcomes):
        result = walk_forward_regression(db_with_outcomes, min_train_months=3, test_months=1,
                                         min_train_samples=30, min_test_samples=5)
        assert isinstance(result, RegressionResult)
        assert result.n_folds > 0
        assert result.oos_ic is not None
        assert result.oos_rmse is not None
        assert result.oos_rmse >= 0

    def test_no_lookahead(self, db_with_outcomes):
        """Train dates must be strictly before test dates in every fold."""
        result = walk_forward_regression(db_with_outcomes, min_train_months=3, test_months=1,
                                          min_train_samples=30, min_test_samples=5)
        for fold in result.folds:
            assert fold['train_end'] <= fold['test_start']


class TestInformationCoefficient:
    def test_perfect_correlation(self):
        predicted = [1, 2, 3, 4, 5]
        actual = [0.01, 0.02, 0.03, 0.04, 0.05]
        ic = compute_information_coefficient(predicted, actual)
        assert ic > 0.9

    def test_no_correlation(self):
        predicted = [1, 2, 3, 4, 5]
        actual = [0.05, 0.01, 0.03, 0.02, 0.04]
        ic = compute_information_coefficient(predicted, actual)
        assert -0.5 < ic < 0.5

    def test_empty_input(self):
        ic = compute_information_coefficient([], [])
        assert ic == 0.0
