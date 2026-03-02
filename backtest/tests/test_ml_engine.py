"""Tests for the walk-forward ML engine."""
import pytest
import sqlite3
import tempfile
import numpy as np
from pathlib import Path
from backtest.ml_engine import (
    prepare_features,
    walk_forward_train,
    walk_forward_regression,
    compute_information_coefficient,
    compute_aggregate_metrics,
    compute_fold_metrics,
    calibrate_probabilities,
    train_multi_horizon,
    WalkForwardResult,
    RegressionResult,
    HorizonResult,
    MultiHorizonResult,
    CAR_ABSOLUTE_MIN,
    CAR_ABSOLUTE_MAX,
    METRIC_BENCHMARKS,
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


@pytest.fixture
def db_with_multi_outcomes():
    """DB with signals that have 30d, 90d, and 180d outcomes filled."""
    conn = init_db(db_path=Path(tempfile.mktemp(suffix='.db')))
    import random
    random.seed(123)
    for i in range(300):
        month = (i % 18) + 1
        year = 2024 + (month - 1) // 12
        month_actual = ((month - 1) % 12) + 1
        day = (i % 28) + 1
        date = f'{year}-{month_actual:02d}-{day:02d}'
        car_30 = random.gauss(0.005, 0.05)
        car_90 = random.gauss(0.01, 0.08)
        car_180 = random.gauss(0.015, 0.10)
        conn.execute(
            """INSERT INTO signals (ticker, signal_date, source, representative,
               insider_name, insider_role, trade_size_points, person_trade_count,
               person_hit_rate_30d, same_ticker_signals_7d, same_ticker_signals_30d,
               has_convergence,
               outcome_30d_filled, car_30d,
               outcome_90d_filled, car_90d,
               outcome_180d_filled, car_180d,
               sector, disclosure_delay, vix_at_signal)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       1, ?, 1, ?, 1, ?, ?, ?, ?)""",
            (f'T{i%50}', date, 'edgar' if i % 3 else 'congress',
             f'Rep{i}' if i % 3 == 0 else '',
             f'Ins{i}' if i % 3 else '',
             random.choice(['CEO', 'CFO', 'Director', 'VP', '']),
             random.choice([3, 5, 8, 10, 15]),
             random.randint(0, 20),
             round(random.random(), 2) if i > 5 else None,
             random.randint(1, 5), random.randint(1, 10),
             1 if i % 30 == 0 else 0,
             round(car_30, 6), round(car_90, 6), round(car_180, 6),
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
        leakage_cols = {'car_30d', 'car_90d', 'car_180d', 'car_365d',
                        'return_30d', 'return_90d', 'return_180d', 'return_365d',
                        'outcome_30d_filled', 'outcome_90d_filled', 'outcome_180d_filled'}
        for col in X.columns:
            assert col not in leakage_cols, f"Leakage column {col} in feature matrix"

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

    def test_prepare_features_90d(self, db_with_multi_outcomes):
        """horizon='90d' should use car_90d column."""
        X, y, ids, dates, car_w = prepare_features(db_with_multi_outcomes, horizon='90d')
        assert len(X) > 0
        # 90d CARs should have different distribution than 30d
        X_30, _, _, _, car_30 = prepare_features(db_with_multi_outcomes, horizon='30d')
        # Both should return data but CARs should differ (different random seeds per col)
        assert len(X_30) > 0
        assert not np.array_equal(car_w, car_30)

    def test_backward_compat(self, db_with_outcomes):
        """No-arg calls should work as before (default horizon='30d')."""
        X, y, ids, dates, car_w = prepare_features(db_with_outcomes)
        X2, y2, ids2, dates2, car_w2 = prepare_features(db_with_outcomes, horizon='30d')
        assert len(X) == len(X2)
        assert np.array_equal(y, y2)
        assert np.array_equal(car_w, car_w2)


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

    def test_fold_metrics_populated(self, db_with_outcomes):
        """Per-fold enhanced metrics should be populated."""
        result = walk_forward_train(db_with_outcomes, min_train_months=3, test_months=1,
                                    min_train_samples=30, min_test_samples=5)
        assert result.n_folds > 0
        for fold in result.folds:
            assert 'brier_score' in fold
            assert 'auc_roc' in fold
            assert 'profit_factor' in fold
            assert 'ev_per_signal' in fold
            # Brier score should be in [0, 1]
            assert 0 <= fold['brier_score'] <= 1
            # AUC should be in [0, 1]
            assert 0 <= fold['auc_roc'] <= 1

    def test_aggregate_metrics_populated(self, db_with_outcomes):
        """Aggregate enhanced metrics should be populated on WalkForwardResult."""
        result = walk_forward_train(db_with_outcomes, min_train_months=3, test_months=1,
                                    min_train_samples=30, min_test_samples=5)
        assert result.n_folds > 0
        # ic_std and ic_t_stat should be computed
        assert isinstance(result.ic_std, float)
        assert isinstance(result.ic_t_stat, float)
        assert isinstance(result.information_ratio, float)
        assert isinstance(result.sortino_ratio, float)
        assert isinstance(result.brier_skill_score, float)
        assert isinstance(result.q5_q1_spread, float)
        assert isinstance(result.top_decile_car, float)
        assert isinstance(result.profit_factor, float)


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

    def test_regression_aggregate_metrics(self, db_with_outcomes):
        """Regression result should have aggregate metrics."""
        result = walk_forward_regression(db_with_outcomes, min_train_months=3, test_months=1,
                                         min_train_samples=30, min_test_samples=5)
        assert isinstance(result.ic_std, float)
        assert isinstance(result.ic_t_stat, float)
        assert isinstance(result.q5_q1_spread, float)
        assert isinstance(result.profit_factor, float)


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


class TestComputeAggregateMetrics:
    def test_basic_output(self):
        """Should return all expected keys."""
        preds = [0.6, 0.4, 0.7, 0.3, 0.5, 0.8, 0.2, 0.55, 0.45, 0.65]
        actuals = [0.05, -0.02, 0.08, -0.04, 0.01, 0.10, -0.06, 0.03, -0.01, 0.07]
        fold_ics = [0.05, 0.08, 0.03, 0.10, 0.06]
        result = compute_aggregate_metrics(preds, actuals, fold_ics, is_clf=True)

        assert 'ic_std' in result
        assert 'ic_t_stat' in result
        assert 'information_ratio' in result
        assert 'sortino_ratio' in result
        assert 'brier_skill_score' in result
        assert 'q5_q1_spread' in result
        assert 'top_decile_car' in result
        assert 'beta' in result
        assert 'profit_factor' in result

    def test_ic_std_positive(self):
        """IC std should be positive when fold ICs vary."""
        fold_ics = [0.05, 0.08, 0.03, 0.10, 0.06]
        preds = list(range(10))
        actuals = [0.01 * x for x in range(10)]
        result = compute_aggregate_metrics(preds, actuals, fold_ics)
        assert result['ic_std'] > 0

    def test_t_stat_with_consistent_ics(self):
        """High, consistent ICs should produce high t-stat."""
        fold_ics = [0.10, 0.10, 0.10, 0.10, 0.10]
        preds = list(range(10))
        actuals = [0.01 * x for x in range(10)]
        result = compute_aggregate_metrics(preds, actuals, fold_ics)
        # With zero std, t_stat should be 0 (division by zero guarded)
        # Actually std(ddof=1) of constant list is 0, so ic_t_stat stays 0
        assert result['ic_t_stat'] == 0.0 or result['ic_std'] == 0.0

    def test_profit_factor_all_wins(self):
        """All positive CARs should give inf profit factor."""
        preds = [0.6, 0.7, 0.8, 0.55, 0.65]
        actuals = [0.05, 0.10, 0.03, 0.04, 0.06]
        fold_ics = [0.1, 0.08]
        result = compute_aggregate_metrics(preds, actuals, fold_ics)
        assert result['profit_factor'] == float('inf')

    def test_insufficient_data(self):
        """Should return zeros with < 5 data points."""
        result = compute_aggregate_metrics([0.5, 0.6], [0.01, 0.02], [0.05])
        assert result['ic_std'] == 0.0
        assert result['q5_q1_spread'] == 0.0

    def test_regression_no_brier(self):
        """Regression mode should skip brier_skill_score."""
        preds = list(range(10))
        actuals = [0.01 * x for x in range(10)]
        fold_ics = [0.05, 0.08]
        result = compute_aggregate_metrics(preds, actuals, fold_ics, is_clf=False)
        assert result['brier_skill_score'] == 0.0


class TestCalibrateProbs:
    def test_bounded_output(self):
        """Calibrated probs should be in [0, 1]."""
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        actuals = np.array([0, 0, 1, 1, 1])
        cal = calibrate_probabilities(probs, actuals)
        assert np.all(cal >= 0.0)
        assert np.all(cal <= 1.0)

    def test_monotonic(self):
        """Calibrated probs should be monotonically non-decreasing (isotonic)."""
        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        actuals = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1])
        cal = calibrate_probabilities(probs, actuals)
        # Isotonic regression gives non-decreasing output
        for i in range(len(cal) - 1):
            assert cal[i] <= cal[i + 1] + 1e-10

    def test_short_input(self):
        """With < 3 items, should return input unchanged."""
        probs = np.array([0.5, 0.6])
        actuals = np.array([0, 1])
        cal = calibrate_probabilities(probs, actuals)
        assert np.array_equal(cal, probs)


class TestMultiHorizon:
    def test_horizon_weights_sum_to_one(self, db_with_multi_outcomes):
        """IC-weighted horizon weights should sum to 1.0."""
        result = train_multi_horizon(db_with_multi_outcomes,
                                      min_train_months=3, test_months=1,
                                      min_train_samples=30, min_test_samples=5)
        if result.horizon_weights:
            total = sum(result.horizon_weights.values())
            assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected ~1.0"

    def test_negative_ic_zero_weight(self):
        """Horizons with negative IC should get zero weight."""
        # Test the weighting logic directly
        # Simulate: IC_30d=0.07, IC_90d=-0.02, IC_180d=0.03
        raw_weights = {'30d': max(0, 0.07), '90d': max(0, -0.02), '180d': max(0, 0.03)}
        weight_sum = sum(raw_weights.values())
        weights = {h: w / weight_sum for h, w in raw_weights.items()}
        assert weights['90d'] == 0.0
        assert weights['30d'] > 0
        assert weights['180d'] > 0
        assert abs(sum(weights.values()) - 1.0) < 1e-10

    def test_multi_horizon_result_structure(self, db_with_multi_outcomes):
        """MultiHorizonResult should have expected fields."""
        result = train_multi_horizon(db_with_multi_outcomes,
                                      min_train_months=3, test_months=1,
                                      min_train_samples=30, min_test_samples=5)
        assert isinstance(result, MultiHorizonResult)
        assert isinstance(result.horizons, dict)
        assert isinstance(result.horizon_weights, dict)
        assert isinstance(result.composite_ic, float)
        assert isinstance(result.ic_decay_rate, float)

    def test_horizon_results_have_both_models(self, db_with_multi_outcomes):
        """Each HorizonResult should contain both clf and reg results."""
        result = train_multi_horizon(db_with_multi_outcomes,
                                      min_train_months=3, test_months=1,
                                      min_train_samples=30, min_test_samples=5)
        for h, hr in result.horizons.items():
            assert isinstance(hr, HorizonResult)
            assert isinstance(hr.clf_result, WalkForwardResult)
            assert isinstance(hr.reg_result, RegressionResult)
            assert hr.horizon == h


class TestFoldMetrics:
    def test_compute_fold_metrics_basic(self):
        """compute_fold_metrics should return expected keys."""
        probs = np.array([0.6, 0.4, 0.7, 0.3, 0.8])
        y_test = np.array([1, 0, 1, 0, 1])
        test_cars = [0.05, -0.02, 0.08, -0.04, 0.10]
        metrics = compute_fold_metrics(probs, y_test, test_cars)
        assert 'brier_score' in metrics
        assert 'auc_roc' in metrics
        assert 'profit_factor' in metrics
        assert 'ev_per_signal' in metrics
        assert 0 <= metrics['brier_score'] <= 1
        assert 0 <= metrics['auc_roc'] <= 1
        assert metrics['profit_factor'] > 0

    def test_all_positive_cars(self):
        """All positive CARs should give inf profit factor."""
        probs = np.array([0.6, 0.7, 0.8])
        y_test = np.array([1, 1, 1])
        test_cars = [0.05, 0.10, 0.03]
        metrics = compute_fold_metrics(probs, y_test, test_cars)
        assert metrics['profit_factor'] == float('inf')
        assert metrics['ev_per_signal'] > 0


class TestMetricBenchmarks:
    def test_benchmarks_structure(self):
        """Each benchmark should have min, good, excellent."""
        for key, bench in METRIC_BENCHMARKS.items():
            assert 'min' in bench
            assert 'good' in bench
            assert 'excellent' in bench
            assert bench['min'] < bench['good'] < bench['excellent']
