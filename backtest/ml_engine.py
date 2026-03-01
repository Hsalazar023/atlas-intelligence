"""Walk-forward ML engine for ATLAS signal scoring.

Uses Random Forest + LightGBM ensemble with walk-forward validation.
Trained on historical signals with filled outcomes, tested out-of-sample.
"""
import sqlite3
import logging
import numpy as np
from dataclasses import dataclass, field
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

log = logging.getLogger(__name__)

# CAR winsorization bounds — clip extreme outliers before ML training
CAR_ABSOLUTE_MIN = -1.0   # -100%
CAR_ABSOLUTE_MAX = 3.0    # +300%

FEATURE_COLUMNS = [
    # Core signal features
    'trade_size_points', 'same_ticker_signals_7d',
    'same_ticker_signals_30d', 'has_convergence', 'convergence_tier',
    'person_trade_count', 'person_hit_rate_30d', 'relative_position_size',
    'insider_role', 'sector', 'price_proximity_52wk', 'market_cap_bucket',
    'cluster_velocity', 'disclosure_delay',
    # Macro context
    'vix_at_signal', 'yield_curve_at_signal', 'credit_spread_at_signal',
    'days_to_earnings', 'days_to_catalyst',
    # Price / momentum features
    'momentum_1m', 'momentum_3m', 'momentum_6m',
    'volume_spike', 'insider_buy_ratio_90d', 'sector_avg_car',
    'vix_regime_interaction',
    # v4: person magnitude + sector momentum + repeat buyer signal
    'person_avg_car_30d', 'sector_momentum', 'days_since_last_buy',
]
# Pruned in v4: 'source' (0.16% imp), 'trade_pattern' (0.40% imp, 31% fill)

CATEGORICAL_FEATURES = [
    'insider_role', 'sector', 'market_cap_bucket',
    'cluster_velocity',
]


@dataclass
class FoldResult:
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    ic: float
    hit_rate: float
    avg_car: float


@dataclass
class WalkForwardResult:
    n_folds: int
    oos_ic: float
    oos_hit_rate: float
    oos_avg_car: float
    feature_importance: dict
    folds: list = field(default_factory=list)
    model_rf: object = None
    model_lgb: object = None


def prepare_features(conn: sqlite3.Connection):
    """Extract feature matrix X, target y, signal IDs, dates, and winsorized CARs.

    Returns (X, y, ids, dates, car_winsorized) or 5 empty arrays if no data.
    CAR winsorization applies percentile clipping (1st/99th) with hard bounds ±300%.
    """
    import pandas as pd

    cols = ', '.join(FEATURE_COLUMNS)
    rows = conn.execute(
        f"SELECT id, signal_date, car_30d, {cols} "
        f"FROM signals WHERE outcome_30d_filled = 1 AND car_30d IS NOT NULL"
    ).fetchall()

    if not rows:
        return pd.DataFrame(), np.array([]), np.array([]), np.array([]), np.array([])

    data = [dict(r) for r in rows]
    df = pd.DataFrame(data)
    ids = df['id'].values
    dates = df['signal_date'].values

    # Winsorize CARs: percentile clip then hard bounds
    car_raw = df['car_30d'].copy()
    p1 = car_raw.quantile(0.01)
    p99 = car_raw.quantile(0.99)
    car_clipped = car_raw.clip(lower=max(p1, CAR_ABSOLUTE_MIN),
                                upper=min(p99, CAR_ABSOLUTE_MAX))

    y = (car_clipped > 0).astype(int).values

    X = df[FEATURE_COLUMNS].copy()

    # Convert categoricals to numeric codes.
    # We use pandas category codes (deterministic mapping based on sorted unique values).
    # LightGBM and RF handle these natively. Encoding is done on full dataset for
    # consistent mapping — this is safe for tree models (no information leakage since
    # encoding is just a label mapping, not derived from the target variable).
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].fillna('unknown').astype(str)
            X[col] = X[col].astype('category').cat.codes

    X = X.fillna(0).infer_objects(copy=False)
    return X, y, ids, dates, car_clipped.values


def prepare_features_all(conn: sqlite3.Connection):
    """Extract features for ALL signals (including those without outcomes).

    Used for scoring — no outcome filter so recent/actionable signals are included.
    Returns (X, ids, dates, tickers, cars, X_raw) where cars may be NaN for pending
    signals and X_raw is the pre-encoded feature DataFrame for factor attribution.
    """
    import pandas as pd

    cols = ', '.join(FEATURE_COLUMNS)
    rows = conn.execute(
        f"SELECT id, signal_date, ticker, car_30d, {cols} FROM signals"
    ).fetchall()

    if not rows:
        empty = pd.DataFrame()
        return empty, np.array([]), np.array([]), np.array([]), np.array([]), empty

    data = [dict(r) for r in rows]
    df = pd.DataFrame(data)
    ids = df['id'].values
    dates = df['signal_date'].values
    tickers = df['ticker'].values
    cars = df['car_30d'].values.astype(float)

    X_raw = df[FEATURE_COLUMNS].copy().fillna(0).infer_objects(copy=False)

    X = df[FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].fillna('unknown').astype(str)
            X[col] = X[col].astype('category').cat.codes

    X = X.fillna(0).infer_objects(copy=False)
    return X, ids, dates, tickers, cars, X_raw


def train_full_sample(conn: sqlite3.Connection):
    """Train 4 models on ALL historical data with outcomes (full sample, not walk-forward).

    Returns (clf_rf, clf_lgb, reg_rf, reg_lgb) or None if insufficient data.
    """
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor

    X, y, ids, dates, car_winsorized = prepare_features(conn)
    if len(X) < 50:
        log.warning(f"Insufficient data for full-sample training ({len(X)} signals)")
        return None

    # Classification: P(beat SPY)
    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    clf_rf.fit(X, y)

    clf_lgb = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42,
                                  verbose=-1, n_jobs=-1)
    clf_lgb.fit(X, y)

    # Regression: predicted CAR magnitude
    reg_rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    reg_rf.fit(X, car_winsorized)

    reg_lgb = lgb.LGBMRegressor(n_estimators=100, max_depth=6, random_state=42,
                                 verbose=-1, n_jobs=-1)
    reg_lgb.fit(X, car_winsorized)

    log.info(f"Full-sample models trained on {len(X)} signals")
    return clf_rf, clf_lgb, reg_rf, reg_lgb


def compute_information_coefficient(predicted, actual) -> float:
    """Spearman rank correlation between predictions and actual returns."""
    if len(predicted) < 3 or len(actual) < 3:
        return 0.0
    corr, _ = spearmanr(predicted, actual)
    return round(corr, 6) if not np.isnan(corr) else 0.0


def walk_forward_train(conn: sqlite3.Connection,
                       min_train_months: int = 6,
                       test_months: int = 1,
                       min_train_samples: int = 200,
                       min_test_samples: int = 20) -> WalkForwardResult:
    """Walk-forward validation with RF + LightGBM ensemble."""
    import pandas as pd

    X, y, ids, dates, car_winsorized = prepare_features(conn)
    if len(X) < 50:
        log.warning(f"Insufficient data for ML training ({len(X)} signals)")
        return WalkForwardResult(n_folds=0, oos_ic=0, oos_hit_rate=0,
                                 oos_avg_car=0, feature_importance={})

    # Sort by date
    date_series = pd.to_datetime(dates)
    sort_idx = date_series.argsort()
    X = X.iloc[sort_idx].reset_index(drop=True)
    y = y[sort_idx]
    ids = ids[sort_idx]
    car_winsorized = car_winsorized[sort_idx]
    date_series = date_series[sort_idx]

    # Walk-forward folds
    max_date = date_series.max()
    folds = []
    all_oos_preds = []
    all_oos_actual = []
    all_fold_importances = []
    rf = None
    lgb_model = None

    train_end = date_series.min() + pd.DateOffset(months=min_train_months)

    while train_end + pd.DateOffset(months=test_months) <= max_date:
        test_end = train_end + pd.DateOffset(months=test_months)

        train_mask = date_series < train_end
        test_mask = (date_series >= train_end) & (date_series < test_end)

        if train_mask.sum() < min_train_samples or test_mask.sum() < min_test_samples:
            train_end += pd.DateOffset(months=1)
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_probs = rf.predict_proba(X_test)[:, 1]

        # Train LightGBM
        lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42,
                                        verbose=-1, n_jobs=-1)
        lgb_model.fit(X_train, y_train)
        lgb_probs = lgb_model.predict_proba(X_test)[:, 1]

        # Ensemble: average probabilities
        ensemble_probs = (rf_probs + lgb_probs) / 2

        # Accumulate per-fold feature importance
        feat_names = list(X.columns)
        rf_imp = rf.feature_importances_
        lgb_imp = lgb_model.feature_importances_ / max(lgb_model.feature_importances_.sum(), 1)
        fold_imp = {}
        for i, name in enumerate(feat_names):
            fold_imp[name] = (rf_imp[i] + lgb_imp[i]) / 2
        all_fold_importances.append(fold_imp)

        # Compute metrics using winsorized CARs
        test_cars = car_winsorized[test_mask].tolist()
        ic = compute_information_coefficient(ensemble_probs, test_cars)
        hit_rate = sum(1 for p, a in zip(ensemble_probs > 0.5, y_test) if p == a) / len(y_test)
        avg_car = np.mean(test_cars) if test_cars else 0

        fold = FoldResult(
            train_start=str(date_series[train_mask].min().date()),
            train_end=str(train_end.date()),
            test_start=str(train_end.date()),
            test_end=str(test_end.date()),
            n_train=int(train_mask.sum()),
            n_test=int(test_mask.sum()),
            ic=ic, hit_rate=round(hit_rate, 4), avg_car=round(avg_car, 6),
        )
        folds.append(fold)
        all_oos_preds.extend(ensemble_probs.tolist())
        all_oos_actual.extend(test_cars)

        train_end += pd.DateOffset(months=1)

    # Aggregate OOS metrics
    oos_ic = compute_information_coefficient(all_oos_preds, all_oos_actual)
    oos_hit = sum(1 for p, c in zip(all_oos_preds, all_oos_actual) if (p > 0.5) == (c > 0)) / max(len(all_oos_preds), 1)
    oos_avg_car = np.mean(all_oos_actual) if all_oos_actual else 0

    # Feature importance — averaged across ALL folds (not just last)
    importance = {}
    if all_fold_importances:
        feat_names = list(X.columns)
        for name in feat_names:
            vals = [fi.get(name, 0) for fi in all_fold_importances]
            importance[name] = round(np.mean(vals), 4)

    # Log features with low importance (debug-level; details in report)
    if importance:
        low_imp = [f"{name} ({imp:.4f})" for name, imp in importance.items() if imp < 0.005]
        if low_imp:
            log.debug(f"Low-importance features (<0.5%): {', '.join(low_imp)}")

    return WalkForwardResult(
        n_folds=len(folds),
        oos_ic=round(oos_ic, 6),
        oos_hit_rate=round(oos_hit, 4),
        oos_avg_car=round(float(oos_avg_car), 6),
        feature_importance=dict(sorted(importance.items(), key=lambda x: -x[1])),
        folds=[{
            'train_start': f.train_start, 'train_end': f.train_end,
            'test_start': f.test_start, 'test_end': f.test_end,
            'n_train': f.n_train, 'n_test': f.n_test,
            'ic': f.ic, 'hit_rate': f.hit_rate,
        } for f in folds],
        model_rf=rf, model_lgb=lgb_model,
    )


@dataclass
class RegressionResult:
    n_folds: int
    oos_ic: float
    oos_rmse: float
    oos_avg_car: float
    feature_importance: dict
    folds: list = field(default_factory=list)
    model_rf: object = None
    model_lgb: object = None


def walk_forward_regression(conn: sqlite3.Connection,
                             min_train_months: int = 6,
                             test_months: int = 1,
                             min_train_samples: int = 200,
                             min_test_samples: int = 20) -> RegressionResult:
    """Walk-forward regression with RF + LightGBM ensemble.

    Targets continuous winsorized car_30d (not binary). Output is predicted
    CAR magnitude for signal ranking — richer than direction-only classification.
    """
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

    X, y_cls, ids, dates, car_winsorized = prepare_features(conn)
    if len(X) < 50:
        log.warning(f"Insufficient data for regression ({len(X)} signals)")
        return RegressionResult(n_folds=0, oos_ic=0, oos_rmse=0,
                                 oos_avg_car=0, feature_importance={})

    # Target: continuous winsorized CAR
    y = car_winsorized

    # Sort by date
    date_series = pd.to_datetime(dates)
    sort_idx = date_series.argsort()
    X = X.iloc[sort_idx].reset_index(drop=True)
    y = y[sort_idx]
    ids = ids[sort_idx]
    date_series = date_series[sort_idx]

    max_date = date_series.max()
    folds = []
    all_oos_preds = []
    all_oos_actual = []
    all_fold_importances = []
    rf = None
    lgb_model = None

    train_end = date_series.min() + pd.DateOffset(months=min_train_months)

    while train_end + pd.DateOffset(months=test_months) <= max_date:
        test_end = train_end + pd.DateOffset(months=test_months)

        train_mask = date_series < train_end
        test_mask = (date_series >= train_end) & (date_series < test_end)

        if train_mask.sum() < min_train_samples or test_mask.sum() < min_test_samples:
            train_end += pd.DateOffset(months=1)
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Train Random Forest Regressor
        rf = RandomForestRegressor(n_estimators=100, max_depth=6,
                                    random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)

        # Train LightGBM Regressor
        lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=6,
                                       random_state=42, verbose=-1, n_jobs=-1)
        lgb_model.fit(X_train, y_train)
        lgb_preds = lgb_model.predict(X_test)

        # Ensemble: average predictions
        ensemble_preds = (rf_preds + lgb_preds) / 2

        # Accumulate per-fold feature importance
        feat_names = list(X.columns)
        rf_imp = rf.feature_importances_
        lgb_imp = lgb_model.feature_importances_ / max(lgb_model.feature_importances_.sum(), 1)
        fold_imp = {}
        for i, name in enumerate(feat_names):
            fold_imp[name] = (rf_imp[i] + lgb_imp[i]) / 2
        all_fold_importances.append(fold_imp)

        # Metrics
        ic = compute_information_coefficient(ensemble_preds, y_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, ensemble_preds)))
        hit_rate = sum(1 for p, a in zip(ensemble_preds > 0, y_test > 0) if p == a) / len(y_test)
        avg_car = float(np.mean(y_test))

        fold = FoldResult(
            train_start=str(date_series[train_mask].min().date()),
            train_end=str(train_end.date()),
            test_start=str(train_end.date()),
            test_end=str(test_end.date()),
            n_train=int(train_mask.sum()),
            n_test=int(test_mask.sum()),
            ic=ic, hit_rate=round(hit_rate, 4), avg_car=round(avg_car, 6),
        )
        folds.append(fold)
        all_oos_preds.extend(ensemble_preds.tolist())
        all_oos_actual.extend(y_test.tolist())

        train_end += pd.DateOffset(months=1)

    # Aggregate OOS metrics
    oos_ic = compute_information_coefficient(all_oos_preds, all_oos_actual)
    oos_rmse = float(np.sqrt(np.mean((np.array(all_oos_preds) - np.array(all_oos_actual))**2))) if all_oos_actual else 0
    oos_avg_car = float(np.mean(all_oos_actual)) if all_oos_actual else 0

    # Feature importance — averaged across ALL folds
    importance = {}
    if all_fold_importances:
        feat_names = list(X.columns)
        for name in feat_names:
            vals = [fi.get(name, 0) for fi in all_fold_importances]
            importance[name] = round(np.mean(vals), 4)

        # Log low-importance features (debug-level; details in report)
        low_imp = [f"{name} ({imp:.4f})" for name, imp in importance.items() if imp < 0.005]
        if low_imp:
            log.debug(f"Regression: low-importance features (<0.5%): {', '.join(low_imp)}")

    return RegressionResult(
        n_folds=len(folds),
        oos_ic=round(oos_ic, 6),
        oos_rmse=round(oos_rmse, 6),
        oos_avg_car=round(oos_avg_car, 6),
        feature_importance=dict(sorted(importance.items(), key=lambda x: -x[1])),
        folds=[{
            'train_start': f.train_start, 'train_end': f.train_end,
            'test_start': f.test_start, 'test_end': f.test_end,
            'n_train': f.n_train, 'n_test': f.n_test,
            'ic': f.ic, 'hit_rate': f.hit_rate,
        } for f in folds],
        model_rf=rf, model_lgb=lgb_model,
    )
