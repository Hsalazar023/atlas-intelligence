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
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

log = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    'source', 'trade_size_points', 'same_ticker_signals_7d',
    'same_ticker_signals_30d', 'has_convergence', 'convergence_tier',
    'person_trade_count', 'person_hit_rate_30d', 'relative_position_size',
    'insider_role', 'sector', 'price_proximity_52wk', 'market_cap_bucket',
    'cluster_velocity', 'trade_pattern', 'disclosure_delay',
]

CATEGORICAL_FEATURES = [
    'source', 'insider_role', 'sector', 'market_cap_bucket',
    'cluster_velocity', 'trade_pattern',
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
    """Extract feature matrix X, target y, signal IDs, and dates from database.
    Returns (X, y, ids, dates) or (empty_df, empty, empty, empty) if no data.
    """
    import pandas as pd

    cols = ', '.join(FEATURE_COLUMNS)
    rows = conn.execute(
        f"SELECT id, signal_date, car_30d, {cols} "
        f"FROM signals WHERE outcome_30d_filled = 1 AND car_30d IS NOT NULL"
    ).fetchall()

    if not rows:
        return pd.DataFrame(), np.array([]), np.array([]), np.array([])

    data = [dict(r) for r in rows]
    df = pd.DataFrame(data)
    ids = df['id'].values
    y = (df['car_30d'] > 0).astype(int).values
    dates = df['signal_date'].values

    X = df[FEATURE_COLUMNS].copy()

    # Encode categoricals
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].fillna('unknown').astype(str)
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    X = X.fillna(0).infer_objects(copy=False)
    return X, y, ids, dates


def compute_information_coefficient(predicted, actual) -> float:
    """Spearman rank correlation between predictions and actual returns."""
    if len(predicted) < 3 or len(actual) < 3:
        return 0.0
    corr, _ = spearmanr(predicted, actual)
    return round(corr, 6) if not np.isnan(corr) else 0.0


def walk_forward_train(conn: sqlite3.Connection,
                       min_train_months: int = 6,
                       test_months: int = 1) -> WalkForwardResult:
    """Walk-forward validation with RF + LightGBM ensemble."""
    import pandas as pd

    X, y, ids, dates = prepare_features(conn)
    if len(X) < 50:
        log.warning(f"Insufficient data for ML training ({len(X)} signals)")
        return WalkForwardResult(n_folds=0, oos_ic=0, oos_hit_rate=0,
                                 oos_avg_car=0, feature_importance={})

    # Get actual CARs for IC computation
    car_values = {}
    for row_id in ids:
        r = conn.execute("SELECT car_30d FROM signals WHERE id=?", (int(row_id),)).fetchone()
        if r:
            car_values[row_id] = r['car_30d']

    # Sort by date
    date_series = pd.to_datetime(dates)
    sort_idx = date_series.argsort()
    X = X.iloc[sort_idx].reset_index(drop=True)
    y = y[sort_idx]
    ids = ids[sort_idx]
    date_series = date_series[sort_idx]

    # Walk-forward folds
    max_date = date_series.max()
    folds = []
    all_oos_preds = []
    all_oos_actual = []
    rf = None
    lgb_model = None

    train_end = date_series.min() + pd.DateOffset(months=min_train_months)

    while train_end + pd.DateOffset(months=test_months) <= max_date:
        test_end = train_end + pd.DateOffset(months=test_months)

        train_mask = date_series < train_end
        test_mask = (date_series >= train_end) & (date_series < test_end)

        if train_mask.sum() < 30 or test_mask.sum() < 5:
            train_end += pd.DateOffset(months=1)
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        test_ids = ids[test_mask]

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

        # Compute metrics
        test_cars = [car_values.get(tid, 0) for tid in test_ids]
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

    # Feature importance (from last fold's models)
    importance = {}
    if folds and rf is not None and lgb_model is not None:
        feat_names = list(X.columns)
        rf_imp = rf.feature_importances_
        lgb_imp = lgb_model.feature_importances_ / max(lgb_model.feature_importances_.sum(), 1)
        for i, name in enumerate(feat_names):
            importance[name] = round((rf_imp[i] + lgb_imp[i]) / 2, 4)

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
