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
from sklearn.metrics import roc_auc_score, brier_score_loss
import lightgbm as lgb

log = logging.getLogger(__name__)

# CAR winsorization bounds — clip extreme outliers before ML training
CAR_ABSOLUTE_MIN = -1.0   # -100%
CAR_ABSOLUTE_MAX = 3.0    # +300%

# Multi-horizon training
HORIZONS = ['30d', '90d', '180d']
MIN_SIGNALS_PER_HORIZON = 220

FEATURE_COLUMNS = [
    # Core signal features
    'trade_size_points', 'same_ticker_signals_7d',
    'same_ticker_signals_30d',
    'person_trade_count', 'person_hit_rate_30d',
    'insider_role', 'sector', 'price_proximity_52wk', 'market_cap_bucket',
    'disclosure_delay',
    # Macro context
    'vix_at_signal', 'yield_curve_at_signal', 'credit_spread_at_signal',
    'days_to_earnings',
    # Price / momentum features
    'momentum_1m', 'momentum_3m', 'momentum_6m',
    'volume_spike', 'insider_buy_ratio_90d', 'sector_avg_car',
    'vix_regime_interaction',
    # v4: person magnitude + sector momentum + repeat buyer signal
    'person_avg_car_30d', 'sector_momentum', 'days_since_last_buy',
    # v5: volume + analyst features
    'volume_dry_up', 'analyst_revision_30d', 'analyst_consensus',
    'analyst_insider_confluence',
    # v6: committee overlap + earnings surprise + news sentiment
    'committee_overlap', 'earnings_surprise', 'news_sentiment_30d',
]
# Pruned in v4: 'source' (0.16%), 'trade_pattern' (0.40%, 31% fill)
# Pruned in v5: 'convergence_tier' (<1%), 'has_convergence' (<1%),
#   'days_to_catalyst' (<1%), 'relative_position_size' (<1%),
#   'cluster_velocity' (<1%) — all <1% importance for 3+ runs

CATEGORICAL_FEATURES = [
    'insider_role', 'sector', 'market_cap_bucket',
]


METRIC_BENCHMARKS = {
    'ic':        {'min': 0.03, 'good': 0.07, 'excellent': 0.12},
    'ic_t_stat': {'min': 2.0,  'good': 3.0,  'excellent': 5.0},
    'hit_rate':  {'min': 0.52, 'good': 0.55, 'excellent': 0.58},
    'info_ratio':{'min': 0.3,  'good': 0.5,  'excellent': 1.0},
    'sortino':   {'min': 0.5,  'good': 1.0,  'excellent': 1.5},
    'bss':       {'min': 0.02, 'good': 0.10, 'excellent': 0.20},
    'auc_roc':   {'min': 0.52, 'good': 0.57, 'excellent': 0.65},
    'q5_q1':     {'min': 0.03, 'good': 0.07, 'excellent': 0.12},
    'profit_fac':{'min': 1.1,  'good': 1.5,  'excellent': 2.0},
}


def _benchmark_label(metric_name: str, value: float) -> str:
    """Return quality label for a metric value based on METRIC_BENCHMARKS."""
    bench = METRIC_BENCHMARKS.get(metric_name)
    if not bench:
        return ''
    if value >= bench['excellent']:
        return 'EXCELLENT'
    if value >= bench['good']:
        return 'GOOD'
    if value >= bench['min']:
        return 'OK'
    return 'WEAK'


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
    brier_score: float = 0.0
    auc_roc: float = 0.5
    profit_factor: float = 0.0
    ev_per_signal: float = 0.0


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
    horizon: str = '30d'
    # Stability metrics
    ic_std: float = 0.0
    ic_t_stat: float = 0.0
    # Risk-adjusted metrics
    information_ratio: float = 0.0
    sortino_ratio: float = 0.0
    # Calibration
    brier_skill_score: float = 0.0
    # Signal discrimination
    q5_q1_spread: float = 0.0
    top_decile_car: float = 0.0
    # Market loading
    beta: float = 0.0
    # Aggregate profit
    profit_factor: float = 0.0


def prepare_features(conn: sqlite3.Connection, horizon: str = '30d'):
    """Extract feature matrix X, target y, signal IDs, dates, and winsorized CARs.

    Args:
        horizon: '30d', '90d', or '180d' — selects car_{horizon} and outcome_{horizon}_filled.

    Returns (X, y, ids, dates, car_winsorized) or 5 empty arrays if no data.
    CAR winsorization applies percentile clipping (1st/99th) with hard bounds ±300%.
    """
    import pandas as pd

    car_col = f'car_{horizon}'
    outcome_col = f'outcome_{horizon}_filled'
    cols = ', '.join(FEATURE_COLUMNS)
    rows = conn.execute(
        f"SELECT id, signal_date, {car_col}, {cols} "
        f"FROM signals WHERE {outcome_col} = 1 AND {car_col} IS NOT NULL"
    ).fetchall()

    if not rows:
        return pd.DataFrame(), np.array([]), np.array([]), np.array([]), np.array([])

    data = [dict(r) for r in rows]
    df = pd.DataFrame(data)
    ids = df['id'].values
    dates = df['signal_date'].values

    # Winsorize CARs: percentile clip then hard bounds
    car_raw = df[car_col].copy()
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


def prepare_features_all(conn: sqlite3.Connection, horizon: str = '30d'):
    """Extract features for ALL signals (including those without outcomes).

    Used for scoring — no outcome filter so recent/actionable signals are included.
    Args:
        horizon: '30d', '90d', or '180d' — selects car_{horizon} for CARs.
    Returns (X, ids, dates, tickers, cars, X_raw) where cars may be NaN for pending
    signals and X_raw is the pre-encoded feature DataFrame for factor attribution.
    """
    import pandas as pd

    car_col = f'car_{horizon}'
    cols = ', '.join(FEATURE_COLUMNS)
    rows = conn.execute(
        f"SELECT id, signal_date, ticker, {car_col}, {cols} FROM signals"
    ).fetchall()

    if not rows:
        empty = pd.DataFrame()
        return empty, np.array([]), np.array([]), np.array([]), np.array([]), empty

    data = [dict(r) for r in rows]
    df = pd.DataFrame(data)
    ids = df['id'].values
    dates = df['signal_date'].values
    tickers = df['ticker'].values
    cars = df[car_col].values.astype(float)

    X_raw = df[FEATURE_COLUMNS].copy().fillna(0).infer_objects(copy=False)

    X = df[FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].fillna('unknown').astype(str)
            X[col] = X[col].astype('category').cat.codes

    X = X.fillna(0).infer_objects(copy=False)
    return X, ids, dates, tickers, cars, X_raw


def _compute_time_weights(dates, half_life_months: int = 12) -> np.ndarray:
    """Compute exponential decay weights so recent signals matter more.

    w_i = 0.5^(age_months / half_life). A 12-month half-life means
    signals from 1 year ago have half the weight of today's signals.
    Weights are clipped to [0.1, 1.0] to avoid zero-weight old signals.
    """
    import pandas as pd
    date_series = pd.to_datetime(dates)
    max_date = date_series.max()
    age_days = (max_date - date_series).dt.days.values.astype(float)
    age_months = age_days / 30.44  # average days per month
    weights = np.power(0.5, age_months / half_life_months)
    return np.clip(weights, 0.1, 1.0)


def train_full_sample(conn: sqlite3.Connection, horizon: str = '30d'):
    """Train 4 models on ALL historical data with outcomes (full sample, not walk-forward).

    Uses time-weighted sampling: recent signals weighted higher (12-month half-life).
    Returns (clf_rf, clf_lgb, reg_rf, reg_lgb) or None if insufficient data.
    """
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor

    X, y, ids, dates, car_winsorized = prepare_features(conn, horizon=horizon)
    if len(X) < 50:
        log.warning(f"Insufficient data for full-sample training ({len(X)} signals)")
        return None

    # Time-weighted: recent signals matter more
    sw = _compute_time_weights(dates)

    # Classification: P(beat SPY)
    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    clf_rf.fit(X, y, sample_weight=sw)

    clf_lgb = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42,
                                  verbose=-1, n_jobs=-1)
    clf_lgb.fit(X, y, sample_weight=sw)

    # Regression: predicted CAR magnitude
    reg_rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    reg_rf.fit(X, car_winsorized, sample_weight=sw)

    reg_lgb = lgb.LGBMRegressor(n_estimators=100, max_depth=6, random_state=42,
                                 verbose=-1, n_jobs=-1)
    reg_lgb.fit(X, car_winsorized, sample_weight=sw)

    log.info(f"Full-sample models trained on {len(X)} signals")
    return clf_rf, clf_lgb, reg_rf, reg_lgb


def compute_information_coefficient(predicted, actual) -> float:
    """Spearman rank correlation between predictions and actual returns."""
    if len(predicted) < 3 or len(actual) < 3:
        return 0.0
    corr, _ = spearmanr(predicted, actual)
    return round(corr, 6) if not np.isnan(corr) else 0.0


def compute_fold_metrics(probs, y_test, test_cars):
    """Compute per-fold enhanced metrics.

    Args:
        probs: ensemble probabilities (clf) or predictions (reg)
        y_test: binary labels (clf) or continuous CARs (reg)
        test_cars: winsorized CARs for this fold

    Returns dict with brier_score, auc_roc, profit_factor, ev_per_signal.
    """
    metrics = {'brier_score': 0.0, 'auc_roc': 0.5, 'profit_factor': 0.0, 'ev_per_signal': 0.0}

    # Brier score — only meaningful for classification probs
    if len(y_test) > 0 and set(np.unique(y_test)).issubset({0, 1}):
        try:
            metrics['brier_score'] = round(float(brier_score_loss(y_test, probs)), 6)
        except Exception:
            pass

    # AUC-ROC — needs both classes present
    if len(y_test) > 1 and len(np.unique(y_test)) == 2:
        try:
            metrics['auc_roc'] = round(float(roc_auc_score(y_test, probs)), 6)
        except Exception:
            pass

    # Profit factor and EV per signal
    cars_arr = np.array(test_cars)
    if len(cars_arr) > 0:
        wins = cars_arr[cars_arr > 0]
        losses = cars_arr[cars_arr < 0]
        total_wins = float(wins.sum()) if len(wins) > 0 else 0.0
        total_losses = abs(float(losses.sum())) if len(losses) > 0 else 0.0
        metrics['profit_factor'] = round(total_wins / total_losses, 4) if total_losses > 0 else (
            float('inf') if total_wins > 0 else 0.0)

        hit_rate = len(wins) / len(cars_arr)
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
        metrics['ev_per_signal'] = round(hit_rate * avg_win + (1 - hit_rate) * avg_loss, 6)

    return metrics


def compute_aggregate_metrics(all_preds, all_actual_cars, fold_ics, is_clf=True):
    """Compute aggregate metrics from accumulated OOS predictions.

    Args:
        all_preds: list of OOS predictions (probs for clf, CARs for reg)
        all_actual_cars: list of actual CARs
        fold_ics: list of per-fold IC values
        is_clf: True for classification, False for regression

    Returns dict with ic_std, ic_t_stat, information_ratio, sortino_ratio,
    brier_skill_score (clf only), q5_q1_spread, top_decile_car, beta, profit_factor.
    """
    result = {
        'ic_std': 0.0, 'ic_t_stat': 0.0,
        'information_ratio': 0.0, 'sortino_ratio': 0.0,
        'brier_skill_score': 0.0,
        'q5_q1_spread': 0.0, 'top_decile_car': 0.0,
        'beta': 0.0, 'profit_factor': 0.0,
    }

    preds = np.array(all_preds)
    actuals = np.array(all_actual_cars)

    if len(preds) < 5 or len(actuals) < 5:
        return result

    # IC stability
    ic_arr = np.array(fold_ics)
    if len(ic_arr) > 1:
        result['ic_std'] = round(float(np.std(ic_arr, ddof=1)), 6)
        if result['ic_std'] > 0:
            result['ic_t_stat'] = round(float(np.mean(ic_arr) / (result['ic_std'] / np.sqrt(len(ic_arr)))), 4)

    # Information ratio: mean(CARs) / std(CARs)
    if np.std(actuals) > 0:
        result['information_ratio'] = round(float(np.mean(actuals) / np.std(actuals)), 4)

    # Sortino ratio: mean(CARs) / std(CARs where CAR < 0)
    downside = actuals[actuals < 0]
    if len(downside) > 1 and np.std(downside) > 0:
        result['sortino_ratio'] = round(float(np.mean(actuals) / np.std(downside)), 4)

    # Brier skill score (clf only)
    if is_clf:
        y_binary = (actuals > 0).astype(int)
        naive_brier = float(np.mean((np.mean(y_binary) - y_binary) ** 2))
        actual_brier = float(np.mean((preds - y_binary) ** 2))
        if naive_brier > 0:
            result['brier_skill_score'] = round(1.0 - (actual_brier / naive_brier), 6)

    # Q5-Q1 spread: mean(top 20% by pred) - mean(bottom 20% by pred)
    sort_idx = np.argsort(preds)
    n = len(preds)
    q_size = max(1, n // 5)
    bottom_20 = actuals[sort_idx[:q_size]]
    top_20 = actuals[sort_idx[-q_size:]]
    result['q5_q1_spread'] = round(float(np.mean(top_20) - np.mean(bottom_20)), 6)

    # Top decile CAR
    d_size = max(1, n // 10)
    top_10 = actuals[sort_idx[-d_size:]]
    result['top_decile_car'] = round(float(np.mean(top_10)), 6)

    # Beta: cov(signal_CARs, SPY_proxy) / var(SPY_proxy)
    # We approximate SPY exposure by using the mean CAR as a constant market return
    # True beta requires SPY returns aligned by date — deferred to Phase 2
    result['beta'] = 0.0  # placeholder until SPY alignment available

    # Aggregate profit factor
    wins = actuals[actuals > 0]
    losses = actuals[actuals < 0]
    total_wins = float(wins.sum()) if len(wins) > 0 else 0.0
    total_losses = abs(float(losses.sum())) if len(losses) > 0 else 0.0
    if total_losses > 0:
        result['profit_factor'] = round(total_wins / total_losses, 4)
    elif total_wins > 0:
        result['profit_factor'] = float('inf')

    return result


def calibrate_probabilities(probs, actuals):
    """Calibrate probability outputs using isotonic regression.

    Args:
        probs: raw probability predictions (array-like)
        actuals: binary outcomes (array-like, 0/1)

    Returns calibrated probability array, monotonic and bounded [0, 1].
    """
    from sklearn.isotonic import IsotonicRegression

    probs = np.asarray(probs, dtype=float)
    actuals = np.asarray(actuals, dtype=float)

    if len(probs) < 3:
        return probs

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    calibrated = iso.fit_transform(probs, actuals)
    return np.clip(calibrated, 0.0, 1.0)


def walk_forward_train(conn: sqlite3.Connection,
                       min_train_months: int = 6,
                       test_months: int = 1,
                       min_train_samples: int = 200,
                       min_test_samples: int = 20,
                       horizon: str = '30d') -> WalkForwardResult:
    """Walk-forward validation with RF + LightGBM ensemble."""
    import pandas as pd

    X, y, ids, dates, car_winsorized = prepare_features(conn, horizon=horizon)
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

        # Time-weighted: recent training signals matter more
        train_sw = _compute_time_weights(date_series[train_mask].values)

        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train, sample_weight=train_sw)
        rf_probs = rf.predict_proba(X_test)[:, 1]

        # Train LightGBM
        lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42,
                                        verbose=-1, n_jobs=-1)
        lgb_model.fit(X_train, y_train, sample_weight=train_sw)
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

        # Per-fold enhanced metrics
        fold_metrics = compute_fold_metrics(ensemble_probs, y_test, test_cars)

        fold = FoldResult(
            train_start=str(date_series[train_mask].min().date()),
            train_end=str(train_end.date()),
            test_start=str(train_end.date()),
            test_end=str(test_end.date()),
            n_train=int(train_mask.sum()),
            n_test=int(test_mask.sum()),
            ic=ic, hit_rate=round(hit_rate, 4), avg_car=round(avg_car, 6),
            brier_score=fold_metrics['brier_score'],
            auc_roc=fold_metrics['auc_roc'],
            profit_factor=fold_metrics['profit_factor'],
            ev_per_signal=fold_metrics['ev_per_signal'],
        )
        folds.append(fold)
        all_oos_preds.extend(ensemble_probs.tolist())
        all_oos_actual.extend(test_cars)

        train_end += pd.DateOffset(months=1)

    # Aggregate OOS metrics
    oos_ic = compute_information_coefficient(all_oos_preds, all_oos_actual)
    oos_hit = sum(1 for p, c in zip(all_oos_preds, all_oos_actual) if (p > 0.5) == (c > 0)) / max(len(all_oos_preds), 1)
    oos_avg_car = np.mean(all_oos_actual) if all_oos_actual else 0

    # Enhanced aggregate metrics
    fold_ics = [f.ic for f in folds]
    agg = compute_aggregate_metrics(all_oos_preds, all_oos_actual, fold_ics, is_clf=True)

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

    # Log key metrics with benchmark labels
    ic_label = _benchmark_label('ic', abs(oos_ic))
    log.info(f"Walk-forward CLF [{horizon}]: IC={oos_ic:.4f} [{ic_label}], "
             f"t={agg['ic_t_stat']:.2f}, hit={oos_hit:.1%}, "
             f"Q5-Q1={agg['q5_q1_spread']:.4f}, PF={agg['profit_factor']:.2f}")

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
            'brier_score': f.brier_score, 'auc_roc': f.auc_roc,
            'profit_factor': f.profit_factor, 'ev_per_signal': f.ev_per_signal,
        } for f in folds],
        model_rf=rf, model_lgb=lgb_model,
        horizon=horizon,
        ic_std=agg['ic_std'],
        ic_t_stat=agg['ic_t_stat'],
        information_ratio=agg['information_ratio'],
        sortino_ratio=agg['sortino_ratio'],
        brier_skill_score=agg['brier_skill_score'],
        q5_q1_spread=agg['q5_q1_spread'],
        top_decile_car=agg['top_decile_car'],
        beta=agg['beta'],
        profit_factor=agg['profit_factor'],
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
    horizon: str = '30d'
    # Stability metrics
    ic_std: float = 0.0
    ic_t_stat: float = 0.0
    # Risk-adjusted metrics
    information_ratio: float = 0.0
    sortino_ratio: float = 0.0
    # Signal discrimination
    q5_q1_spread: float = 0.0
    top_decile_car: float = 0.0
    # Aggregate profit
    profit_factor: float = 0.0


def walk_forward_regression(conn: sqlite3.Connection,
                             min_train_months: int = 6,
                             test_months: int = 1,
                             min_train_samples: int = 200,
                             min_test_samples: int = 20,
                             horizon: str = '30d') -> RegressionResult:
    """Walk-forward regression with RF + LightGBM ensemble.

    Targets continuous winsorized CAR (not binary). Output is predicted
    CAR magnitude for signal ranking — richer than direction-only classification.
    """
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

    X, y_cls, ids, dates, car_winsorized = prepare_features(conn, horizon=horizon)
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

        # Time-weighted: recent training signals matter more
        train_sw = _compute_time_weights(date_series[train_mask].values)

        # Train Random Forest Regressor
        rf = RandomForestRegressor(n_estimators=100, max_depth=6,
                                    random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train, sample_weight=train_sw)
        rf_preds = rf.predict(X_test)

        # Train LightGBM Regressor
        lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=6,
                                       random_state=42, verbose=-1, n_jobs=-1)
        lgb_model.fit(X_train, y_train, sample_weight=train_sw)
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

        # Per-fold enhanced metrics (regression uses CARs as "probs" proxy)
        fold_metrics = compute_fold_metrics(ensemble_preds, (y_test > 0).astype(int), y_test.tolist())

        fold = FoldResult(
            train_start=str(date_series[train_mask].min().date()),
            train_end=str(train_end.date()),
            test_start=str(train_end.date()),
            test_end=str(test_end.date()),
            n_train=int(train_mask.sum()),
            n_test=int(test_mask.sum()),
            ic=ic, hit_rate=round(hit_rate, 4), avg_car=round(avg_car, 6),
            profit_factor=fold_metrics['profit_factor'],
            ev_per_signal=fold_metrics['ev_per_signal'],
        )
        folds.append(fold)
        all_oos_preds.extend(ensemble_preds.tolist())
        all_oos_actual.extend(y_test.tolist())

        train_end += pd.DateOffset(months=1)

    # Aggregate OOS metrics
    oos_ic = compute_information_coefficient(all_oos_preds, all_oos_actual)
    oos_rmse = float(np.sqrt(np.mean((np.array(all_oos_preds) - np.array(all_oos_actual))**2))) if all_oos_actual else 0
    oos_avg_car = float(np.mean(all_oos_actual)) if all_oos_actual else 0

    # Enhanced aggregate metrics
    fold_ics = [f.ic for f in folds]
    agg = compute_aggregate_metrics(all_oos_preds, all_oos_actual, fold_ics, is_clf=False)

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

    # Log key metrics with benchmark labels
    ic_label = _benchmark_label('ic', abs(oos_ic))
    log.info(f"Walk-forward REG [{horizon}]: IC={oos_ic:.4f} [{ic_label}], "
             f"t={agg['ic_t_stat']:.2f}, RMSE={oos_rmse:.4f}, "
             f"Q5-Q1={agg['q5_q1_spread']:.4f}, PF={agg['profit_factor']:.2f}")

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
            'profit_factor': f.profit_factor, 'ev_per_signal': f.ev_per_signal,
        } for f in folds],
        model_rf=rf, model_lgb=lgb_model,
        horizon=horizon,
        ic_std=agg['ic_std'],
        ic_t_stat=agg['ic_t_stat'],
        information_ratio=agg['information_ratio'],
        sortino_ratio=agg['sortino_ratio'],
        q5_q1_spread=agg['q5_q1_spread'],
        top_decile_car=agg['top_decile_car'],
        profit_factor=agg['profit_factor'],
    )


# ---------------------------------------------------------------------------
# Multi-horizon training
# ---------------------------------------------------------------------------

@dataclass
class HorizonResult:
    horizon: str
    clf_result: WalkForwardResult
    reg_result: RegressionResult
    ic_clf: float
    ic_reg: float


@dataclass
class MultiHorizonResult:
    horizons: dict          # {'30d': HorizonResult, ...}
    horizon_weights: dict   # IC-weighted: {'30d': 0.55, '90d': 0.30, '180d': 0.15}
    composite_ic: float
    ic_decay_rate: float    # (IC_30d - IC_180d) / IC_30d — signal persistence


def train_multi_horizon(conn, **kwargs):
    """Train walk-forward CLF + REG for each horizon, compute IC-weighted blend.

    Args:
        conn: SQLite connection
        **kwargs: forwarded to walk_forward_train / walk_forward_regression

    Returns MultiHorizonResult.
    """
    results = {}
    for h in HORIZONS:
        # Check signal count for this horizon
        car_col = f'car_{h}'
        outcome_col = f'outcome_{h}_filled'
        count = conn.execute(
            f"SELECT COUNT(*) as cnt FROM signals WHERE {outcome_col}=1 AND {car_col} IS NOT NULL"
        ).fetchone()['cnt']

        if count < MIN_SIGNALS_PER_HORIZON:
            log.info(f"Skipping horizon {h}: {count} signals < {MIN_SIGNALS_PER_HORIZON} minimum")
            continue

        clf = walk_forward_train(conn, horizon=h, **kwargs)
        reg = walk_forward_regression(conn, horizon=h, **kwargs)

        results[h] = HorizonResult(
            horizon=h,
            clf_result=clf,
            reg_result=reg,
            ic_clf=clf.oos_ic,
            ic_reg=reg.oos_ic,
        )

    if not results:
        log.warning("No horizons had sufficient data for training")
        return MultiHorizonResult(horizons={}, horizon_weights={},
                                   composite_ic=0.0, ic_decay_rate=0.0)

    # IC-weighted blending: weight_h = max(0, IC_h) / sum(max(0, IC_h))
    raw_weights = {h: max(0, r.ic_clf) for h, r in results.items()}
    weight_sum = sum(raw_weights.values())

    if weight_sum > 0:
        horizon_weights = {h: round(w / weight_sum, 4) for h, w in raw_weights.items()}
    else:
        # Uniform fallback
        n = len(results)
        horizon_weights = {h: round(1.0 / n, 4) for h in results}

    # Composite IC
    composite_ic = sum(horizon_weights.get(h, 0) * r.ic_clf for h, r in results.items())

    # IC decay rate: (IC_30d - IC_180d) / IC_30d
    ic_30 = results['30d'].ic_clf if '30d' in results else 0
    ic_180 = results['180d'].ic_clf if '180d' in results else 0
    ic_decay_rate = (ic_30 - ic_180) / ic_30 if ic_30 != 0 else 0.0

    log.info(f"Multi-horizon: weights={horizon_weights}, composite_IC={composite_ic:.4f}, "
             f"decay={ic_decay_rate:.2f}")

    return MultiHorizonResult(
        horizons=results,
        horizon_weights=horizon_weights,
        composite_ic=round(composite_ic, 6),
        ic_decay_rate=round(ic_decay_rate, 4),
    )


def train_full_sample_multi(conn):
    """Train full-sample models for each horizon with sufficient data.

    Returns dict[str, tuple] — horizon → (clf_rf, clf_lgb, reg_rf, reg_lgb).
    Skips horizons with insufficient data.
    """
    models = {}
    for h in HORIZONS:
        car_col = f'car_{h}'
        outcome_col = f'outcome_{h}_filled'
        count = conn.execute(
            f"SELECT COUNT(*) as cnt FROM signals WHERE {outcome_col}=1 AND {car_col} IS NOT NULL"
        ).fetchone()['cnt']

        if count < MIN_SIGNALS_PER_HORIZON:
            log.info(f"Skipping full-sample {h}: {count} signals < {MIN_SIGNALS_PER_HORIZON}")
            continue

        result = train_full_sample(conn, horizon=h)
        if result is not None:
            models[h] = result

    log.info(f"Full-sample multi: trained {list(models.keys())}")
    return models


def evaluate_feature_candidates(conn: sqlite3.Connection,
                                 candidates: list[str],
                                 horizon: str = '30d') -> dict:
    """Test whether adding candidate feature columns improves OOS IC.

    Runs walk-forward with baseline features (current FEATURE_COLUMNS),
    then with each candidate added individually. Reports IC delta.

    Args:
        conn: SQLite connection
        candidates: list of column names to evaluate (must exist in DB)
        horizon: which CAR horizon to evaluate against

    Returns dict with baseline IC and per-candidate IC + delta.
    """
    import pandas as pd

    # 1. Baseline walk-forward with current features
    baseline = walk_forward_train(conn, horizon=horizon)
    baseline_ic = baseline.oos_ic

    results = {
        'baseline_ic': baseline_ic,
        'baseline_features': len(FEATURE_COLUMNS),
        'horizon': horizon,
        'candidates': {},
    }

    log.info(f"Feature eval baseline: IC={baseline_ic:.4f} ({len(FEATURE_COLUMNS)} features)")

    # 2. For each candidate, temporarily add to features and re-run
    for candidate in candidates:
        # Check if column exists in DB
        try:
            conn.execute(f"SELECT {candidate} FROM signals LIMIT 1")
        except Exception:
            results['candidates'][candidate] = {
                'status': 'error',
                'reason': 'column not found in DB',
            }
            continue

        # Check fill rate
        total = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        filled = conn.execute(
            f"SELECT COUNT(*) FROM signals WHERE {candidate} IS NOT NULL"
        ).fetchone()[0]
        fill_rate = filled / max(total, 1)

        if fill_rate < 0.1:
            results['candidates'][candidate] = {
                'status': 'skip',
                'reason': f'fill rate too low ({fill_rate:.1%})',
                'fill_rate': round(fill_rate, 3),
            }
            continue

        # Temporarily extend FEATURE_COLUMNS
        original_cols = FEATURE_COLUMNS.copy()
        FEATURE_COLUMNS.append(candidate)
        try:
            candidate_result = walk_forward_train(conn, horizon=horizon)
            ic_delta = candidate_result.oos_ic - baseline_ic
            results['candidates'][candidate] = {
                'status': 'evaluated',
                'ic': round(candidate_result.oos_ic, 6),
                'ic_delta': round(ic_delta, 6),
                'hit_rate': candidate_result.oos_hit_rate,
                'fill_rate': round(fill_rate, 3),
                'recommendation': 'ADD' if ic_delta > 0.005 else (
                    'NEUTRAL' if ic_delta > -0.005 else 'SKIP'
                ),
            }
            log.info(f"  {candidate}: IC={candidate_result.oos_ic:.4f} "
                     f"(delta={ic_delta:+.4f}) → "
                     f"{'ADD' if ic_delta > 0.005 else 'SKIP'}")
        except Exception as e:
            results['candidates'][candidate] = {
                'status': 'error',
                'reason': str(e),
            }
        finally:
            # Restore original features
            FEATURE_COLUMNS.clear()
            FEATURE_COLUMNS.extend(original_cols)

    return results
