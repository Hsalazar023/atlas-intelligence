"""
learning_engine.py — ATLAS Adaptive Learning Engine (ALE).

Accumulates signal observations, tracks real outcomes, discovers which
features predict positive returns, and tunes scoring weights automatically.

Usage:
    python backtest/learning_engine.py --daily      # daily collect + backfill
    python backtest/learning_engine.py --analyze     # feature analysis + weight update
    python backtest/learning_engine.py --summary     # print status to console
    python backtest/learning_engine.py --bootstrap   # alias for bootstrap_historical.py

Database: data/atlas_signals.db
Dashboard: data/ale_dashboard.json
"""

import sys
import json
import sqlite3
import logging
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.shared import (
    DATA_DIR, SIGNALS_DB, CONGRESS_FEED, EDGAR_FEED, PRICE_HISTORY_DIR,
    OPTIMAL_WEIGHTS, BACKTEST_SUMMARY, DEFAULT_WEIGHTS,
    BRAIN_SIGNALS, BRAIN_STATS, BRAIN_HEALTH, THIRTEENF_FEED,
    load_json, save_json, match_edgar_ticker, range_to_base_points,
)
from backtest.sector_map import get_sector, get_market_cap, get_market_cap_bucket

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logging.getLogger('numexpr').setLevel(logging.WARNING)
log = logging.getLogger(__name__)

ALE_DASHBOARD = DATA_DIR / "ale_dashboard.json"
FEATURE_CANDIDATES = DATA_DIR / "feature_candidates.json"
SIGNAL_HYPOTHESES = DATA_DIR / "signal_hypotheses.json"
ALE_ANALYSIS_REPORT = DATA_DIR / "ale_analysis_report.md"
ALE_DIAGNOSTICS_HTML = DATA_DIR / "ale_diagnostics.html"
ANALYST_REPORT = DATA_DIR / "analyst_report.json"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
MODELS_CACHE = DATA_DIR / "models_cache.pkl"
SIGNAL_INTELLIGENCE = DATA_DIR / "signal_intelligence.json"
PORTFOLIO_STATS = DATA_DIR / "portfolio_stats.json"

# ── Database Setup ───────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    signal_date TEXT NOT NULL,
    source TEXT NOT NULL,

    -- Congressional features
    representative TEXT,
    party TEXT,
    chamber TEXT,
    trade_size_range TEXT,
    trade_size_points REAL,

    -- EDGAR/Insider features
    insider_name TEXT,
    insider_role TEXT,
    transaction_type TEXT,

    -- Aggregate features (computed at insertion)
    same_ticker_signals_7d INTEGER DEFAULT 0,
    same_ticker_signals_30d INTEGER DEFAULT 0,
    has_convergence INTEGER DEFAULT 0,
    convergence_sources TEXT,
    convergence_tier INTEGER DEFAULT 0,
    convergence_sector TEXT,
    convergence_tickers TEXT,
    cluster_velocity TEXT,
    disclosure_delay INTEGER,
    total_score REAL,

    -- Research-backed features
    price_proximity_52wk REAL,
    market_cap_bucket TEXT,
    relative_buy_size REAL,
    sector_momentum REAL,
    trade_pattern TEXT,

    -- Catalyst proximity
    days_to_earnings INTEGER,
    days_to_catalyst INTEGER,

    -- Momentum & derived features
    momentum_1m REAL,
    momentum_3m REAL,
    momentum_6m REAL,
    volume_spike REAL,
    insider_buy_ratio_90d REAL,
    sector_avg_car REAL,
    vix_regime_interaction REAL,

    -- Market context
    sector TEXT,
    vix_at_signal REAL,
    yield_curve_at_signal REAL,
    credit_spread_at_signal REAL,

    -- Outcomes (backfilled as time passes)
    price_at_signal REAL,
    return_5d REAL,
    return_30d REAL,
    return_90d REAL,
    return_180d REAL,
    return_365d REAL,
    car_5d REAL,
    car_30d REAL,
    car_90d REAL,
    car_180d REAL,
    car_365d REAL,
    outcome_5d_filled INTEGER DEFAULT 0,
    outcome_30d_filled INTEGER DEFAULT 0,
    outcome_90d_filled INTEGER DEFAULT 0,
    outcome_180d_filled INTEGER DEFAULT 0,
    outcome_365d_filled INTEGER DEFAULT 0,

    -- Person-level features (computed from historical data)
    person_trade_count INTEGER DEFAULT 0,       -- total prior trades by this person
    person_hit_rate_30d REAL,                   -- their historical 30d hit rate
    person_avg_car_30d REAL,                    -- their historical avg 30d CAR
    person_hit_rate_90d REAL,                   -- their historical 90d hit rate
    person_avg_car_90d REAL,                    -- their historical avg 90d CAR
    relative_position_size REAL,                -- this trade size vs their median (1.0 = typical)

    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(ticker, signal_date, source, representative, insider_name)
);

-- Performance indexes (idempotent via IF NOT EXISTS)
CREATE INDEX IF NOT EXISTS idx_signals_ticker_date ON signals(ticker, signal_date);
CREATE INDEX IF NOT EXISTS idx_signals_source_date ON signals(source, signal_date);
CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(signal_date);
CREATE INDEX IF NOT EXISTS idx_signals_score ON signals(total_score DESC);
CREATE INDEX IF NOT EXISTS idx_signals_sector ON signals(sector);
CREATE INDEX IF NOT EXISTS idx_signals_representative ON signals(representative);
CREATE INDEX IF NOT EXISTS idx_signals_outcome ON signals(outcome_30d_filled, car_30d);

CREATE TABLE IF NOT EXISTS feature_stats (
    feature_name TEXT NOT NULL,
    feature_value TEXT NOT NULL,
    n_observations INTEGER DEFAULT 0,
    positive_rate_30d REAL,
    avg_car_30d REAL,
    avg_car_90d REAL,
    avg_car_180d REAL,
    avg_car_365d REAL,
    last_updated TEXT,
    PRIMARY KEY (feature_name, feature_value)
);

CREATE TABLE IF NOT EXISTS weight_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    weights_json TEXT NOT NULL,
    n_signals INTEGER,
    hit_rate_30d REAL,
    avg_car_30d REAL,
    method TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS feature_importance_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    importance REAL NOT NULL,
    rank INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_feat_hist_date ON feature_importance_history(run_date);
CREATE INDEX IF NOT EXISTS idx_feat_hist_name ON feature_importance_history(feature_name);

CREATE TABLE IF NOT EXISTS brain_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date TEXT NOT NULL,
    run_type TEXT NOT NULL,
    oos_ic REAL,
    oos_hit_rate REAL,
    n_signals INTEGER,
    n_scored INTEGER,
    avg_score REAL,
    max_score REAL,
    top_ticker TEXT,
    top_ticker_pct REAL,
    feature_importance_json TEXT,
    notes TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
"""


def init_db(db_path: Path = None) -> sqlite3.Connection:
    """Create tables if they don't exist, return connection."""
    path = db_path or SIGNALS_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    # Migrate: add new columns to existing tables (ALTER TABLE ADD is safe — no-ops if col exists)
    _migrate_columns(conn)
    conn.commit()
    return conn


def _migrate_columns(conn: sqlite3.Connection) -> None:
    """Add new columns to existing tables if they don't already exist."""
    new_signal_cols = [
        ("return_180d", "REAL"),
        ("return_365d", "REAL"),
        ("car_180d", "REAL"),
        ("car_365d", "REAL"),
        ("outcome_180d_filled", "INTEGER DEFAULT 0"),
        ("outcome_365d_filled", "INTEGER DEFAULT 0"),
        ("person_trade_count", "INTEGER DEFAULT 0"),
        ("person_hit_rate_30d", "REAL"),
        ("person_avg_car_30d", "REAL"),
        ("person_hit_rate_90d", "REAL"),
        ("person_avg_car_90d", "REAL"),
        ("relative_position_size", "REAL"),
        ("convergence_tier", "INTEGER DEFAULT 0"),
        ("convergence_sector", "TEXT"),
        ("convergence_tickers", "TEXT"),
        ("cluster_velocity", "TEXT"),
        ("disclosure_delay", "INTEGER"),
        ("price_proximity_52wk", "REAL"),
        ("market_cap_bucket", "TEXT"),
        ("relative_buy_size", "REAL"),
        ("sector_momentum", "REAL"),
        ("trade_pattern", "TEXT"),
        ("vix_at_signal", "REAL"),
        ("yield_curve_at_signal", "REAL"),
        ("credit_spread_at_signal", "REAL"),
        ("days_to_earnings", "INTEGER"),
        ("days_to_catalyst", "INTEGER"),
        # New alpha features (v3)
        ("momentum_1m", "REAL"),
        ("momentum_3m", "REAL"),
        ("momentum_6m", "REAL"),
        ("volume_spike", "REAL"),
        ("insider_buy_ratio_90d", "REAL"),
        ("sector_avg_car", "REAL"),
        ("vix_regime_interaction", "REAL"),
        # v4: person magnitude + sector momentum + repeat buyer signal
        ("sector_momentum", "REAL"),
        ("days_since_last_buy", "REAL"),
        # v5: volume + analyst features
        ("volume_dry_up", "INTEGER DEFAULT 0"),
        ("analyst_revision_30d", "INTEGER"),
        ("analyst_consensus", "REAL"),
        ("analyst_insider_confluence", "INTEGER DEFAULT 0"),
        # v6: committee overlap + earnings surprise + news sentiment
        ("committee_overlap", "INTEGER DEFAULT 0"),
        ("earnings_surprise", "REAL"),
        ("news_sentiment_30d", "REAL"),
        # v7: FinBERT sentiment features
        ("news_sentiment_score", "REAL"),
        ("news_sentiment_strong_positive", "INTEGER DEFAULT 0"),
        ("news_sentiment_strong_negative", "INTEGER DEFAULT 0"),
        ("news_insider_confluence", "INTEGER DEFAULT 0"),
        ("sentiment_divergence", "INTEGER DEFAULT 0"),
        # v7: market regime
        ("market_regime", "TEXT"),
        # v7: lobbying features
        ("lobbying_active", "INTEGER DEFAULT 0"),
        ("lobbying_trend", "REAL"),
        ("lobby_congress_confluence", "INTEGER DEFAULT 0"),
        # v8: hypothesis-driven interaction features
        ("sect_ticker_momentum", "REAL"),
        ("volume_cluster_signal", "REAL"),
        # v8: market-adjusted returns
        ("spy_return_30d", "REAL"),
        ("market_adj_car_30d", "REAL"),
        # v9: short interest features
        ("short_interest_pct", "REAL"),
        ("short_interest_change", "REAL"),
        ("short_squeeze_signal", "INTEGER DEFAULT 0"),
        # v10: institutional ownership features
        ("institutional_holders", "INTEGER"),
        ("institutional_pct_held", "REAL"),
        ("institutional_insider_confluence", "INTEGER DEFAULT 0"),
        # v10: options flow features
        ("options_bullish", "INTEGER DEFAULT 0"),
        ("options_unusual_calls", "INTEGER DEFAULT 0"),
        ("options_insider_confluence", "INTEGER DEFAULT 0"),
        ("options_bearish_divergence", "INTEGER DEFAULT 0"),
        # v11: accession number for EDGAR dedup
        ("accession_number", "TEXT"),
        # v12: OOS walk-forward predictions (honest scores)
        ("oos_score", "REAL"),
        ("oos_fold", "INTEGER"),
        # v13: liquidity / transaction cost features
        ("avg_daily_volume", "REAL"),
        ("estimated_spread", "REAL"),
        ("liquidity_flag", "TEXT"),
        ("net_expected_return", "REAL"),
        # v14: score breakdown columns (moved from score_all_signals for reliability)
        ("score_base", "REAL"),
        ("score_magnitude", "REAL"),
        ("score_converge", "REAL"),
        ("score_person", "REAL"),
    ]
    new_feature_cols = [
        ("avg_car_180d", "REAL"),
        ("avg_car_365d", "REAL"),
    ]
    for col_name, col_type in new_signal_cols:
        try:
            conn.execute(f"ALTER TABLE signals ADD COLUMN {col_name} {col_type}")
        except sqlite3.OperationalError:
            pass  # column already exists
    for col_name, col_type in new_feature_cols:
        try:
            conn.execute(f"ALTER TABLE feature_stats ADD COLUMN {col_name} {col_type}")
        except sqlite3.OperationalError:
            pass
    # brain_runs migrations
    for col_name, col_type in [("step_counts", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE brain_runs ADD COLUMN {col_name} {col_type}")
        except sqlite3.OperationalError:
            pass


# ── Signal Insertion ─────────────────────────────────────────────────────────

def insert_signal(conn: sqlite3.Connection, signal: dict) -> bool:
    """Insert a signal event. Returns True if inserted, False if duplicate."""
    cols = [
        'ticker', 'signal_date', 'source',
        'representative', 'party', 'chamber',
        'trade_size_range', 'trade_size_points',
        'insider_name', 'insider_role', 'transaction_type',
        'same_ticker_signals_7d', 'same_ticker_signals_30d',
        'has_convergence', 'convergence_sources', 'convergence_tier',
        'convergence_sector', 'convergence_tickers', 'cluster_velocity',
        'disclosure_delay', 'total_score', 'sector', 'price_at_signal',
        'price_proximity_52wk', 'market_cap_bucket', 'relative_buy_size',
        'sector_momentum', 'trade_pattern', 'accession_number',
    ]
    values = {k: signal.get(k) for k in cols}
    # Ensure UNIQUE constraint fields are never NULL (SQLite treats NULLs as distinct)
    for key in ('representative', 'insider_name'):
        if values.get(key) is None:
            values[key] = ''
    placeholders = ', '.join(f':{k}' for k in cols)
    col_names = ', '.join(cols)
    try:
        before = conn.total_changes
        conn.execute(
            f"INSERT OR IGNORE INTO signals ({col_names}) VALUES ({placeholders})",
            values,
        )
        conn.commit()
        return conn.total_changes > before
    except sqlite3.Error as e:
        log.warning(f"Insert failed for {signal.get('ticker')}: {e}")
        return False


def ingest_congress_feed(conn: sqlite3.Connection, feed_path: Path = None) -> int:
    """Load congress_feed.json and insert purchase signals. Returns count inserted."""
    path = feed_path or CONGRESS_FEED
    if not path.exists():
        log.warning(f"Congress feed not found: {path}")
        return 0

    data = load_json(path)
    trades = data.get('trades', [])
    inserted = 0
    buy_count = 0

    for t in trades:
        tx = (t.get('Transaction') or '').lower()
        if 'purchase' not in tx and 'buy' not in tx:
            continue
        buy_count += 1
        ticker = (t.get('Ticker') or '').strip().upper()
        if not ticker or len(ticker) > 5:
            continue

        date = t.get('TransactionDate') or t.get('ReportDate') or t.get('Date') or ''
        if not date:
            continue

        signal = {
            'ticker': ticker,
            'signal_date': date,
            'source': 'congress',
            'representative': t.get('Representative', ''),
            'party': t.get('Party', ''),
            'chamber': t.get('Chamber', '') or t.get('House', ''),
            'trade_size_range': t.get('Range', ''),
            'trade_size_points': range_to_base_points(t.get('Range', '')),
            'insider_name': None,
            'sector': get_sector(ticker),
            'market_cap_bucket': get_market_cap_bucket(ticker),
            'disclosure_delay': t.get('DisclosureDelay'),
        }
        if insert_signal(conn, signal):
            inserted += 1

    log.info(f"Congress feed: {len(trades)} trades loaded, {buy_count} purchases, {inserted} new inserted")
    return inserted


def ingest_edgar_feed(conn: sqlite3.Connection, feed_path: Path = None) -> int:
    """Load edgar_feed.json and insert insider signals using enriched XML fields.

    The daily feed (from fetch_data.py) parses Form 4 XML and provides:
    - ticker: extracted directly from XML (more reliable than company name matching)
    - direction: 'buy', 'sell', 'mixed', or 'other'
    - title/roles: insider's position (CEO, CFO, Director, etc.)
    - buy_value/sell_value: dollar amounts of transactions
    - is_10b5_1: whether a 10b5-1 plan was disclosed

    We only ingest purchases (direction='buy') since those are the actionable signals.
    Falls back to match_edgar_ticker() if the XML ticker field is missing.

    Returns count inserted.
    """
    path = feed_path or EDGAR_FEED
    if not path.exists():
        log.warning(f"EDGAR feed not found: {path}")
        return 0

    data = load_json(path)
    filings = data.get('filings', [])
    inserted = 0
    skipped_non_buy = 0

    for f in filings:
        # Use XML-extracted ticker first, fall back to company name matching
        ticker = f.get('ticker', '').strip().upper()
        if not ticker:
            ticker = match_edgar_ticker(f.get('company', ''))
        if not ticker:
            continue

        date = f.get('date', '')
        if not date:
            continue

        # Only ingest purchases — sells are noise for our signal engine
        direction = (f.get('direction') or '').lower()
        if direction and direction != 'buy':
            skipped_non_buy += 1
            continue

        # Build insider role from title + roles fields
        title = f.get('title', '').strip()
        roles = f.get('roles', [])
        if title:
            insider_role = title
        elif roles:
            insider_role = ', '.join(roles)
        else:
            insider_role = ''

        # Map buy_value to approximate trade_size_points for position sizing
        buy_value = f.get('buy_value', 0) or 0
        trade_size_points = _buy_value_to_points(buy_value)

        signal = {
            'ticker': ticker,
            'signal_date': date,
            'source': 'edgar',
            'insider_name': f.get('insider', ''),
            'insider_role': insider_role,
            'transaction_type': 'Purchase',
            'representative': None,
            'trade_size_points': trade_size_points if trade_size_points > 0 else None,
            'sector': get_sector(ticker),
            'market_cap_bucket': get_market_cap_bucket(ticker),
            'accession_number': f.get('accession', ''),
        }
        if insert_signal(conn, signal):
            inserted += 1

    log.info(f"EDGAR feed: {len(filings)} filings → {skipped_non_buy} non-buy skipped → {inserted} inserted")
    return inserted


def ingest_13f_feed(conn: sqlite3.Connection, feed_path: Path = None) -> int:
    """Load 13f_feed.json and insert institutional signals.

    Source = '13f', role = 'Institutional'. Only ingests new/increased positions
    (bullish signals). Convergence detection automatically picks up the third source.

    Returns count inserted.
    """
    path = feed_path or THIRTEENF_FEED
    if not path.exists():
        log.info("13F feed not found — skipping (run scripts/fetch_13f.py first)")
        return 0

    data = load_json(path)
    filings = data.get('filings', [])
    inserted = 0

    for f in filings:
        ticker = (f.get('ticker') or '').strip().upper()
        if not ticker or len(ticker) > 5:
            continue

        date = f.get('filed_date', '')
        if not date:
            continue

        action = f.get('action', '')
        filer = f.get('filer', '')
        value = f.get('value', 0) or 0

        # Map value to score points
        trade_size_points = _buy_value_to_points(value / 1000)  # 13F values are in dollars

        signal = {
            'ticker': ticker,
            'signal_date': date,
            'source': '13f',
            'insider_name': filer,
            'insider_role': 'Institutional',
            'transaction_type': 'New Position' if action == 'new_position' else 'Increased',
            'representative': None,
            'trade_size_points': trade_size_points if trade_size_points > 0 else None,
            'sector': get_sector(ticker),
            'market_cap_bucket': get_market_cap_bucket(ticker),
        }
        if insert_signal(conn, signal):
            inserted += 1

    log.info(f"13F feed: {len(filings)} changes → {inserted} inserted")
    return inserted


def _buy_value_to_points(value: float) -> int:
    """Map a dollar buy value to base score points (mirrors range_to_base_points)."""
    if value >= 1_000_000: return 15
    if value >= 500_000:   return 12
    if value >= 250_000:   return 10
    if value >= 100_000:   return 8
    if value >= 50_000:    return 6
    if value >= 15_000:    return 5
    if value > 0:          return 3
    return 0


def update_aggregate_features(conn: sqlite3.Connection, since_date: str = None) -> int:
    """Recompute aggregate features with multi-tier convergence detection.

    Convergence Tiers:
      0 — No convergence (single source only)
      1 — Ticker convergence: same ticker in 2+ hubs within window
          (congress 60d lookback, edgar 30d lookback)
      2 — Sector convergence: 3+ signals from 2+ sources in same sector, 30d
      3 — Thematic: sector convergence + active legislation (reserved for frontend)

    Args:
        since_date: If set, only reprocess signals on or after this date.
                    Use in --daily for incremental processing; omit in --analyze for full recompute.
    """
    # Congress gets 60d window (STOCK Act allows 45d disclosure delay)
    CONGRESS_WINDOW = 60
    EDGAR_WINDOW = 30

    if since_date:
        all_signals = conn.execute(
            "SELECT id, ticker, signal_date, source, sector FROM signals "
            "WHERE signal_date >= ?", (since_date,)
        ).fetchall()
    else:
        all_signals = conn.execute(
            "SELECT id, ticker, signal_date, source, sector FROM signals"
        ).fetchall()
    updated = 0

    for sig in all_signals:
        sig_id = sig['id']
        ticker = sig['ticker']
        signal_date = sig['signal_date']
        sector = sig['sector']
        source = sig['source']
        dt = datetime.strptime(signal_date, '%Y-%m-%d')

        # Cluster counts (unchanged — 7d and 30d same-ticker)
        d7 = (dt - timedelta(days=7)).strftime('%Y-%m-%d')
        d30 = (dt - timedelta(days=30)).strftime('%Y-%m-%d')

        count_7d = conn.execute(
            "SELECT COUNT(*) as cnt FROM signals WHERE ticker=? AND signal_date BETWEEN ? AND ? AND id != ?",
            (ticker, d7, signal_date, sig_id)
        ).fetchone()['cnt']
        count_30d = conn.execute(
            "SELECT COUNT(*) as cnt FROM signals WHERE ticker=? AND signal_date BETWEEN ? AND ? AND id != ?",
            (ticker, d30, signal_date, sig_id)
        ).fetchone()['cnt']

        # ── Tier 1: Ticker convergence ──
        # Use 60d lookback for congress, 30d for edgar
        d60 = (dt - timedelta(days=CONGRESS_WINDOW)).strftime('%Y-%m-%d')
        d30e = (dt - timedelta(days=EDGAR_WINDOW)).strftime('%Y-%m-%d')

        # Find distinct sources for this exact ticker within the wider window
        sources_row = conn.execute(
            "SELECT DISTINCT source FROM signals WHERE ticker=? AND signal_date BETWEEN ? AND ?",
            (ticker, d60, signal_date)
        ).fetchall()
        sources = set(r['source'] for r in sources_row)
        has_convergence = 1 if len(sources) > 1 else 0
        convergence_sources = '+'.join(sorted(sources)) if has_convergence else None
        convergence_tier = 1 if has_convergence else 0
        convergence_sector_val = None
        convergence_tickers_val = None

        # ── Tier 2: Sector convergence ──
        # 3+ signals from 2+ sources in the same sector within 30d
        if sector and sector.strip():
            sector_sources = conn.execute(
                "SELECT DISTINCT source FROM signals WHERE sector=? AND signal_date BETWEEN ? AND ? AND id != ?",
                (sector, d30, signal_date, sig_id)
            ).fetchall()
            sector_source_set = set(r['source'] for r in sector_sources)
            # Add current signal's source
            sector_source_set.add(source)

            if len(sector_source_set) >= 2:
                sector_count = conn.execute(
                    "SELECT COUNT(*) as cnt FROM signals WHERE sector=? AND signal_date BETWEEN ? AND ?",
                    (sector, d30, signal_date)
                ).fetchone()['cnt']

                if sector_count >= 3:
                    convergence_tier = max(convergence_tier, 2)
                    has_convergence = 1
                    convergence_sector_val = sector
                    # Get the distinct tickers in this sector cluster
                    sector_tickers = conn.execute(
                        "SELECT DISTINCT ticker FROM signals WHERE sector=? AND signal_date BETWEEN ? AND ?",
                        (sector, d30, signal_date)
                    ).fetchall()
                    convergence_tickers_val = ','.join(sorted(r['ticker'] for r in sector_tickers))
                    convergence_sources = '+'.join(sorted(sector_source_set))

        # Cluster velocity: avg days between consecutive same-ticker signals
        ticker_dates = conn.execute(
            "SELECT DISTINCT signal_date FROM signals WHERE ticker=? AND signal_date <= ? ORDER BY signal_date",
            (ticker, signal_date)
        ).fetchall()
        cluster_velocity = _compute_cluster_velocity([r['signal_date'] for r in ticker_dates])

        conn.execute(
            """UPDATE signals SET
                same_ticker_signals_7d=?, same_ticker_signals_30d=?,
                has_convergence=?, convergence_sources=?,
                convergence_tier=?, convergence_sector=?, convergence_tickers=?,
                cluster_velocity=?
            WHERE id=?""",
            (count_7d, count_30d, has_convergence, convergence_sources,
             convergence_tier, convergence_sector_val, convergence_tickers_val,
             cluster_velocity, sig_id)
        )
        updated += 1

    conn.commit()
    return updated


def _compute_cluster_velocity(dates: list) -> str:
    """Compute average days between consecutive signals.
    burst (<3d), fast (3-7d), moderate (7-14d), slow (>14d), n/a (single)."""
    if len(dates) < 2:
        return 'n/a'
    sorted_dates = sorted(datetime.strptime(d[:10], '%Y-%m-%d') for d in dates)
    gaps = [(sorted_dates[i+1] - sorted_dates[i]).days for i in range(len(sorted_dates)-1)]
    avg_gap = sum(gaps) / len(gaps)
    if avg_gap < 3:
        return 'burst'
    if avg_gap < 7:
        return 'fast'
    if avg_gap < 14:
        return 'moderate'
    return 'slow'


# ── Entry Price Backfill ──────────────────────────────────────────────────────

def backfill_entry_prices(conn: sqlite3.Connection) -> int:
    """Backfill price_at_signal for signals missing it, using price_history cache.

    Looks up the close price on signal_date (or nearest prior trading day).
    Returns count of signals updated.
    """
    rows = conn.execute("""
        SELECT id, ticker, signal_date FROM signals
        WHERE (price_at_signal IS NULL OR price_at_signal = 0)
    """).fetchall()

    if not rows:
        return 0

    _cache = {}
    updated = 0
    for r in rows:
        ticker = r['ticker']
        sig_date = r['signal_date']

        if ticker not in _cache:
            path = PRICE_HISTORY_DIR / f"{ticker}.json"
            if path.exists():
                try:
                    with open(path) as f:
                        _cache[ticker] = json.load(f)
                except Exception:
                    _cache[ticker] = {}
            else:
                _cache[ticker] = {}

        prices = _cache[ticker]
        if not prices:
            continue

        # Try exact date first, then look back up to 5 trading days
        close = None
        if sig_date in prices and isinstance(prices[sig_date], dict):
            close = prices[sig_date].get('c')
        else:
            from datetime import datetime as dt_cls, timedelta
            d = dt_cls.strptime(sig_date, '%Y-%m-%d')
            for offset in range(1, 6):
                prev = (d - timedelta(days=offset)).strftime('%Y-%m-%d')
                if prev in prices and isinstance(prices[prev], dict):
                    close = prices[prev].get('c')
                    if close:
                        break

        if close and close > 0:
            conn.execute("UPDATE signals SET price_at_signal = ? WHERE id = ?",
                         (close, r['id']))
            updated += 1

    if updated:
        conn.commit()
        log.info(f"Backfilled price_at_signal for {updated}/{len(rows)} signals")
    return updated


# ── Liquidity / Transaction Cost Enrichment ──────────────────────────────────

def enrich_liquidity_features(conn: sqlite3.Connection) -> int:
    """Add avg_daily_volume and estimated_spread to signals from price cache.

    Spread estimate by market cap proxy (price × ADV × 252):
      Large cap (>$10B): 0.05%
      Mid cap ($1-10B):  0.20%
      Small cap (<$1B):  0.50%

    Also computes net_expected_return = car_30d − 2×spread (round trip).
    Returns count of signals updated.
    """
    import pandas as pd

    signals = pd.read_sql("""
        SELECT id, ticker, signal_date
        FROM signals WHERE avg_daily_volume IS NULL
    """, conn)

    if signals.empty:
        return 0

    updated = 0
    for _, row in signals.iterrows():
        cache_path = PRICE_HISTORY_DIR / f"{row['ticker']}.json"
        if not cache_path.exists():
            continue

        try:
            with open(cache_path) as f:
                prices = json.load(f)
        except Exception:
            continue

        # Get 30d avg volume around signal date
        signal_dt = pd.to_datetime(row['signal_date'])
        window_start = (signal_dt - pd.Timedelta(days=45)).strftime('%Y-%m-%d')
        window_end = signal_dt.strftime('%Y-%m-%d')

        vols = [
            p.get('v', p.get('volume', 0))
            for date_str, p in prices.items()
            if isinstance(p, dict)
            and window_start <= date_str <= window_end
            and (p.get('v') or p.get('volume'))
        ]

        if not vols:
            continue

        adv = sum(vols) / len(vols)

        # Estimate spread by market cap proxy
        sd_data = prices.get(row['signal_date'])
        price = (sd_data.get('c', sd_data.get('close', 0)) if isinstance(sd_data, dict) else 0)
        if not price:
            # Try nearest date
            for d in sorted(prices.keys(), reverse=True):
                if d <= window_end and isinstance(prices[d], dict):
                    price = prices[d].get('c', prices[d].get('close', 0))
                    if price:
                        break

        market_cap_proxy = price * adv * 252

        if market_cap_proxy > 10_000_000_000:
            spread_est = 0.0005   # large cap: 0.05%
        elif market_cap_proxy > 1_000_000_000:
            spread_est = 0.0020   # mid cap: 0.20%
        else:
            spread_est = 0.0050   # small cap: 0.50%

        flag = 'HIGH_COST' if spread_est >= 0.005 else 'OK'

        conn.execute("""
            UPDATE signals
            SET avg_daily_volume = ?, estimated_spread = ?, liquidity_flag = ?
            WHERE id = ?
        """, (adv, spread_est, flag, row['id']))
        updated += 1

    # Compute net expected return (round trip: buy spread + sell spread)
    conn.execute("""
        UPDATE signals
        SET net_expected_return = car_30d - (2 * estimated_spread)
        WHERE car_30d IS NOT NULL AND estimated_spread IS NOT NULL
        AND net_expected_return IS NULL
    """)
    conn.commit()
    log.info(f"Enriched liquidity for {updated} signals")
    return updated


# ── Person Track Record ──────────────────────────────────────────────────────

def update_person_track_records(conn: sqlite3.Connection) -> int:
    """Compute per-person historical performance and relative position size.
    For each signal, looks at ALL prior signals by the same person and computes
    their hit rate, avg CAR, and how this trade's size compares to their norm."""
    updated = 0

    # Congressional: track by representative
    congress_rows = conn.execute(
        "SELECT id, representative, signal_date, trade_size_points "
        "FROM signals WHERE source='congress' AND representative != '' "
        "ORDER BY signal_date ASC"
    ).fetchall()

    # Build running history per representative
    rep_history = defaultdict(list)  # rep -> [(date, car_30d, car_90d, size_pts)]
    for row in congress_rows:
        rep = row['representative']
        sig_date = row['signal_date']
        sig_id = row['id']
        size_pts = row['trade_size_points'] or 3

        # Look up prior trades' outcomes for this person
        # POINT-IN-TIME: only use outcomes knowable at sig_date
        # (signal_date + 45 cal days < current date → outcome was known)
        prior = rep_history.get(rep, [])
        cutoff_30 = (datetime.strptime(sig_date, '%Y-%m-%d') - timedelta(days=45)).strftime('%Y-%m-%d')
        cutoff_90 = (datetime.strptime(sig_date, '%Y-%m-%d') - timedelta(days=135)).strftime('%Y-%m-%d')
        prior_with_outcomes_30 = [p for p in prior if p['car_30d'] is not None and p['date'] <= cutoff_30]
        prior_with_outcomes_90 = [p for p in prior if p['car_90d'] is not None and p['date'] <= cutoff_90]

        trade_count = len(prior)
        hit_rate_30 = None
        avg_car_30 = None
        hit_rate_90 = None
        avg_car_90 = None
        relative_size = None

        if prior_with_outcomes_30:
            cars_30 = [p['car_30d'] for p in prior_with_outcomes_30]
            hit_rate_30 = round(sum(1 for c in cars_30 if c > 0) / len(cars_30), 4)
            avg_car_30 = round(sum(cars_30) / len(cars_30), 6)

        if prior_with_outcomes_90:
            cars_90 = [p['car_90d'] for p in prior_with_outcomes_90]
            hit_rate_90 = round(sum(1 for c in cars_90 if c > 0) / len(cars_90), 4)
            avg_car_90 = round(sum(cars_90) / len(cars_90), 6)

        if prior:
            prior_sizes = [p['size_pts'] for p in prior if p['size_pts']]
            if prior_sizes:
                median_size = sorted(prior_sizes)[len(prior_sizes) // 2]
                if median_size > 0:
                    relative_size = round(size_pts / median_size, 2)

        conn.execute(
            """UPDATE signals SET
                person_trade_count=?, person_hit_rate_30d=?, person_avg_car_30d=?,
                person_hit_rate_90d=?, person_avg_car_90d=?, relative_position_size=?
            WHERE id=?""",
            (trade_count, hit_rate_30, avg_car_30, hit_rate_90, avg_car_90, relative_size, sig_id)
        )
        updated += 1

        # Fetch this signal's outcomes for future lookups
        outcome = conn.execute(
            "SELECT car_30d, car_90d FROM signals WHERE id=?", (sig_id,)
        ).fetchone()
        rep_history[rep].append({
            'date': sig_date,
            'car_30d': outcome['car_30d'] if outcome else None,
            'car_90d': outcome['car_90d'] if outcome else None,
            'size_pts': size_pts,
        })

    # EDGAR: track by insider_name (with relative position size)
    edgar_rows = conn.execute(
        "SELECT id, insider_name, signal_date, trade_size_points "
        "FROM signals WHERE source='edgar' AND insider_name != '' "
        "ORDER BY signal_date ASC"
    ).fetchall()

    insider_history = defaultdict(list)
    for row in edgar_rows:
        insider = row['insider_name']
        sig_id = row['id']
        size_pts = row['trade_size_points'] or 0

        prior = insider_history.get(insider, [])
        # POINT-IN-TIME: only use outcomes knowable at signal date
        sig_date_str = row['signal_date']
        cutoff_30 = (datetime.strptime(sig_date_str, '%Y-%m-%d') - timedelta(days=45)).strftime('%Y-%m-%d')
        cutoff_90 = (datetime.strptime(sig_date_str, '%Y-%m-%d') - timedelta(days=135)).strftime('%Y-%m-%d')
        prior_with_outcomes_30 = [p for p in prior if p['car_30d'] is not None and p['date'] <= cutoff_30]
        prior_with_outcomes_90 = [p for p in prior if p['car_90d'] is not None and p['date'] <= cutoff_90]

        trade_count = len(prior)
        hit_rate_30 = None
        avg_car_30 = None
        hit_rate_90 = None
        avg_car_90 = None
        relative_size = None

        if prior_with_outcomes_30:
            cars_30 = [p['car_30d'] for p in prior_with_outcomes_30]
            hit_rate_30 = round(sum(1 for c in cars_30 if c > 0) / len(cars_30), 4)
            avg_car_30 = round(sum(cars_30) / len(cars_30), 6)

        if prior_with_outcomes_90:
            cars_90 = [p['car_90d'] for p in prior_with_outcomes_90]
            hit_rate_90 = round(sum(1 for c in cars_90 if c > 0) / len(cars_90), 4)
            avg_car_90 = round(sum(cars_90) / len(cars_90), 6)

        if prior and size_pts > 0:
            prior_sizes = [p['size_pts'] for p in prior if p['size_pts']]
            if prior_sizes:
                median_size = sorted(prior_sizes)[len(prior_sizes) // 2]
                if median_size > 0:
                    relative_size = round(size_pts / median_size, 2)

        conn.execute(
            """UPDATE signals SET
                person_trade_count=?, person_hit_rate_30d=?, person_avg_car_30d=?,
                person_hit_rate_90d=?, person_avg_car_90d=?, relative_position_size=?
            WHERE id=?""",
            (trade_count, hit_rate_30, avg_car_30, hit_rate_90, avg_car_90, relative_size, sig_id)
        )
        updated += 1

        outcome = conn.execute(
            "SELECT car_30d, car_90d FROM signals WHERE id=?", (sig_id,)
        ).fetchone()
        insider_history[insider].append({
            'date': row['signal_date'],
            'car_30d': outcome['car_30d'] if outcome else None,
            'car_90d': outcome['car_90d'] if outcome else None,
            'size_pts': size_pts,
        })

    conn.commit()
    return updated


# ── Feature Enrichment ───────────────────────────────────────────────────────

def enrich_signal_features(conn: sqlite3.Connection) -> int:
    """Populate price_proximity_52wk, market_cap_bucket, trade_pattern, and
    backfill insider_role for signals missing these features.

    This runs after price collection and uses existing price history files
    (data/price_history/*.json) plus the sector map. It does NOT make any
    external API calls — it only uses data already on disk.

    Returns count of signals updated.
    """
    updated = 0

    # ── 1. Price proximity to 52-week range ──
    # Find signals with price_at_signal but missing price_proximity_52wk
    needs_proximity = conn.execute(
        "SELECT DISTINCT ticker FROM signals "
        "WHERE price_at_signal IS NOT NULL AND price_proximity_52wk IS NULL"
    ).fetchall()

    for row in needs_proximity:
        ticker = row['ticker']
        price_index = load_price_index(ticker)
        if not price_index or len(price_index) < 30:
            continue

        # Compute 52-week hi/lo from price history
        all_prices = sorted(price_index.items())
        # For each signal, compute proximity using trailing 252 trading days
        signals = conn.execute(
            "SELECT id, signal_date, price_at_signal FROM signals "
            "WHERE ticker=? AND price_at_signal IS NOT NULL AND price_proximity_52wk IS NULL",
            (ticker,)
        ).fetchall()

        for sig in signals:
            sig_date = sig['signal_date']
            price = sig['price_at_signal']

            # Get prices in the 252 trading days before the signal
            trailing = [p for d, p in all_prices if d <= sig_date]
            trailing = trailing[-252:]  # last ~1 year of trading days

            if len(trailing) < 20:
                continue

            high_52wk = max(trailing)
            low_52wk = min(trailing)
            proximity = compute_52wk_proximity(price, high_52wk, low_52wk)

            if proximity is not None:
                conn.execute(
                    "UPDATE signals SET price_proximity_52wk=? WHERE id=?",
                    (proximity, sig['id'])
                )
                updated += 1

    # ── 1b. Momentum features (from price history — no API calls) ──
    needs_momentum = conn.execute(
        "SELECT DISTINCT ticker FROM signals "
        "WHERE price_at_signal IS NOT NULL AND momentum_1m IS NULL"
    ).fetchall()

    for row in needs_momentum:
        ticker = row['ticker']
        price_index = load_price_index(ticker)
        if not price_index or len(price_index) < 30:
            continue

        signals = conn.execute(
            "SELECT id, signal_date FROM signals "
            "WHERE ticker=? AND price_at_signal IS NOT NULL AND momentum_1m IS NULL",
            (ticker,)
        ).fetchall()

        for sig in signals:
            mom = _compute_momentum_features(price_index, sig['signal_date'])
            updates = {k: v for k, v in mom.items() if v is not None}
            if updates:
                set_clause = ', '.join(f'{k}=?' for k in updates.keys())
                conn.execute(
                    f"UPDATE signals SET {set_clause} WHERE id=?",
                    list(updates.values()) + [sig['id']]
                )
                updated += 1

    # ── 1c. Volume spike (from OHLCV — no API calls) ──
    needs_volume = conn.execute(
        "SELECT DISTINCT ticker FROM signals "
        "WHERE price_at_signal IS NOT NULL AND volume_spike IS NULL"
    ).fetchall()

    for row in needs_volume:
        ticker = row['ticker']
        cache_path = PRICE_HISTORY_DIR / f"{ticker}.json"
        if not cache_path.exists():
            continue
        candles = load_json(cache_path)
        if not candles:
            continue

        signals = conn.execute(
            "SELECT id, signal_date FROM signals "
            "WHERE ticker=? AND price_at_signal IS NOT NULL AND volume_spike IS NULL",
            (ticker,)
        ).fetchall()

        for sig in signals:
            spike = _compute_volume_spike(candles, sig['signal_date'])
            if spike is not None:
                conn.execute(
                    "UPDATE signals SET volume_spike=? WHERE id=?",
                    (spike, sig['id'])
                )
                updated += 1

    # ── 1d. Insider buy ratio 90d (from DB — no API calls) ──
    needs_ratio = conn.execute(
        "SELECT id, ticker, signal_date FROM signals WHERE insider_buy_ratio_90d IS NULL"
    ).fetchall()

    for sig in needs_ratio:
        ratio = _compute_insider_buy_ratio(conn, sig['ticker'], sig['signal_date'], sig['id'])
        conn.execute(
            "UPDATE signals SET insider_buy_ratio_90d=? WHERE id=?",
            (ratio, sig['id'])
        )
        updated += 1

    # ── 1e. Sector avg CAR (from DB — no API calls) ──
    # Point-in-time: compute per-signal using only outcomes from prior signals
    signals_needing_sector_car = conn.execute(
        "SELECT id, sector, signal_date FROM signals "
        "WHERE sector_avg_car IS NULL AND sector IS NOT NULL "
        "ORDER BY signal_date ASC"
    ).fetchall()

    sector_car_updated = 0
    for row in signals_needing_sector_car:
        avg_car = _compute_sector_avg_car(conn, row['sector'], before_date=row['signal_date'])
        if avg_car is not None:
            conn.execute(
                "UPDATE signals SET sector_avg_car=? WHERE id=?",
                (avg_car, row['id'])
            )
            sector_car_updated += 1
    if sector_car_updated:
        log.info(f"Backfilled sector_avg_car for {sector_car_updated} signals (point-in-time)")
    updated += sector_car_updated

    # ── 1f. VIX regime interaction (from DB — no API calls) ──
    needs_vix_int = conn.execute(
        "SELECT id, vix_at_signal, has_convergence FROM signals "
        "WHERE vix_regime_interaction IS NULL AND vix_at_signal IS NOT NULL"
    ).fetchall()

    for sig in needs_vix_int:
        interaction = _compute_vix_regime_interaction(sig['vix_at_signal'], sig['has_convergence'])
        if interaction is not None:
            conn.execute(
                "UPDATE signals SET vix_regime_interaction=? WHERE id=?",
                (interaction, sig['id'])
            )
            updated += 1

    # ── 2. Trade pattern classification (Cohen et al. 2012) ──
    # For EDGAR signals missing trade_pattern, classify as opportunistic vs routine
    edgar_insiders = conn.execute(
        "SELECT DISTINCT insider_name FROM signals "
        "WHERE source='edgar' AND insider_name != '' AND trade_pattern IS NULL"
    ).fetchall()

    for row in edgar_insiders:
        insider = row['insider_name']
        # Get all historical trades by this insider
        history = conn.execute(
            "SELECT id, signal_date as date FROM signals "
            "WHERE source='edgar' AND insider_name=? ORDER BY signal_date ASC",
            (insider,)
        ).fetchall()

        pattern = classify_insider_pattern([dict(h) for h in history])

        # Update all signals for this insider that lack trade_pattern
        conn.execute(
            "UPDATE signals SET trade_pattern=? "
            "WHERE source='edgar' AND insider_name=? AND trade_pattern IS NULL",
            (pattern, insider)
        )
        updated += len(history)

    # ── 3. Backfill insider_role from EDGAR feed + cross-signal propagation ──
    # Historical signals may have insider_name but missing insider_role
    missing_role = conn.execute(
        "SELECT COUNT(*) as cnt FROM signals "
        "WHERE source='edgar' AND insider_name != '' "
        "AND (insider_role IS NULL OR insider_role = '')"
    ).fetchone()['cnt']

    if missing_role > 0:
        role_filled = 0

        # 3a. Cross-signal propagation: if any signal for this insider_name has
        # a role, apply it to all their signals missing a role
        known_roles = conn.execute(
            "SELECT DISTINCT insider_name, insider_role FROM signals "
            "WHERE source='edgar' AND insider_name != '' "
            "AND insider_role IS NOT NULL AND insider_role != ''"
        ).fetchall()
        for row in known_roles:
            cnt = conn.execute(
                "UPDATE signals SET insider_role=? "
                "WHERE source='edgar' AND insider_name=? "
                "AND (insider_role IS NULL OR insider_role = '')",
                (row['insider_role'], row['insider_name'])
            ).rowcount
            role_filled += cnt
        if role_filled:
            log.debug(f"Backfilled insider_role via cross-signal propagation: {role_filled} signals")

        # 3b. Try to backfill remaining from edgar_feed.json
        from backtest.shared import EDGAR_FEED
        if EDGAR_FEED.exists():
            edgar_data = load_json(EDGAR_FEED)
            role_map = {}
            for f in edgar_data.get('filings', []):
                insider = f.get('insider', '').strip()
                if not insider:
                    continue
                title = f.get('title', '').strip()
                roles = f.get('roles', [])
                if title:
                    role_map[insider] = title
                elif roles:
                    role_map[insider] = ', '.join(roles)

            feed_filled = 0
            if role_map:
                for insider, role in role_map.items():
                    normalized = _normalize_role(role)
                    cnt = conn.execute(
                        "UPDATE signals SET insider_role=? "
                        "WHERE source='edgar' AND insider_name=? "
                        "AND (insider_role IS NULL OR insider_role = '')",
                        (normalized, insider)
                    ).rowcount
                    feed_filled += cnt
                log.debug(f"Backfilled insider_role from feed: {len(role_map)} insiders, {feed_filled} signals")

        # 3c. Infer role for corporate/institutional names still missing
        # These are PE/VC firms, holding companies, etc. — almost always 10% Owners
        _CORP_PATTERNS = ('inc', 'corp', 'llc', 'l.p.', ' lp', 'ltd', ' ag ',
                          ' se ', 'holdings', 'partners', 'associates',
                          'management', 'ventures', 'capital', 'fund',
                          'trust', 'group', ' co.', ' co ')
        still_missing = conn.execute(
            "SELECT DISTINCT insider_name FROM signals "
            "WHERE source='edgar' AND insider_name != '' "
            "AND (insider_role IS NULL OR insider_role = '')"
        ).fetchall()
        corp_filled = 0
        for row in still_missing:
            name = row['insider_name']
            if any(p in name.lower() for p in _CORP_PATTERNS):
                cnt = conn.execute(
                    "UPDATE signals SET insider_role='10% Owner' "
                    "WHERE source='edgar' AND insider_name=? "
                    "AND (insider_role IS NULL OR insider_role = '')",
                    (name,)
                ).rowcount
                corp_filled += cnt
        if corp_filled:
            log.debug(f"Backfilled insider_role for {corp_filled} corporate/institutional names")
        role_filled += corp_filled

        # 3d. Assign 'Other' to any remaining individuals without a role
        other_filled = conn.execute(
            "UPDATE signals SET insider_role='Other' "
            "WHERE source='edgar' AND insider_name != '' "
            "AND (insider_role IS NULL OR insider_role = '')"
        ).rowcount
        if other_filled:
            log.debug(f"Assigned 'Other' role to {other_filled} remaining signals")
        role_filled += other_filled

        updated += role_filled

    # ── 4. Market cap bucket ──
    needs_cap = conn.execute(
        "SELECT DISTINCT ticker FROM signals WHERE market_cap_bucket IS NULL"
    ).fetchall()

    # If cache is empty and many signals need caps, rebuild from FMP
    if needs_cap and not get_market_cap('AAPL'):
        import os
        fmp_key = os.environ.get('FMP_API_KEY', 'UefVEEvF1XXtpgWcsidPCGxcDJ6N0kXv')
        if fmp_key:
            from backtest.sector_map import build_sector_map
            tickers_for_map = [r['ticker'] for r in needs_cap]
            log.info(f"Market cap cache empty — rebuilding from FMP for {len(tickers_for_map)} tickers...")
            build_sector_map(api_key=fmp_key, tickers=tickers_for_map)

    cap_updated = 0
    for row in needs_cap:
        ticker = row['ticker']
        bucket = get_market_cap_bucket(ticker)
        if bucket:
            conn.execute(
                "UPDATE signals SET market_cap_bucket=? WHERE ticker=? AND market_cap_bucket IS NULL",
                (bucket, ticker)
            )
            cap_updated += conn.execute("SELECT changes()").fetchone()[0]
    if cap_updated:
        log.info(f"Backfilled market_cap_bucket for {cap_updated} signals")
        updated += cap_updated

    # ── 5. Days to catalyst (upcoming bill/legislation) ──
    needs_catalyst = conn.execute(
        "SELECT id, ticker, signal_date FROM signals WHERE days_to_catalyst IS NULL"
    ).fetchall()
    if needs_catalyst:
        catalyst_updated = _enrich_catalyst_proximity(conn, needs_catalyst)
        updated += catalyst_updated

    # ── 6. Days to earnings ──
    needs_earnings = conn.execute(
        "SELECT DISTINCT ticker FROM signals WHERE days_to_earnings IS NULL"
    ).fetchall()
    if needs_earnings:
        earnings_updated = _enrich_earnings_proximity(conn, [r['ticker'] for r in needs_earnings])
        updated += earnings_updated

    # ── 7. Sector momentum — avg momentum_1m for signals in same sector, trailing 90d ──
    for col in ['sector_momentum', 'days_since_last_buy']:
        try:
            conn.execute(f"ALTER TABLE signals ADD COLUMN {col} REAL")
        except Exception:
            pass

    sector_mom = conn.execute("""
        UPDATE signals SET sector_momentum = (
            SELECT AVG(s2.momentum_1m) FROM signals s2
            WHERE s2.sector = signals.sector
              AND s2.momentum_1m IS NOT NULL
              AND s2.signal_date BETWEEN date(signals.signal_date, '-90 days') AND signals.signal_date
              AND s2.id != signals.id
        ) WHERE sector_momentum IS NULL AND sector IS NOT NULL AND sector != ''
    """).rowcount
    updated += sector_mom

    # ── 8. Days since last buy — for repeat buyer detection ──
    # For each signal, find the most recent prior signal from the same person
    dslb = conn.execute("""
        UPDATE signals SET days_since_last_buy = (
            SELECT MIN(julianday(signals.signal_date) - julianday(s2.signal_date))
            FROM signals s2
            WHERE s2.signal_date < signals.signal_date
              AND s2.id != signals.id
              AND (
                (signals.insider_name IS NOT NULL AND signals.insider_name != '' AND s2.insider_name = signals.insider_name)
                OR (signals.representative IS NOT NULL AND signals.representative != '' AND s2.representative = signals.representative)
              )
        ) WHERE days_since_last_buy IS NULL
    """).rowcount
    updated += dslb

    # ── 9. Volume dry-up (from existing volume_spike) ──
    # Low volume on insider buy often signals accumulation before a move
    vol_dry = conn.execute("""
        UPDATE signals SET volume_dry_up = CASE
            WHEN volume_spike IS NOT NULL AND volume_spike < 0.4 THEN 1
            ELSE 0
        END
        WHERE volume_dry_up IS NULL AND volume_spike IS NOT NULL
    """).rowcount
    updated += vol_dry
    if vol_dry:
        log.info(f"Computed volume_dry_up for {vol_dry} signals")

    # ── 10. Analyst features (from data/analyst_data.json) ──
    analyst_path = DATA_DIR / "analyst_data.json"
    if analyst_path.exists():
        try:
            analyst_raw = load_json(analyst_path)
            analyst_tickers = analyst_raw.get('tickers', {})
            if analyst_tickers:
                needs_analyst = conn.execute(
                    "SELECT id, ticker, source FROM signals WHERE analyst_revision_30d IS NULL"
                ).fetchall()
                analyst_updated = 0
                for sig in needs_analyst:
                    ticker = sig['ticker']
                    a = analyst_tickers.get(ticker)
                    if not a:
                        # No analyst data — fill with 0/null
                        conn.execute(
                            "UPDATE signals SET analyst_revision_30d=0, analyst_consensus=NULL, "
                            "analyst_insider_confluence=0 WHERE id=?",
                            (sig['id'],)
                        )
                    else:
                        rev = a.get('revision_momentum', 0) or 0
                        consensus = a.get('analyst_consensus')
                        # Confluence: positive revision + insider buy
                        is_insider_buy = sig['source'] == 'edgar'
                        confluence = 1 if (rev > 0 and is_insider_buy) else 0
                        conn.execute(
                            "UPDATE signals SET analyst_revision_30d=?, analyst_consensus=?, "
                            "analyst_insider_confluence=? WHERE id=?",
                            (rev, consensus, confluence, sig['id'])
                        )
                    analyst_updated += 1
                updated += analyst_updated
                if analyst_updated:
                    log.info(f"Enriched analyst features for {analyst_updated} signals")
        except Exception as e:
            log.warning(f"Analyst enrichment failed: {e}")

    # ── 11. Committee overlap (from data/committee_data.json) ──
    # If a congress member trades a stock in a sector their committee oversees,
    # that's a high-conviction information-edge signal.
    committee_path = DATA_DIR / "committee_data.json"
    if committee_path.exists():
        try:
            comm_raw = load_json(committee_path)
            comm_members = comm_raw.get('members', {})
            if comm_members:
                needs_committee = conn.execute(
                    "SELECT id, representative, sector FROM signals "
                    "WHERE committee_overlap IS NULL AND source = 'congress' "
                    "AND representative IS NOT NULL AND sector IS NOT NULL"
                ).fetchall()
                comm_updated = 0
                for sig in needs_committee:
                    rep = sig['representative'].strip().lower()
                    sig_sector = sig['sector']
                    overlap = 0
                    # Try matching representative name to committee member
                    # Committee data uses "LastName, FirstName" format
                    for name_key, info in comm_members.items():
                        # Check if representative name matches (fuzzy: last name match)
                        rep_parts = rep.replace(',', ' ').split()
                        key_parts = name_key.replace(',', ' ').split()
                        if rep_parts and key_parts and rep_parts[0] == key_parts[0]:
                            # Last name match — check sector overlap
                            member_sectors = info.get('sectors', [])
                            if sig_sector in member_sectors:
                                overlap = 1
                            break
                    conn.execute(
                        "UPDATE signals SET committee_overlap=? WHERE id=?",
                        (overlap, sig['id'])
                    )
                    comm_updated += 1
                # Fill non-congress signals with 0
                non_congress = conn.execute(
                    "UPDATE signals SET committee_overlap=0 "
                    "WHERE committee_overlap IS NULL AND source != 'congress'"
                ).rowcount
                comm_updated += non_congress
                updated += comm_updated
                if comm_updated:
                    log.info(f"Enriched committee_overlap for {comm_updated} signals")
        except Exception as e:
            log.warning(f"Committee enrichment failed: {e}")

    # ── 12. Earnings surprise (from data/earnings_surprise.json) ──
    # Insider buying after an earnings beat = accumulation; after miss = turnaround bet
    surprise_path = DATA_DIR / "earnings_surprise.json"
    if surprise_path.exists():
        try:
            surprise_raw = load_json(surprise_path)
            surprise_tickers = surprise_raw.get('tickers', {})
            if surprise_tickers:
                needs_surprise = conn.execute(
                    "SELECT id, ticker FROM signals WHERE earnings_surprise IS NULL"
                ).fetchall()
                surprise_updated = 0
                for sig in needs_surprise:
                    ticker = sig['ticker']
                    s = surprise_tickers.get(ticker)
                    if s:
                        conn.execute(
                            "UPDATE signals SET earnings_surprise=? WHERE id=?",
                            (s.get('surprise_pct', 0), sig['id'])
                        )
                    else:
                        conn.execute(
                            "UPDATE signals SET earnings_surprise=0 WHERE id=?",
                            (sig['id'],)
                        )
                    surprise_updated += 1
                updated += surprise_updated
                if surprise_updated:
                    log.info(f"Enriched earnings_surprise for {surprise_updated} signals")
        except Exception as e:
            log.warning(f"Earnings surprise enrichment failed: {e}")

    # ── 13. News sentiment (from data/news_sentiment.json) ──
    sentiment_path = DATA_DIR / "news_sentiment.json"
    if sentiment_path.exists():
        try:
            sent_raw = load_json(sentiment_path)
            sent_tickers = sent_raw.get('tickers', {})
            if sent_tickers:
                needs_sentiment = conn.execute(
                    "SELECT id, ticker FROM signals WHERE news_sentiment_30d IS NULL"
                ).fetchall()
                sent_updated = 0
                for sig in needs_sentiment:
                    ticker = sig['ticker']
                    s = sent_tickers.get(ticker)
                    if s:
                        conn.execute(
                            "UPDATE signals SET news_sentiment_30d=? WHERE id=?",
                            (s.get('sentiment_30d', 0), sig['id'])
                        )
                    else:
                        conn.execute(
                            "UPDATE signals SET news_sentiment_30d=0 WHERE id=?",
                            (sig['id'],)
                        )
                    sent_updated += 1
                updated += sent_updated
                if sent_updated:
                    log.info(f"Enriched news_sentiment_30d for {sent_updated} signals")
        except Exception as e:
            log.warning(f"News sentiment enrichment failed: {e}")

    # ── 14. Sentiment detail features (from data/news_sentiment.json) ──
    # Works with any scorer (VADER, FinBERT, keyword). Enriches:
    #   news_sentiment_score, news_sentiment_strong_positive,
    #   news_sentiment_strong_negative, news_insider_confluence, sentiment_divergence
    if sentiment_path.exists():
        try:
            sent_raw = load_json(sentiment_path)
            sent_tickers = sent_raw.get('tickers', {})
            sent_method = sent_raw.get('method', 'unknown')
            if sent_tickers:
                needs_detail = conn.execute(
                    "SELECT id, ticker, source FROM signals WHERE news_sentiment_score IS NULL"
                ).fetchall()
                detail_updated = 0
                for sig in needs_detail:
                    ticker = sig['ticker']
                    s = sent_tickers.get(ticker)
                    if s:
                        if sent_method == 'finbert':
                            # FinBERT: use dedicated score + confidence fields
                            score = s.get('sentiment_score', 0)
                            conf = s.get('sentiment_confidence', 0)
                            strong_pos = 1 if (score > 0.5 and conf > 0.75) else 0
                            strong_neg = 1 if (score < -0.5 and conf > 0.75) else 0
                        else:
                            # VADER/keyword: use sentiment_30d + headline counts
                            score = s.get('sentiment_30d', 0)
                            n = s.get('article_count', 1)
                            strong_pos = 1 if s.get('strong_positive_count', 0) >= max(1, n * 0.3) else 0
                            strong_neg = 1 if s.get('strong_negative_count', 0) >= max(1, n * 0.3) else 0
                        # news_insider_confluence: strong positive + insider buy
                        is_insider_buy = sig['source'] == 'EDGAR'
                        insider_conf = 1 if (strong_pos and is_insider_buy) else 0
                        # sentiment_divergence: strong negative news + insider buying
                        # (contrarian signal — insiders buying into bad press)
                        divergence = 1 if (strong_neg and is_insider_buy) else 0
                        conn.execute(
                            """UPDATE signals SET
                                news_sentiment_score=?,
                                news_sentiment_strong_positive=?,
                                news_sentiment_strong_negative=?,
                                news_insider_confluence=?,
                                sentiment_divergence=?
                            WHERE id=?""",
                            (score, strong_pos, strong_neg, insider_conf, divergence, sig['id'])
                        )
                    else:
                        conn.execute(
                            """UPDATE signals SET news_sentiment_score=0,
                                news_sentiment_strong_positive=0,
                                news_sentiment_strong_negative=0,
                                news_insider_confluence=0,
                                sentiment_divergence=0
                            WHERE id=?""",
                            (sig['id'],)
                        )
                    detail_updated += 1
                updated += detail_updated
                if detail_updated:
                    log.info(f"Enriched sentiment detail ({sent_method}) for {detail_updated} signals")
        except Exception as e:
            log.warning(f"Sentiment detail enrichment failed: {e}")

    # ── 15. Market regime (from vix_at_signal already in DB) ──
    needs_regime = conn.execute(
        "SELECT id, vix_at_signal FROM signals WHERE market_regime IS NULL"
    ).fetchall()
    regime_updated = 0
    for sig in needs_regime:
        vix = sig['vix_at_signal']
        regime = _vix_to_regime(vix)
        conn.execute("UPDATE signals SET market_regime=? WHERE id=?", (regime, sig['id']))
        regime_updated += 1
    updated += regime_updated
    if regime_updated:
        log.info(f"Assigned market_regime to {regime_updated} signals")

    # ── 16. Lobbying features (from data/lobbying_data.json) ──
    lobby_path = DATA_DIR / "lobbying_data.json"
    if lobby_path.exists():
        try:
            lobby_raw = load_json(lobby_path)
            lobby_tickers = lobby_raw.get('tickers', {})
            if lobby_tickers:
                needs_lobby = conn.execute(
                    "SELECT id, ticker, source, committee_overlap FROM signals "
                    "WHERE lobbying_active IS NULL"
                ).fetchall()
                lobby_updated = 0
                for sig in needs_lobby:
                    ticker = sig['ticker']
                    l = lobby_tickers.get(ticker)
                    if l:
                        active = 1 if l.get('lobbying_active') else 0
                        trend = l.get('lobbying_trend', 0)
                        # lobby_congress_confluence: lobbying + committee match + congress signal
                        is_congress = sig['source'] == 'congress'
                        has_committee = (sig['committee_overlap'] or 0) == 1
                        confluence = 1 if (active and has_committee and is_congress) else 0
                        conn.execute(
                            """UPDATE signals SET lobbying_active=?, lobbying_trend=?,
                                lobby_congress_confluence=? WHERE id=?""",
                            (active, trend, confluence, sig['id'])
                        )
                    else:
                        conn.execute(
                            "UPDATE signals SET lobbying_active=0, lobbying_trend=0, "
                            "lobby_congress_confluence=0 WHERE id=?",
                            (sig['id'],)
                        )
                    lobby_updated += 1
                updated += lobby_updated
                if lobby_updated:
                    log.info(f"Enriched lobbying features for {lobby_updated} signals")
        except Exception as e:
            log.warning(f"Lobbying enrichment failed: {e}")

    # ── 17. Hypothesis-driven interaction features (computed from existing DB cols) ──
    try:
        needs_interactions = conn.execute(
            "SELECT id, sector_momentum, same_ticker_signals_30d, volume_spike "
            "FROM signals WHERE sect_ticker_momentum IS NULL"
        ).fetchall()
        int_updated = 0
        for sig in needs_interactions:
            sm = sig['sector_momentum'] or 0
            st30 = sig['same_ticker_signals_30d'] or 0
            vs = sig['volume_spike'] or 0
            # sect_ticker_momentum: positive sector momentum × clustered insider activity
            # High values = sector trending up AND multiple insiders buying same ticker
            stm = sm * st30 if (sm > 0 and st30 > 1) else 0
            # volume_cluster_signal: elevated volume × clustered insider activity
            # High values = unusual volume AND multiple insiders piling in
            vcs = vs * st30 if (vs > 1.5 and st30 > 1) else 0
            conn.execute(
                "UPDATE signals SET sect_ticker_momentum=?, volume_cluster_signal=? WHERE id=?",
                (round(stm, 4), round(vcs, 4), sig['id'])
            )
            int_updated += 1
        updated += int_updated
        if int_updated:
            log.info(f"Enriched interaction features for {int_updated} signals")
    except Exception as e:
        log.warning(f"Interaction feature enrichment failed: {e}")

    # ── 18. Short interest features (from data/short_interest.json) ──
    try:
        si_path = DATA_DIR / "short_interest.json"
        if si_path.exists():
            with open(si_path) as f:
                si_data = json.load(f).get('tickers', {})
            if si_data:
                needs_si = conn.execute(
                    "SELECT id, ticker FROM signals WHERE short_interest_pct IS NULL"
                ).fetchall()
                si_updated = 0
                for sig in needs_si:
                    info = si_data.get(sig['ticker'])
                    if info:
                        si_pct = info.get('short_pct_float')
                        si_change = info.get('short_change_pct')
                        # Short squeeze signal: high SI (>15%) + SI increasing + insider buying
                        squeeze = 1 if (si_pct and si_pct > 0.15 and
                                        si_change and si_change > 0.05) else 0
                        conn.execute(
                            "UPDATE signals SET short_interest_pct=?, "
                            "short_interest_change=?, short_squeeze_signal=? WHERE id=?",
                            (si_pct, si_change, squeeze, sig['id'])
                        )
                        si_updated += 1
                updated += si_updated
                if si_updated:
                    log.info(f"Enriched short interest for {si_updated} signals")
    except Exception as e:
        log.warning(f"Short interest enrichment failed: {e}")

    # ── 19. Institutional ownership features (from data/institutional_data.json) ──
    try:
        inst_path = DATA_DIR / "institutional_data.json"
        if inst_path.exists():
            with open(inst_path) as f:
                inst_data = json.load(f).get('tickers', {})
            if inst_data:
                needs_inst = conn.execute(
                    "SELECT id, ticker, source FROM signals WHERE institutional_holders IS NULL"
                ).fetchall()
                inst_updated = 0
                for sig in needs_inst:
                    info = inst_data.get(sig['ticker'])
                    if info:
                        n_holders = info.get('n_holders')
                        pct_held = info.get('total_pct_held')
                        # Confluence: institutional attention (5+ holders) + insider buy
                        is_insider = sig['source'] == 'edgar'
                        confluence = 1 if (n_holders and n_holders >= 5 and is_insider) else 0
                        conn.execute(
                            "UPDATE signals SET institutional_holders=?, "
                            "institutional_pct_held=?, institutional_insider_confluence=? "
                            "WHERE id=?",
                            (n_holders, pct_held, confluence, sig['id'])
                        )
                        inst_updated += 1
                updated += inst_updated
                if inst_updated:
                    log.info(f"Enriched institutional data for {inst_updated} signals")
    except Exception as e:
        log.warning(f"Institutional enrichment failed: {e}")

    # ── 20. Options flow features (from data/options_flow.json) ──
    try:
        opt_path = DATA_DIR / "options_flow.json"
        if opt_path.exists():
            with open(opt_path) as f:
                opt_data = json.load(f).get('tickers', {})
            if opt_data:
                needs_opt = conn.execute(
                    "SELECT id, ticker, source FROM signals WHERE options_bullish IS NULL"
                ).fetchall()
                opt_updated = 0
                for sig in needs_opt:
                    info = opt_data.get(sig['ticker'])
                    if info:
                        bullish = 1 if info.get('bullish_options') else 0
                        bearish = info.get('bearish_options', False)
                        unusual = 1 if info.get('unusual_otm_calls') else 0
                        is_insider = sig['source'] == 'edgar'
                        # Confluence: unusual OTM calls + insider buy
                        insider_conf = 1 if (unusual and is_insider) else 0
                        # Divergence: bearish options flow + insider buy anyway
                        bearish_div = 1 if (bearish and is_insider) else 0
                        conn.execute(
                            "UPDATE signals SET options_bullish=?, options_unusual_calls=?, "
                            "options_insider_confluence=?, options_bearish_divergence=? "
                            "WHERE id=?",
                            (bullish, unusual, insider_conf, bearish_div, sig['id'])
                        )
                        opt_updated += 1
                updated += opt_updated
                if opt_updated:
                    log.info(f"Enriched options flow for {opt_updated} signals")
    except Exception as e:
        log.warning(f"Options flow enrichment failed: {e}")

    conn.commit()
    log.info(f"Enriched {updated} features")
    return updated


def _vix_to_regime(vix: float | None) -> str:
    """Map VIX level to market regime category."""
    if vix is None:
        return 'normal'
    if vix < 15:
        return 'low_vol'
    elif vix <= 25:
        return 'normal'
    elif vix <= 35:
        return 'elevated'
    else:
        return 'crisis'


def backfill_features(conn: sqlite3.Connection) -> dict:
    """Re-enrich analyst + volume features for ALL signals (not just NULLs).

    This resets the v5 feature columns to NULL so enrich_signal_features()
    will recompute them from the latest data on disk. Useful when:
      - analyst_data.json was empty during initial enrichment
      - volume data was missing at bootstrap time
      - Feature logic was updated and needs to propagate

    Returns dict with counts of signals re-enriched.
    """
    log.info("=== Backfill Features (v5-v9: volume, analyst, committee, sentiment, interactions, short interest) ===")

    # Step 0: Fill signal outcomes first — ensures car_30d is available
    # before features that depend on it (sector_avg_car, person_avg_car_30d)
    outcome_filled = backfill_outcomes(conn, full=True)
    if outcome_filled:
        log.info(f"Pre-filled {outcome_filled} signal outcomes from price cache")
    # Also fill SPY returns and market-adjusted CARs
    enrich_spy_returns(conn)
    enrich_market_adj_returns(conn)

    # Step 0b: Ingest any new signals from feed JSONs (congress, EDGAR, 13F)
    # This ensures --backfill picks up signals that fetch_data.py downloaded
    c_count = ingest_congress_feed(conn)
    e_count = ingest_edgar_feed(conn)
    f_count = ingest_13f_feed(conn)
    if c_count or e_count or f_count:
        log.info(f"Ingested: {c_count} congress + {e_count} EDGAR + {f_count} 13F new signals")
        # Fill outcomes for newly ingested signals
        new_filled = backfill_outcomes(conn, full=False)
        if new_filled:
            log.info(f"Filled outcomes for {new_filled} newly ingested signals")
        enrich_spy_returns(conn)
        enrich_market_adj_returns(conn)

    BACKFILL_COLS = ('volume_dry_up', 'analyst_revision_30d', 'analyst_consensus',
                     'analyst_insider_confluence', 'committee_overlap',
                     'earnings_surprise', 'news_sentiment_30d',
                     'news_sentiment_score', 'news_sentiment_strong_positive',
                     'news_sentiment_strong_negative', 'news_insider_confluence',
                     'sentiment_divergence', 'market_regime',
                     'lobbying_active', 'lobbying_trend', 'lobby_congress_confluence',
                     'sect_ticker_momentum', 'volume_cluster_signal',
                     'spy_return_30d', 'market_adj_car_30d',
                     'short_interest_pct', 'short_interest_change', 'short_squeeze_signal',
                     'institutional_holders', 'institutional_pct_held',
                     'institutional_insider_confluence',
                     'options_bullish', 'options_unusual_calls',
                     'options_insider_confluence', 'options_bearish_divergence',
                     'person_hit_rate_30d', 'person_avg_car_30d')

    # 1. Count current state
    total = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    nulls_before = {}
    for col in BACKFILL_COLS:
        n = conn.execute(f"SELECT COUNT(*) FROM signals WHERE {col} IS NULL").fetchone()[0]
        nulls_before[col] = n

    log.info(f"Total signals: {total}")
    for col, n in nulls_before.items():
        log.info(f"  {col}: {n}/{total} NULL ({100*n/total:.1f}%)")

    # 2. Reset v5/v6 columns to NULL so enrich picks them all up
    conn.execute("UPDATE signals SET volume_dry_up = NULL")
    conn.execute("""UPDATE signals SET analyst_revision_30d = NULL,
                     analyst_consensus = NULL,
                     analyst_insider_confluence = NULL""")
    conn.execute("UPDATE signals SET committee_overlap = NULL")
    conn.execute("UPDATE signals SET earnings_surprise = NULL")
    conn.execute("UPDATE signals SET news_sentiment_30d = NULL")
    conn.execute("""UPDATE signals SET news_sentiment_score = NULL,
                     news_sentiment_strong_positive = NULL,
                     news_sentiment_strong_negative = NULL,
                     news_insider_confluence = NULL,
                     sentiment_divergence = NULL""")
    conn.execute("UPDATE signals SET market_regime = NULL")
    conn.execute("""UPDATE signals SET lobbying_active = NULL,
                     lobbying_trend = NULL, lobby_congress_confluence = NULL""")
    conn.execute("""UPDATE signals SET sect_ticker_momentum = NULL,
                     volume_cluster_signal = NULL""")
    conn.execute("""UPDATE signals SET spy_return_30d = NULL,
                     market_adj_car_30d = NULL""")
    conn.execute("""UPDATE signals SET short_interest_pct = NULL,
                     short_interest_change = NULL, short_squeeze_signal = NULL""")
    conn.execute("UPDATE signals SET sector_avg_car = NULL")  # recompute point-in-time
    # Reset person track records to recompute with point-in-time outcome filtering
    conn.execute("""UPDATE signals SET person_trade_count = 0, person_hit_rate_30d = NULL,
                     person_avg_car_30d = NULL, person_hit_rate_90d = NULL,
                     person_avg_car_90d = NULL, relative_position_size = NULL""")
    conn.execute("""UPDATE signals SET institutional_holders = NULL,
                     institutional_pct_held = NULL, institutional_insider_confluence = NULL""")
    conn.execute("""UPDATE signals SET options_bullish = NULL, options_unusual_calls = NULL,
                     options_insider_confluence = NULL, options_bearish_divergence = NULL""")
    conn.commit()
    log.info("Reset v5-v10 columns + sector_avg_car to NULL for re-enrichment")

    # 3. Run the standard enrichment (handles the NULL → computed fill)
    enriched = enrich_signal_features(conn)

    # 3b. Recompute person track records (reset at step 2 but not in enrich_signal_features)
    person_updated = update_person_track_records(conn)
    log.info(f"Re-enriched person track records for {person_updated} signals")

    # 3c. Enrich SPY returns + market-adjusted CARs
    enrich_spy_returns(conn)
    enrich_market_adj_returns(conn)

    # 3d. Enrich liquidity / transaction cost features
    enrich_liquidity_features(conn)

    # 4. Report post-backfill state
    nulls_after = {}
    for col in BACKFILL_COLS:
        n = conn.execute(f"SELECT COUNT(*) FROM signals WHERE {col} IS NULL").fetchone()[0]
        nulls_after[col] = n

    log.info("Post-backfill NULL counts:")
    for col, n in nulls_after.items():
        filled = nulls_before[col] - n
        log.info(f"  {col}: {n}/{total} NULL (filled {filled})")

    result = {
        'total_signals': total,
        'enriched': enriched,
        'nulls_before': nulls_before,
        'nulls_after': nulls_after,
    }
    return result


# ── Legislative Catalyst Registry ─────────────────────────────────────────────
# Dynamically loaded from bills_feed.json (populated by scripts/fetch_bills.py).
# Falls back to hardcoded list if bills_feed.json doesn't exist yet.

BILLS_FEED = DATA_DIR / "bills_feed.json"

_HARDCODED_CATALYSTS = [
    {'id': 'HR7821', 'date': '2026-02-25', 'tickers': ['NVDA', 'AMD', 'SMCI', 'MSFT'],
     'sector': 'Technology', 'title': 'American AI Infrastructure Act'},
    {'id': 'SB1882', 'date': '2026-03-03', 'tickers': ['RTX', 'LMT', 'NOC', 'BA'],
     'sector': 'Industrials', 'title': 'National Defense Modernization Act'},
    {'id': 'SB2241', 'date': '2026-03-10', 'tickers': ['PFE', 'MRK', 'ABBV'],
     'sector': 'Healthcare', 'title': 'Drug Price Negotiation Expansion Act'},
    {'id': 'HR6419', 'date': '2026-03-17', 'tickers': ['FCX', 'MP', 'ALB', 'LTHM'],
     'sector': 'Basic Materials', 'title': 'Critical Minerals & Mining Security Act'},
    {'id': 'SB3310', 'date': '2026-04-01', 'tickers': ['ENPH', 'NEE', 'FSLR'],
     'sector': 'Energy', 'title': 'Clean Energy Investment Act'},
]


def _load_legislative_catalysts() -> list:
    """Load bill catalysts from bills_feed.json, falling back to hardcoded list."""
    if BILLS_FEED.exists():
        try:
            data = load_json(BILLS_FEED)
            bills = data.get('bills', [])
            if bills:
                catalysts = []
                for b in bills:
                    date = b.get('action_date') or b.get('introduced_date') or ''
                    tickers = b.get('impact_tickers', [])
                    if date and tickers:
                        catalysts.append({
                            'id': b.get('id', ''),
                            'date': date,
                            'tickers': tickers,
                            'sector': b.get('sector', ''),
                            'title': b.get('title', ''),
                        })
                if catalysts:
                    return catalysts
        except Exception:
            pass
    return _HARDCODED_CATALYSTS


def _build_catalyst_maps() -> tuple:
    """Build ticker → dates and sector → dates lookups from catalysts."""
    catalysts = _load_legislative_catalysts()
    ticker_map = {}
    sector_map = {}
    for bill in catalysts:
        for t in bill['tickers']:
            ticker_map.setdefault(t, []).append(bill['date'])
        sector_map.setdefault(bill['sector'], []).append(bill['date'])
    return ticker_map, sector_map


# Build lookups (refreshed on import)
_CATALYST_TICKER_MAP, _CATALYST_SECTOR_MAP = _build_catalyst_maps()


def _enrich_catalyst_proximity(conn: sqlite3.Connection, signals: list) -> int:
    """Compute days_to_catalyst for each signal.
    Checks if the ticker or its sector has an upcoming bill/catalyst event.
    A negative value means the catalyst already happened (post-catalyst signal).
    """
    updated = 0
    for sig in signals:
        ticker = sig['ticker']
        sig_date = sig['signal_date']
        sig_dt = datetime.strptime(sig_date, '%Y-%m-%d')

        # Check direct ticker match first, then sector match
        catalyst_dates = _CATALYST_TICKER_MAP.get(ticker, [])

        if not catalyst_dates:
            # Try sector-level match
            sector = get_sector(ticker)
            if sector:
                catalyst_dates = _CATALYST_SECTOR_MAP.get(sector, [])

        if not catalyst_dates:
            # No catalyst for this ticker/sector — set to large number (no catalyst)
            conn.execute(
                "UPDATE signals SET days_to_catalyst=? WHERE id=?",
                (999, sig['id'])
            )
            updated += 1
            continue

        # Find the nearest catalyst date (can be before or after the signal)
        min_days = None
        for cat_date in catalyst_dates:
            cat_dt = datetime.strptime(cat_date, '%Y-%m-%d')
            delta = (cat_dt - sig_dt).days
            if min_days is None or abs(delta) < abs(min_days):
                min_days = delta

        if min_days is not None:
            conn.execute(
                "UPDATE signals SET days_to_catalyst=? WHERE id=?",
                (min_days, sig['id'])
            )
            updated += 1

    return updated


def _enrich_earnings_proximity(conn: sqlite3.Connection, tickers: list) -> int:
    """Compute days_to_earnings for each signal using yfinance earnings calendar.
    Uses quarterly earnings dates from yfinance. For each signal, finds the
    nearest upcoming earnings date at signal time.

    This only uses local data if an earnings cache exists, otherwise fetches
    from yfinance (rate-limited). Results are cached to data/earnings_calendar.json.
    """
    import time as _time

    # Load or create earnings cache
    earnings_cache_path = DATA_DIR / "earnings_calendar.json"
    earnings_cache = {}
    if earnings_cache_path.exists():
        try:
            earnings_cache = load_json(earnings_cache_path)
        except (json.JSONDecodeError, OSError):
            pass

    tickers_to_fetch = [t for t in tickers if t not in earnings_cache]
    if tickers_to_fetch:
        log.info(f"Fetching earnings calendars for {len(tickers_to_fetch)} tickers...")
        try:
            import yfinance as yf
        except ImportError:
            log.warning("yfinance not installed — skipping earnings proximity")
            return 0

        fetched = 0
        for i, ticker in enumerate(tickers_to_fetch):
            try:
                stock = yf.Ticker(ticker)
                # Get earnings dates — yfinance provides historical + upcoming
                cal = stock.get_earnings_dates(limit=20)
                if cal is not None and not cal.empty:
                    dates = [d.strftime('%Y-%m-%d') for d in cal.index]
                    earnings_cache[ticker] = dates
                    fetched += 1
                else:
                    earnings_cache[ticker] = []
            except Exception:
                earnings_cache[ticker] = []

            _time.sleep(0.2)  # rate limit

            if (i + 1) % 100 == 0:
                log.info(f"  ...{i+1}/{len(tickers_to_fetch)} earnings calendars fetched")

        log.info(f"Fetched earnings dates for {fetched}/{len(tickers_to_fetch)} tickers")

        # Save cache
        save_json(earnings_cache_path, earnings_cache)

    # Now compute days_to_earnings for each signal
    updated = 0
    for ticker in tickers:
        earning_dates = earnings_cache.get(ticker, [])
        if not earning_dates:
            # No earnings data — set to 999 (unknown)
            conn.execute(
                "UPDATE signals SET days_to_earnings=? WHERE ticker=? AND days_to_earnings IS NULL",
                (999, ticker)
            )
            count = conn.execute(
                "SELECT changes()"
            ).fetchone()[0]
            updated += count
            continue

        # Parse earnings dates once
        parsed_earnings = []
        for d in earning_dates:
            try:
                parsed_earnings.append(datetime.strptime(d[:10], '%Y-%m-%d'))
            except ValueError:
                pass

        if not parsed_earnings:
            conn.execute(
                "UPDATE signals SET days_to_earnings=? WHERE ticker=? AND days_to_earnings IS NULL",
                (999, ticker)
            )
            updated += conn.execute("SELECT changes()").fetchone()[0]
            continue

        signals = conn.execute(
            "SELECT id, signal_date FROM signals WHERE ticker=? AND days_to_earnings IS NULL",
            (ticker,)
        ).fetchall()

        for sig in signals:
            sig_dt = datetime.strptime(sig['signal_date'], '%Y-%m-%d')
            # Find nearest future earnings date at signal time
            future_earnings = [e for e in parsed_earnings if e >= sig_dt]
            if future_earnings:
                nearest = min(future_earnings)
                days = (nearest - sig_dt).days
            else:
                # All earnings in the past — find nearest past
                past_earnings = [e for e in parsed_earnings if e < sig_dt]
                if past_earnings:
                    nearest = max(past_earnings)
                    days = (nearest - sig_dt).days  # negative
                else:
                    days = 999

            conn.execute(
                "UPDATE signals SET days_to_earnings=? WHERE id=?",
                (days, sig['id'])
            )
            updated += 1

    return updated


# ── Market Context (FRED) ────────────────────────────────────────────────────

# FRED series IDs for market context snapshots
FRED_SERIES = {
    'vix_at_signal': 'VIXCLS',           # VIX index
    'yield_curve_at_signal': 'T10Y2Y',   # 10yr-2yr spread
    'credit_spread_at_signal': 'BAMLH0A0HYM2',  # HY credit OAS
}

# Module-level cache for FRED time series (date -> value)
_fred_cache: dict = {}


def _fetch_fred_series(series_id: str, api_key: str) -> dict:
    """Fetch a full FRED time series and return {date_str: float_value}.
    Caches in memory for the session."""
    if series_id in _fred_cache:
        return _fred_cache[series_id]

    import requests
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?api_key={api_key}&series_id={series_id}"
        f"&observation_start=2022-01-01&sort_order=asc&limit=5000&file_type=json"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        result = {}
        for obs in data.get('observations', []):
            val = obs.get('value', '.')
            if val not in ('.', ''):
                try:
                    result[obs['date']] = float(val)
                except (ValueError, TypeError):
                    pass
        _fred_cache[series_id] = result
        log.info(f"FRED {series_id}: fetched {len(result)} observations")
        return result
    except Exception as e:
        log.warning(f"FRED {series_id} fetch failed: {e}")
        _fred_cache[series_id] = {}
        return {}


def _get_fred_value_at_date(series: dict, date_str: str) -> float | None:
    """Get FRED value at a date, with ±5 day tolerance for weekends/holidays."""
    if date_str in series:
        return series[date_str]
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    for offset in range(1, 6):
        for delta in (-offset, offset):
            candidate = (dt + timedelta(days=delta)).strftime('%Y-%m-%d')
            if candidate in series:
                return series[candidate]
    return None


def enrich_market_context(conn: sqlite3.Connection, api_key: str = None) -> int:
    """Populate vix_at_signal, yield_curve_at_signal, credit_spread_at_signal
    for signals missing these values. Fetches from FRED API.

    Args:
        conn: Database connection
        api_key: FRED API key. If None, reads from FRED_KEY env var.

    Returns count of signals updated.
    """
    import os as _os
    if api_key is None:
        api_key = _os.environ.get('FRED_KEY', '71a0b94ed47b56a81f405947f88d08aa')

    if not api_key:
        log.warning("No FRED API key — skipping market context enrichment")
        return 0

    # Find signals missing any market context column
    needs_enrichment = conn.execute(
        "SELECT id, signal_date FROM signals "
        "WHERE vix_at_signal IS NULL OR yield_curve_at_signal IS NULL "
        "OR credit_spread_at_signal IS NULL"
    ).fetchall()

    if not needs_enrichment:
        log.info("All signals already have market context — skipping")
        return 0

    log.info(f"Enriching market context for {len(needs_enrichment)} signals...")

    # Fetch all three FRED series
    fred_data = {}
    for col, series_id in FRED_SERIES.items():
        fred_data[col] = _fetch_fred_series(series_id, api_key)

    updated = 0
    for sig in needs_enrichment:
        sig_date = sig['signal_date']
        updates = {}

        for col, series_id in FRED_SERIES.items():
            series = fred_data.get(col, {})
            if not series:
                continue
            current_val = conn.execute(
                f"SELECT {col} FROM signals WHERE id=?", (sig['id'],)
            ).fetchone()
            if current_val and current_val[col] is not None:
                continue
            val = _get_fred_value_at_date(series, sig_date)
            if val is not None:
                updates[col] = round(val, 4)

        if updates:
            set_clause = ', '.join(f'{k}=?' for k in updates.keys())
            conn.execute(
                f"UPDATE signals SET {set_clause} WHERE id=?",
                list(updates.values()) + [sig['id']]
            )
            updated += 1

    conn.commit()
    log.info(f"Market context: enriched {updated} signals with VIX/yield curve/credit spread")
    return updated


# ── Outcome Backfilling ──────────────────────────────────────────────────────

def load_price_index(ticker: str) -> dict:
    """Load {date: close} price index for a ticker."""
    path = PRICE_HISTORY_DIR / f"{ticker}.json"
    if not path.exists():
        return {}
    candles = load_json(path)
    return {date: v['c'] for date, v in candles.items()}


def get_return(price_index: dict, start_date: str, days: int) -> float | None:
    """Get the return from start_date to start_date+days (with ±5 day tolerance on both ends).

    Uses the nearest available trading day for both base and target prices.
    This handles weekends, holidays, and minor gaps in price data.
    """
    # Find base price with ±5 day tolerance (handles weekends/holidays)
    dt = datetime.strptime(start_date, '%Y-%m-%d')
    base = None
    base_date = None
    for offset in sorted(range(-5, 6), key=abs):
        candidate = (dt + timedelta(days=offset)).strftime('%Y-%m-%d')
        if candidate in price_index and price_index[candidate]:
            base = price_index[candidate]
            base_date = candidate
            break
    if base is None or base == 0:
        return None

    # Find target price with ±5 day tolerance
    target = dt + timedelta(days=days)
    for offset in sorted(range(-5, 6), key=abs):
        candidate = (target + timedelta(days=offset)).strftime('%Y-%m-%d')
        if candidate == base_date:
            continue
        if candidate in price_index and price_index[candidate]:
            return (price_index[candidate] - base) / base
    return None


def backfill_outcomes(conn: sqlite3.Connection, spy_index: dict = None, full: bool = False) -> int:
    """Backfill return/CAR outcomes for signals where enough time has passed.

    Args:
        full: If True, process ALL signals (not just last 400 days).
              Used by --backfill to ensure historical signals get outcomes.
    """
    if spy_index is None:
        spy_index = load_price_index('SPY')

    today = datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')
    today_dt = datetime.strptime(today, '%Y-%m-%d')
    filled = 0

    # Fix prior data: reset filled flags where CAR is NULL (bad prior backfills)
    for window in [5, 30, 90, 180, 365]:
        fixed = conn.execute(
            f"UPDATE signals SET outcome_{window}d_filled = 0 "
            f"WHERE outcome_{window}d_filled = 1 AND car_{window}d IS NULL"
        ).rowcount
        if fixed:
            log.info(f"Reset {fixed} signals with filled={1} but null car_{window}d")

    # Clip existing CARs to hard bounds [-100%, +300%]
    for window in [5, 30, 90, 180, 365]:
        clipped = conn.execute(
            f"UPDATE signals SET car_{window}d = MAX(-1.0, MIN(3.0, car_{window}d)) "
            f"WHERE car_{window}d IS NOT NULL AND (car_{window}d > 3.0 OR car_{window}d < -1.0)"
        ).rowcount
        if clipped:
            log.info(f"Clipped {clipped} out-of-bounds car_{window}d values to [-100%, +300%]")
    conn.commit()

    # Get signals needing backfill (any horizon still unfilled)
    date_filter = "" if full else "AND signal_date >= date('now', '-400 days') "
    rows = conn.execute(
        "SELECT id, ticker, signal_date, outcome_5d_filled, outcome_30d_filled, "
        "outcome_90d_filled, outcome_180d_filled, outcome_365d_filled "
        f"FROM signals WHERE (outcome_5d_filled = 0 OR outcome_30d_filled = 0 "
        "OR outcome_90d_filled = 0 OR outcome_180d_filled = 0 OR outcome_365d_filled = 0) "
        f"{date_filter}"
    ).fetchall()

    windows = [
        (5, 'outcome_5d_filled'),
        (30, 'outcome_30d_filled'),
        (90, 'outcome_90d_filled'),
        (180, 'outcome_180d_filled'),
        (365, 'outcome_365d_filled'),
    ]

    for row in rows:
        sig_dt = datetime.strptime(row['signal_date'], '%Y-%m-%d')
        days_elapsed = (today_dt - sig_dt).days
        ticker = row['ticker']
        price_index = load_price_index(ticker)
        if not price_index:
            continue

        # Get price at signal (with ±5 day tolerance for weekends/holidays)
        price_at = None
        for offset in sorted(range(-5, 6), key=abs):
            candidate = (sig_dt + timedelta(days=offset)).strftime('%Y-%m-%d')
            if candidate in price_index and price_index[candidate]:
                price_at = price_index[candidate]
                break

        updates = {}
        if price_at:
            updates['price_at_signal'] = round(price_at, 4)

        for window, filled_col in windows:
            if row[filled_col]:
                continue
            if days_elapsed < window + 5:  # need window + tolerance
                continue

            stock_ret = get_return(price_index, row['signal_date'], window)
            spy_ret = get_return(spy_index, row['signal_date'], window)

            if stock_ret is not None and spy_ret is not None:
                updates[f'return_{window}d'] = round(stock_ret, 6)
                # Use BHAR (buy-and-hold abnormal return) for industry-standard
                # event study methodology. BHAR = (1+r_stock)/(1+r_bench) - 1
                # This correctly handles compounding over longer horizons.
                bhar = (1 + stock_ret) / (1 + spy_ret) - 1
                # Hard-clip CARs to [-100%, +300%] to prevent outliers
                bhar = max(-1.0, min(3.0, bhar))
                updates[f'car_{window}d'] = round(bhar, 6)
                updates[filled_col] = 1

        if updates:
            set_clause = ', '.join(f'{k}=?' for k in updates.keys())
            conn.execute(
                f"UPDATE signals SET {set_clause} WHERE id=?",
                list(updates.values()) + [row['id']]
            )
            filled += 1

    conn.commit()
    return filled


def enrich_spy_returns(conn: sqlite3.Connection) -> int:
    """Compute date-matched SPY 30d returns for each signal.

    For each signal with a signal_date, compute SPY's actual 30-day return
    starting from that date. This gives a matched benchmark for alpha calculation.
    """
    spy_index = load_price_index('SPY')
    if not spy_index:
        log.warning("SPY price data not available — skipping spy_return enrichment")
        return 0

    rows = conn.execute(
        "SELECT id, signal_date FROM signals WHERE spy_return_30d IS NULL"
    ).fetchall()

    if not rows:
        return 0

    updated = 0
    for row in rows:
        ret = get_return(spy_index, row['signal_date'], days=30)
        if ret is not None:
            conn.execute(
                "UPDATE signals SET spy_return_30d=? WHERE id=?",
                (round(ret, 6), row['id'])
            )
            updated += 1

    conn.commit()
    if updated:
        log.info(f"SPY 30d returns enriched for {updated} signals")
    return updated


def enrich_market_adj_returns(conn: sqlite3.Connection) -> int:
    """Compute market-adjusted CAR = car_30d - spy_return_30d for each signal."""
    result = conn.execute("""
        UPDATE signals
        SET market_adj_car_30d = ROUND(car_30d - spy_return_30d, 6)
        WHERE car_30d IS NOT NULL
          AND spy_return_30d IS NOT NULL
          AND market_adj_car_30d IS NULL
    """)
    conn.commit()
    updated = result.rowcount
    if updated:
        log.info(f"Market-adjusted CARs computed for {updated} signals")
    return updated


def compute_alpha_metrics(conn: sqlite3.Connection) -> dict:
    """Compute date-matched alpha metrics by score band and source.

    Returns dict with:
      alpha_all_signals: mean excess return vs SPY (all signals)
      alpha_80plus: mean excess return for 80+ scored signals
      alpha_congress/edgar: by source
      beta_vs_spy: sensitivity to SPY returns
      sharpe_market_adjusted_80plus: Sharpe on excess returns (80+)
      n_matched_signals: count with both car and spy_return
    """
    import numpy as np

    rows = conn.execute("""
        SELECT total_score, car_30d, spy_return_30d,
               market_adj_car_30d, source
        FROM signals
        WHERE car_30d IS NOT NULL AND spy_return_30d IS NOT NULL
    """).fetchall()

    if len(rows) < 10:
        return {}

    all_adj = [r['market_adj_car_30d'] for r in rows if r['market_adj_car_30d'] is not None]
    cars = np.array([r['car_30d'] for r in rows])
    spy_rets = np.array([r['spy_return_30d'] for r in rows])

    result = {
        'alpha_all_signals': round(float(np.mean(all_adj)), 6) if all_adj else None,
        'n_matched_signals': len(rows),
        'spy_avg_30d_return': round(float(np.mean(spy_rets)), 6),
    }

    # Alpha by score band
    for band_name, lo, hi in [('80plus', 80, 200), ('60_79', 60, 80), ('below_60', 0, 60)]:
        band_adj = [r['market_adj_car_30d'] for r in rows
                    if r['total_score'] is not None and lo <= (r['total_score'] or 0) < hi
                    and r['market_adj_car_30d'] is not None]
        if band_adj:
            result[f'alpha_{band_name}'] = round(float(np.mean(band_adj)), 6)
            result[f'n_{band_name}'] = len(band_adj)

    # Alpha by source
    for src in ['congress', 'EDGAR']:
        src_adj = [r['market_adj_car_30d'] for r in rows
                   if r['source'] == src and r['market_adj_car_30d'] is not None]
        if src_adj:
            result[f'alpha_{src.lower()}'] = round(float(np.mean(src_adj)), 6)

    # Beta = cov(signal_car, spy_return) / var(spy_return)
    # Winsorize at 1st/99th percentile to prevent outlier distortion
    cars_w = np.clip(cars, np.percentile(cars, 1), np.percentile(cars, 99))
    spy_w = np.clip(spy_rets, np.percentile(spy_rets, 1), np.percentile(spy_rets, 99))
    spy_var = float(np.var(spy_w))
    if spy_var > 0:
        cov = float(np.cov(cars_w, spy_w)[0][1])
        result['beta_vs_spy'] = round(cov / spy_var, 4)
        # Also store raw beta for comparison
        raw_var = float(np.var(spy_rets))
        if raw_var > 0:
            raw_cov = float(np.cov(cars, spy_rets)[0][1])
            result['beta_vs_spy_raw'] = round(raw_cov / raw_var, 4)

    # Sharpe on market-adjusted returns (80+ signals)
    adj_80 = [r['market_adj_car_30d'] for r in rows
              if r['total_score'] is not None and (r['total_score'] or 0) >= 80
              and r['market_adj_car_30d'] is not None]
    if len(adj_80) > 10:
        adj_arr = np.array(adj_80)
        std = float(np.std(adj_arr))
        if std > 0:
            result['sharpe_market_adjusted_80plus'] = round(
                float(np.mean(adj_arr)) / std * np.sqrt(12), 4)

    return result


# ── Feature Analysis ─────────────────────────────────────────────────────────

def _normalize_role(role: str) -> str:
    """Bucket insider roles for feature analysis."""
    if not role:
        return 'n/a'
    r = role.upper()
    if 'CEO' in r or 'CHIEF EXECUTIVE' in r:
        return 'CEO'
    if 'CFO' in r or 'CHIEF FINANCIAL' in r:
        return 'CFO'
    if 'COO' in r or 'CHIEF OPERATING' in r:
        return 'COO'
    if 'VP' in r or 'VICE PRESIDENT' in r:
        return 'VP'
    if 'PRESIDENT' in r:
        return 'President'
    if 'DIRECTOR' in r:
        return 'Director'
    if 'OFFICER' in r:
        return 'Officer'
    return 'Other'


def classify_insider_pattern(history: list) -> str:
    """Classify insider as opportunistic vs routine (Cohen, Malloy, Pomorski 2012).
    Routine = trades in same calendar month for 3+ consecutive years."""
    if len(history) < 3:
        return 'insufficient_history'

    dates = sorted(datetime.strptime(h['date'][:10], '%Y-%m-%d') for h in history if h.get('date'))
    if len(dates) < 3:
        return 'insufficient_history'

    # Check if any single month has trades in 3+ different years
    from collections import Counter
    month_years = Counter()
    for d in dates:
        month_years[d.month] += 1

    for month, count in month_years.items():
        if count >= 3:
            years = sorted(set(d.year for d in dates if d.month == month))
            for i in range(len(years) - 2):
                if years[i+2] - years[i] <= 2:
                    return 'routine'

    return 'opportunistic'


def compute_52wk_proximity(price: float, high_52wk: float, low_52wk: float) -> float | None:
    """Compute 0-1 proximity to 52-week range. 0 = at low, 1 = at high."""
    if high_52wk is None or low_52wk is None or price is None:
        return None
    range_val = high_52wk - low_52wk
    if range_val <= 0:
        return None
    return round(max(0.0, min(1.0, (price - low_52wk) / range_val)), 4)


def _compute_momentum_features(price_index: dict, signal_date: str) -> dict:
    """Compute 1m, 3m, 6m momentum (returns) before the signal date.

    Uses trading-day approximations: 1m≈21d, 3m≈63d, 6m≈126d.
    Returns dict with momentum_1m, momentum_3m, momentum_6m (or None if insufficient data).
    """
    if not price_index:
        return {'momentum_1m': None, 'momentum_3m': None, 'momentum_6m': None}

    # Get the signal date's price (with ±5 day tolerance)
    dt = datetime.strptime(signal_date, '%Y-%m-%d')
    base_price = None
    for offset in sorted(range(-5, 6), key=abs):
        candidate = (dt + timedelta(days=offset)).strftime('%Y-%m-%d')
        if candidate in price_index and price_index[candidate]:
            base_price = price_index[candidate]
            break

    if not base_price or base_price == 0:
        return {'momentum_1m': None, 'momentum_3m': None, 'momentum_6m': None}

    result = {}
    for label, cal_days in [('momentum_1m', 30), ('momentum_3m', 90), ('momentum_6m', 180)]:
        past_dt = dt - timedelta(days=cal_days)
        past_price = None
        for offset in sorted(range(-5, 6), key=abs):
            candidate = (past_dt + timedelta(days=offset)).strftime('%Y-%m-%d')
            if candidate in price_index and price_index[candidate]:
                past_price = price_index[candidate]
                break
        if past_price and past_price > 0:
            result[label] = round((base_price - past_price) / past_price, 6)
        else:
            result[label] = None

    return result


def _compute_volume_spike(candles: dict, signal_date: str) -> float | None:
    """Compute volume spike: ratio of signal-date volume to 20-day average.

    Args:
        candles: Full candle dict {date: {o, h, l, c, v}} for the ticker
        signal_date: YYYY-MM-DD

    Returns ratio or None if insufficient data.
    """
    if not candles:
        return None

    # Get signal date volume (with ±2 day tolerance)
    dt = datetime.strptime(signal_date, '%Y-%m-%d')
    signal_vol = None
    for offset in sorted(range(-2, 3), key=abs):
        candidate = (dt + timedelta(days=offset)).strftime('%Y-%m-%d')
        if candidate in candles and candles[candidate].get('v'):
            signal_vol = candles[candidate]['v']
            break

    if not signal_vol:
        return None

    # Compute 20-day average volume before signal date
    sorted_dates = sorted(d for d in candles.keys() if d < signal_date)
    recent = sorted_dates[-20:]  # last 20 trading days before signal
    if len(recent) < 10:
        return None

    avg_vol = sum(candles[d].get('v', 0) for d in recent) / len(recent)
    if avg_vol <= 0:
        return None

    return round(signal_vol / avg_vol, 4)


def _compute_insider_buy_ratio(conn: sqlite3.Connection, ticker: str,
                                signal_date: str, signal_id: int) -> float:
    """Compute log(1 + count of same-ticker buy signals in trailing 90d).

    High count = multiple insiders buying = stronger signal.
    """
    d90 = (datetime.strptime(signal_date, '%Y-%m-%d') - timedelta(days=90)).strftime('%Y-%m-%d')
    count = conn.execute(
        "SELECT COUNT(*) as cnt FROM signals "
        "WHERE ticker=? AND signal_date BETWEEN ? AND ? AND id != ?",
        (ticker, d90, signal_date, signal_id)
    ).fetchone()['cnt']
    import math
    return round(math.log1p(count), 4)


def _compute_sector_avg_car(conn: sqlite3.Connection, sector: str, before_date: str = None) -> float | None:
    """Average car_30d for signals in the same sector, point-in-time.

    POINT-IN-TIME: only uses signals whose 30-day outcome was knowable
    at the current signal's date. A signal's outcome is knowable ~45
    calendar days after its signal_date (30 trading days + buffer).
    So we filter: signal_date < current_signal_date - 45 days.
    """
    if not sector:
        return None
    if before_date:
        # 45 calendar days ≈ 30 trading days + weekends/holidays buffer
        row = conn.execute(
            "SELECT AVG(car_30d) as avg FROM signals "
            "WHERE sector=? AND outcome_30d_filled=1 AND car_30d IS NOT NULL "
            "AND signal_date < date(?, '-45 days')",
            (sector, before_date)
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT AVG(car_30d) as avg FROM signals "
            "WHERE sector=? AND outcome_30d_filled=1 AND car_30d IS NOT NULL",
            (sector,)
        ).fetchone()
    if row and row['avg'] is not None:
        return round(row['avg'], 6)
    return None


def _compute_vix_regime_interaction(vix: float, has_convergence: int) -> float | None:
    """VIX × (1 + has_convergence). High VIX + convergence = strongest contrarian signal."""
    if vix is None:
        return None
    return round(vix * (1 + (has_convergence or 0)), 4)


def compute_feature_stats(conn: sqlite3.Connection) -> dict:
    """Compute per-feature hit rates and average CARs. Updates feature_stats table.

    CARs are winsorized (1st/99th percentile, hard bounds ±300%) before computing
    averages to prevent outliers from skewing feature importance.
    """
    now = datetime.now(tz=timezone.utc).isoformat()

    # Get all signals with 30d outcomes
    rows = conn.execute(
        "SELECT * FROM signals WHERE outcome_30d_filled = 1 AND car_30d IS NOT NULL"
    ).fetchall()

    if not rows:
        log.warning("No signals with 30d outcomes — cannot compute feature stats.")
        return {}

    # Compute winsorization bounds (1st/99th percentile with hard bounds)
    all_cars_30d = sorted(r['car_30d'] for r in rows)
    n = len(all_cars_30d)
    p1_idx = max(0, int(n * 0.01))
    p99_idx = min(n - 1, int(n * 0.99))
    car_lower = max(all_cars_30d[p1_idx], -1.0)
    car_upper = min(all_cars_30d[p99_idx], 3.0)

    def _winsorize_car(val, lower=car_lower, upper=car_upper):
        if val is None:
            return None
        return max(lower, min(upper, val))

    # Define feature extractors: (feature_name, extractor_fn)
    feature_extractors = [
        ('source', lambda r: r['source']),
        ('has_convergence', lambda r: str(r['has_convergence'])),
        ('trade_size_range', lambda r: r['trade_size_range'] or 'n/a'),
        ('cluster_7d', lambda r: '3+' if (r['same_ticker_signals_7d'] or 0) >= 3 else
                                  '2' if (r['same_ticker_signals_7d'] or 0) >= 2 else '1'),
        ('cluster_30d', lambda r: '3+' if (r['same_ticker_signals_30d'] or 0) >= 3 else
                                   '2' if (r['same_ticker_signals_30d'] or 0) >= 2 else '1'),
        ('party', lambda r: r['party'] or 'n/a'),
        ('chamber', lambda r: r['chamber'] or 'n/a'),
        # Person track record features
        ('person_experience', lambda r: 'veteran_10+' if (r['person_trade_count'] or 0) >= 10 else
                                        'experienced_5+' if (r['person_trade_count'] or 0) >= 5 else
                                        'some_2+' if (r['person_trade_count'] or 0) >= 2 else 'first_trade'),
        ('person_accuracy', lambda r: 'n/a' if r['person_hit_rate_30d'] is None else
                                      'strong_70+' if r['person_hit_rate_30d'] >= 0.7 else
                                      'good_55+' if r['person_hit_rate_30d'] >= 0.55 else
                                      'weak_<55'),
        ('relative_position', lambda r: 'n/a' if r['relative_position_size'] is None else
                                        'oversized_2x+' if r['relative_position_size'] >= 2.0 else
                                        'large_1.5x+' if r['relative_position_size'] >= 1.5 else
                                        'typical' if r['relative_position_size'] >= 0.5 else
                                        'small'),
        # ALE v2 features
        ('convergence_tier', lambda r: str(r['convergence_tier'] or 0)),
        ('sector', lambda r: r['sector'] or 'n/a'),
        ('cluster_velocity', lambda r: r['cluster_velocity'] or 'n/a'),
        ('disclosure_delay', lambda r: 'n/a' if r['disclosure_delay'] is None else
                                        'urgent' if r['disclosure_delay'] < 7 else
                                        'normal' if r['disclosure_delay'] < 30 else
                                        'slow' if r['disclosure_delay'] < 45 else
                                        'late'),
        ('insider_role', lambda r: _normalize_role(r['insider_role']) if r['insider_role'] else 'n/a'),
        ('trade_pattern', lambda r: r['trade_pattern'] or 'n/a'),
        ('price_proximity', lambda r: 'n/a' if r['price_proximity_52wk'] is None else
                                       'near_low' if r['price_proximity_52wk'] < 0.2 else
                                       'lower_half' if r['price_proximity_52wk'] < 0.5 else
                                       'upper_half' if r['price_proximity_52wk'] < 0.8 else
                                       'near_high'),
        ('market_cap', lambda r: r['market_cap_bucket'] or 'n/a'),
        # Market context features
        ('vix_regime', lambda r: 'n/a' if r['vix_at_signal'] is None else
                                  'high_fear' if r['vix_at_signal'] >= 30 else
                                  'elevated' if r['vix_at_signal'] >= 20 else
                                  'calm'),
        ('yield_curve', lambda r: 'n/a' if r['yield_curve_at_signal'] is None else
                                   'inverted' if r['yield_curve_at_signal'] < 0 else
                                   'flat' if r['yield_curve_at_signal'] < 0.5 else
                                   'normal'),
        ('credit_stress', lambda r: 'n/a' if r['credit_spread_at_signal'] is None else
                                     'high_stress' if r['credit_spread_at_signal'] >= 5 else
                                     'elevated' if r['credit_spread_at_signal'] >= 4 else
                                     'normal'),
        # Catalyst proximity features
        ('earnings_proximity', lambda r: 'pre_earnings_7d' if 0 < (r['days_to_earnings'] or 999) <= 7 else
                                          'pre_earnings_30d' if 0 < (r['days_to_earnings'] or 999) <= 30 else
                                          'post_earnings_7d' if -7 <= (r['days_to_earnings'] or 999) < 0 else
                                          'no_catalyst' if (r['days_to_earnings'] or 999) >= 999 else
                                          'distant' if r['days_to_earnings'] is not None else 'n/a'),
        ('catalyst_proximity', lambda r: 'pre_catalyst_7d' if 0 < (r['days_to_catalyst'] or 999) <= 7 else
                                          'pre_catalyst_30d' if 0 < (r['days_to_catalyst'] or 999) <= 30 else
                                          'post_catalyst_7d' if -7 <= (r['days_to_catalyst'] or 999) < 0 else
                                          'no_catalyst' if (r['days_to_catalyst'] or 999) >= 999 else
                                          'distant' if r['days_to_catalyst'] is not None else 'n/a'),
        # Trade size bucket (for feature analysis — raw value already in ML)
        ('trade_size_bucket', lambda r: 'xl_15' if (r['trade_size_points'] or 0) >= 15 else
                                         'large_10_14' if (r['trade_size_points'] or 0) >= 10 else
                                         'medium_6_9' if (r['trade_size_points'] or 0) >= 6 else
                                         'small' if r['trade_size_points'] is not None else 'n/a'),
    ]

    stats = {}
    for feature_name, extractor in feature_extractors:
        buckets = defaultdict(list)
        for r in rows:
            val = extractor(r)
            if val and val != 'n/a':
                buckets[val].append(_winsorize_car(r['car_30d']))

        for value, cars in buckets.items():
            cars = [c for c in cars if c is not None]
            if len(cars) < 3:  # minimum observations
                continue
            n = len(cars)
            hit_rate = sum(1 for c in cars if c > 0) / n
            avg_car = sum(cars) / n

            # Also compute longer-horizon stats if available (winsorized)
            extra_cars = {}
            for horizon in ('90d', '180d', '365d'):
                col = f'car_{horizon}'
                vals = [_winsorize_car(r[col]) for r in rows
                        if extractor(r) == value and r[col] is not None]
                vals = [v for v in vals if v is not None]
                extra_cars[horizon] = round(sum(vals) / len(vals), 6) if vals else None

            key = (feature_name, value)
            stats[key] = {
                'n': n,
                'positive_rate_30d': round(hit_rate, 4),
                'avg_car_30d': round(avg_car, 6),
                'avg_car_90d': extra_cars['90d'],
                'avg_car_180d': extra_cars['180d'],
                'avg_car_365d': extra_cars['365d'],
            }

            conn.execute(
                """INSERT OR REPLACE INTO feature_stats
                   (feature_name, feature_value, n_observations, positive_rate_30d,
                    avg_car_30d, avg_car_90d, avg_car_180d, avg_car_365d, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (feature_name, value, n, round(hit_rate, 4), round(avg_car, 6),
                 extra_cars['90d'], extra_cars['180d'], extra_cars['365d'], now)
            )

    conn.commit()
    log.debug(f"Feature stats: {len(stats)} feature-value pairs from {len(rows)} signals")
    return stats


def generate_weights_from_stats(conn: sqlite3.Connection) -> dict:
    """Derive scoring weights from feature analysis. Returns weights dict."""
    import copy
    weights = copy.deepcopy(DEFAULT_WEIGHTS)
    now = datetime.now(tz=timezone.utc)

    # Read feature stats
    rows = conn.execute(
        "SELECT * FROM feature_stats WHERE n_observations >= 5 ORDER BY avg_car_30d DESC"
    ).fetchall()

    if not rows:
        log.info("No feature stats with enough observations — using default weights.")
        return weights

    # Compute overall baseline hit rate (needed for comparisons below)
    all_30d = conn.execute(
        "SELECT car_30d FROM signals WHERE outcome_30d_filled=1 AND car_30d IS NOT NULL"
    ).fetchall()
    overall_hit = sum(1 for r in all_30d if r['car_30d'] > 0) / len(all_30d) if all_30d else 0
    overall_avg = sum(r['car_30d'] for r in all_30d) / len(all_30d) if all_30d else 0

    # Adjust convergence boost based on convergence hit rate vs baseline
    convergence_row = conn.execute(
        "SELECT * FROM feature_stats WHERE feature_name='has_convergence' AND feature_value='1'"
    ).fetchone()
    baseline_row = conn.execute(
        "SELECT * FROM feature_stats WHERE feature_name='has_convergence' AND feature_value='0'"
    ).fetchone()

    if convergence_row and convergence_row['n_observations'] >= 5:
        conv_hit = convergence_row['positive_rate_30d'] or 0.5
        base_hit = baseline_row['positive_rate_30d'] if baseline_row else 0.5
        # Scale convergence boost: higher if convergence signals outperform
        if conv_hit > base_hit:
            # Boost between 15-30 based on outperformance
            outperformance = conv_hit - base_hit
            weights['convergence_boost'] = min(30, max(15, int(20 + outperformance * 50)))
        else:
            weights['convergence_boost'] = 15  # minimum

    # Adjust cluster bonus based on cluster performance
    cluster_3plus = conn.execute(
        "SELECT * FROM feature_stats WHERE feature_name='cluster_30d' AND feature_value='3+'"
    ).fetchone()
    if cluster_3plus and cluster_3plus['n_observations'] >= 5:
        cluster_hit = cluster_3plus['positive_rate_30d'] or 0.5
        if cluster_hit > 0.6:
            weights['congress_cluster_bonus'] = min(20, max(10, int(cluster_hit * 25)))

    # Adjust trade size tiers based on performance by size bucket
    size_stats = conn.execute(
        "SELECT * FROM feature_stats WHERE feature_name='trade_size_range' ORDER BY avg_car_30d DESC"
    ).fetchall()
    if size_stats:
        # If large trades outperform, increase their weight
        for row in size_stats:
            val = row['feature_value']
            if row['avg_car_30d'] and row['avg_car_30d'] > 0.01:
                if '$1,000,001' in val:
                    weights['congress_tiers']['xl'] = min(20, max(12, int(row['positive_rate_30d'] * 25)))
                elif '$250,001' in val or '$500,001' in val:
                    weights['congress_tiers']['significant'] = min(15, max(8, int(row['positive_rate_30d'] * 18)))

    # Adjust person track record bonus based on performance
    veteran_row = conn.execute(
        "SELECT * FROM feature_stats WHERE feature_name='person_accuracy' AND feature_value='strong_70+'"
    ).fetchone()
    if veteran_row and veteran_row['n_observations'] >= 5:
        vet_hit = veteran_row['positive_rate_30d'] or 0.5
        base_hit = overall_hit if overall_hit > 0 else 0.5
        if vet_hit > base_hit:
            outperf = vet_hit - base_hit
            weights['person_track_record_bonus'] = min(15, max(5, int(10 + outperf * 30)))
        else:
            weights['person_track_record_bonus'] = 5
    else:
        weights['person_track_record_bonus'] = 5  # default until enough data

    # Adjust oversized position bonus
    oversized_row = conn.execute(
        "SELECT * FROM feature_stats WHERE feature_name='relative_position' AND feature_value='oversized_2x+'"
    ).fetchone()
    if oversized_row and oversized_row['n_observations'] >= 5:
        over_hit = oversized_row['positive_rate_30d'] or 0.5
        if over_hit > 0.55:
            weights['oversized_position_bonus'] = min(10, max(3, int(over_hit * 12)))

    # Find optimal threshold using current data
    scored = conn.execute(
        "SELECT total_score, car_30d FROM signals WHERE outcome_30d_filled=1 AND car_30d IS NOT NULL AND total_score IS NOT NULL"
    ).fetchall()
    if scored:
        best_threshold = 5
        best_combined = -999
        for threshold in [5, 10, 15, 20, 25, 30, 40, 50, 65]:
            above = [r for r in scored if r['total_score'] >= threshold]
            if len(above) < 5:
                continue
            cars = [r['car_30d'] for r in above]
            hit_rate = sum(1 for c in cars if c > 0) / len(cars)
            avg_car = sum(cars) / len(cars)
            combined = avg_car * hit_rate
            if combined > best_combined:
                best_combined = combined
                best_threshold = threshold
        weights['_optimal_threshold'] = best_threshold

    # Record in weight history
    conn.execute(
        """INSERT INTO weight_history (date, weights_json, n_signals, hit_rate_30d, avg_car_30d, method)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (now.strftime('%Y-%m-%d'), json.dumps(weights), len(all_30d),
         round(overall_hit, 4), round(overall_avg, 6), 'feature_importance')
    )
    conn.commit()

    log.debug(f"Weights updated via feature_importance (n={len(all_30d)}, hit_rate={overall_hit:.1%}, avg_car={overall_avg:.4f})")
    return weights


# ── Dashboard & Summary ──────────────────────────────────────────────────────

def generate_dashboard(conn: sqlite3.Connection, ml_result=None) -> dict:
    """Generate comprehensive ALE status dashboard."""
    now = datetime.now(tz=timezone.utc).isoformat()

    # Database stats
    total = conn.execute("SELECT COUNT(*) as cnt FROM signals").fetchone()['cnt']
    congress_cnt = conn.execute("SELECT COUNT(*) as cnt FROM signals WHERE source='congress'").fetchone()['cnt']
    edgar_cnt = conn.execute("SELECT COUNT(*) as cnt FROM signals WHERE source='edgar'").fetchone()['cnt']
    with_price = conn.execute("SELECT COUNT(*) as cnt FROM signals WHERE price_at_signal IS NOT NULL").fetchone()['cnt']
    filled_5d = conn.execute("SELECT COUNT(*) as cnt FROM signals WHERE outcome_5d_filled=1").fetchone()['cnt']
    filled_30d = conn.execute("SELECT COUNT(*) as cnt FROM signals WHERE outcome_30d_filled=1").fetchone()['cnt']
    filled_90d = conn.execute("SELECT COUNT(*) as cnt FROM signals WHERE outcome_90d_filled=1").fetchone()['cnt']
    filled_180d = conn.execute("SELECT COUNT(*) as cnt FROM signals WHERE outcome_180d_filled=1").fetchone()['cnt']
    filled_365d = conn.execute("SELECT COUNT(*) as cnt FROM signals WHERE outcome_365d_filled=1").fetchone()['cnt']
    unique_tickers = conn.execute("SELECT COUNT(DISTINCT ticker) as cnt FROM signals").fetchone()['cnt']

    date_range_row = conn.execute(
        "SELECT MIN(signal_date) as min_d, MAX(signal_date) as max_d FROM signals"
    ).fetchone()
    date_range = f"{date_range_row['min_d'] or '?'} to {date_range_row['max_d'] or '?'}" if date_range_row else "empty"

    # Scoring performance — all horizons
    horizon_perf = {}
    for horizon in ['5d', '30d', '90d', '180d', '365d']:
        car_col = f'car_{horizon}'
        filled_col = f'outcome_{horizon}_filled'
        all_rows = conn.execute(
            f"SELECT {car_col} FROM signals WHERE {filled_col}=1 AND {car_col} IS NOT NULL"
        ).fetchall()
        conv_rows = conn.execute(
            f"SELECT {car_col} FROM signals WHERE {filled_col}=1 AND {car_col} IS NOT NULL AND has_convergence=1"
        ).fetchall()
        if all_rows:
            hit = sum(1 for r in all_rows if r[car_col] > 0) / len(all_rows)
            avg = sum(r[car_col] for r in all_rows) / len(all_rows)
        else:
            hit, avg = 0, 0
        if conv_rows:
            c_hit = sum(1 for r in conv_rows if r[car_col] > 0) / len(conv_rows)
            c_avg = sum(r[car_col] for r in conv_rows) / len(conv_rows)
        else:
            c_hit, c_avg = 0, 0
        horizon_perf[horizon] = {
            'n': len(all_rows),
            'hit_rate': round(hit, 4),
            'avg_car': round(avg, 6),
            'conv_n': len(conv_rows),
            'conv_hit_rate': round(c_hit, 4),
            'conv_avg_car': round(c_avg, 6),
        }

    # Backward-compatible aliases
    overall_hit = horizon_perf['30d']['hit_rate']
    overall_avg = horizon_perf['30d']['avg_car']
    conv_hit = horizon_perf['30d']['conv_hit_rate']
    conv_avg = horizon_perf['30d']['conv_avg_car']
    all_30d = horizon_perf['30d']
    conv_30d_n = horizon_perf['30d']['conv_n']

    # Top and worst features
    top_features = conn.execute(
        "SELECT feature_name, feature_value, positive_rate_30d, avg_car_30d, n_observations "
        "FROM feature_stats WHERE n_observations >= 5 ORDER BY avg_car_30d DESC LIMIT 10"
    ).fetchall()
    worst_features = conn.execute(
        "SELECT feature_name, feature_value, positive_rate_30d, avg_car_30d, n_observations "
        "FROM feature_stats WHERE n_observations >= 5 ORDER BY avg_car_30d ASC LIMIT 5"
    ).fetchall()

    # Person leaderboards — top and bottom performers
    top_reps = conn.execute(
        """SELECT representative as name, COUNT(*) as trades,
           AVG(car_30d) as avg_car_30, AVG(car_90d) as avg_car_90,
           SUM(CASE WHEN car_30d > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as hit_rate
        FROM signals WHERE source='congress' AND outcome_30d_filled=1
           AND car_30d IS NOT NULL AND representative != ''
        GROUP BY representative HAVING COUNT(*) >= 3
        ORDER BY avg_car_30 DESC LIMIT 10"""
    ).fetchall()

    top_insiders = conn.execute(
        """SELECT insider_name as name, COUNT(*) as trades,
           AVG(car_30d) as avg_car_30, AVG(car_90d) as avg_car_90,
           SUM(CASE WHEN car_30d > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as hit_rate
        FROM signals WHERE source='edgar' AND outcome_30d_filled=1
           AND car_30d IS NOT NULL AND insider_name != ''
        GROUP BY insider_name HAVING COUNT(*) >= 3
        ORDER BY avg_car_30 DESC LIMIT 10"""
    ).fetchall()

    worst_reps = conn.execute(
        """SELECT representative as name, COUNT(*) as trades,
           AVG(car_30d) as avg_car_30,
           SUM(CASE WHEN car_30d > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as hit_rate
        FROM signals WHERE source='congress' AND outcome_30d_filled=1
           AND car_30d IS NOT NULL AND representative != ''
        GROUP BY representative HAVING COUNT(*) >= 3
        ORDER BY avg_car_30 ASC LIMIT 5"""
    ).fetchall()

    # Current weights (from optimal_weights.json)
    current_weights = {}
    if OPTIMAL_WEIGHTS.exists():
        try:
            w = load_json(OPTIMAL_WEIGHTS)
            current_weights = {k: v for k, v in w.items()
                              if k not in ('generated', 'n_events_total', 'stats', 'congress_tiers')}
        except Exception:
            pass

    # Weight history
    weight_hist = conn.execute(
        "SELECT date, method, hit_rate_30d, avg_car_30d, n_signals "
        "FROM weight_history ORDER BY date DESC LIMIT 10"
    ).fetchall()

    # Recent signals (last 20)
    recent = conn.execute(
        "SELECT ticker, signal_date, source, total_score, has_convergence, car_30d "
        "FROM signals ORDER BY signal_date DESC LIMIT 20"
    ).fetchall()

    dashboard = {
        "generated": now,
        "database_stats": {
            "total_signals": total,
            "congress_signals": congress_cnt,
            "edgar_signals": edgar_cnt,
            "with_price_data": with_price,
            "outcomes_filled_5d": filled_5d,
            "outcomes_filled_30d": filled_30d,
            "outcomes_filled_90d": filled_90d,
            "outcomes_filled_180d": filled_180d,
            "outcomes_filled_365d": filled_365d,
            "unique_tickers": unique_tickers,
            "date_range": date_range,
        },
        "scoring_performance": {
            "overall_hit_rate_30d": round(overall_hit, 4),
            "overall_avg_car_30d": round(overall_avg, 6),
            "convergence_hit_rate_30d": round(conv_hit, 4),
            "convergence_avg_car_30d": round(conv_avg, 6),
            "n_with_30d_outcomes": all_30d['n'],
            "n_convergence_signals": conv_30d_n,
            "by_horizon": horizon_perf,
        },
        "top_features": [
            {
                "feature": f"{r['feature_name']}={r['feature_value']}",
                "hit_rate": r['positive_rate_30d'],
                "avg_car_30d": r['avg_car_30d'],
                "n": r['n_observations'],
            } for r in top_features
        ],
        "worst_features": [
            {
                "feature": f"{r['feature_name']}={r['feature_value']}",
                "hit_rate": r['positive_rate_30d'],
                "avg_car_30d": r['avg_car_30d'],
                "n": r['n_observations'],
            } for r in worst_features
        ],
        "top_representatives": [
            {
                "name": r['name'],
                "trades": r['trades'],
                "hit_rate": round(r['hit_rate'], 3) if r['hit_rate'] else 0,
                "avg_car_30d": round(r['avg_car_30'], 6) if r['avg_car_30'] else 0,
                "avg_car_90d": round(r['avg_car_90'], 6) if r['avg_car_90'] else None,
            } for r in top_reps
        ],
        "worst_representatives": [
            {
                "name": r['name'],
                "trades": r['trades'],
                "hit_rate": round(r['hit_rate'], 3) if r['hit_rate'] else 0,
                "avg_car_30d": round(r['avg_car_30'], 6) if r['avg_car_30'] else 0,
            } for r in worst_reps
        ],
        "top_insiders": [
            {
                "name": r['name'],
                "trades": r['trades'],
                "hit_rate": round(r['hit_rate'], 3) if r['hit_rate'] else 0,
                "avg_car_30d": round(r['avg_car_30'], 6) if r['avg_car_30'] else 0,
                "avg_car_90d": round(r['avg_car_90'], 6) if r['avg_car_90'] else None,
            } for r in top_insiders
        ],
        "current_weights": current_weights,
        "weight_history": [
            {
                "date": r['date'],
                "method": r['method'],
                "hit_rate": r['hit_rate_30d'],
                "avg_car": r['avg_car_30d'],
                "n_signals": r['n_signals'],
            } for r in weight_hist
        ],
        "recent_signals": [
            {
                "ticker": r['ticker'],
                "date": r['signal_date'],
                "source": r['source'],
                "score": r['total_score'],
                "convergence": bool(r['has_convergence']),
                "car_30d": r['car_30d'],
            } for r in recent
        ],
    }

    # ML model performance (if available)
    if ml_result and ml_result.n_folds > 0:
        dashboard['ml_model_performance'] = {
            'n_folds': ml_result.n_folds,
            'oos_ic_30d': ml_result.oos_ic,
            'oos_hit_rate': ml_result.oos_hit_rate,
            'oos_avg_car': ml_result.oos_avg_car,
            'feature_importance': ml_result.feature_importance,
        }

    save_json(ALE_DASHBOARD, dashboard)
    return dashboard


def print_summary(conn: sqlite3.Connection) -> None:
    """Print human-readable status to console."""
    dashboard = generate_dashboard(conn)
    db = dashboard['database_stats']
    perf = dashboard['scoring_performance']

    print(f"\n{'='*50}")
    print(f"  ATLAS Learning Engine Status")
    print(f"{'='*50}")

    print(f"\nDatabase: {db['total_signals']:,} signals "
          f"({db['congress_signals']} congress + {db['edgar_signals']} EDGAR)")
    print(f"  With outcomes: {db['outcomes_filled_30d']:,} (30d) / "
          f"{db['outcomes_filled_90d']:,} (90d) / "
          f"{db.get('outcomes_filled_180d', 0):,} (180d) / "
          f"{db.get('outcomes_filled_365d', 0):,} (365d)")
    print(f"  Unique tickers: {db['unique_tickers']}")
    print(f"  Date range: {db['date_range']}")

    if perf['n_with_30d_outcomes'] > 0:
        print(f"\nScoring Performance:")
        by_h = perf.get('by_horizon', {})
        if by_h:
            # Table header
            print(f"  {'Horizon':<10s} {'Hit Rate':>10s} {'Avg CAR':>10s} {'Conv Hit':>10s} {'Conv CAR':>10s} {'N':>8s} {'Conv N':>8s}")
            print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
            for h in ['5d', '30d', '90d', '180d', '365d']:
                hp = by_h.get(h)
                if hp and hp['n'] > 0:
                    print(f"  {h:<10s} {hp['hit_rate']:>9.1%} {hp['avg_car']*100:>+9.2f}%"
                          f" {hp['conv_hit_rate']:>9.1%} {hp['conv_avg_car']*100:>+9.2f}%"
                          f" {hp['n']:>8,} {hp['conv_n']:>8,}")
        else:
            # Fallback to old format
            print(f"  Overall hit rate (30d): {perf['overall_hit_rate_30d']:.1%}")
            print(f"  Overall avg CAR (30d):  {perf['overall_avg_car_30d']*100:+.2f}%")
            if perf['n_convergence_signals'] > 0:
                print(f"  Convergence hit rate:   {perf['convergence_hit_rate_30d']:.1%}"
                      f"  (n={perf['n_convergence_signals']})")
                print(f"  Convergence avg CAR:    {perf['convergence_avg_car_30d']*100:+.2f}%")

    if dashboard['top_features']:
        print(f"\nTop Predictive Features:")
        for i, f in enumerate(dashboard['top_features'][:5], 1):
            hit_pct = f['hit_rate'] * 100 if f['hit_rate'] else 0
            car_pct = f['avg_car_30d'] * 100 if f['avg_car_30d'] else 0
            print(f"  {i}. {f['feature']:<30s} {hit_pct:.0f}% hit rate  "
                  f"{car_pct:+.2f}% CAR  (n={f['n']})")

    if dashboard['worst_features']:
        print(f"\nWeakest Features:")
        for f in dashboard['worst_features'][:3]:
            hit_pct = f['hit_rate'] * 100 if f['hit_rate'] else 0
            car_pct = f['avg_car_30d'] * 100 if f['avg_car_30d'] else 0
            print(f"  - {f['feature']:<30s} {hit_pct:.0f}% hit rate  "
                  f"{car_pct:+.2f}% CAR  (n={f['n']})")

    # Person leaderboards
    if dashboard.get('top_representatives'):
        print(f"\nTop Congressional Traders (by 30d CAR):")
        for i, p in enumerate(dashboard['top_representatives'][:5], 1):
            hit_pct = p['hit_rate'] * 100
            car_pct = p['avg_car_30d'] * 100
            print(f"  {i}. {p['name']:<25s} {hit_pct:.0f}% hit rate  "
                  f"{car_pct:+.2f}% CAR  ({p['trades']} trades)")

    if dashboard.get('worst_representatives'):
        print(f"\nWorst Congressional Traders:")
        for p in dashboard['worst_representatives'][:3]:
            hit_pct = p['hit_rate'] * 100
            car_pct = p['avg_car_30d'] * 100
            print(f"  - {p['name']:<25s} {hit_pct:.0f}% hit rate  "
                  f"{car_pct:+.2f}% CAR  ({p['trades']} trades)")

    if dashboard.get('top_insiders'):
        print(f"\nTop Insiders (by 30d CAR):")
        for i, p in enumerate(dashboard['top_insiders'][:5], 1):
            hit_pct = p['hit_rate'] * 100
            car_pct = p['avg_car_30d'] * 100
            print(f"  {i}. {p['name']:<25s} {hit_pct:.0f}% hit rate  "
                  f"{car_pct:+.2f}% CAR  ({p['trades']} trades)")

    if dashboard['current_weights']:
        w = dashboard['current_weights']
        print(f"\nCurrent Weights: "
              f"convergence={w.get('convergence_boost', '?')}, "
              f"cluster={w.get('congress_cluster_bonus', '?')}, "
              f"decay={w.get('decay_half_life_days', '?')}d")

    if dashboard['weight_history']:
        last = dashboard['weight_history'][0]
        print(f"Last updated: {last['date']} ({last['method']} method)")

    print(f"\nDashboard saved to: {ALE_DASHBOARD}")
    print()


# ── Analysis Report (Markdown) ────────────────────────────────────────────────

def generate_analysis_report(conn: sqlite3.Connection, ml_result=None, reg_result=None) -> Path:
    """Generate a detailed markdown analysis report for deep-dive review.

    Covers: run metadata, formula reference, data quality, scoring performance,
    feature analysis, sector breakdown, person-level stats with significance
    testing, ML diagnostics, anomalies, and auto-generated recommendations.

    Returns the Path to the written markdown file.
    """
    import math

    total = conn.execute("SELECT COUNT(*) as cnt FROM signals").fetchone()['cnt']
    if total == 0:
        ALE_ANALYSIS_REPORT.write_text("# ALE Analysis Report\n\nNo signals in database.\n")
        log.info(f"Analysis report (empty): {ALE_ANALYSIS_REPORT}")
        return ALE_ANALYSIS_REPORT

    now = datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    date_range = conn.execute(
        "SELECT MIN(signal_date) as mn, MAX(signal_date) as mx FROM signals"
    ).fetchone()
    congress_cnt = conn.execute("SELECT COUNT(*) as cnt FROM signals WHERE source='congress'").fetchone()['cnt']
    edgar_cnt = conn.execute("SELECT COUNT(*) as cnt FROM signals WHERE source='edgar'").fetchone()['cnt']
    unique_tickers = conn.execute("SELECT COUNT(DISTINCT ticker) as cnt FROM signals").fetchone()['cnt']

    lines = []
    lines.append("# ALE Analysis Report")
    lines.append(f"\n**Generated:** {now}")
    lines.append(f"**Signals:** {total:,} ({congress_cnt:,} congress + {edgar_cnt:,} EDGAR)")
    lines.append(f"**Unique Tickers:** {unique_tickers:,}")
    lines.append(f"**Date Range:** {date_range['mn'] or '?'} to {date_range['mx'] or '?'}")

    # ── Section 2: Formula Reference ──────────────────────────────────────
    lines.append("\n---\n## Formula Reference\n")
    lines.append("### CAR (Cumulative Abnormal Return)")
    lines.append("```")
    lines.append("BHAR = (1 + stock_return) / (1 + spy_return) - 1")
    lines.append("stock_return = (price_end - price_start) / price_start")
    lines.append("spy_return   = (spy_end - spy_start) / spy_start")
    lines.append("```")
    lines.append("Measured at 5 horizons: 5d, 30d, 90d, 180d, 365d\n")
    lines.append("### Winsorization")
    lines.append("- Percentile clipping: 1st and 99th percentile")
    lines.append("- Hard bounds: [-100%, +300%] (CAR_ABSOLUTE_MIN=-1.0, CAR_ABSOLUTE_MAX=3.0)\n")
    lines.append("### Convergence Scoring")
    lines.append("A signal has convergence when multiple sources (congress + insider + institutional) point at the same ticker.")
    lines.append("- Tier 0: single source only")
    lines.append("- Tier 1: two sources agree")
    lines.append("- Tier 2: three+ sources agree\n")

    # Feature definitions table
    feature_defs = [
        ('source', 'Signal origin (congress/edgar)', 'DB'),
        ('trade_size_points', 'Dollar value mapped to 0-15 point scale', 'DB'),
        ('same_ticker_signals_7d', 'Count of same-ticker signals in trailing 7 days', 'DB'),
        ('same_ticker_signals_30d', 'Count of same-ticker signals in trailing 30 days', 'DB'),
        ('has_convergence', 'Binary: multiple sources agree on ticker', 'DB'),
        ('convergence_tier', 'Convergence level (0/1/2)', 'DB'),
        ('person_trade_count', 'Total historical trades by this person', 'DB'),
        ('person_hit_rate_30d', 'Person\'s 30d hit rate on prior trades', 'DB'),
        ('relative_position_size', 'Trade size relative to person\'s typical', 'DB'),
        ('insider_role', 'Role: CEO/CFO/Director/10% Owner/Other', 'EDGAR'),
        ('sector', 'GICS sector from FMP/yfinance', 'FMP'),
        ('price_proximity_52wk', '0.0 (at low) to 1.0 (at high)', 'Price'),
        ('market_cap_bucket', 'mega/large/mid/small/micro', 'FMP'),
        ('cluster_velocity', 'Speed of signal clustering (fast/medium/slow)', 'DB'),
        ('trade_pattern', 'routine vs opportunistic (3yr month analysis)', 'DB'),
        ('disclosure_delay', 'Days between trade and filing', 'EDGAR'),
        ('vix_at_signal', 'VIX index on signal date', 'FRED'),
        ('yield_curve_at_signal', '10Y-2Y Treasury spread', 'FRED'),
        ('credit_spread_at_signal', 'BAA-AAA corporate bond spread', 'FRED'),
        ('days_to_earnings', 'Calendar days to next earnings', 'FMP'),
        ('days_to_catalyst', 'Calendar days to next catalyst event', 'FMP'),
        ('momentum_1m', '1-month price return before signal', 'Price'),
        ('momentum_3m', '3-month price return before signal', 'Price'),
        ('momentum_6m', '6-month price return before signal', 'Price'),
        ('volume_spike', 'Signal-date volume / 20-day avg volume', 'Price'),
        ('insider_buy_ratio_90d', 'log(1 + same-ticker buy signals in 90d)', 'DB'),
        ('sector_avg_car', 'Historical avg 30d CAR for this sector', 'DB'),
        ('vix_regime_interaction', 'VIX × (1 + has_convergence)', 'Derived'),
    ]

    lines.append("### Feature Definitions (28 ML features)\n")
    lines.append("| Feature | Description | Source | Fill % |")
    lines.append("|---------|-------------|--------|--------|")
    for feat_name, desc, source in feature_defs:
        try:
            non_null = conn.execute(
                f"SELECT COUNT(*) as cnt FROM signals WHERE [{feat_name}] IS NOT NULL"
            ).fetchone()['cnt']
            fill_pct = non_null / total * 100 if total else 0
        except Exception:
            fill_pct = 0.0
        flag = " **LOW**" if fill_pct < 30 else ""
        lines.append(f"| `{feat_name}` | {desc} | {source} | {fill_pct:.1f}%{flag} |")

    # ── Section 3: Data Quality ───────────────────────────────────────────
    lines.append("\n---\n## Data Quality\n")
    lines.append("### Outcome Fill Rates\n")
    lines.append("| Horizon | Filled | With CAR | Fill % | CAR % | Gap |")
    lines.append("|---------|--------|----------|--------|-------|-----|")
    for h in ['5d', '30d', '90d', '180d', '365d']:
        filled = conn.execute(
            f"SELECT COUNT(*) as cnt FROM signals WHERE outcome_{h}_filled=1"
        ).fetchone()['cnt']
        with_car = conn.execute(
            f"SELECT COUNT(*) as cnt FROM signals WHERE outcome_{h}_filled=1 AND car_{h} IS NOT NULL"
        ).fetchone()['cnt']
        fill_pct = filled / total * 100 if total else 0
        car_pct = with_car / total * 100 if total else 0
        gap = filled - with_car
        gap_flag = " **GAP**" if gap > 0 else ""
        lines.append(f"| {h} | {filled:,} | {with_car:,} | {fill_pct:.1f}% | {car_pct:.1f}% | {gap}{gap_flag} |")

    # CAR distribution stats
    lines.append("\n### CAR Distribution (30d)\n")
    car_stats = conn.execute(
        """SELECT COUNT(*) as n, AVG(car_30d) as avg, MIN(car_30d) as mn, MAX(car_30d) as mx,
           SUM(CASE WHEN car_30d > 0 THEN 1 ELSE 0 END) as positive,
           SUM(CASE WHEN ABS(car_30d) > 2.0 THEN 1 ELSE 0 END) as extreme
        FROM signals WHERE car_30d IS NOT NULL"""
    ).fetchone()
    skew = None
    if car_stats['n'] > 0:
        cars = [r['car_30d'] for r in conn.execute(
            "SELECT car_30d FROM signals WHERE car_30d IS NOT NULL ORDER BY car_30d"
        ).fetchall()]
        n = len(cars)
        median_car = cars[n // 2]
        mean_car = sum(cars) / n
        variance = sum((c - mean_car) ** 2 for c in cars) / n
        std_car = math.sqrt(variance)
        # Skewness
        if std_car > 0:
            skew = sum((c - mean_car) ** 3 for c in cars) / (n * std_car ** 3)
        else:
            skew = 0.0
        lines.append(f"- **N:** {n:,}")
        lines.append(f"- **Mean:** {mean_car*100:+.2f}%")
        lines.append(f"- **Median:** {median_car*100:+.2f}%")
        lines.append(f"- **Std Dev:** {std_car*100:.2f}%")
        lines.append(f"- **Skewness:** {skew:.2f}")
        lines.append(f"- **Range:** [{car_stats['mn']*100:+.1f}%, {car_stats['mx']*100:+.1f}%]")
        lines.append(f"- **Extreme (|CAR|>200%):** {car_stats['extreme']}")
    else:
        lines.append("No CAR data available.")

    # Price coverage
    lines.append("\n### Price Coverage\n")
    ticker_rows = conn.execute("SELECT DISTINCT ticker FROM signals").fetchall()
    total_tickers = len(ticker_rows)
    tickers_with_files = sum(
        1 for r in ticker_rows if (PRICE_HISTORY_DIR / f"{r['ticker']}.json").exists()
    )
    coverage_pct = tickers_with_files / total_tickers * 100 if total_tickers else 0
    spy_exists = (PRICE_HISTORY_DIR / "SPY.json").exists()
    lines.append(f"- **Tickers with price files:** {tickers_with_files}/{total_tickers} ({coverage_pct:.0f}%)")
    lines.append(f"- **SPY benchmark file:** {'Yes' if spy_exists else '**MISSING**'}")

    # ── Section 4: Scoring Performance ────────────────────────────────────
    lines.append("\n---\n## Scoring Performance\n")
    lines.append("### Hit Rates & CAR by Horizon\n")
    lines.append("| Horizon | All Hit% | All CAR% | Conv Hit% | Conv CAR% | N (all) | N (conv) |")
    lines.append("|---------|----------|----------|-----------|-----------|---------|----------|")
    for h in ['5d', '30d', '90d', '180d', '365d']:
        car_col = f'car_{h}'
        filled_col = f'outcome_{h}_filled'
        all_rows = conn.execute(
            f"SELECT {car_col} FROM signals WHERE {filled_col}=1 AND {car_col} IS NOT NULL"
        ).fetchall()
        conv_rows = conn.execute(
            f"SELECT {car_col} FROM signals WHERE {filled_col}=1 AND {car_col} IS NOT NULL AND has_convergence=1"
        ).fetchall()
        if all_rows:
            a_hit = sum(1 for r in all_rows if r[car_col] > 0) / len(all_rows) * 100
            a_car = sum(r[car_col] for r in all_rows) / len(all_rows) * 100
        else:
            a_hit, a_car = 0, 0
        if conv_rows:
            c_hit = sum(1 for r in conv_rows if r[car_col] > 0) / len(conv_rows) * 100
            c_car = sum(r[car_col] for r in conv_rows) / len(conv_rows) * 100
        else:
            c_hit, c_car = 0, 0
        lines.append(f"| {h} | {a_hit:.1f}% | {a_car:+.2f}% | {c_hit:.1f}% | {c_car:+.2f}% | {len(all_rows):,} | {len(conv_rows):,} |")

    # Convergence edge summary
    lines.append("\n### Convergence Edge\n")
    for h in ['30d', '90d', '180d']:
        car_col = f'car_{h}'
        filled_col = f'outcome_{h}_filled'
        c_avg = conn.execute(
            f"SELECT AVG({car_col}) as v FROM signals WHERE {filled_col}=1 AND {car_col} IS NOT NULL AND has_convergence=1"
        ).fetchone()['v']
        nc_avg = conn.execute(
            f"SELECT AVG({car_col}) as v FROM signals WHERE {filled_col}=1 AND {car_col} IS NOT NULL AND (has_convergence=0 OR has_convergence IS NULL)"
        ).fetchone()['v']
        edge = ((c_avg or 0) - (nc_avg or 0)) * 100
        lines.append(f"- **{h}:** Convergence {(c_avg or 0)*100:+.2f}% vs Non-convergence {(nc_avg or 0)*100:+.2f}% = **{edge:+.2f}% edge**")

    # Source breakdown
    lines.append("\n### Source Breakdown\n")
    lines.append("| Source | Signals | With Price | With CAR (30d) | Hit Rate | Avg CAR |")
    lines.append("|--------|---------|------------|----------------|----------|---------|")
    for source in ['congress', 'edgar']:
        cnt = conn.execute("SELECT COUNT(*) as cnt FROM signals WHERE source=?", (source,)).fetchone()['cnt']
        wp = conn.execute("SELECT COUNT(*) as cnt FROM signals WHERE source=? AND price_at_signal IS NOT NULL", (source,)).fetchone()['cnt']
        wc = conn.execute("SELECT COUNT(*) as cnt FROM signals WHERE source=? AND car_30d IS NOT NULL", (source,)).fetchone()['cnt']
        rows_src = conn.execute(
            "SELECT car_30d FROM signals WHERE source=? AND outcome_30d_filled=1 AND car_30d IS NOT NULL", (source,)
        ).fetchall()
        if rows_src:
            hit = sum(1 for r in rows_src if r['car_30d'] > 0) / len(rows_src) * 100
            avg = sum(r['car_30d'] for r in rows_src) / len(rows_src) * 100
        else:
            hit, avg = 0, 0
        lines.append(f"| {source} | {cnt:,} | {wp:,} | {wc:,} | {hit:.1f}% | {avg:+.2f}% |")

    # Score-band analysis
    lines.append("\n### Score-Band Analysis (total_score quintiles vs CAR)\n")
    scored = conn.execute(
        "SELECT total_score, car_30d FROM signals WHERE outcome_30d_filled=1 AND car_30d IS NOT NULL AND total_score IS NOT NULL"
    ).fetchall()
    if scored:
        scores = sorted(r['total_score'] for r in scored)
        n_scored = len(scores)
        quintile_bounds = [scores[min(int(n_scored * q), n_scored - 1)] for q in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]] if n_scored >= 5 else []
        if len(quintile_bounds) == 6:
            lines.append("| Quintile | Score Range | N | Hit Rate | Avg CAR |")
            lines.append("|----------|------------|---|----------|---------|")
            for qi in range(5):
                lo = quintile_bounds[qi]
                hi = quintile_bounds[qi + 1]
                band = [r for r in scored if lo <= r['total_score'] <= hi] if qi == 4 else \
                       [r for r in scored if lo <= r['total_score'] < hi]
                if not band:
                    continue
                b_hit = sum(1 for r in band if r['car_30d'] > 0) / len(band) * 100
                b_car = sum(r['car_30d'] for r in band) / len(band) * 100
                lines.append(f"| Q{qi+1} | {lo:.0f}–{hi:.0f} | {len(band)} | {b_hit:.1f}% | {b_car:+.2f}% |")
            lines.append("\n*Higher quintile with higher CAR confirms scoring model adds value.*")
        else:
            lines.append("Insufficient scored signals for quintile analysis.")
    else:
        lines.append("No scored signals with outcomes available.")

    # ── Section 5: Feature Analysis ───────────────────────────────────────
    lines.append("\n---\n## Feature Analysis\n")
    top_features = conn.execute(
        "SELECT feature_name, feature_value, positive_rate_30d, avg_car_30d, n_observations "
        "FROM feature_stats WHERE n_observations >= 5 ORDER BY avg_car_30d DESC LIMIT 10"
    ).fetchall()
    worst_features = conn.execute(
        "SELECT feature_name, feature_value, positive_rate_30d, avg_car_30d, n_observations "
        "FROM feature_stats WHERE n_observations >= 5 ORDER BY avg_car_30d ASC LIMIT 5"
    ).fetchall()

    if top_features:
        lines.append("### Top 10 Predictive Features (by avg CAR)\n")
        lines.append("| Feature | Value | Hit Rate | Avg CAR | N |")
        lines.append("|---------|-------|----------|---------|---|")
        for r in top_features:
            hr = (r['positive_rate_30d'] or 0) * 100
            car = (r['avg_car_30d'] or 0) * 100
            lines.append(f"| {r['feature_name']} | {r['feature_value']} | {hr:.0f}% | {car:+.2f}% | {r['n_observations']} |")

    if worst_features:
        lines.append("\n### Bottom 5 Features (potential pruning candidates)\n")
        lines.append("| Feature | Value | Hit Rate | Avg CAR | N |")
        lines.append("|---------|-------|----------|---------|---|")
        for r in worst_features:
            hr = (r['positive_rate_30d'] or 0) * 100
            car = (r['avg_car_30d'] or 0) * 100
            lines.append(f"| {r['feature_name']} | {r['feature_value']} | {hr:.0f}% | {car:+.2f}% | {r['n_observations']} |")

    # ML feature importance
    if ml_result and ml_result.feature_importance:
        lines.append("\n### ML Feature Importance (Classification)\n")
        lines.append("| Rank | Feature | Importance |")
        lines.append("|------|---------|------------|")
        for rank, (name, imp) in enumerate(ml_result.feature_importance.items(), 1):
            flag = " **(low)**" if imp < 0.005 else ""
            lines.append(f"| {rank} | {name} | {imp:.4f}{flag} |")

    if reg_result and reg_result.feature_importance:
        lines.append("\n### ML Feature Importance (Regression)\n")
        lines.append("| Rank | Feature | Importance |")
        lines.append("|------|---------|------------|")
        for rank, (name, imp) in enumerate(reg_result.feature_importance.items(), 1):
            flag = " **(low)**" if imp < 0.005 else ""
            lines.append(f"| {rank} | {name} | {imp:.4f}{flag} |")

    # ── Section 6: Sector Breakdown ───────────────────────────────────────
    lines.append("\n---\n## Sector Breakdown\n")
    sector_rows = conn.execute(
        """SELECT sector, COUNT(*) as cnt,
           SUM(CASE WHEN car_30d > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as hit_rate,
           AVG(car_30d) as avg_car
        FROM signals WHERE outcome_30d_filled=1 AND car_30d IS NOT NULL
           AND sector IS NOT NULL AND sector != ''
        GROUP BY sector ORDER BY cnt DESC"""
    ).fetchall()
    if sector_rows:
        lines.append("| Sector | Signals | Hit Rate | Avg CAR |")
        lines.append("|--------|---------|----------|---------|")
        total_sector_signals = sum(r['cnt'] for r in sector_rows)
        for r in sector_rows:
            hr = (r['hit_rate'] or 0) * 100
            car = (r['avg_car'] or 0) * 100
            lines.append(f"| {r['sector']} | {r['cnt']:,} | {hr:.1f}% | {car:+.2f}% |")

        # Concentration
        top3 = sum(r['cnt'] for r in sector_rows[:3])
        conc_pct = top3 / total_sector_signals * 100 if total_sector_signals else 0
        lines.append(f"\n**Sector concentration:** Top 3 sectors = {conc_pct:.0f}% of signals")

    # Market cap bucket performance
    lines.append("\n### Market Cap Bucket Performance\n")
    cap_rows = conn.execute(
        """SELECT market_cap_bucket, COUNT(*) as cnt,
           SUM(CASE WHEN car_30d > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as hit_rate,
           AVG(car_30d) as avg_car
        FROM signals WHERE outcome_30d_filled=1 AND car_30d IS NOT NULL
           AND market_cap_bucket IS NOT NULL AND market_cap_bucket != ''
        GROUP BY market_cap_bucket ORDER BY cnt DESC"""
    ).fetchall()
    if cap_rows:
        lines.append("| Bucket | Signals | Hit Rate | Avg CAR |")
        lines.append("|--------|---------|----------|---------|")
        for r in cap_rows:
            hr = (r['hit_rate'] or 0) * 100
            car = (r['avg_car'] or 0) * 100
            lines.append(f"| {r['market_cap_bucket']} | {r['cnt']:,} | {hr:.1f}% | {car:+.2f}% |")
    else:
        lines.append("No market cap data available.")

    # ── Section 7: Person-Level Stats ─────────────────────────────────────
    lines.append("\n---\n## Person-Level Stats\n")

    # Congressional traders
    lines.append("### Congressional Traders (min 3 trades)\n")
    reps = conn.execute(
        """SELECT representative as name, party, COUNT(*) as trades,
           AVG(car_30d) as avg_car_30, AVG(car_90d) as avg_car_90,
           SUM(CASE WHEN car_30d > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as hit_rate
        FROM signals WHERE source='congress' AND outcome_30d_filled=1
           AND car_30d IS NOT NULL AND representative IS NOT NULL AND representative != ''
        GROUP BY representative HAVING COUNT(*) >= 3
        ORDER BY avg_car_30 DESC"""
    ).fetchall()
    if reps:
        lines.append("| Name | Party | Trades | Hit Rate | 30d CAR | 90d CAR | Significance |")
        lines.append("|------|-------|--------|----------|---------|---------|-------------|")
        for r in reps:
            hr = (r['hit_rate'] or 0)
            car30 = (r['avg_car_30'] or 0) * 100
            car90 = (r['avg_car_90'] or 0) * 100 if r['avg_car_90'] else 0
            n = r['trades']
            # Binomial significance: hit_rate > 0.5 + 1.96*sqrt(0.25/n)
            threshold = 0.5 + 1.96 * math.sqrt(0.25 / n) if n >= 10 else 999
            if n >= 10 and hr > threshold:
                sig = "**Reliable**"
            elif n >= 10 and hr < (0.5 - 1.96 * math.sqrt(0.25 / n)):
                sig = "**Fade**"
            else:
                sig = "—"
            lines.append(f"| {r['name']} | {r['party'] or '?'} | {n} | {hr*100:.0f}% | {car30:+.2f}% | {car90:+.2f}% | {sig} |")
    else:
        lines.append("No congressional traders with 3+ trades and outcomes.")

    # Insiders
    lines.append("\n### Insider Traders (min 3 trades)\n")
    insiders = conn.execute(
        """SELECT insider_name as name, COUNT(*) as trades,
           AVG(car_30d) as avg_car_30, AVG(car_90d) as avg_car_90,
           SUM(CASE WHEN car_30d > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as hit_rate
        FROM signals WHERE source='edgar' AND outcome_30d_filled=1
           AND car_30d IS NOT NULL AND insider_name IS NOT NULL AND insider_name != ''
        GROUP BY insider_name HAVING COUNT(*) >= 3
        ORDER BY avg_car_30 DESC"""
    ).fetchall()
    if insiders:
        lines.append("| Name | Trades | Hit Rate | 30d CAR | 90d CAR | Significance |")
        lines.append("|------|--------|----------|---------|---------|-------------|")
        for r in insiders:
            hr = (r['hit_rate'] or 0)
            car30 = (r['avg_car_30'] or 0) * 100
            car90 = (r['avg_car_90'] or 0) * 100 if r['avg_car_90'] else 0
            n = r['trades']
            threshold = 0.5 + 1.96 * math.sqrt(0.25 / n) if n >= 10 else 999
            if n >= 10 and hr > threshold:
                sig = "**Reliable**"
            elif n >= 10 and hr < (0.5 - 1.96 * math.sqrt(0.25 / n)):
                sig = "**Fade**"
            else:
                sig = "—"
            lines.append(f"| {r['name']} | {n} | {hr*100:.0f}% | {car30:+.2f}% | {car90:+.2f}% | {sig} |")
    else:
        lines.append("No insider traders with 3+ trades and outcomes.")

    # ── Section 8: ML Diagnostics ─────────────────────────────────────────
    lines.append("\n---\n## ML Diagnostics\n")
    if ml_result and ml_result.n_folds > 0:
        lines.append("### Classification (Walk-Forward)\n")
        lines.append(f"- **Folds:** {ml_result.n_folds}")
        lines.append(f"- **OOS IC:** {ml_result.oos_ic:.4f}")
        lines.append(f"- **OOS Hit Rate:** {ml_result.oos_hit_rate:.1%}")
        lines.append(f"- **OOS Avg CAR:** {ml_result.oos_avg_car*100:+.2f}%\n")
        if ml_result.folds:
            lines.append("| Fold | Test Period | N Train | N Test | IC | Hit Rate |")
            lines.append("|------|------------|---------|--------|-----|----------|")
            for i, f in enumerate(ml_result.folds, 1):
                lines.append(f"| {i} | {f['test_start']}→{f['test_end']} | {f['n_train']:,} | {f['n_test']} | {f['ic']:.4f} | {f['hit_rate']:.1%} |")
    else:
        lines.append("No classification ML results available. Run `--analyze` to train.\n")

    if reg_result and reg_result.n_folds > 0:
        lines.append("\n### Regression (Walk-Forward)\n")
        lines.append(f"- **Folds:** {reg_result.n_folds}")
        lines.append(f"- **OOS IC:** {reg_result.oos_ic:.4f}")
        lines.append(f"- **OOS RMSE:** {reg_result.oos_rmse:.4f}")
        lines.append(f"- **OOS Avg CAR:** {reg_result.oos_avg_car*100:+.2f}%\n")
        if reg_result.folds:
            lines.append("| Fold | Test Period | N Train | N Test | IC | Hit Rate |")
            lines.append("|------|------------|---------|--------|-----|----------|")
            for i, f in enumerate(reg_result.folds, 1):
                lines.append(f"| {i} | {f['test_start']}→{f['test_end']} | {f['n_train']:,} | {f['n_test']} | {f['ic']:.4f} | {f['hit_rate']:.1%} |")

    # ── Section 9: Anomalies & Warnings ───────────────────────────────────
    lines.append("\n---\n## Anomalies & Warnings\n")
    warnings = []

    # Features with >50% NULL
    for feat_name, _, _ in feature_defs:
        try:
            non_null = conn.execute(
                f"SELECT COUNT(*) as cnt FROM signals WHERE [{feat_name}] IS NOT NULL"
            ).fetchone()['cnt']
            fill_pct = non_null / total * 100 if total else 0
            if fill_pct < 50:
                warnings.append(f"Feature `{feat_name}` has only {fill_pct:.0f}% fill rate (>50% NULL)")
        except Exception:
            pass

    # Convergence underperformance
    c30 = conn.execute(
        "SELECT AVG(car_30d) as v FROM signals WHERE outcome_30d_filled=1 AND car_30d IS NOT NULL AND has_convergence=1"
    ).fetchone()['v']
    nc30 = conn.execute(
        "SELECT AVG(car_30d) as v FROM signals WHERE outcome_30d_filled=1 AND car_30d IS NOT NULL AND (has_convergence=0 OR has_convergence IS NULL)"
    ).fetchone()['v']
    if c30 is not None and nc30 is not None and c30 < nc30:
        warnings.append(f"Convergence signals ({c30*100:+.2f}%) underperform non-convergence ({nc30*100:+.2f}%) at 30d")

    # Sectors with <5 signals
    small_sectors = conn.execute(
        """SELECT sector, COUNT(*) as cnt FROM signals
        WHERE sector IS NOT NULL AND sector != ''
        GROUP BY sector HAVING COUNT(*) < 5"""
    ).fetchall()
    for s in small_sectors:
        warnings.append(f"Sector `{s['sector']}` has only {s['cnt']} signals (insufficient data)")

    # Extreme CARs
    extreme_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM signals WHERE car_30d IS NOT NULL AND ABS(car_30d) > 2.0"
    ).fetchone()['cnt']
    if extreme_count > 0:
        warnings.append(f"{extreme_count} signals have |CAR_30d| > 200% (pre-winsorization)")

    # Tickers with no price data
    no_price = conn.execute(
        "SELECT COUNT(DISTINCT ticker) as cnt FROM signals WHERE price_at_signal IS NULL"
    ).fetchone()['cnt']
    if no_price > 0:
        warnings.append(f"{no_price} tickers have no price data attached")

    if warnings:
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("No anomalies detected.")

    # ── Section 10: Recommendations ───────────────────────────────────────
    lines.append("\n---\n## Recommendations\n")
    recs = []

    # Low-importance features
    if ml_result and ml_result.feature_importance:
        low_imp = [name for name, imp in ml_result.feature_importance.items() if imp < 0.005]
        if low_imp:
            recs.append(f"Consider removing low-importance features (<0.5%): `{'`, `'.join(low_imp)}`")

    # Sector recommendations
    if sector_rows:
        for r in sector_rows:
            hr = (r['hit_rate'] or 0) * 100
            car = (r['avg_car'] or 0) * 100
            if hr < 40 and r['cnt'] >= 10:
                recs.append(f"Sector `{r['sector']}` has {hr:.0f}% hit rate across {r['cnt']} signals — investigate or downweight")

    # Person recommendations
    if reps:
        for r in reps[:5]:
            hr = (r['hit_rate'] or 0)
            n = r['trades']
            if n >= 10 and hr > 0.65:
                recs.append(f"Rep **{r['name']}** has {hr*100:.0f}% hit rate across {n} trades — high-conviction follow")

    # Convergence horizon
    best_horizon = None
    best_edge = -999
    for h in ['30d', '90d', '180d']:
        car_col = f'car_{h}'
        filled_col = f'outcome_{h}_filled'
        c_v = conn.execute(
            f"SELECT AVG({car_col}) as v FROM signals WHERE {filled_col}=1 AND {car_col} IS NOT NULL AND has_convergence=1"
        ).fetchone()['v']
        nc_v = conn.execute(
            f"SELECT AVG({car_col}) as v FROM signals WHERE {filled_col}=1 AND {car_col} IS NOT NULL AND (has_convergence=0 OR has_convergence IS NULL)"
        ).fetchone()['v']
        edge = (c_v or 0) - (nc_v or 0)
        if edge > best_edge:
            best_edge = edge
            best_horizon = h
    if best_horizon and best_horizon != '30d' and best_edge > 0:
        recs.append(f"Convergence edge is strongest at {best_horizon} ({best_edge*100:+.2f}%) — consider emphasizing longer-horizon signals")

    # CAR skewness
    if skew is not None:
        if skew > 1.5:
            recs.append(f"30d CAR has high positive skew ({skew:.1f}) — a few big winners drive returns. Consider asymmetric scoring.")
        elif skew < -1.5:
            recs.append(f"30d CAR has high negative skew ({skew:.1f}) — big losers drag returns. Tighten risk filters.")

    if recs:
        for r in recs:
            lines.append(f"- {r}")
    else:
        lines.append("No specific recommendations at this time.")

    lines.append(f"\n---\n*Report generated by ATLAS ALE v2 — {now}*\n")

    # Write report
    report_text = '\n'.join(lines)
    ALE_ANALYSIS_REPORT.write_text(report_text)
    log.debug(f"Analysis report written to: {ALE_ANALYSIS_REPORT} ({len(lines)} lines)")
    return ALE_ANALYSIS_REPORT


# ── Diagnostics HTML ─────────────────────────────────────────────────────────

def generate_diagnostics_html(conn: sqlite3.Connection, ml_result=None, reg_result=None) -> Path:
    """Generate a self-contained HTML diagnostics dashboard with Chart.js visualizations.

    Produces 10 panels: KPI banner, horizon performance, convergence edge,
    feature importance, CAR distribution, sector heatmap, trader leaderboard,
    ML fold performance, signal timeline, and weight evolution.

    Returns the Path to the written HTML file.
    """
    from datetime import datetime, timezone

    # ── 0. Early exit if database is empty ────────────────────────────────
    total_signals = conn.execute("SELECT COUNT(*) as cnt FROM signals").fetchone()['cnt']
    if total_signals == 0:
        html_content = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>ATLAS ALE Diagnostics</title>
<style>body{background:#0f0f1e;color:#e0e0e0;font-family:sans-serif;display:flex;
align-items:center;justify-content:center;height:100vh;margin:0;}
h1{color:#00d4ff;}</style></head>
<body><div style="text-align:center"><h1>ATLAS ALE Diagnostics</h1>
<p>No data available. Run the daily pipeline first.</p></div></body></html>"""
        ALE_DIAGNOSTICS_HTML.write_text(html_content)
        log.info(f"Diagnostics HTML (empty): {ALE_DIAGNOSTICS_HTML}")
        return ALE_DIAGNOSTICS_HTML

    # ── Feature column list (mirrors ml_engine.py) ────────────────────────
    from backtest.ml_engine import FEATURE_COLUMNS

    # ── 1. KPI data ──────────────────────────────────────────────────────
    with_outcomes_30d = conn.execute(
        "SELECT COUNT(*) as cnt FROM signals WHERE outcome_30d_filled=1"
    ).fetchone()['cnt']
    unique_tickers = conn.execute(
        "SELECT COUNT(DISTINCT ticker) as cnt FROM signals"
    ).fetchone()['cnt']

    # Hit rate 30d
    hit_30d_rows = conn.execute(
        "SELECT car_30d FROM signals WHERE outcome_30d_filled=1 AND car_30d IS NOT NULL"
    ).fetchall()
    if hit_30d_rows:
        hit_rate_30d = sum(1 for r in hit_30d_rows if r['car_30d'] > 0) / len(hit_30d_rows)
        avg_car_30d = sum(r['car_30d'] for r in hit_30d_rows) / len(hit_30d_rows)
    else:
        hit_rate_30d = 0.0
        avg_car_30d = 0.0

    # Convergence edge
    conv_cars = conn.execute(
        "SELECT AVG(car_30d) as avg_car FROM signals "
        "WHERE outcome_30d_filled=1 AND car_30d IS NOT NULL AND has_convergence=1"
    ).fetchone()['avg_car']
    non_conv_cars = conn.execute(
        "SELECT AVG(car_30d) as avg_car FROM signals "
        "WHERE outcome_30d_filled=1 AND car_30d IS NOT NULL AND (has_convergence=0 OR has_convergence IS NULL)"
    ).fetchone()['avg_car']
    conv_edge = ((conv_cars or 0) - (non_conv_cars or 0))

    # ML OOS IC
    ml_oos_ic = ml_result.oos_ic if (ml_result and hasattr(ml_result, 'oos_ic')) else None

    # Feature fill rate
    fill_parts = []
    for fc in FEATURE_COLUMNS:
        try:
            row = conn.execute(
                f"SELECT COUNT(*) as total, SUM(CASE WHEN [{fc}] IS NOT NULL THEN 1 ELSE 0 END) as filled FROM signals"
            ).fetchone()
            if row['total'] > 0:
                fill_parts.append(row['filled'] / row['total'])
        except Exception:
            pass
    feature_fill_pct = (sum(fill_parts) / len(fill_parts)) if fill_parts else 0.0

    kpi_data = {
        'total_signals': total_signals,
        'with_outcomes_30d': with_outcomes_30d,
        'unique_tickers': unique_tickers,
        'hit_rate_30d': round(hit_rate_30d * 100, 1),
        'avg_car_30d': round(avg_car_30d * 100, 2),
        'convergence_edge': round(conv_edge * 100, 2),
        'ml_oos_ic': round(ml_oos_ic, 4) if ml_oos_ic is not None else None,
        'feature_fill_pct': round(feature_fill_pct * 100, 1),
    }

    # ── 2. Horizon performance data ──────────────────────────────────────
    horizon_labels = ['5d', '30d', '90d', '180d', '365d']
    all_hit_rates = []
    conv_hit_rates = []
    for h in horizon_labels:
        car_col = f'car_{h}'
        filled_col = f'outcome_{h}_filled'
        all_rows = conn.execute(
            f"SELECT {car_col} FROM signals WHERE {filled_col}=1 AND {car_col} IS NOT NULL"
        ).fetchall()
        conv_rows = conn.execute(
            f"SELECT {car_col} FROM signals WHERE {filled_col}=1 AND {car_col} IS NOT NULL AND has_convergence=1"
        ).fetchall()
        if all_rows:
            all_hit_rates.append(round(sum(1 for r in all_rows if r[car_col] > 0) / len(all_rows) * 100, 1))
        else:
            all_hit_rates.append(0)
        if conv_rows:
            conv_hit_rates.append(round(sum(1 for r in conv_rows if r[car_col] > 0) / len(conv_rows) * 100, 1))
        else:
            conv_hit_rates.append(0)

    horizon_data = {
        'labels': horizon_labels,
        'all_hit_rates': all_hit_rates,
        'conv_hit_rates': conv_hit_rates,
    }

    # ── 3. Convergence edge chart data ───────────────────────────────────
    conv_edge_data = {'labels': [], 'conv_avg': [], 'nonconv_avg': []}
    for h in ['30d', '90d']:
        car_col = f'car_{h}'
        filled_col = f'outcome_{h}_filled'
        c_avg = conn.execute(
            f"SELECT AVG({car_col}) as v FROM signals WHERE {filled_col}=1 AND {car_col} IS NOT NULL AND has_convergence=1"
        ).fetchone()['v']
        nc_avg = conn.execute(
            f"SELECT AVG({car_col}) as v FROM signals WHERE {filled_col}=1 AND {car_col} IS NOT NULL AND (has_convergence=0 OR has_convergence IS NULL)"
        ).fetchone()['v']
        conv_edge_data['labels'].append(h)
        conv_edge_data['conv_avg'].append(round((c_avg or 0) * 100, 2))
        conv_edge_data['nonconv_avg'].append(round((nc_avg or 0) * 100, 2))

    # ── 4. Feature importance data ───────────────────────────────────────
    if ml_result and hasattr(ml_result, 'feature_importance') and ml_result.feature_importance:
        sorted_fi = sorted(ml_result.feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
        feature_imp_data = {
            'labels': [x[0] for x in sorted_fi],
            'values': [round(x[1], 4) for x in sorted_fi],
            'available': True,
        }
    else:
        feature_imp_data = {'labels': [], 'values': [], 'available': False}

    # ── 5. CAR distribution data ─────────────────────────────────────────
    car_values = conn.execute(
        "SELECT car_30d FROM signals WHERE outcome_30d_filled=1 AND car_30d IS NOT NULL"
    ).fetchall()
    num_bins = 20
    bin_min, bin_max = -0.5, 0.5
    bin_width = (bin_max - bin_min) / num_bins
    bin_counts = [0] * num_bins
    for row in car_values:
        v = row['car_30d']
        # Clip to edge bins
        if v <= bin_min:
            idx = 0
        elif v >= bin_max:
            idx = num_bins - 1
        else:
            idx = int((v - bin_min) / bin_width)
            if idx >= num_bins:
                idx = num_bins - 1
        bin_counts[idx] += 1

    bin_labels = []
    bin_colors = []
    for i in range(num_bins):
        left = bin_min + i * bin_width
        right = left + bin_width
        mid = (left + right) / 2
        bin_labels.append(f"{left*100:.0f}%")
        bin_colors.append('#00e676' if mid >= 0 else '#ff5252')

    car_dist_data = {
        'labels': bin_labels,
        'counts': bin_counts,
        'colors': bin_colors,
    }

    # ── 6. Sector heatmap data ───────────────────────────────────────────
    sector_rows = conn.execute(
        """SELECT sector, COUNT(*) as cnt,
           SUM(CASE WHEN car_30d > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as hit_rate,
           AVG(car_30d) as avg_car
        FROM signals
        WHERE outcome_30d_filled=1 AND car_30d IS NOT NULL AND sector IS NOT NULL AND sector != ''
        GROUP BY sector ORDER BY cnt DESC"""
    ).fetchall()
    sector_data = [
        {
            'sector': r['sector'],
            'count': r['cnt'],
            'hit_rate': round((r['hit_rate'] or 0) * 100, 1),
            'avg_car': round((r['avg_car'] or 0) * 100, 2),
        }
        for r in sector_rows
    ]

    # ── 7. Trader leaderboard data ───────────────────────────────────────
    top_reps = conn.execute(
        """SELECT representative as name, COUNT(*) as trades,
           SUM(CASE WHEN car_30d > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as hit_rate,
           AVG(car_30d) as avg_car
        FROM signals WHERE source='congress' AND outcome_30d_filled=1
           AND car_30d IS NOT NULL AND representative IS NOT NULL AND representative != ''
        GROUP BY representative HAVING COUNT(*) >= 3
        ORDER BY avg_car DESC LIMIT 10"""
    ).fetchall()
    top_insiders = conn.execute(
        """SELECT insider_name as name, COUNT(*) as trades,
           SUM(CASE WHEN car_30d > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as hit_rate,
           AVG(car_30d) as avg_car
        FROM signals WHERE source='edgar' AND outcome_30d_filled=1
           AND car_30d IS NOT NULL AND insider_name IS NOT NULL AND insider_name != ''
        GROUP BY insider_name HAVING COUNT(*) >= 3
        ORDER BY avg_car DESC LIMIT 10"""
    ).fetchall()
    reps_data = [
        {
            'name': r['name'],
            'trades': r['trades'],
            'hit_rate': round((r['hit_rate'] or 0) * 100, 1),
            'avg_car': round((r['avg_car'] or 0) * 100, 2),
        }
        for r in top_reps
    ]
    insiders_data = [
        {
            'name': r['name'],
            'trades': r['trades'],
            'hit_rate': round((r['hit_rate'] or 0) * 100, 1),
            'avg_car': round((r['avg_car'] or 0) * 100, 2),
        }
        for r in top_insiders
    ]

    # ── 8. ML fold performance data ──────────────────────────────────────
    ml_folds_data = {'labels': [], 'cls_ic': [], 'reg_ic': []}
    if ml_result and hasattr(ml_result, 'folds') and ml_result.folds:
        for fold in ml_result.folds:
            label = fold.get('test_start', fold.get('fold', ''))
            ml_folds_data['labels'].append(str(label))
            ml_folds_data['cls_ic'].append(round(fold.get('ic', 0), 4))
    if reg_result and hasattr(reg_result, 'folds') and reg_result.folds:
        # Align regression folds with classification fold labels or add separately
        for i, fold in enumerate(reg_result.folds):
            ic_val = round(fold.get('ic', 0), 4)
            if i < len(ml_folds_data['labels']):
                ml_folds_data['reg_ic'].append(ic_val)
            else:
                label = fold.get('test_start', fold.get('fold', ''))
                ml_folds_data['labels'].append(str(label))
                ml_folds_data['cls_ic'].append(None)
                ml_folds_data['reg_ic'].append(ic_val)
    # Pad reg_ic if shorter
    while len(ml_folds_data['reg_ic']) < len(ml_folds_data['labels']):
        ml_folds_data['reg_ic'].append(None)

    # ── 9. Signal timeline data ──────────────────────────────────────────
    timeline_rows = conn.execute(
        """SELECT strftime('%Y-%m', signal_date) as month,
           COUNT(*) as cnt,
           AVG(car_30d) as avg_car
        FROM signals
        GROUP BY month ORDER BY month"""
    ).fetchall()
    timeline_data = {
        'labels': [r['month'] for r in timeline_rows],
        'counts': [r['cnt'] for r in timeline_rows],
        'avg_cars': [round((r['avg_car'] or 0) * 100, 2) for r in timeline_rows],
    }

    # ── 10. Weight evolution data ────────────────────────────────────────
    weight_rows = conn.execute(
        "SELECT date, hit_rate_30d, avg_car_30d FROM weight_history ORDER BY date ASC"
    ).fetchall()
    weight_evo_data = {
        'labels': [r['date'] for r in weight_rows],
        'hit_rates': [round((r['hit_rate_30d'] or 0) * 100, 1) for r in weight_rows],
        'avg_cars': [round((r['avg_car_30d'] or 0) * 100, 2) for r in weight_rows],
    }

    # ── Build HTML ───────────────────────────────────────────────────────
    generated_at = datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ATLAS ALE Diagnostics</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: #0f0f1e;
    color: #e0e0e0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    padding: 24px;
    min-height: 100vh;
}}
h1 {{
    color: #00d4ff;
    text-align: center;
    margin-bottom: 4px;
    font-size: 1.8rem;
}}
.subtitle {{
    text-align: center;
    color: #666;
    margin-bottom: 24px;
    font-size: 0.85rem;
}}
.grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    max-width: 1400px;
    margin: 0 auto;
}}
@media (max-width: 900px) {{
    .grid {{ grid-template-columns: 1fr; }}
}}
.card {{
    background: #1a1a2e;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #252540;
}}
.card.full-width {{
    grid-column: 1 / -1;
}}
.card h2 {{
    color: #00d4ff;
    font-size: 1rem;
    margin-bottom: 14px;
    border-bottom: 1px solid #252540;
    padding-bottom: 8px;
}}
/* KPI banner */
.kpi-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    justify-content: center;
}}
.kpi-item {{
    background: #252540;
    border-radius: 8px;
    padding: 14px 18px;
    text-align: center;
    min-width: 130px;
    flex: 1;
}}
.kpi-item .kpi-value {{
    font-size: 1.5rem;
    font-weight: 700;
    color: #00d4ff;
}}
.kpi-item .kpi-label {{
    font-size: 0.75rem;
    color: #999;
    margin-top: 4px;
}}
.kpi-item .kpi-value.positive {{ color: #00e676; }}
.kpi-item .kpi-value.negative {{ color: #ff5252; }}
/* Tables */
.data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}}
.data-table th {{
    text-align: left;
    padding: 8px 10px;
    border-bottom: 2px solid #252540;
    color: #00d4ff;
    font-weight: 600;
}}
.data-table td {{
    padding: 6px 10px;
    border-bottom: 1px solid #1f1f35;
}}
.data-table tr:hover td {{
    background: #252540;
}}
.tables-row {{
    display: flex;
    gap: 20px;
}}
.tables-row .table-half {{
    flex: 1;
    min-width: 0;
}}
.tables-row .table-half h3 {{
    color: #ffd740;
    font-size: 0.85rem;
    margin-bottom: 8px;
}}
@media (max-width: 700px) {{
    .tables-row {{ flex-direction: column; }}
}}
canvas {{
    max-height: 340px;
}}
.no-data-msg {{
    color: #666;
    text-align: center;
    padding: 40px;
    font-style: italic;
}}
</style>
</head>
<body>
<h1>ATLAS ALE Diagnostics</h1>
<p class="subtitle">Generated {generated_at} | {total_signals:,} signals</p>

<div class="grid">

<!-- Panel 1: KPI Banner -->
<div class="card full-width">
<h2>Key Performance Indicators</h2>
<div class="kpi-row" id="kpi-row"></div>
</div>

<!-- Panel 2: Horizon Performance -->
<div class="card">
<h2>Horizon Hit Rates</h2>
<canvas id="horizonChart"></canvas>
</div>

<!-- Panel 3: Convergence Edge -->
<div class="card">
<h2>Convergence Edge (Avg CAR %)</h2>
<canvas id="convEdgeChart"></canvas>
</div>

<!-- Panel 4: Feature Importance -->
<div class="card">
<h2>ML Feature Importance (Top 15)</h2>
<div id="featureImpContainer">
<canvas id="featureImpChart"></canvas>
</div>
</div>

<!-- Panel 5: CAR Distribution -->
<div class="card">
<h2>30-Day CAR Distribution</h2>
<canvas id="carDistChart"></canvas>
</div>

<!-- Panel 6: Sector Heatmap -->
<div class="card full-width">
<h2>Sector Performance</h2>
<div id="sectorTableContainer"></div>
</div>

<!-- Panel 7: Trader Leaderboard -->
<div class="card full-width">
<h2>Trader Leaderboard (min 3 trades, 30d outcomes)</h2>
<div class="tables-row" id="leaderboardContainer"></div>
</div>

<!-- Panel 8: ML Fold Performance -->
<div class="card">
<h2>ML Fold Performance (IC)</h2>
<canvas id="mlFoldsChart"></canvas>
</div>

<!-- Panel 9: Signal Timeline -->
<div class="card">
<h2>Signal Timeline</h2>
<canvas id="timelineChart"></canvas>
</div>

<!-- Panel 10: Weight Evolution -->
<div class="card full-width">
<h2>Weight Evolution Over Time</h2>
<canvas id="weightEvoChart"></canvas>
</div>

</div><!-- end grid -->

<script>
const COLORS = {{
    primary: '#00d4ff',
    success: '#00e676',
    danger: '#ff5252',
    warning: '#ffd740',
    muted: '#666',
    bg: '#1a1a2e',
    card: '#252540',
    text: '#e0e0e0'
}};

// ── Data (embedded from Python) ──
const kpiData = {json.dumps(kpi_data)};
const horizonData = {json.dumps(horizon_data)};
const convEdgeData = {json.dumps(conv_edge_data)};
const featureImpData = {json.dumps(feature_imp_data)};
const carDistData = {json.dumps(car_dist_data)};
const sectorData = {json.dumps(sector_data)};
const repsData = {json.dumps(reps_data)};
const insidersData = {json.dumps(insiders_data)};
const mlFoldsData = {json.dumps(ml_folds_data)};
const timelineData = {json.dumps(timeline_data)};
const weightEvoData = {json.dumps(weight_evo_data)};

// ── Panel 1: KPI Banner ──
(function() {{
    const row = document.getElementById('kpi-row');
    const items = [
        {{ label: 'Total Signals', value: kpiData.total_signals.toLocaleString(), cls: '' }},
        {{ label: 'With Outcomes (30d)', value: kpiData.with_outcomes_30d.toLocaleString(), cls: '' }},
        {{ label: 'Unique Tickers', value: kpiData.unique_tickers.toLocaleString(), cls: '' }},
        {{ label: '30d Hit Rate', value: kpiData.hit_rate_30d + '%', cls: kpiData.hit_rate_30d >= 50 ? 'positive' : 'negative' }},
        {{ label: '30d Avg CAR', value: (kpiData.avg_car_30d >= 0 ? '+' : '') + kpiData.avg_car_30d + '%', cls: kpiData.avg_car_30d >= 0 ? 'positive' : 'negative' }},
        {{ label: 'Convergence Edge', value: (kpiData.convergence_edge >= 0 ? '+' : '') + kpiData.convergence_edge + '%', cls: kpiData.convergence_edge >= 0 ? 'positive' : 'negative' }},
        {{ label: 'ML OOS IC', value: kpiData.ml_oos_ic !== null ? kpiData.ml_oos_ic.toFixed(4) : 'N/A', cls: '' }},
        {{ label: 'Feature Fill %', value: kpiData.feature_fill_pct + '%', cls: '' }},
    ];
    items.forEach(item => {{
        const div = document.createElement('div');
        div.className = 'kpi-item';
        div.innerHTML = '<div class="kpi-value ' + item.cls + '">' + item.value + '</div>'
                      + '<div class="kpi-label">' + item.label + '</div>';
        row.appendChild(div);
    }});
}})();

// ── Panel 2: Horizon Performance ──
new Chart(document.getElementById('horizonChart'), {{
    type: 'bar',
    data: {{
        labels: horizonData.labels,
        datasets: [
            {{
                label: 'All Signals',
                data: horizonData.all_hit_rates,
                backgroundColor: COLORS.primary + '99',
                borderColor: COLORS.primary,
                borderWidth: 1,
            }},
            {{
                label: 'Convergence Only',
                data: horizonData.conv_hit_rates,
                backgroundColor: COLORS.success + '99',
                borderColor: COLORS.success,
                borderWidth: 1,
            }},
        ],
    }},
    options: {{
        responsive: true,
        plugins: {{
            legend: {{ labels: {{ color: COLORS.text }} }},
        }},
        scales: {{
            x: {{ ticks: {{ color: COLORS.text }}, grid: {{ color: '#252540' }} }},
            y: {{
                ticks: {{ color: COLORS.text, callback: v => v + '%' }},
                grid: {{ color: '#252540' }},
                suggestedMin: 0,
                suggestedMax: 100,
            }},
        }},
    }},
}});

// ── Panel 3: Convergence Edge ──
new Chart(document.getElementById('convEdgeChart'), {{
    type: 'bar',
    data: {{
        labels: convEdgeData.labels,
        datasets: [
            {{
                label: 'Convergence',
                data: convEdgeData.conv_avg,
                backgroundColor: COLORS.success + 'cc',
                borderColor: COLORS.success,
                borderWidth: 1,
            }},
            {{
                label: 'Non-Convergence',
                data: convEdgeData.nonconv_avg,
                backgroundColor: COLORS.muted + 'cc',
                borderColor: COLORS.muted,
                borderWidth: 1,
            }},
        ],
    }},
    options: {{
        responsive: true,
        plugins: {{
            legend: {{ labels: {{ color: COLORS.text }} }},
        }},
        scales: {{
            x: {{ ticks: {{ color: COLORS.text }}, grid: {{ color: '#252540' }} }},
            y: {{
                ticks: {{ color: COLORS.text, callback: v => v + '%' }},
                grid: {{ color: '#252540' }},
            }},
        }},
    }},
}});

// ── Panel 4: Feature Importance ──
(function() {{
    if (!featureImpData.available) {{
        document.getElementById('featureImpContainer').innerHTML =
            '<p class="no-data-msg">Run --analyze to generate ML feature importance</p>';
        return;
    }}
    new Chart(document.getElementById('featureImpChart'), {{
        type: 'bar',
        data: {{
            labels: featureImpData.labels,
            datasets: [{{
                label: 'Importance',
                data: featureImpData.values,
                backgroundColor: COLORS.warning + 'cc',
                borderColor: COLORS.warning,
                borderWidth: 1,
            }}],
        }},
        options: {{
            indexAxis: 'y',
            responsive: true,
            plugins: {{
                legend: {{ display: false }},
            }},
            scales: {{
                x: {{ ticks: {{ color: COLORS.text }}, grid: {{ color: '#252540' }} }},
                y: {{ ticks: {{ color: COLORS.text, font: {{ size: 11 }} }}, grid: {{ display: false }} }},
            }},
        }},
    }});
}})();

// ── Panel 5: CAR Distribution ──
new Chart(document.getElementById('carDistChart'), {{
    type: 'bar',
    data: {{
        labels: carDistData.labels,
        datasets: [{{
            label: 'Count',
            data: carDistData.counts,
            backgroundColor: carDistData.colors,
            borderWidth: 0,
        }}],
    }},
    options: {{
        responsive: true,
        plugins: {{
            legend: {{ display: false }},
        }},
        scales: {{
            x: {{ ticks: {{ color: COLORS.text, maxRotation: 45 }}, grid: {{ display: false }} }},
            y: {{ ticks: {{ color: COLORS.text }}, grid: {{ color: '#252540' }} }},
        }},
    }},
}});

// ── Panel 6: Sector Heatmap ──
(function() {{
    const container = document.getElementById('sectorTableContainer');
    if (sectorData.length === 0) {{
        container.innerHTML = '<p class="no-data-msg">No sector data available</p>';
        return;
    }}
    let html = '<table class="data-table"><thead><tr>'
             + '<th>Sector</th><th>Signals</th><th>Hit Rate</th><th>Avg CAR</th>'
             + '</tr></thead><tbody>';
    sectorData.forEach(s => {{
        const carColor = s.avg_car >= 0
            ? 'rgba(0,230,118,' + Math.min(Math.abs(s.avg_car) / 10, 1) * 0.6 + ')'
            : 'rgba(255,82,82,' + Math.min(Math.abs(s.avg_car) / 10, 1) * 0.6 + ')';
        html += '<tr><td>' + s.sector + '</td>'
             + '<td>' + s.count + '</td>'
             + '<td>' + s.hit_rate + '%</td>'
             + '<td style="background:' + carColor + ';font-weight:600">'
             + (s.avg_car >= 0 ? '+' : '') + s.avg_car + '%</td></tr>';
    }});
    html += '</tbody></table>';
    container.innerHTML = html;
}})();

// ── Panel 7: Trader Leaderboard ──
(function() {{
    const container = document.getElementById('leaderboardContainer');
    function buildTable(title, data) {{
        if (data.length === 0) return '<div class="table-half"><h3>' + title + '</h3><p class="no-data-msg">No data</p></div>';
        let html = '<div class="table-half"><h3>' + title + '</h3>'
                 + '<table class="data-table"><thead><tr>'
                 + '<th>Name</th><th>Trades</th><th>Hit Rate</th><th>Avg CAR</th>'
                 + '</tr></thead><tbody>';
        data.forEach(d => {{
            html += '<tr><td>' + d.name + '</td>'
                 + '<td>' + d.trades + '</td>'
                 + '<td>' + d.hit_rate + '%</td>'
                 + '<td style="color:' + (d.avg_car >= 0 ? COLORS.success : COLORS.danger) + '">'
                 + (d.avg_car >= 0 ? '+' : '') + d.avg_car + '%</td></tr>';
        }});
        html += '</tbody></table></div>';
        return html;
    }}
    container.innerHTML = buildTable('Top Congress', repsData) + buildTable('Top Insiders', insidersData);
}})();

// ── Panel 8: ML Fold Performance ──
(function() {{
    if (mlFoldsData.labels.length === 0) {{
        document.getElementById('mlFoldsChart').parentElement.querySelector('h2').insertAdjacentHTML(
            'afterend', '<p class="no-data-msg">No ML fold data available. Run --analyze first.</p>');
        document.getElementById('mlFoldsChart').style.display = 'none';
        return;
    }}
    const datasets = [{{
        label: 'Classification IC',
        data: mlFoldsData.cls_ic,
        borderColor: COLORS.primary,
        backgroundColor: COLORS.primary + '33',
        tension: 0.3,
        fill: false,
        pointRadius: 4,
    }}];
    const hasReg = mlFoldsData.reg_ic.some(v => v !== null);
    if (hasReg) {{
        datasets.push({{
            label: 'Regression IC',
            data: mlFoldsData.reg_ic,
            borderColor: COLORS.warning,
            backgroundColor: COLORS.warning + '33',
            tension: 0.3,
            fill: false,
            pointRadius: 4,
        }});
    }}
    new Chart(document.getElementById('mlFoldsChart'), {{
        type: 'line',
        data: {{ labels: mlFoldsData.labels, datasets: datasets }},
        options: {{
            responsive: true,
            plugins: {{ legend: {{ labels: {{ color: COLORS.text }} }} }},
            scales: {{
                x: {{ ticks: {{ color: COLORS.text, maxRotation: 45 }}, grid: {{ color: '#252540' }} }},
                y: {{ ticks: {{ color: COLORS.text }}, grid: {{ color: '#252540' }}, title: {{ display: true, text: 'IC', color: COLORS.text }} }},
            }},
        }},
    }});
}})();

// ── Panel 9: Signal Timeline ──
(function() {{
    if (timelineData.labels.length === 0) {{
        document.getElementById('timelineChart').parentElement.querySelector('h2').insertAdjacentHTML(
            'afterend', '<p class="no-data-msg">No timeline data available</p>');
        document.getElementById('timelineChart').style.display = 'none';
        return;
    }}
    new Chart(document.getElementById('timelineChart'), {{
        type: 'bar',
        data: {{
            labels: timelineData.labels,
            datasets: [
                {{
                    label: 'Signal Count',
                    data: timelineData.counts,
                    backgroundColor: COLORS.primary + '88',
                    borderColor: COLORS.primary,
                    borderWidth: 1,
                    yAxisID: 'y',
                    order: 2,
                }},
                {{
                    label: 'Avg CAR 30d %',
                    data: timelineData.avg_cars,
                    type: 'line',
                    borderColor: COLORS.warning,
                    backgroundColor: COLORS.warning + '33',
                    tension: 0.3,
                    fill: false,
                    pointRadius: 3,
                    yAxisID: 'y1',
                    order: 1,
                }},
            ],
        }},
        options: {{
            responsive: true,
            plugins: {{ legend: {{ labels: {{ color: COLORS.text }} }} }},
            scales: {{
                x: {{ ticks: {{ color: COLORS.text, maxRotation: 45 }}, grid: {{ color: '#252540' }} }},
                y: {{
                    position: 'left',
                    ticks: {{ color: COLORS.text }},
                    grid: {{ color: '#252540' }},
                    title: {{ display: true, text: 'Count', color: COLORS.text }},
                }},
                y1: {{
                    position: 'right',
                    ticks: {{ color: COLORS.warning, callback: v => v + '%' }},
                    grid: {{ drawOnChartArea: false }},
                    title: {{ display: true, text: 'Avg CAR %', color: COLORS.warning }},
                }},
            }},
        }},
    }});
}})();

// ── Panel 10: Weight Evolution ──
(function() {{
    if (weightEvoData.labels.length === 0) {{
        document.getElementById('weightEvoChart').parentElement.querySelector('h2').insertAdjacentHTML(
            'afterend', '<p class="no-data-msg">No weight history data available</p>');
        document.getElementById('weightEvoChart').style.display = 'none';
        return;
    }}
    new Chart(document.getElementById('weightEvoChart'), {{
        type: 'line',
        data: {{
            labels: weightEvoData.labels,
            datasets: [
                {{
                    label: 'Hit Rate 30d %',
                    data: weightEvoData.hit_rates,
                    borderColor: COLORS.success,
                    backgroundColor: COLORS.success + '22',
                    tension: 0.3,
                    fill: true,
                    yAxisID: 'y',
                }},
                {{
                    label: 'Avg CAR 30d %',
                    data: weightEvoData.avg_cars,
                    borderColor: COLORS.primary,
                    backgroundColor: COLORS.primary + '22',
                    tension: 0.3,
                    fill: true,
                    yAxisID: 'y1',
                }},
            ],
        }},
        options: {{
            responsive: true,
            plugins: {{ legend: {{ labels: {{ color: COLORS.text }} }} }},
            scales: {{
                x: {{ ticks: {{ color: COLORS.text, maxRotation: 45 }}, grid: {{ color: '#252540' }} }},
                y: {{
                    position: 'left',
                    ticks: {{ color: COLORS.success, callback: v => v + '%' }},
                    grid: {{ color: '#252540' }},
                    title: {{ display: true, text: 'Hit Rate %', color: COLORS.success }},
                }},
                y1: {{
                    position: 'right',
                    ticks: {{ color: COLORS.primary, callback: v => v + '%' }},
                    grid: {{ drawOnChartArea: false }},
                    title: {{ display: true, text: 'Avg CAR %', color: COLORS.primary }},
                }},
            }},
        }},
    }});
}})();

</script>
</body>
</html>"""

    ALE_DIAGNOSTICS_HTML.write_text(html_content)
    log.debug(f"Diagnostics HTML written to: {ALE_DIAGNOSTICS_HTML}")
    return ALE_DIAGNOSTICS_HTML


# ── Export Pipeline ──────────────────────────────────────────────────────────

# Ticker → company name map for frontend display
TICKER_NAMES = {
    'AAPL': 'Apple', 'ABNB': 'Airbnb', 'AMAT': 'Applied Materials', 'AMD': 'AMD',
    'AMZN': 'Amazon', 'ANET': 'Arista Networks', 'AVGO': 'Broadcom', 'AXP': 'American Express',
    'BA': 'Boeing', 'BABA': 'Alibaba', 'BAC': 'Bank of America', 'BIIB': 'Biogen',
    'BLK': 'BlackRock', 'BMY': 'Bristol-Myers', 'BSX': 'Boston Scientific', 'C': 'Citigroup',
    'CAT': 'Caterpillar', 'CL': 'Colgate-Palmolive', 'CMCSA': 'Comcast', 'COIN': 'Coinbase',
    'COP': 'ConocoPhillips', 'COST': 'Costco', 'CRM': 'Salesforce', 'CRWD': 'CrowdStrike',
    'CSCO': 'Cisco', 'CVS': 'CVS Health', 'CVX': 'Chevron', 'DASH': 'DoorDash',
    'DE': 'Deere', 'DIS': 'Disney', 'DXCM': 'Dexcom', 'ETN': 'Eaton',
    'F': 'Ford', 'FCX': 'Freeport-McMoRan', 'FDX': 'FedEx', 'FISV': 'Fiserv',
    'GD': 'General Dynamics', 'GE': 'GE Aerospace', 'GEV': 'GE Vernova', 'GILD': 'Gilead',
    'GM': 'General Motors', 'GOOG': 'Google', 'GOOGL': 'Google', 'GS': 'Goldman Sachs',
    'HAL': 'Halliburton', 'HD': 'Home Depot', 'HON': 'Honeywell', 'HOOD': 'Robinhood',
    'HSY': 'Hershey', 'IBM': 'IBM', 'INTC': 'Intel', 'ISRG': 'Intuitive Surgical',
    'JNJ': 'Johnson & Johnson', 'JPM': 'JPMorgan Chase', 'KO': 'Coca-Cola', 'LLY': 'Eli Lilly',
    'LMT': 'Lockheed Martin', 'LOW': "Lowe's", 'LRCX': 'Lam Research', 'MA': 'Mastercard',
    'MCD': "McDonald's", 'MELI': 'MercadoLibre', 'META': 'Meta', 'MMM': '3M',
    'MO': 'Altria', 'MRK': 'Merck', 'MSFT': 'Microsoft', 'MU': 'Micron',
    'NEE': 'NextEra Energy', 'NET': 'Cloudflare', 'NFLX': 'Netflix', 'NKE': 'Nike',
    'NOC': 'Northrop Grumman', 'NVDA': 'NVIDIA', 'ORCL': 'Oracle', 'PEP': 'PepsiCo',
    'PFE': 'Pfizer', 'PG': 'Procter & Gamble', 'PGR': 'Progressive', 'PLTR': 'Palantir',
    'PM': 'Philip Morris', 'PYPL': 'PayPal', 'QCOM': 'Qualcomm', 'RTX': 'RTX (Raytheon)',
    'SBUX': 'Starbucks', 'SCHW': 'Schwab', 'SHOP': 'Shopify', 'SNOW': 'Snowflake',
    'SO': 'Southern Co', 'SPGI': 'S&P Global', 'SYK': 'Stryker', 'T': 'AT&T',
    'TGT': 'Target', 'TJX': 'TJX Cos', 'TMO': 'Thermo Fisher', 'TMUS': 'T-Mobile',
    'TSLA': 'Tesla', 'TSM': 'TSMC', 'UNH': 'UnitedHealth', 'UNP': 'Union Pacific',
    'UPS': 'UPS', 'V': 'Visa', 'VRT': 'Vertiv', 'VZ': 'Verizon',
    'WFC': 'Wells Fargo', 'WMT': 'Walmart', 'WPM': 'Wheaton Precious', 'XOM': 'ExxonMobil',
}


def _ticker_display(ticker: str) -> str:
    """Return 'Company (TICK)' if known, else just ticker."""
    name = TICKER_NAMES.get(ticker)
    return f"{name} ({ticker})" if name else ticker


# Representative → committee mapping (most active congressional traders)
# Source: house.gov/committees, senate.gov/committees  (manual, updated periodically)
REP_COMMITTEES: dict[str, str] = {
    # Armed Services
    'Tommy Tuberville': 'Armed Services', 'Jim Risch': 'Armed Services',
    'Mark Kelly': 'Armed Services', 'Tim Kaine': 'Armed Services',
    'Kevin Hern': 'Armed Services', 'Mike Garcia': 'Armed Services',
    'Pat Fallon': 'Armed Services',
    # Finance / Banking
    'Shelley Moore Capito': 'Finance', 'Dan Sullivan': 'Finance',
    'John Hickenlooper': 'Finance', 'Markwayne Mullin': 'Finance',
    'Tim Scott': 'Finance', 'Bill Hagerty': 'Finance',
    # Foreign Affairs
    'Virginia Foxx': 'Foreign Affairs', 'Michael McCaul': 'Foreign Affairs',
    'Gregory Meeks': 'Foreign Affairs',
    # Energy / Environment
    'Garret Graves': 'Energy/Enviro', 'Sheldon Whitehouse': 'Energy/Enviro',
    'Joe Manchin': 'Energy/Enviro', 'John Barrasso': 'Energy/Enviro',
    # Science & Tech
    'Ro Khanna': 'Science & Tech', 'Josh Gottheimer': 'Science & Tech',
    'Suzan DelBene': 'Science & Tech',
    # Health Policy
    'Michael Burgess': 'Health Policy', 'Larry Bucshon': 'Health Policy',
    'Bill Cassidy': 'Health Policy', 'Roger Marshall': 'Health Policy',
    # Appropriations
    'David Joyce': 'Appropriations', 'Hal Rogers': 'Appropriations',
    # Judiciary
    'Nancy Pelosi': 'Judiciary', 'Dan Goldman': 'Judiciary',
    # Commerce
    'Maria Cantwell': 'Commerce', 'Ted Cruz': 'Commerce',
}


def score_all_signals(conn: sqlite3.Connection) -> int:
    """Score ALL signals using full-sample ML models. Writes total_score to DB.

    Scoring formula (0-100):
      base      = clf_probability × 60          (ML confidence)
      magnitude = clamp(reg_car × 200, -20, 25) (predicted return bonus/penalty)
      converge  = convergence_tier × 5          (convergence bonus: 0/5/10)
      person    = clamp(person_hit_rate × 8, 0, 5) (track record bonus)
      raw       = sum of above
      source_mult = learned multiplier by source (edgar=1.0, congress≈0.65, convergence≈1.35)
      total     = clamp(raw × source_mult, 0, 100)
    """
    log.info("=== Scoring All Signals ===")

    try:
        from backtest.ml_engine import (train_full_sample, prepare_features_all,
                                        get_active_features, FEATURE_COLUMNS,
                                        CATEGORICAL_FEATURES)
    except ImportError:
        log.warning("ML dependencies not installed — skipping scoring")
        return 0

    # Apply fill-rate gate: use same active features as training
    try:
        active_feats, active_cats, fill_report = get_active_features(conn)
        FEATURE_COLUMNS.clear()
        FEATURE_COLUMNS.extend(active_feats)
        CATEGORICAL_FEATURES.clear()
        CATEGORICAL_FEATURES.extend(active_cats)
        log.info(f"Fill-rate gate: scoring with {len(active_feats)} features")
    except Exception as e:
        log.warning(f"Fill-rate gate check failed: {e}")

    # Try loading cached models first (skip retraining on daily runs)
    models = None
    if MODELS_CACHE.exists():
        try:
            import pickle
            with open(MODELS_CACHE, 'rb') as f:
                models = pickle.load(f)
            if len(models) == 4:
                log.info("Loaded cached ML models — skipping retraining")
            else:
                models = None
        except Exception as e:
            log.warning(f"Model cache load failed: {e}")
            models = None

    if models is None:
        models = train_full_sample(conn)
        if models is None:
            log.warning("Could not train models — skipping scoring")
            return 0
        # Cache for next daily run
        try:
            import pickle
            with open(MODELS_CACHE, 'wb') as f:
                pickle.dump(models, f)
            log.info(f"ML models cached to {MODELS_CACHE}")
        except Exception as e:
            log.warning(f"Model cache save failed: {e}")

    clf_rf, clf_lgb, reg_rf, reg_lgb = models

    # Ensure score breakdown columns exist
    for col in ['ml_confidence', 'predicted_car',
                'score_base', 'score_magnitude', 'score_converge', 'score_person']:
        try:
            conn.execute(f"ALTER TABLE signals ADD COLUMN {col} REAL")
        except Exception:
            pass  # column already exists

    # Prepare features for ALL signals
    X, ids, dates, tickers, cars, X_raw = prepare_features_all(conn)
    if len(X) == 0:
        log.warning("No signals to score")
        return 0

    # Classification: ensemble P(beat SPY)
    clf_probs = (clf_rf.predict_proba(X)[:, 1] + clf_lgb.predict_proba(X)[:, 1]) / 2

    # Regression: ensemble predicted CAR
    reg_preds = (reg_rf.predict(X) + reg_lgb.predict(X)) / 2

    # Fetch convergence + person + source + role + name + sector + cluster data for bonus terms
    rows = conn.execute(
        "SELECT id, convergence_tier, person_hit_rate_30d, source, has_convergence, "
        "insider_role, representative, insider_name, sector, market_regime, "
        "same_ticker_signals_7d FROM signals"
    ).fetchall()
    meta = {r['id']: (r['convergence_tier'] or 0, r['person_hit_rate_30d'] or 0,
                      r['source'] or '', r['has_convergence'] or 0,
                      r['insider_role'] or '',
                      r['representative'] or r['insider_name'] or '',
                      r['sector'] or 'Unknown',
                      r['market_regime'] or 'normal',
                      r['same_ticker_signals_7d'] or 0) for r in rows}

    # Load optimized coefficients if available
    weights_data = load_json(OPTIMAL_WEIGHTS) if OPTIMAL_WEIGHTS.exists() else {}
    coeffs = weights_data.get('_score_coefficients', {})
    base_mult = coeffs.get('base_mult', 60)
    mag_mult = coeffs.get('magnitude_mult', 200)
    conv_mult = coeffs.get('converge_mult', 5)
    person_mult = coeffs.get('person_mult', 8)
    if coeffs:
        log.info(f"Using optimized coefficients: base={base_mult}, mag={mag_mult}, "
                 f"conv={conv_mult}, person={person_mult}")

    # Load source quality multipliers (learned from --analyze)
    src_quality = weights_data.get('_source_quality', {})
    edgar_sq = src_quality.get('edgar', 1.0)
    congress_sq = src_quality.get('congress', 1.0)
    convergence_sq = src_quality.get('convergence', 1.0)
    if src_quality:
        log.info(f"Source quality multipliers: edgar={edgar_sq:.3f}, congress={congress_sq:.3f}, "
                 f"convergence={convergence_sq:.3f}")

    # Load role quality bonuses (learned from --analyze, or defaults)
    role_quality = weights_data.get('_role_quality', {})
    # Defaults: COO/CFO/President get highest bonus, CEO/Director moderate
    ROLE_QUALITY_DEFAULTS = {
        'COO': 1.25, 'CFO': 1.25, 'President': 1.25,
        'CEO': 1.10, 'Director': 1.10, 'Officer': 1.10,
    }
    if role_quality:
        log.info(f"Role quality bonuses loaded: {len(role_quality)} roles")

    # Load trader tiers for fade signal (learned from --analyze)
    trader_tier_data = weights_data.get('_trader_tiers', {})
    trader_tiers = trader_tier_data.get('tiers', {})
    fade_multiplier = trader_tier_data.get('fade_multiplier', 0.35)
    congress_tier_mults = trader_tier_data.get('congress_tier_multipliers', {
        'elite': 1.20, 'good': 0.80, 'neutral': 0.30, 'fade': 0.10,
    })
    congress_new_mult = trader_tier_data.get('congress_new_multiplier', 0.25)
    if trader_tiers:
        n_fade = sum(1 for v in trader_tiers.values() if v == 'fade')
        n_elite = sum(1 for v in trader_tiers.values() if v == 'elite')
        log.info(f"Trader tiers loaded: {len(trader_tiers)} traders ({n_elite} elite, {n_fade} fade)")

    # Compute scores
    updates = []
    scores = []
    # Load regime caps from optimal_weights (tunable)
    regime_caps = weights_data.get('_regime_caps', {
        'crisis': 70, 'low_vol_momentum_boost': 1.05,
    })
    crisis_cap = regime_caps.get('crisis', 70)
    low_vol_boost = regime_caps.get('low_vol_momentum_boost', 1.05)

    for i in range(len(ids)):
        sig_id = int(ids[i])
        conv_tier, person_hr, source, has_conv, role, person_name, sector, regime, cluster_7d = meta.get(sig_id, (0, 0, '', 0, '', '', 'Unknown', 'normal', 0))

        base = float(clf_probs[i]) * base_mult
        magnitude = max(-20, min(25, float(reg_preds[i]) * mag_mult))
        converge = float(conv_tier) * conv_mult
        person = max(0, min(5, float(person_hr) * person_mult))
        raw_total = base + magnitude + converge + person

        # Apply source quality multiplier
        if has_conv and convergence_sq != 1.0:
            source_mult = convergence_sq
        elif source == 'edgar':
            source_mult = edgar_sq
        elif source == 'congress':
            # Use individual congress tier multiplier if available
            tier = trader_tiers.get(person_name)
            if tier:
                source_mult = congress_tier_mults.get(tier, congress_sq)
            elif person_name:
                source_mult = congress_new_mult  # n<5, insufficient history
            else:
                source_mult = congress_sq
        else:
            source_mult = 1.0

        # Apply role quality bonus (multiplicative, after source_mult)
        role_bonus = 1.0
        if role:
            role_upper = role.strip().split(',')[0].strip()  # take first role if multiple
            role_bonus = role_quality.get(role_upper, ROLE_QUALITY_DEFAULTS.get(role_upper, 1.0))

        # Apply trader tier fade for EDGAR insiders (congress handled above via source_mult)
        trader_mult = 1.0
        if source == 'edgar' and person_name and trader_tiers.get(person_name) == 'fade':
            trader_mult = fade_multiplier

        # Cluster signal bonus — empirically validated (3+ signals: 68% hit, +5.07% CAR, n=490)
        cluster_mult = 1.0
        if cluster_7d >= 3:
            cluster_mult = 1.25
        elif cluster_7d == 2:
            cluster_mult = 1.10

        total = raw_total * source_mult * role_bonus * trader_mult * cluster_mult

        # Regime-conditional adjustment
        if regime == 'crisis':
            total = min(total, crisis_cap)
        elif regime == 'low_vol':
            total = total * low_vol_boost

        total = round(max(0, min(100, total)), 2)

        updates.append((total,
                        round(float(clf_probs[i]), 4),
                        round(float(reg_preds[i]), 6),
                        round(base, 2), round(magnitude, 2),
                        round(converge, 2), round(person, 2),
                        sig_id))
        scores.append(total)

    # Sector rank-normalization: blend absolute score with sector percentile
    # Prevents one sector from monopolizing top signals during sector runs
    sector_scores = {}  # sector → list of (index, score)
    for i, (total, *_, sig_id) in enumerate(updates):
        _, _, _, _, _, _, sector, _, _ = meta.get(sig_id, (0, 0, '', 0, '', '', 'Unknown', 'normal', 0))
        sector_scores.setdefault(sector, []).append((i, total))

    for sector, entries in sector_scores.items():
        if len(entries) < 3:
            continue  # too few signals in sector for meaningful percentile
        sector_vals = sorted(e[1] for e in entries)
        for idx, score in entries:
            # Compute percentile rank within sector (0-100)
            rank = sum(1 for v in sector_vals if v <= score) / len(sector_vals) * 100
            # Blend: 75% absolute + 25% sector percentile
            blended = round(0.75 * score + 0.25 * rank, 2)
            blended = max(0, min(100, blended))
            # Replace total_score in the update tuple
            old_tuple = updates[idx]
            updates[idx] = (blended,) + old_tuple[1:]
            scores[idx] = blended

    # Batch update
    conn.executemany(
        "UPDATE signals SET total_score = ?, ml_confidence = ?, predicted_car = ?, "
        "score_base = ?, score_magnitude = ?, score_converge = ?, score_person = ? "
        "WHERE id = ?",
        updates
    )
    conn.commit()

    # Log compact distribution
    import numpy as np
    scores_arr = np.array(scores)
    n80 = int(np.sum(scores_arr >= 80))
    n60 = int(np.sum((scores_arr >= 60) & (scores_arr < 80)))
    n40 = int(np.sum((scores_arr >= 40) & (scores_arr < 60)))
    nlo = int(np.sum(scores_arr < 40))
    log.info(f"Scored {len(scores)} signals — mean={scores_arr.mean():.1f}, max={scores_arr.max():.1f} "
             f"| 80+:{n80} 60-79:{n60} 40-59:{n40} <40:{nlo}")

    return len(scores)


def log_brain_run(conn: sqlite3.Connection, run_type: str,
                  oos_ic: float = None, oos_hit_rate: float = None,
                  notes: str = None, step_counts: dict = None) -> None:
    """Record a Brain run in brain_runs table for health tracking."""
    import numpy as np

    cur = conn.cursor()
    row = cur.execute(
        "SELECT COUNT(*) as n, AVG(total_score) as avg_s, MAX(total_score) as max_s "
        "FROM signals WHERE total_score IS NOT NULL"
    ).fetchone()
    n_signals = cur.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    n_scored = row['n'] or 0
    avg_score = round(row['avg_s'], 2) if row['avg_s'] else None
    max_score = round(row['max_s'], 2) if row['max_s'] else None

    # Top ticker concentration in top 50
    top50 = cur.execute(
        "SELECT ticker, COUNT(*) as c FROM signals "
        "WHERE total_score IS NOT NULL ORDER BY total_score DESC LIMIT 50"
    ).fetchall()
    ticker_counts = {}
    for r in top50:
        ticker_counts[r['ticker']] = ticker_counts.get(r['ticker'], 0) + r['c']
    top_ticker = max(ticker_counts, key=ticker_counts.get) if ticker_counts else None
    top_ticker_pct = round(ticker_counts.get(top_ticker, 0) / max(len(top50), 1), 3) if top_ticker else None

    # Feature importance from optimal_weights
    weights = load_json(OPTIMAL_WEIGHTS) if OPTIMAL_WEIGHTS.exists() else {}
    fi = weights.get('_feature_importance')
    fi_json = json.dumps(fi) if fi else None

    sc_json = json.dumps(step_counts) if step_counts else None
    cur.execute("""
        INSERT INTO brain_runs (run_date, run_type, oos_ic, oos_hit_rate,
            n_signals, n_scored, avg_score, max_score, top_ticker, top_ticker_pct,
            feature_importance_json, notes, step_counts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now(tz=timezone.utc).strftime('%Y-%m-%d'), run_type,
          oos_ic, oos_hit_rate, n_signals, n_scored, avg_score, max_score,
          top_ticker, top_ticker_pct, fi_json, notes, sc_json))
    conn.commit()
    log.debug(f"Brain run logged: type={run_type}, IC={oos_ic}, scored={n_scored}")


def run_self_check(conn: sqlite3.Connection) -> dict:
    """Run Brain health diagnostics and export brain_health.json.

    Produces structured health report with per-check status (ok/warn/critical),
    overall_status derived from individual checks, and actionable recommendations.
    """
    import numpy as np
    log.info("=== Brain Self-Check ===")
    cur = conn.cursor()
    now_utc = datetime.now(tz=timezone.utc)
    checks = {}
    recommendations = []

    # ── 1. IC Trend ──
    runs = cur.execute(
        "SELECT run_date, oos_ic, oos_hit_rate, run_type FROM brain_runs "
        "WHERE oos_ic IS NOT NULL ORDER BY id DESC LIMIT 10"
    ).fetchall()
    ic_history = [{'date': r['run_date'], 'ic': r['oos_ic'], 'hit_rate': r['oos_hit_rate'], 'type': r['run_type']} for r in runs]

    ic_check = {'status': 'ok', 'threshold_warn': 0.04, 'threshold_critical': 0.0}
    if ic_history:
        current_ic = ic_history[0]['ic']
        prev_ic = ic_history[1]['ic'] if len(ic_history) >= 2 else None
        ic_check['current_ic'] = round(current_ic, 4)
        ic_check['prev_ic'] = round(prev_ic, 4) if prev_ic is not None else None
        if len(ic_history) >= 3:
            last3 = [h['ic'] for h in ic_history[:3]]
            if all(last3[i] < last3[i+1] for i in range(2)):
                ic_check['trend'] = 'degrading'
                ic_check['status'] = 'warn'
                recommendations.append(f"IC declining for 3 runs ({' → '.join(f'{x:.4f}' for x in reversed(last3))}). Run --analyze with fresh data.")
            elif all(last3[i] > last3[i+1] for i in range(2)):
                ic_check['trend'] = 'improving'
            else:
                ic_check['trend'] = 'stable'
        else:
            ic_check['trend'] = 'insufficient_data'
        if current_ic < 0:
            ic_check['status'] = 'critical'
            recommendations.append("IC is negative — model predictions are inversely correlated with outcomes. Retrain immediately.")
        elif current_ic < 0.04:
            ic_check['status'] = 'warn'
    else:
        ic_check['status'] = 'warn'
        ic_check['current_ic'] = None
        ic_check['prev_ic'] = None
        ic_check['trend'] = 'no_data'
        recommendations.append("No IC history. Run --analyze to train ML models.")
    checks['ic_trend'] = ic_check

    # ── 2. Hit Rate ──
    hr_check = {'status': 'ok', 'threshold_warn': 0.50, 'threshold_critical': 0.45}
    if ic_history:
        current_hr = ic_history[0]['hit_rate']
        hr_check['current'] = round(current_hr, 3) if current_hr else None
        # 30d rolling: average of runs in last 30 days
        recent_hrs = [h['hit_rate'] for h in ic_history
                      if h['hit_rate'] is not None and h['date'] >= (now_utc - timedelta(days=30)).strftime('%Y-%m-%d')]
        hr_check['30d_rolling'] = round(sum(recent_hrs) / len(recent_hrs), 3) if recent_hrs else None
        effective = hr_check['30d_rolling'] or hr_check['current']
        if effective is not None:
            if effective < 0.45:
                hr_check['status'] = 'critical'
                recommendations.append(f"Hit rate critically low ({effective:.1%}). Model may need retraining or data quality review.")
            elif effective < 0.50:
                hr_check['status'] = 'warn'
    else:
        hr_check['current'] = None
        hr_check['30d_rolling'] = None
    checks['hit_rate'] = hr_check

    # ── 3. Data Freshness ──
    fresh_check = {'status': 'ok', 'threshold_warn_hours': 48, 'threshold_critical_hours': 96}
    congress_latest = cur.execute("SELECT MAX(signal_date) as d FROM signals WHERE source='congress'").fetchone()['d']
    edgar_latest = cur.execute("SELECT MAX(signal_date) as d FROM signals WHERE source='edgar'").fetchone()['d']
    fresh_check['congress_last_updated'] = congress_latest
    fresh_check['edgar_last_updated'] = edgar_latest

    # Source-specific staleness thresholds (hours)
    # Congress: FMP upstream may have multi-week delays — use relaxed thresholds
    staleness_thresholds = {
        'edgar':    {'warn': 48, 'critical': 96},
        'congress': {'warn': 336, 'critical': 720},  # 14d warn, 30d critical
    }
    for source_name, latest_date in [('congress', congress_latest), ('edgar', edgar_latest)]:
        hrs_key = f'hours_since_{source_name}'
        thresholds = staleness_thresholds.get(source_name, {'warn': 48, 'critical': 96})
        if latest_date:
            delta = now_utc - datetime.fromisoformat(latest_date).replace(tzinfo=timezone.utc)
            hours = delta.total_seconds() / 3600
            fresh_check[hrs_key] = round(hours, 1)
            if hours > thresholds['critical']:
                fresh_check['status'] = 'critical'
                recommendations.append(f"{source_name} data is {hours/24:.0f} days stale. Check fetch pipeline.")
            elif hours > thresholds['warn'] and fresh_check['status'] != 'critical':
                fresh_check['status'] = 'warn'
        else:
            fresh_check[hrs_key] = None
    checks['data_freshness'] = fresh_check

    # ── 4. Feature Drift ──
    drift_check = {'status': 'ok', 'features_below_50pct_fill': [], 'features_degraded_since_last_run': []}
    fill_sql = """
        SELECT
            ROUND(100.0*SUM(CASE WHEN sector IS NOT NULL AND sector != '' THEN 1 ELSE 0 END)/COUNT(*),1) as sector,
            ROUND(100.0*SUM(CASE WHEN market_cap_bucket IS NOT NULL THEN 1 ELSE 0 END)/COUNT(*),1) as market_cap,
            ROUND(100.0*SUM(CASE WHEN momentum_1m IS NOT NULL THEN 1 ELSE 0 END)/COUNT(*),1) as momentum,
            ROUND(100.0*SUM(CASE WHEN person_hit_rate_30d IS NOT NULL THEN 1 ELSE 0 END)/COUNT(*),1) as person_hr,
            ROUND(100.0*SUM(CASE WHEN person_avg_car_30d IS NOT NULL THEN 1 ELSE 0 END)/COUNT(*),1) as person_car,
            ROUND(100.0*SUM(CASE WHEN sector_momentum IS NOT NULL THEN 1 ELSE 0 END)/COUNT(*),1) as sector_mom
        FROM signals
    """
    edgar_fill_sql = """
        SELECT
            ROUND(100.0*SUM(CASE WHEN insider_role IS NOT NULL AND insider_role != '' THEN 1 ELSE 0 END)/COUNT(*),1) as insider_role,
            ROUND(100.0*SUM(CASE WHEN days_since_last_buy IS NOT NULL THEN 1 ELSE 0 END)/COUNT(*),1) as repeat_buyer
        FROM signals WHERE source = 'edgar'
    """
    fill = dict(cur.execute(fill_sql).fetchone())
    fill.update(dict(cur.execute(edgar_fill_sql).fetchone()))
    drift_check['feature_fill_rates'] = fill
    drift_check['features_below_50pct_fill'] = [k for k, v in fill.items() if v is not None and v < 50]
    if drift_check['features_below_50pct_fill']:
        drift_check['status'] = 'warn'
        recommendations.append(f"Low fill features: {', '.join(f'{k}={fill[k]}%' for k in drift_check['features_below_50pct_fill'])}")

    # Feature importance drift between runs
    fi_runs = cur.execute(
        "SELECT feature_importance_json FROM brain_runs "
        "WHERE feature_importance_json IS NOT NULL ORDER BY id DESC LIMIT 2"
    ).fetchall()
    if len(fi_runs) >= 2:
        fi_new = json.loads(fi_runs[0]['feature_importance_json'])
        fi_old = json.loads(fi_runs[1]['feature_importance_json'])
        for feat in fi_new:
            old_v = fi_old.get(feat, 0)
            new_v = fi_new[feat]
            if old_v > 0 and abs(new_v - old_v) / old_v > 0.5:
                drift_check['features_degraded_since_last_run'].append(
                    f"{feat}: {old_v:.3f}→{new_v:.3f}"
                )
    checks['feature_drift'] = drift_check

    # ── 5. Score Concentration ──
    conc_check = {'status': 'ok', 'threshold_warn': 0.30}
    top50 = cur.execute(
        "SELECT ticker FROM signals WHERE total_score IS NOT NULL "
        "ORDER BY total_score DESC LIMIT 50"
    ).fetchall()
    ticker_counts = {}
    for r in top50:
        ticker_counts[r['ticker']] = ticker_counts.get(r['ticker'], 0) + 1
    if ticker_counts:
        top_t = max(ticker_counts, key=ticker_counts.get)
        sorted_counts = sorted(ticker_counts.values(), reverse=True)
        top5_pct = sum(sorted_counts[:5]) / len(top50) if len(top50) > 0 else 0
        conc_check['top_ticker'] = top_t
        conc_check['top_ticker_signal_count'] = ticker_counts[top_t]
        conc_check['pct_signals_in_top5_tickers'] = round(top5_pct, 3)
        if top5_pct > 0.30:
            conc_check['status'] = 'warn'
            recommendations.append(f"Top 5 tickers hold {top5_pct:.0%} of top-50 signals. Consider diversification cap.")
    else:
        conc_check['top_ticker'] = None
        conc_check['top_ticker_signal_count'] = 0
        conc_check['pct_signals_in_top5_tickers'] = 0.0
    checks['score_concentration'] = conc_check

    # ── 6. Harmful Features ──
    harmful_check = {'status': 'ok', 'active_harmful_features': [], 'note': 'features/values with CAR < -2% and n > 30'}
    harmful_rows = cur.execute(
        "SELECT feature_name, feature_value, avg_car_30d, n_observations "
        "FROM feature_stats WHERE avg_car_30d < -0.02 AND n_observations >= 30 "
        "ORDER BY avg_car_30d ASC"
    ).fetchall()
    for r in harmful_rows:
        harmful_check['active_harmful_features'].append({
            'feature': r['feature_name'], 'value': r['feature_value'],
            'avg_car_30d': round(r['avg_car_30d'], 4), 'n': r['n_observations'],
        })
    if harmful_check['active_harmful_features']:
        harmful_check['status'] = 'warn'
        names = ', '.join(f"{h['feature']}={h['value']}" for h in harmful_check['active_harmful_features'][:5])
        recommendations.append(f"Harmful feature values detected: {names}. ML trees handle this but monitor for worsening.")
    checks['harmful_features'] = harmful_check

    # ── 7. Score Band Validation (internal, feeds recommendations) ──
    bands = cur.execute("""
        SELECT
            CASE WHEN total_score >= 80 THEN '80+'
                 WHEN total_score >= 65 THEN '65-79'
                 WHEN total_score >= 40 THEN '40-64'
                 ELSE '<40' END as band,
            COUNT(*) as n, ROUND(AVG(car_30d),4) as avg_car,
            ROUND(100.0*SUM(CASE WHEN car_30d > 0 THEN 1 ELSE 0 END)/NULLIF(SUM(CASE WHEN car_30d IS NOT NULL THEN 1 ELSE 0 END),0),1) as hit_rate
        FROM signals WHERE car_30d IS NOT NULL GROUP BY band ORDER BY band DESC
    """).fetchall()
    score_bands = [{'band': r['band'], 'n': r['n'], 'avg_car': r['avg_car'], 'hit_rate': r['hit_rate']} for r in bands]
    band_cars = {r['band']: r['avg_car'] for r in bands if r['avg_car'] is not None}
    if band_cars.get('80+', 0) < band_cars.get('<40', 0):
        recommendations.append("CRITICAL: 80+ signals underperform <40 — model may be inverting. Retrain immediately.")
    elif band_cars.get('80+', 0) < band_cars.get('65-79', 0):
        recommendations.append("Score band inversion: 80+ CAR < 65-79 CAR. Recalibrate scoring coefficients.")

    # ── 8. Auto-Pruning Candidates (internal, feeds recommendations) ──
    PRUNE_THRESHOLD = 0.01
    PRUNE_CONSECUTIVE_RUNS = 3
    fi_history = cur.execute(
        "SELECT feature_importance_json FROM brain_runs "
        "WHERE feature_importance_json IS NOT NULL ORDER BY id DESC LIMIT ?",
        (PRUNE_CONSECUTIVE_RUNS,)
    ).fetchall()
    prune_candidates = []
    if len(fi_history) >= PRUNE_CONSECUTIVE_RUNS:
        all_fi = [json.loads(r['feature_importance_json']) for r in fi_history]
        all_features = set(all_fi[0].keys())
        for feat in all_features:
            if all(fi.get(feat, 0) < PRUNE_THRESHOLD for fi in all_fi):
                avg_imp = sum(fi.get(feat, 0) for fi in all_fi) / len(all_fi)
                prune_candidates.append({'feature': feat, 'avg_importance': round(avg_imp, 4)})
        prune_candidates.sort(key=lambda x: x['avg_importance'])
        if prune_candidates:
            names = ', '.join(f"{c['feature']}={c['avg_importance']:.3f}" for c in prune_candidates)
            recommendations.append(f"Prune candidates (<1% importance for {PRUNE_CONSECUTIVE_RUNS}+ runs): {names}")

    # ── 9. Source Alpha ──
    source_stats = cur.execute("""
        SELECT source, COUNT(*) as n, ROUND(AVG(car_30d),4) as avg_car,
            ROUND(100.0*SUM(CASE WHEN car_30d > 0 THEN 1 ELSE 0 END)/NULLIF(SUM(CASE WHEN car_30d IS NOT NULL THEN 1 ELSE 0 END),0),1) as hit_rate
        FROM signals WHERE car_30d IS NOT NULL GROUP BY source
    """).fetchall()
    source_alpha = [{'source': r['source'], 'n': r['n'], 'avg_car': r['avg_car'], 'hit_rate': r['hit_rate']} for r in source_stats]
    for s in source_alpha:
        if s['avg_car'] is not None and s['avg_car'] < -0.005:
            recommendations.append(f"{s['source']} signals have negative avg CAR ({s['avg_car']:.2%}). Consider source quality down-weighting.")

    # ── 10. Data Completeness Audit ──
    completeness_check = {'status': 'ok', 'fields': {}}
    completeness_expectations = {
        'price_at_signal': {'expect': 0.95, 'scope': 'outcome_30d_filled = 1'},
        'total_score':     {'expect': 0.90, 'scope': '1=1'},
        'sector':          {'expect': 0.95, 'scope': '1=1'},
        'market_cap_bucket': {'expect': 0.80, 'scope': '1=1'},
        'momentum_1m':     {'expect': 0.75, 'scope': 'outcome_30d_filled = 1'},
        'vix_at_signal':   {'expect': 0.70, 'scope': '1=1'},
    }
    for col, spec in completeness_expectations.items():
        try:
            row = cur.execute(
                f"SELECT COUNT(*) as total, "
                f"SUM(CASE WHEN {col} IS NOT NULL AND {col} != '' AND {col} != 0 THEN 1 ELSE 0 END) as filled "
                f"FROM signals WHERE {spec['scope']}"
            ).fetchone()
            total, filled = row['total'], row['filled']
            rate = filled / total if total > 0 else 0
            completeness_check['fields'][col] = {
                'fill_rate': round(rate, 3), 'expected': spec['expect'],
                'total': total, 'filled': filled, 'missing': total - filled,
            }
            gap = spec['expect'] - rate
            if gap > 0.15:
                completeness_check['status'] = 'critical'
                recommendations.append(f"CRITICAL: {col} fill rate {rate:.1%} (expect {spec['expect']:.0%}). "
                                       f"{total - filled} signals missing. Run --backfill.")
            elif gap > 0.05:
                if completeness_check['status'] != 'critical':
                    completeness_check['status'] = 'warn'
                recommendations.append(f"{col} fill rate {rate:.1%} below target {spec['expect']:.0%} ({total - filled} missing)")
        except Exception:
            pass
    checks['data_completeness'] = completeness_check

    # ── 11. Strategy Readiness ──
    strat_check = {'status': 'ok'}
    try:
        threshold = TRADING_RULES['entry_threshold']
        eligible_row = cur.execute(
            "SELECT COUNT(*) FROM signals "
            "WHERE total_score >= ? AND outcome_30d_filled = 1 AND car_30d IS NOT NULL",
            (threshold,)
        ).fetchone()
        ready_row = cur.execute(
            "SELECT COUNT(*) FROM signals "
            "WHERE total_score >= ? AND outcome_30d_filled = 1 AND car_30d IS NOT NULL "
            "AND price_at_signal IS NOT NULL AND price_at_signal > 0",
            (threshold,)
        ).fetchone()
        eligible = eligible_row[0]
        ready = ready_row[0]
        ratio = ready / eligible if eligible > 0 else 0
        strat_check['eligible'] = eligible
        strat_check['ready'] = ready
        strat_check['readiness'] = round(ratio, 3)
        strat_check['threshold'] = threshold
        if ratio < 0.70:
            strat_check['status'] = 'critical'
            recommendations.append(f"CRITICAL: Strategy readiness {ratio:.1%} — only {ready}/{eligible} signals tradeable. "
                                   f"Run --backfill to recover entry prices.")
        elif ratio < 0.90:
            strat_check['status'] = 'warn'
            recommendations.append(f"Strategy readiness {ratio:.1%} ({eligible - ready} signals missing entry price)")
    except Exception as e:
        strat_check['status'] = 'warn'
        strat_check['error'] = str(e)
    checks['strategy_readiness'] = strat_check

    # ── Determine overall_status ──
    statuses = [c.get('status', 'ok') for c in checks.values()]
    n_critical = statuses.count('critical')
    n_warn = statuses.count('warn')
    if n_critical > 0:
        overall = 'critical'
    elif n_warn >= 2:
        overall = 'degraded'
    else:
        overall = 'healthy'

    # Also escalate if score bands show inversion
    if band_cars.get('80+', 0) < band_cars.get('<40', 0):
        overall = 'critical'

    health = {
        'generated_at': now_utc.isoformat(),
        'overall_status': overall,
        'checks': checks,
        'recommendations': recommendations,
        # Supplementary data for dashboards
        'ic_history': ic_history,
        'score_bands': score_bands,
        'source_alpha': source_alpha,
        'prune_candidates': prune_candidates,
    }

    # ── Fill-rate gate: candidate feature readiness ──
    try:
        from backtest.ml_engine import get_active_features, FILL_GATE_THRESHOLD
        _, _, fill_report = get_active_features(conn)
        health['feature_fill_gate'] = {
            'threshold': FILL_GATE_THRESHOLD,
            'candidates': fill_report,
        }
        waiting = [c for c, r in fill_report.items() if r['status'] == 'candidate']
        promoted = [c for c, r in fill_report.items() if r['status'] == 'active']
        if waiting:
            rates = ', '.join(f"{c}={fill_report[c]['fill_rate']:.0%}" for c in waiting)
            recommendations.append(f"Candidate features below fill threshold ({FILL_GATE_THRESHOLD:.0%}): {rates}")
        if promoted:
            recommendations.append(f"Auto-promoted features: {', '.join(promoted)}")
    except ImportError:
        pass

    # ── Signal hypotheses summary (if available) ──
    if SIGNAL_HYPOTHESES.exists():
        try:
            hyp = load_json(SIGNAL_HYPOTHESES)
            health['signal_hypotheses'] = {
                'high_residual_count': len(hyp.get('high_residual_tickers', [])),
                'top_interaction_candidate': (
                    hyp['feature_interactions'][0]['features']
                    if hyp.get('feature_interactions') else 'none'
                ),
                'worst_regime': (
                    hyp['regime_gaps'][0]['regime']
                    if hyp.get('regime_gaps') else 'none'
                ),
            }
        except Exception:
            pass

    # ── Summary log ──
    log.info(f"Health status: {overall.upper()}")
    for name, check in checks.items():
        log.info(f"  [{check.get('status', '?').upper():>8}] {name}")
    for r in recommendations:
        log.info(f"  [RECOMMEND] {r}")

    # DB health check
    try:
        db_info = db_health_check(conn)
        health['db_health'] = db_info
        if db_info.get('integrity') != 'ok':
            checks['db_integrity'] = {'status': 'critical', 'detail': db_info['integrity']}
            recommendations.append("Database integrity check failed — run PRAGMA integrity_check")
        else:
            checks['db_integrity'] = {'status': 'ok', 'size_mb': db_info.get('db_size_mb', 0),
                                      'indexes': db_info.get('index_count', 0)}
        if db_info.get('db_size_mb', 0) > 500:
            recommendations.append(f"Database is {db_info['db_size_mb']:.0f}MB — consider VACUUM")
    except Exception as e:
        log.warning(f"DB health check failed: {e}")

    save_json(BRAIN_HEALTH, health)
    log.info(f"Exported health → brain_health.json")
    return health


def compute_signal_intelligence(conn: sqlite3.Connection) -> dict:
    """Profile best/worst signals to surface predictive patterns.

    Filters to tradeable universe (mid+ cap) to avoid micro-cap noise.
    Uses quarterly earnings proximity (capped at 90 days) instead of raw
    days_to_earnings which was polluted by annual/missing values.
    """
    import pandas as pd
    min_cap = TRADING_RULES.get('min_market_cap', 'mid')
    min_rank = CAP_ORDER.get(min_cap, 3)
    valid_caps = [cap for cap, rank in CAP_ORDER.items() if rank >= min_rank]

    df = pd.read_sql('''
        SELECT ticker, signal_date, total_score, oos_score,
               car_30d, insider_role, sector, source,
               vix_at_signal, days_to_earnings, market_cap_bucket
        FROM signals
        WHERE car_30d IS NOT NULL AND oos_score IS NOT NULL
        ORDER BY car_30d DESC
    ''', conn)

    # Filter to tradeable universe
    if valid_caps:
        df = df[df['market_cap_bucket'].isin(valid_caps)]

    if len(df) < 100:
        log.info(f"Signal intelligence: only {len(df)} tradeable signals with OOS+CAR, skipping")
        return {}

    # Quarterly earnings: cap at 90 days, exclude 999/bad values
    df['earnings_q'] = df['days_to_earnings'].apply(
        lambda x: x if x is not None and 0 <= x <= 90 else None
    )

    top = df.head(50)
    bot = df.tail(50)

    def profile(group, label):
        eq = group['earnings_q'].dropna()
        result = {
            'label': label, 'n': len(group),
            'avg_score': round(float(group.total_score.mean()), 1),
            'avg_oos': round(float(group.oos_score.mean()), 1),
            'avg_car': round(float(group.car_30d.mean() * 100), 2),
            'avg_vix': round(float(group.vix_at_signal.dropna().mean()), 1) if group.vix_at_signal.notna().any() else None,
            'avg_days_to_earnings': round(float(eq.mean()), 1) if len(eq) > 0 else None,
            'pct_pre_earnings_30d': round(float((eq <= 30).mean()), 3) if len(eq) > 0 else None,
            'top_roles': group.insider_role.value_counts().head(3).to_dict(),
            'top_sectors': group.sector.value_counts().head(3).to_dict(),
            'top_caps': group.market_cap_bucket.value_counts().head(3).to_dict(),
            'source_split': group.source.value_counts().to_dict(),
            'examples': group[['ticker', 'signal_date', 'car_30d', 'insider_role']].head(10).to_dict('records'),
        }
        return result

    top_eq = top['earnings_q'].dropna()
    bot_eq = bot['earnings_q'].dropna()
    earnings_gap = None
    if len(top_eq) > 5 and len(bot_eq) > 5:
        earnings_gap = round(float(top_eq.mean() - bot_eq.mean()), 1)

    output = {
        'generated': datetime.now(tz=timezone.utc).isoformat(),
        'total_signals': len(df),
        'universe_filter': f'{min_cap}+ cap ({len(df)} signals)',
        'best_50': profile(top, 'best_50'),
        'worst_50': profile(bot, 'worst_50'),
        'divergence': {
            'score_gap': round(float(top.total_score.mean() - bot.total_score.mean()), 1),
            'oos_gap': round(float(top.oos_score.mean() - bot.oos_score.mean()), 1),
            'vix_gap': round(float(top.vix_at_signal.dropna().mean() - bot.vix_at_signal.dropna().mean()), 1) if top.vix_at_signal.notna().any() and bot.vix_at_signal.notna().any() else None,
            'earnings_gap': earnings_gap,
        },
    }
    save_json(SIGNAL_INTELLIGENCE, output)
    div = output['divergence']
    log.info(f"Signal intelligence ({min_cap}+ cap): score_gap={div['score_gap']}, oos_gap={div['oos_gap']}, "
             f"vix_gap={div.get('vix_gap', '?')}, earnings_gap={div.get('earnings_gap', '?')}")
    return output


# ── Self-Improving Intelligence ──────────────────────────────────────────────

def analyze_residuals(conn: sqlite3.Connection) -> dict:
    """Examine worst predictions: high score + negative CAR, low score + high CAR.

    Cross-references features to find systematic blind spots:
    - Sector/source concentration in misses
    - VIX regime at time of miss
    - Momentum, market cap, earnings proximity patterns
    - Repeat offender tickers

    Returns structured dict for brain_health.json and dashboard rendering.
    """
    import numpy as np
    cur = conn.cursor()

    # False positives: high score (≥70) but lost >5%
    false_pos = cur.execute("""
        SELECT ticker, sector, source, total_score, oos_score, car_30d, signal_date,
               price_at_signal, vix_at_signal, momentum_1m, momentum_3m,
               market_cap_bucket, days_to_earnings, insider_role, insider_name
        FROM signals
        WHERE total_score >= 70 AND car_30d IS NOT NULL AND car_30d < -0.05
        ORDER BY car_30d ASC LIMIT 30
    """).fetchall()

    # False negatives: low score (<50) but gained >15%
    false_neg = cur.execute("""
        SELECT ticker, sector, source, total_score, oos_score, car_30d, signal_date,
               price_at_signal, vix_at_signal, momentum_1m, momentum_3m,
               market_cap_bucket, days_to_earnings, insider_role, insider_name
        FROM signals
        WHERE total_score < 50 AND car_30d IS NOT NULL AND car_30d > 0.15
        ORDER BY car_30d DESC LIMIT 30
    """).fetchall()

    # Missed gems: scored 50-65 (below threshold) but gained >10%
    missed_gems = cur.execute("""
        SELECT ticker, sector, source, total_score, oos_score, car_30d, signal_date,
               market_cap_bucket, insider_role
        FROM signals
        WHERE total_score >= 50 AND total_score < 65 AND car_30d IS NOT NULL AND car_30d > 0.10
        ORDER BY car_30d DESC LIMIT 20
    """).fetchall()

    def _build_entries(rows):
        entries = []
        for r in rows:
            entries.append({
                'ticker': r['ticker'], 'sector': r['sector'] or 'Unknown',
                'source': r['source'], 'score': round(r['total_score'], 1),
                'oos_score': round(r['oos_score'], 1) if r['oos_score'] else None,
                'car_30d': round(r['car_30d'], 4), 'date': r['signal_date'],
                'entry_price': round(r['price_at_signal'], 2) if r['price_at_signal'] else None,
                'vix': round(r['vix_at_signal'], 1) if r['vix_at_signal'] else None,
                'momentum_1m': round(r['momentum_1m'], 4) if r['momentum_1m'] else None,
                'market_cap': r['market_cap_bucket'] or 'Unknown',
                'days_to_earnings': r['days_to_earnings'],
                'insider': r['insider_role'] or r['insider_name'] or '',
            })
        return entries

    fp_entries = _build_entries(false_pos)
    fn_entries = _build_entries(false_neg)
    mg_entries = _build_entries(missed_gems)

    # Cross-reference patterns
    patterns = []

    def _count_field(entries, field):
        counts = {}
        for e in entries:
            v = e.get(field, 'Unknown')
            counts[v] = counts.get(v, 0) + 1
        return counts

    # Sector patterns
    fp_sectors = _count_field(fp_entries, 'sector')
    fn_sectors = _count_field(fn_entries, 'sector')
    for sector, n in fp_sectors.items():
        if n >= 3:
            patterns.append({
                'type': 'sector_blind_spot',
                'detail': f"{sector}: {n} high-score losers — model overvalues insider buys in this sector",
                'severity': 'high' if n >= 5 else 'medium',
            })
    for sector, n in fn_sectors.items():
        if n >= 3:
            patterns.append({
                'type': 'sector_opportunity',
                'detail': f"{sector}: {n} low-score winners — model undervalues signals here",
                'severity': 'medium',
            })

    # VIX regime at time of miss
    fp_vix = [e['vix'] for e in fp_entries if e['vix']]
    fn_vix = [e['vix'] for e in fn_entries if e['vix']]
    if fp_vix:
        avg_fp_vix = np.mean(fp_vix)
        high_vix_misses = sum(1 for v in fp_vix if v > 25)
        if high_vix_misses >= 3:
            patterns.append({
                'type': 'vix_blind_spot',
                'detail': f"{high_vix_misses}/{len(fp_vix)} false positives occurred during VIX>25 — model doesn't penalize high-vol enough",
                'severity': 'high',
            })

    # Market cap pattern
    fp_caps = _count_field(fp_entries, 'market_cap')
    for cap, n in fp_caps.items():
        if n >= 4 and cap in ('micro', 'small', 'Unknown'):
            patterns.append({
                'type': 'cap_blind_spot',
                'detail': f"{n} false positives in {cap}-cap — small/micro stocks more prone to model errors",
                'severity': 'medium',
            })

    # Momentum at entry — were losers already in downtrend?
    fp_neg_mom = [e for e in fp_entries if e['momentum_1m'] is not None and e['momentum_1m'] < -0.05]
    if len(fp_neg_mom) >= 3:
        patterns.append({
            'type': 'momentum_blind_spot',
            'detail': f"{len(fp_neg_mom)}/{len(fp_entries)} false positives had negative 1m momentum at entry — model should weight momentum higher",
            'severity': 'medium',
        })

    # Repeat offender tickers
    fp_tickers = _count_field(fp_entries, 'ticker')
    repeat_losers = {t: n for t, n in fp_tickers.items() if n >= 2}
    if repeat_losers:
        patterns.append({
            'type': 'repeat_losers',
            'detail': f"Repeat false positives: {', '.join(f'{t}({n}x)' for t, n in sorted(repeat_losers.items(), key=lambda x: -x[1])[:5])}",
            'severity': 'medium',
        })

    # Summary stats
    residuals = {
        'false_positives': fp_entries,
        'false_negatives': fn_entries,
        'missed_gems': mg_entries,
        'patterns': patterns,
        'summary': {
            'n_false_positives': len(fp_entries),
            'n_false_negatives': len(fn_entries),
            'n_missed_gems': len(mg_entries),
            'n_patterns': len(patterns),
            'avg_fp_loss': round(float(np.mean([e['car_30d'] for e in fp_entries])), 4) if fp_entries else 0,
            'avg_fn_gain': round(float(np.mean([e['car_30d'] for e in fn_entries])), 4) if fn_entries else 0,
            'avg_mg_gain': round(float(np.mean([e['car_30d'] for e in mg_entries])), 4) if mg_entries else 0,
            'fp_sectors': dict(sorted(fp_sectors.items(), key=lambda x: -x[1])[:5]),
            'fn_sectors': dict(sorted(fn_sectors.items(), key=lambda x: -x[1])[:5]),
        },
    }

    log.info(f"Residual analysis: {len(fp_entries)} false positives (avg {residuals['summary']['avg_fp_loss']:.2%}), "
             f"{len(fn_entries)} false negatives (avg +{residuals['summary']['avg_fn_gain']:.2%}), "
             f"{len(mg_entries)} missed gems, {len(patterns)} patterns")
    return residuals


def optimize_score_coefficients(conn: sqlite3.Connection) -> dict | None:
    """Grid search for optimal scoring formula coefficients that maximize OOS IC.

    Default formula: base=P×60, magnitude=clamp(CAR×200,-20,25), converge=tier×5, person=clamp(hr×8,0,5)
    Searches around defaults to find the best combo.

    Returns best coefficients dict or None if insufficient data.
    """
    import numpy as np
    from scipy.stats import spearmanr

    rows = conn.execute("""
        SELECT ml_confidence, predicted_car, convergence_tier, person_hit_rate_30d, car_30d
        FROM signals
        WHERE ml_confidence IS NOT NULL AND car_30d IS NOT NULL
    """).fetchall()

    if len(rows) < 100:
        log.info("Insufficient data for coefficient optimization (<100 scored signals with outcomes)")
        return None

    conf = np.array([r['ml_confidence'] or 0 for r in rows])
    pred_car = np.array([r['predicted_car'] or 0 for r in rows])
    conv_tier = np.array([r['convergence_tier'] or 0 for r in rows])
    person_hr = np.array([r['person_hit_rate_30d'] or 0 for r in rows])
    actual_car = np.array([r['car_30d'] for r in rows])

    # Grid search ranges around defaults
    base_mults = [40, 50, 60, 70, 80]
    mag_mults = [100, 150, 200, 250, 300]
    conv_mults = [3, 5, 7, 10]
    person_mults = [5, 8, 10, 12]

    best_ic = -1.0
    best_combo = (60, 200, 5, 8)

    for bm in base_mults:
        for mm in mag_mults:
            for cm in conv_mults:
                for pm in person_mults:
                    base = conf * bm
                    magnitude = np.clip(pred_car * mm, -20, 25)
                    converge = conv_tier * cm
                    person = np.clip(person_hr * pm, 0, 5)
                    total = np.clip(base + magnitude + converge + person, 0, 100)

                    ic, _ = spearmanr(total, actual_car)
                    if not np.isnan(ic) and ic > best_ic:
                        best_ic = ic
                        best_combo = (bm, mm, cm, pm)

    result = {
        'base_mult': best_combo[0],
        'magnitude_mult': best_combo[1],
        'converge_mult': best_combo[2],
        'person_mult': best_combo[3],
        'optimized_ic': round(best_ic, 6),
    }
    log.info(f"Optimal coefficients: base={best_combo[0]}, mag={best_combo[1]}, "
             f"conv={best_combo[2]}, person={best_combo[3]} → IC={best_ic:.4f}")
    return result


def log_feature_importance_history(conn: sqlite3.Connection, feature_importance: dict) -> None:
    """Record per-feature importance for trend analysis across runs."""
    if not feature_importance:
        return

    run_date = datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    rows = []
    for rank, (name, importance) in enumerate(sorted_features, 1):
        rows.append((run_date, name, round(importance, 6), rank))

    conn.executemany(
        "INSERT INTO feature_importance_history (run_date, feature_name, importance, rank) "
        "VALUES (?, ?, ?, ?)",
        rows
    )
    conn.commit()
    log.info(f"Logged {len(rows)} feature importances for {run_date}")


def _compute_source_quality(conn: sqlite3.Connection) -> dict | None:
    """Compute source quality multipliers from historical CAR by source.

    Returns dict with multipliers:
        edgar: baseline 1.0
        congress: ratio of congress_car / edgar_car (clamped 0.3-1.0)
        convergence: bonus multiplier for signals with both sources (clamped 1.0-1.5)

    Multipliers are LEARNED from data, not hardcoded. They update each --analyze run.
    """
    cur = conn.cursor()
    MIN_OBSERVATIONS = 50

    # Get avg CAR by source for signals with outcomes
    source_rows = cur.execute("""
        SELECT source, AVG(car_30d) as avg_car, COUNT(*) as n
        FROM signals
        WHERE outcome_30d_filled = 1 AND car_30d IS NOT NULL
        GROUP BY source
    """).fetchall()
    source_car = {r['source']: (r['avg_car'], r['n']) for r in source_rows}

    edgar_car, edgar_n = source_car.get('edgar', (None, 0))
    congress_car, congress_n = source_car.get('congress', (None, 0))

    if edgar_n < MIN_OBSERVATIONS or congress_n < MIN_OBSERVATIONS:
        return None

    # Convergence signals: both sources agree on same ticker
    conv_row = cur.execute("""
        SELECT AVG(car_30d) as avg_car, COUNT(*) as n
        FROM signals
        WHERE outcome_30d_filled = 1 AND car_30d IS NOT NULL AND has_convergence = 1
    """).fetchone()
    conv_car = conv_row['avg_car'] if conv_row and conv_row['n'] >= 20 else None

    # Compute multipliers relative to EDGAR (best source)
    # If EDGAR has positive CAR, use it as baseline
    if edgar_car is not None and edgar_car > 0:
        base_car = edgar_car
        edgar_mult = 1.0
        # Congress multiplier: ratio of congress/edgar CAR, clamped
        if congress_car is not None:
            raw_ratio = congress_car / base_car if base_car != 0 else 0.5
            congress_mult = round(max(0.3, min(1.0, raw_ratio)), 3)
        else:
            congress_mult = 0.65
    else:
        # Both negative or EDGAR negative — use uniform weights
        edgar_mult = 1.0
        congress_mult = 1.0

    # Convergence bonus: how much better are convergence signals vs average?
    all_avg = cur.execute(
        "SELECT AVG(car_30d) FROM signals WHERE outcome_30d_filled=1 AND car_30d IS NOT NULL"
    ).fetchone()[0]
    if conv_car is not None and all_avg is not None and all_avg > 0:
        conv_ratio = conv_car / all_avg
        conv_mult = round(max(1.0, min(1.5, conv_ratio)), 3)
    else:
        conv_mult = 1.2  # default modest bonus

    result = {
        'edgar': edgar_mult,
        'congress': congress_mult,
        'convergence': conv_mult,
        'edgar_avg_car': round(edgar_car, 6) if edgar_car else None,
        'congress_avg_car': round(congress_car, 6) if congress_car else None,
        'convergence_avg_car': round(conv_car, 6) if conv_car else None,
        'edgar_n': edgar_n,
        'congress_n': congress_n,
    }
    return result


def _compute_role_quality(conn: sqlite3.Connection) -> dict | None:
    """Compute role quality bonuses from historical CAR by insider_role.

    Returns dict mapping normalized role → bonus multiplier (1.0 = neutral).
    Roles with avg_car significantly above baseline get bonus > 1.0.
    Only considers roles with n >= 10 observations.
    """
    cur = conn.cursor()
    MIN_N = 10

    # Baseline: avg CAR across all EDGAR signals with outcomes
    baseline_row = cur.execute("""
        SELECT AVG(car_30d) as avg_car, COUNT(*) as n
        FROM signals
        WHERE source = 'edgar' AND outcome_30d_filled = 1 AND car_30d IS NOT NULL
    """).fetchone()
    if not baseline_row or baseline_row['n'] < 50:
        return None
    baseline_car = baseline_row['avg_car']
    if baseline_car is None or baseline_car <= 0:
        return None

    # Get avg CAR by insider_role
    role_rows = cur.execute("""
        SELECT insider_role, AVG(car_30d) as avg_car, COUNT(*) as n,
               AVG(CASE WHEN car_30d > 0 THEN 1.0 ELSE 0.0 END) as hit_rate
        FROM signals
        WHERE source = 'edgar' AND outcome_30d_filled = 1
          AND car_30d IS NOT NULL
          AND insider_role IS NOT NULL AND insider_role != ''
        GROUP BY insider_role
        HAVING COUNT(*) >= ?
    """, (MIN_N,)).fetchall()

    if not role_rows:
        return None

    role_bonuses = {}
    for r in role_rows:
        role = r['insider_role']
        avg_car = r['avg_car']
        # Bonus = ratio of role_car to baseline, clamped [0.7, 1.5]
        if avg_car is not None and baseline_car > 0:
            ratio = avg_car / baseline_car
            bonus = round(max(0.7, min(1.5, ratio)), 3)
        else:
            bonus = 1.0
        role_bonuses[role] = bonus
        log.info(f"  Role '{role}': CAR={avg_car:.4f}, hit={r['hit_rate']:.2f}, "
                 f"n={r['n']}, bonus={bonus}")

    return role_bonuses


def _compute_trader_tiers(conn: sqlite3.Connection) -> dict:
    """Classify congressional traders into quality tiers based on historical performance.

    Returns dict with:
        'tiers': {person_name: tier_label}  — 'elite'/'good'/'neutral'/'fade'
        'leaderboard': {'elite': [...], 'good': [...], 'fade': [...]}
        'fade_multiplier': float (applied to fade-tier traders' scores)
    """
    cur = conn.cursor()

    # Get person-level stats for congress signals with sufficient history
    person_rows = cur.execute("""
        SELECT representative, COUNT(*) as n,
               AVG(CASE WHEN car_30d > 0 THEN 1.0 ELSE 0.0 END) as hit_rate,
               AVG(car_30d) as avg_car
        FROM signals
        WHERE source = 'congress' AND outcome_30d_filled = 1
          AND car_30d IS NOT NULL AND representative IS NOT NULL AND representative != ''
        GROUP BY representative
        HAVING COUNT(*) >= 5
    """).fetchall()

    # Also get insider stats for EDGAR
    insider_rows = cur.execute("""
        SELECT insider_name, COUNT(*) as n,
               AVG(CASE WHEN car_30d > 0 THEN 1.0 ELSE 0.0 END) as hit_rate,
               AVG(car_30d) as avg_car
        FROM signals
        WHERE source = 'edgar' AND outcome_30d_filled = 1
          AND car_30d IS NOT NULL AND insider_name IS NOT NULL AND insider_name != ''
        GROUP BY insider_name
        HAVING COUNT(*) >= 5
    """).fetchall()

    tiers = {}
    leaderboard = {'elite': [], 'good': [], 'fade': []}

    for rows, name_field in [(person_rows, 'representative'), (insider_rows, 'insider_name')]:
        for r in rows:
            name = r[name_field]
            hr = r['hit_rate'] or 0
            avg = r['avg_car'] or 0
            n = r['n']

            if hr >= 0.65 and avg >= 0.05 and n >= 5:
                tier = 'elite'
            elif hr >= 0.55 and avg >= 0.02 and n >= 5:
                tier = 'good'
            elif avg < -0.03 and n >= 5:
                tier = 'fade'
            else:
                tier = 'neutral'

            tiers[name] = tier

            if tier in ('elite', 'good', 'fade'):
                leaderboard[tier].append({
                    'name': name,
                    'hit_rate': round(hr, 3),
                    'avg_car': round(avg, 4),
                    'n': n,
                })

    # Sort leaderboards
    for key in leaderboard:
        leaderboard[key].sort(key=lambda x: x['avg_car'], reverse=(key != 'fade'))

    log.info(f"Trader tiers: {sum(1 for v in tiers.values() if v == 'elite')} elite, "
             f"{sum(1 for v in tiers.values() if v == 'good')} good, "
             f"{sum(1 for v in tiers.values() if v == 'fade')} fade, "
             f"{sum(1 for v in tiers.values() if v == 'neutral')} neutral")

    return {
        'tiers': tiers,
        'leaderboard': leaderboard,
        'fade_multiplier': 0.35,  # default — can be calibrated
        # Congress tier multipliers: replace flat congress_sq for classified traders
        'congress_tier_multipliers': {
            'elite': 1.20,    # above EDGAR baseline — strongest congressional traders
            'good': 0.80,     # moderate — consistent but not exceptional
            'neutral': 0.30,  # default congress quality
            'fade': 0.10,     # strong contra-signal — near-zero weight
        },
        'congress_new_multiplier': 0.25,  # insufficient history (n<5)
    }


def _compute_regime_stats(conn: sqlite3.Connection) -> dict | None:
    """Compute regime distribution and hit rates from historical data.

    Returns dict with:
        'stats': {regime: {count, hit_rate, avg_car}}
        'caps': regime cap defaults (tunable in optimal_weights.json)
    """
    cur = conn.cursor()
    regimes = cur.execute("""
        SELECT market_regime, COUNT(*) as n,
               AVG(CASE WHEN car_30d > 0 THEN 1.0 ELSE 0.0 END) as hit_rate,
               AVG(car_30d) as avg_car
        FROM signals
        WHERE outcome_30d_filled = 1 AND car_30d IS NOT NULL
          AND market_regime IS NOT NULL
        GROUP BY market_regime
    """).fetchall()

    if not regimes:
        return None

    stats = {}
    for r in regimes:
        stats[r['market_regime']] = {
            'count': r['n'],
            'hit_rate': round(r['hit_rate'], 3),
            'avg_car': round(r['avg_car'], 4),
        }

    # Determine current regime from latest VIX
    latest_vix = cur.execute(
        "SELECT vix_at_signal FROM signals WHERE vix_at_signal IS NOT NULL "
        "ORDER BY signal_date DESC LIMIT 1"
    ).fetchone()
    current = _vix_to_regime(latest_vix['vix_at_signal'] if latest_vix else None)

    return {
        'stats': stats,
        'current_regime': current,
        'caps': {
            'crisis': 70,
            'low_vol_momentum_boost': 1.05,
        },
    }


def generate_signal_hypotheses(conn: sqlite3.Connection) -> dict:
    """Analyze the current model to generate hypotheses for new features.

    Three strategies:
    1. HIGH-RESIDUAL TICKERS: Where model consistently gets it wrong
    2. FEATURE INTERACTION ANALYSIS: Promising untested feature pairs
    3. REGIME PERFORMANCE GAPS: Regimes where model underperforms

    Output saved to data/signal_hypotheses.json.
    """
    import numpy as np
    from backtest.ml_engine import FEATURE_COLUMNS
    cur = conn.cursor()

    hypotheses = {
        'generated_at': datetime.now(tz=timezone.utc).isoformat(),
        'high_residual_tickers': [],
        'feature_interactions': [],
        'regime_gaps': [],
    }

    # 1. HIGH-RESIDUAL TICKERS
    # Find tickers where model is consistently wrong (high absolute residual)
    residual_rows = cur.execute("""
        SELECT ticker, sector, COUNT(*) as n,
               AVG(ABS(car_30d - predicted_car)) as avg_residual,
               AVG(car_30d) as actual_car, AVG(predicted_car) as pred_car
        FROM signals
        WHERE car_30d IS NOT NULL AND predicted_car IS NOT NULL
          AND outcome_30d_filled = 1
        GROUP BY ticker
        HAVING COUNT(*) >= 3 AND AVG(ABS(car_30d - predicted_car)) > 0.10
        ORDER BY AVG(ABS(car_30d - predicted_car)) DESC
        LIMIT 15
    """).fetchall()

    for r in residual_rows:
        direction = 'underestimating' if r['actual_car'] > r['pred_car'] else 'overestimating'
        hypothesis = f"Model {direction} — possible missing signal (M&A, sector catalyst, or unusual activity)"
        hypotheses['high_residual_tickers'].append({
            'ticker': r['ticker'],
            'sector': r['sector'],
            'n_signals': r['n'],
            'avg_residual': round(r['avg_residual'], 4),
            'actual_car': round(r['actual_car'], 4),
            'predicted_car': round(r['pred_car'], 4),
            'hypothesis': hypothesis,
        })

    # 2. FEATURE INTERACTION ANALYSIS
    # Find pairs of features that are both moderately predictive but untested together
    try:
        feature_rows = cur.execute(f"""
            SELECT {', '.join(FEATURE_COLUMNS)}, car_30d
            FROM signals
            WHERE outcome_30d_filled = 1 AND car_30d IS NOT NULL
        """).fetchall()

        if len(feature_rows) > 100:
            import pandas as pd
            df = pd.DataFrame([dict(r) for r in feature_rows])

            # Compute individual feature correlations with CAR
            ics = {}
            for col in FEATURE_COLUMNS:
                if col in df.columns and df[col].dtype in ('float64', 'int64', 'float32'):
                    valid = df[[col, 'car_30d']].dropna()
                    if len(valid) > 30:
                        ic = valid[col].corr(valid['car_30d'])
                        if abs(ic) > 0.02:
                            ics[col] = ic

            # Test interactions of top IC features
            top_features = sorted(ics.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
            for i, (f1, ic1) in enumerate(top_features):
                for f2, ic2 in top_features[i+1:]:
                    # Create interaction: both features above median
                    med1 = df[f1].median()
                    med2 = df[f2].median()
                    if med1 is None or med2 is None:
                        continue
                    interaction = ((df[f1] > med1) & (df[f2] > med2)).astype(int)
                    valid = pd.DataFrame({'interaction': interaction, 'car': df['car_30d']}).dropna()
                    if len(valid) > 30:
                        int_ic = valid['interaction'].corr(valid['car'])
                        # Is interaction IC better than either alone?
                        if abs(int_ic) > max(abs(ic1), abs(ic2)) * 1.1:
                            hypotheses['feature_interactions'].append({
                                'features': [f1, f2],
                                'individual_ics': [round(ic1, 4), round(ic2, 4)],
                                'interaction_ic': round(int_ic, 4),
                                'estimated_ic_gain': round(abs(int_ic) - max(abs(ic1), abs(ic2)), 4),
                                'hypothesis': f"Combined {f1} × {f2} may be stronger than either alone",
                            })

            # Sort by estimated IC gain
            hypotheses['feature_interactions'].sort(
                key=lambda x: x['estimated_ic_gain'], reverse=True
            )
            hypotheses['feature_interactions'] = hypotheses['feature_interactions'][:5]
    except Exception as e:
        log.warning(f"Feature interaction analysis failed: {e}")

    # 3. REGIME PERFORMANCE GAPS
    regime_rows = cur.execute("""
        SELECT market_regime, COUNT(*) as n,
               AVG(CASE WHEN car_30d > 0 THEN 1.0 ELSE 0.0 END) as hit_rate,
               AVG(car_30d) as avg_car
        FROM signals
        WHERE outcome_30d_filled = 1 AND car_30d IS NOT NULL
          AND market_regime IS NOT NULL
        GROUP BY market_regime
        HAVING COUNT(*) >= 10
    """).fetchall()

    for r in regime_rows:
        if r['hit_rate'] < 0.50:
            # Underperforming regime — find which features correlate with outcomes here
            regime_detail = cur.execute(f"""
                SELECT {', '.join(c for c in FEATURE_COLUMNS if c not in ('sector', 'insider_role', 'market_cap_bucket', 'market_regime'))},
                       car_30d
                FROM signals
                WHERE market_regime = ? AND outcome_30d_filled = 1 AND car_30d IS NOT NULL
            """, (r['market_regime'],)).fetchall()

            top_feature = 'unknown'
            if regime_detail:
                import pandas as pd
                rdf = pd.DataFrame([dict(row) for row in regime_detail])
                best_ic = 0
                for col in rdf.columns:
                    if col == 'car_30d' or rdf[col].dtype not in ('float64', 'int64', 'float32'):
                        continue
                    valid = rdf[[col, 'car_30d']].dropna()
                    if len(valid) > 10:
                        ic = abs(valid[col].corr(valid['car_30d']))
                        if ic > best_ic:
                            best_ic = ic
                            top_feature = col

            hypotheses['regime_gaps'].append({
                'regime': r['market_regime'],
                'n_signals': r['n'],
                'current_hit_rate': round(r['hit_rate'], 3),
                'avg_car': round(r['avg_car'], 4),
                'most_predictive_feature': top_feature,
                'hypothesis': f"Model underperforms in {r['market_regime']} regime — "
                              f"best signal: {top_feature}. Consider regime-specific features.",
            })

    # Save to file
    save_json(SIGNAL_HYPOTHESES, hypotheses)

    # Summary
    n_residual = len(hypotheses['high_residual_tickers'])
    n_interact = len(hypotheses['feature_interactions'])
    n_regime = len(hypotheses['regime_gaps'])
    log.info(f"Signal hypotheses: {n_residual} high-residual tickers, "
             f"{n_interact} interaction candidates, {n_regime} regime gaps")

    return hypotheses


# ── Factor attribution labels ────────────────────────────────────────────────
FACTOR_LABELS = {
    'momentum_3m':            ('3-month momentum',       'high', 'Strong 3-month momentum', 'Weak 3-month momentum'),
    'momentum_1m':            ('1-month momentum',       'high', 'Strong recent momentum', 'Weak recent momentum'),
    'momentum_6m':            ('6-month momentum',       'high', 'Strong 6-month trend', 'Weak 6-month trend'),
    'disclosure_delay':       ('Disclosure delay',       'low',  'Fast disclosure filing', 'Slow disclosure filing'),
    'trade_size_points':      ('Trade size',             'high', 'Large trade size', 'Small trade size'),
    'person_hit_rate_30d':    ('Person hit rate',        'high', 'Strong trader track record', 'Weak trader track record'),
    'person_trade_count':     ('Person trade count',     'high', 'Frequent trader', 'Infrequent trader'),
    'person_avg_car_30d':     ('Person avg return',      'high', 'High avg return trader', 'Low avg return trader'),
    'days_since_last_buy':    ('Days since last buy',    'low',  'Repeat buyer signal', 'Long gap since last buy'),
    'insider_buy_ratio_90d':  ('Insider buy ratio',      'high', 'High insider buy ratio', 'Low insider buy ratio'),
    'volume_spike':           ('Volume spike',           'high', 'Unusual volume activity', 'Normal volume'),
    'price_proximity_52wk':   ('52-week position',       'high', 'Near 52-week high', 'Near 52-week low'),
    # convergence_tier, has_convergence pruned in v5
    'same_ticker_signals_7d': ('7-day cluster',          'high', 'Signal cluster forming', 'Isolated signal'),
    'same_ticker_signals_30d':('30-day cluster',         'high', 'Multiple signals in 30d', 'Single signal in 30d'),
    'vix_at_signal':          ('VIX level',              'low',  'Low volatility environment', 'High volatility environment'),
    'days_to_earnings':       ('Days to earnings',       'low',  'Near earnings catalyst', 'Far from earnings'),
    # days_to_catalyst pruned in v5
    'sector_avg_car':         ('Sector returns',         'high', 'Strong sector returns', 'Weak sector returns'),
    'sector_momentum':        ('Sector momentum',        'high', 'Strong sector momentum', 'Weak sector momentum'),
    'cluster_velocity':       ('Cluster velocity',       'high', 'Fast signal clustering', 'Slow signal clustering'),
    'vix_regime_interaction': ('VIX regime',             'high', 'Favorable VIX regime', 'Unfavorable VIX regime'),
}


def compute_signal_factors(X_raw, all_ids, export_ids, feature_importance):
    """Compute top factors explaining why each signal scored high.

    Uses percentile-based attribution: for each feature, computes the signal's
    percentile rank vs the full population, then scores by
    global_importance × abs(percentile - 0.5) × 2.

    Args:
        X_raw: Raw (pre-encoded) feature DataFrame for ALL signals
        all_ids: Array of signal IDs matching X_raw rows
        export_ids: Set of signal IDs being exported (compute factors for these)
        feature_importance: Dict of feature_name → global importance weight

    Returns:
        Dict of signal_id → list of top 5 factor dicts
    """
    import numpy as np
    import pandas as pd

    if X_raw.empty or not feature_importance:
        return {}

    # Skip categorical features — can't compute meaningful percentiles on strings
    from backtest.ml_engine import CATEGORICAL_FEATURES
    numeric_cols = [c for c in X_raw.columns if c not in CATEGORICAL_FEATURES]

    # Compute percentile ranks for each numeric feature across full population
    percentiles = pd.DataFrame(index=X_raw.index, columns=numeric_cols, dtype=float)
    for col in numeric_cols:
        vals = pd.to_numeric(X_raw[col], errors='coerce').fillna(0).values
        ranked = pd.Series(vals).rank(pct=True, method='average').values
        percentiles[col] = ranked

    # Map all_ids to row index
    id_to_idx = {int(sid): i for i, sid in enumerate(all_ids)}

    factors_out = {}
    for sig_id in export_ids:
        idx = id_to_idx.get(sig_id)
        if idx is None:
            factors_out[sig_id] = []
            continue

        scored_features = []
        for feat in numeric_cols:
            imp = feature_importance.get(feat, 0)
            if imp < 0.005:  # skip near-zero importance features
                continue
            pctile = float(percentiles.iloc[idx][feat])
            # How extreme is this feature? 0.5 = median (boring), 0/1 = extreme
            extremity = abs(pctile - 0.5) * 2  # 0-1 scale
            attribution = imp * extremity
            scored_features.append((feat, attribution, pctile))

        # Sort by attribution, take top 5
        scored_features.sort(key=lambda x: x[1], reverse=True)
        top5 = scored_features[:5]

        factors = []
        for feat, attr, pctile in top5:
            label_info = FACTOR_LABELS.get(feat)
            if not label_info:
                continue
            name, direction, bull_text, bear_text = label_info
            # Determine if this factor is bullish or bearish
            is_high = pctile > 0.5
            if direction == 'low':
                is_high = pctile < 0.5  # for "low is good" features
            text = bull_text if is_high else bear_text
            pctile_display = int(round(pctile * 100))
            # Show percentile context
            if pctile >= 0.9:
                pctile_label = f"top {100 - pctile_display}%"
            elif pctile <= 0.1:
                pctile_label = f"bottom {pctile_display}%"
            else:
                pctile_label = f"{pctile_display}th percentile"

            raw_val = float(X_raw.iloc[idx][feat])
            factors.append({
                'feature': feat,
                'label': text,
                'direction': 'bullish' if is_high else 'bearish',
                'percentile': pctile_display,
                'percentile_label': pctile_label,
                'raw_value': round(raw_val, 4),
                'importance': round(attr, 4),
            })

        factors_out[sig_id] = factors

    return factors_out


def compute_smart_targets(price, predicted_car, momentum_1m, direction):
    """Compute smart entry/target/stop from model predictions.

    Args:
        price: Current price at signal
        predicted_car: ML-predicted 30-day cumulative abnormal return (decimal)
        momentum_1m: 1-month momentum (decimal, e.g. 0.05 = 5%)
        direction: 'long' or 'short'

    Returns:
        Dict with entry_lo, entry_hi, target1, target2, stop
    """
    if not price or price <= 0:
        return {'entry_lo': None, 'entry_hi': None, 'target1': None, 'target2': None, 'stop': None}

    # Volatility factor from momentum (more volatile → wider zones)
    vol_factor = max(0.02, min(0.08, abs(momentum_1m or 0) * 1.5 + 0.02))

    if predicted_car is not None and predicted_car != 0:
        car = predicted_car
        if direction == 'long':
            entry_lo = round(price * (1 - vol_factor), 2)
            entry_hi = round(price * (1 + vol_factor * 0.5), 2)
            t1_mult = max(0.03, min(0.50, abs(car)))
            t2_mult = max(0.05, min(0.50, abs(car) * 1.8))
            target1 = round(price * (1 + t1_mult), 2)
            target2 = round(price * (1 + t2_mult), 2)
            stop_pct = max(abs(car) * 0.6, 0.05)
            stop = round(price * (1 - stop_pct), 2)
        else:
            entry_lo = round(price * (1 - vol_factor * 0.5), 2)
            entry_hi = round(price * (1 + vol_factor), 2)
            t1_mult = max(0.03, min(0.50, abs(car)))
            t2_mult = max(0.05, min(0.50, abs(car) * 1.8))
            target1 = round(price * (1 - t1_mult), 2)
            target2 = round(price * (1 - t2_mult), 2)
            stop_pct = max(abs(car) * 0.6, 0.05)
            stop = round(price * (1 + stop_pct), 2)
    else:
        # Fallback: dumb multipliers when no ML prediction
        if direction == 'long':
            entry_lo = round(price * 0.96, 2)
            entry_hi = round(price * 1.04, 2)
            target1 = round(price * 1.20, 2)
            target2 = round(price * 1.35, 2)
            stop = round(price * 0.88, 2)
        else:
            entry_lo = round(price * 0.96, 2)
            entry_hi = round(price * 1.04, 2)
            target1 = round(price * 0.80, 2)
            target2 = round(price * 0.65, 2)
            stop = round(price * 1.12, 2)

    return {
        'entry_lo': entry_lo, 'entry_hi': entry_hi,
        'target1': target1, 'target2': target2, 'stop': stop,
    }


def get_regime_context() -> dict:
    """Get current VIX regime and multiplier from market_data.json.

    Based on OOS IC by VIX bucket:
      VIX 15-25: IC=0.095-0.131 (strong) → 1.00x
      VIX 25-30: IC=0.063 ns (weakening) → 0.85x
      VIX < 15:  IC~0.04 ns (weak)       → 0.75x
      VIX 30-40: limited data             → 0.90x
      VIX > 40:  2020-style crash         → 0.60x
    """
    REGIME_MAP = {
        'OPTIMAL':  (15, 25, 1.00, 'VIX 15-25: strongest prediction zone (IC=0.10-0.13)'),
        'LOW_VOL':  (0,  15, 0.75, 'VIX <15: model weakest, reduce sizing'),
        'ELEVATED': (25, 30, 0.85, 'VIX 25-30: signal weakening'),
        'HIGH_VOL': (30, 40, 0.90, 'VIX 30-40: limited data, cautious'),
        'CRISIS':   (40, 999, 0.60, 'VIX >40: crisis — 2020-style tail risk'),
    }
    mkt_path = DATA_DIR / 'market_data.json'
    current_vix = None
    if mkt_path.exists():
        try:
            mkt = load_json(mkt_path)
            vix_data = mkt.get('vix', {})
            current_vix = vix_data.get('value') if isinstance(vix_data, dict) else vix_data
        except Exception:
            pass

    if current_vix is None:
        return {'current_vix': None, 'regime': 'UNKNOWN', 'multiplier': 1.0,
                'note': 'VIX unavailable — no regime adjustment'}

    for label, (lo, hi, mult, note) in REGIME_MAP.items():
        if lo <= current_vix < hi:
            return {'current_vix': round(current_vix, 2), 'regime': label,
                    'multiplier': mult, 'note': note}

    return {'current_vix': round(current_vix, 2), 'regime': 'NORMAL',
            'multiplier': 1.0, 'note': ''}


def compute_kelly_size(oos_score, hit_rate, avg_win, avg_loss,
                       regime_multiplier=1.0, max_position=0.15,
                       min_position=0.02):
    """Quarter-Kelly position sizing scaled by signal confidence and regime.

    Full Kelly = (p*b - q) / b where b = avg_win/|avg_loss|.
    Uses 1/4 Kelly (conservative for retail). Caps at max_position.
    """
    if avg_loss == 0 or avg_win == 0 or hit_rate <= 0:
        return min_position

    b = avg_win / abs(avg_loss)
    p = hit_rate
    q = 1 - p

    full_kelly = (p * b - q) / b
    if full_kelly <= 0:
        return min_position

    quarter_kelly = full_kelly * 0.25
    confidence_scale = min(oos_score / 100.0, 1.0) if oos_score else 0.5
    sized = quarter_kelly * confidence_scale * regime_multiplier

    return round(max(min_position, min(max_position, sized)), 3)


# ── Trading Rules Engine ─────────────────────────────────────────────────────

TRADING_RULES = {
    'entry_threshold': 65,       # minimum total_score to initiate a buy
    'stop_loss_pct': -0.10,      # -10% hard stop from entry price
    'hold_days': 30,             # max hold period (days)
    'min_market_cap': 'mid',     # minimum cap bucket (options liquidity)
    'position_tiers': [          # (min_score, weight_multiplier)
        (85, 1.5),               # 85+: conviction — 1.5x
        (75, 1.0),               # 75-84: standard — 1.0x
        (65, 0.5),               # 65-74: starter — 0.5x
    ],
    # Trailing stop: activates after gain exceeds trail_activate_pct,
    # then trails at trail_pct below the high-water mark.
    # Score tiers get different trails — higher conviction = wider leash.
    'trail_activate_pct': 0.08,  # trailing stop activates after +8% gain
    'trail_tiers': [             # (min_score, trail_pct_from_peak)
        (85, 0.15),              # 85+: wide trail — give conviction picks room
        (75, 0.12),              # 75-84: standard trail
        (65, 0.10),              # 65-74: tight trail — protect starter gains
    ],
}

# Market cap buckets ordered by size (for min_market_cap filter)
CAP_ORDER = {'mega': 5, 'large': 4, 'mid': 3, 'small': 2, 'micro': 1}


def _passes_cap_filter(cap_bucket: str, rules: dict = None) -> bool:
    """Check if a market cap bucket meets the minimum threshold."""
    rules = rules or TRADING_RULES
    min_cap = rules.get('min_market_cap')
    if not min_cap:
        return True
    min_rank = CAP_ORDER.get(min_cap, 0)
    actual_rank = CAP_ORDER.get(cap_bucket or '', 0)
    return actual_rank >= min_rank


def _position_weight(score: float, rules: dict = None) -> float:
    """Map score to position weight multiplier."""
    rules = rules or TRADING_RULES
    for min_score, weight in rules['position_tiers']:
        if score >= min_score:
            return weight
    return 0.0


def _trail_pct(score: float, rules: dict = None) -> float:
    """Map score to trailing stop percentage (distance from peak)."""
    rules = rules or TRADING_RULES
    for min_score, trail in rules.get('trail_tiers', []):
        if score >= min_score:
            return trail
    return 0.10  # default tight trail


def simulate_trades(conn: sqlite3.Connection, rules: dict = None) -> list[dict]:
    """Simulate trades with entry threshold, hard stop, trailing stop, position sizing.

    For each signal above entry_threshold with a filled outcome:
      1. Entry at price_at_signal on signal_date
      2. Hard stop-loss at stop_loss_pct from entry (always active)
      3. Once gain exceeds trail_activate_pct, trailing stop activates:
         - Tracks high-water mark (HWM) of daily closes
         - Exits when price drops trail_pct below HWM
         - Trail width varies by score tier (higher score = wider trail)
      4. If neither triggered, exit at close on hold_days
      5. Position weight based on score tier

    This lets winners run: no fixed TP cap. High-conviction signals get
    a wider trailing stop so the best picks have room to compound.

    Returns list of trade dicts with entry/exit details and weighted return.
    """
    import pandas as pd
    rules = rules or TRADING_RULES
    threshold = rules['entry_threshold']
    stop_pct = rules['stop_loss_pct']
    activate_pct = rules.get('trail_activate_pct', 0.08)
    max_hold = rules['hold_days']

    rows = conn.execute("""
        SELECT id, ticker, signal_date, total_score, oos_score,
               price_at_signal, car_30d, source, sector, insider_name,
               market_cap_bucket
        FROM signals
        WHERE total_score >= ? AND outcome_30d_filled = 1
          AND car_30d IS NOT NULL AND price_at_signal IS NOT NULL
          AND price_at_signal > 0
        ORDER BY signal_date DESC
    """, (threshold,)).fetchall()

    trades = []
    _price_cache = {}
    skipped_cap = 0

    for r in rows:
        # Market cap filter (options liquidity)
        if not _passes_cap_filter(r['market_cap_bucket'], rules):
            skipped_cap += 1
            continue

        ticker = r['ticker']
        entry_date = r['signal_date']
        entry_price = float(r['price_at_signal'])
        score = float(r['total_score'])
        oos = float(r['oos_score']) if r['oos_score'] else None
        weight = _position_weight(score, rules)
        trail = _trail_pct(score, rules)

        # Load price data
        if ticker not in _price_cache:
            cache_path = PRICE_HISTORY_DIR / f"{ticker}.json"
            if cache_path.exists():
                try:
                    with open(cache_path) as f:
                        _price_cache[ticker] = json.load(f)
                except Exception:
                    _price_cache[ticker] = {}
            else:
                _price_cache[ticker] = {}

        prices = _price_cache[ticker]
        if not prices:
            trades.append({
                'id': r['id'], 'ticker': ticker, 'signal_date': entry_date,
                'score': round(score, 1), 'oos_score': round(oos, 1) if oos else None,
                'entry_price': round(entry_price, 2),
                'exit_price': round(entry_price * (1 + float(r['car_30d'])), 2),
                'exit_date': None, 'exit_reason': 'hold_expire',
                'return_pct': round(float(r['car_30d']), 4),
                'weight': weight,
                'weighted_return': round(float(r['car_30d']) * weight, 4),
                'source': r['source'], 'sector': r['sector'] or 'Unknown',
                'peak_gain': None,
            })
            continue

        # Get sorted trading days after entry
        hard_stop_price = entry_price * (1 + stop_pct)
        activate_price = entry_price * (1 + activate_pct)

        trading_days = sorted([d for d in prices.keys()
                               if d > entry_date and isinstance(prices[d], dict)])[:max_hold]

        exit_price = None
        exit_date = None
        exit_reason = 'hold_expire'
        hwm = entry_price  # high-water mark
        trailing_active = False

        for day in trading_days:
            bar = prices[day]
            low = bar.get('l', bar.get('c', entry_price))
            high = bar.get('h', bar.get('c', entry_price))
            close = bar.get('c', entry_price)

            # Update high-water mark
            if high > hwm:
                hwm = high

            # 1. Hard stop-loss (always active, from entry)
            if low <= hard_stop_price:
                exit_price = hard_stop_price
                exit_date = day
                exit_reason = 'stop_loss'
                break

            # 2. Activate trailing stop once gain exceeds threshold
            if not trailing_active and high >= activate_price:
                trailing_active = True

            # 3. Trailing stop: exit if price drops trail% below HWM
            if trailing_active:
                trail_stop_price = hwm * (1 - trail)
                if low <= trail_stop_price:
                    exit_price = trail_stop_price
                    exit_date = day
                    exit_reason = 'trailing_stop'
                    break

        # If no trigger, exit at last day's close
        if exit_price is None:
            if trading_days:
                last_day = trading_days[-1]
                exit_price = prices[last_day].get('c', entry_price)
                exit_date = last_day
            else:
                exit_price = entry_price * (1 + float(r['car_30d']))
                exit_date = None

        ret = (exit_price - entry_price) / entry_price
        peak_gain = (hwm - entry_price) / entry_price if hwm > entry_price else 0

        trades.append({
            'id': r['id'], 'ticker': ticker, 'signal_date': entry_date,
            'score': round(score, 1), 'oos_score': round(oos, 1) if oos else None,
            'entry_price': round(entry_price, 2),
            'exit_price': round(exit_price, 2),
            'exit_date': exit_date, 'exit_reason': exit_reason,
            'return_pct': round(ret, 4),
            'weight': weight,
            'weighted_return': round(ret * weight, 4),
            'source': r['source'], 'sector': r['sector'] or 'Unknown',
            'peak_gain': round(peak_gain, 4),
        })

    cap_note = f", skipped_cap={skipped_cap}" if skipped_cap else ""
    log.info(f"Simulated {len(trades)} trades (threshold={threshold}, "
             f"stop={stop_pct:.0%}, trail={activate_pct:.0%}, "
             f"min_cap={rules.get('min_market_cap', 'any')}{cap_note})")
    return trades


def compute_strategy_stats(trades: list[dict]) -> dict:
    """Compute strategy-level statistics from simulated trades."""
    import pandas as pd
    if not trades:
        return {}

    df = pd.DataFrame(trades)

    # Exit reason breakdown
    exit_counts = df['exit_reason'].value_counts().to_dict()

    # Weighted returns
    total_weight = df['weight'].sum()
    weighted_avg = (df['return_pct'] * df['weight']).sum() / total_weight if total_weight > 0 else 0

    wins = df[df['return_pct'] > 0]
    losses = df[df['return_pct'] <= 0]

    # Monthly cumulative (weighted)
    df['month'] = pd.to_datetime(df['signal_date']).dt.to_period('M').astype(str)
    monthly = []
    for m, grp in df.groupby('month'):
        m_weight = grp['weight'].sum()
        m_wavg = (grp['return_pct'] * grp['weight']).sum() / m_weight if m_weight > 0 else 0
        monthly.append({
            'month': str(m),
            'n': len(grp),
            'win_rate': round(float((grp['return_pct'] > 0).mean()), 3),
            'avg_return': round(float(grp['return_pct'].mean()), 4),
            'weighted_return': round(float(m_wavg), 4),
            'stops': int((grp['exit_reason'] == 'stop_loss').sum()),
            'targets': int((grp['exit_reason'] == 'take_profit').sum()),
        })
    monthly.sort(key=lambda x: x['month'], reverse=True)

    # Cumulative equity curve (equal-weight per-month)
    monthly_sorted = sorted(monthly, key=lambda x: x['month'])
    cum = 1.0
    for m in monthly_sorted:
        cum *= (1 + m['weighted_return'])
        m['cumulative'] = round(cum, 4)

    # Score tier breakdown
    tier_stats = []
    for label, lo, hi in [('85+', 85, 999), ('75-84', 75, 84.99), ('65-74', 65, 74.99)]:
        tier = df[(df['score'] >= lo) & (df['score'] <= hi)]
        if len(tier) > 0:
            tier_stats.append({
                'tier': label,
                'n': len(tier),
                'weight': round(float(tier['weight'].iloc[0]), 1),
                'win_rate': round(float((tier['return_pct'] > 0).mean()), 3),
                'avg_return': round(float(tier['return_pct'].mean()), 4),
                'avg_weighted': round(float(tier['weighted_return'].mean()), 4),
            })

    # Peak gain analysis — how much upside are we capturing?
    peak_stats = {}
    if 'peak_gain' in df.columns:
        peaks = df[df['peak_gain'].notna() & (df['peak_gain'] > 0)]
        if len(peaks) > 0:
            captured_ratio = peaks['return_pct'] / peaks['peak_gain']
            peak_stats = {
                'avg_peak': round(float(peaks['peak_gain'].mean()), 4),
                'avg_captured': round(float(peaks['return_pct'].mean()), 4),
                'capture_ratio': round(float(captured_ratio.mean()), 3),
                'runners_30pct': int((peaks['peak_gain'] >= 0.30).sum()),
                'runners_50pct': int((peaks['peak_gain'] >= 0.50).sum()),
            }

    return {
        'rules': TRADING_RULES,
        'total_trades': len(df),
        'win_rate': round(float((df['return_pct'] > 0).mean()), 3),
        'avg_return': round(float(df['return_pct'].mean()), 4),
        'weighted_avg_return': round(float(weighted_avg), 4),
        'avg_win': round(float(wins['return_pct'].mean()), 4) if len(wins) > 0 else 0,
        'avg_loss': round(float(losses['return_pct'].mean()), 4) if len(losses) > 0 else 0,
        'best': round(float(df['return_pct'].max()), 4),
        'worst': round(float(df['return_pct'].min()), 4),
        'total_cumulative': round(float(cum), 4) if monthly_sorted else 1.0,
        'exit_reasons': {k: int(v) for k, v in exit_counts.items()},
        'tiers': tier_stats,
        'peak_stats': peak_stats,
        'monthly': monthly[:24],
    }


def export_brain_data(conn: sqlite3.Connection) -> None:
    """Export brain_signals.json + brain_stats.json for the frontend."""
    log.info("=== Exporting Brain Data ===")
    now = datetime.now(tz=timezone.utc).isoformat()
    cur = conn.cursor()

    # ── brain_signals.json ──────────────────────────────────────────────────

    # Top signals from last 90 days by total_score, diversified (max 3 per ticker)
    MAX_PER_TICKER = 3
    EXPORT_LIMIT = 50
    cur.execute("""
        SELECT id, ticker, price_at_signal, signal_date, source, total_score,
               score_base, score_magnitude, score_converge, score_person,
               convergence_tier, has_convergence, representative, insider_name,
               insider_role, transaction_type, person_hit_rate_30d,
               person_trade_count, sector, car_30d,
               same_ticker_signals_7d, ml_confidence, predicted_car,
               party, chamber, trade_size_range,
               momentum_1m, momentum_3m, volume_spike,
               price_proximity_52wk, market_cap_bucket, vix_at_signal,
               days_to_earnings, disclosure_delay, relative_position_size,
               convergence_sources, person_avg_car_30d, sector_avg_car,
               days_since_last_buy, insider_buy_ratio_90d,
               avg_daily_volume, estimated_spread, liquidity_flag,
               oos_score
        FROM signals
        WHERE signal_date >= date('now', '-90 days')
          AND total_score IS NOT NULL
        ORDER BY total_score DESC
        LIMIT 200
    """)
    all_rows = cur.fetchall()
    cols = [d[0] for d in cur.description]

    # Diversify: max N signals per ticker, max M per sector
    MAX_PER_SECTOR = 8
    ticker_counts = {}
    sector_counts = {}
    rows = []
    overflow = []  # signals that exceeded sector cap — used to fill remaining slots
    for row in all_rows:
        r = dict(zip(cols, row))
        t = r['ticker']
        s = r['sector'] or 'Unknown'
        ticker_counts[t] = ticker_counts.get(t, 0) + 1
        if ticker_counts[t] > MAX_PER_TICKER:
            continue
        sector_counts[s] = sector_counts.get(s, 0) + 1
        if sector_counts[s] <= MAX_PER_SECTOR:
            rows.append(row)
        else:
            overflow.append(row)  # over sector cap — save for later fill
        if len(rows) >= EXPORT_LIMIT:
            break

    # If we haven't hit EXPORT_LIMIT, backfill from overflow (sector-capped signals)
    if len(rows) < EXPORT_LIMIT and overflow:
        remaining = EXPORT_LIMIT - len(rows)
        rows.extend(overflow[:remaining])

    # Compute diversification stats on exported signals
    exp_ticker_counts = {}
    exp_sector_counts = {}
    for row in rows:
        r = dict(zip(cols, row))
        t = r['ticker']
        s = r['sector'] or 'Unknown'
        exp_ticker_counts[t] = exp_ticker_counts.get(t, 0) + 1
        exp_sector_counts[s] = exp_sector_counts.get(s, 0) + 1
    sorted_t = sorted(exp_ticker_counts.values(), reverse=True)
    top5_pct = sum(sorted_t[:5]) / max(len(rows), 1) if sorted_t else 0
    diversification_stats = {
        'top_ticker': max(exp_ticker_counts, key=exp_ticker_counts.get) if exp_ticker_counts else None,
        'top_ticker_pct': round(sorted_t[0] / max(len(rows), 1), 3) if sorted_t else 0,
        'top_5_ticker_pct': round(top5_pct, 3),
        'unique_tickers': len(exp_ticker_counts),
        'unique_sectors': len(exp_sector_counts),
        'max_per_ticker_cap': MAX_PER_TICKER,
        'max_per_sector_cap': MAX_PER_SECTOR,
    }
    log.info(f"Export diversification: {diversification_stats['unique_tickers']} tickers, "
             f"{diversification_stats['unique_sectors']} sectors, "
             f"top-5 ticker concentration: {top5_pct:.0%}")

    # Prepare factor attribution data
    export_ids = set()
    rows_dicts = []
    for row in rows:
        r = dict(zip(cols, row))
        rows_dicts.append(r)
        export_ids.add(r['id'])

    # Load feature importance + raw features for factor computation
    weights = load_json(OPTIMAL_WEIGHTS) if OPTIMAL_WEIGHTS.exists() else {}
    feature_importance = weights.get('_feature_importance', {})
    # Load trader tiers for export
    _trader_tier_data = weights.get('_trader_tiers', {})
    _trader_tiers = _trader_tier_data.get('tiers', {})

    factors_map = {}
    try:
        from backtest.ml_engine import prepare_features_all
        X_enc, all_ids, _, _, _, X_raw = prepare_features_all(conn)
        if not X_raw.empty and feature_importance:
            factors_map = compute_signal_factors(X_raw, all_ids, export_ids, feature_importance)
    except Exception as e:
        log.warning(f"Could not compute signal factors: {e}")

    # Fetch live prices for exported tickers (batch via yfinance)
    live_prices = {}
    try:
        import yfinance as yf
        tickers = [
            t for t in {r['ticker'] for r in rows_dicts}
            if t and t.upper() not in ('NONE', 'NULL', 'N/A', '')
            and len(t) <= 5
        ]
        if tickers:
            live = yf.download(
                tickers, period='2d', group_by='ticker',
                auto_adjust=True, progress=False, threads=True
            )
            for t in tickers:
                try:
                    if len(tickers) == 1:
                        live_prices[t] = float(live['Close'].iloc[-1])
                    else:
                        live_prices[t] = float(live[t]['Close'].iloc[-1])
                except Exception:
                    pass
            log.info(f"Live prices fetched for {len(live_prices)}/{len(tickers)} tickers")
    except Exception as e:
        log.warning(f"Live price fetch failed: {e}")

    today = datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')

    # Regime context for all signals
    regime_ctx = get_regime_context()
    regime_mult = regime_ctx.get('multiplier', 1.0)
    log.info(f"Regime context: VIX={regime_ctx.get('current_vix')} → {regime_ctx.get('regime')} "
             f"(multiplier={regime_mult})")

    # Kelly sizing parameters from OOS data
    kelly_params = {'hit_rate': 0.55, 'avg_win': 0.05, 'avg_loss': -0.04}
    try:
        import pandas as pd
        kelly_df = pd.read_sql("""
            SELECT car_30d FROM signals
            WHERE oos_score >= 75 AND car_30d IS NOT NULL
        """, conn)
        if len(kelly_df) >= 50:
            winners = kelly_df[kelly_df.car_30d > 0].car_30d
            losers = kelly_df[kelly_df.car_30d <= 0].car_30d
            kelly_params['hit_rate'] = len(winners) / len(kelly_df)
            kelly_params['avg_win'] = float(winners.mean()) if len(winners) > 0 else 0.05
            kelly_params['avg_loss'] = float(losers.mean()) if len(losers) > 0 else -0.04
            log.info(f"Kelly params (OOS 75+): hit={kelly_params['hit_rate']:.1%} "
                     f"win={kelly_params['avg_win']:+.2%} loss={kelly_params['avg_loss']:+.2%}")
    except Exception as e:
        log.warning(f"Kelly param computation failed: {e}")

    signals_out = []
    for r in rows_dicts:
        price = r['price_at_signal'] or 0
        tx = (r['transaction_type'] or '').lower()
        is_sale = 'sale' in tx or 'disposition' in tx
        direction = 'short' if (r['source'] == 'edgar' and is_sale) else 'long'

        # Build note from available context
        note_parts = []
        if r['representative']:
            note_parts.append(r['representative'])
        if r['insider_name'] and r['insider_role']:
            note_parts.append(f"{r['insider_role']}")
        if r['has_convergence']:
            note_parts.append(f"Tier {r['convergence_tier']} convergence")
        if (r['same_ticker_signals_7d'] or 0) >= 3:
            note_parts.append("cluster")
        company = TICKER_NAMES.get(r['ticker'])
        if company:
            note_parts.append(company)
        note = ' · '.join(note_parts) if note_parts else ''

        person = r['representative'] or r['insider_name'] or ''

        # Smart targets from ML predictions
        targets = compute_smart_targets(
            price, r.get('predicted_car'), r.get('momentum_1m'), direction
        )

        # Score breakdown
        score_breakdown = None
        if r.get('score_base') is not None:
            score_breakdown = {
                'base': round(r['score_base'], 1),
                'magnitude': round(r['score_magnitude'], 1),
                'convergence': round(r['score_converge'], 1),
                'person': round(r['score_person'], 1),
            }

        # Signal context
        context = {
            'person': person,
            'insider_role': r['insider_role'] or '',
            'party': r.get('party') or '',
            'chamber': r.get('chamber') or '',
            'trade_size_range': r.get('trade_size_range') or '',
            'signal_date': r['signal_date'],
            'disclosure_delay': r.get('disclosure_delay'),
            'convergence_sources': r.get('convergence_sources') or '',
        }

        # Predicted move as percentage
        pred_car = r.get('predicted_car')
        predicted_move = round(pred_car * 100, 1) if pred_car is not None else None

        # Win probability from ml_confidence
        ml_conf = r.get('ml_confidence')
        win_probability = round(ml_conf * 100, 1) if ml_conf is not None else None

        # Live position tracking
        current_price = live_prices.get(r['ticker'])
        try:
            days_held = (datetime.strptime(today, '%Y-%m-%d') -
                         datetime.strptime(r['signal_date'], '%Y-%m-%d')).days
        except (ValueError, TypeError):
            days_held = 0
        unrealized_pnl = None
        if current_price and price and price > 0:
            unrealized_pnl = (current_price - price) / price

        signals_out.append({
            'ticker': r['ticker'],
            'price_at_signal': round(price, 2) if price else None,
            'signal_date': r['signal_date'],
            'source': r['source'],
            'dir': direction,
            'total_score': round(r['total_score'], 1) if r['total_score'] else None,
            'convergence_tier': r['convergence_tier'] or 0,
            'has_convergence': bool(r['has_convergence']),
            'note': note,
            'car_30d': round(r['car_30d'], 4) if r['car_30d'] is not None else None,
            'person': person,
            'person_hit_rate': round(r['person_hit_rate_30d'], 2) if r['person_hit_rate_30d'] is not None else None,
            'person_trade_count': r['person_trade_count'] or 0,
            'insider_role': r['insider_role'],
            'trader_tier': _trader_tiers.get(person, 'neutral'),
            'sector': r['sector'],
            'ml_confidence': round(ml_conf, 3) if ml_conf is not None else None,
            'predicted_car': predicted_move,
            # New enriched fields
            'score_breakdown': score_breakdown,
            'factors': factors_map.get(r['id'], []),
            'context': context,
            'win_probability': win_probability,
            'predicted_move': predicted_move,
            'time_horizon': '30d',  # model trains on car_30d
            # Smart targets
            'entry_lo': targets['entry_lo'],
            'entry_hi': targets['entry_hi'],
            'target1': targets['target1'],
            'target2': targets['target2'],
            'stop': targets['stop'],
            # Live position tracking (Task 5)
            'current_price': round(current_price, 2) if current_price else None,
            'unrealized_pnl_pct': round(unrealized_pnl, 4) if unrealized_pnl is not None else None,
            'days_held': days_held,
            'days_remaining': max(0, 30 - days_held),
            'position_status': 'ACTIVE' if days_held <= 30 else 'EXPIRED',
            'stop_loss_price': round(price * 0.88, 2) if price else None,
            'stop_loss_triggered': bool(current_price and price and current_price < price * 0.88),
            # Liquidity / cost (Task 4)
            'estimated_spread_pct': round(r.get('estimated_spread', 0) * 100, 3) if r.get('estimated_spread') else None,
            'avg_daily_volume': round(r.get('avg_daily_volume', 0)) if r.get('avg_daily_volume') else None,
            'liquidity_flag': r.get('liquidity_flag'),
            # OOS score (honest walk-forward)
            'oos_score': round(r.get('oos_score'), 1) if r.get('oos_score') is not None else None,
            # Market cap + tradability
            'market_cap': r.get('market_cap_bucket') or 'unknown',
            'tradeable': _passes_cap_filter(r.get('market_cap_bucket')),
            # Regime context
            'regime_multiplier': regime_mult,
            'regime_label': regime_ctx.get('regime', 'UNKNOWN'),
            # Kelly position sizing
            'kelly_size': compute_kelly_size(
                r.get('oos_score') or (r.get('total_score') or 50),
                kelly_params['hit_rate'], kelly_params['avg_win'],
                kelly_params['avg_loss'], regime_mult),
        })

    # Population stats for context (percentiles across all scored signals)
    import numpy as np
    pop_rows = cur.execute(
        "SELECT total_score, ml_confidence, predicted_car FROM signals "
        "WHERE total_score IS NOT NULL"
    ).fetchall()
    population = {}
    if pop_rows:
        scores_arr = np.array([r[0] or 0 for r in pop_rows])
        conf_arr = np.array([r[1] or 0 for r in pop_rows])
        car_arr = np.array([(r[2] or 0) * 100 for r in pop_rows])
        for name, arr in [('total_score', scores_arr), ('ml_confidence', conf_arr), ('predicted_car', car_arr)]:
            population[name] = {
                'p25': round(float(np.percentile(arr, 25)), 2),
                'p50': round(float(np.percentile(arr, 50)), 2),
                'p75': round(float(np.percentile(arr, 75)), 2),
                'p90': round(float(np.percentile(arr, 90)), 2),
            }

    # Exits: EDGAR sales from last 30 days
    cur.execute("""
        SELECT ticker, insider_name, insider_role, signal_date, transaction_type
        FROM signals
        WHERE source = 'edgar'
          AND signal_date >= date('now', '-30 days')
          AND (transaction_type LIKE '%sale%' OR transaction_type LIKE '%disposition%')
        ORDER BY signal_date DESC
        LIMIT 10
    """)
    exits_out = []
    for row in cur.fetchall():
        exits_out.append({
            'ticker': row[0],
            'insider_name': row[1] or '',
            'insider_role': row[2] or '',
            'signal_date': row[3],
            'transaction_type': row[4] or '',
            'note': '',
        })

    # Closed signals: recently expired (days_held 31-90) with outcomes
    closed_out = []
    cur.execute("""
        SELECT ticker, signal_date, total_score, oos_score,
               price_at_signal, car_30d, insider_name, insider_role,
               source, sector, days_to_earnings, vix_at_signal
        FROM signals
        WHERE signal_date <= date('now', '-30 days')
          AND signal_date >= date('now', '-90 days')
          AND car_30d IS NOT NULL
          AND total_score IS NOT NULL
        ORDER BY signal_date DESC
        LIMIT 50
    """)
    for row in cur.fetchall():
        r = dict(row)
        days_held = (datetime.now(tz=timezone.utc).date() -
                     datetime.strptime(r['signal_date'], '%Y-%m-%d').date()).days
        closed_out.append({
            'ticker': r['ticker'],
            'signal_date': r['signal_date'],
            'total_score': r['total_score'],
            'oos_score': round(r['oos_score'], 1) if r.get('oos_score') is not None else None,
            'price_at_signal': r['price_at_signal'],
            'car_30d': r['car_30d'],
            'person': r.get('insider_name', ''),
            'insider_role': r.get('insider_role', ''),
            'source': r.get('source', ''),
            'sector': r.get('sector', ''),
            'days_held': days_held,
            'position_status': 'EXPIRED',
        })

    brain_signals = {
        'generated': now,
        'signals': signals_out,
        'closed_signals': closed_out,
        'exits': exits_out,
        'population': population,
        'regime_context': regime_ctx,
        'kelly_params': {
            'hit_rate': round(kelly_params['hit_rate'], 3),
            'avg_win': round(kelly_params['avg_win'], 4),
            'avg_loss': round(kelly_params['avg_loss'], 4),
            'regime_multiplier': regime_mult,
        },
    }
    save_json(BRAIN_SIGNALS, brain_signals)
    log.info(f"Exported {len(signals_out)} signals + {len(closed_out)} closed + {len(exits_out)} exits → brain_signals.json")

    # ── portfolio_stats.json ──────────────────────────────────────────────
    # Strategy simulation: apply trading rules (threshold, stop-loss, take-profit, sizing)
    try:
        strategy_trades = simulate_trades(conn)
        strategy_stats = compute_strategy_stats(strategy_trades)
        # Recent closed trades for dashboard (last 200, with full detail)
        recent_trades = sorted(strategy_trades, key=lambda t: t['signal_date'], reverse=True)[:200]
    except Exception as e:
        log.warning(f"Strategy simulation failed: {e}")
        strategy_trades = []
        strategy_stats = {}
        recent_trades = []

    # Raw stats (all signals, no threshold) for comparison
    import pandas as pd
    ps_df = pd.read_sql("""
        SELECT ticker, signal_date, total_score, oos_score,
               price_at_signal, car_30d, source, sector
        FROM signals
        WHERE car_30d IS NOT NULL AND total_score IS NOT NULL
        ORDER BY signal_date DESC
    """, conn)
    raw_wins = ps_df[ps_df.car_30d > 0]
    raw_losses = ps_df[ps_df.car_30d <= 0]

    portfolio_stats = {
        'generated': now,
        'closed_signals': recent_trades,
        'strategy': strategy_stats,
        'raw': {
            'total_signals': len(ps_df),
            'win_rate': round(float((ps_df.car_30d > 0).mean()), 3) if len(ps_df) > 0 else 0,
            'avg_return': round(float(ps_df.car_30d.mean()), 4) if len(ps_df) > 0 else 0,
            'avg_win': round(float(raw_wins.car_30d.mean()), 4) if len(raw_wins) > 0 else 0,
            'avg_loss': round(float(raw_losses.car_30d.mean()), 4) if len(raw_losses) > 0 else 0,
        },
    }
    save_json(PORTFOLIO_STATS, portfolio_stats)
    n_stops = strategy_stats.get('exit_reasons', {}).get('stop_loss', 0)
    n_tp = strategy_stats.get('exit_reasons', {}).get('take_profit', 0)
    log.info(f"Exported portfolio_stats.json: {len(strategy_trades)} strategy trades "
             f"({n_stops} stops, {n_tp} targets), {len(ps_df)} raw signals")

    # ── brain_stats.json ────────────────────────────────────────────────────

    # Alpha: weighted avg CAR of signals with total_score >= 65
    cur.execute("""
        SELECT AVG(car_30d), COUNT(*)
        FROM signals
        WHERE total_score >= 65 AND outcome_30d_filled = 1 AND car_30d IS NOT NULL
    """)
    alpha_row = cur.fetchone()
    alpha_val = alpha_row[0] if alpha_row and alpha_row[0] is not None else 0
    alpha_n = alpha_row[1] if alpha_row else 0

    # Score tiers (IN-SAMPLE: scores from full-sample model applied to training data)
    # Real predictive performance is the walk-forward OOS IC/hit rate in optimal_weights
    score_tiers = []
    for tier_label, lo, hi in [('90+', 90, 999), ('80-89', 80, 89.99), ('65-79', 65, 79.99), ('<65', 0, 64.99)]:
        cur.execute("""
            SELECT COUNT(*),
                   AVG(CASE WHEN car_30d > 0 THEN 1.0 ELSE 0.0 END),
                   AVG(car_30d)
            FROM signals
            WHERE total_score >= ? AND total_score <= ?
              AND outcome_30d_filled = 1 AND car_30d IS NOT NULL
        """, (lo, hi))
        tr = cur.fetchone()
        if tr and tr[0] > 0:
            score_tiers.append({
                'tier': tier_label,
                'n': tr[0],
                'hit_rate': round(tr[1], 2) if tr[1] is not None else None,
                'avg_car_30d': round(tr[2], 4) if tr[2] is not None else None,
                'note': 'in-sample (training data)',
            })

    # Sectors
    cur.execute("""
        SELECT sector, COUNT(*), AVG(car_30d),
               AVG(CASE WHEN car_30d > 0 THEN 1.0 ELSE 0.0 END)
        FROM signals
        WHERE outcome_30d_filled = 1 AND car_30d IS NOT NULL AND sector IS NOT NULL
        GROUP BY sector
        ORDER BY AVG(car_30d) DESC
    """)
    sectors = [{'sector': r[0], 'n': r[1], 'avg_car_30d': round(r[2], 4), 'hit_rate': round(r[3], 2)}
               for r in cur.fetchall()]

    # Congressional heatmap: top tickers by trade count in last 30 days
    cur.execute("""
        SELECT ticker, COUNT(*) as cnt
        FROM signals
        WHERE source = 'congress' AND signal_date >= date('now', '-30 days')
        GROUP BY ticker
        ORDER BY cnt DESC
        LIMIT 16
    """)
    congress_heatmap = []
    for r in cur.fetchall():
        congress_heatmap.append({
            'ticker': r[0],
            'name': TICKER_NAMES.get(r[0], r[0]),
            'count': r[1],
            'heat': min(r[1], 6),  # cap heat level at 6
        })

    # KPIs
    cur.execute("SELECT MAX(total_score) FROM signals WHERE signal_date >= date('now', '-30 days')")
    top_score_row = cur.fetchone()
    top_score = round(top_score_row[0]) if top_score_row and top_score_row[0] else 0

    cur.execute("""
        SELECT ticker FROM signals
        WHERE signal_date >= date('now', '-30 days') AND total_score = (
            SELECT MAX(total_score) FROM signals WHERE signal_date >= date('now', '-30 days')
        ) LIMIT 3
    """)
    top_tickers = ' / '.join(r[0] for r in cur.fetchall())

    cur.execute("SELECT COUNT(*) FROM signals WHERE signal_date >= date('now', '-30 days') AND total_score >= 80")
    exceptional_count = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(DISTINCT ticker) FROM signals
        WHERE signal_date >= date('now', '-30 days') AND same_ticker_signals_7d >= 3
    """)
    cluster_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM signals WHERE source='congress' AND signal_date >= date('now', '-30 days')")
    congress_flags = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM signals WHERE signal_date >= date('now', '-30 days') AND has_convergence = 1")
    convergence_count = cur.fetchone()[0]

    # Insider sectors
    cur.execute("""
        SELECT sector, COUNT(*), AVG(car_30d)
        FROM signals
        WHERE source = 'edgar' AND outcome_30d_filled = 1
          AND car_30d IS NOT NULL AND sector IS NOT NULL
        GROUP BY sector
        ORDER BY AVG(car_30d) DESC
    """)
    insider_sectors = [{'sector': r[0], 'n': r[1], 'avg_car': round(r[2], 4)}
                       for r in cur.fetchall()]

    # ML stats from optimal_weights.json
    ml_stats = {'oos_ic': None, 'oos_hit_rate': None, 'n_folds': None}
    if OPTIMAL_WEIGHTS.exists():
        w = load_json(OPTIMAL_WEIGHTS)
        if '_oos_ic' in w:
            ml_stats = {
                'oos_ic': w.get('_oos_ic'),
                'oos_hit_rate': w.get('_oos_hit_rate'),
                'n_folds': w.get('_n_folds'),
                'pos_folds': w.get('_pos_folds'),
                'ic_t_stat': w.get('_ic_t_stat'),
                'ic_p_value': w.get('_ic_p_value'),
                'sharpe_annual': w.get('_sharpe_annual'),
                'sortino_ratio': w.get('_sortino_ratio'),
                'brier_skill_score': w.get('_brier_skill_score'),
                'q5_q1_spread': w.get('_q5_q1_spread'),
                'profit_factor': w.get('_profit_factor'),
                'strategy_metrics': w.get('_strategy_metrics'),
            }

    # Committees — derive from REP_COMMITTEES mapping
    cur.execute("""
        SELECT representative, COUNT(*) as cnt,
               AVG(CASE WHEN car_30d > 0 THEN 1.0 ELSE 0.0 END) as hit_rate
        FROM signals
        WHERE source = 'congress' AND representative IS NOT NULL
          AND outcome_30d_filled = 1 AND car_30d IS NOT NULL
        GROUP BY representative
    """)
    committee_agg: dict[str, dict] = {}
    for rep, cnt, hr in cur.fetchall():
        comm = REP_COMMITTEES.get(rep)
        if not comm:
            continue
        if comm not in committee_agg:
            committee_agg[comm] = {'n_trades': 0, 'hit_sum': 0.0, 'hit_n': 0}
        committee_agg[comm]['n_trades'] += cnt
        if hr is not None:
            committee_agg[comm]['hit_sum'] += hr * cnt
            committee_agg[comm]['hit_n'] += cnt
    committees = []
    for name in sorted(committee_agg, key=lambda k: committee_agg[k]['n_trades'], reverse=True):
        agg = committee_agg[name]
        match_rate = round(agg['hit_sum'] / agg['hit_n'], 2) if agg['hit_n'] > 0 else 0
        committees.append({'name': name, 'n_trades': agg['n_trades'], 'match_rate': match_rate})

    brain_stats = {
        'generated': now,
        'alpha': {
            'value': round(alpha_val, 4),
            'label': f"+{alpha_val*100:.1f}%" if alpha_val >= 0 else f"{alpha_val*100:.1f}%",
            'horizon': '30d',
            'n_signals': alpha_n,
        },
        'score_tiers': score_tiers,
        'sectors': sectors,
        'committees': committees,
        'congress_heatmap': congress_heatmap,
        'kpis': {
            'top_score': top_score,
            'top_score_tickers': top_tickers,
            'exceptional_count': exceptional_count,
            'cluster_count': cluster_count,
            'congress_flags': congress_flags,
            'convergence_count': convergence_count,
        },
        'insider_sectors': insider_sectors,
        'ml': ml_stats,
    }

    # Trader tiers leaderboard from optimal_weights
    if _trader_tier_data and _trader_tier_data.get('leaderboard'):
        brain_stats['trader_tiers'] = _trader_tier_data['leaderboard']

    # Sector distribution in exported signals
    export_sector_counts = {}
    for s in signals_out:
        sec = s.get('sector') or 'Unknown'
        export_sector_counts[sec] = export_sector_counts.get(sec, 0) + 1
    if export_sector_counts:
        brain_stats['sector_distribution'] = dict(
            sorted(export_sector_counts.items(), key=lambda x: x[1], reverse=True)
        )

    # Ticker distribution in exported signals
    export_ticker_counts = {}
    for s in signals_out:
        t = s['ticker']
        export_ticker_counts[t] = export_ticker_counts.get(t, 0) + 1
    if export_ticker_counts:
        sorted_tickers = sorted(export_ticker_counts.items(), key=lambda x: x[1], reverse=True)
        brain_stats['ticker_distribution'] = {
            'unique_tickers_in_export': len(export_ticker_counts),
            'max_signals_per_ticker': sorted_tickers[0][1] if sorted_tickers else 0,
            'top_5_tickers_by_signal_count': [
                {'ticker': t, 'count': c} for t, c in sorted_tickers[:5]
            ],
        }

    # Pruning log: harmful feature values from feature_stats
    harmful_rows = cur.execute(
        "SELECT feature_name, feature_value, avg_car_30d, n_observations "
        "FROM feature_stats WHERE avg_car_30d < -0.02 AND n_observations >= 30 "
        "ORDER BY avg_car_30d ASC"
    ).fetchall()
    brain_stats['pruning_log'] = [
        {'feature': r[0], 'value': r[1], 'avg_car_30d': round(r[2], 4), 'n': r[3]}
        for r in harmful_rows
    ]

    # Regime stats
    regime_rows = cur.execute("""
        SELECT market_regime, COUNT(*) as n,
               AVG(CASE WHEN car_30d > 0 THEN 1.0 ELSE 0.0 END) as hit_rate,
               AVG(car_30d) as avg_car
        FROM signals
        WHERE outcome_30d_filled = 1 AND car_30d IS NOT NULL
          AND market_regime IS NOT NULL
        GROUP BY market_regime
    """).fetchall()
    regime_dist = {r['market_regime']: r['n'] for r in regime_rows}
    regime_hit = {r['market_regime']: round(r['hit_rate'], 3) for r in regime_rows}
    latest_vix_row = cur.execute(
        "SELECT vix_at_signal FROM signals WHERE vix_at_signal IS NOT NULL "
        "ORDER BY signal_date DESC LIMIT 1"
    ).fetchone()
    brain_stats['regime'] = {
        'current_regime': _vix_to_regime(latest_vix_row['vix_at_signal'] if latest_vix_row else None),
        'distribution': regime_dist,
        'hit_rates': regime_hit,
    }

    # Features pruned from ML training (historical record)
    brain_stats['features_pruned_history'] = {
        'pruned': [
            {'feature': 'source', 'date': '2026-02-28', 'reason': '0.16% importance', 'avg_car': None},
            {'feature': 'trade_pattern', 'date': '2026-02-28', 'reason': '0.40% importance, 31% fill', 'avg_car': None},
            {'feature': 'convergence_tier', 'date': '2026-03-01', 'reason': '<1% importance, 3+ consecutive runs', 'avg_car': None},
            {'feature': 'has_convergence', 'date': '2026-03-01', 'reason': '<1% importance, 3+ consecutive runs', 'avg_car': None},
            {'feature': 'days_to_catalyst', 'date': '2026-03-01', 'reason': '<1% importance, 3+ consecutive runs', 'avg_car': None},
            {'feature': 'relative_position_size', 'date': '2026-03-01', 'reason': '<1% importance, 3+ consecutive runs', 'avg_car': None},
            {'feature': 'cluster_velocity', 'date': '2026-03-01', 'reason': '<1% importance, 3+ consecutive runs', 'avg_car': None},
        ],
        'current_feature_count': 23,
    }

    # Fill-rate gate: candidate feature readiness
    try:
        from backtest.ml_engine import get_active_features, FILL_GATE_THRESHOLD
        _, _, fill_report = get_active_features(conn)
        brain_stats['feature_fill_gate'] = {
            'threshold': FILL_GATE_THRESHOLD,
            'candidates': fill_report,
        }
    except ImportError:
        pass

    # Strategy metrics: market-adjusted alpha by score band
    try:
        alpha_metrics = compute_alpha_metrics(conn)
        if alpha_metrics:
            brain_stats['strategy_metrics'] = alpha_metrics
    except Exception:
        pass

    # Diversification stats from exported signal selection
    brain_stats['signal_concentration'] = diversification_stats

    save_json(BRAIN_STATS, brain_stats)
    log.info(f"Exported brain stats → brain_stats.json")


# ── Daily Pipeline ───────────────────────────────────────────────────────────

def collect_missing_prices(conn: sqlite3.Connection) -> int:
    """Fetch yfinance data for tickers that have signals but no price file.
    Returns count of tickers collected."""
    try:
        from backtest.collect_prices import collect_ticker
    except ImportError:
        log.warning("collect_prices not available — skipping price collection")
        return 0

    rows = conn.execute("SELECT DISTINCT ticker FROM signals").fetchall()
    collected = 0
    for row in rows:
        ticker = row['ticker']
        price_file = PRICE_HISTORY_DIR / f"{ticker}.json"
        if not price_file.exists():
            try:
                if collect_ticker(ticker):
                    collected += 1
            except Exception as e:
                log.warning(f"Price collection failed for {ticker}: {e}")
    return collected


def run_daily(conn: sqlite3.Connection) -> None:
    """Daily pipeline: ingest new signals + backfill outcomes."""
    log.info("=== ALE Daily Pipeline ===")
    steps = {}  # Track step counts for pipeline monitoring

    # 1. Ingest new signals
    c_count = ingest_congress_feed(conn)
    e_count = ingest_edgar_feed(conn)
    f_count = ingest_13f_feed(conn)
    steps['ingest'] = c_count + e_count + f_count
    log.info(f"Ingested: {c_count} congress + {e_count} EDGAR + {f_count} 13F signals")

    # 2. Collect prices for any new tickers missing price history
    new_prices = collect_missing_prices(conn)
    steps['price_collect'] = new_prices or 0
    if new_prices:
        log.info(f"Collected prices for {new_prices} new tickers")

    # 2b. Backfill entry prices from price cache (recovers older signals)
    price_filled = backfill_entry_prices(conn)
    steps['price_backfill'] = price_filled
    if price_filled:
        log.info(f"Backfilled entry prices for {price_filled} signals")

    # 3. Update aggregate features (incremental: last 60 days only)
    since = (datetime.now(tz=timezone.utc) - timedelta(days=60)).strftime('%Y-%m-%d')
    agg = update_aggregate_features(conn, since_date=since)
    steps['aggregate'] = agg
    log.info(f"Updated aggregate features for {agg} ticker-date pairs")

    # 4. Backfill outcomes
    spy_index = load_price_index('SPY')
    filled = backfill_outcomes(conn, spy_index)
    steps['outcomes'] = filled
    log.info(f"Backfilled outcomes for {filled} signals")

    # 4b. Enrich SPY returns + market-adjusted CARs
    spy_enriched = enrich_spy_returns(conn)
    adj_enriched = enrich_market_adj_returns(conn)
    steps['spy_adj'] = (spy_enriched or 0) + (adj_enriched or 0)
    if spy_enriched or adj_enriched:
        log.info(f"SPY returns: {spy_enriched} new, market-adj CARs: {adj_enriched} new")

    # 5. Update person track records
    person_updated = update_person_track_records(conn)
    steps['person'] = person_updated
    log.info(f"Updated person track records for {person_updated} signals")

    # 6. Enrich price-based and research-backed features
    enriched = enrich_signal_features(conn)
    steps['enrich'] = enriched
    log.info(f"Enriched {enriched} signal features")

    # 7. Enrich market context (VIX, yield curve, credit spread from FRED)
    market_enriched = enrich_market_context(conn)
    steps['market_ctx'] = market_enriched
    log.info(f"Enriched {market_enriched} signals with market context")

    # 7b. Enrich liquidity features (ADV, spread estimates)
    liq_enriched = enrich_liquidity_features(conn)
    steps['liquidity'] = liq_enriched
    if liq_enriched:
        log.info(f"Enriched liquidity for {liq_enriched} signals")

    # 8. Generate dashboard + diagnostics
    generate_dashboard(conn)
    log.info(f"Dashboard saved to {ALE_DASHBOARD}")
    generate_analysis_report(conn)
    generate_diagnostics_html(conn)

    # 9. Score all signals with ML models
    scored = 0
    try:
        scored = score_all_signals(conn) or 0
        log.info(f"Scored {scored} signals with ML models")
    except Exception as e:
        log.warning(f"Signal scoring failed: {e}")
    steps['score'] = scored

    # 10. Auto-export brain data for frontend
    export_brain_data(conn)
    steps['export'] = 1

    # 11. Enrichment verification — flag steps that returned 0 unexpectedly
    zero_steps = [k for k, v in steps.items() if v == 0 and k not in ('ingest', 'price_collect', 'price_backfill')]
    if zero_steps:
        log.warning(f"Pipeline steps returned 0: {', '.join(zero_steps)} — verify data feeds")

    # 12. Log brain run with step counts
    log_brain_run(conn, 'daily', step_counts=steps)


def run_analyze(conn: sqlite3.Connection) -> None:
    """Weekly analysis: compute feature stats + ML walk-forward + update weights."""
    log.info("=== ALE Feature Analysis ===")

    # 0. Pre-analysis data cleanup: clip CARs + enrich missing features
    spy_index = load_price_index('SPY')
    backfill_outcomes(conn, spy_index)  # clips out-of-bounds CARs + fills gaps
    enrich_signal_features(conn)        # fills sector_avg_car, market_cap_bucket, roles

    # 1. Compute feature stats
    stats = compute_feature_stats(conn)
    if not stats:
        log.warning("No data to analyze yet.")
        return

    # 2. Generate updated weights from feature stats
    weights = generate_weights_from_stats(conn)

    # 2b. Fill-rate gate: auto-promote candidate features with sufficient data
    fill_report = {}
    try:
        from backtest.ml_engine import (get_active_features, FEATURE_COLUMNS,
                                        CATEGORICAL_FEATURES, CANDIDATE_FEATURES,
                                        FILL_GATE_THRESHOLD)
        active_feats, active_cats, fill_report = get_active_features(conn)
        promoted = [c for c, r in fill_report.items() if r['status'] == 'active']
        waiting = [c for c, r in fill_report.items() if r['status'] == 'candidate']
        log.info(f"Fill-rate gate: {len(active_feats)} active features "
                 f"({len(promoted)} promoted from candidates, {len(waiting)} waiting)")
        for col, info in fill_report.items():
            log.debug(f"  {col}: fill={info['fill_rate']:.1%} → {info['status']}")
        # Patch module-level lists for this analysis session
        FEATURE_COLUMNS.clear()
        FEATURE_COLUMNS.extend(active_feats)
        CATEGORICAL_FEATURES.clear()
        CATEGORICAL_FEATURES.extend(active_cats)
    except ImportError:
        pass

    # 3. Run walk-forward ML training (force retrain — clear model cache)
    if MODELS_CACHE.exists():
        MODELS_CACHE.unlink()
    ml_result = None
    try:
        from backtest.ml_engine import walk_forward_train
        ml_result = walk_forward_train(conn)
        top5_str = ""
        if ml_result.feature_importance:
            top5 = list(ml_result.feature_importance.items())[:5]
            top5_str = f" | top: {', '.join(f'{k}={v:.3f}' for k, v in top5)}"
        log.info(f"Classification: {ml_result.n_folds} folds, IC={ml_result.oos_ic:.4f}, "
                 f"hit={ml_result.oos_hit_rate:.1%}{top5_str}")
        # Persist OOS predictions to DB (honest walk-forward holdout scores)
        if hasattr(ml_result, 'oos_predictions') and ml_result.oos_predictions:
            # Clear previous OOS scores (full retrain replaces all)
            conn.execute("UPDATE signals SET oos_score = NULL, oos_fold = NULL")
            # Each signal appears in exactly one fold's holdout set
            # prob is 0-1 from classifier ensemble; scale to 0-100
            updates = []
            for sid, prob, _car in ml_result.oos_predictions:
                oos_score_100 = round(float(prob) * 100, 2)
                updates.append((oos_score_100, sid))
            conn.executemany(
                "UPDATE signals SET oos_score = ? WHERE id = ?",
                updates
            )
            conn.commit()
            log.info(f"Stored {len(updates)} OOS predictions in DB (oos_score column)")
    except ImportError:
        log.warning("ML dependencies not installed (scikit-learn, lightgbm) — skipping ML training")
    except Exception as e:
        log.warning(f"ML training failed: {e}")
        import traceback
        traceback.print_exc()

    # 3b. Run walk-forward regression (for diagnostics + magnitude validation)
    reg_result = None
    try:
        from backtest.ml_engine import walk_forward_regression
        reg_result = walk_forward_regression(conn)
        log.info(f"Regression:     {reg_result.n_folds} folds, IC={reg_result.oos_ic:.4f}, "
                 f"RMSE={reg_result.oos_rmse:.4f}")
    except ImportError:
        log.warning("ML dependencies not installed — skipping regression")
    except Exception as e:
        log.warning(f"Regression training failed: {e}")
        import traceback
        traceback.print_exc()

    # 3z. Checkpoint before weight update — IC regression guard
    prev_ic = None
    if OPTIMAL_WEIGHTS.exists():
        try:
            prev_data = json.loads(OPTIMAL_WEIGHTS.read_text())
            prev_ic = prev_data.get('_oos_ic')
        except Exception:
            pass
    backup_brain_exports(label="pre_analyze")

    # 4. Save weights — only update if ML outperforms current by >5%
    threshold = weights.pop('_optimal_threshold', 65)
    output = {
        **weights,
        "generated": datetime.now(tz=timezone.utc).isoformat(),
        "optimal_threshold": threshold,
        "method": "feature_importance",
        "stats": {
            "note": "generated by ALE feature analysis",
        },
    }
    if ml_result and ml_result.n_folds > 0:
        output['_oos_ic'] = ml_result.oos_ic
        output['_oos_hit_rate'] = ml_result.oos_hit_rate
        output['_n_folds'] = ml_result.n_folds
        # Store positive fold count from walk-forward folds
        if hasattr(ml_result, 'folds') and ml_result.folds:
            output['_pos_folds'] = sum(1 for f in ml_result.folds if (f.get('ic') if isinstance(f, dict) else getattr(f, 'ic', 0)) > 0)
        # OOS score tiers: compute real predictive performance by probability bucket
        if hasattr(ml_result, 'oos_predictions') and ml_result.oos_predictions:
            oos_tiers = {}
            for _sid, prob, car in ml_result.oos_predictions:
                # Map OOS probability to approximate score bucket
                # prob > 0.7 ≈ high confidence, 0.55-0.7 ≈ moderate, < 0.55 ≈ low
                if prob >= 0.70:
                    bucket = 'high_confidence'
                elif prob >= 0.55:
                    bucket = 'moderate'
                else:
                    bucket = 'low_confidence'
                if bucket not in oos_tiers:
                    oos_tiers[bucket] = {'n': 0, 'hits': 0, 'sum_car': 0.0}
                oos_tiers[bucket]['n'] += 1
                oos_tiers[bucket]['hits'] += 1 if car > 0 else 0
                oos_tiers[bucket]['sum_car'] += car
            oos_score_tiers = {}
            for bucket, data in oos_tiers.items():
                oos_score_tiers[bucket] = {
                    'n': data['n'],
                    'hit_rate': round(data['hits'] / data['n'], 4) if data['n'] > 0 else 0,
                    'avg_car': round(data['sum_car'] / data['n'], 6) if data['n'] > 0 else 0,
                }
            output['_oos_score_tiers'] = oos_score_tiers
            tier_strs = [f"{k}: n={v['n']} hit={v['hit_rate']:.1%}" for k, v in oos_score_tiers.items()]
            log.debug(f"OOS score tiers: {', '.join(tier_strs)}")
        output['_feature_importance'] = ml_result.feature_importance
        # Statistical rigor metrics
        output['_ic_t_stat'] = ml_result.ic_t_stat
        output['_ic_p_value'] = ml_result.ic_p_value
        output['_ic_std'] = ml_result.ic_std
        output['_sharpe_annual'] = ml_result.sharpe_annual
        output['_sortino_ratio'] = ml_result.sortino_ratio
        output['_information_ratio'] = ml_result.information_ratio
        output['_brier_skill_score'] = ml_result.brier_skill_score
        output['_q5_q1_spread'] = ml_result.q5_q1_spread
        output['_profit_factor'] = ml_result.profit_factor
        # Log significance assessment
        sig = "SIGNIFICANT" if ml_result.ic_p_value < 0.05 else "not significant"
        log.debug(f"IC significance: t={ml_result.ic_t_stat:.2f}, p={ml_result.ic_p_value:.4f} ({sig})")
        log.debug(f"Risk-adjusted:  Sharpe={ml_result.sharpe_annual:.2f}, "
                  f"Sortino={ml_result.sortino_ratio:.2f}, "
                  f"BSS={ml_result.brier_skill_score:.3f}")
        # Use ML method whenever we have a positive IC
        if ml_result.oos_ic > 0:
            output['method'] = 'walk_forward_ensemble'
            log.debug(f"Weights method: walk_forward_ensemble (IC {ml_result.oos_ic:.4f})")
        else:
            log.debug(f"Weights method: feature_importance (IC {ml_result.oos_ic:.4f} not positive)")

    # 4b. Optimize score coefficients via grid search
    try:
        coeffs = optimize_score_coefficients(conn)
        if coeffs:
            output['_score_coefficients'] = coeffs
    except Exception as e:
        log.warning(f"Coefficient optimization failed: {e}")

    # 4c. Log feature importance history for trend analysis
    if ml_result and ml_result.feature_importance:
        try:
            log_feature_importance_history(conn, ml_result.feature_importance)
        except Exception as e:
            log.warning(f"Feature importance history log failed: {e}")

    # 4d. Compute source quality multipliers from historical CAR by source
    try:
        source_quality = _compute_source_quality(conn)
        if source_quality:
            output['_source_quality'] = source_quality
            log.debug(f"Source quality: edgar={source_quality.get('edgar', 1.0):.3f}, "
                     f"congress={source_quality.get('congress', 1.0):.3f}, "
                     f"convergence={source_quality.get('convergence', 1.0):.3f}")
    except Exception as e:
        log.warning(f"Source quality computation failed: {e}")

    # 4e. Compute role quality bonuses from historical CAR by insider_role
    try:
        role_quality = _compute_role_quality(conn)
        if role_quality:
            output['_role_quality'] = role_quality
            log.debug(f"Role quality: {len(role_quality)} roles with learned bonuses")
    except Exception as e:
        log.warning(f"Role quality computation failed: {e}")

    # 4f. Compute trader quality tiers (elite/good/neutral/fade)
    try:
        trader_tiers = _compute_trader_tiers(conn)
        output['_trader_tiers'] = trader_tiers
        log.debug(f"Trader tiers computed: {len(trader_tiers.get('tiers', {}))} traders classified")
    except Exception as e:
        log.warning(f"Trader tier computation failed: {e}")

    # 4g. Compute regime stats and store caps
    try:
        regime_stats = _compute_regime_stats(conn)
        if regime_stats:
            output['_regime_caps'] = regime_stats['caps']
            output['_regime_stats'] = regime_stats['stats']
            log.debug(f"Regime stats: {regime_stats['stats']}")
    except Exception as e:
        log.warning(f"Regime stats computation failed: {e}")

    # 4h. Enrich SPY returns + compute market-adjusted alpha
    try:
        enrich_spy_returns(conn)
        enrich_market_adj_returns(conn)
        alpha_metrics = compute_alpha_metrics(conn)
        if alpha_metrics:
            output['_strategy_metrics'] = alpha_metrics
            a80 = alpha_metrics.get('alpha_80plus')
            a_all = alpha_metrics.get('alpha_all_signals')
            beta = alpha_metrics.get('beta_vs_spy')
            sharpe_adj = alpha_metrics.get('sharpe_market_adjusted_80plus')
            if a80 is not None:
                log.debug(f"Alpha (80+ signals, market-adjusted): {a80:+.4f}/signal "
                         f"({a80*12:+.2%} annualized)")
            if a_all is not None:
                log.debug(f"Alpha (all signals, market-adjusted): {a_all:+.4f}/signal")
            if beta is not None:
                log.debug(f"Beta vs SPY: {beta:.3f}")
            if sharpe_adj is not None:
                log.debug(f"Sharpe (80+ market-adjusted): {sharpe_adj:.2f}")
    except Exception as e:
        log.warning(f"Alpha computation failed: {e}")

    save_json(OPTIMAL_WEIGHTS, output)

    # 4z. IC regression guard — auto-rollback if IC dropped >10%
    new_ic = output.get('_oos_ic')
    if prev_ic is not None and new_ic is not None and prev_ic > 0:
        ic_drop = (prev_ic - new_ic) / prev_ic
        if ic_drop > 0.10:
            log.warning(f"IC REGRESSION: {prev_ic:.4f} → {new_ic:.4f} ({ic_drop:.1%} drop). "
                        f"Rolling back to pre-analyze checkpoint.")
            result = rollback_checkpoint()
            if result.get('restored'):
                log.warning(f"Auto-rollback complete: restored {result['checkpoint']}")
            return
        elif ic_drop > 0.05:
            log.warning(f"IC dipped {ic_drop:.1%} ({prev_ic:.4f} → {new_ic:.4f}) — monitoring")

    # 5. Update dashboard (pass ml_result for ML metrics)
    generate_dashboard(conn, ml_result=ml_result)

    # 6. Generate diagnostics
    generate_analysis_report(conn, ml_result=ml_result)
    generate_diagnostics_html(conn, ml_result=ml_result)

    # 7. Score all signals with full-sample ML models
    try:
        score_all_signals(conn)
    except Exception as e:
        log.warning(f"Signal scoring failed: {e}")
        import traceback
        traceback.print_exc()

    # 8. Residual analysis — surface patterns in prediction errors
    try:
        residuals = analyze_residuals(conn)
    except Exception as e:
        log.warning(f"Residual analysis failed: {e}")
        residuals = None

    # 8b. Generate signal hypotheses (autonomous improvement loop)
    try:
        hypotheses = generate_signal_hypotheses(conn)
        n_total = (len(hypotheses.get('high_residual_tickers', [])) +
                   len(hypotheses.get('feature_interactions', [])) +
                   len(hypotheses.get('regime_gaps', [])))
        log.debug(f"Generated {n_total} signal hypotheses → {SIGNAL_HYPOTHESES}")
    except Exception as e:
        log.warning(f"Hypothesis generation failed: {e}")

    # 8c. Signal intelligence — best/worst signal profiling
    try:
        compute_signal_intelligence(conn)
    except Exception as e:
        log.warning(f"Signal intelligence failed: {e}")

    # 9. Auto-export brain data for frontend
    export_brain_data(conn)

    # 10. Log brain run + self-check
    ic = ml_result.oos_ic if ml_result and ml_result.n_folds > 0 else None
    hr = ml_result.oos_hit_rate if ml_result and ml_result.n_folds > 0 else None
    log_brain_run(conn, 'analyze', oos_ic=ic, oos_hit_rate=hr)
    health = run_self_check(conn)

    # 11. Add residual patterns to health output
    if residuals and residuals.get('patterns'):
        health.setdefault('suggestions', []).extend(residuals['patterns'])
        health['residuals'] = residuals
        save_json(BRAIN_HEALTH, health)


# ── DB Health ────────────────────────────────────────────────────────────────

def db_health_check(conn: sqlite3.Connection) -> dict:
    """Check database health: sizes, indexes, integrity, WAL mode."""
    cur = conn.cursor()
    result = {}

    # Table row counts
    tables = {}
    for tbl in ('signals', 'feature_stats', 'weight_history',
                'feature_importance_history', 'brain_runs'):
        try:
            cnt = cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            tables[tbl] = cnt
        except Exception:
            tables[tbl] = -1
    result['tables'] = tables

    # DB file size
    db_path = Path(str(conn.execute("PRAGMA database_list").fetchone()[2]))
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        result['db_size_mb'] = round(size_mb, 2)
    else:
        result['db_size_mb'] = 0

    # Journal mode
    journal = cur.execute("PRAGMA journal_mode").fetchone()[0]
    result['journal_mode'] = journal

    # Index count
    indexes = cur.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='index'"
    ).fetchone()[0]
    result['index_count'] = indexes

    # Quick integrity check
    try:
        integrity = cur.execute("PRAGMA quick_check").fetchone()[0]
        result['integrity'] = integrity  # 'ok' if healthy
    except Exception as e:
        result['integrity'] = str(e)

    # NULL rates for key columns
    total = tables.get('signals', 0)
    if total > 0:
        null_rates = {}
        for col in ('car_30d', 'total_score', 'sector', 'spy_return_30d'):
            try:
                nulls = cur.execute(
                    f"SELECT COUNT(*) FROM signals WHERE {col} IS NULL"
                ).fetchone()[0]
                null_rates[col] = round(nulls / total, 3)
            except Exception:
                pass
        result['null_rates'] = null_rates

    # Outcome fill rate
    if total > 0:
        filled = cur.execute(
            "SELECT COUNT(*) FROM signals WHERE outcome_30d_filled = 1"
        ).fetchone()[0]
        result['outcome_fill_rate'] = round(filled / total, 3)

    # Date range
    try:
        row = cur.execute(
            "SELECT MIN(signal_date), MAX(signal_date) FROM signals"
        ).fetchone()
        if row and row[0]:
            result['date_range'] = {'min': row[0], 'max': row[1]}
    except Exception:
        pass

    return result


# ── Checkpoint / Rollback ────────────────────────────────────────────────────

def backup_brain_exports(label: str = "") -> str:
    """Snapshot brain_signals, brain_stats, optimal_weights, brain_health to timestamped checkpoint.

    Returns checkpoint directory path.
    """
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_{label}" if label else ts
    cp_dir = CHECKPOINTS_DIR / name
    cp_dir.mkdir(parents=True, exist_ok=True)

    files_to_backup = [BRAIN_SIGNALS, BRAIN_STATS, OPTIMAL_WEIGHTS, BRAIN_HEALTH, ANALYST_REPORT]
    backed = 0
    for src in files_to_backup:
        if src.exists():
            import shutil
            shutil.copy2(src, cp_dir / src.name)
            backed += 1
    log.info(f"Checkpoint saved: {cp_dir.name} ({backed} files)")
    return str(cp_dir)


def list_checkpoints() -> list:
    """List available checkpoints, newest first."""
    if not CHECKPOINTS_DIR.exists():
        return []
    dirs = sorted(CHECKPOINTS_DIR.iterdir(), reverse=True)
    result = []
    for d in dirs:
        if not d.is_dir():
            continue
        files = [f.name for f in d.iterdir() if f.is_file()]
        # Read IC from optimal_weights if present
        ic = None
        ow = d / "optimal_weights.json"
        if ow.exists():
            try:
                with open(ow) as f:
                    data = json.load(f)
                ic = data.get('_oos_ic')
            except Exception:
                pass
        result.append({'name': d.name, 'files': files, 'oos_ic': ic})
    return result


def rollback_checkpoint(name: str = None) -> dict:
    """Restore brain exports from a checkpoint. Uses most recent if name is None."""
    checkpoints = list_checkpoints()
    if not checkpoints:
        log.warning("No checkpoints found")
        return {'restored': False, 'reason': 'no checkpoints'}

    if name:
        match = [c for c in checkpoints if c['name'] == name]
        if not match:
            log.warning(f"Checkpoint '{name}' not found")
            return {'restored': False, 'reason': f'checkpoint {name} not found'}
        cp = match[0]
    else:
        cp = checkpoints[0]

    cp_dir = CHECKPOINTS_DIR / cp['name']
    import shutil
    targets = {
        'brain_signals.json': BRAIN_SIGNALS,
        'brain_stats.json': BRAIN_STATS,
        'optimal_weights.json': OPTIMAL_WEIGHTS,
        'brain_health.json': BRAIN_HEALTH,
        'analyst_report.json': ANALYST_REPORT,
    }
    restored = []
    for fname, target in targets.items():
        src = cp_dir / fname
        if src.exists():
            shutil.copy2(src, target)
            restored.append(fname)

    log.info(f"Rolled back to checkpoint {cp['name']}: {len(restored)} files restored")
    return {'restored': True, 'checkpoint': cp['name'], 'files': restored, 'oos_ic': cp.get('oos_ic')}


# ── Analyst Report ───────────────────────────────────────────────────────────

def generate_analyst_report(conn: sqlite3.Connection) -> dict:
    """Generate comprehensive analyst report with console output.

    Reads from signals DB, brain_stats.json, brain_health.json, optimal_weights.json.
    Outputs data/analyst_report.json and prints formatted console report.
    """
    import numpy as np
    import pandas as pd
    cur = conn.cursor()
    now = datetime.now(tz=timezone.utc)

    # ── DATA INVENTORY ──
    total = cur.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    congress = cur.execute("SELECT COUNT(*) FROM signals WHERE LOWER(source)='congress'").fetchone()[0]
    edgar = cur.execute("SELECT COUNT(*) FROM signals WHERE LOWER(source)='edgar'").fetchone()[0]

    date_range = cur.execute(
        "SELECT MIN(signal_date), MAX(signal_date) FROM signals"
    ).fetchone()
    start_date = date_range[0] or 'N/A'
    end_date = date_range[1] or 'N/A'
    months = 0
    if start_date != 'N/A' and end_date != 'N/A':
        d1 = datetime.strptime(start_date[:10], '%Y-%m-%d')
        d2 = datetime.strptime(end_date[:10], '%Y-%m-%d')
        months = max(1, (d2 - d1).days / 30)

    pending = cur.execute(
        "SELECT COUNT(*) FROM signals WHERE outcome_30d_filled = 0"
    ).fetchone()[0]
    with_outcome = cur.execute(
        "SELECT COUNT(*) FROM signals WHERE outcome_30d_filled = 1 AND car_30d IS NOT NULL"
    ).fetchone()[0]

    freshness = {}
    for src in ['congress', 'edgar']:
        newest = cur.execute(
            "SELECT MAX(signal_date) FROM signals WHERE LOWER(source)=?", (src,)
        ).fetchone()[0]
        if newest:
            days_stale = (now - datetime.strptime(newest[:10], '%Y-%m-%d').replace(
                tzinfo=timezone.utc)).days
            freshness[src.lower()] = days_stale
        else:
            freshness[src.lower()] = None

    inventory = {
        'total_signals': total, 'congress': congress, 'edgar': edgar,
        'date_range': [start_date, end_date], 'months': round(months, 1),
        'avg_per_month': round(total / max(months, 1), 1),
        'pending_outcome': pending, 'with_outcome': with_outcome,
        'freshness_days': freshness,
    }

    # ── MODEL PERFORMANCE ──
    w = load_json(OPTIMAL_WEIGHTS) if OPTIMAL_WEIGHTS.exists() else {}
    health = load_json(BRAIN_HEALTH) if BRAIN_HEALTH.exists() else {}

    ic = w.get('_oos_ic')
    p_val = w.get('_ic_p_value')
    ic_std = w.get('_ic_std', 0)
    n_folds = w.get('_n_folds', 0)
    hit_rate = w.get('_oos_hit_rate')
    sharpe = w.get('_sharpe_annual')
    sortino = w.get('_sortino_ratio')
    bss = w.get('_brier_skill_score')
    q5q1 = w.get('_q5_q1_spread')
    pf = w.get('_profit_factor')
    t_stat = w.get('_ic_t_stat')

    # IC 95% CI
    ci_lo = ci_hi = None
    if ic is not None and ic_std and n_folds and n_folds > 1:
        se = ic_std / np.sqrt(n_folds)
        ci_lo = round(ic - 1.96 * se, 4)
        ci_hi = round(ic + 1.96 * se, 4)

    # Positive folds — from walk-forward fold ICs, not analyze run history
    pos_folds = w.get('_pos_folds')

    # Strategy metrics
    strat = w.get('_strategy_metrics', {})
    alpha_80 = strat.get('alpha_80plus')
    beta = strat.get('beta_vs_spy')
    sharpe_adj = strat.get('sharpe_market_adjusted_80plus')

    model_perf = {
        'ic': ic, 'ic_p_value': p_val, 'ic_t_stat': t_stat,
        'ic_95_ci': [ci_lo, ci_hi], 'ic_std': ic_std, 'n_folds': n_folds,
        'pos_folds': pos_folds, 'total_folds': n_folds,
        'hit_rate': hit_rate, 'sharpe_annual': sharpe,
        'sortino_ratio': sortino, 'brier_skill_score': bss,
        'q5_q1_spread': q5q1, 'profit_factor': pf,
        'alpha_80plus': alpha_80, 'beta_vs_spy': beta,
        'sharpe_market_adjusted_80plus': sharpe_adj,
        'with_outcome': with_outcome,
    }

    # ── FEATURE REPORT ──
    from backtest.ml_engine import (FEATURE_COLUMNS, CANDIDATE_FEATURES,
                                    get_active_features)
    active_feats, active_cats, fill_report = get_active_features(conn)
    fi = w.get('_feature_importance', {})
    feature_list = []
    for i, f in enumerate(sorted(fi.items(), key=lambda x: -x[1])):
        feature_list.append({
            'name': f[0], 'importance': f[1], 'rank': i + 1,
        })
    candidates_list = [
        {'name': c, 'fill_rate': fill_report.get(c, {}).get('fill_rate', 0),
         'status': fill_report.get(c, {}).get('status', 'unknown')}
        for c in CANDIDATE_FEATURES
    ]

    # ── SIGNAL QUALITY ──
    bands = cur.execute("""
        SELECT
            CASE WHEN total_score >= 80 THEN '80+'
                 WHEN total_score >= 60 THEN '60-79'
                 WHEN total_score >= 40 THEN '40-59'
                 ELSE '<40' END as band,
            COUNT(*) as n,
            ROUND(AVG(car_30d), 4) as avg_car,
            ROUND(AVG(market_adj_car_30d), 4) as avg_adj_car,
            ROUND(100.0*SUM(CASE WHEN car_30d > 0 THEN 1 ELSE 0 END) /
                  NULLIF(SUM(CASE WHEN car_30d IS NOT NULL THEN 1 ELSE 0 END), 0), 1) as hit_rate
        FROM signals WHERE car_30d IS NOT NULL
        GROUP BY band ORDER BY band DESC
    """).fetchall()
    score_dist = [{'band': r['band'], 'n': r['n'], 'avg_car': r['avg_car'],
                   'avg_adj_car': r['avg_adj_car'], 'hit_rate': r['hit_rate']}
                  for r in bands]
    # Check monotonicity
    band_cars = {r['band']: r['avg_car'] for r in bands if r['avg_car'] is not None}
    monotonic = True
    ordered_bands = ['80+', '60-79', '40-59', '<40']
    for i in range(len(ordered_bands) - 1):
        c1 = band_cars.get(ordered_bands[i])
        c2 = band_cars.get(ordered_bands[i + 1])
        if c1 is not None and c2 is not None and c1 < c2:
            monotonic = False
            break

    # ── PIPELINE HEALTH ──
    db_path = SIGNALS_DB
    db_size = db_path.stat().st_size / (1024 * 1024) if db_path.exists() else 0

    expected_files = ['brain_signals.json', 'brain_stats.json', 'brain_health.json',
                      'optimal_weights.json', 'edgar_feed.json', 'congress_feed.json',
                      'news_sentiment.json', 'institutional_data.json', 'options_flow.json']
    data_files = {}
    missing = []
    for f in expected_files:
        p = DATA_DIR / f
        if p.exists():
            age_h = (now.timestamp() - p.stat().st_mtime) / 3600
            data_files[f] = {'exists': True, 'size_kb': round(p.stat().st_size / 1024, 1),
                             'age_hours': round(age_h, 1)}
        else:
            data_files[f] = {'exists': False}
            missing.append(f)

    # DB health details
    db_info = db_health_check(conn)

    pipeline = {
        'db_size_mb': round(db_size, 2), 'db_total_rows': total,
        'data_files': data_files, 'missing_critical': missing,
        'journal_mode': db_info.get('journal_mode', '?'),
        'index_count': db_info.get('index_count', 0),
        'integrity': db_info.get('integrity', '?'),
        'outcome_fill_rate': db_info.get('outcome_fill_rate'),
        'null_rates': db_info.get('null_rates', {}),
    }

    # ── RECOMMENDATIONS ──
    recs = health.get('recommendations', [])[:3]

    report = {
        'generated_at': now.isoformat(),
        'inventory': inventory,
        'model_performance': model_perf,
        'features': {'active': feature_list, 'candidates': candidates_list,
                     'active_count': len(active_feats)},
        'signal_quality': {'score_distribution': score_dist, 'monotonic': monotonic},
        'pipeline': pipeline,
        'recommendations': recs,
    }

    save_json(ANALYST_REPORT, report)

    # ── Console output ──
    today_str = now.strftime('%Y-%m-%d')
    print(f"""
{'='*54}
ATLAS ANALYST REPORT — {today_str}
{'='*54}

DATA INVENTORY
{'─'*54}
Total signals:       {total} ({congress} congress | {edgar} EDGAR)
Date range:          {start_date[:7]} → {end_date[:7]} ({months:.0f} months)
Avg signals/month:   {total / max(months, 1):.1f}
Pending outcomes:    {pending} signals (no 30d result yet)
Data freshness:      Congress {f"{freshness['congress']}d" if freshness.get('congress') is not None else 'N/A'} {'stale' if (freshness.get('congress') or 0) > 7 else 'current'}
                     EDGAR {f"{freshness['edgar']}d" if freshness.get('edgar') is not None else 'N/A'} {'stale' if (freshness.get('edgar') or 0) > 7 else 'current'}""")

    sig_label = 'SIGNIFICANT' if (p_val or 1) < 0.05 else 'not significant'
    ci_str = f"[{ci_lo:.3f} – {ci_hi:.3f}]" if ci_lo is not None else "N/A"
    pf_folds = pos_folds if pos_folds is not None else '?'
    tf = n_folds or '?'
    print(f"""
MODEL PERFORMANCE
{'─'*54}
IC (30d, walk-forward):  {f'{ic:.4f}' if ic else 'N/A'}
  p-value:   {f'{p_val:.3f}' if p_val else 'N/A'}  [{sig_label}]
  95% CI:    {ci_str}
  Pos folds: {pf_folds}/{tf}
Hit Rate (30d):          {f'{hit_rate:.1%}' if hit_rate else 'N/A'}
Sharpe (annual):         {f'{sharpe:.2f}' if sharpe else 'N/A'}
Alpha vs SPY (80+):      {f'{alpha_80:+.2%}/signal' if alpha_80 is not None else 'N/A'}  [market-adjusted]
Beta vs SPY:             {f'{beta:.2f}' if beta is not None else 'N/A'}  {'[high market exposure — not market-neutral]' if beta is not None and beta > 0.7 else ''}""")

    top3 = [f['name'] for f in feature_list[:3]] if feature_list else ['N/A']
    waiting = [c['name'] for c in candidates_list if c['status'] == 'candidate']
    print(f"""
FEATURE HEALTH
{'─'*54}
Active features:     {len(active_feats)}
Top 3 by importance: {' | '.join(top3)}
Candidates waiting:  {', '.join(waiting) if waiting else 'None'}""")

    mono_str = 'YES' if monotonic else 'NO'
    print(f"""
SIGNAL QUALITY
{'─'*54}
Score monotonic: {mono_str}""")
    for sd in score_dist:
        adj_str = f" | adj: {sd['avg_adj_car']:+.2%}" if sd['avg_adj_car'] is not None else ""
        print(f"  {sd['band']:8s}  {sd['n']:5d} signals | "
              f"avg CAR: {sd['avg_car']:+.2%}{adj_str} | "
              f"hit: {sd['hit_rate']:.1f}%")

    # ── REGIME CONTEXT (current market) ──
    regime_mult = 1.0
    try:
        regime_ctx = get_regime_context()
        report['regime_context'] = regime_ctx
        vix_val = regime_ctx.get('current_vix')
        regime_label = regime_ctx.get('regime', 'UNKNOWN')
        regime_mult = regime_ctx.get('multiplier', 1.0)
        print(f"""
REGIME CONTEXT (current)
{'─'*54}
Current VIX:     {vix_val or 'N/A'} → {regime_label} zone
Model confidence: {'FULL' if regime_mult >= 1.0 else f'REDUCED ({regime_mult:.0%})'}  (multiplier: {regime_mult:.2f}x)
Guardrails:      VIX<15=0.75x | VIX 15-25=1.00x | VIX 25-30=0.85x | VIX>40=0.60x""")
    except Exception as e:
        log.warning(f"Regime context failed: {e}")

    # ── REGIME ROBUSTNESS (IC by VIX bucket — OOS when available) ──
    regime_data = []
    try:
        from scipy import stats as sp_stats
        # Prefer oos_score (honest walk-forward holdout) over total_score (in-sample)
        regime_rows = cur.execute("""
            SELECT oos_score, total_score, car_30d, vix_at_signal, signal_date
            FROM signals
            WHERE car_30d IS NOT NULL AND vix_at_signal IS NOT NULL
            AND (oos_score IS NOT NULL OR total_score IS NOT NULL)
        """).fetchall()
        if regime_rows:
            regime_df = pd.DataFrame([dict(r) for r in regime_rows])
            # Use oos_score where available, fallback to total_score
            has_oos = regime_df['oos_score'].notna().sum()
            if has_oos > 100:
                score_col = 'oos_score'
                score_label = 'OOS (walk-forward holdout)'
                regime_df = regime_df[regime_df['oos_score'].notna()]
            else:
                score_col = 'total_score'
                score_label = 'IN-SAMPLE (total_score — run --analyze to populate oos_score)'
                regime_df = regime_df[regime_df['total_score'].notna()]
            regime_buckets = [
                ('Low vol   (VIX<15)',    regime_df.vix_at_signal < 15),
                ('Normal    (VIX 15-25)', (regime_df.vix_at_signal >= 15) & (regime_df.vix_at_signal < 25)),
                ('Elevated  (VIX 25-35)', (regime_df.vix_at_signal >= 25) & (regime_df.vix_at_signal < 35)),
                ('Crisis    (VIX>35)',    regime_df.vix_at_signal >= 35),
            ]
            passing = 0
            print(f"""
REGIME ROBUSTNESS — {score_label}
{'─'*54}""")
            for name, mask in regime_buckets:
                sub = regime_df[mask].dropna(subset=[score_col, 'car_30d'])
                if len(sub) < 30:
                    print(f"  {name}: n={len(sub)} (insufficient)")
                    regime_data.append({'regime': name, 'n': len(sub), 'ic': None, 'status': 'insufficient'})
                    continue
                ic_r, p_r = sp_stats.spearmanr(sub[score_col], sub.car_30d)
                hit_r = (sub.car_30d > 0).mean()
                sig = '***' if p_r < 0.001 else '**' if p_r < 0.01 else '*' if p_r < 0.05 else 'ns'
                status = 'PASS' if ic_r > 0.05 else 'WARN' if ic_r > 0.02 else 'FAIL'
                if ic_r > 0.05:
                    passing += 1
                print(f"  {name}: IC={ic_r:.4f}{sig} hit={hit_r:.1%} n={len(sub):,} [{status}]")
                regime_data.append({'regime': name, 'n': len(sub), 'ic': round(ic_r, 4),
                                    'p_value': round(p_r, 4), 'hit_rate': round(hit_r, 3),
                                    'status': status, 'score_type': score_col})
            print(f"  Regime robust: {passing}/{sum(1 for r in regime_data if r.get('ic') is not None)} passing (need 3+)")
    except Exception as e:
        log.warning(f"Regime robustness failed: {e}")
        import traceback
        traceback.print_exc()
    report['regime_robustness'] = regime_data

    # Diversification from brain_stats
    conc = load_json(BRAIN_STATS).get('signal_concentration', {}) if BRAIN_STATS.exists() else {}
    if conc:
        print(f"""
DIVERSIFICATION (exported top-{EXPORT_LIMIT if 'EXPORT_LIMIT' in dir() else 50})
{'─'*54}
Unique tickers:  {conc.get('unique_tickers', '?')} | Unique sectors: {conc.get('unique_sectors', '?')}
Top ticker:      {conc.get('top_ticker', '?')} ({conc.get('top_ticker_pct', 0):.0%})
Top-5 tickers:   {conc.get('top_5_ticker_pct', 0):.0%} {'[OK]' if conc.get('top_5_ticker_pct', 1) < 0.25 else '[HIGH]'}
Caps:            {conc.get('max_per_ticker_cap', '?')}/ticker, {conc.get('max_per_sector_cap', '?')}/sector""")

    # ── KELLY SIZING ──
    kelly_data = {}
    try:
        kelly_df = pd.read_sql("""
            SELECT car_30d FROM signals
            WHERE oos_score >= 75 AND car_30d IS NOT NULL
        """, conn)
        if len(kelly_df) >= 50:
            winners = kelly_df[kelly_df.car_30d > 0].car_30d
            losers = kelly_df[kelly_df.car_30d <= 0].car_30d
            k_hit = len(winners) / len(kelly_df)
            k_win = float(winners.mean())
            k_loss = float(losers.mean())
            k_b = k_win / abs(k_loss)
            k_full = (k_hit * k_b - (1 - k_hit)) / k_b
            k_quarter = k_full * 0.25
            kelly_data = {
                'hit_rate': round(k_hit, 3), 'avg_win': round(k_win, 4),
                'avg_loss': round(k_loss, 4), 'win_loss_ratio': round(k_b, 2),
                'full_kelly': round(k_full, 3), 'quarter_kelly': round(k_quarter, 3),
            }
            report['kelly_sizing'] = kelly_data
            print(f"""
KELLY POSITION SIZING (OOS 75+ signals)
{'─'*54}
Hit rate:        {k_hit:.1%}
Avg win:         {k_win:+.2%}
Avg loss:        {k_loss:+.2%}
Win/loss ratio:  {k_b:.2f}x
Full Kelly:      {k_full:.1%}
1/4 Kelly:       {k_quarter:.1%} per position (base)
Regime mult:     {regime_mult:.2f}x → effective: {k_quarter*regime_mult:.1%}
Max position:    15.0% | Min position: 2.0%""")
    except Exception as e:
        log.warning(f"Kelly sizing computation failed: {e}")

    # ── Factor Analysis (Fama-French 6-factor) ──
    try:
        ff_path = DATA_DIR / 'ff5_factors.csv'
        if ff_path.exists():
            ff = pd.read_csv(ff_path, index_col=0)
            # Build monthly portfolio returns from OOS 75+ signals
            factor_df = pd.read_sql("""
                SELECT signal_date, oos_score, car_30d
                FROM signals
                WHERE oos_score >= 75 AND car_30d IS NOT NULL
                AND signal_date < '2026-01-01'
            """, conn)
            if len(factor_df) >= 50:
                factor_df['signal_date'] = pd.to_datetime(factor_df['signal_date'])
                factor_df['month'] = factor_df['signal_date'].dt.to_period('M')
                monthly_ret = factor_df.groupby('month').agg(ret=('car_30d', 'mean')).reset_index()
                monthly_ret.index = monthly_ret['month'].dt.to_timestamp()
                monthly_ret.index = monthly_ret.index.to_period('M')

                # Align FF index to period
                try:
                    ff.index = pd.PeriodIndex(ff.index, freq='M')
                except Exception:
                    ff.index = pd.to_datetime(ff.index).to_period('M')

                merged_ff = monthly_ret.join(ff, how='inner')
                if len(merged_ff) >= 20 and 'RF' in merged_ff.columns and 'Mkt-RF' in merged_ff.columns:
                    merged_ff['excess_ret'] = merged_ff['ret'] - merged_ff['RF']
                    factor_cols = [c for c in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM'] if c in merged_ff.columns]
                    X_ff = merged_ff[factor_cols].values
                    y_ff = merged_ff['excess_ret'].values
                    # Add constant (intercept = alpha)
                    X_const = np.column_stack([np.ones(len(X_ff)), X_ff])
                    coeffs, residuals, _, _ = np.linalg.lstsq(X_const, y_ff, rcond=None)
                    # Compute t-stats
                    n_obs = len(y_ff)
                    n_params = X_const.shape[1]
                    y_pred = X_const @ coeffs
                    sse = np.sum((y_ff - y_pred) ** 2)
                    mse = sse / (n_obs - n_params)
                    try:
                        cov_matrix = mse * np.linalg.inv(X_const.T @ X_const)
                        se = np.sqrt(np.diag(cov_matrix))
                        t_stats = coeffs / se
                    except Exception:
                        se = np.full(n_params, np.nan)
                        t_stats = np.full(n_params, np.nan)

                    names_ff = ['alpha'] + factor_cols
                    monthly_alpha = coeffs[0]
                    ann_alpha = (1 + monthly_alpha) ** 12 - 1
                    r_squared = 1 - sse / np.sum((y_ff - y_ff.mean()) ** 2)

                    factor_result = {
                        'monthly_alpha': round(float(monthly_alpha), 4),
                        'annualized_alpha': round(float(ann_alpha), 4),
                        'r_squared': round(float(r_squared), 3),
                        'n_months': int(n_obs),
                        'loadings': {},
                    }
                    print(f"""
FACTOR ANALYSIS (Fama-French {'6' if 'MOM' in factor_cols else '5'}-factor)
{'─'*54}
Months aligned:  {n_obs}
R²:              {r_squared:.3f}""")
                    for i, name in enumerate(names_ff):
                        sig = ''
                        if not np.isnan(t_stats[i]):
                            ap = abs(t_stats[i])
                            sig = ' ***' if ap > 3.29 else ' **' if ap > 2.58 else ' *' if ap > 1.96 else ' ns'
                        print(f"  {name:12s}  {coeffs[i]:+.4f}  (t={t_stats[i]:+.2f}{sig})")
                        if name != 'alpha':
                            factor_result['loadings'][name] = round(float(coeffs[i]), 4)
                    print(f"""
Monthly alpha:   {monthly_alpha:+.4f} ({monthly_alpha*100:+.2f}%/mo)
Annualized alpha:{ann_alpha:+.1%}""")
                    # Interpretation
                    if ann_alpha > 0.12:
                        verdict = "STRONG — genuine insider edge beyond all factors"
                    elif ann_alpha > 0.06:
                        verdict = "MODERATE — real edge, partially explained by factors"
                    else:
                        verdict = "WEAK — mostly factor exposure"
                    factor_result['verdict'] = verdict
                    print(f"Verdict:         {verdict}")
                    report['factor_analysis'] = factor_result
                else:
                    log.info(f"FF factor alignment: only {len(merged_ff)} months (need 20+)")
            else:
                log.info(f"Only {len(factor_df)} OOS 75+ signals — need 50+ for factor regression")
        else:
            log.info("FF5 factors not found at data/ff5_factors.csv — skipping factor analysis")
            log.info("Download: pip install pandas-datareader && python3 -c \"import pandas_datareader.data as web; ff=web.DataReader('F-F_Research_Data_5_Factors_2x3','famafrench',start='2019-01-01')[0]; mom=web.DataReader('F-F_Momentum_Factor','famafrench',start='2019-01-01')[0]; ff=ff/100; mom=mom/100; mom.columns=['MOM']; ff.join(mom,how='inner').to_csv('data/ff5_factors.csv')\"")
    except Exception as e:
        log.warning(f"Factor analysis failed: {e}")
        import traceback; traceback.print_exc()

    # ── Factor vs Strategy Gap Analysis ──
    try:
        gap_analysis = {}
        # Universe 1: OOS 75+ raw (what factor regression uses)
        oos75_df = pd.read_sql("""
            SELECT car_30d, total_score, oos_score, price_at_signal
            FROM signals WHERE oos_score >= 75 AND car_30d IS NOT NULL
            AND signal_date < '2026-01-01'
        """, conn)
        if len(oos75_df) >= 30:
            gap_analysis['oos75_raw'] = {
                'n': len(oos75_df),
                'avg_car': round(float(oos75_df['car_30d'].mean()), 4),
                'hit_rate': round(float((oos75_df['car_30d'] > 0).mean()), 3),
                'has_price': int((oos75_df['price_at_signal'].notna() & (oos75_df['price_at_signal'] > 0)).sum()),
            }

        # Universe 2: Strategy-eligible (65+ with entry price)
        strat_df = pd.read_sql(f"""
            SELECT car_30d, total_score, oos_score, price_at_signal
            FROM signals WHERE total_score >= {TRADING_RULES['entry_threshold']}
            AND car_30d IS NOT NULL AND price_at_signal > 0
            AND signal_date < '2026-01-01'
        """, conn)
        if len(strat_df) >= 30:
            # Apply hard stop-loss floor only (trailing stop lets winners run)
            sl = TRADING_RULES['stop_loss_pct']
            floored = strat_df['car_30d'].clip(lower=sl)
            gap_analysis['strategy'] = {
                'n': len(strat_df),
                'avg_car_raw': round(float(strat_df['car_30d'].mean()), 4),
                'avg_car_floored': round(float(floored.mean()), 4),
                'hit_rate': round(float((floored > 0).mean()), 3),
                'stop_triggered': int((strat_df['car_30d'] < sl).sum()),
                'runners_20pct': int((strat_df['car_30d'] > 0.20).sum()),
                'runners_30pct': int((strat_df['car_30d'] > 0.30).sum()),
            }

        if gap_analysis:
            # Diagnose gap sources
            sources = []
            oos = gap_analysis.get('oos75_raw', {})
            strat = gap_analysis.get('strategy', {})
            if oos and strat:
                car_diff = oos.get('avg_car', 0) - strat.get('avg_car_floored', strat.get('avg_car_capped', 0))
                if car_diff > 0.01:
                    strat_avg = strat.get('avg_car_floored', strat.get('avg_car_capped', 0))
                    sources.append(f"Threshold: OOS 75+ avg {oos['avg_car']:.2%} vs strategy 65+ {strat_avg:.2%}")
                missing_pct = 1 - oos.get('has_price', 0) / max(oos.get('n', 1), 1)
                if missing_pct > 0.05:
                    sources.append(f"Missing prices: {missing_pct:.0%} of OOS 75+ signals lack entry price")
                if strat.get('stop_triggered', 0) > 0:
                    sources.append(f"Stop-loss: {strat['stop_triggered']} trades hit {sl:.0%} floor")
                if strat.get('runners_20pct', 0) > 0:
                    sources.append(f"Runners: {strat['runners_20pct']} trades gained >20%, {strat.get('runners_30pct', 0)} gained >30%")
            gap_analysis['gap_sources'] = sources
            report['factor_strategy_gap'] = gap_analysis

            print(f"""
FACTOR vs STRATEGY GAP ANALYSIS
{'─'*54}""")
            if oos:
                print(f"OOS 75+ raw:     n={oos['n']} avg={oos['avg_car']:+.2%} hit={oos['hit_rate']:.1%}")
            if strat:
                strat_avg = strat.get('avg_car_floored', strat.get('avg_car_capped', 0))
                print(f"Strategy 65+:    n={strat['n']} avg_raw={strat['avg_car_raw']:+.2%} "
                      f"avg_floored={strat_avg:+.2%} hit={strat['hit_rate']:.1%}")
            for s in sources:
                print(f"  → {s}")

    except Exception as e:
        log.warning(f"Factor gap analysis failed: {e}")

    ofr = db_info.get('outcome_fill_rate')
    ofr_str = f"{ofr:.1%}" if ofr is not None else "?"
    print(f"""
PIPELINE HEALTH
{'─'*54}
DB:              {db_size:.1f} MB | {total:,} rows
Journal mode:    {db_info.get('journal_mode', '?')} | {db_info.get('index_count', 0)} indexes
Integrity:       {db_info.get('integrity', '?')}
Outcome fill:    {ofr_str}
Missing files:   {', '.join(missing) if missing else 'None'}""")

    if recs:
        print(f"""
NEXT ACTIONS
{'─'*54}""")
        for r in recs[:3]:
            print(f"  • {r}")

    print(f"{'='*54}")

    # Re-save with all sections (regime, Kelly added after initial save)
    save_json(ANALYST_REPORT, report)
    print(f"\nSaved to {ANALYST_REPORT}")

    return report


# ── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ATLAS Adaptive Learning Engine')
    parser.add_argument('--daily', action='store_true', help='Run daily collect + backfill')
    parser.add_argument('--analyze', action='store_true', help='Run feature analysis + weight update')
    parser.add_argument('--summary', action='store_true', help='Print status summary')
    parser.add_argument('--bootstrap', action='store_true', help='Run historical bootstrap')
    parser.add_argument('--incremental', action='store_true',
                        help='Incremental bootstrap: only fetch signals older than existing data')
    parser.add_argument('--diagnostics', action='store_true', help='Generate diagnostics HTML + analysis report')
    parser.add_argument('--export', action='store_true', help='Export brain_signals.json + brain_stats.json for frontend')
    parser.add_argument('--score', action='store_true', help='Score all signals with ML + export brain data')
    parser.add_argument('--self-check', action='store_true', dest='self_check', help='Run Brain health diagnostics')
    parser.add_argument('--backfill', action='store_true', help='Re-enrich v5/v6 features (volume, analyst, committee, earnings) for all signals')
    parser.add_argument('--eval-features', nargs='+', dest='eval_features', metavar='COL',
                        help='Evaluate candidate feature columns against baseline IC')
    parser.add_argument('--hypotheses', action='store_true',
                        help='Generate signal hypotheses from model residuals and feature interactions')
    parser.add_argument('--report', action='store_true',
                        help='Generate analyst report (console + JSON)')
    parser.add_argument('--rollback', nargs='?', const='__latest__', default=None,
                        metavar='CHECKPOINT',
                        help='Rollback to checkpoint (most recent if no name given)')
    parser.add_argument('--edgar-days', type=int, default=2555, dest='edgar_days',
                        help='EDGAR lookback days for --bootstrap (default 2555 = ~7 years)')
    args = parser.parse_args()

    conn = init_db()

    if args.bootstrap:
        # Delegate to bootstrap script
        from backtest.bootstrap_historical import bootstrap
        bootstrap(conn, edgar_days=args.edgar_days, incremental=args.incremental)
    elif args.diagnostics:
        generate_analysis_report(conn)
        generate_diagnostics_html(conn)
        print(f"Diagnostics saved to:\n  {ALE_DIAGNOSTICS_HTML}\n  {ALE_ANALYSIS_REPORT}")
    elif args.report:
        report = generate_analyst_report(conn)
        print(f"\nSaved to {ANALYST_REPORT}")
    elif args.rollback is not None:
        cp_name = None if args.rollback == '__latest__' else args.rollback
        # List checkpoints first
        checkpoints = list_checkpoints()
        if not checkpoints:
            print("No checkpoints available.")
        else:
            print(f"\nAvailable checkpoints ({len(checkpoints)}):")
            for i, cp in enumerate(checkpoints[:10]):
                ic_str = f"IC={cp['oos_ic']:.4f}" if cp.get('oos_ic') is not None else "no IC"
                marker = " ← restoring" if (cp_name == cp['name'] or (cp_name is None and i == 0)) else ""
                print(f"  {cp['name']}  ({ic_str}, {len(cp['files'])} files){marker}")
            result = rollback_checkpoint(cp_name)
            if result.get('restored'):
                print(f"\nRestored checkpoint: {result['checkpoint']}")
                print(f"  Files: {', '.join(result['files'])}")
                if result.get('oos_ic') is not None:
                    print(f"  IC: {result['oos_ic']:.4f}")
            else:
                print(f"\nRollback failed: {result.get('reason', 'unknown')}")
    elif args.self_check:
        health = run_self_check(conn)
        overall = str(health.get('overall_status', 'unknown')).upper()
        print(f"\nBrain Health: {overall}")
        for name, check in health.get('checks', {}).items():
            status = str(check.get('status', 'ok')).upper()
            print(f"  [{status:8s}] {name}")
        for rec in health.get('recommendations', []):
            print(f"  [SUGGEST] {rec}")
        print(f"\nSaved to {BRAIN_HEALTH}")
    elif args.backfill:
        # Backfill entry prices first (recovers 800+ older EDGAR signals)
        price_filled = backfill_entry_prices(conn)
        if price_filled:
            print(f"Backfilled entry prices for {price_filled} signals")
        result = backfill_features(conn)
        print(f"\nBackfill complete: {result['enriched']} features re-enriched")
        for col in ('volume_dry_up', 'analyst_revision_30d', 'analyst_consensus',
                     'analyst_insider_confluence', 'committee_overlap',
                     'earnings_surprise', 'news_sentiment_30d',
                     'news_sentiment_score', 'news_sentiment_strong_positive',
                     'news_sentiment_strong_negative', 'news_insider_confluence',
                     'sentiment_divergence',
                     'short_interest_pct', 'short_interest_change', 'short_squeeze_signal'):
            before = result['nulls_before'].get(col, '?')
            after = result['nulls_after'].get(col, '?')
            print(f"  {col}: {before} → {after} NULLs")
    elif args.eval_features:
        from backtest.ml_engine import evaluate_feature_candidates
        result = evaluate_feature_candidates(conn, args.eval_features)
        result['evaluated_at'] = datetime.now(tz=timezone.utc).isoformat()
        save_json(FEATURE_CANDIDATES, result)
        print(f"\nFeature Evaluation (baseline IC={result['baseline_ic']:.4f}, "
              f"{result['baseline_features']} features)")
        recommendations = []
        for name, info in result['candidates'].items():
            if info['status'] == 'evaluated':
                delta = info['ic_delta']
                rec = info['recommendation']
                print(f"  {name}: IC={info['ic']:.4f} (delta={delta:+.4f}) "
                      f"fill={info['fill_rate']:.0%} → {rec}")
                if rec == 'ADD' and delta > 0.005:
                    recommendations.append(f"{name} (+{delta:.4f} IC)")
            else:
                print(f"  {name}: {info['status']} — {info.get('reason', '')}")
        if recommendations:
            print(f"\n*** FEATURE RECOMMENDATIONS: {', '.join(recommendations)} ***")
        print(f"\nSaved to {FEATURE_CANDIDATES}")
    elif args.hypotheses:
        hyp = generate_signal_hypotheses(conn)
        print(f"\n=== Signal Hypotheses ===")
        print(f"\nHigh-Residual Tickers ({len(hyp['high_residual_tickers'])}):")
        for h in hyp['high_residual_tickers'][:5]:
            print(f"  {h['ticker']} ({h['sector']}): residual={h['avg_residual']:.4f} "
                  f"— {h['hypothesis']}")
        print(f"\nFeature Interactions ({len(hyp['feature_interactions'])}):")
        for h in hyp['feature_interactions'][:3]:
            print(f"  {' × '.join(h['features'])}: IC={h['interaction_ic']:.4f} "
                  f"(+{h['estimated_ic_gain']:.4f}) — {h['hypothesis']}")
        print(f"\nRegime Gaps ({len(hyp['regime_gaps'])}):")
        for h in hyp['regime_gaps']:
            print(f"  {h['regime']}: hit_rate={h['current_hit_rate']:.1%} "
                  f"— {h['hypothesis']}")
        print(f"\nSaved to {SIGNAL_HYPOTHESES}")
    elif args.score:
        scored = score_all_signals(conn)
        export_brain_data(conn)
        print(f"Scored {scored} signals + exported:\n  {BRAIN_SIGNALS}\n  {BRAIN_STATS}")
    elif args.export:
        export_brain_data(conn)
        print(f"Brain data exported to:\n  {BRAIN_SIGNALS}\n  {BRAIN_STATS}")
    elif args.daily:
        run_daily(conn)
    elif args.analyze:
        run_analyze(conn)
    elif args.summary:
        print_summary(conn)
    else:
        # Default: daily + analyze if it's Monday
        run_daily(conn)
        today = datetime.now(tz=timezone.utc)
        if today.weekday() == 0:  # Monday
            log.info("Monday — running weekly feature analysis...")
            run_analyze(conn)
        print_summary(conn)

    conn.close()


if __name__ == '__main__':
    main()
