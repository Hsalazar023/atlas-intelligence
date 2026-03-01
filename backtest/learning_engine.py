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
    BRAIN_SIGNALS, BRAIN_STATS, BRAIN_HEALTH,
    load_json, save_json, match_edgar_ticker, range_to_base_points,
)
from backtest.sector_map import get_sector, get_market_cap, get_market_cap_bucket

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

ALE_DASHBOARD = DATA_DIR / "ale_dashboard.json"
ALE_ANALYSIS_REPORT = DATA_DIR / "ale_analysis_report.md"
ALE_DIAGNOSTICS_HTML = DATA_DIR / "ale_diagnostics.html"

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
        'sector_momentum', 'trade_pattern',
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

    for t in trades:
        tx = (t.get('Transaction') or '').lower()
        if 'purchase' not in tx and 'buy' not in tx:
            continue
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
        }
        if insert_signal(conn, signal):
            inserted += 1

    log.info(f"EDGAR feed: {len(filings)} filings → {skipped_non_buy} non-buy skipped → {inserted} inserted")
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


def update_aggregate_features(conn: sqlite3.Connection) -> int:
    """Recompute aggregate features with multi-tier convergence detection.

    Convergence Tiers:
      0 — No convergence (single source only)
      1 — Ticker convergence: same ticker in 2+ hubs within window
          (congress 60d lookback, edgar 30d lookback)
      2 — Sector convergence: 3+ signals from 2+ sources in same sector, 30d
      3 — Thematic: sector convergence + active legislation (reserved for frontend)
    """
    # Congress gets 60d window (STOCK Act allows 45d disclosure delay)
    CONGRESS_WINDOW = 60
    EDGAR_WINDOW = 30

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
        prior = rep_history.get(rep, [])
        prior_with_outcomes_30 = [p for p in prior if p['car_30d'] is not None]
        prior_with_outcomes_90 = [p for p in prior if p['car_90d'] is not None]

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
        prior_with_outcomes_30 = [p for p in prior if p['car_30d'] is not None]
        prior_with_outcomes_90 = [p for p in prior if p['car_90d'] is not None]

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
    # Compute once per sector, then batch-update all signals in that sector
    sectors_needing = conn.execute(
        "SELECT DISTINCT sector FROM signals "
        "WHERE sector_avg_car IS NULL AND sector IS NOT NULL"
    ).fetchall()

    sector_car_updated = 0
    for row in sectors_needing:
        sector = row['sector']
        avg_car = _compute_sector_avg_car(conn, sector)
        if avg_car is not None:
            cnt = conn.execute(
                "UPDATE signals SET sector_avg_car=? "
                "WHERE sector=? AND sector_avg_car IS NULL",
                (avg_car, sector)
            ).rowcount
            sector_car_updated += cnt
    if sector_car_updated:
        log.info(f"Backfilled sector_avg_car for {sector_car_updated} signals")
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
            log.info(f"Backfilled insider_role via cross-signal propagation: {role_filled} signals")

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
                log.info(f"Backfilled insider_role from feed: {len(role_map)} insiders, {feed_filled} signals")

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

    conn.commit()
    log.info(f"Enriched {updated} signal features (52wk, trade pattern, role, catalysts, earnings)")
    return updated


# ── Legislative Catalyst Registry ─────────────────────────────────────────────
# Tracked bills with dates and impacted tickers. Kept in sync with frontend BILLS.
# When adding a new bill: add here AND in atlas-intelligence.html BILLS array.

LEGISLATIVE_CATALYSTS = [
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

# Build a quick lookup: ticker -> list of catalyst dates
_CATALYST_TICKER_MAP: dict = {}
_CATALYST_SECTOR_MAP: dict = {}
for _bill in LEGISLATIVE_CATALYSTS:
    for _t in _bill['tickers']:
        _CATALYST_TICKER_MAP.setdefault(_t, []).append(_bill['date'])
    _CATALYST_SECTOR_MAP.setdefault(_bill['sector'], []).append(_bill['date'])


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


def backfill_outcomes(conn: sqlite3.Connection, spy_index: dict = None) -> int:
    """Backfill return/CAR outcomes for signals where enough time has passed."""
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
    rows = conn.execute(
        "SELECT id, ticker, signal_date, outcome_5d_filled, outcome_30d_filled, "
        "outcome_90d_filled, outcome_180d_filled, outcome_365d_filled "
        "FROM signals WHERE outcome_5d_filled = 0 OR outcome_30d_filled = 0 "
        "OR outcome_90d_filled = 0 OR outcome_180d_filled = 0 OR outcome_365d_filled = 0"
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


def _compute_sector_avg_car(conn: sqlite3.Connection, sector: str) -> float | None:
    """Average car_30d for all signals in the same sector.

    Tells the model: signals in this sector have historically been profitable/unprofitable.
    """
    if not sector:
        return None
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
    log.info(f"Feature stats: {len(stats)} feature-value pairs computed from {len(rows)} signals")
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

    log.info(f"Weights updated via feature_importance (n={len(all_30d)}, hit_rate={overall_hit:.1%}, avg_car={overall_avg:.4f})")
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
    log.info(f"Analysis report written to: {ALE_ANALYSIS_REPORT} ({len(lines)} lines)")
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
    log.info(f"Diagnostics HTML written to: {ALE_DIAGNOSTICS_HTML}")
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
      total     = clamp(sum, 0, 100)
    """
    log.info("=== Scoring All Signals ===")

    try:
        from backtest.ml_engine import train_full_sample, prepare_features_all
    except ImportError:
        log.warning("ML dependencies not installed — skipping scoring")
        return 0

    # Train full-sample models
    models = train_full_sample(conn)
    if models is None:
        log.warning("Could not train models — skipping scoring")
        return 0

    clf_rf, clf_lgb, reg_rf, reg_lgb = models

    # Ensure score breakdown columns exist
    for col in ['ml_confidence', 'predicted_car']:
        try:
            conn.execute(f"ALTER TABLE signals ADD COLUMN {col} REAL")
        except Exception:
            pass  # column already exists

    # Prepare features for ALL signals
    X, ids, dates, tickers, cars = prepare_features_all(conn)
    if len(X) == 0:
        log.warning("No signals to score")
        return 0

    # Classification: ensemble P(beat SPY)
    clf_probs = (clf_rf.predict_proba(X)[:, 1] + clf_lgb.predict_proba(X)[:, 1]) / 2

    # Regression: ensemble predicted CAR
    reg_preds = (reg_rf.predict(X) + reg_lgb.predict(X)) / 2

    # Fetch convergence + person data for bonus terms
    rows = conn.execute(
        "SELECT id, convergence_tier, person_hit_rate_30d FROM signals"
    ).fetchall()
    meta = {r['id']: (r['convergence_tier'] or 0, r['person_hit_rate_30d'] or 0) for r in rows}

    # Compute scores
    updates = []
    scores = []
    for i in range(len(ids)):
        sig_id = int(ids[i])
        conv_tier, person_hr = meta.get(sig_id, (0, 0))

        base = float(clf_probs[i]) * 60
        magnitude = max(-20, min(25, float(reg_preds[i]) * 200))
        converge = float(conv_tier) * 5
        person = max(0, min(5, float(person_hr) * 8))
        total = max(0, min(100, base + magnitude + converge + person))
        total = round(total, 2)

        updates.append((total, round(float(clf_probs[i]), 4), round(float(reg_preds[i]), 6), sig_id))
        scores.append(total)

    # Batch update
    conn.executemany(
        "UPDATE signals SET total_score = ?, ml_confidence = ?, predicted_car = ? WHERE id = ?",
        updates
    )
    conn.commit()

    # Log distribution
    import numpy as np
    scores_arr = np.array(scores)
    log.info(f"Scored {len(scores)} signals")
    log.info(f"  Distribution: mean={scores_arr.mean():.1f}, median={np.median(scores_arr):.1f}, "
             f"min={scores_arr.min():.1f}, max={scores_arr.max():.1f}")

    # Score tier counts
    tiers = {
        '80-100 (strong buy)': int(np.sum(scores_arr >= 80)),
        '60-80 (buy)': int(np.sum((scores_arr >= 60) & (scores_arr < 80))),
        '40-60 (neutral)': int(np.sum((scores_arr >= 40) & (scores_arr < 60))),
        '20-40 (weak)': int(np.sum((scores_arr >= 20) & (scores_arr < 40))),
        '0-20 (avoid)': int(np.sum(scores_arr < 20)),
    }
    for tier, count in tiers.items():
        log.info(f"  {tier}: {count}")

    return len(scores)


def log_brain_run(conn: sqlite3.Connection, run_type: str,
                  oos_ic: float = None, oos_hit_rate: float = None,
                  notes: str = None) -> None:
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

    cur.execute("""
        INSERT INTO brain_runs (run_date, run_type, oos_ic, oos_hit_rate,
            n_signals, n_scored, avg_score, max_score, top_ticker, top_ticker_pct,
            feature_importance_json, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now(tz=timezone.utc).strftime('%Y-%m-%d'), run_type,
          oos_ic, oos_hit_rate, n_signals, n_scored, avg_score, max_score,
          top_ticker, top_ticker_pct, fi_json, notes))
    conn.commit()
    log.info(f"Brain run logged: type={run_type}, IC={oos_ic}, scored={n_scored}")


def run_self_check(conn: sqlite3.Connection) -> dict:
    """Run Brain health diagnostics and export brain_health.json.

    Checks: IC trend, score concentration, data freshness, feature fill rates,
    source alpha balance, and generates warnings/suggestions.
    """
    import numpy as np
    log.info("=== Brain Self-Check ===")
    cur = conn.cursor()
    health = {'generated': datetime.now(tz=timezone.utc).isoformat(), 'status': 'healthy', 'checks': [], 'warnings': [], 'suggestions': []}

    # ── 1. IC Trend ──
    runs = cur.execute(
        "SELECT run_date, oos_ic, oos_hit_rate, run_type FROM brain_runs "
        "WHERE oos_ic IS NOT NULL ORDER BY id DESC LIMIT 10"
    ).fetchall()
    ic_history = [{'date': r['run_date'], 'ic': r['oos_ic'], 'hit_rate': r['oos_hit_rate'], 'type': r['run_type']} for r in runs]
    health['ic_history'] = ic_history

    if len(ic_history) >= 3:
        last3 = [h['ic'] for h in ic_history[:3]]
        if all(last3[i] < last3[i+1] for i in range(2)):
            health['warnings'].append('IC declining for 3 consecutive runs: ' + ' → '.join(f"{x:.4f}" for x in reversed(last3)))
            health['status'] = 'warning'
        health['checks'].append(f"IC trend: {' → '.join(f'{x:.4f}' for x in reversed(last3[-3:]))}")
    elif ic_history:
        health['checks'].append(f"IC latest: {ic_history[0]['ic']:.4f} (need 3+ runs for trend)")
    else:
        health['checks'].append("No IC history yet — run --analyze to populate")

    # ── 2. Score Concentration ──
    top50 = cur.execute(
        "SELECT ticker FROM signals WHERE total_score IS NOT NULL "
        "ORDER BY total_score DESC LIMIT 50"
    ).fetchall()
    ticker_counts = {}
    for r in top50:
        ticker_counts[r['ticker']] = ticker_counts.get(r['ticker'], 0) + 1
    if ticker_counts:
        top_t = max(ticker_counts, key=ticker_counts.get)
        top_pct = ticker_counts[top_t] / len(top50)
        health['score_concentration'] = {
            'top_ticker': top_t, 'count': ticker_counts[top_t],
            'pct_of_top50': round(top_pct, 3), 'unique_in_top50': len(ticker_counts)
        }
        if top_pct > 0.3:
            health['warnings'].append(f"Score concentration: {top_t} is {top_pct:.0%} of top 50 signals")
            health['status'] = 'warning'
        health['checks'].append(f"Top 50 diversity: {len(ticker_counts)} unique tickers, top={top_t} ({ticker_counts[top_t]}/50)")

    # ── 3. Data Freshness ──
    latest = cur.execute("SELECT MAX(signal_date) as d FROM signals").fetchone()['d']
    if latest:
        days_stale = (datetime.now(tz=timezone.utc).date() - datetime.fromisoformat(latest).date()).days
        health['data_freshness'] = {'latest_signal': latest, 'days_since': days_stale}
        if days_stale > 3:
            health['warnings'].append(f"No new signals in {days_stale} days (latest: {latest})")
            health['status'] = 'warning'
        health['checks'].append(f"Data freshness: latest signal {latest} ({days_stale}d ago)")

    # ── 4. Feature Fill Rates ──
    fill_sql = """
        SELECT
            ROUND(100.0*SUM(CASE WHEN insider_role IS NOT NULL AND insider_role != '' THEN 1 ELSE 0 END)/COUNT(*),1) as insider_role,
            ROUND(100.0*SUM(CASE WHEN sector IS NOT NULL AND sector != '' THEN 1 ELSE 0 END)/COUNT(*),1) as sector,
            ROUND(100.0*SUM(CASE WHEN market_cap_bucket IS NOT NULL THEN 1 ELSE 0 END)/COUNT(*),1) as market_cap,
            ROUND(100.0*SUM(CASE WHEN momentum_1m IS NOT NULL THEN 1 ELSE 0 END)/COUNT(*),1) as momentum,
            ROUND(100.0*SUM(CASE WHEN person_hit_rate_30d IS NOT NULL THEN 1 ELSE 0 END)/COUNT(*),1) as person_hr,
            ROUND(100.0*SUM(CASE WHEN trade_pattern IS NOT NULL AND trade_pattern != '' THEN 1 ELSE 0 END)/COUNT(*),1) as trade_pattern
        FROM signals
    """
    fill = dict(cur.execute(fill_sql).fetchone())
    health['feature_fill_rates'] = fill
    low_fills = {k: v for k, v in fill.items() if v < 50}
    if low_fills:
        health['suggestions'].append(f"Low fill features ({', '.join(f'{k}={v}%' for k, v in low_fills.items())}) — consider pruning from ML if <1% importance")
    health['checks'].append(f"Feature fills: {sum(1 for v in fill.values() if v >= 80)}/{len(fill)} above 80%")

    # ── 5. Source Alpha ──
    source_stats = cur.execute("""
        SELECT source, COUNT(*) as n, ROUND(AVG(car_30d),4) as avg_car,
            ROUND(100.0*SUM(CASE WHEN car_30d > 0 THEN 1 ELSE 0 END)/NULLIF(SUM(CASE WHEN car_30d IS NOT NULL THEN 1 ELSE 0 END),0),1) as hit_rate
        FROM signals WHERE car_30d IS NOT NULL GROUP BY source
    """).fetchall()
    health['source_alpha'] = [{'source': r['source'], 'n': r['n'], 'avg_car': r['avg_car'], 'hit_rate': r['hit_rate']} for r in source_stats]
    for s in health['source_alpha']:
        if s['avg_car'] is not None and s['avg_car'] < -0.005:
            health['suggestions'].append(f"{s['source']} signals have negative avg CAR ({s['avg_car']:.2%}) — consider down-weighting")

    # ── 6. Score Band Validation ──
    bands = cur.execute("""
        SELECT
            CASE WHEN total_score >= 80 THEN '80+'
                 WHEN total_score >= 65 THEN '65-79'
                 WHEN total_score >= 40 THEN '40-64'
                 ELSE '<40' END as band,
            COUNT(*) as n,
            ROUND(AVG(car_30d),4) as avg_car,
            ROUND(100.0*SUM(CASE WHEN car_30d > 0 THEN 1 ELSE 0 END)/NULLIF(SUM(CASE WHEN car_30d IS NOT NULL THEN 1 ELSE 0 END),0),1) as hit_rate
        FROM signals WHERE car_30d IS NOT NULL GROUP BY band ORDER BY band DESC
    """).fetchall()
    health['score_bands'] = [{'band': r['band'], 'n': r['n'], 'avg_car': r['avg_car'], 'hit_rate': r['hit_rate']} for r in bands]
    # Check monotonicity: higher bands should have higher CAR
    band_cars = {r['band']: r['avg_car'] for r in bands if r['avg_car'] is not None}
    if band_cars.get('80+', 0) < band_cars.get('65-79', 0):
        health['warnings'].append("Score band inversion: 80+ signals have lower CAR than 65-79 — scoring formula may need recalibration")
        health['status'] = 'warning'
    if band_cars.get('80+', 0) < band_cars.get('<40', 0):
        health['warnings'].append("Critical: 80+ signals underperform <40 — model may be inverting")
        health['status'] = 'critical'
    health['checks'].append(f"Score bands: 80+ CAR={band_cars.get('80+','N/A')}, 65-79={band_cars.get('65-79','N/A')}, <40={band_cars.get('<40','N/A')}")

    # ── 7. Feature Importance Drift ──
    fi_runs = cur.execute(
        "SELECT feature_importance_json FROM brain_runs "
        "WHERE feature_importance_json IS NOT NULL ORDER BY id DESC LIMIT 2"
    ).fetchall()
    if len(fi_runs) >= 2:
        fi_new = json.loads(fi_runs[0]['feature_importance_json'])
        fi_old = json.loads(fi_runs[1]['feature_importance_json'])
        big_shifts = []
        for feat in fi_new:
            old_v = fi_old.get(feat, 0)
            new_v = fi_new[feat]
            if old_v > 0 and abs(new_v - old_v) / old_v > 0.5:
                big_shifts.append(f"{feat}: {old_v:.3f}→{new_v:.3f}")
        if big_shifts:
            health['suggestions'].append(f"Feature importance shifts (>50%): {', '.join(big_shifts[:5])}")
        health['checks'].append(f"Feature drift: {len(big_shifts)} features shifted >50%")

    # ── Summary ──
    log.info(f"Health status: {health['status'].upper()}")
    for c in health['checks']:
        log.info(f"  [CHECK] {c}")
    for w in health['warnings']:
        log.info(f"  [WARN]  {w}")
    for s in health['suggestions']:
        log.info(f"  [SUGGEST] {s}")

    save_json(BRAIN_HEALTH, health)
    log.info(f"Brain health saved to {BRAIN_HEALTH}")
    return health


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
        SELECT ticker, price_at_signal, signal_date, source, total_score,
               convergence_tier, has_convergence, representative, insider_name,
               insider_role, transaction_type, person_hit_rate_30d,
               person_trade_count, sector, car_30d,
               same_ticker_signals_7d, ml_confidence, predicted_car
        FROM signals
        WHERE signal_date >= date('now', '-90 days')
          AND total_score IS NOT NULL
        ORDER BY total_score DESC
        LIMIT 200
    """)
    all_rows = cur.fetchall()
    cols = [d[0] for d in cur.description]

    # Diversify: max N signals per ticker
    ticker_counts = {}
    rows = []
    for row in all_rows:
        r = dict(zip(cols, row))
        t = r['ticker']
        ticker_counts[t] = ticker_counts.get(t, 0) + 1
        if ticker_counts[t] <= MAX_PER_TICKER:
            rows.append(row)
        if len(rows) >= EXPORT_LIMIT:
            break

    signals_out = []
    for row in rows:
        r = dict(zip(cols, row))
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
        # Add company name if known and not already obvious
        company = TICKER_NAMES.get(r['ticker'])
        if company:
            note_parts.append(company)
        note = ' · '.join(note_parts) if note_parts else ''

        person = r['representative'] or r['insider_name'] or ''

        if direction == 'long':
            entry_lo = round(price * 0.96, 2) if price else None
            entry_hi = round(price * 1.04, 2) if price else None
            target1 = round(price * 1.20, 2) if price else None
            target2 = round(price * 1.35, 2) if price else None
            stop = round(price * 0.88, 2) if price else None
        else:
            entry_lo = round(price * 0.96, 2) if price else None
            entry_hi = round(price * 1.04, 2) if price else None
            target1 = round(price * 0.80, 2) if price else None
            target2 = round(price * 0.65, 2) if price else None
            stop = round(price * 1.12, 2) if price else None

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
            'sector': r['sector'],
            'ml_confidence': round(r['ml_confidence'], 3) if r.get('ml_confidence') is not None else None,
            'predicted_car': round(r['predicted_car'] * 100, 1) if r.get('predicted_car') is not None else None,
            'entry_lo': entry_lo,
            'entry_hi': entry_hi,
            'target1': target1,
            'target2': target2,
            'stop': stop,
        })

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

    brain_signals = {
        'generated': now,
        'signals': signals_out,
        'exits': exits_out,
    }
    save_json(BRAIN_SIGNALS, brain_signals)
    log.info(f"Exported {len(signals_out)} signals + {len(exits_out)} exits → {BRAIN_SIGNALS}")

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

    # Score tiers
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
                'n_folds': w.get('stats', {}).get('n_folds'),
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
    save_json(BRAIN_STATS, brain_stats)
    log.info(f"Exported brain stats → {BRAIN_STATS}")


# ── Daily Pipeline ───────────────────────────────────────────────────────────

def run_daily(conn: sqlite3.Connection) -> None:
    """Daily pipeline: ingest new signals + backfill outcomes."""
    log.info("=== ALE Daily Pipeline ===")

    # 1. Ingest new signals
    c_count = ingest_congress_feed(conn)
    e_count = ingest_edgar_feed(conn)
    log.info(f"Ingested: {c_count} congress + {e_count} EDGAR signals")

    # 2. Update aggregate features
    agg = update_aggregate_features(conn)
    log.info(f"Updated aggregate features for {agg} ticker-date pairs")

    # 3. Backfill outcomes
    spy_index = load_price_index('SPY')
    filled = backfill_outcomes(conn, spy_index)
    log.info(f"Backfilled outcomes for {filled} signals")

    # 4. Update person track records
    person_updated = update_person_track_records(conn)
    log.info(f"Updated person track records for {person_updated} signals")

    # 5. Enrich price-based and research-backed features
    enriched = enrich_signal_features(conn)
    log.info(f"Enriched {enriched} signal features")

    # 6. Enrich market context (VIX, yield curve, credit spread from FRED)
    market_enriched = enrich_market_context(conn)
    log.info(f"Enriched {market_enriched} signals with market context")

    # 7. Generate dashboard + diagnostics
    generate_dashboard(conn)
    log.info(f"Dashboard saved to {ALE_DASHBOARD}")
    generate_analysis_report(conn)
    generate_diagnostics_html(conn)

    # 8. Score all signals with ML models
    try:
        scored = score_all_signals(conn)
        log.info(f"Scored {scored} signals with ML models")
    except Exception as e:
        log.warning(f"Signal scoring failed: {e}")

    # 9. Auto-export brain data for frontend
    export_brain_data(conn)

    # 10. Log brain run
    log_brain_run(conn, 'daily')


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

    # 3. Run walk-forward ML training
    ml_result = None
    try:
        from backtest.ml_engine import walk_forward_train
        log.info("Running walk-forward ML training...")
        ml_result = walk_forward_train(conn)
        log.info(f"ML results: {ml_result.n_folds} folds, OOS IC={ml_result.oos_ic}, "
                 f"OOS hit_rate={ml_result.oos_hit_rate}")
        if ml_result.folds:
            for i, fold in enumerate(ml_result.folds):
                log.info(f"  Fold {i+1}: {fold['test_start']}→{fold['test_end']} | "
                         f"n_test={fold['n_test']} IC={fold['ic']:.4f} hit={fold['hit_rate']:.1%}")
            if ml_result.feature_importance:
                top5 = list(ml_result.feature_importance.items())[:5]
                log.info(f"  Top features: {', '.join(f'{k}={v:.3f}' for k, v in top5)}")
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
        log.info("Running walk-forward regression...")
        reg_result = walk_forward_regression(conn)
        log.info(f"Regression results: {reg_result.n_folds} folds, OOS IC={reg_result.oos_ic}, "
                 f"OOS RMSE={reg_result.oos_rmse}")
    except ImportError:
        log.warning("ML dependencies not installed — skipping regression")
    except Exception as e:
        log.warning(f"Regression training failed: {e}")
        import traceback
        traceback.print_exc()

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
        current_weights = load_json(OPTIMAL_WEIGHTS) if OPTIMAL_WEIGHTS.exists() else {}
        current_ic = current_weights.get('_oos_ic', 0)
        # Always save ML metrics so we track IC over time.
        # Only upgrade method to 'walk_forward_ensemble' if IC improved >5%.
        output['_oos_ic'] = ml_result.oos_ic
        output['_oos_hit_rate'] = ml_result.oos_hit_rate
        output['_n_folds'] = ml_result.n_folds
        output['_feature_importance'] = ml_result.feature_importance
        if ml_result.oos_ic > current_ic * 1.05 or current_ic == 0:
            output['method'] = 'walk_forward_ensemble'
            log.info(f"Weights method upgraded to walk_forward_ensemble (IC {ml_result.oos_ic:.4f})")
        else:
            log.info(f"Method stays {output['method']} — IC {ml_result.oos_ic:.4f} vs current {current_ic:.4f} (need >5% improvement)")

    save_json(OPTIMAL_WEIGHTS, output)
    log.info(f"Updated weights saved to {OPTIMAL_WEIGHTS}")

    # 5. Update dashboard (pass ml_result for ML metrics)
    generate_dashboard(conn, ml_result=ml_result)

    # 6. Generate diagnostics
    generate_analysis_report(conn, ml_result=ml_result)
    generate_diagnostics_html(conn, ml_result=ml_result)

    # 7. Score all signals with full-sample ML models
    try:
        scored = score_all_signals(conn)
        log.info(f"Scored {scored} signals with ML models")
    except Exception as e:
        log.warning(f"Signal scoring failed: {e}")
        import traceback
        traceback.print_exc()

    # 8. Auto-export brain data for frontend
    export_brain_data(conn)

    # 9. Log brain run + self-check
    ic = ml_result.oos_ic if ml_result and ml_result.n_folds > 0 else None
    hr = ml_result.oos_hit_rate if ml_result and ml_result.n_folds > 0 else None
    log_brain_run(conn, 'analyze', oos_ic=ic, oos_hit_rate=hr)
    run_self_check(conn)


# ── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ATLAS Adaptive Learning Engine')
    parser.add_argument('--daily', action='store_true', help='Run daily collect + backfill')
    parser.add_argument('--analyze', action='store_true', help='Run feature analysis + weight update')
    parser.add_argument('--summary', action='store_true', help='Print status summary')
    parser.add_argument('--bootstrap', action='store_true', help='Run historical bootstrap')
    parser.add_argument('--diagnostics', action='store_true', help='Generate diagnostics HTML + analysis report')
    parser.add_argument('--export', action='store_true', help='Export brain_signals.json + brain_stats.json for frontend')
    parser.add_argument('--score', action='store_true', help='Score all signals with ML + export brain data')
    parser.add_argument('--self-check', action='store_true', dest='self_check', help='Run Brain health diagnostics')
    args = parser.parse_args()

    conn = init_db()

    if args.bootstrap:
        # Delegate to bootstrap script
        from backtest.bootstrap_historical import bootstrap
        bootstrap(conn)
    elif args.diagnostics:
        generate_analysis_report(conn)
        generate_diagnostics_html(conn)
        print(f"Diagnostics saved to:\n  {ALE_DIAGNOSTICS_HTML}\n  {ALE_ANALYSIS_REPORT}")
    elif args.self_check:
        health = run_self_check(conn)
        print(f"\nBrain Health: {health['status'].upper()}")
        for c in health['checks']:
            print(f"  [CHECK]   {c}")
        for w in health['warnings']:
            print(f"  [WARN]    {w}")
        for s in health['suggestions']:
            print(f"  [SUGGEST] {s}")
        print(f"\nSaved to {BRAIN_HEALTH}")
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
