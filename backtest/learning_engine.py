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
    load_json, save_json, match_edgar_ticker, range_to_base_points,
)
from backtest.sector_map import get_sector

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

ALE_DASHBOARD = DATA_DIR / "ale_dashboard.json"

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

    -- Market context
    sector TEXT,

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
            "SELECT COUNT(*) as cnt FROM signals WHERE ticker=? AND signal_date BETWEEN ? AND ?",
            (ticker, d7, signal_date)
        ).fetchone()['cnt']
        count_30d = conn.execute(
            "SELECT COUNT(*) as cnt FROM signals WHERE ticker=? AND signal_date BETWEEN ? AND ?",
            (ticker, d30, signal_date)
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

    # EDGAR: track by insider_name
    edgar_rows = conn.execute(
        "SELECT id, insider_name, signal_date "
        "FROM signals WHERE source='edgar' AND insider_name != '' "
        "ORDER BY signal_date ASC"
    ).fetchall()

    insider_history = defaultdict(list)
    for row in edgar_rows:
        insider = row['insider_name']
        sig_id = row['id']

        prior = insider_history.get(insider, [])
        prior_with_outcomes_30 = [p for p in prior if p['car_30d'] is not None]
        prior_with_outcomes_90 = [p for p in prior if p['car_90d'] is not None]

        trade_count = len(prior)
        hit_rate_30 = None
        avg_car_30 = None
        hit_rate_90 = None
        avg_car_90 = None

        if prior_with_outcomes_30:
            cars_30 = [p['car_30d'] for p in prior_with_outcomes_30]
            hit_rate_30 = round(sum(1 for c in cars_30 if c > 0) / len(cars_30), 4)
            avg_car_30 = round(sum(cars_30) / len(cars_30), 6)

        if prior_with_outcomes_90:
            cars_90 = [p['car_90d'] for p in prior_with_outcomes_90]
            hit_rate_90 = round(sum(1 for c in cars_90 if c > 0) / len(cars_90), 4)
            avg_car_90 = round(sum(cars_90) / len(cars_90), 6)

        conn.execute(
            """UPDATE signals SET
                person_trade_count=?, person_hit_rate_30d=?, person_avg_car_30d=?,
                person_hit_rate_90d=?, person_avg_car_90d=?
            WHERE id=?""",
            (trade_count, hit_rate_30, avg_car_30, hit_rate_90, avg_car_90, sig_id)
        )
        updated += 1

        outcome = conn.execute(
            "SELECT car_30d, car_90d FROM signals WHERE id=?", (sig_id,)
        ).fetchone()
        insider_history[insider].append({
            'date': row['signal_date'],
            'car_30d': outcome['car_30d'] if outcome else None,
            'car_90d': outcome['car_90d'] if outcome else None,
        })

    conn.commit()
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
    """Get the return from start_date to start_date+days (with ±5 day tolerance)."""
    if start_date not in price_index:
        return None
    base = price_index[start_date]
    if not base or base == 0:
        return None

    dt = datetime.strptime(start_date, '%Y-%m-%d')
    target = dt + timedelta(days=days)
    for offset in sorted(range(-5, 6), key=abs):
        candidate = (target + timedelta(days=offset)).strftime('%Y-%m-%d')
        if candidate == start_date:
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

    # Get signals needing backfill
    rows = conn.execute(
        "SELECT id, ticker, signal_date, outcome_5d_filled, outcome_30d_filled, "
        "outcome_90d_filled, outcome_180d_filled, outcome_365d_filled "
        "FROM signals WHERE outcome_365d_filled = 0"
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

        # Get price at signal
        price_at = price_index.get(row['signal_date'])

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

            if stock_ret is not None:
                updates[f'return_{window}d'] = round(stock_ret, 6)
                if spy_ret is not None:
                    updates[f'car_{window}d'] = round(stock_ret - spy_ret, 6)
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


def compute_feature_stats(conn: sqlite3.Connection) -> dict:
    """Compute per-feature hit rates and average CARs. Updates feature_stats table."""
    now = datetime.now(tz=timezone.utc).isoformat()

    # Get all signals with 30d outcomes
    rows = conn.execute(
        "SELECT * FROM signals WHERE outcome_30d_filled = 1 AND car_30d IS NOT NULL"
    ).fetchall()

    if not rows:
        log.warning("No signals with 30d outcomes — cannot compute feature stats.")
        return {}

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
        ('person_accuracy', lambda r: 'strong_70+' if (r['person_hit_rate_30d'] or 0) >= 0.7 else
                                      'good_55+' if (r['person_hit_rate_30d'] or 0) >= 0.55 else
                                      'weak_<55' if r['person_hit_rate_30d'] is not None else 'n/a'),
        ('relative_position', lambda r: 'oversized_2x+' if (r['relative_position_size'] or 0) >= 2.0 else
                                        'large_1.5x+' if (r['relative_position_size'] or 0) >= 1.5 else
                                        'typical' if (r['relative_position_size'] or 0) >= 0.5 else
                                        'small' if r['relative_position_size'] is not None else 'n/a'),
        # ALE v2 features
        ('convergence_tier', lambda r: str(r['convergence_tier'] or 0)),
        ('sector', lambda r: r['sector'] or 'n/a'),
        ('cluster_velocity', lambda r: r['cluster_velocity'] or 'n/a'),
        ('disclosure_delay', lambda r: 'urgent' if (r['disclosure_delay'] or 999) < 7 else
                                        'normal' if (r['disclosure_delay'] or 999) < 30 else
                                        'slow' if (r['disclosure_delay'] or 999) < 45 else
                                        'late' if r['disclosure_delay'] is not None else 'n/a'),
        ('insider_role', lambda r: _normalize_role(r['insider_role']) if r['insider_role'] else 'n/a'),
        ('trade_pattern', lambda r: r['trade_pattern'] or 'n/a'),
        ('price_proximity', lambda r: 'near_low' if (r['price_proximity_52wk'] or 0.5) < 0.2 else
                                       'lower_half' if (r['price_proximity_52wk'] or 0.5) < 0.5 else
                                       'upper_half' if (r['price_proximity_52wk'] or 0.5) < 0.8 else
                                       'near_high' if r['price_proximity_52wk'] is not None else 'n/a'),
        ('market_cap', lambda r: r['market_cap_bucket'] or 'n/a'),
    ]

    stats = {}
    for feature_name, extractor in feature_extractors:
        buckets = defaultdict(list)
        for r in rows:
            val = extractor(r)
            if val and val != 'n/a':
                buckets[val].append(r['car_30d'])

        for value, cars in buckets.items():
            if len(cars) < 3:  # minimum observations
                continue
            n = len(cars)
            hit_rate = sum(1 for c in cars if c > 0) / n
            avg_car = sum(cars) / n

            # Also compute longer-horizon stats if available
            extra_cars = {}
            for horizon in ('90d', '180d', '365d'):
                col = f'car_{horizon}'
                vals = [r[col] for r in rows if extractor(r) == value and r[col] is not None]
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

    # Scoring performance
    all_30d = conn.execute(
        "SELECT car_30d FROM signals WHERE outcome_30d_filled=1 AND car_30d IS NOT NULL"
    ).fetchall()
    overall_hit = sum(1 for r in all_30d if r['car_30d'] > 0) / len(all_30d) if all_30d else 0
    overall_avg = sum(r['car_30d'] for r in all_30d) / len(all_30d) if all_30d else 0

    conv_30d = conn.execute(
        "SELECT car_30d FROM signals WHERE outcome_30d_filled=1 AND car_30d IS NOT NULL AND has_convergence=1"
    ).fetchall()
    conv_hit = sum(1 for r in conv_30d if r['car_30d'] > 0) / len(conv_30d) if conv_30d else 0
    conv_avg = sum(r['car_30d'] for r in conv_30d) / len(conv_30d) if conv_30d else 0

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
            "n_with_30d_outcomes": len(all_30d),
            "n_convergence_signals": len(conv_30d),
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

    # 5. Generate dashboard
    generate_dashboard(conn)
    log.info(f"Dashboard saved to {ALE_DASHBOARD}")


def run_analyze(conn: sqlite3.Connection) -> None:
    """Weekly analysis: compute feature stats + ML walk-forward + update weights."""
    log.info("=== ALE Feature Analysis ===")

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
    except ImportError:
        log.warning("ML dependencies not installed (scikit-learn, lightgbm) — skipping ML training")
    except Exception as e:
        log.warning(f"ML training failed: {e}")

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
        if ml_result.oos_ic > current_ic * 1.05 or current_ic == 0:
            output['_oos_ic'] = ml_result.oos_ic
            output['_oos_hit_rate'] = ml_result.oos_hit_rate
            output['_feature_importance'] = ml_result.feature_importance
            output['method'] = 'walk_forward_ensemble'
            log.info("Weights updated — ML outperformed previous by >5%")
        else:
            log.info(f"Weights NOT updated — ML IC {ml_result.oos_ic} vs current {current_ic}")

    save_json(OPTIMAL_WEIGHTS, output)
    log.info(f"Updated weights saved to {OPTIMAL_WEIGHTS}")

    # 5. Update dashboard (pass ml_result for ML metrics)
    generate_dashboard(conn, ml_result=ml_result)


# ── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ATLAS Adaptive Learning Engine')
    parser.add_argument('--daily', action='store_true', help='Run daily collect + backfill')
    parser.add_argument('--analyze', action='store_true', help='Run feature analysis + weight update')
    parser.add_argument('--summary', action='store_true', help='Print status summary')
    parser.add_argument('--bootstrap', action='store_true', help='Run historical bootstrap')
    args = parser.parse_args()

    conn = init_db()

    if args.bootstrap:
        # Delegate to bootstrap script
        from backtest.bootstrap_historical import bootstrap
        bootstrap(conn)
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
