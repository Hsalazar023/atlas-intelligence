"""
bootstrap_historical.py — One-time historical data collection for the ATLAS
Adaptive Learning Engine.

Fetches ~21 months of EDGAR Form 4 filings + congressional trades, collects
price history, backfills CARs, and computes initial feature stats.

Usage:
    python backtest/bootstrap_historical.py

This populates data/atlas_signals.db and generates initial weights.
Run once, then switch to daily incremental via learning_engine.py --daily.
"""

import os
import sys
import json
import time
import logging
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.shared import (
    DATA_DIR, PRICE_HISTORY_DIR, OPTIMAL_WEIGHTS, SIGNALS_DB,
    load_json, save_json, match_edgar_ticker, range_to_base_points,
    SEC_USER_AGENT,
)
from backtest.learning_engine import (
    init_db, insert_signal, update_aggregate_features,
    backfill_outcomes, compute_feature_stats, generate_weights_from_stats,
    load_price_index, generate_dashboard, print_summary,
    update_person_track_records,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

SEC_DELAY = 0.5  # seconds between SEC requests


def _parse_efts_hits(hits: list) -> list:
    """Parse EFTS hit objects into filing dicts."""
    import re

    def is_company(name):
        upper = name.upper()
        words = ['INC', ' CORP', ' LLC', ' LTD', ' CO ', ' CO,', ' GROUP',
                 ' HOLDINGS', ' FUND', ' TRUST', ' INTERNATIONAL', ' PARTNERS',
                 ' MANAGEMENT', ' CAPITAL', ' TECHNOLOGIES', ' SYSTEMS', ' PLC']
        return any(w in upper for w in words)

    def clean_name(n):
        return re.sub(r'\s*\(CIK \d+\)\s*', '', n).strip()

    results = []
    for h in hits:
        src = h.get('_source', {})
        names = src.get('display_names', [])
        company_names = [n for n in names if is_company(n)]
        person_names = [n for n in names if not is_company(n)]
        company = clean_name(company_names[-1]) if company_names else clean_name(names[-1]) if names else 'Unknown'
        insider = clean_name(person_names[0]) if person_names else clean_name(names[0]) if names else 'Unknown'

        results.append({
            'company': company,
            'insider': insider,
            'date': src.get('file_date', ''),
            'period': src.get('period_ending', ''),
        })
    return results


def fetch_edgar_historical(days: int = 635, raw_per_month: int = 1500,
                           keep_per_month: int = 500) -> list:
    """Fetch historical Form 4 filings from EDGAR EFTS in monthly chunks,
    prioritizing filings that match known tickers (higher signal quality).

    SEC EFTS returns results newest-first and processes ~1,000+ Form 4s per day.
    We oversample each month (raw_per_month), then filter to keep only filings
    that match a known ticker via the SEC ticker map. This naturally selects
    filings from larger, more liquid companies — the ones most useful for our
    scoring engine.

    Args:
        days: How far back to look (default 635 = ~21 months)
        raw_per_month: Raw filings to fetch per month before filtering (default 1500)
        keep_per_month: Max ticker-matched filings to keep per month (default 500)
    """
    headers = {
        'User-Agent': SEC_USER_AGENT,
        'Accept': 'application/json',
    }

    # Build monthly date windows going backwards
    now = datetime.utcnow()
    month_windows = []
    cursor = now
    earliest = now - timedelta(days=days)

    while cursor > earliest:
        month_end = cursor.strftime('%Y-%m-%d')
        month_start = max(cursor.replace(day=1), earliest).strftime('%Y-%m-%d')
        month_windows.append((month_start, month_end))
        # Move to last day of previous month
        cursor = cursor.replace(day=1) - timedelta(days=1)

    all_filings = []
    page_size = 100

    log.info(f"Fetching EDGAR Form 4 filings: {len(month_windows)} months "
             f"({earliest.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}), "
             f"oversampling {raw_per_month}/month → keeping best {keep_per_month}...")

    for month_start, month_end in month_windows:
        from_idx = 0
        month_raw = []

        # Phase 1: Oversample — fetch up to raw_per_month filings
        while len(month_raw) < raw_per_month:
            url = (
                f'https://efts.sec.gov/LATEST/search-index'
                f'?forms=4'
                f'&dateRange=custom'
                f'&startdt={month_start}'
                f'&enddt={month_end}'
                f'&from={from_idx}'
            )
            try:
                r = requests.get(url, headers=headers, timeout=20)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                log.warning(f"EDGAR fetch error for {month_start}: {e}")
                break

            hits = data.get('hits', {}).get('hits', [])
            if not hits:
                break

            month_raw.extend(_parse_efts_hits(hits))

            total_available = data.get('hits', {}).get('total', {}).get('value', 0)
            from_idx += page_size
            if from_idx >= min(total_available, raw_per_month):
                break

            time.sleep(SEC_DELAY)

        # Phase 2: Filter — keep only ticker-matched filings (known companies)
        matched = []
        unmatched = []
        for f in month_raw:
            ticker = match_edgar_ticker(f.get('company', ''))
            if ticker:
                f['_matched_ticker'] = ticker
                matched.append(f)
            else:
                unmatched.append(f)

        # Take all matched filings up to keep_per_month
        month_kept = matched[:keep_per_month]

        match_rate = len(matched) / len(month_raw) * 100 if month_raw else 0
        log.info(f"  {month_start[:7]}: {len(month_raw)} raw → "
                 f"{len(matched)} matched ({match_rate:.0f}%) → "
                 f"{len(month_kept)} kept")
        all_filings.extend(month_kept)

    log.info(f"Fetched {len(all_filings)} ticker-matched EDGAR filings "
             f"across {len(month_windows)} months")
    return all_filings


def fetch_congress_trades() -> list:
    """Fetch congressional trades. Uses existing feed + QuiverQuant API if available."""
    trades = []

    # 1. Load existing congress_feed.json
    feed_path = DATA_DIR / "congress_feed.json"
    if feed_path.exists():
        data = load_json(feed_path)
        existing = data.get('trades', [])
        trades.extend(existing)
        log.info(f"Loaded {len(existing)} trades from congress_feed.json")

    # 2. Try QuiverQuant historical endpoint
    quiver_key = os.environ.get('QUIVER_KEY', '')
    if quiver_key:
        try:
            # Try historical endpoint (may not be available on free tier)
            url = 'https://api.quiverquant.com/beta/historical/congresstrading'
            headers = {
                'Authorization': f'Token {quiver_key}',
                'Accept': 'application/json',
            }
            r = requests.get(url, headers=headers, timeout=30)
            if r.ok:
                historical = r.json()
                if isinstance(historical, list):
                    log.info(f"QuiverQuant historical: {len(historical)} trades")
                    trades.extend(historical)
                else:
                    log.info("QuiverQuant historical returned non-list response")
            else:
                log.info(f"QuiverQuant historical: HTTP {r.status_code} (may need paid tier)")
        except Exception as e:
            log.warning(f"QuiverQuant historical error: {e}")

    # Deduplicate by (ticker, date, representative)
    seen = set()
    unique = []
    for t in trades:
        key = (
            (t.get('Ticker') or '').upper(),
            t.get('TransactionDate') or t.get('Date') or '',
            t.get('Representative') or '',
        )
        if key not in seen:
            seen.add(key)
            unique.append(t)

    log.info(f"Total congressional trades (deduplicated): {len(unique)}")
    return unique


def collect_price_for_ticker(ticker: str, lookback_days: int = 730) -> bool:
    """Collect price history for a single ticker using yfinance."""
    cache_path = PRICE_HISTORY_DIR / f"{ticker}.json"

    # Skip if we already have recent data
    if cache_path.exists():
        try:
            data = load_json(cache_path)
            if len(data) > 100:  # already have substantial history
                return True
        except Exception:
            pass

    try:
        import yfinance as yf
        end = datetime.now()
        start = end - timedelta(days=lookback_days)
        df = yf.download(ticker, start=start.strftime('%Y-%m-%d'),
                        end=end.strftime('%Y-%m-%d'), progress=False, auto_adjust=True)

        if df is None or df.empty:
            return False

        # yfinance >=1.2 returns MultiIndex columns
        if df.columns.nlevels > 1:
            df = df.droplevel('Ticker', axis=1)

        result = {}
        for idx, row in df.iterrows():
            date_str = idx.strftime('%Y-%m-%d')
            result[date_str] = {
                'o': round(float(row['Open']), 4),
                'h': round(float(row['High']), 4),
                'l': round(float(row['Low']), 4),
                'c': round(float(row['Close']), 4),
                'v': int(row['Volume']),
            }

        if result:
            PRICE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
            save_json(cache_path, result)
            return True
    except Exception as e:
        log.warning(f"Price fetch failed for {ticker}: {e}")

    return False


def bootstrap(conn=None):
    """Main bootstrap pipeline."""
    log.info("=" * 60)
    log.info("  ATLAS Adaptive Learning Engine — Historical Bootstrap")
    log.info("=" * 60)

    if conn is None:
        conn = init_db()

    # ── 1. Fetch historical EDGAR filings ────────────────────────────────
    log.info("\n[1/7] Fetching historical EDGAR Form 4 filings (~21 months)...")
    edgar_filings = fetch_edgar_historical(days=635, raw_per_month=1500, keep_per_month=500)

    edgar_inserted = 0
    for f in edgar_filings:
        # Ticker was pre-matched during fetch (Phase 2 filtering)
        ticker = f.get('_matched_ticker') or match_edgar_ticker(f.get('company', ''))
        if not ticker:
            continue

        signal = {
            'ticker': ticker,
            'signal_date': f.get('date', ''),
            'source': 'edgar',
            'insider_name': f.get('insider', ''),
            'representative': None,
        }
        if insert_signal(conn, signal):
            edgar_inserted += 1

    log.info(f"EDGAR: {len(edgar_filings)} pre-matched filings → {edgar_inserted} inserted")

    # ── 2. Fetch congressional trades ────────────────────────────────────
    log.info("\n[2/7] Fetching congressional trades...")
    congress_trades = fetch_congress_trades()

    congress_inserted = 0
    for t in congress_trades:
        tx = (t.get('Transaction') or '').lower()
        if 'purchase' not in tx and 'buy' not in tx:
            continue
        ticker = (t.get('Ticker') or '').strip().upper()
        if not ticker or len(ticker) > 5:
            continue
        date = t.get('TransactionDate') or t.get('Date') or ''
        if not date:
            continue

        signal = {
            'ticker': ticker,
            'signal_date': date,
            'source': 'congress',
            'representative': t.get('Representative', ''),
            'party': t.get('Party', ''),
            'chamber': t.get('Chamber', ''),
            'trade_size_range': t.get('Range', ''),
            'trade_size_points': range_to_base_points(t.get('Range', '')),
            'insider_name': None,
        }
        if insert_signal(conn, signal):
            congress_inserted += 1

    log.info(f"Congress: {len(congress_trades)} total → {congress_inserted} purchase signals inserted")

    # ── 3. Update aggregate features ─────────────────────────────────────
    log.info("\n[3/7] Computing aggregate features (clusters, convergence)...")
    agg = update_aggregate_features(conn)
    log.info(f"Updated {agg} ticker-date pairs")

    # ── 4. Collect price history ─────────────────────────────────────────
    log.info("\n[4/7] Collecting price history for all tickers...")
    tickers_row = conn.execute("SELECT DISTINCT ticker FROM signals").fetchall()
    tickers = [r['ticker'] for r in tickers_row]

    # Always ensure SPY is collected
    if 'SPY' not in tickers:
        tickers.append('SPY')

    success = 0
    for i, ticker in enumerate(tickers):
        if collect_price_for_ticker(ticker, lookback_days=1000):
            success += 1
        if (i + 1) % 50 == 0:
            log.info(f"  ...{i+1}/{len(tickers)} tickers processed ({success} with data)")
        time.sleep(0.3)  # rate-limit courtesy

    log.info(f"Price history: {success}/{len(tickers)} tickers collected")

    # ── 5. Backfill outcomes ─────────────────────────────────────────────
    log.info("\n[5/7] Backfilling outcomes (returns + CARs)...")
    spy_index = load_price_index('SPY')
    filled = backfill_outcomes(conn, spy_index)
    log.info(f"Backfilled outcomes for {filled} signals")

    # ── 6. Person track records ──────────────────────────────────────────
    log.info("\n[6/7] Computing person track records...")
    person_updated = update_person_track_records(conn)
    log.info(f"Updated track records for {person_updated} signals")

    # ── 7. Feature analysis + weight generation ──────────────────────────
    log.info("\n[7/7] Running feature analysis...")
    stats = compute_feature_stats(conn)
    if stats:
        weights = generate_weights_from_stats(conn)
        threshold = weights.pop('_optimal_threshold', 65)
        output = {
            **weights,
            "generated": datetime.now(tz=timezone.utc).isoformat(),
            "optimal_threshold": threshold,
            "method": "bootstrap_feature_importance",
        }
        save_json(OPTIMAL_WEIGHTS, output)
        log.info(f"Initial weights saved to {OPTIMAL_WEIGHTS}")
    else:
        log.info("Not enough outcome data yet for feature analysis")

    # Generate dashboard
    generate_dashboard(conn)

    # Print summary
    print_summary(conn)

    log.info("Bootstrap complete!")
    return conn


if __name__ == '__main__':
    bootstrap()
