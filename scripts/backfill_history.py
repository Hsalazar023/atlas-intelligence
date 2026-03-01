#!/usr/bin/env python3
"""
ATLAS Historical Data Backfill (One-Time)
==========================================
Fetches deep historical data from FMP (Congress) and SEC EDGAR (Form 4).
Run once to bootstrap the training database with 5+ years of history.

Usage:
  python3 scripts/backfill_history.py --congress   # Deep congress history (100 pages)
  python3 scripts/backfill_history.py --edgar       # Deep EDGAR history (5 years)
  python3 scripts/backfill_history.py --prices      # Extended price backfill (3 years)
  python3 scripts/backfill_history.py --all         # All of the above

Environment variables:
  FMP_API_KEY — Financial Modeling Prep API key (for congress data)

Notes:
  - Congress: 100 pages from FMP → ~5,000-10,000 trades back to ~2019
  - EDGAR: walks EFTS backwards month-by-month for 5 years (rate-limited 0.5s/req)
  - Prices: 3-year lookback for all tickers in the DB, fills 180d/365d CAR gaps
  - After running, use `python -m backtest.learning_engine --daily` to ingest
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Missing 'requests'. Run: pip3 install requests")

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.shared import (
    DATA_DIR, CONGRESS_FEED, EDGAR_FEED, PRICE_HISTORY_DIR,
    load_json, save_json, match_edgar_ticker,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

USER_AGENT = 'ATLAS Intelligence Platform contact@atlasiq.io'
HEADERS = {'User-Agent': USER_AGENT, 'Accept-Encoding': 'gzip, deflate'}
SEC_DELAY = 0.5


# ── Deep Congress History ─────────────────────────────────────────────────────

def backfill_congress(pages: int = 100):
    """Fetch deep congressional trade history from FMP. 100 pages ≈ 5,000-10,000 trades."""
    api_key = os.environ.get('FMP_API_KEY', '')
    if not api_key:
        log.error("FMP_API_KEY not set — cannot fetch congress history")
        return

    # Import the existing FMP fetcher
    sys.path.insert(0, str(Path(__file__).parent))
    from fetch_data import fetch_fmp_congress

    log.info(f"Fetching {pages} pages of congressional trades from FMP...")
    trades = fetch_fmp_congress(api_key, pages=pages)
    log.info(f"Fetched {len(trades)} total congressional trades")

    # Merge with existing feed
    existing = load_json(CONGRESS_FEED) if CONGRESS_FEED.exists() else {}
    existing_trades = existing.get('trades', [])

    # Deduplicate by (ticker, date, representative)
    seen = set()
    merged = []
    for t in trades + existing_trades:
        key = (t.get('Ticker', ''), t.get('TransactionDate', ''), t.get('Representative', ''))
        if key not in seen:
            seen.add(key)
            merged.append(t)

    merged.sort(key=lambda x: x.get('TransactionDate', ''), reverse=True)

    output = {
        'generated': datetime.now(tz=timezone.utc).isoformat(),
        'source': 'FMP Congressional Trading API (deep backfill)',
        'count': len(merged),
        'trades': merged,
    }
    save_json(CONGRESS_FEED, output)
    log.info(f"Saved {len(merged)} trades to congress_feed.json (was {len(existing_trades)})")


# ── Deep EDGAR History ────────────────────────────────────────────────────────

def backfill_edgar(months_back: int = 60):
    """Walk EFTS backwards month-by-month for Form 4 filings.
    SEC rate-limited at 0.5s per request. 60 months = ~120 requests."""
    log.info(f"Fetching EDGAR Form 4 filings for last {months_back} months...")

    all_filings = []
    now = datetime.now(tz=timezone.utc)

    for i in range(months_back):
        end_dt = now - timedelta(days=30 * i)
        start_dt = end_dt - timedelta(days=30)
        start_str = start_dt.strftime('%Y-%m-%d')
        end_str = end_dt.strftime('%Y-%m-%d')

        url = (
            'https://efts.sec.gov/LATEST/search-index'
            '?forms=4'
            '&dateRange=custom'
            f'&startdt={start_str}'
            f'&enddt={end_str}'
            '&from=0&size=200'
        )

        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                hits = data.get('hits', {}).get('hits', [])
                for hit in hits:
                    src = hit.get('_source', {})
                    names = src.get('display_names', [])
                    filing = {
                        'company': names[0] if names else '',
                        'date': src.get('file_date', ''),
                        'insider': names[1] if len(names) > 1 else '',
                    }
                    if filing['company'] and filing['date']:
                        all_filings.append(filing)
                log.info(f"  [{i+1}/{months_back}] {start_str} to {end_str}: {len(hits)} filings")
            else:
                log.warning(f"  [{i+1}/{months_back}] HTTP {resp.status_code}")
        except Exception as e:
            log.warning(f"  [{i+1}/{months_back}] Error: {e}")

        time.sleep(SEC_DELAY)

    # Merge with existing
    existing = load_json(EDGAR_FEED) if EDGAR_FEED.exists() else {}
    existing_filings = existing.get('filings', [])

    # Deduplicate by (company, date, insider)
    seen = set()
    merged = []
    for f in all_filings + existing_filings:
        key = (f.get('company', ''), f.get('date', ''), f.get('insider', ''))
        if key not in seen:
            seen.add(key)
            merged.append(f)

    merged.sort(key=lambda x: x.get('date', ''), reverse=True)

    output = {
        'generated': datetime.now(tz=timezone.utc).isoformat(),
        'source': 'SEC EDGAR EFTS Form 4 (deep backfill)',
        'count': len(merged),
        'filings': merged,
    }
    save_json(EDGAR_FEED, output)
    log.info(f"Saved {len(merged)} filings to edgar_feed.json (was {len(existing_filings)})")


# ── Extended Price Backfill ───────────────────────────────────────────────────

def backfill_prices(lookback_days: int = 1095):
    """Collect 3 years of price history for all tickers in the DB.
    Uses yfinance. Fills gaps for 180d and 365d CAR computation."""
    try:
        from backtest.collect_prices import collect_ticker, extract_tickers
    except ImportError:
        log.error("collect_prices not available")
        return

    # Get all tickers from existing feeds
    congress_data = load_json(CONGRESS_FEED).get('trades', []) if CONGRESS_FEED.exists() else []
    edgar_data = load_json(EDGAR_FEED).get('filings', []) if EDGAR_FEED.exists() else []

    congress_tickers = extract_tickers(congress_data, source='congress')
    edgar_tickers = extract_tickers(edgar_data, source='edgar')
    all_tickers = sorted(set(congress_tickers + edgar_tickers + ['SPY']))

    log.info(f"Backfilling {lookback_days}-day price history for {len(all_tickers)} tickers")

    success, failed = 0, []
    for i, ticker in enumerate(all_tickers):
        log.info(f"  [{i+1}/{len(all_tickers)}] {ticker}")
        try:
            if collect_ticker(ticker, lookback_days=lookback_days):
                success += 1
            else:
                failed.append(ticker)
        except Exception as e:
            log.warning(f"  {ticker}: {e}")
            failed.append(ticker)

    log.info(f"Price backfill: {success} succeeded, {len(failed)} failed")
    if failed:
        log.warning(f"Failed: {failed}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='ATLAS Historical Data Backfill')
    parser.add_argument('--congress', action='store_true', help='Deep congress history (100 pages)')
    parser.add_argument('--edgar', action='store_true', help='Deep EDGAR history (5 years)')
    parser.add_argument('--prices', action='store_true', help='Extended price backfill (3 years)')
    parser.add_argument('--all', action='store_true', help='All of the above')
    parser.add_argument('--pages', type=int, default=100, help='FMP pages for congress (default: 100)')
    parser.add_argument('--months', type=int, default=60, help='Months back for EDGAR (default: 60)')
    parser.add_argument('--lookback', type=int, default=1095, help='Price lookback days (default: 1095)')
    args = parser.parse_args()

    if not any([args.congress, args.edgar, args.prices, args.all]):
        parser.print_help()
        return

    if args.all or args.congress:
        backfill_congress(pages=args.pages)

    if args.all or args.edgar:
        backfill_edgar(months_back=args.months)

    if args.all or args.prices:
        backfill_prices(lookback_days=args.lookback)


if __name__ == '__main__':
    main()
