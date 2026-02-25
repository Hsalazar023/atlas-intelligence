"""
collect_prices.py — fetch and cache Finnhub daily OHLC for all signal tickers.

Usage:
    FINNHUB_KEY=xxx python backtest/collect_prices.py

Output:
    data/price_history/{TICKER}.json  — one file per ticker, date → OHLC dict
    data/price_history/SPY.json       — benchmark

Design:
    - Reads congress_feed.json and edgar_feed.json to find all tickers
    - Fetches 365 days of OHLC from Finnhub /stock/candle
    - Incremental: only fetches dates missing from cache
    - Rate-limited: 1 call/sec (Finnhub free tier: 60/min)
    - Skips tickers with no data (ETFs Finnhub doesn't cover, delisted, etc.)
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.shared import (
    CONGRESS_FEED, EDGAR_FEED, PRICE_HISTORY_DIR, DATA_DIR,
    TICKER_KEYWORDS, LOOKBACK_DAYS, RATE_LIMIT_SLEEP, BENCHMARK,
    load_json, save_json, match_edgar_ticker, date_to_ts, ts_to_date
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

FINNHUB_KEY = os.environ.get('FINNHUB_KEY', '')


def extract_tickers(trades: list, source: str = "congress") -> list:
    """Return unique ticker symbols from a list of trade/filing records."""
    tickers = set()
    for t in trades:
        if source == "congress":
            ticker = (t.get('Ticker') or '').strip().upper()
            if ticker and 1 <= len(ticker) <= 5:
                tickers.add(ticker)
        elif source == "edgar":
            ticker = match_edgar_ticker(t.get('company', ''))
            if ticker:
                tickers.add(ticker)
    return sorted(tickers)


def fetch_candles(ticker: str, from_ts: int, to_ts: int) -> dict:
    """
    Fetch daily OHLCV from Finnhub for a date range.
    Returns dict of {date_str: {o, h, l, c, v}} or empty dict on failure.
    """
    if not FINNHUB_KEY:
        raise EnvironmentError("FINNHUB_KEY not set")
    url = "https://finnhub.io/api/v1/stock/candle"
    params = {
        "symbol": ticker,
        "resolution": "D",
        "from": from_ts,
        "to": to_ts,
        "token": FINNHUB_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get('s') != 'ok' or not data.get('t'):
            return {}
        result = {}
        for i, ts in enumerate(data['t']):
            date_str = ts_to_date(ts)
            result[date_str] = {
                'o': data['o'][i],
                'h': data['h'][i],
                'l': data['l'][i],
                'c': data['c'][i],
                'v': data['v'][i],
            }
        return result
    except Exception as e:
        log.warning(f"Failed to fetch {ticker}: {e}")
        return {}


def merge_candles(existing: dict, new_candles: dict) -> dict:
    """Merge new candle data into existing cache. New data wins on conflict."""
    merged = dict(existing)
    merged.update(new_candles)
    return merged


def build_price_index(candles: dict) -> dict:
    """Return {date_str: close_price} from a candles dict."""
    return {date: v['c'] for date, v in candles.items()}


def load_cached_candles(ticker: str) -> dict:
    """Load existing price cache for a ticker, or return empty dict."""
    path = PRICE_HISTORY_DIR / f"{ticker}.json"
    if path.exists():
        return load_json(path)
    return {}


def collect_ticker(ticker: str, lookback_days: int = LOOKBACK_DAYS) -> bool:
    """
    Fetch and save price history for one ticker.
    Returns True if data was fetched successfully.
    """
    PRICE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_cached_candles(ticker)

    now = datetime.now(tz=timezone.utc)
    to_ts = int(now.timestamp())
    from_ts = int((now - timedelta(days=lookback_days)).timestamp())

    # Find the most recent cached date — only fetch from there forward
    if existing:
        latest_date = max(existing.keys())
        latest_ts = date_to_ts(latest_date)
        if latest_ts >= int((now - timedelta(days=2)).timestamp()):
            log.info(f"{ticker}: cache up to date ({latest_date}), skipping")
            return True
        from_ts = latest_ts  # re-fetch from last known date

    new_candles = fetch_candles(ticker, from_ts, to_ts)
    if not new_candles:
        log.warning(f"{ticker}: no data returned from Finnhub")
        return False

    merged = merge_candles(existing, new_candles)
    save_json(PRICE_HISTORY_DIR / f"{ticker}.json", merged)
    log.info(f"{ticker}: {len(new_candles)} new days, {len(merged)} total cached")
    return True


def main():
    if not FINNHUB_KEY:
        log.error("FINNHUB_KEY environment variable not set. Exiting.")
        sys.exit(1)

    # Load feeds
    congress_data = load_json(CONGRESS_FEED).get('trades', [])
    edgar_data = load_json(EDGAR_FEED).get('filings', [])

    # Build ticker universe
    congress_tickers = extract_tickers(congress_data, source="congress")
    edgar_tickers = extract_tickers(edgar_data, source="edgar")
    all_tickers = sorted(set(congress_tickers + edgar_tickers + [BENCHMARK]))

    log.info(f"Collecting price history for {len(all_tickers)} tickers: {all_tickers}")

    success, failed, skipped = 0, [], 0
    for i, ticker in enumerate(all_tickers):
        log.info(f"[{i+1}/{len(all_tickers)}] {ticker}")
        ok = collect_ticker(ticker)
        if ok:
            success += 1
        else:
            failed.append(ticker)
        time.sleep(RATE_LIMIT_SLEEP)

    log.info(f"\nDone. {success} succeeded, {len(failed)} failed: {failed}")
    if failed:
        log.warning(f"Failed tickers (no data / not on Finnhub): {failed}")


if __name__ == '__main__':
    main()
