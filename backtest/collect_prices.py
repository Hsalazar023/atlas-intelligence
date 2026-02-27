"""
collect_prices.py — fetch and cache daily OHLC via yfinance for all signal tickers.

Usage:
    python backtest/collect_prices.py

Output:
    data/price_history/{TICKER}.json  — one file per ticker, date → OHLC dict
    data/price_history/SPY.json       — benchmark

Design:
    - Reads congress_feed.json and edgar_feed.json to find all tickers
    - Fetches up to 365 days of OHLC from Yahoo Finance (yfinance)
    - Incremental: only fetches dates missing from cache
    - No API key required
    - Skips tickers with no data (delisted, etc.)
"""

import sys
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yfinance as yf

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.shared import (
    CONGRESS_FEED, EDGAR_FEED, PRICE_HISTORY_DIR,
    LOOKBACK_DAYS, BENCHMARK,
    load_json, save_json, match_edgar_ticker,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


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


def fetch_candles(ticker: str, start_date: str, end_date: str) -> dict:
    """
    Fetch daily OHLCV from Yahoo Finance for a date range.
    start_date/end_date are YYYY-MM-DD strings.
    Returns dict of {date_str: {o, h, l, c, v}} or empty dict on failure.
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return {}
        # yfinance >=1.2 returns MultiIndex columns (Price, Ticker) even for single ticker
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
    end_date = (now + timedelta(days=1)).strftime('%Y-%m-%d')  # yfinance end is exclusive

    # Find the most recent cached date — only fetch from there forward
    if existing:
        latest_date = max(existing.keys())
        latest_dt = datetime.strptime(latest_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        if latest_dt >= (now - timedelta(days=2)):
            log.info(f"{ticker}: cache up to date ({latest_date}), skipping")
            return True
        start_date = latest_date  # re-fetch from last known date
    else:
        start_date = (now - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

    new_candles = fetch_candles(ticker, start_date, end_date)
    if not new_candles:
        log.warning(f"{ticker}: no data returned from yfinance")
        return False

    merged = merge_candles(existing, new_candles)
    save_json(PRICE_HISTORY_DIR / f"{ticker}.json", merged)
    log.info(f"{ticker}: {len(new_candles)} new days, {len(merged)} total cached")
    return True


def main():
    # Load feeds
    congress_data = load_json(CONGRESS_FEED).get('trades', [])
    edgar_data = load_json(EDGAR_FEED).get('filings', [])

    # Build ticker universe
    congress_tickers = extract_tickers(congress_data, source="congress")
    edgar_tickers = extract_tickers(edgar_data, source="edgar")
    all_tickers = sorted(set(congress_tickers + edgar_tickers + [BENCHMARK]))

    log.info(f"Collecting price history for {len(all_tickers)} tickers: {all_tickers}")

    success, failed = 0, []
    for i, ticker in enumerate(all_tickers):
        log.info(f"[{i+1}/{len(all_tickers)}] {ticker}")
        ok = collect_ticker(ticker)
        if ok:
            success += 1
        else:
            failed.append(ticker)

    log.info(f"\nDone. {success} succeeded, {len(failed)} failed: {failed}")
    if failed:
        log.warning(f"Failed tickers (no data on Yahoo Finance): {failed}")


if __name__ == '__main__':
    main()
