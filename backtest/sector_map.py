"""
sector_map.py — Ticker-to-GICS-sector mapping for ATLAS signal tagging.

Provides a cached lookup from ticker symbol to sector name. The sector map
can be bootstrapped from yfinance (one-time) or refreshed from the Financial
Modeling Prep (FMP) API.

Usage:
    from backtest.sector_map import get_sector
    sector = get_sector("AAPL")  # "Technology"

    # One-time bootstrap (slow, ~500 tickers):
    from backtest.sector_map import bootstrap_sector_map_yfinance
    bootstrap_sector_map_yfinance(["AAPL", "LMT", "NVDA", ...])

    # Refresh from FMP API:
    from backtest.sector_map import build_sector_map
    build_sector_map(api_key="YOUR_FMP_KEY")
"""

import json
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
SECTOR_MAP_PATH = DATA_DIR / "sector_map.json"

# Module-level cache — loaded lazily on first get_sector() call
_sector_cache: dict | None = None


def _load_cache() -> dict:
    """Load sector map from disk into module cache. Returns empty dict if missing."""
    global _sector_cache
    if SECTOR_MAP_PATH.exists():
        try:
            with open(SECTOR_MAP_PATH, 'r') as f:
                _sector_cache = json.load(f)
            log.info(f"Loaded sector map: {len(_sector_cache)} tickers")
        except (json.JSONDecodeError, IOError) as e:
            log.warning(f"Failed to load sector map: {e}")
            _sector_cache = {}
    else:
        log.warning(f"Sector map not found at {SECTOR_MAP_PATH}")
        _sector_cache = {}
    return _sector_cache


def get_sector(ticker: str) -> str | None:
    """Return the GICS sector for a ticker, or None if unknown.

    Lazy-loads the sector map from data/sector_map.json on first call.
    Case insensitive — input is uppercased before lookup.
    """
    global _sector_cache
    if _sector_cache is None:
        _load_cache()
    return _sector_cache.get(ticker.upper())


def build_sector_map(api_key: str = None) -> dict:
    """Build ticker->sector map from FMP API, or fall back to cached file.

    If api_key is provided, fetches from FMP stock screener and saves to disk.
    If no key or request fails, loads the existing cached file instead.

    Returns dict like {"AAPL": "Technology", "LMT": "Industrials", ...}
    """
    global _sector_cache

    if api_key:
        try:
            import requests
            url = f"https://financialmodelingprep.com/api/v3/stock-screener?limit=10000&apikey={api_key}"
            log.info("Fetching sector map from FMP API...")
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            sector_map = {}
            for item in data:
                sym = (item.get('symbol') or '').strip().upper()
                sector = (item.get('sector') or '').strip()
                if sym and sector:
                    sector_map[sym] = sector

            log.info(f"FMP API returned {len(sector_map)} ticker->sector mappings")
            _save_map(sector_map)
            _sector_cache = sector_map
            return sector_map

        except Exception as e:
            log.warning(f"FMP API failed ({e}), falling back to cached file")

    # Fallback: load cached file
    if SECTOR_MAP_PATH.exists():
        try:
            with open(SECTOR_MAP_PATH, 'r') as f:
                sector_map = json.load(f)
            log.info(f"Loaded cached sector map: {len(sector_map)} tickers")
            _sector_cache = sector_map
            return sector_map
        except (json.JSONDecodeError, IOError) as e:
            log.warning(f"Failed to load cached sector map: {e}")

    log.warning("No sector map available (no API key and no cached file)")
    _sector_cache = {}
    return {}


def bootstrap_sector_map_yfinance(tickers: list) -> dict:
    """One-time bootstrap: fetch sector for each ticker via yfinance.

    Rate-limited to ~3 requests/second (0.3s between calls).
    Saves result to data/sector_map.json and returns the map.

    Args:
        tickers: List of ticker symbols to look up.

    Returns:
        dict mapping ticker -> sector (e.g., {"AAPL": "Technology"}).
    """
    import yfinance as yf

    # Load existing map to avoid re-fetching known tickers
    existing = {}
    if SECTOR_MAP_PATH.exists():
        try:
            with open(SECTOR_MAP_PATH, 'r') as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    sector_map = dict(existing)  # start from existing
    new_count = 0
    errors = 0

    for i, sym in enumerate(tickers):
        sym = sym.upper()
        if sym in sector_map:
            continue  # already have it

        try:
            info = yf.Ticker(sym).info
            sector = info.get('sector', '')
            if sector:
                sector_map[sym] = sector
                new_count += 1
            else:
                log.debug(f"No sector for {sym}")
        except Exception as e:
            log.debug(f"yfinance error for {sym}: {e}")
            errors += 1

        # Rate limit: 0.3s between calls
        time.sleep(0.3)

        # Progress logging every 50 tickers
        if (i + 1) % 50 == 0:
            log.info(f"Bootstrap progress: {i + 1}/{len(tickers)} ({new_count} new sectors, {errors} errors)")

    log.info(f"Bootstrap complete: {new_count} new sectors added, {errors} errors, {len(sector_map)} total")
    _save_map(sector_map)

    global _sector_cache
    _sector_cache = sector_map
    return sector_map


def _save_map(sector_map: dict) -> None:
    """Save sector map to data/sector_map.json."""
    SECTOR_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SECTOR_MAP_PATH, 'w') as f:
        json.dump(sector_map, f, indent=2, sort_keys=True)
    log.info(f"Saved sector map to {SECTOR_MAP_PATH} ({len(sector_map)} tickers)")


def get_top_tickers_from_db(db_path: Path = None, limit: int = 500) -> list:
    """Get the most-signaled tickers from the signals database.

    Args:
        db_path: Path to atlas_signals.db (defaults to data/atlas_signals.db).
        limit: Max tickers to return (default 500).

    Returns:
        List of ticker strings, ordered by signal count descending.
    """
    import sqlite3

    path = db_path or (DATA_DIR / "atlas_signals.db")
    if not path.exists():
        log.warning(f"Signals DB not found: {path}")
        return []

    conn = sqlite3.connect(str(path))
    try:
        rows = conn.execute(
            "SELECT ticker, COUNT(*) as cnt FROM signals GROUP BY ticker ORDER BY cnt DESC LIMIT ?",
            (limit,)
        ).fetchall()
        tickers = [row[0] for row in rows]
        log.info(f"Found {len(tickers)} unique tickers from signals DB (top {limit})")
        return tickers
    finally:
        conn.close()


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    parser = argparse.ArgumentParser(description='ATLAS sector map builder')
    parser.add_argument('--bootstrap', action='store_true',
                        help='Bootstrap sector map from yfinance (top 500 tickers from DB)')
    parser.add_argument('--fmp-key', type=str, default=None,
                        help='FMP API key for full refresh')
    parser.add_argument('--limit', type=int, default=500,
                        help='Max tickers for yfinance bootstrap (default 500)')
    args = parser.parse_args()

    if args.fmp_key:
        result = build_sector_map(api_key=args.fmp_key)
        print(f"FMP refresh: {len(result)} tickers mapped")
    elif args.bootstrap:
        tickers = get_top_tickers_from_db(limit=args.limit)
        if tickers:
            result = bootstrap_sector_map_yfinance(tickers)
            print(f"Bootstrap complete: {len(result)} tickers mapped")
        else:
            print("No tickers found in signals DB. Run --daily first.")
    else:
        # Just report status
        if SECTOR_MAP_PATH.exists():
            with open(SECTOR_MAP_PATH) as f:
                data = json.load(f)
            print(f"Sector map: {len(data)} tickers")
            sectors = set(data.values())
            print(f"Sectors: {sorted(sectors)}")
        else:
            print("No sector map found. Run --bootstrap or --fmp-key.")
