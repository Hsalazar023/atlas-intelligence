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
import os
import time
from pathlib import Path

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
SECTOR_MAP_PATH = DATA_DIR / "sector_map.json"
MARKET_CAP_MAP_PATH = DATA_DIR / "market_cap_map.json"

# Module-level caches — loaded lazily on first call
_sector_cache: dict | None = None
_market_cap_cache: dict | None = None


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


# ── Market Cap ────────────────────────────────────────────────────────────────

def _load_market_cap_cache() -> dict:
    """Load market cap map from disk into module cache. Returns empty dict if missing."""
    global _market_cap_cache
    if MARKET_CAP_MAP_PATH.exists():
        try:
            with open(MARKET_CAP_MAP_PATH, 'r') as f:
                _market_cap_cache = json.load(f)
            log.info(f"Loaded market cap map: {len(_market_cap_cache)} tickers")
        except (json.JSONDecodeError, IOError) as e:
            log.warning(f"Failed to load market cap map: {e}")
            _market_cap_cache = {}
    else:
        _market_cap_cache = {}
    return _market_cap_cache


def _save_market_cap_map(cap_map: dict) -> None:
    """Save market cap map to data/market_cap_map.json."""
    MARKET_CAP_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MARKET_CAP_MAP_PATH, 'w') as f:
        json.dump(cap_map, f, indent=2, sort_keys=True)
    log.info(f"Saved market cap map to {MARKET_CAP_MAP_PATH} ({len(cap_map)} tickers)")


def get_market_cap(ticker: str) -> float | None:
    """Return the market cap (in USD) for a ticker, or None if unknown.

    Lazy-loads from data/market_cap_map.json on first call.
    """
    global _market_cap_cache
    if _market_cap_cache is None:
        _load_market_cap_cache()
    return _market_cap_cache.get(ticker.upper())


def get_market_cap_bucket(ticker: str) -> str | None:
    """Return market cap bucket for a ticker.

    Buckets: mega (>200B), large (10B-200B), mid (2B-10B),
             small (300M-2B), micro (<300M).
    Returns None if market cap is unknown.
    """
    cap = get_market_cap(ticker)
    if cap is None:
        return None
    if cap > 200_000_000_000:
        return 'mega'
    if cap > 10_000_000_000:
        return 'large'
    if cap > 2_000_000_000:
        return 'mid'
    if cap > 300_000_000:
        return 'small'
    return 'micro'


def build_sector_map(api_key: str = None, tickers: list = None) -> dict:
    """Build ticker->sector map from FMP profile API, or fall back to cached file.

    If api_key is provided, fetches sector for each ticker via FMP /stable/profile
    and merges into the existing cached map. Saves result to disk.
    If no key or request fails, loads the existing cached file instead.

    Args:
        api_key: FMP API key. If None, loads cached file only.
        tickers: List of tickers to look up. If None, uses get_top_tickers_from_db().

    Returns dict like {"AAPL": "Technology", "LMT": "Industrials", ...}
    """
    global _sector_cache, _market_cap_cache

    if api_key:
        import requests

        # Load existing maps to avoid re-fetching known tickers
        existing = {}
        if SECTOR_MAP_PATH.exists():
            try:
                with open(SECTOR_MAP_PATH, 'r') as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        existing_caps = {}
        if MARKET_CAP_MAP_PATH.exists():
            try:
                with open(MARKET_CAP_MAP_PATH, 'r') as f:
                    existing_caps = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        sector_map = dict(existing)
        cap_map = dict(existing_caps)

        if tickers is None:
            tickers = get_top_tickers_from_db()
            # Fall back to existing sector map keys if DB is empty/missing
            if not tickers and sector_map:
                tickers = list(sector_map.keys())
                log.info(f"DB empty — using {len(tickers)} tickers from cached sector map")

        # Filter to tickers missing from either sector or market cap map
        to_fetch = [t for t in tickers
                    if t.upper() not in sector_map or t.upper() not in cap_map]
        log.info(f"FMP sector+cap map: {len(tickers)} tickers requested, "
                 f"{len(existing)} sectors cached, {len(existing_caps)} caps cached, "
                 f"{len(to_fetch)} to fetch")

        new_sectors = 0
        new_caps = 0
        errors = 0
        for i, sym in enumerate(to_fetch):
            sym = sym.upper()
            try:
                url = (f"https://financialmodelingprep.com/stable/profile"
                       f"?symbol={sym}&apikey={api_key}")
                resp = requests.get(url, timeout=10)
                if resp.ok:
                    data = resp.json()
                    if isinstance(data, list) and data:
                        profile = data[0]
                        sector = (profile.get('sector') or '').strip()
                        if sector and sym not in sector_map:
                            sector_map[sym] = sector
                            new_sectors += 1
                        mkt_cap = profile.get('marketCap') or profile.get('mktCap')
                        if mkt_cap and sym not in cap_map:
                            cap_map[sym] = float(mkt_cap)
                            new_caps += 1
            except Exception as e:
                errors += 1
                if errors <= 3:
                    log.debug(f"FMP profile error for {sym}: {e}")

            # Rate limit: ~5 req/sec
            time.sleep(0.2)

            if (i + 1) % 50 == 0:
                log.info(f"  ...{i+1}/{len(to_fetch)} fetched "
                         f"({new_sectors} new sectors, {new_caps} new caps)")

        log.info(f"FMP sector map: {new_sectors} new sectors, {errors} errors, "
                 f"{len(sector_map)} total")
        log.info(f"FMP market cap map: {new_caps} new caps, {len(cap_map)} total")
        _save_map(sector_map)
        _save_market_cap_map(cap_map)
        _sector_cache = sector_map
        _market_cap_cache = cap_map
        return sector_map

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

    # Load existing maps to avoid re-fetching known tickers
    existing = {}
    if SECTOR_MAP_PATH.exists():
        try:
            with open(SECTOR_MAP_PATH, 'r') as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    existing_caps = {}
    if MARKET_CAP_MAP_PATH.exists():
        try:
            with open(MARKET_CAP_MAP_PATH, 'r') as f:
                existing_caps = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    sector_map = dict(existing)
    cap_map = dict(existing_caps)
    new_sectors = 0
    new_caps = 0
    errors = 0

    for i, sym in enumerate(tickers):
        sym = sym.upper()
        if sym in sector_map and sym in cap_map:
            continue  # already have both

        try:
            info = yf.Ticker(sym).info
            sector = info.get('sector', '')
            if sector and sym not in sector_map:
                sector_map[sym] = sector
                new_sectors += 1
            mkt_cap = info.get('marketCap')
            if mkt_cap and sym not in cap_map:
                cap_map[sym] = float(mkt_cap)
                new_caps += 1
            if not sector and not mkt_cap:
                log.debug(f"No sector/cap for {sym}")
        except Exception as e:
            log.debug(f"yfinance error for {sym}: {e}")
            errors += 1

        # Rate limit: 0.3s between calls
        time.sleep(0.3)

        # Progress logging every 50 tickers
        if (i + 1) % 50 == 0:
            log.info(f"Bootstrap progress: {i + 1}/{len(tickers)} "
                     f"({new_sectors} new sectors, {new_caps} new caps, {errors} errors)")

    log.info(f"Bootstrap complete: {new_sectors} new sectors, {new_caps} new caps, "
             f"{errors} errors, {len(sector_map)} total sectors, {len(cap_map)} total caps")
    _save_map(sector_map)
    _save_market_cap_map(cap_map)

    global _sector_cache, _market_cap_cache
    _sector_cache = sector_map
    _market_cap_cache = cap_map
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
    parser.add_argument('--fmp-key', type=str,
                        default=os.environ.get('FMP_API_KEY', 'UefVEEvF1XXtpgWcsidPCGxcDJ6N0kXv'),
                        help='FMP API key for profile-based refresh (or set FMP_API_KEY env var)')
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

        if MARKET_CAP_MAP_PATH.exists():
            with open(MARKET_CAP_MAP_PATH) as f:
                cap_data = json.load(f)
            print(f"Market cap map: {len(cap_data)} tickers")
            # Show bucket distribution
            buckets = {'mega': 0, 'large': 0, 'mid': 0, 'small': 0, 'micro': 0}
            for cap in cap_data.values():
                if cap > 200_000_000_000: buckets['mega'] += 1
                elif cap > 10_000_000_000: buckets['large'] += 1
                elif cap > 2_000_000_000: buckets['mid'] += 1
                elif cap > 300_000_000: buckets['small'] += 1
                else: buckets['micro'] += 1
            print(f"Buckets: {buckets}")
        else:
            print("No market cap map found.")
