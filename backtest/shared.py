"""Shared constants and helpers for ATLAS backtest scripts."""

import json
import re
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)

# Paths — all relative to the Atlas project root
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
PRICE_HISTORY_DIR = DATA_DIR / "price_history"
CONGRESS_FEED = DATA_DIR / "congress_feed.json"
EDGAR_FEED = DATA_DIR / "edgar_feed.json"
BACKTEST_RESULTS = DATA_DIR / "backtest_results.json"
OPTIMAL_WEIGHTS = DATA_DIR / "optimal_weights.json"
BACKTEST_SUMMARY = DATA_DIR / "backtest_summary.json"
SIGNALS_DB = DATA_DIR / "atlas_signals.db"
SEC_TICKERS_CACHE = DATA_DIR / "sec_tickers.json"
FMP_CONGRESS_FEED = DATA_DIR / "fmp_congress_feed.json"

# Fallback EDGAR company name → ticker mapping (used when SEC download fails)
_FALLBACK_KEYWORDS = {
    'RTX': ['raytheon', 'rtx corp'],
    'NVDA': ['nvidia'],
    'OXY': ['occidental'],
    'TMDX': ['transmedics'],
    'FCX': ['freeport'],
    'PFE': ['pfizer'],
    'TSM': ['taiwan semiconductor'],
    'META': ['meta platforms'],
    'WFRD': ['weatherford'],
    'SMPL': ['simply good', 'atkins'],
}

# Default scoring weights (hardcoded fallback if no optimal_weights.json exists)
DEFAULT_WEIGHTS = {
    "congress_tiers": {
        "small":       3,   # < $15k
        "medium":      5,   # $15k–$50k
        "large":       6,   # $50k–$100k
        "major":       8,   # $100k–$250k
        "significant": 10,  # $250k–$1M
        "xl":          15,  # $1M+
    },
    "congress_cluster_bonus": 15,     # 3+ members same ticker, 30d
    "congress_track_record_q1": 0,    # top-quartile ExcessReturn history
    "congress_track_record_q2": 0,
    "edgar_base_per_filing": 6,       # per matching Form 4 filing
    "edgar_cluster_2": 10,            # 2 filings
    "edgar_cluster_3plus": 15,        # 3+ filings
    "convergence_boost": 20,          # both congress + insider
    "decay_half_life_days": 21,       # congressional signal half-life
    "edgar_decay_half_life_days": 14, # EDGAR signal half-life
}

BENCHMARK = "SPY"
LOOKBACK_DAYS = 365
RATE_LIMIT_SLEEP = 0.5  # seconds between yfinance calls (rate-limit courtesy)

SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_TICKER_MAP_MAX_AGE = 7 * 86400  # 7 days in seconds
SEC_USER_AGENT = "ATLAS Intelligence Platform contact@atlasiq.io"

# Lazy-loaded SEC ticker map
_sec_ticker_map = None


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_sec_ticker_map() -> dict:
    """Download SEC company_tickers.json and build company_name→ticker map.
    Caches to data/sec_tickers.json (refreshed if >7 days old).
    Falls back to keyword matching on network failure."""
    # Check cache
    if SEC_TICKERS_CACHE.exists():
        age = time.time() - SEC_TICKERS_CACHE.stat().st_mtime
        if age < SEC_TICKER_MAP_MAX_AGE:
            try:
                return load_json(SEC_TICKERS_CACHE)
            except (json.JSONDecodeError, OSError):
                pass  # corrupt cache, re-download

    # Download from SEC
    try:
        import requests
        r = requests.get(
            SEC_TICKER_MAP_URL,
            headers={"User-Agent": SEC_USER_AGENT},
            timeout=15,
        )
        r.raise_for_status()
        raw = r.json()
    except Exception as e:
        log.warning(f"Failed to download SEC ticker map: {e}")
        if SEC_TICKERS_CACHE.exists():
            try:
                return load_json(SEC_TICKERS_CACHE)
            except (json.JSONDecodeError, OSError):
                pass
        return {}  # empty map, will fall back to _FALLBACK_KEYWORDS

    # Build name→ticker map, preferring shortest ticker per CIK (common stock)
    cik_best = {}
    for entry in raw.values():
        cik = entry["cik_str"]
        ticker = entry["ticker"]
        title = entry["title"].lower().strip()
        if cik not in cik_best or len(ticker) < len(cik_best[cik][1]):
            cik_best[cik] = (title, ticker)

    result = {name: ticker for name, ticker in cik_best.values()}
    save_json(SEC_TICKERS_CACHE, result)
    log.info(f"SEC ticker map: {len(result)} companies cached to {SEC_TICKERS_CACHE}")
    return result


def match_edgar_ticker(company_name: str) -> str | None:
    """Return the ticker symbol for an EDGAR company name, or None if no match.
    Uses SEC's company_tickers.json (10,000+ entries) with fallback to hardcoded keywords."""
    global _sec_ticker_map
    if _sec_ticker_map is None:
        _sec_ticker_map = load_sec_ticker_map()

    if not company_name or not company_name.strip():
        return None

    co = company_name.lower().strip()
    # Clean common state suffixes: /DE/, /NJ, /MD/, /OH/, /NV, /WV
    co_clean = re.sub(r'\s*/[A-Za-z]{2,3}/?\s*$', '', co).strip()

    if not co_clean:
        return None

    # Stage 1: SEC map — exact match
    if _sec_ticker_map:
        ticker = _sec_ticker_map.get(co_clean) or _sec_ticker_map.get(co)
        if ticker:
            return ticker
        # Stage 2: SEC map — substring match
        for title, t in _sec_ticker_map.items():
            if title in co_clean or co_clean in title:
                return t

    # Stage 3: Fallback to hardcoded keywords (network failure case)
    for ticker, keywords in _FALLBACK_KEYWORDS.items():
        if any(kw in co for kw in keywords):
            return ticker
    for ticker in _FALLBACK_KEYWORDS:
        if ticker.lower() in co:
            return ticker

    return None


def range_to_base_points(range_str: str) -> int:
    """Map a QuiverQuant Range string to base score points (before decay)."""
    r = range_str or ''
    if '$1,000,001' in r: return 15
    if '$500,001' in r:   return 12
    if '$250,001' in r:   return 10
    if '$100,001' in r:   return 8
    if '$50,001' in r:    return 6
    if '$15,001' in r:    return 5
    return 3


def date_to_ts(date_str: str) -> int:
    """Convert YYYY-MM-DD string to Unix timestamp (UTC midnight)."""
    dt = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def ts_to_date(ts: int) -> str:
    """Convert Unix timestamp to YYYY-MM-DD string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d')
