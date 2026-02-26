"""Shared constants and helpers for ATLAS backtest scripts."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Paths — all relative to the Atlas project root
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
PRICE_HISTORY_DIR = DATA_DIR / "price_history"
CONGRESS_FEED = DATA_DIR / "congress_feed.json"
EDGAR_FEED = DATA_DIR / "edgar_feed.json"
BACKTEST_RESULTS = DATA_DIR / "backtest_results.json"
OPTIMAL_WEIGHTS = DATA_DIR / "optimal_weights.json"
BACKTEST_SUMMARY = DATA_DIR / "backtest_summary.json"

# EDGAR company name → ticker mapping (mirrors TICKER_KEYWORDS in atlas-intelligence.html)
TICKER_KEYWORDS = {
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


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def match_edgar_ticker(company_name: str) -> str | None:
    """Return the ticker symbol for an EDGAR company name, or None if no match."""
    co = company_name.lower()
    for ticker, keywords in TICKER_KEYWORDS.items():
        if any(kw in co for kw in keywords):
            return ticker
    # Stage 2: check if any ticker symbol appears in company name
    for ticker in TICKER_KEYWORDS:
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
