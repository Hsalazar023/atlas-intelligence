# Scoring Backtest Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a self-improving scoring backtest engine that (1) immediately fixes the Watchlist tier so scored tickers show up, (2) runs a proper event-study CAR analysis on all signals, and (3) automatically optimizes score weights weekly via GitHub Actions.

**Architecture:** Three Python scripts in `backtest/` handle price collection, CAR computation, and weight optimization. GitHub Actions runs them every Sunday. `atlas-intelligence.html` loads `data/optimal_weights.json` on startup and applies dynamic scoring weights with signal decay.

**Tech Stack:** Python 3.11, pandas, numpy, finnhub-python, pytest. No new npm packages. No database — flat JSON files only.

---

## Background: Key Data Schemas

**`data/congress_feed.json`:**
```json
{
  "trades": [
    {
      "Representative": "Scott Franklin",
      "TransactionDate": "2026-02-10",
      "ReportDate": "2026-02-23",
      "Ticker": "HSY",
      "Transaction": "Sale",
      "Range": "$1,001 - $15,000",
      "House": "Representatives",
      "Amount": "1001.0",
      "Party": "R",
      "ExcessReturn": 0.807,
      "PriceChange": -0.331,
      "SPYChange": -1.139
    }
  ]
}
```

**`data/edgar_feed.json`:**
```json
{
  "filings": [
    {
      "company": "AppLovin Corp",
      "insider": "Valenzuela Victoria",
      "date": "2026-02-24",
      "period": "2026-02-20",
      "accession": "0001846998-26-000002",
      "link": "https://www.sec.gov/..."
    }
  ]
}
```

**Finnhub candle response:**
```json
{
  "s": "ok",
  "t": [1609459200, 1609545600],
  "o": [128.0, 130.0],
  "h": [133.0, 132.0],
  "l": [126.0, 128.0],
  "c": [131.0, 131.5],
  "v": [100000, 95000]
}
```

**Key constant (mirrors `TICKER_KEYWORDS` in HTML):**
```python
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
```

---

## Task 0: Immediate Frontend Fix — Watchlist Tier + Signal Decay

This is the quick fix for "score is 65, threshold is 85, nothing shows." No Python, no tests — just HTML edits. Do this first so the platform shows something useful while the backtest engine is built.

**Files:**
- Modify: `atlas-intelligence.html`

**Step 1: Add signal decay to `scoreCongressTicker()`**

Find the function `scoreCongressTicker` (around line 1881). Currently it scores all trades equally regardless of age. Add decay so trades lose half their points after 21 days.

Find this line inside `scoreCongressTrades` after `var pts=0;`:
```javascript
  var pts=0;
  buys.forEach(function(t){
    var r=t.Range||'';
    if(r.indexOf('$1,000,001')>=0)       pts+=15;
```

Replace with:
```javascript
  var pts=0;
  var now=Date.now();
  buys.forEach(function(t){
    var r=t.Range||'';
    var rawPts=0;
    if(r.indexOf('$1,000,001')>=0)       rawPts=15;
    else if(r.indexOf('$500,001')>=0)    rawPts=12;
    else if(r.indexOf('$250,001')>=0)    rawPts=10;
    else if(r.indexOf('$100,001')>=0)    rawPts=8;
    else if(r.indexOf('$50,001')>=0)     rawPts=6;
    else if(r.indexOf('$15,001')>=0)     rawPts=5;
    else                                  rawPts=3;
    // Signal decay: half-life = 21 days (will be tuned by backtest)
    var daysSince=(now-new Date(t.TransactionDate||t.Date||now))/86400000;
    var decayFactor=Math.pow(0.5,Math.max(0,daysSince)/21);
    pts+=rawPts*decayFactor;
  });
```

Also remove the old individual `pts+=` lines that follow (they're now inside the loop above — delete:
```javascript
    if(r.indexOf('$1,000,001')>=0)       pts+=15;
    else if(r.indexOf('$500,001')>=0)    pts+=12;
    else if(r.indexOf('$250,001')>=0)    pts+=10;
    else if(r.indexOf('$100,001')>=0)    pts+=8;
    else if(r.indexOf('$50,001')>=0)     pts+=6;
    else if(r.indexOf('$15,001')>=0)     pts+=5;
    else                                  pts+=3;
```

**Step 2: Add Watchlist tier to `renderSignalIdeas()`**

Find `renderSignalIdeas()` (around line 2154). Replace the full function with:

```javascript
function renderSignalIdeas(){
  var container=document.getElementById('idea-cards-container');
  if(!container) return;
  var IDEA_THRESHOLD=window.SCORE_THRESHOLD||65;
  var WATCH_THRESHOLD=40;
  var universe=buildTickerUniverse();
  var all=universe.map(function(ticker){
    var sc=computeConvergenceScore(ticker);
    return {ticker:ticker,sc:sc};
  }).filter(function(r){ return r.sc.total>=WATCH_THRESHOLD; })
    .sort(function(a,b){ return b.sc.total-a.sc.total; });
  var ideas=all.filter(function(r){ return r.sc.total>=IDEA_THRESHOLD; });
  var watchlist=all.filter(function(r){ return r.sc.total>=WATCH_THRESHOLD&&r.sc.total<IDEA_THRESHOLD; });
  var activeEl=document.getElementById('ideas-active-count');
  if(activeEl) activeEl.textContent=ideas.length;
  if(!all.length){
    container.innerHTML='<div style="padding:32px 20px;text-align:center;color:var(--muted)">'
      +'<div style="font-size:22px;margin-bottom:8px">📡</div>'
      +'<div style="font-size:13px;font-weight:600;color:var(--text-s);margin-bottom:6px">No signals above threshold</div>'
      +'<div style="font-size:11px;line-height:1.7">Convergence engine monitoring all feeds.<br>Ideas generated when score ≥ '+IDEA_THRESHOLD+'.</div>'
      +'<div style="margin-top:12px;font-size:10px;color:var(--text-f)">Highest score: <span id="ideas-max-score-inline">—</span></div>'
      +'</div>';
    return;
  }
  var now=new Date().toLocaleDateString('en-US',{month:'short',day:'numeric',year:'numeric'});
  var html='';
  // Trade Ideas section
  if(ideas.length){
    html+='<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:var(--blue-lt);margin-bottom:10px">Trade Ideas — Score ≥ '+IDEA_THRESHOLD+'</div>';
    html+=ideas.map(function(r){
      var sc=r.sc; var ticker=r.ticker;
      var parts=[];
      if(sc.congress.count>0) parts.push(sc.congress.count+' congressional buy'+(sc.congress.count>1?'s':''));
      if(sc.insider.count>0) parts.push(sc.insider.count+' EDGAR filing'+(sc.insider.count>1?'s':''));
      var signal=parts.join(' · ')||'Convergence signal';
      var scoreCls=sc.total>=95?'sc4':sc.total>=IDEA_THRESHOLD?'sc3':'sc1';
      var tag=sc.hasConvergence?'<span class="tag t-clus">Multi-Source</span>':sc.congress.count>0?'<span class="tag t-warn">Congress</span>':'<span class="tag t-buy">EDGAR</span>';
      return '<div class="idea-card" style="border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:12px">'
        +'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px">'
        +'<div><div style="display:flex;align-items:center;gap:6px"><span class="sym" style="font-size:15px;font-weight:700">'+ticker+'</span>'+tag+'</div>'
        +'<div class="sym-sub" style="margin-top:3px">'+signal+'</div></div>'
        +'<span class="score '+scoreCls+'" style="font-size:17px;flex-shrink:0">'+Math.round(sc.total)+'</span>'
        +'</div>'
        +'<div style="font-size:10px;color:var(--muted)">Engine-generated · '+now+' · Convergence boost: +'+sc.boost+'pts</div>'
        +'</div>';
    }).join('');
  }
  // Watchlist section
  if(watchlist.length){
    html+='<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:var(--muted);margin:16px 0 10px">Monitoring — Score '+WATCH_THRESHOLD+'–'+(IDEA_THRESHOLD-1)+'</div>';
    html+=watchlist.slice(0,8).map(function(r){
      var sc=r.sc; var ticker=r.ticker;
      var parts=[];
      if(sc.congress.count>0) parts.push(sc.congress.count+' cong. trade'+(sc.congress.count>1?'s':''));
      if(sc.insider.count>0) parts.push(sc.insider.count+' EDGAR filing'+(sc.insider.count>1?'s':''));
      var signal=parts.join(' · ')||'Signal activity';
      return '<div style="display:flex;justify-content:space-between;align-items:center;padding:10px 12px;border:1px solid var(--border);border-radius:6px;margin-bottom:6px">'
        +'<div><div class="sym" style="font-size:13px;font-weight:600">'+ticker+'</div>'
        +'<div class="sym-sub" style="margin-top:1px">'+signal+'</div></div>'
        +'<span class="score sc1" style="font-size:13px">'+Math.round(sc.total)+'</span>'
        +'</div>';
    }).join('');
  }
  container.innerHTML=html;
  // Update score displays
  var maxScore=all.length?Math.round(all[0].sc.total):0;
  var msEl=document.getElementById('ideas-max-score');
  if(msEl) msEl.textContent=maxScore||'—';
  var msBoxEl=document.getElementById('ideas-max-score-box');
  if(msBoxEl) msBoxEl.textContent=maxScore||'—';
  var msInline=document.getElementById('ideas-max-score-inline');
  if(msInline) msInline.textContent=maxScore||'—';
}
```

**Step 3: Update the static threshold label in HTML**

Find the static label `≥ 85` in the acad-box (around line 942):
```html
<div class="acad-num">≥ 85</div><div class="acad-lbl">Min Threshold</div>
```
Replace with:
```html
<div class="acad-num" id="ideas-threshold-display">≥ 65</div><div class="acad-lbl">Min Threshold</div>
```

**Step 4: Verify manually**

```bash
cd /Users/henrysalazar/Desktop/Atlas && python3 -m http.server 8080
```
Open http://localhost:8080. Navigate to Trade Ideas. Should now see:
- Tickers scoring 40-64 in "Monitoring" section
- Tickers scoring 65+ in "Trade Ideas" section (if any)
- Current highest score displayed

**Step 5: Commit**

```bash
git add atlas-intelligence.html
git commit -m "fix: add Watchlist tier (40-64) + signal decay + lower threshold to 65"
```

---

## Task 1: Python Project Structure

**Files:**
- Create: `backtest/__init__.py`
- Create: `backtest/requirements.txt`
- Create: `backtest/tests/__init__.py`
- Create: `backtest/shared.py` (shared constants + helpers)

**Step 1: Create directory structure**

```bash
mkdir -p /Users/henrysalazar/Desktop/Atlas/backtest/tests
touch /Users/henrysalazar/Desktop/Atlas/backtest/__init__.py
touch /Users/henrysalazar/Desktop/Atlas/backtest/tests/__init__.py
```

**Step 2: Create `backtest/requirements.txt`**

```
finnhub-python==2.4.20
pandas==2.2.0
numpy==1.26.4
pytest==8.1.1
```

**Step 3: Install locally**

```bash
pip install -r /Users/henrysalazar/Desktop/Atlas/backtest/requirements.txt
```

Expected: packages install without errors.

**Step 4: Create `backtest/shared.py`**

```python
"""Shared constants and helpers for ATLAS backtest scripts."""

import os
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
RATE_LIMIT_SLEEP = 1.1  # seconds between Finnhub calls (free tier: 60/min)


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
```

**Step 5: Write tests for shared helpers**

Create `backtest/tests/test_shared.py`:

```python
import pytest
from backtest.shared import match_edgar_ticker, range_to_base_points, date_to_ts, ts_to_date


def test_match_edgar_ticker_nvidia():
    assert match_edgar_ticker("NVIDIA Corporation") == "NVDA"


def test_match_edgar_ticker_raytheon():
    assert match_edgar_ticker("Raytheon Technologies") == "RTX"


def test_match_edgar_ticker_no_match():
    assert match_edgar_ticker("Random Company XYZ") is None


def test_match_edgar_ticker_case_insensitive():
    assert match_edgar_ticker("PFIZER INC") == "PFE"


def test_range_to_base_points_small():
    assert range_to_base_points("$1,001 - $15,000") == 3


def test_range_to_base_points_medium():
    assert range_to_base_points("$15,001 - $50,000") == 5


def test_range_to_base_points_xl():
    assert range_to_base_points("$1,000,001 - $5,000,000") == 15


def test_range_to_base_points_empty():
    assert range_to_base_points("") == 3


def test_date_roundtrip():
    date = "2026-01-15"
    assert ts_to_date(date_to_ts(date)) == date
```

**Step 6: Run tests — expect PASS**

```bash
cd /Users/henrysalazar/Desktop/Atlas && python -m pytest backtest/tests/test_shared.py -v
```
Expected: 9 tests PASS.

**Step 7: Commit**

```bash
git add backtest/
git commit -m "feat: add backtest/ infrastructure — shared helpers + tests"
```

---

## Task 2: Price History Collector (`collect_prices.py`)

**Files:**
- Create: `backtest/collect_prices.py`
- Create: `backtest/tests/test_collect_prices.py`

**Step 1: Write failing test**

Create `backtest/tests/test_collect_prices.py`:

```python
import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from backtest.collect_prices import extract_tickers, merge_candles, build_price_index


def test_extract_tickers_from_congress():
    """Should extract unique ticker symbols from congress purchase trades."""
    trades = [
        {"Ticker": "AAPL", "Transaction": "Purchase"},
        {"Ticker": "AAPL", "Transaction": "Purchase"},  # duplicate
        {"Ticker": "GOOGL", "Transaction": "Sale"},
        {"Ticker": "MSFT", "Transaction": "Purchase"},
        {"Ticker": "TOOLONGNAME", "Transaction": "Purchase"},  # > 5 chars, skip
    ]
    tickers = extract_tickers(trades, source="congress")
    assert "AAPL" in tickers
    assert "MSFT" in tickers
    assert "GOOGL" in tickers  # include sales too — need their history
    assert "TOOLONGNAME" not in tickers
    assert len(tickers) == len(set(tickers))  # no duplicates


def test_merge_candles_appends_new_dates():
    """New candle data should be merged into existing, no duplicates."""
    existing = {"2026-01-01": {"o": 100, "h": 105, "l": 98, "c": 103, "v": 1000}}
    new_candles = {
        "2026-01-01": {"o": 100, "h": 105, "l": 98, "c": 103, "v": 1000},  # dup
        "2026-01-02": {"o": 103, "h": 107, "l": 101, "c": 106, "v": 1200},
    }
    merged = merge_candles(existing, new_candles)
    assert len(merged) == 2
    assert "2026-01-02" in merged


def test_build_price_index_returns_close_by_date():
    """build_price_index should return {date: close_price} dict."""
    candles = {
        "2026-01-01": {"o": 100, "h": 105, "l": 98, "c": 103, "v": 1000},
        "2026-01-02": {"o": 103, "h": 107, "l": 101, "c": 106, "v": 1200},
    }
    index = build_price_index(candles)
    assert index["2026-01-01"] == 103
    assert index["2026-01-02"] == 106
```

**Step 2: Run test — expect FAIL (module not found)**

```bash
cd /Users/henrysalazar/Desktop/Atlas && python -m pytest backtest/tests/test_collect_prices.py -v
```
Expected: ImportError (collect_prices.py doesn't exist yet).

**Step 3: Create `backtest/collect_prices.py`**

```python
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
```

**Step 4: Run the failing test again — expect PASS**

```bash
cd /Users/henrysalazar/Desktop/Atlas && python -m pytest backtest/tests/test_collect_prices.py -v
```
Expected: 3 tests PASS.

**Step 5: Smoke-test the script (requires FINNHUB_KEY)**

```bash
FINNHUB_KEY=d6dnud9r01qm89pkai30d6dnud9r01qm89pkai3g python backtest/collect_prices.py
```
Expected: Creates `data/price_history/` with JSON files for each ticker. Logs like:
```
SPY: 248 new days, 248 total cached
AAPL: 248 new days, 248 total cached
...
Done. 45 succeeded, 3 failed: ['WFRD', 'ITA', ...]
```
ITA (ETF) and some foreign stocks may fail — that's fine.

**Step 6: Commit**

```bash
git add backtest/collect_prices.py backtest/tests/test_collect_prices.py
git commit -m "feat: add collect_prices.py — Finnhub OHLC cache with incremental updates"
```

---

## Task 3: Event Study Engine (`run_event_study.py`)

**Files:**
- Create: `backtest/run_event_study.py`
- Create: `backtest/tests/test_event_study.py`

**Step 1: Write failing tests**

Create `backtest/tests/test_event_study.py`:

```python
import pytest
from backtest.run_event_study import (
    get_forward_return,
    compute_car,
    compute_member_track_records,
    score_congress_event,
    classify_convergence,
)


# Sample price index: date -> close price
PRICES = {
    "2026-01-10": 100.0,
    "2026-01-11": 100.5,
    "2026-01-12": 101.0,
    "2026-01-13": 102.0,
    "2026-01-14": 103.0,
    "2026-01-15": 104.0,
    "2026-02-09": 110.0,   # ~30 days after 2026-01-10
    "2026-04-10": 115.0,   # ~90 days after 2026-01-10
}

SPY_PRICES = {
    "2026-01-10": 500.0,
    "2026-01-14": 502.5,   # +0.5% in 5 days
    "2026-02-09": 510.0,   # +2% in 30 days
    "2026-04-10": 520.0,   # +4% in 90 days
}


def test_get_forward_return_5d():
    """Should find the closest available price ~5 days after event."""
    ret = get_forward_return(PRICES, "2026-01-10", days=5)
    assert ret is not None
    assert abs(ret - 0.04) < 0.01  # ~4% return (100 → 104)


def test_get_forward_return_missing_date():
    """Should return None if no price data exists within tolerance window."""
    sparse_prices = {"2026-01-01": 100.0}
    ret = get_forward_return(sparse_prices, "2026-01-10", days=5)
    assert ret is None


def test_compute_car():
    """CAR = stock return - benchmark return for same period."""
    stock_ret = 0.05   # +5%
    bench_ret = 0.02   # +2%
    car = compute_car(stock_ret, bench_ret)
    assert abs(car - 0.03) < 1e-6  # 3% alpha


def test_compute_member_track_records():
    """Should compute avg ExcessReturn and win_rate per member."""
    trades = [
        {"Representative": "Alice", "Transaction": "Purchase", "ExcessReturn": 5.0},
        {"Representative": "Alice", "Transaction": "Purchase", "ExcessReturn": -2.0},
        {"Representative": "Bob",   "Transaction": "Purchase", "ExcessReturn": 10.0},
        {"Representative": "Alice", "Transaction": "Sale",     "ExcessReturn": -1.0},  # skip (sale)
    ]
    records = compute_member_track_records(trades)
    assert "Alice" in records
    alice = records["Alice"]
    assert abs(alice["avg_excess"] - 1.5) < 0.01  # mean(5, -2) = 1.5
    assert abs(alice["win_rate"] - 0.5) < 0.01    # 1 of 2 were positive
    assert records["Bob"]["avg_excess"] == 10.0


def test_score_congress_event_basic():
    """Should assign base points based on Range."""
    from backtest.shared import DEFAULT_WEIGHTS
    event = {
        "Ticker": "AAPL",
        "Transaction": "Purchase",
        "TransactionDate": "2026-01-10",
        "Range": "$50,001 - $100,000",
        "Representative": "Alice",
        "ExcessReturn": 5.0,
    }
    score = score_congress_event(event, DEFAULT_WEIGHTS, track_records={}, event_date="2026-01-10")
    assert score > 0
    assert score <= 40  # capped


def test_classify_convergence_both():
    """Tickers with both congress and edgar activity = convergence."""
    congress_tickers = {"AAPL", "MSFT"}
    edgar_tickers = {"AAPL", "GOOGL"}
    assert classify_convergence("AAPL", congress_tickers, edgar_tickers) == "convergence"
    assert classify_convergence("MSFT", congress_tickers, edgar_tickers) == "congress"
    assert classify_convergence("GOOGL", congress_tickers, edgar_tickers) == "edgar"
    assert classify_convergence("TSLA", congress_tickers, edgar_tickers) == "none"
```

**Step 2: Run test — expect FAIL**

```bash
cd /Users/henrysalazar/Desktop/Atlas && python -m pytest backtest/tests/test_event_study.py -v
```
Expected: ImportError.

**Step 3: Create `backtest/run_event_study.py`**

```python
"""
run_event_study.py — compute Cumulative Abnormal Returns (CAR) for all signal events.

Usage:
    python backtest/run_event_study.py

Inputs:
    data/congress_feed.json
    data/edgar_feed.json
    data/price_history/{TICKER}.json

Output:
    data/backtest_results.json — one record per event with CAR at 5d/30d/90d
"""

import sys
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.shared import (
    CONGRESS_FEED, EDGAR_FEED, PRICE_HISTORY_DIR, BACKTEST_RESULTS,
    DEFAULT_WEIGHTS, BENCHMARK, load_json, save_json, match_edgar_ticker,
    range_to_base_points, build_price_index, date_to_ts
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

TOLERANCE_DAYS = 5  # look ±5 days around target date for price


def load_price_index(ticker: str) -> dict:
    """Load {date: close} index for a ticker, or empty dict if no cache."""
    path = PRICE_HISTORY_DIR / f"{ticker}.json"
    if not path.exists():
        return {}
    candles = load_json(path)
    return build_price_index(candles)


def get_forward_return(price_index: dict, event_date: str, days: int) -> float | None:
    """
    Find the return from event_date to event_date+days.
    Searches within ±TOLERANCE_DAYS window for available trading day prices.
    Returns None if either price is unavailable.
    """
    if event_date not in price_index:
        return None

    base_price = price_index[event_date]
    if base_price is None or base_price == 0:
        return None

    event_dt = datetime.strptime(event_date, '%Y-%m-%d')
    target_dt = event_dt + timedelta(days=days)

    # Search within tolerance window for the closest available trading day
    for offset in range(-TOLERANCE_DAYS, TOLERANCE_DAYS + 1):
        candidate = (target_dt + timedelta(days=offset)).strftime('%Y-%m-%d')
        if candidate in price_index:
            forward_price = price_index[candidate]
            if forward_price and forward_price > 0:
                return (forward_price - base_price) / base_price

    return None


def compute_car(stock_return: float, benchmark_return: float) -> float:
    """Cumulative Abnormal Return = stock return - benchmark return."""
    return stock_return - benchmark_return


def compute_member_track_records(trades: list) -> dict:
    """
    Build per-member track record from congressional purchase trades.
    Returns {member_name: {avg_excess, win_rate, n_trades}}
    Uses QuiverQuant's pre-computed ExcessReturn as ground truth.
    """
    member_returns = defaultdict(list)
    for t in trades:
        tx = (t.get('Transaction') or '').lower()
        if 'purchase' not in tx and 'buy' not in tx:
            continue
        member = t.get('Representative') or ''
        excess = t.get('ExcessReturn')
        if member and excess is not None:
            try:
                member_returns[member].append(float(excess))
            except (TypeError, ValueError):
                pass

    records = {}
    all_avgs = []
    for member, rets in member_returns.items():
        avg = sum(rets) / len(rets)
        win_rate = sum(1 for r in rets if r > 0) / len(rets)
        records[member] = {'avg_excess': avg, 'win_rate': win_rate, 'n_trades': len(rets)}
        all_avgs.append(avg)

    # Compute quartile thresholds for track record bonuses
    if all_avgs:
        all_avgs.sort()
        n = len(all_avgs)
        q1_threshold = all_avgs[int(n * 0.75)]  # top 25%
        q2_threshold = all_avgs[int(n * 0.50)]  # top 50%
        for member, rec in records.items():
            avg = rec['avg_excess']
            if avg >= q1_threshold:
                rec['quartile'] = 1
            elif avg >= q2_threshold:
                rec['quartile'] = 2
            elif avg >= 0:
                rec['quartile'] = 3
            else:
                rec['quartile'] = 4

    return records


def score_congress_event(event: dict, weights: dict, track_records: dict, event_date: str) -> float:
    """Score a single congressional purchase event using provided weights."""
    base_pts = range_to_base_points(event.get('Range', ''))

    # Apply temporal decay
    half_life = weights.get('decay_half_life_days', 21)
    now = datetime.now(tz=timezone.utc)
    try:
        event_dt = datetime.strptime(event_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        days_since = (now - event_dt).days
    except (ValueError, TypeError):
        days_since = 0

    decay = 0.5 ** (max(0, days_since) / half_life)
    pts = base_pts * decay

    # Track record bonus
    member = event.get('Representative', '')
    rec = track_records.get(member, {})
    quartile = rec.get('quartile', 4)
    if quartile == 1:
        pts += weights.get('congress_track_record_q1', 0)
    elif quartile == 2:
        pts += weights.get('congress_track_record_q2', 0)

    return min(pts, 40)  # cap


def classify_convergence(ticker: str, congress_tickers: set, edgar_tickers: set) -> str:
    """Return 'convergence', 'congress', 'edgar', or 'none'."""
    has_c = ticker in congress_tickers
    has_e = ticker in edgar_tickers
    if has_c and has_e:
        return 'convergence'
    elif has_c:
        return 'congress'
    elif has_e:
        return 'edgar'
    return 'none'


def run_event_study() -> list:
    """
    Main function: compute CAR for all signal events.
    Returns list of event records.
    """
    congress_data = load_json(CONGRESS_FEED).get('trades', [])
    edgar_data = load_json(EDGAR_FEED).get('filings', [])

    # Build sets of active tickers for convergence detection
    congress_purchase_tickers = {
        t['Ticker'] for t in congress_data
        if 'purchase' in (t.get('Transaction') or '').lower() or 'buy' in (t.get('Transaction') or '').lower()
    }
    edgar_matched_tickers = {
        match_edgar_ticker(f.get('company', ''))
        for f in edgar_data
        if match_edgar_ticker(f.get('company', ''))
    }

    # Compute member track records
    track_records = compute_member_track_records(congress_data)

    # Load SPY prices as benchmark
    spy_index = load_price_index(BENCHMARK)
    if not spy_index:
        log.warning("No SPY price history. CARs will use raw returns (no benchmark adjustment).")

    weights = DEFAULT_WEIGHTS
    events = []
    skipped = 0

    # Process congressional purchase events
    log.info(f"Processing {len(congress_data)} congressional trades...")
    for trade in congress_data:
        tx = (trade.get('Transaction') or '').lower()
        if 'purchase' not in tx and 'buy' not in tx:
            continue

        ticker = (trade.get('Ticker') or '').strip().upper()
        if not ticker or len(ticker) > 5:
            continue

        event_date = trade.get('TransactionDate') or trade.get('ReportDate')
        if not event_date:
            continue

        price_index = load_price_index(ticker)
        if not price_index:
            skipped += 1
            continue

        convergence = classify_convergence(ticker, congress_purchase_tickers, edgar_matched_tickers)

        event = {
            "ticker": ticker,
            "event_date": event_date,
            "event_type": "congress",
            "member": trade.get('Representative', ''),
            "range": trade.get('Range', ''),
            "party": trade.get('Party', ''),
            "house": trade.get('House', ''),
            "excess_return_qq": trade.get('ExcessReturn'),  # ground truth from QQ
            "price_change_qq": trade.get('PriceChange'),
            "spy_change_qq": trade.get('SPYChange'),
            "convergence": convergence,
        }

        # Compute forward returns
        for window in [5, 30, 90]:
            stock_ret = get_forward_return(price_index, event_date, window)
            spy_ret = get_forward_return(spy_index, event_date, window) if spy_index else None
            if stock_ret is not None and spy_ret is not None:
                event[f"car_{window}d"] = round(compute_car(stock_ret, spy_ret), 6)
                event[f"stock_ret_{window}d"] = round(stock_ret, 6)
            else:
                event[f"car_{window}d"] = None
                event[f"stock_ret_{window}d"] = None

        # Score this event
        event["score"] = round(score_congress_event(trade, weights, track_records, event_date), 2)
        member_rec = track_records.get(trade.get('Representative', ''), {})
        event["member_quartile"] = member_rec.get('quartile', 4)
        event["member_avg_excess"] = member_rec.get('avg_excess')

        events.append(event)

    # Process EDGAR filing events
    log.info(f"Processing {len(edgar_data)} EDGAR filings...")
    for filing in edgar_data:
        ticker = match_edgar_ticker(filing.get('company', ''))
        if not ticker:
            continue

        event_date = filing.get('date')
        if not event_date:
            continue

        price_index = load_price_index(ticker)
        if not price_index:
            skipped += 1
            continue

        convergence = classify_convergence(ticker, congress_purchase_tickers, edgar_matched_tickers)

        event = {
            "ticker": ticker,
            "event_date": event_date,
            "event_type": "edgar",
            "company": filing.get('company', ''),
            "insider": filing.get('insider', ''),
            "convergence": convergence,
        }

        for window in [5, 30, 90]:
            stock_ret = get_forward_return(price_index, event_date, window)
            spy_ret = get_forward_return(spy_index, event_date, window) if spy_index else None
            if stock_ret is not None and spy_ret is not None:
                event[f"car_{window}d"] = round(compute_car(stock_ret, spy_ret), 6)
            else:
                event[f"car_{window}d"] = None

        events.append(event)

    log.info(f"Event study complete: {len(events)} events processed, {skipped} skipped (no price data)")
    return events


def main():
    events = run_event_study()
    save_json(BACKTEST_RESULTS, {
        "generated": datetime.now(tz=timezone.utc).isoformat(),
        "n_events": len(events),
        "events": events,
    })
    log.info(f"Saved {len(events)} events to {BACKTEST_RESULTS}")


if __name__ == '__main__':
    main()
```

**Step 4: Run the tests — expect PASS**

```bash
cd /Users/henrysalazar/Desktop/Atlas && python -m pytest backtest/tests/test_event_study.py -v
```
Expected: 7 tests PASS.

**Step 5: Smoke-test the script**

```bash
python backtest/run_event_study.py
```
Expected output:
```
Processing 150 congressional trades...
Processing 200 EDGAR filings...
Event study complete: ~130 events processed, ~20 skipped
Saved 130 events to data/backtest_results.json
```
Check output: `cat data/backtest_results.json | python -m json.tool | head -50`

**Step 6: Commit**

```bash
git add backtest/run_event_study.py backtest/tests/test_event_study.py
git commit -m "feat: add run_event_study.py — CAR computation + member track records"
```

---

## Task 4: Weight Optimizer (`optimize_weights.py`)

**Files:**
- Create: `backtest/optimize_weights.py`
- Create: `backtest/tests/test_optimizer.py`

**Step 1: Write failing tests**

Create `backtest/tests/test_optimizer.py`:

```python
import pytest
import copy
from backtest.optimize_weights import (
    score_event_with_weights,
    evaluate_weights,
    find_optimal_threshold,
    grid_search,
)
from backtest.shared import DEFAULT_WEIGHTS

# Sample events for testing
EVENTS = [
    # High-scoring congress event that performed well
    {"event_type": "congress", "range": "$1,000,001 - $5,000,000",
     "car_30d": 0.08, "convergence": "congress", "member_quartile": 1},
    # Low-scoring congress event that underperformed
    {"event_type": "congress", "range": "$1,001 - $15,000",
     "car_30d": -0.02, "convergence": "congress", "member_quartile": 4},
    # Convergence event that performed well
    {"event_type": "congress", "range": "$50,001 - $100,000",
     "car_30d": 0.05, "convergence": "convergence", "member_quartile": 2},
    # EDGAR event with positive return
    {"event_type": "edgar", "car_30d": 0.03, "convergence": "edgar"},
]


def test_score_event_with_weights_congress():
    """Congress event should produce a positive score with default weights."""
    event = EVENTS[0]  # $1M+ purchase
    score = score_event_with_weights(event, DEFAULT_WEIGHTS)
    assert score > 0
    assert score <= 115  # theoretical max


def test_score_event_with_weights_convergence_bonus():
    """Convergence events should score higher than single-source events."""
    single = copy.deepcopy(EVENTS[1])
    single['convergence'] = 'congress'
    conv = copy.deepcopy(EVENTS[1])
    conv['convergence'] = 'convergence'

    score_single = score_event_with_weights(single, DEFAULT_WEIGHTS)
    score_conv = score_event_with_weights(conv, DEFAULT_WEIGHTS)
    assert score_conv > score_single


def test_evaluate_weights_returns_metrics():
    """evaluate_weights should return avg_car, hit_rate, n_events for a threshold."""
    events_with_scores = [
        {"car_30d": 0.05, "score": 70},
        {"car_30d": 0.03, "score": 80},
        {"car_30d": -0.01, "score": 30},
        {"car_30d": None,  "score": 75},   # missing CAR — should be excluded
    ]
    metrics = evaluate_weights(events_with_scores, threshold=65)
    assert metrics["n_events"] == 2  # only 70 and 80 qualify, 30 excluded, None excluded
    assert abs(metrics["avg_car_30d"] - 0.04) < 0.001
    assert abs(metrics["hit_rate"] - 1.0) < 0.001  # both positive


def test_find_optimal_threshold():
    """Should return the threshold where avg_car is best."""
    # Events where high threshold gives better avg return
    events = [
        {"car_30d": 0.10, "score": 90},
        {"car_30d": 0.08, "score": 80},
        {"car_30d": -0.05, "score": 40},
        {"car_30d": -0.03, "score": 50},
    ]
    best_threshold, _ = find_optimal_threshold(events, candidates=[40, 65, 75, 85])
    # At threshold 80, avg_car = mean(0.10) = 0.10 — best
    assert best_threshold >= 80


def test_grid_search_improves_over_default():
    """Grid search should find weights that give >= default weight performance."""
    results = grid_search(EVENTS, n_candidates=3)
    assert "optimal_weights" in results
    assert "stats" in results
    assert results["stats"]["n_events"] >= 0
```

**Step 2: Run test — expect FAIL**

```bash
cd /Users/henrysalazar/Desktop/Atlas && python -m pytest backtest/tests/test_optimizer.py -v
```
Expected: ImportError.

**Step 3: Create `backtest/optimize_weights.py`**

```python
"""
optimize_weights.py — grid search for optimal ATLAS scoring weights.

Usage:
    python backtest/optimize_weights.py

Inputs:
    data/backtest_results.json

Output:
    data/optimal_weights.json  — best weights found
    data/backtest_summary.json — human-readable performance report
"""

import sys
import copy
import logging
import itertools
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.shared import (
    BACKTEST_RESULTS, OPTIMAL_WEIGHTS, BACKTEST_SUMMARY,
    DEFAULT_WEIGHTS, load_json, save_json, range_to_base_points
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


def score_event_with_weights(event: dict, weights: dict) -> float:
    """Compute convergence score for an event record using provided weights."""
    etype = event.get('event_type', 'congress')
    convergence = event.get('convergence', 'none')

    congress_score = 0.0
    edgar_score = 0.0

    if etype == 'congress':
        tiers = weights.get('congress_tiers', DEFAULT_WEIGHTS['congress_tiers'])
        range_str = event.get('range', '')
        r = range_str or ''
        if '$1,000,001' in r:   base = tiers.get('xl', 15)
        elif '$500,001' in r:   base = tiers.get('significant', 10)
        elif '$250,001' in r:   base = tiers.get('significant', 10)
        elif '$100,001' in r:   base = tiers.get('major', 8)
        elif '$50,001' in r:    base = tiers.get('large', 6)
        elif '$15,001' in r:    base = tiers.get('medium', 5)
        else:                    base = tiers.get('small', 3)

        # Track record bonus
        quartile = event.get('member_quartile', 4)
        if quartile == 1:
            base += weights.get('congress_track_record_q1', 0)
        elif quartile == 2:
            base += weights.get('congress_track_record_q2', 0)

        congress_score = min(base, 40)

    elif etype == 'edgar':
        # For optimizer: treat each EDGAR event as 1 filing
        edgar_score = min(weights.get('edgar_base_per_filing', 6), 40)

    # Convergence boost
    boost = 0
    if convergence == 'convergence':
        boost = weights.get('convergence_boost', 20)

    return congress_score + edgar_score + boost


def evaluate_weights(events_with_scores: list, threshold: float) -> dict:
    """
    Evaluate performance of a weight configuration.
    Returns {avg_car_30d, hit_rate, n_events} for events above threshold.
    """
    above = [e for e in events_with_scores if e.get('score', 0) >= threshold and e.get('car_30d') is not None]
    if not above:
        return {"avg_car_30d": 0.0, "hit_rate": 0.0, "n_events": 0}

    cars = [e['car_30d'] for e in above]
    avg_car = float(np.mean(cars))
    hit_rate = sum(1 for c in cars if c > 0) / len(cars)
    return {
        "avg_car_30d": round(avg_car, 6),
        "hit_rate": round(hit_rate, 4),
        "n_events": len(above),
    }


def find_optimal_threshold(events_with_scores: list, candidates=None) -> tuple:
    """
    Find the threshold that maximizes avg_car_30d * hit_rate for above-threshold events.
    Returns (best_threshold, metrics_dict).
    """
    if candidates is None:
        candidates = [40, 50, 55, 60, 65, 70, 75, 80, 85, 90]

    best_score = -999
    best_threshold = 65
    best_metrics = {}

    for threshold in candidates:
        metrics = evaluate_weights(events_with_scores, threshold)
        if metrics['n_events'] < 5:
            continue  # not enough data at this threshold
        combined = metrics['avg_car_30d'] * metrics['hit_rate']
        if combined > best_score:
            best_score = combined
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def grid_search(events: list, n_candidates: int = 4) -> dict:
    """
    Grid search over key weight parameters.
    n_candidates: number of values per parameter to test (use 3-4, full search is 4^N combinations).
    Returns {optimal_weights, stats, threshold}.
    """
    # Parameter search spaces — fewer candidates = faster run
    search_space = {
        'congress_xl':      [12, 15, 18, 20][:n_candidates],
        'congress_cluster': [10, 15, 18, 20][:n_candidates],
        'congress_q1':      [0, 5, 8, 10][:n_candidates],
        'convergence':      [15, 20, 25, 30][:n_candidates],
        'decay_half_life':  [14, 21, 30, 45][:n_candidates],
    }

    best_score = -999
    best_weights = copy.deepcopy(DEFAULT_WEIGHTS)
    best_events_scored = []

    param_names = list(search_space.keys())
    param_values = list(search_space.values())
    combinations = list(itertools.product(*param_values))
    log.info(f"Grid search: testing {len(combinations)} weight combinations...")

    for i, combo in enumerate(combinations):
        candidate = copy.deepcopy(DEFAULT_WEIGHTS)
        params = dict(zip(param_names, combo))

        # Apply this combination
        candidate['congress_tiers']['xl'] = params['congress_xl']
        candidate['congress_cluster_bonus'] = params['congress_cluster']
        candidate['congress_track_record_q1'] = params['congress_q1']
        candidate['convergence_boost'] = params['convergence']
        candidate['decay_half_life_days'] = params['decay_half_life']

        # Score all events
        scored_events = []
        for event in events:
            e = dict(event)
            e['score'] = score_event_with_weights(e, candidate)
            scored_events.append(e)

        # Find best threshold for this weight combo
        threshold, metrics = find_optimal_threshold(scored_events)
        if metrics.get('n_events', 0) < 5:
            continue

        combined = metrics['avg_car_30d'] * metrics['hit_rate']
        if combined > best_score:
            best_score = combined
            best_weights = copy.deepcopy(candidate)
            best_weights['_optimal_threshold'] = threshold
            best_events_scored = scored_events

        if i % 50 == 0:
            log.info(f"  {i}/{len(combinations)} done, best so far: {round(best_score, 4)}")

    # Final evaluation at optimal threshold
    final_threshold = best_weights.get('_optimal_threshold', 65)
    best_weights.pop('_optimal_threshold', None)
    final_metrics = evaluate_weights(best_events_scored, final_threshold)

    return {
        "optimal_weights": best_weights,
        "optimal_threshold": final_threshold,
        "stats": final_metrics,
    }


def main():
    log.info("Loading backtest results...")
    try:
        results = load_json(BACKTEST_RESULTS)
    except FileNotFoundError:
        log.error(f"backtest_results.json not found. Run run_event_study.py first.")
        sys.exit(1)

    events = results.get('events', [])
    if len(events) < 10:
        log.warning(f"Only {len(events)} events — too few to optimize. Using defaults.")
        save_json(OPTIMAL_WEIGHTS, {
            **DEFAULT_WEIGHTS,
            "generated": datetime.now(tz=timezone.utc).isoformat(),
            "optimal_threshold": 65,
            "stats": {"note": "insufficient_data"},
        })
        return

    log.info(f"Optimizing weights over {len(events)} events...")
    result = grid_search(events, n_candidates=4)

    # Build output
    output = {
        **result["optimal_weights"],
        "generated": datetime.now(tz=timezone.utc).isoformat(),
        "n_events_total": len(events),
        "optimal_threshold": result["optimal_threshold"],
        "stats": result["stats"],
    }
    save_json(OPTIMAL_WEIGHTS, output)

    # Summary for humans
    stats = result["stats"]
    summary = {
        "generated": output["generated"],
        "n_events_backtested": len(events),
        "optimal_threshold": result["optimal_threshold"],
        "avg_car_30d_pct": round(stats.get("avg_car_30d", 0) * 100, 2),
        "hit_rate_pct": round(stats.get("hit_rate", 0) * 100, 1),
        "n_events_above_threshold": stats.get("n_events", 0),
        "key_changes_from_default": {
            k: output.get(k) for k in ["convergence_boost", "decay_half_life_days",
                                        "congress_track_record_q1", "optimal_threshold"]
        },
    }
    save_json(BACKTEST_SUMMARY, summary)

    log.info(f"\n=== BACKTEST SUMMARY ===")
    log.info(f"Optimal threshold:  {result['optimal_threshold']}")
    log.info(f"Events above threshold: {stats.get('n_events', 0)}")
    log.info(f"Avg 30d CAR:        {round(stats.get('avg_car_30d', 0)*100, 2)}%")
    log.info(f"Hit rate:           {round(stats.get('hit_rate', 0)*100, 1)}%")
    log.info(f"Weights saved to:   {OPTIMAL_WEIGHTS}")


if __name__ == '__main__':
    main()
```

**Step 4: Run tests — expect PASS**

```bash
cd /Users/henrysalazar/Desktop/Atlas && python -m pytest backtest/tests/test_optimizer.py -v
```
Expected: 5 tests PASS.

**Step 5: Run the full pipeline locally**

```bash
FINNHUB_KEY=d6dnud9r01qm89pkai30d6dnud9r01qm89pkai3g python backtest/run_event_study.py
python backtest/optimize_weights.py
```
Expected: `data/optimal_weights.json` and `data/backtest_summary.json` created. Check:
```bash
python -c "import json; d=json.load(open('data/backtest_summary.json')); print(json.dumps(d, indent=2))"
```
Should print hit rate, avg CAR, optimal threshold.

**Step 6: Run full test suite**

```bash
cd /Users/henrysalazar/Desktop/Atlas && python -m pytest backtest/tests/ -v
```
Expected: All 17 tests PASS.

**Step 7: Commit**

```bash
git add backtest/optimize_weights.py backtest/tests/test_optimizer.py
git commit -m "feat: add optimize_weights.py — grid search for optimal scoring weights"
```

---

## Task 5: GitHub Actions Workflow

**Files:**
- Create: `.github/workflows/backtest.yml`

**Step 1: Create the workflow**

```yaml
name: Weekly Backtest & Weight Optimization

# Runs every Sunday at 02:00 UTC (after weekly data refresh on weekdays)
# Can also be triggered manually from the Actions tab
on:
  schedule:
    - cron: '0 2 * * 0'  # Sundays 02:00 UTC
  workflow_dispatch:      # manual trigger

jobs:
  backtest:
    runs-on: ubuntu-latest
    permissions:
      contents: write  # needed to commit updated weight/results files

    steps:
      - name: Checkout repo
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install finnhub-python pandas numpy

      - name: Collect price history
        env:
          FINNHUB_KEY: ${{ secrets.FINNHUB_KEY }}
        run: python backtest/collect_prices.py

      - name: Run event study
        run: python backtest/run_event_study.py

      - name: Optimize weights
        run: python backtest/optimize_weights.py

      - name: Commit updated results
        run: |
          git config --global user.name  'ATLAS Backtest Bot'
          git config --global user.email 'bot@atlasiq.io'
          git add data/optimal_weights.json data/backtest_results.json data/backtest_summary.json data/price_history/
          git diff --staged --quiet || git commit -m "chore: weekly backtest $(date -u '+%Y-%m-%d') [skip ci]"
          git push
```

**Step 2: Verify `FINNHUB_KEY` secret exists**

Go to: https://github.com/Hsalazar023/atlas-intelligence/settings/secrets/actions

Confirm `FINNHUB_KEY` is listed (it should be — it's used by `fetch_data.py` already).

**Step 3: Push the workflow**

```bash
git add .github/workflows/backtest.yml
git commit -m "ci: add weekly backtest workflow — runs every Sunday 02:00 UTC"
git push
```

**Step 4: Verify in GitHub Actions tab**

Go to: https://github.com/Hsalazar023/atlas-intelligence/actions

Should see "Weekly Backtest & Weight Optimization" in the workflow list. Click "Run workflow" to trigger manually. Watch it run. Expected duration: 5-10 min on first run (price history collection), ~30s on subsequent runs.

---

## Task 6: Frontend Dynamic Weights Integration

**Files:**
- Modify: `atlas-intelligence.html`

**Step 1: Add `SCORE_WEIGHTS` loader at top of JS section**

Find the `window.addEventListener('load', ...)` initialization block (around line 1975). Add a `SCORE_THRESHOLD` and `SCORE_WEIGHTS` global before it, and a loader function:

```javascript
// ═══════════════════════════════════════════════════════
// DYNAMIC SCORING WEIGHTS — loaded from data/optimal_weights.json
// Falls back to hardcoded defaults if file not found
// ═══════════════════════════════════════════════════════
var SCORE_WEIGHTS = null;
var SCORE_THRESHOLD = 65;  // default until backtest runs

function loadOptimalWeights(){
  fetch('data/optimal_weights.json')
    .then(function(r){ return r.json(); })
    .then(function(w){
      SCORE_WEIGHTS=w;
      SCORE_THRESHOLD=w.optimal_threshold||65;
      var threshEl=document.getElementById('ideas-threshold-display');
      if(threshEl) threshEl.textContent='\u2265 '+SCORE_THRESHOLD;
      // Update backtest stats footer if present
      var statsEl=document.getElementById('backtest-stats-bar');
      if(statsEl&&w.stats){
        var hr=w.stats.hit_rate?Math.round(w.stats.hit_rate*100)+'%':'—';
        var car=w.stats.avg_car_30d?'+'+Math.round(w.stats.avg_car_30d*100)+'%':'—';
        var gen=w.generated?w.generated.slice(0,10):'—';
        statsEl.textContent='Last backtest: '+gen+' · '+w.n_events_total+' events · Hit rate: '+hr+' · Avg 30d alpha: '+car;
      }
      refreshConvergenceDisplays();
    })
    .catch(function(){
      // File not found (first run before backtest) — use hardcoded defaults silently
    });
}
```

**Step 2: Wire the loader into the `load` event**

Find the `window.addEventListener('load', ...)` block. At the beginning of its body, add:
```javascript
  loadOptimalWeights();
```

**Step 3: Add backtest stats bar to Ideas page HTML**

Find the Ideas page header section in HTML (near `id="page-ideas"`). Add below the acad-box:
```html
<div id="backtest-stats-bar" style="font-size:10px;color:var(--muted);padding:6px 12px;background:var(--card-bg);border-radius:4px;margin-bottom:16px;border:1px solid var(--border)">
  Backtest engine loading...
</div>
```

**Step 4: Apply dynamic decay in `scoreCongressTicker()` if weights loaded**

In `scoreCongressTicker()`, find the decay line (added in Task 0):
```javascript
    var decayFactor=Math.pow(0.5,Math.max(0,daysSince)/21);
```
Replace with:
```javascript
    var halfLife=(SCORE_WEIGHTS&&SCORE_WEIGHTS.decay_half_life_days)||21;
    var decayFactor=Math.pow(0.5,Math.max(0,daysSince)/halfLife);
```

**Step 5: Verify**

```bash
python3 -m http.server 8080
```
Open http://localhost:8080. Go to Trade Ideas tab.
- If `data/optimal_weights.json` exists: backtest stats bar should show last run date, hit rate, avg alpha
- Watchlist and Trade Ideas sections should be visible based on current scores
- Open browser console — no errors

**Step 6: Commit and push**

```bash
git add atlas-intelligence.html
git commit -m "feat: wire dynamic weights from optimal_weights.json into scoring engine"
git push
```

---

## Verification Checklist

Run this after all tasks are complete:

```bash
# 1. All tests pass
cd /Users/henrysalazar/Desktop/Atlas && python -m pytest backtest/tests/ -v

# 2. Full pipeline runs end to end
FINNHUB_KEY=d6dnud9r01qm89pkai30d6dnud9r01qm89pkai3g python backtest/collect_prices.py
python backtest/run_event_study.py
python backtest/optimize_weights.py

# 3. Output files exist with expected content
ls data/price_history/ | head -10
python -c "import json; d=json.load(open('data/optimal_weights.json')); print('threshold:', d['optimal_threshold']); print('stats:', d['stats'])"

# 4. Frontend loads weights
python3 -m http.server 8080
# Open http://localhost:8080, go to Trade Ideas, verify:
# - Watchlist section shows tickers scoring 40+
# - Backtest stats bar populated (or "loading" if no optimal_weights.json yet)
# - No console errors

# 5. GitHub Actions workflow file exists and is valid YAML
cat .github/workflows/backtest.yml | python -c "import sys,yaml; yaml.safe_load(sys.stdin); print('YAML valid')"
# Note: install pyyaml if needed: pip install pyyaml
```

---

## File Index

| File | Status | Purpose |
|---|---|---|
| `atlas-intelligence.html` | Modified | Task 0 + Task 6: Watchlist tier, signal decay, dynamic weights |
| `backtest/__init__.py` | Created | Python package marker |
| `backtest/shared.py` | Created | Constants, helpers, TICKER_KEYWORDS |
| `backtest/collect_prices.py` | Created | Finnhub OHLC fetcher |
| `backtest/run_event_study.py` | Created | CAR computation, track records |
| `backtest/optimize_weights.py` | Created | Grid search weight optimizer |
| `backtest/requirements.txt` | Created | Python dependencies |
| `backtest/tests/__init__.py` | Created | Test package marker |
| `backtest/tests/test_shared.py` | Created | Unit tests for helpers |
| `backtest/tests/test_collect_prices.py` | Created | Unit tests for price collector |
| `backtest/tests/test_event_study.py` | Created | Unit tests for event study |
| `backtest/tests/test_optimizer.py` | Created | Unit tests for optimizer |
| `.github/workflows/backtest.yml` | Created | Weekly automation |
| `data/price_history/` | Created by script | OHLC cache |
| `data/backtest_results.json` | Created by script | All event CAR results |
| `data/optimal_weights.json` | Created by script | Dynamic weights for frontend |
| `data/backtest_summary.json` | Created by script | Human-readable stats |
