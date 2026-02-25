# ATLAS — Scoring Backtest & Self-Optimizing Engine Design

**Date:** 2026-02-24
**Status:** Approved
**Scope:** CAR-based event study engine + weight optimizer + frontend dynamic weights

---

## Problem

The current scoring system has three failures:

1. **Threshold too high for current data.** Max congress score = 40, max EDGAR = 40, convergence boost = +20, theoretical max = 115. To reach 85 you need strong signals from *both* sources simultaneously. In practice, tickers are scoring 50–65 (one-source signals). Trade ideas never appear.

2. **Weights are guesses, not data-driven.** Scoring tiers ($15k=5pts, cluster=15pts, etc.) were assigned by intuition. No validation against actual measured returns. We don't know if a $1M congressional buy actually predicts better returns than a $50k buy.

3. **No signal decay.** A 30-day-old trade scores the same as yesterday's trade. Signal strength decays with time.

---

## Academic Foundation

Key research findings that inform this design:

- **Congressional trading (post-STOCK Act, 2012+):** Average alpha is ~-17 to -28 bps/month — baseline trades have near-zero edge. But **leaders and committee chairs outperform by 40-50% annually** in high-conviction windows. Signal is in *selective* trades, not all trades.
- **Insider buying:** CFO/Director outperforms CEO (21.5% vs 19.3% annualized). Cluster buying (3+ insiders, 72h) generates 2.1% abnormal return vs 1.2% for singles. No 10b5-1 plan is the single strongest quality filter.
- **Track record matters:** Members of congress with historically high `ExcessReturn` should score more heavily than first-timers or chronic underperformers.
- **Best holding window:** 30 days captures the bulk of predictive signal. After 90 days, alpha decays significantly.

---

## Architecture

```
GitHub Actions (Sundays 02:00 UTC, after weekly data refresh)
    │
    ├─ backtest/collect_prices.py
    │   ├─ Inputs: data/congress_feed.json, data/edgar_feed.json
    │   ├─ Fetches: Finnhub /stock/candle for all tickers + SPY (365d lookback)
    │   ├─ Rate limit: 1 call/sec (Finnhub free tier: 60/min)
    │   └─ Output: data/price_history/{TICKER}.json (OHLC, incremental append)
    │
    ├─ backtest/run_event_study.py
    │   ├─ Inputs: congress_feed.json + edgar_feed.json + price_history/
    │   ├─ Events: congressional purchases + EDGAR filings matched to tickers
    │   ├─ Computes: CAR at 5d/30d/90d (stock return − SPY return)
    │   ├─ Also: member track record (avg ExcessReturn per Representative)
    │   └─ Output: data/backtest_results.json (one record per event)
    │
    ├─ backtest/optimize_weights.py
    │   ├─ Inputs: backtest_results.json
    │   ├─ Method: Grid search → maximize avg CAR at 30d for events where score ≥ threshold
    │   ├─ Also: finds empirically optimal threshold (separates + vs − CAR events)
    │   └─ Output: data/optimal_weights.json + data/backtest_summary.json
    │
    └─ git commit + push → Vercel auto-deploys

atlas-intelligence.html (frontend)
    ├─ Loads optimal_weights.json on startup (fetch, fallback to hardcoded)
    ├─ Scoring functions use dynamic weights
    ├─ Watchlist tier: score 50–84 ("Monitoring" cards)
    ├─ Trade Ideas tier: score ≥ empirical threshold from backtest
    └─ Backtest stats footer: last run, hit rate, avg 30d alpha
```

---

## Component Specifications

### backtest/collect_prices.py

```python
# Pseudocode
TICKERS = extract_all_tickers(congress_feed, edgar_feed)
for ticker in TICKERS + ['SPY']:
    existing = load_cache(f'data/price_history/{ticker}.json')
    missing_dates = compute_missing_dates(existing, lookback_days=365)
    if missing_dates:
        candles = finnhub.stock_candles(ticker, 'D', from_ts, to_ts)
        save_cache(ticker, merge(existing, candles))
    sleep(1)  # rate limit
```

**Failure handling:** Skip tickers with no data (ETFs, foreign stocks Finnhub doesn't cover). Log to `data/backtest_log.txt`.

---

### backtest/run_event_study.py

**Congressional events:**
```
event_date = TransactionDate
ticker     = Ticker
event_type = "congress"
range_str  = Range
party      = Party
member     = Representative
excess_ret = ExcessReturn (already computed by QuiverQuant — ground truth)
car_30d    = computed from price_history if available, else use ExcessReturn as proxy
```

**EDGAR events:**
```
event_date = filing date
ticker     = matched via TICKER_KEYWORDS or ticker-string fallback
event_type = "edgar"
insider    = insider field
car_30d    = computed from price_history
```

**Member track record (per Representative):**
```
all_purchases = filter congData where Transaction=Purchase, group by Representative
member_avg_excess_return = mean(ExcessReturn) for that member
member_win_rate = pct where ExcessReturn > 0
track_record_quartile = quartile rank within all active members
```

**Output schema per event:**
```json
{
  "ticker": "AAPL",
  "event_date": "2026-01-15",
  "event_type": "congress",
  "range": "$50,001 - $100,000",
  "member": "Nancy Pelosi",
  "track_record_quartile": 1,
  "car_5d": 0.023,
  "car_30d": 0.041,
  "car_90d": 0.038,
  "score_current": 28,
  "score_breakdown": {"congress": 28, "edgar": 0, "boost": 0}
}
```

---

### backtest/optimize_weights.py

**Parameters to optimize (grid search):**

| Parameter | Current Value | Search Space |
|---|---|---|
| Congress: small ($1k–$15k) | 3 pts | [1, 2, 3, 4, 5] |
| Congress: medium ($15k–$50k) | 5 pts | [3, 4, 5, 6, 8] |
| Congress: large ($50k–$100k) | 6 pts | [4, 5, 6, 8, 10] |
| Congress: major ($100k–$250k) | 8 pts | [6, 7, 8, 10, 12] |
| Congress: significant ($250k–$1M) | 10 pts | [8, 10, 12, 15] |
| Congress: large ($1M+) | 15 pts | [10, 12, 15, 18, 20] |
| Congress: cluster bonus (3+ members) | 15 pts | [10, 12, 15, 18, 20] |
| Congress: track record Q1 bonus | 0 pts | [3, 5, 8, 10] |
| Congress: track record Q2 bonus | 0 pts | [1, 2, 3, 5] |
| EDGAR: base per filing | 6 pts | [3, 4, 6, 8, 10] |
| EDGAR: cluster 2 bonus | 10 pts | [5, 8, 10, 12, 15] |
| EDGAR: cluster 3+ bonus | 15 pts | [10, 12, 15, 18, 20] |
| Convergence boost | 20 pts | [10, 15, 20, 25, 30] |
| Signal decay half-life (days) | None | [7, 14, 21, 30, 45] |

**Optimization metric:**
- Primary: `avg(car_30d)` for events where `score ≥ threshold` → **maximize expected alpha for displayed signals**
- Secondary: `hit_rate` (pct events with car_30d > 0) → **minimize false positives**
- Combined: `avg(car_30d) × hit_rate` (penalizes high returns on a tiny subset)

**Threshold optimization:**
- For each candidate threshold (40, 50, 60, 65, 70, 75, 80, 85, 90):
  - Compute avg CAR and hit rate for events above that threshold
  - Report the empirically best threshold
  - Also report minimum viable threshold (where avg CAR first turns positive)

**Output (`data/optimal_weights.json`):**
```json
{
  "generated": "2026-02-24T02:00:00Z",
  "n_events": 847,
  "optimal_threshold": 65,
  "congress": {
    "small": 2, "medium": 4, "large": 6, "major": 8,
    "significant": 12, "xl": 15, "cluster_bonus": 18,
    "track_record_q1": 8, "track_record_q2": 3
  },
  "edgar": { "base": 6, "cluster_2": 10, "cluster_3": 15 },
  "convergence_boost": 20,
  "decay_half_life_days": 21,
  "stats": {
    "avg_car_30d_above_threshold": 0.041,
    "hit_rate_above_threshold": 0.61,
    "avg_car_30d_below_threshold": -0.008,
    "n_events_above_threshold": 124
  }
}
```

---

### Frontend Changes (atlas-intelligence.html)

**1. Load dynamic weights on startup:**
```javascript
var SCORE_WEIGHTS = null; // loaded from optimal_weights.json
var SCORE_THRESHOLD = 85; // fallback if weights not loaded

fetch('data/optimal_weights.json')
  .then(r => r.json())
  .then(w => {
    SCORE_WEIGHTS = w;
    SCORE_THRESHOLD = w.optimal_threshold || 85;
    refreshConvergenceDisplays();
  })
  .catch(() => { /* use hardcoded defaults */ });
```

**2. Signal decay in scoreCongressTicker():**
```javascript
var daysSince = (Date.now() - new Date(trade.TransactionDate)) / 86400000;
var halfLife = SCORE_WEIGHTS?.decay_half_life_days || 21;
var decayFactor = Math.pow(0.5, daysSince / halfLife);
pts += basePoints * decayFactor;
```

**3. Watchlist tier + Trade Ideas tier in renderSignalIdeas():**
- **Trade Ideas (score ≥ SCORE_THRESHOLD):** Red/amber score badge, "Engine-generated" label
- **Watchlist (score 50 – SCORE_THRESHOLD-1):** Gray badge, "Monitoring" label
- Both show ticker, signal count, current score, breakdown

**4. Backtest stats footer on Ideas page:**
```
Last backtest: Feb 23, 2026 · 847 events · Hit rate: 61% · Avg 30d alpha: +4.1%
```

---

## GitHub Actions Workflow (.github/workflows/backtest.yml)

```yaml
name: Weekly Backtest & Weight Optimization
on:
  schedule:
    - cron: '0 2 * * 0'  # Sundays 02:00 UTC
  workflow_dispatch:     # manual trigger

jobs:
  backtest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install finnhub-python pandas numpy
      - run: python backtest/collect_prices.py
        env: { FINNHUB_KEY: ${{ secrets.FINNHUB_KEY }} }
      - run: python backtest/run_event_study.py
      - run: python backtest/optimize_weights.py
      - run: |
          git config user.name "ATLAS Bot"
          git config user.email "bot@atlas-intelligence.app"
          git add data/optimal_weights.json data/backtest_results.json data/backtest_summary.json
          git diff --staged --quiet || git commit -m "chore: weekly backtest [skip ci]"
          git push
```

---

## Immediate Fixes (Before Backtest Engine Exists)

1. **Add signal decay** — halve the score of trades older than 21 days (hardcoded until backtest tunes the half-life)
2. **Add Watchlist tier** — show tickers scoring 50–84 as "Monitoring" cards in the Ideas tab
3. **Lower Trade Ideas threshold to 65** initially — empirically where signals start showing positive returns based on academic literature; will be tuned by backtest

---

## What Stays Out of Scope

- EDGAR role enrichment (parsing full Form 4 XML) — Phase 2, tracked separately
- Institutional 13F backtesting — needs full 13F parser first
- ML-based weight optimization — Phase 3, built on top of this event study foundation
- Short signal scoring — only long (purchase) signals backtested initially

---

## Verification

After implementation:
1. Run `python backtest/collect_prices.py` locally — should create `data/price_history/` with OHLC files
2. Run `python backtest/run_event_study.py` — should produce `data/backtest_results.json` with 100+ events
3. Run `python backtest/optimize_weights.py` — should produce `data/optimal_weights.json`
4. Load frontend locally — Dynamic weights loaded, "Watchlist" section shows tickers scoring 50-84
5. Check GitHub Actions tab after push — workflow file appears, can be manually triggered
6. After first automated run (Sunday): `optimal_weights.json` committed automatically, frontend reflects updated weights on next load
