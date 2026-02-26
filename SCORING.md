# ATLAS — Signal Scoring Engine
*Reference document. Version 0.2 — Feb 25, 2026*

This is the canonical spec for how ATLAS scores signals and detects convergence.
Update this file whenever scoring logic changes in the codebase.

---

## Hub 1 — Congressional Score

Derived from QuiverQuant live endpoint. Only **purchase** transactions count.
Sales/exchanges score 0 for convergence purposes (they are displayed but not boosted).

| Factor | Points |
|---|---|
| Purchase filed (base) | +3 |
| Trade size $15K–$50K | +5 |
| Trade size $50K–$100K | +6 |
| Trade size $100K–$250K | +8 |
| Trade size $250K–$500K | +10 |
| Trade size $500K–$1M | +12 |
| Trade size >$1M | +15 |
| Cluster: 3+ members, same ticker, 30d window | +15 |
| Committee with relevant jurisdiction | +10 *(not yet computed)* |
| Same-day as tracked bill activity | +10 *(not yet computed)* |

**Signal decay:** Each trade's points are multiplied by `0.5^(daysSince / halfLife)` where `halfLife` defaults to 21 days (configurable via `data/optimal_weights.json`).

**Rolling window:** 30 days from today.
**Cap per hub:** 40 pts (prevents outlier pile-ups from a single source).
**Data source:** `data/congress_feed.json` — refreshed 4x daily by GitHub Actions.

---

## Hub 2 — Insider Score

Derived from SEC EDGAR Form 4 filings. Matched by company name keywords.
**Limitation (v0.2):** EDGAR EFTS does not return role (CEO/CFO) or transaction type — only filing metadata. Role-based scoring requires parsing the full XML.

| Factor | Points |
|---|---|
| Any Form 4 match for tracked company, 14d window | +6 per filing |
| Cluster: 2+ filings, same company, 14d | +10 |
| Cluster: 3+ filings, same company, 14d | +15 |
| CEO (when role available) | +10 |
| CFO (when role available) | +8 |
| Director (when role available) | +6 |
| No 10b5-1 plan (when available) | +8 |
| Near 52-week low | +5 *(not yet computed)* |

**Rolling window:** 14 days from today.
**Cap per hub:** 40 pts.
**Data source:** `data/edgar_feed.json` — refreshed 4x daily by GitHub Actions.

### Company → Ticker Keyword Map
Used to match EDGAR company names (which have no ticker field) to tracked tickers.
Also used as a fallback: if the ticker string itself appears in the company name, it matches.

| Ticker | Match keywords (case-insensitive) |
|---|---|
| RTX | raytheon, rtx corp |
| NVDA | nvidia |
| OXY | occidental |
| TMDX | transmedics |
| FCX | freeport |
| PFE | pfizer |
| TSM | taiwan semiconductor |
| META | meta platforms |
| WFRD | weatherford |
| SMPL | simply good, atkins |
| ITA | *(ETF — no Form 4)* |

---

## Hub 3 — Institutional Score

**Status: Not yet live.** Currently uses hardcoded demo data.
Will be computed from 13F filings once institutional data pipeline is built.

| Factor | Points |
|---|---|
| Known smart-money manager initiates position | +15 |
| New position (not add-to) | +10 |
| Conviction increase >1% of portfolio | +8 |
| Unusual options sweep (volume vs OI) | +12 |

**Managers tracked:** Berkshire, Druckenmiller, Tepper, Ackman, Third Point, Pershing Square.

---

## Convergence Boosts

Applied once per ticker when multiple hubs fire within their respective windows.

| Hubs Active | Boost |
|---|---|
| Congressional + Insider | +20 |
| Congressional + Institutional | +20 |
| Insider + Institutional | +15 |
| All three | +40 |
| Any convergence + active tracked legislation | +15 additional |

**Active legislation check:** Ticker must appear in `BILLS[n].impactTickers` and the bill's vote date must be within 60 days.

---

## Total Score Calculation

```
total = hubCongress + hubInsider + hubInstitutional + convergenceBoost
```

Theoretical max: 40 + 40 + 40 + 40 + 15 = 175 (with all three hubs + legislation)
Practical max with current data (2 hubs): 40 + 40 + 20 + 15 = 115

| Score | Meaning | Display |
|---|---|---|
| < 40 | Below threshold — not surfaced | Hidden |
| 40–64 | Moderate — worth monitoring | Watchlist / Monitoring tier |
| ≥ 65 | Strong — trade idea generated | Trade Ideas tier |
| ≥ 95 | Exceptional — highest conviction | Highlighted card |

**Threshold is dynamic:** loaded from `data/optimal_weights.json` at startup (`window.SCORE_THRESHOLD`). Default is 65. The backtest engine can tune this value automatically.

---

## Entry Zone Logic

Generated at signal creation time from the price at the moment all hubs converge.

| Bound | Formula |
|---|---|
| Entry Lo | `signal_price × 0.97` |
| Entry Hi | `signal_price × 1.03` |
| Stop | Defined by thesis (below support, above resistance for shorts) |

**Zone statuses:** `in_zone` / `above_zone` (3–10% above hi) / `missed` (>10%) / `stale` (>30 days old)

---

## Signal Decay

Congressional scores now use exponential decay:
```
decayFactor = 0.5 ^ (daysSince / halfLife)
effectivePoints = rawPoints × decayFactor
```

`halfLife` defaults to 21 days, dynamically overridable via `SCORE_WEIGHTS.decay_half_life_days`.

| Condition | Action |
|---|---|
| Trade is recent (0 days) | Full points |
| Trade is 21 days old | Half points |
| Trade is 42 days old | Quarter points |
| `sigDate` older than 30 days | Show stale warning in overlay |
| Price moved >10% above entry hi (long) | Zone status = MISSED |

---

## Backtest Engine

The scoring weights and threshold are tuned automatically by the backtest engine (`backtest/` directory):

1. `collect_prices.py` — fetches historical OHLC from Finnhub
2. `run_event_study.py` — computes CAR (Cumulative Abnormal Return) at 5d/30d/90d for each signal event
3. `optimize_weights.py` — grid search over 1024 weight combinations to maximize `avg_car_30d × hit_rate`
4. Output: `data/optimal_weights.json` — loaded by frontend at startup

**Schedule:** GitHub Actions runs weekly (Sundays 02:00 UTC).
**Self-improving:** Each run incorporates new data, recalculates CARs, and re-optimizes weights.

**Current limitation:** Finnhub free tier does not provide historical OHLC (`/stock/candle` returns 403). Needs either a paid Finnhub plan or alternative data source (yfinance, Alpha Vantage, Polygon).

---

## Roadmap: Scoring Improvements

- [ ] Resolve historical price data source (paid Finnhub or alternative)
- [ ] Parse full EDGAR XML to extract role (CEO/CFO) and transaction type
- [ ] Add 10b5-1 plan flag from Form 4 XML
- [ ] Proximity to 52-week low scoring
- [ ] Committee jurisdiction scoring (requires congressional committee mapping)
- [ ] Institutional 13F pipeline (Phase 2)
- [ ] Historical accuracy per insider (multi-filing tracking)
- [ ] Sector-level convergence (multiple tickers in same sector = sector signal)
- [x] Signal decay (21-day half-life, exponential)
- [x] Dynamic threshold from backtest engine
- [x] Watchlist tier (40–64) + Trade Ideas tier (≥65)
- [x] Open universe scoring (all tickers in feeds, not just TRACKED 11)
