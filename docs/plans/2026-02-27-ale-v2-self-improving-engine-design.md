# ALE v2 — Self-Improving Edge Engine

*Design approved Feb 27, 2026*

## Context

The ALE v1 has 10,445 signals (99 congress + 10,346 EDGAR) across 21 months. Performance: 45.4% hit rate, +0.62% avg CAR at 30d. Convergence is broken (n=1) because congressional data is severely underfed (99 signals vs 10,346 EDGAR). The feature set is shallow (9 features) and misses research-backed predictors. Weight tuning is a simple grid search with no cross-validation.

**Goal:** Transform the ALE from a basic feature tracker into a fully autonomous, self-improving ML system that discovers edge, validates it out-of-sample, and auto-tunes its own scoring weights — while expanding the data foundation to enable real convergence detection.

## Budget

~$15/mo starting point (Financial Modeling Prep starter tier for congressional data).

## Section 1: Data Foundation Expansion

### 1A — FMP Congressional Data Pipeline

Replace QuiverQuant (free tier blocked) with Financial Modeling Prep API:
- **Endpoints:** Senate Trading Activity + House Financial Disclosures
- **Historical depth:** 3+ years of trades (vs current 150 recent)
- **Expected volume:** 99 → 1,500+ congressional buy signals
- **Integration:** New `fetch_fmp_congress()` in fetch_data.py, new `ingest_fmp_congress()` in learning_engine.py
- **Fields to capture:** ticker, transaction_date, representative, party, chamber, amount, transaction_type, disclosure_date
- **Disclosure delay feature:** `disclosure_date - transaction_date` = filing urgency

### 1B — EDGAR Sell Signal Tracking

Currently filtering to buy-only. Expand to store sells in a separate table/flag:
- Not scored as buy signals
- Used by ML to learn exit timing patterns
- Used for "Notable Exit Signals" section (currently hardcoded)

### 1C — Sector Tagging

Every signal gets a GICS sector tag:
- Source: Static ticker→sector CSV (free, ~8,000 tickers from SEC SIC codes or similar)
- Stored in new `sector` column (already exists in schema, currently NULL)
- Enables sector-level convergence detection

## Section 2: Multi-Tier Convergence

### Current (broken)
- Binary: same ticker in both congress + EDGAR within 30d
- Result: 1 convergence event in 21 months

### New Model

| Tier | Definition | Boost | Min Signals |
|---|---|---|---|
| **Tier 1 — Ticker** | Same ticker in 2+ hubs within window | +20 | 2 |
| **Tier 2 — Sector** | 3+ signals from 2+ hubs in same GICS sector | +10 | 3 |
| **Tier 3 — Thematic** | Sector convergence + active legislation | +15 | 3 + bill |

### Window Expansion
- Congress lookback: 30d → **60 days** (STOCK Act allows 45-day disclosure delay)
- EDGAR lookback: stays 30d (Form 4 must be filed within 2 business days)
- ML will eventually learn optimal windows from data

### Convergence in the Database
- New columns: `convergence_tier` (0/1/2/3), `convergence_sector`, `convergence_tickers`
- Replaces binary `has_convergence` flag

## Section 3: New Features (Research-Backed)

### 3A — Opportunistic vs Routine Classification
**Source:** Cohen, Malloy, Pomorski (2012) — "Decoding Inside Information"
- **Method:** Analyze each insider's trading history. If they trade in the same calendar month 3+ years in a row → routine. Otherwise → opportunistic.
- **Impact:** Opportunistic trades show 82 bps/month alpha vs zero for routine.
- **Implementation:** New `classify_insider_pattern()` function. Requires 2+ years of history per insider. Feature bucket: `trade_pattern = routine | opportunistic | insufficient_history`

### 3B — CFO Priority
**Source:** Institutional research showing CFOs have less narrative ambiguity than CEOs.
- **Current:** CEO +4, CFO +3 in frontend scoring
- **Change:** Let ML discover the optimal weighting. Add `is_cfo` as an explicit boolean feature. Initial hypothesis: CFO ≥ CEO.

### 3C — 52-Week Price Proximity
**Already in CLAUDE.md spec, never implemented.**
- **Formula:** `proximity = (price_at_signal - 52wk_low) / (52wk_high - 52wk_low)`
- **Range:** 0.0 (at 52-wk low) to 1.0 (at 52-wk high)
- **Feature buckets:** `near_low` (<0.2), `lower_half` (0.2-0.5), `upper_half` (0.5-0.8), `near_high` (>0.8)
- **Data source:** yfinance 252-day high/low (already fetching price history)

### 3D — Market Cap Bucket
- **Source:** Research shows insider edge strongest in small caps with limited analyst coverage.
- **Buckets:** `micro` (<$300M), `small` ($300M-$2B), `mid` ($2B-$10B), `large` (>$10B)
- **Data source:** yfinance or FMP market cap data

### 3E — Relative Buy Size (vs Market Cap)
- **Formula:** `buy_value / market_cap`
- **Why:** $500K at a $200M company is 25bps of the company. $500K at Apple is noise.
- **Buckets:** `tiny` (<1bp), `small` (1-10bp), `meaningful` (10-50bp), `significant` (>50bp)

### 3F — Sector Momentum
- **Formula:** 30d sector ETF return vs SPY
- **Why:** Insider buying into a weak sector = contrarian conviction
- **Buckets:** `sector_weak` (<-5% vs SPY), `sector_neutral` (-5% to +5%), `sector_strong` (>+5%)

### 3G — Disclosure Delay (Congress only)
- **Formula:** `disclosure_date - transaction_date` in days
- **Why:** Short delay = urgency/conviction. Long delay = routine compliance.
- **Buckets:** `urgent` (<7d), `normal` (7-30d), `slow` (30-45d), `late` (>45d)

### 3H — Cluster Velocity
- **Formula:** Average days between consecutive signals on same ticker
- **Why:** 3 buys in 3 days >> 3 buys over 30 days
- **Buckets:** `burst` (<3d avg), `fast` (3-7d), `moderate` (7-14d), `slow` (>14d)

## Section 4: ML Engine (Self-Improving Core)

### Architecture

```
Daily Pipeline (Mon-Fri, 10 PM UTC via GitHub Actions):

1. COLLECT    — Ingest new FMP congress + EDGAR Form 4 signals
                Tag sectors, compute all features
2. BACKFILL   — Update outcomes at 5d/30d/90d/180d/365d
                CARs vs SPY benchmark
3. TRAIN      — Walk-forward model training (weekly, Mondays)
                Random Forest + LightGBM ensemble
                6-month train window → 1-month test window
                Rolling forward, no lookahead bias
4. EVALUATE   — Out-of-sample metrics
                Information Coefficient (IC)
                Hit rate + avg CAR by prediction decile
                Feature importance via SHAP
5. UPDATE     — Deploy new weights IF OOS improvement > 5%
                Log everything to weight_history
                Safety: auto-revert if live 30d performance degrades
6. EXPLORE    — Auto-generate and test new features (monthly)
                Interaction terms (feature A × feature B)
                New bucketing thresholds
                Report discoveries in dashboard
```

### Walk-Forward Validation
- **Train window:** 6 months of signals with filled 30d outcomes
- **Test window:** Next 1 month (out-of-sample)
- **Slide:** Move forward 1 month, retrain
- **Minimum:** Need 8+ months of data → we have 21 months → 15 walk-forward folds
- **No lookahead:** Test data is always strictly after train data

### Models
- **Random Forest:** Handles non-linear interactions, robust to noise, interpretable feature importance
- **LightGBM:** State-of-the-art for tabular data, fast training, handles categorical features natively
- **Ensemble:** Average predicted probabilities from both models. Ensembles outperform single models (Gu, Kelly, Xiu 2020).
- **Target variable:** Binary — `car_30d > 0` (positive 30-day excess return)

### Safety Rails
- **Minimum improvement:** New weights deploy only if OOS IC improves by >5% over current
- **Rollback trigger:** If live performance (next 30d batch) shows IC decline >10% vs previous, auto-revert
- **Weight history:** Every update logged with timestamp, OOS metrics, model version, feature set
- **Baseline lock:** Keep DEFAULT_WEIGHTS as a floor — never deploy weights that underperform the original static weights on OOS data

### Dependencies
- `scikit-learn` — Random Forest, metrics
- `lightgbm` — Gradient boosting
- `shap` — Feature importance (optional, can use built-in importance initially)

## Section 5: Information Coefficient (IC) as Primary Metric

### Why IC > Hit Rate
- **Hit rate** = % of signals with positive CAR. A 55% hit rate with +0.1% avg winner and -5% avg loser is terrible.
- **IC** = Spearman rank correlation between predicted score and actual forward return. Captures magnitude, not just direction.
- IC of 0.05+ is considered good in institutional quant.
- IC is the standard metric for evaluating alpha factors.

### Implementation
- Compute IC at each horizon (5d, 30d, 90d)
- Report rolling 3-month IC in dashboard
- Use IC as the optimization target for walk-forward training

## Section 6: Dashboard Upgrades

### New ale_dashboard.json Fields

```json
{
  "ml_model_performance": {
    "random_forest_ic_30d": 0.07,
    "lightgbm_ic_30d": 0.08,
    "ensemble_ic_30d": 0.09,
    "oos_hit_rate_30d": 0.54,
    "walk_forward_folds": 15,
    "last_training_date": "2026-02-27"
  },
  "feature_importance": [
    {"feature": "trade_pattern_opportunistic", "importance": 0.15},
    {"feature": "price_proximity_52wk", "importance": 0.12}
  ],
  "convergence_analysis": {
    "tier1_ticker": {"count": 45, "hit_rate": 0.58, "avg_car": 0.032},
    "tier2_sector": {"count": 120, "hit_rate": 0.52, "avg_car": 0.018},
    "tier3_thematic": {"count": 15, "hit_rate": 0.65, "avg_car": 0.045}
  },
  "alpha_decay": {
    "optimal_holding_period_days": 30,
    "ic_by_horizon": {"5d": 0.03, "30d": 0.08, "90d": 0.05}
  },
  "regime": {
    "current": "bull",
    "signal_efficacy_bull": 0.52,
    "signal_efficacy_bear": 0.48
  }
}
```

## Implementation Priority

1. **FMP integration + congress data expansion** (unblocks everything)
2. **Sector tagging + multi-tier convergence**
3. **New features** (opportunistic/routine, 52-wk proximity, market cap, etc.)
4. **ML engine** (Random Forest + LightGBM + walk-forward)
5. **IC metric + dashboard upgrades**
6. **Auto-exploration** (interaction terms, threshold tuning)

## Files to Modify/Create

| File | Action |
|---|---|
| `scripts/fetch_data.py` | Add `fetch_fmp_congress()`, sector tagging |
| `backtest/learning_engine.py` | New features, ML training, IC computation, convergence tiers |
| `backtest/shared.py` | New constants, sector map loader, feature helpers |
| `backtest/ml_engine.py` | **NEW** — Random Forest + LightGBM + walk-forward + SHAP |
| `backtest/bootstrap_historical.py` | FMP historical congress fetch, sector backfill |
| `data/sector_map.json` | **NEW** — ticker → GICS sector mapping |
| `.github/workflows/backtest.yml` | Add ML training step (weekly), FMP API key secret |
| `atlas-intelligence.html` | Updated convergence scoring, new dashboard display |
| `requirements.txt` or setup | Add scikit-learn, lightgbm dependencies |
