# ATLAS Brain — Integration Framework
*The Brain = ALE + ML engine. Powers all scoring, signals, and self-improvement.*

---

## What the Brain Is

The Brain is the unified intelligence layer: `learning_engine.py` + `ml_engine.py` + their outputs. It ingests raw data, learns what predicts alpha, scores every signal, and exports decisions that the frontend renders. No hardcoded trade ideas — everything flows from the Brain.

### Foundational Principle: Data Quality First
The Brain is only as good as its data. Clean, accurate, complete data is the core foundation. All ML accuracy improvements start with data quality — feature engineering and model tuning come second.

---

## Current Brain Capabilities (Built)

| Capability | Status | Output |
|---|---|---|
| Signal ingestion (EDGAR + Congress) | ✅ | `atlas_signals.db` |
| EDGAR purchase filtering (XML parsing) | ✅ | Only genuine buys retained |
| 5-horizon CAR tracking | ✅ | BHAR per signal |
| 27-feature ML scoring (RF + LightGBM) | ✅ | Classification + regression |
| Walk-forward validation | ✅ | OOS IC tracking |
| Person-level track records | ✅ | Hit rate, avg CAR per person |
| Sector-level stats | ✅ | `sector_avg_car` per sector |
| Convergence tier detection | ✅ | Tier 0/1/2 |
| Auto-weight tuning | ✅ | `optimal_weights.json` (5% IC gate) |
| Diagnostics dashboard | ✅ | `ale_diagnostics.html` + report |

---

## Brain → Frontend Pipeline

```
Daily Pipeline (--daily):
  1. Ingest new signals (EDGAR + Congress)
  2. Collect prices (yfinance)
  3. Backfill CARs + enrich features
  4. Score all signals via ML
  5. Export brain_signals.json ← NEW
  6. Export optimal_weights.json (existing)

Frontend loads:
  - data/brain_signals.json   → signal table, trade ideas, alerts, ticker ribbon
  - data/optimal_weights.json → scoring thresholds, decay params
  - data/brain_stats.json     → sector stats, score tier returns, alpha metrics
  - data/congress_feed.json   → congressional table (existing)
  - data/edgar_feed.json      → insider table (existing)
```

### New Export Files Needed

| File | Contents | Replaces |
|---|---|---|
| `brain_signals.json` | Top-scored signals with entry/target/stop, convergence tier, ML confidence | `TRACKED` object |
| `brain_stats.json` | Sector avg CARs, score-tier win rates, overall alpha, committee correlations, heatmap data | All hardcoded stats sections |

---

## What the Brain Needs to Power (Roadmap)

### Phase 1 — Replace Hardcoded Data (Next)
*Goal: Every number on the site comes from the Brain.*

| Frontend Section | Brain Source | Export Key |
|---|---|---|
| Top signals + scores | ML-scored signals ranked by confidence | `brain_signals[].score` |
| Trade ideas (entry/target/stop) | Signal price ± volatility-adjusted zones | `brain_signals[].zones` |
| Ticker ribbon | Top 10 signals by score | `brain_signals[:10]` |
| Score tier returns + win rates | ALE feature_stats by score bucket | `brain_stats.score_tiers` |
| Sector performance | `sector_avg_car` from DB | `brain_stats.sectors` |
| Committee correlation | Computed from congressional DB (member-committee-ticker match rates) | `brain_stats.committees` |
| Congressional heatmap | Frequency count of buys per ticker, 30d window | `brain_stats.congress_heatmap` |
| Signal alpha headline | Weighted avg CAR of high-score signals | `brain_stats.alpha` |
| Notable exits | EDGAR sales filtered by role + size | `brain_signals[].exits` |

### Phase 2 — New Data Sources
*Goal: Feed the Brain more data to improve predictions.*

| Source | What It Adds | Effort |
|---|---|---|
| 13F filings (SEC) | Institutional flows → Hub 3 scoring | Medium — build parser |
| Congress.gov API | Live bill status, vote dates, committee assignments | Low — API is free |
| News sentiment (RSS/API) | Event-driven catalyst detection | Low — RSS free, sentiment model needed |
| FRED macro data | VIX, yield curve, credit spreads (partially built) | Low — expand existing |
| Earnings calendar | `days_to_earnings` feature accuracy | Low — free APIs |

### Phase 3 — Self-Improvement
*Goal: Brain gets smarter autonomously.*

| Feature | How It Works |
|---|---|
| Feature auto-pruning | Drop features with <1% importance for 3 consecutive runs |
| Feature suggestion | When IC stagnates >5 runs, log "try adding X" based on residual analysis |
| Anomaly detection | Flag signals where ML prediction ≠ actual by >2 std dev → investigate |
| Model drift monitoring | Track IC trend over time; alert when declining |
| Hyperparameter auto-tune | Grid search on walk-forward splits (already partially built) |
| Backtest reporting | Auto-generate "what would Brain have caught" reports for historical events |

### Phase 4 — Brain as Full Dashboard Engine
*Goal: Brain powers every dynamic element on the site.*

| Element | Brain Role |
|---|---|
| Live alerts | Brain scores new signals in real-time; push when ≥ threshold |
| Trade ideas | Auto-generated when convergence ≥ Tier 1 + score ≥ 85 |
| Risk warnings | Brain detects negative convergence (multiple bearish signals) |
| Market context | Brain weighs VIX regime, yield curve → adjust all scores |
| Watchlist suggestions | "Brain picked up early signal on X" — below threshold but rising |
| Performance tracking | Track Brain's historical accuracy and display on site |

---

## Brain CLI (Existing + Planned)

| Command | Purpose | Status |
|---|---|---|
| `--daily` | Ingest + enrich + score + export | ✅ Built |
| `--analyze` | Feature analysis + ML + weight update | ✅ Built |
| `--summary` | Console status + leaderboards | ✅ Built |
| `--diagnostics` | HTML dashboard + analysis report | ✅ Built |
| `--export` | Generate `brain_signals.json` + `brain_stats.json` | 🔜 Next |
| `--self-check` | IC trend, feature drift, model health | 🔜 Phase 3 |

---

## Document Map (Updated)

| Doc | Scope | When to Update |
|---|---|---|
| `docs/brain.md` (this) | Brain framework, integration plan, phases | When Brain capabilities change |
| `docs/ale-engine.md` | ALE internals: features, schema, bootstrap | When engine code changes |
| `docs/scoring-logic.md` | Frontend scoring formulas, hubs, zones | When scoring rules change |
| `docs/todo.md` | Immediate tasks + audit findings | Each session |
| `docs/roadmap.md` | Long-term phases | When priorities shift |
| `docs/architecture.md` | Stack, file locations, API keys | When infrastructure changes |
