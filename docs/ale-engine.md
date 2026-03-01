# ATLAS — Adaptive Learning Engine (ALE)
*Reference doc. Update when ALE schema or pipeline changes.*

---

## Overview

Self-improving scoring brain. Accumulates signals, tracks 5 time horizons of forward returns, discovers predictive features, tracks person-level performance, auto-tunes weights.

- **Database:** `data/atlas_signals.db` (SQLite: `signals`, `feature_stats`, `weight_history`)
- **Diagnostics:** `data/ale_diagnostics.html` (visual dashboard) + `data/ale_analysis_report.md` (deep-dive)
- **Weights:** `data/optimal_weights.json` (loaded by frontend)

---

## CLI

| Command | Purpose |
|---|---|
| `--daily` | Ingest + enrich + score + export + diagnostics |
| `--analyze` | Feature analysis + ML walk-forward + weight update + score + export |
| `--score` | Score all signals with ML + export brain data |
| `--summary` | Console status + person leaderboards |
| `--diagnostics` | Generate HTML dashboard + markdown analysis report (standalone) |
| `--export` | Export brain_signals.json + brain_stats.json only |
| `bootstrap_historical.py` | One-time: populate DB with ~39 months of data |
| `backfill_edgar_xml.py` | One-time: parse XML for EDGAR signals, delete non-purchases, enrich buys |

---

## ML Engine

- **Models:** RF + LightGBM ensemble (classification + regression), walk-forward validation
- **27 features** (see list below)
- **CAR:** BHAR `(1+stock)/(1+spy)-1`, winsorized 1st/99th percentile, hard bounds [-100%, +300%]
- **Safety:** Weights only auto-update when OOS IC improves >5%
- **Min samples:** 200 train / 20 test per fold

### Feature List (27)

| Category | Features |
|---|---|
| Source | source |
| Trade | trade_size_points, disclosure_delay, relative_position_size |
| Clustering | same_ticker_signals_7d/30d, has_convergence, convergence_tier, cluster_velocity |
| Person | person_trade_count, person_hit_rate_30d |
| Classification | insider_role, trade_pattern, sector, market_cap_bucket |
| Price-based | price_proximity_52wk, momentum_1m/3m/6m, volume_spike |
| Market context | vix_at_signal, yield_curve_at_signal, credit_spread_at_signal |
| Catalysts | days_to_earnings, days_to_catalyst |
| Derived (DB) | insider_buy_ratio_90d, sector_avg_car, vix_regime_interaction |

---

## Data Quality (Critical)

**EDGAR signals must be filtered to purchases only.** EFTS doesn't expose transaction type — Form 4 XML must be parsed to determine direction (P=purchase, S=sale, M=exercise, A=grant).

- **Bootstrap:** Now enriches via XML before insertion, filters to buys only
- **Backfill:** `backfill_edgar_xml.py` cleans existing DB signals
- **Daily pipeline:** `fetch_data.py` already enriches via `enrich_form4_xml()`

---

## Bootstrap Pipeline

1. EDGAR historical (21mo Form 4 filings) + **XML enrichment + buy filtering**
2. FMP congress (senate + house)
2b. Sector + market cap map (FMP profile API)
3. Aggregate features
4. Price collection (incremental — skip/backfill/forward-fill)
5. Backfill CARs (BHAR, 5 horizons)
6. Person track records
7. Feature enrichment (52wk, trade pattern, momentum, volume, insider ratio, sector CAR, VIX interaction)
7b. Sector + market_cap_bucket NULL backfill
8. Market context (FRED: VIX, T10Y2Y, credit OAS)
9. Feature stats + weight generation
10. ML classification (walk-forward)
11. ML regression (walk-forward)
→ Dashboard + diagnostics + data quality report

---

## Convergence Tiers

| Tier | Condition |
|---|---|
| 0 | No convergence |
| 1 | Same ticker, 2+ sources, 60d window |
| 2 | 3+ signals, 2+ sources, same sector, 30d |
