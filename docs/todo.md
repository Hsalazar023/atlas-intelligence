# ATLAS — Active Todo
*Updated Mar 1, 2026. Move completed items to docs/archive/completed-milestones.md.*

---

## Current System Health

| Metric | Value | Status |
|---|---|---|
| Total signals | 3,212 (2,207 congress + 1,005 EDGAR) | OK |
| Date range | Nov 2022 – Feb 2026 (39 months) | OK |
| OOS IC | +0.0766 (walk_forward_ensemble) | OK |
| OOS Hit Rate | 54.8% (22 folds) | OK |
| insider_role fill | 100% EDGAR | OK |
| Feature fills ≥80% | 5/8 | OK |
| Score bands | 80+ = +61.7% CAR, 65-79 = +25.3% | OK |
| Brain export live on Vercel | Yes | OK |

---

## Session Checkpoint (Mar 1, 2026 — Brain Improvement)

### Completed — All 6 Tasks
- [x] **Task 1 — Harmful feature value detection + pruning visibility.** Added harmful value detection to `run_self_check` (CAR<-2%, n≥30). Added `pruning_log` + `ticker_distribution` to brain_stats.json. Confirmed ML trees handle cluster_velocity=fast via categorical encoding. Pruning is advisory — surfaces harmful values as warnings.
- [x] **Task 2 — Restructured --self-check / brain_health.json.** Refactored output to structured format with named checks (`ic_trend`, `hit_rate`, `data_freshness`, `feature_drift`, `score_concentration`, `harmful_features`), `overall_status` (critical/degraded/healthy), and `recommendations[]`. Added `--self-check` step to backtest.yml after `--daily`.
- [x] **Task 3 — Source-aware scoring.** Added `_compute_source_quality()` to `run_analyze` — computes EDGAR/Congress/convergence multipliers from historical CAR. Stored in `optimal_weights.json → _source_quality`. Applied in `score_all_signals` as `total = clamp(raw × source_mult, 0, 100)`. Multipliers auto-update each analyze run.
- [x] **Task 4 — Ticker diversification cap.** Already existed (MAX_PER_TICKER=3, EXPORT_LIMIT=50). Added `ticker_distribution` stats to brain_stats.json (unique_tickers, max_per_ticker, top_5).
- [x] **Task 5 — Signal expansion roadmap.** Added `Phase 4B — Signal Expansion Roadmap` to docs/roadmap.md. 10 ranked signals with data source, difficulty, ML/convergence role. Tier A (free, now): relative volume, analyst revisions, committee membership, earnings surprise. Tier B (medium): SI change, lobbying, P/C ratio. Tier C (paid): options flow, FinBERT, dark pool.
- [x] **Task 6 — Failure alerting.** Added ntfy.sh notifications to both workflows: success (brain health status) + failure alerts. Added NTFY_CHANNEL to docs/architecture.md API Keys section.

---

## Frontend & UX

- [ ] "Why this signal" context on trade idea cards (role, convergence, person record)
- [ ] Score explanation tooltip: ML confidence + magnitude + convergence breakdown
- [ ] Brain performance dashboard: historical accuracy, IC trend, feature importance
- [ ] Remove remaining hardcoded demo data (BILLS array, institutional cards, options flow)
- [ ] Mobile-responsive improvements
