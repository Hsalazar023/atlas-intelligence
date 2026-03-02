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

## Session 2 Checkpoint (Mar 1, 2026 — Data Freshness & Signal Quality)

### Completed
- [x] **Task 0 — House XML Scraper.** Added `fetch_house_disclosures()` to fetch_data.py. Fetches PTR XML index from disclosures-clerk.house.gov, parses filing pages for trades, normalizes to congress_feed.json schema. Wired into pipeline after FMP with clean fallback. Dedup uses 4-field key (Ticker+Rep+Date+Transaction), prefers House records. No API key needed. Senate deferred (Phase 3).
- [x] **Task 1 — Feature pruning + role_quality_bonus.** Pruned 5 dead-weight features from FEATURE_COLUMNS (convergence_tier, has_convergence, days_to_catalyst, relative_position_size, cluster_velocity). Kept insider_role — role VALUES are predictive. Added `_compute_role_quality()` to --analyze: learns multipliers by role from historical CAR. Applied multiplicatively in `score_all_signals()` after source_mult. Added `features_pruned_history` to brain_stats.json. Feature count: 28→23.
- [x] **Task 2 — Trader quality tiers + fade signals.** Added `_compute_trader_tiers()`: classifies traders as elite/good/neutral/fade based on hit_rate + avg_car. Fade-tier traders get ×0.35 score multiplier (contra-signal). Added trader_tier to brain_signals.json per signal. Added leaderboard to brain_stats.json. Updated scoring-logic.md with tier rules and fade formula.
- [x] **Task 3 — Volume + Analyst features.** volume_spike already is relative volume (signal vol / 20d avg). Added `volume_dry_up` (bool: rel_vol < 0.4 = accumulation signal). Added `fetch_volume_data()` + `fetch_analyst_data()` to fetch_data.py using yfinance. New ML features: `volume_dry_up`, `analyst_revision_30d`, `analyst_consensus`, `analyst_insider_confluence`. Enrichment reads from data/analyst_data.json. Added yfinance to fetch-data.yml. Feature count: 23→27.
- [x] **Task 4 — Sector diversification.** Added MAX_PER_SECTOR=8 cap in export. After ticker cap, sector cap applied. Overflow signals backfill if export < 50. Added sector rank-normalization: `final = 0.75 × absolute + 0.25 × sector_percentile` — prevents one sector from monopolizing. Added `sector_distribution` to brain_stats.json.

---

## Session 1 Checkpoint (Mar 1, 2026 — Brain Improvement)

### Completed — All 6 Tasks
- [x] **Task 1 — Harmful feature value detection + pruning visibility.** Added harmful value detection to `run_self_check` (CAR<-2%, n≥30). Added `pruning_log` + `ticker_distribution` to brain_stats.json. Confirmed ML trees handle cluster_velocity=fast via categorical encoding. Pruning is advisory — surfaces harmful values as warnings.
- [x] **Task 2 — Restructured --self-check / brain_health.json.** Refactored output to structured format with named checks (`ic_trend`, `hit_rate`, `data_freshness`, `feature_drift`, `score_concentration`, `harmful_features`), `overall_status` (critical/degraded/healthy), and `recommendations[]`. Added `--self-check` step to backtest.yml after `--daily`.
- [x] **Task 3 — Source-aware scoring.** Added `_compute_source_quality()` to `run_analyze` — computes EDGAR/Congress/convergence multipliers from historical CAR. Stored in `optimal_weights.json → _source_quality`. Applied in `score_all_signals` as `total = clamp(raw × source_mult, 0, 100)`. Multipliers auto-update each analyze run.
- [x] **Task 4 — Ticker diversification cap.** Already existed (MAX_PER_TICKER=3, EXPORT_LIMIT=50). Added `ticker_distribution` stats to brain_stats.json (unique_tickers, max_per_ticker, top_5).
- [x] **Task 5 — Signal expansion roadmap.** Added `Phase 4B — Signal Expansion Roadmap` to docs/roadmap.md. 10 ranked signals with data source, difficulty, ML/convergence role. Tier A (free, now): relative volume, analyst revisions, committee membership, earnings surprise. Tier B (medium): SI change, lobbying, P/C ratio. Tier C (paid): options flow, FinBERT, dark pool.
- [x] **Task 6 — Failure alerting.** Added ntfy.sh notifications to both workflows: success (brain health status) + failure alerts. Added NTFY_CHANNEL to docs/architecture.md API Keys section.

---

## Session 3 Checkpoint (Mar 1, 2026 — Pipeline Fixes & Signal Expansion)

### Completed — All Tasks
- [x] **Task 0A — Fix FMP Insider Trades 404.** Rewrote `fetch_fmp_insiders()` with multi-endpoint fallback: stable → v4-rss → v4. Auto-detects working endpoint, filters purchases from RSS feed.
- [x] **Task 0B — Fix House XML Scraper URL.** Correct URL `financial-pdfs/{YEAR}FD.xml`. PTR PDFs are encrypted — metadata-only approach.
- [x] **Task 0C — Fix --self-check KeyError.** Fixed dict iteration, missing keys (`recommendations`).
- [x] **Task 1 — Feature backfill.** Added `backfill_features()` + `--backfill` CLI flag. Resets v5/v6 columns to NULL, re-enriches, reports before/after counts.
- [x] **Task 2 — Committee membership.** Added `fetch_committee_data()` (GitHub unitedstates/congress-legislators, free). Maps committees to sectors. Added `committee_overlap` ML feature — 1 if congress member sits on oversight committee for traded sector. Added pyyaml to fetch-data.yml.
- [x] **Task 3 — Earnings surprise.** Added `fetch_earnings_surprise()` (yfinance EPS data). Added `earnings_surprise` ML feature — EPS surprise % for most recent quarter.
- [x] **Task 4 — News sentiment.** Added `fetch_news_sentiment()` (Finnhub headlines + VADER/keyword scoring). Added `news_sentiment_30d` ML feature. Falls back to keyword heuristic if nltk not installed.
- [x] **Task 5 — Time-weighted learning.** Added `_compute_time_weights()` with 12-month half-life. Applied to `walk_forward_train`, `walk_forward_regression`, and `train_full_sample` via `sample_weight`. Recent signals weighted higher.
- [x] **Task 6 — Feature candidate evaluation.** Added `evaluate_feature_candidates()` to ml_engine.py. Tests candidate columns against baseline IC via walk-forward. Added `--eval-features COL [COL...]` CLI flag. Reports IC delta + recommendation (ADD/NEUTRAL/SKIP).
- [x] **Task 7 — Historical data expansion.** Increased EDGAR bootstrap default from 635 days (~21 months) to 900 days (~30 months). Made configurable via `--edgar-days N` CLI parameter.

Feature count: 27 → 30 (v6). New: `committee_overlap`, `earnings_surprise`, `news_sentiment_30d`.

---

## Frontend & UX

- [ ] "Why this signal" context on trade idea cards (role, convergence, person record, trader tier)
- [ ] Score explanation tooltip: ML confidence + magnitude + convergence breakdown
- [ ] Brain performance dashboard: historical accuracy, IC trend, feature importance
- [ ] Remove remaining hardcoded demo data (BILLS array, institutional cards, options flow)
- [ ] Mobile-responsive improvements
- [ ] Trader tier badges (elite/good/fade) on signal cards
