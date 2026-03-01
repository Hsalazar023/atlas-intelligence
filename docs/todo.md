# ATLAS — Active Todo
*Updated Mar 1, 2026. Move completed items to docs/archive/completed-milestones.md.*

---

## Current System Health

| Metric | Value | Status |
|---|---|---|
| Total signals | 3,212 (2,207 congress + 1,005 EDGAR) | ✅ Clean |
| Date range | Nov 2022 – Feb 2026 (39 months) | ✅ |
| OOS IC (classification) | +0.072 | ✅ Positive |
| OOS Hit Rate | 55.3% (22 folds) | ✅ |
| Signals scored | 3,212 / 3,212 (100%) | ✅ |
| Score range | 0 – 93.9 (avg 37.7) | ✅ |
| EDGAR buy-filtered | Yes (946 purchases + 59 unknown) | ✅ |
| Brain export live on Vercel | Yes (brain_signals + brain_stats) | ✅ |
| ML weights saved | ⚠️ IC didn't pass 5% gate — using feature_importance method |

---

## P0 — Immediate Fixes

- [ ] **ML weights not saving** — `optimal_weights.json` has no `_oos_ic` because the 5% gate compares against `current_ic=0` but uses `> 0 * 1.05 = 0` which should pass. Investigate why method is still `feature_importance`.
- [ ] **insider_role fill rate 29.7%** — 51 EDGAR signals + all 2,207 congress signals have no role. Congress signals don't have roles (expected), but EDGAR backfill should have caught more. Check XML parsing for the 51 missing.
- [ ] **trade_pattern fill rate 31.3%** — needs 3yr of history per person to compute. Most people only have 1-2 trades. Consider removing as ML feature or relaxing to 1yr.
- [ ] **person_hit_rate fill rate 73.6%** — 26% of signals have no person track record. Likely new/first-time traders. Fine to leave as 0.
- [ ] **Congress avg CAR is -0.36%** — Congressional trades as a class are slightly negative. EDGAR buys are +3.71%. Scoring formula may be overweighting congress signals.
- [ ] **Top signals are all BITB/congress** — 8 of top 10 signals are BITB congressional trades from Mar-Apr 2025. Concentration risk in scoring. May need ticker diversification in export.

---

## P1 — Brain Improvements

- [ ] Score calibration — top signals shouldn't be dominated by one ticker
- [ ] Deduplicate export — max 2-3 signals per ticker in brain_signals.json
- [ ] Add `--self-check` diagnostic: IC trend, score concentration, feature drift
- [ ] Feature auto-pruning: drop features with <1% importance for 3+ runs
- [ ] Congress vs EDGAR score weighting: EDGAR buys produce 10x more alpha

---

## P2 — Frontend & UX

- [ ] Show "why this signal matters" context on trade idea cards (role, convergence, person record)
- [ ] Score explanation tooltip: show ML confidence, magnitude bonus, convergence bonus breakdown
- [ ] Brain performance dashboard: historical accuracy, IC trend, feature importance chart
- [ ] Remove remaining hardcoded demo data (BILLS array, institutional cards, options flow)
- [ ] Mobile-responsive improvements

---

## Completed (This Session)
- [x] `score_all_signals()` — ML scoring pipeline writes total_score to DB
- [x] `train_full_sample()` + `prepare_features_all()` in ml_engine.py
- [x] `--score` CLI flag for standalone scoring + export
- [x] `--daily` and `--analyze` auto-score + auto-export
- [x] brain_signals.json + brain_stats.json deployed to Vercel
- [x] Frontend top signals + trade ideas now use Brain ML scores
- [x] computeConvergenceScore() overlays Brain totalScore
