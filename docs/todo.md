# ATLAS — Active Todo
*Updated Mar 1, 2026. Move completed items to docs/archive/completed-milestones.md.*

---

## Current System Health

| Metric | Value | Status |
|---|---|---|
| Total signals | 3,212 (2,207 congress + 1,005 EDGAR) | OK |
| Date range | Nov 2022 – Feb 2026 (39 months) | OK |
| OOS IC (classification) | +0.072 | OK |
| OOS Hit Rate | 55.3% (22 folds) | OK |
| Score range | 0 – 93.9 (avg 37.7) | OK |
| Brain export live on Vercel | Yes | OK |
| ML weights saved | IC didn't pass 5% gate — using feature_importance method | Investigate |

---

## P0 — Immediate Fixes

- [ ] **ML weights not saving** — `optimal_weights.json` has no `_oos_ic`. Investigate why method is still `feature_importance` when IC > 0.
- [ ] **insider_role fill rate 29.7%** — 51 EDGAR signals missing role. Check XML parsing.
- [ ] **trade_pattern removed (v4)** — 31% fill, pruned. Verify no regressions in IC.
- [ ] **Congress avg CAR is -0.36%** — EDGAR buys (+3.71%) have 10x more alpha. Scoring may overweight congress.
- [ ] **Top signals all BITB** — 8/10 top signals are one ticker. Needs ticker diversification in export.

---

## P1 — Brain Improvements

- [ ] Score calibration — max 2-3 signals per ticker in brain_signals.json
- [ ] Add `--self-check`: IC trend, score concentration, feature drift
- [ ] Feature auto-pruning: drop features with <1% importance for 3+ runs
- [ ] Congress vs EDGAR source weighting adjustment

---

## P2 — Frontend & UX

- [ ] "Why this signal" context on trade idea cards (role, convergence, person record)
- [ ] Score explanation tooltip: ML confidence + magnitude + convergence breakdown
- [ ] Brain performance dashboard: historical accuracy, IC trend, feature importance
- [ ] Remove remaining hardcoded demo data (BILLS array, institutional cards, options flow)
- [ ] Mobile-responsive improvements
