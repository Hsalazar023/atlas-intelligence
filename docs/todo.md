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
| ML weights saved | Fixed — method upgrades to walk_forward_ensemble when IC > 0 | OK |

---

## P0 — Immediate Fixes

- [x] **ML method not upgrading** — `_oos_ic` was saving but method stayed `feature_importance` because 5% gate never cleared. Fixed: use `walk_forward_ensemble` whenever IC > 0.
- [x] **insider_role fill rate** — Was 94.9% (51 missing). Remaining 51 are corporate/institutional names (KKR, Apollo, etc). Added step 3c: auto-assign `10% Owner` for corporate entity names.
- [x] **trade_pattern removed (v4)** — Confirmed pruned from ML features. IC stable at 0.072. No regression.
- [x] **Congress avg CAR is -0.36%** — Not a scoring bug. ML correctly separates: congress 80+ = +11.6% CAR, 60-79 = +11.5%, <40 = -4.2%. The -0.36% avg is dragged down by 1,263 low-scoring signals. High-score congress signals perform well.
- [x] **Top signals all BITB** — Fixed. MAX_PER_TICKER=3 cap in export_brain_data(). Current export has no ticker >3.

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
