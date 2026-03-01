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

## Frontend & UX

- [ ] "Why this signal" context on trade idea cards (role, convergence, person record)
- [ ] Score explanation tooltip: ML confidence + magnitude + convergence breakdown
- [ ] Brain performance dashboard: historical accuracy, IC trend, feature importance
- [ ] Remove remaining hardcoded demo data (BILLS array, institutional cards, options flow)
- [ ] Mobile-responsive improvements
