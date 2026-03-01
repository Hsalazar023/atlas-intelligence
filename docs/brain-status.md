# ATLAS Brain — Status & Health
*Updated after each analyze run. Current: Mar 1, 2026.*

---

## Latest Run: Mar 1, 2026

### Post-Cleanup Results
| Metric | Pre-Cleanup (Feb 28) | Post-Cleanup (Mar 1) | Change |
|---|---|---|---|
| Total signals | 12,769 | 3,212 | -75% (noise removed) |
| EDGAR signals | 10,563 | 1,005 | -90% (buys only) |
| OOS IC | -0.02 | **+0.072** | Positive! |
| OOS Hit Rate | ~50% | **55.3%** | +5.3pp |
| Walk-forward folds | — | 22 | OK |

### Scoring Pipeline: Live
All 3,212 signals scored (0-100). Brain export deployed to Vercel.

| Tier | Count | Pct |
|---|---|---|
| 80+ (strong buy) | 27 | 0.8% |
| 65-79 (buy) | 125 | 3.9% |
| 40-64 (neutral) | 1,146 | 35.7% |
| <40 (weak/avoid) | 1,914 | 59.6% |

---

## Data Quality

| Feature | Fill Rate | Assessment |
|---|---|---|
| sector | 97.6% | OK |
| market_cap_bucket | 97.6% | OK |
| momentum_1m | 96.2% | OK |
| vix_at_signal | 100% | OK |
| disclosure_delay | 95.0% | OK |
| person_hit_rate_30d | 73.6% | New traders have no history |
| insider_role | 29.7% | Congress has no roles (expected); 51 EDGAR missing |

### Source Quality
| Source | Signals | Avg CAR | Hit Rate | Assessment |
|---|---|---|---|---|
| EDGAR (buys) | 1,005 | **+3.71%** | 47.9% | Real alpha source |
| Congress | 2,207 | -0.36% | 45.0% | Slightly negative avg |

---

## Convergence
| Tier | Count | Pct |
|---|---|---|
| Tier 0 (none) | 844 | 26.3% |
| Tier 1 | 1 | 0.03% |
| Tier 2 | 2,367 | 73.7% |

Tier 2 dominates because many congress signals cluster on the same tickers in the same sector/window.

---

## Known Issues
See `docs/todo.md` P0 for current issues and priorities.
