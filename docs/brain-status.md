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
| Walk-forward folds | — | 22 | ✅ |

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
| sector | 97.6% | ✅ |
| market_cap_bucket | 97.6% | ✅ |
| momentum_1m | 96.2% | ✅ |
| vix_at_signal | 100% | ✅ |
| disclosure_delay | 95.0% | ✅ |
| person_hit_rate_30d | 73.6% | ⚠️ New traders have no history |
| trade_pattern | 31.3% | ❌ Needs 3yr history, most lack it |
| insider_role | 29.7% | ❌ Congress has no roles (expected); 51 EDGAR missing |

### Source Quality
| Source | Signals | Avg CAR | Hit Rate | Assessment |
|---|---|---|---|---|
| EDGAR (buys) | 1,005 | **+3.71%** | 47.9% | ✅ Real alpha source |
| Congress | 2,207 | -0.36% | 45.0% | ⚠️ Slightly negative avg |

**Key insight:** EDGAR insider buys produce 10x more alpha than congressional trades. Congress signals are most valuable as convergence confirmers, not standalone plays.

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
1. **Top signals dominated by BITB** — 8/10 top scores are one ticker. Needs diversification.
2. **ML weights not saving** — IC passed 0 but method still shows `feature_importance`.
3. **Congress negative alpha** — may need down-weighting in scoring formula.
4. **trade_pattern feature** — 31% fill rate, potentially adding noise to ML.
