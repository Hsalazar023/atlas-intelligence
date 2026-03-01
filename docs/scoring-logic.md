# ATLAS — Signal Scoring

---

## Two Scoring Systems

### 1. Brain ML Score (Backend — `total_score` in DB)
The primary score. Trained on 2,921 signals with known outcomes. Written to DB, exported to frontend.

```
base      = clf_probability × 60          (0-60, ML confidence)
magnitude = clamp(reg_car × 200, -20, 25) (predicted return bonus/penalty)
converge  = convergence_tier × 5          (0/5/10 convergence bonus)
person    = clamp(person_hit_rate × 8, 0, 5)
total     = clamp(sum, 0, 100)
```

| Tier | Score | Count | Action |
|---|---|---|---|
| Strong Buy | 80-100 | 27 | Top signals, trade idea cards |
| Buy | 65-79 | 125 | Trade ideas tier |
| Neutral | 40-64 | 1,146 | Watchlist / monitoring |
| Weak | <40 | 1,914 | Hidden from frontend |

### 2. Frontend Heuristic Score (Fallback)
Computed client-side from live feed data when Brain scores aren't available. Used by `computeConvergenceScore()` — now overridden by Brain `totalScore` when present.

#### Hub 1 — Congressional (0-40 pts)
| Factor | Points |
|---|---|
| Purchase filed (base) | +3 |
| Trade size tiers ($15K–$1M+) | +5 to +15 |
| Cluster: 3+ members, same ticker, 30d | +15 |

#### Hub 2 — Insider (0-40 pts)
| Factor | Points |
|---|---|
| Form 4 buy match, 14d window | +6 per filing |
| Cluster: 2+ filings, same co, 14d | +10/+15 |
| CEO +10, CFO +8, Director +6 | Role bonus |
| No 10b5-1 plan | +8 |

#### Convergence Boosts
| Condition | Boost |
|---|---|
| Congressional + Insider | +20 |
| Any convergence + active legislation | +15 |

#### Signal Decay
```
effectivePoints = rawPoints × 0.5^(daysSince / halfLife)
```
Default halfLife: 21d (congress), 14d (insider).

---

## Entry Zone Logic

| Bound | Formula |
|---|---|
| Entry Lo | `signal_price × 0.96` |
| Entry Hi | `signal_price × 1.04` |
| Target 1 | `signal_price × 1.20` |
| Target 2 | `signal_price × 1.35` |
| Stop | `signal_price × 0.88` |

---

## Convergence Tiers (ALE)

| Tier | Condition |
|---|---|
| 0 | Single source only |
| 1 | Same ticker, 2+ sources, 60d window |
| 2 | 3+ signals, 2+ sources, same sector, 30d |
