# ATLAS — Signal Scoring
*Single source of truth for all scoring formulas, tiers, and entry logic.*

---

## Brain ML Score (Primary — `total_score` in DB)

Trained on 2,921 signals with known outcomes. Written to DB, exported to frontend.

```
base      = clf_probability × 60          (0-60, ML confidence)
magnitude = clamp(reg_car × 200, -20, 25) (predicted return bonus/penalty)
converge  = convergence_tier × 5          (0/5/10 convergence bonus)
person    = clamp(person_hit_rate × 8, 0, 5)
total     = clamp(sum, 0, 100)
```

### Score Tiers

| Tier | Score | Action |
|---|---|---|
| Strong Buy | 80-100 | Top signals, trade idea cards |
| Buy | 65-79 | Trade ideas tier |
| Neutral | 40-64 | Watchlist / monitoring |
| Weak | <40 | Hidden from frontend |

---

## Source Quality Multiplier (Learned)

Applied as final multiplier on raw score. Values are computed each `--analyze` run from historical CAR by source.

```
raw       = base + magnitude + converge + person
source_mult = lookup by source type (stored in optimal_weights.json → _source_quality)
total     = clamp(raw × source_mult, 0, 100)
```

| Source | Default | Rationale |
|---|---|---|
| EDGAR (insider buy) | 1.0 | Baseline — highest avg CAR |
| Congress | ~0.65 | Ratio of congress_car / edgar_car, clamped 0.3–1.0 |
| Convergence (both) | ~1.35 | Bonus when EDGAR + Congress agree on ticker |

Multipliers auto-update via `_compute_source_quality()` in `run_analyze`. If Congress alpha improves, its multiplier rises automatically.

---

## Convergence Tiers

| Tier | Condition |
|---|---|
| 0 | Single source only |
| 1 | Same ticker, 2+ sources, 60d window |
| 2 | 3+ signals, 2+ sources, same sector, 30d |

---

## Frontend Heuristic Score (Fallback)

Computed client-side when Brain scores aren't available. Overridden by Brain `totalScore` when present.

### Hub 1 — Congressional (0-40 pts)

| Factor | Points |
|---|---|
| Purchase filed (base) | +3 |
| Trade size tiers ($15K–$1M+) | +5 to +15 |
| Cluster: 3+ members, same ticker, 30d | +15 |

### Hub 2 — Insider (0-40 pts)

| Factor | Points |
|---|---|
| Form 4 buy match, 14d window | +6 per filing |
| Cluster: 2+ filings, same co, 14d | +10/+15 |
| CEO +10, CFO +8, Director +6 | Role bonus |
| No 10b5-1 plan | +8 |

### Convergence Boosts

| Condition | Boost |
|---|---|
| Congressional + Insider | +20 |
| Any convergence + active legislation | +15 |

---

## Signal Decay

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
