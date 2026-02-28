# ATLAS — Signal Scoring Engine
*Canonical reference. Update this file whenever scoring logic changes.*

---

## Hub 1 — Congressional Score

Source: `data/congress_feed.json` + `data/fmp_congress_feed.json` (refreshed 4x daily)
Only **purchase** transactions score. Sales/exchanges display but score 0.

| Factor | Points |
|---|---|
| Purchase filed (base) | +3 |
| Trade size $15K–$50K | +5 |
| Trade size $50K–$100K | +6 |
| Trade size $100K–$250K | +8 |
| Trade size $250K–$500K | +10 |
| Trade size $500K–$1M | +12 |
| Trade size >$1M | +15 |
| Cluster: 3+ members, same ticker, 30d window | +15 |
| Committee with relevant jurisdiction | +10 *(not yet computed)* |
| Same-day as tracked bill activity | +10 *(not yet computed)* |

**Signal decay:** `effectivePoints = rawPoints × 0.5^(daysSince / halfLife)` — `halfLife` defaults to 21 days (configurable via `data/optimal_weights.json`).
**Rolling window:** 30 days. **Cap per hub:** 40 pts.

---

## Hub 2 — Insider Score

Source: `data/edgar_feed.json` (SEC EDGAR Form 4, refreshed 4x daily via XML enrichment)

| Factor | Points |
|---|---|
| Any Form 4 buy match, 14d window | +6 per filing |
| Cluster: 2+ filings, same company, 14d | +10 |
| Cluster: 3+ filings, same company, 14d | +15 |
| CEO | +10 |
| CFO | +8 |
| Director | +6 |
| VP / Officer | +4 |
| No 10b5-1 plan | +8 |
| Near 52-week low | +5 *(not yet computed)* |
| Historical accuracy of this insider | +0–10 |

**Rolling window:** 14 days. **Cap per hub:** 40 pts.
**EDGAR matching:** SEC `company_tickers.json` (137/139 match rate). Ticker extracted directly from XML.
**Direction filter:** buy only — sells are noise for signal generation.

---

## Hub 3 — Institutional Score

**Status: Not yet live.** Demo data displayed; pipeline not built.

| Factor | Points |
|---|---|
| Known smart-money manager initiates | +15 |
| New position (not add-to) | +10 |
| Conviction increase >1% of portfolio | +8 |
| Unusual options sweep (volume vs OI) | +12 |

**Managers tracked:** Berkshire, Druckenmiller, Tepper, Ackman, Third Point, Pershing Square.
**Cap per hub:** 40 pts.

---

## Convergence Boosts

Applied once per ticker when multiple hubs fire within their respective windows.

| Hubs Active | Boost |
|---|---|
| Congressional + Insider | +20 |
| Congressional + Institutional | +20 |
| Insider + Institutional | +15 |
| All three | +40 |
| Any convergence + active tracked legislation | +15 additional |

**Convergence tiers (ALE v2):**
- Tier 0: no convergence
- Tier 1: same ticker, 2+ sources, 60d window
- Tier 2: sector-level — 3+ signals, 2+ sources, same sector, 30d

**Active legislation check:** ticker must appear in `BILLS[n].impactTickers` and vote date within 60 days.

---

## Total Score

```
total = hubCongress + hubInsider + hubInstitutional + convergenceBoost
```

Theoretical max: 40 + 40 + 40 + 55 = 175
Practical max (2 hubs live): 40 + 40 + 20 + 15 = 115

| Score | Tier | Display |
|---|---|---|
| < 40 | Below threshold | Hidden |
| 40–64 | Monitoring | Watchlist cards |
| ≥ 65 | Trade idea | Trade Ideas tier |
| ≥ 95 | Exceptional | Highlighted card |

**Threshold is dynamic:** loaded from `data/optimal_weights.json` at startup (`window.SCORE_THRESHOLD`). Default 65. Backtest engine auto-tunes.

---

## Entry Zone Logic

| Bound | Formula |
|---|---|
| Entry Lo | `signal_price × 0.97` |
| Entry Hi | `signal_price × 1.03` |

**Zone statuses:** `in_zone` / `above_zone` (3–10% above hi) / `missed` (>10%) / `stale` (>30 days old)

---

## Signal Decay

```
decayFactor = 0.5 ^ (daysSince / halfLife)
effectivePoints = rawPoints × decayFactor
```

| Days Old | Multiplier |
|---|---|
| 0 | 1.00 |
| 21 | 0.50 |
| 42 | 0.25 |

`halfLife` dynamically overridable via `SCORE_WEIGHTS.decay_half_life_days`.

---

## Roadmap

- [ ] Committee jurisdiction scoring (requires congressional committee mapping)
- [ ] Proximity to 52-week low scoring
- [ ] Historical accuracy per insider (multi-filing tracking)
- [ ] Institutional 13F pipeline (Phase 4)
- [ ] Sector-level convergence scoring in frontend
- [x] Signal decay (21-day half-life, exponential)
- [x] Dynamic threshold from backtest engine
- [x] Watchlist tier (40–64) + Trade Ideas tier (≥65)
- [x] Open universe scoring (all tickers in feeds)
- [x] EDGAR XML enrichment (role, direction, buy_value)
- [x] ALE v2 multi-tier convergence (Tier 0/1/2)
