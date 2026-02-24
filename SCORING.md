# ATLAS — Signal Scoring Engine
*Reference document. Version 0.1 — Feb 2026*

This is the canonical spec for how ATLAS scores signals and detects convergence.
Update this file whenever scoring logic changes in the codebase.

---

## Hub 1 — Congressional Score

Derived from QuiverQuant live endpoint. Only **purchase** transactions count.
Sales/exchanges score 0 for convergence purposes (they are displayed but not boosted).

| Factor | Points |
|---|---|
| Purchase filed (base) | +3 |
| Trade size $15K–$50K | +5 |
| Trade size $50K–$100K | +6 |
| Trade size $100K–$250K | +8 |
| Trade size $250K–$1M | +10 |
| Trade size >$1M | +15 |
| Cluster: 3+ members, same ticker, 30d window | +15 |
| Committee with relevant jurisdiction | +10 *(manual only — not yet computed)* |
| Same-day as tracked bill activity | +10 *(manual only — not yet computed)* |

**Rolling window:** 30 days from today.
**Cap per hub:** 40 pts (prevents outlier pile-ups from a single source).
**Data source:** `data/congress_feed.json` — refreshed by `scripts/fetch_data.py`.

---

## Hub 2 — Insider Score

Derived from SEC EDGAR Form 4 filings. Matched by company name keywords.
**Limitation (v0.1):** EDGAR EFTS does not return role (CEO/CFO) or transaction type in the search index — only filing metadata. Role-based scoring requires parsing the full XML (Phase 1 Python pipeline).

| Factor | Points |
|---|---|
| Any Form 4 match for tracked company, 14d window | +6 per filing |
| Cluster: 2+ filings, same company, 14d | +10 |
| Cluster: 3+ filings, same company, 14d | +15 |
| CEO (when role available) | +10 |
| CFO (when role available) | +8 |
| Director (when role available) | +6 |
| No 10b5-1 plan (when available) | +8 |
| Near 52-week low | +5 *(not yet computed)* |

**Rolling window:** 14 days from today.
**Cap per hub:** 40 pts.
**Data source:** `data/edgar_feed.json` — refreshed by `scripts/fetch_data.py`.

### Company → Ticker Keyword Map
Used to match EDGAR company names (which have no ticker field) to tracked tickers.

| Ticker | Match keywords (case-insensitive) |
|---|---|
| RTX | raytheon, rtx corp |
| NVDA | nvidia |
| OXY | occidental |
| TMDX | transmedics |
| FCX | freeport |
| PFE | pfizer |
| TSM | taiwan semiconductor |
| META | meta platforms |
| WFRD | weatherford |
| SMPL | simply good, atkins |
| ITA | *(ETF — no Form 4)* |

---

## Hub 3 — Institutional Score

**Status: Phase 2 — not yet live.** Currently uses fallback/demo data.
Will be computed from 13F filings once institutional data pipeline is built.

| Factor | Points |
|---|---|
| Known smart-money manager initiates position | +15 |
| New position (not add-to) | +10 |
| Conviction increase >1% of portfolio | +8 |
| Unusual options sweep (volume vs OI) | +12 |

**Managers tracked:** Berkshire, Druckenmiller, Tepper, Ackman, Third Point, Pershing Square.

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

**Active legislation check:** Ticker must appear in `BILLS[n].impactTickers` and the bill's vote date must be within 60 days.

---

## Total Score Calculation

```
total = hubCongress + hubInsider + hubInstitutional + convergenceBoost
```

| Score | Meaning |
|---|---|
| < 40 | Weak or single-source — do not surface as idea |
| 40–64 | Moderate — worth monitoring |
| 65–84 | Strong — eligible for watchlist |
| ≥ 85 | **Generate trade idea** |
| ≥ 95 | Exceptional — highest conviction |

---

## Entry Zone Logic

Generated at signal creation time from the price at the moment all hubs converge.

| Bound | Formula |
|---|---|
| Entry Lo | `signal_price × 0.97` |
| Entry Hi | `signal_price × 1.03` |
| Stop | Defined by thesis (below support, above resistance for shorts) |

**Zone statuses:** `in_zone` / `above_zone` (3–10% above hi) / `missed` (>10%) / `stale` (>30 days old)

---

## Signal Decay

| Condition | Action |
|---|---|
| `sigDate` older than 30 days | Show ⏰ stale warning in overlay |
| Price moved >10% above entry hi (long) | Zone status = MISSED |
| No new hub activity in 45 days | Flag signal for manual review |

---

## Roadmap: Scoring Improvements

- [ ] Parse full EDGAR XML to extract role (CEO/CFO) and transaction type
- [ ] Add 10b5-1 plan flag from Form 4 XML
- [ ] Proximity to 52-week low (needs Finnhub `stock/candle` data)
- [ ] Committee jurisdiction scoring (requires congressional committee mapping)
- [ ] Institutional 13F pipeline (Phase 2)
- [ ] Historical accuracy per insider (multi-filing tracking, Phase 3)
- [ ] Sector-level convergence (multiple tickers in same sector = sector signal)
