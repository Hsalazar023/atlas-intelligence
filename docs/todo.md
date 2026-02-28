# ATLAS — Active Todo
*Short-term focus. Move completed items to docs/archive/completed-milestones.md.*

---

## Now — EDGAR Data Cleanup (Blocking Everything)
> **Critical discovery:** ~96% of 10,563 EDGAR signals are NOT purchases — they're grants, exercises, sales, and other Form 4 types. Bootstrap ingested all Form 4 filings because EFTS doesn't expose transaction type. Only ~4% are genuine insider buys.

### P0 — In Progress
- [ ] **EDGAR XML backfill running** — `backfill_edgar_xml.py` parsing every EDGAR signal's XML, deleting non-purchases, enriching real buys with role/value/delay
- [ ] **Re-run `--daily` then `--analyze`** after backfill completes — expect dramatic IC improvement from clean data
- [ ] **Bootstrap fixed** — `_parse_efts_hits()` now captures accession + xml_url, enriches via XML before insertion

### P0 — Already Fixed
- [x] CAR winsorization — hard clip [-100%, +300%] applied in `backfill_outcomes()`
- [x] `market_cap_bucket` — FMP key fixed (`marketCap` not `mktCap`), now 99.4% fill
- [x] `sector_avg_car` — batch computation, now 99.4% fill
- [x] `insider_role` — cross-signal propagation added; XML backfill will fill the rest
- [x] ML min samples raised — 200 train / 20 test (was 30/5)
- [x] Feature importance — now averaged across all folds (was last fold only)
- [x] Removed `urgent_filing` derived feature (0% importance)

### P1 — After Clean Data
- [ ] Re-run `--analyze`, compare OOS IC (pre-cleanup baseline: -0.02)
- [ ] Recompute convergence tiers on clean data — Tier 2 may trigger now
- [ ] Recompute person track records on clean data
- [ ] Investigate `cluster_velocity=fast` — -5.15% CAR, actively harmful
- [ ] Review `earnings_proximity=no_catalyst` — -2.97% CAR

---

## Next — Build Brain Export Pipeline
> After data quality fixes land and IC improves.

- [ ] Add `--export` command to `learning_engine.py`
- [ ] Generate `data/brain_signals.json` — top ML-scored signals with entry/target/stop zones
- [ ] Generate `data/brain_stats.json` — sector CARs, score tier stats, committee correlations, heatmap data
- [ ] Frontend: replace `TRACKED` object with `brain_signals.json` loader
- [ ] Frontend: replace hardcoded stats sections with `brain_stats.json` loader

---

## Then — Replace Hardcoded Sections

**Brain can replace (after --export):**
| Section | Lines | Replace With |
|---|---|---|
| TRACKED object | 1076–1088 | `brain_signals.json` |
| Score tier returns | 941–945 | `brain_stats.score_tiers` |
| Signal Alpha box | 869 | `brain_stats.alpha` |
| Sector performance | 875–885 | `brain_stats.sectors` |
| Committee correlation | 635–640 | `brain_stats.committees` |
| Congressional heatmap | 664–679 | `brain_stats.congress_heatmap` |
| Ticker ribbon | 467–488 | Top 10 from `brain_signals.json` |
| Notable exits (DEMO) | 729–740 | EDGAR sales filter |
| Sector insider activity (DEMO) | 744–748 | Computed from EDGAR feed |
| Live alerts carousel | 493–527 | Brain-generated from top signals |
| KPI strip counts | 534–545 | Computed from data feeds |

**Needs new data pipeline (Phase 2+):**
| Section | Lines | Dependency |
|---|---|---|
| BILLS array | 1098–1104 | Congress.gov API |
| Institutional flows cards | 783–815 | 13F pipeline |
| Berkshire 13F tracker | 845–853 | 13F pipeline |
| Options flow table | 826–835 | CBOE/OPRA feed |
| Short interest table | 839–842 | SI data source |
| SMS previews | 1035–1041 | Notification backend |

---

## Backlog
- [ ] Add `FMP_API_KEY` as GitHub repo secret
- [ ] Wire convergence tier badges into frontend
- [ ] Build 13F filing parser (institutional flows)
- [ ] Congress.gov API for live bill tracking
