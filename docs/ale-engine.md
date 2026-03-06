# ATLAS — Adaptive Learning Engine (ALE)
*Technical reference. Update when ALE schema or pipeline changes.*

---

## CLI

| Command | Purpose |
|---|---|
| `--daily` | Ingest + enrich + score + export + diagnostics |
| `--analyze` | Feature analysis + ML walk-forward + weight update + score + export |
| `--score` | Score all signals with ML + export brain data |
| `--summary` | Console status + person leaderboards |
| `--diagnostics` | Generate HTML dashboard + markdown analysis report (standalone) |
| `--export` | Export brain_signals.json + brain_stats.json only |
| `--report` | Analyst report: data inventory, model KPIs, feature health, signal quality → console + analyst_report.json |
| `--self-check` | IC trend, feature drift, model health → brain_health.json |
| `--backfill` | Re-enrich v5/v6/v7 features for all signals |
| `--eval-features COL [COL...]` | Evaluate candidate feature columns against baseline IC → feature_candidates.json |
| `--rollback [NAME]` | List checkpoints + restore (most recent if no name). Auto-triggered on IC regression >10% |
| `--hypotheses` | Generate signal hypotheses from model residuals + feature interactions → signal_hypotheses.json |
| `--edgar-days N` | EDGAR lookback days for --bootstrap (default 2555 = ~7 years) |
| `--incremental` | With --bootstrap: only fetch signals older than existing data (90d overlap) |

**One-time scripts:**
| Script | Purpose |
|---|---|
| `bootstrap_historical.py` | Populate DB with historical data (default ~30 months EDGAR) |
| `backfill_edgar_xml.py` | Parse XML for EDGAR signals, delete non-purchases, enrich buys |

---

## Database

- **File:** `data/atlas_signals.db` (SQLite)
- **Tables:** `signals`, `feature_stats`, `weight_history`
- **Diagnostics:** `data/ale_diagnostics.html` + `data/ale_analysis_report.md`
- **Weights:** `data/optimal_weights.json`

---

## ML Engine

- **Models:** RF + LightGBM ensemble (classification + regression)
- **Validation:** Walk-forward, 6mo min train, 1mo test windows, 200/20 min samples
- **Full-sample training:** For scoring — same hyperparameters, all data with outcomes
- **CAR:** BHAR `(1+stock)/(1+spy)-1`, winsorized 1st/99th percentile, hard bounds [-100%, +300%]
- **Safety:** Weights only auto-update when OOS IC improves >5%
- **Checkpoints:** `data/checkpoints/{timestamp}/` — auto-created before --analyze. IC regression >10% triggers auto-rollback. Manual: `--rollback [NAME]`

---

## Feature List (32) — v10

| Category | Features |
|---|---|
| Trade | trade_size_points, disclosure_delay |
| Clustering | same_ticker_signals_7d/30d |
| Person | person_trade_count, person_hit_rate_30d, person_avg_car_30d |
| Classification | insider_role, sector, market_cap_bucket, **market_regime** |
| Price-based | price_proximity_52wk, momentum_1m/3m/6m, volume_spike, volume_dry_up |
| Market context | vix_at_signal, yield_curve_at_signal, credit_spread_at_signal |
| Catalysts | days_to_earnings, earnings_surprise |
| Analyst | analyst_consensus |
| Committee | committee_overlap |
| Sentiment | news_sentiment_30d |
| Interactions | sect_ticker_momentum, volume_cluster_signal |
| Short Interest | short_interest_pct, short_interest_change, short_squeeze_signal |
| Institutional | institutional_holders, institutional_pct_held, institutional_insider_confluence |
| Options | options_bullish, options_unusual_calls, options_insider_confluence, options_bearish_divergence |
| Derived (DB) | insider_buy_ratio_90d, sector_avg_car, vix_regime_interaction, sector_momentum, days_since_last_buy |

**v10 changes:** Added 7 CANDIDATE features: `institutional_holders`, `institutional_pct_held`, `institutional_insider_confluence` (yfinance, block #19), `options_bullish`, `options_unusual_calls`, `options_insider_confluence`, `options_bearish_divergence` (yfinance options chains, block #20). `fetch_institutional_data()` + `fetch_options_data()` in fetch_data.py. Both run daily.
**v9 changes:** Added `short_interest_pct`, `short_interest_change`, `short_squeeze_signal` (yfinance, CANDIDATE). `fetch_short_interest()` in fetch_data.py. Enrichment block #18.
**v8 changes:** Pruned `analyst_insider_confluence` + `analyst_revision_30d` (0% importance all runs). Added `sect_ticker_momentum`, `volume_cluster_signal` (hypothesis-driven interactions). Added `spy_return_30d`, `market_adj_car_30d` (benchmark columns, not ML features). Fill-rate gate auto-promotes candidates. Net: 34→32 active + 6 candidates.
**v7 changes:** Added `market_regime` (categorical). Regime-conditional capping. 5 sentiment detail features + 3 lobbying features as candidates.
**v6 changes:** Added `committee_overlap`, `earnings_surprise`, `news_sentiment_30d`. Net: 27→30.
**v5 changes:** Pruned 5 dead-weight features. Added volume_dry_up, analyst_revision_30d, analyst_consensus, analyst_insider_confluence. Net: 28→27.

### Pruned Features
| Feature | Pruned Date | Reason | Runs at <1% |
|---|---|---|---|
| source | 2026-02-28 | 0.16% importance | 3+ |
| trade_pattern | 2026-02-28 | 0.40% importance, 31% fill | 3+ |
| convergence_tier | 2026-03-01 | <1% importance | 3+ |
| has_convergence | 2026-03-01 | <1% importance | 3+ |
| days_to_catalyst | 2026-03-01 | <1% importance | 3+ |
| relative_position_size | 2026-03-01 | <1% importance | 3+ |
| cluster_velocity | 2026-03-01 | <1% importance | 3+ |
| analyst_insider_confluence | 2026-03-02 | 0% importance all runs | 5+ |
| analyst_revision_30d | 2026-03-02 | 0% importance all runs | 5+ |

### Feature Importance (Session 8 — first clean run, 74 folds)
| Feature | Importance | Status |
|---|---|---|
| days_to_earnings | 14.0% | CORE — protect |
| sector_avg_car | 13.6% | CORE — audit passed (Session 9: look-ahead fixed) |
| insider_buy_ratio_90d | 5.7% | CORE — protect |
| person_avg_car_30d | ~5% | CORE — look-ahead fixed Session 9 |
| vix_at_signal | ~4% | Active |
| momentum_1m | ~3% | Active |
| (remaining 20) | <3% each | Active |

### Monitor-Prune (Session 9)
Session 8 = first clean-data run. Do NOT prune yet — need 2 more Monday runs.
| Feature | Clean Importance | Theory Basis | Decision |
|---|---|---|---|
| volume_dry_up | 0.2% | Weak | Prune if <1% after 2 more runs |
| analyst_consensus | 0.6% | Moderate | Prune if <1% after 2 more runs |

### Monitor-Extended (strong theory, longer runway)
| Feature | Clean Importance | Theory Basis | Decision |
|---|---|---|---|
| committee_overlap | 0.3% | Strong (congress committee → sector) | Monitor 5+ runs |
| market_regime | 0.3% | Sound (VIX-based regime) | Monitor 5+ runs |
| earnings_surprise | 0.3% | Academic (post-earnings drift) | Monitor 5+ runs |
| sect_ticker_momentum | 0.6% | Hypothesis engine | Monitor 5+ runs |

### Feature Integrity (Session 9)
- **sector_avg_car:** Look-ahead bias FOUND and FIXED. Was `signal_date < current`, now `signal_date < current - 45 days` to ensure 30d outcome was knowable. **IC may drop after --backfill + --analyze.**
- **person_avg_car_30d/90d:** Same look-ahead fix. Added 45-day (30d) and 135-day (90d) buffers.
- **same_ticker_signals_7d:** Clean (counts only, no outcome data).
- **sector_momentum:** Clean (uses momentum_1m, not outcomes).
- **brain_stats score tiers:** Labeled as in-sample. OOS score tiers computed during --analyze.

---

## Data Quality

**EDGAR signals must be filtered to purchases only.** EFTS doesn't expose transaction type — Form 4 XML must be parsed to determine direction (P=purchase, S=sale, M=exercise, A=grant).

- **Bootstrap:** Enriches via XML before insertion, filters to buys only
- **Backfill:** `backfill_edgar_xml.py` cleans existing DB signals
- **Daily pipeline:** `fetch_data.py` enriches via `enrich_form4_xml()`

---

## Bootstrap Pipeline

1. EDGAR historical (21mo Form 4 filings) + XML enrichment + buy filtering
2. FMP congress (senate + house)
2b. Sector + market cap map (FMP profile API)
3. Aggregate features
4. Price collection (incremental — skip/backfill/forward-fill)
5. Backfill CARs (BHAR, 5 horizons)
6. Person track records
7. Feature enrichment (52wk, momentum, volume, insider ratio, sector CAR, VIX interaction)
7b. Sector + market_cap_bucket NULL backfill
8. Market context (FRED: VIX, T10Y2Y, credit OAS)
9. Feature stats + weight generation
10. ML classification (walk-forward)
11. ML regression (walk-forward)
→ Dashboard + diagnostics + data quality report
