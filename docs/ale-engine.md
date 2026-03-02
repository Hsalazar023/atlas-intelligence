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
| `--self-check` | IC trend, feature drift, model health (planned) |
| `--backfill` | Re-enrich v5/v6 features (volume, analyst, committee, earnings, sentiment) for all signals |
| `--eval-features COL [COL...]` | Evaluate candidate feature columns against baseline IC |
| `--edgar-days N` | EDGAR lookback days for --bootstrap (default 900 = ~30 months) |

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

---

## Feature List (30) — v6

| Category | Features |
|---|---|
| Trade | trade_size_points, disclosure_delay |
| Clustering | same_ticker_signals_7d/30d |
| Person | person_trade_count, person_hit_rate_30d, person_avg_car_30d |
| Classification | insider_role, sector, market_cap_bucket |
| Price-based | price_proximity_52wk, momentum_1m/3m/6m, volume_spike, volume_dry_up |
| Market context | vix_at_signal, yield_curve_at_signal, credit_spread_at_signal |
| Catalysts | days_to_earnings, earnings_surprise |
| Analyst | analyst_revision_30d, analyst_consensus, analyst_insider_confluence |
| Committee | committee_overlap |
| Sentiment | news_sentiment_30d |
| Derived (DB) | insider_buy_ratio_90d, sector_avg_car, vix_regime_interaction, sector_momentum, days_since_last_buy |

**v6 changes:** Added `committee_overlap` (congress oversight ↔ stock sector, GitHub data), `earnings_surprise` (EPS surprise %, yfinance), `news_sentiment_30d` (Finnhub news + VADER/keyword scoring). Net: 27→30 features.
**v5 changes:** Pruned 5 features (<1% importance, 3+ runs): `convergence_tier`, `has_convergence`, `days_to_catalyst`, `relative_position_size`, `cluster_velocity`. Added 4 new: `volume_dry_up`, `analyst_revision_30d`, `analyst_consensus`, `analyst_insider_confluence`. Added `_role_quality` bonuses + trader tiers + fade signal. insider_role kept — role VALUES (COO/CEO) are highly predictive. Net: 28→27 features.
**v4 changes:** Pruned `source` (0.16%), `trade_pattern` (0.40%, 31% fill). Added `person_avg_car_30d`, `sector_momentum`, `days_since_last_buy`.

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
