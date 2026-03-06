# ATLAS Brain — Architecture & Integration
*The Brain = ALE + ML engine. Powers all scoring, signals, and self-improvement.*

---

## What the Brain Is

Unified intelligence layer: `learning_engine.py` + `ml_engine.py` + their outputs. Ingests raw data, learns what predicts alpha, scores every signal, and exports to frontend. No hardcoded trade ideas.

---

## Pipeline (Built & Live)

### Automated Loop (GitHub Actions)

```
fetch-data.yml (4x daily Mon-Fri):
  fetch_data.py → data/*.json → git push

backtest.yml (daily Mon-Fri 5 PM ET):
  1. fetch_data.py            → refresh all data feeds
  2. --backfill               → enrich new signals with v5/v6 features
  3. --daily                  → ingest + score + export
  4. --self-check             → brain_health.json
  5. --summary                → print status
  6. git push                 → deploy to Vercel
  7. ntfy.sh                  → push notification (health + feature recs)

  Monday-only additions (after step 3):
  3a. --analyze               → full walk-forward ML retrain + weight update
  3b. --eval-features         → evaluate candidate features against baseline IC
      → feature_candidates.json (recommendations: ADD/NEUTRAL/SKIP)
      → ntfy includes feature promotion recommendations
```

### CLI Pipelines

```
--daily:  ingest → aggregate → BHAR → track records → enrich → FRED
          → diagnostics → ML score → export
--analyze: cleanup → feature stats → walk-forward clf + reg
          → weight update (5% IC gate) → diagnostics → score → export
--eval-features COL [COL...]: baseline IC → test each candidate
          → IC delta + recommendation → feature_candidates.json
```

**Scoring formula** → see `docs/scoring-logic.md`
**CLI commands** → see `docs/ale-engine.md`
**ML engine details** → see `docs/ale-engine.md`

---

## Export Files

| File | Contents | Frontend Consumer |
|---|---|---|
| `brain_signals.json` | Top 50 signals with entry/target/stop, scores, metadata | TRACKED object, ticker ribbon, trade ideas |
| `brain_stats.json` | Alpha, score tiers, sectors, committees, heatmap, KPIs | KPI strip, sector charts, score tier table |
| `optimal_weights.json` | Scoring thresholds, decay params | SCORE_THRESHOLD, decay settings |

---

## Data Integrity (Session 7 Audit)

### Score Tier Metrics
brain_stats.json score tiers (90+: 100% hit, 80-89: 99% hit) are **in-sample** — model trained on all outcomes, then scored the same signals. Not predictive performance.

**Real metrics (walk-forward OOS):** IC=0.0802, hit=55.6%, 22 folds

### Look-Ahead Bias Found & Fixed
- **sector_avg_car (2.84% importance):** Used ALL outcomes across entire dataset regardless of date. Fixed to point-in-time: only uses outcomes from signals BEFORE the current signal's date. Requires `sector_avg_car = NULL` reset + re-enrich.
- **person_avg_car_30d:** CLEAN — processes chronologically, uses only prior trades
- **momentum_3m/1m/6m:** CLEAN — uses signal_date minus offset; ±5d tolerance on base price is minor
- **same_ticker_signals_7d/30d:** CLEAN — counts signals on or before signal_date

### OOS Validation (Session 10, Mar 3 2026)

**Walk-forward IC: 0.1120 [p=0.0003] — this is the TRUE out-of-sample metric.**

The `total_score` in the DB is computed by the full-sample model (trained on ALL signals). Testing `total_score` on any holdout period produces inflated ICs (0.57+) because the model already saw those outcomes. This is expected in-sample behavior, not OOS evidence.

Uncontaminated feature ICs (point-in-time, no ML):
| Feature | IC | Notes |
|---|---|---|
| insider_buy_ratio_90d | +0.077 | Strongest raw predictor |
| days_to_earnings | +0.036 | #1 by ML importance, moderate raw IC |
| convergence_tier | -0.067 | **Negative** — more convergence = worse returns |
| sector_avg_car | -0.035 | Slightly negative |

80+ signal characteristics:
- ALL 290 are EDGAR (zero congress). ALL have convergence_tier=2.
- Concentrated in micro/small-caps (high volatility).
- 93.4% hit rate is in-sample. True OOS hit rate likely 60-70%.
- Avg CAR driven by a few massive micro-cap winners (GLSI +181%, VHAI +96%).

---

## Historical Data Expansion

| Source | Current | Target | Bottleneck |
|---|---|---|---|
| EDGAR Form 4 | ~39mo | ~7yr (2019+) | None — available back to 1994 |
| FMP Congress | ~39mo | ~39mo | FMP API goes back ~2021-2022 at most (50 pages/chamber) |
| yfinance prices | ~39mo | ~7yr | None — 20+ years available |

### Path to Statistical Significance
Current: 39mo, 22 folds, IC=0.0802, p=0.171
After EDGAR expansion to 2019: ~85mo EDGAR, ~39mo congress → ~35-42 folds
Projected p≈0.03-0.05 at same IC (IC may improve with more data)

### Running the Expansion
```bash
python backtest/bootstrap_historical.py --incremental
```
- Fetches EDGAR 7yr back (2555 days), FMP 50 pages/chamber
- Dedup by UNIQUE(ticker, date, source, representative, insider_name)
- Estimated runtime: ~90 minutes (concurrent XML enrichment, 8 req/s)
- After bootstrap: `python backtest/learning_engine.py --analyze`

### Congress Data Bottleneck
FMP free tier doesn't reach 2019. Alternatives to investigate:
- Senate/House disclosure XML archives (efdsearch.senate.gov/search/report/annual/)
- Capitol Trades API (free tier, may have longer history)
- Quiver Quant API

---

## Autonomous Hypothesis Generation

The brain can identify its own blind spots and suggest new features to investigate.
Runs automatically in weekly `--analyze`. Also available via `--hypotheses`.

Three strategies:
1. **High-Residual Tickers:** Finds tickers where model consistently gets it wrong (high |actual - predicted| CAR). These point to missing signals — M&A activity, sector catalysts, or unusual patterns the model doesn't see yet.
2. **Feature Interactions:** Tests pairs of moderately predictive features to see if their combination is stronger. E.g., `committee_sector_match × sentiment_divergence` might have higher IC than either alone.
3. **Regime Performance Gaps:** Identifies market regimes where hit rate < 50%. In those regimes, finds which features are most predictive — suggests regime-specific signals.

Output: `data/signal_hypotheses.json`
Weekly ntfy notification includes hypothesis count.
Human reviews hypotheses and decides which to promote to features.

---

## Key Research Findings

### Days-to-Earnings: #1 Predictive Feature (Session 10, Mar 3 2026)
`days_to_earnings` is the most important feature (0.150 importance, #1 by LightGBM). The 8-30 day pre-earnings window shows the strongest forward returns:

| Window | n | Avg CAR | Hit Rate |
|---|---|---|---|
| 8-30d (near) | 613 | +3.28% | 54.6% |
| 31-90d (medium) | 2,109 | +2.03% | 49.7% |
| >90d (distant) | 2,256 | +1.52% | 51.0% |
| 0-7d (imminent) | 209 | +0.89% | 50.7% |
| Post-earnings | 1,991 | +0.20% | 44.8% |

This is the **earnings catalyst signal**: insiders buying 8-30 days before earnings anticipate positive surprises. Post-earnings trades show near-zero returns — information is already public. LightGBM learns this threshold automatically from the raw feature; no binary encoding needed.

---

## Next Capabilities (Planned)

| Capability | Purpose |
|---|---|
| Feature auto-pruning | Drop <1% importance features after 3 runs |
| IC stagnation detection | Alert when IC trend is declining |
| Score diversification | Max N signals per ticker in export |
