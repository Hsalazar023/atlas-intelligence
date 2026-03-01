# ATLAS Brain — Architecture & Integration
*The Brain = ALE + ML engine. Powers all scoring, signals, and self-improvement.*

---

## What the Brain Is

Unified intelligence layer: `learning_engine.py` + `ml_engine.py` + their outputs. Ingests raw data, learns what predicts alpha, scores every signal, and exports to frontend. No hardcoded trade ideas.

---

## Pipeline (Built & Live)

```
--daily pipeline:
  1. Ingest (EDGAR + Congress)     →  atlas_signals.db
  2. Aggregate features            →  clustering, convergence
  3. Backfill outcomes (BHAR)      →  car_5d through car_365d
  4. Person track records           →  hit_rate, trade_count
  5. Feature enrichment             →  28 features filled
  6. Market context (FRED)          →  VIX, yield curve, credit spread
  7. Dashboard + diagnostics        →  ale_diagnostics.html
  8. Score all signals (ML)         →  total_score 0-100
  9. Export brain data              →  brain_signals.json + brain_stats.json

--analyze pipeline:
  1-2. Pre-analysis cleanup
  3.   Feature stats
  4.   Walk-forward classification (RF + LightGBM)
  4b.  Walk-forward regression
  5.   Weight update (5% IC gate)
  6.   Dashboard + diagnostics
  7.   Score all signals (ML)
  8.   Export brain data
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

## Next Capabilities (Planned)

| Capability | Purpose |
|---|---|
| Feature auto-pruning | Drop <1% importance features after 3 runs |
| IC stagnation detection | Alert when IC trend is declining |
| Score diversification | Max N signals per ticker in export |
| Self-check report | `--self-check` CLI for model health |
| New data sources | 13F, Congress.gov API, news sentiment |
| Automated deploy | GitHub Action: daily → score → commit → push |
