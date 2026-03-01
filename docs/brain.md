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
  5. Feature enrichment             →  27 features filled
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

### Scoring Formula (0-100)
```
base      = clf_probability × 60          (ML confidence, 0-60)
magnitude = clamp(reg_car × 200, -20, 25) (predicted return bonus)
converge  = convergence_tier × 5          (0/5/10)
person    = clamp(person_hit_rate × 8, 0, 5)
total     = clamp(sum, 0, 100)
```

### Export Files
| File | Contents | Frontend Consumer |
|---|---|---|
| `brain_signals.json` | Top 50 signals with entry/target/stop, scores, metadata | TRACKED object, ticker ribbon, trade ideas |
| `brain_stats.json` | Alpha, score tiers, sectors, committees, heatmap, KPIs | KPI strip, sector charts, score tier table |
| `optimal_weights.json` | Scoring thresholds, decay params | SCORE_THRESHOLD, decay settings |

---

## CLI

| Command | Purpose | Status |
|---|---|---|
| `--daily` | Ingest + enrich + score + export | ✅ |
| `--analyze` | Feature analysis + ML + weight update + score + export | ✅ |
| `--score` | Score all signals + export (standalone) | ✅ |
| `--summary` | Console status + leaderboards | ✅ |
| `--diagnostics` | HTML dashboard + analysis report | ✅ |
| `--export` | Export brain_signals + brain_stats only | ✅ |
| `--self-check` | IC trend, feature drift, model health | 🔜 |

---

## ML Engine

- **Models:** RF + LightGBM ensemble
- **27 features** (see `docs/ale-engine.md` for full list)
- **Walk-forward:** 6mo min train, 1mo test windows, 200/20 min samples
- **Full-sample training:** For scoring — same hyperparameters, all data with outcomes
- **CAR:** BHAR, winsorized 1st/99th percentile, hard bounds [-100%, +300%]

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
