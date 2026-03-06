# Multi-Horizon ML Implementation — Progress Tracker

## Status: Phase 1 IN PROGRESS (code not yet written)

## What's Done
- Read all key files fully — have complete context
- Created task list with dependencies
- Identified all edit locations

## Key File Locations (line numbers for edits)

### `backtest/ml_engine.py` (464 lines total) — Phase 1
- `prepare_features()` line 70 — add `horizon='30d'` param
- `prepare_features_all()` line 117 — add `horizon='30d'` param
- `train_full_sample()` line 154 — add `horizon='30d'` param
- `walk_forward_train()` line 195 — add `horizon='30d'` param
- `walk_forward_regression()` line 333 — add `horizon='30d'` param
- NEW: `HorizonResult`, `MultiHorizonResult` dataclasses
- NEW: `train_multi_horizon()`, `train_full_sample_multi()`
- NEW: `calibrate_probabilities()` (isotonic regression)

### `backtest/learning_engine.py` — Phase 2 + 3
- `_migrate_columns()` line 188 — add per-horizon columns
- `score_all_signals()` line 3763 — multi-horizon blend scoring + improved formula
- `compute_smart_targets()` line 4178 — horizon-aware target scaling
- `export_brain_data()` line 4237 — add `horizon_scores` + `time_horizon` per signal
- `run_analyze()` line 4662 — replace single WF with `train_multi_horizon()`
- `run_self_check()` line 3903 — add IC decay monitoring
- `generate_analysis_report()` line 2269 — per-horizon IC table
- `generate_diagnostics_html()` line 2820 — per-horizon visualization

### `backtest/tests/test_ml_engine.py` (141 lines) — Phase 1
- NEW: `db_with_multi_outcomes` fixture (needs 30d/90d/180d CARs)
- NEW: `test_prepare_features_90d()`
- NEW: `test_backward_compat()`
- NEW: `test_horizon_weights_sum_to_one()`
- NEW: `test_negative_ic_zero_weight()`

### `atlas-intelligence.html` — Phase 4
- `loadBrainSignals()` ~line 3370 — map `horizon_scores` into TRACKED
- Card rendering — dynamic `data-horizon` from `best_horizon`
- `renderMLAnalysis()` ~line 2687 — multi-horizon panel
- `computeLiveTargets()` ~line 2710 — horizon-scaled zones

### Docs — Phase 5
- `docs/scoring-logic.md` — new composite formula
- `docs/ale-engine.md` — multi-horizon ML section
- `docs/brain.md` — multi-horizon architecture
- `docs/research-principles.md` — NEW file
- `docs/roadmap.md` — mark Phase 2 items complete
- `docs/todo.md` — update priorities

## Implementation Order
```
Phase 1 (ML Engine)     ← NEXT: write ml_engine.py changes
    ↓
Phase 2 (Scoring)       ← needs Phase 1 functions
    ↓
Phase 3 (Walk-Forward)  ← needs Phase 1 + 2
    ↓
Phase 4 (Frontend)      ← needs Phase 2 export changes
    ↓
Phase 5 (Docs)          ← needs all phases
```

## New Scoring Formula (Phase 2)
```
base      = calibrated_clf_prob × 55    (0-55)
magnitude = clamp(composite_reg × 150, -15, 20)
converge  = tier × 5                    (0/5/10)
person    = clamp(hr × 10, 0, 8)        (0-8)
freshness = clamp((1 - age_days/90) × 2, 0, 2)  (0-2)
total     = clamp(sum, 0, 100)
```

## Multi-Horizon Constants
```python
HORIZONS = ['30d', '90d', '180d']
MIN_SIGNALS_PER_HORIZON = 220
```

## IC-Weighted Blending
```
weight_h = max(0, IC_h) / sum(max(0, IC_h))
Fallback: uniform weights if all ICs ≤ 0
```

## optimal_weights.json New Keys
```json
{
  "_horizon_weights": {"30d": 0.55, "90d": 0.30, "180d": 0.15},
  "_per_horizon_ic": {"30d": {"clf_ic": 0.072, "hit_rate": 0.553}, ...},
  "_composite_ic": 0.089
}
```
