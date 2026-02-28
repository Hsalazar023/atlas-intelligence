# ATLAS Brain — Status & Health
*Updated after each bootstrap/analyze run. Archive old sections when stale.*

---

## Latest Run: Feb 28, 2026

### Critical Finding: EDGAR Data Contamination
**~96% of 10,563 EDGAR signals are NOT purchases.** They're grants (48%), exercises (10%), sales (14%), tax withholding, and other Form 4 transaction types. The bootstrap ingested all Form 4 filings because EFTS doesn't expose transaction type — XML must be parsed to determine direction.

**Impact:** Brain trained on ~10,100 noise signals. OOS IC went negative (-0.02). Insider features (role, trade size, disclosure delay) had near-zero importance because most signals weren't real insider buys.

**Fix:** `backfill_edgar_xml.py` running — parses XML for every EDGAR signal, deletes non-purchases, enriches real buys. Bootstrap also fixed to filter at insertion.

### Summary (Pre-Cleanup — will change after backfill)
| Metric | Value | Assessment |
|---|---|---|
| Total signals | 12,769 (2,206 congress + 10,563 EDGAR) | ⚠️ ~10K EDGAR signals are noise |
| OOS IC | -0.0209 | ❌ Negative — noise is the cause |
| Regression avg IC | 0.0889 | ⚠️ Moderate |
| RMSE | 0.1434 | Baseline |

### Fixed Issues
| Issue | Status |
|---|---|
| CAR winsorization (was +1,733% max) | ✅ Hard clip [-100%, +300%] |
| market_cap_bucket (was 0%) | ✅ 99.4% fill |
| sector_avg_car (was 0.8%) | ✅ 99.4% fill |
| ML min samples (was 30/5) | ✅ 200 train / 20 test |
| Feature importance (last fold only) | ✅ Averaged across all folds |
| urgent_filing (0% importance) | ✅ Removed |

---

## Scoring Performance (Pre-Cleanup)
*These numbers are contaminated by ~10K non-purchase EDGAR signals. Will recompute after backfill.*

| Horizon | Hit Rate | Avg CAR | Conv Hit | Conv CAR |
|---|---|---|---|---|
| 30d | 45.7% | +0.69% | 52.1% | +0.20% |
| 180d | 39.7% | +1.77% | 56.3% | +3.16% |
| 365d | 34.3% | -0.47% | 48.6% | +9.39% |

**Convergence is the edge** at longer horizons — but only 4.4% of signals have it.

---

## ML Feature Importance (Pre-Cleanup)

Top 5: momentum_3m (0.161), momentum_1m (0.135), volume_spike (0.116), momentum_6m (0.112), vix_at_signal (0.095)

*Momentum dominates because insider features have no signal when 96% of "insider" data is noise. Expect insider features to gain importance after cleanup.*

---

## Action Items (After Backfill Completes)

1. Run `--daily` then `--analyze` on clean data
2. Compare OOS IC (baseline: -0.02, target: >0.10)
3. Check if insider_role, trade_size_points, disclosure_delay gain ML importance
4. Recompute convergence tiers — may trigger Tier 2 now
5. Recompute person track records
6. Investigate cluster_velocity=fast (-5.15% CAR)
