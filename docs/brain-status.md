# ATLAS Brain Status
_Last updated: Session 17 — March 5, 2026_

---

## Current Baseline (Session 10 — Clean)

### Model Performance
| Metric | Value |
|---|---|
| IC (walk-forward CLF, 30d) | 0.1092 |
| IC (walk-forward REG, 30d) | 0.0609 |
| p-value | 0.0001 |
| t-statistic | 4.22 |
| Walk-forward folds | 74 |
| Positive folds | 50/74 (67.6%) |
| Hit rate (OOS) | 55.0% |
| Sharpe (IC-based, annual) | 0.21 |
| BSS | -0.029 |

### Alpha Metrics (80+ signals)
| Metric | Value |
|---|---|
| Market-adjusted alpha | +15.85%/signal |
| Beta vs SPY | 0.828 |
| Hit rate (in-sample, 80+) | ~93% — IN-SAMPLE, not OOS |
| Expected OOS hit rate (80+) | ~60-65% |

### Score Distribution (all 7,566 signals)
| Bucket | Signals | Avg CAR | Hit Rate |
|---|---|---|---|
| 80+ | 487 | +24.78% | 93.4% (in-sample) |
| 60-79 | 1,021 | +11.04% | ~70% (in-sample) |
| 40-59 | 1,599 | +4.37% | ~55% (in-sample) |
| <40 | 4,459 | -2.90% | ~38% (in-sample) |

### OOS Score Tiers (walk-forward, authoritative)
| Tier | n | Hit Rate |
|---|---|---|
| High confidence (80+) | 638 | 65.8% |
| Moderate (60-79) | 1,152 | 51.7% |
| Low confidence (<60) | 4,879 | 45.8% |

---

## Data State

| Metric | Value |
|---|---|
| Total signals | 7,566 |
| EDGAR signals | 4,236 |
| Congress signals | 3,330 |
| Date range | 2019-01 → 2026-03 (86 months) |
| Outcome fill rate | 94.9% |
| Training signals (with outcome) | 7,178 |

### Coverage by Year
| Year | Signals | Outcome Fill |
|---|---|---|
| 2019 | 559 | 100% |
| 2020 | 593 | 100% |
| 2021 | 540 | 99.3% |
| 2022 | 943 | 98.2% |
| 2023 | 1,022 | 98.3% |
| 2024 | 955 | 98.1% |
| 2025 | 2,499 | 97.9% |
| 2026 | 455 | 38.5% (recent) |

---

## Feature State

### Active Features (29 = 23 base + 6 auto-promoted)
Ranked by importance (Session 10):

| Rank | Feature | Importance | Status |
|---|---|---|---|
| 1 | days_to_earnings | 0.141 | CORE |
| 2 | sector_avg_car | 0.126 | CORE (look-ahead fixed) |
| 3 | insider_buy_ratio_90d | 0.056 | CORE |
| 4 | disclosure_delay | 0.050 | CORE |
| 5 | sector | 0.047 | CORE |
| 6 | momentum_3m | ~0.04 | Strong |
| 7 | momentum_1m | ~0.04 | Strong |
| 8 | volume_spike | ~0.03 | Good |
| 9 | price_proximity_52wk | ~0.03 | Good |
| 10 | momentum_6m | ~0.03 | Good |
| — | committee_overlap | 0.002 | PRUNE next (3rd flag) |
| — | earnings_surprise | 0.001 | PRUNE next (3rd flag) |
| — | volume_cluster_signal | 0.004 | Monitor |
| — | sect_ticker_momentum | 0.004 | Monitor |
| — | news_sentiment_30d | 0.005 | Monitor |
| — | market_regime | 0.005 | Monitor |

Other active: trade_size_points, same_ticker_signals_7d/30d, person_trade_count, person_hit_rate_30d, person_avg_car_30d, vix_at_signal, yield_curve_at_signal, credit_spread_at_signal, vix_regime_interaction, market_cap_bucket, sector_momentum, days_since_last_buy

### Permanently Excluded
analyst_insider_confluence, analyst_revision_30d, volume_dry_up, analyst_consensus, source, trade_pattern, convergence_tier, has_convergence, days_to_catalyst, relative_position_size, cluster_velocity

### Candidates (waiting for fill rate >= 60%)
short_interest_pct (7%), short_interest_change (7%), institutional_holders (4%), institutional_pct_held (4%), options_bullish (0%), options_unusual_calls (0%), options_insider_confluence (0%), options_bearish_divergence (0%)

---

## Model Configuration

### Scoring Coefficients (optimized Session 10)
base=80, mag=300, conv=3, person=12

### Source Quality Multipliers
edgar=1.000, congress=0.300, convergence=1.000

### Role Bonuses (learned)
| Role | CAR | Hit Rate | n | Bonus |
|---|---|---|---|---|
| CFO | 6.3% | 60% | 225 | 1.5x |
| COO | 5.1% | 61% | 64 | 1.5x |
| Officer | 7.7% | 60% | 125 | 1.5x |
| President | 4.4% | 64% | 139 | 1.49x |
| VP | 4.0% | 54% | 123 | 1.35x |
| CEO | 3.4% | 51% | 791 | 1.15x |
| Director | 2.5% | 54% | 1,683 | 0.85x |
| 10% Owner | 1.4% | 47% | 804 | 0.7x |

### Trader Tiers
11 elite | 27 good | 25 fade | 66 neutral (129 classified)

### Regime Guardrails (Session 14)
Based on honest OOS IC by VIX bucket:
| VIX Zone | OOS IC | Multiplier | Label |
|---|---|---|---|
| 15-25 | 0.095-0.131*** | 1.00x | OPTIMAL |
| 25-30 | 0.063 ns | 0.85x | ELEVATED |
| < 15 | ~0.04 ns | 0.75x | LOW_VOL |
| 30-40 | limited data | 0.90x | HIGH_VOL |
| > 40 | 2020-style | 0.60x | CRISIS |

Multiplier adjusts position sizing, not hard block.

### Kelly Position Sizing (Session 14)
Formula: 1/4 Kelly × (oos_score/100) × regime_multiplier
Parameters from OOS 75+ signals. Capped 2-15% per position.
Kelly params and per-signal `kelly_size` in brain_signals.json.

### Year-by-Year OOS IC (Session 14)
| Year | OOS IC | Significance | Note |
|---|---|---|---|
| 2019 | 0.1334 | * | Pre-COVID bull |
| 2020 | -0.0199 | ns | COVID crash — model failed |
| 2021 | 0.1468 | *** | Recovery — best year |
| 2022 | 0.0992 | ** | Bear market — GENUINE ALPHA |
| 2023 | 0.1161 | *** | |
| 2024 | 0.0399 | ns | Low-vol — WEAK, investigating |
| 2025 | 0.0902 | *** | |

### Market Regime Performance (in-sample, reference only)
| Regime | n | Hit Rate | Avg CAR |
|---|---|---|---|
| Crisis (VIX>35) | 12 | 58.3% | +3.86% |
| Elevated (VIX 25-35) | 599 | 50.8% | +1.47% |
| Normal (VIX 15-25) | 5,318 | 50.7% | +1.85% |
| Low vol (VIX<15) | 1,249 | 41.9% | -0.36% |

---

## Key Research Findings

### Days-to-Earnings: #1 Feature (Session 10)
8-30 days pre-earnings: +3.28% avg CAR, 54.6% hit rate
Post-earnings: +0.20% avg CAR — 16x difference
Earnings catalyst pattern: insiders buy ahead of positive surprises.
LightGBM learns the threshold from raw feature; binary encoding not needed.

### OOS Validation (Session 10)
- Walk-forward IC (0.1092) is the ONLY valid OOS metric
- total_score IC on holdout (0.57) is invalid — full-sample model contamination
- 93.4% hit rate is IN-SAMPLE; true OOS ~60-65%
- ALL 290 80+ signals are EDGAR with convergence_tier=2, concentrated in micro-caps
- Convergence_tier raw IC is negative (-0.067) — more convergence = worse returns

### Uncontaminated Feature ICs (point-in-time, no ML)
| Feature | IC |
|---|---|
| insider_buy_ratio_90d | +0.077 |
| days_to_earnings | +0.036 |
| convergence_tier | -0.067 |
| sector_avg_car | -0.035 |

### Look-Ahead Bias Audit (Session 9)
sector_avg_car: fixed to 45-day buffer. person_avg_car: same fix. IC improved after fix (0.0955 → 0.1120 → 0.1092 post-pruning).

### Congressional Signals
Average CAR: -0.57% (negative). Source quality: 0.300. FMP /stable/ endpoints working (v3/v4 deprecated).
Congress feed staleness: FMP data ends 2026-02-13. House/Senate scrapers extract 0 trades (PDFs encrypted, Cloudflare blocks).

### Signal Decay Analysis (Session 12, IN-SAMPLE)
| Horizon | IC (all) | IC (80+) | Avg CAR (80+) | Hit (80+) |
|---|---|---|---|---|
| 5d | 0.2355 | 0.0507 | +6.30% | 72.3% |
| 30d | 0.4016 | 0.2231 | +19.76% | 90.9% |
| 90d | 0.2686 | 0.0933 | +23.31% | 68.1% |
| 180d | 0.1992 | 0.0921 | +27.12% | 69.7% |
| 365d | 0.1906 | 0.1729 | +35.74% | 67.2% |

Optimal horizon: 30d (confirmed). Model trained on 30d so in-sample IC peaks there (circular), but 5d weakness shows information takes time to price in. Alpha persists to 365d for top signals.

### Regime Robustness (Session 12, IN-SAMPLE — SUPERSEDED)
**These numbers used total_score (in-sample). See Session 13 OOS results below.**
| Regime | IC (in-sample) | Status |
|---|---|---|
| Low vol (VIX<15) | 0.3619 | inflated |
| Normal (VIX 15-25) | 0.4567 | inflated |
| Elevated (VIX 25-35) | 0.3611 | inflated |

### Regime Robustness (Session 15, OOS — AUTHORITATIVE)
| VIX Zone | OOS IC | Significance | Multiplier |
|---|---|---|---|
| 15-25 | 0.1045 | *** | 1.00x (OPTIMAL) |
| 25-35 | 0.0686 | ns | 0.85x (ELEVATED) |
| < 15 | 0.0412 | ns | 0.75x (LOW_VOL) |
| 30-40 | limited data | — | 0.90x (HIGH_VOL) |
| > 40 | 2020-style | — | 0.60x (CRISIS) |

2024 weakness explained: avg VIX=16.1. VIX 20-25 subset: IC=0.3140** (model works). VIX 15-20: IC=-0.04 ns (regime effect).

### Portfolio Simulation (Session 15, OOS — AUTHORITATIVE)
OOS 75+ signals, equal-weight, 30d hold:
| Metric | Value |
|---|---|
| Annualized return | +91.1% |
| Max drawdown | -43.0% (Feb 2020 COVID) |
| Calmar ratio | 2.12 |
| Sharpe | 1.53 |
| Win months | 36/50 (72%) |
| 2022 bear avg monthly | +9.01% |
| 2024 low-vol avg monthly | +21.77% |

Stop-loss: -12% from entry price. `stop_loss_price`, `stop_loss_triggered`, `position_status` in brain_signals.json.

### Factor Analysis (Sessions 15-16)
Fama-French 6-factor regression on OOS 75+ monthly returns (464 signals, 50 months aligned).

| Factor | Loading | t-stat | Sig |
|--------|---------|--------|-----|
| Alpha | +0.0714 | +3.27 | ** |
| Mkt-RF | -0.2295 | -0.51 | ns |
| SMB | +1.2376 | +1.36 | ns |
| HML | -0.6377 | -0.90 | ns |
| RMW | -0.4142 | -0.42 | ns |
| CMA | +0.9555 | +0.83 | ns |
| MOM | +0.0619 | +0.11 | ns |

Monthly alpha: +7.14%/mo. Annualized: +128.9%. R²: 0.082.
Verdict: **STRONG** — genuine insider edge beyond all known factors. Only 8% of returns explained by systematic risk factors.

### Transaction Cost Model (Session 13)
Spread estimates by market cap proxy (price × ADV × 252):
| Tier | Spread | Threshold |
|---|---|---|
| Large cap (>$10B) | 0.05% | |
| Mid cap ($1-10B) | 0.20% | |
| Small cap (<$1B) | 0.50% | HIGH_COST flag |
`net_expected_return = car_30d − 2 × estimated_spread` (round trip)

### Beta Context
Beta=0.828 is real long equity bias. Alpha (market-adjusted) is genuine. In sustained bear markets, strategy will underperform in absolute terms.

---

## Data Integrity Audit Log

| Date | Issue | Resolution | IC Impact |
|---|---|---|---|
| Session 7 | Historical expansion added signals with no outcomes | — | p-value worsened |
| Session 8 | Price cache only went back 365 days | Extended to 2920 days, backfilled 3,817 outcomes | Folds 22→74 |
| Session 9 | sector_avg_car had partial look-ahead | Fixed to 45-day buffer | IC 0.0955→0.1120 |
| Session 9 | person_avg_car_30d same issue | Fixed | Included above |
| Session 10 | person_hr/car reset to 0% by backfill | update_person_track_records() re-added to pipeline | Fixed |
| Session 10 | OOS hit rate 93.4% cited as real | Documented as in-sample | No IC change |

---

## Pipeline Health

| Component | Status |
|---|---|
| EDGAR Form 4 ingestion | OK |
| FMP congressional (/stable/) | OK (fixed Session 10) |
| Price cache | OK (2018→present) |
| Outcome backfill | OK (automated in --backfill) |
| Walk-forward training | OK (74 folds) |
| Checkpoint/rollback | OK (14 checkpoints retained) |
| GitHub Actions Monday pipeline | OK (verified Session 10) |
| Options flow | Fix applied, repopulates next run |
| Lobbying data | Source broken — deprioritized |

---

## IC Progression

| Session | IC | p-value | Folds | Key Change |
|---|---|---|---|---|
| 1 | 0.077 | ns | ~10 | Initial build |
| 2 | 0.086 | ns | ~12 | Feature additions |
| 5 | 0.0808 | ns | ~18 | Post-triage |
| 6 | 0.0824 | 0.171 | 22 | 34 features, 3212 signals |
| 7 | 0.0937 | 0.490 | 22 | Historical expansion (broken outcomes) |
| 8 | 0.0955 | 0.0003 | 74 | Outcome backfill fixed |
| 9 | 0.1120 | 0.0005 | 74 | Look-ahead fixed, >0.10 crossed |
| 10 | 0.1092 | 0.0001 | 74 | Pruned 4 features, t-stat improved |
| 12 | 0.1028 | 0.0006 | 75 | +2 features, liquidity model, OOS stored |
| 14 | 0.1000 | 0.0003 | 75 | Regime guardrails, Kelly sizing, dashboard scaffold |
| 15 | 0.1027 | 0.0003 | 75 | Phase 2 COMPLETE, dashboard built, FF6 added |

---

## Session Carry-Forwards (into Session 16)

### BLOCKER — Dashboard NaN Bug
`brain_signals.json` has Python `NaN` values → invalid JSON → dashboard can't load.
Fix in `shared.py` (`_sanitize_for_json`) applied but not yet verified.
**First step Session 16:** run `--export`, verify valid JSON, test dashboard.
See `docs/todo.md` for quick-fix regex script if --export still produces NaN.

### Immediate
- [x] ~~Fix NaN blocker~~ → dashboard loads (Session 16)
- [x] ~~Download FF5 factors~~ → +128.9% ann alpha (Session 16)
- [x] ~~Delete NONE ticker~~ → 23 records removed (Session 16)
- [x] ~~Congress FMP investigation~~ → upstream delay confirmed, closed (Session 16)

### Signal Intelligence (Session 17)
`compute_signal_intelligence()` profiles best/worst 50 signals by realized CAR:

| Metric | Best 50 | Worst 50 | Gap |
|--------|---------|----------|-----|
| Avg total_score | 72.3 | 28.4 | +43.9 |
| Avg oos_score | ~equal | ~equal | -3.7 |
| Avg days_to_earnings | far | near | -98.9 |

Key findings:
- **Score gap +43.9**: Model total_score strongly separates winners from losers
- **OOS gap -3.7**: OOS scores less discriminating — worth investigating
- **Earnings gap -98.9**: Best signals are far from earnings, worst are near (confirms days_to_earnings as #1 feature)
- Healthcare dominates both lists (largest sector, not a differentiator)

### Dashboard Bugs Fixed (Session 18)
- **win_probability**: Raw value 71.4 (already %) was multiplied by 100 → displayed as 7140%. Fixed.
- **factors field**: Array of objects rendered as [object Object]. Fixed with topFactors() parser.
- **portfolio stats**: Closed signal stats used biased top-50 export. Now from honest full-DB query via portfolio_stats.json.

### Phase 3 Status
- [x] ~~Signal history / closed positions~~ → closed_signals + portfolio_stats.json (Sessions 17-18)
- [x] ~~Portfolio summary~~ → active return card, honest closed stats, monthly log (Session 18)
- [x] ~~Dashboard enhancements~~ → 4-tab architecture, analyst report tab, IC trend chart (Session 18)
- [x] ~~Dashboard accuracy~~ → win_prob fix, factors fix, regime suppression (Session 18)
- [ ] Strategy summary document (investor memo)
- [ ] New signal notifications (80+ alerts via ntfy)
- [ ] Mobile layout polish
