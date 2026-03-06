# ATLAS Roadmap
_Last updated: Session 14 — March 4, 2026_

---

## Vision

Self-improving systematic alternative data engine for a single retail investor. Consistent, research-grade alpha through high-conviction insider signal detection.

**Target:** 15-25% annualized returns, 5-15 active positions, 30-90 day holds, IC consistently above 0.10.

**Comparable:** AQR systematic fundamental / Greenblatt early Gotham — high-conviction, low-frequency, alternative data driven. NOT high-frequency or institutional-scale.

---

## Phase 1 — Signal Research Foundation
### Status: COMPLETE

Proved the signal is real. Clean historical dataset. Statistical significance achieved.

| Milestone | Session |
|---|---|
| EDGAR Form 4 + congressional ingestion | 1-2 |
| LightGBM walk-forward model | 3-4 |
| Historical expansion to 2019 | 7 |
| Outcome backfill (94.9% fill) | 8 |
| Look-ahead bias audit + fix | 9 |
| IC > 0.10, p < 0.001, 74 folds | 9-10 |
| Automated Monday pipeline | 5 |

**Result:** IC=0.1092, p=0.0001, t=4.22, 74 folds. Signal is statistically real and clean.

---

## Phase 2 — Strategy Rigor
### Status: COMPLETE (Sessions 11-15)

Turn a research signal into a credible trading strategy. Understand when it works, when it doesn't, how much to bet, and what returns look like after realistic costs.

### Key Findings
| Metric | Value |
|---|---|
| IC (OOS, walk-forward) | 0.1027 [p=0.0003, SIGNIFICANT] |
| OOS Sharpe | 1.53 |
| OOS Calmar | 2.12 |
| OOS Max DD | -43.0% (Feb 2020 COVID tail) |
| OOS Ann. Return | +91.1% (equal-weight 75+) |
| 2022 Bear | +9.01% avg monthly (GENUINE ALPHA) |
| Kelly base | 11.2% per position (1/4 Kelly) |
| Regime | CONDITIONAL — VIX 15-25 optimal |
| Factor alpha | +128.9% ann (t=3.27**, R²=0.082) |

### 2A — Signal Decay Analysis ✅ (Session 12)
30d confirmed optimal. 5d weakest (IC=0.051). Alpha persists to 365d for top signals.

### 2B — Regime Robustness ✅ (Sessions 12-15)
CONDITIONAL pass (2/3 OOS VIX regimes). VIX 15-25: IC=0.1045***. VIX 25-35: IC=0.0686 ns. VIX <15: IC=0.0412 ns. 2024 weakness explained: avg VIX=16.1 (low-vol regime). 2022 bear IC=0.0992** (genuine alpha). Guardrails implemented.

### 2C — Drawdown Simulation ✅ (Session 15)
OOS 75+ equal-weight: Sharpe 1.53, Calmar 2.12, max DD -43.0% (Feb 2020), annualized +91.1%. Win months 36/50 (72%). 2022 bear: +9.01% avg monthly.

### 2D — Transaction Cost Modeling ✅ (Session 13)
Spread by market cap: large 0.05%, mid 0.20%, small 0.50%. `net_expected_return` and `liquidity_flag` in DB.

### 2E — Kelly Position Sizing ✅ (Session 14)
1/4 Kelly x confidence x regime multiplier. Capped 2-15%. OOS 75+ hit rate: 67.7%, avg win: +15.63%, avg loss: -11.02%.

### 2F — Factor Decomposition ✅ (Sessions 15-16)
Fama-French 6-factor regression added to `--report`. Monthly alpha +7.14%/mo (t=3.27**, significant at 1%). Annualized alpha +128.9%. R²=0.082 — only 8% of returns explained by factors. No significant factor loadings (Mkt-RF, SMB, HML, RMW, CMA, MOM all ns). Verdict: STRONG genuine insider edge beyond all known factors.

### 2G — Multiple Hypothesis Correction (Deferred)
30 features means ~1-2 significant by chance. Benjamini-Hochberg FDR at 5%. Deferred to Phase 4 — 28 active features is manageable.

---

## Phase 3 — Live Trading Tool
### Status: IN PROGRESS (Sessions 14+)

Turn research output into an actionable daily-use trading interface.

### 3A — Active Signals Dashboard ✅ (Session 15)
`data/dashboard.html` — full trading dashboard. Dark theme, sortable tables, responsive. Sections: regime banner (color-coded VIX), stats row (IC, hit rate, Kelly, folds), active positions (P&L, stop loss, Kelly bars), new signals (14d, entry zones, liquidity flags), closed positions (win/loss), model health panel.

### 3B — Signal History / Track Record (Session 15)
All closed signals with realized returns. Score vs return scatter plot. Monthly P&L chart. Running Sharpe.

### 3C — New Signal Feed (Session 15)
Daily notification when 80+ signal appears. Ticker, score, insider role, earnings timing.

### 3D — Model Health Monitor (Session 16)
IC trend chart, feature importance stability, data freshness indicators, alert on IC decline.

### 3E — Portfolio View (Session 16)
Manual position entry → auto-match to Atlas signal → real vs expected P&L.

---

## Phase 4 — Strategy Refinement
### Status: FUTURE (Sessions 17+)

### 4A — Earnings Catalyst Feature
Binary flag: signal 8-30 days before earnings. Directly captures #1 feature pattern.

### 4B — Short Side Exploration
Low scores + insider sells → short-side alpha. Insider sells are noisier but clustered selling may work.

### 4C — Multi-Horizon Optimization
Separate models for 5d, 30d, 90d. CFO buys before earnings = 5d signal. Director buys = 90d signal.

### 4D — Sector-Specific Models
Finance and Healthcare show different patterns. Train sector-specific models for top 3 sectors.

### 4E — FinBERT Upgrade
Replace VADER with financial-domain BERT. 15-20% better accuracy on financial text.

### 4F — Congressional Signal Rehabilitation
Identify top 10-15 profitable congressional traders. Individual quality scores vs blanket 0.300.

---

## Success Metrics

| Metric | Current (S15) | Target Phase 2 | Phase 2 Result |
|---|---|---|---|
| IC | 0.1027 | > 0.10 | 0.1027 ✅ |
| p-value | 0.0003 | < 0.001 | 0.0003 ✅ |
| Sharpe | 1.53 | > 0.40 | 1.53 ✅ |
| Max drawdown | -43.0% | < 25% | -43.0% ❌ (COVID tail) |
| Calmar ratio | 2.12 | > 1.0 | 2.12 ✅ |
| Factor-adj alpha | +128.9% | > 5% annualized | +128.9% ann (t=3.27**) ✅ |
| Kelly sizing | 11.2% | implemented | ✅ |
| Dashboard | built | — | ✅ |
| Kelly sizing | implemented | — | per-signal |
| Regime guardrails | implemented | — | VIX-adaptive |
| Dashboard | scaffold | — | polished |

---

## What We Are NOT Building

- High-frequency trading system
- Institutional-scale portfolio (liquidity irrelevant at retail)
- Automated execution (signals are recommendations, not auto-trades)
- Options trading (yet — Phase 4)
- Short-selling (yet — Phase 4)

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.13 |
| ML Model | LightGBM (walk-forward ensemble) |
| Database | SQLite (migrate to Supabase at 30k signals) |
| Insider data | SEC EDGAR XML (free, authoritative) |
| Congress data | FMP /stable/ endpoints |
| Prices | yfinance |
| Short interest | FINRA API (free) |
| Sentiment | VADER (future: FinBERT) |
| Pipeline | GitHub Actions (Monday weekly) |
| Frontend | TBD (Phase 3) |
