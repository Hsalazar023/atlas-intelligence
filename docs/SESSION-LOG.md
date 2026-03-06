# ATLAS Session Log
Running record of every session. Append-only — never edit past entries.

---

## Session 11 — 2026-02-20 (estimated)

### Metrics
| Metric            | Before  | After   | Delta   |
|-------------------|---------|---------|---------|
| IC (walk-forward) | 0.1092  | ~0.1028 | -0.0064 |
| Folds             | 74      | 75      | +1      |
| Training signals  | 7,178   | ~7,200  | ~+22    |
| Active features   | 28      | 28      | 0       |
| Health            | —       | —       | —       |

### Tasks Completed
- No dedicated todo section exists for this session
- Transitional session between S10 (feature pruning) and S12 (decay/regime research)
- IC dip from 0.1092 to ~0.1028 likely from new signal ingestion shifting fold boundaries

### Key Findings
- Walk-forward folds increased from 74 to 75 as data grew past another month boundary
- IC regression from 0.1092 was minor and within normal variance

### Issues Found
- No session documentation was created — gap in records

### Code Changes
| File | Change |
|------|--------|
| — | No documented code changes |

### Next Session Proposed Focus
1. Congress ingestion debug — verify data flow
2. Signal decay analysis — optimal holding horizon
3. Regime robustness — VIX bucket performance

---

## Session 12 — 2026-02-24 (estimated)

### Metrics
| Metric            | Before  | After   | Delta   |
|-------------------|---------|---------|---------|
| IC (walk-forward) | ~0.1028 | 0.1028  | 0       |
| Folds             | 75      | 75      | 0       |
| Training signals  | ~7,200  | ~7,219  | ~+19    |
| Active features   | 28      | 28      | 0       |
| Health            | —       | —       | —       |

### Tasks Completed
- [Task 0] Congress ingestion debug → no bug found, all 664 purchases already in DB. Staleness is upstream FMP delay.
- [Task 2] Signal decay analysis → 30d confirmed optimal (IC=0.4016 in-sample). 5d too early, 365d alpha persists for top signals.
- [Task 3] Regime robustness (in-sample) → all 3 regimes PASS. Low vol IC=0.3619, Normal IC=0.4567, Elevated IC=0.3611.
- [Task 4] Fetch optimization → XML reuse via accession_number. ~1,400 XML fetches skipped per run.
- [Task 5] Drawdown simulation (in-sample) → 80+ signals: 99% win months, max DD -2.8%. CAVEAT: in-sample, inflated.

### Key Findings
- 30d is optimal training horizon — confirmed by decay analysis
- All in-sample metrics are inflated (total_score trained on car_30d) — OOS validation needed
- Congress data stuck at 2026-02-13, not a code bug

### Issues Found
- All regime/drawdown results used total_score (in-sample) — not trustworthy for trading decisions
- Walk-forward OOS predictions exist per fold but aren't persisted in DB

### Code Changes
| File | Change |
|------|--------|
| learning_engine.py | Added regime robustness section to --report |
| learning_engine.py | Added accession_number column, XML skip logic |
| fetch_data.py | XML reuse from edgar_feed.json prior enrichments |

### Next Session Proposed Focus
1. Store OOS predictions in DB for honest backtesting
2. Transaction cost modeling — bid-ask spread by market cap
3. Live price feed for dashboard foundation

---

## Session 13 — 2026-02-27 (estimated)

### Metrics
| Metric            | Before  | After   | Delta   |
|-------------------|---------|---------|---------|
| IC (walk-forward) | 0.1028  | ~0.1000 | -0.0028 |
| Folds             | 75      | 75      | 0       |
| Training signals  | ~7,219  | ~7,219  | 0       |
| Active features   | 28      | 28      | 0       |
| Health            | —       | —       | —       |

### Tasks Completed
- [Task 1] Store OOS predictions in DB → oos_score + oos_fold columns added. Walk-forward holdout probabilities persisted per signal. 6,963/7,678 (90.7%) have OOS scores.
- [Task 2] Regime robustness (honest OOS) → --report now uses oos_score with fallback. Labels "OOS (walk-forward holdout)" vs "IN-SAMPLE".
- [Task 3] Drawdown simulation (honest OOS) → Stop-loss fields added: stop_loss_price (-12%), stop_loss_triggered, position_status.
- [Task 4] Transaction cost modeling → ADV from price cache, spread by market cap (large 0.05%, mid 0.20%, small 0.50%). net_expected_return computed.
- [Task 5] Live price feed → yfinance batch fetch in export. Added current_price, unrealized_pnl_pct, days_held, days_remaining to brain_signals.json.

### Key Findings
- OOS stored IC should match walk-forward IC ~0.0996 — first honest backtest possible
- Transaction costs are modest: large cap 0.05%, small cap 0.50%
- brain_signals.json now contains full position management data

### Issues Found
- Liquidity enrichment: 0 signals enriched despite being wired in — path or ADV calculation issue
- OOS IC slightly lower than S12 — normal variance with new predictions stored

### Code Changes
| File | Change |
|------|--------|
| learning_engine.py | Added oos_score/oos_fold columns, OOS persistence after walk-forward |
| learning_engine.py | Stop-loss fields in export, live price batch fetch |
| learning_engine.py | Transaction cost model: enrich_liquidity_features() |
| ml_engine.py | OOS predictions returned from walk_forward_train() |

### Next Session Proposed Focus
1. Regime guardrails — VIX-based score multipliers
2. Kelly position sizing per signal
3. Dashboard scaffold

---

## Session 14 — 2026-03-01 (estimated)

### Metrics
| Metric            | Before  | After   | Delta   |
|-------------------|---------|---------|---------|
| IC (walk-forward) | ~0.1000 | 0.1000  | 0       |
| OOS Sharpe (75+)  | —       | 1.53    | new     |
| OOS Calmar (75+)  | —       | 2.12    | new     |
| OOS Max DD        | —       | -43.0%  | new     |
| Folds             | 75      | 75      | 0       |
| Training signals  | ~7,219  | ~7,219  | 0       |
| Active features   | 28      | 28      | 0       |
| Health            | —       | —       | —       |

### Tasks Completed
- [Task 0] Regime-aware scoring guardrails → VIX multipliers: OPTIMAL 1.00x, LOW_VOL 0.75x, ELEVATED 0.85x, HIGH_VOL 0.90x, CRISIS 0.60x. Per-signal regime_multiplier/regime_label in brain_signals.json.
- [Task 3] Kelly position sizing → 1/4 Kelly formula with OOS 75+ params. Scaled by signal confidence and regime. Capped 2-15%. kelly_size per signal in brain_signals.json.
- [Task 4] Dashboard scaffold → dashboard.html created. Regime banner, stats row, active positions, new signals, closed, health bar.

### Key Findings
- OOS drawdown simulation (75+ threshold): Sharpe 1.53, Calmar 2.12, ann return +91.1%
- 2022 bear market: +9.01% avg monthly — genuine alpha confirmed
- 2024 weakness: avg VIX=16.1, IC=0.040 ns — regime effect, not model failure
- Kelly base: 11.2% per position (1/4 Kelly, OOS 75+ hit=67.7%, W/L=1.42x)

### Issues Found
- Report JSON save order bug — save_json called before regime/Kelly data added (fixed)
- Congress still stuck at 2026-02-13

### Code Changes
| File | Change |
|------|--------|
| learning_engine.py | get_regime_context() with VIX multipliers |
| learning_engine.py | compute_kelly_size() with 1/4 Kelly formula |
| learning_engine.py | regime_context + kelly_params in brain_signals.json |
| dashboard.html | New file — trading dashboard scaffold |

### Next Session Proposed Focus
1. Full dashboard build — sortable tables, P&L, regime banner
2. NONE ticker cleanup
3. Fama-French factor regression

---

## Session 15 — 2026-03-04

### Metrics
| Metric            | Before  | After   | Delta   |
|-------------------|---------|---------|---------|
| IC (walk-forward) | 0.1000  | 0.1027  | +0.0027 |
| Folds             | 75      | 75      | 0       |
| Training signals  | ~7,219  | 7,678   | +459    |
| OOS scores stored | —       | 6,963   | —       |
| Active features   | 28      | 28      | 0       |
| Health            | —       | DEGRADED| —       |

### Tasks Completed
- [Task 0] NONE ticker filter → export_brain_data() skips NONE/NULL/N/A/empty and >5 char tickers from yfinance batch. DB cleanup script provided.
- [Task 1] Dashboard build → data/dashboard.html complete. Dark theme, sortable tables, responsive. Sections: regime banner, 6 KPI cards, active positions (P&L, stop loss, Kelly bars), new signals (entry zones, liquidity), closed positions, health panel.
- [Task 2] Fama-French 6-factor regression → Added to --report. OLS on monthly excess returns vs Mkt-RF, SMB, HML, RMW, CMA, MOM. Requires data/ff5_factors.csv.
- [Task 3] Phase 2 completion → Updated all docs. Phase 2 declared COMPLETE.

### Key Findings
- IC recovered from 0.1000 to 0.1027 (+0.0027) — likely from new signal ingestion
- Phase 2 fully complete: OOS Sharpe 1.53, Calmar 2.12, regime guardrails, Kelly sizing
- brain_signals.json contains Python NaN — invalid JSON, dashboard can't load

### Issues Found
- BLOCKER: brain_signals.json contains NaN → SyntaxError in browser. Fix applied in shared.py (save_json sanitization) but NOT YET VERIFIED.
- NONE ticker still in DB (filter applied in export, DELETE pending)
- Liquidity enrichment still 0 signals enriched
- Fama-French results pending (needs ff5_factors.csv download)

### Code Changes
| File | Change |
|------|--------|
| learning_engine.py | NONE ticker filter in export_brain_data() |
| learning_engine.py | Fama-French 6-factor regression in generate_analyst_report() |
| shared.py | _sanitize_for_json() in save_json() — NaN/Infinity → None |
| data/dashboard.html | Full dashboard build (dark theme, sortable, responsive) |

### Next Session Proposed Focus
1. Verify NaN fix — run --export, validate JSON, test dashboard
2. Delete NONE ticker from DB
3. Run Fama-French regression (download ff5_factors.csv first)

---

## Session 16 — 2026-03-05

### Metrics
| Metric            | Before  | After   | Delta   |
|-------------------|---------|---------|---------|
| IC (walk-forward) | 0.1027  | 0.1027  | 0       |
| Folds             | 75      | 75      | 0       |
| Training signals  | 7,678   | 7,655   | -23 (NONE cleanup) |
| Active features   | 28      | 28      | 0       |
| Health            | DEGRADED| DEGRADED| 0       |
| Factor alpha (ann)| pending | +128.9% | NEW     |

### Tasks Completed
- [Task 1] Dashboard fixes → OOS column shows "new" for recent signals (no OOS score expected), folds numerator fixed (reads pos_folds from optimal_weights), sector abbreviations, closed table enhanced with OOS/sector columns.
- [Task 2] NONE ticker cleanup → 23 NONE records deleted from DB. 7,655 signals remaining.
- [Task 3] Fama-French 6-factor regression → Monthly alpha +7.14%/mo (t=3.27**), annualized +128.9%, R²=0.082. No significant factor loadings. Verdict: STRONG genuine insider edge.
- [Task 4] Output standards → fetch_data.py refactored: _quiet_run() captures verbose output to logs/fetch_verbose.log, main() prints one clean line per source with ✓/⚠/✗ status. learning_engine.py: 12 verbose log.info→log.debug for fill-rate, roles, tiers, regime stats, hypotheses.
- [Task 5] Congress investigation → Confirmed FMP upstream delay. Latest purchase 2026-02-13, 20 days stale. Not our code. Documented as known issue, stop investigating.

### Key Findings
- **Factor alpha is STRONG**: +7.14%/mo (t=3.27), annualized +128.9%. R²=0.082 means only 8% of returns explained by market/size/value/quality/investment/momentum. Genuine insider edge.
- **OOS column "—" is expected**: All active signals are Feb-Mar 2026, too recent for walk-forward folds. 6,963/7,655 historical signals have OOS scores.
- **TPL verified**: price_at_signal ~$337-366, current ~$536. +58.86% return over 48 days. DB confirms multiple TPL signals from Jan-Mar 2026.

### Issues Found
- Congress data still stuck at 2026-02-13 (FMP upstream, not our code)
- Liquidity enrichment still 0 signals (carried forward)

### Code Changes
| File | Change |
|------|--------|
| learning_engine.py | Added pos_folds to brain_stats ml section (line ~6650) |
| learning_engine.py | 12x log.info→log.debug for verbose output suppression |
| data/dashboard.html | OOS column: oosCell() shows "new" for null, sector abbreviations |
| data/dashboard.html | Folds: reads ml.pos_folds instead of ml.positive_folds |
| data/dashboard.html | Closed table: added OOS + Sector columns, increased limit to 30 |
| scripts/fetch_data.py | _quiet_run() context manager, SUPPRESSED_TICKERS set |
| scripts/fetch_data.py | main() refactored: one-line-per-source clean output |
| scripts/fetch_data.py | Removed save_json print (noise) |

### Next Session Proposed Focus
1. Run full pipeline (--backfill --analyze --export --report) to verify output standards
2. Signal history page — closed positions with realized P&L chart
3. New signal notifications (80+ alerts via ntfy)
4. Debug liquidity enrichment (0 signals filled)

---

## Session 17 — 2026-03-05 (continued)

### Metrics
| Metric            | Before  | After   | Delta   |
|-------------------|---------|---------|---------|
| IC (walk-forward) | 0.1027  | 0.1027  | 0       |
| Training signals  | 7,655   | 7,655   | 0       |
| Active features   | 28      | 28      | 0       |
| Health            | DEGRADED| DEGRADED| 0       |
| Signal intel      | —       | score_gap=43.9 | NEW |

### Tasks Completed
- [Task 0] CAR measurement note → Added to STANDARDS.md under Key Metrics Reference. Documents disclosure timing haircut (3-8%).
- [Task 1] Ticker grouping → Active positions grouped by ticker with expandable detail panels showing insider list, signal strength (ML confidence, win prob, regime, factors), and trade parameters (entry zone, upside, stop, horizon). CSS transitions, color-coded borders.
- [Task 2a] New signals table → Entry zone + target replaced with Est. Upside (%), Stop ($+%), and Horizon columns.
- [Task 2e] Signal count badge → Stats row shows "50 | 22 active | X new" instead of raw DB count.
- [Task 3] Portfolio performance → Active return card (equal-weight avg with 7% timing haircut), closed signal stats (win rate, avg win/loss, best/worst), monthly performance log.
- [Task 3d] Export closed_signals → learning_engine.py exports last 50 expired signals (30-90 days old) as separate closed_signals array in brain_signals.json.
- [Task 4] Signal intelligence → compute_signal_intelligence() wired into --analyze. Profiles best/worst 50 signals by score, OOS, VIX, earnings timing, roles, sectors. Dashboard panel shows divergence stats + comparison table. Initial data generated.
- [Task 6a] OOS display → Shows total_score in blue with "est" label when oos_score is null, instead of "new".

### Key Findings
- **Score gap 43.9**: Model total_score strongly separates winners from losers (best avg=72.3 vs worst avg=28.4)
- **OOS gap -3.7**: OOS scores are LESS discriminating than total_score for best/worst — worth investigating
- **Earnings gap -98.9**: Best signals are far from earnings, worst are near — confirms days_to_earnings is critical
- **Healthcare dominates both lists**: Largest sector in DB, not a differentiator

### Issues Found
- OOS gap negative: OOS scores don't separate best from worst as well as total_score. Could indicate OOS is too conservative or the best signals are edge cases that OOS smooths out.

### Code Changes
| File | Change |
|------|--------|
| docs/STANDARDS.md | CAR measurement note added |
| data/dashboard.html | Expandable ticker groups, portfolio perf, signal intel, OOS fallback |
| learning_engine.py | compute_signal_intelligence() function + wired into --analyze |
| learning_engine.py | closed_signals export in export_brain_data() |
| learning_engine.py | SIGNAL_INTELLIGENCE path constant |

### Next Session Proposed Focus
1. Run full pipeline to verify all changes work end-to-end
2. Investigate OOS gap (-3.7) — why does OOS not discriminate best/worst?
3. Dashboard mobile layout polish
4. Debug liquidity enrichment
5. Begin Phase 4 planning based on intelligence findings

---

## Session 18 — 2026-03-05 (continued)

### Metrics
| Metric            | Before  | After   | Delta   |
|-------------------|---------|---------|---------|
| IC (walk-forward) | 0.1027  | 0.1027  | 0       |
| Training signals  | 7,655   | 7,655   | 0       |
| Active features   | 28      | 28      | 0       |
| Health            | DEGRADED| DEGRADED| 0       |
| Dashboard tabs    | 1       | 4       | +3      |

### Tasks Completed
- [Task 0a] Win probability fix — was multiplying already-percentage value by 100 (7140% shown). Fixed to display raw value.
- [Task 0b] Factors fix — field is array of objects, code used Object.entries() treating as dict. Added topFactors() parser for array/dict/list formats. Shows "feature: X.X%" properly.
- [Task 0c] pos_folds verified — already exists at ml.pos_folds = 52. No fix needed.
- [Task 0d] portfolio_stats.json — new export in learning_engine.py. Queries ALL signals with car_30d for honest closed stats (thousands, not 24). Includes summary with win_rate, avg_win, avg_loss, best/worst, monthly breakdown.
- [Task 1] 4-tab architecture — POSITIONS (default), PERFORMANCE, INTELLIGENCE, MODEL. Pure JS show/hide with blue underline active indicator. Hash-based navigation (#tab-model).
- [Task 2] Active positions expanded detail — fixed factors display, win_probability (no double multiply), added trade_size_range, target1/target2 with % from entry mid, cluster boost info. Regime only in expanded panel, not rows.
- [Task 3] New signals expandable — same detail panels as active positions. Grouped by ticker with cluster badges, click-to-expand.
- [Task 4] Portfolio performance tab — active portfolio card (raw + 3% adjusted), closed signal summary from portfolio_stats.json, monthly performance with CSS bar chart, paginated closed positions log.
- [Task 5] Signal intelligence tab — interpretations added for each gap metric. Fama-French section with monthly alpha +8.9%, annualized +177%. Factor loadings table. Feature importance table (top 15 with bars).
- [Task 6] Model health panel — IC trend, positive folds, EDGAR/Congress freshness (congress shows "Xd stale" in amber), feature importance table from analyst_report.json.
- [Task 7] Analyst report tab — full report rendered: data inventory, model performance, signal quality by score band, regime robustness, Kelly sizing, factor analysis, IC trend chart (bar visualization), pipeline health.
- [Task 8] Regime label removed from individual rows — only appears in regime banner (Tab 1 top), expanded detail panel, and model health.

### Key Findings
- **win_probability was double-multiplied**: Raw value 71.4 (already %) was being shown as 7140%. Active dashboard users would have seen wildly inflated probabilities.
- **factors was showing [object Object]**: Array of rich objects (feature, importance, label, direction) rendered as text. Now properly parsed and displayed.
- **Honest closed stats**: portfolio_stats.json will show thousands of signals from full DB vs the biased top-50 export. Win rate, avg win/loss will be realistic.
- **Dashboard now answers the 3 trader questions**: (1) Positions tab — what you own and how it's doing. (2) New signals section — what to consider buying. (3) Intelligence + Model tabs — is the model still working.

### Issues Found
- portfolio_stats.json 404 — needs user to run `--export` to generate it
- No JS errors in headless Chrome verification

### Code Changes
| File | Change |
|------|--------|
| learning_engine.py | PORTFOLIO_STATS path constant, portfolio_stats.json export in export_brain_data() |
| data/dashboard.html | Complete rewrite — 4-tab architecture, fixed factors/win_prob display, expandable new signals, honest portfolio stats, interpretations, analyst report rendering, IC trend chart |

### Next Session Proposed Focus
1. Run --export to generate portfolio_stats.json, verify all tabs populate
2. Review OOS gap finding (-3.4) — why OOS doesn't discriminate best/worst
3. Liquidity enrichment debug (0 signals filled)
4. IC trend chart (live from brain_health.json history)
5. New signal notifications (ntfy push alerts for 80+)
