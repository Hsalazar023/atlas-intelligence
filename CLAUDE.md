# ATLAS — Current State

## Metrics (as of 2026-03-05)
- IC: 0.1027 [p=0.0003, SIGNIFICANT]
- OOS IC: 0.1000 [confirmed matches walk-forward]
- Training signals: 7,655
- Active features: 28
- Health: DEGRADED [ic_trend:OK | hit_rate:OK | freshness:WARN]
- Phase: 2 COMPLETE | Phase 3 IN PROGRESS
- Factor alpha: +128.9% annualized (t=3.27**, R²=0.082)

## What Was Just Done (Session 18)
- Dashboard complete rewrite: 4-tab architecture (Positions, Performance,
  Intelligence, Model) with hash-based navigation
- Fixed win_probability double-multiply bug (71.4 shown as 7140%)
- Fixed factors [object Object] — now parsed from array of objects
- New signals have same expandable detail panels as active positions
- portfolio_stats.json export added to --export (honest full-DB closed stats)
- Signal intelligence interpretations for each gap metric
- Fama-French section in Intelligence tab (+177% annualized alpha)
- Full analyst report rendered in Model tab (inventory, performance,
  quality, regime, Kelly, factors, IC trend chart, pipeline health)
- Regime label removed from individual rows (banner + detail only)
- Feature importance table (top 15) in Intelligence tab

## Next Session (19)
1. Dashboard polish — review live dashboard, fix remaining display issues
2. GitHub Actions workflow failures — debug and fix backtest.yml + fetch-data.yml
3. Run --export to generate portfolio_stats.json, verify Performance tab
4. Investigate OOS gap (-3.4) — why OOS doesn't discriminate best/worst
5. Liquidity enrichment debug (0 signals filled)

## File Map
Pipeline:    scripts/fetch_data.py → --backfill → --analyze → --export → --report
Signals:     data/brain_signals.json (top 50 active + 50 closed, live prices)
DB:          data/atlas_signals.db (7,655 rows, 6.5MB)
Dashboard:   data/dashboard.html (4-tab, ~920 lines)
Standards:   docs/STANDARDS.md (read at session start)
Session log: docs/SESSION-LOG.md (append at session end)
Brain stats: docs/brain-status.md (detailed metric history)
FF factors:  data/ff5_factors.csv (85 months, 2019-2026)
Intel:       data/signal_intelligence.json (best/worst signal profiles)
Portfolio:   data/portfolio_stats.json (honest closed stats — needs --export)
Report:      data/analyst_report.json (full analyst report — needs --report)

## Active Known Issues
- Congress stuck at 2026-02-13: FMP upstream delay (not our code)
- Liquidity enrichment: 0 signals filled, path issue
- Prune candidates flagged 3+ runs: holding
- OOS scores: recent signals (Feb 2026+) have no OOS score — expected
- portfolio_stats.json: 404 until user runs --export

## IC Trend (last 5 sessions)
S13: 0.1000 | S14: 0.1027 | S15: 0.1027 | S16: 0.1027 | S17: 0.1027

## Signal Intelligence (first run)
- Score gap: 43.9 — model IS discriminating (best=72.3, worst=28.4)
- OOS gap: -3.7 — OOS scores less discriminating than total_score
- Earnings gap: -98.9 — best signals far from earnings, worst near
- Healthcare dominates both best and worst (largest sector)

## Phase 2 Research Summary (COMPLETE)
OOS Drawdown (75+ threshold):
  Ann return: +91.1% | Max DD: -43.0% (Feb 2020) | Calmar: 2.12 | Sharpe: 1.53
  2022 bear: +9.01% avg monthly — genuine alpha confirmed
Regime (OOS): VIX 15-25 = IC 0.104*** | VIX<15 = 0.041ns | VIX 25-35 = 0.069ns
Kelly: 11.2% base per position (1/4 Kelly, OOS 75+)
Factor alpha: +7.14%/mo (t=3.27**), ann +128.9%, R²=0.082
  No significant factor loadings — pure insider alpha

## Working Rules
- **Don't re-read files** already in context. Grep → Read with offset/limit.
- **Don't read data files** (`data/*.json`, `data/*.db`, price history).
- **Don't run pipelines** — user runs locally. Write the code only.
- **Don't launch agents** without asking first.
- **No hardcoded signals.** All data engine-generated.
- **No keys in frontend.** API keys in env vars / Vercel secrets.
- **Never commit** `data/`, `Skills/`, `.claude/`, `.firecrawl/`.

## Environment
- No `sudo` — use `npm install -g --prefix ~/.npm-global`
- Vercel CLI: `~/.npm-global/bin/vercel`
