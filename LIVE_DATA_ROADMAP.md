# ATLAS — Live Data Roadmap

## ✅ PRIORITY 1 — Data Accuracy Overhaul — COMPLETE
**Rule: No data rather than incorrect data. Everything that appears live must BE live.**

Full plan at: `/Users/henrysalazar/.claude/plans/squishy-churning-fern.md`

### Group A — Fix Factually Wrong Text ✅ DONE (Feb 24, 2026)
Removed all 6 SMPL "pre-earnings" instances. Earnings passed Jan 8, next Apr 8.

### Group B — Congress Table Loading State ✅ DONE
Replaced 9 hardcoded rows with loading spinner. Live data replaces it on QuiverQuant load.

### Group D — Add DEMO Banners ✅ DONE
Added amber labels to: 13F cards, options flow, short interest, committee correlation,
sector performance, Berkshire tracker, market breadth.

### Group E — Congress Page KPI Cleanup ✅ DONE
High Suspicion Trades computed live from congData. Total Vol/Avg Outperformance/Late Filers → "--".

### Group F — Remove All 8 Trade Idea Cards ✅ DONE
All 8 removed (none scored ≥ 40 vs live data). Empty state + convergence engine messaging added.

### Group C1 — renderTopSignals() ✅ DONE
Overview Top Signals table fills from live convergence scores. `id="top-signals-tbl"` added.

### Group C2 — renderLiveAlerts() ✅ DONE
Live alerts panel replaces 4 hardcoded rows. Shows threshold crossings or monitoring message.

### Group C3 — updateOverviewKPIs() ✅ DONE
All 7 Overview KPIs computed live: max score, exceptional count, cluster count, congress buys,
"--" for no-source KPIs, BILLS.length, multi-source count.

### Group C4 — renderInsiderTableLive() ✅ DONE
Insider table populated from EDGAR filings via TICKER_KEYWORDS match. Cluster detection.
Role/amount show N/A (EDGAR EFTS limitation). Wired to renderEdgarFeed().

### Group C5 — Clear Notification Center ✅ DONE
6 hardcoded notifications removed. renderNotifications() generates live alerts in notif center.

---

## Phase 1 — Real Data Pipelines (after accuracy overhaul)
- [ ] Parse full EDGAR XML for role (CEO/CFO) and transaction type
- [ ] Convergence detector: tickers in both feeds within rolling window = first real signal
- [ ] Auto-generate trade idea cards when score ≥ 85

## Phase 2 — Full Stack Rebuild
- [ ] Convert to Next.js 14 App Router
- [ ] Hub pages: /congressional, /insider, /institutional, /convergence
- [ ] Supabase: signals, congressional_trades, form4_filings, convergence_events

## Completed
- [x] Phase 0: Finnhub prices, zone badges, Congress.gov bills, Vercel deploy
- [x] P1: EDGAR Form 4 feed (data/edgar_feed.json)
- [x] P1: QuiverQuant congressional trades (data/congress_feed.json)
- [x] P2.1: rr-block cells synced from TRACKED
- [x] P2.2: Signal staleness detection (>30 days warning)
- [x] P2.3: Convergence scoring engine (computeConvergenceScore)
- [x] FRED market data (VIX + 10yr yield via data/market_data.json)
- [x] GitHub Actions: 4x daily data refresh at market-aligned times
- [x] CSS fix: scroll sections fill card height (card-fill flex utility)
- [x] Data accuracy audit: identified all hardcoded sections + live data gaps
- [x] Data accuracy overhaul: Groups A/B/C1-C5/D/E/F — all hardcoded content replaced or labeled
