# ATLAS — Live Data Roadmap

## ⚠️ PRIORITY 1 — Data Accuracy Overhaul (IN PROGRESS — RESUME HERE)
**Rule: No data rather than incorrect data. Everything that appears live must BE live.**

Full plan at: `/Users/henrysalazar/.claude/plans/squishy-churning-fern.md`

### What's Already Live (don't touch)
- `#cong-tbl tbody` — replaced by `renderCongressTrades()` from QuiverQuant ✅
- `#edgar-feed` — replaced by `renderEdgarFeed()` from EDGAR ✅
- `#congress-feed` — hardcoded bill cards (no pass% API exists, acceptable) ✅
- Finnhub prices, VIX, 10yr yield ✅

### Group A — Fix Factually Wrong Text ❌ NOT DONE
Remove all "pre-earnings" / "18 days pre-earnings" SMPL text (6 locations):
- Line ~477: ticker scroll `Pre-Earnings Warning` → `CEO Exit Warning`
- Line ~546: Top Signals table `· no plan · pre-earnings` → `· no plan`
- Line ~570: Live Alerts `18 days pre-earnings` → remove clause
- Line ~757: Insider Exit table `Pre-earnings` → remove
- Line ~1342: Notification `18 days pre-earnings.` → remove
- Line ~1379: `TRACKED.SMPL.note` `· pre-earnings` → remove

### Group B — Congress Table Loading State ❌ NOT DONE
Replace 9 hardcoded `<tr>` rows in `#cong-tbl tbody` (lines 632–641) with a single loading row.
Live `renderCongressTrades()` replaces it when QuiverQuant data loads (~5s).

### Group D — Add DEMO Banners ❌ NOT DONE
Add clear labels to sections with no live data source:
- Institutional flows cards (~line 800): `Historical · 13F Q4 2025 · Not yet live`
- Options flow table (~line 847): `Demo · CBOE/OPRA integration not yet live`
- Short interest table (~line 860): `Demo · SI data source not yet integrated`
- Committee correlation (~line 644): `(Historical estimate)` in card-sub
- Sector performance (~line 895): `Signal-weighted estimate · Not real-time`
- Berkshire 13F tracker (~line 869): `Q4 2025 13F · Updated quarterly`
- Market breadth (~line 931): `Demo values · Market data source pending`

### Group E — Congress Page KPI Cleanup ❌ NOT DONE
In `renderCongressTrades()` at end, compute:
- High Suspicion Trades → count buys with score ≥ 30
- Total Vol. Disclosed → `--`
- Avg Outperformance → `--`
- Late STOCK Act Filers → `--`

### Group F — Remove All 8 Trade Idea Cards ❌ NOT DONE
**Live signal check result: ALL 8 cards score < 40 pts. Remove all.**
Key finding: SMPL has 2 congressional BUYS in live data — directly contradicts the hardcoded bearish card.
Replace ideas container with "No signals above threshold / Score ≥ 85 required" empty state.

### Group C1 — renderTopSignals() ❌ NOT DONE
New JS function: computes live convergence scores for TRACKED tickers, fills Overview Top Signals table.
Add `id="top-signals-tbl"` to `<table>` at ~line 534.
Also updates `#ideas-max-score` on Trade Ideas page.
Hook: call from end of `refreshConvergenceDisplays()`.

### Group C2 — renderLiveAlerts() ❌ NOT DONE
New JS function: replaces 4 hardcoded `.alert-row` divs.
Add `id="live-alerts-panel"` to wrapping div (~line 564). Clear hardcoded rows.
Checks: bill votes within 7 days + TRACKED scores ≥ 75.
Shows "Monitoring..." empty state if no alerts.
Hook: call from end of `refreshConvergenceDisplays()`.

### Group C3 — updateOverviewKPIs() ❌ NOT DONE
New JS function: replaces 7 hardcoded KPI values (kb1–kb7) with live-computed values.
kb1=max score, kb2=count≥80, kb3=cluster count, kb4=congress buy count,
kb5="--", kb6=BILLS.length, kb7=multi-source count.
Hook: call from end of `refreshConvergenceDisplays()`.

### Group C4 — renderInsiderTableLive() ❌ NOT DONE
New JS function: clears 8 hardcoded `#ins-tbl tbody` rows.
Uses `edgarData` + `TICKER_KEYWORDS` to find tracked-ticker EDGAR matches.
Shows real data: insider name, company, date, SEC link, cluster tag, partial score.
Updates Insider page KPIs (most `--` due to EDGAR EFTS field limits).
Hook: call from end of `renderEdgarFeed()`.

### Group C5 — Clear Notification Center ❌ NOT DONE
Remove 6 hardcoded `<div class="notif-item unread">` from `#notif-list` (lines 1339–1344).
Replace with empty state. Add `renderNotifications()` mirroring `renderLiveAlerts()` logic.
Hook: call from end of `refreshConvergenceDisplays()`.

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
