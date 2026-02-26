# ATLAS — Live Data Roadmap
*Updated Feb 25, 2026*

## Current State Summary

### What's LIVE (dynamically rendered from real data)
| Section | Data Source | Refresh Rate |
|---|---|---|
| Top Signals table | Convergence engine scoring all universe tickers | On data load |
| Trade Ideas / Watchlist | `renderSignalIdeas()` — score ≥ 65 / 40–64 | On data load |
| Congressional Trading table | QuiverQuant API → `data/congress_feed.json` | 4x daily |
| Insider Signals table | SEC EDGAR EFTS → `data/edgar_feed.json` | 4x daily |
| Overview KPI strip | `updateOverviewKPIs()` from live feeds | On data load |
| Live Alerts feed | `renderLiveAlerts()` threshold crossings | On data load |
| Notification center | `renderNotifications()` from live scores | On data load |
| Price strip | Finnhub `/quote` API | 60s intervals |
| Zone badges | `updateIdeaCard()` — IN ZONE / ABOVE / MISSED | On price update |
| Market data | FRED API → VIX + 10yr yield | 4x daily |
| Backtest stats bar | `data/optimal_weights.json` | Weekly (Sundays) |

### What's HARDCODED (static demo content)
| Section | Page | Notes |
|---|---|---|
| Sector Signal Heat | Overview + Insider | Static % (Defense +12.8%, etc.) |
| Legislative Calendar | Congressional Intel | Static bill dates and passage % |
| Industry Strength Matrix | Markets & Sectors | Static scores (+8.4, etc.) |
| Institutional 13F cards | Institutional Flows | Demo Q4 2025 data, marked with amber badge |
| Options Flow table | Institutional Flows | Demo, marked "CBOE/OPRA not yet live" |
| Short Interest Anomalies | Institutional Flows | Demo, marked "SI data source not yet integrated" |
| Berkshire 13F Tracker | Institutional Flows | Static Q4 2025 holdings |
| Insider page KPIs | Insider Signals | Hardcoded (247, 14, 4, 38) |
| Notable Exit Signals | Insider Signals | Hardcoded SMPL, DOCS rows |
| Ticker ribbon marquee | Header | 6 hardcoded tickers with scores |
| Alerts/SMS config | Alerts & SMS | UI only, no Twilio integration |

---

## Completed Milestones

### Phase 0 — Foundation ✅
- [x] Finnhub live prices + 60s refresh
- [x] Zone badges (IN ZONE / ABOVE ZONE / MISSED)
- [x] Congress.gov API key wired in
- [x] Deployed to Vercel + GitHub repo

### Data Accuracy Overhaul ✅ (Feb 24, 2026)
- [x] Group A: Fixed SMPL earnings text
- [x] Group B: Congress table loading state (replaced 9 hardcoded rows)
- [x] Group C1–C5: Dynamic rendering for Top Signals, Live Alerts, KPIs, Insider table, Notifications
- [x] Group D: DEMO banners on all hardcoded sections
- [x] Group E: Congress page KPI cleanup
- [x] Group F: Removed all 8 hardcoded trade idea cards

### Signal Engine Expansion ✅ (Feb 24, 2026)
- [x] Open universe expansion: scores all tickers from feeds (not just TRACKED 11)
- [x] Two-stage EDGAR matching (TICKER_KEYWORDS + ticker-string fallback)
- [x] buildTickerUniverse() function
- [x] fetchTopScorerPrices() for non-cached tickers

### Scoring & Backtest Engine ✅ (Feb 25, 2026)
- [x] Signal decay: 21-day half-life on congressional scores
- [x] Watchlist tier: scores 40–64 shown as Monitoring cards
- [x] Trade Ideas threshold lowered: 85 → 65 (dynamic via SCORE_THRESHOLD)
- [x] backtest/shared.py — constants, helpers, TICKER_KEYWORDS
- [x] backtest/collect_prices.py — Finnhub OHLC cache (needs paid key)
- [x] backtest/run_event_study.py — CAR at 5d/30d/90d + member track records
- [x] backtest/optimize_weights.py — grid search for optimal weights
- [x] 23 unit tests, all passing
- [x] GitHub Actions: weekly backtest (Sundays 02:00 UTC)
- [x] Frontend: loadOptimalWeights() + dynamic decay half-life + backtest stats bar

---

## Resolved: Historical Price Data ✅ (Feb 26, 2026)

Switched from Finnhub (paid-only `/stock/candle`) to **yfinance** (free, no API key). Pipeline now fetches 365 days of daily OHLC for all 138 universe tickers + SPY benchmark. First full run: 95 events processed, 74 with real CARs, 56.8% hit rate.

---

## Next Priorities

### Priority 2 — Replace Remaining Hardcoded Sections
Convert static HTML sections to dynamic rendering:
- Legislative Calendar → render from BILLS array + Congress.gov API
- Sector Signal Heat → compute from live congressional + EDGAR feed sector counts
- Insider page KPIs → compute from edgarData[]
- Ticker ribbon → render from top-scoring universe tickers
- Notable Exit Signals → render from EDGAR filings with sale/disposition keywords

### Priority 3 — EDGAR XML Parsing
Parse full Form 4 XML to get role (CEO/CFO/Director), transaction type (buy/sell/option exercise), and dollar amounts. This unlocks the full Insider scoring rubric.

### Priority 4 — Institutional Flows Data Pipeline
Build 13F filing parser for tracked smart-money managers. This is the third hub — currently entirely demo data.

### Priority 5 — Full Stack Rebuild (Next.js)
Convert to Next.js 14 App Router when signal generation is proven with real data.
