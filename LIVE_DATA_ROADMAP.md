# ATLAS — Live Data Roadmap
*Updated Feb 27, 2026*

## Current State Summary

### What's LIVE (dynamically rendered from real data)
| Section | Data Source | Refresh Rate |
|---|---|---|
| Top Signals table | Convergence engine scoring all universe tickers | On data load |
| Trade Ideas / Watchlist | `renderSignalIdeas()` — score ≥ 65 / 40–64 | On data load |
| Congressional Trading table | QuiverQuant + FMP → `data/congress_feed.json` + `data/fmp_congress_feed.json` | 4x daily |
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
- [x] backtest/shared.py — constants, helpers, SEC ticker matching
- [x] backtest/collect_prices.py — yfinance OHLC cache (free, no API key)
- [x] backtest/run_event_study.py — CAR at 5d/30d/90d + member track records
- [x] backtest/optimize_weights.py — grid search for optimal weights
- [x] 51 unit tests, all passing
- [x] GitHub Actions: daily backtest Mon-Fri 10 PM UTC
- [x] Frontend: loadOptimalWeights() + dynamic decay half-life + backtest stats bar

### Adaptive Learning Engine v1 ✅ (Feb 26, 2026)
- [x] SQLite signals database (`data/atlas_signals.db`) — signals, feature_stats, weight_history tables
- [x] SEC ticker matching: 10,000+ company→ticker from SEC's `company_tickers.json`
- [x] Daily pipeline: collect new signals → backfill outcomes → person track records → feature analysis
- [x] 5 outcome horizons: 5d, 30d, 90d, 180d, 365d (returns + CARs)
- [x] Person track records: per-representative and per-insider hit rates, avg CARs, relative position sizing
- [x] Feature analysis: source, trade_size, convergence, person_experience, person_accuracy, relative_position
- [x] Auto-weight generation from feature stats → `data/optimal_weights.json`
- [x] ALE dashboard: `data/ale_dashboard.json` with person leaderboards
- [x] Historical bootstrap: ~21 months of EDGAR + congressional data, quality-filtered (ticker-matched only)
- [x] Daily EDGAR ingest uses enriched XML fields (ticker from XML, buy-only filter, role, buy_value)
- [x] `--summary` CLI with person leaderboard display

### ALE v2 — Self-Improving ML Engine ✅ (Feb 27, 2026)
- [x] FMP congressional data pipeline: Senate + House trades via Financial Modeling Prep API
- [x] Sector tagging: GICS sector labels on all signals (`backtest/sector_map.py`)
- [x] Multi-tier convergence: Tier 0 (none), Tier 1 (same ticker, 2+ sources, 60d), Tier 2 (sector-level, 3+ signals)
- [x] Cluster velocity: burst/fast/moderate/slow classification
- [x] Insider role normalization: CEO/CFO/COO/VP/President/Director/Officer/Other
- [x] Research-backed features: 52-week proximity, opportunistic vs routine classification (Cohen et al. 2012)
- [x] Walk-forward ML engine: RF + LightGBM ensemble (`backtest/ml_engine.py`)
- [x] Information Coefficient metric (Spearman rank correlation)
- [x] Safety rail: weights auto-update only when OOS IC improves >5%
- [x] ML wired into `--analyze` pipeline with dashboard integration
- [x] GitHub Actions updated: ML deps, FMP fetch step, new data files
- [x] 124 unit tests, all passing
- [ ] **NEXT:** Add FMP_API_KEY as GitHub secret → bootstrap historical congress data

### Resolved: Historical Price Data ✅ (Feb 26, 2026)

Switched from Finnhub (paid-only `/stock/candle`) to **yfinance** (free, no API key). Pipeline now fetches 1000 days of daily OHLC for all universe tickers + SPY benchmark.

### Resolved: EDGAR Bootstrap Depth ✅ (Feb 26, 2026)

EDGAR EFTS returns ~1,000+ Form 4 filings per day. Original approach (flat 5,000 cap) only fetched ~5 business days. Fixed by:
- Monthly chunking: fetch month-by-month going backwards
- Oversampling: fetch 1,500 raw per month, filter to 500 ticker-matched (quality filtering)
- Extended range: 365 days → 635 days (~21 months) so older signals have full 180d/365d outcome data

### Resolved: EDGAR Daily Ingest ✅ (Feb 26, 2026)

Daily `ingest_edgar_feed()` was ignoring enriched XML fields from `fetch_data.py`. Fixed to:
- Use XML-extracted ticker directly (instead of fallback company name matching)
- Filter to `direction='buy'` only (sells are noise for signal generation)
- Extract insider role from `title`/`roles` fields
- Map `buy_value` to `trade_size_points` using same dollar-tier brackets as congressional trades

---

## Next Priorities

### Priority 1 — Add FMP API Key + Bootstrap Historical Congress Data
1. Sign up at https://financialmodelingprep.com ($15/mo starter)
2. Add `FMP_API_KEY` as GitHub repository secret: `https://github.com/Hsalazar023/atlas-intelligence/settings/secrets/actions`
3. Run `FMP_API_KEY=your_key python scripts/fetch_data.py` locally to verify
4. Run bootstrap: `FMP_API_KEY=your_key python backtest/bootstrap_historical.py` (pulls historical congress trades)
5. Run `python backtest/learning_engine.py --daily` to ingest the new data
6. Run `python backtest/learning_engine.py --analyze` to trigger ML training with expanded data
7. Optionally bootstrap sector map: `FMP_API_KEY=your_key python backtest/sector_map.py --bootstrap`

### Priority 2 — Replace Remaining Hardcoded Sections
Convert static HTML sections to dynamic rendering:
- Legislative Calendar → render from BILLS array + Congress.gov API
- Sector Signal Heat → compute from live congressional + EDGAR feed sector counts
- Insider page KPIs → compute from edgarData[]
- Ticker ribbon → render from top-scoring universe tickers
- Notable Exit Signals → render from EDGAR filings with sale/disposition keywords

### Priority 3 — Update Frontend Convergence Scoring
Wire multi-tier convergence (Tier 1/2) into `computeConvergenceScore()` in atlas-intelligence.html. Display convergence tier badges on signal cards.

### Priority 4 — Institutional Flows Data Pipeline
Build 13F filing parser for tracked smart-money managers. This is the third hub — currently entirely demo data.

### Priority 5 — Full Stack Rebuild (Next.js)
Convert to Next.js 14 App Router when signal generation is proven with real data.
