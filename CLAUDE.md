# ATLAS Intelligence Platform

## Core Vision
ATLAS is a signal convergence platform. The edge is not having congressional trades, insider buys, and institutional flows on one site — it's detecting when all three point at the same ticker or sector at the same time, especially when relevant legislation is in motion. That convergence is where the highest-conviction trades live.

**No hardcoded trade ideas.** Trade ideas are generated dynamically by the scoring engine when signals align. Static/manual entries are demo scaffolding only and must be replaced.

---

## The Three Signal Hubs (Primary Pages)

### Hub 1 — Congressional Trading
- STOCK Act disclosures: every buy/sell filed by House and Senate members
- Filter by chamber, committee, party, sector, size
- Highlight cluster activity (3+ members, same ticker, 30-day window)
- Link to source disclosure filing

### Hub 2 — Insider Buying
- SEC EDGAR Form 4: executive and director purchases
- Score each filing: role seniority, size vs. compensation, proximity to 52-wk low, no 10b5-1 plan
- Cluster detection: 3+ insiders, same company, 72-hour window = major bonus
- Link to actual SEC filing

### Hub 3 — Institutional Flows
- 13F filings: new positions, conviction increases, full exits
- Track Berkshire, Druckenmiller, Tepper, Ackman, Pershing Square etc.
- Unusual options flow (sweep volume vs. open interest)
- Highlight when a known smart-money manager initiates or adds significantly

### Layer 4 — Legislative Catalysts (Overlay, Not Standalone)
- Bill status, vote dates, passage probability
- Used as a timing multiplier on signals from the three hubs above
- A cluster of insider buys in defense + SB 1882 markup scheduled = signal score multiplied

---

## The Convergence Engine (The Real Edge)
When signals from two or more hubs align on the same ticker or sector within a rolling window:

| Signal Sources Active | Convergence Score Boost |
|---|---|
| Congressional + Insider | +20 pts |
| Congressional + Institutional | +20 pts |
| Insider + Institutional | +15 pts |
| All three | +40 pts |
| Any above + active legislation | +15 additional |

**A ticker with a CEO buying $4M, 3 congress members buying, Druckenmiller initiating, and a floor vote in 10 days = the highest possible conviction signal.**

Trade ideas are surfaced only when convergence score crosses threshold. They are not manually entered.

---

## Current State
- **Main file**: `atlas-intelligence.html` — single-file HTML/CSS/JS dashboard on Vercel
- **Live URL**: https://atlas-intelligence-wheat.vercel.app
- **GitHub**: https://github.com/Hsalazar023/atlas-intelligence (auto-deploys on push)
- **Data**: Finnhub live prices ✅ | Congress.gov live bills ✅ | EDGAR/institutional = fallback/demo data
- **Trade ideas**: Currently demo/hardcoded — must be replaced with engine-generated signals
- **Known bad data**: ITA fallback=$162 (actual ~$243), WFRD fallback=$17.50 (actual ~$104) — prices were wrong from original build

---

## Tracked Bills
SB 1882 (Defense), HR 7821 (AI Infrastructure), HR 6419 (Critical Minerals), SB 2241 (Drug Pricing)

---

## Architecture

### Now — Single HTML file + Finnhub + Congress.gov
Demo-grade. Proves the concept. Replace trade idea cards with engine-generated ones.

### Next (Phase 1) — Python data pipeline
- EDGAR Form 4 poller → SQLite → scoring engine → signals
- Congressional disclosure scraper (House/Senate disclosure portals)
- FastAPI server serving `/signals`, `/congress`, `/institutional`, `/bills`
- HTML frontend calls local API

### Then (Phase 2) — Full stack
- Next.js App Router (one route per hub + convergence dashboard)
- Supabase for signal storage, price history, convergence events
- API routes proxy all external calls (keys never in frontend)
- Vercel deployment (already set up)

---

## Target Stack
| Layer | Tool | Cost |
|---|---|---|
| Frontend | Next.js 14 App Router | Free |
| Hosting | Vercel | Free |
| Database | Supabase (Postgres) | Free tier |
| Auth | Clerk | Free (≤10k users) |
| Data pipeline | Python + APScheduler | Free |
| Stock prices | Finnhub (now) → Polygon.io (later) | Free / $29/mo |
| SEC EDGAR | EDGAR API (Form 4) | Free |
| Congress trades | House/Senate disclosure portals | Free (scraping) |
| Bill tracking | Congress.gov API ✅ | Free |
| Charts | TradingView Lightweight Charts | Free |
| Notifications | Ntfy.sh | Free |
| Email alerts | Resend.com | Free (3k/mo) |

**Running cost at full build: ~$30–50/month**

---

## Development Phases

### Phase 0 — Foundation ✅ COMPLETE
- [x] Finnhub live prices + 60s refresh
- [x] Zone badges on trade cards (IN ZONE / ABOVE ZONE / MISSED)
- [x] Signal date + price stamps on cards
- [x] SMPL added to price strip
- [x] Congress.gov API key wired in
- [x] Deployed to Vercel + GitHub repo
- [x] Price audit — ITA and WFRD fallback prices corrected

### Phase 1 — Real Data Pipelines ✅ MOSTLY COMPLETE
- [x] Congressional disclosure data via QuiverQuant API (data/congress_feed.json)
- [x] EDGAR Form 4 feed via EFTS (data/edgar_feed.json)
- [x] FRED market data (VIX + 10yr yield)
- [x] Replace hardcoded trade ideas with engine-generated signal cards
- [x] Convergence detector: 2+ hubs fire on same ticker = score boost
- [x] Open universe expansion: scores ALL tickers from feeds, not just TRACKED 11
- [x] Signal decay: 21-day half-life on congressional scores
- [x] Backtest engine: collect_prices, event study, weight optimizer (backtest/)
- [x] GitHub Actions: 4x daily data refresh + weekly backtest
- [ ] **BLOCKER:** Historical price data (Finnhub free tier blocks /stock/candle)
- [ ] EDGAR Form 4 XML parsing: extract role (CEO/CFO), amounts, 10b5-1 plan
- [ ] Replace remaining hardcoded sections (see LIVE_DATA_ROADMAP.md)

### Phase 2 — Full Stack Rebuild (Claude Code)
- [x] Vercel deployment live
- [x] GitHub repo with auto-deploy
- [ ] Convert to Next.js 14 App Router
- [ ] Hub pages: `/congressional`, `/insider`, `/institutional`, `/convergence`
- [ ] API routes proxy all external calls (move keys to Vercel env vars)
- [ ] Supabase: `signals`, `congressional_trades`, `form4_filings`, `institutional_positions`, `convergence_events`

### Phase 3 — Convergence Engine ✅ COMPLETE
- [x] Scoring engine for each hub independently (scoreCongressTicker, scoreEdgarTicker)
- [x] Cross-hub convergence detection with boost multipliers (computeConvergenceScore)
- [x] Legislative catalyst overlay (BILLS array + impactTickers)
- [x] Dynamic trade idea generation when convergence score ≥ threshold
- [x] Signal decay: exponential half-life decay on scores
- [x] Self-improving weights via backtest engine (optimize_weights.py)

### Phase 4 — Charts & Visuals
- [ ] TradingView Lightweight Charts: entry/target/stop overlays per signal
- [ ] Convergence heatmap: sectors on X, signal sources on Y, color = score intensity
- [ ] Timeline view: signals plotted chronologically with legislative events as markers

### Phase 5 — Notifications
- [ ] Ntfy.sh push: convergence event detected, ticker enters zone, bill vote within 48h
- [ ] Resend email: full convergence (all 3 hubs + legislation) events only

### Phase 6 — Auth & Monetization
- [ ] Clerk auth (Google/email)
- [ ] Per-user watchlists in Supabase
- [ ] Stripe paywall ($20–50/mo) if monetizing

---

## Signal Scoring Logic

### Per-Hub Scores

**Insider (Form 4):**
| Factor | Points |
|---|---|
| CEO | 10 |
| CFO | 8 |
| Director | 6 |
| VP | 4 |
| Near 52-week low | +5 |
| No 10b5-1 plan | +8 |
| Historical accuracy of this insider | +0–10 |
| Cluster: 3+ insiders, same ticker, 72h | +15 |

**Congressional:**
| Factor | Points |
|---|---|
| Committee with relevant jurisdiction | +10 |
| Cluster: 3+ members, same ticker, 30d | +15 |
| Trade size >$250k | +5 |
| Trade size >$1M | +10 |
| Same-day as relevant bill activity | +10 |

**Institutional:**
| Factor | Points |
|---|---|
| Known smart-money manager (Berkshire/Druckenmiller/etc.) | +15 |
| New position (not add-to) | +10 |
| >1% portfolio allocation | +8 |
| Unusual options sweep | +12 |

### Convergence Boosts
- Congressional + Insider: +20
- Congressional + Institutional: +20
- Insider + Institutional: +15
- All three: +40
- Any convergence + active legislation: +15

**Thresholds (dynamic — tuned by backtest engine via data/optimal_weights.json):**
- Watchlist / Monitoring tier: score 40–64
- Trade idea generated: score ≥ 65 (default, adjustable)
- Exceptional: ≥ 95

**Signal decay:** Congressional scores use exponential decay with a 21-day half-life (configurable).
Older trades contribute fewer points; a 42-day-old trade is worth 25% of a fresh one.

---

## Entry Zone Logic (for generated signals)
- Lower bound: `signal_generation_price × 0.97`
- Upper bound: `signal_generation_price × 1.03`
- Statuses: `in_zone` | `above_zone` (3–10%) | `missed` (>10%) | `triggered` | `stale` (>30 days old)

---

## API Keys
| Key | Source | Env Var | Status |
|---|---|---|---|
| Finnhub | finnhub.io (free) | `FINNHUB_API_KEY` | ✅ in HTML |
| Congress.gov | api.congress.gov (free) | `CONGRESS_API_KEY` | ✅ in HTML |
| Polygon.io | polygon.io ($29/mo) | `POLYGON_API_KEY` | ⬜ future |
| Resend | resend.com (free) | `RESEND_API_KEY` | ⬜ future |

When converting to Next.js: move all keys to Vercel environment variables. Never commit keys to GitHub in that phase.

---

## Conventions
- **Claude Code** for: multi-file work, Python pipelines, package installs, iterative test loops
- **Chat** for: single-file HTML edits, quick fixes, questions
- No hardcoded trade ideas — all signals must be engine-generated or clearly labeled `[DEMO]`
- Prices in `TRACKED` are fallback only; Finnhub live prices always override
- Deploy: `git add . && git commit -m "msg" && git push` — Vercel auto-deploys

---

## Working in This Codebase

### atlas-intelligence.html
- File is ~2600 lines — always use `offset` + `limit` with the Read tool
- `TRACKED` object at line ~1086 is the source of truth for signal data
- `SCORE_WEIGHTS` / `SCORE_THRESHOLD` globals at line ~1104 — loaded from data/optimal_weights.json
- `TICKER_KEYWORDS` at line ~1860 — maps EDGAR company names to tickers
- Key JS functions:
  - `buildTickerUniverse()` — all tickers from TRACKED + congData
  - `computeConvergenceScore(ticker)` — returns {total, congress, insider, boost, hasConvergence}
  - `renderSignalIdeas()` — two-tier: Trade Ideas (≥65) + Monitoring (40–64)
  - `renderTopSignals()` — dynamic top signals table
  - `renderCongressTrades(data)` — QuiverQuant congressional data
  - `renderInsiderTableLive()` — EDGAR filings via TICKER_KEYWORDS
  - `loadOptimalWeights()` — fetches dynamic scoring weights
  - `updateIdeaCard()` (zone badges), `refreshAllPrices()` (Finnhub loop)
- Price strip pattern: `id="ps-TICKER"` inside `#price-strip`
- Initialization: `window.addEventListener('load', ...)` at line ~2600

### backtest/ directory
- `shared.py` — path constants, TICKER_KEYWORDS, DEFAULT_WEIGHTS, helpers
- `collect_prices.py` — Finnhub OHLC cache with incremental updates
- `run_event_study.py` — CAR computation at 5d/30d/90d + member track records
- `optimize_weights.py` — grid search over 1024 weight combinations
- `tests/` — 23 unit tests (pytest)
- Output files: `data/optimal_weights.json`, `data/backtest_results.json`, `data/backtest_summary.json`

### Environment Quirks
- No `sudo` in terminal — `npm install -g` fails; use `npm install -g --prefix ~/.npm-global` instead
- Vercel CLI binary: `~/.npm-global/bin/vercel`
- Live Finnhub prices are CORS-blocked when opening HTML via `file://` — must serve over HTTP
- Local server: `python3 -m http.server 8080` from project dir
- Git committer name/email not globally configured — shows warning on commits but works fine
