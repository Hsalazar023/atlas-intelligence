# ATLAS — Architecture & Stack
*Reference doc. Update when infrastructure or stack decisions change.*

---

## Current State

| Layer | Tool | Status |
|---|---|---|
| Frontend | Single HTML file (`atlas-intelligence.html`) | ✅ Live |
| Hosting | Vercel (auto-deploy on git push) | ✅ Live |
| Stock prices | Finnhub `/quote` API (60s refresh) | ✅ Live |
| Congressional data | QuiverQuant + FMP → `data/congress_feed.json` | ✅ Live |
| Insider data | SEC EDGAR EFTS + XML → `data/edgar_feed.json` | ✅ Live |
| Market data | FRED API → VIX + 10yr yield | ✅ Live |
| Historical prices | yfinance (free, no key) → `data/price_history/` | ✅ Live |
| Learning engine | SQLite ALE → `data/atlas_signals.db` | ✅ Live |
| Scoring weights | Backtest engine → `data/optimal_weights.json` | ✅ Live |
| Institutional data | 13F pipeline | ❌ Not built — demo only |

---

## Target Stack (Phase 2+)

| Layer | Tool | Cost |
|---|---|---|
| Frontend | Next.js 14 App Router | Free |
| Hosting | Vercel | Free |
| Database | Supabase (Postgres) | Free tier |
| Auth | Clerk | Free (≤10k users) |
| Data pipeline | Python + APScheduler | Free |
| Stock prices | Finnhub → Polygon.io (later) | Free / $29/mo |
| SEC EDGAR | EDGAR API (Form 4) | Free |
| Congress trades | FMP API (Senate + House) | $15/mo |
| Bill tracking | Congress.gov API | Free |
| Charts | TradingView Lightweight Charts | Free |
| Notifications | Ntfy.sh | Free |
| Email alerts | Resend.com | Free (3k/mo) |

**Target running cost at full build: ~$30–50/month**

---

## GitHub Actions Schedule

| Workflow | Schedule | Purpose |
|---|---|---|
| Data refresh | 4x daily | Fetch congress/EDGAR/FRED feeds |
| Backtest | Mon-Fri 10 PM UTC | ALE daily pipeline + weight update |

---

## Key File Locations

| File | Purpose |
|---|---|
| `atlas-intelligence.html` | Main app — ~2600 lines, use offset+limit when reading |
| `scripts/fetch_data.py` | Data fetch pipeline (congress, EDGAR, FRED) |
| `backtest/learning_engine.py` | ALE — daily pipeline, feature analysis, weight generation |
| `backtest/ml_engine.py` | Walk-forward ML engine (RF + LightGBM) |
| `backtest/shared.py` | Path constants, SEC ticker matching, DEFAULT_WEIGHTS |
| `backtest/bootstrap_historical.py` | One-time 21-month historical data load |
| `backtest/sector_map.py` | GICS sector tagging |
| `data/optimal_weights.json` | Live scoring weights loaded by frontend |
| `data/atlas_signals.db` | ALE SQLite database |
| `vercel.json` | Vercel deployment config |
| `.github/workflows/` | GitHub Actions CI/CD |

---

## Key JS Functions (atlas-intelligence.html)

| Function | Purpose |
|---|---|
| `buildTickerUniverse()` | All tickers from TRACKED + congress feed |
| `computeConvergenceScore(ticker)` | Returns `{total, congress, insider, boost, hasConvergence}` — all integers |
| `scoreCongressTicker(ticker, days)` | Hub 1 score (0-40), decay-adjusted, rounded |
| `scoreEdgarTicker(ticker, days)` | Hub 2 score (0-40), role/size bonuses, rounded |
| `renderSignalIdeas()` | Two-tier: Trade Ideas (≥65) + Monitoring (40–64) |
| `renderTopSignals()` | Dynamic top signals table |
| `renderCongressTrades(data)` | QuiverQuant congressional feed |
| `renderInsiderTableLive()` | EDGAR filings via SEC ticker matching |
| `loadOptimalWeights()` | Fetches `data/optimal_weights.json`, sets SCORE_THRESHOLD |
| `refreshAllPrices()` | Finnhub 60s price loop |
| `updateIdeaCard()` | Zone badge updates (IN ZONE / ABOVE / MISSED) |
| `fmtScore(s)` | Format score as integer (Math.round) |
| `fmtPct(p)` | Format percentage with max 2 decimals, sign prefix |

Line references: `TRACKED` ~1086 · `SCORE_WEIGHTS/THRESHOLD` ~1104 · `TICKER_KEYWORDS` ~1860 · `window.addEventListener('load')` ~2600

---

## API Keys

| Key | Source | Env Var | Status |
|---|---|---|---|
| Finnhub | finnhub.io | `FINNHUB_API_KEY` | ✅ in HTML |
| Congress.gov | api.congress.gov | `CONGRESS_API_KEY` | ✅ in HTML |
| FMP | financialmodelingprep.com ($15/mo) | `FMP_API_KEY` | ⬜ Add as GitHub secret |
| Polygon.io | polygon.io ($29/mo) | `POLYGON_API_KEY` | ⬜ Future |
| Resend | resend.com | `RESEND_API_KEY` | ⬜ Future |

*When converting to Next.js: move all keys to Vercel environment variables. Never commit keys.*
