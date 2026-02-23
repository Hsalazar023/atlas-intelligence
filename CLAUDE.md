# ATLAS Intelligence Platform

## What This Is
A financial intelligence dashboard aggregating SEC Form 4 insider trades, congressional stock activity, legislative catalysts, and technical setups into a composite signal scoring engine.

## Current State
- **Main file**: `atlas-intelligence.html` — single-file HTML/CSS/JS dashboard
- **Data**: All static (no live prices, no real-time signals yet)
- **Status**: UI/UX complete; live data layer is the critical missing piece

## Tracked Tickers
RTX, NVDA, OXY, TMDX, FCX, TSM, PFE, META, ITA, WFRD, SMPL

## Tracked Bills
SB 1882, HR 7821, HR 6419, SB 2241

---

## Architecture

### Now (Path A) — Static Site + APIs
Keep `atlas-intelligence.html` as frontend. Add `fetch()` calls to live APIs. Host free on Vercel.

### Later (Path B) — Full Stack
Next.js frontend → Python data pipeline → Supabase → Clerk auth. Graduate here after live data is proven.

---

## Target Stack
| Layer | Tool | Cost |
|---|---|---|
| Frontend | Next.js (after Phase 1) | Free |
| Hosting | Vercel | Free |
| Database | Supabase (Postgres) | Free tier |
| Auth | Clerk | Free (≤10k users) |
| Data pipeline | Python + APScheduler | Free |
| Stock prices | Finnhub → Polygon.io | Free / $29/mo |
| SEC data | SEC EDGAR API | Free |
| Congress data | Congress.gov API | Free |
| Charts | TradingView Lightweight Charts | Free |
| Notifications | Ntfy.sh | Free |
| Email alerts | Resend.com | Free (3k/mo) |

**Running cost at full build: ~$30–50/month (mostly Polygon.io)**

---

## Development Phases

### Phase 0 — Fix What's Broken ✅ COMPLETE (except 2 items)
- [x] Add live Finnhub price validation to trade idea cards — IN ZONE / ABOVE ZONE / MISSED badges
- [x] Replace static ticker bar prices with live quotes — mh-data-strip below ticker scrollbar
- [x] Add "Signal generated [date] at $[price]" stamp to every idea card — rendered in `.price-vs-zone`
- [x] Fix SMPL missing from price strip — `id="ps-SMPL"` element added
- [x] Wire 60-second price refresh interval — `setInterval(refreshAllPrices, 60000)`
- [x] Congress.gov API key added — `CONGRESS_API_KEY` line ~1726
- [ ] **NEXT:** Audit all entry zones, targets, stops against current prices (Finviz/Barchart) + update `TRACKED` fallback prices

### Phase 1 — Live Data Foundation (Claude Code)
- [x] Wire Finnhub prices throughout; refresh every 60s ← done in Phase 0
- [ ] Poll SEC EDGAR Form 4 RSS feed every 90s; parse XML; save to `signals.db` (SQLite)
- [ ] Track bill status via Congress.gov API; write to `bills.json`
- [ ] FastAPI local server: `GET /prices`, `GET /signals`, `GET /bills` with CORS headers

### Phase 2 — Real Website (Claude Code + Vercel)
- [ ] Convert to Next.js 14 App Router (each tab → page route; CSS → `globals.css`)
- [ ] API routes in `/api` folder to proxy Finnhub/EDGAR (never expose keys in frontend)
- [ ] Deploy to Vercel; add env vars in Vercel dashboard

### Phase 3 — Database & Signal Engine (Claude Code)
- [ ] Supabase tables: `signals`, `form4_filings`, `bills`, `price_history`
- [ ] Scoring engine (see logic below); save to Supabase; alert if score ≥ 85
- [ ] Entry zone recalculation loop every 60s

### Phase 4 — Charts & Visuals
- [ ] TradingView Lightweight Charts per trade card: entry band, targets, stop, signal date marker, volume
- [ ] Sector rotation bubble chart (Chart.js): signal strength vs. forward return, bubble size = signal count
- [ ] Score sparklines: 7-day trend line next to each score

### Phase 5 — Notifications
- [ ] Ntfy.sh push: score ≥ 85, ticker enters entry zone, stop approached within 2%, bill vote within 48h
- [ ] Resend email: score ≥ 90 and cluster events

### Phase 6 — Auth & Multi-User (only if sharing/monetizing)
- [ ] Clerk auth (Google + email); protect all routes
- [ ] Per-user watchlists in Supabase linked to Clerk user ID
- [ ] Stripe paywall ($20–50/mo) if monetizing

---

## Signal Scoring Logic
| Factor | Points |
|---|---|
| CEO purchase | 10 |
| CFO purchase | 8 |
| Director purchase | 6 |
| VP purchase | 4 |
| Purchase near 52-week low | bonus |
| No 10b5-1 plan | bonus |
| Historical insider accuracy | bonus |
| Cluster: 3+ insiders, same ticker, 72h window | +15 |

- Alert threshold: **score ≥ 85**
- Exceptional threshold: **score ≥ 90**

## Entry Zone Logic
- Lower bound: `signal_price × 0.97`
- Upper bound: `signal_price × 1.03`
- Statuses: `in_zone` | `above_zone` (3–10% above) | `missed` (>10% above) | `triggered`

---

## API Keys (never hardcode — use env vars)
| Key | Source | Env Var |
|---|---|---|
| Finnhub | finnhub.io (free) | `FINNHUB_API_KEY` |
| Congress.gov | api.congress.gov (free) | `CONGRESS_API_KEY` ✅ |
| Polygon.io | polygon.io ($29/mo) | `POLYGON_API_KEY` |
| Resend | resend.com (free tier) | `RESEND_API_KEY` |

---

## Conventions
- Use **Claude Code** for multi-file work, package installs, iterative scripting, and anything requiring a test loop
- Use **chat interface** only for single-file edits and quick one-shot fixes
- Reference env vars as `process.env.KEY_NAME` (Next.js) or `os.environ["KEY_NAME"]` (Python)
- Keep Phase 0 and Phase 1 changes inside `atlas-intelligence.html` until Next.js conversion
