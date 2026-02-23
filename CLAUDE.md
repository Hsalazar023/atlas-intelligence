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

### Phase 1 — Real Data Pipelines (Claude Code — do next)
- [ ] Fix ITA + WFRD demo cards: update to current prices, mark original signal as stale
- [ ] Replace hardcoded trade ideas with engine-generated signal cards
- [ ] Congressional disclosure scraper (quiverquant.com API or house.gov/senate.gov portals)
- [ ] EDGAR Form 4 poller: parse XML, extract role/value/plan, score each filing
- [ ] Convergence detector: when 2+ hubs fire on same ticker within rolling window
- [ ] FastAPI local server: `/signals`, `/congress`, `/institutional`, `/bills`

### Phase 2 — Full Stack Rebuild (Claude Code)
- [x] Vercel deployment live
- [x] GitHub repo with auto-deploy
- [ ] Convert to Next.js 14 App Router
- [ ] Hub pages: `/congressional`, `/insider`, `/institutional`, `/convergence`
- [ ] API routes proxy all external calls (move keys to Vercel env vars)
- [ ] Supabase: `signals`, `congressional_trades`, `form4_filings`, `institutional_positions`, `convergence_events`

### Phase 3 — Convergence Engine
- [ ] Scoring engine for each hub independently
- [ ] Cross-hub convergence detection with boost multipliers
- [ ] Legislative catalyst overlay (bill status × signal timing)
- [ ] Dynamic trade idea generation when convergence score ≥ threshold
- [ ] Signal decay: reduce score for older filings, flag stale ideas

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

**Generate trade idea when total convergence score ≥ 85. Exceptional ≥ 95.**

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
- File is ~2000 lines / 68k tokens — always use `offset` + `limit` with the Read tool
- `TRACKED` object at line ~1362 is the source of truth for signal data; `data-entry-lo`/`data-entry-hi` HTML attributes on `.idea-card` elements are stale and unused — ignore them
- Key JS functions: `updateIdeaCard()` (zone badges + stamps), `refreshAllPrices()` (Finnhub loop), `fetchPrice()`, `checkPriceAlert()`, `seedPriceCache()`
- Price strip pattern: `id="ps-TICKER"` inside `#price-strip` — one element per tracked ticker
- Initialization block: `window.addEventListener('load', ...)` at line ~1975

### Environment Quirks
- No `sudo` in terminal — `npm install -g` fails; use `npm install -g --prefix ~/.npm-global` instead
- Vercel CLI binary: `~/.npm-global/bin/vercel`
- Live Finnhub prices are CORS-blocked when opening HTML via `file://` — must serve over HTTP
- Local server: `python3 -m http.server 8080` from project dir
- Git committer name/email not globally configured — shows warning on commits but works fine
