# ATLAS — Decisions & Notes

## Platform Assessment
Design, signal framework, trade idea format, and filtering logic are well-conceived. The only missing piece is live data. Once real Form 4s and real prices flow through the scoring engine, this becomes something people would pay for.

---

## Key Decisions & Rationale

**Start with Path A, not Path B**
Don't rewrite to Next.js until Phase 1 live data is working. Prove the concept in the single HTML file first. Premature framework migration risks breaking a working UI.

**Finnhub (free) before Polygon ($29/mo)**
Finnhub's free tier covers real-time quotes for all tracked tickers. Upgrade to Polygon only when the platform is live and the richer data (options flow, historical OHLCV) is actually needed.

**Ntfy.sh over Twilio for notifications**
Zero cost, zero configuration for personal push alerts. Ntfy requires no phone number setup — just install the app and subscribe to a channel. Twilio adds complexity and per-SMS cost before it's warranted.

**Supabase over self-hosted Postgres**
Free tier (500MB storage, 2GB bandwidth/mo) is sufficient for personal use. Managed hosting eliminates DevOps burden at this stage.

**Clerk over custom auth**
Auth is a solved problem. Don't build it. Clerk free tier supports 10k users. Only relevant when sharing the platform with others.

**SQLite (`signals.db`) for local Phase 1 pipeline**
Before committing to Supabase, use SQLite locally so the Python scripts can be developed and tested without a network dependency. Migrate to Supabase in Phase 3.

---

## Monetization Path (if pursued)
- Keep personal: $0/mo
- Invite-only group: Clerk invite restrictions
- Paid access: $20–50/mo via Stripe; Claude Code can scaffold a paywall in one session

---

## Phase 0 Status
- [x] Finnhub API key wired in (`FINNHUB_KEY` at line 1358)
- [x] Live price fetch + 60s refresh active
- [x] Zone badges (IN ZONE / ABOVE ZONE / MISSED) on every trade card
- [x] SMPL added to price strip
- [x] "Signal generated [date] at $[price]" stamp added to every card via `updateIdeaCard()`
- [x] Congress.gov API key added — `CONGRESS_API_KEY` line ~1726
- [ ] Audit TRACKED fallback prices (currently Feb 22, 2026 — update when Finnhub unavailable)
- [ ] Cross-reference all entry zones and stops against Finviz/Barchart for accuracy

## Serving ATLAS locally
```bash
cd /Users/henrysalazar/Desktop/Atlas
python3 -m http.server 8080
```
Open: http://localhost:8080/atlas-intelligence.html
Live prices only work over HTTP (not file://). Permanent fix: deploy to Vercel.

---

## Useful Reference Links
- SEC EDGAR Form 4 RSS: `https://efts.sec.gov/LATEST/search-index?q=%22form+4%22&dateRange=custom&startdt=TODAY`
- Congress.gov API: `https://api.congress.gov`
- TradingView Lightweight Charts: `npm install lightweight-charts`
- Ntfy.sh push syntax: `requests.post("https://ntfy.sh/[channel]", data="message")`

---

## Future Features (backlog)
- Universal search bar filtering across all pages simultaneously
- Signal history + backtesting table (signal date → score → 30d return → 90d return)
- News feed sidebar via NewsAPI for tracked tickers
- Options chain viewer for tickers with sweep signals (Tradier free tier)
- One-click PDF export of trade ideas as research notes
- Correlation matrix showing which tickers share signal sources
- Full calendar view of legislative dates, earnings, FDA decisions
- Dark mode toggle (dark navy terminal aesthetic)
