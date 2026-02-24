# ATLAS — Live Data Roadmap
*Last audited: Feb 24, 2026*

---

## Current State: What's Actually Live vs. Fake

### ✅ Genuinely Live
| Data | Source | Refresh |
|------|--------|---------|
| Stock prices (RTX, NVDA, OXY, TMDX, FCX, PFE, TSM, META, ITA, WFRD, SMPL) | Finnhub REST | 60s |
| Bill status text (latest action, action date) | Congress.gov API | On page load |

### ❌ Hardcoded — Displayed as Current (Misleading)
Everything else on the site is static HTML written in February 2026 and does not update.

**Trade Cards:**
- Entry zones, targets, stops in `rr-block` cells — static HTML, not from TRACKED
- ITA card shows `Entry: $155–$165 / Targets: $182–$196 / Stop: $145` but TRACKED has `$235–$250 / $272–$295 / $220`. Card is actively contradicting itself.
- WFRD card likely has same problem — HTML predates the signal suspension
- Legislative calendar countdowns: "1 day", "7 days" — hardcoded strings, became wrong by Feb 26

**Markets Page:**
- S&P 500: 6,147 | Nasdaq: 21,384 | VIX: 17.4 | 10-yr: 4.42% — static since Feb 2026
- All market breadth stats (A/D ratio, 52-wk highs, put/call, AAII, Fear & Greed, credit spreads) — static
- Sector performance bars and percentages — static
- "ATLAS Signal Alpha +11.4%" — static, made up

**Congressional Intel Page:**
- 9 hardcoded trade rows (Whitfield, Morrison, Chen, Park, Vasquez, Kline, Torres, Harmon, Davis)
- KPI strip (247 filings, 38 high-suspicion, $847M vol, 81 late filers) — static
- Heatmap — static
- Committee correlation bars — static

**Insider Hub:**
- 8 hardcoded insider buy rows, 2 exit rows
- Sector insider activity bars and counts — static
- EDGAR sidebar feed (5 items with yellow "limited results" banner) — fake fallback

**Institutional Flows:**
- KPIs (7 atypical 13Fs, 4 dark pool anomalies, 11 options sweeps, 3 SI spikes) — static
- All flow cards (Berkshire, Druckenmiller, Tepper, options sweeps, dark pool blocks) — static

**Bill tracking (secondary panels):**
- Passage probabilities (85%, 78%, 71%, 52%) — static
- Vote dates in legislative calendar — static, not computed

---

## Priority 0 — Fix Broken Data Today
*No new APIs. Pure JS fixes. ~2–3 hours. These are active contradictions.*

### P0.1 — Fix ITA card `rr-block` HTML
**Problem:** TRACKED has `entryLo:235, entryHi:250, target1:272, target2:295, stop:220` but the HTML card body still shows `$155–$165 / $182–$196 / $145` — the original stale signal. The dynamic zone badge correctly says "ZONE MISSED" but the entry/target cells above it show the wrong zone. User sees contradictory data.

**Fix:** Update the ITA card `rr-block` HTML directly to match TRACKED. Also mark the card with a "SIGNAL STALE — do not act" banner in the HTML itself.

### P0.2 — Fix WFRD card `rr-block` HTML
**Problem:** Same issue. Card HTML predates the WFRD signal suspension. TRACKED was updated but HTML was not.

**Fix:** Update `rr-block` cells + add "SIGNAL SUSPENDED" banner in the card HTML.

### P0.3 — Make legislative calendar countdowns computed
**Problem:** "Feb 25, 2026 — 1 day" is a hardcoded string. By Feb 26 it reads "1 day" forever.

**Fix:** Replace hardcoded day-count strings with JS that computes `Math.ceil((targetDate - now) / 86400000)` at page load. ~10 lines.

### P0.4 — Fix fallback price label
**Problem:** `updateIdeaCard()` hardcodes `'As of Feb 24, 2026'` as the label when Finnhub is unavailable. This string becomes wrong the next day and implies the price shown is current.

**Fix:** Change to `'Fallback price — not live'` and add a note to verify before acting.

---

## Priority 1 — Wire Free APIs That Already Exist
*Each is a 1–3 hour task. No Python required. Each makes one section meaningfully live.*

### P1.1 — Live Market Indices (Finnhub)
**API:** Finnhub already wired. Add these tickers to the price feed:
- `SPY` (S&P 500 proxy) or use Finnhub index quote endpoint
- `QQQ` (Nasdaq proxy)
- `VIX` — Finnhub supports `^VIX`
- `^TNX` — 10-year Treasury yield

**What gets live:** Markets page top KPI strip, replacing static 6,147 / 21,384 / 17.4 / 4.42%.

**Note:** Finnhub free tier supports index symbols. Test `^VIX` and `^TNX` first — these sometimes require a plan upgrade. If blocked, Yahoo Finance CORS proxy is an alternative.

### P1.2 — Live Congressional Trades (QuiverQuant)
**API:** `https://api.quiverquant.com/beta/bulk/congresstrading`
- Free tier: 100 req/day, no charge
- Register at quiverquant.com → API keys → free tier
- Returns: member name, ticker, transaction type, amount, date, committee, party
- No scraping required, clean JSON

**What gets live:** The entire congressional trading table (currently 9 hardcoded rows). This is Hub 1.

**Implementation:**
```js
var QUIVER_KEY = 'your_key_here';
fetch('https://api.quiverquant.com/beta/bulk/congresstrading', {
  headers: { 'X-CSRFToken': QUIVER_KEY, 'Authorization': 'Token ' + QUIVER_KEY }
})
```
Parse the response, score each trade using the congressional scoring logic in CLAUDE.md, render into `#cong-tbl` tbody.

**Fallback:** Keep current hardcoded rows if API call fails.

### P1.3 — Live EDGAR Form 4 Feed (EDGAR ATOM — No Key Required)
**API:** `https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&owner=include&count=40&output=atom`

- Free, no key, no auth
- Returns last 40 Form 4 filings as Atom XML, updates in real time
- Parse with DOMParser in browser

**What gets live:** The EDGAR sidebar feed (currently 5 fake items with warning banner). Insider table on Hub 2 still needs scoring logic.

**CORS note:** EDGAR has permissive CORS headers — fetch works from browser without a proxy.

**Implementation sketch:**
```js
function fetchEdgarFeed() {
  fetch('https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&owner=include&count=40&output=atom')
    .then(r => r.text())
    .then(xml => {
      var doc = new DOMParser().parseFromString(xml, 'text/xml');
      var entries = doc.querySelectorAll('entry');
      // parse title (contains company + insider), updated date, link
      // render into #edgar-feed
    });
}
```

### P1.4 — Live 10-Year Treasury Yield (FRED API)
**API:** `https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key=YOUR_KEY&limit=1&sort_order=desc&file_type=json`
- FRED API key: free, register at fred.stlouisfed.org
- Returns most recent 10-yr Treasury yield
- Updates daily (not intraday, but not stale)

**What gets live:** 10-yr yield on Markets page. More important: this is used in some signal R/R calculations for rate-sensitive plays (PFE short thesis).

### P1.5 — Bill Vote Countdown Logic (No new API)
**What gets live:** The legislative calendar section and bill cards update their countdown automatically.

**Fix:** At page load (and setInterval refresh), recalculate days-until for each bill date:
```js
var billDates = {
  'HR7821': new Date('2026-02-25'),
  'SB1882': new Date('2026-03-03'),
  'SB2241': new Date('2026-03-10'),
  'HR6419': new Date('2026-03-17')
};
```
Render `"N days"` dynamically. Add `PAST` badge if date has passed.

---

## Priority 2 — Make Card Data Fully Dynamic
*Prevents the ITA-style contradiction from happening again. ~4–6 hours.*

### P2.1 — Render `rr-block` from TRACKED, not HTML
**Problem:** Entry zones, targets, and stops are hardcoded in each card's HTML AND in TRACKED. When one is updated, the other falls out of sync (exactly what happened with ITA).

**Fix:** Extend `updateIdeaCard()` to also update the `rr-block` cells:
```js
// inside updateIdeaCard(), after zone badge injection:
var rrEntry = card.querySelector('.rr-cell:nth-child(1) .rr-val');
if (rrEntry) rrEntry.textContent = '$' + info.entryLo + '–$' + info.entryHi;
var rrStop = card.querySelector('.rr-cell:nth-child(3) .rr-val');
if (rrStop) rrStop.textContent = '$' + info.stop;
```
This way TRACKED is the single source of truth. No more HTML/TRACKED divergence.

### P2.2 — Signal staleness detection
**Logic:** If `sigDate` is more than 30 days ago, or if price has moved >10% above the entry hi (for longs), automatically add a `STALE` banner to the card.

```js
var sigAge = (Date.now() - new Date(info.sigDate)) / 86400000;
if (sigAge > 30) { /* inject stale banner */ }
```

### P2.3 — Score cards per-ticker from live Congress.gov + EDGAR data
Once P1.2 (QuiverQuant) and P1.3 (EDGAR) are live, build a simple `scoreFromData(ticker)` function:
- If ticker appears in congressional trades within 30 days → add congressional score
- If ticker appears in recent Form 4 buys → add insider score
- If both → apply convergence boost
- If active bill → apply timing multiplier

This replaces hardcoded scores (91, 94, 88...) with computed ones. First step toward the real convergence engine.

---

## Priority 3 — Python Pipeline (Phase 1)
*Required for real-time insider + congressional data at institutional quality.*

See CLAUDE.md Phase 1. Summary:
1. `edgar_poller.py` — fetch EDGAR Form 4 every 90s, parse XML, extract role/value/plan flag, score, write to SQLite
2. `congress_scraper.py` — QuiverQuant or house.gov/senate.gov disclosure portals, deduplicate, score
3. `api_server.py` — FastAPI serving `/signals`, `/congress`, `/insider`, `/bills`
4. Frontend calls `localhost:8000` → works locally; Vercel calls Supabase in production

**Trigger condition:** Build Phase 3 Python pipeline when QuiverQuant (P1.2) and EDGAR RSS (P1.3) have been running for 1 week and you understand the data shape.

---

## Priority 4 — Market Context Data
*Makes the Markets page useful. Lower priority than signal data.*

| Data Point | Source | Cost |
|---|---|---|
| S&P 500, Nasdaq, Russell intraday | `SPY`, `QQQ`, `IWM` via Finnhub | Free |
| VIX | `^VIX` via Finnhub | Free (test first) |
| 10-yr yield | FRED API `DGS10` | Free |
| Put/Call ratio | CBOE free data download (daily) | Free |
| AAII sentiment | AAII.com (weekly scrape, no official API) | Free |
| Fear & Greed | CNN has no public API — scrape or skip | Free / brittle |
| Market breadth (A/D, 52wk H/L) | Polygon.io `$29/mo` or skip | Paid |

**Recommendation:** Wire SPY/QQQ/IWM/VIX via Finnhub (same code as existing prices). Skip breadth stats for now — they go stale intraday and we don't have a free source. Replace them with a "Market context is updated daily" disclaimer until Phase 3.

---

## Implementation Order (Recommended)

```
Week 1:
  [x] P0.1 Fix ITA rr-block HTML
  [x] P0.2 Fix WFRD rr-block HTML
  [x] P0.3 Computed countdown timers
  [x] P0.4 Fix fallback price label
  [x] P1.1 Live index prices via Finnhub (SPY, QQQ, IWM, ^VIX)
  [x] P1.5 Bill countdown logic (data-bill-date attributes, computed dynamically)

Week 2:
  [x] P1.2 QuiverQuant congressional feed — QUIVER_KEY wired, 150 live trades
  [x] P1.3 EDGAR Form 4 feed — Python fetch_data.py + data/edgar_feed.json
  [x] P1.4 FRED 10-yr yield — FRED_KEY wired, DGS10 live
  [x] P2.1 rr-block rendered from TRACKED — updateIdeaCard() syncs cells on every price refresh
  [x] P2.2 Signal staleness auto-detection — ⏰ warning injected when sigDate >30 days old

Week 3+:
  [ ] P2.3 Computed per-ticker scores from live data (congress + EDGAR cross-reference)
  [ ] GitHub Actions workflow to auto-run fetch_data.py every 4h (auto-commit JSON)
  [ ] Begin Phase 1 Python pipeline (FastAPI server for local dev)
```

---

## Free API Keys Needed (Not Yet Registered)
| Service | Where to Register | Cost | Priority |
|---|---|---|---|
| QuiverQuant | quiverquant.com/quiverapi | Free tier (100 req/day) | P1.2 |
| FRED | fred.stlouisfed.org/docs/api/api_key.html | Free | P1.4 |
| Polygon.io | polygon.io | $29/mo | Phase 2+ |

---

## What "Accurate" Means for Each Data Type

| Data | Accurate = | Not acceptable |
|---|---|---|
| Stock prices | Finnhub live, ≤60s lag | Hardcoded with no timestamp |
| Entry/target/stop | Computed from signal generation price in TRACKED | HTML that diverges from TRACKED |
| Congressional trades | QuiverQuant ≤24hr lag | Hardcoded fictional rows |
| Insider filings | EDGAR RSS ≤15min lag | 5-item fake fallback list |
| Bill status | Congress.gov API (already live) | Hardcoded passage % with no update |
| Bill vote countdown | Computed from actual date | Hardcoded "N days" string |
| Market indices | Finnhub live (SPY/QQQ proxy) | Hardcoded to a single day's close |
| Institutional 13F | Quarterly filing lag is unavoidable — label clearly | Presented as current when months old |
