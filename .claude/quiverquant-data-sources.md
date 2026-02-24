# QuiverQuant API — Available Data Sources for ATLAS
*Based on OpenAPI 3.0.3 spec (document.json). All endpoints use `Authorization: Token YOUR_KEY`.*
*Register at quiverquant.com/quiverapi for a free tier token.*

---

## Authentication
```js
headers: { 'Authorization': 'Token ' + QUIVER_KEY, 'Accept': 'application/json' }
```

---

## Hub 1 — Congressional Trading

| Endpoint | Description | Key Params | ATLAS Use |
|---|---|---|---|
| `GET /beta/live/congresstrading` | Most recent congressional transactions (all members) | `normalized`, `representative` | **Currently wired** — live feed |
| `GET /beta/live/housetrading` | Most recent House rep transactions only | `name`, `options` | Use for chamber filter |
| `GET /beta/live/senatetrading` | Most recent Senate transactions only | `name`, `options` | Use for chamber filter |
| `GET /beta/bulk/congresstrading` | Full history of all congressional transactions | `page`, `page_size`, `ticker`, `date`, `representative` | Per-ticker deep history |
| `GET /beta/historical/congresstrading/{ticker}` | All congressional trades for a specific ticker | `ticker`, `analyst` | Convergence: per-ticker congressional score |
| `GET /beta/historical/housetrading/{ticker}` | House trades for a specific ticker | `ticker` | Chamber-level per-ticker history |
| `GET /beta/historical/senatetrading/{ticker}` | Senate trades for a specific ticker | `ticker` | Chamber-level per-ticker history |

### Sample Response Fields (congresstrading)
```
Representative, Ticker, Transaction (Purchase/Sale), Range, Date, Party, Chamber, State
```

### Scoring Integration (CLAUDE.md)
- Trade size `$250k+` → +5 pts
- Trade size `$1M+` → +10 pts
- Committee with relevant jurisdiction → +10 pts
- 3+ members same ticker/30 days → +15 pts (cluster)

---

## Hub 2 — Insider Trading

| Endpoint | Description | Key Params | ATLAS Use |
|---|---|---|---|
| `GET /beta/live/insiders` | Recent insider transactions (Form 4 equivalent) | `ticker`, `date`, `page`, `page_size`, `limit_codes`, `uploaded` | **Wire next** — replaces EDGAR RSS feed for Hub 2 |

### Sample Response Fields (insiders)
*Likely includes: ticker, filer name/title, transaction type, shares, value, date, plan flag (10b5-1)*

### Scoring Integration (CLAUDE.md)
- CEO → +10, CFO → +8, Director → +6, VP → +4
- No 10b5-1 plan → +8
- Near 52-week low → +5
- 3+ insiders same ticker/72h → +15 (cluster)

### Priority
This is the most important unimplemented endpoint. QuiverQuant insiders replaces/supplements the EDGAR ATOM feed with cleaner JSON, filtering by ticker, and pre-parsed role data.

**Implementation sketch:**
```js
var INSIDER_URL = 'https://api.quiverquant.com/beta/live/insiders';
fetch(INSIDER_URL + '?page_size=50', {
  headers: { 'Authorization': 'Token ' + QUIVER_KEY, 'Accept': 'application/json' }
})
.then(r => r.json())
.then(data => {
  // score each filing per CLAUDE.md logic
  // render into #edgar-feed or insider hub table
});
```

---

## Hub 3 — Institutional Flows (13F)

| Endpoint | Description | Key Params | ATLAS Use |
|---|---|---|---|
| `GET /beta/live/sec13f` | Static 13F portfolio holdings at filing periods | `ticker`, `owner`, `period`, `date`, `today` | Per-manager position lookup |
| `GET /beta/live/sec13fchanges` | 13F changes vs prior period (new/increased/reduced positions) | `ticker`, `owner`, `period`, `most_recent`, `show_new_funds` | **Higher priority** — new positions = strongest signal |

### Scoring Integration (CLAUDE.md)
- Known manager (Berkshire/Druckenmiller/Tepper/Ackman/Pershing) → +15
- New position (not add-to) → +10
- Portfolio allocation >1% → +8

### Usage Note
13F filings are quarterly. Use `most_recent=true` to get latest. Use `show_new_funds=true` to surface new position initiations. Filter `owner` by known smart-money managers for highest signal quality.

---

## Layer 4 — Legislative Catalysts

| Endpoint | Description | ATLAS Use |
|---|---|---|
| `GET /beta/live/legislation` | Recent legislation data (no params) | Cross-reference with BILLS array; timing multiplier (+15) when bill active |

### Note
Congress.gov API is already wired and is the primary bill source. QuiverQuant `/beta/live/legislation` can serve as a cross-reference or fallback.

---

## Signal Amplifiers (Secondary, Future Use)

| Endpoint | Description | Convergence Value |
|---|---|---|
| `GET /beta/live/govcontracts` | Last quarter's gov contract awards for all companies | Defense/GovTech sector signal — amplifies congressional cluster signals in relevant sectors |
| `GET /beta/live/govcontractsall` | Extended gov contracts data | Same as above, more comprehensive |
| `GET /beta/historical/govcontracts/{ticker}` | Contract history for a specific ticker | Per-ticker government revenue trend |
| `GET /beta/live/lobbying` | Recent lobbying spend by company | Companies aggressively lobbying on active bills = timing signal; `date_from`, `date_to` filtering |
| `GET /beta/historical/lobbying/{ticker}` | Lobbying history for a specific ticker | Trend analysis for legislation-sensitive names |
| `GET /beta/live/politicalbeta` | Political beta values for all companies | Sector sensitivity to election/policy risk; useful for position sizing |
| `GET /beta/bulk/politicalbeta` | Full political beta history | Trend analysis |
| `GET /beta/live/offexchange` | Yesterday's off-exchange (dark pool) activity | Unusual dark pool volume → institutional signal; `page`, `page_size` |
| `GET /beta/historical/offexchange/{ticker}` | Off-exchange history for a ticker | Cross-reference with 13F changes |

---

## Niche / Lower Priority

| Endpoint | Notes |
|---|---|
| `GET /beta/live/flights` | Corporate jet flight data — useful for activist situations (confirmed board meetings, M&A travel) |
| `GET /beta/historical/flights/{ticker}` | Flight history for a ticker |
| `GET /beta/live/allpatents` | Recent patent filings | R&D pipeline signal for tech/pharma |
| `GET /beta/historical/allpatents/{ticker}` | Patent history |
| `GET /beta/live/appratings` | App store ratings | Consumer-facing companies only |
| `GET /beta/live/rss` | QuiverQuant RSS feed | General news signal |

---

## Convergence Integration Roadmap

```
Phase 1 (now): Wire /beta/live/congresstrading — DONE (needs QUIVER_KEY token)
               Wire /beta/live/insiders — NEXT, highest ROI for Hub 2

Phase 2: Wire /beta/live/sec13fchanges (filter most_recent + show_new_funds)
         Wire /beta/live/offexchange for dark pool signal

Phase 3: Per-ticker convergence scoring:
         /beta/historical/congresstrading/{ticker} + live insiders + 13f changes
         → scoreFromData(ticker) replaces hardcoded signal scores

Phase 4: Amplifiers
         /beta/live/govcontracts — defense sector signal
         /beta/live/lobbying — pre-floor-vote signal
         /beta/live/politicalbeta — position sizing
```

---

## Free Tier Limits
QuiverQuant free tier: ~100 requests/day. Prioritize:
1. `live/congresstrading` — fetch once on load, re-fetch hourly
2. `live/insiders` — fetch once on load, filter by page_size=50
3. `live/sec13fchanges` — fetch once (quarterly data, no need to poll)
4. `live/offexchange` — daily data, fetch once per session
