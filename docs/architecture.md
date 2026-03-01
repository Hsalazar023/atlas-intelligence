# ATLAS — Architecture & Stack

---

## Current Stack

| Layer | Tool | Status |
|---|---|---|
| Frontend | Single HTML (`atlas-intelligence.html`, ~3400 lines) | ✅ Live |
| Hosting | Vercel (auto-deploy on push) | ✅ Live |
| Stock prices | Finnhub `/quote` (60s refresh) | ✅ Live |
| Congressional data | FMP → `congress_feed.json` | ✅ Live |
| Insider data | SEC EDGAR Form 4 → `edgar_feed.json` | ✅ Live |
| Market data | FRED API → VIX, yield curve | ✅ Live |
| Historical prices | yfinance → `price_history/` | ✅ Live |
| Brain (ALE + ML) | SQLite → `atlas_signals.db` | ✅ Live |
| Brain export | `brain_signals.json` + `brain_stats.json` | ✅ Live |
| Scoring weights | `optimal_weights.json` | ✅ Live |
| Institutional data | 13F pipeline | ❌ Not built |

---

## Key Files

| File | Purpose |
|---|---|
| `atlas-intelligence.html` | Main app (use offset+limit when reading) |
| `scripts/fetch_data.py` | Data fetch pipeline (congress, EDGAR, FRED) |
| `backtest/learning_engine.py` | ALE core: ingest, enrich, score, export, diagnostics |
| `backtest/ml_engine.py` | Walk-forward ML (RF + LightGBM), full-sample training |
| `backtest/shared.py` | Constants, paths, SEC ticker matching |
| `backtest/bootstrap_historical.py` | One-time 39-month historical data load |
| `backtest/backfill_edgar_xml.py` | One-time EDGAR cleanup (parse XML, delete non-buys) |
| `backtest/sector_map.py` | GICS sector + market cap lookup |
| `data/brain_signals.json` | Top 50 ML-scored signals (committed to git) |
| `data/brain_stats.json` | Alpha KPIs, score tiers, sectors (committed to git) |
| `data/optimal_weights.json` | Frontend scoring weights (gitignored) |
| `data/atlas_signals.db` | SQLite DB (gitignored) |

---

## Key JS Functions

| Function | Purpose |
|---|---|
| `loadBrainSignals()` | Fetch brain_signals.json → rebuild TRACKED |
| `loadBrainStats()` | Fetch brain_stats.json → update KPIs, sectors, tiers |
| `computeConvergenceScore(ticker)` | Heuristic score, overridden by Brain totalScore |
| `renderTopSignals()` | Top signals table (uses Brain scores) |
| `renderSignalIdeas()` | Trade Ideas page (uses Brain scores) |
| `refreshConvergenceDisplays()` | Master refresh: calls all render functions |
| `buildTickerUniverse()` | All tickers from TRACKED + feeds |
| `loadOptimalWeights()` | Fetch optimal_weights.json → set SCORE_THRESHOLD |

---

## GitHub Actions

| Workflow | Schedule | Purpose |
|---|---|---|
| Data refresh | 4x daily | Fetch congress/EDGAR/FRED feeds |
| Backtest | Mon-Fri 10 PM UTC | ALE daily pipeline |

---

## API Keys

| Key | Source | Status |
|---|---|---|
| Finnhub | finnhub.io | ✅ in HTML (move to env var) |
| Congress.gov | api.congress.gov | ✅ in HTML (move to env var) |
| FMP | financialmodelingprep.com | ✅ Backend only |
| FRED | fred.stlouisfed.org | ✅ Backend only |
