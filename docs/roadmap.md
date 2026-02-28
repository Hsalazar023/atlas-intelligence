# ATLAS — Long-Term Roadmap
*Active (unchecked) items only. Completed phases live in docs/archive/completed-milestones.md.*
*Update this file as phases complete or priorities shift.*

---

## Phase 0 — Data Quality (Current)
> Clean data is the foundation. Brain can't learn from noise.

- [x] Fix CAR winsorization, market_cap_bucket, sector_avg_car fill rates
- [ ] EDGAR XML backfill — delete non-purchases, enrich real buys
- [ ] Re-run `--analyze` on clean data, validate OOS IC improvement
- [ ] Recompute convergence + person track records on clean data

## Phase 1 — Brain Integration
> Replace every hardcoded value on the site with Brain-generated data.

- [ ] Add `--export` command → generates `brain_signals.json` + `brain_stats.json`
- [ ] Replace TRACKED object with Brain signals
- [ ] Replace hardcoded score tier returns with ALE backtest stats
- [ ] Replace committee correlation, heatmap, sector stats with computed values
- [ ] Wire convergence tier badges into frontend
- [ ] Replace ticker ribbon with top Brain signals

## Phase 2 — New Data Sources
> Feed the Brain more data to improve predictions.

- [ ] 13F filing parser (institutional flows → Hub 3 scoring)
- [ ] Congress.gov API (live bill status + vote dates)
- [ ] News sentiment feed (RSS/API → new ML feature)
- [ ] Earnings calendar integration (`days_to_earnings` accuracy)

## Phase 3 — Full Stack Rebuild
- [ ] Convert to Next.js 14 App Router
- [ ] Hub pages: `/congressional`, `/insider`, `/institutional`, `/convergence`
- [ ] API routes proxy all external calls (move keys to Vercel env vars)
- [ ] Supabase schema: `signals`, `congressional_trades`, `form4_filings`, `institutional_positions`, `convergence_events`

## Phase 4 — Brain Self-Improvement
> Brain gets smarter autonomously.

- [ ] Feature auto-pruning (drop <1% importance features)
- [ ] IC stagnation detection + feature suggestions
- [ ] Model drift monitoring + alerts
- [ ] Auto-generated historical accuracy reports
- [ ] `--self-check` CLI command

## Phase 5 — Charts & Visuals
- [ ] TradingView Lightweight Charts: entry/target/stop overlays per signal
- [ ] Convergence heatmap: sectors × signal sources, color = score intensity
- [ ] Timeline view: signals + legislative events chronologically

## Phase 6 — Notifications
- [ ] Ntfy.sh push: convergence event detected, ticker enters zone, bill vote within 48h
- [ ] Resend email: full convergence (all 3 hubs + legislation) events only

## Phase 7 — Auth & Monetization
- [ ] Clerk auth (Google/email)
- [ ] Per-user watchlists in Supabase
- [ ] Stripe paywall ($20–50/mo)

---

## General Backlog
- [ ] Committee jurisdiction scoring (requires committee mapping)
- [ ] Proximity to 52-week low scoring (live price at signal time)
- [ ] Options flow — unusual sweep detection (CBOE/OPRA source needed)
- [ ] Short interest anomaly detection
- [ ] Historical accuracy per insider (multi-filing tracking)

---

## Running Cost Targets
| Phase | Est. Monthly Cost |
|---|---|
| Current (HTML + GitHub Actions) | ~$0 |
| Phase 1–2 complete (+ FMP) | ~$15/mo |
| Phase 3 complete (+ Supabase/Vercel) | ~$15–20/mo |
| Full build | ~$30–50/mo |
