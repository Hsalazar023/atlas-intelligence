# ATLAS — Roadmap
*Updated Mar 1, 2026. Phases 0-1 complete. Active: Phase 2.*

---

## Phase 0 — Data Quality ✅
## Phase 1 — Brain Integration ✅
See `docs/archive/completed-milestones.md` for details.

---

## Phase 2 — Autonomous Brain Loop (Active)
> Goal: Brain runs itself — collects, learns, scores, deploys, self-monitors.

### 2A. Automated Pipeline (CI/CD)
- [ ] GitHub Action: `--daily` → `--score` → commit brain_signals/stats → push → Vercel auto-deploys
- [ ] Schedule: run after data refresh (4x daily) so scores reflect latest signals
- [ ] Guard: only commit if brain_signals.json actually changed (diff check)
- [ ] Failure alerting: notify on pipeline errors (GitHub Action failure notification)

### 2B. Scoring Calibration
- [ ] **Fix ML weight save** — investigate why `_oos_ic` isn't persisting to optimal_weights.json
- [ ] **Ticker diversification** — max 3 signals per ticker in brain_signals.json export
- [ ] **Source weighting** — EDGAR buys produce +3.71% CAR vs congress -0.36%. Scoring formula may need source-aware adjustments
- [ ] **Regression validation** — use walk-forward regression IC alongside classification for scoring formula tuning
- [ ] **Score band backtesting** — are 80+ signals actually performing better than 65-79? Validate with real forward returns

### 2C. Brain Self-Monitoring (`--self-check`)
- [ ] IC trend tracking: store IC per run, alert when 3 consecutive declines
- [ ] Score concentration report: top tickers by score, flag when >30% of top 50 is one ticker
- [ ] Feature drift: compare current feature distributions to training data
- [ ] Data freshness: alert when no new signals ingested for 48+ hours
- [ ] Person track record recalculation after each run
- [ ] Output: `brain_health.json` — consumed by frontend health dashboard

### 2D. Feature Engineering
- [ ] **Prune trade_pattern** — 31% fill rate, may be adding noise. Test IC with/without.
- [ ] **Test removing cluster_velocity=fast** — -5.15% CAR in analysis, actively harmful
- [ ] **New feature candidates:**
  - `days_since_last_buy` — insider buying frequency (repeat buyers = stronger signal)
  - `sector_momentum` — sector-level momentum (not just ticker)
  - `congressional_committee_match` — is the member on a committee with jurisdiction?
  - `filing_time_of_day` — after-hours filings may signal urgency
  - `price_change_since_signal` — how has stock moved since the signal? (for re-scoring)
- [ ] Auto-pruning: drop features with <1% importance for 3 consecutive runs
- [ ] Feature suggestion log: when IC stagnates >3 runs, log candidates from residual analysis

---

## Phase 3 — Frontend Intelligence
> Goal: The site explains what the Brain sees and why signals matter.

### 3A. Signal Context & Explainability
- [ ] **"Why this signal"** tooltip on each trade idea card:
  - ML confidence (e.g., "73% probability of beating SPY")
  - Key features driving score (top 3 feature contributions for this signal)
  - Person track record ("This insider is 4/6 on 30d buys")
  - Convergence context ("Also flagged by 2 congressional trades")
- [ ] **Score breakdown bar** — visual showing: base (blue) + magnitude (green/red) + convergence (purple) + person (gold)
- [ ] **Signal timeline** per ticker — show all historical signals, their scores, and actual outcomes

### 3B. Brain Performance Dashboard (new page or section)
- [ ] Historical accuracy: "Brain's 80+ signals returned +X% avg over 30 days"
- [ ] IC trend chart (line chart of IC per monthly window)
- [ ] Feature importance chart (horizontal bar, from brain_stats.json)
- [ ] Score distribution histogram
- [ ] Brain health status (from brain_health.json)
- [ ] "What went wrong" section: worst predictions and why (highest score + negative CAR)
- [ ] "What the Brain missed" section: signals that scored low but had high CAR

### 3C. Content Improvements
- [ ] Remove all remaining demo/hardcoded data:
  - BILLS array → replace with Congress.gov API or remove
  - Institutional flow cards → remove until 13F pipeline built
  - Options flow table → remove until data source available
  - SMS previews → simplify to just config UI
- [ ] Add signal source badges: show "Brain Score 82" vs old heuristic scores
- [ ] Sector deep-dive: click sector → see all signals, avg CAR, top performers
- [ ] Person profiles: click insider name → see full trade history, hit rate, avg CAR

---

## Phase 4 — New Data Sources
> Goal: Feed the Brain more data to improve predictions.

| Source | What It Adds | Priority | Effort |
|---|---|---|---|
| Congress.gov API | Live bill status, vote dates, committee assignments | High | Low (free API) |
| 13F filings (SEC) | Institutional flows → new Hub 3 features | High | Medium |
| News sentiment | Event-driven catalyst detection | Medium | Medium |
| Earnings calendar | Improve `days_to_earnings` accuracy | Medium | Low |
| Short interest | SI as contrarian signal feature | Low | Medium |
| Options flow | Unusual activity detection | Low | High (data cost) |

---

## Phase 5 — UI/UX Redesign
> Goal: Professional, clean, fast. Information density without clutter.

- [ ] **Design system:** consistent card sizes, spacing, typography, color usage
- [ ] **Mobile-first:** responsive tables, collapsible sections, touch-friendly filters
- [ ] **TradingView charts:** entry/target/stop overlays per signal
- [ ] **Dark/light mode** toggle
- [ ] **Navigation redesign:** clear hierarchy — Overview → Signals → Brain → Congress → Insider
- [ ] **Loading states:** skeleton screens instead of "feeds loading..."
- [ ] Consider **Next.js migration** for:
  - Server-side rendering (SEO, faster initial load)
  - API routes (move all keys to env vars)
  - Component architecture (break up 3400-line HTML)
  - Supabase integration (persistent user state)

---

## Phase 6 — Notifications & Alerts
- [ ] Ntfy.sh push: convergence events, score threshold crossed, bill vote imminent
- [ ] Email digest: weekly summary of top Brain signals + performance
- [ ] In-app notification center (replace current placeholder)

---

## Phase 7 — Auth & Monetization
- [ ] Clerk auth (Google/email)
- [ ] Per-user watchlists (Supabase)
- [ ] Stripe paywall ($20–50/mo)
- [ ] Free tier: delayed signals, limited history
- [ ] Paid tier: real-time, full Brain access, alerts, API

---

## Running Cost Targets
| Phase | Est. Monthly Cost |
|---|---|
| Current (static HTML + GitHub Actions) | ~$0 |
| Phase 2-3 complete (+ FMP) | ~$15/mo |
| Phase 4-5 complete (+ data sources + Supabase) | ~$30-50/mo |
| Full build (auth + notifications) | ~$50-75/mo |
