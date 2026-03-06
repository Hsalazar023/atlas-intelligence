# ATLAS Quant Standards & Fundamentals

## What This Project Is
A systematic insider trading signal research system for a single
retail investor. Target: 15-25% annualized returns, 5-15 active
positions, 30-90 day holding periods. NOT institutional scale.
Market impact is irrelevant at this scale.

## The Most Important Rules

### 1. Always Use OOS Scores for Analysis
`oos_score` is the honest metric. `total_score` is in-sample
and will always look better than reality.
- Regime analysis: use `oos_score`
- Drawdown simulation: use `oos_score >= 75` as threshold
- Hit rate reporting: use OOS score tiers from --analyze output
- Exception: scoring NEW signals (no OOS score yet) uses
  `total_score` — this is correct and expected

### 2. Walk-Forward Is the Gold Standard
The IC reported by --analyze is walk-forward OOS IC.
This is what matters. Never report in-sample IC as the
primary metric. Current baseline: IC=0.1027, p=0.0003.

### 3. No Look-Ahead Bias
All features must be computable using only data available
AT the time of the signal, not after. Key fixed bugs:
- sector_avg_car: point-in-time (fixed Session 9)
- person track records: point-in-time (verified)
- market regime: uses VIX at signal date (correct)

### 4. IC Thresholds
- IC > 0.10: GOOD — above industry standard for alt data
- IC 0.05-0.10: OK — meaningful but monitor
- IC < 0.05: WEAK — investigate root cause
- p < 0.001: highly significant
- p < 0.01: significant
- p > 0.05: not significant (ns) — treat result with caution

### 5. Feature Promotion Rules
Candidates promoted to active only when fill rate >= 60%.
Prune candidates: features flagged < 1% importance for 3+
consecutive runs. Do NOT prune market_regime until regime
analysis is complete for any given research question.

---

## Key Metrics Reference

### Information Coefficient (IC)
Spearman rank correlation between predicted score and
actual 30d return. Ranges -1 to +1. In finance:
- IC > 0.05: useful signal
- IC > 0.10: strong signal (where we are)
- IC > 0.15: exceptional
Most hedge funds consider IC > 0.05 viable for trading.

### Calmar Ratio
Annualized return / Max drawdown. Target > 1.0.
Current (OOS 75+): 2.12 — strong.

### Sharpe Ratio (monthly)
(Mean monthly return / Std monthly return) * sqrt(12).
Target > 0.35 for long-only strategy.
Current (OOS 75+): 1.53 — strong.

### Kelly Criterion
Optimal position sizing formula.
Full Kelly = (p*b - q) / b where:
  p = hit rate, q = 1-p, b = avg_win/avg_loss
Always use 1/4 Kelly for retail (full Kelly is too aggressive).
Current 1/4 Kelly: 11.2% base per position.
Cap: 15% max, 2% min.

### Factor-Adjusted Alpha
After Fama-French regression, alpha is the return not
explained by market, size, value, momentum, profitability,
investment factors. This is the "pure" insider edge.
Target: > 0.5% monthly (> 6% annualized).

### CAR Measurement Note
All CAR figures measure return from signal_date (disclosure date),
not from insider transaction date. Actual achievable returns are
slightly lower due to:
- Price drift between transaction and disclosure (avg 2-5 days)
- Execution lag after signal appears (1-2 days)
- Model uses closing price on signal_date as entry, not intraday

Practical adjustment: when estimating real portfolio returns,
apply a 3-5% haircut to reported CAR for large-cap liquid names
and 5-8% for small/mid-cap where price reaction is faster.

This is a known and accepted limitation. The signal remains the
disclosure event — the model correctly identifies WHICH stocks
to buy, the timing gap just slightly reduces the magnitude.

---

## Regime Framework (confirmed OOS)

| VIX Range | OOS IC | Status | Score Multiplier |
|-----------|--------|--------|-----------------|
| < 15      | 0.041 ns | WARN | 0.75x |
| 15-25     | 0.104 *** | PASS | 1.00x |
| 25-30     | 0.069 ns | WARN | 0.85x |
| > 40      | insufficient | — | 0.60x |

Current VIX: check market_data.json before each session.
Year-by-year confirmed: 2022 bear market IC=0.099** —
strategy generates alpha independent of market direction.
2020 COVID crash: IC=-0.020 ns — known tail risk.
2024 low-vol: IC=0.040 ns avg VIX=16.1 — regime effect,
not model failure.

---

## Data Sources & Quality

### EDGAR Form 4 (Primary — 1.0x quality)
- Direct SEC filings, highest reliability
- 90-day rolling window, 1,500 filings per fetch
- Skip logic implemented: only new accession numbers fetched
- ~4,340 signals in DB

### FMP Congressional (0.3x quality — DEGRADED)
- Field names: Transaction, Ticker, TransactionDate,
  Representative, Range, Chamber, Party
- Buy filter: 'purchase' in Transaction.lower()
- KNOWN ISSUE: FMP upstream delay, data stuck at 2026-02-13
  despite correct field mapping. Monitor weekly.
- ~3,333 signals in DB
- Individual tier scoring active:
  elite=1.20x, good=0.80x, neutral=0.30x, fade=0.10x

### Price Cache
- Location: data/price_history/[TICKER].json
- Used for: outcomes (car_30d etc), volume features,
  liquidity enrichment
- collect_prices.py updates weekly

---

## Feature Importance Hierarchy (current)

Top features (by ML importance):
1. days_to_earnings (0.142) — MOST IMPORTANT
   Signals far from earnings are cleaner information
2. sector_avg_car (0.129) — sector momentum context
3. insider_buy_ratio_90d (0.055) — conviction signal
4. disclosure_delay (0.049) — faster = more urgent
5. sector (0.047) — sector quality filter

Prune candidates (flagged 3+ runs, < 1% importance):
sect_ticker_momentum, volume_cluster_signal,
market_regime, news_sentiment_30d
→ Do NOT prune yet — monitor one more session

Active features: 28
Candidates (fill < 60%): short_interest_pct,
short_interest_change, institutional_holders,
institutional_pct_held, options_bullish,
options_unusual_calls, options_insider_confluence,
options_bearish_divergence

---

## Scoring Architecture

Score = base(80) + magnitude(300) + convergence(3)
        + person_quality(12)
        × source_quality × role_bonus × tier_multiplier
        × regime_multiplier × cluster_bonus

Source quality: edgar=1.0, congress=0.3, convergence=1.0
Role bonuses: CFO=1.5, COO=1.5, Officer=1.5,
              President=1.49, VP=1.36, Other=1.36,
              CEO=1.15, Director=0.85, 10%Owner=0.70
Cluster bonus: 3+ same ticker 7d = 1.25x, 2 = 1.10x
Regime multiplier: see Regime Framework above

---

## Pipeline Command Reference

Standard session sequence:
  python scripts/fetch_data.py
  python backtest/learning_engine.py --backfill
  python backtest/learning_engine.py --analyze
  python backtest/learning_engine.py --export
  python backtest/learning_engine.py --report

Individual commands:
  --backfill    Ingest new signals, enrich features
  --analyze     Train walk-forward model, store OOS scores
  --export      Update brain_signals.json with live prices
  --report      Print analyst report with regime context
  --summary     Show score distribution and top patterns

Key output files:
  data/brain_signals.json   Top 50 signals + live prices
  data/brain_stats.json     IC, feature importances
  data/brain_health.json    Health check results
  data/analyst_report.json  Full report (machine-readable)

---

## Known Issues & Technical Debt

1. Congress data stuck at 2026-02-13
   Root cause: FMP upstream delay (not our code)
   Status: monitoring, no fix available until FMP updates

2. NONE ticker in DB
   Causes 404 error on every --export
   Fix: DELETE FROM signals WHERE ticker='NONE'
   Status: pending (Task 0, Session 15)

3. Prune candidates flagged 3+ runs
   sect_ticker_momentum, volume_cluster_signal,
   market_regime, news_sentiment_30d
   Status: holding pending regime analysis completion

4. Liquidity enrichment: 0 signals enriched
   enrich_liquidity_features() wired but not filling
   Root cause: price_history path or ADV calculation
   Status: needs debugging

5. Lobbying features: 100% NULL
   lobbying_active, lobbying_trend always NULL
   Status: data source not connected, low priority

---

## Phase Status

Phase 1 (Signal Research): COMPLETE
  IC > 0.10 ✓, p < 0.001 ✓, 75 folds ✓

Phase 2 (Strategy Rigor): COMPLETE
  OOS scores stored ✓, Regime analysis ✓
  Drawdown sim ✓ (Calmar=2.12), Kelly sizing ✓
  Live prices in brain_signals.json ✓
  Fama-French regression: COMPLETE (+128.9% ann alpha, t=3.27**)

Phase 3 (Live Dashboard): IN PROGRESS
  brain_signals.json schema complete ✓
  dashboard.html: PENDING (Session 15 primary task)

Phase 4 (Strategy Refinement): NOT STARTED
  Fundamental analysis layer, short side,
  multi-horizon optimization

---

## Professional Standards Checklist

Before declaring any research finding complete:
☐ Used OOS scores (not total_score)
☐ Sample size adequate (n >= 30 per bucket)
☐ p-value reported alongside IC
☐ Compared to baseline/benchmark
☐ Checked for look-ahead contamination
☐ Documented in docs/brain-status.md

Before any code change to scoring:
☐ Run --analyze before and after
☐ IC stable or improved
☐ Checkpoint saved
☐ Change documented in CLAUDE.md

Before adding a new feature:
☐ Theoretical basis articulated
☐ Fill rate will be >= 60% at maturity
☐ Not correlated with existing top features
☐ Added to CANDIDATE_FEATURES first

---

## Output Standards

### Script Output Format
All pipeline scripts must produce clean, scannable output.
Claude Code should enforce these standards when modifying
any script. If output is messy, fix it before moving on.

#### fetch_data.py
Target format:
  === FETCH [date] [time] UTC ===
  [FRED]      VIX: 23.57  Treasury: 4.06  ✓
  [EDGAR]     1500 fetched  |  149 new XML  |  1351 reused  ✓
  [CONGRESS]  1331 fetched  |  657 purchases  |  0 new inserted  ⚠
  [VOLUME]    96/100 tickers  ✓
  [OPTIONS]   47/50 tickers  ✓
  [SENTIMENT] 95/100 tickers  ✓
  [SHORT]     86/100 tickers  ✓
  [INST]      48/50 tickers  ✓
  === DONE (Xs) ===

Rules:
- One line per data source, counts only
- ✓ = clean, ⚠ = needs attention, ✗ = failed
- Suppress individual 404/delisted ticker noise to stdout
  Log suppressed errors to data/logs/fetch_errors.log instead
- Known-bad tickers (BRK/B, NONE, BHI, AZPN, BSCP):
  suppress entirely — add to a SUPPRESSED_TICKERS set at
  top of fetch_data.py and skip without logging
- Show total elapsed time at end

#### --backfill output
Target format:
  === BACKFILL [date] ===
  Ingested:   2 congress  |  23 EDGAR  |  0 13F
  Features:   72,403 enriched across 7,678 signals
  Outcomes:   2,847 newly filled
  Nulls:      [only show features where filled != 0]
  === DONE (Xs) ===

Rules:
- Suppress NULL count lines where change = 0
- Suppress ALL lobbying lines (always 100% NULL, known issue)
- Suppress sector map loading line
- Suppress "Reset v5-v10 columns" line
- Show only meaningful changes in null counts

#### --analyze output
Target format:
  === ANALYZE [date] ===
  Features:   28 active  |  8 candidates  |  4 promoted
  Walk-fwd:   IC=0.1027 [GOOD]  p=0.0003  t=3.83  folds=75
  OOS stored: 6,963 predictions
  Scoring:    7,678 signals  |  80+:650  60-79:1015  <40:4343
  Regime:     VIX=23.57 → OPTIMAL (1.00x)
  Kelly:      hit=67.7%  win=+15.6%  loss=-11.0%  → 11.2% base
  Health:     DEGRADED  [ic_trend:OK  hit_rate:OK  freshness:WARN]
  === DONE (Xs) ===

Rules:
- Lead with IC — it's the primary metric
- Suppress role bonus list line by line (summarize or skip)
- Suppress feature promotion individual lines
- Suppress "Cleared ML model cache" line
- Suppress "Checkpoint saved" line (keep checkpoint, hide log)
- Health on one line with sub-statuses in brackets
- Suppress "Fill-rate gate promoted" spam on every export step

#### --export output
Target format:
  === EXPORT [date] ===
  Signals:    50 exported  |  37 tickers  |  11 sectors
  Prices:     37/37 live prices fetched
  Regime:     VIX=23.57 → OPTIMAL (1.00x)
  Kelly:      11.2% base per position
  Output:     data/brain_signals.json  ✓
  === DONE (Xs) ===

Rules:
- Suppress NONE/delisted ticker errors (fix NONE bug first)
- Suppress fill-rate gate promotion lines
- Show diversification stats cleanly

#### --report output
Keep current format — it is already well structured.
Add elapsed time at end: "Generated in Xs"

### Error Handling Standards
- Known-bad tickers: suppress entirely (add to SUPPRESSED_TICKERS)
- XML timeouts: show count only — "X XML timeouts (normal, retrying)"
- HTTP 404/500 for data enrichment: suppress per-ticker,
  show summary count only — "X analyst endpoints unavailable"
- New unexpected errors: always show full traceback
- All suppressed errors: write to data/logs/ with timestamp

### What "Clean" Means
Before finishing any session that touched a script:
- Run it once and read the full stdout
- If any line repeats more than 3 times: suppress or summarize
- If a known error appears: suppress it or fix it, never leave it
- If a section takes > 3s with no output: add a progress line
- Goal: a clean run should fit in ~20 lines total stdout

---

## Session Protocol

### At the START of every session, Claude Code must:
1. Read docs/STANDARDS.md (fundamentals and standards)
2. Read CLAUDE.md (current state and next session tasks)
3. Read docs/todo.md (task list)
4. State what it read: current IC, health, top 3 tasks
5. Then begin work

### At the END of every session, Claude Code must:
1. Append entry to docs/SESSION-LOG.md
2. Update CLAUDE.md (metrics, what was done, next session)
3. Update docs/todo.md (mark complete, add new tasks)
4. Update docs/brain-status.md if any metrics changed
5. Run --report and confirm health status

### SESSION-LOG.md entry format (append, never overwrite):

---
## Session [N] — [YYYY-MM-DD]

### Metrics
| Metric            | Before  | After   | Delta   |
|-------------------|---------|---------|---------|
| IC (walk-forward) | 0.1000  | 0.1027  | +0.0027 |
| OOS IC            | 0.1000  | 0.1000  | 0       |
| Training signals  | 7,673   | 7,678   | +5      |
| Active features   | 28      | 28      | 0       |
| Health            | CRITICAL| DEGRADED| ↑       |

### Tasks Completed
- [Task N] Brief description → outcome in one line

### Key Findings
- Most important number or research result
- Second most important

### Issues Found
- Any new bugs, anomalies, or data problems discovered

### Code Changes
| File | Change |
|------|--------|
| learning_engine.py | Added oos_score column (line 297) |

### Next Session Proposed Focus
1. Highest priority task with reason
2. Second priority
3. Third priority
