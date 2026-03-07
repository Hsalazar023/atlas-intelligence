# ATLAS Todo
_Last updated: Session 20 — March 6, 2026_

---

## Session 20 — Data Quality, Model Diagnostics, Liquidity Fix

### Completed (this session)
- [x] **Data quality checks 10-11** — completeness audit + strategy readiness in self-check
- [x] **CI silent failure fix** — `|| true` → explicit file checks + `::warning::` annotations
- [x] **Score vs OOS explainer** — side-by-side card in Intelligence tab
- [x] **75 folds confirmed** — stale stored value, not a code bug
- [x] **Model Mistake Tracker** — `analyze_residuals()` with cross-referencing + dashboard rendering (false positives, missed gems, detected patterns)
- [x] **Liquidity enrichment fix** — price history uses `v`/`c` keys, function was looking for `volume`/`close`. Fixed key lookups.
- [x] **Check 12: Enrichment verification** — step counters in `run_daily`, warnings for steps returning 0
- [x] **Check 13: Pipeline step counters** — `step_counts` column added to `brain_runs` table, JSON persisted per run
- [x] **Check 15: Enhanced ntfy alerts** — now includes brain status, critical checks, trade counts, residual counts
- [x] **Fama-French gap analysis** — compares OOS 75+ raw vs strategy 65+ capped. Dashboard rendering in Intelligence tab.
- [x] **Liquidity in daily pipeline** — `enrich_liquidity_features()` now also runs in `run_daily` (was backfill-only)

### Session 19 Completed
- [x] CI pipeline fix (score_base, TimedeltaIndex)
- [x] .gitignore whitelist (12 JSON files)
- [x] All 3 dashboards deployed
- [x] Trading rules engine (65+, -10% SL, +20% TP, position sizing)
- [x] Entry price backfill (3,477 recovered)
- [x] Dashboard Performance tab (strategy stats, exit reasons, score tiers)

---

## Session 20 Plan Items — DONE (all completed above)

Items 1-4 (P0 trading rules): Done in Session 19
Items 5-7 (P1 dashboard perf): Done in Session 19
Items 8-9 (P2 score/OOS): Done this session
Items 10-11 (P3 diagnostics): Done this session
Checks 12-15 (data quality): Done this session
Liquidity fix: Done this session

---

## Next Session (21) Plan

### Carry Forward
- [ ] **New signal notifications** — 80+ score alerts via ntfy when new high-conviction signals appear
- [ ] **Strategy memo** — investor-format summary with factor alpha, strategy stats, risk metrics

---

## Data Quality & Guardrails Plan

### Why 3,477 missing prices went undetected

**Root cause:** `price_at_signal` was added as a field after thousands of EDGAR signals were already ingested. No check ever verified that critical fields were populated across the full dataset.

**Current monitoring gaps:**
- `run_self_check()` checks IC trend, hit rate, freshness, feature fill, concentration, harmful features
- `db_health_check()` checks NULL rates for `car_30d`, `total_score`, `sector`, `spy_return_30d`
- **Neither checks `price_at_signal`** or strategy-critical fields
- No check compares "how many signals SHOULD be strategy-eligible vs ARE"
- No check runs AFTER enrichment to verify it actually worked
- `|| true` in CI silently swallowed git add failures — errors were invisible
- No ntfy alert when data completeness drops below expected levels

### Implementation Plan (add to `run_self_check`)

**Check 10: Data Completeness Audit**
After every daily run, verify critical fields are populated at expected rates:
```
price_at_signal:     expect >95% (signals with outcomes)
total_score:         expect >90% (scored signals)
sector:              expect >95%
market_cap_bucket:   expect >80%
momentum_1m:         expect >80%
car_30d:             expect >85% (signals older than 35 days)
oos_score:           expect >80% (signals older than 90 days)
```
Status: WARN if any field drops 5% below expectation. CRITICAL if 15% below.
Auto-recommendation: "Run --backfill to recover missing data"

**Check 11: Strategy Readiness**
Compare strategy-eligible signals vs total eligible:
```
eligible = signals with total_score >= 65 AND outcome_30d_filled = 1 AND car_30d IS NOT NULL
ready = eligible AND price_at_signal > 0
ratio = ready / eligible
```
Status: WARN if ratio < 0.90. CRITICAL if < 0.70.
This would have caught the 249/1056 = 23.6% readiness rate immediately.

**Check 12: Enrichment Verification**
After each enrichment step in `run_daily`, log before/after counts:
- Ingestion: "Ingested X, expected Y based on feed size"
- Price backfill: "X signals missing entry price → Y recovered"
- Feature enrichment: "X signals enriched, Z still missing key features"
- Scoring: "X scored, Y failed — verify model loaded"

**Check 13: Pipeline Step Counters**
Track each step's success/failure count in `brain_runs` table:
```sql
ALTER TABLE brain_runs ADD COLUMN step_counts TEXT;
-- JSON: {"ingest": 15, "price_backfill": 0, "enrich": 718, "score": 718, "export": 1}
```
If any step returns 0 when previous runs returned >0, flag as WARN.

**Check 14: Silent Failure Detection**
Replace `|| true` patterns in CI with explicit error handling:
```yaml
# Before (silent failure):
git add data/signal_intelligence.json || true
# After (logged failure):
git add data/signal_intelligence.json 2>/dev/null || echo "::warning::signal_intelligence.json not found"
```

**Check 15: Ntfy Alerts for Data Quality**
Extend the ntfy notification in backtest.yml to include data quality:
```
ATLAS daily complete. Brain: OK | Trades: 1054 | Gaps: 0 critical
```
If any check is CRITICAL, send separate alert:
```
ATLAS DATA QUALITY ALERT: price_at_signal fill rate 23.6% (expect >95%)
```

### Priority Order
1. **Check 11 (Strategy Readiness)** — would have caught this exact bug. Add to `run_self_check`.
2. **Check 10 (Completeness Audit)** — broadens coverage to all critical fields.
3. **Check 14 (Silent Failure)** — fix CI `|| true` patterns.
4. **Check 12 (Enrichment Verification)** — log before/after in each pipeline step.
5. **Check 13 + 15** — pipeline counters + enhanced ntfy alerts.

---

## Session 18 — Dashboard Accuracy, Architecture, Trader-Focused Design

### Completed
- [x] **Task 0a: Win probability fix** — Raw value 71.4 (already %) was multiplied by 100 → 7140%. Fixed to display raw value directly.
- [x] **Task 0b: Factors fix** — Array of objects treated as dict → [object Object]. Added topFactors() parser. Shows "feature: X.X%" properly.
- [x] **Task 0c: pos_folds verified** — Already exists at ml.pos_folds = 52. No fix needed.
- [x] **Task 0d: portfolio_stats.json** — New export in learning_engine.py. Queries ALL signals with car_30d (thousands, not biased top-50). Includes summary, monthly breakdown.
- [x] **Task 1: 4-tab architecture** — POSITIONS (default), PERFORMANCE, INTELLIGENCE, MODEL. Pure JS tabs with hash navigation.
- [x] **Task 2: Active positions detail fix** — Factors parsed correctly, win_prob fixed, trade_size_range shown, target1/target2 with %, cluster boost info.
- [x] **Task 3: New signals expandable** — Same detail panels as active positions, grouped by ticker.
- [x] **Task 4: Portfolio performance tab** — Active card (raw + 3% adjusted), closed summary from portfolio_stats.json, monthly CSS bar chart, paginated closed log.
- [x] **Task 5: Signal intelligence tab** — Interpretations for each gap. Fama-French section. Feature importance table (top 15).
- [x] **Task 6: Model health panel** — Freshness with amber stale warning, feature importance from analyst_report.json.
- [x] **Task 7: Analyst report tab** — Full report: inventory, performance, quality, regime, Kelly, factors, IC trend chart, pipeline health.
- [x] **Task 8: Regime label suppressed** — Removed from individual rows. Only in banner + expanded detail + model health.

### To Run (user)
- [ ] `python backtest/learning_engine.py --export` — generates portfolio_stats.json + brain_signals.json
- [ ] Refresh dashboard — verify all 4 tabs, expandable detail panels, portfolio stats
- [ ] `python backtest/learning_engine.py --report` — regenerate analyst_report.json

### Open Items (carry to Session 19)
- [ ] **Dashboard polish** — review live dashboard, fix remaining display/layout issues
- [ ] **GitHub Actions workflows failing** — debug backtest.yml and fetch-data.yml, fix broken runs
- [ ] **Investigate OOS gap (-3.4)** — why does OOS not discriminate best/worst?
- [ ] **Liquidity enrichment debug** — 0 signals filled, path issue
- [ ] **IC trend chart from live data** — replace hardcoded sessions
- [ ] **New signal notifications** — 80+ alerts via ntfy
- [ ] **Strategy memo** — investor-format summary with factor alpha results

---

## Session 17 — Dashboard Overhaul, Signal Intelligence, Portfolio Performance

### Completed
- [x] **Task 0: CAR Measurement Note** — Added to STANDARDS.md documenting disclosure timing haircut (3-8%).
- [x] **Task 1: Ticker grouping** — Active positions grouped by ticker with expandable detail panels (insiders, signal strength, trade parameters). CSS transitions, color-coded borders, cluster badges.
- [x] **Task 2a: New signals table** — Replaced Entry Zone + Target with Est. Upside (%), Stop ($+%), Horizon columns.
- [x] **Task 2e: Signal count badge** — Stats row shows "50 | 22 active | X new" format.
- [x] **Task 3: Portfolio performance** — Active return card (EW avg with 7% timing haircut), closed signal stats (win rate, avg win/loss, best/worst), monthly performance log.
- [x] **Task 3d: Closed signals export** — learning_engine.py exports last 50 expired signals (30-90 days old) as `closed_signals` array in brain_signals.json.
- [x] **Task 4: Signal intelligence** — `compute_signal_intelligence()` wired into --analyze. Profiles best/worst 50 signals. Dashboard panel shows divergence stats + comparison table. Score gap=43.9, OOS gap=-3.7, earnings gap=-98.9.
- [x] **Task 6a: OOS display fix** — Shows total_score in blue with "est" label when oos_score is null, instead of "new".

### To Run (user)
- [ ] `python backtest/learning_engine.py --analyze` — generates signal_intelligence.json
- [ ] `python backtest/learning_engine.py --export` — regenerate brain_signals.json with closed_signals + pos_folds
- [ ] `python backtest/learning_engine.py --report` — see Fama-French factor analysis section
- [ ] Refresh dashboard — verify ticker grouping, portfolio perf, signal intel panels
- [ ] Run `python scripts/fetch_data.py` — verify clean output format (~15 lines)

### Open Items (carry to Session 18)
- [ ] **New signal notifications** — 80+ alerts via ntfy
- [ ] **Strategy memo** — investor-format summary with factor alpha results
- [ ] **Liquidity enrichment debug** — 0 signals filled, path issue
- [ ] **Investigate OOS gap (-3.7)** — why does OOS not discriminate best/worst?
- [ ] **Dashboard mobile polish** — responsive layout improvements

---

## Session 16 — Dashboard Fixes, NONE Cleanup, Fama-French, Output Standards

### Completed
- [x] **Task 1: Dashboard fixes** — OOS column: shows "new" for recent signals (expected null oos_score). Folds: reads `ml.pos_folds` (was `positive_folds`). Sector: abbreviation map. Closed table: added OOS + Sector columns, limit increased to 30.
- [x] **Task 2: NONE ticker cleanup** — 23 NONE records deleted from DB. 7,655 signals remaining.
- [x] **Task 3: Fama-French 6-factor regression** — Monthly alpha +7.14%/mo (t=3.27**, significant). Annualized +128.9%. R²=0.082. No significant factor loadings. Verdict: STRONG genuine insider edge.
- [x] **Task 4: Output standards** — fetch_data.py: `_quiet_run()` captures verbose output to `data/logs/fetch_verbose.log`, `main()` prints one clean line per source. SUPPRESSED_TICKERS set added. learning_engine.py: 12 verbose `log.info`→`log.debug` for fill-rate, roles, tiers, regime stats, hypotheses.
- [x] **Task 5: Congress investigation** — Confirmed FMP upstream delay. Latest purchase 2026-02-13 (20 days stale). Not our code. Documented, closed.

---

## Session 14 — Regime Guardrails, Kelly Sizing & Dashboard

### Completed
- [x] **Task 0: Regime-aware scoring guardrails** — `get_regime_context()` reads VIX from `market_data.json`. Multipliers: OPTIMAL (VIX 15-25) 1.00x, LOW_VOL (<15) 0.75x, ELEVATED (25-30) 0.85x, HIGH_VOL (30-40) 0.90x, CRISIS (>40) 0.60x. Added `regime_context` to `brain_signals.json` top-level + per-signal `regime_multiplier`/`regime_label`. Congress staleness thresholds relaxed (14d warn, 30d critical). Regime context section added to `--report`.
- [x] **Task 3: Kelly position sizing** — `compute_kelly_size()` uses 1/4 Kelly formula with OOS 75+ hit rate/avg win/loss. Scaled by signal confidence (oos_score/100) and regime multiplier. Capped 2-15%. Added `kelly_size` per signal in `brain_signals.json` and `kelly_params` top-level. Kelly sizing section in `--report`.
- [x] **Task 4: Dashboard scaffold** — `dashboard.html` created. Vanilla JS + CSS, reads `brain_signals.json`. Sections: regime banner, stats row, active positions (P&L, stop loss, Kelly), new signals (entry zone, spread), expired/closed, health bar.

### To Run (user)
- [ ] `python scripts/fetch_data.py` — refresh data (updates market_data.json with current VIX)
- [ ] `python backtest/learning_engine.py --backfill` — enriches liquidity features
- [ ] `python backtest/learning_engine.py --analyze` — trains model + stores OOS predictions
- [ ] `python backtest/learning_engine.py --export` — generates brain_signals.json with regime, Kelly, live prices
- [ ] `python backtest/learning_engine.py --report` — see regime context + Kelly sizing + honest regime robustness
- [ ] Open dashboard: `python3 -m http.server 8080` then go to `localhost:8080/dashboard.html`
- [ ] **Task 1: Honest OOS drawdown simulation** — Run this script:
  ```
python3 -c "
import sqlite3, pandas as pd, numpy as np
conn = sqlite3.connect('data/atlas_signals.db')
df = pd.read_sql('''SELECT ticker, signal_date, oos_score, car_30d, market_adj_car_30d, vix_at_signal
    FROM signals WHERE oos_score IS NOT NULL AND car_30d IS NOT NULL AND signal_date < '2026-01-01' ORDER BY signal_date''', conn)
conn.close()
print(f'Total OOS signals with outcomes: {len(df)}')
for threshold in [80, 75, 70]:
    sub = df[df.oos_score >= threshold].copy()
    if len(sub) < 50: print(f'OOS {threshold}+: only {len(sub)} signals, skipping'); continue
    sub['signal_date'] = pd.to_datetime(sub['signal_date'])
    sub['month'] = sub['signal_date'].dt.to_period('M')
    monthly = sub.groupby('month').agg(n=('car_30d','count'), ret=('car_30d','mean'), hit=('car_30d',lambda x:(x>0).mean()), mkt_adj=('market_adj_car_30d','mean')).reset_index()
    monthly['cum'] = (1+monthly['ret']).cumprod()
    peak = monthly['cum'].expanding().max()
    dd = (monthly['cum']-peak)/peak; max_dd = dd.min(); worst = str(monthly.loc[dd.idxmin(),'month'])
    n_m = len(monthly); tot_ret = monthly['cum'].iloc[-1]-1; ann = (1+tot_ret)**(12/n_m)-1
    calmar = ann/abs(max_dd) if max_dd != 0 else 0
    sharpe = monthly.ret.mean()/monthly.ret.std()*np.sqrt(12)
    print(f'=== OOS {threshold}+ === Signals:{len(sub)} Months:{n_m}')
    print(f'  Ann return: {ann:+.1%} | Max DD: {max_dd:.1%} ({worst}) | Calmar: {calmar:.2f} | Sharpe: {sharpe:.2f}')
    print(f'  Win months: {(monthly.ret>0).sum()}/{n_m} ({100*(monthly.ret>0).mean():.0f}%)')
    for name,(s,e) in {'2020 COVID':('2020-01','2020-06'),'2022 Bear':('2022-01','2022-12'),'2024 Low':('2024-01','2024-12')}.items():
        p = monthly[(monthly['month'].astype(str)>=s)&(monthly['month'].astype(str)<=e)]
        if len(p)==0: continue
        print(f'  {name}: avg={p.ret.mean():+.2%} hit={p.hit.mean():.1%} n={len(p)}mo')
    print()
"
  ```
- [ ] **Task 2: 2024 weakness investigation** — Run this script:
  ```
python3 -c "
import sqlite3, pandas as pd
from scipy import stats
conn = sqlite3.connect('data/atlas_signals.db')
df = pd.read_sql('''SELECT oos_score, car_30d, source, insider_role, vix_at_signal
    FROM signals WHERE oos_score IS NOT NULL AND car_30d IS NOT NULL
    AND signal_date >= '2024-01-01' AND signal_date < '2025-01-01' ''', conn)
print(f'2024 signals: {len(df)}, Avg VIX: {df.vix_at_signal.mean():.1f}')
print('=== IC BY SOURCE ===')
for src in ['edgar','congress']:
    sub = df[df.source==src].dropna(subset=['oos_score','car_30d'])
    if len(sub)<20: continue
    ic,p = stats.spearmanr(sub.oos_score, sub.car_30d)
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    print(f'  {src}: IC={ic:.4f} {sig} n={len(sub)}')
print('=== IC BY ROLE ===')
for role in ['CFO','CEO','Director','Officer','10% Owner']:
    sub = df[df.insider_role==role].dropna(subset=['oos_score','car_30d'])
    if len(sub)<20: continue
    ic,p = stats.spearmanr(sub.oos_score, sub.car_30d)
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    print(f'  {role}: IC={ic:.4f} {sig} n={len(sub)}')
print('=== IC BY VIX IN 2024 ===')
for name,mask in [('VIX<15',df.vix_at_signal<15),('VIX 15-20',(df.vix_at_signal>=15)&(df.vix_at_signal<20)),('VIX 20-25',(df.vix_at_signal>=20)&(df.vix_at_signal<25))]:
    sub = df[mask].dropna(subset=['oos_score','car_30d'])
    if len(sub)<20: continue
    ic,p = stats.spearmanr(sub.oos_score, sub.car_30d)
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    print(f'  {name}: IC={ic:.4f} {sig} n={len(sub)}')
"
  ```

### Fixes Applied
- [x] **Report JSON save order** — `save_json(ANALYST_REPORT, report)` was called before regime/Kelly data was added. Fixed: re-save at end of `generate_analyst_report()`.
- [x] **Regime sections error handling** — Wrapped regime robustness and Kelly sections in try/except to prevent report failure.

### Open Items → Session 15
All carried to Session 15 (now complete — see above).

---

## Session 13 — OOS Predictions, Transaction Costs, Live Price Feed

### Completed
- [x] **Task 0: Congress ingestion** — Already fixed (Session 12 applied capitalized field names). `ingest_congress_feed()` uses `Transaction`, `Ticker`, `TransactionDate`, `Representative` etc.
- [x] **Task 1: Store OOS predictions in DB** — Added `oos_score REAL` and `oos_fold INTEGER` columns. After `walk_forward_train()`, OOS classifier probabilities (0-100 scale) are persisted per signal. Each signal appears in exactly one fold's holdout.
- [x] **Task 2: Regime robustness (honest OOS)** — `--report` now uses `oos_score` when available, with fallback to `total_score`. Adds significance markers and p-values. Labels clearly: "OOS (walk-forward holdout)" vs "IN-SAMPLE".
- [x] **Task 3: Drawdown simulation (honest OOS)** — Stop-loss fields added to `brain_signals.json` export: `stop_loss_price` (-12%), `stop_loss_triggered`, `position_status`.
- [x] **Task 4: Transaction cost modeling** — Added `enrich_liquidity_features()`: ADV from price cache, spread estimate by market cap proxy (large 0.05%, mid 0.20%, small 0.50%). Columns: `avg_daily_volume`, `estimated_spread`, `liquidity_flag`, `net_expected_return`. Wired into `--backfill`.
- [x] **Task 5: Live price feed** — `export_brain_data()` now batch-fetches live prices via yfinance. Added to `brain_signals.json`: `current_price`, `unrealized_pnl_pct`, `days_held`, `days_remaining`, `position_status`, `estimated_spread_pct`, `avg_daily_volume`, `liquidity_flag`, `oos_score`.

### To Run (user)
- [ ] `python scripts/fetch_data.py` — refresh congress + EDGAR data
- [ ] `python backtest/learning_engine.py --backfill` — adds new schema columns, enriches liquidity
- [ ] `python backtest/learning_engine.py --analyze` — trains model + stores OOS predictions
- [ ] Verify OOS predictions stored:
  ```
python3 -c "
import sqlite3
conn = sqlite3.connect('data/atlas_signals.db')
total = conn.execute('SELECT COUNT(*) FROM signals').fetchone()[0]
has_oos = conn.execute('SELECT COUNT(*) FROM signals WHERE oos_score IS NOT NULL').fetchone()[0]
print(f'OOS scores: {has_oos}/{total} ({100*has_oos/total:.1f}%)')
"
  ```
- [ ] Verify OOS IC matches walk-forward IC:
  ```
python3 -c "
import sqlite3, pandas as pd
from scipy import stats
conn = sqlite3.connect('data/atlas_signals.db')
df = pd.read_sql('SELECT oos_score, car_30d FROM signals WHERE oos_score IS NOT NULL AND car_30d IS NOT NULL', conn)
ic, p = stats.spearmanr(df.oos_score, df.car_30d)
print(f'OOS IC (stored): {ic:.4f} [p={p:.6f}] n={len(df)}')
print('Should match walk-forward IC ~0.0996')
"
  ```
- [ ] Honest OOS score distribution:
  ```
  python3 -c "
  import sqlite3, pandas as pd
  conn = sqlite3.connect('data/atlas_signals.db')
  print(pd.read_sql('''
      SELECT
      CASE WHEN oos_score >= 80 THEN '80+' WHEN oos_score >= 60 THEN '60-79'
           WHEN oos_score >= 40 THEN '40-59' ELSE '<40' END as bucket,
      COUNT(*) as n,
      ROUND(AVG(car_30d)*100,2) as avg_car,
      ROUND(AVG(CASE WHEN car_30d>0 THEN 1.0 ELSE 0 END)*100,1) as hit_rate
      FROM signals WHERE oos_score IS NOT NULL AND car_30d IS NOT NULL
      GROUP BY bucket ORDER BY bucket DESC
  ''', conn).to_string())
  "
  ```
- [ ] Honest regime robustness (OOS):
  ```
  python3 -c "
  import sqlite3, pandas as pd
  from scipy import stats
  conn = sqlite3.connect('data/atlas_signals.db')
  df = pd.read_sql('SELECT oos_score, car_30d, vix_at_signal, signal_date FROM signals WHERE oos_score IS NOT NULL AND car_30d IS NOT NULL AND vix_at_signal IS NOT NULL', conn)
  print(f'n={len(df)}')
  for name, mask in [('Low VIX<15', df.vix_at_signal<15), ('Normal 15-25', (df.vix_at_signal>=15)&(df.vix_at_signal<25)), ('Elevated 25-35', (df.vix_at_signal>=25)&(df.vix_at_signal<35)), ('Crisis VIX>35', df.vix_at_signal>=35)]:
      sub = df[mask].dropna(subset=['oos_score','car_30d'])
      if len(sub)<30: print(f'{name}: n={len(sub)} insufficient'); continue
      ic,p = stats.spearmanr(sub.oos_score, sub.car_30d)
      print(f'{name}: IC={ic:.4f} p={p:.4f} n={len(sub)}')
  for yr in range(2019,2027):
      sub = df[df.signal_date.str.startswith(str(yr))].dropna(subset=['oos_score','car_30d'])
      if len(sub)<20: continue
      ic,p = stats.spearmanr(sub.oos_score, sub.car_30d)
      print(f'{yr}: IC={ic:.4f} n={len(sub)}')
  "
  ```
- [ ] Honest drawdown simulation (OOS 80+):
  ```
  python3 -c "
  import sqlite3, pandas as pd, numpy as np
  conn = sqlite3.connect('data/atlas_signals.db')
  df = pd.read_sql(\"SELECT ticker, signal_date, oos_score, car_30d, market_adj_car_30d FROM signals WHERE oos_score >= 80 AND car_30d IS NOT NULL AND signal_date < '2026-01-01' ORDER BY signal_date\", conn)
  if len(df) < 50:
      print(f'WARNING: Only {len(df)} OOS 80+ signals. Trying 70+...')
      df = pd.read_sql(\"SELECT ticker, signal_date, oos_score, car_30d, market_adj_car_30d FROM signals WHERE oos_score >= 70 AND car_30d IS NOT NULL AND signal_date < '2026-01-01' ORDER BY signal_date\", conn)
  df['signal_date'] = pd.to_datetime(df['signal_date'])
  df['month'] = df['signal_date'].dt.to_period('M')
  monthly = df.groupby('month').agg(n=('car_30d','count'), avg_ret=('car_30d','mean'), hit=('car_30d',lambda x:(x>0).mean())).reset_index()
  monthly['cum'] = (1+monthly['avg_ret']).cumprod()
  peak = monthly['cum'].expanding().max()
  dd = (monthly['cum']-peak)/peak
  max_dd = dd.min()
  total_m = len(monthly); total_ret = monthly['cum'].iloc[-1]-1
  ann_ret = (1+total_ret)**(12/total_m)-1
  sharpe = monthly['avg_ret'].mean()/monthly['avg_ret'].std()*np.sqrt(12)
  print(f'OOS 80+ signals: {len(df)}')
  print(f'Ann return: {ann_ret:+.1%} | Max DD: {max_dd:.1%} | Sharpe: {sharpe:.2f}')
  print(f'Win months: {(monthly.avg_ret>0).sum()}/{total_m}')
  "
  ```
- [ ] Liquidity breakdown for 80+ signals:
  ```
  python3 -c "
  import sqlite3, pandas as pd
  conn = sqlite3.connect('data/atlas_signals.db')
  print(pd.read_sql('''
      SELECT liquidity_flag, COUNT(*) as n,
      ROUND(AVG(car_30d)*100,2) as avg_car,
      ROUND(AVG(estimated_spread)*100,3) as avg_spread,
      ROUND(AVG(net_expected_return)*100,2) as avg_net
      FROM signals WHERE total_score >= 80 AND liquidity_flag IS NOT NULL
      GROUP BY liquidity_flag
  ''', conn).to_string())
  "
  ```
- [ ] `python backtest/learning_engine.py --export` — verify new fields in brain_signals.json
- [ ] `python backtest/learning_engine.py --report` — see honest OOS regime robustness
- [ ] Verify brain_signals.json sample:
  ```
  python3 -c "
  import json
  with open('data/brain_signals.json') as f:
      data = json.load(f)
  s = data['signals'][0] if data.get('signals') else {}
  for k in ['ticker','total_score','current_price','unrealized_pnl_pct','days_held','days_remaining','position_status','stop_loss_price','stop_loss_triggered','estimated_spread_pct','liquidity_flag','oos_score']:
      print(f'  {k}: {s.get(k, \"MISSING\")}')
  "
  ```

### Open Items (carry to Session 14)
- [ ] **Kelly position sizing** — size each signal by confidence + variance
- [ ] **Congress scrapers:** House PTR PDFs encrypted, Senate blocked by Cloudflare
- [ ] **Fama-French factor regression** — factor-adjusted alpha
- [ ] **If IC < 0.09:** root cause investigation

---

## Session 12 — Congress Debug, Decay, Regime, Fetch Optimization

### Completed
- [x] **Task 0: Congress ingestion debug** — No bug found. All 664 feed purchases already in DB. Staleness (2026-02-13) is upstream: FMP API has no newer data, House/Senate scrapers extract 0 trades (PDFs encrypted, Cloudflare blocks). Added debug logging to `ingest_congress_feed()`.
- [x] **Task 2: Signal decay analysis** — IC by horizon (in-sample total_score): 5d=0.2355, 30d=0.4016, 90d=0.2686, 180d=0.1992, 365d=0.1906. 30d confirmed as optimal training horizon. For 80+ signals: 30d avg +19.76%, hit 90.9%; 365d avg +35.74%, hit 67.2%.
- [x] **Task 3: Regime robustness** — 3/3 regimes PASS (crisis insufficient n=12). Low vol IC=0.3619, Normal IC=0.4567, Elevated IC=0.3611. Year-by-year: 2022 bear IC=0.3975***, 2025 IC=0.5586***. All in-sample but pattern shows consistency.
- [x] **Task 4: Fetch optimization** — `enrich_form4_xml()` now reuses prior enrichments from `edgar_feed.json`. Only fetches XML for truly new filings. Added `accession_number` column to signals schema + EDGAR ingestion. Expected speedup: ~1,400 XML fetches skipped per run.
- [x] **Task 5: Drawdown simulation (in-sample)** — 80+ signals: 473 signals, 67 months, 99% win months. Max drawdown -2.8%. COVID +9.20%/month, 2022 bear +13.57%/month. CAVEAT: all in-sample (total_score trained on car_30d).
- [x] **Regime robustness added to --report** — New REGIME ROBUSTNESS section in analyst report console output + JSON.

### In-Sample Caveat (Critical)
All decay/regime/drawdown analyses used `total_score` which is IN-SAMPLE (trained on car_30d). The absolute numbers are inflated. Walk-forward OOS IC (0.1028) is the authoritative metric. The walk-forward function stores `oos_predictions` per fold but these aren't persisted in the DB — needed for proper OOS backtest.

### To Run (user)
- [ ] `python scripts/fetch_data.py` — test fetch optimization (should show "X reused, Y new XML")
- [ ] `python backtest/learning_engine.py --backfill` — will add accession_number column
- [ ] `python backtest/learning_engine.py --analyze` — verify IC stable
- [ ] `python backtest/learning_engine.py --report` — see new REGIME ROBUSTNESS section

### Open Items (carry to Session 13)
- [ ] **OOS backtest:** Store walk-forward OOS predictions in DB for proper portfolio simulation
- [ ] **Congress scrapers:** House PTR PDFs encrypted, Senate blocked by Cloudflare. Fix or find alternative source.
- [ ] **Transaction cost modeling** — bid-ask spread by market cap
- [ ] **Kelly position sizing** — add to brain_signals.json
- [ ] **Current price feed** — foundation for live dashboard

---

---

## Backlog (Future Sessions)

### Phase 3 Remaining
- [ ] New signal notifications (80+ alerts)
- [ ] Model health monitor chart
- [ ] Portfolio tracker (manual position entry)

### Phase 4 (Refinement)
- [ ] earnings_catalyst binary feature (8-30d pre-earnings flag)
- [ ] FinBERT upgrade (replace VADER)
- [ ] Congressional individual quality scores
- [ ] Short side exploration (insider sells + low scores)
- [ ] Sector-specific models
- [ ] Multi-horizon model blending (5d + 30d + 90d)
- [ ] Multiple hypothesis testing correction (Benjamini-Hochberg)

---

## Completed (Archive)

### Session 10
- [x] FMP API fix — /stable/ endpoints working, v3/v4 removed
- [x] person_hr/person_car regression fixed (update_person_track_records missing from backfill)
- [x] OOS validation — confirmed walk-forward IC=0.1092 is authoritative; 93.4% hit rate is in-sample
- [x] Feature pruning — volume_dry_up, analyst_consensus removed; short_squeeze_signal, institutional_insider_confluence removed from candidates
- [x] Days-to-earnings deep dive — 8-30d pre-earnings optimal window (+3.28% CAR)
- [x] Pipeline audit — GitHub Actions verified, FMP_API_KEY added to fetch-data.yml

### Session 9
- [x] Beta=0.828 investigated — real long equity characteristic, documented
- [x] sector_avg_car + person_avg_car look-ahead bias found and fixed (45-day buffer)
- [x] Options flow JSON parse error fixed (atomic writes)
- [x] Outcome backfill wired into --backfill permanently
- [x] Diversification cap (3/ticker, 8/sector) with stats

### Session 8
- [x] Outcome backfill for 2019-2023 (3,817 signals filled)
- [x] Price cache extended to 2018 (LOOKBACK=2920)
- [x] Walk-forward folds 22 → 74, p-value 0.490 → 0.0003

### Sessions 1-7
- [x] EDGAR Form 4 + congressional disclosure ingestion
- [x] LightGBM walk-forward model + historical expansion to 2019
- [x] Checkpoint/rollback, DB indexes, health monitoring
- [x] Short interest, institutional, options candidate features
- [x] Analyst report dashboard, self-check system
- [x] Fill-rate gate, hypothesis generation, regime features

pip install pandas-datareader --break-system-packages
python3 -c "
import pandas_datareader.data as web
ff=web.DataReader('F-F_Research_Data_5_Factors_2x3','famafrench',start='2019-01-01')[0]
mom=web.DataReader('F-F_Momentum_Factor','famafrench',start='2019-01-01')[0]
ff=ff/100; mom=mom/100; mom.columns=['MOM']
ff.join(mom,how='inner').to_csv('data/ff5_factors.csv')
print(f'Saved {len(ff)} months of FF5+MOM factors')
"