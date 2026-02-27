# ALE v2 — Self-Improving Edge Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform the ALE from a basic feature tracker into a fully autonomous, self-improving ML system with expanded data sources, multi-tier convergence, research-backed features, and walk-forward validated models.

**Architecture:** Expand data foundation (FMP congress + sector tagging), add 8 new predictive features from academic research, implement multi-tier convergence (ticker/sector/thematic), build a walk-forward ML engine (Random Forest + LightGBM ensemble), and replace hit-rate-only evaluation with Information Coefficient (IC) analysis.

**Tech Stack:** Python 3.11, SQLite, scikit-learn, lightgbm, yfinance, FMP API ($15/mo)

---

## Task 1: FMP Congressional Data Integration

**Files:**
- Modify: `scripts/fetch_data.py` (add `fetch_fmp_congress()` after line 287)
- Modify: `backtest/shared.py` (add FMP constants)
- Test: `backtest/tests/test_fmp_congress.py` (NEW)

**Step 1: Write the failing test**

```python
# backtest/tests/test_fmp_congress.py
"""Tests for FMP congressional data integration."""
import pytest
from unittest.mock import patch, MagicMock
from scripts.fetch_data import fetch_fmp_congress, normalize_fmp_congress


class TestNormalizeFmpCongress:
    def test_normalizes_senate_trade(self):
        """FMP senate trade should normalize to QuiverQuant-compatible format."""
        raw = {
            'transactionDate': '2025-08-15',
            'disclosureDate': '2025-08-20',
            'representative': 'Sen. Jane Smith',
            'type': 'Purchase',
            'amount': '$50,001 - $100,000',
            'symbol': 'NVDA',
            'assetDescription': 'NVIDIA Corp',
            'owner': 'Joint',
        }
        result = normalize_fmp_congress(raw, chamber='Senate')
        assert result['Ticker'] == 'NVDA'
        assert result['TransactionDate'] == '2025-08-15'
        assert result['Representative'] == 'Sen. Jane Smith'
        assert result['Transaction'] == 'Purchase'
        assert result['Range'] == '$50,001 - $100,000'
        assert result['Chamber'] == 'Senate'
        assert result['DisclosureDate'] == '2025-08-20'

    def test_skips_missing_symbol(self):
        """Trades without a ticker symbol should return None."""
        raw = {
            'transactionDate': '2025-08-15',
            'type': 'Purchase',
            'amount': '$1,001 - $15,000',
            'symbol': '',
            'representative': 'Sen. X',
        }
        result = normalize_fmp_congress(raw, chamber='Senate')
        assert result is None

    def test_computes_disclosure_delay(self):
        """Should compute disclosure delay in days."""
        raw = {
            'transactionDate': '2025-08-01',
            'disclosureDate': '2025-08-10',
            'type': 'Purchase',
            'amount': '$15,001 - $50,000',
            'symbol': 'AAPL',
            'representative': 'Rep. Bob',
        }
        result = normalize_fmp_congress(raw, chamber='House')
        assert result['DisclosureDelay'] == 9


class TestFetchFmpCongress:
    @patch('scripts.fetch_data.requests.get')
    def test_fetches_and_normalizes(self, mock_get):
        """Should fetch from both senate and house endpoints."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = [{
            'transactionDate': '2025-08-15',
            'disclosureDate': '2025-08-20',
            'representative': 'Sen. Test',
            'type': 'Purchase',
            'amount': '$50,001 - $100,000',
            'symbol': 'MSFT',
            'assetDescription': 'Microsoft',
            'owner': 'Self',
        }]
        mock_get.return_value = mock_resp

        trades = fetch_fmp_congress('test_key')
        assert len(trades) >= 1
        assert trades[0]['Ticker'] == 'MSFT'
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/henrysalazar/Desktop/Atlas && python -m pytest backtest/tests/test_fmp_congress.py -v`
Expected: FAIL — `ImportError: cannot import name 'fetch_fmp_congress'`

**Step 3: Add FMP constants to shared.py**

Add after line 22 in `backtest/shared.py`:
```python
FMP_CONGRESS_FEED = DATA_DIR / "fmp_congress_feed.json"
```

**Step 4: Implement fetch_fmp_congress() in fetch_data.py**

Add after line 287:
```python
# ── FMP Congressional Trades ─────────────────────────────────────────────────
def normalize_fmp_congress(raw: dict, chamber: str = '') -> dict | None:
    """Normalize an FMP senate/house trade to QuiverQuant-compatible format."""
    symbol = (raw.get('symbol') or '').strip().upper()
    if not symbol or len(symbol) > 5:
        return None
    tx_date = raw.get('transactionDate') or raw.get('transaction_date') or ''
    disc_date = raw.get('disclosureDate') or raw.get('disclosure_date') or ''
    if not tx_date:
        return None

    # Compute disclosure delay
    delay = None
    if tx_date and disc_date:
        from datetime import datetime
        try:
            td = datetime.strptime(tx_date[:10], '%Y-%m-%d')
            dd = datetime.strptime(disc_date[:10], '%Y-%m-%d')
            delay = (dd - td).days
        except ValueError:
            pass

    return {
        'Ticker': symbol,
        'TransactionDate': tx_date[:10],
        'DisclosureDate': disc_date[:10] if disc_date else '',
        'DisclosureDelay': delay,
        'Representative': raw.get('representative') or raw.get('firstName', '') + ' ' + raw.get('lastName', ''),
        'Transaction': raw.get('type') or '',
        'Range': raw.get('amount') or '',
        'Chamber': chamber,
        'Party': raw.get('party') or '',
        'Owner': raw.get('owner') or '',
        'AssetDescription': raw.get('assetDescription') or '',
        'Source': 'fmp',
    }


def fetch_fmp_congress(api_key: str, pages: int = 10) -> list:
    """Fetch congressional trades from Financial Modeling Prep (Senate + House).
    Returns list of normalized trade dicts compatible with congress_feed.json format.
    """
    all_trades = []

    for chamber, endpoint in [('Senate', 'senate-trading'), ('House', 'house-disclosure')]:
        for page in range(pages):
            url = f'https://financialmodelingprep.com/api/v4/{endpoint}?page={page}&apikey={api_key}'
            try:
                r = requests.get(url, timeout=20)
                if not r.ok:
                    print(f'  FMP {chamber} page {page}: HTTP {r.status_code}')
                    break
                data = r.json()
                if not data:
                    break
                for raw in data:
                    normalized = normalize_fmp_congress(raw, chamber=chamber)
                    if normalized:
                        all_trades.append(normalized)
                time.sleep(0.3)  # rate limit courtesy
            except Exception as e:
                print(f'  FMP {chamber} error: {e}')
                break

    # Deduplicate by (ticker, date, representative)
    seen = set()
    unique = []
    for t in all_trades:
        key = (t['Ticker'], t['TransactionDate'], t['Representative'])
        if key not in seen:
            seen.add(key)
            unique.append(t)

    print(f'  FMP congress: {len(unique)} unique trades ({len(all_trades)} raw)')
    return sorted(unique, key=lambda x: x.get('TransactionDate', ''), reverse=True)
```

**Step 5: Run tests to verify they pass**

Run: `cd /Users/henrysalazar/Desktop/Atlas && python -m pytest backtest/tests/test_fmp_congress.py -v`
Expected: PASS

**Step 6: Wire FMP into the main fetch pipeline**

Update `main()` in `fetch_data.py` to call `fetch_fmp_congress()` when `FMP_API_KEY` env var is set, merge results with existing congress_feed.json, and save.

**Step 7: Commit**

```bash
git add scripts/fetch_data.py backtest/shared.py backtest/tests/test_fmp_congress.py
git commit -m "feat(ale-v2): add FMP congressional data pipeline (senate + house)"
```

---

## Task 2: Sector Tagging System

**Files:**
- Create: `data/sector_map.json` (ticker → GICS sector)
- Create: `backtest/sector_map.py` (loader + tagger)
- Test: `backtest/tests/test_sector_map.py` (NEW)
- Modify: `backtest/learning_engine.py` (tag signals with sector on ingest)

**Step 1: Write the failing test**

```python
# backtest/tests/test_sector_map.py
"""Tests for sector mapping."""
import pytest
from backtest.sector_map import get_sector, build_sector_map


class TestGetSector:
    def test_known_ticker(self):
        assert get_sector('AAPL') == 'Technology'

    def test_unknown_ticker(self):
        assert get_sector('ZZZZZ') is None

    def test_case_insensitive(self):
        assert get_sector('aapl') == 'Technology'


class TestBuildSectorMap:
    def test_returns_dict(self):
        result = build_sector_map()
        assert isinstance(result, dict)
        assert len(result) > 100
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/henrysalazar/Desktop/Atlas && python -m pytest backtest/tests/test_sector_map.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement sector_map.py**

```python
# backtest/sector_map.py
"""Ticker → GICS sector mapping for convergence analysis."""
import json
import time
import logging
from pathlib import Path

log = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent.parent / "data"
SECTOR_MAP_PATH = DATA_DIR / "sector_map.json"
_sector_cache = None


def build_sector_map(api_key: str = None) -> dict:
    """Build ticker→sector map. Uses FMP stock list or falls back to yfinance."""
    if api_key:
        import requests
        url = f'https://financialmodelingprep.com/api/v3/stock-screener?limit=10000&apikey={api_key}'
        try:
            r = requests.get(url, timeout=30)
            if r.ok:
                data = r.json()
                result = {}
                for item in data:
                    sym = (item.get('symbol') or '').upper()
                    sector = item.get('sector') or item.get('industry') or ''
                    if sym and sector:
                        result[sym] = sector
                if result:
                    with open(SECTOR_MAP_PATH, 'w') as f:
                        json.dump(result, f, indent=2)
                    log.info(f"Sector map: {len(result)} tickers saved")
                    return result
        except Exception as e:
            log.warning(f"FMP sector map failed: {e}")

    # Fallback: use cached file
    if SECTOR_MAP_PATH.exists():
        with open(SECTOR_MAP_PATH) as f:
            return json.load(f)
    return {}


def get_sector(ticker: str) -> str | None:
    """Return GICS sector for a ticker, or None."""
    global _sector_cache
    if _sector_cache is None:
        if SECTOR_MAP_PATH.exists():
            with open(SECTOR_MAP_PATH) as f:
                _sector_cache = json.load(f)
        else:
            _sector_cache = {}
    return _sector_cache.get(ticker.upper())
```

**Step 4: Run tests, verify pass**

**Step 5: Wire sector tagging into learning_engine.py**

In `ingest_congress_feed()` and `ingest_edgar_feed()`, add:
```python
from backtest.sector_map import get_sector
signal['sector'] = get_sector(ticker)
```

**Step 6: Commit**

```bash
git add backtest/sector_map.py backtest/tests/test_sector_map.py backtest/learning_engine.py
git commit -m "feat(ale-v2): add sector tagging for convergence analysis"
```

---

## Task 3: Multi-Tier Convergence Detection

**Files:**
- Modify: `backtest/learning_engine.py` — rewrite `update_aggregate_features()` (lines 329-376)
- Test: `backtest/tests/test_learning_engine.py` — add convergence tests

**Step 1: Write the failing test**

```python
# Add to backtest/tests/test_learning_engine.py
class TestMultiTierConvergence:
    def test_tier1_ticker_convergence(self, db):
        """Tier 1: same ticker in congress + edgar within 60d."""
        insert_signal(db, {
            'ticker': 'NVDA', 'signal_date': '2025-06-15', 'source': 'congress',
            'representative': 'Rep A', 'insider_name': None,
        })
        insert_signal(db, {
            'ticker': 'NVDA', 'signal_date': '2025-06-20', 'source': 'edgar',
            'representative': None, 'insider_name': 'CEO',
        })
        from backtest.learning_engine import update_aggregate_features
        update_aggregate_features(db)
        row = db.execute(
            "SELECT convergence_tier FROM signals WHERE ticker='NVDA' AND source='edgar'"
        ).fetchone()
        assert row['convergence_tier'] == 1

    def test_tier2_sector_convergence(self, db):
        """Tier 2: 3+ signals from 2+ sources in same sector within window."""
        for i, (src, rep, ins) in enumerate([
            ('congress', 'Rep A', None),
            ('edgar', None, 'CEO X'),
            ('edgar', None, 'CFO Y'),
        ]):
            db.execute(
                """INSERT INTO signals (ticker, signal_date, source, representative,
                   insider_name, sector) VALUES (?, ?, ?, ?, ?, ?)""",
                (f'DEF{i}', f'2025-06-{15+i}', src, rep, ins, 'Industrials')
            )
        db.commit()
        from backtest.learning_engine import update_aggregate_features
        update_aggregate_features(db)
        rows = db.execute(
            "SELECT convergence_tier FROM signals WHERE sector='Industrials'"
        ).fetchall()
        assert any(r['convergence_tier'] == 2 for r in rows)

    def test_no_convergence_single_source(self, db):
        """No convergence if all signals from same source."""
        insert_signal(db, {
            'ticker': 'AAPL', 'signal_date': '2025-06-15', 'source': 'edgar',
            'representative': None, 'insider_name': 'A',
        })
        insert_signal(db, {
            'ticker': 'AAPL', 'signal_date': '2025-06-16', 'source': 'edgar',
            'representative': None, 'insider_name': 'B',
        })
        from backtest.learning_engine import update_aggregate_features
        update_aggregate_features(db)
        row = db.execute(
            "SELECT convergence_tier FROM signals WHERE ticker='AAPL' LIMIT 1"
        ).fetchone()
        assert row['convergence_tier'] == 0
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/henrysalazar/Desktop/Atlas && python -m pytest backtest/tests/test_learning_engine.py::TestMultiTierConvergence -v`
Expected: FAIL — `convergence_tier` column doesn't exist

**Step 3: Add new columns to schema**

In `SCHEMA_SQL` (learning_engine.py ~line 40), add after `convergence_sources TEXT`:
```sql
convergence_tier INTEGER DEFAULT 0,
convergence_sector TEXT,
convergence_tickers TEXT,
disclosure_delay INTEGER,
```

Also add migration logic in `init_db()` to ALTER TABLE for existing databases.

**Step 4: Rewrite update_aggregate_features()**

Replace lines 329-376 with multi-tier convergence logic:
- **Tier 0:** No convergence (single source only)
- **Tier 1:** Same ticker, 2+ sources, 60-day congress / 30-day EDGAR window
- **Tier 2:** Same sector, 3+ signals from 2+ sources within 30 days
- **Tier 3:** Tier 2 + active legislation (check BILLS list)

**Step 5: Run tests, verify pass**

**Step 6: Commit**

```bash
git add backtest/learning_engine.py backtest/tests/test_learning_engine.py
git commit -m "feat(ale-v2): multi-tier convergence detection (ticker/sector/thematic)"
```

---

## Task 4: New Research-Backed Features

**Files:**
- Modify: `backtest/learning_engine.py` — add feature columns + extractors
- Modify: `backtest/shared.py` — add helper functions
- Test: `backtest/tests/test_new_features.py` (NEW)

**Step 1: Write failing tests for each feature**

```python
# backtest/tests/test_new_features.py
"""Tests for ALE v2 research-backed features."""
import pytest
from backtest.learning_engine import (
    classify_insider_pattern,
    compute_52wk_proximity,
    compute_cluster_velocity,
)


class TestOpportunisticVsRoutine:
    def test_routine_same_month_3_years(self):
        """Insider buying in June for 3+ consecutive years = routine."""
        history = [
            {'date': '2023-06-15'}, {'date': '2024-06-20'}, {'date': '2025-06-10'},
        ]
        assert classify_insider_pattern(history) == 'routine'

    def test_opportunistic_irregular(self):
        """Irregular timing = opportunistic."""
        history = [
            {'date': '2023-03-15'}, {'date': '2024-09-20'}, {'date': '2025-06-10'},
        ]
        assert classify_insider_pattern(history) == 'opportunistic'

    def test_insufficient_history(self):
        """<2 years of data = insufficient_history."""
        history = [{'date': '2025-06-15'}]
        assert classify_insider_pattern(history) == 'insufficient_history'


class Test52WeekProximity:
    def test_at_low(self):
        """Price at 52-week low → proximity = 0.0"""
        result = compute_52wk_proximity(price=50, high_52wk=100, low_52wk=50)
        assert result == 0.0

    def test_at_high(self):
        """Price at 52-week high → proximity = 1.0"""
        result = compute_52wk_proximity(price=100, high_52wk=100, low_52wk=50)
        assert result == 1.0

    def test_midpoint(self):
        """Price at midpoint → proximity = 0.5"""
        result = compute_52wk_proximity(price=75, high_52wk=100, low_52wk=50)
        assert result == 0.5

    def test_no_range(self):
        """Same high and low → return None."""
        result = compute_52wk_proximity(price=50, high_52wk=50, low_52wk=50)
        assert result is None


class TestClusterVelocity:
    def test_burst(self):
        """3 signals in 3 days = burst."""
        dates = ['2025-06-15', '2025-06-16', '2025-06-17']
        result = compute_cluster_velocity(dates)
        assert result == 'burst'

    def test_slow(self):
        """Signals spread over weeks = slow."""
        dates = ['2025-06-01', '2025-06-15', '2025-06-30']
        result = compute_cluster_velocity(dates)
        assert result == 'slow'

    def test_single_signal(self):
        """Single signal = n/a."""
        result = compute_cluster_velocity(['2025-06-15'])
        assert result == 'n/a'
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/henrysalazar/Desktop/Atlas && python -m pytest backtest/tests/test_new_features.py -v`
Expected: FAIL — `ImportError`

**Step 3: Implement feature functions in learning_engine.py**

```python
def classify_insider_pattern(history: list) -> str:
    """Classify insider as opportunistic vs routine (Cohen, Malloy, Pomorski 2012).
    Routine = trades in same calendar month for 3+ consecutive years."""
    if len(history) < 3:
        return 'insufficient_history'

    dates = sorted(datetime.strptime(h['date'][:10], '%Y-%m-%d') for h in history if h.get('date'))
    if len(dates) < 3:
        return 'insufficient_history'

    # Check if any single month has trades in 3+ different years
    from collections import Counter
    month_years = Counter()
    for d in dates:
        month_years[d.month] += 1

    # If any month has 3+ trades across years, check if they're consecutive years
    for month, count in month_years.items():
        if count >= 3:
            years = sorted(set(d.year for d in dates if d.month == month))
            # Check for 3 consecutive years in that month
            for i in range(len(years) - 2):
                if years[i+2] - years[i] <= 2:
                    return 'routine'

    return 'opportunistic'


def compute_52wk_proximity(price: float, high_52wk: float, low_52wk: float) -> float | None:
    """Compute 0-1 proximity to 52-week range. 0 = at low, 1 = at high."""
    if high_52wk is None or low_52wk is None or price is None:
        return None
    range_val = high_52wk - low_52wk
    if range_val <= 0:
        return None
    return round(max(0.0, min(1.0, (price - low_52wk) / range_val)), 4)


def compute_cluster_velocity(dates: list) -> str:
    """Compute average days between consecutive signals.
    burst (<3d), fast (3-7d), moderate (7-14d), slow (>14d)."""
    if len(dates) < 2:
        return 'n/a'
    sorted_dates = sorted(datetime.strptime(d[:10], '%Y-%m-%d') for d in dates)
    gaps = [(sorted_dates[i+1] - sorted_dates[i]).days for i in range(len(sorted_dates)-1)]
    avg_gap = sum(gaps) / len(gaps)
    if avg_gap < 3: return 'burst'
    if avg_gap < 7: return 'fast'
    if avg_gap < 14: return 'moderate'
    return 'slow'
```

**Step 4: Add new columns to signals schema**

```sql
-- Add to SCHEMA_SQL after existing columns:
price_proximity_52wk REAL,
market_cap_bucket TEXT,
relative_buy_size REAL,
sector_momentum REAL,
disclosure_delay INTEGER,
cluster_velocity TEXT,
trade_pattern TEXT,
```

**Step 5: Add feature extractors to compute_feature_stats()**

Add to the `feature_extractors` list in `compute_feature_stats()` (after line 637):
```python
('trade_pattern', lambda r: r['trade_pattern'] or 'n/a'),
('price_proximity', lambda r: 'near_low' if (r['price_proximity_52wk'] or 0.5) < 0.2 else
                               'lower_half' if (r['price_proximity_52wk'] or 0.5) < 0.5 else
                               'upper_half' if (r['price_proximity_52wk'] or 0.5) < 0.8 else
                               'near_high' if r['price_proximity_52wk'] is not None else 'n/a'),
('market_cap', lambda r: r['market_cap_bucket'] or 'n/a'),
('cluster_velocity', lambda r: r['cluster_velocity'] or 'n/a'),
('disclosure_delay', lambda r: 'urgent' if (r['disclosure_delay'] or 999) < 7 else
                                'normal' if (r['disclosure_delay'] or 999) < 30 else
                                'slow' if (r['disclosure_delay'] or 999) < 45 else
                                'late' if r['disclosure_delay'] is not None else 'n/a'),
('insider_role', lambda r: r['insider_role'].upper() if r['insider_role'] else 'n/a'),
```

**Step 6: Run tests, verify pass**

**Step 7: Commit**

```bash
git add backtest/learning_engine.py backtest/tests/test_new_features.py
git commit -m "feat(ale-v2): add research-backed features (opportunistic, 52wk, velocity)"
```

---

## Task 5: Walk-Forward ML Engine

**Files:**
- Create: `backtest/ml_engine.py` (NEW — core ML pipeline)
- Test: `backtest/tests/test_ml_engine.py` (NEW)
- Modify: `backtest/learning_engine.py` (call ML engine from --analyze)

**Step 1: Write failing tests**

```python
# backtest/tests/test_ml_engine.py
"""Tests for the walk-forward ML engine."""
import pytest
import sqlite3
import tempfile
from pathlib import Path
from backtest.ml_engine import (
    prepare_features,
    walk_forward_train,
    compute_information_coefficient,
    WalkForwardResult,
)
from backtest.learning_engine import init_db, insert_signal


@pytest.fixture
def db_with_outcomes():
    """DB with enough signals+outcomes for ML training."""
    conn = init_db(db_path=Path(tempfile.mktemp(suffix='.db')))
    import random
    random.seed(42)
    for i in range(200):
        month = (i % 18) + 1
        year = 2024 + (month - 1) // 12
        month_actual = ((month - 1) % 12) + 1
        date = f'{year}-{month_actual:02d}-{(i % 28) + 1:02d}'
        car = random.gauss(0.005, 0.05)
        conn.execute(
            """INSERT INTO signals (ticker, signal_date, source, representative,
               insider_name, insider_role, trade_size_points, person_trade_count,
               person_hit_rate_30d, same_ticker_signals_7d, same_ticker_signals_30d,
               has_convergence, outcome_30d_filled, car_30d, sector)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
            (f'T{i%50}', date, 'edgar' if i % 3 else 'congress',
             f'Rep{i}' if i % 3 == 0 else None,
             f'Ins{i}' if i % 3 else None,
             random.choice(['CEO', 'CFO', 'Director', 'VP', '']),
             random.choice([3, 5, 8, 10, 15]),
             random.randint(0, 20),
             round(random.random(), 2) if i > 5 else None,
             random.randint(1, 5), random.randint(1, 10),
             1 if i % 30 == 0 else 0, round(car, 6),
             random.choice(['Technology', 'Healthcare', 'Industrials', 'Financials']))
        )
    conn.commit()
    yield conn
    conn.close()


class TestPrepareFeatures:
    def test_returns_dataframe(self, db_with_outcomes):
        X, y, ids = prepare_features(db_with_outcomes)
        assert len(X) == len(y) == len(ids)
        assert len(X) > 0
        assert 'source' in X.columns or len(X.columns) > 0

    def test_no_leakage(self, db_with_outcomes):
        """Feature matrix should not contain outcome columns."""
        X, y, ids = prepare_features(db_with_outcomes)
        for col in X.columns:
            assert 'car_' not in col
            assert 'return_' not in col


class TestWalkForward:
    def test_produces_results(self, db_with_outcomes):
        result = walk_forward_train(db_with_outcomes, min_train_months=3, test_months=1)
        assert isinstance(result, WalkForwardResult)
        assert result.n_folds > 0
        assert result.oos_ic is not None
        assert result.oos_hit_rate is not None

    def test_no_lookahead(self, db_with_outcomes):
        """Train dates must be strictly before test dates in every fold."""
        result = walk_forward_train(db_with_outcomes, min_train_months=3, test_months=1)
        for fold in result.folds:
            assert fold['train_end'] < fold['test_start']


class TestInformationCoefficient:
    def test_perfect_correlation(self):
        predicted = [1, 2, 3, 4, 5]
        actual = [0.01, 0.02, 0.03, 0.04, 0.05]
        ic = compute_information_coefficient(predicted, actual)
        assert ic > 0.9

    def test_no_correlation(self):
        predicted = [1, 2, 3, 4, 5]
        actual = [0.05, 0.01, 0.03, 0.02, 0.04]
        ic = compute_information_coefficient(predicted, actual)
        assert -0.5 < ic < 0.5

    def test_empty_input(self):
        ic = compute_information_coefficient([], [])
        assert ic == 0.0
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/henrysalazar/Desktop/Atlas && python -m pytest backtest/tests/test_ml_engine.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Install dependencies**

Run: `pip install scikit-learn lightgbm`

**Step 4: Implement ml_engine.py**

```python
# backtest/ml_engine.py
"""Walk-forward ML engine for ATLAS signal scoring.

Uses Random Forest + LightGBM ensemble with walk-forward validation.
Trained on historical signals with filled outcomes, tested out-of-sample.
"""
import sqlite3
import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

log = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    'source', 'trade_size_points', 'same_ticker_signals_7d',
    'same_ticker_signals_30d', 'has_convergence', 'convergence_tier',
    'person_trade_count', 'person_hit_rate_30d', 'relative_position_size',
    'insider_role', 'sector', 'price_proximity_52wk', 'market_cap_bucket',
    'cluster_velocity', 'trade_pattern', 'disclosure_delay',
]

CATEGORICAL_FEATURES = [
    'source', 'insider_role', 'sector', 'market_cap_bucket',
    'cluster_velocity', 'trade_pattern',
]


@dataclass
class FoldResult:
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    ic: float
    hit_rate: float
    avg_car: float


@dataclass
class WalkForwardResult:
    n_folds: int
    oos_ic: float
    oos_hit_rate: float
    oos_avg_car: float
    feature_importance: dict
    folds: list = field(default_factory=list)
    model_rf: object = None
    model_lgb: object = None


def prepare_features(conn: sqlite3.Connection):
    """Extract feature matrix X, target y, and signal IDs from database."""
    import pandas as pd

    rows = conn.execute(
        f"SELECT id, signal_date, car_30d, {', '.join(FEATURE_COLUMNS)} "
        f"FROM signals WHERE outcome_30d_filled = 1 AND car_30d IS NOT NULL"
    ).fetchall()

    if not rows:
        return pd.DataFrame(), np.array([]), np.array([])

    data = [dict(r) for r in rows]
    df = pd.DataFrame(data)
    ids = df['id'].values
    y = (df['car_30d'] > 0).astype(int).values
    dates = df['signal_date'].values

    X = df[FEATURE_COLUMNS].copy()

    # Encode categoricals
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].fillna('unknown').astype(str)
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    X = X.fillna(0)
    return X, y, ids, dates


def compute_information_coefficient(predicted, actual) -> float:
    """Spearman rank correlation between predictions and actual returns."""
    if len(predicted) < 3 or len(actual) < 3:
        return 0.0
    corr, _ = spearmanr(predicted, actual)
    return round(corr, 6) if not np.isnan(corr) else 0.0


def walk_forward_train(conn: sqlite3.Connection,
                       min_train_months: int = 6,
                       test_months: int = 1) -> WalkForwardResult:
    """Walk-forward validation with RF + LightGBM ensemble."""
    import pandas as pd

    X, y, ids, dates = prepare_features(conn)
    if len(X) < 50:
        log.warning(f"Insufficient data for ML training ({len(X)} signals)")
        return WalkForwardResult(n_folds=0, oos_ic=0, oos_hit_rate=0,
                                 oos_avg_car=0, feature_importance={})

    # Get actual CARs for IC computation
    car_values = {}
    for row_id in ids:
        r = conn.execute("SELECT car_30d FROM signals WHERE id=?", (int(row_id),)).fetchone()
        if r:
            car_values[row_id] = r['car_30d']

    # Sort by date
    date_series = pd.to_datetime(dates)
    sort_idx = date_series.argsort()
    X = X.iloc[sort_idx].reset_index(drop=True)
    y = y[sort_idx]
    ids = ids[sort_idx]
    date_series = date_series[sort_idx]

    # Walk-forward folds
    min_date = date_series.min()
    max_date = date_series.max()
    folds = []
    all_oos_preds = []
    all_oos_actual = []
    all_oos_cars = []

    train_end = min_date + pd.DateOffset(months=min_train_months)

    while train_end + pd.DateOffset(months=test_months) <= max_date:
        test_end = train_end + pd.DateOffset(months=test_months)

        train_mask = date_series < train_end
        test_mask = (date_series >= train_end) & (date_series < test_end)

        if train_mask.sum() < 30 or test_mask.sum() < 5:
            train_end += pd.DateOffset(months=1)
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        test_ids = ids[test_mask]

        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_probs = rf.predict_proba(X_test)[:, 1]

        # Train LightGBM
        lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42,
                                        verbose=-1, n_jobs=-1)
        lgb_model.fit(X_train, y_train)
        lgb_probs = lgb_model.predict_proba(X_test)[:, 1]

        # Ensemble: average probabilities
        ensemble_probs = (rf_probs + lgb_probs) / 2

        # Compute metrics
        test_cars = [car_values.get(tid, 0) for tid in test_ids]
        ic = compute_information_coefficient(ensemble_probs, test_cars)
        hit_rate = sum(1 for p, a in zip(ensemble_probs > 0.5, y_test) if p == a) / len(y_test)
        avg_car = np.mean(test_cars) if test_cars else 0

        fold = FoldResult(
            train_start=str(date_series[train_mask].min().date()),
            train_end=str(train_end.date()),
            test_start=str(train_end.date()),
            test_end=str(test_end.date()),
            n_train=int(train_mask.sum()),
            n_test=int(test_mask.sum()),
            ic=ic, hit_rate=round(hit_rate, 4), avg_car=round(avg_car, 6),
        )
        folds.append(fold)
        all_oos_preds.extend(ensemble_probs.tolist())
        all_oos_actual.extend(test_cars)

        train_end += pd.DateOffset(months=1)

    # Aggregate OOS metrics
    oos_ic = compute_information_coefficient(all_oos_preds, all_oos_actual)
    oos_hit = sum(1 for p, c in zip(all_oos_preds, all_oos_actual) if (p > 0.5) == (c > 0)) / max(len(all_oos_preds), 1)
    oos_avg_car = np.mean(all_oos_actual) if all_oos_actual else 0

    # Feature importance (from last fold's models)
    importance = {}
    if folds:
        feat_names = list(X.columns)
        rf_imp = rf.feature_importances_
        lgb_imp = lgb_model.feature_importances_ / max(lgb_model.feature_importances_.sum(), 1)
        for i, name in enumerate(feat_names):
            importance[name] = round((rf_imp[i] + lgb_imp[i]) / 2, 4)

    return WalkForwardResult(
        n_folds=len(folds),
        oos_ic=round(oos_ic, 6),
        oos_hit_rate=round(oos_hit, 4),
        oos_avg_car=round(float(oos_avg_car), 6),
        feature_importance=dict(sorted(importance.items(), key=lambda x: -x[1])),
        folds=[{
            'train_start': f.train_start, 'train_end': f.train_end,
            'test_start': f.test_start, 'test_end': f.test_end,
            'n_train': f.n_train, 'n_test': f.n_test,
            'ic': f.ic, 'hit_rate': f.hit_rate,
        } for f in folds],
        model_rf=rf, model_lgb=lgb_model,
    )
```

**Step 5: Run tests, verify pass**

Run: `cd /Users/henrysalazar/Desktop/Atlas && python -m pytest backtest/tests/test_ml_engine.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add backtest/ml_engine.py backtest/tests/test_ml_engine.py
git commit -m "feat(ale-v2): walk-forward ML engine (RF + LightGBM ensemble)"
```

---

## Task 6: Wire ML Engine into ALE Pipeline

**Files:**
- Modify: `backtest/learning_engine.py` — integrate ML into `--analyze` flow
- Modify: `backtest/learning_engine.py` — update `generate_weights_from_stats()` to use ML output
- Modify: `backtest/learning_engine.py` — expand `generate_dashboard()` with ML metrics

**Step 1: Import and call ML engine from analyze command**

In the `--analyze` handler (around line 850+), after `compute_feature_stats()`:
```python
from backtest.ml_engine import walk_forward_train

# Walk-forward ML training
log.info("Running walk-forward ML training...")
ml_result = walk_forward_train(conn)
log.info(f"ML results: {ml_result.n_folds} folds, OOS IC={ml_result.oos_ic}, "
         f"OOS hit_rate={ml_result.oos_hit_rate}")

# Only update weights if ML outperforms current by >5%
current_weights = load_json(OPTIMAL_WEIGHTS) if OPTIMAL_WEIGHTS.exists() else DEFAULT_WEIGHTS
current_ic = current_weights.get('_oos_ic', 0)
if ml_result.oos_ic > current_ic * 1.05 or current_ic == 0:
    weights = generate_weights_from_stats(conn)
    weights['_oos_ic'] = ml_result.oos_ic
    weights['_oos_hit_rate'] = ml_result.oos_hit_rate
    weights['_feature_importance'] = ml_result.feature_importance
    weights['method'] = 'walk_forward_ensemble'
    save_json(OPTIMAL_WEIGHTS, weights)
    log.info("Weights updated — ML outperformed previous by >5%")
else:
    log.info(f"Weights NOT updated — ML IC {ml_result.oos_ic} vs current {current_ic} (need >5% improvement)")
```

**Step 2: Expand dashboard with ML metrics**

Add to `generate_dashboard()`:
```python
dashboard['ml_model_performance'] = {
    'n_folds': ml_result.n_folds if 'ml_result' in dir() else 0,
    'oos_ic_30d': ml_result.oos_ic if 'ml_result' in dir() else None,
    'oos_hit_rate': ml_result.oos_hit_rate if 'ml_result' in dir() else None,
    'feature_importance': ml_result.feature_importance if 'ml_result' in dir() else {},
}
```

**Step 3: Run full test suite**

Run: `cd /Users/henrysalazar/Desktop/Atlas && python -m pytest backtest/tests/ -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add backtest/learning_engine.py
git commit -m "feat(ale-v2): wire ML engine into ALE analyze pipeline with safety rails"
```

---

## Task 7: Update GitHub Actions Pipeline

**Files:**
- Modify: `.github/workflows/backtest.yml`

**Step 1: Add FMP_API_KEY secret**

Note: User must add `FMP_API_KEY` as a GitHub repository secret at:
`https://github.com/Hsalazar023/atlas-intelligence/settings/secrets/actions`

**Step 2: Update workflow file**

```yaml
name: Daily Backtest & Learning Engine

on:
  schedule:
    - cron: '0 22 * * 1-5'  # Mon-Fri 10:00 PM UTC (5:00 PM ET)
  workflow_dispatch:

jobs:
  backtest:
    runs-on: ubuntu-latest
    permissions:
      contents: write

    steps:
      - name: Checkout repo
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install yfinance pandas numpy requests scikit-learn lightgbm scipy

      - name: Fetch FMP congressional data
        if: env.FMP_API_KEY != ''
        env:
          FMP_API_KEY: ${{ secrets.FMP_API_KEY }}
        run: python scripts/fetch_data.py --fmp-congress

      - name: Collect price history
        run: python backtest/collect_prices.py

      - name: Run event study
        run: python backtest/run_event_study.py

      - name: Optimize weights (grid search)
        run: python backtest/optimize_weights.py

      - name: Update learning engine (daily collect + backfill)
        run: python backtest/learning_engine.py --daily

      - name: Run ML analysis (weekly on Mondays + manual)
        run: |
          DAY=$(date -u +%u)
          if [ "$DAY" = "1" ] || [ "${{ github.event_name }}" = "workflow_dispatch" ]; then
            python backtest/learning_engine.py --analyze
          fi

      - name: Print learning engine summary
        run: python backtest/learning_engine.py --summary

      - name: Commit updated results
        run: |
          git config --global user.name  'ATLAS Backtest Bot'
          git config --global user.email 'bot@atlasiq.io'
          git add data/optimal_weights.json data/backtest_results.json data/backtest_summary.json data/price_history/ data/ale_dashboard.json data/atlas_signals.db data/sector_map.json data/fmp_congress_feed.json
          git diff --staged --quiet || git commit -m "chore: daily backtest $(date -u '+%Y-%m-%d') [skip ci]"
          git push
```

**Step 3: Commit**

```bash
git add .github/workflows/backtest.yml
git commit -m "feat(ale-v2): update pipeline with FMP congress, ML training, new deps"
```

---

## Task 8: Bootstrap FMP Historical Congress Data

**Files:**
- Modify: `backtest/bootstrap_historical.py` — add FMP historical fetch
- Test: Manual verification via `--summary`

**Step 1: Add FMP historical fetch to bootstrap**

Replace the QuiverQuant historical attempt in `fetch_congress_trades()` (lines 189-209) with FMP pagination:

```python
# Try FMP API for historical congressional trades
fmp_key = os.environ.get('FMP_API_KEY', '')
if fmp_key:
    from scripts.fetch_data import fetch_fmp_congress
    log.info("Fetching historical congressional trades via FMP...")
    fmp_trades = fetch_fmp_congress(fmp_key, pages=50)  # 50 pages = deep history
    trades.extend(fmp_trades)
    log.info(f"FMP historical: {len(fmp_trades)} trades fetched")
```

**Step 2: Run bootstrap with FMP key**

Run: `FMP_API_KEY=<key> python backtest/bootstrap_historical.py`
Expected: Congress signals jump from 99 to 1,000+

**Step 3: Verify with summary**

Run: `python backtest/learning_engine.py --summary`
Expected: Congress signal count significantly increased

**Step 4: Commit**

```bash
git add backtest/bootstrap_historical.py
git commit -m "feat(ale-v2): bootstrap historical congress data via FMP API"
```

---

## Task 9: Backfill New Features for Existing Signals

**Files:**
- Modify: `backtest/learning_engine.py` — add `backfill_new_features()` function

**Step 1: Implement backfill function**

```python
def backfill_new_features(conn: sqlite3.Connection) -> int:
    """Backfill new v2 features for existing signals: sector, 52wk proximity,
    cluster velocity, opportunistic/routine classification."""
    from backtest.sector_map import get_sector
    updated = 0

    # Sector backfill
    rows = conn.execute("SELECT id, ticker FROM signals WHERE sector IS NULL").fetchall()
    for row in rows:
        sector = get_sector(row['ticker'])
        if sector:
            conn.execute("UPDATE signals SET sector=? WHERE id=?", (sector, row['id']))
            updated += 1

    # 52-week proximity backfill
    rows = conn.execute(
        "SELECT id, ticker, signal_date, price_at_signal FROM signals "
        "WHERE price_proximity_52wk IS NULL AND price_at_signal IS NOT NULL"
    ).fetchall()
    for row in rows:
        price_index = load_price_index(row['ticker'])
        if not price_index:
            continue
        sig_dt = datetime.strptime(row['signal_date'], '%Y-%m-%d')
        start = (sig_dt - timedelta(days=252)).strftime('%Y-%m-%d')
        prices_in_range = [v for d, v in price_index.items() if start <= d <= row['signal_date']]
        if prices_in_range:
            high_52 = max(prices_in_range)
            low_52 = min(prices_in_range)
            proximity = compute_52wk_proximity(row['price_at_signal'], high_52, low_52)
            if proximity is not None:
                conn.execute("UPDATE signals SET price_proximity_52wk=? WHERE id=?",
                             (proximity, row['id']))
                updated += 1

    # Cluster velocity backfill
    ticker_dates = conn.execute(
        "SELECT DISTINCT ticker FROM signals WHERE cluster_velocity IS NULL"
    ).fetchall()
    for td in ticker_dates:
        ticker = td['ticker']
        sig_rows = conn.execute(
            "SELECT id, signal_date FROM signals WHERE ticker=? ORDER BY signal_date",
            (ticker,)
        ).fetchall()
        dates = [r['signal_date'] for r in sig_rows]
        velocity = compute_cluster_velocity(dates)
        conn.executemany(
            "UPDATE signals SET cluster_velocity=? WHERE id=?",
            [(velocity, r['id']) for r in sig_rows]
        )
        updated += len(sig_rows)

    conn.commit()
    log.info(f"Backfilled new features for {updated} signal updates")
    return updated
```

**Step 2: Wire into --daily pipeline**

Add call after `backfill_outcomes()`:
```python
backfill_new_features(conn)
```

**Step 3: Commit**

```bash
git add backtest/learning_engine.py
git commit -m "feat(ale-v2): backfill new features for existing signals"
```

---

## Task 10: Update Frontend Convergence Scoring

**Files:**
- Modify: `atlas-intelligence.html` — update `computeConvergenceScore()` (~line 2110)

**Step 1: Expand convergence to include sector-level**

Update `computeConvergenceScore()` to check sector convergence:
- Congress window: 30d → 60d
- Add sector clustering check
- Display convergence tier in signal cards

**Step 2: Update dashboard display to show ML metrics**

In the backtest stats bar, add IC metric display when available from `optimal_weights.json`.

**Step 3: Test locally**

Run: `cd /Users/henrysalazar/Desktop/Atlas && python3 -m http.server 8080`
Verify: Convergence scores and ML metrics render correctly

**Step 4: Commit**

```bash
git add atlas-intelligence.html
git commit -m "feat(ale-v2): update frontend convergence scoring + ML metrics display"
```

---

## Verification Plan

After all tasks are complete:

1. **Run full test suite:**
   ```bash
   cd /Users/henrysalazar/Desktop/Atlas && python -m pytest backtest/tests/ -v
   ```
   Expected: All tests pass (existing 51 + ~25 new = ~76 tests)

2. **Run FMP bootstrap (requires API key):**
   ```bash
   FMP_API_KEY=<key> python backtest/bootstrap_historical.py
   ```
   Expected: Congress signals increase from 99 to 1,000+

3. **Run daily pipeline:**
   ```bash
   python backtest/learning_engine.py --daily
   ```
   Expected: New features computed, sectors tagged

4. **Run ML analysis:**
   ```bash
   python backtest/learning_engine.py --analyze
   ```
   Expected: Walk-forward results printed, weights updated if improvement >5%

5. **Check summary:**
   ```bash
   python backtest/learning_engine.py --summary
   ```
   Expected: New convergence tiers shown, ML IC metric displayed, feature importance listed

6. **Local frontend test:**
   ```bash
   python3 -m http.server 8080
   ```
   Open `http://localhost:8080/atlas-intelligence.html`, verify convergence scores render

7. **Deploy:**
   ```bash
   git push
   ```
   Vercel auto-deploys; GitHub Actions runs nightly pipeline with new features
