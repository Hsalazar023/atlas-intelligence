"""
bootstrap_historical.py — One-time historical data collection for the ATLAS
Adaptive Learning Engine.

Fetches ~21 months of EDGAR Form 4 filings + congressional trades, collects
price history, backfills CARs, and computes initial feature stats.

Usage:
    python backtest/bootstrap_historical.py

This populates data/atlas_signals.db and generates initial weights.
Run once, then switch to daily incremental via learning_engine.py --daily.
"""

import os
import sys
import json
import time
import logging
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.shared import (
    DATA_DIR, PRICE_HISTORY_DIR, OPTIMAL_WEIGHTS, SIGNALS_DB,
    load_json, save_json, match_edgar_ticker, range_to_base_points,
    SEC_USER_AGENT,
)
from backtest.learning_engine import (
    init_db, insert_signal, update_aggregate_features,
    backfill_outcomes, compute_feature_stats, generate_weights_from_stats,
    load_price_index, generate_dashboard, print_summary,
    update_person_track_records, enrich_signal_features, enrich_market_context,
    generate_analysis_report, generate_diagnostics_html,
)
from backtest.sector_map import get_sector, build_sector_map, get_market_cap_bucket

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

SEC_DELAY = 0.12  # seconds between SEC requests (SEC allows 10 req/s)


def _parse_efts_hits(hits: list) -> list:
    """Parse EFTS hit objects into filing dicts."""
    import re

    def is_company(name):
        upper = name.upper()
        words = ['INC', ' CORP', ' LLC', ' LTD', ' CO ', ' CO,', ' GROUP',
                 ' HOLDINGS', ' FUND', ' TRUST', ' INTERNATIONAL', ' PARTNERS',
                 ' MANAGEMENT', ' CAPITAL', ' TECHNOLOGIES', ' SYSTEMS', ' PLC']
        return any(w in upper for w in words)

    def clean_name(n):
        return re.sub(r'\s*\(CIK \d+\)\s*', '', n).strip()

    results = []
    for h in hits:
        src = h.get('_source', {})
        names = src.get('display_names', [])
        _id = h.get('_id', '')
        company_names = [n for n in names if is_company(n)]
        person_names = [n for n in names if not is_company(n)]
        company = clean_name(company_names[-1]) if company_names else clean_name(names[-1]) if names else 'Unknown'
        insider = clean_name(person_names[0]) if person_names else clean_name(names[0]) if names else 'Unknown'

        # Build XML URL for Form 4 enrichment
        accession = src.get('adsh', '')
        ciks = src.get('ciks', [])
        company_cik = ciks[-1] if len(ciks) > 1 else (ciks[0] if ciks else '')
        xml_url = ''
        if company_cik and accession and ':' in _id:
            try:
                numeric_cik = str(int(company_cik))
                clean_acc = accession.replace('-', '')
                xml_filename = _id.split(':', 1)[1]
                xml_url = (
                    f'https://www.sec.gov/Archives/edgar/data/'
                    f'{numeric_cik}/{clean_acc}/{xml_filename}'
                )
            except (ValueError, IndexError):
                pass

        results.append({
            'company': company,
            'insider': insider,
            'date': src.get('file_date', ''),
            'period': src.get('period_ending', ''),
            'accession': accession,
            'xml_url': xml_url,
        })
    return results


def fetch_edgar_historical(days: int = 635, raw_per_month: int = 1500,
                           keep_per_month: int = 500) -> list:
    """Fetch historical Form 4 filings from EDGAR EFTS in monthly chunks,
    prioritizing filings that match known tickers (higher signal quality).

    SEC EFTS returns results newest-first and processes ~1,000+ Form 4s per day.
    We oversample each month (raw_per_month), then filter to keep only filings
    that match a known ticker via the SEC ticker map. This naturally selects
    filings from larger, more liquid companies — the ones most useful for our
    scoring engine.

    Args:
        days: How far back to look (default 635 = ~21 months)
        raw_per_month: Raw filings to fetch per month before filtering (default 1500)
        keep_per_month: Max ticker-matched filings to keep per month (default 500)
    """
    # Reusable session for connection pooling (TCP keep-alive)
    session = requests.Session()
    session.headers.update({
        'User-Agent': SEC_USER_AGENT,
        'Accept': 'application/json',
    })

    # Build monthly date windows going backwards
    now = datetime.utcnow()
    month_windows = []
    cursor = now
    earliest = now - timedelta(days=days)

    while cursor > earliest:
        month_end = cursor.strftime('%Y-%m-%d')
        month_start = max(cursor.replace(day=1), earliest).strftime('%Y-%m-%d')
        month_windows.append((month_start, month_end))
        # Move to last day of previous month
        cursor = cursor.replace(day=1) - timedelta(days=1)

    all_filings = []
    page_size = 100

    log.info(f"Fetching EDGAR Form 4 filings: {len(month_windows)} months "
             f"({earliest.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}), "
             f"oversampling {raw_per_month}/month → keeping best {keep_per_month}...")

    for month_start, month_end in month_windows:
        from_idx = 0
        month_raw = []

        # Phase 1: Oversample — fetch up to raw_per_month filings
        while len(month_raw) < raw_per_month:
            url = (
                f'https://efts.sec.gov/LATEST/search-index'
                f'?forms=4'
                f'&dateRange=custom'
                f'&startdt={month_start}'
                f'&enddt={month_end}'
                f'&from={from_idx}'
            )
            try:
                r = session.get(url, timeout=20)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                log.warning(f"EDGAR fetch error for {month_start}: {e}")
                break

            hits = data.get('hits', {}).get('hits', [])
            if not hits:
                break

            month_raw.extend(_parse_efts_hits(hits))

            total_available = data.get('hits', {}).get('total', {}).get('value', 0)
            from_idx += page_size
            if from_idx >= min(total_available, raw_per_month):
                break

            time.sleep(SEC_DELAY)

        # Phase 2: Filter — keep only ticker-matched filings (known companies)
        matched = []
        unmatched = []
        for f in month_raw:
            ticker = match_edgar_ticker(f.get('company', ''))
            if ticker:
                f['_matched_ticker'] = ticker
                matched.append(f)
            else:
                unmatched.append(f)

        # Take all matched filings up to keep_per_month
        month_kept = matched[:keep_per_month]

        match_rate = len(matched) / len(month_raw) * 100 if month_raw else 0
        log.info(f"  {month_start[:7]}: {len(month_raw)} raw → "
                 f"{len(matched)} matched ({match_rate:.0f}%) → "
                 f"{len(month_kept)} kept")
        all_filings.extend(month_kept)

    session.close()
    log.info(f"Fetched {len(all_filings)} ticker-matched EDGAR filings "
             f"across {len(month_windows)} months")
    return all_filings


def fetch_congress_trades() -> list:
    """Fetch congressional trades from all available sources.

    Priority: FMP API (deep history) > existing congress_feed.json.
    Results are merged, deduplicated, and saved back to congress_feed.json.
    """
    trades = []
    sources = []

    # 1. FMP API — primary source (50 pages/chamber to try reaching 2021+)
    # FMP congress data typically goes back to ~2021-2022 at most
    fmp_key = os.environ.get('FMP_API_KEY', 'UefVEEvF1XXtpgWcsidPCGxcDJ6N0kXv')
    if fmp_key:
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from scripts.fetch_data import fetch_fmp_congress
            log.info("Fetching congressional trades from FMP (50 pages/chamber)...")
            fmp_trades = fetch_fmp_congress(fmp_key, pages=50)
            if fmp_trades:
                trades.extend(fmp_trades)
                sources.append('FMP')
                log.info(f"FMP: {len(fmp_trades)} trades fetched")
        except Exception as e:
            log.warning(f"FMP congress fetch error: {e}")
    else:
        log.info("FMP_API_KEY not set — skipping FMP fetch")

    # 2. Load existing congress_feed.json (may have prior data)
    feed_path = DATA_DIR / "congress_feed.json"
    if feed_path.exists():
        data = load_json(feed_path)
        existing = data.get('trades', [])
        trades.extend(existing)
        sources.append('congress_feed.json')
        log.info(f"Loaded {len(existing)} trades from congress_feed.json")

    # Deduplicate by (ticker, date, representative)
    seen = set()
    unique = []
    for t in trades:
        key = (
            (t.get('Ticker') or '').upper(),
            t.get('TransactionDate') or t.get('Date') or '',
            t.get('Representative') or '',
        )
        if key not in seen:
            seen.add(key)
            unique.append(t)

    # Save merged data back to congress_feed.json for future use
    if unique and fmp_key:
        unique.sort(
            key=lambda x: x.get('TransactionDate', x.get('Date', '')),
            reverse=True,
        )
        from scripts.fetch_data import save_json as save_data_json
        save_data_json('congress_feed.json', {
            'updated': datetime.now(tz=timezone.utc).isoformat(),
            'count': len(unique),
            'source': ' + '.join(sources),
            'trades': unique,
        })

    log.info(f"Total congressional trades (deduplicated): {len(unique)} "
             f"from {', '.join(sources)}")
    return unique


def collect_price_for_ticker(ticker: str, lookback_days: int = 730) -> bool:
    """Collect price history for a single ticker using smart incremental logic.

    Checks existing cache and only fetches what's missing:
    - If no cache exists → full download
    - If cache is missing older data → fetch backward portion
    - If cache is stale (latest date >2d old) → fetch forward portion
    - If cache covers needed lookback and is up-to-date → skip entirely
    """
    from backtest.collect_prices import fetch_candles, merge_candles, load_cached_candles

    PRICE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    try:
        existing = load_cached_candles(ticker)
    except OSError:
        existing = {}  # fd exhaustion — treat as uncached
    now = datetime.now()
    needed_start = (now - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    end_date = (now + timedelta(days=1)).strftime('%Y-%m-%d')

    if not existing:
        # No cache — full download
        new_candles = fetch_candles(ticker, needed_start, end_date)
        if not new_candles:
            return False
        save_json(PRICE_HISTORY_DIR / f"{ticker}.json", new_candles)
        return True

    sorted_dates = sorted(existing.keys())
    cache_min = sorted_dates[0]
    cache_max = sorted_dates[-1]
    cache_max_dt = datetime.strptime(cache_max, '%Y-%m-%d')

    needs_backfill = cache_min > needed_start
    needs_forward = (now - cache_max_dt).days > 2

    if not needs_backfill and not needs_forward:
        # Cache is complete and up-to-date
        return True

    merged = dict(existing)

    if needs_backfill:
        # Fetch the gap: needed_start to cache_min
        backfill = fetch_candles(ticker, needed_start, cache_min)
        if backfill:
            merged = merge_candles(merged, backfill)
            log.debug(f"{ticker}: backfilled {len(backfill)} older days")

    if needs_forward:
        # Fetch from cache_max to now
        forward = fetch_candles(ticker, cache_max, end_date)
        if forward:
            merged = merge_candles(merged, forward)
            log.debug(f"{ticker}: fetched {len(forward)} newer days")

    if len(merged) > len(existing):
        save_json(PRICE_HISTORY_DIR / f"{ticker}.json", merged)

    return len(merged) > 0


def _step_timer():
    """Simple context-manager-like timer. Returns a callable that returns elapsed seconds."""
    start = time.time()
    return lambda: round(time.time() - start, 1)


def _check_spy_coverage(spy_index: dict, conn) -> None:
    """Validate SPY price data covers our full signal date range. Logs warnings."""
    if not spy_index:
        log.error("⚠ SPY price file is EMPTY — no outcomes can be computed!")
        return

    spy_dates = sorted(spy_index.keys())
    spy_min, spy_max = spy_dates[0], spy_dates[-1]

    sig_range = conn.execute(
        "SELECT MIN(signal_date) as min_d, MAX(signal_date) as max_d FROM signals"
    ).fetchone()
    sig_min, sig_max = sig_range['min_d'], sig_range['max_d']

    log.info(f"  SPY price coverage:   {spy_min} to {spy_max} ({len(spy_index)} trading days)")
    log.info(f"  Signal date range:    {sig_min} to {sig_max}")

    # Check: can we compute 365d outcomes for earliest signals?
    if spy_min > sig_min:
        log.warning(f"  ⚠ SPY starts at {spy_min} but signals start at {sig_min} — "
                    f"early signals will miss base price")

    # Check: can we compute outcomes up to the latest signal + window?
    from datetime import datetime as _dt
    spy_max_dt = _dt.strptime(spy_max, '%Y-%m-%d')
    sig_min_dt = _dt.strptime(sig_min, '%Y-%m-%d')
    coverage_days = (spy_max_dt - sig_min_dt).days
    log.info(f"  Effective coverage:   {coverage_days} days from earliest signal")
    if coverage_days < 365:
        log.warning(f"  ⚠ Only {coverage_days}d of coverage — 365d outcomes will be incomplete")
    elif coverage_days < 730:
        log.info(f"  ℹ 365d outcomes available for signals before ~{(spy_max_dt - _dt.strptime(sig_min, '%Y-%m-%d')).days - 365}d ago")


def _print_data_quality_report(conn) -> None:
    """Print comprehensive data quality report after bootstrap."""
    total = conn.execute("SELECT COUNT(*) as cnt FROM signals").fetchone()['cnt']
    if total == 0:
        log.warning("No signals in database — nothing to report.")
        return

    print(f"\n{'='*70}")
    print(f"  DATA QUALITY REPORT")
    print(f"{'='*70}")

    # ── 1. Outcome fill rates ──
    print(f"\n── Outcome Fill Rates ──")
    print(f"  {'Horizon':<12s} {'Filled':>8s} {'With CAR':>10s} {'Fill %':>8s} {'CAR %':>8s} {'Gap':>8s}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for h in ['5d', '30d', '90d', '180d', '365d']:
        filled = conn.execute(
            f"SELECT COUNT(*) as cnt FROM signals WHERE outcome_{h}_filled=1"
        ).fetchone()['cnt']
        with_car = conn.execute(
            f"SELECT COUNT(*) as cnt FROM signals WHERE outcome_{h}_filled=1 AND car_{h} IS NOT NULL"
        ).fetchone()['cnt']
        fill_pct = filled / total * 100 if total else 0
        car_pct = with_car / total * 100 if total else 0
        gap = filled - with_car
        flag = " ⚠ GAP" if gap > 0 else ""
        print(f"  {h:<12s} {filled:>8,} {with_car:>10,} {fill_pct:>7.1f}% {car_pct:>7.1f}% {gap:>8,}{flag}")

    # ── 2. Feature fill rates (ML features) ──
    print(f"\n── ML Feature Fill Rates ──")
    ml_features = [
        'trade_size_points', 'same_ticker_signals_7d', 'same_ticker_signals_30d',
        'has_convergence', 'convergence_tier', 'person_trade_count',
        'person_hit_rate_30d', 'relative_position_size', 'insider_role',
        'sector', 'price_proximity_52wk', 'market_cap_bucket',
        'cluster_velocity', 'trade_pattern', 'disclosure_delay',
        'vix_at_signal', 'yield_curve_at_signal', 'credit_spread_at_signal',
        'days_to_earnings', 'days_to_catalyst',
        'momentum_1m', 'momentum_3m', 'momentum_6m',
        'volume_spike', 'insider_buy_ratio_90d', 'sector_avg_car',
        'vix_regime_interaction',
    ]
    print(f"  {'Feature':<30s} {'Non-NULL':>10s} {'Fill %':>8s}")
    print(f"  {'-'*30} {'-'*10} {'-'*8}")
    for feat in ml_features:
        non_null = conn.execute(
            f"SELECT COUNT(*) as cnt FROM signals WHERE {feat} IS NOT NULL"
        ).fetchone()['cnt']
        pct = non_null / total * 100 if total else 0
        flag = " ⚠ LOW" if pct < 30 else ""
        print(f"  {feat:<30s} {non_null:>10,} {pct:>7.1f}%{flag}")

    # ── 3. CAR sanity checks ──
    print(f"\n── CAR Sanity Checks ──")
    for h in ['5d', '30d', '90d', '180d', '365d']:
        stats = conn.execute(
            f"SELECT COUNT(*) as n, "
            f"AVG(car_{h}) as avg, "
            f"MIN(car_{h}) as mn, "
            f"MAX(car_{h}) as mx, "
            f"SUM(CASE WHEN ABS(car_{h}) > 2.0 THEN 1 ELSE 0 END) as extreme "
            f"FROM signals WHERE car_{h} IS NOT NULL"
        ).fetchone()
        n = stats['n']
        if n == 0:
            print(f"  {h}: no data")
            continue
        avg = stats['avg']
        mn = stats['mn']
        mx = stats['mx']
        extreme = stats['extreme']
        extreme_pct = extreme / n * 100 if n else 0
        flags = []
        if abs(avg) > 0.5:
            flags.append(f"avg={avg*100:+.1f}% seems high")
        if abs(mn) > 5.0 or abs(mx) > 5.0:
            flags.append(f"range [{mn*100:+.0f}%, {mx*100:+.0f}%]")
        if extreme_pct > 5:
            flags.append(f"{extreme_pct:.1f}% outliers >200%")
        flag_str = f"  ⚠ {'; '.join(flags)}" if flags else ""
        print(f"  {h}: n={n:,}  avg={avg*100:+.2f}%  "
              f"min={mn*100:+.1f}%  max={mx*100:+.1f}%  "
              f"|>200%|={extreme}{flag_str}")

    # ── 4. Source breakdown ──
    print(f"\n── Source Breakdown ──")
    for source in ['congress', 'edgar']:
        cnt = conn.execute(
            "SELECT COUNT(*) as cnt FROM signals WHERE source=?", (source,)
        ).fetchone()['cnt']
        with_price = conn.execute(
            "SELECT COUNT(*) as cnt FROM signals WHERE source=? AND price_at_signal IS NOT NULL", (source,)
        ).fetchone()['cnt']
        with_car = conn.execute(
            "SELECT COUNT(*) as cnt FROM signals WHERE source=? AND car_30d IS NOT NULL", (source,)
        ).fetchone()['cnt']
        price_pct = with_price / cnt * 100 if cnt else 0
        car_pct = with_car / cnt * 100 if cnt else 0
        print(f"  {source:<10s}: {cnt:>6,} signals  |  "
              f"price: {with_price:>6,} ({price_pct:.0f}%)  |  "
              f"car_30d: {with_car:>6,} ({car_pct:.0f}%)")

    # ── 5. Convergence breakdown ──
    print(f"\n── Convergence Distribution ──")
    for tier in [0, 1, 2]:
        cnt = conn.execute(
            "SELECT COUNT(*) as cnt FROM signals WHERE convergence_tier=?", (tier,)
        ).fetchone()['cnt']
        pct = cnt / total * 100 if total else 0
        print(f"  Tier {tier}: {cnt:>6,} ({pct:.1f}%)")

    # ── 6. Price file coverage check ──
    print(f"\n── Price File Coverage ──")
    tickers_row = conn.execute("SELECT DISTINCT ticker FROM signals").fetchall()
    total_tickers = len(tickers_row)
    tickers_with_files = 0
    short_files = 0
    for r in tickers_row:
        fpath = PRICE_HISTORY_DIR / f"{r['ticker']}.json"
        if fpath.exists():
            tickers_with_files += 1
            try:
                data = load_json(fpath)
                if len(data) < 600:  # less than ~2.4 years
                    short_files += 1
            except Exception:
                pass

    print(f"  Tickers in DB:     {total_tickers:,}")
    print(f"  With price files:  {tickers_with_files:,} ({tickers_with_files/total_tickers*100:.0f}%)")
    if short_files:
        print(f"  Short files (<600 days): {short_files:,}  ⚠ may miss 365d outcomes")

    print(f"\n{'='*70}\n")


def bootstrap(conn=None, edgar_days: int = 2555, incremental: bool = False):
    """Main bootstrap pipeline.

    Args:
        conn: SQLite connection (created if None)
        edgar_days: How far back to fetch EDGAR filings (default 2555 = ~7 years)
        incremental: If True, only fetch signals older than existing data (avoids re-fetching)
    """
    log.info("=" * 60)
    log.info("  ATLAS Adaptive Learning Engine — Historical Bootstrap")
    log.info("=" * 60)
    pipeline_start = time.time()
    step_times = {}

    if conn is None:
        conn = init_db()

    # Incremental mode: determine how far back we already have data
    if incremental:
        oldest = conn.execute(
            "SELECT MIN(signal_date) as oldest FROM signals WHERE signal_date IS NOT NULL"
        ).fetchone()
        if oldest and oldest['oldest']:
            oldest_date = datetime.strptime(oldest['oldest'], '%Y-%m-%d')
            target_start = datetime(2019, 1, 1)
            overlap_end = oldest_date + timedelta(days=90)
            if oldest_date <= target_start + timedelta(days=30):
                log.info(f"Incremental mode: data already starts at {oldest['oldest']} "
                         f"(target: 2019-01-01). Nothing to expand.")
                edgar_days = 0
            else:
                edgar_days = (datetime.now() - target_start).days
                log.info(f"Incremental mode: existing data starts {oldest['oldest']}, "
                         f"target: {target_start.strftime('%Y-%m-%d')}. "
                         f"Fetching {edgar_days} days back with 90d overlap. "
                         f"Dedup will skip {conn.execute('SELECT COUNT(*) FROM signals').fetchone()[0]} existing signals.")
        else:
            log.info("Incremental mode: no existing data — running full bootstrap")

    months_approx = edgar_days // 30
    # ── 1. Fetch historical EDGAR filings ────────────────────────────────
    log.info(f"\n[1/11] Fetching historical EDGAR Form 4 filings (~{months_approx} months)...")
    elapsed = _step_timer()
    edgar_filings = fetch_edgar_historical(days=edgar_days, raw_per_month=1500, keep_per_month=500)

    # ── 1b. Enrich via Form 4 XML — extract role, direction, buy value ────
    # Without this, we'd ingest grants/sales/exercises as buy signals (noise).
    # Uses concurrent requests with rate limiting (SEC allows 10 req/s).
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    import xml.etree.ElementTree as ET
    from backtest.backfill_edgar_xml import _normalize_role, _buy_value_to_points

    # Rate limiter: max 8 req/s (conservative vs SEC's 10/s limit)
    MAX_RPS = 8
    est_minutes = len(edgar_filings) / MAX_RPS / 60
    log.info(f"\n[1b/11] Enriching {len(edgar_filings)} filings from Form 4 XML "
             f"(~{est_minutes:.0f} min with {MAX_RPS} req/s concurrent)...")

    # Reusable session for connection pooling (TCP keep-alive)
    xml_session = requests.Session()
    xml_session.headers.update({'User-Agent': SEC_USER_AGENT, 'Accept': 'application/json'})

    rate_lock = threading.Lock()
    last_request_times = []

    def _rate_limited_parse(xml_url):
        """Fetch and parse Form 4 XML with rate limiting and session reuse."""
        with rate_lock:
            now = time.monotonic()
            while last_request_times and now - last_request_times[0] > 1.0:
                last_request_times.pop(0)
            if len(last_request_times) >= MAX_RPS:
                sleep_until = last_request_times[0] + 1.0
                wait = sleep_until - now
                if wait > 0:
                    time.sleep(wait)
                    now = time.monotonic()
                    while last_request_times and now - last_request_times[0] > 1.0:
                        last_request_times.pop(0)
            last_request_times.append(time.monotonic())

        try:
            r = xml_session.get(xml_url, timeout=10)
            if r.status_code != 200:
                return None
            root = ET.fromstring(r.content)
        except Exception:
            return None

        # Parse Form 4 XML inline (avoids double HTTP fetch from parse_form4_xml)
        ticker = (root.findtext('.//issuerTradingSymbol') or '').strip().upper()
        title = (root.findtext('.//officerTitle') or '').strip()
        is_officer = root.findtext('.//isOfficer', '0')
        is_director = root.findtext('.//isDirector', '0')
        is_10pct = root.findtext('.//isTenPercentOwner', '0')
        roles = []
        if is_officer in ('1', 'true'):   roles.append('Officer')
        if is_director in ('1', 'true'):  roles.append('Director')
        if is_10pct in ('1', 'true'):     roles.append('10% Owner')

        txn_dates, total_buy_value, total_sell_value, total_buy_shares = [], 0, 0, 0
        txn_codes = set()
        for txn in root.findall('.//nonDerivativeTransaction'):
            code = txn.findtext('.//transactionCode', '')
            txn_codes.add(code)
            try:
                shares = float(txn.findtext('.//transactionShares/value', '0'))
                price_str = txn.findtext('.//transactionPricePerShare/value', '0')
                price = float(price_str) if price_str else 0
            except ValueError:
                shares, price = 0, 0
            value = shares * price
            if code == 'P':
                total_buy_value += value
                total_buy_shares += shares
            elif code == 'S':
                total_sell_value += value
            td = txn.findtext('.//transactionDate/value', '')
            if td:
                txn_dates.append(td)

        if total_buy_value > 0 and total_sell_value == 0:
            direction = 'buy'
        elif total_sell_value > 0 and total_buy_value == 0:
            direction = 'sell'
        elif total_buy_value > 0 and total_sell_value > 0:
            direction = 'mixed'
        elif 'M' in txn_codes:
            direction = 'exercise'
        elif 'A' in txn_codes or 'G' in txn_codes:
            direction = 'grant'
        else:
            direction = 'other'

        disclosure_delay = None
        sig_date = root.findtext('.//signatureDate', '')
        earliest_txn = min(txn_dates) if txn_dates else ''
        if sig_date and earliest_txn:
            try:
                from datetime import datetime as dt_cls
                sig_dt = dt_cls.strptime(sig_date, '%Y-%m-%d')
                txn_dt = dt_cls.strptime(earliest_txn, '%Y-%m-%d')
                disclosure_delay = (sig_dt - txn_dt).days
            except ValueError:
                pass

        return {
            'ticker': ticker,
            'insider_role': _normalize_role(title, roles),
            'direction': direction,
            'buy_value': round(total_buy_value, 2),
            'buy_shares': int(total_buy_shares),
            'disclosure_delay': disclosure_delay,
            'trade_size_points': _buy_value_to_points(total_buy_value) if total_buy_value > 0 else None,
        }

    enriched_count = 0
    skipped_non_buy = 0
    xml_errors = 0
    enriched_filings = []

    filings_with_url = [(i, f) for i, f in enumerate(edgar_filings) if f.get('xml_url')]

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_filing = {}
        for idx, f in filings_with_url:
            future = executor.submit(_rate_limited_parse, f['xml_url'])
            future_to_filing[future] = (idx, f)

        processed = 0
        for future in as_completed(future_to_filing):
            idx, f = future_to_filing[future]
            processed += 1

            xml_data = future.result()
            if not xml_data:
                xml_errors += 1
                continue

            if xml_data['direction'] != 'buy':
                skipped_non_buy += 1
                continue

            f['_xml'] = xml_data
            enriched_filings.append(f)
            enriched_count += 1

            if processed % 500 == 0:
                log.info(f"  XML enrichment: {processed}/{len(filings_with_url)} processed, "
                         f"{enriched_count} buys, {skipped_non_buy} non-buy skipped")

    xml_session.close()
    log.info(f"  XML enrichment complete: {enriched_count} buys / "
             f"{len(edgar_filings)} total ({skipped_non_buy} non-buy, {xml_errors} errors)")

    edgar_inserted = 0
    for f in enriched_filings:
        ticker = f.get('_matched_ticker') or match_edgar_ticker(f.get('company', ''))
        if not ticker:
            continue

        xml = f.get('_xml', {})
        signal = {
            'ticker': ticker,
            'signal_date': f.get('date', ''),
            'source': 'edgar',
            'insider_name': f.get('insider', ''),
            'insider_role': xml.get('insider_role', ''),
            'transaction_type': 'Purchase',
            'trade_size_points': xml.get('trade_size_points'),
            'disclosure_delay': xml.get('disclosure_delay'),
            'representative': None,
            'sector': get_sector(ticker),
            'market_cap_bucket': get_market_cap_bucket(ticker),
        }
        if insert_signal(conn, signal):
            edgar_inserted += 1

    step_times['1_edgar_fetch'] = elapsed()
    log.info(f"EDGAR: {len(edgar_filings)} raw → {enriched_count} buys → {edgar_inserted} inserted  [{elapsed()}s]")

    # ── 2. Fetch congressional trades ────────────────────────────────────
    log.info("\n[2/11] Fetching congressional trades...")
    elapsed = _step_timer()
    congress_trades = fetch_congress_trades()

    congress_inserted = 0
    for t in congress_trades:
        tx = (t.get('Transaction') or '').lower()
        if 'purchase' not in tx and 'buy' not in tx:
            continue
        ticker = (t.get('Ticker') or '').strip().upper()
        if not ticker or len(ticker) > 5:
            continue
        date = t.get('TransactionDate') or t.get('Date') or ''
        if not date:
            continue

        signal = {
            'ticker': ticker,
            'signal_date': date,
            'source': 'congress',
            'representative': t.get('Representative', ''),
            'party': t.get('Party', ''),
            'chamber': t.get('Chamber', ''),
            'trade_size_range': t.get('Range', ''),
            'trade_size_points': range_to_base_points(t.get('Range', '')),
            'insider_name': None,
            'disclosure_delay': t.get('DisclosureDelay'),
            'sector': get_sector(ticker),
        }
        if insert_signal(conn, signal):
            congress_inserted += 1

    step_times['2_congress_fetch'] = elapsed()
    log.info(f"Congress: {len(congress_trades)} total → {congress_inserted} purchase signals inserted  [{elapsed()}s]")

    # ── 2b. Build sector + market cap maps ────────────────────────────────
    log.info("\n[2b/11] Building sector + market cap maps...")
    elapsed = _step_timer()
    tickers_for_map = [r['ticker'] for r in conn.execute("SELECT DISTINCT ticker FROM signals").fetchall()]
    fmp_key = os.environ.get('FMP_API_KEY', 'UefVEEvF1XXtpgWcsidPCGxcDJ6N0kXv')
    if fmp_key:
        build_sector_map(api_key=fmp_key, tickers=tickers_for_map)
        log.info("Sector + market cap maps built from FMP")
    else:
        from backtest.sector_map import bootstrap_sector_map_yfinance
        bootstrap_sector_map_yfinance(tickers_for_map)
        log.info("Sector + market cap maps built from yfinance (fallback)")
    step_times['2b_sector_cap_map'] = elapsed()
    log.info(f"  [{elapsed()}s]")

    # ── 3. Update aggregate features ─────────────────────────────────────
    log.info("\n[3/11] Computing aggregate features (clusters, convergence)...")
    elapsed = _step_timer()
    agg = update_aggregate_features(conn)
    step_times['3_aggregate'] = elapsed()
    log.info(f"Updated {agg} ticker-date pairs  [{elapsed()}s]")

    # ── 4. Collect price history ─────────────────────────────────────────
    log.info("\n[4/11] Collecting price history for all tickers...")
    elapsed = _step_timer()
    tickers_row = conn.execute("SELECT DISTINCT ticker FROM signals").fetchall()
    tickers = [r['ticker'] for r in tickers_row]

    # Always ensure SPY is collected
    if 'SPY' not in tickers:
        tickers.append('SPY')

    # Lookback must cover: oldest signal date + 365d outcome window
    # Signals start ~2022-11, so need ~1300 days from today to cover full range
    lookback = 1300

    import gc

    success = 0
    price_failures = []
    for i, ticker in enumerate(tickers):
        try:
            if collect_price_for_ticker(ticker, lookback_days=lookback):
                success += 1
            else:
                price_failures.append(ticker)
        except OSError as e:
            log.warning(f"OS error for {ticker}: {e} — forcing cleanup")
            price_failures.append(ticker)
            gc.collect()
            time.sleep(2)
        if (i + 1) % 50 == 0:
            log.info(f"  ...{i+1}/{len(tickers)} tickers processed ({success} with data)")
            # Force-close stale yfinance connections to avoid fd exhaustion
            gc.collect()
        time.sleep(0.3)  # rate-limit courtesy

    step_times['4_price_collect'] = elapsed()
    log.info(f"Price history: {success}/{len(tickers)} tickers collected  [{elapsed()}s]")
    if price_failures:
        log.info(f"  Failed tickers ({len(price_failures)}): {', '.join(price_failures[:20])}"
                 f"{'...' if len(price_failures) > 20 else ''}")

    # ── SPY coverage check ──
    spy_index = load_price_index('SPY')
    _check_spy_coverage(spy_index, conn)

    # ── 5. Backfill outcomes ─────────────────────────────────────────────
    log.info("\n[5/11] Backfilling outcomes (returns + CARs)...")
    elapsed = _step_timer()
    filled = backfill_outcomes(conn, spy_index)
    step_times['5_backfill'] = elapsed()
    log.info(f"Backfilled outcomes for {filled} signals  [{elapsed()}s]")

    # Quick sanity check: filled vs CAR match
    for h in ['30d', '365d']:
        f_cnt = conn.execute(f"SELECT COUNT(*) as cnt FROM signals WHERE outcome_{h}_filled=1").fetchone()['cnt']
        c_cnt = conn.execute(f"SELECT COUNT(*) as cnt FROM signals WHERE outcome_{h}_filled=1 AND car_{h} IS NOT NULL").fetchone()['cnt']
        if f_cnt != c_cnt:
            log.warning(f"  ⚠ {h}: {f_cnt} filled but only {c_cnt} have CAR (gap={f_cnt-c_cnt})")
        else:
            log.info(f"  ✓ {h}: {f_cnt} filled, {c_cnt} with CAR — no gap")

    # ── 6. Person track records ──────────────────────────────────────────
    log.info("\n[6/11] Computing person track records...")
    elapsed = _step_timer()
    person_updated = update_person_track_records(conn)
    step_times['6_person_records'] = elapsed()
    log.info(f"Updated track records for {person_updated} signals  [{elapsed()}s]")

    # ── 7. Enrich price-based and research-backed features ────────────────
    log.info("\n[7/11] Enriching signal features (52wk proximity, trade pattern, insider role, market cap)...")
    elapsed = _step_timer()
    enriched = enrich_signal_features(conn)
    step_times['7_enrich_features'] = elapsed()
    log.info(f"Enriched {enriched} signal features  [{elapsed()}s]")

    # ── 7b. Backfill sector + market_cap_bucket for NULL signals ──
    log.info("\n[7b/11] Backfilling sector + market_cap_bucket for NULL signals...")
    elapsed_7b = _step_timer()
    null_sector_cnt = 0
    null_cap_cnt = 0
    for row in conn.execute("SELECT DISTINCT ticker FROM signals WHERE sector IS NULL").fetchall():
        sector = get_sector(row['ticker'])
        if sector:
            conn.execute("UPDATE signals SET sector=? WHERE ticker=? AND sector IS NULL",
                         (sector, row['ticker']))
            null_sector_cnt += conn.execute("SELECT changes()").fetchone()[0]
    for row in conn.execute("SELECT DISTINCT ticker FROM signals WHERE market_cap_bucket IS NULL").fetchall():
        bucket = get_market_cap_bucket(row['ticker'])
        if bucket:
            conn.execute("UPDATE signals SET market_cap_bucket=? WHERE ticker=? AND market_cap_bucket IS NULL",
                         (bucket, row['ticker']))
            null_cap_cnt += conn.execute("SELECT changes()").fetchone()[0]
    conn.commit()
    step_times['7b_sector_cap_backfill'] = elapsed_7b()
    log.info(f"Backfilled {null_sector_cnt} sectors, {null_cap_cnt} market cap buckets  [{elapsed_7b()}s]")

    # ── 8. Enrich market context (VIX, yield curve, credit spread) ────────
    log.info("\n[8/11] Enriching market context from FRED...")
    elapsed = _step_timer()
    market_enriched = enrich_market_context(conn)
    step_times['8_market_context'] = elapsed()
    log.info(f"Enriched {market_enriched} signals with market context  [{elapsed()}s]")

    # ── 9. Feature analysis + weight generation ──────────────────────────
    log.info("\n[9/11] Running feature analysis...")
    elapsed = _step_timer()
    stats = compute_feature_stats(conn)
    if stats:
        weights = generate_weights_from_stats(conn)
        threshold = weights.pop('_optimal_threshold', 65)
        output = {
            **weights,
            "generated": datetime.now(tz=timezone.utc).isoformat(),
            "optimal_threshold": threshold,
            "method": "bootstrap_feature_importance",
        }
        save_json(OPTIMAL_WEIGHTS, output)
        log.info(f"Initial weights saved to {OPTIMAL_WEIGHTS}")
    else:
        log.info("Not enough outcome data yet for feature analysis")
    step_times['9_feature_analysis'] = elapsed()
    log.info(f"  [{elapsed()}s]")

    # ── 10. ML walk-forward training ──────────────────────────────────────
    log.info("\n[10/11] Running ML walk-forward training...")
    elapsed = _step_timer()
    wf_result = None
    reg_result = None
    try:
        from backtest.ml_engine import walk_forward_train
        wf_result = walk_forward_train(conn)
        if wf_result and wf_result.folds:
            avg_ic = sum(f['ic'] for f in wf_result.folds) / len(wf_result.folds)
            log.info(f"ML training complete: {len(wf_result.folds)} folds, "
                     f"avg IC={avg_ic:.4f}, OOS IC={wf_result.oos_ic:.4f}")
            # Per-fold detail
            for i, fold in enumerate(wf_result.folds):
                log.info(f"  Fold {i+1}: {fold['train_start']}→{fold['train_end']} | "
                         f"test {fold['test_start']}→{fold['test_end']} | "
                         f"n_train={fold['n_train']} n_test={fold['n_test']} | "
                         f"IC={fold['ic']:.4f} hit={fold['hit_rate']:.1%}")
            # Top features
            if wf_result.feature_importance:
                top5 = list(wf_result.feature_importance.items())[:5]
                log.info(f"  Top features: {', '.join(f'{k}={v:.3f}' for k, v in top5)}")
        else:
            log.info("ML training: not enough data for walk-forward validation")
    except Exception as e:
        log.warning(f"ML training skipped: {e}")
        import traceback
        traceback.print_exc()
    step_times['10_ml_training'] = elapsed()
    log.info(f"  [{elapsed()}s]")

    # ── 11. ML walk-forward regression ─────────────────────────────────────
    log.info("\n[11/11] Running ML regression (continuous CAR prediction)...")
    elapsed = _step_timer()
    try:
        from backtest.ml_engine import walk_forward_regression
        reg_result = walk_forward_regression(conn)
        if reg_result and reg_result.folds:
            avg_ic = sum(f['ic'] for f in reg_result.folds) / len(reg_result.folds)
            log.info(f"Regression complete: {len(reg_result.folds)} folds, "
                     f"avg IC={avg_ic:.4f}, OOS IC={reg_result.oos_ic:.4f}, "
                     f"RMSE={reg_result.oos_rmse:.4f}")
            if reg_result.feature_importance:
                top5 = list(reg_result.feature_importance.items())[:5]
                log.info(f"  Top regression features: {', '.join(f'{k}={v:.3f}' for k, v in top5)}")
        else:
            log.info("Regression: not enough data for walk-forward validation")
    except Exception as e:
        log.warning(f"Regression skipped: {e}")
        import traceback
        traceback.print_exc()
    step_times['11_regression'] = elapsed()
    log.info(f"  [{elapsed()}s]")

    # ── Generate dashboard + summary ──
    generate_dashboard(conn)
    print_summary(conn)

    # ── Diagnostics ──
    generate_analysis_report(conn, ml_result=wf_result, reg_result=reg_result)
    generate_diagnostics_html(conn, ml_result=wf_result, reg_result=reg_result)

    # ── Data quality report ──
    _print_data_quality_report(conn)

    # ── Pipeline timing summary ──
    total_time = round(time.time() - pipeline_start, 1)
    print(f"\n── Pipeline Timing ──")
    for step, secs in step_times.items():
        pct = secs / total_time * 100 if total_time else 0
        bar = '█' * int(pct / 2)
        print(f"  {step:<25s} {secs:>7.1f}s  {pct:>5.1f}%  {bar}")
    print(f"  {'TOTAL':<25s} {total_time:>7.1f}s")

    log.info("\nBootstrap complete!")
    return conn


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ATLAS Historical Bootstrap')
    parser.add_argument('--edgar-days', type=int, default=2555,
                        help='EDGAR lookback days (default 2555 = ~7 years)')
    parser.add_argument('--incremental', action='store_true',
                        help='Only fetch signals older than existing data (90d overlap)')
    args = parser.parse_args()
    bootstrap(edgar_days=args.edgar_days, incremental=args.incremental)
