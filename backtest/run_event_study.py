"""
run_event_study.py — compute Cumulative Abnormal Returns (CAR) for aggregate
per-ticker convergence signals (mirrors the frontend scoring engine).

Usage:
    python backtest/run_event_study.py

Inputs:
    data/congress_feed.json
    data/edgar_feed.json
    data/price_history/{TICKER}.json

Output:
    data/backtest_results.json — one record per ticker with aggregate score + CAR at 5d/30d/90d
"""

import sys
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.shared import (
    CONGRESS_FEED, EDGAR_FEED, PRICE_HISTORY_DIR, BACKTEST_RESULTS,
    DEFAULT_WEIGHTS, BENCHMARK, load_json, save_json, match_edgar_ticker,
    range_to_base_points
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

TOLERANCE_DAYS = 5  # look ±5 days around target date for price
CONGRESS_WINDOW_DAYS = 30
EDGAR_WINDOW_DAYS = 14


def load_price_index(ticker: str) -> dict:
    """Load {date: close} index for a ticker, or empty dict if no cache."""
    path = PRICE_HISTORY_DIR / f"{ticker}.json"
    if not path.exists():
        return {}
    candles = load_json(path)
    return {date: v['c'] for date, v in candles.items()}


def get_forward_return(price_index: dict, event_date: str, days: int) -> float | None:
    """
    Find the return from event_date to event_date+days.
    Searches within ±TOLERANCE_DAYS window for available trading day prices.
    Returns None if either price is unavailable.
    """
    if event_date not in price_index:
        return None

    base_price = price_index[event_date]
    if base_price is None or base_price == 0:
        return None

    event_dt = datetime.strptime(event_date, '%Y-%m-%d')
    target_dt = event_dt + timedelta(days=days)

    offsets = sorted(range(-TOLERANCE_DAYS, TOLERANCE_DAYS + 1), key=abs)
    for offset in offsets:
        candidate = (target_dt + timedelta(days=offset)).strftime('%Y-%m-%d')
        if candidate == event_date:
            continue
        if candidate in price_index:
            forward_price = price_index[candidate]
            if forward_price and forward_price > 0:
                return (forward_price - base_price) / base_price

    return None


def compute_car(stock_return: float, benchmark_return: float) -> float:
    """Cumulative Abnormal Return = stock return - benchmark return."""
    return stock_return - benchmark_return


def compute_member_track_records(trades: list) -> dict:
    """
    Build per-member track record from congressional purchase trades.
    Returns {member_name: {avg_excess, win_rate, n_trades, quartile}}
    Uses QuiverQuant's pre-computed ExcessReturn as ground truth.
    """
    member_returns = defaultdict(list)
    for t in trades:
        tx = (t.get('Transaction') or '').lower()
        if 'purchase' not in tx and 'buy' not in tx:
            continue
        member = t.get('Representative') or ''
        excess = t.get('ExcessReturn')
        if member and excess is not None:
            try:
                member_returns[member].append(float(excess))
            except (TypeError, ValueError):
                pass

    records = {}
    all_avgs = []
    for member, rets in member_returns.items():
        avg = sum(rets) / len(rets)
        win_rate = sum(1 for r in rets if r > 0) / len(rets)
        records[member] = {'avg_excess': avg, 'win_rate': win_rate, 'n_trades': len(rets)}
        all_avgs.append(avg)

    if all_avgs:
        all_avgs.sort()
        n = len(all_avgs)
        q1_threshold = all_avgs[int(n * 0.75)]
        q2_threshold = all_avgs[int(n * 0.50)]
        for member, rec in records.items():
            avg = rec['avg_excess']
            if avg >= q1_threshold:
                rec['quartile'] = 1
            elif avg >= q2_threshold:
                rec['quartile'] = 2
            elif avg >= 0:
                rec['quartile'] = 3
            else:
                rec['quartile'] = 4

    return records


def score_congress_aggregate(trades: list, weights: dict, track_records: dict) -> dict:
    """
    Score all congressional purchase trades for a single ticker.
    Mirrors frontend scoreCongressTicker() logic but WITHOUT live decay —
    backtest evaluates signal strength at formation time, not decayed by today.
    Returns {score, count, cluster_bonus, members, trade_ranges, raw_points}.
    """
    if not trades:
        return {'score': 0, 'count': 0, 'cluster_bonus': 0, 'members': [],
                'trade_ranges': [], 'raw_points': []}

    pts = 0.0
    members = []
    trade_ranges = []
    raw_points = []

    for t in trades:
        raw_pts = range_to_base_points(t.get('Range', ''))

        # Track record bonus
        member = t.get('Representative', '')
        rec = track_records.get(member, {})
        quartile = rec.get('quartile', 4)
        if quartile == 1:
            raw_pts += weights.get('congress_track_record_q1', 0)
        elif quartile == 2:
            raw_pts += weights.get('congress_track_record_q2', 0)

        pts += raw_pts
        members.append(member)
        trade_ranges.append(t.get('Range', ''))
        raw_points.append(round(raw_pts, 4))

    cluster_bonus = weights.get('congress_cluster_bonus', 15) if len(trades) >= 3 else 0
    score = min(pts + cluster_bonus, 40)

    return {
        'score': round(score, 2),
        'count': len(trades),
        'cluster_bonus': cluster_bonus,
        'members': members,
        'trade_ranges': trade_ranges,
        'raw_points': raw_points,
    }


def score_edgar_aggregate(filings: list, weights: dict) -> dict:
    """
    Score all EDGAR filings for a single ticker.
    Mirrors frontend scoreEdgarTicker() logic.
    Returns {score, count, cluster_bonus}.
    """
    if not filings:
        return {'score': 0, 'count': 0, 'cluster_bonus': 0}

    n = len(filings)
    base = min(n * weights.get('edgar_base_per_filing', 6), 25)

    if n >= 3:
        cluster_bonus = weights.get('edgar_cluster_3plus', 15)
    elif n >= 2:
        cluster_bonus = weights.get('edgar_cluster_2', 10)
    else:
        cluster_bonus = 0

    score = min(base + cluster_bonus, 40)

    return {
        'score': round(score, 2),
        'count': n,
        'cluster_bonus': cluster_bonus,
    }


def build_ticker_events(congress_data: list, edgar_data: list, weights: dict,
                        track_records: dict, spy_index: dict) -> list:
    """
    Build aggregate per-ticker convergence events.
    Groups ALL trades/filings by ticker (no rolling window cutoff — backtest
    evaluates every signal cluster in the feed), computes aggregate scores,
    and CARs from the most recent event date.
    Returns (list of event dicts, skipped count).
    """
    # Group congress purchases by ticker (all purchases in feed)
    congress_by_ticker = defaultdict(list)
    for t in congress_data:
        tx = (t.get('Transaction') or '').lower()
        if 'purchase' not in tx and 'buy' not in tx:
            continue
        ticker = (t.get('Ticker') or '').strip().upper()
        if not ticker or len(ticker) > 5:
            continue
        congress_by_ticker[ticker].append(t)

    # Group EDGAR filings by ticker (all filings in feed)
    edgar_by_ticker = defaultdict(list)
    for f in edgar_data:
        ticker = match_edgar_ticker(f.get('company', ''))
        if not ticker:
            continue
        edgar_by_ticker[ticker].append(f)

    # Build universe of all tickers with any activity
    all_tickers = sorted(set(list(congress_by_ticker.keys()) + list(edgar_by_ticker.keys())))

    events = []
    skipped = 0

    for ticker in all_tickers:
        cong_trades = congress_by_ticker.get(ticker, [])
        edgar_filings = edgar_by_ticker.get(ticker, [])

        # Score each hub
        cong_result = score_congress_aggregate(cong_trades, weights, track_records)
        edgar_result = score_edgar_aggregate(edgar_filings, weights)

        # Convergence boost
        convergence_boost = 0
        if cong_result['score'] > 0 and edgar_result['score'] > 0:
            convergence_boost = weights.get('convergence_boost', 20)

        total_score = cong_result['score'] + edgar_result['score'] + convergence_boost

        # Determine event_date = most recent trade or filing date
        all_dates = []
        for t in cong_trades:
            d = t.get('TransactionDate') or t.get('ReportDate') or ''
            if d:
                all_dates.append(d)
        for f in edgar_filings:
            d = f.get('date', '')
            if d:
                all_dates.append(d)

        if not all_dates:
            continue

        event_date = max(all_dates)

        # Load price data and compute CARs
        price_index = load_price_index(ticker)
        if not price_index:
            skipped += 1
            continue

        # Determine convergence type
        has_congress = cong_result['score'] > 0
        has_edgar = edgar_result['score'] > 0
        if has_congress and has_edgar:
            convergence = 'both'
        elif has_congress:
            convergence = 'congress'
        elif has_edgar:
            convergence = 'edgar'
        else:
            convergence = 'none'

        event = {
            'ticker': ticker,
            'event_date': event_date,
            'convergence': convergence,
            'congress_score': cong_result['score'],
            'congress_count': cong_result['count'],
            'congress_cluster_bonus': cong_result['cluster_bonus'],
            'edgar_score': edgar_result['score'],
            'edgar_count': edgar_result['count'],
            'edgar_cluster_bonus': edgar_result['cluster_bonus'],
            'convergence_boost': convergence_boost,
            'total_score': round(total_score, 2),
            'members': cong_result.get('members', []),
            'trade_ranges': cong_result.get('trade_ranges', []),
            'raw_points': cong_result.get('raw_points', []),
        }

        # Compute CARs at 5d/30d/90d
        for window in [5, 30, 90]:
            stock_ret = get_forward_return(price_index, event_date, window)
            spy_ret = get_forward_return(spy_index, event_date, window) if spy_index else None
            if stock_ret is not None and spy_ret is not None:
                event[f'car_{window}d'] = round(compute_car(stock_ret, spy_ret), 6)
                event[f'stock_ret_{window}d'] = round(stock_ret, 6)
            else:
                event[f'car_{window}d'] = None
                event[f'stock_ret_{window}d'] = None

        events.append(event)

    return events, skipped


def run_event_study() -> list:
    """Main function: compute CAR for all aggregate per-ticker signal events."""
    congress_data = load_json(CONGRESS_FEED).get('trades', [])
    edgar_data = load_json(EDGAR_FEED).get('filings', [])

    track_records = compute_member_track_records(congress_data)
    spy_index = load_price_index(BENCHMARK)
    if not spy_index:
        log.warning("No SPY price history. CARs will use raw returns (no benchmark adjustment).")

    weights = DEFAULT_WEIGHTS

    log.info(f"Processing {len(congress_data)} congressional trades + {len(edgar_data)} EDGAR filings...")
    events, skipped = build_ticker_events(congress_data, edgar_data, weights, track_records, spy_index)

    # Log score distribution
    scores = [e['total_score'] for e in events]
    with_car = [e for e in events if e.get('car_30d') is not None]
    above_65 = [e for e in events if e['total_score'] >= 65]

    log.info(f"Aggregate events: {len(events)} tickers, {skipped} skipped (no price data)")
    if scores:
        log.info(f"Score range: {min(scores):.1f} to {max(scores):.1f}")
    log.info(f"Events with CAR data: {len(with_car)}")
    log.info(f"Events above 65: {len(above_65)}")
    if with_car:
        cars = [e['car_30d'] for e in with_car]
        log.info(f"Avg 30d CAR: {sum(cars)/len(cars):.4f}")

    return events


def main():
    events = run_event_study()
    save_json(BACKTEST_RESULTS, {
        "generated": datetime.now(tz=timezone.utc).isoformat(),
        "n_events": len(events),
        "events": events,
    })
    log.info(f"Saved {len(events)} events to {BACKTEST_RESULTS}")


if __name__ == '__main__':
    main()
