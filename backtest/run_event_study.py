"""
run_event_study.py — compute Cumulative Abnormal Returns (CAR) for all signal events.

Usage:
    python backtest/run_event_study.py

Inputs:
    data/congress_feed.json
    data/edgar_feed.json
    data/price_history/{TICKER}.json

Output:
    data/backtest_results.json — one record per event with CAR at 5d/30d/90d
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


def load_price_index(ticker: str) -> dict:
    """Load {date: close} index for a ticker, or empty dict if no cache."""
    path = PRICE_HISTORY_DIR / f"{ticker}.json"
    if not path.exists():
        return {}
    candles = load_json(path)
    # candles format: {date_str: {o, h, l, c, v}}
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

    # Search within tolerance window for the closest available trading day.
    # Sort offsets by absolute distance so we prefer the nearest date to the target.
    offsets = sorted(range(-TOLERANCE_DAYS, TOLERANCE_DAYS + 1), key=abs)
    for offset in offsets:
        candidate = (target_dt + timedelta(days=offset)).strftime('%Y-%m-%d')
        if candidate == event_date:
            continue  # never use the event date itself as the forward price
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
    Returns {member_name: {avg_excess, win_rate, n_trades}}
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

    # Compute quartile thresholds for track record bonuses
    if all_avgs:
        all_avgs.sort()
        n = len(all_avgs)
        q1_threshold = all_avgs[int(n * 0.75)]  # top 25%
        q2_threshold = all_avgs[int(n * 0.50)]  # top 50%
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


def score_congress_event(event: dict, weights: dict, track_records: dict, event_date: str) -> float:
    """Score a single congressional purchase event using provided weights."""
    base_pts = range_to_base_points(event.get('Range', ''))

    # Apply temporal decay
    half_life = weights.get('decay_half_life_days', 21)
    now = datetime.now(tz=timezone.utc)
    try:
        event_dt = datetime.strptime(event_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        days_since = (now - event_dt).days
    except (ValueError, TypeError):
        days_since = 0

    decay = 0.5 ** (max(0, days_since) / half_life)
    pts = base_pts * decay

    # Track record bonus
    member = event.get('Representative', '')
    rec = track_records.get(member, {})
    quartile = rec.get('quartile', 4)
    if quartile == 1:
        pts += weights.get('congress_track_record_q1', 0)
    elif quartile == 2:
        pts += weights.get('congress_track_record_q2', 0)

    return min(pts, 40)  # cap


def classify_convergence(ticker: str, congress_tickers: set, edgar_tickers: set) -> str:
    """Return 'convergence', 'congress', 'edgar', or 'none'."""
    has_c = ticker in congress_tickers
    has_e = ticker in edgar_tickers
    if has_c and has_e:
        return 'convergence'
    elif has_c:
        return 'congress'
    elif has_e:
        return 'edgar'
    return 'none'


def run_event_study() -> list:
    """
    Main function: compute CAR for all signal events.
    Returns list of event records.
    """
    congress_data = load_json(CONGRESS_FEED).get('trades', [])
    edgar_data = load_json(EDGAR_FEED).get('filings', [])

    # Build sets of active tickers for convergence detection
    congress_purchase_tickers = {
        t['Ticker'] for t in congress_data
        if 'purchase' in (t.get('Transaction') or '').lower() or 'buy' in (t.get('Transaction') or '').lower()
        if t.get('Ticker')
    }
    edgar_matched_tickers = {
        match_edgar_ticker(f.get('company', ''))
        for f in edgar_data
        if match_edgar_ticker(f.get('company', ''))
    }

    # Compute member track records
    track_records = compute_member_track_records(congress_data)

    # Load SPY prices as benchmark
    spy_index = load_price_index(BENCHMARK)
    if not spy_index:
        log.warning("No SPY price history. CARs will use raw returns (no benchmark adjustment).")

    weights = DEFAULT_WEIGHTS
    events = []
    skipped = 0

    # Process congressional purchase events
    log.info(f"Processing {len(congress_data)} congressional trades...")
    for trade in congress_data:
        tx = (trade.get('Transaction') or '').lower()
        if 'purchase' not in tx and 'buy' not in tx:
            continue

        ticker = (trade.get('Ticker') or '').strip().upper()
        if not ticker or len(ticker) > 5:
            continue

        event_date = trade.get('TransactionDate') or trade.get('ReportDate')
        if not event_date:
            continue

        price_index = load_price_index(ticker)
        if not price_index:
            skipped += 1
            continue

        convergence = classify_convergence(ticker, congress_purchase_tickers, edgar_matched_tickers)

        event = {
            "ticker": ticker,
            "event_date": event_date,
            "event_type": "congress",
            "member": trade.get('Representative', ''),
            "range": trade.get('Range', ''),
            "party": trade.get('Party', ''),
            "house": trade.get('House', ''),
            "excess_return_qq": trade.get('ExcessReturn'),  # ground truth from QQ
            "price_change_qq": trade.get('PriceChange'),
            "spy_change_qq": trade.get('SPYChange'),
            "convergence": convergence,
        }

        # Compute forward returns
        for window in [5, 30, 90]:
            stock_ret = get_forward_return(price_index, event_date, window)
            spy_ret = get_forward_return(spy_index, event_date, window) if spy_index else None
            if stock_ret is not None and spy_ret is not None:
                event[f"car_{window}d"] = round(compute_car(stock_ret, spy_ret), 6)
                event[f"stock_ret_{window}d"] = round(stock_ret, 6)
            else:
                event[f"car_{window}d"] = None
                event[f"stock_ret_{window}d"] = None

        # Score this event
        event["score"] = round(score_congress_event(trade, weights, track_records, event_date), 2)
        member_rec = track_records.get(trade.get('Representative', ''), {})
        event["member_quartile"] = member_rec.get('quartile', 4)
        event["member_avg_excess"] = member_rec.get('avg_excess')

        events.append(event)

    # Process EDGAR filing events
    log.info(f"Processing {len(edgar_data)} EDGAR filings...")
    for filing in edgar_data:
        ticker = match_edgar_ticker(filing.get('company', ''))
        if not ticker:
            continue

        event_date = filing.get('date')
        if not event_date:
            continue

        price_index = load_price_index(ticker)
        if not price_index:
            skipped += 1
            continue

        convergence = classify_convergence(ticker, congress_purchase_tickers, edgar_matched_tickers)

        event = {
            "ticker": ticker,
            "event_date": event_date,
            "event_type": "edgar",
            "company": filing.get('company', ''),
            "insider": filing.get('insider', ''),
            "convergence": convergence,
        }

        for window in [5, 30, 90]:
            stock_ret = get_forward_return(price_index, event_date, window)
            spy_ret = get_forward_return(spy_index, event_date, window) if spy_index else None
            if stock_ret is not None and spy_ret is not None:
                event[f"car_{window}d"] = round(compute_car(stock_ret, spy_ret), 6)
            else:
                event[f"car_{window}d"] = None

        events.append(event)

    log.info(f"Event study complete: {len(events)} events processed, {skipped} skipped (no price data)")
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
