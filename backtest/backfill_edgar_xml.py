"""
backfill_edgar_xml.py — Backfill missing EDGAR signal data from Form 4 XML.

The bootstrap fetched EDGAR filings from EFTS but only captured company name,
insider name, and date — no role, buy value, direction, or disclosure delay.
This script re-fetches those filings from EFTS to get accession numbers, then
fetches the actual Form 4 XML to extract the missing fields.

CRITICAL: The bootstrap ingested ALL Form 4 types (sales, exercises, grants)
because it had no direction data. This script identifies and removes non-purchase
signals, keeping only genuine insider buys.

Usage:
    python backtest/backfill_edgar_xml.py --dry-run --limit 20   # Preview
    python backtest/backfill_edgar_xml.py                         # Full run

SEC rate limit: ~8 req/sec (0.12s delay per XML, 0.5s per EFTS search).
Estimated time: ~60-90 min for 10K signals.
"""

import os
import sys
import time
import logging
import sqlite3
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.shared import SIGNALS_DB, SEC_USER_AGENT, match_edgar_ticker
from backtest.learning_engine import init_db

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

SEC_DELAY = 0.12  # ~8 req/sec for XML fetches
EFTS_DELAY = 0.5  # Slower for EFTS search queries


def _normalize_role(title: str, roles: list) -> str:
    """Normalize insider role from Form 4 XML fields."""
    if title:
        t = title.upper()
        if 'CEO' in t or 'CHIEF EXECUTIVE' in t:
            return 'CEO'
        if 'CFO' in t or 'CHIEF FINANCIAL' in t:
            return 'CFO'
        if 'COO' in t or 'CHIEF OPERATING' in t:
            return 'COO'
        if 'PRESIDENT' in t:
            return 'President'
        if 'VP' in t or 'VICE PRESIDENT' in t:
            return 'VP'
        if 'GENERAL COUNSEL' in t:
            return 'Officer'
    if roles:
        if '10% Owner' in roles:
            return '10% Owner'
        if 'Director' in roles:
            return 'Director'
        if 'Officer' in roles:
            return 'Officer'
    return 'Other'


def _buy_value_to_points(value: float) -> int:
    """Map dollar buy value to trade size points."""
    if value >= 5_000_000: return 15
    if value >= 1_000_000: return 15
    if value >= 500_000:   return 12
    if value >= 250_000:   return 10
    if value >= 100_000:   return 8
    if value >= 50_000:    return 6
    if value >= 15_000:    return 5
    return 3


def fetch_efts_for_insider(insider_name: str, signal_date: str,
                           headers: dict) -> dict | None:
    """Search EFTS for a Form 4 filing matching this insider + date.

    Returns dict with xml_url if found, None otherwise.
    """
    try:
        d = datetime.strptime(signal_date, '%Y-%m-%d')
    except ValueError:
        return None

    start = (d - timedelta(days=1)).strftime('%Y-%m-%d')
    end = (d + timedelta(days=1)).strftime('%Y-%m-%d')

    search_name = insider_name.strip()
    if not search_name:
        return None

    url = (
        f'https://efts.sec.gov/LATEST/search-index'
        f'?forms=4'
        f'&dateRange=custom'
        f'&startdt={start}'
        f'&enddt={end}'
        f'&q=%22{requests.utils.quote(search_name)}%22'
    )

    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:
        return None

    hits = data.get('hits', {}).get('hits', [])
    for h in hits:
        src = h.get('_source', {})
        _id = h.get('_id', '')
        accession = src.get('adsh', '')
        ciks = src.get('ciks', [])
        company_cik = ciks[-1] if len(ciks) > 1 else (ciks[0] if ciks else '')

        if company_cik and accession and ':' in _id:
            try:
                numeric_cik = str(int(company_cik))
                clean_acc = accession.replace('-', '')
                xml_filename = _id.split(':', 1)[1]
                xml_url = (
                    f'https://www.sec.gov/Archives/edgar/data/'
                    f'{numeric_cik}/{clean_acc}/{xml_filename}'
                )
                return {'xml_url': xml_url, 'accession': accession}
            except (ValueError, IndexError):
                continue

    return None


def parse_form4_xml(xml_url: str, headers: dict) -> dict | None:
    """Fetch and parse a Form 4 XML, extracting all key fields."""
    try:
        r = requests.get(xml_url, headers=headers, timeout=10)
        if r.status_code != 200:
            return None
        root = ET.fromstring(r.content)
    except Exception:
        return None

    # Ticker
    ticker = (root.findtext('.//issuerTradingSymbol') or '').strip().upper()

    # Role / relationship
    title = (root.findtext('.//officerTitle') or '').strip()
    is_officer = root.findtext('.//isOfficer', '0')
    is_director = root.findtext('.//isDirector', '0')
    is_10pct = root.findtext('.//isTenPercentOwner', '0')
    roles = []
    if is_officer in ('1', 'true'):   roles.append('Officer')
    if is_director in ('1', 'true'):  roles.append('Director')
    if is_10pct in ('1', 'true'):     roles.append('10% Owner')

    # 10b5-1 plan
    plan = root.findtext('.//aff10b5One', '')
    is_10b5_1 = plan in ('1', 'true')

    # Parse ALL transactions (non-derivative)
    txn_dates = []
    total_buy_value = 0
    total_sell_value = 0
    total_buy_shares = 0
    txn_codes = set()

    for txn in root.findall('.//nonDerivativeTransaction'):
        code = txn.findtext('.//transactionCode', '')
        txn_codes.add(code)
        shares_str = txn.findtext('.//transactionShares/value', '0')
        price_str = txn.findtext('.//transactionPricePerShare/value', '0')
        acq_disp = txn.findtext('.//transactionAcquiredDisposedCode/value', '')

        try:
            shares = float(shares_str)
            price = float(price_str) if price_str else 0
        except ValueError:
            shares, price = 0, 0

        value = shares * price

        # P = open market purchase, S = sale, M = exercise, A/G = grant
        if code == 'P':
            total_buy_value += value
            total_buy_shares += shares
        elif code == 'S':
            total_sell_value += value

        td = txn.findtext('.//transactionDate/value', '')
        if td:
            txn_dates.append(td)

    # Determine direction
    if total_buy_value > 0 and total_sell_value == 0:
        direction = 'buy'
    elif total_sell_value > 0 and total_buy_value == 0:
        direction = 'sell'
    elif total_buy_value > 0 and total_sell_value > 0:
        direction = 'mixed'
    else:
        # No P or S codes — exercises, grants, gifts, etc.
        if 'M' in txn_codes:
            direction = 'exercise'
        elif 'A' in txn_codes or 'G' in txn_codes:
            direction = 'grant'
        else:
            direction = 'other'

    # Disclosure delay = signature date - earliest transaction date
    sig_date = root.findtext('.//signatureDate', '')
    earliest_txn = min(txn_dates) if txn_dates else ''
    disclosure_delay = None
    if sig_date and earliest_txn:
        try:
            sig_dt = datetime.strptime(sig_date, '%Y-%m-%d')
            txn_dt = datetime.strptime(earliest_txn, '%Y-%m-%d')
            disclosure_delay = (sig_dt - txn_dt).days
        except ValueError:
            pass

    return {
        'ticker': ticker,
        'insider_role': _normalize_role(title, roles),
        'title_raw': title,
        'roles': roles,
        'is_10b5_1': is_10b5_1,
        'direction': direction,
        'txn_codes': txn_codes,
        'buy_value': round(total_buy_value, 2),
        'buy_shares': int(total_buy_shares),
        'disclosure_delay': disclosure_delay,
        'trade_size_points': _buy_value_to_points(total_buy_value) if total_buy_value > 0 else None,
    }


def backfill_edgar_signals(conn: sqlite3.Connection, dry_run: bool = False,
                            limit: int = None):
    """Main backfill: fetch Form 4 XMLs, enrich fields, remove non-purchases."""

    # Get unprocessed edgar signals (skip already-enriched purchases on resume)
    query = """
        SELECT id, ticker, signal_date, insider_name, insider_role,
               trade_size_points, disclosure_delay, transaction_type
        FROM signals
        WHERE source = 'edgar'
          AND (transaction_type IS NULL OR transaction_type = '')
        ORDER BY signal_date DESC
    """
    signals = conn.execute(query).fetchall()
    total = len(signals)

    if limit:
        signals = signals[:limit]

    log.info(f"Total EDGAR signals: {total}, processing {len(signals)}")

    headers = {
        'User-Agent': SEC_USER_AGENT,
        'Accept': 'application/json',
    }

    # Group signals by (insider_name, signal_date) to batch EFTS lookups
    groups = defaultdict(list)
    for s in signals:
        key = (s['insider_name'], s['signal_date'])
        groups[key].append(s)

    log.info(f"Grouped into {len(groups)} unique (insider, date) pairs")

    stats = {
        'enriched': 0,
        'confirmed_buy': 0,
        'not_buy': 0,
        'not_found': 0,
        'xml_errors': 0,
        'skipped_no_name': 0,
    }
    direction_counts = defaultdict(int)

    for i, ((insider_name, signal_date), group_signals) in enumerate(groups.items()):
        if not insider_name:
            stats['skipped_no_name'] += len(group_signals)
            continue

        # Step 1: Find this filing on EFTS
        efts_result = fetch_efts_for_insider(insider_name, signal_date, headers)
        time.sleep(EFTS_DELAY)

        if not efts_result:
            stats['not_found'] += len(group_signals)
            continue

        # Step 2: Fetch and parse the Form 4 XML
        xml_data = parse_form4_xml(efts_result['xml_url'], headers)
        time.sleep(SEC_DELAY)

        if not xml_data:
            stats['xml_errors'] += len(group_signals)
            continue

        direction = xml_data['direction']
        direction_counts[direction] += len(group_signals)

        # Step 3: Process each signal in this group
        for sig in group_signals:
            is_buy = direction == 'buy'

            if dry_run:
                action = "KEEP+ENRICH" if is_buy else f"DELETE ({direction})"
                log.info(
                    f"  {action}: signal {sig['id']} ({sig['ticker']}) — "
                    f"role={xml_data['insider_role']}, "
                    f"buy_value=${xml_data['buy_value']:,.0f}, "
                    f"delay={xml_data['disclosure_delay']}d, "
                    f"codes={xml_data['txn_codes']}"
                )
                if is_buy:
                    stats['confirmed_buy'] += 1
                else:
                    stats['not_buy'] += 1
                stats['enriched'] += 1
                continue

            if not is_buy:
                # Remove non-purchase signals — they're noise
                conn.execute("DELETE FROM signals WHERE id = ?", (sig['id'],))
                stats['not_buy'] += 1
                continue

            # It's a genuine buy — enrich with all available data
            stats['confirmed_buy'] += 1
            update_parts = ['transaction_type = ?']
            update_vals = ['Purchase']

            if xml_data['insider_role']:
                update_parts.append('insider_role = ?')
                update_vals.append(xml_data['insider_role'])

            if xml_data['trade_size_points'] is not None:
                update_parts.append('trade_size_points = ?')
                update_vals.append(xml_data['trade_size_points'])

            if xml_data['disclosure_delay'] is not None:
                update_parts.append('disclosure_delay = ?')
                update_vals.append(xml_data['disclosure_delay'])

            update_vals.append(sig['id'])
            conn.execute(
                f"UPDATE signals SET {', '.join(update_parts)} WHERE id = ?",
                update_vals
            )
            stats['enriched'] += 1

        # Progress every 100 groups
        if (i + 1) % 100 == 0:
            conn.commit()
            pct = (i + 1) / len(groups) * 100
            log.info(
                f"  [{pct:.0f}%] {i+1}/{len(groups)} pairs — "
                f"buys={stats['confirmed_buy']}, "
                f"removed={stats['not_buy']}, "
                f"not_found={stats['not_found']}, "
                f"errors={stats['xml_errors']}"
            )

    conn.commit()

    # Summary
    log.info(f"\n{'='*60}")
    log.info(f"BACKFILL COMPLETE")
    log.info(f"{'='*60}")
    log.info(f"  Processed: {stats['enriched']}")
    log.info(f"  Confirmed buys (kept+enriched): {stats['confirmed_buy']}")
    log.info(f"  Non-buys {'(would remove)' if dry_run else '(REMOVED)'}: {stats['not_buy']}")
    log.info(f"  Not found on EFTS: {stats['not_found']}")
    log.info(f"  XML errors: {stats['xml_errors']}")
    log.info(f"  Skipped (no name): {stats['skipped_no_name']}")

    log.info(f"\n  Direction breakdown:")
    for direction, count in sorted(direction_counts.items(), key=lambda x: -x[1]):
        log.info(f"    {direction}: {count}")

    if not dry_run:
        # Report final state
        remaining = conn.execute(
            "SELECT COUNT(*) FROM signals WHERE source='edgar'"
        ).fetchone()[0]
        total_signals = conn.execute(
            "SELECT COUNT(*) FROM signals"
        ).fetchone()[0]
        log.info(f"\n  DB after cleanup: {remaining} EDGAR + "
                 f"{total_signals - remaining} congress = {total_signals} total")

        for col in ['insider_role', 'trade_size_points', 'disclosure_delay']:
            filled = conn.execute(
                f"SELECT COUNT(*) FROM signals WHERE source='edgar' "
                f"AND {col} IS NOT NULL AND {col} != ''"
            ).fetchone()[0]
            pct = filled / remaining * 100 if remaining else 0
            log.info(f"  {col}: {filled}/{remaining} ({pct:.1f}%)")


def backfill_relative_position_size(conn: sqlite3.Connection):
    """After trade_size_points is filled, compute relative_position_size.

    relative_position_size = this trade's points / person's median points.
    Values > 1.0 mean unusually large trade for this person.
    """
    insiders = conn.execute("""
        SELECT insider_name, COUNT(*) as cnt
        FROM signals
        WHERE source='edgar' AND insider_name != '' AND trade_size_points IS NOT NULL
        GROUP BY insider_name
        HAVING cnt >= 2
    """).fetchall()

    updated = 0
    for row in insiders:
        insider = row['insider_name']
        sizes = conn.execute(
            "SELECT trade_size_points FROM signals "
            "WHERE source='edgar' AND insider_name=? AND trade_size_points IS NOT NULL "
            "ORDER BY trade_size_points",
            (insider,)
        ).fetchall()

        points = [s['trade_size_points'] for s in sizes]
        n = len(points)
        median = points[n // 2] if n % 2 == 1 else (points[n // 2 - 1] + points[n // 2]) / 2
        if median <= 0:
            continue

        cnt = conn.execute(
            "UPDATE signals SET relative_position_size = trade_size_points / ? "
            "WHERE source='edgar' AND insider_name=? AND trade_size_points IS NOT NULL "
            "AND relative_position_size IS NULL",
            (median, insider)
        ).rowcount
        updated += cnt

    conn.commit()
    log.info(f"Computed relative_position_size for {updated} signals ({len(insiders)} insiders)")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Backfill EDGAR signal data from Form 4 XML')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be updated/deleted without changing DB')
    parser.add_argument('--limit', type=int, default=None,
                        help='Max signals to process (for testing)')
    parser.add_argument('--skip-xml', action='store_true',
                        help='Skip XML fetch, only compute relative_position_size')
    args = parser.parse_args()

    conn = init_db()

    if not args.skip_xml:
        backfill_edgar_signals(conn, dry_run=args.dry_run, limit=args.limit)

    if not args.dry_run:
        backfill_relative_position_size(conn)

    conn.close()
    log.info("Done.")
