#!/usr/bin/env python3
"""
ATLAS 13F Fetcher — SEC EDGAR EFTS
===================================
Queries SEC EDGAR for recent 13F-HR filings, parses XML information tables,
compares to prior quarter, and surfaces new positions and >25% increases.

Output: data/13f_feed.json

Usage:
  python3 scripts/fetch_13f.py

No API key required — SEC EDGAR is free. Rate limit: 10 req/sec, we use 0.5s delay.
"""

import os
import sys
import json
import time
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

try:
    import requests
except ImportError:
    sys.exit("Missing 'requests'. Run: pip3 install requests")

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
FEED_PATH = DATA_DIR / '13f_feed.json'
CACHE_PATH = DATA_DIR / '13f_cache.json'  # prior quarter for diffing

USER_AGENT = 'ATLAS Intelligence Platform contact@atlasiq.io'
HEADERS = {'User-Agent': USER_AGENT, 'Accept-Encoding': 'gzip, deflate'}
SEC_DELAY = 0.5  # seconds between SEC requests

# Notable filers we track (CIK numbers)
# Add more as needed — these are well-known institutional investors
NOTABLE_FILERS = {
    '1067983': 'Berkshire Hathaway',
    '1649339': 'Duquesne Family Office',
    '1541996': 'Pershing Square',
    '1037389': 'Renaissance Technologies',
    '1061768': 'Third Point',
    '1336528': 'Citadel Advisors',
    '1350694': 'Appaloosa Management',
    '1273087': 'Tiger Global',
    '1656456': 'Coatue Management',
    '0896159': 'DE Shaw',
    '1167483': 'Viking Global',
    '1040273': 'Lone Pine Capital',
    '1730073': 'Whale Rock Capital',
    '1037389': 'Renaissance Technologies',
    '1364742': 'Elliott Management',
    '0001159159': 'RTW Investments',
    '0001044316': 'Soros Fund Management',
}


def fetch_recent_13f_filings(days_back: int = 95) -> list:
    """Fetch recent 13F-HR filings from SEC EDGAR EFTS.
    Uses 95 days to capture the full quarterly filing window (45 days after quarter end).
    """
    end_date = datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')
    start_date = (datetime.now(tz=timezone.utc) - timedelta(days=days_back)).strftime('%Y-%m-%d')

    url = (
        'https://efts.sec.gov/LATEST/search-index'
        '?forms=13F-HR'
        '&dateRange=custom'
        f'&startdt={start_date}'
        f'&enddt={end_date}'
        '&from=0&size=100'
    )

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get('hits', {}).get('hits', [])
        log.info(f"Found {len(hits)} 13F-HR filings in last {days_back} days")
        return hits
    except requests.RequestException as e:
        log.warning(f"EFTS search failed: {e}")
        return []


def parse_13f_filing(filing: dict) -> dict | None:
    """Extract key info from a 13F EFTS hit."""
    source = filing.get('_source', {})

    cik = str(source.get('entity_id', '')).lstrip('0')
    filer_name = source.get('display_names', [''])[0] if source.get('display_names') else ''
    # Clean CIK suffix
    filer_name = re.sub(r'\s*\(CIK \d+\)\s*', '', filer_name).strip()

    filed_date = source.get('file_date', '')
    period = source.get('period_of_report', '')
    accession = source.get('file_num', '')

    # Check if this is a notable filer
    is_notable = cik in NOTABLE_FILERS
    if is_notable:
        filer_name = NOTABLE_FILERS[cik]

    return {
        'cik': cik,
        'filer_name': filer_name,
        'filed_date': filed_date,
        'period': period,
        'accession': accession,
        'is_notable': is_notable,
        'holdings': [],
    }


def fetch_13f_holdings(cik: str, accession: str) -> list:
    """Fetch and parse the 13F information table XML for a filing.
    Returns list of holdings with ticker, shares, value."""
    # Normalize accession number for URL
    acc_clean = accession.replace('-', '')

    # Try to find the infotable XML
    index_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=13F-HR&dateb=&owner=include&count=10&search_text=&action=getcompany"

    # Direct approach: use EDGAR Archives
    acc_parts = accession.split('-')
    if len(acc_parts) >= 3:
        acc_path = f"{acc_parts[0]}/{acc_parts[1]}/{acc_parts[2]}"
    else:
        acc_path = acc_clean

    # Try to get filing index to find infotable
    idx_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/index.json"

    time.sleep(SEC_DELAY)
    try:
        resp = requests.get(idx_url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return []

        idx_data = resp.json()
        items = idx_data.get('directory', {}).get('item', [])

        # Find the infotable XML file
        infotable_file = None
        for item in items:
            name = item.get('name', '').lower()
            if 'infotable' in name and name.endswith('.xml'):
                infotable_file = item['name']
                break
            if name.endswith('.xml') and '13f' in name:
                infotable_file = item['name']
                break

        if not infotable_file:
            return []

        # Fetch the infotable XML
        xml_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{infotable_file}"
        time.sleep(SEC_DELAY)
        resp = requests.get(xml_url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return []

        return parse_infotable_xml(resp.text)

    except Exception as e:
        log.warning(f"Could not fetch holdings for CIK {cik}: {e}")
        return []


def parse_infotable_xml(xml_text: str) -> list:
    """Parse 13F information table XML into list of holdings."""
    holdings = []
    try:
        # Handle various XML namespaces
        xml_text_clean = re.sub(r'\sxmlns[^"]*"[^"]*"', '', xml_text, count=1)
        root = ET.fromstring(xml_text_clean)

        for info in root.iter():
            if 'infoTable' in info.tag:
                holding = {}

                for child in info:
                    tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                    tag_lower = tag.lower()

                    if tag_lower == 'nameofissuer':
                        holding['issuer'] = (child.text or '').strip()
                    elif tag_lower == 'titleofclass':
                        holding['title'] = (child.text or '').strip()
                    elif tag_lower == 'cusip':
                        holding['cusip'] = (child.text or '').strip()
                    elif tag_lower == 'value':
                        try:
                            holding['value'] = int(child.text or 0) * 1000  # value in thousands
                        except (ValueError, TypeError):
                            holding['value'] = 0
                    elif 'sshprnamt' in tag_lower:
                        for sub in child:
                            sub_tag = sub.tag.split('}')[-1] if '}' in sub.tag else sub.tag
                            if 'sshprnamt' in sub_tag.lower() and sub_tag.lower() != 'sshprnamttype':
                                try:
                                    holding['shares'] = int(sub.text or 0)
                                except (ValueError, TypeError):
                                    holding['shares'] = 0
                    elif tag_lower == 'putcall':
                        holding['put_call'] = (child.text or '').strip()

                if holding.get('issuer'):
                    holdings.append(holding)

    except ET.ParseError as e:
        log.warning(f"XML parse error: {e}")

    return holdings


def compute_position_changes(current: list, prior: list) -> list:
    """Compare current vs prior quarter holdings to find new/increased positions.
    Returns list of notable changes."""
    # Build prior lookup by CUSIP
    prior_map = {}
    for h in prior:
        cusip = h.get('cusip', '')
        if cusip:
            prior_map[cusip] = h

    changes = []
    for h in current:
        cusip = h.get('cusip', '')
        value = h.get('value', 0)
        shares = h.get('shares', 0)
        issuer = h.get('issuer', '')

        if cusip in prior_map:
            prior_h = prior_map[cusip]
            prior_shares = prior_h.get('shares', 0)
            prior_value = prior_h.get('value', 0)

            if prior_shares > 0:
                pct_change = (shares - prior_shares) / prior_shares
                if pct_change >= 0.25:  # >25% increase
                    changes.append({
                        'issuer': issuer,
                        'cusip': cusip,
                        'action': 'increased',
                        'shares': shares,
                        'value': value,
                        'prior_shares': prior_shares,
                        'prior_value': prior_value,
                        'pct_change': round(pct_change, 4),
                    })
        else:
            # New position
            if value >= 100_000_000:  # Only surface new positions >= $100M
                changes.append({
                    'issuer': issuer,
                    'cusip': cusip,
                    'action': 'new_position',
                    'shares': shares,
                    'value': value,
                    'prior_shares': 0,
                    'prior_value': 0,
                    'pct_change': None,
                })

    return changes


# CUSIP → Ticker mapping for common securities
# Expanded at runtime from SEC ticker map if available
CUSIP_TICKER_MAP = {
    '037833100': 'AAPL',
    '594918104': 'MSFT',
    '67066G104': 'NVDA',
    '30303M102': 'META',
    '02079K305': 'GOOGL',
    '023135106': 'AMZN',
    '88160R101': 'TSLA',
    '22160K105': 'COST',
    '931142103': 'WMT',
    '46625H100': 'JPM',
    '060505104': 'BAC',
    '882508104': 'OXY',
    '30231G102': 'XOM',
    '166764100': 'CVX',
    '35671D857': 'FCX',
    '718546104': 'PFE',
    '58933Y105': 'MRK',
    '00287Y109': 'ABBV',
    '75513E101': 'RTX',
    '539830109': 'LMT',
    '666807102': 'NOC',
    '097023105': 'BA',
}


def cusip_to_ticker(cusip: str) -> str:
    """Map CUSIP to ticker symbol. Returns empty string if unknown."""
    return CUSIP_TICKER_MAP.get(cusip, '')


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Fetch recent 13F filings
    filings = fetch_recent_13f_filings(days_back=95)

    # 2. Parse and filter for notable filers
    parsed = []
    for f in filings:
        info = parse_13f_filing(f)
        if info and info['is_notable']:
            parsed.append(info)

    log.info(f"Found {len(parsed)} notable filer 13F filings")

    # 3. Load prior quarter cache for diffing
    prior_cache = {}
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH) as f:
                prior_cache = json.load(f)
        except Exception:
            prior_cache = {}

    # 4. Fetch holdings and compute changes for notable filers
    all_changes = []
    current_cache = {}

    for filer in parsed[:20]:  # cap at 20 filers to respect rate limits
        cik = filer['cik']
        accession = filer['accession']
        filer_name = filer['filer_name']

        if not accession:
            continue

        log.info(f"Fetching holdings: {filer_name} (CIK {cik})")
        holdings = fetch_13f_holdings(cik, accession)

        if holdings:
            current_cache[cik] = holdings
            prior = prior_cache.get(cik, [])
            changes = compute_position_changes(holdings, prior)

            for change in changes:
                ticker = cusip_to_ticker(change['cusip'])
                all_changes.append({
                    'filer': filer_name,
                    'cik': cik,
                    'filed_date': filer['filed_date'],
                    'period': filer['period'],
                    'ticker': ticker,
                    'issuer': change['issuer'],
                    'action': change['action'],
                    'shares': change['shares'],
                    'value': change['value'],
                    'prior_shares': change['prior_shares'],
                    'prior_value': change['prior_value'],
                    'pct_change': change['pct_change'],
                })

    # 5. Save current holdings as cache for next quarter's diff
    if current_cache:
        with open(CACHE_PATH, 'w') as f:
            json.dump(current_cache, f)
        log.info(f"Cached {len(current_cache)} filer holdings for next quarter diff")

    # 6. Sort by value descending and save feed
    all_changes.sort(key=lambda x: x.get('value', 0), reverse=True)

    output = {
        'generated': datetime.now(tz=timezone.utc).isoformat(),
        'source': 'SEC EDGAR 13F-HR',
        'count': len(all_changes),
        'filings': all_changes,
    }

    with open(FEED_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    log.info(f"Saved {len(all_changes)} 13F changes to {FEED_PATH}")

    # Summary
    new_positions = [c for c in all_changes if c['action'] == 'new_position']
    increases = [c for c in all_changes if c['action'] == 'increased']
    log.info(f"Summary: {len(new_positions)} new positions, {len(increases)} increased (>25%)")


if __name__ == '__main__':
    main()
