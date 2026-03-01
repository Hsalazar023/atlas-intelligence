#!/usr/bin/env python3
"""
ATLAS Bill Fetcher — Congress.gov API v3
========================================
Fetches recent bills by keyword categories mapped to market sectors.
Extracts: bill ID, title, status, latest action, action date, committees.
Maps bills to impact sectors + tickers via keyword matching.

Output: data/bills_feed.json

Usage:
  python3 scripts/fetch_bills.py

Environment variables:
  CONGRESS_API_KEY — Congress.gov API key (free, from api.congress.gov)
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Missing 'requests'. Run: pip3 install requests")

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
BILLS_FEED = DATA_DIR / 'bills_feed.json'

API_BASE = 'https://api.congress.gov/v3'

# Keyword categories → sector + impact tickers
# Each search query maps to a sector and set of tickers that would be impacted
BILL_CATEGORIES = [
    {
        'keywords': ['artificial intelligence', 'AI infrastructure', 'machine learning', 'semiconductor'],
        'sector': 'Technology',
        'tickers': ['NVDA', 'AMD', 'SMCI', 'MSFT', 'GOOGL', 'META', 'INTC', 'AVGO'],
    },
    {
        'keywords': ['defense modernization', 'military spending', 'national defense', 'defense authorization'],
        'sector': 'Industrials',
        'tickers': ['RTX', 'LMT', 'NOC', 'BA', 'GD', 'HII', 'LHX'],
    },
    {
        'keywords': ['clean energy', 'renewable energy', 'solar', 'wind energy', 'electric vehicle'],
        'sector': 'Energy',
        'tickers': ['ENPH', 'NEE', 'FSLR', 'TSLA', 'PLUG'],
    },
    {
        'keywords': ['drug pricing', 'pharmaceutical', 'healthcare reform', 'Medicare'],
        'sector': 'Healthcare',
        'tickers': ['PFE', 'MRK', 'ABBV', 'UNH', 'JNJ', 'LLY'],
    },
    {
        'keywords': ['critical minerals', 'mining', 'rare earth', 'lithium'],
        'sector': 'Basic Materials',
        'tickers': ['FCX', 'MP', 'ALB', 'NEM', 'AA'],
    },
    {
        'keywords': ['cryptocurrency', 'digital assets', 'stablecoin', 'blockchain'],
        'sector': 'Crypto',
        'tickers': ['COIN', 'MSTR', 'SQ', 'MARA', 'RIOT'],
    },
    {
        'keywords': ['cybersecurity', 'data privacy', 'critical infrastructure'],
        'sector': 'Cybersecurity',
        'tickers': ['CRWD', 'PANW', 'ZS', 'FTNT', 'S'],
    },
    {
        'keywords': ['oil and gas', 'energy independence', 'drilling', 'fossil fuel', 'natural gas'],
        'sector': 'Oil & Gas',
        'tickers': ['OXY', 'XOM', 'CVX', 'COP', 'EOG', 'DVN'],
    },
]


def fetch_bills_for_query(query: str, api_key: str, limit: int = 10) -> list:
    """Search Congress.gov for bills matching a query. Returns list of bill dicts."""
    url = f"{API_BASE}/bill"
    params = {
        'api_key': api_key,
        'format': 'json',
        'limit': limit,
        'sort': 'updateDate+desc',
        'query': query,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get('bills', [])
    except requests.RequestException as e:
        log.warning(f"Congress.gov API error for '{query}': {e}")
        return []


def fetch_bill_detail(congress: int, bill_type: str, number: int, api_key: str) -> dict:
    """Fetch detailed info for a specific bill."""
    url = f"{API_BASE}/bill/{congress}/{bill_type}/{number}"
    params = {'api_key': api_key, 'format': 'json'}

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get('bill', {})
    except requests.RequestException as e:
        log.warning(f"Bill detail error for {bill_type}{number}: {e}")
        return {}


def extract_bill_info(bill: dict, category: dict) -> dict:
    """Extract relevant fields from a Congress.gov bill response."""
    bill_type = bill.get('type', '').lower()
    number = bill.get('number', '')
    congress = bill.get('congress', 119)

    # Build bill ID like HR7821 or SB1882
    type_map = {'hr': 'HR', 's': 'SB', 'hres': 'HRES', 'sres': 'SRES',
                'hjres': 'HJRES', 'sjres': 'SJRES', 'hconres': 'HCONRES', 'sconres': 'SCONRES'}
    bill_id = f"{type_map.get(bill_type, bill_type.upper())}{number}"

    # Latest action
    latest_action = bill.get('latestAction', {})
    action_date = latest_action.get('actionDate', '')
    action_text = latest_action.get('text', '')

    # Chamber
    chamber = 'house' if bill_type.startswith('h') else 'senate'

    return {
        'id': bill_id,
        'title': bill.get('title', ''),
        'congress': congress,
        'bill_type': bill_type,
        'number': number,
        'chamber': chamber,
        'status': action_text[:100] if action_text else 'Unknown',
        'action_date': action_date,
        'introduced_date': bill.get('introducedDate', ''),
        'url': bill.get('url', f"https://www.congress.gov/bill/{congress}th-congress/{chamber}-bill/{number}"),
        'sector': category['sector'],
        'impact_tickers': category['tickers'],
        'search_category': category['keywords'][0],
    }


def main():
    api_key = os.environ.get('CONGRESS_API_KEY', '')
    if not api_key:
        log.warning("CONGRESS_API_KEY not set — skipping bill fetch")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_bills = {}  # keyed by bill_id to deduplicate
    for category in BILL_CATEGORIES:
        for keyword in category['keywords']:
            log.info(f"Searching: '{keyword}' ({category['sector']})")
            bills = fetch_bills_for_query(keyword, api_key, limit=5)

            for bill in bills:
                info = extract_bill_info(bill, category)
                bill_id = info['id']

                # Deduplicate: keep the version with more tickers
                if bill_id in all_bills:
                    existing = all_bills[bill_id]
                    # Merge tickers from multiple categories
                    merged_tickers = list(set(existing['impact_tickers'] + info['impact_tickers']))
                    existing['impact_tickers'] = merged_tickers
                else:
                    all_bills[bill_id] = info

            # Rate limit: Congress.gov recommends 1 req/sec
            time.sleep(1.0)

    bills_list = sorted(all_bills.values(), key=lambda b: b.get('action_date', ''), reverse=True)

    output = {
        'generated': datetime.now(tz=timezone.utc).isoformat(),
        'source': 'Congress.gov API v3',
        'count': len(bills_list),
        'bills': bills_list,
    }

    with open(BILLS_FEED, 'w') as f:
        json.dump(output, f, indent=2)

    log.info(f"Saved {len(bills_list)} bills to {BILLS_FEED}")


if __name__ == '__main__':
    main()
