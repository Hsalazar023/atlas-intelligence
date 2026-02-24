#!/usr/bin/env python3
"""
ATLAS Data Fetcher
==================
Fetches live data from EDGAR EFTS and QuiverQuant, saves to data/ directory
as static JSON files served by Vercel. No CORS issues — runs server-side.

Usage:
  python3 scripts/fetch_data.py

Schedule:
  Cron (local):  0 */4 * * * cd /path/to/atlas && python3 scripts/fetch_data.py
  GitHub Actions: see .github/workflows/fetch-data.yml (runs every 4h, auto-commits)

Environment variables:
  QUIVER_KEY   — QuiverQuant API token (register at quiverquant.com/quiverapi)
  FRED_KEY     — FRED API key (register at fred.stlouisfed.org/docs/api/api_key.html)
  FINNHUB_KEY  — Finnhub API key (for market_data.json index prices)
"""

import os, json, re, time, datetime, sys
try:
    import requests
except ImportError:
    sys.exit("Missing 'requests'. Run: pip3 install requests")

# ── Config ──────────────────────────────────────────────────────────────────
USER_AGENT   = 'ATLAS Intelligence Platform contact@atlasiq.io'
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')
QUIVER_KEY   = os.environ.get('QUIVER_KEY',  '')
FRED_KEY     = os.environ.get('FRED_KEY',    '71a0b94ed47b56a81f405947f88d08aa')
FINNHUB_KEY  = os.environ.get('FINNHUB_KEY', 'd6dnud9r01qm89pkai30d6dnud9r01qm89pkai3g')

# SEC rate limit: max 10 req/sec, recommend 1 req/sec for pollers
SEC_DELAY   = 0.5  # seconds between SEC requests

# ── Helpers ──────────────────────────────────────────────────────────────────
def clean_name(n):
    """Strip (CIK 0001234567) suffix from display_name."""
    return re.sub(r'\s*\(CIK \d+\)\s*', '', n).strip()

def is_company(name):
    """Heuristic: does this display_name look like a company, not a person?"""
    upper = name.upper()
    company_words = ['INC', ' CORP', ' LLC', ' LTD', ' CO ', ' CO,', ' GROUP',
                     ' HOLDINGS', ' FUND', ' TRUST', ' INTERNATIONAL', ' PARTNERS',
                     ' MANAGEMENT', ' CAPITAL', ' TECHNOLOGIES', ' SYSTEMS', ' PLC']
    return any(w in upper for w in company_words)

def save_json(filename, data):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f'  → Saved {path}')

# ── EDGAR Form 4 Feed ─────────────────────────────────────────────────────────
def fetch_edgar_form4(days=7, max_results=200):
    """
    Fetch recent Form 4 filings from EDGAR EFTS full-text search.
    Returns list of filing dicts. Covers ALL companies — not limited to tracked tickers.
    """
    startdt = (datetime.datetime.utcnow() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
    base_url = (
        'https://efts.sec.gov/LATEST/search-index'
        '?forms=4'
        '&dateRange=custom'
        f'&startdt={startdt}'
    )
    headers = {
        'User-Agent': USER_AGENT,
        'Accept': 'application/json',
    }

    filings = []
    from_idx = 0
    page_size = 100

    while len(filings) < max_results:
        url = base_url + f'&from={from_idx}&hits.hits.total.value=true'
        try:
            r = requests.get(url, headers=headers, timeout=20)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f'  EDGAR fetch error at from={from_idx}: {e}')
            break

        hits = data.get('hits', {}).get('hits', [])
        if not hits:
            break

        for h in hits:
            src = h.get('_source', {})
            names = src.get('display_names', [])

            # Separate person (insider) from company (issuer)
            company_names = [n for n in names if is_company(n)]
            person_names  = [n for n in names if not is_company(n)]

            company = clean_name(company_names[-1]) if company_names else clean_name(names[-1]) if names else 'Unknown'
            insider = clean_name(person_names[0])   if person_names  else clean_name(names[0])  if names else 'Unknown'

            # Build direct SEC filing link from accession number + company CIK
            accession = src.get('adsh', '')
            ciks = src.get('ciks', [])
            # Company CIK is typically the last one; insider CIK is first
            company_cik = ciks[-1] if len(ciks) > 1 else (ciks[0] if ciks else '')
            link = ''
            if company_cik and accession:
                try:
                    numeric_cik = str(int(company_cik))
                    clean_acc = accession.replace('-', '')
                    link = (
                        f'https://www.sec.gov/Archives/edgar/data/'
                        f'{numeric_cik}/{clean_acc}/{accession}-index.htm'
                    )
                except ValueError:
                    pass

            filings.append({
                'company':   company,
                'insider':   insider,
                'date':      src.get('file_date', ''),
                'period':    src.get('period_ending', ''),
                'accession': accession,
                'link':      link,
            })

        from_idx += page_size
        total_available = data.get('hits', {}).get('total', {}).get('value', 0)
        if from_idx >= min(total_available, max_results):
            break

        time.sleep(SEC_DELAY)

    # Sort newest first
    filings.sort(key=lambda x: x.get('date', ''), reverse=True)
    return filings


# ── QuiverQuant Congressional Trades ─────────────────────────────────────────
def fetch_quiver_congress(api_key):
    """
    Fetch recent congressional trades from QuiverQuant live endpoint.
    Requires a free API token from quiverquant.com/quiverapi.
    """
    url = 'https://api.quiverquant.com/beta/live/congresstrading'
    headers = {
        'Authorization': f'Token {api_key}',
        'Accept': 'application/json',
    }
    r = requests.get(url, headers=headers, timeout=20)
    if not r.ok:
        print(f'  QuiverQuant error: HTTP {r.status_code}')
        return []
    data = r.json()
    if not isinstance(data, list):
        print(f'  QuiverQuant unexpected response: {str(data)[:200]}')
        return []
    # Sort most recent first
    return sorted(data, key=lambda x: x.get('Date', ''), reverse=True)[:150]


# ── QuiverQuant Insider Trades ─────────────────────────────────────────────
def fetch_quiver_insiders(api_key, page_size=100):
    """
    Fetch recent insider transactions from QuiverQuant.
    Tier 2 endpoint — included with free token if access is granted.
    Fields: ticker, date, owner name, transaction type, shares, value, role
    """
    url = f'https://api.quiverquant.com/beta/live/insiders?page_size={page_size}'
    headers = {
        'Authorization': f'Token {api_key}',
        'Accept': 'application/json',
    }
    r = requests.get(url, headers=headers, timeout=20)
    if not r.ok:
        print(f'  QuiverQuant insiders error: HTTP {r.status_code}')
        return []
    data = r.json()
    if not isinstance(data, list):
        return []
    return sorted(data, key=lambda x: x.get('Date', x.get('date', '')), reverse=True)


# ── Market Context Data (VIX + 10yr Treasury) ────────────────────────────────
def fetch_market_data():
    """
    Fetch VIX and 10-yr Treasury yield from FRED.
    Saves to data/market_data.json — frontend reads this file to avoid CORS.
    """
    if not FRED_KEY:
        print('  FRED_KEY not set — skipping market data')
        return

    base = (
        'https://api.stlouisfed.org/fred/series/observations'
        f'?api_key={FRED_KEY}&limit=5&sort_order=desc&file_type=json&series_id='
    )

    market = {'updated': datetime.datetime.utcnow().isoformat() + 'Z', 'source': 'FRED'}

    for series_id, key, label in [('DGS10', 'treasury_10yr', '10yr Treasury'),
                                   ('VIXCLS', 'vix', 'VIX')]:
        try:
            r = requests.get(base + series_id, timeout=15)
            r.raise_for_status()
            data = r.json()
            obs = [o for o in data.get('observations', []) if o.get('value') not in ('.', '')]
            if obs:
                val  = float(obs[0]['value'])
                prev = float(obs[1]['value']) if len(obs) > 1 else None
                market[key] = {
                    'value': val,
                    'date':  obs[0]['date'],
                    'change': round(val - prev, 4) if prev is not None else None,
                }
                print(f'  {label}: {val}  ({obs[0]["date"]})')
            else:
                print(f'  {label}: no valid observations returned')
        except Exception as e:
            print(f'  FRED {series_id} error: {e}')

    save_json('market_data.json', market)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f'\n=== ATLAS Data Fetcher — {datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M")} UTC ===\n')

    # ── Market context (VIX + 10yr yield from FRED) ────────────────────────
    print('Fetching market context data (FRED)...')
    fetch_market_data()

    # ── EDGAR Form 4 feed ──────────────────────────────────────────────────
    print('Fetching EDGAR Form 4 filings (last 7 days)...')
    try:
        filings = fetch_edgar_form4(days=7, max_results=200)
        save_json('edgar_feed.json', {
            'updated': datetime.datetime.utcnow().isoformat() + 'Z',
            'count': len(filings),
            'source': 'SEC EDGAR EFTS',
            'filings': filings,
        })
        print(f'  {len(filings)} Form 4 filings saved')
    except Exception as e:
        print(f'  EDGAR error: {e}')

    # ── Congressional trades (QuiverQuant) ─────────────────────────────────
    if QUIVER_KEY:
        print('\nFetching congressional trades (QuiverQuant)...')
        try:
            trades = fetch_quiver_congress(QUIVER_KEY)
            save_json('congress_feed.json', {
                'updated': datetime.datetime.utcnow().isoformat() + 'Z',
                'count': len(trades),
                'source': 'QuiverQuant',
                'trades': trades,
            })
            print(f'  {len(trades)} congressional trades saved')
        except Exception as e:
            print(f'  QuiverQuant congress error: {e}')

        # ── Insider trades (QuiverQuant Tier 2) ───────────────────────────
        print('\nFetching insider trades (QuiverQuant)...')
        try:
            insiders = fetch_quiver_insiders(QUIVER_KEY)
            if insiders:
                save_json('insiders_feed.json', {
                    'updated': datetime.datetime.utcnow().isoformat() + 'Z',
                    'count': len(insiders),
                    'source': 'QuiverQuant',
                    'trades': insiders,
                })
                print(f'  {len(insiders)} insider trades saved')
            else:
                print('  No insider data returned (Tier 2 access may not be included)')
        except Exception as e:
            print(f'  QuiverQuant insiders error: {e}')
    else:
        print('\nQUIVER_KEY not set — skipping congressional and insider trade fetch')
        print('  Register at quiverquant.com/quiverapi, then:')
        print('  export QUIVER_KEY=your_token && python3 scripts/fetch_data.py')

    print('\nDone.\n')


if __name__ == '__main__':
    main()
