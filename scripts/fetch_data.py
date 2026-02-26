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
from xml.etree import ElementTree as ET
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
            _id = h.get('_id', '')

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
            xml_url = ''
            if company_cik and accession:
                try:
                    numeric_cik = str(int(company_cik))
                    clean_acc = accession.replace('-', '')
                    link = (
                        f'https://www.sec.gov/Archives/edgar/data/'
                        f'{numeric_cik}/{clean_acc}/{accession}-index.htm'
                    )
                    # Extract XML filename from EFTS _id: "{accession}:{filename}"
                    xml_filename = _id.split(':', 1)[1] if ':' in _id else ''
                    if xml_filename:
                        xml_url = (
                            f'https://www.sec.gov/Archives/edgar/data/'
                            f'{numeric_cik}/{clean_acc}/{xml_filename}'
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
                'xml_url':   xml_url,
            })

        from_idx += page_size
        total_available = data.get('hits', {}).get('total', {}).get('value', 0)
        if from_idx >= min(total_available, max_results):
            break

        time.sleep(SEC_DELAY)

    # Sort newest first
    filings.sort(key=lambda x: x.get('date', ''), reverse=True)
    return filings


def enrich_form4_xml(filings):
    """
    Fetch each Form 4 XML and extract: ticker, role, transaction type,
    shares, price, total value, 10b5-1 plan flag.
    SEC rate limit: 10 req/sec — we use 0.12s delay (~8 req/sec).
    """
    headers = {'User-Agent': USER_AGENT}
    enriched = 0
    errors = 0
    total = len(filings)

    for i, f in enumerate(filings):
        xml_url = f.get('xml_url', '')
        if not xml_url:
            continue

        try:
            r = requests.get(xml_url, headers=headers, timeout=10)
            if r.status_code != 200:
                errors += 1
                continue

            root = ET.fromstring(r.content)

            # Ticker
            f['ticker'] = (root.findtext('.//issuerTradingSymbol') or '').strip().upper()

            # Role / relationship
            f['title'] = (root.findtext('.//officerTitle') or '').strip()
            is_officer  = root.findtext('.//isOfficer', '0')
            is_director = root.findtext('.//isDirector', '0')
            is_10pct    = root.findtext('.//isTenPercentOwner', '0')
            roles = []
            if is_officer in ('1', 'true'):   roles.append('Officer')
            if is_director in ('1', 'true'):  roles.append('Director')
            if is_10pct in ('1', 'true'):     roles.append('10% Owner')
            f['roles'] = roles

            # 10b5-1 plan
            plan = root.findtext('.//aff10b5One', '')
            f['is_10b5_1'] = plan in ('1', 'true')

            # Transactions (non-derivative)
            txns = []
            total_buy_value = 0
            total_sell_value = 0
            total_buy_shares = 0
            total_sell_shares = 0
            for txn in root.findall('.//nonDerivativeTransaction'):
                code = txn.findtext('.//transactionCode', '')
                shares_str = txn.findtext('.//transactionShares/value', '0')
                price_str = txn.findtext('.//transactionPricePerShare/value', '0')
                acq_disp = txn.findtext('.//transactionAcquiredDisposedCode/value', '')
                try:
                    shares = float(shares_str)
                    price = float(price_str) if price_str else 0
                except ValueError:
                    shares, price = 0, 0
                value = shares * price
                txns.append({
                    'code': code,  # P=purchase, S=sale, M=exercise, A=grant
                    'shares': shares,
                    'price': round(price, 2),
                    'value': round(value, 2),
                    'acquired': acq_disp == 'A',
                })
                if code == 'P':
                    total_buy_value += value
                    total_buy_shares += shares
                elif code == 'S':
                    total_sell_value += value
                    total_sell_shares += shares

            f['transactions'] = txns
            f['buy_value'] = round(total_buy_value, 2)
            f['sell_value'] = round(total_sell_value, 2)
            f['buy_shares'] = int(total_buy_shares)
            f['sell_shares'] = int(total_sell_shares)
            # Net direction: P = purchase, S = sale, M = mixed/exercise
            if total_buy_value > 0 and total_sell_value == 0:
                f['direction'] = 'buy'
            elif total_sell_value > 0 and total_buy_value == 0:
                f['direction'] = 'sell'
            elif total_buy_value > 0 and total_sell_value > 0:
                f['direction'] = 'mixed'
            else:
                # Check for M (exercise) or A (grant) codes
                codes = set(t['code'] for t in txns)
                f['direction'] = 'exercise' if 'M' in codes else 'grant' if 'A' in codes else 'other'

            enriched += 1

        except ET.ParseError:
            errors += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f'  XML parse error [{i+1}/{total}]: {e}')

        # Rate limit: ~8 req/sec
        time.sleep(0.12)

        # Progress indicator every 50
        if (i + 1) % 50 == 0:
            print(f'  Enriched {enriched}/{i+1} filings ({errors} errors)...')

    print(f'  Enrichment complete: {enriched}/{total} enriched, {errors} errors')
    # Remove xml_url from output (internal only)
    for f in filings:
        f.pop('xml_url', None)
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
    print('Fetching EDGAR Form 4 filings (last 90 days)...')
    try:
        filings = fetch_edgar_form4(days=90, max_results=1000)
        print(f'  {len(filings)} filings from EFTS index')

        # Enrich with Form 4 XML data (ticker, role, amounts, buy/sell)
        print('  Enriching filings from Form 4 XML (~25s per 200 filings)...')
        filings = enrich_form4_xml(filings)

        save_json('edgar_feed.json', {
            'updated': datetime.datetime.utcnow().isoformat() + 'Z',
            'count': len(filings),
            'source': 'SEC EDGAR EFTS + Form 4 XML',
            'filings': filings,
        })
        print(f'  {len(filings)} enriched Form 4 filings saved')
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

    # ── SEC ticker map (for EDGAR→ticker matching) ────────────────────
    print('\nFetching SEC company→ticker map...')
    try:
        sec_url = 'https://www.sec.gov/files/company_tickers.json'
        r = requests.get(sec_url, headers={'User-Agent': USER_AGENT}, timeout=15)
        r.raise_for_status()
        raw = r.json()
        # Build name→ticker (prefer shortest ticker = common stock)
        cik_best = {}
        for entry in raw.values():
            cik = entry['cik_str']
            ticker = entry['ticker']
            title = entry['title'].lower().strip()
            if cik not in cik_best or len(ticker) < len(cik_best[cik][1]):
                cik_best[cik] = (title, ticker)
        sec_map = {name: ticker for name, ticker in cik_best.values()}
        save_json('sec_tickers.json', sec_map)
        print(f'  {len(sec_map)} company→ticker mappings saved')
    except Exception as e:
        print(f'  SEC ticker map error: {e}')

    print('\nDone.\n')


if __name__ == '__main__':
    main()
