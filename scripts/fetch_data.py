#!/usr/bin/env python3
"""
ATLAS Data Fetcher
==================
Fetches live data from EDGAR EFTS, FMP (congress + insiders), and FRED (market context).
Saves to data/ directory as static JSON files served by Vercel. No CORS issues — runs server-side.

Usage:
  python3 scripts/fetch_data.py

Schedule:
  Cron (local):  0 */4 * * * cd /path/to/atlas && python3 scripts/fetch_data.py
  GitHub Actions: see .github/workflows/fetch-data.yml (runs every 4h, auto-commits)

Environment variables:
  FMP_API_KEY  — Financial Modeling Prep API key (congress + insider trades)
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
FRED_KEY     = os.environ.get('FRED_KEY',    '71a0b94ed47b56a81f405947f88d08aa')
FINNHUB_KEY  = os.environ.get('FINNHUB_KEY', 'd6dnud9r01qm89pkai30d6dnud9r01qm89pkai3g')
QUIVER_KEY   = os.environ.get('QUIVER_KEY', '')

# SEC rate limit: max 10 req/sec, recommend 1 req/sec for pollers
SEC_DELAY   = 0.5  # seconds between SEC requests

# Tickers known to 404/error — suppress from stdout entirely
SUPPRESSED_TICKERS = {'BRK/B', 'NONE', 'BHI', 'AZPN', 'BSCP', 'N/A', ''}

LOG_DIR = os.path.join(DATA_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

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

def _sanitize_for_json(obj):
    """Sanitize values for valid JSON output (numpy types, NaN, Infinity)."""
    import math
    try:
        import numpy as np
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
    except ImportError:
        pass
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def save_json(filename, data):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    tmp_path = path + '.tmp'
    data = _sanitize_for_json(data)
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)  # atomic on POSIX

# ── EDGAR Form 4 Feed ─────────────────────────────────────────────────────────
def _fetch_edgar_window(startdt: str, enddt: str, max_per_window: int, headers: dict) -> list:
    """Fetch Form 4 filings for a single date window from EFTS."""
    base_url = (
        'https://efts.sec.gov/LATEST/search-index'
        '?forms=4'
        '&dateRange=custom'
        f'&startdt={startdt}'
        f'&enddt={enddt}'
    )
    filings = []
    from_idx = 0
    page_size = 100

    while len(filings) < max_per_window:
        url = base_url + f'&from={from_idx}&hits.hits.total.value=true'
        try:
            r = requests.get(url, headers=headers, timeout=20)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f'    EFTS error window {startdt}→{enddt} from={from_idx}: {e}')
            break

        hits = data.get('hits', {}).get('hits', [])
        if not hits:
            break

        for h in hits:
            src = h.get('_source', {})
            names = src.get('display_names', [])
            _id = h.get('_id', '')

            company_names = [n for n in names if is_company(n)]
            person_names  = [n for n in names if not is_company(n)]

            company = clean_name(company_names[-1]) if company_names else clean_name(names[-1]) if names else 'Unknown'
            insider = clean_name(person_names[0])   if person_names  else clean_name(names[0])  if names else 'Unknown'

            accession = src.get('adsh', '')
            ciks = src.get('ciks', [])
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
        if from_idx >= min(total_available, max_per_window):
            break

        time.sleep(SEC_DELAY)

    return filings


def fetch_edgar_form4(days=7, max_results=200, window_days=7):
    """
    Fetch recent Form 4 filings from EDGAR EFTS in weekly windows.
    This ensures even date coverage instead of only getting the most recent filings.
    """
    headers = {
        'User-Agent': USER_AGENT,
        'Accept': 'application/json',
    }
    now = datetime.datetime.now(datetime.UTC)
    filings = []
    n_windows = max(1, days // window_days)
    per_window = max(50, max_results // n_windows)

    for i in range(n_windows):
        w_end = now - datetime.timedelta(days=i * window_days)
        w_start = now - datetime.timedelta(days=(i + 1) * window_days)
        startdt = w_start.strftime('%Y-%m-%d')
        enddt = w_end.strftime('%Y-%m-%d')
        window_filings = _fetch_edgar_window(startdt, enddt, per_window, headers)
        filings.extend(window_filings)
        print(f'    Window {startdt}→{enddt}: {len(window_filings)} filings')
        if len(filings) >= max_results:
            break

    # Deduplicate by accession number
    seen = set()
    unique = []
    for f in filings:
        acc = f.get('accession', '')
        if acc and acc in seen:
            continue
        seen.add(acc)
        unique.append(f)

    unique.sort(key=lambda x: x.get('date', ''), reverse=True)
    return unique[:max_results]


def enrich_form4_xml(filings):
    """
    Fetch each Form 4 XML and extract: ticker, role, transaction type,
    shares, price, total value, 10b5-1 plan flag.
    SEC rate limit: 10 req/sec — we use 0.12s delay (~8 req/sec).

    Optimization: reuses enrichment data from previous edgar_feed.json
    for filings already processed. Only fetches XML for new filings.
    """
    headers = {'User-Agent': USER_AGENT}
    enriched = 0
    reused = 0
    errors = 0
    total = len(filings)

    # Load previously enriched filings to skip re-fetching XML
    prev_feed_path = os.path.join(DATA_DIR, 'edgar_feed.json')
    prev_by_accession = {}
    if os.path.exists(prev_feed_path):
        try:
            with open(prev_feed_path) as pf:
                prev_data = json.load(pf)
            for pf_entry in prev_data.get('filings', []):
                acc = pf_entry.get('accession', '')
                # Only reuse if it was actually enriched (has ticker)
                if acc and pf_entry.get('ticker'):
                    prev_by_accession[acc] = pf_entry
            print(f'  Loaded {len(prev_by_accession)} previously enriched filings for skip-logic')
        except Exception as e:
            print(f'  Could not load previous feed for skip-logic: {e}')

    for i, f in enumerate(filings):
        # Skip XML fetch if this filing was already enriched in previous run
        acc = f.get('accession', '')
        if acc and acc in prev_by_accession:
            prev = prev_by_accession[acc]
            for key in ('ticker', 'title', 'roles', 'is_10b5_1', 'transactions',
                        'buy_value', 'sell_value', 'buy_shares', 'sell_shares', 'direction'):
                if key in prev:
                    f[key] = prev[key]
            reused += 1
            enriched += 1
            continue

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

    print(f'  Enrichment complete: {enriched}/{total} enriched ({reused} reused, {enriched - reused} new XML), {errors} errors')
    # Remove xml_url from output (internal only)
    for f in filings:
        f.pop('xml_url', None)
    return filings


# ── FMP Congressional Trades ──────────────────────────────────────────────
FMP_API_KEY = os.environ.get('FMP_API_KEY', 'UefVEEvF1XXtpgWcsidPCGxcDJ6N0kXv')

def normalize_fmp_congress(raw, chamber):
    """
    Normalize a single FMP stable API response dict into the standard
    congress_feed.json format.

    FMP stable API fields: transactionDate, disclosureDate, firstName, lastName,
                           office, district, type, amount, symbol, assetDescription,
                           owner, link

    Returns a normalized dict, or None if no valid ticker symbol.
    """
    symbol = (raw.get('symbol') or '').strip().upper()
    # Skip entries with no ticker or tickers >5 chars (likely invalid)
    if not symbol or len(symbol) > 5:
        return None

    # Map FMP 'type' to standard 'Transaction' format
    fmp_type = (raw.get('type') or '').strip()
    if fmp_type.lower().startswith('purchase'):
        transaction = 'Purchase'
    elif fmp_type.lower().startswith('sale'):
        transaction = 'Sale'
    else:
        transaction = fmp_type or 'Unknown'

    # Compute disclosure delay (days between transaction and disclosure)
    txn_date = (raw.get('transactionDate') or '').strip()
    disc_date = (raw.get('disclosureDate') or '').strip()
    disclosure_delay = None
    if txn_date and disc_date:
        try:
            td = datetime.datetime.strptime(txn_date, '%Y-%m-%d')
            dd = datetime.datetime.strptime(disc_date, '%Y-%m-%d')
            disclosure_delay = (dd - td).days
        except ValueError:
            pass

    # Build representative name from firstName + lastName (stable API)
    # Fall back to 'representative' field (legacy compat) or 'office'
    first = (raw.get('firstName') or '').strip()
    last = (raw.get('lastName') or '').strip()
    if first and last:
        representative = f'{first} {last}'
    else:
        representative = (raw.get('representative') or raw.get('office') or '').strip()

    # Party not available in stable API — leave empty
    party = (raw.get('party') or '').strip()

    return {
        'Ticker': symbol,
        'TransactionDate': txn_date,
        'Representative': representative,
        'Transaction': transaction,
        'Range': (raw.get('amount') or '').strip(),
        'Chamber': chamber,
        'Party': party,
        'DisclosureDate': disc_date,
        'DisclosureDelay': disclosure_delay,
        'Source': 'FMP',
    }


def fetch_fmp_congress(api_key, pages=10):
    """
    Fetch congressional trades from FMP (Financial Modeling Prep) stable API.
    Pulls from both Senate and House endpoints, normalizes to standard format,
    deduplicates, and returns sorted list (most recent first).
    """
    all_trades = []
    seen = set()  # (ticker, date, representative) for dedup

    endpoints = [
        ('https://financialmodelingprep.com/stable/senate-latest', 'Senate'),
        ('https://financialmodelingprep.com/stable/house-latest', 'House'),
    ]

    for base_url, chamber in endpoints:
        print(f'  Fetching FMP {chamber} trades ({pages} pages)...')
        for page in range(pages):
            url = f'{base_url}?page={page}&apikey={api_key}'
            try:
                r = requests.get(url, timeout=20)
                if not r.ok:
                    body = r.text[:200] if r.text else ''
                    if r.status_code in (401, 403):
                        print(f'  FMP {chamber}: HTTP {r.status_code} — API key expired or plan lacks congress data')
                        print(f'  Renew at: https://financialmodelingprep.com/developer/docs')
                    else:
                        print(f'  FMP {chamber} page {page} error: HTTP {r.status_code} {body}')
                    break
                data = r.json()
                if isinstance(data, dict) and data.get('Error Message'):
                    print(f'  FMP {chamber}: {data["Error Message"]}')
                    break
                if not isinstance(data, list) or len(data) == 0:
                    break

                for raw_trade in data:
                    normalized = normalize_fmp_congress(raw_trade, chamber)
                    if normalized is None:
                        continue
                    # Dedup key: (ticker, date, representative)
                    dedup_key = (
                        normalized['Ticker'],
                        normalized['TransactionDate'],
                        normalized['Representative'],
                    )
                    if dedup_key not in seen:
                        seen.add(dedup_key)
                        all_trades.append(normalized)

            except Exception as e:
                print(f'  FMP {chamber} page {page} error: {e}')
                break

            time.sleep(0.3)  # Rate limit

    # Sort most recent first
    all_trades.sort(key=lambda x: x.get('TransactionDate', ''), reverse=True)
    print(f'  FMP total: {len(all_trades)} unique trades ({len(seen)} deduped)')
    return all_trades


# ── House Financial Disclosures (Direct Scraper) ─────────────────────────
# No API key needed. Fetches PTR XML index directly from the Clerk of the
# House, then parses individual filing pages for trade details.
# Rate limit: 1 req/sec for all government site requests.

# Words that look like tickers but aren't
_NON_TICKER_WORDS = frozenset([
    'AND', 'THE', 'FOR', 'LLC', 'INC', 'ETF', 'USA', 'NEW', 'ALL', 'ARE',
    'HAS', 'HIS', 'HER', 'ITS', 'NOT', 'OUR', 'OUT', 'OWN', 'CAN', 'DID',
    'GET', 'GOT', 'HAD', 'HAS', 'LET', 'MAY', 'OLD', 'PUT', 'RAN', 'SAY',
    'SHE', 'TOO', 'USE', 'WAR', 'WAY', 'WHO', 'BOY', 'HOW', 'MAN', 'TRY',
    'ASK', 'BIG', 'END', 'FAR', 'FEW', 'RUN', 'SET', 'TOP', 'NAN', 'USD',
    'ACT', 'AGO', 'BUY', 'TWO', 'COM', 'LTD', 'CEO', 'CFO', 'COO', 'CTO',
    'EST', 'AVG', 'MAX', 'MIN', 'NET', 'TAX', 'SEC', 'PER', 'JAN', 'FEB',
    'MAR', 'APR', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'REP',
    'PTR', 'PDF', 'XML', 'DOC', 'NONE', 'FUND', 'BOND', 'CASH', 'DEBT',
    'CORP', 'MISC', 'PART', 'EACH', 'FROM', 'THAT', 'THIS', 'WILL', 'WITH',
    'TYPE', 'DATE', 'NAME', 'FILE', 'FORM', 'SALE', 'SOLD',
])

# Regex for dollar amount ranges (e.g., "$1,001 - $15,000")
_AMOUNT_RE = re.compile(r'\$[\d,]+\s*-\s*\$[\d,]+')

# Regex for ticker-like strings (1-5 uppercase letters)
_TICKER_RE = re.compile(r'\b([A-Z]{1,5})\b')

# Date patterns: MM/DD/YYYY or YYYY-MM-DD
_DATE_MDY_RE = re.compile(r'(\d{1,2})/(\d{1,2})/(\d{4})')
_DATE_YMD_RE = re.compile(r'(\d{4})-(\d{2})-(\d{2})')


def fetch_house_disclosures(lookback_days=30):
    """
    Fetch recent PTR (Periodic Transaction Report) filings from the
    House of Representatives Financial Disclosures.

    Returns list of normalized trades matching congress_feed.json schema,
    or [] on any failure.
    """
    try:
        return _fetch_house_disclosures_inner(lookback_days)
    except Exception as e:
        print(f'  House scraper error (safe fallback to []): {e}')
        return []


def _fetch_house_disclosures_inner(lookback_days):
    """Inner implementation — may raise; caller wraps in try/except.

    XML index URL: https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{YEAR}FD.xml
    (NOT ptr-pdfs — that's for individual filing PDFs)

    XML structure per <Member>:
        <Prefix>, <Last>, <First>, <Suffix>, <FilingType>, <StateDst>,
        <Year>, <FilingDate> (M/D/YYYY), <DocID>
    FilingType: P=PTR (periodic transaction report), others: C/D/X/W/A/E/G

    Individual PTR docs are encrypted PDFs at:
        https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{YEAR}/{DocID}.pdf
    No HTML viewer available — trade detail extraction is NOT possible from PDFs.
    This scraper returns filing metadata only (member, date, filing count).
    """
    today = datetime.date.today()
    cutoff = today - datetime.timedelta(days=lookback_days)
    headers = {'User-Agent': 'ATLAS Research Tool'}

    # Determine which years of XML index to fetch
    years = [today.year]
    if today.month <= 2:
        years.append(today.year - 1)

    # ── STEP 1: Fetch & parse FD XML index ──────────────────────────────
    all_filings = []

    for year in years:
        xml_url = f'https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.xml'
        print(f'  Fetching House FD index: {year}...')
        try:
            r = requests.get(xml_url, headers=headers, timeout=15)
            if not r.ok:
                print(f'  House XML index {year}: HTTP {r.status_code}')
                continue
        except Exception as e:
            print(f'  House XML index {year} fetch error: {e}')
            continue

        time.sleep(1)  # rate limit

        # ── STEP 2: Parse XML, filter to PTR filings in lookback window ─
        try:
            root = ET.fromstring(r.content)
        except ET.ParseError as e:
            print(f'  House XML parse error for {year}: {e}')
            continue

        year_count = 0
        for member in root.iter('Member'):
            # Extract fields using exact tag names from the XML structure
            filing_type = _get_child_text(member, 'FilingType')
            # Only PTR filings (FilingType=P) contain transaction reports
            if filing_type != 'P':
                continue

            filing_date_str = _get_child_text(member, 'FilingDate')
            doc_id = _get_child_text(member, 'DocID')
            last = _get_child_text(member, 'Last')
            first = _get_child_text(member, 'First')
            state_dst = _get_child_text(member, 'StateDst')

            if not doc_id or not filing_date_str:
                continue

            filing_date = _parse_date_flex(filing_date_str)
            if not filing_date or filing_date < cutoff:
                continue

            member_name = _clean_member_name(f'{last}, {first}' if last else first)

            all_filings.append({
                'doc_id': doc_id,
                'member_name': member_name,
                'filing_date': filing_date.isoformat(),
                'state_district': state_dst,
                'year': year,
            })
            year_count += 1

        print(f'  House {year}: {year_count} PTR filings in last {lookback_days}d')

    print(f'  House PTR index total: {len(all_filings)} recent filings')
    if not all_filings:
        return []

    # ── STEP 3: Save filing metadata ────────────────────────────────────
    # PTR PDFs are encrypted — cannot extract trade details from them.
    # Save filing metadata for: (1) freshness tracking, (2) filing frequency
    # signals, (3) future integration if HTML viewer becomes available.
    save_json('house_filings.json', {
        'updated': datetime.datetime.now(datetime.UTC).isoformat() + 'Z',
        'count': len(all_filings),
        'source': 'House Clerk XML Index',
        'note': 'Metadata only — PTR PDFs are encrypted, trade details not extractable',
        'filings': all_filings,
    })

    # ── STEP 4: Generate trades from filing metadata ────────────────────
    # Without trade details from the PDFs, we generate one "filing signal"
    # per member per filing date. This at minimum tells us WHO filed WHEN,
    # which the Brain can cross-reference with FMP data for freshness.
    # These are NOT full trade records — they're filing activity signals.
    all_trades = []
    for filing in all_filings:
        all_trades.append({
            'Ticker': '',  # unknown without PDF parsing
            'TransactionDate': filing['filing_date'],
            'Representative': filing['member_name'],
            'Transaction': 'PTR Filing',  # indicates filing activity, not a specific trade
            'Range': '',
            'Chamber': 'House',
            'Party': '',
            'DisclosureDate': filing['filing_date'],
            'DisclosureDelay': 0,
            'Source': 'House',
            'DocID': filing['doc_id'],
        })

    # Filter out trades with no ticker — they can't merge into congress_feed
    # but we still saved the metadata in house_filings.json
    trades_with_ticker = [t for t in all_trades if t['Ticker']]

    print(f'  House scraper: {len(all_filings)} filing metadata records saved, '
          f'{len(trades_with_ticker)} trades with tickers (0 expected — PDFs encrypted)')

    return trades_with_ticker


def _parse_house_filing_page(html, member_name, filing_date):
    """
    Parse a House PTR filing HTML page for individual trades.
    Returns list of normalized trade dicts.
    """
    trades = []
    text = html

    # Look for table rows or structured data
    # House filing pages typically have tabular trade data with:
    # Asset/Ticker, Transaction Type, Date, Amount
    # Try to find ticker symbols in context of transaction keywords

    # Split into transaction blocks — look for purchase/sale keywords
    # Each "block" around a transaction keyword likely describes one trade
    lines = text.split('\n')
    full_text = ' '.join(lines)

    # Strategy: find all ticker-like symbols that appear near transaction keywords
    # Extract tickers with surrounding context
    ticker_matches = list(_TICKER_RE.finditer(full_text))

    # Find transaction types
    txn_purchase = [(m.start(), 'Purchase') for m in re.finditer(r'(?i)\b(purchase[d]?|bought|buy)\b', full_text)]
    txn_sale = [(m.start(), 'Sale') for m in re.finditer(r'(?i)\b(sale|sold|sell|disposition)\b', full_text)]
    all_txns = sorted(txn_purchase + txn_sale, key=lambda x: x[0])

    # Find dates in the text
    date_matches = []
    for m in _DATE_MDY_RE.finditer(full_text):
        try:
            d = datetime.date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
            date_matches.append((m.start(), d))
        except ValueError:
            pass
    for m in _DATE_YMD_RE.finditer(full_text):
        try:
            d = datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            date_matches.append((m.start(), d))
        except ValueError:
            pass
    date_matches.sort(key=lambda x: x[0])

    # Find amount ranges
    amount_matches = [(m.start(), m.group()) for m in _AMOUNT_RE.finditer(full_text)]

    # For each transaction keyword, find the nearest valid ticker, date, and amount
    seen_trades = set()
    for txn_pos, txn_type in all_txns:
        # Look for nearest ticker within ±500 chars
        best_ticker = None
        best_dist = 500
        for tm in ticker_matches:
            t = tm.group(1)
            if t in _NON_TICKER_WORDS or len(t) < 2:
                continue
            dist = abs(tm.start() - txn_pos)
            if dist < best_dist:
                best_dist = dist
                best_ticker = t

        if not best_ticker:
            continue

        # Nearest date within ±1000 chars
        best_date = None
        best_date_dist = 1000
        for dp, d in date_matches:
            dist = abs(dp - txn_pos)
            if dist < best_date_dist:
                best_date_dist = dist
                best_date = d

        # Use filing date as fallback for transaction date
        txn_date = best_date.isoformat() if best_date else filing_date

        # Nearest amount
        best_amount = ''
        best_amt_dist = 500
        for ap, amt in amount_matches:
            dist = abs(ap - txn_pos)
            if dist < best_amt_dist:
                best_amt_dist = dist
                best_amount = amt

        # ── STEP 4: Normalize to congress_feed.json schema ──────────────
        # Dedup key within this filing
        dedup_key = (best_ticker, txn_date, txn_type)
        if dedup_key in seen_trades:
            continue
        seen_trades.add(dedup_key)

        # Compute disclosure delay
        disclosure_delay = None
        if best_date:
            try:
                fd = datetime.date.fromisoformat(filing_date)
                disclosure_delay = (fd - best_date).days
            except (ValueError, TypeError):
                pass

        trades.append({
            'Ticker': best_ticker,
            'TransactionDate': txn_date,
            'Representative': member_name,
            'Transaction': txn_type,
            'Range': best_amount,
            'Chamber': 'House',
            'Party': '',  # not reliably available in filing page
            'DisclosureDate': filing_date,
            'DisclosureDelay': disclosure_delay,
            'Source': 'House',
        })

    return trades


def _get_child_text(elem, tag_name):
    """Get text of a child element by tag name (case-insensitive)."""
    for child in elem:
        if child.tag and child.tag.lower() == tag_name.lower():
            return (child.text or '').strip()
    return ''


def _clean_member_name(name):
    """Normalize 'LAST, First' or 'First Last' to 'First Last'."""
    if not name:
        return ''
    name = name.strip()
    # Handle "LAST, First" format
    if ',' in name:
        parts = name.split(',', 1)
        last = parts[0].strip().title()
        first = parts[1].strip().title()
        return f'{first} {last}'
    return name.title()


def _parse_date_flex(date_str):
    """Parse a date string in MM/DD/YYYY or YYYY-MM-DD format."""
    if not date_str:
        return None
    date_str = date_str.strip()
    for fmt in ('%m/%d/%Y', '%Y-%m-%d', '%m/%d/%y'):
        try:
            return datetime.datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


# ── Senate Financial Disclosures ───────────────────────────────────────────

def fetch_senate_disclosures(lookback_days=30):
    """
    Fetch recent PTR filings from Senate Electronic Financial Disclosures.

    Uses CSRF-protected search at efdsearch.senate.gov.
    Returns list of normalized trades matching congress_feed.json schema,
    or [] on any failure (Cloudflare, CSRF change, etc.)
    """
    try:
        return _fetch_senate_disclosures_inner(lookback_days)
    except Exception as e:
        print(f'  Senate scraper error (safe fallback to []): {e}')
        return []


def _fetch_senate_disclosures_inner(lookback_days):
    """Inner implementation — may raise; caller wraps in try/except."""
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=lookback_days)

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'ATLAS Research Tool/1.0',
    })

    # STEP 1 — Acquire CSRF token + session cookie
    print('  Senate: acquiring CSRF token...')
    try:
        r = session.get('https://efdsearch.senate.gov/search/', timeout=15)
    except Exception as e:
        print(f'  Senate: connection failed — {e}')
        return []

    if r.status_code == 403:
        print('  Senate: Cloudflare blocked (403)')
        return []

    csrf = re.search(r'csrfmiddlewaretoken.*?value="([^"]+)"', r.text)
    if not csrf:
        print('  Senate: CSRF token not found — page structure may have changed')
        return []
    csrf_token = csrf.group(1)

    # STEP 2 — POST search for recent PTRs
    print(f'  Senate: searching PTRs from {start_date} to {today}...')
    search_url = 'https://efdsearch.senate.gov/search/report/data/'
    payload = {
        'start': '0',
        'length': '100',
        'report_types': '[11]',  # 11 = Periodic Transaction Report
        'submitted_start_date': f'{start_date.strftime("%m/%d/%Y")} 00:00:00',
        'submitted_end_date': f'{today.strftime("%m/%d/%Y")} 23:59:59',
        'csrfmiddlewaretoken': csrf_token,
    }
    search_headers = {
        'X-CSRFToken': csrf_token,
        'Referer': 'https://efdsearch.senate.gov/search/',
        'Content-Type': 'application/x-www-form-urlencoded',
    }

    try:
        r = session.post(search_url, data=payload, headers=search_headers, timeout=20)
    except Exception as e:
        print(f'  Senate: search POST failed — {e}')
        return []

    if r.status_code in (403, 429):
        print(f'  Senate: Cloudflare blocked ({r.status_code})')
        return []

    try:
        data = r.json()
    except (json.JSONDecodeError, ValueError):
        print('  Senate: search response not JSON — blocked or structure changed')
        return []

    records = data.get('data', [])
    total = data.get('recordsTotal', 0)
    print(f'  Senate: {len(records)} PTR filings found (total: {total})')

    if not records:
        return []

    # STEP 3 — Parse filing metadata from search results
    # Each record is a list: [first_name, last_name, office, report_type_link, date_received]
    all_trades = []
    filings_metadata = []

    for record in records:
        try:
            if not isinstance(record, (list, tuple)) or len(record) < 5:
                continue

            first_name = _strip_html(str(record[0]))
            last_name = _strip_html(str(record[1]))
            # record[3] contains link to the filing detail page
            link_html = str(record[3])
            date_received = _strip_html(str(record[4]))

            member_name = _clean_member_name(f'{last_name}, {first_name}')

            # Extract detail URL from link HTML
            link_match = re.search(r'href="(/search/view/[^"]+)"', link_html)
            detail_path = link_match.group(1) if link_match else None

            filing_date = _parse_date_flex(date_received)
            if not filing_date:
                continue

            filings_metadata.append({
                'member_name': member_name,
                'filing_date': filing_date.isoformat(),
                'detail_path': detail_path,
            })
        except Exception:
            continue

    print(f'  Senate: {len(filings_metadata)} filing metadata records parsed')

    # STEP 4 — Fetch detail pages and parse trade tables
    trades_found = 0
    for filing in filings_metadata:
        if not filing['detail_path']:
            continue

        detail_url = f"https://efdsearch.senate.gov{filing['detail_path']}"
        try:
            time.sleep(2)  # rate limit
            r = session.get(detail_url, timeout=15)
            if r.status_code != 200:
                continue

            page_trades = _parse_senate_ptr_page(
                r.text, filing['member_name'], filing['filing_date']
            )
            all_trades.extend(page_trades)
            trades_found += len(page_trades)
        except Exception:
            continue

    # Save metadata regardless of trade extraction success
    save_json('senate_filings.json', {
        'updated': datetime.datetime.now(datetime.UTC).isoformat() + 'Z',
        'count': len(filings_metadata),
        'trades_extracted': trades_found,
        'source': 'Senate EFD Search',
        'filings': filings_metadata,
    })

    print(f'  Senate scraper: {len(filings_metadata)} filings, '
          f'{trades_found} trades extracted')

    return all_trades


def _parse_senate_ptr_page(html, member_name, filing_date):
    """Parse a Senate PTR detail page for individual trades.

    Senate PTR pages have a table with columns:
    Transaction Date, Owner, Ticker, Asset Name, Type, Amount
    Returns list of normalized trade dicts.
    """
    trades = []

    # Find table rows — Senate uses standard HTML tables
    # Look for rows containing transaction data
    row_pattern = re.compile(
        r'<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE
    )
    cell_pattern = re.compile(
        r'<td[^>]*>(.*?)</td>', re.DOTALL | re.IGNORECASE
    )

    for row_match in row_pattern.finditer(html):
        row_html = row_match.group(1)
        cells = [_strip_html(c.group(1)).strip() for c in cell_pattern.finditer(row_html)]

        if len(cells) < 4:
            continue

        # Try to identify transaction rows by looking for ticker-like patterns
        # and transaction types (Purchase, Sale, Exchange)
        ticker = None
        txn_date = None
        txn_type = None
        amount = None

        for cell in cells:
            # Check for ticker (1-5 uppercase letters)
            if not ticker and re.match(r'^[A-Z]{1,5}$', cell.strip()):
                ticker = cell.strip()
            # Check for date
            if not txn_date:
                parsed_date = _parse_date_flex(cell)
                if parsed_date:
                    txn_date = parsed_date.isoformat()
            # Check for transaction type
            cell_lower = cell.lower().strip()
            if not txn_type and cell_lower in ('purchase', 'sale', 'sale (full)',
                                                'sale (partial)', 'exchange'):
                txn_type = 'Purchase' if 'purchase' in cell_lower else 'Sale'
            # Check for amount range
            if not amount and '$' in cell:
                amount = cell.strip()

        if ticker and txn_type:
            trades.append({
                'Ticker': ticker,
                'TransactionDate': txn_date or filing_date,
                'Representative': member_name,
                'Transaction': txn_type,
                'Range': amount or '',
                'Chamber': 'Senate',
                'Party': '',
                'DisclosureDate': filing_date,
                'DisclosureDelay': 0,
                'Source': 'Senate',
            })

    return trades


def _strip_html(text):
    """Remove HTML tags from a string."""
    return re.sub(r'<[^>]+>', '', text).strip()


# ── Quiver Quant — Lobbying + Committee Data ──────────────────────────────

COMMITTEE_SECTOR_MAP = {
    "Armed Services": ["Aerospace & Defense", "Government"],
    "Financial Services": ["Financials", "Banking"],
    "Energy and Commerce": ["Energy", "Utilities", "Healthcare"],
    "Intelligence": ["Technology", "Defense"],
    "Agriculture": ["Consumer Staples", "Materials"],
    "Science, Space, and Technology": ["Technology", "Semiconductors"],
    "Health, Education, Labor, and Pensions": ["Healthcare", "Biotech"],
    "Banking, Housing, and Urban Affairs": ["Financials", "Banking"],
    "Commerce, Science, and Transportation": ["Technology", "Industrials"],
    "Environment and Public Works": ["Utilities", "Materials"],
    "Appropriations": [],  # broad — no sector signal
    "Judiciary": [],
}


def fetch_lobbying_data(tickers):
    """Fetch lobbying spending data from Quiver Quant for each ticker.

    Computes:
      lobbying_active: bool (spend in last 2 quarters)
      lobbying_trend: (recent_spend - older_spend) / older_spend
    Saves to data/lobbying_data.json.
    Skips gracefully if QUIVER_KEY not set.
    """
    if not QUIVER_KEY:
        print('  QUIVER_KEY not set — lobbying features will remain NULL')
        save_json('lobbying_data.json', {
            'updated': datetime.datetime.now(datetime.UTC).isoformat() + 'Z',
            'count': 0, 'source': 'QuiverQuant', 'tickers': {},
        })
        return
    if not tickers:
        return

    print(f'  Fetching lobbying data for {len(tickers)} tickers...')
    lobbying = {}
    fetched = 0
    failed = 0
    headers = {
        'Authorization': f'Token {QUIVER_KEY}',
        'Accept': 'application/json',
    }

    for ticker in tickers:
        try:
            url = f'https://api.quiverquant.com/beta/live/lobbying/{ticker}'
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 429:
                print('  Quiver rate limit hit — stopping')
                break
            if r.status_code != 200:
                failed += 1
                continue

            data = r.json()
            if not data:
                continue

            # Get last 4 quarters of spending
            quarters = sorted(data, key=lambda x: x.get('Date', ''), reverse=True)[:4]
            if not quarters:
                continue

            recent = quarters[:2]  # last 2 quarters
            older = quarters[2:4]  # older 2 quarters

            recent_spend = sum(q.get('Amount', 0) for q in recent)
            older_spend = sum(q.get('Amount', 0) for q in older)

            lobbying[ticker] = {
                'lobbying_active': recent_spend > 0,
                'lobbying_trend': round(
                    (recent_spend - older_spend) / max(older_spend, 1), 4
                ) if older_spend > 0 else 0,
                'recent_spend': recent_spend,
                'quarters_found': len(quarters),
            }
            fetched += 1
        except Exception:
            failed += 1
        time.sleep(1)

    save_json('lobbying_data.json', {
        'updated': datetime.datetime.now(datetime.UTC).isoformat() + 'Z',
        'count': len(lobbying),
        'source': 'QuiverQuant',
        'tickers': lobbying,
    })
    print(f'  Lobbying data: {fetched} tickers fetched, {failed} failed')


def fetch_quiver_committee_data():
    """Fetch committee membership data from Quiver Quant.

    Builds COMMITTEE_MAP: {rep_name: [committees]}
    Saves to data/committee_data.json — refresh weekly.
    Skips gracefully if QUIVER_KEY not set.
    """
    if not QUIVER_KEY:
        print('  QUIVER_KEY not set — skipping Quiver committee data')
        return
    print('  Fetching committee membership from Quiver Quant...')

    headers = {
        'Authorization': f'Token {QUIVER_KEY}',
        'Accept': 'application/json',
    }

    # Load existing congress feed to get rep names
    cong_path = os.path.join(DATA_DIR, 'congress_feed.json')
    if not os.path.exists(cong_path):
        print('  No congress_feed.json — skipping Quiver committee data')
        return

    with open(cong_path) as f:
        cong_data = json.load(f)
    reps = list({t.get('Representative', '') for t in cong_data.get('trades', []) if t.get('Representative')})

    committee_map = {}
    fetched = 0
    for rep in reps[:50]:  # limit to avoid rate limit
        try:
            # URL-encode the rep name
            encoded = requests.utils.quote(rep)
            url = f'https://api.quiverquant.com/beta/live/congresstrading/{encoded}'
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 429:
                print('  Quiver rate limit hit — stopping')
                break
            if r.status_code != 200:
                continue

            data = r.json()
            if not data:
                continue

            # Extract committees from trading data (Quiver includes committee info)
            committees = set()
            for trade in data:
                comm = trade.get('Committee', '')
                if comm:
                    committees.add(comm)
            if committees:
                committee_map[rep] = list(committees)
                fetched += 1
        except Exception:
            continue
        time.sleep(1)

    if committee_map:
        save_json('committee_data.json', {
            'updated': datetime.datetime.now(datetime.UTC).isoformat() + 'Z',
            'count': len(committee_map),
            'source': 'QuiverQuant',
            'sector_map': COMMITTEE_SECTOR_MAP,
            'members': committee_map,
        })
    print(f'  Quiver committee data: {fetched} reps fetched')


# ── FMP Insider Trades ─────────────────────────────────────────────────────
def fetch_fmp_insiders(api_key, pages=10):
    """
    Fetch recent insider trades from FMP (Financial Modeling Prep).
    NOTE (Mar 2026): FMP deprecated v3/v4 endpoints. No stable insider-trading
    endpoint exists yet. This function probes for one; if none found, returns []
    and EDGAR Form 4 (direct SEC) remains the sole insider source.
    """
    all_trades = []
    seen = set()  # (symbol, date, reportingName) for dedup

    # FMP killed legacy v3/v4 endpoints (403 "Legacy Endpoint no longer supported").
    # Stable insider-trading endpoint doesn't exist as of Mar 2026.
    # Keep probing in case FMP adds one later.
    ENDPOINTS = [
        ('stable', 'https://financialmodelingprep.com/stable/insider-trading'
                   '?transactionType=P-Purchase&page={page}&apikey={key}'),
        ('stable-latest', 'https://financialmodelingprep.com/stable/insider-latest'
                          '?page={page}&apikey={key}'),
    ]

    working_endpoint = None
    for ep_name, ep_template in ENDPOINTS:
        test_url = ep_template.format(page=0, key=api_key)
        try:
            r = requests.get(test_url, timeout=20)
            if r.ok:
                data = r.json()
                if isinstance(data, list) and len(data) > 0:
                    print(f'  FMP insider endpoint: {ep_name} working ({len(data)} items on page 0)')
                    working_endpoint = (ep_name, ep_template)
                    break
        except Exception as e:
            print(f'  FMP insider endpoint {ep_name}: {e}')
        time.sleep(0.3)

    if not working_endpoint:
        print('  FMP insider endpoints: no stable endpoint available (v3/v4 deprecated).')
        print('  EDGAR Form 4 (direct SEC) remains the primary insider data source.')
        return []

    ep_name, ep_template = working_endpoint
    is_rss = 'rss' in ep_name

    for page in range(pages):
        url = ep_template.format(page=page, key=api_key)
        try:
            r = requests.get(url, timeout=20)
            if not r.ok:
                print(f'  FMP insiders page {page} error: HTTP {r.status_code}')
                break
            data = r.json()
            if not isinstance(data, list) or len(data) == 0:
                break

            for raw in data:
                symbol = (raw.get('symbol') or '').strip().upper()
                if not symbol or len(symbol) > 5:
                    continue

                # Filter to purchases — RSS feed returns all types
                txn_type = raw.get('transactionType', '')
                if is_rss and 'P' not in txn_type.upper().split('-')[0]:
                    continue

                txn_date = raw.get('transactionDate', '')
                reporter = raw.get('reportingName', '')
                dedup_key = (symbol, txn_date, reporter)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                shares = raw.get('securitiesTransacted', 0) or 0
                owned = raw.get('securitiesOwned', 0) or 0

                all_trades.append({
                    'Ticker': symbol,
                    'Date': txn_date,
                    'Owner': reporter,
                    'Transaction': txn_type,
                    'Shares': shares,
                    'SecuritiesOwned': owned,
                    'SecurityName': raw.get('securityName', ''),
                    'FormType': raw.get('formType', ''),
                    'Link': raw.get('link', ''),
                    'Source': 'FMP',
                })

        except Exception as e:
            print(f'  FMP insiders page {page} error: {e}')
            break

        time.sleep(0.3)  # Rate limit

    all_trades.sort(key=lambda x: x.get('Date', ''), reverse=True)
    print(f'  FMP total: {len(all_trades)} insider purchases ({len(seen)} deduped) via {ep_name}')
    return all_trades


# ── Volume Data (yfinance) ──────────────────────────────────────────────────
def fetch_volume_data(tickers):
    """
    Fetch 35-day volume history for tickers with recent signals.
    Computes relative volume (latest / 30-day avg).
    Saves to data/volume_data.json.
    """
    try:
        import yfinance as yf
    except ImportError:
        print('  yfinance not installed — skipping volume data')
        return

    if not tickers:
        print('  No tickers for volume data')
        return

    print(f'  Fetching volume data for {len(tickers)} tickers...')
    volume_data = {}
    fetched = 0
    failed = 0

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period='35d')
            if hist.empty or 'Volume' not in hist.columns:
                continue
            vols = hist['Volume'].dropna().values
            if len(vols) < 5:
                continue
            avg_30d = float(vols[:-1].mean()) if len(vols) > 1 else float(vols.mean())
            latest = float(vols[-1])
            rel_vol = round(latest / avg_30d, 4) if avg_30d > 0 else None
            volume_data[ticker] = {
                'rel_volume': rel_vol,
                'avg_30d': round(avg_30d),
                'latest_volume': round(latest),
                'date': hist.index[-1].strftime('%Y-%m-%d'),
            }
            fetched += 1
        except Exception as e:
            failed += 1
        time.sleep(0.2)

    if volume_data:
        save_json('volume_data.json', {
            'updated': datetime.datetime.now(datetime.UTC).isoformat() + 'Z',
            'count': len(volume_data),
            'tickers': volume_data,
        })
    print(f'  Volume data: {fetched} tickers fetched, {failed} failed')


# ── Analyst Data (yfinance) ──────────────────────────────────────────────────
def fetch_analyst_data(tickers):
    """
    Fetch analyst recommendation and price target data for signal tickers.
    Computes revision momentum (upgrades - downgrades in last 30d) and consensus.
    Saves to data/analyst_data.json.
    """
    try:
        import yfinance as yf
    except ImportError:
        print('  yfinance not installed — skipping analyst data')
        return

    if not tickers:
        print('  No tickers for analyst data')
        return

    print(f'  Fetching analyst data for {len(tickers)} tickers...')
    analyst_data = {}
    fetched = 0
    failed = 0
    cutoff_30d = (datetime.date.today() - datetime.timedelta(days=30)).isoformat()

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            entry = {
                'upgrades_30d': 0,
                'downgrades_30d': 0,
                'revision_momentum': 0,
                'analyst_consensus': None,
            }

            # Get recommendations
            try:
                recs = t.recommendations
                if recs is not None and not recs.empty:
                    # Filter to last 30 days
                    if hasattr(recs.index, 'strftime'):
                        recent = recs[recs.index >= cutoff_30d]
                    else:
                        recent = recs.tail(10)  # fallback: last 10 entries

                    for _, row in recent.iterrows():
                        to_grade = str(row.get('To Grade', row.get('toGrade', ''))).lower()
                        if any(w in to_grade for w in ['buy', 'strong buy', 'overweight', 'outperform']):
                            entry['upgrades_30d'] += 1
                        elif any(w in to_grade for w in ['sell', 'underweight', 'underperform', 'reduce']):
                            entry['downgrades_30d'] += 1

                    entry['revision_momentum'] = entry['upgrades_30d'] - entry['downgrades_30d']
            except Exception:
                pass

            # Get analyst consensus from recommendations_summary if available
            try:
                summary = getattr(t, 'recommendations_summary', None)
                if summary is not None and not summary.empty:
                    row = summary.iloc[0] if len(summary) > 0 else None
                    if row is not None:
                        strong_buy = int(row.get('strongBuy', 0))
                        buy = int(row.get('buy', 0))
                        hold = int(row.get('hold', 0))
                        sell = int(row.get('sell', 0))
                        strong_sell = int(row.get('strongSell', 0))
                        total = strong_buy + buy + hold + sell + strong_sell
                        if total > 0:
                            entry['analyst_consensus'] = round((strong_buy + buy) / total, 3)
            except Exception:
                pass

            analyst_data[ticker] = entry
            fetched += 1
        except Exception:
            failed += 1
        time.sleep(0.2)

    if analyst_data:
        save_json('analyst_data.json', {
            'updated': datetime.datetime.now(datetime.UTC).isoformat() + 'Z',
            'count': len(analyst_data),
            'tickers': analyst_data,
        })
    print(f'  Analyst data: {fetched} tickers fetched, {failed} failed')


# ── Earnings Surprise Data (yfinance) ────────────────────────────────────────
def fetch_earnings_surprise(tickers):
    """
    Fetch most recent earnings surprise for signal tickers.
    Computes surprise_pct = (actual - estimate) / |estimate| × 100.
    Saves to data/earnings_surprise.json.
    """
    try:
        import yfinance as yf
    except ImportError:
        print('  yfinance not installed — skipping earnings surprise')
        return

    if not tickers:
        print('  No tickers for earnings surprise')
        return

    print(f'  Fetching earnings surprise for {len(tickers)} tickers...')
    surprise_data = {}
    fetched = 0
    failed = 0

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            # get_earnings_dates returns recent + upcoming with EPS estimates/actuals
            dates = t.get_earnings_dates(limit=8)
            if dates is None or dates.empty:
                continue

            # Find most recent row with actual EPS (reported earnings)
            for _, row in dates.iterrows():
                actual = row.get('Reported EPS')
                estimate = row.get('EPS Estimate')
                if actual is not None and estimate is not None:
                    try:
                        actual_f = float(actual)
                        estimate_f = float(estimate)
                    except (ValueError, TypeError):
                        continue
                    if abs(estimate_f) > 0.001:
                        surprise_pct = round((actual_f - estimate_f) / abs(estimate_f) * 100, 2)
                    elif actual_f > 0:
                        surprise_pct = 100.0
                    elif actual_f < 0:
                        surprise_pct = -100.0
                    else:
                        surprise_pct = 0.0
                    surprise_data[ticker] = {
                        'surprise_pct': surprise_pct,
                        'actual_eps': actual_f,
                        'estimate_eps': estimate_f,
                        'date': _.strftime('%Y-%m-%d') if hasattr(_, 'strftime') else str(_),
                    }
                    fetched += 1
                    break  # most recent only
        except Exception:
            failed += 1
        time.sleep(0.2)

    save_json('earnings_surprise.json', {
        'updated': datetime.datetime.now(datetime.UTC).isoformat() + 'Z',
        'count': len(surprise_data),
        'tickers': surprise_data,
    })
    print(f'  Earnings surprise: {fetched} tickers fetched, {failed} failed')


# ── News Sentiment (Yahoo RSS + VADER, Finnhub optional) ─────────────────────

def _fetch_yahoo_rss_headlines(ticker, max_items=20):
    """Fetch recent headlines from Yahoo Finance RSS (free, no API key)."""
    url = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US'
    try:
        r = requests.get(url, timeout=10, headers={'User-Agent': USER_AGENT})
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.content)
        headlines = []
        for item in root.iter('item'):
            title = item.findtext('title', '')
            if title:
                headlines.append(title)
            if len(headlines) >= max_items:
                break
        return headlines
    except Exception:
        return []


def _fetch_finnhub_headlines(ticker, finnhub_key, max_items=20):
    """Fetch recent headlines from Finnhub API (requires API key)."""
    today = datetime.date.today()
    from_date = (today - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
    to_date = today.strftime('%Y-%m-%d')
    try:
        url = (f'https://finnhub.io/api/v1/company-news'
               f'?symbol={ticker}&from={from_date}&to={to_date}'
               f'&token={finnhub_key}')
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return []
        articles = r.json()
        return [a.get('headline', '') for a in articles[:max_items] if a.get('headline')]
    except Exception:
        return []


def _score_headlines_vader(headlines, vader):
    """Score headlines with VADER. Returns list of compound scores."""
    return [vader.polarity_scores(h)['compound'] for h in headlines if h]


def _score_headlines_keyword(headlines):
    """Score headlines with keyword heuristic (fallback when VADER unavailable)."""
    scores = []
    pos_words = ['beat', 'surge', 'gain', 'profit', 'upgrade',
                 'growth', 'strong', 'record', 'bullish', 'buy',
                 'raise', 'exceed', 'outperform', 'rally']
    neg_words = ['miss', 'fall', 'loss', 'downgrade', 'weak',
                 'decline', 'bearish', 'sell', 'cut', 'warning',
                 'layoff', 'recall', 'lawsuit', 'investigation']
    for h in headlines:
        if not h:
            continue
        hl = h.lower()
        pos = sum(1 for w in pos_words if w in hl)
        neg = sum(1 for w in neg_words if w in hl)
        scores.append((pos - neg) / max(pos + neg, 1))
    return scores


def fetch_news_sentiment(tickers):
    """
    Fetch recent news headlines and score sentiment.
    Primary source: Yahoo Finance RSS (free, no API key).
    Upgrade: Finnhub headlines when FINNHUB_KEY is set.
    Scoring: VADER (vaderSentiment) → keyword heuristic fallback.
    Saves to data/news_sentiment.json.
    """
    if not tickers:
        print('  No tickers for news sentiment')
        save_json('news_sentiment.json', {
            'updated': datetime.datetime.now(datetime.UTC).isoformat() + 'Z',
            'count': 0, 'method': 'none', 'tickers': {},
        })
        return

    # Load VADER scorer (graceful fallback to keyword heuristic)
    vader = None
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        vader = SentimentIntensityAnalyzer()
    except ImportError:
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            import nltk
            try:
                vader = SentimentIntensityAnalyzer()
            except LookupError:
                nltk.download('vader_lexicon', quiet=True)
                vader = SentimentIntensityAnalyzer()
        except ImportError:
            pass

    use_finnhub = bool(FINNHUB_KEY)
    method = 'vader' if vader else 'keyword'
    source = 'finnhub+yahoo' if use_finnhub else 'yahoo_rss'
    print(f'  Fetching news sentiment for {len(tickers)} tickers '
          f'(source: {source}, scorer: {method})...')

    sentiment_data = {}
    fetched = 0
    failed = 0

    for ticker in tickers:
        try:
            # Primary: Yahoo RSS (always available, no API key)
            headlines = _fetch_yahoo_rss_headlines(ticker)

            # Upgrade: merge Finnhub headlines if key available
            if use_finnhub:
                fh_headlines = _fetch_finnhub_headlines(ticker, FINNHUB_KEY)
                # Deduplicate by lowercase title
                seen = {h.lower() for h in headlines}
                for h in fh_headlines:
                    if h.lower() not in seen:
                        headlines.append(h)
                        seen.add(h.lower())

            if not headlines:
                continue

            # Score headlines
            if vader:
                scores = _score_headlines_vader(headlines, vader)
            else:
                scores = _score_headlines_keyword(headlines)

            if scores:
                avg_score = round(sum(scores) / len(scores), 4)
                strong_pos = sum(1 for s in scores if s > 0.5)
                strong_neg = sum(1 for s in scores if s < -0.5)
                sentiment_data[ticker] = {
                    'sentiment_30d': avg_score,
                    'article_count': len(scores),
                    'positive_pct': round(sum(1 for s in scores if s > 0.05) / len(scores), 3),
                    'strong_positive_count': strong_pos,
                    'strong_negative_count': strong_neg,
                }
                fetched += 1
        except Exception:
            failed += 1
        time.sleep(0.25)  # Rate-limit for RSS

    save_json('news_sentiment.json', {
        'updated': datetime.datetime.now(datetime.UTC).isoformat() + 'Z',
        'count': len(sentiment_data),
        'method': method,
        'source': source,
        'tickers': sentiment_data,
    })
    print(f'  News sentiment: {fetched} tickers scored, {failed} failed '
          f'({method} via {source})')


# ── Committee Membership Data (GitHub unitedstates/congress-legislators) ─────

# Map committee names/jurisdiction keywords to GICS sectors for overlap detection
_COMMITTEE_SECTOR_MAP = {
    # Finance / Banking
    'banking':        'Financial Services',
    'finance':        'Financial Services',
    'financial':      'Financial Services',
    'securities':     'Financial Services',
    'insurance':      'Financial Services',
    # Energy
    'energy':         'Energy',
    'natural resources': 'Energy',
    'nuclear':        'Energy',
    'oil':            'Energy',
    'gas':            'Energy',
    # Technology
    'science':        'Technology',
    'technology':     'Technology',
    'space':          'Technology',
    'innovation':     'Technology',
    'cyber':          'Technology',
    # Healthcare
    'health':         'Healthcare',
    'drug':           'Healthcare',
    'pharmaceutical': 'Healthcare',
    'biotech':        'Healthcare',
    'medical':        'Healthcare',
    # Defense / Industrials
    'armed services': 'Industrials',
    'defense':        'Industrials',
    'military':       'Industrials',
    'veterans':       'Industrials',
    'homeland security': 'Industrials',
    # Telecom
    'commerce':       'Communication Services',
    'communications': 'Communication Services',
    'telecommunications': 'Communication Services',
    # Real Estate / Utilities
    'housing':        'Real Estate',
    'infrastructure': 'Industrials',
    'transportation': 'Industrials',
    # Agriculture / Consumer Staples
    'agriculture':    'Consumer Staples',
    'nutrition':      'Consumer Staples',
    'food':           'Consumer Staples',
    'forestry':       'Basic Materials',
}


def fetch_committee_data():
    """
    Download current congress committee membership from GitHub
    (unitedstates/congress-legislators). Builds member→committees→sectors
    mapping. Saves to data/committee_data.json.

    No API key needed — public GitHub raw content.
    """
    try:
        import yaml
    except ImportError:
        print('  PyYAML not installed — skipping committee data')
        return

    BASE = 'https://raw.githubusercontent.com/unitedstates/congress-legislators/main'
    print('  Fetching committee definitions...')
    try:
        r_comms = requests.get(f'{BASE}/committees-current.yaml', timeout=30)
        r_comms.raise_for_status()
        committees_raw = yaml.safe_load(r_comms.text)
    except Exception as e:
        print(f'  Committee definitions failed: {e}')
        return

    # Build committee_id → {name, type, sectors} mapping
    committee_info = {}
    for comm in committees_raw:
        cid = comm.get('thomas_id', '')
        name = comm.get('name', '')
        ctype = comm.get('type', '')
        jurisdiction = (comm.get('jurisdiction', '') or '').lower()
        name_lower = name.lower()

        # Map committee to sectors via keyword matching
        sectors = set()
        for keyword, sector in _COMMITTEE_SECTOR_MAP.items():
            if keyword in name_lower or keyword in jurisdiction:
                sectors.add(sector)

        committee_info[cid] = {
            'name': name,
            'type': ctype,
            'sectors': sorted(sectors),
        }

    print(f'  Parsed {len(committee_info)} committees')

    print('  Fetching committee membership...')
    try:
        r_members = requests.get(f'{BASE}/committee-membership-current.yaml', timeout=30)
        r_members.raise_for_status()
        membership_raw = yaml.safe_load(r_members.text)
    except Exception as e:
        print(f'  Committee membership failed: {e}')
        return

    # Build member_name → {committees, sectors} mapping
    # Membership YAML: { committee_id: [ {name, party, rank, bioguide, ...} ] }
    member_map = {}  # name_key → { committees: [...], sectors: set() }
    for comm_id, members in membership_raw.items():
        # Strip subcommittee suffix (e.g., SSAF13 → SSAF)
        parent_id = comm_id[:4] if len(comm_id) > 4 else comm_id
        info = committee_info.get(parent_id, {})
        comm_name = info.get('name', comm_id)
        comm_sectors = info.get('sectors', [])

        for m in (members or []):
            name = m.get('name', '')
            if not name:
                continue
            # Normalize name for matching: "LastName, FirstName" → "firstname lastname"
            name_key = name.strip().lower()
            if name_key not in member_map:
                member_map[name_key] = {
                    'name': name,
                    'bioguide': m.get('bioguide', ''),
                    'committees': [],
                    'sectors': set(),
                }
            # Only add parent committee (not subcommittees) to avoid duplication
            if comm_id == parent_id:
                member_map[name_key]['committees'].append(comm_name)
            member_map[name_key]['sectors'].update(comm_sectors)

    # Convert sets to sorted lists for JSON serialization
    for key in member_map:
        member_map[key]['sectors'] = sorted(member_map[key]['sectors'])

    save_json('committee_data.json', {
        'updated': datetime.datetime.now(datetime.UTC).isoformat() + 'Z',
        'members_count': len(member_map),
        'committees_count': len(committee_info),
        'committees': committee_info,
        'members': {k: v for k, v in member_map.items()},
    })
    print(f'  Committee data: {len(member_map)} members across {len(committee_info)} committees')


# ── Short Interest Data (yfinance) ─────────────────────────────────────────────

def fetch_short_interest(tickers):
    """Fetch short interest data for tickers via yfinance.

    Collects: shortPercentOfFloat, sharesShort, sharesShortPriorMonth, shortRatio (days to cover).
    Saves to data/short_interest.json.
    """
    if not tickers:
        print('  No tickers for short interest')
        return

    try:
        import yfinance as yf
    except ImportError:
        print('  yfinance not installed — skipping short interest')
        return

    si_data = {}
    errors = 0
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            short_pct = info.get('shortPercentOfFloat')
            shares_short = info.get('sharesShort')
            shares_prior = info.get('sharesShortPriorMonth')
            short_ratio = info.get('shortRatio')

            if short_pct is not None or shares_short is not None:
                entry = {
                    'short_pct_float': round(short_pct, 4) if short_pct else None,
                    'shares_short': shares_short,
                    'shares_short_prior': shares_prior,
                    'short_ratio': round(short_ratio, 2) if short_ratio else None,
                }
                # Compute change vs prior month
                if shares_short and shares_prior and shares_prior > 0:
                    entry['short_change_pct'] = round(
                        (shares_short - shares_prior) / shares_prior, 4)
                si_data[ticker] = entry
            time.sleep(0.15)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f'  Short interest error for {ticker}: {e}')

    result = {
        'updated': datetime.datetime.now(datetime.UTC).isoformat() + 'Z',
        'source': 'yfinance',
        'tickers': si_data,
    }
    save_json('short_interest.json', result)
    print(f'  Short interest: {len(si_data)} tickers ({errors} errors)')


# ── Options Flow Data (yfinance) ───────────────────────────────────────────────

def fetch_options_data(tickers):
    """Fetch options chain data for tickers via yfinance.

    Computes: put/call ratio, bullish/bearish signal, unusual OTM call activity.
    Saves to data/options_flow.json.
    """
    if not tickers:
        print('  No tickers for options data')
        return

    try:
        import yfinance as yf
    except ImportError:
        print('  yfinance not installed — skipping options data')
        return

    options_data = {}
    errors = 0
    for ticker in tickers[:50]:  # limit to top 50 signal tickers
        try:
            t = yf.Ticker(ticker)
            expirations = t.options
            if not expirations:
                continue

            # Use nearest 2 expiries for signal
            near_expiries = expirations[:2]
            total_call_vol = 0
            total_put_vol = 0
            unusual_strikes = []

            for expiry in near_expiries:
                try:
                    chain = t.option_chain(expiry)
                    calls = chain.calls
                    puts = chain.puts

                    call_vol = calls['volume'].fillna(0).sum()
                    put_vol = puts['volume'].fillna(0).sum()
                    total_call_vol += call_vol
                    total_put_vol += put_vol

                    # Flag unusual volume: OTM calls with high vol
                    price = t.info.get('currentPrice') or t.info.get('regularMarketPrice', 0)
                    if price and price > 0:
                        otm_calls = calls[calls['strike'] > price * 1.05]
                        if not otm_calls.empty and otm_calls['volume'].mean() > 0:
                            high_vol_otm = otm_calls[
                                otm_calls['volume'] > otm_calls['volume'].mean() * 3
                            ]
                            if not high_vol_otm.empty:
                                unusual_strikes.append({
                                    'expiry': expiry,
                                    'strike': float(high_vol_otm['strike'].iloc[0]),
                                    'volume': int(high_vol_otm['volume'].iloc[0]),
                                    'type': 'OTM_call',
                                })
                except Exception:
                    continue

            if total_call_vol + total_put_vol == 0:
                continue

            pc_ratio = (total_put_vol / total_call_vol
                        if total_call_vol > 0 else None)

            options_data[ticker] = {
                'call_volume': int(total_call_vol),
                'put_volume': int(total_put_vol),
                'put_call_ratio': round(pc_ratio, 3) if pc_ratio else None,
                'bullish_options': pc_ratio < 0.7 if pc_ratio else False,
                'bearish_options': pc_ratio > 1.5 if pc_ratio else False,
                'unusual_otm_calls': len(unusual_strikes) > 0,
                'unusual_strikes': unusual_strikes[:3],
            }
            time.sleep(0.3)

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f'  Options data error for {ticker}: {e}')

    result = {
        'updated': datetime.datetime.now(datetime.UTC).isoformat() + 'Z',
        'source': 'yfinance',
        'tickers': options_data,
    }
    save_json('options_flow.json', result)
    print(f'  Options data: {len(options_data)} tickers ({errors} errors)')


# ── 13F Institutional Ownership (yfinance) ────────────────────────────────────

def fetch_institutional_data(tickers):
    """Fetch institutional holder data for tickers via yfinance.

    Collects: number of institutional holders, total institutional ownership %,
    top holder names. Source: yfinance .institutional_holders property.
    Saves to data/institutional_data.json.
    """
    if not tickers:
        print('  No tickers for institutional data')
        return

    try:
        import yfinance as yf
    except ImportError:
        print('  yfinance not installed — skipping institutional data')
        return

    inst_data = {}
    errors = 0
    for ticker in tickers[:50]:  # limit to top 50 signal tickers
        try:
            t = yf.Ticker(ticker)
            holders = t.institutional_holders
            if holders is not None and not holders.empty:
                n_holders = len(holders)
                # pctHeld is fractional (0.08 = 8%)
                total_pct = holders['pctHeld'].sum() if 'pctHeld' in holders.columns else None
                top_names = holders['Holder'].head(5).tolist() if 'Holder' in holders.columns else []
                inst_data[ticker] = {
                    'n_holders': n_holders,
                    'total_pct_held': round(float(total_pct), 4) if total_pct is not None else None,
                    'top_holders': top_names,
                }
            time.sleep(0.3)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f'  Institutional data error for {ticker}: {e}')

    result = {
        'updated': datetime.datetime.now(datetime.UTC).isoformat() + 'Z',
        'source': 'yfinance',
        'tickers': inst_data,
    }
    save_json('institutional_data.json', result)
    print(f'  Institutional data: {len(inst_data)} tickers ({errors} errors)')


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

    market = {'updated': datetime.datetime.now(datetime.UTC).isoformat() + 'Z', 'source': 'FRED'}

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
def _quiet_run(fn, *args, **kwargs):
    """Run fn with stdout captured to log file. Returns (result, captured_output)."""
    import io, contextlib
    buf = io.StringIO()
    result = None
    try:
        with contextlib.redirect_stdout(buf):
            result = fn(*args, **kwargs)
    except Exception as e:
        buf.write(f'ERROR: {e}\n')
        raise
    finally:
        output = buf.getvalue()
        if output.strip():
            log_path = os.path.join(LOG_DIR, 'fetch_verbose.log')
            with open(log_path, 'a') as f:
                f.write(f'\n--- {fn.__name__} {datetime.datetime.now(datetime.UTC).isoformat()} ---\n')
                f.write(output)
    return result


def main():
    t0 = time.time()
    verbose = '--verbose' in sys.argv
    now_str = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M")
    print(f'=== FETCH {now_str} UTC ===')

    # ── Market context (VIX + 10yr yield from FRED) ────────────────────────
    try:
        _quiet_run(fetch_market_data)
        # Read back VIX/Treasury from saved file
        mkt_path = os.path.join(DATA_DIR, 'market_data.json')
        vix_str = treasury_str = '?'
        if os.path.exists(mkt_path):
            with open(mkt_path) as f:
                mkt = json.load(f)
            vix_str = str(round(mkt.get('vix', {}).get('value', 0), 2))
            treasury_str = str(round(mkt.get('treasury_10y', {}).get('value', 0), 2))
        print(f'[FRED]      VIX: {vix_str}  Treasury: {treasury_str}  ✓')
    except Exception as e:
        print(f'[FRED]      ✗ {e}')

    # ── EDGAR Form 4 feed ──────────────────────────────────────────────────
    try:
        filings = _quiet_run(fetch_edgar_form4, days=90, max_results=1500, window_days=7)
        n_fetched = len(filings) if filings else 0
        filings = _quiet_run(enrich_form4_xml, filings)
        n_enriched = len(filings) if filings else 0
        _quiet_run(save_json, 'edgar_feed.json', {
            'updated': datetime.datetime.now(datetime.UTC).isoformat() + 'Z',
            'count': n_enriched,
            'source': 'SEC EDGAR EFTS + Form 4 XML',
            'filings': filings,
        })
        print(f'[EDGAR]     {n_fetched} fetched  |  {n_enriched} enriched  ✓')
    except Exception as e:
        print(f'[EDGAR]     ✗ {e}')

    # ── Congressional trades ───────────────────────────────────────────────
    congress_trades = []
    congress_source = []
    fmp_pages = 30 if '--bootstrap' in sys.argv else 10

    if FMP_API_KEY:
        try:
            fmp_trades = _quiet_run(fetch_fmp_congress, FMP_API_KEY, pages=fmp_pages)
            if fmp_trades:
                _quiet_run(save_json, 'fmp_congress_feed.json', {
                    'updated': datetime.datetime.now(datetime.UTC).isoformat() + 'Z',
                    'count': len(fmp_trades), 'source': 'FMP', 'trades': fmp_trades,
                })
                congress_trades.extend(fmp_trades)
                congress_source.append('FMP')
        except Exception as e:
            print(f'[CONGRESS]  FMP error: {e}')

    # House scraper
    try:
        house_trades = _quiet_run(fetch_house_disclosures)
        if house_trades:
            congress_source.append('House')
            congress_trades.extend(house_trades)
    except Exception:
        pass

    # Senate scraper
    try:
        senate_trades = _quiet_run(fetch_senate_disclosures)
        if senate_trades:
            congress_source.append('Senate')
            congress_trades.extend(senate_trades)
    except Exception:
        pass

    # Merge and deduplicate
    n_purchases = 0
    if congress_trades:
        seen = {}
        for t in congress_trades:
            key = (t.get('Ticker', ''), t.get('TransactionDate', t.get('Date', '')),
                   t.get('Representative', ''), t.get('Transaction', ''))
            if key not in seen:
                seen[key] = t
            elif t.get('Source') in ('House', 'Senate') and seen[key].get('Source') not in ('House', 'Senate'):
                seen[key] = t
        merged = sorted(seen.values(), key=lambda x: x.get('TransactionDate', x.get('Date', '')), reverse=True)
        n_purchases = sum(1 for t in merged if 'purchase' in str(t.get('Transaction', '')).lower())
        source_label = ' + '.join(congress_source)
        _quiet_run(save_json, 'congress_feed.json', {
            'updated': datetime.datetime.now(datetime.UTC).isoformat() + 'Z',
            'count': len(merged), 'source': source_label, 'trades': merged,
        })
        status = '✓' if n_purchases > 0 else '⚠'
        print(f'[CONGRESS]  {len(merged)} fetched  |  {n_purchases} purchases  {status}')
    else:
        print('[CONGRESS]  0 fetched  ⚠')

    # ── FMP Insider trades ─────────────────────────────────────────────────
    if FMP_API_KEY:
        insider_pages = 30 if '--bootstrap' in sys.argv else 10
        try:
            insiders = _quiet_run(fetch_fmp_insiders, FMP_API_KEY, pages=insider_pages)
            if insiders:
                _quiet_run(save_json, 'insiders_feed.json', {
                    'updated': datetime.datetime.now(datetime.UTC).isoformat() + 'Z',
                    'count': len(insiders), 'source': 'FMP', 'trades': insiders,
                })
                print(f'[INSIDERS]  {len(insiders)} purchases  ✓')
            else:
                print('[INSIDERS]  0 returned  ⚠')
        except Exception as e:
            print(f'[INSIDERS]  ✗ {e}')

    # ── SEC ticker map ─────────────────────────────────────────────────────
    try:
        sec_url = 'https://www.sec.gov/files/company_tickers.json'
        r = requests.get(sec_url, headers={'User-Agent': USER_AGENT}, timeout=15)
        r.raise_for_status()
        raw = r.json()
        cik_best = {}
        for entry in raw.values():
            cik = entry['cik_str']
            ticker = entry['ticker']
            title = entry['title'].lower().strip()
            if cik not in cik_best or len(ticker) < len(cik_best[cik][1]):
                cik_best[cik] = (title, ticker)
        sec_map = {name: ticker for name, ticker in cik_best.values()}
        _quiet_run(save_json, 'sec_tickers.json', sec_map)
    except Exception:
        pass

    # ── Ticker universe for enrichment ─────────────────────────────────────
    signal_tickers = set()
    if congress_trades:
        for t in congress_trades:
            ticker = t.get('Ticker', '')
            if ticker and len(ticker) <= 5 and ticker not in SUPPRESSED_TICKERS:
                signal_tickers.add(ticker)
    try:
        brain_path = os.path.join(DATA_DIR, 'brain_signals.json')
        if os.path.exists(brain_path):
            with open(brain_path) as f:
                brain = json.load(f)
            for s in brain.get('signals', []):
                t = s.get('ticker', '')
                if t and t not in SUPPRESSED_TICKERS:
                    signal_tickers.add(t)
    except Exception:
        pass

    tickers_list = sorted(signal_tickers)[:100] if signal_tickers else []
    n_tickers = len(tickers_list)

    if tickers_list:
        enrichment_sources = [
            ('VOLUME',    fetch_volume_data,        tickers_list),
            ('ANALYST',   fetch_analyst_data,        tickers_list),
            ('EARNINGS',  fetch_earnings_surprise,   tickers_list),
            ('SENTIMENT', fetch_news_sentiment,      tickers_list),
            ('SHORT',     fetch_short_interest,      tickers_list),
            ('INST',      fetch_institutional_data,  tickers_list),
            ('OPTIONS',   fetch_options_data,        tickers_list),
        ]
        for label, fn, args in enrichment_sources:
            try:
                _quiet_run(fn, args)
                print(f'[{label:10s}] {n_tickers} tickers  ✓')
            except Exception as e:
                print(f'[{label:10s}] ✗ {e}')

    # ── Committee membership ───────────────────────────────────────────────
    try:
        _quiet_run(fetch_committee_data)
    except Exception:
        pass

    # ── Quiver Quant (Mondays only) ────────────────────────────────────────
    if datetime.date.today().weekday() == 0 or '--force-quiver' in sys.argv:
        if tickers_list:
            try:
                _quiet_run(fetch_lobbying_data, tickers_list)
            except Exception:
                pass
        try:
            _quiet_run(fetch_quiver_committee_data)
        except Exception:
            pass

    elapsed = time.time() - t0
    print(f'=== DONE ({elapsed:.0f}s) ===')


if __name__ == '__main__':
    main()
