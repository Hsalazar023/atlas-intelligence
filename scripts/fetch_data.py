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
                    print(f'  FMP {chamber} page {page} error: HTTP {r.status_code}')
                    break
                data = r.json()
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
    """Inner implementation — may raise; caller wraps in try/except."""
    today = datetime.date.today()
    cutoff = today - datetime.timedelta(days=lookback_days)
    headers = {'User-Agent': 'ATLAS Research Tool'}

    # Determine which years of XML index to fetch
    years = [today.year]
    if today.month <= 2:
        years.append(today.year - 1)

    # ── STEP 1: Fetch & parse PTR XML index ─────────────────────────────
    all_filings = []
    filings_parsed = 0
    filings_failed = 0

    for year in years:
        xml_url = f'https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}FD.xml'
        print(f'  Fetching House PTR index: {year}...')
        try:
            r = requests.get(xml_url, headers=headers, timeout=15)
            if not r.ok:
                print(f'  House XML index {year}: HTTP {r.status_code}')
                continue
        except Exception as e:
            print(f'  House XML index {year} fetch error: {e}')
            continue

        time.sleep(1)  # rate limit

        # ── STEP 2: Parse XML, filter to recent filings ─────────────────
        try:
            root = ET.fromstring(r.content)
        except ET.ParseError as e:
            print(f'  House XML parse error for {year}: {e}')
            continue

        # The XML structure: <FinancialDisclosure> → <Member> elements
        members = root.findall('.//Member')
        if not members:
            # Try alternate structure
            members = root.findall('.//')

        for member in root.iter('Member'):
            filing_date_str = ''
            doc_id = ''
            member_name = ''

            # Extract fields — try common element names
            for child in member:
                tag = child.tag.lower() if child.tag else ''
                text = (child.text or '').strip()
                if 'filingdate' in tag or tag == 'filing_date':
                    filing_date_str = text
                elif 'docid' in tag or tag == 'doc_id':
                    doc_id = text
                elif tag in ('prefix', 'last', 'first', 'suffix'):
                    # Build name from parts
                    if tag == 'last':
                        member_name = text + (', ' + member_name if member_name else '')
                    elif tag == 'first':
                        member_name = (member_name + ' ' if member_name else '') + text
                elif 'name' in tag or tag == 'membername':
                    member_name = text
                elif tag == 'filingtype' or 'type' in tag:
                    # Only process PTR filings
                    if text and 'ptr' not in text.lower() and 'transaction' not in text.lower():
                        continue

            # Also try attributes
            if not doc_id:
                doc_id = member.get('DocID', member.get('docid', ''))
            if not filing_date_str:
                filing_date_str = member.get('FilingDate', member.get('filing_date', ''))

            if not doc_id or not filing_date_str:
                continue

            # Parse filing date
            filing_date = _parse_date_flex(filing_date_str)
            if not filing_date or filing_date < cutoff:
                continue

            # Build name from child elements if not found yet
            if not member_name:
                last = _get_child_text(member, 'Last')
                first = _get_child_text(member, 'First')
                if last and first:
                    member_name = f'{first} {last}'
                elif last:
                    member_name = last

            all_filings.append({
                'doc_id': doc_id,
                'member_name': _clean_member_name(member_name),
                'filing_date': filing_date.isoformat(),
            })

    print(f'  House PTR index: {len(all_filings)} recent filings found (last {lookback_days}d)')
    if not all_filings:
        return []

    # ── STEP 3: Fetch individual filing pages for trade details ──────
    all_trades = []
    for filing in all_filings:
        doc_id = filing['doc_id']
        # Try the HTML report URL
        doc_url = f'https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{doc_id}.pdf'
        # Also try the structured HTML view
        html_url = f'https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{doc_id}.htm'

        trades_from_filing = []
        try:
            # Try HTML first (more parseable than PDF)
            r = requests.get(html_url, headers=headers, timeout=15)
            if r.ok and len(r.text) > 200:
                trades_from_filing = _parse_house_filing_page(
                    r.text, filing['member_name'], filing['filing_date']
                )
                filings_parsed += 1
            else:
                # If HTML fails, we can't parse PDF — log and skip
                filings_failed += 1
        except Exception as e:
            print(f'  Filing {doc_id} parse error: {e}')
            filings_failed += 1

        all_trades.extend(trades_from_filing)
        time.sleep(1)  # rate limit: 1 req/sec for government sites

    # ── STEP 5: Return normalized trades ────────────────────────────────
    # Sort most recent first
    all_trades.sort(key=lambda x: x.get('TransactionDate', ''), reverse=True)

    print(f'  House scraper: {len(all_trades)} trades extracted '
          f'({filings_parsed} filings parsed, {filings_failed} failed)')

    return all_trades


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


# ── FMP Insider Trades ─────────────────────────────────────────────────────
def fetch_fmp_insiders(api_key, pages=10):
    """
    Fetch recent insider trades from FMP (Financial Modeling Prep).
    Uses the stable API insider-trading endpoint with pagination.
    Filters to purchases only (P-Purchase) to match Brain's buy-signal focus.
    """
    all_trades = []
    seen = set()  # (symbol, date, reportingName) for dedup

    for page in range(pages):
        url = (
            f'https://financialmodelingprep.com/stable/insider-trading'
            f'?transactionType=P-Purchase&page={page}&apikey={api_key}'
        )
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

                txn_date = raw.get('transactionDate', '')
                reporter = raw.get('reportingName', '')
                dedup_key = (symbol, txn_date, reporter)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                shares = raw.get('securitiesTransacted', 0) or 0
                owned = raw.get('securitiesOwned', 0) or 0
                txn_type = raw.get('transactionType', '')

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
    print(f'  FMP total: {len(all_trades)} insider purchases ({len(seen)} deduped)')
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
def main():
    print(f'\n=== ATLAS Data Fetcher — {datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M")} UTC ===\n')

    # ── Market context (VIX + 10yr yield from FRED) ────────────────────────
    print('Fetching market context data (FRED)...')
    fetch_market_data()

    # ── EDGAR Form 4 feed ──────────────────────────────────────────────────
    print('Fetching EDGAR Form 4 filings (last 90 days, weekly windows)...')
    try:
        filings = fetch_edgar_form4(days=90, max_results=1500, window_days=7)
        print(f'  {len(filings)} filings from EFTS index (across {90 // 7} windows)')

        # Enrich with Form 4 XML data (ticker, role, amounts, buy/sell)
        print('  Enriching filings from Form 4 XML (~25s per 200 filings)...')
        filings = enrich_form4_xml(filings)

        save_json('edgar_feed.json', {
            'updated': datetime.datetime.now(datetime.UTC).isoformat() + 'Z',
            'count': len(filings),
            'source': 'SEC EDGAR EFTS + Form 4 XML',
            'filings': filings,
        })
        print(f'  {len(filings)} enriched Form 4 filings saved')
    except Exception as e:
        import traceback
        print(f'  EDGAR error: {e}')
        traceback.print_exc()

    # ── Congressional trades (FMP) ────────────────────────────────────────
    congress_trades = []
    congress_source = []

    # FMP congressional trades (primary source if API key set)
    # --bootstrap flag fetches 30 pages per chamber (~6000 trades, back to ~2022)
    fmp_pages = 30 if '--bootstrap' in sys.argv else 10
    if FMP_API_KEY:
        print(f'\nFetching congressional trades (FMP, {fmp_pages} pages/chamber)...')
        try:
            fmp_trades = fetch_fmp_congress(FMP_API_KEY, pages=fmp_pages)
            if fmp_trades:
                # Save raw FMP data separately
                save_json('fmp_congress_feed.json', {
                    'updated': datetime.datetime.now(datetime.UTC).isoformat() + 'Z',
                    'count': len(fmp_trades),
                    'source': 'FMP',
                    'trades': fmp_trades,
                })
                print(f'  {len(fmp_trades)} FMP congressional trades saved')
                congress_trades.extend(fmp_trades)
                congress_source.append('FMP')
        except Exception as e:
            print(f'  FMP congress error: {e}')
    else:
        print('\nFMP_API_KEY not set — skipping FMP congressional trades')

    # House Financial Disclosures (direct scraper — no API key needed)
    house_trades = []
    try:
        print('\nFetching House financial disclosures (direct scraper)...')
        house_trades = fetch_house_disclosures()
        if house_trades:
            congress_source.append('House')
            # Count how many are genuinely new (not in FMP data)
            fmp_keys = set()
            for t in congress_trades:
                fmp_keys.add((
                    t.get('Ticker', ''),
                    t.get('TransactionDate', ''),
                    t.get('Representative', ''),
                    t.get('Transaction', ''),
                ))
            house_new = sum(
                1 for t in house_trades
                if (t['Ticker'], t['TransactionDate'], t['Representative'], t['Transaction']) not in fmp_keys
            )
            congress_trades.extend(house_trades)
            print(f'  House scraper: {len(house_trades)} total, {house_new} new (not in FMP)')
    except Exception as e:
        print(f'  House scraper failed, FMP-only fallback: {e}')

    # ── Insider trades (FMP) ─────────────────────────────────────────────
    if FMP_API_KEY:
        insider_pages = 30 if '--bootstrap' in sys.argv else 10
        print(f'\nFetching insider trades (FMP, {insider_pages} pages, purchases only)...')
        try:
            insiders = fetch_fmp_insiders(FMP_API_KEY, pages=insider_pages)
            if insiders:
                save_json('insiders_feed.json', {
                    'updated': datetime.datetime.now(datetime.UTC).isoformat() + 'Z',
                    'count': len(insiders),
                    'source': 'FMP',
                    'trades': insiders,
                })
                print(f'  {len(insiders)} insider purchase trades saved')
            else:
                print('  No insider trades returned from FMP')
        except Exception as e:
            print(f'  FMP insiders error: {e}')
    else:
        print('\nFMP_API_KEY not set — skipping insider trades')

    # Merge and deduplicate combined congressional trades
    if congress_trades:
        # Dedup by (Ticker, TransactionDate, Representative, Transaction)
        # When duplicate exists in both FMP + House, prefer House record (fresher)
        seen = {}  # key → trade dict
        for t in congress_trades:
            key = (
                t.get('Ticker', ''),
                t.get('TransactionDate', t.get('Date', '')),
                t.get('Representative', ''),
                t.get('Transaction', ''),
            )
            if key not in seen:
                seen[key] = t
            elif t.get('Source') == 'House' and seen[key].get('Source') != 'House':
                seen[key] = t  # prefer House record
        merged = list(seen.values())
        # Sort most recent first
        merged.sort(
            key=lambda x: x.get('TransactionDate', x.get('Date', '')),
            reverse=True,
        )
        source_label = ' + '.join(congress_source)
        save_json('congress_feed.json', {
            'updated': datetime.datetime.now(datetime.UTC).isoformat() + 'Z',
            'count': len(merged),
            'source': source_label,
            'trades': merged,
        })
        print(f'  Combined: {len(merged)} unique congressional trades ({source_label})')

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

    # ── Volume + Analyst data (yfinance) ─────────────────────────────────
    # Collect unique tickers from recent signals (congress + insiders)
    signal_tickers = set()
    if congress_trades:
        for t in congress_trades:
            ticker = t.get('Ticker', '')
            if ticker and len(ticker) <= 5:
                signal_tickers.add(ticker)
    # Also try loading existing brain signals for ticker list
    try:
        brain_path = os.path.join(DATA_DIR, 'brain_signals.json')
        if os.path.exists(brain_path):
            with open(brain_path) as f:
                brain = json.load(f)
            for s in brain.get('signals', []):
                if s.get('ticker'):
                    signal_tickers.add(s['ticker'])
    except Exception:
        pass

    if signal_tickers:
        tickers_list = sorted(signal_tickers)[:100]  # cap at 100 to limit API calls

        print(f'\nFetching volume data ({len(tickers_list)} tickers)...')
        try:
            fetch_volume_data(tickers_list)
        except Exception as e:
            print(f'  Volume data error (skipped): {e}')

        print(f'\nFetching analyst data ({len(tickers_list)} tickers)...')
        try:
            fetch_analyst_data(tickers_list)
        except Exception as e:
            print(f'  Analyst data error (skipped): {e}')

    print('\nDone.\n')


if __name__ == '__main__':
    main()
