"""Tests for FMP congressional data integration in scripts/fetch_data.py."""

import pytest
from unittest.mock import patch, MagicMock

import sys
import os

# Ensure scripts/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.fetch_data import normalize_fmp_congress, fetch_fmp_congress
from backtest.shared import FMP_CONGRESS_FEED


# ── Sample FMP API Response Fixtures ──────────────────────────────────────────

@pytest.fixture
def sample_senate_trade():
    """A typical FMP Senate trading response dict."""
    return {
        'transactionDate': '2026-02-10',
        'disclosureDate': '2026-02-15',
        'representative': 'Nancy Pelosi',
        'type': 'Purchase',
        'amount': '$1,000,001 - $5,000,000',
        'symbol': 'NVDA',
        'assetDescription': 'NVIDIA Corporation',
        'owner': 'Spouse',
        'party': 'Democrat',
    }


@pytest.fixture
def sample_house_trade():
    """A typical FMP House disclosure response dict."""
    return {
        'transactionDate': '2026-01-20',
        'disclosureDate': '2026-02-01',
        'representative': 'Dan Crenshaw',
        'type': 'Sale (Full)',
        'amount': '$15,001 - $50,000',
        'symbol': 'AAPL',
        'assetDescription': 'Apple Inc',
        'owner': 'Self',
        'party': 'Republican',
    }


# ── normalize_fmp_congress Tests ──────────────────────────────────────────────

class TestNormalizeFmpCongress:

    def test_normalizes_senate_trade_correctly(self, sample_senate_trade):
        """Should normalize all FMP fields to QuiverQuant-compatible format."""
        result = normalize_fmp_congress(sample_senate_trade, 'Senate')

        assert result is not None
        assert result['Ticker'] == 'NVDA'
        assert result['TransactionDate'] == '2026-02-10'
        assert result['Representative'] == 'Nancy Pelosi'
        assert result['Transaction'] == 'Purchase'
        assert result['Range'] == '$1,000,001 - $5,000,000'
        assert result['Chamber'] == 'Senate'
        assert result['Party'] == 'Democrat'
        assert result['DisclosureDate'] == '2026-02-15'
        assert result['Source'] == 'FMP'

    def test_normalizes_house_trade_correctly(self, sample_house_trade):
        """Should normalize House trade with Sale type."""
        result = normalize_fmp_congress(sample_house_trade, 'House')

        assert result is not None
        assert result['Ticker'] == 'AAPL'
        assert result['Chamber'] == 'House'
        assert result['Transaction'] == 'Sale'
        assert result['Party'] == 'Republican'

    def test_returns_none_when_symbol_empty(self):
        """Should return None when symbol is empty string."""
        raw = {
            'transactionDate': '2026-02-10',
            'representative': 'Test Rep',
            'type': 'Purchase',
            'symbol': '',
        }
        result = normalize_fmp_congress(raw, 'Senate')
        assert result is None

    def test_returns_none_when_symbol_missing(self):
        """Should return None when symbol key is missing entirely."""
        raw = {
            'transactionDate': '2026-02-10',
            'representative': 'Test Rep',
            'type': 'Purchase',
        }
        result = normalize_fmp_congress(raw, 'Senate')
        assert result is None

    def test_returns_none_when_symbol_too_long(self):
        """Should return None when symbol is >5 chars (likely invalid)."""
        raw = {
            'transactionDate': '2026-02-10',
            'representative': 'Test Rep',
            'type': 'Purchase',
            'symbol': 'TOOLONG',
        }
        result = normalize_fmp_congress(raw, 'Senate')
        assert result is None

    def test_accepts_5_char_symbol(self):
        """Should accept symbols with exactly 5 chars (e.g. GOOGL)."""
        raw = {
            'transactionDate': '2026-02-10',
            'representative': 'Test Rep',
            'type': 'Purchase',
            'symbol': 'GOOGL',
        }
        result = normalize_fmp_congress(raw, 'Senate')
        assert result is not None
        assert result['Ticker'] == 'GOOGL'

    def test_computes_disclosure_delay_correctly(self, sample_senate_trade):
        """Should compute disclosure delay as days between transaction and disclosure."""
        result = normalize_fmp_congress(sample_senate_trade, 'Senate')
        # 2026-02-15 minus 2026-02-10 = 5 days
        assert result['DisclosureDelay'] == 5

    def test_disclosure_delay_larger_gap(self):
        """Should correctly compute delay for longer gaps."""
        raw = {
            'transactionDate': '2026-01-01',
            'disclosureDate': '2026-02-15',
            'representative': 'Test Rep',
            'type': 'Purchase',
            'symbol': 'MSFT',
        }
        result = normalize_fmp_congress(raw, 'House')
        assert result['DisclosureDelay'] == 45

    def test_disclosure_delay_none_when_missing_dates(self):
        """Should set DisclosureDelay to None when dates are missing."""
        raw = {
            'transactionDate': '2026-02-10',
            'representative': 'Test Rep',
            'type': 'Purchase',
            'symbol': 'TSLA',
        }
        result = normalize_fmp_congress(raw, 'Senate')
        assert result['DisclosureDelay'] is None

    def test_disclosure_delay_none_when_invalid_date_format(self):
        """Should set DisclosureDelay to None when date format is unexpected."""
        raw = {
            'transactionDate': '02/10/2026',
            'disclosureDate': 'invalid-date',
            'representative': 'Test Rep',
            'type': 'Purchase',
            'symbol': 'TSLA',
        }
        result = normalize_fmp_congress(raw, 'Senate')
        assert result['DisclosureDelay'] is None

    def test_maps_purchase_type(self):
        """Should map 'Purchase' type correctly."""
        raw = {'type': 'Purchase', 'symbol': 'AAPL', 'transactionDate': '2026-01-01'}
        result = normalize_fmp_congress(raw, 'Senate')
        assert result['Transaction'] == 'Purchase'

    def test_maps_sale_full_type(self):
        """Should map 'Sale (Full)' to 'Sale'."""
        raw = {'type': 'Sale (Full)', 'symbol': 'AAPL', 'transactionDate': '2026-01-01'}
        result = normalize_fmp_congress(raw, 'Senate')
        assert result['Transaction'] == 'Sale'

    def test_maps_sale_partial_type(self):
        """Should map 'Sale (Partial)' to 'Sale'."""
        raw = {'type': 'Sale (Partial)', 'symbol': 'AAPL', 'transactionDate': '2026-01-01'}
        result = normalize_fmp_congress(raw, 'Senate')
        assert result['Transaction'] == 'Sale'

    def test_maps_unknown_type(self):
        """Should pass through unknown transaction types."""
        raw = {'type': 'Exchange', 'symbol': 'AAPL', 'transactionDate': '2026-01-01'}
        result = normalize_fmp_congress(raw, 'Senate')
        assert result['Transaction'] == 'Exchange'

    def test_missing_fields_defaults(self):
        """Should handle missing optional fields gracefully with empty strings."""
        raw = {'symbol': 'TSLA'}
        result = normalize_fmp_congress(raw, 'House')
        assert result is not None
        assert result['Ticker'] == 'TSLA'
        assert result['Representative'] == ''
        assert result['TransactionDate'] == ''
        assert result['Range'] == ''
        assert result['Party'] == ''
        assert result['Transaction'] == 'Unknown'

    def test_symbol_normalized_to_uppercase(self):
        """Should uppercase the ticker symbol."""
        raw = {'symbol': 'nvda', 'transactionDate': '2026-01-01'}
        result = normalize_fmp_congress(raw, 'Senate')
        assert result['Ticker'] == 'NVDA'

    def test_symbol_whitespace_stripped(self):
        """Should strip whitespace from symbol."""
        raw = {'symbol': '  AAPL  ', 'transactionDate': '2026-01-01'}
        result = normalize_fmp_congress(raw, 'Senate')
        assert result['Ticker'] == 'AAPL'


# ── fetch_fmp_congress Tests ──────────────────────────────────────────────────

class TestFetchFmpCongress:

    def _mock_response(self, data, status_code=200):
        """Create a mock requests.Response."""
        mock = MagicMock()
        mock.ok = status_code == 200
        mock.status_code = status_code
        mock.json.return_value = data
        return mock

    @patch('scripts.fetch_data.time.sleep')
    @patch('scripts.fetch_data.requests.get')
    def test_fetches_both_senate_and_house(self, mock_get, mock_sleep):
        """Should call both Senate and House endpoints."""
        senate_data = [
            {'symbol': 'NVDA', 'transactionDate': '2026-02-10',
             'representative': 'Rep A', 'type': 'Purchase'},
        ]
        house_data = [
            {'symbol': 'AAPL', 'transactionDate': '2026-02-11',
             'representative': 'Rep B', 'type': 'Sale'},
        ]
        # First call = senate page 0, second = senate page 1 (empty),
        # third = house page 0, fourth = house page 1 (empty)
        mock_get.side_effect = [
            self._mock_response(senate_data),
            self._mock_response([]),
            self._mock_response(house_data),
            self._mock_response([]),
        ]

        result = fetch_fmp_congress('test_key', pages=2)

        assert len(result) == 2
        tickers = {t['Ticker'] for t in result}
        assert 'NVDA' in tickers
        assert 'AAPL' in tickers

        # Verify API URLs contain the right endpoints
        calls = mock_get.call_args_list
        assert any('senate-latest' in str(c) for c in calls)
        assert any('house-latest' in str(c) for c in calls)

    @patch('scripts.fetch_data.time.sleep')
    @patch('scripts.fetch_data.requests.get')
    def test_deduplicates_by_ticker_date_rep(self, mock_get, mock_sleep):
        """Should deduplicate trades by (ticker, date, representative)."""
        duplicate_data = [
            {'symbol': 'NVDA', 'transactionDate': '2026-02-10',
             'representative': 'Rep A', 'type': 'Purchase'},
            {'symbol': 'NVDA', 'transactionDate': '2026-02-10',
             'representative': 'Rep A', 'type': 'Purchase'},
        ]
        mock_get.side_effect = [
            self._mock_response(duplicate_data),
            self._mock_response([]),
            self._mock_response([]),
        ]

        result = fetch_fmp_congress('test_key', pages=1)
        assert len(result) == 1

    @patch('scripts.fetch_data.time.sleep')
    @patch('scripts.fetch_data.requests.get')
    def test_sorts_most_recent_first(self, mock_get, mock_sleep):
        """Should return trades sorted by date, most recent first."""
        data = [
            {'symbol': 'AAPL', 'transactionDate': '2026-01-01',
             'representative': 'Rep A', 'type': 'Purchase'},
            {'symbol': 'NVDA', 'transactionDate': '2026-02-15',
             'representative': 'Rep B', 'type': 'Purchase'},
            {'symbol': 'TSLA', 'transactionDate': '2026-01-20',
             'representative': 'Rep C', 'type': 'Sale'},
        ]
        mock_get.side_effect = [
            self._mock_response(data),
            self._mock_response([]),
            self._mock_response([]),
        ]

        result = fetch_fmp_congress('test_key', pages=1)
        dates = [t['TransactionDate'] for t in result]
        assert dates == sorted(dates, reverse=True)

    @patch('scripts.fetch_data.time.sleep')
    @patch('scripts.fetch_data.requests.get')
    def test_handles_http_error(self, mock_get, mock_sleep):
        """Should handle HTTP errors gracefully (break inner loop, continue)."""
        mock_get.return_value = self._mock_response([], status_code=403)

        result = fetch_fmp_congress('bad_key', pages=2)
        # Should return empty list, not raise
        assert result == []

    @patch('scripts.fetch_data.time.sleep')
    @patch('scripts.fetch_data.requests.get')
    def test_handles_request_exception(self, mock_get, mock_sleep):
        """Should handle request exceptions gracefully."""
        mock_get.side_effect = Exception('Connection timeout')

        result = fetch_fmp_congress('test_key', pages=1)
        assert result == []

    @patch('scripts.fetch_data.time.sleep')
    @patch('scripts.fetch_data.requests.get')
    def test_filters_invalid_symbols(self, mock_get, mock_sleep):
        """Should skip trades where normalize returns None (invalid symbols)."""
        data = [
            {'symbol': 'NVDA', 'transactionDate': '2026-02-10',
             'representative': 'Rep A', 'type': 'Purchase'},
            {'symbol': '', 'transactionDate': '2026-02-10',
             'representative': 'Rep B', 'type': 'Purchase'},
            {'symbol': 'TOOLONG', 'transactionDate': '2026-02-10',
             'representative': 'Rep C', 'type': 'Purchase'},
        ]
        mock_get.side_effect = [
            self._mock_response(data),
            self._mock_response([]),
            self._mock_response([]),
        ]

        result = fetch_fmp_congress('test_key', pages=1)
        assert len(result) == 1
        assert result[0]['Ticker'] == 'NVDA'

    @patch('scripts.fetch_data.time.sleep')
    @patch('scripts.fetch_data.requests.get')
    def test_paginates_correctly(self, mock_get, mock_sleep):
        """Should paginate through multiple pages."""
        page0 = [
            {'symbol': 'NVDA', 'transactionDate': '2026-02-10',
             'representative': 'Rep A', 'type': 'Purchase'},
        ]
        page1 = [
            {'symbol': 'AAPL', 'transactionDate': '2026-02-11',
             'representative': 'Rep B', 'type': 'Sale'},
        ]
        mock_get.side_effect = [
            # Senate pages
            self._mock_response(page0),
            self._mock_response(page1),
            self._mock_response([]),  # page 2 empty
            # House pages
            self._mock_response([]),
        ]

        result = fetch_fmp_congress('test_key', pages=3)
        assert len(result) == 2

    @patch('scripts.fetch_data.time.sleep')
    @patch('scripts.fetch_data.requests.get')
    def test_api_key_in_url(self, mock_get, mock_sleep):
        """Should include the API key in request URLs."""
        mock_get.return_value = self._mock_response([])

        fetch_fmp_congress('my_secret_key', pages=1)

        for call in mock_get.call_args_list:
            url = call[0][0] if call[0] else call[1].get('url', '')
            assert 'apikey=my_secret_key' in url


# ── FMP_CONGRESS_FEED Constant Test ───────────────────────────────────────────

class TestFmpCongressFeedConstant:

    def test_fmp_congress_feed_path_exists(self):
        """FMP_CONGRESS_FEED should be a Path pointing to data/fmp_congress_feed.json."""
        assert str(FMP_CONGRESS_FEED).endswith('data/fmp_congress_feed.json')
