"""Tests for backtest/sector_map.py — sector mapping module."""

import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from backtest.sector_map import (
    get_sector, build_sector_map, get_market_cap, get_market_cap_bucket,
)


SAMPLE_MAP = {
    "AAPL": "Technology",
    "LMT": "Industrials",
    "NVDA": "Technology",
    "PFE": "Healthcare",
    "JPM": "Financial Services",
}


@pytest.fixture(autouse=True)
def reset_cache():
    """Reset the module-level caches before each test."""
    import backtest.sector_map as sm
    sm._sector_cache = None
    sm._market_cap_cache = None
    yield
    sm._sector_cache = None
    sm._market_cap_cache = None


class TestGetSector:
    """Tests for get_sector() function."""

    @patch('backtest.sector_map._load_cache')
    def test_returns_correct_sector(self, mock_load):
        """get_sector() returns the correct sector for a known ticker."""
        import backtest.sector_map as sm
        sm._sector_cache = dict(SAMPLE_MAP)

        assert get_sector("AAPL") == "Technology"
        assert get_sector("LMT") == "Industrials"
        assert get_sector("PFE") == "Healthcare"

    @patch('backtest.sector_map._load_cache')
    def test_returns_none_for_unknown(self, mock_load):
        """get_sector() returns None for a ticker not in the map."""
        import backtest.sector_map as sm
        sm._sector_cache = dict(SAMPLE_MAP)

        assert get_sector("ZZZZZ") is None
        assert get_sector("FAKECO") is None

    @patch('backtest.sector_map._load_cache')
    def test_case_insensitive(self, mock_load):
        """get_sector() works with lowercase or mixed-case input."""
        import backtest.sector_map as sm
        sm._sector_cache = dict(SAMPLE_MAP)

        assert get_sector("aapl") == "Technology"
        assert get_sector("lmt") == "Industrials"
        assert get_sector("Nvda") == "Technology"
        assert get_sector("jPm") == "Financial Services"

    def test_lazy_loads_cache(self, tmp_path):
        """get_sector() lazy-loads cache from disk on first call."""
        import backtest.sector_map as sm

        # Write a sector map file
        map_file = tmp_path / "sector_map.json"
        map_file.write_text(json.dumps(SAMPLE_MAP))

        # Point the module to our temp file
        sm.SECTOR_MAP_PATH = map_file
        sm._sector_cache = None  # ensure not loaded

        result = get_sector("AAPL")
        assert result == "Technology"
        assert sm._sector_cache is not None
        assert len(sm._sector_cache) == len(SAMPLE_MAP)


class TestBuildSectorMap:
    """Tests for build_sector_map() function."""

    @patch('backtest.sector_map.time.sleep')
    @patch('requests.get')
    def test_returns_dict_from_api(self, mock_get, mock_sleep, tmp_path):
        """build_sector_map() returns a dict when API call succeeds."""
        import backtest.sector_map as sm
        sm.SECTOR_MAP_PATH = tmp_path / "sector_map.json"

        # Mock FMP per-ticker profile API responses (uses 'marketCap' key)
        def make_resp(symbol, sector, market_cap=None):
            resp = MagicMock()
            resp.ok = True
            profile = {"symbol": symbol, "sector": sector}
            if market_cap:
                profile["marketCap"] = market_cap
            resp.json.return_value = [profile]
            return resp

        tickers = ["AAPL", "LMT", "PFE"]
        mock_get.side_effect = [
            make_resp("AAPL", "Technology", 3_000_000_000_000),
            make_resp("LMT", "Industrials", 120_000_000_000),
            make_resp("PFE", "Healthcare", 150_000_000_000),
        ]

        sm.MARKET_CAP_MAP_PATH = tmp_path / "market_cap_map.json"
        result = build_sector_map(api_key="test_key", tickers=tickers)

        assert isinstance(result, dict)
        assert result["AAPL"] == "Technology"
        assert result["LMT"] == "Industrials"
        assert result["PFE"] == "Healthcare"
        assert len(result) == 3

        # Verify sector map saved to disk
        assert sm.SECTOR_MAP_PATH.exists()
        saved = json.loads(sm.SECTOR_MAP_PATH.read_text())
        assert saved["AAPL"] == "Technology"

        # Verify market cap map saved to disk
        assert sm.MARKET_CAP_MAP_PATH.exists()
        cap_saved = json.loads(sm.MARKET_CAP_MAP_PATH.read_text())
        assert cap_saved["AAPL"] == 3_000_000_000_000
        assert get_market_cap_bucket("AAPL") == "mega"
        assert get_market_cap_bucket("LMT") == "large"

    def test_falls_back_to_cached_file(self, tmp_path):
        """build_sector_map() loads cached file when API key is None."""
        import backtest.sector_map as sm

        # Write a cached map
        map_file = tmp_path / "sector_map.json"
        map_file.write_text(json.dumps(SAMPLE_MAP))
        sm.SECTOR_MAP_PATH = map_file

        result = build_sector_map(api_key=None)

        assert isinstance(result, dict)
        assert result["AAPL"] == "Technology"
        assert len(result) == len(SAMPLE_MAP)

    @patch('backtest.sector_map.time.sleep')
    @patch('requests.get')
    def test_falls_back_on_api_error(self, mock_get, mock_sleep, tmp_path):
        """build_sector_map() falls back to cache when API request fails."""
        import backtest.sector_map as sm

        # Write a cached map
        map_file = tmp_path / "sector_map.json"
        map_file.write_text(json.dumps(SAMPLE_MAP))
        sm.SECTOR_MAP_PATH = map_file

        # Make the API call raise an exception
        mock_get.side_effect = Exception("Connection refused")

        # Pass a ticker not in cache to force an API call
        result = build_sector_map(api_key="bad_key", tickers=["ZZZZZ"])

        assert isinstance(result, dict)
        # Should still have cached entries even though API failed
        assert result["AAPL"] == "Technology"
        assert len(result) == len(SAMPLE_MAP)

    def test_returns_empty_dict_no_cache_no_key(self, tmp_path):
        """build_sector_map() returns empty dict when no API key and no cached file."""
        import backtest.sector_map as sm
        sm.SECTOR_MAP_PATH = tmp_path / "nonexistent_sector_map.json"

        result = build_sector_map(api_key=None)

        assert isinstance(result, dict)
        assert len(result) == 0


class TestMarketCapBucket:
    """Tests for get_market_cap_bucket() and get_market_cap()."""

    def test_mega_cap(self):
        """Companies with >$200B market cap are 'mega'."""
        import backtest.sector_map as sm
        sm._market_cap_cache = {"AAPL": 3_000_000_000_000}
        assert get_market_cap_bucket("AAPL") == "mega"

    def test_large_cap(self):
        """Companies with $10B-$200B market cap are 'large'."""
        import backtest.sector_map as sm
        sm._market_cap_cache = {"CAT": 50_000_000_000}
        assert get_market_cap_bucket("CAT") == "large"

    def test_mid_cap(self):
        """Companies with $2B-$10B market cap are 'mid'."""
        import backtest.sector_map as sm
        sm._market_cap_cache = {"EPAC": 5_000_000_000}
        assert get_market_cap_bucket("EPAC") == "mid"

    def test_small_cap(self):
        """Companies with $300M-$2B market cap are 'small'."""
        import backtest.sector_map as sm
        sm._market_cap_cache = {"XYZ": 800_000_000}
        assert get_market_cap_bucket("XYZ") == "small"

    def test_micro_cap(self):
        """Companies with <$300M market cap are 'micro'."""
        import backtest.sector_map as sm
        sm._market_cap_cache = {"TINY": 100_000_000}
        assert get_market_cap_bucket("TINY") == "micro"

    def test_unknown_ticker(self):
        """Unknown tickers return None."""
        import backtest.sector_map as sm
        sm._market_cap_cache = {}
        assert get_market_cap_bucket("ZZZZ") is None

    def test_case_insensitive(self):
        """get_market_cap_bucket() works with lowercase input."""
        import backtest.sector_map as sm
        sm._market_cap_cache = {"AAPL": 3_000_000_000_000}
        assert get_market_cap_bucket("aapl") == "mega"

    def test_get_market_cap_returns_value(self):
        """get_market_cap() returns the raw market cap value."""
        import backtest.sector_map as sm
        sm._market_cap_cache = {"NVDA": 2_500_000_000_000}
        assert get_market_cap("NVDA") == 2_500_000_000_000

    def test_get_market_cap_unknown(self):
        """get_market_cap() returns None for unknown tickers."""
        import backtest.sector_map as sm
        sm._market_cap_cache = {}
        assert get_market_cap("UNKNOWN") is None
