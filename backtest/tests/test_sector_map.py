"""Tests for backtest/sector_map.py — sector mapping module."""

import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from backtest.sector_map import get_sector, build_sector_map


SAMPLE_MAP = {
    "AAPL": "Technology",
    "LMT": "Industrials",
    "NVDA": "Technology",
    "PFE": "Healthcare",
    "JPM": "Financial Services",
}


@pytest.fixture(autouse=True)
def reset_cache():
    """Reset the module-level sector cache before each test."""
    import backtest.sector_map as sm
    sm._sector_cache = None
    yield
    sm._sector_cache = None


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

    @patch('requests.get')
    def test_returns_dict_from_api(self, mock_get, tmp_path):
        """build_sector_map() returns a dict when API call succeeds."""
        import backtest.sector_map as sm
        sm.SECTOR_MAP_PATH = tmp_path / "sector_map.json"

        # Mock FMP API response
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"symbol": "AAPL", "sector": "Technology"},
            {"symbol": "LMT", "sector": "Industrials"},
            {"symbol": "PFE", "sector": "Healthcare"},
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = build_sector_map(api_key="test_key")

        assert isinstance(result, dict)
        assert result["AAPL"] == "Technology"
        assert result["LMT"] == "Industrials"
        assert result["PFE"] == "Healthcare"
        assert len(result) == 3

        # Verify it was saved to disk
        assert sm.SECTOR_MAP_PATH.exists()
        saved = json.loads(sm.SECTOR_MAP_PATH.read_text())
        assert saved["AAPL"] == "Technology"

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

    @patch('requests.get')
    def test_falls_back_on_api_error(self, mock_get, tmp_path):
        """build_sector_map() falls back to cache when API request fails."""
        import backtest.sector_map as sm

        # Write a cached map
        map_file = tmp_path / "sector_map.json"
        map_file.write_text(json.dumps(SAMPLE_MAP))
        sm.SECTOR_MAP_PATH = map_file

        # Make the API call raise an exception
        mock_get.side_effect = Exception("Connection refused")

        result = build_sector_map(api_key="bad_key")

        assert isinstance(result, dict)
        assert result["AAPL"] == "Technology"
        assert len(result) == len(SAMPLE_MAP)

    def test_returns_empty_dict_no_cache_no_key(self, tmp_path):
        """build_sector_map() returns empty dict when no API key and no cached file."""
        import backtest.sector_map as sm
        sm.SECTOR_MAP_PATH = tmp_path / "nonexistent_sector_map.json"

        result = build_sector_map(api_key=None)

        assert isinstance(result, dict)
        assert len(result) == 0
