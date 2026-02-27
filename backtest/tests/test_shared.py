import pytest
from backtest.shared import match_edgar_ticker, range_to_base_points, date_to_ts, ts_to_date


# ── SEC Ticker Matching Tests ────────────────────────────────────────────────

def test_match_edgar_ticker_nvidia():
    assert match_edgar_ticker("NVIDIA Corporation") == "NVDA"


def test_match_edgar_ticker_nvidia_sec_exact():
    """SEC map has 'nvidia corp' as exact entry."""
    assert match_edgar_ticker("NVIDIA CORP") == "NVDA"


def test_match_edgar_ticker_raytheon():
    assert match_edgar_ticker("Raytheon Technologies") == "RTX"


def test_match_edgar_ticker_no_match():
    assert match_edgar_ticker("Random Company XYZ") is None


def test_match_edgar_ticker_case_insensitive():
    assert match_edgar_ticker("PFIZER INC") == "PFE"


def test_match_edgar_ticker_empty():
    assert match_edgar_ticker("") is None


def test_match_edgar_ticker_with_state_suffix():
    """Companies with /DE/ or /NJ suffixes should still match."""
    result = match_edgar_ticker("DANAHER CORP /DE/")
    assert result == "DHR"


def test_match_edgar_ticker_servicenow():
    assert match_edgar_ticker("ServiceNow, Inc.") == "NOW"


def test_match_edgar_ticker_ibm():
    assert match_edgar_ticker("INTERNATIONAL BUSINESS MACHINES CORP") == "IBM"


def test_match_edgar_ticker_colgate():
    assert match_edgar_ticker("COLGATE PALMOLIVE CO") == "CL"


def test_match_edgar_ticker_fallback_keywords():
    """When SEC map is empty, fallback keywords should still work."""
    import backtest.shared as shared
    saved = shared._sec_ticker_map
    try:
        shared._sec_ticker_map = {}  # force empty
        assert match_edgar_ticker("nvidia corp") == "NVDA"
        assert match_edgar_ticker("Raytheon Technologies") == "RTX"
    finally:
        shared._sec_ticker_map = saved


# ── Range to Base Points Tests ───────────────────────────────────────────────

def test_range_to_base_points_small():
    assert range_to_base_points("$1,001 - $15,000") == 3


def test_range_to_base_points_medium():
    assert range_to_base_points("$15,001 - $50,000") == 5


def test_range_to_base_points_xl():
    assert range_to_base_points("$1,000,001 - $5,000,000") == 15


def test_range_to_base_points_empty():
    assert range_to_base_points("") == 3


# ── Date Utilities ───────────────────────────────────────────────────────────

def test_date_roundtrip():
    date = "2026-01-15"
    assert ts_to_date(date_to_ts(date)) == date
