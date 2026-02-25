import pytest
from backtest.shared import match_edgar_ticker, range_to_base_points, date_to_ts, ts_to_date


def test_match_edgar_ticker_nvidia():
    assert match_edgar_ticker("NVIDIA Corporation") == "NVDA"


def test_match_edgar_ticker_raytheon():
    assert match_edgar_ticker("Raytheon Technologies") == "RTX"


def test_match_edgar_ticker_no_match():
    assert match_edgar_ticker("Random Company XYZ") is None


def test_match_edgar_ticker_case_insensitive():
    assert match_edgar_ticker("PFIZER INC") == "PFE"


def test_range_to_base_points_small():
    assert range_to_base_points("$1,001 - $15,000") == 3


def test_range_to_base_points_medium():
    assert range_to_base_points("$15,001 - $50,000") == 5


def test_range_to_base_points_xl():
    assert range_to_base_points("$1,000,001 - $5,000,000") == 15


def test_range_to_base_points_empty():
    assert range_to_base_points("") == 3


def test_date_roundtrip():
    date = "2026-01-15"
    assert ts_to_date(date_to_ts(date)) == date
