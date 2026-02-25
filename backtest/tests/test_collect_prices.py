import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from backtest.collect_prices import extract_tickers, merge_candles, build_price_index


def test_extract_tickers_from_congress():
    """Should extract unique ticker symbols from congress purchase trades."""
    trades = [
        {"Ticker": "AAPL", "Transaction": "Purchase"},
        {"Ticker": "AAPL", "Transaction": "Purchase"},  # duplicate
        {"Ticker": "GOOGL", "Transaction": "Sale"},
        {"Ticker": "MSFT", "Transaction": "Purchase"},
        {"Ticker": "TOOLONGNAME", "Transaction": "Purchase"},  # > 5 chars, skip
    ]
    tickers = extract_tickers(trades, source="congress")
    assert "AAPL" in tickers
    assert "MSFT" in tickers
    assert "GOOGL" in tickers  # include sales too — need their history
    assert "TOOLONGNAME" not in tickers
    assert len(tickers) == len(set(tickers))  # no duplicates


def test_merge_candles_appends_new_dates():
    """New candle data should be merged into existing, no duplicates."""
    existing = {"2026-01-01": {"o": 100, "h": 105, "l": 98, "c": 103, "v": 1000}}
    new_candles = {
        "2026-01-01": {"o": 100, "h": 105, "l": 98, "c": 103, "v": 1000},  # dup
        "2026-01-02": {"o": 103, "h": 107, "l": 101, "c": 106, "v": 1200},
    }
    merged = merge_candles(existing, new_candles)
    assert len(merged) == 2
    assert "2026-01-02" in merged


def test_build_price_index_returns_close_by_date():
    """build_price_index should return {date: close_price} dict."""
    candles = {
        "2026-01-01": {"o": 100, "h": 105, "l": 98, "c": 103, "v": 1000},
        "2026-01-02": {"o": 103, "h": 107, "l": 101, "c": 106, "v": 1200},
    }
    index = build_price_index(candles)
    assert index["2026-01-01"] == 103
    assert index["2026-01-02"] == 106
