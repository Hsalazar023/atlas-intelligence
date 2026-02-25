import pytest
from backtest.run_event_study import (
    get_forward_return,
    compute_car,
    compute_member_track_records,
    score_congress_event,
    classify_convergence,
)


# Sample price index: date -> close price
PRICES = {
    "2026-01-10": 100.0,
    "2026-01-11": 100.5,
    "2026-01-12": 101.0,
    "2026-01-13": 102.0,
    "2026-01-14": 103.0,
    "2026-01-15": 104.0,
    "2026-02-09": 110.0,   # ~30 days after 2026-01-10
    "2026-04-10": 115.0,   # ~90 days after 2026-01-10
}

SPY_PRICES = {
    "2026-01-10": 500.0,
    "2026-01-14": 502.5,   # +0.5% in 5 days
    "2026-02-09": 510.0,   # +2% in 30 days
    "2026-04-10": 520.0,   # +4% in 90 days
}


def test_get_forward_return_5d():
    """Should find the closest available price ~5 days after event."""
    ret = get_forward_return(PRICES, "2026-01-10", days=5)
    assert ret is not None
    assert abs(ret - 0.04) < 0.01  # ~4% return (100 → 104)


def test_get_forward_return_missing_date():
    """Should return None if no price data exists within tolerance window."""
    sparse_prices = {"2026-01-01": 100.0}
    ret = get_forward_return(sparse_prices, "2026-01-10", days=5)
    assert ret is None


def test_compute_car():
    """CAR = stock return - benchmark return for same period."""
    stock_ret = 0.05   # +5%
    bench_ret = 0.02   # +2%
    car = compute_car(stock_ret, bench_ret)
    assert abs(car - 0.03) < 1e-6  # 3% alpha


def test_compute_member_track_records():
    """Should compute avg ExcessReturn and win_rate per member."""
    trades = [
        {"Representative": "Alice", "Transaction": "Purchase", "ExcessReturn": 5.0},
        {"Representative": "Alice", "Transaction": "Purchase", "ExcessReturn": -2.0},
        {"Representative": "Bob",   "Transaction": "Purchase", "ExcessReturn": 10.0},
        {"Representative": "Alice", "Transaction": "Sale",     "ExcessReturn": -1.0},  # skip (sale)
    ]
    records = compute_member_track_records(trades)
    assert "Alice" in records
    alice = records["Alice"]
    assert abs(alice["avg_excess"] - 1.5) < 0.01  # mean(5, -2) = 1.5
    assert abs(alice["win_rate"] - 0.5) < 0.01    # 1 of 2 were positive
    assert records["Bob"]["avg_excess"] == 10.0


def test_score_congress_event_basic():
    """Should assign base points based on Range."""
    from backtest.shared import DEFAULT_WEIGHTS
    event = {
        "Ticker": "AAPL",
        "Transaction": "Purchase",
        "TransactionDate": "2026-01-10",
        "Range": "$50,001 - $100,000",
        "Representative": "Alice",
        "ExcessReturn": 5.0,
    }
    score = score_congress_event(event, DEFAULT_WEIGHTS, track_records={}, event_date="2026-01-10")
    assert score > 0
    assert score <= 40  # capped


def test_classify_convergence_both():
    """Tickers with both congress and edgar activity = convergence."""
    congress_tickers = {"AAPL", "MSFT"}
    edgar_tickers = {"AAPL", "GOOGL"}
    assert classify_convergence("AAPL", congress_tickers, edgar_tickers) == "convergence"
    assert classify_convergence("MSFT", congress_tickers, edgar_tickers) == "congress"
    assert classify_convergence("GOOGL", congress_tickers, edgar_tickers) == "edgar"
    assert classify_convergence("TSLA", congress_tickers, edgar_tickers) == "none"
