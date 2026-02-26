import pytest
from backtest.run_event_study import (
    get_forward_return,
    compute_car,
    compute_member_track_records,
    score_congress_aggregate,
    score_edgar_aggregate,
)
from backtest.shared import DEFAULT_WEIGHTS


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


def test_score_congress_aggregate_single_trade():
    """Single trade should produce base points with decay."""
    trades = [
        {"Range": "$50,001 - $100,000", "TransactionDate": "2026-02-25",
         "Representative": "Alice"},
    ]
    result = score_congress_aggregate(trades, DEFAULT_WEIGHTS, track_records={})
    assert result['count'] == 1
    assert result['score'] > 0
    assert result['score'] <= 40
    assert result['cluster_bonus'] == 0  # only 1 trade, no cluster


def test_score_congress_aggregate_cluster():
    """3+ trades for same ticker should get cluster bonus."""
    trades = [
        {"Range": "$15,001 - $50,000", "TransactionDate": "2026-02-20", "Representative": "A"},
        {"Range": "$15,001 - $50,000", "TransactionDate": "2026-02-22", "Representative": "B"},
        {"Range": "$15,001 - $50,000", "TransactionDate": "2026-02-24", "Representative": "C"},
    ]
    result = score_congress_aggregate(trades, DEFAULT_WEIGHTS, track_records={})
    assert result['count'] == 3
    assert result['cluster_bonus'] == 15
    assert result['score'] > 0
    assert len(result['members']) == 3


def test_score_congress_aggregate_capped_at_40():
    """Congress score should be capped at 40."""
    trades = [
        {"Range": "$1,000,001 - $5,000,000", "TransactionDate": "2026-02-25", "Representative": "A"},
        {"Range": "$1,000,001 - $5,000,000", "TransactionDate": "2026-02-25", "Representative": "B"},
        {"Range": "$1,000,001 - $5,000,000", "TransactionDate": "2026-02-25", "Representative": "C"},
    ]
    result = score_congress_aggregate(trades, DEFAULT_WEIGHTS, track_records={})
    assert result['score'] == 40  # 3×15 + 15 cluster = 60, but capped at 40


def test_score_edgar_aggregate_basic():
    """Single filing should produce base score without cluster bonus."""
    filings = [{"company": "NVIDIA", "date": "2026-02-20"}]
    result = score_edgar_aggregate(filings, DEFAULT_WEIGHTS)
    assert result['count'] == 1
    assert result['score'] == 6  # 1 × 6pts
    assert result['cluster_bonus'] == 0


def test_score_edgar_aggregate_cluster():
    """3+ filings should get cluster bonus."""
    filings = [
        {"company": "NVIDIA", "date": "2026-02-20"},
        {"company": "NVIDIA", "date": "2026-02-21"},
        {"company": "NVIDIA", "date": "2026-02-22"},
    ]
    result = score_edgar_aggregate(filings, DEFAULT_WEIGHTS)
    assert result['count'] == 3
    assert result['cluster_bonus'] == 15
    assert result['score'] == min(3 * 6 + 15, 40)  # 18 + 15 = 33


def test_score_edgar_aggregate_capped_at_40():
    """EDGAR score should be capped at 40."""
    filings = [{"company": f"CO{i}", "date": "2026-02-20"} for i in range(10)]
    result = score_edgar_aggregate(filings, DEFAULT_WEIGHTS)
    assert result['score'] == 40  # 10×6=60 base (capped 25) + 15 cluster = 40
