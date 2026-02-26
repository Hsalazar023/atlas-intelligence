import pytest
import copy
from backtest.optimize_weights import (
    score_event_with_weights,
    evaluate_weights,
    find_optimal_threshold,
    grid_search,
)
from backtest.shared import DEFAULT_WEIGHTS

# Sample aggregate events (per-ticker, not per-trade)
EVENTS = [
    # Ticker with 1 large congress trade, no EDGAR — score ~15
    {"ticker": "AAPL", "congress_count": 1, "edgar_count": 0,
     "trade_ranges": ["$1,000,001 - $5,000,000"], "raw_points": [15.0],
     "convergence": "congress", "car_30d": 0.08},
    # Ticker with 1 small congress trade — score ~3
    {"ticker": "MSFT", "congress_count": 1, "edgar_count": 0,
     "trade_ranges": ["$1,001 - $15,000"], "raw_points": [3.0],
     "convergence": "congress", "car_30d": -0.02},
    # Ticker with congress + EDGAR convergence — score should include boost
    {"ticker": "NVDA", "congress_count": 2, "edgar_count": 2,
     "trade_ranges": ["$50,001 - $100,000", "$15,001 - $50,000"],
     "raw_points": [6.0, 5.0],
     "convergence": "both", "car_30d": 0.05},
    # Ticker with only EDGAR filings — score ~6
    {"ticker": "META", "congress_count": 0, "edgar_count": 1,
     "trade_ranges": [], "raw_points": [],
     "convergence": "edgar", "car_30d": 0.03},
]


def test_score_event_with_weights_congress():
    """Congress-only aggregate event should produce a positive score."""
    event = EVENTS[0]  # AAPL, 1 trade $1M+
    score = score_event_with_weights(event, DEFAULT_WEIGHTS)
    assert score == 15.0  # decayed_points=[15.0], no cluster, no EDGAR


def test_score_event_with_weights_convergence_bonus():
    """Convergence events (both hubs) should score higher than single-hub."""
    # NVDA has both congress and EDGAR
    conv_event = EVENTS[2]
    score_conv = score_event_with_weights(conv_event, DEFAULT_WEIGHTS)

    # Compare to same event but with 0 EDGAR
    single_event = copy.deepcopy(conv_event)
    single_event['edgar_count'] = 0
    score_single = score_event_with_weights(single_event, DEFAULT_WEIGHTS)

    assert score_conv > score_single
    assert score_conv - score_single >= 20  # convergence boost (20) + EDGAR score


def test_score_event_with_weights_edgar_cluster():
    """EDGAR events with 3+ filings should get cluster bonus."""
    event = {"congress_count": 0, "edgar_count": 3,
             "trade_ranges": [], "raw_points": []}
    score = score_event_with_weights(event, DEFAULT_WEIGHTS)
    # 3×6=18 base + 15 cluster = 33
    assert score == 33.0


def test_evaluate_weights_returns_metrics():
    """evaluate_weights should return avg_car, hit_rate, n_events for a threshold."""
    events_with_scores = [
        {"car_30d": 0.05, "score": 70},
        {"car_30d": 0.03, "score": 80},
        {"car_30d": -0.01, "score": 30},
        {"car_30d": None,  "score": 75},   # missing CAR — should be excluded
    ]
    metrics = evaluate_weights(events_with_scores, threshold=65)
    assert metrics["n_events"] == 2
    assert abs(metrics["avg_car_30d"] - 0.04) < 0.001
    assert abs(metrics["hit_rate"] - 1.0) < 0.001


def test_find_optimal_threshold():
    """Should return the threshold where avg_car * hit_rate is best."""
    events = [
        {"car_30d": 0.10, "score": 90},
        {"car_30d": 0.08, "score": 80},
        {"car_30d": -0.05, "score": 40},
        {"car_30d": -0.03, "score": 50},
    ]
    best_threshold, _ = find_optimal_threshold(events, candidates=[40, 65, 75, 85], min_events=1)
    assert best_threshold >= 80


def test_grid_search_produces_output():
    """Grid search should produce optimal_weights and stats."""
    results = grid_search(EVENTS, n_candidates=2)
    assert "optimal_weights" in results
    assert "stats" in results
    assert results["stats"]["n_events"] >= 0
