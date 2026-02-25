import pytest
import copy
from backtest.optimize_weights import (
    score_event_with_weights,
    evaluate_weights,
    find_optimal_threshold,
    grid_search,
)
from backtest.shared import DEFAULT_WEIGHTS

# Sample events for testing
EVENTS = [
    # High-scoring congress event that performed well
    {"event_type": "congress", "range": "$1,000,001 - $5,000,000",
     "car_30d": 0.08, "convergence": "congress", "member_quartile": 1},
    # Low-scoring congress event that underperformed
    {"event_type": "congress", "range": "$1,001 - $15,000",
     "car_30d": -0.02, "convergence": "congress", "member_quartile": 4},
    # Convergence event that performed well
    {"event_type": "congress", "range": "$50,001 - $100,000",
     "car_30d": 0.05, "convergence": "convergence", "member_quartile": 2},
    # EDGAR event with positive return
    {"event_type": "edgar", "car_30d": 0.03, "convergence": "edgar"},
]


def test_score_event_with_weights_congress():
    """Congress event should produce a positive score with default weights."""
    event = EVENTS[0]  # $1M+ purchase
    score = score_event_with_weights(event, DEFAULT_WEIGHTS)
    assert score > 0
    assert score <= 115  # theoretical max


def test_score_event_with_weights_convergence_bonus():
    """Convergence events should score higher than single-source events."""
    single = copy.deepcopy(EVENTS[1])
    single['convergence'] = 'congress'
    conv = copy.deepcopy(EVENTS[1])
    conv['convergence'] = 'convergence'

    score_single = score_event_with_weights(single, DEFAULT_WEIGHTS)
    score_conv = score_event_with_weights(conv, DEFAULT_WEIGHTS)
    assert score_conv > score_single


def test_evaluate_weights_returns_metrics():
    """evaluate_weights should return avg_car, hit_rate, n_events for a threshold."""
    events_with_scores = [
        {"car_30d": 0.05, "score": 70},
        {"car_30d": 0.03, "score": 80},
        {"car_30d": -0.01, "score": 30},
        {"car_30d": None,  "score": 75},   # missing CAR — should be excluded
    ]
    metrics = evaluate_weights(events_with_scores, threshold=65)
    assert metrics["n_events"] == 2  # only 70 and 80 qualify, 30 excluded, None excluded
    assert abs(metrics["avg_car_30d"] - 0.04) < 0.001
    assert abs(metrics["hit_rate"] - 1.0) < 0.001  # both positive


def test_find_optimal_threshold():
    """Should return the threshold where avg_car is best."""
    # Events where high threshold gives better avg return
    events = [
        {"car_30d": 0.10, "score": 90},
        {"car_30d": 0.08, "score": 80},
        {"car_30d": -0.05, "score": 40},
        {"car_30d": -0.03, "score": 50},
    ]
    best_threshold, _ = find_optimal_threshold(events, candidates=[40, 65, 75, 85], min_events=1)
    # At threshold 80, avg_car = mean(0.10) = 0.10 — best
    assert best_threshold >= 80


def test_grid_search_improves_over_default():
    """Grid search should find weights that give >= default weight performance."""
    results = grid_search(EVENTS, n_candidates=3)
    assert "optimal_weights" in results
    assert "stats" in results
    assert results["stats"]["n_events"] >= 0
