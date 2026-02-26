"""
optimize_weights.py — grid search for optimal ATLAS scoring weights.

Usage:
    python backtest/optimize_weights.py

Inputs:
    data/backtest_results.json  (aggregate per-ticker events from run_event_study.py)

Output:
    data/optimal_weights.json  — best weights found
    data/backtest_summary.json — human-readable performance report
"""

import sys
import copy
import logging
import itertools
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.shared import (
    BACKTEST_RESULTS, OPTIMAL_WEIGHTS, BACKTEST_SUMMARY,
    DEFAULT_WEIGHTS, load_json, save_json, range_to_base_points
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


def score_event_with_weights(event: dict, weights: dict) -> float:
    """
    Re-score an aggregate per-ticker event using provided weights.
    Uses trade_ranges and edgar_count stored in the event to recompute
    the score with different weight parameters.
    """
    # Congress: re-score from stored trade_ranges using provided weight tiers
    congress_score = 0.0
    trade_ranges = event.get('trade_ranges', [])
    tiers = weights.get('congress_tiers', DEFAULT_WEIGHTS['congress_tiers'])
    for r in trade_ranges:
        rr = r or ''
        if '$1,000,001' in rr:   congress_score += tiers.get('xl', 15)
        elif '$500,001' in rr:   congress_score += tiers.get('significant', 10)
        elif '$250,001' in rr:   congress_score += tiers.get('significant', 10)
        elif '$100,001' in rr:   congress_score += tiers.get('major', 8)
        elif '$50,001' in rr:    congress_score += tiers.get('large', 6)
        elif '$15,001' in rr:    congress_score += tiers.get('medium', 5)
        else:                    congress_score += tiers.get('small', 3)

    congress_count = event.get('congress_count', 0)
    if congress_count >= 3:
        congress_score += weights.get('congress_cluster_bonus', 15)
    congress_score = min(congress_score, 40)

    # EDGAR: re-score from filing count
    edgar_count = event.get('edgar_count', 0)
    edgar_score = 0.0
    if edgar_count > 0:
        edgar_score = min(edgar_count * weights.get('edgar_base_per_filing', 6), 25)
        if edgar_count >= 3:
            edgar_score += weights.get('edgar_cluster_3plus', 15)
        elif edgar_count >= 2:
            edgar_score += weights.get('edgar_cluster_2', 10)
        edgar_score = min(edgar_score, 40)

    # Convergence boost
    boost = 0
    if congress_score > 0 and edgar_score > 0:
        boost = weights.get('convergence_boost', 20)

    return congress_score + edgar_score + boost


def evaluate_weights(events_with_scores: list, threshold: float) -> dict:
    """
    Evaluate performance of a weight configuration.
    Returns {avg_car_30d, hit_rate, n_events} for events above threshold.
    """
    above = [e for e in events_with_scores if e.get('score', 0) >= threshold and e.get('car_30d') is not None]
    if not above:
        return {"avg_car_30d": 0.0, "hit_rate": 0.0, "n_events": 0}

    cars = [e['car_30d'] for e in above]
    avg_car = float(np.mean(cars))
    hit_rate = sum(1 for c in cars if c > 0) / len(cars)
    return {
        "avg_car_30d": round(avg_car, 6),
        "hit_rate": round(hit_rate, 4),
        "n_events": len(above),
    }


def find_optimal_threshold(events_with_scores: list, candidates=None, min_events: int = 5) -> tuple:
    """
    Find the threshold that maximizes avg_car_30d * hit_rate for above-threshold events.
    Returns (best_threshold, metrics_dict).
    """
    if candidates is None:
        candidates = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 65, 70, 75, 80]

    best_score = -999
    best_threshold = 65
    best_metrics = {}

    for threshold in candidates:
        metrics = evaluate_weights(events_with_scores, threshold)
        if metrics['n_events'] < min_events:
            continue
        combined = metrics['avg_car_30d'] * metrics['hit_rate']
        if combined > best_score:
            best_score = combined
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def grid_search(events: list, n_candidates: int = 4) -> dict:
    """
    Grid search over key weight parameters.
    Returns {optimal_weights, stats, threshold}.
    """
    search_space = {
        'congress_xl':      [12, 15, 18, 20][:n_candidates],
        'congress_cluster': [10, 15, 18, 20][:n_candidates],
        'congress_q1':      [0, 5, 8, 10][:n_candidates],
        'convergence':      [15, 20, 25, 30][:n_candidates],
        'decay_half_life':  [14, 21, 30, 45][:n_candidates],
    }

    best_score = -999
    best_weights = copy.deepcopy(DEFAULT_WEIGHTS)
    best_events_scored = []

    param_names = list(search_space.keys())
    param_values = list(search_space.values())
    combinations = list(itertools.product(*param_values))
    log.info(f"Grid search: testing {len(combinations)} weight combinations...")

    for i, combo in enumerate(combinations):
        candidate = copy.deepcopy(DEFAULT_WEIGHTS)
        params = dict(zip(param_names, combo))

        candidate['congress_tiers']['xl'] = params['congress_xl']
        candidate['congress_cluster_bonus'] = params['congress_cluster']
        candidate['congress_track_record_q1'] = params['congress_q1']
        candidate['convergence_boost'] = params['convergence']
        candidate['decay_half_life_days'] = params['decay_half_life']

        # Score all events with this weight combination
        scored_events = []
        for event in events:
            e = dict(event)
            e['score'] = score_event_with_weights(e, candidate)
            scored_events.append(e)

        # Find best threshold for this weight combo
        threshold, metrics = find_optimal_threshold(scored_events)
        if metrics.get('n_events', 0) < 5:
            continue

        combined = metrics['avg_car_30d'] * metrics['hit_rate']
        if combined > best_score:
            best_score = combined
            best_weights = copy.deepcopy(candidate)
            best_weights['_optimal_threshold'] = threshold
            best_events_scored = scored_events

    # Final evaluation at optimal threshold
    final_threshold = best_weights.get('_optimal_threshold', 65)
    best_weights.pop('_optimal_threshold', None)
    final_metrics = evaluate_weights(best_events_scored, final_threshold)

    return {
        "optimal_weights": best_weights,
        "optimal_threshold": final_threshold,
        "stats": final_metrics,
    }


def main():
    log.info("Loading backtest results...")
    try:
        results = load_json(BACKTEST_RESULTS)
    except FileNotFoundError:
        log.error("backtest_results.json not found. Run run_event_study.py first.")
        sys.exit(1)

    events = results.get('events', [])
    if len(events) < 10:
        log.warning(f"Only {len(events)} events — too few to optimize. Using defaults.")
        save_json(OPTIMAL_WEIGHTS, {
            **DEFAULT_WEIGHTS,
            "generated": datetime.now(tz=timezone.utc).isoformat(),
            "optimal_threshold": 65,
            "stats": {"note": "insufficient_data"},
        })
        return

    log.info(f"Optimizing weights over {len(events)} aggregate events...")
    result = grid_search(events, n_candidates=4)

    output = {
        **result["optimal_weights"],
        "generated": datetime.now(tz=timezone.utc).isoformat(),
        "n_events_total": len(events),
        "optimal_threshold": result["optimal_threshold"],
        "stats": result["stats"],
    }
    save_json(OPTIMAL_WEIGHTS, output)

    # Summary for humans
    stats = result["stats"]
    summary = {
        "generated": output["generated"],
        "n_events_backtested": len(events),
        "optimal_threshold": result["optimal_threshold"],
        "avg_car_30d_pct": round(stats.get("avg_car_30d", 0) * 100, 2),
        "hit_rate_pct": round(stats.get("hit_rate", 0) * 100, 1),
        "n_events_above_threshold": stats.get("n_events", 0),
        "key_changes_from_default": {
            k: output.get(k) for k in ["convergence_boost", "decay_half_life_days",
                                        "congress_track_record_q1", "optimal_threshold"]
        },
    }
    save_json(BACKTEST_SUMMARY, summary)

    log.info(f"\n=== BACKTEST SUMMARY ===")
    log.info(f"Optimal threshold:  {result['optimal_threshold']}")
    log.info(f"Events above threshold: {stats.get('n_events', 0)}")
    log.info(f"Avg 30d CAR:        {round(stats.get('avg_car_30d', 0)*100, 2)}%")
    log.info(f"Hit rate:           {round(stats.get('hit_rate', 0)*100, 1)}%")
    log.info(f"Weights saved to:   {OPTIMAL_WEIGHTS}")


if __name__ == '__main__':
    main()
