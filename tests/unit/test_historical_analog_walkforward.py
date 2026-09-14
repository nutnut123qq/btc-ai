import numpy as np

from historical_analog_walkforward import (
    WalkForwardConfig,
    build_returns_shape_vectors,
    classify_return,
    evaluate_walk_forward,
    select_non_overlapping,
)


def _synthetic_ohlc(count: int) -> np.ndarray:
    indices = np.arange(count, dtype=np.float64)
    close = 100.0 + indices * 0.08 + np.sin(indices / 4.0) * 1.5
    open_price = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_price, close) + 0.4
    low = np.minimum(open_price, close) - 0.4
    return np.column_stack((open_price, high, low, close))


def test_returns_shape_matches_backend_layout_and_normalizes_rows():
    vectors = build_returns_shape_vectors(_synthetic_ohlc(20), 10)
    assert vectors.shape == (11, 40)
    assert vectors[0, 0] == 0.0
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_economic_threshold_keeps_small_moves_neutral():
    assert classify_return(0.14, 0.15) == 0
    assert classify_return(-0.14, 0.15) == 0
    assert classify_return(0.16, 0.15) == 1
    assert classify_return(-0.16, 0.15) == -1


def test_non_overlap_selection_uses_window_plus_future_horizon():
    assert select_non_overlapping([100, 101, 120, 121, 142], 3, 21) == [100, 121, 142]


def test_walk_forward_is_explicitly_leakage_safe_and_reference_only_when_small():
    report = evaluate_walk_forward(
        _synthetic_ohlc(500),
        WalkForwardConfig(window_size=10, neighbour_count=20, candidate_lookback=200, evaluation_points=12),
    )
    assert report["parametersTunedOnEvaluationPeriod"] is False
    assert report["candidateFutureStrictlyBeforeQuery"] is True
    assert report["maximumCandidateFutureOffsetBars"] == -1
    assert report["selectedNeighboursNonOverlapping"] is True
    assert report["minimumSelectedStartGapBars"] >= report["querySpacingBars"]
    assert report["querySpacingBars"] == 16
    assert 0 < report["evaluatedQueries"] <= 12
    assert report["passesAllReferenceGates"] is False
    assert report["decision"] == "reference_only"


def test_walk_forward_skips_query_windows_that_cross_a_time_gap():
    ohlc = _synthetic_ohlc(500)
    times = np.arange(500, dtype=np.int64) * 3_600_000
    times[450:] += 3_600_000
    report = evaluate_walk_forward(
        ohlc,
        WalkForwardConfig(window_size=10, neighbour_count=20, candidate_lookback=200, evaluation_points=12),
        times,
        3_600_000,
    )
    assert report["candidateFutureStrictlyBeforeQuery"] is True
    assert report["evaluatedQueries"] < 12
