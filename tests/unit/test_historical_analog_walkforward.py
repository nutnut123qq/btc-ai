import numpy as np

from historical_analog_walkforward import (
    WalkForwardConfig,
    build_returns_shape_vectors,
    build_returns_shape_v2_vectors,
    classify_return,
    classify_evidence_status,
    evaluate_walk_forward,
    paired_block_bootstrap_accuracy_lift,
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
    assert report["decision"] == "insufficient_evidence"


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


def test_v2_separates_opposite_direction_shapes_that_v1_nearly_conflates():
    def directional_window(direction: float) -> np.ndarray:
        rows = []
        previous = 100.0
        for _ in range(10):
            open_price = previous
            close = open_price + direction
            rows.append((open_price, max(open_price, close) + 0.5, min(open_price, close) - 0.5, close))
            previous = close
        return np.asarray(rows, dtype=np.float64)

    bullish = directional_window(1.0)
    bearish = directional_window(-1.0)
    combined = np.concatenate((bullish, bearish))
    v1 = build_returns_shape_vectors(combined, 10)
    v2 = build_returns_shape_v2_vectors(combined, 10)
    v1_similarity = float(v1[0] @ v1[10])
    v2_similarity = float(v2[0] @ v2[10])
    assert v1_similarity > 0.99
    assert v2_similarity < 0.25


def test_appending_future_data_cannot_change_fixed_past_evaluation():
    original = _synthetic_ohlc(500)
    extended = _synthetic_ohlc(560)
    config = WalkForwardConfig(
        window_size=10,
        neighbour_count=20,
        candidate_lookback=200,
        evaluation_points=12,
        bootstrap_repetitions=100,
    )
    past_end = 470
    first = evaluate_walk_forward(original, config, evaluation_end_index=past_end)
    second = evaluate_walk_forward(extended, config, evaluation_end_index=past_end)
    comparable_keys = (
        "attemptedQueries",
        "evaluatedQueries",
        "acceptedQueryEndTimesMs",
        "horizons",
        "decision",
    )
    assert {key: first[key] for key in comparable_keys} == {key: second[key] for key in comparable_keys}


def test_missing_values_fail_closed_instead_of_becoming_similarity_scores():
    ohlc = _synthetic_ohlc(100)
    ohlc[50, 3] = np.nan
    with np.testing.assert_raises_regex(ValueError, "NaN or infinite"):
        build_returns_shape_v2_vectors(ohlc, 10)


def test_negative_and_inconclusive_evidence_are_first_class_outcomes():
    actual = [1, 1, -1, -1] * 75
    candidate = [-value for value in actual]
    baseline = list(actual)
    interval = paired_block_bootstrap_accuracy_lift(
        actual,
        candidate,
        baseline,
        repetitions=300,
        block_size=8,
        random_seed=7,
    )
    config = WalkForwardConfig(minimum_evidence_samples=200, minimum_coverage=0.2)
    assert interval["upper"] < 0
    assert classify_evidence_status(len(actual), 1.0, interval, config) == "adverse"

    inconclusive_interval = {"lower": -0.05, "upper": 0.05}
    assert classify_evidence_status(250, 1.0, inconclusive_interval, config) == "inconclusive"


def test_quality_gate_abstains_and_reports_coverage():
    report = evaluate_walk_forward(
        _synthetic_ohlc(500),
        WalkForwardConfig(
            window_size=10,
            neighbour_count=20,
            candidate_lookback=200,
            evaluation_points=12,
            minimum_mean_similarity=1.0,
            bootstrap_repetitions=10,
        ),
    )
    assert report["attemptedQueries"] > 0
    assert report["abstainedLowQuality"] > 0
    assert report["acceptedQueryCoverage"] < 1.0
    assert report["candidateFutureStrictlyBeforeQuery"] is True
