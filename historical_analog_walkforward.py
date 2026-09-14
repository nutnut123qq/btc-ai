"""Leakage-safe walk-forward evaluation for historical candle analogs.

This module is deliberately read-only.  It evaluates the same candle-shape
representation used by the backend (return, body, upper wick, lower wick) and
compares nearest historical neighbours with an expanding-history majority
baseline.  It does not tune parameters on the evaluation period.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from db_config import get_db_connection


ACTIVE_TIMEFRAMES = ("1h", "4h", "1d")
HORIZONS = (1, 3, 6)
INTERVAL_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


@dataclass(frozen=True)
class WalkForwardConfig:
    window_size: int = 15
    neighbour_count: int = 50
    candidate_lookback: int = 5_000
    evaluation_points: int = 250
    round_trip_cost_pct: float = 0.15
    atr_multiplier: float = 0.25

    @property
    def exclusion_bars(self) -> int:
        return self.window_size + max(HORIZONS)


def build_returns_shape_vectors(ohlc: np.ndarray, window_size: int) -> np.ndarray:
    """Build backend-compatible returns_shape vectors for every full window."""
    if ohlc.ndim != 2 or ohlc.shape[1] != 4:
        raise ValueError("ohlc must have shape (n, 4) in open/high/low/close order")
    if window_size < 5 or len(ohlc) < window_size:
        raise ValueError("window_size must be >= 5 and no larger than the input")

    vectors = np.zeros((len(ohlc) - window_size + 1, window_size * 4), dtype=np.float32)
    for start in range(len(vectors)):
        window = ohlc[start : start + window_size]
        previous_close = window[0, 3]
        offset = 0
        for index, (open_price, high, low, close) in enumerate(window):
            candle_range = high - low
            candle_return = 0.0 if index == 0 or abs(previous_close) <= 1e-12 else close / previous_close - 1.0
            if candle_range > 1e-12:
                body = abs(close - open_price) / candle_range
                upper_wick = (high - max(open_price, close)) / candle_range
                lower_wick = (min(open_price, close) - low) / candle_range
            else:
                body = upper_wick = lower_wick = 0.0
            vectors[start, offset : offset + 4] = (candle_return, body, upper_wick, lower_wick)
            previous_close = close
            offset += 4

    norms = np.linalg.norm(vectors, axis=1)
    nonzero = norms > 0
    vectors[nonzero] /= norms[nonzero, None]
    return vectors


def atr14_percent(ohlc: np.ndarray) -> np.ndarray:
    """Return trailing Wilder-style 14-bar mean true range as percent of close."""
    high, low, close = ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]
    previous_close = np.concatenate(([close[0]], close[:-1]))
    true_range = np.maximum(high - low, np.maximum(np.abs(high - previous_close), np.abs(low - previous_close)))
    result = np.full(len(ohlc), np.nan, dtype=np.float64)
    for index in range(13, len(ohlc)):
        mean_range = float(np.mean(true_range[index - 13 : index + 1]))
        if abs(close[index]) > 1e-12:
            result[index] = mean_range / close[index] * 100.0
    return result


def classify_return(return_pct: float, threshold_pct: float) -> int:
    if return_pct > threshold_pct:
        return 1
    if return_pct < -threshold_pct:
        return -1
    return 0


def select_non_overlapping(starts_by_rank: Iterable[int], limit: int, exclusion_bars: int) -> list[int]:
    selected: list[int] = []
    for start in starts_by_rank:
        if all(abs(start - prior) >= exclusion_bars for prior in selected):
            selected.append(int(start))
            if len(selected) == limit:
                break
    return selected


def _dominant_direction(labels: np.ndarray) -> int:
    counts = {label: int(np.sum(labels == label)) for label in (-1, 0, 1)}
    highest = max(counts.values())
    winners = [label for label, count in counts.items() if count == highest]
    return winners[0] if len(winners) == 1 else 0


def _accuracy(actual: list[int], predicted: list[int]) -> float | None:
    if not actual:
        return None
    return sum(a == p for a, p in zip(actual, predicted, strict=True)) / len(actual)


def evaluate_walk_forward(
    ohlc: np.ndarray,
    config: WalkForwardConfig,
    open_times_ms: np.ndarray | None = None,
    interval_ms: int | None = None,
) -> dict:
    """Evaluate fixed analog parameters on chronological, independent query windows."""
    minimum = config.window_size + max(HORIZONS) + 30
    if len(ohlc) < minimum:
        raise ValueError(f"at least {minimum} candles are required")
    if open_times_ms is not None:
        if interval_ms is None or interval_ms <= 0:
            raise ValueError("interval_ms is required with open_times_ms")
        if len(open_times_ms) != len(ohlc):
            raise ValueError("open_times_ms and ohlc must contain the same number of rows")
        break_flags = np.zeros(len(open_times_ms), dtype=np.int64)
        break_flags[1:] = np.diff(open_times_ms) != interval_ms
        break_prefix = np.cumsum(break_flags)
    else:
        open_times_ms = np.arange(len(ohlc), dtype=np.int64)
        interval_ms = 1
        break_prefix = np.zeros(len(ohlc), dtype=np.int64)

    def contiguous(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        return (break_prefix[ends] - break_prefix[starts]) == 0

    vectors = build_returns_shape_vectors(ohlc, config.window_size)
    atr_pct = atr14_percent(ohlc)
    closes = ohlc[:, 3]
    thresholds = np.maximum(config.round_trip_cost_pct, np.nan_to_num(atr_pct, nan=0.0) * config.atr_multiplier)
    labels = {
        horizon: np.array(
            [
                classify_return((closes[end + horizon] / closes[end] - 1.0) * 100.0, thresholds[end])
                if end + horizon < len(closes)
                else 0
                for end in range(len(closes))
            ],
            dtype=np.int8,
        )
        for horizon in HORIZONS
    }

    exclusion = config.exclusion_bars
    last_query_end = len(ohlc) - max(HORIZONS) - 1
    first_query_end = max(config.window_size - 1 + exclusion * 3, last_query_end - exclusion * (config.evaluation_points - 1))
    query_ends = list(range(first_query_end, last_query_end + 1, exclusion))[-config.evaluation_points :]

    actual_by_horizon = {horizon: [] for horizon in HORIZONS}
    analog_by_horizon = {horizon: [] for horizon in HORIZONS}
    baseline_by_horizon = {horizon: [] for horizon in HORIZONS}
    evaluated_queries = 0
    minimum_neighbours = min(5, config.neighbour_count)
    maximum_future_offset: int | None = None
    minimum_selected_gap: int | None = None

    for query_end in query_ends:
        query_start = query_end - config.window_size + 1
        if not bool(contiguous(np.asarray([query_start]), np.asarray([query_end + max(HORIZONS)]))[0]):
            continue
        # A candidate's six-bar future must end strictly before the query starts.
        maximum_candidate_end = query_start - max(HORIZONS) - 1
        minimum_candidate_end = max(config.window_size - 1, maximum_candidate_end - config.candidate_lookback + 1)
        if maximum_candidate_end < minimum_candidate_end:
            continue

        candidate_ends = np.arange(minimum_candidate_end, maximum_candidate_end + 1, dtype=np.int64)
        candidate_starts = candidate_ends - config.window_size + 1
        candidate_future_ends = candidate_ends + max(HORIZONS)
        eligible = contiguous(candidate_starts, candidate_future_ends)
        eligible &= open_times_ms[candidate_future_ends] < open_times_ms[query_start]
        candidate_ends = candidate_ends[eligible]
        candidate_starts = candidate_starts[eligible]
        candidate_future_ends = candidate_future_ends[eligible]
        if len(candidate_ends) < minimum_neighbours:
            continue
        query_vector = vectors[query_start]
        similarities = vectors[candidate_starts] @ query_vector
        ranked_starts = candidate_starts[np.argsort(-similarities, kind="stable")]
        selected_starts = select_non_overlapping(ranked_starts, config.neighbour_count, exclusion)
        if len(selected_starts) < minimum_neighbours:
            continue

        candidate_future_offset = int(
            np.max((open_times_ms[candidate_future_ends] - open_times_ms[query_start]) // interval_ms)
        )
        maximum_future_offset = (
            candidate_future_offset
            if maximum_future_offset is None
            else max(maximum_future_offset, candidate_future_offset)
        )
        if len(selected_starts) > 1:
            ordered_starts = sorted(selected_starts)
            selected_gap = min(right - left for left, right in zip(ordered_starts, ordered_starts[1:]))
            minimum_selected_gap = selected_gap if minimum_selected_gap is None else min(minimum_selected_gap, selected_gap)

        selected_ends = np.asarray(selected_starts, dtype=np.int64) + config.window_size - 1
        evaluated_queries += 1
        for horizon in HORIZONS:
            neighbour_labels = labels[horizon][selected_ends]
            history_labels = labels[horizon][candidate_ends]
            actual_by_horizon[horizon].append(int(labels[horizon][query_end]))
            analog_by_horizon[horizon].append(_dominant_direction(neighbour_labels))
            baseline_by_horizon[horizon].append(_dominant_direction(history_labels))

    horizons: dict[str, dict] = {}
    all_horizons_pass = evaluated_queries >= 200
    for horizon in HORIZONS:
        actual = actual_by_horizon[horizon]
        analog = analog_by_horizon[horizon]
        baseline = baseline_by_horizon[horizon]
        analog_accuracy = _accuracy(actual, analog)
        baseline_accuracy = _accuracy(actual, baseline)
        lift = None if analog_accuracy is None or baseline_accuracy is None else analog_accuracy - baseline_accuracy
        directional_predictions = [index for index, value in enumerate(analog) if value != 0]
        directional_hits = sum(analog[index] == actual[index] for index in directional_predictions)
        directional_accuracy = directional_hits / len(directional_predictions) if directional_predictions else None
        directional_coverage = len(directional_predictions) / len(analog) if analog else 0.0
        passes = bool(
            evaluated_queries >= 200
            and lift is not None
            and lift >= 0.02
            and directional_coverage >= 0.20
            and directional_accuracy is not None
            and directional_accuracy >= 0.52
        )
        all_horizons_pass = all_horizons_pass and passes
        horizons[str(horizon)] = {
            "evaluated": len(actual),
            "analogAccuracy": analog_accuracy,
            "expandingMajorityAccuracy": baseline_accuracy,
            "accuracyLift": lift,
            "directionalAccuracy": directional_accuracy,
            "directionalCoverage": directional_coverage,
            "passesReferenceGate": passes,
        }

    return {
        "method": "historical-analog-returns-shape-v1",
        "rankingMethod": "shape-similarity-desc-context-audit-only",
        "evaluation": "chronological-walk-forward-fixed-parameters",
        "parametersTunedOnEvaluationPeriod": False,
        "querySpacingBars": exclusion,
        "candidateFutureStrictlyBeforeQuery": maximum_future_offset is not None and maximum_future_offset < 0,
        "maximumCandidateFutureOffsetBars": maximum_future_offset,
        "selectedNeighboursNonOverlapping": minimum_selected_gap is not None and minimum_selected_gap >= exclusion,
        "minimumSelectedStartGapBars": minimum_selected_gap,
        "evaluatedQueries": evaluated_queries,
        "horizons": horizons,
        "passesAllReferenceGates": all_horizons_pass,
        "decision": "reference_only" if not all_horizons_pass else "eligible_for_further_validation",
    }


def load_market_data(symbol: str, timeframe: str) -> tuple[np.ndarray, np.ndarray]:
    with get_db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            '''
            SELECT "OpenTimeMs", "Open", "High", "Low", "Close"
            FROM "Klines"
            WHERE "Symbol" = %s AND "Timeframe" = %s
            ORDER BY "OpenTimeMs" ASC
            ''',
            (symbol, timeframe),
        )
        rows = cursor.fetchall()
    data = np.asarray(rows, dtype=np.float64)
    return data[:, 0].astype(np.int64), data[:, 1:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", choices=ACTIVE_TIMEFRAMES, default="4h")
    parser.add_argument("--window-size", type=int, choices=(10, 15, 20, 25), default=15)
    parser.add_argument("--neighbour-count", type=int, default=50)
    parser.add_argument("--candidate-lookback", type=int, default=5_000)
    parser.add_argument("--evaluation-points", type=int, default=250)
    parser.add_argument("--round-trip-cost-pct", type=float, default=0.15)
    parser.add_argument("--atr-multiplier", type=float, default=0.25)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = WalkForwardConfig(
        window_size=args.window_size,
        neighbour_count=args.neighbour_count,
        candidate_lookback=args.candidate_lookback,
        evaluation_points=args.evaluation_points,
        round_trip_cost_pct=args.round_trip_cost_pct,
        atr_multiplier=args.atr_multiplier,
    )
    open_times_ms, ohlc = load_market_data(args.symbol, args.timeframe)
    report = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "config": {
            "windowSize": config.window_size,
            "neighbourCount": config.neighbour_count,
            "candidateLookback": config.candidate_lookback,
            "evaluationPoints": config.evaluation_points,
            "roundTripCostPct": config.round_trip_cost_pct,
            "atrMultiplier": config.atr_multiplier,
        },
        **evaluate_walk_forward(ohlc, config, open_times_ms, INTERVAL_MS[args.timeframe]),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
