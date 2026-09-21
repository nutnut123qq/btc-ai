"""Point-in-time walk-forward evidence for BTC historical candle analogs.

The evaluator permits negative and inconclusive results. It compares analog
decisions with a rolling majority baseline on identical timestamps, records
abstentions, and reports a paired time-block bootstrap interval. The cost floor
is a classification dead zone, not a fill simulation or net-PnL calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from db_config import get_db_connection
from research_contract import DEFAULT_EXECUTION_COSTS, HISTORICAL_ANALOG_TRIAL_FAMILY, ResearchManifest
from trading_config import DEFAULT_SYMBOL


ACTIVE_TIMEFRAMES = ("1h", "4h", "1d")
HORIZONS = (1, 3, 6)
INTERVAL_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
DEFAULT_TRIAL_LEDGER = Path("docs/research/historical_analog_trials.jsonl")


@dataclass(frozen=True)
class WalkForwardConfig:
    window_size: int = 15
    neighbour_count: int = 50
    candidate_lookback: int = 20_000
    evaluation_points: int = 250
    round_trip_cost_pct_points: float = DEFAULT_EXECUTION_COSTS.round_trip_pct_points
    atr_multiplier: float = 0.25
    representation: str = "returns_shape_v2_signed"
    minimum_mean_similarity: float = 0.72
    minimum_evidence_samples: int = 200
    minimum_coverage: float = 0.20
    bootstrap_repetitions: int = 1_000
    bootstrap_block_size: int = 12
    random_seed: int = 17

    def __post_init__(self) -> None:
        if self.window_size < 5:
            raise ValueError("window_size must be >= 5")
        if self.neighbour_count < 1 or self.candidate_lookback < 1:
            raise ValueError("neighbour_count and candidate_lookback must be positive")
        if self.representation not in HISTORICAL_ANALOG_TRIAL_FAMILY:
            raise ValueError(f"unsupported representation: {self.representation}")
        if not -1.0 <= self.minimum_mean_similarity <= 1.0:
            raise ValueError("minimum_mean_similarity must be within [-1, 1]")
        if self.bootstrap_repetitions < 0 or self.bootstrap_block_size < 1:
            raise ValueError("invalid bootstrap configuration")

    @property
    def exclusion_bars(self) -> int:
        return self.window_size + max(HORIZONS)

    def to_manifest_parameters(self) -> dict[str, object]:
        return {
            "windowSize": self.window_size,
            "neighbourCount": self.neighbour_count,
            "candidateLookbackBars": self.candidate_lookback,
            "evaluationPoints": self.evaluation_points,
            "economicDeadZoneFloorPctPoints": self.round_trip_cost_pct_points,
            "atrDeadZoneMultiplier": self.atr_multiplier,
            "representation": self.representation,
            "minimumMeanSimilarity": self.minimum_mean_similarity,
            "minimumEvidenceSamples": self.minimum_evidence_samples,
            "minimumCoverage": self.minimum_coverage,
            "bootstrap": {
                "method": "paired-circular-moving-block",
                "repetitions": self.bootstrap_repetitions,
                "blockSizeQueries": self.bootstrap_block_size,
                "randomSeed": self.random_seed,
            },
        }


def _validate_ohlc(ohlc: np.ndarray, window_size: int) -> None:
    if ohlc.ndim != 2 or ohlc.shape[1] != 4:
        raise ValueError("ohlc must have shape (n, 4) in open/high/low/close order")
    if len(ohlc) < window_size:
        raise ValueError("window_size must be no larger than the input")
    if not np.isfinite(ohlc).all():
        raise ValueError("ohlc contains NaN or infinite values")
    if np.any(ohlc <= 0):
        raise ValueError("ohlc prices must be positive")
    if np.any(ohlc[:, 1] < np.maximum(ohlc[:, 0], ohlc[:, 3])):
        raise ValueError("high must be at least open and close")
    if np.any(ohlc[:, 2] > np.minimum(ohlc[:, 0], ohlc[:, 3])):
        raise ValueError("low must be at most open and close")


def _normalise_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    nonzero = norms > 0
    vectors[nonzero] /= norms[nonzero, None]
    return vectors


def build_returns_shape_vectors(ohlc: np.ndarray, window_size: int) -> np.ndarray:
    """Build the legacy v1 representation for audit comparisons only."""

    _validate_ohlc(ohlc, window_size)
    vectors = np.zeros((len(ohlc) - window_size + 1, window_size * 4), dtype=np.float32)
    for start in range(len(vectors)):
        window = ohlc[start : start + window_size]
        previous_close = window[0, 3]
        for index, (open_price, high, low, close) in enumerate(window):
            candle_range = high - low
            candle_return = 0.0 if index == 0 else close / previous_close - 1.0
            if candle_range > 1e-12:
                body = abs(close - open_price) / candle_range
                upper_wick = (high - max(open_price, close)) / candle_range
                lower_wick = (min(open_price, close) - low) / candle_range
            else:
                body = upper_wick = lower_wick = 0.0
            offset = index * 4
            vectors[start, offset : offset + 4] = (candle_return, body, upper_wick, lower_wick)
            previous_close = close
    return _normalise_rows(vectors)


def build_returns_shape_v2_vectors(ohlc: np.ndarray, window_size: int) -> np.ndarray:
    """Build a signed, scale-balanced causal candle-shape representation."""

    _validate_ohlc(ohlc, window_size)
    vectors = np.zeros((len(ohlc) - window_size + 1, window_size * 4), dtype=np.float32)
    for start in range(len(vectors)):
        window = ohlc[start : start + window_size]
        previous_close = window[0, 3]
        for index, (open_price, high, low, close) in enumerate(window):
            candle_range = high - low
            candle_return = 0.0 if index == 0 else close / previous_close - 1.0
            range_fraction = candle_range / previous_close if previous_close > 1e-12 else 0.0
            scaled_return = float(np.clip(candle_return / max(range_fraction, 1e-12), -1.0, 1.0))
            if candle_range > 1e-12:
                signed_body = (close - open_price) / candle_range
                upper_wick = (high - max(open_price, close)) / candle_range
                lower_wick = (min(open_price, close) - low) / candle_range
            else:
                signed_body = upper_wick = lower_wick = 0.0
            offset = index * 4
            vectors[start, offset : offset + 4] = (
                scaled_return,
                signed_body,
                upper_wick,
                lower_wick,
            )
            previous_close = close
    return _normalise_rows(vectors)


def build_representation(ohlc: np.ndarray, config: WalkForwardConfig) -> np.ndarray:
    if config.representation == "returns_shape_v1_legacy":
        return build_returns_shape_vectors(ohlc, config.window_size)
    return build_returns_shape_v2_vectors(ohlc, config.window_size)


def atr14_percent(ohlc: np.ndarray) -> np.ndarray:
    """Return trailing 14-bar mean true range in percentage points of close."""

    high, low, close = ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]
    previous_close = np.concatenate(([close[0]], close[:-1]))
    true_range = np.maximum(high - low, np.maximum(np.abs(high - previous_close), np.abs(low - previous_close)))
    result = np.full(len(ohlc), np.nan, dtype=np.float64)
    for index in range(13, len(ohlc)):
        mean_range = float(np.mean(true_range[index - 13 : index + 1]))
        result[index] = mean_range / close[index] * 100.0
    return result


def classify_return(return_pct_points: float, threshold_pct_points: float) -> int:
    if return_pct_points > threshold_pct_points:
        return 1
    if return_pct_points < -threshold_pct_points:
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
    return None if not actual else float(np.mean(np.asarray(actual) == np.asarray(predicted)))


def paired_block_bootstrap_accuracy_lift(
    actual: list[int],
    candidate: list[int],
    baseline: list[int],
    *,
    repetitions: int,
    block_size: int,
    random_seed: int,
) -> dict[str, float | int | str | None]:
    """Paired circular moving-block interval for candidate-minus-baseline accuracy."""

    n = len(actual)
    if not (n == len(candidate) == len(baseline)):
        raise ValueError("paired bootstrap inputs must have equal lengths")
    if n == 0:
        return {"method": "paired-circular-moving-block", "lower": None, "upper": None, "samples": 0}
    differences = (
        (np.asarray(candidate) == np.asarray(actual)).astype(np.float64)
        - (np.asarray(baseline) == np.asarray(actual)).astype(np.float64)
    )
    point = float(np.mean(differences))
    if repetitions == 0 or n < 2:
        return {
            "method": "paired-circular-moving-block",
            "pointEstimate": point,
            "lower": None,
            "upper": None,
            "samples": n,
            "repetitions": repetitions,
            "blockSize": min(block_size, n),
        }
    effective_block = min(block_size, n)
    blocks_needed = int(np.ceil(n / effective_block))
    offsets = np.arange(effective_block)
    rng = np.random.default_rng(random_seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        starts = rng.integers(0, n, size=blocks_needed)
        sampled = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        estimates[index] = float(np.mean(differences[sampled]))
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "method": "paired-circular-moving-block",
        "pointEstimate": point,
        "lower": float(lower),
        "upper": float(upper),
        "samples": n,
        "repetitions": repetitions,
        "blockSize": effective_block,
    }


def classify_evidence_status(
    evaluated: int,
    coverage: float,
    interval: dict[str, float | int | str | None],
    config: WalkForwardConfig,
) -> str:
    if evaluated < config.minimum_evidence_samples or coverage < config.minimum_coverage:
        return "insufficient_evidence"
    lower, upper = interval.get("lower"), interval.get("upper")
    if lower is None or upper is None:
        return "inconclusive"
    if float(lower) > 0.0:
        return "supported_lift"
    if float(upper) < 0.0:
        return "adverse"
    return "inconclusive"


def evaluate_walk_forward(
    ohlc: np.ndarray,
    config: WalkForwardConfig,
    open_times_ms: np.ndarray | None = None,
    interval_ms: int | None = None,
    *,
    evaluation_end_index: int | None = None,
) -> dict:
    """Evaluate fixed analog parameters using only information available at each query."""

    minimum = config.window_size + max(HORIZONS) + 30
    _validate_ohlc(ohlc, config.window_size)
    if len(ohlc) < minimum:
        raise ValueError(f"at least {minimum} candles are required")
    if open_times_ms is not None:
        if interval_ms is None or interval_ms <= 0:
            raise ValueError("interval_ms is required with open_times_ms")
        if len(open_times_ms) != len(ohlc):
            raise ValueError("open_times_ms and ohlc must contain the same number of rows")
        if not np.issubdtype(open_times_ms.dtype, np.integer):
            raise ValueError("open_times_ms must contain integer epoch milliseconds")
        if np.any(np.diff(open_times_ms) <= 0):
            raise ValueError("open_times_ms must be strictly increasing")
        break_flags = np.zeros(len(open_times_ms), dtype=np.int64)
        break_flags[1:] = np.diff(open_times_ms) != interval_ms
        break_prefix = np.cumsum(break_flags)
    else:
        open_times_ms = np.arange(len(ohlc), dtype=np.int64)
        interval_ms = 1
        break_prefix = np.zeros(len(ohlc), dtype=np.int64)

    def contiguous(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        return (break_prefix[ends] - break_prefix[starts]) == 0

    vectors = build_representation(ohlc, config)
    atr_pct_points = atr14_percent(ohlc)
    closes = ohlc[:, 3]
    dead_zone_pct_points = np.maximum(
        config.round_trip_cost_pct_points,
        np.nan_to_num(atr_pct_points, nan=0.0) * config.atr_multiplier,
    )
    labels = {
        horizon: np.array(
            [
                classify_return(
                    (closes[end + horizon] / closes[end] - 1.0) * 100.0,
                    dead_zone_pct_points[end],
                )
                if end + horizon < len(closes)
                else 0
                for end in range(len(closes))
            ],
            dtype=np.int8,
        )
        for horizon in HORIZONS
    }

    exclusion = config.exclusion_bars
    safe_last_query_end = len(ohlc) - max(HORIZONS) - 1
    last_query_end = safe_last_query_end if evaluation_end_index is None else evaluation_end_index
    if last_query_end > safe_last_query_end or last_query_end < config.window_size - 1:
        raise ValueError("evaluation_end_index must identify a query with fully observed outcomes")
    first_query_end = max(
        config.window_size - 1 + exclusion * 3,
        last_query_end - exclusion * (config.evaluation_points - 1),
    )
    query_ends = list(range(first_query_end, last_query_end + 1, exclusion))[-config.evaluation_points :]

    actual_by_horizon = {horizon: [] for horizon in HORIZONS}
    analog_by_horizon = {horizon: [] for horizon in HORIZONS}
    baseline_by_horizon = {horizon: [] for horizon in HORIZONS}
    attempted_queries = 0
    abstained_low_quality = 0
    evaluated_queries = 0
    minimum_neighbours = min(5, config.neighbour_count)
    maximum_future_offset: int | None = None
    minimum_selected_gap: int | None = None
    accepted_similarities: list[float] = []
    accepted_query_end_times: list[int] = []

    for query_end in query_ends:
        query_start = query_end - config.window_size + 1
        if not bool(contiguous(np.asarray([query_start]), np.asarray([query_end + max(HORIZONS)]))[0]):
            continue
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

        attempted_queries += 1
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
            candidate_future_offset if maximum_future_offset is None else max(maximum_future_offset, candidate_future_offset)
        )
        if len(selected_starts) > 1:
            ordered_starts = sorted(selected_starts)
            selected_gap = min(right - left for left, right in zip(ordered_starts, ordered_starts[1:]))
            minimum_selected_gap = selected_gap if minimum_selected_gap is None else min(minimum_selected_gap, selected_gap)

        similarity_by_start = {int(start): float(similarity) for start, similarity in zip(candidate_starts, similarities)}
        mean_similarity = float(np.mean([similarity_by_start[start] for start in selected_starts]))
        if mean_similarity < config.minimum_mean_similarity:
            abstained_low_quality += 1
            continue

        selected_ends = np.asarray(selected_starts, dtype=np.int64) + config.window_size - 1
        evaluated_queries += 1
        accepted_similarities.append(mean_similarity)
        accepted_query_end_times.append(int(open_times_ms[query_end]))
        for horizon in HORIZONS:
            neighbour_labels = labels[horizon][selected_ends]
            rolling_history_labels = labels[horizon][candidate_ends]
            actual_by_horizon[horizon].append(int(labels[horizon][query_end]))
            analog_by_horizon[horizon].append(_dominant_direction(neighbour_labels))
            baseline_by_horizon[horizon].append(_dominant_direction(rolling_history_labels))

    coverage = evaluated_queries / attempted_queries if attempted_queries else 0.0
    horizons: dict[str, dict] = {}
    statuses: list[str] = []
    for horizon in HORIZONS:
        actual = actual_by_horizon[horizon]
        analog = analog_by_horizon[horizon]
        baseline = baseline_by_horizon[horizon]
        analog_accuracy = _accuracy(actual, analog)
        baseline_accuracy = _accuracy(actual, baseline)
        lift_interval = paired_block_bootstrap_accuracy_lift(
            actual,
            analog,
            baseline,
            repetitions=config.bootstrap_repetitions,
            block_size=config.bootstrap_block_size,
            random_seed=config.random_seed + horizon,
        )
        directional_predictions = [index for index, value in enumerate(analog) if value != 0]
        directional_accuracy = (
            sum(analog[index] == actual[index] for index in directional_predictions) / len(directional_predictions)
            if directional_predictions
            else None
        )
        directional_coverage = len(directional_predictions) / len(analog) if analog else 0.0
        status = classify_evidence_status(len(actual), coverage, lift_interval, config)
        statuses.append(status)
        horizons[str(horizon)] = {
            "evaluated": len(actual),
            "analogAccuracy": analog_accuracy,
            "rollingCandidateMajorityAccuracy": baseline_accuracy,
            "baselineProvenance": (
                "majority label among all eligible, contiguous candidates inside the configured "
                "rolling lookback; evaluated on the same accepted query timestamps"
            ),
            "accuracyLift": lift_interval.get("pointEstimate"),
            "accuracyLift95PctInterval": lift_interval,
            "directionalAccuracy": directional_accuracy,
            "directionalCoverageWithinAcceptedQueries": directional_coverage,
            "evidenceStatus": status,
        }

    if any(status == "adverse" for status in statuses):
        decision = "retain_exploratory_adverse_evidence"
    elif statuses and all(status == "supported_lift" for status in statuses):
        decision = "eligible_for_further_validation"
    elif all(status == "insufficient_evidence" for status in statuses):
        decision = "insufficient_evidence"
    else:
        decision = "retain_exploratory_inconclusive"

    return {
        "method": "historical-analog-returns-shape-v2",
        "representation": config.representation,
        "declaredTrialFamily": list(HISTORICAL_ANALOG_TRIAL_FAMILY),
        "rankingMethod": "cosine-similarity-desc-point-in-time",
        "evaluation": "chronological-walk-forward-fixed-parameters",
        "parametersTunedOnEvaluationPeriod": False,
        "querySpacingBars": exclusion,
        "candidateFutureStrictlyBeforeQuery": maximum_future_offset is not None and maximum_future_offset < 0,
        "maximumCandidateFutureOffsetBars": maximum_future_offset,
        "selectedNeighboursNonOverlapping": minimum_selected_gap is not None and minimum_selected_gap >= exclusion,
        "minimumSelectedStartGapBars": minimum_selected_gap,
        "attemptedQueries": attempted_queries,
        "evaluatedQueries": evaluated_queries,
        "abstainedLowQuality": abstained_low_quality,
        "acceptedQueryCoverage": coverage,
        "meanAcceptedSimilarity": float(np.mean(accepted_similarities)) if accepted_similarities else None,
        "acceptedQueryEndTimesMs": accepted_query_end_times,
        "horizons": horizons,
        "passesAllReferenceGates": bool(statuses) and all(status == "supported_lift" for status in statuses),
        "decision": decision,
        "limitations": [
            "Cost is used only as a classification dead-zone floor; returns are not a fill simulation.",
            "OHLC bars cannot establish intrabar execution ordering.",
            "A supported interval is evidence for further validation, not live-trading approval.",
        ],
    }


def _market_data_sha256(open_times_ms: np.ndarray, ohlc: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(open_times_ms, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(ohlc, dtype=np.float64).tobytes())
    return digest.hexdigest()


def build_manifest(
    symbol: str,
    timeframe: str,
    config: WalkForwardConfig,
    open_times_ms: np.ndarray,
    ohlc: np.ndarray,
    decision_cutoff_ms: int,
) -> dict:
    provenance = {
        "source": 'PostgreSQL table "Klines"',
        "orderedBy": "OpenTimeMs ASC",
        "rowCount": len(ohlc),
        "firstOpenTimeMs": int(open_times_ms[0]),
        "lastOpenTimeMs": int(open_times_ms[-1]),
        "decisionCutoffMs": decision_cutoff_ms,
        "finalizedCandlePredicate": "CloseTimeMs <= decisionCutoffMs",
        "contentSha256": _market_data_sha256(open_times_ms, ohlc),
    }
    code_provenance = _code_provenance()
    runtime_dependencies = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "psycopg2-binary": _dependency_version("psycopg2-binary"),
    }
    return ResearchManifest(
        experiment="historical-analog-walk-forward",
        symbol=symbol,
        timeframe=timeframe,
    ).to_dict(
        parameters=config.to_manifest_parameters(),
        data_provenance=provenance,
        code_provenance=code_provenance,
        runtime_dependencies=runtime_dependencies,
    )


def _dependency_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _code_provenance() -> dict[str, object]:
    root = Path(__file__).resolve().parent
    relative_files = (
        "historical_analog_walkforward.py",
        "research_contract.py",
        "trading_config.py",
        "db_config.py",
    )
    file_hashes: dict[str, str] = {}
    combined = hashlib.sha256()
    for relative in relative_files:
        content = (root / relative).read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        file_hashes[relative] = digest
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(content)
        combined.update(b"\0")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--", *relative_files],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
        status = "unavailable"
    return {
        "gitCommit": commit,
        "dirtyRelativeToCommit": status not in ("", "unavailable"),
        "implementationContentSha256": combined.hexdigest(),
        "fileSha256": file_hashes,
    }


def append_trial_ledger(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "recordedAtUtc": datetime.now(UTC).isoformat(),
        "manifestSha256": report["manifest"]["manifestSha256"],
        "symbol": report["symbol"],
        "timeframe": report["timeframe"],
        "representation": report["representation"],
        "declaredTrialFamily": report["declaredTrialFamily"],
        "decision": report["decision"],
        "evaluatedQueries": report["evaluatedQueries"],
        "horizons": report["horizons"],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def load_market_data(
    symbol: str,
    timeframe: str,
    decision_cutoff_ms: int,
) -> tuple[np.ndarray, np.ndarray]:
    with get_db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            '''
            SELECT "OpenTimeMs", "Open", "High", "Low", "Close"
            FROM "Klines"
            WHERE "Symbol" = %s AND "Timeframe" = %s
              AND "CloseTimeMs" <= %s
            ORDER BY "OpenTimeMs" ASC
            ''',
            (symbol, timeframe, decision_cutoff_ms),
        )
        rows = cursor.fetchall()
    if not rows:
        raise ValueError(f"no market data for {symbol} {timeframe}")
    data = np.asarray(rows, dtype=np.float64)
    return data[:, 0].astype(np.int64), data[:, 1:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, choices=(DEFAULT_SYMBOL,))
    parser.add_argument("--timeframe", choices=ACTIVE_TIMEFRAMES, default="4h")
    parser.add_argument("--window-size", type=int, choices=(10, 15, 20, 25), default=15)
    parser.add_argument("--neighbour-count", type=int, default=50)
    parser.add_argument("--candidate-lookback", type=int, default=20_000)
    parser.add_argument("--evaluation-points", type=int, default=250)
    parser.add_argument(
        "--round-trip-cost-pct-points",
        type=float,
        default=DEFAULT_EXECUTION_COSTS.round_trip_pct_points,
        help="Round-trip screening cost in percentage points (default 0.30 = 30 bps).",
    )
    parser.add_argument("--atr-multiplier", type=float, default=0.25)
    parser.add_argument("--representation", choices=HISTORICAL_ANALOG_TRIAL_FAMILY, default="returns_shape_v2_signed")
    parser.add_argument("--minimum-mean-similarity", type=float, default=0.72)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1_000)
    parser.add_argument("--bootstrap-block-size", type=int, default=12)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trial-ledger", type=Path, default=DEFAULT_TRIAL_LEDGER)
    args = parser.parse_args()

    config = WalkForwardConfig(
        window_size=args.window_size,
        neighbour_count=args.neighbour_count,
        candidate_lookback=args.candidate_lookback,
        evaluation_points=args.evaluation_points,
        round_trip_cost_pct_points=args.round_trip_cost_pct_points,
        atr_multiplier=args.atr_multiplier,
        representation=args.representation,
        minimum_mean_similarity=args.minimum_mean_similarity,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_block_size=args.bootstrap_block_size,
    )
    decision_cutoff_ms = int(datetime.now(UTC).timestamp() * 1000)
    open_times_ms, ohlc = load_market_data(args.symbol, args.timeframe, decision_cutoff_ms)
    report = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "manifest": build_manifest(
            args.symbol,
            args.timeframe,
            config,
            open_times_ms,
            ohlc,
            decision_cutoff_ms,
        ),
        **evaluate_walk_forward(ohlc, config, open_times_ms, INTERVAL_MS[args.timeframe]),
    }
    append_trial_ledger(args.trial_ledger, report)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
