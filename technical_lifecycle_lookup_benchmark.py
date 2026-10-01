#!/usr/bin/env python3
"""Production-shaped benchmark for indexed technical-event lifecycle lookup.

The reference sample intentionally uses never-crossed levels, the historical
worst case that scanned to the end of a contiguous candle segment.  The index
is measured across the full requested event count; reference full-run time is
projected from a deterministic evenly-spaced sample and clearly labelled.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from technical_event_descriptive_evidence import (
    TIMEFRAME_MS,
    _level_hit,
    _time_to_level,
    parse_snapshot,
)


HASH = "b" * 64


def _snapshot(candle_count: int) -> dict[str, Any]:
    interval = TIMEFRAME_MS["1h"]
    candles = []
    for index in range(candle_count):
        open_ms = index * interval
        close = 30_000.0 + 500.0 * math.sin(index / 97.0)
        candles.append(
            {
                "openTimeMs": open_ms,
                "closeTimeMs": open_ms + interval - 1,
                "open": close,
                "high": close + 50.0,
                "low": close - 50.0,
                "close": close,
                "volume": 1.0,
                "availableTimeMs": open_ms + interval - 1,
                "finalized": True,
                "context": {"trend": "sideways", "volatility": "normal"},
            }
        )
    return {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "lineage": {"source": "benchmark", "sourceVersion": "v1", "contentSha256": HASH},
        "candles": candles,
        "events": [],
    }


def _linear_reference(dataset: Any, decision_index: int, level: dict[str, Any], cutoff_ms: int) -> Any:
    for index in range(decision_index + 1, len(dataset.candles)):
        candle = dataset.candles[index]
        if candle.available_ms > cutoff_ms:
            break
        if candle.open_ms - dataset.candles[index - 1].open_ms != dataset.interval_ms:
            break
        if _level_hit(candle, level):
            return index
    return None


def run_benchmark(candle_count: int, event_count: int, reference_sample_size: int) -> dict[str, Any]:
    raw = _snapshot(candle_count)
    cutoff = raw["candles"][-1]["closeTimeMs"]
    build_started = time.perf_counter()
    dataset = parse_snapshot(raw, cutoff)
    index_build_seconds = time.perf_counter() - build_started

    event_count = min(event_count, candle_count - 1)
    decisions = [index * (candle_count - 1) // event_count for index in range(event_count)]
    reference_sample_size = min(reference_sample_size, event_count)
    sampled = [decisions[index * event_count // reference_sample_size] for index in range(reference_sample_size)]
    level = {"operator": "at_or_above", "price": 1_000_000.0}

    indexed_started = time.perf_counter()
    indexed_results = [_time_to_level(dataset, decision, level, cutoff) for decision in decisions]
    indexed_seconds = time.perf_counter() - indexed_started

    reference_started = time.perf_counter()
    reference_results = [_linear_reference(dataset, decision, level, cutoff) for decision in sampled]
    reference_sample_seconds = time.perf_counter() - reference_started
    if any(result is not None for result in indexed_results) or any(result is not None for result in reference_results):
        raise AssertionError("never-crossed benchmark level unexpectedly crossed")

    indexed_per_query = indexed_seconds / event_count
    reference_per_query = reference_sample_seconds / reference_sample_size
    projected_reference_seconds = reference_per_query * event_count
    return {
        "schema": "btc-technical-lifecycle-lookup-benchmark/v1",
        "workload": {
            "timeframe": "1h",
            "candles": candle_count,
            "indexedQueries": event_count,
            "referenceSampleQueries": reference_sample_size,
            "case": "never-crossed level in one contiguous segment",
        },
        "indexBuildSeconds": index_build_seconds,
        "indexedAllQueriesSeconds": indexed_seconds,
        "referenceSampleSeconds": reference_sample_seconds,
        "projectedReferenceAllQueriesSeconds": projected_reference_seconds,
        "indexedMicrosecondsPerQuery": indexed_per_query * 1_000_000.0,
        "referenceMicrosecondsPerQuery": reference_per_query * 1_000_000.0,
        "perQuerySpeedup": reference_per_query / indexed_per_query,
        "parity": True,
        "projectionDisclosure": "reference full-run time is projected from deterministic evenly-spaced queries",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark indexed lifecycle first-crossing queries")
    parser.add_argument("--candles", type=int, default=56_884)
    parser.add_argument("--events", type=int, default=10_307)
    parser.add_argument("--reference-sample-size", type=int, default=300)
    parser.add_argument("--minimum-speedup", type=float, default=20.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.candles < 2 or args.events < 1 or args.reference_sample_size < 1:
        parser.error("candles >= 2, events >= 1 and reference-sample-size >= 1 are required")
    result = run_benchmark(args.candles, args.events, args.reference_sample_size)
    result["minimumRequiredSpeedup"] = args.minimum_speedup
    result["passed"] = result["perQuerySpeedup"] >= args.minimum_speedup
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
