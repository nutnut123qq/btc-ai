#!/usr/bin/env python3
"""Bounded production-shaped benchmark for statistical evidence assembly."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import tracemalloc
from pathlib import Path
from typing import Any

from technical_evidence_statistics import build_statistical_evidence, declared_event_families


HORIZONS = (1, 3, 6)
METRICS = ("forwardReturn", "mfe", "mae")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _summary(count: int, p_value: float) -> dict[str, Any]:
    return {
        "count": count,
        "mean": 0.0,
        "median": 0.0,
        "q25": 0.0,
        "q75": 0.0,
        "meanBlockBootstrapInterval": {"lower": -0.01, "upper": 0.01},
        "centeredNullTwoSidedPValue": p_value,
    }


def _fixture(row_count: int, event_type_count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    families, _ = declared_event_families()
    identities = [
        (module, event_type)
        for module, event_types in sorted(families.items())
        for event_type in event_types
    ][:event_type_count]
    if len(identities) != event_type_count:
        raise ValueError("event-type count exceeds the predeclared family")
    shared_control = {
        str(horizon): {metric: 0.0 for metric in METRICS}
        for horizon in HORIZONS
    }
    rows = []
    counts = [0] * event_type_count
    start_ms = 1_577_836_800_000
    interval = 3_600_000
    for index in range(row_count):
        family = index % event_type_count
        module, event_type = identities[family]
        counts[family] += 1
        value = ((index % 17) - 8) / 10_000.0
        event_horizons = {
            str(horizon): {
                "forwardReturn": value,
                "mfe": abs(value) + 0.001,
                "mae": -abs(value) - 0.001,
            }
            for horizon in HORIZONS
        }
        trend = "up" if index % 2 == 0 else "down"
        volatility = "normal" if index % 3 else "high"
        signature = f"trend={trend}|volatility={volatility}"
        rows.append(
            {
                "eventId": f"event-{index}",
                "module": module,
                "eventType": event_type,
                "status": "eligible",
                "decisionIndex": index * 2,
                "decisionCloseTimeMs": start_ms + index * interval,
                "context": {"trend": trend, "volatility": volatility},
                "contextSignature": signature,
                "horizons": event_horizons,
                "control": {
                    "status": "matched",
                    "outcomeAvailableBeforeEvent": True,
                    "contextSignature": signature,
                    "horizons": shared_control,
                },
            }
        )
    summaries = {
        identities[family][1]: {
            "horizons": {
                str(horizon): {
                    "metrics": {
                        metric: {"matchedControlDifference": _summary(counts[family], 0.25)}
                        for metric in METRICS
                    }
                }
                for horizon in HORIZONS
            }
        }
        for family in range(event_type_count)
    }
    return rows, summaries


def run_benchmark(row_count: int, event_type_count: int) -> dict[str, Any]:
    rows, summaries = _fixture(row_count, event_type_count)
    arguments = dict(
        timeframe="1h",
        horizons=HORIZONS,
        metrics=METRICS,
        block_size_events=8,
        bootstrap_samples=2_000,
        random_seed=42,
        sensitivity={
            "preRegisteredGrid": {
                "declaredGridSha256": "d" * 64,
                "executedVariantIds": ["fixed:a", "fixed:b"],
                "allExecutedVariantsRetained": True,
            }
        },
    )
    tracemalloc.start()
    started = time.perf_counter()
    first = build_statistical_evidence(rows, summaries, **arguments)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    first_sha = hashlib.sha256(_canonical(first).encode()).hexdigest()
    second_sha = hashlib.sha256(
        _canonical(build_statistical_evidence(rows, summaries, **arguments)).encode()
    ).hexdigest()
    return {
        "schema": "btc-technical-evidence-statistics-benchmark/v1",
        "workload": {
            "rows": row_count,
            "eventTypes": event_type_count,
            "hypotheses": len(first["hypotheses"]),
            "timeframe": "1h",
        },
        "elapsedSeconds": elapsed,
        "peakTracedBytes": peak,
        "reportSha256": first_sha,
        "deterministic": first_sha == second_sha,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark descriptive statistical evidence assembly")
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--event-types", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=45.0)
    parser.add_argument("--max-peak-memory-mib", type=float, default=512.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rows < 1 or args.event_types < 1 or args.event_types > args.rows:
        parser.error("require rows >= event-types >= 1")
    result = run_benchmark(args.rows, args.event_types)
    result["limits"] = {
        "maxSeconds": args.max_seconds,
        "maxPeakMemoryMiB": args.max_peak_memory_mib,
    }
    result["passed"] = bool(
        result["deterministic"]
        and result["elapsedSeconds"] <= args.max_seconds
        and result["peakTracedBytes"] <= args.max_peak_memory_mib * 1024 * 1024
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
