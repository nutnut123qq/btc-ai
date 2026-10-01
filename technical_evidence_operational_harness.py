#!/usr/bin/env python3
"""Deterministic, production-DB-free reliability harness for technical evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import tempfile
import time
import tracemalloc
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from run_technical_evidence_pipeline import (
    RUN_POINTER_FILE,
    EvaluationConfig,
    _publish_staged_bundle,
    canonical_json,
    read_verified_success,
    run_pipeline,
    sha256_bytes,
)
from technical_event_descriptive_evidence import (
    TIMEFRAME_MS,
    build_postgresql_snapshot_from_rows,
    evaluate,
)


HARNESS_SCHEMA = "btc-technical-evidence-operational-harness/v1"
FIXED_NOW = datetime(2026, 9, 22, 3, 0, tzinfo=timezone.utc)


def _snapshot(timeframe: str, cutoff_ms: int, bars: int) -> dict[str, Any]:
    interval = TIMEFRAME_MS[timeframe]
    rows = []
    for index in range(bars):
        base = 100_000 + 13 * index + ((index % 12) - 6) * 200
        close = base + ((index % 5) - 2) * 40
        rows.append({
            "OpenTimeMs": index * interval,
            "CloseTimeMs": (index + 1) * interval - 1,
            "Open": float(base),
            "High": float(max(base, close) + 100 + (index % 3) * 10),
            "Low": float(min(base, close) - 110 - (index % 4) * 10),
            "Close": float(close),
            "Volume": float((1_000 + (index % 7) * 100) * (4 if index % 29 == 0 else 1)),
        })
    return build_postgresql_snapshot_from_rows(
        timeframe,
        cutoff_ms,
        rows,
        [],
        smc_columns=[],
    )


def run_harness(
    *,
    bars: int = 240,
    iterations: int = 2,
    bootstrap_samples: int = 5,
    max_peak_memory_mib: int = 512,
) -> dict[str, Any]:
    if bars < 80 or iterations < 2 or bootstrap_samples <= 0:
        raise ValueError("harness requires bars>=80, iterations>=2 and positive bootstrap samples")
    cutoff = 10**15
    snapshots = {timeframe: _snapshot(timeframe, cutoff, bars) for timeframe in TIMEFRAME_MS}
    timings: list[float] = []
    semantic_hashes: list[str] = []
    tracemalloc.start()
    for _ in range(iterations):
        started = time.perf_counter()
        _, _, report = evaluate(
            snapshots["1h"],
            EvaluationConfig(cutoff_ms=cutoff, bootstrap_samples=bootstrap_samples),
        )
        timings.append(time.perf_counter() - started)
        semantic_hashes.append(sha256_bytes(canonical_json(report).encode()))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if len(set(semantic_hashes)) != 1:
        raise RuntimeError("repeatable benchmark produced different semantic hashes")
    if peak > max_peak_memory_mib * 1024 * 1024:
        raise RuntimeError("repeatable benchmark exceeded bounded peak-memory budget")

    with tempfile.TemporaryDirectory(prefix="btc-technical-evidence-harness-") as raw:
        directory = Path(raw) / "published"
        run_pipeline(
            directory,
            forced_cutoff_ms=cutoff,
            bootstrap_samples=bootstrap_samples,
            now=lambda: FIXED_NOW,
            snapshot_loader=lambda timeframe, value: snapshots[timeframe],
        )
        pointer_before = (directory / RUN_POINTER_FILE).read_bytes()
        calls = 0

        def fail_second(paths: Any, output_dir: Path) -> Path:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected second-bundle publication failure")
            return _publish_staged_bundle(paths, output_dir)

        failure_observed = False
        try:
            run_pipeline(
                directory,
                forced_cutoff_ms=cutoff,
                bootstrap_samples=bootstrap_samples,
                dedup_bars=5,
                now=lambda: FIXED_NOW + timedelta(minutes=1),
                snapshot_loader=lambda timeframe, value: snapshots[timeframe],
                bundle_publisher=fail_second,
            )
        except OSError:
            failure_observed = True
        pointer_preserved = (directory / RUN_POINTER_FILE).read_bytes() == pointer_before
        prior_still_valid = read_verified_success(directory)["valid"] is True
        run_pipeline(
            directory,
            forced_cutoff_ms=cutoff,
            bootstrap_samples=bootstrap_samples,
            dedup_bars=5,
            now=lambda: FIXED_NOW + timedelta(minutes=2),
            snapshot_loader=lambda timeframe, value: snapshots[timeframe],
        )
        retry = read_verified_success(directory)

        restored = Path(raw) / "restored"
        shutil.copytree(directory, restored)
        restore = read_verified_success(restored)
        scenarios = {
            "partialPublishFailureInjected": failure_observed,
            "activePointerPreservedAfterFailure": pointer_preserved,
            "priorTrioValidAfterFailure": prior_still_valid,
            "retryPublishedExactTrio": [item["timeframe"] for item in retry["bundles"]] == ["1h", "4h", "1d"],
            "copiedArtifactRestoreVerified": restore["runIndexSha256"] == retry["runIndexSha256"],
        }
    if not all(scenarios.values()):
        raise RuntimeError("one or more operational harness scenarios failed")
    return {
        "schema": HARNESS_SCHEMA,
        "status": "passed",
        "workload": {
            "barsPerTimeframe": bars,
            "iterations": iterations,
            "bootstrapSamples": bootstrap_samples,
            "timeframes": ["1h", "4h", "1d"],
            "syntheticOnly": True,
        },
        "repeatability": {
            "semanticReportSha256": semantic_hashes[0],
            "allIterationsEqual": True,
            "medianEvaluationSeconds": statistics.median(timings),
            "maximumEvaluationSeconds": max(timings),
            "peakTracedMemoryBytes": peak,
            "peakMemoryLimitBytes": max_peak_memory_mib * 1024 * 1024,
        },
        "scenarios": scenarios,
        "limitations": [
            "Synthetic fixtures validate protocol behavior, not production database capacity or PostgreSQL restore correctness.",
            "tracemalloc measures Python allocations and does not include every native-library or operating-system allocation.",
            "Artifact copy verification is not a substitute for an isolated pg_restore drill of the source database.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Technical evidence operational reliability harness")
    parser.add_argument("--bars", type=int, default=240)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=5)
    parser.add_argument("--max-peak-memory-mib", type=int, default=512)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = run_harness(
        bars=args.bars,
        iterations=args.iterations,
        bootstrap_samples=args.bootstrap_samples,
        max_peak_memory_mib=args.max_peak_memory_mib,
    )
    content = (canonical_json(report) + "\n").encode()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / f"{sha256_bytes(content)}.operational-harness.json"
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError("conflicting operational harness output")
        path.write_bytes(content)
        report = {**report, "outputFile": str(path), "outputSha256": sha256_bytes(content)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
