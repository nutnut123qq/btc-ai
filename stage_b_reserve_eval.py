#!/usr/bin/env python3
"""Stage B — one-shot confirmatory gate on the no-peek reserve.

Protocol (frozen 2026-10-23, before any reserve scoring): candidate = the
pre-selected v3 causal HGB (+ PriorOnlyCalibrator) only; fit/calibration come
exclusively from the hash-bound 2026-10-05 v3 freeze snapshot minus a fixed
list of 10 rows that break the label contract (last 120 remaining rows
calibrate); reserve = live rows with decision_time > FROZEN_CUTOFF_MS; gate =
rolling_retrainer.PROMOTION_THRESHOLDS + v3 baseline intervals; minimum 150
reserve rows. Labels must be PriceTargets.TargetDirection4h and are re-derived
from Klines closes (close-to-close, +-0.6% dead zone) before any fit.

Usage:
    python stage_b_reserve_eval.py --count-only --bundle <freeze>.manifest.json
    python stage_b_reserve_eval.py --score --bundle <freeze>.manifest.json --out <new>.json

--count-only never fits or scores and never loads a reserve label. --score is
the single explicit scoring action; it refuses to run if a Stage B report
already exists next to --out. Exit: 0 PASSED, 10 FAILED, 20 NOT_READY,
21 READY_BY_COUNT_ONLY, 30 EVIDENCE_ERROR (hash/lineage/runtime/schema; no
verdict, no report). Exit 1 = crash and 2 = CLI usage error, neither a verdict.
"""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg2
import sklearn

sys.path.insert(0, str(Path(__file__).parent))
import ml_evidence_walkforward as v2
import ml_evidence_v3 as v3
from db_config import get_db_connection
from rolling_retrainer import (
    LABEL_REMAP,
    PROMOTION_THRESHOLDS,
    evaluate_probabilities,
    majority_baseline_metrics,
)

PROTOCOL_VERSION = "btc-4h-stage-b-confirmatory/2026-10-23-v2"
FROZEN_CUTOFF_MS = 1_791_151_530_514  # 2026-10-04T22:05:30Z (2026-10-05 05:05 +07) — v3 snapshot cutoff
FROZEN_BUNDLE_MANIFEST_SHA256 = "7d662f64c72a45634dc8f3eaa3b5f8a9793b724ec1abf113d3b8cc726c99af19"
FROZEN_DATASET_SHA256 = "e876003bc8907aacf58a669a16867f806720b119dc2162259b9b8f0c5975bb95"
FROZEN_ROWS = 14_674
# Protocol v2 (2026-10-23): frozen rows whose stored label breaks the 4h close-to-close contract,
# found by lineage before any reserve scoring. Excluded from fit, calibration and baseline history.
LABEL_EXCLUDED_DECISION_MS = (
    1_582_113_600_000,  # 2020-02-19T12:00Z — next 4h Kline missing; stored label is an 8h move
    1_789_416_000_000, 1_789_430_400_000, 1_789_444_800_000,  # 2026-09-14T20:00Z ..
    1_789_459_200_000, 1_789_473_600_000, 1_789_488_000_000,  # .. stored TargetReturn4h differs
    1_789_502_400_000, 1_789_516_800_000, 1_789_531_200_000,  # .. from Klines .. 2026-09-16T04:00Z
)
STAGE_B_INPUT_ROWS = 14_664
STAGE_B_INPUT_SHA256 = "3e2e73b48831ca940064d6871c32856fd0b2d74d024e3e3b173eb7ac88543f74"
LABEL_SOURCE = "PriceTargets.TargetDirection4h"
DIRECTION_THRESHOLD_PCT = 0.6         # backend MlDatasetService.DirectionThreshold("4h")
CANDIDATE = v3.FULL_MODEL             # hist_gradient_boosting — the only Stage B candidate
WINDOW_SIZE = 5
CALIBRATION_ROWS = 120                # V3Config.calibration_rows of the freeze
MIN_RESERVE_ROWS = 150                # protocol minimum, not configurable
BLOCK_SIZES = (6, 12, 24)
BOOTSTRAP_SAMPLES = 1000
SEED = 42

EXIT_PASSED, EXIT_FAILED, EXIT_NOT_READY, EXIT_READY_BY_COUNT, EXIT_EVIDENCE = 0, 10, 20, 21, 30
MISSING = 99  # sentinel label for a signal bar whose closes are not both present

RESERVE_COUNT_SQL = """
    SELECT count(*), min(w."WindowEndMs"), max(w."WindowEndMs")
    FROM "WindowClassificationDatasets" AS w
    INNER JOIN "PriceTargets" AS p
      ON p."Symbol" = w."Symbol" AND p."Timeframe" = w."Timeframe" AND p."OpenTimeMs" = w."WindowEndMs"
    WHERE w."Symbol" = 'BTCUSDT' AND w."Timeframe" = '4h' AND w."WindowSize" = %(ws)s
      AND w."Horizon" = '4h' AND w."FeatureDim" = %(dim)s
      AND p."TargetDirection4h" IS NOT NULL
      AND w."WindowEndMs" + 2 * %(i)s <= %(now)s
      AND w."WindowEndMs" + %(i)s > %(cut)s
"""


class EvidenceError(RuntimeError):
    """Evidence is broken or missing — never a statistical verdict."""


def _runtime() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikitLearn": sklearn.__version__,
        "psycopg2": psycopg2.__version__,
    }


def _subset(data: v2.BenchmarkData, mask: np.ndarray) -> v2.BenchmarkData:
    return v2.BenchmarkData(data.features[mask], data.labels[mask], data.decision_times_ms[mask],
                            data.label_available_times_ms[mask], data.last_returns[mask])


def load_frozen_input(bundle_path: Path, *, recompute_statistics: bool) -> tuple[v2.BenchmarkData, dict]:
    """Verify the freeze bundle end to end and return its snapshot + research manifest."""
    try:
        bundle = v3.verify_evidence_bundle(bundle_path, recompute_statistics=recompute_statistics)
    except (OSError, ValueError, KeyError) as exc:
        raise EvidenceError(f"freeze bundle verification failed: {exc}") from exc
    if bundle["bundleManifestSha256"] != FROZEN_BUNDLE_MANIFEST_SHA256:
        raise EvidenceError("bundle is not the pinned 2026-10-05 freeze")
    report = json.loads((bundle_path.parent / bundle["artifacts"]["report"]["file"]).read_text(encoding="utf-8"))
    manifest = report["researchManifest"]
    prov = manifest["dataProvenance"]
    params = manifest["parameters"]
    checks = {
        "decisionCutoffMs": manifest["decisionCutoffMs"] == FROZEN_CUTOFF_MS,
        "labelSource": prov["labelSource"] == LABEL_SOURCE,
        "datasetSha256": prov["datasetSha256"] == FROZEN_DATASET_SHA256,
        "rowCount": prov["rowCount"] == FROZEN_ROWS,
        "featureDim": prov["featureDim"] == WINDOW_SIZE * len(v2.CAUSAL_FEATURE_NAMES),
        "calibrationRows": params["calibration_rows"] == CALIBRATION_ROWS,
        "randomState": params["random_state"] == SEED,
        "windowSize": params["window_size"] == WINDOW_SIZE,
    }
    if not all(checks.values()):
        raise EvidenceError(f"freeze manifest violates protocol: {[k for k, ok in checks.items() if not ok]}")
    runtime = _runtime()
    if runtime != manifest["runtimeDependencies"]:
        raise EvidenceError(f"runtime {runtime} != freeze runtime {manifest['runtimeDependencies']}")
    snapshot = bundle_path.parent / bundle["artifacts"]["datasetSnapshot"]["file"]
    frozen = v3.load_v3_npz(snapshot, decision_cutoff_ms=FROZEN_CUTOFF_MS, window_size=WINDOW_SIZE)
    if v2.dataset_sha256(frozen) != FROZEN_DATASET_SHA256 or len(frozen.labels) != FROZEN_ROWS:
        raise EvidenceError("frozen snapshot content does not match pinned dataset hash")
    return frozen, manifest


def stage_b_input(frozen: v2.BenchmarkData) -> v2.BenchmarkData:
    """Frozen snapshot minus exactly the pinned label-contract exclusions."""
    excluded = np.isin(frozen.decision_times_ms, LABEL_EXCLUDED_DECISION_MS)
    if int(excluded.sum()) != len(set(LABEL_EXCLUDED_DECISION_MS)):
        raise EvidenceError("pinned label exclusions are not all present in the frozen snapshot")
    used = _subset(frozen, ~excluded)
    if len(used.labels) != STAGE_B_INPUT_ROWS or v2.dataset_sha256(used) != STAGE_B_INPUT_SHA256:
        raise EvidenceError("Stage B input (snapshot minus exclusions) does not match its pinned hash")
    return used


def drift_check(frozen: v2.BenchmarkData, live_pre: v2.BenchmarkData) -> dict:
    """Live pre-cutoff rows must reproduce the freeze exactly; extra (late) rows are excluded, not used."""
    if live_pre.features.shape[1] != frozen.features.shape[1]:
        raise EvidenceError("live feature schema differs from frozen snapshot")
    common, fi, li = np.intersect1d(frozen.decision_times_ms, live_pre.decision_times_ms, return_indices=True)
    if len(common) != len(frozen.labels):
        raise EvidenceError(f"{len(frozen.labels) - len(common)} frozen rows missing from live DB")
    if not (np.array_equal(frozen.features[fi], live_pre.features[li])
            and np.array_equal(frozen.labels[fi], live_pre.labels[li])):
        raise EvidenceError("live pre-cutoff rows drifted from frozen snapshot")
    late = np.setdiff1d(live_pre.decision_times_ms, frozen.decision_times_ms)
    return {"frozenRowsMatched": int(len(common)), "lateRowsExcludedFromFitCalibration": int(len(late)),
            "lateRowDecisionTimesMs": [int(t) for t in late]}


def ctc_direction(return_pct: np.ndarray) -> np.ndarray:
    # Mirrors MlDatasetService.Direction: strictly beyond the band is a move; the band edge is sideways.
    return np.where(return_pct > DIRECTION_THRESHOLD_PCT, 1,
                    np.where(return_pct < -DIRECTION_THRESHOLD_PCT, -1, 0)).astype(np.int8)


def label_lineage(decision_times_ms: np.ndarray, labels: np.ndarray, closes: dict) -> dict:
    """Re-derive TargetDirection4h from the signal bar close and the very next 4h bar close."""
    signal_open = decision_times_ms - v2.INTERVAL_MS
    c0 = np.array([closes.get(int(t), np.nan) for t in signal_open], dtype=np.float64)
    c1 = np.array([closes.get(int(t) + v2.INTERVAL_MS, np.nan) for t in signal_open], dtype=np.float64)
    present = np.isfinite(c0) & np.isfinite(c1) & (c0 > 0)
    expected = np.full(len(labels), MISSING, dtype=np.int8)
    expected[present] = ctc_direction((c1[present] - c0[present]) / c0[present] * 100.0)
    missing = int((~present).sum())
    mismatches = int((expected[present] != labels[present]).sum())
    return {"labelSource": LABEL_SOURCE, "rule": f"(close[t+4h]-close[t])/close[t]*100 vs +-{DIRECTION_THRESHOLD_PCT}",
            "rows": int(len(labels)), "missingBars": missing, "mismatches": mismatches,
            "complete": len(labels) > 0 and missing == 0 and mismatches == 0}


def fetch_closes(open_times_ms: np.ndarray) -> dict:
    wanted = sorted({int(t) for t in open_times_ms} | {int(t) + v2.INTERVAL_MS for t in open_times_ms})
    conn = get_db_connection()
    try:
        conn.set_session(readonly=True, autocommit=False)
        with conn.cursor() as cur:
            cur.execute('SELECT "OpenTimeMs", "Close" FROM "Klines" WHERE "Symbol" = %s AND "Timeframe" = %s '
                        'AND "OpenTimeMs" = ANY(%s)', ("BTCUSDT", "4h", wanted))
            rows = cur.fetchall()
        conn.rollback()
    finally:
        conn.close()
    return {int(t): float(c) for t, c in rows}


def count_reserve_rows(now_ms: int) -> dict:
    """Same cohort/predicates as v2.load_btc_4h_benchmark, but selects no feature or label value."""
    conn = get_db_connection()
    try:
        conn.set_session(readonly=True, autocommit=False)
        with conn.cursor() as cur:
            cur.execute(RESERVE_COUNT_SQL, {"ws": WINDOW_SIZE, "dim": WINDOW_SIZE * v2.STORED_FEATURES_PER_BAR,
                                            "i": v2.INTERVAL_MS, "now": now_ms, "cut": FROZEN_CUTOFF_MS})
            n, first, last = cur.fetchone()
        conn.rollback()
    finally:
        conn.close()
    return {"rows": int(n), "firstDecisionMs": first and first + v2.INTERVAL_MS,
            "lastDecisionMs": last and last + v2.INTERVAL_MS}


def _require_lineage(name: str, data: v2.BenchmarkData) -> dict:
    lineage = label_lineage(data.decision_times_ms, data.labels, fetch_closes(data.decision_times_ms - v2.INTERVAL_MS))
    if not lineage["complete"]:
        raise EvidenceError(f"{name} label lineage incomplete: {lineage}")
    return lineage


def _reserve_gate(cand: dict, base: dict) -> dict:
    lim = PROMOTION_THRESHOLDS
    checks = {
        "sample count": cand["samples"] >= lim["minimum_samples"],
        "class support": min(cand["class_counts"]) >= lim["minimum_class_samples"],
        "macro F1": cand["f1_macro"] >= lim["minimum_macro_f1"],
        "balanced accuracy": cand["balanced_accuracy"] >= lim["minimum_balanced_accuracy"],
        "MCC": cand["mcc"] >= lim["minimum_mcc"],
        "per-class F1": min(cand["f1_per_class"].values()) >= lim["minimum_per_class_f1"],
        "ECE": cand["ece"] <= lim["maximum_ece"],
        "Brier improvement": cand["brier_score"] <= base["brier_score"] - lim["minimum_loss_improvement"],
        "log-loss improvement": cand["log_loss"] <= base["log_loss"] - lim["minimum_loss_improvement"],
    }
    return {"passed": all(checks.values()), "checks": checks,
            "failures": [k for k, ok in checks.items() if not ok]}


def _baseline_intervals(labels_map: np.ndarray, cand_probs: np.ndarray,
                        baseline_probs: dict, alpha: float) -> dict:
    out = {}
    for name, probs in baseline_probs.items():
        entry = {}
        for metric, row_fn in (("brier", v2._brier_rows), ("logLoss", v3._log_loss_rows)):
            lift = row_fn(labels_map, probs) - row_fn(labels_map, cand_probs)
            intervals = {}
            for bs in BLOCK_SIZES:
                lo, hi = v2._paired_block_interval(
                    lift, block_size=bs, bootstrap_samples=BOOTSTRAP_SAMPLES,
                    random_state=SEED + bs, alpha=alpha)
                intervals[str(bs)] = [lo, hi]
            entry[metric] = {
                "meanLift": float(np.mean(lift)),
                "intervals": intervals,
                "robustAcrossBlockSizes": all(lo > 0 for lo, _ in intervals.values()),
            }
        out[name] = entry
    return out


def score(frozen: v2.BenchmarkData, reserve: v2.BenchmarkData) -> dict:
    """Fit on input[:-120], calibrate on input[-120:], score reserve once (input = snapshot minus exclusions)."""
    data = v2.BenchmarkData(*(np.concatenate([a, b]) for a, b in (
        (frozen.features, reserve.features), (frozen.labels, reserve.labels),
        (frozen.decision_times_ms, reserve.decision_times_ms),
        (frozen.label_available_times_ms, reserve.label_available_times_ms),
        (frozen.last_returns, reserve.last_returns))))
    data.validate()
    n = len(frozen.labels)
    fit_idx, cal_idx = np.arange(n - CALIBRATION_ROWS), np.arange(n - CALIBRATION_ROWS, n)
    reserve_idx = np.arange(n, len(data.labels))
    columns = v3._feature_columns(WINDOW_SIZE, None)
    cand_probs = v3._fit_calibrated_hgb(data, fit_idx, cal_idx, reserve_idx, columns, SEED)

    y_map = np.array([LABEL_REMAP[int(v)] for v in data.labels[reserve_idx]])
    cand = evaluate_probabilities(y_map, cand_probs)
    base = majority_baseline_metrics(np.array([LABEL_REMAP[int(v)] for v in data.labels[fit_idx]]), y_map)
    gate = _reserve_gate(cand, base)
    config = v3.V3Config()
    base_probs = v3._baseline_probabilities(data, reserve_idx, config)
    alpha = config.familywise_alpha / (len(base_probs) * 2)
    comparisons = _baseline_intervals(data.labels[reserve_idx], cand_probs, base_probs, alpha)
    intervals_positive = all(m["robustAcrossBlockSizes"] for comp in comparisons.values() for m in comp.values())
    return {
        "status": "STAGE_B_PASSED" if (gate["passed"] and intervals_positive) else "STAGE_B_FAILED",
        "split": {"fitRows": int(len(fit_idx)), "calibrationRows": int(len(cal_idx)),
                  "reserveRows": int(len(reserve_idx)), "featureColumns": int(len(columns))},
        "candidate": cand,
        "majorityBaseline": base,
        "promotionGate": gate,
        "baselineIntervalsPositive": intervals_positive,
        "baselineComparisons": comparisons,
    }


def _emit(payload: dict, code: int) -> int:
    print(json.dumps(payload, indent=2, default=str))
    return code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--count-only", action="store_true", help="preflight + count; never fits or scores")
    mode.add_argument("--score", action="store_true", help="the one-shot confirmatory scoring run")
    ap.add_argument("--bundle", required=True, type=Path, help="freeze <sha>.manifest.json path")
    ap.add_argument("--out", type=Path, help="new report path (required with --score)")
    args = ap.parse_args(argv)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    head = {"protocolVersion": PROTOCOL_VERSION, "checkedAtMs": now_ms, "cutoffMs": FROZEN_CUTOFF_MS,
            "minReserveRows": MIN_RESERVE_ROWS, "candidate": CANDIDATE}

    if args.score:
        if args.out is None:
            ap.error("--score requires --out")
        prior = sorted(args.out.parent.glob("stage_b_report*.json")) if args.out.parent.is_dir() else []
        if args.out.exists() or prior:
            return _emit({**head, "status": "EVIDENCE_ERROR",
                          "reason": f"a Stage B report already exists; scoring is one-shot: {prior or [args.out]}"},
                         EXIT_EVIDENCE)
    count = count_reserve_rows(now_ms) if args.count_only else None  # metadata only, safe before evidence checks
    try:
        frozen, manifest = load_frozen_input(args.bundle, recompute_statistics=args.score)
        used = stage_b_input(frozen)
        if args.count_only:
            live_pre = v2.load_btc_4h_benchmark(decision_cutoff_ms=FROZEN_CUTOFF_MS, window_size=WINDOW_SIZE)
            drift = drift_check(frozen, live_pre)
            frozen_lineage = _require_lineage("frozen", used)
            ready = count["rows"] >= MIN_RESERVE_ROWS
            return _emit({**head, "status": "READY_BY_COUNT_ONLY" if ready else "NOT_READY",
                          "reserve": count, "missingToMin": max(0, MIN_RESERVE_ROWS - count["rows"]),
                          "drift": drift, "labelExcludedRows": len(LABEL_EXCLUDED_DECISION_MS),
                          "labelLineage": {"frozen": frozen_lineage}},
                         EXIT_READY_BY_COUNT if ready else EXIT_NOT_READY)

        live = v2.load_btc_4h_benchmark(decision_cutoff_ms=now_ms, window_size=WINDOW_SIZE)
        reserve = _subset(live, live.decision_times_ms > FROZEN_CUTOFF_MS)
        if len(reserve.labels) < MIN_RESERVE_ROWS:
            return _emit({**head, "status": "RESERVE_NOT_READY", "reserveRows": int(len(reserve.labels))},
                         EXIT_NOT_READY)
        drift = drift_check(frozen, _subset(live, live.label_available_times_ms <= FROZEN_CUTOFF_MS))
        lineage = {"frozen": _require_lineage("frozen", used), "reserve": _require_lineage("reserve", reserve)}
    except (EvidenceError, ValueError, KeyError) as exc:  # data/evidence defects, never a verdict
        return _emit({**head, "status": "EVIDENCE_ERROR", "reason": str(exc), "reserve": count}, EXIT_EVIDENCE)

    here = Path(__file__).resolve()
    result = score(used, reserve)
    report = {
        **head,
        "status": result.pop("status"),
        "frozenInput": {"bundleManifestSha256": FROZEN_BUNDLE_MANIFEST_SHA256, "datasetSha256": FROZEN_DATASET_SHA256,
                        "researchManifestSha256": manifest["manifestSha256"], "rows": FROZEN_ROWS,
                        "stageAGit": manifest["codeProvenance"]["git"],
                        "labelExcludedDecisionTimesMs": list(LABEL_EXCLUDED_DECISION_MS),
                        "stageBInputRows": STAGE_B_INPUT_ROWS, "stageBInputSha256": STAGE_B_INPUT_SHA256},
        "drift": drift,
        "reserve": {"rows": int(len(reserve.labels)), "firstDecisionMs": int(reserve.decision_times_ms[0]),
                    "lastDecisionMs": int(reserve.decision_times_ms[-1]),
                    "datasetSha256": v2.dataset_sha256(reserve)},
        "labelLineage": lineage,
        "code": {"git": v2._git_provenance(here.parent), "stageBSha256": v2._file_hash(here),
                 "evaluatorSha256": v2._file_hash(here.parent / "ml_evidence_walkforward.py"),
                 "v3EvaluatorSha256": v2._file_hash(here.parent / "ml_evidence_v3.py"),
                 "rollingRetrainerSha256": v2._file_hash(here.parent / "rolling_retrainer.py")},
        "runtime": _runtime(),
        **result,
        "scoredOnce": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "x", encoding="utf-8") as handle:  # "x": never overwrite a prior report
        handle.write(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ("status", "reserve", "labelLineage")}, indent=2, default=str))
    print(f"Report -> {args.out}")
    return EXIT_PASSED if report["status"] == "STAGE_B_PASSED" else EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
