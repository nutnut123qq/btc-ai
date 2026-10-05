#!/usr/bin/env python3
"""Stage B — one-shot confirmatory gate on the no-peek reserve.

Fit/calibrate strictly on rows whose label was available by the frozen v3
snapshot cutoff, then score the post-cutoff reserve exactly once and apply
the promotion gate (rolling_retrainer.PROMOTION_THRESHOLDS) plus the v3
selection-aware baseline comparisons on the reserve.

Refuses to run before the reserve reaches the protocol minimum so the
confirmatory row set is never peeked at incrementally.

Usage:
    python stage_b_reserve_eval.py [--min-reserve 150] [--strict]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ml_evidence_walkforward as v2
import ml_evidence_v3 as v3
from rolling_retrainer import (
    LABEL_REMAP,
    PROMOTION_THRESHOLDS,
    evaluate_probabilities,
    majority_baseline_metrics,
)

FROZEN_CUTOFF_MS = 1_791_151_530_514  # 2026-10-05T04:45:30Z — v3 snapshot cutoff
CALIBRATION_ROWS = 120                # V3Config.calibration_rows default
MIN_RESERVE_ROWS = 150                # protocol minimum
STRICT_RESERVE_ROWS = 240             # stricter bound from evaluation-protocol
BLOCK_SIZES = (6, 12, 24)
BOOTSTRAP_SAMPLES = 1000
SEED = 42


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-reserve", type=int, default=MIN_RESERVE_ROWS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    data = v2.load_btc_4h_benchmark(decision_cutoff_ms=now_ms, window_size=5)

    # Frozen side: label knowable at the v3 cutoff. Reserve: decision after it.
    fit_pool = np.flatnonzero(data.label_available_times_ms <= FROZEN_CUTOFF_MS)
    reserve = np.flatnonzero(data.decision_times_ms > FROZEN_CUTOFF_MS)
    reserve_rows = int(len(reserve))
    if reserve_rows < args.min_reserve:
        print(json.dumps({
            "status": "RESERVE_NOT_READY",
            "reserveRows": reserve_rows,
            "required": args.min_reserve,
            "strictBound": STRICT_RESERVE_ROWS,
            "cutoffMs": FROZEN_CUTOFF_MS,
        }, indent=2))
        sys.exit(2)

    cal_idx = fit_pool[-CALIBRATION_ROWS:]
    fit_idx = fit_pool[:-CALIBRATION_ROWS]
    columns = v3._feature_columns(5, None)
    cand_probs = v3._fit_calibrated_hgb(data, fit_idx, cal_idx, reserve, columns, SEED)

    y_map = np.array([LABEL_REMAP[int(v)] for v in data.labels[reserve]])
    cand = evaluate_probabilities(y_map, cand_probs)
    base = majority_baseline_metrics(
        np.array([LABEL_REMAP[int(v)] for v in data.labels[fit_idx]]), y_map)

    gate = _reserve_gate(cand, base)

    config = v3.V3Config()
    base_probs = v3._baseline_probabilities(data, reserve, config)
    alpha = config.familywise_alpha / (len(base_probs) * 2)
    comparisons = _baseline_intervals(
        data.labels[reserve], cand_probs, base_probs, alpha)
    intervals_positive = all(
        m["robustAcrossBlockSizes"]
        for comp in comparisons.values() for m in comp.values())

    verdict = "STAGE_B_PASSED" if (gate["passed"] and intervals_positive) else "STAGE_B_FAILED"
    report = {
        "status": verdict,
        "cutoffMs": FROZEN_CUTOFF_MS,
        "reserveRows": reserve_rows,
        "strictBound": STRICT_RESERVE_ROWS,
        "candidate": cand,
        "majorityBaseline": base,
        "promotionGate": gate,
        "baselineIntervalsPositive": intervals_positive,
        "baselineComparisons": comparisons,
        "scoredOnce": True,
    }
    text = json.dumps(report, indent=2)
    out = args.out or f"stage_b_report_{datetime.now(timezone.utc):%Y%m%d}.json"
    Path(out).write_text(text, encoding="utf-8")
    print(text)
    print(f"Report -> {out}")
    sys.exit(0 if verdict == "STAGE_B_PASSED" else 1)


if __name__ == "__main__":
    main()
