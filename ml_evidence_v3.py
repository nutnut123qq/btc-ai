#!/usr/bin/env python3
"""Reproducible, leakage-safe BTCUSDT 4h ML evidence bundle (schema v3).

This module deliberately does not register, promote, or deploy a model.  It emits
an immutable dataset snapshot, row-level out-of-sample predictions, a compact
report, and a content-addressed bundle manifest so an independent reviewer can
recompute every reported metric from frozen inputs and predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import ml_evidence_walkforward as v2


SCHEMA_VERSION = "btc-ml-evidence-bundle/v3"
SNAPSHOT_SCHEMA_VERSION = "btc-ml-evidence-snapshot/v3"
EVALUATOR_VERSION = "btc-4h-next-bar-ml-evidence-v3"
FULL_MODEL = "hist_gradient_boosting"
EXPANDING_PRIOR = "expanding_historical_class_prior"
EXPANDING_MAJORITY = "expanding_historical_majority"
RULE_BASELINES = ("momentum", "reversion")
RESEARCH_MANIFEST_KEYS = {
    "schemaVersion", "experiment", "symbol", "timeframe", "decisionCutoffMs",
    "parameters", "featureGroups", "dataProvenance", "codeProvenance",
    "runtimeDependencies", "sourceDisclosure", "manifestSha256",
}

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "price_returns": (
        "CloseZscore", "ClosePctChange1", "ClosePctChange4", "ClosePctChange24",
        "HighLowRangePct", "BodyPct", "UpperWickPct", "LowerWickPct",
    ),
    "momentum": ("Rsi14", "Rsi14Slope", "MacdNorm", "MacdSignalNorm", "MacdHistogramNorm"),
    "trend": ("Ema12Dist", "Ema26Dist", "Ema50Dist", "Ema200Dist", "Sma50Dist", "Sma200Dist"),
    "volatility": ("BollingerWidth", "BollingerPosition", "Atr14Pct"),
    "volume_flow": ("ObvEmaDist", "VwapDist", "RollingVwapDist", "VolumeZscore", "VolumeSma20Ratio", "TakerBuyRatio"),
    "pattern": ("RecentPatternEncoded",),
    "time": ("HourSin", "HourCos", "DayOfWeekSin", "DayOfWeekCos", "IsWeekend"),
}


@dataclass(frozen=True)
class V3Config:
    window_size: int = 5
    min_fit_rows: int = 500
    calibration_rows: int = 120
    test_rows: int = 120
    step_rows: int = 120
    label_horizon_bars: int = 1
    laplace_alpha: float = 1.0
    rule_confidence: float = 0.80
    reliability_bins: int = 10
    bootstrap_samples: int = 1000
    bootstrap_block_sizes_rows: tuple[int, ...] = (6, 12, 24)
    adaptive_prior_windows_rows: tuple[int, ...] = (180, 540)
    familywise_alpha: float = 0.05
    random_state: int = 42
    minimum_gate_samples: int = 240
    minimum_class_samples: int = 20

    def validate(self) -> None:
        positive = (
            "window_size", "min_fit_rows", "calibration_rows", "test_rows",
            "step_rows", "label_horizon_bars", "reliability_bins", "bootstrap_samples",
            "minimum_gate_samples", "minimum_class_samples",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.bootstrap_block_sizes_rows or min(self.bootstrap_block_sizes_rows) <= 0:
            raise ValueError("bootstrap block sizes must be positive")
        if not self.adaptive_prior_windows_rows or min(self.adaptive_prior_windows_rows) <= 0:
            raise ValueError("adaptive prior windows must be positive")
        if len(set(self.bootstrap_block_sizes_rows)) != len(self.bootstrap_block_sizes_rows):
            raise ValueError("bootstrap block sizes must be unique")
        if len(set(self.adaptive_prior_windows_rows)) != len(self.adaptive_prior_windows_rows):
            raise ValueError("adaptive prior windows must be unique")
        if not 1 / 3 < self.rule_confidence < 1:
            raise ValueError("rule_confidence must be between 1/3 and 1")
        if not 0 < self.familywise_alpha < 1:
            raise ValueError("familywise_alpha must be in (0, 1)")
        _validate_feature_groups()


@dataclass(frozen=True)
class V3Result:
    evaluation: dict[str, Any]
    prediction_rows: tuple[dict[str, Any], ...]


def _validate_data_schema(data: v2.BenchmarkData, window_size: int) -> None:
    data.validate()
    expected = window_size * len(v2.CAUSAL_FEATURE_NAMES)
    if data.features.shape[1] != expected:
        raise ValueError(
            f"causal feature matrix must have exactly {expected} columns for "
            f"window_size={window_size}; got {data.features.shape[1]}"
        )


def freeze_at_cutoff(
    data: v2.BenchmarkData,
    *,
    decision_cutoff_ms: int,
    window_size: int,
) -> v2.BenchmarkData:
    """Return the only rows whose decisions and realized labels existed by cutoff."""
    _validate_data_schema(data, window_size)
    cutoff = int(decision_cutoff_ms)
    mask = (data.decision_times_ms <= cutoff) & (data.label_available_times_ms <= cutoff)
    frozen = v2.BenchmarkData(
        data.features[mask],
        data.labels[mask],
        data.decision_times_ms[mask],
        data.label_available_times_ms[mask],
        data.last_returns[mask],
    )
    _validate_data_schema(frozen, window_size)
    return frozen


def _require_frozen_at_cutoff(
    data: v2.BenchmarkData,
    *,
    decision_cutoff_ms: int,
    window_size: int,
) -> None:
    frozen = freeze_at_cutoff(
        data,
        decision_cutoff_ms=decision_cutoff_ms,
        window_size=window_size,
    )
    if v2.dataset_sha256(frozen) != v2.dataset_sha256(data):
        raise ValueError("dataset contains decisions or labels beyond decision cutoff")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _validate_feature_groups() -> None:
    flattened = [name for names in FEATURE_GROUPS.values() for name in names]
    if len(flattened) != len(set(flattened)):
        raise ValueError("feature groups overlap")
    missing = set(v2.CAUSAL_FEATURE_NAMES).difference(flattened)
    extra = set(flattened).difference(v2.CAUSAL_FEATURE_NAMES)
    if missing or extra:
        raise ValueError(f"feature groups do not partition causal schema; missing={sorted(missing)}, extra={sorted(extra)}")


def _v2_config(config: V3Config) -> v2.EvaluatorConfig:
    return v2.EvaluatorConfig(
        window_size=config.window_size,
        min_fit_rows=config.min_fit_rows,
        calibration_rows=config.calibration_rows,
        test_rows=config.test_rows,
        step_rows=config.step_rows,
        label_horizon_bars=config.label_horizon_bars,
        laplace_alpha=config.laplace_alpha,
        rule_confidence=config.rule_confidence,
        reliability_bins=config.reliability_bins,
        block_size_rows=config.bootstrap_block_sizes_rows[0],
        bootstrap_samples=config.bootstrap_samples,
        familywise_alpha=config.familywise_alpha,
        random_state=config.random_state,
        minimum_gate_samples=config.minimum_gate_samples,
        minimum_class_samples=config.minimum_class_samples,
    )


def _adaptive_name(rows: int) -> str:
    return f"adaptive_class_prior_{rows}_rows"


def declared_baselines(config: V3Config) -> tuple[str, ...]:
    return (
        EXPANDING_PRIOR,
        EXPANDING_MAJORITY,
        *(_adaptive_name(rows) for rows in config.adaptive_prior_windows_rows),
        *RULE_BASELINES,
    )


def baseline_definitions(config: V3Config) -> dict[str, str]:
    definitions = {
        EXPANDING_PRIOR: "Laplace-smoothed class prior over every label available by decision time; expanding, not rolling",
        EXPANDING_MAJORITY: "Soft majority decision derived from the expanding historical class prior",
        "momentum": "Soft class rule from the sign of the final signal bar ClosePctChange1",
        "reversion": "Soft class rule opposite to the final signal bar ClosePctChange1",
    }
    for rows in config.adaptive_prior_windows_rows:
        definitions[_adaptive_name(rows)] = (
            f"Laplace-smoothed class prior over the most recent {rows} labels available by decision time"
        )
    return definitions


def _history_indices(data: v2.BenchmarkData, row_index: int, decision_time: int) -> np.ndarray:
    return np.flatnonzero(
        (np.arange(len(data.labels)) < row_index)
        & (data.label_available_times_ms <= decision_time)
    )


def _prior(labels: np.ndarray, alpha: float) -> np.ndarray:
    counts = np.asarray([np.sum(labels == value) for value in v2.CLASSES], dtype=np.float64)
    return (counts + alpha) / (counts.sum() + alpha * len(v2.CLASSES))


def _baseline_probabilities(
    data: v2.BenchmarkData,
    test_indices: np.ndarray,
    config: V3Config,
) -> dict[str, np.ndarray]:
    names = declared_baselines(config)
    outputs = {name: np.empty((len(test_indices), 3), dtype=np.float64) for name in names}
    for output_index, row_index in enumerate(test_indices):
        decision_time = int(data.decision_times_ms[row_index])
        known = _history_indices(data, int(row_index), decision_time)
        expanding = _prior(data.labels[known], config.laplace_alpha)
        outputs[EXPANDING_PRIOR][output_index] = expanding
        majority = int(v2.CLASSES[np.argmax(expanding)])
        outputs[EXPANDING_MAJORITY][output_index] = v2._soft_rule_probability(majority, config.rule_confidence)
        for rows in config.adaptive_prior_windows_rows:
            recent = known[-rows:]
            outputs[_adaptive_name(rows)][output_index] = _prior(data.labels[recent], config.laplace_alpha)
        last_return = data.last_returns[row_index]
        momentum = 1 if last_return > 0 else (-1 if last_return < 0 else 0)
        outputs["momentum"][output_index] = v2._soft_rule_probability(momentum, config.rule_confidence)
        outputs["reversion"][output_index] = v2._soft_rule_probability(-momentum, config.rule_confidence)
    return outputs


def _feature_columns(window_size: int, omitted_group: str | None = None) -> np.ndarray:
    omitted = set(FEATURE_GROUPS[omitted_group]) if omitted_group else set()
    per_bar = [index for index, name in enumerate(v2.CAUSAL_FEATURE_NAMES) if name not in omitted]
    stride = len(v2.CAUSAL_FEATURE_NAMES)
    return np.asarray(
        [bar * stride + offset for bar in range(window_size) for offset in per_bar],
        dtype=np.int64,
    )


def _fit_calibrated_hgb(
    data: v2.BenchmarkData,
    fit_indices: np.ndarray,
    calibration_indices: np.ndarray,
    test_indices: np.ndarray,
    columns: np.ndarray,
    random_state: int,
) -> np.ndarray:
    estimator = v2._build_estimator(FULL_MODEL, random_state)
    estimator.fit(data.features[np.ix_(fit_indices, columns)], data.labels[fit_indices])
    calibration_base = v2._aligned_probabilities(
        estimator, data.features[np.ix_(calibration_indices, columns)]
    )
    calibrator = v2.PriorOnlyCalibrator(random_state)
    calibrator.fit(calibration_base, data.labels[calibration_indices])
    test_base = v2._aligned_probabilities(estimator, data.features[np.ix_(test_indices, columns)])
    return calibrator.transform(test_base)


def _log_loss_rows(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    class_indices = np.searchsorted(v2.CLASSES, labels)
    selected = probabilities[np.arange(len(labels)), class_indices]
    return -np.log(np.clip(selected, 1e-12, 1.0))


def _select_rows_by_timestamps(
    data: v2.BenchmarkData,
    timestamps: np.ndarray,
) -> v2.BenchmarkData:
    indices = np.searchsorted(data.decision_times_ms, timestamps)
    if (
        np.any(indices >= len(data.decision_times_ms))
        or not np.array_equal(data.decision_times_ms[indices], timestamps)
    ):
        raise ValueError("evaluation timestamps are not an exact subset of the frozen dataset")
    return v2.BenchmarkData(
        data.features[indices],
        data.labels[indices],
        data.decision_times_ms[indices],
        data.label_available_times_ms[indices],
        data.last_returns[indices],
    )


def _statistical_evidence(
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    config: V3Config,
) -> dict[str, Any]:
    baseline_names = declared_baselines(config)
    ablation_names = tuple(f"hgb_without_{group}" for group in FEATURE_GROUPS)
    expected = {FULL_MODEL, *ablation_names, *baseline_names}
    if set(probabilities) != expected:
        raise ValueError("probability trials do not match the declared v3 family")
    metrics = {
        name: v2._metrics(labels, values, config.reliability_bins)
        for name, values in probabilities.items()
    }
    strongest_by_metric = {
        "brier": min(baseline_names, key=lambda name: metrics[name]["brier"]),
        "logLoss": min(baseline_names, key=lambda name: metrics[name]["logLoss"]),
    }
    candidate_losses = {
        "brier": v2._brier_rows(labels, probabilities[FULL_MODEL]),
        "logLoss": _log_loss_rows(labels, probabilities[FULL_MODEL]),
    }
    comparison_alpha = config.familywise_alpha / (len(baseline_names) * len(candidate_losses))
    baseline_comparisons: list[dict[str, Any]] = []
    for baseline_name in baseline_names:
        baseline_losses = {
            "brier": v2._brier_rows(labels, probabilities[baseline_name]),
            "logLoss": _log_loss_rows(labels, probabilities[baseline_name]),
        }
        metric_results: dict[str, Any] = {}
        for metric_name in ("brier", "logLoss"):
            lift = baseline_losses[metric_name] - candidate_losses[metric_name]
            intervals = []
            for block_size in config.bootstrap_block_sizes_rows:
                low, high = v2._paired_block_interval(
                    lift,
                    block_size=block_size,
                    bootstrap_samples=config.bootstrap_samples,
                    random_state=config.random_state
                    + block_size
                    + (1000 * baseline_names.index(baseline_name))
                    + (100_000 if metric_name == "logLoss" else 0),
                    alpha=comparison_alpha,
                )
                intervals.append({
                    "blockSizeRows": block_size,
                    "ci": [low, high],
                    "status": "positive" if low > 0 else ("negative" if high < 0 else "inconclusive"),
                })
            metric_results[metric_name] = {
                "meanLift": float(np.mean(lift)),
                "candidateBetterPointEstimate": metrics[FULL_MODEL][metric_name] < metrics[baseline_name][metric_name],
                "robustAcrossBlockSizes": all(item["ci"][0] > 0 for item in intervals),
                "intervals": intervals,
            }
        baseline_comparisons.append({"baseline": baseline_name, "metrics": metric_results})

    strongest_brier = strongest_by_metric["brier"]
    strongest_brier_comparison = next(
        item for item in baseline_comparisons if item["baseline"] == strongest_brier
    )
    sensitivity = [
        {
            "blockSizeRows": item["blockSizeRows"],
            "meanPairedBrierLift": strongest_brier_comparison["metrics"]["brier"]["meanLift"],
            "confidenceLevel": 1.0 - comparison_alpha,
            "ci": item["ci"],
            "status": item["status"],
        }
        for item in strongest_brier_comparison["metrics"]["brier"]["intervals"]
    ]

    full_loss = candidate_losses["brier"]
    ablations = []
    for group, name in zip(FEATURE_GROUPS, ablation_names, strict=True):
        delta = v2._brier_rows(labels, probabilities[name]) - full_loss
        low, high = v2._paired_block_interval(
            delta,
            block_size=config.bootstrap_block_sizes_rows[0],
            bootstrap_samples=config.bootstrap_samples,
            random_state=config.random_state,
            alpha=config.familywise_alpha / len(ablation_names),
        )
        ablations.append({
            "group": group,
            "trial": name,
            "omittedFeatures": list(FEATURE_GROUPS[group]),
            "metrics": metrics[name],
            "meanBrierDamageWhenOmitted": float(np.mean(delta)),
            "familywiseCi": [low, high],
            "interpretation": "useful" if low > 0 else ("harmful" if high < 0 else "inconclusive"),
            "comparison": "same HGB hyperparameters, folds, calibration partitions, and OOS timestamps as full model",
        })

    class_counts = metrics[FULL_MODEL]["classCounts"]
    all_comparisons = [
        metric
        for comparison in baseline_comparisons
        for metric in comparison["metrics"].values()
    ]
    checks = {
        "minimumSamples": len(labels) >= config.minimum_gate_samples,
        "minimumClassSupport": min(class_counts.values()) >= config.minimum_class_samples,
        "brierBetterThanStrongestBrierBaseline": metrics[FULL_MODEL]["brier"]
        < metrics[strongest_by_metric["brier"]]["brier"],
        "logLossBetterThanStrongestLogLossBaseline": metrics[FULL_MODEL]["logLoss"]
        < metrics[strongest_by_metric["logLoss"]]["logLoss"],
        "allBaselineMetricIntervalsPositiveAcrossBlockSizes": all(
            result["robustAcrossBlockSizes"] for result in all_comparisons
        ),
    }
    return {
        "metrics": {name: metrics[name] for name in (FULL_MODEL, *baseline_names)},
        "strongestBaseline": strongest_by_metric["brier"],
        "strongestBaselineByMetric": strongest_by_metric,
        "baselineComparisons": baseline_comparisons,
        "familywiseControl": {
            "method": "Bonferroni over every declared baseline and both proper scoring metrics",
            "familyAlpha": config.familywise_alpha,
            "comparisons": len(baseline_names) * len(candidate_losses),
            "perComparisonAlpha": comparison_alpha,
        },
        "pairedBrierSensitivity": sensitivity,
        "hgbFeatureGroupAblation": ablations,
        "gateChecks": checks,
    }


def _protocol_contract() -> dict[str, Any]:
    return {
        "outerValidation": "expanding walk-forward; non-overlapping test blocks",
        "purge": "fit and calibration rows are admitted only after their labels are available",
        "calibration": "penalized multinomial logistic recalibration over log base probabilities on the later calibration partition; if it contains one class, probabilities are left uncalibrated",
        "modelSelection": "retrospective selection-aware screen: HGB was selected in v2 using overlapping historical OOS rows; this run is not an independent confirmatory holdout",
        "ablation": "leave-one-feature-group-out HGB with identical folds/calibration/timestamps",
        "uncertainty": "paired circular block bootstrap over row-level Brier loss differences",
        "classOrder": [int(value) for value in v2.CLASSES],
    }


def _limitations_contract() -> list[str]:
    return [
        "This is one BTCUSDT 4h close-to-close target and does not establish trading profitability.",
        "Brier score combines calibration and discrimination; inspect reliability tables and log loss as well.",
        "Feature-group ablation estimates conditional contribution in this fixed HGB protocol, not causal market effects.",
        "Adaptive windows and bootstrap blocks are predeclared sensitivity checks, not post-hoc promotion choices.",
        "HGB was chosen in the earlier v2 study on overlapping OOS history, so this bundle cannot claim independent confirmation.",
        "No fees, slippage, latency, fill model, capacity, paper outcome, or live outcome is evaluated here.",
        "There are no per-fold checkpoints; an interrupted production-size run must restart.",
    ]


def _execution_plan(fold_count: int) -> dict[str, Any]:
    return {
        "folds": fold_count,
        "hgbFits": fold_count * (1 + len(FEATURE_GROUPS)),
        "checkpointResume": False,
        "note": "The current implementation writes only a complete immutable bundle; interrupted runs must restart.",
    }


def evaluate_walk_forward_v3(
    data: v2.BenchmarkData,
    config: V3Config,
    *,
    evaluation_end_ms: int | None = None,
) -> V3Result:
    """Evaluate the full HGB, same-protocol leave-one-group-out HGB, and baselines."""
    config.validate()
    _validate_data_schema(data, config.window_size)
    if evaluation_end_ms is not None:
        data = freeze_at_cutoff(
            data,
            decision_cutoff_ms=int(evaluation_end_ms),
            window_size=config.window_size,
        )

    folds = v2._folds(data, _v2_config(config))
    ablation_names = tuple(f"hgb_without_{group}" for group in FEATURE_GROUPS)
    model_names = (FULL_MODEL, *ablation_names)
    baseline_names = declared_baselines(config)
    outputs: dict[str, list[np.ndarray]] = {name: [] for name in model_names + baseline_names}
    timestamps_by_trial: dict[str, list[np.ndarray]] = {name: [] for name in outputs}
    labels_by_fold: list[np.ndarray] = []
    timestamps_by_fold: list[np.ndarray] = []
    fold_ids: list[np.ndarray] = []
    fold_contracts: list[dict[str, Any]] = []

    for fold in folds:
        fold_id = int(fold["fold"])
        fit_indices = np.asarray(fold["fit"], dtype=np.int64)
        calibration_indices = np.asarray(fold["calibration"], dtype=np.int64)
        test_indices = np.asarray(fold["test"], dtype=np.int64)
        labels_by_fold.append(data.labels[test_indices])
        timestamps_by_fold.append(data.decision_times_ms[test_indices])
        fold_ids.append(np.full(len(test_indices), fold_id, dtype=np.int32))
        fold_contracts.append(
            {key: value for key, value in fold.items() if key not in {"fit", "calibration", "test"}}
            | {
                "fitRows": int(len(fit_indices)),
                "calibrationRows": int(len(calibration_indices)),
                "testRows": int(len(test_indices)),
                "maxFitLabelAvailableMs": int(np.max(data.label_available_times_ms[fit_indices])),
                "maxCalibrationLabelAvailableMs": int(np.max(data.label_available_times_ms[calibration_indices])),
            }
        )
        for name in model_names:
            omitted = name.removeprefix("hgb_without_") if name.startswith("hgb_without_") else None
            probabilities = _fit_calibrated_hgb(
                data, fit_indices, calibration_indices, test_indices,
                _feature_columns(config.window_size, omitted),
                config.random_state + fold_id,
            )
            outputs[name].append(probabilities)
            timestamps_by_trial[name].append(data.decision_times_ms[test_indices])
        baselines = _baseline_probabilities(data, test_indices, config)
        for name in baseline_names:
            outputs[name].append(baselines[name])
            timestamps_by_trial[name].append(data.decision_times_ms[test_indices])

    labels = np.concatenate(labels_by_fold)
    timestamps = np.concatenate(timestamps_by_fold)
    row_fold_ids = np.concatenate(fold_ids)
    if len(np.unique(timestamps)) != len(timestamps):
        raise ValueError("outer test folds overlap")
    probabilities = {name: np.concatenate(parts) for name, parts in outputs.items()}
    expected_names = model_names + baseline_names
    timestamp_hashes: dict[str, str] = {}
    for name in expected_names:
        covered = np.concatenate(timestamps_by_trial[name])
        if not np.array_equal(covered, timestamps):
            raise ValueError(f"timestamp coverage mismatch for {name}")
        values = probabilities[name]
        if values.shape != (len(timestamps), 3) or not np.isfinite(values).all():
            raise ValueError(f"invalid probability coverage for {name}")
        if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-9):
            raise ValueError(f"probabilities do not sum to one for {name}")
        timestamp_hashes[name] = _sha256(covered.astype("<i8").tobytes())

    statistics = _statistical_evidence(labels, probabilities, config)
    statistics["gateChecks"] = {
        "identicalTimestampCoverage": len(set(timestamp_hashes.values())) == 1,
        **statistics["gateChecks"],
    }
    prediction_rows = tuple(
        {
            "foldId": int(row_fold_ids[index]),
            "decisionTimeMs": int(timestamps[index]),
            "label": int(labels[index]),
            "probabilities": {
                name: [float(value) for value in probabilities[name][index]]
                for name in expected_names
            },
        }
        for index in range(len(timestamps))
    )
    evaluation_data = _select_rows_by_timestamps(data, timestamps)
    evaluation = {
        "schemaVersion": SCHEMA_VERSION,
        "evaluatorVersion": EVALUATOR_VERSION,
        "scope": "BTCUSDT 4h next-bar close-to-close direction benchmark",
        "hypothesis": "A causally computed HGB feature model improves proper probabilistic scores over predeclared historical, adaptive, momentum, and reversion baselines on future walk-forward blocks.",
        "protocol": _protocol_contract(),
        "declaredBaselines": list(baseline_names),
        "baselineDefinitions": baseline_definitions(config),
        "folds": fold_contracts,
        "evaluationRows": int(len(timestamps)),
        "evaluationTimestampSha256": next(iter(timestamp_hashes.values())),
        "evaluationDatasetSha256": v2.dataset_sha256(evaluation_data),
        "evidenceTier": "retrospective_selection_aware",
        "executionPlan": _execution_plan(len(folds)),
        **statistics,
        "promotionGate": {
            "passed": all(statistics["gateChecks"].values()),
            "status": "retrospective_screen_passed" if all(statistics["gateChecks"].values()) else "retrospective_screen_not_passed",
            "confirmatory": False,
            "promotionAllowed": False,
            "checks": statistics["gateChecks"],
            "policy": "retrospective research screen only; it is not confirmatory and never mutates registry, paper, or live trading",
        },
        "limitations": _limitations_contract(),
        "negativeResultPolicy": "Negative and inconclusive results are retained and are valid outcomes.",
    }
    return V3Result(evaluation=evaluation, prediction_rows=prediction_rows)


def build_v3_manifest(
    data: v2.BenchmarkData,
    config: V3Config,
    *,
    decision_cutoff_ms: int,
    source: str,
) -> dict[str, Any]:
    config.validate()
    _require_frozen_at_cutoff(
        data,
        decision_cutoff_ms=decision_cutoff_ms,
        window_size=config.window_size,
    )
    base = v2.build_manifest(
        data,
        _v2_config(config),
        decision_cutoff_ms=decision_cutoff_ms,
        source=source,
    )
    core = {
        "schemaVersion": SCHEMA_VERSION,
        "experiment": "btc-4h-next-bar-ml-evidence-v3",
        "symbol": "BTCUSDT",
        "timeframe": "4h",
        "decisionCutoffMs": int(decision_cutoff_ms),
        "parameters": asdict(config),
        "featureGroups": {name: list(features) for name, features in FEATURE_GROUPS.items()},
        "dataProvenance": base["dataProvenance"],
        "codeProvenance": base["codeProvenance"] | {
            "v3EvaluatorSha256": v2._file_hash(Path(__file__).resolve())
        },
        "runtimeDependencies": base["runtimeDependencies"],
        "sourceDisclosure": "source kind only; secrets and database credentials excluded",
    }
    core["manifestSha256"] = _sha256(_canonical_json(core).encode("utf-8"))
    _validate_research_manifest(core)
    _validate_manifest_data(core, data)
    return core


def _snapshot_bytes(data: v2.BenchmarkData) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        X=data.features.astype("<f8"),
        y=data.labels.astype("i1"),
        decision_ms=data.decision_times_ms.astype("<i8"),
        label_available_ms=data.label_available_times_ms.astype("<i8"),
        last_return=data.last_returns.astype("<f8"),
        causal_feature_names=np.asarray(v2.CAUSAL_FEATURE_NAMES, dtype="U32"),
        schema_version=np.asarray(SNAPSHOT_SCHEMA_VERSION),
        window_size=np.asarray(data.features.shape[1] // len(v2.CAUSAL_FEATURE_NAMES), dtype="<i8"),
    )
    return buffer.getvalue()


def _load_snapshot(path: Path, *, expected_window_size: int) -> v2.BenchmarkData:
    with np.load(path, allow_pickle=False) as snapshot:
        required = {
            "X", "y", "decision_ms", "label_available_ms", "last_return",
            "causal_feature_names", "schema_version", "window_size",
        }
        if set(snapshot.files) != required:
            raise ValueError(
                f"v3 snapshot arrays must be exactly {sorted(required)}; got {sorted(snapshot.files)}"
            )
        expected_dtypes = {
            "X": np.dtype("<f8"),
            "y": np.dtype("i1"),
            "decision_ms": np.dtype("<i8"),
            "label_available_ms": np.dtype("<i8"),
            "last_return": np.dtype("<f8"),
            "causal_feature_names": np.dtype("<U32"),
            "schema_version": np.dtype(f"<U{len(SNAPSHOT_SCHEMA_VERSION)}"),
            "window_size": np.dtype("<i8"),
        }
        for name, expected_dtype in expected_dtypes.items():
            if snapshot[name].dtype != expected_dtype:
                raise ValueError(
                    f"snapshot dtype mismatch for {name}: expected {expected_dtype}, "
                    f"got {snapshot[name].dtype}"
                )
        if snapshot["schema_version"].shape != () or snapshot["window_size"].shape != ():
            raise ValueError("snapshot schema_version and window_size must be scalar arrays")
        if snapshot["causal_feature_names"].shape != (len(v2.CAUSAL_FEATURE_NAMES),):
            raise ValueError("snapshot causal_feature_names shape does not match evaluator schema")
        schema_version = str(np.asarray(snapshot["schema_version"]).item())
        if schema_version != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(f"unsupported snapshot schema: {schema_version}")
        window_size = int(np.asarray(snapshot["window_size"]).item())
        if window_size != expected_window_size:
            raise ValueError(
                f"snapshot window_size mismatch: expected {expected_window_size}, got {window_size}"
            )
        names = tuple(str(value) for value in np.asarray(snapshot["causal_feature_names"]).tolist())
        if names != v2.CAUSAL_FEATURE_NAMES:
            raise ValueError("snapshot causal feature names or ordering do not match evaluator schema")
        data = v2.BenchmarkData(
            np.asarray(snapshot["X"], dtype=np.float64),
            np.asarray(snapshot["y"], dtype=np.int8),
            np.asarray(snapshot["decision_ms"], dtype=np.int64),
            np.asarray(snapshot["label_available_ms"], dtype=np.int64),
            np.asarray(snapshot["last_return"], dtype=np.float64),
        )
    _validate_data_schema(data, expected_window_size)
    return data


def load_v3_npz(
    path: Path,
    *,
    decision_cutoff_ms: int,
    window_size: int,
) -> v2.BenchmarkData:
    data = _load_snapshot(path, expected_window_size=window_size)
    _require_frozen_at_cutoff(
        data,
        decision_cutoff_ms=decision_cutoff_ms,
        window_size=window_size,
    )
    return data


def _atomic_immutable_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise FileExistsError(f"refusing to overwrite immutable evidence artifact: {path}")
    finally:
        temporary_path.unlink(missing_ok=True)


def _config_from_manifest(manifest: dict[str, Any]) -> V3Config:
    parameters = dict(manifest.get("parameters", {}))
    expected = set(V3Config.__dataclass_fields__)
    if set(parameters) != expected:
        raise ValueError("research manifest parameters do not match V3Config schema")
    parameters["bootstrap_block_sizes_rows"] = tuple(parameters["bootstrap_block_sizes_rows"])
    parameters["adaptive_prior_windows_rows"] = tuple(parameters["adaptive_prior_windows_rows"])
    config = V3Config(**parameters)
    config.validate()
    return config


def _validate_research_manifest(manifest: dict[str, Any]) -> V3Config:
    if set(manifest) != RESEARCH_MANIFEST_KEYS:
        raise ValueError("research manifest schema mismatch")
    if manifest.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("unsupported research manifest schema")
    if (
        manifest.get("experiment") != "btc-4h-next-bar-ml-evidence-v3"
        or manifest.get("symbol") != "BTCUSDT"
        or manifest.get("timeframe") != "4h"
    ):
        raise ValueError("research manifest scope mismatch")
    cutoff = manifest.get("decisionCutoffMs")
    if isinstance(cutoff, bool) or not isinstance(cutoff, int):
        raise ValueError("research manifest decision cutoff must be an integer")
    if manifest.get("sourceDisclosure") != (
        "source kind only; secrets and database credentials excluded"
    ):
        raise ValueError("research manifest source disclosure mismatch")
    expected_groups = {name: list(features) for name, features in FEATURE_GROUPS.items()}
    if manifest.get("featureGroups") != expected_groups:
        raise ValueError("research manifest feature groups mismatch")
    data_provenance = manifest.get("dataProvenance")
    required_data_keys = {
        "source", "datasetSha256", "rowCount", "featureDim", "firstDecisionTimeMs",
        "lastDecisionTimeMs", "decisionCutoffMs", "featureSource", "labelSource",
        "rowJoin", "closedBarPredicate", "realizedLabelPredicate",
    }
    if not isinstance(data_provenance, dict) or frozenset(data_provenance) not in {
        frozenset(required_data_keys),
        frozenset(required_data_keys | {"databaseIdentity"}),
    }:
        raise ValueError("research manifest data provenance schema mismatch")
    if "databaseIdentity" in data_provenance:
        database_identity = data_provenance["databaseIdentity"]
        if (
            not isinstance(database_identity, dict)
            or set(database_identity) != {"host", "port", "database", "passwordExcluded"}
            or database_identity.get("passwordExcluded") is not True
        ):
            raise ValueError("research manifest database identity schema mismatch")
    if data_provenance.get("decisionCutoffMs") != cutoff:
        raise ValueError("research manifest cutoff references disagree")
    code_provenance = manifest.get("codeProvenance")
    if not isinstance(code_provenance, dict) or set(code_provenance) != {
        "evaluatorSha256", "researchContractSha256", "git", "v3EvaluatorSha256",
    }:
        raise ValueError("research manifest code provenance schema mismatch")
    if not isinstance(code_provenance.get("git"), dict) or set(code_provenance["git"]) != {
        "commit", "dirty",
    }:
        raise ValueError("research manifest git provenance schema mismatch")
    runtime = manifest.get("runtimeDependencies")
    if not isinstance(runtime, dict) or set(runtime) != {
        "python", "numpy", "scikitLearn", "psycopg2",
    }:
        raise ValueError("research manifest runtime dependency schema mismatch")
    return _config_from_manifest(manifest)


def _validate_manifest_data(manifest: dict[str, Any], data: v2.BenchmarkData) -> None:
    provenance = manifest["dataProvenance"]
    expected = {
        "datasetSha256": v2.dataset_sha256(data),
        "rowCount": len(data.labels),
        "featureDim": data.features.shape[1],
        "firstDecisionTimeMs": int(data.decision_times_ms[0]),
        "lastDecisionTimeMs": int(data.decision_times_ms[-1]),
        "decisionCutoffMs": int(manifest["decisionCutoffMs"]),
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"research manifest data provenance mismatch: {key}")


def _verify_self_hash(payload: dict[str, Any], field: str, *, label: str) -> str:
    core = dict(payload)
    supplied = core.pop(field, None)
    actual = _sha256(_canonical_json(core).encode("utf-8"))
    if supplied != actual:
        raise ValueError(f"{label} self-hash mismatch")
    return actual


def _assert_json_equivalent(actual: Any, expected: Any, path: str = "root") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"semantic report mismatch at {path}")
        for key in expected:
            _assert_json_equivalent(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"semantic report mismatch at {path}")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            _assert_json_equivalent(actual_item, expected_item, f"{path}[{index}]")
        return
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not np.isclose(
            float(actual), expected, rtol=1e-12, atol=1e-12
        ):
            raise ValueError(f"semantic report mismatch at {path}")
        return
    if isinstance(expected, bool):
        if not isinstance(actual, bool) or actual is not expected:
            raise ValueError(f"semantic report mismatch at {path}")
        return
    if isinstance(expected, int):
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
            raise ValueError(f"semantic report mismatch at {path}")
        return
    if isinstance(expected, str):
        if not isinstance(actual, str) or actual != expected:
            raise ValueError(f"semantic report mismatch at {path}")
        return
    if actual != expected:
        raise ValueError(f"semantic report mismatch at {path}")


def _expected_fold_contracts(
    data: v2.BenchmarkData,
    config: V3Config,
) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for fold in v2._folds(data, _v2_config(config)):
        fit_indices = np.asarray(fold["fit"], dtype=np.int64)
        calibration_indices = np.asarray(fold["calibration"], dtype=np.int64)
        test_indices = np.asarray(fold["test"], dtype=np.int64)
        contracts.append(
            {key: value for key, value in fold.items() if key not in {"fit", "calibration", "test"}}
            | {
                "fitRows": int(len(fit_indices)),
                "calibrationRows": int(len(calibration_indices)),
                "testRows": int(len(test_indices)),
                "maxFitLabelAvailableMs": int(np.max(data.label_available_times_ms[fit_indices])),
                "maxCalibrationLabelAvailableMs": int(
                    np.max(data.label_available_times_ms[calibration_indices])
                ),
            }
        )
    return contracts


def _validate_evaluation_contract(
    data: v2.BenchmarkData,
    manifest: dict[str, Any],
    evaluation: dict[str, Any],
    expected_statistics: dict[str, Any],
) -> None:
    expected_keys = {
        "schemaVersion", "evaluatorVersion", "scope", "hypothesis", "protocol",
        "declaredBaselines", "baselineDefinitions", "folds", "evaluationRows",
        "evaluationTimestampSha256", "evaluationDatasetSha256", "evidenceTier",
        "executionPlan", "metrics", "strongestBaseline", "strongestBaselineByMetric",
        "baselineComparisons", "familywiseControl", "pairedBrierSensitivity",
        "hgbFeatureGroupAblation", "gateChecks", "promotionGate", "limitations",
        "negativeResultPolicy",
    }
    if not isinstance(evaluation, dict) or set(evaluation) != expected_keys:
        raise ValueError("evaluation report schema mismatch")
    config = _validate_research_manifest(manifest)
    folds = _expected_fold_contracts(data, config)
    expected_passed = all(expected_statistics["gateChecks"].values())
    expected_static = {
        "schemaVersion": SCHEMA_VERSION,
        "evaluatorVersion": EVALUATOR_VERSION,
        "scope": "BTCUSDT 4h next-bar close-to-close direction benchmark",
        "hypothesis": "A causally computed HGB feature model improves proper probabilistic scores over predeclared historical, adaptive, momentum, and reversion baselines on future walk-forward blocks.",
        "protocol": _protocol_contract(),
        "declaredBaselines": list(declared_baselines(config)),
        "baselineDefinitions": baseline_definitions(config),
        "folds": folds,
        "evidenceTier": "retrospective_selection_aware",
        "executionPlan": _execution_plan(len(folds)),
        "promotionGate": {
            "passed": expected_passed,
            "status": "retrospective_screen_passed" if expected_passed else "retrospective_screen_not_passed",
            "confirmatory": False,
            "promotionAllowed": False,
            "checks": expected_statistics["gateChecks"],
            "policy": "retrospective research screen only; it is not confirmatory and never mutates registry, paper, or live trading",
        },
        "limitations": _limitations_contract(),
        "negativeResultPolicy": "Negative and inconclusive results are retained and are valid outcomes.",
    }
    for key, expected in expected_static.items():
        _assert_json_equivalent(evaluation.get(key), expected, f"evaluation.{key}")


def _validate_prediction_semantics(
    data: v2.BenchmarkData,
    manifest: dict[str, Any],
    evaluation: dict[str, Any],
    prediction_rows: Iterable[dict[str, Any]],
    *,
    recompute_statistics: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    config = _validate_research_manifest(manifest)
    cutoff = int(manifest["decisionCutoffMs"])
    _require_frozen_at_cutoff(
        data,
        decision_cutoff_ms=cutoff,
        window_size=config.window_size,
    )
    rows = tuple(prediction_rows)
    if not rows or len(rows) != evaluation.get("evaluationRows"):
        raise ValueError("prediction row coverage mismatch")
    expected_trials = {
        FULL_MODEL,
        *(f"hgb_without_{group}" for group in FEATURE_GROUPS),
        *declared_baselines(config),
    }
    row_keys = {"foldId", "decisionTimeMs", "label", "probabilities"}
    for row in rows:
        if not isinstance(row, dict) or set(row) != row_keys:
            raise ValueError("prediction row schema mismatch")
        for key in ("foldId", "decisionTimeMs", "label"):
            if isinstance(row[key], bool) or not isinstance(row[key], int):
                raise ValueError(f"prediction row {key} must be an integer")
        if not isinstance(row["probabilities"], dict):
            raise ValueError("prediction probabilities must be an object")
    timestamps = np.asarray([row["decisionTimeMs"] for row in rows], dtype=np.int64)
    if np.any(timestamps[1:] <= timestamps[:-1]) or int(timestamps[-1]) > cutoff:
        raise ValueError("prediction timestamps must be unique, increasing, and within cutoff")
    selected = _select_rows_by_timestamps(data, timestamps)
    labels = np.asarray([row.get("label") for row in rows], dtype=np.int8)
    if not np.array_equal(labels, selected.labels):
        raise ValueError("prediction labels do not match frozen dataset")

    folds = v2._folds(data, _v2_config(config))
    expected_timestamps = np.concatenate(
        [data.decision_times_ms[np.asarray(fold["test"], dtype=np.int64)] for fold in folds]
    )
    expected_fold_ids = np.concatenate(
        [np.full(len(fold["test"]), int(fold["fold"]), dtype=np.int32) for fold in folds]
    )
    actual_fold_ids = np.asarray([row.get("foldId") for row in rows], dtype=np.int32)
    if not np.array_equal(timestamps, expected_timestamps) or not np.array_equal(
        actual_fold_ids, expected_fold_ids
    ):
        raise ValueError("prediction rows do not match deterministic fold coverage")

    probabilities: dict[str, np.ndarray] = {}
    for name in sorted(expected_trials):
        if any(set(row.get("probabilities", {})) != expected_trials for row in rows):
            raise ValueError("prediction trials do not match declared model and baseline family")
        for row in rows:
            raw = row["probabilities"][name]
            if (
                not isinstance(raw, list)
                or len(raw) != len(v2.CLASSES)
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw)
            ):
                raise ValueError(f"invalid probability schema for {name}")
        values = np.asarray([row["probabilities"][name] for row in rows], dtype=np.float64)
        if (
            values.shape != (len(rows), len(v2.CLASSES))
            or not np.isfinite(values).all()
            or np.any(values < 0.0)
            or np.any(values > 1.0)
            or not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-9)
        ):
            raise ValueError(f"invalid probabilities for {name}")
        probabilities[name] = values

    timestamp_hash = _sha256(timestamps.astype("<i8").tobytes())
    if timestamp_hash != evaluation.get("evaluationTimestampSha256"):
        raise ValueError("prediction timestamp coverage mismatch")
    if v2.dataset_sha256(selected) != evaluation.get("evaluationDatasetSha256"):
        raise ValueError("evaluated dataset semantic hash mismatch")
    if evaluation.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("unsupported evaluation schema")
    if evaluation.get("evaluatorVersion") != EVALUATOR_VERSION:
        raise ValueError("unsupported evaluator version")

    if recompute_statistics:
        expected_statistics = _statistical_evidence(labels, probabilities, config)
        expected_statistics["gateChecks"] = {
            "identicalTimestampCoverage": True,
            **expected_statistics["gateChecks"],
        }
        for key, expected in expected_statistics.items():
            _assert_json_equivalent(evaluation.get(key), expected, f"evaluation.{key}")
        _validate_evaluation_contract(data, manifest, evaluation, expected_statistics)
    return labels, probabilities


def write_evidence_bundle(
    output_dir: Path,
    data: v2.BenchmarkData,
    manifest: dict[str, Any],
    result: V3Result,
) -> dict[str, Path]:
    """Write and cross-hash a complete immutable bundle with no circular hashes."""
    _verify_self_hash(manifest, "manifestSha256", label="research manifest")
    config = _validate_research_manifest(manifest)
    _require_frozen_at_cutoff(
        data,
        decision_cutoff_ms=int(manifest["decisionCutoffMs"]),
        window_size=config.window_size,
    )
    _validate_manifest_data(manifest, data)
    if len(result.prediction_rows) != result.evaluation["evaluationRows"]:
        raise ValueError("row predictions do not match reported evaluation coverage")
    _validate_prediction_semantics(
        data,
        manifest,
        result.evaluation,
        result.prediction_rows,
        recompute_statistics=True,
    )
    snapshot = _snapshot_bytes(data)
    predictions = (
        "".join(_canonical_json(row) + "\n" for row in result.prediction_rows)
    ).encode("utf-8")
    report_core = {"schemaVersion": SCHEMA_VERSION, "researchManifest": manifest, "evaluation": result.evaluation}
    report_core["reportPayloadSha256"] = _sha256(_canonical_json(report_core).encode("utf-8"))
    report = (json.dumps(report_core, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")

    artifact_bytes = {"datasetSnapshot": snapshot, "rowPredictions": predictions, "report": report}
    suffixes = {"datasetSnapshot": "dataset.npz", "rowPredictions": "predictions.jsonl", "report": "report.json"}
    artifacts: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for key, content in artifact_bytes.items():
        digest = _sha256(content)
        filename = f"{digest}.{suffixes[key]}"
        path = output_dir / filename
        _atomic_immutable_write(path, content)
        artifacts[key] = {"file": filename, "sha256": digest, "bytes": len(content)}
        paths[key] = path

    bundle_core = {
        "schemaVersion": SCHEMA_VERSION,
        "researchManifestSha256": manifest["manifestSha256"],
        "artifacts": artifacts,
        "integrityPolicy": "verify every artifact byte hash before use; fail closed on missing or mismatched files",
    }
    bundle_hash = _sha256(_canonical_json(bundle_core).encode("utf-8"))
    bundle = bundle_core | {"bundleManifestSha256": bundle_hash}
    bundle_content = (json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    bundle_path = output_dir / f"{bundle_hash}.manifest.json"
    _atomic_immutable_write(bundle_path, bundle_content)
    paths["bundleManifest"] = bundle_path
    return paths


def verify_evidence_bundle(bundle_path: Path) -> dict[str, Any]:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if set(bundle) != {
        "schemaVersion", "researchManifestSha256", "artifacts",
        "integrityPolicy", "bundleManifestSha256",
    } or bundle.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("unsupported bundle manifest schema")
    actual_hash = _verify_self_hash(bundle, "bundleManifestSha256", label="bundle manifest")
    if bundle_path.name != f"{actual_hash}.manifest.json":
        raise ValueError("bundle manifest hash mismatch")
    expected_suffixes = {
        "datasetSnapshot": "dataset.npz",
        "rowPredictions": "predictions.jsonl",
        "report": "report.json",
    }
    if set(bundle["artifacts"]) != set(expected_suffixes):
        raise ValueError("bundle artifact keys do not match v3 schema")
    research_hash = bundle.get("researchManifestSha256")
    if (
        not isinstance(research_hash, str)
        or len(research_hash) != 64
        or any(character not in "0123456789abcdef" for character in research_hash)
    ):
        raise ValueError("bundle research manifest hash is invalid")
    for key, artifact in bundle["artifacts"].items():
        if set(artifact) != {"file", "sha256", "bytes"}:
            raise ValueError(f"artifact metadata schema mismatch: {key}")
        if (
            not isinstance(artifact["file"], str)
            or not isinstance(artifact["sha256"], str)
            or len(artifact["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in artifact["sha256"])
            or isinstance(artifact["bytes"], bool)
            or not isinstance(artifact["bytes"], int)
            or artifact["bytes"] < 0
        ):
            raise ValueError(f"artifact metadata value mismatch: {key}")
        name = artifact["file"]
        if Path(name).name != name:
            raise ValueError("artifact path must be a basename")
        expected_name = f"{artifact['sha256']}.{expected_suffixes[key]}"
        if name != expected_name:
            raise ValueError(f"artifact filename is not content-addressed: {name}")
        path = bundle_path.parent / name
        if not path.is_file():
            raise ValueError(f"artifact missing: {name}")
        content = path.read_bytes()
        if len(content) != artifact["bytes"] or _sha256(content) != artifact["sha256"]:
            raise ValueError(f"artifact hash mismatch: {name}")

    report_path = bundle_path.parent / bundle["artifacts"]["report"]["file"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if set(report) != {"schemaVersion", "researchManifest", "evaluation", "reportPayloadSha256"}:
        raise ValueError("report schema mismatch")
    report_without_hash = dict(report)
    supplied_report_hash = report_without_hash.pop("reportPayloadSha256", None)
    if supplied_report_hash != _sha256(_canonical_json(report_without_hash).encode("utf-8")):
        raise ValueError("report payload hash mismatch")
    if report.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("unsupported report schema")
    research_manifest = report.get("researchManifest", {})
    manifest_hash = _verify_self_hash(
        research_manifest, "manifestSha256", label="research manifest"
    )
    if manifest_hash != bundle["researchManifestSha256"]:
        raise ValueError("research manifest reference mismatch")
    config = _validate_research_manifest(research_manifest)

    snapshot_path = bundle_path.parent / bundle["artifacts"]["datasetSnapshot"]["file"]
    snapshot_data = _load_snapshot(snapshot_path, expected_window_size=config.window_size)
    _require_frozen_at_cutoff(
        snapshot_data,
        decision_cutoff_ms=int(research_manifest["decisionCutoffMs"]),
        window_size=config.window_size,
    )
    _validate_manifest_data(research_manifest, snapshot_data)

    predictions_path = bundle_path.parent / bundle["artifacts"]["rowPredictions"]["file"]
    prediction_rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    evaluation = report.get("evaluation", {})
    _validate_prediction_semantics(
        snapshot_data,
        research_manifest,
        evaluation,
        prediction_rows,
        recompute_statistics=True,
    )
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="BTCUSDT 4h reproducible ML evidence bundle v3")
    parser.add_argument("--decision-cutoff-ms", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-npz", help="optional frozen replay dataset; otherwise read PostgreSQL")
    parser.add_argument("--min-fit-rows", type=int, default=500)
    parser.add_argument("--calibration-rows", type=int, default=120)
    parser.add_argument("--test-rows", type=int, default=120)
    parser.add_argument("--step-rows", type=int, default=120)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    config = V3Config(
        min_fit_rows=args.min_fit_rows,
        calibration_rows=args.calibration_rows,
        test_rows=args.test_rows,
        step_rows=args.step_rows,
        bootstrap_samples=args.bootstrap_samples,
    )
    if args.input_npz:
        data = load_v3_npz(
            Path(args.input_npz),
            decision_cutoff_ms=args.decision_cutoff_ms,
            window_size=config.window_size,
        )
        source = "immutable-npz-replay"
    else:
        data = v2.load_btc_4h_benchmark(
            decision_cutoff_ms=args.decision_cutoff_ms,
            window_size=config.window_size,
        )
        data = freeze_at_cutoff(
            data,
            decision_cutoff_ms=args.decision_cutoff_ms,
            window_size=config.window_size,
        )
        source = "postgresql-readonly"
    result = evaluate_walk_forward_v3(data, config)
    manifest = build_v3_manifest(
        data,
        config,
        decision_cutoff_ms=args.decision_cutoff_ms,
        source=source,
    )
    paths = write_evidence_bundle(Path(args.output_dir), data, manifest, result)
    verify_evidence_bundle(paths["bundleManifest"])
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
