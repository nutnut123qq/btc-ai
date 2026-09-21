#!/usr/bin/env python3
"""Common leakage-safe evidence evaluator for the BTCUSDT 4h next-bar benchmark.

The command is read-only with respect to PostgreSQL. It evaluates a fixed, declared
family (scaled logistic regression and histogram gradient boosting) with expanding
outer walk-forward folds. Every transform, model and probability calibrator is fit
only on data available before its evaluation interval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import psycopg2
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from db_config import get_db_connection, get_db_params
from research_contract import ResearchManifest


CLASSES = np.array([-1, 0, 1], dtype=np.int8)
INTERVAL_MS = 4 * 60 * 60 * 1000
FEATURES_PER_BAR = 35
ACTIVE_RULE_COUNT_OFFSET = 29
EVALUATOR_VERSION = "btc-4h-next-bar-ml-evidence-v1"
DECLARED_MODEL_FAMILY = ("logistic_scaled", "hist_gradient_boosting")
DECLARED_BASELINES = ("rolling_class_probs", "rolling_majority", "momentum", "reversion")


@dataclass(frozen=True)
class EvaluatorConfig:
    window_size: int = 5
    min_fit_rows: int = 500
    calibration_rows: int = 120
    test_rows: int = 120
    step_rows: int = 120
    label_horizon_bars: int = 1
    laplace_alpha: float = 1.0
    rule_confidence: float = 0.80
    reliability_bins: int = 10
    block_size_rows: int = 12
    bootstrap_samples: int = 1000
    familywise_alpha: float = 0.05
    random_state: int = 42
    minimum_gate_samples: int = 240
    minimum_class_samples: int = 20

    def validate(self) -> None:
        for name in (
            "window_size",
            "min_fit_rows",
            "calibration_rows",
            "test_rows",
            "step_rows",
            "label_horizon_bars",
            "reliability_bins",
            "block_size_rows",
            "bootstrap_samples",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 1 / 3 < self.rule_confidence < 1:
            raise ValueError("rule_confidence must be between 1/3 and 1")
        if not 0 < self.familywise_alpha < 1:
            raise ValueError("familywise_alpha must be in (0, 1)")


@dataclass(frozen=True)
class BenchmarkData:
    features: np.ndarray
    labels: np.ndarray
    decision_times_ms: np.ndarray
    label_available_times_ms: np.ndarray
    last_returns: np.ndarray

    def validate(self) -> None:
        n = len(self.features)
        if self.features.ndim != 2 or n == 0:
            raise ValueError("features must be a non-empty 2D matrix")
        for name in ("labels", "decision_times_ms", "label_available_times_ms", "last_returns"):
            value = getattr(self, name)
            if value.ndim != 1 or len(value) != n:
                raise ValueError(f"{name} must align with features")
        if not np.isfinite(self.features).all() or not np.isfinite(self.last_returns).all():
            raise ValueError("features and last_returns must be finite")
        if not np.isin(self.labels, CLASSES).all():
            raise ValueError("labels must be -1, 0, or 1")
        if np.any(self.decision_times_ms[1:] <= self.decision_times_ms[:-1]):
            raise ValueError("decision timestamps must be strictly increasing")
        if np.any(self.label_available_times_ms < self.decision_times_ms):
            raise ValueError("label availability cannot precede the decision")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(parts: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def dataset_sha256(data: BenchmarkData) -> str:
    parts: list[bytes] = []
    for array in (
        data.features.astype("<f8"),
        data.labels.astype("i1"),
        data.decision_times_ms.astype("<i8"),
        data.label_available_times_ms.astype("<i8"),
        data.last_returns.astype("<f8"),
    ):
        parts.extend((str(array.shape).encode("ascii"), array.tobytes(order="C")))
    return _sha256_bytes(parts)


def drop_noncausal_stored_features(features: np.ndarray, window_size: int) -> np.ndarray:
    expected = window_size * FEATURES_PER_BAR
    if features.ndim != 2 or features.shape[1] != expected:
        raise ValueError(f"stored feature matrix must have {expected} columns")
    excluded_columns = [
        bar_index * FEATURES_PER_BAR + ACTIVE_RULE_COUNT_OFFSET
        for bar_index in range(window_size)
    ]
    return np.delete(features, excluded_columns, axis=1)


def load_btc_4h_benchmark(*, decision_cutoff_ms: int, window_size: int = 5) -> BenchmarkData:
    """Read the fixed benchmark from PostgreSQL without modifying any table."""
    feature_dim = window_size * FEATURES_PER_BAR
    query = """
        SELECT w."FeatureVector", p."TargetDirection4h", w."WindowEndMs", w."FeatureDim"
        FROM "WindowClassificationDatasets" AS w
        INNER JOIN "PriceTargets" AS p
          ON p."Symbol" = w."Symbol"
         AND p."Timeframe" = w."Timeframe"
         AND p."OpenTimeMs" = w."WindowEndMs"
        WHERE w."Symbol" = 'BTCUSDT'
          AND w."Timeframe" = '4h'
          AND w."WindowSize" = %s
          AND w."Horizon" = '4h'
          AND w."FeatureDim" = %s
          AND p."TargetDirection4h" IS NOT NULL
          AND w."WindowEndMs" + %s <= %s
        ORDER BY w."WindowEndMs"
    """
    conn = get_db_connection()
    try:
        conn.set_session(readonly=True, autocommit=False)
        with conn.cursor() as cursor:
            cursor.execute(query, (window_size, feature_dim, INTERVAL_MS, decision_cutoff_ms))
            rows = cursor.fetchall()
        conn.rollback()
    finally:
        conn.close()
    if not rows:
        raise ValueError("BTCUSDT 4h benchmark query returned no rows")

    features = np.asarray([row[0] for row in rows], dtype=np.float64)
    # ActiveRuleCount is a mutable *current* database setting that the backend
    # historically copied onto every row. It has no point-in-time rule-version
    # lineage, so it is inadmissible in a causal benchmark. Keep the stored
    # vectors intact and remove this column for every bar at evaluation time.
    features = drop_noncausal_stored_features(features, window_size)
    labels = np.asarray([row[1] for row in rows], dtype=np.int8)
    # WindowEndMs is the open time of the signal bar in the current backend schema.
    signal_open = np.asarray([row[2] for row in rows], dtype=np.int64)
    decision_times = signal_open + INTERVAL_MS
    label_available = decision_times + INTERVAL_MS
    final_mask = label_available <= int(decision_cutoff_ms)
    features = features[final_mask]
    labels = labels[final_mask]
    decision_times = decision_times[final_mask]
    label_available = label_available[final_mask]
    # Feature index 1 is ClosePctChange1; the final bar starts at this offset.
    last_returns = features[:, (window_size - 1) * FEATURES_PER_BAR + 1]
    data = BenchmarkData(features, labels, decision_times, label_available, last_returns)
    data.validate()
    return data


def _aligned_probabilities(estimator: Any, features: np.ndarray) -> np.ndarray:
    raw = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    result = np.zeros((len(features), len(CLASSES)), dtype=np.float64)
    for source_index, class_value in enumerate(estimator.classes_):
        target = int(np.flatnonzero(CLASSES == class_value)[0])
        result[:, target] = raw[:, source_index]
    result = np.clip(result, 1e-12, 1.0)
    return result / result.sum(axis=1, keepdims=True)


class PriorOnlyCalibrator:
    """Multinomial sigmoid calibration fit on a later, prior-only partition."""

    def __init__(self, random_state: int):
        self.random_state = random_state
        self.model: LogisticRegression | None = None

    def fit(self, base_probabilities: np.ndarray, labels: np.ndarray) -> "PriorOnlyCalibrator":
        if len(np.unique(labels)) < 2:
            self.model = None
            return self
        logits = np.log(np.clip(base_probabilities, 1e-12, 1.0))
        self.model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=self.random_state,
        ).fit(logits, labels)
        return self

    def transform(self, base_probabilities: np.ndarray) -> np.ndarray:
        if self.model is None:
            return base_probabilities
        return _aligned_probabilities(self.model, np.log(np.clip(base_probabilities, 1e-12, 1.0)))


def _build_estimator(name: str, random_state: int) -> Any:
    if name == "logistic_scaled":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=100,
            max_depth=3,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=random_state,
        )
    raise ValueError(f"undeclared model: {name}")


def _soft_rule_probability(direction: int, confidence: float) -> np.ndarray:
    residual = (1.0 - confidence) / 2.0
    probability = np.full(3, residual, dtype=np.float64)
    probability[int(np.flatnonzero(CLASSES == direction)[0])] = confidence
    return probability


def _baseline_probabilities(
    data: BenchmarkData,
    test_indices: np.ndarray,
    config: EvaluatorConfig,
) -> dict[str, np.ndarray]:
    outputs = {name: np.empty((len(test_indices), 3), dtype=np.float64) for name in DECLARED_BASELINES}
    for output_index, row_index in enumerate(test_indices):
        decision_time = data.decision_times_ms[row_index]
        known = (data.label_available_times_ms <= decision_time) & (
            np.arange(len(data.labels)) < row_index
        )
        counts = np.asarray([np.sum(data.labels[known] == value) for value in CLASSES], dtype=np.float64)
        rolling = (counts + config.laplace_alpha) / (
            counts.sum() + config.laplace_alpha * len(CLASSES)
        )
        outputs["rolling_class_probs"][output_index] = rolling
        majority = int(CLASSES[np.argmax(rolling)])
        outputs["rolling_majority"][output_index] = _soft_rule_probability(
            majority, config.rule_confidence
        )
        last_return = data.last_returns[row_index]
        momentum = 1 if last_return > 0 else (-1 if last_return < 0 else 0)
        reversion = -momentum
        outputs["momentum"][output_index] = _soft_rule_probability(
            momentum, config.rule_confidence
        )
        outputs["reversion"][output_index] = _soft_rule_probability(
            reversion, config.rule_confidence
        )
    return outputs


def _brier_rows(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    one_hot = np.zeros_like(probabilities)
    for class_index, class_value in enumerate(CLASSES):
        one_hot[:, class_index] = labels == class_value
    return np.sum((probabilities - one_hot) ** 2, axis=1)


def _reliability(labels: np.ndarray, probabilities: np.ndarray, bins: int) -> dict[str, list[dict[str, Any]]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result: dict[str, list[dict[str, Any]]] = {}
    for class_index, class_value in enumerate(CLASSES):
        rows: list[dict[str, Any]] = []
        for bin_index in range(bins):
            lower, upper = edges[bin_index], edges[bin_index + 1]
            mask = (probabilities[:, class_index] >= lower) & (
                probabilities[:, class_index] < upper
                if bin_index < bins - 1
                else probabilities[:, class_index] <= upper
            )
            if not mask.any():
                continue
            rows.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": int(mask.sum()),
                    "meanForecast": float(np.mean(probabilities[mask, class_index])),
                    "observedFrequency": float(np.mean(labels[mask] == class_value)),
                }
            )
        result[str(int(class_value))] = rows
    return result


def _metrics(labels: np.ndarray, probabilities: np.ndarray, bins: int) -> dict[str, Any]:
    predicted = CLASSES[np.argmax(probabilities, axis=1)]
    return {
        "samples": int(len(labels)),
        "classCounts": {str(int(value)): int(np.sum(labels == value)) for value in CLASSES},
        "brier": float(np.mean(_brier_rows(labels, probabilities))),
        "logLoss": float(log_loss(labels, probabilities, labels=CLASSES)),
        "balancedAccuracy": float(balanced_accuracy_score(labels, predicted)),
        "confusionMatrix": confusion_matrix(labels, predicted, labels=CLASSES).tolist(),
        "reliability": _reliability(labels, probabilities, bins),
    }


def _paired_block_interval(
    paired_values: np.ndarray,
    *,
    block_size: int,
    bootstrap_samples: int,
    random_state: int,
    alpha: float,
) -> tuple[float, float]:
    if len(paired_values) == 0:
        raise ValueError("paired bootstrap requires observations")
    rng = np.random.default_rng(random_state)
    block_size = max(1, min(block_size, len(paired_values)))
    estimates = np.empty(bootstrap_samples, dtype=np.float64)
    for iteration in range(bootstrap_samples):
        sampled: list[float] = []
        while len(sampled) < len(paired_values):
            start = int(rng.integers(0, len(paired_values)))
            for offset in range(block_size):
                sampled.append(float(paired_values[(start + offset) % len(paired_values)]))
                if len(sampled) == len(paired_values):
                    break
        estimates[iteration] = np.mean(sampled)
    low, high = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def _folds(data: BenchmarkData, config: EvaluatorConfig) -> list[dict[str, np.ndarray | int]]:
    folds: list[dict[str, np.ndarray | int]] = []
    first_test = config.min_fit_rows + config.calibration_rows
    for test_start in range(first_test, len(data.labels) - config.test_rows + 1, config.step_rows):
        test_end = test_start + config.test_rows
        calibration_start = test_start - config.calibration_rows
        calibration_boundary = data.decision_times_ms[calibration_start]
        test_boundary = data.decision_times_ms[test_start]
        fit_indices = np.flatnonzero(
            (np.arange(len(data.labels)) < calibration_start)
            & (data.label_available_times_ms <= calibration_boundary)
        )
        calibration_indices = np.flatnonzero(
            (np.arange(len(data.labels)) >= calibration_start)
            & (np.arange(len(data.labels)) < test_start)
            & (data.label_available_times_ms <= test_boundary)
        )
        test_indices = np.arange(test_start, test_end, dtype=np.int64)
        if len(fit_indices) < config.min_fit_rows - config.label_horizon_bars:
            continue
        if len(calibration_indices) < max(20, config.calibration_rows // 2):
            continue
        folds.append(
            {
                "fold": len(folds),
                "fit": fit_indices,
                "calibration": calibration_indices,
                "test": test_indices,
                "fitThroughMs": int(data.decision_times_ms[fit_indices[-1]]),
                "calibrationStartMs": int(data.decision_times_ms[calibration_indices[0]]),
                "calibrationThroughMs": int(data.decision_times_ms[calibration_indices[-1]]),
                "testStartMs": int(data.decision_times_ms[test_indices[0]]),
                "testEndMs": int(data.decision_times_ms[test_indices[-1]]),
            }
        )
    if not folds:
        raise ValueError("insufficient rows for a complete outer walk-forward fold")
    return folds


def evaluate_walk_forward(
    data: BenchmarkData,
    config: EvaluatorConfig,
    *,
    evaluation_end_ms: int | None = None,
) -> dict[str, Any]:
    config.validate()
    data.validate()
    if evaluation_end_ms is not None:
        mask = data.decision_times_ms <= int(evaluation_end_ms)
        data = BenchmarkData(
            data.features[mask],
            data.labels[mask],
            data.decision_times_ms[mask],
            data.label_available_times_ms[mask],
            data.last_returns[mask],
        )
        data.validate()

    folds = _folds(data, config)
    model_outputs: dict[str, list[np.ndarray]] = {name: [] for name in DECLARED_MODEL_FAMILY}
    baseline_outputs: dict[str, list[np.ndarray]] = {name: [] for name in DECLARED_BASELINES}
    labels_by_fold: list[np.ndarray] = []
    timestamps_by_fold: list[np.ndarray] = []
    fold_contracts: list[dict[str, Any]] = []

    for fold in folds:
        fit_indices = np.asarray(fold["fit"])
        calibration_indices = np.asarray(fold["calibration"])
        test_indices = np.asarray(fold["test"])
        labels_by_fold.append(data.labels[test_indices])
        timestamps_by_fold.append(data.decision_times_ms[test_indices])
        fold_contracts.append({key: value for key, value in fold.items() if key not in {"fit", "calibration", "test"}} | {
            "fitRows": int(len(fit_indices)),
            "calibrationRows": int(len(calibration_indices)),
            "testRows": int(len(test_indices)),
            "maxFitLabelAvailableMs": int(np.max(data.label_available_times_ms[fit_indices])),
            "maxCalibrationLabelAvailableMs": int(np.max(data.label_available_times_ms[calibration_indices])),
        })
        for model_name in DECLARED_MODEL_FAMILY:
            estimator = _build_estimator(model_name, config.random_state + int(fold["fold"]))
            estimator.fit(data.features[fit_indices], data.labels[fit_indices])
            calibration_base = _aligned_probabilities(estimator, data.features[calibration_indices])
            calibrator = PriorOnlyCalibrator(config.random_state + int(fold["fold"]))
            calibrator.fit(calibration_base, data.labels[calibration_indices])
            test_base = _aligned_probabilities(estimator, data.features[test_indices])
            model_outputs[model_name].append(calibrator.transform(test_base))
        fold_baselines = _baseline_probabilities(data, test_indices, config)
        for baseline_name in DECLARED_BASELINES:
            baseline_outputs[baseline_name].append(fold_baselines[baseline_name])

    labels = np.concatenate(labels_by_fold)
    timestamps = np.concatenate(timestamps_by_fold)
    if len(np.unique(timestamps)) != len(timestamps):
        raise ValueError("outer test folds overlap; identical timestamp comparison is not possible")
    probabilities = {
        name: np.concatenate(values)
        for name, values in (model_outputs | baseline_outputs).items()
    }
    metrics = {
        name: _metrics(labels, model_probabilities, config.reliability_bins)
        for name, model_probabilities in probabilities.items()
    }
    strongest_baseline = min(DECLARED_BASELINES, key=lambda name: metrics[name]["brier"])
    baseline_loss = _brier_rows(labels, probabilities[strongest_baseline])
    timestamp_hash = hashlib.sha256(timestamps.astype("<i8").tobytes()).hexdigest()

    trials: list[dict[str, Any]] = []
    per_candidate_alpha = config.familywise_alpha / len(DECLARED_MODEL_FAMILY)
    for name in DECLARED_MODEL_FAMILY:
        candidate_loss = _brier_rows(labels, probabilities[name])
        paired_lift = baseline_loss - candidate_loss
        interval = _paired_block_interval(
            paired_lift,
            block_size=config.block_size_rows,
            bootstrap_samples=config.bootstrap_samples,
            random_state=config.random_state,
            alpha=per_candidate_alpha,
        )
        trials.append(
            {
                "trial": name,
                "kind": "candidate",
                "status": "positive" if interval[0] > 0 else ("negative" if interval[1] < 0 else "inconclusive"),
                "metrics": metrics[name],
                "comparisonBaseline": strongest_baseline,
                "pairedBrierLift": float(np.mean(paired_lift)),
                "pairedBrierLiftFamilywiseCi": list(interval),
                "familywiseControl": {
                    "method": "Bonferroni over declared candidate family",
                    "familyAlpha": config.familywise_alpha,
                    "perCandidateAlpha": per_candidate_alpha,
                    "confidenceLevel": 1.0 - per_candidate_alpha,
                },
                "coverage": 1.0,
                "evaluationTimestampSha256": timestamp_hash,
            }
        )
    for name in DECLARED_BASELINES:
        trials.append(
            {
                "trial": name,
                "kind": "baseline",
                "status": "reference",
                "metrics": metrics[name],
                "coverage": 1.0,
                "evaluationTimestampSha256": timestamp_hash,
            }
        )

    selected = min(DECLARED_MODEL_FAMILY, key=lambda name: metrics[name]["brier"])
    selected_trial = next(trial for trial in trials if trial["trial"] == selected)
    class_counts = metrics[selected]["classCounts"]
    checks = {
        "declaredFamilyComplete": {trial["trial"] for trial in trials}
        == set(DECLARED_MODEL_FAMILY + DECLARED_BASELINES),
        "identicalTimestampCoverage": all(len(probabilities[name]) == len(timestamps) for name in probabilities),
        "minimumSamples": len(timestamps) >= config.minimum_gate_samples,
        "minimumClassSupport": min(class_counts.values()) >= config.minimum_class_samples,
        "brierBetterThanStrongestBaseline": metrics[selected]["brier"] < metrics[strongest_baseline]["brier"],
        "logLossBetterThanStrongestBaseline": metrics[selected]["logLoss"] < metrics[strongest_baseline]["logLoss"],
        "pairedIntervalExcludesZero": selected_trial["pairedBrierLiftFamilywiseCi"][0] > 0,
    }
    promotion_passed = all(checks.values())
    return {
        "evaluatorVersion": EVALUATOR_VERSION,
        "scope": "BTCUSDT 4h next-bar direction benchmark",
        "declaredModelFamily": list(DECLARED_MODEL_FAMILY),
        "declaredBaselines": list(DECLARED_BASELINES),
        "folds": fold_contracts,
        "evaluationRows": int(len(timestamps)),
        "evaluationTimestampsMs": timestamps.tolist(),
        "coverage": 1.0,
        "strongestBaseline": strongest_baseline,
        "trials": trials,
        "selectedCandidate": selected,
        "promotionGate": {
            "passed": promotion_passed,
            "promotionAllowed": promotion_passed,
            "checks": checks,
            "policy": "fail-closed; no registry or production mutation is performed",
        },
        "negativeResultIsValid": True,
    }


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_provenance(repo_dir: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repo_dir, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def build_manifest(
    data: BenchmarkData,
    config: EvaluatorConfig,
    *,
    decision_cutoff_ms: int,
    source: str,
) -> dict[str, Any]:
    here = Path(__file__).resolve()
    contract = here.with_name("research_contract.py")
    db_params = get_db_params() if source == "postgresql-readonly" else {}
    provenance = {
        "source": source,
        "datasetSha256": dataset_sha256(data),
        "rowCount": int(len(data.labels)),
        "featureDim": int(data.features.shape[1]),
        "firstDecisionTimeMs": int(data.decision_times_ms[0]),
        "lastDecisionTimeMs": int(data.decision_times_ms[-1]),
        "decisionCutoffMs": int(decision_cutoff_ms),
        "featureSource": "WindowClassificationDatasets.FeatureVector excluding ActiveRuleCount for every bar",
        "labelSource": "PriceTargets.TargetDirection4h",
        "rowJoin": "Symbol + Timeframe + WindowEndMs=OpenTimeMs",
        "closedBarPredicate": "WindowEndMs + 4h <= cutoff",
        "realizedLabelPredicate": "decisionTimeMs + 4h <= cutoff",
    }
    if db_params:
        provenance["databaseIdentity"] = {
            "host": db_params["host"],
            "port": db_params["port"],
            "database": db_params["database"],
            "passwordExcluded": True,
        }
    manifest = ResearchManifest(
        experiment="btc-4h-next-bar-ml-evidence",
        symbol="BTCUSDT",
        timeframe="4h",
        outcome_price_basis="signal-bar-close-to-next-bar-close",
    )
    return manifest.to_dict(
        parameters={
            **asdict(config),
            "declaredModelFamily": list(DECLARED_MODEL_FAMILY),
            "declaredBaselines": list(DECLARED_BASELINES),
            "featureSchema": "WindowClassificationDataset 35 features/bar flattened; ActiveRuleCount offset 29 removed because it lacks point-in-time rule-version lineage",
            "label": "PriceTargets.TargetDirection4h close-to-close (not triple-barrier); available at next 4h bar close",
        },
        data_provenance=provenance,
        code_provenance={
            "evaluatorSha256": _file_hash(here),
            "researchContractSha256": _file_hash(contract),
            "git": _git_provenance(here.parent),
        },
        runtime_dependencies={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikitLearn": sklearn.__version__,
            "psycopg2": psycopg2.__version__,
        },
    )


def write_evidence(output_dir: Path, manifest: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_hash = manifest["manifestSha256"]
    report = {"manifest": manifest, "evaluation": evaluation}
    report["reportSha256"] = hashlib.sha256(_canonical_json(report).encode("utf-8")).hexdigest()
    report_path = output_dir / f"{manifest_hash}.report.json"
    ledger_path = output_dir / f"{manifest_hash}.trials.jsonl"
    report_content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ledger_content = "".join(
        _canonical_json({"manifestSha256": manifest_hash, **trial}) + "\n"
        for trial in evaluation["trials"]
    )
    for path, content in ((report_path, report_content), (ledger_path, ledger_content)):
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite immutable evidence artifact: {path}")
        if not path.exists():
            path.write_text(content, encoding="utf-8")
    return {"report": report_path, "trialLedger": ledger_path}


def _load_npz(path: Path, decision_cutoff_ms: int) -> BenchmarkData:
    with np.load(path, allow_pickle=False) as payload:
        required = {"X", "y", "decision_ms", "label_available_ms", "last_return"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"NPZ missing arrays: {sorted(missing)}")
        data = BenchmarkData(
            np.asarray(payload["X"], dtype=np.float64),
            np.asarray(payload["y"], dtype=np.int8),
            np.asarray(payload["decision_ms"], dtype=np.int64),
            np.asarray(payload["label_available_ms"], dtype=np.int64),
            np.asarray(payload["last_return"], dtype=np.float64),
        )
    mask = (data.decision_times_ms <= decision_cutoff_ms) & (
        data.label_available_times_ms <= decision_cutoff_ms
    )
    filtered = BenchmarkData(
        data.features[mask],
        data.labels[mask],
        data.decision_times_ms[mask],
        data.label_available_times_ms[mask],
        data.last_returns[mask],
    )
    filtered.validate()
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="BTCUSDT 4h next-bar common ML evidence evaluator")
    parser.add_argument("--decision-cutoff-ms", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-npz", help="optional replay dataset; otherwise use read-only PostgreSQL")
    parser.add_argument("--min-fit-rows", type=int, default=500)
    parser.add_argument("--calibration-rows", type=int, default=120)
    parser.add_argument("--test-rows", type=int, default=120)
    parser.add_argument("--step-rows", type=int, default=120)
    args = parser.parse_args()
    config = EvaluatorConfig(
        min_fit_rows=args.min_fit_rows,
        calibration_rows=args.calibration_rows,
        test_rows=args.test_rows,
        step_rows=args.step_rows,
    )
    if args.input_npz:
        data = _load_npz(Path(args.input_npz), args.decision_cutoff_ms)
        source = f"npz:{Path(args.input_npz).resolve()}"
    else:
        data = load_btc_4h_benchmark(
            decision_cutoff_ms=args.decision_cutoff_ms,
            window_size=config.window_size,
        )
        source = "postgresql-readonly"
    manifest = build_manifest(
        data,
        config,
        decision_cutoff_ms=args.decision_cutoff_ms,
        source=source,
    )
    evaluation = evaluate_walk_forward(data, config)
    paths = write_evidence(Path(args.output_dir), manifest, evaluation)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
