#!/usr/bin/env python3
"""Reproducible BTCUSDT 4h feature-group contribution evaluator.

This is a read-only research diagnostic.  It measures whether declared groups of
technical features add out-of-sample classification information on a fixed
chronological protocol.  It deliberately makes no trading or PnL claim.
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
from typing import Any, Iterable, Iterator

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from db_config import get_db_connection
from research_contract import ResearchManifest


CLASSES = np.array([-1, 0, 1], dtype=np.int8)
INTERVAL_MS = 4 * 60 * 60 * 1000
EVALUATOR_VERSION = "btc-4h-feature-group-ablation-v1"

# This order is the persisted WindowDatasetService schema. ActiveRuleCount is
# retained here only so its raw offset can be identified and excluded.
STORED_FEATURE_NAMES = (
    "CloseZscore",
    "ClosePctChange1",
    "ClosePctChange4",
    "ClosePctChange24",
    "HighLowRangePct",
    "BodyPct",
    "UpperWickPct",
    "LowerWickPct",
    "Rsi14",
    "Rsi14Slope",
    "MacdNorm",
    "MacdSignalNorm",
    "MacdHistogramNorm",
    "Ema12Dist",
    "Ema26Dist",
    "Ema50Dist",
    "Ema200Dist",
    "Sma50Dist",
    "Sma200Dist",
    "BollingerWidth",
    "BollingerPosition",
    "Atr14Pct",
    "ObvEmaDist",
    "VwapDist",
    "RollingVwapDist",
    "VolumeZscore",
    "VolumeSma20Ratio",
    "TakerBuyRatio",
    "RecentPatternEncoded",
    "ActiveRuleCount",
    "HourSin",
    "HourCos",
    "DayOfWeekSin",
    "DayOfWeekCos",
    "IsWeekend",
)

EXCLUDED_STORED_FEATURES = {
    "ActiveRuleCount": (
        "mutable current rule-admin state without point-in-time rule-version lineage"
    )
}

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "price_returns": (
        "CloseZscore",
        "ClosePctChange1",
        "ClosePctChange4",
        "ClosePctChange24",
    ),
    "candle_geometry": (
        "HighLowRangePct",
        "BodyPct",
        "UpperWickPct",
        "LowerWickPct",
    ),
    "trend": (
        "Ema12Dist",
        "Ema26Dist",
        "Ema50Dist",
        "Ema200Dist",
        "Sma50Dist",
        "Sma200Dist",
    ),
    "momentum": (
        "Rsi14",
        "Rsi14Slope",
        "MacdNorm",
        "MacdSignalNorm",
        "MacdHistogramNorm",
    ),
    "volatility": (
        "BollingerWidth",
        "BollingerPosition",
        "Atr14Pct",
    ),
    "volume": (
        "ObvEmaDist",
        "VwapDist",
        "RollingVwapDist",
        "VolumeZscore",
        "VolumeSma20Ratio",
        "TakerBuyRatio",
    ),
    "pattern": ("RecentPatternEncoded",),
    "time": (
        "HourSin",
        "HourCos",
        "DayOfWeekSin",
        "DayOfWeekCos",
        "IsWeekend",
    ),
}

CAUSAL_FEATURE_NAMES = tuple(
    name for name in STORED_FEATURE_NAMES if name not in EXCLUDED_STORED_FEATURES
)


@dataclass(frozen=True)
class AblationConfig:
    window_size: int = 5
    min_fit_rows: int = 1000
    test_rows: int = 240
    step_rows: int = 240
    minimum_test_rows: int = 60
    laplace_alpha: float = 1.0
    logistic_c: float = 1.0
    max_iter: int = 1000
    block_size_rows: int = 12
    bootstrap_samples: int = 1000
    familywise_alpha: float = 0.05
    random_state: int = 42

    def validate(self) -> None:
        positive = (
            "window_size",
            "min_fit_rows",
            "test_rows",
            "step_rows",
            "minimum_test_rows",
            "max_iter",
            "block_size_rows",
            "bootstrap_samples",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.minimum_test_rows > self.test_rows:
            raise ValueError("minimum_test_rows cannot exceed test_rows")
        if self.laplace_alpha <= 0 or self.logistic_c <= 0:
            raise ValueError("laplace_alpha and logistic_c must be positive")
        if not 0 < self.familywise_alpha < 1:
            raise ValueError("familywise_alpha must be in (0, 1)")


@dataclass(frozen=True)
class AblationData:
    features: np.ndarray
    labels: np.ndarray
    decision_times_ms: np.ndarray
    label_available_times_ms: np.ndarray

    def validate(self) -> None:
        rows = len(self.features)
        per_bar = len(CAUSAL_FEATURE_NAMES)
        if (
            self.features.ndim != 2
            or rows == 0
            or self.features.shape[1] == 0
            or self.features.shape[1] % per_bar != 0
        ):
            raise ValueError(f"features must be non-empty with a multiple of {per_bar} columns")
        for name in ("labels", "decision_times_ms", "label_available_times_ms"):
            value = getattr(self, name)
            if value.ndim != 1 or len(value) != rows:
                raise ValueError(f"{name} must align with features")
        if not np.isfinite(self.features).all():
            raise ValueError("features must be finite")
        if not np.isin(self.labels, CLASSES).all():
            raise ValueError("labels must be -1, 0, or 1")
        if np.any(self.decision_times_ms[1:] <= self.decision_times_ms[:-1]):
            raise ValueError("decision timestamps must be strictly increasing")
        if np.any(self.label_available_times_ms < self.decision_times_ms):
            raise ValueError("label availability cannot precede the decision")


@dataclass(frozen=True)
class Fold:
    number: int
    train_indices: np.ndarray
    test_indices: np.ndarray


def validate_feature_groups() -> None:
    flattened = [name for names in FEATURE_GROUPS.values() for name in names]
    if len(flattened) != len(set(flattened)):
        raise ValueError("feature groups overlap")
    if set(flattened) != set(CAUSAL_FEATURE_NAMES):
        missing = sorted(set(CAUSAL_FEATURE_NAMES) - set(flattened))
        unknown = sorted(set(flattened) - set(CAUSAL_FEATURE_NAMES))
        raise ValueError(f"feature groups must partition causal schema; missing={missing}, unknown={unknown}")


def group_columns(window_size: int) -> dict[str, np.ndarray]:
    """Return flattened causal-vector columns for every declared feature group."""
    validate_feature_groups()
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    per_bar = len(CAUSAL_FEATURE_NAMES)
    offsets = {name: index for index, name in enumerate(CAUSAL_FEATURE_NAMES)}
    result: dict[str, np.ndarray] = {}
    for group, names in FEATURE_GROUPS.items():
        result[group] = np.asarray(
            [bar * per_bar + offsets[name] for bar in range(window_size) for name in names],
            dtype=np.int64,
        )
    return result


def flatten_causal_window(stored: np.ndarray, window_size: int) -> np.ndarray:
    """Preserve the stored bar order while excluding every noncausal field."""
    expected = window_size * len(STORED_FEATURE_NAMES)
    matrix = np.asarray(stored, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != expected:
        raise ValueError(f"stored feature matrix must have {expected} columns")
    cube = matrix.reshape(len(matrix), window_size, len(STORED_FEATURE_NAMES))
    keep = [index for index, name in enumerate(STORED_FEATURE_NAMES) if name in CAUSAL_FEATURE_NAMES]
    return cube[:, :, keep].reshape(len(matrix), window_size * len(keep))


def load_btc_4h_data(*, decision_cutoff_ms: int, window_size: int = 5) -> AblationData:
    """Read the fixed BTC 4h benchmark from PostgreSQL without modifying it."""
    feature_dim = window_size * len(STORED_FEATURE_NAMES)
    query = """
        SELECT w."FeatureVector", p."TargetDirection4h", w."WindowEndMs"
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
        raise ValueError("BTCUSDT 4h feature-ablation query returned no rows")

    stored = np.asarray([row[0] for row in rows], dtype=np.float64)
    features = flatten_causal_window(stored, window_size)
    labels = np.asarray([row[1] for row in rows], dtype=np.int8)
    signal_open = np.asarray([row[2] for row in rows], dtype=np.int64)
    decisions = signal_open + INTERVAL_MS
    availability = decisions + INTERVAL_MS
    final = availability <= int(decision_cutoff_ms)
    data = AblationData(features[final], labels[final], decisions[final], availability[final])
    data.validate()
    return data


def chronological_folds(data: AblationData, config: AblationConfig) -> Iterator[Fold]:
    """Yield expanding folds whose training outcomes exist before each test begins."""
    data.validate()
    config.validate()
    cursor = config.min_fit_rows
    number = 0
    while cursor < len(data.features):
        stop = min(cursor + config.test_rows, len(data.features))
        if stop - cursor < config.minimum_test_rows:
            break
        test = np.arange(cursor, stop, dtype=np.int64)
        test_start = int(data.decision_times_ms[cursor])
        train = np.flatnonzero(
            (data.decision_times_ms < test_start)
            & (data.label_available_times_ms <= test_start)
        )
        if len(train) >= config.min_fit_rows:
            yield Fold(number, train, test)
            number += 1
        cursor += config.step_rows


def _trial_columns(config: AblationConfig) -> dict[str, np.ndarray | None]:
    groups = group_columns(config.window_size)
    all_columns = np.arange(
        config.window_size * len(CAUSAL_FEATURE_NAMES), dtype=np.int64
    )
    trials: dict[str, np.ndarray | None] = {"rolling_class_prior": None, "full": all_columns}
    for name, columns in groups.items():
        trials[f"only:{name}"] = columns
        trials[f"without:{name}"] = np.setdiff1d(all_columns, columns, assume_unique=True)
    return trials


def _aligned_probabilities(model: Any, features: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict_proba(features), dtype=np.float64)
    result = np.full((len(features), len(CLASSES)), 1e-12, dtype=np.float64)
    for source, label in enumerate(model.classes_):
        target = int(np.flatnonzero(CLASSES == label)[0])
        result[:, target] = raw[:, source]
    return result / result.sum(axis=1, keepdims=True)


def _prior_probabilities(labels: np.ndarray, rows: int, alpha: float) -> np.ndarray:
    counts = np.asarray([(labels == label).sum() for label in CLASSES], dtype=np.float64)
    prior = (counts + alpha) / (len(labels) + alpha * len(CLASSES))
    return np.repeat(prior[None, :], rows, axis=0)


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    one_hot = (labels[:, None] == CLASSES[None, :]).astype(np.float64)
    predictions = CLASSES[np.argmax(probabilities, axis=1)]
    return {
        "rows": int(len(labels)),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "logLoss": float(log_loss(labels, probabilities, labels=CLASSES)),
        "balancedAccuracy": float(balanced_accuracy_score(labels, predictions)),
        "predictedClassCounts": {
            str(int(label)): int((predictions == label).sum()) for label in CLASSES
        },
    }


def _brier_losses(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    one_hot = (labels[:, None] == CLASSES[None, :]).astype(np.float64)
    return np.sum((probabilities - one_hot) ** 2, axis=1)


def _block_bootstrap_ci(
    values: np.ndarray,
    *,
    block_size: int,
    samples: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        raise ValueError("cannot bootstrap an empty vector")
    rng = np.random.default_rng(seed)
    blocks_needed = int(np.ceil(len(values) / block_size))
    max_start = max(1, len(values) - block_size + 1)
    estimates = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        starts = rng.integers(0, max_start, size=blocks_needed)
        indices = np.concatenate(
            [np.arange(start, min(start + block_size, len(values))) for start in starts]
        )[: len(values)]
        estimates[sample] = float(np.mean(values[indices]))
    return (
        float(np.quantile(estimates, alpha / 2)),
        float(np.quantile(estimates, 1 - alpha / 2)),
    )


def evaluate_feature_groups(data: AblationData, config: AblationConfig) -> dict[str, Any]:
    """Evaluate the declared full/group-only/leave-one-group-out trial family."""
    data.validate()
    config.validate()
    expected_dim = config.window_size * len(CAUSAL_FEATURE_NAMES)
    if data.features.shape[1] != expected_dim:
        raise ValueError(
            f"config expects {expected_dim} feature columns, got {data.features.shape[1]}"
        )
    trials = _trial_columns(config)
    labels_parts: list[np.ndarray] = []
    times_parts: list[np.ndarray] = []
    probability_parts: dict[str, list[np.ndarray]] = {name: [] for name in trials}
    fold_records: list[dict[str, Any]] = []

    for fold in chronological_folds(data, config):
        train_y = data.labels[fold.train_indices]
        test_y = data.labels[fold.test_indices]
        if len(np.unique(train_y)) < 2:
            continue
        labels_parts.append(test_y)
        times_parts.append(data.decision_times_ms[fold.test_indices])
        probability_parts["rolling_class_prior"].append(
            _prior_probabilities(train_y, len(fold.test_indices), config.laplace_alpha)
        )
        for name, columns in trials.items():
            if columns is None:
                continue
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            C=config.logistic_c,
                            max_iter=config.max_iter,
                            class_weight="balanced",
                            random_state=config.random_state,
                        ),
                    ),
                ]
            )
            model.fit(data.features[fold.train_indices][:, columns], train_y)
            probability_parts[name].append(
                _aligned_probabilities(model, data.features[fold.test_indices][:, columns])
            )
        fold_records.append(
            {
                "fold": fold.number,
                "fitRows": int(len(fold.train_indices)),
                "testRows": int(len(fold.test_indices)),
                "maxFitLabelAvailableMs": int(data.label_available_times_ms[fold.train_indices].max()),
                "testStartMs": int(data.decision_times_ms[fold.test_indices].min()),
                "testEndMs": int(data.decision_times_ms[fold.test_indices].max()),
            }
        )

    if not labels_parts:
        raise ValueError("configuration produced no evaluable chronological folds")
    labels = np.concatenate(labels_parts)
    times = np.concatenate(times_parts)
    probabilities = {name: np.concatenate(parts) for name, parts in probability_parts.items()}
    timestamp_hash = hashlib.sha256(times.astype("<i8").tobytes()).hexdigest()
    baseline_loss = _brier_losses(labels, probabilities["rolling_class_prior"])
    full_loss = _brier_losses(labels, probabilities["full"])
    # Every non-baseline trial is compared with the common prior, and each
    # leave-one-group-out trial is additionally compared with the full model.
    comparisons = (len(trials) - 1) + len(FEATURE_GROUPS)
    adjusted_alpha = config.familywise_alpha / comparisons

    trial_results: dict[str, dict[str, Any]] = {}
    for index, (name, trial_probs) in enumerate(probabilities.items()):
        metrics = _metrics(labels, trial_probs)
        loss = _brier_losses(labels, trial_probs)
        lift = baseline_loss - loss
        low, high = _block_bootstrap_ci(
            lift,
            block_size=config.block_size_rows,
            samples=config.bootstrap_samples,
            alpha=adjusted_alpha,
            seed=config.random_state + index,
        )
        status = "positive" if low > 0 else "negative" if high < 0 else "inconclusive"
        trial_results[name] = {
            **metrics,
            "coverage": 1.0,
            "evaluationTimestampSha256": timestamp_hash,
            "brierLiftVsRollingPrior": float(np.mean(lift)),
            "brierLiftFamilywiseCi": [low, high],
            "statusVsRollingPrior": status,
        }

    group_results: dict[str, dict[str, Any]] = {}
    for index, group in enumerate(FEATURE_GROUPS):
        without_loss = _brier_losses(labels, probabilities[f"without:{group}"])
        contribution = without_loss - full_loss
        low, high = _block_bootstrap_ci(
            contribution,
            block_size=config.block_size_rows,
            samples=config.bootstrap_samples,
            alpha=adjusted_alpha,
            seed=config.random_state + 100 + index,
        )
        status = "positive" if low > 0 else "harmful" if high < 0 else "inconclusive"
        group_results[group] = {
            "features": list(FEATURE_GROUPS[group]),
            "columns": group_columns(config.window_size)[group].tolist(),
            "groupOnly": trial_results[f"only:{group}"],
            "withoutGroup": trial_results[f"without:{group}"],
            "incrementalBrierContribution": float(np.mean(contribution)),
            "incrementalContributionFamilywiseCi": [low, high],
            "incrementalStatus": status,
        }

    eligible = max(0, len(data.features) - config.min_fit_rows)
    return {
        "evaluatorVersion": EVALUATOR_VERSION,
        "purpose": "feature-group contribution and ablation; no trading or PnL claim",
        "model": "scaled multinomial logistic regression with fixed hyperparameters",
        "featureObservation": f"{config.window_size} closed 4h bars ending at each decision",
        "causalFeatureCountPerBar": len(CAUSAL_FEATURE_NAMES),
        "featureVectorDimension": config.window_size * len(CAUSAL_FEATURE_NAMES),
        "excludedStoredFeatures": EXCLUDED_STORED_FEATURES,
        "evaluationRows": int(len(labels)),
        "eligibleRowsAfterWarmup": int(eligible),
        "protocolCoverage": float(len(labels) / eligible) if eligible else 0.0,
        "folds": fold_records,
        "baseline": trial_results["rolling_class_prior"],
        "fullModel": trial_results["full"],
        "groups": group_results,
        "trials": trial_results,
        "multipleComparison": {
            "method": "Bonferroni familywise intervals over declared Brier comparisons",
            "familywiseAlpha": config.familywise_alpha,
            "comparisons": comparisons,
            "perComparisonAlpha": adjusted_alpha,
        },
        "negativeResultIsValid": True,
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def dataset_sha256(data: AblationData) -> str:
    digest = hashlib.sha256()
    for array in (
        data.features.astype("<f8"),
        data.labels.astype("i1"),
        data.decision_times_ms.astype("<i8"),
        data.label_available_times_ms.astype("<i8"),
    ):
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=Path(__file__).parent, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def build_manifest(
    data: AblationData,
    config: AblationConfig,
    *,
    decision_cutoff_ms: int,
    source: str,
) -> dict[str, Any]:
    contract = ResearchManifest(experiment="btc-4h-feature-group-ablation")
    return contract.to_dict(
        parameters={
            **asdict(config),
            "evaluatorVersion": EVALUATOR_VERSION,
            "label": "TargetDirection4h",
            "featureObservation": f"{config.window_size} closed bars",
            "storedFeatureNames": list(STORED_FEATURE_NAMES),
            "causalFeatureNames": list(CAUSAL_FEATURE_NAMES),
            "featureGroups": {name: list(features) for name, features in FEATURE_GROUPS.items()},
            "excludedStoredFeatures": EXCLUDED_STORED_FEATURES,
        },
        data_provenance={
            "source": source,
            "readOnly": True,
            "decisionCutoffMs": int(decision_cutoff_ms),
            "rows": len(data.features),
            "firstDecisionMs": int(data.decision_times_ms[0]),
            "lastDecisionMs": int(data.decision_times_ms[-1]),
            "datasetSha256": dataset_sha256(data),
        },
        code_provenance={
            "gitCommit": _git_value("rev-parse", "HEAD"),
            "gitWorkingTree": "dirty" if _git_value("status", "--porcelain") else "clean",
            "evaluatorSha256": _file_sha256(Path(__file__)),
            "researchContractSha256": _file_sha256(Path(__file__).with_name("research_contract.py")),
        },
        runtime_dependencies={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikitLearn": sklearn.__version__,
        },
    )


def _write_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"immutable evidence conflict: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def write_evidence(
    output_dir: Path, manifest: dict[str, Any], evaluation: dict[str, Any]
) -> dict[str, Path]:
    evidence_id = manifest["manifestSha256"]
    manifest_path = output_dir / f"{evidence_id}.manifest.json"
    report_path = output_dir / f"{evidence_id}.report.json"
    report = {"manifestSha256": evidence_id, "evaluation": evaluation}
    _write_immutable(manifest_path, (json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    _write_immutable(report_path, (json.dumps(report, indent=2) + "\n").encode("utf-8"))
    return {"manifest": manifest_path, "report": report_path}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-cutoff-ms", type=int, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("docs/research/evidence/feature-groups")
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = AblationConfig(bootstrap_samples=args.bootstrap_samples)
    data = load_btc_4h_data(
        decision_cutoff_ms=args.decision_cutoff_ms, window_size=config.window_size
    )
    manifest = build_manifest(
        data,
        config,
        decision_cutoff_ms=args.decision_cutoff_ms,
        source="postgresql:WindowClassificationDatasets+PriceTargets",
    )
    evaluation = evaluate_feature_groups(data, config)
    paths = write_evidence(args.output_dir, manifest, evaluation)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
