#!/usr/bin/env python3
"""Leakage-safe temporal archetype fitting and out-of-sample evaluation.

This module intentionally does not write the application database.  A model version
is a frozen, file-backed research artifact: its scaler, centroids, distance gates and
cluster outcome probabilities are learned from rows available at ``train_through_ms``.
Later windows are assigned with that frozen state and can never refit the version.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


SCHEMA_VERSION = "btc-archetype-temporal-v1"
CLASS_VALUES = np.array([-1, 0, 1], dtype=np.int8)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _as_feature_matrix(value: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("features must be a non-empty 2D matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("features contain NaN or infinite values")
    return matrix


def _as_labels(value: np.ndarray | Iterable[int], expected: int) -> np.ndarray:
    labels = np.asarray(value, dtype=np.int8)
    if labels.ndim != 1 or len(labels) != expected:
        raise ValueError("labels must be a 1D array aligned with features")
    if not np.isin(labels, CLASS_VALUES).all():
        raise ValueError("labels must be one of -1, 0, 1")
    return labels


def _as_times(value: np.ndarray | Iterable[int], expected: int) -> np.ndarray:
    times = np.asarray(value, dtype=np.int64)
    if times.ndim != 1 or len(times) != expected:
        raise ValueError("end_times_ms must be a 1D array aligned with features")
    if len(times) > 1 and np.any(times[1:] < times[:-1]):
        raise ValueError("end_times_ms must be chronological")
    return times


@dataclass(frozen=True)
class TemporalArchetypeModel:
    schema_version: str
    version_id: str
    symbol: str
    timeframe: str
    window_size: int
    horizon: str
    train_through_ms: int
    feature_dim: int
    n_clusters: int
    random_state: int
    training_rows: int
    training_data_hash: str
    training_label_policy: str
    scaler_mean: list[float]
    scaler_scale: list[float]
    centroids_scaled: list[list[float]]
    archetype_ids: list[str]
    train_member_counts: list[int]
    distance_thresholds: list[float]
    class_probabilities: list[list[float]]
    distance_quantile: float
    min_cluster_members: int
    laplace_alpha: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TemporalArchetypeModel":
        model = cls(**payload)
        if model.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported archetype schema: {model.schema_version}")
        model._validate()
        return model

    def _validate(self) -> None:
        if self.feature_dim <= 0 or self.n_clusters <= 0:
            raise ValueError("invalid model dimensions")
        if len(self.scaler_mean) != self.feature_dim or len(self.scaler_scale) != self.feature_dim:
            raise ValueError("scaler dimension mismatch")
        if len(self.centroids_scaled) != self.n_clusters:
            raise ValueError("centroid count mismatch")
        if any(len(row) != self.feature_dim for row in self.centroids_scaled):
            raise ValueError("centroid dimension mismatch")
        for field in (
            self.archetype_ids,
            self.train_member_counts,
            self.distance_thresholds,
            self.class_probabilities,
        ):
            if len(field) != self.n_clusters:
                raise ValueError("cluster metadata count mismatch")
        if any(len(row) != len(CLASS_VALUES) for row in self.class_probabilities):
            raise ValueError("class probability dimension mismatch")


@dataclass(frozen=True)
class AssignmentBatch:
    cluster_indices: np.ndarray
    archetype_ids: tuple[str | None, ...]
    distances: np.ndarray
    accepted: np.ndarray
    class_probabilities: np.ndarray


def _canonical_cluster_order(centroids: np.ndarray) -> np.ndarray:
    """Return a deterministic order independent of KMeans' arbitrary label numbers."""
    rounded = np.round(centroids, decimals=12)
    keys = [tuple(row.tolist()) for row in rounded]
    return np.asarray(sorted(range(len(keys)), key=keys.__getitem__), dtype=np.int32)


def _training_hash(features: np.ndarray, labels: np.ndarray, times: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (features.astype("<f8"), labels.astype("i1"), times.astype("<i8")):
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def fit_temporal_archetypes(
    features: np.ndarray | Iterable[Iterable[float]],
    labels: np.ndarray | Iterable[int],
    end_times_ms: np.ndarray | Iterable[int],
    *,
    train_through_ms: int,
    label_available_times_ms: np.ndarray | Iterable[int],
    symbol: str = "BTCUSDT",
    timeframe: str = "4h",
    window_size: int = 5,
    horizon: str = "4h",
    n_clusters: int = 8,
    random_state: int = 42,
    distance_quantile: float = 0.99,
    min_cluster_members: int = 5,
    laplace_alpha: float = 1.0,
) -> TemporalArchetypeModel:
    """Fit one immutable version from rows at or before the declared cutoff only."""
    all_features = _as_feature_matrix(features)
    all_labels = _as_labels(labels, len(all_features))
    all_times = _as_times(end_times_ms, len(all_features))
    label_times = _as_times(label_available_times_ms, len(all_features))
    if np.any(label_times < all_times):
        raise ValueError("label availability cannot precede feature availability")
    # The second predicate purges labels whose future outcome crosses the split.
    mask = (all_times <= int(train_through_ms)) & (label_times <= int(train_through_ms))
    train_x = all_features[mask]
    train_y = all_labels[mask]
    train_times = all_times[mask]
    if len(train_x) < max(n_clusters, 2):
        raise ValueError("insufficient training rows at train_through_ms")
    if not 0.5 <= distance_quantile <= 1.0:
        raise ValueError("distance_quantile must be in [0.5, 1.0]")
    if min_cluster_members < 1 or laplace_alpha <= 0:
        raise ValueError("min_cluster_members and laplace_alpha must be positive")

    scaler = StandardScaler().fit(train_x)
    train_scaled = scaler.transform(train_x)
    clusterer = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300,
    ).fit(train_scaled)

    order = _canonical_cluster_order(clusterer.cluster_centers_)
    old_to_canonical = np.empty(n_clusters, dtype=np.int32)
    old_to_canonical[order] = np.arange(n_clusters, dtype=np.int32)
    assignments = old_to_canonical[clusterer.labels_]
    centroids = clusterer.cluster_centers_[order]

    member_counts: list[int] = []
    thresholds: list[float] = []
    probabilities: list[list[float]] = []
    for cluster_index in range(n_clusters):
        cluster_mask = assignments == cluster_index
        member_counts.append(int(cluster_mask.sum()))
        distances = np.linalg.norm(train_scaled[cluster_mask] - centroids[cluster_index], axis=1)
        thresholds.append(float(np.quantile(distances, distance_quantile)))
        counts = np.array(
            [np.sum(train_y[cluster_mask] == class_value) for class_value in CLASS_VALUES],
            dtype=np.float64,
        )
        smoothed = (counts + laplace_alpha) / (counts.sum() + laplace_alpha * len(CLASS_VALUES))
        probabilities.append(smoothed.tolist())

    train_label_times = label_times[mask]
    data_hash = _training_hash(train_x, train_y, np.column_stack([train_times, train_label_times]))
    identity = {
        "schema_version": SCHEMA_VERSION,
        "scope": [symbol, timeframe, int(window_size), horizon],
        "train_through_ms": int(train_through_ms),
        "training_data_hash": data_hash,
        "n_clusters": int(n_clusters),
        "random_state": int(random_state),
        "distance_quantile": float(distance_quantile),
        "min_cluster_members": int(min_cluster_members),
        "laplace_alpha": float(laplace_alpha),
        "scaler_mean": np.asarray(scaler.mean_).tolist(),
        "scaler_scale": np.asarray(scaler.scale_).tolist(),
        "centroids_scaled": centroids.tolist(),
    }
    version_id = f"arch-{_sha256_json(identity)[:20]}"
    archetype_ids = [f"{version_id}:A-{index:04d}" for index in range(n_clusters)]
    model = TemporalArchetypeModel(
        schema_version=SCHEMA_VERSION,
        version_id=version_id,
        symbol=symbol,
        timeframe=timeframe,
        window_size=int(window_size),
        horizon=horizon,
        train_through_ms=int(train_through_ms),
        feature_dim=int(train_x.shape[1]),
        n_clusters=int(n_clusters),
        random_state=int(random_state),
        training_rows=int(len(train_x)),
        training_data_hash=data_hash,
        training_label_policy="label_available_time_ms <= train_through_ms",
        scaler_mean=np.asarray(scaler.mean_).tolist(),
        scaler_scale=np.asarray(scaler.scale_).tolist(),
        centroids_scaled=centroids.tolist(),
        archetype_ids=archetype_ids,
        train_member_counts=member_counts,
        distance_thresholds=thresholds,
        class_probabilities=probabilities,
        distance_quantile=float(distance_quantile),
        min_cluster_members=int(min_cluster_members),
        laplace_alpha=float(laplace_alpha),
    )
    model._validate()
    return model


def assign_frozen(
    model: TemporalArchetypeModel,
    features: np.ndarray | Iterable[Iterable[float]],
) -> AssignmentBatch:
    """Assign windows with a frozen model. No estimator is fitted in this path."""
    matrix = _as_feature_matrix(features)
    if matrix.shape[1] != model.feature_dim:
        raise ValueError("feature dimension does not match frozen archetype version")
    mean = np.asarray(model.scaler_mean, dtype=np.float64)
    scale = np.asarray(model.scaler_scale, dtype=np.float64)
    centroids = np.asarray(model.centroids_scaled, dtype=np.float64)
    scaled = (matrix - mean) / scale
    all_distances = np.linalg.norm(scaled[:, None, :] - centroids[None, :, :], axis=2)
    indices = np.argmin(all_distances, axis=1).astype(np.int32)
    distances = all_distances[np.arange(len(matrix)), indices]
    thresholds = np.asarray(model.distance_thresholds, dtype=np.float64)[indices]
    member_counts = np.asarray(model.train_member_counts, dtype=np.int64)[indices]
    accepted = (distances <= thresholds) & (member_counts >= model.min_cluster_members)
    probs = np.asarray(model.class_probabilities, dtype=np.float64)[indices]
    ids = tuple(model.archetype_ids[index] if ok else None for index, ok in zip(indices, accepted))
    return AssignmentBatch(indices, ids, distances, accepted, probs)


def save_model_immutable(model: TemporalArchetypeModel, path: str | Path) -> Path:
    """Write once. Repeating identical bytes is idempotent; mutation is rejected."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = _canonical_json(model.to_dict()) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") == content:
            return target
        raise FileExistsError(f"refusing to overwrite immutable archetype model: {target}")
    target.write_text(content, encoding="utf-8")
    return target


def load_model(path: str | Path) -> TemporalArchetypeModel:
    return TemporalArchetypeModel.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _brier_rows(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    targets = np.zeros((len(labels), len(CLASS_VALUES)), dtype=np.float64)
    for class_index, class_value in enumerate(CLASS_VALUES):
        targets[:, class_index] = labels == class_value
    return np.sum((probabilities - targets) ** 2, axis=1)


def _block_bootstrap_mean_interval(
    paired_lift: np.ndarray,
    *,
    block_size: int,
    bootstrap_samples: int,
    random_state: int,
) -> tuple[float, float]:
    if len(paired_lift) == 0:
        return float("nan"), float("nan")
    block_size = max(1, min(int(block_size), len(paired_lift)))
    rng = np.random.default_rng(random_state)
    sample_means = np.empty(bootstrap_samples, dtype=np.float64)
    for iteration in range(bootstrap_samples):
        sampled: list[float] = []
        while len(sampled) < len(paired_lift):
            start = int(rng.integers(0, len(paired_lift)))
            for offset in range(block_size):
                sampled.append(float(paired_lift[(start + offset) % len(paired_lift)]))
                if len(sampled) == len(paired_lift):
                    break
        sample_means[iteration] = np.mean(sampled)
    low, high = np.quantile(sample_means, [0.025, 0.975])
    return float(low), float(high)


def evaluate_temporal_archetypes(
    features: np.ndarray | Iterable[Iterable[float]],
    labels: np.ndarray | Iterable[int],
    end_times_ms: np.ndarray | Iterable[int],
    *,
    train_through_ms: int,
    label_available_times_ms: np.ndarray | Iterable[int],
    embargo_rows: int = 0,
    min_evaluation_samples: int = 30,
    block_size: int = 8,
    bootstrap_samples: int = 1000,
    **fit_kwargs: Any,
) -> tuple[TemporalArchetypeModel, dict[str, Any]]:
    """Evaluate cluster-conditioned probabilities on frozen future rows.

    Both candidate and unconditional training baseline are scored on exactly the
    accepted future timestamps. Positive lift means lower Brier loss than baseline.
    """
    matrix = _as_feature_matrix(features)
    aligned_labels = _as_labels(labels, len(matrix))
    times = _as_times(end_times_ms, len(matrix))
    label_times = _as_times(label_available_times_ms, len(matrix))
    model = fit_temporal_archetypes(
        matrix,
        aligned_labels,
        times,
        train_through_ms=train_through_ms,
        label_available_times_ms=label_times,
        **fit_kwargs,
    )
    future_indices = np.flatnonzero(times > int(train_through_ms))
    if embargo_rows:
        future_indices = future_indices[int(embargo_rows) :]
    if len(future_indices) == 0:
        raise ValueError("no evaluation rows after cutoff and embargo")

    assignments = assign_frozen(model, matrix[future_indices])
    accepted_local = np.flatnonzero(assignments.accepted)
    accepted_global = future_indices[accepted_local]
    evaluation_labels = aligned_labels[accepted_global]
    candidate_probs = assignments.class_probabilities[accepted_local]

    # Use the exact same point-in-time eligibility as the fitted candidate.
    # Including boundary labels that mature after the cutoff would leak future
    # information into the baseline and make the paired comparison invalid.
    train_eligible = (times <= int(train_through_ms)) & (label_times <= int(train_through_ms))
    train_labels = aligned_labels[train_eligible]
    train_counts = np.array([np.sum(train_labels == value) for value in CLASS_VALUES], dtype=np.float64)
    alpha = model.laplace_alpha
    baseline_probs_single = (train_counts + alpha) / (len(train_labels) + alpha * len(CLASS_VALUES))
    baseline_probs = np.repeat(baseline_probs_single[None, :], len(evaluation_labels), axis=0)

    if len(evaluation_labels):
        candidate_loss = _brier_rows(candidate_probs, evaluation_labels)
        baseline_loss = _brier_rows(baseline_probs, evaluation_labels)
        paired_lift = baseline_loss - candidate_loss
        lift = float(np.mean(paired_lift))
        interval = _block_bootstrap_mean_interval(
            paired_lift,
            block_size=block_size,
            bootstrap_samples=bootstrap_samples,
            random_state=model.random_state,
        )
        candidate_predictions = CLASS_VALUES[np.argmax(candidate_probs, axis=1)]
        baseline_predictions = CLASS_VALUES[np.argmax(baseline_probs, axis=1)]
        candidate_accuracy = float(np.mean(candidate_predictions == evaluation_labels))
        baseline_accuracy = float(np.mean(baseline_predictions == evaluation_labels))
        candidate_brier = float(np.mean(candidate_loss))
        baseline_brier = float(np.mean(baseline_loss))
    else:
        lift = candidate_accuracy = baseline_accuracy = candidate_brier = baseline_brier = None
        interval = (None, None)

    if len(evaluation_labels) < min_evaluation_samples:
        conclusion = "insufficient_evidence"
    elif interval[0] is not None and interval[0] > 0:
        conclusion = "positive"
    elif interval[1] is not None and interval[1] < 0:
        conclusion = "negative"
    else:
        conclusion = "inconclusive"

    report = {
        "schemaVersion": SCHEMA_VERSION,
        "method": "training-only-standard-scaler+kmeans; frozen nearest-centroid assignment",
        "versionId": model.version_id,
        "symbol": model.symbol,
        "timeframe": model.timeframe,
        "windowSize": model.window_size,
        "horizon": model.horizon,
        "trainThroughMs": model.train_through_ms,
        "trainingRows": model.training_rows,
        "trainingLabelPolicy": model.training_label_policy,
        "evaluationRows": int(len(future_indices)),
        "acceptedRows": int(len(evaluation_labels)),
        "coverage": float(len(evaluation_labels) / len(future_indices)),
        "abstentionRate": float(1.0 - len(evaluation_labels) / len(future_indices)),
        "baseline": "Laplace-smoothed unconditional class probabilities from training rows",
        "candidateBrier": candidate_brier,
        "baselineBrier": baseline_brier,
        "brierLift": lift,
        "brierLift95Ci": [interval[0], interval[1]],
        "candidateAccuracy": candidate_accuracy,
        "baselineAccuracy": baseline_accuracy,
        "blockBootstrap": {
            "blockSizeRows": int(block_size),
            "samples": int(bootstrap_samples),
            "seed": model.random_state,
        },
        "conclusion": conclusion,
        "acceptedEndTimesMs": times[accepted_global].tolist(),
        "negativeResultIsValid": True,
    }
    report["reportHash"] = _sha256_json(report)
    return model, report


def _write_report_immutable(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            return
        raise FileExistsError(f"refusing to overwrite immutable evaluation report: {path}")
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe BTC archetype temporal evaluation")
    parser.add_argument("--input-npz", help="Optional NPZ; otherwise load the read-only BTCUSDT 4h benchmark from PostgreSQL")
    parser.add_argument("--train-through-ms", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--symbol", default="BTCUSDT", choices=["BTCUSDT"])
    parser.add_argument("--timeframe", default="4h", choices=["1h", "4h", "1d"])
    parser.add_argument("--window-size", default=5, type=int)
    parser.add_argument("--horizon", default="4h")
    parser.add_argument("--clusters", default=8, type=int)
    parser.add_argument("--embargo-rows", default=0, type=int)
    args = parser.parse_args()

    if args.input_npz:
        with np.load(args.input_npz, allow_pickle=False) as dataset:
            if "label_available_ms" not in dataset:
                raise ValueError("NPZ must contain label_available_ms to prove split-boundary purge")
            features = dataset["X"]
            labels = dataset["y"]
            end_ms = dataset["end_ms"]
            label_available_ms = dataset["label_available_ms"]
    else:
        if args.timeframe != "4h" or args.horizon != "4h":
            raise ValueError("The direct PostgreSQL benchmark is frozen to BTCUSDT 4h / next 4h outcome")
        from ml_evidence_walkforward import load_btc_4h_benchmark

        # Load all outcomes mature by the current dataset cutoff. The archetype
        # fit itself still uses only labels mature by train_through_ms.
        replay = load_btc_4h_benchmark(
            decision_cutoff_ms=2**63 - 1,
            window_size=args.window_size,
        )
        features = replay.features
        labels = replay.labels
        end_ms = replay.decision_times_ms
        label_available_ms = replay.label_available_times_ms

    model, report = evaluate_temporal_archetypes(
        features,
        labels,
        end_ms,
        train_through_ms=args.train_through_ms,
        label_available_times_ms=label_available_ms,
        symbol=args.symbol,
        timeframe=args.timeframe,
        window_size=args.window_size,
        horizon=args.horizon,
        n_clusters=args.clusters,
        embargo_rows=args.embargo_rows,
    )
    output_dir = Path(args.output_dir)
    save_model_immutable(model, output_dir / f"{model.version_id}.json")
    _write_report_immutable(report, output_dir / f"{model.version_id}.evaluation.json")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
