"""Fail-closed model registry and prediction utilities."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

MODELS_DIR = Path(__file__).with_name("models")
REGISTRY_PATH = MODELS_DIR / "model_registry.json"

_MODEL_CACHE: dict[str, tuple[Any, dict, float]] = {}
_CACHE_TTL_SECONDS = 3600.0
_REQUIRED_MANIFEST_FIELDS = {
    "artifact_sha256",
    "feature_schema_version",
    "feature_schema_hash",
    "library_versions",
    "class_mapping",
    "feature_names",
    "data_provenance",
    "promotion_gate",
}
_REQUIRED_LIBRARY_VERSIONS = {"joblib", "scikit-learn", "xgboost"}


class ModelArtifactIncompatibleError(RuntimeError):
    """The selected registry artifact lacks evidence required for safe inference."""


def _registry_models() -> dict[str, dict]:
    try:
        payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelArtifactIncompatibleError("Model registry is unavailable.") from exc
    models = payload.get("models")
    if not isinstance(models, dict):
        raise ModelArtifactIncompatibleError("Model registry is invalid.")
    return models


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_schema_hash(feature_names: list[str]) -> str:
    encoded = json.dumps(feature_names, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_active_artifact(
    symbol: str,
    timeframe: str,
    window_size: int,
    horizon: str,
    model_name: str | None = None,
) -> tuple[Path, dict]:
    key = f"{symbol}_{timeframe}_ws{window_size}_h{horizon}"
    entry = _registry_models().get(key)
    if not isinstance(entry, dict) or entry.get("status") != "active":
        raise ModelArtifactIncompatibleError(f"No compatible active artifact is registered for {key}.")

    filename = entry.get("active_model_file")
    if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".joblib"):
        raise ModelArtifactIncompatibleError(f"Registry artifact path is invalid for {key}.")

    artifact = MODELS_DIR / filename
    manifest_path = artifact.with_suffix(".json")
    if not artifact.is_file() or not manifest_path.is_file():
        raise ModelArtifactIncompatibleError(f"Artifact or manifest is missing for {key}.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelArtifactIncompatibleError(f"Manifest is invalid for {key}.") from exc

    missing = sorted(field for field in _REQUIRED_MANIFEST_FIELDS if not manifest.get(field))
    if missing:
        raise ModelArtifactIncompatibleError(
            f"Artifact {filename} is quarantined: manifest lacks {', '.join(missing)}."
        )

    if str(manifest["artifact_sha256"]).lower() != _sha256(artifact):
        raise ModelArtifactIncompatibleError(f"Artifact checksum mismatch for {filename}.")

    versions = manifest["library_versions"]
    if not isinstance(versions, dict) or not _REQUIRED_LIBRARY_VERSIONS.issubset(versions):
        raise ModelArtifactIncompatibleError(f"Library version evidence is invalid for {filename}.")
    for package, expected in versions.items():
        try:
            actual = importlib.metadata.version(str(package))
        except importlib.metadata.PackageNotFoundError as exc:
            raise ModelArtifactIncompatibleError(f"Required runtime package is missing for {filename}.") from exc
        if actual != str(expected):
            raise ModelArtifactIncompatibleError(f"Runtime version mismatch for {filename}.")

    class_mapping = manifest["class_mapping"]
    if class_mapping != {"0": -1, "1": 0, "2": 1}:
        raise ModelArtifactIncompatibleError(f"Class mapping is unsupported for {filename}.")

    feature_names = manifest["feature_names"]
    if (
        not isinstance(feature_names, list)
        or not feature_names
        or not all(isinstance(name, str) and name for name in feature_names)
        or len(feature_names) != manifest.get("feature_dim")
        or _feature_schema_hash(feature_names) != str(manifest["feature_schema_hash"]).lower()
    ):
        raise ModelArtifactIncompatibleError(f"Feature schema evidence is invalid for {filename}.")

    expected = {
        "symbol": symbol,
        "timeframe": timeframe,
        "window_size": window_size,
        "horizon": horizon,
    }
    if any(manifest.get(field) != value for field, value in expected.items()):
        raise ModelArtifactIncompatibleError(f"Manifest identity mismatch for {filename}.")
    if manifest.get("version") != entry.get("version"):
        raise ModelArtifactIncompatibleError(f"Manifest version mismatch for {filename}.")

    provenance = manifest["data_provenance"]
    if (
        not isinstance(provenance, dict)
        or provenance.get("identity") != key
        or not isinstance(provenance.get("row_count"), int)
        or provenance["row_count"] <= 0
        or not isinstance(provenance.get("dataset_sha256"), str)
        or len(provenance["dataset_sha256"]) != 64
        or not isinstance(provenance.get("label_lineage"), dict)
        or provenance["label_lineage"].get("complete") is not True
        or not isinstance(provenance["label_lineage"].get("source_column"), str)
    ):
        raise ModelArtifactIncompatibleError(f"Dataset provenance is invalid for {filename}.")
    promotion_gate = manifest["promotion_gate"]
    if not isinstance(promotion_gate, dict) or promotion_gate.get("passed") is not True:
        raise ModelArtifactIncompatibleError(f"Artifact {filename} did not pass the promotion gate.")

    if model_name and model_name not in {filename, artifact.stem, manifest.get("model_name")}:
        raise ModelArtifactIncompatibleError("Requested model is not the registered active artifact.")
    return artifact, manifest


def load_model(symbol: str, timeframe: str, window_size: int, horizon: str, model_name: str | None = None):
    """Load only a manifest-verified active artifact with a short in-memory cache."""
    artifact, manifest = _resolve_active_artifact(symbol, timeframe, window_size, horizon, model_name)
    cache_key = f"{artifact.name}:{manifest['artifact_sha256']}"
    now = time.time()
    cached = _MODEL_CACHE.get(cache_key)
    if cached and now - cached[2] < _CACHE_TTL_SECONDS:
        return cached[0], cached[1]

    try:
        model = joblib.load(artifact)
    except Exception as exc:
        raise ModelArtifactIncompatibleError(f"Artifact cannot be loaded: {artifact.name}.") from exc
    expected_dim = manifest.get("feature_dim")
    if not isinstance(expected_dim, int) or expected_dim <= 0:
        raise ModelArtifactIncompatibleError(f"Feature dimension is invalid for {artifact.name}.")
    if getattr(model, "n_features_in_", None) != expected_dim:
        raise ModelArtifactIncompatibleError(f"Feature dimension mismatch for {artifact.name}.")
    if not hasattr(model, "predict_proba"):
        raise ModelArtifactIncompatibleError(f"Artifact has no probability interface: {artifact.name}.")
    classes = np.asarray(getattr(model, "classes_", []))
    if classes.tolist() != [0, 1, 2]:
        raise ModelArtifactIncompatibleError(f"Model class order is unsupported for {artifact.name}.")

    _MODEL_CACHE.clear()
    _MODEL_CACHE[cache_key] = (model, manifest, now)
    return model, manifest


def predict_from_vector(
    feature_vector: list[float],
    symbol: str,
    timeframe: str,
    window_size: int,
    horizon: str,
    model_name: str | None = None,
):
    model, manifest = load_model(symbol, timeframe, window_size, horizon, model_name)
    expected_dim = manifest["feature_dim"]
    if len(feature_vector) != expected_dim:
        raise ValueError(f"Feature vector length {len(feature_vector)} != expected {expected_dim}")

    features = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
    started = time.time()
    try:
        probability_matrix = np.asarray(model.predict_proba(features), dtype=np.float64)
    except Exception as exc:
        raise ModelArtifactIncompatibleError("Model probability inference failed.") from exc
    if (
        probability_matrix.shape != (1, 3)
        or not np.isfinite(probability_matrix).all()
        or not np.isclose(probability_matrix.sum(axis=1)[0], 1.0, atol=1e-6)
    ):
        raise ModelArtifactIncompatibleError("Model probability output is invalid.")
    probabilities = probability_matrix[0]
    predicted_index = int(np.argmax(probabilities))
    mapping = manifest["class_mapping"]
    return {
        "label": int(mapping[str(predicted_index)]),
        "confidence": float(probabilities[predicted_index]),
        "prob_down": float(probabilities[0]),
        "prob_sideways": float(probabilities[1]),
        "prob_up": float(probabilities[2]),
        "model_version": manifest.get("version") or manifest.get("model_name"),
        "inference_ms": (time.time() - started) * 1000.0,
    }


def list_available_models() -> list[dict]:
    """Return only artifacts that are safe to serve; quarantined files stay hidden."""
    available: list[dict] = []
    for key, entry in _registry_models().items():
        if not isinstance(entry, dict) or entry.get("status") != "active":
            continue
        try:
            artifact, manifest = _resolve_active_artifact(
                str(entry.get("symbol")),
                str(entry.get("timeframe")),
                int(entry.get("window_size")),
                str(entry.get("horizon")),
            )
            model, _ = load_model(
                str(entry.get("symbol")),
                str(entry.get("timeframe")),
                int(entry.get("window_size")),
                str(entry.get("horizon")),
            )
            try:
                probabilities = np.asarray(
                    model.predict_proba(np.zeros((1, manifest["feature_dim"]), dtype=np.float32))
                )
            except Exception as exc:
                raise ModelArtifactIncompatibleError(f"Probability smoke failed for {artifact.name}.") from exc
            if (
                probabilities.shape != (1, 3)
                or not np.isfinite(probabilities).all()
                or not np.isclose(probabilities.sum(axis=1)[0], 1.0, atol=1e-6)
            ):
                raise ModelArtifactIncompatibleError(f"Probability smoke failed for {artifact.name}.")
        except (ModelArtifactIncompatibleError, TypeError, ValueError):
            continue
        available.append({
            "key": key,
            "file": artifact.name,
            "symbol": manifest["symbol"],
            "timeframe": manifest["timeframe"],
            "window_size": manifest["window_size"],
            "horizon": manifest["horizon"],
            "model_name": manifest.get("model_name"),
            "is_active": True,
            "metrics": manifest.get("oos_metrics", {}),
        })
    return available
