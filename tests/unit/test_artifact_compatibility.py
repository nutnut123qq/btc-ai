import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
from prediction_service import list_available_models, load_model


ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"

ARTIFACT_RUNTIME_PINS = {
    "joblib": "1.5.3",
    "scikit-learn": "1.9.0",
    "xgboost": "3.3.0",
}


def _active_entries():
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return [
        (key, entry)
        for key, entry in registry.get("models", {}).items()
        if entry.get("status") == "active"
    ]


def test_active_registry_entries_match_sidecar_manifests():
    entries = _active_entries()
    assert entries, "model registry must contain at least one active artifact"

    for key, entry in entries:
        filename = entry["active_model_file"]
        assert Path(filename).name == filename, f"unsafe artifact path in registry: {filename}"

        artifact_path = MODELS_DIR / filename
        manifest_path = artifact_path.with_suffix(".json")
        assert artifact_path.is_file(), f"missing active artifact: {filename}"
        assert manifest_path.is_file(), f"missing sidecar manifest: {manifest_path.name}"

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_key = (
            f"{entry['symbol']}_{entry['timeframe']}_"
            f"ws{entry['window_size']}_h{entry['horizon']}"
        )
        assert key == expected_key
        for field in ("symbol", "timeframe", "window_size", "horizon", "version"):
            assert manifest[field] == entry[field], f"{filename}: mismatched {field}"
        assert manifest.get("feature_dim", 0) > 0, f"{filename}: invalid feature_dim"


def test_active_sidecars_have_complete_lineage_and_are_servable():
    required = {
        "artifact_sha256",
        "feature_schema_version",
        "feature_schema_hash",
        "library_versions",
        "class_mapping",
        "feature_names",
    }
    for _, entry in _active_entries():
        artifact = MODELS_DIR / entry["active_model_file"]
        manifest = json.loads(artifact.with_suffix(".json").read_text(encoding="utf-8"))
        assert not (required - set(manifest)), f"incomplete lineage for {artifact.name}"
        model, loaded_manifest = load_model(
            entry["symbol"], entry["timeframe"], entry["window_size"], entry["horizon"]
        )
        assert model is not None
        assert loaded_manifest["artifact_sha256"] == manifest["artifact_sha256"]
    assert len(list_available_models()) == len(_active_entries())


def test_active_artifacts_load_with_pinned_runtime():
    for package, expected_version in ARTIFACT_RUNTIME_PINS.items():
        assert importlib.metadata.version(package) == expected_version

    for _, entry in _active_entries():
        artifact_path = MODELS_DIR / entry["active_model_file"]
        manifest = json.loads(artifact_path.with_suffix(".json").read_text(encoding="utf-8"))
        model = joblib.load(artifact_path)

        assert hasattr(model, "predict")
        assert getattr(model, "n_features_in_", None) == manifest["feature_dim"]
        probabilities = model.predict_proba(np.zeros((1, manifest["feature_dim"]), dtype=np.float32))
        assert probabilities.shape == (1, 3)
        assert np.asarray(model.classes_).tolist() == [0, 1, 2]
        assert np.isfinite(probabilities).all()
        assert np.isclose(probabilities.sum(axis=1)[0], 1.0, atol=1e-6)
