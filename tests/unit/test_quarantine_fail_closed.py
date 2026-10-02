"""Fixture tests for the quarantine fail-closed guarantees surveyed in
docs/research/ml-evaluation/quarantine-provenance.md.

These tests pin the two independent blocks that keep the quarantined
BTCUSDT_4h_ws5_h4h artifact out of production:

1. registry `status != "active"` is rejected before any manifest work, and
2. the quarantined manifest *shape* (no ``data_provenance`` /
   ``promotion_gate`` fields) is rejected even if the status were flipped
   back to ``active``.

All fixtures are in-memory/tempdir; the real 9.8MB joblib is never loaded.
"""

import hashlib
import importlib.metadata
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prediction_service import (
    ModelArtifactIncompatibleError,
    list_available_models,
    load_model,
)
from rolling_retrainer import assess_promotion_gate

MODEL_KEY = "BTCUSDT_4h_ws5_h4h"
ARTIFACT_NAME = "BTCUSDT_4h_ws5_h4h_XGB_v20260901025507.joblib"


def _feature_names(dim: int = 1) -> list[str]:
    return ["ws5_bar0_Feature"] * dim


def _feature_schema_hash(names: list[str]) -> str:
    payload = json.dumps(names, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _manifest(artifact_bytes: bytes, *, lineage_complete: bool = True,
              gate: dict | None = None, include_gate: bool = True,
              include_provenance: bool = True) -> dict:
    """Minimal manifest in the shape the hardened loader inspects."""
    manifest = {
        "symbol": "BTCUSDT",
        "timeframe": "4h",
        "window_size": 5,
        "horizon": "4h",
        "version": "v20260901025507",
        "model_name": "XGB_v20260901025507",
        "feature_dim": 1,
        "feature_names": _feature_names(),
        "feature_schema_version": "window-dataset-35-v1",
        "feature_schema_hash": _feature_schema_hash(_feature_names()),
        "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "library_versions": {"joblib": importlib.metadata.version("joblib")},
        "class_mapping": {"0": -1, "1": 0, "2": 1},
    }
    if include_provenance:
        manifest["data_provenance"] = {
            "identity": MODEL_KEY,
            "row_count": 14380,
            "dataset_sha256": "0" * 64,
            "label_lineage": {
                "complete": lineage_complete,
                "source_column": "PriceTargets.TargetDirectionTb4h",
            },
        }
    if include_gate:
        manifest["promotion_gate"] = gate if gate is not None else {"passed": True}
    return manifest


def _registry(status: str, filename: str = ARTIFACT_NAME) -> dict:
    return {
        "models": {
            MODEL_KEY: {
                "symbol": "BTCUSDT",
                "timeframe": "4h",
                "window_size": 5,
                "horizon": "4h",
                "status": status,
                "active_model_file": filename,
                "version": "v20260901025507",
            }
        }
    }


def _fixture_dir(models_dir: Path, manifest: dict | None = None,
                 registry: dict | None = None) -> Path:
    """Populate a temp models dir with a stub artifact + optional files."""
    artifact = models_dir / ARTIFACT_NAME
    artifact_bytes = b"stub-artifact-not-a-real-joblib"
    artifact.write_bytes(artifact_bytes)
    if manifest is not None:
        artifact.with_suffix(".json").write_text(json.dumps(manifest), encoding="utf-8")
    if registry is not None:
        (models_dir / "model_registry.json").write_text(
            json.dumps(registry), encoding="utf-8"
        )
    return artifact


def test_quarantined_status_fails_closed_before_manifest_inspection():
    """status='quarantined' must reject even when every other field is perfect."""
    with tempfile.TemporaryDirectory() as temp:
        models_dir = Path(temp)
        manifest = _manifest(b"unused-by-this-test")  # complete, gate passed
        _fixture_dir(models_dir, manifest=manifest, registry=_registry("quarantined"))
        with (
            patch("prediction_service.MODELS_DIR", models_dir),
            patch("prediction_service.REGISTRY_PATH", models_dir / "model_registry.json"),
            patch("prediction_service._REQUIRED_LIBRARY_VERSIONS", {"joblib"}),
            patch("prediction_service._sha256", return_value="0" * 64) as sha_probe,
        ):
            try:
                load_model("BTCUSDT", "4h", 5, "4h")
            except ModelArtifactIncompatibleError as exc:
                assert "No compatible active artifact" in str(exc)
            else:
                raise AssertionError("quarantined status must fail closed")
            sha_probe.assert_not_called()  # rejected before artifact bytes are touched


def test_manifest_shape_of_quarantined_artifact_is_rejected():
    """The real artifact's manifest lacks data_provenance/promotion_gate.
    Even with status='active' that shape must never serve."""
    with tempfile.TemporaryDirectory() as temp:
        models_dir = Path(temp)
        artifact = _fixture_dir(models_dir, registry=_registry("active"))
        # Mirror the deployed manifest: every old field present, the two
        # lineage fields introduced at 06f2df7 absent.
        manifest = _manifest(artifact.read_bytes(), include_provenance=False,
                           include_gate=False)
        artifact.with_suffix(".json").write_text(json.dumps(manifest), encoding="utf-8")
        with (
            patch("prediction_service.MODELS_DIR", models_dir),
            patch("prediction_service.REGISTRY_PATH", models_dir / "model_registry.json"),
        ):
            try:
                load_model("BTCUSDT", "4h", 5, "4h")
            except ModelArtifactIncompatibleError as exc:
                message = str(exc)
            else:
                raise AssertionError("manifest missing lineage fields must fail closed")
        assert "quarantined" in message
        assert "data_provenance" in message
        assert "promotion_gate" in message


def test_quarantined_entry_is_hidden_from_available_models():
    """list_available_models must not surface quarantined artifacts."""
    with tempfile.TemporaryDirectory() as temp:
        models_dir = Path(temp)
        _fixture_dir(
            models_dir,
            manifest=_manifest(b"unused"),
            registry=_registry("quarantined"),
        )
        with (
            patch("prediction_service.MODELS_DIR", models_dir),
            patch("prediction_service.REGISTRY_PATH", models_dir / "model_registry.json"),
        ):
            assert list_available_models() == []


def test_ambiguous_label_lineage_is_rejected():
    """label_lineage.complete=False is the guard the Sep-21 dataset rebuild
    made load-bearing: two plausible label columns must never serve."""
    with tempfile.TemporaryDirectory() as temp:
        models_dir = Path(temp)
        artifact = _fixture_dir(models_dir, registry=_registry("active"))
        manifest = _manifest(artifact.read_bytes(), lineage_complete=False)
        artifact.with_suffix(".json").write_text(json.dumps(manifest), encoding="utf-8")
        with (
            patch("prediction_service.MODELS_DIR", models_dir),
            patch("prediction_service.REGISTRY_PATH", models_dir / "model_registry.json"),
            patch("prediction_service._REQUIRED_LIBRARY_VERSIONS", {"joblib"}),
        ):
            try:
                load_model("BTCUSDT", "4h", 5, "4h")
            except ModelArtifactIncompatibleError as exc:
                assert "Dataset provenance is invalid" in str(exc)
            else:
                raise AssertionError("ambiguous label lineage must fail closed")


def test_truthy_nonboolean_gate_verdict_is_rejected():
    """promotion_gate.passed must be literal True; 'yes'/'1' must not pass."""
    with tempfile.TemporaryDirectory() as temp:
        models_dir = Path(temp)
        artifact = _fixture_dir(models_dir, registry=_registry("active"))
        for bad_gate in ({"passed": "yes"}, {"passed": 1}, {"passed": None}, {}):
            manifest = _manifest(artifact.read_bytes(), gate=bad_gate)
            artifact.with_suffix(".json").write_text(json.dumps(manifest), encoding="utf-8")
            with (
                patch("prediction_service.MODELS_DIR", models_dir),
                patch("prediction_service.REGISTRY_PATH", models_dir / "model_registry.json"),
                patch("prediction_service._REQUIRED_LIBRARY_VERSIONS", {"joblib"}),
            ):
                try:
                    load_model("BTCUSDT", "4h", 5, "4h")
                except ModelArtifactIncompatibleError as exc:
                    # Either the missing-field check ("manifest lacks
                    # promotion_gate") or the verdict check ("did not pass
                    # the promotion gate") must fire.
                    assert "promotion" in str(exc)
                else:
                    raise AssertionError(f"gate {bad_gate} must fail closed")


def test_recorded_oos_collapse_reproduces_gate_failures():
    """Pin the measured post-quarantine replay: the frozen artifact scored on
    the stored test window (2026-07-25→2026-08-24, n=181) under today's
    triple-barrier labels fails the discrimination checks of
    two-independent-windows-v1, while still beating the majority baseline on
    both losses — matching the 'class-discrimination' half of the recorded
    quarantine reason."""
    validation_metrics = {  # frozen artifact on registry val window (547 rows)
        "samples": 547,
        "class_counts": [166, 212, 169],
        "f1_per_class": {"down": 0.3585, "sideways": 0.6254, "up": 0.3431},
        "f1_macro": 0.4423,
        "balanced_accuracy": 0.4787,
        "mcc": 0.3487,
        "ece": 0.1583,
        "brier_score": 0.6156,
        "log_loss": 1.0224,
    }
    oos_metrics = {  # frozen artifact on registry test window (181 rows)
        "samples": 181,
        "class_counts": [28, 101, 52],
        "f1_per_class": {"down": 0.0556, "sideways": 0.7299, "up": 0.0},
        "f1_macro": 0.2618,
        "balanced_accuracy": 0.3419,
        "mcc": 0.0804,
        "ece": 0.1521,
        "brier_score": 0.5932,
        "log_loss": 1.0168,
    }
    # majority_baseline over the same windows' train priors [0.288, 0.308, 0.404]
    validation_baseline = {"brier_score": 0.6777, "log_loss": 1.1141}
    oos_baseline = {"brier_score": 0.6760, "log_loss": 1.1101}

    verdict = assess_promotion_gate(
        validation_metrics, oos_metrics, validation_baseline, oos_baseline
    )

    assert verdict["passed"] is False
    assert verdict["policy"] == "two-independent-windows-v1"
    assert set(verdict["failures"]) == {
        "oos: macro F1",
        "oos: balanced accuracy",
        "oos: MCC",
        "oos: per-class F1",
    }


if __name__ == "__main__":
    import unittest

    unittest.main()
