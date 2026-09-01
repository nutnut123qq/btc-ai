import unittest
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

# Add btc-ai to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prediction_service import ModelArtifactIncompatibleError, list_available_models, load_model, predict_from_vector


class TestPredictionService(unittest.TestCase):
    """
    Hermetic unit tests for ML model loading, vector dimension validation,
    and prediction probability mappings.
    """

    def test_list_available_models(self):
        models = list_available_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["symbol"], "BTCUSDT")
        self.assertTrue(models[0]["is_active"])

    def test_predict_from_vector_dimension_mismatch(self):
        # Passing an invalid feature vector of length 5 (expected 175 for ws=5)
        with self.assertRaises(ValueError):
            predict_from_vector(
                feature_vector=[1.0, 2.0, 3.0],
                symbol="BTCUSDT",
                timeframe="4h",
                window_size=5,
                horizon="4h"
            )

    def test_predict_from_vector_valid_shape(self):
        # 5 bars * 35 features = 175 features
        vec = [0.0] * 175
        result = predict_from_vector(
            feature_vector=vec,
            symbol="BTCUSDT",
            timeframe="4h",
            window_size=5,
            horizon="4h"
        )
        self.assertIn(result["label"], (-1, 0, 1))
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertAlmostEqual(
            result["prob_down"] + result["prob_sideways"] + result["prob_up"], 1.0
        )

    def test_arbitrary_legacy_filename_cannot_bypass_registry(self):
        with tempfile.TemporaryDirectory() as temp:
            models_dir = Path(temp)
            legacy = models_dir / "BTCUSDT_4h_ws5_h4h_XGB_calibrated.joblib"
            legacy.write_bytes(b"legacy file must never be opened")
            registry = models_dir / "model_registry.json"
            registry.write_text(json.dumps({"models": {}}), encoding="utf-8")
            with (
                patch("prediction_service.MODELS_DIR", models_dir),
                patch("prediction_service.REGISTRY_PATH", registry),
                self.assertRaises(ModelArtifactIncompatibleError),
            ):
                load_model("BTCUSDT", "4h", 5, "4h", legacy.name)

    def test_live_probability_exception_is_sanitized_as_artifact_error(self):
        class BrokenModel:
            def predict_proba(self, _features):
                raise RuntimeError("provider internals must not escape")

        manifest = {"feature_dim": 3, "class_mapping": {"0": -1, "1": 0, "2": 1}, "version": "test"}
        with (
            patch("prediction_service.load_model", return_value=(BrokenModel(), manifest)),
            self.assertRaises(ModelArtifactIncompatibleError),
        ):
            predict_from_vector([0.0, 0.0, 0.0], "BTCUSDT", "4h", 5, "4h")

    def test_live_probability_output_must_be_finite_and_normalized(self):
        class FixedModel:
            def __init__(self, probabilities):
                self.probabilities = probabilities

            def predict_proba(self, _features):
                return self.probabilities

        manifest = {"feature_dim": 3, "class_mapping": {"0": -1, "1": 0, "2": 1}, "version": "test"}
        invalid_outputs = (
            np.asarray([[np.nan, 0.5, 0.5]]),
            np.asarray([[np.inf, 0.0, 0.0]]),
            np.asarray([[0.2, 0.2, 0.2]]),
            np.asarray([[0.5, 0.5]]),
        )
        for probabilities in invalid_outputs:
            with self.subTest(probabilities=probabilities), patch(
                "prediction_service.load_model", return_value=(FixedModel(probabilities), manifest)
            ), self.assertRaises(ModelArtifactIncompatibleError):
                predict_from_vector([0.0, 0.0, 0.0], "BTCUSDT", "4h", 5, "4h")


if __name__ == "__main__":
    unittest.main()
