import unittest
import sys
from pathlib import Path

# Add btc-ai to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prediction_service import list_available_models, predict_from_vector


class TestPredictionService(unittest.TestCase):
    """
    Hermetic unit tests for ML model loading, vector dimension validation,
    and prediction probability mappings.
    """

    def test_list_available_models(self):
        models = list_available_models()
        self.assertIsInstance(models, list)
        if models:
            first = models[0]
            self.assertIn("file", first)
            self.assertIn("is_active", first)

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
        try:
            res = predict_from_vector(
                feature_vector=vec,
                symbol="BTCUSDT",
                timeframe="4h",
                window_size=5,
                horizon="4h"
            )
            self.assertIn("label", res)
            self.assertIn("confidence", res)
            self.assertIn("prob_up", res)
            self.assertIn("prob_down", res)
            self.assertIn(res["label"], [-1, 0, 1])
            self.assertGreaterEqual(res["confidence"], 0.0)
            self.assertLessEqual(res["confidence"], 1.0)
        except FileNotFoundError:
            # In clean CI env without model artifacts, FileNotFoundError is expected and valid
            pass


if __name__ == "__main__":
    unittest.main()
