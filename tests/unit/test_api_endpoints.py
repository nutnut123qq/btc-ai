import unittest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add btc-ai to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from main import app


class TestApiEndpoints(unittest.TestCase):
    """
    Hermetic API endpoint testing for FastAPI service.
    """

    def setUp(self):
        self.client = TestClient(app)

    def test_list_models_endpoint(self):
        response = self.client.get("/api/predict/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("models", data)
        self.assertIsInstance(data["models"], list)

    def test_predict_endpoint_validation_error(self):
        # Empty feature vector should fail validation or return 422/500
        payload = {
            "symbol": "BTCUSDT",
            "timeframe": "4h",
            "window_size": 5,
            "horizon": "4h",
            "feature_vector": [1.0, 2.0]
        }
        response = self.client.post("/api/predict", json=payload)
        # Should return 500 (ValueError from dimension mismatch) or 422
        self.assertIn(response.status_code, [400, 422, 500])


if __name__ == "__main__":
    unittest.main()
