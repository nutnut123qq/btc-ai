import unittest
import sys
import os
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient

# Add btc-ai to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from main import app
from prediction_service import ModelArtifactIncompatibleError


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
        # A quarantined model fails closed before model-specific shape validation.
        payload = {
            "symbol": "BTCUSDT",
            "timeframe": "4h",
            "window_size": 5,
            "horizon": "4h",
            "feature_vector": [1.0, 2.0]
        }
        with patch(
            "main.predict_from_vector",
            side_effect=ModelArtifactIncompatibleError("quarantined"),
        ):
            response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["code"], "MODEL_ARTIFACT_INCOMPATIBLE")

    def test_capabilities_reports_llm_disabled_but_ml_available(self):
        with (
            patch.dict(os.environ, {"LLM_PROVIDER": "none"}),
            patch("main.list_available_models", return_value=[{"file": "model.joblib"}]),
        ):
            response = self.client.get("/api/capabilities")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "mlInference": True,
            "llmExplanation": False,
            "provider": "none",
            "reason": "LLM provider is disabled.",
            "mlReason": None,
        })

    def test_capabilities_reports_ml_unavailable_when_registry_cannot_load(self):
        with patch("main.list_available_models", side_effect=ValueError("secret registry error")):
            response = self.client.get("/api/capabilities")

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["mlInference"])
        self.assertEqual(response.json()["mlReason"], "Model registry is unavailable.")
        self.assertNotIn("secret registry error", response.text)

    def test_predict_does_not_require_llm(self):
        expected = {"label": 1, "probability": 0.7}
        with (
            patch.dict(os.environ, {"LLM_PROVIDER": "none"}),
            patch("main.predict_from_vector", return_value=expected),
        ):
            response = self.client.post("/api/predict", json={
                "symbol": "BTCUSDT",
                "timeframe": "4h",
                "window_size": 5,
                "horizon": "4h",
                "feature_vector": [1.0],
            })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected)

    def test_llm_endpoints_return_structured_503_when_disabled(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "none"}):
            responses = [
                self.client.post("/api/analyze", json={"symbol": "BTC"}),
                self.client.post("/api/explain", json={"prompt": "Explain"}),
                self.client.post("/api/explain/stream", json={"prompt": "Explain"}),
            ]

        for response in responses:
            self.assertEqual(response.status_code, 503)
            body = response.json()
            self.assertEqual(body["code"], "LLM_NOT_CONFIGURED")
            self.assertFalse(body["retryable"])
            self.assertTrue(body["requestId"])
            self.assertEqual(set(body), {"code", "message", "retryable", "requestId"})

    def test_missing_provider_key_does_not_expose_configuration_details(self):
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "blackbox", "BLACKBOX_API_KEY": ""},
        ):
            response = self.client.post("/api/explain", json={"prompt": "Explain"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["code"], "LLM_NOT_CONFIGURED")
        self.assertNotIn("BLACKBOX_API_KEY", response.text)

    def test_unknown_provider_returns_sanitized_unavailable_error(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "not-a-provider"}):
            response = self.client.post("/api/explain", json={"prompt": "Explain"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["code"], "LLM_PROVIDER_UNAVAILABLE")
        self.assertFalse(response.json()["retryable"])
        self.assertNotIn("not-a-provider", response.text)

    def test_provider_runtime_error_is_sanitized(self):
        class BrokenLlm:
            async def ainvoke(self, _messages):
                raise RuntimeError("secret provider response")

        with patch("main._build_llm", return_value=BrokenLlm()):
            response = self.client.post("/api/explain", json={"prompt": "Explain"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["code"], "LLM_PROVIDER_UNAVAILABLE")
        self.assertTrue(response.json()["retryable"])
        self.assertNotIn("secret provider response", response.text)

    def test_analyze_graph_runtime_error_is_sanitized(self):
        class BrokenLlm:
            async def ainvoke(self, _messages):
                raise RuntimeError("secret graph provider response")

        with (
            patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}),
            patch("main._build_llm", return_value=BrokenLlm()),
        ):
            response = self.client.post("/api/analyze", json={
                "symbol": "BTC",
                "news_context": "known news",
                "tech_context": "known technical context",
            })

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["code"], "LLM_PROVIDER_UNAVAILABLE")
        self.assertNotIn("secret graph provider response", response.text)

    def test_stream_runtime_error_is_sanitized(self):
        class BrokenLlm:
            async def astream(self, _messages):
                if False:
                    yield None
                raise RuntimeError("secret streaming response")

        with patch("main._build_llm", return_value=BrokenLlm()):
            response = self.client.post("/api/explain/stream", json={"prompt": "Explain"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["code"], "LLM_PROVIDER_UNAVAILABLE")
        self.assertNotIn("secret streaming response", response.text)

    def test_empty_stream_returns_sanitized_503(self):
        class EmptyLlm:
            async def astream(self, _messages):
                if False:
                    yield None

        with patch("main._build_llm", return_value=EmptyLlm()):
            response = self.client.post("/api/explain/stream", json={"prompt": "Explain"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["code"], "LLM_PROVIDER_UNAVAILABLE")

    def test_stream_error_after_first_token_is_sanitized(self):
        class PartiallyBrokenLlm:
            async def astream(self, _messages):
                yield "safe token"
                raise RuntimeError("secret mid-stream response")

        with patch("main._build_llm", return_value=PartiallyBrokenLlm()):
            response = self.client.post(
                "/api/explain/stream",
                json={"prompt": "Explain"},
                headers={"x-request-id": "test-request-id"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("safe token", response.text)
        self.assertIn("LLM_PROVIDER_UNAVAILABLE", response.text)
        self.assertIn("test-request-id", response.text)
        self.assertNotIn("secret mid-stream response", response.text)


if __name__ == "__main__":
    unittest.main()
