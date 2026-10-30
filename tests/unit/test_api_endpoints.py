"""AI service routes after VNONLY-1: only the VN equity lane is served."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from main import app  # noqa: E402

RETIRED = [
    ("post", "/api/predict"), ("get", "/api/predict/models"), ("post", "/api/analyze"),
    ("post", "/api/explain"), ("post", "/api/explain/stream"), ("post", "/api/rl/predict"), ("post", "/api/rl/train"),
]
KEPT = {"/api/capabilities", "/api/current-conditions", "/api/equity-base-rates/build",
        "/api/equity-llm/run", "/api/equity-llm/status"}


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.parametrize(("method", "path"), RETIRED)
def test_retired_btc_routes_are_not_served(client, method, path):
    assert client.request(method, path).status_code == 404


def test_equity_routes_are_served():
    assert KEPT <= {route.path for route in app.routes}


def test_capabilities_keeps_shape_and_reports_retired_btc_features(client):
    with patch.dict(os.environ, {"LLM_PROVIDER": "none"}):
        response = client.get("/api/capabilities")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"mlInference", "llmExplanation", "provider", "reason", "mlReason"}
    assert (body["mlInference"], body["llmExplanation"], body["provider"]) == (False, False, "none")
    assert "retired" in body["reason"] and "retired" in body["mlReason"]


def test_importing_main_does_not_load_btc_modules():
    # Fresh interpreter: other tests in this process import these modules directly.
    heavy = ("graph", "prediction_service", "rl_agent", "langgraph", "langchain_core", "joblib")
    code = f"import sys, main; print(','.join(m for m in {heavy!r} if m in sys.modules))"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                            env={**os.environ, "LLM_PROVIDER": "none"}, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""
