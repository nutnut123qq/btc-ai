"""Hermetic import/startup smoke check used by CI."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["LLM_PROVIDER"] = "none"
os.environ.pop("GOOGLE_API_KEY", None)
os.environ.pop("BLACKBOX_API_KEY", None)

from main import app  # noqa: E402

# VN-only service (VNONLY-1): equity routes must exist, the retired BTC routes must not.
REQUIRED = {"/api/capabilities", "/api/current-conditions", "/api/equity-base-rates/build", "/api/equity-llm/run"}
RETIRED = {"/api/predict", "/api/predict/models", "/api/analyze", "/api/explain", "/api/explain/stream",
           "/api/rl/predict", "/api/rl/train"}


def main() -> None:
    routes = {route.path for route in app.routes}
    missing = sorted(REQUIRED - routes)
    if missing:
        raise SystemExit(f"missing API routes: {missing}")
    retired = sorted(RETIRED & routes)
    if retired:
        raise SystemExit(f"retired BTC routes still served: {retired}")
    loaded = sorted(m for m in ("graph", "prediction_service", "rl_agent", "langgraph", "langchain_core") if m in sys.modules)
    if loaded:
        raise SystemExit(f"retired BTC modules imported at startup: {loaded}")

    print("AI startup smoke passed (LLM_PROVIDER=none).")


if __name__ == "__main__":
    main()
