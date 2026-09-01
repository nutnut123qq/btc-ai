"""Hermetic import/startup smoke check used by CI."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["LLM_PROVIDER"] = "none"
os.environ.pop("GOOGLE_API_KEY", None)
os.environ.pop("BLACKBOX_API_KEY", None)

from main import app, _provider_capability  # noqa: E402


def main() -> None:
    provider, available, _ = _provider_capability()
    if provider != "none" or available:
        raise SystemExit("provider-none startup contract failed")

    routes = {route.path for route in app.routes}
    required = {"/api/capabilities", "/api/predict", "/api/analyze", "/api/explain"}
    missing = sorted(required - routes)
    if missing:
        raise SystemExit(f"missing API routes: {missing}")

    print("AI startup smoke passed (LLM_PROVIDER=none).")


if __name__ == "__main__":
    main()
