"""Model loading and prediction utilities for the FastAPI service."""

import json
import time
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np

MODELS_DIR = Path(__file__).with_name("models")
REGISTRY_PATH = MODELS_DIR / "model_registry.json"

# Cache loaded models in memory: key -> (model, metadata, loaded_at)
_MODEL_CACHE: dict[str, tuple[Any, dict, float]] = {}
_CACHE_TTL_SECONDS = 3600.0


def _get_active_model_from_registry(symbol: str, timeframe: str, window_size: int, horizon: str) -> Optional[Path]:
    """Look up active model file from model_registry.json."""
    if not REGISTRY_PATH.exists():
        return None
    try:
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        key = f"{symbol}_{timeframe}_ws{window_size}_h{horizon}"
        entry = registry.get("models", {}).get(key)
        if entry and entry.get("status") == "active":
            active_file = entry.get("active_model_file")
            if active_file:
                path = MODELS_DIR / active_file
                if path.exists():
                    return path
    except Exception as e:
        print(f"[WARN] Error reading model registry in prediction_service: {e}")
    return None


def _find_model_file(symbol: str, timeframe: str, window_size: int, horizon: str, model_name: str | None = None) -> Path | None:
    """Find the best matching model file."""
    if model_name:
        pattern = f"{symbol}_{timeframe}_ws{window_size}_h{horizon}_{model_name}.joblib"
        path = MODELS_DIR / pattern
        if path.exists():
            return path
        # Direct filename match
        exact_path = MODELS_DIR / (model_name if model_name.endswith(".joblib") else f"{model_name}.joblib")
        if exact_path.exists():
            return exact_path

    # 1. Check active model from model_registry.json first
    active_path = _get_active_model_from_registry(symbol, timeframe, window_size, horizon)
    if active_path is not None:
        return active_path

    # 2. Prefer calibrated models first, then uncalibrated (XGB -> LGB -> HGB -> RF -> LR)
    for prefix in ["XGB", "LGB", "HGB", "RF", "LR"]:
        calib_pattern = f"{symbol}_{timeframe}_ws{window_size}_h{horizon}_{prefix}_calibrated.joblib"
        calib_path = MODELS_DIR / calib_pattern
        if calib_path.exists():
            return calib_path

    for prefix in ["XGB", "LGB", "HGB", "RF", "LR"]:
        pattern = f"{symbol}_{timeframe}_ws{window_size}_h{horizon}_{prefix}*.joblib"
        matches = sorted(MODELS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
    return None


def load_model(symbol: str, timeframe: str, window_size: int, horizon: str, model_name: str | None = None):
    """Load model + metadata with simple TTL cache."""
    key = f"{symbol}_{timeframe}_ws{window_size}_h{horizon}_{model_name or 'auto'}"
    now = time.time()

    if key in _MODEL_CACHE:
        model, meta, loaded_at = _MODEL_CACHE[key]
        if now - loaded_at < _CACHE_TTL_SECONDS:
            return model, meta

    path = _find_model_file(symbol, timeframe, window_size, horizon, model_name)
    if path is None:
        raise FileNotFoundError(f"No model found for {symbol} {timeframe} ws={window_size} h={horizon} model={model_name}")

    model = joblib.load(path)
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    _MODEL_CACHE[key] = (model, meta, now)
    return model, meta


def predict_from_vector(feature_vector: list[float], symbol: str, timeframe: str, window_size: int, horizon: str, model_name: str | None = None):
    """Run inference on a single feature vector."""
    model, meta = load_model(symbol, timeframe, window_size, horizon, model_name)
    expected_dim = meta.get("features_dim") or meta.get("feature_dim") or len(meta.get("feature_names", []))
    if expected_dim and len(feature_vector) != expected_dim:
        raise ValueError(f"Feature vector length {len(feature_vector)} != expected {expected_dim}")

    X = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
    t0 = time.time()

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        confidence = float(proba[pred_idx])
    else:
        pred_idx = int(model.predict(X)[0])
        proba = None
        confidence = 1.0

    inference_ms = (time.time() - t0) * 1000.0
    resolved_model_name = meta.get("model_name", "unknown")

    if "XGB" in resolved_model_name:
        # Trained on mapped labels {0,1,2} -> original {-1,0,1}
        label = pred_idx - 1
        if proba is not None and len(proba) == 3:
            prob_down, prob_sideways, prob_up = float(proba[0]), float(proba[1]), float(proba[2])
        else:
            prob_down = prob_sideways = prob_up = 0.0
    else:
        label = pred_idx
        if proba is not None and len(proba) == 3:
            prob_down, prob_sideways, prob_up = float(proba[0]), float(proba[1]), float(proba[2])
        else:
            prob_down = prob_sideways = prob_up = 0.0

    return {
        "label": label,
        "confidence": confidence,
        "prob_down": prob_down,
        "prob_sideways": prob_sideways,
        "prob_up": prob_up,
        "model_version": resolved_model_name,
        "inference_ms": inference_ms,
    }


def list_available_models():
    """List all saved models with metadata and active registry status."""
    active_files = set()
    if REGISTRY_PATH.exists():
        try:
            reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
            for m in reg.get("models", {}).values():
                if m.get("status") == "active" and m.get("active_model_file"):
                    active_files.add(m["active_model_file"])
        except Exception:
            pass

    models = []
    for path in sorted(MODELS_DIR.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True):
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        models.append({
            "file": path.name,
            "symbol": meta.get("symbol"),
            "timeframe": meta.get("timeframe"),
            "window_size": meta.get("window_size"),
            "horizon": meta.get("horizon"),
            "model_name": meta.get("model_name"),
            "is_active": path.name in active_files,
            "metrics": meta.get("oos_metrics") or meta.get("metrics") or meta.get("test_metrics", {}),
        })
    return models
