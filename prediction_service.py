"""Model loading and prediction utilities for the FastAPI service."""

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

MODELS_DIR = Path(__file__).with_name("models")

# Cache loaded models in memory: key -> (model, metadata, loaded_at)
_MODEL_CACHE: dict[str, tuple[Any, dict, float]] = {}
_CACHE_TTL_SECONDS = 3600.0


def _find_model_file(symbol: str, timeframe: str, window_size: int, horizon: str, model_name: str | None = None) -> Path | None:
    """Find the best matching model file."""
    if model_name:
        pattern = f"{symbol}_{timeframe}_ws{window_size}_h{horizon}_{model_name}.joblib"
        path = MODELS_DIR / pattern
        return path if path.exists() else None

    # Prefer XGB, then LGB, then HGB, then RF, then LR
    for prefix in ["XGB", "LGB", "HGB", "RF", "LR"]:
        pattern = f"{symbol}_{timeframe}_ws{window_size}_h{horizon}_{prefix}*.joblib"
        matches = sorted(MODELS_DIR.glob(pattern), reverse=True)
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
    expected_dim = meta.get("feature_dim") or len(meta.get("feature_names", []))
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
    """List all saved models with metadata."""
    models = []
    for path in sorted(MODELS_DIR.glob("*.joblib")):
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        models.append({
            "file": path.name,
            "symbol": meta.get("symbol"),
            "timeframe": meta.get("timeframe"),
            "window_size": meta.get("window_size"),
            "horizon": meta.get("horizon"),
            "model_name": meta.get("model_name"),
            "metrics": meta.get("metrics", {}),
        })
    return models
