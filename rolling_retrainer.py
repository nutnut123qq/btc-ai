#!/usr/bin/env python3
"""
Walk-Forward Rolling Retrainer Pipeline (FreqAI-inspired)
=========================================================
Monitors model performance / concept drift on recent market data
and performs walk-forward rolling window retraining with Isotonic Calibration.

Key capabilities:
1. Drift & Error Evaluation:
   - Measures multi-class Brier Score and Log-Loss on recent N bars (~30 days).
   - Retraining trigger: Brier Score > 0.25 OR Model Age > 30 days OR forced CLI flag.
2. Rolling Window Retraining:
   - Rolling Train Window: ~18 months prior to validation period.
   - Calibration Window then independent promotion-gate Window: ~3 months.
   - Test / Drift Window: Last ~30 days.
   - Base XGBoost trained before isotonic calibration; no split is reused for fitting and scoring.
3. Model Registry Management:
   - Saves versioned model: ai/models/{Symbol}_{TF}_ws{WS}_h{H}_XGB_v{YYYYMMDD}.joblib
   - Updates ai/models/model_registry.json for active inference serving.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import psycopg2
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

# Suppress sklearn/xgboost user warnings
warnings.filterwarnings("ignore")

# Add ai/ to python path
AI_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(AI_DIR))

from db_config import get_db_params
from train_baseline_advanced import infer_feature_names

MODELS_DIR = AI_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_PATH = MODELS_DIR / "model_registry.json"

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
LABEL_REMAP = {-1: 0, 0: 1, 1: 2}
LABEL_INV = {0: -1, 1: 0, 2: 1}
ARTIFACT_RUNTIME_PACKAGES = ("joblib", "scikit-learn", "xgboost")
FEATURE_SCHEMA_VERSION = "window-dataset-35-v1"
PURGE_BARS = 5
PROMOTION_THRESHOLDS = {
    "minimum_samples": 150,
    "minimum_class_samples": 15,
    "minimum_macro_f1": 0.40,
    "minimum_balanced_accuracy": 0.40,
    "minimum_mcc": 0.10,
    "minimum_per_class_f1": 0.10,
    "maximum_ece": 0.20,
    "minimum_loss_improvement": 0.01,
}


def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


# -----------------------------------------------------------------------------
# Database Ingestion
# -----------------------------------------------------------------------------

def fetch_sliding_windows(
    symbol: str,
    timeframe: str = "4h",
    window_size: int = 5,
    horizon: str = "4h",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fetch window classification dataset from PostgreSQL.
    Returns:
        times (np.ndarray): WindowEndMs array (sorted asc)
        X (np.ndarray): Feature vectors (2D float32)
        y (np.ndarray): Remapped labels {0, 1, 2} (1D int64)
    """
    db_params = get_db_params()
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    
    cur.execute(
        """
        SELECT "WindowEndMs", "FeatureVector", "Label"
        FROM "WindowClassificationDatasets"
        WHERE "Symbol" = %s
          AND "Timeframe" = %s
          AND "WindowSize" = %s
          AND "Horizon" = %s
          AND "FeatureVector" IS NOT NULL
          AND "Label" IS NOT NULL
        ORDER BY "WindowEndMs" ASC
        """,
        (symbol, timeframe, window_size, horizon),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32), np.array([], dtype=np.int64)

    times = np.array([int(r[0]) for r in rows], dtype=np.int64)
    X = np.array([list(r[1]) for r in rows], dtype=np.float32)
    y = np.array([LABEL_REMAP[int(r[2])] for r in rows], dtype=np.int64)

    return times, X, y


# -----------------------------------------------------------------------------
# Metric Calculations
# -----------------------------------------------------------------------------

def compute_multiclass_brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Compute multi-class Brier Score:
    Brier = (1/N) * sum_i sum_k (p_{ik} - y_{ik})^2
    """
    if len(y_true) == 0 or len(y_proba) == 0:
        return 0.0
    n_classes = y_proba.shape[1]
    y_one_hot = np.zeros_like(y_proba)
    for idx, label_idx in enumerate(y_true):
        if 0 <= label_idx < n_classes:
            y_one_hot[idx, label_idx] = 1.0
    return float(np.mean(np.sum((y_proba - y_one_hot) ** 2, axis=1)))


def expected_calibration_error(y: np.ndarray, probas: np.ndarray, bins: int = 10) -> float:
    predictions = probas.argmax(axis=1)
    confidences = probas.max(axis=1)
    correct = predictions == y
    error = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidences >= lower) & (confidences < upper)
        if upper == 1.0:
            mask |= confidences == 1.0
        if mask.any():
            error += float(mask.mean() * abs(correct[mask].mean() - confidences[mask].mean()))
    return error


def evaluate_probabilities(y: np.ndarray, probas: np.ndarray) -> Dict[str, Any]:
    """Return discrimination and calibration metrics without hiding class collapse."""
    if len(y) == 0 or len(probas) == 0:
        return {}
    probas = np.asarray(probas, dtype=np.float64)
    preds = probas.argmax(axis=1)

    brier = compute_multiclass_brier_score(y, probas)
    
    # Clip probas for robust log-loss calculation
    clipped_probas = np.clip(probas, 1e-15, 1.0 - 1e-15)
    clipped_probas /= clipped_probas.sum(axis=1, keepdims=True)
    try:
        ll = float(log_loss(y, clipped_probas, labels=[0, 1, 2]))
    except Exception:
        ll = float(-np.mean([np.log(clipped_probas[i, y[i]]) for i in range(len(y))]))

    acc = float(accuracy_score(y, preds))
    balanced_acc = float(balanced_accuracy_score(y, preds))
    mcc = float(matthews_corrcoef(y, preds))
    per_class_f1 = f1_score(y, preds, labels=[0, 1, 2], average=None, zero_division=0)
    f1_m = float(f1_score(y, preds, average="macro", zero_division=0))
    f1_w = float(f1_score(y, preds, average="weighted", zero_division=0))

    return {
        "samples": int(len(y)),
        "class_counts": np.bincount(y, minlength=3).astype(int).tolist(),
        "prediction_counts": np.bincount(preds, minlength=3).astype(int).tolist(),
        "brier_score": round(brier, 4),
        "log_loss": round(ll, 4),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "mcc": round(mcc, 4),
        "f1_per_class": {
            "down": round(float(per_class_f1[0]), 4),
            "sideways": round(float(per_class_f1[1]), 4),
            "up": round(float(per_class_f1[2]), 4),
        },
        "f1_macro": round(f1_m, 4),
        "f1_weighted": round(f1_w, 4),
        "ece": round(expected_calibration_error(y, probas), 4),
    }


def evaluate_model_performance(model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    if len(X) == 0 or len(y) == 0:
        return {}
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)
    else:
        predictions = model.predict(X)
        probas = np.eye(3, dtype=np.float32)[predictions.astype(int)]
    return evaluate_probabilities(y, probas)


def majority_baseline_metrics(y_train: np.ndarray, y_eval: np.ndarray) -> Dict[str, Any]:
    """Use training priors only; evaluation labels never influence the baseline."""
    counts = np.bincount(y_train, minlength=3).astype(np.float64)
    priors = counts / counts.sum()
    return evaluate_probabilities(y_eval, np.repeat(priors[None, :], len(y_eval), axis=0))


def assess_promotion_gate(
    validation_metrics: Dict[str, Any],
    oos_metrics: Dict[str, Any],
    validation_baseline: Dict[str, Any],
    oos_baseline: Dict[str, Any],
) -> Dict[str, Any]:
    """Fail closed unless two consecutive untouched windows beat naive priors."""
    failures: List[str] = []
    limits = PROMOTION_THRESHOLDS
    for window, metrics, baseline in (
        ("validation", validation_metrics, validation_baseline),
        ("oos", oos_metrics, oos_baseline),
    ):
        if not metrics or not baseline:
            failures.append(f"{window}: metrics unavailable")
            continue
        checks = {
            "sample count": metrics["samples"] >= limits["minimum_samples"],
            "class support": min(metrics["class_counts"]) >= limits["minimum_class_samples"],
            "macro F1": metrics["f1_macro"] >= limits["minimum_macro_f1"],
            "balanced accuracy": metrics["balanced_accuracy"] >= limits["minimum_balanced_accuracy"],
            "MCC": metrics["mcc"] >= limits["minimum_mcc"],
            "per-class F1": min(metrics["f1_per_class"].values()) >= limits["minimum_per_class_f1"],
            "ECE": metrics["ece"] <= limits["maximum_ece"],
            "Brier improvement": metrics["brier_score"] <= baseline["brier_score"] - limits["minimum_loss_improvement"],
            "log-loss improvement": metrics["log_loss"] <= baseline["log_loss"] - limits["minimum_loss_improvement"],
        }
        failures.extend(f"{window}: {name}" for name, passed in checks.items() if not passed)
    return {
        "passed": not failures,
        "policy": "two-independent-windows-v1",
        "thresholds": limits,
        "failures": failures,
    }


def dataset_provenance(
    symbol: str,
    timeframe: str,
    window_size: int,
    horizon: str,
    times: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, Any]:
    digest = hashlib.sha256()
    for array in (times, X, y):
        digest.update(np.ascontiguousarray(array).tobytes())
    label_columns = {
        "1h": ('"TargetDirection1h"', '"TargetDirectionTb1h"'),
        "4h": ('"TargetDirection4h"', '"TargetDirectionTb4h"'),
        "1d": ('"TargetDirection1d"', '"TargetDirectionTb1d"'),
    }
    if horizon not in label_columns:
        raise ValueError(f"Unsupported horizon for label lineage: {horizon}")
    close_column, triple_barrier_column = label_columns[horizon]
    with psycopg2.connect(**get_db_params()) as connection, connection.cursor() as cursor:
        cursor.execute(
            f'''SELECT count(*),
                count(*) FILTER (WHERE w."Label" = p.{close_column}),
                count(*) FILTER (WHERE w."Label" = p.{triple_barrier_column})
                FROM "WindowClassificationDatasets" w
                LEFT JOIN "PriceTargets" p
                  ON p."Symbol" = w."Symbol" AND p."Timeframe" = w."Timeframe"
                 AND p."OpenTimeMs" = w."WindowEndMs"
                WHERE w."Symbol" = %s AND w."Timeframe" = %s
                  AND w."WindowSize" = %s AND w."Horizon" = %s''',
            (symbol, timeframe, window_size, horizon),
        )
        total, close_matches, triple_barrier_matches = map(int, cursor.fetchone())
    matching_sources = []
    if total == len(y) and close_matches == total:
        matching_sources.append(f"PriceTargets.TargetDirection{horizon}")
    if total == len(y) and triple_barrier_matches == total:
        matching_sources.append(f"PriceTargets.TargetDirectionTb{horizon}")

    return {
        "source_table": "WindowClassificationDatasets",
        "identity": f"{symbol}_{timeframe}_ws{window_size}_h{horizon}",
        "row_count": int(len(y)),
        "first_window_end_ms": int(times[0]),
        "last_window_end_ms": int(times[-1]),
        "dataset_sha256": digest.hexdigest(),
        "label_lineage": {
            "complete": len(matching_sources) == 1,
            "source_column": matching_sources[0] if len(matching_sources) == 1 else None,
            "matching_sources": matching_sources,
            "close_to_close_matches": close_matches,
            "triple_barrier_matches": triple_barrier_matches,
        },
    }


# -----------------------------------------------------------------------------
# Model Registry Helpers
# -----------------------------------------------------------------------------

def load_model_registry() -> Dict[str, Any]:
    """Load model registry from disk or return initialized structure."""
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            log(f"[WARN] Error reading model registry: {e}. Reinitializing.")
    return {"updated_at_utc": datetime.now(timezone.utc).isoformat(), "models": {}, "history": []}


def save_model_registry(registry: Dict[str, Any]):
    """Atomically save the model registry."""
    registry["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    temporary_path = REGISTRY_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    os.replace(temporary_path, REGISTRY_PATH)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_schema_hash(feature_names: List[str]) -> str:
    payload = json.dumps(feature_names, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def get_active_model_for_symbol(
    symbol: str,
    timeframe: str = "4h",
    window_size: int = 5,
    horizon: str = "4h",
) -> Tuple[Optional[Any], Optional[Dict[str, Any]], Optional[str]]:
    """
    Get only the registry-promoted model; legacy filesystem fallbacks are unsafe.
    Returns: (model_obj, model_metadata, model_file_name)
    """
    registry = load_model_registry()
    key = f"{symbol}_{timeframe}_ws{window_size}_h{horizon}"
    
    entry = registry.get("models", {}).get(key)
    if isinstance(entry, dict) and entry.get("status") == "active":
        try:
            from prediction_service import load_model

            model, metadata = load_model(symbol, timeframe, window_size, horizon)
            return model, metadata, str(entry["active_model_file"])
        except Exception as e:
            log(f"[WARN] Registered model is not promotion-safe: {e}")

    return None, None, None


# -----------------------------------------------------------------------------
# Rolling Window Retraining Logic
# -----------------------------------------------------------------------------

def retrain_symbol_rolling(
    symbol: str,
    timeframe: str = "4h",
    window_size: int = 5,
    horizon: str = "4h",
    drift_days: int = 30,
    brier_threshold: float = 0.25,
    train_months: int = 18,
    val_months: int = 3,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates drift on recent data for `symbol` and performs rolling retrain if triggered.
    """
    log("=" * 70)
    log(f"Processing Rolling Retrainer for {symbol} ({timeframe} ws={window_size} h={horizon})")
    log("=" * 70)

    # 1. Fetch historical windows
    times, X, y = fetch_sliding_windows(symbol, timeframe, window_size, horizon)
    if len(X) < 200:
        log(f"[ERROR] Insufficient dataset for {symbol}: only {len(X)} samples available.")
        return {
            "symbol": symbol,
            "status": "skipped_insufficient_data",
            "retrained": False,
        }

    total_samples = len(X)
    max_time_ms = int(times[-1])
    max_dt = datetime.fromtimestamp(max_time_ms / 1000, timezone.utc)
    log(f"Total dataset: {total_samples:,} windows | Latest bar: {max_dt.isoformat()} (dim={X.shape[1]})")

    # 2. Define rolling time splits
    drift_ms = int(drift_days * 24 * 3600 * 1000)
    val_ms = int(val_months * 30.44 * 24 * 3600 * 1000)
    train_ms = int(train_months * 30.44 * 24 * 3600 * 1000)
    timeframe_ms = {"15m": 900_000, "30m": 1_800_000, "1h": 3_600_000,
                    "4h": 14_400_000, "1d": 86_400_000}.get(timeframe)
    if timeframe_ms is None:
        raise ValueError(f"Unsupported timeframe for temporal purge: {timeframe}")
    purge_ms = PURGE_BARS * timeframe_ms

    test_start_ms = max_time_ms - drift_ms
    val_start_ms = test_start_ms - val_ms
    gate_start_ms = test_start_ms - min(int(30 * 24 * 3600 * 1000), val_ms // 3)
    train_start_ms = max(int(times[0]), val_start_ms - train_ms)

    test_mask = times >= test_start_ms
    gate_mask = (times >= gate_start_ms) & (times < test_start_ms - purge_ms)
    calibration_mask = (times >= val_start_ms) & (times < gate_start_ms - purge_ms)
    train_mask = (times >= train_start_ms) & (times < val_start_ms - purge_ms)

    X_train, y_train = X[train_mask], y[train_mask]
    X_calibration, y_calibration = X[calibration_mask], y[calibration_mask]
    X_val, y_val = X[gate_mask], y[gate_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    log(
        f"Purged temporal splits ({PURGE_BARS} bars): Train={len(X_train):,} | "
        f"Calibration={len(X_calibration):,} | Gate={len(X_val):,} | Test={len(X_test):,}"
    )
    if min(len(X_train), len(X_calibration), len(X_val), len(X_test)) < 100:
        return {"symbol": symbol, "status": "skipped_insufficient_temporal_splits", "retrained": False}

    # 3. Check Current Active Model Performance (Drift Detection)
    active_model, active_meta, active_file = get_active_model_for_symbol(symbol, timeframe, window_size, horizon)
    
    before_metrics = {}
    should_retrain = force
    retrain_reason = []

    if active_model is not None:
        before_metrics = evaluate_model_performance(active_model, X_test, y_test)
        brier_curr = before_metrics["brier_score"]
        log(f"Active Model [{active_file}]:")
        log(f"  Brier Score (recent {drift_days}d): {brier_curr:.4f} (Threshold: {brier_threshold})")
        log(f"  Log-Loss    (recent {drift_days}d): {before_metrics['log_loss']:.4f}")
        log(f"  Accuracy    (recent {drift_days}d): {before_metrics['accuracy']*100:.2f}%")
        log(f"  Macro F1    (recent {drift_days}d): {before_metrics['f1_macro']:.4f}")

        if brier_curr > brier_threshold:
            should_retrain = True
            retrain_reason.append(f"Concept drift detected: Brier {brier_curr:.4f} > {brier_threshold}")

        # Check model age from metadata or file creation date
        created_str = active_meta.get("created_at_utc") or active_meta.get("trained_at_utc")
        if created_str:
            try:
                created_dt = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                age_days = (datetime.now(timezone.utc) - created_dt).total_days()
                if age_days > 30:
                    should_retrain = True
                    retrain_reason.append(f"Model cycle expired: age {age_days:.1f} days > 30 days")
            except Exception:
                pass
    else:
        should_retrain = True
        retrain_reason.append("No active model found in registry or models directory")

    if force:
        retrain_reason.append("Force retrain flag enabled")

    if not should_retrain:
        log(f"[INFO] No retrain required for {symbol}. Model is healthy and within tolerances.")
        return {
            "symbol": symbol,
            "status": "healthy_no_retrain_needed",
            "retrained": False,
            "active_model_file": active_file,
            "before_metrics": before_metrics,
            "after_metrics": before_metrics,
        }

    log(f"[TRIGGER] Initiating Rolling Retrain for {symbol}. Reason(s): {', '.join(retrain_reason)}")

    # 4. Train New XGBoost Classifier on Rolling Train Window
    sample_weights = compute_sample_weight("balanced", y_train)

    base_xgb = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss",
        tree_method="hist",
    )

    log(f"Fitting XGBoost on train only ({len(X_train):,} samples)...")
    base_xgb.fit(X_train, y_train, sample_weight=sample_weights)
    log(f"Fitting isotonic calibration on later calibration-only data ({len(X_calibration):,} samples)...")
    calibrated_clf = CalibratedClassifierCV(estimator=FrozenEstimator(base_xgb), method="isotonic")
    calibrated_clf.fit(X_calibration, y_calibration)

    # 5. Evaluate on Validation & Test Windows
    val_metrics = evaluate_model_performance(calibrated_clf, X_val, y_val)
    after_metrics = evaluate_model_performance(calibrated_clf, X_test, y_test)
    val_baseline = majority_baseline_metrics(y_train, y_val)
    test_baseline = majority_baseline_metrics(y_train, y_test)
    provenance = dataset_provenance(symbol, timeframe, window_size, horizon, times, X, y)
    promotion_gate = assess_promotion_gate(val_metrics, after_metrics, val_baseline, test_baseline)
    if not provenance["label_lineage"]["complete"]:
        promotion_gate["passed"] = False
        promotion_gate["failures"].append("dataset: label lineage is ambiguous or incomplete")

    # Threshold scan for optimal trade filtering on validation set
    val_probas = calibrated_clf.predict_proba(X_val)
    best_thr = 0.55
    best_thr_acc = 0.0
    for thr in np.arange(0.50, 0.75, 0.02):
        max_p = val_probas.max(axis=1)
        preds = val_probas.argmax(axis=1)
        mask = max_p >= thr
        if mask.sum() >= 20:
            acc = float((preds[mask] == y_val[mask]).mean())
            if acc > best_thr_acc:
                best_thr_acc = acc
                best_thr = float(round(thr, 2))

    log("Rolling Retrain Evaluation Results:")
    log(f"  Validation Accuracy : {val_metrics['accuracy']*100:.2f}% | Val Brier: {val_metrics['brier_score']:.4f}")
    log(f"  OOS Test Accuracy   : {after_metrics['accuracy']*100:.2f}% (Before: {before_metrics.get('accuracy', 0)*100:.2f}%)")
    log(f"  OOS Test Brier Score: {after_metrics['brier_score']:.4f} (Before: {before_metrics.get('brier_score', 0):.4f})")
    log(f"  OOS Test Log-Loss   : {after_metrics['log_loss']:.4f} (Before: {before_metrics.get('log_loss', 0):.4f})")
    log(f"  Optimal Threshold   : {best_thr:.2f} (Val Acc @ thr: {best_thr_acc*100:.2f}%)")
    log(f"  Promotion Gate      : {'PASS' if promotion_gate['passed'] else 'FAIL'}")

    if not promotion_gate["passed"]:
        registry = load_model_registry()
        model_key = f"{symbol}_{timeframe}_ws{window_size}_h{horizon}"
        existing = registry.get("models", {}).get(model_key)
        if isinstance(existing, dict) and existing.get("status") == "active":
            existing["status"] = "quarantined"
            existing["quarantine_reason"] = "Artifact lacks a passing independent-window promotion gate."
            save_model_registry(registry)
        log(f"[REJECTED] Candidate not saved or promoted: {', '.join(promotion_gate['failures'])}")
        return {
            "symbol": symbol,
            "status": "candidate_rejected",
            "retrained": False,
            "before_metrics": before_metrics,
            "validation_metrics": val_metrics,
            "oos_metrics": after_metrics,
            "after_metrics": after_metrics,
            "validation_baseline": val_baseline,
            "oos_baseline": test_baseline,
            "promotion_gate": promotion_gate,
        }

    # 6. Save Versioned Artifacts
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    version_tag = f"v{date_str}"
    model_filename = f"{symbol}_{timeframe}_ws{window_size}_h{horizon}_XGB_{version_tag}.joblib"
    json_filename = f"{symbol}_{timeframe}_ws{window_size}_h{horizon}_XGB_{version_tag}.json"
    
    model_path = MODELS_DIR / model_filename
    json_path = MODELS_DIR / json_filename
    temporary_model_path = model_path.with_suffix(".joblib.tmp")
    joblib.dump(calibrated_clf, temporary_model_path)
    os.replace(temporary_model_path, model_path)

    feature_names = infer_feature_names(window_size, int(X.shape[1]))
    if len(feature_names) != int(X.shape[1]):
        raise RuntimeError("Could not prove the feature schema for the trained artifact.")

    metadata = {
        "symbol": symbol,
        "timeframe": timeframe,
        "window_size": window_size,
        "horizon": horizon,
        "model_name": f"XGB_{version_tag}",
        "version": version_tag,
        "base_model": "XGBoost (n_estimators=200, depth=6, lr=0.04)",
        "calibration": "isotonic on later calibration-only window",
        "feature_dim": int(X.shape[1]),
        "feature_names": feature_names,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_hash": feature_schema_hash(feature_names),
        "artifact_sha256": sha256_file(model_path),
        "library_versions": {
            package: importlib.metadata.version(package)
            for package in ARTIFACT_RUNTIME_PACKAGES
        },
        "class_mapping": {"0": -1, "1": 0, "2": 1},
        "data_provenance": provenance,
        "promotion_gate": promotion_gate,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_window_start": datetime.fromtimestamp(train_start_ms / 1000, timezone.utc).isoformat(),
        "train_window_end": datetime.fromtimestamp((val_start_ms - purge_ms) / 1000, timezone.utc).isoformat(),
        "calibration_window_start": datetime.fromtimestamp(val_start_ms / 1000, timezone.utc).isoformat(),
        "calibration_window_end": datetime.fromtimestamp((gate_start_ms - purge_ms) / 1000, timezone.utc).isoformat(),
        "val_window_start": datetime.fromtimestamp(gate_start_ms / 1000, timezone.utc).isoformat(),
        "val_window_end": datetime.fromtimestamp((test_start_ms - purge_ms) / 1000, timezone.utc).isoformat(),
        "test_window_start": datetime.fromtimestamp(test_start_ms / 1000, timezone.utc).isoformat(),
        "test_window_end": max_dt.isoformat(),
        "sample_counts": {
            "train": len(X_train),
            "calibration": len(X_calibration),
            "val": len(X_val),
            "test": len(X_test),
        },
        "optimal_threshold": best_thr,
        "validation_metrics": val_metrics,
        "validation_baseline": val_baseline,
        "oos_metrics": after_metrics,
        "oos_baseline": test_baseline,
        "before_metrics": before_metrics,
        "retrain_reasons": retrain_reason,
        "note": "labels remapped {-1,0,1}->{0,1,2}; argmax(proba)-1 = direction",
    }
    temporary_json_path = json_path.with_suffix(".json.tmp")
    temporary_json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(temporary_json_path, json_path)
    log(f"[OK] Saved model artifact: {model_filename}")

    # 7. Update Model Registry
    registry = load_model_registry()
    model_key = f"{symbol}_{timeframe}_ws{window_size}_h{horizon}"
    
    # Archive previous active entry if present
    if model_key in registry.get("models", {}):
        prev_entry = dict(registry["models"][model_key])
        prev_entry["archived_at_utc"] = datetime.now(timezone.utc).isoformat()
        prev_entry["status"] = "archived"
        registry.setdefault("history", []).append(prev_entry)

    # Register new active model
    registry.setdefault("models", {})[model_key] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "window_size": window_size,
        "horizon": horizon,
        "active_model_file": model_filename,
        "version": version_tag,
        "created_at_utc": metadata["created_at_utc"],
        "status": "active",
        "validation_accuracy": val_metrics["accuracy"],
        "oos_brier_score": after_metrics["brier_score"],
        "oos_log_loss": after_metrics["log_loss"],
        "oos_accuracy": after_metrics["accuracy"],
        "optimal_threshold": best_thr,
        "train_window_start": metadata["train_window_start"],
        "train_window_end": metadata["train_window_end"],
        "val_window_start": metadata["val_window_start"],
        "val_window_end": metadata["val_window_end"],
        "test_window_start": metadata["test_window_start"],
        "test_window_end": metadata["test_window_end"],
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
    }
    save_model_registry(registry)
    log(f"[OK] Updated model registry for {model_key} -> {model_filename}")

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "window_size": window_size,
        "horizon": horizon,
        "status": "retrained_successfully",
        "retrained": True,
        "version": version_tag,
        "model_file": model_filename,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "val_metrics": val_metrics,
        "optimal_threshold": best_thr,
    }


# -----------------------------------------------------------------------------
# Main Runner & Reporting
# -----------------------------------------------------------------------------

def run_rolling_retrainer_suite(
    symbols: List[str],
    timeframe: str = "4h",
    window_size: int = 5,
    horizon: str = "4h",
    drift_days: int = 30,
    brier_threshold: float = 0.25,
    train_months: int = 18,
    val_months: int = 3,
    force: bool = False,
) -> List[Dict[str, Any]]:
    """Run rolling retrain loop across target symbols and produce summary report."""
    results = []
    for sym in symbols:
        res = retrain_symbol_rolling(
            symbol=sym,
            timeframe=timeframe,
            window_size=window_size,
            horizon=horizon,
            drift_days=drift_days,
            brier_threshold=brier_threshold,
            train_months=train_months,
            val_months=val_months,
            force=force,
        )
        results.append(res)
    
    # Print Acceptance Report Table
    print("\n" + "=" * 92)
    print("                      WALK-FORWARD ROLLING RETRAINER REPORT")
    print("=" * 92)
    header = f"{'Symbol':<10} | {'Active Model Version':<22} | {'Brier (Before->After)':<23} | {'Acc (Before->After)':<21} | {'Status':<10}"
    print(header)
    print("-" * 92)

    for r in results:
        sym = r.get("symbol", "N/A")
        status = r.get("status", "unknown")
        ver = r.get("model_file", r.get("active_model_file", "N/A"))
        
        before = r.get("before_metrics", {})
        after = r.get("after_metrics", {})
        
        brier_before = f"{before.get('brier_score', 0):.4f}" if "brier_score" in before else "N/A"
        brier_after = f"{after.get('brier_score', 0):.4f}" if "brier_score" in after else "N/A"
        brier_str = f"{brier_before} -> {brier_after}"

        acc_before = f"{before.get('accuracy', 0)*100:.1f}%" if "accuracy" in before else "N/A"
        acc_after = f"{after.get('accuracy', 0)*100:.1f}%" if "accuracy" in after else "N/A"
        acc_str = f"{acc_before} -> {acc_after}"

        if r.get("retrained"):
            status_str = "RETRAINED"
        elif status == "candidate_rejected":
            status_str = "REJECTED"
        elif status == "healthy_no_retrain_needed":
            status_str = "HEALTHY"
        else:
            status_str = "SKIPPED"
        print(f"{sym:<10} | {ver:<22} | {brier_str:<23} | {acc_str:<21} | {status_str:<10}")

    print("=" * 92)
    return results


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Rolling Retrainer Pipeline")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT", help="Comma-separated list of symbols")
    parser.add_argument("--timeframe", default="4h", help="Timeframe (default: 4h)")
    parser.add_argument("--ws", type=int, default=5, help="Window size (default: 5)")
    parser.add_argument("--horizon", default="4h", help="Prediction horizon (default: 4h)")
    parser.add_argument("--drift-days", type=int, default=30, help="Evaluation drift window in days (default: 30)")
    parser.add_argument("--brier-threshold", type=float, default=0.25, help="Brier score drift threshold (default: 0.25)")
    parser.add_argument("--train-months", type=int, default=18, help="Rolling train window in months (default: 18)")
    parser.add_argument("--val-months", type=int, default=3, help="Validation window in months (default: 3)")
    parser.add_argument("--force", action="store_true", help="Force retraining even if drift threshold is not exceeded")
    
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    run_rolling_retrainer_suite(
        symbols=symbols,
        timeframe=args.timeframe,
        window_size=args.ws,
        horizon=args.horizon,
        drift_days=args.drift_days,
        brier_threshold=args.brier_threshold,
        train_months=args.train_months,
        val_months=args.val_months,
        force=args.force,
    )


if __name__ == "__main__":
    main()
