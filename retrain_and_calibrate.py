#!/usr/bin/env python3
"""
Tự động huấn luyện, Calibrate và quét ngưỡng tối ưu cho các mô hình XGBoost
trên nhiều khung thời gian (4h, 1h, 30m).

Pipeline:
  1. Đọc tập dữ liệu WindowClassificationDatasets từ PostgreSQL.
  2. Chia dữ liệu theo mốc thời gian (Train, Calibrate, Test).
  3. Huấn luyện XGBoost Classifier.
  4. Hiệu chỉnh xác suất (Isotonic Calibration).
  5. Tìm ngưỡng tin cậy tối ưu (Threshold Scan).
  6. Lưu mô hình .joblib và metadata .json vào ai/models/.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import psycopg2
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from db_config import get_db_params

DB = get_db_params()

LABEL_REMAP = {-1: 0, 0: 1, 1: 2}
MODELS_DIR = Path(__file__).parent / "models"


def fetch_windows(symbol, timeframe, ws, horizon):
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute(
        """SELECT "WindowEndMs", "FeatureVector", "Label"
           FROM "WindowClassificationDatasets"
           WHERE "Symbol"=%s AND "Timeframe"=%s AND "WindowSize"=%s AND "Horizon"=%s
             AND "FeatureVector" IS NOT NULL AND "Label" IS NOT NULL
           ORDER BY "WindowEndMs" """,
        (symbol, timeframe, ws, horizon),
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return np.array([]), np.array([]), np.array([])
    times = np.array([r[0] for r in rows], dtype=np.int64)
    X = np.array([r[1] for r in rows], dtype=np.float32)
    y = np.array([LABEL_REMAP[r[2]] for r in rows], dtype=np.int64)
    return times, X, y


def retrain_and_calibrate(symbol="BTCUSDT", timeframe="4h", ws=5, horizon="4h"):
    print(f"\n--- Processing {symbol} {timeframe} ws={ws} h={horizon} ---")
    times, X, y = fetch_windows(symbol, timeframe, ws, horizon)
    if len(X) < 100:
        print(f"Not enough data for {timeframe} (got {len(X)} samples), skip.")
        return

    n = len(X)
    n_train = int(n * 0.6)
    n_cal = int(n * 0.8)
    purge = ws + 5  # Purge overlapping windows and forward label horizons

    X_train, y_train = X[:n_train], y[:n_train]
    X_cal, y_cal = X[n_train + purge : n_cal], y[n_train + purge : n_cal]
    X_test, y_test = X[n_cal + purge :], y[n_cal + purge :]

    print(f"Samples: Train={len(X_train)}, Calibrate={len(X_cal)}, Test={len(X_test)} (Purge={purge} bars)")

    base_xgb = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss",
    )
    base_xgb.fit(X_train, y_train)

    calibrated_clf = CalibratedClassifierCV(estimator=base_xgb, method="isotonic", cv="prefit")
    calibrated_clf.fit(X_cal, y_cal)

    # Scan threshold on calibration set
    cal_probas = calibrated_clf.predict_proba(X_cal)
    best_thr = 0.55
    best_acc = 0.0

    for thr in np.arange(0.50, 0.75, 0.02):
        max_p = cal_probas.max(axis=1)
        preds = cal_probas.argmax(axis=1)
        mask = max_p >= thr
        if mask.sum() > 20:
            acc = (preds[mask] == y_cal[mask]).mean()
            if acc > best_acc:
                best_acc = acc
                best_thr = float(thr)

    print(f"Optimal threshold found: {best_thr:.2f} (Calibration Acc: {best_acc * 100:.2f}%)")

    # Evaluate on OOS Test set
    test_probas = calibrated_clf.predict_proba(X_test)
    test_preds = test_probas.argmax(axis=1)
    mask = test_probas.max(axis=1) >= best_thr
    if mask.sum() > 0:
        test_acc = (test_preds[mask] == y_test[mask]).mean()
        print(f"Test Accuracy @ Thr {best_thr:.2f}: {test_acc * 100:.2f}% ({mask.sum()}/{len(X_test)} trades)")

    # Save model and metadata
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_filename = f"{symbol}_{timeframe}_ws{ws}_h{horizon}_XGB_calibrated.joblib"
    model_path = MODELS_DIR / model_filename
    joblib.dump(calibrated_clf, model_path)

    meta_filename = f"{symbol}_{timeframe}_ws{ws}_h{horizon}_XGB_calibrated.json"
    meta_path = MODELS_DIR / meta_filename
    metadata = {
        "symbol": symbol,
        "timeframe": timeframe,
        "window_size": ws,
        "horizon": horizon,
        "model_name": "XGB_calibrated",
        "feature_dim": X.shape[1],
        "optimal_threshold": best_thr,
        "test_accuracy": float(test_acc) if mask.sum() > 0 else 0.0,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved calibrated model to {model_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Retrain & Calibrate multi-timeframe models.")
    parser.add_argument("--symbol", default="BTCUSDT")
    args = parser.parse_args()

    configs = [
        ("4h", 5, "4h"),
        ("1h", 5, "1h"),
        ("30m", 5, "1h"),
    ]

    for tf, ws, h in configs:
        retrain_and_calibrate(args.symbol, tf, ws, h)


if __name__ == "__main__":
    main()
