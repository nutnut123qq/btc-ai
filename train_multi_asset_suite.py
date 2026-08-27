#!/usr/bin/env python3
"""
Multi-Asset Multi-Timeframe ML Model Training & Isotonic Calibration Suite
==========================================================================
Trains and calibrates XGBoost & LightGBM models for BTCUSDT, ETHUSDT, SOLUSDT
across 1h, 4h, and 1d timeframes with strict temporal partitions.
Computes Brier Score, ECE, OOS Accuracy, F1-Score, and Honest OOS Win Rate.
Saves .joblib and .json files into ai/models/.
"""

import sys
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import psycopg2
import joblib
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from db_config import get_db_params

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

LABEL_REMAP = {-1: 0, 0: 1, 1: 2}
LABEL_UNMAP = {0: -1, 1: 0, 2: 1}

# Strict temporal splits
TRAIN_END = "2024-07-01"
CAL_START = "2024-07-08"
CAL_END = "2025-07-01"
TEST_START = "2025-07-08"


def ms(iso):
    return int(datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp() * 1000)


def fetch_dataset(symbol, timeframe, ws, horizon):
    conn = psycopg2.connect(**get_db_params())
    cur = conn.cursor()
    cur.execute('''
        SELECT "WindowEndMs", "FeatureVector", "Label"
        FROM "WindowClassificationDatasets"
        WHERE "Symbol"=%s AND "Timeframe"=%s AND "WindowSize"=%s AND "Horizon"=%s
          AND "FeatureVector" IS NOT NULL AND "Label" IS NOT NULL
        ORDER BY "WindowEndMs"
    ''', (symbol, timeframe, ws, horizon))
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return None, None, None

    times = np.array([int(r[0]) for r in rows], dtype=np.int64)
    X = np.array([list(r[1]) for r in rows], dtype=np.float32)
    y = np.array([int(r[2]) for r in rows], dtype=np.int8)
    return times, X, y


def fetch_klines(symbol, timeframe, start_ms, end_ms):
    conn = psycopg2.connect(**get_db_params())
    cur = conn.cursor()
    cur.execute('''
        SELECT "OpenTimeMs", "Close"
        FROM "Klines"
        WHERE "Symbol"=%s AND "Timeframe"=%s AND "OpenTimeMs" >= %s AND "OpenTimeMs" <= %s
        ORDER BY "OpenTimeMs"
    ''', (symbol, timeframe, start_ms, end_ms))
    rows = cur.fetchall()
    conn.close()
    return {int(r[0]): float(r[1]) for r in rows}


def multiclass_brier_score(y_true, proba):
    """
    Multi-class Brier Score: 1/N * sum_{i} sum_{k} (p_{ik} - y_{ik})^2
    Lower is better (0.0 = perfect probabilistic calibration).
    """
    n_samples, n_classes = proba.shape
    y_one_hot = np.zeros((n_samples, n_classes))
    for i, label in enumerate(y_true):
        y_one_hot[i, label] = 1.0
    return float(np.mean(np.sum((proba - y_one_hot) ** 2, axis=1)))


def expected_calibration_error(y_true, proba, n_bins=10):
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1])
        if m.sum() > 0:
            e += m.sum() / len(conf) * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def simulate_trades(times_i, labels, confs, threshold, close_by_time, horizon_ms, fee=0.001, slip=0.0005):
    trades = []
    for t, lab, c in zip(times_i, labels, confs):
        if c < threshold or lab == 0:
            continue
        entry, exit_ = int(t), int(t) + horizon_ms
        if entry not in close_by_time or exit_ not in close_by_time:
            continue
        ep, xp = close_by_time[entry], close_by_time[exit_]
        if lab == 1:
            gross = (xp * (1 - slip) - ep * (1 + slip)) / (ep * (1 + slip))
        else:
            gross = (ep * (1 - slip) - xp * (1 + slip)) / (ep * (1 + slip))
        trades.append(gross - 2 * fee)
    return np.array(trades)


def train_and_calibrate(symbol, timeframe, ws, horizon, model_type="XGB"):
    times, X, y = fetch_dataset(symbol, timeframe, ws, horizon)
    if times is None or len(times) < 100:
        print(f"[{symbol} {timeframe} ws={ws} h={horizon}] Dataset too small or empty.")
        return None

    horizon_ms = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}[horizon]
    bars_per_year = 365.25 * 24 * 3600 * 1000 / horizon_ms

    # Temporal split
    tr_mask = times < ms(TRAIN_END)
    cal_mask = (times >= ms(CAL_START)) & (times < ms(CAL_END))
    te_mask = times >= ms(TEST_START)

    if tr_mask.sum() < 50 or cal_mask.sum() < 30 or te_mask.sum() < 30:
        # Fallback split for shorter series (e.g. 1d with fewer total bars)
        n = len(times)
        n_tr = int(n * 0.65)
        n_cal = int(n * 0.80)
        tr_mask = np.zeros(n, dtype=bool)
        cal_mask = np.zeros(n, dtype=bool)
        te_mask = np.zeros(n, dtype=bool)
        tr_mask[:n_tr] = True
        cal_mask[n_tr:n_cal] = True
        te_mask[n_cal:] = True

    y_map = np.array([LABEL_REMAP[v] for v in y])

    # 1. Base Model Fit
    if model_type == "XGB":
        base_model = xgb.XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=42
        )
    else:
        # LightGBM
        import lightgbm as lgb
        base_model = lgb.LGBMClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            n_jobs=-1,
            random_state=42,
            verbose=-1
        )

    base_model.fit(X[tr_mask], y_map[tr_mask])

    # 2. Isotonic Calibration
    calib_model = CalibratedClassifierCV(base_model, method="isotonic", cv="prefit")
    calib_model.fit(X[cal_mask], y_map[cal_mask])

    # 3. Fetch Klines for trade simulation
    klines_dict = fetch_klines(symbol, timeframe, int(times[0]), int(times[-1]) + horizon_ms * 2)

    # 4. Calibration threshold optimization
    cal_probs = calib_model.predict_proba(X[cal_mask])
    cal_confs = cal_probs.max(axis=1)
    cal_preds = np.array([LABEL_UNMAP[p] for p in cal_probs.argmax(axis=1)])
    cal_times = times[cal_mask]

    best_thr = 0.50
    best_ret = -999.0
    for thr in np.arange(0.35, 0.75, 0.01):
        tr_sim = simulate_trades(cal_times, cal_preds, cal_confs, thr, klines_dict, horizon_ms)
        if len(tr_sim) >= 10:
            ret = float(np.prod(1 + tr_sim) - 1) * 100.0
            if ret > best_ret:
                best_ret = ret
                best_thr = round(float(thr), 2)

    # 5. Honest OOS Test Evaluation
    te_probs = calib_model.predict_proba(X[te_mask])
    te_confs = te_probs.max(axis=1)
    te_preds_map = te_probs.argmax(axis=1)
    te_preds = np.array([LABEL_UNMAP[p] for p in te_preds_map])
    te_y_map = y_map[te_mask]
    te_times = times[te_mask]

    oos_acc = accuracy_score(te_y_map, te_preds_map)
    oos_f1 = f1_score(te_y_map, te_preds_map, average="weighted")
    brier = multiclass_brier_score(te_y_map, te_probs)
    ece_val = expected_calibration_error(te_y_map, te_probs)

    # OOS simulated trading with calibrated threshold
    oos_trades = simulate_trades(te_times, te_preds, te_confs, best_thr, klines_dict, horizon_ms)
    oos_win_rate = float((oos_trades > 0).mean()) if len(oos_trades) > 0 else 0.0
    oos_total_pnl = float(np.prod(1 + oos_trades) - 1) * 100.0 if len(oos_trades) > 0 else 0.0
    oos_sharpe = float(oos_trades.mean() / oos_trades.std() * np.sqrt(bars_per_year)) if len(oos_trades) > 1 and oos_trades.std() > 0 else 0.0

    # Save calibrated model and metadata
    model_slug = f"{symbol}_{timeframe}_ws{ws}_h{horizon}_{model_type}_calibrated"
    joblib_path = MODELS_DIR / f"{model_slug}.joblib"
    json_path = MODELS_DIR / f"{model_slug}.json"

    joblib.dump(calib_model, joblib_path)

    meta = {
        "model_name": model_slug,
        "symbol": symbol,
        "timeframe": timeframe,
        "window_size": ws,
        "horizon": horizon,
        "algorithm": model_type,
        "calibrated": True,
        "calibration_method": "isotonic",
        "optimal_threshold": best_thr,
        "features_dim": int(X.shape[1]),
        "total_samples": len(times),
        "train_samples": int(tr_mask.sum()),
        "cal_samples": int(cal_mask.sum()),
        "test_samples": int(te_mask.sum()),
        "metrics": {
            "oos_accuracy": round(float(oos_acc), 4),
            "oos_f1_weighted": round(float(oos_f1), 4),
            "brier_score": round(float(brier), 4),
            "expected_calibration_error": round(float(ece_val), 4),
            "oos_win_rate": round(float(oos_win_rate), 4),
            "oos_trades_count": len(oos_trades),
            "oos_total_return_pct": round(float(oos_total_pnl), 2),
            "oos_sharpe": round(float(oos_sharpe), 3),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[{symbol} {timeframe:<3} ws={ws:<2} h={horizon:<3} {model_type}] "
          f"Acc={oos_acc*100:.1f}% | Brier={brier:.4f} | WR={oos_win_rate*100:.1f}% | Thr={best_thr} -> Saved {model_slug}")

    return meta


def main():
    print("=" * 110)
    print("        STARTING MULTI-ASSET MULTI-TIMEFRAME ML TRAINING & ISOTONIC CALIBRATION")
    print("=" * 110)

    configs = [
        # BTCUSDT
        ("BTCUSDT", "4h", 5, "4h", "XGB"),
        ("BTCUSDT", "4h", 5, "1d", "XGB"),
        ("BTCUSDT", "1h", 5, "1h", "XGB"),
        ("BTCUSDT", "1h", 10, "1h", "XGB"),
        ("BTCUSDT", "1h", 5, "4h", "XGB"),
        ("BTCUSDT", "1d", 5, "1d", "XGB"),
        
        # ETHUSDT
        ("ETHUSDT", "4h", 5, "4h", "XGB"),
        ("ETHUSDT", "4h", 5, "1d", "XGB"),
        ("ETHUSDT", "1h", 5, "1h", "XGB"),
        ("ETHUSDT", "1h", 10, "1h", "XGB"),
        ("ETHUSDT", "1h", 5, "4h", "XGB"),
        ("ETHUSDT", "1d", 5, "1d", "XGB"),
        
        # SOLUSDT
        ("SOLUSDT", "4h", 5, "4h", "XGB"),
        ("SOLUSDT", "4h", 5, "1d", "XGB"),
        ("SOLUSDT", "1h", 5, "1h", "XGB"),
        ("SOLUSDT", "1h", 10, "1h", "XGB"),
        ("SOLUSDT", "1h", 5, "4h", "XGB"),
        ("SOLUSDT", "1d", 5, "1d", "XGB"),
    ]

    all_results = []
    for sym, tf, ws, h, alg in configs:
        res = train_and_calibrate(sym, tf, ws, h, alg)
        if res:
            all_results.append(res)

    print("=" * 110)
    print(f"COMPLETED TRAINING & CALIBRATION: {len(all_results)} MODELS READY IN ai/models/")
    print("=" * 110)


if __name__ == "__main__":
    main()
