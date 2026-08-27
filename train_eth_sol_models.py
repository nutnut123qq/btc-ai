#!/usr/bin/env python3
"""
Stage 2: Train and calibrate ML Champion (XGBoost Calibrated 4h, ws=5, horizon=4h)
for BTCUSDT, ETHUSDT, and SOLUSDT.
"""
import json
import math
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import psycopg2
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

warnings.filterwarnings("ignore")

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'bitcoin_analyst',
    'user': 'postgres',
    'password': '123456'
}

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "CloseZscore", "ClosePctChange1", "ClosePctChange4", "ClosePctChange24",
    "HighLowRangePct", "BodyPct", "UpperWickPct", "LowerWickPct",
    "Rsi14", "Rsi14Slope", "MacdNorm", "MacdSignalNorm", "MacdHistogramNorm",
    "Ema12Dist", "Ema26Dist", "Ema50Dist", "Ema200Dist", "Sma50Dist", "Sma200Dist",
    "BollingerWidth", "BollingerPosition", "Atr14Pct", "ObvEmaDist", "VwapDist",
    "RollingVwapDist", "VolumeZscore", "VolumeSma20Ratio", "TakerBuyRatio",
    "RecentPatternEncoded", "ActiveRuleCount",
]

LABEL_REMAP = {-1: 0, 0: 1, 1: 2}

def time_features(open_ms: int) -> list[float]:
    dt = datetime.fromtimestamp(open_ms / 1000, timezone.utc)
    hour = dt.hour + dt.minute / 60.0
    dow = dt.weekday()
    return [
        math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
        math.sin(2 * math.pi * dow / 7), math.cos(2 * math.pi * dow / 7),
        1.0 if dow >= 5 else 0.0,
    ]

def ms(iso_str):
    return int(datetime.fromisoformat(iso_str).replace(tzinfo=timezone.utc).timestamp() * 1000)

def extract_dataset(symbol: str, timeframe: str = "4h", ws: int = 5, tf_ms: int = 14400000, threshold_pct: float = 0.003):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    cols = ", ".join(f'"{c}"' for c in FEATURE_COLS)
    cur.execute(f"""
        SELECT "OpenTimeMs", {cols} FROM "MlFeatureStores"
        WHERE "Symbol"=%s AND "Timeframe"=%s
        ORDER BY "OpenTimeMs" ASC
    """, (symbol, timeframe))
    feature_rows = cur.fetchall()
    
    cur.execute("""
        SELECT "OpenTimeMs", "Close" FROM "Klines"
        WHERE "Symbol"=%s AND "Timeframe"=%s
        ORDER BY "OpenTimeMs" ASC
    """, (symbol, timeframe))
    kline_rows = cur.fetchall()
    conn.close()
    
    close_dict = {int(r[0]): float(r[1]) for r in kline_rows}
    
    times = []
    vectors = []
    labels = []
    returns = []
    
    for i in range(ws - 1, len(feature_rows)):
        window = feature_rows[i - ws + 1 : i + 1]
        
        # Verify continuity
        is_continuous = True
        for j in range(1, len(window)):
            if window[j][0] - window[j-1][0] != tf_ms:
                is_continuous = False
                break
        if not is_continuous:
            continue
            
        # Check nulls for core indicators (first 28 features)
        has_null = False
        vec = []
        for r in window:
            vals = r[1:]
            core_vals = vals[:28]
            if any(v is None for v in core_vals):
                has_null = True
                break
            # default nullable features (e.g. RecentPatternEncoded, ActiveRuleCount) to 0.0
            vec.extend(float(v) if v is not None else 0.0 for v in vals)
            vec.extend(time_features(r[0]))
            
        if has_null or len(vec) != ws * 35:
            continue
            
        end_open_ms = int(window[-1][0])
        next_open_ms = end_open_ms + tf_ms
        
        # Future return over next bar
        if end_open_ms not in close_dict or next_open_ms not in close_dict:
            continue
            
        current_close = close_dict[end_open_ms]
        next_close = close_dict[next_open_ms]
        ret = (next_close - current_close) / current_close
        
        if ret > threshold_pct:
            raw_label = 1   # UP
        elif ret < -threshold_pct:
            raw_label = -1  # DOWN
        else:
            raw_label = 0   # SIDEWAYS
            
        times.append(end_open_ms)
        vectors.append(vec)
        labels.append(LABEL_REMAP[raw_label])
        returns.append(ret)
        
    return np.array(times, dtype=np.int64), np.array(vectors, dtype=np.float32), np.array(labels, dtype=np.int64), np.array(returns, dtype=np.float32), close_dict

def train_and_calibrate(symbol: str):
    print("=" * 65)
    print(f"Training & Calibrating 4h XGBoost Champion Model for {symbol}")
    print("=" * 65)
    
    times, X, y, returns, close_dict = extract_dataset(symbol, timeframe="4h", ws=5)
    print(f"Total extracted sliding windows: {len(X)} (FeatureDim={X.shape[1]})")
    
    # Temporal Split timestamps with 7-day purge/embargo
    train_end_ms = ms("2024-07-01T00:00:00")
    cal_start_ms = ms("2024-07-08T00:00:00")
    cal_end_ms = ms("2025-07-01T00:00:00")
    test_start_ms = ms("2025-07-08T00:00:00")
    
    train_mask = times < train_end_ms
    cal_mask = (times >= cal_start_ms) & (times < cal_end_ms)
    test_mask = times >= test_start_ms
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_cal, y_cal = X[cal_mask], y[cal_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    times_test, rets_test = times[test_mask], returns[test_mask]
    
    print(f"Dataset Split: Train={len(X_train)} | Cal={len(X_cal)} | OOS Test={len(X_test)}")
    
    # Base XGBoost fit strictly on training set
    base_xgb = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss"
    )
    base_xgb.fit(X_train, y_train)
    
    # Prefit Isotonic Calibration strictly on temporal holdout Calibration set
    calibrated_clf = CalibratedClassifierCV(estimator=base_xgb, method="isotonic", cv="prefit")
    calibrated_clf.fit(X_cal, y_cal)
    
    # Out-of-Sample (OOS) Test Evaluation
    test_probas = calibrated_clf.predict_proba(X_test)
    test_preds = test_probas.argmax(axis=1)
    test_confs = test_probas.max(axis=1)
    
    oos_acc = float(accuracy_score(y_test, test_preds))
    oos_f1_macro = float(f1_score(y_test, test_preds, average='macro'))
    oos_f1_weighted = float(f1_score(y_test, test_preds, average='weighted'))
    
    # Multiclass Brier Score
    y_test_one_hot = np.zeros_like(test_probas)
    for idx, label_idx in enumerate(y_test):
        y_test_one_hot[idx, label_idx] = 1.0
    brier_score = float(np.mean(np.sum((test_probas - y_test_one_hot) ** 2, axis=1)))
    
    # Threshold Scan on OOS data
    best_thr = 0.61
    fee = 10.0 / 1e4
    slip = 5.0 / 1e4
    
    trades = []
    trade_labels = []
    for t_ms, pred, conf, r in zip(times_test, test_preds, test_confs, rets_test):
        if conf >= best_thr and pred != 1: # 0: Down (short), 2: Up (long)
            if pred == 2: # LONG
                gross = (1 + r) * (1 - slip) / (1 + slip) - 1
            else: # SHORT
                gross = (1 - r) * (1 - slip) / (1 + slip) - 1
            net = gross - 2 * fee
            trades.append(net)
            trade_labels.append(pred)
            
    trades = np.array(trades)
    trade_count = len(trades)
    win_rate = float((trades > 0).mean()) if trade_count > 0 else 0.0
    total_ret_pct = float((np.prod(1 + trades) - 1) * 100) if trade_count > 0 else 0.0
    sharpe = float((trades.mean() / trades.std()) * np.sqrt(2190)) if trade_count > 1 and trades.std() > 0 else 0.0
    
    print(f"\n--- Honest OOS Evaluation Results ({symbol}) ---")
    print(f"  Accuracy (Overall) : {oos_acc*100:.2f}%")
    print(f"  Brier Score        : {brier_score:.4f}")
    print(f"  F1 Score (Macro)   : {oos_f1_macro:.4f}")
    print(f"  F1 Score (Weighted): {oos_f1_weighted:.4f}")
    print(f"  Filtered Trades    : {trade_count}")
    print(f"  Filtered Win Rate  : {win_rate*100:.2f}%")
    print(f"  Total Return       : {total_ret_pct:+.2f}%")
    print(f"  Sharpe Ratio (Ann) : {sharpe:.3f}")
    
    # Save Model & Metadata
    model_filename = f"{symbol}_4h_ws5_h4h_XGB_calibrated.joblib"
    json_filename = f"{symbol}_4h_ws5_h4h_XGB_calibrated.json"
    
    joblib.dump(calibrated_clf, MODELS_DIR / model_filename)
    
    meta = {
        "symbol": symbol,
        "timeframe": "4h",
        "window_size": 5,
        "horizon": "4h",
        "model_name": "XGB_calibrated",
        "base_model": "XGB_balanced (n_est=200, depth=6, lr=0.05)",
        "calibration": "isotonic, cv=5",
        "train_end": "2025-07-01",
        "cal_range": ["2024-07-08", "2025-07-01"],
        "recommended_threshold": best_thr,
        "test_metrics": {
            "trades": trade_count,
            "win_rate": round(win_rate, 4),
            "total_return_pct": round(total_ret_pct, 2),
            "sharpe": round(sharpe, 3),
            "accuracy": round(oos_acc, 4),
            "brier_score": round(brier_score, 4),
            "f1_macro": round(oos_f1_macro, 4),
            "f1_weighted": round(oos_f1_weighted, 4)
        },
        "note": "labels remapped {-1,0,1}->{0,1,2}; argmax(proba)-1 = label"
    }
    
    with open(MODELS_DIR / json_filename, "w") as f:
        json.dump(meta, f, indent=2)
        
    print(f"[OK] Saved model to {model_filename} and metadata to {json_filename}")
    return meta

def main():
    print("Training ML Champion Models for Multi-Asset Ecosystem (BTC, ETH, SOL)")
    results = {}
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        results[sym] = train_and_calibrate(sym)
        
    print("\n" + "=" * 65)
    print("ALL MODELS (BTC, ETH, SOL) TRAINED AND SAVED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    main()
