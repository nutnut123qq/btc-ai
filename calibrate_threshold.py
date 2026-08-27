#!/usr/bin/env python3
"""
D1: Probability calibration + threshold tuning cho 4h model.

Pipeline (time-split, purged 2 bars ở mỗi ranh giới):
  train:      < 2024-07-01        -> fit XGB (params giống production)
  calibrate:  2024-07-08..2025-07 -> fit isotonic CalibratedClassifierCV(prefit)
                                     + scan threshold tối ưu net PnL
  test:       2025-07-08..now     -> đánh giá honest OOS

So sánh trên test:
  A. all trades (không threshold)
  B. raw max-prob >= best threshold (uncalibrated)
  C. calibrated prob >= best threshold (isotonic)
Kèm ECE (expected calibration error) trước/sau calibration.

Usage: python calibrate_threshold.py [--timeframe 4h --window-size 5 --horizon 4h]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg2
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, str(Path(__file__).parent))
from backtest_strategy import fetch_klines
from db_config import get_db_params

DB = get_db_params()

LABEL_REMAP = {-1: 0, 0: 1, 1: 2}

TRAIN_END = "2024-07-01"
CAL_START = "2024-07-08"
CAL_END = "2025-07-01"
TEST_START = "2025-07-08"


def ms(iso):
    return int(datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp() * 1000)


def fetch_windows(symbol, timeframe, ws, horizon):
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute(
        """SELECT "WindowEndMs", "FeatureVector", "Label"
           FROM "WindowClassificationDatasets"
           WHERE "Symbol"=%s AND "Timeframe"=%s AND "WindowSize"=%s AND "Horizon"=%s
             AND "FeatureVector" IS NOT NULL AND "Label" IS NOT NULL
           ORDER BY "WindowEndMs" """,
        (symbol, timeframe, ws, horizon))
    rows = cur.fetchall()
    conn.close()
    times = np.array([int(r[0]) for r in rows], dtype=np.int64)
    X = np.array([list(r[1]) for r in rows], dtype=np.float32)
    y = np.array([int(r[2]) for r in rows], dtype=np.int8)
    return times, X, y


def simulate(times_i, labels, confs, threshold, close_by_time, horizon_ms, fee, slip):
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


def metrics(trades, bars_per_year):
    if len(trades) == 0:
        return {"trades": 0}
    eq = float(np.prod(1 + trades))
    sharpe = float(trades.mean() / trades.std() * np.sqrt(bars_per_year)) if len(trades) > 1 and trades.std() > 0 else 0.0
    return {
        "trades": len(trades),
        "win_rate": round(float((trades > 0).mean()), 4),
        "total_return_pct": round((eq - 1) * 100, 2),
        "avg_trade_pct": round(float(trades.mean()) * 100, 4),
        "sharpe": round(sharpe, 3),
    }


def ece(y_true, proba, n_bins=10):
    """Multiclass ECE: confidence = max prob, correct = argmax == y."""
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1])
        if m.sum() > 0:
            e += m.sum() / len(conf) * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--timeframe", default="4h")
    p.add_argument("--window-size", type=int, default=5)
    p.add_argument("--horizon", default="4h")
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    args = p.parse_args()

    horizon_ms = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}[args.horizon]
    bars_per_year = 365.25 * 24 * 3600 * 1000 / horizon_ms
    fee, slip = args.fee_bps / 1e4, args.slippage_bps / 1e4

    times, X, y = fetch_windows(args.symbol, args.timeframe, args.window_size, args.horizon)
    print(f"Loaded {len(times)} windows")

    tr = times < ms(TRAIN_END)
    cal = (times >= ms(CAL_START)) & (times < ms(CAL_END))
    te = times >= ms(TEST_START)
    print(f"train={tr.sum()} cal={cal.sum()} test={te.sum()}")

    y_map = np.array([LABEL_REMAP[v] for v in y])
    model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                              eval_metric="mlogloss", n_jobs=-1, random_state=42)
    model.fit(X[tr], y_map[tr])

    calib = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calib.fit(X[cal], y_map[cal])

    klines = fetch_klines(args.symbol, args.timeframe, int(times[0]), int(times[-1]) + horizon_ms * 2)
    close_by_time = {int(r[0]): float(r[4]) for r in klines}

    # --- Calibration slice: scan thresholds ---
    raw_cal = model.predict_proba(X[cal])
    cal_cal = calib.predict_proba(X[cal])
    y_cal = y[cal]
    times_cal = times[cal]
    raw_pred_cal = raw_cal.argmax(axis=1) - 1
    cal_pred_cal = cal_cal.argmax(axis=1) - 1
    raw_conf_cal = raw_cal.max(axis=1)
    cal_conf_cal = cal_cal.max(axis=1)

    def scan(pred, conf, tag):
        best = None
        for thr in np.arange(0.34, 0.76, 0.01):
            tr_ = simulate(times_cal, pred, conf, thr, close_by_time, horizon_ms, fee, slip)
            m = metrics(tr_, bars_per_year)
            if m["trades"] < 50:
                continue
            score = m["total_return_pct"]
            if best is None or score > best[1]["total_return_pct"]:
                best = (round(float(thr), 2), m)
        print(f"  cal scan [{tag}]: best thr={best[0]} -> {best[1]}")
        return best

    best_raw = scan(raw_pred_cal, raw_conf_cal, "raw")
    best_cal = scan(cal_pred_cal, cal_conf_cal, "calibrated")

    # --- Test slice ---
    raw_te = model.predict_proba(X[te])
    cal_te = calib.predict_proba(X[te])
    y_te = y[te]
    times_te = times[te]
    raw_pred_te = raw_te.argmax(axis=1) - 1
    cal_pred_te = cal_te.argmax(axis=1) - 1
    y_te_map = y_map[te]

    ece_raw = ece(y_te_map, raw_te)
    ece_cal = ece(y_te_map, cal_te)
    print(f"\nECE test: raw={ece_raw:.4f} calibrated={ece_cal:.4f}")

    results = {
        "A_all_trades_raw": metrics(simulate(times_te, raw_pred_te, raw_te.max(axis=1), 0.0, close_by_time, horizon_ms, fee, slip), bars_per_year),
        "B_raw_best_thr": {"threshold": best_raw[0], **metrics(simulate(times_te, raw_pred_te, raw_te.max(axis=1), best_raw[0], close_by_time, horizon_ms, fee, slip), bars_per_year)},
        "C_calibrated_best_thr": {"threshold": best_cal[0], **metrics(simulate(times_te, cal_pred_te, cal_te.max(axis=1), best_cal[0], close_by_time, horizon_ms, fee, slip), bars_per_year)},
        "D_raw_conf_0.5_baseline": metrics(simulate(times_te, raw_pred_te, raw_te.max(axis=1), 0.5, close_by_time, horizon_ms, fee, slip), bars_per_year),
    }
    print("\n=== TEST RESULTS (honest OOS, 2025-07-08 -> now) ===")
    for k, v in results.items():
        print(f"  {k}: {v}")

    report = {
        "config": vars(args),
        "splits": {"train_end": TRAIN_END, "cal": [CAL_START, CAL_END], "test_start": TEST_START},
        "ece": {"raw": round(ece_raw, 4), "calibrated": round(ece_cal, 4)},
        "cal_scan": {"raw_best": {"threshold": best_raw[0], **best_raw[1]},
                     "calibrated_best": {"threshold": best_cal[0], **best_cal[1]}},
        "test": results,
    }
    out = f"calibration_report_{args.symbol}_{args.timeframe}_ws{args.window_size}_h{args.horizon}.json"
    Path(out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report -> {out}")


if __name__ == "__main__":
    main()
