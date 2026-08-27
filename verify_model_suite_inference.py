#!/usr/bin/env python3
"""
Multi-Asset ML Model Suite Inference Verification & Acceptance Benchmark Matrix
===============================================================================
Performs direct inference tests on latest features for BTCUSDT, ETHUSDT, SOLUSDT
across 1h, 4h, 1d timeframes.
Verifies probability normalization (sum == 1.0) and model version alignment.
Prints the complete benchmark matrix.
"""

import sys
import json
from pathlib import Path
import psycopg2
import numpy as np
from prediction_service import load_model, predict_from_vector
from db_config import get_db_params

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

MODELS_DIR = Path(__file__).parent / "models"


def get_latest_window_feature(symbol, timeframe, ws, horizon):
    conn = psycopg2.connect(**get_db_params())
    cur = conn.cursor()
    cur.execute('''
        SELECT "FeatureVector", "WindowEndMs"
        FROM "WindowClassificationDatasets"
        WHERE "Symbol"=%s AND "Timeframe"=%s AND "WindowSize"=%s AND "Horizon"=%s
          AND "FeatureVector" IS NOT NULL
        ORDER BY "WindowEndMs" DESC
        LIMIT 1
    ''', (symbol, timeframe, ws, horizon))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None, None
    return list(row[0]), int(row[1])


def main():
    print("=" * 115)
    print("           MULTI-ASSET REALTIME INFERENCE HEALTHCHECK & MODEL MATRIX AUDIT")
    print("=" * 115)

    test_matrix = [
        ("BTCUSDT", "1h", 5, "1h", "XGB"),
        ("BTCUSDT", "1h", 10, "1h", "XGB"),
        ("BTCUSDT", "1h", 5, "4h", "XGB"),
        ("BTCUSDT", "4h", 5, "4h", "XGB"),
        ("BTCUSDT", "4h", 5, "1d", "XGB"),
        ("BTCUSDT", "1d", 5, "1d", "XGB"),

        ("ETHUSDT", "1h", 5, "1h", "XGB"),
        ("ETHUSDT", "1h", 10, "1h", "XGB"),
        ("ETHUSDT", "1h", 5, "4h", "XGB"),
        ("ETHUSDT", "4h", 5, "4h", "XGB"),
        ("ETHUSDT", "4h", 5, "1d", "XGB"),
        ("ETHUSDT", "1d", 5, "1d", "XGB"),

        ("SOLUSDT", "1h", 5, "1h", "XGB"),
        ("SOLUSDT", "1h", 10, "1h", "XGB"),
        ("SOLUSDT", "1h", 5, "4h", "XGB"),
        ("SOLUSDT", "4h", 5, "4h", "XGB"),
        ("SOLUSDT", "4h", 5, "1d", "XGB"),
        ("SOLUSDT", "1d", 5, "1d", "XGB"),
    ]

    benchmark_rows = []
    print(f"{'Symbol':<10} | {'TF':<4} | {'WS':<3} | {'H':<4} | {'Model Version':<38} | {'Prob(D/S/U)':<24} | {'Sum(P)':<7} | {'Status'}")
    print("-" * 115)

    for sym, tf, ws, h, alg in test_matrix:
        feat_vec, end_ms = get_latest_window_feature(sym, tf, ws, h)
        if not feat_vec:
            print(f"{sym:<10} | {tf:<4} | {ws:<3} | {h:<4} | NO DATA IN WindowClassificationDatasets")
            continue

        try:
            res = predict_from_vector(feat_vec, sym, tf, ws, h)
            p_down = res["prob_down"]
            p_side = res["prob_sideways"]
            p_up = res["prob_up"]
            p_sum = p_down + p_side + p_up
            mv = res["model_version"]

            # Verify no fallback: mv must contain sym
            is_valid_sym = sym in mv
            is_valid_sum = abs(p_sum - 1.0) < 1e-4

            status = "PASS (Valid)" if (is_valid_sym and is_valid_sum) else "FAIL"
            p_str = f"{p_down:.2f} / {p_side:.2f} / {p_up:.2f}"

            print(f"{sym:<10} | {tf:<4} | {ws:<3} | {h:<4} | {mv:<38} | {p_str:<24} | {p_sum:.4f}  | {status}")

            # Read json metadata for benchmark matrix
            meta_file = MODELS_DIR / f"{mv}.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                metrics = meta.get("metrics", {})
                benchmark_rows.append({
                    "symbol": sym,
                    "tf": tf,
                    "ws": ws,
                    "horizon": h,
                    "algo": meta.get("algorithm", "XGBoost") + " (Isotonic)",
                    "oos_acc": f"{metrics.get('oos_accuracy', 0)*100:.1f}%",
                    "brier": f"{metrics.get('brier_score', 0):.4f}",
                    "ece": f"{metrics.get('expected_calibration_error', 0):.4f}",
                    "oos_wr": f"{metrics.get('oos_win_rate', 0)*100:.1f}%" if metrics.get("oos_trades_count", 0) > 0 else "N/A (Hold)",
                    "threshold": meta.get("optimal_threshold", 0.5),
                    "file": f"{mv}.joblib"
                })

        except Exception as e:
            print(f"{sym:<10} | {tf:<4} | {ws:<3} | {h:<4} | ERROR: {e}")

    print("=" * 115)
    print("\n")
    print("=" * 125)
    print("                   MULTI-ASSET ML MODEL SUITE & BENCHMARK ACCEPTANCE MATRIX")
    print("=" * 125)
    print(f"| {'Symbol':<8} | {'Khung (TF)':<10} | {'Horizon':<8} | {'Thuật Toán':<20} | {'OOS Accuracy':<13} | {'Brier Score':<12} | {'ECE':<8} | {'OOS Win Rate':<13} | {'Model File':<38} |")
    print("|" + "-"*10 + "|" + "-"*12 + "|" + "-"*10 + "|" + "-"*22 + "|" + "-"*15 + "|" + "-"*14 + "|" + "-"*10 + "|" + "-"*15 + "|" + "-"*40 + "|")

    for r in benchmark_rows:
        tf_ws = f"{r['tf']} (ws={r['ws']})"
        print(f"| {r['symbol']:<8} | {tf_ws:<10} | {r['horizon']:<8} | {r['algo']:<20} | {r['oos_acc']:<13} | {r['brier']:<12} | {r['ece']:<8} | {r['oos_wr']:<13} | {r['file']:<38} |")

    print("=" * 125)


if __name__ == "__main__":
    main()
