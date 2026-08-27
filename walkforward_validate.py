#!/usr/bin/env python3
"""
Walk-forward retrain validation cho model direction (mặc định 4h ws5 h4h).

Mục đích: kiểm chứng kết quả backtest single-split có bền không khi phải
retrain định kỳ trên dữ liệu quá khứ và trade OOS liên tục (chống
multiple-testing bias / backtest overfitting).

Thiết kế:
  - Expanding window: train = tất cả dữ liệu trước fold_start - purge_gap.
  - Purge gap = purge_bars bars (mặc định 2 bars của timeframe) — tránh leak
    do triple-barrier label là interval [t, t1] chồng lấn ranh giới fold.
  - Mỗi fold: train XGB (params giống train_baseline_advanced), predict OOS,
    simulate trades (reuse backtest_strategy.simulate_trades) với fee/slippage
    + confidence threshold, tính metrics.
  - Aggregate: per-fold Sharpe/return/win-rate, stitched equity, DSR-lite
    (mean/std Sharpe, % folds dương), so với buy&hold.

Usage:
  python walkforward_validate.py --timeframe 4h --window-size 5 --horizon 4h \
      --fold-months 6 --confidence-threshold 0.5
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

sys.path.insert(0, str(Path(__file__).parent))
from backtest_strategy import fetch_klines
from db_config import get_db_connection

LABEL_REMAP = {-1: 0, 0: 1, 1: 2}
LABEL_INV = {0: -1, 1: 0, 2: 1}


def get_connection():
    return get_db_connection()


def fetch_all_windows(symbol, timeframe, window_size, horizon):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT "WindowEndMs", "FeatureVector", "Label"
        FROM "WindowClassificationDatasets"
        WHERE "Symbol" = %s AND "Timeframe" = %s
          AND "WindowSize" = %s AND "Horizon" = %s
          AND "FeatureVector" IS NOT NULL AND "Label" IS NOT NULL
        ORDER BY "WindowEndMs"
        """,
        (symbol, timeframe, window_size, horizon),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    times = np.array([int(r[0]) for r in rows], dtype=np.int64)
    X = np.array([list(r[1]) for r in rows], dtype=np.float32)
    y = np.array([int(r[2]) for r in rows], dtype=np.int8)
    return times, X, y


def make_folds(times, fold_months, min_train_months, purge_ms):
    """Sinh các fold (train_end_ms, test_start_ms, test_end_ms) theo mốc tháng UTC."""
    t_min = datetime.fromtimestamp(times[0] / 1000, timezone.utc)
    t_max = datetime.fromtimestamp(times[-1] / 1000, timezone.utc)
    # Mốc đầu mỗi tháng UTC
    boundaries = []
    y, m = t_min.year, t_min.month
    while True:
        b = datetime(y, m, 1, tzinfo=timezone.utc)
        if b > t_max:
            break
        boundaries.append(int(b.timestamp() * 1000))
        m += 1
        if m > 12:
            y, m = y + 1, 1
    folds = []
    fold_ms = fold_months * 30.44 * 24 * 3600 * 1000
    min_train_ms = min_train_months * 30.44 * 24 * 3600 * 1000
    for i in range(0, len(boundaries) - fold_months, fold_months):
        test_start = boundaries[i]
        test_end = boundaries[min(i + fold_months, len(boundaries) - 1)]
        train_end = test_start - purge_ms
        if train_end - times[0] < min_train_ms:
            continue
        if test_end - test_start < fold_ms * 0.5:
            continue
        folds.append((train_end, test_start, test_end))
    return folds


def annualized_sharpe(trades, bars_per_year):
    rets = np.array([t["net_return"] for t in trades], dtype=np.float64)
    if len(rets) < 2 or rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(bars_per_year))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--timeframe", default="4h")
    p.add_argument("--window-size", type=int, default=5)
    p.add_argument("--horizon", default="4h")
    p.add_argument("--fold-months", type=int, default=6)
    p.add_argument("--min-train-months", type=int, default=18)
    p.add_argument("--purge-bars", type=int, default=2)
    p.add_argument("--confidence-threshold", type=float, default=0.5)
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    tf_ms_map = {"15m": 900_000, "30m": 1_800_000, "1h": 3_600_000,
                 "4h": 14_400_000, "1d": 86_400_000}
    horizon_ms_map = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
    tf_ms = tf_ms_map[args.timeframe]
    horizon_ms = horizon_ms_map[args.horizon]
    purge_ms = args.purge_bars * tf_ms
    bars_per_year = 365.25 * 24 * 3600 * 1000 / horizon_ms

    print(f"Walk-forward: {args.symbol} {args.timeframe} ws={args.window_size} h={args.horizon}")
    print(f"Folds: {args.fold_months}mo, min train {args.min_train_months}mo, purge {args.purge_bars} bars, conf>={args.confidence_threshold}")

    times, X, y = fetch_all_windows(args.symbol, args.timeframe, args.window_size, args.horizon)
    print(f"Loaded {len(times)} windows: {datetime.fromtimestamp(times[0]/1000, timezone.utc):%Y-%m-%d} -> {datetime.fromtimestamp(times[-1]/1000, timezone.utc):%Y-%m-%d}")

    folds = make_folds(times, args.fold_months, args.min_train_months, purge_ms)
    print(f"{len(folds)} folds")

    klines = fetch_klines(args.symbol, args.timeframe, int(times[0]), int(times[-1]) + horizon_ms * 2)
    close_by_time = {int(r[0]): float(r[4]) for r in klines}

    fold_results = []
    all_trades = []
    for fi, (train_end, test_start, test_end) in enumerate(folds):
        tr_mask = times < train_end
        te_mask = (times >= test_start) & (times < test_end)
        n_tr, n_te = int(tr_mask.sum()), int(te_mask.sum())
        if n_tr < 500 or n_te < 30:
            continue

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            eval_metric="mlogloss", n_jobs=-1, random_state=42,
        )
        model.fit(X[tr_mask], np.array([LABEL_REMAP[v] for v in y[tr_mask]]))
        proba = model.predict_proba(X[te_mask])
        pred = np.array([LABEL_INV[i] for i in proba.argmax(axis=1)], dtype=np.int8)
        conf = proba.max(axis=1)

        acc = float((pred == y[te_mask]).mean())

        # Simulate trades trực tiếp từ pred/conf của fold (không qua model global)
        te_times = times[te_mask]
        fee = args.fee_bps / 10000.0
        slip = args.slippage_bps / 10000.0
        trades = []
        for i in range(n_te):
            if conf[i] < args.confidence_threshold or pred[i] == 0:
                continue
            entry_time = int(te_times[i])
            exit_time = entry_time + horizon_ms
            if entry_time not in close_by_time or exit_time not in close_by_time:
                continue
            ep = close_by_time[entry_time]
            xp = close_by_time[exit_time]
            if pred[i] == 1:  # long
                gross = (xp * (1 - slip) - ep * (1 + slip)) / (ep * (1 + slip))
            else:  # short
                gross = (ep * (1 - slip) - xp * (1 + slip)) / (ep * (1 + slip))
            trades.append({
                "entry_time": entry_time, "exit_time": exit_time,
                "side": "long" if pred[i] == 1 else "short",
                "net_return": gross - 2.0 * fee,
            })
        d0 = datetime.fromtimestamp(test_start / 1000, timezone.utc)
        d1 = datetime.fromtimestamp(test_end / 1000, timezone.utc)
        res = {
            "fold": fi,
            "test_start": f"{d0:%Y-%m-%d}", "test_end": f"{d1:%Y-%m-%d}",
            "n_train": n_tr, "n_test": n_te,
            "acc": round(acc, 4),
            "trades": len(trades),
        }
        if trades:
            wins = sum(1 for t in trades if t["net_return"] > 0)
            eq = 1.0
            for t in trades:
                eq *= 1 + t["net_return"]
            res.update({
                "win_rate": round(wins / len(trades), 4),
                "total_return_pct": round((eq - 1) * 100, 2),
                "avg_trade_pct": round(float(np.mean([t["net_return"] for t in trades])) * 100, 4),
                "sharpe": round(annualized_sharpe(trades, bars_per_year), 3),
            })
            all_trades.extend(trades)
        fold_results.append(res)
        print(f"  fold {fi}: {d0:%Y-%m-%d}->{d1:%Y-%m-%d} train={n_tr} acc={acc:.3f} "
              f"trades={res['trades']} ret={res.get('total_return_pct', 0):+.1f}% sharpe={res.get('sharpe', 0):.2f}", flush=True)

    # Aggregate
    sharpes = [f["sharpe"] for f in fold_results if "sharpe" in f]
    rets = [f["total_return_pct"] for f in fold_results if "total_return_pct" in f]
    eq = 1.0
    for t in sorted(all_trades, key=lambda t: t["entry_time"]):
        eq *= 1 + t["net_return"]
    wins = sum(1 for t in all_trades if t["net_return"] > 0)
    agg = {
        "folds": len(fold_results),
        "folds_profitable": len([r for r in rets if r > 0]),
        "mean_fold_sharpe": round(float(np.mean(sharpes)), 3) if sharpes else 0,
        "std_fold_sharpe": round(float(np.std(sharpes)), 3) if sharpes else 0,
        "total_trades": len(all_trades),
        "overall_win_rate": round(wins / len(all_trades), 4) if all_trades else 0,
        "stitched_total_return_pct": round((eq - 1) * 100, 2),
    }
    print("\n=== AGGREGATE ===")
    for k, v in agg.items():
        print(f"  {k}: {v}")

    report = {"config": vars(args), "aggregate": agg, "folds": fold_results}
    out = args.out or f"walkforward_report_{args.symbol}_{args.timeframe}_ws{args.window_size}_h{args.horizon}.json"
    Path(out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report -> {out}")


if __name__ == "__main__":
    main()
