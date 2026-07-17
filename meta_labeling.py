#!/usr/bin/env python3
"""
D2: Meta-labeling — secondary model quyết định ACT/PASS trên tín hiệu primary.

Primary: XGB 3-class (giống production), train < 2024-07.
Meta:    XGB binary, target = 1 nếu primary đúng (và không flat), 0 nếu sai.
         Features = futures metrics (OI, funding, L/S, taker) + regime (vol, trend)
         tại thởi điểm window end — KHÔNG dùng lại 35 features của primary
         (de Prado: meta-labeling cần thông tin MỚI).

Split (purged 2 bars ở ranh giới):
  primary train:  < 2024-07-01
  meta train:     2024-07-08 .. 2025-07-01 (chỉ rows primary có signal)
  test (honest):  2025-07-08 .. now

So sánh trên test:
  A. primary all signals
  B. primary raw conf >= 0.52 (best raw từ D1)
  C. primary + meta filter (threshold scan trên meta-train)
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg2
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from backtest_strategy import fetch_klines
from calibrate_threshold import fetch_windows, ms, TRAIN_END, CAL_START, CAL_END, TEST_START, DB, LABEL_REMAP

FUTURES_FEATURE_NAMES = [
    "funding_rate", "funding_z", "oi_d1h", "oi_d4h", "oi_d1d",
    "global_ls", "top_ls_count", "top_ls_sum", "taker_ratio", "taker_ma24",
    "rv20", "dist_sma200", "ret_7d",
]


def fetch_futures_metrics(symbol):
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute(
        """SELECT "OpenTimeMs", "OpenInterest", "GlobalLsRatio",
                  "TopTraderLsCountRatio", "TopTraderLsSumRatio", "TakerBuySellVolRatio",
                  "FundingRate"
           FROM "FuturesMetrics" WHERE "Symbol"=%s ORDER BY "OpenTimeMs" """,
        (symbol,))
    rows = cur.fetchall()
    conn.close()
    arr = {
        "ts": np.array([r[0] for r in rows], dtype=np.int64),
        "oi": np.array([r[1] if r[1] is not None else np.nan for r in rows]),
        "gls": np.array([r[2] if r[2] is not None else np.nan for r in rows]),
        "tls_c": np.array([r[3] if r[3] is not None else np.nan for r in rows]),
        "tls_s": np.array([r[4] if r[4] is not None else np.nan for r in rows]),
        "taker": np.array([r[5] if r[5] is not None else np.nan for r in rows]),
        "fund": np.array([r[6] if r[6] is not None else np.nan for r in rows]),
    }
    return arr


def last_valid(series, idx):
    """Giá trị non-NaN gần nhất tại hoặc trước idx (quay lui tối đa 300 rows)."""
    j = idx
    while j >= 0 and j >= idx - 300 and np.isnan(series[j]):
        j -= 1
    return series[j] if j >= 0 and not np.isnan(series[j]) else np.nan


def pct_change_back(series, idx, back):
    """% thay đổi so với giá trị non-NaN gần nhất cách idx `back` steps."""
    now = last_valid(series, idx)
    prev = last_valid(series, idx - back)
    if np.isnan(now) or np.isnan(prev) or prev == 0:
        return np.nan
    return (now - prev) / prev * 100.0


def mean_back(series, idx, back):
    vals = series[max(0, idx - back + 1): idx + 1]
    vals = vals[~np.isnan(vals)]
    return float(vals.mean()) if len(vals) else np.nan


def build_meta_features(win_times, fut, klines):
    """Ma trận (n_windows, 13) futures + regime features tại mỗi window end."""
    fut_ts = fut["ts"]
    k_ts = np.array([int(r[0]) for r in klines], dtype=np.int64)
    k_close = np.array([float(r[4]) for r in klines])

    n = len(win_times)
    F = np.full((n, len(FUTURES_FEATURE_NAMES)), np.nan, dtype=np.float64)
    fund_hist = []  # funding rate history for z-score
    fund_ts = fut_ts[~np.isnan(fut["fund"])]
    fund_vals = fut["fund"][~np.isnan(fut["fund"])]

    for i, t in enumerate(win_times):
        j = np.searchsorted(fut_ts, t, side="right") - 1
        if j < 0:
            continue
        # funding: last funding event <= t, z-score vs last 90 events
        fj = np.searchsorted(fund_ts, t, side="right") - 1
        if fj >= 0:
            F[i, 0] = fund_vals[fj]
            lo = max(0, fj - 90)
            hist = fund_vals[lo:fj]
            if len(hist) >= 30 and hist.std() > 0:
                F[i, 1] = (fund_vals[fj] - hist.mean()) / hist.std()
        F[i, 2] = pct_change_back(fut["oi"], j, 12)     # 1h
        F[i, 3] = pct_change_back(fut["oi"], j, 48)     # 4h
        F[i, 4] = pct_change_back(fut["oi"], j, 288)    # 1d
        F[i, 5] = last_valid(fut["gls"], j)
        F[i, 6] = last_valid(fut["tls_c"], j)
        F[i, 7] = last_valid(fut["tls_s"], j)
        F[i, 8] = last_valid(fut["taker"], j)
        F[i, 9] = mean_back(fut["taker"], j, 288)

        # regime từ klines 4h
        kj = np.searchsorted(k_ts, t, side="right") - 1
        if kj >= 200:
            rets = np.diff(np.log(k_close[kj - 20: kj + 1]))
            F[i, 10] = rets.std() * 100
            sma = k_close[kj - 199: kj + 1].mean()
            F[i, 11] = (k_close[kj] - sma) / sma * 100
        if kj >= 42:
            F[i, 12] = (k_close[kj] / k_close[kj - 42] - 1) * 100
    return F


def simulate(times_i, labels, keep_mask, close_by_time, horizon_ms, fee, slip):
    trades = []
    for t, lab, k in zip(times_i, labels, keep_mask):
        if not k or lab == 0:
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
    return {"trades": len(trades), "win_rate": round(float((trades > 0).mean()), 4),
            "total_return_pct": round((eq - 1) * 100, 2),
            "avg_trade_pct": round(float(trades.mean()) * 100, 4),
            "sharpe": round(sharpe, 3)}


def main():
    symbol, timeframe, ws, horizon = "BTCUSDT", "4h", 5, "4h"
    horizon_ms = 14_400_000
    bars_per_year = 365.25 * 24 * 3600 * 1000 / horizon_ms
    fee, slip = 10 / 1e4, 5 / 1e4

    times, X, y = fetch_windows(symbol, timeframe, ws, horizon)
    tr = times < ms(TRAIN_END)
    mt = (times >= ms(CAL_START)) & (times < ms(CAL_END))
    te = times >= ms(TEST_START)
    print(f"windows={len(times)} primary_train={tr.sum()} meta_train={mt.sum()} test={te.sum()}")

    # --- Primary model ---
    y_map = np.array([LABEL_REMAP[v] for v in y])
    primary = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                eval_metric="mlogloss", n_jobs=-1, random_state=42)
    primary.fit(X[tr], y_map[tr])

    proba = primary.predict_proba(X)
    pred = proba.argmax(axis=1) - 1
    conf = proba.max(axis=1)

    # --- Meta features ---
    print("Building meta features (futures + regime)...", flush=True)
    fut = fetch_futures_metrics(symbol)
    klines = fetch_klines(symbol, timeframe, int(times[0]), int(times[-1]) + horizon_ms * 2)
    close_by_time = {int(r[0]): float(r[4]) for r in klines}
    F = build_meta_features(times, fut, klines)
    valid = ~np.isnan(F).any(axis=1)
    print(f"meta features valid: {valid.sum()}/{len(times)}")

    # --- Meta labels: chỉ trên rows primary có signal (pred != 0) ---
    meta_label = (pred == y).astype(np.int8)  # primary đúng = 1
    sig_mt = mt & valid & (pred != 0)
    print(f"meta train samples: {sig_mt.sum()} (positive rate {meta_label[sig_mt].mean():.3f})")

    meta = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                             eval_metric="logloss", n_jobs=-1, random_state=42)
    meta.fit(F[sig_mt], meta_label[sig_mt])
    meta_prob = meta.predict_proba(F)[:, 1]

    # --- Threshold scan trên meta-train slice ---
    best = None
    for thr in np.arange(0.30, 0.85, 0.01):
        keep = (meta_prob >= thr) & (pred != 0)
        trades = simulate(times[mt], pred[mt], keep[mt], close_by_time, horizon_ms, fee, slip)
        m = metrics(trades, bars_per_year)
        if m["trades"] >= 50 and (best is None or m["total_return_pct"] > best[1]["total_return_pct"]):
            best = (round(float(thr), 2), m)
    print(f"meta-train scan: best thr={best[0]} -> {best[1]}")

    # --- Honest test ---
    sig_te = te & valid
    keep_all = (pred != 0)
    keep_raw = (pred != 0) & (conf >= 0.52)
    keep_meta = (pred != 0) & (meta_prob >= best[0])

    results = {
        "A_primary_all": metrics(simulate(times[sig_te], pred[sig_te], keep_all[sig_te], close_by_time, horizon_ms, fee, slip), bars_per_year),
        "B_raw_conf_0.52": metrics(simulate(times[sig_te], pred[sig_te], keep_raw[sig_te], close_by_time, horizon_ms, fee, slip), bars_per_year),
        "C_meta_filter": {"threshold": best[0], **metrics(simulate(times[sig_te], pred[sig_te], keep_meta[sig_te], close_by_time, horizon_ms, fee, slip), bars_per_year)},
    }
    print("\n=== TEST (honest OOS 2025-07-08 -> now) ===")
    for k, v in results.items():
        print(f"  {k}: {v}")

    # Feature importance của meta model
    imp = meta.feature_importances_
    order = np.argsort(imp)[::-1]
    print("\nMeta feature importances:")
    for idx in order:
        print(f"  {FUTURES_FEATURE_NAMES[idx]:15s} {imp[idx]:.4f}")

    report = {"splits": {"primary_train_end": TRAIN_END, "meta_train": [CAL_START, CAL_END], "test_start": TEST_START},
              "meta_train_samples": int(sig_mt.sum()),
              "meta_positive_rate": round(float(meta_label[sig_mt].mean()), 4),
              "meta_threshold": best[0],
              "test": results,
              "feature_importance": {FUTURES_FEATURE_NAMES[i]: round(float(imp[i]), 4) for i in order}}
    out = f"meta_labeling_report_{symbol}_{timeframe}_ws{ws}_h{horizon}.json"
    Path(out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report -> {out}")


if __name__ == "__main__":
    main()
