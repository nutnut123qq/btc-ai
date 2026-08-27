#!/usr/bin/env python3
"""
Backtest a trained direction-prediction model against buy-and-hold.

Usage:
    python backtest_strategy.py --model models/BTCUSDT_1h_ws5_h1h_XGB_balanced.joblib \
        --start 2025-01-01 --end 2026-07-01 --fee-bps 10 --slippage-bps 5
"""

import argparse
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import psycopg2
from psycopg2.extensions import AsIs, register_adapter

# psycopg2 cannot adapt numpy scalars (raises `schema "np" does not exist`).
register_adapter(np.float64, lambda v: AsIs(float(v)))
register_adapter(np.float32, lambda v: AsIs(float(v)))
register_adapter(np.int64, lambda v: AsIs(int(v)))
register_adapter(np.int32, lambda v: AsIs(int(v)))
register_adapter(np.int8, lambda v: AsIs(int(v)))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

from db_config import get_db_connection
from trading_config import FEE_BPS, SLIPPAGE_BPS, DEFAULT_SYMBOL

LABEL_TO_SIDE = {1: "long", -1: "short", 0: "flat"}


def get_connection():
    return get_db_connection()


def load_model(model_path: Path):
    model = joblib.load(model_path)
    meta_path = model_path.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {}
    return model, meta


def fetch_test_windows(symbol, timeframe, window_size, horizon, start_ms, end_ms):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT "WindowEndMs", "FeatureVector", "Label", "TargetReturn"
        FROM "WindowClassificationDatasets"
        WHERE "Symbol" = %s AND "Timeframe" = %s
          AND "WindowSize" = %s AND "Horizon" = %s
          AND "WindowEndMs" >= %s AND "WindowEndMs" <= %s
        ORDER BY "WindowEndMs"
        """,
        (symbol, timeframe, window_size, horizon, start_ms, end_ms),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def fetch_klines(symbol, timeframe, start_ms, end_ms):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT "OpenTimeMs", "Open", "High", "Low", "Close", "Volume"
        FROM "Klines"
        WHERE "Symbol" = %s AND "Timeframe" = %s
          AND "OpenTimeMs" >= %s AND "OpenTimeMs" <= %s
        ORDER BY "OpenTimeMs"
        """,
        (symbol, timeframe, start_ms, end_ms),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def simulate_trades(rows, klines, horizon_ms, fee_bps=10.0, slippage_bps=5.0, confidence_threshold=0.0, side_filter=None):
    """Simulate long/short entries based on predicted label."""
    if not rows or not klines:
        return []

    close_by_time = {int(r[0]): float(r[4]) for r in klines}
    times = sorted(close_by_time.keys())
    trades = []
    fee = fee_bps / 10000.0
    slippage = slippage_bps / 10000.0

    for window_end_ms, vec, true_label, target_return in rows:
        if vec is None or len(vec) == 0:
            continue
        X = np.array(vec, dtype=np.float32).reshape(1, -1)

        # Model prediction
        pred = model_predict(model, X, meta)
        if confidence_threshold > 0 and pred["confidence"] < confidence_threshold:
            continue

        side = LABEL_TO_SIDE[pred["label"]]
        if side == "flat":
            continue
        if side_filter and side != side_filter:
            continue

        entry_time = int(window_end_ms)
        exit_time = entry_time + horizon_ms

        if entry_time not in close_by_time or exit_time not in close_by_time:
            continue

        entry_price = close_by_time[entry_time]
        exit_price = close_by_time[exit_time]

        if side == "long":
            entry_price_adj = entry_price * (1.0 + slippage)
            exit_price_adj = exit_price * (1.0 - slippage)
            gross_return = (exit_price_adj - entry_price_adj) / entry_price_adj
        else:
            entry_price_adj = entry_price * (1.0 - slippage)
            exit_price_adj = exit_price * (1.0 + slippage)
            gross_return = (entry_price_adj - exit_price_adj) / entry_price_adj

        net_return = gross_return - 2.0 * fee
        pnl_pct = net_return * 100.0

        trades.append({
            "entry_time": entry_time,
            "exit_time": exit_time,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_return": gross_return,
            "net_return": net_return,
            "pnl_pct": pnl_pct,
            "confidence": pred["confidence"],
            "true_label": int(true_label),
            "target_return": float(target_return) if target_return is not None else None,
        })

    return trades


def model_predict(model, X, meta):
    """Predict label and confidence; handles XGB label remapping if needed."""
    model_name = meta.get("model_name", "")
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        confidence = float(proba[pred_idx])
    else:
        pred_idx = int(model.predict(X)[0])
        proba = None
        confidence = 1.0

    if "XGB" in model_name:
        # Model was trained on labels {0,1,2} mapped from {-1,0,1}
        label = pred_idx - 1
        if proba is not None and len(proba) == 3:
            # proba order corresponds to mapped classes 0,1,2 -> original -1,0,1
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
    }


def compute_metrics(trades, klines, initial_capital=10000.0):
    if not trades:
        return {}

    equity = [initial_capital]
    equity_times = [trades[0]["entry_time"]]
    peak = initial_capital
    max_drawdown = 0.0
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0

    for t in trades:
        ret = t["net_return"]
        new_equity = equity[-1] * (1.0 + ret)
        equity.append(new_equity)
        equity_times.append(t["exit_time"])

        if new_equity > peak:
            peak = new_equity
        dd = (peak - new_equity) / peak
        if dd > max_drawdown:
            max_drawdown = dd

        if ret > 0:
            wins += 1
            gross_profit += ret
        else:
            losses += 1
            gross_loss += abs(ret)

    total_return = (equity[-1] - initial_capital) / initial_capital * 100.0
    win_rate = wins / len(trades) if trades else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    returns = [t["net_return"] for t in trades]
    avg_return = np.mean(returns) if returns else 0.0
    std_return = np.std(returns) if len(returns) > 1 else 0.0
    sharpe = (avg_return / std_return * math.sqrt(252)) if std_return > 0 else 0.0

    downside = [r for r in returns if r < 0]
    downside_std = np.std(downside) if len(downside) > 1 else 0.0
    sortino = (avg_return / downside_std * math.sqrt(252)) if downside_std > 0 else 0.0

    # Buy-and-hold comparison
    if klines and len(klines) >= 2:
        bh_start = float(klines[0][4])
        bh_end = float(klines[-1][4])
        buy_hold_return = (bh_end - bh_start) / bh_start * 100.0
    else:
        buy_hold_return = 0.0

    return {
        "total_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_return_pct": total_return,
        "buy_hold_return_pct": buy_hold_return,
        "excess_return_pct": total_return - buy_hold_return,
        "max_drawdown_pct": max_drawdown * 100.0,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "profit_factor": profit_factor,
        "avg_return_per_trade_pct": avg_return * 100.0,
        "final_equity": equity[-1],
        "equity_curve": [{"time": t, "equity": e} for t, e in zip(equity_times, equity)],
    }


def save_backtest_to_db(run_info, trades):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO "BacktestRuns" (
            "Symbol", "Timeframe", "WindowSize", "Horizon", "ModelName",
            "StartTimeMs", "EndTimeMs", "FeeBps", "SlippageBps",
            "TotalTrades", "WinRate", "TotalReturnPct", "BuyHoldReturnPct",
            "MaxDrawdownPct", "SharpeRatio", "SortinoRatio", "ProfitFactor",
            "FinalEquity", "MetricsJson", "EquityCurveJson", "CreatedAtUtc"
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        RETURNING "Id"
        """,
        (
            run_info["symbol"], run_info["timeframe"], run_info["window_size"], run_info["horizon"], run_info["model_name"],
            run_info["start_ms"], run_info["end_ms"], run_info["fee_bps"], run_info["slippage_bps"],
            run_info["metrics"]["total_trades"], run_info["metrics"]["win_rate"], run_info["metrics"]["total_return_pct"],
            run_info["metrics"]["buy_hold_return_pct"], run_info["metrics"]["max_drawdown_pct"],
            run_info["metrics"]["sharpe_ratio"], run_info["metrics"]["sortino_ratio"], run_info["metrics"]["profit_factor"],
            run_info["metrics"]["final_equity"], json.dumps(run_info["metrics"]), json.dumps(run_info["metrics"]["equity_curve"]),
            datetime.now(timezone.utc),
        ),
    )
    run_id = cur.fetchone()[0]

    for t in trades:
        cur.execute(
            """
            INSERT INTO "BacktestTrades" (
                "BacktestRunId", "EntryTimeMs", "ExitTimeMs", "Side",
                "EntryPrice", "ExitPrice", "GrossReturn", "NetReturn",
                "PnlPct", "Confidence", "TrueLabel", "TargetReturn"
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                run_id, t["entry_time"], t["exit_time"], t["side"],
                t["entry_price"], t["exit_price"], t["gross_return"], t["net_return"],
                t["pnl_pct"], t["confidence"], t["true_label"], t["target_return"],
            ),
        )

    conn.commit()
    cur.close()
    conn.close()
    return run_id


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to .joblib model")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD (UTC)")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD (UTC)")
    p.add_argument("--fee-bps", type=float, default=FEE_BPS)
    p.add_argument("--slippage-bps", type=float, default=SLIPPAGE_BPS)
    p.add_argument("--confidence-threshold", type=float, default=0.0)
    p.add_argument("--side", choices=["long", "short"], default=None, help="Only trade one side")
    p.add_argument("--exit-horizon", choices=["1h", "4h", "1d"], default=None, help="Override holding period (default: model horizon)")
    p.add_argument("--save-db", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    model, meta = load_model(model_path)
    symbol = meta.get("symbol", "BTCUSDT")
    timeframe = meta.get("timeframe", "1h")
    window_size = meta.get("window_size", 5)
    horizon = meta.get("horizon", "1h")
    model_name = meta.get("model_name", model_path.stem)

    start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    horizon_ms_map = {"1h": 3600_000, "4h": 14_400_000, "1d": 86_400_000}
    exit_horizon = args.exit_horizon or horizon
    horizon_ms = horizon_ms_map.get(exit_horizon, 3600_000)

    print(f"Backtest {model_name} on {symbol} {timeframe} ws={window_size} h={horizon}")
    print(f"Period: {args.start} -> {args.end}")

    rows = fetch_test_windows(symbol, timeframe, window_size, horizon, start_ms, end_ms)
    print(f"Loaded {len(rows)} test windows")

    if not rows:
        print("No data. Abort.")
        sys.exit(1)

    # Extend kline range to cover exit times
    klines = fetch_klines(symbol, timeframe, start_ms, end_ms + horizon_ms)
    print(f"Loaded {len(klines)} klines")

    trades = simulate_trades(rows, klines, horizon_ms, args.fee_bps, args.slippage_bps, args.confidence_threshold, args.side)
    print(f"Simulated {len(trades)} trades")

    metrics = compute_metrics(trades, klines)
    if not metrics:
        print("No trades executed.")
        sys.exit(0)

    print("\n=== Results ===")
    for k, v in metrics.items():
        if k != "equity_curve":
            print(f"  {k}: {v}")

    if args.save_db:
        run_info = {
            "symbol": symbol, "timeframe": timeframe, "window_size": window_size,
            "horizon": horizon, "model_name": model_name,
            "start_ms": start_ms, "end_ms": end_ms,
            "fee_bps": args.fee_bps, "slippage_bps": args.slippage_bps,
            "metrics": metrics,
        }
        run_id = save_backtest_to_db(run_info, trades)
        print(f"\nSaved backtest run to DB with Id={run_id}")

    report_path = Path(f"backtest_report_{model_path.stem}.json")
    report_path.write_text(json.dumps({"run_info": run_info if args.save_db else None, "metrics": metrics, "trades": trades}, indent=2, cls=NumpyEncoder), encoding="utf-8")
    print(f"Report written: {report_path}")
