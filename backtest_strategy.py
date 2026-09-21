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
from execution_engine import (
    ExecutionContract,
    ExecutionCostSpec,
    ExecutionMode,
    calculate_round_trip,
)

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


def simulate_trades(
    rows,
    klines,
    horizon_ms,
    model=None,
    meta=None,
    fee_bps=10.0,
    slippage_bps=5.0,
    confidence_threshold=0.0,
    side_filter=None,
    execution_mode=ExecutionMode.SPOT_REFERENCE_DERIVATIVE_SIMULATION,
    entry_delay_bars=1,
    initial_capital=10_000.0,
    capital_fraction_per_trade=1.0,
):
    """Sequentially simulate decisions with fills no earlier than a later bar.

    ``WindowEndMs`` is the opening timestamp of the finalized signal bar.  Its
    information becomes available at the next bar open, which is also the
    earliest permitted fill.  ``entry_delay_bars=1`` is therefore the baseline;
    larger values are delayed-entry stress scenarios.
    """
    if not rows or not klines:
        return []
    if entry_delay_bars < 1:
        raise ValueError("entry_delay_bars must be at least 1 (next-bar-open).")

    # Fallback to global model/meta if not passed
    m = model if model is not None else globals().get("model")
    mt = meta if meta is not None else globals().get("meta", {})

    bars_by_time = {
        int(r[0]): {
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
        }
        for r in klines
    }
    times = sorted(bars_by_time)
    time_index = {timestamp: index for index, timestamp in enumerate(times)}
    trades = []
    costs = ExecutionCostSpec(fee_bps, slippage_bps)
    mode = execution_mode if isinstance(execution_mode, ExecutionMode) else ExecutionMode(execution_mode)
    contract = ExecutionContract(
        mode=mode,
        capital_fraction_per_trade=capital_fraction_per_trade,
    )
    if mode == ExecutionMode.PERPETUAL:
        raise ValueError(
            "Perpetual backtests require an explicit historical funding series; "
            "this spot-candle runner cannot supply one."
        )
    capital = float(initial_capital)
    active_until_ms = -1

    for window_end_ms, vec, true_label, target_return in sorted(rows, key=lambda row: int(row[0])):
        if vec is None or len(vec) == 0:
            continue
        X = np.array(vec, dtype=np.float32).reshape(1, -1)

        # Model prediction
        pred = model_predict(m, X, mt)
        if confidence_threshold > 0 and pred["confidence"] < confidence_threshold:
            continue

        side = LABEL_TO_SIDE[pred["label"]]
        if side == "flat":
            continue
        if side_filter and side != side_filter:
            continue

        signal_time = int(window_end_ms)
        signal_index = time_index.get(signal_time)
        if signal_index is None:
            continue
        entry_index = signal_index + entry_delay_bars
        if entry_index >= len(times):
            continue
        entry_time = times[entry_index]
        exit_time = entry_time + horizon_ms

        if exit_time not in bars_by_time:
            continue
        if entry_time < active_until_ms:
            continue

        entry_price = bars_by_time[entry_time]["open"]
        exit_price = bars_by_time[exit_time]["open"]
        try:
            fill = calculate_round_trip(
                side=side,
                entry_reference_price=entry_price,
                exit_reference_price=exit_price,
                costs=costs,
                contract=contract,
            )
        except ValueError:
            if contract.mode == ExecutionMode.SPOT and side == "short":
                # A spot contract truthfully abstains from short signals.
                continue
            raise
        # Reserve entry fee inside the allocation so a 100%-capital strategy
        # cannot spend slightly more cash than it owns.
        position_notional = capital * capital_fraction_per_trade / (1.0 + fill.entry_fee_fraction)
        pnl_usdt = position_notional * fill.net_return
        capital_before = capital
        capital += pnl_usdt
        active_until_ms = exit_time

        trades.append({
            "signal_time": signal_time,
            "decision_time": times[signal_index + 1],
            "entry_time": entry_time,
            "exit_time": exit_time,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "entry_fill_price": fill.entry_fill_price,
            "exit_fill_price": fill.exit_fill_price,
            "gross_return": fill.gross_return,
            "net_return": fill.net_return,
            "pnl_pct": fill.net_return * 100.0,
            "position_notional": position_notional,
            "pnl_usdt": pnl_usdt,
            "capital_before": capital_before,
            "capital_after": capital,
            "confidence": pred["confidence"],
            "true_label": int(true_label),
            "target_return": float(target_return) if target_return is not None else None,
            "execution_mode": contract.mode.value,
            "fill_policy": contract.fill_policy.value,
            "price_source_market": contract.price_source_market,
            "costs": costs.to_dict(),
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
    equity_times = [trades[0]["decision_time"]]
    peak = initial_capital
    max_drawdown = 0.0
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0

    for t in trades:
        ret = t["net_return"]
        new_equity = float(t.get("capital_after", equity[-1] * (1.0 + ret)))
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

    kline_times = sorted(int(row[0]) for row in klines)
    close_by_time = {int(row[0]): float(row[4]) for row in klines}
    mtm_curve = []
    for timestamp in kline_times:
        realized = initial_capital
        for trade in trades:
            if trade["exit_time"] <= timestamp:
                realized = float(trade.get("capital_after", realized))
            else:
                break
        active = next(
            (
                trade
                for trade in trades
                if trade["entry_time"] <= timestamp < trade["exit_time"]
            ),
            None,
        )
        if active is None:
            mtm_equity = realized
        else:
            cost_values = active.get("costs", {})
            costs = ExecutionCostSpec(
                float(cost_values.get("feePerSideBps", 0.0)),
                float(cost_values.get("slippagePerSideBps", 0.0)),
            )
            contract = ExecutionContract(mode=ExecutionMode(active["execution_mode"]))
            liquidation = calculate_round_trip(
                side=active["side"],
                entry_reference_price=active["entry_price"],
                exit_reference_price=close_by_time[timestamp],
                costs=costs,
                contract=contract,
            )
            mtm_equity = float(active["capital_before"]) + float(active["position_notional"]) * liquidation.net_return
        mtm_curve.append({"time": timestamp, "equity": mtm_equity})

    if mtm_curve:
        mtm_peak = mtm_curve[0]["equity"]
        mtm_drawdown = 0.0
        for point in mtm_curve:
            mtm_peak = max(mtm_peak, point["equity"])
            if mtm_peak > 0:
                mtm_drawdown = max(mtm_drawdown, (mtm_peak - point["equity"]) / mtm_peak)
        max_drawdown = max(max_drawdown, mtm_drawdown)
        clock_returns = [
            mtm_curve[index]["equity"] / mtm_curve[index - 1]["equity"] - 1.0
            for index in range(1, len(mtm_curve))
            if mtm_curve[index - 1]["equity"] > 0
        ]
        if len(kline_times) > 1 and clock_returns:
            intervals = np.diff(kline_times)
            interval_ms = float(np.median(intervals))
            periods_per_year = (365.25 * 86_400_000.0) / interval_ms if interval_ms > 0 else 0.0
            clock_mean = float(np.mean(clock_returns))
            clock_std = float(np.std(clock_returns)) if len(clock_returns) > 1 else 0.0
            sharpe = clock_mean / clock_std * math.sqrt(periods_per_year) if clock_std > 0 else 0.0
            clock_downside = [value for value in clock_returns if value < 0]
            downside_std = float(np.std(clock_downside)) if len(clock_downside) > 1 else 0.0
            sortino = clock_mean / downside_std * math.sqrt(periods_per_year) if downside_std > 0 else 0.0
    if len(kline_times) > 1:
        active_bars = sum(
            sum(1 for ts in kline_times if t["entry_time"] <= ts < t["exit_time"])
            for t in trades
        )
        exposure = active_bars / len(kline_times)
    else:
        exposure = 0.0
    turnover = sum(2.0 * float(t.get("position_notional", initial_capital)) for t in trades) / initial_capital

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
        "exposure_fraction": exposure,
        "turnover_initial_capital_multiple": turnover,
        "equity_curve": [{"time": t, "equity": e} for t, e in zip(equity_times, equity)],
        "mark_to_market_equity_curve": mtm_curve,
    }


def run_execution_stress_scenarios(
    rows,
    klines,
    horizon_ms,
    *,
    model,
    meta,
    fee_bps,
    slippage_bps,
    confidence_threshold=0.0,
    side_filter=None,
    execution_mode=ExecutionMode.SPOT_REFERENCE_DERIVATIVE_SIMULATION,
):
    """Evaluate declared cost and one-bar delay stresses on identical signals."""
    scenarios = {}
    for multiplier in (1.0, 1.5, 2.0):
        for delay in (1, 2):
            name = f"cost_{multiplier:g}x_delay_{delay}_bar"
            scenario_trades = simulate_trades(
                rows,
                klines,
                horizon_ms,
                model=model,
                meta=meta,
                fee_bps=fee_bps * multiplier,
                slippage_bps=slippage_bps * multiplier,
                confidence_threshold=confidence_threshold,
                side_filter=side_filter,
                execution_mode=execution_mode,
                entry_delay_bars=delay,
            )
            scenarios[name] = compute_metrics(scenario_trades, klines)
    return scenarios


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
                t.get("entry_fill_price", t["entry_price"]), t.get("exit_fill_price", t["exit_price"]),
                t["gross_return"], t["net_return"],
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
    p.add_argument(
        "--execution-mode",
        choices=[mode.value for mode in ExecutionMode],
        default=ExecutionMode.SPOT_REFERENCE_DERIVATIVE_SIMULATION.value,
        help="Explicit instrument semantics. Spot rejects short signals.",
    )
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
    timeframe_ms = horizon_ms_map.get(timeframe, 3600_000)
    klines = fetch_klines(symbol, timeframe, start_ms, end_ms + horizon_ms + 2 * timeframe_ms)
    print(f"Loaded {len(klines)} klines")
    print(f"Execution: {args.execution_mode}; signal after finalized bar; fill next bar open")

    trades = simulate_trades(
        rows,
        klines,
        horizon_ms,
        model=model,
        meta=meta,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        confidence_threshold=args.confidence_threshold,
        side_filter=args.side,
        execution_mode=args.execution_mode,
    )
    print(f"Simulated {len(trades)} trades")

    metrics = compute_metrics(trades, klines)
    if not metrics:
        print("No trades executed.")
        sys.exit(0)

    print("\n=== Results ===")
    for k, v in metrics.items():
        if k != "equity_curve":
            print(f"  {k}: {v}")

    stress_scenarios = run_execution_stress_scenarios(
        rows,
        klines,
        horizon_ms,
        model=model,
        meta=meta,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        confidence_threshold=args.confidence_threshold,
        side_filter=args.side,
        execution_mode=args.execution_mode,
    )

    if args.save_db:
        run_info = {
            "symbol": symbol, "timeframe": timeframe, "window_size": window_size,
            "horizon": horizon, "model_name": model_name,
            "start_ms": start_ms, "end_ms": end_ms,
            "fee_bps": args.fee_bps, "slippage_bps": args.slippage_bps,
            "execution_mode": args.execution_mode,
            "metrics": metrics,
        }
        run_id = save_backtest_to_db(run_info, trades)
        print(f"\nSaved backtest run to DB with Id={run_id}")

    report_path = Path(f"backtest_report_{model_path.stem}.json")
    report_path.write_text(json.dumps({"run_info": run_info if args.save_db else None, "metrics": metrics, "stress_scenarios": stress_scenarios, "trades": trades}, indent=2, cls=NumpyEncoder), encoding="utf-8")
    print(f"Report written: {report_path}")
