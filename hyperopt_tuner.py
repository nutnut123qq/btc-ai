#!/usr/bin/env python3
"""
Bayesian Hyperparameter Optimization via Optuna (Freqtrade-style Hyperopt)
========================================================================
Performs automated hyperparameter tuning for trading threshold, ATR Take-Profit,
ATR Stop-Loss, and Max-Hold-Bars across BTCUSDT, ETHUSDT, SOLUSDT on 4h timeframe.

Objective Function:
    Maximize: Sharpe_Ratio * sqrt(Profit_Factor) - 2.0 * max(0, MDD - 0.10)
    Subject to: Min 20 trades to prevent overfitting to sparse outliers.
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any

import numpy as np
import psycopg2
import joblib
import optuna
from optuna.samplers import TPESampler

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

optuna.logging.set_verbosity(optuna.logging.WARNING)

from db_config import get_db_params, get_db_connection
from trading_config import (
    FEE_BPS,
    SLIPPAGE_BPS,
    INITIAL_BALANCE_USDT,
    TRAILING_STOP_ATR_TRIGGER,
    TRAILING_STOP_ATR_DIST,
    MAX_KELLY_FRACTION,
    MIN_KELLY_FRACTION,
    KELLY_SAFETY_FACTOR,
)

MODELS_DIR = Path(__file__).parent / "models"
OUTPUT_FILE = Path(__file__).parent / "optuna_best_params.json"

FEATURE_COLS = [
    "CloseZscore", "ClosePctChange1", "ClosePctChange4", "ClosePctChange24",
    "HighLowRangePct", "BodyPct", "UpperWickPct", "LowerWickPct",
    "Rsi14", "Rsi14Slope", "MacdNorm", "MacdSignalNorm", "MacdHistogramNorm",
    "Ema12Dist", "Ema26Dist", "Ema50Dist", "Ema200Dist", "Sma50Dist", "Sma200Dist",
    "BollingerWidth", "BollingerPosition", "Atr14Pct", "ObvEmaDist", "VwapDist",
    "RollingVwapDist", "VolumeZscore", "VolumeSma20Ratio", "TakerBuyRatio",
    "RecentPatternEncoded", "ActiveRuleCount",
]


def time_features(open_ms: int) -> list[float]:
    dt = datetime.fromtimestamp(open_ms / 1000, timezone.utc)
    hour = dt.hour + dt.minute / 60.0
    dow = dt.weekday()
    return [
        math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
        math.sin(2 * math.pi * dow / 7), math.cos(2 * math.pi * dow / 7),
        1.0 if dow >= 5 else 0.0,
    ]


def load_dataset_for_symbol(symbol: str, timeframe: str = "4h") -> Tuple[List[dict], Any]:
    """Preloads klines, ATR, and pre-computed model probabilities for rapid simulation."""
    conn = get_db_connection()
    cur = conn.cursor()

    model_file = MODELS_DIR / f"{symbol}_{timeframe}_ws5_h4h_XGB_calibrated.joblib"
    if not model_file.exists():
        conn.close()
        raise FileNotFoundError(f"Model file not found: {model_file}")

    model = joblib.load(model_file)

    # 1. Load Klines
    cur.execute("""
        SELECT "OpenTimeMs", "Open", "High", "Low", "Close", "Volume"
        FROM "Klines"
        WHERE "Symbol" = %s AND "Timeframe" = %s
        ORDER BY "OpenTimeMs"
    """, (symbol, timeframe))
    klines_raw = cur.fetchall()

    if not klines_raw:
        conn.close()
        raise ValueError(f"No klines found for {symbol} {timeframe}")

    # 2. Load ML Feature Stores
    cols = ", ".join(f'"{c}"' for c in FEATURE_COLS)
    cur.execute(f"""
        SELECT "OpenTimeMs", {cols}
        FROM "MlFeatureStores"
        WHERE "Symbol" = %s AND "Timeframe" = %s
        ORDER BY "OpenTimeMs"
    """, (symbol, timeframe))
    feat_raw = cur.fetchall()
    conn.close()

    feat_by_time = {r[0]: r[1:] for r in feat_raw}
    tf_ms = 14_400_000

    bars = []
    tr_window = []

    for i in range(len(klines_raw)):
        open_ms, o, h, l, c, v = klines_raw[i]
        o, h, l, c, v = float(o), float(h), float(l), float(c), float(v)

        # Compute ATR(14)
        if i > 0:
            prev_c = float(klines_raw[i - 1][4])
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        else:
            tr = h - l
        tr_window.append(tr)
        if len(tr_window) > 14:
            tr_window.pop(0)
        atr14 = sum(tr_window) / len(tr_window) if tr_window else (h - l)

        # Build 5-bar vector if history exists
        proba = None
        if i >= 4:
            prev_times = [klines_raw[i - k][0] for k in range(4, -1, -1)]
            is_contig = all(prev_times[k] - prev_times[k - 1] == tf_ms for k in range(1, 5))
            if is_contig:
                vec = []
                valid = True
                for t in prev_times:
                    if t not in feat_by_time:
                        valid = False
                        break
                    vals = feat_by_time[t]
                    if any(v is None for v in vals[:28]):
                        valid = False
                        break
                    vec.extend(float(x) if x is not None else 0.0 for x in vals)
                    vec.extend(time_features(t))
                if valid:
                    proba = model.predict_proba(np.array(vec, dtype=np.float32).reshape(1, -1))[0]

        bars.append({
            "time_ms": open_ms,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "atr": atr14,
            "proba": proba,  # [P_down, P_sideways, P_up]
        })

    return bars, model


def simulate_strategy(
    bars: List[dict],
    threshold: float,
    tp_mult: float,
    sl_mult: float,
    max_hold_bars: int,
) -> Dict[str, Any]:
    """Runs high-speed backtest simulation of Quarter-Kelly & ATR Trailing Stop."""
    fee = FEE_BPS / 1e4
    slip = SLIPPAGE_BPS / 1e4

    balance = INITIAL_BALANCE_USDT
    equity_curve = [balance]
    trades = []
    open_trade = None

    for i, bar in enumerate(bars):
        h, l, c = bar["high"], bar["low"], bar["close"]
        atr = bar["atr"]

        # 1. Manage open trade
        if open_trade is not None:
            side = open_trade["side"]
            entry_p = open_trade["entry_price"]
            sl_p = open_trade["sl_price"]
            tp_p = open_trade["tp_price"]
            trade_atr = open_trade["atr"]

            # Dynamic ATR Trailing Stop:
            if trade_atr > 0:
                if side == "long" and h >= entry_p + TRAILING_STOP_ATR_TRIGGER * trade_atr:
                    cand_sl = max(entry_p, h - TRAILING_STOP_ATR_DIST * trade_atr)
                    if cand_sl > sl_p:
                        sl_p = cand_sl
                        open_trade["sl_price"] = sl_p
                elif side == "short" and l <= entry_p - TRAILING_STOP_ATR_TRIGGER * trade_atr:
                    cand_sl = min(entry_p, l + TRAILING_STOP_ATR_DIST * trade_atr)
                    if cand_sl < sl_p:
                        sl_p = cand_sl
                        open_trade["sl_price"] = sl_p

            exit_price = None
            exit_reason = None

            # Check SL
            if side == "long" and l <= sl_p:
                exit_price = sl_p
                exit_reason = "SL"
            elif side == "short" and h >= sl_p:
                exit_price = sl_p
                exit_reason = "SL"

            # Check TP
            if exit_price is None:
                if side == "long" and h >= tp_p:
                    exit_price = tp_p
                    exit_reason = "TP"
                elif side == "short" and l <= tp_p:
                    exit_price = tp_p
                    exit_reason = "TP"

            # Check Timeout
            bars_held = i - open_trade["entry_idx"]
            if exit_price is None and bars_held >= max_hold_bars:
                exit_price = c
                exit_reason = "TIMEOUT"

            if exit_price is not None:
                if side == "long":
                    gross_ret = (exit_price * (1 - slip) - entry_p * (1 + slip)) / (entry_p * (1 + slip))
                else:
                    gross_ret = (entry_p * (1 - slip) - exit_price * (1 + slip)) / (entry_p * (1 + slip))
                net_ret = gross_ret - 2 * fee
                pnl = open_trade["size"] * net_ret
                balance += pnl
                equity_curve.append(balance)

                trades.append({
                    "net_ret": net_ret,
                    "pnl": pnl,
                    "win": net_ret > 0,
                    "reason": exit_reason,
                })
                open_trade = None

        # 2. Check entry signal if flat
        if open_trade is None and bar["proba"] is not None:
            proba = bar["proba"]
            p_down, p_side, p_up = proba[0], proba[1], proba[2]

            pred_idx = int(np.argmax(proba))
            conf = float(proba[pred_idx])

            if pred_idx != 1 and conf >= threshold:
                side = "long" if pred_idx == 2 else "short"
                entry_p = c

                if side == "long":
                    tp_p = entry_p + tp_mult * atr
                    sl_p = entry_p - sl_mult * atr
                else:
                    tp_p = entry_p - tp_mult * atr
                    sl_p = entry_p + sl_mult * atr

                # Quarter-Kelly Sizing
                b_ratio = tp_mult / sl_mult
                p = conf
                q = 1.0 - p
                f_star = (b_ratio * p - q) / b_ratio
                frac = min(max(f_star * KELLY_SAFETY_FACTOR, MIN_KELLY_FRACTION), MAX_KELLY_FRACTION)
                pos_size = balance * frac

                open_trade = {
                    "side": side,
                    "entry_price": entry_p,
                    "entry_idx": i,
                    "tp_price": tp_p,
                    "sl_price": sl_p,
                    "size": pos_size,
                    "atr": atr,
                }

    total_trades = len(trades)
    if total_trades < 10:
        return {
            "total_trades": total_trades,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": -10.0,
            "max_drawdown": 1.0,
            "net_return_pct": -100.0,
            "score": -50.0,
        }

    wins = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]

    win_rate = len(wins) / total_trades
    gross_profits = sum(t["pnl"] for t in wins)
    gross_losses = abs(sum(t["pnl"] for t in losses))
    profit_factor = (gross_profits / gross_losses) if gross_losses > 0 else 5.0

    # Max Drawdown
    eq_arr = np.array(equity_curve)
    peaks = np.maximum.accumulate(eq_arr)
    drawdowns = (eq_arr - peaks) / peaks
    max_drawdown = float(np.min(drawdowns))  # negative float

    # Sharpe Ratio
    returns = np.array([t["net_ret"] for t in trades])
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    sharpe = float(mean_ret / std_ret * np.sqrt(6 * 365)) if std_ret > 0 else 0.0

    net_return_pct = ((balance - INITIAL_BALANCE_USDT) / INITIAL_BALANCE_USDT) * 100.0

    # Objective Score
    mdd_penalty = 2.0 * max(0.0, abs(max_drawdown) - 0.10)
    score = sharpe * math.sqrt(max(0.1, profit_factor)) - mdd_penalty
    if total_trades < 25:
        score -= (25 - total_trades) * 0.5

    return {
        "total_trades": total_trades,
        "win_rate": win_rate * 100.0,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown * 100.0,
        "net_return_pct": net_return_pct,
        "score": score,
    }


def optimize_for_symbol(symbol: str, n_trials: int = 30) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f" OPTUNA BAYESIAN HYPEROPT: {symbol}")
    print(f"=======================================================")

    bars, _ = load_dataset_for_symbol(symbol)
    print(f"Loaded {len(bars):,} 4h bars for {symbol}. Running {n_trials} trials...")

    def objective(trial: optuna.Trial) -> float:
        threshold = trial.suggest_float("threshold", 0.40, 0.75, step=0.01)
        tp_mult = trial.suggest_float("tp_mult", 1.0, 3.5, step=0.1)
        sl_mult = trial.suggest_float("sl_mult", 0.8, 2.0, step=0.1)
        max_hold_bars = trial.suggest_int("max_hold_bars", 3, 12, step=1)

        res = simulate_strategy(bars, threshold, tp_mult, sl_mult, max_hold_bars)
        trial.set_user_attr("win_rate", res["win_rate"])
        trial.set_user_attr("profit_factor", res["profit_factor"])
        trial.set_user_attr("sharpe_ratio", res["sharpe_ratio"])
        trial.set_user_attr("max_drawdown", res["max_drawdown"])
        trial.set_user_attr("total_trades", res["total_trades"])
        trial.set_user_attr("net_return_pct", res["net_return_pct"])

        return res["score"]

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    print(f"\n>> Best Trial #{best.number} for {symbol}:")
    print(f"   Threshold: {best.params['threshold']:.2f}")
    print(f"   ATR TP Mult: {best.params['tp_mult']:.1f}")
    print(f"   ATR SL Mult: {best.params['sl_mult']:.1f}")
    print(f"   Max Hold Bars: {best.params['max_hold_bars']} ({best.params['max_hold_bars']*4}h)")
    print(f"   Win Rate: {best.user_attrs['win_rate']:.1f}% | Profit Factor: {best.user_attrs['profit_factor']:.2f}")
    print(f"   Sharpe: {best.user_attrs['sharpe_ratio']:.2f} | MDD: {best.user_attrs['max_drawdown']:.1f}% | Net Ret: {best.user_attrs['net_return_pct']:+.1f}% | Trades: {best.user_attrs['total_trades']}")

    return {
        "symbol": symbol,
        "best_params": best.params,
        "metrics": best.user_attrs,
        "best_score": best.value,
    }


def main():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuner for AIFinance")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT", help="Comma-separated symbols")
    parser.add_argument("--trials", type=int, default=30, help="Number of trials per symbol")
    parser.add_argument("--out", default=str(OUTPUT_FILE), help="Output JSON path")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    results = {}

    for sym in symbols:
        try:
            res = optimize_for_symbol(sym, n_trials=args.trials)
            results[sym] = res
        except Exception as e:
            print(f"ERROR optimizing {sym}: {e}", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] All best hyperopt parameters saved to {args.out}")


if __name__ == "__main__":
    main()
