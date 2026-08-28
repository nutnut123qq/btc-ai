#!/usr/bin/env python3
"""
Phase 5: Historical Out-Of-Sample (OOS) Blind Replay & Performance Audit
========================================================================
Performs strict point-in-time blind simulation on unseen post-2025 data.
Evaluates:
  - Engine A: ML Champion Model (XGB Calibrated 4h & Balanced 1h)
  - Engine B: Same-Timeframe Regime / 5-Bar Momentum Rule Blend
Applies real-world transaction costs (10 bps fee + 5 bps slippage per side).
Computes quant metrics, calibration quality, and adversarial regime breakdown.
"""

import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import psycopg2
from sklearn.metrics import brier_score_loss, log_loss

# UTF-8 stdout configuration for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from db_config import get_db_connection
from trading_config import (
    DEFAULT_SYMBOL,
    FEE_BPS,
    SLIPPAGE_BPS,
    SPLIT_TIMESTAMP_MS,
    TIMEFRAME_THRESHOLDS,
    TOTAL_COST_PER_SIDE_BPS,
    TOTAL_ROUNDTRIP_COST_PCT,
)

MODELS_DIR = Path(__file__).parent / "models"
OUTPUT_REPORT_MD = Path(__file__).parent / "oos_blind_performance_report.md"
OUTPUT_REPORT_JSON = Path(__file__).parent / "oos_blind_performance_report.json"


def get_connection():
    return get_db_connection()


def fetch_oos_windows(
    symbol: str, timeframe: str, window_size: int, horizon: str, start_ms: int
) -> List[Tuple[int, List[float], int, Optional[float]]]:
    """Fetch out-of-sample window feature vectors strictly >= start_ms."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT "WindowEndMs", "FeatureVector", "Label", "TargetReturn"
        FROM "WindowClassificationDatasets"
        WHERE "Symbol" = %s AND "Timeframe" = %s
          AND "WindowSize" = %s AND "Horizon" = %s
          AND "WindowEndMs" >= %s
        ORDER BY "WindowEndMs" ASC
        """,
        (symbol, timeframe, window_size, horizon, start_ms),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [(int(r[0]), list(r[1]) if r[1] else [], int(r[2]), float(r[3]) if r[3] is not None else None) for r in rows]


def fetch_klines_map(symbol: str, timeframe: str, start_ms: int) -> Dict[int, Dict[str, float]]:
    """Fetch klines as a lookup map."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT "OpenTimeMs", "Open", "High", "Low", "Close", "Volume"
        FROM "Klines"
        WHERE "Symbol" = %s AND "Timeframe" = %s
          AND "OpenTimeMs" >= %s
        ORDER BY "OpenTimeMs" ASC
        """,
        (symbol, timeframe, start_ms - 200 * 3600 * 1000),  # extra lookback for SMAs
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {
        int(r[0]): {
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
            "volume": float(r[5]),
        }
        for r in rows
    }


def classify_market_regime(
    times: List[int], kline_map: Dict[int, Dict[str, float]], current_time: int
) -> str:
    """Classifies market regime into Bull Trend, Bear Trend, or Chop/Sideways."""
    idx = None
    for i, t in enumerate(times):
        if t == current_time:
            idx = i
            break
    if idx is None or idx < 50:
        return "Chop / Sideways"

    closes = [kline_map[times[j]]["close"] for j in range(idx - 49, idx + 1)]
    sma20 = float(np.mean(closes[-20:]))
    sma50 = float(np.mean(closes))
    current_close = closes[-1]
    ret20 = (current_close - closes[-20]) / closes[-20]

    if current_close > sma50 and sma20 > sma50 and ret20 > 0.01:
        return "Bull Trend"
    elif current_close < sma50 and sma20 < sma50 and ret20 < -0.01:
        return "Bear Trend"
    else:
        return "Chop / Sideways"


def run_engine_a_simulation(
    model_path: Path,
    timeframe: str,
    window_size: int,
    horizon: str,
    horizon_ms: int,
    start_ms: int,
    confidence_threshold: float,
) -> Dict[str, Any]:
    """Simulates trades for Engine A (ML Champion Model)."""
    if not model_path.exists():
        return {"error": f"Model file not found: {model_path}"}

    model = joblib.load(model_path)
    meta_path = model_path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    model_name = meta.get("model_name", model_path.stem)

    windows = fetch_oos_windows(DEFAULT_SYMBOL, timeframe, window_size, horizon, start_ms)
    if not windows:
        return {"error": f"No OOS window data found for {timeframe} ws={window_size} h={horizon}"}

    kline_map = fetch_klines_map(DEFAULT_SYMBOL, timeframe, start_ms)
    sorted_kline_times = sorted(kline_map.keys())

    trades = []
    probabilities = []
    y_true_binary = []
    y_pred_proba_up = []

    fee_rate = FEE_BPS / 10000.0
    slippage_rate = SLIPPAGE_BPS / 10000.0

    for window_end_ms, vec, true_label, target_ret in windows:
        if not vec:
            continue

        X = np.array(vec, dtype=np.float32).reshape(1, -1)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            pred_idx = int(np.argmax(proba))
            conf = float(proba[pred_idx])
            prob_down, prob_sideways, prob_up = float(proba[0]), float(proba[1]), float(proba[2])
        else:
            pred_idx = int(model.predict(X)[0])
            conf = 1.0
            prob_down = prob_sideways = prob_up = 0.33

        # XGB label mapping {0,1,2} -> {-1,0,1}
        label = pred_idx - 1 if "XGB" in model_name else pred_idx
        probabilities.append(conf)

        if true_label != 0:
            y_true_binary.append(1 if true_label == 1 else 0)
            y_pred_proba_up.append(prob_up / (prob_up + prob_down) if (prob_up + prob_down) > 0 else 0.5)

        if label == 0 or conf < confidence_threshold:
            continue

        side = "long" if label == 1 else "short"
        entry_time = window_end_ms
        exit_time = entry_time + horizon_ms

        if entry_time not in kline_map or exit_time not in kline_map:
            continue

        entry_price = kline_map[entry_time]["close"]
        exit_price = kline_map[exit_time]["close"]

        # Point-in-time slippage & cost model
        if side == "long":
            entry_adj = entry_price * (1.0 + slippage_rate)
            exit_adj = exit_price * (1.0 - slippage_rate)
            gross_return = (exit_adj - entry_adj) / entry_adj
        else:
            entry_adj = entry_price * (1.0 - slippage_rate)
            exit_adj = exit_price * (1.0 + slippage_rate)
            gross_return = (entry_adj - exit_adj) / entry_adj

        net_return = gross_return - 2.0 * fee_rate
        regime = classify_market_regime(sorted_kline_times, kline_map, entry_time)

        trades.append({
            "entry_time": entry_time,
            "exit_time": exit_time,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "confidence": conf,
            "gross_return": gross_return,
            "net_return": net_return,
            "regime": regime,
            "true_label": true_label,
        })

    metrics = calculate_quant_metrics(trades, sorted_kline_times, kline_map, start_ms)
    metrics["model_name"] = model_name
    metrics["timeframe"] = timeframe
    metrics["horizon"] = horizon
    metrics["threshold"] = confidence_threshold
    metrics["prob_distribution"] = summarize_distribution(probabilities)

    # Calibration Metrics
    if len(y_true_binary) > 10:
        metrics["brier_score"] = float(brier_score_loss(y_true_binary, y_pred_proba_up))
        metrics["log_loss"] = float(log_loss(y_true_binary, y_pred_proba_up, labels=[0, 1]))
    else:
        metrics["brier_score"] = None
        metrics["log_loss"] = None

    metrics["regime_breakdown"] = analyze_regime_breakdown(trades)
    return metrics


def simulate_engine_b_windows(
    windows: List[Tuple[int, List[float], int, Optional[float]]],
    kline_map: Dict[int, Dict[str, float]],
    horizon_ms: int,
    confidence_threshold: float,
    ml_model: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], List[float], List[int]]:
    """Run Engine B on supplied point-in-time inputs.

    Labels and target returns are present in the dataset rows for later scoring, but
    this decision path deliberately discards them. Keeping this production seam
    separate makes that invariant directly testable without a database.
    """
    sorted_kline_times = sorted(kline_map.keys())
    index_by_time = {timestamp: index for index, timestamp in enumerate(sorted_kline_times)}
    fee_rate = FEE_BPS / 10000.0
    slippage_rate = SLIPPAGE_BPS / 10000.0
    trades = []
    probabilities = []

    for window_end_ms, vec, *_outcomes in windows:
        if not vec:
            continue

        regime = classify_market_regime(sorted_kline_times, kline_map, window_end_ms)
        is_trending = regime in ("Bull Trend", "Bear Trend")

        # Primary same-timeframe regime direction score.
        l1_up = 0.70 if regime == "Bull Trend" else 0.20 if regime == "Bear Trend" else 0.35
        l1_down = 0.70 if regime == "Bear Trend" else 0.20 if regime == "Bull Trend" else 0.35

        # Five-bar momentum score using prices available at window_end_ms.
        idx = index_by_time.get(window_end_ms)
        if idx is not None and idx >= 4:
            past_closes = [kline_map[sorted_kline_times[j]]["close"] for j in range(idx - 4, idx + 1)]
            past_ret = (past_closes[-1] - past_closes[0]) / past_closes[0]
            l2_momentum_up = 0.65 if past_ret > 0.005 else 0.25 if past_ret < -0.005 else 0.40
            l2_momentum_down = 0.65 if past_ret < -0.005 else 0.25 if past_ret > 0.005 else 0.40
        else:
            l2_momentum_up = 0.33
            l2_momentum_down = 0.33

        # Secondary weight from the same SMA/return regime classifier (not ADX).
        l3_up = 0.75 if regime == "Bull Trend" else 0.20
        l3_down = 0.75 if regime == "Bear Trend" else 0.20

        # Symmetric trending/chop constant (not support/resistance or order flow).
        l4_up = 0.60 if is_trending else 0.40
        l4_down = 0.60 if is_trending else 0.40

        # Model probability when available; otherwise a regime-dependent fallback.
        if ml_model is not None and hasattr(ml_model, "predict_proba"):
            X = np.array(vec, dtype=np.float32).reshape(1, -1)
            proba = ml_model.predict_proba(X)[0]
            l5_down, _l5_side, l5_up = float(proba[0]), float(proba[1]), float(proba[2])
        else:
            l5_up = 0.60 if regime == "Bull Trend" else 0.40
            l5_down = 0.60 if regime == "Bear Trend" else 0.40

        w_primary_regime = 0.35 if is_trending else 0.25
        w_momentum = 0.20 if is_trending else 0.15
        w_secondary_regime = 0.15 if is_trending else 0.10
        w_market_state = 0.10 if is_trending else 0.30
        w_ml = 0.20

        w_total = w_primary_regime + w_momentum + w_secondary_regime + w_market_state + w_ml
        agg_up = (
            w_primary_regime * l1_up
            + w_momentum * l2_momentum_up
            + w_secondary_regime * l3_up
            + w_market_state * l4_up
            + w_ml * l5_up
        ) / w_total
        agg_down = (
            w_primary_regime * l1_down
            + w_momentum * l2_momentum_down
            + w_secondary_regime * l3_down
            + w_market_state * l4_down
            + w_ml * l5_down
        ) / w_total
        agg_side = max(0.05, 1.0 - agg_up - agg_down)

        if agg_up >= agg_down and agg_up >= agg_side:
            conf = agg_up
            label = 1
        elif agg_down >= agg_up and agg_down >= agg_side:
            conf = agg_down
            label = -1
        else:
            conf = agg_side
            label = 0

        probabilities.append(conf)

        if label == 0 or conf < confidence_threshold:
            continue

        side = "long" if label == 1 else "short"
        entry_time = window_end_ms
        exit_time = entry_time + horizon_ms

        if entry_time not in kline_map or exit_time not in kline_map:
            continue

        entry_price = kline_map[entry_time]["close"]
        exit_price = kline_map[exit_time]["close"]

        if side == "long":
            entry_adj = entry_price * (1.0 + slippage_rate)
            exit_adj = exit_price * (1.0 - slippage_rate)
            gross_return = (exit_adj - entry_adj) / entry_adj
        else:
            entry_adj = entry_price * (1.0 - slippage_rate)
            exit_adj = exit_price * (1.0 + slippage_rate)
            gross_return = (entry_adj - exit_adj) / entry_adj

        net_return = gross_return - 2.0 * fee_rate

        trades.append({
            "entry_time": entry_time,
            "exit_time": exit_time,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "confidence": conf,
            "gross_return": gross_return,
            "net_return": net_return,
            "regime": regime,
        })

    return trades, probabilities, sorted_kline_times


def run_engine_b_simulation(
    timeframe: str,
    window_size: int,
    horizon: str,
    horizon_ms: int,
    start_ms: int,
    confidence_threshold: float = 0.58,
    model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Simulates Engine B's same-timeframe regime/momentum rule blend."""
    windows = fetch_oos_windows(DEFAULT_SYMBOL, timeframe, window_size, horizon, start_ms)
    if not windows:
        return {"error": f"No OOS window data for Engine B {timeframe}"}

    kline_map = fetch_klines_map(DEFAULT_SYMBOL, timeframe, start_ms)

    ml_model = None
    if model_path and model_path.exists():
        try:
            ml_model = joblib.load(model_path)
        except Exception:
            ml_model = None

    trades, probabilities, sorted_kline_times = simulate_engine_b_windows(
        windows,
        kline_map,
        horizon_ms,
        confidence_threshold,
        ml_model,
    )
    metrics = calculate_quant_metrics(trades, sorted_kline_times, kline_map, start_ms)
    metrics["model_name"] = "Same-Timeframe Regime / 5-Bar Momentum Rule Blend (Engine B)"
    metrics["timeframe"] = timeframe
    metrics["horizon"] = horizon
    metrics["threshold"] = confidence_threshold
    metrics["prob_distribution"] = summarize_distribution(probabilities)
    metrics["regime_breakdown"] = analyze_regime_breakdown(trades)
    return metrics


def calculate_quant_metrics(
    trades: List[Dict[str, Any]],
    kline_times: List[int],
    kline_map: Dict[int, Dict[str, float]],
    start_ms: int,
    initial_capital: float = 10_000.0,
) -> Dict[str, Any]:
    """Computes comprehensive quantitative performance statistics."""
    if not trades:
        return {
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "net_return_pct": 0.0,
            "buy_and_hold_return_pct": 0.0,
            "trade_frequency_per_day": 0.0,
        }

    equity = initial_capital
    peak = initial_capital
    max_dd = 0.0
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    returns = []

    for t in trades:
        ret = t["net_return"]
        returns.append(ret)
        equity *= 1.0 + ret
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd:
            max_dd = dd

        if ret > 0:
            wins += 1
            gross_profit += ret
        else:
            losses += 1
            gross_loss += abs(ret)

    total_return_pct = (equity - initial_capital) / initial_capital * 100.0
    win_rate_pct = (wins / len(trades)) * 100.0 if trades else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    avg_ret = float(np.mean(returns)) if returns else 0.0
    std_ret = float(np.std(returns)) if len(returns) > 1 else 0.0
    # Annualized Sharpe ratio (using ~365*6 = 2190 bars/year for 4h or 8760 for 1h)
    annual_factor = math.sqrt(2190)
    sharpe = (avg_ret / std_ret * annual_factor) if std_ret > 0 else 0.0

    downside_returns = [r for r in returns if r < 0]
    downside_std = float(np.std(downside_returns)) if len(downside_returns) > 1 else 0.0
    sortino = (avg_ret / downside_std * annual_factor) if downside_std > 0 else 0.0

    # Buy & Hold Return for the period
    oos_kline_times = [t for t in kline_times if t >= start_ms]
    if oos_kline_times:
        bh_start = kline_map[oos_kline_times[0]]["close"]
        bh_end = kline_map[oos_kline_times[-1]]["close"]
        bh_return_pct = ((bh_end - bh_start) / bh_start) * 100.0
        duration_days = max(1.0, (oos_kline_times[-1] - oos_kline_times[0]) / (86400.0 * 1000.0))
    else:
        bh_return_pct = 0.0
        duration_days = 1.0

    trade_freq = len(trades) / duration_days

    return {
        "total_trades": len(trades),
        "winning_trades": wins,
        "losing_trades": losses,
        "win_rate_pct": round(win_rate_pct, 2),
        "profit_factor": round(profit_factor, 2) if not math.isinf(profit_factor) else 999.0,
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "net_return_pct": round(total_return_pct, 2),
        "buy_and_hold_return_pct": round(bh_return_pct, 2),
        "trade_frequency_per_day": round(trade_freq, 2),
        "duration_days": round(duration_days, 1),
    }


def summarize_distribution(values: List[float]) -> Dict[str, float]:
    """Summarizes probability score distribution."""
    if not values:
        return {}
    arr = np.array(values)
    return {
        "count": len(values),
        "mean": round(float(np.mean(arr)), 3),
        "std": round(float(np.std(arr)), 3),
        "min": round(float(np.min(arr)), 3),
        "p25": round(float(np.percentile(arr, 25)), 3),
        "median": round(float(np.median(arr)), 3),
        "p75": round(float(np.percentile(arr, 75)), 3),
        "p90": round(float(np.percentile(arr, 90)), 3),
        "p99": round(float(np.percentile(arr, 99)), 3),
        "max": round(float(np.max(arr)), 3),
    }


def analyze_regime_breakdown(trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Analyzes performance split by Market Regime."""
    regimes = defaultdict(list)
    for t in trades:
        regimes[t["regime"]].append(t)

    breakdown = {}
    for reg_name, reg_trades in regimes.items():
        wins = sum(1 for t in reg_trades if t["net_return"] > 0)
        gross_p = sum(t["net_return"] for t in reg_trades if t["net_return"] > 0)
        gross_l = sum(abs(t["net_return"]) for t in reg_trades if t["net_return"] <= 0)
        pf = (gross_p / gross_l) if gross_l > 0 else 999.0
        cum_ret = (float(np.prod([1.0 + t["net_return"] for t in reg_trades])) - 1.0) * 100.0

        breakdown[reg_name] = {
            "trades_count": len(reg_trades),
            "win_rate_pct": round(wins / len(reg_trades) * 100.0, 2),
            "profit_factor": round(pf, 2),
            "cumulative_net_return_pct": round(cum_ret, 2),
        }
    return breakdown


def _fmt_pct(val: Optional[float]) -> str:
    if val is None:
        return "N/A"
    return f"+{val:.2f}%" if val > 0 else f"{val:.2f}%"


def generate_markdown_report(results: Dict[str, Any]) -> str:
    """Generates an honest, dynamically computed Markdown audit report."""
    md = []
    md.append("# Bitcoin AI Analyst — Out-of-Sample (OOS) Blind Performance Audit\n")
    md.append(f"**Execution Timestamp:** {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S UTC}\n")
    md.append(f"**Out-Of-Sample Start Date:** `2025-01-01 00:00:00 UTC`\n")
    md.append(f"**Transaction Costs Enforced:** Fee = `{FEE_BPS} bps/side`, Slippage = `{SLIPPAGE_BPS} bps/side` (Roundtrip = `{TOTAL_ROUNDTRIP_COST_PCT*100:.2f}%`)\n\n")

    md.append("## 1. Engine B Architecture & Semantics Disclosure\n\n")
    md.append(
        "Engine B combines two scores from the same SMA/return regime classifier, "
        "a five-bar momentum score, a symmetric trending/chop constant, and model "
        "probabilities (or a regime-dependent fallback). It does not calculate "
        "multi-timeframe confluence, ADX, market structure, liquidity, or order flow. "
        "Its production decision seam ignores dataset labels and target returns. "
        "Permutation tests cover that seam but do not independently certify upstream "
        "feature construction.\n\n"
    )

    md.append("## 2. Executive Summary & Benchmark Comparison\n\n")
    md.append("| Metric | Engine A (Champion XGB 4h) | Engine A (Balanced XGB 1h) | Engine B (Same-Timeframe Regime / 5-Bar Momentum Rule Blend) | Benchmark (Buy & Hold) |\n")
    md.append("|---|---|---|---|---|\n")

    res_4h = results.get("engine_a_4h", {})
    res_1h = results.get("engine_a_1h", {})
    res_ens = results.get("engine_b_ensemble", {})

    ret_4h = res_4h.get("net_return_pct")
    ret_1h = res_1h.get("net_return_pct")
    ret_ens = res_ens.get("net_return_pct")
    ret_bh = res_4h.get("buy_and_hold_return_pct")

    md.append(f"| **Total Trades** | {res_4h.get('total_trades', 0)} | {res_1h.get('total_trades', 0)} | {res_ens.get('total_trades', 0)} | N/A |\n")
    md.append(f"| **Win Rate (Post-Fee)** | **{res_4h.get('win_rate_pct', 0)}%** | {res_1h.get('win_rate_pct', 0)}% | **{res_ens.get('win_rate_pct', 0)}%** | N/A |\n")
    md.append(f"| **Profit Factor** | **{res_4h.get('profit_factor', 0)}** | {res_1h.get('profit_factor', 0)} | **{res_ens.get('profit_factor', 0)}** | N/A |\n")
    md.append(f"| **Sharpe Ratio (Ann.)** | **{res_4h.get('sharpe_ratio', 0)}** | {res_1h.get('sharpe_ratio', 0)} | **{res_ens.get('sharpe_ratio', 0)}** | — |\n")
    md.append(f"| **Sortino Ratio (Ann.)** | **{res_4h.get('sortino_ratio', 0)}** | {res_1h.get('sortino_ratio', 0)} | **{res_ens.get('sortino_ratio', 0)}** | — |\n")
    md.append(f"| **Max Drawdown (MDD)** | **{res_4h.get('max_drawdown_pct', 0)}%** | {res_1h.get('max_drawdown_pct', 0)}% | **{res_ens.get('max_drawdown_pct', 0)}%** | — |\n")
    md.append(f"| **Net Return (%)** | **{_fmt_pct(ret_4h)}** | {_fmt_pct(ret_1h)} | **{_fmt_pct(ret_ens)}** | **{_fmt_pct(ret_bh)}** |\n")
    md.append(f"| **Trade Freq (trades/day)** | {res_4h.get('trade_frequency_per_day', 0)} | {res_1h.get('trade_frequency_per_day', 0)} | {res_ens.get('trade_frequency_per_day', 0)} | — |\n\n")

    md.append("## 3. Confidence Calibration & Probability Distribution\n\n")
    md.append("Statistical validation of model confidence scores across all unseen test windows:\n\n")
    md.append("| Model | Mean Conf | Median | 25th % | 75th % | 90th % | 99th % | Brier Score | Log-Loss |\n")
    md.append("|---|---|---|---|---|---|---|---|---|\n")

    for k, title in [("engine_a_4h", "XGB 4h Calibrated"), ("engine_a_1h", "XGB 1h Balanced"), ("engine_b_ensemble", "Same-Timeframe Regime / 5-Bar Momentum Rule Blend")]:
        r = results.get(k, {})
        d = r.get("prob_distribution", {})
        brier = f"{r.get('brier_score'):.4f}" if r.get("brier_score") is not None else "N/A"
        ll = f"{r.get('log_loss'):.4f}" if r.get("log_loss") is not None else "N/A"
        md.append(f"| **{title}** | {d.get('mean', 0)} | {d.get('median', 0)} | {d.get('p25', 0)} | {d.get('p75', 0)} | {d.get('p90', 0)} | {d.get('p99', 0)} | `{brier}` | `{ll}` |\n")

    md.append("\n## 4. Market Regime Breakdown\n\n")
    md.append("Performance segmented by underlying market regime:\n\n")

    for k, title in [("engine_a_4h", "Engine A (Champion XGB 4h)"), ("engine_b_ensemble", "Engine B (Same-Timeframe Regime / 5-Bar Momentum Rule Blend)")]:
        r = results.get(k, {})
        rb = r.get("regime_breakdown", {})
        md.append(f"### {title}\n\n")
        md.append("| Market Regime | Trade Count | Win Rate (%) | Profit Factor | Cumulative Net Return (%) |\n")
        md.append("|---|---|---|---|---|\n")
        for reg, s in rb.items():
            cum_r = s.get('cumulative_net_return_pct')
            md.append(f"| **{reg}** | {s.get('trades_count', 0)} | {s.get('win_rate_pct', 0)}% | {s.get('profit_factor', 0)} | {_fmt_pct(cum_r)} |\n")
        md.append("\n")

    md.append("## 5. Quantitative Findings & Derived Conclusions\n\n")
    # Dynamically evaluate findings based strictly on actual metrics
    findings = []

    # Engine A 4h evaluation
    if res_4h.get("profit_factor", 0) > 1.2 and res_4h.get("win_rate_pct", 0) > 52.0:
        findings.append(f"1. **Engine A (4h XGB Calibrated)**: Demonstrated positive edge post-fees with Win Rate = `{res_4h.get('win_rate_pct')}%`, Profit Factor = `{res_4h.get('profit_factor')}`, and Net Return = `{_fmt_pct(ret_4h)}`.")
    else:
        findings.append(f"1. **Engine A (4h XGB Calibrated)**: Achieved Win Rate = `{res_4h.get('win_rate_pct', 0)}%` and Profit Factor = `{res_4h.get('profit_factor', 0)}`; requires conservative threshold gating to control drawdowns.")

    # Engine A 1h evaluation
    if res_1h.get("net_return_pct", 0) < 0:
        findings.append(f"2. **Engine A (1h XGB Balanced)**: Underperformed on 1h OOS data with Net Return = `{_fmt_pct(ret_1h)}` (Max Drawdown = `{res_1h.get('max_drawdown_pct', 0)}%`), confirming that 1h signals suffer from elevated transaction drag (30 bps roundtrip).")
    else:
        findings.append(f"2. **Engine A (1h XGB Balanced)**: Net Return = `{_fmt_pct(ret_1h)}`, Win Rate = `{res_1h.get('win_rate_pct', 0)}%`.")

    # Engine B evaluation
    if res_ens.get("profit_factor", 0) > 1.1 and res_ens.get("net_return_pct", 0) > 0:
        findings.append(f"3. **Engine B (Same-Timeframe Regime / 5-Bar Momentum Rule Blend)**: This replay reported Net Return = `{_fmt_pct(ret_ens)}` and Win Rate = `{res_ens.get('win_rate_pct')}%`.")
    else:
        findings.append(f"3. **Engine B (Same-Timeframe Regime / 5-Bar Momentum Rule Blend)**: This replay reported Net Return = `{_fmt_pct(ret_ens)}` and Win Rate = `{res_ens.get('win_rate_pct', 0)}%`. The result does not support live allocation.")

    findings.append(f"4. **Benchmark Comparison**: Buy & Hold OOS Return was `{_fmt_pct(ret_bh)}`.")
    md.append("\n".join(findings) + "\n")
    return "".join(md)


def main():
    print("=== STARTING PHASE 5: OUT-OF-SAMPLE BLIND REPLAY & PERFORMANCE AUDIT ===")

    start_ms = SPLIT_TIMESTAMP_MS  # 2025-01-01 00:00:00 UTC
    dt_str = datetime.fromtimestamp(start_ms / 1000, timezone.utc).strftime("%Y-%m-%d")
    print(f"-> Blind Test Period Start: {dt_str}")
    print(f"-> Applied Costs: Fee={FEE_BPS} bps/side, Slippage={SLIPPAGE_BPS} bps/side")

    results = {}

    # 1. Engine A: Champion 4h Calibrated Model
    print("\n--- Running Engine A: Champion Model (BTCUSDT 4h ws=5 h=4h XGB Calibrated) ---")
    model_4h = MODELS_DIR / "BTCUSDT_4h_ws5_h4h_XGB_calibrated.joblib"
    thr_4h = TIMEFRAME_THRESHOLDS.get("4h", 0.61)
    res_4h = run_engine_a_simulation(model_4h, "4h", 5, "4h", 14_400_000, start_ms, thr_4h)
    results["engine_a_4h"] = res_4h
    print(f"  Trades: {res_4h.get('total_trades')} | Win Rate: {res_4h.get('win_rate_pct')}% | Profit Factor: {res_4h.get('profit_factor')} | Net Return: {_fmt_pct(res_4h.get('net_return_pct'))} | MDD: {res_4h.get('max_drawdown_pct')}%")

    # 2. Engine A: Balanced 1h Model
    print("\n--- Running Engine A: Balanced Model (BTCUSDT 1h ws=5 h=1h XGB Balanced) ---")
    model_1h = MODELS_DIR / "BTCUSDT_1h_ws5_h1h_XGB_balanced.joblib"
    thr_1h = TIMEFRAME_THRESHOLDS.get("1h", 0.58)
    res_1h = run_engine_a_simulation(model_1h, "1h", 5, "1h", 3_600_000, start_ms, thr_1h)
    results["engine_a_1h"] = res_1h
    print(f"  Trades: {res_1h.get('total_trades')} | Win Rate: {res_1h.get('win_rate_pct')}% | Profit Factor: {res_1h.get('profit_factor')} | Net Return: {_fmt_pct(res_1h.get('net_return_pct'))} | MDD: {res_1h.get('max_drawdown_pct')}%")

    # 3. Engine B: Point-in-Time Multi-Layer Ensemble
    print("\n--- Running Engine B: Same-Timeframe Regime / 5-Bar Momentum Rule Blend (BTCUSDT 1h ws=5 h=1h) ---")
    res_ens = run_engine_b_simulation("1h", 5, "1h", 3_600_000, start_ms, 0.58, model_path=model_1h)
    results["engine_b_ensemble"] = res_ens
    print(f"  Trades: {res_ens.get('total_trades')} | Win Rate: {res_ens.get('win_rate_pct')}% | Profit Factor: {res_ens.get('profit_factor')} | Net Return: {_fmt_pct(res_ens.get('net_return_pct'))} | MDD: {res_ens.get('max_drawdown_pct')}%")

    # 4. Generate & Save Reports
    OUTPUT_REPORT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    report_md = generate_markdown_report(results)
    OUTPUT_REPORT_MD.write_text(report_md, encoding="utf-8")

    print(f"\n-> Saved JSON Report: {OUTPUT_REPORT_JSON}")
    print(f"-> Saved Markdown Report: {OUTPUT_REPORT_MD}")
    print("\n=== PHASE 5 BLIND REPLAY COMPLETED SUCCESSFULLY! ===")


if __name__ == "__main__":
    main()
