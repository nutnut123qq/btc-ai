#!/usr/bin/env python3
"""
Phase 12: Point-in-Time Historical Blind Replay & Adversarial Audit for Multi-Asset ML Champion Models
(BTCUSDT vs ETHUSDT vs SOLUSDT) on Out-Of-Sample (OOS) Test Set (412 days).
Compares Before vs After Sniper Calibration, Dynamic ATR Sizing & BTC Confluence Gatekeeper.
"""

import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import joblib
import numpy as np
import psycopg2
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'bitcoin_analyst',
    'user': 'postgres',
    'password': '123456'
}

MODELS_DIR = Path(__file__).parent / "models"

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

FEE_BPS = 10.0      # 0.10% per side
SLIPPAGE_BPS = 5.0  # 0.05% per side
ROUNDTRIP_COST_PCT = (FEE_BPS + SLIPPAGE_BPS) * 2.0 / 10000.0  # 0.0030 (0.30%)

def time_features(open_ms: int) -> list:
    dt = datetime.fromtimestamp(open_ms / 1000, timezone.utc)
    hour = dt.hour + dt.minute / 60.0
    dow = dt.weekday()
    return [
        math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
        math.sin(2 * math.pi * dow / 7), math.cos(2 * math.pi * dow / 7),
        1.0 if dow >= 5 else 0.0,
    ]

def load_data(symbol: str, timeframe: str = "4h", ws: int = 5, tf_ms: int = 14400000):
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
        SELECT "OpenTimeMs", "Open", "High", "Low", "Close", "Volume"
        FROM "Klines"
        WHERE "Symbol"=%s AND "Timeframe"=%s
        ORDER BY "OpenTimeMs" ASC
    """, (symbol, timeframe))
    kline_rows = cur.fetchall()
    
    cur.execute("""
        SELECT "OpenTimeMs", "Atr14", "Rsi14", "Ema50", "Ema200"
        FROM "TechnicalIndicators"
        WHERE "Symbol"=%s AND "Timeframe"=%s
        ORDER BY "OpenTimeMs" ASC
    """, (symbol, timeframe))
    tech_rows = cur.fetchall()
    conn.close()
    
    klines_dict = {int(r[0]): {"open": float(r[1]), "high": float(r[2]), "low": float(r[3]), "close": float(r[4]), "vol": float(r[5])} for r in kline_rows}
    tech_dict = {int(r[0]): {"atr": float(r[1]) if r[1] is not None else 0.0, "rsi": float(r[2]) if r[2] is not None else 50.0, "ema50": float(r[3]) if r[3] is not None else 0.0, "ema200": float(r[4]) if r[4] is not None else 0.0} for r in tech_rows}
    
    times = []
    vectors = []
    regimes = []
    
    for i in range(ws - 1, len(feature_rows)):
        window = feature_rows[i - ws + 1 : i + 1]
        
        is_continuous = True
        for j in range(1, len(window)):
            if window[j][0] - window[j-1][0] != tf_ms:
                is_continuous = False
                break
        if not is_continuous:
            continue
            
        has_null = False
        vec = []
        for r in window:
            vals = r[1:]
            core_vals = vals[:28]
            if any(v is None for v in core_vals):
                has_null = True
                break
            vec.extend(float(v) if v is not None else 0.0 for v in vals)
            vec.extend(time_features(r[0]))
            
        if has_null or len(vec) != ws * 35:
            continue
            
        end_ms = int(window[-1][0])
        if end_ms not in klines_dict:
            continue
            
        t_info = tech_dict.get(end_ms, {})
        close_p = klines_dict[end_ms]["close"]
        ema50 = t_info.get("ema50", 0.0)
        ema200 = t_info.get("ema200", 0.0)
        rsi = t_info.get("rsi", 50.0)
        
        if ema50 > 0 and ema200 > 0:
            if close_p > ema50 and ema50 > ema200:
                regime = "Bull"
            elif close_p < ema50 and ema50 < ema200:
                regime = "Bear"
            else:
                regime = "Sideways"
        else:
            if rsi > 55:
                regime = "Bull"
            elif rsi < 45:
                regime = "Bear"
            else:
                regime = "Sideways"
                
        times.append(end_ms)
        vectors.append(vec)
        regimes.append(regime)
        
    return np.array(times, dtype=np.int64), np.array(vectors, dtype=np.float32), regimes, klines_dict, tech_dict

def run_simulation(
    symbol: str,
    threshold: float,
    tp_mult: float,
    sl_mult: float,
    use_btc_confluence: bool = False,
    btc_data_bundle: Optional[Tuple] = None,
    start_test_ms: int = 1751932800000, # 2025-07-08T00:00:00Z
    tf_ms: int = 14400000,
    max_hold_bars: int = 6,
) -> Dict[str, Any]:
    # 1. Load Model
    model_path = MODELS_DIR / f"{symbol}_4h_ws5_h4h_XGB_calibrated.joblib"
    model = joblib.load(model_path)
    
    # 2. Load Data
    times, X, regimes, klines_dict, tech_dict = load_data(symbol)
    
    test_mask = times >= start_test_ms
    times_test = times[test_mask]
    X_test = X[test_mask]
    regimes_test = [regimes[i] for i, m in enumerate(test_mask) if m]
    
    probas = model.predict_proba(X_test)
    preds = np.argmax(probas, axis=1) # 0: Down, 1: Sideways, 2: Up
    confs = np.max(probas, axis=1)
    
    # Pre-compute BTC confluence lookup if required
    btc_lookup = {}
    if use_btc_confluence and btc_data_bundle:
        btc_times, btc_X, btc_tech = btc_data_bundle
        btc_model = joblib.load(MODELS_DIR / "BTCUSDT_4h_ws5_h4h_XGB_calibrated.joblib")
        btc_probs = btc_model.predict_proba(btc_X)
        for t_ms, probs in zip(btc_times, btc_probs):
            btc_rsi = btc_tech.get(t_ms, {}).get("rsi", 50.0)
            btc_lookup[t_ms] = {
                "p_down": float(probs[0]),
                "p_up": float(probs[2]),
                "rsi": btc_rsi
            }
            
    # Simulate Trades Point-in-Time
    trades = []
    in_pos = False
    pos_side = None
    entry_price = 0.0
    entry_time_ms = 0
    tp_price = 0.0
    sl_price = 0.0
    entry_regime = "Sideways"
    bars_held = 0
    entry_conf = 0.0
    
    first_p = klines_dict[times_test[0]]["open"]
    last_p = klines_dict[times_test[-1]]["close"]
    buy_hold_return = (last_p - first_p) / first_p
    
    blocked_by_confluence = 0
    
    for i in range(len(times_test) - 1):
        t_cur = times_test[i]
        t_next = times_test[i+1]
        
        # Check active position exit
        if in_pos:
            bars_held += 1
            cur_k = klines_dict.get(t_cur)
            if not cur_k:
                continue
                
            high_p = cur_k["high"]
            low_p = cur_k["low"]
            close_p = cur_k["close"]
            
            exited = False
            exit_price = 0.0
            exit_reason = ""
            
            if pos_side == "LONG":
                if low_p <= sl_price:
                    exit_price = sl_price
                    exit_reason = "SL"
                    exited = True
                elif high_p >= tp_price:
                    exit_price = tp_price
                    exit_reason = "TP"
                    exited = True
                elif bars_held >= max_hold_bars:
                    exit_price = close_p
                    exit_reason = "TIMEOUT"
                    exited = True
            elif pos_side == "SHORT":
                if high_p >= sl_price:
                    exit_price = sl_price
                    exit_reason = "SL"
                    exited = True
                elif low_p <= tp_price:
                    exit_price = tp_price
                    exit_reason = "TP"
                    exited = True
                elif bars_held >= max_hold_bars:
                    exit_price = close_p
                    exit_reason = "TIMEOUT"
                    exited = True
                    
            if exited:
                if pos_side == "LONG":
                    gross_ret = (exit_price - entry_price) / entry_price
                else:
                    gross_ret = (entry_price - exit_price) / entry_price
                net_ret = gross_ret - ROUNDTRIP_COST_PCT
                
                trades.append({
                    "entry_time_ms": entry_time_ms,
                    "exit_time_ms": t_cur,
                    "side": pos_side,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_return": gross_ret,
                    "net_return": net_ret,
                    "exit_reason": exit_reason,
                    "bars_held": bars_held,
                    "regime": entry_regime,
                    "confidence": entry_conf,
                })
                in_pos = False
                bars_held = 0
                
        # Signal Generation
        if not in_pos:
            p_pred = preds[i]
            p_conf = confs[i]
            
            if p_pred != 1 and p_conf >= threshold:
                side = "LONG" if p_pred == 2 else "SHORT"
                
                # Check BTC Confluence Gatekeeper
                if use_btc_confluence and symbol != "BTCUSDT":
                    btc_info = btc_lookup.get(t_cur)
                    if btc_info:
                        if side == "LONG" and (btc_info["p_down"] >= 0.40 or btc_info["rsi"] <= 45.0):
                            blocked_by_confluence += 1
                            continue
                        elif side == "SHORT" and (btc_info["p_up"] >= 0.40 or btc_info["rsi"] >= 65.0):
                            blocked_by_confluence += 1
                            continue
                            
                next_k = klines_dict.get(t_next)
                if next_k:
                    entry_p = next_k["open"]
                    t_info = tech_dict.get(t_cur, {})
                    atr_val = t_info.get("atr", entry_p * 0.015)
                    if atr_val <= 0:
                        atr_val = entry_p * 0.015
                        
                    if side == "LONG":
                        tp = entry_p + (tp_mult * atr_val)
                        sl = entry_p - (sl_mult * atr_val)
                    else:
                        tp = entry_p - (tp_mult * atr_val)
                        sl = entry_p + (sl_mult * atr_val)
                        
                    in_pos = True
                    pos_side = side
                    entry_price = entry_p
                    entry_time_ms = t_next
                    tp_price = tp
                    sl_price = sl
                    entry_regime = regimes_test[i]
                    entry_conf = float(p_conf)
                    bars_held = 0
                    
    total_trades = len(trades)
    if total_trades == 0:
        return {
            "symbol": symbol,
            "threshold": threshold,
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "cumulative_net_return_pct": 0.0,
            "buy_hold_return_pct": buy_hold_return * 100.0,
            "blocked_by_confluence": blocked_by_confluence,
            "trades": []
        }
        
    net_returns = np.array([tr["net_return"] for tr in trades])
    wins = [r for r in net_returns if r > 0]
    losses = [r for r in net_returns if r <= 0]
    win_rate = len(wins) / total_trades * 100.0
    
    gross_profit = sum(wins) if len(wins) > 0 else 0.0
    gross_loss = abs(sum(losses)) if len(losses) > 0 else 1e-6
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    
    equity = np.cumprod(1.0 + net_returns)
    peak = np.maximum.accumulate(equity)
    drawdowns = (equity - peak) / peak
    max_dd = float(np.min(drawdowns)) * 100.0
    cum_net_ret = float(equity[-1] - 1.0) * 100.0
    
    days_scope = (times_test[-1] - times_test[0]) / (86400 * 1000)
    trades_per_year = (total_trades / max(days_scope, 1)) * 365.0
    mean_ret = np.mean(net_returns)
    std_ret = np.std(net_returns) if np.std(net_returns) > 0 else 1e-6
    sharpe_ann = float((mean_ret / std_ret) * math.sqrt(trades_per_year))
    
    return {
        "symbol": symbol,
        "threshold": threshold,
        "tp_mult": tp_mult,
        "sl_mult": sl_mult,
        "total_trades": total_trades,
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ann,
        "max_drawdown_pct": max_dd,
        "cumulative_net_return_pct": cum_net_ret,
        "buy_hold_return_pct": buy_hold_return * 100.0,
        "blocked_by_confluence": blocked_by_confluence,
        "trades": trades,
    }

def simulate_combined_portfolio(trade_sets: Dict[str, List[Dict]], pos_size_pct: float = 0.20, initial_capital: float = 10000.0):
    """
    Simulate a portfolio balance across all trades in chronological order.
    """
    all_events = []
    for sym, trs in trade_sets.items():
        for tr in trs:
            all_events.append({
                "symbol": sym,
                "entry_time_ms": tr["entry_time_ms"],
                "exit_time_ms": tr["exit_time_ms"],
                "net_return": tr["net_return"],
                "exit_reason": tr["exit_reason"]
            })
    # Sort events by exit time (when PnL is realized)
    all_events.sort(key=lambda x: x["exit_time_ms"])
    
    balance = initial_capital
    balance_history = [balance]
    total_trades = len(all_events)
    if total_trades == 0:
        return {"return_pct": 0.0, "max_drawdown_pct": 0.0, "win_rate_pct": 0.0, "total_trades": 0}
        
    wins = 0
    for ev in all_events:
        trade_pos_size = balance * pos_size_pct
        pnl = trade_pos_size * ev["net_return"]
        balance += pnl
        balance_history.append(balance)
        if ev["net_return"] > 0:
            wins += 1
            
    balance_arr = np.array(balance_history)
    peak = np.maximum.accumulate(balance_arr)
    dd = (balance_arr - peak) / peak
    mdd = float(np.min(dd)) * 100.0
    total_return = (balance - initial_capital) / initial_capital * 100.0
    win_rate = (wins / total_trades) * 100.0
    
    return {
        "initial_capital": initial_capital,
        "final_capital": balance,
        "return_pct": total_return,
        "max_drawdown_pct": mdd,
        "win_rate_pct": win_rate,
        "total_trades": total_trades,
        "wins": wins,
        "losses": total_trades - wins
    }

def main():
    print("=" * 80)
    print("PHASE 12: HISTORICAL BLIND REPLAY & ADVERSARIAL AUDIT")
    print("Comparison: BEFORE Calibration vs AFTER Sniper + BTC Confluence Gatekeeper")
    print("=" * 80)
    
    # Load BTC technicals & vectors for confluence checks
    btc_times, btc_X, btc_regimes, btc_klines, btc_tech = load_data("BTCUSDT")
    btc_bundle = (btc_times, btc_X, btc_tech)
    
    # ── 1. BEFORE CALIBRATION (Default Unchecked) ─────────────────────────────
    print("\n[1] Evaluating BEFORE Calibration (Default Altcoin Thresholds & Uniform ATR)...")
    before_btc = run_simulation("BTCUSDT", threshold=0.61, tp_mult=1.5, sl_mult=1.0)
    before_eth = run_simulation("ETHUSDT", threshold=0.40, tp_mult=1.5, sl_mult=1.0)
    before_sol = run_simulation("SOLUSDT", threshold=0.50, tp_mult=1.5, sl_mult=1.0)
    
    before_portfolio = simulate_combined_portfolio({
        "BTCUSDT": before_btc["trades"],
        "ETHUSDT": before_eth["trades"],
        "SOLUSDT": before_sol["trades"],
    })
    
    # ── 2. AFTER CALIBRATION (Sniper + BTC Confluence + Dynamic ATR) ──────────
    print("\n[2] Evaluating AFTER Calibration (Sniper Thresholds, Dynamic ATR & BTC Gate)...")
    after_btc = run_simulation("BTCUSDT", threshold=0.61, tp_mult=1.5, sl_mult=1.0)
    after_eth = run_simulation("ETHUSDT", threshold=0.58, tp_mult=1.8, sl_mult=1.2, use_btc_confluence=True, btc_data_bundle=btc_bundle)
    after_sol = run_simulation("SOLUSDT", threshold=0.55, tp_mult=2.2, sl_mult=1.5, use_btc_confluence=True, btc_data_bundle=btc_bundle)
    
    after_portfolio = simulate_combined_portfolio({
        "BTCUSDT": after_btc["trades"],
        "ETHUSDT": after_eth["trades"],
        "SOLUSDT": after_sol["trades"],
    })
    
    # ── 3. PRINT COMPARISON SUMMARY ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("                 PORTFOLIO OPTIMIZATION REPORT (BEFORE VS AFTER)")
    print("=" * 80)
    print(f" {'Metric':<32} | {'Before Calibration':<20} | {'After Sniper + BTC Gate':<20}")
    print("-" * 80)
    print(f" {'BTCUSDT Win Rate (OOS)':<32} | {before_btc['win_rate_pct']:>18.1f}% | {after_btc['win_rate_pct']:>18.1f}%")
    print(f" {'BTCUSDT Profit Factor':<32} | {before_btc['profit_factor']:>18.2f}  | {after_btc['profit_factor']:>18.2f} ")
    print(f" {'BTCUSDT Total Trades':<32} | {before_btc['total_trades']:>18d}  | {after_btc['total_trades']:>18d} ")
    print("-" * 80)
    print(f" {'ETHUSDT Win Rate (OOS)':<32} | {before_eth['win_rate_pct']:>18.1f}% | {after_eth['win_rate_pct']:>18.1f}%")
    print(f" {'ETHUSDT Profit Factor':<32} | {before_eth['profit_factor']:>18.2f}  | {after_eth['profit_factor']:>18.2f} ")
    print(f" {'ETHUSDT Net Return':<32} | {before_eth['cumulative_net_return_pct']:>+17.2f}% | {after_eth['cumulative_net_return_pct']:>+17.2f}%")
    print(f" {'ETHUSDT Max Drawdown':<32} | {before_eth['max_drawdown_pct']:>18.1f}% | {after_eth['max_drawdown_pct']:>18.1f}%")
    print(f" {'ETHUSDT Trades (Filtered)':<32} | {before_eth['total_trades']:>18d}  | {after_eth['total_trades']:>18d} (Blocked: {after_eth['blocked_by_confluence']})")
    print("-" * 80)
    print(f" {'SOLUSDT Win Rate (OOS)':<32} | {before_sol['win_rate_pct']:>18.1f}% | {after_sol['win_rate_pct']:>18.1f}%")
    print(f" {'SOLUSDT Profit Factor':<32} | {before_sol['profit_factor']:>18.2f}  | {after_sol['profit_factor']:>18.2f} ")
    print(f" {'SOLUSDT Net Return':<32} | {before_sol['cumulative_net_return_pct']:>+17.2f}% | {after_sol['cumulative_net_return_pct']:>+17.2f}%")
    print(f" {'SOLUSDT Max Drawdown':<32} | {before_sol['max_drawdown_pct']:>18.1f}% | {after_sol['max_drawdown_pct']:>18.1f}%")
    print(f" {'SOLUSDT Trades (Filtered)':<32} | {before_sol['total_trades']:>18d}  | {after_sol['total_trades']:>18d} (Blocked: {after_sol['blocked_by_confluence']})")
    print("=" * 80)
    print(f" {'COMBINED PORTFOLIO PERFORMANCE':<32} | {'Before Calibration':<20} | {'After Sniper + BTC Gate':<20}")
    print("-" * 80)
    print(f" {'Portfolio Initial Capital':<32} | ${before_portfolio['initial_capital']:>17,.2f} | ${after_portfolio['initial_capital']:>17,.2f}")
    print(f" {'Portfolio Final Capital':<32} | ${before_portfolio['final_capital']:>17,.2f} | ${after_portfolio['final_capital']:>17,.2f}")
    print(f" {'Portfolio Win Rate':<32} | {before_portfolio['win_rate_pct']:>18.1f}% | {after_portfolio['win_rate_pct']:>18.1f}%")
    print(f" {'Portfolio Total Trades':<32} | {before_portfolio['total_trades']:>18d}  | {after_portfolio['total_trades']:>18d} ")
    print(f" {'Portfolio Max Drawdown (MDD)':<32} | {before_portfolio['max_drawdown_pct']:>18.1f}% | {after_portfolio['max_drawdown_pct']:>18.1f}%")
    print(f" {'Portfolio Cumulative Return':<32} | {before_portfolio['return_pct']:>+17.2f}% | {after_portfolio['return_pct']:>+17.2f}%")
    print("=" * 80)
    
    # Save Report to JSON
    report_data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "before": {
            "BTCUSDT": {k: v for k, v in before_btc.items() if k != "trades"},
            "ETHUSDT": {k: v for k, v in before_eth.items() if k != "trades"},
            "SOLUSDT": {k: v for k, v in before_sol.items() if k != "trades"},
            "portfolio": before_portfolio
        },
        "after": {
            "BTCUSDT": {k: v for k, v in after_btc.items() if k != "trades"},
            "ETHUSDT": {k: v for k, v in after_eth.items() if k != "trades"},
            "SOLUSDT": {k: v for k, v in after_sol.items() if k != "trades"},
            "portfolio": after_portfolio
        }
    }
    
    report_file = Path(__file__).parent / "phase12_optimization_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n[OK] Phase 12 Report successfully written to {report_file}")

if __name__ == "__main__":
    main()
