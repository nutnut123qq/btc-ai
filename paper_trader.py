#!/usr/bin/env python3
"""
Regime-Adaptive Multi-Strategy Automated Live Paper Trading Worker
with Portfolio Circuit Breaker & Hard Risk Limits (NautilusTrader/Hummingbot standard)
=======================================================================================
Implements an institutional-grade Quantitative Trading Worker with:
1. Multi-Strategy Regime-Adaptive Router (Momentum Trend-Following & Mean-Reversion).
2. Portfolio Circuit Breaker & Kill-Switch:
   - 24h Rolling Drawdown Limit (MAX_DAILY_DRAWDOWN = 4.0%)
   - Consecutive Loss Limit (MAX_CONSECUTIVE_LOSSES = 4)
   - BTC Flash Crash / Black Swan Volatility Halt (VOLATILITY_HALT_THRESHOLD = 8.0%)
   - Automated Position Liquidation on Breaker Trigger & Cooldown enforcement (24h)
   - Realtime Alert dispatching to AppAlerts
"""

import json
import math
import os
import sys
import urllib.request
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional, List, Tuple, Dict

import joblib
import numpy as np
import psycopg2

# Add ai/ to python path
AI_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(AI_DIR))

from db_config import get_db_params, get_db_connection
from prediction_service import load_model
from trading_config import (
    DEFAULT_SYMBOL,
    ACTIVE_SYMBOLS,
    ASSET_4H_THRESHOLDS,
    ASSET_ATR_SETTINGS,
    FEE_BPS,
    SLIPPAGE_BPS,
    TIMEFRAME_THRESHOLDS,
    INITIAL_BALANCE_USDT,
    POSITION_SIZE_PCT,
    MAX_CONCURRENT_PER_SYMBOL,
    ATR_TP_MULTIPLIER,
    ATR_SL_MULTIPLIER,
    MAX_HOLD_BARS,
    BACKEND_BASE_URL,
    MAX_KELLY_FRACTION,
    MIN_KELLY_FRACTION,
    KELLY_SAFETY_FACTOR,
    TRAILING_STOP_ATR_TRIGGER,
    TRAILING_STOP_ATR_DIST,
    REGIME_ADX_THRESHOLD,
    MEAN_REVERSION_POSITION_PCT,
    MEAN_REVERSION_RSI_OVERSOLD,
    MEAN_REVERSION_RSI_OVERBOUGHT,
    MEAN_REVERSION_TP_ATR_MULT,
    MEAN_REVERSION_SL_ATR_MULT,
    TREND_MOMENTUM_TP_ATR_MULT,
    TREND_MOMENTUM_SL_ATR_MULT,
    MAX_DAILY_DRAWDOWN,
    VOLATILITY_HALT_THRESHOLD,
    MAX_CONSECUTIVE_LOSSES,
    CIRCUIT_BREAKER_COOLDOWN_HOURS,
    ALTCOIN_VOLATILITY_HALT_BARS,
)

MODELS_DIR = AI_DIR / "models"

FEATURE_COLS = [
    "CloseZscore", "ClosePctChange1", "ClosePctChange4", "ClosePctChange24",
    "HighLowRangePct", "BodyPct", "UpperWickPct", "LowerWickPct",
    "Rsi14", "Rsi14Slope", "MacdNorm", "MacdSignalNorm", "MacdHistogramNorm",
    "Ema12Dist", "Ema26Dist", "Ema50Dist", "Ema200Dist", "Sma50Dist", "Sma200Dist",
    "BollingerWidth", "BollingerPosition", "Atr14Pct", "ObvEmaDist", "VwapDist",
    "RollingVwapDist", "VolumeZscore", "VolumeSma20Ratio", "TakerBuyRatio",
    "RecentPatternEncoded", "ActiveRuleCount",
]

_MODEL_CACHE: dict[str, Any] = {}


def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    try:
        print(f"[{ts}] {msg}")
    except UnicodeEncodeError:
        print(f"[{ts}] {msg.encode('ascii', 'replace').decode()}")


def get_conn():
    return get_db_connection()


def ensure_schema(conn, cur):
    """Ensure PaperTrades schema includes StrategyType and related columns."""
    cur.execute("""
        ALTER TABLE "PaperTrades" ADD COLUMN IF NOT EXISTS "StrategyType" character varying(50);
    """)
    conn.commit()


def push_system_alert(cur, conn, alert_type: str, title: str, message: str, price_snapshot: float = 0.0):
    """Dispatch system alert into AppAlerts table."""
    alert_id = str(uuid.uuid4())
    cur.execute(
        """INSERT INTO "AppAlerts" ("Id", "UserId", "Type", "Title", "Message", "PriceSnapshot", "CreatedAt", "IsRead")
           VALUES (%s, 'default', %s, %s, %s, %s, NOW(), FALSE)""",
        (alert_id, alert_type, title, message, price_snapshot)
    )
    conn.commit()


def get_model_for_symbol(symbol: str) -> Tuple[Optional[Any], str]:
    """Load active model from registry or fallback."""
    try:
        model, meta = load_model(symbol, "4h", 5, "4h")
        model_name = meta.get("model_name") or f"{symbol}_4h_ws5_h4h_XGB_active"
        return model, model_name
    except Exception as e:
        log(f"[WARN] Failed to load active model from registry for {symbol}: {e}. Trying fallback.")
        fallback_file = f"{symbol}_4h_ws5_h4h_XGB_calibrated.joblib"
        path = MODELS_DIR / fallback_file
        if path.exists():
            if fallback_file not in _MODEL_CACHE:
                _MODEL_CACHE[fallback_file] = joblib.load(path)
            return _MODEL_CACHE[fallback_file], fallback_file
        return None, fallback_file


def time_features(open_ms: int) -> list[float]:
    dt = datetime.fromtimestamp(open_ms / 1000, timezone.utc)
    hour = dt.hour + dt.minute / 60.0
    dow = dt.weekday()
    return [
        math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
        math.sin(2 * math.pi * dow / 7), math.cos(2 * math.pi * dow / 7),
        1.0 if dow >= 5 else 0.0,
    ]


# ── Feature & Technical Indicator Extraction ────────────────────────────────

def build_vector_at(cur, symbol: str, timeframe: str, window_size: int, tf_ms: int, end_open_time_ms: int):
    """Build feature vector from MlFeatureStores for the window ending at end_open_time_ms."""
    cols = ", ".join(f'"{c}"' for c in FEATURE_COLS)
    cur.execute(
        f"""SELECT "OpenTimeMs", {cols} FROM "MlFeatureStores"
            WHERE "Symbol"=%s AND "Timeframe"=%s AND "OpenTimeMs" <= %s
            ORDER BY "OpenTimeMs" DESC LIMIT %s""",
        (symbol, timeframe, end_open_time_ms, window_size))
    rows = cur.fetchall()
    if len(rows) < window_size:
        return None
    rows = list(reversed(rows))
    for i in range(1, len(rows)):
        if rows[i][0] - rows[i - 1][0] != tf_ms:
            return None
    vector = []
    for r in rows:
        vals = r[1:]
        core_vals = vals[:28]
        if any(v is None for v in core_vals):
            return None
        vector.extend(float(v) if v is not None else 0.0 for v in vals)
        vector.extend(time_features(r[0]))
    return rows[-1][0], np.array(vector, dtype=np.float32)


def compute_atr14(cur, symbol: str, timeframe: str, before_ms: int) -> Optional[float]:
    cur.execute(
        """SELECT "Open", "High", "Low", "Close" FROM "Klines"
           WHERE "Symbol"=%s AND "Timeframe"=%s AND "OpenTimeMs" <= %s
           ORDER BY "OpenTimeMs" DESC LIMIT 15""",
        (symbol, timeframe, before_ms))
    rows = cur.fetchall()
    if len(rows) < 15:
        return None
    rows = list(reversed(rows))
    tr_values = []
    for i in range(1, len(rows)):
        high = float(rows[i][1])
        low = float(rows[i][2])
        prev_close = float(rows[i - 1][3])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_values.append(tr)
    return sum(tr_values) / len(tr_values) if tr_values else None


def compute_adx14(cur, symbol: str, timeframe: str, before_ms: int, period: int = 14) -> Tuple[float, bool]:
    """Computes standard Wilder's ADX(14) from Klines."""
    cur.execute(
        """SELECT "OpenTimeMs", "High", "Low", "Close" FROM "Klines"
           WHERE "Symbol"=%s AND "Timeframe"=%s AND "OpenTimeMs" <= %s
           ORDER BY "OpenTimeMs" DESC LIMIT %s""",
        (symbol, timeframe, before_ms, period * 3)
    )
    rows = cur.fetchall()
    if len(rows) < period * 2:
        return 20.0, False
    rows = list(reversed(rows))
    highs = np.array([float(r[1]) for r in rows], dtype=np.float64)
    lows = np.array([float(r[2]) for r in rows], dtype=np.float64)
    closes = np.array([float(r[3]) for r in rows], dtype=np.float64)

    tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    up_move = highs[1:] - highs[:-1]
    down_move = lows[:-1] - lows[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_smooth = np.zeros(len(tr))
    plus_dm_smooth = np.zeros(len(tr))
    minus_dm_smooth = np.zeros(len(tr))

    tr_smooth[period - 1] = np.sum(tr[:period])
    plus_dm_smooth[period - 1] = np.sum(plus_dm[:period])
    minus_dm_smooth[period - 1] = np.sum(minus_dm[:period])

    for i in range(period, len(tr)):
        tr_smooth[i] = tr_smooth[i - 1] - (tr_smooth[i - 1] / period) + tr[i]
        plus_dm_smooth[i] = plus_dm_smooth[i - 1] - (plus_dm_smooth[i - 1] / period) + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i - 1] - (minus_dm_smooth[i - 1] / period) + minus_dm[i]

    plus_di = 100.0 * (plus_dm_smooth[period - 1:] / np.maximum(tr_smooth[period - 1:], 1e-9))
    minus_di = 100.0 * (minus_dm_smooth[period - 1:] / np.maximum(tr_smooth[period - 1:], 1e-9))
    dx = 100.0 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-9)

    if len(dx) < period:
        return float(np.mean(dx)) if len(dx) > 0 else 20.0, True

    adx = np.zeros(len(dx))
    adx[period - 1] = np.mean(dx[:period])
    for i in range(period, len(dx)):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return float(adx[-1]), True


def get_technical_indicators_at(cur, symbol: str, timeframe: str, bar_open_ms: int) -> Optional[Dict[str, Any]]:
    """Fetch RSI, Bollinger Bands, and ATR from TechnicalIndicators table."""
    cur.execute(
        """SELECT "OpenTimeMs", "Rsi14", "BollingerUpper", "BollingerMiddle", "BollingerLower", "Atr14"
           FROM "TechnicalIndicators"
           WHERE "Symbol"=%s AND "Timeframe"=%s AND "OpenTimeMs" <= %s
           ORDER BY "OpenTimeMs" DESC LIMIT 10""",
        (symbol, timeframe, bar_open_ms)
    )
    rows = cur.fetchall()
    if not rows or rows[0][0] != bar_open_ms:
        return None

    r = rows[0]
    rsi = float(r[1]) if r[1] is not None else 50.0
    bb_upper = float(r[2]) if r[2] is not None else 0.0
    bb_middle = float(r[3]) if r[3] is not None else 0.0
    bb_lower = float(r[4]) if r[4] is not None else 0.0
    atr = float(r[5]) if r[5] is not None else 0.0

    bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0.0

    prev_widths = []
    for pr in rows[1:]:
        if pr[2] and pr[3] and pr[4] and float(pr[3]) > 0:
            pw = (float(pr[2]) - float(pr[4])) / float(pr[3])
            prev_widths.append(pw)

    avg_prev_width = sum(prev_widths) / len(prev_widths) if prev_widths else bb_width
    is_bb_expanding = bb_width >= avg_prev_width

    return {
        "rsi14": rsi,
        "bb_upper": bb_upper,
        "bb_middle": bb_middle,
        "bb_lower": bb_lower,
        "bb_width": bb_width,
        "is_bb_expanding": is_bb_expanding,
        "atr14": atr,
    }


def classify_market_regime(
    cur,
    symbol: str,
    timeframe: str,
    bar_open_ms: int,
    adx_threshold: float = REGIME_ADX_THRESHOLD,
) -> Tuple[str, Dict[str, Any]]:
    """
    Market Regime Classifier:
    - 'TRENDING_REGIME': ADX(14) > 25 OR (ADX > 20 and Bollinger Bands expanding).
    - 'CHOPPY_SIDEWAYS_REGIME': ADX(14) <= 25 and narrow/contracting BB width.
    """
    adx, adx_ok = compute_adx14(cur, symbol, timeframe, bar_open_ms, period=14)
    tech = get_technical_indicators_at(cur, symbol, timeframe, bar_open_ms)

    is_expanding = tech["is_bb_expanding"] if tech else False
    bb_width = tech["bb_width"] if tech else 0.0

    if adx > adx_threshold or (adx > 20.0 and is_expanding):
        regime = "TRENDING_REGIME"
    else:
        regime = "CHOPPY_SIDEWAYS_REGIME"

    info = {
        "regime": regime,
        "adx14": round(adx, 2),
        "is_bb_expanding": is_expanding,
        "bb_width": round(bb_width, 4),
        "tech": tech,
    }
    return regime, info


# ── Ensemble & Confluence Filters ───────────────────────────────────────────

def get_ensemble_direction(symbol: str) -> Optional[str]:
    try:
        url = f"{BACKEND_BASE_URL}/api/ensemble/predict?symbol={symbol}&timeframe=4h"
        req = urllib.request.Request(url, headers={"User-Agent": "PaperTrader/1.0"})
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read().decode())
            return data.get("finalDirection", None)
    except Exception:
        return None


def check_btc_confluence(cur, bar_open_ms: int, target_side: str) -> Tuple[bool, str]:
    """BTC Confluence Gatekeeper for Altcoin trades (ETH, SOL)."""
    cur.execute(
        """SELECT "Rsi14" FROM "TechnicalIndicators"
           WHERE "Symbol"='BTCUSDT' AND "Timeframe"='4h' AND "OpenTimeMs"=%s""",
        (bar_open_ms,))
    row = cur.fetchone()
    btc_rsi = float(row[0]) if (row and row[0] is not None) else 50.0

    btc_model, _ = get_model_for_symbol("BTCUSDT")
    if btc_model is None:
        return True, ""

    res = build_vector_at(cur, "BTCUSDT", "4h", 5, 14_400_000, bar_open_ms)
    if not res:
        return True, ""

    _, btc_vec = res
    btc_probas = btc_model.predict_proba(btc_vec.reshape(1, -1))[0]
    p_down = float(btc_probas[0])
    p_up = float(btc_probas[2])

    if target_side == "long":
        if p_down >= 0.40:
            return False, f"BTC P(Down)={p_down*100:.1f}% >= 40%"
        if btc_rsi <= 45.0:
            return False, f"BTC RSI={btc_rsi:.1f} <= 45"
    elif target_side == "short":
        if p_up >= 0.40:
            return False, f"BTC P(Up)={p_up*100:.1f}% >= 40%"
        if btc_rsi >= 65.0:
            return False, f"BTC RSI={btc_rsi:.1f} >= 65"

    return True, ""


# ── Position Sizing ─────────────────────────────────────────────────────────

def get_portfolio_balance(cur) -> float:
    """Calculates overall virtual portfolio balance across all closed trades."""
    cur.execute(
        """SELECT "PositionSizeUsdt", "NetReturn" FROM "PaperTrades"
           WHERE "Status"='closed'
           AND "PositionSizeUsdt" IS NOT NULL AND "NetReturn" IS NOT NULL""")
    rows = cur.fetchall()
    balance = INITIAL_BALANCE_USDT
    for pos_size, net_return in rows:
        balance += float(pos_size) * float(net_return)
    return balance


def compute_quarter_kelly_size(
    balance: float,
    win_prob: float,
    tp_mult: float,
    sl_mult: float,
    min_pct: float = MIN_KELLY_FRACTION,
    max_pct: float = MAX_KELLY_FRACTION,
    safety: float = KELLY_SAFETY_FACTOR,
) -> Tuple[float, float]:
    """Quarter-Kelly Position Sizing."""
    if sl_mult <= 0 or balance <= 0:
        return balance * min_pct, min_pct
    b = tp_mult / sl_mult
    p = win_prob
    q = 1.0 - p
    f_star = (b * p - q) / b
    if f_star <= 0:
        frac = min_pct
    else:
        frac = min(max(f_star * safety, min_pct), max_pct)
    size_usdt = balance * frac
    return size_usdt, frac


def get_kline(cur, symbol: str, timeframe: str, open_ms: int):
    cur.execute(
        """SELECT "Open", "High", "Low", "Close", "Volume" FROM "Klines"
           WHERE "Symbol"=%s AND "Timeframe"=%s AND "OpenTimeMs"=%s""",
        (symbol, timeframe, open_ms))
    return cur.fetchone()


# ── Portfolio Circuit Breaker & Risk Constraints ────────────────────────────

def evaluate_portfolio_circuit_breaker(cur, conn, bar_open_ms: int) -> Tuple[bool, str]:
    """
    Evaluates Circuit Breaker & Hard Risk Constraints:
    1. 24h Rolling Drawdown >= MAX_DAILY_DRAWDOWN (4.0%)
    2. Consecutive Losses >= MAX_CONSECUTIVE_LOSSES (4)
    Returns: (is_triggered, reason_message)
    """
    # 1. 24h Rolling Window Drawdown Check
    day_window_ms = 24 * 3600 * 1000
    start_24h_ms = bar_open_ms - day_window_ms

    cur.execute(
        """SELECT "PositionSizeUsdt", "NetReturn" FROM "PaperTrades"
           WHERE "Status"='closed' AND "ExitTimeMs" >= %s AND "ExitTimeMs" <= %s
           AND "PositionSizeUsdt" IS NOT NULL AND "NetReturn" IS NOT NULL""",
        (start_24h_ms, bar_open_ms)
    )
    closed_24h = cur.fetchall()
    pnl_24h = sum(float(pos) * float(ret) for pos, ret in closed_24h)
    curr_balance = get_portfolio_balance(cur)
    balance_24h_ago = curr_balance - pnl_24h

    if balance_24h_ago > 0 and pnl_24h < 0:
        dd_pct = abs(pnl_24h) / balance_24h_ago
        if dd_pct >= MAX_DAILY_DRAWDOWN:
            reason = f"24h Daily Drawdown {dd_pct*100:.2f}% exceeded limit of {MAX_DAILY_DRAWDOWN*100:.1f}% (PnL: ${pnl_24h:+.2f})"
            return True, reason

    # 2. Consecutive Losses Check
    cur.execute(
        """SELECT "NetReturn" FROM "PaperTrades"
           WHERE "Status"='closed' AND "ExitTimeMs" <= %s AND "NetReturn" IS NOT NULL
           ORDER BY "ExitTimeMs" DESC, "Id" DESC LIMIT %s""",
        (bar_open_ms, MAX_CONSECUTIVE_LOSSES)
    )
    recent_rets = [float(r[0]) for r in cur.fetchall()]
    if len(recent_rets) >= MAX_CONSECUTIVE_LOSSES and all(r < 0 for r in recent_rets):
        reason = f"Hit {len(recent_rets)} consecutive losing trades (Limit: {MAX_CONSECUTIVE_LOSSES})"
        return True, reason

    return False, ""


def trigger_circuit_breaker_halt(conn, cur, bar_open_ms: int, reason: str) -> int:
    """
    Executes Circuit Breaker Kill-Switch:
    1. Closes ALL currently open positions at current bar Close with ExitReason='CIRCUIT_BREAKER'.
    2. Updates portfolio balance.
    3. Dispatches high-priority alert to AppAlerts.
    Returns: cooldown_until_ms
    """
    cooldown_ms = CIRCUIT_BREAKER_COOLDOWN_HOURS * 3600 * 1000
    halt_until_ms = bar_open_ms + cooldown_ms
    halt_dt = datetime.fromtimestamp(halt_until_ms / 1000, timezone.utc)

    # Close all open positions across all symbols
    cur.execute("""SELECT "Id", "Symbol", "EntryPrice", "PositionSizeUsdt", "Side" FROM "PaperTrades" WHERE "Status"='open'""")
    open_positions = cur.fetchall()

    for tid, sym, ep, pos_size, side in open_positions:
        kl = get_kline(cur, sym, "4h", bar_open_ms)
        exit_price = float(kl[3]) if kl else float(ep)
        ep_val = float(ep)
        pos_val = float(pos_size)
        fee = FEE_BPS / 1e4
        slip = SLIPPAGE_BPS / 1e4

        if side == "long":
            gross_ret = (exit_price * (1 - slip) - ep_val * (1 + slip)) / (ep_val * (1 + slip))
        else:
            gross_ret = (ep_val * (1 - slip) - exit_price * (1 + slip)) / (ep_val * (1 + slip))
        net_ret = gross_ret - 2 * fee

        cur.execute(
            """UPDATE "PaperTrades"
               SET "Status"='closed', "ExitPrice"=%s, "ExitTimeMs"=%s,
                   "NetReturn"=%s, "ExitReason"='CIRCUIT_BREAKER', "ClosedAtUtc"=NOW()
               WHERE "Id"=%s""",
            (exit_price, bar_open_ms, net_ret, tid))
        conn.commit()

    curr_bal = get_portfolio_balance(cur)
    cur.execute("""UPDATE "PaperTrades" SET "BalanceAfter"=%s WHERE "Status"='closed' AND "BalanceAfter" IS NULL""", (curr_bal,))
    conn.commit()

    alert_title = "[CIRCUIT BREAKER ACTIVATED]"
    alert_msg = f"System transitioned to HALTED_CIRCUIT_BREAKER. Reason: {reason}. Liquidated {len(open_positions)} open positions. Trading paused until {halt_dt.isoformat()}."
    push_system_alert(cur, conn, "CIRCUIT_BREAKER", alert_title, alert_msg, curr_bal)

    log("=" * 80)
    log(f"🚨 [CIRCUIT BREAKER ACTIVATED] {reason}")
    log(f"   Liquidated {len(open_positions)} open positions | Current Capital: ${curr_bal:,.2f}")
    log(f"   Status: HALTED_CIRCUIT_BREAKER until {halt_dt.strftime('%Y-%m-%d %H:%M:%S UTC')} ({CIRCUIT_BREAKER_COOLDOWN_HOURS}h cooldown)")
    log("=" * 80)

    return halt_until_ms


def check_btc_volatility_halt(cur, conn, bar_open_ms: int) -> Tuple[bool, str, float]:
    """
    Checks if BTC experienced a Flash Crash / Black Swan candle (> 8.0% amplitude/body).
    """
    kl = get_kline(cur, "BTCUSDT", "4h", bar_open_ms)
    if not kl:
        return False, "", 0.0

    b_open, b_high, b_low, b_close, _ = [float(x) for x in kl]
    if b_open <= 0:
        return False, "", 0.0

    amplitude = (b_high - b_low) / b_open
    body = abs(b_close - b_open) / b_open
    max_vol = max(amplitude, body)

    if max_vol >= VOLATILITY_HALT_THRESHOLD:
        reason = f"BTC 4h Candle Amplitude/Body {max_vol*100:.2f}% >= {VOLATILITY_HALT_THRESHOLD*100:.1f}% (Open: ${b_open:,.2f}, High: ${b_high:,.2f}, Low: ${b_low:,.2f}, Close: ${b_close:,.2f})"
        return True, reason, max_vol

    return False, "", max_vol


# ── Step Single Symbol Bar with Adaptive Strategy Routing ───────────────────

def step_symbol_bar(conn, cur, symbol: str, bar_open_ms: int):
    """Executes a single bar step for a symbol with trailing stops and strategy routing."""
    tf = "4h"
    tf_ms = 14_400_000
    fee = FEE_BPS / 1e4
    slip = SLIPPAGE_BPS / 1e4

    kl = get_kline(cur, symbol, tf, bar_open_ms)
    if not kl:
        return
    bar_open, bar_high, bar_low, bar_close, bar_vol = [float(x) for x in kl]

    # 1. Manage open trades for this symbol
    cur.execute(
        """SELECT "Id", "Side", "EntryPrice", "EntryTimeMs", "TakeProfitPrice",
                  "StopLossPrice", "PositionSizeUsdt", "Atr14", "StrategyType"
           FROM "PaperTrades"
           WHERE "Symbol"=%s AND "Timeframe"=%s AND "Status"='open'""",
        (symbol, tf))
    open_trades = cur.fetchall()

    for tid, side, entry_price, entry_ms, tp_price, sl_price, pos_size, atr_val, strat_type in open_trades:
        entry_price = float(entry_price)
        tp_price = float(tp_price) if tp_price else None
        sl_price = float(sl_price) if sl_price else None
        pos_size = float(pos_size) if pos_size else 0.0
        trade_atr = float(atr_val) if (atr_val is not None and float(atr_val) > 0) else (compute_atr14(cur, symbol, tf, bar_open_ms) or 0.0)

        # Dynamic Trailing Stop for TREND_MOMENTUM
        if strat_type == "TREND_MOMENTUM" and trade_atr > 0:
            if side == "long":
                if bar_high >= entry_price + TRAILING_STOP_ATR_TRIGGER * trade_atr:
                    candidate_sl = max(entry_price, bar_high - TRAILING_STOP_ATR_DIST * trade_atr)
                    if sl_price is None or candidate_sl > sl_price:
                        sl_price = candidate_sl
                        cur.execute("""UPDATE "PaperTrades" SET "StopLossPrice"=%s WHERE "Id"=%s""", (sl_price, tid))
                        conn.commit()
                        log(f"[{symbol}] TRAILING SL UPDATED: #{tid} (LONG) -> ${sl_price:,.2f} (+1.0x ATR Profit Lock)")
            elif side == "short":
                if bar_low <= entry_price - TRAILING_STOP_ATR_TRIGGER * trade_atr:
                    candidate_sl = min(entry_price, bar_low + TRAILING_STOP_ATR_DIST * trade_atr)
                    if sl_price is None or candidate_sl < sl_price:
                        sl_price = candidate_sl
                        cur.execute("""UPDATE "PaperTrades" SET "StopLossPrice"=%s WHERE "Id"=%s""", (sl_price, tid))
                        conn.commit()
                        log(f"[{symbol}] TRAILING SL UPDATED: #{tid} (SHORT) -> ${sl_price:,.2f} (+1.0x ATR Profit Lock)")

        exit_price = None
        exit_reason = None

        # Check Stop Loss
        if sl_price:
            if side == "long" and bar_low <= sl_price:
                exit_price = sl_price
                exit_reason = "TRAILING_SL" if sl_price >= entry_price else "SL"
            elif side == "short" and bar_high >= sl_price:
                exit_price = sl_price
                exit_reason = "TRAILING_SL" if sl_price <= entry_price else "SL"

        # Check Take Profit
        if exit_price is None and tp_price:
            if side == "long" and bar_high >= tp_price:
                exit_price = tp_price
                exit_reason = "TP"
            elif side == "short" and bar_low <= tp_price:
                exit_price = tp_price
                exit_reason = "TP"

        # Time horizon timeout (6 bars = 24h)
        bars_held = (bar_open_ms - entry_ms) // tf_ms
        if exit_price is None and bars_held >= MAX_HOLD_BARS:
            exit_price = bar_close
            exit_reason = "TIMEOUT"

        if exit_price is not None:
            if side == "long":
                gross_ret = (exit_price * (1 - slip) - entry_price * (1 + slip)) / (entry_price * (1 + slip))
            else:
                gross_ret = (entry_price * (1 - slip) - exit_price * (1 + slip)) / (entry_price * (1 + slip))
            net_ret = gross_ret - 2 * fee
            pnl_usdt = pos_size * net_ret

            cur.execute(
                """UPDATE "PaperTrades"
                   SET "Status"='closed', "ExitPrice"=%s, "ExitTimeMs"=%s,
                       "NetReturn"=%s, "ExitReason"=%s, "ClosedAtUtc"=NOW()
                   WHERE "Id"=%s""",
                (exit_price, bar_open_ms, net_ret, exit_reason, tid))
            conn.commit()

            balance_now = get_portfolio_balance(cur)
            cur.execute("""UPDATE "PaperTrades" SET "BalanceAfter"=%s WHERE "Id"=%s""",
                        (balance_now, tid))
            conn.commit()

            log(f"[{symbol}] CLOSED #{tid} ({strat_type} {side.upper()}) | Exit=${exit_price:,.2f} ({exit_reason}) | Net={net_ret*100:+.2f}% | PnL=${pnl_usdt:+.2f} | Balance=${balance_now:,.2f}")

    # 2. Check position limit and duplicate bar guard
    cur.execute(
        """SELECT COUNT(*) FROM "PaperTrades"
           WHERE "Symbol"=%s AND "Timeframe"=%s AND "Status"='open'""",
        (symbol, tf))
    if cur.fetchone()[0] >= MAX_CONCURRENT_PER_SYMBOL:
        return

    cur.execute(
        """SELECT COUNT(*) FROM "PaperTrades"
           WHERE "Symbol"=%s AND "Timeframe"=%s AND "WindowEndMs"=%s""",
        (symbol, tf, bar_open_ms))
    if cur.fetchone()[0] > 0:
        return

    # 3. Market Regime Classification
    regime, regime_info = classify_market_regime(cur, symbol, tf, bar_open_ms)
    tech = regime_info.get("tech")
    atr = tech["atr14"] if (tech and tech.get("atr14", 0) > 0) else (compute_atr14(cur, symbol, tf, bar_open_ms) or 0.0)

    # ═════════════════════════════════════════════════════════════════════════
    # STRATEGY 1: Momentum Trend-Following (Active in TRENDING_REGIME)
    # ═════════════════════════════════════════════════════════════════════════
    if regime == "TRENDING_REGIME":
        model, model_file = get_model_for_symbol(symbol)
        if model is None:
            return

        res = build_vector_at(cur, symbol, tf, 5, tf_ms, bar_open_ms)
        if not res:
            return
        _, feat_vec = res

        threshold = ASSET_4H_THRESHOLDS.get(symbol, 0.58)
        probas = model.predict_proba(feat_vec.reshape(1, -1))[0]
        pred_idx = int(np.argmax(probas))
        conf = float(probas[pred_idx])

        # 0: Down (Short), 1: Sideways, 2: Up (Long)
        if pred_idx == 1 or conf < threshold:
            return

        side = "long" if pred_idx == 2 else "short"

        # Confluence & Ensemble filters
        if symbol != "BTCUSDT":
            confluence_ok, conf_reason = check_btc_confluence(cur, bar_open_ms, side)
            if not confluence_ok:
                log(f"[{symbol}] [TREND_MOMENTUM] SKIP {side.upper()} at {bar_open_ms} -> BTC Confluence Block ({conf_reason})")
                return

        ens_dir = get_ensemble_direction(symbol)
        if ens_dir:
            if side == "long" and ens_dir == "Bearish":
                log(f"[{symbol}] [TREND_MOMENTUM] SKIP LONG at {bar_open_ms} -> Bearish Ensemble")
                return
            if side == "short" and ens_dir == "Bullish":
                log(f"[{symbol}] [TREND_MOMENTUM] SKIP SHORT at {bar_open_ms} -> Bullish Ensemble")
                return

        entry_price = bar_close
        tp_mult = TREND_MOMENTUM_TP_ATR_MULT
        sl_mult = TREND_MOMENTUM_SL_ATR_MULT

        if atr and atr > 0:
            if side == "long":
                tp_price = entry_price + tp_mult * atr
                sl_price = entry_price - sl_mult * atr
            else:
                tp_price = entry_price - tp_mult * atr
                sl_price = entry_price + sl_mult * atr
        else:
            tp_price, sl_price = None, None

        curr_bal = get_portfolio_balance(cur)
        pos_size, kelly_frac = compute_quarter_kelly_size(curr_bal, conf, tp_mult, sl_mult)

        cur.execute(
            """INSERT INTO "PaperTrades" (
                   "Symbol", "Timeframe", "WindowEndMs", "EntryTimeMs", "ExitTimeMs", "EntryPrice",
                   "Side", "Status", "Confidence", "ModelVersion", "PositionSizeUsdt",
                   "TakeProfitPrice", "StopLossPrice", "Atr14", "EnsembleDirection", "StrategyType", "CreatedAtUtc"
               ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'open', %s, %s, %s, %s, %s, %s, %s, %s, NOW())
               RETURNING "Id" """,
            (symbol, tf, bar_open_ms, bar_open_ms, 0, entry_price, side, conf,
             model_file, pos_size, tp_price, sl_price, atr, ens_dir, "TREND_MOMENTUM"))
        new_id = cur.fetchone()[0]
        conn.commit()

        qty = pos_size / entry_price if entry_price > 0 else 0.0
        log(f"[{symbol}] OPENED [TREND_MOMENTUM] Trade #{new_id} ({side.upper()}) | Entry=${entry_price:,.2f} | Conf={conf*100:.1f}% | ADX={regime_info['adx14']:.1f} | Kelly={kelly_frac*100:.1f}% (${pos_size:,.2f}) | TP=${tp_price or 0:,.2f} | SL=${sl_price or 0:,.2f}")

    # ═════════════════════════════════════════════════════════════════════════
    # STRATEGY 2: Mean-Reversion Choppy Engine (Active in CHOPPY_SIDEWAYS_REGIME)
    # ═════════════════════════════════════════════════════════════════════════
    elif regime == "CHOPPY_SIDEWAYS_REGIME":
        if not tech or tech["bb_upper"] == 0 or tech["bb_lower"] == 0:
            return

        rsi = tech["rsi14"]
        bb_upper = tech["bb_upper"]
        bb_middle = tech["bb_middle"]
        bb_lower = tech["bb_lower"]
        entry_price = bar_close

        side = None
        # Mean Reversion Long: Price <= Lower BB and RSI < 30 (Oversold bounce)
        if (bar_close <= bb_lower or bar_low <= bb_lower) and rsi < MEAN_REVERSION_RSI_OVERSOLD:
            side = "long"
            tp_price = bb_middle if bb_middle > entry_price else (entry_price + MEAN_REVERSION_TP_ATR_MULT * atr if atr else entry_price * 1.02)
            sl_price = entry_price - MEAN_REVERSION_SL_ATR_MULT * atr if atr else entry_price * 0.98

        # Mean Reversion Short: Price >= Upper BB and RSI > 70 (Overbought reversal)
        elif (bar_close >= bb_upper or bar_high >= bb_upper) and rsi > MEAN_REVERSION_RSI_OVERBOUGHT:
            side = "short"
            tp_price = bb_middle if bb_middle < entry_price else (entry_price - MEAN_REVERSION_TP_ATR_MULT * atr if atr else entry_price * 0.98)
            sl_price = entry_price + MEAN_REVERSION_SL_ATR_MULT * atr if atr else entry_price * 1.02

        if side is None:
            return

        curr_bal = get_portfolio_balance(cur)
        pos_size = curr_bal * MEAN_REVERSION_POSITION_PCT  # Fixed 8% NAV

        cur.execute(
            """INSERT INTO "PaperTrades" (
                   "Symbol", "Timeframe", "WindowEndMs", "EntryTimeMs", "ExitTimeMs", "EntryPrice",
                   "Side", "Status", "Confidence", "ModelVersion", "PositionSizeUsdt",
                   "TakeProfitPrice", "StopLossPrice", "Atr14", "EnsembleDirection", "StrategyType", "CreatedAtUtc"
               ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'open', %s, %s, %s, %s, %s, %s, %s, %s, NOW())
               RETURNING "Id" """,
            (symbol, tf, bar_open_ms, bar_open_ms, 0, entry_price, side, 1.0,
             "MeanReversionEngine", pos_size, tp_price, sl_price, atr, None, "MEAN_REVERSION"))
        new_id = cur.fetchone()[0]
        conn.commit()

        qty = pos_size / entry_price if entry_price > 0 else 0.0
        log(f"[{symbol}] OPENED [MEAN_REVERSION] Trade #{new_id} ({side.upper()}) | Entry=${entry_price:,.2f} | RSI={rsi:.1f} | ADX={regime_info['adx14']:.1f} | Fixed NAV={MEAN_REVERSION_POSITION_PCT*100:.0f}% (${pos_size:,.2f}) | TP=${tp_price:,.2f} | SL=${sl_price:,.2f}")


# ── Multi-Asset Execution Loop with Circuit Breaker ─────────────────────────

def run_multi_asset_loop(symbols: List[str] = None, start_ms: Optional[int] = None, end_ms: Optional[int] = None):
    symbols = symbols or ACTIVE_SYMBOLS
    conn = get_conn()
    cur = conn.cursor()

    ensure_schema(conn, cur)

    log("=" * 75)
    log(f"STARTING REGIME-ADAPTIVE MULTI-STRATEGY PAPER TRADER: {', '.join(symbols)}")
    log("=" * 75)

    tf = "4h"

    if start_ms is None:
        start_ms = int(datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)

    if end_ms is None:
        cur.execute("""SELECT MAX("OpenTimeMs") FROM "Klines" WHERE "Timeframe"='4h'""")
        end_ms = cur.fetchone()[0]

    # Get distinct 4h timestamps in range
    cur.execute(
        """SELECT DISTINCT "OpenTimeMs" FROM "Klines"
           WHERE "Timeframe"=%s AND "OpenTimeMs" >= %s AND "OpenTimeMs" <= %s
           ORDER BY "OpenTimeMs" ASC""",
        (tf, start_ms, end_ms))
    bar_times = [r[0] for r in cur.fetchall()]

    log(f"Evaluating {len(bar_times)} 4h cycles across {len(symbols)} symbols from {datetime.fromtimestamp(start_ms/1000, timezone.utc)} to {datetime.fromtimestamp(end_ms/1000, timezone.utc)}")

    circuit_breaker_halt_until_ms = 0
    altcoin_volatility_halt_until_ms = 0

    for t in bar_times:
        # Check active Circuit Breaker status
        if t < circuit_breaker_halt_until_ms:
            t_dt = datetime.fromtimestamp(t / 1000, timezone.utc)
            halt_dt = datetime.fromtimestamp(circuit_breaker_halt_until_ms / 1000, timezone.utc)
            log(f"⏸️ [HALTED_CIRCUIT_BREAKER] Bar {t_dt.strftime('%Y-%m-%d %H:%M')} skipped -> Circuit breaker cooldown active until {halt_dt.strftime('%Y-%m-%d %H:%M')}")
            continue

        # Check BTC Volatility Flash Crash condition
        is_flash_crash, flash_reason, vol_val = check_btc_volatility_halt(cur, conn, t)
        if is_flash_crash:
            altcoin_volatility_halt_until_ms = t + ALTCOIN_VOLATILITY_HALT_BARS * 14_400_000
            halt_dt = datetime.fromtimestamp(altcoin_volatility_halt_until_ms / 1000, timezone.utc)
            log(f"⚠️ [VOLATILITY HALT] {flash_reason}. Pausing all Altcoin entries until {halt_dt.strftime('%Y-%m-%d %H:%M')}")
            push_system_alert(cur, conn, "VOLATILITY_HALT", "[VOLATILITY HALT TRIGGERED]", flash_reason, vol_val)

        # Evaluate Portfolio Circuit Breaker
        cb_triggered, cb_reason = evaluate_portfolio_circuit_breaker(cur, conn, t)
        if cb_triggered:
            circuit_breaker_halt_until_ms = trigger_circuit_breaker_halt(conn, cur, t, cb_reason)
            continue

        # Execute symbol steps
        for sym in symbols:
            # Check Altcoin halt during BTC flash crashes
            if sym != "BTCUSDT" and t < altcoin_volatility_halt_until_ms:
                log(f"[{sym}] SKIP BAR -> Altcoin entries suspended during BTC flash volatility cooldown")
                continue

            step_symbol_bar(conn, cur, sym, t)

    # Print Final Summary Report with Strategy Breakdown
    print_portfolio_report(cur, symbols)
    conn.close()


def print_portfolio_report(cur, symbols: List[str]):
    print("\n" + "=" * 96)
    print("                 REGIME-ADAPTIVE MULTI-STRATEGY PAPER TRADING REPORT")
    print("=" * 96)

    cur.execute("""
        SELECT "Id", "Symbol", "StrategyType", "Side", "EntryPrice", "ExitPrice",
               "PositionSizeUsdt", "NetReturn", "Status", "ExitReason", "EntryTimeMs"
        FROM "PaperTrades"
        ORDER BY "Id" ASC
    """)
    trades = cur.fetchall()

    header = f"{'ID':<4} | {'Symbol':<8} | {'Strategy':<15} | {'Side':<5} | {'Entry Price':<12} | {'Exit Price':<12} | {'Net Return':<10} | {'PnL (USDT)':<10} | {'Status':<6} | {'Reason':<15}"
    print(header)
    print("-" * 96)

    total_pnl = 0.0
    wins = 0
    losses = 0
    strategy_stats: Dict[str, Dict[str, Any]] = {
        "TREND_MOMENTUM": {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
        "MEAN_REVERSION": {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
    }

    for r in trades:
        tid, sym, strat, side, ep, xp, pos_size, net_ret, status, exit_reason, entry_ms = r
        strat_s = strat or "TREND_MOMENTUM"
        ep_s = f"${float(ep):,.2f}" if ep else "-"
        xp_s = f"${float(xp):,.2f}" if xp else "-"

        if net_ret is not None and pos_size is not None:
            net_pct = float(net_ret) * 100
            pnl = float(pos_size) * float(net_ret)
            total_pnl += pnl
            is_win = net_ret > 0
            if is_win:
                wins += 1
            else:
                losses += 1
            ret_s = f"{net_pct:+.2f}%"
            pnl_s = f"${pnl:+.2f}"

            if strat_s in strategy_stats:
                strategy_stats[strat_s]["trades"] += 1
                if is_win:
                    strategy_stats[strat_s]["wins"] += 1
                else:
                    strategy_stats[strat_s]["losses"] += 1
                strategy_stats[strat_s]["pnl"] += pnl
        else:
            ret_s = "-"
            pnl_s = "-"

        reason_s = exit_reason or "-"
        print(f"{tid:<4} | {sym:<8} | {strat_s:<15} | {side.upper():<5} | {ep_s:<12} | {xp_s:<12} | {ret_s:<10} | {pnl_s:<10} | {status:<6} | {reason_s:<15}")

    final_balance = INITIAL_BALANCE_USDT + total_pnl
    total_closed = wins + losses
    overall_win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0

    print("-" * 96)
    print(f"Overall Portfolio Performance:")
    print(f"  Initial Capital : ${INITIAL_BALANCE_USDT:,.2f} USDT")
    print(f"  Current Capital : ${final_balance:,.2f} USDT (Net Return: {total_pnl/INITIAL_BALANCE_USDT*100:+.2f}%)")
    print(f"  Total Trades    : {total_closed} (Wins: {wins} | Losses: {losses} | Win Rate: {overall_win_rate:.1f}%)")
    print("\nStrategy Performance Breakdown:")
    for s_name, stats in strategy_stats.items():
        st_closed = stats["wins"] + stats["losses"]
        st_wr = (stats["wins"] / st_closed * 100) if st_closed > 0 else 0.0
        print(f"  [{s_name:<14}] Trades: {st_closed:<3} | Win Rate: {st_wr:5.1f}% | PnL: ${stats['pnl']:+8.2f}")
    print("=" * 96 + "\n")


# ── Stress-Test Mock Scenario ───────────────────────────────────────────────

def run_stress_test_mock():
    """
    Stress-Test Mock:
    Simulates a sequence of adverse market conditions (Drawdown >= 4.0%)
    to verify automated Kill-Switch trigger, position liquidation, and trading lockdown.
    """
    log("=" * 80)
    log("RUNNING PORTFOLIO CIRCUIT BREAKER STRESS-TEST MOCK")
    log("=" * 80)

    conn = get_conn()
    cur = conn.cursor()
    ensure_schema(conn, cur)

    # 1. Clear trades and seed a realistic initial portfolio
    cur.execute('DELETE FROM "PaperTrades"')
    conn.commit()

    base_time_ms = int(datetime(2026, 8, 15, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)

    log("\n[Step 1] Seeding initial normal trades ($10,000 USDT NAV)...")
    # Trade 1: Normal profitable trade
    cur.execute(
        """INSERT INTO "PaperTrades" (
               "Symbol", "Timeframe", "WindowEndMs", "EntryTimeMs", "ExitTimeMs", "EntryPrice",
               "ExitPrice", "Side", "Status", "Confidence", "ModelVersion", "PositionSizeUsdt",
               "NetReturn", "ExitReason", "StrategyType", "BalanceAfter", "CreatedAtUtc", "ClosedAtUtc"
           ) VALUES ('BTCUSDT', '4h', %s, %s, %s, 60000, 60900, 'long', 'closed', 0.65,
                     'XGB_test', 1000, 0.015, 'TP', 'TREND_MOMENTUM', 10015.0, NOW(), NOW())""",
        (base_time_ms, base_time_ms, base_time_ms + 14400000)
    )
    conn.commit()
    log("  Trade #1 seeded: Profitable (+1.5%, PnL=+$15.00)")

    # 2. Inject a rapid sequence of adverse losing trades causing > 4.0% daily loss ($400+ loss)
    log("\n[Step 2] Injecting extreme adverse market shocks causing > 4.0% 24h drawdown...")
    t2 = base_time_ms + 28800000
    t3 = base_time_ms + 43200000

    # Trade 2: -2.5% loss on $2,500 position (-$62.50)
    cur.execute(
        """INSERT INTO "PaperTrades" (
               "Symbol", "Timeframe", "WindowEndMs", "EntryTimeMs", "ExitTimeMs", "EntryPrice",
               "ExitPrice", "Side", "Status", "Confidence", "ModelVersion", "PositionSizeUsdt",
               "NetReturn", "ExitReason", "StrategyType", "BalanceAfter", "CreatedAtUtc", "ClosedAtUtc"
           ) VALUES ('SOLUSDT', '4h', %s, %s, %s, 80, 78, 'long', 'closed', 0.62,
                     'XGB_test', 2500, -0.025, 'SL', 'TREND_MOMENTUM', 9952.5, NOW(), NOW())""",
        (t2, t2, t2 + 14400000)
    )
    # Trade 3: -4.2% loss on $9,000 position (-$378.00) -> Total 24h loss = $425.50 (4.25% DD)
    cur.execute(
        """INSERT INTO "PaperTrades" (
               "Symbol", "Timeframe", "WindowEndMs", "EntryTimeMs", "ExitTimeMs", "EntryPrice",
               "ExitPrice", "Side", "Status", "Confidence", "ModelVersion", "PositionSizeUsdt",
               "NetReturn", "ExitReason", "StrategyType", "BalanceAfter", "CreatedAtUtc", "ClosedAtUtc"
           ) VALUES ('ETHUSDT', '4h', %s, %s, %s, 2200, 2107.6, 'long', 'closed', 0.64,
                     'XGB_test', 9000, -0.042, 'SL', 'TREND_MOMENTUM', 9574.5, NOW(), NOW())""",
        (t3, t3, t3 + 14400000)
    )
    conn.commit()
    log("  Trade #2 seeded: Loss (-$62.50)")
    log("  Trade #3 seeded: Loss (-$378.00) -> Total 24h Drawdown: $425.50 (~4.25% > 4.0% limit)")


    # 3. Seed an active OPEN position that should be automatically liquidated by the Circuit Breaker
    log("\n[Step 3] Seeding an active OPEN trade on BTCUSDT to verify auto-liquidation...")
    t4 = base_time_ms + 57600000
    cur.execute(
        """INSERT INTO "PaperTrades" (
               "Symbol", "Timeframe", "WindowEndMs", "EntryTimeMs", "ExitTimeMs", "EntryPrice",
               "Side", "Status", "Confidence", "ModelVersion", "PositionSizeUsdt",
               "StrategyType", "CreatedAtUtc"
           ) VALUES ('BTCUSDT', '4h', %s, %s, 0, 61000, 'long', 'open', 0.60,
                     'XGB_test', 1200, 'TREND_MOMENTUM', NOW())
           RETURNING "Id" """,
        (t4, t4)
    )
    open_tid = cur.fetchone()[0]
    conn.commit()
    log(f"  Active Open Trade #{open_tid} seeded on BTCUSDT ($1,200 position)")

    # 4. Trigger Circuit Breaker evaluation at bar t4
    log(f"\n[Step 4] Running Circuit Breaker evaluation at bar {datetime.fromtimestamp(t4/1000, timezone.utc)}...")
    is_triggered, reason = evaluate_portfolio_circuit_breaker(cur, conn, t4)
    log(f"  Circuit Breaker Evaluation: Triggered={is_triggered} | Reason: {reason}")

    if is_triggered:
        halt_until_ms = trigger_circuit_breaker_halt(conn, cur, t4, reason)
        halt_dt = datetime.fromtimestamp(halt_until_ms / 1000, timezone.utc)
        log(f"  ✅ Kill-Switch Successfully Executed! System status -> HALTED_CIRCUIT_BREAKER until {halt_dt}")

    # 5. Verify the active trade was closed with CIRCUIT_BREAKER
    cur.execute("""SELECT "Id", "Status", "ExitReason", "ExitPrice", "NetReturn" FROM "PaperTrades" WHERE "Id"=%s""", (open_tid,))
    closed_check = cur.fetchone()
    log(f"\n[Step 5] Verifying Open Trade Status:")
    log(f"  Trade #{closed_check[0]} Status: {closed_check[1]} | ExitReason: {closed_check[2]} | ExitPrice: ${float(closed_check[3] or 0):,.2f}")

    # 6. Verify alert in AppAlerts
    cur.execute("""SELECT "Type", "Title", "Message", "CreatedAt" FROM "AppAlerts" WHERE "Type"='CIRCUIT_BREAKER' ORDER BY "CreatedAt" DESC LIMIT 1""")
    alert_row = cur.fetchone()
    if alert_row:
        log(f"\n[Step 6] Verifying System Alert in AppAlerts:")
        log(f"  Type: {alert_row[0]} | Title: {alert_row[1]}")
        log(f"  Message: {alert_row[2]}")

    print_portfolio_report(cur, ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    conn.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Regime-Adaptive Multi-Strategy Paper Trader with Circuit Breaker")
    parser.add_argument("--symbols", default=",".join(ACTIVE_SYMBOLS), help="Comma-separated symbols")
    parser.add_argument("--start-ms", type=int, default=None, help="Start timestamp in ms")
    parser.add_argument("--end-ms", type=int, default=None, help="End timestamp in ms")
    parser.add_argument("--clear-previous", action="store_true", help="Clear previous paper trades before running")
    parser.add_argument("--mock-stress-test", action="store_true", help="Run Circuit Breaker Stress-Test Mock scenario")
    args = parser.parse_args()

    if args.mock_stress-test if hasattr(args, "mock_stress-test") else args.mock_stress_test:
        run_stress_test_mock()
        return

    if args.clear_previous:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute('DELETE FROM "PaperTrades"')
        conn.commit()
        conn.close()
        log("[INFO] Cleared previous PaperTrades records.")

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    run_multi_asset_loop(symbols=symbols, start_ms=args.start_ms, end_ms=args.end_ms)


if __name__ == "__main__":
    main()
