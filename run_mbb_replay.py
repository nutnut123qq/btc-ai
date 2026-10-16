"""MBB (HOSE equity) rule-based walk-forward replay — MBB-04C.

Lightweight replay on the ``InstrumentCandles`` store, deliberately NOT a
port of ``backtest_strategy.py`` (which is BTC-coupled: Binance costs,
WindowClassificationDatasets, ML model outputs).  There is no MBB model
yet, so this v1 evaluates two declared rule-based strategies plus honest
baselines and emits a provenance-complete JSON artifact under
``docs/research/evidence/mbb-rule-replay/``.

Declared protocol (fixed before looking at results, per the plan's
evaluation rules):

- Chronological replay only ("walk-forward" in the time sense; rule
  strategies have nothing to refit).  Signals use only bars already
  closed at decision time — strictly causal, no look-ahead.
- Signal at a closed bar -> fill no earlier than the next expected
  session open via ``vn_execution_model`` (lot 100, ±7% band on the
  previous closed same-timeframe bar, T+2 settlement, sell tax, broker
  fee).  Off-session anomaly bars are excluded from signals AND fills.
- Single position, all-in on settled cash, whole-position exit.  One
  resting order at a time; a band-rejected order is cancelled, not
  clamped.
- Baselines on the same cost model: buy-and-hold (max lots at the first
  fillable open, liquidated at the last close) and flat (no trade).
- Verdict rule (declared): ``unavailable`` when the usable bar count is
  below the strategy's declared minimum; ``inconclusive`` when closed
  round trips < ``--min-trades`` OR the strategy's net return does not
  exceed every available baseline; ``supported`` otherwise.
- All prices/PnL are adjusted basis (``priceBasis: "adjusted"`` is
  stamped on the report and every trade) — not real matched prices.
- MBB-only tool: symbol must resolve to an equity instrument in
  ``trading_config.INSTRUMENTS``; anything else fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import trading_config
import vietnam_market_calendar as vn_calendar
from vn_execution_model import (
    PRICE_BASIS,
    VnEquityAccount,
    VnEquityCostSpec,
    VnFill,
    VnRejection,
    close_round_trip,
)

SCHEMA_NAME = "vn-equity-rule-replay/v1"
DEFAULT_OUT_DIR = (
    Path(__file__).resolve().parent / "docs" / "research" / "evidence" / "mbb-rule-replay"
)
DEFAULT_INITIAL_CAPITAL_VND = 50_000_000.0
DEFAULT_MIN_TRADES = 10
VERDICTS = ("supported", "inconclusive", "unavailable")

CANDLE_FIELDS = ("OpenTimeMs", "CloseTimeMs", "Open", "High", "Low", "Close", "Volume")


# ---------------------------------------------------------------------------
# Declared rule-based strategies (params fixed here; no sweeping).
# ---------------------------------------------------------------------------


def _sma_at(closes: list[float], i: int, n: int) -> float:
    return sum(closes[i - n + 1 : i + 1]) / n


def _sma_cross_signals(closes: list[float], fast: int, slow: int) -> list[str | None]:
    """SMA(fast) crosses above/below SMA(slow); both windows causal."""
    signals: list[str | None] = [None] * len(closes)
    for i in range(slow, len(closes)):  # i-1 must have a full slow window
        f_prev, s_prev = _sma_at(closes, i - 1, fast), _sma_at(closes, i - 1, slow)
        f_cur, s_cur = _sma_at(closes, i, fast), _sma_at(closes, i, slow)
        if f_prev <= s_prev and f_cur > s_cur:
            signals[i] = "buy"
        elif f_prev >= s_prev and f_cur < s_cur:
            signals[i] = "sell"
    return signals


def _rsi_wilder(closes: list[float], period: int) -> list[float | None]:
    rsi: list[float | None] = [None] * len(closes)
    if len(closes) <= period:
        return rsi
    gains, losses = [], []
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    def _rsi(ag: float, al: float) -> float:
        if al == 0:
            return 100.0
        return 100.0 - 100.0 / (1.0 + ag / al)

    rsi[period] = _rsi(avg_gain, avg_loss)
    for i in range(period + 1, len(closes)):
        diff = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(diff, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-diff, 0.0)) / period
        rsi[i] = _rsi(avg_gain, avg_loss)
    return rsi


def _rsi_mean_revert_signals(
    closes: list[float], period: int, oversold: float, exit_level: float
) -> list[str | None]:
    rsi = _rsi_wilder(closes, period)
    signals: list[str | None] = [None] * len(closes)
    for i, value in enumerate(rsi):
        if value is None:
            continue
        if value < oversold:
            signals[i] = "buy"
        elif value > exit_level:
            signals[i] = "sell"
    return signals


StrategySpec = dict[str, Any]

STRATEGIES: dict[str, StrategySpec] = {
    "sma-cross-20-50": {
        "name": "sma-cross-20-50",
        "timeframe": "1d",
        "minBars": 52,
        "params": {"fast": 20, "slow": 50},
        "signals": lambda closes: _sma_cross_signals(closes, 20, 50),
        "description": "SMA(20)/SMA(50) golden/death cross on 1d closes.",
    },
    "rsi-mean-revert-14": {
        "name": "rsi-mean-revert-14",
        "timeframe": "1h",
        "minBars": 16,
        "params": {"period": 14, "oversold": 30.0, "exitLevel": 50.0},
        "signals": lambda closes: _rsi_mean_revert_signals(closes, 14, 30.0, 50.0),
        "description": "Wilder RSI(14) < 30 buys, > 50 exits, on 1h closes.",
    },
}


# ---------------------------------------------------------------------------
# Data loading + canonical hashing
# ---------------------------------------------------------------------------


def load_candles_db(symbol: str, timeframe: str, cutoff_ms: int) -> list[dict[str, Any]]:
    """Read InstrumentCandles for a symbol/timeframe (ascending, finalized)."""
    from db_config import get_db_connection

    connection = get_db_connection()
    try:
        connection.set_session(
            readonly=True,
            autocommit=False,
            isolation_level="REPEATABLE READ",
        )
        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT "OpenTimeMs","CloseTimeMs","Open","High","Low","Close",'
                '"Volume","PriceBasis" '
                'FROM "InstrumentCandles" WHERE "Symbol"=%s AND "Timeframe"=%s '
                'AND "CloseTimeMs"<=%s ORDER BY "OpenTimeMs" ASC',
                (symbol, timeframe, cutoff_ms),
            )
            names = CANDLE_FIELDS + ("PriceBasis",)
            rows = [dict(zip(names, row)) for row in cursor.fetchall()]
        connection.rollback()
        return rows
    finally:
        connection.close()


def normalize_candles(
    rows: Iterable[Mapping[str, Any]], cutoff_ms: int
) -> list[dict[str, Any]]:
    """Coerce fixture/DB rows to float bars, filter to finalized, sort ascending."""
    bars = []
    for row in rows:
        bar = {
            "open_ms": int(row["OpenTimeMs"]),
            "close_ms": int(row["CloseTimeMs"]),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"]),
            "price_basis": row.get("PriceBasis"),
        }
        if bar["close_ms"] <= cutoff_ms:
            bars.append(bar)
    bars.sort(key=lambda b: b["open_ms"])
    declared = {b["price_basis"] for b in bars if b["price_basis"] is not None}
    bad = declared - {PRICE_BASIS}
    if bad:
        raise ValueError(
            f"InstrumentCandles rows declare price basis {sorted(bad)}; this replay "
            "is adjusted-basis only and refuses to mix bases."
        )
    return bars


def dataset_sha256(bars: list[dict[str, Any]]) -> str:
    """sha256 over the canonical candle serialization (deterministic)."""
    lines = [
        ",".join(
            [
                str(bar["open_ms"]),
                str(bar["close_ms"]),
                repr(bar["open"]),
                repr(bar["high"]),
                repr(bar["low"]),
                repr(bar["close"]),
                repr(bar["volume"]),
            ]
        )
        for bar in bars
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Replay core
# ---------------------------------------------------------------------------


def _resolve_symbol(symbol: str) -> str:
    canonical = trading_config.require_known_symbol(symbol)
    if trading_config.INSTRUMENTS[canonical]["asset_class"] != "equity":
        raise ValueError(
            f"run_mbb_replay is equity-only; '{canonical}' is "
            f"{trading_config.INSTRUMENTS[canonical]['asset_class']}."
        )
    return canonical


def _resolve_strategy(strategy: str | StrategySpec) -> StrategySpec:
    if isinstance(strategy, Mapping):
        for key in ("name", "timeframe", "minBars", "params", "signals"):
            if key not in strategy:
                raise ValueError(f"Custom strategy spec missing '{key}'.")
        return dict(strategy)
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Declared: {', '.join(STRATEGIES)}."
        )
    return STRATEGIES[strategy]


def _max_drawdown_fraction(equity: list[float]) -> float:
    peak = 0.0
    max_dd = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            max_dd = min(max_dd, value / peak - 1.0)
    return -max_dd


def _buy_and_hold_baseline(
    bars: list[dict[str, Any]],
    initial_cash_vnd: float,
    costs: VnEquityCostSpec,
    non_trading_dates: frozenset[date],
    timeframe: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {"id": "buy-and-hold", "priceBasis": PRICE_BASIS}
    if len(bars) < 2:
        result.update({"status": "unavailable", "reason": "insufficient-bars"})
        return result
    account = VnEquityAccount(initial_cash_vnd, costs, non_trading_dates)
    expected = vn_calendar.next_expected_open_ms(
        timeframe, bars[0]["open_ms"], non_trading_dates
    )
    fill_bar = next((b for b in bars[1:] if expected and b["open_ms"] >= expected), None)
    if fill_bar is None:
        result.update({"status": "unavailable", "reason": "no-fill-bar"})
        return result
    fill = account.execute_buy(
        fill_open_ms=fill_bar["open_ms"],
        open_price=fill_bar["open"],
        reference_close=bars[0]["close"],
        expected_open_ms=expected,
    )
    if not isinstance(fill, VnFill):
        result.update(
            {"status": "unavailable", "reason": f"entry-{fill.reason}"}
        )
        return result
    account.settle_as_of(date(3000, 1, 1))
    last_close = bars[-1]["close"]
    exit_costs = fill.shares * last_close * (
        costs.broker_fee_fraction + costs.sell_tax_fraction
    )
    gross_value = account.settled_cash_vnd + fill.shares * last_close
    net_value = gross_value - exit_costs
    result.update(
        {
            "status": "evaluated",
            "entryOpenMs": fill.fill_open_ms,
            "shares": fill.shares,
            "entryPrice": fill.price,
            "markClose": last_close,
            "grossReturnPct": (gross_value / initial_cash_vnd - 1.0) * 100.0,
            "netReturnPct": (net_value / initial_cash_vnd - 1.0) * 100.0,
            "note": "Entry at first fillable session open; liquidation costs "
            "(fee+tax on last close) netted for comparability.",
        }
    )
    return result


def run_replay(
    *,
    candles: Iterable[Mapping[str, Any]],
    strategy: str | StrategySpec,
    symbol: str = "MBB",
    timeframe: str | None = None,
    initial_cash_vnd: float = DEFAULT_INITIAL_CAPITAL_VND,
    min_trades: int = DEFAULT_MIN_TRADES,
    cutoff_ms: int | None = None,
    costs: VnEquityCostSpec | None = None,
    non_trading_dates: frozenset[date] | None = None,
) -> dict[str, Any]:
    """Chronological rule replay -> provenance-complete report dict.

    ``candles`` are InstrumentCandles-shaped rows (fixture-injectable; the
    DB loader produces the same shape).  Pure: never touches the DB.
    """
    canonical = _resolve_symbol(symbol)
    spec = _resolve_strategy(strategy)
    tf = timeframe or spec["timeframe"]
    if tf != spec["timeframe"]:
        raise ValueError(
            f"Strategy '{spec['name']}' is bound to {spec['timeframe']}; "
            f"got timeframe '{tf}'."
        )
    tf = trading_config.require_active_timeframe(tf, canonical)
    costs = costs or VnEquityCostSpec()
    ntd = (
        frozenset(non_trading_dates)
        if non_trading_dates is not None
        else vn_calendar.non_trading_dates_from_env()
    )
    cutoff = int(cutoff_ms if cutoff_ms is not None else time.time() * 1000)
    if min_trades < 1:
        raise ValueError("min_trades must be at least 1.")

    raw_bars = normalize_candles(candles, cutoff)
    bars = [
        b
        for b in raw_bars
        if vn_calendar.is_expected_open(tf, b["open_ms"], ntd)
    ]
    excluded_anomalies = len(raw_bars) - len(bars)

    report: dict[str, Any] = {
        "schema": SCHEMA_NAME,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "symbol": canonical,
        "timeframe": tf,
        "priceBasis": PRICE_BASIS,
        "evaluationMode": "chronological-walk-forward-no-refit",
        "strategy": {
            "name": spec["name"],
            "timeframe": spec["timeframe"],
            "params": spec["params"],
            "description": spec.get("description"),
        },
        "costs": costs.to_dict(),
        "execution": {
            "fillPolicy": "next-session-open",
            "bandReference": "previous-closed-bar-same-timeframe",
            "bandBreachPolicy": "reject",
            "settlementTradingDays": costs.settlement_trading_days,
            "lotSize": costs.lot_size,
            "shortSelling": "not-supported",
            "positionModel": "single-position-all-in-settled-cash",
        },
        "calendar": {
            "nonTradingDatesEnv": vn_calendar.NON_TRADING_DATES_ENV,
            "nonTradingDates": sorted(d.isoformat() for d in ntd),
        },
        "dataset": {
            "candleRows": len(raw_bars),
            "usableSessionBars": len(bars),
            "excludedAnomalyBars": excluded_anomalies,
            "firstOpenMs": bars[0]["open_ms"] if bars else None,
            "lastCloseMs": bars[-1]["close_ms"] if bars else None,
            "sha256": dataset_sha256(raw_bars),
            "cutoffMs": cutoff,
        },
    }

    verdict_rule = (
        f"supported requires >= {min_trades} closed round trips and net "
        "return above every evaluated baseline; fewer trades or no baseline "
        "beat -> inconclusive; too few usable bars -> unavailable."
    )
    report["verdictRule"] = verdict_rule

    # -- replay loop ----------------------------------------------------
    account = VnEquityAccount(initial_cash_vnd, costs, ntd)
    signals = spec["signals"]([b["close"] for b in bars]) if bars else []
    pending: dict[str, Any] | None = None
    entry_fill: VnFill | None = None
    fills: list[VnFill] = []
    rejections: list[VnRejection] = []
    round_trips = []
    equity_curve: list[float] = []

    for i, bar in enumerate(bars):
        account.settle_as_of(vn_calendar.to_local(bar["open_ms"]).date())

        # 1) resting order fills at this session open (or a later one after
        #    a missing expected open — the order stays resting).
        if pending is not None and bar["open_ms"] >= pending["expected_open_ms"]:
            executor = (
                account.execute_buy if pending["side"] == "buy" else account.execute_sell
            )
            outcome = executor(
                fill_open_ms=bar["open_ms"],
                open_price=bar["open"],
                reference_close=pending["reference_close"],
                expected_open_ms=pending["expected_open_ms"],
            )
            if isinstance(outcome, VnFill):
                fills.append(outcome)
                if outcome.side == "buy":
                    entry_fill = outcome
                else:
                    round_trips.append(close_round_trip(entry_fill, outcome))
                    entry_fill = None
            else:
                rejections.append(outcome)
            pending = None

        # 2) signal on this just-closed bar; only one resting order.
        if pending is None and i < len(signals):
            signal = signals[i]
            want = (signal == "buy" and account.total_shares == 0) or (
                signal == "sell" and account.total_shares > 0
            )
            if want:
                expected = vn_calendar.next_expected_open_ms(tf, bar["open_ms"], ntd)
                if expected is not None:
                    pending = {
                        "side": signal,
                        "signal_open_ms": bar["open_ms"],
                        "expected_open_ms": expected,
                        "reference_close": bar["close"],
                    }

        equity_curve.append(account.equity_vnd(bar["close"]))

    # -- accounting ------------------------------------------------------
    account.settle_as_of(date(3000, 1, 1))  # flush pending at horizon end
    last_close = bars[-1]["close"] if bars else 0.0
    final_equity = (
        account.settled_cash_vnd + account.total_shares * last_close
        if bars
        else account.settled_cash_vnd
    )
    gross_pnl = sum(t.gross_pnl_vnd for t in round_trips)
    net_pnl = sum(t.net_pnl_vnd for t in round_trips)
    fee_total = sum(f.fee_vnd for f in fills)
    tax_total = sum(f.tax_vnd for f in fills)
    winners = sum(1 for t in round_trips if t.net_pnl_vnd > 0)

    open_position = None
    if entry_fill is not None and bars:
        invested = entry_fill.gross_value_vnd + entry_fill.fee_vnd
        open_position = {
            "entryOpenMs": entry_fill.fill_open_ms,
            "shares": entry_fill.shares,
            "entryPrice": entry_fill.price,
            "markClose": last_close,
            "unrealizedNetPct": (last_close * entry_fill.shares / invested - 1.0)
            * 100.0,
            "priceBasis": PRICE_BASIS,
        }

    report["accounting"] = {
        "initialCashVnd": initial_cash_vnd,
        "finalEquityVnd": final_equity,
        "netReturnPct": (final_equity / initial_cash_vnd - 1.0) * 100.0
        if initial_cash_vnd > 0
        else 0.0,
        "grossReturnPct": ((final_equity + fee_total + tax_total) / initial_cash_vnd - 1.0)
        * 100.0
        if initial_cash_vnd > 0
        else 0.0,
        "maxDrawdownPct": _max_drawdown_fraction(equity_curve) * 100.0,
    }
    report["trades"] = {
        "closedRoundTrips": len(round_trips),
        "winners": winners,
        "hitRate": winners / len(round_trips) if round_trips else None,
        "grossPnlVnd": gross_pnl,
        "netPnlVnd": net_pnl,
        "feesVnd": fee_total,
        "taxesVnd": tax_total,
        "fills": [f.to_dict() for f in fills],
        "rejections": [r.to_dict() for r in rejections],
        "unfilledOrderAtEnd": pending["side"] if pending else None,
        "openPosition": open_position,
        "roundTrips": [t.to_dict() for t in round_trips],
    }

    # -- baselines + verdict ---------------------------------------------
    baselines = [
        _buy_and_hold_baseline(bars, initial_cash_vnd, costs, ntd, tf),
        {"id": "flat", "status": "evaluated", "netReturnPct": 0.0, "grossReturnPct": 0.0},
    ]
    report["baselines"] = baselines

    if len(bars) < spec["minBars"]:
        verdict, reason = (
            "unavailable",
            f"insufficient-bars:{len(bars)}<{spec['minBars']}",
        )
    elif len(round_trips) < min_trades:
        verdict, reason = (
            "inconclusive",
            f"insufficient-trades:{len(round_trips)}<{min_trades}",
        )
    else:
        net_pct = report["accounting"]["netReturnPct"]
        best = max(
            (b["netReturnPct"] for b in baselines if b.get("status") == "evaluated"),
            default=None,
        )
        if best is not None and net_pct > best:
            verdict, reason = "supported", "net-return-above-baselines"
        else:
            verdict, reason = "inconclusive", "net-return-not-above-baselines"
    report["verdict"] = {"value": verdict, "reason": reason}
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _artifact_name(report: dict[str, Any]) -> str:
    return (
        f"{report['strategy']['name']}-{report['symbol']}-{report['timeframe']}"
        f"-cutoff{report['dataset']['cutoffMs']}.json"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MBB (HOSE equity) rule-based replay on InstrumentCandles."
    )
    parser.add_argument("--symbol", default="MBB")
    parser.add_argument(
        "--strategy",
        default="all",
        choices=sorted(STRATEGIES) + ["all"],
    )
    parser.add_argument(
        "--timeframe",
        default=None,
        help="Optional; must match the strategy's declared timeframe.",
    )
    parser.add_argument("--cutoff-ms", type=int, default=None)
    parser.add_argument("--initial-capital-vnd", type=float, default=DEFAULT_INITIAL_CAPITAL_VND)
    parser.add_argument("--min-trades", type=int, default=DEFAULT_MIN_TRADES)
    parser.add_argument("--broker-fee-fraction", type=float, default=None)
    parser.add_argument("--sell-tax-fraction", type=float, default=None)
    parser.add_argument(
        "--fixture-json",
        default=None,
        help="JSON file of InstrumentCandles rows; bypasses the DB.",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    canonical = _resolve_symbol(args.symbol)
    costs = VnEquityCostSpec(
        **{
            k: v
            for k, v in {
                "broker_fee_fraction": args.broker_fee_fraction,
                "sell_tax_fraction": args.sell_tax_fraction,
            }.items()
            if v is not None
        }
    )
    cutoff = int(args.cutoff_ms if args.cutoff_ms is not None else time.time() * 1000)

    if args.fixture_json:
        with open(args.fixture_json, "r", encoding="utf-8") as handle:
            candles = json.load(handle)
    else:
        tf_for_load = args.timeframe
        reports = []
        names = sorted(STRATEGIES) if args.strategy == "all" else [args.strategy]
        # Each strategy binds its own timeframe; load per strategy inside the loop.
        for name in names:
            spec = _resolve_strategy(name)
            tf = tf_for_load or spec["timeframe"]
            try:
                candles_db = load_candles_db(canonical, tf, cutoff)
            except Exception as exc:  # DB unreachable -> honest unavailable
                report = {
                    "schema": SCHEMA_NAME,
                    "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
                    "symbol": canonical,
                    "timeframe": tf,
                    "priceBasis": PRICE_BASIS,
                    "strategy": {"name": spec["name"], "params": spec["params"]},
                    "verdict": {
                        "value": "unavailable",
                        "reason": f"data-load-failed:{type(exc).__name__}",
                    },
                }
                reports.append(report)
                continue
            reports.append(
                run_replay(
                    candles=candles_db,
                    strategy=spec,
                    symbol=canonical,
                    timeframe=tf,
                    initial_cash_vnd=args.initial_capital_vnd,
                    min_trades=args.min_trades,
                    cutoff_ms=cutoff,
                    costs=costs,
                )
            )
        _emit(reports, args)
        return 0

    names = sorted(STRATEGIES) if args.strategy == "all" else [args.strategy]
    reports = [
        run_replay(
            candles=candles,
            strategy=_resolve_strategy(name),
            symbol=canonical,
            timeframe=args.timeframe,
            initial_cash_vnd=args.initial_capital_vnd,
            min_trades=args.min_trades,
            cutoff_ms=cutoff,
            costs=costs,
        )
        for name in names
    ]
    _emit(reports, args)
    return 0


def _emit(reports: list[dict[str, Any]], args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    for report in reports:
        verdict = report["verdict"]["value"]
        print(
            f"{report['strategy']['name']} {report['symbol']} {report['timeframe']}: "
            f"verdict={verdict} reason={report['verdict']['reason']}"
        )
        if not args.no_write:
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / _artifact_name(report)
            path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"  artifact: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
