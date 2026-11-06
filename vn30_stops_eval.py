"""STOPS-1 (roadmap 2.5): backtest of the decision engine's declared -8% stop / -4% reduce rules.

The backend (btc-be/Services/MbbDecisionService.cs, declared-rules-v4) exits a position on a 1d close at or below
avgCost x 0.92 and reduces it on a 1d close at or below avgCost x 0.96 when the trend is bearish (close < SMA50
and RSI14 < 50); both levels are rounded up onto the HOSE tick grid. Those constants were never backtested.

This module replays them, as frozen in docs/research/protocols/stops-1.json (registry entry ``hose-stops-1``,
registered with the spec's sha256 before any result), through ``vn_portfolio_backtest.run_backtest``'s
``risk_exit`` hook on two VN30 portfolios (equal weight, B-mom_6 top 10), development window
2018-01-22..2023-12-31 only (bars after it are removed before the simulation; the VN30-2 holdout is not read).

    python vn30_stops_eval.py run [--snapshot <dir>] [--out-root <dir>]
    python vn30_stops_eval.py record <result.json>      # supersede the planned entry with status run

Pure computation besides reading the snapshot and writing the result; no network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import trial_registry
import vn30_panel_eval as pe
import vn_portfolio_backtest as pb

SPEC_ID = "stops-1"
SPEC_PATH = Path(__file__).resolve().parent / "docs" / "research" / "protocols" / f"{SPEC_ID}.json"
PROTOCOL_PATH = Path(__file__).resolve().parent / "docs" / "research" / "protocols" / "vn30-2.json"
REGISTRY_ID = "hose-stops-1"
STOP_LOSS_FRACTION = 0.08    # MbbDecisionService.StopLossFraction
REDUCE_LOSS_FRACTION = 0.04  # MbbDecisionService.ReduceLossFraction
SMA_BARS = 50
RSI_BARS = 14
RSI_MID = 50.0
REDUCE_SHARE = 0.5  # declared in the spec: the backend's reduce carries no quantity
MOM6_BARS = 126
WINDOW_FROM = "2018-01-22"
WINDOW_TO = "2023-12-31"
SNAPSHOT_ID = "20261106-9afb8020ec16"
SNAPSHOT_SHA256 = "9afb8020ec16a1c26669b046b5621453012e3cd09a5dc24b70da754648acc94e"
PERIODS_PER_YEAR = 252
DSR_MIN = 0.95
PORTFOLIOS = ("P-ew", "P-mom6")
VARIANTS = ("none", "stop-8", "stop-8+reduce-4")


# --------------------------------------------------------------------------------------------- indicators


def round_up_to_tick(price: float) -> float:
    """Smallest HOSE tick-grid price at or above ``price`` (VietnamExecutionRules.RoundUpToTick)."""
    tick = pb.hose_tick(price)
    return math.ceil(price / tick - 1e-9) * tick


def wilder_rsi(closes: Sequence[float], period: int = RSI_BARS) -> list[float | None]:
    """Wilder RSI as the backend's TechnicalIndicatorIndexer.ComputeRsi: simple seed over the first ``period``
    deltas, then Wilder smoothing; None while warming up; 50 when flat, 100 without losses."""
    out: list[float | None] = []
    avg_gain = avg_loss = 0.0
    for i, c in enumerate(closes):
        if i == 0:
            out.append(None)
            continue
        delta = float(c) - float(closes[i - 1])
        gain, loss = max(delta, 0.0), max(-delta, 0.0)
        if i <= period:
            avg_gain += gain
            avg_loss += loss
            if i < period:
                out.append(None)
                continue
            avg_gain /= period
            avg_loss /= period
        else:
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            out.append(50.0 if avg_gain == 0 else 100.0)
        else:
            out.append(100.0 - 100.0 / (1.0 + avg_gain / avg_loss))
    return out


def sma(values: Sequence[float], n: int) -> list[float | None]:
    out: list[float | None] = []
    run = 0.0
    for i, v in enumerate(values):
        run += float(v)
        if i >= n:
            run -= float(values[i - n])
        out.append(run / n if i >= n - 1 else None)
    return out


@dataclass(frozen=True)
class DayIndicators:
    close: float
    sma50: float | None
    rsi14: float | None

    @property
    def bearish(self) -> bool:
        """MbbDecisionService: trend down (close < SMA50) and RSI14 below the midline."""
        return (self.sma50 is not None and self.rsi14 is not None
                and self.close < self.sma50 and self.rsi14 < RSI_MID)


def indicators_by_date(bars: Mapping[str, Sequence[float]]) -> dict[str, DayIndicators]:
    """date -> close, SMA50, RSI14 of one symbol; each value uses only bars dated <= that date."""
    dates = sorted(bars)
    closes = [float(bars[d][3]) for d in dates]
    s, r = sma(closes, SMA_BARS), wilder_rsi(closes, RSI_BARS)
    return {d: DayIndicators(closes[k], s[k], r[k]) for k, d in enumerate(dates)}


# --------------------------------------------------------------------------------------------- the rule


@dataclass
class StopReduceRule:
    """``risk_exit`` hook replaying the stop (and optionally the bearish reduce) on an average-cost book."""

    indicators: Mapping[str, Mapping[str, DayIndicators]]
    reduce: bool = False
    stop_fraction: float = STOP_LOSS_FRACTION
    reduce_fraction: float = REDUCE_LOSS_FRACTION
    lot_size: int = 100
    avg_cost: dict[str, float] = field(default_factory=dict)
    held: dict[str, int] = field(default_factory=dict)
    reduced: set[str] = field(default_factory=set)
    events: list[dict[str, Any]] = field(default_factory=list)

    def _book(self, fills: Sequence[Mapping[str, Any]]) -> None:
        for t in fills:
            s, qty, px = t["symbol"], int(t["shares"]), float(t["price"])
            have = self.held.get(s, 0)
            if t["side"] == "buy":
                self.avg_cost[s] = (self.avg_cost.get(s, 0.0) * have + px * qty) / (have + qty)
                self.held[s] = have + qty
            else:
                self.held[s] = have - qty
                if self.held[s] <= 0:
                    self.held.pop(s, None)
                    self.avg_cost.pop(s, None)

    def __call__(self, ctx: pb.RiskContext) -> dict[str, int]:
        self._book(ctx.fills)
        if any(self.held.get(s, 0) != n for s, n in ctx.shares.items()) or set(self.held) - set(ctx.shares):
            raise RuntimeError(f"{ctx.date}: average-cost book out of sync with the backtester positions")
        if ctx.rebalance_entry:
            self.reduced.clear()
        if ctx.decision_session:
            return {}
        exits: dict[str, int] = {}
        for s, shares in sorted(ctx.shares.items()):
            ind = self.indicators.get(s, {}).get(ctx.date)
            if shares <= 0 or ind is None or ctx.open_orders.get(s) == "sell":
                continue
            cost = self.avg_cost[s]
            stop_level = round_up_to_tick(cost * (1.0 - self.stop_fraction))
            reduce_level = round_up_to_tick(cost * (1.0 - self.reduce_fraction))
            base = {"date": ctx.date, "symbol": s, "close": ind.close, "avgCost": cost, "shares": shares}
            if ind.close <= stop_level:
                exits[s] = shares
                self.events.append({**base, "kind": "stop", "level": stop_level, "sellShares": shares})
            elif self.reduce and s not in self.reduced and ind.bearish and ind.close <= reduce_level:
                qty = pb.lot_floor(shares * REDUCE_SHARE, self.lot_size) or shares
                exits[s] = qty
                self.reduced.add(s)
                self.events.append({**base, "kind": "reduce", "level": reduce_level, "sellShares": qty,
                                    "sma50": ind.sma50, "rsi14": ind.rsi14})
        return exits


# --------------------------------------------------------------------------------------------- scores


def mom6_score(day: str, view: pb.PointInTimeView) -> dict[str, float]:
    """B-mom_6: close(d) / close 126 bars earlier - 1, over the symbol's own bars dated <= d."""
    out = {}
    for s in view.members:
        closes = view.closes(s)
        if len(closes) > MOM6_BARS and closes[-1 - MOM6_BARS] > 0:
            out[s] = closes[-1] / closes[-1 - MOM6_BARS] - 1.0
    return out


def ew_score(day: str, view: pb.PointInTimeView) -> dict[str, float]:
    return {s: 0.0 for s in view.members}


SCORES = {"P-ew": (ew_score, {"top_k": None, "keep_rank": None}), "P-mom6": (mom6_score, {})}


# --------------------------------------------------------------------------------------------- metrics


def nav_metrics(daily: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> dict[str, Any]:
    nav = np.array([float(r["nav"]) for r in daily])
    rets = nav[1:] / nav[:-1] - 1.0
    years = float(summary["years"])
    cagr = float((nav[-1] / nav[0]) ** (1.0 / years) - 1.0) if years > 0 else None
    sd = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    peak = np.maximum.accumulate(nav)
    return {"from": daily[0]["date"], "to": daily[-1]["date"], "sessions": len(daily), "years": years,
            "endNavVnd": float(nav[-1]), "totalReturn": float(nav[-1] / nav[0] - 1.0), "netCagr": cagr,
            "sharpeAnnualized": float(rets.mean() / sd * math.sqrt(PERIODS_PER_YEAR)) if sd else 0.0,
            "maxDrawdown": float((nav / peak - 1.0).min()),
            "annualTurnoverTwoSided": summary["annualTurnoverTwoSided"], "trades": summary["trades"],
            "costVnd": summary["costVnd"]}


def daily_returns(daily: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {daily[k]["date"]: daily[k]["nav"] / daily[k - 1]["nav"] - 1.0 for k in range(1, len(daily))}


def dsr_table(series: Mapping[str, Mapping[str, float]], n_trials: int) -> dict[str, Any]:
    """DSR of each series (date -> return) with N = ``n_trials`` and V[SR] over these series."""
    arrays = {k: np.array([v[d] for d in sorted(v)]) for k, v in series.items()}
    var = pe.sr_variance(arrays)
    out = {k: pe.deflated_sharpe(a, n_trials=n_trials, sr_variance=var["value"]) for k, a in arrays.items()}
    return {"srVariance": var, "series": out}


def decide(control: Mapping[str, Any], variant: Mapping[str, Any], excess_dsr: float) -> dict[str, Any]:
    checks = {"netCagrHigher": bool(variant["netCagr"] > control["netCagr"]),
              "sharpeHigher": bool(variant["sharpeAnnualized"] > control["sharpeAnnualized"]),
              "drawdownShallower": bool(variant["maxDrawdown"] > control["maxDrawdown"]),
              "excessDsrPass": bool(excess_dsr >= DSR_MIN)}
    return {**checks, "verdict": "supported" if all(checks.values()) else "not supported"}


# --------------------------------------------------------------------------------------------- run


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registered_entry(registry_path: Path | str = trial_registry.REGISTRY_PATH,
                     spec_path: Path | str = SPEC_PATH) -> dict[str, Any]:
    """The active stops-1 registry entry; fails unless it was registered for this exact spec file."""
    sha = sha256_file(spec_path)
    found = [e for e in trial_registry.load(registry_path).active()
             if e["id"] == REGISTRY_ID or e.get("supersedes") == REGISTRY_ID]
    if not found:
        raise ValueError(f"no active registry entry {REGISTRY_ID}: register the trial before running it")
    entry = found[-1]
    if entry.get("specSha256") != sha:
        raise ValueError(f"registry entry {entry['id']} was registered for another version of {SPEC_ID}.json")
    return entry


def default_snapshot_dir() -> Path:
    return pe.repo_main_checkout().parent / "data" / "vn30-research" / "snapshots" / SNAPSHOT_ID


def load_inputs(snapshot_dir: Path | str, end: str = WINDOW_TO) -> dict[str, Any]:
    """Calendar, bars and periods from the verified snapshot, every bar dated after ``end`` removed."""
    import vn30_research_data as vrd

    snap = vrd.load_snapshot(Path(snapshot_dir), SNAPSHOT_SHA256)
    periods = snap.periods
    ever = set().union(*(m for _, m in periods))
    bars = {}
    for s in sorted(ever & set(snap.symbols)):
        rows = {b.date: (b.open, b.high, b.low, b.close, b.volume) for b in snap.bars(s) if b.date <= end}
        if rows:
            bars[s] = rows
    calendar = [d for d in snap.sessions if d <= end]
    return {"calendar": calendar, "bars": bars, "periods": periods, "snapshotSha256": snap.sha256,
            "missingMembers": sorted(ever - set(bars))}


def run_variant(inputs: Mapping[str, Any], portfolio: str, variant: str,
                indicators: Mapping[str, Mapping[str, DayIndicators]],
                protocol_path: Path | str = PROTOCOL_PATH) -> tuple[pb.BacktestResult, StopReduceRule | None]:
    score, overrides = SCORES[portfolio]
    rules = pb.PortfolioRules.from_protocol(Path(protocol_path), **overrides)
    rule = None if variant == "none" else StopReduceRule(indicators, reduce=variant == "stop-8+reduce-4",
                                                         lot_size=rules.costs.lot_size)
    res = pb.run_backtest(inputs["calendar"], inputs["bars"], inputs["periods"], score, rules,
                          end=WINDOW_TO, risk_exit=rule)
    return res, rule


def evaluate(inputs: Mapping[str, Any], *, registry_path: Path | str = trial_registry.REGISTRY_PATH,
             protocol_path: Path | str = PROTOCOL_PATH, spec_path: Path | str = SPEC_PATH) -> dict[str, Any]:
    entry = registered_entry(registry_path, spec_path)
    count = pe.registry_trial_count(registry_path, "HOSE")
    indicators = {s: indicators_by_date(b) for s, b in inputs["bars"].items()}
    runs: dict[str, dict[str, Any]] = {}
    rets: dict[str, dict[str, float]] = {}
    for p in PORTFOLIOS:
        for v in VARIANTS:
            res, rule = run_variant(inputs, p, v, indicators, protocol_path)
            key = f"{p}/{v}"
            if res.daily[-1]["date"] > WINDOW_TO or res.daily[0]["date"] < WINDOW_FROM:
                raise RuntimeError(f"{key}: simulation left the development window")
            exits = [t for t in res.trades if t.get("origin") == "risk-exit"]
            events = rule.events if rule else []
            runs[key] = {**nav_metrics(res.daily, res.summary),
                         "stopSignals": sum(e["kind"] == "stop" for e in events),
                         "reduceSignals": sum(e["kind"] == "reduce" for e in events),
                         "riskExitFills": len(exits),
                         "riskExitTradedVnd": sum(t["grossVnd"] for t in exits),
                         "supersededByRiskExit": sum(r["reason"] == "superseded-by-risk-exit"
                                                     for r in res.rejections),
                         "rules": res.rules.to_dict()}
            rets[key] = daily_returns(res.daily)
    n = count["trials"]
    absolute = dsr_table(rets, n)
    excess_series = {f"{p}/{v} - none": {d: rets[f"{p}/{v}"][d] - rets[f"{p}/none"][d] for d in rets[f"{p}/none"]}
                     for p in PORTFOLIOS for v in VARIANTS[1:]}
    excess = dsr_table(excess_series, n)
    verdicts = {f"{p}/{v}": decide(runs[f"{p}/none"], runs[f"{p}/{v}"],
                                   excess["series"][f"{p}/{v} - none"]["dsr"])
                for p in PORTFOLIOS for v in VARIANTS[1:]}
    return {"schema": "stops-eval/v1", "spec": SPEC_ID, "specSha256": sha256_file(spec_path),
            "protocolSha256": sha256_file(protocol_path), "evaluatorSha256": pe.evaluator_sha256(),
            "registryEntry": entry["id"], "registryCount": count, "snapshot": SNAPSHOT_ID,
            "snapshotSha256": inputs["snapshotSha256"],
            "window": {"from": WINDOW_FROM, "to": WINDOW_TO}, "missingMembers": inputs["missingMembers"],
            "runs": runs, "dsrAbsolute": _strip(absolute), "dsrExcessVsControl": _strip(excess),
            "verdicts": verdicts,
            "conclusion": ("supported" if any(x["verdict"] == "supported" for x in verdicts.values())
                           else "not supported")}


def _strip(table: Mapping[str, Any]) -> dict[str, Any]:
    keep = ("n", "sharpe", "skew", "kurtosis", "nTrials", "expectedMaxSharpe", "dsr", "psrVsZero")
    return {"srVariance": {k: table["srVariance"][k] for k in ("estimator", "crossTrial", "estimatorFloor", "value")},
            "series": {k: {f: v[f] for f in keep} for k, v in table["series"].items()}}


def result_line(result: Mapping[str, Any]) -> str:
    """One-line registry summary of a result."""
    parts = []
    for key, r in result["runs"].items():
        parts.append(f"{key}: CAGR {r['netCagr']:.4f}, Sharpe {r['sharpeAnnualized']:.3f}, MDD {r['maxDrawdown']:.4f}")
    dsr = ", ".join(f"{k} {v['dsr']:.3f}" for k, v in result["dsrExcessVsControl"]["series"].items())
    return (f"{result['conclusion']} (dev 2018-01-22..2023-12-31, N={result['registryCount']['trials']}). "
            + "; ".join(parts) + f". Excess DSR: {dsr}.")


def record_run(result_path: Path | str, *, registry_path: Path | str = trial_registry.REGISTRY_PATH,
               evaluated_at: str | None = None) -> dict[str, Any]:
    """Supersede the planned entry with a status-run entry carrying the result."""
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    planned = registered_entry(registry_path)
    if planned["status"] != "planned" or planned["id"] != REGISTRY_ID:
        raise ValueError(f"{REGISTRY_ID} is not an active planned entry (already recorded?)")
    if result["specSha256"] != planned["specSha256"]:
        raise ValueError("result was produced under another spec version")
    return trial_registry.append({
        "id": f"{REGISTRY_ID}-run", "registeredAt": datetime.now(timezone.utc).date().isoformat(),
        "market": planned["market"], "family": planned["family"], "status": "run", "trials": planned["trials"],
        "trialsExact": planned["trialsExact"], "description": planned["description"], "source": planned["source"],
        "specSha256": planned["specSha256"], "supersedes": REGISTRY_ID,
        "evaluatedAt": evaluated_at or datetime.now(timezone.utc).date().isoformat(),
        "result": result_line(result)}, registry_path)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="STOPS-1: backtest of the -8% stop / -4% reduce rules")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--snapshot", default=None)
    run.add_argument("--out-root", default=None)
    rec = sub.add_parser("record")
    rec.add_argument("result")
    args = parser.parse_args(argv)
    if args.command == "record":
        print(json.dumps(record_run(args.result), ensure_ascii=False))
        return 0
    inputs = load_inputs(args.snapshot or default_snapshot_dir())
    result = evaluate(inputs)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"  # serialize before creating the folder
    out_dir = Path(args.out_root or pe.default_evidence_root()) / f"{SPEC_ID}-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "result.json").write_text(payload, encoding="utf-8")
    print(json.dumps({"out": str(out_dir / "result.json"), "conclusion": result["conclusion"],
                      "summary": result_line(result)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
