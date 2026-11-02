"""VN30-1: point-in-time VN30 rotation backtest, as pre-registered in evidence/vn30-1-20261102/spec.json.

On the last session of each month, rank the VN30 members in effect that day by trailing return over
21 x lookbackMonths sessions (only sessions <= that day), buy the top K equal-weight at the next session's
open, hold to the next rebalance. Fees and the sell tax are charged on traded value. Benchmarks: VNINDEX
buy-and-hold and an equal-weight VN30 basket with the same execution and costs.

Usage (from ai/, venv python; not -I, which drops ai/ from sys.path):
  python vn30_rotation.py --spec <spec.json> --membership <vn30-membership.json> --data <dir> --out <results.json>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from bisect import bisect_right
from datetime import date
from pathlib import Path
from statistics import NormalDist

from equity_base_rates import Costs

SESSIONS_PER_MONTH = 21
INDEX = "VNINDEX"
EULER_GAMMA = 0.5772156649015329

Prices = dict[str, dict[str, tuple[float, float]]]  # symbol -> session date -> (open, close)
Periods = list[tuple[str, frozenset[str]]]  # (effectiveDate, constituents), chronological


def load_membership(path: Path) -> Periods:
    periods = []
    for p in json.loads(path.read_text(encoding="utf-8"))["periods"]:
        members = p["constituents"]
        if len(members) != 30 or len(set(members)) != 30:
            raise ValueError(f"{p['effectiveDate']}: expected 30 unique constituents, got {len(set(members))}")
        periods.append((p["effectiveDate"], frozenset(members)))
    dates = [d for d, _ in periods]
    if dates != sorted(set(dates)):
        raise ValueError("membership periods must have unique, chronological effective dates")
    return periods


def members_on(periods: Periods, day: str) -> frozenset[str] | None:
    i = bisect_right([d for d, _ in periods], day) - 1
    return periods[i][1] if i >= 0 else None


def load_prices(data_dir: Path) -> Prices:
    """Zero-volume bars are not sessions (spec-addendum.json): pre-listing placeholders, provider holes."""
    prices: Prices = {}
    for path in sorted(data_dir.glob("*.csv")):
        with path.open(encoding="utf-8") as f:
            prices[path.stem] = {r["date"]: (float(r["open"]), float(r["close"]))
                                 for r in csv.DictReader(f) if float(r["volume"]) > 0}
    return prices


def select(calendar: list[str], prices: Prices, members: frozenset[str], i: int,
           lookback_months: int, top_k: int | None) -> list[str]:
    """Members to hold after decision session calendar[i]; reads no session after it.

    lookback_months = 0 is the equal-weight benchmark: every member with a close that day.
    """
    day = calendar[i]
    if lookback_months == 0:
        return sorted(s for s in members if day in prices.get(s, {}))
    j = i - SESSIONS_PER_MONTH * lookback_months
    if j < 0:
        return []
    past = calendar[j]
    scored = []
    for s in members:
        bars = prices.get(s, {})
        if day in bars and past in bars:
            scored.append((-(bars[day][1] / bars[past][1] - 1.0), s))
    return [s for _, s in sorted(scored)[:top_k]]


def month_end_indices(calendar: list[str], start: str, end: str) -> list[int]:
    """Last session of each month inside [start, end] that still has a next session to trade on."""
    return [i for i in range(len(calendar) - 1)
            if start <= calendar[i] <= end and calendar[i + 1][:7] != calendar[i][:7]]


def _exec_price(prices: Prices, symbol: str, day: str) -> float | None:
    bar = prices.get(symbol, {}).get(day)
    if bar is None:
        return None
    return bar[0] if bar[0] > 0 else bar[1]


def rebalance(units: dict[str, float], cash: float, targets: list[str], weight: float, prices: Prices,
              day: str, last_close: dict[str, float], costs: Costs) -> tuple[dict[str, float], float, float]:
    """Trade at the open of `day` to `weight` per target; returns (units, cash, turnover / value).

    A held symbol with no bar that day cannot trade and keeps its units; targets with no bar stay cash.
    Costs are solved by fixed point so the post-cost value is what gets allocated.
    """
    price = {s: _exec_price(prices, s, day) for s in set(units) | set(targets)}
    frozen = {s for s in units if price[s] is None}
    mark = {s: (price[s] if price[s] is not None else last_close[s]) for s in units}
    value = cash + sum(u * mark[s] for s, u in units.items())
    frozen_value = sum(units[s] * mark[s] for s in frozen)
    tradable = [s for s in targets if price[s] is not None and s not in frozen]
    net, sells, buys, target_value = value, 0.0, 0.0, {}
    for _ in range(50):
        per_name = min(net * weight, (net - frozen_value) / len(tradable)) if tradable else 0.0
        target_value = {s: per_name for s in tradable}
        sells = sum(max(0.0, units[s] * mark[s] - target_value.get(s, 0.0)) for s in units if s not in frozen)
        buys = sum(max(0.0, v - units.get(s, 0.0) * mark.get(s, 0.0)) for s, v in target_value.items())
        new_net = value - sells * (costs.sell_fee + costs.sell_tax) - buys * costs.buy_fee
        if abs(new_net - net) < 1e-13:
            break
        net = new_net
    new_units = {s: units[s] for s in frozen}
    for s, v in target_value.items():
        new_units[s] = v / price[s]
        last_close[s] = price[s]
    new_cash = net - frozen_value - sum(target_value.values())
    return new_units, new_cash, (sells + buys) / value


def simulate(calendar: list[str], prices: Prices, periods: Periods, lookback_months: int,
             top_k: int | None, costs: Costs, start: str, end: str) -> dict:
    decisions = set(month_end_indices(calendar, start, end))
    units: dict[str, float] = {}
    last_close: dict[str, float] = {}
    cash, pending, started = 1.0, None, False
    dates, values, turnover, holdings = [], [], [], []
    for i, day in enumerate(calendar):
        if day > end:
            break
        if pending is not None:
            weight = 1.0 / (top_k if top_k else max(len(pending), 1))
            units, cash, t = rebalance(units, cash, pending, weight, prices, day, last_close, costs)
            turnover.append(t)
            holdings.append({"date": day, "held": sorted(units)})
            started, pending = True, None
        for s in units:
            bar = prices.get(s, {}).get(day)
            if bar is not None:
                last_close[s] = bar[1]
        if started:
            dates.append(day)
            values.append(cash + sum(u * last_close[s] for s, u in units.items()))
        if i in decisions:
            members = members_on(periods, day)
            if members:
                pending = select(calendar, prices, members, i, lookback_months, top_k)
    return {"dates": dates, "values": values, "turnover": turnover, "holdings": holdings}


def _moments(rets: list[float]) -> tuple[float, float, float, float]:
    mean = statistics.fmean(rets)
    sd = statistics.pstdev(rets)
    skew = sum((r - mean) ** 3 for r in rets) / len(rets) / sd ** 3 if sd else 0.0
    kurt = sum((r - mean) ** 4 for r in rets) / len(rets) / sd ** 4 if sd else 3.0
    return mean, sd, skew, kurt


def psr(sr: float, sr_star: float, n: int, skew: float, kurt: float) -> float:
    """Probabilistic Sharpe ratio (per-period SR; kurt is non-excess)."""
    den = math.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr))
    return NormalDist().cdf((sr - sr_star) * math.sqrt(n - 1) / den)


def expected_max_sr(trial_srs: list[float]) -> float:
    """Expected maximum per-period SR among len(trial_srs) unskilled trials (Bailey & Lopez de Prado)."""
    n = len(trial_srs)
    if n < 2:
        return 0.0
    z = NormalDist().inv_cdf
    return math.sqrt(statistics.variance(trial_srs)) * (
        (1 - EULER_GAMMA) * z(1 - 1 / n) + EULER_GAMMA * z(1 - 1 / (n * math.e)))


def metrics(dates: list[str], values: list[float], turnover: list[float] | None = None) -> dict:
    rets = [b / a - 1.0 for a, b in zip(values, values[1:])]
    years = (date.fromisoformat(dates[-1]) - date.fromisoformat(dates[0])).days / 365.25
    mean, sd, skew, kurt = _moments(rets)
    peak, max_dd = values[0], 0.0
    for v in values:
        peak = max(peak, v)
        max_dd = min(max_dd, v / peak - 1.0)
    out = {"from": dates[0], "to": dates[-1], "sessions": len(values),
           "cagr": (values[-1] / values[0]) ** (1 / years) - 1.0 if years > 0 else 0.0,
           "volatility": sd * math.sqrt(252), "sharpe": mean / sd * math.sqrt(252) if sd else 0.0,
           "maxDrawdown": max_dd, "dailySharpe": mean / sd if sd else 0.0, "skew": skew, "kurtosis": kurt,
           "psrVsZero": psr(mean / sd, 0.0, len(rets), skew, kurt) if sd else 0.5}
    if turnover is not None and years > 0:
        out["turnoverPerYear"] = sum(turnover) / years
    return out


def window(dates: list[str], values: list[float], start: str, end: str) -> tuple[list[str], list[float]]:
    keep = [(d, v) for d, v in zip(dates, values) if start <= d <= end]
    return [d for d, _ in keep], [v for _, v in keep]


def run(spec: dict, periods: Periods, prices: Prices) -> dict:
    calendar = sorted(prices[INDEX])
    start = spec["periods"]["design"]["from"]
    design_end = spec["periods"]["design"]["to"]
    oos_start = spec["periods"]["outOfSample"]["from"]
    end = calendar[-1]
    cost_sets = {k: Costs(v["buyFeeFraction"], v["sellFeeFraction"], v["sellTaxFraction"])
                 for k, v in spec["costs"].items() if k in ("base", "sensitivity")}

    def score(sim: dict) -> dict:
        d, v = sim["dates"], sim["values"]
        return {"full": metrics(d, v, sim["turnover"]),
                "design": metrics(*window(d, v, start, design_end)),
                "outOfSample": metrics(*window(d, v, oos_start, end))}

    configs = []
    for lookback in spec["signal"]["lookbackMonths"]:
        for k in spec["portfolio"]["topK"]:
            row = {"lookbackMonths": lookback, "topK": k}
            for name, costs in cost_sets.items():
                sim = simulate(calendar, prices, periods, lookback, k, costs, start, end)
                row[name] = score(sim)
                if name == "base":
                    row["lastHoldings"] = sim["holdings"][-1] if sim["holdings"] else None
            configs.append(row)
    first, last = configs[0]["base"]["full"]["from"], end
    ew = simulate(calendar, prices, periods, 0, None, cost_sets["base"], start, end)
    index_dates = [d for d in calendar if first <= d <= last]
    index_values = [prices[INDEX][d][1] for d in index_dates]

    def bench(dates: list[str], values: list[float]) -> dict:
        return {"full": metrics(*window(dates, values, first, last)),
                "design": metrics(*window(dates, values, first, design_end)),
                "outOfSample": metrics(*window(dates, values, oos_start, last))}

    benchmarks = {"VNINDEX": bench(index_dates, index_values), "equalWeightVN30": bench(ew["dates"], ew["values"])}
    trial_srs = [c["base"]["full"]["dailySharpe"] for c in configs]
    sr_star = expected_max_sr(trial_srs)
    for c in configs:
        f = c["base"]["full"]
        c["base"]["full"]["deflatedSharpe"] = psr(f["dailySharpe"], sr_star, f["sessions"] - 1, f["skew"], f["kurtosis"])
    best = max(configs, key=lambda c: c["base"]["design"]["sharpe"])
    oos = best["base"]["outOfSample"]
    worth = (oos["cagr"] > benchmarks["VNINDEX"]["outOfSample"]["cagr"]
             and oos["cagr"] > benchmarks["equalWeightVN30"]["outOfSample"]["cagr"]
             and best["base"]["full"]["deflatedSharpe"] >= 0.95)
    return {"configs": configs, "benchmarks": benchmarks, "trials": len(trial_srs), "expectedMaxDailySharpe": sr_star,
            "selected": {"lookbackMonths": best["lookbackMonths"], "topK": best["topK"]},
            "verdict": "worth-following" if worth else "not-enough-evidence"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--membership", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = run(json.loads(args.spec.read_text(encoding="utf-8")), load_membership(args.membership),
                 load_prices(args.data))
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"selected": result["selected"], "verdict": result["verdict"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
