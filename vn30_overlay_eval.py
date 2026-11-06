"""OVERLAY-1 (roadmap 3.2): regime and volatility-target exposure overlays on VN30 portfolios.

Roadmap item 3.2 asks whether a regime filter (MA200 of the index) or volatility targeting reduces the drawdown
of the VN30 portfolios. It is not part of the frozen vn30-2 protocol, so it is its own trial:
docs/research/protocols/overlay-1.json, registered in the trial registry (entry ``hose-overlay-1``, status planned,
4 trials) with the spec's sha256 before any result was computed.

Design (read the spec for the exact wording):

- Portfolios: P-ew (B-ew, every eligible member) and P-composite (B-composite, top 10 keep 15), because VN30-2 run
  20261106T121249Z found no development winner. Engine: ``vn_portfolio_backtest.run_backtest`` with the vn30-2
  rules (cash_settlement_sessions 0).
- Variants: none (control), ma200-cash (exposure 0 when the VN30 index close of the decision session is below its
  200-close mean, else 1), vol-target-15 (exposure min(1, 0.15 / trailing 60-session annualized vol of the
  equal-weight basket of the decision's targets)). The exposure is fixed at the decision close and scales the
  rebalance budget at the next open (``run_backtest(exposure=...)``).
- Window: development only, 2018-01-22..2023-12-31; every bar after 2023-12-31 is removed before the simulation, so
  the vn30-2 holdout is never read.
- Metrics: net CAGR, Sharpe, max drawdown vs the control and vs P-ew/none; DSR with N = HOSE trials (run + planned)
  from the registry at evaluation time. Decision rule as stops-1 (CAGR, Sharpe, MDD all better AND excess DSR >= 0.95).

    python vn30_overlay_eval.py run [--snapshot <dir>] [--out-root <dir>]
    python vn30_overlay_eval.py record <result.json>    # supersede the planned entry with status run

Pure computation besides reading the snapshot and writing the result; no network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import trial_registry
import vn30_2_run as v2
import vn30_panel_eval as pe
import vn30_stops_eval as se
import vn_portfolio_backtest as pb

SPEC_ID = "overlay-1"
REPO = Path(__file__).resolve().parent
SPEC_PATH = REPO / "docs" / "research" / "protocols" / f"{SPEC_ID}.json"
PROTOCOL_PATH = REPO / "docs" / "research" / "protocols" / "vn30-2.json"
REGISTRY_ID = "hose-overlay-1"
SNAPSHOT_ID = "20261106-9afb8020ec16"
SNAPSHOT_SHA256 = "9afb8020ec16a1c26669b046b5621453012e3cd09a5dc24b70da754648acc94e"
WINDOW_FROM = "2018-01-22"
WINDOW_TO = "2023-12-31"
REGIME_INDEX = "VN30"
MA_BARS = 200
VOL_TARGET = 0.15
VOL_BARS = 60
PERIODS_PER_YEAR = 252
PORTFOLIOS = ("P-ew", "P-composite")
VARIANTS = ("none", "ma200-cash", "vol-target-15")
CONTROL = "P-ew/none"
RESULT_SCHEMA = "overlay-eval/v1"


# --------------------------------------------------------------------------------------------- scores


def ew_score(day: str, view: pb.PointInTimeView) -> dict[str, float]:
    return {s: 0.0 for s in view.members}


@dataclass
class CompositeScore:
    """B-composite at the decision close: mean of the 8 vn30-2 features, each rank-normalized over the eligible
    members (``view.members``: in force with >= 260 prior bars), as vn30_2_run.build_panel computes them."""

    series: Mapping[str, v2.SymbolSeries]

    def __call__(self, day: str, view: pb.PointInTimeView) -> dict[str, float]:
        names = sorted(s for s in view.members if s in self.series)
        if not names:
            return {}
        raw = [v2.raw_features(self.series[s], day) for s in names]
        ranks = np.column_stack([v2.rank_normalize([r[f] for r in raw]) for f in v2.FEATURES])
        return {s: float(x) for s, x in zip(names, ranks.mean(axis=1))}


# --------------------------------------------------------------------------------------------- overlays


@dataclass
class Ma200Cash:
    """Exposure 1 when the index close of the decision session is >= the mean of its last 200 closes dated <= d
    (d included), 0 below; 1 while fewer than 200 closes exist."""

    index_closes: Mapping[str, float]
    bars: int = MA_BARS
    log: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        self._dates = sorted(self.index_closes)
        self._closes = np.array([float(self.index_closes[d]) for d in self._dates])
        if self.log is None:
            self.log = []

    def state(self, day: str) -> dict[str, Any]:
        k = bisect_right(self._dates, day)
        if k < self.bars:
            return {"date": day, "close": float(self._closes[k - 1]) if k else None, "ma": None, "exposure": 1.0}
        window = self._closes[k - self.bars:k]
        close, ma = float(window[-1]), float(window.mean())
        return {"date": day, "indexDate": self._dates[k - 1], "close": close, "ma": ma,
                "exposure": 1.0 if close >= ma else 0.0}

    def __call__(self, day: str, view: pb.PointInTimeView, targets: Sequence[str]) -> float:
        row = self.state(day)
        self.log.append(row)
        return row["exposure"]


def basket_returns(calendar: Sequence[str], bars: Mapping[str, Mapping[str, tuple]], names: Sequence[str],
                   day: str, n: int = VOL_BARS) -> np.ndarray:
    """The last ``n`` daily returns (sessions ending at ``day``) of the equal-weight basket of ``names``; a name's
    return on session t uses its last close dated <= t and <= t-1 (0 when it did not trade on t)."""
    k = bisect_right(calendar, day) - 1
    if k < n or not names:
        return np.array([])
    sessions = calendar[k - n:k + 1]  # n + 1 sessions -> n returns
    cols = []
    for s in names:
        rows = bars.get(s, {})
        ds = sorted(d for d in rows if d <= day)
        if not ds:
            continue
        closes = []
        for t in sessions:
            j = bisect_right(ds, t) - 1
            closes.append(float(rows[ds[j]][3]) if j >= 0 else math.nan)
        c = np.array(closes)
        r = c[1:] / c[:-1] - 1.0
        cols.append(np.where(np.isfinite(r), r, 0.0))
    if not cols:
        return np.array([])
    return np.column_stack(cols).mean(axis=1)


@dataclass
class VolTarget:
    """Exposure min(1, target / sigma), sigma = sd (ddof 1) of the last 60 basket returns of the targets x sqrt(252);
    1 when sigma cannot be computed or is 0."""

    calendar: Sequence[str]
    bars: Mapping[str, Mapping[str, tuple]]
    target: float = VOL_TARGET
    window: int = VOL_BARS
    log: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.log is None:
            self.log = []

    def state(self, day: str, targets: Sequence[str]) -> dict[str, Any]:
        r = basket_returns(self.calendar, self.bars, targets, day, self.window)
        sigma = float(np.std(r, ddof=1) * math.sqrt(PERIODS_PER_YEAR)) if len(r) == self.window else math.nan
        exposure = min(1.0, self.target / sigma) if math.isfinite(sigma) and sigma > 0 else 1.0
        return {"date": day, "names": len(targets), "sigma": sigma if math.isfinite(sigma) else None,
                "exposure": exposure}

    def __call__(self, day: str, view: pb.PointInTimeView, targets: Sequence[str]) -> float:
        row = self.state(day, targets)
        self.log.append(row)
        return row["exposure"]


# --------------------------------------------------------------------------------------------- registry / inputs


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registered_entry(registry_path: Path | str = trial_registry.REGISTRY_PATH,
                     spec_path: Path | str = SPEC_PATH) -> dict[str, Any]:
    """The active overlay-1 registry entry; fails unless it was registered for this exact spec file."""
    sha = sha256_file(spec_path)
    found = [e for e in trial_registry.load(registry_path).active()
             if e["id"] == REGISTRY_ID or e.get("supersedes") == REGISTRY_ID]
    if not found:
        raise ValueError(f"no active registry entry {REGISTRY_ID}: register the trial before running it")
    entry = found[-1]
    if entry.get("specSha256") != sha:
        raise ValueError(f"registry entry {entry['id']} was registered for another version of {SPEC_ID}.json")
    return entry


def check_spec(spec_path: Path | str = SPEC_PATH) -> dict[str, Any]:
    """The spec declares exactly the portfolios, variants, window and snapshot implemented here."""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    problems = []
    if spec.get("id") != SPEC_ID:
        problems.append("id")
    if tuple(p["id"] for p in spec["portfolios"]["list"]) != PORTFOLIOS:
        problems.append("portfolios")
    if tuple(v["id"] for v in spec["variants"]["list"]) != VARIANTS:
        problems.append("variants")
    if (spec["window"]["from"], spec["window"]["to"]) != (WINDOW_FROM, WINDOW_TO):
        problems.append("window")
    if (spec["data"]["snapshot"], spec["data"]["snapshotSha256"]) != (SNAPSHOT_ID, SNAPSHOT_SHA256):
        problems.append("snapshot")
    if problems:
        raise ValueError(f"{SPEC_ID}.json does not match the implementation: {', '.join(problems)}")
    return spec


def default_snapshot_dir() -> Path:
    return pe.repo_main_checkout().parent / "data" / "vn30-research" / "snapshots" / SNAPSHOT_ID


def load_market(snapshot_dir: Path | str, end: str = WINDOW_TO) -> tuple[v2.Market, str]:
    """vn30-2 market (members, leavers, VN30 / E1VFVN30 closes) from the verified snapshot, cut at ``end``."""
    import vn30_research_data as vrd

    snap = vrd.load_snapshot(Path(snapshot_dir), SNAPSHOT_SHA256)
    market = v2.market_from_snapshot(snap).truncate(end)
    if REGIME_INDEX not in market.benchmarks:
        raise ValueError(f"snapshot has no {REGIME_INDEX} index series")
    return market, snap.sha256


# --------------------------------------------------------------------------------------------- run


def make_overlay(variant: str, market: v2.Market) -> Callable | None:
    if variant == "none":
        return None
    if variant == "ma200-cash":
        return Ma200Cash(market.benchmarks[REGIME_INDEX])
    if variant == "vol-target-15":
        return VolTarget(market.calendar, market.bars)
    raise ValueError(f"unknown variant {variant}")


def portfolio_setup(portfolio: str, market: v2.Market, protocol_path: Path | str = PROTOCOL_PATH,
                    series: Mapping[str, v2.SymbolSeries] | None = None) -> tuple[Callable, pb.PortfolioRules]:
    if portfolio == "P-ew":
        return ew_score, pb.PortfolioRules.from_protocol(Path(protocol_path), cash_settlement_sessions=0,
                                                         top_k=None, keep_rank=None)
    if portfolio == "P-composite":
        series = series if series is not None else v2.symbol_series(market.bars)
        return CompositeScore(series), pb.PortfolioRules.from_protocol(Path(protocol_path),
                                                                       cash_settlement_sessions=0)
    raise ValueError(f"unknown portfolio {portfolio}")


def check_composite_against_panel(market: v2.Market, score: CompositeScore, decisions: Sequence[int]) -> int:
    """Integrity check: the on-the-fly composite equals vn30_2_run's panel composite on every decision date."""
    panel = v2.build_panel(market, decisions, series=score.series)
    comp = panel[list(v2.FCOLS)].to_numpy(dtype=float).mean(axis=1)
    checked = 0
    for day, rows in panel.assign(comp=comp).groupby("date"):
        view_members = frozenset(rows["symbol"])
        mine = score.__call__(str(day), _MembersOnly(view_members))
        for s, c in zip(rows["symbol"], rows["comp"]):
            if abs(mine[s] - c) > 1e-12:
                raise RuntimeError(f"composite mismatch {day} {s}: {mine[s]} vs panel {c}")
            checked += 1
    return checked


@dataclass(frozen=True)
class _MembersOnly:
    members: frozenset[str]


def exposure_stats(rebalances: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    xs = [float(r.get("exposure", 1.0)) for r in rebalances]
    return {"decisions": len(xs), "meanExposure": float(np.mean(xs)) if xs else None,
            "minExposure": min(xs) if xs else None, "shareZeroExposure": float(np.mean([x == 0 for x in xs]))
            if xs else None}


def comparison(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, float]:
    """a minus b in net CAGR, Sharpe and max drawdown (a positive MDD difference = shallower)."""
    return {"netCagr": a["netCagr"] - b["netCagr"], "sharpe": a["sharpeAnnualized"] - b["sharpeAnnualized"],
            "maxDrawdown": a["maxDrawdown"] - b["maxDrawdown"]}


def evaluate(market: v2.Market, snapshot_sha256: str, *, registry_path: Path | str = trial_registry.REGISTRY_PATH,
             protocol_path: Path | str = PROTOCOL_PATH, spec_path: Path | str = SPEC_PATH,
             verify_composite: bool = True) -> dict[str, Any]:
    check_spec(spec_path)
    entry = registered_entry(registry_path, spec_path)
    if market.last_date() > WINDOW_TO or any(d > WINDOW_TO for rows in market.bars.values() for d in rows) \
            or any(d > WINDOW_TO for rows in market.benchmarks.values() for d in rows):
        raise RuntimeError("market contains data after the development window")
    count = pe.registry_trial_count(registry_path, "HOSE")
    series = v2.symbol_series(market.bars)
    first = pb.first_decision_index(market.calendar, market.bars, market.periods)
    if first is None:
        raise ValueError("no session satisfies the first-decision history rule")
    runs: dict[str, dict[str, Any]] = {}
    rets: dict[str, dict[str, float]] = {}
    overlay_logs: dict[str, list[dict[str, Any]]] = {}
    composite_checked = None
    for p in PORTFOLIOS:
        score, rules = portfolio_setup(p, market, protocol_path, series)
        if verify_composite and isinstance(score, CompositeScore):
            decisions = list(range(first, len(market.calendar) - 1, rules.rebalance_every))
            composite_checked = check_composite_against_panel(market, score, decisions)
        for v in VARIANTS:
            overlay = make_overlay(v, market)
            res = pb.run_backtest(market.calendar, market.bars, market.periods, score, rules,
                                  first_decision=first, end=WINDOW_TO, exposure=overlay)
            key = f"{p}/{v}"
            if res.daily[-1]["date"] > WINDOW_TO or res.daily[0]["date"] < WINDOW_FROM:
                raise RuntimeError(f"{key}: simulation left the development window")
            runs[key] = {**se.nav_metrics(res.daily, res.summary), **exposure_stats(res.rebalances),
                         "rules": res.rules.to_dict()}
            rets[key] = se.daily_returns(res.daily)
            if overlay is not None:
                overlay_logs[key] = overlay.log
    dates = list(rets[CONTROL])
    if any(list(r) != dates for r in rets.values()):
        raise RuntimeError("daily rows differ between runs")
    n = count["trials"]
    absolute = se.dsr_table(rets, n)
    excess_ctrl = {f"{p}/{v} - none": {d: rets[f"{p}/{v}"][d] - rets[f"{p}/none"][d] for d in dates}
                   for p in PORTFOLIOS for v in VARIANTS[1:]}
    dsr_ctrl = se.dsr_table(excess_ctrl, n)
    excess_ew = {f"{k} - {CONTROL}": {d: rets[k][d] - rets[CONTROL][d] for d in dates} for k in rets if k != CONTROL}
    dsr_ew = se.dsr_table(excess_ew, n)
    vs_control = {f"{p}/{v}": comparison(runs[f"{p}/{v}"], runs[f"{p}/none"]) for p in PORTFOLIOS for v in VARIANTS[1:]}
    vs_ew = {k: comparison(runs[k], runs[CONTROL]) for k in runs if k != CONTROL}
    verdicts = {}
    for p in PORTFOLIOS:
        for v in VARIANTS[1:]:
            key = f"{p}/{v}"
            verdicts[key] = {**se.decide(runs[f"{p}/none"], runs[key], dsr_ctrl["series"][f"{key} - none"]["dsr"]),
                             "excessDsr": dsr_ctrl["series"][f"{key} - none"]["dsr"]}
    return {"schema": RESULT_SCHEMA, "spec": SPEC_ID, "specSha256": sha256_file(spec_path),
            "protocolSha256": sha256_file(protocol_path), "evaluatorSha256": pe.evaluator_sha256(),
            "registryEntry": entry["id"], "registryCount": count,
            "snapshot": SNAPSHOT_ID, "snapshotSha256": snapshot_sha256,
            "window": {"from": WINDOW_FROM, "to": WINDOW_TO, "firstDecision": market.calendar[first],
                       "lastSession": market.last_date()},
            "regimeIndex": REGIME_INDEX, "compositeRowsVerifiedAgainstPanel": composite_checked,
            "codeSha256": {m: sha256_file(REPO / m) for m in
                           ("vn30_overlay_eval.py", "vn_portfolio_backtest.py", "vn30_2_run.py", "vn30_stops_eval.py",
                            "vn30_panel_eval.py")},
            "runs": runs, "vsControl": vs_control, "vsEqualWeight": vs_ew,
            "dsrAbsolute": se._strip(absolute), "dsrExcessVsControl": se._strip(dsr_ctrl),
            "dsrExcessVsEqualWeight": se._strip(dsr_ew), "verdicts": verdicts,
            "conclusion": ("supported" if any(x["verdict"] == "supported" for x in verdicts.values())
                           else "not supported"),
            "overlayLog": overlay_logs}


def result_line(result: Mapping[str, Any]) -> str:
    """One-line registry summary of a result."""
    parts = [f"{k}: CAGR {r['netCagr']:.4f}, Sharpe {r['sharpeAnnualized']:.3f}, MDD {r['maxDrawdown']:.4f}, "
             f"mean exposure {r['meanExposure']:.3f}" for k, r in result["runs"].items()]
    dsr = ", ".join(f"{k} {v['dsr']:.3f}" for k, v in result["dsrExcessVsControl"]["series"].items())
    return (f"{result['conclusion']} (dev {WINDOW_FROM}..{WINDOW_TO}, N={result['registryCount']['trials']}). "
            + "; ".join(parts) + f". Excess DSR vs control: {dsr}.")


def record_run(result_path: Path | str, *, registry_path: Path | str = trial_registry.REGISTRY_PATH,
               spec_path: Path | str = SPEC_PATH, evaluated_at: str | None = None,
               evidence: str | None = None) -> dict[str, Any]:
    """Supersede the planned entry with a status-run entry carrying the result."""
    raw = Path(result_path).read_bytes()
    result = json.loads(raw.decode("utf-8"))
    planned = registered_entry(registry_path, spec_path)
    if planned["status"] != "planned" or planned["id"] != REGISTRY_ID:
        raise ValueError(f"{REGISTRY_ID} is not an active planned entry (already recorded?)")
    if result["specSha256"] != planned["specSha256"]:
        raise ValueError("result was produced under another spec version")
    today = datetime.now(timezone.utc).date().isoformat()
    fields = {"id": f"{REGISTRY_ID}-run", "registeredAt": today, "market": planned["market"],
              "family": planned["family"], "status": "run", "trials": planned["trials"],
              "trialsExact": planned["trialsExact"], "description": planned["description"],
              "source": planned["source"], "specSha256": planned["specSha256"], "supersedes": REGISTRY_ID,
              "evaluatedAt": evaluated_at or today, "result": result_line(result),
              "resultsSha256": hashlib.sha256(raw).hexdigest(), "snapshotSha256": result["snapshotSha256"]}
    if evidence:
        fields["evidence"] = evidence
    return trial_registry.append(fields, registry_path)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="OVERLAY-1: MA200 regime and 15% vol-target exposure overlays")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--snapshot", default=None)
    run.add_argument("--out-root", default=None)
    rec = sub.add_parser("record")
    rec.add_argument("result")
    args = parser.parse_args(argv)
    if args.command == "record":
        path = Path(args.result).resolve()
        evidence = f"docs/research/evidence/vn30-2/{path.parent.name}/{path.name}"
        print(json.dumps(record_run(path, evidence=evidence), ensure_ascii=False))
        return 0
    market, snap_sha = load_market(args.snapshot or default_snapshot_dir())
    result = evaluate(market, snap_sha)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = json.dumps(v2._clean(result), ensure_ascii=False, indent=2) + "\n"  # serialize before the folder
    out_dir = Path(args.out_root or pe.default_evidence_root()) / f"{SPEC_ID}-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "result.json").write_text(payload, encoding="utf-8", newline="\n")
    print(json.dumps({"out": str(out_dir / "result.json"), "conclusion": result["conclusion"],
                      "summary": result_line(result)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
