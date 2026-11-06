"""VN30-2 runner (roadmap 3.1): run the frozen protocol docs/research/protocols/vn30-2.json once.

Order of a run (``python vn30_2_run.py run``), every step fail-closed:

1. Preflight: the protocol file's sha256 must equal ``specSha256`` of the active, still *planned* registry entry
   ``hose-vn30-2``; the registry chain must verify; the protocol must declare exactly the 8 features and 8 models
   implemented here; the holdout lock of vn30-2 must not exist yet; the candle snapshot must load with its expected
   sha256 (``vn30_research_data.load_snapshot``).
2. Development: the market is truncated at the last session <= 2023-12-31 (nothing later is read). Features and
   targets are computed point in time, the 8 models (baselines included) are scored by purged expanding
   walk-forward (yearly refit, sealed folds of ``vn30_panel_eval.yearly_folds`` over the session calendar), rank IC
   (``vn30_panel_eval.window_rank_ic``, test rows of those folds only) and portfolios (``vn_portfolio_backtest``)
   are evaluated, the
   development gate is applied, and ``dev-results.json`` is written to the run folder BEFORE the holdout is opened.
3. Holdout: ``HoldoutLock.acquire`` (read once, shared by every worktree); the market is truncated at the last
   session <= 2026-09-18 (the post-FTSE regime is forward-only and never read); every model is evaluated for the
   report; the holdout gate is applied to the development winner only (no winner = no-go).
4. ``results.json``, ``report.md`` and ``daily-nav.csv`` in ``docs/research/evidence/vn30-2/<run-id>/`` of the main
   checkout, with the snapshot sha256, the registry head and a hash of the code state (git HEAD + diff + untracked
   files); then a registry entry superseding ``hose-vn30-2`` with status ``run`` and the result summary.

Choices made where the frozen protocol is silent (fixed in code before any result was seen, written into
``results.json`` under ``implementationChoices``):

* Sessions: the snapshot calendar (VNINDEX bars with volume > 0); a symbol's own bars exclude zero-volume bars and
  documented ``excludeBefore`` bars. Feature windows count the symbol's own bars dated <= d (the last bar <= d stands
  for d when the symbol did not trade on d); a feature without its full window is missing (rank 0).
* Universe per decision: members in force at d with >= 260 bars strictly before d (the backtester's filter, identical
  for every model). The target's equal-weight mean is over ALL members in force at d with a valid window return.
  A member without a bar at the open of d+1 or d+21 has no target (dropped from the IC of that date).
* Decision grid: every 20 sessions from ``first_decision_index`` (>= 260 prior bars for >= 20 members). The first
  calendar year of decisions has no earlier training data, so the development comparison covers the fold years
  2019-2023 for every model alike (baselines are scored on the same dates as the ML models).
* Portfolios start in cash at the first test decision of the window and end at the open of d+21 of the last decision
  whose label lies inside the window (that next decision gets no score, so everything is sold there). Per-period
  returns are NAV at the entry open to NAV at the next entry open; daily returns are close-to-close NAV changes.
* Net excess vs B-ew (annualized) = CAGR(model) - CAGR(B-ew) over the same span (calendar days / 365.25); the index
  and E1VFVN30 comparisons use their close at the first and last day of the span (price index, no costs).
* DSR and PBO use the DAILY net excess return series (model daily NAV return - B-ew daily NAV return) of the 7 ranked
  models over the window; DSR of the gated model with N from the registry (run + planned HOSE trials) and V[SR] by
  ``vn30_panel_eval.SR_VARIANCE_ESTIMATOR``; PBO = ``gatePbo`` of CSCV with 16 blocks. The holdout gate uses the
  holdout series; development-window values are reported for information.
* Development winner: M-ridge / M-lgbm passing the development gate; if both pass, the higher development mean IC.
* Cash: sale proceeds are buying power in the same session (``cash_settlement_sessions = 0``, the HOSE practice of
  buying with matched-but-unsettled sale proceeds; no advance interest charged). Shares are sellable from the open of
  T+2 (backtester default). Initial NAV 10 bn VND.
* Limit fills: the protocol's adjusted-series proxy is PRIMARY. When the raw CafeF snapshot of item 1.1 loads with
  every per-file sha256 verified, the 8 portfolios are also run with raw limit-lock detection (exchange reference,
  +/-7%, +/-20% after a > 25-session gap) as a SENSITIVITY registered beforehand as its own planned trial
  (``hose-vn30-2-rawlimits``); IC does not depend on fills. Sessions without usable raw data fall back to the proxy.

CLI (from the repo root, venv python):
  python vn30_2_run.py register-raw-sensitivity       # once, before any result is computed
  python vn30_2_run.py dry-run --out <dir outside the evidence root>   # dev phase + pseudo-holdout inside dev
  python vn30_2_run.py run                            # the one official run (opens the holdout)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
import sys
import traceback
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

import trial_registry
import vn30_panel_eval as ev
from vn30_rotation import Periods, members_on
from vn_portfolio_backtest import (BacktestResult, LimitInfo, PortfolioRules, first_decision_index, run_backtest,
                                   validate_limits)

REPO = Path(__file__).resolve().parent
PROTOCOL_PATH = ev.PROTOCOL_PATH
PROTOCOL_ID = ev.PROTOCOL_ID
REGISTRY_ENTRY_ID = "hose-vn30-2"
RAW_TRIAL_ID = "hose-vn30-2-rawlimits"
WORKSPACE = REPO.parent
DEFAULT_SNAPSHOT = WORKSPACE / "data" / "vn30-research" / "snapshots" / "20261106-9afb8020ec16"
DEFAULT_SNAPSHOT_SHA256 = "9afb8020ec16a1c26669b046b5621453012e3cd09a5dc24b70da754648acc94e"
DEFAULT_RAW_DIR = WORKSPACE / "data" / "vn30-research" / "raw"
RESULTS_SCHEMA = "vn30-2-results/v1"

FEATURES = ("mom_12_1", "mom_6", "mom_3", "rev_1", "vol_60", "liq_20", "dist_52w_high", "trend_ma200")
FCOLS = tuple(f"f_{f}" for f in FEATURES)
MODELS = ("B-ew", "B-mom_12_1", "B-mom_6", "B-rev_1", "B-lowvol", "B-composite", "M-ridge", "M-lgbm")
RANKED = MODELS[1:]
ML_MODELS = ("M-ridge", "M-lgbm")
BASELINE_FEATURE = {"B-mom_12_1": "mom_12_1", "B-mom_6": "mom_6", "B-rev_1": "rev_1", "B-lowvol": "vol_60"}
SEMI_SEALED = ("B-mom_6", "B-mom_12_1")  # protocol: seen on the holdout by VN30-1, reported but cannot confirm
RIDGE_ALPHA = 1.0
LGBM_SEED = 20261106
LGBM_PARAMS = {"num_leaves": 15, "n_estimators": 200, "learning_rate": 0.05, "min_child_samples": 50,
               "colsample_bytree": 0.8, "subsample": 1.0, "subsample_freq": 0, "random_state": LGBM_SEED,
               "n_jobs": 1, "deterministic": True, "force_row_wise": True, "verbose": -1}
MIN_IC_NAMES = 10
TRADING_DAYS = 252
LABEL_SESSIONS = 21  # entry at the open of d+1, exit at the open of d+21
BENCHMARK_SERIES = ("VN30", "E1VFVN30")
GATE_DEV_T_MIN = 2.0

IMPLEMENTATION_CHOICES = [" ".join(item.split()) for item in
                          __doc__.split("Choices made where the frozen protocol is silent")[1]
                          .split("CLI (from")[0].split("\n* ")[1:]]


class RunRefused(RuntimeError):
    """A precondition of the official run does not hold."""


# --------------------------------------------------------------------------------------------- market


@dataclass(frozen=True)
class Market:
    """Calendar, member bars (symbol -> date -> (open, high, low, close, volume)), membership and benchmark closes."""

    calendar: tuple[str, ...]
    bars: Mapping[str, Mapping[str, tuple]]
    periods: Periods
    benchmarks: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def truncate(self, last_day: str) -> "Market":
        """Everything dated after ``last_day`` removed (the only data later phases may read)."""
        return Market(tuple(d for d in self.calendar if d <= last_day),
                      {s: {d: b for d, b in rows.items() if d <= last_day} for s, rows in self.bars.items()},
                      self.periods,
                      {k: {d: v for d, v in rows.items() if d <= last_day} for k, rows in self.benchmarks.items()})

    def last_date(self) -> str:
        return self.calendar[-1]


def market_from_snapshot(snapshot: Any) -> Market:
    """Member bars (roles member / leaver) and benchmark closes of a verified ``vn30_research_data.Snapshot``."""
    bars = {}
    for sym in snapshot.symbols:
        if snapshot.role(sym) not in ("member", "leaver-at-first-period"):
            continue
        bars[sym] = {b.date: (b.open, b.high, b.low, b.close, b.volume) for b in snapshot.bars(sym)}
    benchmarks = {k: {b.date: b.close for b in snapshot.bars(k)} for k in BENCHMARK_SERIES if k in snapshot.symbols}
    return Market(tuple(snapshot.sessions), bars, snapshot.periods, benchmarks)


# --------------------------------------------------------------------------------------------- features / panel


@dataclass(frozen=True)
class SymbolSeries:
    dates: list[str]
    high: np.ndarray
    close: np.ndarray
    volume: np.ndarray


def symbol_series(bars: Mapping[str, Mapping[str, tuple]]) -> dict[str, SymbolSeries]:
    out = {}
    for s, rows in bars.items():
        ds = sorted(rows)
        out[s] = SymbolSeries(ds, np.array([rows[d][1] for d in ds], dtype=float),
                              np.array([rows[d][3] for d in ds], dtype=float),
                              np.array([rows[d][4] for d in ds], dtype=float))
    return out


def raw_features(series: SymbolSeries, day: str) -> dict[str, float]:
    """The 8 protocol features from the symbol's bars dated <= ``day`` (NaN without a full window)."""
    out = dict.fromkeys(FEATURES, math.nan)
    k = bisect_right(series.dates, day) - 1
    if k < 0:
        return out
    c, h, v = series.close, series.high, series.volume
    if k >= 252:
        out["mom_12_1"] = c[k - 21] / c[k - 252] - 1.0
    if k >= 126:
        out["mom_6"] = c[k] / c[k - 126] - 1.0
    if k >= 63:
        out["mom_3"] = c[k] / c[k - 63] - 1.0
    if k >= 21:
        out["rev_1"] = -(c[k] / c[k - 21] - 1.0)
    if k >= 60:
        out["vol_60"] = -float(np.std(np.diff(np.log(c[k - 60:k + 1])), ddof=1))
    if k >= 19:
        med = float(np.median(c[k - 19:k + 1] * v[k - 19:k + 1]))
        out["liq_20"] = math.log(med) if med > 0 else math.nan
    if k >= 251:
        out["dist_52w_high"] = c[k] / float(np.max(h[k - 251:k + 1])) - 1.0
    if k >= 199:
        out["trend_ma200"] = c[k] / float(np.mean(c[k - 199:k + 1])) - 1.0
    return {f: float(x) for f, x in out.items()}


def rank_normalize(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Cross-sectional rank (average ties) scaled to [-0.5, 0.5]; missing -> 0; a single value -> 0."""
    x = np.asarray(values, dtype=float)
    out = np.zeros(len(x))
    ok = np.isfinite(x)
    n = int(ok.sum())
    if n >= 2:
        out[ok] = (pd.Series(x[ok]).rank(method="average").to_numpy() - 1.0) / (n - 1) - 0.5
    return out


def decision_indices(market: Market, rebalance: int = 20, min_history: int = 260) -> list[int]:
    first = first_decision_index(market.calendar, market.bars, market.periods, min_history)
    if first is None:
        raise ValueError("no session satisfies the first-decision history rule")
    return list(range(first, len(market.calendar), rebalance))


def build_panel(market: Market, decisions: Sequence[int], *, min_history: int = 260,
                series: Mapping[str, SymbolSeries] | None = None) -> pd.DataFrame:
    """One row per (decision, eligible member): raw features, rank-normalized ``f_*`` features, return and target.

    The target needs the open of d+21, so it exists only when that session is inside ``market`` (a truncated market
    therefore never yields a label beyond its last session).
    """
    series = series if series is not None else symbol_series(market.bars)
    cal = market.calendar
    rows: list[dict[str, Any]] = []
    for i in decisions:
        day = cal[i]
        in_force = members_on(market.periods, day)
        if not in_force:
            continue
        rets: dict[str, float] = {}
        if i + LABEL_SESSIONS < len(cal):
            entry, exit_ = cal[i + 1], cal[i + LABEL_SESSIONS]
            for s in in_force:
                b0, b1 = market.bars.get(s, {}).get(entry), market.bars.get(s, {}).get(exit_)
                if b0 is not None and b1 is not None and b0[0] > 0:
                    rets[s] = b1[0] / b0[0] - 1.0
        bench = float(np.mean(list(rets.values()))) if rets else math.nan
        for s in sorted(in_force):
            ser = series.get(s)
            if ser is None or bisect_left(ser.dates, day) < min_history:
                continue
            r = rets.get(s, math.nan)
            rows.append({"session": i, "date": day, "symbol": s, **raw_features(ser, day), "ret": r,
                         "benchmark": bench, "target": r - bench if math.isfinite(r) else math.nan})
    panel = pd.DataFrame(rows, columns=["session", "date", "symbol", *FEATURES, "ret", "benchmark", "target"])
    for f, col in zip(FEATURES, FCOLS):
        panel[col] = 0.0
        for _, idx in panel.groupby("session").groups.items():
            panel.loc[idx, col] = rank_normalize(panel.loc[idx, f].to_numpy(dtype=float))
    return panel.reset_index(drop=True)


# --------------------------------------------------------------------------------------------- models


def _finite_train(train: pd.DataFrame) -> pd.DataFrame:
    return train[np.isfinite(train["target"].to_numpy(dtype=float))]


def fit_predict_ridge(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    from sklearn.linear_model import Ridge

    tr = _finite_train(train)
    model = Ridge(alpha=RIDGE_ALPHA).fit(tr[list(FCOLS)].to_numpy(), tr["target"].to_numpy())
    return model.predict(test[list(FCOLS)].to_numpy())


def fit_predict_lgbm(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    from lightgbm import LGBMRegressor

    tr = _finite_train(train)
    model = LGBMRegressor(**LGBM_PARAMS).fit(tr[list(FCOLS)].to_numpy(), tr["target"].to_numpy())
    return model.predict(test[list(FCOLS)].to_numpy())


def _baseline(feature: str) -> Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]:
    col = f"f_{feature}"
    return lambda train, test: test[col].to_numpy(dtype=float)


def _composite(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return test[list(FCOLS)].to_numpy(dtype=float).mean(axis=1)


def _equal_weight(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return np.zeros(len(test))


FIT_PREDICT: dict[str, Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]] = {
    "B-ew": _equal_weight, **{m: _baseline(f) for m, f in BASELINE_FEATURE.items()},
    "B-composite": _composite, "M-ridge": fit_predict_ridge, "M-lgbm": fit_predict_lgbm}


# --------------------------------------------------------------------------------------------- protocol / registry


def check_protocol(protocol_path: Path = PROTOCOL_PATH,
                   registry_path: Path = trial_registry.REGISTRY_PATH) -> dict[str, Any]:
    """Protocol sha256 == specSha256 of the active, planned ``hose-vn30-2``; declared features/models == implemented."""
    sha = ev.protocol_sha256(protocol_path)
    spec = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    registry = trial_registry.load(registry_path)
    active = {e["id"]: e for e in registry.active()}
    entry = active.get(REGISTRY_ENTRY_ID)
    if entry is None:
        raise RunRefused(f"registry entry {REGISTRY_ENTRY_ID} is not active (already run or superseded)")
    if entry["status"] != "planned":
        raise RunRefused(f"registry entry {REGISTRY_ENTRY_ID} has status {entry['status']!r}, expected 'planned'")
    if entry.get("specSha256") != sha:
        raise RunRefused(f"protocol sha256 {sha} != registered specSha256 {entry.get('specSha256')}")
    if spec["id"] != PROTOCOL_ID or not spec.get("frozenBeforeResults"):
        raise RunRefused("protocol id or frozen flag unexpected")
    if tuple(spec["features"]["list"]) != FEATURES:
        raise RunRefused(f"protocol features {list(spec['features']['list'])} != implemented {list(FEATURES)}")
    if tuple(m["id"] for m in spec["models"]["list"]) != MODELS:
        raise RunRefused("protocol models differ from the implemented list")
    if entry["trials"] != len(MODELS):
        raise RunRefused(f"registry entry declares {entry['trials']} trials, protocol has {len(MODELS)} models")
    rules = {m["id"]: m["rule"] for m in spec["models"]["list"]}
    for literal in ("num_leaves 15", "n_estimators 200", "learning_rate 0.05", "min_data_in_leaf 50",
                    "feature_fraction 0.8", "bagging off", f"seed {LGBM_SEED}"):
        if literal not in rules["M-lgbm"]:
            raise RunRefused(f"M-lgbm rule no longer contains {literal!r}")
    if f"alpha = {RIDGE_ALPHA}" not in rules["M-ridge"]:
        raise RunRefused("M-ridge alpha differs")
    val = spec["validation"]
    if (val["purgeSessions"], val["embargoSessions"], spec["decisionSchedule"]["rebalanceEverySessions"]) != (
            ev.PURGE_SESSIONS, ev.EMBARGO_SESSIONS, ev.REBALANCE_SESSIONS):
        raise RunRefused("purge/embargo/rebalance differ from the evaluator constants")
    return {"protocolId": spec["id"], "protocolSha256": sha, "evaluatorSha256": ev.evaluator_sha256(),
            "registryEntry": entry["id"], "registryHead": registry.head, "spec": spec}


def raw_sensitivity_entry(protocol_sha: str, today: str) -> dict[str, Any]:
    return {"id": RAW_TRIAL_ID, "registeredAt": today, "market": "HOSE", "family": "hose-vn30-cross-section",
            "status": "planned", "trials": len(MODELS), "trialsExact": True,
            "description": ("VN30-2 execution sensitivity: the 8 declared portfolios re-run with raw (CafeF) limit-lock "
                            "detection instead of the adjusted-series proxy (exchange reference, +/-7%, +/-20% after a "
                            "> 25-session gap; proxy where raw data is unusable). Reported beside the frozen proxy "
                            "result, never used for selection or gates."),
            "source": "vn30_2_run.py", "relatedProtocolSha256": protocol_sha, "evaluatedAt": None, "result": None}


def register_raw_sensitivity(protocol_path: Path = PROTOCOL_PATH,
                             registry_path: Path = trial_registry.REGISTRY_PATH, today: str | None = None) -> dict:
    registry = trial_registry.load(registry_path)
    if any(e["id"] == RAW_TRIAL_ID or e.get("supersedes") == RAW_TRIAL_ID for e in registry.entries):
        raise RunRefused(f"{RAW_TRIAL_ID} is already registered")
    return trial_registry.append(raw_sensitivity_entry(ev.protocol_sha256(protocol_path),
                                                       today or date.today().isoformat()), registry_path)


def raw_trial_planned(registry_path: Path = trial_registry.REGISTRY_PATH) -> bool:
    return any(e["id"] == RAW_TRIAL_ID and e["status"] == "planned"
               for e in trial_registry.load(registry_path).active())


# --------------------------------------------------------------------------------------------- raw limits


def load_raw_limits(raw_dir: Path, market: Market, symbols: Sequence[str], start: str, end: str
                    ) -> tuple[dict[str, dict[str, LimitInfo]], dict[str, Any]]:
    """Raw limit info (VND) per member and session in [start, end]; every file's sha256 is verified on load.

    Sessions without a raw bar, without a reference, without an opening print or with an open outside the bar's
    low..high are left out (the backtester then uses the proxy there)."""
    import vn_raw_prices as raw

    prices = raw.RawPrices(raw_dir, verify=True)
    if prices.manifest is None:
        raise ValueError(f"{raw_dir}: no manifest.json")
    sessions = [d for d in market.calendar if start <= d <= end]
    market_sessions = list(market.calendar)
    limits: dict[str, dict[str, LimitInfo]] = {}
    skipped = {"noRawBar": 0, "noReference": 0, "noOpenOrOutsideRange": 0}
    used = 0
    for s in symbols:
        if s not in prices.manifest["symbols"]:
            skipped["noRawBar"] += sum(1 for d in sessions if d in market.bars.get(s, {}))
            continue
        ser = prices.series(s)
        rows: dict[str, LimitInfo] = {}
        for d in sessions:
            if d not in market.bars.get(s, {}):
                continue
            bar = ser.bar(d)
            if bar is None:
                skipped["noRawBar"] += 1
                continue
            if bar.open <= 0 or not bar.low <= bar.open <= bar.high:
                skipped["noOpenOrOutsideRange"] += 1
                continue
            ref = raw.reference_price(ser, d, "exchange")
            if not ref:
                skipped["noReference"] += 1
                continue
            band = raw.band_fraction(ser, d, market_sessions)
            floor, ceiling = raw.price_limits(ref, band)
            rows[d] = LimitInfo(float(bar.open), float(floor), float(ceiling))
            used += 1
        if rows:
            limits[s] = rows
    validate_limits(limits)
    manifest_bytes = (Path(raw_dir) / "manifest.json").read_bytes()
    return limits, {"rawDir": str(raw_dir), "manifestSha256": hashlib.sha256(manifest_bytes).hexdigest(),
                    "sessionsWithRawLimits": used, "sessionsFallingBackToProxy": skipped}


# --------------------------------------------------------------------------------------------- evaluation


def _clean(x: Any) -> Any:
    """JSON-safe: numpy -> python, NaN/inf -> None."""
    if isinstance(x, Mapping):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return f if math.isfinite(f) else None
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    return x


def window_folds(panel: pd.DataFrame, market: Market, test_from: str, test_to: str,
                 holdout_lock: Mapping[str, Any] | None = None,
                 evidence_root: Path | None = None) -> list[ev.Fold]:
    """Sealed yearly folds of the window; the evaluator derives the last session <= test_to from the calendar."""
    sessions = sorted(set(int(s) for s in panel["session"]))
    dates = [market.calendar[s] for s in sessions]
    return ev.yearly_folds(sessions, dates, test_from=test_from, test_to=test_to, calendar=market.calendar,
                           holdout_lock=holdout_lock, evidence_root=evidence_root)


def score_models(panel: pd.DataFrame, folds: Sequence[ev.Fold]) -> dict[str, pd.Series]:
    return {m: ev.walk_forward_scores(panel, folds, FIT_PREDICT[m]) for m in MODELS}


def scores_by_date(panel: pd.DataFrame, scores: pd.Series) -> dict[str, dict[str, float]]:
    ok = scores.notna()
    out: dict[str, dict[str, float]] = {}
    for day, sym, val in zip(panel.loc[ok, "date"], panel.loc[ok, "symbol"], scores[ok]):
        out.setdefault(str(day), {})[str(sym)] = float(val)
    return out


def _cagr(start: float, end: float, years: float) -> float | None:
    if years <= 0 or start <= 0 or end <= 0:
        return None
    return (end / start) ** (1.0 / years) - 1.0


def _close_on_or_before(series: Mapping[str, float], day: str) -> float | None:
    ds = sorted(series)
    k = bisect_right(ds, day) - 1
    return float(series[ds[k]]) if k >= 0 else None


def portfolio_metrics(res: BacktestResult, ew: BacktestResult | None,
                      benchmarks: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    nav = np.array([r["nav"] for r in res.daily], dtype=float)
    dates = [r["date"] for r in res.daily]
    daily = nav[1:] / nav[:-1] - 1.0
    years = (date.fromisoformat(dates[-1]) - date.fromisoformat(dates[0])).days / 365.25
    sd = float(daily.std(ddof=1)) if len(daily) > 1 else math.nan
    downside = float(np.sqrt(np.mean(np.minimum(daily, 0.0) ** 2))) if len(daily) else math.nan
    peak = np.maximum.accumulate(nav)
    period_nav = [r["navAtOpen"] for r in res.rebalances]
    periods = [b / a - 1.0 for a, b in zip(period_nav, period_nav[1:])]
    reasons: dict[str, int] = {}
    for r in res.rejections:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    out: dict[str, Any] = {
        "from": dates[0], "to": dates[-1], "sessions": len(dates), "years": years,
        "totalReturn": float(nav[-1] / nav[0] - 1.0), "cagr": _cagr(nav[0], nav[-1], years),
        "sharpeAnnualized": float(daily.mean() / sd * math.sqrt(TRADING_DAYS)) if sd and sd > 0 else None,
        "sortinoAnnualized": (float(daily.mean() / downside * math.sqrt(TRADING_DAYS))
                              if downside and downside > 0 else None),
        "maxDrawdown": float(np.min(nav / peak - 1.0)),
        "annualTurnoverTwoSided": res.summary.get("annualTurnoverTwoSided"),
        "costVnd": res.summary["costVnd"], "trades": res.summary["trades"], "periods": len(periods),
        "periodReturns": periods, "rejectionsByReason": reasons,
        "stalePositionSessions": res.summary.get("stalePositionSessions"),
    }
    for name, series in benchmarks.items():
        a, b = _close_on_or_before(series, dates[0]), _close_on_or_before(series, dates[-1])
        bc = _cagr(a, b, years) if a and b else None
        out[f"vs{name}"] = {"benchmarkTotalReturn": (b / a - 1.0) if a and b else None, "benchmarkCagr": bc,
                            "netExcessAnnualized": (out["cagr"] - bc) if bc is not None and out["cagr"] is not None
                            else None}
    if ew is not None:
        ew_nav = np.array([r["nav"] for r in ew.daily], dtype=float)
        if [r["date"] for r in ew.daily] != dates:
            raise ValueError("model and B-ew daily rows differ")
        ew_daily = ew_nav[1:] / ew_nav[:-1] - 1.0
        ew_nav_open = [r["navAtOpen"] for r in ew.rebalances]
        ew_periods = [b / a - 1.0 for a, b in zip(ew_nav_open, ew_nav_open[1:])]
        if len(ew_periods) != len(periods):
            raise ValueError("model and B-ew rebalance counts differ")
        excess_p = np.array(periods) - np.array(ew_periods)
        ew_cagr = _cagr(ew_nav[0], ew_nav[-1], years)
        out["vsEqualWeight"] = {
            "netExcessAnnualized": (out["cagr"] - ew_cagr) if out["cagr"] is not None and ew_cagr is not None
            else None,
            "netExcessCumulative": float(nav[-1] / nav[0] - ew_nav[-1] / ew_nav[0]),
            "periodExcess": ev.mean_t(excess_p),
            "hitRate": float(np.mean(excess_p > 0)) if len(excess_p) else None,
        }
        out["_dailyNetExcess"] = daily - ew_daily
    return out


def run_portfolios(market: Market, panel: pd.DataFrame, scores: Mapping[str, pd.Series], folds: Sequence[ev.Fold],
                   rules: PortfolioRules, ew_rules: PortfolioRules,
                   limits: Mapping[str, Mapping[str, LimitInfo]] | None = None) -> dict[str, Any]:
    tested = sorted(s for f in folds for s in f.test_sessions)
    first, last_full = tested[0], tested[-1]
    end = last_full + LABEL_SESSIONS
    if end >= len(market.calendar):
        raise ValueError("portfolio end beyond the market window")
    results: dict[str, BacktestResult] = {}
    for m in MODELS:
        lookup = scores_by_date(panel, scores[m])
        results[m] = run_backtest(market.calendar, market.bars, market.periods,
                                  lambda day, view, lookup=lookup: lookup.get(day, {}),
                                  ew_rules if m == "B-ew" else rules, first_decision=first,
                                  end=market.calendar[end], limits=limits)
    metrics = {m: portfolio_metrics(results[m], None if m == "B-ew" else results["B-ew"], market.benchmarks)
               for m in MODELS}
    daily_excess = {m: metrics[m].pop("_dailyNetExcess") for m in RANKED}
    nav = {m: [(r["date"], r["nav"]) for r in results[m].daily] for m in MODELS}
    return {"firstDecision": market.calendar[first], "lastDecisionScored": market.calendar[last_full],
            "end": market.calendar[end], "metrics": metrics, "dailyNetExcess": daily_excess, "nav": nav}


def ic_block(panel: pd.DataFrame, scores: Mapping[str, pd.Series], folds: Sequence[ev.Fold]) -> dict[str, Any]:
    """Rank IC of every ranked model, baselines included, through the guarded ``window_rank_ic`` (fold test rows)."""
    tables = {m: ev.window_rank_ic(panel, scores[m], folds, min_names=MIN_IC_NAMES) for m in RANKED}
    tables = {m: t[np.isin(t["session"], panel.loc[scores[m].notna(), "session"].unique())] for m, t in tables.items()}
    out = {m: {"summary": ev.ic_summary(t), "byDate": t.to_dict(orient="records")} for m, t in tables.items()}
    for m in ML_MODELS:
        out[m]["minusComposite"] = ev.paired_ic_difference(tables[m], tables["B-composite"])
    return out


def multiple_testing(daily_excess: Mapping[str, np.ndarray], best: str, registry_path: Path) -> dict[str, Any]:
    try:
        return ev.multiple_testing_report({m: daily_excess[m] for m in RANKED}, best, registry_path=registry_path)
    except ValueError as exc:  # e.g. too few periods for 16 blocks: the gate cannot pass
        return {"best": best, "error": str(exc),
                "gates": {"dsrMin": ev.GATE_DSR_MIN, "dsr": None, "dsrPass": False, "pboMax": ev.GATE_PBO_MAX,
                          "pbo": None, "pboPass": False}}


def evaluate_window(market: Market, panel: pd.DataFrame, test_from: str, test_to: str, *,
                    rules: PortfolioRules, ew_rules: PortfolioRules, registry_path: Path,
                    holdout_lock: Mapping[str, Any] | None = None, raw_limits: Mapping | None = None,
                    best_for_report: str | None = None, evidence_root: Path | None = None) -> dict[str, Any]:
    """Walk-forward scores, IC, portfolios (proxy, and raw limits when given) and multiple testing for one window."""
    folds = window_folds(panel, market, test_from, test_to, holdout_lock, evidence_root)
    if not folds:
        raise ValueError(f"no walk-forward fold in [{test_from}, {test_to}]")
    scores = score_models(panel, folds)
    ic = ic_block(panel, scores, folds)
    port = run_portfolios(market, panel, scores, folds, rules, ew_rules)
    out: dict[str, Any] = {
        "window": {"from": test_from, "to": test_to, "lastSession": market.last_date()},
        "folds": [f.as_dict() for f in folds], "ic": ic,
        "portfolio": {k: port[k] for k in ("firstDecision", "lastDecisionScored", "end", "metrics")},
        "_dailyNetExcess": port["dailyNetExcess"], "_nav": port["nav"], "_scores": scores,
    }
    if best_for_report is None:
        best_for_report = max(RANKED, key=lambda m: _num(ic[m]["summary"]["mean"]))
    out["multipleTesting"] = multiple_testing(port["dailyNetExcess"], best_for_report, registry_path)
    if raw_limits is not None:
        raw_port = run_portfolios(market, panel, scores, folds, rules, ew_rules, limits=raw_limits)
        out["portfolioRawLimits"] = {k: raw_port[k] for k in ("firstDecision", "lastDecisionScored", "end", "metrics")}
    return out


def _num(x: Any) -> float:
    return float(x) if x is not None and math.isfinite(float(x)) else -math.inf


def development_gate(ic: Mapping[str, Any]) -> dict[str, Any]:
    """M-ridge or M-lgbm passes only if its dev mean rank IC > B-composite's and > 0 with t >= 2."""
    comp = ic["B-composite"]["summary"]["mean"]
    rows = {}
    for m in ML_MODELS:
        s = ic[m]["summary"]
        checks = {"meanIcAboveComposite": s["mean"] is not None and comp is not None and s["mean"] > comp,
                  "meanIcPositive": s["mean"] is not None and s["mean"] > 0,
                  "tAtLeast2": s["t"] is not None and s["t"] >= GATE_DEV_T_MIN}
        rows[m] = {"meanIc": s["mean"], "t": s["t"], "dates": s["datesNonOverlapping"], "compositeMeanIc": comp,
                   **checks, "pass": all(checks.values())}
    passing = [m for m in ML_MODELS if rows[m]["pass"]]
    winner = max(passing, key=lambda m: rows[m]["meanIc"]) if passing else None
    return {"rule": "M-ridge or M-lgbm passes only if its dev mean rank IC > B-composite's and > 0 with t >= 2",
            "models": rows, "winner": winner, "verdict": "pass" if winner else "no-go"}


def holdout_gate(winner: str | None, window: Mapping[str, Any]) -> dict[str, Any]:
    """The dev winner must show holdout mean rank IC > 0, net excess vs B-ew > 0, DSR >= 0.95 and PBO < 0.3."""
    rule = ("the dev winner must show holdout mean rank IC > 0, top-10 net excess vs B-ew > 0, DSR >= 0.95 "
            "and PBO < 0.3")
    if winner is None:
        return {"rule": rule, "applied": False, "model": None, "verdict": "no-go",
                "reason": "no model passed the development gate"}
    ic_mean = window["ic"][winner]["summary"]["mean"]
    excess = window["portfolio"]["metrics"][winner]["vsEqualWeight"]["netExcessAnnualized"]
    g = window["multipleTesting"]["gates"]
    checks = {"meanIcPositive": ic_mean is not None and ic_mean > 0,
              "netExcessVsEwPositive": excess is not None and excess > 0,
              "dsrAtLeast095": bool(g["dsrPass"]), "pboBelow03": bool(g["pboPass"])}
    passed = all(checks.values())
    return {"rule": rule, "applied": True, "model": winner, "meanIc": ic_mean, "netExcessVsEwAnnualized": excess,
            "dsr": g["dsr"], "pbo": g["pbo"], **checks, "pass": passed,
            "verdict": "pass: proceed to >= 6 months of forward paper" if passed else "no-go"}


# --------------------------------------------------------------------------------------------- run state / files


def code_state(repo: Path = REPO) -> dict[str, Any]:
    """git HEAD, sha256 of ``git diff HEAD --binary`` + every untracked file (path and bytes), per-module sha256."""

    def git(*args: str) -> bytes:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True).stdout

    modules = ["vn30_2_run.py", "vn30_panel_eval.py", "vn_portfolio_backtest.py", "vn30_research_data.py",
               "vn_raw_prices.py", "vn30_rotation.py", "vn_execution_model.py", "trial_registry.py"]
    state: dict[str, Any] = {"modules": {m: hashlib.sha256((repo / m).read_bytes()).hexdigest()
                                         for m in modules if (repo / m).exists()}}
    try:
        head = git("rev-parse", "HEAD").decode().strip()
        diff = git("diff", "HEAD", "--binary")
        untracked = sorted(p for p in git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0") if p)
        h = hashlib.sha256(diff)
        for p in untracked:
            h.update(b"\0" + p.encode() + b"\0" + (repo / p).read_bytes())
        state.update({"gitHead": head, "diffSha256": h.hexdigest(), "dirty": bool(diff or untracked),
                      "untrackedFiles": untracked})
    except (OSError, subprocess.CalledProcessError) as exc:
        state.update({"gitHead": None, "diffSha256": None, "error": str(exc)})
    return state


def _write_json(path: Path, obj: Any) -> str:
    data = (json.dumps(_clean(obj), ensure_ascii=False, indent=2, sort_keys=False) + "\n").encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _public(window: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in window.items() if not k.startswith("_")}


def nav_csv(windows: Mapping[str, Mapping[str, Any]]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["window", "date", *MODELS])
    for name, win in windows.items():
        navs = win["_nav"]
        for k, (day, _) in enumerate(navs["B-ew"]):
            w.writerow([name, day, *(f"{navs[m][k][1]:.2f}" for m in MODELS)])
    return buf.getvalue().encode("utf-8")


def _fmt(x: Any, pct: bool = False, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    return f"{100 * x:.2f}%" if pct else f"{x:.{digits}f}"


def report_markdown(results: Mapping[str, Any]) -> str:
    lines = [f"# VN30-2 run {results['runId']}", "",
             f"- Protocol `{results['protocol']['id']}` sha256 `{results['protocol']['sha256']}` "
             f"(registry entry `{results['protocol']['registryEntry']}`); evaluator `vn30_panel_eval.py` sha256 "
             f"`{results['protocol'].get('evaluatorSha256')}`",
             f"- Snapshot `{results['data']['snapshotSha256']}`",
             f"- Registry head at evaluation `{results['registry']['headAtEvaluation']}`; HOSE trials N = "
             f"{results['registry']['hoseTrialsRunPlanned']}",
             f"- Code: git `{results['code'].get('gitHead')}`, diff sha256 `{results['code'].get('diffSha256')}`",
             f"- Limit-fill rule (primary): adjusted-series proxy of the protocol; raw-limit sensitivity: "
             f"{'reported' if results.get('rawLimits', {}).get('used') else 'not run'}",
             "", f"## Verdict: {results['verdict']}", ""]
    dg = results["development"]["gate"]
    lines += ["## Development gate (2018-01-22 .. 2023-12-31, fold years with training data)", "",
              "| model | mean IC | t | dates | composite IC | > composite | > 0 | t >= 2 | pass |",
              "|---|---|---|---|---|---|---|---|---|"]
    for m, r in dg["models"].items():
        lines.append(f"| {m} | {_fmt(r['meanIc'])} | {_fmt(r['t'], digits=2)} | {r['dates']} | "
                     f"{_fmt(r['compositeMeanIc'])} | {_fmt(r['meanIcAboveComposite'])} | {_fmt(r['meanIcPositive'])} | "
                     f"{_fmt(r['tAtLeast2'])} | {_fmt(r['pass'])} |")
    lines += ["", f"Development winner: {dg['winner'] or 'none'}", ""]
    hg = results["holdout"]["gate"]
    lines += ["## Holdout gate (2024-01-01 .. 2026-09-18, read once)", ""]
    if hg["applied"]:
        lines += [f"Model {hg['model']}: mean IC {_fmt(hg['meanIc'])} (> 0: {_fmt(hg['meanIcPositive'])}), "
                  f"net excess vs B-ew {_fmt(hg['netExcessVsEwAnnualized'], pct=True)}/yr "
                  f"(> 0: {_fmt(hg['netExcessVsEwPositive'])}), DSR {_fmt(hg['dsr'])} (>= 0.95: "
                  f"{_fmt(hg['dsrAtLeast095'])}), PBO {_fmt(hg['pbo'])} (< 0.3: {_fmt(hg['pboBelow03'])}). "
                  f"Verdict: {hg['verdict']}.", ""]
    else:
        lines += [f"Not applied: {hg['reason']}. Verdict: {hg['verdict']}.", ""]
    for name in ("development", "holdout"):
        win = results[name]["window"]
        lines += [f"## {name.capitalize()}: every model ({win['portfolio']['firstDecision']} .. "
                  f"{win['portfolio']['end']})", "",
                  "| model | mean IC | t | dates | net excess vs B-ew /yr | vs VN30 /yr | vs E1VFVN30 /yr | CAGR | "
                  "Sharpe | Sortino | MDD | turnover /yr | hit rate |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for m in MODELS:
            pm = win["portfolio"]["metrics"][m]
            ic = win["ic"].get(m, {}).get("summary", {})
            vs_ew = pm.get("vsEqualWeight", {})
            note = " (semi-sealed: cannot confirm)" if name == "holdout" and m in SEMI_SEALED else ""
            lines.append(f"| {m}{note} | {_fmt(ic.get('mean'))} | {_fmt(ic.get('t'), digits=2)} | "
                         f"{ic.get('datesNonOverlapping', '')} | {_fmt(vs_ew.get('netExcessAnnualized'), pct=True)} | "
                         f"{_fmt(pm.get('vsVN30', {}).get('netExcessAnnualized'), pct=True)} | "
                         f"{_fmt(pm.get('vsE1VFVN30', {}).get('netExcessAnnualized'), pct=True)} | "
                         f"{_fmt(pm['cagr'], pct=True)} | {_fmt(pm['sharpeAnnualized'], digits=2)} | "
                         f"{_fmt(pm['sortinoAnnualized'], digits=2)} | {_fmt(pm['maxDrawdown'], pct=True)} | "
                         f"{_fmt(pm['annualTurnoverTwoSided'], digits=2)} | {_fmt(vs_ew.get('hitRate'), pct=True)} |")
        mt = win["multipleTesting"]
        g = mt["gates"]
        lines += ["", f"Multiple testing ({'gate' if name == 'holdout' else 'information'}; series = daily net "
                  f"excess vs B-ew of the 7 ranked models; DSR of {mt['best']}): DSR {_fmt(g['dsr'])}, "
                  f"PBO {_fmt(g['pbo'])}" + (f" (error: {mt['error']})" if mt.get("error") else "") + ".", ""]
        raw = win.get("portfolioRawLimits")
        if raw:
            lines += ["Raw-limit sensitivity (registered trial hose-vn30-2-rawlimits; not used for gates):", "",
                      "| model | net excess vs B-ew /yr | CAGR | limit-locked rejections (proxy -> raw) |",
                      "|---|---|---|---|"]
            for m in MODELS:
                pm, pp = raw["metrics"][m], win["portfolio"]["metrics"][m]
                lines.append(f"| {m} | {_fmt(pm.get('vsEqualWeight', {}).get('netExcessAnnualized'), pct=True)} | "
                             f"{_fmt(pm['cagr'], pct=True)} | {pp['rejectionsByReason'].get('limit-locked', 0)} -> "
                             f"{pm['rejectionsByReason'].get('limit-locked', 0)} |")
            lines.append("")
    lines += ["## Implementation choices (protocol silent)", ""] + [f"- {c}" for c in results["implementationChoices"]]
    lines += ["", "## Known limitations", ""] + [f"- {x}" for x in results["knownLimitations"]]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------- phases


@dataclass
class RunConfig:
    snapshot: Path = DEFAULT_SNAPSHOT
    snapshot_sha256: str = DEFAULT_SNAPSHOT_SHA256
    raw_dir: Path | None = DEFAULT_RAW_DIR
    registry_path: Path = trial_registry.REGISTRY_PATH
    protocol_path: Path = PROTOCOL_PATH
    evidence_root: Path | None = None  # None = vn30_panel_eval.default_evidence_root() (main checkout)


def portfolio_rules(protocol_path: Path) -> tuple[PortfolioRules, PortfolioRules]:
    rules = PortfolioRules.from_protocol(Path(protocol_path), cash_settlement_sessions=0)
    return rules, PortfolioRules.from_protocol(Path(protocol_path), cash_settlement_sessions=0, top_k=None,
                                               keep_rank=None)


def development_phase(market_full: Market, spec: Mapping[str, Any], *, rules: PortfolioRules,
                      ew_rules: PortfolioRules, registry_path: Path, raw_dir: Path | None,
                      pseudo_holdout: tuple[str, str] | None = None) -> dict[str, Any]:
    dev = spec["validation"]["developmentWindow"]
    market = market_full.truncate(dev["to"])
    min_hist = int(spec["universe"]["minimumHistorySessions"])
    decisions = decision_indices(market, int(spec["decisionSchedule"]["rebalanceEverySessions"]), min_hist)
    panel = build_panel(market, decisions, min_history=min_hist)
    raw_limits, raw_info = _raw_limits_for(raw_dir, market, market.calendar[decisions[0]], market.last_date())
    window = evaluate_window(market, panel, dev["from"], dev["to"], rules=rules, ew_rules=ew_rules,
                             registry_path=registry_path, raw_limits=raw_limits)
    gate = development_gate(window["ic"])
    out = {"market": {"firstSession": market.calendar[0], "lastSession": market.last_date(),
                      "firstDecision": market.calendar[decisions[0]], "decisions": len(decisions),
                      "panelRows": len(panel), "panelRowsWithTarget": int(panel["target"].notna().sum())},
           "window": window, "gate": gate, "rawLimits": raw_info}
    if pseudo_holdout is not None:  # dry run only: the holdout code path on a development sub-window, no lock
        best = gate["winner"] or max(ML_MODELS, key=lambda m: _num(window["ic"][m]["summary"]["mean"]))
        pw = evaluate_window(market, panel, pseudo_holdout[0], pseudo_holdout[1], rules=rules, ew_rules=ew_rules,
                             registry_path=registry_path, raw_limits=raw_limits, best_for_report=best)
        out["pseudoHoldout"] = {"window": pw, "gate": holdout_gate(gate["winner"], pw)}
    return out


def _raw_limits_for(raw_dir: Path | None, market: Market, start: str, end: str):
    if raw_dir is None:
        return None, {"used": False, "reason": "disabled"}
    try:
        ever_in_force = sorted(set().union(*(members for _, members in market.periods)) & set(market.bars))
        limits, info = load_raw_limits(Path(raw_dir), market, ever_in_force, start, end)
    except (OSError, ValueError, KeyError) as exc:
        return None, {"used": False, "reason": f"raw data unavailable or not verified: {exc}"}
    return limits, {"used": True, **info}


def holdout_phase(market_full: Market, spec: Mapping[str, Any], lock: Mapping[str, Any], winner: str | None,
                  dev_ic: Mapping[str, Any], *, rules: PortfolioRules, ew_rules: PortfolioRules,
                  registry_path: Path, raw_dir: Path | None, evidence_root: Path | None = None) -> dict[str, Any]:
    hold = spec["validation"]["holdout"]
    market = market_full.truncate(hold["to"])
    min_hist = int(spec["universe"]["minimumHistorySessions"])
    decisions = decision_indices(market, int(spec["decisionSchedule"]["rebalanceEverySessions"]), min_hist)
    panel = build_panel(market, decisions, min_history=min_hist)
    first_hold = next(i for i in decisions if market.calendar[i] >= hold["from"])
    raw_limits, raw_info = _raw_limits_for(raw_dir, market, market.calendar[first_hold], market.last_date())
    best = winner or max(ML_MODELS, key=lambda m: _num(dev_ic[m]["summary"]["mean"]))
    window = evaluate_window(market, panel, hold["from"], hold["to"], rules=rules, ew_rules=ew_rules,
                             registry_path=registry_path, holdout_lock=lock, raw_limits=raw_limits,
                             best_for_report=best, evidence_root=evidence_root)
    window["multipleTesting"]["bestRole"] = ("development winner (gate)" if winner else
                                             "ML model with the higher development IC (information only, no winner)")
    return {"market": {"lastSession": market.last_date(), "decisions": len(decisions), "panelRows": len(panel)},
            "window": window, "gate": holdout_gate(winner, window), "rawLimits": raw_info}


def preflight(cfg: RunConfig) -> dict[str, Any]:
    proto = check_protocol(cfg.protocol_path, cfg.registry_path)
    lock = ev.HoldoutLock(PROTOCOL_ID, proto["protocolSha256"], cfg.evidence_root)
    if lock.locked:
        raise RunRefused(f"the holdout of {PROTOCOL_ID} was already opened: {lock.existing()[0].get('path')}")
    if cfg.raw_dir is not None and not raw_trial_planned(cfg.registry_path):
        raise RunRefused(f"raw-limit sensitivity requested but {RAW_TRIAL_ID} is not registered as planned: "
                         "run register-raw-sensitivity first (or pass --no-raw)")
    return {**proto, "lock": lock}


def load_market(cfg: RunConfig) -> tuple[Market, str]:
    import vn30_research_data as rd

    snap = rd.load_snapshot(cfg.snapshot, cfg.snapshot_sha256)
    return market_from_snapshot(snap), snap.sha256


KNOWN_LIMITATIONS = [
    "Adjusted prices only for fills and ticks; the limit-fill proxy is approximate (raw-limit sensitivity reported).",
    "30 names give low power: a mean rank IC of about 0.035 is needed for t = 2 over the development window.",
    "Snapshot issues (manifest knownIssues): 2018-01-23/24 provider hole (not sessions); VHM bars before 2018-05-17 "
    "excluded (reused ticker); pre-HOSE history of tickers that moved to HOSE feeds warm-up features.",
    "Sale proceeds are used as buying power in the same session; advance interest is not charged.",
    "The development window was evaluated during dry runs before the official run (no protocol parameter exists to "
    "tune; only the holdout is read once).",
]


def official_run(cfg: RunConfig, run_id: str | None = None, now: datetime | None = None,
                 market: tuple[Market, str] | None = None) -> dict[str, Any]:
    """The one official run. ``market`` = (Market, label) replaces the snapshot (tests only)."""
    now = now or datetime.now(timezone.utc)
    pre = preflight(cfg)
    market_full, snap_sha = market if market is not None else load_market(cfg)
    return run_with_market(cfg, pre, market_full, snap_sha, run_id or now.strftime("%Y%m%dT%H%M%SZ"), now)


def run_with_market(cfg: RunConfig, pre: Mapping[str, Any], market_full: Market, snap_sha: str, run_id: str,
                    now: datetime) -> dict[str, Any]:
    spec = pre["spec"]
    lock: ev.HoldoutLock = pre["lock"]
    rules, ew_rules = portfolio_rules(cfg.protocol_path)
    code = code_state()
    run_dir = lock.evidence_root / run_id
    if run_dir.exists():
        raise RunRefused(f"run folder already exists: {run_dir}")
    count_before = ev.registry_trial_count(cfg.registry_path)

    dev = development_phase(market_full, spec, rules=rules, ew_rules=ew_rules, registry_path=cfg.registry_path,
                            raw_dir=cfg.raw_dir)
    run_dir.mkdir(parents=True)
    dev_doc = {"schema": RESULTS_SCHEMA + "/development", "runId": run_id, "protocolSha256": pre["protocolSha256"],
               "evaluatorSha256": pre["evaluatorSha256"], "snapshotSha256": snap_sha,
               "registryHead": count_before["registryHead"], "code": code,
               "market": dev["market"], "gate": dev["gate"], "window": _public(dev["window"]),
               "rawLimits": dev["rawLimits"], "writtenAtUtc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    dev_sha = _write_json(run_dir / "dev-results.json", dev_doc)

    record = lock.acquire(run_dir, run_id=run_id, registry_head=count_before["registryHead"], models=MODELS)
    try:
        hold = holdout_phase(market_full, spec, record, dev["gate"]["winner"], dev["window"]["ic"], rules=rules,
                             ew_rules=ew_rules, registry_path=cfg.registry_path, raw_dir=cfg.raw_dir,
                             evidence_root=lock.evidence_root)
    except BaseException:
        (run_dir / "holdout-error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise

    verdict = hold["gate"]["verdict"] if dev["gate"]["winner"] else "no-go (no model passed the development gate)"
    count_now = ev.registry_trial_count(cfg.registry_path)
    results = {
        "schema": RESULTS_SCHEMA, "runId": run_id, "startedAtUtc": now.isoformat(timespec="seconds"),
        "finishedAtUtc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": {"id": PROTOCOL_ID, "sha256": pre["protocolSha256"], "evaluatorSha256": pre["evaluatorSha256"],
                     "registryEntry": pre["registryEntry"]},
        "data": {"snapshot": str(cfg.snapshot), "snapshotSha256": snap_sha},
        "registry": {"headAtEvaluation": count_now["registryHead"], "hoseTrialsRunPlanned": count_now["trials"],
                     "hoseEntriesRunPlanned": count_now["entries"], "lowerBound": count_now["lowerBound"]},
        "code": code, "models": list(MODELS), "features": list(FEATURES),
        "modelParameters": {"M-ridge": {"alpha": RIDGE_ALPHA, "fitIntercept": True},
                            "M-lgbm": {k: v for k, v in LGBM_PARAMS.items()}},
        "rules": {"model": rules.to_dict(), "equalWeight": ew_rules.to_dict()},
        "verdict": verdict,
        "development": {"market": dev["market"], "gate": dev["gate"], "window": _public(dev["window"]),
                        "resultsFileSha256": dev_sha},
        "holdout": {"lock": {k: v for k, v in record.items() if k != "path"}, "market": hold["market"],
                    "gate": hold["gate"], "window": _public(hold["window"]), "semiSealed": list(SEMI_SEALED)},
        "rawLimits": {"used": bool(dev["rawLimits"].get("used") and hold["rawLimits"].get("used")),
                      "trialId": RAW_TRIAL_ID, "development": dev["rawLimits"], "holdout": hold["rawLimits"]},
        "implementationChoices": IMPLEMENTATION_CHOICES, "knownLimitations": KNOWN_LIMITATIONS,
    }
    for win in (results["development"]["window"], results["holdout"]["window"]):
        for m, v in win["ic"].items():
            v["byDate"] = [{k: r[k] for k in ("date", "n", "ic")} for r in v["byDate"]]
    (run_dir / "daily-nav.csv").write_bytes(nav_csv({"development": dev["window"], "holdout": hold["window"]}))
    results_sha = _write_json(run_dir / "results.json", results)
    (run_dir / "report.md").write_text(report_markdown(results), encoding="utf-8", newline="\n")

    summary = registry_result_summary(results)
    entry = trial_registry.append({
        "id": f"{REGISTRY_ENTRY_ID}-run-{run_id}", "registeredAt": now.date().isoformat(), "market": "HOSE",
        "family": "hose-vn30-cross-section", "status": "run", "trials": len(MODELS), "trialsExact": True,
        "description": "VN30-2 frozen protocol, run once (development then holdout); 8 declared models.",
        "source": "docs/research/protocols/vn30-2.json", "specSha256": pre["protocolSha256"],
        "supersedes": REGISTRY_ENTRY_ID, "evaluatedAt": now.date().isoformat(), "runId": run_id,
        "snapshotSha256": snap_sha, "resultsSha256": results_sha,
        "evidence": f"docs/research/evidence/{PROTOCOL_ID}/{run_id}/results.json", "result": summary},
        cfg.registry_path)
    raw_entry = None
    if results["rawLimits"]["used"] and raw_trial_planned(cfg.registry_path):
        raw_entry = trial_registry.append({
            **raw_sensitivity_entry(pre["protocolSha256"], now.date().isoformat()),
            "id": f"{RAW_TRIAL_ID}-run-{run_id}", "status": "run", "supersedes": RAW_TRIAL_ID,
            "evaluatedAt": now.date().isoformat(), "runId": run_id, "resultsSha256": results_sha,
            "evidence": f"docs/research/evidence/{PROTOCOL_ID}/{run_id}/results.json",
            "result": raw_result_summary(results)}, cfg.registry_path)
    return {"runDir": str(run_dir), "resultsSha256": results_sha, "verdict": verdict, "registryEntry": entry["id"],
            "rawRegistryEntry": raw_entry["id"] if raw_entry else None, "results": results}


def registry_result_summary(results: Mapping[str, Any]) -> str:
    dg, hg = results["development"]["gate"], results["holdout"]["gate"]
    parts = [f"verdict: {results['verdict']}."]
    for m in ML_MODELS:
        r = dg["models"][m]
        parts.append(f"dev {m} mean IC {_fmt(r['meanIc'])} t {_fmt(r['t'], digits=2)} "
                     f"({'pass' if r['pass'] else 'fail'});")
    parts.append(f"dev B-composite mean IC {_fmt(dg['models']['M-ridge']['compositeMeanIc'])}.")
    if hg["applied"]:
        parts.append(f"holdout {hg['model']}: IC {_fmt(hg['meanIc'])}, net excess vs B-ew "
                     f"{_fmt(hg['netExcessVsEwAnnualized'], pct=True)}/yr, DSR {_fmt(hg['dsr'])}, PBO {_fmt(hg['pbo'])}.")
    else:
        hw = results["holdout"]["window"]
        parts.append("holdout (report only): " + ", ".join(
            f"{m} IC {_fmt(hw['ic'][m]['summary']['mean'])}" for m in ML_MODELS) + ".")
    return " ".join(parts)


def raw_result_summary(results: Mapping[str, Any]) -> str:
    out = []
    for name in ("development", "holdout"):
        raw = results[name]["window"].get("portfolioRawLimits")
        if not raw:
            continue
        vals = ", ".join(f"{m} {_fmt(raw['metrics'][m]['vsEqualWeight']['netExcessAnnualized'], pct=True)}"
                         for m in RANKED)
        out.append(f"{name} net excess vs B-ew /yr with raw limits: {vals}.")
    return " ".join(out) or "not run"


def dry_run(cfg: RunConfig, out_dir: Path, market: tuple[Market, str] | None = None) -> dict[str, Any]:
    """Development phase + the holdout code path on a development sub-window; no lock, no registry write."""
    out_dir = Path(out_dir).resolve()
    root = (cfg.evidence_root or ev.default_evidence_root()).resolve()
    if out_dir == root or root in out_dir.parents:
        raise RunRefused("a dry run must not write into the protocol evidence root")
    proto = check_protocol(cfg.protocol_path, cfg.registry_path)
    market_full, snap_sha = market if market is not None else load_market(cfg)
    rules, ew_rules = portfolio_rules(cfg.protocol_path)
    dev = development_phase(market_full, proto["spec"], rules=rules, ew_rules=ew_rules,
                            registry_path=cfg.registry_path, raw_dir=cfg.raw_dir,
                            pseudo_holdout=("2022-01-01", "2023-12-31"))
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {"schema": RESULTS_SCHEMA + "/dry-run", "protocolSha256": proto["protocolSha256"],
           "evaluatorSha256": proto["evaluatorSha256"], "snapshotSha256": snap_sha, "market": dev["market"],
           "gate": dev["gate"], "window": _public(dev["window"]), "rawLimits": dev["rawLimits"],
           "pseudoHoldout": {"gate": dev["pseudoHoldout"]["gate"], "window": _public(dev["pseudoHoldout"]["window"])}}
    _write_json(out_dir / "dry-run.json", doc)
    return doc


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Run the frozen VN30-2 protocol")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("register-raw-sensitivity")
    for name in ("dry-run", "run"):
        p = sub.add_parser(name)
        p.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
        p.add_argument("--snapshot-sha256", default=DEFAULT_SNAPSHOT_SHA256)
        p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
        p.add_argument("--no-raw", action="store_true")
        if name == "dry-run":
            p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "register-raw-sensitivity":
            print(json.dumps(register_raw_sensitivity(), ensure_ascii=False))
            return 0
        cfg = RunConfig(snapshot=args.snapshot, snapshot_sha256=args.snapshot_sha256,
                        raw_dir=None if args.no_raw else args.raw_dir)
        if args.cmd == "dry-run":
            doc = dry_run(cfg, args.out)
            print(json.dumps(_clean({"market": doc["market"], "rawLimits": doc["rawLimits"],
                                     "out": str(args.out)}), indent=2))
            return 0
        out = official_run(cfg)
        print(json.dumps(_clean({k: v for k, v in out.items() if k != "results"}), indent=2))
        print(report_markdown(out["results"]))
        return 0
    except RunRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
