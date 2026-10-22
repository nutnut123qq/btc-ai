"""Equity quant base-rates (EQX P1) — protocol ``equity-base-rates-protocol-v1``.

Implements the protocol frozen on 2026-10-16 in
``backend/docs/plans/equity-expansion/spec.md`` §5, before any result was seen.
Nothing here is tuned on outcomes; every constant below is a declared protocol
value.  Interpretations of points the spec leaves open are listed in
``PROTOCOL_INTERPRETATIONS`` and stamped into the artifact.

Pipeline (per symbol, never pooled, only that symbol's own adjusted 1d OHLCV):

1. Bars: finalized ``InstrumentCandles`` 1d rows (adjusted basis), off-session
   anomaly bars excluded (same calendar predicate as ``run_mbb_replay``).
2. Labels: signal at the close of bar t, simulated entry at the open of the next
   stored session t+1; triple barrier from the entry price over sessions
   entry..entry+H-1 (gap-through-stop exits at the open, gap-through-target exits
   at the target, same-bar double touch = stop-first, timeout exits at the last
   close, not enough bars = immature).  Label availability = close of the
   resolving bar.
3. Features at t: ret5, ret20, ret60, vol20, distSma50, distSma200, rsi14,
   atr14Pct, volumeZ20; 200-bar warm-up.
4. kNN (Euclidean, standardized): for each forecast bar t, the candidate set is
   every sample whose label resolved strictly before t; the scaler is fit on
   exactly that set.  k in {20, 50, 100} is chosen per walk-forward fold by an
   inner 80/20 time split of the fold's training data.
5. Walk-forward expanding, calendar-year folds, OOS starting at the first bar
   with >= 500 matured samples; expanding class-frequency prior on the same
   timestamps.
6. Metrics, gate, current forecast, atomic artifact write.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import trading_config
import vietnam_market_calendar as vn_calendar
from vn_execution_model import PRICE_BASIS, VnEquityCostSpec

# ---------------------------------------------------------------------------
# Frozen protocol constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "equity-base-rates-artifact-v1"
REPORT_SCHEMA_VERSION = "equity-base-rates-report-v1"
PROTOCOL_VERSION = "equity-base-rates-protocol-v1"
PROTOCOL_FROZEN_AT = "2026-10-16"
METHOD_NAME = "knn-euclidean-standardized"
METHOD_VERSION = "equity-knn-v1"
TIMEFRAME = "1d"

FEATURES: tuple[str, ...] = (
    "ret5",
    "ret20",
    "ret60",
    "vol20",
    "distSma50",
    "distSma200",
    "rsi14",
    "atr14Pct",
    "volumeZ20",
)
WARMUP_BARS = 200
K_GRID: tuple[int, ...] = (20, 50, 100)
INNER_TRAIN_FRACTION = 0.80
MIN_MATURED_FOR_OOS = 500
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20261016
CI_LEVEL = 0.90
RELIABILITY_BINS = 5
GATE_MIN_N_EFFECTIVE = 30
GATE_MAX_ECE = 0.05

CLASSES: tuple[str, ...] = ("target-first", "stop-first", "timeout")
TARGET, STOP, TIMEOUT = 0, 1, 2


@dataclass(frozen=True)
class BarrierConfig:
    config_id: str
    target_pct: float  # fraction, e.g. 0.05
    stop_pct: float  # fraction, e.g. 0.04
    horizon_sessions: int


CONFIGS: tuple[BarrierConfig, ...] = (
    BarrierConfig("tb-p5-s4-h20", 0.05, 0.04, 20),
    BarrierConfig("tb-p10-s6-h60", 0.10, 0.06, 60),
)

PROTOCOL_INTERPRETATIONS: tuple[str, ...] = (
    "entry = open of the next STORED session bar after the signal bar (index t+1), "
    "consistent with the journal resolver rule 'first 1d bar open >= AvailableTimeMs'.",
    "a label is matured once a barrier is touched inside the available bars, even if fewer "
    "than H bars follow; immature = window runs past the last stored bar without a touch.",
    "neighbor/prior/scaler eligibility at forecast bar t = samples whose resolving bar index "
    "is strictly < t (a label resolving at the close of t itself is NOT eligible).",
    "per-point scaler (fit on each forecast point's eligible set) supersedes the per-fold "
    "'refit scaler' wording; k is refit per fold.",
    "inner 80/20 split: fold training = samples resolved before the fold's first forecast bar, "
    "ordered by signal time; validation points use only inner-train samples resolved before "
    "them; k minimises validation binary Brier, ties broken toward the larger k.",
    "folds = calendar years (Asia/Ho_Chi_Minh date of the signal bar); the first fold starts at "
    "the first bar with >= 500 eligible matured samples.",
    "EV per forecast = sum_c p_c * r_c where r_c is the mean realized net return of class c over "
    "the same eligible set (fallback: nominal barrier return, timeout = round-trip cost only).",
    "evNetOosMean / evNetOosWhenPositive = mean REALIZED net return of OOS samples (all / those "
    "with predicted kNN EV > 0).",
    "n_effective = greedy non-overlapping OOS samples (signal indices >= H apart); the CI is a "
    "moving-block bootstrap of the per-sample Brier difference series (block = H sessions).",
    "the current forecast re-selects k with the inner split on all samples resolved before the "
    "latest bar.",
    "features: vol20 = sample std (ddof=1) of the last 20 log returns; volumeZ20 over the 20-bar "
    "window including t (ddof=1); RSI14/ATR14 use Wilder smoothing; zero-variance scaler columns "
    "use std 1.",
)


class BaseRatesError(ValueError):
    """Protocol/scope violation (fail closed)."""


class NonDecisionSymbolError(BaseRatesError):
    """Symbol is not a decision-capable equity (only MBB, BID)."""


class InsufficientDataError(BaseRatesError):
    """No usable data to build an artifact (reason in ``.reason``)."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


DEFAULT_EVIDENCE_ROOT = Path(__file__).resolve().parent / "docs" / "research" / "evidence"
_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_DIR = _WORKSPACE_ROOT / "evidence" / "equity-expansion" / "base-rates"


def now_ms() -> int:
    return int(time.time() * 1000)


def resolve_decision_symbol(symbol: str) -> str:
    """Only decision-capable equities (MBB, BID) get base rates; fail closed otherwise."""
    try:
        canonical = trading_config.require_known_symbol(symbol)
    except ValueError as exc:
        raise NonDecisionSymbolError(str(exc)) from exc
    descriptor = trading_config.INSTRUMENTS[canonical]
    if descriptor["asset_class"] != "equity" or not descriptor["capabilities"].get("decision"):
        raise NonDecisionSymbolError(
            f"Base rates require a decision-capable equity; '{canonical}' is not."
        )
    if TIMEFRAME not in descriptor["timeframes"]:
        raise NonDecisionSymbolError(f"'{canonical}' does not declare the {TIMEFRAME} timeframe.")
    return canonical


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_candles_db(symbol: str, cutoff_ms: int) -> list[dict[str, Any]]:
    """Finalized 1d InstrumentCandles rows of exactly ``symbol`` (ascending).

    Same read-only repeatable-read pattern as ``run_mbb_replay.load_candles_db``;
    connection settings come from ``db_config`` (DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASS).
    """
    from db_config import get_db_connection

    connection = get_db_connection()
    try:
        connection.set_session(readonly=True, autocommit=False, isolation_level="REPEATABLE READ")
        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT "OpenTimeMs","CloseTimeMs","Open","High","Low","Close","Volume",'
                '"PriceBasis","Provider","UpdatedAtUtc","Revision" '
                'FROM "InstrumentCandles" WHERE "Symbol"=%s AND "Timeframe"=%s '
                'AND "CloseTimeMs"<=%s ORDER BY "OpenTimeMs" ASC',
                (symbol, TIMEFRAME, cutoff_ms),
            )
            names = (
                "OpenTimeMs", "CloseTimeMs", "Open", "High", "Low", "Close", "Volume",
                "PriceBasis", "Provider", "UpdatedAtUtc", "Revision",
            )
            rows = [dict(zip(names, row)) for row in cursor.fetchall()]
        connection.rollback()
        return rows
    finally:
        connection.close()


def _iso_utc(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    text = str(value).strip()
    if not text:
        return None
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return _iso_utc(dt)


@dataclass
class Bars:
    open_ms: np.ndarray
    close_ms: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __len__(self) -> int:
        return int(self.open_ms.shape[0])


def prepare_bars(
    rows: Iterable[Mapping[str, Any]],
    *,
    cutoff_ms: int,
    non_trading_dates: frozenset[date] = frozenset(),
) -> tuple[Bars, dict[str, Any], str | None]:
    """Finalize/sort rows, fingerprint the raw finalized set, drop off-session bars.

    Returns ``(usable_bars, dataset_fingerprint, provider)``.  The fingerprint is
    computed on every finalized stored row (before anomaly filtering) so the
    backend can recompute it from the same rows.
    """
    raw = []
    for row in rows:
        close_ms = int(row["CloseTimeMs"])
        if close_ms > cutoff_ms:
            continue
        raw.append(row)
    raw.sort(key=lambda r: int(r["OpenTimeMs"]))
    opens = [int(r["OpenTimeMs"]) for r in raw]
    if len(set(opens)) != len(opens):
        raise BaseRatesError("Duplicate OpenTimeMs rows in the candle set.")
    bases = {r.get("PriceBasis") for r in raw if r.get("PriceBasis") is not None}
    if bases - {PRICE_BASIS}:
        raise BaseRatesError(
            f"Candles declare price basis {sorted(bases)}; base rates are adjusted-basis only."
        )
    providers = {r.get("Provider") for r in raw if r.get("Provider")}
    if len(providers) > 1:
        raise BaseRatesError(f"Mixed providers in one series: {sorted(providers)}.")
    provider = next(iter(providers)) if providers else None

    lines = []
    for r in raw:
        lines.append(
            ",".join(
                [str(int(r["OpenTimeMs"])), str(int(r["CloseTimeMs"]))]
                + [repr(float(r[k])) for k in ("Open", "High", "Low", "Close", "Volume")]
            )
        )
    updated = [_iso_utc(r.get("UpdatedAtUtc")) for r in raw]
    updated = [u for u in updated if u is not None]
    revisions = [int(r["Revision"]) for r in raw if r.get("Revision") is not None]
    fingerprint = {
        "barCount": len(raw),
        "firstOpenTimeMs": opens[0] if opens else None,
        "lastOpenTimeMs": opens[-1] if opens else None,
        "maxUpdatedAtUtc": max(updated) if updated else None,
        "maxRevision": max(revisions) if revisions else None,
        "sha256": hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest(),
    }

    usable = [r for r in raw if vn_calendar.is_expected_open(TIMEFRAME, int(r["OpenTimeMs"]), non_trading_dates)]
    fingerprint["usableBars"] = len(usable)
    fingerprint["excludedOffSessionBars"] = len(raw) - len(usable)

    def col(name: str, dtype=float) -> np.ndarray:
        return np.array([dtype(r[name]) for r in usable], dtype=np.float64 if dtype is float else np.int64)

    bars = Bars(
        open_ms=col("OpenTimeMs", int),
        close_ms=col("CloseTimeMs", int),
        open=col("Open"),
        high=col("High"),
        low=col("Low"),
        close=col("Close"),
        volume=col("Volume"),
    )
    if len(bars) and (
        np.any(bars.open <= 0) or np.any(bars.close <= 0) or np.any(bars.low <= 0)
        or np.any(bars.high < np.maximum(bars.open, bars.close))
        or np.any(bars.low > np.minimum(bars.open, bars.close))
    ):
        raise BaseRatesError("Invalid OHLC rows (non-positive price or high/low inconsistent).")
    return bars, fingerprint, provider


# ---------------------------------------------------------------------------
# Costs + labels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Costs:
    buy_fee: float
    sell_fee: float
    sell_tax: float

    @classmethod
    def from_execution_model(cls, spec: VnEquityCostSpec | None = None) -> "Costs":
        spec = spec or VnEquityCostSpec()
        return cls(spec.broker_fee_fraction, spec.broker_fee_fraction, spec.sell_tax_fraction)

    def net_return(self, entry_price: float, exit_price: float) -> float:
        """Net long return: exit proceeds after sell fee+tax over entry cost incl. buy fee."""
        return exit_price * (1.0 - self.sell_fee - self.sell_tax) / (entry_price * (1.0 + self.buy_fee)) - 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": "vn_execution_model.VnEquityCostSpec",
            "buyFeeFraction": self.buy_fee,
            "sellFeeFraction": self.sell_fee,
            "sellTaxFraction": self.sell_tax,
            "netReturnFormula": "exit*(1-sellFee-sellTax)/(entry*(1+buyFee)) - 1",
        }


@dataclass
class Labels:
    """Per signal index t: label class, resolve index, flags, realized net return."""

    label: np.ndarray  # int, -1 = immature / no entry
    resolve_idx: np.ndarray  # int, -1 when immature
    double_touch: np.ndarray  # bool
    gap_exit: np.ndarray  # bool
    net_return: np.ndarray  # float, nan when immature
    entry_idx: np.ndarray  # int, -1 when no next bar


def label_barriers(bars: Bars, cfg: BarrierConfig, costs: Costs) -> Labels:
    n = len(bars)
    label = np.full(n, -1, dtype=np.int64)
    resolve = np.full(n, -1, dtype=np.int64)
    double = np.zeros(n, dtype=bool)
    gap = np.zeros(n, dtype=bool)
    net = np.full(n, np.nan)
    entry_idx = np.full(n, -1, dtype=np.int64)
    for t in range(n - 1):
        e = t + 1
        entry_idx[t] = e
        entry = bars.open[e]
        target = entry * (1.0 + cfg.target_pct)
        stop = entry * (1.0 - cfg.stop_pct)
        last = e + cfg.horizon_sessions - 1
        for j in range(e, min(last, n - 1) + 1):
            o, h, lo = bars.open[j], bars.high[j], bars.low[j]
            exit_price = None
            if o <= stop:
                label[t], exit_price, gap[t] = STOP, o, True
            elif o >= target:
                label[t], exit_price, gap[t] = TARGET, target, True
            elif h >= target and lo <= stop:
                label[t], exit_price, double[t] = STOP, stop, True
            elif lo <= stop:
                label[t], exit_price = STOP, stop
            elif h >= target:
                label[t], exit_price = TARGET, target
            elif j == last:
                label[t], exit_price = TIMEOUT, bars.close[j]
            if exit_price is not None:
                resolve[t] = j
                net[t] = costs.net_return(entry, exit_price)
                break
    return Labels(label, resolve, double, gap, net, entry_idx)


# ---------------------------------------------------------------------------
# Features (causal, symbol's own OHLCV only)
# ---------------------------------------------------------------------------


def _wilder(values: np.ndarray, period: int, start: int) -> np.ndarray:
    """Wilder smoothing of ``values`` seeded by the mean of values[start:start+period]."""
    out = np.full(values.shape[0], np.nan)
    seed_end = start + period
    if values.shape[0] < seed_end:
        return out
    avg = float(np.mean(values[start:seed_end]))
    out[seed_end - 1] = avg
    for i in range(seed_end, values.shape[0]):
        avg = (avg * (period - 1) + values[i]) / period
        out[i] = avg
    return out


def compute_features(bars: Bars) -> np.ndarray:
    """(n, 9) feature matrix; rows < WARMUP_BARS are NaN (warm-up)."""
    n = len(bars)
    c, h, lo, v = bars.close, bars.high, bars.low, bars.volume
    X = np.full((n, len(FEATURES)), np.nan)
    if n == 0:
        return X
    diff = np.zeros(n)
    diff[1:] = np.diff(c)
    gains = np.where(diff > 0, diff, 0.0)
    losses = np.where(diff < 0, -diff, 0.0)
    avg_gain = _wilder(gains, 14, 1)
    avg_loss = _wilder(losses, 14, 1)
    tr = np.zeros(n)
    if n > 1:
        prev = c[:-1]
        tr[1:] = np.maximum.reduce([h[1:] - lo[1:], np.abs(h[1:] - prev), np.abs(lo[1:] - prev)])
    atr = _wilder(tr, 14, 1)
    logret = np.full(n, np.nan)
    logret[1:] = np.log(c[1:] / c[:-1])
    for t in range(WARMUP_BARS, n):
        ag, al = avg_gain[t], avg_loss[t]
        rsi = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
        vwin = v[t - 19 : t + 1]
        vstd = float(np.std(vwin, ddof=1))
        X[t] = (
            c[t] / c[t - 5] - 1.0,
            c[t] / c[t - 20] - 1.0,
            c[t] / c[t - 60] - 1.0,
            float(np.std(logret[t - 19 : t + 1], ddof=1)),
            c[t] / float(np.mean(c[t - 49 : t + 1])) - 1.0,
            c[t] / float(np.mean(c[t - 199 : t + 1])) - 1.0,
            rsi,
            atr[t] / c[t],
            0.0 if vstd == 0 else (v[t] - float(np.mean(vwin))) / vstd,
        )
    return X


# ---------------------------------------------------------------------------
# kNN + prior
# ---------------------------------------------------------------------------


def eligible_mask(labels: Labels, has_features: np.ndarray, t: int) -> np.ndarray:
    """Samples usable at forecast bar t: features present and label resolved strictly before t."""
    return has_features & (labels.resolve_idx >= 0) & (labels.resolve_idx < t)


def fit_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return mean, std


def _class_returns(labels: Labels, idx: np.ndarray, cfg: BarrierConfig, costs: Costs) -> np.ndarray:
    nominal = np.array(
        [
            costs.net_return(1.0, 1.0 + cfg.target_pct),
            costs.net_return(1.0, 1.0 - cfg.stop_pct),
            costs.net_return(1.0, 1.0),
        ]
    )
    out = nominal.copy()
    lab = labels.label[idx]
    ret = labels.net_return[idx]
    for c in range(3):
        sel = lab == c
        if np.any(sel):
            out[c] = float(np.mean(ret[sel]))
    return out


def class_frequencies(lab: np.ndarray) -> np.ndarray:
    counts = np.bincount(lab, minlength=3).astype(float)
    return counts / counts.sum()


def _ordered_neighbors(
    X: np.ndarray,
    labels: Labels,
    has_features: np.ndarray,
    t: int,
    k_max: int,
    candidate_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """(neighbors ordered by distance [<= k_max], eligible idx, scaler mean, scaler std)."""
    if not has_features[t]:
        return None
    mask = eligible_mask(labels, has_features, t)
    if candidate_mask is not None:
        mask &= candidate_mask
    idx = np.flatnonzero(mask)
    if idx.shape[0] == 0:
        return None
    mean, std = fit_scaler(X[idx])
    Z = (X[idx] - mean) / std
    z = (X[t] - mean) / std
    dist = np.sqrt(np.sum((Z - z) ** 2, axis=1))
    order = np.argsort(dist, kind="stable")[:k_max]
    return idx[order], idx, mean, std


def knn_forecast(
    X: np.ndarray,
    labels: Labels,
    has_features: np.ndarray,
    t: int,
    k: int,
    *,
    candidate_mask: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """kNN class probabilities for forecast bar t, or None when it must abstain.

    Candidates = eligible (resolved strictly before t) ∩ ``candidate_mask``; the
    scaler is fit on exactly those candidate rows.  Ties in distance resolve to
    the earlier sample (stable sort) so results are deterministic.  Abstains
    when fewer than k candidates exist.
    """
    found = _ordered_neighbors(X, labels, has_features, t, k, candidate_mask)
    if found is None or found[1].shape[0] < k:
        return None
    neighbors, idx, mean, std = found
    return {
        "probs": class_frequencies(labels.label[neighbors]),
        "neighbors": neighbors,
        "eligible": idx,
        "scalerMean": mean,
        "scalerStd": std,
    }


def prior_forecast(labels: Labels, has_features: np.ndarray, t: int) -> np.ndarray | None:
    idx = np.flatnonzero(eligible_mask(labels, has_features, t))
    if idx.shape[0] == 0:
        return None
    return class_frequencies(labels.label[idx])


def select_k(
    X: np.ndarray,
    labels: Labels,
    has_features: np.ndarray,
    fold_start: int,
) -> dict[str, Any]:
    """Inner 80/20 time split inside the fold's training data (resolved before fold_start)."""
    train = np.flatnonzero(eligible_mask(labels, has_features, fold_start))
    n_inner = int(math.floor(train.shape[0] * INNER_TRAIN_FRACTION))
    inner_train, inner_val = train[:n_inner], train[n_inner:]
    cand = np.zeros(X.shape[0], dtype=bool)
    cand[inner_train] = True
    errs: dict[int, list[float]] = {k: [] for k in K_GRID}
    for v in inner_val:
        found = _ordered_neighbors(X, labels, has_features, int(v), max(K_GRID), cand)
        if found is None:
            continue
        neighbors, eligible = found[0], found[1]
        y = 1.0 if labels.label[v] == TARGET else 0.0
        for k in K_GRID:
            if eligible.shape[0] >= k:  # same abstention rule as knn_forecast
                p_target = float(np.mean(labels.label[neighbors[:k]] == TARGET))
                errs[k].append((p_target - y) ** 2)
    scores: dict[int, float | None] = {k: (float(np.mean(e)) if e else None) for k, e in errs.items()}
    scored = [(s, -k, k) for k, s in scores.items() if s is not None]
    chosen = min(scored)[2] if scored else max(K_GRID)
    return {
        "k": chosen,
        "innerTrainN": int(inner_train.shape[0]),
        "innerValN": int(inner_val.shape[0]),
        "innerBrierByK": {str(k): s for k, s in scores.items()},
        "kSelectedByDefault": not scored,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def reliability(probs: np.ndarray, outcomes: np.ndarray) -> tuple[list[dict[str, Any]], float | None]:
    bins = []
    n = probs.shape[0]
    ece = 0.0
    edges = np.linspace(0.0, 1.0, RELIABILITY_BINS + 1)
    for b in range(RELIABILITY_BINS):
        lo, hi = edges[b], edges[b + 1]
        sel = (probs >= lo) & ((probs < hi) if b < RELIABILITY_BINS - 1 else (probs <= hi))
        cnt = int(np.sum(sel))
        entry = {"lower": float(lo), "upper": float(hi), "n": cnt, "meanPredicted": None, "observedRate": None}
        if cnt:
            mp, obs = float(np.mean(probs[sel])), float(np.mean(outcomes[sel]))
            entry["meanPredicted"], entry["observedRate"] = mp, obs
            ece += cnt / n * abs(mp - obs)
        bins.append(entry)
    return bins, (float(ece) if n else None)


def non_overlapping_count(signal_idx: Sequence[int], horizon: int) -> int:
    """Greedy count of samples whose signal indices are >= horizon sessions apart."""
    count, last = 0, None
    for i in sorted(int(x) for x in signal_idx):
        if last is None or i - last >= horizon:
            count += 1
            last = i
    return count


def block_bootstrap_ci(
    values: np.ndarray,
    block: int,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    level: float = CI_LEVEL,
) -> list[float] | None:
    """Moving-block bootstrap CI of the mean (blocks of ``block`` consecutive samples)."""
    n = values.shape[0]
    if n == 0:
        return None
    block = max(1, min(int(block), n))
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(resamples, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(resamples, -1)[:, :n]
    means = values[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    return [float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))]


def _round(x: Any, nd: int = 6) -> Any:
    if x is None:
        return None
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return round(float(x), nd)


# ---------------------------------------------------------------------------
# Walk-forward evaluation per config
# ---------------------------------------------------------------------------


def _local_year(open_ms: int) -> int:
    return vn_calendar.to_local(int(open_ms)).year


def evaluate_config(
    bars: Bars,
    X: np.ndarray,
    cfg: BarrierConfig,
    costs: Costs,
) -> dict[str, Any]:
    n = len(bars)
    labels = label_barriers(bars, cfg, costs)
    has_features = np.all(np.isfinite(X), axis=1) if n else np.zeros(0, dtype=bool)
    matured = has_features & (labels.resolve_idx >= 0)

    # OOS start: first bar with >= MIN_MATURED_FOR_OOS eligible matured samples.
    oos_start = None
    if n:
        resolved_sorted = np.sort(labels.resolve_idx[matured])
        if resolved_sorted.shape[0] >= MIN_MATURED_FOR_OOS:
            # eligible count at t = #(resolve_idx < t) -> need resolve of the 500th < t
            oos_start = int(resolved_sorted[MIN_MATURED_FOR_OOS - 1]) + 1

    folds: list[dict[str, Any]] = []
    rec: dict[str, list] = {k: [] for k in ("t", "pk", "pp", "y", "ret", "ev")}
    n_candidates = 0
    n_immature_oos = 0
    if oos_start is not None and oos_start < n:
        years = sorted({_local_year(bars.open_ms[t]) for t in range(oos_start, n)})
        for year in years:
            fold_ts = [t for t in range(oos_start, n) if _local_year(bars.open_ms[t]) == year and has_features[t]]
            if not fold_ts:
                continue
            sel = select_k(X, labels, has_features, fold_ts[0])
            k = sel["k"]
            f_rec: dict[str, list] = {"pk": [], "pp": [], "y": []}
            f_abst = 0
            f_imm = 0
            for t in fold_ts:
                if labels.resolve_idx[t] < 0:
                    f_imm += 1
                    continue  # immature: excluded from evaluation (pending)
                fc = knn_forecast(X, labels, has_features, t, k)
                pr = prior_forecast(labels, has_features, t)
                if fc is None or pr is None:
                    f_abst += 1
                    continue
                elig = fc["eligible"]
                ev = float(np.dot(fc["probs"], _class_returns(labels, elig, cfg, costs)))
                y = 1.0 if labels.label[t] == TARGET else 0.0
                for key, val in (("t", t), ("pk", fc["probs"][TARGET]), ("pp", pr[TARGET]), ("y", y),
                                 ("ret", labels.net_return[t]), ("ev", ev)):
                    rec[key].append(val)
                f_rec["pk"].append(fc["probs"][TARGET])
                f_rec["pp"].append(pr[TARGET])
                f_rec["y"].append(y)
            n_candidates += len(fold_ts) - f_imm
            n_immature_oos += f_imm
            pk, pp, yy = (np.array(f_rec[k_]) for k_ in ("pk", "pp", "y"))
            folds.append(
                {
                    "year": year,
                    "firstSignalOpenTimeMs": int(bars.open_ms[fold_ts[0]]),
                    "lastSignalOpenTimeMs": int(bars.open_ms[fold_ts[-1]]),
                    "kSelection": sel,
                    "nForecastBars": len(fold_ts),
                    "nImmatureExcluded": f_imm,
                    "nAbstained": f_abst,
                    "nOos": int(pk.shape[0]),
                    "brierKnn": _round(np.mean((pk - yy) ** 2)) if pk.shape[0] else None,
                    "brierPrior": _round(np.mean((pp - yy) ** 2)) if pk.shape[0] else None,
                    "targetFirstRate": _round(np.mean(yy)) if pk.shape[0] else None,
                }
            )

    t_arr = np.array(rec["t"], dtype=np.int64)
    pk, pp, y = (np.array(rec[k_], dtype=float) for k_ in ("pk", "pp", "y"))
    ret, ev = np.array(rec["ret"], dtype=float), np.array(rec["ev"], dtype=float)
    n_oos = int(t_arr.shape[0])
    validation: dict[str, Any] = {
        "nOos": n_oos,
        "nEffective": non_overlapping_count(t_arr.tolist(), cfg.horizon_sessions),
        "nImmatureExcluded": n_immature_oos,
        "oosStartSignalOpenTimeMs": int(bars.open_ms[oos_start]) if oos_start is not None and oos_start < n else None,
        "brierKnn": None,
        "brierPrior": None,
        "brierDiffMean": None,
        "brierDiffCi90": None,
        "ece": None,
        "calibrationBins": [],
        "coverage": (n_oos / n_candidates) if n_candidates else None,
        "abstentionRate": (1.0 - n_oos / n_candidates) if n_candidates else None,
        "evNetOosMean": None,
        "evNetOosWhenPositive": None,
        "evNetOosWhenPositiveN": int(np.sum(ev > 0)) if n_oos else 0,
        "evPredictedOosMean": None,
        "targetFirstRateOos": None,
        "bootstrap": {"method": "moving-block", "block": cfg.horizon_sessions,
                      "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "level": CI_LEVEL},
    }
    if n_oos:
        bk, bp = (pk - y) ** 2, (pp - y) ** 2
        bins, ece = reliability(pk, y)
        ci = block_bootstrap_ci(bk - bp, cfg.horizon_sessions)
        validation.update(
            {
                "brierKnn": _round(np.mean(bk)),
                "brierPrior": _round(np.mean(bp)),
                "brierDiffMean": _round(np.mean(bk - bp)),
                "brierDiffCi90": [_round(ci[0]), _round(ci[1])] if ci else None,
                "ece": _round(ece),
                "calibrationBins": [
                    {**b, "meanPredicted": _round(b["meanPredicted"]), "observedRate": _round(b["observedRate"])}
                    for b in bins
                ],
                "evNetOosMean": _round(np.mean(ret)),
                "evNetOosWhenPositive": _round(np.mean(ret[ev > 0])) if np.any(ev > 0) else None,
                "evPredictedOosMean": _round(np.mean(ev)),
                "targetFirstRateOos": _round(np.mean(y)),
            }
        )
        validation["coverage"] = _round(validation["coverage"])
        validation["abstentionRate"] = _round(validation["abstentionRate"])
    gate_passed, reasons = evaluate_gate(validation)
    validation["gatePassed"] = gate_passed
    validation["gateReasons"] = reasons

    current = current_forecast(bars, X, labels, has_features, cfg, costs)
    label_counts = {
        CLASSES[c]: int(np.sum(labels.label[matured] == c)) for c in range(3)
    }
    return {
        "configId": cfg.config_id,
        "targetPct": cfg.target_pct,
        "stopPct": cfg.stop_pct,
        "horizonSessions": cfg.horizon_sessions,
        "validation": validation,
        "current": current,
        "_folds": folds,
        "_oos": {"t": t_arr, "probTargetKnn": pk, "probTargetPrior": pp, "outcome": y, "evPredicted": ev},
        "_labels": {
            "matured": int(np.sum(matured)),
            "immature": int(np.sum(has_features & (labels.resolve_idx < 0))),
            "classCounts": label_counts,
            "doubleTouch": int(np.sum(labels.double_touch & matured)),
            "gapExits": int(np.sum(labels.gap_exit & matured)),
        },
    }


def evaluate_gate(validation: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    if not validation.get("nOos"):
        reasons.append("no-oos-samples")
    if (validation.get("nEffective") or 0) < GATE_MIN_N_EFFECTIVE:
        reasons.append(f"n-effective-below-{GATE_MIN_N_EFFECTIVE}")
    ci = validation.get("brierDiffCi90")
    if not ci or ci[1] is None or not ci[1] < 0:
        reasons.append("brier-diff-ci90-upper-not-below-zero")
    ece = validation.get("ece")
    if ece is None or ece > GATE_MAX_ECE:
        reasons.append(f"ece-above-{GATE_MAX_ECE}")
    return (not reasons), reasons


def current_forecast(
    bars: Bars,
    X: np.ndarray,
    labels: Labels,
    has_features: np.ndarray,
    cfg: BarrierConfig,
    costs: Costs,
) -> dict[str, Any]:
    n = len(bars)
    out: dict[str, Any] = {
        "signalBarOpenTimeMs": int(bars.open_ms[-1]) if n else None,
        "probTargetFirst": None,
        "probStopFirst": None,
        "probTimeout": None,
        "evNetFraction": None,
        "k": None,
        "nEligible": 0,
        "prior": {"probTargetFirst": None, "probStopFirst": None, "probTimeout": None},
        "abstained": True,
        "abstainReason": None,
    }
    if n == 0:
        out["abstainReason"] = "no-bars"
        return out
    t = n - 1
    eligible = np.flatnonzero(eligible_mask(labels, has_features, t))
    out["nEligible"] = int(eligible.shape[0])
    pr = prior_forecast(labels, has_features, t)
    if pr is not None:
        out["prior"] = {"probTargetFirst": _round(pr[TARGET]), "probStopFirst": _round(pr[STOP]),
                        "probTimeout": _round(pr[TIMEOUT])}
    if not has_features[t]:
        out["abstainReason"] = "insufficient-warmup"
        return out
    if eligible.shape[0] < MIN_MATURED_FOR_OOS:
        out["abstainReason"] = f"matured-samples-below-{MIN_MATURED_FOR_OOS}"
        return out
    sel = select_k(X, labels, has_features, t)
    fc = knn_forecast(X, labels, has_features, t, sel["k"])
    if fc is None:
        out["abstainReason"] = "insufficient-eligible-neighbors"
        return out
    ev = float(np.dot(fc["probs"], _class_returns(labels, fc["eligible"], cfg, costs)))
    out.update(
        {
            "probTargetFirst": _round(fc["probs"][TARGET]),
            "probStopFirst": _round(fc["probs"][STOP]),
            "probTimeout": _round(fc["probs"][TIMEOUT]),
            "evNetFraction": _round(ev),
            "k": sel["k"],
            "abstained": False,
            "kSelection": sel,
        }
    )
    return out


# ---------------------------------------------------------------------------
# Build + artifact
# ---------------------------------------------------------------------------


def build_artifact(
    rows: Iterable[Mapping[str, Any]],
    symbol: str,
    *,
    cutoff_ms: int | None = None,
    non_trading_dates: frozenset[date] | None = None,
    costs: Costs | None = None,
    generated_at: datetime | None = None,
    calendar: vn_calendar.MarketCalendar | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pure build: ``(artifact, report)``.  Never touches the DB or the filesystem.

    Calendar precedence: ``calendar`` > ``non_trading_dates`` > MBB_NON_TRADING_DATES
    override; none of them ⇒ ``InsufficientDataError('calendar-unavailable')``.
    """
    canonical = resolve_decision_symbol(symbol)
    descriptor = trading_config.INSTRUMENTS[canonical]
    cutoff = int(cutoff_ms if cutoff_ms is not None else time.time() * 1000)
    try:
        market_calendar = vn_calendar.require_calendar(non_trading_dates, calendar)
    except vn_calendar.CalendarUnavailableError as exc:
        raise InsufficientDataError(exc.reason, str(exc)) from exc
    ntd = market_calendar.non_trading_dates
    costs = costs or Costs.from_execution_model()
    bars, fingerprint, provider = prepare_bars(rows, cutoff_ms=cutoff, non_trading_dates=ntd)
    if fingerprint["barCount"] == 0:
        raise InsufficientDataError("no-finalized-candles", f"No finalized {TIMEFRAME} candles for {canonical}.")
    try:
        market_calendar.require_covered(
            vn_calendar.to_local(int(fingerprint["lastOpenTimeMs"])).date(), "latest finalized bar"
        )
    except vn_calendar.CalendarCoverageError as exc:
        raise InsufficientDataError(exc.reason, str(exc)) from exc
    calendar_provenance = {
        **market_calendar.provenance(),
        # Flagged, not dropped: the frozen protocol labels bar-indexed horizons, but
        # sessions before coverage start were filtered with an unknown holiday set.
        "barsBeforeCoverage": int(sum(
            market_calendar.is_before_coverage(vn_calendar.to_local(int(o)).date()) for o in bars.open_ms
        )),
    }
    if len(bars) <= WARMUP_BARS:
        raise InsufficientDataError(
            "insufficient-bars", f"{len(bars)} usable bars <= warm-up {WARMUP_BARS} for {canonical}."
        )
    X = compute_features(bars)
    results = [evaluate_config(bars, X, cfg, costs) for cfg in CONFIGS]

    generated = (generated_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    artifact = {
        "schemaVersion": SCHEMA_VERSION,
        "symbol": canonical,
        "timeframe": TIMEFRAME,
        "generatedAtUtc": generated.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "dataAsOfOpenTimeMs": fingerprint["lastOpenTimeMs"],
        "priceBasis": PRICE_BASIS,
        "provider": provider or descriptor["data_provider"],
        "dataset": fingerprint,
        "calendar": calendar_provenance,
        "method": {
            "name": METHOD_NAME,
            "version": METHOD_VERSION,
            "features": list(FEATURES),
            "kGrid": list(K_GRID),
            "warmupBars": WARMUP_BARS,
            "distance": "euclidean",
            "scaler": "z-score fit on the forecast point's eligible set",
            "innerSplit": INNER_TRAIN_FRACTION,
            "minMaturedForOos": MIN_MATURED_FOR_OOS,
            "folds": "expanding walk-forward, calendar years (Asia/Ho_Chi_Minh)",
            "baseline": "expanding class-frequency prior on the same timestamps",
            "pooling": "none (single symbol, own OHLCV only)",
        },
        "protocol": {
            "version": PROTOCOL_VERSION,
            "frozenAt": PROTOCOL_FROZEN_AT,
            "entry": "next-session-open (simulated, not a matched order)",
            "labelAvailability": "close of the resolving bar",
            "doubleTouch": "stop-first (flagged doubleTouch)",
            "gap": {"throughStop": "exit at open", "throughTarget": "exit at target"},
            "timeout": "exit at close of the last horizon bar",
            "immature": "excluded from train/eval (pending)",
            "costs": costs.to_dict(),
            "gate": {
                "minNEffective": GATE_MIN_N_EFFECTIVE,
                "brierDiffCi90UpperBelow": 0.0,
                "maxEce": GATE_MAX_ECE,
                "brierDiff": "kNN - prior (binary target-first)",
            },
            "interpretations": list(PROTOCOL_INTERPRETATIONS),
        },
        "configs": [
            {k: v for k, v in r.items() if not k.startswith("_")} for r in results
        ],
    }
    report = {
        "schemaVersion": REPORT_SCHEMA_VERSION,
        "artifact": artifact,
        "perConfig": [
            {"configId": r["configId"], "labels": r["_labels"], "folds": r["_folds"]} for r in results
        ],
        "reproduce": {
            "command": f"python equity_base_rates.py --symbol {canonical}",
            "cutoffMs": cutoff,
            "nonTradingDates": sorted(d.isoformat() for d in ntd),
            "calendar": calendar_provenance,
            "bootstrapSeed": BOOTSTRAP_SEED,
        },
    }
    return artifact, report


def artifact_path(out_root: Path | str, symbol: str) -> Path:
    return Path(out_root) / "equity-base-rates" / symbol / "base-rates.json"


def write_json_atomic(path: Path | str, payload: Mapping[str, Any]) -> Path:
    """Write JSON via a temp file in the same directory + ``os.replace`` (no partial file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    return path


def build_and_write(
    symbol: str,
    *,
    rows: Iterable[Mapping[str, Any]] | None = None,
    out_root: Path | str = DEFAULT_EVIDENCE_ROOT,
    report_dir: Path | str | None = None,
    cutoff_ms: int | None = None,
    calendar: vn_calendar.MarketCalendar | None = None,
) -> dict[str, Any]:
    """Build + write. Without an explicit ``calendar`` this is a DB-backed job: the
    HOSE calendar comes from ``MarketCalendarDays`` (MBB_NON_TRADING_DATES overrides)
    and an empty/unreadable table fails closed (``calendar-unavailable``)."""
    canonical = resolve_decision_symbol(symbol)
    cutoff = int(cutoff_ms if cutoff_ms is not None else time.time() * 1000)
    if calendar is None:
        try:
            calendar = vn_calendar.load_calendar_db()
        except vn_calendar.CalendarUnavailableError as exc:
            raise InsufficientDataError(exc.reason, str(exc)) from exc
    if rows is None:
        rows = load_candles_db(canonical, cutoff)
    artifact, report = build_artifact(rows, canonical, cutoff_ms=cutoff, calendar=calendar)
    path = write_json_atomic(artifact_path(out_root, canonical), artifact)
    report_path = None
    if report_dir is not None:
        report_path = write_json_atomic(Path(report_dir) / f"{canonical}-report.json", report)
    return {"artifact": artifact, "artifactPath": path, "reportPath": report_path}


def summarize(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "built",
        "schemaVersion": artifact["schemaVersion"],
        "symbol": artifact["symbol"],
        "timeframe": artifact["timeframe"],
        "generatedAtUtc": artifact["generatedAtUtc"],
        "dataAsOfOpenTimeMs": artifact["dataAsOfOpenTimeMs"],
        "datasetSha256": artifact["dataset"]["sha256"],
        "calendar": artifact.get("calendar"),
        "configs": [
            {
                "configId": c["configId"],
                "gatePassed": c["validation"]["gatePassed"],
                "gateReasons": c["validation"]["gateReasons"],
                "nEffective": c["validation"]["nEffective"],
                "currentAbstained": c["current"]["abstained"],
            }
            for c in artifact["configs"]
        ],
    }


def _load_fixture(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, Mapping):
        data = data.get("candles", [])
    return list(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Equity quant base rates (equity-base-rates-protocol-v1).")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--candles-json", default=None, help="InstrumentCandles-shaped JSON fixture; bypasses the DB.")
    parser.add_argument("--out-root", "--out", dest="out_root", default=str(DEFAULT_EVIDENCE_ROOT),
                        help="Evidence root (EvidenceCatalog:RootPath); artifact goes to equity-base-rates/{SYMBOL}/.")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--cutoff-ms", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        canonical = resolve_decision_symbol(args.symbol)
        rows = _load_fixture(args.candles_json) if args.candles_json else None
        # Fixture runs never read the DB: the calendar must come from the env override.
        calendar = None
        if rows is not None:
            try:
                calendar = vn_calendar.require_calendar()
            except vn_calendar.CalendarUnavailableError as exc:
                raise InsufficientDataError(exc.reason, str(exc)) from exc
        result = build_and_write(
            canonical,
            rows=rows,
            out_root=args.out_root,
            report_dir=None if args.no_report else args.report_dir,
            cutoff_ms=args.cutoff_ms,
            calendar=calendar,
        )
    except BaseRatesError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    summary = summarize(result["artifact"])
    print(json.dumps(summary, indent=2))
    print(f"artifact: {result['artifactPath']}")
    if result["reportPath"]:
        print(f"report: {result['reportPath']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
