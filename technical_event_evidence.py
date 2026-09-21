#!/usr/bin/env python3
"""Read-only, causal evidence audit for BTCUSDT 4h technical events.

This evaluator measures next-bar direction and close-to-close return after stored
technical events.  It is intentionally descriptive research: no trading fills,
PnL, model promotion, registry, or application tables are written.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np
import psycopg2

from db_config import get_db_connection, get_db_params
from research_contract import ResearchManifest


SYMBOL = "BTCUSDT"
TIMEFRAME = "4h"
INTERVAL_MS = 4 * 60 * 60 * 1000
EVALUATOR_VERSION = "btc-4h-technical-event-evidence-v1"
SMC_CAUSAL_VERSION = "smc-causal-v2"
MODULES = ("candle_patterns", "volume_anomaly", "market_regime", "causal_smc")
DOUBLE_CANDLE_PATTERNS = frozenset(
    {
        "BullishEngulfing",
        "BearishEngulfing",
        "PiercingLine",
        "DarkCloudCover",
        "BullishHarami",
        "BearishHarami",
        "TweezerBottoms",
        "TweezerTops",
    }
)
TRIPLE_CANDLE_PATTERNS = frozenset(
    {
        "MorningStar",
        "EveningStar",
        "ThreeWhiteSoldiers",
        "ThreeBlackCrows",
        "ThreeInsideUp",
        "ThreeInsideDown",
    }
)


@dataclass(frozen=True)
class EvaluatorConfig:
    min_event_history: int = 30
    min_context_history: int = 100
    bootstrap_samples: int = 2000
    block_size_events: int = 8
    alpha: float = 0.05
    random_seed: int = 42

    def validate(self) -> None:
        if self.min_event_history <= 0 or self.min_context_history <= 0:
            raise ValueError("history thresholds must be positive")
        if self.bootstrap_samples <= 0 or self.block_size_events <= 0:
            raise ValueError("bootstrap parameters must be positive")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")


@dataclass(frozen=True)
class BarData:
    open_ms: np.ndarray
    close_ms: np.ndarray
    close: np.ndarray
    contexts: tuple[str, ...]
    next_returns: np.ndarray
    next_positive: np.ndarray
    label_available_ms: np.ndarray

    def validate(self) -> None:
        n = len(self.open_ms)
        if n < 2:
            raise ValueError("at least two closed bars are required")
        for name in ("close_ms", "close", "next_returns", "next_positive", "label_available_ms"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} must align with open_ms")
        if len(self.contexts) != n:
            raise ValueError("contexts must align with open_ms")
        if np.any(self.open_ms[1:] <= self.open_ms[:-1]):
            raise ValueError("bar open timestamps must be strictly increasing")
        if not np.isfinite(self.close).all() or np.any(self.close <= 0):
            raise ValueError("close prices must be positive and finite")


@dataclass(frozen=True)
class EventOccurrence:
    event_type: str
    decision_index: int
    context: str


@dataclass(frozen=True)
class ModuleEvents:
    status: str
    reason: str | None
    availability_basis: str
    events: tuple[EventOccurrence, ...]
    diagnostics: dict[str, Any]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_parts(parts: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def _trend_contexts(close: np.ndarray) -> tuple[str, ...]:
    """Match the backend's causal six-close trend context."""
    contexts: list[str] = []
    for end in range(len(close)):
        start = max(0, end - 5)
        if end - start < 3:
            contexts.append("Sideways")
            continue
        changes = np.diff(close[start : end + 1])
        up = int(np.sum(changes > 0))
        down = int(np.sum(changes < 0))
        if up >= 4 and down <= 1:
            contexts.append("Uptrend")
        elif down >= 4 and up <= 1:
            contexts.append("Downtrend")
        else:
            contexts.append("Sideways")
    return tuple(contexts)


def build_bar_data(rows: list[tuple[Any, ...]]) -> BarData:
    if len(rows) < 2:
        raise ValueError("at least two closed bars are required")
    open_ms = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
    close_ms = np.asarray([int(row[1]) for row in rows], dtype=np.int64)
    close = np.asarray([float(row[2]) for row in rows], dtype=np.float64)
    next_returns = np.full(len(rows), np.nan, dtype=np.float64)
    next_positive = np.full(len(rows), -1, dtype=np.int8)
    label_available = np.full(len(rows), -1, dtype=np.int64)
    contiguous = open_ms[1:] - open_ms[:-1] == INTERVAL_MS
    valid_indices = np.flatnonzero(contiguous)
    next_returns[valid_indices] = close[valid_indices + 1] / close[valid_indices] - 1.0
    next_positive[valid_indices] = (next_returns[valid_indices] > 0).astype(np.int8)
    label_available[valid_indices] = close_ms[valid_indices + 1]
    result = BarData(
        open_ms=open_ms,
        close_ms=close_ms,
        close=close,
        contexts=_trend_contexts(close),
        next_returns=next_returns,
        next_positive=next_positive,
        label_available_ms=label_available,
    )
    result.validate()
    return result


def wilson_interval(successes: int, total: int, alpha: float = 0.05) -> dict[str, float] | None:
    if total <= 0:
        return None
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return {"lower": max(0.0, center - radius), "upper": min(1.0, center + radius)}


def moving_block_bootstrap_mean_ci(
    values: np.ndarray,
    *,
    samples: int,
    block_size: int,
    alpha: float,
    seed: int,
) -> dict[str, float] | None:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        return None
    rng = np.random.default_rng(seed)
    block = min(block_size, len(values))
    start_count = len(values) - block + 1
    full_blocks, remainder = divmod(len(values), block)
    block_prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    block_sums = block_prefix[block:] - block_prefix[:-block]
    estimates = np.empty(samples, dtype=np.float64)
    # Batch the bootstrap to keep peak memory bounded for regime-sized samples.
    for batch_start in range(0, samples, 128):
        batch_count = min(128, samples - batch_start)
        total = np.zeros(batch_count, dtype=np.float64)
        if full_blocks:
            starts = rng.integers(0, start_count, size=(batch_count, full_blocks))
            total += np.sum(block_sums[starts], axis=1)
        if remainder:
            starts = rng.integers(0, len(values) - remainder + 1, size=batch_count)
            total += block_prefix[starts + remainder] - block_prefix[starts]
        estimates[batch_start : batch_start + batch_count] = total / len(values)
    return {
        "lower": float(np.quantile(estimates, alpha / 2.0)),
        "upper": float(np.quantile(estimates, 1.0 - alpha / 2.0)),
    }


def _dataset_sha256(bars: BarData, modules: dict[str, ModuleEvents]) -> str:
    parts: list[bytes] = []
    for array in (bars.open_ms.astype("<i8"), bars.close_ms.astype("<i8"), bars.close.astype("<f8")):
        parts.extend((str(array.shape).encode("ascii"), array.tobytes(order="C")))
    serialized = {
        name: {
            "status": module.status,
            "reason": module.reason,
            "availabilityBasis": module.availability_basis,
            "events": [asdict(event) for event in module.events],
            "diagnostics": module.diagnostics,
        }
        for name, module in sorted(modules.items())
    }
    parts.append(_canonical_json(serialized).encode("utf-8"))
    return _sha256_parts(parts)


def _deduplicate(events: Iterable[EventOccurrence]) -> tuple[EventOccurrence, ...]:
    return tuple(sorted(set(events), key=lambda item: (item.decision_index, item.event_type, item.context)))


def pattern_decision_offset(pattern_type: str, category: str) -> int | None:
    """Return the final defining-bar offset without trusting known bad category rows."""
    if pattern_type in TRIPLE_CANDLE_PATTERNS:
        return 2
    if pattern_type in DOUBLE_CANDLE_PATTERNS:
        return 1
    if category == "Single":
        return 0
    return None


def _unavailable(reason: str, availability_basis: str, **diagnostics: Any) -> ModuleEvents:
    return ModuleEvents("unavailable", reason, availability_basis, (), diagnostics)


def _table_columns(cursor: Any, table_name: str) -> set[str]:
    cursor.execute(
        """
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        """,
        (table_name,),
    )
    return {str(row[0]) for row in cursor.fetchall()}


def _load_patterns(cursor: Any, bars: BarData) -> ModuleEvents:
    required = {"Symbol", "Timeframe", "OpenTimeMs", "PatternType", "PatternCategory", "TrendDirection"}
    columns = _table_columns(cursor, "CandlePatterns")
    if not columns:
        return _unavailable("CandlePatterns table is absent", "unavailable")
    missing = sorted(required - columns)
    if missing:
        return _unavailable(f"missing required columns: {missing}", "unavailable", missingColumns=missing)
    cursor.execute(
        'SELECT "OpenTimeMs", "PatternType", "PatternCategory", "TrendDirection" '
        'FROM "CandlePatterns" WHERE "Symbol"=%s AND "Timeframe"=%s ORDER BY "OpenTimeMs", "PatternType"',
        (SYMBOL, TIMEFRAME),
    )
    index = {int(value): i for i, value in enumerate(bars.open_ms)}
    events: list[EventOccurrence] = []
    skipped = 0
    corrected_category_rows = 0
    for start_ms, pattern_type, category, trend in cursor.fetchall():
        start_idx = index.get(int(start_ms))
        pattern_name = str(pattern_type)
        category_name = str(category)
        offset = pattern_decision_offset(pattern_name, category_name)
        category_offset = {"Single": 0, "Double": 1, "Triple": 2}.get(category_name)
        if offset is not None and category_offset != offset:
            corrected_category_rows += 1
        if start_idx is None or offset is None or start_idx + offset >= len(bars.open_ms):
            skipped += 1
            continue
        decision_idx = start_idx + offset
        if int(bars.open_ms[decision_idx]) - int(bars.open_ms[start_idx]) != offset * INTERVAL_MS:
            skipped += 1
            continue
        context = str(trend) if str(trend) in {"Uptrend", "Downtrend", "Sideways"} else bars.contexts[start_idx]
        events.append(EventOccurrence(pattern_name, decision_idx, context))
    return ModuleEvents(
        "evaluable",
        None,
        "Availability is reconstructed from the recognizer's frozen pattern-length map: Single at its close, known Double at the second contiguous close, and known Triple at the third; unknown multi-pattern names fail closed",
        _deduplicate(events),
        {
            "storedRows": len(events) + skipped,
            "skippedUnjoinableOrUnknownRows": skipped,
            "rowsWhoseStoredCategoryWasCorrectedByFrozenLengthMap": corrected_category_rows,
        },
    )


def _load_volume(cursor: Any, bars: BarData) -> ModuleEvents:
    required = {"Symbol", "Timeframe", "OpenTimeMs", "VolumeAnomalyRatio"}
    columns = _table_columns(cursor, "CandleVolumeStats")
    if not columns:
        return _unavailable("CandleVolumeStats table is absent", "unavailable")
    missing = sorted(required - columns)
    if missing:
        return _unavailable(f"missing required columns: {missing}", "unavailable", missingColumns=missing)
    cursor.execute(
        'SELECT "OpenTimeMs", "VolumeAnomalyRatio" FROM "CandleVolumeStats" '
        'WHERE "Symbol"=%s AND "Timeframe"=%s ORDER BY "OpenTimeMs"',
        (SYMBOL, TIMEFRAME),
    )
    index = {int(value): i for i, value in enumerate(bars.open_ms)}
    events: list[EventOccurrence] = []
    unjoinable = 0
    for open_ms, ratio in cursor.fetchall():
        idx = index.get(int(open_ms))
        if idx is None:
            unjoinable += 1
            continue
        value = float(ratio)
        if not math.isfinite(value):
            continue
        if value >= 1.5:
            events.append(EventOccurrence("volume_ratio_gte_1_5", idx, bars.contexts[idx]))
        if value >= 2.0:
            events.append(EventOccurrence("volume_ratio_gte_2_0", idx, bars.contexts[idx]))
    return ModuleEvents(
        "evaluable",
        None,
        "Current-candle volume statistics use only the finalized candle and prior rolling volume; available at that candle close",
        _deduplicate(events),
        {"predeclaredThresholds": [1.5, 2.0], "skippedUnjoinableRows": unjoinable},
    )


def _load_regimes(cursor: Any, bars: BarData) -> ModuleEvents:
    required = {"Symbol", "Timeframe", "OpenTimeMs", "RegimeType"}
    columns = _table_columns(cursor, "MarketRegimes")
    if not columns:
        return _unavailable("MarketRegimes table is absent", "unavailable")
    missing = sorted(required - columns)
    if missing:
        return _unavailable(f"missing required columns: {missing}", "unavailable", missingColumns=missing)
    cursor.execute(
        'SELECT "OpenTimeMs", "RegimeType" FROM "MarketRegimes" '
        'WHERE "Symbol"=%s AND "Timeframe"=%s ORDER BY "OpenTimeMs"',
        (SYMBOL, TIMEFRAME),
    )
    index = {int(value): i for i, value in enumerate(bars.open_ms)}
    events: list[EventOccurrence] = []
    unjoinable = 0
    for open_ms, regime in cursor.fetchall():
        idx = index.get(int(open_ms))
        if idx is None:
            unjoinable += 1
            continue
        events.append(EventOccurrence(str(regime), idx, bars.contexts[idx]))
    return ModuleEvents(
        "evaluable",
        None,
        "Stored regime indicators include the finalized current candle and prior bars; available at the matching candle close",
        _deduplicate(events),
        {"skippedUnjoinableRows": unjoinable},
    )


def _load_smc(cursor: Any, bars: BarData) -> ModuleEvents:
    required = {"Symbol", "Timeframe", "OriginTimeMs", "AvailableTimeMs", "EventType", "CalculationVersion"}
    columns = _table_columns(cursor, "SmartMoneyStructures")
    if not columns:
        return _unavailable("SmartMoneyStructures table is absent", "explicit AvailableTimeMs required")
    missing = sorted(required - columns)
    if missing:
        return _unavailable(
            f"causal availability cannot be established; missing required columns: {missing}",
            "explicit AvailableTimeMs required",
            missingColumns=missing,
        )
    cursor.execute(
        'SELECT "OriginTimeMs", "AvailableTimeMs", "EventType", "CalculationVersion" '
        'FROM "SmartMoneyStructures" WHERE "Symbol"=%s AND "Timeframe"=%s ORDER BY "AvailableTimeMs", "EventType"',
        (SYMBOL, TIMEFRAME),
    )
    close_index = {int(value): i for i, value in enumerate(bars.close_ms)}
    events: list[EventOccurrence] = []
    excluded_legacy = 0
    invalid_availability = 0
    unjoinable = 0
    for origin_ms, available_ms, event_type, version in cursor.fetchall():
        if str(version) != SMC_CAUSAL_VERSION:
            excluded_legacy += 1
            continue
        if available_ms is None or int(available_ms) < int(origin_ms):
            invalid_availability += 1
            continue
        idx = close_index.get(int(available_ms))
        if idx is None:
            unjoinable += 1
            continue
        events.append(EventOccurrence(str(event_type), idx, bars.contexts[idx]))
    if not events:
        return _unavailable(
            "no causal-v2 rows with valid availability join to closed 4h candles",
            "explicit AvailableTimeMs from smc-causal-v2",
            excludedLegacyRows=excluded_legacy,
            invalidAvailabilityRows=invalid_availability,
            skippedUnjoinableRows=unjoinable,
        )
    return ModuleEvents(
        "evaluable",
        None,
        "Explicit AvailableTimeMs from smc-causal-v2, joined exactly to a finalized candle close; mitigation state is excluded because it can be updated later",
        _deduplicate(events),
        {
            "excludedLegacyRows": excluded_legacy,
            "invalidAvailabilityRows": invalid_availability,
            "skippedUnjoinableRows": unjoinable,
        },
    )


def load_postgresql(decision_cutoff_ms: int) -> tuple[BarData, dict[str, ModuleEvents]]:
    conn = get_db_connection()
    try:
        conn.set_session(readonly=True, autocommit=False)
        with conn.cursor() as cursor:
            cursor.execute(
                'SELECT "OpenTimeMs", "CloseTimeMs", "Close" FROM "Klines" '
                'WHERE "Symbol"=%s AND "Timeframe"=%s AND "CloseTimeMs" <= %s '
                'ORDER BY "OpenTimeMs"',
                (SYMBOL, TIMEFRAME, decision_cutoff_ms),
            )
            bars = build_bar_data(cursor.fetchall())
            modules = {
                "candle_patterns": _load_patterns(cursor, bars),
                "volume_anomaly": _load_volume(cursor, bars),
                "market_regime": _load_regimes(cursor, bars),
                "causal_smc": _load_smc(cursor, bars),
            }
        conn.rollback()
        return bars, modules
    finally:
        conn.close()


def _history_by_context(bars: BarData) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for context in sorted(set(bars.contexts)):
        indices = [
            i
            for i, value in enumerate(bars.contexts)
            if value == context and bars.label_available_ms[i] >= 0
        ]
        labels = [int(bars.next_positive[i]) for i in indices]
        returns = [float(bars.next_returns[i]) for i in indices]
        result[context] = {
            "times": [int(bars.label_available_ms[i]) for i in indices],
            "indices": indices,
            "winsPrefix": np.concatenate(([0], np.cumsum(labels, dtype=np.int64))),
            "returnsPrefix": np.concatenate(([0.0], np.cumsum(returns, dtype=np.float64))),
        }
    return result


def evaluate_event_type(
    bars: BarData,
    occurrences: list[EventOccurrence],
    config: EvaluatorConfig,
    *,
    seed_offset: int = 0,
    inference_alpha: float | None = None,
) -> dict[str, Any]:
    adjusted_alpha = config.alpha if inference_alpha is None else inference_alpha
    context_history = _history_by_context(bars)
    realized = [event for event in occurrences if bars.label_available_ms[event.decision_index] >= 0]
    event_available_times = [int(bars.label_available_ms[event.decision_index]) for event in realized]
    order = np.argsort([event.decision_index for event in realized], kind="stable")
    realized = [realized[int(i)] for i in order]
    event_available_times = [event_available_times[int(i)] for i in order]

    actual: list[int] = []
    returns: list[float] = []
    candidate_probs: list[float] = []
    baseline_probs: list[float] = []
    baseline_return_means: list[float] = []
    decision_times: list[int] = []
    event_labels: list[int] = []
    event_label_available: list[int] = []
    event_wins_prefix: list[int] = [0]

    for event, available in zip(realized, event_available_times):
        decision_time = int(bars.close_ms[event.decision_index])
        prior_event_count = bisect.bisect_right(event_label_available, decision_time)
        baseline_history = context_history[event.context]
        prior_context_count = bisect.bisect_right(baseline_history["times"], decision_time)
        if prior_event_count >= config.min_event_history and prior_context_count >= config.min_context_history:
            event_wins = event_wins_prefix[prior_event_count]
            context_wins = int(baseline_history["winsPrefix"][prior_context_count])
            candidate_probs.append((event_wins + 1.0) / (prior_event_count + 2.0))
            baseline_probs.append((context_wins + 1.0) / (prior_context_count + 2.0))
            baseline_return_means.append(
                float(baseline_history["returnsPrefix"][prior_context_count]) / prior_context_count
            )
            actual.append(int(bars.next_positive[event.decision_index]))
            returns.append(float(bars.next_returns[event.decision_index]))
            decision_times.append(decision_time)
        event_labels.append(int(bars.next_positive[event.decision_index]))
        event_wins_prefix.append(event_wins_prefix[-1] + event_labels[-1])
        event_label_available.append(available)

    evaluated = len(actual)
    if evaluated == 0:
        return {
            "status": "insufficient_history",
            "storedOccurrences": len(occurrences),
            "realizedOccurrences": len(realized),
            "evaluatedOosOccurrences": 0,
            "coverageOfRealized": 0.0,
            "minimumEventHistory": config.min_event_history,
            "minimumContextHistory": config.min_context_history,
        }

    y = np.asarray(actual, dtype=np.float64)
    ret = np.asarray(returns, dtype=np.float64)
    event_p = np.asarray(candidate_probs, dtype=np.float64)
    base_p = np.asarray(baseline_probs, dtype=np.float64)
    base_return = np.asarray(baseline_return_means, dtype=np.float64)
    candidate_loss = np.square(y - event_p)
    baseline_loss = np.square(y - base_p)
    brier_lift = baseline_loss - candidate_loss
    positive_count = int(np.sum(y))
    status = "descriptive_only"
    lift_ci = moving_block_bootstrap_mean_ci(
        brier_lift,
        samples=config.bootstrap_samples,
        block_size=config.block_size_events,
        alpha=adjusted_alpha,
        seed=config.random_seed + seed_offset,
    )
    if evaluated < max(config.min_event_history, 30):
        status = "insufficient_evidence"
    elif lift_ci is not None and lift_ci["lower"] > 0:
        status = "positive_predictive_evidence"
    elif lift_ci is not None and lift_ci["upper"] < 0:
        status = "adverse_predictive_evidence"
    else:
        status = "inconclusive"
    return {
        "status": status,
        "storedOccurrences": len(occurrences),
        "realizedOccurrences": len(realized),
        "evaluatedOosOccurrences": evaluated,
        "coverageOfRealized": evaluated / len(realized) if realized else 0.0,
        "firstOosDecisionTimeMs": decision_times[0],
        "lastOosDecisionTimeMs": decision_times[-1],
        "outcome": {
            "positiveCount": positive_count,
            "positiveRate": positive_count / evaluated,
            "positiveRateWilsonCi": wilson_interval(positive_count, evaluated, config.alpha),
            "positiveRateWilsonFamilywiseCi": wilson_interval(positive_count, evaluated, adjusted_alpha),
            "meanReturnFraction": float(np.mean(ret)),
            "meanReturnBlockBootstrapCi": moving_block_bootstrap_mean_ci(
                ret,
                samples=config.bootstrap_samples,
                block_size=config.block_size_events,
                alpha=config.alpha,
                seed=config.random_seed + seed_offset + 10_000,
            ),
            "meanReturnBlockBootstrapFamilywiseCi": moving_block_bootstrap_mean_ci(
                ret,
                samples=config.bootstrap_samples,
                block_size=config.block_size_events,
                alpha=adjusted_alpha,
                seed=config.random_seed + seed_offset + 30_000,
            ),
        },
        "causalExpandingPriorComparison": {
            "eventPriorBrier": float(np.mean(candidate_loss)),
            "contextBaselineBrier": float(np.mean(baseline_loss)),
            "pairedBrierLift": float(np.mean(brier_lift)),
            "pairedBrierLiftBlockBootstrapCi": lift_ci,
            "eventPriorAccuracy": float(np.mean((event_p >= 0.5) == y)),
            "contextBaselineAccuracy": float(np.mean((base_p >= 0.5) == y)),
            "meanReturnMinusPriorContextMean": float(np.mean(ret - base_return)),
            "meanReturnMinusPriorContextMeanBlockBootstrapCi": moving_block_bootstrap_mean_ci(
                ret - base_return,
                samples=config.bootstrap_samples,
                block_size=config.block_size_events,
                alpha=config.alpha,
                seed=config.random_seed + seed_offset + 20_000,
            ),
            "baseline": "all prior BTCUSDT 4h bars in the same causal six-close trend context whose next-bar outcomes were available by the event decision",
            "inferenceAlpha": adjusted_alpha,
        },
    }


def evaluate_all(
    bars: BarData,
    modules: dict[str, ModuleEvents],
    config: EvaluatorConfig,
) -> dict[str, Any]:
    config.validate()
    module_reports: dict[str, Any] = {}
    ledger: list[dict[str, Any]] = []
    seed_offset = 0
    declared_trial_count = sum(
        len({event.event_type for event in module.events})
        for module in modules.values()
        if module.status == "evaluable"
    )
    adjusted_alpha = config.alpha / max(1, declared_trial_count)
    required_bootstrap_samples = math.ceil(2.0 / adjusted_alpha)
    if declared_trial_count and config.bootstrap_samples < required_bootstrap_samples:
        raise ValueError(
            "bootstrap_samples is too small for the Bonferroni-adjusted two-sided tail: "
            f"need at least {required_bootstrap_samples}, got {config.bootstrap_samples}"
        )
    for module_name in MODULES:
        module = modules[module_name]
        if module.status != "evaluable":
            report = {
                "status": "unavailable",
                "reason": module.reason,
                "availabilityBasis": module.availability_basis,
                "diagnostics": module.diagnostics,
                "events": {},
            }
            module_reports[module_name] = report
            ledger.append({"module": module_name, "eventType": None, "status": "unavailable", "reason": module.reason})
            continue
        grouped: dict[str, list[EventOccurrence]] = {}
        for occurrence in module.events:
            grouped.setdefault(occurrence.event_type, []).append(occurrence)
        event_reports: dict[str, Any] = {}
        for event_type in sorted(grouped):
            event_report = evaluate_event_type(
                bars,
                grouped[event_type],
                config,
                seed_offset=seed_offset,
                inference_alpha=adjusted_alpha,
            )
            seed_offset += 1
            event_reports[event_type] = event_report
            ledger.append({"module": module_name, "eventType": event_type, **event_report})
        module_reports[module_name] = {
            "status": "evaluable" if event_reports else "no_events",
            "reason": None if event_reports else "table is available but no predeclared events matched",
            "availabilityBasis": module.availability_basis,
            "diagnostics": module.diagnostics,
            "eventTypeCount": len(event_reports),
            "events": event_reports,
        }
    eligible = int(np.sum(bars.label_available_ms >= 0))
    return {
        "evaluatorVersion": EVALUATOR_VERSION,
        "scope": "BTCUSDT 4h closed-bar technical event evidence",
        "outcome": "signal close to next contiguous 4h close; positive iff return > 0",
        "eligibleOutcomeBars": eligible,
        "totalClosedBars": len(bars.open_ms),
        "modules": module_reports,
        "trials": ledger,
        "multipleTesting": {
            "method": "Bonferroni across every reported event type",
            "declaredTrialCount": declared_trial_count,
            "familywiseAlpha": config.alpha,
            "perTrialInferenceAlpha": adjusted_alpha,
            "minimumBootstrapSamplesForAdjustedTail": required_bootstrap_samples,
            "statusUsesFamilywisePairedBrierInterval": True,
        },
        "parametersTunedOnEvaluationPeriod": False,
        "chronologicalOos": True,
        "promotionAllowed": False,
        "pnlClaim": False,
        "negativeResultIsValid": True,
        "interpretation": "Predictive evidence here is conditional next-bar evidence only; it is not a tradable strategy or PnL estimate.",
    }


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_provenance(repo_dir: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repo_dir, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def build_manifest(
    bars: BarData,
    modules: dict[str, ModuleEvents],
    config: EvaluatorConfig,
    decision_cutoff_ms: int,
) -> dict[str, Any]:
    here = Path(__file__).resolve()
    contract_path = here.with_name("research_contract.py")
    db_params = get_db_params()
    module_inventory = {
        name: {
            "status": module.status,
            "eventRows": len(module.events),
            "availabilityBasis": module.availability_basis,
            "reason": module.reason,
        }
        for name, module in modules.items()
    }
    manifest = ResearchManifest(
        experiment="btc-4h-technical-event-evidence",
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        outcome_price_basis="signal-bar-close-to-next-contiguous-bar-close",
    )
    return manifest.to_dict(
        parameters={
            **asdict(config),
            "evaluatorVersion": EVALUATOR_VERSION,
            "modules": list(MODULES),
            "volumeThresholds": [1.5, 2.0],
            "doubleCandlePatterns": sorted(DOUBLE_CANDLE_PATTERNS),
            "tripleCandlePatterns": sorted(TRIPLE_CANDLE_PATTERNS),
            "smcCalculationVersion": SMC_CAUSAL_VERSION,
            "directionDefinition": "next return > 0",
            "selection": "none; every stored event type and both predeclared volume thresholds are reported",
        },
        data_provenance={
            "source": "postgresql-readonly",
            "datasetSha256": _dataset_sha256(bars, modules),
            "decisionCutoffMs": int(decision_cutoff_ms),
            "closedBarPredicate": "Klines.CloseTimeMs <= decision cutoff",
            "outcomeRealization": "next open is exactly +4h and next close is within cutoff",
            "rowCount": len(bars.open_ms),
            "firstOpenTimeMs": int(bars.open_ms[0]),
            "lastCloseTimeMs": int(bars.close_ms[-1]),
            "moduleInventory": module_inventory,
            "databaseIdentity": {
                "host": db_params["host"],
                "port": db_params["port"],
                "database": db_params["database"],
                "passwordExcluded": True,
            },
        },
        code_provenance={
            "evaluatorSha256": _file_hash(here),
            "researchContractSha256": _file_hash(contract_path),
            "git": _git_provenance(here.parent),
        },
        runtime_dependencies={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "psycopg2": psycopg2.__version__,
        },
    )


def _write_immutable(path: Path, content: str) -> None:
    """Publish a complete artifact atomically without ever replacing a peer."""
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != content:
                raise FileExistsError(f"refusing to overwrite immutable evidence artifact: {path}")
    finally:
        temporary_path.unlink(missing_ok=True)


def write_evidence(output_dir: Path, manifest: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_id = manifest["manifestSha256"]
    report = {"manifest": manifest, "evaluation": evaluation}
    report["reportSha256"] = hashlib.sha256(_canonical_json(report).encode("utf-8")).hexdigest()
    report_path = output_dir / f"{artifact_id}.report.json"
    ledger_path = output_dir / f"{artifact_id}.trials.jsonl"
    report_content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ledger_content = "".join(
        _canonical_json({"manifestSha256": artifact_id, **trial}) + "\n"
        for trial in evaluation["trials"]
    )
    for path, content in ((report_path, report_content), (ledger_path, ledger_content)):
        _write_immutable(path, content)
    return {"report": report_path, "trialLedger": ledger_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="BTCUSDT 4h technical event evidence evaluator")
    parser.add_argument("--decision-cutoff-ms", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-event-history", type=int, default=30)
    parser.add_argument("--min-context-history", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    config = EvaluatorConfig(
        min_event_history=args.min_event_history,
        min_context_history=args.min_context_history,
        bootstrap_samples=args.bootstrap_samples,
    )
    bars, modules = load_postgresql(args.decision_cutoff_ms)
    manifest = build_manifest(bars, modules, config, args.decision_cutoff_ms)
    evaluation = evaluate_all(bars, modules, config)
    paths = write_evidence(Path(args.output_dir), manifest, evaluation)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
