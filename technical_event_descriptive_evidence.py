#!/usr/bin/env python3
"""Causal, descriptive evidence bundle for BTCUSDT technical events.

The evaluator deliberately does not train, predict, tune, backtest trades, or
calculate PnL.  It freezes finalized candles and causally available technical
events, then describes what happened after each event.  Every aggregate can be
recomputed from the row-level ledger by :func:`verify_bundle`.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from technical_evidence_profiles import build_evidence_profiles
from technical_evidence_statistics import (
    STATISTICS_OUTPUT_SCHEMA_PATH,
    STATISTICS_SCHEMA,
    STATISTICS_SPEC_PATH,
    build_statistical_evidence,
    load_statistics_spec,
)
from technical_event_modules import (
    CONTRACT_PATH,
    build_causal_technical_events,
    build_sensitivity_variants,
    load_contract,
)


SCHEMA = "btc-technical-event-descriptive-evidence/v1"
SYMBOL = "BTCUSDT"
TIMEFRAME_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
HORIZONS = (1, 3, 6)
METRICS = ("forwardReturn", "mfe", "mae")
DERIVED_MODULES = (
    "technicalIndicators",
    "candlePatterns",
    "volumeAnomaly",
    "marketRegime",
    "fibonacci",
    "volumeProfile",
    "confluence",
)
EXCLUSION_REASONS = (
    "unknown_event_type",
    "unknown_event_lineage",
    "unknown_availability",
    "decision_bar_not_found",
    "decision_after_cutoff",
    "source_bar_not_found",
    "source_bar_not_available",
    "non_contiguous_followup",
    "overlap_deduplicated",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_chunks(value: Any) -> Iterable[bytes]:
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
    for chunk in encoder.iterencode(value):
        yield chunk.encode("utf-8")


def sha256_canonical(value: Any) -> str:
    digest = hashlib.sha256()
    for chunk in _canonical_chunks(value):
        digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


@dataclass(frozen=True)
class EvaluationConfig:
    cutoff_ms: int
    bootstrap_samples: int = 2_000
    block_size_events: int = 8
    alpha: float = 0.05
    dedup_bars: int = 6
    context_keys: tuple[str, ...] = ("trend", "volatility")
    random_seed: int = 42

    def validate(self) -> None:
        if self.cutoff_ms <= 0:
            raise ValueError("cutoff_ms must be positive")
        if self.bootstrap_samples <= 0 or self.block_size_events <= 0:
            raise ValueError("bootstrap settings must be positive")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if self.dedup_bars < 0:
            raise ValueError("dedup_bars must be non-negative")
        if len(set(self.context_keys)) != len(self.context_keys):
            raise ValueError("context_keys must be unique")


@dataclass(frozen=True)
class Candle:
    open_ms: int
    close_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    available_ms: int
    context: Mapping[str, str]


class _LifecycleLevelIndex:
    """First threshold crossing within a candle's contiguous segment.

    The evaluator previously scanned every later candle for every lifecycle
    level. A max/min segment tree preserves the same left-most-hit semantics
    while pruning ranges that cannot cross the requested price level.
    """

    __slots__ = (
        "_candles",
        "_segment_ends",
        "_size",
        "_high_max",
        "_low_min",
        "_available_max",
    )

    def __init__(self, candles: Sequence[Candle], interval_ms: int) -> None:
        self._candles = candles
        count = len(candles)
        segment_ends = [count - 1] * count
        segment_start = 0
        for index in range(1, count):
            if candles[index].open_ms - candles[index - 1].open_ms != interval_ms:
                for member in range(segment_start, index):
                    segment_ends[member] = index - 1
                segment_start = index
        self._segment_ends = tuple(segment_ends)

        size = 1
        while size < count:
            size *= 2
        self._size = size
        high_max = [-math.inf] * (2 * size)
        low_min = [math.inf] * (2 * size)
        available_max = [-math.inf] * (2 * size)
        for index, candle in enumerate(candles):
            high_max[size + index] = candle.high
            low_min[size + index] = candle.low
            available_max[size + index] = candle.available_ms
        for node in range(size - 1, 0, -1):
            high_max[node] = max(high_max[node * 2], high_max[node * 2 + 1])
            low_min[node] = min(low_min[node * 2], low_min[node * 2 + 1])
            available_max[node] = max(available_max[node * 2], available_max[node * 2 + 1])
        self._high_max = high_max
        self._low_min = low_min
        self._available_max = available_max

    def _first_matching(
        self,
        left: int,
        right: int,
        aggregates: Sequence[float],
        can_match: Any,
    ) -> int | None:
        def search(node: int, node_left: int, node_right: int) -> int | None:
            if node_right < left or right < node_left or not can_match(aggregates[node]):
                return None
            if node_left == node_right:
                return node_left
            midpoint = (node_left + node_right) // 2
            match = search(node * 2, node_left, midpoint)
            if match is not None:
                return match
            return search(node * 2 + 1, midpoint + 1, node_right)

        return search(1, 0, self._size - 1)

    def first_crossing(
        self,
        decision_index: int,
        operator: str,
        price: float,
        cutoff_ms: int,
    ) -> int | None:
        left = decision_index + 1
        if left >= len(self._candles):
            return None
        right = self._segment_ends[decision_index]
        if left > right:
            return None
        first_unavailable = self._first_matching(
            left,
            right,
            self._available_max,
            lambda value: value > cutoff_ms,
        )
        if first_unavailable is not None:
            right = first_unavailable - 1
        if left > right:
            return None
        if operator == "at_or_above":
            aggregates = self._high_max
            can_cross = lambda value: value >= price
        elif operator == "at_or_below":
            aggregates = self._low_min
            can_cross = lambda value: value <= price
        else:
            raise ValueError("level.operator must be at_or_above or at_or_below")
        return self._first_matching(left, right, aggregates, can_cross)


@dataclass(frozen=True)
class Dataset:
    timeframe: str
    interval_ms: int
    candles: tuple[Candle, ...]
    events: tuple[dict[str, Any], ...]
    lineage: Mapping[str, Any]
    modules: Mapping[str, Any]
    lifecycle_index: _LifecycleLevelIndex


def _validate_lineage(lineage: Any, owner: str) -> dict[str, Any]:
    if not isinstance(lineage, Mapping):
        raise ValueError(f"{owner} lineage is required")
    required = ("source", "sourceVersion", "contentSha256")
    normalized: dict[str, Any] = {}
    for key in required:
        value = lineage.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{owner} lineage.{key} is required")
        normalized[key] = value.strip()
    if len(normalized["contentSha256"]) != 64 or any(
        char not in "0123456789abcdef" for char in normalized["contentSha256"].lower()
    ):
        raise ValueError(f"{owner} lineage.contentSha256 must be a SHA-256 hex digest")
    normalized["contentSha256"] = normalized["contentSha256"].lower()
    if "sourceCutoffMs" in lineage:
        source_cutoff = lineage.get("sourceCutoffMs")
        if not isinstance(source_cutoff, int) or source_cutoff <= 0:
            raise ValueError(f"{owner} lineage.sourceCutoffMs must be a positive integer")
        normalized["sourceCutoffMs"] = source_cutoff
    return normalized


def parse_snapshot(raw: Mapping[str, Any], cutoff_ms: int) -> Dataset:
    if raw.get("symbol") != SYMBOL:
        raise ValueError("only BTCUSDT is supported")
    timeframe = str(raw.get("timeframe", ""))
    if timeframe not in TIMEFRAME_MS:
        raise ValueError("timeframe must be one of 1h, 4h, 1d")
    lineage = _validate_lineage(raw.get("lineage"), "dataset")
    interval = TIMEFRAME_MS[timeframe]
    candles: list[Candle] = []
    for index, item in enumerate(raw.get("candles") or []):
        if item.get("finalized") is not True:
            raise ValueError(f"candle[{index}] is not explicitly finalized")
        available_ms = item.get("availableTimeMs")
        if not isinstance(available_ms, int):
            raise ValueError(f"candle[{index}] has unknown availability")
        open_ms = int(item["openTimeMs"])
        close_ms = int(item["closeTimeMs"])
        if available_ms < close_ms:
            raise ValueError(f"candle[{index}] is available before it closes")
        if available_ms > cutoff_ms:
            continue
        open_price = _finite(item["open"], f"candle[{index}].open")
        high = _finite(item["high"], f"candle[{index}].high")
        low = _finite(item["low"], f"candle[{index}].low")
        close = _finite(item["close"], f"candle[{index}].close")
        volume = _finite(item.get("volume", 0.0), f"candle[{index}].volume")
        if min(open_price, high, low, close) <= 0 or volume < 0 or low > min(open_price, close) or high < max(open_price, close):
            raise ValueError(f"candle[{index}] has invalid OHLC")
        context_raw = item.get("context") or {}
        if not isinstance(context_raw, Mapping):
            raise ValueError(f"candle[{index}].context must be an object")
        candles.append(
            Candle(
                open_ms,
                close_ms,
                open_price,
                high,
                low,
                close,
                volume,
                available_ms,
                {str(key): str(value) for key, value in sorted(context_raw.items())},
            )
        )
    if not candles:
        raise ValueError("snapshot has no finalized candles available by cutoff")
    candles.sort(key=lambda candle: candle.open_ms)
    if len({candle.open_ms for candle in candles}) != len(candles):
        raise ValueError("candle open times must be unique")
    for index in range(1, len(candles)):
        if candles[index].available_ms < candles[index - 1].available_ms:
            raise ValueError("candle availability must be non-decreasing")
    for candle in candles:
        if candle.close_ms - candle.open_ms + 1 != interval:
            raise ValueError("candle duration does not match timeframe")
    events = tuple(dict(item) for item in (raw.get("events") or []))
    event_ids = [str(event.get("eventId", "")) for event in events]
    if any(not event_id for event_id in event_ids) or len(set(event_ids)) != len(event_ids):
        raise ValueError("eventId must be present and unique")
    modules = raw.get("modules") or {}
    if not isinstance(modules, Mapping):
        raise ValueError("modules must be an object")
    frozen_candles = tuple(candles)
    return Dataset(
        timeframe,
        interval,
        frozen_candles,
        events,
        lineage,
        dict(modules),
        _LifecycleLevelIndex(frozen_candles, interval),
    )


def _causal_contexts(kline_rows: Sequence[Mapping[str, Any]], interval_ms: int) -> list[dict[str, str]]:
    closes = np.asarray([float(row["Close"]) for row in kline_rows], dtype=np.float64)
    highs = np.asarray([float(row["High"]) for row in kline_rows], dtype=np.float64)
    lows = np.asarray([float(row["Low"]) for row in kline_rows], dtype=np.float64)
    true_range_pct = np.empty(len(closes), dtype=np.float64)
    for index in range(len(closes)):
        contiguous = index > 0 and int(kline_rows[index]["OpenTimeMs"]) - int(kline_rows[index - 1]["OpenTimeMs"]) == interval_ms
        if not contiguous:
            true_range = highs[index] - lows[index]
        else:
            true_range = max(highs[index] - lows[index], abs(highs[index] - closes[index - 1]), abs(lows[index] - closes[index - 1]))
        true_range_pct[index] = true_range / closes[index]
    contexts: list[dict[str, str]] = []
    segment_start = 0
    for end in range(len(kline_rows)):
        if end > 0 and int(kline_rows[end]["OpenTimeMs"]) - int(kline_rows[end - 1]["OpenTimeMs"]) != interval_ms:
            segment_start = end
        start = max(segment_start, end - 5)
        changes = np.diff(closes[start : end + 1])
        up = int(np.sum(changes > 0))
        down = int(np.sum(changes < 0))
        trend = "sideways"
        if len(changes) >= 3 and up >= 4 and down <= 1:
            trend = "up"
        elif len(changes) >= 3 and down >= 4 and up <= 1:
            trend = "down"
        historical = true_range_pct[max(segment_start, end - 20) : end]
        volatility = "insufficient_history"
        if len(historical) >= 10:
            median = float(np.median(historical))
            if median > 0 and true_range_pct[end] >= 1.5 * median:
                volatility = "high"
            elif median > 0 and true_range_pct[end] <= 0.67 * median:
                volatility = "low"
            else:
                volatility = "normal"
        contexts.append({"trend": trend, "volatility": volatility})
    return contexts


def _source_times_for_smc(row: Mapping[str, Any], interval_ms: int) -> list[int]:
    event_type = str(row.get("EventType") or "")
    origin = row.get("OriginTimeMs") if row.get("OriginTimeMs") is not None else row.get("TimeMs")
    if not isinstance(origin, int):
        return []
    if event_type.startswith("FVG_"):
        return [origin - interval_ms, origin, origin + interval_ms]
    if event_type.startswith("SWING_"):
        return [origin - 2 * interval_ms, origin - interval_ms, origin, origin + interval_ms, origin + 2 * interval_ms]
    reference = row.get("ReferenceTimeMs")
    if event_type.startswith(("BOS_", "CHOCH_")) and isinstance(reference, int):
        return sorted(
            {
                reference - 2 * interval_ms,
                reference - interval_ms,
                reference,
                reference + interval_ms,
                reference + 2 * interval_ms,
                origin,
            }
        )
    return [origin]


def _smc_lifecycle(row: Mapping[str, Any]) -> dict[str, Any] | None:
    event_type = str(row.get("EventType") or "")
    high = row.get("HighPrice")
    low = row.get("LowPrice")
    if high is None or low is None or not event_type.startswith("FVG_"):
        return None
    high_value = _finite(high, "SmartMoneyStructures.HighPrice")
    low_value = _finite(low, "SmartMoneyStructures.LowPrice")
    if low_value >= high_value:
        return None
    if event_type == "FVG_BULL":
        touch = {"operator": "at_or_below", "price": high_value}
        mitigation = {"operator": "at_or_below", "price": low_value}
    elif event_type == "FVG_BEAR":
        touch = {"operator": "at_or_above", "price": low_value}
        mitigation = {"operator": "at_or_above", "price": high_value}
    else:
        return None
    return {
        "semanticsVersion": "smc-causal-fvg-zone-v1",
        "touchLevel": touch,
        "mitigationLevel": mitigation,
        "invalidationUnavailableReason": "smc-causal-v2 defines complete fill as mitigation and has no validated invalidation rule",
    }


def build_postgresql_snapshot_from_rows(
    timeframe: str,
    cutoff_ms: int,
    kline_rows: Sequence[Mapping[str, Any]],
    smc_rows: Sequence[Mapping[str, Any]],
    *,
    smc_columns: Sequence[str],
    causal_smc_rows: Sequence[Mapping[str, Any]] | None = None,
    causal_smc_columns: Sequence[str] = (),
    causal_smc_checkpoint: Mapping[str, Any] | None = None,
    causal_smc_checkpoint_columns: Sequence[str] = (),
) -> dict[str, Any]:
    """Pure mapping used by the read-only PostgreSQL loader and unit tests."""
    if timeframe not in TIMEFRAME_MS:
        raise ValueError("timeframe must be one of 1h, 4h, 1d")
    interval = TIMEFRAME_MS[timeframe]
    all_klines = sorted(kline_rows, key=lambda row: int(row["OpenTimeMs"]))
    ordered_klines = [
        row
        for row in all_klines
        if int(row["CloseTimeMs"]) <= cutoff_ms
        and int(row["CloseTimeMs"]) - int(row["OpenTimeMs"]) + 1 == interval
    ]
    invalid_duration_count = sum(
        int(row["CloseTimeMs"]) <= cutoff_ms
        and int(row["CloseTimeMs"]) - int(row["OpenTimeMs"]) + 1 != interval
        for row in all_klines
    )
    contexts = _causal_contexts(ordered_klines, interval) if ordered_klines else []
    candles = [
        {
            "openTimeMs": int(row["OpenTimeMs"]),
            "closeTimeMs": int(row["CloseTimeMs"]),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row.get("Volume") or 0.0),
            "availableTimeMs": int(row["CloseTimeMs"]),
            "finalized": True,
            "context": contexts[index],
        }
        for index, row in enumerate(ordered_klines)
    ]
    candle_context = {item["closeTimeMs"]: item["context"] for item in candles}
    canonical_required = {
        "EventId", "EventType", "OriginTimeMs", "AvailableTimeMs",
        "CalculationVersion", "DecisionSourceOpenTimeMsJson",
        "DecisionEvidenceJson", "DecisionEvidenceSha256",
    }
    canonical_schema = canonical_required.issubset(set(causal_smc_columns))
    derived_events, derived_modules = build_causal_technical_events(
        timeframe, interval, ordered_klines, contexts
    )
    events: list[dict[str, Any]] = list(derived_events)
    causal_count = 0
    future_causal_count = 0
    legacy_count = 0
    for raw_row in sorted(
        causal_smc_rows or [],
        key=lambda row: (int(row.get("AvailableTimeMs") or 2**63 - 1), str(row.get("EventId") or "")),
    ):
        if not canonical_schema:
            raise ValueError("CausalSmartMoneyEvents schema is incomplete")
        row = dict(raw_row)
        if not isinstance(row.get("AvailableTimeMs"), int) or int(row["AvailableTimeMs"]) > cutoff_ms:
            future_causal_count += 1
            continue
        evidence_json = row.get("DecisionEvidenceJson")
        evidence_sha = str(row.get("DecisionEvidenceSha256") or "").lower()
        if not isinstance(evidence_json, str) or sha256_bytes(evidence_json.encode("utf-8")) != evidence_sha:
            raise ValueError("CausalSmartMoneyEvents decision evidence hash mismatch")
        try:
            evidence = json.loads(evidence_json)
            source_times = json.loads(str(row.get("DecisionSourceOpenTimeMsJson") or "[]"))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("CausalSmartMoneyEvents decision evidence JSON is invalid") from exc
        if (
            not isinstance(evidence, Mapping)
            or not isinstance(source_times, list)
            or not source_times
            or any(not isinstance(value, int) for value in source_times)
            or len(set(source_times)) != len(source_times)
        ):
            raise ValueError("CausalSmartMoneyEvents decision source list is invalid")
        evidence_sources = evidence.get("sourceCandles")
        if not isinstance(evidence_sources, list) or [item.get("openTimeMs") for item in evidence_sources] != source_times:
            raise ValueError("CausalSmartMoneyEvents source list disagrees with decision evidence")
        if row.get("DecisionSourceCandleCount") is not None and int(row["DecisionSourceCandleCount"]) != len(source_times):
            raise ValueError("CausalSmartMoneyEvents decision source count mismatch")
        for column, evidence_key in (
            ("EventId", "eventId"),
            ("EventType", "eventType"),
            ("OriginTimeMs", "originTimeMs"),
            ("AvailableTimeMs", "availableTimeMs"),
            ("CalculationVersion", "calculationVersion"),
        ):
            if evidence.get(evidence_key) != row.get(column):
                raise ValueError(f"CausalSmartMoneyEvents immutable decision mismatch: {column}")
        event_type = str(row["EventType"])
        expected_state = "active" if event_type in ("FVG_BULL", "FVG_BEAR") else "confirmed"
        if evidence.get("stateAtAsOf") != expected_state or evidence.get("mitigatedAtMs") is not None:
            raise ValueError("CausalSmartMoneyEvents decision evidence contains future lifecycle state")
        available = int(row["AvailableTimeMs"])
        event = {
            "eventId": str(row["EventId"]),
            "module": "causalSmc",
            "eventType": event_type,
            "formedTimeMs": int(row["OriginTimeMs"]),
            "confirmedTimeMs": available,
            "availableTimeMs": available,
            "sourceCandleOpenTimesMs": source_times,
            "context": candle_context.get(available, {}),
            "lineage": {
                "source": "postgresql:CausalSmartMoneyEvents",
                "sourceVersion": str(row["CalculationVersion"]),
                "contentSha256": evidence_sha,
            },
        }
        lifecycle = _smc_lifecycle(row)
        if lifecycle:
            event["lifecycle"] = lifecycle
        events.append(event)
        causal_count += 1

    for raw_row in sorted(smc_rows, key=lambda row: (row.get("AvailableTimeMs") or 2**63 - 1, str(row.get("EventType")))):
        row = dict(raw_row)
        version = str(row.get("CalculationVersion") or "unknown")
        available = None
        legacy_count += 1
        immutable_core = {
            key: row.get(key)
            for key in (
                "Id",
                "TimeMs",
                "OriginTimeMs",
                "ReferenceTimeMs",
                "EventType",
                "CalculationVersion",
                "Price",
                "HighPrice",
                "LowPrice",
            )
        }
        # Legacy availability is deliberately normalized away. A later database
        # backfill must not rewrite the hash of an earlier fail-closed snapshot.
        immutable_core["AvailableTimeMs"] = available
        raw_digest = sha256_bytes(canonical_json(immutable_core).encode())
        identity = {
            "rowId": row.get("Id"),
            "symbol": SYMBOL,
            "timeframe": timeframe,
            "eventType": row.get("EventType"),
            "originTimeMs": row.get("OriginTimeMs") or row.get("TimeMs"),
            "referenceTimeMs": row.get("ReferenceTimeMs"),
            "availableTimeMs": available,
            "calculationVersion": version,
            "rowSha256": raw_digest,
        }
        event = {
            "eventId": "smc-" + sha256_bytes(canonical_json(identity).encode())[:32],
            "eventType": str(row.get("EventType") or ""),
            "formedTimeMs": row.get("OriginTimeMs") if row.get("OriginTimeMs") is not None else row.get("TimeMs"),
            "confirmedTimeMs": available,
            "availableTimeMs": available,
            "sourceCandleOpenTimesMs": _source_times_for_smc(row, interval),
            "context": candle_context.get(available, {}),
            "lineage": {
                "source": "postgresql:SmartMoneyStructures",
                "sourceVersion": version,
                "contentSha256": raw_digest,
            },
        }
        events.append(event)
    first_requested_open = int(ordered_klines[0]["OpenTimeMs"]) if ordered_klines else None
    last_requested_open = int(ordered_klines[-1]["OpenTimeMs"]) if ordered_klines else None
    checkpoint = dict(causal_smc_checkpoint or {})
    checkpoint_version_ok = checkpoint.get("CalculationVersion") == "smc-causal-v2"
    coverage_start = checkpoint.get("CoverageStartOpenTimeMs")
    last_processed = checkpoint.get("LastProcessedOpenTimeMs")
    checkpoint_status = str(checkpoint.get("Status") or "not_started")
    start_covered = (
        first_requested_open is not None
        and isinstance(coverage_start, int)
        and coverage_start <= first_requested_open
    )
    end_covered = (
        last_requested_open is not None
        and isinstance(last_processed, int)
        and last_processed >= last_requested_open
    )
    coverage_complete = (
        canonical_schema
        and checkpoint_version_ok
        and checkpoint_status in {"complete", "checkpointed"}
        and start_covered
        and end_covered
    )
    if coverage_complete:
        module_status = "evaluable"
        coverage_reason = None
    elif canonical_schema and checkpoint:
        module_status = "partial"
        coverage_reason = "canonical event ledger does not cover the full requested finalized-candle window at this cutoff"
    else:
        module_status = "unavailable"
        coverage_reason = "canonical table, immutable decision schema, or versioned rebuild checkpoint is unavailable"
    modules = {
        "finalizedCandles": {
            "status": "evaluable" if candles else "unavailable",
            "includedRows": len(candles),
            "excludedInvalidDurationRows": invalid_duration_count,
            "availabilityBasis": "CloseTimeMs at or before cutoff; exact timeframe duration",
        },
        "causalSmc": {
            "status": module_status,
            "canonicalTable": "CausalSmartMoneyEvents",
            "causalRows": causal_count,
            "futureRowsExcludedByCutoff": future_causal_count,
            "excludedLegacyOrUnknownAvailabilityRows": legacy_count,
            "reason": coverage_reason,
            "requiredColumns": sorted(canonical_required),
            "presentColumns": sorted(set(causal_smc_columns)),
            "legacyAuditTable": "SmartMoneyStructures",
            "legacyPresentColumns": sorted(set(smc_columns)),
            "coverage": {
                "calculationVersion": checkpoint.get("CalculationVersion"),
                "checkpointStatus": checkpoint_status,
                "coverageStartOpenTimeMs": coverage_start,
                "lastProcessedOpenTimeMs": last_processed,
                "requestedFirstOpenTimeMs": first_requested_open,
                "requestedLastOpenTimeMs": last_requested_open,
                "coversRequestedCutoff": coverage_complete,
                "presentColumns": sorted(set(causal_smc_checkpoint_columns)),
            },
        }
    }
    modules.update(derived_modules)
    source_payload = {"timeframe": timeframe, "cutoffMs": cutoff_ms, "candles": candles, "events": events, "modules": modules}
    schema_digest = sha256_bytes(canonical_json({
        "canonical": sorted(set(causal_smc_columns)),
        "canonicalCheckpoint": sorted(set(causal_smc_checkpoint_columns)),
        "legacyAudit": sorted(set(smc_columns)),
    }).encode())
    return {
        "symbol": SYMBOL,
        "timeframe": timeframe,
        "lineage": {
            "source": "postgresql:Klines+CausalSmartMoneyEvents+SmartMoneyStructuresLegacyAudit",
            "sourceVersion": schema_digest,
            "contentSha256": sha256_canonical(source_payload),
            "sourceCutoffMs": cutoff_ms,
        },
        "candles": candles,
        "events": events,
        "modules": modules,
    }


def _table_columns(cursor: Any, table_name: str) -> set[str]:
    cursor.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name=%s",
        (table_name,),
    )
    return {str(row[0]) for row in cursor.fetchall()}


def load_postgresql_snapshot(timeframe: str, cutoff_ms: int) -> dict[str, Any]:
    """Read a cutoff snapshot inside a PostgreSQL read-only transaction."""
    from db_config import get_db_connection

    connection = get_db_connection()
    try:
        # PostgreSQL's default READ COMMITTED isolation takes a fresh snapshot
        # for every statement.  This loader reads Klines, table metadata and SMC
        # rows in separate statements, so an ingestion commit between them could
        # otherwise produce a bundle assembled from different database states.
        connection.set_session(
            readonly=True,
            autocommit=False,
            isolation_level="REPEATABLE READ",
        )
        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT "OpenTimeMs","CloseTimeMs","Open","High","Low","Close","Volume" FROM "Klines" '
                'WHERE "Symbol"=%s AND "Timeframe"=%s AND "CloseTimeMs"<=%s ORDER BY "OpenTimeMs"',
                (SYMBOL, timeframe, cutoff_ms),
            )
            names = ("OpenTimeMs", "CloseTimeMs", "Open", "High", "Low", "Close", "Volume")
            klines = [dict(zip(names, row)) for row in cursor.fetchall()]
            causal_columns = _table_columns(cursor, "CausalSmartMoneyEvents")
            causal_rows: list[dict[str, Any]] = []
            if causal_columns:
                causal_selected = [
                    name
                    for name in (
                        "Id", "EventId", "EventType", "OriginTimeMs", "AvailableTimeMs",
                        "ReferenceTimeMs", "Price", "HighPrice", "LowPrice", "State",
                        "MitigatedAtMs", "MitigationSourceOpenTimeMs", "CalculationVersion",
                        "SegmentStartOpenTimeMs", "DecisionSourceCandleCount",
                        "DecisionSourceOpenTimeMsJson", "DecisionEvidenceJson",
                        "DecisionEvidenceSha256",
                    )
                    if name in causal_columns
                ]
                causal_quoted = ",".join(f'"{name}"' for name in causal_selected)
                cursor.execute(
                    f'SELECT {causal_quoted} FROM "CausalSmartMoneyEvents" '
                    'WHERE "Symbol"=%s AND "Timeframe"=%s AND "AvailableTimeMs"<=%s '
                    'ORDER BY "AvailableTimeMs", "EventId"',
                    (SYMBOL, timeframe, cutoff_ms),
                )
                causal_rows = [dict(zip(causal_selected, row)) for row in cursor.fetchall()]

            checkpoint_columns = _table_columns(cursor, "CausalSmartMoneyRebuildCheckpoints")
            causal_checkpoint: dict[str, Any] | None = None
            if checkpoint_columns:
                checkpoint_selected = [
                    name
                    for name in (
                        "CalculationVersion", "LastProcessedOpenTimeMs", "CoverageStartOpenTimeMs",
                        "ProcessedCandleCount", "MaterializedEventCount", "Status", "LastError",
                    )
                    if name in checkpoint_columns
                ]
                checkpoint_quoted = ",".join(f'"{name}"' for name in checkpoint_selected)
                cursor.execute(
                    f'SELECT {checkpoint_quoted} FROM "CausalSmartMoneyRebuildCheckpoints" '
                    'WHERE "Symbol"=%s AND "Timeframe"=%s AND "CalculationVersion"=%s '
                    'ORDER BY "UpdatedAtUtc" DESC LIMIT 1',
                    (SYMBOL, timeframe, "smc-causal-v2"),
                )
                checkpoint_row = cursor.fetchone()
                if checkpoint_row is not None:
                    causal_checkpoint = dict(zip(checkpoint_selected, checkpoint_row))

            columns = _table_columns(cursor, "SmartMoneyStructures")
            smc_rows: list[dict[str, Any]] = []
            if columns:
                selected = [
                    name
                    for name in (
                        "Id", "TimeMs", "OriginTimeMs", "ReferenceTimeMs", "AvailableTimeMs", "EventType",
                        "CalculationVersion", "Price", "HighPrice", "LowPrice",
                    )
                    if name in columns
                ]
                quoted = ",".join(f'"{name}"' for name in selected)
                sql = (
                    f'SELECT {quoted} FROM "SmartMoneyStructures" WHERE "Symbol"=%s AND "Timeframe"=%s '
                    'AND "TimeMs"<=%s ORDER BY "TimeMs", "EventType"'
                )
                params: tuple[Any, ...] = (SYMBOL, timeframe, cutoff_ms)
                cursor.execute(sql, params)
                smc_rows = [dict(zip(selected, row)) for row in cursor.fetchall()]
        connection.rollback()
        return build_postgresql_snapshot_from_rows(
            timeframe,
            cutoff_ms,
            klines,
            smc_rows,
            smc_columns=sorted(columns),
            causal_smc_rows=causal_rows,
            causal_smc_columns=sorted(causal_columns),
            causal_smc_checkpoint=causal_checkpoint,
            causal_smc_checkpoint_columns=sorted(checkpoint_columns),
        )
    finally:
        connection.close()


def _context_signature(context: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    values: list[tuple[str, str]] = []
    for key in keys:
        value = context.get(key)
        if value is None or str(value).strip() == "":
            return None
        values.append((key, str(value)))
    return canonical_json(values)


def _event_exclusion(
    event: Mapping[str, Any],
    dataset: Dataset,
    config: EvaluationConfig,
    close_to_index: Mapping[int, int],
    open_to_index: Mapping[int, int],
) -> tuple[str | None, int | None]:
    if not isinstance(event.get("eventType"), str) or not str(event.get("eventType")).strip():
        return "unknown_event_type", None
    try:
        _validate_lineage(event.get("lineage"), f"event {event.get('eventId')}")
    except ValueError:
        return "unknown_event_lineage", None
    available_ms = event.get("availableTimeMs")
    confirmed_ms = event.get("confirmedTimeMs")
    if not isinstance(available_ms, int) or not isinstance(confirmed_ms, int) or available_ms < confirmed_ms:
        return "unknown_availability", None
    if available_ms > config.cutoff_ms:
        return "decision_after_cutoff", None
    decision_index = close_to_index.get(available_ms)
    if decision_index is None:
        return "decision_bar_not_found", None
    source_times = event.get("sourceCandleOpenTimesMs")
    if not isinstance(source_times, list) or not source_times or any(not isinstance(value, int) for value in source_times):
        return "unknown_availability", None
    source_indices: list[int] = []
    for source_time in source_times:
        source_index = open_to_index.get(source_time)
        if source_index is None:
            return "source_bar_not_found", None
        source_indices.append(source_index)
    if any(dataset.candles[index].available_ms > available_ms for index in source_indices):
        return "source_bar_not_available", None
    if max(source_indices) > decision_index:
        return "source_bar_not_available", None
    if _context_signature(event.get("context") or {}, config.context_keys) is None:
        return "unknown_availability", None
    return None, decision_index


def _is_contiguous(candles: Sequence[Candle], start: int, end: int, interval_ms: int) -> bool:
    return all(candles[index].open_ms - candles[index - 1].open_ms == interval_ms for index in range(start + 1, end + 1))


def _level_hit(candle: Candle, level: Mapping[str, Any] | None) -> bool:
    if not level:
        return False
    price = _finite(level.get("price"), "level.price")
    operator = level.get("operator")
    if operator == "at_or_above":
        return candle.high >= price
    if operator == "at_or_below":
        return candle.low <= price
    raise ValueError("level.operator must be at_or_above or at_or_below")


def _horizon_metrics(dataset: Dataset, decision_index: int, horizon: int, cutoff_ms: int) -> dict[str, Any] | None:
    end = decision_index + horizon
    if end >= len(dataset.candles):
        return None
    if dataset.candles[end].available_ms > cutoff_ms or not _is_contiguous(dataset.candles, decision_index, end, dataset.interval_ms):
        return None
    entry = dataset.candles[decision_index].close
    followup = dataset.candles[decision_index + 1 : end + 1]
    return {
        "bars": horizon,
        "elapsedMs": dataset.candles[end].close_ms - dataset.candles[decision_index].close_ms,
        "availableTimeMs": dataset.candles[end].available_ms,
        "forwardReturn": dataset.candles[end].close / entry - 1.0,
        "mfe": max(candle.high / entry - 1.0 for candle in followup),
        "mae": min(candle.low / entry - 1.0 for candle in followup),
    }


def _time_to_level(
    dataset: Dataset,
    decision_index: int,
    level: Mapping[str, Any] | None,
    cutoff_ms: int,
) -> dict[str, int] | None:
    if not level:
        return None
    next_index = decision_index + 1
    if (
        next_index >= len(dataset.candles)
        or dataset.candles[next_index].available_ms > cutoff_ms
        or dataset.candles[next_index].open_ms - dataset.candles[decision_index].open_ms
        != dataset.interval_ms
    ):
        return None
    price = _finite(level.get("price"), "level.price")
    operator = level.get("operator")
    index = dataset.lifecycle_index.first_crossing(decision_index, operator, price, cutoff_ms)
    if index is None:
        return None
    candle = dataset.candles[index]
    return {
        "bars": index - decision_index,
        "elapsedMs": candle.close_ms - dataset.candles[decision_index].close_ms,
        "timeMs": candle.close_ms,
    }


def _lifecycle(
    dataset: Dataset, decision_index: int, event: Mapping[str, Any], cutoff_ms: int
) -> dict[str, Any]:
    lifecycle = event.get("lifecycle")
    if not isinstance(lifecycle, Mapping):
        return {
            "status": "unavailable",
            "reason": "event type has no declared lifecycle semantics",
            "firstTouch": None,
            "mitigation": None,
            "invalidation": None,
        }
    version = lifecycle.get("semanticsVersion")
    if not isinstance(version, str) or not version.strip():
        return {
            "status": "unavailable",
            "reason": "lifecycle semantics version is unknown",
            "firstTouch": None,
            "mitigation": None,
            "invalidation": None,
        }
    touch_level = lifecycle.get("touchLevel")
    mitigation_level = lifecycle.get("mitigationLevel")
    invalidation_level = lifecycle.get("invalidationLevel")
    if touch_level is None and mitigation_level is None and invalidation_level is None:
        return {
            "status": "unavailable",
            "reason": "lifecycle has no touch, mitigation, or invalidation level",
            "firstTouch": None,
            "mitigation": None,
            "invalidation": None,
        }
    return {
        "status": "supported",
        "reason": None,
        "semanticsVersion": version,
        "firstTouch": _time_to_level(dataset, decision_index, touch_level, cutoff_ms),
        "mitigation": _time_to_level(dataset, decision_index, mitigation_level, cutoff_ms),
        "invalidation": _time_to_level(dataset, decision_index, invalidation_level, cutoff_ms),
        "invalidationUnavailableReason": lifecycle.get("invalidationUnavailableReason")
        if invalidation_level is None
        else None,
    }


def _moving_block_inference(
    values: Sequence[float],
    config: EvaluationConfig,
    seed_offset: int = 0,
) -> dict[str, Any] | None:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2 or not np.isfinite(array).all():
        return None
    # Keep at least two possible block starts when the sample has more than one
    # row; using one full-length block would report a misleading zero-width CI.
    block = min(config.block_size_events, max(1, len(array) // 2))
    blocks = math.ceil(len(array) / block)
    rng = np.random.default_rng(config.random_seed + seed_offset)
    estimates = np.empty(config.bootstrap_samples, dtype=np.float64)
    # The legacy implementation materialized every sampled value in nested
    # Python loops.  A bootstrap mean only needs each sampled block's sum.  Draw
    # starts in row-major chunks so NumPy consumes the exact same RNG sequence;
    # the final block uses only its leading remainder, matching picked[:n].
    prefix = np.concatenate(([0.0], np.cumsum(array, dtype=np.float64)))
    full_sums = prefix[block:] - prefix[:-block]
    final_length = len(array) - (blocks - 1) * block
    final_sums = prefix[final_length:] - prefix[:-final_length]
    max_start = len(array) - block + 1
    chunk_size = max(1, min(config.bootstrap_samples, 1_000_000 // blocks))
    for first in range(0, config.bootstrap_samples, chunk_size):
        count = min(chunk_size, config.bootstrap_samples - first)
        starts = rng.integers(0, max_start, size=(count, blocks))
        totals = final_sums[starts[:, -1]].copy()
        if blocks > 1:
            totals += np.sum(full_sums[starts[:, :-1]], axis=1)
        estimates[first : first + count] = totals / len(array)
    observed = float(np.mean(array))
    centered_null = estimates - observed
    return {
        "interval": {
            "lower": float(np.quantile(estimates, config.alpha / 2)),
            "upper": float(np.quantile(estimates, 1 - config.alpha / 2)),
        },
        "centeredNullTwoSidedPValue": float(
            (1 + np.count_nonzero(np.abs(centered_null) >= abs(observed)))
            / (config.bootstrap_samples + 1)
        ),
    }


def _moving_block_ci(values: Sequence[float], config: EvaluationConfig, seed_offset: int = 0) -> dict[str, float] | None:
    inference = _moving_block_inference(values, config, seed_offset)
    return None if inference is None else inference["interval"]


def _summary(
    values: Sequence[float],
    config: EvaluationConfig,
    seed_offset: int,
    *,
    include_centered_null_p: bool = False,
) -> dict[str, Any] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    inference = _moving_block_inference(values, config, seed_offset)
    result = {
        "count": len(values),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "meanBlockBootstrapInterval": None if inference is None else inference["interval"],
    }
    if include_centered_null_p:
        result["centeredNullTwoSidedPValue"] = (
            None if inference is None else inference["centeredNullTwoSidedPValue"]
        )
    return result


def _select_controls(rows: list[dict[str, Any]], dataset: Dataset, config: EvaluationConfig) -> None:
    all_candidates: dict[str, list[int]] = {}
    candidate_horizons: dict[int, dict[str, Any]] = {}
    for index, candle in enumerate(dataset.candles):
        signature = _context_signature(candle.context, config.context_keys)
        if signature is None:
            continue
        horizons = {
            str(horizon): _horizon_metrics(dataset, index, horizon, config.cutoff_ms)
            for horizon in HORIZONS
        }
        if horizons[str(max(HORIZONS))] is None:
            continue
        all_candidates.setdefault(signature, []).append(index)
        candidate_horizons[index] = horizons

    families: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        identity = (str(row.get("module") or "unknown"), str(row.get("eventType") or "unknown"))
        families.setdefault(identity, []).append(row)

    for family_rows in families.values():
        event_decisions = {
            int(row["decisionIndex"])
            for row in family_rows
            if row.get("decisionIndex") is not None
        }
        reserved = {
            index
            for decision in event_decisions
            for index in range(
                max(0, decision - max(HORIZONS)),
                min(len(dataset.candles), decision + max(HORIZONS) + 1),
            )
        }
        candidates = {
            signature: [index for index in sequence if index not in reserved]
            for signature, sequence in all_candidates.items()
        }
        positions: dict[str, int] = {}
        available: dict[str, list[int]] = {}
        for row in sorted(
            (item for item in family_rows if item["status"] == "eligible"),
            key=lambda item: item["decisionIndex"],
        ):
            decision_index = row["decisionIndex"]
            decision_available_ms = dataset.candles[decision_index].available_ms
            event_available = row.get("availableTimeMs")
            if isinstance(event_available, int):
                decision_available_ms = min(decision_available_ms, event_available)
            signature = row["contextSignature"]
            sequence = candidates.get(signature, [])
            position = positions.get(signature, 0)
            heap = available.setdefault(signature, [])
            max_horizon_key = str(max(HORIZONS))
            while (
                position < len(sequence)
                and candidate_horizons[sequence[position]][max_horizon_key]["availableTimeMs"] < decision_available_ms
            ):
                heapq.heappush(heap, -sequence[position])
                position += 1
            positions[signature] = position
            while heap and candidate_horizons[-heap[0]][max_horizon_key]["availableTimeMs"] >= decision_available_ms:
                heapq.heappop(heap)
            if not heap:
                row["control"] = {"status": "unavailable", "reason": "no_prior_context_match"}
                continue
            chosen = -heapq.heappop(heap)
            row["control"] = {
                "status": "matched",
                "decisionOpenTimeMs": dataset.candles[chosen].open_ms,
                "decisionCloseTimeMs": dataset.candles[chosen].close_ms,
                "controlOutcomeEndIndex": chosen + max(HORIZONS),
                "outcomeAvailableBeforeEvent": candidate_horizons[chosen][max_horizon_key]["availableTimeMs"] < decision_available_ms,
                "contextSignature": signature,
                "horizons": candidate_horizons[chosen],
            }


def _evaluate_event_rows(
    events: Sequence[Mapping[str, Any]],
    dataset: Dataset,
    config: EvaluationConfig,
    *,
    lifecycle: bool,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    last_kept: dict[tuple[str, str], int] = {}
    overlap_candidates = 0
    close_to_index = {candle.close_ms: index for index, candle in enumerate(dataset.candles)}
    open_to_index = {candle.open_ms: index for index, candle in enumerate(dataset.candles)}
    def event_sort_key(item: Mapping[str, Any]) -> tuple[int, str]:
        available = item.get("availableTimeMs")
        return (available if isinstance(available, int) else 2**63 - 1, str(item.get("eventId")))

    for event in sorted(events, key=event_sort_key):
        event_type = str(event.get("eventType") or "")
        base = {
            "eventId": str(event["eventId"]),
            "module": str(event.get("module") or ("causalSmc" if str(event.get("eventId", "")).startswith("smc-") else "unknown")),
            "eventType": event_type,
            "formedTimeMs": event.get("formedTimeMs"),
            "confirmedTimeMs": event.get("confirmedTimeMs"),
            "availableTimeMs": event.get("availableTimeMs"),
            "lineage": event.get("lineage"),
            "status": "excluded",
            "exclusionReason": None,
            "decisionIndex": None,
            "decisionOpenTimeMs": None,
            "decisionCloseTimeMs": None,
            "context": event.get("context") or {},
            "contextSignature": None,
            "horizons": {},
            "lifecycle": {
                "status": "not_evaluated",
                "reason": None,
                "firstTouch": None,
                "mitigation": None,
                "invalidation": None,
            },
            "control": {"status": "not_attempted"},
        }
        reason, decision_index = _event_exclusion(
            event,
            dataset,
            config,
            close_to_index,
            open_to_index,
        )
        if reason:
            base["exclusionReason"] = reason
            rows.append(base)
            continue
        assert decision_index is not None
        signature = _context_signature(event.get("context") or {}, config.context_keys)
        assert signature is not None
        key = (event_type, signature)
        if key in last_kept and decision_index - last_kept[key] <= config.dedup_bars:
            base["decisionIndex"] = decision_index
            base["decisionOpenTimeMs"] = dataset.candles[decision_index].open_ms
            base["decisionCloseTimeMs"] = dataset.candles[decision_index].close_ms
            base["contextSignature"] = signature
            base["exclusionReason"] = "overlap_deduplicated"
            overlap_candidates += 1
            rows.append(base)
            continue
        last_kept[key] = decision_index
        horizons = {
            str(horizon): _horizon_metrics(dataset, decision_index, horizon, config.cutoff_ms)
            for horizon in HORIZONS
        }
        base.update(
            {
                "status": "eligible",
                "decisionIndex": decision_index,
                "decisionOpenTimeMs": dataset.candles[decision_index].open_ms,
                "decisionCloseTimeMs": dataset.candles[decision_index].close_ms,
                "contextSignature": signature,
                "horizons": horizons,
                "lifecycle": _lifecycle(dataset, decision_index, event, config.cutoff_ms) if lifecycle else {
                    "status": "not_evaluated",
                    "reason": "sensitivity audit does not infer lifecycle semantics",
                    "firstTouch": None,
                    "mitigation": None,
                    "invalidation": None,
                },
            }
        )
        if horizons[str(max(HORIZONS))] is None and decision_index + max(HORIZONS) < len(dataset.candles):
            base["status"] = "excluded"
            base["exclusionReason"] = "non_contiguous_followup"
        rows.append(base)
    return rows, overlap_candidates


def evaluate(snapshot: Mapping[str, Any], config: EvaluationConfig) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    config.validate()
    dataset = parse_snapshot(snapshot, config.cutoff_ms)
    rows, overlap_candidates = _evaluate_event_rows(dataset.events, dataset, config, lifecycle=True)
    _select_controls(rows, dataset, config)
    report = recompute_report(rows, dataset, config)
    report["dependence"] = {
        "deduplication": "keep earliest event per eventType and declared context within dedupBars",
        "dedupBars": config.dedup_bars,
        "overlapCandidatesExcluded": overlap_candidates,
        "remainingOutcomeWindowsMayOverlap": True,
        "intervalMethod": "moving block bootstrap over chronological event rows",
        "independenceClaimed": False,
    }
    filtered_snapshot = {
        "schema": SCHEMA,
        "symbol": SYMBOL,
        "timeframe": dataset.timeframe,
        "cutoffMs": config.cutoff_ms,
        "lineage": dict(dataset.lineage),
        "candles": [
            {
                "openTimeMs": candle.open_ms,
                "closeTimeMs": candle.close_ms,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
                "availableTimeMs": candle.available_ms,
                "finalized": True,
                "context": dict(candle.context),
            }
            for candle in dataset.candles
        ],
        "events": list(dataset.events),
        "modules": dict(dataset.modules),
    }
    return filtered_snapshot, rows, report


def _dataset_rows(dataset: Dataset) -> list[dict[str, Any]]:
    return [
        {
            "OpenTimeMs": candle.open_ms,
            "CloseTimeMs": candle.close_ms,
            "Open": candle.open,
            "High": candle.high,
            "Low": candle.low,
            "Close": candle.close,
            "Volume": candle.volume,
        }
        for candle in dataset.candles
    ]


def _sensitivity_summary(
    dataset: Dataset,
    config: EvaluationConfig,
    baseline_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    kline_rows = _dataset_rows(dataset)
    contexts = [dict(candle.context) for candle in dataset.candles]
    derived_modules = {
        "technicalIndicators",
        "candlePatterns",
        "volumeAnomaly",
        "marketRegime",
        "fibonacci",
        "volumeProfile",
    }
    baseline_events = [event for event in dataset.events if event.get("module") in derived_modules]
    contract, contract_sha = load_contract()
    declared_grid = {
        module: contract["modules"][module]["sensitivity"]
        for module in sorted(contract["modules"])
    }
    variants = build_sensitivity_variants(
        dataset.timeframe,
        dataset.interval_ms,
        kline_rows,
        contexts,
        baseline_events,
    )

    def summarize(rows: Sequence[Mapping[str, Any]], seed_offset: int) -> dict[str, Any]:
        eligible = [row for row in rows if row.get("status") == "eligible"]
        exclusions: dict[str, int] = {}
        for row in rows:
            reason = row.get("exclusionReason")
            if reason:
                exclusions[str(reason)] = exclusions.get(str(reason), 0) + 1
        horizons: dict[str, Any] = {}
        for horizon in HORIZONS:
            key = str(horizon)
            realized = [row for row in eligible if row.get("horizons", {}).get(key) is not None]
            metrics = {
                metric: _summary(
                    [float(row["horizons"][key][metric]) for row in realized],
                    config,
                    seed_offset + horizon * 10 + metric_index,
                )
                for metric_index, metric in enumerate(METRICS)
            }
            horizons[key] = {"realized": len(realized), "metrics": metrics}
        return {
            "stored": len(rows),
            "eligible": len(eligible),
            "excluded": len(rows) - len(eligible),
            "realizedAtMaxHorizon": sum(
                row.get("horizons", {}).get(str(max(HORIZONS))) is not None for row in eligible
            ),
            "exclusionReasons": dict(sorted(exclusions.items())),
            "eligibleDecisionTimes": sorted(
                {int(row["decisionCloseTimeMs"]) for row in eligible if row.get("decisionCloseTimeMs") is not None}
            ),
            "horizons": horizons,
        }

    baseline_by_module: dict[str, dict[str, Any]] = {}
    for module in sorted(derived_modules | {"confluence"}):
        module_rows = [row for row in baseline_rows if row.get("module") == module]
        baseline_by_module[module] = summarize(module_rows, 100_000 + len(baseline_by_module) * 1_000)

    audits: list[dict[str, Any]] = []
    for index, variant in enumerate(variants):
        variant_rows, overlap = _evaluate_event_rows(variant["events"], dataset, config, lifecycle=False)
        summary = summarize(variant_rows, 200_000 + index * 1_000)
        baseline = baseline_by_module[variant["module"]]
        left = set(baseline["eligibleDecisionTimes"])
        right = set(summary["eligibleDecisionTimes"])
        union = left | right
        horizon_comparison: dict[str, Any] = {}
        for horizon in HORIZONS:
            key = str(horizon)
            metrics: dict[str, Any] = {}
            for metric in METRICS:
                variant_metric = summary["horizons"][key]["metrics"][metric] or {
                    "count": 0, "mean": None, "median": None, "lower": None, "upper": None
                }
                baseline_metric = baseline["horizons"][key]["metrics"][metric] or {
                    "count": 0, "mean": None, "median": None, "lower": None, "upper": None
                }
                variant_mean = variant_metric["mean"]
                baseline_mean = baseline_metric["mean"]
                metrics[metric] = {
                    "variant": variant_metric,
                    "baselineMean": baseline_mean,
                    "meanDeltaVsBaseline": None
                    if variant_mean is None or baseline_mean is None
                    else variant_mean - baseline_mean,
                }
            horizon_comparison[key] = {
                "elapsedTimeMs": horizon * dataset.interval_ms,
                "realized": summary["horizons"][key]["realized"],
                "metrics": metrics,
            }
        audits.append(
            {
                "module": variant["module"],
                "variantId": variant["variantId"],
                "parameters": variant["parameters"],
                "stored": summary["stored"],
                "eligible": summary["eligible"],
                "excluded": summary["excluded"],
                "realizedAtMaxHorizon": summary["realizedAtMaxHorizon"],
                "exclusionReasons": summary["exclusionReasons"],
                "overlapCandidatesExcluded": overlap,
                "eligibleDecisionTimeJaccardVsBaseline": 1.0 if not union else len(left & right) / len(union),
                "horizons": horizon_comparison,
            }
        )
    executed_variant_ids = [f"{item['module']}:{item['variantId']}" for item in audits]
    return {
        "method": "one_axis_at_a_time",
        "resultSelection": False,
        "promotionAllowed": False,
        "preRegisteredGrid": {
            "contractDefinitionsSha256": contract_sha,
            "declaredGrid": declared_grid,
            "declaredGridSha256": sha256_bytes(canonical_json(declared_grid).encode()),
            "baselineParametersFixedInContract": {
                module: contract["modules"][module]["parameters"]
                for module in sorted(contract["modules"])
            },
            "executedVariantIds": executed_variant_ids,
            "allExecutedVariantsRetained": len(executed_variant_ids) == len(set(executed_variant_ids)) == len(audits),
            "outcomeDrivenSelectionAllowed": False,
        },
        "variantCount": len(audits),
        "variants": audits,
    }


def recompute_report(rows: Sequence[Mapping[str, Any]], dataset: Dataset, config: EvaluationConfig) -> dict[str, Any]:
    stored = len(rows)
    eligible_rows = [row for row in rows if row.get("status") == "eligible"]
    exclusions = {reason: 0 for reason in EXCLUSION_REASONS}
    for row in rows:
        reason = row.get("exclusionReason")
        if reason:
            exclusions[reason] = exclusions.get(reason, 0) + 1
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in eligible_rows:
        groups.setdefault(str(row["eventType"]), []).append(row)
    event_types: dict[str, Any] = {}
    for group_index, (event_type, group) in enumerate(sorted(groups.items())):
        horizons: dict[str, Any] = {}
        for horizon in HORIZONS:
            key = str(horizon)
            realized = [row for row in group if row.get("horizons", {}).get(key) is not None]
            controls = [row for row in realized if row.get("control", {}).get("status") == "matched"]
            metrics: dict[str, Any] = {}
            for metric_index, metric in enumerate(METRICS):
                values = [float(row["horizons"][key][metric]) for row in realized]
                paired = [
                    float(row["horizons"][key][metric]) - float(row["control"]["horizons"][key][metric])
                    for row in controls
                ]
                metrics[metric] = {
                    "event": _summary(values, config, group_index * 100 + horizon * 10 + metric_index),
                    "matchedControlDifference": _summary(
                        paired,
                        config,
                        10_000 + group_index * 100 + horizon * 10 + metric_index,
                        include_centered_null_p=True,
                    ),
                }
            horizons[key] = {
                "elapsedTimeMs": horizon * dataset.interval_ms,
                "eligible": len(group),
                "realized": len(realized),
                "matchedControls": len(controls),
                "metrics": metrics,
            }
        touches = [
            int(row["lifecycle"]["firstTouch"]["bars"])
            for row in group
            if row.get("lifecycle", {}).get("firstTouch")
        ]
        invalidations = [
            int(row["lifecycle"]["invalidation"]["bars"])
            for row in group
            if row.get("lifecycle", {}).get("invalidation")
        ]
        mitigations = [
            int(row["lifecycle"]["mitigation"]["bars"])
            for row in group
            if row.get("lifecycle", {}).get("mitigation")
        ]
        lifecycle_supported = sum(row.get("lifecycle", {}).get("status") == "supported" for row in group)
        event_types[event_type] = {
            "stored": sum(1 for row in rows if row.get("eventType") == event_type),
            "eligible": len(group),
            "horizons": horizons,
            "timeToFirstTouchBars": _summary(touches, config, 20_000 + group_index),
            "timeToMitigationBars": _summary(mitigations, config, 25_000 + group_index),
            "timeToInvalidationBars": _summary(invalidations, config, 30_000 + group_index),
            "lifecycleCoverage": {
                "supported": lifecycle_supported,
                "unavailable": len(group) - lifecycle_supported,
            },
        }
    report = {
        "schema": SCHEMA,
        "claimType": "descriptive_technical_event_history",
        "predictiveEvidence": False,
        "probabilityClaim": False,
        "economicClaim": False,
        "promotionAllowed": False,
        "scope": {"symbol": SYMBOL, "timeframe": dataset.timeframe, "cutoffMs": config.cutoff_ms},
        "modules": dict(dataset.modules),
        "declaredFamily": {
            "eventTypes": sorted(groups),
            "horizonsBars": list(HORIZONS),
            "metrics": list(METRICS),
            "contextKeys": list(config.context_keys),
            "selection": "all causally eligible stored event types; no result-based filtering",
        },
        "counts": {
            "stored": stored,
            "eligible": len(eligible_rows),
            "excluded": stored - len(eligible_rows),
            "realizedAtMaxHorizon": sum(
                row.get("horizons", {}).get(str(max(HORIZONS))) is not None for row in eligible_rows
            ),
            "exclusionReasons": {key: value for key, value in sorted(exclusions.items()) if value},
        },
        "eventTypes": event_types,
        "limitations": [
            "Historical descriptions are not probabilities or predictions for a current event.",
            "Outcome windows can overlap after deterministic same-type deduplication.",
            "Context controls are historical comparisons, not randomized counterfactuals.",
            "No fees, fills, slippage, position sizing, trades, or PnL are evaluated.",
        ],
    }
    sensitivity = _sensitivity_summary(dataset, config, rows)
    report["sensitivityAudit"] = sensitivity
    report["statisticalEvidence"] = build_statistical_evidence(
        rows,
        event_types,
        timeframe=dataset.timeframe,
        horizons=HORIZONS,
        metrics=METRICS,
        block_size_events=config.block_size_events,
        bootstrap_samples=config.bootstrap_samples,
        random_seed=config.random_seed,
        sensitivity=sensitivity,
    )
    report["evidenceProfiles"] = build_evidence_profiles(
        rows,
        [
            {
                "openTimeMs": candle.open_ms,
                "closeTimeMs": candle.close_ms,
                "context": dict(candle.context),
            }
            for candle in dataset.candles
        ],
        timeframe=dataset.timeframe,
        interval_ms=dataset.interval_ms,
        modules=dataset.modules,
        horizons=HORIZONS,
        sensitivity=sensitivity,
    )
    return report


def _git_state(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def _write_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"refusing to overwrite conflicting artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _write_content_addressed(
    output_dir: Path,
    suffix: str,
    chunks: Iterable[bytes],
) -> tuple[Path, str, int]:
    """Stream an immutable artifact without materializing its full JSON bytes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / f".artifact-{uuid.uuid4().hex}.tmp"
    digest = hashlib.sha256()
    size = 0
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            for chunk in chunks:
                stream.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        sha = digest.hexdigest()
        destination = output_dir / f"{sha}.{suffix}"
        if destination.exists():
            if destination.stat().st_size != size or sha256_file(destination) != sha:
                raise FileExistsError(f"refusing conflicting content-addressed artifact: {destination}")
            temporary.unlink()
        else:
            os.replace(temporary, destination)
        return destination, sha, size
    finally:
        temporary.unlink(missing_ok=True)


def _code_provenance() -> dict[str, Any]:
    evaluator = Path(__file__).resolve()
    modules = evaluator.with_name("technical_event_modules.py")
    profiles = evaluator.with_name("technical_evidence_profiles.py")
    statistics = evaluator.with_name("technical_evidence_statistics.py")
    profile_schema = evaluator.with_name("contracts") / "technical-evidence-profile.schema.json"
    contract, definitions_sha = load_contract()
    statistical_spec, statistical_spec_sha = load_statistics_spec()
    components = [
        {"role": "evaluator", "fileName": evaluator.name, "sha256": sha256_file(evaluator)},
        {"role": "technicalModules", "fileName": modules.name, "sha256": sha256_file(modules)},
        {"role": "evidenceProfiles", "fileName": profiles.name, "sha256": sha256_file(profiles)},
        {"role": "evidenceProfileSchema", "fileName": profile_schema.name, "sha256": sha256_file(profile_schema)},
        {"role": "evidenceStatistics", "fileName": statistics.name, "sha256": sha256_file(statistics)},
        {
            "role": "evidenceStatisticalSpec",
            "fileName": STATISTICS_SPEC_PATH.name,
            "sha256": sha256_file(STATISTICS_SPEC_PATH),
            "definitionsSha256": statistical_spec_sha,
            "specVersion": statistical_spec["specVersion"],
        },
        {
            "role": "evidenceStatisticsSchema",
            "fileName": STATISTICS_OUTPUT_SCHEMA_PATH.name,
            "sha256": sha256_file(STATISTICS_OUTPUT_SCHEMA_PATH),
        },
        {
            "role": "technicalModuleContract",
            "fileName": CONTRACT_PATH.name,
            "sha256": sha256_file(CONTRACT_PATH),
            "definitionsSha256": definitions_sha,
            "contractVersion": contract["contractVersion"],
        },
    ]
    return {
        "fileName": evaluator.name,
        "sha256": sha256_bytes(canonical_json(components).encode()),
        "components": components,
    }


def _runtime_provenance(manifest_path: Path) -> dict[str, Any]:
    return {
        "schema": "btc-technical-evidence-runtime/v1",
        "semanticManifestFileName": manifest_path.name,
        "semanticManifestSha256": manifest_path.name.split(".", 1)[0],
        "python": {
            "implementation": sys.implementation.name,
            "version": sys.version.split()[0],
            "executable": sys.executable,
        },
        "platform": {
            "sysPlatform": sys.platform,
            "osName": os.name,
            "processorArchitecture": os.environ.get("PROCESSOR_ARCHITECTURE", "unknown"),
        },
        "numpyVersion": np.__version__,
        "git": _git_state(Path(__file__).resolve().parent),
        "semanticHashIncludesRuntimeMetadata": False,
    }


def write_bundle(
    output_dir: Path,
    snapshot: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    config: EvaluationConfig,
) -> dict[str, Path]:
    artifacts: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    streams = (
        ("snapshot", "snapshot.json", chain(_canonical_chunks(snapshot), (b"\n",))),
        (
            "ledger",
            "ledger.jsonl",
            (chunk for row in rows for chunk in chain(_canonical_chunks(row), (b"\n",))),
        ),
        ("report", "report.json", chain(_canonical_chunks(report), (b"\n",))),
    )
    for role, suffix, chunks in streams:
        path, digest, size = _write_content_addressed(output_dir, suffix, chunks)
        paths[role] = path
        artifacts[role] = {"fileName": path.name, "sha256": digest, "sizeBytes": size}
    manifest = {
        "schema": SCHEMA,
        "createdFromCutoffMs": config.cutoff_ms,
        "scope": report["scope"],
        "claimType": report["claimType"],
        "configuration": {
            "horizonsBars": list(HORIZONS),
            "bootstrapSamples": config.bootstrap_samples,
            "blockSizeEvents": config.block_size_events,
            "alpha": config.alpha,
            "dedupBars": config.dedup_bars,
            "contextKeys": list(config.context_keys),
            "randomSeed": config.random_seed,
        },
        "sourceLineage": snapshot["lineage"],
        "code": _code_provenance(),
        "statisticalSpec": {
            "schema": STATISTICS_SCHEMA,
            "sha256": report["statisticalEvidence"]["specSha256"],
        },
        "artifacts": artifacts,
        "immutability": "content-addressed; conflicting overwrite refused",
        "runtimeMetadata": "separate content-addressed sidecar; excluded from semantic manifest hash",
    }
    manifest_bytes = (canonical_json(manifest) + "\n").encode()
    manifest_digest = sha256_bytes(manifest_bytes)
    manifest_path = output_dir / f"{manifest_digest}.manifest.json"
    _write_immutable(manifest_path, manifest_bytes)
    paths["manifest"] = manifest_path
    runtime = _runtime_provenance(manifest_path)
    runtime_path, _, _ = _write_content_addressed(
        output_dir,
        "runtime.json",
        chain(_canonical_chunks(runtime), (b"\n",)),
    )
    paths["runtime"] = runtime_path
    return paths


def verify_bundle(manifest_path: Path) -> dict[str, Any]:
    manifest_bytes = manifest_path.read_bytes()
    if manifest_path.name != f"{sha256_bytes(manifest_bytes)}.manifest.json":
        raise ValueError("manifest filename hash mismatch")
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != SCHEMA or manifest.get("claimType") != "descriptive_technical_event_history":
        raise ValueError("unsupported or non-descriptive manifest")
    declared_code = manifest.get("code")
    current_code = _code_provenance()
    if not isinstance(declared_code, Mapping) or declared_code.get("sha256") != current_code["sha256"]:
        raise ValueError("evaluator/module/contract provenance hash mismatch")
    if declared_code.get("components") != current_code["components"]:
        raise ValueError("evaluator provenance components mismatch")
    _, statistical_spec_sha = load_statistics_spec()
    if manifest.get("statisticalSpec") != {
        "schema": STATISTICS_SCHEMA,
        "sha256": statistical_spec_sha,
    }:
        raise ValueError("statistical evidence spec provenance mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {"snapshot", "ledger", "report"}:
        raise ValueError("manifest artifact roles mismatch")
    artifact_paths: dict[str, Path] = {}
    for role in ("snapshot", "ledger", "report"):
        item = artifacts[role]
        if not isinstance(item, Mapping) or set(item) != {"fileName", "sha256", "sizeBytes"}:
            raise ValueError(f"{role} artifact declaration mismatch")
        path = manifest_path.parent / item["fileName"]
        if path.name != item["fileName"] or path.parent.resolve() != manifest_path.parent.resolve():
            raise ValueError("artifact path traversal rejected")
        if sha256_file(path) != item["sha256"] or path.stat().st_size != item["sizeBytes"]:
            raise ValueError(f"{role} integrity mismatch")
        if not path.name.startswith(item["sha256"] + "."):
            raise ValueError(f"{role} filename is not content-addressed")
        artifact_paths[role] = path
    with artifact_paths["snapshot"].open("r", encoding="utf-8") as stream:
        snapshot = json.load(stream)
    with artifact_paths["report"].open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    configuration = manifest["configuration"]
    if configuration.get("horizonsBars") != list(HORIZONS):
        raise ValueError("manifest horizon declaration mismatch")
    config = EvaluationConfig(
        cutoff_ms=int(manifest["createdFromCutoffMs"]),
        bootstrap_samples=int(configuration["bootstrapSamples"]),
        block_size_events=int(configuration["blockSizeEvents"]),
        alpha=float(configuration["alpha"]),
        dedup_bars=int(configuration["dedupBars"]),
        context_keys=tuple(configuration["contextKeys"]),
        random_seed=int(configuration["randomSeed"]),
    )
    dataset = parse_snapshot(snapshot, config.cutoff_ms)
    if manifest.get("scope") != report.get("scope") or manifest.get("scope") != {
        "symbol": SYMBOL,
        "timeframe": dataset.timeframe,
        "cutoffMs": config.cutoff_ms,
    }:
        raise ValueError("manifest scope does not match frozen evidence")
    if manifest.get("sourceLineage") != snapshot.get("lineage"):
        raise ValueError("manifest source lineage does not match frozen snapshot")
    reconstructed_events, reconstructed_metadata = build_causal_technical_events(
        dataset.timeframe,
        dataset.interval_ms,
        _dataset_rows(dataset),
        [dict(candle.context) for candle in dataset.candles],
    )
    stored_derived = [event for event in dataset.events if event.get("module") in DERIVED_MODULES]
    if stored_derived != reconstructed_events:
        raise ValueError("frozen derived technical events do not match causal reconstruction")
    for module in (*DERIVED_MODULES, "contract"):
        if dataset.modules.get(module) != reconstructed_metadata.get(module):
            raise ValueError(f"frozen module metadata mismatch: {module}")
    for event in dataset.events:
        if event.get("module") in DERIVED_MODULES:
            continue
        if event.get("module") not in (None, "causalSmc"):
            raise ValueError("unsupported stored event module")
        _validate_lineage(event.get("lineage"), f"stored event {event.get('eventId')}")
    if (
        str(dataset.lineage.get("source", "")).startswith("postgresql:Klines+")
        and isinstance(dataset.lineage.get("sourceCutoffMs"), int)
    ):
        source_payload = {
            "timeframe": dataset.timeframe,
            "cutoffMs": dataset.lineage["sourceCutoffMs"],
            "candles": snapshot.get("candles"),
            "events": snapshot.get("events"),
            "modules": snapshot.get("modules"),
        }
        if dataset.lineage.get("contentSha256") != sha256_canonical(source_payload):
            raise ValueError("PostgreSQL source lineage content hash mismatch")
    expected_snapshot, expected_rows, expected_report = evaluate(snapshot, config)
    if expected_snapshot != snapshot:
        raise ValueError("snapshot is not in canonical cutoff-filtered form")
    with artifact_paths["ledger"].open("r", encoding="utf-8") as stream:
        for index, expected_row in enumerate(expected_rows):
            line = stream.readline()
            if not line or json.loads(line) != expected_row:
                raise ValueError("ledger does not match causal recomputation from snapshot")
        if stream.readline():
            raise ValueError("ledger does not match causal recomputation from snapshot")
    if expected_report != report:
        raise ValueError("report does not match semantic recomputation from ledger")
    return {
        "valid": True,
        "schema": SCHEMA,
        "manifestSha256": sha256_bytes(manifest_bytes),
        "rows": len(expected_rows),
        "eligible": report["counts"]["eligible"],
        "claimType": report["claimType"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC technical-event descriptive evidence bundle")
    parser.add_argument("--input-json", type=Path)
    parser.add_argument("--from-postgresql", action="store_true")
    parser.add_argument("--timeframe", choices=sorted(TIMEFRAME_MS))
    parser.add_argument("--verify-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/research/evidence/technical-descriptive"))
    parser.add_argument("--cutoff-ms", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--block-size-events", type=int, default=8)
    parser.add_argument("--dedup-bars", type=int, default=6)
    parser.add_argument("--context-keys", default="trend,volatility")
    args = parser.parse_args()
    if args.verify_manifest:
        print(json.dumps(verify_bundle(args.verify_manifest), indent=2))
        return
    if args.cutoff_ms is None:
        parser.error("--cutoff-ms is required when building a bundle")
    if bool(args.input_json) == bool(args.from_postgresql):
        parser.error("choose exactly one of --input-json or --from-postgresql")
    if args.from_postgresql:
        if not args.timeframe:
            parser.error("--timeframe is required with --from-postgresql")
        snapshot = load_postgresql_snapshot(args.timeframe, args.cutoff_ms)
    else:
        snapshot = json.loads(args.input_json.read_text(encoding="utf-8"))
    config = EvaluationConfig(
        cutoff_ms=args.cutoff_ms,
        bootstrap_samples=args.bootstrap_samples,
        block_size_events=args.block_size_events,
        dedup_bars=args.dedup_bars,
        context_keys=tuple(key.strip() for key in args.context_keys.split(",") if key.strip()),
    )
    filtered_snapshot, rows, report = evaluate(snapshot, config)
    paths = write_bundle(args.output_dir, filtered_snapshot, rows, report, config)
    result = verify_bundle(paths["manifest"])
    print(json.dumps({"paths": {key: str(path) for key, path in paths.items()}, "verification": result}, indent=2))


if __name__ == "__main__":
    main()
