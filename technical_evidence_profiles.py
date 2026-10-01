"""Pre-declared descriptive profiles for causal technical-event evidence.

The profiles in this module are deliberately exhaustive summaries.  They do
not select event families, thresholds, years, regimes, or sensitivity variants
from observed outcomes.  Empty and non-positive result groups remain visible.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Mapping, Sequence


PROFILE_SCHEMA = "btc-technical-evidence-profiles/v1"
BASE_MODULES = (
    "technicalIndicators",
    "candlePatterns",
    "volumeAnomaly",
    "marketRegime",
    "fibonacci",
    "volumeProfile",
)
METRICS = ("forwardReturn", "mfe", "mae")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _year_utc(value: Any) -> str:
    if not isinstance(value, int) or value <= 0:
        return "unavailable"
    return str(datetime.fromtimestamp(value / 1000, tz=timezone.utc).year)


def _regime_key(row: Mapping[str, Any]) -> str:
    context = row.get("context")
    if not isinstance(context, Mapping):
        return "unavailable"
    trend = str(context.get("trend") or "unavailable")
    volatility = str(context.get("volatility") or "unavailable")
    return f"trend={trend}|volatility={volatility}"


def _metric_summary(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    midpoint = len(ordered) // 2
    median = (
        ordered[midpoint]
        if len(ordered) % 2
        else (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
    )
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "median": median,
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def _group_summary(rows: Sequence[Mapping[str, Any]], horizons: Sequence[int]) -> dict[str, Any]:
    eligible = [row for row in rows if row.get("status") == "eligible"]
    exclusions = Counter(
        str(row.get("exclusionReason"))
        for row in rows
        if row.get("exclusionReason")
    )
    horizon_profiles: dict[str, Any] = {}
    for horizon in horizons:
        key = str(horizon)
        realized = [row for row in eligible if row.get("horizons", {}).get(key) is not None]
        returns = [float(row["horizons"][key]["forwardReturn"]) for row in realized]
        horizon_profiles[key] = {
            "eligible": len(eligible),
            "realized": len(realized),
            "unrealized": len(eligible) - len(realized),
            "nonPositiveForwardReturnCount": sum(value <= 0 for value in returns),
            "positiveForwardReturnCount": sum(value > 0 for value in returns),
            "metrics": {
                metric: _metric_summary(
                    [float(row["horizons"][key][metric]) for row in realized]
                )
                for metric in METRICS
            },
        }
    return {
        "stored": len(rows),
        "eligible": len(eligible),
        "excluded": len(rows) - len(eligible),
        "exclusionReasons": dict(sorted(exclusions.items())),
        "horizons": horizon_profiles,
    }


def _dimension(
    rows: Sequence[Mapping[str, Any]],
    key_selector: Any,
    horizons: Sequence[int],
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key_selector(row))].append(row)
    return {key: _group_summary(groups[key], horizons) for key in sorted(groups)}


def _coverage(
    rows: Sequence[Mapping[str, Any]],
    candles: Sequence[Mapping[str, Any]],
    interval_ms: int,
    modules: Mapping[str, Any],
    horizons: Sequence[int],
) -> dict[str, Any]:
    opens = sorted(int(candle["openTimeMs"]) for candle in candles)
    missing_bars = 0
    gap_count = 0
    irregular_spacing_count = 0
    for previous, current in zip(opens, opens[1:]):
        delta = current - previous
        if delta == interval_ms:
            continue
        gap_count += 1
        if delta > interval_ms and delta % interval_ms == 0:
            missing_bars += delta // interval_ms - 1
        else:
            irregular_spacing_count += 1

    module_rows = Counter(str(row.get("module") or "unknown") for row in rows)
    module_coverage = {
        name: {
            "status": value.get("status") if isinstance(value, Mapping) else "unknown",
            "eventRowsInLedger": module_rows.get(name, 0),
            "declaredEventRows": value.get("eventRows") if isinstance(value, Mapping) else None,
        }
        for name, value in sorted(modules.items())
        if name != "contract"
    }
    eligible = [row for row in rows if row.get("status") == "eligible"]
    return {
        "candles": {
            "rows": len(candles),
            "firstOpenTimeMs": opens[0] if opens else None,
            "lastOpenTimeMs": opens[-1] if opens else None,
            "gapCount": gap_count,
            "estimatedMissingBars": missing_bars,
            "irregularSpacingCount": irregular_spacing_count,
            "segmentCount": 0 if not opens else gap_count + 1,
        },
        "events": {
            "rows": len(rows),
            "withoutDecisionTime": sum(row.get("decisionCloseTimeMs") is None for row in rows),
            "withoutUsableContext": sum(_regime_key(row) == "unavailable" for row in rows),
            "withoutKnownLineage": sum(row.get("exclusionReason") == "unknown_event_lineage" for row in rows),
            "unmatchedHistoricalControls": sum(
                row.get("control", {}).get("status") == "unavailable" for row in eligible
            ),
            "horizonMissingness": {
                str(horizon): sum(
                    row.get("horizons", {}).get(str(horizon)) is None for row in eligible
                )
                for horizon in horizons
            },
        },
        "modules": module_coverage,
    }


def _dependence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row.get("status") == "eligible"]
    by_close: dict[int, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in eligible:
        close = row.get("decisionCloseTimeMs")
        if isinstance(close, int):
            by_close[close][str(row.get("module") or "unknown")].append(row)

    module_pairs: Counter[str] = Counter()
    unique_by_module: Counter[str] = Counter()
    distinct_module_count: Counter[str] = Counter()
    base_overlap_by_close: dict[int, set[str]] = {}
    for close, grouped in by_close.items():
        base = sorted(set(grouped).intersection(BASE_MODULES))
        base_overlap_by_close[close] = set(base)
        distinct_module_count[str(len(base))] += 1
        if len(base) == 1:
            unique_by_module[base[0]] += 1
        for left, right in combinations(base, 2):
            module_pairs[f"{left}|{right}"] += 1

    confluence_rows = [row for row in eligible if row.get("module") == "confluence"]
    confluence_support = Counter(
        str(len(base_overlap_by_close.get(int(row["decisionCloseTimeMs"]), set())))
        for row in confluence_rows
        if isinstance(row.get("decisionCloseTimeMs"), int)
    )
    return {
        "unit": "distinct module presence at the same finalized decision close",
        "eligibleDecisionCloses": len(by_close),
        "baseModuleCountPerClose": dict(sorted(distinct_module_count.items(), key=lambda item: int(item[0]))),
        "uniqueDecisionClosesByModule": dict(sorted(unique_by_module.items())),
        "pairwiseSameCloseCounts": dict(sorted(module_pairs.items())),
        "confluence": {
            "eligibleRows": len(confluence_rows),
            "coLocatedBaseModuleCount": dict(sorted(confluence_support.items(), key=lambda item: int(item[0]))),
            "rowsWithFewerThanTwoBaseModules": sum(
                len(base_overlap_by_close.get(int(row["decisionCloseTimeMs"]), set())) < 2
                for row in confluence_rows
                if isinstance(row.get("decisionCloseTimeMs"), int)
            ),
        },
        "independenceClaimed": False,
        "limitation": "same-close and forward-outcome overlap create dependence; counts are descriptive and are not independent trials",
    }


def _negative_result_retention(
    rows: Sequence[Mapping[str, Any]],
    horizons: Sequence[int],
    sensitivity: Mapping[str, Any],
) -> dict[str, Any]:
    max_key = str(max(horizons))
    by_type: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[(
            str(row.get("module") or "unknown"),
            str(row.get("eventType") or "unknown"),
        )].append(row)
    retained_types: dict[str, Any] = {}
    for module, event_type in sorted(by_type):
        group = by_type[(module, event_type)]
        realized = [
            row for row in group
            if row.get("status") == "eligible" and row.get("horizons", {}).get(max_key) is not None
        ]
        values = [float(row["horizons"][max_key]["forwardReturn"]) for row in realized]
        mean = None if not values else sum(values) / len(values)
        retained_types[f"{module}:{event_type}"] = {
            "module": module,
            "eventType": event_type,
            "stored": len(group),
            "eligible": sum(row.get("status") == "eligible" for row in group),
            "realizedAtMaxHorizon": len(values),
            "meanForwardReturnAtMaxHorizon": mean,
            "retentionClassification": (
                "no_realized_sample_retained"
                if mean is None
                else "non_positive_mean_retained"
                if mean <= 0
                else "positive_mean_retained"
            ),
        }

    variants: dict[str, Any] = {}
    for item in sensitivity.get("variants", []):
        horizon = item.get("horizons", {}).get(max_key, {})
        metric = horizon.get("metrics", {}).get("forwardReturn", {}).get("variant") or {}
        mean = metric.get("mean")
        key = f"{item.get('module')}:{item.get('variantId')}"
        variants[key] = {
            "eligible": item.get("eligible"),
            "realizedAtMaxHorizon": item.get("realizedAtMaxHorizon"),
            "meanForwardReturnAtMaxHorizon": mean,
            "retentionClassification": (
                "no_realized_sample_retained"
                if mean is None
                else "non_positive_mean_retained"
                if float(mean) <= 0
                else "positive_mean_retained"
            ),
        }
    return {
        "rule": "retain every declared event type and every pre-declared sensitivity variant regardless of sign, sample size, or availability",
        "thresholdSelectionAllowed": False,
        "eventTypes": retained_types,
        "sensitivityVariants": dict(sorted(variants.items())),
    }


def build_evidence_profiles(
    rows: Sequence[Mapping[str, Any]],
    candles: Sequence[Mapping[str, Any]],
    *,
    timeframe: str,
    interval_ms: int,
    modules: Mapping[str, Any],
    horizons: Sequence[int],
    sensitivity: Mapping[str, Any],
) -> dict[str, Any]:
    """Build exhaustive, deterministic profiles without outcome-driven selection."""
    declared_dimensions = ["yearUtc", "timeframe", "regime"]
    breakdowns = {
        "yearUtc": _dimension(rows, lambda row: _year_utc(row.get("decisionCloseTimeMs")), horizons),
        "timeframe": {timeframe: _group_summary(rows, horizons)},
        "regime": _dimension(rows, _regime_key, horizons),
    }
    module_names = sorted({str(row.get("module") or "unknown") for row in rows} | {
        str(name) for name in modules if name != "contract"
    })
    by_module: dict[str, Any] = {}
    for module in module_names:
        module_rows = [row for row in rows if str(row.get("module") or "unknown") == module]
        module_sensitivity = {
            **dict(sensitivity),
            "variants": [
                item for item in sensitivity.get("variants", [])
                if str(item.get("module")) == module
            ],
        }
        by_module[module] = {
            "breakdowns": {
                "yearUtc": _dimension(module_rows, lambda row: _year_utc(row.get("decisionCloseTimeMs")), horizons),
                "timeframe": {timeframe: _group_summary(module_rows, horizons)},
                "regime": _dimension(module_rows, _regime_key, horizons),
            },
            "coverageAndMissingness": _coverage(
                module_rows,
                candles,
                interval_ms,
                {module: modules.get(module, {})},
                horizons,
            ),
            "negativeResultRetention": _negative_result_retention(
                module_rows,
                horizons,
                module_sensitivity,
            ),
        }

    profiles = {
        "schema": PROFILE_SCHEMA,
        "selectionPolicy": {
            "dimensionsDeclaredBeforeOutcomes": declared_dimensions,
            "minimumSampleFilter": None,
            "outcomeThresholdFilter": None,
            "rankingOrWinnerSelection": False,
        },
        "breakdowns": breakdowns,
        "byModule": by_module,
        "coverageAndMissingness": _coverage(rows, candles, interval_ms, modules, horizons),
        "overlapAndDependence": _dependence(rows),
        "negativeResultRetention": _negative_result_retention(rows, horizons, sensitivity),
    }
    profiles["profileDefinitionsSha256"] = _sha(
        {
            "schema": PROFILE_SCHEMA,
            "dimensions": declared_dimensions,
            "moduleScoping": "byModule",
            "horizons": list(horizons),
            "baseModules": list(BASE_MODULES),
            "metrics": list(METRICS),
        }
    )
    return profiles
