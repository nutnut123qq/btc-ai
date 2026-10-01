"""Versioned statistical diagnostics for descriptive technical evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from technical_event_modules import load_contract


STATISTICS_SCHEMA = "btc-technical-evidence-statistics/v1"
STATISTICS_SPEC_PATH = Path(__file__).with_name("contracts") / "technical-evidence-statistical-spec.json"
STATISTICS_OUTPUT_SCHEMA_PATH = Path(__file__).with_name("contracts") / "technical-evidence-statistics.schema.json"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_statistics_spec() -> tuple[dict[str, Any], str]:
    value = json.loads(STATISTICS_SPEC_PATH.read_text(encoding="utf-8"))
    if value.get("specVersion") != STATISTICS_SCHEMA:
        raise ValueError("unsupported technical evidence statistical spec")
    scope = value.get("scope") or {}
    if scope.get("symbol") != "BTCUSDT" or scope.get("timeframes") != ["1h", "4h", "1d"]:
        raise ValueError("statistical spec scope mismatch")
    digest = hashlib.sha256(_canonical(value).encode()).hexdigest()
    return value, digest


def declared_event_families() -> tuple[dict[str, list[str]], str]:
    spec, _ = load_statistics_spec()
    module_contract, module_contract_sha = load_contract()
    families = {
        str(module): [str(event_type) for event_type in value["parameters"]["eventTypes"]]
        for module, value in sorted(module_contract["modules"].items())
    }
    for module, event_types in sorted(spec["declaredEventFamilies"]["external"].items()):
        if module in families:
            raise ValueError(f"duplicate declared event-family module: {module}")
        families[str(module)] = [str(event_type) for event_type in event_types]
    seen: dict[str, str] = {}
    for module, event_types in sorted(families.items()):
        if not event_types or len(set(event_types)) != len(event_types):
            raise ValueError(f"empty or duplicate event types in declared family: {module}")
        for event_type in event_types:
            previous = seen.setdefault(event_type, module)
            if previous != module:
                raise ValueError(f"event type is not module-unique: {event_type}")
    return dict(sorted(families.items())), module_contract_sha


def _median(values: Sequence[float]) -> float | None:
    return None if not values else float(np.median(np.asarray(values, dtype=np.float64)))


def _effect_size(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {
            "count": 0,
            "meanPairedDifference": None,
            "medianPairedDifference": None,
            "pairedStandardizedMeanDifference": None,
            "positiveFraction": None,
            "tieFraction": None,
            "negativeFraction": None,
        }
    mean = float(np.mean(array))
    deviation = float(np.std(array, ddof=1)) if len(array) >= 2 else 0.0
    return {
        "count": len(array),
        "meanPairedDifference": mean,
        "medianPairedDifference": float(np.median(array)),
        "pairedStandardizedMeanDifference": mean / deviation if deviation > 0 else None,
        "positiveFraction": float(np.mean(array > 0)),
        "tieFraction": float(np.mean(array == 0)),
        "negativeFraction": float(np.mean(array < 0)),
    }


def _effective_sample_size(values: Sequence[float], max_lags: int = 20) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    count = len(array)
    if count < 2:
        return {"estimate": float(count), "positiveAutocorrelationLagsUsed": 0, "maxLags": max_lags}
    centered = array - np.mean(array)
    denominator = float(np.dot(centered, centered))
    if denominator == 0:
        return {"estimate": float(count), "positiveAutocorrelationLagsUsed": 0, "maxLags": max_lags}
    positive_sum = 0.0
    used = 0
    for lag in range(1, min(max_lags, count - 1) + 1):
        rho = float(np.dot(centered[:-lag], centered[lag:]) / denominator)
        if not math.isfinite(rho) or rho <= 0:
            break
        positive_sum += rho
        used += 1
    estimate = count / (1.0 + 2.0 * positive_sum)
    return {
        "estimate": max(1.0, min(float(count), float(estimate))),
        "positiveAutocorrelationLagsUsed": used,
        "maxLags": max_lags,
    }


def _sample_diagnostics(rows: Sequence[Mapping[str, Any]], values: Sequence[float], horizon: int) -> dict[str, Any]:
    windows = sorted(
        (int(row["decisionIndex"]) + 1, int(row["decisionIndex"]) + horizon)
        for row in rows
    )
    non_overlapping = 0
    last_end = -1
    for start, end in sorted(windows, key=lambda item: (item[1], item[0])):
        if start > last_end:
            non_overlapping += 1
            last_end = end
    unique_times = {
        int(row["decisionCloseTimeMs"])
        for row in rows
        if isinstance(row.get("decisionCloseTimeMs"), int)
    }
    return {
        "nominalMatchedPairs": len(rows),
        "uniqueDecisionTimes": len(unique_times),
        "maximumGreedyNonOverlappingOutcomeWindows": non_overlapping,
        "observationsExcludedForMaximumNonOverlappingSet": len(rows) - non_overlapping,
        "eventOrderAutocorrelationEffectiveSampleSize": _effective_sample_size(values),
        "independenceClaimed": False,
    }


def _year(row: Mapping[str, Any]) -> str:
    value = row.get("decisionCloseTimeMs")
    if not isinstance(value, int) or value <= 0:
        return "unavailable"
    return str(datetime.fromtimestamp(value / 1000, tz=timezone.utc).year)


def _regime(row: Mapping[str, Any]) -> str:
    context = row.get("context")
    if not isinstance(context, Mapping):
        return "unavailable"
    return f"trend={context.get('trend', 'unavailable')}|volatility={context.get('volatility', 'unavailable')}"


def _stability(
    rows_and_values: Sequence[tuple[Mapping[str, Any], float]],
    pooled_mean: float | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, selector in (("yearUtc", _year), ("regime", _regime)):
        grouped: dict[str, list[float]] = defaultdict(list)
        for row, value in rows_and_values:
            grouped[selector(row)].append(value)
        groups = {
            key: {
                "count": len(values),
                "meanPairedDifference": sum(values) / len(values),
                "medianPairedDifference": _median(values),
                "sign": "positive" if sum(values) > 0 else "negative" if sum(values) < 0 else "zero",
            }
            for key, values in sorted(grouped.items())
        }
        pooled_sign = None if pooled_mean is None else (1 if pooled_mean > 0 else -1 if pooled_mean < 0 else 0)
        agreeing = sum(
            (1 if item["meanPairedDifference"] > 0 else -1 if item["meanPairedDifference"] < 0 else 0) == pooled_sign
            for item in groups.values()
        )
        result[name] = {
            "groups": groups,
            "availableGroupCount": len(groups),
            "pooledSignAgreementFraction": None if not groups else agreeing / len(groups),
            "minimumSampleFilter": None,
        }
    return result


def _benjamini_yekutieli(hypotheses: list[dict[str, Any]], alpha: float) -> None:
    testable = sorted(
        (
            item
            for item in hypotheses
            if isinstance(item.get("rawPValue"), (int, float)) and item.get("status") == "tested"
        ),
        key=lambda item: (float(item["rawPValue"]), item["hypothesisId"]),
    )
    count = len(testable)
    if not count:
        return
    harmonic = sum(1.0 / rank for rank in range(1, count + 1))
    adjusted = [1.0] * count
    running = 1.0
    for offset in range(count - 1, -1, -1):
        rank = offset + 1
        candidate = min(1.0, float(testable[offset]["rawPValue"]) * count * harmonic / rank)
        running = min(running, candidate)
        adjusted[offset] = running
    for item, q_value in zip(testable, adjusted):
        item["adjustedQValue"] = q_value
        item["passesDeclaredFdr"] = q_value <= alpha


def build_statistical_evidence(
    rows: Sequence[Mapping[str, Any]],
    event_type_summaries: Mapping[str, Any],
    *,
    timeframe: str,
    horizons: Sequence[int],
    metrics: Sequence[str],
    block_size_events: int,
    bootstrap_samples: int,
    random_seed: int,
    sensitivity: Mapping[str, Any],
) -> dict[str, Any]:
    spec, spec_sha = load_statistics_spec()
    if (
        timeframe not in spec["scope"]["timeframes"]
        or list(horizons) != spec["scope"]["horizonsBars"]
        or list(metrics) != spec["scope"]["metrics"]
    ):
        raise ValueError("evaluation configuration disagrees with statistical spec")
    minimum_non_overlapping = spec["multipleTesting"].get("minimumNonOverlappingPairs")
    if (
        not isinstance(minimum_non_overlapping, int)
        or isinstance(minimum_non_overlapping, bool)
        or minimum_non_overlapping < 1
    ):
        raise ValueError("statistical spec minimumNonOverlappingPairs is missing or invalid")
    declared_families, module_contract_sha = declared_event_families()
    declared_identities = {
        (module, event_type)
        for module, event_types in declared_families.items()
        for event_type in event_types
    }
    by_identity: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    event_type_modules: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        identity = (str(row.get("module") or "unknown"), str(row.get("eventType") or "unknown"))
        by_identity[identity].append(row)
        event_type_modules[identity[1]].add(identity[0])
    undeclared = sorted(set(by_identity) - declared_identities)
    if undeclared:
        raise ValueError(f"observed statistical event identity is not predeclared: {undeclared}")
    collisions = {key: sorted(value) for key, value in event_type_modules.items() if len(value) > 1}
    if collisions:
        raise ValueError(f"event type is not module-unique for statistical summary reuse: {collisions}")

    hypotheses: list[dict[str, Any]] = []
    for module, event_type in sorted(declared_identities):
        group = by_identity.get((module, event_type), [])
        summary = event_type_summaries.get(event_type, {})
        for horizon in horizons:
            key = str(horizon)
            for metric in metrics:
                matched: list[Mapping[str, Any]] = []
                values: list[float] = []
                for row in group:
                    outcome = row.get("horizons", {}).get(key)
                    control = row.get("control", {})
                    control_outcome = control.get("horizons", {}).get(key) if isinstance(control, Mapping) else None
                    if row.get("status") != "eligible" or outcome is None or control_outcome is None:
                        continue
                    if control.get("status") != "matched" or control.get("outcomeAvailableBeforeEvent") is not True:
                        continue
                    if control.get("contextSignature") != row.get("contextSignature"):
                        raise ValueError("matched null control regime differs from event regime")
                    matched.append(row)
                    values.append(float(outcome[metric]) - float(control_outcome[metric]))
                effect = _effect_size(values)
                paired_summary = (
                    summary.get("horizons", {}).get(key, {}).get("metrics", {}).get(metric, {}).get("matchedControlDifference")
                )
                if isinstance(paired_summary, Mapping) and paired_summary.get("count") != len(values):
                    raise ValueError("paired statistical summary count disagrees with matched-control ledger")
                raw_p = paired_summary.get("centeredNullTwoSidedPValue") if isinstance(paired_summary, Mapping) else None
                if raw_p is not None and (
                    not isinstance(raw_p, (int, float)) or not math.isfinite(float(raw_p)) or not 0 <= float(raw_p) <= 1
                ):
                    raise ValueError("paired statistical summary p-value is invalid")
                diagnostics = _sample_diagnostics(matched, values, horizon)
                hypothesis = {
                    "hypothesisId": f"{module}:{event_type}:{horizon}:{metric}",
                    "module": module,
                    "eventType": event_type,
                    "horizonBars": horizon,
                    "metric": metric,
                    "status": "tested" if len(values) >= 2 and raw_p is not None else "insufficient_or_no_matched_sample",
                    "nullBaseline": "strict-prior exact trend+volatility regime match without replacement within focal module:eventType",
                    "effectSize": effect,
                    "blockBootstrap": {
                        "samples": bootstrap_samples,
                        "blockSizeEvents": min(block_size_events, max(1, len(values) // 2)) if values else None,
                        "meanDifferenceInterval": paired_summary.get("meanBlockBootstrapInterval")
                        if isinstance(paired_summary, Mapping)
                        else None,
                        "centeredTwoSidedPValue": raw_p,
                    },
                    "rawPValue": raw_p,
                    "adjustedQValue": None,
                    "passesDeclaredFdr": None,
                    "sufficientSample": diagnostics["maximumGreedyNonOverlappingOutcomeWindows"] >= minimum_non_overlapping,
                    "minimumNonOverlappingPairs": minimum_non_overlapping,
                    "sampleDiagnostics": diagnostics,
                    "stability": _stability(list(zip(matched, values)), effect["meanPairedDifference"]),
                }
                hypotheses.append(hypothesis)

    alpha = float(spec["multipleTesting"]["declaredQAlpha"])
    _benjamini_yekutieli(hypotheses, alpha)
    # Apply the declared sufficiency gate on top of the BY verdict: a tested
    # hypothesis below the non-overlapping-pair floor keeps its honest
    # rawPValue/adjustedQValue but reports passesDeclaredFdr false. Untested
    # hypotheses were never admitted to the BY family and keep None.
    for item in hypotheses:
        if item["status"] == "tested":
            item["passesDeclaredFdr"] = bool(item["passesDeclaredFdr"]) and item["sufficientSample"]
    tested = [item for item in hypotheses if item["status"] == "tested"]
    return {
        "schema": STATISTICS_SCHEMA,
        "specSha256": spec_sha,
        "scope": {"symbol": "BTCUSDT", "timeframe": timeframe},
        "claimType": "descriptive_only",
        "declaredFamily": {
            "technicalModuleContractDefinitionsSha256": module_contract_sha,
            "moduleEventTypes": declared_families,
            "moduleEventTypeIdentityCount": len(declared_identities),
            "cartesianHypothesisCount": len(declared_identities) * len(horizons) * len(metrics),
            "unknownObservedIdentityPolicy": "fail_closed",
            "zeroEventPolicy": "retain_with_null_inferential_values",
        },
        "nullBaseline": spec["nullBaseline"],
        "dependence": {
            **spec["dependence"],
            "configuredBlockSizeEvents": block_size_events,
            "bootstrapSamples": bootstrap_samples,
            "randomSeed": random_seed,
        },
        "multipleTesting": {
            **spec["multipleTesting"],
            "familySizeAllRetained": len(hypotheses),
            "testableFamilySize": len(tested),
        },
        "sampleDiagnosticsDefinition": spec["sampleDiagnostics"],
        "stabilityDefinition": spec["stability"],
        "sensitivityGrid": {
            "declaredGridSha256": sensitivity.get("preRegisteredGrid", {}).get("declaredGridSha256"),
            "executedVariantIds": sensitivity.get("preRegisteredGrid", {}).get("executedVariantIds", []),
            "allExecutedVariantsRetained": sensitivity.get("preRegisteredGrid", {}).get("allExecutedVariantsRetained"),
            "outcomeDrivenSelectionAllowed": False,
        },
        "retentionPolicy": spec["hypothesisFamily"]["retention"],
        "hypotheses": hypotheses,
    }
