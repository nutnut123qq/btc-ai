"""Frozen causal technical-event reconstruction from finalized BTC candles.

Definitions live in ``technical_module_contract.json``.  The functions here do
not read derived database tables because those tables currently lack the point-
in-time availability and calculation-version lineage required by the evidence
contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import copy
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


CONTRACT_PATH = Path(__file__).with_name("contracts") / "technical-module-contract.json"

_STORED_LAYER_TABLES = {
    "technicalIndicators": "TechnicalIndicators",
    "candlePatterns": "CandlePatterns",
    "volumeAnomaly": "CandleVolumeStats",
    "marketRegime": "MarketRegimes",
    "fibonacci": None,
    "volumeProfile": "VolumeProfileSnapshots",
    "confluence": "ConfluenceSnapshots",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def load_contract() -> tuple[dict[str, Any], str]:
    raw = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    definitions = {key: value for key, value in raw.items() if key != "goldenFixture"}
    definitions_sha = _sha(definitions)
    fixture = raw.get("goldenFixture")
    if not isinstance(fixture, dict) or not isinstance(fixture.get("generator"), dict) or not isinstance(fixture.get("expected"), dict):
        raise ValueError("technical module contract goldenFixture is required")
    expected = fixture["expected"]
    required = {
        "definitionsSha256", "totalEvents", "orderedEventLedgerSha256",
        "crossLanguageSemanticLedgerSha256", "eventRowsByModule",
    }
    if not required.issubset(expected):
        raise ValueError("technical module contract goldenFixture.expected is incomplete")
    declared = expected.get("definitionsSha256")
    if declared != definitions_sha:
        raise ValueError("technical module contract definitionsSha256 does not match golden fixture")
    return raw, definitions_sha


def verify_golden_fixture() -> dict[str, Any]:
    contract, definitions_sha = load_contract()
    fixture = contract["goldenFixture"]
    generator = fixture["generator"]
    expected = fixture["expected"]
    count = int(generator["bars"])
    interval_ms = int(generator["intervalMs"])
    timeframe = str(generator["timeframe"])
    rows: list[dict[str, Any]] = []
    for index in range(count):
        base = 100_000 + 13 * index + ((index % 12) - 6) * 200
        open_price = base
        close = base + ((index % 5) - 2) * 40
        rows.append(
            {
                "OpenTimeMs": index * interval_ms,
                "CloseTimeMs": (index + 1) * interval_ms - 1,
                "Open": float(open_price),
                "High": float(max(open_price, close) + 100 + (index % 3) * 10),
                "Low": float(min(open_price, close) - 110 - (index % 4) * 10),
                "Close": float(close),
                "Volume": float((1_000 + (index % 7) * 100) * (4 if index % 29 == 0 else 1)),
            }
        )
    contexts = sensitivity_contexts(rows, interval_ms, 1.5, 0.67)
    events, metadata = build_causal_technical_events(timeframe, interval_ms, rows, contexts)
    ordered_payload = [
        {
            "eventId": event["eventId"],
            "module": event["module"],
            "eventType": event["eventType"],
            "availableTimeMs": event["availableTimeMs"],
            "sources": event["sourceCandleOpenTimesMs"],
        }
        for event in events
    ]
    semantic_payload = sorted(
        (
            {
                "availableTimeMs": event["availableTimeMs"],
                "eventType": event["eventType"],
                "module": event["module"],
            }
            for event in events
        ),
        key=lambda item: (item["availableTimeMs"], item["module"], item["eventType"]),
    )
    actual = {
        "definitionsSha256": definitions_sha,
        "totalEvents": len(events),
        "orderedEventLedgerSha256": _sha(ordered_payload),
        "crossLanguageSemanticLedgerSha256": _sha(semantic_payload),
        "eventRowsByModule": {
            name: value["eventRows"]
            for name, value in metadata.items()
            if isinstance(value, dict) and "eventRows" in value
        },
    }
    if actual != expected:
        raise ValueError("technical module contract golden semantic self-check failed")
    return actual


def _module_metadata(name: str, spec: Mapping[str, Any], contract_sha: str, event_rows: int, basis: str) -> dict[str, Any]:
    table = _STORED_LAYER_TABLES[name]
    if table is None:
        stored_audit = {
            "status": "unavailable",
            "table": None,
            "reason": "no persisted derived table with explicit point-in-time availability and calculation version",
        }
    else:
        stored_audit = {
            "status": "unavailable",
            "table": table,
            "reason": "persisted derived table is not used because explicit point-in-time availability and calculation-version lineage are not both contractually guaranteed",
        }
    return {
        "status": "evaluable",
        "availabilityBasis": basis,
        "calculationVersion": spec["calculationVersion"],
        "parameters": spec["parameters"],
        "sensitivity": {
            "status": "declared_parameter_grid_not_result_selected",
            "variants": spec["sensitivity"],
            "interpretation": "disclosed robustness grid; baseline parameters are fixed before outcomes and no variant is promoted by observed return",
        },
        "eventRows": event_rows,
        "contractDefinitionsSha256": contract_sha,
        "storedDerivedTableRowsUsed": False,
        "storedLayerAudit": stored_audit,
    }


def _segments(rows: Sequence[Mapping[str, Any]], interval_ms: int) -> list[tuple[int, int]]:
    if not rows:
        return []
    result: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(rows)):
        if int(rows[index]["OpenTimeMs"]) - int(rows[index - 1]["OpenTimeMs"]) != interval_ms:
            result.append((start, index))
            start = index
    result.append((start, len(rows)))
    return result


def sensitivity_contexts(
    rows: Sequence[Mapping[str, Any]], interval_ms: int, high_multiplier: float, low_multiplier: float
) -> list[dict[str, str]]:
    """Contract-exact contexts for regime sensitivity variants."""
    result: list[dict[str, str]] = []
    segment_start = 0
    true_range_pct: list[float] = []
    for index, row in enumerate(rows):
        if index and int(row["OpenTimeMs"]) - int(rows[index - 1]["OpenTimeMs"]) != interval_ms:
            segment_start = index
        close = float(row["Close"])
        high = float(row["High"])
        low = float(row["Low"])
        if index == segment_start:
            true_range = high - low
        else:
            previous_close = float(rows[index - 1]["Close"])
            true_range = max(high - low, abs(high - previous_close), abs(low - previous_close))
        true_range_pct.append(true_range / close)
        change_start = max(segment_start, index - 5)
        changes = [
            float(rows[position]["Close"]) - float(rows[position - 1]["Close"])
            for position in range(change_start + 1, index + 1)
        ]
        up = sum(change > 0 for change in changes)
        down = sum(change < 0 for change in changes)
        trend = "sideways"
        if len(changes) >= 3 and up >= 4 and down <= 1:
            trend = "up"
        elif len(changes) >= 3 and down >= 4 and up <= 1:
            trend = "down"
        history = true_range_pct[max(segment_start, index - 20) : index]
        volatility = "insufficient_history"
        if len(history) >= 10:
            median = statistics.median(history)
            if median > 0 and true_range_pct[index] >= high_multiplier * median:
                volatility = "high"
            elif median > 0 and true_range_pct[index] <= low_multiplier * median:
                volatility = "low"
            else:
                volatility = "normal"
        result.append({"trend": trend, "volatility": volatility})
    return result


def _event(
    module: str,
    event_type: str,
    decision_index: int,
    source_indices: Sequence[int],
    rows: Sequence[Mapping[str, Any]],
    contexts: Sequence[Mapping[str, str]],
    version: str,
    contract_sha: str,
    *,
    formed_index: int | None = None,
    parameters: Mapping[str, Any] | None = None,
    lifecycle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    unique_sources = sorted(set(int(index) for index in source_indices))
    source_payload = [
        {
            "openTimeMs": int(rows[index]["OpenTimeMs"]),
            "closeTimeMs": int(rows[index]["CloseTimeMs"]),
            "open": float(rows[index]["Open"]),
            "high": float(rows[index]["High"]),
            "low": float(rows[index]["Low"]),
            "close": float(rows[index]["Close"]),
            "volume": float(rows[index].get("Volume") or 0.0),
        }
        for index in unique_sources
    ]
    core = {
        "module": module,
        "eventType": event_type,
        "availableTimeMs": int(rows[decision_index]["CloseTimeMs"]),
        "source": source_payload,
        "version": version,
        "contractSha256": contract_sha,
        "parameters": dict(parameters or {}),
    }
    digest = _sha(core)
    result = {
        "eventId": f"{module}-{digest[:32]}",
        "module": module,
        "eventType": event_type,
        "formedTimeMs": int(rows[formed_index if formed_index is not None else decision_index]["CloseTimeMs"]),
        "confirmedTimeMs": int(rows[decision_index]["CloseTimeMs"]),
        "availableTimeMs": int(rows[decision_index]["CloseTimeMs"]),
        "sourceCandleOpenTimesMs": [int(rows[index]["OpenTimeMs"]) for index in unique_sources],
        "context": dict(contexts[decision_index]),
        "lineage": {
            "source": f"finalized-klines:{module}",
            "sourceVersion": version,
            "contentSha256": digest,
        },
    }
    if lifecycle:
        result["lifecycle"] = dict(lifecycle)
    return result


def _ema(values: Sequence[float], period: int) -> float:
    alpha = 2.0 / (period + 1.0)
    value = float(values[0])
    for item in values[1:]:
        value = alpha * float(item) + (1.0 - alpha) * value
    return value


def _rsi(values: Sequence[float], period: int) -> float | None:
    if len(values) < period + 1:
        return None
    changes = [float(values[index]) - float(values[index - 1]) for index in range(len(values) - period, len(values))]
    gain = sum(max(change, 0.0) for change in changes) / period
    loss = sum(max(-change, 0.0) for change in changes) / period
    if loss == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + gain / loss)


def _indicator_events(
    rows: Sequence[Mapping[str, Any]], contexts: Sequence[Mapping[str, str]], interval_ms: int,
    spec: Mapping[str, Any], contract_sha: str,
) -> list[dict[str, Any]]:
    params = spec["parameters"]
    version = str(spec["calculationVersion"])
    window = int(params["emaSourceBars"])
    rsi_period = int(params["rsiPeriod"])
    sma_period = int(params["smaPeriod"])
    events: list[dict[str, Any]] = []
    for start, end in _segments(rows, interval_ms):
        closes = [float(row["Close"]) for row in rows[start:end]]
        for local in range(max(window, sma_period, rsi_period + 2), len(closes)):
            index = start + local
            ema_source_start = index - window
            # Use only the finite windows declared by the contract.  Slicing
            # closes[:local] and closes[:local+1] first made a full-history run
            # quadratic even though those prefixes were immediately trimmed.
            ema_previous = closes[local - window : local]
            ema_current = closes[local - window + 1 : local + 1]
            fast_prev = _ema(ema_previous, int(params["emaFast"]))
            slow_prev = _ema(ema_previous, int(params["emaSlow"]))
            fast_now = _ema(ema_current, int(params["emaFast"]))
            slow_now = _ema(ema_current, int(params["emaSlow"]))
            if fast_prev <= slow_prev and fast_now > slow_now:
                events.append(_event("technicalIndicators", "EMA_BULL_CROSS", index, range(ema_source_start, index + 1), rows, contexts, version, contract_sha, parameters=params))
            elif fast_prev >= slow_prev and fast_now < slow_now:
                events.append(_event("technicalIndicators", "EMA_BEAR_CROSS", index, range(ema_source_start, index + 1), rows, contexts, version, contract_sha, parameters=params))
            rsi_prev = _rsi(closes[local - rsi_period - 1 : local], rsi_period)
            rsi_now = _rsi(closes[local - rsi_period : local + 1], rsi_period)
            lower, upper = float(params["rsiLower"]), float(params["rsiUpper"])
            if rsi_prev is not None and rsi_now is not None:
                label = None
                if rsi_prev >= lower and rsi_now < lower:
                    label = "RSI_ENTER_OVERSOLD"
                elif rsi_prev <= lower and rsi_now > lower:
                    label = "RSI_EXIT_OVERSOLD"
                elif rsi_prev <= upper and rsi_now > upper:
                    label = "RSI_ENTER_OVERBOUGHT"
                elif rsi_prev >= upper and rsi_now < upper:
                    label = "RSI_EXIT_OVERBOUGHT"
                if label:
                    events.append(_event("technicalIndicators", label, index, range(index - rsi_period - 1, index + 1), rows, contexts, version, contract_sha, parameters=params))
            sma_prev = sum(closes[local - sma_period : local]) / sma_period
            sma_now = sum(closes[local - sma_period + 1 : local + 1]) / sma_period
            if closes[local - 1] <= sma_prev and closes[local] > sma_now:
                events.append(_event("technicalIndicators", "CLOSE_ABOVE_SMA50", index, range(index - sma_period, index + 1), rows, contexts, version, contract_sha, parameters=params))
            elif closes[local - 1] >= sma_prev and closes[local] < sma_now:
                events.append(_event("technicalIndicators", "CLOSE_BELOW_SMA50", index, range(index - sma_period, index + 1), rows, contexts, version, contract_sha, parameters=params))
    return events


def _pattern_events(
    rows: Sequence[Mapping[str, Any]], contexts: Sequence[Mapping[str, str]], interval_ms: int,
    spec: Mapping[str, Any], contract_sha: str,
) -> list[dict[str, Any]]:
    params = spec["parameters"]
    version = str(spec["calculationVersion"])
    events: list[dict[str, Any]] = []
    for start, end in _segments(rows, interval_ms):
        for index in range(start, end):
            row = rows[index]
            open_price, high, low, close = map(float, (row["Open"], row["High"], row["Low"], row["Close"]))
            body = abs(close - open_price)
            span = high - low
            if span <= 0:
                continue
            labels: list[tuple[str, int]] = []
            if body / span <= float(params["dojiMaxBodyToRange"]):
                labels.append(("DOJI", index))
            safe_body = max(body, span * 1e-6)
            upper = high - max(open_price, close)
            lower = min(open_price, close) - low
            if lower >= float(params["longWickToBody"]) * safe_body and upper <= float(params["shortOppositeWickToBody"]) * safe_body:
                labels.append(("HAMMER_SHAPE", index))
            if upper >= float(params["longWickToBody"]) * safe_body and lower <= float(params["shortOppositeWickToBody"]) * safe_body:
                labels.append(("SHOOTING_STAR_SHAPE", index))
            if index > start:
                previous = rows[index - 1]
                po, pc = float(previous["Open"]), float(previous["Close"])
                if pc < po and close > open_price and open_price <= pc and close >= po:
                    labels.append(("BULLISH_ENGULFING", index - 1))
                if pc > po and close < open_price and open_price >= pc and close <= po:
                    labels.append(("BEARISH_ENGULFING", index - 1))
            for label, source_start in labels:
                events.append(_event("candlePatterns", label, index, range(source_start, index + 1), rows, contexts, version, contract_sha, parameters=params, formed_index=source_start))
    return events


def _volume_events(
    rows: Sequence[Mapping[str, Any]], contexts: Sequence[Mapping[str, str]], interval_ms: int,
    spec: Mapping[str, Any], contract_sha: str,
) -> list[dict[str, Any]]:
    params = spec["parameters"]
    version = str(spec["calculationVersion"])
    prior = int(params["priorBars"])
    events: list[dict[str, Any]] = []
    for start, end in _segments(rows, interval_ms):
        for index in range(start + prior, end):
            history = [float(rows[position].get("Volume") or 0.0) for position in range(index - prior, index)]
            baseline = sum(history) / prior
            current = float(rows[index].get("Volume") or 0.0)
            if baseline <= 0 or current < 0:
                continue
            ratio = current / baseline
            for threshold in params["thresholds"]:
                if ratio >= float(threshold):
                    label = f"VOLUME_ANOMALY_{str(threshold).replace('.', '_')}X"
                    events.append(_event("volumeAnomaly", label, index, range(index - prior, index + 1), rows, contexts, version, contract_sha, parameters=params))
    return events


def _regime_events(
    rows: Sequence[Mapping[str, Any]], contexts: Sequence[Mapping[str, str]], interval_ms: int,
    spec: Mapping[str, Any], contract_sha: str,
) -> list[dict[str, Any]]:
    version = str(spec["calculationVersion"])
    params = spec["parameters"]
    events: list[dict[str, Any]] = []
    for start, end in _segments(rows, interval_ms):
        for index in range(start + int(params["volatilityMinimumPriorBars"]) + 1, end):
            if contexts[index] == contexts[index - 1]:
                continue
            trend_token = {
                "up": "BULL",
                "down": "BEAR",
                "sideways": "SIDEWAYS",
            }[contexts[index]["trend"]]
            label = f"REGIME_{trend_token}_{contexts[index]['volatility'].upper()}"
            source_start = max(start, index - int(params["volatilityPriorBars"]) - 1)
            events.append(_event("marketRegime", label, index, range(source_start, index + 1), rows, contexts, version, contract_sha, parameters=params))
    return events


def _fibonacci_events(
    rows: Sequence[Mapping[str, Any]], contexts: Sequence[Mapping[str, str]], interval_ms: int,
    spec: Mapping[str, Any], contract_sha: str,
) -> list[dict[str, Any]]:
    params = spec["parameters"]
    version = str(spec["calculationVersion"])
    radius = int(params["pivotRadiusBars"])
    events: list[dict[str, Any]] = []
    for start, end in _segments(rows, interval_ms):
        last_pivot: tuple[str, int, float] | None = None
        for center in range(start + radius, end - radius):
            high = float(rows[center]["High"])
            low = float(rows[center]["Low"])
            is_high = all(high > float(rows[index]["High"]) for index in range(center - radius, center + radius + 1) if index != center)
            is_low = all(low < float(rows[index]["Low"]) for index in range(center - radius, center + radius + 1) if index != center)
            current = ("HIGH", center, high) if is_high else ("LOW", center, low) if is_low else None
            if current is None:
                continue
            decision = center + radius
            if last_pivot and last_pivot[0] != current[0]:
                direction = "BULL" if last_pivot[0] == "LOW" else "BEAR"
                sources = list(range(max(start, last_pivot[1] - radius), center + radius + 1))
                events.append(_event("fibonacci", f"FIBONACCI_LEG_{direction}", decision, sources, rows, contexts, version, contract_sha, parameters=params, formed_index=center))
            last_pivot = current
    return events


def _profile(rows: Sequence[Mapping[str, Any]], start: int, end: int, bins: int, fraction: float) -> tuple[float, float, float] | None:
    low = min(float(rows[index]["Low"]) for index in range(start, end))
    high = max(float(rows[index]["High"]) for index in range(start, end))
    if high <= low:
        return None
    weights = [0.0] * bins
    width = (high - low) / bins
    for index in range(start, end):
        typical = (float(rows[index]["High"]) + float(rows[index]["Low"]) + float(rows[index]["Close"])) / 3.0
        slot = min(bins - 1, max(0, int((typical - low) / width)))
        weights[slot] += max(0.0, float(rows[index].get("Volume") or 0.0))
    total = sum(weights)
    if total <= 0:
        return None
    poc = max(range(bins), key=lambda slot: weights[slot])
    selected: set[int] = set()
    cumulative = 0.0
    for slot in sorted(range(bins), key=lambda item: (-weights[item], item)):
        selected.add(slot)
        cumulative += weights[slot]
        if cumulative >= fraction * total:
            break
    midpoint = lambda slot: low + (slot + 0.5) * width
    return midpoint(poc), midpoint(max(selected)), midpoint(min(selected))


def _volume_profile_events(
    rows: Sequence[Mapping[str, Any]], contexts: Sequence[Mapping[str, str]], interval_ms: int,
    spec: Mapping[str, Any], contract_sha: str,
) -> list[dict[str, Any]]:
    params = spec["parameters"]
    version = str(spec["calculationVersion"])
    window, bins, fraction = int(params["windowBars"]), int(params["bins"]), float(params["valueAreaFraction"])
    events: list[dict[str, Any]] = []
    for start, end in _segments(rows, interval_ms):
        previous_profile = None
        for index in range(start + window - 1, end):
            current_profile = _profile(rows, index - window + 1, index + 1, bins, fraction)
            if current_profile is None:
                previous_profile = None
                continue
            if previous_profile is not None:
                previous_close = float(rows[index - 1]["Close"])
                current_close = float(rows[index]["Close"])
                for label, previous_level, current_level in zip(("POC", "VAH", "VAL"), previous_profile, current_profile):
                    event_type = None
                    if previous_close < previous_level and current_close >= current_level:
                        event_type = f"VOLUME_PROFILE_CLOSE_ABOVE_{label}"
                    elif previous_close > previous_level and current_close <= current_level:
                        event_type = f"VOLUME_PROFILE_CLOSE_BELOW_{label}"
                    if event_type:
                        events.append(_event("volumeProfile", event_type, index, range(index - window, index + 1), rows, contexts, version, contract_sha, parameters=params))
            previous_profile = current_profile
    return events


def _direction(event_type: str) -> int:
    upper = event_type.upper()
    bullish = ("BULL", "ABOVE", "EXIT_OVERSOLD", "ENTER_OVERSOLD")
    bearish = ("BEAR", "BELOW", "EXIT_OVERBOUGHT", "ENTER_OVERBOUGHT")
    if any(token in upper for token in bullish):
        return 1
    if any(token in upper for token in bearish):
        return -1
    return 0


def _confluence_events(
    rows: Sequence[Mapping[str, Any]], contexts: Sequence[Mapping[str, str]], base_events: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any], contract_sha: str,
) -> list[dict[str, Any]]:
    params = spec["parameters"]
    version = str(spec["calculationVersion"])
    minimum = int(params["minimumAlignedModules"])
    allowed = set(params["allowedInputs"])
    by_time: dict[int, list[Mapping[str, Any]]] = {}
    for event in base_events:
        if event.get("module") in allowed and _direction(str(event["eventType"])):
            by_time.setdefault(int(event["availableTimeMs"]), []).append(event)
    close_index = {int(row["CloseTimeMs"]): index for index, row in enumerate(rows)}
    result: list[dict[str, Any]] = []
    for available, events in sorted(by_time.items()):
        index = close_index.get(available)
        if index is None:
            continue
        module_directions: dict[str, set[int]] = {}
        for event in events:
            module_directions.setdefault(str(event["module"]), set()).add(_direction(str(event["eventType"])))
        for direction, label in ((1, "BULL"), (-1, "BEAR")):
            modules = sorted(
                module for module, directions in module_directions.items() if directions == {direction}
            )
            if len(modules) < minimum:
                continue
            aligned = [
                event for event in events
                if str(event["module"]) in modules and _direction(str(event["eventType"])) == direction
            ]
            opens = {int(value) for event in aligned for value in event["sourceCandleOpenTimesMs"]}
            open_index = {int(row["OpenTimeMs"]): position for position, row in enumerate(rows)}
            sources = [open_index[value] for value in opens if value in open_index]
            result.append(_event("confluence", f"CONFLUENCE_{label}_{minimum}PLUS", index, sources, rows, contexts, version, contract_sha, parameters={**params, "alignedModules": modules}))
    return result


def build_sensitivity_variants(
    timeframe: str,
    interval_ms: int,
    rows: Sequence[Mapping[str, Any]],
    contexts: Sequence[Mapping[str, str]],
    baseline_events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Execute every declared one-axis-at-a-time sensitivity variant.

    The returned events are ephemeral inputs to the descriptive sensitivity
    audit.  They are never mixed into the baseline declared family and no
    outcome-dependent variant selection is performed.
    """
    contract, contract_sha = load_contract()
    specs = contract["modules"]
    result: list[dict[str, Any]] = []

    def run(
        module: str,
        variant_id: str,
        updates: Mapping[str, Any],
        builder: Any,
        variant_contexts: Sequence[Mapping[str, str]] = contexts,
    ) -> None:
        spec = copy.deepcopy(specs[module])
        spec["parameters"].update(updates)
        events = builder(rows, variant_contexts, interval_ms, spec, contract_sha)
        result.append(
            {
                "module": module,
                "variantId": variant_id,
                "parameters": {key: spec["parameters"][key] for key in sorted(updates)},
                "events": events,
            }
        )

    indicator = specs["technicalIndicators"]
    for lower, upper in indicator["sensitivity"]["rsiBands"]:
        run(
            "technicalIndicators",
            f"rsiBands={lower:g}_{upper:g}",
            {"rsiLower": float(lower), "rsiUpper": float(upper)},
            _indicator_events,
        )
    for value in indicator["sensitivity"]["emaSourceBars"]:
        run(
            "technicalIndicators",
            f"emaSourceBars={int(value)}",
            {"emaSourceBars": int(value)},
            _indicator_events,
        )

    pattern = specs["candlePatterns"]
    for value in pattern["sensitivity"]["dojiMaxBodyToRange"]:
        run(
            "candlePatterns",
            f"dojiMaxBodyToRange={float(value):g}",
            {"dojiMaxBodyToRange": float(value)},
            _pattern_events,
        )
    for value in pattern["sensitivity"]["longWickToBody"]:
        run(
            "candlePatterns",
            f"longWickToBody={float(value):g}",
            {"longWickToBody": float(value)},
            _pattern_events,
        )

    volume = specs["volumeAnomaly"]
    for value in volume["sensitivity"]["thresholds"]:
        run(
            "volumeAnomaly",
            f"threshold={float(value):g}",
            {"thresholds": [float(value)]},
            _volume_events,
        )

    regime = specs["marketRegime"]
    for high, low in regime["sensitivity"]["volatilityMultipliers"]:
        variant_context = sensitivity_contexts(rows, interval_ms, float(high), float(low))
        run(
            "marketRegime",
            f"volatilityMultipliers={float(high):g}_{float(low):g}",
            {"highVolatilityMultiplier": float(high), "lowVolatilityMultiplier": float(low)},
            _regime_events,
            variant_context,
        )

    fibonacci = specs["fibonacci"]
    for value in fibonacci["sensitivity"]["pivotRadiusBars"]:
        run(
            "fibonacci",
            f"pivotRadiusBars={int(value)}",
            {"pivotRadiusBars": int(value)},
            _fibonacci_events,
        )

    profile = specs["volumeProfile"]
    for value in profile["sensitivity"]["windowBars"]:
        run(
            "volumeProfile",
            f"windowBars={int(value)}",
            {"windowBars": int(value)},
            _volume_profile_events,
        )
    for value in profile["sensitivity"]["bins"]:
        run(
            "volumeProfile",
            f"bins={int(value)}",
            {"bins": int(value)},
            _volume_profile_events,
        )

    confluence = specs["confluence"]
    for value in confluence["sensitivity"]["minimumAlignedModules"]:
        spec = copy.deepcopy(confluence)
        spec["parameters"]["minimumAlignedModules"] = int(value)
        events = _confluence_events(rows, contexts, baseline_events, spec, contract_sha)
        result.append(
            {
                "module": "confluence",
                "variantId": f"minimumAlignedModules={int(value)}",
                "parameters": {"minimumAlignedModules": int(value)},
                "events": events,
            }
        )
    return result


def build_causal_technical_events(
    timeframe: str,
    interval_ms: int,
    rows: Sequence[Mapping[str, Any]],
    contexts: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(rows) != len(contexts):
        raise ValueError("rows and contexts must align")
    contract, contract_sha = load_contract()
    specs = contract["modules"]
    builders = (
        ("technicalIndicators", _indicator_events),
        ("candlePatterns", _pattern_events),
        ("volumeAnomaly", _volume_events),
        ("marketRegime", _regime_events),
        ("fibonacci", _fibonacci_events),
        ("volumeProfile", _volume_profile_events),
    )
    all_events: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    for name, builder in builders:
        events = builder(rows, contexts, interval_ms, specs[name], contract_sha)
        all_events.extend(events)
        metadata[name] = _module_metadata(
            name,
            specs[name],
            contract_sha,
            len(events),
            "reconstructed from finalized contiguous Klines; available at decision candle close",
        )
    confluence = _confluence_events(rows, contexts, all_events, specs["confluence"], contract_sha)
    all_events.extend(confluence)
    metadata["confluence"] = _module_metadata(
        "confluence",
        specs["confluence"],
        contract_sha,
        len(confluence),
        "same-close votes from causal reconstructed technical modules only",
    )
    metadata["contract"] = {
        "contractVersion": contract["contractVersion"],
        "definitionsSha256": contract_sha,
        "rawFileSha256": hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
        "timeframe": timeframe,
    }
    def event_order(event: Mapping[str, Any]) -> tuple[int, str, int, str]:
        priority = 0
        if event.get("module") == "confluence":
            priority = 0 if "_BULL_" in str(event["eventType"]) else 1
        return int(event["availableTimeMs"]), str(event["module"]), priority, str(event["eventId"])

    return sorted(all_events, key=event_order), metadata
