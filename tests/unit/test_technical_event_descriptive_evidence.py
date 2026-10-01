import json
import tracemalloc
from copy import deepcopy

import pytest
import math
import numpy as np
import technical_event_descriptive_evidence as evidence_module

from technical_event_descriptive_evidence import (
    EvaluationConfig,
    TIMEFRAME_MS,
    build_postgresql_snapshot_from_rows,
    evaluate,
    parse_snapshot,
    sha256_canonical,
    verify_bundle,
    _write_content_addressed,
    _canonical_chunks,
    write_bundle,
    _context_signature,
    _evaluate_event_rows,
    _horizon_metrics,
    _level_hit,
    _moving_block_ci,
    _select_controls,
    _time_to_level,
    HORIZONS,
)


HASH = "a" * 64


def _canonical_smc_row(
    event_id,
    event_type,
    origin,
    available,
    source_times,
    *,
    reference=None,
    high=None,
    low=None,
    state=None,
    mitigated_at=None,
):
    version = "smc-causal-v2"
    decision_state = "active" if event_type.startswith("FVG_") else "confirmed"
    evidence = {
        "eventId": event_id,
        "eventType": event_type,
        "description": "fixture",
        "originTimeMs": origin,
        "availableTimeMs": available,
        "referenceTimeMs": reference,
        "price": 101.0,
        "highPrice": high,
        "lowPrice": low,
        "calculationVersion": version,
        "stateAtAsOf": decision_state,
        "mitigatedAtMs": None,
        "invalidatedAtMs": None,
        "mitigationRule": None,
        "invalidationRule": None,
        "sourceCandles": [
            {
                "role": "decision-source",
                "openTimeMs": value,
                "closeTimeMs": value + TIMEFRAME_MS["4h"] - 1,
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 1.0,
            }
            for value in source_times
        ],
        "detectionConditions": [],
        "limitations": [],
    }
    evidence_json = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
    return {
        "EventId": event_id,
        "EventType": event_type,
        "OriginTimeMs": origin,
        "AvailableTimeMs": available,
        "ReferenceTimeMs": reference,
        "Price": 101.0,
        "HighPrice": high,
        "LowPrice": low,
        "State": state or decision_state,
        "MitigatedAtMs": mitigated_at,
        "CalculationVersion": version,
        "DecisionSourceCandleCount": len(source_times),
        "DecisionSourceOpenTimeMsJson": json.dumps(source_times, separators=(",", ":")),
        "DecisionEvidenceJson": evidence_json,
        "DecisionEvidenceSha256": __import__("hashlib").sha256(evidence_json.encode()).hexdigest(),
    }


def _complete_checkpoint(klines, status="complete", *, start=None, last=None):
    return {
        "CalculationVersion": "smc-causal-v2",
        "CoverageStartOpenTimeMs": klines[0]["OpenTimeMs"] if start is None else start,
        "LastProcessedOpenTimeMs": klines[-1]["OpenTimeMs"] if last is None else last,
        "ProcessedCandleCount": len(klines),
        "MaterializedEventCount": 1,
        "Status": status,
        "LastError": None,
    }


def test_content_addressed_writer_streams_large_json_with_bounded_extra_memory(tmp_path):
    payload = {"rows": [f"{index:06d}-" + "x" * 250 for index in range(20_000)]}
    tracemalloc.start()
    from itertools import chain
    path, digest, size = _write_content_addressed(
        tmp_path,
        "snapshot.json",
        chain(_canonical_chunks(payload), (b"\n",)),
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert size > 5_000_000
    assert peak < 8 * 1024 * 1024
    assert path.name == f"{digest}.snapshot.json"


def _legacy_moving_block_ci(values, config, seed_offset=0):
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2 or not np.isfinite(array).all():
        return None
    block = min(config.block_size_events, max(1, len(array) // 2))
    blocks = math.ceil(len(array) / block)
    rng = np.random.default_rng(config.random_seed + seed_offset)
    estimates = np.empty(config.bootstrap_samples, dtype=np.float64)
    for sample in range(config.bootstrap_samples):
        picked = []
        for _ in range(blocks):
            start = int(rng.integers(0, len(array) - block + 1))
            picked.extend(array[start : start + block])
        estimates[sample] = float(np.mean(picked[: len(array)]))
    return {
        "lower": float(np.quantile(estimates, config.alpha / 2)),
        "upper": float(np.quantile(estimates, 1 - config.alpha / 2)),
    }


def test_vectorized_block_bootstrap_matches_legacy_sampling_and_is_deterministic():
    values = [0.1, -0.2, 0.33, 0.04, -0.05, 0.16, 0.27, -0.18, 0.09, 0.3, -0.11]
    config = EvaluationConfig(cutoff_ms=1, bootstrap_samples=257, block_size_events=4, random_seed=73)
    expected = _legacy_moving_block_ci(values, config, 9)
    actual = _moving_block_ci(values, config, 9)
    assert actual == _moving_block_ci(values, config, 9)
    assert actual["lower"] == pytest.approx(expected["lower"], abs=1e-15)
    assert actual["upper"] == pytest.approx(expected["upper"], abs=1e-15)


def _snapshot(timeframe: str = "4h", count: int = 32):
    interval = TIMEFRAME_MS[timeframe]
    candles = []
    price = 100.0
    for index in range(count):
        open_ms = index * interval
        price *= 1.01 if index % 4 < 2 else 0.995
        candles.append(
            {
                "openTimeMs": open_ms,
                "closeTimeMs": open_ms + interval - 1,
                "open": price * 0.998,
                "high": price * 1.02,
                "low": price * 0.98,
                "close": price,
                "availableTimeMs": open_ms + interval - 1,
                "finalized": True,
                "context": {
                    "trend": "up" if index % 2 == 0 else "down",
                    "volatility": "normal",
                },
            }
        )

    def event(event_id: str, index: int, event_type: str = "BOS_BULL", lifecycle=True):
        result = {
            "eventId": event_id,
            "module": "causalSmc",
            "eventType": event_type,
            "formedTimeMs": candles[index - 1]["closeTimeMs"],
            "confirmedTimeMs": candles[index]["closeTimeMs"],
            "availableTimeMs": candles[index]["closeTimeMs"],
            "sourceCandleOpenTimesMs": [candles[index - 1]["openTimeMs"], candles[index]["openTimeMs"]],
            "context": candles[index]["context"],
            "lineage": {"source": "causal-smc", "sourceVersion": "v3", "contentSha256": HASH},
        }
        if lifecycle:
            result["lifecycle"] = {
                "semanticsVersion": "smc-zone-v1",
                "touchLevel": {"operator": "at_or_above", "price": candles[index + 1]["high"] - 0.001},
                "invalidationLevel": {"operator": "at_or_below", "price": candles[index + 2]["low"] + 0.001},
            }
        return result

    return {
        "symbol": "BTCUSDT",
        "timeframe": timeframe,
        "lineage": {"source": "fixture", "sourceVersion": "v1", "contentSha256": HASH},
        "candles": candles,
        "events": [event("e1", 12), event("e-overlap", 14), event("e2", 22, "FVG_BULL", lifecycle=False)],
    }


def _causal_snapshot(timeframe: str = "4h"):
    from tests.unit.test_technical_event_modules import golden_rows

    rows = golden_rows(240, timeframe)
    return build_postgresql_snapshot_from_rows(
        timeframe, rows[-1]["CloseTimeMs"], rows, [], smc_columns=[]
    )


@pytest.mark.parametrize("timeframe", ["1h", "4h", "1d"])
def test_supported_timeframes_emit_fixed_horizons_and_elapsed_time(timeframe):
    snapshot = _snapshot(timeframe)
    cutoff = snapshot["candles"][-1]["closeTimeMs"]
    _, rows, report = evaluate(snapshot, EvaluationConfig(cutoff_ms=cutoff, bootstrap_samples=20))
    e1 = next(row for row in rows if row["eventId"] == "e1")
    assert e1["status"] == "eligible"
    assert set(e1["horizons"]) == {"1", "3", "6"}
    assert e1["horizons"]["6"]["elapsedMs"] == 6 * TIMEFRAME_MS[timeframe]
    assert report["claimType"] == "descriptive_technical_event_history"
    assert report["predictiveEvidence"] is False
    assert report["probabilityClaim"] is False
    assert report["economicClaim"] is False
    assert report["promotionAllowed"] is False


def test_counts_dedup_overlap_and_lifecycle_unavailability_are_explicit():
    snapshot = _snapshot()
    cutoff = snapshot["candles"][-1]["closeTimeMs"]
    _, rows, report = evaluate(snapshot, EvaluationConfig(cutoff_ms=cutoff, bootstrap_samples=20))
    overlap = next(row for row in rows if row["eventId"] == "e-overlap")
    no_lifecycle = next(row for row in rows if row["eventId"] == "e2")
    assert overlap["status"] == "excluded"
    assert overlap["exclusionReason"] == "overlap_deduplicated"
    assert no_lifecycle["lifecycle"]["status"] == "unavailable"
    assert "no declared lifecycle semantics" in no_lifecycle["lifecycle"]["reason"]
    assert report["counts"] == {
        "stored": 3,
        "eligible": 2,
        "excluded": 1,
        "realizedAtMaxHorizon": 2,
        "exclusionReasons": {"overlap_deduplicated": 1},
    }
    assert report["dependence"]["remainingOutcomeWindowsMayOverlap"] is True
    assert report["dependence"]["independenceClaimed"] is False


def test_empty_event_ledger_retains_complete_predeclared_statistical_family():
    snapshot = _snapshot()
    snapshot["events"] = []
    cutoff = snapshot["candles"][-1]["closeTimeMs"]
    _, rows, report = evaluate(snapshot, EvaluationConfig(cutoff_ms=cutoff, bootstrap_samples=5))
    statistical = report["statisticalEvidence"]
    assert rows == []
    assert report["counts"]["stored"] == 0
    assert statistical["declaredFamily"]["moduleEventTypeIdentityCount"] == 42
    assert statistical["declaredFamily"]["cartesianHypothesisCount"] == 378
    assert len(statistical["hypotheses"]) == 378
    assert statistical["multipleTesting"]["testableFamilySize"] == 0
    assert all(item["status"] == "insufficient_or_no_matched_sample" for item in statistical["hypotheses"])


def test_row_metrics_include_return_mfe_mae_touch_and_invalidation():
    snapshot = _snapshot()
    cutoff = snapshot["candles"][-1]["closeTimeMs"]
    _, rows, _ = evaluate(snapshot, EvaluationConfig(cutoff_ms=cutoff, bootstrap_samples=20))
    row = next(row for row in rows if row["eventId"] == "e1")
    assert set(row["horizons"]["3"]) >= {
        "bars",
        "elapsedMs",
        "forwardReturn",
        "mfe",
        "mae",
    }
    assert row["horizons"]["3"]["mfe"] >= row["horizons"]["3"]["forwardReturn"]
    assert row["horizons"]["3"]["mae"] <= row["horizons"]["3"]["forwardReturn"]
    assert row["lifecycle"]["status"] == "supported"
    assert row["lifecycle"]["firstTouch"]["bars"] == 1
    assert row["lifecycle"]["invalidation"]["bars"] <= 2


def _linear_time_to_level_reference(dataset, decision_index, level, cutoff_ms):
    if not level:
        return None
    for index in range(decision_index + 1, len(dataset.candles)):
        candle = dataset.candles[index]
        if candle.available_ms > cutoff_ms:
            break
        if candle.open_ms - dataset.candles[index - 1].open_ms != dataset.interval_ms:
            break
        if _level_hit(candle, level):
            return {
                "bars": index - decision_index,
                "elapsedMs": candle.close_ms - dataset.candles[decision_index].close_ms,
                "timeMs": candle.close_ms,
            }
    return None


def _random_lifecycle_snapshot(seed=917, count=480):
    rng = np.random.default_rng(seed)
    interval = TIMEFRAME_MS["1h"]
    candles = []
    open_ms = 0
    close = 30_000.0
    for index in range(count):
        if index in {91, 233, 401}:
            open_ms += interval
        next_close = close * (1.0 + float(rng.normal(0.0, 0.002)))
        high = max(close, next_close) * (1.0 + float(rng.uniform(0.0001, 0.004)))
        low = min(close, next_close) * (1.0 - float(rng.uniform(0.0001, 0.004)))
        candles.append(
            {
                "openTimeMs": open_ms,
                "closeTimeMs": open_ms + interval - 1,
                "open": close,
                "high": high,
                "low": low,
                "close": next_close,
                "availableTimeMs": open_ms + interval - 1,
                "finalized": True,
                "context": {"trend": "sideways", "volatility": "normal"},
            }
        )
        close = next_close
        open_ms += interval
    return {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "lineage": {"source": "random-fixture", "sourceVersion": "v1", "contentSha256": HASH},
        "candles": candles,
        "events": [],
    }


def test_indexed_lifecycle_lookup_matches_linear_reference_randomized():
    raw = _random_lifecycle_snapshot()
    cutoff = raw["candles"][-1]["closeTimeMs"]
    dataset = parse_snapshot(raw, cutoff)
    rng = np.random.default_rng(731)
    for _ in range(4_000):
        decision = int(rng.integers(0, len(dataset.candles)))
        operator = "at_or_above" if int(rng.integers(0, 2)) == 0 else "at_or_below"
        target = dataset.candles[int(rng.integers(0, len(dataset.candles)))]
        price = target.high if operator == "at_or_above" else target.low
        level = {"operator": operator, "price": price}
        assert _time_to_level(dataset, decision, level, cutoff) == _linear_time_to_level_reference(
            dataset, decision, level, cutoff
        )


def test_indexed_lifecycle_lookup_preserves_equality_gap_cutoff_and_tail_semantics():
    raw = _random_lifecycle_snapshot(count=480)
    full_cutoff = raw["candles"][-1]["closeTimeMs"]
    dataset = parse_snapshot(raw, full_cutoff)

    equality_level = {"operator": "at_or_above", "price": dataset.candles[1].high}
    assert _time_to_level(dataset, 0, equality_level, full_cutoff)["bars"] == 1

    gap_decision = 90
    beyond_gap = {"operator": "at_or_below", "price": max(candle.low for candle in dataset.candles)}
    assert _time_to_level(dataset, gap_decision, beyond_gap, full_cutoff) is None

    cutoff = dataset.candles[40].close_ms
    later_only = {"operator": "at_or_above", "price": max(candle.high for candle in dataset.candles[41:])}
    assert _time_to_level(dataset, 10, later_only, cutoff) == _linear_time_to_level_reference(
        dataset, 10, later_only, cutoff
    )
    assert _time_to_level(dataset, len(dataset.candles) - 1, equality_level, full_cutoff) is None
    assert _time_to_level(
        dataset,
        len(dataset.candles) - 1,
        {"operator": "unsupported", "price": "not-a-number"},
        full_cutoff,
    ) is None
    assert _time_to_level(
        dataset,
        gap_decision,
        {"operator": "unsupported", "price": "not-a-number"},
        full_cutoff,
    ) is None

    with pytest.raises(ValueError, match="level.operator"):
        _time_to_level(dataset, 0, {"operator": "unsupported", "price": 1.0}, full_cutoff)


def test_unknown_dataset_lineage_and_candle_availability_fail_closed():
    snapshot = _snapshot()
    snapshot["lineage"].pop("contentSha256")
    with pytest.raises(ValueError, match="lineage.contentSha256"):
        parse_snapshot(snapshot, snapshot["candles"][-1]["closeTimeMs"])
    snapshot = _snapshot()
    snapshot["candles"][0].pop("availableTimeMs")
    with pytest.raises(ValueError, match="unknown availability"):
        parse_snapshot(snapshot, snapshot["candles"][-1]["closeTimeMs"])


def test_unknown_event_lineage_is_excluded_and_counted():
    snapshot = _snapshot()
    snapshot["events"][0].pop("lineage")
    cutoff = snapshot["candles"][-1]["closeTimeMs"]
    _, rows, report = evaluate(snapshot, EvaluationConfig(cutoff_ms=cutoff, bootstrap_samples=20))
    assert rows[0]["exclusionReason"] == "unknown_event_lineage"
    assert report["counts"]["exclusionReasons"]["unknown_event_lineage"] == 1


def test_tail_event_is_eligible_but_unrealized_instead_of_zero_outcome():
    snapshot = _snapshot()
    event = deepcopy(snapshot["events"][0])
    event["eventId"] = "tail"
    event["availableTimeMs"] = snapshot["candles"][-2]["closeTimeMs"]
    event["confirmedTimeMs"] = event["availableTimeMs"]
    event["sourceCandleOpenTimesMs"] = [snapshot["candles"][-2]["openTimeMs"]]
    event["context"] = snapshot["candles"][-2]["context"]
    event.pop("lifecycle")
    snapshot["events"] = [event]
    cutoff = snapshot["candles"][-1]["closeTimeMs"]
    _, rows, report = evaluate(snapshot, EvaluationConfig(cutoff_ms=cutoff, bootstrap_samples=20))
    assert rows[0]["status"] == "eligible"
    assert rows[0]["horizons"]["1"] is not None
    assert rows[0]["horizons"]["3"] is None
    assert report["counts"]["realizedAtMaxHorizon"] == 0


def test_matched_controls_use_only_prior_context_and_are_descriptive():
    snapshot = _snapshot()
    cutoff = snapshot["candles"][-1]["closeTimeMs"]
    _, rows, report = evaluate(snapshot, EvaluationConfig(cutoff_ms=cutoff, bootstrap_samples=20))
    row = next(row for row in rows if row["eventId"] == "e1")
    assert row["control"]["status"] == "matched"
    assert row["control"]["decisionCloseTimeMs"] < row["decisionCloseTimeMs"]
    difference = report["eventTypes"]["BOS_BULL"]["horizons"]["1"]["metrics"]["forwardReturn"][
        "matchedControlDifference"
    ]
    assert difference["count"] == 1
    assert "probabilities" in report["limitations"][0]


def test_indexed_control_selection_matches_naive_most_recent_unused_prior_scan():
    raw = _snapshot(count=80)
    cutoff = raw["candles"][-1]["closeTimeMs"]
    config = EvaluationConfig(cutoff_ms=cutoff, bootstrap_samples=10)
    dataset = parse_snapshot(raw, cutoff)
    rows, _ = _evaluate_event_rows(dataset.events, dataset, config, lifecycle=True)
    optimized = deepcopy(rows)
    naive = deepcopy(rows)
    _select_controls(optimized, dataset, config)

    family_of = lambda row: (str(row.get("module") or "unknown"), str(row.get("eventType") or "unknown"))
    family_decisions: dict = {}
    for row in naive:
        if row.get("decisionIndex") is not None:
            family_decisions.setdefault(family_of(row), set()).add(row["decisionIndex"])
    reserved_by_family = {
        family: {
            index
            for decision in decisions
            for index in range(max(0, decision - max(HORIZONS)), min(len(dataset.candles), decision + max(HORIZONS) + 1))
        }
        for family, decisions in family_decisions.items()
    }
    used_by_family: dict = {}
    for row in sorted((item for item in naive if item["status"] == "eligible"), key=lambda item: item["decisionIndex"]):
        family = family_of(row)
        reserved = reserved_by_family.get(family, set())
        used = used_by_family.setdefault(family, set())
        decision_available = dataset.candles[row["decisionIndex"]].available_ms
        chosen = None
        for index in range(row["decisionIndex"] - 1, -1, -1):
            if index in reserved or index in used:
                continue
            if _context_signature(dataset.candles[index].context, config.context_keys) != row["contextSignature"]:
                continue
            outcome = _horizon_metrics(dataset, index, max(HORIZONS), cutoff)
            if outcome is None or outcome["availableTimeMs"] >= decision_available:
                continue
            chosen = index
            break
        if chosen is None:
            row["control"] = {"status": "unavailable", "reason": "no_prior_context_match"}
        else:
            used.add(chosen)
            row["control"] = {
                "status": "matched",
                "decisionOpenTimeMs": dataset.candles[chosen].open_ms,
                "decisionCloseTimeMs": dataset.candles[chosen].close_ms,
                "controlOutcomeEndIndex": chosen + max(HORIZONS),
                "outcomeAvailableBeforeEvent": _horizon_metrics(dataset, chosen, max(HORIZONS), cutoff)["availableTimeMs"] < decision_available,
                "contextSignature": row["contextSignature"],
                "horizons": {
                    str(horizon): _horizon_metrics(dataset, chosen, horizon, cutoff)
                    for horizon in HORIZONS
                },
            }
    assert [row["control"] for row in optimized] == [row["control"] for row in naive]
    assert all(
        row["control"].get("outcomeAvailableBeforeEvent") is True
        for row in optimized if row["control"]["status"] == "matched"
    )


def _event_at(snapshot, event_id, index, event_type, module="causalSmc", context=None):
    candles = snapshot["candles"]
    interval = TIMEFRAME_MS[snapshot["timeframe"]]
    return {
        "eventId": event_id,
        "module": module,
        "eventType": event_type,
        "formedTimeMs": candles[index - 1]["closeTimeMs"],
        "confirmedTimeMs": candles[index]["closeTimeMs"],
        "availableTimeMs": candles[index]["closeTimeMs"],
        "sourceCandleOpenTimesMs": [candles[index - 1]["openTimeMs"], candles[index]["openTimeMs"]],
        "context": dict(context if context is not None else candles[index]["context"]),
        "lineage": {"source": "causal-smc", "sourceVersion": "v3", "contentSha256": HASH},
    }


def test_dense_other_family_events_do_not_starve_same_regime_controls():
    snapshot = _snapshot(count=80)
    for candle in snapshot["candles"]:
        candle["context"] = {"trend": "down", "volatility": "normal"}
    focal_regime = {"trend": "up", "volatility": "normal"}
    # The only candle matching the focal regime sits inside the ±max(HORIZONS)
    # reserved zone of an unrelated family (BOS_BULL decision at index 40).
    # Old all-family exclusion starved this control; per-family selection must keep it.
    snapshot["candles"][46]["context"] = focal_regime
    snapshot["events"] = [
        _event_at(snapshot, "bos", 40, "BOS_BULL"),
        _event_at(snapshot, "fvg", 55, "FVG_BULL", context=focal_regime),
    ]
    cutoff = snapshot["candles"][-1]["closeTimeMs"]
    _, rows, _ = evaluate(snapshot, EvaluationConfig(cutoff_ms=cutoff, bootstrap_samples=10))
    bos = next(row for row in rows if row["eventId"] == "bos")
    fvg = next(row for row in rows if row["eventId"] == "fvg")
    assert bos["status"] == "eligible" and bos["control"]["status"] == "matched"
    assert fvg["control"]["status"] == "matched"
    assert fvg["control"]["decisionOpenTimeMs"] == snapshot["candles"][46]["openTimeMs"]
    assert fvg["control"]["outcomeAvailableBeforeEvent"] is True


def test_non_monotone_candle_availability_is_rejected():
    snapshot = _snapshot(count=80)
    snapshot["candles"][39]["availableTimeMs"] = snapshot["candles"][40]["closeTimeMs"] + 1
    cutoff = snapshot["candles"][-1]["closeTimeMs"]
    with pytest.raises(ValueError, match="non-decreasing"):
        parse_snapshot(snapshot, cutoff)


def test_control_outcome_not_strictly_prior_to_decision_is_rejected():
    snapshot = _snapshot(count=80)
    for candle in snapshot["candles"]:
        candle["context"] = {"trend": "up", "volatility": "normal"}
    event = _event_at(snapshot, "evt", 40, "BOS_BULL")
    event["formedTimeMs"] = event["confirmedTimeMs"]
    event["sourceCandleOpenTimesMs"] = [snapshot["candles"][40]["openTimeMs"]]
    snapshot["events"] = [event]
    # Index 33 is the most recent candidate whose 6-bar window ends at 39,
    # strictly before the decision at 40 by index — but its outcome bar is only
    # published at the same instant as the decision bar, not strictly before.
    snapshot["candles"][39]["availableTimeMs"] = snapshot["candles"][40]["availableTimeMs"]
    cutoff = snapshot["candles"][-1]["closeTimeMs"]
    _, rows, _ = evaluate(snapshot, EvaluationConfig(cutoff_ms=cutoff, bootstrap_samples=10))
    control = rows[0]["control"]
    assert control["status"] == "matched"
    assert control["decisionOpenTimeMs"] == snapshot["candles"][32]["openTimeMs"]
    assert control["horizons"]["6"]["availableTimeMs"] < snapshot["candles"][40]["availableTimeMs"]
    assert control["outcomeAvailableBeforeEvent"] is True


def test_future_data_and_new_events_do_not_change_past_control_matching():
    snapshot = _snapshot(count=80)
    cutoff = snapshot["candles"][-1]["closeTimeMs"]
    _, rows_before, _ = evaluate(snapshot, EvaluationConfig(cutoff_ms=cutoff, bootstrap_samples=10))
    controls_before = {row["eventId"]: row["control"] for row in rows_before}

    extended = _snapshot(count=100)
    extended["events"] = deepcopy(extended["events"])
    extended["events"].append(_event_at(extended, "future", 90, "FVG_BULL"))
    extended_cutoff = extended["candles"][-1]["closeTimeMs"]
    _, rows_after, _ = evaluate(extended, EvaluationConfig(cutoff_ms=extended_cutoff, bootstrap_samples=10))
    controls_after = {row["eventId"]: row["control"] for row in rows_after}

    assert set(controls_before) <= set(controls_after)
    for event_id, control in controls_before.items():
        assert controls_after[event_id] == control


def test_bundle_is_content_addressed_and_semantically_verified(tmp_path):
    snapshot = _causal_snapshot()
    cutoff = snapshot["candles"][-1]["closeTimeMs"]
    config = EvaluationConfig(cutoff_ms=cutoff, bootstrap_samples=20)
    frozen, rows, report = evaluate(snapshot, config)
    paths = write_bundle(tmp_path, frozen, rows, report, config)
    result = verify_bundle(paths["manifest"])
    assert result["valid"] is True
    assert result["schema"] == "btc-technical-event-descriptive-evidence/v1"
    assert result["manifestSha256"] == paths["manifest"].name.split(".")[0]
    assert paths["runtime"].name.endswith(".runtime.json")
    assert result["rows"] == len(rows)
    assert result["eligible"] == report["counts"]["eligible"]
    assert result["claimType"] == "descriptive_technical_event_history"
    manifest = json.loads(paths["manifest"].read_text())
    for artifact in manifest["artifacts"].values():
        assert artifact["fileName"].startswith(artifact["sha256"] + ".")


def test_semantic_manifest_is_independent_from_runtime_git_metadata(tmp_path, monkeypatch):
    snapshot = _causal_snapshot()
    cutoff = snapshot["candles"][-1]["closeTimeMs"]
    config = EvaluationConfig(cutoff_ms=cutoff, bootstrap_samples=10)
    frozen, rows, report = evaluate(snapshot, config)
    monkeypatch.setattr(evidence_module, "_git_state", lambda repo: {"commit": "first", "dirty": False})
    first = write_bundle(tmp_path / "first", frozen, rows, report, config)
    monkeypatch.setattr(evidence_module, "_git_state", lambda repo: {"commit": "second", "dirty": True})
    second = write_bundle(tmp_path / "second", frozen, rows, report, config)
    assert first["manifest"].name == second["manifest"].name
    assert first["manifest"].read_bytes() == second["manifest"].read_bytes()
    assert first["runtime"].name != second["runtime"].name
    assert verify_bundle(first["manifest"])["valid"] is True
    assert verify_bundle(second["manifest"])["valid"] is True


def test_semantic_verifier_rejects_forged_row_even_with_consistent_file_hashes(tmp_path):
    snapshot = _causal_snapshot()
    cutoff = snapshot["candles"][-1]["closeTimeMs"]
    config = EvaluationConfig(cutoff_ms=cutoff, bootstrap_samples=20)
    frozen, rows, report = evaluate(snapshot, config)
    forged = deepcopy(rows)
    target = next(row for row in forged if row.get("horizons", {}).get("1") is not None)
    target["horizons"]["1"]["forwardReturn"] = 9.0
    paths = write_bundle(tmp_path, frozen, forged, report, config)
    with pytest.raises(ValueError, match="ledger does not match causal recomputation"):
        verify_bundle(paths["manifest"])


def test_semantic_verifier_rejects_forged_derived_event_even_when_bundle_hashes_are_consistent(tmp_path):
    snapshot = _causal_snapshot()
    cutoff = snapshot["candles"][-1]["closeTimeMs"]
    config = EvaluationConfig(cutoff_ms=cutoff, bootstrap_samples=10)
    frozen, rows, report = evaluate(snapshot, config)
    forged = deepcopy(frozen)
    derived = next(event for event in forged["events"] if event.get("module") == "candlePatterns")
    derived["eventType"] = "FORGED_PATTERN"
    forged["lineage"]["contentSha256"] = sha256_canonical({
        "timeframe": forged["timeframe"],
        "cutoffMs": forged["lineage"]["sourceCutoffMs"],
        "candles": forged["candles"],
        "events": forged["events"],
        "modules": forged["modules"],
    })
    paths = write_bundle(tmp_path, forged, rows, report, config)
    with pytest.raises(ValueError, match="do not match causal reconstruction"):
        verify_bundle(paths["manifest"])


def test_verifier_rejects_manifest_bound_to_different_module_code(tmp_path):
    snapshot = _causal_snapshot()
    cutoff = snapshot["candles"][-1]["closeTimeMs"]
    config = EvaluationConfig(cutoff_ms=cutoff, bootstrap_samples=10)
    frozen, rows, report = evaluate(snapshot, config)
    paths = write_bundle(tmp_path, frozen, rows, report, config)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    module_component = next(
        item for item in manifest["code"]["components"] if item["role"] == "technicalModules"
    )
    module_component["sha256"] = "0" * 64
    manifest["code"]["sha256"] = __import__("hashlib").sha256(
        json.dumps(manifest["code"]["components"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    content = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    forged_path = tmp_path / f"{__import__('hashlib').sha256(content).hexdigest()}.manifest.json"
    forged_path.write_bytes(content)
    with pytest.raises(ValueError, match="provenance"):
        verify_bundle(forged_path)


@pytest.mark.parametrize("field", ["scope", "sourceLineage"])
def test_verifier_rejects_manifest_metadata_not_bound_to_frozen_evidence(tmp_path, field):
    snapshot = _causal_snapshot()
    cutoff = snapshot["candles"][-1]["closeTimeMs"]
    config = EvaluationConfig(cutoff_ms=cutoff, bootstrap_samples=20)
    frozen, rows, report = evaluate(snapshot, config)
    paths = write_bundle(tmp_path, frozen, rows, report, config)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest[field] = dict(manifest[field])
    manifest[field]["tampered"] = True
    content = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    forged_path = tmp_path / f"{__import__('hashlib').sha256(content).hexdigest()}.manifest.json"
    forged_path.write_bytes(content)
    with pytest.raises(ValueError, match="scope|source lineage"):
        verify_bundle(forged_path)


def test_verifier_rejects_false_postgresql_source_content_hash(tmp_path):
    snapshot = _causal_snapshot()
    cutoff = snapshot["candles"][-1]["closeTimeMs"]
    config = EvaluationConfig(cutoff_ms=cutoff, bootstrap_samples=20)
    frozen, rows, report = evaluate(snapshot, config)
    frozen["lineage"] = dict(frozen["lineage"])
    frozen["lineage"]["contentSha256"] = "0" * 64
    paths = write_bundle(tmp_path, frozen, rows, report, config)
    with pytest.raises(ValueError, match="source lineage content hash"):
        verify_bundle(paths["manifest"])


def test_postgresql_mapping_builds_causal_context_lineage_and_fvg_lifecycle():
    interval = TIMEFRAME_MS["4h"]
    klines = []
    price = 100.0
    for index in range(30):
        price += 1 if index % 3 else -0.5
        klines.append(
            {
                "OpenTimeMs": index * interval,
                "CloseTimeMs": (index + 1) * interval - 1,
                "Open": price - 0.25,
                "High": price + 1,
                "Low": price - 1,
                "Close": price,
            }
        )
    canonical = [_canonical_smc_row(
        "persisted-fvg-bull",
        "FVG_BULL",
        12 * interval,
        14 * interval - 1,
        [11 * interval, 12 * interval, 13 * interval],
        high=102.0,
        low=100.0,
        state="mitigated",
        mitigated_at=20 * interval - 1,
    )]
    smc = [
        {
            "TimeMs": 16 * interval,
            "OriginTimeMs": 16 * interval,
            "ReferenceTimeMs": None,
            "AvailableTimeMs": None,
            "EventType": "SWING_HIGH",
            "CalculationVersion": "legacy-uncausal-v1",
            "Price": 110.0,
            "HighPrice": None,
            "LowPrice": None,
            "MitigatedAtMs": None,
        },
    ]
    columns = list(smc[0])
    cutoff = klines[-1]["CloseTimeMs"]
    snapshot = build_postgresql_snapshot_from_rows(
        "4h", cutoff, klines, smc, smc_columns=columns,
        causal_smc_rows=canonical, causal_smc_columns=list(canonical[0]),
        causal_smc_checkpoint=_complete_checkpoint(klines),
        causal_smc_checkpoint_columns=list(_complete_checkpoint(klines)),
    )
    assert snapshot["lineage"]["source"] == "postgresql:Klines+CausalSmartMoneyEvents+SmartMoneyStructuresLegacyAudit"
    assert snapshot["modules"]["causalSmc"]["status"] == "evaluable"
    assert snapshot["modules"]["causalSmc"]["coverage"]["coversRequestedCutoff"] is True
    causal = next(event for event in snapshot["events"] if event.get("module") == "causalSmc" and event["eventType"] == "FVG_BULL")
    assert causal["eventId"] == "persisted-fvg-bull"
    assert causal["lineage"]["contentSha256"] == canonical[0]["DecisionEvidenceSha256"]
    assert causal["sourceCandleOpenTimesMs"] == [11 * interval, 12 * interval, 13 * interval]
    assert causal["context"] == snapshot["candles"][13]["context"]
    assert causal["lifecycle"] == {
        "semanticsVersion": "smc-causal-fvg-zone-v1",
        "touchLevel": {"operator": "at_or_below", "price": 102.0},
        "mitigationLevel": {"operator": "at_or_below", "price": 100.0},
        "invalidationUnavailableReason": "smc-causal-v2 defines complete fill as mitigation and has no validated invalidation rule",
    }
    # Stored mitigation is intentionally ignored; lifecycle is reconstructed.
    assert "MitigatedAtMs" not in causal["lifecycle"]
    _, rows, report = evaluate(snapshot, EvaluationConfig(cutoff_ms=cutoff, bootstrap_samples=20))
    assert next(row for row in rows if row["eventId"] == causal["eventId"])["status"] == "eligible"
    assert report["counts"]["exclusionReasons"]["unknown_availability"] == 1


def test_fvg_lifecycle_reads_hash_bound_evidence_not_mutable_row_columns():
    interval = TIMEFRAME_MS["4h"]
    klines = [
        {
            "OpenTimeMs": index * interval,
            "CloseTimeMs": (index + 1) * interval - 1,
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
        }
        for index in range(20)
    ]
    row = _canonical_smc_row(
        "fvg-zone", "FVG_BULL", 5 * interval, 7 * interval - 1,
        [4 * interval, 5 * interval, 6 * interval], high=102.0, low=101.0,
    )
    # HighPrice/LowPrice row columns sit outside the immutable-decision check;
    # drifting them must not move the lifecycle zone bound by
    # DecisionEvidenceJson (which still hashes and validates cleanly).
    row["HighPrice"] = 250.0
    row["LowPrice"] = 240.0
    snapshot = build_postgresql_snapshot_from_rows(
        "4h", klines[-1]["CloseTimeMs"], klines, [], smc_columns=[],
        causal_smc_rows=[row], causal_smc_columns=list(row),
    )
    event = next(event for event in snapshot["events"] if event["eventId"] == "fvg-zone")
    assert event["lifecycle"]["touchLevel"] == {"operator": "at_or_below", "price": 102.0}
    assert event["lifecycle"]["mitigationLevel"] == {"operator": "at_or_below", "price": 101.0}


def test_postgresql_mapping_bos_sources_include_full_confirmed_reference_pivot():
    interval = TIMEFRAME_MS["4h"]
    row = _canonical_smc_row(
        "persisted-bos-bull", "BOS_BULL", 20 * interval, 21 * interval - 1,
        [10 * interval, 11 * interval, 12 * interval, 13 * interval, 14 * interval, 20 * interval],
        reference=12 * interval,
    )
    klines = [
        {
            "OpenTimeMs": index * interval,
            "CloseTimeMs": (index + 1) * interval - 1,
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
        }
        for index in range(30)
    ]
    snapshot = build_postgresql_snapshot_from_rows(
        "4h", klines[-1]["CloseTimeMs"], klines, [], smc_columns=[],
        causal_smc_rows=[row], causal_smc_columns=list(row),
    )
    bos = next(event for event in snapshot["events"] if event.get("module") == "causalSmc" and event["eventType"] == "BOS_BULL")
    assert bos["sourceCandleOpenTimesMs"] == [
        10 * interval,
        11 * interval,
        12 * interval,
        13 * interval,
        14 * interval,
        20 * interval,
    ]


def test_postgresql_lineage_ignores_mutable_mitigation_observation():
    interval = TIMEFRAME_MS["4h"]
    klines = [
        {
            "OpenTimeMs": index * interval,
            "CloseTimeMs": (index + 1) * interval - 1,
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
        }
        for index in range(12)
    ]
    row = _canonical_smc_row(
        "persisted-fvg-bear", "FVG_BEAR", 5 * interval, 7 * interval - 1,
        [4 * interval, 5 * interval, 6 * interval], high=102.0, low=101.0,
    )
    cutoff = klines[-1]["CloseTimeMs"]
    before = build_postgresql_snapshot_from_rows(
        "4h", cutoff, klines, [], smc_columns=[],
        causal_smc_rows=[row], causal_smc_columns=list(row),
    )
    mutated = dict(row, MitigatedAtMs=10 * interval, State="mitigated")
    after = build_postgresql_snapshot_from_rows(
        "4h", cutoff, klines, [], smc_columns=[],
        causal_smc_rows=[mutated], causal_smc_columns=list(mutated),
    )
    assert before["events"] == after["events"]
    assert before["lineage"]["contentSha256"] == after["lineage"]["contentSha256"]


def test_legacy_event_hash_ignores_later_availability_backfill():
    interval = TIMEFRAME_MS["1h"]
    klines = [
        {
            "OpenTimeMs": index * interval,
            "CloseTimeMs": (index + 1) * interval - 1,
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
        }
        for index in range(12)
    ]
    legacy = {
        "Id": 12,
        "TimeMs": 5 * interval,
        "OriginTimeMs": 5 * interval,
        "ReferenceTimeMs": None,
        "AvailableTimeMs": None,
        "EventType": "SWING_HIGH",
        "CalculationVersion": "legacy-uncausal-v1",
        "Price": 101.0,
        "HighPrice": None,
        "LowPrice": None,
    }
    cutoff = klines[-1]["CloseTimeMs"]
    before = build_postgresql_snapshot_from_rows("1h", cutoff, klines, [legacy], smc_columns=list(legacy))
    after = build_postgresql_snapshot_from_rows(
        "1h", cutoff, klines, [dict(legacy, AvailableTimeMs=8 * interval - 1)], smc_columns=list(legacy)
    )
    assert before["events"] == after["events"]
    assert before["lineage"]["contentSha256"] == after["lineage"]["contentSha256"]


def test_postgresql_mapping_marks_module_unavailable_when_causal_columns_are_missing():
    interval = TIMEFRAME_MS["1h"]
    klines = [
        {
            "OpenTimeMs": index * interval,
            "CloseTimeMs": (index + 1) * interval - 1,
            "Open": 100 + index,
            "High": 101 + index,
            "Low": 99 + index,
            "Close": 100.5 + index,
        }
        for index in range(20)
    ]
    legacy = [{"TimeMs": 10 * interval, "EventType": "SWING_HIGH", "Price": 110.0}]
    snapshot = build_postgresql_snapshot_from_rows(
        "1h", klines[-1]["CloseTimeMs"], klines, legacy, smc_columns=list(legacy[0])
    )
    module = snapshot["modules"]["causalSmc"]
    assert module["status"] == "unavailable"
    assert "canonical table" in module["reason"]
    _, rows, report = evaluate(
        snapshot, EvaluationConfig(cutoff_ms=klines[-1]["CloseTimeMs"], bootstrap_samples=20)
    )
    legacy_id = next(event["eventId"] for event in snapshot["events"] if event.get("module") is None)
    assert next(row for row in rows if row["eventId"] == legacy_id)["exclusionReason"] == "unknown_availability"
    assert report["modules"]["causalSmc"]["status"] == "unavailable"


def test_canonical_empty_or_incomplete_checkpoint_is_not_interpreted_as_zero_events():
    interval = TIMEFRAME_MS["4h"]
    klines = [
        {
            "OpenTimeMs": index * interval,
            "CloseTimeMs": (index + 1) * interval - 1,
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
        }
        for index in range(20)
    ]
    canonical_columns = list(_canonical_smc_row(
        "shape", "BOS_BULL", 5 * interval, 6 * interval - 1, [5 * interval]
    ))
    checkpoint = _complete_checkpoint(klines, status="checkpointed", last=10 * interval)
    snapshot = build_postgresql_snapshot_from_rows(
        "4h", klines[-1]["CloseTimeMs"], klines, [], smc_columns=[],
        causal_smc_rows=[], causal_smc_columns=canonical_columns,
        causal_smc_checkpoint=checkpoint, causal_smc_checkpoint_columns=list(checkpoint),
    )
    module = snapshot["modules"]["causalSmc"]
    assert module["status"] == "partial"
    assert module["causalRows"] == 0
    assert module["coverage"]["coversRequestedCutoff"] is False
    assert "does not cover" in module["reason"]


def test_canonical_cutoff_ignores_latest_future_mitigation_state_and_future_rows():
    interval = TIMEFRAME_MS["4h"]
    klines = [
        {
            "OpenTimeMs": index * interval,
            "CloseTimeMs": (index + 1) * interval - 1,
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
        }
        for index in range(9)
    ]
    cutoff = klines[-1]["CloseTimeMs"]
    visible = _canonical_smc_row(
        "visible-fvg", "FVG_BEAR", 5 * interval, 7 * interval - 1,
        [4 * interval, 5 * interval, 6 * interval], high=102.0, low=101.0,
        state="mitigated", mitigated_at=11 * interval - 1,
    )
    future = _canonical_smc_row(
        "future-bos", "BOS_BULL", 9 * interval, 10 * interval - 1,
        [9 * interval],
    )
    checkpoint = _complete_checkpoint(klines)
    snapshot = build_postgresql_snapshot_from_rows(
        "4h", cutoff, klines, [], smc_columns=[],
        causal_smc_rows=[visible, future], causal_smc_columns=list(visible),
        causal_smc_checkpoint=checkpoint, causal_smc_checkpoint_columns=list(checkpoint),
    )
    assert snapshot["modules"]["causalSmc"]["futureRowsExcludedByCutoff"] == 1
    assert all(event["eventId"] != "future-bos" for event in snapshot["events"])
    _, rows, _ = evaluate(snapshot, EvaluationConfig(cutoff_ms=cutoff, bootstrap_samples=5))
    visible_row = next(row for row in rows if row["eventId"] == "visible-fvg")
    assert visible_row["status"] == "eligible"
    assert visible_row["lifecycle"]["mitigation"] is None


def test_legacy_smc_causal_version_is_still_excluded_without_canonical_table():
    interval = TIMEFRAME_MS["1h"]
    klines = [
        {
            "OpenTimeMs": index * interval,
            "CloseTimeMs": (index + 1) * interval - 1,
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
        }
        for index in range(12)
    ]
    legacy = {
        "Id": 99,
        "TimeMs": 5 * interval,
        "OriginTimeMs": 5 * interval,
        "AvailableTimeMs": 7 * interval - 1,
        "ReferenceTimeMs": None,
        "EventType": "BOS_BULL",
        "CalculationVersion": "smc-causal-v2",
        "Price": 100.0,
        "HighPrice": None,
        "LowPrice": None,
    }
    cutoff = klines[-1]["CloseTimeMs"]
    snapshot = build_postgresql_snapshot_from_rows(
        "1h", cutoff, klines, [legacy], smc_columns=list(legacy)
    )
    _, rows, report = evaluate(snapshot, EvaluationConfig(cutoff_ms=cutoff, bootstrap_samples=5))
    legacy_row = next(row for row in rows if row["eventId"].startswith("smc-"))
    assert legacy_row["exclusionReason"] == "unknown_availability"
    assert report["modules"]["causalSmc"]["status"] == "unavailable"


def test_sensitivity_grid_is_executed_and_reports_causal_counts_overlap_and_outcomes():
    from tests.unit.test_technical_event_modules import golden_rows

    klines = golden_rows()
    cutoff = klines[-1]["CloseTimeMs"]
    snapshot = build_postgresql_snapshot_from_rows("4h", cutoff, klines, [], smc_columns=[])
    _, _, report = evaluate(snapshot, EvaluationConfig(cutoff_ms=cutoff, bootstrap_samples=10))

    audit = report["sensitivityAudit"]
    assert audit["resultSelection"] is False
    assert audit["promotionAllowed"] is False
    assert audit["variantCount"] == 31
    assert {row["module"] for row in audit["variants"]} == {
        "technicalIndicators", "candlePatterns", "volumeAnomaly", "marketRegime",
        "fibonacci", "volumeProfile", "confluence",
    }
    assert any(row["stored"] > 0 and row["eligible"] > 0 for row in audit["variants"])
    for row in audit["variants"]:
        assert 0.0 <= row["eligibleDecisionTimeJaccardVsBaseline"] <= 1.0
        assert row["stored"] == row["eligible"] + row["excluded"]
        assert set(row["horizons"]) == {"1", "3", "6"}
        assert set(row["horizons"]["1"]["metrics"]) == {"forwardReturn", "mfe", "mae"}
