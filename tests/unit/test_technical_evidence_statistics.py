from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from technical_evidence_statistics import (
    STATISTICS_SCHEMA,
    build_statistical_evidence,
    declared_event_families,
    load_statistics_spec,
)


HORIZONS = (1, 3, 6)
METRICS = ("forwardReturn", "mfe", "mae")


def _milliseconds(year, month, day):
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp() * 1000)


def _row(event_id, event_type, module, decision, close_ms, value, regime="up|normal"):
    trend, volatility = regime.split("|")
    event_horizons = {
        str(horizon): {metric: value * (metric_index + 1) for metric_index, metric in enumerate(METRICS)}
        for horizon in HORIZONS
    }
    control_horizons = {
        str(horizon): {metric: 0.0 for metric in METRICS}
        for horizon in HORIZONS
    }
    signature = f"trend={trend}|volatility={volatility}"
    return {
        "eventId": event_id,
        "module": module,
        "eventType": event_type,
        "status": "eligible",
        "exclusionReason": None,
        "decisionIndex": decision,
        "decisionCloseTimeMs": close_ms,
        "context": {"trend": trend, "volatility": volatility},
        "contextSignature": signature,
        "horizons": event_horizons,
        "control": {
            "status": "matched",
            "outcomeAvailableBeforeEvent": True,
            "contextSignature": signature,
            "horizons": control_horizons,
        },
    }


def _summary(count, p_value):
    return {
        "count": count,
        "mean": 0.0,
        "median": 0.0,
        "q25": 0.0,
        "q75": 0.0,
        "meanBlockBootstrapInterval": {"lower": -0.1, "upper": 0.1},
        "centeredNullTwoSidedPValue": p_value,
    }


def _event_summary(count, p_value):
    return {
        "horizons": {
            str(horizon): {
                "metrics": {
                    metric: {"matchedControlDifference": _summary(count, p_value)}
                    for metric in METRICS
                }
            }
            for horizon in HORIZONS
        }
    }


def _sensitivity():
    return {
        "preRegisteredGrid": {
            "declaredGridSha256": "c" * 64,
            "executedVariantIds": ["technicalIndicators:fixed-a", "technicalIndicators:fixed-b"],
            "allExecutedVariantsRetained": True,
        }
    }


def test_statistics_spec_is_versioned_deterministic_and_explicit_about_assumptions():
    first, first_sha = load_statistics_spec()
    second, second_sha = load_statistics_spec()
    assert first == second
    assert first_sha == second_sha
    assert first["specVersion"] == STATISTICS_SCHEMA
    assert first["multipleTesting"]["method"] == "Benjamini-Yekutieli"
    assert first["multipleTesting"]["dependenceAssumption"]
    assert len(first["dependence"]["assumptions"]) >= 3
    assert first["sensitivity"]["outcomeDrivenSelectionAllowed"] is False


def test_machine_readable_output_schema_and_example_are_versioned_and_nullable():
    contracts = Path(__file__).parents[2] / "contracts"
    schema = json.loads((contracts / "technical-evidence-statistics.schema.json").read_text(encoding="utf-8"))
    example = json.loads((contracts / "technical-evidence-statistics.example.json").read_text(encoding="utf-8"))
    _, spec_sha = load_statistics_spec()
    assert schema["$id"] == example["schema"] == STATISTICS_SCHEMA
    assert example["specSha256"] == spec_sha
    expected = build_statistical_evidence(
        [],
        {},
        timeframe="1h",
        horizons=HORIZONS,
        metrics=METRICS,
        block_size_events=8,
        bootstrap_samples=2_000,
        random_seed=42,
        sensitivity={
            "preRegisteredGrid": {
                "declaredGridSha256": None,
                "executedVariantIds": [],
                "allExecutedVariantsRetained": True,
            }
        },
    )
    assert example == expected
    assert len(example["hypotheses"]) == 378
    assert example["multipleTesting"]["familySizeAllRetained"] == 378
    assert example["multipleTesting"]["testableFamilySize"] == 0
    assert all(item["rawPValue"] is None for item in example["hypotheses"])
    assert all(item["adjustedQValue"] is None for item in example["hypotheses"])
    assert all(item["passesDeclaredFdr"] is None for item in example["hypotheses"])
    try:
        import jsonschema
    except ImportError:
        jsonschema = None
    if jsonschema is not None:
        jsonschema.Draft202012Validator(schema).validate(example)


def test_effect_null_multiple_testing_stability_and_sample_diagnostics_are_complete():
    rows = [
        _row("p1", "EMA_BULL_CROSS", "technicalIndicators", 10, _milliseconds(2024, 1, 1), 0.03, "up|normal"),
        _row("p2", "EMA_BULL_CROSS", "technicalIndicators", 13, _milliseconds(2024, 2, 1), 0.02, "up|normal"),
        _row("p3", "EMA_BULL_CROSS", "technicalIndicators", 30, _milliseconds(2025, 1, 1), -0.01, "down|high"),
        _row("p4", "EMA_BULL_CROSS", "technicalIndicators", 50, _milliseconds(2025, 2, 1), 0.04, "down|high"),
        _row("n1", "DOJI", "candlePatterns", 70, _milliseconds(2025, 3, 1), -0.03, "down|normal"),
        _row("n2", "DOJI", "candlePatterns", 90, _milliseconds(2025, 4, 1), -0.02, "down|normal"),
    ]
    report = build_statistical_evidence(
        rows,
        {"EMA_BULL_CROSS": _event_summary(4, 0.001), "DOJI": _event_summary(2, 0.2)},
        timeframe="1h",
        horizons=HORIZONS,
        metrics=METRICS,
        block_size_events=8,
        bootstrap_samples=2_000,
        random_seed=42,
        sensitivity=_sensitivity(),
    )
    assert report["schema"] == STATISTICS_SCHEMA
    assert report["multipleTesting"]["familySizeAllRetained"] == 378
    assert report["multipleTesting"]["testableFamilySize"] == 18
    assert len(report["hypotheses"]) == 378

    positive = next(item for item in report["hypotheses"] if item["hypothesisId"] == "technicalIndicators:EMA_BULL_CROSS:6:forwardReturn")
    negative = next(item for item in report["hypotheses"] if item["hypothesisId"] == "candlePatterns:DOJI:1:forwardReturn")
    assert positive["effectSize"]["meanPairedDifference"] == pytest.approx(0.02)
    assert positive["effectSize"]["positiveFraction"] == 0.75
    assert negative["effectSize"]["meanPairedDifference"] < 0
    assert positive["adjustedQValue"] >= positive["rawPValue"]
    assert negative["adjustedQValue"] >= negative["rawPValue"]
    assert positive["sampleDiagnostics"]["nominalMatchedPairs"] == 4
    assert positive["sampleDiagnostics"]["maximumGreedyNonOverlappingOutcomeWindows"] < 4
    assert positive["sampleDiagnostics"]["observationsExcludedForMaximumNonOverlappingSet"] > 0
    assert "outcomeWindowOverlapPairs" not in positive["sampleDiagnostics"]
    assert positive["sampleDiagnostics"]["independenceClaimed"] is False
    assert set(positive["stability"]["yearUtc"]["groups"]) == {"2024", "2025"}
    assert set(positive["stability"]["regime"]["groups"]) == {
        "trend=down|volatility=high",
        "trend=up|volatility=normal",
    }
    assert report["sensitivityGrid"]["allExecutedVariantsRetained"] is True
    assert report["sensitivityGrid"]["outcomeDrivenSelectionAllowed"] is False


def test_empty_ledger_emits_every_declared_no_sample_hypothesis_without_fake_p_values():
    report = build_statistical_evidence(
        [],
        {},
        timeframe="4h",
        horizons=HORIZONS,
        metrics=METRICS,
        block_size_events=8,
        bootstrap_samples=100,
        random_seed=7,
        sensitivity=_sensitivity(),
    )
    families, _ = declared_event_families()
    declared = {
        f"{module}:{event_type}:{horizon}:{metric}"
        for module, event_types in families.items()
        for event_type in event_types
        for horizon in HORIZONS
        for metric in METRICS
    }
    assert len(report["hypotheses"]) == report["declaredFamily"]["cartesianHypothesisCount"] == 378
    assert {item["hypothesisId"] for item in report["hypotheses"]} == declared
    assert report["multipleTesting"]["testableFamilySize"] == 0
    assert all(item["status"] == "insufficient_or_no_matched_sample" for item in report["hypotheses"])
    assert all(item["effectSize"]["count"] == 0 for item in report["hypotheses"])
    assert all(item["rawPValue"] is None and item["adjustedQValue"] is None for item in report["hypotheses"])


def test_same_regime_null_and_summary_count_mismatches_fail_closed():
    rows = [_row("a", "FVG_BULL", "causalSmc", 10, _milliseconds(2025, 1, 1), 0.1)]
    mismatched_regime = deepcopy(rows)
    mismatched_regime[0]["control"]["contextSignature"] = "trend=down|volatility=high"
    with pytest.raises(ValueError, match="regime differs"):
        build_statistical_evidence(
            mismatched_regime,
            {"FVG_BULL": _event_summary(1, None)},
            timeframe="1d",
            horizons=HORIZONS,
            metrics=METRICS,
            block_size_events=8,
            bootstrap_samples=100,
            random_seed=1,
            sensitivity=_sensitivity(),
        )
    with pytest.raises(ValueError, match="count disagrees"):
        build_statistical_evidence(
            rows,
            {"FVG_BULL": _event_summary(2, 0.1)},
            timeframe="1d",
            horizons=HORIZONS,
            metrics=METRICS,
            block_size_events=8,
            bootstrap_samples=100,
            random_seed=1,
            sensitivity=_sensitivity(),
        )


def test_benjamini_yekutieli_matches_hand_computed_example():
    from technical_evidence_statistics import _benjamini_yekutieli

    hypotheses = [
        {"hypothesisId": "a", "rawPValue": 0.5, "status": "tested"},
        {"hypothesisId": "b", "rawPValue": 0.001, "status": "tested"},
        {"hypothesisId": "c", "rawPValue": 0.01, "status": "tested"},
        {"hypothesisId": "d", "rawPValue": 0.9, "status": "insufficient_or_no_matched_sample"},
    ]
    _benjamini_yekutieli(hypotheses, 0.05)
    # m = 3 testable hypotheses (status "insufficient_or_no_matched_sample" is
    # never admitted to the BY family even when it carries a forged p-value).
    harmonic = 1 + 1 / 2 + 1 / 3
    by_id = {item["hypothesisId"]: item for item in hypotheses}
    assert by_id["b"]["adjustedQValue"] == pytest.approx(0.001 * 3 * harmonic / 1)
    assert by_id["c"]["adjustedQValue"] == pytest.approx(0.01 * 3 * harmonic / 2)
    assert by_id["a"]["adjustedQValue"] == pytest.approx(0.5 * 3 * harmonic / 3)
    assert by_id["b"]["passesDeclaredFdr"] is True
    assert by_id["c"]["passesDeclaredFdr"] is True
    assert by_id["a"]["passesDeclaredFdr"] is False
    assert "adjustedQValue" not in by_id["d"]
    assert "passesDeclaredFdr" not in by_id["d"]
