from copy import deepcopy
import json
from pathlib import Path

from technical_evidence_profiles import build_evidence_profiles


INTERVAL = 3_600_000
YEAR_2025 = 1_735_689_600_000


def _horizons(value):
    return {
        str(horizon): {
            "forwardReturn": value,
            "mfe": max(value, 0.01),
            "mae": min(value, -0.01),
        }
        for horizon in (1, 3, 6)
    }


def _row(module, event_type, close, value, *, status="eligible", context=None):
    return {
        "module": module,
        "eventType": event_type,
        "status": status,
        "exclusionReason": None if status == "eligible" else "unknown_event_lineage",
        "decisionCloseTimeMs": close if status == "eligible" else None,
        "context": context or {"trend": "up", "volatility": "normal"},
        "horizons": _horizons(value) if status == "eligible" else {},
        "control": {"status": "matched"} if status == "eligible" else {"status": "not_attempted"},
    }


def _fixture():
    first = YEAR_2025 + INTERVAL - 1
    second = YEAR_2025 + 3 * INTERVAL - 1
    rows = [
        _row("technicalIndicators", "EMA_BULL_CROSS", first, 0.02),
        _row("candlePatterns", "DOJI", first, -0.01),
        _row("confluence", "CONFLUENCE_BULL_2PLUS", first, -0.005),
        _row("volumeAnomaly", "VOLUME_ANOMALY_2_0X", second, -0.03, context={"trend": "down", "volatility": "high"}),
        _row("fibonacci", "FIBONACCI_LEG_BEAR", second, 0.0, status="excluded"),
    ]
    candles = [
        {"openTimeMs": YEAR_2025, "closeTimeMs": first},
        {"openTimeMs": YEAR_2025 + 2 * INTERVAL, "closeTimeMs": second},
    ]
    modules = {
        name: {"status": "evaluable", "eventRows": sum(row["module"] == name for row in rows)}
        for name in ("technicalIndicators", "candlePatterns", "volumeAnomaly", "marketRegime", "fibonacci", "volumeProfile", "confluence")
    }
    sensitivity = {
        "variants": [
            {
                "module": "technicalIndicators",
                "variantId": "rsiBands=30_70",
                "eligible": 3,
                "realizedAtMaxHorizon": 3,
                "horizons": {"6": {"metrics": {"forwardReturn": {"variant": {"mean": -0.02}}}}},
            },
            {
                "module": "volumeProfile",
                "variantId": "bins=32",
                "eligible": 0,
                "realizedAtMaxHorizon": 0,
                "horizons": {"6": {"metrics": {"forwardReturn": {"variant": None}}}},
            },
        ]
    }
    return rows, candles, modules, sensitivity


def test_profiles_keep_fixed_dimensions_missingness_and_negative_results():
    rows, candles, modules, sensitivity = _fixture()
    result = build_evidence_profiles(
        rows,
        candles,
        timeframe="1h",
        interval_ms=INTERVAL,
        modules=modules,
        horizons=(1, 3, 6),
        sensitivity=sensitivity,
    )

    assert set(result["breakdowns"]) == {"yearUtc", "timeframe", "regime"}
    assert result["breakdowns"]["yearUtc"]["2025"]["stored"] == 4
    assert result["breakdowns"]["yearUtc"]["unavailable"]["excluded"] == 1
    assert result["coverageAndMissingness"]["candles"]["gapCount"] == 1
    assert result["coverageAndMissingness"]["candles"]["estimatedMissingBars"] == 1
    assert result["coverageAndMissingness"]["events"]["withoutKnownLineage"] == 1
    retained = result["negativeResultRetention"]
    assert retained["thresholdSelectionAllowed"] is False
    assert retained["eventTypes"]["candlePatterns:DOJI"]["retentionClassification"] == "non_positive_mean_retained"
    assert retained["eventTypes"]["fibonacci:FIBONACCI_LEG_BEAR"]["retentionClassification"] == "no_realized_sample_retained"
    assert retained["sensitivityVariants"]["technicalIndicators:rsiBands=30_70"]["retentionClassification"] == "non_positive_mean_retained"
    assert retained["sensitivityVariants"]["volumeProfile:bins=32"]["retentionClassification"] == "no_realized_sample_retained"


def test_profiles_disclose_overlap_unique_and_confluence_dependence_deterministically():
    rows, candles, modules, sensitivity = _fixture()
    first = build_evidence_profiles(
        rows, candles, timeframe="1h", interval_ms=INTERVAL,
        modules=modules, horizons=(1, 3, 6), sensitivity=sensitivity,
    )
    second = build_evidence_profiles(
        deepcopy(rows), deepcopy(candles), timeframe="1h", interval_ms=INTERVAL,
        modules=deepcopy(modules), horizons=(1, 3, 6), sensitivity=deepcopy(sensitivity),
    )

    dependence = first["overlapAndDependence"]
    assert dependence["pairwiseSameCloseCounts"]["candlePatterns|technicalIndicators"] == 1
    assert dependence["uniqueDecisionClosesByModule"]["volumeAnomaly"] == 1
    assert dependence["confluence"]["rowsWithFewerThanTwoBaseModules"] == 0
    assert dependence["independenceClaimed"] is False
    assert first == second


def test_module_profiles_do_not_mix_same_event_type_or_opposite_results():
    rows, candles, modules, sensitivity = _fixture()
    close = rows[0]["decisionCloseTimeMs"]
    rows.extend([
        _row("technicalIndicators", "SHARED_LABEL", close, 0.05),
        _row("candlePatterns", "SHARED_LABEL", close, -0.05),
    ])
    result = build_evidence_profiles(
        rows, candles, timeframe="1h", interval_ms=INTERVAL,
        modules=modules, horizons=(1, 3, 6), sensitivity=sensitivity,
    )
    indicator = result["byModule"]["technicalIndicators"]
    patterns = result["byModule"]["candlePatterns"]
    assert indicator["breakdowns"]["timeframe"]["1h"]["horizons"]["6"]["metrics"]["forwardReturn"]["mean"] > 0
    assert patterns["breakdowns"]["timeframe"]["1h"]["horizons"]["6"]["metrics"]["forwardReturn"]["mean"] < 0
    retained = result["negativeResultRetention"]["eventTypes"]
    assert retained["technicalIndicators:SHARED_LABEL"]["retentionClassification"] == "positive_mean_retained"
    assert retained["candlePatterns:SHARED_LABEL"]["retentionClassification"] == "non_positive_mean_retained"


def test_machine_readable_schema_and_example_are_versioned_and_hash_bound():
    contract_dir = Path(__file__).resolve().parents[2] / "contracts"
    schema = json.loads((contract_dir / "technical-evidence-profile.schema.json").read_text(encoding="utf-8"))
    example = json.loads((contract_dir / "technical-evidence-profile.example.json").read_text(encoding="utf-8"))
    assert schema["$id"] == example["schema"] == "btc-technical-evidence-profiles/v1"
    assert "byModule" in schema["required"]
    assert len(example["profileDefinitionsSha256"]) == 64
