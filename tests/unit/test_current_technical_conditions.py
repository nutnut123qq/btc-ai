"""Fixture-based tests for ``current_technical_conditions`` (no DB, no network)."""

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import current_technical_conditions as ctc
from current_technical_conditions import (
    CANONICAL_SMC_COLUMNS,
    TIMEFRAME_MS,
    WINDOW_BARS,
    build_conditions_payload,
)
from technical_event_descriptive_evidence import _causal_contexts
from technical_event_modules import _direction, build_causal_technical_events


INTERVAL = TIMEFRAME_MS["4h"]
START = 1_700_000_000_000


def _row(index, open_, high, low, close, volume=10.0, *, interval=INTERVAL, start=START):
    return {
        "OpenTimeMs": start + index * interval,
        "CloseTimeMs": start + (index + 1) * interval - 1,
        "Open": float(open_),
        "High": float(high),
        "Low": float(low),
        "Close": float(close),
        "Volume": float(volume),
    }


def _uptrend_rows(count=140, *, interval=INTERVAL, start=START, first_close=100.0):
    rows = []
    for i in range(count):
        close = first_close + i * 0.5
        open_ = close - 0.25
        rows.append(_row(i, open_, max(open_, close) + 0.4, min(open_, close) - 0.4, close,
                         interval=interval, start=start))
    return rows


def _smc_row(event_id, event_type, origin_ms, available_ms, source_times, *,
             high=None, low=None, price=100.0, interval=INTERVAL):
    version = "smc-causal-v2"
    evidence = {
        "eventId": event_id,
        "eventType": event_type,
        "description": "fixture",
        "originTimeMs": origin_ms,
        "availableTimeMs": available_ms,
        "referenceTimeMs": None,
        "price": price,
        "highPrice": high,
        "lowPrice": low,
        "calculationVersion": version,
        "stateAtAsOf": "active" if event_type.startswith("FVG_") else "confirmed",
        "mitigatedAtMs": None,
        "invalidatedAtMs": None,
        "mitigationRule": None,
        "invalidationRule": None,
        "sourceCandles": [
            {
                "role": "decision-source",
                "openTimeMs": t,
                "closeTimeMs": t + interval - 1,
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 1.0,
            }
            for t in source_times
        ],
        "detectionConditions": [],
        "limitations": [],
    }
    evidence_json = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
    return {
        "EventId": event_id,
        "EventType": event_type,
        "OriginTimeMs": origin_ms,
        "AvailableTimeMs": available_ms,
        "CalculationVersion": version,
        "DecisionSourceCandleCount": len(source_times),
        "DecisionSourceOpenTimeMsJson": json.dumps(source_times, separators=(",", ":")),
        "DecisionEvidenceJson": evidence_json,
        "DecisionEvidenceSha256": hashlib.sha256(evidence_json.encode()).hexdigest(),
    }


def _fvg_row(rows, decision_index, zone_low, zone_high, *, event_type="FVG_BULL",
             event_id="fvg-1"):
    origin = int(rows[decision_index - 1]["OpenTimeMs"])
    available = int(rows[decision_index]["CloseTimeMs"])
    source_times = [
        int(rows[decision_index - 2]["OpenTimeMs"]),
        origin,
        int(rows[decision_index]["OpenTimeMs"]),
    ]
    mid = (zone_low + zone_high) / 2.0
    return _smc_row(
        event_id, event_type, origin, available, source_times,
        high=zone_high, low=zone_low, price=mid,
    )


def _conditions_by_kind(payload, kind):
    return [c for c in payload["conditions"] if c["kind"] == kind]


def test_triggered_on_bar_requires_equality_with_latest_close():
    rows = _uptrend_rows(140)
    # Force a DOJI on an early bar and on the latest bar.
    for index in (10, 139):
        row = rows[index]
        row["Open"] = row["Close"] = 100.0
        row["High"] = 102.0
        row["Low"] = 98.0
    now = int(rows[-1]["CloseTimeMs"])
    payload = build_conditions_payload("4h", rows, [], sorted(CANONICAL_SMC_COLUMNS), now_ms=now)

    events, _ = build_causal_technical_events("4h", INTERVAL, rows, _causal_contexts(rows, INTERVAL))
    earlier_ids = {
        e["eventId"] for e in events if int(e["availableTimeMs"]) < payload["asOfMs"]
    }
    assert earlier_ids, "fixture must produce events on earlier bars"
    triggered = _conditions_by_kind(payload, "triggeredOnBar")
    assert triggered, "latest-bar DOJI must surface as triggeredOnBar"
    assert all(c["availableTimeMs"] == payload["asOfMs"] for c in triggered)
    assert earlier_ids.isdisjoint(c["eventId"] for c in triggered)
    assert any(c["eventType"] == "DOJI" for c in triggered)


def test_state_conditions_on_uptrend_window():
    rows = _uptrend_rows(140)
    now = int(rows[-1]["CloseTimeMs"])
    payload = build_conditions_payload("4h", rows, [], sorted(CANONICAL_SMC_COLUMNS), now_ms=now)
    states = {c["eventType"]: c for c in _conditions_by_kind(payload, "state")}

    assert "REGIME_BULL_NORMAL" in states
    assert states["REGIME_BULL_NORMAL"]["module"] == "marketRegime"
    assert states["REGIME_BULL_NORMAL"]["direction"] == "bullish"

    assert "CLOSE_ABOVE_SMA50" in states
    assert states["CLOSE_ABOVE_SMA50"]["direction"] == "bullish"
    assert states["CLOSE_ABOVE_SMA50"]["details"]["sma"] < rows[-1]["Close"]

    assert "EMA_FAST_ABOVE_SLOW" in states
    assert states["EMA_FAST_ABOVE_SLOW"]["direction"] == "bullish"

    assert "RSI_ZONE_OVERBOUGHT" in states
    assert states["RSI_ZONE_OVERBOUGHT"]["details"]["rsi"] == 100.0

    for label in ("POC", "VAH", "VAL"):
        assert f"VOLUME_PROFILE_CLOSE_ABOVE_{label}" in states

    for condition in states.values():
        assert condition["kind"] == "state"
        assert condition["eventId"] is None
        assert condition["availableTimeMs"] == payload["asOfMs"]
        assert condition["formedTimeMs"] == payload["asOfMs"]


def test_fvg_mitigation_uses_wick_low_not_close():
    rows = _uptrend_rows(140)
    # Zone low is 96. Bar 60 wicks to 95.5 (below zone low) but closes at 130 —
    # the wick must mitigate the zone even though the close never crossed it.
    zone_low, zone_high = 96.0, 99.0
    rows[60]["Low"] = 95.5
    smc = _fvg_row(rows, 30, zone_low, zone_high)
    payload = build_conditions_payload(
        "4h", rows, [smc], list(smc), now_ms=int(rows[-1]["CloseTimeMs"])
    )
    zones = _conditions_by_kind(payload, "activeZone")
    assert not any(c["eventId"] == "fvg-1" for c in zones)


def test_fvg_zone_stays_active_when_wick_never_reaches():
    rows = _uptrend_rows(140)
    # All later lows stay above the zone low → mitigation never happens.
    zone_low = min(r["Low"] for r in rows[31:]) - 1.0
    smc = _fvg_row(rows, 30, zone_low, zone_low + 1.0)
    payload = build_conditions_payload(
        "4h", rows, [smc], list(smc), now_ms=int(rows[-1]["CloseTimeMs"])
    )
    zones = [c for c in _conditions_by_kind(payload, "activeZone") if c["eventId"] == "fvg-1"]
    assert len(zones) == 1
    zone = zones[0]
    assert zone["eventType"] == "FVG_BULL"
    assert zone["direction"] == "bullish"
    assert zone["details"]["zoneLowPrice"] == zone_low
    assert zone["details"]["semanticsVersion"] == "smc-causal-fvg-zone-v1"
    assert zone["details"]["decisionPredatesWindow"] is False


def test_fvg_bear_mitigated_by_high_wick():
    rows = _uptrend_rows(140)
    rows[60]["High"] = 150.0  # wick crosses zone high while close stays below
    smc = _fvg_row(rows, 30, 140.0, 145.0, event_type="FVG_BEAR", event_id="fvg-bear")
    payload = build_conditions_payload(
        "4h", rows, [smc], list(smc), now_ms=int(rows[-1]["CloseTimeMs"])
    )
    assert not any(c["eventId"] == "fvg-bear" for c in _conditions_by_kind(payload, "activeZone"))


def test_fvg_decision_predating_window_is_honest_not_unmitigated():
    rows = _uptrend_rows(140)
    # Decision candle is before the loaded window entirely.
    smc = _smc_row(
        "fvg-old", "FVG_BULL",
        START - 2 * INTERVAL,
        START - INTERVAL - 1,          # close before first window bar
        [START - 3 * INTERVAL, START - 2 * INTERVAL, START - INTERVAL],
        high=60.0, low=50.0,
    )
    payload = build_conditions_payload(
        "4h", rows, [smc], list(smc), now_ms=int(rows[-1]["CloseTimeMs"])
    )
    zones = [c for c in _conditions_by_kind(payload, "activeZone") if c["eventId"] == "fvg-old"]
    assert len(zones) == 1
    assert zones[0]["details"]["decisionPredatesWindow"] is True
    assert "unobservable" in zones[0]["details"]["mitigationObservation"]


def test_smc_hash_mismatch_marks_whole_module_unavailable():
    rows = _uptrend_rows(140)
    smc = _fvg_row(rows, 30, 50.0, 60.0)
    smc["DecisionEvidenceSha256"] = "0" * 64
    payload = build_conditions_payload(
        "4h", rows, [smc], list(smc), now_ms=int(rows[-1]["CloseTimeMs"])
    )
    assert payload["unavailableModules"]
    assert payload["unavailableModules"][0]["module"] == "causalSmc"
    assert "hash" in payload["unavailableModules"][0]["reason"]
    assert not any(c["module"] == "causalSmc" for c in payload["conditions"])


def test_smc_missing_canonical_schema_is_unavailable():
    rows = _uptrend_rows(140)
    payload = build_conditions_payload(
        "4h", rows, [], ["EventId", "EventType"], now_ms=int(rows[-1]["CloseTimeMs"])
    )
    assert payload["unavailableModules"] == [
        {
            "module": "causalSmc",
            "reason": "CausalSmartMoneyEvents canonical schema is unavailable",
        }
    ]


def test_gap_truncates_window_with_warnings():
    head = _uptrend_rows(50)
    tail_start = START + 60 * INTERVAL  # 10-bar gap between head and tail
    tail = _uptrend_rows(70, start=tail_start, first_close=200.0)
    rows = head + tail
    payload = build_conditions_payload(
        "4h", rows, [], sorted(CANONICAL_SMC_COLUMNS), now_ms=int(tail[-1]["CloseTimeMs"])
    )
    assert payload["analysisWindow"]["bars"] == 70
    assert payload["analysisWindow"]["firstOpenTimeMs"] == tail[0]["OpenTimeMs"]
    assert payload["analysisWindow"]["contiguous"] is True
    assert "analysis-window-truncated-at-non-contiguous-gap" in payload["warnings"]
    assert "fibonacci-anchor-may-predate-window" in payload["warnings"]


def test_window_cap_truncates_and_warns():
    rows = _uptrend_rows(140)
    payload = build_conditions_payload(
        "4h", rows, [], sorted(CANONICAL_SMC_COLUMNS),
        now_ms=int(rows[-1]["CloseTimeMs"]), window_bars=100,
    )
    assert payload["analysisWindow"]["bars"] == 100
    assert payload["analysisWindow"]["firstOpenTimeMs"] == rows[40]["OpenTimeMs"]
    assert "fibonacci-anchor-may-predate-window" in payload["warnings"]
    assert "analysis-window-truncated-at-non-contiguous-gap" not in payload["warnings"]


def test_in_flight_bar_is_never_used():
    rows = _uptrend_rows(140)
    in_flight = _row(140, rows[-1]["Close"], rows[-1]["Close"] + 1, rows[-1]["Close"] - 1,
                   rows[-1]["Close"])
    now = int(rows[-1]["CloseTimeMs"])  # in-flight bar has CloseTimeMs > now
    assert in_flight["CloseTimeMs"] > now
    payload = build_conditions_payload(
        "4h", rows + [in_flight], [], sorted(CANONICAL_SMC_COLUMNS), now_ms=now
    )
    assert payload["asOfMs"] == int(rows[-1]["CloseTimeMs"])
    assert payload["latestClosedBar"]["openTimeMs"] == rows[-1]["OpenTimeMs"]


def test_operative_leg_is_most_recent_fibonacci_leg():
    rows = _uptrend_rows(140)
    # Deep low wick → pivot LOW at 30 (strictly below neighbors ±2); tall high
    # wick → pivot HIGH at 40 (strictly above neighbors ±2). Confirmed leg at
    # decision bar 42: FIBONACCI_LEG_BULL.
    rows[30]["Low"] = rows[30]["Low"] - 50.0
    rows[40]["High"] = rows[40]["High"] + 50.0
    payload = build_conditions_payload(
        "4h", rows, [], sorted(CANONICAL_SMC_COLUMNS), now_ms=int(rows[-1]["CloseTimeMs"])
    )
    legs = _conditions_by_kind(payload, "operativeLeg")
    assert len(legs) == 1
    leg = legs[0]
    assert leg["eventType"] == "FIBONACCI_LEG_BULL"
    assert leg["direction"] == "bullish"
    assert leg["availableTimeMs"] == int(rows[42]["CloseTimeMs"])
    assert leg["availableTimeMs"] <= payload["asOfMs"]
    assert leg["eventId"]


def test_direction_mapping_uses_contract_tokens():
    rows = _uptrend_rows(140)
    payload = build_conditions_payload(
        "4h", rows, [], sorted(CANONICAL_SMC_COLUMNS), now_ms=int(rows[-1]["CloseTimeMs"])
    )
    expected = {1: "bullish", -1: "bearish", 0: "neutral"}
    for condition in payload["conditions"]:
        assert condition["direction"] == expected[_direction(condition["eventType"])]
    # Spot-check the token mapping survives through the payload.
    assert _direction("CLOSE_ABOVE_SMA50") == 1
    assert _direction("CLOSE_BELOW_SMA50") == -1
    assert _direction("FIBONACCI_LEG_BEAR") == -1


def test_bad_timeframe_rejected():
    with pytest.raises(ctc.UnknownTimeframeError):
        build_conditions_payload("15m", [], [], [], now_ms=START)
    with pytest.raises(ctc.UnknownTimeframeError):
        ctc.get_current_conditions("5m")


def test_no_finalized_bars_is_unavailable():
    with pytest.raises(ctc.ConditionsUnavailableError):
        build_conditions_payload("4h", [], [], sorted(CANONICAL_SMC_COLUMNS), now_ms=START)


class TestEndpoint:
    def setup_method(self):
        ctc.clear_cache()
        self.client = TestClient(__import__("main").app)

    def test_bad_timeframe_returns_400(self):
        response = self.client.get("/api/current-conditions?timeframe=15m")
        assert response.status_code == 400

    def test_contract_shape_with_mocked_db(self):
        rows = _uptrend_rows(140)
        smc = _fvg_row(rows, 30, 50.0, 60.0)
        with patch.object(
            ctc, "_load_inputs", return_value=(rows, [smc], list(smc), None)
        ):
            response = self.client.get("/api/current-conditions?timeframe=4h")
        assert response.status_code == 200
        payload = response.json()
        assert payload["schemaVersion"] == "current-conditions-v1"
        assert payload["symbol"] == "BTCUSDT"
        assert payload["timeframe"] == "4h"
        assert payload["asOfMs"] == int(rows[-1]["CloseTimeMs"])
        for key in ("openTimeMs", "closeTimeMs", "open", "high", "low", "close", "volume"):
            assert key in payload["latestClosedBar"]
        assert set(payload["analysisWindow"]) == {
            "firstOpenTimeMs", "bars", "contiguous", "windowBars"
        }
        assert payload["analysisWindow"]["windowBars"] == WINDOW_BARS
        assert set(payload["contract"]) == {
            "contractVersion", "definitionsSha256", "rawFileSha256"
        }
        assert isinstance(payload["generatedAtMs"], int)
        assert isinstance(payload["conditions"], list)
        assert isinstance(payload["warnings"], list)
        assert isinstance(payload["unavailableModules"], list)
        for condition in payload["conditions"]:
            assert {"module", "eventType", "kind", "direction", "eventId",
                    "formedTimeMs", "availableTimeMs", "context"} <= set(condition)
            assert condition["kind"] in {
                "triggeredOnBar", "state", "activeZone", "operativeLeg"
            }
        assert any(c["kind"] == "activeZone" for c in payload["conditions"])

    def test_no_klines_returns_503(self):
        with patch.object(ctc, "_load_inputs", return_value=([], [], [], None)):
            response = self.client.get("/api/current-conditions?timeframe=4h")
        assert response.status_code == 503
        assert response.json()["detail"]

    def test_cache_hit_returns_same_payload_except_generated_at(self):
        rows = _uptrend_rows(140)
        loader = patch.object(ctc, "_load_inputs", return_value=(rows, [], [], None))
        with loader as mocked:
            first = self.client.get("/api/current-conditions?timeframe=4h").json()
            second = self.client.get("/api/current-conditions?timeframe=4h").json()
        # Klines are still read each request to learn asOfMs, but the payload
        # body is served from the (timeframe, asOfMs) cache.
        assert {k: v for k, v in first.items() if k != "generatedAtMs"} == {
            k: v for k, v in second.items() if k != "generatedAtMs"
        }
