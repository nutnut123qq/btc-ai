import hashlib
import json
from pathlib import Path

import pytest

import technical_event_modules as modules_under_test
from technical_event_descriptive_evidence import TIMEFRAME_MS, _causal_contexts
from technical_event_modules import build_causal_technical_events, load_contract


def golden_rows(count: int = 240, timeframe: str = "4h"):
    interval = TIMEFRAME_MS[timeframe]
    rows = []
    for index in range(count):
        base = 100_000 + 13 * index + ((index % 12) - 6) * 200
        open_price = base
        close = base + ((index % 5) - 2) * 40
        high = max(open_price, close) + 100 + (index % 3) * 10
        low = min(open_price, close) - 110 - (index % 4) * 10
        volume = (1_000 + (index % 7) * 100) * (4 if index % 29 == 0 else 1)
        rows.append(
            {
                "OpenTimeMs": index * interval,
                "CloseTimeMs": (index + 1) * interval - 1,
                "Open": float(open_price),
                "High": float(high),
                "Low": float(low),
                "Close": float(close),
                "Volume": float(volume),
            }
        )
    return rows


def _build(rows):
    interval = TIMEFRAME_MS["4h"]
    contexts = _causal_contexts(rows, interval)
    return build_causal_technical_events("4h", interval, rows, contexts)


def _event_digest(events):
    payload = [
        {
            "eventId": event["eventId"],
            "module": event["module"],
            "eventType": event["eventType"],
            "availableTimeMs": event["availableTimeMs"],
            "sources": event["sourceCandleOpenTimesMs"],
        }
        for event in events
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _cross_language_semantic_digest(events):
    payload = sorted(
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
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_contract_and_golden_fixture_are_byte_stable():
    contract, contract_hash = load_contract()
    events, modules = _build(golden_rows())
    expected = contract["goldenFixture"]["expected"]
    assert contract_hash == expected["definitionsSha256"]
    assert len(events) == expected["totalEvents"]
    assert _event_digest(events) == expected["orderedEventLedgerSha256"]
    assert _cross_language_semantic_digest(events) == expected["crossLanguageSemanticLedgerSha256"]
    assert {name: value["eventRows"] for name, value in modules.items() if "eventRows" in value} == expected["eventRowsByModule"]


def test_contract_load_fails_closed_when_definitions_do_not_match_golden(tmp_path, monkeypatch):
    contract = json.loads(Path(modules_under_test.CONTRACT_PATH).read_text(encoding="utf-8"))
    contract["modules"]["fibonacci"]["parameters"]["pivotRadiusBars"] = 99
    changed = tmp_path / "contract.json"
    changed.write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.setattr(modules_under_test, "CONTRACT_PATH", changed)
    with pytest.raises(ValueError, match="does not match golden fixture"):
        modules_under_test.load_contract()


def test_contract_load_fails_closed_when_golden_fixture_is_missing(tmp_path, monkeypatch):
    contract = json.loads(Path(modules_under_test.CONTRACT_PATH).read_text(encoding="utf-8"))
    contract.pop("goldenFixture")
    changed = tmp_path / "contract.json"
    changed.write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.setattr(modules_under_test, "CONTRACT_PATH", changed)
    with pytest.raises(ValueError, match="goldenFixture is required"):
        modules_under_test.load_contract()


def test_every_reconstructed_event_has_explicit_availability_sources_and_version():
    rows = golden_rows()
    events, modules = _build(rows)
    close_times = {row["CloseTimeMs"] for row in rows}
    open_times = {row["OpenTimeMs"] for row in rows}
    assert events
    assert set(modules) >= {
        "technicalIndicators",
        "candlePatterns",
        "volumeAnomaly",
        "marketRegime",
        "fibonacci",
        "volumeProfile",
        "confluence",
        "contract",
    }
    for event in events:
        assert event["availableTimeMs"] in close_times
        assert event["confirmedTimeMs"] == event["availableTimeMs"]
        assert event["sourceCandleOpenTimesMs"]
        assert set(event["sourceCandleOpenTimesMs"]) <= open_times
        assert max(event["sourceCandleOpenTimesMs"]) <= event["availableTimeMs"]
        assert event["lineage"]["sourceVersion"]
        assert len(event["lineage"]["contentSha256"]) == 64


def test_crossing_and_transition_sources_cover_previous_and_current_states():
    rows = golden_rows()
    events, _ = _build(rows)
    interval = TIMEFRAME_MS["4h"]
    by_type = {}
    for event in events:
        by_type.setdefault(event["eventType"], []).append(event)

    for event_type in ("EMA_BULL_CROSS", "EMA_BEAR_CROSS"):
        assert {len(event["sourceCandleOpenTimesMs"]) for event in by_type[event_type]} == {61}
    for event_type in ("CLOSE_ABOVE_SMA50", "CLOSE_BELOW_SMA50"):
        assert {len(event["sourceCandleOpenTimesMs"]) for event in by_type[event_type]} == {51}
    for event_type, group in by_type.items():
        if event_type.startswith("VOLUME_PROFILE_CLOSE_"):
            assert {len(event["sourceCandleOpenTimesMs"]) for event in group} == {101}
        if event_type.startswith("REGIME_"):
            for event in group:
                index = event["availableTimeMs"] // interval
                expected_first = max(0, index - 21) * interval
                sources = event["sourceCandleOpenTimesMs"]
                assert sources[0] == expected_first
                assert sources[-1] == index * interval
                assert len(sources) == index - max(0, index - 21) + 1
        if event_type.startswith("FIBONACCI_LEG_"):
            for event in group:
                sources = event["sourceCandleOpenTimesMs"]
                assert sources[-1] == (event["availableTimeMs"] // interval) * interval
                assert all(right - left == interval for left, right in zip(sources, sources[1:]))


def test_rsi_cross_source_is_period_plus_two_candles():
    interval = TIMEFRAME_MS["4h"]
    closes = [100 + (index % 2) * 0.2 for index in range(61)]
    closes += [100 - index * 2 for index in range(1, 16)]
    closes += [70 + index * 3 for index in range(1, 25)]
    rows = [
        {
            "OpenTimeMs": index * interval,
            "CloseTimeMs": (index + 1) * interval - 1,
            "Open": close + 0.1,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 1_000.0,
        }
        for index, close in enumerate(closes)
    ]
    events, _ = _build(rows)
    rsi_events = [event for event in events if event["eventType"].startswith("RSI_")]
    assert rsi_events
    assert {len(event["sourceCandleOpenTimesMs"]) for event in rsi_events} == {16}


def test_appending_future_candles_does_not_rewrite_existing_events():
    prefix = golden_rows(220)
    full = golden_rows(240)
    before, _ = _build(prefix)
    after, _ = _build(full)
    cutoff = prefix[-1]["CloseTimeMs"]
    after_prefix = [event for event in after if event["availableTimeMs"] <= cutoff]
    assert before == after_prefix


def test_gap_resets_all_module_sources():
    rows = golden_rows()
    interval = TIMEFRAME_MS["4h"]
    for index in range(120, len(rows)):
        rows[index] = {
            **rows[index],
            "OpenTimeMs": rows[index]["OpenTimeMs"] + interval,
            "CloseTimeMs": rows[index]["CloseTimeMs"] + interval,
        }
    events, _ = _build(rows)
    gap_left = rows[119]["OpenTimeMs"]
    gap_right = rows[120]["OpenTimeMs"]
    for event in events:
        sources = event["sourceCandleOpenTimesMs"]
        assert not (min(sources) <= gap_left and max(sources) >= gap_right)
