import json
import math
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock

import pytest

import paper_trader
from forward_paper_recorder import LiveQuote


def test_paper_trader_fails_closed_when_registry_model_is_unavailable(monkeypatch):
    paper_trader._UNAVAILABLE_MODELS.clear()
    loader = Mock(side_effect=RuntimeError("quarantined"))
    monkeypatch.setattr(paper_trader, "load_model", loader)

    first = paper_trader.get_model_for_symbol("BTCUSDT")
    second = paper_trader.get_model_for_symbol("BTCUSDT")

    assert first == (None, "unavailable")
    assert second == (None, "unavailable")
    loader.assert_called_once_with("BTCUSDT", "4h", 5, "4h")


def test_paper_trader_rejects_non_btc_symbol_before_model_loading(monkeypatch):
    loader = Mock()
    monkeypatch.setattr(paper_trader, "load_model", loader)

    with pytest.raises(ValueError, match="Unsupported symbol"):
        paper_trader.get_model_for_symbol("ETHUSDT")

    loader.assert_not_called()


def test_forward_feature_vector_follows_promoted_manifest_schema_and_can_exclude_active_rules():
    cursor = Mock()
    rows = []
    for bar_index in range(5):
        values = [float(bar_index * 100 + index) for index in range(len(paper_trader.FEATURE_COLS))]
        values[paper_trader.FEATURE_COLS.index("ActiveRuleCount")] = 999_999.0
        rows.append((bar_index * paper_trader.FORWARD_PAPER_TIMEFRAME_MS, *values))
    cursor.fetchall.return_value = list(reversed(rows))
    expected_names = [
        f"ws5_bar{bar_index}_{name}"
        for bar_index in range(5)
        for name in paper_trader.FEATURE_COLS + paper_trader.TIME_FEATURE_COLS
        if name != "ActiveRuleCount"
    ]

    _, vector = paper_trader.build_vector_at(
        cursor,
        "BTCUSDT",
        "4h",
        5,
        paper_trader.FORWARD_PAPER_TIMEFRAME_MS,
        rows[-1][0],
        expected_feature_names=expected_names,
    )

    assert len(vector) == 5 * 34
    assert 999_999.0 not in vector


def test_default_mode_is_fail_closed_forward_paper():
    args = paper_trader.build_parser().parse_args([])
    assert args.mode == paper_trader.FORWARD_PAPER_MODE


@pytest.mark.parametrize(
    ("iso_time", "dotnet_day_of_week", "is_weekend"),
    [
        ("2026-09-20T04:00:00+00:00", 0, 1.0),  # Sunday
        ("2026-09-21T04:00:00+00:00", 1, 0.0),  # Monday
        ("2026-09-26T04:00:00+00:00", 6, 1.0),  # Saturday
    ],
)
def test_time_features_match_backend_system_day_of_week(
    iso_time, dotnet_day_of_week, is_weekend
):
    observed = paper_trader.time_features(
        int(datetime.fromisoformat(iso_time).timestamp() * 1000)
    )

    assert observed[0] == pytest.approx(math.sin(2 * math.pi * 4 / 24))
    assert observed[1] == pytest.approx(math.cos(2 * math.pi * 4 / 24))
    assert observed[2] == pytest.approx(math.sin(2 * math.pi * dotnet_day_of_week / 7))
    assert observed[3] == pytest.approx(math.cos(2 * math.pi * dotnet_day_of_week / 7))
    assert observed[4] == is_weekend


def test_forward_paper_records_model_unavailable_as_abstention(monkeypatch):
    conn = Mock()
    cursor = Mock()
    conn.cursor.return_value = cursor
    observed_ms = 1_800_000_000_000
    signal_open_ms = observed_ms - paper_trader.FORWARD_PAPER_TIMEFRAME_MS
    signal_close_ms = observed_ms - 1
    quote = LiveQuote(
        source="test-quote",
        price=Decimal("50000"),
        received_at_utc=paper_trader.datetime.fromtimestamp(observed_ms / 1000, paper_trader.timezone.utc),
        received_time_ms=observed_ms,
    )
    inserted = []

    monkeypatch.setattr(paper_trader, "get_conn", lambda: conn)
    monkeypatch.setattr(paper_trader, "ensure_observation_schema", Mock())
    monkeypatch.setattr(
        paper_trader,
        "_latest_finalized_signal_bar",
        lambda *_args: (signal_open_ms, signal_close_ms),
    )
    monkeypatch.setattr(
        paper_trader,
        "_prospective_decision",
        lambda *_args: ("abstain", None, None, "model-unavailable", {"featureVectorAvailable": False}),
    )
    monkeypatch.setattr(
        paper_trader,
        "insert_observation",
        lambda _conn, _cursor, observation: inserted.append(observation) or True,
    )

    result = paper_trader._run_forward_paper_once(
        ["BTCUSDT"],
        observed_at=paper_trader.datetime.fromtimestamp(observed_ms / 1000, paper_trader.timezone.utc),
        quote_fetcher=lambda _symbol: quote,
        is_prospective=False,
    )

    assert result == {"BTCUSDT": "inserted"}
    assert inserted[0].decision == "abstain"
    assert inserted[0].abstention_reason == "model-unavailable"
    assert inserted[0].available_time_ms >= quote.received_time_ms
    assert inserted[0].evidence_provenance["nextBarOpenRead"] is False
    assert inserted[0].evidence_provenance["fillObserved"] is False
    assert inserted[0].evidence_provenance["isProspectivePaperEvidence"] is False
    cursor.close.assert_called_once()
    conn.close.assert_called_once()


def test_forward_paper_stale_signal_abstains_before_model_inference(monkeypatch):
    conn = Mock()
    cursor = Mock()
    conn.cursor.return_value = cursor
    observed_ms = 1_800_000_000_000
    signal_close_ms = observed_ms - paper_trader.FORWARD_PAPER_MAX_SIGNAL_AGE_MS - 1
    signal_open_ms = signal_close_ms - paper_trader.FORWARD_PAPER_TIMEFRAME_MS + 1
    observed_at = paper_trader.datetime.fromtimestamp(observed_ms / 1000, paper_trader.timezone.utc)
    quote = LiveQuote("test-quote", Decimal("50000"), observed_at, observed_ms)
    recorded = []

    monkeypatch.setattr(paper_trader, "get_conn", lambda: conn)
    monkeypatch.setattr(paper_trader, "ensure_observation_schema", Mock())
    monkeypatch.setattr(
        paper_trader,
        "_latest_finalized_signal_bar",
        lambda *_args: (signal_open_ms, signal_close_ms),
    )
    inference = Mock(side_effect=AssertionError("stale data must not reach inference"))
    monkeypatch.setattr(paper_trader, "_prospective_decision", inference)
    monkeypatch.setattr(
        paper_trader,
        "insert_observation",
        lambda _conn, _cursor, observation: recorded.append(observation) or True,
    )

    result = paper_trader._run_forward_paper_once(
        ["BTCUSDT"],
        observed_at=observed_at,
        quote_fetcher=lambda _symbol: quote,
        is_prospective=False,
    )

    assert result == {"BTCUSDT": "inserted"}
    assert recorded[0].decision == "abstain"
    assert recorded[0].abstention_reason == "signal-data-stale"
    inference.assert_not_called()


def test_public_forward_boundary_rejects_injected_clock_and_quote():
    with pytest.raises(TypeError):
        paper_trader.run_forward_paper(
            ["BTCUSDT"],
            now_utc=datetime.now(paper_trader.timezone.utc),
            quote_fetcher=lambda _symbol: None,
        )


def test_forward_decision_uses_model_classes_and_manifest_mapping(monkeypatch):
    model = Mock()
    model.classes_ = paper_trader.np.array([2, 0, 1])
    model.predict_proba.return_value = paper_trader.np.array([[0.8, 0.1, 0.1]])
    manifest = {
        "feature_names": ["f"],
        "class_mapping": {"0": -1, "1": 0, "2": 1},
    }
    monkeypatch.setattr(
        paper_trader,
        "get_model_bundle_for_symbol",
        lambda _symbol: (model, manifest, "mapped-model"),
    )
    monkeypatch.setattr(
        paper_trader,
        "build_vector_at",
        lambda *_args, **_kwargs: (123, paper_trader.np.array([1.0])),
    )

    decision, confidence, version, reason, evidence = paper_trader._prospective_decision(
        Mock(), "BTCUSDT", 123
    )

    assert decision == "long"
    assert confidence == pytest.approx(0.8)
    assert version == "mapped-model"
    assert reason is None
    assert evidence["predictedClassIndex"] == 0
    assert evidence["predictedModelClass"] == 2
    assert evidence["predictedSemanticClass"] == 1


def test_replay_requires_explicit_mode_and_routes_to_replay_loop(monkeypatch):
    replay = Mock()
    forward = Mock(side_effect=AssertionError("wrong mode"))
    monkeypatch.setattr(paper_trader, "run_replay_loop", replay)
    monkeypatch.setattr(paper_trader, "run_forward_paper", forward)

    paper_trader.main(["--mode", "replay", "--start-ms", "100", "--end-ms", "200"])

    replay.assert_called_once_with(symbols=["BTCUSDT"], start_ms=100, end_ms=200)
    forward.assert_not_called()


def test_replay_provenance_cannot_be_mistaken_for_forward_paper():
    payload = json.loads(paper_trader.build_replay_execution_provenance(1234))
    assert payload["runMode"] == "replay"
    assert payload["signalBarOpenTimeMs"] == 1234
    assert payload["isProspectivePaperEvidence"] is False
    assert payload["fillSource"] == "next-stored-bar-open"
