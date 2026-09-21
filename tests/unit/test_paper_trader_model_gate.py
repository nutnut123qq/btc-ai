import json
from unittest.mock import Mock

import pytest

import paper_trader


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


def test_default_mode_is_fail_closed_forward_paper():
    args = paper_trader.build_parser().parse_args([])
    assert args.mode == paper_trader.FORWARD_PAPER_MODE


def test_forward_paper_unavailable_never_opens_database(monkeypatch):
    connect = Mock(side_effect=AssertionError("must not connect"))
    monkeypatch.setattr(paper_trader, "get_conn", connect)

    with pytest.raises(RuntimeError, match="No trade was written"):
        paper_trader.run_forward_paper(["BTCUSDT"])

    connect.assert_not_called()


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
