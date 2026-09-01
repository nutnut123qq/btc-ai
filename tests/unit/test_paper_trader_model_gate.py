from unittest.mock import Mock

import paper_trader


def test_paper_trader_fails_closed_when_registry_model_is_unavailable(monkeypatch):
    paper_trader._UNAVAILABLE_MODELS.clear()
    loader = Mock(side_effect=RuntimeError("quarantined"))
    monkeypatch.setattr(paper_trader, "load_model", loader)

    first = paper_trader.get_model_for_symbol("ETHUSDT")
    second = paper_trader.get_model_for_symbol("ETHUSDT")

    assert first == (None, "unavailable")
    assert second == (None, "unavailable")
    loader.assert_called_once_with("ETHUSDT", "4h", 5, "4h")
