"""Shared instrument contract tests (MBB-01): registry, per-symbol timeframes."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trading_config import (
    ACTIVE_PRODUCTION_TIMEFRAMES,
    ACTIVE_SYMBOLS,
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    INSTRUMENTS,
    KNOWN_SYMBOLS,
    active_timeframes_for,
    canonicalize_symbol,
    default_timeframe_for,
    require_active_timeframe,
    require_known_symbol,
)


def test_registry_declares_btc_and_mbb_contract():
    assert KNOWN_SYMBOLS == ("BTCUSDT", "MBB", "BID", "VCB", "CTG", "VNINDEX")

    btc = INSTRUMENTS["BTCUSDT"]
    assert btc["asset_class"] == "crypto"
    assert btc["currency"] == "USDT"
    assert btc["timezone"] == "UTC"
    assert btc["data_provider"] == "Binance"
    assert btc["price_basis"] == "raw"
    assert btc["enabled"] is True
    assert btc["timeframes"] == ACTIVE_PRODUCTION_TIMEFRAMES == ("1h", "4h", "1d")
    assert btc["default_timeframe"] == DEFAULT_TIMEFRAME == "4h"
    crypto_flags = {name: v for name, v in btc["capabilities"].items()
                    if name not in ("ingestion", "decision")}
    assert all(crypto_flags.values())
    # Equity-lane flags are false for the Binance lane (it has its own ingestion).
    assert btc["capabilities"]["ingestion"] is False
    assert btc["capabilities"]["decision"] is False

    mbb = INSTRUMENTS["MBB"]
    assert mbb["asset_class"] == "equity"
    assert mbb["venue"] == "HOSE"
    assert mbb["currency"] == "VND"
    assert mbb["timezone"] == "Asia/Ho_Chi_Minh"
    assert mbb["data_provider"] == "VCI"
    assert mbb["cross_check_provider"] == "KBS"
    assert mbb["price_basis"] == "adjusted"
    assert mbb["enabled"] is False
    assert mbb["timeframes"] == ("1h", "1d")  # no synthetic 4h
    assert mbb["default_timeframe"] == "1d"
    assert mbb["capabilities"]["technical_analysis"] is True
    assert mbb["capabilities"]["ingestion"] is True
    assert mbb["capabilities"]["decision"] is True
    assert all(
        not enabled
        for name, enabled in mbb["capabilities"].items()
        if name not in ("technical_analysis", "ingestion", "decision")
    )


def test_canonicalize_symbol_resolves_aliases_and_case():
    assert canonicalize_symbol(" btc ") == "BTCUSDT"
    assert canonicalize_symbol("BTCUSDT") == "BTCUSDT"
    assert canonicalize_symbol("mbb") == "MBB"
    assert canonicalize_symbol("ETHUSDT") == "ETHUSDT"  # unknown passes through


def test_require_known_symbol_fails_closed():
    assert require_known_symbol("MBB") == "MBB"
    assert require_known_symbol("BTC") == "BTCUSDT"
    for symbol in ("ETHUSDT", "SOLUSDT", ""):
        with pytest.raises(ValueError, match="Unknown symbol"):
            require_known_symbol(symbol)


def test_symbol_aware_timeframes_default_to_btc_scope():
    assert require_active_timeframe("4H") == "4h"
    assert require_active_timeframe("1d") == "1d"
    with pytest.raises(ValueError, match="Inactive production timeframe"):
        require_active_timeframe("15m")


def test_mbb_accepts_only_1h_and_1d():
    assert require_active_timeframe("1h", "MBB") == "1h"
    assert require_active_timeframe(" 1D ", "mbb") == "1d"
    assert active_timeframes_for("MBB") == ("1h", "1d")
    assert default_timeframe_for("MBB") == "1d"
    for timeframe in ("4h", "15m", "1w"):
        with pytest.raises(ValueError, match="Inactive production timeframe"):
            require_active_timeframe(timeframe, "MBB")


def test_symbol_aware_timeframes_fail_closed_on_unknown_symbols():
    with pytest.raises(ValueError, match="Unknown symbol"):
        require_active_timeframe("1d", "ETHUSDT")
    with pytest.raises(ValueError, match="Unknown symbol"):
        active_timeframes_for("DOGEUSDT")
    with pytest.raises(ValueError, match="Unknown symbol"):
        default_timeframe_for("")
    assert default_timeframe_for(DEFAULT_SYMBOL) == "4h"


# --- EQX: registry parity with backend InstrumentCatalog (spec §1, instrument-catalog-v2) ---

# symbol -> (asset_class, currency, price_basis, timeframes, ingestion, technical_analysis, decision)
_SPEC_SCOPE = {
    "BTCUSDT": ("crypto", "USDT", "raw", ("1h", "4h", "1d"), False, True, False),
    "MBB": ("equity", "VND", "adjusted", ("1h", "1d"), True, True, True),
    "BID": ("equity", "VND", "adjusted", ("1d",), True, True, True),
    "VCB": ("equity", "VND", "adjusted", ("1d",), True, False, False),
    "CTG": ("equity", "VND", "adjusted", ("1d",), True, False, False),
    "VNINDEX": ("index", "index-points", "raw", ("1d",), True, False, False),
}
_CRYPTO_ONLY_FLAGS = ("news", "sentiment", "futures", "liquidation", "order_book",
                      "execution", "paper_trading")


@pytest.mark.parametrize("symbol", sorted(_SPEC_SCOPE))
def test_capabilities_parity_with_backend_catalog(symbol):
    asset_class, currency, basis, timeframes, ingestion, ta, decision = _SPEC_SCOPE[symbol]
    descriptor = INSTRUMENTS[symbol]
    caps = descriptor["capabilities"]
    assert descriptor["asset_class"] == asset_class
    assert descriptor["currency"] == currency
    assert descriptor["price_basis"] == basis
    assert descriptor["timeframes"] == timeframes
    assert caps["ingestion"] is ingestion
    assert caps["technical_analysis"] is ta
    assert caps["decision"] is decision
    if symbol != "BTCUSDT":
        # Crypto flags stay false for every equity/index; none is enabled for crypto traffic.
        assert descriptor["enabled"] is False
        assert all(caps[name] is False for name in _CRYPTO_ONLY_FLAGS)
        assert descriptor["default_timeframe"] == "1d"
    # Decision instruments always carry technical analysis (spec §1).
    if decision:
        assert ta is True


def test_bid_is_daily_only():
    assert require_active_timeframe("1d", "BID") == "1d"
    assert require_active_timeframe(" 1D ", "bid") == "1d"
    assert active_timeframes_for("BID") == ("1d",)
    assert default_timeframe_for("BID") == "1d"
    for timeframe in ("1h", "4h", "15m"):
        with pytest.raises(ValueError, match="Inactive production timeframe"):
            require_active_timeframe(timeframe, "BID")


def test_vnindex_is_an_index_in_points_without_vnd_costs():
    vnindex = INSTRUMENTS["VNINDEX"]
    assert vnindex["asset_class"] == "index"
    assert vnindex["currency"] == "index-points"
    assert vnindex["price_basis"] == "raw"
    assert vnindex["capabilities"]["decision"] is False
    assert require_active_timeframe("1d", "VNINDEX") == "1d"
    with pytest.raises(ValueError, match="Inactive production timeframe"):
        require_active_timeframe("1h", "VNINDEX")


def test_active_symbols_stay_btc_only():
    from trading_config import require_active_symbol

    assert ACTIVE_SYMBOLS == ["BTCUSDT"]
    for symbol in ("BID", "MBB", "VCB", "CTG", "VNINDEX"):
        assert require_known_symbol(symbol) == symbol
        with pytest.raises(ValueError, match="Unsupported symbol"):
            require_active_symbol(symbol)
