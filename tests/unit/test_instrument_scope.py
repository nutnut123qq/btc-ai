"""Shared instrument contract tests (MBB-01): registry, per-symbol timeframes."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trading_config import (
    ACTIVE_PRODUCTION_TIMEFRAMES,
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
    assert KNOWN_SYMBOLS == ("BTCUSDT", "MBB")

    btc = INSTRUMENTS["BTCUSDT"]
    assert btc["asset_class"] == "crypto"
    assert btc["currency"] == "USDT"
    assert btc["timezone"] == "UTC"
    assert btc["data_provider"] == "Binance"
    assert btc["price_basis"] == "raw"
    assert btc["enabled"] is True
    assert btc["timeframes"] == ACTIVE_PRODUCTION_TIMEFRAMES == ("1h", "4h", "1d")
    assert btc["default_timeframe"] == DEFAULT_TIMEFRAME == "4h"
    assert all(btc["capabilities"].values())

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
    assert all(
        not enabled
        for name, enabled in mbb["capabilities"].items()
        if name != "technical_analysis"
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
