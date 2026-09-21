import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cleanup_btc_scope import CONFIRMATION_TOKEN, build_delete_specs, parse_args
from trading_config import (
    ACTIVE_PRODUCTION_TIMEFRAMES,
    ACTIVE_SYMBOLS,
    require_active_symbol,
    require_active_symbols,
    require_active_timeframe,
)


def test_production_scope_is_btc_hourly_and_daily_only():
    assert ACTIVE_SYMBOLS == ["BTCUSDT"]
    assert ACTIVE_PRODUCTION_TIMEFRAMES == ("1h", "4h", "1d")
    assert require_active_symbol(" btcusdt ") == "BTCUSDT"
    assert require_active_symbols(["BTCUSDT"]) == ["BTCUSDT"]
    assert require_active_timeframe("4H") == "4h"


@pytest.mark.parametrize("symbol", ["ETHUSDT", "SOLUSDT", "BTC", ""])
def test_non_production_symbols_are_rejected(symbol):
    with pytest.raises(ValueError, match="Unsupported symbol"):
        require_active_symbol(symbol)


@pytest.mark.parametrize("timeframe", ["1m", "5m", "15m", "30m"])
def test_subhourly_candle_timeframes_are_rejected(timeframe):
    with pytest.raises(ValueError, match="Inactive production timeframe"):
        require_active_timeframe(timeframe)


def test_cleanup_plan_never_targets_news_and_preserves_btc_futures_by_predicate():
    specs = build_delete_specs()
    tables = {spec.table for spec in specs}
    assert "NewsArticles" not in tables
    assert "NewsChunks" not in tables
    futures = next(spec for spec in specs if spec.table == "FuturesMetrics")
    market_metrics = next(spec for spec in specs if spec.table == "MarketMetrics")
    assert "Timeframe" not in futures.predicate
    assert "Timeframe" not in market_metrics.predicate
    assert futures.params == ("BTCUSDT",)
    assert market_metrics.params == ("BTCUSDT",)
    assert all(spec.predicate != "TRUE" for spec in specs)


def test_cleanup_is_dry_run_by_default_and_execute_requires_exact_token():
    assert parse_args([]).execute is False
    with pytest.raises(SystemExit):
        parse_args(["--execute"])
    args = parse_args(["--execute", "--confirmation", CONFIRMATION_TOKEN])
    assert args.execute is True
