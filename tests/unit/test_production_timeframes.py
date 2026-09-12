from full_multi_asset_backfill import CONFIGS
from graph import TECH_ANALYSIS_TIMEFRAME
from graph import TECH_ANALYSIS_TIMEFRAME
from trading_config import (
    ACTIVE_PRODUCTION_TIMEFRAMES,
    DEFAULT_TIMEFRAME,
    SUPPORTED_TIMEFRAMES,
    TIMEFRAME_THRESHOLDS,
)


def test_production_timeframes_are_hourly_and_daily_only():
    expected = ("1h", "4h", "1d")

    assert ACTIVE_PRODUCTION_TIMEFRAMES == expected
    assert SUPPORTED_TIMEFRAMES == list(expected)
    assert DEFAULT_TIMEFRAME == "4h"
    assert TECH_ANALYSIS_TIMEFRAME == DEFAULT_TIMEFRAME
    assert TECH_ANALYSIS_TIMEFRAME == DEFAULT_TIMEFRAME
    assert set(TIMEFRAME_THRESHOLDS).issubset(expected)


def test_full_backfill_cannot_schedule_minute_candles():
    expected = list(ACTIVE_PRODUCTION_TIMEFRAMES)

    assert CONFIGS
    assert all(config["timeframes"] == expected for config in CONFIGS)
