"""Replay-level tests for run_mbb_replay (MBB-04C).

Fixture candles only — never touches the DB.  Custom strategy specs are
injected through the same StrategySpec shape the declared strategies use,
so expected PnL can be computed by hand.
"""

import sys
from datetime import date, time, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import run_mbb_replay as replay
import vietnam_market_calendar as vn
from vn_execution_model import PRICE_BASIS


def _trading_days(start: date, count: int) -> list[date]:
    days, cursor = [], start
    while len(days) < count:
        if vn.is_trading_day(cursor):
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _daily_bar(day: date, open_: float, close: float, price_basis="adjusted"):
    open_ms = vn.daily_open_ms(day)
    return {
        "OpenTimeMs": open_ms,
        "CloseTimeMs": vn.session_close_ms("1d", open_ms),
        "Open": open_,
        "High": max(open_, close) + 100.0,
        "Low": min(open_, close) - 100.0,
        "Close": close,
        "Volume": 1_000_000.0,
        "PriceBasis": price_basis,
    }


DAYS = _trading_days(date(2026, 8, 3), 12)


def _buy_first_sell_fifth():
    def signal(closes, i):  # noqa: ANN001 - matches StrategySpec signature
        return {0: "buy", 4: "sell"}.get(i)

    return {
        "name": "test-buy-first-sell-fifth",
        "timeframe": "1d",
        "minBars": 2,
        "params": {},
        "signals": lambda closes: [signal(closes, i) for i in range(len(closes))],
    }


def _fixture_bars():
    opens = [20_000, 20_200, 20_500, 20_800, 21_000, 22_000, 21_500]
    closes = [20_100, 20_400, 20_700, 20_900, 21_800, 21_700, 23_000]
    return [
        _daily_bar(DAYS[i], float(opens[i]), float(closes[i])) for i in range(7)
    ]


def test_schema_provenance_and_deterministic_pnl():
    bars = _fixture_bars()
    cutoff = bars[-1]["CloseTimeMs"] + 1
    report = replay.run_replay(
        candles=bars,
        strategy=_buy_first_sell_fifth(),
        initial_cash_vnd=5_000_000,
        cutoff_ms=cutoff,
    )
    assert report["schema"] == "vn-equity-rule-replay/v1"
    assert report["symbol"] == "MBB"
    assert report["timeframe"] == "1d"
    assert report["priceBasis"] == PRICE_BASIS == "adjusted"
    assert report["dataset"]["cutoffMs"] == cutoff
    assert report["dataset"]["sha256"] == replay.dataset_sha256(
        replay.normalize_candles(bars, cutoff)
    )
    assert report["dataset"]["usableSessionBars"] == 7
    assert report["calendar"]["nonTradingDatesEnv"] == "MBB_NON_TRADING_DATES"
    assert report["execution"]["lotSize"] == 100
    assert report["execution"]["settlementTradingDays"] == 2

    # Signal buy at bar0 close -> fill at bar1 open (20,200).
    # 5,000,000 / (20,200*1.0015) = 247.0 -> 200 shares (2 lots).
    # notional 4,040,000 + fee 6,060 -> cash left 953,940.
    # Signal sell at bar4 close -> fill at bar5 open (22,000); shares settled
    # (fill day1 + 2 trading days = day3).
    trips = report["trades"]["roundTrips"]
    assert len(trips) == 1
    trip = trips[0]
    assert trip["shares"] == 200
    assert trip["entryOpenMs"] == bars[1]["OpenTimeMs"]
    assert trip["exitOpenMs"] == bars[5]["OpenTimeMs"]
    assert trip["entryPrice"] == 20_200.0 and trip["exitPrice"] == 22_000.0
    assert trip["grossPnlVnd"] == pytest.approx(360_000.0)
    assert trip["feeVnd"] == pytest.approx(6_060.0 + 6_600.0)
    assert trip["taxVnd"] == pytest.approx(4_400.0)
    assert trip["netPnlVnd"] == pytest.approx(360_000 - 12_660 - 4_400)
    assert trip["priceBasis"] == "adjusted"

    assert report["accounting"]["finalEquityVnd"] == pytest.approx(
        953_940.0 + (4_400_000 - 6_600 - 4_400)
    )
    assert {b["id"] for b in report["baselines"]} == {"buy-and-hold", "flat"}
    assert report["verdict"]["value"] == "inconclusive"
    assert report["verdict"]["reason"].startswith("insufficient-trades:1<")


def test_verdict_unavailable_when_too_few_bars():
    report = replay.run_replay(
        candles=_fixture_bars()[:3],
        strategy={**_buy_first_sell_fifth(), "minBars": 10},
        cutoff_ms=10**15,
    )
    assert report["verdict"] == {
        "value": "unavailable",
        "reason": "insufficient-bars:3<10",
    }
    assert report["trades"]["closedRoundTrips"] == 0


def test_t2_blocks_fast_flip_inside_replay():
    # Buy signal day0 -> fill day1; sell signal day1 -> fill day2, but the
    # shares only settle on day3 -> the sell is rejected, position stays open.
    spec = _buy_first_sell_fifth()
    spec["signals"] = lambda closes: [
        {0: "buy", 1: "sell"}.get(i) for i in range(len(closes))
    ]
    report = replay.run_replay(
        candles=_fixture_bars(),
        strategy=spec,
        initial_cash_vnd=5_000_000,
        cutoff_ms=10**15,
    )
    rejections = report["trades"]["rejections"]
    assert [r["reason"] for r in rejections] == ["shares-not-settled"]
    assert report["trades"]["closedRoundTrips"] == 0
    assert report["trades"]["openPosition"]["shares"] == 200


def test_off_session_anomaly_bar_is_excluded():
    bars = _fixture_bars()
    anomaly_day = DAYS[2]
    anomaly_open = vn.to_utc_ms(anomaly_day, time(12, 0))
    bars.append(
        {
            "OpenTimeMs": anomaly_open,
            "CloseTimeMs": vn.session_close_ms("1h", anomaly_open),
            "Open": 99.0,
            "High": 99.0,
            "Low": 99.0,
            "Close": 99.0,
            "Volume": 1.0,
            "PriceBasis": "adjusted",
        }
    )
    report = replay.run_replay(
        candles=bars,
        strategy=_buy_first_sell_fifth(),
        cutoff_ms=10**15,
    )
    assert report["dataset"]["candleRows"] == 8
    assert report["dataset"]["excludedAnomalyBars"] == 1
    assert report["dataset"]["usableSessionBars"] == 7
    # Anomaly prices must not leak into fills.
    for fill in report["trades"]["fills"]:
        assert fill["price"] != 99.0


def test_fail_closed_symbol_timeframe_and_basis():
    bars = _fixture_bars()
    with pytest.raises(ValueError, match="equity-only"):
        replay.run_replay(candles=bars, strategy=_buy_first_sell_fifth(), symbol="BTCUSDT")
    with pytest.raises(ValueError, match="Unknown symbol"):
        replay.run_replay(candles=bars, strategy=_buy_first_sell_fifth(), symbol="ETHUSDT")
    with pytest.raises(ValueError, match="bound to 1d"):
        replay.run_replay(
            candles=bars, strategy=_buy_first_sell_fifth(), timeframe="4h"
        )
    with pytest.raises(ValueError, match="Unknown strategy"):
        replay.run_replay(candles=bars, strategy="macd-voodoo")
    bad = [dict(row, PriceBasis="raw") for row in bars]
    with pytest.raises(ValueError, match="price basis"):
        replay.run_replay(candles=bad, strategy=_buy_first_sell_fifth())


def test_deterministic_dataset_hash_and_declared_strategies():
    bars = _fixture_bars()
    h1 = replay.dataset_sha256(replay.normalize_candles(bars, 10**15))
    h2 = replay.dataset_sha256(replay.normalize_candles(bars, 10**15))
    assert h1 == h2 and len(h1) == 64
    # Declared strategies stay bound to their timeframes and keep fixed params.
    assert replay.STRATEGIES["sma-cross-20-50"]["timeframe"] == "1d"
    assert replay.STRATEGIES["sma-cross-20-50"]["params"] == {"fast": 20, "slow": 50}
    assert replay.STRATEGIES["rsi-mean-revert-14"]["timeframe"] == "1h"
    assert replay.STRATEGIES["rsi-mean-revert-14"]["params"] == {
        "period": 14,
        "oversold": 30.0,
        "exitLevel": 50.0,
    }
