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


@pytest.fixture(autouse=True)
def _explicit_fixture_calendar(monkeypatch):
    """EQX v2 A1: pure runs without a calendar fail closed, so every fixture here
    declares one explicitly via the env override (a date outside all fixture ranges)."""
    monkeypatch.setenv(vn.NON_TRADING_DATES_ENV, "2000-01-03")


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
    # MBBREPLAY-1 (c): no open position at the end -> same figures as before.
    assert report["accounting"]["netReturnPct"] == pytest.approx(6.8588)
    assert report["accounting"]["grossReturnPct"] == pytest.approx(7.2)
    assert report["accounting"]["maxDrawdownPct"] == 0.0
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


# --- EQX: BID (decision equity, 1h + 1d since EQTECH-1 H) and capability gating ---


def test_bid_replay_runs_on_daily_fixture():
    bars = _fixture_bars()
    report = replay.run_replay(
        candles=bars,
        strategy=_buy_first_sell_fifth(),
        symbol="bid",
        initial_cash_vnd=5_000_000,
        cutoff_ms=bars[-1]["CloseTimeMs"] + 1,
    )
    assert report["symbol"] == "BID"
    assert report["timeframe"] == "1d"
    assert report["priceBasis"] == "adjusted"
    assert report["trades"]["closedRoundTrips"] == 1
    assert report["trades"]["roundTrips"][0]["entryPrice"] == 20_200.0


def test_bid_declares_hourly_strategy_like_mbb_and_rejects_undeclared_timeframes():
    # EQTECH-1 H: BID declares 1h, so "all" includes the 1h RSI strategy exactly as for MBB.
    assert replay.strategies_for_symbol("BID") == sorted(replay.STRATEGIES)
    assert replay.strategies_for_symbol("MBB") == sorted(replay.STRATEGIES)
    assert replay.strategies_for_symbol("BID", "rsi-mean-revert-14") == ["rsi-mean-revert-14"]
    four_hour = dict(_buy_first_sell_fifth(), timeframe="4h")
    with pytest.raises(ValueError, match="Inactive production timeframe"):
        replay.run_replay(candles=_fixture_bars(), strategy=four_hour, symbol="BID")


@pytest.mark.parametrize(
    ("symbol", "message"),
    [("VNINDEX", "equity-only"), ("VCB", "decision capability"), ("CTG", "decision capability")],
)
def test_index_and_context_only_symbols_fail_closed(symbol, message):
    with pytest.raises(ValueError, match=message):
        replay.run_replay(candles=_fixture_bars(), strategy=_buy_first_sell_fifth(), symbol=symbol)
    with pytest.raises(ValueError, match=message):
        replay.strategies_for_symbol(symbol)


def test_cli_bid_fixture_all_runs_both_declared_timeframes(tmp_path, capsys):
    bars = _fixture_bars()
    fixture = tmp_path / "bid.json"
    import json

    fixture.write_text(json.dumps(bars), encoding="utf-8")
    code = replay.main(
        ["--symbol", "BID", "--fixture-json", str(fixture), "--out-dir", str(tmp_path / "out"),
         "--cutoff-ms", str(bars[-1]["CloseTimeMs"] + 1)]
    )
    assert code == 0
    written = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert len(written) == 2
    assert written[0].startswith("rsi-mean-revert-14-BID-1h-")
    assert written[1].startswith("sma-cross-20-50-BID-1d-")
    reports = [json.loads((tmp_path / "out" / name).read_text(encoding="utf-8")) for name in written]
    assert [r["symbol"] for r in reports] == ["BID", "BID"]
    # Daily fixture: no expected 1h opens (all 7 rows excluded) and 7 bars < minBars 52.
    assert [r["verdict"]["value"] for r in reports] == ["unavailable", "unavailable"]


# --- MBBREPLAY-1: symmetric exit costs + uncertainty before "supported" ------


def _scripted(signals_by_index):
    return {
        **_buy_first_sell_fifth(),
        "name": "test-scripted",
        "signals": lambda closes: [signals_by_index.get(i) for i in range(len(closes))],
    }


def _replay(pairs, signals, **kwargs):
    bars = [_daily_bar(DAYS[i], float(o), float(c)) for i, (o, c) in enumerate(pairs)]
    return replay.run_replay(
        candles=bars,
        strategy=_scripted(signals),
        initial_cash_vnd=5_000_000,
        cutoff_ms=10**15,
        **kwargs,
    )


# Two winning round trips, 200 shares each, respecting T+2 (shares and sale cash
# settle 2 sessions later): buy 20,000 -> sell 21,000, buy 19,500 -> sell 20,500.
TWO_WINS = [
    (20_000, 20_000), (20_000, 20_500), (20_500, 21_000), (21_000, 20_000),
    (20_000, 19_500), (19_500, 20_000), (20_000, 20_500), (20_500, 19_500),
    (19_500, 19_500),
]
TWO_WINS_SIGNALS = {0: "buy", 2: "sell", 4: "buy", 6: "sell"}


def test_open_position_is_netted_like_buy_and_hold():
    # Buy once, never sell: the strategy IS buy-and-hold, so both report one net.
    report = replay.run_replay(
        candles=_fixture_bars(),
        strategy=_scripted({0: "buy"}),
        initial_cash_vnd=5_000_000,
        cutoff_ms=10**15,
    )
    acct = report["accounting"]
    # cash 953,940 + 200 x 23,000 = 5,553,940 marked; liquidation fee + tax
    # 4,600,000 x (0.15% + 0.1%) = 11,500 -> 5,542,440.
    assert acct["netReturnPct"] == pytest.approx(10.8488)
    assert acct["finalEquityVnd"] == pytest.approx(5_542_440.0)
    assert acct["netReturnPct"] == report["baselines"][0]["netReturnPct"]
    assert acct["markToMarketEquityVnd"] == pytest.approx(5_553_940.0)
    assert acct["assumedExitCostsVnd"] == pytest.approx(11_500.0)
    assert acct["grossReturnPct"] == pytest.approx(11.2)  # before any cost: unchanged
    assert report["trades"]["openPosition"]["shares"] == 200


def test_strategy_cannot_beat_baselines_by_skipping_exit_costs():
    # Two wins, then a re-entry at 19,500 drifting to 17,700: marked to market
    # the strategy is +0.031% (above flat), but selling 200 x 17,700 costs
    # 8,850 -> -0.146%, below flat.  The CI is above 0, so only the
    # accounting stands between this run and "supported".
    pairs = TWO_WINS + [(19_500, 18_300), (18_300, 17_900), (17_900, 17_700)]
    report = _replay(pairs, {**TWO_WINS_SIGNALS, 8: "buy"}, min_trades=2)
    assert report["verdict"] == {
        "value": "inconclusive",
        "reason": "net-return-not-above-baselines",
    }
    acct = report["accounting"]
    assert acct["netReturnPct"] == pytest.approx(-0.146)
    assert acct["markToMarketEquityVnd"] == pytest.approx(5_001_550.0)
    assert acct["assumedExitCostsVnd"] == pytest.approx(8_850.0)
    assert report["trades"]["closedRoundTrips"] == 2
    assert report["trades"]["openPosition"]["shares"] == 200
    assert report["uncertainty"]["ciLowerPct"] > 0


def test_supported_needs_baseline_beat_and_ci_lower_above_zero():
    report = _replay(TWO_WINS + [(19_500, 19_000)], TWO_WINS_SIGNALS, min_trades=2)
    assert report["verdict"] == {"value": "supported", "reason": "net-return-above-baselines"}
    unc = report["uncertainty"]
    keys = ("method", "seed", "resamples", "confidenceLevel", "sampleSize")
    assert {k: unc[k] for k in keys} == {
        "method": "percentile-bootstrap-iid",
        "seed": replay.BOOTSTRAP_SEED,
        "resamples": replay.BOOTSTRAP_RESAMPLES,
        "confidenceLevel": 0.95,
        "sampleSize": 2,
    }
    trips = [t["netReturnPct"] for t in report["trades"]["roundTrips"]]
    assert unc["ciLowerPct"] == pytest.approx(min(trips))
    assert unc["ciLowerPct"] > 0
    assert unc["ciUpperPct"] == pytest.approx(max(trips))
    assert report["accounting"]["assumedExitCostsVnd"] == 0.0


def test_beating_baselines_with_ci_reaching_zero_is_inconclusive():
    # Trip 2 loses (19,500 -> 19,200): net still beats both baselines, but the
    # bootstrap CI of the mean trip return reaches the losing trip.
    pairs = TWO_WINS[:6] + [(19_500, 19_300), (19_200, 19_000), (19_000, 18_800)]
    first = _replay(pairs, TWO_WINS_SIGNALS, min_trades=2)
    assert first["verdict"] == {
        "value": "inconclusive",
        "reason": "round-trip-ci-lower-not-above-zero",
    }
    best = max(b["netReturnPct"] for b in first["baselines"])
    assert first["accounting"]["netReturnPct"] > best
    assert first["uncertainty"]["ciLowerPct"] <= 0
    second = _replay(pairs, TWO_WINS_SIGNALS, min_trades=2)
    assert first["uncertainty"] == second["uncertainty"]


def test_single_round_trip_has_no_ci_and_cannot_be_supported():
    report = _replay(TWO_WINS[:5] + [(19_500, 19_000)], {0: "buy", 2: "sell"}, min_trades=1)
    assert report["verdict"] == {
        "value": "inconclusive",
        "reason": "uncertainty-unavailable:1<2",
    }
    assert report["uncertainty"]["ciLowerPct"] is None
    assert report["uncertainty"]["ciUpperPct"] is None


def test_bootstrap_block_is_deterministic():
    returns = [4.2, -1.3, 0.8, 2.5, -0.4, 3.1, 1.7]
    block = replay._bootstrap_uncertainty(returns)
    assert block == replay._bootstrap_uncertainty(list(returns))
    # Pinned: same values in a fresh process (fixed seed, no hash/dict ordering).
    assert block["ciLowerPct"] == pytest.approx(1.0 / 7.0)
    assert block["ciUpperPct"] == pytest.approx(19.8 / 7.0)
    assert block["ciLowerPct"] < block["meanNetReturnPct"] < block["ciUpperPct"]
