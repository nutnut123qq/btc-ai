"""MBB (HOSE equity) coverage for the descriptive evidence pipeline.

Fixtures emulate ``InstrumentCandles`` rows: session-aware opens/closes
(09/10/11/13/14 local, the 11:00 bar closes 11:30, the 14:00 bar closes 14:45,
1d opens 07:00 local).  Contiguity is the expected-open sequence — lunch,
overnight, weekend and configured non-trading boundaries are contiguous and a
missing expected open is a real gap.  No DB and no network: the loader tests
use a fake connection, the pipeline tests inject snapshot/cutoff loaders.
"""

import json
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import run_technical_evidence_pipeline as pipeline
import vietnam_market_calendar as vn
from technical_event_descriptive_evidence import (
    EvaluationConfig,
    build_postgresql_snapshot_from_rows,
    evaluate,
    load_postgresql_snapshot,
    parse_snapshot,
)


@pytest.fixture(autouse=True)
def _explicit_fixture_calendar(monkeypatch):
    """EQX v2 A1: pure runs without a calendar fail closed, so every fixture here
    declares one explicitly via the env override (a date outside all fixture ranges)."""
    monkeypatch.setenv(vn.NON_TRADING_DATES_ENV, "2000-01-03")


FIXED =datetime(2026, 10, 16, 1, 2, 3, tzinfo=timezone.utc)
HASH = "b" * 64
CANDLE_COLUMNS = ("OpenTimeMs", "CloseTimeMs", "Open", "High", "Low", "Close", "Volume")


def _trading_days(start: date, count: int, non_trading=frozenset()) -> list[date]:
    days = []
    cursor = start
    while len(days) < count:
        if vn.is_trading_day(cursor, non_trading):
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _bar(open_ms, close_ms, price):
    return {
        "OpenTimeMs": open_ms,
        "CloseTimeMs": close_ms,
        "Open": price - 0.5,
        "High": price + 1.0,
        "Low": price - 1.0,
        "Close": price,
        "Volume": 10_000.0,
    }


def _mbb_hourly_rows(days, first_close=20_000.0):
    rows = []
    price = first_close
    for day in days:
        for local_time in vn.HOURLY_OPENS_LOCAL:
            open_ms = vn.to_utc_ms(day, local_time)
            rows.append(_bar(open_ms, vn.session_close_ms("1h", open_ms), price))
            price += 1.0
    return rows


def _mbb_daily_rows(days, first_close=20_000.0):
    rows = []
    price = first_close
    for day in days:
        open_ms = vn.daily_open_ms(day)
        rows.append(_bar(open_ms, vn.session_close_ms("1d", open_ms), price))
        price += 1.0
    return rows


def _mbb_snapshot(timeframe: str, rows, cutoff_ms: int | None = None):
    cutoff = cutoff_ms if cutoff_ms is not None else int(rows[-1]["CloseTimeMs"])
    return build_postgresql_snapshot_from_rows(
        timeframe, cutoff, rows, [], smc_columns=[], symbol="MBB"
    )


def _mbb_event(event_id: str, candles, index: int):
    return {
        "eventId": event_id,
        "module": "causalSmc",
        "eventType": "BOS_BULL",
        "formedTimeMs": candles[index]["closeTimeMs"],
        "confirmedTimeMs": candles[index]["closeTimeMs"],
        "availableTimeMs": candles[index]["closeTimeMs"],
        "sourceCandleOpenTimesMs": [candles[index]["openTimeMs"]],
        "context": candles[index]["context"],
        "lineage": {"source": "fixture", "sourceVersion": "v1", "contentSha256": HASH},
    }


class _FakeCursor:
    def __init__(self, connection):
        self._connection = connection
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        self._connection.queries.append((sql, params))
        self._rows = self._connection.response_rows(sql)

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeConnection:
    """Minimal psycopg2 stand-in recording executed SQL."""

    def __init__(self, response_rows):
        self.queries = []
        self._response_rows = response_rows
        self.closed = False

    def set_session(self, **kwargs):
        pass

    def cursor(self):
        return _FakeCursor(self)

    def response_rows(self, sql):
        return self._response_rows(sql)

    def rollback(self):
        pass

    def close(self):
        self.closed = True


def _candle_tuples(rows):
    return [tuple(row[name] for name in CANDLE_COLUMNS) for row in rows]


def test_mbb_loader_reads_instrument_candles_and_skips_btc_tables():
    days = _trading_days(date(2026, 8, 3), 12)
    rows = _mbb_hourly_rows(days)
    cutoff = int(rows[-1]["CloseTimeMs"])

    def respond(sql):
        assert '"InstrumentCandles"' in sql
        return _candle_tuples(rows)

    connection = _FakeConnection(respond)
    with patch("db_config.get_db_connection", return_value=connection):
        snapshot = load_postgresql_snapshot("1h", cutoff, symbol="MBB")

    # Exactly one statement: the candle read.  The BTC-only lineage tables and
    # information_schema must never be touched for equity instruments.
    assert len(connection.queries) == 1
    sql, params = connection.queries[0]
    assert params == ("MBB", "1h", cutoff)
    for forbidden in ('"Klines"', "CausalSmartMoneyEvents", "SmartMoneyStructures", "information_schema"):
        assert forbidden not in sql
    assert connection.closed

    assert snapshot["symbol"] == "MBB"
    assert snapshot["lineage"]["source"] == "postgresql:InstrumentCandles"
    assert snapshot["lineage"]["sourceCutoffMs"] == cutoff
    assert len(snapshot["candles"]) == len(rows)
    assert snapshot["modules"]["causalSmc"]["status"] == "unavailable"
    assert "BTCUSDT-only" in snapshot["modules"]["causalSmc"]["reason"]


def test_mbb_loader_rejects_undeclared_timeframe_before_touching_db():
    with pytest.raises(ValueError):
        load_postgresql_snapshot("4h", 10**15, symbol="MBB")


def test_mbb_builder_excludes_off_session_anomaly_and_counts_it():
    days = _trading_days(date(2026, 8, 3), 10)
    rows = _mbb_hourly_rows(days)
    anomaly_open = vn.to_utc_ms(days[2], time(12, 0))  # lunch break (provider anomaly)
    anomaly = _bar(anomaly_open, vn.session_close_ms("1h", anomaly_open), 99_999.0)
    snapshot = build_postgresql_snapshot_from_rows(
        "1h", int(rows[-1]["CloseTimeMs"]), rows + [anomaly], [],
        smc_columns=[], symbol="MBB",
    )
    candles = snapshot["modules"]["finalizedCandles"]
    assert candles["includedRows"] == 50
    assert candles["excludedOffSessionRows"] == 1
    opens = {c["openTimeMs"] for c in snapshot["candles"]}
    assert anomaly_open not in opens


def test_mbb_builder_rejects_4h_and_off_session_bars_fail_closed():
    with pytest.raises(ValueError):
        build_postgresql_snapshot_from_rows("4h", 10**15, [], [], smc_columns=[], symbol="MBB")
    days = _trading_days(date(2026, 8, 3), 2)
    rows = _mbb_hourly_rows(days)
    snapshot = _mbb_snapshot("1h", rows)
    # Rewrite the day-0 11:00 bar as an off-session 12:00 anomaly while keeping
    # open ordering and availability monotonicity intact.
    anomaly_open = vn.to_utc_ms(days[0], time(12, 0))
    anomaly_close = vn.to_utc_ms(days[0], time(12, 30))
    snapshot["candles"][2] = {
        **snapshot["candles"][2],
        "openTimeMs": anomaly_open,
        "closeTimeMs": anomaly_close,
        "availableTimeMs": anomaly_close,
    }
    with pytest.raises(ValueError, match="declared session open"):
        parse_snapshot(snapshot, int(rows[-1]["CloseTimeMs"]))


def test_mbb_evaluation_horizons_realize_across_lunch_and_weekend():
    days = _trading_days(date(2026, 8, 3), 15)  # 75 hourly bars, 3 trading weeks
    rows = _mbb_hourly_rows(days)
    snapshot = _mbb_snapshot("1h", rows)
    candles = snapshot["candles"]
    cutoff = int(candles[-1]["closeTimeMs"])

    # Decision on the 11:00 bar: horizon 1 lands on the 13:00 bar (lunch break
    # is contiguous), horizon 6 lands next trading day.
    lunch_index = next(
        i for i, c in enumerate(candles)
        if c["openTimeMs"] == vn.to_utc_ms(days[0], time(11, 0))
    )
    # Decision on Friday's 14:00 bar: horizon 1 lands Monday 09:00.
    friday = next(day for day in days if day.weekday() == 4)
    weekend_index = next(
        i for i, c in enumerate(candles)
        if c["openTimeMs"] == vn.to_utc_ms(friday, time(14, 0))
    )
    snapshot["events"] = [
        _mbb_event("e-lunch", candles, lunch_index),
        _mbb_event("e-weekend", candles, weekend_index),
    ]
    _, ledger, report = evaluate(
        snapshot, EvaluationConfig(cutoff_ms=cutoff, bootstrap_samples=10)
    )
    by_id = {row["eventId"]: row for row in ledger}
    assert by_id["e-lunch"]["status"] == "eligible"
    assert by_id["e-weekend"]["status"] == "eligible"
    assert by_id["e-lunch"]["horizons"]["1"] is not None
    assert by_id["e-weekend"]["horizons"]["1"] is not None
    # Friday 14:00 + 6 session bars = Mon(5) + Tue 09:00.
    assert by_id["e-weekend"]["horizons"]["6"] is not None
    assert report["scope"]["symbol"] == "MBB"
    # Equity horizons carry no fixed wall-clock interval claim.
    for group in report["eventTypes"].values():
        assert all(h["elapsedTimeMs"] is None for h in group["horizons"].values())


@pytest.mark.parametrize("symbol", ["MBB", "BID"])
@pytest.mark.parametrize("timeframe", ["1h", "1d"])
def test_equity_statistical_evidence_scope_names_the_instrument(symbol, timeframe):
    # CurrentConditionsService rejects a bundle whose statisticalEvidence scope
    # names another instrument (evidence-scope-mismatch), so the equity report
    # must carry its own symbol there, not only in report.scope.
    rows = (
        _mbb_hourly_rows(_trading_days(date(2026, 8, 3), 15))
        if timeframe == "1h"
        else _mbb_daily_rows(_trading_days(date(2026, 6, 1), 60))
    )
    snapshot = build_postgresql_snapshot_from_rows(
        timeframe, int(rows[-1]["CloseTimeMs"]), rows, [], smc_columns=[], symbol=symbol
    )
    cutoff = int(snapshot["candles"][-1]["closeTimeMs"])
    _, _, report = evaluate(snapshot, EvaluationConfig(cutoff_ms=cutoff, bootstrap_samples=10))
    assert report["scope"]["symbol"] == symbol
    assert report["statisticalEvidence"]["scope"] == {"symbol": symbol, "timeframe": timeframe}


def test_mbb_evaluation_missing_expected_open_is_a_real_gap():
    days = _trading_days(date(2026, 8, 3), 15)
    rows = _mbb_hourly_rows(days)
    dropped_open = vn.to_utc_ms(days[5], time(13, 0))
    kept = [row for row in rows if row["OpenTimeMs"] != dropped_open]
    snapshot = _mbb_snapshot("1h", kept)
    candles = snapshot["candles"]
    cutoff = int(candles[-1]["closeTimeMs"])
    gap_index = next(
        i for i, c in enumerate(candles)
        if c["openTimeMs"] == vn.to_utc_ms(days[5], time(11, 0))
    )
    safe_index = next(
        i for i, c in enumerate(candles)
        if c["openTimeMs"] == vn.to_utc_ms(days[3], time(11, 0))
    )
    snapshot["events"] = [
        _mbb_event("e-gap", candles, gap_index),
        _mbb_event("e-safe", candles, safe_index),
    ]
    _, ledger, _ = evaluate(
        snapshot, EvaluationConfig(cutoff_ms=cutoff, bootstrap_samples=10)
    )
    by_id = {row["eventId"]: row for row in ledger}
    # The 14:00 bar after the dropped 13:00 open is NOT contiguous.
    assert by_id["e-gap"]["status"] == "excluded"
    assert by_id["e-gap"]["exclusionReason"] == "non_contiguous_followup"
    assert by_id["e-safe"]["status"] == "eligible"
    assert by_id["e-safe"]["horizons"]["6"] is not None


def test_mbb_lifecycle_lookup_uses_session_segments_across_weekend():
    days = _trading_days(date(2026, 8, 3), 15)
    rows = _mbb_hourly_rows(days)
    snapshot = _mbb_snapshot("1h", rows)
    candles = snapshot["candles"]
    cutoff = int(candles[-1]["closeTimeMs"])
    friday = next(day for day in days if day.weekday() == 4)
    decision_index = next(
        i for i, c in enumerate(candles)
        if c["openTimeMs"] == vn.to_utc_ms(friday, time(14, 0))
    )
    monday_low = candles[decision_index + 1]["low"]
    event = _mbb_event("e-touch", candles, decision_index)
    event["lifecycle"] = {
        "semanticsVersion": "fixture-v1",
        "touchLevel": {"operator": "at_or_below", "price": monday_low},
    }
    snapshot["events"] = [event]
    _, ledger, _ = evaluate(
        snapshot, EvaluationConfig(cutoff_ms=cutoff, bootstrap_samples=10)
    )
    row = next(item for item in ledger if item["eventId"] == "e-touch")
    assert row["lifecycle"]["status"] == "supported"
    # The Monday 09:00 bar is bar +1 despite the weekend between.
    assert row["lifecycle"]["firstTouch"]["bars"] == 1


def test_mbb_latest_cutoffs_reads_instrument_candles():
    def respond(sql):
        assert '"InstrumentCandles"' in sql
        assert '"Klines"' not in sql
        return [(1_800_000_000_000,)]

    connection = _FakeConnection(respond)
    with patch("db_config.get_db_connection", return_value=connection):
        result = pipeline.latest_cutoffs("MBB", ("1h", "1d"))
    assert result == {"1h": 1_800_000_000_000, "1d": 1_800_000_000_000}
    assert [params[0] for _, params in connection.queries] == ["MBB", "MBB"]
    assert [params[1] for _, params in connection.queries] == ["1h", "1d"]


def test_mbb_run_publishes_two_bundles_in_per_symbol_dir(tmp_path):
    hourly = _mbb_hourly_rows(_trading_days(date(2026, 8, 3), 20))
    daily = _mbb_daily_rows(_trading_days(date(2026, 6, 1), 60))
    snapshots = {
        "1h": _mbb_snapshot("1h", hourly, cutoff_ms=10**15),
        "1d": _mbb_snapshot("1d", daily, cutoff_ms=10**15),
    }

    status = pipeline.run_pipeline(
        tmp_path,
        symbol="MBB",
        forced_cutoff_ms=10**15,
        bootstrap_samples=10,
        now=lambda: FIXED,
        snapshot_loader=lambda timeframe, cutoff: snapshots[timeframe],
    )

    run_dir = tmp_path / "MBB"
    manifests = list(run_dir.glob("*.manifest.json"))
    assert len(manifests) == 2
    assert all(pipeline.verify_bundle(path)["valid"] for path in manifests)
    assert {item["timeframe"] for item in status["timeframes"]} == {"1h", "1d"}

    # The BTC legacy top-level layout is untouched.
    assert not (tmp_path / pipeline.RUN_POINTER_FILE).exists()
    assert not (tmp_path / pipeline.STATUS_FILE).exists()
    assert not (tmp_path / pipeline.LOCK_FILE).exists()

    pointer = json.loads((run_dir / pipeline.RUN_POINTER_FILE).read_text(encoding="utf-8"))
    index = json.loads((run_dir / pointer["runIndexFileName"]).read_text(encoding="utf-8"))
    assert index["schema"] == pipeline.RUN_INDEX_SCHEMA
    assert index["symbol"] == "MBB"
    assert [item["timeframe"] for item in index["bundles"]] == ["1h", "1d"]

    verified = pipeline.read_verified_success(
        run_dir, symbol="MBB", timeframes=("1h", "1d")
    )
    assert verified["valid"] is True
    assert [item["timeframe"] for item in verified["bundles"]] == ["1h", "1d"]

    # Default reader (BTCUSDT trio) fails closed on the MBB run index.
    with pytest.raises(ValueError):
        pipeline.read_verified_success(run_dir)
    # No pointer exists at the BTC top level.
    with pytest.raises(FileNotFoundError):
        pipeline.read_verified_success(tmp_path)


def test_mbb_pipeline_rejects_undeclared_timeframes_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="declared timeframes"):
        pipeline.run_pipeline(
            tmp_path,
            symbol="MBB",
            timeframes=("1h", "4h", "1d"),
            forced_cutoff_ms=10**15,
            bootstrap_samples=10,
            now=lambda: FIXED,
            snapshot_loader=lambda timeframe, cutoff: _mbb_snapshot(
                timeframe,
                _mbb_hourly_rows(_trading_days(date(2026, 8, 3), 20)),
                cutoff_ms=10**15,
            ),
        )
    with pytest.raises(ValueError):
        pipeline.run_pipeline(
            tmp_path,
            symbol="VNM",
            forced_cutoff_ms=10**15,
            bootstrap_samples=10,
            now=lambda: FIXED,
            snapshot_loader=lambda timeframe, cutoff: {},
        )
