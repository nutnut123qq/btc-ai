"""MBB (HOSE equity) coverage for ``current_technical_conditions``.

Fixtures emulate InstrumentCandles rows: session-aware opens/closes (09/10/11/13/14
local, 11:00 closes 11:30, 14:00 closes 14:45, 1d opens 07:00 local). No DB and no
network — ``_load_inputs`` is patched or fed a fake connection.
"""

import sys
from datetime import date, time, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import current_technical_conditions as ctc
import vietnam_market_calendar as vn
from current_technical_conditions import (
    CANONICAL_SMC_COLUMNS,
    build_conditions_payload,
)


@pytest.fixture(autouse=True)
def _explicit_fixture_calendar(monkeypatch):
    """EQX v2 A1: pure runs without a calendar fail closed, so every fixture here
    declares one explicitly via the env override (a date outside all fixture ranges)."""
    monkeypatch.setenv(vn.NON_TRADING_DATES_ENV, "2000-01-03")


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


def test_load_inputs_reads_instrument_candles_for_mbb():
    connection = _FakeConnection(lambda sql: [])
    with patch("db_config.get_db_connection", return_value=connection):
        rows, causal_rows, causal_columns, causal_error = ctc._load_inputs(
            "MBB", "1h", now_ms=1_800_000_000_000
        )
    assert '"InstrumentCandles"' in connection.queries[0][0]
    assert '"Klines"' not in connection.queries[0][0]
    assert connection.queries[0][1][0] == "MBB"
    # No exact-duration predicate on the equity store.
    assert '"CloseTimeMs"-"OpenTimeMs"' not in connection.queries[0][0]
    assert rows == [] and connection.closed


def test_load_inputs_reads_klines_for_btc():
    connection = _FakeConnection(lambda sql: [])
    with patch("db_config.get_db_connection", return_value=connection):
        ctc._load_inputs("BTCUSDT", "4h", now_ms=1_800_000_000_000)
    assert '"Klines"' in connection.queries[0][0]
    assert '"InstrumentCandles"' not in connection.queries[0][0]


def test_mbb_hourly_contiguous_across_lunch_and_weekend():
    # Two full trading weeks: every day has 5 session opens; lunch (11->13),
    # overnight and the Fri->Mon weekend must not be reported as gaps.
    days = _trading_days(date(2026, 8, 3), 10)
    rows = _mbb_hourly_rows(days)
    assert len(rows) == 50
    payload = build_conditions_payload(
        "1h", rows, [], sorted(CANONICAL_SMC_COLUMNS),
        now_ms=int(rows[-1]["CloseTimeMs"]), symbol="MBB",
    )
    assert payload["symbol"] == "MBB"
    assert payload["timeframe"] == "1h"
    assert payload["analysisWindow"]["bars"] == 50
    assert payload["analysisWindow"]["contiguous"] is True
    assert "analysis-window-truncated-at-non-contiguous-gap" not in payload["warnings"]


def test_mbb_hourly_missing_expected_open_is_a_real_gap():
    days = _trading_days(date(2026, 8, 3), 10)
    rows = _mbb_hourly_rows(days)
    # Drop the 13:00 bar of day 8 -> the contiguous run must restart at 14:00.
    dropped_open = vn.to_utc_ms(days[8], vn.HOURLY_OPENS_LOCAL[3])
    kept = [r for r in rows if r["OpenTimeMs"] != dropped_open]
    payload = build_conditions_payload(
        "1h", kept, [], sorted(CANONICAL_SMC_COLUMNS),
        now_ms=int(kept[-1]["CloseTimeMs"]), symbol="MBB",
    )
    # Day 8 keeps only its 14:00 bar, then 5 bars of day 9.
    assert payload["analysisWindow"]["bars"] == 6
    assert "analysis-window-truncated-at-non-contiguous-gap" in payload["warnings"]
    assert "fibonacci-anchor-may-predate-window" in payload["warnings"]


def test_mbb_off_session_anomaly_bar_is_excluded_not_a_gap():
    days = _trading_days(date(2026, 8, 3), 6)
    rows = _mbb_hourly_rows(days)
    # Provider anomaly like the real 2024-06-20 12:00 bar: inside lunch break.
    anomaly_open = vn.to_utc_ms(days[2], time(12, 0))
    anomaly = _bar(anomaly_open, vn.session_close_ms("1h", anomaly_open), 99_999.0)
    payload = build_conditions_payload(
        "1h", rows + [anomaly], [], sorted(CANONICAL_SMC_COLUMNS),
        now_ms=int(rows[-1]["CloseTimeMs"]), symbol="MBB",
    )
    assert payload["analysisWindow"]["bars"] == 30
    assert "analysis-window-truncated-at-non-contiguous-gap" not in payload["warnings"]
    assert any("off-session" in w for w in payload["warnings"])
    # The anomalous bar must not leak into the emitted window.
    assert payload["latestClosedBar"]["openTimeMs"] == int(rows[-1]["OpenTimeMs"])


def test_mbb_daily_contiguous_across_weekend_and_holiday():
    holiday = date(2026, 9, 2)  # configured non-trading date (VN National Day)
    days = _trading_days(date(2026, 8, 24), 12, non_trading=frozenset({holiday}))
    assert holiday not in days
    rows = _mbb_daily_rows(days)
    with patch.dict("os.environ", {"MBB_NON_TRADING_DATES": "2026-09-02"}):
        payload = build_conditions_payload(
            "1d", rows, [], sorted(CANONICAL_SMC_COLUMNS),
            now_ms=int(rows[-1]["CloseTimeMs"]), symbol="MBB",
        )
    assert payload["analysisWindow"]["bars"] == 12
    assert "analysis-window-truncated-at-non-contiguous-gap" not in payload["warnings"]


def test_mbb_daily_missing_trading_day_is_a_real_gap():
    days = _trading_days(date(2026, 8, 24), 12)
    rows = _mbb_daily_rows(days)
    kept = rows[:-3] + rows[-2:]  # drop the bar of day index 9
    payload = build_conditions_payload(
        "1d", kept, [], sorted(CANONICAL_SMC_COLUMNS),
        now_ms=int(kept[-1]["CloseTimeMs"]), symbol="MBB",
    )
    assert payload["analysisWindow"]["bars"] == 2
    assert "analysis-window-truncated-at-non-contiguous-gap" in payload["warnings"]


def test_mbb_rejects_4h_and_unknown_symbol():
    with pytest.raises(ctc.UnknownTimeframeError):
        build_conditions_payload("4h", [], [], [], now_ms=0, symbol="MBB")
    with pytest.raises(ctc.UnknownSymbolError):
        build_conditions_payload("1h", [], [], [], now_ms=0, symbol="VNM")
    with pytest.raises(ctc.UnknownSymbolError):
        ctc.get_current_conditions("1h", symbol="VNM")
    with pytest.raises(ctc.UnknownTimeframeError):
        ctc.get_current_conditions("4h", symbol="MBB")


class TestEndpoint:
    def setup_method(self):
        ctc.clear_cache()
        self.client = TestClient(__import__("main").app)

    def test_mbb_4h_returns_400(self):
        response = self.client.get("/api/current-conditions?timeframe=4h&symbol=MBB")
        assert response.status_code == 400

    def test_unknown_symbol_returns_400(self):
        response = self.client.get("/api/current-conditions?timeframe=1h&symbol=XYZ")
        assert response.status_code == 400

    def test_mbb_payload_echoes_symbol(self):
        days = _trading_days(date(2026, 8, 3), 10)
        rows = _mbb_hourly_rows(days)
        with patch.object(
            ctc, "_load_inputs", return_value=(rows, [], ["EventId"], None)
        ):
            response = self.client.get("/api/current-conditions?timeframe=1h&symbol=MBB")
        assert response.status_code == 200
        payload = response.json()
        assert payload["schemaVersion"] == "current-conditions-v1"
        assert payload["symbol"] == "MBB"
        assert payload["timeframe"] == "1h"
        # causalSmc degrades honestly: canonical columns missing -> unavailable.
        assert payload["unavailableModules"][0]["module"] == "causalSmc"

    def test_btc_request_path_unchanged(self):
        # Default symbol stays BTCUSDT and reaches the loader unchanged.
        with patch.object(ctc, "_load_inputs", return_value=([], [], [], None)) as loader:
            response = self.client.get("/api/current-conditions?timeframe=4h")
        assert response.status_code == 503
        assert loader.call_args[0][0] == "BTCUSDT"


# --- EQX: BID (1d-only decision equity) + capability gating -----------------


def test_bid_daily_conditions_on_instrument_candles():
    days = _trading_days(date(2026, 8, 3), 40)
    rows = _mbb_daily_rows(days, first_close=40_000.0)
    payload = build_conditions_payload(
        "1d", rows, [], sorted(CANONICAL_SMC_COLUMNS),
        now_ms=int(rows[-1]["CloseTimeMs"]), symbol="bid",
    )
    assert payload["schemaVersion"] == "current-conditions-v1"
    assert payload["symbol"] == "BID"
    assert payload["timeframe"] == "1d"
    assert payload["analysisWindow"]["bars"] == 40
    assert payload["analysisWindow"]["contiguous"] is True
    assert payload["latestClosedBar"]["openTimeMs"] == int(rows[-1]["OpenTimeMs"])


def test_bid_load_inputs_reads_instrument_candles():
    connection = _FakeConnection(lambda sql: [])
    with patch("db_config.get_db_connection", return_value=connection):
        ctc._load_inputs("BID", "1d", now_ms=1_800_000_000_000)
    assert '"InstrumentCandles"' in connection.queries[0][0]
    assert connection.queries[0][1][:2] == ("BID", "1d")


def test_bid_rejects_hourly_and_context_only_symbols_rejected():
    with pytest.raises(ctc.UnknownTimeframeError):
        build_conditions_payload("1h", [], [], [], now_ms=0, symbol="BID")
    for symbol in ("VCB", "CTG", "VNINDEX"):
        with pytest.raises(ctc.UnsupportedCapabilityError):
            build_conditions_payload("1d", [], [], [], now_ms=0, symbol=symbol)
        with pytest.raises(ctc.UnsupportedCapabilityError):
            ctc.get_current_conditions("1d", symbol=symbol)


class TestEndpointBid:
    def setup_method(self):
        ctc.clear_cache()
        self.client = TestClient(__import__("main").app)

    def test_bid_daily_payload(self):
        rows = _mbb_daily_rows(_trading_days(date(2026, 8, 3), 30), first_close=40_000.0)
        with patch.object(ctc, "_load_inputs", return_value=(rows, [], ["EventId"], None)) as loader:
            response = self.client.get("/api/current-conditions?timeframe=1d&symbol=BID")
        assert response.status_code == 200
        assert response.json()["symbol"] == "BID"
        assert loader.call_args[0][:2] == ("BID", "1d")

    def test_bid_hourly_and_vnindex_return_400(self):
        assert self.client.get("/api/current-conditions?timeframe=1h&symbol=BID").status_code == 400
        assert self.client.get("/api/current-conditions?timeframe=1d&symbol=VNINDEX").status_code == 400
        assert self.client.get("/api/current-conditions?timeframe=1d&symbol=VCB").status_code == 400
