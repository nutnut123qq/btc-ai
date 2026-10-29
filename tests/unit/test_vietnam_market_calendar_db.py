"""EQX v2 A1 — HOSE calendar from ``MarketCalendarDays`` (fail closed).

No real database: a fake DB-API connection/cursor stands in for psycopg2 and is
installed as the ``db_config`` module, so nothing here can reach PostgreSQL.
"""

from __future__ import annotations

import sys
import types
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import current_technical_conditions as ctc
import equity_base_rates as ebr
import run_mbb_replay as replay
import vietnam_market_calendar as vn
from current_technical_conditions import CANONICAL_SMC_COLUMNS, build_conditions_payload

HOLIDAY = date(2026, 9, 2)  # VN National Day (Wednesday)


class FakeCursor:
    def __init__(self, rows=None, error: Exception | None = None):
        self.rows = rows or []
        self.error = error
        self.executed: list[tuple[str, tuple]] = []

    def execute(self, sql, params=()):
        self.executed.append((sql, params))
        if self.error is not None:
            raise self.error

    def fetchall(self):
        return list(self.rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConnection:
    def __init__(self, cursor: FakeCursor):
        self._cursor = cursor
        self.closed = False
        self.rolled_back = False

    def cursor(self):
        return self._cursor

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def _install_db(monkeypatch, cursor: FakeCursor) -> FakeConnection:
    connection = FakeConnection(cursor)
    module = types.ModuleType("db_config")
    module.get_db_connection = lambda: connection
    monkeypatch.setitem(sys.modules, "db_config", module)
    return connection


COVERAGE = (date(2020, 1, 1), date(2026, 12, 31))


def _calendar_rows(*days: date, verification="verified", version="vn-hose-calendar-v1", coverage=COVERAGE):
    return [(d, verification, version, coverage[0], coverage[1]) for d in days]


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    monkeypatch.delenv(vn.NON_TRADING_DATES_ENV, raising=False)


def _trading_days(start: date, count: int, non_trading=frozenset()) -> list[date]:
    days, cursor = [], start
    while len(days) < count:
        if vn.is_trading_day(cursor, non_trading):
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _daily_rows(days, first_close=20_000.0):
    rows, price = [], first_close
    for day in days:
        open_ms = vn.daily_open_ms(day)
        rows.append({
            "OpenTimeMs": open_ms, "CloseTimeMs": vn.session_close_ms("1d", open_ms),
            "Open": price - 0.5, "High": price + 1.0, "Low": price - 1.0, "Close": price,
            "Volume": 10_000.0,
        })
        price += 1.0
    return rows


# --- loader ---------------------------------------------------------------------


def test_load_from_cursor_reads_hose_rows_with_provenance():
    cursor = FakeCursor(
        _calendar_rows(date(2026, 4, 30), date(2026, 5, 1))
        + _calendar_rows(date(2019, 2, 4), verification="unverified-legacy")
    )
    cal = vn.load_calendar_from_cursor(cursor)
    assert cal.source == "db:MarketCalendarDays"
    assert cal.non_trading_dates == frozenset({date(2026, 4, 30), date(2026, 5, 1), date(2019, 2, 4)})
    prov = cal.provenance()
    assert prov["count"] == 3 and prov["unverifiedCount"] == 1
    assert prov["version"] == "vn-hose-calendar-v1"
    assert prov["firstDate"] == "2019-02-04" and len(prov["sha256"]) == 64
    sql, params = cursor.executed[0]
    assert '"MarketCalendarDays"' in sql and params == ("HOSE",)


def test_empty_table_fails_closed():
    with pytest.raises(vn.CalendarUnavailableError, match="calendar-unavailable"):
        vn.load_calendar_from_cursor(FakeCursor([]))


def test_unreadable_table_fails_closed():
    with pytest.raises(vn.CalendarUnavailableError, match="could not be read"):
        vn.load_calendar_from_cursor(FakeCursor(error=RuntimeError("relation does not exist")))


def test_db_loader_uses_table_and_closes_connection(monkeypatch):
    connection = _install_db(monkeypatch, FakeCursor(_calendar_rows(HOLIDAY)))
    cal = vn.load_calendar_db()
    assert cal.non_trading_dates == frozenset({HOLIDAY})
    assert connection.closed and connection.rolled_back


def test_env_override_wins_without_touching_db(monkeypatch):
    def boom():
        raise AssertionError("DB must not be opened when the env override is set")

    module = types.ModuleType("db_config")
    module.get_db_connection = boom
    monkeypatch.setitem(sys.modules, "db_config", module)
    monkeypatch.setenv(vn.NON_TRADING_DATES_ENV, "2026-09-02,2026-09-01")
    cal = vn.load_calendar_db()
    assert cal.source == "env:MBB_NON_TRADING_DATES"
    assert cal.non_trading_dates == frozenset({date(2026, 9, 1), HOLIDAY})


def test_blank_env_is_not_an_override(monkeypatch):
    monkeypatch.setenv(vn.NON_TRADING_DATES_ENV, "  ")
    assert vn.calendar_from_env() is None
    with pytest.raises(vn.CalendarUnavailableError):
        vn.require_calendar()


def test_db_unreachable_fails_closed(monkeypatch):
    module = types.ModuleType("db_config")

    def unreachable():
        raise ConnectionError("refused")

    module.get_db_connection = unreachable
    monkeypatch.setitem(sys.modules, "db_config", module)
    with pytest.raises(vn.CalendarUnavailableError, match="unreachable"):
        vn.load_calendar_db()


# --- conditions: continuity across a valid holiday vs a real gap ------------------


def test_conditions_holiday_continuity_with_db_calendar():
    days = _trading_days(date(2026, 8, 24), 12, non_trading=frozenset({HOLIDAY}))
    assert HOLIDAY not in days
    rows = _daily_rows(days)
    cal = vn.load_calendar_from_cursor(FakeCursor(_calendar_rows(HOLIDAY)))
    payload = build_conditions_payload(
        "1d", rows, [], sorted(CANONICAL_SMC_COLUMNS),
        now_ms=int(rows[-1]["CloseTimeMs"]), symbol="MBB", calendar=cal,
    )
    assert payload["analysisWindow"]["bars"] == 12
    assert payload["analysisWindow"]["firstOpenTimeMs"] == rows[0]["OpenTimeMs"]
    assert "analysis-window-truncated-at-non-contiguous-gap" not in payload["warnings"]
    assert payload["calendar"]["source"] == "db:MarketCalendarDays"
    assert payload["calendar"]["count"] == 1


def test_conditions_without_the_holiday_resets_window():
    """Before the fix (no calendar rows) the same bars truncate at the holiday."""
    days = _trading_days(date(2026, 8, 24), 12, non_trading=frozenset({HOLIDAY}))
    rows = _daily_rows(days)
    payload = build_conditions_payload(
        "1d", rows, [], sorted(CANONICAL_SMC_COLUMNS),
        now_ms=int(rows[-1]["CloseTimeMs"]), symbol="MBB", non_trading_dates=frozenset(),
    )
    after_holiday = [r for r in rows if vn.to_local(r["OpenTimeMs"]).date() > HOLIDAY]
    assert payload["analysisWindow"]["bars"] == len(after_holiday)
    assert "analysis-window-truncated-at-non-contiguous-gap" in payload["warnings"]


def test_conditions_real_gap_still_breaks_continuity_with_calendar():
    days = _trading_days(date(2026, 8, 24), 12, non_trading=frozenset({HOLIDAY}))
    rows = _daily_rows(days)
    kept = rows[:-3] + rows[-2:]  # a trading day (not in the calendar) has no bar
    cal = vn.calendar_from_dates({HOLIDAY})
    payload = build_conditions_payload(
        "1d", kept, [], sorted(CANONICAL_SMC_COLUMNS),
        now_ms=int(kept[-1]["CloseTimeMs"]), symbol="MBB", calendar=cal,
    )
    assert payload["analysisWindow"]["bars"] == 2
    assert "analysis-window-truncated-at-non-contiguous-gap" in payload["warnings"]


def test_conditions_pure_call_without_calendar_fails_closed():
    rows = _daily_rows(_trading_days(date(2026, 8, 24), 5))
    with pytest.raises(vn.CalendarUnavailableError):
        build_conditions_payload("1d", rows, [], [], now_ms=int(rows[-1]["CloseTimeMs"]), symbol="BID")


def test_get_current_conditions_db_backed_fail_closed_on_empty_table(monkeypatch):
    _install_db(monkeypatch, FakeCursor([]))
    ctc.clear_cache()
    with patch.object(ctc, "_load_inputs") as loader:
        with pytest.raises(ctc.ConditionsUnavailableError, match="calendar-unavailable"):
            ctc.get_current_conditions("1d", symbol="BID", now_ms=1_800_000_000_000)
    loader.assert_not_called()


def test_get_current_conditions_db_backed_uses_table(monkeypatch):
    days = _trading_days(date(2026, 8, 24), 12, non_trading=frozenset({HOLIDAY}))
    rows = _daily_rows(days)
    _install_db(monkeypatch, FakeCursor(_calendar_rows(HOLIDAY)))
    ctc.clear_cache()
    with patch.object(ctc, "_load_inputs", return_value=(rows, [], sorted(CANONICAL_SMC_COLUMNS), None)):
        payload = ctc.get_current_conditions("1d", symbol="BID", now_ms=int(rows[-1]["CloseTimeMs"]) + 1)
    ctc.clear_cache()
    assert payload["analysisWindow"]["bars"] == 12
    assert payload["calendar"]["source"] == "db:MarketCalendarDays"


def test_btc_conditions_never_need_the_hose_calendar():
    with patch.object(ctc, "_load_calendar", side_effect=AssertionError("BTC must not load HOSE")), \
            patch.object(ctc, "_load_inputs", return_value=([], [], [], None)):
        with pytest.raises(ctc.ConditionsUnavailableError):
            ctc.get_current_conditions("4h", symbol="BTCUSDT", now_ms=1_800_000_000_000)


# --- base rates + replay provenance / fail closed ---------------------------------


def _walk_rows(n=260):
    days = _trading_days(date(2025, 1, 2), n, non_trading=frozenset({date(2025, 4, 30), date(2025, 5, 1)}))
    out = []
    price = 20_000.0
    for i, day in enumerate(days):
        open_ms = vn.daily_open_ms(day)
        price *= 1.0 + (0.01 if i % 3 else -0.012)
        out.append({
            "OpenTimeMs": open_ms, "CloseTimeMs": vn.session_close_ms("1d", open_ms),
            "Open": price, "High": price * 1.01, "Low": price * 0.99, "Close": price,
            "Volume": 1_000_000.0, "PriceBasis": "adjusted", "Provider": "VCI",
        })
    return out


def test_base_rates_records_calendar_provenance():
    cal = vn.calendar_from_dates({date(2025, 4, 30), date(2025, 5, 1)}, source="db:MarketCalendarDays")
    artifact, report = ebr.build_artifact(_walk_rows(), "BID", cutoff_ms=10**15, calendar=cal)
    assert artifact["calendar"]["source"] == "db:MarketCalendarDays"
    assert artifact["calendar"]["count"] == 2
    assert report["reproduce"]["calendar"] == artifact["calendar"]
    assert ebr.summarize(artifact)["calendar"]["count"] == 2


def test_base_rates_without_calendar_fails_closed():
    with pytest.raises(ebr.InsufficientDataError) as info:
        ebr.build_artifact(_walk_rows(), "BID", cutoff_ms=10**15)
    assert info.value.reason == "calendar-unavailable"


def test_base_rates_db_job_fails_closed_on_empty_table(monkeypatch, tmp_path):
    _install_db(monkeypatch, FakeCursor([]))
    with pytest.raises(ebr.InsufficientDataError) as info:
        ebr.build_and_write("BID", rows=_walk_rows(), out_root=tmp_path)
    assert info.value.reason == "calendar-unavailable"
    assert not (tmp_path / "equity-base-rates").exists()


def test_replay_db_run_fails_closed_on_empty_table(monkeypatch, tmp_path, capsys):
    _install_db(monkeypatch, FakeCursor([]))
    with patch.object(replay, "load_candles_db", side_effect=AssertionError("no candle read without calendar")):
        code = replay.main(["--symbol", "BID", "--strategy", "all", "--out-dir", str(tmp_path)])
    assert code == 2
    assert "reason=calendar-unavailable" in capsys.readouterr().out


def test_replay_records_db_calendar(monkeypatch, tmp_path):
    _install_db(monkeypatch, FakeCursor(_calendar_rows(date(2025, 4, 30), date(2025, 5, 1))))
    rows = _walk_rows(80)
    with patch.object(replay, "load_candles_db", side_effect=lambda symbol, tf, cutoff: rows if tf == "1d" else []):
        code = replay.main(["--symbol", "BID", "--strategy", "all", "--out-dir", str(tmp_path)])
    assert code == 0
    import json

    # EQTECH-1 H: BID declares 1h, so "all" writes the 1h RSI report too; each carries the DB calendar.
    artifacts = sorted(tmp_path.glob("*.json"))
    assert [json.loads(a.read_text(encoding="utf-8"))["timeframe"] for a in artifacts] == ["1h", "1d"]
    for artifact in artifacts:
        report = json.loads(artifact.read_text(encoding="utf-8"))
        assert report["calendar"]["source"] == "db:MarketCalendarDays"
        assert report["calendar"]["count"] == 2
        assert report["calendar"]["nonTradingDates"] == ["2025-04-30", "2025-05-01"]


# --- technical descriptive evidence (DB-backed MBB bundles) -----------------------


class SequencedCursor(FakeCursor):
    """Answers the calendar query, then the candle query (in that order)."""

    def __init__(self, calendar_rows, candle_rows):
        super().__init__()
        self._answers = [calendar_rows, candle_rows]

    def execute(self, sql, params=()):
        self.executed.append((sql, params))
        self.rows = self._answers.pop(0) if self._answers else []


class SessionConnection(FakeConnection):
    def set_session(self, **kwargs):
        self.session = kwargs


def _candle_tuples(rows):
    return [(r["OpenTimeMs"], r["CloseTimeMs"], r["Open"], r["High"], r["Low"], r["Close"], r["Volume"]) for r in rows]


def test_descriptive_evidence_loader_reads_calendar_in_same_snapshot(monkeypatch):
    import technical_event_descriptive_evidence as ted

    days = _trading_days(date(2026, 8, 24), 12, non_trading=frozenset({HOLIDAY}))
    rows = _daily_rows(days)
    cursor = SequencedCursor(_calendar_rows(HOLIDAY), _candle_tuples(rows))
    connection = SessionConnection(cursor)
    module = types.ModuleType("db_config")
    module.get_db_connection = lambda: connection
    monkeypatch.setitem(sys.modules, "db_config", module)

    snapshot = ted.load_postgresql_snapshot("1d", int(rows[-1]["CloseTimeMs"]), symbol="MBB")

    assert '"MarketCalendarDays"' in cursor.executed[0][0]
    assert '"InstrumentCandles"' in cursor.executed[1][0]
    assert connection.session["isolation_level"] == "REPEATABLE READ"
    session = snapshot["modules"]["finalizedCandles"]["sessionCalendar"]
    assert session["nonTradingDates"] == ["2026-09-02"]
    assert session["provenance"]["source"] == "db:MarketCalendarDays"
    # The bundle re-parses with its embedded calendar even with no env/DB.
    dataset = ted.parse_snapshot(snapshot, int(rows[-1]["CloseTimeMs"]))
    assert len(dataset.candles) == 12


def test_descriptive_evidence_loader_fails_closed_on_empty_table(monkeypatch):
    import technical_event_descriptive_evidence as ted

    cursor = SequencedCursor([], [])
    module = types.ModuleType("db_config")
    module.get_db_connection = lambda: SessionConnection(cursor)
    monkeypatch.setitem(sys.modules, "db_config", module)
    with pytest.raises(vn.CalendarUnavailableError):
        ted.load_postgresql_snapshot("1d", 1_800_000_000_000, symbol="MBB")
    assert len(cursor.executed) == 1  # no candle read without a calendar


def test_descriptive_evidence_pure_paths_fail_closed_without_calendar():
    import technical_event_descriptive_evidence as ted

    rows = _daily_rows(_trading_days(date(2026, 8, 24), 5))
    with pytest.raises(vn.CalendarUnavailableError):
        ted.build_postgresql_snapshot_from_rows("1d", int(rows[-1]["CloseTimeMs"]), rows, [], smc_columns=[], symbol="MBB")
    snapshot = ted.build_postgresql_snapshot_from_rows(
        "1d", int(rows[-1]["CloseTimeMs"]), rows, [], smc_columns=[], symbol="MBB", non_trading_dates=frozenset())
    assert snapshot["modules"]["finalizedCandles"]["sessionCalendar"]["provenance"]["source"] == "explicit"
    stripped = {**snapshot, "modules": {k: v for k, v in snapshot["modules"].items() if k != "finalizedCandles"}}
    with pytest.raises(vn.CalendarUnavailableError):
        ted.parse_snapshot(stripped, int(rows[-1]["CloseTimeMs"]))


def test_base_rates_build_endpoint_maps_calendar_unavailable_to_503(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import main

    _install_db(monkeypatch, FakeCursor([]))
    monkeypatch.setenv("EQUITY_BASE_RATES_EVIDENCE_ROOT", str(tmp_path))
    with patch.object(ebr, "load_candles_db", return_value=_walk_rows()):
        response = TestClient(main.app).post("/api/equity-base-rates/build?symbol=BID")
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "CALENDAR_UNAVAILABLE" and detail["reason"] == "calendar-unavailable"
    assert not (tmp_path / "equity-base-rates").exists()


# --- coverage window (review-v2 #1/#7) ---------------------------------------------


def test_loader_reads_coverage_and_intersects_versions():
    rows = _calendar_rows(date(2026, 9, 2)) + _calendar_rows(
        date(2021, 2, 11), version="older", coverage=(date(2019, 1, 1), date(2026, 6, 30)))
    cal = vn.load_calendar_from_cursor(FakeCursor(rows))
    assert (cal.coverage_from, cal.coverage_to) == (date(2020, 1, 1), date(2026, 6, 30))
    assert cal.provenance()["coverageTo"] == "2026-06-30"
    with pytest.raises(vn.CalendarUnavailableError):
        vn.load_calendar_from_cursor(FakeCursor([(date(2026, 9, 2), "verified", "v", None, None)]))


def _cal_2026():
    return vn.MarketCalendar(frozenset({HOLIDAY, date(2026, 1, 1)}), vn.SOURCE_DB,
                             coverage_from=COVERAGE[0], coverage_to=COVERAGE[1])


def _rows_into_2027(n=12):
    return _daily_rows(_trading_days(date(2026, 12, 21), n))  # runs into January 2027


def test_conditions_fail_closed_after_coverage_end_2027():
    rows = _rows_into_2027()
    assert vn.to_local(rows[-1]["OpenTimeMs"]).date().year == 2027
    with pytest.raises(ctc.ConditionsUnavailableError, match="calendar-coverage-expired"):
        build_conditions_payload("1d", rows, [], sorted(CANONICAL_SMC_COLUMNS),
                                 now_ms=int(rows[-1]["CloseTimeMs"]), symbol="MBB", calendar=_cal_2026())
    inside = [r for r in rows if vn.to_local(r["OpenTimeMs"]).date().year == 2026]
    payload = build_conditions_payload("1d", inside, [], sorted(CANONICAL_SMC_COLUMNS),
                                       now_ms=int(inside[-1]["CloseTimeMs"]), symbol="MBB", calendar=_cal_2026())
    assert payload["calendar"]["coverageTo"] == "2026-12-31"


def test_conditions_window_stops_at_coverage_start_with_explicit_note():
    rows = _daily_rows(_trading_days(date(2019, 12, 16), 20, non_trading=frozenset({date(2020, 1, 1)})))
    payload = build_conditions_payload("1d", rows, [], sorted(CANONICAL_SMC_COLUMNS),
                                       now_ms=int(rows[-1]["CloseTimeMs"]), symbol="MBB", calendar=_cal_2026())
    first = vn.to_local(payload["analysisWindow"]["firstOpenTimeMs"]).date()
    assert first == date(2020, 1, 2)
    assert "below-calendar-coverage" in payload["warnings"]
    assert "analysis-window-truncated-at-non-contiguous-gap" not in payload["warnings"]


def test_base_rates_fail_closed_after_coverage_end(monkeypatch, tmp_path):
    rows = _walk_rows(300)
    shifted = _daily_rows(_trading_days(date(2027, 1, 4), 1))[0] | {"PriceBasis": "adjusted", "Provider": "VCI"}
    with pytest.raises(ebr.InsufficientDataError) as info:
        ebr.build_artifact(rows + [shifted], "BID", cutoff_ms=10**15, calendar=_cal_2026())
    assert info.value.reason == "calendar-coverage-expired"
    artifact, _ = ebr.build_artifact(rows, "BID", cutoff_ms=10**15, calendar=vn.MarketCalendar(
        frozenset({date(2025, 4, 30)}), vn.SOURCE_DB, coverage_from=date(2025, 6, 2), coverage_to=COVERAGE[1]))
    assert artifact["calendar"]["barsBeforeCoverage"] > 0  # flagged, protocol unchanged

    from fastapi.testclient import TestClient

    import main

    _install_db(monkeypatch, FakeCursor(_calendar_rows(HOLIDAY)))
    monkeypatch.setenv("EQUITY_BASE_RATES_EVIDENCE_ROOT", str(tmp_path))
    monkeypatch.setattr(ebr, "time", types.SimpleNamespace(time=lambda: 1_801_000_000.0))  # 2027-01-26
    with patch.object(ebr, "load_candles_db", return_value=rows + [shifted]):
        response = TestClient(main.app).post("/api/equity-base-rates/build?symbol=BID")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "CALENDAR_COVERAGE_EXPIRED"
    assert not (tmp_path / "equity-base-rates").exists()


def test_replay_fails_closed_after_coverage_end(monkeypatch, tmp_path, capsys):
    _install_db(monkeypatch, FakeCursor(_calendar_rows(HOLIDAY)))
    rows = [r | {"PriceBasis": "adjusted"} for r in _rows_into_2027(80)]
    with patch.object(replay, "load_candles_db", return_value=rows):
        code = replay.main(["--symbol", "BID", "--strategy", "all", "--out-dir", str(tmp_path),
                            "--cutoff-ms", str(rows[-1]["CloseTimeMs"])])
    assert code == 2
    assert "reason=calendar-coverage-expired" in capsys.readouterr().out


def test_descriptive_evidence_fails_closed_after_coverage_end():
    import technical_event_descriptive_evidence as ted

    rows = _rows_into_2027()
    with pytest.raises(vn.CalendarCoverageError):
        ted.build_postgresql_snapshot_from_rows("1d", int(rows[-1]["CloseTimeMs"]), rows, [], smc_columns=[],
                                                symbol="MBB", calendar=_cal_2026())


def test_embedded_empty_calendar_does_not_bypass_fail_closed():
    import technical_event_descriptive_evidence as ted

    rows = _daily_rows(_trading_days(date(2026, 8, 24), 5))
    snapshot = ted.build_postgresql_snapshot_from_rows(
        "1d", int(rows[-1]["CloseTimeMs"]), rows, [], smc_columns=[], symbol="MBB", non_trading_dates=frozenset())
    assert snapshot["modules"]["finalizedCandles"]["sessionCalendar"]["nonTradingDates"] == []
    with pytest.raises(vn.CalendarUnavailableError):  # pre-v2 style bundle: empty list ⇒ missing
        ted.parse_snapshot(snapshot, int(rows[-1]["CloseTimeMs"]))
    # A bundle built with a real calendar keeps its coverage when re-parsed.
    built = ted.build_postgresql_snapshot_from_rows(
        "1d", int(rows[-1]["CloseTimeMs"]), rows, [], smc_columns=[], symbol="MBB", calendar=_cal_2026())
    assert built["modules"]["finalizedCandles"]["sessionCalendar"]["provenance"]["coverageTo"] == "2026-12-31"
    assert len(ted.parse_snapshot(built, int(rows[-1]["CloseTimeMs"])).candles) == 5
