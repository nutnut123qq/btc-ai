"""Forward-paper fill/outcome observer.

Unit tests use a fake clock + fake quote fetcher (no network). The PostgreSQL
tests create a uniquely named temporary database on localhost, apply the real
recorder SCHEMA_SQL (constraints + write-once trigger), and drop only that DB.
They skip when psycopg2 or local PostgreSQL is unavailable. Because the UPDATEs
are bounded by the database NOW(), PG tests use the real clock and place rows
in the past with an explicit CreatedAtUtc instead of faking the observer clock.
"""

from __future__ import annotations

import itertools
import json
import os
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import Mock

import pytest

import paper_observer as po
from forward_paper_recorder import RECORDER_VERSION, LiveQuote, deterministic_decision_id, ensure_observation_schema

T0 = datetime(2026, 10, 23, 4, 0, 5, tzinfo=timezone.utc)
MIN = timedelta(minutes=1)
H4 = timedelta(milliseconds=po.OUTCOME_HORIZON_MS)
WINDOW = timedelta(milliseconds=po.FILL_WINDOW_MS)
MS = timedelta(milliseconds=1)


def _quote(at: datetime, price: str = "50000") -> LiveQuote:
    return LiveQuote("binance-spot-bookTicker-bid-ask-mid", Decimal(price), at, po._ms(at))


def _row(decision="long", observed=T0, fill_price=None, fill_at=None, row_id="r1"):
    return (row_id, decision, observed, po._ms(observed), fill_price, fill_at)


# ---------------------------------------------------------------- pure planner

def test_abstain_never_gets_fill_or_outcome():
    assert po.plan_writes([_row("abstain")], _quote(T0 + MIN)) == []
    assert po.plan_writes([_row("abstain", fill_price=Decimal(1), fill_at=T0)], _quote(T0 + H4 + MIN)) == []


@pytest.mark.parametrize(("offset", "fills"), [(-MS, False), (timedelta(0), True), (WINDOW, True), (WINDOW + MS, False)])
def test_fill_window_boundaries(offset, fills):
    assert bool(po.plan_writes([_row()], _quote(T0 + offset))) is fills


@pytest.mark.parametrize(("offset", "writes"), [(H4 - MS, False), (H4, True), (H4 + WINDOW, True), (H4 + WINDOW + MS, False)])
def test_outcome_window_boundaries(offset, writes):
    row = _row(fill_price=Decimal("50000"), fill_at=T0)
    assert bool(po.plan_writes([row], _quote(T0 + offset))) is writes


def test_valid_quote_fills_at_observed_mid_with_source_and_receipt_time():
    quote = _quote(T0 + MIN, "50123.5")
    [(kind, row_id, price, at, source)] = po.plan_writes([_row()], quote)
    assert (kind, row_id, price, at) == ("fill", "r1", Decimal("50123.5"), quote.received_at_utc)
    assert source.startswith("binance-spot-bookTicker-bid-ask-mid|receivedTimeMs=")
    assert po.OBSERVER_VERSION in source


@pytest.mark.parametrize(("decision", "expected"), [("long", 0.02), ("short", -0.02)])
def test_outcome_in_window_is_gross_directional_return(decision, expected):
    row = _row(decision, fill_price=Decimal("50000"), fill_at=T0)
    [(kind, _, gross, at, horizon)] = po.plan_writes([row], _quote(T0 + H4 + MIN, "51000"))
    assert kind == "outcome" and horizon == "4h-gross" and at == T0 + H4 + MIN
    assert gross == pytest.approx(expected)


def test_quote_rejection_rules():
    now = T0 + MIN
    assert po.quote_rejection(_quote(now), now) is None
    assert po.quote_rejection(_quote(now - timedelta(seconds=6)), now) == "quote-stale"
    assert po.quote_rejection(_quote(now + timedelta(seconds=2)), now) == "quote-from-future"
    assert po.quote_rejection(_quote(now, "0"), now) == "quote-nonpositive"
    skewed = LiveQuote("x", Decimal(1), now, po._ms(now) - 500)
    assert po.quote_rejection(skewed, now) == "quote-timestamp-mismatch"
    assert po.quote_rejection(None, now) == "quote-invalid-type"


def test_updates_are_conditional_server_clocked_and_never_touch_decision_columns():
    assert '"FillPrice" IS NULL' in po.FILL_UPDATE_SQL
    assert '"OutcomeReturn" IS NULL' in po.OUTCOME_UPDATE_SQL
    for sql in (po.FILL_UPDATE_SQL, po.OUTCOME_UPDATE_SQL):
        assert "NOW()" in sql
        set_clause = sql.split("SET", 1)[1].split("WHERE", 1)[0]
        assert "Decision" not in set_clause and "Quote" not in set_clause
    assert all("Klines" not in sql for sql in (po.SELECT_CANDIDATES_SQL, po.FILL_UPDATE_SQL, po.OUTCOME_UPDATE_SQL))


# ---------------------------------------------------------- fake connection

class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.executed = []

    def execute(self, sql, params=()):
        self.executed.append(sql)

    def fetchall(self):
        return list(self.rows)

    def fetchone(self):
        return ("id",)

    def close(self):
        pass


class FakeConn:
    def __init__(self, rows):
        self.cur = FakeCursor(rows)

    def cursor(self):
        return self.cur

    def commit(self):
        pass


def _updates(conn):
    return [sql for sql in conn.cur.executed if sql.lstrip().startswith("UPDATE")]


def test_no_candidates_means_no_quote_fetch_and_no_write():
    conn, fetcher = FakeConn([]), Mock()
    assert po._observe_once(conn, quote_fetcher=fetcher, clock=lambda: T0)["candidates"] == 0
    fetcher.assert_not_called()
    assert _updates(conn) == []


@pytest.mark.parametrize(
    ("fetcher", "reason"),
    [
        (Mock(side_effect=OSError("down")), "unavailable:OSError"),
        (Mock(return_value=_quote(T0 + MIN - timedelta(seconds=30))), "quote-stale"),
        (Mock(return_value=None), "quote-invalid-type"),
    ],
)
def test_missing_or_stale_quote_writes_nothing(fetcher, reason):
    conn = FakeConn([_row()])
    summary = po._observe_once(conn, quote_fetcher=fetcher, clock=lambda: T0 + MIN)
    assert summary["quote"] == reason and summary["fills"] == 0
    assert _updates(conn) == []


def test_valid_quote_issues_one_conditional_fill_update():
    conn = FakeConn([_row()])
    summary = po._observe_once(conn, quote_fetcher=lambda s: _quote(T0 + MIN), clock=lambda: T0 + MIN)
    assert summary == {"candidates": 1, "fills": 1, "outcomes": 0, "quote": "ok"}
    assert _updates(conn) == [po.FILL_UPDATE_SQL]


def test_public_observe_once_takes_no_clock_or_fetcher():
    with pytest.raises(TypeError):
        po.observe_once(FakeConn([]), clock=lambda: T0)


def test_cli_requires_once():
    with pytest.raises(SystemExit):
        po.main([])


def test_cli_db_failure_logs_json_to_stdout_and_exits_nonzero(monkeypatch, capsys):
    import db_config

    def refuse():
        raise ConnectionRefusedError("db down")

    monkeypatch.setattr(db_config, "get_db_connection", refuse)
    assert po.main(["--once"]) == 1
    out, err = capsys.readouterr()
    line = json.loads(out.strip())
    assert line["error"] == "ConnectionRefusedError" and "ranAtUtc" in line and err == ""


# ------------------------------------------------------------- PostgreSQL

PG = dict(host="localhost", port=5432, user="postgres", password=os.getenv("DB_PASS", "123456"))
_bars = itertools.count(1)


@pytest.fixture
def pg_connect():
    """One fresh temp DB per test, so rows from other tests cannot enter a window."""
    psycopg2 = pytest.importorskip("psycopg2")
    try:
        admin = psycopg2.connect(dbname="postgres", connect_timeout=3, **PG)
    except psycopg2.OperationalError as exc:
        pytest.skip(f"local PostgreSQL unreachable: {exc}")
    admin.autocommit = True
    name = f"btc_observer_test_{uuid.uuid4().hex[:12]}"
    try:
        with admin.cursor() as cur:
            cur.execute(f'CREATE DATABASE "{name}"')

        def connect():
            return psycopg2.connect(dbname=name, **PG)

        conn = connect()
        try:
            ensure_observation_schema(conn, conn.cursor())
        finally:
            conn.close()
        connect.admin, connect.dbname = admin, name
        yield connect
    finally:
        assert name.startswith("btc_observer_test_") and name != "bitcoin_analyst"
        with admin.cursor() as cur:
            cur.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        admin.close()


def _now():
    return datetime.now(timezone.utc)


def _insert(conn, decision, observed, fill_at=None, fill_price="50000"):
    """Insert a row as if recorded at `observed` (explicit CreatedAtUtc satisfies
    CK_PaperObservations_ProspectiveClock); optionally with an existing fill."""
    available_ms = po._ms(observed)
    close_ms = available_ms - next(_bars) * 1000  # unique signal bar per row
    open_ms = close_ms - 4 * 3600 * 1000
    directional = decision != "abstain"
    fill = (Decimal(fill_price), fill_at, "test-fixture-fill") if fill_at else (None, None, None)
    cur = conn.cursor()
    cur.execute(
        '''INSERT INTO "PaperObservations" (
               "Id", "DecisionId", "RecorderVersion", "Symbol", "Timeframe",
               "SignalBarOpenTimeMs", "SignalBarCloseTimeMs", "ObservedAtUtc", "AvailableTimeMs",
               "ModelVersion", "Decision", "Confidence", "AbstentionReason",
               "QuoteSource", "QuotePrice", "QuoteReceivedAtUtc", "QuoteReceivedTimeMs",
               "ConfigProvenanceJson", "EvidenceProvenanceJson",
               "FillPrice", "FillObservedAtUtc", "FillSource", "CreatedAtUtc")
           VALUES (%s,%s,%s,'BTCUSDT','4h',%s,%s,%s,%s,%s,%s,%s,%s,
                   'binance-spot-bookTicker-bid-ask-mid',49900,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s,%s)
           RETURNING "Id"''',
        (
            str(uuid.uuid4()), deterministic_decision_id("BTCUSDT", "4h", open_ms, close_ms), RECORDER_VERSION,
            open_ms, close_ms, observed, available_ms,
            "test-model" if directional else None, decision,
            0.7 if directional else None, None if directional else "model-unavailable",
            observed, available_ms,
            json.dumps({"costSpec": {"feePerSideBps": 10.0, "slippagePerSideBps": 5.0, "roundTripBps": 30.0}}),
            json.dumps({"isProspectivePaperEvidence": True, "nextBarOpenRead": False}),
            *fill, observed,
        ),
    )
    row_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    return row_id


def _state(conn, row_id):
    cur = conn.cursor()
    cur.execute('SELECT * FROM "PaperObservations" WHERE "Id"=%s', (row_id,))
    names = [d[0] for d in cur.description]
    row = dict(zip(names, cur.fetchone()))
    conn.commit()
    cur.close()
    return row


def _sql(conn, sql, params):
    cur = conn.cursor()
    cur.execute(sql, params)
    written = cur.fetchone() is not None
    conn.commit()
    cur.close()
    return written


FILL_OUTCOME = {"FillPrice", "FillObservedAtUtc", "FillSource", "OutcomeReturn", "OutcomeObservedAtUtc", "OutcomeHorizon"}


def _run(conn, price):
    """Real clock; the fake fetcher returns a quote received 'now'."""
    return po._observe_once(conn, quote_fetcher=lambda s: _quote(_now(), price), clock=_now)


def _core(row):
    return {k: v for k, v in row.items() if k not in FILL_OUTCOME}


def test_pg_fill_written_once_decision_core_unchanged(pg_connect):
    import psycopg2

    conn = pg_connect()
    try:
        t = _now() - MIN
        ids = {d: _insert(conn, d, t) for d in ("long", "short", "abstain")}
        before = {d: _state(conn, i) for d, i in ids.items()}

        assert _run(conn, "50000")["fills"] == 2
        assert _run(conn, "60000")["fills"] == 0  # retry: existing fill unchanged
        after = {d: _state(conn, i) for d, i in ids.items()}
        assert after["long"]["FillPrice"] == after["short"]["FillPrice"] == Decimal("50000")
        assert after["long"]["FillSource"].endswith(po.OBSERVER_VERSION)
        assert all(after["abstain"][c] is None for c in FILL_OUTCOME)
        assert all(_core(after[d]) == _core(before[d]) for d in ids)

        cur = conn.cursor()  # backstop: the trigger rejects a fill rewrite outright
        with pytest.raises(psycopg2.Error, match="write-once"):
            cur.execute('UPDATE "PaperObservations" SET "FillPrice"=1 WHERE "Id"=%s', (ids["long"],))
        conn.rollback()
    finally:
        conn.close()


def test_pg_outcome_written_once_as_gross(pg_connect):
    conn = pg_connect()
    try:
        fill_at = _now() - H4 - MIN
        long_id = _insert(conn, "long", fill_at - MIN, fill_at=fill_at)
        short_id = _insert(conn, "short", fill_at - MIN, fill_at=fill_at)
        early_id = _insert(conn, "long", _now() - H4 + 2 * MIN, fill_at=_now() - H4 + 3 * MIN)
        before = _state(conn, long_id)

        assert _run(conn, "51000")["outcomes"] == 2
        assert _run(conn, "70000")["outcomes"] == 0  # write-once
        long_row, short_row = _state(conn, long_id), _state(conn, short_id)
        assert long_row["OutcomeReturn"] == pytest.approx(0.02)
        assert short_row["OutcomeReturn"] == pytest.approx(-0.02)
        assert long_row["OutcomeHorizon"] == "4h-gross"
        assert _core(long_row) == _core(before)
        assert _state(conn, early_id)["OutcomeReturn"] is None  # horizon not reached
    finally:
        conn.close()


def test_pg_backdated_quote_rejected_by_sql_even_when_planning_is_bypassed(pg_connect):
    conn = pg_connect()
    try:
        t = _now() - timedelta(hours=1)
        unfilled = _insert(conn, "long", t)
        # in the row's own window, but in the past per the database clock
        assert not _sql(conn, po.FILL_UPDATE_SQL, po.fill_params(unfilled, Decimal("50000"), t + MIN, "backfill"))

        fill_at = _now() - timedelta(hours=8)
        filled = _insert(conn, "long", fill_at - MIN, fill_at=fill_at)
        assert not _sql(conn, po.OUTCOME_UPDATE_SQL, po.outcome_params(filled, 0.01, fill_at + H4 + MIN))
        # also a quote dated in the future per the database clock
        assert not _sql(conn, po.FILL_UPDATE_SQL, po.fill_params(unfilled, Decimal("1"), _now() + MIN, "future"))

        assert all(_state(conn, i)[c] is None for i in (unfilled,) for c in FILL_OUTCOME)
        assert _state(conn, filled)["OutcomeReturn"] is None
    finally:
        conn.close()


def test_pg_sql_window_guards_hold_without_planner(pg_connect):
    conn = pg_connect()
    try:
        now = _now()
        stale = _insert(conn, "long", now - WINDOW - MIN)  # fill window already closed
        future = _insert(conn, "long", now + timedelta(seconds=30))  # not yet available
        abstain = _insert(conn, "abstain", now - MIN)
        early = _insert(conn, "long", now - H4 + MIN, fill_at=now - H4 + 2 * MIN)  # before horizon
        late = _insert(conn, "long", now - H4 - WINDOW - 2 * MIN, fill_at=now - H4 - WINDOW - MIN)  # past window

        q = _now()
        for row_id in (stale, future, abstain):
            assert not _sql(conn, po.FILL_UPDATE_SQL, po.fill_params(row_id, Decimal("50000"), q, "direct"))
        for row_id in (early, late):
            assert not _sql(conn, po.OUTCOME_UPDATE_SQL, po.outcome_params(row_id, 0.01, q))
        for row_id in (stale, future, abstain):
            assert _state(conn, row_id)["FillPrice"] is None
        for row_id in (early, late):
            assert _state(conn, row_id)["OutcomeReturn"] is None

        control = _insert(conn, "long", now - MIN)  # same SQL succeeds inside every guard
        assert _sql(conn, po.FILL_UPDATE_SQL, po.fill_params(control, Decimal("50000"), _now(), "direct"))
    finally:
        conn.close()


def test_pg_two_concurrent_fills_leave_a_single_fill(pg_connect):
    a, b = pg_connect(), pg_connect()
    try:
        row_id = _insert(a, "long", _now() - MIN)
        cur_a = a.cursor()  # A holds the row lock with an uncommitted fill
        cur_a.execute(po.FILL_UPDATE_SQL, po.fill_params(row_id, Decimal("50000"), _now(), "racer-a"))
        assert cur_a.fetchone() is not None

        result = {}
        racer = threading.Thread(target=lambda: result.update(_run(b, "60000")))
        racer.start()
        deadline = time.monotonic() + 5  # stays well inside the server-clock tolerance
        while time.monotonic() < deadline:  # wait until B is blocked on A's row lock
            with pg_connect.admin.cursor() as cur:
                cur.execute("SELECT count(*) FROM pg_stat_activity WHERE datname=%s AND wait_event_type='Lock'",
                            (pg_connect.dbname,))
                if cur.fetchone()[0]:
                    break
            time.sleep(0.05)
        else:
            a.rollback()
            pytest.fail("racer never blocked on the row lock")
        a.commit()
        racer.join(10)
        assert not racer.is_alive()
        assert result["candidates"] == 1 and result["fills"] == 0
        row = _state(a, row_id)
        assert row["FillPrice"] == Decimal("50000") and row["FillSource"] == "racer-a"
    finally:
        a.close()
        b.close()


def test_pg_downtime_past_window_never_fills_or_scores(pg_connect):
    conn = pg_connect()
    try:
        missed_fill = _insert(conn, "long", _now() - WINDOW - 2 * MIN)
        missed_both = _insert(conn, "long", _now() - H4 - 5 * MIN)
        assert _run(conn, "50000") == {"candidates": 0, "fills": 0, "outcomes": 0, "quote": "not-needed"}
        assert all(_state(conn, i)[c] is None for i in (missed_fill, missed_both) for c in FILL_OUTCOME)
    finally:
        conn.close()
