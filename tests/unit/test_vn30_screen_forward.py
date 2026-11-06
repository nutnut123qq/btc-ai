"""vn30_screen_forward: immutable forward ledger of the screen's top 10 and its 20-session scoring."""

import json
from datetime import datetime, timedelta, timezone

import pytest

import vn30_screen_forward as fw

NOW = datetime(2026, 11, 6, 9, 30, tzinfo=timezone.utc)
SYMS = [f"S{i:02d}" for i in range(30)]


def _screen(tmp_path, session: str, name: str | None = None):
    rows = [{"symbol": s, "close": 100.0 + i, "returns": {}, "ranks": {"3": 1, "6": i + 1, "12": 1}} for i, s in enumerate(SYMS)]
    path = tmp_path / f"{name or session}.json"
    path.write_text(json.dumps({"schema": "vn30-screen/v1", "session": session, "generatedAtUtc": "x", "rows": rows}),
                    encoding="utf-8")
    return path


def test_record_is_chained_idempotent_and_never_backfills(tmp_path):
    ledger = tmp_path / "l.jsonl"
    r1 = fw.record_from_screen(_screen(tmp_path, "2026-11-05"), NOW)
    assert r1["top"] == SYMS[:10]
    assert fw.append_record(r1, ledger) == "recorded"
    assert fw.append_record(r1, ledger) == "unchanged"
    assert fw.append_record(fw.record_from_screen(_screen(tmp_path, "2026-11-06"), NOW), ledger) == "recorded"
    with pytest.raises(fw.LedgerError, match="backfill"):
        fw.append_record(fw.record_from_screen(_screen(tmp_path, "2026-11-04"), NOW), ledger)
    rows = fw.load_ledger(ledger)
    assert [r["session"] for r in rows] == ["2026-11-05", "2026-11-06"]
    assert rows[1]["prevSha256"] == rows[0]["sha256"]


def test_tampering_is_detected(tmp_path):
    ledger = tmp_path / "l.jsonl"
    fw.append_record(fw.record_from_screen(_screen(tmp_path, "2026-11-05"), NOW), ledger)
    fw.append_record(fw.record_from_screen(_screen(tmp_path, "2026-11-06"), NOW), ledger)
    lines = ledger.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["top"] = list(reversed(first["top"]))
    lines[0] = json.dumps(first, sort_keys=True)
    ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(fw.LedgerError, match="sha256"):
        fw.load_ledger(ledger)


def _calendar(n: int):
    start = datetime(2026, 11, 5)
    return [(start + timedelta(days=i)).date().isoformat() for i in range(n)]


def test_score_buys_next_open_and_sells_open_21_sessions_later(tmp_path):
    cal = _calendar(30)
    record = fw.record_from_screen(_screen(tmp_path, cal[0]), NOW)
    # Top 10 (S00..S09) gain 10% between the open of cal[1] and the open of cal[21]; the rest are flat.
    bars = {s: {d: (100.0, 100.0) for d in cal} for s in SYMS}
    for s in SYMS[:10]:
        bars[s][cal[21]] = (110.0, 110.0)
        bars[s][cal[20]] = (999.0, 999.0)  # a move before the exit open does not count
    scored = fw.score_record(record, cal, bars)
    assert scored["status"] == "scored" and (scored["entry"], scored["exit"]) == (cal[1], cal[21])
    assert scored["top10Gross"] == pytest.approx(0.10)
    assert scored["basketGross"] == pytest.approx(0.10 / 3)
    assert scored["spreadGross"] == pytest.approx(0.10 - 0.10 / 3)
    assert scored["excessNet"] == pytest.approx(0.10 - fw.ROUND_TRIP_COST - 0.10 / 3)
    assert scored["rankIc"] > 0.5


def test_immature_until_the_exit_session_exists(tmp_path):
    cal = _calendar(21)  # d + 21 not stored yet
    record = fw.record_from_screen(_screen(tmp_path, cal[0]), NOW)
    bars = {s: {d: (100.0, 100.0) for d in cal} for s in SYMS}
    assert fw.score_record(record, cal, bars) is None


def test_summary_uses_every_20th_record_as_non_overlapping():
    rows = [{"status": "scored", "spreadGross": 0.01, "excessNet": 0.0, "rankIc": 0.1} for _ in range(41)]
    s = fw.summarize(rows + [{"status": "immature"}])
    assert (s["scored"], s["pending"]) == (41, 1)
    assert s["nonOverlapping"]["spreadGross"]["n"] == 3
