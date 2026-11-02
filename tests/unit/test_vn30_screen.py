"""VN30-SCREEN: the daily table ranks the basket in effect with the vn30_rotation rule, reads no bar after the
ranking session, leaves gaps blank instead of guessing, and flags stale data."""
import hashlib
import json
import sys
import urllib.error
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

AI_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(AI_ROOT))

import vn30_rotation as vr  # noqa: E402
import vn30_screen as screen  # noqa: E402

EVIDENCE_MEMBERSHIP = AI_ROOT.parent / "evidence" / "vn30-1-20261102" / "membership" / "vn30-membership.json"


def _sessions(n: int, start: date = date(2025, 1, 1)) -> list[str]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _ms(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000)


def _candles(closes: dict[str, float], volume: dict[str, float] | None = None) -> list[dict]:
    # open/close ratio varies by session, so returns computed from opens would differ from close-to-close
    return [{"openTimeMs": _ms(d), "open": c * (0.95 + 0.01 * (i % 5)), "high": c * 1.01, "low": c * 0.94,
             "close": c, "volume": (volume or {}).get(d, 1000.0)} for i, (d, c) in enumerate(sorted(closes.items()))]


class FakeBackend:
    """Honours `limit` like the backend (latest N bars) but ignores endTimeMs, so the script's own cut-off is what
    gets tested. Unknown symbols answer 404 like EquitiesController."""

    def __init__(self, bars: dict[str, list[dict]], not_up_to_date: set[str] = frozenset()):
        self.bars, self.not_up_to_date, self.paths = bars, not_up_to_date, []

    def __call__(self, path: str):
        self.paths.append(path)
        if path == "/api/equities":
            return {"items": [{"symbol": s, "ingestion": {"timeframes": [
                {"timeframe": "1d", "upToDate": s not in self.not_up_to_date}]}} for s in self.bars]}
        symbol = path.split("/")[3]
        if symbol not in self.bars:
            raise urllib.error.HTTPError(path, 404, "Not Found", {}, None)
        limit = int(path.split("limit=")[1].split("&")[0])
        return self.bars[symbol][-limit:]


def _market(n: int = 300, members: int = 12):
    days = _sessions(n)
    closes = {"VNINDEX": {d: 1000.0 + i for i, d in enumerate(days)}}
    for k in range(members):
        # distinct growth rates and a wobble so 3/6/12-month ranks disagree
        closes[f"S{k:02d}"] = {d: 50.0 * (1 + 0.0004 * (k - 5)) ** i * (1 + 0.05 * ((i // 40 + k) % 3))
                               for i, d in enumerate(days)}
    return days, closes


def _periods(symbols, effective: str = "2024-12-01") -> vr.Periods:
    return [(effective, frozenset(symbols))]


def test_top_ten_is_exactly_the_engine_selection_and_the_ranks_agree():
    days, closes = _market()
    members = sorted(s for s in closes if s != "VNINDEX")
    out = screen.build(FakeBackend({s: _candles(c) for s, c in closes.items()}), _periods(members))
    prices = {s: {d: (c * 0.97, c) for d, c in closes[s].items()} for s in closes}
    expected = vr.select(sorted(closes["VNINDEX"]), prices, frozenset(members), len(days) - 1, 6, 10)
    top = {r["symbol"] for r in out["rows"] if r["topByRankMonths"]}
    assert top == set(expected) and len(top) == 10
    assert top == {r["symbol"] for r in out["rows"] if r["ranks"]["6"] <= 10}
    assert [r["ranks"]["6"] for r in out["rows"]] == list(range(1, len(members) + 1))
    assert any(r["ranks"]["3"] != r["ranks"]["6"] for r in out["rows"])


def test_returns_use_closes_over_21_sessions_per_month_and_excess_is_against_vnindex():
    days, closes = _market()
    out = screen.build(FakeBackend({s: _candles(c) for s, c in closes.items()}), _periods(["S03", "S07"]))
    day = days[-1]
    assert out["session"] == day
    for m in (3, 6, 12):
        past = days[-1 - 21 * m]
        assert out["anchorSessions"][str(m)] == past
        idx = closes["VNINDEX"][day] / closes["VNINDEX"][past] - 1
        assert out["indexReturns"][str(m)] == pytest.approx(idx)
        for row in out["rows"]:
            r = closes[row["symbol"]][day] / closes[row["symbol"]][past] - 1
            assert row["returns"][str(m)] == pytest.approx(r)
            assert row["excessVsIndex"][str(m)] == pytest.approx(r - idx)
            assert row["close"] == closes[row["symbol"]][day]


def test_zero_volume_bars_are_not_sessions():
    days, closes = _market(members=3)
    vol = {"S00": {days[-1]: 0.0}, "VNINDEX": {days[-2]: 0.0}}
    out = screen.build(FakeBackend({s: _candles(c, vol.get(s)) for s, c in closes.items()}),
                       _periods(["S00", "S01", "S02"]))
    calendar = [d for d in days if d != days[-2]]
    assert out["anchorSessions"]["3"] == calendar[-1 - 63]  # the zero-volume VNINDEX day is not counted
    s00 = next(r for r in out["rows"] if r["symbol"] == "S00")
    assert s00["returns"] == {"3": None, "6": None, "12": None} and s00["close"] is None
    assert s00["ranks"] == {"3": None, "6": None, "12": None} and not s00["topByRankMonths"]
    assert out["missingSessionBar"] == ["S00"]
    assert out["rows"][-1]["symbol"] == "S00"  # unranked rows go last


def test_a_member_without_the_anchor_bar_or_enough_history_is_left_blank():
    days, closes = _market(members=3)
    del closes["S01"][days[-1 - 63]]  # halted on the 3-month anchor
    closes["S02"] = {d: c for d, c in closes["S02"].items() if d >= days[-200]}  # listed ~200 sessions ago
    out = screen.build(FakeBackend({s: _candles(c) for s, c in closes.items()}), _periods(["S00", "S01", "S02"]))
    rows = {r["symbol"]: r for r in out["rows"]}
    assert rows["S01"]["returns"]["3"] is None and rows["S01"]["returns"]["6"] is not None
    assert rows["S01"]["ranks"]["3"] is None
    assert rows["S02"]["returns"]["12"] is None and rows["S02"]["returns"]["6"] is not None
    assert rows["S02"]["excessVsIndex"]["12"] is None
    assert out["missingSessionBar"] == []


def test_as_of_reads_no_bar_after_the_ranking_session():
    days, closes = _market()
    members = sorted(s for s in closes if s != "VNINDEX")
    cut = days[-30]
    truncated = {s: {d: c for d, c in cs.items() if d <= cut} for s, cs in closes.items()}
    later = {s: dict(cs) for s, cs in truncated.items()}
    for s, cs in closes.items():  # future bars that would reorder everything if they leaked in
        for d in days:
            if d > cut:
                later[s][d] = cs[d] * (10.0 if s in ("S00", "S01") else 0.1)
    a = screen.build(FakeBackend({s: _candles(c) for s, c in later.items()}), _periods(members), as_of=cut)
    b = screen.build(FakeBackend({s: _candles(c) for s, c in truncated.items()}), _periods(members))
    assert a["session"] == cut
    assert [(r["symbol"], r["returns"], r["ranks"], r["topByRankMonths"]) for r in a["rows"]] == \
           [(r["symbol"], r["returns"], r["ranks"], r["topByRankMonths"]) for r in b["rows"]]


def test_as_of_asks_the_backend_to_stop_at_the_end_of_that_vietnam_day():
    days, closes = _market(members=1)
    fake = FakeBackend({s: _candles(c) for s, c in closes.items()})
    screen.build(fake, _periods(["S00"]), as_of=days[-5])
    next_day = (date.fromisoformat(days[-5]) + timedelta(days=1)).isoformat()
    end_ms = _ms(next_day) - 7 * 3600 * 1000 - 1  # 23:59:59.999 in Vietnam on the ranking day
    assert all(f"endTimeMs={end_ms}" in p for p in fake.paths if "/candles" in p)


def test_staleness_flags():
    days, closes = _market(members=2)
    fake = FakeBackend({s: _candles(c) for s, c in closes.items()}, not_up_to_date={"S01"})
    recent = (date.fromisoformat(days[-1]) - timedelta(days=100)).isoformat()
    fresh = screen.build(fake, _periods(["S00", "S01"], effective=recent))
    assert fresh["notUpToDateNow"] == ["S01"]
    assert fresh["basket"]["possiblyStale"] is False
    old = (date.fromisoformat(days[-1]) - timedelta(days=191)).isoformat()
    assert screen.build(fake, _periods(["S00", "S01"], effective=old))["basket"]["possiblyStale"] is True
    edge = (date.fromisoformat(days[-1]) - timedelta(days=190)).isoformat()
    assert screen.build(fake, _periods(["S00", "S01"], effective=edge))["basket"]["possiblyStale"] is False


def test_a_late_vnindex_is_flagged_because_it_sets_the_ranking_session():
    days, closes = _market(members=2)
    fake = FakeBackend({s: _candles(c) for s, c in closes.items()}, not_up_to_date={"VNINDEX"})
    assert screen.build(fake, _periods(["S00", "S01"]))["notUpToDateNow"] == ["VNINDEX"]


def test_every_warning_reaches_the_markdown_the_user_reads():
    days, closes = _market(members=3)
    vol = {"S00": {days[-1]: 0.0}}
    fake = FakeBackend({s: _candles(c, vol.get(s)) for s, c in closes.items()}, not_up_to_date={"S01"})
    old = (date.fromisoformat(days[-1]) - timedelta(days=200)).isoformat()
    md = screen.to_markdown(screen.build(fake, _periods(["S00", "S01", "S02", "ZZZ"], effective=old)))
    assert "**Rổ có thể đã cũ**" in md
    assert f"Chưa có nến phiên {days[-1]}: S00, ZZZ." in md
    assert "Backend báo chưa nạp tới phiên mới nhất (lúc chạy): S01, ZZZ." in md
    assert "**Backend chưa có mã** (cập nhật danh mục mã của backend): ZZZ." in md
    table = [line for line in md.splitlines() if line.startswith("| ") and "Mã" not in line]
    assert table[0].startswith("| 1 | ") and table[-1].startswith("| — | ")


def test_a_member_the_backend_does_not_know_leaves_a_blank_row_not_a_failed_table():
    days, closes = _market(members=2)
    out = screen.build(FakeBackend({s: _candles(c) for s, c in closes.items()}), _periods(["S00", "S01", "ZZZ"]))
    rows = {r["symbol"]: r for r in out["rows"]}
    assert out["unknownToBackend"] == ["ZZZ"] and rows["ZZZ"]["returns"] == {"3": None, "6": None, "12": None}
    assert rows["S00"]["ranks"]["6"] is not None and out["missingSessionBar"] == ["ZZZ"]

    def server_error(path):
        if "/ZZZ/" in path:
            raise urllib.error.HTTPError(path, 500, "Server Error", {}, None)
        return FakeBackend({s: _candles(c) for s, c in closes.items()})(path)
    with pytest.raises(urllib.error.HTTPError):
        screen.build(server_error, _periods(["S00", "ZZZ"]))


def test_sessions_are_vietnam_calendar_days():
    assert screen.session_of(_ms("2026-11-01") + 17 * 3600 * 1000) == "2026-11-02"  # 00:00 in Vietnam
    assert screen.session_of(_ms("2026-11-02")) == "2026-11-02"


def test_the_basket_in_effect_on_the_ranking_session_is_used():
    days, closes = _market(members=4)
    periods = [("2024-01-01", frozenset({"S00", "S01"})), (days[-10], frozenset({"S02", "S03"}))]
    fake = FakeBackend({s: _candles(c) for s, c in closes.items()})
    assert {r["symbol"] for r in screen.build(fake, periods)["rows"]} == {"S02", "S03"}
    early = screen.build(fake, periods, as_of=days[-11])
    assert {r["symbol"] for r in early["rows"]} == {"S00", "S01"} and early["basket"]["effectiveDate"] == "2024-01-01"


def test_main_writes_json_and_markdown_and_maps_failures_to_exit_codes(tmp_path):
    days, closes = _market(members=3)
    members_file = tmp_path / "m.json"
    syms = [f"S{k:02d}" for k in range(3)] + [f"X{k:02d}" for k in range(27)]
    members_file.write_text(json.dumps({"periods": [{"effectiveDate": "2024-12-01", "constituents": syms}]}))
    bars = {s: _candles(c) for s, c in closes.items()} | {s: [] for s in syms[3:]}
    out = tmp_path / "out"
    assert screen.main(["--out-dir", str(out), "--membership", str(members_file)], fetch=FakeBackend(bars)) == 0
    data = json.loads((out / f"{days[-1]}.json").read_text(encoding="utf-8"))
    assert data["schema"] == "vn30-screen/v1" and len(data["rows"]) == 30
    md = (out / "latest.md").read_text(encoding="utf-8")
    assert md == (out / f"{days[-1]}.md").read_text(encoding="utf-8")
    assert screen.CAVEAT in md and "| S" in md and "X26" in md

    def broken(path):
        raise screen.urllib.error.URLError("connection refused")
    assert screen.main(["--out-dir", str(tmp_path / "b"), "--membership", str(members_file)], fetch=broken) == 1
    weekend = date.fromisoformat(days[100])
    while weekend.weekday() < 5:
        weekend += timedelta(days=1)
    assert screen.main(["--out-dir", str(tmp_path / "c"), "--membership", str(members_file),
                        "--as-of", weekend.isoformat()], fetch=FakeBackend(bars)) == 2  # not a VNINDEX session
    assert screen.main(["--out-dir", str(tmp_path / "d"), "--membership", str(members_file)],
                       fetch=FakeBackend(bars | {"VNINDEX": []})) == 2
    bad = [dict(c, close="n/a") if i == len(bars["S00"]) - 1 else c for i, c in enumerate(bars["S00"])]
    assert screen.main(["--out-dir", str(tmp_path / "e"), "--membership", str(members_file)],
                       fetch=FakeBackend(bars | {"S00": bad})) == 1  # malformed payload is a backend fault
    for d in "bcde":
        assert not (tmp_path / d).exists()

    back = tmp_path / "back"
    assert screen.main(["--out-dir", str(back), "--membership", str(members_file), "--as-of", days[-5]],
                       fetch=FakeBackend(bars)) == 0
    assert (back / f"{days[-5]}.md").exists() and not (back / "latest.md").exists()


def test_repo_membership_is_the_vn30_1_file_and_ends_on_the_august_2026_basket():
    periods = vr.load_membership(screen.DEFAULT_MEMBERSHIP)
    assert periods[-1][0] == "2026-08-03" and len(periods) == 19
    assert {"MBB", "BID", "VCB", "CTG", "TCX", "MCH", "BSR"} <= periods[-1][1]
    if EVIDENCE_MEMBERSHIP.exists():  # the frozen VN30-1 copy lives outside the repo and is CRLF
        lf = EVIDENCE_MEMBERSHIP.read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(screen.DEFAULT_MEMBERSHIP.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == \
               hashlib.sha256(lf).hexdigest()
