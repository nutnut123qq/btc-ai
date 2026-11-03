"""VN30-WEB exporter: only contract fields reach the public files (never a forbidden note field), parts the backend
does not serve become null/empty plus a meta.unavailable entry, failed candles keep the previous file, /technical
stays under the backend's 10/min limit, and a file is rewritten only when its bytes change."""
import io
import json
import os
import sys
import urllib.error
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

AI_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(AI_ROOT))

import export_vn30_web as web  # noqa: E402

KEY = "test-admin-key-0123456789"
SESSION = "2026-11-02"
SYMBOLS = ("VIC", "MBB", "VCB", "ACB")  # the screen's rank order, not alphabetical
FORBIDDEN = {"actionView", "actionBasis", "probability", "scenarios", "evidence", "counterEvidence", "snapshotId",
             "snapshotSha256", "noteId", "mode", "requestedAsOfUtc", "skillVersion", "methodVersion", "forecastIds",
             "limitations", "missingData", "outcomeCheck", "signalClose", "targetClose", "realized", "baselineCall",
             "methods", "obv", "vwap", "macdNorm", "noteBinding", "timeIso", "caveat", "anchorSessions",
             "unknownToBackend", "ranks", "position", "cash"}
META_KEYS = ["schema", "generatedAtUtc", "session", "symbols", "unavailable", "source"]
SCREEN_KEYS = ["session", "basket", "indexReturns", "rankingMonths", "topK", "missingSessionBar", "notUpToDate", "rows"]
ROW_KEYS = ["symbol", "close", "returns", "excessVsIndex", "rank", "top"]
SYMBOL_KEYS = ["symbol", "label", "isBank", "candles", "indicators", "relativeStrength", "notes", "score"]
INDICATOR_KEYS = ["date", "close", *web.INDICATORS]
NOTE_KEYS = ["signalDate", "horizonSessions", "direction", "thesis", "status", "pendingReason", "targetDate",
             "returnPct", "hit"]


def _sessions(n: int, start: date) -> list[str]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _ms(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000)


def _bars(n: int, base: int, zero: tuple[int, ...] = ()) -> list[dict]:
    # open/high/low/close/volume all differ, so a swapped column shows
    return [{"openTimeMs": _ms(d), "timeIso": d + "T00:00:00Z", "open": base + 10 * i - 3, "high": base + 10 * i + 7,
             "low": base + 10 * i - 9, "close": base + 10 * i, "volume": 0 if i in zero else 1000 + i,
             "closeTimeMs": _ms(d) + 27_900_000, "quoteVolume": None} for i, d in enumerate(_sessions(n, date(2025, 8, 1)))]


def _technical(bars: list[dict], vnindex_status: str = "ok") -> dict:
    last = bars[-1]
    day_ind = {f: 100.0 + k + 0.5 for k, f in enumerate(web.INDICATORS)}
    return {
        "contractVersion": "equity-technical-v1", "symbol": "X", "asOfUtc": "2026-11-02T08:00:00Z",
        "timeframes": [
            {"timeframe": "1h", "candles": [dict(last, openTimeMs=last["openTimeMs"] + 3_600_000, close=1)],
             "indicators": {"openTimeMs": last["openTimeMs"] + 3_600_000, **{f: -1.0 for f in web.INDICATORS}}},
            {"timeframe": "1d", "candles": [{k: last[k] for k in ("openTimeMs", "open", "high", "low", "close", "volume")}],
             "indicators": {"openTimeMs": last["openTimeMs"], "isLatestCandle": True, **day_ind, "macdNorm": 0.25,
                            "obv": 123456, "vwap": 1.5, "methods": {"rsi14": "Wilder RSI(14)"}}},
        ],
        "relativeStrength": {"status": vnindex_status, "reason": None, "windows": [20, 60], "benchmarks": [
            {"symbol": "BID", "kind": "peer", "status": "ok", "returns": {
                "20": {"instrument": 0.011, "benchmark": 0.5, "excess": 0.9},
                "60": {"instrument": -0.07, "benchmark": 0.4, "excess": 0.8}}},
            {"symbol": "VNINDEX", "kind": "index", "status": vnindex_status, "returns": {
                "20": {"instrument": 0.011, "benchmark": -0.02, "excess": 0.031},
                "60": {"instrument": -0.07, "benchmark": -0.03, "excess": -0.04}}}]},
        "noteBinding": {"snapshotId": "snap-secret-id", "snapshotSha256": "b" * 64, "requestedAsOf": "1",
                        "signalBarOpenTimeMs": last["openTimeMs"]},
        "limitations": ["text"],
    }


def _note(signal_day: str, n: int, mode: str, direction: str, available: int, thesis: str) -> dict:
    ms = _ms(signal_day)
    return {"noteId": f"session/MBB/{ms}/{n:08x}", "symbol": "MBB", "mode": mode, "requestedAsOfUtc": "2026-11-02",
            "signalBarOpenTimeMs": ms, "signalBarCloseTimeMs": ms + 1, "horizonSessions": 20,
            "snapshotId": "snap-secret-id", "snapshotSha256": "a" * 64, "snapshotVerified": True,
            "skillVersion": "equity-technical-v3", "methodVersion": "m1", "direction": direction,
            "actionView": "hold", "actionBasis": "own-position",
            "body": {"thesis": thesis, "evidence": [{"claim": "c", "evidenceIds": ["e1"]}],
                     "scenarios": [{"name": "s", "description": "d", "invalidation": "i", "horizonSessions": 20}],
                     "limitations": ["l"], "missingData": ["m"], "probability": 0.7,
                     "outcomeCheck": {"how": "h", "dueAfterSessions": 20}, "schemaVersion": "equity-research-note-v1",
                     "counterEvidence": [{"claim": "x", "evidenceIds": ["e2"]}]},
            "availableTimeMs": ms + available, "createdAtUtc": "2026-11-02T10:00:00Z", "forecastIds": ["fc-1"]}


def _score(note: dict, status: str, pending=None, ret=None, hit=None, target_day=None) -> dict:
    return {"noteId": note["noteId"], "mode": note["mode"], "signalBarOpenTimeMs": note["signalBarOpenTimeMs"],
            "horizonSessions": 20, "direction": note["direction"], "availableTimeMs": note["availableTimeMs"],
            "status": status, "pendingReason": pending, "signalClose": 18750,
            "targetBarOpenTimeMs": None if target_day is None else _ms(target_day), "targetClose": 19000,
            "returnPct": ret, "realized": "up", "baselineCall": "down", "hit": hit}


def _mbb_notes() -> tuple[list[dict], list[dict], dict]:
    """81 notes on 80 signal bars, newest first in `days`; 3 retrospective, one bar with an earlier slot holder and a
    later superseded note. Returns notes, direction-score items and the hand-written expectation per signal day."""
    days = list(reversed(_sessions(80, date(2026, 6, 1))))  # days[0] = newest signal bar
    notes, scores = [], []
    for i, d in enumerate(days):
        if i in (1, 4, 7):
            n = _note(d, i, "retrospective", "up", 5_000, f"retro {i}")
            notes.append(n)
            scores.append(_score(n, "retrospective-not-scored"))
        elif i == 0:
            n = _note(d, i, "forward", "up", 5_000, "x" * 2000)
            notes.append(n)
            scores.append(_score(n, "resolved", ret=0.061, hit=True, target_day="2026-11-30"))
        elif i == 2:
            n = _note(d, i, "forward", "down", 5_000, "thesis 2")
            notes.append(n)
            scores.append(_score(n, "resolved", ret=0.012, hit=False, target_day="2026-11-27"))
        elif i == 3:
            early = _note(d, 300, "forward", "sideways", 1_000, "early")
            late = _note(d, 301, "forward", "down", 9_000, "late")
            notes += [early, late]
            scores += [_score(early, "pending", pending="immature", target_day="2026-11-26"),
                       _score(late, "superseded-not-scored")]
        elif i == 5:
            n = _note(d, i, "forward", "no-view", 5_000, "thesis 5")
            notes.append(n)
            scores.append(_score(n, "no-view", pending="immature", target_day="2026-11-24"))
        elif i == 6:
            n = _note(d, i, "forward", "sideways", 5_000, "thesis 6")
            notes.append(n)
            scores.append(_score(n, "pending", pending="calendar-coverage-expired"))
        else:
            n = _note(d, i, "forward", "sideways", 5_000, f"thesis {i}")
            notes.append(n)
            scores.append(_score(n, "pending", pending="immature", target_day="2026-12-31"))
    return notes, scores, {"days": days}


def _http(code: int, api_code: str | None = None) -> urllib.error.HTTPError:
    body = json.dumps({"code": api_code, "message": "m", "retryable": False, "requestId": "r1"}).encode()
    return urllib.error.HTTPError("http://fake.invalid/x", code, "error", {}, io.BytesIO(body))


def _query(path: str) -> dict[str, str]:
    q = path.split("?", 1)[1] if "?" in path else ""
    return dict(kv.split("=", 1) for kv in q.split("&") if kv)


class Clock:
    def __init__(self):
        self.t, self.sleeps = 1000.0, []

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


class FakeBackend:
    """GET-only fake of the endpoints the exporter reads. Honours candles `limit` (latest N bars) and the notes/scores
    `take` clamps and orderings like the backend; AdminGuard (401) runs before the capability check (400)."""

    def __init__(self, bars, technical, notes, scores, unsupported=frozenset(), errors=None, clock=None):
        self.bars, self.technical, self.notes, self.scores = bars, technical, notes, scores
        self.unsupported, self.errors, self.clock, self.calls = set(unsupported), errors or {}, clock, []

    def __call__(self, path: str, headers: dict):
        self.calls.append((path, dict(headers or {}), self.clock() if self.clock else None))  # record even bad calls
        if path == "/api/equities":
            return {"contractVersion": "equity-instruments-v1",
                    "items": [{"symbol": s, "label": f"{s} (Name {s})", "venue": "HOSE"} for s in self.bars]}
        parts = path.split("?")[0].split("/")
        symbol, kind = parts[3], "/".join(parts[4:])
        queue = self.errors.get((symbol, kind))
        if queue:
            raise _http(queue.pop(0))
        if symbol not in self.bars:
            raise _http(404, "UNKNOWN_SYMBOL")
        if kind == "candles":
            return self.bars[symbol][-int(_query(path)["limit"]):]
        if kind.startswith("research-notes") and headers.get("X-Admin-Key") != KEY:
            raise _http(401)
        if symbol in self.unsupported:
            raise _http(400, "UNSUPPORTED_CAPABILITY")
        if kind == "technical":
            return self.technical[symbol]
        if kind == "research-notes":
            take = max(1, min(int(_query(path).get("take", 50)), 200))
            items = sorted(self.notes.get(symbol, []),
                           key=lambda n: (-n["signalBarOpenTimeMs"], -n["availableTimeMs"], n["noteId"]))
            return {"contractVersion": "equity-research-notes-v1", "symbol": symbol, "items": items[:take]}
        if kind == "research-notes/direction-scores":
            take = max(1, min(int(_query(path).get("take", 200)), 500))
            items = sorted(self.scores.get(symbol, []), key=lambda i: (-i["signalBarOpenTimeMs"], i["availableTimeMs"]))
            return {"contractVersion": "equity-direction-scores-v1", "symbol": symbol, "rule": "r", "items": items[:take]}
        raise AssertionError(path)


def _market(clock=None, errors=None) -> FakeBackend:
    bars = {s: _bars(300, 20000 + 5000 * k, zero=(10, 250, 299)) for k, s in enumerate(SYMBOLS)}
    technical = {"MBB": _technical(bars["MBB"]), "VCB": _technical(bars["VCB"]),
                 "VIC": _technical(bars["VIC"], vnindex_status="stale")}
    notes, scores, _ = _mbb_notes()
    return FakeBackend(bars, technical, {"MBB": notes}, {"MBB": scores}, unsupported={"ACB"}, errors=errors,
                       clock=clock)


def _screen(session: str = SESSION, schema: str = "vn30-screen/v1", symbols=SYMBOLS) -> dict:
    rows = [{"symbol": s, "close": 1000.0 * n, "returns": {"3": 0.01 * n, "6": 0.02 * n, "12": 0.03 * n},
             "excessVsIndex": {"3": -0.01 * n, "6": -0.02 * n, "12": -0.03 * n},
             "ranks": {"3": 10 + n, "6": n, "12": 20 + n}, "topByRankMonths": n <= 2} for n, s in enumerate(symbols, 1)]
    rows[-1]["ranks"]["6"] = None
    return {"schema": schema, "generatedAtUtc": "2026-11-02T09:20:00+00:00", "session": session,
            "asOfRequested": None, "caveat": "Quan sát", "basket": {"effectiveDate": "2026-08-03", "possiblyStale": False},
            "anchorSessions": {"3": "2026-08-01"}, "indexReturns": {"3": -0.014, "6": -0.088, "12": 0.035},
            "ranking": {"lookbackMonths": 6, "topK": 10}, "missingSessionBar": ["VIC"], "unknownToBackend": [],
            "notUpToDateNow": ["ACB"], "rows": rows}


@pytest.fixture
def screen_dir(tmp_path: Path) -> Path:
    d = tmp_path / "screen"
    d.mkdir()
    (d / f"{SESSION}.json").write_text(json.dumps(_screen()), encoding="utf-8")
    (d / "2026-10-30.json").write_text(json.dumps(_screen("2026-10-30", symbols=("BID", "FPT"))), encoding="utf-8")
    (d / "zz-notes.json").write_text("not a screen", encoding="utf-8")  # not <session>.json: ignored
    return d


def _run(fake, clock, screen_dir: Path, out: Path, monkeypatch, key: str | None = KEY) -> int:
    if key is None:
        monkeypatch.delenv("EQX_ADMIN_KEY", raising=False)
    else:
        monkeypatch.setenv("EQX_ADMIN_KEY", key)
    return web.main(["--out-dir", str(out), "--screen-dir", str(screen_dir)], fetch=fake, sleep=clock.sleep,
                    clock=clock)


def _load(out: Path) -> dict[str, dict]:
    return {p.relative_to(out).as_posix(): json.loads(p.read_text(encoding="utf-8")) for p in out.rglob("*.json")}


def _keys(x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield k
            yield from _keys(v)
    elif isinstance(x, list):
        for v in x:
            yield from _keys(v)


def _symbol(out: Path, s: str) -> dict:
    return json.loads((out / "symbols" / f"{s}.json").read_text(encoding="utf-8"))


def test_every_output_file_holds_only_contract_fields(screen_dir, tmp_path, monkeypatch):
    out = tmp_path / "data"
    assert _run(_market(Clock()), Clock(), screen_dir, out, monkeypatch) == 0
    files = _load(out)
    assert sorted(files) == ["meta.json", "screen.json"] + [f"symbols/{s}.json" for s in sorted(SYMBOLS)]
    for name, doc in files.items():
        leaked = FORBIDDEN & set(_keys(doc))
        assert not leaked, f"{name} leaks {leaked}"
        text = (out / name).read_text(encoding="utf-8")
        for secret in (KEY, "snap-secret-id", "session/MBB/", "fake.invalid", "http", "own-position", "a" * 64):
            assert secret not in text, f"{name} contains {secret!r}"
    assert list(files["meta.json"]) == META_KEYS
    assert all(list(u) == ["symbol", "part", "reason"] for u in files["meta.json"]["unavailable"])
    assert list(files["screen.json"]) == SCREEN_KEYS
    assert all(list(r) == ROW_KEYS for r in files["screen.json"]["rows"])
    for s in SYMBOLS:
        doc = files[f"symbols/{s}.json"]
        assert list(doc) == SYMBOL_KEYS and list(doc["score"]) == ["scored", "hits"]
        assert all(list(n) == NOTE_KEYS for n in doc["notes"])
        assert doc["indicators"] is None or list(doc["indicators"]) == INDICATOR_KEYS
        assert doc["relativeStrength"] is None or list(doc["relativeStrength"]) == ["20", "60"]
    assert len(files["symbols/MBB.json"]["notes"]) == 56  # the fake really served notes with forbidden fields


def test_candles_are_the_latest_260_bars_without_zero_volume(screen_dir, tmp_path, monkeypatch):
    fake = _market(Clock())
    out = tmp_path / "data"
    assert _run(fake, Clock(), screen_dir, out, monkeypatch) == 0
    bars = fake.bars["MBB"]
    expected = [[web.session_of(b["openTimeMs"]), b["open"], b["high"], b["low"], b["close"], b["volume"]]
                for b in bars[-260:] if b["volume"] > 0]
    got = _symbol(out, "MBB")["candles"]
    assert got == expected and len(got) == 258  # bars 250 and 299 had zero volume; bar 10 was outside the window
    assert got[-1][0] == web.session_of(bars[-2]["openTimeMs"])
    assert all(p.endswith("candles?timeframe=1d&limit=260") for p, _, _ in fake.calls if "/candles" in p)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert all(u["part"] != "candles" for u in meta["unavailable"])


def test_indicators_are_the_1d_row_and_strength_is_the_vnindex_excess(screen_dir, tmp_path, monkeypatch):
    fake = _market(Clock())
    out = tmp_path / "data"
    assert _run(fake, Clock(), screen_dir, out, monkeypatch) == 0
    mbb, last = _symbol(out, "MBB"), fake.bars["MBB"][-1]
    assert mbb["indicators"] == {"date": web.session_of(last["openTimeMs"]), "close": last["close"],
                                 **{f: 100.0 + k + 0.5 for k, f in enumerate(web.INDICATORS)}}
    assert mbb["relativeStrength"] == {"20": 0.031, "60": -0.04}
    assert mbb["label"] == "MBB (Name MBB)" and mbb["isBank"] is True and _symbol(out, "VIC")["isBank"] is False
    vic = _symbol(out, "VIC")  # VNINDEX stale: its old window is not the latest session
    assert vic["relativeStrength"] is None and vic["indicators"] is not None
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert [u["part"] for u in meta["unavailable"] if u["symbol"] == "VIC"] == ["technical"]
    assert "VNINDEX" in next(u["reason"] for u in meta["unavailable"] if u["symbol"] == "VIC")


def test_unsupported_capability_is_not_available_yet_not_a_failure(screen_dir, tmp_path, monkeypatch):
    out = tmp_path / "data"
    fake = _market(Clock(), errors={("VCB", "technical"): [500]})
    assert _run(fake, Clock(), screen_dir, out, monkeypatch) == 0
    acb = _symbol(out, "ACB")
    assert acb["indicators"] is None and acb["relativeStrength"] is None
    assert acb["notes"] == [] and acb["score"] == {"scored": 0, "hits": 0}
    assert len(acb["candles"]) == 258  # candles need only ingestion
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    acb_parts = {u["part"]: u["reason"] for u in meta["unavailable"] if u["symbol"] == "ACB"}
    assert set(acb_parts) == {"technical", "notes"} and all("UNSUPPORTED_CAPABILITY" in r for r in acb_parts.values())
    vcb = _symbol(out, "VCB")
    assert vcb["indicators"] is None and vcb["notes"] == [] and vcb["score"] == {"scored": 0, "hits": 0}
    assert {"symbol": "VCB", "part": "technical", "reason": "HTTP 500"} in meta["unavailable"]


def test_technical_calls_are_spaced_and_a_429_waits_a_minute_and_retries_once(screen_dir, tmp_path, monkeypatch):
    clock = Clock()
    fake = _market(clock, errors={("MBB", "technical"): [429], ("VCB", "technical"): [429, 429],
                                  ("VIC", "candles"): [429]})
    out = tmp_path / "data"
    assert _run(fake, clock, screen_dir, out, monkeypatch) == 0
    assert clock.sleeps.count(60.0) == 3  # MBB once, VCB once (then gives up), VIC candles once
    times = [t for p, _, t in fake.calls if p.endswith("/technical?bars=1")]
    assert len(times) == 6  # ACB, MBB x2, VCB x2, VIC
    assert all(b - a >= 7.0 for a, b in zip(times, times[1:]))
    assert all(sum(1 for u in times if t <= u < t + 60) <= 10 for t in times)
    assert _symbol(out, "MBB")["indicators"] is not None and len(_symbol(out, "VIC")["candles"]) == 258
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert {"symbol": "VCB", "part": "technical", "reason": "HTTP 429"} in meta["unavailable"]


def test_without_the_admin_key_notes_are_unavailable_and_never_requested(screen_dir, tmp_path, monkeypatch):
    fake = _market(Clock())
    out = tmp_path / "data"
    assert _run(fake, Clock(), screen_dir, out, monkeypatch, key=None) == 0
    assert not any("research-notes" in p for p, _, _ in fake.calls)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert sorted(u["symbol"] for u in meta["unavailable"] if u["part"] == "notes") == sorted(SYMBOLS)
    mbb = _symbol(out, "MBB")
    assert mbb["notes"] == [] and mbb["score"] == {"scored": 0, "hits": 0}

    with_key = _market(Clock())
    assert _run(with_key, Clock(), screen_dir, tmp_path / "keyed", monkeypatch) == 0
    for path, headers, _ in with_key.calls:  # the key goes only to the two admin endpoints
        assert ("X-Admin-Key" in headers) == ("research-notes" in path)
        assert headers.get("X-Admin-Key") in (None, KEY)


def test_notes_are_forward_newest_first_joined_by_note_id_and_scored(screen_dir, tmp_path, monkeypatch):
    out = tmp_path / "data"
    assert _run(_market(Clock()), Clock(), screen_dir, out, monkeypatch) == 0
    mbb = _symbol(out, "MBB")
    notes, days = mbb["notes"], _mbb_notes()[2]["days"]
    # the backend's newest 60 notes (take=60) hold 57 forward ones: days 0..58 plus a superseded second note on day 3,
    # which is dropped (never scored, and it would show as the latest call of that bar)
    expected_days = [d for i, d in enumerate(days[:59]) if i not in (1, 4, 7)]
    assert [n["signalDate"] for n in notes] == expected_days
    assert all(n["status"] != "superseded-not-scored" and n["thesis"] != "late" for n in notes)
    assert notes[0] == {"signalDate": days[0], "horizonSessions": 20, "direction": "up", "thesis": "x" * 1500,
                        "status": "scored", "pendingReason": None, "targetDate": "2026-11-30", "returnPct": 0.061,
                        "hit": True}
    assert notes[1] == {"signalDate": days[2], "horizonSessions": 20, "direction": "down", "thesis": "thesis 2",
                        "status": "scored", "pendingReason": None, "targetDate": "2026-11-27", "returnPct": 0.012,
                        "hit": False}
    early = notes[2]  # the slot holder of day 3 stays
    assert (early["thesis"], early["status"], early["pendingReason"], early["targetDate"]) == \
           ("early", "pending", "immature", "2026-11-26")
    assert (notes[3]["direction"], notes[3]["status"]) == ("no-view", "no-view")
    assert (notes[4]["pendingReason"], notes[4]["targetDate"]) == ("calendar-coverage-expired", None)
    assert mbb["score"] == {"scored": 2, "hits": 1}

    unjoined = _market(Clock())  # a note without its score: the part is unavailable, never a note without a status
    unjoined.scores["MBB"] = [i for i in unjoined.scores["MBB"] if i["status"] != "no-view"]
    assert _run(unjoined, Clock(), screen_dir, tmp_path / "unjoined", monkeypatch) == 0
    assert _symbol(tmp_path / "unjoined", "MBB")["notes"] == []
    meta = json.loads((tmp_path / "unjoined" / "meta.json").read_text(encoding="utf-8"))
    assert {"symbol": "MBB", "part": "notes", "reason": "phản hồi sai định dạng (KeyError)"} in meta["unavailable"]


def test_unchanged_files_are_not_rewritten(screen_dir, tmp_path, monkeypatch):
    out = tmp_path / "data"
    fake = _market(Clock())
    assert _run(fake, Clock(), screen_dir, out, monkeypatch) == 0
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    (out / "meta.json").write_bytes(web._dump(dict(meta, generatedAtUtc="2026-01-01T00:00:00Z")))
    files = sorted(out.rglob("*.json"))
    before = {p: p.read_bytes() for p in files}
    for p in files:
        os.utime(p, ns=(1_000_000_000, 1_000_000_000))
    assert _run(_market(Clock()), Clock(), screen_dir, out, monkeypatch) == 0
    assert all(p.stat().st_mtime_ns == 1_000_000_000 and p.read_bytes() == before[p] for p in files)

    newer = _market(Clock())
    last = newer.bars["MBB"][-1]
    newer.bars["MBB"].append(dict(last, openTimeMs=last["openTimeMs"] + 3 * 86_400_000, volume=5))
    assert _run(newer, Clock(), screen_dir, out, monkeypatch) == 0
    rewritten = {p.relative_to(out).as_posix() for p in files if p.stat().st_mtime_ns != 1_000_000_000}
    assert rewritten == {"symbols/MBB.json", "meta.json"}
    assert json.loads((out / "meta.json").read_text(encoding="utf-8"))["generatedAtUtc"] != "2026-01-01T00:00:00Z"
    assert not list(out.rglob("*.tmp"))
    assert (out / "meta.json").read_bytes().endswith(b"\n") and b"\r\n" not in (out / "meta.json").read_bytes()


def test_a_ticker_whose_candles_fail_keeps_its_previous_file(screen_dir, tmp_path, monkeypatch):
    out = tmp_path / "data"
    assert _run(_market(Clock()), Clock(), screen_dir, out, monkeypatch) == 0
    previous = (out / "symbols" / "VCB.json").read_bytes()
    fake = _market(Clock(), errors={("VCB", "candles"): [500]})
    fake.technical["VCB"]["timeframes"][1]["indicators"]["rsi14"] = 1.0  # would change the file if it were rebuilt
    assert _run(fake, Clock(), screen_dir, out, monkeypatch) == 0
    assert (out / "symbols" / "VCB.json").read_bytes() == previous
    assert not any(p.startswith("/api/equities/VCB/technical") for p, _, _ in fake.calls)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert {"symbol": "VCB", "part": "candles", "reason": "HTTP 500; giữ nguyên file phiên trước"} in meta["unavailable"]

    fresh = tmp_path / "fresh"
    assert _run(_market(Clock(), errors={("VCB", "candles"): [500]}), Clock(), screen_dir, fresh, monkeypatch) == 0
    vcb = _symbol(fresh, "VCB")
    assert vcb["candles"] == [] and vcb["indicators"] is not None
    meta = json.loads((fresh / "meta.json").read_text(encoding="utf-8"))
    assert {"symbol": "VCB", "part": "candles", "reason": "HTTP 500; chưa có file phiên trước"} in meta["unavailable"]


def test_screen_json_follows_the_newest_screen_file(screen_dir, tmp_path, monkeypatch):
    out = tmp_path / "data"
    assert _run(_market(Clock()), Clock(), screen_dir, out, monkeypatch) == 0
    files = _load(out)
    screen, meta = files["screen.json"], files["meta.json"]
    assert screen["session"] == SESSION and meta["session"] == SESSION
    assert [r["symbol"] for r in screen["rows"]] == list(SYMBOLS)  # the screen's order is kept
    assert meta["symbols"] == sorted(SYMBOLS) and meta["schema"] == "vn30-web-data/v1"
    assert screen["rows"][1] == {"symbol": "MBB", "close": 2000.0, "returns": {"3": 0.02, "6": 0.04, "12": 0.06},
                                 "excessVsIndex": {"3": -0.02, "6": -0.04, "12": -0.06}, "rank": 2, "top": True}
    assert screen["rows"][2]["top"] is False and screen["rows"][3]["rank"] is None
    assert (screen["basket"], screen["rankingMonths"], screen["topK"]) == \
           ({"effectiveDate": "2026-08-03", "possiblyStale": False}, 6, 10)
    assert screen["indexReturns"] == {"3": -0.014, "6": -0.088, "12": 0.035}
    assert (screen["missingSessionBar"], screen["notUpToDate"]) == (["VIC"], ["ACB"])


def test_exit_codes(screen_dir, tmp_path, monkeypatch):
    def unreachable(path, headers):
        raise urllib.error.URLError("connection refused")
    assert _run(unreachable, Clock(), screen_dir, tmp_path / "a", monkeypatch) == 1

    def broken_list(path, headers):
        raise _http(500)
    assert _run(broken_list, Clock(), screen_dir, tmp_path / "b", monkeypatch) == 1

    fake = _market(Clock())
    real = fake.__call__

    def dies_midway(path, headers):  # the backend stops after the first tickers: nothing is written
        if path.startswith("/api/equities/VCB/"):
            raise urllib.error.URLError("connection refused")
        return real(path, headers)
    assert _run(dies_midway, Clock(), screen_dir, tmp_path / "c", monkeypatch) == 1

    assert _run(_market(Clock()), Clock(), tmp_path / "missing", tmp_path / "d", monkeypatch) == 2
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "2026-11-03.json").write_text(json.dumps(_screen(schema="vn30-screen/v0")), encoding="utf-8")
    assert _run(_market(Clock()), Clock(), bad, tmp_path / "e", monkeypatch) == 2
    (bad / "2026-11-04.json").write_text(json.dumps(_screen(symbols=("MBB", "../X"))), encoding="utf-8")
    assert _run(_market(Clock()), Clock(), bad, tmp_path / "f", monkeypatch) == 2
    for d in "abcdef":
        assert not (tmp_path / d).exists()


def test_http_fetch_sends_the_headers_it_is_given(monkeypatch):
    seen = {}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(request, timeout):
        seen["url"], seen["key"], seen["timeout"] = request.full_url, request.get_header("X-admin-key"), timeout
        return Response(b'{"items": []}')
    monkeypatch.setattr(web.urllib.request, "urlopen", urlopen)
    assert web.http_fetch("http://127.0.0.1:1/")("/api/x?y=1", {"X-Admin-Key": KEY}) == {"items": []}
    assert seen == {"url": "http://127.0.0.1:1/api/x?y=1", "key": KEY, "timeout": 120.0}
