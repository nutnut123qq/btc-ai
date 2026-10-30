"""ai/research_snapshot.py against a fake backend (stdlib http.server on 127.0.0.1)."""

from __future__ import annotations

import hashlib
import json
import socket
import sys
import threading
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import research_snapshot as rs

ADMIN_KEY = "test-admin-key-not-real"
DAY = 86_400_000
FIRST_OPEN = 1_600_000_000_000


def _bars(n: int) -> list[dict]:
    out = []
    for i in range(n):
        open_ms = FIRST_OPEN + i * DAY
        out.append({
            "openTimeMs": open_ms, "closeTimeMs": open_ms + 27_900_000,
            "open": f"{40000 + i}.50", "high": f"{40100 + i}.00", "low": f"{39900 + i}", "close": "40000.0000",
            "volume": f"{1000000 + i}.000",
        })
    return out


class FakeBackend:
    """Serves research-snapshot/readiness/candles/position/decision/research-notes like the backend."""

    def __init__(self, bars: list[dict], cutoff_index: int, tamper: bool = False) -> None:
        self.bars = bars
        self.cutoff = bars[cutoff_index]["closeTimeMs"]
        kept = [b for b in bars if b["closeTimeMs"] <= self.cutoff]
        lines = [rs.series_line(json.loads(json.dumps(b), parse_float=Decimal)) for b in kept]
        sha = rs.fingerprint(lines)
        if tamper:
            sha = "0" * 64
        self.snapshot = {
            "contractVersion": "equity-research-snapshot-v1", "snapshotId": f"snap/BID/{self.cutoff}/{sha[:12]}",
            "snapshotSha256": "a" * 64, "symbol": "BID", "requestedAsOf": "2026-10-14T09:00:00Z",
            "requestedAsOfMs": self.cutoff + 1000, "effectiveCutoffMs": self.cutoff,
            "signalBar": {"openTimeMs": kept[-1]["openTimeMs"], "closeTimeMs": kept[-1]["closeTimeMs"]},
            "dataset": {"1d": {"barCount": len(kept), "sha256": sha}},
            "series": [{"timeframe": "1d", "endpoint": "/api/equities/BID/candles?timeframe=1d", "endTimeMs": self.cutoff,
                        "barCount": len(kept), "fingerprint": sha, "fingerprintMethod": rs.FINGERPRINT_METHOD,
                        "pageLimit": 2000}],
        }
        self.requests: list[tuple[str, str, dict, dict, bytes]] = []
        self.notes: list[dict] = []

    def handle(self, method: str, url: str, headers: dict, body: bytes):
        parsed = urlparse(url)
        q = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        self.requests.append((method, parsed.path, q, headers, body))
        admin_ok = headers.get("X-Admin-Key") == ADMIN_KEY
        path = parsed.path
        if path == "/api/equities/BID/research-snapshot":
            return 200, json.dumps(self.snapshot, indent=1).encode()
        if path == "/api/equities/BID/research-readiness":
            return 200, json.dumps({"contractVersion": "equity-research-readiness-v1", "asOf": q.get("asOf"),
                                    "purposes": {"portfolio-action": {"status": "requires-admin"}}}).encode()
        if path == "/api/equities/BID/candles":
            end = int(q["endTimeMs"])
            limit = int(q["limit"])
            rows = [b for b in self.bars if b["openTimeMs"] <= end][-limit:]
            text = "[" + ",".join(
                '{"openTimeMs":%d,"closeTimeMs":%d,"open":%s,"high":%s,"low":%s,"close":%s,"volume":%s,"quoteVolume":null}'
                % (b["openTimeMs"], b["closeTimeMs"], b["open"], b["high"], b["low"], b["close"], b["volume"])
                for b in rows) + "]"
            return 200, text.encode()
        if path in ("/api/equities/BID/position", "/api/equities/BID/decision", "/api/equities/BID/research-notes"):
            if not admin_ok:
                return 401, b'{"Message":"Unauthorized"}'
            if path.endswith("position"):
                return 200, b'{"symbol":"BID","lots":[{"quantity":700}],"availableCashVnd":9876543}'
            if path.endswith("decision"):
                return 200, b'{"symbol":"BID","decision":"hold"}'
            if method == "POST":
                note = json.loads(body)
                if note.get("mode") not in ("forward", "retrospective"):
                    return 400, b'{"code":"INVALID_NOTE","message":"mode must be forward or retrospective."}'
                self.notes.append(note)
                return 201, json.dumps({"noteId": "session/BID/1/abcd1234", "snapshotVerified": True}).encode()
            return 200, json.dumps({"items": self.notes, "take": q.get("take")}).encode()
        return 404, b'{"code":"NOT_FOUND","message":"nope"}'


@pytest.fixture
def backend(request):
    fake: FakeBackend = request.param if hasattr(request, "param") else FakeBackend(_bars(4500), 4300)

    class Handler(BaseHTTPRequestHandler):
        def _serve(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            status, payload = fake.handle(self.command, self.path, dict(self.headers), body)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_GET = _serve
        do_POST = _serve

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    fake.url = f"http://127.0.0.1:{server.server_address[1]}"
    yield fake
    server.shutdown()
    server.server_close()


def _run(backend, argv, monkeypatch, key: str | None = None) -> int:
    if key is None:
        monkeypatch.delenv(rs.ADMIN_KEY_ENV, raising=False)
    else:
        monkeypatch.setenv(rs.ADMIN_KEY_ENV, key)
    return rs.main(["--backend", backend.url, *argv])


# --- canonical line format ------------------------------------------------------------


def test_normalize_and_cross_language_vector():
    assert rs.normalize_number(Decimal("27650.50")) == "27650.5"
    assert rs.normalize_number(40000) == "40000"
    assert rs.normalize_number(Decimal("40000.000")) == "40000"
    assert rs.normalize_number(Decimal("0.00")) == "0"
    assert rs.normalize_number(Decimal("1E+3")) == "1000"
    lines = [
        rs.series_line({"openTimeMs": 1000, "closeTimeMs": 2000, "open": Decimal("27650.50"), "high": Decimal("27700.00"),
                        "low": Decimal("27600.0"), "close": Decimal("27650.5"), "volume": Decimal("1234567.000")}),
        rs.series_line({"openTimeMs": 3000, "closeTimeMs": 4000, "open": 40000, "high": 40100, "low": 39900,
                        "close": 40000, "volume": 0}),
    ]
    # Same vector is pinned in backend Tests/Backend.Tests/EquityResearchSnapshotTests.cs.
    assert rs.fingerprint(lines) == "a3ad6448775a161503acd32ddbc6ec328472ae4107052eb54c2b86790683932f"


# --- export ---------------------------------------------------------------------------


def test_export_pages_verifies_and_writes_manifest(backend, tmp_path, monkeypatch, capsys):
    out = tmp_path / "exp"
    assert _run(backend, ["export", "--symbol", "BID", "--as-of", "2026-10-14T09:00:00Z", "--out-dir", str(out)], monkeypatch) == 0

    candle_calls = [r for r in backend.requests if r[1].endswith("/candles")]
    assert len(candle_calls) == 3  # 4301 bars <= cutoff in pages of 2000
    assert int(candle_calls[0][2]["endTimeMs"]) == backend.cutoff
    snap_call = next(r for r in backend.requests if r[1].endswith("/research-snapshot"))
    assert snap_call[2]["asOf"] == "2026-10-14T09:00:00Z"
    assert "X-Admin-Key" not in snap_call[3]

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schemaVersion"] == "equity-research-export-v1"
    assert manifest["snapshotId"] == backend.snapshot["snapshotId"]
    assert manifest["series"] == [{"timeframe": "1d", "barCount": 4301,
                                   "fingerprint": backend.snapshot["series"][0]["fingerprint"], "verified": True}]
    names = {f["path"] for f in manifest["files"]}
    assert names == {"snapshot.json", "series-1d.csv"}
    for item in manifest["files"]:
        assert hashlib.sha256((out / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    # snapshot.json is exactly what the backend served
    assert json.loads((out / "snapshot.json").read_text(encoding="utf-8")) == backend.snapshot
    csv_lines = (out / "series-1d.csv").read_text(encoding="utf-8").splitlines()
    assert csv_lines[0] == rs.CSV_HEADER and len(csv_lines) == 4302
    assert csv_lines[1] == f"{FIRST_OPEN},{FIRST_OPEN + 27_900_000},40000.5,40100,39900,40000,1000000"
    assert int(csv_lines[-1].split(",")[1]) == backend.cutoff
    assert rs.fingerprint(csv_lines[1:]) == manifest["series"][0]["fingerprint"]
    assert not (out / "portfolio.json").exists()


@pytest.mark.parametrize("backend", [FakeBackend(_bars(300), 250, tamper=True)], indirect=True)
def test_export_fingerprint_mismatch_fails_without_manifest(backend, tmp_path, monkeypatch, capsys):
    out = tmp_path / "exp"
    code = _run(backend, ["export", "--symbol", "BID", "--out-dir", str(out)], monkeypatch)
    assert code == rs.EXIT_FINGERPRINT
    assert "fingerprint mismatch" in capsys.readouterr().err
    assert not (out / "manifest.json").exists()


def test_include_portfolio_requires_key_and_stays_separate(backend, tmp_path, monkeypatch, capsys):
    out = tmp_path / "nokey"
    assert _run(backend, ["export", "--symbol", "BID", "--out-dir", str(out), "--include-portfolio"], monkeypatch) == rs.EXIT_USAGE
    assert "EQX_ADMIN_KEY" in capsys.readouterr().err
    assert not out.exists()

    out = tmp_path / "withkey"
    assert _run(backend, ["export", "--symbol", "BID", "--out-dir", str(out), "--include-portfolio"], monkeypatch,
                key=ADMIN_KEY) == 0
    captured = capsys.readouterr()
    portfolio = json.loads((out / "portfolio.json").read_text(encoding="utf-8"))
    assert portfolio["position"]["availableCashVnd"] == 9876543
    assert portfolio["decision"]["decision"] == "hold"
    snapshot_text = (out / "snapshot.json").read_text(encoding="utf-8")
    assert "9876543" not in snapshot_text and "lots" not in snapshot_text
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert "portfolio.json" in {f["path"] for f in manifest["files"]}
    # key sent only as header to admin routes, never printed or written
    admin_calls = [r for r in backend.requests if r[3].get("X-Admin-Key")]
    assert {r[1] for r in admin_calls} == {"/api/equities/BID/position", "/api/equities/BID/decision"}
    for path in out.iterdir():
        assert ADMIN_KEY not in path.read_text(encoding="utf-8")
    assert ADMIN_KEY not in captured.out + captured.err


# --- readiness / journal ----------------------------------------------------------------


def test_readiness_passes_as_of(backend, monkeypatch, capsys):
    assert _run(backend, ["readiness", "--symbol", "BID", "--as-of", "1790000000000"], monkeypatch) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["asOf"] == "1790000000000"
    assert payload["purposes"]["portfolio-action"]["status"] == "requires-admin"


def test_journal_record_fills_from_snapshot_and_lists(backend, tmp_path, monkeypatch, capsys):
    snap = tmp_path / "snapshot.json"
    snap.write_text(json.dumps(backend.snapshot), encoding="utf-8")
    note = tmp_path / "note.json"
    note.write_text(json.dumps({"mode": "forward", "horizonSessions": 20, "skillVersion": "s", "methodVersion": "m",
                                "direction": "sideways", "actionView": "wait", "actionBasis": "market-only",
                                "body": {"schemaVersion": "equity-research-note-v1",
                                         "probability": {"probTargetFirst": 0.35}}}), encoding="utf-8")

    assert _run(backend, ["journal", "record", "--symbol", "BID", "--file", str(note)], monkeypatch) == rs.EXIT_USAGE
    assert backend.notes == []

    code = _run(backend, ["journal", "record", "--symbol", "BID", "--file", str(note), "--snapshot", str(snap)],
                monkeypatch, key=ADMIN_KEY)
    assert code == 0
    sent = backend.notes[-1]
    assert sent["snapshotId"] == backend.snapshot["snapshotId"]
    assert sent["snapshotSha256"] == backend.snapshot["snapshotSha256"]
    assert sent["requestedAsOf"] == str(backend.snapshot["requestedAsOfMs"])
    assert sent["signalBarOpenTimeMs"] == backend.snapshot["signalBar"]["openTimeMs"]
    assert sent["body"]["probability"]["probTargetFirst"] == 0.35
    assert json.loads(capsys.readouterr().out)["snapshotVerified"] is True

    assert _run(backend, ["journal", "list", "--symbol", "BID", "--take", "5"], monkeypatch, key=ADMIN_KEY) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["take"] == "5" and len(listed["items"]) == 1


def test_api_error_envelope_is_reported(backend, tmp_path, monkeypatch, capsys):
    note = tmp_path / "bad.json"
    note.write_text(json.dumps({"mode": "sideways"}), encoding="utf-8")
    code = _run(backend, ["journal", "record", "--symbol", "BID", "--file", str(note)], monkeypatch, key=ADMIN_KEY)
    assert code == rs.EXIT_HTTP
    err = capsys.readouterr().err
    assert "HTTP 400 INVALID_NOTE" in err and ADMIN_KEY not in err


def test_backend_read_timeout_is_an_http_failure(monkeypatch, capsys):
    monkeypatch.delenv(rs.ADMIN_KEY_ENV, raising=False)
    with socket.socket() as silent:  # accepts the connection (backlog) but never answers
        silent.bind(("127.0.0.1", 0))
        silent.listen(1)
        port = silent.getsockname()[1]
        code = rs.main(["--backend", f"http://127.0.0.1:{port}", "--timeout", "0.5", "readiness", "--symbol", "BID"])
    assert code == rs.EXIT_HTTP
    assert "BACKEND_TIMEOUT" in capsys.readouterr().err


def test_unreachable_backend_fails_cleanly(monkeypatch, capsys):
    monkeypatch.delenv(rs.ADMIN_KEY_ENV, raising=False)
    code = rs.main(["--backend", "http://127.0.0.1:9", "--timeout", "2", "readiness", "--symbol", "BID"])
    assert code == rs.EXIT_HTTP
    assert "BACKEND_UNREACHABLE" in capsys.readouterr().err
