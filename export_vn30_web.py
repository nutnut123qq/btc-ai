"""VN30-WEB: export the data of the public site (contract vn30-web-data/v1, evidence/vn30-web-20261103/data-contract.md).

Reads the newest VN30-SCREEN table (<screen-dir>/<session>.json) and the backend HTTP API, GET only (--backend or
BACKEND_BASE_URL): /api/equities (labels) and per ticker /candles (1d, 260 bars), /technical?bars=1 (latest 1d
indicators, relative strength) and, with X-Admin-Key from EQX_ADMIN_KEY, research-notes and direction-scores.
Writes meta.json, screen.json and symbols/<TICKER>.json to --out-dir.

Everything written is published. Every output object is built field by field from the contract and no backend
object is copied whole: notes carry actionView/actionBasis/probability/scenarios/snapshot ids, which never leave.

Units: every return is a fraction (0.05 = +5 %). relativeStrength["20"/"60"] is the `excess` of the VNINDEX
benchmark (instrument minus VNINDEX close-to-close return over 20/60 aligned 1d sessions), exported only while that
benchmark's status is "ok". returnPct is direction-score-v1's target close / signal close - 1. The backend status
"resolved" (scored, hit true/false) is exported as "scored"; score counts those notes and their hits among the
exported notes (0/0 while the notes part is unavailable, which meta.unavailable says).

/technical shares the backend's "expensive" limit (10 per minute per client, no queue): its calls start at least
TECH_GAP_S apart, and an HTTP 429 on any call waits RETRY_AFTER_S and retries once. A part the backend does not serve
(400 UNSUPPORTED_CAPABILITY before the VN30-TA deploy) or answers with another HTTP error is exported as null/empty
and listed in meta.unavailable. A ticker whose candles cannot be read keeps its previous file. A file is rewritten
only when its bytes change, and meta.json keeps its generatedAtUtc when nothing else changed.
Exit 1 when the backend is unreachable (nothing is written), 2 when the screen file is missing or invalid.

Usage (from ai/, venv python):
  python export_vn30_web.py [--out-dir DIR] [--screen-dir DIR] [--backend URL]
"""
from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from vn30_screen import DEFAULT_BACKEND, session_of
from vn30_screen import DEFAULT_OUT as DEFAULT_SCREEN_DIR

SCHEMA = "vn30-web-data/v1"
SCREEN_SCHEMA = "vn30-screen/v1"
SOURCE = "VCI, giá điều chỉnh, qua backend của dự án"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "web" / "data"
CANDLE_LIMIT = 260
NOTE_LIMIT = 60
THESIS_MAX = 1500
TECH_GAP_S = 7.0  # Program.cs "expensive": fixed window of 10 per minute, QueueLimit 0
RETRY_AFTER_S = 60.0
BANKS = frozenset({"MBB", "BID", "VCB", "CTG", "ACB", "HDB", "LPB", "SHB", "SSB", "STB", "TCB", "VIB", "VPB"})
INDICATORS = ("sma50", "sma200", "ema12", "ema26", "ema50", "ema200", "rsi14", "macd", "macdSignal",
              "macdHistogram", "bollingerUpper", "bollingerMiddle", "bollingerLower", "atr14")
LOOKBACK = ("3", "6", "12")
RS_WINDOWS = ("20", "60")
TICKER = re.compile(r"[A-Z0-9]{2,10}")  # also a file name: no path characters
TOKEN = re.compile(r"[a-z][a-z-]{0,40}")

Fetch = Callable[[str, dict], Any]


class ScreenError(ValueError):
    pass


class BackendError(RuntimeError):
    """The backend answered /api/equities, but not with the payload shape this script reads."""


def http_fetch(backend: str, timeout: float = 120.0) -> Fetch:
    def fetch(path: str, headers: dict) -> Any:
        request = urllib.request.Request(backend.rstrip("/") + path, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    return fetch


def _num(x: Any) -> int | float | None:
    if x is None:
        return None
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise TypeError(f"not a finite number: {x!r}")
    return x


def _token(x: Any) -> str | None:
    if x is None:
        return None
    if not isinstance(x, str) or not TOKEN.fullmatch(x):
        raise TypeError(f"not a status token: {x!r}")
    return x


def _ticker(x: Any) -> str:
    if not isinstance(x, str) or not TICKER.fullmatch(x):
        raise ScreenError(f"not a ticker: {x!r}")
    return x


def screen_doc(raw: dict) -> dict:
    if raw["schema"] != SCREEN_SCHEMA:
        raise ScreenError(f"schema {raw['schema']!r}")
    months = raw["ranking"]["lookbackMonths"]
    rows = [{"symbol": _ticker(r["symbol"]), "close": _num(r["close"]),
             "returns": {m: _num(r["returns"][m]) for m in LOOKBACK},
             "excessVsIndex": {m: _num(r["excessVsIndex"][m]) for m in LOOKBACK},
             "rank": _num(r["ranks"][str(months)]), "top": r["topByRankMonths"] is True} for r in raw["rows"]]
    if not rows or len({r["symbol"] for r in rows}) != len(rows):
        raise ScreenError("rows empty or a symbol repeats")
    return {
        "session": date.fromisoformat(raw["session"]).isoformat(),
        "basket": {"effectiveDate": date.fromisoformat(raw["basket"]["effectiveDate"]).isoformat(),
                   "possiblyStale": raw["basket"]["possiblyStale"] is True},
        "indexReturns": {m: _num(raw["indexReturns"][m]) for m in LOOKBACK},
        "rankingMonths": _num(months), "topK": _num(raw["ranking"]["topK"]),
        "missingSessionBar": [_ticker(s) for s in raw["missingSessionBar"]],
        "notUpToDate": [_ticker(s) for s in raw["notUpToDateNow"]],
        "rows": rows,
    }


def latest_screen(screen_dir: Path) -> dict:
    files = sorted(p for p in screen_dir.glob("*.json") if re.fullmatch(r"\d{4}-\d{2}-\d{2}\.json", p.name))
    if not files:
        raise ScreenError(f"no <session>.json in {screen_dir}")
    try:
        return screen_doc(json.loads(files[-1].read_text(encoding="utf-8")))
    except (KeyError, TypeError, AttributeError, ValueError) as e:  # ScreenError and JSONDecodeError are ValueErrors
        raise ScreenError(f"{files[-1].name}: {type(e).__name__}: {e}") from e


class Pacer:
    """Starts calls at least `gap` seconds apart."""

    def __init__(self, gap: float, sleep: Callable[[float], None], clock: Callable[[], float]):
        self.gap, self.sleep, self.clock, self.last = gap, sleep, clock, None

    def __call__(self) -> None:
        if self.last is not None and (wait := self.last + self.gap - self.clock()) > 0:
            self.sleep(wait)
        self.last = self.clock()


def _get(fetch: Fetch, path: str, headers: dict, sleep: Callable[[float], None], pace: Pacer | None = None) -> Any:
    for attempt in (1, 2):
        if pace is not None:
            pace()
        try:
            return fetch(path, headers)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 2:
                raise
        sleep(RETRY_AFTER_S)


def _reason(e: urllib.error.HTTPError) -> str:
    try:
        code = json.loads(e.read().decode("utf-8")).get("code")
    except Exception:  # an error body that is not the JSON envelope
        code = None
    if e.code == 400 and code == "UNSUPPORTED_CAPABILITY":
        return "backend chưa hỗ trợ mã này (UNSUPPORTED_CAPABILITY)"
    return f"HTTP {e.code}" + (f" {code}" if isinstance(code, str) and re.fullmatch(r"[A-Z_]{1,40}", code) else "")


def _part(read: Callable[[], Any]) -> tuple[Any, str | None]:
    """(value, None) or (None, reason) for an HTTP error or a malformed payload; unreachable still raises."""
    try:
        return read(), None
    except urllib.error.HTTPError as e:
        return None, _reason(e)
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        return None, f"phản hồi sai định dạng ({type(e).__name__})"


def candle_rows(payload: list) -> list[list]:
    rows = [[session_of(c["openTimeMs"]), _num(c["open"]), _num(c["high"]), _num(c["low"]), _num(c["close"]),
             _num(c["volume"])] for c in payload]
    return [r for r in rows if r[5] > 0][-CANDLE_LIMIT:]


def technical_parts(payload: dict) -> tuple[dict | None, dict | None]:
    """(latest 1d indicators, excess vs VNINDEX over 20/60 sessions); either may be None."""
    day = next((t for t in payload["timeframes"] if t["timeframe"] == "1d"), None)
    indicators = None
    if day is not None and day["indicators"] is not None:
        i = day["indicators"]
        close = next((c["close"] for c in day["candles"] if c["openTimeMs"] == i["openTimeMs"]), None)
        indicators = {"date": session_of(i["openTimeMs"]), "close": _num(close),
                      **{f: _num(i.get(f)) for f in INDICATORS}}
    bench = next((b for b in payload["relativeStrength"]["benchmarks"] if b["symbol"] == "VNINDEX"), None)
    strength = None
    if bench is not None and bench["status"] == "ok":  # "stale" keeps an old window, not the latest session
        strength = {w: _num((bench["returns"].get(w) or {}).get("excess")) for w in RS_WINDOWS}
    return indicators, strength


def note_rows(notes: dict, scores: dict) -> list[dict]:
    by_id = {s["noteId"]: s for s in scores["items"]}
    forward = sorted((n for n in notes["items"] if n["mode"] == "forward"),
                     key=lambda n: (n["signalBarOpenTimeMs"], n["availableTimeMs"]), reverse=True)
    out = []
    for n in forward[:NOTE_LIMIT]:
        s = by_id[n["noteId"]]  # a note without its score fails the whole part: the site needs a status
        body = n.get("body")
        thesis = body.get("thesis") if isinstance(body, dict) else None
        status, target, hit = _token(s["status"]), s.get("targetBarOpenTimeMs"), s.get("hit")
        if status is None:
            raise TypeError("note score without a status")
        if status == "superseded-not-scored":  # a later note on a bar already taken: never scored, would pose as the latest call
            continue
        out.append({"signalDate": session_of(n["signalBarOpenTimeMs"]),
                    "horizonSessions": _num(n["horizonSessions"]), "direction": _token(n["direction"]),
                    "thesis": thesis[:THESIS_MAX] if isinstance(thesis, str) else None,
                    "status": "scored" if status == "resolved" else status,
                    "pendingReason": _token(s.get("pendingReason")),
                    "targetDate": None if target is None else session_of(target),
                    "returnPct": _num(s.get("returnPct")), "hit": hit if isinstance(hit, bool) else None})
    return out


def build(fetch: Fetch, screen: dict, admin_key: str | None, out_dir: Path,
          sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic
          ) -> tuple[dict[str, dict], list[dict]]:
    """Symbol -> document (a symbol whose candles failed and has a previous file is absent: kept), unavailable."""
    try:
        labels = {i["symbol"]: i["label"] for i in _get(fetch, "/api/equities", {}, sleep)["items"]}
    except (KeyError, TypeError, AttributeError) as e:
        raise BackendError(f"/api/equities: {type(e).__name__}: {e}") from e
    pace = Pacer(TECH_GAP_S, sleep, clock)
    admin = {"X-Admin-Key": admin_key} if admin_key else None
    docs, unavailable = {}, []
    for s in sorted(r["symbol"] for r in screen["rows"]):
        def miss(part: str, reason: str) -> None:
            unavailable.append({"symbol": s, "part": part, "reason": reason})

        candles, why = _part(lambda: candle_rows(
            _get(fetch, f"/api/equities/{s}/candles?timeframe=1d&limit={CANDLE_LIMIT}", {}, sleep)))
        if why is not None:
            kept = (out_dir / "symbols" / f"{s}.json").exists()
            miss("candles", why + ("; giữ nguyên file phiên trước" if kept else "; chưa có file phiên trước"))
            if kept:
                continue
            candles = []
        technical, why = _part(lambda: technical_parts(
            _get(fetch, f"/api/equities/{s}/technical?bars=1", {}, sleep, pace)))
        indicators, strength = technical or (None, None)
        if why is not None:
            miss("technical", why)
        else:
            if indicators is None:
                miss("technical", "backend chưa có chỉ báo 1d")
            if strength is None:
                miss("technical", "chưa có sức mạnh tương đối so VNINDEX ở phiên mới nhất")
        notes, why = (None, "thiếu quyền đọc nhật ký nhận định") if admin is None else _part(lambda: note_rows(
            _get(fetch, f"/api/equities/{s}/research-notes?take={NOTE_LIMIT}", admin, sleep),
            _get(fetch, f"/api/equities/{s}/research-notes/direction-scores", admin, sleep)))
        if why is not None:
            miss("notes", why)
            notes = []
        scored = [n for n in notes if n["status"] == "scored"]
        label = labels.get(s)
        docs[s] = {"symbol": s, "label": label if isinstance(label, str) else s, "isBank": s in BANKS,
                   "candles": candles, "indicators": indicators, "relativeStrength": strength, "notes": notes,
                   "score": {"scored": len(scored), "hits": sum(n["hit"] is True for n in scored)}}
    return docs, unavailable


def _dump(doc: dict) -> bytes:
    return (json.dumps(doc, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")


def write_if_changed(path: Path, data: bytes) -> bool:
    if path.exists() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return True


def write_outputs(out_dir: Path, screen: dict, docs: dict[str, dict], unavailable: list[dict]) -> list[str]:
    files = [("screen.json", screen)] + [(f"symbols/{s}.json", d) for s, d in docs.items()]
    changed = [rel for rel, doc in files if write_if_changed(out_dir / rel, _dump(doc))]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = {"schema": SCHEMA, "generatedAtUtc": now, "session": screen["session"],
            "symbols": sorted(r["symbol"] for r in screen["rows"]), "unavailable": unavailable, "source": SOURCE}
    try:
        old = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = None
    if not changed and isinstance(old, dict) and {**old, "generatedAtUtc": now} == meta:
        meta = old  # nothing changed: keep the timestamp, so an idle day makes no commit
    if write_if_changed(out_dir / "meta.json", _dump(meta)):
        changed.append("meta.json")
    return changed


def main(argv: list[str] | None = None, fetch: Fetch | None = None,
         sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--screen-dir", type=Path, default=DEFAULT_SCREEN_DIR)
    ap.add_argument("--backend", default=os.getenv("BACKEND_BASE_URL") or DEFAULT_BACKEND)
    args = ap.parse_args(argv)
    try:
        screen = latest_screen(args.screen_dir)
    except ScreenError as e:
        print(f"screen error: {e}", file=sys.stderr)
        return 2
    try:
        docs, unavailable = build(fetch or http_fetch(args.backend), screen, os.getenv("EQX_ADMIN_KEY") or None,
                                  args.out_dir, sleep, clock)
    except (urllib.error.URLError, http.client.HTTPException, OSError, BackendError, ValueError) as e:
        print(f"backend error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    changed = write_outputs(args.out_dir, screen, docs, unavailable)
    parts: dict[str, int] = {}
    for u in unavailable:
        parts[u["part"]] = parts.get(u["part"], 0) + 1
    print(json.dumps({"session": screen["session"], "symbols": len(screen["rows"]), "written": len(docs),
                      "changed": changed, "unavailableByPart": parts}))  # ASCII: a redirected stdout is cp1252
    return 0


if __name__ == "__main__":
    sys.exit(main())
