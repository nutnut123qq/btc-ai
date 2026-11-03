"""VN30-WEB: export the data of the public site (contract vn30-web-data/v2, evidence/vn30-web-20261103/data-contract.md).

Reads the newest VN30-SCREEN table (<screen-dir>/<session>.json) and the backend HTTP API, GET only (--backend or
BACKEND_BASE_URL): /api/equities (labels) and per ticker /candles (1d, 260 bars), /technical?bars=1 (latest 1d
indicators, relative strength) and, with X-Admin-Key from EQX_ADMIN_KEY, research-notes (take NOTES_TAKE, the
backend's cap; every mode and horizon counts against it) and research-notes/direction-scores (take SCORES_TAKE items;
horizons[] always covers the whole journal). Writes meta.json, screen.json and symbols/<TICKER>.json to --out-dir.

Everything written is published. Every output object is built field by field from the contract and no backend
object is copied whole: notes carry actionView/actionBasis/probability/scenarios/snapshot ids, which never leave.

Units: every return is a fraction (0.05 = +5 %). relativeStrength["20"/"60"] is the `excess` of the VNINDEX
benchmark (instrument minus VNINDEX close-to-close return over 20/60 aligned 1d sessions), exported only while that
benchmark's status is "ok". notes are the newest forward notes with horizonSessions HORIZON (at most NOTE_LIMIT; a
superseded note, a later one on a bar already taken, is dropped), joined with their direction scores by note id. The
backend status "resolved" is exported as "scored"; a direction, status or pendingReason outside the contract makes
the notes part malformed. returnPct is direction-score-v1's target close / signal close - 1. score is the H=20 total
over the ticker's whole journal, copied from the direction-scores horizons[] entry with horizonSessions HORIZON
(resolved, hits, baselinePrior.hits, status), so it does not depend on which notes are exported.

thesis is published verbatim, so it passes a fail-closed filter (PRIVATE_TEXT): anything about positions, holdings,
cash, budget, cost basis, T+2, stops or trading actions (English or Vietnamese), URLs, host names, IP addresses,
ports, file paths, e-mail addresses, long hex strings, internal ids or secrets withholds the whole thesis (null). A
kept thesis longer than THESIS_MAX characters is cut to THESIS_MAX - 1 characters followed by CUT.

Parts: candles, technical (indicators + relativeStrength) and notes (notes + score) are read independently. A part
that fails (an HTTP error, a 429 after the retry, a malformed payload or, for notes, a missing admin key) keeps the
previous symbols/<TICKER>.json's values when the previous meta.json has schema SCHEMA and those values validate; a
candles failure keeps the whole previous file and reads nothing else for that ticker. Otherwise the part is written
empty (null, [], score null). A truthful empty answer is never replaced: no 1d indicator row, a VNINDEX benchmark that
is not "ok", or 400 UNSUPPORTED_CAPABILITY (a part the backend does not serve for that ticker). meta.unavailable
lists every part not read fresh; a failure's reason ends with KEPT or NOTHING_KEPT.

/technical shares the backend's "expensive" limit (10 per minute per client, no queue) with the daily analysis run
that writes the notes: its calls start at least TECH_GAP_S apart (at most 4 per minute), and an HTTP 429 on any call
waits RETRY_AFTER_S and retries once. A file is rewritten only when its bytes change, and meta.json keeps its
generatedAtUtc when nothing else changed. Exit 1 when the backend is unreachable (nothing is written), 2 when the
screen file is missing or invalid. The summary line on stdout (ASCII) counts the unavailable parts, the parts kept
from the previous run and the withheld theses.

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
import unicodedata
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from vn30_screen import DEFAULT_BACKEND, session_of
from vn30_screen import DEFAULT_OUT as DEFAULT_SCREEN_DIR

SCHEMA = "vn30-web-data/v2"
SCREEN_SCHEMA = "vn30-screen/v1"
SOURCE = "VCI, giá điều chỉnh, qua backend của dự án"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "web" / "data"
CANDLE_LIMIT = 260
HORIZON = 20  # the site text, the daily task and the scoring copy are all 20 sessions
NOTE_LIMIT = 60
NOTES_TAKE = 200  # EquityResearchNoteService.ListAsync clamps take to 1..200
SCORES_TAKE = 500  # EquityDirectionScoringService.MaxTake: take cuts items only, never horizons[]
THESIS_MAX = 1500
CUT = chr(0x2026)  # U+2026 HORIZONTAL ELLIPSIS, one character
TECH_GAP_S = 15.0  # Program.cs "expensive": fixed window of 10 per minute, QueueLimit 0
RETRY_AFTER_S = 60.0
KEPT = "; giữ dữ liệu lượt trước"
NOTHING_KEPT = "; chưa có dữ liệu lượt trước"
BANKS = frozenset({"MBB", "BID", "VCB", "CTG", "ACB", "HDB", "LPB", "SHB", "SSB", "STB", "TCB", "VIB", "VPB"})
INDICATORS = ("sma50", "sma200", "ema12", "ema26", "ema50", "ema200", "rsi14", "macd", "macdSignal",
              "macdHistogram", "bollingerUpper", "bollingerMiddle", "bollingerLower", "atr14")
LOOKBACK = ("3", "6", "12")
RS_WINDOWS = ("20", "60")
TICKER = re.compile(r"[A-Z0-9]{2,10}")  # also a file name: no path characters
DIRECTIONS = frozenset({"up", "down", "sideways", "no-view"})
SUPERSEDED = "superseded-not-scored"
STATUS_OUT = {"resolved": "scored", "pending": "pending", "no-view": "no-view",
              "out-of-window-not-scored": "out-of-window-not-scored",
              "window-undetermined-not-scored": "window-undetermined-not-scored"}
NOTE_STATUSES = frozenset(STATUS_OUT.values())
PENDING_REASONS = frozenset({"immature", "target-bar-missing", "calendar-coverage-expired", "signal-bar-missing"})
SAMPLE_STATUSES = frozenset({"no-data", "insufficient-sample", "ok"})
SYMBOL_FIELDS = ("symbol", "label", "isBank", "candles", "indicators", "relativeStrength", "notes", "score")
INDICATOR_FIELDS = ("date", "close", *INDICATORS)
NOTE_FIELDS = ("signalDate", "horizonSessions", "direction", "thesis", "status", "pendingReason", "targetDate",
               "returnPct", "hit")
SCORE_FIELDS = ("scored", "hits", "baselineHits", "sampleStatus")

# Technical phrases blanked before the privacy check, so that a range position or "selling pressure" alone does not
# withhold a thesis. Every other "position", "selling" or "sell" still does. Whitespace is a single space by then.
TECHNICAL_PHRASES = re.compile(
    r"\bposition ?[:=]? ?(?:0?\.\d+|\d{1,3}(?:\.\d+)? ?%)(?!\d|[.,]\d)"
    r"|\b(?:range|band|channel) position\b"
    r"|\bposition (?:with)?in (?:the|its|a) (?:[\w%-]{1,30} ){0,3}?(?:range|band|channel|window)\b"
    r"|\b(?:buying|selling) (?:pressure|interest|climax|volume|activity|flows?|waves?|power|momentum|exhaustion)\b"
    r"|\bsell-?offs?\b",
    re.IGNORECASE)
# Fail-closed: one match anywhere withholds the whole thesis. When in doubt a word is listed here, so a normal
# technical sentence can lose its thesis ("support may hold"), never the other way round. Every repetition is
# bounded or delimited, so a 4000-character thesis is scanned in linear time.
PRIVATE_TEXT = re.compile("|".join((
    # positions, holdings, money and settlement
    r"\bpositions?\b", r"\bholding", r"holders?\b", r"\bhold\b", r"\bportfolios?\b", r"\bexposure\b",
    r"\ballocat", r"\b(?:over|under)weight\b", r"\bcash\b", r"\bbudget", r"\bcapital\b", r"\bcosts?\b",
    r"\bmargin (?:loans?|calls?|accounts?|debt|trading|ratio|requirements?)\b", r"\bon margin\b",
    r"\bavg\.? ?price\b", r"\bbreak-?even\b", r"\bp ?[&/] ?l\b", r"\bpnl\b", r"\bunreali[sz]ed\b",
    r"\d[\d,.]{0,20} ?(?:k|m|mn|million)? ?(?:shares?|cp|cổ phiếu)\b", r"\bt ?\+ ?\d", r"\bsellable\b",
    r"\bsettlement\b", r"\bbrokers?\b", r"\busers?\b", r"\byou(?:rs?)?\b",
    # stops and trading actions
    r"\bstop(?:s|-?loss(?:es)?)?\b", r"\btake[- ]?profits?\b", r"\bcut(?:ting)? (?:the |your )?loss",
    r"\bbuys?\b", r"\bbuying\b", r"\bbought\b", r"\bsells?\b", r"\bselling\b", r"\bsold\b", r"\breduce\b",
    r"\badd\b", r"\baccumulate\b", r"\btrim\b", r"\bexits?\b", r"\benter\b", r"\bentr(?:y|ies)\b", r"\bwait\b",
    r"\bgo(?:ing)? (?:long|short)\b", r"\bshorting\b", r"\bshort[- ]?sell", r"\brecommend",
    # the same in Vietnamese: mua/bán/giữ/cắt lỗ, chờ, danh mục, vị thế, tiền mặt, ngân sách, giá vốn...
    r"\b(?:mua|bán|giữ|gom|chờ|nên|lệnh|lãi|lỗ|vốn)\b", r"\bchốt (?:lời|lãi)\b", r"\bgiải ngân\b",
    r"\bt[ỷỉ] trọng\b", r"\bdanh mục\b", r"\bvị thế\b", r"\btiền mặt\b", r"\bngân sách\b",
    r"\bkhuy(?:ên|ến)\b", r"\bhành động\b", r"\bđứng ngoài\b", r"\bk[ýí] quỹ\b",
    # URLs, host names (label.tld), IP addresses, ports
    r"[a-z][a-z0-9+.-]{0,30}://", r"\bwww\.", r"\blocalhost\b", r"\b(?:tailscale|ngrok)\b", r"\bbtc-(?:ai|be|fe)\b",
    r"[a-z0-9-]\.[a-z]{2,24}\b", r"\b\d{1,3}(?:\.\d{1,3}){3}\b", r"::",
    r"\b(?:[0-9a-f]{1,4}:){7}[0-9a-f]{1,4}\b", r"\b[a-z][a-z0-9.-]{0,62}:\d{2,5}\b", r"\bports?\b",
    # file paths
    r"\\", r"\b[a-z]:/", r"(?<![\w)\]~./])[~.]{0,2}/[\w.-]+/",
    r"(?<![\w)\]~./])/(?:users|home|tmp|var|etc|mnt|opt|srv|root|api)\b",
    r"\w\.(?:json|md|py|ps1|psm1|cs|csproj|sln|tsx?|m?js|cjs|txt|log|csv|tsv|xml|html?|exe|dll|db|sqlite|sql|env"
    r"|ya?ml|ini|cfg|conf|toml|bat|cmd|sh|joblib|pkl|parquet|zip|7z|bak|dump|clixml|pem|crt)\b",
    r"\b(?:research-sessions|node_modules|appdata)\b", r"\.ops\b",
    # e-mail addresses, long hex strings, tokens, internal ids and secrets
    r"@", r"\b[0-9a-f]{16,}\b", r"\b0x[0-9a-f]{8,}", r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b",
    r"[a-z0-9+/=_-]{32,}", r"\bsnap[/:_-]", r"\bsnapshot", r"\bsessions?/",
    r"\b(?:note|forecast|request|snapshot)[ _-]?ids?\b", r"\bx-admin", r"\badmin[ _-]?key", r"\bapi[ _-]?keys?\b",
    r"\bbearer\b", r"\bpassw", r"\bsecrets?\b", r"\btokens?\b", r"\bcredentials?\b", r"\beqx_",
)), re.IGNORECASE)

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


def _count(x: Any) -> int:
    if isinstance(x, bool) or not isinstance(x, int) or x < 0:
        raise TypeError(f"not a count: {x!r}")
    return x


def _enum(x: Any, allowed: frozenset[str] | dict[str, str]) -> str:
    if not isinstance(x, str) or x not in allowed:
        raise ValueError(f"not a contract value: {x!r}")
    return x


def _iso(x: Any) -> str:
    if not isinstance(x, str) or date.fromisoformat(x).isoformat() != x:
        raise ValueError(f"not an ISO date: {x!r}")
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


def _reason(e: urllib.error.HTTPError) -> tuple[str, bool]:
    """(Vietnamese reason, whether it is 400 UNSUPPORTED_CAPABILITY: a part the backend does not serve)."""
    try:
        code = json.loads(e.read().decode("utf-8")).get("code")
    except Exception:  # an error body that is not the JSON envelope
        code = None
    if e.code == 400 and code == "UNSUPPORTED_CAPABILITY":
        return "backend chưa hỗ trợ mã này (UNSUPPORTED_CAPABILITY)", True
    return f"HTTP {e.code}" + (f" {code}" if isinstance(code, str) and re.fullmatch(r"[A-Z_]{1,40}", code) else ""), False


def _part(read: Callable[[], Any]) -> tuple[Any, str | None, bool]:
    """(value, None, False), or (None, reason, keep) for an HTTP error or a malformed payload, where keep says the
    previous run's values may stand in (not for UNSUPPORTED_CAPABILITY); an unreachable backend still raises."""
    try:
        return read(), None, False
    except urllib.error.HTTPError as e:
        reason, unsupported = _reason(e)
        return None, reason, not unsupported
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        return None, f"phản hồi sai định dạng ({type(e).__name__})", True


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


def _probe(text: str) -> str:
    """What the privacy filter reads: NFKC (full-width and compatibility forms folded, Vietnamese composed), format
    and control characters and stray combining marks removed (a zero-width space cannot split a word), whitespace
    runs as one space, technical phrases blanked."""
    folded = "".join(c for c in unicodedata.normalize("NFKC", text)
                     if c in "\t\n\r" or unicodedata.category(c) not in ("Cc", "Cf", "Mn", "Me"))
    return TECHNICAL_PHRASES.sub(" ", re.sub(r"\s+", " ", folded))


def public_thesis(raw: Any) -> tuple[str | None, bool]:
    """(the thesis to publish or None, whether the privacy filter withheld it). The filter reads the whole text."""
    if not isinstance(raw, str) or not raw.strip():
        return None, False
    if PRIVATE_TEXT.search(_probe(raw)):
        return None, True
    return (raw if len(raw) <= THESIS_MAX else raw[:THESIS_MAX - 1] + CUT), False


def score_of(horizons: list) -> dict:
    """The contract's score: the direction-scores horizons[] entry with horizonSessions HORIZON (whole journal)."""
    entries = [h for h in horizons if h["horizonSessions"] == HORIZON]
    if len(entries) != 1:
        raise ValueError(f"{len(entries)} horizons[] entries with horizonSessions {HORIZON}")
    h = entries[0]
    score = {"scored": _count(h["resolved"]), "hits": _count(h["hits"]),
             "baselineHits": _count(h["baselinePrior"]["hits"]), "sampleStatus": _enum(h["status"], SAMPLE_STATUSES)}
    if max(score["hits"], score["baselineHits"]) > score["scored"]:
        raise ValueError("more hits than scored notes")
    return score


def note_rows(notes: dict, scores: dict) -> tuple[list[dict], dict, int]:
    """(newest forward H=20 notes, whole-journal H=20 score, theses withheld); a malformed payload raises."""
    score = score_of(scores["horizons"])
    by_id = {i["noteId"]: i for i in scores["items"]}
    forward = sorted((n for n in notes["items"] if n["mode"] == "forward" and n["horizonSessions"] == HORIZON),
                     key=lambda n: (n["signalBarOpenTimeMs"], n["availableTimeMs"]), reverse=True)
    out, withheld = [], 0
    for n in forward:
        if len(out) == NOTE_LIMIT:
            break
        s = by_id[n["noteId"]]  # a note without its score fails the whole part: the site needs a status
        if (s["signalBarOpenTimeMs"], s["horizonSessions"], s["direction"]) != \
                (n["signalBarOpenTimeMs"], n["horizonSessions"], n["direction"]):
            raise ValueError("a direction score does not match its note")
        if s["status"] == SUPERSEDED:  # a later note on a bar already taken: never scored, would pose as the latest call
            continue
        status = STATUS_OUT[_enum(s["status"], STATUS_OUT)]
        hit = s["hit"] if status == "scored" else None
        if status == "scored" and not isinstance(hit, bool):
            raise TypeError("a scored note without a boolean hit")
        target, reason = s["targetBarOpenTimeMs"], s["pendingReason"]
        body = n.get("body")
        thesis, hidden = public_thesis(body.get("thesis") if isinstance(body, dict) else None)
        withheld += hidden
        out.append({"signalDate": session_of(_num(n["signalBarOpenTimeMs"])), "horizonSessions": HORIZON,
                    "direction": _enum(n["direction"], DIRECTIONS), "thesis": thesis, "status": status,
                    "pendingReason": None if reason is None else _enum(reason, PENDING_REASONS),
                    "targetDate": None if target is None else session_of(_num(target)),
                    "returnPct": _num(s["returnPct"]), "hit": hit})
    return out, score, withheld


# --- the previous run's parts: our own earlier output, validated again (and theses filtered again) before reuse ---

def _fields(x: Any, names: tuple[str, ...]) -> dict:
    if not isinstance(x, dict) or set(x) != set(names):
        raise ValueError("fields differ from the contract")
    return x


def _old_indicators(x: Any) -> dict | None:
    if x is None:
        return None
    x = _fields(x, INDICATOR_FIELDS)
    return {"date": _iso(x["date"]), **{f: _num(x[f]) for f in INDICATOR_FIELDS[1:]}}


def _old_strength(x: Any) -> dict | None:
    if x is None:
        return None
    x = _fields(x, RS_WINDOWS)
    return {w: _num(x[w]) for w in RS_WINDOWS}


def _old_score(x: Any) -> dict | None:
    if x is None:
        return None
    x = _fields(x, SCORE_FIELDS)
    score = {"scored": _count(x["scored"]), "hits": _count(x["hits"]), "baselineHits": _count(x["baselineHits"]),
             "sampleStatus": _enum(x["sampleStatus"], SAMPLE_STATUSES)}
    if max(score["hits"], score["baselineHits"]) > score["scored"]:
        raise ValueError("more hits than scored notes")
    return score


def _old_notes(x: Any) -> tuple[list[dict], int]:
    if not isinstance(x, list) or len(x) > NOTE_LIMIT:
        raise ValueError("notes")
    out, withheld = [], 0
    for n in x:
        n = _fields(n, NOTE_FIELDS)
        status = _enum(n["status"], NOTE_STATUSES)
        hit = n["hit"]
        if (status == "scored") != isinstance(hit, bool) or (status != "scored" and hit is not None):
            raise ValueError("hit")
        if n["horizonSessions"] != HORIZON:
            raise ValueError("horizon")
        thesis, hidden = public_thesis(n["thesis"])
        withheld += hidden
        out.append({"signalDate": _iso(n["signalDate"]), "horizonSessions": HORIZON,
                    "direction": _enum(n["direction"], DIRECTIONS), "thesis": thesis, "status": status,
                    "pendingReason": None if n["pendingReason"] is None else _enum(n["pendingReason"], PENDING_REASONS),
                    "targetDate": None if n["targetDate"] is None else _iso(n["targetDate"]),
                    "returnPct": _num(n["returnPct"]), "hit": hit})
    if any(a["signalDate"] < b["signalDate"] for a, b in zip(out, out[1:])):
        raise ValueError("notes not newest first")
    return out, withheld


def _old_candles(x: Any) -> list[list]:
    if not isinstance(x, list) or len(x) > CANDLE_LIMIT:
        raise ValueError("candles")
    rows = []
    for r in x:
        if not isinstance(r, list) or len(r) != 6 or any(v is None for v in r[1:]) or _num(r[5]) <= 0:
            raise ValueError("candle row")
        rows.append([_iso(r[0]), *(_num(v) for v in r[1:])])
    if any(a[0] >= b[0] for a, b in zip(rows, rows[1:])):
        raise ValueError("candles not ascending")
    return rows


class PreviousRun:
    """The previous run's files in out_dir; usable only when its meta.json has schema SCHEMA. Each kept_* method
    returns None when there is nothing to keep: no usable file, values that do not validate, or a part that was
    itself empty (not read) in the previous run."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        try:
            meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = None
        self.usable = isinstance(meta, dict) and meta.get("schema") == SCHEMA

    def doc(self, symbol: str) -> dict | None:
        if not self.usable:
            return None
        try:
            doc = json.loads((self.out_dir / "symbols" / f"{symbol}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return doc if isinstance(doc, dict) and doc.get("symbol") == symbol else None

    def kept_technical(self, symbol: str) -> tuple[dict | None, dict | None] | None:
        try:
            doc = self.doc(symbol)
            kept = None if doc is None else (_old_indicators(doc["indicators"]), _old_strength(doc["relativeStrength"]))
        except (KeyError, TypeError, ValueError, AttributeError):
            return None
        return None if kept is None or kept == (None, None) else kept

    def kept_notes(self, symbol: str) -> tuple[list[dict], dict, int] | None:
        try:
            doc = self.doc(symbol)
            score = None if doc is None else _old_score(doc["score"])
            if score is None:  # no file, or the previous run had no notes part either
                return None
            notes, withheld = _old_notes(doc["notes"])
        except (KeyError, TypeError, ValueError, AttributeError):
            return None
        return notes, score, withheld

    def kept_whole(self, symbol: str) -> tuple[dict, int] | None:
        try:
            doc = self.doc(symbol)
            if doc is None:
                return None
            doc = _fields(doc, SYMBOL_FIELDS)
            candles = _old_candles(doc["candles"])
            if not candles:  # the previous run had no candles either: read the other parts fresh
                return None
            if not isinstance(doc["label"], str) or not isinstance(doc["isBank"], bool):
                raise TypeError("label/isBank")
            score = _old_score(doc["score"])
            notes, withheld = _old_notes(doc["notes"])
            if score is None and notes:
                raise ValueError("notes without a score")
            kept = {"symbol": symbol, "label": doc["label"], "isBank": doc["isBank"], "candles": candles,
                    "indicators": _old_indicators(doc["indicators"]),
                    "relativeStrength": _old_strength(doc["relativeStrength"]), "notes": notes, "score": score}
        except (KeyError, TypeError, ValueError, AttributeError):
            return None
        return kept, withheld


def build(fetch: Fetch, screen: dict, admin_key: str | None, out_dir: Path,
          sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic
          ) -> tuple[dict[str, dict], list[dict], int]:
    """(symbol -> document, meta.unavailable, theses withheld). A document kept whole is in the result too."""
    try:
        labels = {i["symbol"]: i["label"] for i in _get(fetch, "/api/equities", {}, sleep)["items"]}
    except (KeyError, TypeError, AttributeError) as e:
        raise BackendError(f"/api/equities: {type(e).__name__}: {e}") from e
    previous = PreviousRun(out_dir)
    pace = Pacer(TECH_GAP_S, sleep, clock)
    admin = {"X-Admin-Key": admin_key} if admin_key else None
    docs, unavailable, withheld = {}, [], 0
    for s in sorted(r["symbol"] for r in screen["rows"]):
        def miss(part: str, reason: str, keep: bool = False, kept: bool = False) -> None:
            suffix = "" if not keep else KEPT if kept else NOTHING_KEPT  # never kept: UNSUPPORTED_CAPABILITY
            unavailable.append({"symbol": s, "part": part, "reason": reason + suffix})

        candles, why, keep = _part(lambda: candle_rows(
            _get(fetch, f"/api/equities/{s}/candles?timeframe=1d&limit={CANDLE_LIMIT}", {}, sleep)))
        if why is not None:
            whole = previous.kept_whole(s) if keep else None
            miss("candles", why, keep, whole is not None)
            if whole is not None:
                docs[s] = whole[0]
                withheld += whole[1]
                continue
            candles = []

        technical, why, keep = _part(lambda: technical_parts(
            _get(fetch, f"/api/equities/{s}/technical?bars=1", {}, sleep, pace)))
        if why is None:
            indicators, strength = technical
            if indicators is None:
                miss("technical", "backend chưa có chỉ báo 1d")
            if strength is None:
                miss("technical", "chưa có sức mạnh tương đối so VNINDEX ở phiên mới nhất")
        else:
            kept = previous.kept_technical(s) if keep else None
            miss("technical", why, keep, kept is not None)
            indicators, strength = kept or (None, None)

        if admin is None:
            part, why, keep = None, "thiếu quyền đọc nhật ký nhận định", True
        else:
            part, why, keep = _part(lambda: note_rows(
                _get(fetch, f"/api/equities/{s}/research-notes?take={NOTES_TAKE}", admin, sleep),
                _get(fetch, f"/api/equities/{s}/research-notes/direction-scores?take={SCORES_TAKE}", admin, sleep)))
        if why is not None:
            kept = previous.kept_notes(s) if keep else None
            miss("notes", why, keep, kept is not None)
            part = kept or ([], None, 0)
        notes, score, hidden = part
        withheld += hidden

        label = labels.get(s)
        docs[s] = {"symbol": s, "label": label if isinstance(label, str) else s, "isBank": s in BANKS,
                   "candles": candles, "indicators": indicators, "relativeStrength": strength, "notes": notes,
                   "score": score}
    return docs, unavailable, withheld


def _dump(doc: dict) -> bytes:
    return (json.dumps(doc, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")


def write_if_changed(path: Path, data: bytes) -> bool:
    if path.exists() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)  # readers never see a half-written file
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return True


def write_outputs(out_dir: Path, screen: dict, docs: dict[str, dict], unavailable: list[dict]) -> list[str]:
    for stale in [*out_dir.glob("*.json.tmp"), *(out_dir / "symbols").glob("*.json.tmp")]:
        stale.unlink()  # left by an interrupted run: never published
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
        docs, unavailable, withheld = build(fetch or http_fetch(args.backend), screen,
                                            os.getenv("EQX_ADMIN_KEY") or None, args.out_dir, sleep, clock)
    except (urllib.error.URLError, http.client.HTTPException, OSError, BackendError, ValueError) as e:
        print(f"backend error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    changed = write_outputs(args.out_dir, screen, docs, unavailable)
    parts: dict[str, int] = {}
    kept: dict[str, int] = {}
    for u in unavailable:
        parts[u["part"]] = parts.get(u["part"], 0) + 1
        if u["reason"].endswith(KEPT):
            kept[u["part"]] = kept.get(u["part"], 0) + 1
    print(json.dumps({"session": screen["session"], "symbols": len(screen["rows"]),
                      "written": len(docs) - kept.get("candles", 0), "changed": changed,
                      "unavailableByPart": parts, "keptByPart": kept,
                      "thesesWithheld": withheld}))  # ASCII: a redirected stdout is cp1252
    return 0


if __name__ == "__main__":
    sys.exit(main())
