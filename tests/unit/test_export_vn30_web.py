"""VN30-WEB exporter, contract vn30-web-data/v2: only contract fields reach the public files; notes are the newest
forward 20-session notes and score is the backend's whole-journal H=20 total; a value outside the contract enums makes
the notes part malformed; a thesis passes a fail-closed privacy filter and carries a cut marker; a part that fails
keeps the previous v2 run's values (400 UNSUPPORTED_CAPABILITY never does); /technical calls start 15 s apart; a file
is rewritten only when its bytes change."""
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
KEPT = "; giữ dữ liệu lượt trước"
NOTHING_KEPT = "; chưa có dữ liệu lượt trước"
UNSUPPORTED = "backend chưa hỗ trợ mã này (UNSUPPORTED_CAPABILITY)"
NO_KEY = "thiếu quyền đọc nhật ký nhận định"
NO_STRENGTH = "chưa có sức mạnh tương đối so VNINDEX ở phiên mới nhất"
INDICATOR_NAMES = ["sma50", "sma200", "ema12", "ema26", "ema50", "ema200", "rsi14", "macd", "macdSignal",
                   "macdHistogram", "bollingerUpper", "bollingerMiddle", "bollingerLower", "atr14"]
META_KEYS = ["schema", "generatedAtUtc", "session", "symbols", "unavailable", "source"]
SCREEN_KEYS = ["session", "basket", "indexReturns", "rankingMonths", "topK", "missingSessionBar", "notUpToDate", "rows"]
ROW_KEYS = ["symbol", "close", "returns", "excessVsIndex", "rank", "top"]
SYMBOL_KEYS = ["symbol", "label", "isBank", "candles", "indicators", "relativeStrength", "notes", "score"]
INDICATOR_KEYS = ["date", "close", *INDICATOR_NAMES]
NOTE_KEYS = ["signalDate", "horizonSessions", "direction", "thesis", "status", "pendingReason", "targetDate",
             "returnPct", "hit"]
SCORE_KEYS = ["scored", "hits", "baselineHits", "sampleStatus"]
FORBIDDEN = {"actionView", "actionBasis", "probability", "scenarios", "evidence", "counterEvidence", "snapshotId",
             "snapshotSha256", "noteId", "mode", "requestedAsOfUtc", "skillVersion", "methodVersion", "forecastIds",
             "limitations", "missingData", "outcomeCheck", "signalClose", "targetClose", "realized", "baselineCall",
             "methods", "obv", "vwap", "macdNorm", "noteBinding", "timeIso", "caveat", "anchorSessions",
             "unknownToBackend", "ranks", "position", "cash", "body", "horizons", "items", "resolved", "baselinePrior",
             "hitRate", "nEffective", "configId", "bestConstantInHindsight", "pendingReasons", "rule", "totalNotes"}
DIRS = ("up", "down", "sideways")
# Planted in one fixture thesis: the filter must withhold that thesis, so none of these may reach any output file.
THESIS_SECRETS = ("100.83.92.101:5197", "someone@example.com", "D:\\code\\btc\\data\\research-sessions",
                  "120,000,000 VND", "f0e364fd")
PLANTED_THESIS = ("MBB 1d closed below SMA50. User holds 3,000 MBB at avg cost 21,500, cash 120,000,000 VND; see "
                  "100.83.92.101:5197 and D:\\code\\btc\\data\\research-sessions\\MBB, note session/MBB/1792108800000/"
                  "f0e364fd, mail someone@example.com.")
LONG_THESIS = ("The 1d close sits above SMA50 while RSI14 stays near 50 and the 20-session range is intact. " * 25)[:2000]
EXACT_THESIS = ("Bollinger middle band 18,900 caps the 1d close; ATR14 is flat. " * 30)[:1500]
# The three theses published in web/data on 2026-11-03 (copied verbatim from symbols/MBB.json and BID.json).
REAL_THESES = (
    "MBB 1d signal bar 2026-11-02 closed 18750 (-2.85% on the day, volume 2.1x its 20-session average), the low end "
    "of the 100-bar window (20/60-session low 18600 touched intraday). Trend states are bearish (close below SMA50 "
    "19877 and below the 100-bar volume-profile VAL 18852, EMA12 < EMA26, MACD histogram negative), but momentum is "
    "stretched rather than accelerating: Wilder RSI14 34.0, close near the lower Bollinger band 18730, 1h Wilder "
    "RSI14 21.3. Relative strength vs VNINDEX is flat over 20 sessions. Mixed evidence over 20 sessions: view "
    "sideways (R between -4% and +5%), for reference only.",
    "MBB 1d signal 2026-10-30 close 19300 (adjusted). Mixed technical picture: 1d regime is sideways/high-vol, price "
    "has held the 20-session range 18650-20100 (position 0.45) while sitting below SMA50 (-3.0%) and SMA200 (-5.9%); "
    "1h is in a bear/high-vol regime and the signal bar closed at its low (BearishMarubozu). 1d MACD histogram turned "
    "slightly positive while EMA12 < EMA26. Neither the +5% (>= 20265) nor the -4% (<= 18528) H20 barrier is "
    "favoured by the evidence: both lie just outside the 20-session range. View: sideways over 20 sessions, for "
    "reference only.",
    "BID 1d signal bar 2026-11-02 closed 32000 (-4.19% on the day, volume 2.85x its 20-session average), a new low "
    "of the 100-bar window (intraday low 31700), below the lower Bollinger band 32625. Trend is clearly bearish "
    "(20-session -7.5%, 60-session -10.0%, close below SMA50 35177 and SMA200 38502, EMA12 < EMA26, underperforming "
    "VNINDEX by 5.1 points over 20 sessions), but the move is stretched: Wilder RSI14 28.0 on 1d, 1h Wilder RSI14 "
    "25.1 (Cutler 7.0). Continuation and mean reversion both have support; over 20 sessions the view is sideways (R "
    "between -4% and +5%), for reference only.",
)
TECHNICAL_THESES = REAL_THESES + (
    "Buyers stepped in at the 20-session low and the support holds; price held above SMA50.",
    "Selling pressure eased after a sell-off on 2.1x volume; buying pressure returned near the lower band.",
    "RSI14 is oversold on 1d and overbought on 1h; short-term momentum is weak and the long lower shadow shows demand.",
    "The range position is 0.45 (position 0.45, i.e. position within the 20-session range) after a selloff.",
    "Money flow and accumulation/distribution diverge; volume-weighted average price 18,900; +5%/-4% barriers.",
    "Price stopped at the 60-session low and rebounded; sellers were absorbed, e.g. on 2026-11-02 (R(20) inside).",
)
# The strings cold-check round 1 planted in a thesis (coldcheck-r1.md, lens-leak and its triage).
COLDCHECK_PRIVATE = (
    "http://127.0.0.1:5197/api/equities/MBB/technical",
    "10.11.12.13:5197",
    "D:\\code\\btc\\data\\research-sessions\\MBB\\20261102-1726-technical",
    "C:\\Users\\someone\\AppData",
    "snap/MBB/1793346300000/2bb65c32d3d5",
    "session/MBB/1792108800000/f0e364fd",
    "0f" * 32,
    "someone@example.com",
    "User holds 3,000 MBB at avg cost 21,500 (T+2 sellable 1,000); stop -8%; cash 120,000,000 VND, buy budget 50m",
    "View: buy more",
    "Hành động: hold - không bán bớt; người đang giữ giữ trên 19.500",
)
PRIVATE_THESES = COLDCHECK_PRIVATE + (
    "for a holder the market view alone supports hold", "Market-only: for new money wait for a close above 20,100",
    "the stop at the low of 18,600", "your position is 2,000 shares", "portfolio weight 10%",
    "take profit at 21,000", "cost basis 21,500", "T+2 sellable", "reduce on a close below 18,500",
    "add on a retest of 18,600", "Khuyến nghị mua thêm khi vượt 20.100", "cắt lỗ dưới 18.500", "danh mục chưa có",
    "tiền mặt 50 triệu", "ngân sách 100 triệu", "chờ xác nhận", "btc-be.example.com", "see vietstock.vn",
    "localhost:3000", "100.83.92.101", "fe80::1", "(/api/equities/MBB/candles)", "~/code/btc/ai", "D:\\work\\notes",
    "saved under D:/notes", "noteId attached",
    "X-Admin-Key header", "ai/skills/equity-technical/SKILL.md",
    "b\u200buy now",  # a zero-width space inside the word
    "ba\u0301n bớt",  # Vietnamese typed decomposed (NFD)
    "\uff42\uff55\uff59 the dip",  # full-width letters
    "T\uff0b2",  # a full-width plus sign
)


def _sessions(n: int, start: date) -> list[str]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _ms(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000)


def _target(day: str) -> str:
    return (date.fromisoformat(day) + timedelta(days=28)).isoformat()


def _thesis(symbol: str, i: int) -> str:
    return f"{symbol} thesis {i}: the 1d close sits above SMA50 and the 20-session support holds."


def _ret(i: int) -> float:
    return round(0.0005 * i - 0.02, 4)


def _bars(n: int, base: int, zero: tuple[int, ...] = ()) -> list[dict]:
    # open/high/low/close/volume all differ, so a swapped column shows
    return [{"openTimeMs": _ms(d), "timeIso": d + "T00:00:00Z", "open": base + 10 * i - 3, "high": base + 10 * i + 7,
             "low": base + 10 * i - 9, "close": base + 10 * i, "volume": 0 if i in zero else 1000 + i,
             "closeTimeMs": _ms(d) + 27_900_000, "quoteVolume": None} for i, d in enumerate(_sessions(n, date(2025, 8, 1)))]


def _technical(bars: list[dict], vnindex_status: str = "ok") -> dict:
    last = bars[-1]
    day_ind = {f: 100.0 + k + 0.5 for k, f in enumerate(INDICATOR_NAMES)}
    return {
        "contractVersion": "equity-technical-v1", "symbol": "X", "asOfUtc": "2026-11-02T08:00:00Z",
        "timeframes": [
            {"timeframe": "1h", "candles": [dict(last, openTimeMs=last["openTimeMs"] + 3_600_000, close=1)],
             "indicators": {"openTimeMs": last["openTimeMs"] + 3_600_000, **{f: -1.0 for f in INDICATOR_NAMES}}},
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


def _entry(symbol: str, day: str, n: int, *, mode: str = "forward", direction: str = "up", available: int = 5_000,
           thesis: str = "", horizon: int = 20, status: str = "pending", pending: str | None = None,
           ret: float | None = None, hit: bool | None = None, target: str | None = None, realized: str | None = None,
           baseline: str | None = None) -> tuple[dict, dict]:
    """One journal note as GET research-notes serves it, and its GET direction-scores item."""
    ms = _ms(day)
    note = {"noteId": f"session/{symbol}/{ms}/{n:08x}", "symbol": symbol, "mode": mode,
            "requestedAsOfUtc": day + "T08:00:00Z", "signalBarOpenTimeMs": ms, "signalBarCloseTimeMs": ms + 27_900_000,
            "horizonSessions": horizon, "snapshotId": f"snap/{symbol}/{ms}/snap-secret-id", "snapshotSha256": "a" * 64,
            "snapshotVerified": True, "skillVersion": "equity-technical-v6", "methodVersion": "m1",
            "direction": direction, "actionView": "hold", "actionBasis": "own-position",
            "body": {"schemaVersion": "equity-research-note-v1", "thesis": thesis,
                     "evidence": [{"claim": "c", "evidenceIds": ["e1"]}],
                     "counterEvidence": [{"claim": "x", "evidenceIds": ["e2"]}],
                     "scenarios": [{"name": "s", "horizonSessions": horizon, "description": "d", "invalidation": ["i"]}],
                     "limitations": ["l"], "missingData": ["m"],
                     "outcomeCheck": {"how": "h", "dueAfterSessions": horizon}, "probability": 0.7},
            "availableTimeMs": ms + available, "createdAtUtc": day + "T10:00:00Z", "forecastIds": ["fc-1"]}
    item = {"noteId": note["noteId"], "mode": mode, "signalBarOpenTimeMs": ms, "horizonSessions": horizon,
            "direction": direction, "availableTimeMs": note["availableTimeMs"], "status": status,
            "pendingReason": pending, "signalClose": 18750,
            "targetBarOpenTimeMs": None if target is None else _ms(target),
            "targetClose": None if target is None else 19000, "returnPct": ret, "realized": realized,
            "baselineCall": baseline, "hit": hit}
    return note, item


def _mbb_days() -> list[str]:
    return list(reversed(_sessions(100, date(2026, 6, 8))))  # [0] = the newest signal bar


def _mbb_journal() -> list[tuple[dict, dict]]:
    """One forward H=20 note on each of 100 signal bars (days[0] newest) plus a superseded one, 2 H=60 and 2
    retrospective notes. days[0..19]: not resolved yet, with every status and pendingReason of the contract and a
    thesis with private text (days[2]), a long one (days[11]) and one of exactly 1500 characters (days[12]); days[3]
    also has a later note on the taken bar (superseded). days[20..99]: 80 resolved notes, hit unless i % 3 == 0,
    the baseline right when i % 4 == 0."""
    days, pairs = _mbb_days(), []
    for i, d in enumerate(days):
        kw = {"direction": DIRS[i % 3], "thesis": _thesis("MBB", i), "status": "pending", "pending": "immature",
              "target": _target(d)}
        if i >= 20:
            hit = i % 3 != 0
            realized = DIRS[i % 3] if hit else DIRS[(i + 1) % 3]
            kw.update(status="resolved", pending=None, ret=_ret(i), hit=hit, realized=realized,
                      baseline=realized if i % 4 == 0 else DIRS[(DIRS.index(realized) + 1) % 3])
        elif i == 2:
            kw.update(thesis=PLANTED_THESIS)
        elif i == 3:
            kw.update(thesis="first note on the bar", available=1_000)
            pairs.append(_entry("MBB", d, 300, direction="down", thesis="late second note", available=9_000,
                                status="superseded-not-scored"))
        elif i == 5:
            kw.update(direction="no-view", status="no-view")
        elif i == 6:
            kw.update(pending="calendar-coverage-expired", target=None)
        elif i == 7:
            kw.update(pending="target-bar-missing")
        elif i == 8:
            kw.update(pending="signal-bar-missing")
        elif i in (9, 10):
            kw.update(status=("out-of-window-not-scored", "window-undetermined-not-scored")[i - 9], pending=None,
                      target=None)
        elif i == 11:
            kw.update(thesis=LONG_THESIS)
        elif i == 12:
            kw.update(thesis=EXACT_THESIS)
        pairs.append(_entry("MBB", d, i, **kw))
    pairs.append(_entry("MBB", days[0], 600, direction="down", thesis="h60 newest", available=9_000, horizon=60,
                        pending="immature", target=_target(days[0])))
    pairs.append(_entry("MBB", days[30], 630, direction="up", thesis="h60 resolved", horizon=60, status="resolved",
                        ret=0.12, hit=True, target=_target(days[30]), realized="up", baseline="up"))
    for i in (1, 4):
        pairs.append(_entry("MBB", days[i], 400 + i, mode="retrospective", thesis=f"retro {i}", available=7_000,
                            status="retrospective-not-scored"))
    return pairs


def _vcb_journal() -> list[tuple[dict, dict]]:
    """31 resolved notes 21 sessions apart (nEffective 31: "ok") and a newer pending one."""
    days = list(reversed(_sessions(31 * 21 + 1, date(2024, 1, 1))))
    pairs = [_entry("VCB", days[0], 0, thesis=_thesis("VCB", 0), pending="immature", target=_target(days[0]))]
    for k in range(1, 32):
        d, hit = days[21 * k], k % 5 != 0
        realized = DIRS[k % 3] if hit else DIRS[(k + 1) % 3]
        pairs.append(_entry("VCB", d, k, direction=DIRS[k % 3], thesis=_thesis("VCB", k), status="resolved",
                            ret=_ret(k), hit=hit, target=_target(d), realized=realized, baseline="sideways"))
    return pairs


def _vic_journal() -> list[tuple[dict, dict]]:
    days = list(reversed(_sessions(2, date(2026, 10, 29))))
    return [_entry("VIC", d, i, direction="down", thesis=_thesis("VIC", i), pending="immature", target=_target(d))
            for i, d in enumerate(days)]


def _acb_journal() -> list[tuple[dict, dict]]:
    d = "2026-09-01"
    return [_entry("ACB", d, 0, thesis=_thesis("ACB", 0), status="resolved", ret=0.06, hit=True, target=_target(d),
                   realized="up", baseline="sideways")]


def _crowded_journal() -> tuple[list[tuple[dict, dict]], list[str]]:
    """199 retrospective notes on the 199 newest bars; days[199] holds a retrospective note and, written later, the
    only forward note among the backend's newest 200 notes; days[200..279]: 80 older resolved forward notes."""
    days = list(reversed(_sessions(280, date(2025, 9, 1))))
    pairs = [_entry("MBB", days[i], i, mode="retrospective", thesis=f"retro {i}", available=1_000,
                    status="retrospective-not-scored") for i in range(200)]
    pairs.append(_entry("MBB", days[199], 199_000, direction="down", thesis="the only forward note in the take",
                        pending="target-bar-missing", target=_target(days[199])))
    for i in range(200, 280):
        hit = i % 2 == 0
        realized = DIRS[i % 3] if hit else DIRS[(i + 2) % 3]
        pairs.append(_entry("MBB", days[i], i, direction=DIRS[i % 3], thesis=_thesis("MBB", i), status="resolved",
                            ret=_ret(i), hit=hit, target=_target(days[i]), realized=realized,
                            baseline=realized if i % 5 == 0 else DIRS[(DIRS.index(realized) + 1) % 3]))
    return pairs, days


def _busday(ms: int) -> int:
    weeks, rest = divmod((datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date() - date(2024, 1, 1)).days, 7)
    return weeks * 5 + min(rest, 5)  # 2024-01-01 is a Monday


def _summary(h: int, items: list[dict]) -> dict:
    """One direction-scores horizons[] entry as EquityDirectionScoringService.Summarize builds it: over every item."""
    fwd = [i for i in items if i["mode"] == "forward" and i["horizonSessions"] == h]

    def count(status: str) -> int:
        return sum(i["status"] == status for i in fwd)

    def rate(k: int) -> float | None:
        return k / len(resolved) if resolved else None

    resolved = [i for i in fwd if i["status"] == "resolved"]
    pending = [i for i in fwd if i["status"] == "pending"]
    n_eff, last = 0, None
    for b in sorted(_busday(i["signalBarOpenTimeMs"]) for i in resolved):
        if last is None or b - last >= h:
            n_eff, last = n_eff + 1, b
    hits = sum(i["hit"] is True for i in resolved)
    prior = sum(i["baselineCall"] == i["realized"] for i in resolved)
    scored = count("no-view") + len(resolved) + len(pending)
    return {"horizonSessions": h, "configId": f"tb-h{h}", "upThresholdPct": 0.05 if h == 20 else 0.1,
            "downThresholdPct": -0.04 if h == 20 else -0.06,
            "status": "no-data" if not resolved else "insufficient-sample" if n_eff < 30 else "ok",
            "forwardNotes": len(fwd), "outOfWindow": count("out-of-window-not-scored"),
            "windowUndetermined": count("window-undetermined-not-scored"),
            "eligible": scored + count("superseded-not-scored"), "superseded": count("superseded-not-scored"),
            "scored": scored, "noView": count("no-view"), "directional": len(resolved) + len(pending),
            "resolved": len(resolved), "pending": len(pending),
            "pendingReasons": {k: sum(i["pendingReason"] == v for i in pending) for k, v in (
                ("immature", "immature"), ("targetBarMissing", "target-bar-missing"),
                ("calendarCoverageExpired", "calendar-coverage-expired"), ("signalBarMissing", "signal-bar-missing"))},
            "hits": hits, "hitRate": rate(hits), "coverage": (len(resolved) + len(pending)) / scored if scored else None,
            "nEffective": n_eff, "minEffectiveForScoring": 30, "baselinePrior": {"hits": prior, "hitRate": rate(prior)},
            "bestConstantInHindsight": {"label": None, "hits": 0, "hitRate": None}}


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
    """GET-only fake of the endpoints the exporter reads, mimicking the backend. candles honour `limit` (the latest N
    bars). research-notes: take defaults to 50, clamped to 1..200, every mode and horizon counts, newest signal bar
    first, then availableTimeMs descending, then noteId. direction-scores: items take defaults to 200, clamped to
    1..500, newest signal bar first, then horizon, availableTimeMs and noteId; horizons[] (one entry per barrier
    config, ascending) summarise the WHOLE journal whatever take is. AdminGuard (401) runs before the capability
    check (400)."""

    def __init__(self, bars, technical, journals, unsupported=frozenset(), errors=None, clock=None):
        self.bars, self.technical, self.journals = bars, technical, journals
        self.unsupported, self.errors, self.clock, self.calls = set(unsupported), errors or {}, clock, []
        # per symbol: barrier configs (the backend has H=20 and H=60), a horizons[] rewrite, items left out
        self.configs, self.patch_horizons, self.hidden_items = {}, {}, set()

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
        journal = self.journals.get(symbol, [])
        if kind == "research-notes":
            take = max(1, min(int(_query(path).get("take", 50)), 200))
            notes = sorted((n for n, _ in journal),
                           key=lambda n: (-n["signalBarOpenTimeMs"], -n["availableTimeMs"], n["noteId"]))
            return {"contractVersion": "equity-research-notes-v1", "symbol": symbol, "items": notes[:take]}
        if kind == "research-notes/direction-scores":
            take = max(1, min(int(_query(path).get("take", 200)), 500))
            every = [i for _, i in journal]
            items = sorted((i for i in every if i["noteId"] not in self.hidden_items),
                           key=lambda i: (-i["signalBarOpenTimeMs"], i["horizonSessions"], i["availableTimeMs"],
                                          i["noteId"]))
            horizons = [_summary(h, every) for h in sorted(self.configs.get(symbol, (20, 60)))]
            patch = self.patch_horizons.get(symbol)
            return {"contractVersion": "equity-direction-scores-v1", "symbol": symbol,
                    "ruleVersion": "direction-score-v1", "rule": "r", "generatedAtUtc": "2026-11-02T12:00:00Z",
                    "calendarCoverageTo": "2026-12-31", "totalNotes": len(every),
                    "retrospectiveNotes": sum(i["mode"] != "forward" for i in every), "noDeclaredThresholdNotes": 0,
                    "horizons": patch(horizons) if patch else horizons,
                    "items": items[:take]}
        raise AssertionError(path)


def _market(clock=None, errors=None, unsupported=frozenset({"ACB"})) -> FakeBackend:
    bars = {s: _bars(300, 20000 + 5000 * k, zero=(10, 250, 299)) for k, s in enumerate(SYMBOLS)}
    technical = {s: _technical(bars[s], vnindex_status="stale" if s == "VIC" else "ok") for s in SYMBOLS}
    journals = {"MBB": _mbb_journal(), "VCB": _vcb_journal(), "VIC": _vic_journal(), "ACB": _acb_journal()}
    return FakeBackend(bars, technical, journals, unsupported=unsupported, errors=errors, clock=clock)


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


def _summary_line(capsys) -> dict:
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line.isascii()
    return json.loads(line)


def _load(out: Path) -> dict[str, dict]:
    return {p.relative_to(out).as_posix(): json.loads(p.read_text(encoding="utf-8")) for p in out.rglob("*.json")}


def _meta(out: Path) -> dict:
    return json.loads((out / "meta.json").read_text(encoding="utf-8"))


def _symbol(out: Path, s: str) -> dict:
    return json.loads((out / "symbols" / f"{s}.json").read_text(encoding="utf-8"))


def _keys(x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield k
            yield from _keys(v)
    elif isinstance(x, list):
        for v in x:
            yield from _keys(v)


def _strings(x):
    """Every decoded key and string value (a Windows path is escaped in the raw JSON text)."""
    if isinstance(x, dict):
        for k, v in x.items():
            yield k
            yield from _strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from _strings(v)
    elif isinstance(x, str):
        yield x


def _newest(fake: FakeBackend, symbol: str = "MBB") -> tuple[dict, dict]:
    """The newest forward H=20 note that holds its bar's slot, and its direction-score item."""
    return max((p for p in fake.journals[symbol] if p[0]["mode"] == "forward" and p[0]["horizonSessions"] == 20
                and p[1]["status"] != "superseded-not-scored"),
               key=lambda p: (p[0]["signalBarOpenTimeMs"], p[0]["availableTimeMs"]))


def _resolved(fake: FakeBackend) -> tuple[dict, dict]:
    """MBB's newest resolved H=20 note (days[20], inside the exported 60)."""
    day = _ms(_mbb_days()[20])
    return next(p for p in fake.journals["MBB"] if p[0]["signalBarOpenTimeMs"] == day and p[0]["horizonSessions"] == 20)


def _patch_h20(**changes):
    def patch(horizons):
        return [dict(h, **{k: (v(h) if callable(v) else v) for k, v in changes.items()})
                if h["horizonSessions"] == 20 else h for h in horizons]
    return patch


CORRUPTIONS = {
    "direction-buy": lambda f: [d.update(direction="buy") for d in _newest(f)],
    "status-reduce": lambda f: _newest(f)[1].update(status="reduce"),
    "status-retrospective-on-a-forward-note": lambda f: _newest(f)[1].update(status="retrospective-not-scored"),
    "status-no-declared-thresholds": lambda f: _newest(f)[1].update(status="no-declared-thresholds"),
    "status-capitalised": lambda f: _resolved(f)[1].update(status="Resolved"),
    "pending-reason-unknown": lambda f: _newest(f)[1].update(pendingReason="soon"),
    "hit-not-boolean": lambda f: _resolved(f)[1].update(hit="yes"),
    "score-of-another-note": lambda f: _newest(f)[1].update(direction="down"),
    "note-without-score": lambda f: f.hidden_items.add(_newest(f)[0]["noteId"]),
    "sample-status-unknown": lambda f: f.patch_horizons.update(MBB=_patch_h20(status="great")),
    "no-h20-entry": lambda f: f.configs.update(MBB=(60,)),
    "two-h20-entries": lambda f: f.patch_horizons.update(
        MBB=lambda hs: hs + [h for h in hs if h["horizonSessions"] == 20]),
    "hits-above-resolved": lambda f: f.patch_horizons.update(MBB=_patch_h20(hits=lambda h: h["resolved"] + 1)),
    "baseline-above-resolved": lambda f: f.patch_horizons.update(MBB=_patch_h20(
        baselinePrior=lambda h: {"hits": h["resolved"] + 1, "hitRate": None})),
    "resolved-not-a-count": lambda f: f.patch_horizons.update(MBB=_patch_h20(resolved=lambda h: float(h["resolved"]))),
}


def test_every_output_file_holds_only_contract_fields_and_no_planted_secret(screen_dir, tmp_path, monkeypatch):
    out = tmp_path / "data"
    assert _run(_market(Clock()), Clock(), screen_dir, out, monkeypatch) == 0
    files = _load(out)
    assert sorted(files) == ["meta.json", "screen.json"] + [f"symbols/{s}.json" for s in sorted(SYMBOLS)]
    for name, doc in files.items():
        leaked = FORBIDDEN & set(_keys(doc))
        assert not leaked, f"{name} leaks {leaked}"
        text, values = (out / name).read_text(encoding="utf-8"), list(_strings(doc))
        for secret in (KEY, "snap-secret-id", "session/MBB/", "fake.invalid", "http", "own-position", "a" * 64,
                       *THESIS_SECRETS):
            assert secret not in text and not any(secret in v for v in values), f"{name} contains {secret!r}"
    meta, screen = files["meta.json"], files["screen.json"]
    assert list(meta) == META_KEYS and meta["schema"] == "vn30-web-data/v2"
    assert all(list(u) == ["symbol", "part", "reason"] for u in meta["unavailable"])
    assert list(screen) == SCREEN_KEYS and all(list(r) == ROW_KEYS for r in screen["rows"])
    for s in SYMBOLS:
        doc = files[f"symbols/{s}.json"]
        assert list(doc) == SYMBOL_KEYS
        assert doc["score"] is None or list(doc["score"]) == SCORE_KEYS
        assert all(list(n) == NOTE_KEYS for n in doc["notes"])
        assert doc["indicators"] is None or list(doc["indicators"]) == INDICATOR_KEYS
        assert doc["relativeStrength"] is None or list(doc["relativeStrength"]) == ["20", "60"]
    assert len(files["symbols/MBB.json"]["notes"]) == 60  # the fake really served notes with forbidden fields
    assert list(files["symbols/MBB.json"]["score"]) == SCORE_KEYS and files["symbols/ACB.json"]["score"] is None


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
    assert all(u["part"] != "candles" for u in _meta(out)["unavailable"])


def test_indicators_are_the_1d_row_and_strength_is_the_vnindex_excess(screen_dir, tmp_path, monkeypatch):
    fake = _market(Clock())
    out = tmp_path / "data"
    assert _run(fake, Clock(), screen_dir, out, monkeypatch) == 0
    mbb, last = _symbol(out, "MBB"), fake.bars["MBB"][-1]
    assert mbb["indicators"] == {"date": web.session_of(last["openTimeMs"]), "close": last["close"],
                                 **{f: 100.0 + k + 0.5 for k, f in enumerate(INDICATOR_NAMES)}}
    assert mbb["relativeStrength"] == {"20": 0.031, "60": -0.04}
    assert mbb["label"] == "MBB (Name MBB)" and mbb["isBank"] is True and _symbol(out, "VIC")["isBank"] is False
    vic = _symbol(out, "VIC")  # VNINDEX stale: its old window is not the latest session
    assert vic["relativeStrength"] is None and vic["indicators"] is not None
    assert [u for u in _meta(out)["unavailable"] if u["symbol"] == "VIC"] == [
        {"symbol": "VIC", "part": "technical", "reason": NO_STRENGTH}]


def test_a_truthful_empty_answer_is_never_replaced_by_the_previous_run(screen_dir, tmp_path, monkeypatch):
    out = tmp_path / "data"
    fake = _market(Clock())
    fake.technical["VIC"] = _technical(fake.bars["VIC"])  # VNINDEX ok on the first run
    assert _run(fake, Clock(), screen_dir, out, monkeypatch) == 0
    assert _symbol(out, "VIC")["relativeStrength"] is not None and _symbol(out, "MBB")["indicators"] is not None
    later = _market(Clock())  # VIC: VNINDEX stale; MBB: no 1d indicator row yet
    later.technical["MBB"]["timeframes"][1]["indicators"] = None
    assert _run(later, Clock(), screen_dir, out, monkeypatch) == 0
    vic, mbb = _symbol(out, "VIC"), _symbol(out, "MBB")
    assert vic["relativeStrength"] is None and vic["indicators"] is not None
    assert mbb["indicators"] is None and mbb["relativeStrength"] == {"20": 0.031, "60": -0.04}
    unavailable = _meta(out)["unavailable"]
    assert {"symbol": "VIC", "part": "technical", "reason": NO_STRENGTH} in unavailable
    assert {"symbol": "MBB", "part": "technical", "reason": "backend chưa có chỉ báo 1d"} in unavailable


def test_notes_are_the_newest_forward_20_session_notes_joined_by_note_id(screen_dir, tmp_path, monkeypatch):
    fake = _market(Clock())
    out = tmp_path / "data"
    assert _run(fake, Clock(), screen_dir, out, monkeypatch) == 0
    notes, days = _symbol(out, "MBB")["notes"], _mbb_days()
    # the newest 60 bars; the superseded second note of days[3], the H=60 notes (one written last on days[0]) and the
    # retrospective notes are not exported
    assert [n["signalDate"] for n in notes] == days[:60]
    assert {n["horizonSessions"] for n in notes} == {20}
    assert not {"late second note", "h60 newest", "h60 resolved", "retro 1", "retro 4"} & {n["thesis"] for n in notes}
    assert notes[0] == {"signalDate": days[0], "horizonSessions": 20, "direction": "up", "thesis": _thesis("MBB", 0),
                        "status": "pending", "pendingReason": "immature", "targetDate": _target(days[0]),
                        "returnPct": None, "hit": None}
    assert (notes[2]["thesis"], notes[2]["direction"], notes[2]["status"]) == (None, "sideways", "pending")
    assert (notes[3]["thesis"], notes[3]["direction"], notes[3]["status"]) == ("first note on the bar", "up", "pending")
    assert notes[5] == {"signalDate": days[5], "horizonSessions": 20, "direction": "no-view",
                        "thesis": _thesis("MBB", 5), "status": "no-view", "pendingReason": "immature",
                        "targetDate": _target(days[5]), "returnPct": None, "hit": None}
    assert (notes[6]["pendingReason"], notes[6]["targetDate"]) == ("calendar-coverage-expired", None)
    assert [n["pendingReason"] for n in notes[7:9]] == ["target-bar-missing", "signal-bar-missing"]
    assert notes[9] == {"signalDate": days[9], "horizonSessions": 20, "direction": "up", "thesis": _thesis("MBB", 9),
                        "status": "out-of-window-not-scored", "pendingReason": None, "targetDate": None,
                        "returnPct": None, "hit": None}
    assert (notes[10]["status"], notes[10]["direction"]) == ("window-undetermined-not-scored", "down")
    assert notes[11]["thesis"] == LONG_THESIS[:1499] + "\u2026" and len(notes[11]["thesis"]) == 1500
    assert notes[12]["thesis"] == EXACT_THESIS
    assert notes[20] == {"signalDate": days[20], "horizonSessions": 20, "direction": "sideways",
                         "thesis": _thesis("MBB", 20), "status": "scored", "pendingReason": None,
                         "targetDate": _target(days[20]), "returnPct": _ret(20), "hit": True}
    assert (notes[21]["direction"], notes[21]["status"], notes[21]["hit"], notes[21]["returnPct"]) == \
           ("up", "scored", False, _ret(21))
    assert [p for p, _, _ in fake.calls if "/MBB/research-notes" in p] == [
        "/api/equities/MBB/research-notes?take=200", "/api/equities/MBB/research-notes/direction-scores?take=500"]


def test_score_is_the_whole_journal_h20_total_from_direction_scores(screen_dir, tmp_path, monkeypatch):
    out = tmp_path / "data"
    assert _run(_market(Clock()), Clock(), screen_dir, out, monkeypatch) == 0
    mbb = _symbol(out, "MBB")
    # 80 resolved H=20 notes in the journal (53 hits, prior right on 20), only 40 of them among the exported 60 notes;
    # the resolved H=60 note counts nowhere
    assert mbb["score"] == {"scored": 80, "hits": 53, "baselineHits": 20, "sampleStatus": "insufficient-sample"}
    exported = [n for n in mbb["notes"] if n["status"] == "scored"]
    assert (len(exported), sum(n["hit"] for n in exported)) == (40, 27)
    assert _symbol(out, "VCB")["score"] == {"scored": 31, "hits": 25, "baselineHits": 10, "sampleStatus": "ok"}
    assert _symbol(out, "VIC")["score"] == {"scored": 0, "hits": 0, "baselineHits": 0, "sampleStatus": "no-data"}
    assert _symbol(out, "ACB")["score"] is None  # UNSUPPORTED_CAPABILITY: no score at all, never 0/0

    extra = _market(Clock())  # a further barrier config listed first: the entry is chosen by horizonSessions
    extra.configs["MBB"] = (10, 20, 60)
    assert _run(extra, Clock(), screen_dir, tmp_path / "extra", monkeypatch) == 0
    assert _symbol(tmp_path / "extra", "MBB")["score"] == mbb["score"]


def test_retrospective_notes_take_slots_and_the_scores_take_covers_the_notes_take(screen_dir, tmp_path, monkeypatch):
    fake = _market(Clock())
    fake.journals["MBB"], days = _crowded_journal()
    out = tmp_path / "data"
    assert _run(fake, Clock(), screen_dir, out, monkeypatch) == 0
    mbb = _symbol(out, "MBB")
    # the backend's newest 200 notes are 199 retrospective ones and the forward note of days[199], whose score item is
    # the 201st item (the bar's retrospective note was written first): it is exported with its status, while the score
    # still counts the 80 older resolved notes
    assert mbb["notes"] == [{"signalDate": days[199], "horizonSessions": 20, "direction": "down",
                             "thesis": "the only forward note in the take", "status": "pending",
                             "pendingReason": "target-bar-missing", "targetDate": _target(days[199]),
                             "returnPct": None, "hit": None}]
    assert mbb["score"] == {"scored": 80, "hits": 40, "baselineHits": 16, "sampleStatus": "insufficient-sample"}
    assert not any(u["symbol"] == "MBB" for u in _meta(out)["unavailable"])


@pytest.mark.parametrize("corruption", sorted(CORRUPTIONS))
def test_a_value_outside_the_contract_makes_the_notes_part_malformed(corruption, screen_dir, tmp_path, monkeypatch):
    out = tmp_path / "data"
    assert _run(_market(Clock()), Clock(), screen_dir, out, monkeypatch) == 0
    before = _symbol(out, "MBB")
    fake = _market(Clock())
    CORRUPTIONS[corruption](fake)
    assert _run(fake, Clock(), screen_dir, out, monkeypatch) == 0
    mbb = _symbol(out, "MBB")
    assert (mbb["notes"], mbb["score"]) == (before["notes"], before["score"])  # kept from the previous run
    reasons = [u["reason"] for u in _meta(out)["unavailable"] if u["symbol"] == "MBB" and u["part"] == "notes"]
    assert len(reasons) == 1 and reasons[0].startswith("phản hồi sai định dạng (") and reasons[0].endswith(KEPT)

    fresh = tmp_path / "fresh"  # no previous run: no notes and no score, never 0/0
    fake = _market(Clock())
    CORRUPTIONS[corruption](fake)
    assert _run(fake, Clock(), screen_dir, fresh, monkeypatch) == 0
    assert (_symbol(fresh, "MBB")["notes"], _symbol(fresh, "MBB")["score"]) == ([], None)
    assert _symbol(fresh, "VCB")["score"]["scored"] == 31  # one ticker's payload never touches another


@pytest.mark.parametrize("private", COLDCHECK_PRIVATE)
def test_a_thesis_naming_private_or_action_text_is_withheld(private, screen_dir, tmp_path, monkeypatch, capsys):
    fake = _market(Clock())
    _newest(fake)[0]["body"]["thesis"] = f"MBB 1d closed below SMA50, RSI14 34. {private}. View: sideways."
    out = tmp_path / "data"
    assert _run(fake, Clock(), screen_dir, out, monkeypatch) == 0
    summary = _summary_line(capsys)
    notes = _symbol(out, "MBB")["notes"]
    assert notes[0]["thesis"] is None and (notes[0]["direction"], notes[0]["status"]) == ("up", "pending")
    assert notes[1]["thesis"] == _thesis("MBB", 1)  # a clean thesis is still published
    assert summary["thesesWithheld"] == 2  # this one and the fixture's own (days[2])
    for p in out.rglob("*.json"):
        assert not any(private in v for v in _strings(json.loads(p.read_text(encoding="utf-8")))), p.name


@pytest.mark.parametrize("text", TECHNICAL_THESES)
def test_a_technical_thesis_is_published_unchanged(text):
    assert web.public_thesis(text) == (text, False)


@pytest.mark.parametrize("text", PRIVATE_THESES)
def test_private_or_action_text_anywhere_withholds_the_thesis(text):
    assert web.public_thesis(f"MBB 1d closes near SMA50. {text}. View: sideways, for reference only.") == (None, True)


def test_a_long_thesis_is_cut_to_1500_characters_with_a_marker():
    base = "Bollinger middle 18,900 caps the 1d close. "
    exact, longer = (base * 40)[:1500], (base * 40)[:1501]
    assert web.public_thesis(exact) == (exact, False)
    assert web.public_thesis(longer) == (longer[:1499] + "\u2026", False)
    assert web.public_thesis(exact + " Hold.") == (None, True)  # the filter reads the whole text, past the cut
    assert web.public_thesis("  \n ") == (None, False) and web.public_thesis(None) == (None, False)
    assert web.public_thesis(["not", "text"]) == (None, False)


FAILURES = {  # part, run-2 change, reason before the suffix
    "technical-500": ("technical", {"errors": {("MBB", "technical"): [500]}}, "HTTP 500"),
    "technical-429-twice": ("technical", {"errors": {("MBB", "technical"): [429, 429]}}, "HTTP 429"),
    "technical-malformed": ("technical", {"technical": lambda t: t.pop("relativeStrength")},
                            "phản hồi sai định dạng (KeyError)"),
    "notes-500": ("notes", {"errors": {("MBB", "research-notes"): [500]}}, "HTTP 500"),
    "scores-401": ("notes", {"errors": {("MBB", "research-notes/direction-scores"): [401]}}, "HTTP 401"),
    "notes-without-admin-key": ("notes", {"key": None}, NO_KEY),
}


@pytest.mark.parametrize("failure", sorted(FAILURES))
def test_a_failed_part_keeps_the_previous_v2_run(failure, screen_dir, tmp_path, monkeypatch, capsys):
    part, change, reason = FAILURES[failure]
    out = tmp_path / "data"
    assert _run(_market(Clock()), Clock(), screen_dir, out, monkeypatch) == 0
    before = _symbol(out, "MBB")
    fake = _market(Clock(), errors=change.get("errors"))
    if "technical" in change:
        change["technical"](fake.technical["MBB"])
    last = fake.bars["MBB"][-1]  # a newer bar: candles are read fresh, the file is not kept whole
    fake.bars["MBB"].append(dict(last, openTimeMs=last["openTimeMs"] + 3 * 86_400_000, volume=5))
    capsys.readouterr()
    assert _run(fake, Clock(), screen_dir, out, monkeypatch, key=change.get("key", KEY)) == 0
    summary = _summary_line(capsys)
    mbb = _symbol(out, "MBB")
    kept = ("indicators", "relativeStrength") if part == "technical" else ("notes", "score")
    fresh = ("notes", "score") if part == "technical" else ("indicators", "relativeStrength")
    assert all(before[f] and mbb[f] == before[f] for f in kept + fresh)  # the backend data did not change
    assert mbb["candles"][-1][0] == web.session_of(fake.bars["MBB"][-1]["openTimeMs"]) != before["candles"][-1][0]
    assert {"symbol": "MBB", "part": part, "reason": reason + KEPT} in _meta(out)["unavailable"]
    assert summary["keptByPart"][part] >= 1 and summary["written"] == len(SYMBOLS)


def test_a_kept_thesis_passes_the_privacy_filter_again(screen_dir, tmp_path, monkeypatch, capsys):
    out = tmp_path / "data"
    assert _run(_market(Clock()), Clock(), screen_dir, out, monkeypatch) == 0
    for s in ("MBB", "VCB"):  # as if an older filter had let them through
        doc = _symbol(out, s)
        doc["notes"][1]["thesis"] = "Hold above 19,500; cash 50m."
        (out / "symbols" / f"{s}.json").write_text(json.dumps(doc), encoding="utf-8")
    capsys.readouterr()
    errors = {("MBB", "research-notes"): [500], ("VCB", "candles"): [500]}  # MBB notes kept, VCB kept whole
    assert _run(_market(Clock(), errors=errors), Clock(), screen_dir, out, monkeypatch) == 0
    summary = _summary_line(capsys)
    mbb, vcb = _symbol(out, "MBB"), _symbol(out, "VCB")
    assert mbb["notes"][1]["thesis"] is None and vcb["notes"][1]["thesis"] is None
    assert mbb["notes"][0]["thesis"] == _thesis("MBB", 0) and vcb["notes"][0]["thesis"] == _thesis("VCB", 0)
    assert summary["thesesWithheld"] == 2 and summary["keptByPart"] == {"notes": 1, "candles": 1}
    assert summary["written"] == len(SYMBOLS) - 1
    assert not any("cash" in v for p in out.rglob("*.json") for v in _strings(json.loads(p.read_text("utf-8"))))


def _errors() -> dict:
    return {("MBB", "technical"): [500], ("MBB", "research-notes"): [500], ("VCB", "candles"): [500],
            ("ACB", "technical"): [500]}


def test_without_a_previous_v2_run_a_failed_part_is_written_empty(screen_dir, tmp_path, monkeypatch):
    fresh = tmp_path / "fresh"
    assert _run(_market(Clock(), errors=_errors()), Clock(), screen_dir, fresh, monkeypatch) == 0
    mbb, vcb = _symbol(fresh, "MBB"), _symbol(fresh, "VCB")
    assert (mbb["indicators"], mbb["relativeStrength"], mbb["notes"], mbb["score"]) == (None, None, [], None)
    assert vcb["candles"] == [] and vcb["indicators"] is not None and vcb["score"]["scored"] == 31
    unavailable = _meta(fresh)["unavailable"]
    for u in ({"symbol": "MBB", "part": "technical", "reason": "HTTP 500" + NOTHING_KEPT},
              {"symbol": "MBB", "part": "notes", "reason": "HTTP 500" + NOTHING_KEPT},
              {"symbol": "VCB", "part": "candles", "reason": "HTTP 500" + NOTHING_KEPT},
              {"symbol": "ACB", "part": "technical", "reason": "HTTP 500" + NOTHING_KEPT}):
        assert u in unavailable

    v1 = tmp_path / "v1"  # files from a v1 data set (other meaning of notes/score/thesis) are never a source
    assert _run(_market(Clock()), Clock(), screen_dir, v1, monkeypatch) == 0
    meta = _meta(v1)
    (v1 / "meta.json").write_text(json.dumps(dict(meta, schema="vn30-web-data/v1")), encoding="utf-8")
    assert _run(_market(Clock(), errors=_errors()), Clock(), screen_dir, v1, monkeypatch) == 0
    assert {s: _symbol(v1, s) for s in SYMBOLS} == {s: _symbol(fresh, s) for s in SYMBOLS}
    assert _meta(v1)["unavailable"] == unavailable

    gone = tmp_path / "gone"  # a previous v2 run whose meta.json cannot be read
    assert _run(_market(Clock()), Clock(), screen_dir, gone, monkeypatch) == 0
    (gone / "meta.json").write_text("{not json", encoding="utf-8")
    assert _run(_market(Clock(), errors=_errors()), Clock(), screen_dir, gone, monkeypatch) == 0
    assert _symbol(gone, "MBB") == mbb and _meta(gone)["unavailable"] == unavailable


def test_a_previous_part_is_kept_only_when_it_validates_and_was_read(screen_dir, tmp_path, monkeypatch):
    out = tmp_path / "data"
    assert _run(_market(Clock()), Clock(), screen_dir, out, monkeypatch) == 0  # ACB: UNSUPPORTED, nothing read
    before = _symbol(out, "MBB")
    (out / "symbols" / "MBB.json").write_text(json.dumps(dict(before, score={"scored": 2, "hits": 1})),
                                              encoding="utf-8")  # a v1-shaped score
    assert _run(_market(Clock(), errors=_errors(), unsupported=frozenset()), Clock(), screen_dir, out,
                monkeypatch) == 0
    mbb, acb = _symbol(out, "MBB"), _symbol(out, "ACB")
    assert (mbb["notes"], mbb["score"]) == ([], None) and mbb["indicators"] == before["indicators"]
    assert (acb["indicators"], acb["relativeStrength"]) == (None, None) and acb["score"]["scored"] == 1
    unavailable = _meta(out)["unavailable"]
    assert {"symbol": "MBB", "part": "notes", "reason": "HTTP 500" + NOTHING_KEPT} in unavailable
    assert {"symbol": "MBB", "part": "technical", "reason": "HTTP 500" + KEPT} in unavailable
    assert {"symbol": "ACB", "part": "technical", "reason": "HTTP 500" + NOTHING_KEPT} in unavailable


def test_unsupported_capability_is_listed_without_a_suffix_and_never_kept(screen_dir, tmp_path, monkeypatch):
    out = tmp_path / "data"
    assert _run(_market(Clock(), unsupported=frozenset()), Clock(), screen_dir, out, monkeypatch) == 0
    acb = _symbol(out, "ACB")
    assert acb["indicators"] is not None and acb["notes"] and acb["score"]["scored"] == 1  # served before
    assert _run(_market(Clock()), Clock(), screen_dir, out, monkeypatch) == 0  # now 400 UNSUPPORTED_CAPABILITY
    acb = _symbol(out, "ACB")
    assert (acb["indicators"], acb["relativeStrength"], acb["notes"], acb["score"]) == (None, None, [], None)
    assert len(acb["candles"]) == 258  # candles need only ingestion
    assert [u for u in _meta(out)["unavailable"] if u["symbol"] == "ACB"] == [
        {"symbol": "ACB", "part": "technical", "reason": UNSUPPORTED},
        {"symbol": "ACB", "part": "notes", "reason": UNSUPPORTED}]


def test_technical_calls_are_spaced_15_s_and_a_429_waits_a_minute_and_retries_once(screen_dir, tmp_path, monkeypatch):
    clock = Clock()
    fake = _market(clock, errors={("MBB", "technical"): [429], ("VCB", "technical"): [429, 429],
                                  ("VIC", "candles"): [429]})
    out = tmp_path / "data"
    assert _run(fake, clock, screen_dir, out, monkeypatch) == 0
    assert clock.sleeps.count(60.0) == 3  # MBB once, VCB once (then gives up), VIC candles once
    times = [t for p, _, t in fake.calls if p.endswith("/technical?bars=1")]
    assert len(times) == 6  # ACB, MBB x2, VCB x2, VIC
    assert all(b - a >= 15.0 for a, b in zip(times, times[1:]))
    assert all(sum(1 for u in times if t <= u < t + 60) <= 4 for t in times)
    assert _symbol(out, "MBB")["indicators"] is not None and len(_symbol(out, "VIC")["candles"]) == 258
    assert {"symbol": "VCB", "part": "technical", "reason": "HTTP 429" + NOTHING_KEPT} in _meta(out)["unavailable"]


def test_without_the_admin_key_notes_are_never_requested(screen_dir, tmp_path, monkeypatch):
    fake = _market(Clock())
    out = tmp_path / "data"
    assert _run(fake, Clock(), screen_dir, out, monkeypatch, key=None) == 0
    assert not any("research-notes" in p for p, _, _ in fake.calls)
    notes_parts = [u for u in _meta(out)["unavailable"] if u["part"] == "notes"]
    assert sorted(u["symbol"] for u in notes_parts) == sorted(SYMBOLS)
    assert all(u["reason"] == NO_KEY + NOTHING_KEPT for u in notes_parts)
    assert (_symbol(out, "MBB")["notes"], _symbol(out, "MBB")["score"]) == ([], None)

    with_key = _market(Clock())
    assert _run(with_key, Clock(), screen_dir, tmp_path / "keyed", monkeypatch) == 0
    for path, headers, _ in with_key.calls:  # the key goes only to the two admin endpoints
        assert ("X-Admin-Key" in headers) == ("research-notes" in path)
        assert headers.get("X-Admin-Key") in (None, KEY)


def test_unchanged_files_are_not_rewritten(screen_dir, tmp_path, monkeypatch):
    out = tmp_path / "data"
    fake = _market(Clock())
    assert _run(fake, Clock(), screen_dir, out, monkeypatch) == 0
    meta = _meta(out)
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
    assert _meta(out)["generatedAtUtc"] != "2026-01-01T00:00:00Z"
    assert not list(out.rglob("*.tmp"))
    assert (out / "meta.json").read_bytes().endswith(b"\n") and b"\r\n" not in (out / "meta.json").read_bytes()


def test_a_file_is_replaced_whole_and_no_tmp_file_is_left(screen_dir, tmp_path, monkeypatch):
    out = tmp_path / "data"
    assert _run(_market(Clock()), Clock(), screen_dir, out, monkeypatch) == 0
    before = (out / "symbols" / "MBB.json").read_bytes()
    (out / "symbols" / "ZZZ.json.tmp").write_bytes(b"left by an interrupted run")
    real_replace = os.replace

    def replace(src, dst):
        if str(dst).endswith("MBB.json"):
            raise PermissionError("the file is held by another process")
        real_replace(src, dst)

    monkeypatch.setattr(web.os, "replace", replace)
    newer = _market(Clock())
    last = newer.bars["MBB"][-1]
    newer.bars["MBB"].append(dict(last, openTimeMs=last["openTimeMs"] + 3 * 86_400_000, volume=5))
    with pytest.raises(PermissionError):
        _run(newer, Clock(), screen_dir, out, monkeypatch)
    assert (out / "symbols" / "MBB.json").read_bytes() == before  # never half-written
    assert not list(out.rglob("*.tmp"))


def test_a_ticker_whose_candles_fail_keeps_its_previous_file(screen_dir, tmp_path, monkeypatch):
    out = tmp_path / "data"
    assert _run(_market(Clock()), Clock(), screen_dir, out, monkeypatch) == 0
    previous = (out / "symbols" / "VCB.json").read_bytes()
    fake = _market(Clock(), errors={("VCB", "candles"): [500]})
    fake.technical["VCB"]["timeframes"][1]["indicators"]["rsi14"] = 1.0  # would change the file if it were rebuilt
    assert _run(fake, Clock(), screen_dir, out, monkeypatch) == 0
    assert (out / "symbols" / "VCB.json").read_bytes() == previous
    assert not any(p.startswith("/api/equities/VCB/") and "/candles" not in p for p, _, _ in fake.calls)
    assert {"symbol": "VCB", "part": "candles", "reason": "HTTP 500" + KEPT} in _meta(out)["unavailable"]

    fresh = tmp_path / "fresh"
    assert _run(_market(Clock(), errors={("VCB", "candles"): [500]}), Clock(), screen_dir, fresh, monkeypatch) == 0
    vcb = _symbol(fresh, "VCB")
    assert vcb["candles"] == [] and vcb["indicators"] is not None and vcb["notes"]
    assert {"symbol": "VCB", "part": "candles", "reason": "HTTP 500" + NOTHING_KEPT} in _meta(fresh)["unavailable"]

    again = _market(Clock(), errors={("VCB", "candles"): [500]})  # the previous file has no candles either: nothing
    assert _run(again, Clock(), screen_dir, fresh, monkeypatch) == 0  # to keep, so the other parts are read again
    assert any(p.startswith("/api/equities/VCB/technical") for p, _, _ in again.calls)
    assert {"symbol": "VCB", "part": "candles", "reason": "HTTP 500" + NOTHING_KEPT} in _meta(fresh)["unavailable"]


def test_screen_json_follows_the_newest_screen_file(screen_dir, tmp_path, monkeypatch):
    out = tmp_path / "data"
    assert _run(_market(Clock()), Clock(), screen_dir, out, monkeypatch) == 0
    files = _load(out)
    screen, meta = files["screen.json"], files["meta.json"]
    assert screen["session"] == SESSION and meta["session"] == SESSION
    assert [r["symbol"] for r in screen["rows"]] == list(SYMBOLS)  # the screen's order is kept
    assert meta["symbols"] == sorted(SYMBOLS) and meta["schema"] == "vn30-web-data/v2"
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
