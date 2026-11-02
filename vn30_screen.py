"""VN30-SCREEN: daily VN30 strength table. An observation, not a signal: the VN30-1 backtest of ranking the
basket by trailing return ended not-enough-evidence (evidence/vn30-1-20261102/README.md).

Reads the backend HTTP API only (--backend or BACKEND_BASE_URL): /api/equities for per-symbol freshness and
/api/equities/{symbol}/candles for 1d bars of VNINDEX and the basket in effect on the ranking session. Returns
follow vn30_rotation.select: close-to-close over 21 x months sessions of the VNINDEX calendar, zero-volume bars
are not sessions, a member without a bar on either end gets no return. Writes <session>.json, <session>.md and
latest.md to --out-dir. Never writes to the backend or the database.

Usage (from ai/, venv python; not -I, which drops ai/ from sys.path):
  python vn30_screen.py [--as-of YYYY-MM-DD] [--out-dir DIR] [--backend URL]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import vn30_rotation as vr

SCHEMA = "vn30-screen/v1"
LOOKBACK_MONTHS = (3, 6, 12)
RANK_MONTHS, TOP_K = 6, 10  # the configuration VN30-1 selected on design-period Sharpe
CANDLE_LIMIT = 420  # 252 sessions for 12 months plus room for halts and zero-volume bars
STALE_BASKET_DAYS = 190  # VN30 is reviewed every six months
VN_OFFSET = timedelta(hours=7)
CAVEAT = "Quan sát, không phải tín hiệu: VN30-1 kết luận not-enough-evidence cho chiến lược xếp hạng này."
AI_ROOT = Path(__file__).resolve().parent
DEFAULT_MEMBERSHIP = AI_ROOT / "contracts" / "vn30-membership.json"
DEFAULT_OUT = AI_ROOT / "docs" / "research" / "evidence" / "vn30-screen"
DEFAULT_BACKEND = "http://127.0.0.1:5197"  # "localhost" costs ~2 s per urllib request on Windows (IPv6 tried first)

Fetch = Callable[[str], Any]


class UsageError(ValueError):
    pass


class BackendError(RuntimeError):
    """The backend answered, but not with the payload shape this script reads."""


def session_of(open_time_ms: int) -> str:
    return (datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc) + VN_OFFSET).date().isoformat()


def http_fetch(backend: str, timeout: float = 30.0) -> Fetch:
    def fetch(path: str) -> Any:
        with urllib.request.urlopen(backend.rstrip("/") + path, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    return fetch


def load_bars(fetch: Fetch, symbol: str, end_ms: int | None) -> dict[str, tuple[float, float]]:
    """Session -> (open, close), zero-volume bars dropped (vn30_rotation.load_prices)."""
    path = f"/api/equities/{symbol}/candles?timeframe=1d&limit={CANDLE_LIMIT}"
    if end_ms is not None:
        path += f"&endTimeMs={end_ms}"
    try:
        return {session_of(c["openTimeMs"]): (float(c["open"]), float(c["close"]))
                for c in fetch(path) if float(c["volume"]) > 0}
    except (KeyError, TypeError, ValueError) as e:
        raise BackendError(f"{symbol} candles: {type(e).__name__}: {e}") from e


def build(fetch: Fetch, periods: vr.Periods, as_of: str | None = None) -> dict:
    end_ms = None
    if as_of is not None:
        end_ms = int((datetime.fromisoformat(as_of) + timedelta(days=1) - VN_OFFSET)
                     .replace(tzinfo=timezone.utc).timestamp() * 1000) - 1
    index = {d: b for d, b in load_bars(fetch, vr.INDEX, end_ms).items() if as_of is None or d <= as_of}
    if not index:
        raise UsageError("VNINDEX has no 1d bars up to the ranking session")
    calendar = sorted(index)
    day = calendar[-1]
    if as_of is not None and day != as_of:
        raise UsageError(f"{as_of} is not a VNINDEX session (latest session on or before it: {day})")
    effective = max((d for d, _ in periods if d <= day), default=None)
    members = vr.members_on(periods, day)
    if members is None or effective is None:
        raise UsageError(f"no VN30 basket in effect on {day}")
    prices: vr.Prices = {vr.INDEX: index}
    unknown = []
    for s in sorted(members):
        try:
            bars = load_bars(fetch, s, end_ms)
        except urllib.error.HTTPError as e:
            # a basket file updated before the backend catalog: blank that row, keep the table
            if e.code != 404:
                raise
            unknown.append(s)
            bars = {}
        prices[s] = {d: b for d, b in bars.items() if d <= day}

    i = len(calendar) - 1
    anchors = {m: calendar[i - vr.SESSIONS_PER_MONTH * m] if i >= vr.SESSIONS_PER_MONTH * m else None
               for m in LOOKBACK_MONTHS}

    def ret(symbol: str, m: int) -> float | None:
        bars, past = prices[symbol], anchors[m]
        if past is None or day not in bars or past not in bars:
            return None
        return bars[day][1] / bars[past][1] - 1.0

    index_ret = {m: ret(vr.INDEX, m) for m in LOOKBACK_MONTHS}
    rows = []
    for s in sorted(members):
        r = {m: ret(s, m) for m in LOOKBACK_MONTHS}
        rows.append({"symbol": s, "close": prices[s][day][1] if day in prices[s] else None,
                     "returns": {str(m): r[m] for m in LOOKBACK_MONTHS},
                     "excessVsIndex": {str(m): (r[m] - index_ret[m] if r[m] is not None and index_ret[m] is not None
                                                else None) for m in LOOKBACK_MONTHS},
                     "ranks": {}})
    for m in LOOKBACK_MONTHS:
        scored = sorted((-row["returns"][str(m)], row["symbol"]) for row in rows if row["returns"][str(m)] is not None)
        rank = {s: n for n, (_, s) in enumerate(scored, start=1)}
        for row in rows:
            row["ranks"][str(m)] = rank.get(row["symbol"])
    top = set(vr.select(calendar, prices, members, i, RANK_MONTHS, TOP_K))
    for row in rows:
        row["topByRankMonths"] = row["symbol"] in top
    rows.sort(key=lambda row: (row["ranks"][str(RANK_MONTHS)] is None, row["ranks"][str(RANK_MONTHS)] or 0, row["symbol"]))

    try:
        status = {item["symbol"]: item for item in fetch("/api/equities")["items"]}
        not_up_to_date = sorted(s for s in members | {vr.INDEX}
                                if not any(t["timeframe"] == "1d" and t["upToDate"]
                                           for t in status.get(s, {}).get("ingestion", {}).get("timeframes", [])))
    except (KeyError, TypeError, AttributeError) as e:
        raise BackendError(f"/api/equities: {type(e).__name__}: {e}") from e
    return {
        "schema": SCHEMA,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session": day,
        "asOfRequested": as_of,
        "caveat": CAVEAT,
        "basket": {"effectiveDate": effective,
                   "possiblyStale": (date.fromisoformat(day) - date.fromisoformat(effective)).days > STALE_BASKET_DAYS},
        "anchorSessions": {str(m): anchors[m] for m in LOOKBACK_MONTHS},
        "indexReturns": {str(m): index_ret[m] for m in LOOKBACK_MONTHS},
        "ranking": {"lookbackMonths": RANK_MONTHS, "topK": TOP_K},
        "missingSessionBar": sorted(s for s in members if day not in prices[s]),
        "unknownToBackend": unknown,
        "notUpToDateNow": not_up_to_date,
        "rows": rows,
    }


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:+.1f}%"


def to_markdown(screen: dict) -> str:
    basket = screen["basket"]
    lines = [f"# Bảng lọc VN30 — phiên {screen['session']}", "",
             f"> {screen['caveat']}", "",
             f"- Rổ VN30 hiệu lực từ {basket['effectiveDate']}."
             + (" **Rổ có thể đã cũ**: đã quá 190 ngày, kiểm kỳ xét duyệt mới." if basket["possiblyStale"] else ""),
             "- VNINDEX: " + ", ".join(f"{m} tháng {_pct(screen['indexReturns'][str(m)])}" for m in LOOKBACK_MONTHS) + ".",
             f"- Xếp theo lợi nhuận {RANK_MONTHS} tháng; ★ = top {TOP_K} (cấu hình VN30-1 đã chọn)."]
    if screen["unknownToBackend"]:
        lines.append(f"- **Backend chưa có mã** (cập nhật danh mục mã của backend): {', '.join(screen['unknownToBackend'])}.")
    if screen["missingSessionBar"]:
        lines.append(f"- Chưa có nến phiên {screen['session']}: {', '.join(screen['missingSessionBar'])}.")
    if screen["notUpToDateNow"]:
        lines.append(f"- Backend báo chưa nạp tới phiên mới nhất (lúc chạy): {', '.join(screen['notUpToDateNow'])}.")
    lines += ["", "| # | Mã | Giá đóng | 3T | so VNI | 6T | so VNI | 12T | so VNI | Top |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|:---:|"]
    for row in screen["rows"]:
        cells = [str(row["ranks"][str(RANK_MONTHS)] or "—"), row["symbol"],
                 "—" if row["close"] is None else f"{row['close']:,.0f}".replace(",", ".")]
        for m in LOOKBACK_MONTHS:
            cells += [_pct(row["returns"][str(m)]), _pct(row["excessVsIndex"][str(m)])]
        cells.append("★" if row["topByRankMonths"] else "")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(screen: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = to_markdown(screen)
    paths = [out_dir / f"{screen['session']}.json", out_dir / f"{screen['session']}.md"]
    if screen["asOfRequested"] is None:  # a backdated run must not replace the latest table
        paths.append(out_dir / "latest.md")
    paths[0].write_text(json.dumps(screen, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    for p in paths[1:]:
        p.write_text(md, encoding="utf-8")
    return paths


def main(argv: list[str] | None = None, fetch: Fetch | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", help="ranking session YYYY-MM-DD (default: latest VNINDEX session)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--membership", type=Path, default=DEFAULT_MEMBERSHIP)
    ap.add_argument("--backend", default=os.getenv("BACKEND_BASE_URL") or DEFAULT_BACKEND)
    args = ap.parse_args(argv)
    try:
        screen = build(fetch or http_fetch(args.backend), vr.load_membership(args.membership), args.as_of)
    except (urllib.error.URLError, TimeoutError, OSError, BackendError, json.JSONDecodeError) as e:
        print(f"backend error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    except ValueError as e:  # UsageError, a bad --as-of or membership file
        print(f"usage error: {e}", file=sys.stderr)
        return 2
    paths = write_outputs(screen, args.out_dir)
    top = [r["symbol"] for r in screen["rows"] if r["topByRankMonths"]]
    print(json.dumps({"session": screen["session"], "members": len(screen["rows"]), "top": top,
                      "missingSessionBar": screen["missingSessionBar"], "unknownToBackend": screen["unknownToBackend"],
                      "notUpToDateNow": screen["notUpToDateNow"],
                      "files": [str(p) for p in paths]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
