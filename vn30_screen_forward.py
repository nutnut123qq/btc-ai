"""Forward ledger of the daily VN30 screen's top 10 (roadmap 0.7).

The screen ranks the VN30 members by 6-month return; VN30-1 found not enough historical evidence that holding the
top 10 beats the basket. This ledger starts the forward record that history cannot give: one immutable line per
screen session, written the evening of that session, never backfilled, and scored once 20 sessions have passed.

    python vn30_screen_forward.py record     # after vn30_screen.py (task BTCVn30Screen)
    python vn30_screen_forward.py evaluate   # matured records only; prints a JSON summary

Ledger: ``<workspace>/data/forward/vn30-screen-top10.jsonl`` (outside the public repos). Each line carries
``prevSha256`` and its own ``sha256`` over the canonical JSON of the other fields, so an edited or deleted line
fails verification. ``record`` refuses a session older than the last recorded one and is a no-op for the same session.

Scoring (per record of session d, on the VNINDEX session calendar): every member is bought at the open of d+1 and
sold at the open of d+21 (T+2 satisfied). ``spreadGross`` = mean return of the top 10 minus the mean return of all
members; ``top10Net`` subtracts one round trip (0.15% + 0.15% fees, 0.1% sell tax) from the top-10 return while the
equal-weight basket is the gross benchmark; ``rankIc`` = Spearman correlation between -rank6 and the member return.
Records overlap (daily records, 20-session windows), so the summary also reports every 20th record, which do not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import vn30_screen as vs

LEDGER_SCHEMA = "vn30-screen-forward/v1"
AI_ROOT = Path(__file__).resolve().parent
DEFAULT_LEDGER = AI_ROOT.parent / "data" / "forward" / "vn30-screen-top10.jsonl"
DEFAULT_SCREEN_DIR = vs.DEFAULT_OUT
GENESIS = "0" * 64
HOLD_SESSIONS = 20
ROUND_TRIP_COST = 0.0015 + 0.0015 + 0.001
RANK_KEY = "6"
TOP_K = 10


class LedgerError(ValueError):
    """The ledger was modified after the fact or a record would break its rules."""


def _hash(entry: Mapping[str, Any]) -> str:
    body = {k: v for k, v in entry.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def load_ledger(path: Path | str = DEFAULT_LEDGER) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows, prev, last_session = [], GENESIS, ""
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        entry = json.loads(line)
        if entry.get("schema") != LEDGER_SCHEMA:
            raise LedgerError(f"line {n}: unexpected schema")
        if entry.get("prevSha256") != prev:
            raise LedgerError(f"line {n}: prevSha256 does not chain to the previous record")
        if _hash(entry) != entry.get("sha256"):
            raise LedgerError(f"line {n}: sha256 mismatch (record {entry.get('session')} was modified)")
        if entry["session"] <= last_session:
            raise LedgerError(f"line {n}: session {entry['session']} is not after {last_session}")
        prev, last_session = entry["sha256"], entry["session"]
        rows.append(entry)
    return rows


def latest_screen(screen_dir: Path | str = DEFAULT_SCREEN_DIR) -> Path:
    files = sorted(p for p in Path(screen_dir).glob("????-??-??.json"))
    if not files:
        raise LedgerError(f"no screen file in {screen_dir}")
    return files[-1]


def record_from_screen(screen_path: Path | str, recorded_at: datetime) -> dict[str, Any]:
    raw = Path(screen_path).read_bytes()
    screen = json.loads(raw.decode("utf-8"))
    if screen.get("schema") != "vn30-screen/v1":
        raise LedgerError(f"{screen_path}: not a vn30-screen/v1 file")
    members = []
    for row in screen["rows"]:
        rank = row.get("ranks", {}).get(RANK_KEY)
        if rank is None:
            continue
        members.append({"symbol": row["symbol"], "rank6": int(rank), "close": float(row["close"])})
    if len(members) < TOP_K * 2:
        raise LedgerError(f"{screen_path}: only {len(members)} ranked members")
    members.sort(key=lambda m: (m["rank6"], m["symbol"]))
    return {
        "schema": LEDGER_SCHEMA,
        "session": screen["session"],
        "recordedAtUtc": recorded_at.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "screenGeneratedAtUtc": screen.get("generatedAtUtc"),
        "screenSha256": hashlib.sha256(raw).hexdigest(),
        "rankBy": f"return-{RANK_KEY}m",
        "top": [m["symbol"] for m in members[:TOP_K]],
        "members": members,
    }


def append_record(record: Mapping[str, Any], path: Path | str = DEFAULT_LEDGER) -> str:
    """Append ``record``; returns "recorded" or "unchanged" (same session already recorded)."""
    path = Path(path)
    rows = load_ledger(path)
    if rows and rows[-1]["session"] == record["session"]:
        return "unchanged"
    if rows and record["session"] < rows[-1]["session"]:
        raise LedgerError(f"refusing to backfill {record['session']}: the ledger is already at {rows[-1]['session']}")
    entry = {**dict(record), "prevSha256": rows[-1]["sha256"] if rows else GENESIS}
    entry["sha256"] = _hash(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    return "recorded"


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    def ranks(v: Sequence[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return None if sx == 0 or sy == 0 else cov / (sx * sy)


Bars = Mapping[str, Mapping[str, tuple[float, float]]]  # symbol -> session -> (open, close)


def score_record(record: Mapping[str, Any], calendar: Sequence[str], bars: Bars) -> dict[str, Any] | None:
    """Score one record, or None while d+21 has no stored session yet."""
    try:
        i = calendar.index(record["session"])
    except ValueError:
        return None
    if i + HOLD_SESSIONS + 1 >= len(calendar):
        return None
    entry_day, exit_day = calendar[i + 1], calendar[i + HOLD_SESSIONS + 1]
    returns, missing = {}, []
    for m in record["members"]:
        b = bars.get(m["symbol"], {})
        if entry_day in b and exit_day in b and b[entry_day][0] > 0:
            returns[m["symbol"]] = b[exit_day][0] / b[entry_day][0] - 1.0
        else:
            missing.append(m["symbol"])
    top = [s for s in record["top"] if s in returns]
    if len(top) < TOP_K // 2 or len(returns) < TOP_K * 2:
        return {"session": record["session"], "entry": entry_day, "exit": exit_day, "status": "insufficient-bars",
                "missing": missing}
    all_mean = sum(returns.values()) / len(returns)
    top_mean = sum(returns[s] for s in top) / len(top)
    ranked = [m for m in record["members"] if m["symbol"] in returns]
    return {
        "session": record["session"], "entry": entry_day, "exit": exit_day, "status": "scored",
        "basketGross": all_mean, "top10Gross": top_mean, "spreadGross": top_mean - all_mean,
        "top10Net": top_mean - ROUND_TRIP_COST, "excessNet": top_mean - ROUND_TRIP_COST - all_mean,
        "rankIc": _spearman([-m["rank6"] for m in ranked], [returns[m["symbol"]] for m in ranked]),
        "missing": missing,
    }


def _mean_t(values: Sequence[float]) -> dict[str, Any]:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "t": None}
    mean = sum(values) / n
    if n < 2:
        return {"n": n, "mean": mean, "t": None}
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    return {"n": n, "mean": mean, "t": None if sd == 0 else mean / (sd / math.sqrt(n))}


def summarize(scored: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ok = [s for s in scored if s.get("status") == "scored"]
    non_overlap = ok[::HOLD_SESSIONS]
    def block(rows):
        return {
            "spreadGross": _mean_t([r["spreadGross"] for r in rows]),
            "excessNet": _mean_t([r["excessNet"] for r in rows]),
            "rankIc": _mean_t([r["rankIc"] for r in rows if r["rankIc"] is not None]),
        }
    return {"scored": len(ok), "pending": len(scored) - len(ok), "all": block(ok),
            "nonOverlapping": block(non_overlap),
            "caveat": "forward paper record of a strategy VN30-1 found not-enough-evidence for; overlapping daily "
                      "records are not independent, judge on nonOverlapping"}


def evaluate(fetch: Callable[[str], Any], ledger: Path | str = DEFAULT_LEDGER) -> dict[str, Any]:
    rows = load_ledger(ledger)
    if not rows:
        return {"records": 0, **summarize([])}
    calendar = sorted(vs.load_bars(fetch, "VNINDEX", None))
    symbols = sorted({m["symbol"] for r in rows for m in r["members"]})
    bars = {s: vs.load_bars(fetch, s, None) for s in symbols}
    scored = [score_record(r, calendar, bars) for r in rows]
    scored = [s if s is not None else {"session": r["session"], "status": "immature"} for s, r in zip(scored, rows)]
    return {"records": len(rows), **summarize(scored), "lastSession": rows[-1]["session"]}


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Forward ledger of the VN30 screen's top 10")
    ap.add_argument("command", choices=("record", "evaluate", "verify"))
    ap.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    ap.add_argument("--screen-dir", type=Path, default=DEFAULT_SCREEN_DIR)
    ap.add_argument("--backend", default=os.getenv("BACKEND_BASE_URL") or vs.DEFAULT_BACKEND)
    args = ap.parse_args(argv)
    try:
        if args.command == "record":
            screen = latest_screen(args.screen_dir)
            status = append_record(record_from_screen(screen, datetime.now(timezone.utc)), args.ledger)
            out = {"status": status, "screen": screen.name, "ledger": str(args.ledger)}
        elif args.command == "verify":
            rows = load_ledger(args.ledger)
            out = {"ok": True, "records": len(rows), "lastSession": rows[-1]["session"] if rows else None}
        else:
            out = evaluate(vs.http_fetch(args.backend), args.ledger)
    except (LedgerError, vs.BackendError, OSError) as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
