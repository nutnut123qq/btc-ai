"""VN30-2 research data: an immutable, hash-verified snapshot of adjusted daily candles.

Layout of one snapshot (written once by `write_snapshot`, never overwritten):

    <root>/<YYYYMMDD>-<sha12>/
        <SYMBOL>.csv       date,open,high,low,close,volume (date = HOSE local session date, LF line ends)
        membership.json    byte copy of contracts/vn30-membership.json used for point-in-time membership
        manifest.json      source, fetch time (UTC), per-symbol first/last session, bar count, sha256 per file
        SNAPSHOT.sha256    "<sha256 of manifest.json>  manifest.json"; <sha12> in the folder name is its prefix

The snapshot sha256 is the sha256 of the manifest bytes; since the manifest lists the sha256 of every other
file, it pins the whole snapshot. `load_snapshot` recomputes every hash, rejects unlisted or missing files,
re-checks the per-symbol counts and dates, and raises `SnapshotError` on any mismatch (fail closed).

Zero-volume bars are kept as the provider sent them (volume 0) and listed in the manifest; they are never
filled in. `bars()` drops them by default and `sessions` (the VNINDEX calendar) excludes them, matching the
VN30-1 convention that a zero-volume bar is not a session. A symbol can carry a documented `excludeBefore`
date (e.g. a reused ticker whose early bars belong to another security): the bars stay in the file and the
hash, but `bars()` leaves them out unless asked.

CLI: python vn30_research_data.py verify <snapshot-dir> [--sha256 <expected>]
"""
from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import io
import json
import os
import re
import secrets
import stat
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping

from vn30_rotation import Periods, load_membership
from vn30_rotation import members_on as _members_on

SCHEMA = "vn30-research-snapshot/v1"
MANIFEST = "manifest.json"
SHA_FILE = "SNAPSHOT.sha256"
MEMBERSHIP_FILE = "membership.json"
CALENDAR_SYMBOL = "VNINDEX"
CSV_FIELDS = ("date", "open", "high", "low", "close", "volume")
SHORT_SHA = 12
_SYMBOL_RE = re.compile(r"^[A-Z0-9]{1,16}$")
_STAMP_RE = re.compile(r"^\d{8}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DIR_RE = re.compile(r"^\d{8}-([0-9a-f]{%d})$" % SHORT_SHA)


class SnapshotError(RuntimeError):
    """The snapshot is missing, altered or inconsistent with its manifest."""


class SnapshotExistsError(SnapshotError):
    """A snapshot with the same folder name already exists; snapshots are never overwritten."""


@dataclass(frozen=True)
class Bar:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def zero_volume(self) -> bool:
        return self.volume == 0


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fmt(x: float) -> str:
    x = float(x)
    return str(int(x)) if x.is_integer() else repr(x)


def render_csv(rows: Iterable[Mapping[str, float | str]]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_FIELDS)
    for r in rows:
        w.writerow([r["date"]] + [_fmt(r[k]) for k in CSV_FIELDS[1:]])
    return buf.getvalue().encode("utf-8")


def _parse_csv(symbol: str, data: bytes) -> tuple[Bar, ...]:
    reader = csv.reader(io.StringIO(data.decode("utf-8")))
    header = next(reader, None)
    if tuple(header or ()) != CSV_FIELDS:
        raise SnapshotError(f"{symbol}: unexpected CSV header {header}")
    out = []
    for row in reader:
        if len(row) != len(CSV_FIELDS):
            raise SnapshotError(f"{symbol}: malformed CSV row {row}")
        out.append(Bar(row[0], *(float(v) for v in row[1:])))
    return tuple(out)


def summarize(symbol: str, rows: list[Mapping[str, float | str]]) -> dict:
    """Validate one symbol's rows and return its manifest statistics (raises ValueError if unusable)."""
    if not rows:
        raise ValueError(f"{symbol}: no bars")
    dates = [str(r["date"]) for r in rows]
    if any(not _DATE_RE.match(d) for d in dates):
        raise ValueError(f"{symbol}: dates must be YYYY-MM-DD")
    if any(a >= b for a, b in zip(dates, dates[1:])):
        raise ValueError(f"{symbol}: dates must be strictly increasing (no duplicates)")
    if any(float(r["close"]) <= 0 for r in rows):
        raise ValueError(f"{symbol}: non-positive close")
    if any(float(r["volume"]) < 0 for r in rows):
        raise ValueError(f"{symbol}: negative volume")
    zero = [str(r["date"]) for r in rows if float(r["volume"]) == 0]
    anomalies = [str(r["date"]) for r in rows
                 if not (0 < float(r["low"]) <= min(float(r["open"]), float(r["close"]))
                         <= max(float(r["open"]), float(r["close"])) <= float(r["high"]))]
    return {"bars": len(rows), "first": dates[0], "last": dates[-1],
            "sessionsWithVolume": len(rows) - len(zero), "zeroVolumeBars": len(zero), "zeroVolumeDates": zero,
            "ohlcAnomalies": len(anomalies), "ohlcAnomalyDates": anomalies}


def _in_force_ranges(periods: Periods, symbol: str) -> list[dict]:
    """Membership spells [from, to) of a symbol; `to` None = still a member in the last period."""
    spells, start = [], None
    for eff, members in periods:
        if symbol in members and start is None:
            start = eff
        elif symbol not in members and start is not None:
            spells.append({"from": start, "toExclusive": eff})
            start = None
    if start is not None:
        spells.append({"from": start, "toExclusive": None})
    return spells


def _dump(obj: dict) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _make_read_only(path: Path) -> None:
    os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def write_snapshot(root: Path, stamp: str, series: Mapping[str, list[Mapping[str, float | str]]],
                   roles: Mapping[str, str], membership_path: Path, meta: Mapping | None = None,
                   missing: Mapping[str, str] | None = None,
                   exclusions: Mapping[str, Mapping[str, str]] | None = None) -> Path:
    """Write a new immutable snapshot under `root` and return its folder (<stamp>-<sha12>).

    `series` maps symbol -> rows (dicts with CSV_FIELDS); `roles` gives each symbol's role (member, index, ...);
    `meta` is copied into the manifest (source, fetch times, notes); `missing` lists symbols that could not be
    fetched with the reason; `exclusions` maps symbol -> {"before": date, "reason": text} for bars that are kept
    but are not the symbol's own history. Refuses to overwrite: raises SnapshotExistsError if the folder already exists.
    """
    if not _STAMP_RE.match(stamp):
        raise ValueError("stamp must be YYYYMMDD")
    if CALENDAR_SYMBOL not in series:
        raise ValueError(f"{CALENDAR_SYMBOL} is required: it defines the session calendar")
    for sym in series:
        if not _SYMBOL_RE.match(sym):
            raise ValueError(f"unsafe symbol name {sym!r}")
        if sym not in roles:
            raise ValueError(f"{sym}: no role given")
    for sym, exc in (exclusions or {}).items():
        if sym not in series or not _DATE_RE.match(str(exc.get("before", ""))) or not exc.get("reason"):
            raise ValueError(f"{sym}: an exclusion needs a fetched symbol, a YYYY-MM-DD 'before' and a reason")
    membership_bytes = Path(membership_path).read_bytes()
    periods = load_membership(Path(membership_path))
    calendar = {str(r["date"]) for r in series[CALENDAR_SYMBOL] if float(r["volume"]) > 0}

    files: dict[str, bytes] = {MEMBERSHIP_FILE: membership_bytes}
    symbols = {}
    for sym in sorted(series):
        rows = list(series[sym])
        entry = {"file": f"{sym}.csv", "role": roles[sym], **summarize(sym, rows)}
        off = [str(r["date"]) for r in rows if float(r["volume"]) > 0 and str(r["date"]) not in calendar]
        exc = (exclusions or {}).get(sym)
        if exc:
            kept = [r for r in rows if str(r["date"]) >= exc["before"]]
            entry.update({"excludeBefore": exc["before"], "exclusionReason": exc["reason"],
                          "barsExcluded": len(rows) - len(kept),
                          "usableFirst": str(kept[0]["date"]) if kept else None})
        entry["datesOutsideCalendar"] = len(off)
        entry["datesOutsideCalendarSample"] = off[:20]
        spells = _in_force_ranges(periods, sym)
        if spells:
            entry["inForce"] = spells
        files[entry["file"]] = render_csv(rows)
        entry["sha256"] = sha256_bytes(files[entry["file"]])
        symbols[sym] = entry

    manifest = {
        **dict(meta or {}),
        "schema": SCHEMA,
        "calendarSymbol": CALENDAR_SYMBOL,
        "csvFields": list(CSV_FIELDS),
        "membership": {"file": MEMBERSHIP_FILE, "sha256": sha256_bytes(membership_bytes), "periods": len(periods),
                       "firstEffectiveDate": periods[0][0], "lastEffectiveDate": periods[-1][0]},
        "symbols": symbols,
        "missing": dict(sorted((missing or {}).items())),
        "files": {name: sha256_bytes(data) for name, data in sorted(files.items())},
    }
    manifest_bytes = _dump(manifest)
    snap_sha = sha256_bytes(manifest_bytes)

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"{stamp}-{snap_sha[:SHORT_SHA]}"
    if final.exists():
        raise SnapshotExistsError(f"snapshot already exists, refusing to overwrite: {final}")
    staging = root / f".staging-{stamp}-{secrets.token_hex(6)}"
    staging.mkdir()
    try:
        for name, data in files.items():
            (staging / name).write_bytes(data)
        (staging / MANIFEST).write_bytes(manifest_bytes)
        (staging / SHA_FILE).write_bytes(f"{snap_sha}  {MANIFEST}\n".encode("ascii"))
        if final.exists():
            raise SnapshotExistsError(f"snapshot already exists, refusing to overwrite: {final}")
        staging.rename(final)
    except BaseException:
        for p in staging.glob("*"):
            p.unlink()
        staging.rmdir()
        raise
    for p in final.iterdir():
        _make_read_only(p)
    return final


class Snapshot:
    """A verified snapshot. Construct with `load_snapshot`."""

    def __init__(self, path: Path, sha256: str, manifest: dict, periods: Periods, data: dict[str, tuple[Bar, ...]]):
        self.path = path
        self.sha256 = sha256
        self.manifest = manifest
        self._periods = periods
        self._data = data
        self.sessions: tuple[str, ...] = tuple(b.date for b in data[CALENDAR_SYMBOL] if not b.zero_volume)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self._data))

    @property
    def periods(self) -> Periods:
        return list(self._periods)

    def role(self, symbol: str) -> str:
        return self.manifest["symbols"][symbol]["role"]

    def bars(self, symbol: str, include_zero_volume: bool = False,
             include_excluded: bool = False) -> tuple[Bar, ...]:
        if symbol not in self._data:
            raise KeyError(f"{symbol} is not in snapshot {self.sha256[:SHORT_SHA]}"
                           + (f" (missing: {self.manifest['missing'][symbol]})"
                              if symbol in self.manifest.get("missing", {}) else ""))
        rows = self._data[symbol]
        before = None if include_excluded else self.manifest["symbols"][symbol].get("excludeBefore")
        return tuple(b for b in rows if (include_zero_volume or not b.zero_volume)
                     and (before is None or b.date >= before))

    def members_on(self, day: str) -> frozenset[str] | None:
        """VN30 constituents in force on `day` (point in time); None before the first period."""
        return _members_on(self._periods, day)

    def coverage(self) -> list[dict]:
        return [{"symbol": s, **{k: e[k] for k in ("role", "bars", "first", "last", "zeroVolumeBars",
                                                   "ohlcAnomalies", "datesOutsideCalendar")},
                 "usableFirst": e.get("usableFirst", e["first"]), "barsExcluded": e.get("barsExcluded", 0)}
                for s, e in sorted(self.manifest["symbols"].items())]


def load_snapshot(path: Path, expected_sha256: str | None = None) -> Snapshot:
    """Verify every hash of the snapshot at `path` and return it; raises SnapshotError on any mismatch.

    `expected_sha256` (full hash or a prefix of at least 12 hex chars), when given, must match the
    snapshot hash; research runs pass the hash they recorded so a different snapshot cannot be swapped in.
    """
    path = Path(path)
    if not path.is_dir():
        raise SnapshotError(f"not a snapshot folder: {path}")
    try:
        manifest_bytes = (path / MANIFEST).read_bytes()
        sha_line = (path / SHA_FILE).read_text(encoding="ascii").split()
    except OSError as exc:
        raise SnapshotError(f"{path}: {exc}") from exc
    snap_sha = sha256_bytes(manifest_bytes)
    if len(sha_line) != 2 or sha_line[1] != MANIFEST or sha_line[0] != snap_sha:
        raise SnapshotError(f"{path}: manifest hash does not match {SHA_FILE}")
    m = _DIR_RE.match(path.name)
    if not m or not snap_sha.startswith(m.group(1)):
        raise SnapshotError(f"{path}: folder name does not end with the snapshot hash prefix {snap_sha[:SHORT_SHA]}")
    if expected_sha256 is not None:
        exp = expected_sha256.strip().lower()
        if len(exp) < SHORT_SHA or not snap_sha.startswith(exp):
            raise SnapshotError(f"{path}: snapshot hash {snap_sha} is not the expected {expected_sha256}")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise SnapshotError(f"{path}: unsupported schema {manifest.get('schema')!r}")
    listed = manifest.get("files") or {}
    present = {p.name for p in path.iterdir()} - {MANIFEST, SHA_FILE}
    if present != set(listed):
        raise SnapshotError(f"{path}: files differ from the manifest (unlisted {sorted(present - set(listed))}, "
                            f"absent {sorted(set(listed) - present)})")
    blobs = {}
    for name, digest in listed.items():
        blobs[name] = (path / name).read_bytes()
        if sha256_bytes(blobs[name]) != digest:
            raise SnapshotError(f"{path}: sha256 mismatch for {name}")
    if manifest["membership"]["sha256"] != listed.get(MEMBERSHIP_FILE):
        raise SnapshotError(f"{path}: membership hash inconsistent")
    try:
        periods = load_membership(path / MEMBERSHIP_FILE)
    except (ValueError, KeyError) as exc:
        raise SnapshotError(f"{path}: invalid membership: {exc}") from exc
    data = {}
    for sym, entry in manifest["symbols"].items():
        if listed.get(entry["file"]) != entry["sha256"] or entry["file"] != f"{sym}.csv":
            raise SnapshotError(f"{path}: {sym} entry inconsistent with the file list")
        bars = _parse_csv(sym, blobs[entry["file"]])
        if (len(bars), bars[0].date if bars else None, bars[-1].date if bars else None) != \
                (entry["bars"], entry["first"], entry["last"]):
            raise SnapshotError(f"{path}: {sym} bars/first/last differ from the manifest")
        data[sym] = bars
    if set(f"{s}.csv" for s in data) | {MEMBERSHIP_FILE} != set(listed):
        raise SnapshotError(f"{path}: file list has entries without a symbol")
    if CALENDAR_SYMBOL not in data:
        raise SnapshotError(f"{path}: {CALENDAR_SYMBOL} calendar missing")
    return Snapshot(path, snap_sha, manifest, periods, data)


def third_thursday(year: int, month: int) -> str:
    weeks = calendar.monthcalendar(year, month)
    day = [w[calendar.THURSDAY] for w in weeks if w[calendar.THURSDAY]][2]
    return date(year, month, day).isoformat()


def verify_front_month_roll(continuous: list[Mapping], contracts: Mapping[tuple[int, int], list[Mapping]]) -> dict:
    """Check a provider's continuous front-month futures series against individual monthly contracts.

    `contracts` maps (year, month) -> that contract's daily rows, for the contracts the provider still serves.
    A contract expires on its last bar when it stopped trading before the continuous series ends, else on the
    third Thursday of its month. The front contract on day t is the earliest month whose expiry is >= t. Days
    whose front contract is not served are counted apart, not compared.
    """
    by_month = {k: {str(r["date"]): r for r in rows} for k, rows in contracts.items() if rows}
    if not by_month or not continuous:
        return {"contracts": {}, "window": None, "daysChecked": 0, "daysMatchingFront": 0,
                "daysFrontNotServed": 0, "mismatches": []}
    series_end = max(str(r["date"]) for r in continuous)
    window_start = min(min(d) for d in by_month.values())

    def expiry(y: int, m: int) -> str:
        days = by_month.get((y, m))
        if days and max(days) < series_end:
            return max(days)
        return third_thursday(y, m)

    checked, not_served, mismatches, last_day = 0, 0, [], None
    for r in continuous:
        day = str(r["date"])
        if day < window_start:
            continue
        y, m = int(day[:4]), int(day[5:7])
        if expiry(y, m) < day:
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        front = by_month.get((y, m), {}).get(day)
        last_day = day
        if front is None:
            not_served += 1
            continue
        checked += 1
        if abs(float(front["close"]) - float(r["close"])) > 1e-9 or abs(float(front["open"]) - float(r["open"])) > 1e-9:
            mismatches.append({"date": day, "front": f"{y:04d}-{m:02d}"})
    return {"contracts": {f"{y:04d}-{m:02d}": {"first": min(d), "last": max(d), "expiry": expiry(y, m)}
                          for (y, m), d in sorted(by_month.items())},
            "window": [window_start, last_day], "daysChecked": checked,
            "daysMatchingFront": checked - len(mismatches), "daysFrontNotServed": not_served,
            "mismatches": mismatches[:20]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("path", type=Path)
    v.add_argument("--sha256")
    args = ap.parse_args(argv)
    try:
        snap = load_snapshot(args.path, args.sha256)
    except SnapshotError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"sha256": snap.sha256, "sessions": len(snap.sessions),
                      "firstSession": snap.sessions[0], "lastSession": snap.sessions[-1],
                      "missing": snap.manifest.get("missing", {}), "coverage": snap.coverage()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
