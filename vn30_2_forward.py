"""VN30-2 forward paper recorder (roadmap 2.6 + 3.3): one immutable record per decision date, all 8 models.

The frozen protocol docs/research/protocols/vn30-2.json declares the forward phase: "from 2026-11-09: one immutable
record per decision date and model (paper only)". The official run (20261106T121249Z) was a no-go, so no model has
a holdout pass; the forward record is still kept for every declared model, because the post-FTSE regime
(2026-09-21 onward) may only be judged on forward data.

    python vn30_2_forward.py record     # daily after the screen (task BTCVn30Screen) and as retries (task
                                        # BTCVn30TwoForward, 19:30 and 08:30); writes only on a decision date
    python vn30_2_forward.py evaluate   # matured records; prints a JSON summary
    python vn30_2_forward.py verify     # hash chain only

Exit codes: 0 ok (recorded, unchanged or nothing to do), 1 error (no record: data gap, splice mismatch, stale or
mismatched membership, protocol mismatch, calendar coverage ended), 3 missed-deadline (a decision date can no longer
be recorded), 4 data-not-ready (a decision date is due but the backend has not ingested its bars yet; rerun before
the deadline, the retry task does).

Ledger: ``<workspace>/data/forward/vn30-2.jsonl`` (outside the public repos). The rules are those of
``vn30_screen_forward.py``: every line carries ``prevSha256`` and its own ``sha256`` over the canonical JSON of the
other fields; a decision date older than the last recorded one is refused (no backfill); the same date again is a
no-op.

Schedule: the protocol grid, every 20 sessions from ``firstDecision`` (first session with >= 260 prior sessions for
>= 20 members, ``vn30_2_run.decision_indices``), continued on the live session calendar. ``record`` writes only when
the latest finalized session is a grid date >= 2026-11-09, and only before the next session opens (09:00 ICT of the
next HOSE trading day); a grid date that passed unrecorded is never written later and is listed as missed in the
next record. Late ingest: the session expected by the HOSE calendar at run time (today from 16:00 ICT on a trading
day, else the previous trading day) is compared with the backend's latest VNINDEX bar; when a grid date is among the
sessions the backend has not ingested yet, the run returns ``data-not-ready`` (exit 4) instead of a silent no-op, and
so does a decision date on which any in-force member lacks its bar of that session (the backend ingests per symbol,
so a run during the ingest can see some members without bar d). The retry task (``scripts/run_vn30_2_forward.ps1``,
installed by ``scripts/install_vn30_2_forward_task.ps1``) reruns ``record`` at 19:30 and 08:30, inside the deadline;
repeated runs are idempotent.

Scores (identical to the protocol's walk-forward): the market is cut at the close of decision d. Features are
computed point in time and rank-normalized (``vn30_2_run.build_panel``); every model is the protocol's
``vn30_2_run.FIT_PREDICT`` fitted on the yearly-refit training set of d's calendar year: all grid decisions s with
s + purge (21) + embargo (20) < f, where f is the first grid decision of that year (``yearly_folds`` of the official
run gives the same set). The record keeps the raw features, every model's score, ranking and selection (top 10;
a holding of the previous record of the same model is kept while its rank <= 15; B-ew = every eligible member).

Data source switch (history -> live):
* History = the frozen, hash-verified research snapshot 20261106-9afb8020ec16 (VCI adjusted bars through
  2026-11-06, the snapshot's last session L), the same file the official run read.
* Live = the local backend ``GET /api/equities/{SYM}/candles?timeframe=1d`` (same VCI adjusted series; on 2026-11-06
  its closes equalled the snapshot's on every common bar for MBB, FPT, VIC, SSB and VNINDEX, 1404-2000 bars each) for
  every session after L. Live sessions are the
  backend VNINDEX bars with volume > 0 dated after L; they must be exactly the HOSE trading days of the backend's
  checked-in calendar (``btc-be/Data/Calendars/vn-hose-non-trading-days.json``), so a missing or extra live session
  fails closed instead of shifting the 20-session grid. A bar dated today counts only from 16:00 ICT (the
  snapshot's lastSessionRule). Hard stop: the calendar file covers a fixed range (``coverage.toDate``, 2027-12-31
  in btc-be as of 2026-11; 2027 still provisional), so every ``record`` and ``evaluate`` fails once a live session
  or the expected session is after it, until btc-be extends the file; from 60 days before that date every output
  carries a warning.
* Completeness: on a decision date every in-force member must have a bar dated d (a missing one means the ingest
  has not finished: ``data-not-ready``). Earlier live sessions without a bar (e.g. a suspension) are allowed, as in
  the snapshot, and listed in the record (``data.liveGaps``).
* Splice: per symbol, the backend and the snapshot closes are compared on the common sessions of the last 60
  snapshot sessions. Their ratio must be constant (spread <= 0.1%); the snapshot history is then multiplied by that
  ratio (1 while the provider has not re-adjusted; < 1 after a later dividend re-adjusted the backend's history), so
  returns inside the snapshot are unchanged and the live bars continue the same adjusted level. Volumes are kept.
* Membership = ``contracts/vn30-membership.json`` of this checkout; its periods must start with exactly the
  snapshot's periods, and the latest period must be at most 190 days old at the decision date (a warning from 170
  days: a semi-annual review is due). On a decision date it is also cross-checked against the backend catalog
  (``GET /api/equities``): every member in force at d must be in the catalog, and a catalog equity that no
  membership period lists (a new constituent, e.g. an ad-hoc change like 2026-05-13) means the membership file is
  out of date: no record. Catalog equities that are former members are allowed (reported).
* A symbol the backend does not serve (HTTP 404, e.g. a leaver) keeps its snapshot bars only; a member in force at
  the decision date that the backend does not serve is an error (no record rather than stale features).
* Provenance: every record carries the sha256 of each imported research module, the git HEAD of this checkout and
  the modules that differ from HEAD (``code``).

Evaluation (``evaluate``): a record of decision d is matured when the open of d+21 exists. Member return =
open(d+21) / open(d+1) - 1; target = return minus the equal-weight mean return of every member in force at d
(protocol target); rank IC = Spearman(score, target) over eligible members with a target (>= 10 names). Grid
records are 20 sessions apart, so matured records do not overlap (``vn30_panel_eval.non_overlapping`` is applied
anyway). Survivorship: every member in force at d of a matured record must have both opens; a leaver the backend no
longer serves (404 after a review) is read from ``--leaver-bars DIR`` (``<SYM>.csv`` written by
``scripts/fetch_vn30_candles.py`` from the public VCI source into ``<workspace>/data``, spliced like live bars);
otherwise evaluate fails, unless ``--allow-missing-members`` (then the affected periods are counted in the output).
Portfolio = the protocol portfolio: ``vn_portfolio_backtest.run_backtest`` with ``PortfolioRules.from_protocol``
(as ``vn30_2_run.portfolio_rules``: 100-share lots, limit-fill proxy with the miss carried over, T+2, costs on every
trade including equal-weight re-balancing of kept names and B-ew drift) driven by the recorded scores, run over each
run of contiguous matured records; the run ends with a liquidation at the open of d+21 of its last record, and a
period return is NAV(entry open of a record) -> NAV(next entry open, or the close after the final liquidation).
Net excess = period net return of the model minus that of B-ew; annualized = CAGR difference with 252/20 periods
per year.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from bisect import bisect_right
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

import trial_registry
import vn30_2_run as run
import vn30_panel_eval as ev
from vn30_rotation import Periods, load_membership, members_on
from vn_portfolio_backtest import PortfolioRules, first_decision_index, rank_members, run_backtest, select_targets

SCHEMA = "vn30-2-forward/v1"
REPO = Path(__file__).resolve().parent
WORKSPACE = REPO.parent
DEFAULT_LEDGER = WORKSPACE / "data" / "forward" / "vn30-2.jsonl"
DEFAULT_MEMBERSHIP = REPO / "contracts" / "vn30-membership.json"
CALENDAR_CANDIDATES = (WORKSPACE / "btc-be" / "Data" / "Calendars" / "vn-hose-non-trading-days.json",
                       WORKSPACE / "backend" / "Data" / "Calendars" / "vn-hose-non-trading-days.json")
DEFAULT_BACKEND = "http://127.0.0.1:5197"  # never "localhost" (~2 s per request on Windows)
DEFAULT_RUN_ID = "20261106T121249Z"
GENESIS = "0" * 64
FORWARD_START = "2026-11-09"  # protocol validation.forward
CALENDAR_SYMBOL = "VNINDEX"
REBALANCE = ev.REBALANCE_SESSIONS
LABEL = run.LABEL_SESSIONS
TOP_K = 10
KEEP_RANK = 15
MIN_HISTORY = 260
SPLICE_OVERLAP_SESSIONS = 60
SPLICE_TOLERANCE = 1e-3
STALE_MEMBERSHIP_DAYS = 190
REVIEW_DUE_DAYS = 170  # warning: a semi-annual review (about 182 days apart) is due
COVERAGE_WARN_DAYS = 60
EXIT_MISSED_DEADLINE = 3
EXIT_DATA_NOT_READY = 4
CODE_MODULES = ("vn30_2_forward.py", "vn30_2_run.py", "vn30_panel_eval.py", "vn_portfolio_backtest.py",
                "vn30_research_data.py", "vn_raw_prices.py", "vn30_rotation.py", "vn_execution_model.py",
                "trial_registry.py")
FINAL_BAR_HOUR_ICT = 16
NEXT_OPEN_HOUR_ICT = 9
CANDLE_LIMIT_MAX = 2000
USER_AGENT = "btc-ai-vn30-2-forward/1.0 (local research recorder)"
ICT = timezone(timedelta(hours=7))
PERIODS_PER_YEAR = run.TRADING_DAYS / REBALANCE

Bar = tuple  # (open, high, low, close, volume)
Fetch = Callable[[str], Any]


class ForwardError(RuntimeError):
    """Inputs do not allow a faithful record (data gap, splice mismatch, stale membership, protocol mismatch)."""


class LedgerError(ValueError):
    """The ledger was modified after the fact or a record would break its rules."""


class NotServed(LookupError):
    """The backend does not serve this symbol (HTTP 404)."""


# --------------------------------------------------------------------------------------------- ledger


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash(entry: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical({k: v for k, v in entry.items() if k != "sha256"})).hexdigest()


def load_ledger(path: Path | str = DEFAULT_LEDGER) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows, prev, last = [], GENESIS, ""
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        entry = json.loads(line)
        if entry.get("schema") != SCHEMA:
            raise LedgerError(f"line {n}: unexpected schema")
        if entry.get("prevSha256") != prev:
            raise LedgerError(f"line {n}: prevSha256 does not chain to the previous record")
        if _hash(entry) != entry.get("sha256"):
            raise LedgerError(f"line {n}: sha256 mismatch (record {entry.get('decisionDate')} was modified)")
        if entry["decisionDate"] <= last:
            raise LedgerError(f"line {n}: decision {entry['decisionDate']} is not after {last}")
        prev, last = entry["sha256"], entry["decisionDate"]
        rows.append(entry)
    return rows


def append_record(record: Mapping[str, Any], path: Path | str = DEFAULT_LEDGER) -> str:
    """Append ``record``; returns "recorded" or "unchanged" (same decision date already recorded)."""
    path = Path(path)
    rows = load_ledger(path)
    if rows and rows[-1]["decisionDate"] == record["decisionDate"]:
        return "unchanged"
    if rows and record["decisionDate"] < rows[-1]["decisionDate"]:
        raise LedgerError(f"refusing to backfill {record['decisionDate']}: the ledger is already at "
                          f"{rows[-1]['decisionDate']}")
    entry = {**dict(record), "schema": SCHEMA, "prevSha256": rows[-1]["sha256"] if rows else GENESIS}
    entry["sha256"] = _hash(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    return "recorded"


# --------------------------------------------------------------------------------------------- live data


def http_fetch(backend: str, timeout: float = 30.0, sleep: Callable[[float], None] = time.sleep,
               pause: float = 0.25) -> Fetch:
    """GET JSON from the local backend; 404 -> NotServed; 429 -> wait 60 s and retry once."""
    def fetch(path: str) -> Any:
        request = urllib.request.Request(backend.rstrip("/") + path, headers={"User-Agent": USER_AGENT})
        for attempt in (1, 2):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                sleep(pause)
                return payload
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    raise NotServed(path) from exc
                if exc.code == 429 and attempt == 1:
                    sleep(60.0)
                    continue
                raise
        raise AssertionError("unreachable")
    return fetch


def session_of(open_time_ms: int) -> str:
    return datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc).astimezone(ICT).date().isoformat()


def fetch_bars(fetch: Fetch, symbol: str, limit: int) -> dict[str, Bar]:
    """Session -> (open, high, low, close, volume) from the backend, zero-volume bars dropped."""
    payload = fetch(f"/api/equities/{symbol}/candles?timeframe=1d&limit={int(limit)}")
    try:
        return {session_of(int(c["openTimeMs"])): (float(c["open"]), float(c["high"]), float(c["low"]),
                                                   float(c["close"]), float(c["volume"]))
                for c in payload if float(c["volume"]) > 0}
    except (KeyError, TypeError, ValueError) as exc:
        raise ForwardError(f"{symbol} candles: unexpected payload ({type(exc).__name__}: {exc})") from exc


def finalized(bars: Mapping[str, Bar], now_utc: datetime) -> dict[str, Bar]:
    """Drop a bar dated today (ICT) before 16:00 ICT, and anything dated after today."""
    local = now_utc.astimezone(ICT)
    today = local.date().isoformat()
    return {d: b for d, b in bars.items()
            if d < today or (d == today and local.hour >= FINAL_BAR_HOUR_ICT)}


def load_non_trading_days(path: Path | str | None = None) -> tuple[frozenset[str], str]:
    """(non-trading dates, coverage end) of the backend's checked-in HOSE calendar."""
    candidates = [Path(path)] if path is not None else list(CALENDAR_CANDIDATES)
    for p in candidates:
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("venue") != "HOSE":
                raise ForwardError(f"{p}: not a HOSE calendar")
            return frozenset(d["date"] for d in data["days"]), str(data["coverage"]["toDate"])
    raise ForwardError(f"HOSE calendar file not found ({', '.join(str(p) for p in candidates)})")


def trading_days(after: str, through: str, non_trading: frozenset[str], coverage_to: str) -> list[str]:
    """HOSE trading days in (after, through]: weekdays that are not listed non-trading days."""
    if through > coverage_to:
        raise ForwardError(f"HOSE calendar covers up to {coverage_to}, not {through}")
    out, day, end = [], date.fromisoformat(after) + timedelta(days=1), date.fromisoformat(through)
    while day <= end:
        iso = day.isoformat()
        if day.weekday() < 5 and iso not in non_trading:
            out.append(iso)
        day += timedelta(days=1)
    return out


def next_trading_day(after: str, non_trading: frozenset[str]) -> str:
    day = date.fromisoformat(after) + timedelta(days=1)
    while day.weekday() >= 5 or day.isoformat() in non_trading:
        day += timedelta(days=1)
    return day.isoformat()


def is_trading_day(day: str, non_trading: frozenset[str]) -> bool:
    return date.fromisoformat(day).weekday() < 5 and day not in non_trading


def expected_session(now_utc: datetime, non_trading: frozenset[str]) -> str:
    """The latest session the HOSE calendar says is finalized at ``now``: today from 16:00 ICT on a trading day,
    otherwise the previous trading day."""
    local = now_utc.astimezone(ICT)
    today = local.date().isoformat()
    if is_trading_day(today, non_trading) and local.hour >= FINAL_BAR_HOUR_ICT:
        return today
    day = local.date() - timedelta(days=1)
    while not is_trading_day(day.isoformat(), non_trading):
        day -= timedelta(days=1)
    return day.isoformat()


def coverage_warnings(now_utc: datetime, coverage_to: str) -> list[str]:
    left = (date.fromisoformat(coverage_to) - now_utc.astimezone(ICT).date()).days
    if left > COVERAGE_WARN_DAYS:
        return []
    return [f"the HOSE calendar file covers up to {coverage_to} ({left} days left): every record and evaluate fails "
            "after that date until btc-be/Data/Calendars/vn-hose-non-trading-days.json is extended"]


def live_calendar(history_calendar: Sequence[str], index_bars: Mapping[str, Bar], non_trading: frozenset[str],
                  coverage_to: str) -> list[str]:
    """Live sessions after the snapshot's last session, validated against the history overlap and the HOSE calendar."""
    last = history_calendar[-1]
    overlap = list(history_calendar[-SPLICE_OVERLAP_SESSIONS:])
    have = sorted(d for d in index_bars if overlap[0] <= d <= last)
    if have != overlap:
        missing = sorted(set(overlap) - set(have))
        extra = sorted(set(have) - set(overlap))
        raise ForwardError(f"backend {CALENDAR_SYMBOL} disagrees with the snapshot calendar on the overlap "
                           f"(missing {missing[:5]}, extra {extra[:5]})")
    live = sorted(d for d in index_bars if d > last)
    if not live:
        return []
    expected = trading_days(last, live[-1], non_trading, coverage_to)
    if live != expected:
        raise ForwardError(f"live sessions differ from the HOSE calendar (missing {sorted(set(expected) - set(live))[:5]}"
                           f", extra {sorted(set(live) - set(expected))[:5]})")
    return live


def splice(history: Mapping[str, Bar], live: Mapping[str, Bar], history_calendar: Sequence[str],
           live_sessions: Sequence[str], symbol: str = "") -> tuple[dict[str, Bar], float]:
    """History bars scaled to the live level, followed by the live bars of the live sessions."""
    last = history_calendar[-1]
    overlap = [d for d in history_calendar[-SPLICE_OVERLAP_SESSIONS:] if d in history and d in live]
    if not overlap:
        raise ForwardError(f"{symbol}: no common session with the backend in the last {SPLICE_OVERLAP_SESSIONS} "
                           "snapshot sessions, cannot splice")
    ratios = np.array([live[d][3] / history[d][3] for d in overlap], dtype=float)
    if not np.all(np.isfinite(ratios)) or ratios.min() <= 0 or ratios.max() / ratios.min() - 1 > SPLICE_TOLERANCE:
        raise ForwardError(f"{symbol}: backend/snapshot close ratio not constant on the overlap "
                           f"({ratios.min():.6f}..{ratios.max():.6f}): the series differ beyond a re-adjustment")
    ratio = float(np.median(ratios))
    if abs(ratio - 1.0) < 1e-9:
        ratio = 1.0
    out = {d: (b[0] * ratio, b[1] * ratio, b[2] * ratio, b[3] * ratio, b[4]) if ratio != 1.0 else tuple(b)
           for d, b in history.items() if d <= last}
    live_set = set(live_sessions)
    out.update({d: tuple(b) for d, b in live.items() if d in live_set})
    return out, ratio


# --------------------------------------------------------------------------------------------- inputs


def check_membership(history_periods: Periods, periods: Periods) -> None:
    if list(periods[:len(history_periods)]) != list(history_periods):
        raise ForwardError("contracts/vn30-membership.json does not start with the snapshot's membership periods")


def symbols_in_force_between(periods: Periods, start: str, end: str) -> set[str]:
    """Every constituent of a period whose in-force range intersects [start, end]."""
    out: set[str] = set()
    for k, (eff, members) in enumerate(periods):
        until = periods[k + 1][0] if k + 1 < len(periods) else "9999-12-31"
        if eff <= end and until > start:
            out |= set(members)
    return out


def check_protocol(protocol_path: Path = run.PROTOCOL_PATH,
                   registry_path: Path = trial_registry.REGISTRY_PATH) -> dict[str, Any]:
    """The frozen protocol sha256 must equal the registered ``hose-vn30-2`` specSha256; models/features unchanged."""
    sha = ev.protocol_sha256(protocol_path)
    spec = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    registry = trial_registry.load(registry_path)
    entries = [e for e in registry.entries if e["id"] == run.REGISTRY_ENTRY_ID]
    if not entries or entries[0].get("specSha256") != sha:
        raise ForwardError(f"protocol sha256 {sha} is not the registered {run.REGISTRY_ENTRY_ID} spec")
    if tuple(spec["features"]["list"]) != run.FEATURES or tuple(m["id"] for m in spec["models"]["list"]) != run.MODELS:
        raise ForwardError("protocol features/models differ from vn30_2_run")
    if FORWARD_START not in spec["validation"]["forward"]:
        raise ForwardError("protocol forward clause changed")
    if spec["decisionSchedule"]["rebalanceEverySessions"] != REBALANCE:
        raise ForwardError("protocol rebalance differs")
    return {"protocolSha256": sha, "evaluatorSha256": ev.evaluator_sha256(),
            "costs": dict(spec["portfolio"]["costs"]), "registryHead": registry.head}


def _sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes() -> dict[str, str]:
    return {m: _sha_file(REPO / m) for m in CODE_MODULES if (REPO / m).is_file()}


def code_provenance(repo: Path = REPO) -> dict[str, Any]:
    """Module sha256s, the git HEAD of the checkout and the modules that differ from it (None when git fails)."""
    out: dict[str, Any] = {"modules": code_hashes(), "gitHead": None, "modulesDifferFromHead": None}
    try:
        def git(*args: str) -> str:
            return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True, timeout=30,
                                  text=True).stdout
        out["gitHead"] = git("rev-parse", "HEAD").strip()
        status = git("status", "--porcelain", "--untracked-files=all", "--", *CODE_MODULES)
        out["modulesDifferFromHead"] = sorted(line[3:].strip() for line in status.splitlines() if line.strip())
    except (OSError, subprocess.SubprocessError) as exc:
        out["gitError"] = f"{type(exc).__name__}: {exc}"
    return out


def check_catalog(catalog: Any, periods: Periods, in_force: Iterable[str], day: str) -> dict[str, Any]:
    """Cross-check the membership file against the backend catalog (``GET /api/equities``) at decision ``day``."""
    try:
        items = catalog["items"]
        symbols = {str(it["symbol"]) for it in items if it.get("assetClass", "equity") == "equity"}
    except (KeyError, TypeError) as exc:
        raise ForwardError(f"/api/equities: unexpected payload ({type(exc).__name__}: {exc})") from exc
    symbols.discard(CALENDAR_SYMBOL)
    known = {s for _, members in periods for s in members}
    in_force = set(in_force)
    absent = sorted(in_force - symbols)
    if absent:
        raise ForwardError(f"members in force at {day} per contracts/vn30-membership.json are not in the backend "
                           f"catalog: {absent} (update the membership file or the backend catalog)")
    unknown = sorted(symbols - known)
    if unknown:
        raise ForwardError(f"the backend catalog lists {unknown}, which no period of contracts/vn30-membership.json "
                           f"contains: the membership file is probably missing a newer period (update it)")
    return {"catalogNotInForce": sorted(symbols - in_force)}


# --------------------------------------------------------------------------------------------- market


def build_live_market(history: run.Market, live: Mapping[str, Mapping[str, Bar] | None],
                      live_sessions: Sequence[str], periods: Periods) -> tuple[run.Market, dict[str, Any]]:
    """History + live sessions. ``live[sym]`` None = not served by the backend (snapshot bars only)."""
    bars: dict[str, dict[str, Bar]] = {}
    ratios, not_served, digest_src = {}, [], {}
    live_set, last = set(live_sessions), history.calendar[-1]
    gaps: dict[str, list[str]] = {}
    in_force_on = {d: members_on(periods, d) or frozenset() for d in live_sessions}
    for sym in sorted(set(history.bars) | set(live)):
        hist = history.bars.get(sym, {})
        lv = live.get(sym)
        if lv is None:
            bars[sym] = {d: tuple(b) for d, b in hist.items()}
            if sym in live:
                not_served.append(sym)
            continue
        if hist:
            bars[sym], ratio = splice(hist, lv, history.calendar, live_sessions, sym)
            if ratio != 1.0:
                ratios[sym] = ratio
        else:  # a new member without snapshot history (bars of the live sessions only would be too short)
            bars[sym] = {d: tuple(b) for d, b in lv.items() if d <= last or d in live_set}
        digest_src[sym] = [[d, *bars[sym][d]] for d in live_sessions if d in bars[sym]]
        missing = [d for d in live_sessions if sym in in_force_on[d] and d not in bars[sym]]
        if missing:
            gaps[sym] = missing
    calendar = tuple(history.calendar) + tuple(live_sessions)
    provenance = {
        "historyLastSession": history.calendar[-1],
        "liveSource": "backend GET /api/equities/{SYM}/candles?timeframe=1d (VCI adjusted)",
        "liveSessions": {"count": len(live_sessions), "first": live_sessions[0] if live_sessions else None,
                         "last": live_sessions[-1] if live_sessions else None},
        "spliceRatios": ratios, "notServed": not_served, "liveGaps": gaps,
        "liveBarsSha256": hashlib.sha256(_canonical({"calendar": list(live_sessions), "bars": digest_src})).hexdigest(),
    }
    return run.Market(calendar, bars, periods, {}), provenance


def grid_first(history: run.Market) -> int:
    first = first_decision_index(history.calendar, history.bars, history.periods, MIN_HISTORY)
    if first is None:
        raise ForwardError("no session satisfies the first-decision rule")
    return first


def fit_plan(calendar: Sequence[str], first: int, i: int) -> dict[str, Any]:
    """Yearly-refit training set for decision index ``i`` (as ``vn30_panel_eval.yearly_folds``)."""
    if i < first or (i - first) % REBALANCE:
        raise ValueError(f"session {i} is not a grid decision")
    grid = list(range(first, i + 1, REBALANCE))
    year = calendar[i][:4]
    fit_at = next(g for g in grid if calendar[g][:4] == year)
    candidates = np.array([g for g in grid if g < fit_at], dtype=np.int64)
    keep = ev.purge_embargo_mask(candidates, fit_at, i) if len(candidates) else np.zeros(0, dtype=bool)
    return {"year": int(year), "fitAt": fit_at, "train": [int(x) for x in candidates[keep]],
            "purged": [int(x) for x in candidates[~keep]]}


def decision_scores(market: run.Market, first: int, i: int) -> dict[str, Any]:
    """Point-in-time features and the 8 model scores at decision index ``i`` (market cut at its close)."""
    cut = market.truncate(market.calendar[i])
    plan = fit_plan(cut.calendar, first, i)
    panel = run.build_panel(cut, plan["train"] + [i], min_history=MIN_HISTORY)
    test = panel[panel["session"] == i]
    if test.empty:
        raise ForwardError(f"{cut.calendar[i]}: no eligible member")
    grid = list(range(first, i + 1, REBALANCE))
    fold = ev.forward_fold(grid, cut.calendar)  # sealed: the label of i lies past the cut calendar
    if (fold.fit_session, list(fold.train_sessions), list(fold.purged_sessions)) != (
            plan["fitAt"], plan["train"], plan["purged"]):
        raise ForwardError(f"{cut.calendar[i]}: fit plan differs from the evaluator's forward fold")
    scores = {}
    for m in run.MODELS:
        if m.startswith("M-") and not plan["train"]:
            raise ForwardError(f"{m}: no training decision before {cut.calendar[plan['fitAt']]}")
        s = ev.walk_forward_scores(panel, [fold], run.FIT_PREDICT[m])
        scores[m] = {sym: float(s.loc[idx]) for idx, sym in zip(test.index, test["symbol"])}
    train_rows = panel[panel["session"].isin(plan["train"])]
    features = {row["symbol"]: {f: (None if not math.isfinite(row[f]) else float(row[f])) for f in run.FEATURES}
                for _, row in test.iterrows()}
    return {
        "eligible": sorted(test["symbol"]),
        "features": features,
        "scores": scores,
        "fit": {"year": plan["year"], "fitDecisionDate": cut.calendar[plan["fitAt"]],
                "trainDecisions": len(plan["train"]),
                "trainFrom": cut.calendar[plan["train"][0]] if plan["train"] else None,
                "trainTo": cut.calendar[plan["train"][-1]] if plan["train"] else None,
                "trainRowsWithTarget": int(np.isfinite(train_rows["target"].to_numpy(dtype=float)).sum()),
                "purgedDecisions": len(plan["purged"])},
    }


def select(model: str, scores: Mapping[str, float], eligible: Iterable[str], held: set[str]) -> list[str]:
    ranked = rank_members(scores, frozenset(eligible))
    if model == "B-ew":
        return select_targets(ranked, held, None, None)
    return select_targets(ranked, held, TOP_K, KEEP_RANK)


def build_record(market: run.Market, first: int, i: int, previous: Mapping[str, Any] | None, *,
                 recorded_at: datetime, meta: Mapping[str, Any], missed: Sequence[str] = ()) -> dict[str, Any]:
    cal = market.calendar
    day = cal[i]
    res = decision_scores(market, first, i)
    contiguous = (previous is not None and i - REBALANCE >= 0 and previous.get("decisionDate") == cal[i - REBALANCE])
    models = {}
    for m in run.MODELS:
        held = set(previous["models"][m]["selection"]) if contiguous else set()
        sc = res["scores"][m]
        models[m] = {"scores": sc, "ranking": rank_members(sc, frozenset(res["eligible"])),
                     "selection": select(m, sc, res["eligible"], held)}
    return {
        "schema": SCHEMA,
        "protocolId": run.PROTOCOL_ID,
        "decisionDate": day,
        "decisionSession": i,
        "gridNumber": (i - first) // REBALANCE,
        "firstDecisionDate": cal[first],
        "recordedAtUtc": recorded_at.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "execution": "enter at the open of the next session, exit at the open of the 21st session after (paper)",
        "previousRecordContiguous": bool(contiguous),
        "missedDecisions": list(missed),
        "inForce": sorted(members_on(market.periods, day) or ()),
        "eligible": res["eligible"],
        "fit": res["fit"],
        "features": res["features"],
        "models": models,
        **dict(meta),
    }


# --------------------------------------------------------------------------------------------- record


def _limit_for(history_last: str, now_utc: datetime) -> int:
    days = (now_utc.astimezone(ICT).date() - date.fromisoformat(history_last)).days
    limit = int(days * 5 / 7) + SPLICE_OVERLAP_SESSIONS + 30
    if limit > CANDLE_LIMIT_MAX:
        raise ForwardError(f"the snapshot ends {history_last}: more than {CANDLE_LIMIT_MAX} bars would be needed")
    return limit


def load_live(fetch: Fetch, symbols: Iterable[str], limit: int, now_utc: datetime) -> dict[str, dict | None]:
    out: dict[str, dict | None] = {}
    for sym in sorted(set(symbols)):
        try:
            out[sym] = finalized(fetch_bars(fetch, sym, limit), now_utc)
        except NotServed:
            out[sym] = None
    return out


def _deadline(day: str, non_trading: frozenset[str]) -> datetime:
    """A decision of ``day`` must be recorded before the next session opens (09:00 ICT)."""
    return datetime.fromisoformat(next_trading_day(day, non_trading)).replace(hour=NEXT_OPEN_HOUR_ICT, tzinfo=ICT)


def membership_warnings(periods: Periods, day: str) -> list[str]:
    latest_period = max((eff for eff, _ in periods if eff <= day), default=None)
    if latest_period is None:
        return []
    age = (date.fromisoformat(day) - date.fromisoformat(latest_period)).days
    if age <= REVIEW_DUE_DAYS:
        return []
    return [f"the latest membership period ({latest_period}) is {age} days old at {day}: a VN30 review is due; "
            f"update contracts/vn30-membership.json (records stop after {STALE_MEMBERSHIP_DAYS} days)"]


def record(history: run.Market, fetch: Fetch, *, now_utc: datetime, periods: Periods,
           non_trading: frozenset[str], coverage_to: str, meta: Mapping[str, Any],
           ledger: Path | str = DEFAULT_LEDGER, forward_start: str = FORWARD_START) -> dict[str, Any]:
    """One daily call: append the record of the latest session when it is a forward grid decision.

    Statuses: recorded, unchanged, not-a-decision-date, before-forward-start (exit 0); missed-deadline (exit 3);
    data-not-ready (exit 4: a grid date is due but the backend lacks its VNINDEX bar or an in-force member's bar)."""
    check_membership(history.periods, periods)
    limit = _limit_for(history.calendar[-1], now_utc)
    try:
        index_bars = finalized(fetch_bars(fetch, CALENDAR_SYMBOL, limit), now_utc)
    except NotServed as exc:
        raise ForwardError(f"backend does not serve {CALENDAR_SYMBOL}") from exc
    live_sessions = live_calendar(history.calendar, index_bars, non_trading, coverage_to)
    cal = list(history.calendar) + live_sessions
    first = grid_first(history)
    i = len(cal) - 1
    latest = cal[i]
    rows = load_ledger(ledger)
    last = rows[-1] if rows else None
    expected = expected_session(now_utc, non_trading)
    warnings = coverage_warnings(now_utc, coverage_to) + membership_warnings(periods, max(latest, expected))
    out: dict[str, Any] = {"latestSession": latest, "expectedSession": expected, "ledger": str(ledger)}
    if warnings:
        out["warnings"] = warnings
    if expected > latest:  # the backend has not ingested every session the calendar says is finalized
        pending = trading_days(latest, expected, non_trading, coverage_to)
        out["backendLagSessions"] = pending
        due = [d for k, d in enumerate(pending) if (i + 1 + k - first) % REBALANCE == 0 and d >= forward_start
               and (last is None or d > last["decisionDate"])]
        for d in due:
            if now_utc < _deadline(d, non_trading):
                return {**out, "status": "data-not-ready", "decisionDate": d,
                        "deadlineIct": _deadline(d, non_trading).isoformat(),
                        "reason": f"backend {CALENDAR_SYMBOL} has no bar for {pending}: rerun before the deadline"}
        if due:
            return {**out, "status": "missed-deadline", "decisionDate": due[-1],
                    "deadlineIct": _deadline(due[-1], non_trading).isoformat(),
                    "reason": f"backend {CALENDAR_SYMBOL} still has no bar for {due}"}
    offset = (i - first) % REBALANCE
    if offset:
        return {**out, "status": "not-a-decision-date", "lastDecision": cal[i - offset],
                "sessionsToNextDecision": REBALANCE - offset}
    if latest < forward_start:
        return {**out, "status": "before-forward-start", "forwardStart": forward_start}
    if last is not None and last["decisionDate"] == latest:
        return {**out, "status": "unchanged"}
    if last is not None and last["decisionDate"] > latest:
        raise LedgerError(f"ledger is at {last['decisionDate']}, after the latest session {latest}")
    deadline = _deadline(latest, non_trading)
    if now_utc >= deadline:
        return {**out, "status": "missed-deadline", "decisionDate": latest, "deadlineIct": deadline.isoformat()}
    latest_period = max((eff for eff, _ in periods if eff <= latest), default=None)
    age = None if latest_period is None else (date.fromisoformat(latest) - date.fromisoformat(latest_period)).days
    if age is None or age > STALE_MEMBERSHIP_DAYS:
        raise ForwardError(f"membership period {latest_period} is older than {STALE_MEMBERSHIP_DAYS} days at {latest}: "
                           "update contracts/vn30-membership.json")
    in_force = members_on(periods, latest) or frozenset()
    try:
        catalog = check_catalog(fetch("/api/equities"), periods, in_force, latest)
    except NotServed as exc:
        raise ForwardError("backend does not serve /api/equities") from exc
    grid = range(first, i, REBALANCE)
    missed = [cal[g] for g in grid if cal[g] >= forward_start and (last is None or cal[g] > last["decisionDate"])]
    live = load_live(fetch, symbols_in_force_between(periods, history.calendar[-1], latest), limit, now_utc)
    unserved = sorted(s for s in in_force if live.get(s) is None)
    if unserved and live_sessions:
        raise ForwardError(f"members in force at {latest} not served by the backend: {unserved}")
    incomplete = sorted(s for s in in_force if live.get(s) is not None and latest not in live[s])
    if incomplete and live_sessions:
        return {**out, "status": "data-not-ready", "decisionDate": latest, "deadlineIct": deadline.isoformat(),
                "membersWithoutDecisionBar": incomplete,
                "reason": f"{len(incomplete)} in-force member(s) have no bar dated {latest} yet: rerun before the "
                          "deadline"}
    market, provenance = build_live_market(history, live, live_sessions, periods)
    rec = build_record(market, first, i, last, recorded_at=now_utc,
                       meta={**dict(meta), "data": {**dict(meta.get("data", {})), **provenance, **catalog}},
                       missed=missed)
    status = append_record(rec, ledger)
    return {**out, "status": status, "decisionDate": latest, "missedDecisions": missed,
            "liveGaps": provenance["liveGaps"], "top10": {m: rec["models"][m]["selection"] for m in run.RANKED}}


# --------------------------------------------------------------------------------------------- evaluate


def period_returns(market: run.Market, i: int, symbols: Iterable[str]) -> tuple[dict[str, float], str, str]:
    cal = market.calendar
    entry, exit_ = cal[i + 1], cal[i + LABEL]
    rets = {}
    for s in symbols:
        b0, b1 = market.bars.get(s, {}).get(entry), market.bars.get(s, {}).get(exit_)
        if b0 is not None and b1 is not None and b0[0] > 0:
            rets[s] = b1[0] / b0[0] - 1.0
    return rets, entry, exit_


def score_records(rows: Sequence[Mapping[str, Any]], market: run.Market) -> list[dict]:
    """Per record: immature, or the realized target, the basket return and every ranked model's rank IC."""
    cal = market.calendar
    pos = {d: k for k, d in enumerate(cal)}
    out = []
    for r in rows:
        i = pos.get(r["decisionDate"])
        if i is None or i != r["decisionSession"]:
            raise ForwardError(f"{r['decisionDate']}: decision session differs from the current calendar")
        if i + LABEL >= len(cal):
            out.append({"decisionDate": r["decisionDate"], "session": i, "status": "immature"})
            continue
        rets, entry, exit_ = period_returns(market, i, r["inForce"])
        bench = float(np.mean(list(rets.values()))) if rets else math.nan
        res: dict[str, Any] = {"decisionDate": r["decisionDate"], "session": i, "status": "scored", "entry": entry,
                               "exit": exit_, "basketGross": bench,
                               "missing": sorted(s for s in r["inForce"] if s not in rets), "models": {}}
        for m in run.MODELS:
            ic = math.nan
            if m != "B-ew":
                names = [s for s in r["eligible"] if s in rets and r["models"][m]["scores"].get(s) is not None]
                if len(names) >= run.MIN_IC_NAMES:
                    ic = ev.spearman([r["models"][m]["scores"][s] for s in names], [rets[s] - bench for s in names])
            res["models"][m] = {"rankIc": None if not math.isfinite(ic) else ic}
        out.append(res)
    return out


def contiguous_runs(rows: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    """Records split where a grid decision is missing (sessions not exactly REBALANCE apart)."""
    runs: list[list[Mapping[str, Any]]] = []
    for r in sorted(rows, key=lambda x: x["decisionSession"]):
        if runs and r["decisionSession"] - runs[-1][-1]["decisionSession"] == REBALANCE:
            runs[-1].append(r)
        else:
            runs.append([r])
    return runs


def protocol_portfolio(rows: Sequence[Mapping[str, Any]], market: run.Market, rules: PortfolioRules,
                       ew_rules: PortfolioRules) -> dict[str, Any]:
    """The protocol portfolio of every model driven by the recorded scores (matured records only).

    One backtest per run of contiguous records, from the first record's decision to the open of d+21 of the last
    one, where the decision of d+20 has no scores and so sells everything. Period return k = NAV at the entry open
    of record k+1 (or the close of the final session) / NAV at the entry open of record k - 1."""
    periods: dict[str, dict[str, float]] = {m: {} for m in run.MODELS}
    runs_out = []
    for seg in contiguous_runs(rows):
        s0, sk = seg[0]["decisionSession"], seg[-1]["decisionSession"]
        end = market.calendar[sk + LABEL]
        dates = [r["decisionDate"] for r in seg]
        info: dict[str, Any] = {"firstDecision": dates[0], "lastDecision": dates[-1], "end": end,
                                "records": len(seg), "rejections": {}, "trades": {}}
        for m in run.MODELS:
            lookup = {r["decisionDate"]: dict(r["models"][m]["scores"]) for r in seg}
            res = run_backtest(market.calendar, market.bars, market.periods,
                               lambda day, view, lookup=lookup: lookup.get(day, {}),
                               ew_rules if m == "B-ew" else rules, first_decision=s0, end=end)
            opens = {x["decisionDate"]: x["navAtOpen"] for x in res.rebalances}
            navs = [opens[d] for d in dates] + [res.daily[-1]["nav"]]
            for d, a, b in zip(dates, navs, navs[1:]):
                periods[m][d] = b / a - 1.0
            info["rejections"][m] = len(res.rejections)
            info["trades"][m] = res.summary["trades"]
        runs_out.append(info)
    return {"rules": rules.to_dict(), "ewRules": ew_rules.to_dict(), "runs": runs_out, "periodNet": periods}


def _cagr(rets: Sequence[float]) -> float | None:
    if not rets:
        return None
    growth = float(np.prod([1.0 + x for x in rets]))
    return growth ** (PERIODS_PER_YEAR / len(rets)) - 1.0 if growth > 0 else -1.0


def dev_ic_status(results_path: Path | None) -> tuple[dict[str, float], str | None]:
    """(model -> development mean IC of the official run, None) or ({}, why it could not be read)."""
    if results_path is None:
        return {}, "no results.json path"
    try:
        data = json.loads(Path(results_path).read_text(encoding="utf-8"))
    except OSError as exc:
        return {}, f"cannot read {results_path}: {type(exc).__name__}: {exc}"
    except ValueError as exc:
        return {}, f"{results_path} is not JSON: {exc}"
    try:
        ic = data["development"]["window"]["ic"]
        out = {m: float(v["summary"]["mean"]) for m, v in ic.items()}
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        return {}, (f"{results_path}: development.window.ic.<model>.summary.mean not found "
                    f"({type(exc).__name__}: {exc})")
    return (out, None) if out else ({}, f"{results_path}: development.window.ic is empty")


def dev_ic(results_path: Path | None) -> dict[str, float]:
    return dev_ic_status(results_path)[0]


def summarize(scored: Sequence[Mapping[str, Any]], dev: Mapping[str, float] | None = None) -> dict[str, Any]:
    ok = [s for s in scored if s.get("status") == "scored"]
    keep = ev.non_overlapping([s["session"] for s in ok], REBALANCE) if ok else np.zeros(0, dtype=bool)
    rows = [s for s, k in zip(ok, keep) if k]
    ew_nets = [r["models"]["B-ew"]["net"] for r in rows if r["models"]["B-ew"].get("net") is not None]
    models = {}
    for m in run.MODELS:
        nets = [r["models"][m]["net"] for r in rows if r["models"][m].get("net") is not None]
        block: dict[str, Any] = {"periods": len(nets), "netPerPeriod": ev.mean_t(nets), "cagrNet": _cagr(nets)}
        if m != "B-ew":
            excess = [r["models"][m]["netExcess"] for r in rows if r["models"][m].get("netExcess") is not None]
            ics = [r["models"][m]["rankIc"] for r in rows if r["models"][m]["rankIc"] is not None]
            cagr_m, cagr_ew = _cagr(nets), _cagr(ew_nets)
            block.update({
                "rankIc": ev.mean_t(ics),
                "rankIcHitRate": (sum(1 for x in ics if x > 0) / len(ics)) if ics else None,
                "netExcessPerPeriod": ev.mean_t(excess),
                "netExcessAnnualized": (cagr_m - cagr_ew) if cagr_m is not None and cagr_ew is not None else None,
                "hitRateVsEw": (sum(1 for x in excess if x > 0) / len(excess)) if excess else None,
            })
            if dev and m in dev:
                live = block["rankIc"]["mean"]
                block["forwardGateInfo"] = {"devMeanIc": dev[m], "halfDevIc": dev[m] / 2,
                                            "liveIcAtLeastHalfDev": None if live is None else live >= dev[m] / 2}
        models[m] = block
    span = None
    if rows:
        months = (date.fromisoformat(rows[-1]["exit"]) - date.fromisoformat(rows[0]["entry"])).days / 30.4375
        span = {"firstDecision": rows[0]["decisionDate"], "lastExit": rows[-1]["exit"], "months": months}
    return {"scored": len(ok), "nonOverlapping": len(rows), "immature": len(scored) - len(ok), "span": span,
            "periodsWithMissingMembers": sum(1 for s in ok if s.get("missing")),
            "models": models,
            "caveat": ("net returns are the protocol portfolio (vn_portfolio_backtest, PortfolioRules.from_protocol) "
                       "driven by the recorded scores; VN30-2 run 20261106T121249Z was no-go, so the protocol "
                       "forward gate (>= 6 months, live IC >= half the dev IC) applies to no model: these numbers "
                       "are information only")}


def load_bars_dir(directory: Path | str, symbols: Iterable[str]) -> dict[str, dict[str, Bar]]:
    """``<dir>/<SYM>.csv`` (date, open, high, low, close, volume: scripts/fetch_vn30_candles.py) for the symbols
    that have one; zero-volume bars dropped."""
    out = {}
    for sym in sorted(set(symbols)):
        path = Path(directory) / f"{sym}.csv"
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                out[sym] = {row["date"]: (float(row["open"]), float(row["high"]), float(row["low"]),
                                          float(row["close"]), float(row["volume"]))
                            for row in csv.DictReader(handle) if float(row["volume"]) > 0}
        except (KeyError, ValueError) as exc:
            raise ForwardError(f"{path}: unexpected CSV ({type(exc).__name__}: {exc})") from exc
    return out


def evaluate(history: run.Market, fetch: Fetch, *, now_utc: datetime, periods: Periods, non_trading: frozenset[str],
             coverage_to: str, ledger: Path | str = DEFAULT_LEDGER, snapshot_sha256: str | None = None,
             rules: tuple[PortfolioRules, PortfolioRules] | None = None, results_path: Path | None = None,
             leaver_bars: Mapping[str, Mapping[str, Bar]] | None = None,
             allow_missing_members: bool = False) -> dict[str, Any]:
    """Score the matured records (rank IC, protocol portfolio). ``leaver_bars``: bars of symbols the backend no
    longer serves; a matured record with an in-force member lacking an entry or exit open fails unless
    ``allow_missing_members`` (then it is counted)."""
    rows = load_ledger(ledger)
    warnings = coverage_warnings(now_utc, coverage_to)
    if not rows:
        return {"records": 0, **({"warnings": warnings} if warnings else {}), **summarize([])}
    if snapshot_sha256 is not None:
        bad = [r["decisionDate"] for r in rows if r.get("data", {}).get("snapshotSha256") != snapshot_sha256]
        if bad:
            raise ForwardError(f"records {bad[:3]} were made from another snapshot")
    check_membership(history.periods, periods)
    limit = _limit_for(history.calendar[-1], now_utc)
    try:
        index_bars = finalized(fetch_bars(fetch, CALENDAR_SYMBOL, limit), now_utc)
    except NotServed as exc:
        raise ForwardError(f"backend does not serve {CALENDAR_SYMBOL}") from exc
    live_sessions = live_calendar(history.calendar, index_bars, non_trading, coverage_to)
    symbols = {s for r in rows for s in r["inForce"]}
    live = load_live(fetch, symbols, limit, now_utc)
    from_files = sorted(s for s, v in live.items() if v is None and leaver_bars and s in leaver_bars)
    for s in from_files:
        live[s] = finalized(leaver_bars[s], now_utc)
    market, provenance = build_live_market(history, live, live_sessions, periods)
    scored = score_records(rows, market)
    affected = [{"decisionDate": s["decisionDate"], "missing": s["missing"]} for s in scored if s.get("missing")]
    if affected and not allow_missing_members:
        syms = sorted({x for a in affected for x in a["missing"]})
        raise ForwardError(f"{len(affected)} matured record(s) have in-force members without an entry or exit open "
                           f"({syms}; first {affected[0]['decisionDate']}): fetch them with "
                           "scripts/fetch_vn30_candles.py into <workspace>/data and pass --leaver-bars DIR "
                           "(or --allow-missing-members to report anyway)")
    matured = [r for r, s in zip(rows, scored) if s["status"] == "scored"]
    portfolio = None
    if matured:
        rules_pair = rules or run.portfolio_rules(run.PROTOCOL_PATH)
        portfolio = protocol_portfolio(matured, market, *rules_pair)
        for s in scored:
            if s["status"] != "scored":
                continue
            for m in run.MODELS:
                s["models"][m]["net"] = portfolio["periodNet"][m][s["decisionDate"]]
            for m in run.MODELS:
                s["models"][m]["netExcess"] = s["models"][m]["net"] - s["models"]["B-ew"]["net"]
    dev, dev_error = dev_ic_status(results_path)
    out = {"records": len(rows), "lastDecision": rows[-1]["decisionDate"], "latestSession": market.calendar[-1],
           **summarize(scored, dev),
           "devIc": {"path": None if results_path is None else str(results_path), "models": len(dev),
                     "error": dev_error},
           "membersFromLeaverBars": from_files, "missingMemberPeriods": affected,
           "portfolioRuns": None if portfolio is None else portfolio["runs"],
           "portfolioRules": None if portfolio is None else {"model": portfolio["rules"], "ew": portfolio["ewRules"]},
           "liveGaps": provenance["liveGaps"], "periods": scored}
    if warnings:
        out["warnings"] = warnings
    return out


# --------------------------------------------------------------------------------------------- CLI


def load_history(snapshot: Path, sha256: str) -> tuple[run.Market, str]:
    import vn30_research_data as rd

    snap = rd.load_snapshot(snapshot, sha256)
    return run.market_from_snapshot(snap), snap.sha256


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="VN30-2 forward paper recorder")
    ap.add_argument("command", choices=("record", "evaluate", "verify"))
    ap.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    ap.add_argument("--backend", default=os.getenv("BACKEND_BASE_URL") or DEFAULT_BACKEND)
    ap.add_argument("--snapshot", type=Path, default=run.DEFAULT_SNAPSHOT)
    ap.add_argument("--snapshot-sha256", default=run.DEFAULT_SNAPSHOT_SHA256)
    ap.add_argument("--membership", type=Path, default=DEFAULT_MEMBERSHIP)
    ap.add_argument("--calendar-file", type=Path, default=None)
    ap.add_argument("--results", type=Path, default=None,
                    help="results.json of the official run (dev IC for the forward-gate information)")
    ap.add_argument("--details", action="store_true", help="evaluate: include every scored period")
    ap.add_argument("--leaver-bars", type=Path, default=None,
                    help="evaluate: directory of <SYM>.csv (scripts/fetch_vn30_candles.py) for symbols the backend "
                         "no longer serves")
    ap.add_argument("--allow-missing-members", action="store_true",
                    help="evaluate: report even when an in-force member of a matured record has no entry/exit open")
    args = ap.parse_args(argv)
    try:
        if args.command == "verify":
            rows = load_ledger(args.ledger)
            out = {"ok": True, "records": len(rows), "lastDecision": rows[-1]["decisionDate"] if rows else None}
        else:
            proto = check_protocol()
            history, snap_sha = load_history(args.snapshot, args.snapshot_sha256)
            periods = load_membership(args.membership)
            non_trading, coverage_to = load_non_trading_days(args.calendar_file)
            fetch = http_fetch(args.backend)
            now = datetime.now(timezone.utc)
            if args.command == "record":
                meta = {"protocolSha256": proto["protocolSha256"], "evaluatorSha256": proto["evaluatorSha256"],
                        "registryHead": proto["registryHead"],
                        "membershipSha256": _sha_file(args.membership), "code": code_provenance(),
                        "data": {"snapshotSha256": snap_sha, "snapshot": args.snapshot.name}}
                out = record(history, fetch, now_utc=now, periods=periods, non_trading=non_trading,
                             coverage_to=coverage_to, meta=meta, ledger=args.ledger)
                codes = {"missed-deadline": EXIT_MISSED_DEADLINE, "data-not-ready": EXIT_DATA_NOT_READY}
                if out["status"] in codes:
                    print(json.dumps(run._clean({"ok": False, **out}), ensure_ascii=False))
                    return codes[out["status"]]
            else:
                results = args.results
                if results is None:
                    try:
                        results = ev.default_evidence_root() / DEFAULT_RUN_ID / "results.json"
                    except OSError:
                        results = None
                leavers = None
                if args.leaver_bars is not None:
                    leavers = load_bars_dir(args.leaver_bars, {s for _, members in periods for s in members})
                out = evaluate(history, fetch, now_utc=now, periods=periods, non_trading=non_trading,
                               coverage_to=coverage_to, ledger=args.ledger, snapshot_sha256=snap_sha,
                               rules=run.portfolio_rules(run.PROTOCOL_PATH), results_path=results,
                               leaver_bars=leavers, allow_missing_members=args.allow_missing_members)
                if not args.details:
                    out.pop("periods", None)
            out = {"ok": True, **out}
    except (ForwardError, LedgerError, OSError, ValueError, urllib.error.URLError) as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1
    print(json.dumps(run._clean(out), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
