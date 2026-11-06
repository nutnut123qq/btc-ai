"""Fetch and freeze the VN30-2 research snapshot (docs/research/protocols/vn30-2.json, data.snapshot).

Adjusted daily OHLCV from the public VCI gap-chart endpoint (the same fetch as scripts/fetch_vn30_candles.py,
reused unmodified) for every ticker that was ever a VN30 member in the membership file (current members and
leavers), the two tickers removed at the first listed review, VNINDEX and VN30 (indices), the ETFs E1VFVN30 and
FUEVFVND, and the provider's continuous front-month future VN30F1M, from --since (or listing) to the latest
finalized session. The VN30F1M roll is checked against the individual contracts the provider still serves.

Requests are sequential with --delay seconds between them; a failure is retried once (after 60 s on HTTP 429)
and then recorded under "missing", never dropped silently. Writes one new immutable snapshot folder via
vn30_research_data.write_snapshot under --root (default <workspace>/data/vn30-research/snapshots).

Usage (from the repo root, venv python):
  python scripts/build_vn30_research_snapshot.py [--membership contracts/vn30-membership.json] [--root DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import fetch_vn30_candles as fvc  # noqa: E402
from probe_mbb_market_data import USER_AGENT, VCI_CHART_URL, VN_TZ, _vci_rows  # noqa: E402
import vn30_research_data as vrd  # noqa: E402

INDICES = ("VNINDEX", "VN30")
ETFS = ("E1VFVN30", "FUEVFVND")
FUTURE = "VN30F1M"
# KRX-era contract codes (41I1 + year letter + month 1-9/A-C + 000); the provider serves recent ones only.
CONTRACT_YEARS = {"F": 2025, "G": 2026}
MONTH_CODES = "123456789ABC"
DEFAULT_ROOT = REPO.parent / "data" / "vn30-research" / "snapshots"
# Bars kept in the file but not the symbol's own history (documented, applied by the loader).
EXCLUSIONS = {
    "VHM": {"before": "2018-05-17",
            "reason": "ticker reuse: Vinhomes first traded on HOSE on 2018-05-17; the provider's earlier VHM bars "
                      "(2011-11-10..2018-05-16: 105 sparse low-volume trades at 8-11k VND, then zero-volume "
                      "placeholders) belong to an earlier security and must not feed features"},
}
KNOWN_ISSUES = [
    "2018-01-23 and 2018-01-24 carry zero volume and unchanged prices for every HOSE symbol and VNINDEX/VN30 "
    "(the local backend, fed by the same provider, shows the same for MBB), while HNX/UPCoM names and VN30F1M traded: "
    "most likely a provider hole, not a halt; kept as zero-volume bars, so they are not sessions",
    "tickers that moved to HOSE (e.g. ACB, SHB, DGC from HNX; VIB, LPB, BCM, GVR, POW, BSR, MCH from UPCoM) "
    "include their pre-HOSE history; this source does not say the venue, and pre-HOSE bars had other price bands",
    "early bars of some names (NT2, PDR, CTD, BMP, PNJ, DGC, BCM, MCH, E1VFVN30) have many zero-volume days and a "
    "few OHLC anomalies (open or close outside low..high), listed per symbol; VN30 and VNINDEX have OHLC "
    "anomalies in 2011-2013 index bars",
    "ROS was delisted from HOSE in 2022; its bars end on 2022-08-11",
]


def symbol_roles(membership: Path) -> dict[str, str]:
    periods = json.loads(membership.read_text(encoding="utf-8"))["periods"]
    roles = {t: "member" for t in fvc.tickers(membership)}
    for t in periods[0].get("removed", []):
        roles.setdefault(t, "leaver-at-first-period")  # in VN30 before 2018-01-22, never in force afterwards
    roles.update({s: "index" for s in INDICES})
    roles.update({s: "etf" for s in ETFS})
    roles[FUTURE] = "future-continuous"
    return roles


def cutoff_exclusive(now_local: datetime, final_after: str) -> str:
    """Bars dated before the returned day are final; today's bar counts only after `final_after` (HH:MM ICT)."""
    hh, mm = (int(x) for x in final_after.split(":"))
    today = now_local.date()
    final_today = (now_local.hour, now_local.minute) >= (hh, mm)
    return (today + timedelta(days=1) if final_today else today).isoformat()


def fetch_with_retry(symbol: str, since: str, cutoff: str, delay: float) -> list[dict]:
    try:
        return fvc.fetch(symbol, since, cutoff)[0]
    except urllib.error.HTTPError as exc:
        time.sleep(60 if exc.code == 429 else max(delay, 5))
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        time.sleep(max(delay, 5))
    return fvc.fetch(symbol, since, cutoff)[0]


def contract_rows(code: str, cutoff: str) -> list[dict]:
    rows = [{"date": b["open_time_local"][:10], **{k: float(b[k]) for k in ("open", "high", "low", "close", "volume")}}
            for b in _vci_rows(code, "ONE_DAY", 400)]
    return sorted((r for r in rows if r["date"] < cutoff), key=lambda r: r["date"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--membership", type=Path, default=REPO / "contracts" / "vn30-membership.json")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--since", default="2011-01-01")
    ap.add_argument("--delay", type=float, default=1.2)
    ap.add_argument("--final-after", default="16:00", help="ICT time after which the fetch day's bar is final")
    args = ap.parse_args()
    if args.delay < 1.0:
        ap.error("--delay must be >= 1 s (polite crawling)")

    started = datetime.now(UTC)
    now_local = started.astimezone(VN_TZ)
    cutoff = cutoff_exclusive(now_local, args.final_after)
    roles = symbol_roles(args.membership)
    series, missing = {}, {}
    for sym in sorted(roles, key=lambda s: (roles[s] != "member", s)):
        try:
            rows = fetch_with_retry(sym, args.since, cutoff, args.delay)
            if not rows:
                raise ValueError("provider returned no bars in the window")
            series[sym] = rows
            print(f"{sym:9s} {roles[sym]:22s} {len(rows):5d} bars {rows[0]['date']}..{rows[-1]['date']}")
        except Exception as exc:  # recorded per symbol; one failure must not hide the others
            missing[sym] = f"{type(exc).__name__}: {exc}"
            print(f"{sym:9s} FAILED {missing[sym]}")
        time.sleep(args.delay)

    contracts = {}
    for letter in CONTRACT_YEARS:
        for month in MONTH_CODES:
            code = f"41I1{letter}{month}000"
            try:
                rows = contract_rows(code, cutoff)
            except Exception as exc:  # absence is expected for old or unlisted months
                rows = []
                print(f"{code}: {type(exc).__name__}")
            if rows:
                contracts[(CONTRACT_YEARS[letter], MONTH_CODES.index(month) + 1)] = rows
            time.sleep(args.delay)
    roll = vrd.verify_front_month_roll(series.get(FUTURE, []), contracts)
    print(f"VN30F1M roll check: {roll['daysMatchingFront']}/{roll['daysChecked']} days match the front contract "
          f"over {roll['window']} ({roll['daysFrontNotServed']} days with the front contract not served)")

    finished = datetime.now(UTC)
    meta = {
        "protocol": "vn30-2",
        "source": {
            "name": "VCI (Vietcap) gap-chart, public, no login",
            "urlPattern": f"POST {VCI_CHART_URL} "
                          '{"timeFrame":"ONE_DAY","symbols":["<SYMBOL>"],"to":<unix now>,"countBack":%d}'
                          % fvc.COUNT_BACK,
            "userAgent": USER_AGENT,
            "adjusted": "prices adjusted by the provider for corporate actions as of the fetch time; a later "
                        "fetch can re-adjust history, which is why the snapshot is frozen",
            "units": "stocks and ETFs in VND, indices and VN30F1M in index points; volume in shares/contracts "
                     "as the provider reports it",
            "requestDelaySeconds": args.delay,
        },
        "fetchStartedUtc": started.isoformat(timespec="seconds"),
        "fetchFinishedUtc": finished.isoformat(timespec="seconds"),
        "fetchLocalDate": now_local.date().isoformat(),
        "since": args.since,
        "lastSessionRule": f"bars dated before {cutoff} only; the fetch day's bar is kept only when the fetch "
                           f"started at or after {args.final_after} ICT (HOSE closes 14:45)",
        "zeroVolumeRule": "zero-volume bars are kept as served (volume 0), listed per symbol, never filled; "
                          "they are not sessions",
        "futures": {
            "symbol": FUTURE,
            "construction": "provider-built continuous front month, not back-adjusted (price jumps at rolls stay)",
            "roll": "the front contract until its last trading day (expiry, third Thursday of the month), then the "
                    "next month; checked against the individual KRX-coded contracts (41I1<year><month>000) the "
                    "provider still serves (see rollCheck, months keyed YYYY-MM), assumed before that window",
            "rollCheck": roll,
        },
        "knownIssues": KNOWN_ISSUES,
        "roles": {
            "member": "in at least one VN30 period of membership.json (current members and leavers)",
            "leaver-at-first-period": "removed at the first listed review (2018-01-22); never in force in "
                                      "membership.json, kept for completeness",
            "index": "index level series", "etf": "VN30-tracking ETF", "future-continuous": "see futures",
        },
    }
    snap = vrd.write_snapshot(args.root, now_local.strftime("%Y%m%d"), series, {s: roles[s] for s in series},
                              args.membership, meta=meta, missing=missing,
                              exclusions={s: e for s, e in EXCLUSIONS.items() if s in series})
    loaded = vrd.load_snapshot(snap)
    print(f"snapshot {snap}")
    print(f"sha256 {loaded.sha256}  sessions={len(loaded.sessions)} "
          f"{loaded.sessions[0]}..{loaded.sessions[-1]}  symbols={len(loaded.symbols)}  missing={sorted(missing)}")
    return 0 if not missing else 2


if __name__ == "__main__":
    sys.exit(main())
