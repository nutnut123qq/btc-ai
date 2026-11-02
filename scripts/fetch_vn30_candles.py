"""Fetch the frozen VN30-1 research snapshot: adjusted daily candles for every ticker that was ever in the
VN30 basket (per the membership file) plus VNINDEX, from the same VCI gap-chart endpoint the backend uses.

Writes <out>/<SYMBOL>.csv (date, open, high, low, close, volume; date = HOSE local session date) and
<out>/manifest.json (per symbol: bars, first/last date, sha256, or the fetch error). A symbol the provider
does not serve (delisted, renamed) is listed under "missing", never dropped silently. No DB, no secrets.

Usage: python -I scripts/fetch_vn30_candles.py --membership <vn30-membership.json> --out <dir> [--from 2016-01-01]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_mbb_market_data import VN_TZ, _vci_rows  # noqa: E402  (reused, unmodified)

INDEX = "VNINDEX"
COUNT_BACK = 5000  # daily bars; reaches back to ~2006, well before the earliest member date needed
DELAY_S = 0.5


def tickers(membership_path: Path) -> list[str]:
    periods = json.loads(membership_path.read_text(encoding="utf-8"))["periods"]
    return sorted({t for p in periods for t in p["constituents"]})


def fetch(symbol: str, since: str, today: str) -> tuple[list[dict], list[str]]:
    """Finalized bars since `since`; today's bar is dropped (the session may still be forming).

    Returns (rows, dates whose open/close fall outside low..high), the latter recorded, not dropped.
    """
    rows = []
    for b in _vci_rows(symbol, "ONE_DAY", COUNT_BACK):
        day = b["open_time_local"][:10]
        if since <= day < today:
            rows.append({"date": day, **{k: float(b[k]) for k in ("open", "high", "low", "close", "volume")}})
    rows.sort(key=lambda r: r["date"])
    if len({r["date"] for r in rows}) != len(rows):
        raise ValueError(f"{symbol}: duplicate session dates in provider response")
    if any(r["close"] <= 0 for r in rows):
        raise ValueError(f"{symbol}: non-positive close")
    anomalies = [r["date"] for r in rows
                 if not (0 < r["low"] <= min(r["open"], r["close"]) <= max(r["open"], r["close"]) <= r["high"])]
    return rows, anomalies


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--membership", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--from", dest="since", default="2016-01-01")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    today = datetime.now(VN_TZ).date().isoformat()
    manifest = {"source": "VCI gap-chart ONE_DAY (adjusted)", "since": args.since, "fetchedOnLocal": today,
                "excluded": "bars dated on the fetch day (possibly still forming)", "symbols": {}, "missing": {}}
    for symbol in tickers(args.membership) + [INDEX]:
        try:
            rows, anomalies = fetch(symbol, args.since, today)
        except Exception as exc:  # recorded per symbol; one failure must not hide the others
            manifest["missing"][symbol] = f"{type(exc).__name__}: {exc}"
            print(f"{symbol}: FAILED {exc}")
            time.sleep(DELAY_S)
            continue
        if not rows:
            manifest["missing"][symbol] = "provider returned no bars"
            print(f"{symbol}: no bars")
            time.sleep(DELAY_S)
            continue
        path = args.out / f"{symbol}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["date", "open", "high", "low", "close", "volume"])
            w.writeheader()
            w.writerows(rows)
        manifest["symbols"][symbol] = {"bars": len(rows), "first": rows[0]["date"], "last": rows[-1]["date"],
                                       "ohlcAnomalies": len(anomalies), "ohlcAnomalyDates": anomalies[:10],
                                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(f"{symbol}: {len(rows)} bars {rows[0]['date']}..{rows[-1]['date']}")
        time.sleep(DELAY_S)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"fetched={len(manifest['symbols'])} missing={sorted(manifest['missing'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
