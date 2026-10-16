"""Read-only MBB market-data probe for the MBB-00 feasibility gate.

Uses public VCI/KBS endpoints directly so the probe does not require a vnstock
install. It prints a compact JSON evidence report instead of persisting bars.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
VCI_CHART_URL = "https://trading.vietcap.com.vn/api/chart/OHLCChart/gap-chart"
KBS_BASE_URL = "https://kbbuddywts.kbsec.com.vn/iis-server/investment/stocks"
USER_AGENT = "Mozilla/5.0 (compatible; MBB-market-data-probe/1.0)"


def _request_json(url: str, *, payload: dict[str, Any] | None = None) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def _vci_rows(symbol: str, interval: str, count_back: int, to: int | None = None) -> list[dict[str, Any]]:
    payload = {
        "timeFrame": interval,
        "symbols": [symbol],
        "to": to or int(time.time()),
        "countBack": count_back,
    }
    data = _request_json(VCI_CHART_URL, payload=payload)
    if not isinstance(data, list) or not data:
        return []
    row = data[0]
    bars = []
    for idx, raw_ts in enumerate(row.get("t", [])):
        opened_at = datetime.fromtimestamp(int(raw_ts), UTC).astimezone(VN_TZ)
        bars.append(
            {
                "open_time_local": opened_at.isoformat(),
                "open": row["o"][idx],
                "high": row["h"][idx],
                "low": row["l"][idx],
                "close": row["c"][idx],
                "volume": row["v"][idx],
            }
        )
    return bars


def _kbs_rows(symbol: str, interval_suffix: str, start: str, end: str) -> list[dict[str, Any]]:
    key = f"data_{interval_suffix}"
    query = urllib.parse.urlencode({"sdate": start, "edate": end})
    url = f"{KBS_BASE_URL}/{symbol}/{key}?{query}"
    data = _request_json(url)
    bars = []
    for row in data.get(key, []) if isinstance(data, dict) else []:
        opened_at = datetime.strptime(row["t"], "%Y-%m-%d %H:%M").replace(tzinfo=VN_TZ)
        bars.append(
            {
                "open_time_local": opened_at.isoformat(),
                "open": row["o"],
                "high": row["h"],
                "low": row["l"],
                "close": row["c"],
                "volume": row["v"],
            }
        )
    return sorted(bars, key=lambda item: item["open_time_local"])


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "first_open_time_local": rows[0]["open_time_local"],
        "last_open_time_local": rows[-1]["open_time_local"],
        "first_row": rows[0],
        "last_row": rows[-1],
        "fractional_price_rows": sum(
            any(float(row[field]) % 1 != 0 for field in ("open", "high", "low", "close"))
            for row in rows
        ),
    }


def _validate_ohlc(rows: list[dict[str, Any]]) -> dict[str, Any]:
    timestamps = [row["open_time_local"] for row in rows]
    invalid = [
        row["open_time_local"]
        for row in rows
        if not (
            row["low"] <= row["open"] <= row["high"]
            and row["low"] <= row["close"] <= row["high"]
            and row["low"] <= row["high"]
            and row["volume"] >= 0
        )
    ]
    return {
        "is_monotonic": timestamps == sorted(timestamps),
        "duplicate_open_times": len(timestamps) - len(set(timestamps)),
        "invalid_ohlcv_rows": len(invalid),
        "invalid_examples": invalid[:3],
    }


def _hour_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hours = Counter(row["open_time_local"][11:13] for row in rows)
    dates = [row["open_time_local"][:10] for row in rows]
    return {
        "local_open_hours": dict(sorted(hours.items())),
        "trading_dates": len(set(dates)),
        "bars_per_date_counts": dict(sorted(Counter(dates).items())) if len(set(dates)) <= 10 else None,
    }


def _session_half_bars(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_date: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_date.setdefault(row["open_time_local"][:10], []).append(row)

    bars = []
    for trading_date, day_rows in sorted(by_date.items()):
        for session_name, session_rows in (
            ("morning", [r for r in day_rows if r["open_time_local"][11:16] < "12:00"]),
            ("afternoon", [r for r in day_rows if r["open_time_local"][11:16] >= "12:00"]),
        ):
            if not session_rows:
                continue
            ordered = sorted(session_rows, key=lambda item: item["open_time_local"])
            bars.append(
                {
                    "trading_date": trading_date,
                    "session_block": session_name,
                    "open_time_local": ordered[0]["open_time_local"],
                    "source_bars": len(ordered),
                    "open": ordered[0]["open"],
                    "high": max(r["high"] for r in ordered),
                    "low": min(r["low"] for r in ordered),
                    "close": ordered[-1]["close"],
                    "volume": sum(r["volume"] for r in ordered),
                }
            )
    return bars


def _daily_crosscheck(vci_rows: list[dict[str, Any]], kbs_rows: list[dict[str, Any]]) -> dict[str, Any]:
    vci_by_date = {row["open_time_local"][:10]: row for row in vci_rows}
    kbs_by_date = {row["open_time_local"][:10]: row for row in kbs_rows}
    dates = sorted(set(vci_by_date) & set(kbs_by_date))
    mismatches = []
    for date in dates:
        left, right = vci_by_date[date], kbs_by_date[date]
        diff = max(
            abs(float(left[field]) - float(right[field]))
            for field in ("open", "high", "low", "close")
        )
        if diff > 1.0:
            mismatches.append(
                {
                    "date": date,
                    "vci_close": left["close"],
                    "kbs_close": right["close"],
                    "max_price_diff": diff,
                }
            )
    return {
        "overlap_dates": len(dates),
        "price_mismatch_over_one_vnd": len(mismatches),
        "examples": mismatches[:5],
    }


def build_report(symbol: str) -> dict[str, Any]:
    vci_daily = _vci_rows(symbol, "ONE_DAY", 5000)
    vci_hourly = _vci_rows(symbol, "ONE_HOUR", 5000)
    vci_minute = _vci_rows(symbol, "ONE_MINUTE", 1000)
    kbs_daily = _kbs_rows(symbol, "day", "01-01-2020", "31-12-2026")
    kbs_hourly = _kbs_rows(symbol, "60P", "01-10-2026", "31-12-2026")

    hourly_checks = _validate_ohlc(vci_hourly)
    if not hourly_checks["is_monotonic"] or hourly_checks["invalid_ohlcv_rows"]:
        raise RuntimeError(f"VCI hourly validation failed: {hourly_checks}")

    half_bars = _session_half_bars(vci_hourly)
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "symbol": symbol,
        "price_unit": "VND (providers return full dong for recent rows; historical rows may be adjusted fractional dong)",
        "vci": {
            "daily": _summarize(vci_daily),
            "hourly": {**_summarize(vci_hourly), **_hour_profile(vci_hourly)},
            "minute_recent_1000": _summarize(vci_minute),
        },
        "kbs": {
            "daily_since_2020": _summarize(kbs_daily),
            "hourly_oct_2026_onward": {**_summarize(kbs_hourly), **_hour_profile(kbs_hourly)},
        },
        "crosscheck_daily_vci_vs_kbs": _daily_crosscheck(vci_daily, kbs_daily),
        "derived_4h_candidates": {
            "recommended_convention": "HOSE session-half bars: morning 09:00-11:30 and afternoon 13:00-14:45 ICT",
            "bars_per_full_trading_day": 2,
            "latest_examples": half_bars[-4:],
            "warning": "Not a continuous 240-minute crypto-style candle; produced from completed 1h source bars only.",
        },
        "known_corporate_action_dates_for_adjustment_checks": [
            "2024-05-23",
            "2025-01-07",
            "2025-08-13",
            "2026-07-09",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="MBB")
    parser.add_argument("--out", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    report = build_report(args.symbol.strip().upper())
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
