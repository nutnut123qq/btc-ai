"""Read-only provider probe for the equity-expansion W0 gate (BID / VCB / CTG / VNINDEX).

Answers, with live public requests, the provider facts needed before extending the MBB
equity lane: VCI gap-chart symbol acceptance, bar coverage, timestamp convention, units,
partial-bar behaviour, adjusted-vs-raw price basis (VCI vs KBS around ex-rights dates),
KBS cross-check coverage (incl. index path), and whether VCI/KBS expose corporate
actions, events and quarterly bank fundamentals with a point-in-time publish date.

Nothing is persisted except the JSON report (and an optional raw-response cache so a
rerun does not re-hit providers). No DB access, no auth, no secrets.

Usage:
    python ai/scripts/probe_equity_market_data.py \
        --out evidence/equity-expansion/w0-provider-probe.json --cache-dir <scratch>/eqx-cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_mbb_market_data import _validate_ohlc  # noqa: E402  (reused, unmodified)

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
VCI_CHART_URL = "https://trading.vietcap.com.vn/api/chart/OHLCChart/gap-chart"
VCI_GRAPHQL_URL = "https://trading.vietcap.com.vn/data-mt/graphql"   # vnstock 3.5.1 (legacy)
VCI_IQ = "https://iq.vietcap.com.vn/api/iq-insight-service"           # vnstock main (current)
VCI_EVENT_CODES = "DIV,ISS,DDIND,DDINS,DDRP,AGME,AGMR,EGME,AIS,MA,MOVE,NLIS,OTHE,RETU,SUSP"
KBS_IIS = "https://kbbuddywts.kbsec.com.vn/iis-server/investment"
KBS_SAS = "https://kbbuddywts.kbsec.com.vn/sas"
KBS_FINANCE_INFO = f"{KBS_IIS}/stock/finance-info"                        # current path
KBS_FINANCE_INFO_OLD = f"{KBS_SAS}/kbsv-stock-data-store/stock/finance-info"  # vnstock 3.5.1 path
USER_AGENT = "Mozilla/5.0 (compatible; equity-expansion-w0-probe/1.0)"
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/120.0.0.0 Safari/537.36")
VCI_HEADERS = {"Referer": "https://trading.vietcap.com.vn/", "Origin": "https://trading.vietcap.com.vn",
               "User-Agent": BROWSER_UA}
KBS_HEADERS = {"User-Agent": BROWSER_UA}
REQUEST_DELAY_S = 0.5

STOCKS = ["BID", "VCB", "CTG"]
INDEX = "VNINDEX"
BANK_RATIO_KEYS = ["pb", "pe", "roe", "roa", "bookValuePerShare", "netInterestMargin", "npl",
                   "loansLossReservesToNPLs", "loansLossReserveToLoans", "provisionToOutstandingLoans",
                   "car", "ldrLoanDepositRatio", "casaRatio", "cir", "marketCap", "numberOfSharesMktCap"]


class Fetcher:
    """HTTP with polite delay, request log, error capture and an optional raw cache."""

    def __init__(self, cache_dir: Path | None) -> None:
        self.cache_dir = cache_dir
        self.log: list[dict[str, Any]] = []
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(method: str, url: str, payload: Any) -> str:
        raw = f"{method} {url} {json.dumps(payload, sort_keys=True)}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def get(self, url: str, *, payload: Any = None, headers: dict[str, str] | None = None,
            label: str) -> dict[str, Any]:
        method = "POST" if payload is not None else "GET"
        cached = self.cache_dir / f"{self._key(method, url, payload)}.json" if self.cache_dir else None
        if cached and cached.exists():
            result = json.loads(cached.read_text(encoding="utf-8"))
            self.log.append({**result["meta"], "label": label, "from_cache": True})
            return result

        body = None if payload is None else json.dumps(payload).encode("utf-8")
        hdrs = {"Accept": "application/json", "User-Agent": USER_AGENT, **(headers or {})}
        if body is not None:
            hdrs["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=body, headers=hdrs, method=method)
        started = time.monotonic()
        meta: dict[str, Any] = {"method": method, "url": url.split("?")[0],
                                "query": urllib.parse.urlparse(url).query or None,
                                "requested_at_utc": datetime.now(UTC).isoformat()}
        data: Any = None
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                text = resp.read().decode("utf-8", errors="replace")
                meta["status"] = resp.status
                meta["bytes"] = len(text)
                meta["rate_limit_headers"] = {k: v for k, v in resp.headers.items()
                                              if "ratelimit" in k.lower() or k.lower() == "retry-after"}
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    meta["error"] = "non-JSON body"
                    data = text[:300]
        except urllib.error.HTTPError as exc:
            meta["status"] = exc.code
            meta["error"] = f"HTTPError {exc.code} {exc.reason}"
            try:
                data = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:  # noqa: BLE001
                data = None
        except Exception as exc:  # noqa: BLE001  (record any network failure verbatim)
            meta["status"] = None
            meta["error"] = f"{type(exc).__name__}: {exc}"
        meta["elapsed_ms"] = round((time.monotonic() - started) * 1000)
        result = {"meta": meta, "data": data}
        self.log.append({**meta, "label": label, "from_cache": False})
        if cached and meta.get("status") is not None:
            cached.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        time.sleep(REQUEST_DELAY_S)
        return result


# ---------------------------------------------------------------- OHLC parsing

def num(value: Any) -> float | int:
    """Providers mix JSON numbers and numeric strings; keep ints as ints for readability."""
    x = float(value)
    return int(x) if x.is_integer() else x


def vci_bars(resp: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = resp["data"]
    info: dict[str, Any] = {"status": resp["meta"].get("status"), "error": resp["meta"].get("error")}
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        info["shape"] = type(data).__name__ if data is not None else None
        info["raw_sample"] = data if not isinstance(data, list) else data[:1]
        return [], info
    row = data[0]
    info["keys"] = sorted(row.keys())
    info["t_raw_type"] = type(row["t"][0]).__name__ if row.get("t") else None
    info["t_raw_sample"] = row["t"][:2] if row.get("t") else None
    info["field_raw_types"] = {f: type(row[f][0]).__name__ for f in ("o", "c", "v")} if row.get("t") else None
    bars = []
    for i, raw_ts in enumerate(row.get("t", [])):
        utc = datetime.fromtimestamp(int(float(raw_ts)), UTC)
        bars.append({"utc": utc.isoformat(), "local": utc.astimezone(VN_TZ).isoformat(),
                     **{f: num(row[f][i]) for f in ("o", "h", "l", "c", "v")}})
    bars.sort(key=lambda b: b["utc"])
    return bars, info


def kbs_bars(resp: dict[str, Any], key: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = resp["data"]
    info: dict[str, Any] = {"status": resp["meta"].get("status"), "error": resp["meta"].get("error")}
    if not isinstance(data, dict):
        info["raw_sample"] = data if isinstance(data, str) else str(type(data))
        return [], info
    info["root_keys"] = sorted(data.keys())[:10]
    rows = data.get(key)
    if not isinstance(rows, list):
        info["rows_kind"] = type(rows).__name__
        return [], info
    info["row_keys"] = sorted(rows[0].keys()) if rows else None
    info["t_raw_sample"] = [r.get("t") for r in rows[:2]]
    info["field_raw_types"] = {f: type(rows[0].get(f)).__name__ for f in ("o", "c", "v")} if rows else None
    bars = []
    for r in rows:
        t = str(r["t"]).strip()
        fmt = "%Y-%m-%d %H:%M" if len(t) > 10 else "%Y-%m-%d"
        local = datetime.strptime(t[:16], fmt).replace(tzinfo=VN_TZ)
        bars.append({"utc": local.astimezone(UTC).isoformat(), "local": local.isoformat(),
                     **{f: num(r[f]) for f in ("o", "h", "l", "c", "v")}})
    bars.sort(key=lambda b: b["utc"])
    return bars, info


def summarize(bars: list[dict[str, Any]], now_local: datetime) -> dict[str, Any]:
    if not bars:
        return {"count": 0}
    hours = sorted({b["local"][11:16] for b in bars})
    checks = _validate_ohlc([{"open_time_local": b["utc"], "open": b["o"], "high": b["h"],
                              "low": b["l"], "close": b["c"], "volume": b["v"]} for b in bars])

    def fractional(rows: list[dict[str, Any]]) -> int:
        return sum(any(float(b[k]) % 1 for k in ("o", "h", "l", "c")) for b in rows)

    return {
        "count": len(bars),
        "first_local": bars[0]["local"],
        "last_local": bars[-1]["local"],
        "distinct_local_open_hhmm": hours[:12],
        "first_bar": bars[0],
        "last_3_bars": bars[-3:],
        "fractional_price_rows": fractional(bars),
        "recent_100_fractional_rows": fractional(bars[-100:]),
        "last_bar_is_today_local": bars[-1]["local"][:10] == now_local.date().isoformat(),
        "probe_local_time": now_local.isoformat(timespec="minutes"),
        **{f"ohlc_{k}": v for k, v in checks.items()},
    }


def by_date(bars: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {b["local"][:10]: b for b in bars}


def crosscheck(vci: list[dict[str, Any]], kbs: list[dict[str, Any]], tail: int = 30,
               exclude_date: str | None = None) -> dict[str, Any]:
    """Daily VCI-vs-KBS diff stats. `exclude_date` drops today's still-forming bar (snapshot skew)."""
    v, k = by_date(vci), by_date(kbs)
    common = sorted(d for d in set(v) & set(k) if d != exclude_date)
    if not common:
        return {"overlap_dates": 0}

    def stats(dates: list[str]) -> dict[str, Any]:
        rel = [abs(float(v[d]["c"]) / float(k[d]["c"]) - 1) for d in dates]
        absd = [abs(float(v[d]["c"]) - float(k[d]["c"])) for d in dates]
        return {"n": len(dates), "close_abs_diff_max": round(max(absd), 4),
                "close_abs_diff_mean": round(statistics.mean(absd), 4),
                "close_rel_diff_max_pct": round(max(rel) * 100, 4),
                "exact_close_matches": sum(a == 0 for a in absd),
                "exact_volume_matches": sum(int(float(v[d]["v"])) == int(float(k[d]["v"])) for d in dates),
                "first": dates[0], "last": dates[-1]}

    out = {"overlap_dates": len(common),
           "vci_only_dates_in_kbs_span": len([d for d in v if common[0] <= d <= common[-1] and d not in k]),
           "kbs_only_dates": len([d for d in k if d not in v]),
           "last_n_excluding_today": stats(common[-tail:]), "all_overlap_excluding_today": stats(common),
           "vci_over_kbs_close_ratio_first_overlap": round(float(v[common[0]]["c"]) / float(k[common[0]]["c"]), 6)}
    if exclude_date and exclude_date in v and exclude_date in k:
        out["today_forming_bar"] = {"vci": v[exclude_date], "kbs": k[exclude_date]}
    return out


def ex_date_gap_analysis(events: list[dict[str, Any]], vci: list[dict[str, Any]],
                         kbs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each ex-rights date: overnight gap in each series vs the gap a RAW series would show.

    A raw (unadjusted) series gaps by ~-r/(1+r) on a stock-dividend/bonus ex-date (and by
    ~-cash/prev_close on a cash-dividend ex-date); a back-adjusted series does not. If the
    VCI/KBS close ratio steps across the ex-date, the providers use different bases.
    """
    v, k = by_date(vci), by_date(kbs)
    vd, kd = sorted(v), sorted(k)
    out = []
    for ev in events:
        ex = ev.get("exright_date")
        if not ex:
            continue
        ratio = ev.get("exercise_ratio")
        row: dict[str, Any] = {"exright_date": ex, "code": ev.get("code"), "title": ev.get("title"),
                               "exercise_ratio": ratio, "value_per_share": ev.get("value_per_share")}
        for name, series, dates in (("vci", v, vd), ("kbs", k, kd)):
            if ex not in series:
                row[name] = "ex-date not in series"
                continue
            idx = dates.index(ex)
            if idx == 0:
                row[name] = "no prior bar"
                continue
            prev, cur = series[dates[idx - 1]], series[ex]
            row[name] = {"prev_date": dates[idx - 1], "prev_close": prev["c"], "ex_open": cur["o"],
                         "ex_close": cur["c"],
                         "overnight_gap_pct": round((float(cur["o"]) / float(prev["c"]) - 1) * 100, 3)}
            if ev.get("code") == "DIV" and ev.get("value_per_share"):
                row[f"{name}_raw_expected_gap_pct"] = round(
                    -float(ev["value_per_share"]) / float(prev["c"]) * 100, 3)
            elif ev.get("code") == "ISS" and ratio:
                row[f"{name}_raw_expected_gap_pct"] = round(-float(ratio) / (1 + float(ratio)) * 100, 3)
        if isinstance(row.get("vci"), dict) and isinstance(row.get("kbs"), dict):
            row["vci_over_kbs_close_ratio_prev"] = round(float(row["vci"]["prev_close"])
                                                         / float(row["kbs"]["prev_close"]), 6)
            row["vci_over_kbs_close_ratio_ex"] = round(float(row["vci"]["ex_close"])
                                                       / float(row["kbs"]["ex_close"]), 6)
        out.append(row)
    return out


def truncate(obj: Any, n: int = 3) -> Any:
    if isinstance(obj, list):
        return [truncate(x, n) for x in obj[:n]] + ([f"...(+{len(obj) - n})"] if len(obj) > n else [])
    if isinstance(obj, dict):
        return {k: truncate(v, n) for k, v in obj.items()}
    if isinstance(obj, str) and len(obj) > 240:
        return obj[:240] + "...(trunc)"
    return obj


def day(value: Any) -> str | None:
    return str(value)[:10] if value else None


# ---------------------------------------------------------------- probe sections

def probe_ohlc(f: Fetcher, report: dict[str, Any], now_utc: datetime, now_local: datetime
               ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    to_sec = int(now_utc.timestamp())
    today = now_local.strftime("%d-%m-%Y")
    vci_daily: dict[str, list[dict[str, Any]]] = {}
    kbs_daily: dict[str, list[dict[str, Any]]] = {}
    for sym in STOCKS + [INDEX]:
        entry: dict[str, Any] = {}
        resp = f.get(VCI_CHART_URL, payload={"timeFrame": "ONE_DAY", "symbols": [sym], "to": to_sec,
                                             "countBack": 6000}, label=f"vci ONE_DAY {sym}")
        bars, info = vci_bars(resp)
        entry["vci_daily"] = {"request_symbol": sym, **info, **summarize(bars, now_local)}
        vci_daily[sym] = bars

        if sym == INDEX:
            # KBS: index path (vnstock uses /index/{sym}/data_day) and the stocks path for comparison
            for kind, url in (("index_path", f"{KBS_IIS}/index/{sym}/data_day"),
                              ("stocks_path", f"{KBS_IIS}/stocks/{sym}/data_day")):
                r = f.get(f"{url}?sdate=01-01-2000&edate={today}", label=f"kbs data_day {sym} {kind}")
                kb, kinfo = kbs_bars(r, "data_day")
                entry[f"kbs_daily_{kind}"] = {"url": r["meta"]["url"], **kinfo, **summarize(kb, now_local)}
                if kb and not kbs_daily.get(sym):
                    kbs_daily[sym] = kb
        else:
            r = f.get(f"{KBS_IIS}/stocks/{sym}/data_day?sdate=01-01-2000&edate={today}",
                      label=f"kbs data_day {sym}")
            kb, kinfo = kbs_bars(r, "data_day")
            entry["kbs_daily_requested_from_2000"] = {**kinfo, **summarize(kb, now_local)}
            kbs_daily[sym] = kb

        if sym == "BID":
            resp = f.get(VCI_CHART_URL, payload={"timeFrame": "ONE_HOUR", "symbols": [sym], "to": to_sec,
                                                 "countBack": 5000}, label="vci ONE_HOUR BID")
            hb, hinfo = vci_bars(resp)
            entry["vci_hourly"] = {**hinfo, **summarize(hb, now_local)}
            r = f.get(f"{KBS_IIS}/stocks/{sym}/data_60P?sdate=01-10-2026&edate={today}",
                      label="kbs data_60P BID")
            kh, khinfo = kbs_bars(r, "data_60P")
            entry["kbs_hourly_oct_2026"] = {**khinfo, **summarize(kh, now_local)}
        report["ohlc"][sym] = entry

    # Batching: does gap-chart serve several symbols in one request? (+ single-symbol control)
    for label, symbols in (("vci_multi_symbol_request", ["BID", "VCB"]),
                           ("vci_single_symbol_countback3_control", ["BID"])):
        resp = f.get(VCI_CHART_URL, payload={"timeFrame": "ONE_DAY", "symbols": symbols, "to": to_sec,
                                             "countBack": 3}, label=f"vci ONE_DAY {label}")
        d = resp["data"]
        report["ohlc"][label] = {
            "symbols": symbols, "status": resp["meta"].get("status"),
            "elements": len(d) if isinstance(d, list) else None,
            "bars_per_element": [len(e.get("t", [])) for e in d] if isinstance(d, list) else None,
            "element_symbols": [e.get("symbol") for e in d] if isinstance(d, list) else None}

    for sym in STOCKS + [INDEX]:
        report["crosscheck"][sym] = crosscheck(vci_daily[sym], kbs_daily.get(sym, []),
                                               exclude_date=now_local.date().isoformat())
    return vci_daily, kbs_daily


def probe_vci_corporate(f: Fetcher, corp: dict[str, Any], now_local: datetime) -> dict[str, list[dict[str, Any]]]:
    # (a) legacy GraphQL (vnstock 3.5.1 path) — negative check with minimal queries
    r1 = f.get(VCI_GRAPHQL_URL, payload={"query": "{ __typename }"}, headers=VCI_HEADERS,
               label="vci graphql __typename")
    r2 = f.get(VCI_GRAPHQL_URL, payload={"query": "query Q($ticker: String!) { OrganizationEvents(ticker: $ticker)"
                                                  " { id eventTitle exrightDate ratio } }",
                                         "variables": {"ticker": "BID"}},
               headers=VCI_HEADERS, label="vci graphql OrganizationEvents BID")
    corp["vci_graphql_legacy"] = {
        "url": VCI_GRAPHQL_URL, "method": "POST",
        "typename_probe": {"status": r1["meta"].get("status"), "body": truncate(r1["data"])},
        "events_probe": {"status": r2["meta"].get("status"), "body": truncate(r2["data"])}}

    # (b) IQ events (vnstock main path)
    events_by_sym: dict[str, list[dict[str, Any]]] = {}
    to_ymd = now_local.strftime("%Y%m%d")
    corp["vci_iq_events"] = {"url_template": f"{VCI_IQ}/v1/events?ticker={{sym}}&fromDate=YYYYMMDD&toDate=YYYYMMDD"
                                             f"&eventCode={VCI_EVENT_CODES}&page=0&size=N", "method": "GET"}
    for sym in STOCKS + [INDEX]:
        url = (f"{VCI_IQ}/v1/events?ticker={sym}&fromDate=20080101&toDate={to_ymd}"
               f"&eventCode={VCI_EVENT_CODES}&page=0&size=200")
        r = f.get(url, headers=VCI_HEADERS, label=f"vci iq events {sym}")
        body = r["data"] if isinstance(r["data"], dict) else {}
        data = body.get("data") if isinstance(body.get("data"), dict) else {}
        content = data.get("content") or []
        norm = [{"code": e.get("eventCode"), "type_vi": e.get("eventNameVi"), "title": e.get("eventTitleVi"),
                 "public_date": day(e.get("publicDate")), "record_date": day(e.get("recordDate")),
                 "exright_date": day(e.get("exrightDate")), "issue_date": day(e.get("issueDate")),
                 "listing_date": day(e.get("listingDate")), "payout_date": day(e.get("payoutDate")),
                 "exercise_ratio": e.get("exerciseRatio"), "value_per_share": e.get("valuePerShare")}
                for e in content]
        events_by_sym[sym] = norm
        codes: dict[str, int] = {}
        presence: dict[str, int] = {}
        for e in norm:
            codes[f'{e["code"]} | {e["type_vi"]}'] = codes.get(f'{e["code"]} | {e["type_vi"]}', 0) + 1
        for e in content:
            for kk in e:
                presence[kk] = presence.get(kk, 0) + 1
        corp["vci_iq_events"][sym] = {
            "status": r["meta"].get("status"), "envelope_status": body.get("status"), "msg": body.get("msg"),
            "total_elements": data.get("totalElements"), "returned": len(content),
            "raw_field_presence": presence, "event_code_counts": codes,
            "oldest_public_date": min((e["public_date"] for e in norm if e["public_date"]), default=None),
            "newest_public_date": max((e["public_date"] for e in norm if e["public_date"]), default=None),
            "with_exright_date": sum(bool(e["exright_date"]) for e in norm),
            "DIV_ISS_with_exright_latest": [e for e in norm if e["code"] in ("DIV", "ISS")
                                            and e["exright_date"]][:12],
            "other_types_sample": [e for e in norm if e["code"] not in ("DIV", "ISS")][:4],
            "raw_first_item": truncate(content[0], 3) if content else None,
        }

    # (c) IQ ratios (statistics-financial), company details, financial-statement section probe
    corp["vci_iq_statistics_financial"] = {"url_template": f"{VCI_IQ}/v1/company/{{sym}}/statistics-financial",
                                           "method": "GET"}
    for sym in STOCKS:
        r = f.get(f"{VCI_IQ}/v1/company/{sym}/statistics-financial", headers=VCI_HEADERS,
                  label=f"vci iq statistics-financial {sym}")
        body = r["data"] if isinstance(r["data"], dict) else {}
        rows = body.get("data") if isinstance(body.get("data"), list) else []
        types: dict[str, int] = {}
        for x in rows:
            types[str(x.get("ratioType"))] = types.get(str(x.get("ratioType")), 0) + 1
        q = sorted([x for x in rows if x.get("ratioType") == "RATIO_TTM"],
                   key=lambda x: (int(x.get("year") or 0), int(x.get("quarter") or 0)))
        corp["vci_iq_statistics_financial"][sym] = {
            "status": r["meta"].get("status"), "rows": len(rows), "ratio_type_counts": types,
            "all_fields": sorted(rows[0].keys()) if rows else None,
            "date_like_fields": sorted({kk for x in rows for kk in x
                                        if "date" in kk.lower() or "time" in kk.lower()}),
            "ttm_quarter_range": ([f'{q[0]["year"]}-Q{q[0]["quarter"]}', f'{q[-1]["year"]}-Q{q[-1]["quarter"]}']
                                  if q else None),
            "latest_4_ttm_quarters_bank_fields": [{"period": f'{x["year"]}-Q{x["quarter"]}',
                                                   **{kk: x.get(kk) for kk in BANK_RATIO_KEYS}} for x in q[-4:]],
        }
    r = f.get(f"{VCI_IQ}/v1/company/details?ticker=BID", headers=VCI_HEADERS, label="vci iq company details BID")
    det = r["data"].get("data") if isinstance(r["data"], dict) else None
    corp["vci_iq_company_details_BID"] = {
        "status": r["meta"].get("status"),
        "fields": sorted(det.keys()) if isinstance(det, dict) else None,
        "sample": ({kk: det.get(kk) for kk in ("ticker", "comTypeCode", "comGroupCode", "currentPrice",
                                                "issueShare", "numberOfSharesMktCap", "rating", "ratingAsOf")}
                   if isinstance(det, dict) else None)}
    r = f.get(f"{VCI_IQ}/v1/company/BID/financial-statement?section=RATIO", headers=VCI_HEADERS,
              label="vci iq financial-statement section=RATIO BID")
    corp["vci_iq_financial_statement_section_RATIO_BID"] = {"status": r["meta"].get("status"),
                                                            "body": truncate(r["data"])}
    return events_by_sym


def probe_kbs_corporate(f: Fetcher, corp: dict[str, Any]) -> None:
    kbs: dict[str, Any] = {}
    r = f.get(f"{KBS_FINANCE_INFO_OLD}/BID?page=1&pageSize=8&type=CSTC&unit=1000&termtype=2&languageid=1",
              headers=KBS_HEADERS, label="kbs sas finance-info (old path) BID")
    kbs["finance_info_old_sas_path"] = {"url": KBS_FINANCE_INFO_OLD, "status": r["meta"].get("status"),
                                        "error": r["meta"].get("error"), "body": truncate(r["data"])}
    kbs["finance_info_CSTC_quarterly"] = {
        "url_template": f"{KBS_FINANCE_INFO}/{{sym}}?page=1&pageSize=8&type=CSTC&unit=1000&termtype=2&languageid=1",
        "method": "GET"}
    for sym in STOCKS:
        r = f.get(f"{KBS_FINANCE_INFO}/{sym}?page=1&pageSize=8&type=CSTC&unit=1000&termtype=2&languageid=1",
                  headers=KBS_HEADERS, label=f"kbs finance-info CSTC Q {sym}")
        d = r["data"] if isinstance(r["data"], dict) else {}
        heads = d.get("Head") or []
        content = d.get("Content") if isinstance(d.get("Content"), dict) else {}
        items = [{"group": grp, "id": it.get("ReportNormID"), "name": it.get("Name"), "name_en": it.get("NameEn"),
                  "unit": it.get("Unit"), "Value1": it.get("Value1"), "Value2": it.get("Value2")}
                 for grp, lst in content.items() for it in (lst or [])]
        kbs["finance_info_CSTC_quarterly"][sym] = {
            "status": r["meta"].get("status"), "error": r["meta"].get("error"), "top_keys": sorted(d.keys()),
            "head_fields": sorted(heads[0].keys()) if heads else None,
            "periods": [{kk: h.get(kk) for kk in ("YearPeriod", "TermCode", "PeriodEnd", "ReportDate",
                                                  "DatePubDepartment", "CreatedDate", "LastUpdate")}
                        for h in heads],
            "item_count": len(items), "items": items,
        }
    for label, url in (("events_BID", f"{KBS_IIS}/stockinfo/event/BID?l=1&p=1&s=20"),
                       ("events_BID_eID2_dividend", f"{KBS_IIS}/stockinfo/event/BID?l=1&p=1&s=20&eID=2"),
                       ("news_BID", f"{KBS_IIS}/stockinfo/news/BID?l=1&p=1&s=5"),
                       ("profile_BID", f"{KBS_IIS}/stockinfo/profile/BID?l=1")):
        r = f.get(url, headers=KBS_HEADERS, label=f"kbs {label}")
        d = r["data"]
        entry: dict[str, Any] = {"status": r["meta"].get("status"), "error": r["meta"].get("error"),
                                 "url": r["meta"]["url"], "query": r["meta"]["query"]}
        if isinstance(d, dict):
            entry["top_keys"] = sorted(d.keys())
            if label == "profile_BID":
                cc = d.get("CharterCapital")
                entry["CharterCapital_rows"] = len(cc) if isinstance(cc, list) else None
                entry["CharterCapital_sample"] = truncate(cc, 4)
            else:
                entry["sample"] = truncate(d, 2)
        else:
            entry["sample"] = truncate(d, 3)
        kbs[label] = entry
    corp["kbs"] = kbs


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path)
    args = ap.parse_args()

    f = Fetcher(args.cache_dir)
    now_utc = datetime.now(UTC)
    now_local = now_utc.astimezone(VN_TZ)
    report: dict[str, Any] = {"generated_at_utc": now_utc.isoformat(), "probe_local_time": now_local.isoformat(),
                              "ohlc": {}, "crosscheck": {}, "corporate": {}}

    vci_daily, kbs_daily = probe_ohlc(f, report, now_utc, now_local)
    events_by_sym = probe_vci_corporate(f, report["corporate"], now_local)
    probe_kbs_corporate(f, report["corporate"])

    report["adjustment_checks"] = {
        sym: ex_date_gap_analysis([e for e in events_by_sym.get(sym, [])
                                   if e.get("exright_date") and e["code"] in ("DIV", "ISS")
                                   and e["exright_date"] >= "2016-08-01"][:10],
                                  vci_daily[sym], kbs_daily.get(sym, []))
        for sym in STOCKS}
    report["request_log"] = f.log
    report["request_count_network"] = sum(not x["from_cache"] for x in f.log)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(f.log)} requests, {report['request_count_network']} network)")


if __name__ == "__main__":
    main()
