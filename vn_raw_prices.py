"""Raw (unadjusted) HOSE daily prices: reference price, ceiling/floor, limit-locked opens, traded value.

Roadmap item 1.1-raw. The VN30 research snapshot (VCI gap-chart) is back-adjusted, so limit-lock detection, tick
sizes and traded value computed on it are approximate. This module fetches the RAW daily history that CafeF serves
publicly behind its price-history pages and derives the exchange-level quantities from it.

Source (public, no login, robots.txt of cafef.vn is ``Allow: /``)::

    GET https://cafef.vn/du-lieu/Ajax/PageNew/DataHistory/PriceHistory.ashx
        ?Symbol=MBB&StartDate=MM/DD/YYYY&EndDate=MM/DD/YYYY&PageIndex=1&PageSize=30

Observed provider behaviour (probed 2026-11-06, pinned by the tests): dates are month-first; a query never spans
more than about one year (older bars are silently cut); at most 30 records per page; the end date is not reliably
inclusive. The fetcher therefore asks for half-year windows that overlap by one day, pages each window until its
``TotalCount`` is reached, and de-duplicates the overlap (a duplicate must be identical or the fetch fails).

Provider record fields and units (thousand VND for prices, billion VND for values)::

    Ngay            session date DD/MM/YYYY
    GiaMoCua/GiaCaoNhat/GiaThapNhat/GiaDongCua   raw open/high/low/close
    GiaDieuChinh    CafeF's own adjusted close
    ThayDoi         "-0,25 (-1,28%)": change of the close vs the session's reference price
    KhoiLuongKhopLenh / GiaTriKhopLenh   matched (order-book) volume / value
    KLThoaThuan / GtThoaThuan            put-through volume / value

Files: ``<out>/<SYMBOL>.json`` holds the provider records VERBATIM (sorted by date) so a parsing change never needs a
refetch; ``<out>/manifest.json`` records bars, first/last date and sha256 per symbol. A symbol file is write-once: a
rerun keeps an existing file whose sha256 matches the manifest and refuses to overwrite a different one.

HOSE price rules (broker summary of the HOSE trading regulation, Dec 2023, cross-checked empirically against the raw
highs/lows in ``crosscheck``): tick 10 VND below 10,000, 50 VND from 10,000 to 49,950, 100 VND from 50,000; daily band
+/-7% of the reference price (+/-20% on the first listing session, on the first session after a suspension of more
than 25 sessions, and on some ex-right sessions); the reference price is the previous session's close, adjusted by the
exchange on ex-right sessions. The rounding of the band edge to a valid tick is a declared choice
(``LIMIT_ROUNDING``) fixed by the empirical check, not by the text.

Usage (from the repo root, venv python):
  python vn_raw_prices.py fetch --membership contracts/vn30-membership.json --out <ROOT>/data/vn30-research/raw
  python vn_raw_prices.py crosscheck --raw <dir> --adjusted <VCI csv dir> --membership contracts/vn30-membership.json
      --out <report.json>

Known provider defects (found by ``crosscheck``, handled or reported, never silently patched in the files):
- market-wide holes: some sessions are absent for every symbol (e.g. 2025-02-03, 2026-01-29, 2026-01-30);
  ``traded_value`` is None there and the next session's reference comes from the provider's change field;
- the change field (hence the provider reference) is sometimes plain wrong or misses a corporate action, and
  the raw close occasionally differs from the official close that the next session's reference reveals;
  ``reference_price(..., "exchange")`` arbitrates between the candidates with the session's own range;
- before joining HOSE several VN30 members traded on HNX or UPCoM (other bands, ticks and reference rules):
  the HOSE helpers are valid only while the symbol is HOSE-listed, which holds for every VN30 member session.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from bisect import bisect_left
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from vn_execution_model import DEFAULT_PRICE_BAND_FRACTION

SCHEMA = "vn-raw-daily/v1"
SOURCE = "CafeF price history (PriceHistory.ashx), raw + CafeF-adjusted close, matched and put-through volume/value"
SOURCE_URL = "https://cafef.vn/du-lieu/Ajax/PageNew/DataHistory/PriceHistory.ashx"
USER_AGENT = "Mozilla/5.0 (compatible; research-data-fetcher/1.0; public daily price history, sequential, >= 1 s between requests)"
PAGE_SIZE = 30  # provider cap: a larger PageSize still returns 30 records
DELAY_S = 1.2  # minimum gap between request starts (sequential requests only)
MIN_PAUSE_S = 0.2
TIMEOUT_S = 30
RETRY_WAIT_S = 60
VN_TZ = timezone(timedelta(hours=7))
SESSION_FINAL_LOCAL_HOUR = 16  # put-through trading ends 15:00; a fetch at/after 16:00 local keeps that day's bar
EXCLUDED = "bars dated on or after the cutoff day (the fetch day itself when fetched before 16:00 local)"

PRICE_UNIT_VND = 1000  # provider prices are in thousand VND
VALUE_UNIT_VND = 1_000_000_000  # provider values are in billion VND (2 decimals -> 10 million VND resolution)
FIRST_DAY_BAND_FRACTION = 0.20
LONG_SUSPENSION_SESSIONS = 25
REFERENCE_NOISE = 0.001
LIMIT_ROUNDING = "inward"  # ceiling = largest valid tick <= ref*(1+b); floor = smallest valid tick >= ref*(1-b)
ROUNDING_MODES = ("inward", "nearest")

Side = Literal["buy", "sell"]

# --------------------------------------------------------------------------------------------- HOSE price rules

def hose_tick_size(price_vnd: float) -> int:
    """HOSE tick for stocks / closed-end fund certificates at this price level (VND)."""
    if price_vnd <= 0:
        raise ValueError("price must be positive")
    if price_vnd < 10_000:
        return 10
    if price_vnd < 50_000:
        return 50
    return 100


def is_valid_tick_price(price_vnd: int) -> bool:
    return price_vnd > 0 and price_vnd % hose_tick_size(price_vnd) == 0


def round_down_to_tick(price_vnd: float) -> int:
    """Largest valid HOSE price <= price (the tick depends on the level of the result)."""
    p = int(math.floor(price_vnd + 1e-9))
    while p > 0 and not is_valid_tick_price(p):
        p -= 1
    if p <= 0:
        raise ValueError("no valid tick price at or below the input")
    return p


def round_up_to_tick(price_vnd: float) -> int:
    """Smallest valid HOSE price >= price."""
    p = max(10, int(math.ceil(price_vnd - 1e-9)))
    while not is_valid_tick_price(p):
        p += 1
    return p


def round_nearest_to_tick(price_vnd: float) -> int:
    """Nearest valid HOSE price (ties go up)."""
    lo, hi = round_down_to_tick(price_vnd), round_up_to_tick(price_vnd)
    return lo if price_vnd - lo < hi - price_vnd else hi


def price_limits(reference_vnd: float, band_fraction: float = DEFAULT_PRICE_BAND_FRACTION,
                 rounding: str | None = None) -> tuple[int, int]:
    """(floor, ceiling) in VND for a reference price, rounded to valid ticks.

    rounding "inward" (default, ``LIMIT_ROUNDING``): ceiling down, floor up, so both stay inside the +/- band;
    "nearest": both to the nearest valid tick (kept for the empirical check in ``crosscheck``).
    """
    if reference_vnd <= 0:
        raise ValueError("reference price must be positive")
    if not 0 < band_fraction < 1:
        raise ValueError("band_fraction must be in (0, 1)")
    rounding = rounding or LIMIT_ROUNDING
    hi, lo = reference_vnd * (1 + band_fraction), reference_vnd * (1 - band_fraction)
    if rounding == "inward":
        return round_up_to_tick(lo), round_down_to_tick(hi)
    if rounding == "nearest":
        return round_nearest_to_tick(lo), round_nearest_to_tick(hi)
    raise ValueError(f"unknown rounding {rounding!r}")


# ----------------------------------------------------------------------------------------------- record parsing

_CHANGE_RE = re.compile(r"^\s*([+-]?\d+(?:[.,]\d+)?)")


def parse_change(text: Any) -> float | None:
    """'-0,25 (-1,28%)' -> -0.25 (thousand VND); None when absent or unparseable."""
    if not isinstance(text, str):
        return None
    m = _CHANGE_RE.match(text)
    return float(m.group(1).replace(",", ".")) if m else None


def _vnd(x: Any, unit: int) -> int:
    return int(round(float(x) * unit))


@dataclass(frozen=True)
class RawBar:
    date: str  # ISO session date
    open: int  # raw VND (0 = provider had no open)
    high: int
    low: int
    close: int
    adj_close: float  # CafeF adjusted close, VND
    provider_reference: int | None  # close - ThayDoi, VND (exchange reference incl. ex-right adjustment)
    matched_volume: int
    matched_value_vnd: int
    put_through_volume: int
    put_through_value_vnd: int


def parse_record(rec: dict[str, Any]) -> RawBar:
    day = datetime.strptime(rec["Ngay"], "%d/%m/%Y").date().isoformat()
    close = _vnd(rec["GiaDongCua"], PRICE_UNIT_VND)
    change = parse_change(rec.get("ThayDoi"))
    ref = close - _vnd(change, PRICE_UNIT_VND) if change is not None else None
    return RawBar(
        date=day,
        open=_vnd(rec.get("GiaMoCua") or 0, PRICE_UNIT_VND),
        high=_vnd(rec.get("GiaCaoNhat") or 0, PRICE_UNIT_VND),
        low=_vnd(rec.get("GiaThapNhat") or 0, PRICE_UNIT_VND),
        close=close,
        adj_close=float(rec["GiaDieuChinh"]) * PRICE_UNIT_VND,
        provider_reference=ref if ref is not None and ref > 0 else None,
        matched_volume=int(rec.get("KhoiLuongKhopLenh") or 0),
        matched_value_vnd=_vnd(rec.get("GiaTriKhopLenh") or 0, VALUE_UNIT_VND),
        put_through_volume=int(rec.get("KLThoaThuan") or 0),
        put_through_value_vnd=_vnd(rec.get("GtThoaThuan") or 0, VALUE_UNIT_VND),
    )


# ------------------------------------------------------------------------------------------------------ fetching

Getter = Callable[[str], Any]


def _http_get_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503) and attempt == 0:
                time.sleep(RETRY_WAIT_S)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 0:
                time.sleep(RETRY_WAIT_S)
                continue
            raise
    raise RuntimeError("unreachable")


def query_url(symbol: str, start: date, end: date, page: int) -> str:
    q = urllib.parse.urlencode({"Symbol": symbol, "StartDate": start.strftime("%m/%d/%Y"),
                                "EndDate": end.strftime("%m/%d/%Y"), "PageIndex": page, "PageSize": PAGE_SIZE})
    return f"{SOURCE_URL}?{q}"


def windows(since: date, until: date) -> list[tuple[date, date]]:
    """Half-year windows covering since..until, each overlapping the next by one day (end not reliably inclusive)."""
    out, start = [], since
    while start <= until:
        nxt = date(start.year, 7, 1) if start.month < 7 else date(start.year + 1, 1, 1)
        out.append((start, min(nxt, until)))
        start = nxt
    return out


def pause_after(elapsed_s: float) -> float:
    """Pause after a response so request starts are >= DELAY_S apart, and never less than MIN_PAUSE_S."""
    return max(MIN_PAUSE_S, DELAY_S - elapsed_s)


def fetch_window(symbol: str, start: date, end: date, get: Getter, sleep: Callable[[float], None]) -> list[dict]:
    """All provider records of one window (paged to TotalCount)."""
    records: list[dict] = []
    page, total = 1, None
    while True:
        t0 = time.monotonic()
        payload = get(query_url(symbol, start, end, page))
        sleep(pause_after(time.monotonic() - t0))
        if not isinstance(payload, dict) or not payload.get("Success", False) or not isinstance(payload.get("Data"), dict):
            raise ValueError(f"{symbol} {start}..{end} page {page}: unexpected payload")
        data = payload["Data"]
        total = int(data.get("TotalCount") or 0) if total is None else total
        batch = data.get("Data") or []
        if any(r.get("Symbol", symbol).upper() != symbol.upper() for r in batch):
            raise ValueError(f"{symbol}: provider returned another symbol")
        records.extend(batch)
        if not batch or len(records) >= total or len(batch) < PAGE_SIZE:
            break
        page += 1
    if len(records) != total:
        raise ValueError(f"{symbol} {start}..{end}: got {len(records)} records, provider TotalCount {total}")
    return records


def fetch_cutoff(now_local: datetime) -> date:
    """First excluded session date: the fetch day before 16:00 local (session may be forming), else the next day."""
    d = now_local.date()
    return d if now_local.hour < SESSION_FINAL_LOCAL_HOUR else d + timedelta(days=1)


PRICE_FIELDS = ("GiaDieuChinh", "GiaDongCua", "ThayDoi", "GiaMoCua", "GiaCaoNhat", "GiaThapNhat")


def collapse_duplicate(symbol: str, day: str, a: dict, b: dict) -> dict:
    """Two provider records for one session: identical -> either; same prices -> the one with the larger matched
    volume (the provider serves zero-volume shadow rows, e.g. GVR on UPCoM in 2018); else a hard error."""
    if a == b:
        return a
    if any(a.get(f) != b.get(f) for f in PRICE_FIELDS):
        raise ValueError(f"{symbol} {day}: provider records for the same session disagree")
    key = lambda r: (r.get("KhoiLuongKhopLenh") or 0, r.get("KLThoaThuan") or 0)  # noqa: E731
    return a if key(a) >= key(b) else b


def fetch_symbol(symbol: str, since: date, cutoff: date, get: Getter = _http_get_json,
                 sleep: Callable[[float], None] = time.sleep, stats: dict | None = None) -> list[dict]:
    """Verbatim provider records dated since <= d < cutoff, one per session, oldest first.

    `stats["duplicatesCollapsed"]` counts same-session records that differed only in volume fields."""
    by_date: dict[str, dict] = {}
    collapsed = 0
    for start, end in windows(since, cutoff):
        for rec in fetch_window(symbol, start, end, get, sleep):
            day = datetime.strptime(rec["Ngay"], "%d/%m/%Y").date()
            if not since <= day < cutoff:
                continue
            key = day.isoformat()
            if key in by_date:
                collapsed += by_date[key] != rec
                rec = collapse_duplicate(symbol, key, by_date[key], rec)
            by_date[key] = rec
    if stats is not None:
        stats["duplicatesCollapsed"] = collapsed
    return [by_date[k] for k in sorted(by_date)]


def validate_records(symbol: str, records: list[dict]) -> list[str]:
    """Hard errors raise; soft anomalies (OHLC outside low..high, VWAP outside the range) are returned as dates."""
    anomalies = []
    for rec in records:
        bar = parse_record(rec)
        if bar.close <= 0:
            raise ValueError(f"{symbol} {bar.date}: non-positive close")
        prices = [p for p in (bar.open, bar.close) if p > 0]
        if bar.low <= 0 or bar.high < bar.low or any(not bar.low <= p <= bar.high for p in prices):
            anomalies.append(bar.date)
            continue
        if bar.matched_volume > 0 and bar.matched_value_vnd > 0:
            vwap = bar.matched_value_vnd / bar.matched_volume
            slack = VALUE_UNIT_VND * 0.005 / bar.matched_volume + 1.0  # half of the 0.01-billion value resolution
            if not bar.low - slack <= vwap <= bar.high + slack:
                anomalies.append(bar.date)
    return anomalies


def tickers(membership_path: Path) -> list[str]:
    periods = json.loads(membership_path.read_text(encoding="utf-8"))["periods"]
    return sorted({t for p in periods for t in p["constituents"]})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_symbol_file(out: Path, symbol: str, records: list[dict], since: str, fetched_on: str) -> Path:
    """Write-once: keep an identical existing file, refuse to replace a different one."""
    path = out / f"{symbol}.json"
    body = json.dumps({"schema": SCHEMA, "symbol": symbol, "source": SOURCE, "sourceUrl": SOURCE_URL,
                       "since": since, "fetchedOnLocal": fetched_on, "excluded": EXCLUDED,
                       "units": {"prices": "thousand VND", "values": "billion VND", "volumes": "shares"},
                       "records": records}, ensure_ascii=False, indent=1).encode("utf-8")
    if path.exists():
        raise FileExistsError(f"{path} already exists; raw snapshot files are immutable")
    path.write_bytes(body)
    return path


def run_fetch(symbols: Iterable[str], out: Path, since: date, cutoff: date, get: Getter = _http_get_json,
              sleep: Callable[[float], None] = time.sleep, log: Callable[[str], None] = print,
              fetched_on: str | None = None) -> dict:
    """Fetch every symbol not yet in `out` (resumable); bars are dated since <= d < cutoff."""
    fetched_on = fetched_on or datetime.now(VN_TZ).date().isoformat()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else
                {"schema": SCHEMA, "source": SOURCE, "sourceUrl": SOURCE_URL, "since": since.isoformat(),
                 "fetchedOnLocal": fetched_on, "cutoffExclusive": cutoff.isoformat(), "excluded": EXCLUDED,
                 "symbols": {}, "missing": {}})
    if manifest["since"] != since.isoformat():
        raise ValueError("existing manifest was fetched with another --from; use a new output directory")
    cutoff = date.fromisoformat(manifest["cutoffExclusive"])  # a resumed fetch keeps the snapshot's cutoff
    for symbol in symbols:
        path = out / f"{symbol}.json"
        entry = manifest["symbols"].get(symbol)
        if path.exists():
            if not entry or entry["sha256"] != _sha256(path):
                raise ValueError(f"{path} exists but does not match the manifest; refusing to continue")
            log(f"{symbol}: kept ({entry['bars']} bars)")
            continue
        try:
            stats: dict = {}
            records = fetch_symbol(symbol, since, cutoff, get, sleep, stats)
            anomalies = validate_records(symbol, records)
        except Exception as exc:  # recorded per symbol; one failure must not hide the others
            manifest["missing"][symbol] = f"{type(exc).__name__}: {exc}"
            log(f"{symbol}: FAILED {exc}")
            _write_manifest(manifest_path, manifest)
            continue
        if not records:
            manifest["missing"][symbol] = "provider returned no bars"
            log(f"{symbol}: no bars")
            _write_manifest(manifest_path, manifest)
            continue
        written = write_symbol_file(out, symbol, records, since.isoformat(), fetched_on)
        first, last = parse_record(records[0]).date, parse_record(records[-1]).date
        manifest["missing"].pop(symbol, None)
        manifest["symbols"][symbol] = {"bars": len(records), "first": first, "last": last,
                                       "anomalies": len(anomalies), "anomalyDates": anomalies[:20],
                                       "duplicatesCollapsed": stats["duplicatesCollapsed"],
                                       "sha256": _sha256(written)}
        _write_manifest(manifest_path, manifest)
        log(f"{symbol}: {len(records)} bars {first}..{last} anomalies={len(anomalies)}")
    return manifest


def _write_manifest(path: Path, manifest: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# ----------------------------------------------------------------------------------------- loaded series helpers

class RawSeries:
    """One symbol's raw daily bars, oldest first."""

    def __init__(self, symbol: str, bars: list[RawBar]):
        self.symbol = symbol
        self.bars = sorted(bars, key=lambda b: b.date)
        self.dates = [b.date for b in self.bars]
        if len(set(self.dates)) != len(self.dates):
            raise ValueError(f"{symbol}: duplicate session dates")
        self._index = {d: i for i, d in enumerate(self.dates)}

    @classmethod
    def from_records(cls, symbol: str, records: list[dict]) -> "RawSeries":
        return cls(symbol, [parse_record(r) for r in records])

    def bar(self, session: str) -> RawBar | None:
        i = self._index.get(session)
        return self.bars[i] if i is not None else None

    def previous(self, session: str) -> RawBar | None:
        i = self._index.get(session)
        if i is None:
            i = bisect_left(self.dates, session)
        return self.bars[i - 1] if i > 0 else None


def adjustment_ratio(bar: RawBar) -> float:
    """adjusted / raw close; constant between corporate actions."""
    return bar.adj_close / bar.close


def reference_price(series: RawSeries, session: str, basis: str = "previous-close") -> int | None:
    """Reference price of `session` in VND.

    basis "previous-close": the previous raw close in the series (the HOSE rule outside ex-right sessions).
    basis "exchange": the best estimate of the exchange's reference. Candidates, in order of preference:
      1. the provider's reference (close - change) when it differs from the previous raw close by more than
         ``reference_noise_vnd``: it carries the exchange's ex-right adjustment, bridges a session missing from the
         provider's history and corrects a provider close that differs from the official close. Within the noise
         the provider value is a rounding artefact of its 2-decimal change field and is ignored;
      2. when the provider's adjusted/raw ratio steps at this session (a corporate action the change field did
         not carry), the previous raw close scaled by the ratio step;
      3. the previous raw close.
    The first candidate whose +/-7% limits contain the session's raw low..high is returned (the change field is
    occasionally plain wrong, e.g. ACB 2021-08-25); if none does, the first candidate. Using the session's own
    range is not look-ahead: the exchange published the reference before the open; the range only identifies it.
    """
    if basis not in ("previous-close", "exchange"):
        raise ValueError(f"unknown basis {basis!r}")
    bar, prev = series.bar(session), series.previous(session)
    if basis == "previous-close" or bar is None:
        return prev.close if prev is not None else None
    cands = reference_candidates(bar, prev)
    for c in cands:
        floor, ceiling = price_limits(c)
        if bar.low > 0 and floor <= bar.low and bar.high <= ceiling:
            return c
    return cands[0] if cands else None


def reference_candidates(bar: RawBar, prev: RawBar | None) -> list[int]:
    """Candidate exchange references of `bar`, most preferred first (see ``reference_price``)."""
    out: list[int] = []
    pr = bar.provider_reference
    if pr is not None and (prev is None or abs(pr - prev.close) > reference_noise_vnd(prev.close)):
        out.append(pr)
    if prev is not None:
        if ratio_stepped(prev, bar):
            out.append(int(round(prev.close * adjustment_ratio(prev) / adjustment_ratio(bar))))
        out.append(prev.close)
    return out


def ratio_stepped(prev: RawBar, bar: RawBar) -> bool:
    """True when the provider's adjusted/raw ratio changes between two consecutive bars beyond its rounding."""
    if prev.adj_close <= 0 or bar.adj_close <= 0:
        return False
    tol = _cafef_tol(bar.adj_close) + _cafef_tol(prev.adj_close)
    return abs(adjustment_ratio(bar) / adjustment_ratio(prev) - 1) > tol


def reference_noise_vnd(price_vnd: float) -> float:
    """Largest provider-vs-previous-close gap treated as rounding noise: one tick or 0.1%, whichever is larger."""
    return max(hose_tick_size(price_vnd), REFERENCE_NOISE * price_vnd)


def band_fraction(series: RawSeries, session: str, market_sessions: list[str] | None = None) -> float:
    """+/-20% on the first bar of a series that starts at listing or after > 25 missed market sessions, else 7%.

    Without `market_sessions` (the union of sessions across symbols) only the 7% default can be inferred.
    The ex-right 20% cases need corporate-action data and are not inferred.
    """
    i = series._index.get(session)
    if i is None or market_sessions is None:
        return DEFAULT_PRICE_BAND_FRACTION
    if i == 0:
        return FIRST_DAY_BAND_FRACTION if market_sessions and market_sessions[0] < session else DEFAULT_PRICE_BAND_FRACTION
    prev = series.dates[i - 1]
    missed = bisect_left(market_sessions, session) - bisect_left(market_sessions, prev) - 1
    return FIRST_DAY_BAND_FRACTION if missed > LONG_SUSPENSION_SESSIONS else DEFAULT_PRICE_BAND_FRACTION


def limit_locked_open_bar(bar: RawBar, reference_vnd: int, side: Side,
                          band: float = DEFAULT_PRICE_BAND_FRACTION) -> bool:
    """True when a buy (sell) at the open cannot be assumed filled: the open printed at/above the ceiling
    (at/below the floor). An order queued at the limit may still partly fill; this flags the session, it does
    not model the queue."""
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    if bar.open <= 0:
        return True  # no opening print: nothing can be assumed filled at the open
    floor, ceiling = price_limits(reference_vnd, band)
    return bar.open >= ceiling if side == "buy" else bar.open <= floor


class RawPrices:
    """Raw snapshot directory (``<SYMBOL>.json`` + ``manifest.json``) with symbol-keyed helpers."""

    def __init__(self, directory: Path, verify: bool = True):
        self.directory = Path(directory)
        manifest_path = self.directory / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
        self.verify = verify
        self._series: dict[str, RawSeries] = {}

    def symbols(self) -> list[str]:
        return sorted(p.stem for p in self.directory.glob("*.json") if p.name != "manifest.json")

    def series(self, symbol: str) -> RawSeries:
        if symbol not in self._series:
            path = self.directory / f"{symbol}.json"
            if self.verify and self.manifest is not None:
                expected = self.manifest["symbols"].get(symbol, {}).get("sha256")
                if expected != _sha256(path):
                    raise ValueError(f"{path}: sha256 does not match the manifest")
            doc = json.loads(path.read_text(encoding="utf-8"))
            if doc.get("schema") != SCHEMA:
                raise ValueError(f"{path}: unexpected schema {doc.get('schema')!r}")
            self._series[symbol] = RawSeries.from_records(symbol, doc["records"])
        return self._series[symbol]

    def reference_price(self, symbol: str, session: str, basis: str = "previous-close") -> int | None:
        return reference_price(self.series(symbol), session, basis)

    def price_limits(self, symbol: str, session: str, basis: str = "exchange",
                     band: float = DEFAULT_PRICE_BAND_FRACTION) -> tuple[int, int] | None:
        ref = self.reference_price(symbol, session, basis)
        return price_limits(ref, band) if ref else None

    def limit_locked_open(self, symbol: str, session: str, side: Side, basis: str = "exchange",
                          band: float = DEFAULT_PRICE_BAND_FRACTION) -> bool | None:
        """None when unknown: no bar, no reference (first bar without a provider reference), or a provider open
        outside the session's low..high (the provider serves such opens on some dates, e.g. 2019-07-04)."""
        s = self.series(symbol)
        bar = s.bar(session)
        ref = reference_price(s, session, basis)
        if bar is None or ref is None or (bar.open > 0 and not bar.low <= bar.open <= bar.high):
            return None
        return limit_locked_open_bar(bar, ref, side, band)

    def traded_value(self, symbol: str, session: str, include_put_through: bool = False) -> int | None:
        """Traded value in VND (matched order book; + put-through when asked). None when there is no bar."""
        bar = self.series(symbol).bar(session)
        if bar is None:
            return None
        return bar.matched_value_vnd + (bar.put_through_value_vnd if include_put_through else 0)


# ------------------------------------------------------------------------------------------------- cross-checks

def ratio_steps(pairs: list[tuple[str, float, float]], rel_tol: Callable[[float], float]) -> dict[str, Any]:
    """Segment the adjusted/raw ratio of (date, raw_close, adjusted_close) rows into constant runs.

    A step is a day whose ratio differs from the previous day's by more than the rounding tolerance (a candidate
    corporate action). A *transient* is a one-day deviation that returns to the previous level the next day: the
    pattern of a data error, not of a corporate action. Returns steps and transients.
    """
    steps, transients = [], []
    for i in range(1, len(pairs)):
        d, raw, adj = pairs[i]
        _, praw, padj = pairs[i - 1]
        r, pr = adj / raw, padj / praw
        tol = rel_tol(adj) + rel_tol(padj)
        if abs(r / pr - 1) <= tol:
            continue
        if i + 1 < len(pairs):
            _, nraw, nadj = pairs[i + 1]
            nr = nadj / nraw
            if abs(nr / pr - 1) <= rel_tol(nadj) + rel_tol(padj):
                transients.append({"date": d, "ratioPrev": pr, "ratio": r, "ratioNext": nr})
                continue
        steps.append({"date": d, "ratioPrev": pr, "ratio": r, "change": r / pr - 1})
    # the day after a transient is its own return to level, not a step
    tdates = {t["date"] for t in transients}
    steps = [s for s in steps if pairs[[p[0] for p in pairs].index(s["date"]) - 1][0] not in tdates]
    return {"steps": steps, "transients": transients}


def load_adjusted_csv(path: Path) -> dict[str, dict[str, float]]:
    """VCI snapshot CSV (scripts/fetch_vn30_candles.py) -> date -> {open, high, low, close, volume}."""
    with path.open(encoding="utf-8") as f:
        return {r["date"]: {k: float(r[k]) for k in ("open", "high", "low", "close", "volume")}
                for r in csv.DictReader(f)}


def _cafef_tol(adj_vnd: float) -> float:
    return 5.0 / adj_vnd + 1e-6  # CafeF adjusted close has 0.01 thousand-VND (10 VND) resolution


def _vci_tol(adj_vnd: float) -> float:
    return 0.5 / adj_vnd + 5e-4  # VCI adjusted values are rounded; 0.05% floor covers its float noise


def crosscheck_symbol(series: RawSeries, adjusted: dict[str, dict[str, float]] | None,
                      market_sessions: list[str] | None = None,
                      in_vn30: Callable[[str], bool] | None = None) -> dict[str, Any]:
    """Data-quality report for one symbol. With `in_vn30` (session -> member?) the band check, limit-lock counts
    and the proxy comparison cover only sessions while the symbol was a VN30 member (HOSE-listed by construction;
    several members traded on HNX or UPCoM before, where other bands and ticks apply)."""
    bars = series.bars
    relevant = in_vn30 or (lambda d: True)
    out: dict[str, Any] = {"bars": len(bars), "first": bars[0].date if bars else None,
                           "last": bars[-1].date if bars else None}
    # 1. CafeF internal: raw close vs CafeF adjusted close
    internal = ratio_steps([(b.date, b.close, b.adj_close) for b in bars], _cafef_tol)
    out["cafefSteps"] = [s["date"] for s in internal["steps"]]
    out["cafefTransients"] = [t["date"] for t in internal["transients"]]
    # 2. where the exchange reference came from (VN30 member sessions only when `in_vn30` is given)
    source = {"provider": 0, "ratio": 0, "previous-close": 0}
    noise, rejected, provider_no_step, step_no_provider = 0, [], [], []
    for i in range(1, len(bars)):
        b, p = bars[i], bars[i - 1]
        if not relevant(b.date):
            continue
        cands = reference_candidates(b, p)
        ref = reference_price(series, b.date, "exchange")
        provider_dev = b.provider_reference is not None and cands[0] == b.provider_reference and len(cands) > 1 \
            and b.provider_reference != p.close
        stepped = ratio_stepped(p, b)
        if provider_dev and ref == b.provider_reference:
            source["provider"] += 1
            if not stepped:
                provider_no_step.append(b.date)
        elif stepped and ref != p.close:
            source["ratio"] += 1
        else:
            source["previous-close"] += 1
        if provider_dev and ref != b.provider_reference:
            rejected.append(b.date)
        if stepped and not provider_dev:
            step_no_provider.append(b.date)
        if b.provider_reference is not None and b.provider_reference != p.close and not provider_dev:
            noise += 1
    out["referenceSource"] = source
    out["providerReferenceNoiseSessions"] = noise
    out["providerReferenceRejected"] = rejected
    out["providerReferenceWithoutStep"] = provider_no_step
    out["stepsWithoutProviderReference"] = step_no_provider
    step_set = set(out["cafefSteps"])
    # 3a. rule check (not circular): on plain sessions - provider reference within noise of the previous close and
    #     no adjustment-ratio step - the raw high/low must lie within the +/-7% limits of the previous raw close,
    #     for each rounding rule; touches tell the rules apart.
    # 3b. sessions whose range no reference candidate explains (bad provider data or an unmodelled band).
    checks = {mode: {"violations": [], "highAtCeiling": 0, "lowAtFloor": 0} for mode in ROUNDING_MODES}
    locks = {"buy": 0, "sell": 0}
    outside, plain_n, unexplained = 0, 0, []
    for i in range(1, len(bars)):
        b, prev = bars[i], bars[i - 1]
        ref = reference_price(series, b.date, "exchange")
        band = band_fraction(series, b.date, market_sessions)
        floor, ceiling = price_limits(ref, band)
        fits = b.low > 0 and floor <= b.low and b.high <= ceiling
        if not relevant(b.date):
            outside += not fits
            continue
        if not fits:
            unexplained.append({"date": b.date, "reference": ref, "candidates": reference_candidates(b, prev),
                                "band": band, "floor": floor, "ceiling": ceiling, "low": b.low, "high": b.high})
        if b.open > 0 and b.low <= b.open <= b.high:
            locks["buy"] += b.open >= ceiling
            locks["sell"] += b.open <= floor
        if reference_candidates(b, prev) != [prev.close] or band != DEFAULT_PRICE_BAND_FRACTION:
            continue
        plain_n += 1
        for mode, c in checks.items():
            f, cl = price_limits(prev.close, band, mode)
            if b.high > cl or (b.low > 0 and b.low < f):
                c["violations"].append({"date": b.date, "reference": prev.close, "floor": f, "ceiling": cl,
                                        "low": b.low, "high": b.high})
            c["highAtCeiling"] += b.high == cl
            c["lowAtFloor"] += b.low == f
    out["checkedSessions"] = sum(1 for b in bars[1:] if relevant(b.date))
    out["plainSessions"] = plain_n
    out["unexplainedOutsideVn30"] = outside
    out["unexplainedSessions"] = unexplained
    out["bandCheck"] = checks
    out["bandViolations"] = checks[LIMIT_ROUNDING]["violations"]
    out["limitLockedOpens"] = locks
    # 4. raw vs VCI adjusted close (the research snapshot's basis)
    if adjusted is not None:
        common = [(b.date, b.close, adjusted[b.date]["close"]) for b in bars
                  if b.date in adjusted and adjusted[b.date]["close"] > 0]
        vci = ratio_steps(common, _vci_tol)
        vci_steps = {s["date"] for s in vci["steps"]}
        only_raw = sorted(d for d in series.dates if d not in adjusted)
        only_vci = sorted(d for d, r in adjusted.items() if d not in series._index and r["volume"] > 0
                          and bars and bars[0].date <= d <= bars[-1].date)
        out["vciCommonSessions"] = len(common)
        out["sessionsOnlyRaw"], out["sessionsOnlyRawCount"] = only_raw[:50], len(only_raw)
        out["sessionsOnlyVci"], out["sessionsOnlyVciCount"] = only_vci[:50], len(only_vci)
        out["vciSteps"] = sorted(vci_steps)
        out["vciTransients"] = [t["date"] for t in vci["transients"]]
        out["stepsOnlyVci"] = sorted(vci_steps - step_set)
        out["stepsOnlyCafef"] = sorted(step_set - vci_steps)
        # proxy of the frozen protocol (open >= 1.065 x previous adjusted close) vs raw limit-lock, same sessions
        agree = {"buy": [0, 0, 0, 0], "sell": [0, 0, 0, 0]}  # [both, proxyOnly, rawOnly, neither]
        adates = sorted(adjusted)
        for i in range(1, len(bars)):
            b = bars[i]
            if b.date not in adjusted or not 0 < b.low <= b.open <= b.high or not relevant(b.date):
                continue
            j = bisect_left(adates, b.date)
            if j == 0:
                continue
            pa = adjusted[adates[j - 1]]["close"]
            ao = adjusted[b.date]["open"]
            ref = reference_price(series, b.date, "exchange")
            floor, ceiling = price_limits(ref, band_fraction(series, b.date, market_sessions))
            for side, proxy, raw in (("buy", ao >= 1.065 * pa, b.open >= ceiling),
                                     ("sell", ao <= 0.935 * pa, b.open <= floor)):
                agree[side][0 if proxy and raw else 1 if proxy else 2 if raw else 3] += 1
        out["proxyVsRawLock"] = {s: dict(zip(("both", "proxyOnly", "rawOnly", "neither"), v))
                                 for s, v in agree.items()}
    return out


def run_crosscheck(raw_dir: Path, adjusted_dir: Path | None, membership: Path | None = None) -> dict[str, Any]:
    prices = RawPrices(raw_dir)
    symbols = prices.symbols()
    market = sorted({d for s in symbols for d in prices.series(s).dates})
    periods = None
    if membership is not None:
        from vn30_rotation import load_membership, members_on
        periods = load_membership(membership)
    report: dict[str, Any] = {"rawDir": str(raw_dir), "adjustedDir": str(adjusted_dir) if adjusted_dir else None,
                              "membership": str(membership) if membership else None,
                              "limitRounding": LIMIT_ROUNDING, "symbols": {}}
    for s in symbols:
        adj_path = adjusted_dir / f"{s}.csv" if adjusted_dir else None
        adjusted = load_adjusted_csv(adj_path) if adj_path and adj_path.exists() else None
        in_vn30 = None
        if periods is not None:
            in_vn30 = (lambda sym: lambda d: sym in (members_on(periods, d) or frozenset()))(s)
        report["symbols"][s] = crosscheck_symbol(prices.series(s), adjusted, market, in_vn30)
    syms = report["symbols"].values()
    report["totals"] = {
        "symbols": len(symbols),
        "bars": sum(v["bars"] for v in syms),
        "cafefSteps": sum(len(v["cafefSteps"]) for v in syms),
        "cafefTransients": sum(len(v["cafefTransients"]) for v in syms),
        "referenceSource": {k: sum(v["referenceSource"][k] for v in syms)
                            for k in ("provider", "ratio", "previous-close")},
        "providerReferenceRejected": sum(len(v["providerReferenceRejected"]) for v in syms),
        "providerReferenceWithoutStep": sum(len(v["providerReferenceWithoutStep"]) for v in syms),
        "stepsWithoutProviderReference": sum(len(v["stepsWithoutProviderReference"]) for v in syms),
        "checkedSessions": sum(v["checkedSessions"] for v in syms),
        "plainSessions": sum(v["plainSessions"] for v in syms),
        "bandViolations": sum(len(v["bandViolations"]) for v in syms),
        "unexplainedSessions": sum(len(v["unexplainedSessions"]) for v in syms),
        "unexplainedOutsideVn30": sum(v["unexplainedOutsideVn30"] for v in syms),
        "bandCheckByRounding": {m: {k: sum(len(v["bandCheck"][m]["violations"]) if k == "violations"
                                           else v["bandCheck"][m][k] for v in syms)
                                    for k in ("violations", "highAtCeiling", "lowAtFloor")}
                                for m in ROUNDING_MODES},
        "providerReferenceNoiseSessions": sum(v["providerReferenceNoiseSessions"] for v in syms),
        "limitLockedOpensBuy": sum(v["limitLockedOpens"]["buy"] for v in syms),
        "limitLockedOpensSell": sum(v["limitLockedOpens"]["sell"] for v in syms),
    }
    with_vci = [v for v in syms if "vciSteps" in v]
    if with_vci:
        report["totals"].update({
            "symbolsWithVci": len(with_vci),
            "vciSteps": sum(len(v["vciSteps"]) for v in with_vci),
            "vciTransients": sum(len(v["vciTransients"]) for v in with_vci),
            "stepsOnlyVci": sum(len(v["stepsOnlyVci"]) for v in with_vci),
            "stepsOnlyCafef": sum(len(v["stepsOnlyCafef"]) for v in with_vci),
            "sessionsOnlyRaw": sum(v["sessionsOnlyRawCount"] for v in with_vci),
            "sessionsOnlyVci": sum(v["sessionsOnlyVciCount"] for v in with_vci),
            "proxyVsRawLock": {side: {k: sum(v["proxyVsRawLock"][side][k] for v in with_vci)
                                      for k in ("both", "proxyOnly", "rawOnly", "neither")}
                               for side in ("buy", "sell")},
        })
    return report


# ---------------------------------------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--membership", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--from", dest="since", default="2016-01-01")
    f.add_argument("--symbols", default="", help="comma-separated subset (default: every ticker ever in VN30)")
    c = sub.add_parser("crosscheck")
    c.add_argument("--raw", type=Path, required=True)
    c.add_argument("--adjusted", type=Path)
    c.add_argument("--membership", type=Path, help="restrict the band/lock checks to VN30 member sessions")
    c.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "fetch":
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or tickers(args.membership)
        cutoff = fetch_cutoff(datetime.now(VN_TZ))
        manifest = run_fetch(symbols, args.out, date.fromisoformat(args.since), cutoff)
        print(f"fetched={len(manifest['symbols'])} missing={sorted(manifest['missing'])}")
        return 0
    report = run_crosscheck(args.raw, args.adjusted, args.membership)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["totals"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
