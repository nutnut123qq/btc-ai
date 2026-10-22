"""Document period + column-role resolution (pure).

Column roles come from the printed column headers ("Quý này/Năm nay", "Lũy kế từ đầu năm", explicit
"Từ dd/mm/yyyy đến dd/mm/yyyy", balance-sheet dates). Position is only a fallback inside a header group
(left = current, right = comparative — the Vietnamese statement convention), and ambiguities fail closed.

Period label conventions (stable, documented in metric_catalog.json "periodConventions"):
- flows, quarter column of a Qn report: "YYYYQn" quarter
- flows, cumulative column of a Qn report: "YYYY-3nM" ytd (n<4); Q4 cumulative (12M, unaudited): "YYYY" ytd
- flows, H1 report: "YYYYH1" half-year; FY report: "YYYY" annual
- balance sheet / note balances: point-in-time, label = quarter of the balance date ("2025Q4" for 31/12/2025)
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date

from .textnorm import find_dates, fold

CUM_WORDS = ("LUY KE", "TU DAU NAM", "DAU NAM DEN", "LUY KE TU")
QUARTER_WORDS = ("QUY", "TRONG KY", "PHAT SINH")
CUR_WORDS = ("NAM NAY", "KY NAY", "SO CUOI NAM", "SO CUOI KY", "CUOI KY")
COMP_WORDS = ("NAM TRUOC", "KY TRUOC", "SO DAU NAM", "SO DAU KY")


@dataclass(frozen=True)
class DocPeriod:
    kind: str  # Q | H | FY
    year: int
    n: int  # quarter 1-4, half 1-2, FY 0
    start: date
    end: date

    @property
    def label(self) -> str:
        return f"{self.year}Q{self.n}" if self.kind == "Q" else (f"{self.year}H{self.n}" if self.kind == "H" else str(self.year))


def doc_period(reporting_period: str | None) -> DocPeriod:
    rp = (reporting_period or "").strip().upper()
    m = re.fullmatch(r"(\d{4})(?:Q([1-4])|H([12]))?", rp)
    if not m:
        raise ValueError(f"unsupported ReportingPeriod {reporting_period!r}")
    y = int(m.group(1))
    if m.group(2):
        n = int(m.group(2))
        return DocPeriod("Q", y, n, date(y, 3 * n - 2, 1), month_end(y, 3 * n))
    if m.group(3):
        n = int(m.group(3))
        return DocPeriod("H", y, n, date(y, 1 if n == 1 else 7, 1), month_end(y, 6 if n == 1 else 12))
    return DocPeriod("FY", y, 0, date(y, 1, 1), date(y, 12, 31))


def month_end(y: int, m: int) -> date:
    return date(y, m, calendar.monthrange(y, m)[1])


def quarter_label(d: date) -> str:
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def _has(text: str, words) -> bool:
    return any(w in text for w in words)


def _dates(text: str) -> list[date]:
    out = []
    for y, m, d in find_dates(text):
        try:
            out.append(date(y, m, d))
        except ValueError:
            continue
    return sorted(set(out))


def assign_columns(headers: dict[str, str], ncols: int, statement: str, dp: DocPeriod) -> dict:
    """Map value columns to (role, period). statement: balance-sheet | note | income-statement | cash-flow.

    Returns {"columns": [spec|None,...], "errors": [...]} where spec has role/periodLabel/periodType/start/end.
    """
    rest = _restatement_layout(headers, ncols, statement, dp)
    if rest is not None:
        return rest
    if statement in ("balance-sheet", "note"):
        return _assign_point_in_time(headers, ncols, dp)
    return _assign_flows(headers, ncols, statement, dp)


PUBLISHED_WORDS = ("DA CONG BO", "CONG BO")
ADJUSTMENT_WORDS = ("DIEU CHINH",)


def _restatement_layout(headers: dict[str, str], ncols: int, statement: str, dp: DocPeriod) -> dict | None:
    """Restatement documents print published | adjustment | restated columns for the same period.

    Only the restated column becomes a fact (ColumnRole current, flagged restated); the published figure already
    lives in its original document and the adjustment column is a difference, not a period value.
    """
    folded = {int(k): fold(v) for k, v in headers.items() if int(k) < ncols}
    pub = [k for k, h in folded.items() if any(w in h for w in PUBLISHED_WORDS)]
    adj = [k for k, h in folded.items() if any(w in h for w in ADJUSTMENT_WORDS)]
    rest = [k for k, h in folded.items() if "TRINH BAY LAI" in h and k not in pub and k not in adj]
    if not (pub or adj) or len(rest) != 1:
        return None
    cols: list[dict | None] = [None] * ncols
    k = rest[0]
    raw = headers.get(str(k), "")
    if statement in ("balance-sheet", "note"):
        spec = _spec("current", quarter_label(dp.end), "point-in-time", None, dp.end, "restatement-layout", raw, True)
    elif dp.kind == "FY":
        spec = _spec("current", str(dp.year), "annual", dp.start, dp.end, "restatement-layout", raw, True)
    else:
        return {"columns": cols, "errors": [f"restatement layout in a non-annual report ({dp.label}) not supported"],
                "restatement": True}
    cols[k] = spec
    return {"columns": cols, "errors": [], "restatement": True,
            "skipped": {str(i): ("published" if i in pub else "adjustment" if i in adj else "unrecognised") for i in range(ncols) if i != k}}


def _spec(role, label, ptype, start, end, method, header, restated=False) -> dict:
    return {"role": role, "periodLabel": label, "periodType": ptype, "start": start.isoformat() if start else None,
            "end": end.isoformat(), "fiscalYear": end.year, "method": method, "header": header, "restated": restated}


def _restated(raw: str) -> bool:
    return "TRINH BAY LAI" in fold(raw) or "(*)" in (raw or "")


def _assign_point_in_time(headers: dict[str, str], ncols: int, dp: DocPeriod) -> dict:
    cols: list[dict | None] = [None] * ncols
    errors: list[str] = []
    for k in range(ncols):
        raw = headers.get(str(k), "")
        f = fold(raw)
        ds = [d for d in _dates(raw) if d <= dp.end]
        if ds:
            end = ds[-1]
            if end == dp.end:
                cols[k] = _spec("current", quarter_label(end), "point-in-time", None, end, "date", raw, _restated(raw))
            else:
                cols[k] = _spec("comparative", quarter_label(end), "point-in-time", None, end, "date", raw, _restated(raw))
        elif _has(f, CUR_WORDS) or "CUOI NAM" in f:
            cols[k] = _spec("current", quarter_label(dp.end), "point-in-time", None, dp.end, "words", raw, _restated(raw))
        elif _has(f, COMP_WORDS) or "DAU NAM" in f:
            end = date(dp.year - 1, 12, 31)
            cols[k] = _spec("comparative", quarter_label(end), "point-in-time", None, end, "words", raw, _restated(raw))
    # elimination: one column resolved, the other unreadable (e.g. "37/12/2025") -> the remaining role
    if ncols == 2 and (cols[0] is None) != (cols[1] is None):
        k = 0 if cols[0] is None else 1
        other = cols[1 - k]
        raw = headers.get(str(k), "")
        if other["role"] == "comparative" and k == 0:
            cols[0] = _spec("current", quarter_label(dp.end), "point-in-time", None, dp.end, "elimination", raw, _restated(raw))
        elif other["role"] == "current" and k == 1:
            end = date(dp.year - 1, 12, 31)
            cols[1] = _spec("comparative", quarter_label(end), "point-in-time", None, end, "elimination", raw, _restated(raw))
    # positional fallback for unresolved columns (left current, right prior year-end)
    if ncols == 2 and cols[0] is None and cols[1] is None:
        cols[0] = _spec("current", quarter_label(dp.end), "point-in-time", None, dp.end, "position", headers.get("0", ""))
        end = date(dp.year - 1, 12, 31)
        cols[1] = _spec("comparative", quarter_label(end), "point-in-time", None, end, "position", headers.get("1", ""),
                        _restated(headers.get("1", "")))
    for k, c in enumerate(cols):
        if c is None:
            errors.append(f"col{k}: unresolved balance date from header {headers.get(str(k), '')!r}")
    _check_unique(cols, errors)
    return {"columns": cols, "errors": errors}


def _assign_flows(headers: dict[str, str], ncols: int, statement: str, dp: DocPeriod) -> dict:
    errors: list[str] = []
    info = []
    for k in range(ncols):
        raw = headers.get(str(k), "")
        f = fold(raw)
        ds = _dates(raw)
        # a flow column can only carry dates inside the current or the prior year: anything else is an OCR misread
        # ("30/6/2026 30/8/2028" for "kết thúc ngày 30/6/2026"); drop it and say so
        implausible = [d for d in ds if not (date(dp.year - 1, 1, 1) <= d <= dp.end)]
        if len(implausible) == len(ds):  # nothing plausible left: a header of another period, fail closed below
            implausible = []
        ds = [d for d in ds if d not in implausible]
        rng = (ds[0], ds[-1]) if len(ds) == 2 else None
        end_only = ds[0] if len(ds) == 1 and implausible else None
        group = None
        if rng:
            months = (rng[1].year - rng[0].year) * 12 + rng[1].month - rng[0].month + 1
            if rng[0].month == 1 and rng[0].day == 1 and (months > 3 or _has(f, CUM_WORDS)):
                group = "cum"
            elif months == 3:
                group = "q"
        if group is None:
            if _has(f, CUM_WORDS):
                group = "cum"
            elif _has(f, QUARTER_WORDS) and dp.kind == "Q":
                group = "q"
        yrs = [int(y) for y in re.findall(r"NAM (\d{4})", f)]
        role = None
        if rng:
            role = "cur" if rng[1].year == dp.year else ("comp" if rng[1].year == dp.year - 1 else None)
            if role is None:
                errors.append(f"col{k}: header dates {rng[0]}..{rng[1]} not in {dp.label} or prior year")
        elif end_only and (end_only.month, end_only.day) == (dp.end.month, dp.end.day):
            role = "cur" if end_only.year == dp.year else "comp"  # the surviving period-end date still decides
        elif _has(f, CUR_WORDS):
            role = "cur"
        elif _has(f, COMP_WORDS):
            role = "comp"
        elif len(set(yrs)) == 1:
            role = "cur" if yrs[0] == dp.year else ("comp" if yrs[0] == dp.year - 1 else None)
        info.append({"group": group, "role": role, "range": rng, "raw": raw,
                     "implausibleDates": [d.isoformat() for d in implausible]})

    if dp.kind == "Q":
        if not any(i["group"] for i in info):
            if ncols == 4:
                for k, g in enumerate(("q", "q", "cum", "cum")):
                    info[k]["group"] = g
            elif ncols == 2:
                for i in info:
                    i["group"] = "cum" if statement == "cash-flow" else "q"
        else:  # columns without group words inherit the group of their header-group neighbour
            for k, i in enumerate(info):
                if i["group"] is None:
                    if ncols == 4:
                        i["group"] = "q" if k < 2 else "cum"
                    elif ncols == 2:
                        other = info[1 - k]["group"]
                        i["group"] = other or ("cum" if statement == "cash-flow" else "q")
    else:
        for i in info:
            i["group"] = "fy" if dp.kind == "FY" else "h"
    # role by position inside each group
    for g in {i["group"] for i in info}:
        members = [i for i in info if i["group"] == g]
        unresolved = [i for i in members if i["role"] is None]
        if len(members) == 2 and len(unresolved) == 2:
            members[0]["role"], members[1]["role"] = "cur", "comp"
        elif len(members) == 2 and len(unresolved) == 1:
            known = [i for i in members if i["role"]][0]["role"]
            unresolved[0]["role"] = "comp" if known == "cur" else "cur"
        elif len(members) == 1 and members[0]["role"] is None:
            members[0]["role"] = "cur"

    cols: list[dict | None] = []
    for k, i in enumerate(info):
        if i["group"] is None or i["role"] is None:
            errors.append(f"col{k}: unresolved group/role from header {i['raw']!r}")
            cols.append(None)
            continue
        ry = dp.year if i["role"] == "cur" else dp.year - 1
        method = "dates" if i["range"] else "words/position"
        if i["group"] == "q":
            start, end = date(ry, 3 * dp.n - 2, 1), month_end(ry, 3 * dp.n)
            spec = _spec("current" if i["role"] == "cur" else "comparative", f"{ry}Q{dp.n}", "quarter", start, end, method,
                         i["raw"], _restated(i["raw"]))
        elif i["group"] == "cum":
            start, end = date(ry, 1, 1), month_end(ry, 3 * dp.n)
            label = f"{ry}-{3 * dp.n}M" if dp.n < 4 else str(ry)
            spec = _spec("cumulative-current" if i["role"] == "cur" else "cumulative-comparative", label, "ytd", start, end,
                         method, i["raw"], _restated(i["raw"]))
        elif i["group"] == "h":
            start, end = (date(ry, 1, 1), date(ry, 6, 30)) if dp.n == 1 else (date(ry, 7, 1), date(ry, 12, 31))
            spec = _spec("current" if i["role"] == "cur" else "comparative", f"{ry}H{dp.n}", "half-year", start, end, method,
                         i["raw"], _restated(i["raw"]))
        else:
            start, end = date(ry, 1, 1), date(ry, 12, 31)
            spec = _spec("current" if i["role"] == "cur" else "comparative", str(ry), "annual", start, end, method,
                         i["raw"], _restated(i["raw"]))
        if i["implausibleDates"]:
            spec["implausibleDates"] = i["implausibleDates"]
            spec["method"] = f"{method}+implausible-dates-dropped"
        if i["range"] and (i["range"][0] != start or i["range"][1] != end):
            errors.append(f"col{k}: header dates {i['range'][0]}..{i['range'][1]} != expected {start}..{end}")
            cols.append(None)
            continue
        cols.append(spec)
    _check_unique(cols, errors)
    return {"columns": cols, "errors": errors}


def _check_unique(cols: list[dict | None], errors: list[str]) -> None:
    seen: dict[tuple, int] = {}
    for k, c in enumerate(cols):
        if c is None:
            continue
        key = (c["role"], c["periodLabel"], c["periodType"])
        if key in seen:
            errors.append(f"ambiguous columns {seen[key]} and {k} both map to {key}")
            cols[k] = None
            cols[seen[key]] = None
        else:
            seen[key] = k
