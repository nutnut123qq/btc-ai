"""Bounded collector for official issuer source documents (EQX v2 C1): BID (BIDV) and MBB (MB Bank).

Coverage target per symbol: the 4 most recent consolidated quarterly financial statements plus the
latest audited consolidated annual financial statements, taken only from the issuer's own investor
relations listing. Every document is stored as untrusted data under
``{root}/{SYMBOL}/{sha256}.{ext}`` (root = ``EQX_DOCUMENTS_ROOT`` or ``D:\\code\\btc\\data\\equity-documents``),
deduplicated by sha256, never executed. Text is read with the pure-Python ``pypdf`` reader only.

Commands (run with ``python -I`` because ``extract`` reads downloaded files)::

    python -I ai/equity_documents.py discover --symbol BID     # list IR page(s) -> catalog.json
    python -I ai/equity_documents.py fetch    --symbol BID     # download selected targets (resumable)
    python -I ai/equity_documents.py extract  --symbol BID     # pypdf text -> metrics with page provenance
    python -I ai/equity_documents.py manifest --symbol BID     # manifest.json in EquitySourceDocument shape
    python -I ai/equity_documents.py register --symbol BID --base-url http://localhost:5197  # needs EQX_ADMIN_KEY

Every HTTP request is appended to ``{root}/{SYMBOL}/requests.jsonl`` (url, status, bytes, sha256). A run
stops at ``--budget`` requests (default 40). Values are only ever copied from the document text with the
page number and the matched line; nothing is computed or inferred. Metrics not found are listed as
missing with a reason. NPL / CAR / COVERAGE are taken only when the document states the ratio together
with its definition text; otherwise they are reported missing.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
import time
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import unquote_plus, urljoin, urlparse

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

COLLECTOR_VERSION = "equity-documents-v1"
EXTRACTOR_VERSION = "equity-documents-extract-v1"
DEFAULT_ROOT = Path(os.environ.get("EQX_DOCUMENTS_ROOT", r"D:\code\btc\data\equity-documents"))
DEFAULT_BUDGET = 40
DEFAULT_DELAY_S = 1.5
MAX_DOCUMENT_BYTES = 80 * 1024 * 1024
QUARTERS_TARGET = 4
SOURCE = "issuer-report"
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/130.0.0.0 Safari/537.36")
DOC_NAMESPACE = uuid.UUID("6f1e2c43-7a52-4d0e-9a51-0f3c2b7e9d10")


@dataclass(frozen=True)
class Issuer:
    symbol: str
    entity: str
    provider: str
    listing_url: str
    allowed_hosts: frozenset[str]


ISSUERS: dict[str, Issuer] = {
    "BID": Issuer(
        symbol="BID",
        entity="Ngân hàng TMCP Đầu tư và Phát triển Việt Nam (BIDV)",
        provider="issuer:BIDV",
        listing_url="https://bidv.com.vn/vn/quan-he-nha-dau-tu/bao-cao-va-tai-lieu/",
        allowed_hosts=frozenset({"bidv.com.vn", "www.bidv.com.vn"}),
    ),
    "MBB": Issuer(
        symbol="MBB",
        entity="Ngân hàng TMCP Quân đội (MB)",
        provider="issuer:MB",
        listing_url="https://www.mbbank.com.vn/Investor/bao-cao-tai-chinh",
        allowed_hosts=frozenset({"mbbank.com.vn", "www.mbbank.com.vn"}),
    ),
}
MB_API = "https://www.mbbank.com.vn/api"
MB_CONSOLIDATED_CATEGORY_TITLE = "Báo cáo tài chính hợp nhất"

ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4}


# --------------------------------------------------------------------------------------------
# small helpers


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fold(text: str) -> str:
    """Lower-case and strip Vietnamese diacritics (đ -> d) for matching only; never for values."""
    text = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn").lower()


def document_id(symbol: str, sha256: str) -> str:
    """Deterministic id so reruns produce the same manifest (backend may assign its own)."""
    return str(uuid.uuid5(DOC_NAMESPACE, f"{symbol}:{sha256}"))


def symbol_dir(root: Path, symbol: str) -> Path:
    return root / symbol


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------------------------
# classification of listing titles (pure, tested offline)


def classify_title(title: str) -> dict[str, Any]:
    """Classify an IR listing title. Only what the title states; unknown otherwise."""
    f = fold(title)
    consolidation = "consolidated" if "hop nhat" in f else ("separate" if "rieng" in f else "unknown")
    if "kiem toan" in f and "chua kiem toan" not in f:
        audit = "audited"
    elif "soat xet" in f:
        audit = "reviewed"
    else:
        audit = "unknown"
    period: str | None = None
    kind = "other"
    q = re.search(r"quy\s+([ivx]+|\d)\s*(?:[./]|nam)?\s*(\d{4})", f)
    if q:
        token = q.group(1).upper()
        n = ROMAN.get(token) or (int(token) if token.isdigit() else None)
        if n in (1, 2, 3, 4):
            period, kind = f"{q.group(2)}Q{n}", "quarterly"
    elif re.search(r"6 thang|ban nien|sau thang|giua nien do \d{4} soat xet", f):
        kind = "semiannual"
    else:
        y = re.search(r"nam\s+(\d{4})|ngan hang\s+(\d{4})|nh\s+(\d{4})", f)
        if y and audit == "audited":
            period, kind = next(g for g in y.groups() if g), "annual"
    is_fs = "bao cao tai chinh" in f or "bctc" in f
    return {"period": period, "kind": kind, "consolidation": consolidation, "auditStatus": audit,
            "isFinancialStatement": is_fs}


def _period_key(period: str) -> tuple[int, int]:
    if "Q" in period:
        y, q = period.split("Q")
        return int(y), int(q)
    return int(period), 5


def select_targets(candidates: list[dict[str, Any]], quarters: int = QUARTERS_TARGET) -> list[dict[str, Any]]:
    """Latest `quarters` consolidated quarterly FS + latest audited consolidated annual FS."""
    fs = [c for c in candidates if c["isFinancialStatement"] and c["consolidation"] == "consolidated"]
    quarterly: dict[str, dict[str, Any]] = {}
    for c in fs:
        if c["kind"] == "quarterly" and c["period"] not in quarterly:
            quarterly[c["period"]] = c
    annual = [c for c in fs if c["kind"] == "annual" and c["auditStatus"] == "audited"]
    picked = sorted(quarterly.values(), key=lambda c: _period_key(c["period"]), reverse=True)[:quarters]
    if annual:
        picked.append(max(annual, key=lambda c: _period_key(c["period"])))
    return picked


# --------------------------------------------------------------------------------------------
# listing parsers (pure, tested offline)

_BIDV_ITEM = re.compile(
    r"var item = \{\s*type_pdf:[^\n]*\n\s*parenttitle: formatTitle\('(?P<parent>[^']*)'\),\s*"
    r"parentname: formatTitle\('[^']*'\),\s*title: formatTitle\('(?P<title>[^']*)'\),\s*"
    r"publishdate: '(?P<date>[^']*)',\s*file_title: formatTitle\('(?P<file>[^']*)'\),\s*"
    r"file_size: '(?P<size>[^']*)'")


def parse_bidv_listing(page_html: str, page_url: str) -> list[dict[str, Any]]:
    out = []
    for m in _BIDV_ITEM.finditer(page_html):
        title = html.unescape(m.group("title")).strip()
        href = html.unescape(m.group("file")).strip()
        if not href.lower().split("?")[0].endswith(".pdf"):
            continue
        d = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", m.group("date").strip())
        published = f"{d.group(3)}-{d.group(2)}-{d.group(1)}" if d else None
        out.append({
            "title": title, "sourceUrl": urljoin(page_url, href), "listingUrl": page_url,
            "listingGroup": m.group("parent"), "publishedDate": published,
            "publishedDateBasis": "issuer IR listing 'publishdate' (date only, ICT)" if published else None,
            "listedSize": m.group("size"), **classify_title(title),
        })
    return out


def parse_mb_listing(payload: dict[str, Any], page_url: str) -> list[dict[str, Any]]:
    out = []
    for x in payload.get("lst") or []:
        path = (x.get("file_path") or "").strip()
        title = (x.get("title") or "").strip()
        if not path.lower().endswith(".pdf") or not title:
            continue
        out.append({
            "title": title, "titleEn": x.get("title_en"), "sourceUrl": urljoin("https://www.mbbank.com.vn/", path),
            "listingUrl": page_url, "listingGroup": str(x.get("year")), "publishedDate": None,
            "publishedDateBasis": None,
            # CMS save timestamp of the listing entry; not a publication statement, kept for audit only.
            "listingLastSaveLocal": x.get("last_Save_Date"), "listedSize": x.get("file_zise"),
            **classify_title(title),
        })
    return out


# --------------------------------------------------------------------------------------------
# HTTP with budget + request log


class BudgetExceeded(RuntimeError):
    pass


class Http:
    def __init__(self, sdir: Path, issuer: Issuer, budget: int, delay_s: float, command: str,
                 session: Any = None):
        import requests  # local: tests construct Http with a fake session

        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": BROWSER_UA, "Accept-Language": "vi,en;q=0.8"})
        self.log_path = sdir / "requests.jsonl"
        self.issuer, self.budget, self.delay_s, self.command = issuer, budget, delay_s, command
        self.used = 0
        self._last = 0.0

    def _check_host(self, url: str) -> None:
        host = (urlparse(url).hostname or "").lower()
        if urlparse(url).scheme != "https" or host not in self.issuer.allowed_hosts:
            raise ValueError(f"refusing non-issuer URL: {url}")

    def log(self, **entry: Any) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"at": utc_now(), "command": self.command, **entry}, ensure_ascii=False) + "\n")

    def request(self, url: str, headers: dict[str, str] | None = None, stream: bool = False) -> Any:
        self._check_host(url)
        if self.used >= self.budget:
            raise BudgetExceeded(f"request budget {self.budget} reached")
        wait = self.delay_s - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self.used += 1
        self._last = time.monotonic()
        resp = self.session.get(url, headers=headers or {}, timeout=(15, 120), stream=stream,
                                allow_redirects=True)
        self._check_host(resp.url)
        return resp


# --------------------------------------------------------------------------------------------
# discover


def discover_bidv(http: Http) -> list[dict[str, Any]]:
    url = http.issuer.listing_url
    r = http.request(url)
    http.log(method="GET", url=url, status=r.status_code, bytes=len(r.content), contentType=r.headers.get("Content-Type"))
    r.raise_for_status()
    return parse_bidv_listing(r.text, url)


def discover_mb(http: Http, years: Iterable[int]) -> list[dict[str, Any]]:
    page = http.issuer.listing_url
    r = http.request(page)
    http.log(method="GET", url=page, status=r.status_code, bytes=len(r.content), contentType=r.headers.get("Content-Type"))
    r.raise_for_status()
    # Anti-forgery token the IR page itself embeds for its own XHR calls (public, per-session; not a credential).
    m = re.search(r'name="__RequestVerificationToken" type="hidden" value="([^"]+)"', r.text)
    headers = {"Referer": page, "X-Requested-With": "XMLHttpRequest"}
    if m:
        headers["MB-XSRF-Token-FormOnline"] = m.group(1)

    def api(path: str) -> Any:
        url = f"{MB_API}/{path}"
        for attempt in (1, 2):
            resp = http.request(url, headers=headers)
            http.log(method="GET", url=url, status=resp.status_code, bytes=len(resp.content),
                     contentType=resp.headers.get("Content-Type"), attempt=attempt)
            if resp.status_code == 403 and attempt == 1:
                time.sleep(3)  # the IR WAF intermittently rejects the first XHR; one polite retry
                continue
            resp.raise_for_status()
            return resp.json()
        return None

    cats = api("GetAttachmentCategoryFinal") or {}
    cat_id = next((c["id"] for c in cats.get("categories", []) if c.get("title") == MB_CONSOLIDATED_CATEGORY_TITLE), None)
    if cat_id is None:
        raise RuntimeError("MB IR: consolidated financial statements category not found")
    out: list[dict[str, Any]] = []
    for y in years:
        payload = api(f"GetListFinance/{cat_id}/1/{y}") or {}
        info = (payload.get("currentPageinfo") or [{}])[0]
        if int(info.get("totalPage") or 1) > 1:
            http.log(method="NOTE", url=f"{MB_API}/GetListFinance/{cat_id}/*/{y}", note="more than one page; only page 1 read")
        out.extend(parse_mb_listing(payload, f"{MB_API}/GetListFinance/{cat_id}/1/{y}"))
    return out


def cmd_discover(root: Path, symbol: str, budget: int, delay_s: float, session: Any = None) -> dict[str, Any]:
    issuer, sdir = ISSUERS[symbol], symbol_dir(root, symbol)
    sdir.mkdir(parents=True, exist_ok=True)
    http = Http(sdir, issuer, budget, delay_s, "discover", session)
    if symbol == "BID":
        cands = discover_bidv(http)
    else:
        this_year = datetime.now(UTC).year
        cands = discover_mb(http, [this_year, this_year - 1])
    targets = select_targets(cands)
    catalog = {"schemaVersion": COLLECTOR_VERSION, "symbol": symbol, "discoveredAtUtc": utc_now(),
               "listingUrl": issuer.listing_url, "requestsUsed": http.used,
               "candidates": cands, "targets": [t["sourceUrl"] for t in targets]}
    write_json(sdir / "catalog.json", catalog)
    return catalog


# --------------------------------------------------------------------------------------------
# fetch (resumable, dedup by sha256)


def cmd_fetch(root: Path, symbol: str, budget: int, delay_s: float, session: Any = None) -> dict[str, Any]:
    issuer, sdir = ISSUERS[symbol], symbol_dir(root, symbol)
    catalog = read_json(sdir / "catalog.json", None)
    if not catalog:
        raise SystemExit("run discover first")
    state = read_json(sdir / "documents.json", {"documents": []})
    by_url = {d["sourceUrl"]: d for d in state["documents"]}
    by_sha = {d["contentSha256"]: d for d in state["documents"]}
    http = Http(sdir, issuer, budget, delay_s, "fetch", session)
    cand_by_url = {c["sourceUrl"]: c for c in catalog["candidates"]}
    report = {"fetched": [], "skipped": [], "failed": []}
    for url in catalog["targets"]:
        prev = by_url.get(url)
        if prev and (root / prev["storagePath"]).exists():
            report["skipped"].append({"url": url, "reason": "already-stored", "sha256": prev["contentSha256"]})
            continue
        try:
            resp = http.request(url, stream=True)
        except BudgetExceeded:
            report["failed"].append({"url": url, "reason": "budget-exhausted"})
            break
        status, ctype = resp.status_code, resp.headers.get("Content-Type", "")
        if status != 200:
            http.log(method="GET", url=url, status=status, bytes=0, contentType=ctype)
            report["failed"].append({"url": url, "reason": f"http-{status}"})
            continue
        h, size = hashlib.sha256(), 0
        tmp = sdir / f".partial-{uuid.uuid4().hex}"
        try:
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(1 << 16):
                    size += len(chunk)
                    if size > MAX_DOCUMENT_BYTES:
                        raise ValueError("document exceeds size cap")
                    h.update(chunk)
                    fh.write(chunk)
            with tmp.open("rb") as fh:
                magic = fh.read(5)
            if magic != b"%PDF-":
                raise ValueError(f"not a PDF (magic {magic!r}, content-type {ctype})")
        except Exception as exc:  # noqa: BLE001 - logged and reported, file discarded
            tmp.unlink(missing_ok=True)
            http.log(method="GET", url=url, status=status, bytes=size, contentType=ctype, error=str(exc))
            report["failed"].append({"url": url, "reason": str(exc)})
            continue
        sha = h.hexdigest()
        rel = f"{symbol}/{sha}.pdf"
        dest = root / rel
        fetched_at = utc_now()
        http.log(method="GET", url=url, status=status, bytes=size, sha256=sha, contentType=ctype,
                 finalUrl=resp.url)
        if dest.exists():
            tmp.unlink()
        else:
            tmp.replace(dest)
        cand = cand_by_url[url]
        if sha in by_sha:  # same bytes already known under another URL: keep one document
            by_sha[sha].setdefault("alsoSeenAt", []).append({"sourceUrl": url, "fetchedAtUtc": fetched_at})
            report["skipped"].append({"url": url, "reason": "duplicate-content", "sha256": sha})
            continue
        older = [d for d in state["documents"] if d["sourceUrl"] == url]
        doc = {
            "documentId": document_id(symbol, sha), "sourceUrl": url, "contentSha256": sha,
            "storagePath": rel, "byteSize": size, "mimeType": "application/pdf",
            "httpContentType": ctype, "firstSeenAtUtc": catalog["discoveredAtUtc"], "fetchedAtUtc": fetched_at,
            "revision": len(older) + 1, "supersedesDocumentId": older[-1]["documentId"] if older else None,
            "listing": cand,
        }
        state["documents"].append(doc)
        by_url[url], by_sha[sha] = doc, doc
        report["fetched"].append({"url": url, "sha256": sha, "bytes": size})
        write_json(sdir / "documents.json", state)
    write_json(sdir / "documents.json", state)
    report["requestsUsed"] = http.used
    return report


# --------------------------------------------------------------------------------------------
# extraction (pure on page texts; tested offline with synthetic pages)

_NUM = r"\(?-?\d{1,3}(?:[.,]\d{3})+\)?|\(?-?\d+\)?"
_NUM_RE = re.compile(_NUM)


def parse_number(token: str) -> int | None:
    """'1.234.567' / '1,234,567' / '(1.234)' -> int. Returns None for anything ambiguous.

    Thousands groups may mix '.' and ',' because OCR text layers confuse them; a group that is not
    exactly 3 digits (a decimal such as '26.340,00') makes the token ambiguous -> None.
    """
    t = token.strip()
    neg = (t.startswith("(") and t.endswith(")")) or t.startswith("-")
    t = t.strip("()-")
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", t):
        v = int(re.sub(r"[.,]", "", t))
    elif re.fullmatch(r"\d+", t):
        v = int(t)
    else:
        return None
    return -v if neg else v


UNIT_PATTERNS = [
    # (folded regex, unit label as stated, multiplier to VND)
    (r"don vi\s*(?:tinh)?\s*:?\s*trieu\s*(?:dong|vnd)", "triệu đồng", 1_000_000),
    (r"don vi\s*(?:tinh)?\s*:?\s*ty\s*(?:dong|vnd)", "tỷ đồng", 1_000_000_000),
    (r"don vi\s*(?:tinh)?\s*:?\s*(?:dong|vnd)\b", "đồng", 1),
    (r"(?:unit|currency)\s*:?\s*(?:vnd\s*)?million", "VND million", 1_000_000),
    (r"(?:unit|currency)\s*:?\s*vnd\b", "VND", 1),
]


def detect_unit(page_text: str) -> tuple[str, int] | None:
    f = fold(page_text)
    for pat, label, mult in UNIT_PATTERNS:
        if re.search(pat, f):
            return label, mult
    return None


# Line labels (folded) per metric. A line must START with the label (after an optional code/roman prefix).
METRIC_LABELS: dict[str, list[str]] = {
    "TOTAL_ASSETS": [r"tong (?:cong )?tai san\b(?! (?:co|thue|ngan))", r"total assets\b"],
    "EQUITY": [r"(?:tong )?von chu so huu\b", r"total (?:owners'?|shareholders'?) equity\b"],
    "NET_PROFIT": [r"loi nhuan sau thue\b", r"net profit after tax\b", r"profit after tax\b"],
}
LINE_PREFIX = r"^\s*(?:[a-z]{1,4}[.)]?\s+|\d{1,3}[.)]?\s+)?"
BALANCE_SHEET_HINTS = [r"bang can doi ke toan", r"tinh hinh tai chinh", r"statement of financial position", r"balance sheet"]
INCOME_HINTS = [r"ket qua hoat dong kinh doanh", r"ket qua kinh doanh", r"income statement", r"profit or loss"]
RATIO_RULES: dict[str, dict[str, Any]] = {
    # metric: definition text that must be on the same page + the ratio line pattern
    "NPL": {"definition": r"(?:ty le )?no xau\s*(?:/|tren)\s*tong du no|nhom 3.{0,20}nhom 5|non-performing loans? ratio",
            "line": r"ty le no xau|npl ratio"},
    "CAR": {"definition": r"ty le an toan von|capital adequacy ratio",
            "line": r"ty le an toan von|capital adequacy ratio|\bcar\b"},
    "COVERAGE": {"definition": r"ty le bao phu no xau|du phong.{0,40}/\s*no xau|loan loss coverage",
                 "line": r"ty le bao phu|coverage ratio"},
}


def _first_numbers(line: str) -> list[int]:
    vals = []
    for tok in _NUM_RE.findall(line):
        v = parse_number(tok)
        if v is not None:
            vals.append(v)
    return vals


def _label_match(folded_line: str, patterns: list[str]) -> bool:
    return any(re.match(LINE_PREFIX + p, folded_line) for p in patterns)


def extract_metrics(pages: list[str], period: str | None, consolidation: str) -> dict[str, Any]:
    """Return {found: [...], missing: [...]} from page texts. Values are copied, never computed.

    For each metric the first page whose header matches the expected statement and whose unit line is
    explicit is used; the first numeric column on the labelled line is taken as the current period
    (Vietnamese bank statements list the current period first). The matched line is kept verbatim.
    """
    found: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for metric, labels in METRIC_LABELS.items():
        hints = INCOME_HINTS if metric == "NET_PROFIT" else BALANCE_SHEET_HINTS
        hit = None
        reasons: list[str] = []
        for pno, text in enumerate(pages, start=1):
            ft = fold(text)
            if not any(re.search(h, ft) for h in hints):
                continue
            unit = detect_unit(text)
            for raw in text.splitlines():
                fl = fold(raw)
                if not _label_match(fl, labels):
                    continue
                # strip leading code/note columns like "200" "IX" "V.12" so only amounts remain
                body = re.sub(LINE_PREFIX, "", fl)
                body = re.sub(r"^[^\d(]*", "", body)
                nums = [n for n in _first_numbers(body) if abs(n) >= 1000]
                if not nums:
                    reasons.append(f"p{pno}: label without amount")
                    continue
                if unit is None:
                    reasons.append(f"p{pno}: amount without explicit unit on page")
                    continue
                hit = (pno, raw.strip(), nums[0], unit)
                break
            if hit:
                break
        if hit:
            pno, line, val, (label, mult) = hit
            found.append({"metric": metric, "period": period, "valueOriginal": val, "unitOriginal": label,
                          "value": val * mult, "unit": "VND", "consolidation": consolidation,
                          "page": pno, "lineText": line[:300],
                          "sourceLocation": f"page {pno}: \"{line[:160]}\""})
        else:
            missing.append({"metric": metric, "reason": "; ".join(reasons[:3]) or
                            "label not found on a page with the expected statement header (text layer)"})
    for metric, rule in RATIO_RULES.items():
        hit = None
        for pno, text in enumerate(pages, start=1):
            ft = fold(text)
            if not re.search(rule["definition"], ft):
                continue
            for raw in text.splitlines():
                fl = fold(raw)
                if re.search(rule["line"], fl):
                    pm = re.search(r"(\d{1,3}(?:[.,]\d{1,4})?)\s*%", raw)
                    if pm:
                        hit = (pno, raw.strip(), pm.group(1))
                        break
            if hit:
                break
        if hit:
            pno, line, tok = hit
            d = re.search(rule["definition"], fold(pages[pno - 1]))
            found.append({"metric": metric, "period": period, "valueOriginal": tok, "unitOriginal": "%",
                          "value": float(tok.replace(",", ".")), "unit": "percent", "consolidation": consolidation,
                          "page": pno, "lineText": line[:300],
                          "definitionText": fold(pages[pno - 1])[max(0, d.start() - 60):d.end() + 120],
                          "sourceLocation": f"page {pno}: \"{line[:160]}\""})
        else:
            missing.append({"metric": metric,
                            "reason": "ratio with an explicit definition not stated in the document text layer"})
    return {"found": found, "missing": missing}


# --- OCR'd income statements (labels and amount columns come out as separate blocks) -----------
#
# The issuer PDFs are scans; a few statement pages carry an OCR text layer in which the label column
# and each amount column are emitted as separate blocks, so a label and its amount are never on the
# same line. The amount is located positionally and the position is ACCEPTED only when the
# statement's own printed totals agree exactly at that position:
#   profit before tax + total CIT expense == profit after tax
#   profit after tax - non-controlling interests == profit after tax attributable to the bank
# These identities only validate the alignment; the stored value is the printed token, copied.

_TOKEN_RE = re.compile(r"\(?-?\d{1,3}(?:[.,]\d{3})+\)?|\(?-?\d+\)?")
OCR_INCOME_HEADER = r"k.{1,3}t qua ho.{1,8}ng\s?h.{1,3}p nh"
OCR_QUARTER_COLUMN = r"s.{1,2} phat sinh quy"
OCR_CURRENT_YEAR = r"n_?a_?m_?\s?n_?a"  # "nam nay"; OCR also gives "N_a_m_na....:cy"
OCR_UNIT_MILLION = r"tri.{1,3}u\s*\S{0,3}ng\b"
OCR_NPAT_LABEL = r"^l.{1,4}\s?nhu.{1,4}n sau thu\S{0,3}\s*$"


def _amount_line_tokens(raw: str) -> list[str] | None:
    """Amount tokens of a line that holds only amounts (else None)."""
    s = raw.replace("{", "(").replace("}", ")")
    if "/" in s or ":" in s:  # dates / times / page refs are not amounts
        return None
    s = re.sub(r"\s*([.,])\s*(?=\d)", r"\1", s)
    toks = _TOKEN_RE.findall(s)
    rest = _TOKEN_RE.sub("", s)
    if not toks or re.search(r"[A-Za-zÀ-ỹ]", rest):
        return None
    return toks


def extract_ocr_quarter_net_profit(pages: list[str], period: str | None) -> tuple[dict[str, Any] | None, str]:
    """NET_PROFIT (consolidated profit after tax, current quarter) from an OCR'd income statement page."""
    if not period or not re.fullmatch(r"\d{4}Q[1-4]", period):
        return None, "document period is not a quarter"
    reasons = []
    for pno, text in enumerate(pages, start=1):
        ft = fold(text)
        if not re.search(OCR_INCOME_HEADER, ft):
            continue
        gates = {"quarter-column header": OCR_QUARTER_COLUMN, "current-year header": OCR_CURRENT_YEAR,
                 "million-VND unit": OCR_UNIT_MILLION}
        failed = [name for name, pat in gates.items() if not re.search(pat, ft)]
        lines = text.splitlines()
        label = next((ln.strip() for ln in lines if re.match(OCR_NPAT_LABEL, fold(ln).strip())), None)
        if label is None:
            failed.append("profit-after-tax label")
        if failed:
            reasons.append(f"p{pno}: missing {', '.join(failed)}")
            continue
        stream: list[tuple[int, str, str]] = []  # (value, token, raw line); first amount of each amount-only line
        for ln in lines:
            toks = _amount_line_tokens(ln)
            if toks:
                v = parse_number(toks[0].replace("{", "(").replace("}", ")"))
                if v is not None:
                    stream.append((v, toks[0], ln.strip()))
        vals = [v for v, _, _ in stream]
        # A column opens with two printed triples: interest income + interest expense = net interest income,
        # then fee income + fee expense = net fee income. Column 1 is the first such opening in the stream.
        def triple(s: int) -> bool:
            return s + 2 < len(vals) and vals[s] > 0 > vals[s + 1] and vals[s] + vals[s + 1] == vals[s + 2]

        tops = [s for s in range(len(vals)) if triple(s) and triple(s + 3)]
        if not tops:
            reasons.append(f"p{pno}: no amount column opens with the interest and fee subtotal rows")
            continue
        start = tops[0]
        for j in range(start + 3, len(vals) - 3):
            tax, npat, nci, parent = vals[j:j + 4]
            if not (tax < 0 < npat and npat - nci == parent):
                continue
            pbt_k = next((k for k in range(1, 5) if vals[j - k] + tax == npat and vals[j - k] > npat), None)
            if pbt_k is None:
                continue
            if any(start < s < j for s in tops):
                # a second column began before column 1 closed: column 1's bottom rows are unreadable
                reasons.append(f"p{pno}: column 1 bottom rows unreadable (next column starts before a valid match)")
                break
            unit_m = re.search(OCR_UNIT_MILLION, ft)
            return {
                "metric": "NET_PROFIT", "period": period, "valueOriginal": npat,
                "unitOriginal": f"triệu đồng (OCR text '{unit_m.group(0)}')", "value": npat * 1_000_000, "unit": "VND",
                "definition": "consolidated profit after tax incl. non-controlling interests, current quarter "
                              "(column 'Số phát sinh quý – Năm nay')",
                "page": pno, "lineText": stream[j + 1][2], "labelText": label,
                "method": "ocr-column-alignment",
                "alignmentCheck": {"profitBeforeTax": vals[j - pbt_k], "totalCitExpense": tax, "profitAfterTax": npat,
                                   "nonControllingInterests": nci, "profitAfterTaxAttributableToBank": parent},
                "sourceLocation": (f"page {pno}: consolidated income statement, column 'Số phát sinh quý – Năm nay', "
                                   f"row 'LỢI NHUẬN SAU THUẾ' = '{stream[j + 1][1]}' (OCR text layer; label read as '{label[:40]}')"),
            }, ""
        else:
            reasons.append(f"p{pno}: no amount position satisfies PBT+CIT=PAT and PAT-NCI=PAT(bank)")
    return None, "; ".join(reasons) or "no page with an OCR'd consolidated income statement header"


def read_pdf_pages(path: Path) -> tuple[list[str], dict[str, Any]]:
    from pypdf import PdfReader  # pure-Python reader; no rendering, no JS, no external tools

    reader = PdfReader(str(path), strict=False)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception as exc:  # noqa: BLE001 - per-page failure is recorded, not fatal
            pages.append(f"\x00extract-error: {exc}")
    chars = sum(len(t.strip()) for t in pages)
    info = {"tool": "pypdf", "toolVersion": __import__("pypdf").__version__, "pageCount": len(pages),
            "textChars": chars, "pagesWithText": sum(1 for t in pages if len(t.strip()) > 40)}
    return pages, info


def detect_audit_statement(pages: list[str]) -> dict[str, Any] | None:
    """Audit status only when the document says so on its first pages."""
    for pno, text in enumerate(pages[:6], start=1):
        f = fold(text)
        for pat, status in ((r"chua (?:duoc )?kiem toan|unaudited", "unaudited"),
                            (r"bao cao kiem toan doc lap|independent auditors?'? report", "audited"),
                            (r"bao cao (?:ket qua )?(?:cong tac )?soat xet|review report", "reviewed")):
            m = re.search(pat, f)
            if m:
                return {"status": status, "page": pno, "text": f[max(0, m.start() - 40):m.end() + 60]}
    return None


def cmd_extract(root: Path, symbol: str) -> dict[str, Any]:
    sdir = symbol_dir(root, symbol)
    state = read_json(sdir / "documents.json", {"documents": []})
    out = {}
    for doc in state["documents"]:
        path = root / doc["storagePath"]
        pages, info = read_pdf_pages(path)
        listing = doc["listing"]
        audit = detect_audit_statement(pages)
        if info["pagesWithText"] == 0:
            result = {"found": [], "missing": [{"metric": m, "reason": "no text layer on any page (scanned PDF)"}
                                               for m in [*METRIC_LABELS, *RATIO_RULES]]}
        else:
            result = extract_metrics(pages, listing["period"], listing["consolidation"])
            if not any(f["metric"] == "NET_PROFIT" for f in result["found"]) and listing["consolidation"] == "consolidated":
                hit, why = extract_ocr_quarter_net_profit(pages, listing["period"])
                np_missing = next(m for m in result["missing"] if m["metric"] == "NET_PROFIT")
                if hit:
                    hit["consolidation"] = listing["consolidation"]
                    result["found"].append(hit)
                    result["missing"].remove(np_missing)
                else:
                    np_missing["reason"] += f"; OCR-column fallback: {why}"
        text_pages = [i + 1 for i, t in enumerate(pages) if len(t.strip()) > 40]
        for m in result["missing"]:
            m["reason"] += f" [text layer on pages {text_pages} of {len(pages)}; other pages are scanned images, OCR not run]"
        doc["extraction"] = {"extractorVersion": EXTRACTOR_VERSION, "extractedAtUtc": utc_now(), **info,
                             "auditStatement": audit, **result}
        out[doc["contentSha256"]] = {"found": len(result["found"]), "missing": [m["metric"] for m in result["missing"]],
                                     **{k: info[k] for k in ("pageCount", "pagesWithText")}}
    write_json(sdir / "documents.json", state)
    return out


# --------------------------------------------------------------------------------------------
# manifest + register


def published_at_from_date(day: str | None) -> str | None:
    """Issuer-stated date-only publication (yyyy-MM-dd, ICT) -> 00:00 ICT of the next day, as UTC ISO.

    A same-day time is not stated, so the document is treated as available only once that day is over.
    """
    if not day:
        return None
    d = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC)
    # 00:00 ICT (UTC+7) on day+1 == 17:00 UTC on day.
    return d.replace(hour=17).isoformat().replace("+00:00", "Z")


def _doc_type(listing: dict[str, Any]) -> str:
    return "financial-statement" if listing.get("isFinancialStatement") else "other"


def build_manifest(symbol: str, state: dict[str, Any]) -> dict[str, Any]:
    issuer = ISSUERS[symbol]
    docs, fundamentals = [], []
    for d in state["documents"]:
        lst, ext = d["listing"], d.get("extraction")
        audit = lst["auditStatus"]
        if audit == "unknown" and ext and ext.get("auditStatement"):
            audit = ext["auditStatement"]["status"]
        extraction_json = None
        if ext:
            extraction_json = {k: ext[k] for k in ext if k not in ("found",)}
            extraction_json["found"] = [{k: v for k, v in f.items()} for f in ext["found"]]
            extraction_json["listing"] = {k: lst.get(k) for k in ("title", "titleEn", "listingUrl", "listingGroup",
                                                                  "publishedDate", "publishedDateBasis",
                                                                  "listingLastSaveLocal", "listedSize")}
        docs.append({
            "DocumentId": d["documentId"], "Symbol": symbol, "Entity": issuer.entity, "DocType": _doc_type(lst),
            "Title": lst["title"], "SourceUrl": d["sourceUrl"], "Provider": issuer.provider,
            "ContentSha256": d["contentSha256"], "StoragePath": d["storagePath"], "ByteSize": d["byteSize"],
            "MimeType": d["mimeType"], "ReportingPeriod": lst["period"], "Consolidation": lst["consolidation"],
            "AuditStatus": audit,
            # Only a date is stated by the IR listing (BIDV): availability is 00:00 ICT of the NEXT day (the
            # backend's date-only convention). Registration sends PublishedDate; the backend derives the same
            # PublishedAtUtc. MB states no date -> both null (availability = first-seen).
            "PublishedAtUtc": published_at_from_date(lst.get("publishedDate")), "PublishedDate": lst.get("publishedDate"),
            "PublishedAtBasis": ((lst.get("publishedDateBasis") or "issuer-stated date") + " -> 00:00 ICT next day") if lst.get("publishedDate") else None,
            "FirstSeenAtUtc": d["firstSeenAtUtc"], "FetchedAtUtc": d["fetchedAtUtc"], "Revision": d["revision"],
            "SupersedesDocumentId": d["supersedesDocumentId"], "ExtractionJson": extraction_json,
        })
        for f in (ext or {}).get("found", []):
            period = f["period"]
            fundamentals.append({
                "period": period, "metric": f["metric"], "value": f["value"], "unit": f["unit"],
                "unitOriginal": f["unitOriginal"], "valueOriginal": f["valueOriginal"],
                "publishedDate": lst.get("publishedDate"), "source": SOURCE, "sourceUrl": d["sourceUrl"],
                "documentId": d["documentId"], "contentSha256": d["contentSha256"],
                "sourceLocation": f["sourceLocation"], "consolidation": f["consolidation"],
                "definition": f.get("definition"),
                # The admin import accepts only YYYYQn periods; annual figures (period "YYYY") wait for a contract.
                "importable": bool(period and re.fullmatch(r"\d{4}Q[1-4]", period)),
            })
    return {"schemaVersion": "equity-source-documents-manifest-v1", "symbol": symbol, "generatedAtUtc": utc_now(),
            "collectorVersion": COLLECTOR_VERSION, "documents": docs, "fundamentals": fundamentals}


def cmd_manifest(root: Path, symbol: str) -> dict[str, Any]:
    sdir = symbol_dir(root, symbol)
    manifest = build_manifest(symbol, read_json(sdir / "documents.json", {"documents": []}))
    write_json(sdir / "manifest.json", manifest)
    return manifest


REGISTRATION_FIELDS = ("Entity", "DocType", "Title", "SourceUrl", "Provider", "ContentSha256", "StoragePath", "MimeType",
                       "ReportingPeriod", "Consolidation", "AuditStatus", "FirstSeenAtUtc", "FetchedAtUtc")


def register_payload(doc: dict[str, Any]) -> dict[str, Any]:
    """Body for POST /api/equities/{symbol}/admin/source-documents (EquitySourceDocumentRegistration).

    The backend assigns DocumentId/Revision itself and re-hashes the file under its RootPath; the
    extraction provenance travels as a JSON object (`extraction`). Local supersedes ids are not sent
    (they are not backend ids); every document here is revision 1.
    """
    body = {k[0].lower() + k[1:]: doc[k] for k in REGISTRATION_FIELDS}
    body["extraction"] = doc.get("ExtractionJson")
    # Date-only issuer statement: send publishedDate (the backend converts it, write-once; it rejects
    # publishedDate + publishedAtUtc together). Re-registering an existing document fills it once.
    if doc.get("PublishedDate"):
        body["publishedDate"] = doc["PublishedDate"]
    return body


def fundamentals_payload(manifest: dict[str, Any], registered: dict[str, Any]) -> list[dict[str, Any]]:
    """Batch body for POST /api/equities/{symbol}/admin/fundamentals (EquityFundamentalImport + C1 fields).

    `documentId` must be the BACKEND id of the registered document (from registered.json), so a row is
    only importable once its document is registered.
    """
    rows = []
    for f in manifest["fundamentals"]:
        reg = registered.get(f["contentSha256"])
        if not f["importable"] or not reg:
            continue
        rows.append({"period": f["period"], "metric": f["metric"], "value": f["value"], "unit": f["unit"],
                     "unavailableReason": None, "publishedDate": f["publishedDate"], "source": f["source"],
                     "sourceUrl": f["sourceUrl"], "documentId": reg["documentId"], "sourceLocation": f["sourceLocation"]})
    return rows


def cmd_register(root: Path, symbol: str, base_url: str, what: str = "documents",
                 poster: Callable[..., Any] | None = None) -> list[dict[str, Any]]:
    key = os.environ.get("EQX_ADMIN_KEY")
    if not key:
        raise SystemExit("EQX_ADMIN_KEY is not set")
    sdir = symbol_dir(root, symbol)
    manifest = read_json(sdir / "manifest.json", None)
    if not manifest:
        raise SystemExit("run manifest first")
    if poster is None:
        import requests

        poster = requests.post
    headers = {"X-Admin-Key": key}
    base = f"{base_url.rstrip('/')}/api/equities/{symbol}/admin"
    registered = read_json(sdir / "registered.json", {})
    if what == "fundamentals":
        body = fundamentals_payload(manifest, registered)
        if not body:
            return [{"status": None, "body": "no importable fundamentals with a registered document"}]
        r = poster(f"{base}/fundamentals", json=body, headers=headers, timeout=60)
        return [{"rows": len(body), "status": r.status_code, "body": r.text[:500]}]
    results = []
    for doc in manifest["documents"]:
        r = poster(f"{base}/source-documents", json=register_payload(doc), headers=headers, timeout=60)
        entry: dict[str, Any] = {"sha256": doc["ContentSha256"], "status": r.status_code}
        if r.status_code in (200, 201):
            item = (r.json() or {}).get("item") or {}
            payload = r.json() or {}
            entry.update(documentId=item.get("documentId"), created=r.status_code == 201,
                         publishedAtUtc=item.get("publishedAtUtc"), publishedAtFilled=bool(payload.get("publishedAtFilled")))
            prior = registered.get(doc["ContentSha256"], {})
            registered[doc["ContentSha256"]] = {**prior, "documentId": item.get("documentId"), "backend": base_url,
                                                "registeredAtUtc": prior.get("registeredAtUtc") or utc_now(),
                                                "created": prior.get("created", r.status_code == 201),
                                                "publishedAtUtc": item.get("publishedAtUtc"),
                                                "lastRegisterAtUtc": utc_now()}
        else:
            entry["body"] = r.text[:300]
        results.append(entry)
        if r.status_code in (401, 404):
            break  # wrong key, or endpoint not available yet: do not hammer it
    write_json(sdir / "registered.json", registered)
    return results


# ============================================================================================
# v3 (spec-v3-document-pipeline.md §6): full-history discovery, classification and download.
# Pure parsing/classification plus a polite HTTP client; persistence lives in ai/equity_pipeline.py.
# The legacy v1 functions above stay untouched so the old CLI commands keep working.
# ============================================================================================

V3_COLLECTOR_VERSION = "equity-documents-v3"
BIDV_LIBRARY_VI = "https://bidv.com.vn/vn/quan-he-nha-dau-tu/bao-cao-va-tai-lieu/"
BIDV_DISCLOSURE_FEED = "https://bidv.com.vn/vn/quan-he-nha-dau-tu/thong-tin-co-dong/"
MB_ORIGIN = "https://www.mbbank.com.vn"
MB_FS_PAGE = f"{MB_ORIGIN}/Investor/bao-cao-tai-chinh"
MB_NOTICE_PAGE = f"{MB_ORIGIN}/Investor/thong-bao-nha-dau-tu/{{year}}/0//0"
MB_FS_CATEGORIES = {7: "consolidated", 13: "separate"}  # GetAttachmentCategoryFinal ids (A0, 2026-10-19)
ICT_OFFSET_H = 7


def _js_field(body: str, key: str) -> str | None:
    # (?<!...) so that "title:" does not match inside "parenttitle:".
    m = re.search(r"(?<![A-Za-z_])" + key + r":\s*(?:formatTitle\()?'(.*?)'\)?\s*,?\s*\n", body)
    return html.unescape(m.group(1)).strip() if m else None


def parse_bidv_items(page_html: str, page_url: str) -> list[dict[str, Any]]:
    """Every inline `var item = {...}` of a BIDV IR page (library or disclosure feed), PDF or not."""
    out = []
    for body in re.findall(r"var item = \{(.*?)\};", page_html, re.S):
        title = _js_field(body + "\n", "title")
        if not title:
            continue
        href = _js_field(body + "\n", "file_title") or ""
        d = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", (_js_field(body + "\n", "publishdate") or "").strip())
        path_only = href.split("?")[0]
        uuid_m = re.search(r"/connect/([0-9a-fA-F-]{36})/", href)
        cache_m = re.search(r"CACHEID=ROOTWORKSPACE-[0-9a-fA-F-]{36}-([A-Za-z0-9_-]+)", href)
        out.append({
            "title": title, "url": urljoin(page_url, href) if href else None, "listingUrl": page_url,
            "listedDate": f"{d.group(3)}-{d.group(2)}-{d.group(1)}" if d else None,
            "listedSize": _js_field(body + "\n", "file_size") or None,
            "group": _js_field(body + "\n", "parentname"), "contentPath": _js_field(body + "\n", "path"),
            "fileName": path_only.rsplit("/", 1)[-1] if path_only else "",
            "fileType": path_only.rsplit(".", 1)[-1].lower() if "." in path_only.rsplit("/", 1)[-1] else None,
            "identity": f"bidv-wcm:{uuid_m.group(1).lower()}" if uuid_m else None,
            "versionToken": cache_m.group(1) if cache_m else None,
        })
    return out


def parse_mb_finance_page(payload: dict[str, Any], category: int) -> tuple[list[dict[str, Any]], int]:
    """One GetListFinance/{category}/{page}/0 page -> (items, totalPage)."""
    items = []
    for x in payload.get("lst") or []:
        path = (x.get("file_path") or "").strip()
        fname = path.rsplit("/", 1)[-1]
        pref = re.match(r"(20\d{2})(\d{2})(\d{2})---", fname)
        items.append({
            "title": re.sub(r"\s+", " ", (x.get("title") or "")).strip(), "url": urljoin(MB_ORIGIN + "/", path) if path else None,
            "listingUrl": f"{MB_ORIGIN}/api/GetListFinance/{category}/{{page}}/0", "listedDate": None,
            "listedSize": x.get("file_zise"), "group": str(x.get("year")), "category": category,
            "mbId": x.get("id"), "cmsLastSaveLocal": x.get("last_Save_Date"), "fileName": fname,
            "fileType": (x.get("file_type") or fname.rsplit(".", 1)[-1]).lower() if fname else None,
            "filenameDatePrefix": f"{pref.group(1)}-{pref.group(2)}-{pref.group(3)}" if pref else None,
            "identity": f"url:{urljoin(MB_ORIGIN + '/', path)}" if path else None, "versionToken": None,
        })
    info = (payload.get("currentPageinfo") or [{}])[0]
    return items, int(info.get("totalPage") or 1)


def parse_mb_notices(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """GetListMessage/{page}/{year} -> (notices, totalPage). Only `created`/`start_date` are dates the CMS
    states for the post; `last_save_date` is an edit time and is never used."""
    out, seen = [], set()
    for x in payload.get("otherNews") or []:
        if not isinstance(x, dict) or x.get("id") in seen:
            continue
        seen.add(x.get("id"))
        out.append({"id": str(x.get("id")), "title": re.sub(r"\s+", " ", x.get("title") or "").strip(),
                    "createdLocal": x.get("created") or x.get("start_date")})
    info = (payload.get("currentPageinfo") or [{}])[0]
    return out, int(info.get("totalPage") or 1)


_ORDINAL = {"first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th"}


def classify_v3(title: str, file_name: str = "", category_consolidation: str | None = None) -> dict[str, Any]:
    """Classify an issuer listing title (v3). States only what the title says; filename / MB category are
    fallbacks for consolidation and are recorded in `basis`. Non-statements get a `skipReason`."""
    f = fold(title)
    fn = fold(unquote_plus(file_name or "")).replace("-", " ").replace("_", " ")
    out: dict[str, Any] = {"periodType": None, "fiscalYear": None, "reportingPeriod": None, "consolidation": "unknown",
                           "covers": [], "auditStatus": "unknown", "isFinancialStatement": False, "restatement": False,
                           "skipReason": None, "basis": {}}
    is_fs = bool(re.search(r"bao cao tai chinh|\bbctc\b|financial statement", f))
    if re.search(r"giai trinh|explanation", f):
        out["skipReason"] = "profit-variance explanation letter (giải trình), not a financial statement"
        return out
    if not is_fs:
        out["skipReason"] = "title does not name a financial statement"
        return out
    out["restatement"] = bool(re.search(r"dieu chinh (?:bao cao tai chinh|bctc)|bo sung (?:bao cao tai chinh|bctc)|restat", f))
    has_c = bool(re.search(r"hop nhat|consolidat", f))
    has_s = bool(re.search(r"\brieng\b|separate|seperate|cong ty me", f))
    if has_c and has_s:
        out["covers"] = ["consolidated", "separate"]
    elif has_c or has_s:
        out["consolidation"] = "consolidated" if has_c else "separate"
        out["covers"] = [out["consolidation"]]
    elif re.search(r"\bhn\b|hop ?nhat|consol", fn):
        out["consolidation"], out["covers"], out["basis"]["consolidation"] = "consolidated", ["consolidated"], "filename"
    elif re.search(r"\brl\b|rieng|separate|cong ?ty ?me", fn):
        out["consolidation"], out["covers"], out["basis"]["consolidation"] = "separate", ["separate"], "filename"
    elif category_consolidation:
        out["consolidation"], out["covers"], out["basis"]["consolidation"] = category_consolidation, [category_consolidation], "listing category"
    f2 = re.sub(r"\b(first|second|third|fourth)\b(?= quarter)", lambda m: _ORDINAL[m.group(1)], f)
    q = (re.search(r"\b([1-4])(?:st|nd|rd|th) quarter(?: of)?\s*(20\d\d)?", f2)
         or re.search(r"(?:quy|quarter)\s*([ivx]+|[1-4])\b(?!\d)\s*(?:[./]|nam|of)?\s*(20\d\d)?", f2)
         or re.search(r"\bq([1-4])\s*[./]?\s*(20\d\d)?\b", f2))
    years = [int(y) for y in re.findall(r"(?<!\d)(20\d\d)(?!\d)", f)]
    if q:
        tok = q.group(1)
        n = ROMAN.get(tok.upper()) or (int(tok) if tok.isdigit() else None)
        if n in (1, 2, 3, 4):
            out["periodType"] = f"Q{n}"
            out["fiscalYear"] = int(q.group(2)) if q.group(2) else (years[-1] if years else None)
    if out["periodType"] is None:
        if re.search(r"6 thang|sau thang|ban nien|giua nien do|soat xet|6-month|half|1h20\d\d|30/0?6/", f):
            out["periodType"] = "H1"
        elif re.search(r"kiem toan|audited", f):
            out["periodType"] = "FY"
        out["fiscalYear"] = years[-1] if years else None
    if out["fiscalYear"] is None:
        fy = re.findall(r"(?<!\d)(20\d\d)(?!\d)", fn)
        if fy:
            out["fiscalYear"], out["basis"]["fiscalYear"] = int(fy[-1]), "filename"
    if re.search(r"kiem toan|audited", f) and not re.search(r"chua (?:duoc )?kiem toan|unaudited", f):
        out["auditStatus"] = "audited"
    elif re.search(r"soat xet|reviewed", f):
        out["auditStatus"] = "reviewed"
    if out["periodType"] is None or out["fiscalYear"] is None:
        out["skipReason"] = "reporting period not determinable from title"
        return out
    y, pt = out["fiscalYear"], out["periodType"]
    out["reportingPeriod"] = f"{y}{pt}" if pt.startswith("Q") or pt == "H1" else str(y)
    out["isFinancialStatement"] = True
    return out


def classify_notice(title: str) -> list[tuple[str, str]]:
    """MB investor notice announcing financial statements -> [(reportingPeriod, consolidation)], else []."""
    f = fold(title)
    if re.search(r"gia han|hop dong|lua chon|giai trinh|dinh chinh|chap thuan|tam ung", f):
        return []
    c = classify_v3(title)
    if not c["isFinancialStatement"]:
        return []
    covers = c["covers"] or ["consolidated", "separate"]  # "BCTC quý 4" without a word covers both statements
    return [(c["reportingPeriod"], cons) for cons in covers]


def period_end(reporting_period: str) -> datetime:
    m = re.fullmatch(r"(\d{4})(?:Q([1-4])|(H1))?", reporting_period)
    if not m:
        raise ValueError(reporting_period)
    y = int(m.group(1))
    if m.group(2):
        month = int(m.group(2)) * 3
    elif m.group(3):
        month = 6
    else:
        month = 12
    last = {3: 31, 6: 30, 9: 30, 12: 31}[month]
    return datetime(y, month, last, tzinfo=UTC)


def listed_size_matches(listed: str | None, byte_size: int) -> bool | None:
    """Listing size text ('12.84 MB', '487 KB') vs stored bytes. None when the listing states no usable size.
    Both issuers print bytes/2^20 (or /2^10) rounded to 2 decimals (A0 §5)."""
    m = re.fullmatch(r"\s*([\d.]+)\s*(MB|KB)\s*", listed or "", re.I)
    if not m:
        return None
    val = float(m.group(1))
    if val == 0:
        return None
    if m.group(2).upper() == "MB":
        return abs(byte_size / 2**20 - val) <= 0.0051 + 1e-9
    return abs(byte_size / 2**10 - val) <= 0.51


class FetchOutcome(dict):
    """{outcome: ok|http-error|rate-limited|timeout|too-large|parse-error, status, detail, ...}."""


class PoliteHttp:
    """Budgeted, rate-limited HTTP for one issuer. Retries 403/429/5xx/timeouts with backoff; every network call
    counts against the budget. `sleep`/`now` are injectable for offline tests."""

    RETRY_STATUSES = (403, 429, 500, 502, 503, 504)

    def __init__(self, allowed_hosts: Iterable[str], budget: int, delay_s: float = DEFAULT_DELAY_S,
                 session: Any = None, sleep: Callable[[float], None] = time.sleep,
                 monotonic: Callable[[], float] = time.monotonic, backoff: tuple[float, ...] = (3, 6, 12),
                 loopback_http: bool = False):
        if session is None:
            import requests

            session = requests.Session()
        self.session = session
        self.session.headers.update({"User-Agent": BROWSER_UA, "Accept-Language": "vi,en;q=0.8"})
        self.allowed_hosts = frozenset(h.lower() for h in allowed_hosts)
        # Test-only (fixture server drills): plain http allowed solely for loopback hosts.
        self.loopback_http = loopback_http and self.allowed_hosts <= {"127.0.0.1", "localhost"}
        self.budget, self.delay_s, self.sleep, self.monotonic, self.backoff = budget, delay_s, sleep, monotonic, backoff
        self.used = 0
        self.retries = 0
        self.by_outcome: dict[str, int] = {}
        self._last = -1e9

    def _check(self, url: str) -> None:
        p = urlparse(url)
        scheme_ok = p.scheme == "https" or (self.loopback_http and p.scheme == "http")
        if not scheme_ok or (p.hostname or "").lower() not in self.allowed_hosts:
            raise ValueError(f"refusing non-issuer URL: {url}")

    def _one(self, method: str, url: str, headers: dict[str, str] | None, stream: bool) -> Any:
        if self.used >= self.budget:
            raise BudgetExceeded(f"request budget {self.budget} reached")
        wait = self.delay_s - (self.monotonic() - self._last)
        if wait > 0:
            self.sleep(wait)
        self.used += 1
        self._last = self.monotonic()
        return self.session.request(method, url, headers=headers or {}, timeout=(15, 180), stream=stream,
                                    allow_redirects=True)

    def request(self, url: str, *, method: str = "GET", headers: dict[str, str] | None = None,
                stream: bool = False, max_retries: int = 3) -> FetchOutcome:
        """Returns FetchOutcome with `resp` on success (outcome 'ok'). Raises BudgetExceeded."""
        self._check(url)
        last: FetchOutcome | None = None
        for attempt in range(max_retries + 1):
            try:
                resp = self._one(method, url, headers, stream)
            except BudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 - network errors are outcomes, not crashes
                name = type(exc).__name__
                kind = "timeout" if "timeout" in name.lower() else "http-error"
                last = FetchOutcome(outcome=kind, status=None, detail=f"{name}: {str(exc)[:200]}", attempts=attempt + 1)
            else:
                try:
                    self._check(resp.url)
                except ValueError as exc:
                    return FetchOutcome(outcome="http-error", status=resp.status_code, detail=str(exc), attempts=attempt + 1)
                if resp.status_code < 400:
                    out = FetchOutcome(outcome="ok", status=resp.status_code, resp=resp, attempts=attempt + 1, detail=None)
                    self.by_outcome["ok"] = self.by_outcome.get("ok", 0) + 1
                    return out
                kind = "rate-limited" if resp.status_code in (403, 429) else "http-error"
                last = FetchOutcome(outcome=kind, status=resp.status_code, detail=f"HTTP {resp.status_code}",
                                    attempts=attempt + 1)
                if resp.status_code not in self.RETRY_STATUSES:
                    break
                retry_after = resp.headers.get("Retry-After") if hasattr(resp, "headers") else None
                if retry_after and str(retry_after).isdigit():
                    last["retryAfter"] = int(retry_after)
            if attempt < max_retries:
                self.retries += 1
                pause = self.backoff[min(attempt, len(self.backoff) - 1)]
                self.sleep(max(pause, float(last.get("retryAfter") or 0)))
        assert last is not None
        self.by_outcome[last["outcome"]] = self.by_outcome.get(last["outcome"], 0) + 1
        return last


def download_pdf(http: PoliteHttp, url: str, dest_dir: Path, max_bytes: int = MAX_DOCUMENT_BYTES) -> FetchOutcome:
    """GET a document into dest_dir/.partial-*; verifies the %PDF- magic (BIDV answers a missing/old version
    token with an HTML error page and HTTP 200) and the size cap. On success returns outcome 'ok' with
    sha256, bytes, tmpPath, etag, lastModified, contentType. Never leaves a partial file behind."""
    r = http.request(url, stream=True)
    if r["outcome"] != "ok":
        return r
    resp = r["resp"]
    headers = getattr(resp, "headers", {}) or {}
    ctype = headers.get("Content-Type", "")
    declared = headers.get("Content-Length")
    if declared and str(declared).isdigit() and int(declared) > max_bytes:
        return FetchOutcome(outcome="too-large", status=resp.status_code, detail=f"Content-Length {declared} > cap {max_bytes}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest_dir / f".partial-{uuid.uuid4().hex}"
    h, size = hashlib.sha256(), 0
    try:
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(1 << 16):
                if not chunk:
                    continue
                size += len(chunk)
                if size > max_bytes:
                    raise _TooLarge(f"body exceeds cap {max_bytes} bytes")
                h.update(chunk)
                fh.write(chunk)
        with tmp.open("rb") as fh:
            magic = fh.read(5)
        if magic != b"%PDF-":
            raise _NotPdf(f"not a PDF (magic {magic!r}, content-type {ctype or '?'}, final url {getattr(resp, 'url', url)})")
    except _TooLarge as exc:
        tmp.unlink(missing_ok=True)
        return FetchOutcome(outcome="too-large", status=resp.status_code, detail=str(exc))
    except _NotPdf as exc:
        tmp.unlink(missing_ok=True)
        return FetchOutcome(outcome="parse-error", status=resp.status_code, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - interrupted body: an outcome, file discarded
        tmp.unlink(missing_ok=True)
        kind = "timeout" if "timeout" in type(exc).__name__.lower() else "http-error"
        return FetchOutcome(outcome=kind, status=resp.status_code, detail=f"{type(exc).__name__}: {str(exc)[:200]}")
    return FetchOutcome(outcome="ok", status=resp.status_code, sha256=h.hexdigest(), bytes=size, tmpPath=str(tmp),
                        etag=headers.get("ETag"), lastModified=headers.get("Last-Modified"), contentType=ctype,
                        finalUrl=getattr(resp, "url", url), detail=None)


class _TooLarge(Exception):
    pass


class _NotPdf(Exception):
    pass


# --------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["discover", "fetch", "extract", "manifest", "register"])
    ap.add_argument("--symbol", required=True, choices=sorted(ISSUERS))
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY_S)
    ap.add_argument("--base-url", default="http://localhost:5197")
    ap.add_argument("--what", choices=["documents", "fundamentals"], default="documents", help="register only")
    a = ap.parse_args(argv)
    if a.command == "discover":
        c = cmd_discover(a.root, a.symbol, a.budget, a.delay)
        print(json.dumps({"candidates": len(c["candidates"]), "targets": c["targets"], "requests": c["requestsUsed"]},
                         ensure_ascii=False, indent=2))
    elif a.command == "fetch":
        print(json.dumps(cmd_fetch(a.root, a.symbol, a.budget, a.delay), ensure_ascii=False, indent=2))
    elif a.command == "extract":
        print(json.dumps(cmd_extract(a.root, a.symbol), ensure_ascii=False, indent=2))
    elif a.command == "manifest":
        m = cmd_manifest(a.root, a.symbol)
        print(json.dumps({"documents": len(m["documents"]), "fundamentals": len(m["fundamentals"])}, indent=2))
    else:
        print(json.dumps(cmd_register(a.root, a.symbol, a.base_url, a.what), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
