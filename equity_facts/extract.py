"""extract_document: issuer PDF -> classified pages -> hybrid OCR -> tables -> normalized, validated facts.

Idempotent per (DocumentId, EXTRACTOR_VERSION). Writes on the caller's connection and never commits.
Intermediate artifacts (page plan, Tesseract words, RapidOCR boxes, reconstructed tables, facts) are kept under
{root}/{SYM}/extractions/{sha256}/{EXTRACTOR_VERSION}/ so every fact can be re-checked against its page.
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from . import catalog as cat_mod
from .classify import classify_text, plan_pages, scan_done
from .periods import COMP_WORDS, CUR_WORDS, DocPeriod, _dates, _spec, assign_columns, doc_period, quarter_label
from .tables import build_table, headers_for_row
from .textnorm import detect_unit, fold, norm_label, parse_amount
from .validate import _key as _vkey
from .validate import evaluate

log = logging.getLogger("equity_facts")

EXTRACTOR_VERSION = "eqx-facts-v4"
CHEAP_DPI = 150
FULL_DPI = 300
REREAD_DPIS = (200, 250)
KIND_STATEMENT = {"balance_sheet": "balance-sheet", "income_statement": "income-statement", "cash_flow": "cash-flow",
                  "note_loan_quality": "note"}
SYMBOLS = {"BID", "MBB"}


class OcrEngine:
    """Real engines (pypdfium2 + Tesseract + RapidOCR). Tests inject a fake with the same methods."""

    def __init__(self) -> None:
        from . import ocr

        self.ocr = ocr
        self.rotations: dict[int, int] = {}  # page -> extra clockwise rotation found by OSD (sideways scans)

    def open(self, path: Path):
        self.rotations = {}
        return self.ocr.PdfPages(path)

    def cheap_text(self, pages, page_no: int) -> str:
        img, _ = pages.render(page_no, CHEAP_DPI)
        text = self.ocr.tesseract_text(img, CHEAP_DPI)
        if not _looks_upright(text):  # sideways table scanned without /Rotate (MB 2018-2019 income statements)
            rot = self.ocr.osd_rotation(img)
            if rot in (90, 180, 270):
                turned = self.ocr.tesseract_text(img.rotate(-rot, expand=True), CHEAP_DPI)
                if _looks_upright(turned) or len(turned.split()) > 2 * len(text.split()):
                    self.rotations[page_no] = rot
                    text = turned
        return text

    def full_page(self, pages, page_no: int) -> dict:
        img, rotation = pages.render(page_no, FULL_DPI)
        if self.rotations.get(page_no):
            img = img.rotate(-self.rotations[page_no], expand=True)
        skew = self.ocr.estimate_skew(img)
        img = self.ocr.deskew(img, skew)
        words = self.ocr.tesseract_words(img, FULL_DPI)
        if len(words) < 30:
            rot = self.ocr.osd_rotation(img)
            if rot:
                img = img.rotate(-rot, expand=True)
                words = self.ocr.tesseract_words(img, FULL_DPI)
        boxes = self.ocr.rapid_boxes(img)
        long_boxes = [b for b in boxes if len(b["text"]) >= 4]
        vertical = [b for b in long_boxes if (b["box"][3] - b["box"][1]) > 1.5 * (b["box"][2] - b["box"][0])]
        if long_boxes and len(vertical) > 0.5 * len(long_boxes) and not self.rotations.get(page_no):
            # whole page printed sideways without /Rotate (MB 2019Q3 income statement): turn it and read again
            rot = self.ocr.osd_rotation(img) or 90
            self.rotations[page_no] = rot
            return self.full_page(pages, page_no)
        return {"words": words, "boxes": boxes, "width": img.width, "height": img.height, "skewDeg": skew,
                "pdfRotation": rotation, "osdRotation": self.rotations.get(page_no, 0)}

    def reread(self, pages, page_no: int, boxes: list[list[float]], skew: float) -> list[list[str]]:
        """RapidOCR texts of each cell crop (300-dpi deskewed coordinates) re-rendered at REREAD_DPIS."""
        from PIL import Image

        out: list[list[str]] = [[] for _ in boxes]
        for dpi in REREAD_DPIS:
            img, _ = pages.render(page_no, dpi)
            if self.rotations.get(page_no):
                img = img.rotate(-self.rotations[page_no], expand=True)
            img = self.ocr.deskew(img, skew)
            s = dpi / FULL_DPI
            for i, (x0, y0, x1, y1) in enumerate(boxes):
                # the box crop (+-25 px) and a crop centred on the cell: a merged two-row box brings the neighbouring
                # rows in, which made RapidOCR read "663.089" as "633089" (MB 2017Q4 p27); some cells read only in
                # the wider crop (BIDV 2018Q4 p5 "99,621"), so both readings are kept
                yc, half = (y0 + y1) / 2, min((y1 - y0) / 2, 20) + 6
                for top, bottom in ((y0 - 25, y1 + 25), (yc - half, yc + half)):
                    crop = img.crop((int((x0 - 60) * s), int(top * s), int((x1 + 40) * s), int(bottom * s)))
                    canvas = Image.new("L", (crop.width + 200, crop.height + 120), 255)
                    canvas.paste(crop, (100, 60))
                    out[i].append(" ".join(b["text"] for b in self.ocr.rapid_boxes(canvas)))
        return out

    def text_layer(self, pages, page_no: int) -> str:
        try:
            return pages.text_layer(page_no)
        except Exception:  # noqa: BLE001 - text layer is only a cross-check flag
            return ""

    def versions(self) -> dict:
        return self.ocr.tool_versions()


LOAN_GROUP_CODES = ("LOAN_GROUP1", "NPL_GROUP2", "NPL_GROUP3", "NPL_GROUP4", "NPL_GROUP5")


def _confirmed_by_complete_identity(f: dict, results: list[dict]) -> bool:
    """A weak label match is kept only if a passing identity used it with every term present: in a 'sum' identity a
    combined line can stand in for an absent one (MB 2021 "Lãi thuần từ mua bán chứng khoán kinh doanh, chứng khoán
    đầu tư …" is trading + investment, not investment securities alone)."""
    k = list(_vkey(f))
    return f.get("status") == "validated" and any(
        r["outcome"] == "pass" and r["key"] == k and f["metric"] in r["metrics"] and not r.get("missing") for r in results)


def _assign_group1_definition(facts: list[dict], sym: str) -> None:
    """MB prints 'Nợ đủ tiêu chuẩn' with the MBS margin line inside (quarterly) or separately (annual/H1): two
    definitions, never one series. The loan-group identity has already decided the layout per column."""
    if sym != "MBB":
        return
    defs = cat_mod.metrics_by_code()
    for f in facts:
        if f["metric"] != "LOAN_GROUP1":
            continue
        treat = {c.get("marginTreatment") for c in f.get("checks", [])} - {None}
        if "separate" in treat:
            f["definition"] = "excl. MBS margin line (printed separately; groups 1-5 + margin = total)"
            continue
        f["metric"], f["metricDef"] = "LOAN_GROUP1_INCL_MARGIN", defs["LOAN_GROUP1_INCL_MARGIN"]
        f["definition"] = ("incl. MBS margin line (of-which inside group 1; groups 1-5 = total)" if "inGroup1" in treat
                           else "as printed; no separate MBS margin line in this column")


def _note_amount_columns(res: dict, hdr: dict[str, str], n: int, dp: DocPeriod, group_rows: list[dict]) -> dict:
    """Loan-quality tables printing "amount | %" per date (or a date header centred over two columns): when the
    header-based map leaves the amount columns unresolved, take the two columns that hold amounts in every group row
    (left = current period end, right = the latest earlier date printed in the headers)."""
    amounts = []
    for k in range(n):
        vals = [r["cells"][str(k)]["value"] for r in group_rows
                if str(k) in r["cells"] and r["cells"][str(k)]["kind"] == "number"]
        if len(vals) >= 4 and len(vals) == len(group_rows) and all(abs(v) >= 10_000 for v in vals):
            amounts.append(k)
    if len(amounts) != 2 or all(res["columns"][k] is not None for k in amounts):
        return res
    def role(text: str) -> str | None:  # per-column evidence: dates, "cuối/đầu kỳ" words, or the bare year
        f, ds = fold(text), [d for d in _dates(text) if d <= dp.end]
        if dp.end in ds or any(w in f for w in CUR_WORDS) or (str(dp.year) in f and str(dp.year - 1) not in f):
            return "cur"
        if (ds and max(ds) < dp.end) or any(w in f for w in COMP_WORDS) or str(dp.year - 1) in f:
            return "comp"
        return None

    left, right = (role(hdr.get(str(k), "")) for k in amounts)
    if left == "comp" or right == "cur":
        return res  # the headers contradict "left = current"
    allh = " ".join(hdr.values())
    dates = sorted({d for d in _dates(allh) if d <= dp.end})
    prior = [d for d in dates if d < dp.end]
    if not (left == "cur" or dp.end in dates) or not (right == "comp" or prior):
        return res  # neither the column headers nor the table header name both periods
    comp_end = prior[-1] if prior else date(dp.year - 1, 12, 31)
    cols: list[dict | None] = [None] * n
    method = "amount-columns+header"
    cols[amounts[0]] = _spec("current", quarter_label(dp.end), "point-in-time", None, dp.end, method, hdr.get(str(amounts[0]), ""))
    cols[amounts[1]] = _spec("comparative", quarter_label(comp_end), "point-in-time", None, comp_end, method,
                             hdr.get(str(amounts[1]), ""))
    return {"columns": cols, "errors": [], "fallbackFrom": res["errors"]}


def _looks_upright(text: str) -> bool:
    """Every BIDV/MB statement page carries the bank name / form header in its first lines."""
    head = fold(" ".join([ln for ln in (text or "").splitlines() if ln.strip()][:15]))
    return bool(re.search(r"NGAN HANG|BIDV|QUAN DOI|THUONG MAI|TCTD|BAO CAO|THUYET MINH", head))


def _utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def document_available_at(doc: dict) -> datetime:
    """Same rule as the backend (EquityResearchNoteService.AvailableAt): issuer date, else first-seen."""
    first = _utc(doc["FirstSeenAtUtc"])
    pub = _utc(doc.get("PublishedAtUtc"))
    if pub is None:
        return first
    return first if doc.get("SupersedesDocumentId") and first > pub else pub


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)


def extract_with(repo, engine, document: dict, root: Path, *, run_id: str | None,
                 heartbeat: Callable[[], None], clock: Callable[[], datetime] = lambda: datetime.now(UTC)) -> dict:
    doc_id = str(document["Id"])
    existing = repo.get_extraction(doc_id, EXTRACTOR_VERSION)
    if existing and existing["Status"] in ("succeeded", "partial"):
        return {"skipped": True, "status": existing["Status"], "extractionId": existing["Id"]}
    if existing:
        repo.delete_failed_extraction(existing["Id"])
    started = clock()
    t0 = time.perf_counter()
    extraction_id = str(uuid.uuid4())
    sym = str(document.get("Symbol") or "").upper()
    sha = str(document.get("ContentSha256") or "")
    art_rel = f"{sym}/extractions/{sha}/{EXTRACTOR_VERSION}"
    art = Path(root) / art_rel
    quality: dict[str, Any] = {"extractorVersion": EXTRACTOR_VERSION, "runId": run_id, "problems": []}

    def fail(msg: str, page_count: int = 0, pages_ocr: int = 0) -> dict:
        repo.insert_extraction({"Id": extraction_id, "DocumentId": doc_id, "ExtractorVersion": EXTRACTOR_VERSION,
                                "Method": "ocr", "Status": "failed", "PageCount": page_count, "PagesOcr": pages_ocr,
                                "ArtifactPath": art_rel if art.exists() else None, "QualityJson": quality, "Error": msg,
                                "StartedAtUtc": started, "FinishedAtUtc": max(clock(), started)})
        return {"status": "failed", "method": "ocr", "pageCount": page_count, "pagesOcr": pages_ocr, "facts": 0,
                "validated": 0, "reviewNeeded": 0, "quarantined": 0, "extractionId": extraction_id, "error": msg}

    if sym not in SYMBOLS:
        return fail(f"symbol {sym!r} out of scope")
    try:
        dp = doc_period(document.get("ReportingPeriod"))
    except ValueError as exc:
        return fail(str(exc))
    pdf_path = Path(root) / str(document.get("StoragePath") or "")
    if not pdf_path.is_file():
        return fail(f"document file missing: {document.get('StoragePath')}")

    pages = engine.open(pdf_path)
    try:
        n_pages = len(pages)
        # ---- 1. cheap classification pass (stops once statements + loan-quality note are found)
        infos = []
        for p in range(1, n_pages + 1):
            info = {"page": p, **classify_text(engine.cheap_text(pages, p))}
            infos.append(info)
            heartbeat()
            plan = plan_pages(infos)
            if scan_done(plan["kinds"], plan["notesStart"]):
                break
        plan = plan_pages(infos)
        quality["cheapPagesScanned"] = len(infos)
        _write_json(art / "pages.json", {"documentId": doc_id, "reportingPeriod": dp.label, "pages": infos, "plan": plan})
        kinds = {p: k for p, k in plan["kinds"].items() if k in KIND_STATEMENT}
        if not kinds:
            quality["problems"].append("no statement pages found by the classification pass")

        # ---- 2. full hybrid OCR on the needed pages only
        tables: dict[int, dict] = {}
        units: dict[int, tuple[int, str] | None] = {}
        text_layers: dict[int, str] = {}
        page_meta: dict[int, dict] = {}
        for p in sorted(kinds):
            res = engine.full_page(pages, p)
            heartbeat()
            table = build_table(res["words"], res["boxes"], res["width"], res["height"])
            quality.setdefault("rereads", {})[str(p)] = _reread_cells(engine, pages, p, table, res.get("skewDeg") or 0.0)
            words_text = "\n".join(" ".join(w["text"] for w in res["words"] if w.get("line") == ln)
                                   for ln in dict.fromkeys(w.get("line") for w in res["words"]))
            units[p] = detect_unit(words_text)
            text_layers[p] = engine.text_layer(pages, p)
            page_meta[p] = {"kind": kinds[p], "skewDeg": res.get("skewDeg"), "pdfRotation": res.get("pdfRotation"),
                            "osdRotation": res.get("osdRotation", 0),
                            "unit": units[p], "words": len(res["words"]), "boxes": len(res["boxes"]),
                            "columns": len(table["columns"]), "textLayerChars": len(text_layers[p])}
            _write_json(art / f"p{p:03d}.words.json", res["words"])
            _write_json(art / f"p{p:03d}.numbers.json", res["boxes"])
            _write_json(art / f"p{p:03d}.table.json", table)
            tables[p] = table
        pages_ocr = len({i["page"] for i in infos} | set(kinds))
    finally:
        pages.close()

    facts, dashes, unreadable, problems = build_facts(document, dp, kinds, tables, units, text_layers, plan)
    quality["problems"].extend(problems)
    quality["pages"] = page_meta
    results = evaluate(facts, dashes, unreadable)
    weak = [f for f in facts if f.get("weakAnchor") and not _confirmed_by_complete_identity(f, results)]
    if weak or any(d.get("weakAnchor") for d in dashes + unreadable):
        # a weak label match must never decide anyone else's status: drop the unconfirmed ones and re-evaluate
        if weak:
            quality["problems"].append("weak label matches not confirmed by an identity (not persisted): "
                                       + ", ".join(sorted({f"p{f['page']} {f['metric']}" for f in weak})))
        weak_ids = {id(f) for f in weak}
        facts = [f for f in facts if id(f) not in weak_ids]
        dashes = [d for d in dashes if not d.get("weakAnchor")]
        unreadable = [u for u in unreadable if not u.get("weakAnchor")]
        for f in facts:
            f.pop("checks", None)
            f.pop("status", None)
        results = evaluate(facts, dashes, unreadable)
    _assign_group1_definition(facts, sym)
    real = [f for f in facts if not f.get("aux")]
    of_which = [f for f in real if f.get("marginInGroup1")]
    if of_which:
        quality["problems"].append(f"margin line is an of-which part of group 1 (groups 1-5 = total): {len(of_which)} "
                                   "MARGIN_LOANS_GROUP1 cells not persisted")
        real = [f for f in real if not f.get("marginInGroup1")]
    quality["consolidationByPage"] = {str(p): c for p, c in (plan.get("pageConsolidation") or {}).items() if p in kinds}
    unknown = [f for f in real if f.get("cons") not in ("consolidated", "separate")]
    if unknown:
        quality["problems"].append(f"consolidation unknown for {len(unknown)} facts: not persisted")
        real = [f for f in real if f.get("cons") in ("consolidated", "separate")]

    available = document_available_at(document)
    now = clock()
    rows = []
    for f in real:
        rows.append(_fact_row(f, document, sym, f["cons"], extraction_id, available, now))

    stmt_kinds = {f["pageKind"] for f in real if f["cons"] == (document.get("Consolidation") if document.get("Consolidation")
                                                                in ("consolidated", "separate") else f["cons"])}
    status = "succeeded" if {"balance_sheet", "income_statement"} <= stmt_kinds else "partial"
    counts = {s: sum(1 for r in rows if r["Status"] == s) for s in ("validated", "review-needed", "quarantined", "extracted")}
    quality.update({"identities": results, "counts": counts, "facts": len(rows), "seconds": round(time.perf_counter() - t0, 1),
                    "tools": _safe_versions(engine), "pagesFullOcr": sorted(kinds)})
    _write_json(art / "facts.json", rows)
    _write_json(art / "quality.json", quality)
    repo.insert_extraction({"Id": extraction_id, "DocumentId": doc_id, "ExtractorVersion": EXTRACTOR_VERSION, "Method": "ocr",
                            "Status": status, "PageCount": n_pages, "PagesOcr": min(n_pages, pages_ocr), "ArtifactPath": art_rel,
                            "QualityJson": _quality_summary(quality), "Error": None, "StartedAtUtc": started,
                            "FinishedAtUtc": max(clock(), started)})
    repo.insert_facts(rows)
    return {"status": status, "method": "ocr", "pageCount": n_pages, "pagesOcr": min(n_pages, pages_ocr), "facts": len(rows),
            "validated": counts["validated"], "reviewNeeded": counts["review-needed"], "quarantined": counts["quarantined"],
            "extracted": counts["extracted"], "extractionId": extraction_id, "artifactPath": art_rel,
            "seconds": quality["seconds"]}


def _reread_cells(engine, pages, page_no: int, table: dict, skew: float) -> dict:
    """Second and third readings of unreadable cells; accept a value only when two well-formed readings agree."""
    todo = [(r, k, c) for r in table["rows"] for k, c in r["cells"].items() if c.get("needsReread") and c.get("box")]
    if not todo or not hasattr(engine, "reread"):
        return {"cells": len(todo), "resolved": 0}
    try:
        texts = engine.reread(pages, page_no, [c["box"] for _, _, c in todo], skew)
    except Exception as exc:  # noqa: BLE001 - a failed re-read leaves the cell unreadable
        return {"cells": len(todo), "resolved": 0, "error": f"{type(exc).__name__}: {exc}"[:200]}
    resolved = 0
    for (row, k, cell), reads in zip(todo, texts):
        labels = [f"rapid{crop}@{d}" for d in REREAD_DPIS for crop in ("", "-centred")]
        readings = list(zip(labels, reads)) + [("tesseract@300", cell.get("tessRaw") or "")]
        parsed = [(src, t, parse_amount(t)) for src, t in readings]
        values = [p["value"] for _, _, p in parsed if p["kind"] == "number"]
        agreed = [v for v in set(values) if values.count(v) >= 2]
        cell["reread"] = [{"source": src, "text": t} for src, t, _ in parsed]
        dashes = sum(1 for _, _, p in parsed if p["kind"] == "dash")
        if not values and dashes >= 2:  # printed "-" (nil) that RapidOCR first read as "01"
            cell.update({"kind": "dash", "value": None, "engine": "rapid-reread", "rapidRaw": cell["raw"], "raw": "-"})
            resolved += 1
            continue
        if len(agreed) == 1:
            v = agreed[0]
            cell["rapidRaw"] = cell["raw"]
            cell["raw"] = next(t for _, t, p in parsed if p["kind"] == "number" and p["value"] == v)
            cell.update({"kind": "number", "value": v, "engine": "rapid-reread",
                         "engineAgree": cell.get("tessRaw") is not None and parse_amount(cell["tessRaw"])["value"] == v})
            resolved += 1
    return {"cells": len(todo), "resolved": resolved}


def _safe_versions(engine) -> dict:
    try:
        return engine.versions()
    except Exception:  # noqa: BLE001
        return {}


def _quality_summary(q: dict) -> dict:
    """QualityJson stored in PG: compact (identity details stay in the artifact quality.json)."""
    ids = q.get("identities") or []
    return {"extractorVersion": q["extractorVersion"], "runId": q.get("runId"), "problems": q.get("problems", [])[:50],
            "counts": q.get("counts"), "facts": q.get("facts"), "seconds": q.get("seconds"), "tools": q.get("tools"),
            "cheapPagesScanned": q.get("cheapPagesScanned"), "pagesFullOcr": q.get("pagesFullOcr"),
            "consolidationInferred": q.get("consolidationInferred"),
            "identities": {o: sum(1 for r in ids if r["outcome"] == o) for o in ("pass", "weak", "fail", "incomplete")},
            "failedIdentities": [{k: r.get(k) for k in ("id", "key", "diff")} for r in ids if r["outcome"] == "fail"][:30],
            "pages": q.get("pages")}


def _text_layer_digits(text: str) -> set[str]:
    import re

    return {re.sub(r"\D", "", m) for m in re.findall(r"\d[\d.,]{3,}\d", text or "")}


def build_facts(document: dict, dp: DocPeriod, kinds: dict[int, str], tables: dict[int, dict],
                units: dict[int, Any], text_layers: dict[int, str], plan: dict) -> tuple[list, list, list, list]:
    """Anchor catalog metrics in the reconstructed tables and turn cells into candidate facts (pure)."""
    catalog = cat_mod.load_catalog()
    issuer = str(document.get("Symbol") or "").upper()
    facts: list[dict] = []
    dashes: list[dict] = []
    unreadable: list[dict] = []
    problems: list[str] = []
    found_units = {u for u in units.values() if u}
    doc_unit = min(found_units) if len({u[0] for u in found_units}) == 1 else None
    if len({u[0] for u in found_units}) > 1:
        problems.append(f"conflicting units across pages: {sorted(found_units)}")
    doc_cons = document.get("Consolidation") if document.get("Consolidation") in ("consolidated", "separate") else None
    page_cons = plan.get("pageConsolidation") or {}
    titles = {c for c in (plan.get("consolidationTitles") or []) if c}
    groups: dict[tuple, list[int]] = {}
    for p, k in sorted(kinds.items()):
        groups.setdefault((k, page_cons.get(p) if doc_cons is None else None), []).append(p)
    for (kind, group_cons), kpages in groups.items():
        cons = doc_cons or group_cons or (next(iter(titles)) if len(titles) == 1 else None)
        rows = []
        for p in kpages:
            for r in tables[p]["rows"]:
                rows.append({**r, "page": p})
        if kind == "balance_sheet":
            cat_mod.tag_bs_sections(rows)
        all_rows = rows
        if kind == "note_loan_quality":  # the group table starts at its "Nợ đủ tiêu chuẩn" row (earlier tables differ)
            start = next((i for i, r in enumerate(rows) if r.get("cells") and any(
                n.startswith("NO DU TIEU CHUAN") for n in r.get("labelNorm", []))), 0)
            rows = rows[start:]
        metrics = cat_mod.extractable(catalog, issuer, kind)
        anchored = cat_mod.anchor_metrics(rows, metrics, issuer)
        margin_def = [m for m in metrics if m["code"] == "MARGIN_LOANS_GROUP1"]
        if kind == "note_loan_quality" and margin_def and "MARGIN_LOANS_GROUP1" not in anchored:
            # pre-2018 MB notes print the MBS line only in the by-type table above the group table
            anchored.update(cat_mod.anchor_metrics(all_rows, margin_def, issuer))
        if kind != "note_loan_quality":
            # second pass for OCR-garbled labels on the rows nobody claimed ("Thu nhận từ gúp vẫn mua cả nhẫn",
            # BIDV 2025Q3 p7): such a fact is kept only if an identity then validates it
            used = {id(a["row"]) for a in anchored.values()}
            left = [m for m in metrics if m["code"] not in anchored]
            weak = cat_mod.anchor_metrics([r for r in rows if id(r) not in used], left, issuer, cat_mod.WEAK_THRESHOLD)
            for a in weak.values():
                a["weak"] = True
            anchored.update(weak)
        missing = [m["code"] for m in metrics if m["code"] not in anchored and not m.get("optional")]
        if missing:
            problems.append(f"{kind}: metrics not found {missing}")
        col_cache: dict[tuple, dict] = {}
        statement = KIND_STATEMENT[kind]
        by_code = {m["code"]: m for m in metrics}
        last_good: dict[int, tuple[int, dict]] = {}  # ncols -> (page, resolved map) for header-less continuation pages

        def columns_for(page: int, row_y: float, col_cache=col_cache, statement=statement, kind=kind,
                        last_good=last_good, anchored=anchored) -> dict:
            t = tables[page]
            hdr = headers_for_row(t, row_y)
            n = len(t["columns"])
            key = (page, tuple(sorted(hdr.items())), n)
            if key not in col_cache:
                res = assign_columns(hdr, n, statement, dp)
                if kind == "note_loan_quality":
                    res = _note_amount_columns(res, hdr, n, dp, [a["row"] for c, a in anchored.items()
                                                                 if c in LOAN_GROUP_CODES and a["row"]["page"] == page])
                if all(c is None for c in res["columns"]) and n in last_good and last_good[n][0] < page \
                        and not res.get("restatement"):
                    # continuation page whose own header was not printed / not read: same layout as the previous page
                    src_page, src = last_good[n]
                    res = {"columns": [dict(c, method=f"inherited-p{src_page}") if c else None for c in src["columns"]],
                           "errors": [], "inheritedFrom": src_page}
                elif any(c is not None for c in res["columns"]):
                    last_good[n] = (page, res)
                col_cache[key] = res
                for e in res["errors"]:
                    problems.append(f"p{page} {kind}: {e}")
            return col_cache[key]

        extra = []
        if kind == "note_loan_quality" and "NPL_GROUP5" in anchored:
            total_row = _loan_total_row(rows, anchored["NPL_GROUP5"]["row"])
            if total_row is not None:
                extra.append(("LOAN_QUALITY_TOTAL", {"row": total_row, "score": 1.0, "label": "", "text": ""}))
        for code, a in list(anchored.items()) + extra:
            r = a["row"]
            page = r["page"]
            colmap = columns_for(page, r["y"])
            unit = units.get(page)
            unit_flag = None
            if unit is None and doc_unit is not None:
                unit, unit_flag = doc_unit, "unit-from-other-statement-page"
            elif unit is not None and doc_unit is None and len(found_units) > 1:
                unit, unit_flag = None, "unit-conflict"
            for k_str, cell in r["cells"].items():
                k = int(k_str)
                spec = colmap["columns"][k] if k < len(colmap["columns"]) else None
                if spec is None:
                    continue
                base = {"metric": code, "pageKind": kind, "page": page, "cons": cons, "row": r["index"], "col": k,
                        "role": spec["role"],
                        "periodLabel": spec["periodLabel"], "periodType": spec["periodType"], "start": spec["start"],
                        "end": spec["end"], "fiscalYear": spec["fiscalYear"], "weakAnchor": bool(a.get("weak"))}
                if cell["kind"] == "dash":
                    dashes.append(base)
                    continue
                if cell["kind"] != "number":
                    unreadable.append({**base, "raw": cell["raw"]})
                    problems.append(f"p{page} {code} col{k}: unreadable cell {cell['raw']!r}")
                    continue
                raw_label = _raw_label(r, a["text"])
                fact = {**base, "value": cell["value"], "raw": cell["raw"], "engine": cell.get("engine"),
                        "engineAgree": cell.get("engineAgree"), "tessRaw": cell.get("tessRaw"), "box": cell.get("box"),
                        "rawLabel": raw_label, "anchorScore": a["score"], "catalogLabel": a["label"],
                        "column": {"method": spec["method"], "header": spec["header"][:200], "restated": spec["restated"],
                                   **({"implausibleDates": spec["implausibleDates"]} if spec.get("implausibleDates") else {})},
                        "unit": unit, "metricDef": by_code.get(code), "flags": []}
                if code == "LOAN_QUALITY_TOTAL":
                    fact["aux"] = True
                tl = text_layers.get(page) or ""
                if len(tl.strip()) > 50:
                    fact["textLayer"] = str(abs(cell["value"])) in _text_layer_digits(tl)
                else:
                    fact["textLayer"] = None
                if unit_flag:
                    fact["flags"].append(unit_flag)
                if unit is None:
                    fact["forceStatus"] = "quarantined"
                    fact["flags"].append("unit-unknown")
                pc = page_cons.get(page)
                if doc_cons and pc and pc != doc_cons:
                    fact["forceStatus"] = "quarantined"
                    fact["flags"].append(f"consolidation-mismatch: document {doc_cons}, page title {pc}")
                if spec.get("method") == "restatement-layout":
                    fact["flags"].append("restatement-document: restated column (Số trình bày lại); "
                                         "published figure stays in the original report")
                facts.append(fact)
    return facts, dashes, unreadable, problems


def _loan_total_row(rows: list[dict], g5_row: dict) -> dict | None:
    """Unlabelled (or 'Tổng'/'Cộng') total row printed right under 'Nợ có khả năng mất vốn'."""
    same = [r for r in rows if r["page"] == g5_row["page"] and r["y"] > g5_row["y"] and r.get("cells")]
    for r in same[:3]:  # MB annual reports print the margin-lending group-1 line between group 5 and the total
        lab = norm_label(r.get("label") or "")
        if not lab or lab.startswith(("TONG", "CONG")) or len(lab) <= 3:
            return r
    return None


def _raw_label(row: dict, used_norm: str) -> str:
    for raw, nrm in zip(row.get("labelVariants", []), row.get("labelNorm", [])):
        if nrm == used_norm:
            return raw
    if row.get("rapidLabel") and norm_label(row["rapidLabel"]) == used_norm:
        return row["rapidLabel"]
    return row.get("label") or (row.get("labelVariants") or [""])[0]


def _fact_row(f: dict, doc: dict, sym: str, cons: str, extraction_id: str, available: datetime, now: datetime) -> dict:
    m = f["metricDef"]
    scale, unit_raw = f["unit"] if f["unit"] else (1, None)
    key = f"{sym}/{doc['Id']}/{EXTRACTOR_VERSION}/{cons}/{f['metric']}/{f['periodLabel']}/{f['periodType']}/{f['role']}"
    location = (f"p{f['page']} row{f['row']} col{f['col']} | {fold(f['rawLabel'])[:120]} | "
                f"{f['column']['header'][:120]}")
    validation = {"extractorVersion": EXTRACTOR_VERSION, "extractionStatus": f["status"], "checks": f.get("checks", []),
                  "anchor": {"score": f["anchorScore"], "catalogLabel": f["catalogLabel"]},
                  "engine": f.get("engine"), "engineAgree": f.get("engineAgree"), "tesseractRaw": f.get("tessRaw"),
                  "textLayerAgrees": f.get("textLayer"), "column": f["column"], "box": f.get("box"),
                  "flags": f.get("flags", [])}
    if f.get("definition"):
        validation["definition"] = f["definition"]
    return {"Id": str(uuid.uuid4()), "FactKey": key, "Symbol": sym, "MetricCode": f["metric"],
            "MetricDefinitionVersion": m.get("definitionVersion", "v1"), "MetricGroup": m["group"], "Statement": m["statement"],
            "PeriodLabel": f["periodLabel"], "PeriodType": f["periodType"], "PeriodStart": f["start"], "PeriodEnd": f["end"],
            "FiscalYear": f["fiscalYear"], "Consolidation": cons, "Value": f["value"], "Unit": m.get("unit", "VND"),
            "Scale": scale, "RawLabel": (f["rawLabel"] or f["metric"])[:500], "RawValueText": f["raw"][:100],
            "RawUnitText": (unit_raw or None) and unit_raw[:100], "ColumnRole": f["role"], "PageNumber": f["page"],
            "Location": location[:500], "DocumentId": str(doc["Id"]), "ExtractionId": extraction_id, "Status": f["status"],
            "ValidationJson": validation, "IsDerived": False, "DerivationJson": None, "AvailableAtUtc": available,
            "CreatedAtUtc": now}
