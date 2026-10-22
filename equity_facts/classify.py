"""Cheap page classification: low-dpi Tesseract text of each page -> statement title / note table detection.

Statements are recognised by their printed title in the first lines (plus enough grouped numbers so a table
of contents or the auditor's report never qualifies); an untitled numeric page right after a statement page is
its continuation. Notes are recognised by structure, not keyword alone: the loan-quality note must show the
row "Nợ đủ tiêu chuẩn ... <amount>" AND "Nợ có khả năng mất vốn ... <amount>" as table rows on the page.
"""
from __future__ import annotations

import re

from .textnorm import fold

TITLE_RULES = [  # order matters: off-balance pages carry the BS title too
    ("off_balance", r"CHI TIEU NGOAI BANG|NGOAI BAO CAO TINH HINH TAI CHINH|NGOAI BANG CAN DOI"),
    # OCR noise tolerated inside the words ("LUU CHUYEEN TIEN TE", "KET QUÁ HOAT ĐQNG" ...)
    ("cash_flow", r"L\w{1,2}\s?[CG]H\w{0,4}N\s?TI\w{0,2}N\s?TE\b"),
    ("income_statement", r"BAO\s?CAO\s?K\w{0,2}T\s?QU\w{0,2}\s?HO\w{0,2}T\s?D\w{0,2}NG|KET QUA HOAT DONG KINH DOANH"),
    ("balance_sheet", r"BAO\s?CAO\s?T\w{0,2}NH\s?H\w{0,2}NH\s?T\w{0,2}I\s?CH\w{0,2}NH|BANG CAN DOI KE TOAN"),
]
FORM_KINDS = {"2": "balance_sheet", "3": "income_statement", "4": "cash_flow"}
NOTES_TITLE = r"THUYET MINH BAO CAO TAI CH|THUYET MINH BAO CAO"
GROUPED = re.compile(r"\(?\d{1,3}(?:[.,]\d{3})+\)?")
AMOUNT_AT_END = r".{0,80}?\d{1,3}(?:[.,]\s?\d{3}){1,}"
GROUP_PREFIX = r"^\W*(?:NHOM\s?\S{1,2}\W*)?"  # MB annual notes: "Nhóm 1 - Nợ đủ tiêu chuẩn"
LOAN_GROUP_LABELS = ["NO DU TIEU CHUAN", "NO CAN CHU Y", "NO DUOI TIEU CHUAN", "NO NGHI NGO", "NO CO KHA NANG MAT VON"]


def classify_text(text: str) -> dict:
    """Classify one page from its (low-dpi) OCR text."""
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    head = fold(" ".join(lines[:14]))
    full_lines = [fold(ln) for ln in lines]
    nums = len(GROUPED.findall(text or ""))
    title = next((k for k, pat in TITLE_RULES if re.search(pat, head)), None)
    form = re.search(r"\bB\s?0([2345])\s?A?\s?/\s?TCTD", head)  # SBV form code printed on every page (B02 BS ... B05 notes)
    if title is None and form and form.group(1) in FORM_KINDS and not re.search(r"CHI TIEU NGOAI", head):
        title = FORM_KINDS[form.group(1)]
    consolidation = "consolidated" if "HOP NHAT" in head else ("separate" if re.search(r"\bRIENG\b", head) else None)
    # a table row: the group label starts the line and is followed by amounts, or stands alone on a short line
    # (low-dpi OCR sometimes emits the amount columns separately); narrative policy text never qualifies
    group_rows = sum(1 for g in LOAN_GROUP_LABELS
                     if any(re.match(GROUP_PREFIX + g + AMOUNT_AT_END, ln) or (re.match(GROUP_PREFIX + g + r"\b", ln) and len(ln) <= 40)
                            for ln in full_lines))
    full = " ".join(full_lines)
    loan_quality = (group_rows >= 4 and nums >= 8
                    and ("CHO VAY KHACH HANG" in full or "CHAT LUONG NO CHO VAY" in full))
    notes = bool(re.search(NOTES_TITLE, head)) or bool(form and form.group(1) == "5")
    return {"title": title, "notesTitle": notes, "groupedNumbers": nums, "formCode": form.group(0) if form else None,
            "continued": "TIEP THEO" in head, "consolidation": consolidation, "loanQualityTable": loan_quality,
            "head": head[:300]}


MAX_CONTINUATION = {"balance_sheet": 3, "income_statement": 2, "cash_flow": 2, "off_balance": 1}


def plan_pages(page_infos: list[dict], min_numbers: int = 8) -> dict:
    """Decide the page kind of each classified page (in page order) and which pages need full OCR.

    page_infos: [{"page": n, **classify_text(...)}] in order (may stop early).
    Returns {"kinds": {page: kind}, "notesStart": page|None, "consolidationTitles": [...], "pageConsolidation": {...}}.
    Untitled numeric pages continue the previous statement only for a bounded number of pages (MAX_CONTINUATION):
    when the notes title is unreadable, numeric note pages must not turn into "balance sheet" pages (MB Q1/2025).
    """
    kinds: dict[int, str] = {}
    page_cons: dict[int, str | None] = {}
    last_cons: str | None = None
    last_stmt: str | None = None
    run = 0
    notes_start = None
    titles = []
    for info in page_infos:
        p = info["page"]
        if notes_start is None and info["notesTitle"] and not info["title"]:
            notes_start = p
        if info["loanQualityTable"] and not info["title"]:
            if "note_loan_quality" not in kinds.values():
                kinds[p] = "note_loan_quality"
            last_stmt = None
            continue
        if notes_start is None:
            if info["title"] and info["groupedNumbers"] >= min_numbers:
                kinds[p] = info["title"]
                last_stmt, run = info["title"], 0
                last_cons = info.get("consolidation") or last_cons
                page_cons[p] = info.get("consolidation") or last_cons
                if info.get("consolidation"):
                    titles.append(info["consolidation"])
            elif info["title"] is None and last_stmt and info["groupedNumbers"] >= min_numbers \
                    and run < MAX_CONTINUATION.get(last_stmt, 1):
                kinds[p] = last_stmt  # continuation page (BIDV prints only the running header)
                page_cons[p] = info.get("consolidation") or last_cons
                run += 1
            else:
                last_stmt = None
    return {"kinds": kinds, "notesStart": notes_start, "consolidationTitles": titles, "pageConsolidation": page_cons}


def scan_done(kinds: dict[int, str], notes_start: int | None) -> bool:
    """Stop the cheap pass once the statements and the loan-quality note are found."""
    have = set(kinds.values())
    return {"balance_sheet", "income_statement", "note_loan_quality"} <= have
