"""equity_facts v2 (round 2): regressions found on the full v1 backfill."""
import math
import subprocess
import sys
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from equity_facts import derive, ocr
from equity_facts.classify import classify_text, plan_pages
from equity_facts.periods import assign_columns, doc_period
from equity_facts.tables import build_table
from equity_facts.textnorm import detect_unit


def test_tesseract_near_empty_page_is_no_text_not_a_failure(monkeypatch):
    """BID 2020Q2/Q3 v1 failures: tesseract exits 1 with "Too few characters. Skipping this page"."""
    from PIL import Image

    def fake_run(*a, **k):
        return subprocess.CompletedProcess(a, 1, b"", b"Too few characters. Skipping this page\nError during processing.")

    monkeypatch.setattr(ocr.subprocess, "run", fake_run)
    img = Image.new("L", (50, 50), 255)
    assert ocr.tesseract_text(img, 150) == ""
    assert ocr.tesseract_words(img, 300) == []

    def broken(*a, **k):
        return subprocess.CompletedProcess(a, 1, b"", b"Error opening data file vie.traineddata")

    monkeypatch.setattr(ocr.subprocess, "run", broken)
    try:
        ocr.tesseract_text(img, 150)
    except RuntimeError as exc:
        assert "traineddata" in str(exc)
    else:
        raise AssertionError("a real tesseract error must still raise")


def test_deskew_rotates_the_right_way():
    """v1 rotated by -angle and doubled the skew (BID 2025Q2 loan note: labels drifted half a row)."""
    from PIL import Image, ImageDraw

    for ang in (0.8, -1.2):
        img = Image.new("L", (1600, 1200), 255)
        d = ImageDraw.Draw(img)
        for y0 in range(100, 1100, 50):
            for x in range(50, 1550, 4):
                y = y0 + math.tan(math.radians(ang)) * x
                d.rectangle([x, y, x + 2, y + 6], fill=0)
        est = ocr.estimate_skew(img)
        assert abs(est - ang) <= 0.2
        assert abs(ocr.estimate_skew(ocr.deskew(img, est))) <= 0.2


def test_units_on_old_bidv_scans():
    assert detect_unit("Em vị: Triệu đẳng")[0] == 1_000_000


def test_cash_flow_title_with_ocr_noise_and_page_consolidation():
    nums = "\n".join(f"Dòng {i} 1.234.567 (2.345.678)" for i in range(10))
    cf = classify_text("BÁO CÁO LƯU CHUYEEN TIỀN TỆ HỢP NHẤT\n" + nums)
    assert cf["title"] == "cash_flow"
    pages = [{"page": 1, **classify_text("BÁO CÁO TÌNH HÌNH TÀI CHÍNH RIÊNG\n" + nums)},
             {"page": 2, **classify_text("Báo cáo tài chính\n" + nums)},
             {"page": 3, **classify_text("BÁO CÁO TÌNH HÌNH TÀI CHÍNH HỢP NHẤT\n" + nums)}]
    plan = plan_pages(pages)
    assert plan["pageConsolidation"] == {1: "separate", 2: "separate", 3: "consolidated"}


def test_restatement_layout_keeps_only_the_restated_column():
    hdr = {"0": "2024 Năm (Số liệu đã công bố)", "1": "Điều chỉnh của KTNN", "2": "Đơn vị: triệu đồng 2024 Năm (Số trình bày lại)"}
    bs = assign_columns(hdr, 3, "balance-sheet", doc_period("2024"))
    assert bs["columns"][:2] == [None, None] and bs["skipped"] == {"0": "published", "1": "adjustment"}
    c = bs["columns"][2]
    assert (c["role"], c["periodLabel"], c["periodType"], c["restated"], c["method"]) == (
        "current", "2024Q4", "point-in-time", True, "restatement-layout")
    is_ = assign_columns(hdr, 3, "income-statement", doc_period("2024"))["columns"][2]
    assert (is_["periodLabel"], is_["periodType"]) == ("2024", "annual")
    # an ordinary restated comparative column is NOT the restatement layout
    fy = assign_columns({"0": "Năm nay", "1": "Năm trước (Trình bày lại)"}, 2, "income-statement", doc_period("2025"))
    assert [c["role"] for c in fy["columns"]] == ["current", "comparative"]


def _w(text, x0, y, line):
    return {"text": text, "box": [x0, y - 12, x0 + 14 * len(text), y + 12], "line": line}


def test_label_rows_follow_tesseract_lines_under_residual_skew():
    """Labels sit half a row lower than their amounts (skewed scan): Tesseract's line id still pairs them."""
    words, boxes = [], []
    labels = ["Nợ đủ tiêu chuẩn", "Nợ cần chú ý", "Nợ dưới tiêu chuẩn", "Nợ nghi ngờ", "Nợ có khả năng mất vốn"]
    amounts = ["2.103.191.893", "33.547.271", "6.392.458", "8.046.072", "28.701.914"]
    for i, (lab, amt) in enumerate(zip(labels, amounts)):
        y = 1000 + 60 * i
        x = 300
        for tok in lab.split():
            words.append(_w(tok, x, y + 28, f"1.1.{i}"))
            x += 14 * len(tok) + 12
        words.append(_w(amt, 1500, y, f"1.1.{i}"))
        boxes.append({"text": amt, "box": [1500, y - 12, 1500 + 14 * len(amt), y + 12]})
    table = build_table(words, boxes, 2400, 3400)
    got = {r["label"]: r["cells"]["0"]["raw"] for r in table["rows"] if r["cells"]}
    assert got == dict(zip(labels, amounts))


def test_unreadable_cell_is_reread_and_needs_two_agreeing_readings():
    from equity_facts.extract import _reread_cells

    words, boxes = [], []
    for i in range(4):
        y = 500 + 60 * i
        words.append(_w(f"Dòng{i}", 300, y, f"1.1.{i}"))
        boxes.append({"text": f"{i + 1}.234.567", "box": [1500, y - 12, 1682, y + 12]})
        words.append(_w(f"{i + 1}.234.567", 1500, y, f"1.1.{i}"))
    boxes[3]["text"] = "776.657846"            # RapidOCR dropped a separator
    words[-1]["text"] = "776.657.846"          # Tesseract read the same cell cleanly

    class Eng:
        def __init__(self, reads):
            self.reads = reads

        def reread(self, pages, page_no, cell_boxes, skew):
            return [self.reads for _ in cell_boxes]

    def cell_after(reads):
        table = build_table(words, boxes, 2400, 3400)
        cell = [r for r in table["rows"] if r["label"] == "Dòng3"][0]["cells"]["0"]
        assert cell["kind"] == "malformed" and cell["needsReread"]
        info = _reread_cells(Eng(reads), None, 1, table, 0.0)
        return cell, info

    cell, info = cell_after(["776.657.846", "776.657846"])   # one re-read + Tesseract agree
    assert (cell["kind"], cell["value"], cell["engine"], cell["rapidRaw"]) == ("number", 776657846, "rapid-reread", "776.657846")
    assert info == {"cells": 1, "resolved": 1}
    cell, info = cell_after(["7'5376", ""])                 # only Tesseract is well-formed -> still unreadable
    assert cell["kind"] == "malformed" and cell["value"] is None and info["resolved"] == 0
    cell, _ = cell_after(["776.657.846", "776.657.846"])    # two re-reads agree, Tesseract differs: accepted
    assert cell["value"] == 776657846


def _row(metric, label, ptype, role, value, doc, avail, restated=False, ext="e1", fin=datetime(2026, 10, 19, tzinfo=UTC),
         audit=None):
    key = f"BID/{doc}/{ext}/consolidated/{metric}/{label}/{ptype}/{role}"
    return {"Id": key, "FactKey": key, "Symbol": "BID", "MetricCode": metric, "MetricDefinitionVersion": "v1",
            "MetricGroup": "income-statement", "PeriodLabel": label, "PeriodType": ptype, "PeriodStart": None,
            "PeriodEnd": date(2024, 12, 31), "Consolidation": "consolidated", "Value": Decimal(value), "Unit": "VND",
            "Scale": Decimal(1_000_000), "ColumnRole": role, "DocumentId": doc, "Status": "validated",
            "ValidationJson": {"extractionStatus": "validated", "column": {"restated": restated}}, "IsDerived": False,
            "DerivationJson": None, "AvailableAtUtc": avail, "DocPeriod": label, "DocRevision": 1, "ExtractionId": ext,
            "ExtractorVersion": ext, "ExtractionFinishedAtUtc": fin, "DocAuditStatus": audit}


def test_declared_restatements_are_not_review_needed():
    orig = _row("PROFIT_BEFORE_TAX", "2024", "annual", "current", 100, "fy2024", datetime(2025, 3, 31, tzinfo=UTC))
    later_cmp = _row("PROFIT_BEFORE_TAX", "2024", "annual", "comparative", 95, "fy2025", datetime(2026, 3, 31, tzinfo=UTC),
                     restated=True)
    restated_doc = _row("PROFIT_BEFORE_TAX", "2024", "annual", "current", 95, "fy2024-restated",
                        datetime(2025, 10, 23, tzinfo=UTC), restated=True)
    upd = {f["FactKey"]: (st, vj["crossDocument"]) for f, st, vj in derive.cross_validate([orig, later_cmp, restated_doc])}
    assert upd[orig["FactKey"]][0] == "validated" and upd[orig["FactKey"]][1]["supersededBy"] == sorted([restated_doc["FactKey"], later_cmp["FactKey"]])
    assert upd[orig["FactKey"]][1]["supersessionReason"] == "restatement"
    assert upd[later_cmp["FactKey"]][0] == "validated"
    unflagged = dict(later_cmp, ValidationJson={"extractionStatus": "validated", "column": {"restated": False}})
    upd = {f["FactKey"]: st for f, st, _ in derive.cross_validate([orig, unflagged])}
    assert upd[unflagged["FactKey"]] == "review-needed"


def test_only_the_latest_extraction_per_document_is_cross_checked():
    old = _row("NET_INTEREST_INCOME", "2025Q2", "quarter", "current", 999, "q2", datetime(2025, 7, 30, tzinfo=UTC),
               ext="v1", fin=datetime(2026, 10, 19, 10, tzinfo=UTC))
    new = _row("NET_INTEREST_INCOME", "2025Q2", "quarter", "current", 100, "q2", datetime(2025, 7, 30, tzinfo=UTC),
               ext="v2", fin=datetime(2026, 10, 20, tzinfo=UTC))
    derived_old = dict(old, FactKey="BID/NII/derived/eqx-derive-v1/x", IsDerived=True,
                       DerivationJson={"derivationVersion": "eqx-derive-v1"})
    assert [f["FactKey"] for f in derive.current_facts([old, new, derived_old])] == [new["FactKey"]]


def test_form_code_fallback_and_bounded_continuation():
    nums = "\n".join(f"Dòng {i} 1.234.567 (2.345.678)" for i in range(10))
    cf = classify_text("Ngân hàng TMCP Quân đội B04a/TCTD-HN\nBÁO CÁO LƯU GHUYẾN TIỀN TỆ HỢP NHÁT\n" + nums)
    assert cf["title"] == "cash_flow"                       # "LƯU GHUYẾN" (C read as G) and the B04a form code
    assert classify_text("Ngân hàng B03a/TCTD-HN\nxyz\n" + nums)["title"] == "income_statement"
    assert classify_text("Ngân hàng B05a/TCTD-HN\n1. Đặc điểm\n" + nums)["notesTitle"] is True
    pages = [{"page": 1, **classify_text("BÁO CÁO TÌNH HÌNH TÀI CHÍNH HỢP NHẤT\n" + nums)}]
    pages += [{"page": p, **classify_text("Ngân hàng\n" + nums)} for p in range(2, 9)]  # notes title unreadable
    kinds = plan_pages(pages)["kinds"]
    assert sorted(kinds) == [1, 2, 3, 4]                     # continuation bounded (3 pages), notes not swallowed


def test_sideways_page_detection_heuristic():
    from equity_facts.extract import _looks_upright

    assert _looks_upright("Ngân hàng Thương mại Cổ phần Quân đội\nBÁO CÁO KẾT QUẢ HOẠT ĐỘNG")
    assert not _looks_upright('"=ma sex# s%œsnh" s“ ® ®$\n>> >> zx>z è š š\ns -\nH kị')


def test_mb_group1_definitions_are_never_compared():
    """v4: the quarterly 'incl. margin' figure and the annual 'excl. margin' figure are two metrics (round-4 split)."""
    q = _row("LOAN_GROUP1_INCL_MARGIN", "2025Q4", "point-in-time", "current", 1_059_781_834, "q4", datetime(2026, 1, 28, tzinfo=UTC))
    fy = _row("LOAN_GROUP1", "2025Q4", "point-in-time", "current", 1_044_741_249, "fy", datetime(2026, 3, 11, tzinfo=UTC))
    upd = {f["FactKey"]: st for f, st, _ in derive.cross_validate([q, fy])}
    assert upd == {}  # different definitions: nothing to compare, nothing demoted


def test_liabilities_section_survives_ocr_noise_in_its_header():
    from equity_facts.catalog import tag_bs_sections

    rows = [{"labelNorm": [t]} for t in ("TIEN MAT VANG BAC", "NO PHAAI TRA VA VON CHU SO HUU", "TIEN GUI CUA KHACH HANG",
                                         "VON VA CAC QUY")]
    tag_bs_sections(rows)
    assert [r["section"] for r in rows] == ["assets", "liabilities", "liabilities", "equity"]
    rows = [{"labelNorm": ["CAC KHOAN PHAI THU"]}, {"labelNorm": ["CHI PHI TRA TRUOC"]}]
    tag_bs_sections(rows)
    assert [r["section"] for r in rows] == ["assets", "assets"]
    rows = [{"labelNorm": ["NOPHAI TRA VA VON CHU SO HUU"]}]  # BIDV 2026Q1 p5: words run together
    tag_bs_sections(rows)
    assert rows[0]["section"] == "liabilities"


def test_ocr_mangled_roman_numbering_is_stripped():
    from equity_facts.textnorm import norm_label

    assert norm_label("XIHI. Lợi nhuận sau thuế TNDN") == "LOI NHUAN SAU THUE TNDN"
    assert norm_label("IHI Tiền gửi của khách hàng") == "TIEN GUI CUA KHACH HANG"
    assert norm_label("HII. Tiền gửi và cho vay các TCTD khác") == "TIEN GUI VA CHO VAY CAC TCTD KHAC"


def test_wrapped_label_with_numbering_on_its_second_line():
    """BIDV 2024 restated p6: '... và cho vay các' / 'III. khác' — the STT sits on the 2nd line, mid-label."""
    words, boxes = [], []
    labels = ["HI. khác", "Nợ cần chú ý", "Nợ dưới tiêu chuẩn", "Nợ nghi ngờ", "Nợ có khả năng mất vốn"]
    amounts = ["279.971.515", "33.547.271", "6.392.458", "8.046.072", "28.701.914"]
    for i, (lab, amt) in enumerate(zip(labels, amounts)):
        y = 1000 + 80 * i
        x = 300
        for tok in lab.split():
            words.append(_w(tok, x, y, f"1.1.{i}"))
            x += 14 * len(tok) + 12
        words.append(_w(amt, 1500, y, f"1.1.{i}"))
        boxes.append({"text": amt, "box": [1500, y - 12, 1500 + 14 * len(amt), y + 12]})
    x = 300
    for tok in ["Tiền,", "vàng", "gửi", "tại", "các", "TCTD", "khác", "và", "cho", "vay", "các"]:
        words.append(_w(tok, x, 970, "1.0.9"))
        x += 14 * len(tok) + 12
    row = next(r for r in build_table(words, boxes, 2400, 3400)["rows"] if r["cells"])
    assert "TIEN VANG GUI TAI CAC TCTD KHAC VA CHO VAY CAC KHAC" in row["labelNorm"]


def test_audit_adjustment_is_supersession_not_review():
    q4 = _row("TOTAL_ASSETS", "2024Q4", "point-in-time", "current", 2_760_693_352, "q4", datetime(2025, 1, 24, tzinfo=UTC))
    fy = _row("TOTAL_ASSETS", "2024Q4", "point-in-time", "current", 2_760_791_854, "fy", datetime(2025, 3, 28, tzinfo=UTC),
              audit="audited")
    upd = {f["FactKey"]: (st, vj["crossDocument"]) for f, st, vj in derive.cross_validate([q4, fy])}
    assert upd[q4["FactKey"]][0] == "validated"
    assert upd[q4["FactKey"]][1]["supersededBy"] == [fy["FactKey"]]
    assert upd[q4["FactKey"]][1]["supersessionReason"] == "audit-adjustment"
    assert upd[fy["FactKey"]][0] == "validated" and upd[fy["FactKey"]][1]["supersedes"] == [q4["FactKey"]]
    # audited FY report's comparative revising the unaudited Q4 report: also supersession
    fy_cmp = dict(fy, FactKey="BID/fy25/e1/consolidated/TOTAL_ASSETS/2024Q4/point-in-time/comparative", ColumnRole="comparative",
                  DocumentId="fy25", AvailableAtUtc=datetime(2026, 3, 30, tzinfo=UTC))
    upd = {f["FactKey"]: st for f, st, _ in derive.cross_validate([q4, fy_cmp])}
    assert upd[fy_cmp["FactKey"]] == "validated" and upd[q4["FactKey"]] == "validated"


def test_same_assurance_revision_stays_review_needed():
    q2 = _row("FEE_INCOME", "2025Q2", "quarter", "current", 3_590_791, "q2", datetime(2025, 7, 30, tzinfo=UTC))
    cmp26 = _row("FEE_INCOME", "2025Q2", "quarter", "comparative", 3_590_607, "q2-2026", datetime(2026, 7, 30, tzinfo=UTC))
    later_q = _row("FEE_INCOME", "2025Q2", "quarter", "current", 3_590_000, "q2-dup", datetime(2025, 8, 30, tzinfo=UTC))
    upd = {f["FactKey"]: (st, vj["crossDocument"]) for f, st, vj in derive.cross_validate([q2, cmp26, later_q])}
    assert upd[cmp26["FactKey"]][0] == "review-needed"                    # unaudited vs unaudited, no restatement note
    assert upd[q2["FactKey"]][0] == "review-needed" and "supersededBy" not in upd[q2["FactKey"]][1]


def test_derived_q4_after_audit_supersedes_the_unaudited_q4_report():
    def mk(label, ptype, value, doc_period, avail, audit=None):
        return dict(_row("PROFIT_BEFORE_TAX", label, ptype, "current" if ptype != "ytd" else "cumulative-current", value,
                         doc_period, avail, audit=audit), DocPeriod=doc_period)
    fy = mk("2024", "annual", 1000, "2024", datetime(2025, 3, 28, tzinfo=UTC), audit="audited")
    m9 = mk("2024-9M", "ytd", 700, "2024Q3", datetime(2024, 10, 30, tzinfo=UTC))
    q4 = mk("2024Q4", "quarter", 290, "2024Q4", datetime(2025, 1, 24, tzinfo=UTC))
    facts = [fy, m9, q4]
    d = [x for x in derive.plan_derivations(facts, {}) if x["PeriodLabel"] == "2024Q4"]
    assert d and d[0]["Status"] == "validated" and d[0]["Value"] == 300
    assert d[0]["ValidationJson"]["printedQuarter"]["supersedes"] == [q4["FactKey"]]
    assert derive.superseded_by_derived(d)[q4["FactKey"]][0] == [d[0]["FactKey"]]
    fy_unaudited = dict(fy, DocAuditStatus=None, AvailableAtUtc=datetime(2025, 1, 20, tzinfo=UTC))
    d = [x for x in derive.plan_derivations([fy_unaudited, m9, q4], {}) if x["PeriodLabel"] == "2024Q4"]
    assert d[0]["Status"] == "review-needed"
