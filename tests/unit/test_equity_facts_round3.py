"""equity_facts v3 (round 3): findings of the independent v2 data review."""
import sys
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from equity_facts import derive
from equity_facts.catalog import anchor_metrics, extractable, load_catalog
from equity_facts.extract import _note_amount_columns
from equity_facts.periods import assign_columns, doc_period
from equity_facts.tables import build_table
from equity_facts.textnorm import norm_label
from equity_facts.validate import evaluate


def _row(metric, label, ptype, role, value, doc, avail, audit=None):
    key = f"BID/{doc}/v3/consolidated/{metric}/{label}/{ptype}/{role}"
    return {"Id": key, "FactKey": key, "Symbol": "BID", "MetricCode": metric, "MetricDefinitionVersion": "v1",
            "MetricGroup": "income-statement", "PeriodLabel": label, "PeriodType": ptype, "PeriodStart": None,
            "PeriodEnd": date(2020, 12, 31), "Consolidation": "consolidated", "Value": Decimal(value), "Unit": "VND",
            "Scale": Decimal(1_000_000), "ColumnRole": role, "DocumentId": doc, "Status": "validated",
            "ValidationJson": {"extractionStatus": "validated", "column": {"restated": False}}, "IsDerived": False,
            "DerivationJson": None, "AvailableAtUtc": avail, "DocPeriod": doc, "DocRevision": 1, "ExtractionId": doc,
            "ExtractorVersion": "v3", "ExtractionFinishedAtUtc": avail, "DocAuditStatus": audit}


def test_q4_twelve_month_column_is_superseded_by_the_audited_annual():
    """BID 2020 PAT: Q4 report 12M 7,362,643 (unaudited) vs FY2020 7,223,565 (audited); FY2021's and the 2021Q4
    report's comparative print the audited figure."""
    q4 = _row("PROFIT_AFTER_TAX", "2020", "ytd", "cumulative-current", 7_362_643, "2020Q4", datetime(2021, 1, 30, tzinfo=UTC))
    fy = _row("PROFIT_AFTER_TAX", "2020", "annual", "current", 7_223_565, "2020", datetime(2021, 4, 1, tzinfo=UTC), "audited")
    q4_next = _row("PROFIT_AFTER_TAX", "2020", "ytd", "cumulative-comparative", 7_223_565, "2021Q4",
                   datetime(2022, 1, 30, tzinfo=UTC))
    upd = {f["FactKey"]: (st, vj["crossDocument"]) for f, st, vj in derive.cross_validate([q4, fy, q4_next])}
    assert upd[q4["FactKey"]][0] == "validated"
    assert upd[q4["FactKey"]][1]["supersededBy"] == [fy["FactKey"]]
    assert upd[q4["FactKey"]][1]["supersessionReason"] == "audit-adjustment"
    assert upd[fy["FactKey"]][1]["supersedes"] == [q4["FactKey"]]
    assert upd[q4_next["FactKey"]][0] == "validated"  # agrees with the audited annual: not a mismatch
    # an unaudited later comparative that matches nothing stays review-needed
    odd = dict(q4_next, Value=Decimal(7_000_000), FactKey="x", Id="x")
    assert {f["FactKey"]: st for f, st, _ in derive.cross_validate([q4, fy, odd])}["x"] == "review-needed"


def _fact(metric, value):
    return {"metric": metric, "value": value, "periodLabel": "2019Q4", "periodType": "point-in-time", "role": "current",
            "end": "2019-12-31", "cons": "consolidated"}


def test_mbs_line_inside_or_outside_group1_is_decided_by_exact_arithmetic():
    groups = {"LOAN_GROUP1": 244_536_712, "NPL_GROUP2": 3_000_000, "NPL_GROUP3": 1_000_000, "NPL_GROUP4": 800_000,
              "NPL_GROUP5": 304_512}
    margin = 2_689_399
    total = sum(groups.values()) + margin  # MBB 2019Q4 p24: groups + MBS line = total
    facts = [_fact(m, v) for m, v in groups.items()] + [_fact("MARGIN_LOANS_GROUP1", margin), _fact("LOAN_QUALITY_TOTAL", total)]
    evaluate(facts, [], [])
    assert all(f["status"] == "validated" for f in facts)
    assert not any(f.get("marginInGroup1") for f in facts)
    # MBB 2019Q3 p24: "+ Trong đó … tại MBS" already inside group 1 -> groups = total, line is not this metric
    facts = [_fact(m, v) for m, v in groups.items()] + [_fact("MARGIN_LOANS_GROUP1", margin),
                                                        _fact("LOAN_QUALITY_TOTAL", sum(groups.values()))]
    evaluate(facts, [], [])
    m = next(f for f in facts if f["metric"] == "MARGIN_LOANS_GROUP1")
    assert m["marginInGroup1"] and all(f["status"] == "validated" for f in facts if f is not m)
    # neither closes -> still a failure (nothing loosened)
    facts = [_fact(m, v) for m, v in groups.items()] + [_fact("MARGIN_LOANS_GROUP1", margin),
                                                        _fact("LOAN_QUALITY_TOTAL", total + 5_000)]
    evaluate(facts, [], [])
    assert all(f["status"] == "quarantined" for f in facts)


def test_mb_label_variants_for_parent_profit_and_the_mbs_line():
    cat = load_catalog()
    is_metrics = extractable(cat, "MBB", "income_statement")
    rows = [{"page": 13, "y": 2361.0, "labelNorm": [norm_label("Lợi nhuận ròng trong năm")], "cells": {"0": {"kind": "number"}}}]
    assert "PROFIT_AFTER_TAX_PARENT" in anchor_metrics(rows, is_metrics, "MBB")
    note = extractable(cat, "MBB", "note_loan_quality")
    for label in ("+ Trong đó các khoản cho vay tại MBS", "Các khoản cho vay giao dịch ký quỹ và ứng trước cho khách hàng",
                  "Các hợp đồng REPO, hỗ trợ tài chính và ứng trước cho khách hàng của MBS"):
        rows = [{"page": 24, "y": 2261.0, "labelNorm": [norm_label(label)], "cells": {"0": {"kind": "number"}}}]
        assert "MARGIN_LOANS_GROUP1" in anchor_metrics(rows, note, "MBB"), label


def test_implausible_header_date_falls_back_and_is_recorded():
    """MBB 2026H1 p13: 'kết thúc ngày 30/6/2026' read as '30/6/2026 30/8/2028'."""
    res = assign_columns({"0": "Kỳ sáu tháng 30/6/2026 30/8/2028", "1": "Kỳ sáu tháng 30/6/2025 30/6/2025"}, 2,
                         "income-statement", doc_period("2026H1"))
    c0, c1 = res["columns"]
    assert (c0["role"], c0["periodLabel"], c1["role"], c1["periodLabel"]) == ("current", "2026H1", "comparative", "2025H1")
    assert c0["implausibleDates"] == ["2028-08-30"] and "implausible-dates-dropped" in c0["method"]
    # every date of another period: still fail closed
    res = assign_columns({"0": "Từ 01/04/2024 đến 30/06/2024", "1": "Năm trước"}, 2, "income-statement", doc_period("2026Q2"))
    assert res["columns"][0] is None


def _w(text, x0, y, line, w=None, h=24):
    return {"text": text, "box": [x0, y - h // 2, x0 + (w or 14 * len(text)), y + h // 2], "line": line}


def test_small_printed_dash_is_a_nil_cell():
    """BID 2018H1 p14: current-column '-' read by Tesseract as a 13x5 px quote mark."""
    words, boxes = [], []
    rows = [("Tiền thu từ cổ tức", "37.016", "49.188"), ("Mua sắm tài sản cố định", "(512.711)", "(260.541)"),
            ("Lưu chuyển tiền thuần từ hoạt động đầu tư", "(471.917)", "371.150"),
            ("Tăng vốn cổ phần", None, "223.951"), ("Cổ tức trả cho cổ đông", None, "(11.447)"),
            ("Lưu chuyển tiền thuần từ hoạt động tài chính", None, "212.504"),
            ("Lưu chuyển tiền thuần trong kỳ", "12.524.168", "2.648.426"),
            ("Tiền và tương đương tiền đầu kỳ", "60.110.234", "57.461.808")]
    for i, (lab, a, b) in enumerate(rows):
        y = 1000 + 60 * i
        x = 300
        for tok in lab.split():
            words.append(_w(tok, x, y, f"1.1.{i}"))
            x += 14 * len(tok) + 12
        for right, v in ((1860, a), (2190, b)):
            if v:
                box = [right - 14 * len(v), y - 12, right, y + 12]
                words.append({"text": v, "box": box, "line": f"1.1.{i}"})
                boxes.append({"text": v, "box": box})
        if a is None:
            words.append(_w("“", 1830, y + 3, f"1.1.{i}", w=13, h=5))
    table = build_table(words, boxes, 2400, 3400)
    fin = next(r for r in table["rows"] if r["label"] == "Lưu chuyển tiền thuần từ hoạt động tài chính")
    assert fin["cells"]["0"]["kind"] == "dash" and fin["cells"]["1"]["value"] == 212504
    inv = next(r for r in table["rows"] if r["label"] == "Lưu chuyển tiền thuần từ hoạt động đầu tư")
    assert inv["cells"]["0"]["value"] == -471917


def test_loan_note_amount_columns_fallback():
    """BID 2021Q2 p18: each date header is centred over 'amount | %' -> the amount columns are left unresolved."""
    hdr = {"0": "30/06/2021 30/06/2021", "2": "31/12/2020 31/12/2020"}
    res = assign_columns(hdr, 4, "note", doc_period("2021Q2"))
    vals = [(1_262_191_978, 1_179_268_926), (14_112_507, 13_657_572), (3_330_642, 2_382_136), (1_000_000, 900_000),
            (9_000_000, 8_000_000)]
    group_rows = [{"page": 18, "cells": {"1": {"kind": "number", "value": a}, "3": {"kind": "number", "value": b}}}
                  for a, b in vals]
    out = _note_amount_columns(res, hdr, 4, doc_period("2021Q2"), group_rows)
    assert out["columns"][1]["role"] == "current" and out["columns"][1]["end"] == "2021-06-30"
    assert out["columns"][3]["role"] == "comparative" and out["columns"][3]["end"] == "2020-12-31"
    assert out["columns"][0] is None and out["columns"][1]["method"] == "amount-columns+header"
    # percentages in the second column of each pair are never taken as amounts
    pct = [{"page": 18, "cells": {"0": {"kind": "number", "value": a}, "1": {"kind": "number", "value": 95}}} for a, _ in vals]
    assert _note_amount_columns(res, hdr, 4, doc_period("2021Q2"), pct) is res


def test_equity_section_header_with_ocr_noise():
    from equity_facts.catalog import tag_bs_sections

    for head in ("VAN VA CAC QUY", "MON CHU SO HUU", "VON CHU SO HUU"):
        rows = [{"labelNorm": ["NO PHAI TRA"]}, {"labelNorm": ["TIEN GUI CUA KHACH HANG"]}, {"labelNorm": [head]},
                {"labelNorm": ["VON DIEU LE"]}]
        tag_bs_sections(rows)
        assert [r["section"] for r in rows] == ["liabilities", "liabilities", "equity", "equity"], head


def test_unreadable_optional_margin_line_does_not_block_closing_groups():
    groups = {"LOAN_GROUP1": 208_168_412, "NPL_GROUP2": 3_657_858, "NPL_GROUP3": 1_099_661, "NPL_GROUP4": 797_692,
              "NPL_GROUP5": 962_335}
    for total, want in ((sum(groups.values()), "validated"), (sum(groups.values()) + 2_300_284, "extracted")):
        facts = [_fact(m, v) for m, v in groups.items()] + [_fact("LOAN_QUALITY_TOTAL", total)]
        bad = [{**_fact("MARGIN_LOANS_GROUP1", None), "raw": "230284"}]
        evaluate(facts, [], bad)
        assert all(f["status"] == want for f in facts), (total, [f["status"] for f in facts])


def test_loan_note_amount_columns_from_words_or_years():
    vals = [(1_262_191_237, 1_179_268_926), (14_077_536, 13_657_572), (3_335_142, 2_382_136), (2_150_626, 2_462_228),
            (15_690_546, 16_525_054)]
    rows = [{"page": 32, "cells": {"0": {"kind": "number", "value": a}, "1": {"kind": "malformed", "value": None},
                                   "3": {"kind": "number", "value": b}}} for a, b in vals]
    hdr = {"0": "Số cuối kỳ Triệu VND", "1": "Số cuối kỳ %", "2": "Số đầu kỳ Triệu VND", "3": "Số đầu kỳ Triệu VND", "4": "%"}
    dp = doc_period("2021H1")
    res = assign_columns(hdr, 5, "note", dp)  # BID 2021H1 p32: duplicates -> ambiguous -> nothing resolved
    out = _note_amount_columns(res, hdr, 5, dp, rows)
    assert (out["columns"][0]["end"], out["columns"][3]["end"]) == ("2021-06-30", "2020-12-31")
    swapped = {"0": "Số đầu kỳ", "3": "Số cuối kỳ"}
    assert _note_amount_columns(res, swapped, 5, dp, rows) is res  # headers contradict left = current


def test_operating_cash_flow_is_not_lost_to_the_investing_line():
    """BID FY2016 p11/p12: OCR noise puts the investing line within 0.06 of the operating label."""
    cat = load_catalog()
    ms = extractable(cat, "BID", "cash_flow")
    labels = ["Lưu chuyển tiền thuần từ hoạt động kinh doanh trước những thay đổi về tài sản và công nợ hoạt động",
              "Lưu chuyển tiễn thuẳn từ huạt động kinh duanh", "Lưu chuyển tiền thuần từ huạt động đầu tư",
              "Lưu chuyển tiền thuần từ hoạt động tài chính", "Lưu chuyển tiền thuần trong năm"]
    rows = [{"page": 11, "index": i, "y": 100.0 * i, "labelNorm": [norm_label(t)], "cells": {"0": {"kind": "number"}}}
            for i, t in enumerate(labels)]
    got = {k: v["row"]["index"] for k, v in anchor_metrics(rows, ms, "BID").items()}
    assert got["NET_CASH_FROM_OPERATING"] == 1 and got["NET_CASH_FROM_INVESTING"] == 2


def test_comparative_quarter_matching_the_audited_derivation_is_not_review_needed():
    def mk(label, ptype, role, value, doc, avail, audit=None):
        return dict(_row("INTEREST_INCOME", label, ptype, role, value, doc, avail, audit), DocPeriod=doc)
    fy = mk("2020", "annual", "current", 100_687_502, "2020", datetime(2021, 4, 1, tzinfo=UTC), "audited")
    m9 = mk("2020-9M", "ytd", "cumulative-current", 74_728_327, "2020Q3", datetime(2020, 10, 30, tzinfo=UTC))
    q4 = mk("2020Q4", "quarter", "current", 25_959_761, "2020Q4", datetime(2021, 1, 30, tzinfo=UTC))
    cmp21 = mk("2020Q4", "quarter", "comparative", 25_959_175, "2021Q4", datetime(2022, 1, 30, tzinfo=UTC))
    facts = [fy, m9, q4, cmp21]
    upd = derive.cross_validate(facts)
    assert {f["FactKey"]: st for f, st, _ in upd}[cmp21["FactKey"]] == "review-needed"
    status_of = {f["FactKey"]: f["Status"] for f in facts} | {f["FactKey"]: st for f, st, _ in upd}
    upd = derive.reconcile_with_derived(upd, derive.plan_derivations(facts, status_of))
    st, vj = {f["FactKey"]: (st, vj) for f, st, vj in upd}[cmp21["FactKey"]]
    assert st == "validated" and vj["crossDocument"]["supersessionReason"] == "audit-adjustment"
    off = dict(cmp21, Value=Decimal(25_000_000), FactKey="off", Id="off")
    upd = derive.cross_validate([fy, m9, q4, off])
    upd = derive.reconcile_with_derived(upd, derive.plan_derivations([fy, m9, q4, off], status_of))
    assert {f["FactKey"]: st for f, st, _ in upd}["off"] == "review-needed"


def test_six_month_column_is_compared_with_the_reviewed_half_year_report():
    """BID 2017Q2 report: 6M-2016 comparative = reviewed H1 2016 report, not the earlier Q2 report's 6M column."""
    q2 = _row("NET_INTEREST_INCOME", "2016-6M", "ytd", "cumulative-current", 10_334_096, "2016Q2", datetime(2016, 7, 30, tzinfo=UTC))
    h1 = _row("NET_INTEREST_INCOME", "2016H1", "half-year", "current", 10_201_903, "2016H1", datetime(2016, 8, 30, tzinfo=UTC),
              "reviewed")
    cmp17 = _row("NET_INTEREST_INCOME", "2016-6M", "ytd", "cumulative-comparative", 10_201_903, "2017Q2",
                 datetime(2017, 7, 30, tzinfo=UTC))
    upd = {f["FactKey"]: (st, vj["crossDocument"]) for f, st, vj in derive.cross_validate([q2, h1, cmp17])}
    assert upd[cmp17["FactKey"]][0] == "validated"
    assert upd[q2["FactKey"]][0] == "validated" and upd[q2["FactKey"]][1]["supersededBy"] == [h1["FactKey"]]


def test_group_rows_with_nhom_prefix_are_a_loan_quality_table():
    """MBB FY2016 p39: 'Nhóm 1 - Nợ đủ tiêu chuẩn 144.555.920 115.624.100'."""
    from equity_facts.classify import classify_text

    text = ("Thuyết minh báo cáo tài chính hợp nhất\nCho vay khách hàng\n"
            "Nhóm I - Nợ đủ tiêu chuẩn 144.555.920 115.624.100\nNhóm 2 - Nợ cân chú ý 1.904.761 2.381.530\n"
            "Nhóm 3 - Nợ dưới tiêu chuẩn 896.027 425.343\nNhóm 4 - Nợ nghi ngờ 476.547 442.136\n"
            "Nhóm 5 - Nợ có khả năng mất vốn 614.623 1.082.115\n150.737.702 121.348.630\n")
    assert classify_text(text)["loanQualityTable"]
    assert norm_label("Nhóm 1 - Nợ đủ tiêu chuẩn") == "NO DU TIEU CHUAN"
    assert norm_label("Nhằm 4 - Nợ nghỉ ngữ") == "NO NGHI NGU"
    assert norm_label("Nhóm Nơ nghi ngò") == "NO NGHI NGO"  # RapidOCR dropped the number
