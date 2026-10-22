"""equity_facts: column-role / period detection from printed headers (V3-B)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from equity_facts.periods import assign_columns, doc_period, quarter_label


def roles(res):
    return [(c["role"], c["periodLabel"], c["periodType"]) if c else None for c in res["columns"]]


def test_doc_period():
    q2 = doc_period("2026Q2")
    assert (q2.kind, q2.start.isoformat(), q2.end.isoformat()) == ("Q", "2026-04-01", "2026-06-30")
    assert doc_period("2025").end.isoformat() == "2025-12-31"
    assert doc_period("2018H1").end.isoformat() == "2018-06-30"
    with pytest.raises(ValueError):
        doc_period(None)
    assert quarter_label(doc_period("2025").end) == "2025Q4"


def test_mbb_quarter_and_cumulative_words():
    hdr = {"0": "Số phát sinh quý II Năm nay triệu đồng", "1": "Năm trước triệu đồng",
           "2": "Lũy kê từ đầu năm đến cuỗi quý Năm nay triệu đồng", "3": "Lũy kê từ đầu năm đến cuỗi quý Năm trước triệu đồng"}
    res = assign_columns(hdr, 4, "income-statement", doc_period("2026Q2"))
    assert not res["errors"]
    assert roles(res) == [("current", "2026Q2", "quarter"), ("comparative", "2025Q2", "quarter"),
                          ("cumulative-current", "2026-6M", "ytd"), ("cumulative-comparative", "2025-6M", "ytd")]


def test_bid_explicit_date_ranges():
    hdr = {"0": "Trong kỳ Từ 01/04/2026 đến 30/06/2026", "1": "Từ 01/04/2025 đến 30/06/2025",
           "2": "Lũy kế đầu năm đến 01/01/2026 Từ 30/06/2026 đến", "3": "Lũy kế 01/01/2025 Từ 30/06/2025 đến"}
    res = assign_columns(hdr, 4, "income-statement", doc_period("2026Q2"))
    assert not res["errors"]
    assert [c["method"] for c in res["columns"]] == ["dates"] * 4
    assert roles(res)[2] == ("cumulative-current", "2026-6M", "ytd")


def test_q1_two_columns_and_q4_cumulative_label():
    hdr = {"0": "Kỳ này 01/01/2026 đến 31/03/2026", "1": "Kỳ trước 01/01/2025 đến 31/03/2025"}
    assert roles(assign_columns(hdr, 2, "income-statement", doc_period("2026Q1"))) == [
        ("current", "2026Q1", "quarter"), ("comparative", "2025Q1", "quarter")]
    hdr4 = {"0": "Quý IV năm nay", "1": "Quý IV năm trước", "2": "Lũy kế từ đầu năm năm nay", "3": "Lũy kế từ đầu năm năm trước"}
    r = roles(assign_columns(hdr4, 4, "income-statement", doc_period("2025Q4")))
    assert r[2] == ("cumulative-current", "2025", "ytd")  # 12M unaudited != audited annual
    assert r[3] == ("cumulative-comparative", "2024", "ytd")


def test_annual_report_years_and_restated_flag():
    res = assign_columns({"0": "Năm 2025 triệu đồng", "1": "Năm 2024 triệu đồng"}, 2, "income-statement", doc_period("2025"))
    assert roles(res) == [("current", "2025", "annual"), ("comparative", "2024", "annual")]
    res = assign_columns({"0": "Năm nay", "1": "Năm trước (Trình bày lại)"}, 2, "income-statement", doc_period("2025"))
    assert res["columns"][1]["restated"] is True and res["columns"][0]["restated"] is False


def test_cash_flow_two_columns_default_to_cumulative():
    res = assign_columns({}, 2, "cash-flow", doc_period("2026Q2"))
    assert roles(res) == [("cumulative-current", "2026-6M", "ytd"), ("cumulative-comparative", "2025-6M", "ytd")]


def test_balance_sheet_dates_words_and_position():
    res = assign_columns({"0": "30/06/2026 triệu đồng", "1": "31/12/2025 đã kiểm toán"}, 2, "balance-sheet", doc_period("2026Q2"))
    assert roles(res) == [("current", "2026Q2", "point-in-time"), ("comparative", "2025Q4", "point-in-time")]
    assert res["columns"][1]["end"] == "2025-12-31"
    res = assign_columns({"0": "Số cuối năm", "1": "Số đầu năm (Trình bày lại)"}, 2, "balance-sheet", doc_period("2025"))
    assert roles(res) == [("current", "2025Q4", "point-in-time"), ("comparative", "2024Q4", "point-in-time")]
    res = assign_columns({}, 2, "note", doc_period("2026Q1"))
    assert [c["method"] for c in res["columns"]] == ["position", "position"]


def test_inconsistent_headers_fail_closed():
    # header dates of another year -> column unusable, never guessed
    res = assign_columns({"0": "Từ 01/04/2024 đến 30/06/2024", "1": "Năm trước"}, 2, "income-statement", doc_period("2026Q2"))
    assert res["columns"][0] is None and res["errors"]
    # two columns claiming the same period -> both dropped
    res = assign_columns({"0": "30/06/2026", "1": "30/06/2026"}, 2, "balance-sheet", doc_period("2026Q2"))
    assert res["columns"] == [None, None] and any("ambiguous" in e for e in res["errors"])
