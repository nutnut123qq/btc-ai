"""equity_facts v4 (round 4): findings of the v3 delta review."""
import sys
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from equity_facts import derive
from equity_facts.catalog import WEAK_THRESHOLD, anchor_metrics, extractable, load_catalog
from equity_facts.extract import _assign_group1_definition
from equity_facts.textnorm import norm_label


def _row(metric, label, ptype, role, value, doc, avail, audit=None, status="validated", sym="BID", restated=False):
    key = f"{sym}/{doc}/v4/consolidated/{metric}/{label}/{ptype}/{role}"
    return {"Id": key, "FactKey": key, "Symbol": sym, "MetricCode": metric, "MetricDefinitionVersion": "v1",
            "MetricGroup": "income-statement", "PeriodLabel": label, "PeriodType": ptype, "PeriodStart": None,
            "PeriodEnd": date(2017, 12, 31), "Consolidation": "consolidated", "Value": Decimal(value), "Unit": "VND",
            "Scale": Decimal(1_000_000), "ColumnRole": role, "DocumentId": doc, "Status": status,
            "ValidationJson": {"extractionStatus": status, "column": {"restated": restated}}, "IsDerived": False,
            "DerivationJson": None, "AvailableAtUtc": avail, "DocPeriod": doc, "DocRevision": 1, "ExtractionId": doc,
            "ExtractorVersion": "v4", "ExtractionFinishedAtUtc": avail, "DocAuditStatus": audit}


def test_audited_annual_without_identity_is_confirmed_by_later_reports():
    """BIDV FY2017 interest income 78,628,515: identity could not run (extracted); two later reports print the same
    figure; the earlier unaudited 2017Q4 12M column (78,682,431) is what it supersedes, so it must not block."""
    q4 = _row("INTEREST_INCOME", "2017", "ytd", "cumulative-current", 78_682_431, "2017Q4", datetime(2018, 1, 30, tzinfo=UTC))
    fy = _row("INTEREST_INCOME", "2017", "annual", "current", 78_628_515, "2017", datetime(2018, 3, 30, tzinfo=UTC),
              "audited", status="extracted")
    c1 = _row("INTEREST_INCOME", "2017", "annual", "comparative", 78_628_515, "2018", datetime(2019, 3, 30, tzinfo=UTC), "audited")
    c2 = _row("INTEREST_INCOME", "2017", "ytd", "cumulative-comparative", 78_628_515, "2018Q4", datetime(2019, 1, 30, tzinfo=UTC))
    upd = {f["FactKey"]: (st, vj["crossDocument"]) for f, st, vj in derive.cross_validate([q4, fy, c1, c2])}
    st, cross = upd[fy["FactKey"]]
    assert st == "validated" and cross["promoted"]["confirmedBy"] == sorted([c1["FactKey"], c2["FactKey"]])
    assert cross["supersedes"] == [q4["FactKey"]]
    assert upd[q4["FactKey"]][0] == "validated" and fy["FactKey"] in upd[q4["FactKey"]][1]["supersededBy"]
    # a credible same-assurance primary that disagrees still blocks the promotion
    other = _row("INTEREST_INCOME", "2017", "annual", "current", 70_000_000, "2017-dup", datetime(2018, 4, 30, tzinfo=UTC), "audited")
    assert {f["FactKey"]: st for f, st, _ in derive.cross_validate([fy, c1, other])}.get(fy["FactKey"], "extracted") != "validated"


def test_unvalidated_counterpart_never_demotes():
    """MB 2017 interest income: FY2018 comparative 19,876,026 vs the quarantined 2017Q4 OCR misread 19,026."""
    bad = _row("INTEREST_INCOME", "2017", "ytd", "cumulative-current", 19_026, "2017Q4", datetime(2018, 1, 30, tzinfo=UTC),
               status="quarantined", sym="MBB")
    cmp18 = _row("INTEREST_INCOME", "2017", "annual", "comparative", 19_876_026, "2018", datetime(2019, 3, 30, tzinfo=UTC),
                 "audited", sym="MBB")
    st, vj = {f["FactKey"]: (st, vj) for f, st, vj in derive.cross_validate([bad, cmp18])}[cmp18["FactKey"]]
    assert st == "validated" and vj["crossDocument"]["disagreesWithUnvalidated"] == [bad["FactKey"]]


def test_restated_comparative_supersedes_the_earlier_annual():
    """BIDV FY2018 opex -16,116,922 vs FY2019 'Năm trước (Trình bày lại)' -16,016,084."""
    fy18 = _row("OPERATING_EXPENSES", "2018", "annual", "current", -16_116_922, "2018", datetime(2019, 3, 30, tzinfo=UTC), "audited")
    cmp19 = _row("OPERATING_EXPENSES", "2018", "annual", "comparative", -16_016_084, "2019", datetime(2020, 3, 30, tzinfo=UTC),
                 "audited", restated=True)
    upd = {f["FactKey"]: (st, vj["crossDocument"]) for f, st, vj in derive.cross_validate([fy18, cmp19])}
    assert upd[fy18["FactKey"]][0] == "validated"
    assert upd[fy18["FactKey"]][1]["supersededBy"] == [cmp19["FactKey"]]
    assert upd[fy18["FactKey"]][1]["supersessionReason"] == "restatement"
    assert upd[cmp19["FactKey"]][0] == "validated" and upd[cmp19["FactKey"]][1]["supersedes"] == [fy18["FactKey"]]


def test_mb_group1_definition_follows_the_identity_layout():
    def g1(treat):
        checks = [{"id": "NOTE_LOAN_GROUPS_EQ_TABLE_TOTAL", "outcome": "pass", **({"marginTreatment": treat} if treat else {})}]
        return {"metric": "LOAN_GROUP1", "metricDef": {}, "checks": checks}
    sep, inner, none = g1("separate"), g1("inGroup1"), g1(None)
    _assign_group1_definition([sep, inner, none], "MBB")
    assert sep["metric"] == "LOAN_GROUP1" and "excl." in sep["definition"]
    assert inner["metric"] == "LOAN_GROUP1_INCL_MARGIN" and "of-which" in inner["definition"]
    assert none["metric"] == "LOAN_GROUP1_INCL_MARGIN"
    bid = g1(None)
    _assign_group1_definition([bid], "BID")
    assert bid["metric"] == "LOAN_GROUP1"  # BIDV has no margin line: one definition


def test_npl_ratio_uses_one_group1_definition():
    def pit(metric, v):
        return dict(_row(metric, "2022Q4", "point-in-time", "current", v, "2022Q4", datetime(2023, 1, 30, tzinfo=UTC), sym="MBB"),
                    MetricGroup="credit-quality", PeriodEnd=date(2022, 12, 31))
    facts = [pit("LOAN_GROUP1_INCL_MARGIN", 447_734_256), pit("NPL_GROUP2", 7_808_932), pit("NPL_GROUP3", 1_517_212),
             pit("NPL_GROUP4", 1_000_000), pit("NPL_GROUP5", 2_514_082), pit("MARGIN_LOANS_GROUP1", 3_759_500)]
    d = next(x for x in derive.plan_derivations(facts, {}) if x["MetricCode"] == "NPL_RATIO_CALC")
    assert "MARGIN_LOANS_GROUP1" not in d["DerivationJson"]["formula"]  # the margin line is already inside group 1
    assert d["Value"] == (Decimal(5_031_294) / Decimal(460_574_482)).quantize(derive.RATIO_PLACES)


def test_weak_second_pass_threshold():
    """BIDV 2025Q3 p7 'VII Thu nhập từ góp vốn, mua cổ phần' OCR'd as 'Thu nhận từ gúp vẫn mua cả nhẫn' (0.839)."""
    ms = [m for m in extractable(load_catalog(), "BID", "income_statement") if m["code"] == "EQUITY_INVESTMENT_INCOME"]
    rows = [{"page": 7, "index": 24, "y": 1762.0, "labelNorm": [norm_label("VI Thu nhận từ gúp vẫn mua cả nhẫn 18")],
             "cells": {"0": {"kind": "number"}}}]
    assert anchor_metrics(rows, ms, "BID") == {}
    assert "EQUITY_INVESTMENT_INCOME" in anchor_metrics(rows, ms, "BID", WEAK_THRESHOLD)


def test_weak_match_needs_a_complete_passing_identity():
    from equity_facts.extract import _confirmed_by_complete_identity

    f = {"metric": "INVESTMENT_SECURITIES_INCOME_NET", "periodType": "quarter", "periodLabel": "2021Q4", "role": "current",
         "cons": "consolidated", "status": "validated"}
    key = ["quarter", "2021Q4", "current", "consolidated"]
    absorbed = {"outcome": "pass", "key": key, "metrics": ["TOTAL_OPERATING_INCOME", f["metric"]],
                "missing": ["TRADING_SECURITIES_INCOME_NET"]}
    assert not _confirmed_by_complete_identity(f, [absorbed])
    assert _confirmed_by_complete_identity(f, [dict(absorbed, missing=[])])


def test_mangled_numbering_and_note_ref_on_the_trading_row():
    """BIDV FY2017 p10 'IW Lãi thuần từ mua bán chứng khoán kinh 3ũ.1 / doanh' (IV, note 30.1): without stripping,
    the subtotal 'Lãi thuần từ mua bán chứng khoán' (IV + V) outscored it."""
    assert norm_label("IW Lãi thuần tử mua hản chứng khoản kinh 3ũ.1") == "LAI THUAN TU MUA HAN CHUNG KHOAN KINH"
