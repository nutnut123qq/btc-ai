"""equity_facts: cross-document validation + derived facts with lineage guards (V3-B)."""
import sys
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from equity_facts import derive


def row(metric, label, ptype, role, value, doc, doc_period, status="validated", cons="consolidated", scale=1_000_000,
        group="income-statement", end=None, rev=1, avail=datetime(2026, 1, 31, tzinfo=UTC)):
    key = f"BID/{doc}/eqx-facts-v1/{metric}/{label}/{ptype}/{role}"
    return {"Id": key, "FactKey": key, "MetricCode": metric, "MetricDefinitionVersion": "v1", "MetricGroup": group,
            "Statement": "income-statement", "PeriodLabel": label, "PeriodType": ptype, "PeriodStart": None,
            "PeriodEnd": end or date(int(label[:4]), 12, 31), "FiscalYear": int(label[:4]), "Consolidation": cons,
            "Value": Decimal(value), "Unit": "VND", "Scale": Decimal(scale), "ColumnRole": role, "DocumentId": doc,
            "Status": status, "ValidationJson": {"extractionStatus": status}, "IsDerived": False, "DerivationJson": None,
            "AvailableAtUtc": avail, "PageNumber": 7, "DocPeriod": doc_period, "DocRevision": rev, "DocType": "financial-statement",
            "DocSupersedes": None, "DocSha": doc}


def fy_and_9m(**kw):
    fy = row("PROFIT_BEFORE_TAX", "2025", "annual", "current", 37_787_518, "fy2025", "2025",
             avail=datetime(2026, 3, 31, tzinfo=UTC), **kw.get("fy", {}))
    nine = row("PROFIT_BEFORE_TAX", "2025-9M", "ytd", "cumulative-current", 27_000_000, "q3-2025", "2025Q3",
               end=date(2025, 9, 30), **kw.get("nine", {}))
    return [fy, nine]


def statuses(facts):
    return {f["FactKey"]: f["Status"] for f in facts}


def q4(derived):
    return [d for d in derived if d["PeriodLabel"] == "2025Q4" and d["MetricCode"] == "PROFIT_BEFORE_TAX"]


def test_q4_is_fy_minus_9m_with_lineage():
    facts = fy_and_9m()
    d = q4(derive.plan_derivations(facts, statuses(facts)))
    assert len(d) == 1
    d = d[0]
    assert d["Value"] == Decimal(10_787_518) and d["Scale"] == Decimal(1_000_000)
    assert d["IsDerived"] and d["ColumnRole"] == "derived" and d["ExtractionId"] is None and d["Statement"] == "derived"
    assert d["DerivationJson"]["inputs"] == sorted(f["FactKey"] for f in facts)
    assert d["AvailableAtUtc"] == datetime(2026, 3, 31, tzinfo=UTC)  # max of inputs
    assert d["PeriodStart"] == date(2025, 10, 1) and d["PeriodEnd"] == date(2025, 12, 31)
    assert d["Status"] == "validated"


def test_q4_fact_key_is_stable_across_runs():
    facts = fy_and_9m()
    a = q4(derive.plan_derivations(facts, statuses(facts)))[0]["FactKey"]
    b = q4(derive.plan_derivations(facts, statuses(facts)))[0]["FactKey"]
    assert a == b


def test_q4_lineage_guards():
    for kw in ({"nine": {"scale": 1}}, {"nine": {"cons": "separate"}}, {"nine": {"status": "review-needed"}},
               {"fy": {"status": "quarantined"}}, {"fy": {"group": "balance-sheet"}}):
        facts = fy_and_9m(**kw)
        assert q4(derive.plan_derivations(facts, statuses(facts))) == [], kw


def test_q4_uses_latest_revision_only_and_skips_conflicts():
    facts = fy_and_9m()
    newer = row("PROFIT_BEFORE_TAX", "2025", "annual", "current", 37_800_000, "fy2025-r2", "2025", rev=2)
    d = q4(derive.plan_derivations(facts + [newer], statuses(facts + [newer])))
    assert len(d) == 1 and d[0]["Value"] == Decimal(37_800_000 - 27_000_000)


def test_derived_q4_disagreeing_with_printed_quarter_needs_review():
    facts = fy_and_9m()
    printed = row("PROFIT_BEFORE_TAX", "2025Q4", "quarter", "current", 10_000_000, "q4-2025", "2025Q4")
    d = q4(derive.plan_derivations(facts + [printed], statuses(facts + [printed])))
    assert d[0]["Status"] == "review-needed" and d[0]["ValidationJson"]["printedQuarter"]["agrees"] is False


def test_cross_validation_promotes_agreeing_extracted_and_flags_comparative_mismatch():
    own = row("NET_INTEREST_INCOME", "2025Q2", "quarter", "current", 100, "q2-2025", "2025Q2")
    later_ok = row("NET_INTEREST_INCOME", "2025Q2", "quarter", "comparative", 100, "q2-2026", "2026Q2", status="extracted")
    later_bad = row("NET_INTEREST_INCOME", "2025Q2", "quarter", "comparative", 120, "q2-2026b", "2026Q2")
    upd = {f["FactKey"]: (st, vj) for f, st, vj in derive.cross_validate([own, later_ok, later_bad])}
    assert upd[later_ok["FactKey"]][0] == "validated"
    assert upd[later_bad["FactKey"]][0] == "review-needed"
    assert "restatementOrMismatch" in upd[later_bad["FactKey"]][1]["crossDocument"]
    assert upd[own["FactKey"]][0] == "validated"  # agrees with one later comparative -> not a restatement
    assert upd[own["FactKey"]][1]["crossDocument"]["disagreesWith"] == [later_bad["FactKey"]]


def test_unaudited_primary_differing_from_later_audited_primary_needs_review():
    q4 = row("TOTAL_EQUITY", "2025Q4", "point-in-time", "current", 167_988_959, "q4-2025", "2025Q4",
             avail=datetime(2026, 1, 31, tzinfo=UTC))
    fy = row("TOTAL_EQUITY", "2025Q4", "point-in-time", "current", 168_045_367, "fy-2025", "2025",
             avail=datetime(2026, 3, 31, tzinfo=UTC))
    upd = {f["FactKey"]: (st, vj) for f, st, vj in derive.cross_validate([q4, fy])}
    assert upd[q4["FactKey"]][0] == "review-needed"
    assert upd[q4["FactKey"]][1]["crossDocument"]["laterReportDiffers"] == [fy["FactKey"]]
    assert upd[fy["FactKey"]][0] == "validated"  # values are never overwritten, only flagged


def test_cross_validation_never_promotes_quarantined_and_is_idempotent():
    own = row("NET_INTEREST_INCOME", "2025Q2", "quarter", "current", 100, "a", "2025Q2")
    q = row("NET_INTEREST_INCOME", "2025Q2", "quarter", "comparative", 100, "b", "2026Q2", status="quarantined")
    upd = derive.cross_validate([own, q])
    st = {f["FactKey"]: s for f, s, _ in upd}
    assert st.get(q["FactKey"], "quarantined") == "quarantined"
    for f, s, vj in upd:  # apply, then a second run changes nothing
        f["Status"], f["ValidationJson"] = s, vj
    assert derive.cross_validate([own, q]) == []


def test_ratios_and_parent_equity_need_full_inputs():
    pit = dict(ptype="point-in-time", role="current", doc="q2", doc_period="2026Q2", group="credit-quality", end=date(2026, 6, 30))

    def r(m, v, **kw):
        return row(m, "2026Q2", pit["ptype"], pit["role"], v, pit["doc"], pit["doc_period"], group=pit["group"], end=pit["end"], **kw)

    groups = [r("LOAN_GROUP1", 900), r("NPL_GROUP2", 50), r("NPL_GROUP3", 20), r("NPL_GROUP4", 10), r("NPL_GROUP5", 20)]
    d = derive.plan_derivations(groups, statuses(groups))
    npl = [x for x in d if x["MetricCode"] == "NPL_RATIO_CALC"]
    assert len(npl) == 1 and npl[0]["Value"] == Decimal("0.05") and npl[0]["Unit"] == "ratio" and npl[0]["Scale"] == 1
    assert npl[0]["DerivationJson"]["formula"].startswith("(NPL_GROUP3")
    assert not [x for x in derive.plan_derivations(groups[:4], statuses(groups[:4])) if x["MetricCode"] == "NPL_RATIO_CALC"]
    eq = [r("TOTAL_EQUITY", 198_778_010), r("MINORITY_INTEREST", 5_728_181)]
    par = [x for x in derive.plan_derivations(eq, statuses(eq)) if x["MetricCode"] == "EQUITY_ATTRIBUTABLE_PARENT"]
    assert par[0]["Value"] == Decimal(193_049_829)


def test_existing_derived_fact_follows_input_status():
    facts = fy_and_9m()
    d = q4(derive.plan_derivations(facts, statuses(facts)))[0]
    d["Symbol"] = "BID"
    st = statuses(facts)
    st[facts[1]["FactKey"]] = "review-needed"
    upd = derive.recheck_existing_derived(facts + [d], st)
    assert [(u[0]["FactKey"], u[1]) for u in upd] == [(d["FactKey"], "review-needed")]
