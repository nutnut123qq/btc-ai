"""equity_facts: accounting identities + fail-closed statuses (V3-B)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from equity_facts.validate import evaluate

Q = {"periodLabel": "2026Q2", "periodType": "quarter", "role": "current", "end": "2026-06-30"}
BS = {"periodLabel": "2026Q2", "periodType": "point-in-time", "role": "current", "end": "2026-06-30"}


def fact(metric, value, period=Q):
    return {"metric": metric, "value": value, **period}


def is_page(nii=16_893_654):
    """MBB Q2/2026 income statement, quarter column (ground truth b0-samples)."""
    return [fact("INTEREST_INCOME", 34_006_838), fact("INTEREST_EXPENSE", -17_113_184), fact("NET_INTEREST_INCOME", nii),
            fact("FEE_INCOME_GROSS", 5_249_714), fact("FEE_EXPENSE", -3_444_039), fact("FEE_INCOME", 1_805_675),
            fact("PROFIT_BEFORE_TAX", 10_559_872), fact("INCOME_TAX_EXPENSE", -2_114_399), fact("PROFIT_AFTER_TAX", 8_445_473),
            fact("MINORITY_INTEREST_PROFIT", 216_411), fact("PROFIT_AFTER_TAX_PARENT", 8_229_062)]


def status(facts, metric):
    return next(f["status"] for f in facts if f["metric"] == metric)


def test_identities_pass_validates_participants():
    facts = is_page()
    res = evaluate(facts, [], [])
    assert {r["id"] for r in res if r["outcome"] == "pass"} >= {"IS_NET_INTEREST_INCOME", "IS_NET_FEE_INCOME",
                                                               "IS_PROFIT_AFTER_TAX", "IS_PAT_PARENT_PLUS_MINORITY"}
    assert all(f["status"] == "validated" for f in facts)
    nii = next(f for f in facts if f["metric"] == "NET_INTEREST_INCOME")
    assert nii["checks"][0]["id"] == "IS_NET_INTEREST_INCOME" and nii["checks"][0]["outcome"] == "pass"


def test_misread_digit_quarantines_identity_members():
    facts = is_page(nii=46_893_654)  # the Tesseract '1'->'4' failure mode seen in V3-B0
    evaluate(facts, [], [])
    for m in ("NET_INTEREST_INCOME", "INTEREST_INCOME", "INTEREST_EXPENSE"):
        assert status(facts, m) == "quarantined"
    assert status(facts, "PROFIT_AFTER_TAX") == "validated"  # unrelated identities still pass


def test_rounding_size_difference_is_review_needed():
    facts = is_page(nii=16_893_655)
    evaluate(facts, [], [])
    assert status(facts, "NET_INTEREST_INCOME") == "review-needed"


def test_fact_in_failing_and_passing_identity_is_review_needed():
    facts = [fact("PROFIT_BEFORE_TAX", 100), fact("INCOME_TAX_EXPENSE", -20), fact("PROFIT_AFTER_TAX", 80),
             fact("PROFIT_AFTER_TAX_PARENT", 70), fact("MINORITY_INTEREST_PROFIT", 5)]
    evaluate(facts, [], [])
    assert status(facts, "PROFIT_AFTER_TAX") == "review-needed"   # PAT identity passes, parent identity fails
    assert status(facts, "PROFIT_AFTER_TAX_PARENT") == "quarantined"


def test_parent_identity_accepts_both_minority_sign_conventions():
    bid = [fact("PROFIT_AFTER_TAX", 8_294_898), fact("MINORITY_INTEREST_PROFIT", -148_650), fact("PROFIT_AFTER_TAX_PARENT", 8_146_248)]
    evaluate(bid, [], [])
    assert all(f["status"] == "validated" for f in bid)


def test_dash_counts_as_zero_and_unreadable_blocks():
    facts = [fact("INCOME_TAX_EXPENSE", -2_038_020), fact("CIT_CURRENT", -2_038_020)]
    evaluate(facts, [{"metric": "CIT_DEFERRED", **Q}], [])
    assert all(f["status"] == "validated" for f in facts)
    facts = [fact("NET_INTEREST_INCOME", 10), fact("INTEREST_INCOME", 15)]
    res = evaluate(facts, [], [{"metric": "INTEREST_EXPENSE", **Q}])
    assert [r["outcome"] for r in res if r["id"] == "IS_NET_INTEREST_INCOME"] == ["incomplete"]
    assert all(f["status"] == "extracted" for f in facts)


def test_sum_identity_tolerates_absent_rows_only_when_it_adds_up():
    ok = [fact("TOTAL_ASSETS", 100, BS), fact("CASH_AND_GOLD", 30, BS), fact("CUSTOMER_LOANS_NET", 70, BS)]
    evaluate(ok, [], [])
    assert all(f["status"] == "validated" for f in ok)
    gap = [fact("TOTAL_ASSETS", 100, BS), fact("CASH_AND_GOLD", 30, BS), fact("CUSTOMER_LOANS_NET", 60, BS)]
    evaluate(gap, [], [])  # a component row may simply be missing -> incomplete, not a verdict
    assert all(f["status"] == "extracted" for f in gap)


def test_balance_sheet_and_loan_note_share_point_in_time_key():
    bs = [fact("CUSTOMER_LOANS_GROSS", 2_501_807_043, BS)]
    groups = [fact(m, v, BS) for m, v in [("LOAN_GROUP1", 2_425_506_836), ("NPL_GROUP2", 30_581_448), ("NPL_GROUP3", 12_003_167),
                                          ("NPL_GROUP4", 7_293_693), ("NPL_GROUP5", 26_421_899)]]
    facts = bs + groups
    evaluate(facts, [], [])
    assert all(f["status"] == "validated" for f in facts)
    groups[2]["value"] = 12_003_197
    for f in facts:
        f.pop("checks", None)
    evaluate(facts, [], [])
    assert status(facts, "NPL_GROUP3") == "quarantined"


def test_force_status_wins():
    facts = is_page()
    facts[0]["forceStatus"] = "quarantined"  # e.g. unit unknown / consolidation mismatch
    evaluate(facts, [], [])
    assert facts[0]["status"] == "quarantined"
