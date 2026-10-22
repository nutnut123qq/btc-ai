"""Per-document accounting validation (fail closed, pure).

Every candidate fact carries ``period key`` (periodLabel, periodType, role). Identities are evaluated per key
(balance sheet and notes are point-in-time; their key ignores the page). A value printed as "-" counts as 0
in an identity but never becomes a fact. Outcome per fact:
- in a failing identity and no passing one  -> quarantined
- in a failing identity but also a passing one, or only weakly inconsistent (rounding-size diff) -> review-needed
- in at least one passing identity -> validated
- covered by no evaluable identity -> extracted (derive_and_cross_validate may confirm it across documents)
"""
from __future__ import annotations

from typing import Any

ASSET_PARTS = ["CASH_AND_GOLD", "DEPOSITS_AT_SBV", "INTERBANK_PLACEMENTS_NET", "TRADING_SECURITIES_NET", "DERIVATIVE_ASSETS",
               "CUSTOMER_LOANS_NET", "DEBT_PURCHASE_NET", "INVESTMENT_SECURITIES_NET", "LONG_TERM_INVESTMENTS_NET",
               "FIXED_ASSETS", "INVESTMENT_PROPERTY", "OTHER_ASSETS"]
LIAB_PARTS = ["DUE_TO_GOV_AND_SBV", "INTERBANK_LIABILITIES", "CUSTOMER_DEPOSITS", "DERIVATIVE_LIABILITIES", "ENTRUSTED_FUNDS",
              "VALUABLE_PAPERS_ISSUED", "OTHER_LIABILITIES"]
INCOME_PARTS = ["NET_INTEREST_INCOME", "FEE_INCOME", "FX_INCOME_NET", "TRADING_SECURITIES_INCOME_NET",
                "INVESTMENT_SECURITIES_INCOME_NET", "OTHER_INCOME_NET", "EQUITY_INVESTMENT_INCOME"]
LOAN_GROUPS = ["LOAN_GROUP1", "NPL_GROUP2", "NPL_GROUP3", "NPL_GROUP4", "NPL_GROUP5"]

# (id, lhs, rhs terms, mode) ; mode strict = every term must be readable; sum = missing rhs rows tolerated if it adds up
IDENTITIES: list[tuple[str, str, list[str], str]] = [
    ("BS_TOTAL_ASSETS_EQ_LIAB_EQUITY", "TOTAL_ASSETS", ["TOTAL_LIABILITIES_AND_EQUITY"], "strict"),
    ("BS_ASSETS_SUM", "TOTAL_ASSETS", ASSET_PARTS, "sum"),
    ("BS_LIAB_EQUITY_SUM", "TOTAL_LIABILITIES_AND_EQUITY", LIAB_PARTS + ["TOTAL_EQUITY"], "sum"),
    ("BS_LIABILITIES_SUM", "TOTAL_LIABILITIES", LIAB_PARTS, "sum"),
    ("BS_LIABILITIES_PLUS_EQUITY", "TOTAL_LIABILITIES_AND_EQUITY", ["TOTAL_LIABILITIES", "TOTAL_EQUITY"], "strict"),
    ("BS_BANK_CAPITAL_SUM", "BANK_CAPITAL", ["CHARTER_CAPITAL", "SHARE_PREMIUM", "OTHER_CAPITAL", "TREASURY_SHARES"], "sum"),
    ("BS_EQUITY_SUM", "TOTAL_EQUITY", ["BANK_CAPITAL", "BANK_FUNDS", "FX_TRANSLATION_DIFF", "REVALUATION_DIFF",
                                       "RETAINED_EARNINGS", "MINORITY_INTEREST"], "sum"),
    ("BS_LOANS_NET", "CUSTOMER_LOANS_NET", ["CUSTOMER_LOANS_GROSS", "LOAN_LOSS_PROVISION_BALANCE"], "strict"),
    ("IS_NET_INTEREST_INCOME", "NET_INTEREST_INCOME", ["INTEREST_INCOME", "INTEREST_EXPENSE"], "strict"),
    ("IS_NET_FEE_INCOME", "FEE_INCOME", ["FEE_INCOME_GROSS", "FEE_EXPENSE"], "strict"),
    ("IS_TOTAL_OPERATING_INCOME", "TOTAL_OPERATING_INCOME", INCOME_PARTS, "sum"),
    ("IS_PRE_PROVISION_FROM_TOI", "PRE_PROVISION_PROFIT", ["TOTAL_OPERATING_INCOME", "OPERATING_EXPENSES"], "strict"),
    ("IS_PRE_PROVISION_FROM_PARTS", "PRE_PROVISION_PROFIT", INCOME_PARTS + ["OPERATING_EXPENSES"], "sum"),
    ("IS_PROFIT_BEFORE_TAX", "PROFIT_BEFORE_TAX", ["PRE_PROVISION_PROFIT", "CREDIT_PROVISION_EXPENSE"], "strict"),
    ("IS_TAX_EXPENSE", "INCOME_TAX_EXPENSE", ["CIT_CURRENT", "CIT_DEFERRED"], "sum"),
    ("IS_PROFIT_AFTER_TAX", "PROFIT_AFTER_TAX", ["PROFIT_BEFORE_TAX", "INCOME_TAX_EXPENSE"], "strict"),
    ("CF_NET_CHANGE", "NET_CHANGE_IN_CASH", ["NET_CASH_FROM_OPERATING", "NET_CASH_FROM_INVESTING", "NET_CASH_FROM_FINANCING"], "strict"),
    ("NOTE_LOAN_GROUPS_EQ_GROSS_LOANS", "CUSTOMER_LOANS_GROSS", LOAN_GROUPS + ["MARGIN_LOANS_GROUP1"], "groups"),
    ("NOTE_LOAN_GROUPS_EQ_TABLE_TOTAL", "LOAN_QUALITY_TOTAL", LOAN_GROUPS + ["MARGIN_LOANS_GROUP1"], "groups"),
]


def _key(f: dict) -> tuple:
    cons = f.get("cons")
    if f["periodType"] == "point-in-time":
        return ("pit", f["periodLabel"], f["end"], cons)
    return (f["periodType"], f["periodLabel"], f["role"], cons)


def _tolerance(n_terms: int) -> int:
    return 1 if n_terms <= 2 else min(3, n_terms - 1)


def evaluate(facts: list[dict], dashes: list[dict], unreadable: list[dict]) -> list[dict]:
    """Evaluate identities; mutate facts with status + checks. Returns the list of identity results.

    facts: candidate facts (metric, value, periodLabel, periodType, role, end, ...).
    dashes: same keys for cells printed as "-" (value 0 in identities, never facts).
    unreadable: same keys for malformed/ambiguous cells (identity -> incomplete).
    """
    vals: dict[tuple, dict[str, Any]] = {}
    for f in facts:
        vals.setdefault(_key(f), {})[f["metric"]] = f
    for d in dashes:
        vals.setdefault(_key(d), {}).setdefault(d["metric"], "dash")
    for u in unreadable:
        vals.setdefault(_key(u), {}).setdefault(u["metric"], "unreadable")

    results = []
    for key, mv in vals.items():
        for ident, lhs, rhs, mode in IDENTITIES:
            if lhs not in mv:
                continue
            if not any(t in mv for t in rhs):
                continue
            res = _check(ident, lhs, rhs, mode, mv)
            if res is None:
                continue
            res["key"] = list(key)
            results.append(res)
        par = _check_parent(mv)
        if par:
            par["key"] = list(key)
            results.append(par)

    for f in facts:
        f.setdefault("checks", [])
    by_fact: dict[int, list[dict]] = {}
    for r in results:
        for m in r["metrics"]:
            obj = vals[tuple(r["key"])].get(m)
            if isinstance(obj, dict):
                by_fact.setdefault(id(obj), []).append(r)
                obj["checks"].append({k: r[k] for k in ("id", "outcome", "diff", "terms")}
                                     | ({"marginTreatment": r["marginTreatment"]} if r.get("marginTreatment") else {}))
        if r.get("marginTreatment") == "inGroup1" and r["outcome"] == "pass":
            m = vals[tuple(r["key"])].get("MARGIN_LOANS_GROUP1")
            if isinstance(m, dict):
                m["marginInGroup1"] = True  # an of-which line inside group 1, not the separate line this metric means: dropped
    for f in facts:
        outs = [r["outcome"] for r in by_fact.get(id(f), [])]
        if f.get("forceStatus"):
            f["status"] = f["forceStatus"]
        elif "fail" in outs:
            f["status"] = "review-needed" if "pass" in outs else "quarantined"
        elif "weak" in outs:
            f["status"] = "validated" if "pass" in outs else "review-needed"
        elif "pass" in outs:
            f["status"] = "validated"
        else:
            f["status"] = "extracted"
    return results


def _num(v) -> int | None:
    if v == "dash":
        return 0
    if isinstance(v, dict):
        return v["value"]
    return None


def _check(ident: str, lhs: str, rhs: list[str], mode: str, mv: dict) -> dict | None:
    left = _num(mv.get(lhs))
    if left is None:
        return {"id": ident, "outcome": "incomplete", "diff": None, "terms": {lhs: mv.get(lhs)}, "metrics": []}
    present = {t: _num(mv[t]) for t in rhs if t in mv}
    unreadable = [t for t, v in present.items() if v is None]
    missing = [t for t in rhs if t not in mv]
    terms = {lhs: left, **{t: v for t, v in present.items()}}
    treatment = None
    if mode == "groups":  # the five debt groups are mandatory, the separately printed margin line is optional
        missing = [t for t in missing if t != "MARGIN_LOANS_GROUP1"]
        mode = "strict"
        if "MARGIN_LOANS_GROUP1" in unreadable and len(unreadable) == 1:
            # the optional line is unreadable: groups 1-5 alone may still close (of-which layout); otherwise no verdict
            unreadable, treatment = [], "margin-unreadable"
            present.pop("MARGIN_LOANS_GROUP1")
            if left != sum(present.values()):
                return {"id": ident, "outcome": "incomplete", "diff": None, "terms": terms, "missing": missing,
                        "unreadable": ["MARGIN_LOANS_GROUP1"], "metrics": [], "marginTreatment": treatment}
        if present.get("MARGIN_LOANS_GROUP1") is not None and not missing and not unreadable:
            treatment = "separate"
            groups_only = sum(v for t, v in present.items() if t != "MARGIN_LOANS_GROUP1")
            if left == groups_only and left != groups_only + present["MARGIN_LOANS_GROUP1"]:
                # "+ Trong đó … cho vay tại MBS": an of-which line already inside group 1 (exact arithmetic decides)
                treatment = "inGroup1"
                present.pop("MARGIN_LOANS_GROUP1")
    if unreadable or (mode == "strict" and missing):
        return {"id": ident, "outcome": "incomplete", "diff": None, "terms": terms,
                "missing": missing, "unreadable": unreadable, "metrics": []}
    total = sum(v for v in present.values() if v is not None)
    diff = left - total
    n = 1 + len(present)
    if diff == 0:
        outcome = "pass"
    elif missing:
        outcome = "incomplete"
    elif abs(diff) <= _tolerance(n):
        outcome = "weak"
    else:
        outcome = "fail"
    metrics = [] if outcome == "incomplete" else [lhs, *present.keys()]
    res = {"id": ident, "outcome": outcome, "diff": diff, "terms": terms, "missing": missing, "metrics": metrics}
    if treatment:
        res["marginTreatment"] = treatment
    return res


def _check_parent(mv: dict) -> dict | None:
    """PAT_parent = PAT - minority share (MB prints the share positive) or PAT + minority (BIDV prints it negative)."""
    if "PROFIT_AFTER_TAX_PARENT" not in mv or "PROFIT_AFTER_TAX" not in mv:
        return None
    parent, pat = _num(mv["PROFIT_AFTER_TAX_PARENT"]), _num(mv["PROFIT_AFTER_TAX"])
    if "MINORITY_INTEREST_PROFIT" not in mv and parent is not None and pat is not None and parent != pat:
        # minority row not found: an absent row is not a zero -> no verdict
        return {"id": "IS_PAT_PARENT_PLUS_MINORITY", "outcome": "incomplete", "diff": None,
                "terms": {"PROFIT_AFTER_TAX_PARENT": parent, "PROFIT_AFTER_TAX": pat}, "missing": ["MINORITY_INTEREST_PROFIT"],
                "metrics": []}
    mi = _num(mv["MINORITY_INTEREST_PROFIT"]) if "MINORITY_INTEREST_PROFIT" in mv else 0
    terms = {"PROFIT_AFTER_TAX_PARENT": parent, "PROFIT_AFTER_TAX": pat, "MINORITY_INTEREST_PROFIT": mi}
    if parent is None or pat is None or mi is None:
        return {"id": "IS_PAT_PARENT_PLUS_MINORITY", "outcome": "incomplete", "diff": None, "terms": terms, "metrics": []}
    diffs = [parent - (pat - mi), parent - (pat + mi)]
    diff = min(diffs, key=abs)
    outcome = "pass" if diff == 0 else ("weak" if abs(diff) <= 1 else "fail")
    metrics = ["PROFIT_AFTER_TAX_PARENT", "PROFIT_AFTER_TAX"] + (["MINORITY_INTEREST_PROFIT"] if "MINORITY_INTEREST_PROFIT" in mv else [])
    return {"id": "IS_PAT_PARENT_PLUS_MINORITY", "outcome": outcome, "diff": diff, "terms": terms, "metrics": metrics}
