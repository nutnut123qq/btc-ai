"""derive_and_cross_validate: cross-document checks + derived facts (pure core + thin SQL wrapper).

Cross-validation (per metric, period, period type, consolidation, definition version; values compared in VND):
- an 'extracted' fact (no in-document identity) agreeing with a 'validated' fact of ANOTHER document -> validated
- a comparative / cumulative-comparative column disagreeing with the primary fact of the period
  (current / cumulative-current of the report for that period) -> review-needed (restatement flag), never overwritten
Statuses are recomputed from the extraction-time status on every run, so the step is deterministic and idempotent;
quarantined facts are never promoted.

Derived facts (IsDerived, Statement 'derived', ColumnRole 'derived', FactKey idempotent):
- Q4 = FY(annual, audited report) - 9M (ytd from the Q3 report); Q2 = 6M - 3M, Q3 = 9M - 6M (ytd columns)
  only for flows, inputs validated, identical consolidation / unit / scale / definition version, latest revision
- EQUITY_ATTRIBUTABLE_PARENT = TOTAL_EQUITY - MINORITY_INTEREST (same document column)
- NPL_RATIO_CALC, LOAN_TO_DEPOSIT_CALC (declared formulas in metric_catalog.json), full inputs only
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from .catalog import load_catalog, metrics_by_code
from .extract import EXTRACTOR_VERSION
from .periods import month_end

DERIVATION_VERSION = "eqx-derive-v4"
PRIMARY_ROLES = {"current", "cumulative-current"}
FLOW_GROUPS = {"income-statement", "cash-flow"}
RATIO_PLACES = Decimal("0.0000001")


def _d(v) -> date | None:
    if v is None or isinstance(v, date):
        return v
    return date.fromisoformat(str(v)[:10])


def _vnd(f: dict) -> Decimal:
    return Decimal(f["Value"]) * Decimal(f["Scale"])


def _period_class(f: dict) -> tuple[str, str]:
    """Same period, different report kinds: the Q4 report's 12-month column ("YYYY", ytd) is the audited annual's
    "YYYY"; the Q2 report's 6-month column ("YYYY-6M", ytd) is the reviewed half-year report's "YYYYH1"."""
    label = str(f["PeriodLabel"])
    if f["PeriodType"] == "ytd" and len(label) == 4:
        return label, "annual"
    if f["PeriodType"] == "ytd" and label.endswith("-6M"):
        return f"{label[:4]}H1", "half-year"
    return label, f["PeriodType"]


def _base_status(f: dict) -> str:
    vj = f.get("ValidationJson") or {}
    return vj.get("extractionStatus") or f["Status"]


ASSURANCE = {"audited": 2, "reviewed": 1}  # EquitySourceDocuments.AuditStatus; anything else = unaudited (0)


def _restated(f: dict) -> bool:
    return bool(((f.get("ValidationJson") or {}).get("column") or {}).get("restated"))


def _assurance(f: dict) -> int:
    return ASSURANCE.get(str(f.get("DocAuditStatus") or ""), 0)


def _supersedes(new: dict, old: dict) -> bool:
    """`new` is a later report that legitimately revises `old`: higher assurance, or a column declared restated
    (restatement document, or a later report's comparative headed "Trình bày lại")."""
    if new["AvailableAtUtc"] <= old["AvailableAtUtc"]:
        return False
    return _restated(new) or _assurance(new) > _assurance(old)


def _reason(fs: list[dict]) -> str:
    return "restatement" if any(_restated(f) for f in fs) else "audit-adjustment"


def cross_validate(facts: list[dict]) -> list[tuple[dict, str, dict]]:
    """Return [(fact, new_status, new_validation_json)] for facts whose status/annotation must change.

    Only a credible counterpart (validated by its own document's identities, or confirmed below) can demote or be
    superseded; an unvalidated / quarantined reading never demotes anything (MB 2017 interest income 19,876,026 vs the
    quarantined OCR misread 19,026)."""
    groups: dict[tuple, list[dict]] = {}
    for f in facts:
        if f["IsDerived"]:
            continue
        groups.setdefault((f.get("Symbol"), f["MetricCode"], *_period_class(f), f["Consolidation"],
                           f["MetricDefinitionVersion"]), []).append(f)

    # pass 1: an 'extracted' fact (no in-document identity could run) agreeing with a validated fact of another
    # report is confirmed, unless a credible primary report disagrees that it does not itself supersede
    # (BIDV FY2017 audited interest income 78,628,515 = two later reports; it supersedes the unaudited 2017Q4 12M)
    promoted: dict[str, list[str]] = {}
    for fs in groups.values():
        for f in fs:
            if _base_status(f) != "extracted":
                continue
            others = [o for o in fs if o["DocumentId"] != f["DocumentId"]]
            agree = [o for o in others if _vnd(o) == _vnd(f) and _base_status(o) == "validated"]
            block = [o for o in others if _vnd(o) != _vnd(f) and _base_status(o) == "validated"
                     and o["ColumnRole"] in PRIMARY_ROLES and not _supersedes(f, o)]
            if agree and not block:
                promoted[f["FactKey"]] = sorted(o["FactKey"] for o in agree)

    def credible(o: dict) -> bool:
        return _base_status(o) == "validated" or o["FactKey"] in promoted

    updates = []
    for key, fs in groups.items():
        docs = {f["DocumentId"] for f in fs}
        for f in fs:
            base = _base_status(f)
            vj = dict(f.get("ValidationJson") or {})
            vj.setdefault("extractionStatus", base)
            cross: dict[str, Any] = {"version": DERIVATION_VERSION}
            status = base
            others = [o for o in fs if o["DocumentId"] != f["DocumentId"]]
            if len(docs) > 1 and others:
                agree = [o for o in others if _vnd(o) == _vnd(f)]
                disagree_all = [o for o in others if _vnd(o) != _vnd(f)]
                disagree = [o for o in disagree_all if credible(o)]
                cross["agreesWith"] = sorted(o["FactKey"] for o in agree)
                cross["disagreesWith"] = sorted(o["FactKey"] for o in disagree)
                if len(disagree) != len(disagree_all):
                    cross["disagreesWithUnvalidated"] = sorted(o["FactKey"] for o in disagree_all if not credible(o))
                if f["FactKey"] in promoted:
                    status = "validated"
                    cross["promoted"] = {"reason": "agrees with validated facts of other reports", "confirmedBy": promoted[f["FactKey"]]}
                if disagree:
                    primaries = [o for o in disagree if o["ColumnRole"] in PRIMARY_ROLES]
                    # e.g. a 2026 report's 31/12/2025 column equals the AUDITED annual figure but not the earlier
                    # unaudited Q4 report: agreeing with one primary report is consistent, not a mismatch
                    agrees_primary = any(o["ColumnRole"] in PRIMARY_ROLES for o in agree)
                    # supersession: a LATER report of HIGHER assurance (audited FY, reviewed H1) or a declared
                    # restatement revises the figure; the earlier fact stays validated as then-published
                    sup_by = [o for o in disagree if _supersedes(o, f)]
                    sups = [o for o in disagree if _supersedes(f, o)] if credible(f) else []
                    if sup_by and credible(f):
                        cross["supersededBy"] = sorted(o["FactKey"] for o in sup_by)
                        cross["supersessionReason"] = _reason(sup_by)
                    if sups:
                        cross["supersedes"] = sorted(o["FactKey"] for o in sups)
                        cross.setdefault("supersessionReason", _reason([f]))
                    if f["ColumnRole"] not in PRIMARY_ROLES and primaries and not agrees_primary and _restated(f):
                        # the column header itself says "(*) / Trình bày lại": a declared restatement, not a misread
                        cross["declaredRestatement"] = "comparative column is marked restated; original report differs"
                    elif f["ColumnRole"] not in PRIMARY_ROLES and primaries and not agrees_primary:
                        # an audited/reviewed report's comparative revising an unaudited report: supersession
                        if not all(_supersedes(f, o) for o in primaries):
                            if status != "quarantined":
                                status = "review-needed"
                            cross["restatementOrMismatch"] = "comparative column differs from the period's own report"
                    elif f["ColumnRole"] in PRIMARY_ROLES:
                        later = [o for o in primaries if o["AvailableAtUtc"] > f["AvailableAtUtc"]]
                        unexplained = [o for o in later if not _supersedes(o, f)]
                        if unexplained:  # same or lower assurance and no declared restatement: a real conflict
                            cross["laterReportDiffers"] = sorted(o["FactKey"] for o in unexplained)
                            if status != "quarantined":
                                status = "review-needed"
                        elif not later:
                            cross["earlierReportDiffers"] = sorted(o["FactKey"] for o in primaries)
            if len(docs) == 1:  # nothing to compare against: leave the row untouched
                cross = vj.get("crossDocument")
            if status != f["Status"] or vj.get("crossDocument") != cross:
                vj["crossDocument"] = cross
                updates.append((f, status, vj))
    return updates


def _latest_by_doc_identity(fs: list[dict]) -> list[dict]:
    """Keep facts of the newest revision per (doc period) — superseded revisions do not feed derivations."""
    best: dict[str, dict] = {}
    for f in fs:
        k = f["DocPeriod"]
        if k not in best or (f["DocRevision"] or 0) > (best[k]["DocRevision"] or 0):
            best[k] = f
    return list(best.values())


def _derived(code: str, mdef: dict, label: str, ptype: str, start: date | None, end: date, cons: str, value: Decimal,
             unit: str, scale: Decimal, inputs: list[dict], formula: str, status: str, notes: dict) -> dict:
    end, start = _d(end), _d(start)
    keys = sorted(i["FactKey"] for i in inputs)
    h = hashlib.sha1("|".join(keys).encode()).hexdigest()[:16]
    latest = max(inputs, key=lambda i: i["AvailableAtUtc"])
    return {"Id": str(uuid.uuid4()), "Symbol": latest.get("Symbol"), "FactKey": f"{latest.get('Symbol') or ''}/{code}/derived/{DERIVATION_VERSION}/{label}/{ptype}/{cons}/{h}",
            "MetricCode": code, "MetricDefinitionVersion": mdef.get("definitionVersion", "v1"), "MetricGroup": mdef["group"],
            "Statement": "derived", "PeriodLabel": label, "PeriodType": ptype, "PeriodStart": start, "PeriodEnd": end,
            "FiscalYear": end.year, "Consolidation": cons, "Value": value, "Unit": unit, "Scale": scale, "RawLabel": None,
            "RawValueText": None, "RawUnitText": None, "ColumnRole": "derived", "PageNumber": None, "Location": None,
            "DocumentId": latest["DocumentId"], "ExtractionId": None, "Status": status,
            "ValidationJson": {"derivationVersion": DERIVATION_VERSION, "inputsValidated": status != "review-needed", **notes},
            "IsDerived": True,
            "DerivationJson": {"formula": formula, "inputs": keys, "derivationVersion": DERIVATION_VERSION,
                               "extractorVersion": EXTRACTOR_VERSION},
            "AvailableAtUtc": latest["AvailableAtUtc"], "CreatedAtUtc": datetime.now(UTC)}


def plan_derivations(facts: list[dict], status_of: dict[str, str]) -> list[dict]:
    """Derived facts from the current (post cross-validation) statuses. Pure; returns candidate rows."""
    symbols = {f.get("Symbol") for f in facts}
    if len(symbols) > 1:  # never mix issuers
        return [d for sym in sorted(symbols, key=str) for d in plan_derivations([f for f in facts if f.get("Symbol") == sym], status_of)]
    cat = load_catalog()
    mdefs = metrics_by_code(cat)
    src = [f for f in facts if not f["IsDerived"] and status_of.get(f["FactKey"], f["Status"]) == "validated"]
    out: list[dict] = []

    # --- single quarters from cumulative/annual flows
    idx: dict[tuple, list[dict]] = {}
    for f in src:
        if f["MetricGroup"] not in FLOW_GROUPS or f["ColumnRole"] not in PRIMARY_ROLES:
            continue
        idx.setdefault((f["MetricCode"], f["Consolidation"], f["PeriodLabel"], f["PeriodType"]), []).append(f)

    def one(code, cons, label, ptype, want_doc: str | None) -> dict | None:
        fs = [f for f in idx.get((code, cons, label, ptype), []) if want_doc is None or f["DocPeriod"] == want_doc]
        fs = _latest_by_doc_identity(fs)
        vals = {_vnd(f) for f in fs}
        return fs[0] if len(fs) >= 1 and len(vals) == 1 else None

    codes_cons = {(c, cons) for (c, cons, _, _) in idx}
    years = {int(str(lbl)[:4]) for (_, _, lbl, _) in idx}
    for code, cons in sorted(codes_cons):
        mdef = mdefs.get(code)
        if not mdef:
            continue
        for y in sorted(years):
            specs = [  # (quarter n, later input (label, type, doc), earlier input)
                (4, (str(y), "annual", str(y)), (f"{y}-9M", "ytd", f"{y}Q3")),
                (3, (f"{y}-9M", "ytd", f"{y}Q3"), (f"{y}-6M", "ytd", f"{y}Q2")),
                (2, (f"{y}-6M", "ytd", f"{y}Q2"), (f"{y}Q1", "quarter", f"{y}Q1")),
            ]
            for n, (l1, t1, d1), (l0, t0, d0) in specs:
                a, b = one(code, cons, l1, t1, d1), one(code, cons, l0, t0, d0)
                if not a or not b:
                    continue
                if (a["Unit"], Decimal(a["Scale"]), a["MetricDefinitionVersion"]) != (b["Unit"], Decimal(b["Scale"]), b["MetricDefinitionVersion"]):
                    continue
                value = Decimal(a["Value"]) - Decimal(b["Value"])
                label = f"{y}Q{n}"
                printed = [f for f in facts if not f["IsDerived"] and f["MetricCode"] == code and f["Consolidation"] == cons
                           and f["PeriodLabel"] == label and f["PeriodType"] == "quarter" and f["ColumnRole"] == "current"]
                notes: dict[str, Any] = {}
                status = "validated"
                if printed:
                    same = [p for p in printed if Decimal(p["Value"]) * Decimal(p["Scale"]) == value * Decimal(a["Scale"])]
                    notes["printedQuarter"] = {"agrees": bool(same), "factKeys": sorted(p["FactKey"] for p in printed)}
                    if not same:
                        # Q4 = FY(audited) - 9M vs the earlier unaudited Q4 report: the audit adjustment supersedes it
                        sup = [p for p in printed if any(_supersedes(i, p) for i in (a, b))]
                        if len(sup) == len(printed):
                            notes["printedQuarter"]["supersedes"] = sorted(p["FactKey"] for p in sup)
                            notes["printedQuarter"]["supersessionReason"] = _reason([a, b])
                        else:
                            status = "review-needed"
                start, end = date(y, 3 * n - 2, 1), month_end(y, 3 * n)
                out.append(_derived(code, mdef, label, "quarter", start, end, cons, value, a["Unit"], Decimal(a["Scale"]), [a, b],
                                    f"{l1} - {l0}", status, notes))

    # --- same-column identities / ratios (point-in-time, same document)
    pit: dict[tuple, dict[str, dict]] = {}
    for f in src:
        if f["PeriodType"] != "point-in-time":
            continue
        pit.setdefault((f["DocumentId"], f["PeriodLabel"], f["PeriodEnd"], f["Consolidation"]), {})[f["MetricCode"]] = f
    for (doc_id, label, end, cons), mv in sorted(pit.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        if "TOTAL_EQUITY" in mv and "MINORITY_INTEREST" in mv:
            te, mi = mv["TOTAL_EQUITY"], mv["MINORITY_INTEREST"]
            if Decimal(te["Scale"]) == Decimal(mi["Scale"]):
                out.append(_derived("EQUITY_ATTRIBUTABLE_PARENT", mdefs["EQUITY_ATTRIBUTABLE_PARENT"], label, "point-in-time", None,
                                    end, cons, Decimal(te["Value"]) - Decimal(mi["Value"]), "VND", Decimal(te["Scale"]), [te, mi],
                                    "TOTAL_EQUITY - MINORITY_INTEREST", "validated", {}))
        # group 1 in one of its two definitions; the separate MB margin line is added only to the "excl." one
        g1 = "LOAN_GROUP1" if "LOAN_GROUP1" in mv else "LOAN_GROUP1_INCL_MARGIN"
        groups = [g1, "NPL_GROUP2", "NPL_GROUP3", "NPL_GROUP4", "NPL_GROUP5"]
        denom = groups + (["MARGIN_LOANS_GROUP1"] if g1 == "LOAN_GROUP1" and "MARGIN_LOANS_GROUP1" in mv else [])
        if all(g in mv for g in groups) and len({Decimal(mv[g]["Scale"]) for g in denom}) == 1:
            tot = sum(Decimal(mv[g]["Value"]) for g in denom)
            if tot > 0:
                npl = sum(Decimal(mv[g]["Value"]) for g in groups[2:])
                out.append(_derived("NPL_RATIO_CALC", mdefs["NPL_RATIO_CALC"], label, "point-in-time", None, end, cons,
                                    (npl / tot).quantize(RATIO_PLACES), "ratio", Decimal(1), [mv[g] for g in denom],
                                    "(NPL_GROUP3+NPL_GROUP4+NPL_GROUP5)/(" + "+".join(denom) + ")", "validated", {}))
        if "CUSTOMER_LOANS_GROSS" in mv and "CUSTOMER_DEPOSITS" in mv:
            ln, dep = mv["CUSTOMER_LOANS_GROSS"], mv["CUSTOMER_DEPOSITS"]
            if Decimal(dep["Value"]) > 0 and Decimal(ln["Scale"]) == Decimal(dep["Scale"]):
                out.append(_derived("LOAN_TO_DEPOSIT_CALC", mdefs["LOAN_TO_DEPOSIT_CALC"], label, "point-in-time", None, end, cons,
                                    (Decimal(ln["Value"]) / Decimal(dep["Value"])).quantize(RATIO_PLACES), "ratio", Decimal(1),
                                    [ln, dep], "CUSTOMER_LOANS_GROSS / CUSTOMER_DEPOSITS", "validated", {}))
    return out


def reconcile_with_derived(updates: list[tuple[dict, str, dict]], derived: list[dict]) -> list[tuple[dict, str, dict]]:
    """A later report's comparative QUARTER that differs from the unaudited quarterly report but equals the audited
    derivation (Q4 = audited FY - 9M) is the audit adjustment carried forward, not a mismatch (BID 2021Q4 report,
    2020Q4 column: 25,959,175 = 100,687,502 - 74,728,327)."""
    audited_q = {}
    for d in derived:
        pq = (d.get("ValidationJson") or {}).get("printedQuarter") or {}
        if d["Status"] == "validated" and pq.get("supersedes"):
            audited_q[(d["MetricCode"], d["PeriodLabel"], d["PeriodType"], d["Consolidation"])] = d
    out = []
    for f, st, vj in updates:
        cross = (vj.get("crossDocument") or {})
        d = audited_q.get((f["MetricCode"], f["PeriodLabel"], f["PeriodType"], f["Consolidation"]))
        if st == "review-needed" and cross.get("restatementOrMismatch") and d is not None and _vnd(d) == _vnd(f):
            cross = {k: v for k, v in cross.items() if k != "restatementOrMismatch"}
            cross.update({"agreesWithDerived": d["FactKey"], "supersessionReason": "audit-adjustment"})
            vj = {**vj, "crossDocument": cross}
            st = _base_status(f)
        out.append((f, st, vj))
    return out


def superseded_by_derived(derived: list[dict]) -> dict[str, tuple[list[str], str]]:
    """printed-quarter FactKey -> (derived FactKeys superseding it, reason)."""
    out: dict[str, tuple[list[str], str]] = {}
    for d in derived:
        pq = (d.get("ValidationJson") or {}).get("printedQuarter") or {}
        for k in pq.get("supersedes") or []:
            out[k] = (sorted({*out.get(k, ([], ""))[0], d["FactKey"]}), pq.get("supersessionReason") or "audit-adjustment")
    return out


def recheck_existing_derived(facts: list[dict], status_of: dict[str, str]) -> list[tuple[dict, str, dict]]:
    """Existing derived facts whose inputs are no longer all validated -> review-needed (and back)."""
    updates = []
    for f in facts:
        if not f["IsDerived"]:
            continue
        inputs = (f.get("DerivationJson") or {}).get("inputs") or []
        ok = all(status_of.get(k) == "validated" for k in inputs)
        vj = dict(f.get("ValidationJson") or {})
        pq = vj.get("printedQuarter") or {}
        printed_ok = pq.get("agrees", True) or bool(pq.get("supersedes"))
        want = "validated" if ok and printed_ok else "review-needed"
        if want != f["Status"] or vj.get("inputsValidated") != ok:
            vj["inputsValidated"] = ok
            updates.append((f, want, vj))
    return updates


def current_facts(facts: list[dict]) -> list[dict]:
    """Only the latest extraction of each document takes part (older extractor versions stay as history), and only
    derived facts of the current derivation version are re-checked."""
    latest: dict[str, tuple] = {}
    for f in facts:
        if f["IsDerived"] or not f.get("ExtractionId"):
            continue
        k = (f.get("ExtractionFinishedAtUtc") or datetime.min.replace(tzinfo=UTC), f.get("ExtractorVersion") or "")
        if f["DocumentId"] not in latest or k > latest[f["DocumentId"]][0]:
            latest[f["DocumentId"]] = (k, f["ExtractionId"])
    out = []
    for f in facts:
        if f["IsDerived"]:
            if (f.get("DerivationJson") or {}).get("derivationVersion") == DERIVATION_VERSION:
                out.append(f)
        elif not f.get("ExtractionId") or latest.get(f["DocumentId"], (None, f.get("ExtractionId")))[1] == f.get("ExtractionId"):
            out.append(f)
    return out


def run(repo, symbol: str, run_id: str | None = None) -> dict:
    all_facts = repo.facts_for_symbol(symbol)
    facts = current_facts(all_facts)
    updates = cross_validate(facts)
    status_of = {f["FactKey"]: f["Status"] for f in facts}
    for f, st, _ in updates:
        status_of[f["FactKey"]] = st
    updates = reconcile_with_derived(updates, plan_derivations(facts, status_of))
    for f, st, vj in updates:
        status_of[f["FactKey"]] = st
        repo.update_fact_status(f["Id"], st, vj)
    upd2 = recheck_existing_derived(facts, status_of)
    for f, st, vj in upd2:
        repo.update_fact_status(f["Id"], st, vj)
    existing_keys = {f["FactKey"] for f in all_facts}
    planned = plan_derivations(facts, status_of)
    new = [d for d in planned if d["FactKey"] not in existing_keys]
    # printed quarters superseded by a derived quarter: own key, because cross_validate rewrites crossDocument
    vj_of = {f["FactKey"]: vj for f, _, vj in updates}
    by_key = {f["FactKey"]: f for f in facts}
    marked = 0
    for k, (keys, reason) in superseded_by_derived(planned).items():
        p = by_key.get(k)
        if p is None:
            continue
        vj = dict(vj_of.get(k) or p.get("ValidationJson") or {})
        want = {"factKeys": keys, "reason": reason}
        if vj.get("supersededByDerived") != want:
            vj["supersededByDerived"] = want
            repo.update_fact_status(p["Id"], status_of.get(k, p["Status"]), vj)
            marked += 1
    for d in new:
        d["Symbol"] = symbol
    inserted = repo.insert_facts(new, on_conflict_skip=True)
    counts: dict[str, Any] = {"facts": len(facts), "crossUpdated": len(updates), "derivedRechecked": len(upd2),
                              "derivedInserted": inserted, "derivationVersion": DERIVATION_VERSION, "runId": run_id,
                              "printedSupersededByDerived": marked}
    counts["promoted"] = sum(1 for _, st, vj in updates if st == "validated" and (vj.get("crossDocument") or {}).get("promoted"))
    counts["reviewNeededByCrossCheck"] = sum(1 for _, st, _ in updates if st == "review-needed")
    return counts
