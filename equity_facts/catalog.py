"""Metric catalog loading and row -> metric anchoring (pure)."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .textnorm import label_score, norm_label

CATALOG_PATH = Path(__file__).with_name("metric_catalog.json")
MATCH_THRESHOLD = 0.86
WEAK_THRESHOLD = 0.8  # second pass on left-over rows; kept only if an identity then validates the fact
RAPID_LABEL_PENALTY = 0.02  # RapidOCR drops Vietnamese diacritic letters: a small tie-breaker in favour of Tesseract


@lru_cache(maxsize=4)
def load_catalog(path: str | None = None) -> dict[str, Any]:
    data = json.loads(Path(path or CATALOG_PATH).read_text(encoding="utf-8"))
    for m in data["metrics"]:
        labels = m.get("labels", {})
        # own issuer's wording first, then the other issuer's (both use the SBV template; wording drifts by year)
        m["_labels_norm"] = {iss: list(dict.fromkeys(
            [norm_label(x) for x in labs] + [norm_label(x) for o, ol in labels.items() if o != iss for x in ol]))
            for iss, labs in labels.items()}
    return data


def metrics_by_code(cat: dict | None = None) -> dict[str, dict]:
    cat = cat or load_catalog()
    return {m["code"]: m for m in cat["metrics"]}


def extractable(cat: dict, issuer: str, page_kind: str) -> list[dict]:
    return [m for m in cat["metrics"] if m.get("pageKind") == page_kind and m["_labels_norm"].get(issuer)
            and not m.get("derived") and m.get("extracted", True)]


def row_score(row: dict, labels_norm: list[str]) -> tuple[float, str, str]:
    """Best similarity of a table row against label variants -> (score, matched catalog label, row text used)."""
    best = (0.0, "", "")
    texts = [(t, 0.0) for t in row.get("labelNorm", [])]
    rv = row.get("rapidVariants") or ([row["rapidLabel"]] if row.get("rapidLabel") else [])
    for t in rv:
        texts.append((norm_label(t), -RAPID_LABEL_PENALTY))
    for lab in labels_norm:
        for t, w in texts:
            s = label_score(lab, t) + w
            if s > best[0]:
                best = (s, lab, t)
    return best


def tag_bs_sections(rows: list[dict]) -> None:
    """Assets -> liabilities -> equity, moving forward only (rows already in page/y order)."""
    section = "assets"
    for r in rows:
        texts = r.get("labelNorm", [])[:1] + ([norm_label(r["rapidLabel"])] if r.get("rapidLabel") else [])
        if section == "assets" and any(re.search(r"PH\w{2,3}\s?TRA\b", t) for t in texts):  # not "phải thu"/"chi phí trả trước" (assets); OCR "PHẢÁI TRẢ"
            section = "liabilities"
        # "VỐN CHỦ SỞ HỮU" / "Vốn và các quỹ", also OCR'd "Vẫn và các quỹ" (BIDV 2016-19), "MÔN CHỦ SỞ HỮU" (MB 2020)
        elif section == "liabilities" and any(re.match(r"^\w{0,2}N\s?(CHU SO HUU|VA CAC QU)", t) for t in texts):
            section = "equity"
        r["section"] = section


def anchor_metrics(rows: list[dict], metrics: list[dict], issuer: str,
                   threshold: float = MATCH_THRESHOLD) -> dict[str, dict]:
    """Assign each metric to at most one numeric row; a row serves at most one metric.

    rows: numeric + heading rows of all pages of one page kind, in reading order, each with "page".
    Returns {code: {"row": row, "score": s, "label": matched, "text": used}}.
    """
    numeric = [r for r in rows if r.get("cells")]
    cands: dict[str, list[tuple[float, int, str, str]]] = {}
    for m in metrics:
        labs = m["_labels_norm"].get(issuer, [])
        section = m.get("section")
        excl = [x.upper() for x in m.get("exclude", [])]
        lst = []
        for i, r in enumerate(numeric):
            if section and r.get("section") and r["section"] != section:
                continue
            rapid_n = [norm_label(t) for t in (r.get("rapidVariants") or [r.get("rapidLabel") or ""])]
            if excl and any(x in t for x in excl for t in r.get("labelNorm", []) + rapid_n):
                continue
            if m.get("labelPriority"):
                for li, lab in enumerate(labs):
                    s, used_lab, used_text = row_score(r, [lab])
                    if s >= threshold:
                        lst.append((s - 0.1 * li, i, used_lab, used_text))
                        break
                continue
            s, used_lab, used_text = row_score(r, labs)
            if s >= threshold:
                lst.append((s, i, used_lab, used_text))
        cands[m["code"]] = lst

    by_code = {m["code"]: m for m in metrics}

    def pick(code: str, taken: set[int]) -> tuple[float, int, str, str] | None:
        lst = [c for c in cands[code] if c[1] not in taken]
        if not lst:
            return None
        occ = by_code[code].get("occurrence")
        if occ:
            best = max(c[0] for c in cands[code])  # same label printed several times: OCR noise costs <= ~0.05
            floor = 0.9 if occ > 0 else max(threshold, best - 0.06)  # k-th printed duplicate vs "last of the best"
            strong = sorted((c for c in cands[code] if c[0] >= floor), key=lambda c: c[1])
            if occ < 0:  # a later row won by a better-matching metric (investing CF line) is not "the last occurrence"
                strong = [c for c in strong if c[1] not in taken]
            if occ < 0 and len(strong) >= -occ and strong[occ][1] not in taken:
                return strong[occ]  # -1 = last occurrence (operating cash flow comes after the pre-working-capital line)
            if occ > 0 and len(strong) >= occ and strong[occ - 1][1] not in taken:
                return strong[occ - 1]
            return None
        return max(lst, key=lambda c: (round(c[0], 4), -c[1]))

    assigned: dict[str, tuple[float, int, str, str]] = {}
    taken: set[int] = set()
    pending = [m["code"] for m in metrics]
    for _ in range(len(pending) + 5):
        if not pending:
            break
        proposals = {code: pick(code, taken) for code in pending}
        proposals = {c: p for c, p in proposals.items() if p}
        if not proposals:
            break
        # resolve: per row keep the strongest claim
        by_row: dict[int, tuple[str, tuple]] = {}
        for code, p in proposals.items():
            cur = by_row.get(p[1])
            if cur is None or p[0] > cur[1][0]:
                by_row[p[1]] = (code, p)
        for row_i, (code, p) in by_row.items():
            assigned[code] = p
            taken.add(row_i)
        pending = [c for c in pending if c not in assigned and any(x[1] not in taken for x in cands[c])
                   and not (by_code[c].get("occurrence") or 0) > 0]  # "last occurrence" retries on untaken rows
    out = {}
    for code, (s, i, lab, txt) in assigned.items():
        out[code] = {"row": numeric[i], "score": round(s, 4), "label": lab, "text": txt}
    return out
