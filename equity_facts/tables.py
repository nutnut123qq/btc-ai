"""Table reconstruction from OCR boxes (pure, testable without OCR engines).

Inputs per page: Tesseract word boxes (labels/headings) and RapidOCR boxes (numeric cells), both in the
pixel frame of the same (deskewed) 300-dpi render. Output: value columns (x-clusters of right edges of
grouped numbers), header text per column, and rows (label variants + one cell per column), all with the raw
text so any fact can be re-checked.
"""
from __future__ import annotations

import collections
import re
import statistics
from typing import Any

from .textnorm import (
    GROUPED_ANYWHERE,
    NOTE_REF,
    STT,
    fold,
    is_numeric_token,
    norm_label,
    parse_amount,
)

YEAR = re.compile(r"^(?:19|20)\d{2}$")
DATE_OR_YEAR = re.compile(r"\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{4}|\b(?:19|20)\d{2}\b")

HEADER_KEYWORDS = ("NAM NAY", "NAM TRUOC", "KY NAY", "KY TRUOC", "LUY KE", "TU DAU NAM", "QUY", "TRONG KY",
                   "SO CUOI", "SO DAU", "CUOI NAM", "DAU NAM", "TRIEU", "THANG", "TRINH BAY LAI", "DA KIEM TOAN")


def _dash_glyph(tokens: list[dict], col: dict, halfw: float, y: float, mh: float) -> dict | None:
    """A flat, short, non-digit mark inside a value column on the row's line: the printed nil dash."""
    for t in tokens:
        w, h = t["x1"] - t["x0"], t["y1"] - t["y0"]
        xc = (t["x0"] + t["x1"]) / 2
        if (len(t["text"]) <= 2 and not any(ch.isalnum() and ch.isascii() for ch in t["text"])
                and 0 < h <= 0.45 * mh and 1.5 * h <= w <= 1.2 * mh
                and col["left"] - 0.25 * halfw <= xc <= col["right"] + 0.1 * halfw and abs(t["yc"] - y) < 0.6 * mh):
            return t
    return None


def _split_tokens(items: list[dict]) -> list[dict]:
    """Split multi-word boxes into word tokens with proportional x (line-level engines)."""
    out = []
    for it in items:
        text = str(it.get("text") or "")
        x0, y0, x1, y1 = (float(v) for v in it["box"])
        if not text.strip():
            continue
        n = max(len(text), 1)
        start = None
        for i, ch in enumerate(text + " "):
            if ch.isspace():
                if start is not None:
                    tok = text[start:i]
                    out.append({"text": tok, "x0": x0 + (x1 - x0) * start / n, "x1": x0 + (x1 - x0) * i / n,
                                "y0": y0, "y1": y1, "yc": (y0 + y1) / 2, "h": y1 - y0,
                                "conf": it.get("conf"), "line": it.get("line")})
                    start = None
            elif start is None:
                start = i
    return out


def _merge_numeric_fragments(cells: list[dict]) -> list[dict]:
    """Join adjacent numeric fragments on one visual line ("(1.452." + "742)") before parsing."""
    cells = sorted(cells, key=lambda c: (round(c["yc"] / 8), c["x0"]))
    out: list[dict] = []
    for c in cells:
        if out:
            m = out[-1]
            h = max(m["h"], c["h"], 10.0)
            if abs(m["yc"] - c["yc"]) < 0.5 * h and 0 <= c["x0"] - m["x1"] < 0.6 * h and not m["text"].endswith(")"):
                m["text"] += c["text"]
                m["x1"] = c["x1"]
                continue
        out.append(dict(c))
    return out


def _cluster_columns(cells: list[dict], width: float) -> list[dict]:
    xs = sorted(c["x1"] for c in cells if GROUPED_ANYWHERE.search(c["text"]))
    if not xs:
        return []
    clusters, cur = [], [xs[0]]
    for x in xs[1:]:
        if x - cur[-1] < 0.02 * width:
            cur.append(x)
        else:
            clusters.append(cur)
            cur = [x]
    clusters.append(cur)
    clusters = [c for c in clusters if len(c) >= 3]
    cols = []
    for c in clusters:
        right = statistics.median(c)
        members = [d for d in cells if abs(d["x1"] - right) < 0.03 * width and GROUPED_ANYWHERE.search(d["text"])]
        left = statistics.median([d["x0"] for d in members]) if members else right - 0.08 * width
        cols.append({"right": right, "left": left, "center": (left + right) / 2, "n": len(c)})
    return cols


def _cluster_lines(items: list[dict], tol: float) -> list[list[dict]]:
    items = sorted(items, key=lambda w: w["yc"])
    lines: list[list[dict]] = []
    for w in items:
        if lines and abs(w["yc"] - statistics.fmean(x["yc"] for x in lines[-1])) < tol:
            lines[-1].append(w)
        else:
            lines.append([w])
    for ln in lines:
        ln.sort(key=lambda w: w["x0"])
    return lines


def _segments(line: list[dict], gap: float) -> list[dict]:
    segs: list[dict] = []
    for w in line:
        if segs and w["x0"] - segs[-1]["x1"] <= gap:
            segs[-1]["text"] += " " + w["text"]
            segs[-1]["x1"] = max(segs[-1]["x1"], w["x1"])
        else:
            segs.append({"text": w["text"], "x0": w["x0"], "x1": w["x1"]})
    return segs


def build_table(words: list[dict], boxes: list[dict], width: float, height: float) -> dict[str, Any]:
    """Reconstruct the value table of one page. See module docstring."""
    tw = _split_tokens(words)
    rb = _split_tokens(boxes)
    rapid_cells = _merge_numeric_fragments([t for t in rb if is_numeric_token(t["text"]) and not YEAR.match(t["text"].strip())])
    rapid_words = [t for t in rb if not is_numeric_token(t["text"])]
    tess_nums = _merge_numeric_fragments([t for t in tw if is_numeric_token(t["text"]) and GROUPED_ANYWHERE.search(t["text"])])
    label_words = [t for t in tw if not (is_numeric_token(t["text"]) and GROUPED_ANYWHERE.search(t["text"]))]

    cols = _cluster_columns(rapid_cells, width)
    result: dict[str, Any] = {"width": width, "height": height, "columns": [], "rows": [], "headerBlocks": []}
    if not cols:
        return result
    spacing = min((b["center"] - a["center"] for a, b in zip(cols, cols[1:])), default=0.16 * width)
    halfw = max(0.45 * spacing, 0.03 * width)
    first_left = min(c["left"] for c in cols)
    label_limit = first_left - 0.005 * width

    def col_of(cell: dict) -> int | None:
        d = [abs(cell["x1"] - c["right"]) for c in cols]
        k = min(range(len(cols)), key=d.__getitem__)
        return k if d[k] < min(0.06 * width, 0.9 * halfw + 0.02 * width) else None

    value_cells = []
    for c in rapid_cells:
        k = col_of(c)
        if k is not None and c["x0"] > first_left - 0.12 * width:
            value_cells.append({**c, "col": k, "engine": "rapid"})
    tess_cells = []
    for c in tess_nums:
        k = col_of(c)
        if k is not None:
            tess_cells.append({**c, "col": k, "engine": "tesseract"})

    heights = [c["h"] for c in value_cells] or [30.0]
    mh = statistics.median(heights)
    num_rows = _cluster_lines(value_cells, 0.5 * mh)

    lw_heights = [w["h"] for w in label_words] or [mh]
    lmh = statistics.median(lw_heights)
    all_lines = _cluster_lines(label_words, 0.45 * lmh)

    # header blocks: lines with no value cells, carrying dates / period keywords over value columns.
    # RapidOCR reads digits better than Tesseract ("37/12/2025"), so its date/year tokens join the header pool.
    numeric_ys = sorted(statistics.fmean(c["yc"] for c in r) for r in num_rows)
    rapid_hdr = [t for t in rapid_words if DATE_OR_YEAR.search(t["text"])]
    header_lines = []
    for ln in _cluster_lines(label_words + rapid_hdr, 0.45 * lmh):
        y = statistics.fmean(w["yc"] for w in ln)
        if any(abs(y - ny) < 0.6 * mh for ny in numeric_ys):
            continue
        ws = [w for w in ln if w["x1"] > first_left - halfw]
        if ws and _is_header_text(" ".join(w["text"] for w in ws)):
            header_lines.append({"y": y, "words": ws})

    for i, c in enumerate(cols):
        result["columns"].append({"index": i, "left": round(c["left"], 1), "right": round(c["right"], 1),
                                  "center": round(c["center"], 1), "cells": c["n"]})

    # label lines (left of the value area)
    lab_lines = []
    for ln in all_lines:
        ws = [w for w in ln if w["x1"] <= label_limit + 0.01 * width]
        if not ws:
            continue
        text = " ".join(w["text"] for w in ws)
        if sum(ch.isalpha() for ch in norm_label(text)) < 2:  # speckle / rule fragments
            continue
        y = statistics.fmean(w["yc"] for w in ws)
        lab_lines.append({"y": y, "words": ws, "text": text, "used": False,
                          "lines": {w.get("line") for w in ws if w.get("line")},
                          "continuation": text.lstrip(" -+*_.'\"")[:1].islower()})

    rapid_lab = [w for w in rapid_words if w["x1"] <= label_limit + 0.01 * width]
    rap_lines = [{"y": statistics.fmean(w["yc"] for w in ln), "text": " ".join(w["text"] for w in ln)}
                 for ln in _cluster_lines(rapid_lab, 0.4 * mh)]
    num_ys = [statistics.fmean(c["yc"] for c in r) for r in num_rows]

    def rapid_variants(y: float) -> list[str]:
        own = [ln for ln in rap_lines if abs(ln["y"] - y) < 0.6 * mh]
        if not own:
            return []
        own_t = " ".join(ln["text"] for ln in own)
        lo, hi = min(ln["y"] for ln in own), max(ln["y"] for ln in own)
        free = [ln for ln in rap_lines if all(abs(ln["y"] - ny) >= 0.6 * mh for ny in num_ys)]
        prev = [ln for ln in free if 0 < lo - ln["y"] < 1.9 * max(mh, lmh)]
        nxt = [ln for ln in free if 0 < ln["y"] - hi < 1.9 * max(mh, lmh)]
        out = [own_t]
        if prev:
            out.append(f"{prev[-1]['text']} {own_t}")
        if nxt:
            out.append(f"{own_t} {nxt[0]['text']}")
        return out

    rows: list[dict] = []
    for r in num_rows:
        y = statistics.fmean(c["yc"] for c in r)
        rl = [w for w in rapid_lab if abs(w["yc"] - y) < 0.6 * mh]
        rapid_label = " ".join(w["text"] for ln in _cluster_lines(rl, 0.4 * mh) for w in ln)
        own = None
        # 1) Tesseract reads label + amounts as one text line: its line id links them even on a residual skew
        linked = []
        for c in r:  # per RapidOCR cell: the nearest overlapping Tesseract token (adjacent rows are ~1 cell height apart)
            near = [t for t in tess_cells if t.get("line") and t["x1"] > c["x0"] and t["x0"] < c["x1"]
                    and abs(t["yc"] - c["yc"]) < 0.9 * mh]
            if near:
                linked.append(min(near, key=lambda t: abs(t["yc"] - c["yc"]))["line"])
        top = collections.Counter(linked).most_common(2)
        if top and (len(top) == 1 or top[0][1] > top[1][1]):  # a tied vote is no linkage: fall back to geometry
            lid = top[0][0]
            same =[ln for ln in lab_lines if lid in ln["lines"] and abs(ln["y"] - y) < 1.6 * max(mh, lmh)]
            if same:
                own = min(same, key=lambda ln: abs(ln["y"] - y))
        # 2) geometry: nearest line, preferring unused lines that start a new item (not a lowercase continuation)
        if own is None:
            cand = [ln for ln in lab_lines if abs(ln["y"] - y) < 0.8 * max(mh, lmh)]
            if cand:
                own = min(cand, key=lambda ln: abs(ln["y"] - y) + (0.5 * lmh if ln["continuation"] else 0)
                          + (0.4 * lmh if ln["used"] else 0))
        if own is not None:
            own["used"] = True
        cells: dict[int, dict] = {}
        for k in range(len(cols)):
            in_col = [c for c in r if c["col"] == k]
            tess = [c for c in tess_cells if c["col"] == k and abs(c["yc"] - y) < 0.6 * mh]
            tess_raw = min(tess, key=lambda c: abs(c["yc"] - y))["text"] if tess else None
            if len(in_col) > 1:
                vals = {parse_amount(c["text"])["value"] for c in in_col}
                if len(vals) > 1:
                    cells[k] = {"raw": " | ".join(c["text"] for c in in_col), "kind": "ambiguous", "value": None,
                                "engine": "rapid", "tessRaw": tess_raw, "needsReread": True,
                                "box": [round(min(c["x0"] for c in in_col)), round(min(c["y0"] for c in in_col)),
                                        round(max(c["x1"] for c in in_col)), round(max(c["y1"] for c in in_col))]}
                    continue
                in_col = in_col[:1]
            if in_col:
                c = in_col[0]
                p = parse_amount(c["text"])
                tp = parse_amount(tess_raw) if tess_raw else {"kind": None, "value": None}
                box = [round(c["x0"]), round(c["y0"]), round(c["x1"]), round(c["y1"])]
                if p["kind"] == "malformed":
                    # RapidOCR lost a separator ("776.657846", "7'5376"): the extractor re-reads the cell crop at other
                    # resolutions; a value is accepted only when two independent readings agree (never guessed)
                    cells[k] = {"raw": c["text"], "kind": "malformed", "value": None, "engine": "rapid",
                                "tessRaw": tess_raw, "needsReread": True, "box": box}
                    continue
                cells[k] = {"raw": c["text"], "kind": p["kind"], "value": p["value"], "engine": "rapid",
                            "conf": c.get("conf"), "tessRaw": tess_raw,
                            "engineAgree": (tp["value"] == p["value"]) if (tess_raw and p["kind"] == "number") else None,
                            "box": box}
            elif tess:
                c = min(tess, key=lambda c: abs(c["yc"] - y))
                p = parse_amount(c["text"])
                cells[k] = {"raw": c["text"], "kind": p["kind"], "value": p["value"], "engine": "tesseract",
                            "conf": c.get("conf"), "tessRaw": c["text"], "engineAgree": None,
                            "box": [round(c["x0"]), round(c["y0"]), round(c["x1"]), round(c["y1"])]}
            else:
                g = _dash_glyph(tw, cols[k], halfw, y, mh)
                if g is not None:  # a printed "-" (nil) too small for RapidOCR; Tesseract reads it as “ / š / ~
                    cells[k] = {"raw": "-", "kind": "dash", "value": None, "engine": "glyph", "tessRaw": g["text"],
                                "box": [round(g["x0"]), round(g["y0"]), round(g["x1"]), round(g["y1"])]}
        rows.append({"y": round(y, 1), "own": own, "cells": cells, "rapidLabel": rapid_label,
                     "rapidVariants": rapid_variants(y)})

    # label-only lines become heading rows (also used as continuation candidates)
    for ln in lab_lines:
        if not ln["used"]:
            rows.append({"y": round(ln["y"], 1), "own": ln, "cells": {}, "labelOnly": True})
    rows.sort(key=lambda r: r["y"])

    out_rows = []
    for i, r in enumerate(rows):
        own_text = r["own"]["text"] if r["own"] else ""
        variants: list[tuple[str, ...]] = [(own_text,)] if own_text else []
        if r["cells"]:
            prev = rows[i - 1] if i > 0 else None
            nxt = rows[i + 1] if i + 1 < len(rows) else None
            near = 1.9 * max(mh, lmh)
            p_txt = prev["own"]["text"] if prev and prev.get("labelOnly") and r["y"] - prev["y"] < near else None
            n_txt = nxt["own"]["text"] if nxt and nxt.get("labelOnly") and nxt["y"] - r["y"] < near else None
            if p_txt:
                variants.append((p_txt, own_text))
            if n_txt:
                variants.append((own_text, n_txt))
            if p_txt and n_txt:
                variants.append((p_txt, own_text, n_txt))
            prev2 = rows[i - 2] if i > 1 else None  # three-line labels with the amounts on the last line
            if p_txt and prev2 and prev2.get("labelOnly") and prev["y"] - prev2["y"] < near:
                variants.append((prev2["own"]["text"], p_txt, own_text))
        first = r["own"]["words"][0]["text"] if r["own"] else ""
        uniq = [v for v in dict.fromkeys(variants) if any(v)]
        out_rows.append({
            "index": len(out_rows), "y": r["y"], "label": own_text,
            "labelVariants": [" ".join(x for x in v if x) for v in uniq],
            # each line normalized on its own: a wrapped label's STT often sits on its 2nd line ("... cho vay các / III. khác")
            "labelNorm": [" ".join(n for n in map(norm_label, v) if n) for v in uniq],
            "rapidLabel": r.get("rapidLabel") or None,
            "rapidVariants": r.get("rapidVariants") or [],
            "stt": first if (first and STT.match(first)) else None,
            "labelOnly": bool(r.get("labelOnly")),
            "cells": {str(k): v for k, v in sorted(r["cells"].items())},
        })
    result["rows"] = out_rows

    # header text per column: for each header line, segments overlapping the column's x-range
    blocks = []
    for hl in header_lines:
        per_col: dict[str, list[str]] = {}
        plain = [w for w in hl["words"] if not any(ch.isdigit() for ch in w["text"])]
        for w in hl["words"]:  # dates/years: nearest column only (headers of adjacent columns sit close together)
            if any(ch.isdigit() for ch in w["text"]):
                xc = (w["x0"] + w["x1"]) / 2
                d = [abs(xc - c["center"]) for c in cols]
                k = min(range(len(cols)), key=d.__getitem__)
                if d[k] <= halfw + 0.03 * width:
                    per_col.setdefault(str(k), []).append(w["text"])
        for seg in _segments(plain, 1.5 * lmh):  # words: spanning group headers attach to every column they cover
            for k, c in enumerate(cols):
                lo, hi = c["center"] - halfw, c["right"] + 0.02 * width
                if seg["x1"] >= lo and seg["x0"] <= hi:
                    per_col.setdefault(str(k), []).append(seg["text"])
        if per_col:
            blocks.append({"y": round(hl["y"], 1), "columns": {k: " ".join(v) for k, v in per_col.items()}})
    result["headerBlocks"] = blocks
    result["medianCellHeight"] = round(mh, 1)
    return result


def headers_for_row(table: dict, row_y: float, max_gap_rows: int = 40) -> dict[str, str]:
    """Column header text of the header block nearest above ``row_y`` (statement: page header; notes: table header)."""
    blocks = [b for b in table.get("headerBlocks", []) if b["y"] < row_y]
    if not blocks:
        return {}
    # header block = run of header lines ending at the nearest one above the row
    blocks.sort(key=lambda b: b["y"])
    mh = table.get("medianCellHeight") or 30
    chosen = [blocks[-1]]
    for b in reversed(blocks[:-1]):
        if chosen[-1]["y"] - b["y"] <= 3.2 * mh:
            chosen.append(b)
        else:
            break
    out: dict[str, list[str]] = {}
    for b in reversed(chosen):
        for k, t in b["columns"].items():
            out.setdefault(k, []).append(t)
    return {k: " ".join(v) for k, v in out.items()}


def _is_header_text(text: str) -> bool:
    f = fold(text)
    if re.search(r"\d{1,2}\s*[/.\-]\s*\d{1,2}\s*[/.\-]\s*\d{4}", f) or re.search(r"\b(19|20)\d{2}\b", f):
        return True
    return any(k in f for k in HEADER_KEYWORDS) and not NOTE_REF.match(text.strip())


def is_note_ref(text: str) -> bool:
    return bool(NOTE_REF.match((text or "").strip()))
