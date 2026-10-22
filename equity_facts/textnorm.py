"""Text / number / unit normalization for Vietnamese bank statements (pure functions, no I/O).

Numbers are parsed fail-closed: a value is only produced for a clean grouped integer such as
``1.234.567`` / ``1,234,567`` / ``(1.234.567)`` / ``-`` (nil). Anything else (mixed separators,
broken groups such as ``1.452. 7 42``, letters) is ``malformed`` and never guessed.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

_SPECKLE = ".,·'`’‘:;|_~"
_DASHES = {"-", "–", "—", "−", "--"}
_GROUPED = re.compile(r"^\d{1,3}(?:[.,]\d{3})+$")
_PLAIN = re.compile(r"^\d{1,3}$")
NUMERIC_TOKEN = re.compile(r"^[\s\d.,()\-–—−·'`’‘]+$")
GROUPED_ANYWHERE = re.compile(r"\d[.,]\d{3}")
NOTE_REF = re.compile(r"^(?:[IVXivx]{1,5}(?:\.\d{1,2}){0,3}|\d{1,2}(?:[.,]\d{1,2}){0,3})\.?$")
# roman numerals may carry an H or W: OCR reads "II" as "HI" (BIDV "XIHI.", "IHI"), "IV" as "IW"
STT = re.compile(r"^(?:[IVX][IVXHW]{0,4}|[A-H]|[a-h]|\d{1,2})[.)_:;]?$")


def fold(text: str) -> str:
    """Upper-case, diacritic-free, single-spaced form used for matching (đ -> D)."""
    s = unicodedata.normalize("NFKC", text or "")  # "Ⅲ" (RapidOCR) -> "III"
    s = s.replace("đ", "d").replace("Đ", "D")
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s.upper()).strip()


def norm_label(text: str) -> str:
    """Folded label without punctuation, leading STT numbering or trailing note references."""
    # upper-case "HI." / "HII." = mangled II / III; case-sensitive so "hi phí" ("Chi phí" minus C) survives
    s = fold(re.sub(r"^\s*H[IH]{1,3}[.)]\s+", "", text or ""))
    s = re.sub(r"[“”\"'`’‘()\[\]{}*,;:]", " ", s)
    s = re.sub(r"[^\w/ ]|_", " ", s)
    toks = s.split()
    while toks and (STT.match(toks[0]) or toks[0] in {"-", "+"}):
        toks.pop(0)
    if len(toks) > 3 and re.fullmatch(r"NH[OA]M", toks[0]):  # "Nhóm 1 - Nợ đủ tiêu chuẩn" (OCR "Nhằm 4", "Nhóm l")
        del toks[:2 if re.fullmatch(r"[0-9IVXLS]{1,2}", toks[1]) else 1]
    # trailing note references, also OCR-mangled ones ("III.5" read as "lII.B" / "1I1.5")
    while toks and (NOTE_REF.match(toks[-1]) or re.fullmatch(r"[IVXL1]{1,4}|[A-Z0-9]|\d[A-Z]", toks[-1])):
        toks.pop()
    return " ".join(toks)


def label_score(a: str, b: str) -> float:
    """Similarity of two already-normalized labels (0..1)."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    r = SequenceMatcher(None, a, b, autojunk=False).ratio()
    a2, b2 = a.replace(" ", ""), b.replace(" ", "")
    if a2 == b2:
        return max(r, 0.97)
    r2 = SequenceMatcher(None, a2, b2, autojunk=False).ratio()
    toks = b.split()
    r3 = 0.0
    if len(toks) > 2 and len(toks[0]) <= 3:  # unreadable STT / bullet glued in front ("VH CHUNG KHOAN...")
        r3 = 0.97 * SequenceMatcher(None, a, " ".join(toks[1:]), autojunk=False).ratio()
    r4 = 0.0
    if 0.6 * len(a) <= len(b) < len(a):  # wrapped label whose continuation line was unreadable
        r4 = 0.95 * SequenceMatcher(None, a[:len(b)], b, autojunk=False).ratio()
    return max(r, r2, r3, r4)


def parse_amount(raw: str) -> dict:
    """Parse one printed amount cell.

    Returns {"kind": "number"|"dash"|"malformed", "value": int|None, "text": cleaned}.
    Parentheses (either side, scanners sometimes lose one) or a leading minus mean negative.
    """
    t = (raw or "").strip().replace(" ", "").replace(" ", "")
    t = t.strip(_SPECKLE)
    if t in _DASHES or (t and set(t) <= set("-–—−")):
        return {"kind": "dash", "value": None, "text": t}
    neg = t.startswith("(") or t.endswith(")") or t[:1] in {"-", "–", "—", "−"}
    core = t.strip("()-–—−").strip(_SPECKLE)
    if _GROUPED.match(core):
        seps = {c for c in core if c in ".,"}
        if len(seps) != 1:  # 1.234,567 -> never guess which one is decimal
            return {"kind": "malformed", "value": None, "text": t}
    elif not _PLAIN.match(core) or (len(core) > 1 and core.startswith("0")):  # "01" = a misread "-"
        return {"kind": "malformed", "value": None, "text": t}
    v = int(re.sub(r"\D", "", core))
    return {"kind": "number", "value": -v if neg else v, "text": t}


def is_numeric_token(text: str) -> bool:
    t = (text or "").strip()
    return bool(t) and bool(NUMERIC_TOKEN.match(t)) and (any(ch.isdigit() for ch in t) or t.strip(_SPECKLE) in _DASHES)


UNIT_PATTERNS = [  # (folded regex, scale)
    (r"TRIEU\s*(?:VND|ND\b|VN\b|D\w{0,2}NG\b|D\b)", 1_000_000),
    (r"\bTY\s*(?:VND|DONG)", 1_000_000_000),
    (r"NGHIN\s*(?:VND|DONG)", 1_000),
    (r"DON\s*VI(?:\s*TINH)?\s*[:.]?\s*(?:VND|DONG)\b", 1),
]


def detect_unit(text: str) -> tuple[int, str] | None:
    """Find the amount unit declared on a page ("Đơn vị: Triệu VND", "triệu đồng" column headers...).

    Returns (scale, raw text as printed) or None. Several conflicting units on one page -> None (fail closed).
    """
    f = fold(text)
    found: dict[int, str] = {}
    for pat, scale in UNIT_PATTERNS:
        m = re.search(pat, f)
        if m:
            found.setdefault(scale, _raw_slice(text, m.group(0)))
    if len(found) == 1:
        scale, raw = next(iter(found.items()))
        return scale, raw
    if len(found) > 1 and 1_000_000 in found and 1 in found:
        # "Đơn vị: VND" never co-occurs with "triệu" on a real page; ambiguous -> None
        return None
    return None


def _raw_slice(text: str, folded_match: str) -> str:
    """Best-effort original-text snippet for a folded match (for RawUnitText)."""
    words = folded_match.split()
    for line in (text or "").splitlines():
        if all(w in fold(line) for w in words):
            m = re.search(r"(?i)(đơn\s*vị[^:]{0,6}:\s*)?((triệu|tỷ|nghìn)\s*)?(vnd|nd|đ\w{0,2}ng|dong)\b", line)
            return (m.group(0) if m else line).strip()[:100]
    return folded_match[:100]


DATE_RE = re.compile(r"(\d{1,2})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*(\d{4})")


def find_dates(text: str) -> list[tuple[int, int, int]]:
    """All dd/mm/yyyy dates in reading order as (y, m, d), invalid ones dropped."""
    out = []
    for d, m, y in DATE_RE.findall(text or ""):
        d, m, y = int(d), int(m), int(y)
        if 1 <= m <= 12 and 1 <= d <= 31 and 1990 <= y <= 2100:
            out.append((y, m, d))
    return out
