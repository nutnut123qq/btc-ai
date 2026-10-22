"""Local OCR engines (V3-B0 decision): Tesseract 5 `vie` for words/labels, RapidOCR PP-OCRv6 for numbers.

All binaries are referenced by explicit path (no PATH / global install): the portable Tesseract lives under
``tools/tesseract`` (see tools/README.md) and can be overridden with EQX_TESSERACT_EXE / EQX_TESSDATA_DIR.
CPU use is bounded: Tesseract runs with OMP_THREAD_LIMIT=1, onnxruntime with EQX_OCR_THREADS (default 4).
"""
from __future__ import annotations

import csv
import io
import math
import os
import re
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TESS_DIR = REPO_ROOT / "tools" / "tesseract"
# Tesseract exits 1 on pages with (almost) no text: "Too few characters. Skipping this page" / "Empty page!!"
EMPTY_PAGE_MARKERS = re.compile(r"Too few characters|Empty page", re.IGNORECASE)


def tesseract_paths() -> tuple[Path, Path]:
    exe = Path(os.environ.get("EQX_TESSERACT_EXE") or DEFAULT_TESS_DIR / "Library" / "bin" / "tesseract.exe")
    data = Path(os.environ.get("EQX_TESSDATA_DIR") or DEFAULT_TESS_DIR / "share" / "tessdata")
    return exe, data


def ocr_available() -> tuple[bool, str | None]:
    exe, data = tesseract_paths()
    if not exe.is_file():
        return False, f"tesseract not found at {exe}"
    if not (data / "vie.traineddata").is_file():
        return False, f"vie.traineddata not found in {data}"
    try:
        import pypdfium2  # noqa: F401
        import rapidocr  # noqa: F401
    except ImportError as exc:
        return False, f"python OCR deps missing: {exc}"
    return True, None


def tool_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    exe, _ = tesseract_paths()
    try:
        r = subprocess.run([str(exe), "--version"], capture_output=True, timeout=30)
        out["tesseract"] = (r.stdout or r.stderr).decode("utf-8", "replace").splitlines()[0].strip()
    except Exception as exc:  # noqa: BLE001
        out["tesseract"] = f"unavailable: {exc}"
    for mod in ("rapidocr", "onnxruntime", "pypdfium2"):
        try:
            m = __import__(mod)
            out[mod] = str(getattr(m, "__version__", None) or getattr(m, "V_PYPDFIUM2", None) or "?")
        except Exception:  # noqa: BLE001
            out[mod] = "unavailable"
    try:
        from importlib.metadata import version
        for dist in ("rapidocr", "pypdfium2", "onnxruntime"):
            out[dist] = version(dist)
    except Exception:  # noqa: BLE001
        pass
    return out


# ------------------------------------------------------------------------------------------------
# PDF rendering
# ------------------------------------------------------------------------------------------------

class PdfPages:
    """Thin pypdfium2 wrapper. Rendering applies the page /Rotate (MBB stores BS pages rotated 270)."""

    def __init__(self, path: Path):
        import pypdfium2 as pdfium

        self._pdf = pdfium.PdfDocument(str(path))
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._pdf)

    def render(self, page_no: int, dpi: int):
        with self._lock:
            page = self._pdf[page_no - 1]
            img = page.render(scale=dpi / 72, grayscale=True).to_pil()
            rotation = page.get_rotation()
            page.close()
        return img, rotation

    def text_layer(self, page_no: int) -> str:
        with self._lock:
            page = self._pdf[page_no - 1]
            tp = page.get_textpage()
            try:
                return tp.get_text_range() or ""
            finally:
                tp.close()
                page.close()

    def close(self) -> None:
        self._pdf.close()


# ------------------------------------------------------------------------------------------------
# Skew / orientation
# ------------------------------------------------------------------------------------------------

def estimate_skew(img, max_deg: float = 3.0, step: float = 0.1) -> float:
    """Projection-profile skew estimate in degrees (positive = text lines rise to the right)."""
    import numpy as np

    small = img.convert("L")
    small = small.resize((max(1, small.width // 4), max(1, small.height // 4)))
    a = np.asarray(small) < 140
    ys, xs = np.nonzero(a)
    if len(xs) < 500:
        return 0.0
    if len(xs) > 200_000:
        idx = np.random.default_rng(0).choice(len(xs), 200_000, replace=False)
        ys, xs = ys[idx], xs[idx]
    best, best_ang = -1.0, 0.0
    n = a.shape[0] + 2 * a.shape[1] + 10
    off = a.shape[1] + 5
    for ang in np.arange(-max_deg, max_deg + 1e-9, step):
        yy = np.round(ys - math.tan(math.radians(ang)) * xs).astype(int) + off
        ok = (yy >= 0) & (yy < n)
        prof = np.bincount(yy[ok], minlength=n).astype("float64")
        v = float((prof ** 2).sum())
        if v > best:
            best, best_ang = v, float(ang)
    return round(best_ang, 2)


def deskew(img, angle: float):
    """Rotate so text lines are horizontal (white fill)."""
    if abs(angle) < 0.25:
        return img
    return img.rotate(angle, resample=3, expand=False, fillcolor=255)  # PIL: positive = counter-clockwise


def osd_rotation(img) -> int | None:
    """Tesseract OSD page orientation (0/90/180/270) or None when unknown."""
    exe, data = tesseract_paths()
    if not (data / "osd.traineddata").is_file():
        return None
    out = _run_tesseract(img, ["--psm", "0", "-l", "osd"], dpi=150)
    m = re.search(r"Rotate:\s*(\d+)", out)
    return int(m.group(1)) if m else None


# ------------------------------------------------------------------------------------------------
# Tesseract
# ------------------------------------------------------------------------------------------------

def _run_tesseract(img, args: list[str], dpi: int) -> str:
    exe, data = tesseract_paths()
    fd, tmp = tempfile.mkstemp(prefix="eqx_ocr_", suffix=".png")
    os.close(fd)
    try:
        img.save(tmp, dpi=(dpi, dpi))
        env = dict(os.environ, OMP_THREAD_LIMIT=os.environ.get("EQX_TESS_THREADS", "1"))
        r = subprocess.run([str(exe), tmp, "stdout", "--tessdata-dir", str(data), *args],
                           capture_output=True, timeout=300, env=env)
        if r.returncode != 0:
            err = r.stderr.decode("utf-8", "replace")
            if EMPTY_PAGE_MARKERS.search(err):
                return ""  # blank / near-empty page (separator sheet, signature page): no text, not a failure
            raise RuntimeError(f"tesseract failed ({r.returncode}): {err[:300]}")
        return r.stdout.decode("utf-8", "replace")
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def tesseract_text(img, dpi: int, psm: int = 4) -> str:
    return _run_tesseract(img, ["-l", "vie", "--oem", "1", "--psm", str(psm)], dpi)


def tesseract_words(img, dpi: int, psm: int = 4) -> list[dict[str, Any]]:
    """Word boxes: {text, box:[x0,y0,x1,y1], conf, line:"block.par.line"} in image pixels."""
    out = _run_tesseract(img, ["-l", "vie", "--oem", "1", "--psm", str(psm), "-c", "tessedit_create_tsv=1"], dpi)
    words = []
    for w in csv.DictReader(io.StringIO(out), delimiter="\t", quoting=csv.QUOTE_NONE):
        if w.get("level") != "5" or not (w.get("text") or "").strip():
            continue
        x, y, ww, hh = (int(w[k]) for k in ("left", "top", "width", "height"))
        words.append({"text": w["text"], "box": [x, y, x + ww, y + hh], "conf": float(w["conf"]),
                      "line": f'{w["block_num"]}.{w["par_num"]}.{w["line_num"]}'})
    return words


# ------------------------------------------------------------------------------------------------
# RapidOCR
# ------------------------------------------------------------------------------------------------

_rapid = None
_rapid_lock = threading.Lock()


def _rapid_engine():
    global _rapid
    with _rapid_lock:
        if _rapid is None:
            from rapidocr import RapidOCR

            threads = int(os.environ.get("EQX_OCR_THREADS", "4"))
            _rapid = RapidOCR(params={"EngineConfig.onnxruntime.intra_op_num_threads": threads,
                                      "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                                      "Global.log_level": "warning"})
        return _rapid


def rapid_boxes(img) -> list[dict[str, Any]]:
    """Text boxes from RapidOCR (PP-OCRv6 small, bundled ONNX models, offline)."""
    import numpy as np

    eng = _rapid_engine()
    arr = np.asarray(img.convert("RGB"))[:, :, ::-1].copy()
    with _rapid_lock:
        res = eng(arr)
    out = []
    if res is None or res.boxes is None:
        return out
    for box, txt, score in zip(res.boxes, res.txts, res.scores):
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        out.append({"text": str(txt), "box": [min(xs), min(ys), max(xs), max(ys)], "conf": float(score)})
    return out
