"""Multi-format document to plain-text conversion.

Converts DOCX, DOC, RTF, and PDF files to UTF-8 text.

- DOCX: python-docx first, LibreOffice headless fallback
- DOC / RTF: LibreOffice headless
- PDF: pdfplumber first, OCR fallback (pdf2image + tesseract)

Vendored from Legal_Schema_Generator; adapted for use as a library inside
the ScheMatiQ backend (no CLI entry point, no process-wide signal handlers,
structured logging instead of print()).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import pdfplumber
import pytesseract
from docx import Document
from pdf2image import convert_from_path
from pdfplumber.utils import cluster_objects

logger = logging.getLogger(__name__)

LIBREOFFICE_EXTENSIONS = {".rtf", ".docx", ".doc"}
PDF_EXTENSION = ".pdf"
SUPPORTED_EXTENSIONS = LIBREOFFICE_EXTENSIONS | {PDF_EXTENSION}


# ---------------------------------------------------------------------------
# LibreOffice helpers
# ---------------------------------------------------------------------------

def get_libreoffice_path() -> str:
    """Return the LibreOffice soffice executable path.

    Raises FileNotFoundError when LibreOffice is not installed.
    """
    candidates = {
        "darwin": "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        "linux": "/usr/bin/soffice",
        "win32": r"C:\Program Files\LibreOffice\program\soffice.exe",
    }
    default_path = candidates.get(sys.platform, "/usr/bin/soffice")
    if os.path.exists(default_path):
        return default_path

    found = shutil.which("soffice")
    if found:
        return found

    raise FileNotFoundError(
        "LibreOffice not found. Install it or add soffice to PATH."
    )


def _run_libreoffice(
    input_path: Path,
    output_dir: Path,
    soffice_path: str,
    *,
    profile_dir: Optional[Path] = None,
    max_retries: int = 3,
) -> Tuple[bool, str]:
    """Run LibreOffice headless to convert *input_path* to text.

    Each call uses its own user-profile directory (under *profile_dir*, or a
    fresh tempdir) so concurrent invocations don't collide.
    """
    cleanup_profile = False
    if profile_dir is None:
        profile_dir = Path(tempfile.mkdtemp(prefix="lo_profile_"))
        cleanup_profile = True

    cmd = [
        soffice_path,
        "--headless",
        f"-env:UserInstallation=file://{profile_dir}",
        "--convert-to", "txt:Text",
        "--outdir", str(output_dir),
        str(input_path),
    ]

    last_error = ""
    try:
        for attempt in range(max_retries):
            try:
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=120,
                )
                expected = output_dir / (input_path.stem + ".txt")
                if result.returncode == 0 and expected.exists():
                    return True, f"Converted: {expected}"

                last_error = (result.stderr.decode() if result.stderr else "Unknown error")
            except subprocess.TimeoutExpired:
                last_error = "Conversion timed out after 120 s"
            except Exception as e:
                last_error = str(e)

            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
    finally:
        if cleanup_profile:
            shutil.rmtree(profile_dir, ignore_errors=True)

    return False, f"LibreOffice conversion failed after {max_retries} attempts: {last_error}"


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------

def convert_docx_to_txt(input_path: Path, output_dir: Path) -> Tuple[bool, str]:
    """Extract text from a DOCX using python-docx (paragraphs + tables)."""
    output_path = output_dir / (input_path.stem + ".txt")
    try:
        doc = Document(input_path)
        parts: list[str] = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    t = cell.text.strip()
                    if t:
                        parts.append(t)
        if not parts:
            return False, "No text extracted from DOCX"
        output_path.write_text("\n".join(parts), encoding="utf-8")
        return True, f"Converted: {output_path}"
    except Exception as e:
        return False, f"DOCX extraction failed: {e}"


# ---------------------------------------------------------------------------
# PDF (pdfplumber → OCR fallback)
# ---------------------------------------------------------------------------

_CHARS_PER_PAGE_THRESHOLD = 200

# pdfplumber's default extract_text() reads a page in one top-to-bottom sweep
# across the full page width. On a two-column layout this interleaves the two
# columns word-by-word/line-by-line instead of reading one column fully
# before the other, scrambling sentences that span the interleaving.
# _detect_column_gutter/_extract_page_text_column_aware below detect a
# genuine column gutter (a consistent vertical gap in word positions, not
# just normal ragged-right word spacing) and, only when confident, read the
# left column fully then the right column fully instead.

_MIN_WORDS_FOR_COLUMN_DETECTION = 40
_LINE_CLUSTER_TOLERANCE_PT = 3
_MIN_RAW_GAP_PT = 12  # per-line candidate gap threshold (normal word spacing is ~2-3pt)
_MIN_FINAL_GUTTER_WIDTH_PT = 8  # width required after intersecting the gap across lines
_GUTTER_ZONE_FRACTION = (0.30, 0.70)  # gutter midpoint must fall in the page's middle band
_MIN_GAP_HEIGHT_FRACTION = 0.55  # gap's vertical span / page height
_MIN_GAP_LINE_FRACTION = 0.55  # lines-with-the-gap / lines-within-that-vertical-span
_MIN_LINES_WITH_GAP = 8
_MIN_SIDE_WORD_FRACTION = 0.15  # each side must hold a real share of the page's words
_MIN_AVG_WORDS_PER_LINE_PER_SIDE = 3  # guards against two-column numeric tables


def _detect_column_gutter(page: "pdfplumber.page.Page") -> Optional[Tuple[float, float]]:
    """Return (gutter_x0, gutter_x1) if *page* has a confident two-column
    layout, else None. A pure function of page geometry -- no I/O.
    """
    if getattr(page, "rotation", 0) not in (0, None):
        return None  # rotated pages: x/top axes don't map to visual columns reliably
    if not page.width or not page.height:
        return None  # degenerate/malformed page geometry -- nothing safe to measure

    words = [w for w in page.extract_words() if w.get("upright", True)]
    if len(words) < _MIN_WORDS_FOR_COLUMN_DETECTION:
        return None

    lines = cluster_objects(words, "top", _LINE_CLUSTER_TOLERANCE_PT)
    lines = [sorted(line, key=lambda w: w["x0"]) for line in lines]
    line_spans = [(min(w["top"] for w in line), max(w["bottom"] for w in line), line) for line in lines]

    # Per-line candidate gaps wide enough, and roughly centered, to plausibly
    # be a column gutter rather than ordinary word spacing.
    candidates = []  # (x0, x1, top, bottom)
    for top, bottom, line in line_spans:
        for a, b in zip(line, line[1:]):
            gap = b["x0"] - a["x1"]
            if gap < _MIN_RAW_GAP_PT:
                continue
            mid_frac = ((a["x1"] + b["x0"]) / 2) / page.width
            if _GUTTER_ZONE_FRACTION[0] <= mid_frac <= _GUTTER_ZONE_FRACTION[1]:
                candidates.append((a["x1"], b["x0"], top, bottom))

    if not candidates:
        return None

    # Merge candidates whose x-ranges mutually intersect, tracking the running
    # intersection -- this is what narrows ragged-edge per-line gaps down to
    # the one true, tight gutter shared across most lines.
    candidates.sort(key=lambda c: c[0])
    groups: list[dict] = []
    for c in candidates:
        merged = False
        for g in groups:
            new_x0, new_x1 = max(g["x0"], c[0]), min(g["x1"], c[1])
            if new_x1 - new_x0 >= _MIN_FINAL_GUTTER_WIDTH_PT:
                g.update(x0=new_x0, x1=new_x1, top=min(g["top"], c[2]),
                         bottom=max(g["bottom"], c[3]), count=g["count"] + 1)
                merged = True
                break
        if not merged:
            groups.append({"x0": c[0], "x1": c[1], "top": c[2], "bottom": c[3], "count": 1})

    if not groups:
        return None
    best = max(groups, key=lambda g: g["count"])

    if best["x1"] - best["x0"] < _MIN_FINAL_GUTTER_WIDTH_PT:
        return None
    if best["count"] < _MIN_LINES_WITH_GAP:
        return None
    if (best["bottom"] - best["top"]) / page.height < _MIN_GAP_HEIGHT_FRACTION:
        return None

    lines_in_span = [ls for ls in line_spans
                      if ls[0] >= best["top"] - 1 and ls[1] <= best["bottom"] + 1]
    if not lines_in_span or best["count"] / len(lines_in_span) < _MIN_GAP_LINE_FRACTION:
        return None

    mid = (best["x0"] + best["x1"]) / 2

    def _in_span(w: dict) -> bool:
        return w["top"] >= best["top"] - 1 and w["bottom"] <= best["bottom"] + 1

    left_words = [w for w in words if w["x1"] <= mid and _in_span(w)]
    right_words = [w for w in words if w["x0"] >= mid and _in_span(w)]
    total = len(left_words) + len(right_words)
    if total == 0:
        return None
    if (len(left_words) / total < _MIN_SIDE_WORD_FRACTION
            or len(right_words) / total < _MIN_SIDE_WORD_FRACTION):
        return None

    # Guard against two-column numeric tables (a wide gap but few words per
    # line on each side).
    n_lines = max(1, best["count"])
    if (len(left_words) / n_lines < _MIN_AVG_WORDS_PER_LINE_PER_SIDE
            or len(right_words) / n_lines < _MIN_AVG_WORDS_PER_LINE_PER_SIDE):
        return None

    # _MIN_GAP_LINE_FRACTION tolerates some lines within the gutter's
    # vertical span not having the gap -- but a line that's genuinely
    # full-width (a caption, table row, or equation spanning the whole
    # page) has a word straddling `mid` itself. within_bbox only keeps
    # objects FULLY inside a bbox, so such a word would fall in neither the
    # left nor the right crop and be silently dropped from the output. Bail
    # out to today's plain extract_text() rather than lose text.
    if any(w["x0"] < mid < w["x1"] for w in words if _in_span(w)):
        return None

    return best["x0"], best["x1"]


def _extract_page_text_column_aware(page: "pdfplumber.page.Page") -> str:
    """Extract a page's text, reading a confidently-detected two-column page
    column-major (full left column, then full right column) instead of
    pdfplumber's default row-major sweep. Single-column pages -- and any page
    where no confident column gutter is found -- are extracted exactly as
    `page.extract_text()` would today.

    Column detection and splitting run on arbitrary, uncontrolled real-world
    PDFs (malformed fonts/encodings, degenerate geometry); any unexpected
    failure there falls back to today's plain extraction rather than
    failing the whole document's conversion.
    """
    try:
        gutter = _detect_column_gutter(page)
        if gutter is None:
            return page.extract_text() or ""

        mid = (gutter[0] + gutter[1]) / 2
        left = page.within_bbox((0, 0, mid, page.height)).extract_text() or ""
        right = page.within_bbox((mid, 0, page.width, page.height)).extract_text() or ""

        # Cheap RTL accommodation: if most words on the page are RTL-directed,
        # the visually-first column is the right one.
        words = page.extract_words()
        if words and sum(1 for w in words if w.get("direction") == "rtl") / len(words) > 0.5:
            left, right = right, left

        return "\n\n".join(t for t in (left, right) if t)
    except Exception:
        return page.extract_text() or ""


def convert_pdf_to_txt(input_path: Path, output_dir: Path) -> Tuple[bool, str]:
    """Convert a PDF to text via pdfplumber; fall back to OCR for scanned pages."""
    output_path = output_dir / (input_path.stem + ".txt")

    with pdfplumber.open(input_path) as pdf:
        page_count = max(1, len(pdf.pages))
        text_parts = [_extract_page_text_column_aware(p) for p in pdf.pages]

    total_chars = sum(len(t) for t in text_parts)

    if total_chars / page_count >= _CHARS_PER_PAGE_THRESHOLD:
        output_path.write_text(
            "\n\n".join(t for t in text_parts if t), encoding="utf-8",
        )
        return True, f"Converted: {output_path}"

    # Scanned PDF — OCR page-by-page to avoid loading all images into RAM.
    with tempfile.TemporaryDirectory(prefix="ocr_pages_") as tmpdir:
        try:
            page_images = convert_from_path(
                str(input_path),
                dpi=300,
                output_folder=tmpdir,
                fmt="png",
            )
        except Exception as e:
            return False, f"pdf2image failed: {e}"

        ocr_parts: list[str] = []
        for img in page_images:
            text = pytesseract.image_to_string(img)
            if text.strip():
                ocr_parts.append(text)
            # Let each PIL image be freed before the next page is processed.
            img.close()

    if not ocr_parts:
        return False, "No text extracted from PDF (tried pdfplumber and OCR)"

    output_path.write_text("\n\n".join(ocr_parts), encoding="utf-8")
    return True, f"Converted (OCR): {output_path}"


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def convert_file(
    input_path: Path,
    output_dir: Path,
    soffice_path: str,
    worker_id: Optional[str] = None,
) -> Tuple[bool, str]:
    """Convert a single file to text based on its extension.

    For DOCX, tries python-docx first and falls back to LibreOffice.
    """
    ext = input_path.suffix.lower()

    if ext == ".docx":
        ok, msg = convert_docx_to_txt(input_path, output_dir)
        if ok:
            return ok, msg
        logger.debug("python-docx failed for %s, falling back to LibreOffice", input_path.name)
        return _run_libreoffice(input_path, output_dir, soffice_path)

    if ext in {".doc", ".rtf"}:
        return _run_libreoffice(input_path, output_dir, soffice_path)

    if ext == PDF_EXTENSION:
        return convert_pdf_to_txt(input_path, output_dir)

    return False, f"Unsupported file type: {ext}"
