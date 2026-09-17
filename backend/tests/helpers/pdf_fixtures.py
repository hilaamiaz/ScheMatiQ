"""Build small, exact-geometry PDFs for testing column-aware text extraction.

Uses reportlab's low-level `Canvas` (not Platypus) so column x-positions and
page margins are known constants the tests can assert against, and
`simpleSplit` for accurate per-column line wrapping (naive word-chunking can
silently let a line overflow into the gutter and defeat column detection).
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen.canvas import Canvas

_FONT_NAME = "Helvetica"
_FONT_SIZE = 10
_LINE_HEIGHT = 14
_TOP_MARGIN = 50
_BOTTOM_MARGIN = 50


def _wrapped_lines(text: str, col_width: float, *, repeat: int = 20) -> list[str]:
    """Wrap `text` (repeated so it's long enough to fill a full page) to `col_width`."""
    return simpleSplit(text * repeat, _FONT_NAME, _FONT_SIZE, col_width)


def _draw_column(c: Canvas, lines: list[str], x: float, page_height: float) -> None:
    """Draw `lines` top-down starting near the top margin, stopping at the bottom margin."""
    y = page_height - _TOP_MARGIN
    for line in lines:
        if y < _BOTTOM_MARGIN:
            break
        c.drawString(x, y, line)
        y -= _LINE_HEIGHT


def build_two_column_pdf(
    path: Path,
    left_text: str,
    right_text: str,
    *,
    page_size=letter,
) -> None:
    """A single page with two genuine, full-height text columns."""
    width, height = page_size
    left_x, right_x = 50, width / 2 + 15
    col_width = width / 2 - 65

    c = Canvas(str(path), pagesize=page_size)
    c.setFont(_FONT_NAME, _FONT_SIZE)
    _draw_column(c, _wrapped_lines(left_text, col_width), left_x, height)
    _draw_column(c, _wrapped_lines(right_text, col_width), right_x, height)
    c.showPage()
    c.save()


def build_single_column_pdf(path: Path, text: str, *, page_size=letter) -> None:
    """A single page of ordinary full-width single-column prose."""
    width, height = page_size
    x = 72
    col_width = width - 144

    c = Canvas(str(path), pagesize=page_size)
    c.setFont(_FONT_NAME, _FONT_SIZE)
    _draw_column(c, _wrapped_lines(text, col_width), x, height)
    c.showPage()
    c.save()


def build_figure_page_pdf(path: Path, caption_text: str, *, page_size=letter) -> None:
    """A page with one full-height column of prose and nothing on the other
    half (simulating a page mostly occupied by a figure/image)."""
    width, height = page_size
    x = 50
    col_width = width / 2 - 65

    c = Canvas(str(path), pagesize=page_size)
    c.setFont(_FONT_NAME, _FONT_SIZE)
    _draw_column(c, _wrapped_lines(caption_text, col_width), x, height)
    c.showPage()
    c.save()


def build_two_column_table_pdf(path: Path, *, rows: int = 40, page_size=letter) -> None:
    """A page of two sparse numeric columns (1-2 short tokens per line, wide
    gap between) -- geometrically similar to a two-column gutter but with far
    too little content per line to be real prose."""
    width, height = page_size
    left_x, right_x = 60, width / 2 + 20

    c = Canvas(str(path), pagesize=page_size)
    c.setFont(_FONT_NAME, _FONT_SIZE)
    y = height - _TOP_MARGIN
    for i in range(rows):
        if y < _BOTTOM_MARGIN:
            break
        c.drawString(left_x, y, f"{i}.{i}")
        c.drawString(right_x, y, f"{i * 10}.{i}")
        y -= _LINE_HEIGHT
    c.showPage()
    c.save()


def build_short_gap_heading_pdf(path: Path, heading: str, body_text: str, *, page_size=letter) -> None:
    """A single-column page whose heading has a wide letter/word-spaced gap on
    only its first couple of lines, followed by ordinary single-column body
    text -- should not be mistaken for a two-column layout."""
    width, height = page_size
    x = 72
    col_width = width - 144

    c = Canvas(str(path), pagesize=page_size)
    c.setFont(_FONT_NAME, 14)
    left_half, right_half = heading[: len(heading) // 2], heading[len(heading) // 2 :]
    y = height - _TOP_MARGIN
    c.drawString(x, y, left_half)
    c.drawString(x + width / 2, y, right_half)
    y -= 20

    c.setFont(_FONT_NAME, _FONT_SIZE)
    _draw_column_from_y(c, _wrapped_lines(body_text, col_width), x, y)
    c.showPage()
    c.save()


def _draw_column_from_y(c: Canvas, lines: list[str], x: float, start_y: float) -> None:
    y = start_y
    for line in lines:
        if y < _BOTTOM_MARGIN:
            break
        c.drawString(x, y, line)
        y -= _LINE_HEIGHT


def build_mixed_layout_pdf(
    path: Path,
    title: str,
    left_text: str,
    right_text: str,
    *,
    page_size=letter,
) -> None:
    """A two-page PDF: page 1 single-column (title/abstract), page 2 two-column body."""
    width, height = page_size
    c = Canvas(str(path), pagesize=page_size)

    c.setFont(_FONT_NAME, _FONT_SIZE)
    _draw_column(c, _wrapped_lines(title, width - 144, repeat=1), 72, height)
    c.showPage()

    left_x, right_x = 50, width / 2 + 15
    col_width = width / 2 - 65
    c.setFont(_FONT_NAME, _FONT_SIZE)
    _draw_column(c, _wrapped_lines(left_text, col_width), left_x, height)
    _draw_column(c, _wrapped_lines(right_text, col_width), right_x, height)
    c.showPage()
    c.save()


def build_sparse_pdf(path: Path, text: str = "Hi.", *, page_size=letter) -> None:
    """A near-blank page with only a couple of words."""
    c = Canvas(str(path), pagesize=page_size)
    c.setFont(_FONT_NAME, _FONT_SIZE)
    c.drawString(72, page_size[1] - 72, text)
    c.showPage()
    c.save()
