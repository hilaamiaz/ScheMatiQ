"""Tests for column-aware PDF text extraction (convert_to_txt.py).

pdfplumber's default page.extract_text() reads a two-column page in one
top-to-bottom sweep across the full page width, interleaving the two columns
word-by-word instead of reading one column fully before the other. These
tests build small, exact-geometry PDFs at test time (via reportlab) to prove
the fix: two-column pages are read column-major, while every other page
shape (single-column, figure pages, tables, sparse pages) is extracted
byte-for-byte identically to today.
"""

import re

import pytest

pytest.importorskip("reportlab")

import pdfplumber

from app.services.document_conversion.convert_to_txt import (
    _detect_column_gutter,
    _extract_page_text_column_aware,
    convert_pdf_to_txt,
)
from tests.helpers.pdf_fixtures import (
    build_figure_page_pdf,
    build_mixed_layout_pdf,
    build_short_gap_heading_pdf,
    build_single_column_pdf,
    build_sparse_pdf,
    build_two_column_pdf,
    build_two_column_table_pdf,
)

LEFT_SENTENCE = (
    "Like many other RPTPs this protein consists of an extracellular receptor "
    "like region a short transmembrane segment and a cytoplasmic region"
)
RIGHT_SENTENCE = (
    "This sidebar describes most known signaling functions of the receptor "
    "phosphatase family across many cell types and tissues throughout the body"
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def test_two_column_page_read_in_column_major_order(tmp_path):
    pdf_path = tmp_path / "two_col.pdf"
    build_two_column_pdf(pdf_path, LEFT_SENTENCE, RIGHT_SENTENCE)

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        column_aware = _extract_page_text_column_aware(page)
        default = page.extract_text() or ""

    assert _norm(LEFT_SENTENCE) in _norm(column_aware)
    assert _norm(column_aware).index(_norm(LEFT_SENTENCE).split()[0]) < _norm(column_aware).index(
        _norm(RIGHT_SENTENCE).split()[0]
    )
    # Direct regression proof: today's default scrambles the same page.
    assert _norm(LEFT_SENTENCE) not in _norm(default)


def test_single_column_page_unaffected(tmp_path):
    pdf_path = tmp_path / "one_col.pdf"
    build_single_column_pdf(pdf_path, "A perfectly ordinary single-column paragraph of text. ")

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        assert _extract_page_text_column_aware(page) == (page.extract_text() or "")


def test_detect_column_gutter_returns_none_for_single_column(tmp_path):
    pdf_path = tmp_path / "one_col.pdf"
    build_single_column_pdf(pdf_path, "A perfectly ordinary single-column paragraph of text. ")

    with pdfplumber.open(pdf_path) as pdf:
        assert _detect_column_gutter(pdf.pages[0]) is None


def test_detect_column_gutter_returns_expected_bbox_for_two_column(tmp_path):
    pdf_path = tmp_path / "two_col.pdf"
    build_two_column_pdf(pdf_path, LEFT_SENTENCE, RIGHT_SENTENCE)

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        gutter = _detect_column_gutter(page)

    assert gutter is not None
    x0, x1 = gutter
    assert x0 < x1
    # The gutter should sit roughly in the middle third of the page.
    mid_frac = ((x0 + x1) / 2) / page.width
    assert 0.35 <= mid_frac <= 0.65


def test_figure_page_not_falsely_detected_as_two_column(tmp_path):
    pdf_path = tmp_path / "figure.pdf"
    build_figure_page_pdf(
        pdf_path,
        "Figure caption describing the diagram shown here with extra detail. ",
    )

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        assert _detect_column_gutter(page) is None
        assert _extract_page_text_column_aware(page) == (page.extract_text() or "")


def test_narrow_table_not_falsely_detected_as_two_column(tmp_path):
    pdf_path = tmp_path / "table.pdf"
    build_two_column_table_pdf(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        assert _detect_column_gutter(page) is None


def test_short_gap_run_not_falsely_detected(tmp_path):
    pdf_path = tmp_path / "heading.pdf"
    build_short_gap_heading_pdf(
        pdf_path,
        "A Wide Spaced Heading",
        "Ordinary single-column body text following the heading. ",
    )

    with pdfplumber.open(pdf_path) as pdf:
        assert _detect_column_gutter(pdf.pages[0]) is None


def test_mixed_layout_document_end_to_end(tmp_path):
    pdf_path = tmp_path / "mixed.pdf"
    build_mixed_layout_pdf(pdf_path, "A Single Column Title Page", LEFT_SENTENCE, RIGHT_SENTENCE)

    output_dir = tmp_path
    success, message = convert_pdf_to_txt(pdf_path, output_dir)
    assert success, message

    text = (output_dir / "mixed.txt").read_text(encoding="utf-8")
    assert "a single column title page" in _norm(text)
    assert _norm(LEFT_SENTENCE) in _norm(text)


def test_sparse_page_does_not_crash(tmp_path):
    pdf_path = tmp_path / "sparse.pdf"
    build_sparse_pdf(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        assert _detect_column_gutter(page) is None
        assert _extract_page_text_column_aware(page) == (page.extract_text() or "")
