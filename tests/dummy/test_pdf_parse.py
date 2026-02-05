"""Tests for PDF text extraction and cleanup logic."""

from __future__ import annotations

import pytest

from litgraph.analysis.paper import cleanup_pdf_text, extract_pdf_text


@pytest.fixture
def fixture_pdf(tmp_path):
    """Create a small test PDF with pymupdf programmatically."""
    import pymupdf

    pdf_path = tmp_path / "test.pdf"
    doc = pymupdf.open()

    # Page 1
    page = doc.new_page()
    text_point = pymupdf.Point(72, 100)
    page.insert_text(text_point, "This is the first page of the paper.")
    text_point2 = pymupdf.Point(72, 120)
    page.insert_text(text_point2, "It discusses founda-")

    # Page 2
    page2 = doc.new_page()
    text_point3 = pymupdf.Point(72, 100)
    page2.insert_text(text_point3, "tion models for single-cell analysis.")
    text_point4 = pymupdf.Point(72, 120)
    page2.insert_text(text_point4, "The results show improvement.")
    # Page number
    text_point5 = pymupdf.Point(300, 750)
    page2.insert_text(text_point5, "2")

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


class TestExtractPdfText:
    def test_extracts_text(self, fixture_pdf):
        text = extract_pdf_text(fixture_pdf)
        assert "first page" in text
        assert "single-cell" in text

    def test_max_pages(self, fixture_pdf):
        """Should respect max_pages limit."""
        text = extract_pdf_text(fixture_pdf, max_pages=1)
        assert "first page" in text
        # Second page content should not be present
        assert "results show" not in text

    def test_returns_string(self, fixture_pdf):
        text = extract_pdf_text(fixture_pdf)
        assert isinstance(text, str)


class TestCleanupPdfText:
    def test_merge_hyphenated_breaks(self):
        text = "This discusses founda-\ntion models."
        cleaned = cleanup_pdf_text(text)
        assert "foundation" in cleaned

    def test_page_number_removal(self):
        text = "Some content.\n\n  42  \n\nMore content."
        cleaned = cleanup_pdf_text(text)
        assert "42" not in cleaned

    def test_collapse_blank_lines(self):
        text = "Line 1\n\n\n\n\nLine 2"
        cleaned = cleanup_pdf_text(text)
        assert "\n\n\n" not in cleaned

    def test_preserves_normal_content(self):
        text = "This is normal text.\nWith line breaks.\n\nAnd paragraphs."
        cleaned = cleanup_pdf_text(text)
        assert "normal text" in cleaned
        assert "paragraphs" in cleaned

    def test_strip_whitespace(self):
        text = "\n\n  Some content  \n\n"
        cleaned = cleanup_pdf_text(text)
        assert cleaned == "Some content"
