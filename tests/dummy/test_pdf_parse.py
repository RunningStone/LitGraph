"""Tests for PDF text extraction and cleanup logic."""

from __future__ import annotations

import pytest

from litgraph.analysis.paper import cleanup_pdf_text, extract_pdf_text


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
