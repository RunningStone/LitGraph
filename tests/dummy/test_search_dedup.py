"""Tests for paper deduplication: key generation, single-run dedup, cross-run merge."""

from __future__ import annotations

import json

import pytest

from litgraph.search.dedup import (
    _title_hash,
    dedup_key,
    dedup_paper_list,
    merge_into_index,
)


class TestDedupKey:
    def test_arxiv_priority(self):
        """arxiv_id takes highest priority."""
        paper = {"arxiv_id": "2401.12345", "doi": "10.1234/test", "title": "Some Title"}
        assert dedup_key(paper) == "arxiv:2401.12345"

    def test_doi_second(self):
        """doi used when no arxiv_id."""
        paper = {"arxiv_id": None, "doi": "10.1234/test", "title": "Some Title"}
        assert dedup_key(paper) == "doi:10.1234/test"

    def test_title_hash_fallback(self):
        """title_hash used when neither arxiv_id nor doi."""
        paper = {"arxiv_id": None, "doi": None, "title": "Some Title"}
        key = dedup_key(paper)
        assert key.startswith("title:")
        assert len(key) == len("title:") + 16

    def test_empty_arxiv_uses_doi(self):
        """Empty string arxiv_id should fall through to doi."""
        paper = {"arxiv_id": "", "doi": "10.1234/test", "title": "T"}
        assert dedup_key(paper) == "doi:10.1234/test"

    def test_missing_fields_uses_title(self):
        """Missing arxiv_id and doi fields → title hash."""
        paper = {"title": "Just a Title"}
        key = dedup_key(paper)
        assert key.startswith("title:")


class TestTitleHash:
    def test_normalization_lowercase(self):
        """Title hash should be case-insensitive."""
        assert _title_hash("Hello World") == _title_hash("hello world")

    def test_normalization_punctuation(self):
        """Punctuation should be stripped."""
        assert _title_hash("Hello, World!") == _title_hash("Hello World")

    def test_normalization_whitespace(self):
        """Extra whitespace should be collapsed."""
        assert _title_hash("Hello   World") == _title_hash("Hello World")

    def test_hash_length(self):
        """Hash should be exactly 16 hex chars."""
        h = _title_hash("Test Title")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_titles_different_hashes(self):
        """Different titles should produce different hashes."""
        assert _title_hash("Paper A") != _title_hash("Paper B")


class TestDedupPaperList:
    def test_basic_dedup(self, fixture_papers):
        """Should remove duplicates within a single list."""
        papers = fixture_papers + [fixture_papers[0].copy()]  # Duplicate first paper
        result = dedup_paper_list(papers)
        assert len(result) == 3

    def test_keeps_first_occurrence(self):
        """When duplicates exist, keep the first one."""
        papers = [
            {"arxiv_id": "2401.12345", "title": "First", "source": "arxiv"},
            {"arxiv_id": "2401.12345", "title": "Second", "source": "semantic"},
        ]
        result = dedup_paper_list(papers)
        assert len(result) == 1
        assert result[0]["title"] == "First"

    def test_adds_dedup_key(self):
        """Each paper in result should have a dedup_key field."""
        papers = [{"arxiv_id": "2401.12345", "title": "Test"}]
        result = dedup_paper_list(papers)
        assert "dedup_key" in result[0]
        assert result[0]["dedup_key"] == "arxiv:2401.12345"

    def test_empty_list(self):
        """Empty list should return empty."""
        assert dedup_paper_list([]) == []


class TestMergeIntoIndex:
    def test_add_new_papers(self, data_dir, fixture_papers):
        """New papers should be added to index."""
        index_path = data_dir / "papers" / "index.json"
        added, updated = merge_into_index(fixture_papers, index_path)
        assert len(added) == 3
        assert len(updated) == 0
        assert index_path.exists()

        with open(index_path) as f:
            data = json.load(f)
        assert len(data) == 3

    def test_skip_existing(self, data_dir, fixture_papers):
        """Existing papers should not be re-added."""
        index_path = data_dir / "papers" / "index.json"
        merge_into_index(fixture_papers, index_path)
        added, updated = merge_into_index(fixture_papers, index_path)
        assert len(added) == 0

        with open(index_path) as f:
            data = json.load(f)
        assert len(data) == 3

    def test_update_meta_fields(self, data_dir, fixture_papers):
        """Updated meta fields (citations, doi, pdf_url) should be refreshed."""
        index_path = data_dir / "papers" / "index.json"
        merge_into_index(fixture_papers, index_path)

        updated_paper = fixture_papers[0].copy()
        updated_paper["citations"] = 999
        added, updated = merge_into_index([updated_paper], index_path)
        assert len(added) == 0
        assert len(updated) == 1

        with open(index_path) as f:
            data = json.load(f)
        matching = [p for p in data if p.get("arxiv_id") == "2401.12345"]
        assert matching[0]["citations"] == 999

    def test_add_and_update_mixed(self, data_dir, fixture_papers):
        """Mix of new and existing papers."""
        index_path = data_dir / "papers" / "index.json"
        merge_into_index(fixture_papers[:1], index_path)

        new_paper = {"arxiv_id": "9999.99999", "title": "Brand New", "doi": None}
        added, updated = merge_into_index([fixture_papers[0], new_paper], index_path)
        assert len(added) == 1
        assert added[0]["title"] == "Brand New"

    def test_empty_new_papers(self, data_dir):
        """Empty list should create empty index."""
        index_path = data_dir / "papers" / "index.json"
        added, updated = merge_into_index([], index_path)
        assert len(added) == 0
        assert len(updated) == 0
