"""Live tests for real arxiv + semantic scholar search."""

from __future__ import annotations

import pytest

from litgraph.settings import reset_settings


@pytest.fixture(autouse=True)
def _clean():
    reset_settings()
    yield
    reset_settings()


@pytest.mark.live
def test_arxiv_search():
    from litgraph.search.arxiv import search_arxiv

    results = search_arxiv(["transformer"], max_results=3)
    assert len(results) > 0
    for p in results:
        assert "title" in p
        assert "abstract" in p
        assert p["source"] == "arxiv"


@pytest.mark.live
def test_semantic_scholar_search():
    from litgraph.search.semantic import search_semantic_scholar

    results = search_semantic_scholar(["single-cell foundation model"], max_results=3)
    assert len(results) > 0
    for p in results:
        assert "title" in p
        assert "abstract" in p
        assert p["source"] == "semantic_scholar"
