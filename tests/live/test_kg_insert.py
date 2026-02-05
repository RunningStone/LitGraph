"""Live test for KG insert and query."""

from __future__ import annotations

import pytest

from litgraph.settings import reset_settings


@pytest.fixture(autouse=True)
def _clean():
    reset_settings()
    yield
    reset_settings()


@pytest.mark.live
def test_insert_and_query(tmp_path, monkeypatch):
    """Insert a paper text and query — verify non-empty response."""
    monkeypatch.setenv("LITGRAPH_MODE", "pro")

    from litgraph.kg.graph import insert_texts, query_graph
    from litgraph.settings import get_settings

    get_settings(force_reload=True)

    working_dir = str(tmp_path / "kg_test")

    text = (
        "ProtoCORAL is a foundation model for single-cell RNA sequencing analysis. "
        "It uses a variational autoencoder architecture and is evaluated on the PBMC dataset. "
        "The model achieves state-of-the-art performance on cell type annotation tasks."
    )

    insert_texts([text], working_dir=working_dir)
    result = query_graph("What methods are used for single-cell analysis?", working_dir=working_dir)
    assert isinstance(result, str)
    assert len(result) > 0
