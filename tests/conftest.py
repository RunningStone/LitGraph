"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import networkx as nx
import pytest


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


@pytest.fixture
def fixture_papers():
    """3 hardcoded paper metadata dicts (no network needed)."""
    return [
        {
            "paper_id": "arxiv:2401.12345",
            "arxiv_id": "2401.12345",
            "doi": "10.1234/test1",
            "title": "ProtoCORAL: A Foundation Model for Single-Cell Analysis",
            "authors": ["Alice Smith", "Bob Jones"],
            "year": 2024,
            "abstract": "We propose ProtoCORAL, a foundation model for single-cell RNA sequencing data.",
            "source": "arxiv",
            "citations": 15,
            "pdf_url": "https://arxiv.org/pdf/2401.12345.pdf",
        },
        {
            "paper_id": "arxiv:2402.67890",
            "arxiv_id": "2402.67890",
            "doi": None,
            "title": "Spatial Transcriptomics with Graph Neural Networks",
            "authors": ["Charlie Lee"],
            "year": 2024,
            "abstract": "A GNN-based approach for spatial transcriptomics analysis.",
            "source": "arxiv",
            "citations": 8,
            "pdf_url": "https://arxiv.org/pdf/2402.67890.pdf",
        },
        {
            "paper_id": "doi:10.5678/test3",
            "arxiv_id": None,
            "doi": "10.5678/test3",
            "title": "Deep Learning for Cell Type Annotation",
            "authors": ["Diana Wu", "Eve Zhang"],
            "year": 2023,
            "abstract": "A deep learning framework for automatic cell type annotation from scRNA-seq data.",
            "source": "semantic_scholar",
            "citations": 42,
            "pdf_url": None,
        },
    ]


@pytest.fixture
def data_dir(tmp_path):
    """Temporary DATA directory structure."""
    dirs = ["papers", "analysis", "reports", "runs", "kg_store"]
    for d in dirs:
        (tmp_path / d).mkdir()
    return tmp_path


@pytest.fixture
def mock_llm():
    """Returns a mock LLM complete function that returns fixed text."""
    mock = MagicMock()
    mock.return_value = (
        "## What problem does this paper address?\n\n"
        "This paper addresses the problem of single-cell analysis.\n\n"
        "## What is the core innovation?\n\n"
        "The core innovation is a foundation model approach.\n"
    )
    return mock


@pytest.fixture
def small_graph():
    """5-node, 8-edge NetworkX graph for testing."""
    G = nx.Graph()
    G.add_node("single-cell RNA sequencing", entity_type="Concept", name="single-cell RNA sequencing")
    G.add_node("foundation model", entity_type="Concept", name="foundation model")
    G.add_node("ProtoCORAL", entity_type="Method", name="ProtoCORAL")
    G.add_node("cell type annotation", entity_type="Task", name="cell type annotation")
    G.add_node("PBMC dataset", entity_type="Dataset", name="PBMC dataset")

    G.add_edge("ProtoCORAL", "single-cell RNA sequencing", relation_type="studies_topic")
    G.add_edge("ProtoCORAL", "foundation model", relation_type="studies_topic")
    G.add_edge("ProtoCORAL", "cell type annotation", relation_type="uses_method")
    G.add_edge("ProtoCORAL", "PBMC dataset", relation_type="evaluated_on")
    G.add_edge("foundation model", "single-cell RNA sequencing", relation_type="part_of")
    G.add_edge("foundation model", "cell type annotation", relation_type="related_to")
    G.add_edge("single-cell RNA sequencing", "cell type annotation", relation_type="related_to")
    G.add_edge("cell type annotation", "PBMC dataset", relation_type="related_to")
    return G


@pytest.fixture
def sample_schema():
    """Test schema dict matching config/schema.yaml structure."""
    return {
        "node_types": {
            "Paper": {"required": ["title", "year", "source"], "optional": ["doi", "arxiv_id"]},
            "Concept": {"required": ["name"], "optional": ["aliases"]},
            "Method": {"required": ["name"], "optional": ["category"]},
            "Dataset": {"required": ["name"]},
            "Finding": {"required": ["description", "paper_id"]},
            "Task": {"required": ["name"]},
        },
        "relation_types": {
            "uses_method": {"from": "Paper", "to": "Method"},
            "studies_topic": {"from": "Paper", "to": "Concept"},
            "proposes": {"from": "Paper", "to": "Method"},
            "extends": {"from": "Paper", "to": "Paper"},
            "contradicts": {"from": "Finding", "to": "Finding"},
            "evaluated_on": {"from": "Method", "to": "Dataset"},
            "part_of": {"from": "Concept", "to": "Concept"},
        },
        "aliases": {
            "scRNA-seq": "single-cell RNA sequencing",
            "scRNAseq": "single-cell RNA sequencing",
            "sc-RNA-seq": "single-cell RNA sequencing",
            "VAE": "variational autoencoder",
            "GAN": "generative adversarial network",
        },
    }
