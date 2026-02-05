"""Tests for direct NetworkX graph operations: expand keywords, subgraph, stats."""

from __future__ import annotations

import networkx as nx
import pytest

from litgraph.kg.direct import expand_keywords, get_stats, get_subgraph, load_graph


class TestExpandKeywords:
    def test_basic_expansion(self, small_graph):
        """Should find related concepts from seed."""
        results = expand_keywords(small_graph, ["ProtoCORAL"])
        assert len(results) > 0
        # ProtoCORAL is a Method, should be in results
        assert "ProtoCORAL" in results

    def test_bfs_neighbors(self, small_graph):
        """Should find neighbors of seed nodes."""
        results = expand_keywords(small_graph, ["ProtoCORAL"], max_hops=1)
        # Direct neighbors of ProtoCORAL include Concept and Task nodes
        assert "single-cell RNA sequencing" in results or "foundation model" in results or "cell type annotation" in results

    def test_max_hops_limit(self, small_graph):
        """max_hops=0 should only return the seed itself."""
        results = expand_keywords(small_graph, ["ProtoCORAL"], max_hops=0)
        assert results == ["ProtoCORAL"]

    def test_max_results(self, small_graph):
        """Should respect max_results."""
        results = expand_keywords(small_graph, ["ProtoCORAL"], max_hops=3, max_results=2)
        assert len(results) <= 2

    def test_nonexistent_seed(self, small_graph):
        """Non-existent seed should return empty."""
        results = expand_keywords(small_graph, ["nonexistent_node"])
        assert results == []

    def test_empty_graph(self):
        """Empty graph should return empty."""
        G = nx.Graph()
        results = expand_keywords(G, ["anything"])
        assert results == []

    def test_case_insensitive_seed(self, small_graph):
        """Seed matching should be case-insensitive."""
        results = expand_keywords(small_graph, ["protocoral"])
        assert len(results) > 0


class TestGetSubgraph:
    def test_basic_subgraph(self, small_graph):
        sub = get_subgraph(small_graph, "ProtoCORAL", max_hops=1)
        assert "ProtoCORAL" in sub.nodes
        assert sub.number_of_nodes() > 1

    def test_subgraph_depth(self, small_graph):
        sub0 = get_subgraph(small_graph, "ProtoCORAL", max_hops=0)
        sub1 = get_subgraph(small_graph, "ProtoCORAL", max_hops=1)
        assert sub0.number_of_nodes() <= sub1.number_of_nodes()

    def test_nonexistent_center(self, small_graph):
        sub = get_subgraph(small_graph, "nonexistent")
        assert sub.number_of_nodes() == 0

    def test_preserves_edges(self, small_graph):
        sub = get_subgraph(small_graph, "ProtoCORAL", max_hops=1)
        assert sub.number_of_edges() > 0


class TestGetStats:
    def test_basic_stats(self, small_graph):
        stats = get_stats(small_graph)
        assert stats["node_count"] == 5
        assert stats["edge_count"] == 8
        assert "Concept" in stats["node_types"]
        assert "Method" in stats["node_types"]
        assert stats["node_types"]["Concept"] == 2
        assert stats["node_types"]["Method"] == 1

    def test_relation_types(self, small_graph):
        stats = get_stats(small_graph)
        assert "studies_topic" in stats["relation_types"]
        assert "related_to" in stats["relation_types"]

    def test_empty_graph(self):
        G = nx.Graph()
        stats = get_stats(G)
        assert stats["node_count"] == 0
        assert stats["edge_count"] == 0
        assert stats["node_types"] == {}
        assert stats["relation_types"] == {}


class TestLoadGraph:
    def test_load_nonexistent(self, tmp_path):
        """Should return empty graph for non-existent file."""
        G = load_graph(tmp_path / "nonexistent.graphml")
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == 0

    def test_load_roundtrip(self, tmp_path, small_graph):
        """Save and reload should preserve graph."""
        path = tmp_path / "test.graphml"
        nx.write_graphml(small_graph, str(path))
        G = load_graph(path)
        assert G.number_of_nodes() == small_graph.number_of_nodes()
        assert G.number_of_edges() == small_graph.number_of_edges()
