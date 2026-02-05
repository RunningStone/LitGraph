"""Direct NetworkX graph operations: load, expand keywords, subgraph, stats."""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)


def load_graph(graphml_path: Path | str) -> nx.Graph:
    """Load a NetworkX graph from a GraphML file.

    Returns an empty graph if the file doesn't exist.
    """
    path = Path(graphml_path)
    if not path.exists():
        logger.info("No graph file at %s, returning empty graph", path)
        return nx.Graph()
    return nx.read_graphml(str(path))


def expand_keywords(
    graph: nx.Graph,
    seeds: list[str],
    max_hops: int = 2,
    max_results: int = 20,
) -> list[str]:
    """BFS expand from seed nodes to find related keywords.

    Only considers Concept, Method, and Task nodes.

    Args:
        graph: NetworkX graph.
        seeds: Seed node names.
        max_hops: Maximum BFS depth.
        max_results: Maximum number of results.

    Returns:
        List of expanded keyword strings.
    """
    expandable_types = {"Concept", "Method", "Task"}
    results = set()
    visited = set()

    queue: deque[tuple[str, int]] = deque()

    for seed in seeds:
        # Find matching nodes (case-insensitive)
        seed_lower = seed.lower()
        for node in graph.nodes:
            if str(node).lower() == seed_lower:
                queue.append((node, 0))
                visited.add(node)
                break

    while queue and len(results) < max_results:
        node, depth = queue.popleft()

        # Add to results if it's an expandable type
        node_data = graph.nodes.get(node, {})
        node_type = node_data.get("entity_type", "")
        if node_type in expandable_types:
            node_name = node_data.get("name", str(node))
            results.add(node_name)

        if depth < max_hops:
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

    return list(results)[:max_results]


def get_subgraph(
    graph: nx.Graph,
    center_node: str,
    max_hops: int = 2,
) -> nx.Graph:
    """Extract a subgraph centered on a node within max_hops.

    Args:
        graph: Source graph.
        center_node: Center node name.
        max_hops: Maximum distance from center.

    Returns:
        Subgraph as a new nx.Graph.
    """
    if center_node not in graph:
        return nx.Graph()

    nodes = set()
    visited = set()
    queue: deque[tuple[str, int]] = deque([(center_node, 0)])
    visited.add(center_node)

    while queue:
        node, depth = queue.popleft()
        nodes.add(node)

        if depth < max_hops:
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

    return graph.subgraph(nodes).copy()


def get_stats(graph: nx.Graph) -> dict:
    """Get basic statistics about the graph.

    Returns:
        Dict with node_count, edge_count, node_types, relation_types.
    """
    if graph.number_of_nodes() == 0:
        return {
            "node_count": 0,
            "edge_count": 0,
            "node_types": {},
            "relation_types": {},
        }

    node_types: dict[str, int] = {}
    for _, data in graph.nodes(data=True):
        nt = data.get("entity_type", "unknown")
        node_types[nt] = node_types.get(nt, 0) + 1

    relation_types: dict[str, int] = {}
    for _, _, data in graph.edges(data=True):
        rt = data.get("relation_type", "unknown")
        relation_types[rt] = relation_types.get(rt, 0) + 1

    return {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "node_types": node_types,
        "relation_types": relation_types,
    }
