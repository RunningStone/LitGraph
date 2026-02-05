"""Innovation analysis: identify research gaps and novel opportunities."""

from __future__ import annotations

import logging
from pathlib import Path

from litgraph.llm.client import complete
from litgraph.llm.prompts import load_prompt
from litgraph.settings import get_settings

logger = logging.getLogger(__name__)


def identify_innovations(scope: str = "all", data_dir: Path | None = None) -> str:
    """Identify innovations by combining KG context with paper analyses.

    Args:
        scope: "all" for all analyses, "last-run" for most recent run.
        data_dir: DATA directory path.

    Returns:
        Innovation report as Markdown string.
    """
    if data_dir is None:
        data_dir = get_settings().data_dir

    # Gather paper analysis summaries
    papers_summary = _gather_analyses(scope, data_dir)

    # Gather KG context
    kg_context = _gather_kg_context(data_dir)

    # Render innovation prompt
    system_prompt, user_prompt = load_prompt(
        "innovation",
        kg_context=kg_context,
        papers_summary=papers_summary,
        scope=scope,
    )

    # Call LLM
    response = complete(user_prompt, system_prompt=system_prompt, model="best")
    return response


def _gather_analyses(scope: str, data_dir: Path) -> str:
    """Read analysis Markdown files and create a summary block."""
    analysis_dir = data_dir / "analysis"
    if not analysis_dir.exists():
        return "(No paper analyses available)"

    md_files = sorted(analysis_dir.glob("*.md"))
    # Exclude versioned backups
    md_files = [f for f in md_files if not f.stem.endswith(tuple(f".v{i}" for i in range(100)))]

    if scope == "last-run":
        # Take the most recently modified files (up to 20)
        md_files = sorted(md_files, key=lambda f: f.stat().st_mtime, reverse=True)[:20]

    if not md_files:
        return "(No paper analyses available)"

    summaries = []
    for md in md_files[:50]:  # Cap at 50 to avoid token overflow
        try:
            content = md.read_text(encoding="utf-8")
            # Take first 500 chars as summary
            summary = content[:500]
            if len(content) > 500:
                summary += "..."
            summaries.append(f"### {md.stem}\n{summary}")
        except Exception as e:
            logger.warning("Failed to read %s: %s", md, e)

    return "\n\n".join(summaries) if summaries else "(No paper analyses available)"


def _gather_kg_context(data_dir: Path) -> str:
    """Get KG stats and a global context summary."""
    try:
        from litgraph.kg.direct import get_stats, load_graph

        # Try to find the graph file in kg_store
        kg_dir = data_dir / "kg_store"
        graphml_candidates = list(kg_dir.glob("*.graphml")) if kg_dir.exists() else []

        if not graphml_candidates:
            return "(Knowledge graph is empty or not built yet)"

        graph = load_graph(graphml_candidates[0])
        stats = get_stats(graph)

        lines = [
            f"- Nodes: {stats['node_count']}",
            f"- Edges: {stats['edge_count']}",
            "- Node types: " + ", ".join(f"{k}: {v}" for k, v in stats["node_types"].items()),
            "- Relation types: " + ", ".join(f"{k}: {v}" for k, v in stats["relation_types"].items()),
        ]
        return "\n".join(lines)
    except Exception as e:
        logger.warning("Failed to gather KG context: %s", e)
        return "(Knowledge graph context unavailable)"
