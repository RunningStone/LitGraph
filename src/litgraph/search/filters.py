"""Paper filtering: citation threshold + optional LLM relevance scoring."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def filter_papers(
    papers: list[dict],
    keywords: list[str] | None = None,
    min_citations: int = 0,
    use_llm_filter: bool = False,
) -> list[dict]:
    """Filter papers by citation count and optionally LLM relevance.

    Args:
        papers: List of paper dicts.
        keywords: Original search keywords (for LLM relevance context).
        min_citations: Minimum citation threshold.
        use_llm_filter: Whether to use LLM for relevance scoring.

    Returns:
        Filtered list of papers.
    """
    result = []

    for paper in papers:
        citations = paper.get("citations", 0) or 0
        if citations < min_citations:
            continue

        if use_llm_filter and keywords:
            if not _llm_relevance_check(paper, keywords):
                continue

        paper["relevant"] = True
        result.append(paper)

    logger.info("Filtered %d → %d papers (min_citations=%d, llm=%s)",
                len(papers), len(result), min_citations, use_llm_filter)
    return result


def _llm_relevance_check(paper: dict, keywords: list[str]) -> bool:
    """Use LLM to check if a paper is relevant to the search keywords."""
    from litgraph.llm.client import complete
    from litgraph.llm.prompts import load_prompt

    topic = ", ".join(keywords)
    try:
        system, user = load_prompt(
            "relevance_filter",
            topic=topic,
            title=paper.get("title", ""),
            abstract=paper.get("abstract", ""),
        )
        response = complete(user, system_prompt=system, model="cheap")
        data = json.loads(response)
        return data.get("relevant", True)
    except Exception as e:
        logger.warning("LLM relevance check failed for '%s': %s", paper.get("title"), e)
        return True  # Default to keeping the paper on failure
