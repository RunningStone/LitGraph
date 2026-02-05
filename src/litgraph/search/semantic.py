"""Semantic Scholar paper search."""

from __future__ import annotations

import logging

from litgraph.retry import RateLimiter

logger = logging.getLogger(__name__)

_rate_limiter: RateLimiter | None = None


def _get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def search_semantic_scholar(
    keywords: list[str],
    max_results: int = 50,
    year_from: int | None = None,
) -> list[dict]:
    """Search Semantic Scholar and return normalized paper dicts.

    Args:
        keywords: List of search keywords.
        max_results: Maximum total results.
        year_from: Only include papers from this year onwards.

    Returns:
        List of paper dicts with standardized fields.
    """
    from semanticscholar import SemanticScholar

    sch = SemanticScholar()
    limiter = _get_rate_limiter()
    all_papers = []

    for kw in keywords:
        limiter.acquire_sync()
        try:
            year_range = f"{year_from}-" if year_from else None
            results = sch.search_paper(
                kw,
                limit=min(max_results, 100),
                year=year_range,
                fields=[
                    "title", "abstract", "authors", "year",
                    "externalIds", "citationCount", "openAccessPdf",
                ],
            )
        except Exception as e:
            logger.warning("Semantic Scholar search failed for '%s': %s", kw, e)
            continue

        if results is None:
            continue

        for item in results:
            paper = _normalize_ss_paper(item)
            if paper:
                all_papers.append(paper)

        if len(all_papers) >= max_results:
            break

    return all_papers[:max_results]


def _normalize_ss_paper(item) -> dict | None:
    """Normalize a Semantic Scholar result to standard format."""
    title = getattr(item, "title", None) or ""
    if not title.strip():
        return None

    ext_ids = getattr(item, "externalIds", {}) or {}
    arxiv_id = ext_ids.get("ArXiv")
    doi = ext_ids.get("DOI")

    authors = []
    raw_authors = getattr(item, "authors", []) or []
    for a in raw_authors:
        name = getattr(a, "name", None) or (a.get("name") if isinstance(a, dict) else str(a))
        if name:
            authors.append(name)

    pdf_url = None
    oaPdf = getattr(item, "openAccessPdf", None)
    if oaPdf and isinstance(oaPdf, dict):
        pdf_url = oaPdf.get("url")

    paper_id = f"arxiv:{arxiv_id}" if arxiv_id else (f"doi:{doi}" if doi else None)

    return {
        "paper_id": paper_id,
        "arxiv_id": arxiv_id,
        "doi": doi,
        "title": title.strip(),
        "authors": authors,
        "year": getattr(item, "year", None),
        "abstract": getattr(item, "abstract", "") or "",
        "source": "semantic_scholar",
        "citations": getattr(item, "citationCount", 0) or 0,
        "pdf_url": pdf_url,
    }
