"""Arxiv paper search via paperscraper."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def search_arxiv(keywords: list[str], max_results: int = 50) -> list[dict]:
    """Search arxiv via paperscraper and return normalized paper dicts.

    Args:
        keywords: List of search keywords.
        max_results: Maximum results per keyword.

    Returns:
        List of paper dicts with standardized fields.
    """
    from paperscraper.arxiv import get_and_dump_arxiv_papers

    all_papers = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for kw in keywords:
            outfile = Path(tmpdir) / f"{kw.replace(' ', '_')}.jsonl"
            try:
                get_and_dump_arxiv_papers(kw, output_filepath=str(outfile))
            except Exception as e:
                logger.warning("Arxiv search failed for '%s': %s", kw, e)
                continue

            if outfile.exists():
                with open(outfile) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            raw = json.loads(line)
                            paper = _normalize_arxiv_paper(raw)
                            if paper:
                                all_papers.append(paper)
                        except json.JSONDecodeError:
                            continue

            if len(all_papers) >= max_results:
                break

    return all_papers[:max_results]


def _normalize_arxiv_paper(raw: dict) -> dict | None:
    """Normalize a paperscraper arxiv result to standard format."""
    title = raw.get("title", "").strip()
    if not title:
        return None

    # Extract arxiv_id from the paper URL or doi
    arxiv_id = None
    doi = raw.get("doi", "")
    if doi and "arxiv" in doi.lower():
        parts = doi.split("/")
        if parts:
            arxiv_id = parts[-1]
    # Try from paperscraper's externalIds or direct field
    if not arxiv_id:
        arxiv_id = raw.get("arxiv_id") or raw.get("paperId")

    return {
        "paper_id": f"arxiv:{arxiv_id}" if arxiv_id else None,
        "arxiv_id": arxiv_id,
        "doi": raw.get("doi"),
        "title": title,
        "authors": raw.get("authors", []),
        "year": raw.get("year") or raw.get("date", "")[:4] if raw.get("date") else None,
        "abstract": raw.get("abstract", ""),
        "source": "arxiv",
        "citations": raw.get("citationCount") or raw.get("citations", 0),
        "pdf_url": raw.get("url"),
    }
