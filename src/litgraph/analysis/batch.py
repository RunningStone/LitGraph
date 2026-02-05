"""Batch paper analysis — sequential loop with progress bar."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from tqdm import tqdm

from litgraph.analysis.paper import analyze_paper

logger = logging.getLogger(__name__)


def analyze_batch(
    paper_ids: list[str] | None = None,
    all_pending: bool = False,
    data_dir: Path = None,
) -> dict:
    """Analyze multiple papers sequentially.

    Args:
        paper_ids: Specific paper IDs to analyze.
        all_pending: If True, analyze all papers in index that lack analysis.
        data_dir: DATA directory path.

    Returns:
        Dict with {analyzed, skipped, failed, errors}.
    """
    from litgraph.settings import get_settings
    if data_dir is None:
        data_dir = get_settings().data_dir

    papers = _resolve_papers(paper_ids, all_pending, data_dir)

    stats = {"analyzed": 0, "skipped": 0, "failed": 0, "errors": []}

    for paper in tqdm(papers, desc="Analyzing papers"):
        try:
            result = analyze_paper(paper, data_dir)
            if result is not None:
                stats["analyzed"] += 1
            else:
                stats["failed"] += 1
        except Exception as e:
            pid = paper.get("paper_id", "unknown")
            logger.error("Analysis failed for %s: %s", pid, e)
            stats["failed"] += 1
            stats["errors"].append({"paper_id": pid, "error": str(e)})

    return stats


def _resolve_papers(
    paper_ids: list[str] | None,
    all_pending: bool,
    data_dir: Path,
) -> list[dict]:
    """Load papers from index.json based on selection criteria."""
    index_path = data_dir / "papers" / "index.json"
    if not index_path.exists():
        logger.warning("No index.json found at %s", index_path)
        return []

    with open(index_path) as f:
        all_papers = json.load(f)

    if paper_ids:
        return [p for p in all_papers if p.get("paper_id") in paper_ids or p.get("dedup_key") in paper_ids]

    if all_pending:
        # Return papers that don't have analysis files yet
        analysis_dir = data_dir / "analysis"
        pending = []
        for p in all_papers:
            pid = p.get("paper_id") or p.get("dedup_key", "unknown")
            safe_id = pid.replace(":", "_").replace("/", "_")
            md_path = analysis_dir / f"{safe_id}.md"
            if not md_path.exists():
                pending.append(p)
        return pending

    return all_papers
