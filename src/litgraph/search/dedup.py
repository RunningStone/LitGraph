"""Paper deduplication: key generation, single-run dedup, cross-run index merge."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def _title_hash(title: str) -> str:
    """Normalize title → SHA256[:16] for dedup fallback.

    Steps: lowercase → strip punctuation → collapse whitespace → SHA256[:16].
    """
    t = title.lower()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]


def dedup_key(paper: dict) -> str:
    """Return the unique dedup key for a paper.

    Priority: arxiv_id > doi > title_hash.
    """
    arxiv_id = paper.get("arxiv_id")
    if arxiv_id:
        return f"arxiv:{arxiv_id}"

    doi = paper.get("doi")
    if doi:
        return f"doi:{doi}"

    title = paper.get("title", "")
    return f"title:{_title_hash(title)}"


def dedup_paper_list(papers: list[dict]) -> list[dict]:
    """Deduplicate a list of papers within a single run. Keeps first occurrence."""
    seen = set()
    result = []
    for p in papers:
        key = dedup_key(p)
        if key not in seen:
            seen.add(key)
            p["dedup_key"] = key
            result.append(p)
    return result


def merge_into_index(
    new_papers: list[dict], index_path: Path
) -> tuple[list[dict], list[dict]]:
    """Merge new papers into the persistent index.json.

    Returns (added, updated) lists.
    Updated papers get their meta fields refreshed (citations, doi, pdf_url).
    """
    # Load existing index
    existing = {}
    if index_path.exists():
        with open(index_path) as f:
            data = json.load(f)
            for entry in data:
                key = entry.get("dedup_key", dedup_key(entry))
                existing[key] = entry

    added = []
    updated = []

    for paper in new_papers:
        key = paper.get("dedup_key", dedup_key(paper))
        paper["dedup_key"] = key

        if key in existing:
            # Update meta fields that may have changed
            old = existing[key]
            changed = False
            for field in ("citations", "doi", "pdf_url"):
                new_val = paper.get(field)
                if new_val is not None and new_val != old.get(field):
                    old[field] = new_val
                    changed = True
            if changed:
                updated.append(old)
        else:
            existing[key] = paper
            added.append(paper)

    # Write back
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w") as f:
        json.dump(list(existing.values()), f, indent=2, ensure_ascii=False)

    return added, updated
