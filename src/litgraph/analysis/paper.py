"""Single paper analysis: PDF download → extract → LLM analysis → Markdown output."""

from __future__ import annotations

import logging
import re
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

from litgraph.llm.client import complete
from litgraph.llm.prompts import (
    format_questions_block,
    get_questions_version,
    load_prompt,
    load_questions,
)
from litgraph.retry import with_retry
from litgraph.settings import get_settings

logger = logging.getLogger(__name__)


def analyze_paper(paper: dict, data_dir: Path) -> Path | None:
    """Full analysis flow for a single paper.

    1. Check existing analysis → skip if questions_version matches.
    2. Download PDF to tempfile → extract text → delete PDF.
    3. Render prompt → LLM → write Markdown with YAML front matter.

    Returns:
        Path to the analysis Markdown file, or None on failure.
    """
    settings = get_settings()
    paper_id = paper.get("paper_id") or paper.get("dedup_key", "unknown")
    safe_id = paper_id.replace(":", "_").replace("/", "_")

    analysis_dir = data_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    md_path = analysis_dir / f"{safe_id}.md"

    # Check existing analysis
    current_version = get_questions_version()
    if md_path.exists():
        existing_version = _extract_questions_version(md_path)
        if existing_version == current_version:
            logger.info("Skipping %s — already analyzed with questions v%d", paper_id, current_version)
            return md_path
        else:
            # Rename old analysis
            old_name = f"{safe_id}.v{existing_version}.md"
            old_path = analysis_dir / old_name
            md_path.rename(old_path)
            logger.info("Renamed old analysis to %s (v%s → v%d)", old_name, existing_version, current_version)

    # Try to get full text from PDF
    full_text = None
    source_type = "abstract_only"
    pdf_url = paper.get("pdf_url")
    if pdf_url:
        full_text = _download_and_extract_pdf(pdf_url)
        if full_text:
            source_type = "full_text"

    # Build prompt
    questions = load_questions()
    questions_block = format_questions_block(questions)

    system_prompt, user_prompt = load_prompt(
        "paper_analysis",
        title=paper.get("title", "Unknown"),
        authors=", ".join(paper.get("authors", [])) if isinstance(paper.get("authors"), list) else paper.get("authors", ""),
        year=paper.get("year", "Unknown"),
        source=paper.get("source", "Unknown"),
        abstract=paper.get("abstract", ""),
        full_text=full_text,
        questions_block=questions_block,
    )

    # Call LLM
    try:
        response = complete(user_prompt, system_prompt=system_prompt, model="best")
    except Exception as e:
        logger.error("LLM analysis failed for %s: %s", paper_id, e)
        return None

    # Write Markdown with YAML front matter
    front_matter = {
        "paper_id": paper_id,
        "title": paper.get("title", ""),
        "authors": paper.get("authors", []),
        "year": paper.get("year"),
        "source": paper.get("source", ""),
        "doi": paper.get("doi"),
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "analyzed_by": settings.llm.best_model,
        "mode": settings.mode,
        "source_type": source_type,
        "questions_version": current_version,
    }

    content = "---\n"
    content += yaml.dump(front_matter, default_flow_style=False, allow_unicode=True)
    content += "---\n\n"
    content += f"# {paper.get('title', 'Unknown')}\n\n"
    content += response

    md_path.write_text(content, encoding="utf-8")
    logger.info("Wrote analysis: %s (%s)", md_path.name, source_type)
    return md_path


def extract_pdf_text(pdf_path: str | Path, max_pages: int = 50) -> str:
    """Extract text from a PDF using pymupdf."""
    import pymupdf

    doc = pymupdf.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


def cleanup_pdf_text(text: str) -> str:
    """Clean up extracted PDF text.

    - Strip page numbers (standalone digits on a line).
    - Merge hyphenated line breaks (e.g. 'founda-\\ntion' → 'foundation').
    """
    # Merge hyphenated breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Remove standalone page numbers
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@with_retry
def _download_and_extract_pdf(pdf_url: str) -> str | None:
    """Download PDF to temp file, extract text, delete PDF."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
            urllib.request.urlretrieve(pdf_url, tmp_path)

        text = extract_pdf_text(tmp_path)
        text = cleanup_pdf_text(text)

        # Delete temp PDF
        Path(tmp_path).unlink(missing_ok=True)

        if len(text.strip()) < 100:
            logger.warning("PDF text too short (%d chars), likely extraction failure", len(text))
            return None

        return text
    except Exception as e:
        logger.warning("PDF download/extraction failed for %s: %s", pdf_url, e)
        # Clean up temp file if it exists
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except (NameError, OSError):
            pass
        return None


def _extract_questions_version(md_path: Path) -> int | None:
    """Extract questions_version from YAML front matter of an analysis Markdown."""
    try:
        content = md_path.read_text(encoding="utf-8")
        if not content.startswith("---"):
            return None
        end = content.index("---", 3)
        front_matter = yaml.safe_load(content[3:end])
        return front_matter.get("questions_version")
    except Exception:
        return None
