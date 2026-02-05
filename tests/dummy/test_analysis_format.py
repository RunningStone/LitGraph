"""Tests for analysis Markdown output format, front matter, skip logic, version handling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from litgraph.analysis.paper import _extract_questions_version, analyze_paper
from litgraph.settings import reset_settings


@pytest.fixture(autouse=True)
def _clean_settings(monkeypatch):
    reset_settings()
    # Ensure settings can load without real .env
    monkeypatch.setenv("LITGRAPH_MODE", "pro")
    yield
    reset_settings()


@pytest.fixture
def paper():
    return {
        "paper_id": "arxiv:2401.12345",
        "title": "Test Paper Title",
        "authors": ["Alice", "Bob"],
        "year": 2024,
        "source": "arxiv",
        "doi": "10.1234/test",
        "abstract": "This is a test abstract.",
        "pdf_url": None,
    }


@pytest.fixture
def project_root(tmp_path):
    """Create minimal config for settings."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    prompts_dir = config_dir / "prompts"
    prompts_dir.mkdir()

    (config_dir / "questions.yaml").write_text(
        "version: 1\nquestions:\n"
        "  - id: problem\n    text: What problem?\n"
        "  - id: method\n    text: What method?\n"
    )
    (config_dir / "schema.yaml").write_text("node_types: {}\n")
    (config_dir / "config.default.yaml").write_text("{}\n")
    (prompts_dir / "paper_analysis.yaml").write_text(
        "version: 1\n"
        "system: You are a test analyst.\n"
        "template: |\n"
        "  Title: {{ title }}\n"
        "  Abstract: {{ abstract }}\n"
        "  {{ questions_block }}\n"
    )
    return tmp_path


class TestAnalyzePaper:
    @patch("litgraph.analysis.paper.complete")
    def test_creates_markdown(self, mock_complete, paper, data_dir, project_root, monkeypatch):
        mock_complete.return_value = "## What problem?\n\nThe problem is X.\n"

        from litgraph.settings import get_settings
        get_settings(project_root=project_root, force_reload=True)

        result = analyze_paper(paper, data_dir)
        assert result is not None
        assert result.exists()
        assert result.suffix == ".md"

    @patch("litgraph.analysis.paper.complete")
    def test_front_matter_fields(self, mock_complete, paper, data_dir, project_root):
        mock_complete.return_value = "## Answer\n\nSome answer.\n"

        from litgraph.settings import get_settings
        get_settings(project_root=project_root, force_reload=True)

        result = analyze_paper(paper, data_dir)
        content = result.read_text()

        assert content.startswith("---\n")
        end = content.index("---", 3)
        front_matter = yaml.safe_load(content[3:end])

        assert front_matter["paper_id"] == "arxiv:2401.12345"
        assert front_matter["title"] == "Test Paper Title"
        assert front_matter["source_type"] == "abstract_only"
        assert front_matter["questions_version"] == 1
        assert "analyzed_at" in front_matter
        assert "analyzed_by" in front_matter
        assert "mode" in front_matter

    @patch("litgraph.analysis.paper.complete")
    def test_skip_existing(self, mock_complete, paper, data_dir, project_root):
        """Already analyzed paper with matching version should be skipped."""
        mock_complete.return_value = "## Answer\nSome answer.\n"

        from litgraph.settings import get_settings
        get_settings(project_root=project_root, force_reload=True)

        # Analyze once
        result1 = analyze_paper(paper, data_dir)
        # Analyze again — should skip
        result2 = analyze_paper(paper, data_dir)
        assert result2 == result1
        # LLM should only be called once
        assert mock_complete.call_count == 1

    @patch("litgraph.analysis.paper.complete")
    def test_version_mismatch_renames(self, mock_complete, paper, data_dir, project_root):
        """Version mismatch should rename old file and re-analyze."""
        mock_complete.return_value = "## Answer\nSome answer.\n"

        from litgraph.settings import get_settings
        get_settings(project_root=project_root, force_reload=True)

        # Create analysis with version 1
        result = analyze_paper(paper, data_dir)
        assert result.exists()

        # Update questions.yaml to version 2
        q_path = project_root / "config" / "questions.yaml"
        q_path.write_text(
            "version: 2\nquestions:\n"
            "  - id: problem\n    text: What problem?\n"
            "  - id: method\n    text: What method?\n"
        )

        # Re-analyze — should rename old and create new
        result2 = analyze_paper(paper, data_dir)
        assert result2.exists()
        assert mock_complete.call_count == 2

        # Check old file was renamed
        old_files = list(data_dir.glob("analysis/*.v1.md"))
        assert len(old_files) == 1

    @patch("litgraph.analysis.paper.complete")
    def test_source_type_abstract_only(self, mock_complete, paper, data_dir, project_root):
        """Paper without pdf_url should be abstract_only."""
        mock_complete.return_value = "## Answer\nAbstract analysis.\n"
        paper["pdf_url"] = None

        from litgraph.settings import get_settings
        get_settings(project_root=project_root, force_reload=True)

        result = analyze_paper(paper, data_dir)
        content = result.read_text()
        end = content.index("---", 3)
        fm = yaml.safe_load(content[3:end])
        assert fm["source_type"] == "abstract_only"


class TestExtractQuestionsVersion:
    def test_extracts_version(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("---\nquestions_version: 2\ntitle: Test\n---\n\n# Content\n")
        assert _extract_questions_version(md) == 2

    def test_missing_version(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("---\ntitle: Test\n---\n\n# Content\n")
        assert _extract_questions_version(md) is None

    def test_no_front_matter(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("# Just content\nNo front matter.\n")
        assert _extract_questions_version(md) is None
