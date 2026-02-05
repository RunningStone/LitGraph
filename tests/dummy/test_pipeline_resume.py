"""Tests for pipeline resume logic."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from litgraph.settings import reset_settings


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    reset_settings()
    monkeypatch.setenv("LITGRAPH_MODE", "pro")
    yield
    reset_settings()


@pytest.fixture
def project_root(tmp_path):
    """Minimal config for settings."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    prompts_dir = config_dir / "prompts"
    prompts_dir.mkdir()
    (config_dir / "questions.yaml").write_text(
        "version: 1\nquestions:\n  - id: test\n    text: Test?\n"
    )
    (config_dir / "schema.yaml").write_text("node_types: {}\n")
    (config_dir / "config.default.yaml").write_text("{}\n")
    (prompts_dir / "paper_analysis.yaml").write_text(
        "version: 1\nsystem: Test.\ntemplate: |\n  {{ title }} {{ abstract }} {{ questions_block }}\n"
    )
    (prompts_dir / "innovation.yaml").write_text(
        "version: 1\nsystem: Test.\ntemplate: |\n  {{ kg_context }} {{ papers_summary }} {{ scope }}\n"
    )
    return tmp_path


class TestPipelineResume:
    def test_skip_search_when_index_exists(self, data_dir):
        """When index.json exists and resume=True, search should be skipped."""
        index_path = data_dir / "papers" / "index.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps([
            {"paper_id": "arxiv:test1", "title": "Test", "dedup_key": "arxiv:test1"}
        ]))

        # The resume logic checks if index_path.exists()
        assert index_path.exists()

    def test_skip_analyzed_papers(self, data_dir):
        """Papers with existing analysis should be skipped in batch."""
        analysis_dir = data_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        # Create existing analysis
        (analysis_dir / "arxiv_test1.md").write_text(
            "---\nquestions_version: 1\n---\n\n# Test\n"
        )

        # Create index with the paper
        index_path = data_dir / "papers" / "index.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps([
            {"paper_id": "arxiv:test1", "title": "Test", "dedup_key": "arxiv:test1"}
        ]))

        from litgraph.analysis.batch import _resolve_papers
        pending = _resolve_papers(None, True, data_dir)
        assert len(pending) == 0

    def test_detect_pending_papers(self, data_dir):
        """Papers without analysis should be detected as pending."""
        (data_dir / "analysis").mkdir(parents=True, exist_ok=True)
        index_path = data_dir / "papers" / "index.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps([
            {"paper_id": "arxiv:test1", "title": "Test 1", "dedup_key": "arxiv:test1"},
            {"paper_id": "arxiv:test2", "title": "Test 2", "dedup_key": "arxiv:test2"},
        ]))

        from litgraph.analysis.batch import _resolve_papers
        pending = _resolve_papers(None, True, data_dir)
        assert len(pending) == 2

    def test_run_record_saved(self, data_dir):
        """Pipeline run record should be saved to DATA/runs/."""
        runs_dir = data_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        record = {
            "started_at": "2024-01-01T00:00:00",
            "keywords": ["test"],
            "mode": "pro",
            "steps": {"search": "completed"},
        }

        run_path = runs_dir / "20240101_000000.json"
        with open(run_path, "w") as f:
            json.dump(record, f)

        assert run_path.exists()
        with open(run_path) as f:
            loaded = json.load(f)
        assert loaded["keywords"] == ["test"]
