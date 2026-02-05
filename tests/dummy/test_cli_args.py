"""Tests for CLI argument parsing, mode routing, and subcommand structure."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from litgraph.cli import main
from litgraph.settings import reset_settings


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def runner():
    return CliRunner()


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
    for name in ["paper_analysis", "relevance_filter", "entity_extraction", "innovation", "keyword_expansion"]:
        (prompts_dir / f"{name}.yaml").write_text(
            f"version: 1\nsystem: Test.\ntemplate: |\n  test\n"
        )
    return tmp_path


class TestModeRouting:
    def test_lite_mode_best_equals_cheap(self, monkeypatch):
        monkeypatch.setenv("LITGRAPH_MODE", "lite")
        monkeypatch.setenv("LITGRAPH_OLLAMA_MODEL", "qwen2.5:7b")

        from litgraph.settings import get_settings
        settings = get_settings(force_reload=True)
        assert settings.llm.best_model == settings.llm.cheap_model

    def test_pro_mode_different_models(self, monkeypatch):
        monkeypatch.setenv("LITGRAPH_MODE", "pro")
        monkeypatch.setenv("LITGRAPH_PRO_BEST_MODEL", "claude-sonnet-4")
        monkeypatch.setenv("LITGRAPH_PRO_CHEAP_MODEL", "claude-haiku-4")

        from litgraph.settings import get_settings
        settings = get_settings(force_reload=True)
        assert settings.llm.best_model != settings.llm.cheap_model


class TestCLISubcommands:
    def test_main_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "LitGraph" in result.output

    def test_search_help(self, runner):
        result = runner.invoke(main, ["search", "--help"])
        assert result.exit_code == 0
        assert "--keywords" in result.output

    def test_filter_help(self, runner):
        result = runner.invoke(main, ["filter", "--help"])
        assert result.exit_code == 0
        assert "--min-citations" in result.output

    def test_analyze_help(self, runner):
        result = runner.invoke(main, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "--paper-ids" in result.output
        assert "--all-pending" in result.output

    def test_kg_help(self, runner):
        result = runner.invoke(main, ["kg", "--help"])
        assert result.exit_code == 0
        assert "update" in result.output
        assert "query" in result.output
        assert "expand" in result.output
        assert "stats" in result.output

    def test_innovate_help(self, runner):
        result = runner.invoke(main, ["innovate", "--help"])
        assert result.exit_code == 0
        assert "--scope" in result.output

    def test_run_help(self, runner):
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--keywords" in result.output
        assert "--resume" in result.output

    def test_config_show_help(self, runner):
        result = runner.invoke(main, ["config", "show", "--help"])
        assert result.exit_code == 0

    def test_config_validate_help(self, runner):
        result = runner.invoke(main, ["config", "validate", "--help"])
        assert result.exit_code == 0


class TestCLIConfigShow:
    def test_config_show(self, runner, monkeypatch):
        monkeypatch.setenv("LITGRAPH_MODE", "pro")
        result = runner.invoke(main, ["config", "show"])
        assert result.exit_code == 0
        assert "Mode: pro" in result.output
        assert "Best model:" in result.output


class TestCLILiteWarning:
    def test_lite_mode_warning(self, monkeypatch):
        monkeypatch.setenv("LITGRAPH_MODE", "lite")
        monkeypatch.setenv("LITGRAPH_OLLAMA_MODEL", "qwen2.5:7b")
        r = CliRunner(mix_stderr=False)
        result = r.invoke(main, ["config", "show"])
        # Warning goes to stderr via click.secho(err=True)
        assert "WARNING" in result.stderr
        assert result.exit_code == 0

    def test_mode_override_via_flag(self, runner, monkeypatch):
        monkeypatch.setenv("LITGRAPH_MODE", "pro")
        result = runner.invoke(main, ["--mode", "lite", "config", "show"])
        assert result.exit_code == 0
        assert "Mode: lite" in result.output
