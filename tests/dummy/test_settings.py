"""Tests for settings loading, path resolution, mode switch, retry/rate_limit defaults."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from litgraph.settings import (
    LLMConfig,
    RateLimitConfig,
    RetryConfig,
    Settings,
    _resolve_data_dir,
    get_settings,
    reset_settings,
)


@pytest.fixture(autouse=True)
def _clean_settings():
    """Reset singleton before and after each test."""
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def project_root(tmp_path):
    """Create a minimal project structure for settings loading."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    prompts_dir = config_dir / "prompts"
    prompts_dir.mkdir()

    # config.default.yaml
    (config_dir / "config.default.yaml").write_text(
        "search:\n  max_results: 50\n"
    )
    # questions.yaml
    (config_dir / "questions.yaml").write_text(
        "version: 1\nquestions:\n  - id: test\n    text: test question\n"
    )
    # schema.yaml
    (config_dir / "schema.yaml").write_text(
        "node_types:\n  Paper:\n    required: [title]\n"
    )
    return tmp_path


class TestPathResolution:
    def test_relative_path(self, tmp_path):
        result = _resolve_data_dir("../DATA", tmp_path)
        assert result == (tmp_path / ".." / "DATA").resolve()

    def test_absolute_path(self, tmp_path):
        abs_path = "/tmp/litgraph_data"
        result = _resolve_data_dir(abs_path, tmp_path)
        assert result == Path(abs_path)

    def test_dot_relative(self, tmp_path):
        result = _resolve_data_dir("./mydata", tmp_path)
        assert result == (tmp_path / "mydata").resolve()


class TestEnvLoading:
    def test_loads_env_file(self, project_root, monkeypatch):
        # Write a .env file
        env_file = project_root / ".env"
        env_file.write_text("LITGRAPH_MODE=lite\nLITGRAPH_OLLAMA_MODEL=llama3.1:8b\n")

        # Clear any existing env vars that might interfere
        monkeypatch.delenv("LITGRAPH_MODE", raising=False)
        monkeypatch.delenv("LITGRAPH_OLLAMA_MODEL", raising=False)

        settings = get_settings(project_root=project_root)
        assert settings.mode == "lite"
        assert settings.llm.best_model == "llama3.1:8b"
        assert settings.llm.cheap_model == "llama3.1:8b"

    def test_env_var_override(self, project_root, monkeypatch):
        """Environment variables should work even without .env file."""
        monkeypatch.setenv("LITGRAPH_MODE", "pro")
        monkeypatch.setenv("LITGRAPH_PRO_BEST_MODEL", "claude-opus-4")

        settings = get_settings(project_root=project_root)
        assert settings.mode == "pro"
        assert settings.llm.best_model == "claude-opus-4"

    def test_defaults_without_env(self, project_root, monkeypatch):
        """Without .env or env vars, defaults should apply."""
        # Clear all LITGRAPH_ env vars
        for key in list(os.environ):
            if key.startswith("LITGRAPH_"):
                monkeypatch.delenv(key, raising=False)

        settings = get_settings(project_root=project_root)
        assert settings.mode == "pro"
        assert settings.llm.base_url == "http://localhost:3456/v1"
        assert settings.embedding_model == "all-MiniLM-L6-v2"


class TestModeSwitch:
    def test_pro_mode(self, project_root, monkeypatch):
        monkeypatch.setenv("LITGRAPH_MODE", "pro")
        monkeypatch.setenv("LITGRAPH_PRO_BEST_MODEL", "claude-sonnet-4")
        monkeypatch.setenv("LITGRAPH_PRO_CHEAP_MODEL", "claude-haiku-4")

        settings = get_settings(project_root=project_root)
        assert settings.mode == "pro"
        assert settings.llm.best_model == "claude-sonnet-4"
        assert settings.llm.cheap_model == "claude-haiku-4"
        assert settings.llm.best_model != settings.llm.cheap_model

    def test_lite_mode_same_model(self, project_root, monkeypatch):
        monkeypatch.setenv("LITGRAPH_MODE", "lite")
        monkeypatch.setenv("LITGRAPH_OLLAMA_MODEL", "qwen2.5:7b")

        settings = get_settings(project_root=project_root)
        assert settings.mode == "lite"
        assert settings.llm.best_model == settings.llm.cheap_model
        assert settings.llm.best_model == "qwen2.5:7b"
        assert "11434" in settings.llm.base_url

    def test_lite_mode_ollama_url(self, project_root, monkeypatch):
        monkeypatch.setenv("LITGRAPH_MODE", "lite")
        monkeypatch.setenv("LITGRAPH_OLLAMA_BASE_URL", "http://myhost:11434/v1")

        settings = get_settings(project_root=project_root)
        assert settings.llm.base_url == "http://myhost:11434/v1"


class TestRetryAndRateLimit:
    def test_default_retry(self, project_root, monkeypatch):
        for key in list(os.environ):
            if key.startswith("LITGRAPH_"):
                monkeypatch.delenv(key, raising=False)

        settings = get_settings(project_root=project_root)
        assert settings.retry.max_retries == 3
        assert settings.retry.backoff_base == 2.0

    def test_custom_retry(self, project_root, monkeypatch):
        monkeypatch.setenv("LITGRAPH_MAX_RETRIES", "5")
        monkeypatch.setenv("LITGRAPH_RETRY_BACKOFF_BASE", "3.0")

        settings = get_settings(project_root=project_root)
        assert settings.retry.max_retries == 5
        assert settings.retry.backoff_base == 3.0

    def test_default_rate_limit(self, project_root, monkeypatch):
        for key in list(os.environ):
            if key.startswith("LITGRAPH_"):
                monkeypatch.delenv(key, raising=False)

        settings = get_settings(project_root=project_root)
        assert settings.rate_limit.search_max_calls == 90
        assert settings.rate_limit.search_period == 300

    def test_custom_rate_limit(self, project_root, monkeypatch):
        monkeypatch.setenv("LITGRAPH_SEARCH_RATE_LIMIT", "50")
        monkeypatch.setenv("LITGRAPH_SEARCH_RATE_PERIOD", "60")

        settings = get_settings(project_root=project_root)
        assert settings.rate_limit.search_max_calls == 50
        assert settings.rate_limit.search_period == 60


class TestSingleton:
    def test_singleton_returns_same(self, project_root, monkeypatch):
        monkeypatch.setenv("LITGRAPH_MODE", "pro")
        s1 = get_settings(project_root=project_root)
        s2 = get_settings()
        assert s1 is s2

    def test_force_reload(self, project_root, monkeypatch):
        monkeypatch.setenv("LITGRAPH_MODE", "pro")
        s1 = get_settings(project_root=project_root)

        monkeypatch.setenv("LITGRAPH_MODE", "lite")
        s2 = get_settings(project_root=project_root, force_reload=True)
        assert s2.mode == "lite"
        assert s1 is not s2


class TestConfigPaths:
    def test_prompts_dir(self, project_root, monkeypatch):
        for key in list(os.environ):
            if key.startswith("LITGRAPH_"):
                monkeypatch.delenv(key, raising=False)
        settings = get_settings(project_root=project_root)
        assert settings.prompts_dir == project_root / "config" / "prompts"
        assert settings.questions_path == project_root / "config" / "questions.yaml"
        assert settings.schema_path == project_root / "config" / "schema.yaml"
