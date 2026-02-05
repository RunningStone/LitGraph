"""Tests for CLI argument parsing and mode routing (partial — Phase 1)."""

from __future__ import annotations

import pytest

from litgraph.settings import reset_settings


@pytest.fixture(autouse=True)
def _clean_settings():
    reset_settings()
    yield
    reset_settings()


class TestModeRouting:
    def test_lite_mode_best_equals_cheap(self, monkeypatch):
        """In Lite mode, best_model and cheap_model should be the same."""
        monkeypatch.setenv("LITGRAPH_MODE", "lite")
        monkeypatch.setenv("LITGRAPH_OLLAMA_MODEL", "qwen2.5:7b")

        from litgraph.settings import get_settings
        settings = get_settings(force_reload=True)
        assert settings.llm.best_model == settings.llm.cheap_model

    def test_pro_mode_different_models(self, monkeypatch):
        """In Pro mode, best_model and cheap_model should differ."""
        monkeypatch.setenv("LITGRAPH_MODE", "pro")
        monkeypatch.setenv("LITGRAPH_PRO_BEST_MODEL", "claude-sonnet-4")
        monkeypatch.setenv("LITGRAPH_PRO_CHEAP_MODEL", "claude-haiku-4")

        from litgraph.settings import get_settings
        settings = get_settings(force_reload=True)
        assert settings.llm.best_model != settings.llm.cheap_model
