"""Live tests for LLM client — requires running proxy or Ollama."""

from __future__ import annotations

import pytest

from litgraph.llm.client import complete, reset_client
from litgraph.settings import get_settings, reset_settings


@pytest.fixture(autouse=True)
def _clean():
    reset_settings()
    reset_client()
    yield
    reset_settings()
    reset_client()


@pytest.mark.live
def test_pro_mode_complete(monkeypatch):
    """Pro mode should return a non-empty string."""
    monkeypatch.setenv("LITGRAPH_MODE", "pro")
    get_settings(force_reload=True)

    result = complete("Say hello in one word.", model="best")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.live
def test_lite_mode_complete(monkeypatch):
    """Lite mode should return a non-empty string."""
    monkeypatch.setenv("LITGRAPH_MODE", "lite")
    get_settings(force_reload=True)

    result = complete("Say hello in one word.", model="best")
    assert isinstance(result, str)
    assert len(result) > 0
