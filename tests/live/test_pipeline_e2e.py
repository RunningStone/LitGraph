"""Live end-to-end pipeline test — 2 papers in lite mode."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from litgraph.settings import reset_settings


@pytest.fixture(autouse=True)
def _clean():
    reset_settings()
    yield
    reset_settings()


@pytest.mark.live
def test_pipeline_e2e(monkeypatch, tmp_path):
    """Run full pipeline with 2 papers in lite mode."""
    monkeypatch.setenv("LITGRAPH_MODE", "lite")
    monkeypatch.setenv("LITGRAPH_DATA_DIR", str(tmp_path))

    from litgraph.cli import main

    runner = CliRunner()
    result = runner.invoke(main, [
        "run",
        "--keywords", "transformer",
        "--max-results", "2",
        "--min-citations", "0",
    ])

    assert result.exit_code == 0 or result.exit_code == 1  # Partial failure OK
    assert "Run record:" in result.output
