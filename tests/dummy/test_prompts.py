"""Tests for Jinja2 YAML prompt loading and rendering."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from jinja2 import UndefinedError

from litgraph.llm.prompts import (
    format_questions_block,
    get_questions_version,
    load_prompt_from_path,
    load_questions,
)
from litgraph.settings import reset_settings


@pytest.fixture(autouse=True)
def _clean_settings():
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def prompt_yaml(tmp_path):
    """Create a test prompt YAML file."""
    data = {
        "version": 1,
        "system": "You are a test assistant.",
        "template": "Title: {{ title }}\nAbstract: {{ abstract }}\n{% if full_text is defined and full_text %}Full: {{ full_text }}{% endif %}",
    }
    path = tmp_path / "test_prompt.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


@pytest.fixture
def questions_yaml(tmp_path):
    """Create a test questions YAML file."""
    data = {
        "version": 1,
        "questions": [
            {"id": "problem", "text": "What problem does this paper address?"},
            {"id": "method", "text": "What is the core innovation?"},
            {"id": "data", "text": "What datasets are used?"},
            {"id": "results", "text": "Key experimental results?"},
            {"id": "limitations", "text": "Limitations and future directions?"},
            {"id": "connection", "text": "Connections to single-cell foundation models?"},
        ],
    }
    path = tmp_path / "questions.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


class TestLoadPrompt:
    def test_basic_render(self, prompt_yaml):
        system, user = load_prompt_from_path(
            prompt_yaml,
            title="Test Paper",
            abstract="This is an abstract.",
        )
        assert system == "You are a test assistant."
        assert "Title: Test Paper" in user
        assert "Abstract: This is an abstract." in user

    def test_optional_full_text(self, prompt_yaml):
        """Optional full_text should render when provided."""
        _, user = load_prompt_from_path(
            prompt_yaml,
            title="Test",
            abstract="Abstract",
            full_text="The full paper text here.",
        )
        assert "Full: The full paper text here." in user

    def test_optional_full_text_absent(self, prompt_yaml):
        """Without full_text, the conditional block should be omitted."""
        _, user = load_prompt_from_path(
            prompt_yaml,
            title="Test",
            abstract="Abstract",
        )
        assert "Full:" not in user

    def test_missing_required_var_raises(self, prompt_yaml):
        """Missing required template variables should raise UndefinedError."""
        with pytest.raises(UndefinedError):
            load_prompt_from_path(prompt_yaml, title="Test")
            # 'abstract' is required but missing

    def test_curly_braces_in_content_safe(self, prompt_yaml):
        """Content with single curly braces (math, code) should not break Jinja2."""
        _, user = load_prompt_from_path(
            prompt_yaml,
            title="Math Paper",
            abstract="Given set S = {x | x > 0}, the function f(x) = max{a, b}",
        )
        assert "{x | x > 0}" in user
        assert "max{a, b}" in user


class TestLoadQuestions:
    def test_load_questions_count(self, questions_yaml):
        questions = load_questions(questions_path=questions_yaml)
        assert len(questions) == 6

    def test_load_questions_structure(self, questions_yaml):
        questions = load_questions(questions_path=questions_yaml)
        for q in questions:
            assert "id" in q
            assert "text" in q

    def test_load_questions_ids(self, questions_yaml):
        questions = load_questions(questions_path=questions_yaml)
        ids = [q["id"] for q in questions]
        assert "problem" in ids
        assert "method" in ids
        assert "connection" in ids


class TestGetQuestionsVersion:
    def test_version(self, questions_yaml):
        assert get_questions_version(questions_path=questions_yaml) == 1

    def test_default_version(self, tmp_path):
        """If version key is missing, default to 1."""
        path = tmp_path / "q.yaml"
        path.write_text("questions:\n  - id: test\n    text: test\n")
        assert get_questions_version(questions_path=path) == 1


class TestFormatQuestionsBlock:
    def test_format(self):
        questions = [
            {"id": "q1", "text": "First question?"},
            {"id": "q2", "text": "Second question?"},
        ]
        block = format_questions_block(questions)
        assert "1. First question?" in block
        assert "2. Second question?" in block
        assert block.count("\n") == 1  # Two lines, one newline
