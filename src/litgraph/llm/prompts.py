"""Jinja2 prompt loader — loads YAML templates and renders them. Contains zero prompt text."""

from __future__ import annotations

from pathlib import Path

import yaml
from jinja2 import BaseLoader, Environment, StrictUndefined

from litgraph.settings import get_settings

_jinja_env = Environment(loader=BaseLoader(), undefined=StrictUndefined)


def load_prompt(name: str, **kwargs) -> tuple[str, str]:
    """Load a prompt YAML and render the template with Jinja2.

    Args:
        name: Prompt name (without .yaml extension), e.g. "paper_analysis".
        **kwargs: Template variables for Jinja2 rendering.

    Returns:
        (system_prompt, user_prompt) tuple.
    """
    settings = get_settings()
    yaml_path = settings.prompts_dir / f"{name}.yaml"
    return load_prompt_from_path(yaml_path, **kwargs)


def load_prompt_from_path(yaml_path: Path, **kwargs) -> tuple[str, str]:
    """Load a prompt YAML from a specific path and render with Jinja2.

    Args:
        yaml_path: Path to the YAML file.
        **kwargs: Template variables for Jinja2 rendering.

    Returns:
        (system_prompt, user_prompt) tuple.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    system_prompt = data["system"].strip()

    template = _jinja_env.from_string(data["template"])
    user_prompt = template.render(**kwargs).strip()

    return system_prompt, user_prompt


def load_questions(questions_path: Path | None = None) -> list[dict]:
    """Load questions from questions.yaml.

    Returns:
        List of question dicts with 'id' and 'text' keys.
    """
    if questions_path is None:
        settings = get_settings()
        questions_path = settings.questions_path

    with open(questions_path) as f:
        data = yaml.safe_load(f)

    return data["questions"]


def get_questions_version(questions_path: Path | None = None) -> int:
    """Return the version number from questions.yaml."""
    if questions_path is None:
        settings = get_settings()
        questions_path = settings.questions_path

    with open(questions_path) as f:
        data = yaml.safe_load(f)

    return data.get("version", 1)


def format_questions_block(questions: list[dict]) -> str:
    """Format a list of questions into a numbered block for prompt inclusion."""
    lines = []
    for i, q in enumerate(questions, 1):
        lines.append(f"{i}. {q['text']}")
    return "\n".join(lines)
