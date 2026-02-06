"""Configuration center: loads .env + config.default.yaml → global Settings singleton."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


# OAuth token prefix for Anthropic subscription tokens
ANTHROPIC_OAUTH_PREFIX = "sk-ant-oat"


@dataclass
class LLMConfig:
    """LLM configuration for Pro mode (Anthropic) or Lite mode (Ollama)."""

    # For Pro mode: Anthropic API key or OAuth token
    api_key: str = ""
    best_model: str = "claude-sonnet-4-20250514"
    cheap_model: str = "claude-haiku-4-20250414"

    # For Lite mode: Ollama base URL
    base_url: str = "http://localhost:11434/v1"

    @property
    def is_oauth_token(self) -> bool:
        """Check if the API key is an OAuth subscription token."""
        return self.api_key.startswith(ANTHROPIC_OAUTH_PREFIX)


@dataclass
class RetryConfig:
    max_retries: int = 3
    backoff_base: float = 2.0


@dataclass
class RateLimitConfig:
    search_max_calls: int = 90
    search_period: int = 300


@dataclass
class Settings:
    mode: str = "pro"
    data_dir: Path = field(default_factory=lambda: Path("../DATA"))
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding_model: str = "all-MiniLM-L6-v2"
    retry: RetryConfig = field(default_factory=RetryConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    prompts_dir: Path = field(default=None)
    questions_path: Path = field(default=None)
    schema_path: Path = field(default=None)

    def __post_init__(self):
        if self.prompts_dir is None:
            self.prompts_dir = self.project_root / "config" / "prompts"
        if self.questions_path is None:
            self.questions_path = self.project_root / "config" / "questions.yaml"
        if self.schema_path is None:
            self.schema_path = self.project_root / "config" / "schema.yaml"


_settings: Settings | None = None


def _resolve_data_dir(raw: str, project_root: Path) -> Path:
    """Resolve data_dir: absolute path stays, relative resolves from project_root."""
    p = Path(raw)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def get_settings(project_root: Path | None = None, force_reload: bool = False) -> Settings:
    """Return the global Settings singleton, loading from .env + config.default.yaml."""
    global _settings
    if _settings is not None and not force_reload:
        return _settings

    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent

    # Load .env from project root
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)

    # Load config.default.yaml
    default_config_path = project_root / "config" / "config.default.yaml"
    defaults = {}
    if default_config_path.exists():
        with open(default_config_path) as f:
            defaults = yaml.safe_load(f) or {}

    mode = os.environ.get("LITGRAPH_MODE", "pro").lower()

    # Build LLM config based on mode
    if mode == "lite":
        ollama_model = os.environ.get("LITGRAPH_OLLAMA_MODEL", "qwen2.5:7b")
        llm = LLMConfig(
            base_url=os.environ.get("LITGRAPH_OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",  # Ollama doesn't need a real key
            best_model=ollama_model,
            cheap_model=ollama_model,
        )
    else:
        # Pro mode: use Anthropic API directly
        # Priority: ANTHROPIC_OAUTH_TOKEN > LITGRAPH_ANTHROPIC_API_KEY > ANTHROPIC_API_KEY
        api_key = (
            os.environ.get("ANTHROPIC_OAUTH_TOKEN", "").strip()
            or os.environ.get("LITGRAPH_ANTHROPIC_API_KEY", "").strip()
            or os.environ.get("ANTHROPIC_API_KEY", "").strip()
        )
        llm = LLMConfig(
            api_key=api_key,
            best_model=os.environ.get("LITGRAPH_PRO_BEST_MODEL", "claude-sonnet-4-20250514"),
            cheap_model=os.environ.get("LITGRAPH_PRO_CHEAP_MODEL", "claude-haiku-4-20250414"),
        )

    retry = RetryConfig(
        max_retries=int(os.environ.get("LITGRAPH_MAX_RETRIES", "3")),
        backoff_base=float(os.environ.get("LITGRAPH_RETRY_BACKOFF_BASE", "2.0")),
    )

    rate_limit = RateLimitConfig(
        search_max_calls=int(os.environ.get("LITGRAPH_SEARCH_RATE_LIMIT", "90")),
        search_period=int(os.environ.get("LITGRAPH_SEARCH_RATE_PERIOD", "300")),
    )

    data_dir = _resolve_data_dir(
        os.environ.get("LITGRAPH_DATA_DIR", defaults.get("data_dir", "../DATA")),
        project_root,
    )

    _settings = Settings(
        mode=mode,
        data_dir=data_dir,
        llm=llm,
        embedding_model=os.environ.get("LITGRAPH_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        retry=retry,
        rate_limit=rate_limit,
        project_root=project_root,
        prompts_dir=project_root / "config" / "prompts",
        questions_path=project_root / "config" / "questions.yaml",
        schema_path=project_root / "config" / "schema.yaml",
    )
    return _settings


def reset_settings() -> None:
    """Clear the singleton (used in tests)."""
    global _settings
    _settings = None
