"""Unified LLM interface — Anthropic SDK for Pro mode, OpenAI SDK for Lite mode (Ollama)."""

from __future__ import annotations

import asyncio
import logging

import httpx
import anthropic
from openai import OpenAI

from litgraph.retry import with_retry
from litgraph.settings import get_settings

logger = logging.getLogger(__name__)

# Client singletons
_anthropic_client: anthropic.Anthropic | None = None
_httpx_client: httpx.Client | None = None
_openai_client: OpenAI | None = None
_lite_warned: bool = False
_using_oauth: bool = False

# Anthropic API settings
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
OAUTH_BETA_HEADER = "oauth-2025-04-20"


def _get_httpx_client() -> httpx.Client:
    """Lazy-initialize the httpx client singleton for OAuth requests."""
    global _httpx_client
    if _httpx_client is None:
        _httpx_client = httpx.Client(timeout=120.0)
    return _httpx_client


def _get_anthropic_client() -> anthropic.Anthropic:
    """Lazy-initialize the Anthropic SDK client singleton (for API key auth)."""
    global _anthropic_client
    if _anthropic_client is None:
        settings = get_settings()
        token = settings.llm.api_key

        if not token:
            raise ValueError(
                "Anthropic API key not configured. Set ANTHROPIC_OAUTH_TOKEN or LITGRAPH_ANTHROPIC_API_KEY."
            )

        logger.debug("Using standard API key with Anthropic SDK")
        _anthropic_client = anthropic.Anthropic(api_key=token)

    return _anthropic_client


def _get_openai_client() -> OpenAI:
    """Lazy-initialize the OpenAI client singleton (for Lite mode / Ollama)."""
    global _openai_client
    if _openai_client is None:
        settings = get_settings()
        _openai_client = OpenAI(
            base_url=settings.llm.base_url,
            api_key=settings.llm.api_key,
        )
    return _openai_client


def reset_client() -> None:
    """Reset the client singletons (for tests)."""
    global _anthropic_client, _httpx_client, _openai_client, _lite_warned, _using_oauth
    _anthropic_client = None
    if _httpx_client is not None:
        _httpx_client.close()
        _httpx_client = None
    _openai_client = None
    _lite_warned = False
    _using_oauth = False


def _maybe_warn_lite() -> None:
    """Log a warning on first call in Lite mode."""
    global _lite_warned
    settings = get_settings()
    if settings.mode == "lite" and not _lite_warned:
        logger.warning(
            "Lite mode: best_model and cheap_model point to the same model (%s), "
            "quality may be limited",
            settings.llm.best_model,
        )
        _lite_warned = True


def _complete_anthropic_oauth(prompt: str, system_prompt: str | None, model: str) -> str:
    """Call Anthropic API with OAuth token using httpx directly.

    OAuth tokens require:
    - Authorization: Bearer header
    - anthropic-beta: oauth-2025-04-20 header

    See: https://deepwiki.com/sst/opencode-anthropic-auth
    """
    settings = get_settings()
    model_name = settings.llm.best_model if model == "best" else settings.llm.cheap_model
    token = settings.llm.api_key

    # Build request headers
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "anthropic-version": ANTHROPIC_VERSION,
        "anthropic-beta": OAUTH_BETA_HEADER,
    }

    # Build request payload
    payload = {
        "model": model_name,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        payload["system"] = system_prompt

    # Make request
    client = _get_httpx_client()
    response = client.post(ANTHROPIC_API_URL, headers=headers, json=payload)

    # Handle errors
    if response.status_code != 200:
        error_data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        error_msg = error_data.get("error", {}).get("message", response.text)
        raise RuntimeError(f"Anthropic API error ({response.status_code}): {error_msg}")

    # Extract text from response
    data = response.json()
    content = data.get("content", [])
    if content and len(content) > 0:
        return content[0].get("text", "")
    return ""


def _complete_anthropic_sdk(prompt: str, system_prompt: str | None, model: str) -> str:
    """Call Anthropic API with standard API key using the SDK."""
    settings = get_settings()
    model_name = settings.llm.best_model if model == "best" else settings.llm.cheap_model

    client = _get_anthropic_client()

    # Build messages
    messages = [{"role": "user", "content": prompt}]

    # Call Anthropic API
    kwargs = {
        "model": model_name,
        "max_tokens": 4096,
        "messages": messages,
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    response = client.messages.create(**kwargs)

    # Extract text from response
    content = response.content
    if content and len(content) > 0:
        return content[0].text
    return ""


def _complete_anthropic(prompt: str, system_prompt: str | None, model: str) -> str:
    """Call Anthropic API directly.

    Uses httpx for OAuth tokens, SDK for standard API keys.
    """
    settings = get_settings()

    if settings.llm.is_oauth_token:
        logger.debug("Using OAuth token with httpx client")
        return _complete_anthropic_oauth(prompt, system_prompt, model)
    else:
        logger.debug("Using standard API key with Anthropic SDK")
        return _complete_anthropic_sdk(prompt, system_prompt, model)


def _complete_openai(prompt: str, system_prompt: str | None, model: str) -> str:
    """Call OpenAI-compatible API (Ollama)."""
    settings = get_settings()
    model_name = settings.llm.best_model if model == "best" else settings.llm.cheap_model

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    client = _get_openai_client()
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )
    return response.choices[0].message.content


@with_retry
def complete(prompt: str, system_prompt: str | None = None, model: str = "best") -> str:
    """Synchronous LLM completion.

    Args:
        prompt: User prompt text.
        system_prompt: Optional system prompt.
        model: "best" or "cheap" — selects from settings.

    Returns:
        LLM response text.
    """
    _maybe_warn_lite()
    settings = get_settings()

    if settings.mode == "lite":
        return _complete_openai(prompt, system_prompt, model)
    else:
        return _complete_anthropic(prompt, system_prompt, model)


async def best_model_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list | None = None,
    **kwargs,
) -> str:
    """Async wrapper for nano-graphrag — uses best model.

    Handles the hashing_kv caching kwarg from nano-graphrag.
    Uses asyncio.to_thread to wrap sync API call.
    """
    # Handle nano-graphrag caching
    hashing_kv = kwargs.get("hashing_kv")
    if hashing_kv is not None:
        cache_key = f"{prompt}_{system_prompt}"
        cached = await hashing_kv.get_by_id(cache_key)
        if cached and "return" in cached:
            return cached["return"]

    result = await asyncio.to_thread(complete, prompt, system_prompt, "best")

    if hashing_kv is not None:
        await hashing_kv.upsert({cache_key: {"return": result}})

    return result


async def cheap_model_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list | None = None,
    **kwargs,
) -> str:
    """Async wrapper for nano-graphrag — uses cheap model."""
    hashing_kv = kwargs.get("hashing_kv")
    if hashing_kv is not None:
        cache_key = f"{prompt}_{system_prompt}"
        cached = await hashing_kv.get_by_id(cache_key)
        if cached and "return" in cached:
            return cached["return"]

    result = await asyncio.to_thread(complete, prompt, system_prompt, "cheap")

    if hashing_kv is not None:
        await hashing_kv.upsert({cache_key: {"return": result}})

    return result
