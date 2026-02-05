"""Unified LLM interface — sync client with async wrappers for nano-graphrag."""

from __future__ import annotations

import asyncio
import logging

from openai import OpenAI

from litgraph.retry import with_retry
from litgraph.settings import get_settings

logger = logging.getLogger(__name__)

_client: OpenAI | None = None
_lite_warned: bool = False


def _get_client() -> OpenAI:
    """Lazy-initialize the OpenAI client singleton."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = OpenAI(
            base_url=settings.llm.base_url,
            api_key=settings.llm.api_key,
        )
    return _client


def reset_client() -> None:
    """Reset the client singleton (for tests)."""
    global _client, _lite_warned
    _client = None
    _lite_warned = False


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

    model_name = settings.llm.best_model if model == "best" else settings.llm.cheap_model

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    client = _get_client()
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )
    return response.choices[0].message.content


async def best_model_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list | None = None,
    **kwargs,
) -> str:
    """Async wrapper for nano-graphrag — uses best model.

    Handles the hashing_kv caching kwarg from nano-graphrag.
    Uses asyncio.to_thread to wrap sync OpenAI call.
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
