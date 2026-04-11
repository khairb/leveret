"""Thin wrapper around the Anthropic Python SDK for tool-use calls.

Includes automatic prompt caching — system prompts, tool definitions,
and the stable conversation prefix are cached across turns to reduce
input token costs by ~40-60%.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

_CACHE_CONTROL = {"type": "ephemeral"}


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for the Anthropic LLM client."""

    model: str = "claude-haiku-4-5"
    temperature: float = 0.0
    max_tokens: int = 16384
    api_key: str | None = None  # falls back to ANTHROPIC_API_KEY env var


# Retry settings
_MAX_RETRIES = 10
_INITIAL_DELAY = 10.0  # seconds
_MAX_DELAY = 120.0  # seconds


def _make_cached_system(system: str) -> list[dict[str, Any]]:
    """Wrap a plain-text system prompt as a cached content block."""
    return [
        {
            "type": "text",
            "text": system,
            "cache_control": _CACHE_CONTROL,
        }
    ]


def _make_cached_tools(tools: list[dict]) -> list[dict]:
    """Add a cache breakpoint to the last tool definition."""
    if not tools:
        return tools
    cached = copy.deepcopy(tools)
    cached[-1]["cache_control"] = _CACHE_CONTROL
    return cached


def _add_conversation_cache_breakpoint(
    messages: list[dict],
) -> list[dict]:
    """Add a cache breakpoint to the last user message before the final turn.

    This caches the stable conversation prefix so that only the newest
    turn's tokens are processed at full price.  We mark the second-to-last
    user-role message (the last one that won't change on the next call).

    If the conversation has fewer than 4 messages, the prefix is too small
    to benefit from caching — return as-is.
    """
    if len(messages) < 4:
        return messages

    # Work on a shallow copy so we don't mutate the caller's list.
    msgs = list(messages)

    # Find the second-to-last user message.
    user_indices = [i for i, m in enumerate(msgs) if m.get("role") == "user"]
    if len(user_indices) < 2:
        return msgs

    target_idx = user_indices[-2]
    target = msgs[target_idx]
    content = target.get("content")

    # Deep-copy only the message we're modifying.
    target = dict(target)
    msgs[target_idx] = target

    if isinstance(content, str):
        # Plain text message — wrap in a content block with cache marker.
        target["content"] = [
            {
                "type": "text",
                "text": content,
                "cache_control": _CACHE_CONTROL,
            }
        ]
    elif isinstance(content, list) and content:
        # List of content blocks (e.g., tool_result blocks).
        # Add cache marker to the last block.
        content = [dict(b) for b in content]  # shallow copy of each block
        content[-1] = {**content[-1], "cache_control": _CACHE_CONTROL}
        target["content"] = content

    return msgs


async def call_llm(
    config: LLMConfig,
    *,
    system: str,
    messages: list[dict],
    tools: list[dict] | None = None,
) -> anthropic.types.Message:
    """Make a single Anthropic API call with tool-use support and caching.

    Three cache breakpoints are placed automatically:
    1. System prompt — cached across all turns.
    2. Last tool definition — cached across all turns.
    3. Second-to-last user message — caches the stable conversation prefix.

    Retries up to 3 times on transient errors (connection errors, rate
    limits, server errors) with exponential backoff from 10s to 30s.

    Args:
        config: LLM configuration.
        system: The system prompt (passed as top-level ``system`` param).
        messages: Conversation messages in Anthropic format.
        tools: Tool definitions in Anthropic format (optional).

    Returns:
        The raw ``anthropic.types.Message`` response.
    """
    client = anthropic.AsyncAnthropic(
        api_key=config.api_key or os.environ.get("ANTHROPIC_API_KEY"),
    )

    kwargs: dict[str, Any] = {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "system": _make_cached_system(system),
        "messages": _add_conversation_cache_breakpoint(messages),
    }
    if tools:
        kwargs["tools"] = _make_cached_tools(tools)

    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = await client.messages.create(**kwargs)
            # Log cache performance on first successful call.
            usage = response.usage
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
            if cache_read or cache_create:
                logger.debug(
                    "Cache stats: %d read, %d created, %d uncached input tokens",
                    cache_read, cache_create, usage.input_tokens,
                )
            return response
        except (
            anthropic.APIConnectionError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
        ) as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                delay = min(_INITIAL_DELAY * (2 ** attempt), _MAX_DELAY)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. "
                    "Retrying in %.0fs...",
                    attempt + 1, _MAX_RETRIES, exc, delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "LLM call failed after %d attempts: %s",
                    _MAX_RETRIES, exc,
                )

    raise last_exc  # type: ignore[misc]
