"""Multi-provider LLM wrapper using Pydantic AI's direct model API.

Translates between Scout's internal message format (Anthropic-style dicts)
and Pydantic AI's message types.  Supports any provider Pydantic AI
supports — Anthropic, OpenAI, Google, Groq, Mistral, etc.

For Anthropic, prompt caching is enabled automatically (system prompt,
tool definitions, and conversation messages) to reduce input token costs.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, NoReturn, Union

from pydantic_ai.direct import model_request
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    InstructionPart,
    ModelRequest as PydanticRequest,
    ModelResponse as PydanticResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import KnownModelName, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

#: Type alias for model names. Accepts any known ``"provider:model"``
#: string (with IDE autocomplete) or an arbitrary string for custom models.
ModelName = Union[KnownModelName, str]

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════


_WARNED_BARE_MODELS: set[str] = set()


def _normalize_model(model: str) -> str:
    """Auto-prefix 'anthropic:' for bare model names without a provider.

    Allows backward-compatible usage: ``"claude-haiku-4-5"`` is treated
    as ``"anthropic:claude-haiku-4-5"``.

    Logs a warning (once per model name) if the bare name looks like a
    non-Anthropic model (e.g. ``"gpt-4o"``) to help catch
    misconfiguration early.
    """
    if ":" in model:
        return model

    # Heuristic: Anthropic models start with "claude".
    if not model.startswith("claude") and model not in _WARNED_BARE_MODELS:
        _WARNED_BARE_MODELS.add(model)
        logger.warning(
            "Model '%s' has no provider prefix — assuming 'anthropic:%s'. "
            "If this is not an Anthropic model, use the full format: "
            "model='provider:%s' (e.g. 'openai:%s').",
            model, model, model, model,
        )

    return f"anthropic:{model}"


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for the LLM provider.

    The ``model`` field uses Pydantic AI's ``provider:model`` format,
    e.g. ``"anthropic:claude-haiku-4-5"``, ``"openai:gpt-4o"``.
    Bare model names (no colon) are assumed to be Anthropic models.
    """

    model: ModelName = "claude-haiku-4-5"
    temperature: float = 0.0
    max_tokens: int = 16384
    api_key: str | None = None


# Retry settings
_MAX_RETRIES = 10
_INITIAL_DELAY = 10.0  # seconds
_MAX_DELAY = 120.0  # seconds


# ═══════════════════════════════════════════════════════════════
#  Response types — drop-in replacements for Anthropic SDK types
# ═══════════════════════════════════════════════════════════════


@dataclass
class ContentBlock:
    """A single content block in an LLM response."""

    type: str  # "text" or "tool_use"
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class Usage:
    """Token usage statistics."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class LLMResponse:
    """Provider-agnostic LLM response.

    Mirrors the shape of ``anthropic.types.Message`` so that all existing
    callers (loop, validator, requirements, trace) work without changes.
    """

    content: list[ContentBlock]
    usage: Usage
    stop_reason: str  # "end_turn", "tool_use", "max_tokens"


# ═══════════════════════════════════════════════════════════════
#  Translation: Scout → Pydantic AI
# ═══════════════════════════════════════════════════════════════


def _to_pydantic_messages(
    messages: list[dict],
) -> list[PydanticRequest | PydanticResponse]:
    """Convert Scout's Anthropic-format messages to Pydantic AI format."""
    result: list[PydanticRequest | PydanticResponse] = []

    # Build a lookup from tool_use_id → tool_name across all assistant
    # messages. Scout's tool_result blocks don't carry tool_name, but
    # Pydantic AI's ToolReturnPart requires it for some providers (e.g.
    # OpenAI). The mapping comes from the preceding assistant message's
    # tool_use blocks.
    tool_id_to_name: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "tool_use":
                        tid = block.get("id", "")
                        if tid:
                            tool_id_to_name[tid] = block.get("name", "")

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            if isinstance(content, str):
                result.append(
                    PydanticRequest(parts=[UserPromptPart(content=content)])
                )
            elif isinstance(content, list):
                parts = []
                for block in content:
                    btype = block.get("type")
                    if btype == "tool_result":
                        tool_use_id = block.get("tool_use_id", "")
                        parts.append(
                            ToolReturnPart(
                                tool_name=tool_id_to_name.get(
                                    tool_use_id, "",
                                ),
                                content=block.get("content", ""),
                                tool_call_id=tool_use_id,
                            )
                        )
                    elif btype == "text":
                        parts.append(
                            UserPromptPart(content=block.get("text", ""))
                        )
                if parts:
                    result.append(PydanticRequest(parts=parts))

        elif role == "assistant":
            if isinstance(content, list):
                parts = []
                for block in content:
                    btype = block.get("type")
                    if btype == "text":
                        parts.append(TextPart(content=block.get("text", "")))
                    elif btype == "tool_use":
                        parts.append(
                            ToolCallPart(
                                tool_name=block.get("name", ""),
                                args=block.get("input", {}),
                                tool_call_id=block.get("id", ""),
                            )
                        )
                if parts:
                    result.append(PydanticResponse(parts=parts))

    return result


def _to_pydantic_tools(tools: list[dict]) -> list[ToolDefinition]:
    """Convert Anthropic-format tool schemas to Pydantic AI ToolDefinition."""
    return [
        ToolDefinition(
            name=t["name"],
            description=t.get("description", ""),
            parameters_json_schema=t.get("input_schema", {"type": "object", "properties": {}}),
        )
        for t in tools
    ]


# ═══════════════════════════════════════════════════════════════
#  Translation: Pydantic AI → Scout
# ═══════════════════════════════════════════════════════════════

_FINISH_REASON_MAP = {
    "stop": "end_turn",
    "end_turn": "end_turn",
    "tool_calls": "tool_use",
    "tool-calls": "tool_use",
    "tool_use": "tool_use",
    "length": "max_tokens",
    "max_tokens": "max_tokens",
}


def _from_pydantic_response(response: PydanticResponse) -> LLMResponse:
    """Convert a Pydantic AI ModelResponse to Scout's LLMResponse."""
    blocks: list[ContentBlock] = []
    has_tool_use = False

    for part in response.parts:
        if isinstance(part, TextPart):
            blocks.append(ContentBlock(type="text", text=part.content))
        elif isinstance(part, ToolCallPart):
            has_tool_use = True
            args = part.args if isinstance(part.args, dict) else part.args_as_dict()
            blocks.append(
                ContentBlock(
                    type="tool_use",
                    id=part.tool_call_id,
                    name=part.tool_name,
                    input=args,
                )
            )

    usage = Usage(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cache_read_input_tokens=response.usage.cache_read_tokens,
        cache_creation_input_tokens=response.usage.cache_write_tokens,
    )

    raw_reason = response.finish_reason or ""
    stop_reason = _FINISH_REASON_MAP.get(
        raw_reason, "tool_use" if has_tool_use else "end_turn"
    )

    return LLMResponse(content=blocks, usage=usage, stop_reason=stop_reason)


# ═══════════════════════════════════════════════════════════════
#  Model settings (provider-aware caching)
# ═══════════════════════════════════════════════════════════════


def _build_model_settings(config: LLMConfig) -> ModelSettings:
    """Build provider-appropriate model settings.

    For Anthropic, enables automatic prompt caching on system prompt,
    tool definitions, and conversation messages.
    """
    model = _normalize_model(config.model)
    provider = model.split(":")[0]

    base: dict[str, Any] = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }

    if provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModelSettings

        return AnthropicModelSettings(
            **base,
            anthropic_cache=True,
            anthropic_cache_instructions=True,
            anthropic_cache_tool_definitions=True,
        )

    return ModelSettings(**base)


def _resolve_model(config: LLMConfig) -> Any:
    """Resolve a model string to a Pydantic AI model instance.

    When an explicit API key is provided, constructs the provider with
    that key.  Otherwise, returns the model string and lets Pydantic AI
    resolve it (reading env vars automatically).

    Works with ANY provider Pydantic AI supports — no provider-specific
    code needed.
    """
    from pydantic_ai.models import infer_model
    from pydantic_ai.providers import infer_provider, infer_provider_class

    model = _normalize_model(config.model)

    if not config.api_key:
        return model

    def _provider_with_key(provider_name: str) -> Any:
        """Create a provider instance with the explicit API key."""
        if provider_name.startswith("gateway/"):
            return infer_provider(provider_name)

        if provider_name in ("google-vertex", "google-gla", "vertexai"):
            from pydantic_ai.providers.google import GoogleProvider

            return GoogleProvider(
                api_key=config.api_key,
                vertexai=provider_name in ("google-vertex", "vertexai"),
            )

        provider_cls = infer_provider_class(provider_name)
        return provider_cls(api_key=config.api_key)

    return infer_model(model, provider_factory=_provider_with_key)


# ═══════════════════════════════════════════════════════════════
#  Provider error handling
# ═══════════════════════════════════════════════════════════════

# Every provider Pydantic AI supports (used for typo suggestions).
_KNOWN_PROVIDERS = (
    "anthropic", "openai", "google-gla", "google-vertex", "groq",
    "mistral", "cohere", "bedrock", "deepseek", "azure", "grok", "xai",
    "ollama", "together", "fireworks", "cerebras", "huggingface",
    "openrouter", "github", "sambanova", "nebius", "moonshotai",
)


def _raise_provider_error(config: LLMConfig, exc: Exception) -> NoReturn:
    """Raise a clear, actionable error when a provider can't be loaded."""
    model = _normalize_model(config.model)
    provider = model.split(":")[0]

    if isinstance(exc, ImportError):
        raise ImportError(
            f"\n"
            f"  Provider '{provider}' requires its SDK to be installed.\n"
            f"\n"
            f"  To fix this, run:\n"
            f"\n"
            f"    pip install {provider}\n"
            f"\n"
            f"  You passed: model=\"{config.model}\"\n"
        ) from exc

    if isinstance(exc, ValueError) and "Unknown provider" in str(exc):
        # Find similar provider names to suggest.
        from difflib import get_close_matches

        matches = get_close_matches(provider, _KNOWN_PROVIDERS, n=3, cutoff=0.5)
        suggestion = ""
        if matches:
            suggestion = (
                f"\n"
                f"  Did you mean one of these?\n"
                + "".join(f"    - {m}\n" for m in matches)
            )

        raise ValueError(
            f"\n"
            f"  Unknown provider '{provider}'.\n"
            f"{suggestion}"
            f"\n"
            f"  Supported providers include:\n"
            f"    anthropic, openai, google-gla, groq, mistral, cohere,\n"
            f"    bedrock, deepseek, azure, ollama, together, fireworks,\n"
            f"    and more.\n"
            f"\n"
            f"  Usage: model=\"provider:model-name\"\n"
            f"  Example: model=\"openai:gpt-4o\"\n"
            f"\n"
            f"  You passed: model=\"{config.model}\"\n"
        ) from exc

    raise exc


# ═══════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════


async def call_llm(
    config: LLMConfig,
    *,
    system: str,
    messages: list[dict],
    tools: list[dict] | None = None,
) -> LLMResponse:
    """Make an LLM API call with tool-use support and provider-aware caching.

    For Anthropic models, prompt caching is enabled automatically on the
    system prompt, tool definitions, and conversation messages.

    Retries up to ``_MAX_RETRIES`` times on transient errors with
    exponential backoff.

    Args:
        config: LLM configuration (model, temperature, api key, etc.).
        system: The system prompt.
        messages: Conversation messages in Anthropic dict format.
        tools: Tool definitions in Anthropic dict format (optional).

    Returns:
        An :class:`LLMResponse` with content blocks, usage, and stop reason.
    """
    try:
        model = _resolve_model(config)
    except (ImportError, ValueError) as exc:
        _raise_provider_error(config, exc)

    settings = _build_model_settings(config)
    pydantic_messages = _to_pydantic_messages(messages)

    mrp = ModelRequestParameters(
        function_tools=_to_pydantic_tools(tools) if tools else [],
        allow_text_output=True,
        instruction_parts=[InstructionPart(content=system)],
    )

    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = await model_request(
                model,
                pydantic_messages,
                model_settings=settings,
                model_request_parameters=mrp,
            )

            result = _from_pydantic_response(response)

            cache_read = result.usage.cache_read_input_tokens
            cache_create = result.usage.cache_creation_input_tokens
            if cache_read or cache_create:
                logger.debug(
                    "Cache stats: %d read, %d created, %d uncached input tokens",
                    cache_read,
                    cache_create,
                    result.usage.input_tokens,
                )

            return result

        except (ImportError, ValueError) as exc:
            _raise_provider_error(config, exc)

        except (ModelHTTPError, ConnectionError, TimeoutError) as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                delay = min(_INITIAL_DELAY * (2**attempt), _MAX_DELAY)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. "
                    "Retrying in %.0fs...",
                    attempt + 1,
                    _MAX_RETRIES,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "LLM call failed after %d attempts: %s",
                    _MAX_RETRIES,
                    exc,
                )

    raise last_exc  # type: ignore[misc]
