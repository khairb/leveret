"""Scout — AI agent that writes web scraping scripts."""

from .agent.llm import ModelName
from .errors import (
    # New short names (preferred)
    AutoFixError,
    ConfigError,
    Error,
    GenerationError,
    SchemaError,
    ScriptError,
    ScriptLoadError,
    ScriptRuntimeError,
    ScriptTimeoutError,
    ValidationError,
    # Backward compatibility aliases
    ScoutAutoFixError,
    ScoutConfigError,
    ScoutError,
    ScoutGenerationError,
    ScoutSchemaError,
    ScoutScriptError,
    ScoutScriptLoadError,
    ScoutScriptRuntimeError,
    ScoutScriptTimeoutError,
    ScoutValidationError,
)
from .schema.types import Field, Items, List, SchemaType
from .scraper import Scraper, ScraperResult

__all__ = [
    # Core API
    "Scraper",
    "ScraperResult",
    "ModelName",
    "Field",
    "Items",
    "SchemaType",
    # Errors (short names)
    "AutoFixError",
    "ConfigError",
    "Error",
    "GenerationError",
    "SchemaError",
    "ScriptError",
    "ScriptLoadError",
    "ScriptRuntimeError",
    "ScriptTimeoutError",
    "ValidationError",
    # Backward compatibility
    "List",
    "ScoutAutoFixError",
    "ScoutConfigError",
    "ScoutError",
    "ScoutGenerationError",
    "ScoutSchemaError",
    "ScoutScriptError",
    "ScoutScriptLoadError",
    "ScoutScriptRuntimeError",
    "ScoutScriptTimeoutError",
    "ScoutValidationError",
]
