"""Scout — AI agent that writes web scraping scripts."""

from .errors import (
    AutoFixError,
    ConfigError,
    Error,
    GenerationError,
    SandboxViolationError,
    SchemaError,
    ScriptError,
    ScriptLoadError,
    ScriptRuntimeError,
    ScriptTimeoutError,
    ValidationError,
)
from .schema.types import Field, Items
from .scraper import Scraper, ScraperResult

# Backward compatibility — importable but not in __all__
from .agent.llm import ModelName as ModelName
from .errors import (
    ScoutAutoFixError as ScoutAutoFixError,
    ScoutConfigError as ScoutConfigError,
    ScoutError as ScoutError,
    ScoutGenerationError as ScoutGenerationError,
    ScoutSchemaError as ScoutSchemaError,
    ScoutScriptError as ScoutScriptError,
    ScoutScriptLoadError as ScoutScriptLoadError,
    ScoutScriptRuntimeError as ScoutScriptRuntimeError,
    ScoutScriptTimeoutError as ScoutScriptTimeoutError,
    ScoutValidationError as ScoutValidationError,
)
from .schema.types import List as List, SchemaType as SchemaType

__all__ = [
    # Core
    "Scraper",
    "ScraperResult",
    "Field",
    "Items",
    "List",
    # Errors
    "AutoFixError",
    "ConfigError",
    "Error",
    "GenerationError",
    "SchemaError",
    "ScriptError",
    "ScriptLoadError",
    "ScriptRuntimeError",
    "SandboxViolationError",
    "ScriptTimeoutError",
    "ValidationError",
]
