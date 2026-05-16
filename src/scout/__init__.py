"""Scout — AI agent that writes web scraping scripts."""

from __future__ import annotations

from typing import Union

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
from .autofix.types import AutoFixMode
from .browser import Browser, LaunchOptions
from .inputs import Input
from .schema.tolerance import Tolerance
from .schema.types import Field, Items
from .scraper import Scraper, ScraperResult

# Schema type alias — gives IDE users a concrete type for the ``schema=``
# parameter instead of ``Any``.
Schema = Union[
    type,       # str, int, float, bool, dict
    Field,
    Items,
    dict,       # {"key": SchemaType, ...}
    list,       # [SchemaType] — always exactly one element
]

# Backward compatibility — importable and in __all__ with deprecation path.
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
    "Browser",
    "Scraper",
    "ScraperResult",
    "Schema",
    "Field",
    "Input",
    "Items",
    "AutoFixMode",
    "LaunchOptions",
    "Tolerance",
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
    # Backward compatibility aliases (deprecated — use the short names above)
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
