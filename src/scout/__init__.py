"""Scout — AI agent that writes web scraping scripts."""

from __future__ import annotations

from typing import Union

from .errors import (
    AutoRegenerateError,
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
from .autofix.types import RegenerateMode
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
    ScoutAutoRegenerateError as ScoutAutoRegenerateError,
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
    "RegenerateMode",
    "LaunchOptions",
    "Tolerance",
    # Errors
    "AutoRegenerateError",
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


# ── Backward compatibility: deprecated names importable but not in __all__ ──

def __getattr__(name: str):
    import warnings as _warnings
    _deprecated = {
        "AutoFixMode": ("RegenerateMode", RegenerateMode),
        "AutoFixError": ("AutoRegenerateError", AutoRegenerateError),
    }
    if name in _deprecated:
        new_name, obj = _deprecated[name]
        _warnings.warn(
            f"{name} is deprecated, use {new_name} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
