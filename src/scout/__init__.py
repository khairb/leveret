"""Scout — AI agent that writes web scraping scripts."""

from __future__ import annotations

from typing import Union

from .autofix.types import RegenerateMode
from .browser import Browser, LaunchOptions
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
from .inputs import Input
from .schema.tolerance import Tolerance
from .schema.types import Field, Items
from .scraper import Scraper, ScraperResult

# Schema type alias — gives IDE users a concrete type for the ``schema=``
# parameter instead of ``Any``.
Schema = Union[
    type,  # str, int, float, bool, dict
    Field,
    Items,
    dict,  # {"key": SchemaType, ...}
    list,  # [SchemaType] — always exactly one element
]

# Backward compatibility — importable and in __all__ with deprecation path.
from .agent.llm import ModelName as ModelName
from .errors import (
    ScoutAutoFixError as ScoutAutoFixError,
)
from .errors import (
    ScoutAutoRegenerateError as ScoutAutoRegenerateError,
)
from .errors import (
    ScoutConfigError as ScoutConfigError,
)
from .errors import (
    ScoutError as ScoutError,
)
from .errors import (
    ScoutGenerationError as ScoutGenerationError,
)
from .errors import (
    ScoutSchemaError as ScoutSchemaError,
)
from .errors import (
    ScoutScriptError as ScoutScriptError,
)
from .errors import (
    ScoutScriptLoadError as ScoutScriptLoadError,
)
from .errors import (
    ScoutScriptRuntimeError as ScoutScriptRuntimeError,
)
from .errors import (
    ScoutScriptTimeoutError as ScoutScriptTimeoutError,
)
from .errors import (
    ScoutValidationError as ScoutValidationError,
)
from .schema.types import List as List
from .schema.types import SchemaType as SchemaType

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
