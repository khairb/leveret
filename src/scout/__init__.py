"""Scout — AI agent that writes web scraping scripts."""

from .errors import (
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
from .schema.types import Field, List, SchemaType
from .scraper import Scraper, ScraperResult

__all__ = [
    # Core API
    "Scraper",
    "ScraperResult",
    "Field",
    "List",
    "SchemaType",
    # Errors
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
