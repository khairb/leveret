"""Scout exception hierarchy.

All Scout exceptions inherit from Error, allowing broad catching
with ``except Error`` or specific catching with subclasses.

Hierarchy::

    Error
    ├── SchemaError
    ├── ConfigError
    ├── GenerationError
    ├── AutoFixError
    ├── ScriptError
    │   ├── ScriptLoadError
    │   ├── ScriptRuntimeError
    │   └─�� ScriptTimeoutError
    └── ValidationError
"""


class Error(Exception):
    """Base exception for all Scout errors."""


class SchemaError(Error):
    """Invalid schema definition.

    Raised at schema construction time when the user provides an invalid
    schema — e.g. ``Field(str, min=5)`` (min doesn't apply to str),
    or a list schema with more than one element.
    """


class ConfigError(Error):
    """Configuration mismatch.

    Raised when a cached script's metadata conflicts with the current
    Scraper configuration — e.g. the script was generated for a different
    domain than the current URL.
    """


class GenerationError(Error):
    """AI failed to generate a valid scraping function.

    Raised when the agent exhausts all retry attempts without producing
    a function that passes validation, or when the LLM API returns
    an unrecoverable error.
    """


class SandboxViolationError(Error):
    """Potential security threat detected in AI-generated code.

    Scout runs AI-generated scraping code in a sandbox that restricts
    access to the filesystem, network, and system utilities. This error
    is raised when the AI repeatedly generates code that violates these
    restrictions — which may indicate that the target website is
    attempting to manipulate the AI through prompt injection.

    The sandbox is enabled by default. If you trust the target website
    and believe this is a false positive, you can disable it::

        Scraper(..., sandbox=False)
    """


class AutoFixError(Error):
    """Auto-fix regenerated but the new script failed the same way.

    Raised when the auto-fix system regenerated a script, but the new
    script failed with the same error pattern as the old one. This
    indicates the problem is not the script — it's something
    regeneration cannot fix (unknown anti-bot, page genuinely doesn't
    have what the schema expects, etc.).

    The user should check the URL manually or adjust the task/schema.
    """


class ScriptError(Error):
    """Base class for saved-script execution failures.

    Catch this to handle any script failure regardless of cause.
    Catch a subclass for specific failure modes.
    """


class ScriptLoadError(ScriptError):
    """Saved script file is malformed.

    The file has a syntax error, is missing the ``scrape`` function,
    or is otherwise not valid Python. Fix the file manually or
    regenerate with ``scraper.regenerate()``.
    """


class ScriptRuntimeError(ScriptError):
    """Saved script crashed during execution.

    The script loaded and started running but raised an exception —
    typically because the website structure changed and selectors
    no longer match. Regenerate with ``scraper.regenerate()``.
    """


class ScriptTimeoutError(ScriptError):
    """Saved script exceeded the execution timeout.

    The script did not complete within the configured timeout.
    Increase ``timeout=`` or investigate why the script is slow.
    """


class ValidationError(Error):
    """Script output does not match the schema.

    The script ran successfully but returned data that fails schema
    validation. This can happen when a previously working script
    encounters a changed website, or when the schema was modified
    after the script was generated.
    """


# ── Backward compatibility aliases ──────────────────────────────
# These ensure existing code using Scout-prefixed names still works.

ScoutError = Error
ScoutSchemaError = SchemaError
ScoutConfigError = ConfigError
ScoutGenerationError = GenerationError
ScoutAutoFixError = AutoFixError
ScoutScriptError = ScriptError
ScoutScriptLoadError = ScriptLoadError
ScoutScriptRuntimeError = ScriptRuntimeError
ScoutScriptTimeoutError = ScriptTimeoutError
ScoutValidationError = ValidationError
