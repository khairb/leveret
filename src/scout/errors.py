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

    Attributes:
        status_code: HTTP status code from the API, if the failure was
            an API error. ``None`` for non-API failures (agent exhausted
            retries, connection error, etc.).
        is_transient: ``True`` when the failure is likely temporary
            (rate limit, server error) and retrying may succeed.
            ``False`` for permanent failures (auth error, agent failure).
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code

    @property
    def is_transient(self) -> bool:
        """Whether retrying the same call may succeed."""
        if self.status_code is None:
            return False
        return self.status_code == 429 or self.status_code >= 500


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


class AutoRegenerateError(Error):
    """Auto-regeneration produced a script that failed the same way.

    Raised when auto-regeneration replaced a broken script, but the new
    script failed with the same error pattern as the old one. This
    indicates the problem is not the script — it's something
    regeneration cannot fix (unknown anti-bot, page genuinely doesn't
    have what the schema expects, etc.).

    The user should check the URL manually or adjust the task/schema.
    """


# Backward compatibility alias.
AutoFixError = AutoRegenerateError


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
ScoutAutoFixError = AutoRegenerateError
ScoutAutoRegenerateError = AutoRegenerateError
ScoutScriptError = ScriptError
ScoutScriptLoadError = ScriptLoadError
ScoutScriptRuntimeError = ScriptRuntimeError
ScoutScriptTimeoutError = ScriptTimeoutError
ScoutValidationError = ValidationError
