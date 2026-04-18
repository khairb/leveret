"""Scout exception hierarchy.

All Scout exceptions inherit from ScoutError, allowing broad catching
with ``except ScoutError`` or specific catching with subclasses.

Hierarchy::

    ScoutError
    ├── ScoutSchemaError
    ├── ScoutConfigError
    ├── ScoutGenerationError
    ├── ScoutScriptError
    │   ├── ScoutScriptLoadError
    │   ├── ScoutScriptRuntimeError
    │   └── ScoutScriptTimeoutError
    └── ScoutValidationError
"""


class ScoutError(Exception):
    """Base exception for all Scout errors."""


class ScoutSchemaError(ScoutError):
    """Invalid schema definition.

    Raised at schema construction time when the user provides an invalid
    schema — e.g. ``Field(str, min=5)`` (min doesn't apply to str),
    or a list schema with more than one element.
    """


class ScoutConfigError(ScoutError):
    """Configuration mismatch.

    Raised when a cached script's metadata conflicts with the current
    Scraper configuration — e.g. the script was generated for a different
    domain than the current URL.
    """


class ScoutGenerationError(ScoutError):
    """AI failed to generate a valid scraping function.

    Raised when the agent exhausts all retry attempts without producing
    a function that passes validation, or when the Anthropic API returns
    an unrecoverable error.
    """


class ScoutScriptError(ScoutError):
    """Base class for saved-script execution failures.

    Catch this to handle any script failure regardless of cause.
    Catch a subclass for specific failure modes.
    """


class ScoutScriptLoadError(ScoutScriptError):
    """Saved script file is malformed.

    The file has a syntax error, is missing the ``scrape`` function,
    or is otherwise not valid Python. Fix the file manually or
    regenerate with ``force=True``.
    """


class ScoutScriptRuntimeError(ScoutScriptError):
    """Saved script crashed during execution.

    The script loaded and started running but raised an exception —
    typically because the website structure changed and selectors
    no longer match. Regenerate with ``force=True``.
    """


class ScoutScriptTimeoutError(ScoutScriptError):
    """Saved script exceeded the execution timeout.

    The script did not complete within the configured timeout.
    Increase ``timeout=`` or investigate why the script is slow.
    """


class ScoutValidationError(ScoutError):
    """Script output does not match the schema.

    The script ran successfully but returned data that fails schema
    validation. This can happen when a previously working script
    encounters a changed website, or when the schema was modified
    after the script was generated.
    """
