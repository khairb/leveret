"""Scout public API — Scraper class and ScraperResult."""

from __future__ import annotations

import asyncio
import ast
import errno
import hashlib
import inspect
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from .errors import (
    AutoFixError,
    ConfigError,
    Error,
    GenerationError,
    SchemaError,
    ScriptLoadError,
    ScriptRuntimeError,
    ScriptTimeoutError,
    ValidationError,
)
from .agent.llm import ModelName
from .schema.compiler import compile_schema
from .schema.tolerance import Tolerance
from ._logging import logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from .schema.compiler import CompiledSchema

from .autofix.types import AutoFixMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _preview(data: Any) -> str:
    """Format a short preview of data for repr."""
    if isinstance(data, list):
        return f"items={len(data)}"
    elif isinstance(data, dict):
        keys = list(data.keys())[:3]
        return f"keys={keys}"
    else:
        s = repr(data)
        if len(s) > 50:
            s = s[:50] + "..."
        return f"data={s}"


def _normalize_domain(url: str) -> str:
    """Extract and normalize a URL's hostname for domain comparison.

    Strips ``www.`` prefix and lowercases. Only ``www.`` is stripped —
    other subdomains (``m.``, ``api.``, ``app.``) are treated as
    different sites.
    """
    host = urlparse(url).hostname or ""
    if host.startswith("www."):
        host = host[4:]
    return host.lower()


# ---------------------------------------------------------------------------
# Metadata escaping
# ---------------------------------------------------------------------------

def _escape_metadata(value: str) -> str:
    """Escape a string for inclusion in the metadata docstring.

    Handles: triple-quotes, backslashes, newlines.
    """
    value = value.replace("\\", "\\\\")
    value = value.replace('"""', '\\"\\"\\"')
    value = value.replace("\n", "\\n")
    return value


def _unescape_metadata(value: str) -> str:
    """Reverse of :func:`_escape_metadata`."""
    value = value.replace("\\n", "\n")
    value = value.replace('\\"\\"\\"', '"""')
    value = value.replace("\\\\", "\\")
    return value


# ---------------------------------------------------------------------------
# Pre-import namespace for in-process execution
# ---------------------------------------------------------------------------

def _build_pre_import_namespace(*, sandbox: bool = False) -> dict[str, Any]:
    """Build namespace with pre-imported modules for in-process execution.

    Matches the pre-imports from the subprocess wrapper template so that
    agent-authored functions work identically in both execution modes.

    When ``sandbox=True``, excludes ``os``, ``shutil``, ``tempfile`` and
    uses the safe asyncio proxy.
    """
    if sandbox:
        from .runtime.sandbox import build_safe_pre_imports
        return build_safe_pre_imports()

    import math
    from urllib.parse import urljoin

    return {
        "json": json,
        "re": re,
        "math": math,
        "os": os,
        "time": time,
        "asyncio": asyncio,
        "tempfile": tempfile,
        "shutil": shutil,
        "datetime": datetime,
        "urljoin": urljoin,
        "urlparse": urlparse,
    }


# ---------------------------------------------------------------------------
# Script version
# ---------------------------------------------------------------------------

def _get_scout_version() -> str:
    """Get the current scout package version."""
    try:
        from importlib.metadata import version
        return version("scout")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Script save / load
# ---------------------------------------------------------------------------

def _build_metadata_docstring(
    url: str, task: str, model: str, timestamp: str,
    schema_hash: str = "",
    content_hash: str = "",
) -> str:
    """Build the metadata docstring for a saved script file."""
    return (
        '"""\n'
        "Scout Script\n"
        "\n"
        f"url:           {_escape_metadata(url)}\n"
        f"task:          {_escape_metadata(task)}\n"
        f"generated:     {timestamp}\n"
        f"model:         {model}\n"
        f"scout_version: {_get_scout_version()}\n"
        + (f"schema_hash:   {schema_hash}\n" if schema_hash else "")
        + (f"content_hash:  {content_hash}\n" if content_hash else "")
        + '"""\n'
    )


_METADATA_RE = re.compile(
    r'^""".*?^"""',
    re.MULTILINE | re.DOTALL,
)

_FIELD_RE = re.compile(r"^(\w+):\s+(.+)$", re.MULTILINE)


def _parse_script_metadata(source: str) -> dict[str, str]:
    """Extract metadata fields from a script's docstring.

    Returns a dict with keys: url, task, generated, model, scout_version.
    Missing fields map to empty strings.
    """
    m = _METADATA_RE.search(source)
    if m is None:
        return {}

    docstring = m.group(0)
    fields: dict[str, str] = {}
    for fm in _FIELD_RE.finditer(docstring):
        key = fm.group(1).strip()
        val = fm.group(2).strip()
        if key in ("url", "task"):
            val = _unescape_metadata(val)
        fields[key] = val
    return fields


def _save_script(
    code: str,
    path: Path,
    url: str,
    task: str,
    model: str,
    schema_hash: str = "",
) -> None:
    """Write a script file with metadata docstring.

    Auto-creates parent directories. Wraps filesystem errors in
    Error with actionable messages.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    content_hash = hashlib.sha256(code.strip().encode()).hexdigest()[:16]
    docstring = _build_metadata_docstring(
        url, task, model, timestamp,
        schema_hash=schema_hash,
        content_hash=content_hash,
    )
    content = docstring + "\n" + code

    # Back up existing script before overwriting (best-effort).
    # This protects user edits during regenerate() and auto-fix.
    if path.exists():
        bak = path.with_suffix(".py.bak")
        try:
            shutil.copy2(path, bak)
        except OSError:
            pass

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        raise Error(
            f'Cannot create directory "{path.parent}" '
            f"— a file with that name already exists."
        ) from None
    except OSError as exc:
        raise Error(
            f'Could not write script to "{path}" — {_describe_os_error(exc)}.'
        ) from None

    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise Error(
            f'Could not write script to "{path}" — {_describe_os_error(exc)}.'
        ) from None


def _describe_os_error(exc: OSError) -> str:
    """Map an OSError to a human-readable description."""
    if exc.errno == errno.ENOSPC:
        return "disk full"
    if exc.errno == errno.EACCES:
        return "permission denied. Check directory permissions"
    if exc.errno == errno.EROFS:
        return "read-only filesystem"
    return str(exc)


def _load_script(
    path: Path, *, sandbox: bool = False,
) -> tuple[Callable[..., Any], dict[str, str]]:
    """Load a saved script file, validate it, and return the scrape function.

    Returns:
        (scrape_fn, metadata_dict)

    Raises:
        ScriptLoadError: If the file has syntax errors, is missing the
            scrape function, or has the wrong signature.
    """
    # Read the file
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        if exc.errno == errno.EACCES:
            raise ScriptLoadError(
                f'Could not read script "{path}" — permission denied.'
            ) from None
        raise ScriptLoadError(
            f'Could not read script "{path}" — {exc}.'
        ) from None

    if not source.strip():
        raise ScriptLoadError(
            f'Script "{path}" is empty. '
            f"Regenerate with scraper.regenerate()."
        )

    # Syntax check
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise ScriptLoadError(
            f'Script "{path}" has a syntax error: {exc.msg} '
            f"(line {exc.lineno}). Fix the file or "
            f"regenerate with scraper.regenerate()."
        ) from None

    # Find async def scrape
    scrape_func = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "scrape":
                scrape_func = node
                break

    if scrape_func is None:
        raise ScriptLoadError(
            f'Script "{path}" has no function named "scrape". '
            f"The file must define: async def scrape(page, start_url, checkpoint). "
            f"Fix the file or regenerate with scraper.regenerate()."
        )

    if not isinstance(scrape_func, ast.AsyncFunctionDef):
        raise ScriptLoadError(
            f'Script "{path}" defines "scrape" as a sync function. '
            f"It must be async: async def scrape(page, start_url, checkpoint). "
            f"Fix the file or regenerate with scraper.regenerate()."
        )

    # Check signature: exactly (page, start_url, checkpoint).
    # Also accept the legacy (page, url, checkpoint) for backward compat.
    _ACCEPTED_PARAMS = [
        ["page", "start_url", "checkpoint"],
        ["page", "url", "checkpoint"],
    ]
    actual_params = [arg.arg for arg in scrape_func.args.args]
    if actual_params not in _ACCEPTED_PARAMS:
        raise ScriptLoadError(
            f'Script "{path}" has wrong signature for scrape(). '
            f"Expected: scrape(page, start_url, checkpoint). "
            f"Got: scrape({', '.join(actual_params)}). "
            f"Fix the file or regenerate with scraper.regenerate()."
        )

    # Execute the source in an isolated namespace. We use compile+exec
    # instead of importlib to avoid bytecode caching issues when the same
    # file path is overwritten with new content (e.g. auto_fix='always').
    if sandbox:
        from .runtime.sandbox import (
            compile_restricted_agent_code,
            build_restricted_globals,
            build_safe_pre_imports,
            SandboxError,
        )
        try:
            compiled_code = compile_restricted_agent_code(
                source, filename=str(path),
            )
        except SandboxError as exc:
            raise ScriptLoadError(
                f'Script "{path}" failed sandbox validation: {exc}'
            ) from None
        namespace = build_restricted_globals(build_safe_pre_imports())
        namespace["__file__"] = str(path)
    else:
        compiled_code = compile(source, str(path), "exec")
        namespace = {
            "__file__": str(path),
            **_build_pre_import_namespace(),
        }

    try:
        exec(compiled_code, namespace)  # noqa: S102
    except Exception as exc:
        raise ScriptLoadError(
            f'Script "{path}" failed to load: {exc}. '
            f"Fix the file or regenerate with scraper.regenerate()."
        ) from None

    fn = namespace.get("scrape")
    if fn is None or not callable(fn):
        raise ScriptLoadError(
            f'Script "{path}" does not export a callable "scrape" function.'
        )

    if not inspect.iscoroutinefunction(fn):
        raise ScriptLoadError(
            f'Script "{path}": scrape is not async. '
            f"It must be: async def scrape(page, start_url, checkpoint)."
        )

    metadata = _parse_script_metadata(source)
    return fn, metadata


# ---------------------------------------------------------------------------
# Config mismatch detection
# ---------------------------------------------------------------------------

def _check_domain_mismatch(
    script_path: Path,
    script_url: str,
    current_url: str,
) -> None:
    """Raise ConfigError if script was generated for a different domain."""
    script_domain = _normalize_domain(script_url)
    current_domain = _normalize_domain(current_url)

    if not script_domain or not current_domain:
        return  # can't compare — skip

    if script_domain != current_domain:
        raise ConfigError(
            f"Script '{script_path}' was generated for a different site.\n"
            f"\n"
            f"  Script:  {script_url}\n"
            f"  Current: {current_url}\n"
            f"\n"
            f"  To regenerate for the new site: scraper.regenerate()\n"
            f'  To use a different script file:  script="<new_path>.py"'
        )


def _check_task_mismatch(script_task: str, current_task: str) -> None:
    """Log a warning if the task description changed since generation."""
    if script_task and script_task != current_task:
        logger.warning(
            "Note: Task description has changed since script was generated.\n"
            "        Using cached script. To regenerate: scraper.regenerate()"
        )


def _is_script_user_edited(path: Path) -> bool:
    """Check if a script file was edited after Scout generated it.

    Compares the current function code hash against the content_hash
    stored in the script's metadata. Returns False if no hash is stored
    (pre-Round-3 scripts) or if the file can't be read.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return False

    metadata = _parse_script_metadata(source)
    stored_hash = metadata.get("content_hash", "")
    if not stored_hash:
        return False  # No hash stored — can't compare

    # Extract function code (everything after the metadata docstring)
    m = _METADATA_RE.search(source)
    if m:
        function_code = source[m.end():].strip()
    else:
        function_code = source

    current_hash = hashlib.sha256(function_code.encode()).hexdigest()[:16]
    return current_hash != stored_hash


# ---------------------------------------------------------------------------
# ScraperResult
# ---------------------------------------------------------------------------

@dataclass
class ScraperResult:
    """The result of a successful scraping run.

    Attributes:
        data: Validated Python data matching the schema.
        url: The URL that was scraped.
        timestamp: ISO 8601 UTC with ``Z`` suffix,
            e.g. ``"2024-03-15T10:30:00.123456Z"``.
        cached: ``True`` if loaded from a saved script on disk,
            ``False`` if freshly generated by the AI agent.
        script_path: Absolute path to the script file, or ``None``
            if ``script=`` was not set on the Scraper.
        auto_fixed: ``True`` when auto-fix triggered regeneration.
            Implies ``cached=False`` (a fresh script was generated).
    """

    data: Any
    url: str
    timestamp: str
    cached: bool
    script_path: str | None
    auto_fixed: bool = False

    def __repr__(self) -> str:
        parts = [
            f"ScraperResult("
            f"url={self.url!r}, "
            f"{_preview(self.data)}, "
            f"cached={self.cached}",
        ]
        if self.auto_fixed:
            parts.append(", auto_fixed=True")
        parts.append(")")
        return "".join(parts)


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class Scraper:
    """AI-powered web scraper — generate once, run forever.

    The AI agent is a build step: the first ``run()`` generates a scraping
    function, validates it against the schema, and (optionally) saves it to
    disk. Every subsequent ``run()`` loads the saved function and executes
    it — no AI, no API cost.
    """

    def __init__(
        self,
        url: str,
        task: str,
        *,
        schema: Any,
        tolerance: str | Tolerance = "balanced",
        script: str | Path | None = None,
        model: ModelName = "claude-haiku-4-5",
        headless: bool = True,
        api_key: str | None = None,
        timeout: int = 600,
        max_attempts: int = 6,
        auto_fix: bool | str = False,
        sandbox: bool = True,
        protect_script: bool = False,
        launch_options: dict | None = None,
    ) -> None:
        # -- url --
        if not isinstance(url, str) or not url.strip():
            raise Error(f'url must be a valid HTTP(S) URL (got {url!r})')
        url = url.strip()
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            # Detect missing scheme — "example.com" instead of "https://example.com"
            if not parsed.scheme and "." in url:
                raise Error(
                    f"url must start with https:// or http://\n\n"
                    f"  Got:          {url!r}\n"
                    f"  Did you mean: 'https://{url}'?"
                )
            raise Error(
                f"url must start with https:// or http:// (got {url!r})"
            )
        if not parsed.hostname:
            raise Error(f'url must include a hostname (got {url!r})')

        # -- task --
        if not isinstance(task, str) or not task.strip():
            raise Error("task must not be empty")

        # -- schema --
        if schema is None:
            raise SchemaError(
                "schema is required.\n\n"
                "  Quick examples:\n"
                "    schema={'title': str, 'price': float}       # single object\n"
                "    schema=[{'title': str, 'price': float}]     # list of objects\n"
                "    schema=Items({'title': str}, min=10)         # list with constraints"
            )
        compiled: CompiledSchema = compile_schema(schema)

        # -- tolerance --
        resolved_tolerance = self._resolve_tolerance_value(tolerance)

        # -- script (path normalization) --
        script_path: Path | None = None
        if script is not None:
            path = Path(script).expanduser().resolve()
            if path.is_dir():
                raise Error(
                    f'script must be a file path, not a directory '
                    f'(got {str(script)!r})'
                )
            if path.suffix != ".py":
                raise Error(
                    f'script must be a .py file path (got {str(script)!r})'
                )
            script_path = path

        # -- timeout --
        if not isinstance(timeout, int) or timeout <= 0:
            raise Error(
                f"timeout must be a positive integer (got {timeout!r})"
            )

        # -- max_attempts --
        if not isinstance(max_attempts, int) or max_attempts < 1:
            raise Error(
                f"max_attempts must be at least 1 (got {max_attempts!r})"
            )

        # -- model --
        if not isinstance(model, str) or not model.strip():
            raise Error("model must not be empty")

        # Detect common model names used without the provider: prefix.
        # e.g. "gpt-4o" → should be "openai:gpt-4o"
        if ":" not in model:
            suggestion = self._suggest_model_provider(model)
            if suggestion:
                raise ConfigError(
                    f"Model {model!r} looks like it belongs to "
                    f"{suggestion[0]}.\n\n"
                    f"  Use the provider:model format:\n"
                    f'    model="{suggestion[1]}"'
                )

        # -- auto_fix --
        auto_fix_mode = None
        if auto_fix is not False:
            auto_fix_mode = self._resolve_auto_fix_value(auto_fix)

            # §3/§11: auto_fix without script= has no effect
            # (except "always" which forces regeneration regardless)
            if script_path is None and auto_fix_mode != AutoFixMode.ALWAYS:
                logger.warning(
                    "auto_fix has no effect without script= — "
                    "every run generates from scratch."
                )
                auto_fix_mode = None

        # -- Store validated state --
        self._url = url
        self._task = task
        self._compiled_schema = compiled
        self._tolerance = resolved_tolerance
        self._script_path = script_path
        self._model = model
        self._headless = headless
        self._api_key = api_key
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._auto_fix_mode = auto_fix_mode
        self._sandbox = sandbox
        self._protect_script = protect_script
        self._launch_options = launch_options

        # -- Runtime state --
        self._cached_fn: Any = None
        self._cached_source: str | None = None
        self._schema_changed: bool = False
        self._context_managed: bool = False
        self._browser_mgr: Any = None
        self._bg_loop: Any = None
        self._bg_thread: threading.Thread | None = None
        self._cm_page_count: int = 0
        self._cm_start_time: float = 0.0

        # -- Developer hints --
        if script_path is None:
            logger.warning(
                "No script= path set — each run() will generate a new "
                "script from scratch (costs API credits every time). "
                "Add script='scrapers/my_scraper.py' to cache it."
            )

        # -- Fail-fast: check browser at construction time --
        # The browser check is a one-time setup issue (install command).
        # Checking early surfaces this before the developer waits for a
        # run() call. The API key check stays at run() time because the
        # key may be set between construction and execution.
        if script_path is None or not script_path.exists():
            self._check_playwright()

    # -- Public properties --

    @property
    def url(self) -> str:
        """The target URL."""
        return self._url

    @property
    def task(self) -> str:
        """The task description."""
        return self._task

    @property
    def script_path(self) -> Path | None:
        """Absolute path to the script file, or None."""
        return self._script_path

    @property
    def has_script(self) -> bool:
        """Whether a cached script exists (on disk or in memory).

        When ``True``, the next ``run()`` will use the cached script
        — no AI, no API cost.  When ``False``, the next ``run()``
        will generate a new script via the AI agent.
        """
        return self._cached_fn is not None or self._has_script_on_disk()

    def __repr__(self) -> str:
        parts = [f"Scraper({self._url!r}"]
        if self._script_path:
            parts.append(f", script={str(self._script_path)!r}")
            if self._cached_fn is not None:
                parts.append(", cached=True")
        parts.append(")")
        return "".join(parts)

    def _get_schema_hash(self) -> str:
        """Compute a short hash of the compiled schema prompt.

        Used to detect when the user changed the schema after generation.
        """
        return hashlib.sha256(
            self._compiled_schema.prompt.encode()
        ).hexdigest()[:16]

    def _get_resolved_launch_options(self) -> dict:
        """Resolve launch options, merging user options with stealth defaults."""
        from .browser import resolve_launch_options
        return resolve_launch_options(
            self._launch_options, headless=self._headless,
        )

    # -- Auto-fix mode resolution --

    # Mapping from user-facing auto_fix values to AutoFixMode enums.
    _VALID_AUTO_FIX = {
        True: AutoFixMode.BALANCED,
        "balanced": AutoFixMode.BALANCED,
        "conservative": AutoFixMode.CONSERVATIVE,
        "aggressive": AutoFixMode.AGGRESSIVE,
        "always": AutoFixMode.ALWAYS,
        "regenerate": AutoFixMode.ALWAYS,
    }

    # Well-known model prefixes → (provider display name, provider prefix).
    _MODEL_PROVIDER_HINTS: list[tuple[list[str], str, str]] = [
        (["gpt-", "o1-", "o3-", "o4-"], "OpenAI", "openai"),
        (["gemini-", "gemma-"], "Google", "google-gla"),
        (["llama-", "mixtral-", "llama3"], "Groq", "groq"),
        (["mistral-", "codestral-", "pixtral-"], "Mistral", "mistral"),
        (["deepseek-"], "DeepSeek", "deepseek"),
        (["command-"], "Cohere", "cohere"),
    ]

    @staticmethod
    def _suggest_model_provider(model: str) -> tuple[str, str] | None:
        """Check if a model name matches a known non-Anthropic provider.

        Returns (provider_name, suggested_format) or None.
        """
        lower = model.lower()
        for prefixes, provider_name, provider_prefix in Scraper._MODEL_PROVIDER_HINTS:
            for prefix in prefixes:
                if lower.startswith(prefix):
                    return (provider_name, f"{provider_prefix}:{model}")
        return None

    @staticmethod
    def _resolve_auto_fix_value(value: bool | str | AutoFixMode) -> AutoFixMode:
        """Convert a user-facing auto_fix value to an AutoFixMode enum.

        Raises ConfigError for invalid values.
        """
        # Accept AutoFixMode enum values directly
        if isinstance(value, AutoFixMode):
            return value
        mode = Scraper._VALID_AUTO_FIX.get(value)
        if mode is None:
            raise ConfigError(
                f"auto_fix must be False, True, 'conservative', "
                f"'balanced', 'aggressive', 'always', or "
                f"'regenerate' (got {value!r})"
            )
        return mode

    # Mapping from user-facing tolerance values to Tolerance enums.
    _VALID_TOLERANCE: dict[str, Tolerance] = {
        "strict": Tolerance.STRICT,
        "balanced": Tolerance.BALANCED,
        "tolerant": Tolerance.TOLERANT,
    }

    @staticmethod
    def _resolve_tolerance_value(value: str | Tolerance) -> Tolerance:
        """Convert a user-facing tolerance value to a Tolerance enum.

        Raises ConfigError for invalid values.
        """
        if isinstance(value, Tolerance):
            return value
        mode = Scraper._VALID_TOLERANCE.get(value)
        if mode is None:
            raise ConfigError(
                f"tolerance must be 'strict', 'balanced', or 'tolerant' "
                f"(got {value!r})"
            )
        return mode

    # -- Context manager --

    def __enter__(self) -> Scraper:
        if self._context_managed:
            raise Error(
                "Scraper is already inside a 'with' block — "
                "nested context managers are not supported."
            )
        self._context_managed = True
        self._cm_start_time = time.monotonic()
        self._cm_page_count = 0
        # Persistent event loop in a background thread so that the shared
        # browser stays alive across multiple sync run() calls.
        self._bg_loop = asyncio.new_event_loop()
        self._bg_thread = threading.Thread(
            target=self._bg_loop.run_forever,
            daemon=True,
            name="scout-browser",
        )
        self._bg_thread.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        try:
            self._sync_close_browser()
        finally:
            if self._bg_loop is not None:
                self._bg_loop.call_soon_threadsafe(self._bg_loop.stop)
            if self._bg_thread is not None:
                self._bg_thread.join(timeout=10)
            if self._bg_loop is not None:
                self._bg_loop.close()
            self._bg_loop = None
            self._bg_thread = None
            self._context_managed = False
        return False

    async def __aenter__(self) -> Scraper:
        if self._context_managed:
            raise Error(
                "Scraper is already inside a 'with' block — "
                "nested context managers are not supported."
            )
        self._context_managed = True
        self._cm_start_time = time.monotonic()
        self._cm_page_count = 0
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        try:
            await self._close_browser()
        finally:
            self._context_managed = False
        return False

    def close(self) -> None:
        """Close the browser and release resources. Idempotent."""
        if self._bg_loop is not None:
            self._sync_close_browser()
            self._bg_loop.call_soon_threadsafe(self._bg_loop.stop)
            if self._bg_thread is not None:
                self._bg_thread.join(timeout=10)
            self._bg_loop.close()
            self._bg_loop = None
            self._bg_thread = None
        elif self._browser_mgr is not None:
            # Async-only case (user didn't use async with but has a browser)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                loop.create_task(self._close_browser())
            else:
                asyncio.run(self._close_browser())
        self._context_managed = False

    def __del__(self) -> None:
        if getattr(self, "_browser_mgr", None) is not None:
            warnings.warn(
                "Scraper was garbage-collected with an open browser. "
                "Use 'with scraper:' or call 'scraper.close()' "
                "to avoid resource leaks.",
                ResourceWarning,
                stacklevel=2,
            )

    # -- Public methods --

    async def async_run(
        self,
        *args: Any,
        url: str | None = None,
        auto_fix: bool | str | None = None,
    ) -> ScraperResult:
        """Generate (if needed) and execute the scraping function.

        Args:
            url: Override the constructor URL for this execution only.
                 Must be on the same domain as the original URL.
            auto_fix: Override the constructor's auto_fix setting for
                this run only. Pass ``"always"`` to force regeneration
                (discards any cached script).

        Returns:
            ScraperResult with validated data.
        """
        if args:
            hint = args[0]
            raise Error(
                f"run() does not accept positional arguments.\n\n"
                f"  You wrote:    scraper.run({hint!r})\n"
                f"  Did you mean: scraper.run(url={hint!r})"
            )
        effective_url = self._resolve_url(url)
        start_time = time.monotonic()

        # Resolve effective auto_fix mode for this call
        if auto_fix is not None:
            if auto_fix is False:
                effective_mode = None
            else:
                effective_mode = self._resolve_auto_fix_value(auto_fix)
        else:
            effective_mode = self._auto_fix_mode

        force_regenerate = (effective_mode == AutoFixMode.ALWAYS)

        # -- Cached path: load from disk or memory --
        if not force_regenerate and (self._cached_fn is not None or self._has_script_on_disk()):
            return await self._run_cached(effective_url, effective_mode)

        # -- Generation path --
        if force_regenerate and self._script_path and self._script_path.exists():
            # Warn if user edited the script (protection is checked in
            # regenerate(); this covers run(auto_fix="always") directly)
            if _is_script_user_edited(self._script_path):
                if self._protect_script:
                    raise ConfigError(
                        f"Script '{self._script_path}' has been manually "
                        f"edited and protect_script=True.\n\n"
                        f"  To overwrite:  scraper.regenerate(force=True)\n"
                        f"  To keep edits: set protect_script=False"
                    )
                logger.warning(
                    "Script '%s' has been manually edited. "
                    "Regenerating will overwrite your changes. "
                    "A backup will be saved to '%s.bak'.",
                    self._script_path,
                    self._script_path,
                )
            logger.info(
                "Regenerating script (overwriting %s)", self._script_path
            )
        elif self._script_path:
            logger.info(
                "No cached script found. Generating with %s...",
                self._model,
            )
            logger.info(
                "Note: First run calls the AI model API. "
                "Subsequent runs use the cached script."
            )
        else:
            logger.info(
                "Generating scraping function with %s...", self._model
            )

        # Clear in-memory cache when regenerating
        if force_regenerate:
            self._cached_fn = None
            self._cached_source = None
            self._schema_changed = False

        self._check_api_key()
        self._check_playwright()

        return await self._run_generate(effective_url, start_time)

    def run(
        self,
        *args: Any,
        url: str | None = None,
        auto_fix: bool | str | None = None,
    ) -> ScraperResult:
        """Synchronous version of :meth:`async_run`.

        Inside a ``with scraper:`` block, dispatches to the background
        event loop where the shared browser lives.  Outside a ``with``
        block, uses ``asyncio.run()``.

        Raises Error if called inside a running event loop
        (e.g. Jupyter) without a context manager.
        Use ``await scraper.async_run()`` instead.
        """
        if args:
            hint = args[0]
            raise Error(
                f"run() does not accept positional arguments.\n\n"
                f"  You wrote:    scraper.run({hint!r})\n"
                f"  Did you mean: scraper.run(url={hint!r})"
            )
        # Context-managed: dispatch to the background loop that owns
        # the shared browser.
        if self._context_managed and self._bg_loop is not None:
            future = asyncio.run_coroutine_threadsafe(
                self.async_run(url=url, auto_fix=auto_fix),
                self._bg_loop,
            )
            return future.result()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # Inside a running event loop (Jupyter, async server, etc.)
            # Try nest_asyncio to make run() work seamlessly in notebooks.
            try:
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(
                    self.async_run(url=url, auto_fix=auto_fix)
                )
            except ImportError:
                raise Error(
                    "scraper.run() cannot be used inside a running event "
                    "loop (e.g. Jupyter notebook, async web server).\n\n"
                    "  Option 1 — install nest-asyncio (recommended for "
                    "notebooks):\n"
                    "    pip install nest-asyncio\n"
                    "    # Then scraper.run() will work automatically.\n\n"
                    "  Option 2 — use the async API:\n"
                    "    result = await scraper.async_run()"
                ) from None

        return asyncio.run(self.async_run(url=url, auto_fix=auto_fix))

    async def async_regenerate(
        self, *, url: str | None = None, force: bool = False,
    ) -> ScraperResult:
        """Force-regenerate the scraping script, discarding any cached version.

        Args:
            url: Override the constructor URL for this execution only.
            force: Override ``protect_script`` for this call. Required
                when ``protect_script=True`` and the script has been
                edited by the user.

        Equivalent to ``async_run(auto_fix="always")``.
        """
        self._check_script_protection(force=force)
        return await self.async_run(url=url, auto_fix="always")

    def regenerate(
        self, *, url: str | None = None, force: bool = False,
    ) -> ScraperResult:
        """Force-regenerate the scraping script, discarding any cached version.

        Args:
            url: Override the constructor URL for this execution only.
            force: Override ``protect_script`` for this call. Required
                when ``protect_script=True`` and the script has been
                edited by the user.

        Equivalent to ``run(auto_fix="always")``.
        """
        self._check_script_protection(force=force)
        return self.run(url=url, auto_fix="always")

    def export(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
    ) -> None:
        """Export a standalone runnable script for debugging.

        The exported file launches a browser, calls the scrape function,
        and pretty-prints the result as JSON.

        Args:
            path: Target ``.py`` file path. Parent directories are
                created automatically.
            overwrite: Allow overwriting an existing file.
        """
        if self._script_path is None:
            raise Error(
                "Cannot export — Scraper has no script= path. "
                "Generate a script first."
            )
        if not self._script_path.exists():
            raise Error(
                f"Cannot export — script not found at {self._script_path}. "
                "Run scraper.run() to generate it first."
            )

        target = Path(path).expanduser().resolve()
        if target.suffix != ".py":
            raise Error(
                f"Export path must be a .py file (got {str(path)!r})"
            )
        if target.exists() and not overwrite:
            raise Error(
                f"File already exists: {target}. "
                "Pass overwrite=True to replace it."
            )

        from .agent.wrapper import generate_standalone_script

        function_source = self._get_function_source()
        standalone = generate_standalone_script(
            function_source, self._url, self._task,
        )

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise Error(
                f'Could not create directory "{target.parent}" — '
                f"{_describe_os_error(exc)}."
            ) from None
        try:
            target.write_text(standalone, encoding="utf-8")
        except OSError as exc:
            raise Error(
                f'Could not write export to "{target}" — '
                f"{_describe_os_error(exc)}."
            ) from None

        logger.info("Exported standalone script → %s", target)

    # -- Internal: URL resolution --

    def _resolve_url(self, override_url: str | None) -> str:
        """Validate and return the effective URL for this run."""
        if override_url is None:
            return self._url

        if not isinstance(override_url, str) or not override_url.strip():
            raise Error(
                f"url must be a valid HTTP(S) URL (got {override_url!r})"
            )
        override_url = override_url.strip()
        parsed = urlparse(override_url)
        if parsed.scheme not in ("http", "https"):
            if not parsed.scheme and "." in override_url:
                raise Error(
                    f"url must start with https:// or http://\n\n"
                    f"  Got:          {override_url!r}\n"
                    f"  Did you mean: 'https://{override_url}'?"
                )
            raise Error(
                f"url must start with https:// or http:// (got {override_url!r})"
            )
        if not parsed.hostname:
            raise Error(
                f"url must include a hostname (got {override_url!r})"
            )
        return override_url

    # -- Internal: disk check --

    def _has_script_on_disk(self) -> bool:
        """Check if a script file exists on disk."""
        return self._script_path is not None and self._script_path.exists()

    # -- Internal: prerequisite checks --

    # Provider name → environment variable for API key.
    # Covers the top providers bundled in ``pip install scout``.
    # Unknown providers fall back to ``{PROVIDER}_API_KEY``.
    _API_KEY_ENV_VARS: dict[str, str] = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google-gla": "GOOGLE_API_KEY",
        "google-vertex": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "cohere": "CO_API_KEY",
        "bedrock": "AWS_ACCESS_KEY_ID",
        "deepseek": "DEEPSEEK_API_KEY",
        "together": "TOGETHER_API_KEY",
        "fireworks": "FIREWORKS_API_KEY",
    }

    def _check_script_protection(self, *, force: bool = False) -> None:
        """Check if the script is protected from overwriting.

        Raises ConfigError if ``protect_script=True`` and the script
        exists and was edited by the user, unless ``force=True``.
        Always warns (regardless of protect_script) if user edits are
        detected.
        """
        if self._script_path is None or not self._script_path.exists():
            return

        if not _is_script_user_edited(self._script_path):
            return

        if self._protect_script and not force:
            raise ConfigError(
                f"Script '{self._script_path}' has been manually edited "
                f"and protect_script=True.\n\n"
                f"  To overwrite anyway:  scraper.regenerate(force=True)\n"
                f"  To keep your edits:   set protect_script=False or "
                f"edit the script directly"
            )

        logger.warning(
            "Script '%s' has been manually edited since generation. "
            "Regenerating will overwrite your changes. "
            "A backup will be saved to '%s.bak'.",
            self._script_path,
            self._script_path,
        )

    def _check_api_key(self) -> None:
        """Verify an API key is available before generation."""
        if self._api_key:
            return

        # Determine which env var to check based on the model provider.
        provider = (
            self._model.split(":")[0]
            if ":" in self._model
            else "anthropic"
        )
        env_var = self._API_KEY_ENV_VARS.get(
            provider,
            f"{provider.upper().replace('-', '_')}_API_KEY",
        )
        if os.environ.get(env_var):
            return

        raise ConfigError(
            f"API key not found for provider '{provider}'.\n\n"
            f"  Set the environment variable:\n"
            f"    export {env_var}=...\n\n"
            f"  Or pass it directly:\n"
            f'    Scraper(..., api_key="...")'
        )

    def _check_playwright(self) -> None:
        """Verify Patchright package and browser binaries are installed."""
        try:
            import patchright  # noqa: F401
        except ImportError:
            raise ConfigError(
                "Patchright is not installed.\n\n"
                "  Run:\n"
                "    pip install patchright\n"
                "    patchright install chromium"
            ) from None

        # Check that the browser binary is actually installed, not just
        # the Python package.  Use the CLI's --dry-run to find the
        # expected install location, then verify it exists.
        try:
            import subprocess
            from patchright._impl._driver import compute_driver_executable
            node, cli_js = compute_driver_executable()
            result = subprocess.run(
                [str(node), str(cli_js), "install", "--dry-run", "chromium"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and "Install location:" in result.stdout:
                # Parse the first install location line
                for line in result.stdout.splitlines():
                    if "Install location:" in line:
                        install_dir = Path(line.split(":", 1)[1].strip())
                        marker = install_dir / "INSTALLATION_COMPLETE"
                        if not install_dir.exists() or not marker.exists():
                            raise ConfigError(
                                "Chromium browser is not installed.\n\n"
                                "  Patchright is installed, but the browser "
                                "binary is missing.\n\n"
                                "  Run:\n"
                                "    patchright install chromium\n\n"
                                "  This downloads Chromium (~150 MB, "
                                "one-time setup)."
                            )
                        break  # First location (chromium) is enough
        except Error:
            raise
        except Exception:
            # Can't verify — proceed and let the browser launch fail
            # with its own error if the binary is truly missing.
            pass

    # -- Internal: browser lifecycle --

    async def _close_browser(self) -> None:
        """Close the shared browser. Idempotent."""
        if self._browser_mgr is not None:
            elapsed = time.monotonic() - self._cm_start_time
            count = self._cm_page_count
            try:
                await self._browser_mgr.stop()
            except Exception:
                pass
            self._browser_mgr = None
            if count > 0:
                logger.info(
                    "Browser closed (scraped %d pages in %ds)",
                    count, int(elapsed),
                )

    def _sync_close_browser(self) -> None:
        """Close browser from sync context via the background loop."""
        if self._browser_mgr is not None and self._bg_loop is not None:
            future = asyncio.run_coroutine_threadsafe(
                self._close_browser(), self._bg_loop,
            )
            try:
                future.result(timeout=10)
            except Exception:
                pass

    # -- Internal: cached execution path --

    async def _run_cached(
        self,
        effective_url: str,
        auto_fix_mode: AutoFixMode | None = None,
    ) -> ScraperResult:
        """Execute a cached scraping function."""
        # Load from disk if not in memory
        if self._cached_fn is None:
            assert self._script_path is not None
            fn, metadata = _load_script(
                self._script_path, sandbox=self._sandbox,
            )

            # Config mismatch checks
            script_url = metadata.get("url", "")
            if script_url:
                _check_domain_mismatch(
                    self._script_path, script_url, effective_url,
                )
            _check_task_mismatch(
                metadata.get("task", ""), self._task,
            )

            # Check for schema changes since generation
            stored_schema_hash = metadata.get("schema_hash", "")
            if stored_schema_hash:
                current_hash = self._get_schema_hash()
                if stored_schema_hash != current_hash:
                    self._schema_changed = True
                    logger.warning(
                        "Schema has changed since this script was generated. "
                        "If the script fails, regenerate: scraper.regenerate()"
                    )

            self._cached_fn = fn

        # Auto-fix path: diagnosis handles execution with signal collection
        if auto_fix_mode is not None:
            return await self._run_cached_with_autofix(effective_url, auto_fix_mode)

        # Branch: in-process (context-managed) vs subprocess
        if self._context_managed:
            if self._browser_mgr is None:
                logger.info(
                    "Running cached script → %s", self._script_path
                )
            else:
                logger.info("Scraping → %s", effective_url)

            return_value_json = await self._run_in_process(effective_url)
        else:
            logger.info(
                "Running cached script → %s", self._script_path
            )

            stdout, return_value_json, stderr, returncode = (
                await self._execute_function(
                    self._get_function_source(),
                    effective_url,
                )
            )

            if returncode == -1 and "timed out" in stderr:
                raise ScriptTimeoutError(
                    f"Script exceeded the {self._timeout}s timeout.\n\n"
                    f"  Increase the timeout:\n"
                    f"    Scraper(..., timeout={self._timeout * 2})"
                )

            if returncode != 0:
                err_preview = stderr.strip()
                if len(err_preview) > 500:
                    err_preview = err_preview[-500:]
                raise ScriptRuntimeError(
                    f"Script crashed during execution.\n\n"
                    f"  {err_preview}\n\n"
                    f"  The website may have changed. "
                    f"Try: scraper.regenerate()"
                )

        # Validate return value against schema
        data = self._validate_return_value(return_value_json)

        timestamp = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        return ScraperResult(
            data=data,
            url=effective_url,
            timestamp=timestamp,
            cached=True,
            script_path=(
                str(self._script_path) if self._script_path else None
            ),
        )

    # -- Internal: auto-fix execution path --

    async def _run_cached_with_autofix(
        self,
        effective_url: str,
        auto_fix_mode: AutoFixMode | None = None,
    ) -> ScraperResult:
        """Execute cached script with auto-fix diagnosis (spec §7).

        Delegates all execution to the diagnosis loop, which runs
        up to 3 attempts with page signal collection and error
        fingerprinting. On success, returns the data. On failure,
        either regenerates the script or raises an enhanced error.
        """
        from .autofix import diagnose
        from .autofix.types import (
            AttemptResult,
            AutoFixAction,
            DiagnosisResult,
        )

        assert auto_fix_mode is not None

        if self._context_managed:
            if self._browser_mgr is None:
                logger.info(
                    "Running cached script → %s", self._script_path,
                )
            else:
                logger.info("Scraping → %s", effective_url)
        else:
            logger.info(
                "Running cached script → %s", self._script_path,
            )

        # Build execute_fn adapter for the current execution mode
        if self._context_managed:
            execute_fn = self._make_in_process_execute_fn(effective_url)
        else:
            execute_fn = self._make_subprocess_execute_fn(effective_url)

        logger.debug(
            "Auto-fix: diagnosing cached script (%s mode)...",
            auto_fix_mode.value,
        )

        # §7: Diagnosis loop — up to 3 attempts with signals
        result = await diagnose(
            execute_fn, effective_url, auto_fix_mode,
        )

        # Success on any attempt — return data
        if isinstance(result, AttemptResult):
            timestamp = datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ",
            )
            return ScraperResult(
                data=result.data,
                url=effective_url,
                timestamp=timestamp,
                cached=True,
                script_path=(
                    str(self._script_path)
                    if self._script_path else None
                ),
            )

        # All attempts failed — act on the decision
        assert isinstance(result, DiagnosisResult)

        if result.action == AutoFixAction.RAISE:
            self._raise_autofix_error(result)

        # §9/§10: REGENERATE — generate a new script
        assert result.action == AutoFixAction.REGENERATE

        # Respect protect_script: if user edited the script, don't
        # auto-overwrite — raise the original error instead.
        if (
            self._protect_script
            and self._script_path
            and self._script_path.exists()
            and _is_script_user_edited(self._script_path)
        ):
            logger.info(
                "Auto-fix: would regenerate, but script is protected "
                "(manually edited + protect_script=True). Raising error."
            )
            self._raise_autofix_error(result)

        logger.info(
            "Auto-fix: regenerating script with %s...",
            self._model,
        )

        # §10: Clear cached function before regeneration
        self._cached_fn = None
        self._cached_source = None
        self._schema_changed = False

        # Pre-flight checks for generation
        self._check_api_key()
        self._check_playwright()

        start_time = time.monotonic()
        try:
            regen_result = await self._run_generate(
                effective_url, start_time,
            )
        except GenerationError as exc:
            # Agent could not produce a valid script at all
            logger.info("Auto-fix: regeneration failed — %s", exc)
            raise AutoFixError(
                f"Auto-fix triggered regeneration, but the AI agent "
                f"could not produce a valid script.\n\n"
                f"  Original failure: {result.message}\n\n"
                f"  Generation error: {exc}\n\n"
                f"  Check the URL manually, or adjust the task/schema."
            ) from exc
        except ValidationError as exc:
            # §10: New script's output also failed schema validation.
            # Re-raise as ValidationError (not AutoFixError)
            # per spec §10: same schema constraint → ValidationError.
            logger.info(
                "Auto-fix: new script failed schema validation — %s",
                exc,
            )
            raise ValidationError(
                f"Auto-fix regenerated a script, but it also produced "
                f"output that does not match the schema.\n\n"
                f"  {exc}\n\n"
                f"  The page may have fewer items than expected, "
                f"or the data structure has changed.\n\n"
                f"  Check the URL or adjust the schema."
            ) from exc

        # Regeneration succeeded
        logger.info("Auto-fix: new script succeeded.")
        regen_result.auto_fixed = True
        return regen_result

    def _raise_autofix_error(self, result: Any) -> None:
        """Raise the appropriate exception for a RAISE diagnosis.

        Maps error categories to Scout exception types and includes
        the full diagnostic message from the diagnosis loop.
        """
        from .autofix.types import ErrorCategory

        category = result.category
        message = (
            f"Cached script failed.\n\n"
            f"{result.message}\n\n"
            f"  To regenerate: scraper.regenerate()"
        )

        if category in (ErrorCategory.D, ErrorCategory.F2):
            raise ScriptTimeoutError(message)
        if category == ErrorCategory.G:
            raise ValidationError(message)
        if category == ErrorCategory.F3:
            raise Error(message)
        # B, C, E, F1, and any other
        raise ScriptRuntimeError(message)

    # -- Internal: in-process execution --

    async def _run_in_process(self, effective_url: str) -> str | None:
        """Execute cached function in-process with shared browser.

        Returns the return value as a JSON string, or None on failure.
        """
        from .runtime.environment import BrowserManager

        # Launch browser on first in-process call
        if self._browser_mgr is None:
            self._browser_mgr = BrowserManager(
                    headless=self._headless,
                    launch_options=self._get_resolved_launch_options(),
                )
            await self._browser_mgr.start()
            logger.info(
                "Launching browser (will reuse for subsequent runs)"
            )

        page = await self._browser_mgr.new_page()
        try:
            await page.goto(effective_url, wait_until="domcontentloaded")

            # Lightweight checkpoint for cached runs
            async def checkpoint(label: str, data_preview: Any = None) -> None:
                dp = (
                    f" | {len(data_preview)} items"
                    if data_preview else ""
                )
                logger.debug("[checkpoint] %s%s", label, dp)

            try:
                data = await asyncio.wait_for(
                    self._cached_fn(page, effective_url, checkpoint),
                    timeout=self._timeout,
                )
            except asyncio.TimeoutError:
                raise ScriptTimeoutError(
                    f"Script exceeded the {self._timeout}s timeout.\n\n"
                    f"  Increase the timeout:\n"
                    f"    Scraper(..., timeout={self._timeout * 2})"
                ) from None

            self._cm_page_count += 1

            try:
                rv_json = json.dumps(
                    data, ensure_ascii=False, indent=2, default=str,
                )
            except TypeError as exc:
                type_name = type(data).__name__
                raise ScriptRuntimeError(
                    f"scrape() returned type '{type_name}' which is not "
                    f"JSON-serializable: {exc}"
                ) from None

            return rv_json

        except (
            ScriptTimeoutError,
            ScriptRuntimeError,
        ):
            raise
        except Exception as exc:
            raise ScriptRuntimeError(
                f"Script crashed during execution.\n\n"
                f"  {exc}\n\n"
                f"  The website may have changed. "
                f"Try: scraper.regenerate()"
            ) from exc
        finally:
            try:
                await page.close()
            except Exception:
                pass

    def _get_function_source(self) -> str:
        """Get the source code of the cached function."""
        if self._cached_source is not None:
            return self._cached_source
        if self._script_path is None:
            raise Error("No script path — cannot get function source")
        source = self._script_path.read_text(encoding="utf-8")
        # Strip the metadata docstring to get just the function code
        m = _METADATA_RE.search(source)
        if m:
            return source[m.end():].strip()
        return source

    async def _execute_function(
        self,
        function_source: str,
        url: str,
    ) -> tuple[str, str | None, str, int]:
        """Run a scrape function in a fresh subprocess.

        Returns (stdout, return_value_json, stderr, returncode).
        """
        from .agent.wrapper import (
            generate_subprocess_wrapper,
            parse_return_value,
        )

        cp_dir = tempfile.mkdtemp(prefix="scrape_cp_")
        wrapper_code = generate_subprocess_wrapper(
            function_source, url, cp_dir,
            sandbox=self._sandbox,
            launch_options=self._get_resolved_launch_options(),
        )

        script_dir = Path(tempfile.mkdtemp(prefix="scrape_run_"))
        script_path = script_dir / "script.py"
        try:
            script_path.write_text(wrapper_code, encoding="utf-8")

            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=self._timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return (
                    "",
                    None,
                    f"Function timed out after {self._timeout} seconds",
                    -1,
                )

            raw_stdout = stdout_bytes.decode(errors="replace")
            stderr = stderr_bytes.decode(errors="replace")
            clean_stdout, return_value_json = parse_return_value(raw_stdout)

            return (
                clean_stdout,
                return_value_json,
                stderr,
                proc.returncode or 0,
            )
        finally:
            shutil.rmtree(script_dir, ignore_errors=True)
            shutil.rmtree(cp_dir, ignore_errors=True)

    def _validate_return_value(
        self, return_value_json: str | None,
    ) -> Any:
        """Parse and validate the return value against the schema.

        Returns the validated Python data.
        Raises ValidationError on failure.
        """
        if return_value_json is None:
            data = None
        else:
            try:
                data = json.loads(return_value_json)
            except (ValueError, TypeError):
                data = None

        valid, feedback = self._compiled_schema.validate(
            data, tolerance=self._tolerance,
        )
        if not valid:
            if self._schema_changed:
                raise ValidationError(
                    f"Script output does not match the schema.\n\n"
                    f"{feedback}\n\n"
                    f"The schema has changed since this script was "
                    f"generated. Regenerate to match the new schema:\n"
                    f"  scraper.regenerate()"
                )
            raise ValidationError(
                f"Script output does not match the schema.\n\n"
                f"{feedback}\n\n"
                f"The website may have changed. "
                f"Try: scraper.regenerate()"
            )
        return data

    # -- Internal: auto-fix execute_fn adapters --

    def _make_subprocess_execute_fn(
        self,
        effective_url: str,
    ) -> Callable[[], Any]:
        """Build an execute_fn adapter for subprocess execution.

        Returns an async callable that runs the cached script in a
        subprocess with page signal collection, and returns an
        ``AttemptResult``.

        Used by the auto-fix diagnosis loop. Each call spawns a fresh
        subprocess with a fresh browser — independent of previous attempts.
        """
        from .autofix.types import AttemptResult, PageSignals

        async def execute() -> AttemptResult:
            from .agent.wrapper import (
                generate_subprocess_wrapper,
                parse_page_signals,
                parse_return_value,
            )

            function_source = self._get_function_source()
            cp_dir = tempfile.mkdtemp(prefix="scrape_cp_")
            wrapper_code = generate_subprocess_wrapper(
                function_source, effective_url, cp_dir,
                collect_page_signals=True,
                launch_options=self._get_resolved_launch_options(),
            )

            script_dir = Path(tempfile.mkdtemp(prefix="scrape_run_"))
            script_path = script_dir / "script.py"
            try:
                script_path.write_text(wrapper_code, encoding="utf-8")

                proc = await asyncio.create_subprocess_exec(
                    sys.executable, str(script_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        proc.communicate(), timeout=self._timeout,
                    )
                except asyncio.TimeoutError:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass  # Process already exited
                    await proc.communicate()
                    return AttemptResult(
                        success=False,
                        error=(
                            f"Function timed out after "
                            f"{self._timeout} seconds"
                        ),
                        exit_code=-1,
                    )

                raw_stdout = stdout_bytes.decode(errors="replace")
                stderr = stderr_bytes.decode(errors="replace")
                returncode = proc.returncode or 0

                _, return_value_json = parse_return_value(raw_stdout)

                # Extract page signals from subprocess output.
                # The wrapper emits signals on both success and failure
                # when collect_page_signals=True (needed for Category G).
                signals_dict = parse_page_signals(raw_stdout)
                page_signals: PageSignals | None = None
                if signals_dict is not None:
                    page_signals = PageSignals(
                        http_status=signals_dict.get("http_status"),
                        page_url=signals_dict.get("page_url"),
                        content=signals_dict.get("content"),
                        headers=signals_dict.get("headers", {}),
                        cookies=signals_dict.get("cookies", []),
                    )

                # Check for failure
                if returncode != 0 or return_value_json is None:
                    return AttemptResult(
                        success=False,
                        error=stderr or None,
                        exit_code=returncode,
                        page_signals=page_signals,
                    )

                # Parse the return value
                try:
                    data = json.loads(return_value_json)
                except (ValueError, TypeError):
                    return AttemptResult(
                        success=False,
                        error="Script output is not valid JSON",
                        page_signals=page_signals,
                    )

                # Schema validation (Category G)
                valid, feedback = self._compiled_schema.validate(
                    data, tolerance=self._tolerance,
                )
                if not valid:
                    return AttemptResult(
                        success=False,
                        data=data,
                        schema_error=feedback,
                        page_signals=page_signals,
                    )

                return AttemptResult(success=True, data=data)

            finally:
                shutil.rmtree(script_dir, ignore_errors=True)
                shutil.rmtree(cp_dir, ignore_errors=True)

        return execute

    def _make_in_process_execute_fn(
        self,
        effective_url: str,
    ) -> Callable[[], Any]:
        """Build an execute_fn adapter for in-process execution.

        Returns an async callable that runs the cached script in-process
        with page signal collection, and returns an ``AttemptResult``.

        Uses the shared browser from the context manager. Each call
        creates a fresh page (navigate independently per attempt).
        """
        from .autofix.types import AttemptResult, PageSignals

        async def execute() -> AttemptResult:
            from .runtime.environment import BrowserManager

            # Ensure browser is launched
            if self._browser_mgr is None:
                self._browser_mgr = BrowserManager(
                    headless=self._headless,
                    launch_options=self._get_resolved_launch_options(),
                )
                await self._browser_mgr.start()
                logger.info(
                    "Launching browser (will reuse for subsequent runs)"
                )

            page = await self._browser_mgr.new_page()
            document_responses: list[Any] = []

            def on_response(response: Any) -> None:
                try:
                    if response.request.resource_type == "document":
                        document_responses.append(response)
                except Exception:
                    pass

            page_signals: PageSignals | None = None

            try:
                # §6: Attach response listener BEFORE navigation
                page.on("response", on_response)

                await page.goto(
                    effective_url, wait_until="domcontentloaded",
                )

                # Lightweight checkpoint
                async def checkpoint(
                    label: str, data_preview: Any = None,
                ) -> None:
                    dp = (
                        f" | {len(data_preview)} items"
                        if data_preview else ""
                    )
                    logger.debug("[checkpoint] %s%s", label, dp)

                try:
                    data = await asyncio.wait_for(
                        self._cached_fn(page, effective_url, checkpoint),
                        timeout=self._timeout,
                    )
                except asyncio.TimeoutError:
                    # Collect signals before returning failure
                    page_signals = await self._collect_in_process_signals(
                        page, document_responses,
                    )
                    return AttemptResult(
                        success=False,
                        error=(
                            f"Script execution timed out after "
                            f"{self._timeout} seconds"
                        ),
                        page_signals=page_signals,
                    )

                # Serialize return value
                try:
                    rv_json = json.dumps(
                        data, ensure_ascii=False, indent=2, default=str,
                    )
                except TypeError as exc:
                    return AttemptResult(
                        success=False,
                        error=(
                            f"scrape() returned non-serializable type: {exc}"
                        ),
                        page_signals=await self._collect_in_process_signals(
                            page, document_responses,
                        ),
                    )

                # Schema validation (Category G)
                parsed = json.loads(rv_json)
                valid, feedback = self._compiled_schema.validate(
                    parsed, tolerance=self._tolerance,
                )
                if not valid:
                    return AttemptResult(
                        success=False,
                        data=parsed,
                        schema_error=feedback,
                        page_signals=await self._collect_in_process_signals(
                            page, document_responses,
                        ),
                    )

                self._cm_page_count += 1
                return AttemptResult(success=True, data=parsed)

            except Exception as exc:
                # Script or navigation failure
                page_signals = await self._collect_in_process_signals(
                    page, document_responses,
                )
                return AttemptResult(
                    success=False,
                    error=f"{type(exc).__name__}: {exc}",
                    page_signals=page_signals,
                )
            finally:
                try:
                    page.remove_listener("response", on_response)
                except Exception:
                    pass
                try:
                    await page.close()
                except Exception:
                    pass

        return execute

    @staticmethod
    async def _collect_in_process_signals(
        page: Any,
        document_responses: list[Any],
    ) -> Any:
        """Collect page-level signals after a script failure (in-process).

        Every access is defensive — the page may be crashed or closed.
        Returns a ``PageSignals`` dataclass.

        Spec reference: §6 signal collection.
        """
        from .autofix.types import PageSignals

        http_status: int | None = None
        page_url: str | None = None
        content: str | None = None
        headers: dict[str, str] = {}
        cookies: list[dict[str, str]] = []

        try:
            page_url = page.url
        except Exception:
            pass

        try:
            content = await asyncio.wait_for(
                page.content(), timeout=5.0,
            )
        except Exception:
            pass

        if document_responses:
            last_resp = document_responses[-1]
            try:
                http_status = last_resp.status
            except Exception:
                pass
            try:
                headers = dict(last_resp.headers)
            except Exception:
                pass

        try:
            context = page.context
            raw_cookies = await context.cookies()
            cookies = [
                {"name": c["name"], "value": c.get("value", "")}
                for c in raw_cookies
            ]
        except Exception:
            pass

        return PageSignals(
            http_status=http_status,
            page_url=page_url,
            content=content,
            headers=headers,
            cookies=cookies,
        )

    # -- Internal: generation path --

    async def _run_generate(
        self,
        effective_url: str,
        start_time: float,
    ) -> ScraperResult:
        """Generate a scraping function via the AI agent."""
        from .agent.loop import AgentLoop
        from .agent.llm import LLMConfig

        llm_config = LLMConfig(
            model=self._model,
            api_key=self._api_key,
        )

        agent = AgentLoop(
            llm_config=llm_config,
            headless=self._headless,
            script_timeout=self._timeout,
            max_script_attempts=self._max_attempts,
            compiled_schema=self._compiled_schema,
            approval_mode="auto",
            sandbox=self._sandbox,
            launch_options=self._get_resolved_launch_options(),
            tolerance=self._tolerance,
        )

        try:
            result = await agent.run(url=effective_url, task=self._task)
        except Exception as exc:
            raise self._map_generation_error(exc) from exc

        if not result.success:
            model_hint = ""
            lower_model = self._model.lower()
            if "haiku" in lower_model or "mini" in lower_model or "flash" in lower_model:
                model_hint = (
                    "\n\n  The page may be too complex for this model. "
                    "Try a more capable one:\n"
                    '    model="claude-sonnet-4-5"'
                )
            raise GenerationError(
                (result.error
                or "AI failed to generate a valid scraping function "
                f"after {self._max_attempts} attempts.")
                + model_hint
            )

        # Validate return value against schema
        data = self._validate_return_value(result.return_value)

        # Save script if script= was set
        if self._script_path is not None:
            _save_script(
                result.final_script,
                self._script_path,
                effective_url,
                self._task,
                self._model,
                schema_hash=self._get_schema_hash(),
            )
            elapsed = time.monotonic() - start_time
            # Count lines in the function
            line_count = len(result.final_script.strip().splitlines())
            logger.info(
                "Script saved → %s (%d lines, %ds)",
                self._script_path,
                line_count,
                int(elapsed),
            )
        else:
            elapsed = time.monotonic() - start_time
            logger.info(
                "Done (%ds). Pass script= to cache the function "
                "for future runs.",
                int(elapsed),
            )

        # Cache function and source in memory for reuse
        self._cached_source = result.final_script
        if self._script_path:
            fn, _meta = _load_script(
                self._script_path, sandbox=self._sandbox,
            )
            self._cached_fn = fn
        else:
            # No script on disk — compile and cache from source directly
            try:
                if self._sandbox:
                    from .runtime.sandbox import (
                        compile_restricted_agent_code,
                        build_restricted_globals,
                        build_safe_pre_imports,
                    )
                    ns = build_restricted_globals(build_safe_pre_imports())
                    ns["__file__"] = "<scout-generated>"
                    exec(  # noqa: S102
                        compile_restricted_agent_code(result.final_script),
                        ns,
                    )
                else:
                    ns = {
                        "__file__": "<scout-generated>",
                        **_build_pre_import_namespace(),
                    }
                    exec(  # noqa: S102
                        compile(result.final_script, "<scout-generated>", "exec"),
                        ns,
                    )
                fn = ns.get("scrape")
                if fn is not None and callable(fn):
                    self._cached_fn = fn
            except Exception:
                # Caching failed but data is already validated — skip silently
                pass

        timestamp = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        return ScraperResult(
            data=data,
            url=effective_url,
            timestamp=timestamp,
            cached=False,
            script_path=(
                str(self._script_path) if self._script_path else None
            ),
        )

    def _map_generation_error(self, exc: Exception) -> Error:
        """Map LLM API errors to GenerationError."""
        from pydantic_ai.exceptions import ModelHTTPError

        if isinstance(exc, ModelHTTPError):
            status = exc.status_code
            if status == 429:
                return GenerationError(
                    "API rate limit exceeded. "
                    "Retry in a few minutes.",
                    status_code=429,
                )
            if status == 401:
                return GenerationError(
                    "API rejected the API key. "
                    "Check your API key environment variable or "
                    "the api_key= argument.",
                    status_code=401,
                )
            if status and status >= 500:
                return GenerationError(
                    f"API returned a server error ({status}). "
                    "This is usually transient — retry shortly.",
                    status_code=status,
                )
            return GenerationError(
                f"API error ({status}): {exc}",
                status_code=status,
            )
        if isinstance(exc, ConnectionError):
            return GenerationError(
                "Could not reach the API. "
                "Check your network connection."
            )
        return GenerationError(str(exc))
