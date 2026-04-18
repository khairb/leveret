"""Scout public API — Scraper class and ScraperResult."""

from __future__ import annotations

import asyncio
import ast
import errno
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
    ScoutConfigError,
    ScoutError,
    ScoutGenerationError,
    ScoutSchemaError,
    ScoutScriptLoadError,
    ScoutScriptRuntimeError,
    ScoutScriptTimeoutError,
    ScoutValidationError,
)
from .schema.compiler import compile_schema
from ._logging import logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from .schema.compiler import CompiledSchema


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

def _build_pre_import_namespace() -> dict[str, Any]:
    """Build namespace with pre-imported modules for in-process execution.

    Matches the pre-imports from the subprocess wrapper template so that
    agent-authored functions work identically in both execution modes.
    """
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
        '"""\n'
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
) -> None:
    """Write a script file with metadata docstring.

    Auto-creates parent directories. Wraps filesystem errors in
    ScoutError with actionable messages.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    docstring = _build_metadata_docstring(url, task, model, timestamp)
    content = docstring + "\n" + code

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        raise ScoutError(
            f'Cannot create directory "{path.parent}" '
            f"— a file with that name already exists."
        ) from None
    except OSError as exc:
        raise ScoutError(
            f'Could not write script to "{path}" — {_describe_os_error(exc)}.'
        ) from None

    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise ScoutError(
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


def _load_script(path: Path) -> tuple[Callable[..., Any], dict[str, str]]:
    """Load a saved script file, validate it, and return the scrape function.

    Returns:
        (scrape_fn, metadata_dict)

    Raises:
        ScoutScriptLoadError: If the file has syntax errors, is missing the
            scrape function, or has the wrong signature.
    """
    # Read the file
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        if exc.errno == errno.EACCES:
            raise ScoutScriptLoadError(
                f'Could not read script "{path}" — permission denied.'
            ) from None
        raise ScoutScriptLoadError(
            f'Could not read script "{path}" — {exc}.'
        ) from None

    if not source.strip():
        raise ScoutScriptLoadError(
            f'Script "{path}" is empty. '
            f"Regenerate with scraper.run(force=True)."
        )

    # Syntax check
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise ScoutScriptLoadError(
            f'Script "{path}" has a syntax error: {exc.msg} '
            f"(line {exc.lineno}). Fix the file or "
            f"regenerate with scraper.run(force=True)."
        ) from None

    # Find async def scrape
    scrape_func = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "scrape":
                scrape_func = node
                break

    if scrape_func is None:
        raise ScoutScriptLoadError(
            f'Script "{path}" has no function named "scrape". '
            f"The file must define: async def scrape(page, url, checkpoint). "
            f"Fix the file or regenerate with scraper.run(force=True)."
        )

    if not isinstance(scrape_func, ast.AsyncFunctionDef):
        raise ScoutScriptLoadError(
            f'Script "{path}" defines "scrape" as a sync function. '
            f"It must be async: async def scrape(page, url, checkpoint). "
            f"Fix the file or regenerate with scraper.run(force=True)."
        )

    # Check signature: exactly (page, url, checkpoint)
    expected_params = ["page", "url", "checkpoint"]
    actual_params = [arg.arg for arg in scrape_func.args.args]
    if actual_params != expected_params:
        raise ScoutScriptLoadError(
            f'Script "{path}" has wrong signature for scrape(). '
            f"Expected: scrape(page, url, checkpoint). "
            f"Got: scrape({', '.join(actual_params)}). "
            f"Fix the file or regenerate with scraper.run(force=True)."
        )

    # Execute the source in an isolated namespace. We use compile+exec
    # instead of importlib to avoid bytecode caching issues when the same
    # file path is overwritten with new content (e.g. force=True).
    compiled_code = compile(source, str(path), "exec")
    namespace: dict[str, Any] = {
        "__file__": str(path),
        **_build_pre_import_namespace(),
    }
    try:
        exec(compiled_code, namespace)  # noqa: S102
    except Exception as exc:
        raise ScoutScriptLoadError(
            f'Script "{path}" failed to load: {exc}. '
            f"Fix the file or regenerate with scraper.run(force=True)."
        ) from None

    fn = namespace.get("scrape")
    if fn is None or not callable(fn):
        raise ScoutScriptLoadError(
            f'Script "{path}" does not export a callable "scrape" function.'
        )

    if not inspect.iscoroutinefunction(fn):
        raise ScoutScriptLoadError(
            f'Script "{path}": scrape is not async. '
            f"It must be: async def scrape(page, url, checkpoint)."
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
    """Raise ScoutConfigError if script was generated for a different domain."""
    script_domain = _normalize_domain(script_url)
    current_domain = _normalize_domain(current_url)

    if not script_domain or not current_domain:
        return  # can't compare — skip

    if script_domain != current_domain:
        raise ScoutConfigError(
            f"Script '{script_path}' was generated for a different site.\n"
            f"\n"
            f"  Script:  {script_url}\n"
            f"  Current: {current_url}\n"
            f"\n"
            f"  To regenerate for the new site: scraper.run(force=True)\n"
            f'  To use a different script file:  script="<new_path>.py"'
        )


def _check_task_mismatch(script_task: str, current_task: str) -> None:
    """Log a warning if the task description changed since generation."""
    if script_task and script_task != current_task:
        logger.warning(
            "Note: Task description has changed since script was generated.\n"
            "        Using cached script. To regenerate: scraper.run(force=True)"
        )


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
    """

    data: Any
    url: str
    timestamp: str
    cached: bool
    script_path: str | None

    def __repr__(self) -> str:
        return (
            f"ScraperResult("
            f"url={self.url!r}, "
            f"{_preview(self.data)}, "
            f"cached={self.cached})"
        )


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
        script: str | Path | None = None,
        model: str = "claude-haiku-4-5",
        headless: bool = True,
        api_key: str | None = None,
        timeout: int = 600,
        max_retries: int = 6,
    ) -> None:
        # -- url --
        if not isinstance(url, str) or not url.strip():
            raise ScoutError(f'url must be a valid HTTP(S) URL (got {url!r})')
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ScoutError(f'url must be a valid HTTP(S) URL (got {url!r})')
        if not parsed.hostname:
            raise ScoutError(f'url must be a valid URL (got {url!r})')

        # -- task --
        if not isinstance(task, str) or not task.strip():
            raise ScoutError("task must not be empty")

        # -- schema --
        if schema is None:
            raise ScoutSchemaError(
                "schema is required — pass a dict, list, or Field/List schema"
            )
        compiled: CompiledSchema = compile_schema(schema)

        # -- script (path normalization) --
        script_path: Path | None = None
        if script is not None:
            path = Path(script).expanduser().resolve()
            if path.is_dir():
                raise ScoutError(
                    f'script must be a file path, not a directory '
                    f'(got {str(script)!r})'
                )
            if path.suffix != ".py":
                raise ScoutError(
                    f'script must be a .py file path (got {str(script)!r})'
                )
            script_path = path

        # -- timeout --
        if not isinstance(timeout, int) or timeout <= 0:
            raise ScoutError(
                f"timeout must be a positive integer (got {timeout!r})"
            )

        # -- max_retries --
        if not isinstance(max_retries, int) or max_retries < 1:
            raise ScoutError(
                f"max_retries must be at least 1 (got {max_retries!r})"
            )

        # -- model --
        if not isinstance(model, str) or not model.strip():
            raise ScoutError("model must not be empty")

        # -- Store validated state --
        self._url = url
        self._task = task
        self._compiled_schema = compiled
        self._script_path = script_path
        self._model = model
        self._headless = headless
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries

        # -- Runtime state --
        self._cached_fn: Any = None
        self._cached_source: str | None = None
        self._context_managed: bool = False
        self._browser_mgr: Any = None
        self._bg_loop: Any = None
        self._bg_thread: threading.Thread | None = None
        self._cm_page_count: int = 0
        self._cm_start_time: float = 0.0

    # -- Public properties --

    @property
    def url(self) -> str:
        """The target URL."""
        return self._url

    @property
    def script_path(self) -> Path | None:
        """Absolute path to the script file, or None."""
        return self._script_path

    def __repr__(self) -> str:
        parts = [f"Scraper({self._url!r}"]
        if self._script_path:
            parts.append(f", script={str(self._script_path)!r}")
            if self._cached_fn is not None:
                parts.append(", cached=True")
        parts.append(")")
        return "".join(parts)

    # -- Context manager --

    def __enter__(self) -> Scraper:
        if self._context_managed:
            raise ScoutError(
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
            raise ScoutError(
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
        *,
        url: str | None = None,
        force: bool = False,
    ) -> ScraperResult:
        """Generate (if needed) and execute the scraping function.

        Args:
            url: Override the constructor URL for this execution only.
                 Must be on the same domain as the original URL.
            force: Discard any cached function and regenerate from scratch.

        Returns:
            ScraperResult with validated data.
        """
        effective_url = self._resolve_url(url)
        start_time = time.monotonic()

        # -- Cached path: load from disk or memory --
        if not force and (self._cached_fn is not None or self._has_script_on_disk()):
            return await self._run_cached(effective_url)

        # -- Generation path --
        if force and self._script_path and self._script_path.exists():
            logger.info(
                "Regenerating script (overwriting %s)", self._script_path
            )
        elif self._script_path:
            logger.info(
                "No cached script found. Generating with %s...",
                self._model,
            )
            logger.info(
                "Note: First run calls the Anthropic API. "
                "Subsequent runs use the cached script."
            )
        else:
            logger.info(
                "Generating scraping function with %s...", self._model
            )

        # Clear in-memory cache when forcing
        if force:
            self._cached_fn = None
            self._cached_source = None

        self._check_api_key()
        self._check_playwright()

        return await self._run_generate(effective_url, start_time)

    def run(
        self,
        *,
        url: str | None = None,
        force: bool = False,
    ) -> ScraperResult:
        """Synchronous version of :meth:`async_run`.

        Inside a ``with scraper:`` block, dispatches to the background
        event loop where the shared browser lives.  Outside a ``with``
        block, uses ``asyncio.run()``.

        Raises ScoutError if called inside a running event loop
        (e.g. Jupyter) without a context manager.
        Use ``await scraper.async_run()`` instead.
        """
        # Context-managed: dispatch to the background loop that owns
        # the shared browser.
        if self._context_managed and self._bg_loop is not None:
            future = asyncio.run_coroutine_threadsafe(
                self.async_run(url=url, force=force),
                self._bg_loop,
            )
            return future.result()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            raise ScoutError(
                "scraper.run() cannot be called inside a running event loop "
                "(e.g. Jupyter, async web server).\n\n"
                "  Use instead:\n"
                "    result = await scraper.async_run()"
            )

        return asyncio.run(self.async_run(url=url, force=force))

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
            raise ScoutError(
                "Cannot export — Scraper has no script= path. "
                "Generate a script first."
            )
        if not self._script_path.exists():
            raise ScoutError(
                f"Cannot export — script not found at {self._script_path}. "
                "Run scraper.run() to generate it first."
            )

        target = Path(path).expanduser().resolve()
        if target.suffix != ".py":
            raise ScoutError(
                f"Export path must be a .py file (got {str(path)!r})"
            )
        if target.exists() and not overwrite:
            raise ScoutError(
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
            raise ScoutError(
                f'Could not create directory "{target.parent}" — '
                f"{_describe_os_error(exc)}."
            ) from None
        try:
            target.write_text(standalone, encoding="utf-8")
        except OSError as exc:
            raise ScoutError(
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
            raise ScoutError(
                f"url must be a valid HTTP(S) URL (got {override_url!r})"
            )
        parsed = urlparse(override_url)
        if parsed.scheme not in ("http", "https"):
            raise ScoutError(
                f"url must be a valid HTTP(S) URL (got {override_url!r})"
            )
        if not parsed.hostname:
            raise ScoutError(
                f"url must be a valid URL (got {override_url!r})"
            )
        return override_url

    # -- Internal: disk check --

    def _has_script_on_disk(self) -> bool:
        """Check if a script file exists on disk."""
        return self._script_path is not None and self._script_path.exists()

    # -- Internal: prerequisite checks --

    def _check_api_key(self) -> None:
        """Verify an API key is available before generation."""
        if self._api_key:
            return
        if os.environ.get("ANTHROPIC_API_KEY"):
            return
        raise ScoutError(
            "Anthropic API key not found.\n\n"
            "  Set the environment variable:\n"
            "    export ANTHROPIC_API_KEY=sk-ant-...\n\n"
            "  Or pass it directly:\n"
            '    Scraper(..., api_key="sk-ant-...")'
        )

    def _check_playwright(self) -> None:
        """Verify Playwright/Patchright browsers are installed."""
        try:
            import patchright  # noqa: F401
        except ImportError:
            raise ScoutError(
                "Playwright browsers not installed.\n\n"
                "  Run this command to install:\n"
                "    playwright install chromium"
            ) from None

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

    async def _run_cached(self, effective_url: str) -> ScraperResult:
        """Execute a cached scraping function."""
        # Load from disk if not in memory
        if self._cached_fn is None:
            assert self._script_path is not None
            fn, metadata = _load_script(self._script_path)

            # Config mismatch checks
            script_url = metadata.get("url", "")
            if script_url:
                _check_domain_mismatch(
                    self._script_path, script_url, effective_url,
                )
            _check_task_mismatch(
                metadata.get("task", ""), self._task,
            )

            self._cached_fn = fn

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
                raise ScoutScriptTimeoutError(
                    f"Script exceeded the {self._timeout}s timeout.\n\n"
                    f"  Increase the timeout:\n"
                    f"    Scraper(..., timeout={self._timeout * 2})"
                )

            if returncode != 0:
                err_preview = stderr.strip()
                if len(err_preview) > 500:
                    err_preview = err_preview[-500:]
                raise ScoutScriptRuntimeError(
                    f"Script crashed during execution.\n\n"
                    f"  {err_preview}\n\n"
                    f"  The website may have changed. "
                    f"Try: scraper.run(force=True)"
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

    # -- Internal: in-process execution --

    async def _run_in_process(self, effective_url: str) -> str | None:
        """Execute cached function in-process with shared browser.

        Returns the return value as a JSON string, or None on failure.
        """
        from .runtime.environment import BrowserManager

        # Launch browser on first in-process call
        if self._browser_mgr is None:
            self._browser_mgr = BrowserManager(headless=self._headless)
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
                raise ScoutScriptTimeoutError(
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
                raise ScoutScriptRuntimeError(
                    f"scrape() returned type '{type_name}' which is not "
                    f"JSON-serializable: {exc}"
                ) from None

            return rv_json

        except (
            ScoutScriptTimeoutError,
            ScoutScriptRuntimeError,
        ):
            raise
        except Exception as exc:
            raise ScoutScriptRuntimeError(
                f"Script crashed during execution.\n\n"
                f"  {exc}\n\n"
                f"  The website may have changed. "
                f"Try: scraper.run(force=True)"
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
            raise ScoutError("No script path — cannot get function source")
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
        Raises ScoutValidationError on failure.
        """
        if return_value_json is None:
            data = None
        else:
            try:
                data = json.loads(return_value_json)
            except (ValueError, TypeError):
                data = None

        valid, feedback = self._compiled_schema.validate(data)
        if not valid:
            raise ScoutValidationError(
                f"Script output does not match the schema.\n\n"
                f"{feedback}\n\n"
                f"The website may have changed. "
                f"Try: scraper.run(force=True)"
            )
        return data

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
            max_script_attempts=self._max_retries,
            compiled_schema=self._compiled_schema,
            approval_mode="auto",
        )

        try:
            result = await agent.run(url=effective_url, task=self._task)
        except Exception as exc:
            raise self._map_generation_error(exc) from exc

        if not result.success:
            raise ScoutGenerationError(
                result.error
                or "AI failed to generate a valid scraping function "
                f"after {self._max_retries} attempts."
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
            fn, _meta = _load_script(self._script_path)
            self._cached_fn = fn
        else:
            # No script on disk — compile and cache from source directly
            try:
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

    def _map_generation_error(self, exc: Exception) -> ScoutError:
        """Map Anthropic SDK errors to ScoutGenerationError."""
        try:
            import anthropic
        except ImportError:
            return ScoutGenerationError(str(exc))

        if isinstance(exc, anthropic.RateLimitError):
            return ScoutGenerationError(
                "Anthropic API rate limit exceeded. "
                "Retry in a few minutes, or check your usage "
                "at console.anthropic.com."
            )
        if isinstance(exc, anthropic.AuthenticationError):
            return ScoutGenerationError(
                "Anthropic API rejected the API key. "
                "Check ANTHROPIC_API_KEY or the api_key= argument."
            )
        if isinstance(exc, anthropic.APIConnectionError):
            return ScoutGenerationError(
                "Could not reach the Anthropic API. "
                "Check your network connection."
            )
        if isinstance(exc, anthropic.APIStatusError):
            status = getattr(exc, "status_code", "unknown")
            if status and int(status) >= 500:
                return ScoutGenerationError(
                    f"Anthropic API returned a server error ({status}). "
                    "This is usually transient — retry shortly."
                )
            return ScoutGenerationError(
                f"Anthropic API rejected the request: {exc.message}."
            )
        # Not an Anthropic error — wrap generically
        return ScoutGenerationError(str(exc))
