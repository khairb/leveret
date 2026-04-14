"""
scraping_runtime.py — Stateful Python runtime for AI scraping agents.

Usage:
    from scraping_runtime import ScrapingRuntime

    runtime = ScrapingRuntime(
        headless=False,
        post_exec_hook=my_hook,
    )
    await runtime.start()
    result = await runtime.execute('await page.goto("https://example.com")')
    await runtime.stop()

Architecture:
    ScrapingRuntime
    ├── BaseREPL (abstract — swap implementations later)
    │   └── LocalREPL (exec/eval with shared globals)
    ├── BrowserManager (Playwright lifecycle)
    ├── SnapshotCapture (page state extraction)
    └── ExecutionHistory (step log)

To upgrade later, implement BaseREPL with IPythonREPL or DockerREPL
and pass it to ScrapingRuntime. Nothing else changes.
"""

from __future__ import annotations

import abc
import asyncio
import base64
import io
import json
import logging
import shutil
import sys
import tempfile
import textwrap
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from patchright.async_api import (
    Browser,
    BrowserContext,
    ConsoleMessage,
    Page,
    Playwright,
    Request,
    Response,
    async_playwright,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════


@dataclass
class PageSnapshot:
    """Structured capture of page state at a point in time."""

    url: str = ""
    title: str = ""
    viewport_size: dict = field(default_factory=dict)
    html: str = ""
    text_content: str = ""
    screenshot_b64: str = ""
    timestamp: str = ""

    def summary(self, max_html: int = 2000, max_text: int = 1000) -> str:
        """Compact text for feeding back to an LLM."""
        parts = [
            f"URL: {self.url}",
            f"Title: {self.title}",
            f"Viewport: {self.viewport_size}",
        ]
        if self.text_content:
            parts.append(f"Text (first {max_text} chars):\n{self.text_content[:max_text]}")
        if self.html:
            parts.append(f"HTML (first {max_html} chars):\n{self.html[:max_html]}")
        return "\n".join(parts)


@dataclass
class ExecutionResult:
    """Returned to the caller after each code execution."""

    code: str = ""
    output: str = ""
    error: str = ""
    success: bool = True
    snapshot: Optional[PageSnapshot] = None
    hook_data: Any = None
    duration_ms: float = 0
    step: int = 0
    diagnostics: Optional[TimeoutDiagnostics] = None


@dataclass
class TimeoutDiagnostics:
    """Diagnostic data collected when a timeout occurs.

    Captures everything needed to understand *why* execution timed out:
    browser console logs, pending network requests, partial stdout, and
    a snapshot of the page state at the moment the timeout was detected.
    """

    partial_stdout: str = ""
    """Any stdout captured before the timeout killed execution."""

    console_logs: list[dict] = field(default_factory=list)
    """Browser console messages collected during execution.
    Each dict: {level, text, url, line_number, timestamp}."""

    pending_requests: list[dict] = field(default_factory=list)
    """Network requests that were still in-flight when timeout hit.
    Each dict: {url, method, resource_type, started_at}."""

    completed_requests: list[dict] = field(default_factory=list)
    """Network requests completed during this execution step.
    Each dict: {url, method, status, resource_type, duration_ms}."""

    failed_requests: list[dict] = field(default_factory=list)
    """Network requests that failed during execution.
    Each dict: {url, method, resource_type, error}."""

    page_url: str = ""
    """Page URL at the time the timeout was detected."""

    page_title: str = ""
    """Page title at timeout detection time."""

    screenshot_b64: str = ""
    """Base64-encoded PNG screenshot at timeout detection time."""

    elapsed_ms: float = 0
    """Wall-clock time elapsed before timeout."""

    timeout_limit_s: float = 0
    """The timeout limit that was exceeded."""

    code_executed: str = ""
    """The code that was being executed when timeout occurred."""

    def summary(self) -> str:
        """Human-readable summary for quick debugging."""
        parts = [
            f"=== TIMEOUT DIAGNOSTICS (limit={self.timeout_limit_s}s, "
            f"elapsed={self.elapsed_ms:.0f}ms) ===",
            f"Page URL: {self.page_url}",
            f"Page title: {self.page_title}",
        ]
        if self.partial_stdout:
            parts.append(f"\n-- Partial stdout ({len(self.partial_stdout)} chars) --")
            parts.append(self.partial_stdout[:2000])
        if self.console_logs:
            parts.append(f"\n-- Console logs ({len(self.console_logs)} entries) --")
            for log in self.console_logs[-20:]:
                parts.append(f"  [{log.get('level', '?')}] {log.get('text', '')[:200]}")
        if self.pending_requests:
            parts.append(
                f"\n-- Pending network requests ({len(self.pending_requests)}) --"
            )
            for req in self.pending_requests:
                parts.append(
                    f"  {req.get('method', '?')} {req.get('url', '?')[:120]} "
                    f"[{req.get('resource_type', '?')}]"
                )
        if self.failed_requests:
            parts.append(
                f"\n-- Failed network requests ({len(self.failed_requests)}) --"
            )
            for req in self.failed_requests:
                parts.append(
                    f"  {req.get('method', '?')} {req.get('url', '?')[:120]} "
                    f"err={req.get('error', '?')}"
                )
        return "\n".join(parts)

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dictionary."""
        return {
            "partial_stdout": self.partial_stdout,
            "console_logs": self.console_logs,
            "pending_requests": self.pending_requests,
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "page_url": self.page_url,
            "page_title": self.page_title,
            "has_screenshot": bool(self.screenshot_b64),
            "elapsed_ms": self.elapsed_ms,
            "timeout_limit_s": self.timeout_limit_s,
            "code_executed": self.code_executed,
        }

    def save(self, path: str | Path) -> Path:
        """Write full diagnostics to a JSON file (+ screenshot as separate PNG).

        Returns the path to the JSON file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save JSON (everything except screenshot blob).
        json_path = path.with_suffix(".json")
        json_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Save screenshot separately if present.
        if self.screenshot_b64:
            png_path = path.with_suffix(".png")
            png_path.write_bytes(base64.b64decode(self.screenshot_b64))

        return json_path


# ═══════════════════════════════════════════════
#  Browser Instrumentation — Console & Network
# ═══════════════════════════════════════════════


class PageInstrumentation:
    """Collects browser console logs and network request data from a Page.

    Attaches event listeners on ``start()`` and detaches on ``stop()``.
    Between those calls, all console messages and network events are
    buffered.  Call ``drain_since(step_start)`` to get only the events
    that occurred during the current execution step.
    """

    def __init__(self) -> None:
        self._console_logs: list[dict] = []
        self._inflight: dict[str, dict] = {}  # url+method → request info
        self._completed: list[dict] = []
        self._failed: list[dict] = []
        self._page: Optional[Page] = None
        self._attached = False

    def attach(self, page: Page) -> None:
        """Start listening to page events."""
        if self._attached:
            return
        self._page = page
        page.on("console", self._on_console)
        page.on("request", self._on_request)
        page.on("requestfinished", self._on_request_finished)
        page.on("requestfailed", self._on_request_failed)
        page.on("response", self._on_response)
        self._attached = True

    def detach(self) -> None:
        """Stop listening to page events."""
        if not self._attached or not self._page:
            return
        try:
            self._page.remove_listener("console", self._on_console)
            self._page.remove_listener("request", self._on_request)
            self._page.remove_listener("requestfinished", self._on_request_finished)
            self._page.remove_listener("requestfailed", self._on_request_failed)
            self._page.remove_listener("response", self._on_response)
        except Exception:
            pass  # Page may already be closed.
        self._attached = False

    def mark_step_start(self) -> float:
        """Return a timestamp marking the start of an execution step."""
        return time.monotonic()

    def drain_since(self, step_start: float) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
        """Return (console_logs, pending, completed, failed) since step_start.

        Also returns still-inflight requests as "pending".
        """
        console = [l for l in self._console_logs if l.get("_mono", 0) >= step_start]
        completed = [r for r in self._completed if r.get("_mono", 0) >= step_start]
        failed = [r for r in self._failed if r.get("_mono", 0) >= step_start]

        # Anything still in _inflight that started after step_start is pending.
        pending = [
            r for r in self._inflight.values()
            if r.get("_mono", 0) >= step_start
        ]

        # Strip internal monotonic timestamps from output copies.
        def _clean(items: list[dict]) -> list[dict]:
            return [{k: v for k, v in d.items() if k != "_mono"} for d in items]

        return _clean(console), _clean(pending), _clean(completed), _clean(failed)

    def clear(self) -> None:
        """Clear all accumulated data."""
        self._console_logs.clear()
        self._inflight.clear()
        self._completed.clear()
        self._failed.clear()

    # ── Event handlers ────────────────────────────────────────

    def _on_console(self, msg: ConsoleMessage) -> None:
        try:
            location = msg.location
            self._console_logs.append({
                "level": msg.type,
                "text": msg.text,
                "url": location.get("url", "") if location else "",
                "line_number": location.get("lineNumber", 0) if location else 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "_mono": time.monotonic(),
            })
        except Exception:
            pass

    def _on_request(self, request: Request) -> None:
        try:
            key = f"{request.method}:{request.url}"
            self._inflight[key] = {
                "url": request.url,
                "method": request.method,
                "resource_type": request.resource_type,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "_mono": time.monotonic(),
            }
        except Exception:
            pass

    def _on_request_finished(self, request: Request) -> None:
        try:
            key = f"{request.method}:{request.url}"
            info = self._inflight.pop(key, None)
            if info:
                info["duration_ms"] = (time.monotonic() - info["_mono"]) * 1000
                response = request.response
                info["status"] = None
                # response is a coroutine in some cases — skip if so
                self._completed.append(info)
        except Exception:
            pass

    def _on_request_failed(self, request: Request) -> None:
        try:
            key = f"{request.method}:{request.url}"
            info = self._inflight.pop(key, {})
            info.update({
                "url": request.url,
                "method": request.method,
                "resource_type": request.resource_type,
                "error": request.failure or "unknown",
                "_mono": time.monotonic(),
            })
            self._failed.append(info)
        except Exception:
            pass

    def _on_response(self, response: Response) -> None:
        """Update completed requests with status code when response arrives."""
        try:
            key = f"{response.request.method}:{response.request.url}"
            # Check completed list (most recent first).
            for entry in reversed(self._completed):
                if f"{entry.get('method')}:{entry.get('url')}" == key:
                    entry["status"] = response.status
                    break
        except Exception:
            pass


# ═══════════════════════════════════════════════
#  REPL Layer — Abstract Base + Local Implementation
# ═══════════════════════════════════════════════


class BaseREPL(abc.ABC):
    """
    Abstract interface for a stateful Python execution environment.

    To upgrade from prototype to production, implement this with
    IPythonREPL, DockerREPL, etc. The ScrapingRuntime doesn't care
    which implementation it's using.
    """

    @abc.abstractmethod
    def inject(self, **kwargs: Any) -> None:
        """Insert named objects into the execution namespace."""

    @abc.abstractmethod
    def get(self, name: str, default: Any = None) -> Any:
        """Retrieve an object from the namespace."""

    @abc.abstractmethod
    async def execute(self, code: str, timeout: float | None = None) -> tuple[str, str]:
        """
        Execute code. Returns (stdout, error_traceback).
        Both are empty strings on success with no output.
        """

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear all state. Called by ScrapingRuntime.reset()."""


class LocalREPL(BaseREPL):
    """
    Lightweight REPL using exec/eval with a shared globals dict.
    No dependencies. No safety. For local prototyping only.
    """

    def __init__(self) -> None:
        self._globals: dict[str, Any] = {"__builtins__": __builtins__}
        self._active_stdout_buffer: Optional[io.StringIO] = None

    # ── Public interface ──

    def inject(self, **kwargs: Any) -> None:
        self._globals.update(kwargs)

    def get(self, name: str, default: Any = None) -> Any:
        return self._globals.get(name, default)

    def reset(self) -> None:
        self._globals.clear()
        self._globals["__builtins__"] = __builtins__

    @property
    def partial_stdout(self) -> str:
        """Retrieve whatever has been captured so far in the active buffer.

        This is crucial for timeout debugging — when asyncio.wait_for
        cancels the coroutine, _run() never returns, so its captured
        output is lost. This property lets us recover it.
        """
        if self._active_stdout_buffer is not None:
            return self._active_stdout_buffer.getvalue()
        return ""

    async def execute(self, code: str, timeout: float | None = None) -> tuple[str, str]:
        coro = self._run(code)
        if timeout is not None:
            coro = asyncio.wait_for(coro, timeout=timeout)
        try:
            return await coro
        except asyncio.TimeoutError:
            # Recover any partial stdout that was captured before timeout.
            partial = self.partial_stdout
            # Restore sys.stdout — _capture_stdout's finally block won't run
            # because the coroutine was cancelled.
            if self._active_stdout_buffer is not None:
                sys.stdout = sys.__stdout__
                self._active_stdout_buffer = None
            error_msg = f"TimeoutError: Execution exceeded {timeout}s limit."
            if partial:
                error_msg += f"\n\n[Partial stdout captured before timeout ({len(partial)} chars)]:\n{partial}"
            return partial, error_msg

    # ── Internals ──

    async def _run(self, code: str) -> tuple[str, str]:
        with self._capture_stdout() as captured:
            try:
                if self._is_expression(code):
                    result = await self._eval_expression(code)
                    if result is not None:
                        print(repr(result))
                elif self._needs_async(code):
                    await self._exec_async(code)
                else:
                    exec(compile(code, "<agent>", "exec"), self._globals)
            except Exception:
                return captured.getvalue(), traceback.format_exc()

        return captured.getvalue(), ""

    def _is_expression(self, code: str) -> bool:
        """Check if code is a single expression (not a statement)."""
        try:
            compile(code, "<test>", "eval")
            return True
        except SyntaxError:
            return False

    def _needs_async(self, code: str) -> bool:
        """Check if code contains top-level await or async constructs."""
        return "await " in code or code.strip().startswith("async ")

    async def _eval_expression(self, code: str) -> Any:
        """Evaluate a single expression, awaiting if it returns a coroutine."""
        result = eval(compile(code, "<agent>", "eval"), self._globals)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    async def _exec_async(self, code: str) -> None:
        """
        Execute code containing top-level await.

        Wraps the code in an async function, executes it, then copies
        any new/modified local variables back into the persistent globals.
        This is the fix for the scoping bug: without the copy-back step,
        variables defined inside the async wrapper would be lost.
        """
        # Snapshot current keys so we can detect new variables
        keys_before = set(self._globals.keys())

        # Build the async wrapper. We pass `__globals__` so the inner
        # function can both read from and write to our persistent namespace.
        wrapper_name = "__async_exec_wrapper__"
        indented = textwrap.indent(code, "    ")
        wrapper = (
            f"async def {wrapper_name}(__globals__):\n"
            f"{indented}\n"
            f"    # Copy all new locals back into globals\n"
            f"    __globals__.update({{k: v for k, v in locals().items() if k != '__globals__'}})"
        )

        exec(compile(wrapper, "<agent-async>", "exec"), self._globals)
        await self._globals[wrapper_name](self._globals)

        # Clean up the wrapper itself
        self._globals.pop(wrapper_name, None)

    @contextmanager
    def _capture_stdout(self):
        """
        Temporarily redirect stdout to a StringIO buffer.
        Uses a contextmanager for guaranteed cleanup.

        Also stores a reference to the buffer in ``_active_stdout_buffer``
        so that :attr:`partial_stdout` can recover output if the coroutine
        is cancelled by a timeout.
        """
        buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buffer
        self._active_stdout_buffer = buffer
        try:
            yield buffer
        finally:
            sys.stdout = old_stdout
            self._active_stdout_buffer = None


# ═══════════════════════════════════════════════
#  Browser Layer
# ═══════════════════════════════════════════════


class BrowserManager:
    """Manages Playwright browser lifecycle with anti-bot detection measures.

    Uses ``launch_persistent_context`` with real Chrome channel, a temporary
    user-data dir, and stealth flags to avoid common bot fingerprinting.
    """

    # Chrome args that reduce automation fingerprint.
    _STEALTH_ARGS: list[str] = [
        "--disable-blink-features=AutomationControlled",
        "--no-first-run",
        "--disable-extensions",
        "--disable-default-apps",
        "--disable-sync",
        "--disable-background-networking",
    ]

    def __init__(
        self,
        headless: bool = False,
        browser_type: str = "chromium",
        viewport: dict | None = None,
        user_agent: str | None = None,
        launch_args: list[str] | None = None,
    ) -> None:
        self._config = {
            "headless": headless,
            "browser_type": browser_type,
            "viewport": viewport or {"width": 1920, "height": 1080},
            "user_agent": user_agent,
            "launch_args": launch_args or [],
        }
        self._pw: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._profile_dir: Optional[str] = None

    @property
    def page(self) -> Optional[Page]:
        return self._page

    @property
    def browser(self) -> Optional[Browser]:
        return self._browser

    @property
    def context(self) -> Optional[BrowserContext]:
        return self._context

    async def start(self) -> Page:
        self._pw = await async_playwright().start()

        # Create a temporary profile directory for persistent context.
        self._profile_dir = tempfile.mkdtemp(prefix="scraping_agent_profile_")

        # Merge stealth args with any user-supplied args.
        args = list(self._STEALTH_ARGS) + self._config["launch_args"]

        launcher = getattr(self._pw, self._config["browser_type"])
        # launch_persistent_context combines browser + context into one object.
        self._context = await launcher.launch_persistent_context(
            user_data_dir=self._profile_dir,
            channel="chrome",
            headless=self._config["headless"],
            no_viewport=True,
            bypass_csp=True,
            locale="en-US",
            timezone_id="America/New_York",
            args=args,
        )

        # Lower the default timeout so a single hung operation doesn't eat
        # the entire execution budget.  Legitimate interactions complete in
        # well under these limits; the agent can always pass an explicit
        # ``timeout`` kwarg to individual calls when more time is needed.
        self._context.set_default_timeout(10_000)           # 10 s for most ops
        self._context.set_default_navigation_timeout(15_000) # 15 s for goto/reload

        # Set viewport explicitly (avoids Playwright emulation detection).
        self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()
        await self._page.set_viewport_size(self._config["viewport"])

        return self._page

    async def new_page(self) -> Page:
        if not self._context:
            raise RuntimeError("Browser not started.")
        page = await self._context.new_page()
        await page.set_viewport_size(self._config["viewport"])
        return page

    async def stop(self) -> None:
        if self._context:
            await self._context.close()
        if self._pw:
            await self._pw.stop()
        # Clean up temporary profile directory.
        if self._profile_dir:
            shutil.rmtree(self._profile_dir, ignore_errors=True)
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        self._profile_dir = None


# ═══════════════════════════════════════════════
#  Snapshot Capture
# ═══════════════════════════════════════════════


@dataclass
class SnapshotConfig:
    """Controls what gets captured after each execution step."""

    include_html: bool = True
    include_text: bool = True
    include_screenshot: bool = False


async def capture_snapshot(page: Page, config: SnapshotConfig) -> PageSnapshot:
    """
    Capture page state. Each field is captured independently
    so one failure doesn't prevent capturing the rest.
    """
    snap = PageSnapshot(timestamp=datetime.now(timezone.utc).isoformat())

    snap.url = _safe(lambda: page.url, "")
    snap.title = await _safe_async(page.title, "")
    snap.viewport_size = page.viewport_size or {}

    if config.include_html:
        snap.html = await _safe_async(page.content, "")

    if config.include_text:
        snap.text_content = await _safe_async(lambda: page.inner_text("body"), "")

    if config.include_screenshot:
        raw = await _safe_async(lambda: page.screenshot(type="png", full_page=False), b"")
        snap.screenshot_b64 = base64.b64encode(raw).decode() if raw else ""

    return snap


def _safe(fn: Callable, default: Any) -> Any:
    """Call a sync function, return default on any exception."""
    try:
        return fn()
    except Exception:
        return default


async def _safe_async(fn: Callable, default: Any) -> Any:
    """Call an async function (or lambda returning a coroutine), return default on error."""
    try:
        result = fn()
        if asyncio.iscoroutine(result):
            return await result
        return result
    except Exception:
        return default


# ═══════════════════════════════════════════════
#  Execution History
# ═══════════════════════════════════════════════


class ExecutionHistory:
    """Ordered log of all execution steps. Useful for debugging and replay."""

    def __init__(self, max_size: int = 200) -> None:
        self._entries: list[ExecutionResult] = []
        self._max_size = max_size

    def append(self, result: ExecutionResult) -> None:
        self._entries.append(result)
        if len(self._entries) > self._max_size:
            self._entries = self._entries[-self._max_size:]

    @property
    def entries(self) -> list[ExecutionResult]:
        return list(self._entries)

    @property
    def last(self) -> Optional[ExecutionResult]:
        return self._entries[-1] if self._entries else None

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def errors_only(self) -> list[ExecutionResult]:
        return [e for e in self._entries if not e.success]

    def summary(self, last_n: int = 10) -> str:
        """Text summary of recent steps for debugging."""
        recent = self._entries[-last_n:]
        lines = []
        for r in recent:
            status = "✓" if r.success else "✗"
            code_preview = r.code.replace("\n", " ")[:60]
            lines.append(f"  [{r.step}] {status} {code_preview}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════
#  Main Runtime
# ═══════════════════════════════════════════════

PostExecHook = Callable[[Page, ExecutionResult], Awaitable[Any]]


class ScrapingRuntime:
    """
    Ties together REPL + Browser + Snapshots + History + Hooks.

    This is the only class your agent integration needs to touch.
    """

    def __init__(
        self,
        # Browser config
        headless: bool = False,
        browser_type: str = "chromium",
        viewport: dict | None = None,
        user_agent: str | None = None,
        launch_args: list[str] | None = None,
        # Execution config
        default_timeout: float = 30.0,
        max_history: int = 200,
        # Snapshot config
        snapshot_config: SnapshotConfig | None = None,
        # Hook — your system's post-processing
        post_exec_hook: PostExecHook | None = None,
        # REPL implementation — swap this to upgrade later
        repl: BaseREPL | None = None,
        # Diagnostics
        diagnostics_dir: str | Path | None = None,
    ) -> None:
        self._browser_mgr = BrowserManager(
            headless=headless,
            browser_type=browser_type,
            viewport=viewport,
            user_agent=user_agent,
            launch_args=launch_args,
        )
        self._repl = repl or LocalREPL()
        self._hook = post_exec_hook
        self._snap_config = snapshot_config or SnapshotConfig()
        self._default_timeout = default_timeout
        self._history = ExecutionHistory(max_size=max_history)
        self._step_counter = 0
        self._started = False
        self._instrumentation = PageInstrumentation()
        self._diagnostics_dir = Path(diagnostics_dir) if diagnostics_dir else None

    # ── Properties ──

    @property
    def page(self) -> Optional[Page]:
        """Direct page access for your system code (outside the agent)."""
        return self._browser_mgr.page

    @property
    def browser(self) -> Optional[Browser]:
        return self._browser_mgr.browser

    @property
    def context(self) -> Optional[BrowserContext]:
        return self._browser_mgr.context

    @property
    def repl(self) -> BaseREPL:
        return self._repl

    @property
    def history(self) -> ExecutionHistory:
        return self._history

    @property
    def is_running(self) -> bool:
        return self._started

    @property
    def instrumentation(self) -> PageInstrumentation:
        """Access the page instrumentation (console logs, network data)."""
        return self._instrumentation

    @property
    def diagnostics_dir(self) -> Path | None:
        return self._diagnostics_dir

    @diagnostics_dir.setter
    def diagnostics_dir(self, path: str | Path | None) -> None:
        self._diagnostics_dir = Path(path) if path else None

    # ── Lifecycle ──

    async def start(self) -> None:
        """Launch browser and wire up the REPL namespace."""
        page = await self._browser_mgr.start()

        # Attach instrumentation to collect console logs and network data.
        self._instrumentation.attach(page)

        self._repl.inject(
            page=page,
            browser=self._browser_mgr.browser,
            context=self._browser_mgr.context,
            new_page=self._browser_mgr.new_page,
            asyncio=asyncio,
        )
        self._started = True

    async def stop(self) -> None:
        """Shut down browser. REPL state and history are preserved."""
        self._instrumentation.detach()
        await self._browser_mgr.stop()
        self._started = False

    async def reset(self) -> None:
        """Full reset: stop browser, clear REPL state, clear history, restart."""
        await self.stop()
        self._repl.reset()
        self._history.clear()
        self._step_counter = 0
        await self.start()

    # ── Core execution ──

    async def execute(self, code: str, timeout: float | None = None) -> ExecutionResult:
        """
        Execute one step of agent code.

        Flow:
            1. Run code in the stateful REPL
            2. If timeout → collect diagnostics
            3. Capture page snapshot
            4. Call post_exec_hook (your system processing)
            5. Log to history
            6. Return bundled result

        Args:
            code: Python code from the agent. Supports top-level await.
            timeout: Override default timeout for this step.
        """
        if not self._started:
            raise RuntimeError("Runtime not started. Call await runtime.start() first.")

        self._step_counter += 1
        effective_timeout = timeout if timeout is not None else self._default_timeout

        # Mark step start for instrumentation.
        step_start = self._instrumentation.mark_step_start()

        # 1. Execute
        t0 = asyncio.get_event_loop().time()
        output, error = await self._repl.execute(code, timeout=effective_timeout)
        duration_ms = (asyncio.get_event_loop().time() - t0) * 1000

        is_timeout = "TimeoutError: Execution exceeded" in error

        result = ExecutionResult(
            code=code,
            output=output.strip(),
            error=error.strip(),
            success=(error == ""),
            duration_ms=round(duration_ms, 2),
            step=self._step_counter,
        )

        # 2. If timeout → collect and save diagnostics
        if is_timeout:
            result.diagnostics = await self._collect_timeout_diagnostics(
                code=code,
                partial_stdout=output,
                step_start=step_start,
                elapsed_ms=duration_ms,
                timeout_limit=effective_timeout,
            )
            if self._diagnostics_dir:
                diag_path = (
                    self._diagnostics_dir
                    / f"timeout_step_{self._step_counter}"
                )
                try:
                    saved = result.diagnostics.save(diag_path)
                    logger.info("Timeout diagnostics saved to %s", saved)
                except Exception:
                    logger.exception("Failed to save timeout diagnostics")

        # 3. Snapshot — only if page is still alive
        page = self._get_live_page()
        if page:
            result.snapshot = await capture_snapshot(page, self._snap_config)

        # 4. Hook — your system's processing of the page
        if self._hook and page:
            result.hook_data = await self._run_hook(page, result)

        # 5. History
        self._history.append(result)

        return result

    async def _collect_timeout_diagnostics(
        self,
        code: str,
        partial_stdout: str,
        step_start: float,
        elapsed_ms: float,
        timeout_limit: float,
    ) -> TimeoutDiagnostics:
        """Gather all available diagnostic data after a timeout.

        This runs *after* the timeout has already occurred, so we need
        to be careful not to block forever on Playwright calls. Each
        piece of data is collected independently with its own try/except.
        """
        diag = TimeoutDiagnostics(
            partial_stdout=partial_stdout,
            elapsed_ms=elapsed_ms,
            timeout_limit_s=timeout_limit,
            code_executed=code,
        )

        # Drain instrumentation data for this step.
        console, pending, completed, failed = self._instrumentation.drain_since(
            step_start
        )
        diag.console_logs = console
        diag.pending_requests = pending
        diag.completed_requests = completed
        diag.failed_requests = failed

        # Try to get page state (with a short timeout to avoid blocking).
        page = self._get_live_page()
        if page:
            try:
                diag.page_url = page.url
            except Exception:
                pass

            try:
                diag.page_title = await asyncio.wait_for(
                    page.title(), timeout=3.0
                )
            except Exception:
                pass

            try:
                screenshot_bytes = await asyncio.wait_for(
                    page.screenshot(type="png", full_page=False),
                    timeout=5.0,
                )
                diag.screenshot_b64 = base64.b64encode(screenshot_bytes).decode()
            except Exception:
                pass

        logger.warning(
            "Timeout diagnostics collected for step %d:\n%s",
            self._step_counter,
            diag.summary(),
        )

        return diag

    # ── Helpers ──

    def _get_live_page(self) -> Optional[Page]:
        """
        Safely retrieve the current page object.

        Guards against two failure modes:
        - The agent called page.close()
        - The agent reassigned `page` to something else in the REPL
        """
        # Prefer what's actually in the REPL namespace (agent may have reassigned)
        repl_page = self._repl.get("page")
        if isinstance(repl_page, Page) and not repl_page.is_closed():
            return repl_page

        # Fall back to the browser manager's page
        mgr_page = self._browser_mgr.page
        if mgr_page and not mgr_page.is_closed():
            return mgr_page

        return None

    async def _run_hook(self, page: Page, result: ExecutionResult) -> Any:
        """Run the post-exec hook with error isolation."""
        try:
            return await self._hook(page, result)
        except Exception as e:
            return {"hook_error": type(e).__name__, "hook_message": str(e)}

    # ── Context manager ──

    async def __aenter__(self) -> ScrapingRuntime:
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()


# ═══════════════════════════════════════════════
#  MCP Tool Schema
# ═══════════════════════════════════════════════


def mcp_tool_schema() -> dict:
    """
    JSON schema for exposing `execute` as an MCP tool or
    OpenAI-style function call.
    """
    return {
        "name": "execute_browser_code",
        "description": (
            "Execute Python code in a stateful environment with a live "
            "Playwright browser. The `page` object is pre-injected and "
            "persists across calls. Use `await page.goto(url)` to navigate, "
            "`await page.click(selector)` to interact. All variables "
            "defined in one call are available in the next."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Supports top-level await.",
                }
            },
            "required": ["code"],
        },
    }