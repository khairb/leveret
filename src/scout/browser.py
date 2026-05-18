"""Browser configuration for Scout.

Provides :class:`LaunchOptions` — a TypedDict matching the kwargs of
Patchright's ``launch_persistent_context()``.  Users get full IDE
autocomplete; Scout gets a serializable config it can forward to all
browser launch sites (BrowserManager, subprocess wrapper, in-process
runner) without maintaining its own option list.

Usage::

    from scout import Scraper, LaunchOptions

    scraper = Scraper(
        url, task, schema=...,
        launch_options=LaunchOptions(
            proxy={"server": "http://proxy:8080"},
            locale="de-DE",
            timezone_id="Europe/Berlin",
        ),
    )

    # Plain dict also works — LaunchOptions is just a TypedDict
    scraper = Scraper(..., launch_options={"locale": "de-DE"})
"""

from __future__ import annotations

import asyncio
import os
import re
import threading
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

# Import Patchright's own sub-option TypedDicts so users get native types.
try:
    from patchright._impl._api_structures import (
        ClientCertificate,
        Geolocation,
        HttpCredentials,
        ProxySettings,
        ViewportSize,
    )
except ImportError:
    # Patchright not installed — define stubs so the TypedDict can
    # still be defined and used for type-checking without the runtime
    # dependency.
    ProxySettings = dict[str, Any]  # type: ignore[assignment,misc]
    ViewportSize = dict[str, int]  # type: ignore[assignment,misc]
    Geolocation = dict[str, Any]  # type: ignore[assignment,misc]
    HttpCredentials = dict[str, Any]  # type: ignore[assignment,misc]
    ClientCertificate = dict[str, Any]  # type: ignore[assignment,misc]

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


class LaunchOptions(TypedDict, total=False):
    """Browser launch options — forwarded to ``launch_persistent_context()``.

    Every key maps 1:1 to a Patchright/Playwright kwarg.  See the
    `Playwright docs <https://playwright.dev/python/docs/api/class-browsertype#browser-type-launch-persistent-context>`_
    for full descriptions.

    All fields are optional.  Unset fields use Scout's stealth defaults
    (``channel="chrome"``, ``locale="en-US"``, etc.).  To override a
    default, set it explicitly.
    """

    # ── Browser-level options ────────────────────────────────────
    channel: str
    executable_path: str | Path
    args: Sequence[str]
    ignore_default_args: bool | Sequence[str]
    handle_sigint: bool
    handle_sigterm: bool
    handle_sighup: bool
    timeout: float
    env: dict[str, str | float | bool]
    proxy: ProxySettings
    downloads_path: str | Path
    slow_mo: float
    chromium_sandbox: bool
    traces_dir: str | Path

    # ── Context-level options ────────────────────────────────────
    viewport: ViewportSize | None
    screen: ViewportSize
    no_viewport: bool
    ignore_https_errors: bool
    java_script_enabled: bool
    bypass_csp: bool
    user_agent: str
    locale: str
    timezone_id: str
    geolocation: Geolocation
    permissions: Sequence[str]
    extra_http_headers: dict[str, str]
    offline: bool
    http_credentials: HttpCredentials
    device_scale_factor: float
    is_mobile: bool
    has_touch: bool
    color_scheme: Literal["dark", "light", "no-preference", "null"]
    reduced_motion: Literal["no-preference", "null", "reduce"]
    forced_colors: Literal["active", "none", "null"]
    contrast: Literal["more", "no-preference", "null"]
    accept_downloads: bool
    base_url: str
    strict_selectors: bool
    service_workers: Literal["allow", "block"]
    storage_state: str | Path | dict[str, Any]
    client_certificates: list[ClientCertificate]

    # ── Recording options ────────────────────────────────────────
    record_har_path: str | Path
    record_har_omit_content: bool
    record_har_mode: Literal["full", "minimal"]
    record_har_content: Literal["attach", "embed", "omit"]
    record_har_url_filter: str | re.Pattern[str]
    record_video_dir: str | Path
    record_video_size: ViewportSize

    # ── Firefox-specific ─────────────────────────────────────────
    firefox_user_prefs: dict[str, str | float | bool]


# ── Stealth defaults ─────────────────────────────────────────────

STEALTH_ARGS: list[str] = [
    "--disable-blink-features=AutomationControlled",
    "--no-first-run",
    "--disable-extensions",
    "--disable-default-apps",
    "--disable-sync",
    "--disable-background-networking",
]


def _detect_browser_channel() -> str:
    """Detect which browser channel is available.

    Priority:
    1. ``SCOUT_BROWSER_CHANNEL`` env var (explicit override)
    2. ``chrome`` if Google Chrome is installed at the standard path
    3. ``chromium`` as fallback
    """
    env_channel = os.environ.get("SCOUT_BROWSER_CHANNEL")
    if env_channel:
        return env_channel

    # Check standard Google Chrome install locations
    chrome_paths = [
        Path("/opt/google/chrome/chrome"),  # Linux
        Path("/usr/bin/google-chrome"),  # Linux (symlink)
        Path("/usr/bin/google-chrome-stable"),  # Linux (alt)
    ]

    # Also check macOS
    mac_chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if mac_chrome.exists():
        return "chrome"

    for p in chrome_paths:
        if p.exists():
            return "chrome"

    return "chromium"


SCOUT_DEFAULTS: LaunchOptions = {
    "no_viewport": True,
    "bypass_csp": True,
    "locale": "en-US",
    "timezone_id": "America/New_York",
}


def resolve_launch_options(
    user_options: LaunchOptions | None,
    *,
    headless: bool,
    demo: bool = False,
) -> dict[str, Any]:
    """Merge user options with Scout's stealth defaults.

    Merge rules:
    - User options override Scout defaults.
    - ``channel`` is auto-detected if not set by user (chrome → chromium fallback).
    - ``args`` are *extended* (stealth args + user args), not replaced.
    - ``headless`` comes from Scraper, not from launch_options.
    - ``user_data_dir`` is always set by Scout (temp dir) — cannot be
      overridden.
    - When ``demo`` is True, a ``_demo`` sentinel is included in the
      returned dict so that ``BrowserManager`` can skip the manual
      ``set_viewport_size`` call — letting the page fill the window
      at whatever size the OS provides.

    Returns a dict ready to be unpacked into
    ``launch_persistent_context(**result)``.
    """
    merged: dict[str, Any] = {
        **SCOUT_DEFAULTS,
        "channel": _detect_browser_channel(),
        "headless": headless,
    }

    if demo:
        merged["_demo"] = True

    if user_options:
        # Merge args: stealth + user-provided, no duplicates
        user_args = list(user_options.get("args", []))
        if "args" in user_options:
            # Don't let user args clobber stealth args
            user_opts = {k: v for k, v in user_options.items() if k != "args"}
            merged.update(user_opts)
        else:
            merged.update(user_options)
        merged["args"] = STEALTH_ARGS + [a for a in user_args if a not in STEALTH_ARGS]
    else:
        merged["args"] = list(STEALTH_ARGS)

    return merged


# ── Demo layout ─────────────────────────────────────────────────


def compute_demo_layout(
    screen_width: int,
    screen_height: int,
) -> dict[str, int]:
    """Compute window sizes for the demo 20/80 split.

    Returns a dict with keys: panel_width, page_width, panel_x, height.
    """
    if screen_width < 800:
        screen_width = 1920
    if screen_height < 400:
        screen_height = 1080
    panel_width = max(320, min(500, int(screen_width * 0.2)))
    page_width = screen_width - panel_width
    return {
        "panel_width": panel_width,
        "page_width": page_width,
        "panel_x": page_width,
        "height": screen_height,
    }


def compute_expanded_layout(
    screen_width: int,
    screen_height: int,
) -> dict[str, int]:
    """Compute window sizes for the expanded 50/50 results view.

    Returns a dict with keys: panel_width, page_width, panel_x, height.
    """
    if screen_width < 800:
        screen_width = 1920
    if screen_height < 400:
        screen_height = 1080
    panel_width = screen_width // 2
    page_width = screen_width - panel_width
    return {
        "panel_width": panel_width,
        "page_width": page_width,
        "panel_x": page_width,
        "height": screen_height,
    }


# ── Shared Browser ──────────────────────────────────────────────


class Browser:
    """Shared browser for running multiple scrapers efficiently.

    Instead of each :class:`Scraper` launching its own Chrome instance,
    pass a shared ``Browser`` and all scrapers open tabs in the same
    browser process.

    Usage::

        from scout import Scraper, Browser

        # One browser, multiple scrapers
        with Browser(headless=True) as browser:
            s1 = Scraper(url1, task1, schema=..., script="s1.py", browser=browser)
            s2 = Scraper(url2, task2, schema=..., script="s2.py", browser=browser)
            r1 = s1.run()
            r2 = s2.run()

        # Async
        async with Browser(headless=True) as browser:
            r1, r2 = await asyncio.gather(
                s1.async_run(),
                s2.async_run(),
            )

        # Without browser= everything works as before (backward compatible)
        scraper = Scraper(url, task, schema=..., script="s.py")
        scraper.run()
    """

    def __init__(
        self,
        *,
        headless: bool = True,
        launch_options: LaunchOptions | None = None,
    ) -> None:
        self._headless = headless
        self._user_launch_options = launch_options
        self._browser_mgr: Any = None
        self._bg_loop: Any = None
        self._bg_thread: threading.Thread | None = None
        self._started = False
        self._start_time: float = 0.0

    @property
    def is_running(self) -> bool:
        """Whether the browser is currently running."""
        return self._started

    def _get_resolved_options(self) -> dict[str, Any]:
        """Resolve launch options with defaults."""
        return resolve_launch_options(
            self._user_launch_options,
            headless=self._headless,
        )

    # ── Async interface ──────────────────────────────────────

    async def _start(self) -> None:
        """Launch the browser (async)."""
        if self._started:
            return
        from .runtime.environment import BrowserManager

        self._browser_mgr = BrowserManager(
            headless=self._headless,
            launch_options=self._get_resolved_options(),
        )
        await self._browser_mgr.start()
        self._started = True
        self._start_time = time.monotonic()

    async def _stop(self) -> None:
        """Close the browser (async). Idempotent."""
        if self._browser_mgr is not None:
            try:
                await self._browser_mgr.stop()
            except Exception:
                pass
            self._browser_mgr = None
        self._started = False

    async def _new_page(self) -> Any:
        """Create a new page (tab) in the shared browser.

        Returns a Playwright Page object.

        Raises:
            RuntimeError: If the browser is not running.
        """
        if not self._started or self._browser_mgr is None:
            raise RuntimeError(
                "Browser is not running. "
                "Use 'with Browser() as browser:' or call 'await browser.start()'."
            )
        return await self._browser_mgr.new_page()

    async def new_page(self) -> Any:
        """Deprecated — use _new_page() internally."""
        import warnings

        warnings.warn("Browser.new_page() is deprecated", DeprecationWarning, stacklevel=2)
        return await self._new_page()

    async def __aenter__(self) -> Browser:
        await self._start()
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        await self._stop()
        return False

    # ── Sync interface ───────────────────────────────────────

    def __enter__(self) -> Browser:
        self._bg_loop = asyncio.new_event_loop()
        self._bg_thread = threading.Thread(
            target=self._bg_loop.run_forever,
            daemon=True,
            name="scout-shared-browser",
        )
        self._bg_thread.start()

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._start(),
                self._bg_loop,
            )
            future.result(timeout=60)
        except Exception:
            # Browser failed to start — clean up the thread
            self._bg_loop.call_soon_threadsafe(self._bg_loop.stop)
            self._bg_thread.join(timeout=5)
            self._bg_loop.close()
            self._bg_loop = None
            self._bg_thread = None
            raise
        return self

    def __exit__(self, *exc: Any) -> bool:
        if self._bg_loop is not None:
            future = asyncio.run_coroutine_threadsafe(
                self._stop(),
                self._bg_loop,
            )
            try:
                future.result(timeout=15)
            except Exception:
                pass
            self._bg_loop.call_soon_threadsafe(self._bg_loop.stop)
            if self._bg_thread is not None:
                self._bg_thread.join(timeout=10)
            self._bg_loop.close()
            self._bg_loop = None
            self._bg_thread = None
        return False

    def close(self) -> None:
        """Close the browser and release resources. Idempotent.

        From sync context: runs cleanup directly.
        From async context: use ``await browser._stop()`` or the
        ``async with`` pattern instead.
        """
        if self._bg_loop is not None:
            # Sync path — running in background thread
            self.__exit__(None, None, None)
        elif self._started:
            asyncio.run(self._stop())

    def __del__(self) -> None:
        if getattr(self, "_started", False):
            warnings.warn(
                "Browser was garbage-collected while still running. "
                "Use 'with Browser():' or call 'browser.close()' "
                "to avoid resource leaks.",
                ResourceWarning,
                stacklevel=2,
            )
