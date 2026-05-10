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

import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

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
    ProxySettings = Dict[str, Any]  # type: ignore[assignment,misc]
    ViewportSize = Dict[str, int]  # type: ignore[assignment,misc]
    Geolocation = Dict[str, Any]  # type: ignore[assignment,misc]
    HttpCredentials = Dict[str, Any]  # type: ignore[assignment,misc]
    ClientCertificate = Dict[str, Any]  # type: ignore[assignment,misc]

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
    executable_path: Union[str, Path]
    args: Sequence[str]
    ignore_default_args: Union[bool, Sequence[str]]
    handle_sigint: bool
    handle_sigterm: bool
    handle_sighup: bool
    timeout: float
    env: Dict[str, Union[str, float, bool]]
    proxy: ProxySettings
    downloads_path: Union[str, Path]
    slow_mo: float
    chromium_sandbox: bool
    traces_dir: Union[str, Path]

    # ── Context-level options ────────────────────────────────────
    viewport: Optional[ViewportSize]
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
    extra_http_headers: Dict[str, str]
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
    storage_state: Union[str, Path, Dict[str, Any]]
    client_certificates: List[ClientCertificate]

    # ── Recording options ────────────────────────────────────────
    record_har_path: Union[str, Path]
    record_har_omit_content: bool
    record_har_mode: Literal["full", "minimal"]
    record_har_content: Literal["attach", "embed", "omit"]
    record_har_url_filter: Union[str, "re.Pattern[str]"]
    record_video_dir: Union[str, Path]
    record_video_size: ViewportSize

    # ── Firefox-specific ─────────────────────────────────────────
    firefox_user_prefs: Dict[str, Union[str, float, bool]]


# ── Stealth defaults ─────────────────────────────────────────────

STEALTH_ARGS: list[str] = [
    "--disable-blink-features=AutomationControlled",
    "--no-first-run",
    "--disable-extensions",
    "--disable-default-apps",
    "--disable-sync",
    "--disable-background-networking",
]

SCOUT_DEFAULTS: LaunchOptions = {
    "channel": "chrome",
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
    merged: dict[str, Any] = {**SCOUT_DEFAULTS, "headless": headless}

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
        merged["args"] = STEALTH_ARGS + [
            a for a in user_args if a not in STEALTH_ARGS
        ]
    else:
        merged["args"] = list(STEALTH_ARGS)

    return merged


# ── Demo layout ─────────────────────────────────────────────────

def compute_demo_layout(
    screen_width: int, screen_height: int,
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
