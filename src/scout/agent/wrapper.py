"""Engine wrapper generation for the structured function format.

The agent writes ``async def scrape(page, start_url, checkpoint) -> JsonValue``.
This module generates:

1. A subprocess wrapper that launches a browser, calls the function,
   and serializes the return value (used by the engine for validation).
2. A standalone script the user can run directly (``python scraper.py``).
3. Utilities for parsing the return value from subprocess stdout.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

# ═══════════════════════════════════════════════════════════════
#  Markers for return value extraction from subprocess stdout
# ═══════════════════════════════════════════════════════════════

RETURN_VALUE_START = "__SCOUT_RETURN_VALUE_START_a7b3__"
RETURN_VALUE_END = "__SCOUT_RETURN_VALUE_END_a7b3__"

# Markers for page signal extraction (auto-fix S6/S7).
# Printed to stdout on script failure when collect_page_signals=True.
PAGE_SIGNALS_START = "__SCOUT_PAGE_SIGNALS_START_c9e1__"
PAGE_SIGNALS_END = "__SCOUT_PAGE_SIGNALS_END_c9e1__"

# Pre-imported modules available in the function's execution scope.
_PRE_IMPORTS = """\
import json
import re
import math
import os
import time
import asyncio
import tempfile
import shutil
from datetime import datetime
from urllib.parse import urljoin, urlparse
"""


# ═══════════════════════════════════════════════════════════════
#  Checkpoint function (embedded in the wrapper, not imported)
# ═══════════════════════════════════════════════════════════════

_CHECKPOINT_TEMPLATE = '''\
_CP_DIR = {checkpoint_dir!r}
_CP_START = time.time()
_CP_COUNTER = 0


async def _raw_checkpoint(page, label, *, data_preview=None):
    """Capture a checkpoint: one-line summary to stdout, full state to disk."""
    global _CP_COUNTER
    _CP_COUNTER += 1
    cp_id = f"CP-{{_CP_COUNTER}}"

    url = page.url
    title = await page.title()
    elapsed = time.time() - _CP_START

    info = await page.evaluate(
        """() => {{
            const text = document.body ? document.body.innerText : "";
            const count = document.querySelectorAll("*").length;
            return {{ text: text.substring(0, 5000), count }};
        }}"""
    )

    visible_text = info.get("text", "")
    element_count = info.get("count", 0)

    # Capture raw HTML for sectioning at expansion time.
    try:
        raw_html = await page.content()
    except Exception:
        raw_html = ""

    data = {{
        "id": cp_id,
        "label": label,
        "url": url,
        "title": title,
        "timestamp_s": round(elapsed, 1),
        "element_count": element_count,
        "visible_text": visible_text,
        "html": raw_html,
        "data_preview": data_preview,
    }}
    os.makedirs(_CP_DIR, exist_ok=True)
    with open(os.path.join(_CP_DIR, f"{{cp_id}}.json"), "w") as f:
        json.dump(data, f, indent=2, default=str)

    t = (title[:50] + "\\u2026") if len(title) > 50 else title
    dp = f" | data_preview={{len(data_preview)}} items" if data_preview else ""
    print(
        f"[{{cp_id}} {{label}}] url={{url}} | "
        f"title=\\"{{t}}\\" | elements={{element_count}}{{dp}} | {{elapsed:.1f}}s"
    )
'''


# ═══════════════════════════════════════════════════════════════
#  Engine runner template (shared between subprocess and standalone)
# ═══════════════════════════════════════════════════════════════

_ENGINE_RUNNER = """\
async def _run():
    start_url = {url!r}
    _launch_opts = {launch_options!r}
    _user_profile = _launch_opts.pop("user_data_dir", None)
    profile_dir = {profile_dir!r} or _user_profile or tempfile.mkdtemp(prefix="scraper_profile_")
    _owns_profile = ({profile_dir!r} is None and _user_profile is None)
    try:
        async with async_playwright() as p:
            context = await p.chromium.launch_persistent_context(
                user_data_dir=profile_dir,
                **_launch_opts,
            )
            page = context.pages[0] if context.pages else await context.new_page()
            _viewport = _launch_opts.get("viewport") or {{"width": 1920, "height": 1080}}
            await page.set_viewport_size(_viewport)
            _goto_response = await page.goto(start_url, wait_until="domcontentloaded")
{inputs_line}
{call_section}

            await context.close()
    finally:
        if _owns_profile:
            shutil.rmtree(profile_dir, ignore_errors=True)
"""


# ═══════════════════════════════════════════════════════════════
#  Subprocess wrapper (used by the engine for validation)
# ═══════════════════════════════════════════════════════════════


def generate_subprocess_wrapper(
    agent_code: str,
    url: str,
    checkpoint_dir: str,
    *,
    collect_page_signals: bool = False,
    sandbox: bool = False,
    launch_options: dict | None = None,
    profile_dir: str | None = None,
    inputs: dict | None = None,
) -> str:
    """Generate the subprocess wrapper script.

    The wrapper:
    - Pre-imports common modules
    - Embeds the agent's function code
    - Defines an inline checkpoint function
    - Launches a browser with production defaults
    - Navigates to the URL
    - Calls ``scrape(page, start_url, checkpoint)``
    - Serializes the return value between markers

    Args:
        agent_code: The ``async def scrape(...)`` function source code.
        url: Target URL for navigation.
        checkpoint_dir: Directory for checkpoint files.
        collect_page_signals: When True, attach a response listener and
            collect page-level signals (URL, content, HTTP status, headers,
            cookies) on script failure. The signals are serialized as JSON
            between ``PAGE_SIGNALS_START/END`` markers in stdout. Used by
            the auto-fix diagnosis loop (spec §6/§7).
        launch_options: Resolved browser launch options dict. Forwarded
            as ``**kwargs`` to ``launch_persistent_context()``.
    """
    if launch_options is None:
        from ..browser import resolve_launch_options

        launch_options = resolve_launch_options(None, headless=False)

    # Strip internal sentinels that are not valid Playwright kwargs.
    launch_options = {k: v for k, v in launch_options.items() if not k.startswith("_")}
    checkpoint_code = _CHECKPOINT_TEMPLATE.format(checkpoint_dir=checkpoint_dir)

    # When profile_dir is None, the subprocess creates its own at runtime
    # (via the `or tempfile.mkdtemp(...)` in the template).
    # When the parent provides one, it owns cleanup — no orphan risk.

    has_inputs = inputs is not None and len(inputs) > 0
    if collect_page_signals:
        call_section = _build_call_section_with_signals(has_inputs=has_inputs)
    else:
        call_section = _build_call_section_basic(has_inputs=has_inputs)

    # Embed inputs as a JSON literal in the wrapper when provided.
    if has_inputs:
        inputs_json = json.dumps(inputs, ensure_ascii=False)
        inputs_line = f"\n            _INPUTS = json.loads({inputs_json!r})\n"
    else:
        inputs_line = ""

    runner = _ENGINE_RUNNER.format(
        url=url,
        launch_options=launch_options,
        profile_dir=profile_dir,
        inputs_line=inputs_line,
        call_section=call_section,
    )

    if sandbox:
        # In sandbox mode, the agent code runs in a restricted namespace
        # that does NOT have os/shutil/tempfile. The checkpoint and engine
        # runner still have full access (they're trusted wrapper code).
        sandbox_setup = _build_sandbox_setup(agent_code)
        parts = [
            "#!/usr/bin/env python3",
            "# Scout engine wrapper — auto-generated, do not edit.\n",
            _PRE_IMPORTS,
            "from patchright.async_api import async_playwright\n",
            "# ── Checkpoint ──\n",
            checkpoint_code,
            "# ── Sandbox: Agent Code in restricted namespace ──\n",
            sandbox_setup,
            "\n# ── Engine Runner ──\n",
            runner,
            "asyncio.run(_run())",
        ]
    else:
        parts = [
            "#!/usr/bin/env python3",
            "# Scout engine wrapper — auto-generated, do not edit.\n",
            _PRE_IMPORTS,
            "from patchright.async_api import async_playwright\n",
            "# ── Checkpoint ──\n",
            checkpoint_code,
            "# ── Agent Code ──\n",
            agent_code,
            "\n# ── Engine Runner ──\n",
            runner,
            "asyncio.run(_run())",
        ]
    return "\n".join(parts) + "\n"


def _build_sandbox_setup(agent_code: str) -> str:
    """Embed agent code as a string that runs in a restricted namespace.

    The restricted namespace has:
    - Import whitelist (no os, subprocess, etc.)
    - Safe asyncio proxy (no create_subprocess_exec, etc.)
    - No os, shutil, tempfile access
    - Restricted builtins (no exec, eval, open, etc.)
    """
    # Embed agent code safely using repr() for proper escaping
    escaped_code = repr(agent_code)
    return f"""\
from scout.runtime.sandbox import (
    compile_restricted_agent_code,
    build_restricted_globals,
    build_safe_pre_imports,
)

_agent_source = {escaped_code}
_agent_compiled = compile_restricted_agent_code(_agent_source, "<agent>")
_agent_ns = build_restricted_globals(build_safe_pre_imports())
exec(_agent_compiled, _agent_ns)
scrape = _agent_ns["scrape"]
"""


def _build_call_section_basic(*, has_inputs: bool = False) -> str:
    """Build the call section without signal collection (default)."""
    scrape_call = (
        "data = await scrape(page, start_url, _INPUTS, checkpoint)"
        if has_inputs
        else "data = await scrape(page, start_url, checkpoint)"
    )
    return f"""\
            # Create checkpoint closure: agent calls checkpoint("label"),
            # wrapper injects page automatically.
            async def checkpoint(label, data_preview=None):
                await _raw_checkpoint(page, label, data_preview=data_preview)

            {scrape_call}

            # Serialize return value between markers.
            try:
                rv_json = json.dumps(data, ensure_ascii=False, indent=2, default=str)
            except TypeError as exc:
                type_name = type(data).__name__
                print(
                    f"ERROR: scrape() returned type '{{type_name}}' which is not "
                    f"JSON-serializable: {{exc}}",
                    file=__import__('sys').stderr,
                )
                __import__('sys').exit(1)

            print("{RETURN_VALUE_START}")
            print(rv_json)
            print("{RETURN_VALUE_END}")"""


def _build_call_section_with_signals(*, has_inputs: bool = False) -> str:
    """Build the call section with page signal collection (auto-fix §6/§7).

    Collects page signals (URL, content, HTTP status, headers, cookies)
    on EVERY attempt — both success and failure. Signals are needed on
    failure for all categories, and on success for Category G (script
    succeeded but schema validation fails later).

    The initial document response from ``page.goto()`` is seeded from
    ``_goto_response`` (captured in the engine runner template). A
    response listener captures any subsequent document responses from
    navigations triggered by the scrape function.

    Every signal access is wrapped in try/except — the diagnostic system
    must never crash (dev guide).
    """
    return f"""\
            # Seed with the goto response (auto-fix S6).
            # _goto_response is set by the engine runner template.
            _doc_responses = []
            if _goto_response is not None:
                _doc_responses.append(_goto_response)

            # Also capture subsequent document responses from the script.
            def _on_doc_response(response):
                try:
                    if response.request.resource_type == "document":
                        _doc_responses.append(response)
                except Exception:
                    pass
            page.on("response", _on_doc_response)

            # Create checkpoint closure.
            async def checkpoint(label, data_preview=None):
                await _raw_checkpoint(page, label, data_preview=data_preview)

            # Run the scrape function, capturing any exception.
            _scrape_exc = None
            _scrape_data = None
            try:
                _scrape_data = await scrape(page, start_url, {"_INPUTS, checkpoint" if has_inputs else "checkpoint"})
            except Exception as _exc:
                _scrape_exc = _exc

            # Collect page signals on every attempt (auto-fix S6/S7).
            # Needed on failure for all categories, and on success
            # for Category G (schema validation may fail later).
            # Every access is defensive — page may be crashed/closed.
            _signals = {{}}
            try:
                _signals["page_url"] = page.url
            except Exception:
                pass
            try:
                _signals["content"] = await asyncio.wait_for(
                    page.content(), timeout=5.0,
                )
            except Exception:
                pass
            if _doc_responses:
                _last_resp = _doc_responses[-1]
                try:
                    _signals["http_status"] = _last_resp.status
                except Exception:
                    pass
                try:
                    _signals["headers"] = dict(_last_resp.headers)
                except Exception:
                    pass
            try:
                _cookies = await context.cookies()
                _signals["cookies"] = [
                    {{"name": c["name"], "value": c.get("value", "")}}
                    for c in _cookies
                ]
            except Exception:
                pass
            # Serialize signals between markers.
            try:
                _sig_json = json.dumps(_signals, ensure_ascii=False)
                print("{PAGE_SIGNALS_START}")
                print(_sig_json)
                print("{PAGE_SIGNALS_END}")
            except Exception:
                pass

            # Re-raise if the script failed.
            if _scrape_exc is not None:
                raise _scrape_exc

            # Success — serialize return value between markers.
            try:
                rv_json = json.dumps(_scrape_data, ensure_ascii=False, indent=2, default=str)
            except TypeError as exc:
                type_name = type(_scrape_data).__name__
                print(
                    f"ERROR: scrape() returned type '{{type_name}}' which is not "
                    f"JSON-serializable: {{exc}}",
                    file=__import__('sys').stderr,
                )
                __import__('sys').exit(1)

            print("{RETURN_VALUE_START}")
            print(rv_json)
            print("{RETURN_VALUE_END}")"""


# ═══════════════════════════════════════════════════════════════
#  Standalone script (for the user to run directly)
# ═══════════════════════════════════════════════════════════════


def generate_standalone_script(
    agent_code: str,
    url: str,
    task: str,
    inputs: dict | None = None,
) -> str:
    """Generate a self-contained, runnable script for the user.

    Combines the agent's function with a lightweight engine wrapper.
    The user can ``python scraper.py`` with no Scout dependency.
    """
    from ..browser import resolve_launch_options

    default_opts = resolve_launch_options(None, headless=False)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    safe_task = task.replace('"""', r"\"\"\"")

    has_inputs = inputs is not None and len(inputs) > 0
    if has_inputs:
        scrape_call = "data = await scrape(page, start_url, _INPUTS, _checkpoint)"
    else:
        scrape_call = "data = await scrape(page, start_url, _checkpoint)"

    call_section = f"""\
            {scrape_call}

            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))"""

    if has_inputs:
        inputs_json = json.dumps(inputs, ensure_ascii=False)
        inputs_line = f"\n            _INPUTS = json.loads({inputs_json!r})\n"
    else:
        inputs_line = ""

    runner = _ENGINE_RUNNER.format(
        url=url,
        launch_options=default_opts,
        profile_dir=None,
        inputs_line=inputs_line,
        call_section=call_section,
    )

    header = f'''\
#!/usr/bin/env python3
"""Scout — standalone scraper.

Task: {safe_task}
URL:  {url}
Generated: {now}
"""
'''

    checkpoint_fn = """\
async def _checkpoint(label, data_preview=None):
    dp = f" | data_preview={len(data_preview)} items" if data_preview else ""
    print(f"[checkpoint] {label}{dp}")
"""

    parts = [
        header,
        _PRE_IMPORTS,
        "from patchright.async_api import async_playwright\n",
        "# ── Scraping Logic ──\n",
        agent_code,
        "\n# ── Runner (auto-generated by Scout) ──\n",
        checkpoint_fn,
        runner,
        'if __name__ == "__main__":',
        "    asyncio.run(_run())",
    ]
    return "\n".join(parts) + "\n"


# ═══════════════════════════════════════════════════════════════
#  Return value parsing from subprocess stdout
# ═══════════════════════════════════════════════════════════════


def parse_return_value(raw_stdout: str) -> tuple[str, str | None]:
    """Split subprocess stdout into progress output and return value.

    Returns:
        ``(clean_stdout, return_value_json)`` — *return_value_json* is
        ``None`` if the markers were not found (e.g. the function raised
        before returning).
    """
    start_idx = raw_stdout.find(RETURN_VALUE_START)
    end_idx = raw_stdout.find(RETURN_VALUE_END)

    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return raw_stdout, None

    clean_stdout = raw_stdout[:start_idx].rstrip()
    return_value_json = raw_stdout[start_idx + len(RETURN_VALUE_START) : end_idx].strip()

    # Include any output after the end marker (unlikely but safe).
    after = raw_stdout[end_idx + len(RETURN_VALUE_END) :].strip()
    if after:
        clean_stdout = clean_stdout + "\n" + after if clean_stdout else after

    return clean_stdout, return_value_json


def parse_page_signals(raw_stdout: str) -> dict[str, Any] | None:
    """Extract page signals JSON from subprocess stdout.

    Looks for the ``PAGE_SIGNALS_START/END`` markers and parses the
    JSON between them. Returns a dict with keys like ``http_status``,
    ``page_url``, ``content``, ``headers``, ``cookies``.

    Returns ``None`` if markers are not found (e.g. old wrapper format,
    successful execution, or signal serialization failed).

    The caller converts the dict to a ``PageSignals`` dataclass.
    """
    start_idx = raw_stdout.find(PAGE_SIGNALS_START)
    end_idx = raw_stdout.find(PAGE_SIGNALS_END)

    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return None

    signals_json = raw_stdout[start_idx + len(PAGE_SIGNALS_START) : end_idx].strip()

    if not signals_json:
        return None

    try:
        signals = json.loads(signals_json)
    except (ValueError, TypeError):
        return None

    if not isinstance(signals, dict):
        return None

    return signals


def build_combined_output(
    stdout: str,
    return_value_json: str | None,
) -> str:
    """Combine stdout and return value for the validator.

    The validator sees both as a single text block, processed through
    its existing output management logic.
    """
    parts = []

    if stdout.strip():
        parts.append(stdout.strip())

    parts.append("\n── Return Value (JSON) ──")

    if return_value_json is not None:
        parts.append(return_value_json)
    else:
        parts.append("(no return value — function raised or did not return)")

    return "\n".join(parts)
