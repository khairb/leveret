"""Engine wrapper generation for the structured function format.

The agent writes ``async def scrape(page, url, checkpoint) -> JsonValue``.
This module generates:

1. A subprocess wrapper that launches a browser, calls the function,
   and serializes the return value (used by the engine for validation).
2. A standalone script the user can run directly (``python scraper.py``).
3. Utilities for parsing the return value from subprocess stdout.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..runtime.environment import BrowserManager

# ═══════════════════════════════════════════════════════════════
#  Markers for return value extraction from subprocess stdout
# ═══════════════════════════════════════════════════════════════

RETURN_VALUE_START = "__SCOUT_RETURN_VALUE_START_a7b3__"
RETURN_VALUE_END = "__SCOUT_RETURN_VALUE_END_a7b3__"

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

    data = {{
        "id": cp_id,
        "label": label,
        "url": url,
        "title": title,
        "timestamp_s": round(elapsed, 1),
        "element_count": element_count,
        "visible_text": visible_text,
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
    url = {url!r}
    profile_dir = tempfile.mkdtemp(prefix="scraper_profile_")
    try:
        async with async_playwright() as p:
            context = await p.chromium.launch_persistent_context(
                user_data_dir=profile_dir,
                channel="chrome",
                headless=False,
                no_viewport=True,
                bypass_csp=True,
                locale="en-US",
                timezone_id="America/New_York",
                args={stealth_args!r},
            )
            page = context.pages[0] if context.pages else await context.new_page()
            await page.set_viewport_size({{"width": 1920, "height": 1080}})
            await page.goto(url, wait_until="domcontentloaded")

{call_section}

            await context.close()
    finally:
        shutil.rmtree(profile_dir, ignore_errors=True)
"""


# ═══════════════════════════════════════════════════════════════
#  Subprocess wrapper (used by the engine for validation)
# ═══════════════════════════════════════════════════════════════

def generate_subprocess_wrapper(
    agent_code: str,
    url: str,
    checkpoint_dir: str,
) -> str:
    """Generate the subprocess wrapper script.

    The wrapper:
    - Pre-imports common modules
    - Embeds the agent's function code
    - Defines an inline checkpoint function
    - Launches a browser with production defaults
    - Navigates to the URL
    - Calls ``scrape(page, url, checkpoint)``
    - Serializes the return value between markers
    """
    stealth_args = list(BrowserManager._STEALTH_ARGS)
    checkpoint_code = _CHECKPOINT_TEMPLATE.format(checkpoint_dir=checkpoint_dir)

    call_section = f"""\
            # Create checkpoint closure: agent calls checkpoint("label"),
            # wrapper injects page automatically.
            async def checkpoint(label, data_preview=None):
                await _raw_checkpoint(page, label, data_preview=data_preview)

            data = await scrape(page, url, checkpoint)

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

    runner = _ENGINE_RUNNER.format(
        url=url,
        stealth_args=stealth_args,
        call_section=call_section,
    )

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


# ═══════════════════════════════════════════════════════════════
#  Standalone script (for the user to run directly)
# ═══════════════════════════════════════════════════════════════

def generate_standalone_script(
    agent_code: str,
    url: str,
    task: str,
) -> str:
    """Generate a self-contained, runnable script for the user.

    Combines the agent's function with a lightweight engine wrapper.
    The user can ``python scraper.py`` with no Scout dependency.
    """
    stealth_args = list(BrowserManager._STEALTH_ARGS)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    safe_task = task.replace('"""', r'\"\"\"')

    call_section = """\
            data = await scrape(page, url, _checkpoint)

            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))"""

    runner = _ENGINE_RUNNER.format(
        url=url,
        stealth_args=stealth_args,
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
    return_value_json = raw_stdout[
        start_idx + len(RETURN_VALUE_START) : end_idx
    ].strip()

    # Include any output after the end marker (unlikely but safe).
    after = raw_stdout[end_idx + len(RETURN_VALUE_END) :].strip()
    if after:
        clean_stdout = clean_stdout + "\n" + after if clean_stdout else after

    return clean_stdout, return_value_json


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
        parts.append(
            "(no return value — function raised or did not return)"
        )

    return "\n".join(parts)
