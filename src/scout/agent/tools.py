"""Tool definitions and execution dispatch for the scraping agent.

Single tool:
    - ``python``: Execute code in the stateful Patchright environment.

The agent also has two built-in REPL functions (``show_page`` and
``zoom_section``) which are injected separately — see ``bridge.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .timeout_predict import predict_timeout

if TYPE_CHECKING:
    from ..runtime.environment import ScrapingRuntime

# ═══════════════════════════════════════════════════════════════
#  Timeout Limits
# ═══════════════════════════════════════════════════════════════

MIN_TIMEOUT: float = 1.0
"""Floor for any explicit timeout the agent provides."""

MAX_TIMEOUT: float = 300.0
"""Ceiling for any timeout (explicit or predicted)."""


# ═══════════════════════════════════════════════════════════════
#  Raw HTML blocker
# ═══════════════════════════════════════════════════════════════

# Patterns that dump full-page HTML into stdout, wasting tokens.
_RAW_HTML_PATTERNS = [
    # page.content() — returns entire page HTML.
    (re.compile(r'\bpage\s*\.\s*content\s*\('), "page.content()"),
    # page.inner_html("html") / page.inner_html("body") — same effect.
    (re.compile(r'\.\s*inner_html\s*\(\s*["\'](?:html|body)["\']'), '.inner_html("html"/"body")'),
    # document.documentElement.outerHTML inside evaluate().
    (re.compile(r'document\s*\.\s*documentElement\s*\.\s*outerHTML'), "document.documentElement.outerHTML"),
    # document.body.innerHTML in evaluate() targeting the whole body.
    (re.compile(r'document\s*\.\s*body\s*\.\s*innerHTML'), "document.body.innerHTML"),
]

_RAW_HTML_REJECTION = (
    "Script blocked: your code uses {pattern}, which dumps the entire "
    "page HTML into the conversation and wastes context.\n\n"
    "This is unnecessary — you already have structured tools for "
    "viewing the page:\n"
    "  - `await show_page(page)` — shows a clean, structured view "
    "of the full page with section IDs\n"
    "  - `await zoom_section(page, 'S-3')` — shows the full DOM of "
    "a specific section for detailed inspection\n\n"
    "Use these instead. They give you better information in less space."
)


def _check_raw_html_patterns(code: str) -> str:
    """Return a rejection message if *code* contains raw HTML dump patterns.

    Returns an empty string if the code is clean.
    """
    for pattern, label in _RAW_HTML_PATTERNS:
        if pattern.search(code):
            return _RAW_HTML_REJECTION.format(pattern=label)
    return ""


# ═══════════════════════════════════════════════════════════════
#  Tool Schemas
# ═══════════════════════════════════════════════════════════════

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "python",
        "description": (
            "Execute Python code in a stateful environment with a live "
            "Patchright browser. The `page` object is pre-injected and "
            "variables persist across calls. Supports top-level `await`. "
            "Two built-in functions are available: "
            "`await show_page(page)` to see the page, and "
            "`await zoom_section(page, section_id)` to inspect HTML structure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python code to execute. "
                        "Use `await` for Patchright async calls."
                    ),
                },
                "timeout": {
                    "type": "number",
                    "description": (
                        "Max seconds to wait for this code to finish. "
                        "Auto-calculated from code complexity when omitted. "
                        "Override for slow operations "
                        "like page loads or large extractions (max 300)."
                    ),
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "last_resort_antibot_escape",
        "description": (
            "LAST RESORT ONLY — Signal that the target website has an "
            "anti-bot or CAPTCHA system that you cannot bypass after "
            "exhausting every possible strategy. Calling this tool "
            "terminates the run immediately.\n\n"
            "Do NOT call this tool unless you have genuinely tried "
            "everything: waiting for challenges to resolve, navigating "
            "around blocked pages, trying alternative URLs or entry "
            "points, adjusting timing, and using different interaction "
            "patterns. This is your absolute last option when the "
            "website is fundamentally inaccessible to automation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "provider": {
                    "type": "string",
                    "description": (
                        "The anti-bot provider or CAPTCHA system "
                        "blocking access (e.g. 'Cloudflare', "
                        "'reCAPTCHA', 'DataDome', 'Akamai', "
                        "'unknown')."
                    ),
                },
                "strategies_tried": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of every strategy you attempted to "
                        "bypass the block. Be specific — e.g. "
                        "'waited 10s for Cloudflare challenge to "
                        "auto-resolve', 'tried navigating to /api "
                        "endpoint directly', 'attempted different "
                        "interaction timing'."
                    ),
                },
                "page_evidence": {
                    "type": "string",
                    "description": (
                        "What you observed on the page that confirms "
                        "the block — e.g. 'Cloudflare challenge page "
                        "with title Just a moment, challenge form "
                        "present, no way to proceed without human "
                        "CAPTCHA solving'."
                    ),
                },
            },
            "required": [
                "provider",
                "strategies_tried",
                "page_evidence",
            ],
        },
    },
]


# ═══════════════════════════════════════════════════════════════
#  Tool Result
# ═══════════════════════════════════════════════════════════════

@dataclass
class ToolResult:
    """The outcome of executing a single tool call."""

    tool_use_id: str
    name: str
    content: str
    is_error: bool = False


# ═══════════════════════════════════════════════════════════════
#  Execution Dispatch
# ═══════════════════════════════════════════════════════════════

async def execute_tool(
    name: str,
    arguments: dict[str, Any],
    tool_use_id: str,
    runtime: ScrapingRuntime,
) -> ToolResult:
    """Execute a tool call and return the formatted result."""
    try:
        if name == "python":
            return await _exec_python(
                arguments.get("code", ""),
                tool_use_id,
                runtime,
                timeout=arguments.get("timeout"),
            )
        else:
            return ToolResult(
                tool_use_id=tool_use_id,
                name=name,
                content=f"Unknown tool: {name}",
                is_error=True,
            )
    except Exception as exc:
        return ToolResult(
            tool_use_id=tool_use_id,
            name=name,
            content=f"Tool execution error: {exc}",
            is_error=True,
        )


# ── Individual tool executors ─────────────────────────────────

async def _exec_python(
    code: str,
    tool_use_id: str,
    runtime: ScrapingRuntime,
    timeout: float | None = None,
) -> ToolResult:
    """Execute Python code and return output."""
    if timeout is not None:
        # Agent provided an explicit timeout — clamp it.
        timeout = max(MIN_TIMEOUT, min(float(timeout), MAX_TIMEOUT))
    else:
        # No explicit timeout — predict from code structure.
        timeout = predict_timeout(code)

    # Block code that dumps raw HTML into the conversation.
    blocked = _check_raw_html_patterns(code)
    if blocked:
        return ToolResult(
            tool_use_id=tool_use_id,
            name="python",
            content=blocked,
            is_error=True,
        )

    # Capture URL before execution to detect navigation.
    url_before = runtime.page.url if runtime.page else None

    result = await runtime.execute(code, timeout=timeout)

    parts: list[str] = []
    parts.append(f"Executed in {result.duration_ms:.0f}ms (step {result.step}).")

    if result.output:
        parts.append(f"\nOutput:\n{result.output}")

    if result.error:
        parts.append(f"\nError:\n{result.error}")

    # If timeout diagnostics were collected, include a summary for the agent.
    if result.diagnostics:
        diag = result.diagnostics
        diag_parts = ["\n--- Timeout Diagnostics ---"]
        diag_parts.append(f"Page URL at timeout: {diag.page_url}")
        if diag.pending_requests:
            diag_parts.append(
                f"Pending network requests ({len(diag.pending_requests)}):"
            )
            for req in diag.pending_requests[:5]:
                diag_parts.append(
                    f"  {req.get('method', '?')} {req.get('url', '?')[:100]}"
                )
        if diag.console_logs:
            errors = [l for l in diag.console_logs if l.get("level") in ("error", "warning")]
            if errors:
                diag_parts.append(f"Console errors/warnings ({len(errors)}):")
                for log in errors[-5:]:
                    diag_parts.append(f"  [{log.get('level')}] {log.get('text', '')[:150]}")
        if runtime.diagnostics_dir:
            diag_parts.append(
                f"Full diagnostics saved to: timeout_step_{result.step}"
            )
        parts.append("\n".join(diag_parts))

    # Notify the agent if the URL changed (navigation occurred).
    url_after = runtime.page.url if runtime.page else None
    if url_before and url_after and url_after != url_before:
        parts.append(f"\nInfo: page URL changed to {url_after}")

    return ToolResult(
        tool_use_id=tool_use_id,
        name="python",
        content="\n".join(parts),
        is_error=not result.success,
    )
