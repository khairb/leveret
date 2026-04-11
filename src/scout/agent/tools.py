"""Tool definitions and execution dispatch for the scraping agent.

Single tool:
    - ``python``: Execute code in the stateful Patchright environment.

The agent also has two built-in REPL functions (``show_page`` and
``zoom_section``) which are injected separately — see ``bridge.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..runtime.environment import ScrapingRuntime


# ═══════════════════════════════════════════════════════════════
#  Tool Schemas  (Anthropic native format)
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
                        "Defaults to 30. Increase for slow operations "
                        "like page loads or large extractions (max 120)."
                    ),
                },
            },
            "required": ["code"],
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
    # Clamp timeout to [1, 120] if provided.
    if timeout is not None:
        timeout = max(1.0, min(float(timeout), 120.0))

    # Capture URL before execution to detect navigation.
    url_before = runtime.page.url if runtime.page else None

    result = await runtime.execute(code, timeout=timeout)

    parts: list[str] = []
    parts.append(f"Executed in {result.duration_ms:.0f}ms (step {result.step}).")

    if result.output:
        parts.append(f"\nOutput:\n{result.output}")

    if result.error:
        parts.append(f"\nError:\n{result.error}")

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
