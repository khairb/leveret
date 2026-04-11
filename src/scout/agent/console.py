"""Live console output for the scraping agent.

Prints a friendly, readable view of what the agent is doing in real-time
so you can follow along while waiting.
"""

from __future__ import annotations

import asyncio
import textwrap

# ── ANSI colors ───────────────────────────────────────────────

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_MAGENTA = "\033[35m"
_BLUE = "\033[34m"


def _c(color: str, text: str) -> str:
    return f"{color}{text}{_RESET}"


def _header(text: str) -> str:
    return f"\n{_BOLD}{_CYAN}{'=' * 60}{_RESET}\n{_BOLD}{_CYAN}  {text}{_RESET}\n{_BOLD}{_CYAN}{'=' * 60}{_RESET}"


def _subheader(text: str) -> str:
    return f"\n{_BOLD}{text}{_RESET}"


def _indent(text: str, prefix: str = "  ") -> str:
    return textwrap.indent(text, prefix)


def _truncate(text: str, max_lines: int = 15, max_chars: int = 2000) -> str:
    """Truncate text for display, showing start and end."""
    if len(text) <= max_chars:
        lines = text.split("\n")
        if len(lines) <= max_lines:
            return text

    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text[:max_chars] + f"\n{_DIM}... ({len(text) - max_chars} more chars){_RESET}"

    head = "\n".join(lines[:max_lines // 2])
    tail = "\n".join(lines[-(max_lines // 2):])
    omitted = len(lines) - max_lines
    return f"{head}\n{_DIM}  ... ({omitted} lines omitted) ...{_RESET}\n{tail}"


# ═══════════════════════════════════════════════════════════════
#  Public print functions — called from the loop
# ═══════════════════════════════════════════════════════════════

def print_init(url: str, task: str, model: str) -> None:
    print(_header("Scraping Agent"))
    print(f"  {_BOLD}URL:{_RESET}   {url}")
    print(f"  {_BOLD}Task:{_RESET}  {task}")
    print(f"  {_BOLD}Model:{_RESET} {model}")


def print_page_loaded(url: str, section_count: int) -> None:
    print(f"\n  {_c(_GREEN, 'Page loaded')} - {section_count} sections detected")
    print(f"  {_DIM}{url}{_RESET}")


def print_turn_start(turn: int) -> None:
    print(f"\n{_BOLD}{_BLUE}--- Turn {turn} ---{_RESET}")


def print_llm_thinking(
    duration_ms: float,
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    cache_create: int = 0,
) -> None:
    parts = [
        f"  {_DIM}LLM responded in {duration_ms / 1000:.1f}s "
        f"({input_tokens:,} in, {output_tokens:,} out)",
    ]
    if cache_read or cache_create:
        cache_parts = []
        if cache_read:
            cache_parts.append(f"{cache_read:,} cached")
        if cache_create:
            cache_parts.append(f"{cache_create:,} cache-write")
        parts.append(f" | {', '.join(cache_parts)}")
    parts.append(f"{_RESET}")
    print("".join(parts))


def print_agent_text(text: str) -> None:
    """Print text the agent says (not a tool call)."""
    if not text.strip():
        return
    print(f"\n  {_c(_MAGENTA, 'Agent:')}")
    print(_indent(_truncate(text.strip(), max_lines=10, max_chars=500), "    "))


def print_tool_call(name: str, arguments: dict, step: int, total_steps: int) -> None:
    """Print a tool call with its arguments in a friendly way."""
    badge = _c(_YELLOW, f"[{step}/{total_steps}]")

    if name == "python":
        code = arguments.get("code", "")
        print(f"\n  {badge} {_c(_BOLD, 'python')}  {_DIM}executing code{_RESET}")
        print(_indent(_truncate(code, max_lines=20, max_chars=1500), f"    {_DIM}|{_RESET} "))

    else:
        print(f"\n  {badge} {_c(_BOLD, name)}")


def print_tool_result(name: str, is_error: bool, duration_ms: float, content: str) -> None:
    """Print the result of a tool call."""
    status = _c(_RED, "ERROR") if is_error else _c(_GREEN, "OK")
    dur = f"{duration_ms / 1000:.1f}s"

    if name == "python":
        # Show a brief summary — output lines, whether page view is included.
        lines = content.split("\n")
        # Find the execution summary line.
        summary = lines[0] if lines else ""
        has_page_view = "=== Page State" in content
        has_error = "Error:" in content
        has_output = "Output:" in content

        parts = [f"{status} {_DIM}({dur}){_RESET}"]
        if has_output:
            # Extract just the output section.
            out_start = content.find("Output:\n")
            out_end = content.find("\n\n", out_start) if out_start != -1 else -1
            if out_start != -1:
                output_text = content[out_start + 8 : out_end if out_end != -1 else out_start + 500]
                output_preview = _truncate(output_text.strip(), max_lines=5, max_chars=300)
                parts.append(f"\n    {_DIM}Output:{_RESET}\n{_indent(output_preview, '      ')}")
        if has_error:
            err_start = content.find("Error:\n")
            err_end = content.find("\n\n", err_start) if err_start != -1 else -1
            if err_start != -1:
                error_text = content[err_start + 7 : err_end if err_end != -1 else err_start + 500]
                error_preview = _truncate(error_text.strip(), max_lines=5, max_chars=300)
                parts.append(f"\n    {_c(_RED, 'Error:')}\n{_indent(error_preview, '      ')}")
        if has_page_view:
            parts.append(f"  {_DIM}(page view updated){_RESET}")

        print(f"    -> {' '.join(parts[:1])}" + "".join(parts[1:]))

    else:
        print(f"    -> {status} {_DIM}({dur}){_RESET}")


def print_budget_warning(message: str) -> None:
    print(f"\n  {_c(_YELLOW, 'Budget:')} {message}")


def print_nudge() -> None:
    print(f"\n  {_DIM}(nudging agent to continue...){_RESET}")


def print_script_found(valid: bool, error: str = "") -> None:
    if valid:
        print(f"\n  {_c(_GREEN, 'Final script extracted!')} {_DIM}(valid syntax){_RESET}")
    else:
        print(f"\n  {_c(_RED, 'Script extracted but has syntax error:')} {error}")


def print_result(result) -> None:
    """Print the final result summary."""
    print(_header("Result"))
    if result.success:
        print(f"  {_c(_GREEN, 'SUCCESS')}")
        print(f"  Steps: {result.steps_executed} total, {result.python_steps} python")
        print(f"  Messages: {result.conversation_length}")
    else:
        print(f"  {_c(_RED, 'FAILED')}: {result.error}")
        print(f"  Steps: {result.steps_executed} total, {result.python_steps} python")

    if result.run_dir:
        print(f"  Run dir: {result.run_dir}")
    print()


def print_running_script() -> None:
    print(_header("Running Generated Script"))


def print_script_output(output: str, error: str, returncode: int) -> None:
    if output:
        print(f"\n{_c(_GREEN, 'Output:')}")
        print(_indent(output, "  "))
    if error:
        print(f"\n{_c(_RED, 'Stderr:')}")
        print(_indent(error, "  "))
    if returncode == 0:
        print(f"\n  {_c(_GREEN, 'Script executed successfully')}")
    else:
        print(f"\n  {_c(_RED, f'Script exited with code {returncode}')}")


async def ask_user_approval() -> tuple[bool, str]:
    """Ask the user to approve or reject the script output.

    Returns:
        ``(approved, feedback)`` — feedback is empty if approved.
    """
    print(f"\n{_BOLD}{_CYAN}{'─' * 60}{_RESET}")
    print(f"  {_BOLD}Review the script output above.{_RESET}")
    print(f"  {_c(_GREEN, '[a]pprove')}  —  output is correct, accept the script")
    print(f"  {_c(_RED, '[r]eject')}   —  output is wrong, provide feedback")
    print(f"{_BOLD}{_CYAN}{'─' * 60}{_RESET}")

    choice = await asyncio.to_thread(
        input, f"\n  {_BOLD}Approve or reject? [a/r]: {_RESET}"
    )
    choice = choice.strip().lower()

    if choice in ("a", "approve", "y", "yes"):
        print(f"\n  {_c(_GREEN, 'Script approved.')}")
        return True, ""

    feedback = await asyncio.to_thread(
        input, f"  {_BOLD}What's wrong? {_RESET}"
    )
    return False, feedback.strip()


def print_generating_requirements() -> None:
    print(f"\n  {_DIM}Generating success criteria from exploration history...{_RESET}")


def print_requirements_generated(requirements: str) -> None:
    print(f"\n  {_c(_GREEN, 'Success criteria generated:')}")
    print(_indent(_truncate(requirements, max_lines=20, max_chars=2000), "    "))


def print_validator_approved() -> None:
    print(f"\n  {_c(_GREEN, 'Validator approved the script.')}")


def print_validator_rejected(feedback: str) -> None:
    print(f"\n  {_c(_RED, 'Validator rejected:')} {feedback[:200]}")


def print_checkpoints_summary(checkpoints: list[dict]) -> None:
    """Print a brief checkpoint summary after script execution."""
    if not checkpoints:
        return
    count = len(checkpoints)
    print(f"\n  {_c(_DIM, f'{count} checkpoint(s) captured:')}")
    for cp in checkpoints:
        cp_id = cp.get("id", "?")
        label = cp.get("label", "?")
        print(f"    {_c(_DIM, f'{cp_id}: {label}')}")


def print_script_rejected(attempt: int, max_attempts: int) -> None:
    print(
        f"\n  {_c(_RED, 'Script rejected')} "
        f"(attempt {attempt}/{max_attempts}). "
        f"Feeding back to agent..."
    )
