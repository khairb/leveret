"""AgentLoop — main orchestration for the scraping agent.

Usage::

    from agent.loop import AgentLoop, AgentResult
    from agent.llm import LLMConfig

    agent = AgentLoop(llm_config=LLMConfig())
    result = await agent.run(
        url="https://example.com/products",
        task="Extract all product names and prices",
    )
    print(result.final_script)
"""

from __future__ import annotations

import ast
import asyncio
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..page.manager import PageStateManager
from ..runtime.environment import (
    ScrapingRuntime,
    SnapshotConfig,
)

from .checkpoint import (
    create_checkpoint_guard,
    create_expand_checkpoint_function,
    format_checkpoint_summary,
    read_checkpoints,
    write_scraping_utils,
)
from .context import ConversationManager
from .bridge import (
    create_post_exec_hook,
    create_show_page_function,
    create_zoom_section_function,
)
from .llm import LLMConfig, call_llm
from .requirements import generate_requirements
from .validator import AttemptRecord, validate_output
from .prompt import build_initial_user_message, build_system_prompt
from .tools import TOOL_SCHEMAS, ToolResult, execute_tool
from .trace import Tracer
from . import console

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

# Regex to extract fenced Python code blocks from agent text.
_PYTHON_FENCE_RE = re.compile(
    r"```python\s*\n(.*?)```", re.DOTALL
)


# ═══════════════════════════════════════════════════════════════
#  Result
# ═══════════════════════════════════════════════════════════════

@dataclass
class AgentResult:
    """The output of a completed agent run."""

    final_script: str = ""
    conversation_length: int = 0
    steps_executed: int = 0
    python_steps: int = 0
    success: bool = False
    error: str | None = None
    run_dir: str | None = None


# ═══════════════════════════════════════════════════════════════
#  Agent Loop
# ═══════════════════════════════════════════════════════════════

class AgentLoop:
    """Orchestrates the scraping agent from initialization to script output.

    Typical lifecycle::

        agent = AgentLoop(llm_config=LLMConfig())
        result = await agent.run(url, task)
    """

    def __init__(
        self,
        llm_config: LLMConfig | None = None,
        *,
        max_steps: int = 50,
        max_python_steps: int = 30,
        max_syntax_retries: int = 10,
        headless: bool = False,
        browser_type: str = "chromium",
        default_timeout: float = 30.0,
        trace_dir: str | Path = "./traces",
        script_timeout: int = 600,
        max_script_attempts: int = 10,
        approval_mode: str = "human",
        validator_config: LLMConfig | None = None,
    ) -> None:
        self._llm_config = llm_config or LLMConfig()
        self._max_steps = max_steps
        self._max_python_steps = max_python_steps
        self._max_syntax_retries = max_syntax_retries
        self._headless = headless
        self._browser_type = browser_type
        self._default_timeout = default_timeout
        self._trace_dir = Path(trace_dir)
        self._script_timeout = script_timeout
        self._max_script_attempts = max_script_attempts
        self._approval_mode = approval_mode
        self._validator_config = validator_config or LLMConfig(
            model="claude-haiku-4-5",
            max_tokens=8192,
        )

    # ── Main entry point ──────────────────────────────────────

    async def run(self, url: str, task: str) -> AgentResult:
        """Run the agent to completion.

        Args:
            url: The target URL to scrape.
            task: Natural language description of what to extract.

        Returns:
            :class:`AgentResult` with the final script or an error.
        """
        result = AgentResult()
        runtime: ScrapingRuntime | None = None
        tracer = Tracer(output_dir=self._trace_dir)

        try:
            # ── Phase 0: Initialization ───────────────────────
            console.print_init(url, task, self._llm_config.model)

            runtime, psm, initial_page_view = await self._initialize(url)

            # Count sections for display.
            section_count = (
                len(psm.current_state.sections) if psm.current_state else 0
            )
            console.print_page_loaded(url, section_count)

            system_prompt = build_system_prompt()
            initial_msg = build_initial_user_message(task, url)

            # Start trace.
            tracer.start(
                url=url,
                task=task,
                model=self._llm_config.model,
                system_prompt=system_prompt,
            )
            state = psm.current_state
            tracer.log_initial_page_view(
                page_view=initial_page_view,
                raw_html=state.raw_html if state else "",
                sanitized_html=state.html if state else "",
            )

            conversation = ConversationManager()
            conversation.add_user_message(initial_msg)

            step_count = 0
            python_step_count = 0
            script_attempts = 0
            attempt_history: list[AttemptRecord] = []
            requirements: str | None = None
            turn_number = 0
            checkpoint_base_dir = Path(
                tempfile.mkdtemp(prefix="scrape_checkpoints_")
            )

            # ── Main loop ─────────────────────────────────────
            while step_count < self._max_steps:
                turn_number += 1
                console.print_turn_start(turn_number)

                # Call the LLM.
                messages_for_api = conversation.get_messages()
                tracer.log_llm_request(messages_for_api, TOOL_SCHEMAS)

                llm_start = time.time()
                response = await call_llm(
                    self._llm_config,
                    system=system_prompt,
                    messages=messages_for_api,
                    tools=TOOL_SCHEMAS,
                )
                llm_duration_ms = (time.time() - llm_start) * 1000

                tracer.log_llm_response(response, llm_duration_ms)
                usage = response.usage
                console.print_llm_thinking(
                    llm_duration_ms,
                    usage.input_tokens,
                    usage.output_tokens,
                    cache_read=getattr(usage, "cache_read_input_tokens", 0) or 0,
                    cache_create=getattr(usage, "cache_creation_input_tokens", 0) or 0,
                )

                # Parse the response into content blocks.
                content_blocks = response.content
                stop_reason = response.stop_reason

                # Print any text the agent says.
                for b in content_blocks:
                    if b.type == "text" and b.text.strip():
                        console.print_agent_text(b.text)

                # Store the assistant message.
                conversation.add_assistant_message(
                    _serialize_content_blocks(content_blocks)
                )

                # Collect tool_use blocks.
                tool_uses = [
                    b for b in content_blocks if b.type == "tool_use"
                ]

                if not tool_uses:
                    # No tool calls — the agent responded with text.
                    # Check for a fenced Python code block (final script).
                    text = _extract_text(content_blocks)
                    script = _extract_final_script(text)
                    if script:
                        valid, error_msg = _validate_script(script)
                        tracer.log_script_extracted(script, valid, error_msg)
                        console.print_script_found(valid, error_msg)
                        if valid:
                            # Generate requirements once (before
                            # the first validation).
                            if (
                                requirements is None
                                and self._approval_mode == "auto"
                            ):
                                console.print_generating_requirements()
                                requirements = (
                                    await generate_requirements(
                                        task=task,
                                        conversation_history=(
                                            conversation.messages
                                        ),
                                        llm_config=self._validator_config,
                                    )
                                )
                                console.print_requirements_generated(
                                    requirements,
                                )
                                tracer.log_system_event(
                                    "requirements_generated",
                                    requirements=requirements,
                                )

                            # Run the script in a fresh subprocess.
                            console.print_running_script()
                            run_number = script_attempts + 1
                            run_dir = checkpoint_base_dir / f"run_{run_number}"
                            stdout, stderr, returncode = (
                                await _run_script_subprocess(
                                    script,
                                    timeout=self._script_timeout,
                                    checkpoint_dir=run_dir,
                                )
                            )
                            # Read checkpoints and update the
                            # expand_checkpoint ref.
                            checkpoints = read_checkpoints(run_dir)
                            self._checkpoint_run_dir_ref[0] = run_dir
                            console.print_script_output(
                                stdout, stderr, returncode,
                            )
                            console.print_checkpoints_summary(checkpoints)
                            tracer.log_system_event(
                                "script_executed",
                                returncode=returncode,
                                stdout_len=len(stdout),
                                stderr_len=len(stderr),
                                checkpoints=len(checkpoints),
                            )

                            # Approve or reject the script output.
                            cp_summary = format_checkpoint_summary(
                                checkpoints,
                            )

                            # Short-circuit: skip validator when
                            # the script crashed (non-zero exit).
                            if returncode != 0:
                                approved = False
                                feedback = (
                                    "Script crashed with exit code "
                                    f"{returncode}. Fix the error and "
                                    "try again."
                                )
                            elif self._approval_mode == "auto":
                                approved, feedback = await validate_output(
                                    task=task,
                                    script=script,
                                    stdout=stdout,
                                    stderr=stderr,
                                    returncode=returncode,
                                    llm_config=self._validator_config,
                                    attempt_history=(
                                        attempt_history
                                        if attempt_history
                                        else None
                                    ),
                                    requirements=requirements,
                                    trace_dir=tracer.run_dir,
                                    checkpoints_summary=cp_summary,
                                )
                                if approved:
                                    console.print_validator_approved()
                                else:
                                    console.print_validator_rejected(feedback)
                            else:
                                approved, feedback = (
                                    await console.ask_user_approval()
                                )

                            if approved:
                                result.final_script = script
                                result.success = True
                                tracer.log_system_event(
                                    "script_approved",
                                    attempt=script_attempts + 1,
                                )
                                # Save script output to the run dir.
                                if tracer.run_dir:
                                    out_path = tracer.run_dir / "output.txt"
                                    out_path.write_text(
                                        stdout, encoding="utf-8",
                                    )
                                break

                            # Rejected — record attempt and feed back.
                            script_attempts += 1
                            attempt_history.append(AttemptRecord(
                                attempt_number=script_attempts,
                                script=script,
                                stdout_sample=stdout[:3000],
                                stderr=stderr,
                                returncode=returncode,
                                rejection_feedback=feedback,
                            ))
                            tracer.log_system_event(
                                "script_rejected",
                                feedback=feedback,
                                attempt=script_attempts,
                            )
                            console.print_script_rejected(
                                script_attempts,
                                self._max_script_attempts,
                            )

                            if script_attempts >= self._max_script_attempts:
                                result.final_script = script
                                result.error = (
                                    f"Script rejected "
                                    f"{script_attempts} times. "
                                    f"Last feedback: {feedback}"
                                )
                                break

                            rejection_msg = _build_rejection_message(
                                feedback, stdout, stderr,
                                returncode, script_attempts,
                                self._max_script_attempts,
                                checkpoints=checkpoints,
                            )
                            conversation.add_user_message(rejection_msg)
                            continue
                        else:
                            # Ask the agent to fix the syntax error.
                            fix_msg = (
                                f"The script has a syntax error: {error_msg}\n"
                                f"Please fix it and respond again with the "
                                f"corrected script in a Python code block."
                            )
                            tracer.log_system_event(
                                "script_syntax_error", error=error_msg,
                            )
                            conversation.add_user_message(fix_msg)
                            continue

                    # No code block found — nudge the agent.
                    if stop_reason == "end_turn":
                        nudge = (
                            "Continue working. Use your tools to explore "
                            "the page. When done, respond with the final "
                            "script in a ```python code block."
                        )
                        tracer.log_system_event("nudge_sent", text=nudge)
                        console.print_nudge()
                        conversation.add_user_message(nudge)
                    continue

                # Execute each tool call.
                tool_results: list[dict] = []
                for block in tool_uses:
                    step_count += 1
                    if block.name == "python":
                        python_step_count += 1

                    console.print_tool_call(
                        block.name, block.input,
                        step_count, self._max_steps,
                    )
                    tracer.log_tool_call(block.name, block.input, block.id)

                    tool_start = time.time()
                    tool_result = await execute_tool(
                        name=block.name,
                        arguments=block.input,
                        tool_use_id=block.id,
                        runtime=runtime,
                    )
                    tool_duration_ms = (time.time() - tool_start) * 1000

                    tracer.log_tool_result(
                        name=block.name,
                        tool_use_id=block.id,
                        content=tool_result.content,
                        is_error=tool_result.is_error,
                        duration_ms=tool_duration_ms,
                    )
                    console.print_tool_result(
                        block.name,
                        tool_result.is_error,
                        tool_duration_ms,
                        tool_result.content,
                    )

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_result.tool_use_id,
                        "content": tool_result.content,
                        **({"is_error": True} if tool_result.is_error else {}),
                    })

                conversation.add_tool_results(tool_results)

                # Budget checks.
                if python_step_count >= self._max_python_steps:
                    budget_msg = (
                        f"You have used {python_step_count} code executions "
                        f"(limit: {self._max_python_steps}). Stop using tools "
                        f"and respond with the final script in a ```python "
                        f"code block."
                    )
                    tracer.log_system_event("budget_warning", text=budget_msg)
                    console.print_budget_warning(budget_msg)
                    conversation.add_user_message(budget_msg)
                elif step_count >= self._max_steps - 2:
                    budget_msg = (
                        f"You have {self._max_steps - step_count} tool calls "
                        f"remaining. Stop using tools and respond with the "
                        f"final script in a ```python code block."
                    )
                    tracer.log_system_event("budget_warning", text=budget_msg)
                    console.print_budget_warning(budget_msg)
                    conversation.add_user_message(budget_msg)

            # If we exited the loop without a script:
            if not result.success:
                result.error = (
                    f"Agent exhausted budget ({step_count} steps, "
                    f"{python_step_count} python) without producing a script."
                )

            result.conversation_length = len(conversation.messages)
            result.steps_executed = step_count
            result.python_steps = python_step_count

        except Exception as exc:
            logger.exception("Agent run failed")
            result.error = str(exc)
            tracer.log_system_event("exception", error=str(exc))

        finally:
            if runtime is not None:
                try:
                    await runtime.stop()
                except Exception:
                    pass

            # Clean up checkpoint temp directory.
            try:
                shutil.rmtree(checkpoint_base_dir, ignore_errors=True)
            except NameError:
                pass  # checkpoint_base_dir was never created

            # Set run_dir early — the directory exists from start().
            if tracer.run_dir:
                result.run_dir = str(tracer.run_dir)

            # Rewrite trace with final stats (incremental file already on disk).
            try:
                tracer.finish(result)
            except Exception:
                logger.exception("Failed to finalize trace")

        console.print_result(result)
        return result

    # ── Initialization ────────────────────────────────────────

    async def _initialize(
        self, url: str
    ) -> tuple[ScrapingRuntime, PageStateManager, str]:
        """Launch browser, navigate, capture initial page view.

        Returns:
            ``(runtime, page_state_manager, initial_page_view)``
        """
        # Mutable reference for the hook closure.
        psm_ref: list[Any] = [None]
        hook = create_post_exec_hook(psm_ref)

        runtime = ScrapingRuntime(
            headless=self._headless,
            browser_type=self._browser_type,
            default_timeout=self._default_timeout,
            snapshot_config=SnapshotConfig(
                include_html=False,
                include_text=False,
                include_screenshot=False,
            ),
            post_exec_hook=hook,
        )

        await runtime.start()

        # Now that the page exists, create the PageStateManager.
        psm = PageStateManager(runtime.page)
        psm_ref[0] = psm

        # Inject show_page(page), zoom_section(page, ...), and
        # expand_checkpoint(...) into the REPL.
        show_page_fn = create_show_page_function(psm_ref)
        zoom_section_fn = create_zoom_section_function(psm_ref)
        self._checkpoint_run_dir_ref: list[Any] = [None]
        expand_cp_fn = create_expand_checkpoint_function(
            self._checkpoint_run_dir_ref,
        )
        checkpoint_guard = create_checkpoint_guard()
        runtime.repl.inject(
            show_page=show_page_fn,
            zoom_section=zoom_section_fn,
            expand_checkpoint=expand_cp_fn,
            checkpoint=checkpoint_guard,
        )

        # Navigate to the target URL — triggers the hook.
        goto_code = (
            f'await page.goto("{url}", '
            f'wait_until="domcontentloaded", timeout=30000)'
        )
        initial_result = await runtime.execute(goto_code)

        if not initial_result.success:
            raise RuntimeError(
                f"Failed to navigate to {url}: {initial_result.error}"
            )

        # Wait for dynamic content to load before capturing page state.
        await asyncio.sleep(10)

        # Capture the fully-loaded page (ignore the hook's early capture).
        await psm.capture()
        page_view = psm.get_page_view()

        return runtime, psm, page_view


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

def _serialize_content_blocks(blocks: list) -> list[dict]:
    """Convert Anthropic content blocks to plain dicts for storage."""
    serialized = []
    for b in blocks:
        if b.type == "text":
            serialized.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            serialized.append({
                "type": "tool_use",
                "id": b.id,
                "name": b.name,
                "input": b.input,
            })
        else:
            # Unknown block type — store as text.
            serialized.append({"type": "text", "text": str(b)})
    return serialized


def _extract_text(blocks: list) -> str:
    """Extract all text from content blocks."""
    parts = []
    for b in blocks:
        if b.type == "text":
            parts.append(b.text)
    return "\n".join(parts)


def _extract_final_script(text: str) -> str | None:
    """Extract a Python script from fenced code blocks.

    Looks for ```python ... ``` blocks.  If multiple are found, returns
    the last one (the agent may show small snippets earlier and the full
    script last).
    """
    matches = _PYTHON_FENCE_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def _validate_script(script: str) -> tuple[bool, str]:
    """Check that the script is valid Python syntax."""
    try:
        ast.parse(script)
        return True, ""
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


async def _run_script_subprocess(
    script: str,
    timeout: int = 120,
    checkpoint_dir: Path | None = None,
) -> tuple[str, str, int]:
    """Run a Python script in a fresh subprocess.

    If *checkpoint_dir* is given, ``scraping_utils.py`` is written next
    to the script and ``SCRAPE_CHECKPOINT_DIR`` is set so checkpoints
    land in that directory.

    Returns:
        ``(stdout, stderr, returncode)``
    """
    script_dir = Path(tempfile.mkdtemp(prefix="scrape_run_"))
    script_path = script_dir / "script.py"
    try:
        script_path.write_text(script, encoding="utf-8")

        # Write scraping_utils.py alongside the script so
        # ``from scraping_utils import checkpoint`` works.
        write_scraping_utils(script_dir)

        env = dict(os.environ)
        if checkpoint_dir is not None:
            env["SCRAPE_CHECKPOINT_DIR"] = str(checkpoint_dir)

        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return "", f"Script timed out after {timeout} seconds", -1

        return (
            stdout_bytes.decode(errors="replace"),
            stderr_bytes.decode(errors="replace"),
            proc.returncode or 0,
        )
    finally:
        shutil.rmtree(script_dir, ignore_errors=True)


def _build_rejection_message(
    feedback: str,
    stdout: str,
    stderr: str,
    returncode: int,
    attempt: int,
    max_attempts: int,
    checkpoints: list[dict] | None = None,
) -> str:
    """Build the message sent to the agent after user rejects a script."""
    parts = [
        "## Script Rejected\n",
        f"**User feedback:** {feedback}\n",
    ]

    if stdout.strip():
        output = stdout.strip()
        if len(output) > 3000:
            output = (
                output[:1500]
                + "\n\n... (truncated) ...\n\n"
                + output[-1500:]
            )
        parts.append(f"**Script output:**\n```\n{output}\n```\n")

    if stderr.strip():
        parts.append(
            f"**Script errors:**\n```\n{stderr.strip()[:2000]}\n```\n"
        )

    parts.append(f"**Exit code:** {returncode}\n")
    parts.append(f"**Attempt:** {attempt}/{max_attempts}\n")

    # Include checkpoint summary if available.
    if checkpoints:
        cp_summary = format_checkpoint_summary(checkpoints)
        parts.append(f"\n{cp_summary}\n")

    parts.append(
        "\n---\n\n"
        "Do not guess at a fix. Be systematic:\n\n"
        "1. **Analyze** the feedback, script output, and checkpoints "
        "carefully. Understand exactly what went wrong — which part of "
        "the script failed and why.\n"
        "2. **Expand checkpoints** to see what the page looked like at "
        "key moments: call `expand_checkpoint(\"CP-1\")` to inspect "
        "the full page state at any checkpoint.\n"
        "3. **Investigate** using your Python environment. The browser is "
        "still open from your exploration. Test selectors, check page "
        "state, verify your assumptions. Use `await show_page(page)` and "
        "`await zoom_section(page, ...)` to re-examine the page if "
        "needed.\n"
        "4. **Verify your fix** before writing the final script. Test the "
        "corrected logic in your Python environment first.\n"
        "5. **Write the corrected script** in a ```python code block.\n\n"
        "The user will test this script again. Make sure your fix "
        "addresses their feedback."
    )

    return "\n".join(parts)
