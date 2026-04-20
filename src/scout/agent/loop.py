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
import json
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
)
from .wrapper import (
    build_combined_output,
    generate_subprocess_wrapper,
    parse_return_value,
)
from .context import ConversationManager, _PAGE_VIEW_START, _ZOOM_START
from .bridge import (
    ShowPageResult,
    create_post_exec_hook,
    create_show_page_function,
    create_zoom_section_function,
)
from .llm import LLMConfig, call_llm
from .requirements import generate_requirements, revise_requirements
from .validator import AttemptRecord, validate_output
from .prompt import (
    build_initial_user_message,
    build_show_page_analysis_prompt_a,
    build_show_page_analysis_prompt_b,
    build_show_page_debugging_prompt_a,
    build_show_page_debugging_prompt_b,
    build_system_prompt,
    build_zoom_structural_capture_prompt,
)
from .show_page_context import (
    NEIGHBOR_RADIUS,
    ElementMatch,
    ShowPageAnalysisLog,
    ShowPageState,
    build_filtered_output,
    get_referenced_sections,
    get_sections_by_id,
    page_similarity,
)
from .tools import TOOL_SCHEMAS, ToolResult, execute_tool
from .trace import Tracer
from . import console

# Import for type checking only — CompiledSchema is optional at init.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..schema.compiler import CompiledSchema

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
    return_value: str | None = None
    conversation_length: int = 0
    steps_executed: int = 0
    python_steps: int = 0
    success: bool = False
    last_resort: bool = False
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
        max_steps: int = 75,
        max_python_steps: int = 50,
        max_syntax_retries: int = 10,
        headless: bool = False,
        browser_type: str = "chromium",
        default_timeout: float = 30.0,
        trace_dir: str | Path = "./traces",
        script_timeout: int = 600,
        max_script_attempts: int = 10,
        approval_mode: str = "human",
        validator_config: LLMConfig | None = None,
        compiled_schema: CompiledSchema | None = None,
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
            model="anthropic:claude-haiku-4-5",
            max_tokens=8192,
        )
        self._compiled_schema = compiled_schema

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

            schema_prompt = (
                self._compiled_schema.prompt
                if self._compiled_schema is not None
                else ""
            )
            system_prompt = build_system_prompt(
                schema_prompt=schema_prompt,
            )
            initial_msg = build_initial_user_message(task, url)

            # Start trace.
            tracer.start(
                url=url,
                task=task,
                model=self._llm_config.model,
                system_prompt=system_prompt,
            )

            # Wire diagnostics directory so timeout dumps are saved
            # alongside the trace.
            if tracer.run_dir:
                runtime.diagnostics_dir = tracer.run_dir / "timeout_diagnostics"
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
            needs_debugging = False
            debugging_turn_count = 0  # turns since last rejection
            active_script_result: InProcessScriptResult | None = None
            exploration_page = runtime.page  # Save original page ref
            checkpoint_base_dir = Path(
                tempfile.mkdtemp(prefix="scrape_checkpoints_")
            )

            # Show-page context management state.
            show_page_state = ShowPageState()
            pending_show_page: ShowPageResult | None = None
            pending_is_variant_a = False
            analysis_prompt_msg_index: int | None = None

            # Zoom-section structural capture state.
            zoom_prompt_msg_index: int | None = None

            # ── Main loop ─────────────────────────────────────
            while step_count < self._max_steps:
                turn_number += 1
                console.print_turn_start(turn_number)

                # Call the LLM.
                current_tools = TOOL_SCHEMAS
                messages_for_api = conversation.get_messages()
                tracer.log_llm_request(messages_for_api, current_tools)

                llm_start = time.time()
                response = await call_llm(
                    self._llm_config,
                    system=system_prompt,
                    messages=messages_for_api,
                    tools=current_tools,
                )
                llm_duration_ms = (time.time() - llm_start) * 1000

                tracer.log_llm_response(response, llm_duration_ms)
                usage = response.usage
                console.print_llm_thinking(
                    llm_duration_ms,
                    usage.input_tokens,
                    usage.output_tokens,
                    cache_read=usage.cache_read_input_tokens,
                    cache_create=usage.cache_creation_input_tokens,
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

                # ── Phase 2+3: show_page analysis & filtering ────
                was_analysis_turn = pending_show_page is not None
                if pending_show_page is not None:
                    reasoning = _extract_text(content_blocks)
                    if reasoning.strip():
                        # Phase 3 — filter the page view.
                        sections_for_ref = [
                            (s.section_id, s.content,
                             s.interactive_elements)
                            for s in pending_show_page.sections
                        ]
                        referenced, el_matches = (
                            get_referenced_sections(
                                reasoning, sections_for_ref,
                            )
                        )
                        sections_for_filter = [
                            (s.section_id, s.content,
                             s.semantic_role, s.interactive_count)
                            for s in pending_show_page.sections
                        ]
                        # Extract the page header line so it
                        # survives filtering (preserves URL).
                        pv_text = pending_show_page.text_output
                        first_line = pv_text.split("\n", 1)[0]
                        page_header = (
                            first_line
                            if first_line.startswith("===")
                            else None
                        )
                        filtered = build_filtered_output(
                            sections_for_filter, referenced,
                            page_header=page_header,
                        )
                        conversation.replace_last_show_page_result(
                            filtered,
                        )

                        # ── Observability logging ────────────
                        _emit_show_page_log(
                            tracer=tracer,
                            turn_number=turn_number,
                            url=runtime.page.url if runtime.page else "",
                            pending_show_page=pending_show_page,
                            show_page_state=show_page_state,
                            pending_is_variant_a=pending_is_variant_a,
                            reasoning=reasoning,
                            referenced=referenced,
                            el_matches=el_matches,
                            sections_for_filter=sections_for_filter,
                            filtered=filtered,
                        )
                    # Remove ephemeral analysis prompt.
                    if analysis_prompt_msg_index is not None:
                        conversation.remove_message(
                            analysis_prompt_msg_index,
                        )
                        analysis_prompt_msg_index = None
                    if pending_is_variant_a:
                        show_page_state.mark_analyzed(
                            pending_show_page.raw_text,
                        )
                    pending_show_page = None

                # Remove ephemeral zoom structural capture prompt.
                if zoom_prompt_msg_index is not None:
                    conversation.remove_message(zoom_prompt_msg_index)
                    zoom_prompt_msg_index = None

                # Collect tool_use blocks.
                tool_uses = [
                    b for b in content_blocks if b.type == "tool_use"
                ]

                if not tool_uses:
                    # No tool calls — the agent responded with text.
                    # Check for a fenced Python code block (final function).
                    text = _extract_text(content_blocks)
                    script = _extract_final_script(text)
                    if script and needs_debugging and debugging_turn_count == 0:
                        # The agent tried to resubmit without
                        # investigating.  Block it and insist on
                        # root-cause analysis first.
                        debug_msg = (
                            "Script not accepted. You have not "
                            "investigated the failure yet.\n\n"
                            "Use the `python` tool to debug before "
                            "resubmitting. `page` now points to "
                            "the page from your last script "
                            "execution — it is in the exact state "
                            "where the script ended or crashed. "
                            "Call `await show_page(page)` to see "
                            "what the page looks like, "
                            "`await zoom_section(page, ...)` to "
                            "inspect DOM sections, and test your "
                            "selectors against `page` to see what "
                            "they actually return."
                            "\n\nIdentify the exact root cause, "
                            "then write the corrected function."
                        )
                        tracer.log_system_event(
                            "debug_required", text=debug_msg,
                        )
                        console.print_debug_required()
                        conversation.add_user_message(debug_msg)
                        continue
                    if script:
                        valid, error_msg = _validate_script(script)
                        tracer.log_script_extracted(script, valid, error_msg)
                        console.print_script_found(valid, error_msg)
                        if valid:
                            # Agent submitted a script — clear
                            # debugging state.
                            needs_debugging = False
                            debugging_turn_count = 0

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

                            # Clean up previous script browser and
                            # restore exploration page as safety net.
                            if active_script_result is not None:
                                await active_script_result.cleanup()
                                active_script_result = None
                                runtime.repl.inject(
                                    page=exploration_page,
                                )
                                psm.page = exploration_page

                            # Run the function in a fresh in-process
                            # browser (page stays alive for debugging).
                            console.print_running_script()
                            run_number = script_attempts + 1
                            run_dir = checkpoint_base_dir / f"run_{run_number}"
                            script_run = (
                                await _run_script_in_process(
                                    script,
                                    start_url=url,
                                    timeout=self._script_timeout,
                                    checkpoint_dir=run_dir,
                                )
                            )
                            # Track for cleanup (finally block).
                            active_script_result = script_run
                            stdout = script_run.stdout
                            return_value_json = script_run.return_value_json
                            stderr = script_run.stderr
                            returncode = script_run.returncode
                            # Build combined output for display and
                            # validation (stdout + return value JSON).
                            combined_output = build_combined_output(
                                stdout, return_value_json,
                            )

                            # Read checkpoints and update the
                            # expand_checkpoint ref.
                            checkpoints = read_checkpoints(run_dir)
                            self._checkpoint_run_dir_ref[0] = run_dir
                            console.print_script_output(
                                combined_output, stderr, returncode,
                            )
                            console.print_checkpoints_summary(checkpoints)
                            tracer.log_system_event(
                                "script_executed",
                                returncode=returncode,
                                stdout_len=len(stdout),
                                stderr_len=len(stderr),
                                checkpoints=len(checkpoints),
                                has_return_value=return_value_json is not None,
                            )

                            # Approve or reject the function output.
                            cp_summary = format_checkpoint_summary(
                                checkpoints,
                            )

                            # Short-circuit: skip validator when
                            # the function crashed (non-zero exit).
                            approved = False
                            feedback = ""

                            if returncode != 0:
                                feedback = (
                                    "Function crashed with exit code "
                                    f"{returncode}. Fix the error and "
                                    "try again."
                                )
                            else:
                                # Schema validation gate — fast,
                                # deterministic, free (no LLM call).
                                # Runs before the LLM validator.
                                schema_passed = True
                                if self._compiled_schema is not None:
                                    if return_value_json is not None:
                                        try:
                                            return_data = json.loads(
                                                return_value_json,
                                            )
                                        except (ValueError, TypeError):
                                            return_data = None
                                    else:
                                        # No return value markers —
                                        # validate None.
                                        return_data = None

                                    valid, schema_feedback = (
                                        self._compiled_schema.validate(
                                            return_data,
                                        )
                                    )
                                    tracer.log_system_event(
                                        "schema_validation",
                                        valid=valid,
                                    )
                                    if not valid:
                                        schema_passed = False
                                        feedback = schema_feedback

                                if not schema_passed:
                                    # Schema errors are definitive —
                                    # skip the LLM validator.
                                    pass
                                elif self._approval_mode == "auto":
                                    approved, feedback = (
                                        await validate_output(
                                            task=task,
                                            script=script,
                                            stdout=combined_output,
                                            stderr=stderr,
                                            returncode=returncode,
                                            llm_config=(
                                                self._validator_config
                                            ),
                                            attempt_history=(
                                                attempt_history
                                                if attempt_history
                                                else None
                                            ),
                                            requirements=requirements,
                                            trace_dir=tracer.run_dir,
                                            checkpoints_summary=(
                                                cp_summary
                                            ),
                                            attempt_number=(
                                                script_attempts + 1
                                            ),
                                            max_attempts=(
                                                self._max_script_attempts
                                            ),
                                        )
                                    )
                                    if approved:
                                        console.print_validator_approved()
                                    else:
                                        console.print_validator_rejected(
                                            feedback,
                                        )
                                else:
                                    approved, feedback = (
                                        await console.ask_user_approval()
                                    )

                            if approved:
                                result.final_script = script
                                result.return_value = return_value_json
                                result.success = True
                                tracer.log_system_event(
                                    "script_approved",
                                    attempt=script_attempts + 1,
                                )
                                # Save combined output to the run dir.
                                if tracer.run_dir:
                                    out_path = tracer.run_dir / "output.txt"
                                    out_path.write_text(
                                        combined_output, encoding="utf-8",
                                    )
                                # Script approved — close script browser.
                                await active_script_result.cleanup()
                                active_script_result = None
                                break

                            # Rejected — swap the REPL's `page` to the
                            # script's page so the agent debugs on the
                            # actual page the script ran on.
                            if script_run.page is not None:
                                runtime.repl.inject(
                                    page=script_run.page,
                                )
                                psm.page = script_run.page
                                console.print_script_page_injected()
                            # Record attempt and feed back.
                            script_attempts += 1
                            attempt_history.append(AttemptRecord(
                                attempt_number=script_attempts,
                                script=script,
                                stdout_sample=combined_output[:3000],
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

                            # Revise requirements after 2 validator
                            # rejections — execution evidence may
                            # show that page-reported numbers do not
                            # match what is actually extractable.
                            if (
                                script_attempts == 2
                                and requirements is not None
                                and self._approval_mode == "auto"
                            ):
                                console.print_revising_requirements()
                                evidence = [
                                    {
                                        "attempt_number": rec.attempt_number,
                                        "rejection_feedback": (
                                            rec.rejection_feedback
                                        ),
                                        "stdout_sample": rec.stdout_sample,
                                        "returncode": rec.returncode,
                                    }
                                    for rec in attempt_history
                                ]
                                requirements = (
                                    await revise_requirements(
                                        task=task,
                                        original_requirements=requirements,
                                        attempt_history=evidence,
                                        llm_config=(
                                            self._validator_config
                                        ),
                                    )
                                )
                                console.print_requirements_revised(
                                    requirements,
                                )
                                tracer.log_system_event(
                                    "requirements_revised",
                                    requirements=requirements,
                                    after_attempt=script_attempts,
                                )

                            if script_attempts >= self._max_script_attempts:
                                result.final_script = script
                                result.error = (
                                    f"Function rejected "
                                    f"{script_attempts} times. "
                                    f"Last feedback: {feedback}"
                                )
                                break

                            rejection_msg = _build_rejection_message(
                                feedback, stdout, stderr,
                                returncode, script_attempts,
                                self._max_script_attempts,
                                checkpoints=checkpoints,
                                return_value_json=return_value_json,
                                script_source=script,
                            )
                            conversation.add_user_message(rejection_msg)
                            needs_debugging = True
                            debugging_turn_count = 0
                            continue
                        else:
                            # Ask the agent to fix the validation error.
                            fix_msg = (
                                f"Your code has an issue: {error_msg}\n"
                                f"Please fix it and respond again with the "
                                f"corrected function in a Python code block."
                            )
                            tracer.log_system_event(
                                "script_syntax_error", error=error_msg,
                            )
                            conversation.add_user_message(fix_msg)
                            continue

                    # No code block found — nudge the agent.
                    if stop_reason == "end_turn":
                        if was_analysis_turn:
                            # Analysis turn — the text-only response
                            # is expected.  Send a brief follow-up so
                            # the conversation ends on a user message
                            # (required by the API).
                            if needs_debugging:
                                conversation.add_user_message(
                                    "Analysis received. Continue "
                                    "investigating the root cause "
                                    "of the failure."
                                )
                            else:
                                conversation.add_user_message(
                                    "Analysis received. Continue with "
                                    "your task."
                                )
                        else:
                            if needs_debugging:
                                nudge = (
                                    "Use the `python` tool to "
                                    "investigate the failure — "
                                    "try `show_page` or `zoom_section` "
                                    "if needed. Once you understand "
                                    "the root cause, submit "
                                    "the corrected function in a "
                                    "```python code block."
                                )
                            else:
                                nudge = (
                                    "Continue working. Use your tools "
                                    "to explore the page. When done, "
                                    "respond with the final scraping "
                                    "function in a ```python code "
                                    "block."
                                )
                            tracer.log_system_event(
                                "nudge_sent", text=nudge,
                            )
                            console.print_nudge()
                            conversation.add_user_message(nudge)
                    # Save snapshot for text-only turns too.
                    tracer.save_history_snapshot(
                        turn_number=turn_number,
                        messages=conversation.messages,
                        usage={
                            "input_tokens": usage.input_tokens,
                            "output_tokens": usage.output_tokens,
                            "cache_read": usage.cache_read_input_tokens,
                            "cache_create": usage.cache_creation_input_tokens,
                        },
                        step_count=step_count,
                        python_step_count=python_step_count,
                    )
                    continue

                # Execute each tool call.
                tool_results: list[dict] = []
                for block in tool_uses:
                    step_count += 1
                    if block.name == "python":
                        python_step_count += 1
                        if needs_debugging:
                            debugging_turn_count += 1

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

                    # Log and print timeout diagnostics if present.
                    if (
                        runtime.history.last
                        and runtime.history.last.diagnostics
                    ):
                        diag = runtime.history.last.diagnostics
                        tracer.log_system_event(
                            "timeout_diagnostics",
                            step=runtime.history.last.step,
                            page_url=diag.page_url,
                            pending_requests=len(diag.pending_requests),
                            failed_requests=len(diag.failed_requests),
                            console_errors=len(
                                [l for l in diag.console_logs
                                 if l.get("level") in ("error", "warning")]
                            ),
                            partial_stdout_len=len(diag.partial_stdout),
                            diagnostics_summary=diag.summary()[:2000],
                        )
                        saved_path = ""
                        if runtime.diagnostics_dir:
                            saved_path = str(
                                runtime.diagnostics_dir
                                / f"timeout_step_{runtime.history.last.step}"
                            )
                        console.print_timeout_diagnostics(
                            step=runtime.history.last.step,
                            diag_summary=diag.summary(),
                            saved_path=saved_path,
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

                # Neutral turn status — inform the agent of its
                # position without pressuring it to finish early.
                steps_remaining = self._max_steps - step_count
                python_remaining = (
                    self._max_python_steps - python_step_count
                )
                status_msg = (
                    f"[Turn {step_count}/{self._max_steps} — "
                    f"{steps_remaining} remaining"
                    f" | Code executions: {python_step_count}/"
                    f"{self._max_python_steps},"
                    f" {python_remaining} remaining]"
                )
                tracer.log_system_event(
                    "turn_status", text=status_msg,
                )
                console.print_turn_status(status_msg)
                if step_count % 10 == 0:
                    conversation.add_user_message(status_msg)

                # Periodic debugging reminders — escalating but
                # never aggressive.  Only appended when the agent
                # is using the python tool while a rejection is
                # still pending.
                if needs_debugging:
                    reminder = _debugging_reminder(
                        debugging_turn_count,
                    )
                    if reminder:
                        conversation.add_user_message(reminder)
                        tracer.log_system_event(
                            "debugging_reminder",
                            turn=debugging_turn_count,
                            text=reminder,
                        )

                # Save history snapshot for post-run analysis.
                tracer.save_history_snapshot(
                    turn_number=turn_number,
                    messages=conversation.messages,
                    usage={
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "cache_read": usage.cache_read_input_tokens,
                        "cache_create": usage.cache_creation_input_tokens,
                    },
                    step_count=step_count,
                    python_step_count=python_step_count,
                )

                # ── Phase 1: detect show_page and inject prompt ──
                is_show_page_turn = any(
                    isinstance(tr.get("content"), str)
                    and _PAGE_VIEW_START in tr["content"]
                    for tr in tool_results
                )
                sp_result = self._show_page_result_ref[0]
                if is_show_page_turn and sp_result is not None:
                    self._show_page_result_ref[0] = None
                    pending_is_variant_a = (
                        show_page_state.should_force_full_analysis(
                            sp_result.raw_text,
                        )
                    )
                    if needs_debugging and pending_is_variant_a:
                        prompt = build_show_page_debugging_prompt_a()
                    elif needs_debugging:
                        prompt = build_show_page_debugging_prompt_b()
                    elif pending_is_variant_a:
                        prompt = build_show_page_analysis_prompt_a()
                    else:
                        prompt = build_show_page_analysis_prompt_b()
                    conversation.add_user_message(prompt)
                    analysis_prompt_msg_index = (
                        len(conversation.messages) - 1
                    )
                    pending_show_page = sp_result
                    tracer.log_system_event(
                        "show_page_phase1",
                        variant="A" if pending_is_variant_a else "B",
                    )

                # ── Zoom structural capture prompt ────────────
                # Inject a lightweight prompt when zoom_section was
                # called, unless show_page already triggered an
                # analysis prompt (which covers structural capture).
                if not is_show_page_turn:
                    is_zoom_turn = any(
                        isinstance(tr.get("content"), str)
                        and _ZOOM_START in tr["content"]
                        for tr in tool_results
                    )
                    if is_zoom_turn:
                        zoom_prompt = (
                            build_zoom_structural_capture_prompt()
                        )
                        conversation.add_user_message(zoom_prompt)
                        zoom_prompt_msg_index = (
                            len(conversation.messages) - 1
                        )
                        tracer.log_system_event(
                            "zoom_structural_capture",
                        )

            # If we exited the loop without a script, make one
            # final LLM call with no tools available — the agent
            # has no choice but to generate the script.
            if not result.success:
                tracer.log_system_event("final_call_no_tools")
                console.print_final_call()

                final_msg = (
                    "All your exploration turns are exhausted. "
                    "You cannot use any more tools. Write the "
                    "best possible scraping function now based "
                    "on everything you have learned. Respond "
                    "with the final `async def scrape(page, start_url,"
                    " checkpoint)` function in a ```python code "
                    "block."
                )
                conversation.add_user_message(final_msg)

                messages_for_api = conversation.get_messages()
                tracer.log_llm_request(messages_for_api, [])

                llm_start = time.time()
                response = await call_llm(
                    self._llm_config,
                    system=system_prompt,
                    messages=messages_for_api,
                    tools=[],
                )
                llm_duration_ms = (time.time() - llm_start) * 1000

                tracer.log_llm_response(response, llm_duration_ms)
                usage = response.usage
                console.print_llm_thinking(
                    llm_duration_ms,
                    usage.input_tokens,
                    usage.output_tokens,
                    cache_read=usage.cache_read_input_tokens,
                    cache_create=usage.cache_creation_input_tokens,
                )

                content_blocks = response.content
                for b in content_blocks:
                    if b.type == "text" and b.text.strip():
                        console.print_agent_text(b.text)

                conversation.add_assistant_message(
                    _serialize_content_blocks(content_blocks),
                )

                text = _extract_text(content_blocks)
                script = _extract_final_script(text)
                if script:
                    valid, error_msg = _validate_script(script)
                    tracer.log_script_extracted(
                        script, valid, error_msg,
                    )
                    if valid:
                        result.final_script = script
                        # Skip validation — this is a last-resort
                        # script; let the caller decide quality.
                        result.success = True
                        result.last_resort = True
                        tracer.log_system_event(
                            "last_resort_script_accepted",
                        )

                if not result.success:
                    result.error = (
                        f"Agent exhausted budget ({step_count} "
                        f"steps, {python_step_count} python) "
                        f"without producing a function."
                    )

            result.conversation_length = len(conversation.messages)
            result.steps_executed = step_count
            result.python_steps = python_step_count

        except Exception as exc:
            logger.exception("Agent run failed")
            result.error = str(exc)
            tracer.log_system_event("exception", error=str(exc))

        finally:
            # Clean up any leftover script browser.
            try:
                if active_script_result is not None:
                    await active_script_result.cleanup()
            except (NameError, Exception):
                pass

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
            # diagnostics_dir is set later once tracer creates run_dir.
        )

        await runtime.start()

        # Now that the page exists, create the PageStateManager.
        psm = PageStateManager(runtime.page)
        psm_ref[0] = psm

        # Inject show_page(page), zoom_section(page, ...), and
        # expand_checkpoint(...) into the REPL.
        self._show_page_result_ref: list[Any] = [None]
        show_page_fn = create_show_page_function(psm_ref, self._show_page_result_ref)
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

def _debugging_reminder(turn_count: int) -> str | None:
    """Return a periodic debugging reminder, or None if not due.

    Reminders fire at turns 3, 4, 7, 8, 11, 12, … (pairs with a gap).
    Each tier uses different wording so the model doesn't tune them out.
    """
    if turn_count < 3:
        return None

    # Tier boundaries: 3-4, 7-8, 11-12, ...
    cycle = (turn_count - 3) % 4
    if cycle >= 2:
        # Gap turns — no reminder.
        return None

    tier = (turn_count - 3) // 4
    reminders = [
        # Tier 0 (turns 3-4): gentle
        (
            "Once you understand the root cause, write and submit "
            "the corrected `scrape` function."
        ),
        (
            "Focus on what specifically failed and why. Then submit "
            "the fixed `scrape` function."
        ),
        # Tier 1 (turns 7-8): moderate
        (
            "You have enough context to fix the `scrape` function. "
            "Submit the corrected version when ready."
        ),
        (
            "The goal is a working `scrape` function, not a complete "
            "manual walkthrough. Fix the issue and resubmit."
        ),
        # Tier 2+ (turns 11-12, 15-16, ...): direct
        (
            "Submit the corrected `scrape` function now — further "
            "manual exploration is unlikely to reveal new information."
        ),
        (
            "Time to write the fix. Apply what you learned and "
            "submit the corrected `scrape` function."
        ),
    ]
    # Pick from the appropriate tier, alternating within the pair.
    idx = min(tier * 2 + cycle, len(reminders) - 1)
    return reminders[idx]


def _serialize_content_blocks(blocks: list) -> list[dict]:
    """Convert LLM content blocks to plain dicts for storage."""
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


_EXPECTED_PARAMS = ["page", "start_url", "checkpoint"]
_REQUIRED_SIG = "async def scrape(page, start_url, checkpoint) -> JsonValue:"


def _validate_script(script: str) -> tuple[bool, str]:
    """Validate the agent's function code.

    Checks:
    1. Valid Python syntax
    2. Contains a function named ``scrape``
    3. The function is async
    4. Parameters are exactly ``(page, start_url, checkpoint)`` in order

    Returns ``(True, "")`` on success or ``(False, error_message)`` on
    the first failure.
    """
    # Step 1: Syntax.
    try:
        tree = ast.parse(script)
    except SyntaxError as e:
        return False, f"Syntax error on line {e.lineno}: {e.msg}"

    # Step 2: Find the scrape function.
    scrape_funcs = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "scrape"
    ]
    if not scrape_funcs:
        return False, (
            "No function named `scrape` found. Your code must define:\n"
            f"    {_REQUIRED_SIG}"
        )

    func = scrape_funcs[-1]  # last definition wins

    # Step 3: Must be async.
    if not isinstance(func, ast.AsyncFunctionDef):
        return False, (
            "`scrape` must be an async function. Use "
            "`async def scrape(...)`, not `def scrape(...)`."
        )

    # Step 4: Parameter signature.
    params = [arg.arg for arg in func.args.args]
    if params == _EXPECTED_PARAMS:
        return True, ""

    actual_sig = f"async def scrape({', '.join(params)})"

    # Check for missing parameters.
    for p in _EXPECTED_PARAMS:
        if p not in params:
            return False, (
                f"Parameter `{p}` is missing from `scrape`. "
                f"Required signature:\n    {_REQUIRED_SIG}\n"
                f"Your signature:\n    {actual_sig}"
            )

    # Check for extra parameters.
    extras = [p for p in params if p not in _EXPECTED_PARAMS]
    if extras:
        return False, (
            f"Unexpected parameter `{extras[0]}` in `scrape`. "
            f"Required signature:\n    {_REQUIRED_SIG}\n"
            f"Your signature:\n    {actual_sig}"
        )

    # Must be wrong order.
    return False, (
        f"Parameters are in the wrong order. "
        f"Required signature:\n    {_REQUIRED_SIG}\n"
        f"Your signature:\n    {actual_sig}"
    )


async def _run_script_subprocess(
    script: str,
    *,
    url: str,
    timeout: int = 120,
    checkpoint_dir: Path | None = None,
) -> tuple[str, str | None, str, int]:
    """Run the agent's scrape function in a fresh subprocess.

    Generates an engine wrapper around the agent's code, launches a
    browser, navigates to *url*, calls ``scrape(page, start_url, checkpoint)``,
    and captures the return value.

    Returns:
        ``(stdout, return_value_json, stderr, returncode)`` —
        *return_value_json* is ``None`` if the function raised before
        returning.
    """
    cp_dir = str(checkpoint_dir) if checkpoint_dir else "/tmp/scrape_checkpoints"
    wrapper_code = generate_subprocess_wrapper(script, url, cp_dir)

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
                proc.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return "", None, f"Function timed out after {timeout} seconds", -1

        raw_stdout = stdout_bytes.decode(errors="replace")
        stderr = stderr_bytes.decode(errors="replace")

        # Separate progress output from the serialized return value.
        clean_stdout, return_value_json = parse_return_value(raw_stdout)

        return (
            clean_stdout,
            return_value_json,
            stderr,
            proc.returncode or 0,
        )
    finally:
        shutil.rmtree(script_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════
#  In-process script runner (keeps the page alive for debugging)
# ═══════════════════════════════════════════════════════════════

class InProcessScriptResult:
    """Result of running the scrape function in-process.

    Unlike the subprocess runner, the browser and page stay alive
    after execution so the agent can inspect them for debugging.
    """

    __slots__ = (
        "stdout", "return_value_json", "stderr", "returncode",
        "page", "context", "pw", "profile_dir",
    )

    def __init__(self) -> None:
        self.stdout: str = ""
        self.return_value_json: str | None = None
        self.stderr: str = ""
        self.returncode: int = 0
        self.page: Any = None          # Patchright Page — still live
        self.context: Any = None       # BrowserContext — for cleanup
        self.pw: Any = None            # Playwright instance — for cleanup
        self.profile_dir: str = ""     # Temp profile — for cleanup

    async def cleanup(self) -> None:
        """Close the script browser and free resources."""
        if self.context:
            try:
                await self.context.close()
            except Exception:
                pass
            self.context = None
        if self.pw:
            try:
                await self.pw.stop()
            except Exception:
                pass
            self.pw = None
        if self.profile_dir:
            shutil.rmtree(self.profile_dir, ignore_errors=True)
            self.profile_dir = ""
        self.page = None


async def _run_script_in_process(
    script: str,
    *,
    start_url: str,
    timeout: int = 600,
    checkpoint_dir: Path | None = None,
) -> InProcessScriptResult:
    """Run the agent's scrape function in-process with a second browser.

    Launches a fresh Playwright browser (same stealth config as the
    subprocess wrapper), navigates to *url*, defines the agent's
    ``scrape()`` function, and calls it.  The browser and page stay
    alive after execution so the agent can inspect the page for
    debugging.

    Returns:
        An :class:`InProcessScriptResult` containing stdout, return
        value, error info, **and the live page object**.
    """
    from patchright.async_api import async_playwright
    from ..runtime.environment import BrowserManager

    cp_dir = str(checkpoint_dir) if checkpoint_dir else "/tmp/scrape_checkpoints"
    stealth_args = list(BrowserManager._STEALTH_ARGS)
    profile_dir = tempfile.mkdtemp(prefix="scraper_profile_")

    result = InProcessScriptResult()
    result.profile_dir = profile_dir

    pw = None
    context = None
    try:
        pw = await async_playwright().start()
        context = await pw.chromium.launch_persistent_context(
            user_data_dir=profile_dir,
            channel="chrome",
            headless=False,
            no_viewport=True,
            bypass_csp=True,
            locale="en-US",
            timezone_id="America/New_York",
            args=stealth_args,
        )
        page = (
            context.pages[0]
            if context.pages
            else await context.new_page()
        )
        await page.set_viewport_size({"width": 1920, "height": 1080})
        await page.goto(start_url, wait_until="domcontentloaded")

        # ── Build checkpoint function (same logic as wrapper) ──
        cp_start = time.time()
        cp_counter = [0]

        async def checkpoint(label, data_preview=None):
            cp_counter[0] += 1
            cp_id = f"CP-{cp_counter[0]}"
            cp_url = page.url
            title = await page.title()
            elapsed = time.time() - cp_start
            info = await page.evaluate(
                """() => {
                    const text = document.body
                        ? document.body.innerText : "";
                    const count = document.querySelectorAll("*").length;
                    return { text: text.substring(0, 5000), count };
                }"""
            )
            visible_text = info.get("text", "")
            element_count = info.get("count", 0)
            data = {
                "id": cp_id,
                "label": label,
                "url": cp_url,
                "title": title,
                "timestamp_s": round(elapsed, 1),
                "element_count": element_count,
                "visible_text": visible_text,
                "data_preview": data_preview,
            }
            os.makedirs(cp_dir, exist_ok=True)
            cp_path = os.path.join(cp_dir, f"{cp_id}.json")
            with open(cp_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            t = (
                (title[:50] + "\u2026")
                if len(title) > 50
                else title
            )
            dp = (
                f" | data_preview={len(data_preview)} items"
                if data_preview
                else ""
            )
            print(
                f"[{cp_id} {label}] url={cp_url} | "
                f'title="{t}" | elements={element_count}'
                f"{dp} | {elapsed:.1f}s"
            )

        # ── Define the scrape function in a clean namespace ──
        import io as _io
        exec_globals: dict[str, Any] = {"__builtins__": __builtins__}
        # Pre-imports (same as subprocess wrapper).
        exec(
            compile(
                "import json, re, math, os, time, asyncio, "
                "tempfile, shutil\n"
                "from datetime import datetime\n"
                "from urllib.parse import urljoin, urlparse\n",
                "<script-imports>",
                "exec",
            ),
            exec_globals,
        )
        # Compile and exec the agent's script code.
        exec(compile(script, "<agent-script>", "exec"), exec_globals)

        scrape_fn = exec_globals.get("scrape")
        if scrape_fn is None:
            result.stderr = "Script does not define a scrape() function."
            result.returncode = 1
            result.context = context
            result.pw = pw
            result.page = page
            return result

        # ── Run scrape() capturing stdout ──
        stdout_buf = _io.StringIO()
        old_stdout = sys.stdout
        scrape_error = None
        return_data = None

        try:
            sys.stdout = stdout_buf
            return_data = await asyncio.wait_for(
                scrape_fn(page, start_url, checkpoint),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            scrape_error = f"Function timed out after {timeout} seconds"
        except Exception:
            import traceback
            scrape_error = traceback.format_exc()
        finally:
            sys.stdout = old_stdout

        result.stdout = stdout_buf.getvalue()
        result.page = page
        result.context = context
        result.pw = pw

        if scrape_error:
            result.stderr = scrape_error
            result.returncode = 1
            return result

        # Serialize return value.
        try:
            result.return_value_json = json.dumps(
                return_data,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        except TypeError as exc:
            type_name = type(return_data).__name__
            result.stderr = (
                f"scrape() returned type '{type_name}' which is "
                f"not JSON-serializable: {exc}"
            )
            result.returncode = 1

        return result

    except Exception:
        # Browser launch or navigation failure.
        import traceback
        result.stderr = traceback.format_exc()
        result.returncode = 1
        result.context = context
        result.pw = pw
        return result


def _build_rejection_message(
    feedback: str,
    stdout: str,
    stderr: str,
    returncode: int,
    attempt: int,
    max_attempts: int,
    checkpoints: list[dict] | None = None,
    return_value_json: str | None = None,
    script_source: str | None = None,
) -> str:
    """Build the message sent to the agent after a function is rejected."""
    parts = [
        "## Function Rejected\n",
        f"**Feedback:** {feedback}\n",
    ]

    if script_source:
        src = script_source.strip()
        if len(src) > 4000:
            src = (
                src[:2000]
                + "\n\n... (truncated) ...\n\n"
                + src[-2000:]
            )
        parts.append(f"**Your function:**\n```python\n{src}\n```\n")

    if stdout.strip():
        output = stdout.strip()
        if len(output) > 3000:
            output = (
                output[:1500]
                + "\n\n... (truncated) ...\n\n"
                + output[-1500:]
            )
        parts.append(f"**Progress output:**\n```\n{output}\n```\n")

    if return_value_json is not None:
        rv_display = return_value_json.strip()
        if len(rv_display) > 3000:
            rv_display = (
                rv_display[:1500]
                + "\n\n... (truncated) ...\n\n"
                + rv_display[-1500:]
            )
        parts.append(f"**Return value:**\n```json\n{rv_display}\n```\n")
    elif returncode == 0:
        parts.append(
            "**Return value:** (none — function did not return a value)\n"
        )

    if stderr.strip():
        parts.append(
            f"**Errors:**\n```\n{stderr.strip()[:2000]}\n```\n"
        )

    parts.append(f"**Exit code:** {returncode}\n")
    parts.append(f"**Attempt:** {attempt}/{max_attempts}\n")

    # Include checkpoint summary if available.
    if checkpoints:
        cp_summary = format_checkpoint_summary(checkpoints)
        parts.append(f"\n{cp_summary}\n")

    parts.append(
        "\n---\n\n"
        "Read the error and your function to understand what went "
        "wrong. `page` is in the state where your function ended or "
        "crashed — you can inspect it directly with the `python` "
        "tool.\n\n"
        "Call `await show_page(page)` to see what the page looks "
        "like at the point of failure — this reveals whether the "
        "expected elements are present or whether the page is in an "
        "unexpected state. Call `await zoom_section(page, ...)` to "
        "inspect the actual HTML structure of specific sections. "
        "Zoom is especially important during debugging because the "
        "DOM may differ from what your selectors assumed — the real "
        "attributes, nesting, and classes are only visible in the "
        "zoom output.\n\n"
        "Investigate the root cause, verify your fix, then submit "
        "the corrected function in a ```python block."
    )

    return "\n".join(parts)


def _emit_show_page_log(
    *,
    tracer: Tracer,
    turn_number: int,
    url: str,
    pending_show_page: ShowPageResult,
    show_page_state: ShowPageState,
    pending_is_variant_a: bool,
    reasoning: str,
    referenced: set[str],
    el_matches: list,
    sections_for_filter: list[tuple[str, str]],
    filtered: str,
) -> None:
    """Build and emit the observability log after Phase 3."""
    import time as _time

    variant = "A" if pending_is_variant_a else "B"
    total_sections = len(sections_for_filter)
    total_page_chars = len(pending_show_page.text_output)
    filtered_page_chars = len(filtered)
    compression_ratio = (
        filtered_page_chars / total_page_chars
        if total_page_chars > 0
        else 0.0
    )

    # Similarity score.
    if show_page_state.last_analyzed_text is not None:
        similarity_score = page_similarity(
            pending_show_page.raw_text,
            show_page_state.last_analyzed_text,
        )
    else:
        similarity_score = 0.0  # First page, no baseline.

    # Section ID mentions vs element matches.
    all_section_ids = [s[0] for s in sections_for_filter]
    id_mentioned = get_sections_by_id(reasoning, all_section_ids)
    element_matched_sections = referenced - id_mentioned

    # Compute kept / neighbor / distant counts.
    kept_indices = {
        i for i, s in enumerate(sections_for_filter)
        if s[0] in referenced
    }
    neighbor_indices: set[int] = set()
    for ki in kept_indices:
        for offset in range(-NEIGHBOR_RADIUS, NEIGHBOR_RADIUS + 1):
            idx = ki + offset
            if 0 <= idx < total_sections and idx not in kept_indices:
                neighbor_indices.add(idx)
    total_kept = len(kept_indices)
    total_neighbor = len(neighbor_indices)
    total_distant = total_sections - total_kept - total_neighbor

    # Build ElementMatch log entries from ElementMatchResults.
    element_match_logs: list[ElementMatch] = []
    for m in el_matches:
        if m.matched_attributes:
            best = m.matched_attributes[0]
            attr = best.get("attr", "unknown")
            value = best.get("value", "")
            match_type = best.get("match", "")
            if match_type == "full":
                marker = f'{attr}="{value}"'
            elif attr == "text":
                marker = str(value)
            else:
                marker = str(value)
            # Extract a reasoning snippet around the marker.
            snippet_value = str(value)
            pos = reasoning.find(snippet_value)
            if pos >= 0:
                start = max(0, pos - 40)
                end = min(len(reasoning), pos + len(snippet_value) + 40)
                context = reasoning[start:end].replace("\n", " ").strip()
            else:
                context = ""
            element_match_logs.append(ElementMatch(
                marker=marker,
                marker_type=attr,
                reasoning_context=context,
                matched_sections=m.sections_with_same_element,
                is_ambiguous=len(m.sections_with_same_element) > 1,
            ))

    log_entry = ShowPageAnalysisLog(
        timestamp=_time.time(),
        turn_number=turn_number,
        url=url,
        similarity_score=similarity_score,
        variant_used=variant,
        total_sections=total_sections,
        total_page_chars=total_page_chars,
        analysis_char_count=len(reasoning),
        sections_mentioned_by_id=sorted(id_mentioned),
        sections_matched_by_element=sorted(element_matched_sections),
        total_sections_kept=total_kept,
        total_sections_neighbor=total_neighbor,
        total_sections_distant=total_distant,
        filtered_page_chars=filtered_page_chars,
        compression_ratio=compression_ratio,
        element_matches=element_match_logs,
    )

    tracer.log_show_page_analysis(log_entry)

    reduction_pct = (1.0 - compression_ratio) * 100
    console.print_show_page_analysis(
        variant=variant,
        similarity=similarity_score,
        total_sections=total_sections,
        kept=total_kept,
        neighbor=total_neighbor,
        distant=total_distant,
        original_kb=total_page_chars / 1024,
        filtered_kb=filtered_page_chars / 1024,
        reduction_pct=reduction_pct,
    )
