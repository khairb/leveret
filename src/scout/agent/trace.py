"""Full observability trace for the scraping agent.

Produces a markdown file that shows the complete agent session exactly
as the agent experienced it — every message, every tool call, every
tool result, every LLM response, with timing and token usage.

**Incremental writing** — each event is appended to ``trace.md`` as it
happens, so if the agent crashes mid-run you still have a partial trace.
When ``finish()`` is called, the file is rewritten with final summary
stats in the header.

Usage::

    from agent.trace import Tracer

    tracer = Tracer(output_dir="./traces")
    tracer.start(url="...", task="...", model="...", system_prompt="...")

    # In the loop:
    tracer.log_llm_request(messages, tools)
    tracer.log_llm_response(response)
    tracer.log_tool_call(name, arguments, tool_use_id)
    tracer.log_tool_result(tool_result)
    tracer.log_system_event("Budget warning sent")

    tracer.finish(agent_result)
    # → writes traces/run_2026-03-28_14-30-22/trace.md
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class _StepEntry:
    """One discrete event in the trace."""

    timestamp: float
    kind: str  # "llm_request", "llm_response", "tool_call", "tool_result", "system"
    data: dict[str, Any]


class Tracer:
    """Records a full agent session, writing events incrementally to disk."""

    def __init__(self, output_dir: str | Path = "./traces") -> None:
        self._output_dir = Path(output_dir)
        self._entries: list[_StepEntry] = []
        self._start_time: float = 0
        self._url: str = ""
        self._task: str = ""
        self._model: str = ""
        self._system_prompt: str = ""
        self._initial_raw_html: str = ""
        self._initial_sanitized_html: str = ""
        self._initial_page_view: str = ""
        self._turn_number: int = 0
        self._tool_call_counter: int = 0
        self._run_dir: Path | None = None
        self._trace_path: Path | None = None

    # ── Properties ─────────────────────────────────────────────

    @property
    def run_dir(self) -> Path | None:
        """The run directory, available after ``start()``."""
        return self._run_dir

    # ── Lifecycle ─────────────────────────────────────────────

    def start(
        self,
        url: str,
        task: str,
        model: str,
        system_prompt: str,
    ) -> None:
        """Begin a new trace session.

        Creates the run directory and writes the initial header + system
        prompt to ``trace.md`` immediately.
        """
        self._start_time = time.time()
        self._url = url
        self._task = task
        self._model = model
        self._system_prompt = system_prompt
        self._entries = []
        self._turn_number = 0
        self._tool_call_counter = 0

        # Create run directory.
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._run_dir = self._output_dir / f"run_{ts}"
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._trace_path = self._run_dir / "trace.md"

        # Write initial header (stats will be rewritten in finish()).
        lines = [
            f"# Agent Trace — {ts}",
            "",
            f"- **URL**: `{url}`",
            f"- **Task**: {task}",
            f"- **Model**: `{model}`",
            f"- **Status**: IN PROGRESS...",
            "",
            "---",
            "",
            "## System Prompt",
            "",
            "<details>",
            "<summary>Click to expand system prompt</summary>",
            "",
            "````",
            system_prompt,
            "````",
            "",
            "</details>",
            "",
            "---",
            "",
            "## Conversation",
            "",
        ]
        self._trace_path.write_text("\n".join(lines), encoding="utf-8")

        self._add("system", {
            "event": "session_start",
            "url": url,
            "task": task,
            "model": model,
        })

    def finish(self, agent_result: Any) -> Path:
        """End the session and rewrite the trace with final summary stats.

        Also saves initial HTML as ``page.html``.

        Returns:
            Path to the **run directory** containing all artifacts.
        """
        self._add("system", {
            "event": "session_end",
            "success": agent_result.success,
            "error": agent_result.error,
            "steps_executed": agent_result.steps_executed,
            "python_steps": agent_result.python_steps,
            "conversation_length": agent_result.conversation_length,
        })

        # Rewrite the full trace with final header stats.
        self._rewrite_final_trace()

        return self._run_dir

    # ── Logging methods ───────────────────────────────────────

    def log_initial_page_view(
        self,
        page_view: str,
        raw_html: str = "",
        sanitized_html: str = "",
    ) -> None:
        """Log the initial page view and save all representations to disk."""
        self._initial_raw_html = raw_html
        self._initial_sanitized_html = sanitized_html
        self._initial_page_view = page_view

        # Save all three representations immediately (crash-safe).
        if self._run_dir:
            if raw_html:
                (self._run_dir / "page_raw.html").write_text(
                    raw_html, encoding="utf-8",
                )
            if sanitized_html:
                (self._run_dir / "page_sanitized.html").write_text(
                    sanitized_html, encoding="utf-8",
                )
            if page_view:
                (self._run_dir / "page_view.txt").write_text(
                    page_view, encoding="utf-8",
                )

        self._add("system", {
            "event": "initial_page_view",
            "page_view": page_view,
        })

    def log_llm_request(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> None:
        """Log the full request sent to the LLM."""
        self._turn_number += 1
        self._add("llm_request", {
            "turn": self._turn_number,
            "message_count": len(messages),
            "has_tools": tools is not None,
            "latest_message": messages[-1] if messages else None,
        })

    def log_llm_response(
        self,
        response: Any,
        duration_ms: float,
    ) -> None:
        """Log the full LLM response."""
        content_blocks = []
        for b in response.content:
            if b.type == "text":
                content_blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                content_blocks.append({
                    "type": "tool_use",
                    "id": b.id,
                    "name": b.name,
                    "input": b.input,
                })

        self._add("llm_response", {
            "turn": self._turn_number,
            "stop_reason": response.stop_reason,
            "content_blocks": content_blocks,
            "duration_ms": round(duration_ms, 1),
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        })

    def log_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
        tool_use_id: str,
    ) -> None:
        """Log a tool call before execution."""
        self._tool_call_counter += 1
        self._add("tool_call", {
            "number": self._tool_call_counter,
            "name": name,
            "tool_use_id": tool_use_id,
            "arguments": arguments,
        })

    def log_tool_result(
        self,
        name: str,
        tool_use_id: str,
        content: str,
        is_error: bool,
        duration_ms: float,
    ) -> None:
        """Log a tool result after execution."""
        self._add("tool_result", {
            "name": name,
            "tool_use_id": tool_use_id,
            "content": content,
            "is_error": is_error,
            "duration_ms": round(duration_ms, 1),
        })

    def log_system_event(self, message: str, **extra: Any) -> None:
        """Log a system-level event (budget warnings, nudges, errors)."""
        self._add("system", {"event": message, **extra})

    def log_script_extracted(
        self,
        script: str,
        valid: bool,
        error: str = "",
    ) -> None:
        """Log when a final script is detected and validated."""
        self._add("system", {
            "event": "script_extracted",
            "valid": valid,
            "error": error,
            "script_length": len(script),
            "script_preview": script[:500] + ("..." if len(script) > 500 else ""),
        })

    # ── Internal ──────────────────────────────────────────────

    def _add(self, kind: str, data: dict[str, Any]) -> None:
        """Store an entry in memory AND append its markdown to disk."""
        entry = _StepEntry(
            timestamp=time.time(),
            kind=kind,
            data=data,
        )
        self._entries.append(entry)

        # Append to the live trace file.
        if self._trace_path and self._trace_path.exists():
            md = self._render_entry(entry)
            if md:
                with open(self._trace_path, "a", encoding="utf-8") as f:
                    f.write(md)

    def _render_entry(self, entry: _StepEntry) -> str:
        """Render a single entry as markdown text."""
        lines: list[str] = []
        _a = lines.append
        elapsed = entry.timestamp - self._start_time
        ts_str = f"`[{elapsed:7.1f}s]`"

        if entry.kind == "system":
            event = entry.data.get("event", "")

            if event == "session_start":
                pass  # Already in header.

            elif event == "initial_page_view":
                _a(f"### {ts_str} Initial Page View")
                _a("")
                _a("````")
                _a(entry.data["page_view"])
                _a("````")
                _a("")

            elif event == "session_end":
                _a("---")
                _a("")
                _a(f"### {ts_str} Session End")
                _a("")
                success = entry.data.get("success", False)
                _a(f"- **Success**: {success}")
                if entry.data.get("error"):
                    _a(f"- **Error**: {entry.data['error']}")
                _a(f"- **Steps**: {entry.data.get('steps_executed', 0)} total, {entry.data.get('python_steps', 0)} python")
                _a(f"- **Conversation messages**: {entry.data.get('conversation_length', 0)}")
                _a("")

            elif event == "script_extracted":
                _a(f"### {ts_str} Script Extracted")
                _a("")
                _a(f"- **Valid syntax**: {entry.data.get('valid', False)}")
                if entry.data.get("error"):
                    _a(f"- **Syntax error**: {entry.data['error']}")
                _a(f"- **Length**: {entry.data.get('script_length', 0)} chars")
                _a("")
                _a("```python")
                _a(entry.data.get("script_preview", ""))
                _a("```")
                _a("")

            else:
                _a(f"#### {ts_str} System: {event}")
                extra = {k: v for k, v in entry.data.items() if k != "event"}
                if extra:
                    _a("")
                    for k, v in extra.items():
                        _a(f"- **{k}**: {v}")
                _a("")

        elif entry.kind == "llm_request":
            turn = entry.data["turn"]
            _a("---")
            _a("")
            _a(f"### {ts_str} Turn {turn} — LLM Request")
            _a("")
            _a(f"- **Messages in context**: {entry.data['message_count']}")
            _a("")

            latest = entry.data.get("latest_message")
            if latest:
                _a("**Latest message sent** (what triggered this call):")
                _a("")
                content = latest.get("content", "")
                if isinstance(content, str):
                    _a("````")
                    _a(content)
                    _a("````")
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            btype = block.get("type", "")
                            if btype == "tool_result":
                                tid = block.get("tool_use_id", "?")
                                is_err = block.get("is_error", False)
                                _a(f"**Tool result** `{tid}`" + (" (ERROR)" if is_err else "") + ":")
                                _a("")
                                _a("````")
                                _a(str(block.get("content", "")))
                                _a("````")
                                _a("")
                            else:
                                _a("````")
                                _a(json.dumps(block, indent=2, default=str))
                                _a("````")
                        else:
                            _a(f"```\n{block}\n```")
                _a("")

        elif entry.kind == "llm_response":
            turn = entry.data["turn"]
            dur = entry.data.get("duration_ms", 0)
            usage = entry.data.get("usage", {})
            stop = entry.data.get("stop_reason", "?")

            _a(f"### {ts_str} Turn {turn} — LLM Response")
            _a("")
            _a(f"- **Stop reason**: `{stop}`")
            _a(f"- **Duration**: {dur:.0f}ms")
            _a(f"- **Tokens**: {usage.get('input_tokens', 0):,} in, {usage.get('output_tokens', 0):,} out")
            _a("")

            for block in entry.data.get("content_blocks", []):
                btype = block.get("type", "")
                if btype == "text":
                    _a("**Agent says:**")
                    _a("")
                    _a("````")
                    _a(block["text"])
                    _a("````")
                    _a("")
                elif btype == "tool_use":
                    name = block.get("name", "?")
                    tid = block.get("id", "?")
                    args = block.get("input", {})
                    _a(f"**Tool call -> `{name}`** (id: `{tid}`)")
                    _a("")
                    if name == "python":
                        _a("```python")
                        _a(args.get("code", ""))
                        _a("```")
                    else:
                        _a("```json")
                        _a(json.dumps(args, indent=2, default=str))
                        _a("```")
                    _a("")

        elif entry.kind == "tool_call":
            num = entry.data["number"]
            name = entry.data["name"]
            _a(f"#### {ts_str} Executing tool #{num}: `{name}`")
            _a("")

        elif entry.kind == "tool_result":
            name = entry.data["name"]
            dur = entry.data.get("duration_ms", 0)
            is_err = entry.data.get("is_error", False)

            status = "ERROR" if is_err else "OK"
            _a(f"#### {ts_str} Tool result: `{name}` [{status}] ({dur:.0f}ms)")
            _a("")
            _a("<details>")
            _a(f"<summary>Full tool output ({len(entry.data.get('content', ''))} chars)</summary>")
            _a("")
            _a("````")
            _a(entry.data.get("content", ""))
            _a("````")
            _a("")
            _a("</details>")
            _a("")

        if not lines:
            return ""
        return "\n".join(lines) + "\n"

    def _rewrite_final_trace(self) -> None:
        """Rewrite the trace file with final summary stats in the header."""
        if not self._trace_path:
            return

        ts = self._trace_path.parent.name.replace("run_", "")
        lines: list[str] = []
        _a = lines.append

        # ── Header with final stats ────────────────────────────
        total_duration = (
            (self._entries[-1].timestamp - self._start_time)
            if self._entries
            else 0
        )

        total_input = 0
        total_output = 0
        for e in self._entries:
            if e.kind == "llm_response" and "usage" in e.data:
                total_input += e.data["usage"]["input_tokens"]
                total_output += e.data["usage"]["output_tokens"]

        _a(f"# Agent Trace — {ts}")
        _a("")
        _a(f"- **URL**: `{self._url}`")
        _a(f"- **Task**: {self._task}")
        _a(f"- **Model**: `{self._model}`")
        _a(f"- **Duration**: {total_duration:.1f}s")
        _a(f"- **Turns**: {self._turn_number}")
        _a(f"- **Tool calls**: {self._tool_call_counter}")
        _a(f"- **Tokens**: {total_input:,} input, {total_output:,} output, {total_input + total_output:,} total")
        _a("")

        # ── System prompt ──────────────────────────────────────
        _a("---")
        _a("")
        _a("## System Prompt")
        _a("")
        _a("<details>")
        _a("<summary>Click to expand system prompt</summary>")
        _a("")
        _a("````")
        _a(self._system_prompt)
        _a("````")
        _a("")
        _a("</details>")
        _a("")

        # ── All entries ────────────────────────────────────────
        _a("---")
        _a("")
        _a("## Conversation")
        _a("")

        for entry in self._entries:
            md = self._render_entry(entry)
            if md:
                lines.append(md.rstrip("\n"))
                lines.append("")

        self._trace_path.write_text("\n".join(lines), encoding="utf-8")
