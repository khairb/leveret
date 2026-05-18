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

        # Save system prompt as standalone file for trace_viewer.
        (self._run_dir / "system_prompt.txt").write_text(
            system_prompt,
            encoding="utf-8",
        )

        # Write initial header (stats will be rewritten in finish()).
        lines = [
            f"# Agent Trace — {ts}",
            "",
            f"- **URL**: `{url}`",
            f"- **Task**: {task}",
            f"- **Model**: `{model}`",
            "- **Status**: IN PROGRESS...",
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

        self._add(
            "system",
            {
                "event": "session_start",
                "url": url,
                "task": task,
                "model": model,
            },
        )

    def finish(self, agent_result: Any) -> Path:
        """End the session and rewrite the trace with final summary stats.

        Also saves initial HTML as ``page.html``.

        Returns:
            Path to the **run directory** containing all artifacts.
        """
        self._add(
            "system",
            {
                "event": "session_end",
                "success": agent_result.success,
                "error": agent_result.error,
                "steps_executed": agent_result.steps_executed,
                "python_steps": agent_result.python_steps,
                "conversation_length": agent_result.conversation_length,
            },
        )

        # Rewrite the full trace with final header stats.
        self._rewrite_final_trace()

        # Generate history_stats.txt from saved snapshots.
        self._generate_history_stats()

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
                    raw_html,
                    encoding="utf-8",
                )
            if sanitized_html:
                (self._run_dir / "page_sanitized.html").write_text(
                    sanitized_html,
                    encoding="utf-8",
                )
            if page_view:
                (self._run_dir / "page_view.txt").write_text(
                    page_view,
                    encoding="utf-8",
                )

        self._add(
            "system",
            {
                "event": "initial_page_view",
                "page_view": page_view,
            },
        )

    def log_llm_request(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> None:
        """Log the full request sent to the LLM."""
        self._turn_number += 1
        self._add(
            "llm_request",
            {
                "turn": self._turn_number,
                "message_count": len(messages),
                "has_tools": tools is not None,
                "latest_message": messages[-1] if messages else None,
            },
        )

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
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": b.id,
                        "name": b.name,
                        "input": b.input,
                    }
                )

        self._add(
            "llm_response",
            {
                "turn": self._turn_number,
                "stop_reason": response.stop_reason,
                "content_blocks": content_blocks,
                "duration_ms": round(duration_ms, 1),
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cache_read_input_tokens": response.usage.cache_read_input_tokens,
                    "cache_creation_input_tokens": response.usage.cache_creation_input_tokens,
                },
            },
        )

    def log_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
        tool_use_id: str,
    ) -> None:
        """Log a tool call before execution."""
        self._tool_call_counter += 1
        self._add(
            "tool_call",
            {
                "number": self._tool_call_counter,
                "name": name,
                "tool_use_id": tool_use_id,
                "arguments": arguments,
            },
        )

    def log_tool_result(
        self,
        name: str,
        tool_use_id: str,
        content: str,
        is_error: bool,
        duration_ms: float,
    ) -> None:
        """Log a tool result after execution."""
        self._add(
            "tool_result",
            {
                "name": name,
                "tool_use_id": tool_use_id,
                "content": content,
                "is_error": is_error,
                "duration_ms": round(duration_ms, 1),
            },
        )

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
        self._add(
            "system",
            {
                "event": "script_extracted",
                "valid": valid,
                "error": error,
                "script_length": len(script),
                "script_preview": script[:500] + ("..." if len(script) > 500 else ""),
            },
        )

    def log_compression(self, meta: dict[str, Any]) -> None:
        """Log a history compression event."""
        self._add(
            "system",
            {
                "event": "history_compression",
                **meta,
            },
        )

    def log_show_page_analysis(self, log: Any) -> None:
        """Log a show_page analysis cycle.

        Args:
            log: A :class:`ShowPageAnalysisLog` dataclass instance.
        """
        from dataclasses import asdict

        self._add(
            "system",
            {
                "event": "show_page_analysis",
                **asdict(log),
            },
        )

    # ── History snapshots ─────────────────────────────────────

    def save_history_snapshot(
        self,
        turn_number: int,
        messages: list[dict],
        usage: dict[str, Any] | None = None,
        step_count: int = 0,
        python_step_count: int = 0,
    ) -> None:
        """Save a JSON snapshot of the conversation history after a turn.

        Creates ``snapshots/turn_{N}.json`` in the run directory with:
        - Full message history (roles, content sizes, types)
        - Per-message token estimates (chars / 4 as rough approximation)
        - Cumulative token usage from LLM responses
        - Timing information
        """
        if not self._run_dir:
            return

        snap_dir = self._run_dir / "snapshots"
        snap_dir.mkdir(exist_ok=True)

        elapsed = time.time() - self._start_time

        # Build per-message metadata AND store full content for
        # debugging.  The block-level stats (types, sizes) are kept
        # for the history_stats report; the raw content is added so
        # snapshots can be inspected directly.
        message_stats = []
        for idx, msg in enumerate(messages):
            role = msg.get("role", "?")
            content = msg.get("content", "")

            if isinstance(content, str):
                char_count = len(content)
                content_types = ["text"]
                block_details = [{"type": "text", "chars": char_count}]
            elif isinstance(content, list):
                char_count = 0
                content_types = []
                block_details = []
                for block in content:
                    btype = block.get("type", "unknown")
                    content_types.append(btype)
                    if btype == "text":
                        text = block.get("text", "")
                        bchars = len(text)
                    elif btype == "tool_use":
                        code = block.get("input", {}).get("code", "")
                        bchars = len(json.dumps(block.get("input", {})))
                        block_details.append(
                            {
                                "type": "tool_use",
                                "name": block.get("name", "?"),
                                "chars": bchars,
                                "code_chars": len(code) if code else 0,
                            }
                        )
                        char_count += bchars
                        continue
                    elif btype == "tool_result":
                        result_content = block.get("content", "")
                        bchars = (
                            len(result_content)
                            if isinstance(result_content, str)
                            else len(json.dumps(result_content))
                        )
                        block_details.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.get("tool_use_id", "?"),
                                "chars": bchars,
                                "is_error": block.get("is_error", False),
                            }
                        )
                        char_count += bchars
                        continue
                    else:
                        bchars = len(json.dumps(block, default=str))
                    block_details.append({"type": btype, "chars": bchars})
                    char_count += bchars
            else:
                char_count = len(str(content))
                content_types = ["unknown"]
                block_details = [{"type": "unknown", "chars": char_count}]

            message_stats.append(
                {
                    "index": idx,
                    "role": role,
                    "chars": char_count,
                    "estimated_tokens": char_count // 4,
                    "content_types": list(set(content_types)),
                    "block_count": len(block_details),
                    "blocks": block_details,
                    "content": content,
                }
            )

        # Aggregate LLM usage from trace entries up to this turn.
        cumulative_usage = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cache_read": 0,
            "total_cache_create": 0,
        }
        per_turn_usage = []
        for e in self._entries:
            if e.kind == "llm_response" and "usage" in e.data:
                u = e.data["usage"]
                cumulative_usage["total_input_tokens"] += u.get("input_tokens", 0)
                cumulative_usage["total_output_tokens"] += u.get("output_tokens", 0)
                cumulative_usage["total_cache_read"] += u.get("cache_read_input_tokens", 0)
                cumulative_usage["total_cache_create"] += u.get("cache_creation_input_tokens", 0)
                per_turn_usage.append(
                    {
                        "turn": e.data.get("turn", 0),
                        "input_tokens": u.get("input_tokens", 0),
                        "output_tokens": u.get("output_tokens", 0),
                        "cache_read": u.get("cache_read_input_tokens", 0),
                        "cache_create": u.get("cache_creation_input_tokens", 0),
                        "duration_ms": e.data.get("duration_ms", 0),
                    }
                )

        snapshot = {
            "turn": turn_number,
            "timestamp": time.time(),
            "elapsed_s": round(elapsed, 1),
            "step_count": step_count,
            "python_step_count": python_step_count,
            "message_count": len(messages),
            "total_chars": sum(m["chars"] for m in message_stats),
            "total_estimated_tokens": sum(m["estimated_tokens"] for m in message_stats),
            "messages": message_stats,
            "cumulative_usage": cumulative_usage,
            "per_turn_usage": per_turn_usage,
            "current_usage": usage or {},
        }

        path = snap_dir / f"turn_{turn_number:03d}.json"
        path.write_text(
            json.dumps(snapshot, indent=2, default=str),
            encoding="utf-8",
        )

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
                _a(
                    f"- **Steps**: {entry.data.get('steps_executed', 0)} total, {entry.data.get('python_steps', 0)} python"
                )
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
                                _a(
                                    f"**Tool result** `{tid}`"
                                    + (" (ERROR)" if is_err else "")
                                    + ":"
                                )
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
            _a(
                f"- **Tokens**: {usage.get('input_tokens', 0):,} in, {usage.get('output_tokens', 0):,} out"
            )
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

    def _generate_history_stats(self) -> None:
        """Generate ``history_stats.txt`` from saved snapshots.

        Reads all ``snapshots/turn_*.json`` files and writes a
        comprehensive plain-text statistics report into the run
        directory.  Called automatically by ``finish()``.
        """
        if not self._run_dir:
            return

        snap_dir = self._run_dir / "snapshots"
        if not snap_dir.is_dir():
            return
        files = sorted(snap_dir.glob("turn_*.json"))
        if not files:
            return

        snapshots = [json.loads(f.read_text()) for f in files]
        report = _build_stats_report(
            snapshots,
            self._entries,
            self._model,
        )
        (self._run_dir / "history_stats.txt").write_text(report, encoding="utf-8")

    def _rewrite_final_trace(self) -> None:
        """Rewrite the trace file with final summary stats in the header."""
        if not self._trace_path:
            return

        ts = self._trace_path.parent.name.replace("run_", "")
        lines: list[str] = []
        _a = lines.append

        # ── Header with final stats ────────────────────────────
        total_duration = (self._entries[-1].timestamp - self._start_time) if self._entries else 0

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
        _a(
            f"- **Tokens**: {total_input:,} input, {total_output:,} output, {total_input + total_output:,} total"
        )
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


# ═══════════════════════════════════════════════════════════════
#  History stats report builder (plain text, no ANSI)
# ═══════════════════════════════════════════════════════════════

_BAR = "\u2588"  # █
_BAR_EMPTY = "\u2591"  # ░
_SPARKS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"


def _bar(value: float, max_val: float, width: int = 30) -> str:
    if max_val <= 0:
        return ""
    filled = min(int((value / max_val) * width), width)
    return _BAR * filled + _BAR_EMPTY * (width - filled)


def _spark(values: list[float]) -> str:
    if not values:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    return "".join(_SPARKS[min(int((v - mn) / rng * 7), 7)] for v in values)


def _pct(part: float, total: float) -> str:
    if total <= 0:
        return "  0.0%"
    return f"{part / total * 100:5.1f}%"


def _fmt_tok(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _fmt_ch(n: int) -> str:
    return _fmt_tok(n)


_MODEL_PRICING: dict[str, dict[str, float]] = {
    # Anthropic — prices per million tokens (USD).
    # Cache read = 0.1× input, cache write = 1.25× input.
    "claude-opus-4": {"input": 15.0, "output": 75.0, "cache_read": 1.50, "cache_write": 18.75},
    "claude-opus-4-5": {"input": 5.0, "output": 25.0, "cache_read": 0.50, "cache_write": 6.25},
    "claude-opus-4-6": {"input": 5.0, "output": 25.0, "cache_read": 0.50, "cache_write": 6.25},
    "claude-sonnet-4": {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_write": 3.75},
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_write": 3.75},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_write": 3.75},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0, "cache_read": 0.10, "cache_write": 1.25},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.0, "cache_read": 0.08, "cache_write": 1.00},
    # OpenAI — prices per million tokens (USD).
    # No separate cache-write fee; cache_write = input price.
    "gpt-4o": {"input": 2.50, "output": 10.0, "cache_read": 1.25, "cache_write": 2.50},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cache_read": 0.075, "cache_write": 0.15},
    "gpt-4.1": {"input": 2.0, "output": 8.0, "cache_read": 0.50, "cache_write": 2.0},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "cache_read": 0.10, "cache_write": 0.40},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40, "cache_read": 0.025, "cache_write": 0.10},
    "o1": {"input": 15.0, "output": 60.0, "cache_read": 7.50, "cache_write": 15.0},
    "o1-mini": {"input": 3.0, "output": 12.0, "cache_read": 1.50, "cache_write": 3.0},
    "o3": {"input": 2.0, "output": 8.0, "cache_read": 0.50, "cache_write": 2.0},
    "o3-mini": {"input": 1.10, "output": 4.40, "cache_read": 0.55, "cache_write": 1.10},
    "o4-mini": {"input": 1.10, "output": 4.40, "cache_read": 0.275, "cache_write": 1.10},
}


def _resolve_pricing(model: str) -> dict[str, float] | None:
    """Resolve a model string like 'anthropic:claude-sonnet-4-20250514' to pricing."""
    # Strip provider prefix.
    name = model.split(":", 1)[-1] if ":" in model else model
    # Exact match first.
    if name in _MODEL_PRICING:
        return _MODEL_PRICING[name]
    # Strip date suffixes (e.g. claude-sonnet-4-20250514 → claude-sonnet-4).
    for key in sorted(_MODEL_PRICING, key=len, reverse=True):
        if name.startswith(key):
            return _MODEL_PRICING[key]
    return None


def _calc_cost(
    pricing: dict[str, float],
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_create_tokens: int = 0,
) -> float:
    """Calculate cost in USD given per-million-token prices and token counts."""
    m = 1_000_000
    return (
        (input_tokens / m) * pricing["input"]
        + (output_tokens / m) * pricing["output"]
        + (cache_read_tokens / m) * pricing["cache_read"]
        + (cache_create_tokens / m) * pricing["cache_write"]
    )


def _build_stats_report(
    snapshots: list[dict],
    entries: list[_StepEntry] | None = None,
    model: str = "",
) -> str:
    """Build a complete plain-text statistics report from snapshot data."""
    o: list[str] = []
    w = o.append

    latest = snapshots[-1]
    total_chars = latest["total_chars"]
    total_est_tok = latest["total_estimated_tokens"]
    messages = latest["messages"]
    cu = latest.get("cumulative_usage", {})
    per_turn = latest.get("per_turn_usage", [])
    total_input = cu.get("total_input_tokens", 0)
    total_output = cu.get("total_output_tokens", 0)
    grand_total = total_input + total_output

    # ── Header ────────────────────────────────────────────────
    w("=" * 72)
    w(f"  HISTORY STATISTICS — {len(snapshots)} turns")
    w("=" * 72)
    w("")
    w(f"  Turns:        {latest['turn']}")
    w(f"  Messages:     {latest['message_count']}")
    w(f"  History:      {_fmt_ch(total_chars)} chars  (~{_fmt_tok(total_est_tok)} tokens)")
    w(
        f"  LLM tokens:   {_fmt_tok(grand_total)} total  "
        f"({_fmt_tok(total_input)} in, {_fmt_tok(total_output)} out)"
    )
    w(
        f"  Steps:        {latest.get('step_count', '?')} "
        f"({latest.get('python_step_count', '?')} python)"
    )
    w(f"  Duration:     {latest['elapsed_s']:.0f}s")
    if total_output > 0:
        w(
            f"  I/O ratio:    {total_input / total_output:.1f}x  "
            f"(high = big context, small responses)"
        )
    w("")

    # ── Growth over turns ─────────────────────────────────────
    if len(snapshots) >= 2:
        w("-" * 72)
        w("  HISTORY GROWTH")
        w("-" * 72)
        w("")
        chars_list = [s["total_chars"] for s in snapshots]
        msg_list = [s["message_count"] for s in snapshots]
        tok_list = [s["total_estimated_tokens"] for s in snapshots]
        w(
            f"  Chars:    {_spark([float(v) for v in chars_list])}  "
            f"({_fmt_ch(chars_list[0])} -> {_fmt_ch(chars_list[-1])})"
        )
        w(f"  Messages: {_spark([float(v) for v in msg_list])}  ({msg_list[0]} -> {msg_list[-1]})")
        w(
            f"  Tokens:   {_spark([float(v) for v in tok_list])}  "
            f"({_fmt_tok(tok_list[0])} -> {_fmt_tok(tok_list[-1])})"
        )
        w("")

        max_ch = max(chars_list)
        prev_ch = 0
        w(
            f"  {'Turn':>5} {'Msgs':>5} {'Chars':>10} {'Est.Tok':>10} "
            f"{'Delta':>10} {'Elapsed':>8}  Growth"
        )
        w(f"  {'─' * 5} {'─' * 5} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 8}  {'─' * 25}")
        for s in snapshots:
            delta = s["total_chars"] - prev_ch
            d_str = f"+{_fmt_ch(delta)}" if delta > 0 else _fmt_ch(delta)
            w(
                f"  {s['turn']:>5} {s['message_count']:>5} "
                f"{_fmt_ch(s['total_chars']):>10} "
                f"{_fmt_tok(s['total_estimated_tokens']):>10} "
                f"{d_str:>10} {s['elapsed_s']:>7.0f}s  "
                f"{_bar(s['total_chars'], max_ch, 25)}"
            )
            prev_ch = s["total_chars"]
        w("")

        total_growth = chars_list[-1] - chars_list[0]
        n_turns = snapshots[-1]["turn"] - snapshots[0]["turn"]
        if n_turns > 0:
            avg = total_growth / n_turns
            w(
                f"  Avg growth: {_fmt_ch(int(avg))} chars/turn  "
                f"(~{_fmt_tok(int(avg) // 4)} tokens/turn)"
            )
        w("")

    # ── LLM token usage ───────────────────────────────────────
    if grand_total > 0:
        w("-" * 72)
        w("  LLM TOKEN USAGE")
        w("-" * 72)
        w("")
        w(f"  {'Category':<25} {'Tokens':>12} {'%':>7}  Bar")
        w(f"  {'─' * 25} {'─' * 12} {'─' * 7}  {'─' * 30}")
        w(
            f"  {'Input':<25} {_fmt_tok(total_input):>12} "
            f"{_pct(total_input, grand_total):>7}  "
            f"{_bar(total_input, grand_total)}"
        )
        w(
            f"  {'Output':<25} {_fmt_tok(total_output):>12} "
            f"{_pct(total_output, grand_total):>7}  "
            f"{_bar(total_output, grand_total)}"
        )
        cache_read = cu.get("total_cache_read", 0)
        cache_create = cu.get("total_cache_create", 0)
        if cache_read:
            w(f"  {'Cache read':<25} {_fmt_tok(cache_read):>12}")
        if cache_create:
            w(f"  {'Cache write':<25} {_fmt_tok(cache_create):>12}")
        w(f"  {'─' * 25} {'─' * 12}")
        w(f"  {'TOTAL':<25} {_fmt_tok(grand_total):>12}")
        w("")

    # ── Per-turn token table ──────────────────────────────────
    if per_turn:
        input_vals = [t["input_tokens"] for t in per_turn]
        output_vals = [t["output_tokens"] for t in per_turn]
        max_inp = max(input_vals) if input_vals else 1
        w(
            f"  Input:  {_spark([float(v) for v in input_vals])}  "
            f"(min={_fmt_tok(min(input_vals))}, max={_fmt_tok(max(input_vals))}, "
            f"avg={_fmt_tok(sum(input_vals) // len(input_vals))})"
        )
        w(
            f"  Output: {_spark([float(v) for v in output_vals])}  "
            f"(min={_fmt_tok(min(output_vals))}, max={_fmt_tok(max(output_vals))}, "
            f"avg={_fmt_tok(sum(output_vals) // len(output_vals))})"
        )
        w("")
        w(f"  {'Turn':>5} {'Input':>10} {'Output':>10} {'Duration':>10}  Bar")
        w(f"  {'─' * 5} {'─' * 10} {'─' * 10} {'─' * 10}  {'─' * 30}")
        for t in per_turn:
            dur = f"{t['duration_ms'] / 1000:.1f}s" if t.get("duration_ms") else "?"
            w(
                f"  {t['turn']:>5} {_fmt_tok(t['input_tokens']):>10} "
                f"{_fmt_tok(t['output_tokens']):>10} {dur:>10}  "
                f"{_bar(t['input_tokens'], max_inp)}"
            )
        w("")
        if len(per_turn) >= 2:
            growths = [
                per_turn[i]["input_tokens"] - per_turn[i - 1]["input_tokens"]
                for i in range(1, len(per_turn))
            ]
            avg_g = sum(growths) / len(growths)
            max_g_idx = growths.index(max(growths))
            w(f"  Avg input growth/turn: {_fmt_tok(int(avg_g))} tokens")
            w(
                f"  Largest jump: Turn {per_turn[max_g_idx + 1]['turn']} "
                f"(+{_fmt_tok(max(growths))} tokens)"
            )
            w("")

    # ── Cost estimate ─────────────────────────────────────────
    pricing = _resolve_pricing(model) if model else None
    if pricing and grand_total > 0:
        cache_read = cu.get("total_cache_read", 0)
        cache_create = cu.get("total_cache_create", 0)
        # Actual input tokens billed = input - cache_read - cache_create
        # (cache tokens are billed separately at their own rates).
        billed_input = max(total_input - cache_read - cache_create, 0)
        total_cost = _calc_cost(
            pricing,
            billed_input,
            total_output,
            cache_read,
            cache_create,
        )

        w("-" * 72)
        w("  COST ESTIMATE")
        w("-" * 72)
        w("")
        model_display = model.split(":", 1)[-1] if ":" in model else model
        w(f"  Model: {model_display}")
        w("  Pricing (per 1M tokens):")
        w(
            f"    Input: ${pricing['input']:.2f}  |  Output: ${pricing['output']:.2f}  "
            f"|  Cache read: ${pricing['cache_read']:.3f}  "
            f"|  Cache write: ${pricing['cache_write']:.2f}"
        )
        w("")
        w(f"  {'Component':<25} {'Tokens':>12} {'Rate':>10} {'Cost':>10}")
        w(f"  {'─' * 25} {'─' * 12} {'─' * 10} {'─' * 10}")

        def _d(v: float, decimals: int = 4) -> str:
            """Format a dollar amount."""
            return f"${v:.{decimals}f}"

        input_cost = (billed_input / 1_000_000) * pricing["input"]
        output_cost = (total_output / 1_000_000) * pricing["output"]
        cr_cost = (cache_read / 1_000_000) * pricing["cache_read"]
        cw_cost = (cache_create / 1_000_000) * pricing["cache_write"]

        w(
            f"  {'Input (non-cached)':<25} {_fmt_tok(billed_input):>12} "
            f"{_d(pricing['input'], 2):>10} {_d(input_cost):>10}"
        )
        w(
            f"  {'Output':<25} {_fmt_tok(total_output):>12} "
            f"{_d(pricing['output'], 2):>10} {_d(output_cost):>10}"
        )
        if cache_read:
            w(
                f"  {'Cache read':<25} {_fmt_tok(cache_read):>12} "
                f"{_d(pricing['cache_read'], 3):>10} {_d(cr_cost):>10}"
            )
        if cache_create:
            w(
                f"  {'Cache write':<25} {_fmt_tok(cache_create):>12} "
                f"{_d(pricing['cache_write'], 2):>10} {_d(cw_cost):>10}"
            )
        w(f"  {'─' * 25} {'':>12} {'':>10} {'─' * 10}")
        w(f"  {'TOTAL':<25} {'':>12} {'':>10} {_d(total_cost):>10}")
        w("")

        # Per-turn cost breakdown.
        if per_turn:
            w("  PER-TURN COST")
            w(
                f"  {'Turn':>5} {'Input':>10} {'Output':>10} "
                f"{'CacheRd':>10} {'CacheWr':>10} {'Cost':>10}"
            )
            w(f"  {'─' * 5} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10}")
            cumulative_cost = 0.0
            for t in per_turn:
                t_cr = t.get("cache_read", 0)
                t_cw = t.get("cache_create", 0)
                t_inp = max(t["input_tokens"] - t_cr - t_cw, 0)
                t_cost = _calc_cost(
                    pricing,
                    t_inp,
                    t["output_tokens"],
                    t_cr,
                    t_cw,
                )
                cumulative_cost += t_cost
                w(
                    f"  {t['turn']:>5} {_fmt_tok(t_inp):>10} "
                    f"{_fmt_tok(t['output_tokens']):>10} "
                    f"{_fmt_tok(t_cr):>10} {_fmt_tok(t_cw):>10} "
                    f"{_d(t_cost):>10}"
                )
            w(f"  {'─' * 5} {'':>10} {'':>10} {'':>10} {'':>10} {'─' * 10}")
            w(f"  {'Total':>5} {'':>10} {'':>10} {'':>10} {'':>10} {_d(cumulative_cost):>10}")
            w("")

        # What-if: show cost for other models.
        w("  COST COMPARISON (same token usage, different models)")
        w(f"  {'Model':<30} {'Est. Cost':>10}")
        w(f"  {'─' * 30} {'─' * 10}")
        for alt_name in sorted(_MODEL_PRICING):
            alt_p = _MODEL_PRICING[alt_name]
            alt_cost = _calc_cost(
                alt_p,
                billed_input,
                total_output,
                cache_read,
                cache_create,
            )
            marker = " <-- this run" if alt_p is pricing else ""
            w(f"  {alt_name:<30} {_d(alt_cost):>10}{marker}")
        w("")

    # ── Message breakdown by role ─────────────────────────────
    w("-" * 72)
    w("  MESSAGE BREAKDOWN")
    w("-" * 72)
    w("")
    w(
        f"  Turn: {latest['turn']}  |  Messages: {latest['message_count']}  |  "
        f"Elapsed: {latest['elapsed_s']:.0f}s"
    )
    w(f"  Total: {_fmt_ch(total_chars)} chars  (~{_fmt_tok(total_est_tok)} tokens)")
    w("")

    role_stats: dict[str, dict] = {}
    for m in messages:
        r = m["role"]
        if r not in role_stats:
            role_stats[r] = {"count": 0, "chars": 0, "tokens": 0}
        role_stats[r]["count"] += 1
        role_stats[r]["chars"] += m["chars"]
        role_stats[r]["tokens"] += m["estimated_tokens"]

    w(f"  {'Role':<12} {'#':>5} {'Chars':>10} {'Tokens':>10} {'%':>7}  Bar")
    w(f"  {'─' * 12} {'─' * 5} {'─' * 10} {'─' * 10} {'─' * 7}  {'─' * 25}")
    for role, st in sorted(role_stats.items()):
        w(
            f"  {role:<12} {st['count']:>5} {_fmt_ch(st['chars']):>10} "
            f"{_fmt_tok(st['tokens']):>10} "
            f"{_pct(st['chars'], total_chars):>7}  "
            f"{_bar(st['chars'], total_chars, 25)}"
        )
    w("")

    # ── By content type ───────────────────────────────────────
    type_stats: dict[str, dict] = {}
    for m in messages:
        for block in m["blocks"]:
            bt = block["type"]
            key = f"tool_use:{block.get('name', '?')}" if bt == "tool_use" else bt
            if key not in type_stats:
                type_stats[key] = {"count": 0, "chars": 0}
            type_stats[key]["count"] += 1
            type_stats[key]["chars"] += block["chars"]

    w(f"  {'Content Type':<28} {'#':>5} {'Chars':>10} {'%':>7}  Bar")
    w(f"  {'─' * 28} {'─' * 5} {'─' * 10} {'─' * 7}  {'─' * 25}")
    for key, st in sorted(type_stats.items(), key=lambda x: -x[1]["chars"]):
        w(
            f"  {key:<28} {st['count']:>5} {_fmt_ch(st['chars']):>10} "
            f"{_pct(st['chars'], total_chars):>7}  "
            f"{_bar(st['chars'], total_chars, 25)}"
        )
    w("")

    # ── Top 10 largest messages ───────────────────────────────
    sorted_msgs = sorted(messages, key=lambda m: -m["chars"])[:10]
    w("  TOP 10 LARGEST MESSAGES")
    w(f"  {'#':>3} {'Idx':>4} {'Role':<10} {'Types':<24} {'Chars':>10} {'%':>7}  Bar")
    w(f"  {'─' * 3} {'─' * 4} {'─' * 10} {'─' * 24} {'─' * 10} {'─' * 7}  {'─' * 20}")
    for rank, m in enumerate(sorted_msgs, 1):
        types_str = ",".join(m["content_types"])[:23]
        w(
            f"  {rank:>3} {m['index']:>4} {m['role']:<10} {types_str:<24} "
            f"{_fmt_ch(m['chars']):>10} "
            f"{_pct(m['chars'], total_chars):>7}  "
            f"{_bar(m['chars'], total_chars, 20)}"
        )
    w("")

    # ── Tool result distribution ──────────────────────────────
    tool_results = [b for m in messages for b in m["blocks"] if b["type"] == "tool_result"]
    if tool_results:
        total_tr = sum(b["chars"] for b in tool_results)
        w(
            f"  TOOL RESULTS: {len(tool_results)} total  "
            f"| {_fmt_ch(total_tr)} chars  "
            f"| {_pct(total_tr, total_chars)} of history"
        )
        w("")
        buckets = {"<1K": 0, "1-5K": 0, "5-20K": 0, "20-100K": 0, ">100K": 0}
        for tr in tool_results:
            ch = tr["chars"]
            if ch < 1000:
                buckets["<1K"] += 1
            elif ch < 5000:
                buckets["1-5K"] += 1
            elif ch < 20000:
                buckets["5-20K"] += 1
            elif ch < 100000:
                buckets["20-100K"] += 1
            else:
                buckets[">100K"] += 1
        mx_b = max(buckets.values()) if buckets else 1
        for label, count in buckets.items():
            w(f"  {label:>8}: {count:>4}  {_bar(count, mx_b, 25)}")
        w("")

    # ── Cost hotspots ─────────────────────────────────────────
    w("-" * 72)
    w("  COST HOTSPOTS")
    w("-" * 72)
    w("")
    categories: dict[str, int] = {
        "system/task messages": 0,
        "agent text": 0,
        "agent tool_use (code)": 0,
        "tool results (page views)": 0,
        "tool results (other)": 0,
        "turn status/nudges": 0,
    }
    for m in messages:
        if m["role"] == "assistant":
            for block in m["blocks"]:
                if block["type"] == "text":
                    categories["agent text"] += block["chars"]
                elif block["type"] == "tool_use":
                    categories["agent tool_use (code)"] += block["chars"]
        elif m["role"] == "user":
            has_tr = any(b["type"] == "tool_result" for b in m["blocks"])
            if has_tr:
                for block in m["blocks"]:
                    if block["type"] == "tool_result":
                        if block["chars"] > 5000:
                            categories["tool results (page views)"] += block["chars"]
                        else:
                            categories["tool results (other)"] += block["chars"]
            elif m["chars"] < 200:
                categories["turn status/nudges"] += m["chars"]
            else:
                categories["system/task messages"] += m["chars"]

    mx_cat = max(categories.values()) if categories else 1
    for cat, chars in sorted(categories.items(), key=lambda x: -x[1]):
        if chars == 0:
            continue
        w(
            f"  {cat:<30} {_fmt_ch(chars):>10} {_pct(chars, total_chars):>7}  "
            f"{_bar(chars, mx_cat, 25)}"
        )
    w("")

    # 80% coverage
    cum = 0
    for i, m in enumerate(sorted(messages, key=lambda m: -m["chars"]), 1):
        cum += m["chars"]
        if cum >= total_chars * 0.8:
            w(f"  {i} messages ({_pct(i, len(messages))}) account for 80% of the history")
            break
    w("")

    # ── Tool execution summary ───────────────────────────────
    if entries:
        tool_results_entries = [e for e in entries if e.kind == "tool_result"]
        if tool_results_entries:
            # Per-tool-name aggregation.
            tool_agg: dict[str, dict] = {}
            for e in tool_results_entries:
                name = e.data.get("name", "?")
                dur = e.data.get("duration_ms", 0)
                content_len = len(e.data.get("content", ""))
                is_err = e.data.get("is_error", False)
                if name not in tool_agg:
                    tool_agg[name] = {
                        "count": 0,
                        "total_ms": 0,
                        "max_ms": 0,
                        "total_chars": 0,
                        "max_chars": 0,
                        "errors": 0,
                        "durations": [],
                    }
                a = tool_agg[name]
                a["count"] += 1
                a["total_ms"] += dur
                a["max_ms"] = max(a["max_ms"], dur)
                a["total_chars"] += content_len
                a["max_chars"] = max(a["max_chars"], content_len)
                a["durations"].append(dur)
                if is_err:
                    a["errors"] += 1

            total_tool_ms = sum(a["total_ms"] for a in tool_agg.values())
            total_tool_calls = sum(a["count"] for a in tool_agg.values())

            w("-" * 72)
            w("  TOOL EXECUTION SUMMARY")
            w("-" * 72)
            w("")
            w(
                f"  Total tool calls: {total_tool_calls}  |  "
                f"Total tool time: {total_tool_ms / 1000:.1f}s"
            )
            w("")
            w(
                f"  {'Tool':<20} {'#':>4} {'Err':>4} {'Total':>8} "
                f"{'Avg':>8} {'Max':>8} {'OutChars':>10} "
                f"{'%Time':>7}  Bar"
            )
            w(
                f"  {'─' * 20} {'─' * 4} {'─' * 4} {'─' * 8} "
                f"{'─' * 8} {'─' * 8} {'─' * 10} "
                f"{'─' * 7}  {'─' * 20}"
            )
            for name, a in sorted(
                tool_agg.items(),
                key=lambda x: -x[1]["total_ms"],
            ):
                avg_ms = a["total_ms"] / a["count"] if a["count"] else 0
                err_str = str(a["errors"]) if a["errors"] else ""
                w(
                    f"  {name:<20} {a['count']:>4} {err_str:>4} "
                    f"{a['total_ms'] / 1000:>7.1f}s "
                    f"{avg_ms / 1000:>7.1f}s "
                    f"{a['max_ms'] / 1000:>7.1f}s "
                    f"{_fmt_ch(a['total_chars']):>10} "
                    f"{_pct(a['total_ms'], total_tool_ms):>7}  "
                    f"{_bar(a['total_ms'], total_tool_ms, 20)}"
                )
            w("")

            # Top 10 slowest individual tool calls.
            slowest = sorted(
                tool_results_entries,
                key=lambda e: -e.data.get("duration_ms", 0),
            )[:10]
            w("  TOP 10 SLOWEST TOOL CALLS")
            w(f"  {'#':>3} {'Tool':<20} {'Duration':>10} {'Output':>10} {'Error':>6}")
            w(f"  {'─' * 3} {'─' * 20} {'─' * 10} {'─' * 10} {'─' * 6}")
            for rank, e in enumerate(slowest, 1):
                name = e.data.get("name", "?")
                dur = e.data.get("duration_ms", 0)
                out_ch = len(e.data.get("content", ""))
                is_err = "YES" if e.data.get("is_error") else ""
                w(f"  {rank:>3} {name:<20} {dur / 1000:>9.1f}s {_fmt_ch(out_ch):>10} {is_err:>6}")
            w("")

    # ── Top largest content blocks ────────────────────────────
    # Show the biggest individual blocks in the conversation
    # history, so you can see what's bloating context.
    all_blocks: list[dict] = []
    for m in messages:
        role = m["role"]
        for block in m["blocks"]:
            all_blocks.append(
                {
                    "msg_idx": m["index"],
                    "role": role,
                    "type": block.get("type", "?"),
                    "name": block.get("name", ""),
                    "chars": block.get("chars", 0),
                    "tool_use_id": block.get("tool_use_id", ""),
                }
            )
    biggest_blocks = sorted(all_blocks, key=lambda b: -b["chars"])[:15]
    if biggest_blocks:
        w("-" * 72)
        w("  TOP 15 LARGEST CONTENT BLOCKS")
        w("-" * 72)
        w("")
        w(f"  {'#':>3} {'Msg':>4} {'Role':<10} {'Type':<24} {'Chars':>10} {'%':>7}  Bar")
        w(f"  {'─' * 3} {'─' * 4} {'─' * 10} {'─' * 24} {'─' * 10} {'─' * 7}  {'─' * 20}")
        for rank, b in enumerate(biggest_blocks, 1):
            btype = b["type"]
            if btype == "tool_use" and b["name"]:
                btype = f"tool_use:{b['name']}"
            elif btype == "tool_result" and b["tool_use_id"]:
                btype = "tool_result"
            btype = btype[:23]
            w(
                f"  {rank:>3} {b['msg_idx']:>4} {b['role']:<10} {btype:<24} "
                f"{_fmt_ch(b['chars']):>10} "
                f"{_pct(b['chars'], total_chars):>7}  "
                f"{_bar(b['chars'], total_chars, 20)}"
            )
        w("")

    # ── Show page / context management stats ──────────────────
    if entries:
        sp_events = [
            e for e in entries if e.kind == "system" and e.data.get("event") == "show_page_analysis"
        ]
        if sp_events:
            w("-" * 72)
            w("  SHOW PAGE / CONTEXT MANAGEMENT")
            w("-" * 72)
            w("")
            w(f"  show_page calls: {len(sp_events)}")
            w("")
            w(
                f"  {'#':>3} {'Sections':>9} {'Kept':>5} {'Nbr':>4} "
                f"{'Dist':>5} {'FullChars':>10} {'Filtered':>10} "
                f"{'Ratio':>7} {'Sim':>5} {'Var':>4}"
            )
            w(
                f"  {'─' * 3} {'─' * 9} {'─' * 5} {'─' * 4} "
                f"{'─' * 5} {'─' * 10} {'─' * 10} "
                f"{'─' * 7} {'─' * 5} {'─' * 4}"
            )
            for i, e in enumerate(sp_events, 1):
                d = e.data
                total_s = d.get("total_sections", 0)
                kept = d.get("total_sections_kept", 0)
                nbr = d.get("total_sections_neighbor", 0)
                dist = d.get("total_sections_distant", 0)
                orig = d.get("total_page_chars", 0)
                filt = d.get("filtered_page_chars", 0)
                ratio = d.get("compression_ratio", 0)
                sim = d.get("similarity_score", 0)
                var = d.get("variant_used", "?")
                w(
                    f"  {i:>3} {total_s:>9} {kept:>5} {nbr:>4} "
                    f"{dist:>5} {_fmt_ch(orig):>10} {_fmt_ch(filt):>10} "
                    f"{ratio:>6.2f}x {sim:>4.2f} {var:>4}"
                )
            total_orig = sum(e.data.get("total_page_chars", 0) for e in sp_events)
            total_filt = sum(e.data.get("filtered_page_chars", 0) for e in sp_events)
            if total_orig > 0:
                savings = total_orig - total_filt
                w("")
                w(
                    f"  Total page chars produced: {_fmt_ch(total_orig)}  "
                    f"| After filtering: {_fmt_ch(total_filt)}  "
                    f"| Saved: {_fmt_ch(savings)} "
                    f"({savings / total_orig * 100:.0f}%)"
                )
            w("")

    # ── Heatmap ───────────────────────────────────────────────
    if messages:
        w("-" * 72)
        w("  MESSAGE SIZE HEATMAP")
        w("-" * 72)
        w("  Each cell = 1 message. Brightness = relative size.")
        w("  Row = role (U=user, A=assistant). Column = message index.")
        w("")
        max_mc = max(m["chars"] for m in messages)
        heat = " \u2591\u2592\u2593\u2588"
        for role, label in [("user", "U"), ("assistant", "A")]:
            cells = []
            for m in messages:
                if m["role"] == role:
                    level = int((m["chars"] / max_mc) * 4) if max_mc > 0 else 0
                    cells.append(heat[min(level, 4)])
                else:
                    cells.append(" ")
            line = "".join(cells)
            for start in range(0, len(line), 70):
                chunk = line[start : start + 70]
                if chunk.strip():
                    w(f"  {label} {start:>4} |{chunk}|")
        w("")
        w("  Legend: ' '=other role  \u2591=small  \u2592=medium  \u2593=large  \u2588=largest")
        w("")

    return "\n".join(o) + "\n"
