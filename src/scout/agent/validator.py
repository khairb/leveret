"""Validation agent for scraping script output.

Runs a separate LLM that inspects script output and decides whether
it satisfies the original task.  Replaces manual ``ask_user_approval()``
with the same ``(approved, feedback)`` interface.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .llm import LLMConfig, call_llm

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════


@dataclass
class AttemptRecord:
    """History of a previous script attempt, for context on retries."""

    attempt_number: int
    script: str
    stdout_sample: str  # first 3000 chars of stdout
    stderr: str
    returncode: int
    rejection_feedback: str


# ═══════════════════════════════════════════════════════════════
#  Tool Schemas
# ═══════════════════════════════════════════════════════════════

VALIDATOR_TOOL_SCHEMAS = [
    {
        "name": "search_output",
        "description": (
            "Search the script's stdout for a keyword or phrase. "
            "Returns matching lines with surrounding context and line numbers. "
            "Use this to verify specific requirements (e.g., search for "
            "'[page' to check pagination progress)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search term or phrase to find in output",
                },
                "context_lines": {
                    "type": "integer",
                    "description": ("Lines of context before and after each match (default: 3)"),
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "view_lines",
        "description": (
            "View a specific range of lines from the script's stdout. "
            "Line numbers are 1-indexed. Max 200 lines per call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_line": {
                    "type": "integer",
                    "description": "First line number (1-indexed)",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last line number (1-indexed)",
                },
            },
            "required": ["start_line", "end_line"],
        },
    },
    {
        "name": "decide",
        "description": (
            "Make your final approval or rejection decision. "
            "This ends the validation. You MUST call this tool to finish."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "approved": {
                    "type": "boolean",
                    "description": ("True if the output is correct, False if not"),
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation of your decision",
                },
                "feedback": {
                    "type": "string",
                    "description": (
                        "If rejected: specific, actionable feedback for the "
                        "scraping agent explaining what went wrong and what "
                        "to fix. Reference line numbers when relevant. "
                        "If approved: empty string."
                    ),
                    "default": "",
                },
            },
            "required": ["approved", "reasoning"],
        },
    },
]

# ═══════════════════════════════════════════════════════════════
#  System Prompt
# ═══════════════════════════════════════════════════════════════

_VALIDATOR_SYSTEM_PROMPT = """\
You are a senior data quality engineer reviewing web scraping output. You \
think like someone who will actually use this data downstream — you care \
about whether this data genuinely fulfills what the user asked for.

You are experienced and pragmatic. You maintain high standards, but you \
also know that web scraping operates against live, unpredictable websites. \
Pages change between exploration and execution. Displayed counts include \
items that are hidden, filtered, or inaccessible. Server-side limits cap \
what is actually served. A script that thoroughly extracts everything \
available is doing its job — even if the result does not perfectly match \
what the page once claimed to have.

## Your Core Judgment

Before every decision, ask yourself two questions:

1. **Is this output genuinely useful for what the user wanted?** \
The user asked for data. Did they get it? Is it complete enough to use? \
Are the fields real and meaningful?

2. **If I reject, will the next attempt realistically be better?** \
If the scraping agent has tried multiple times and the output is stable — \
same item count, same structure, same minor shortfall — then rejecting \
again will not produce a different result. A website can only give what it \
has. Repeating the same rejection is waste, not quality control.

## What You Receive

1. The original task — the user's actual intent in their own words
2. Success criteria — requirements derived from the exploration phase \
(page counts, item counts, available fields, navigation prerequisites). \
Note: quantitative estimates in these criteria are based on what the page \
displayed during exploration — they are informed expectations, not exact \
thresholds.
3. The generated script — to understand what the script was supposed to do
4. Script stdout, stderr, and exit code
5. A sampled overview of the output with line numbers
6. Previous attempt history (if this is a retry) — including the scripts \
used, outputs produced, and rejection feedback given

## Your Reasoning Process

### Step 1 — Understand the User's Intent

Read the task and success criteria. Before anything else, answer: \
**What does success look like for this user?**

Write down:
- The scope: how many pages, categories, or items are expected
- What "complete" means for each required field
- What would genuinely satisfy this request vs. what would disappoint

Be literal about the user's language. "All pages" means every page. \
"Each item" means every item. A requested field means the full value — \
not a truncated fragment.

### Step 2 — Verify Scope Coverage

This is the most important check.

- What is the total item count in the output? (Search for a count in \
logs, or estimate by counting JSON objects.)
- How many pages or sections were actually covered?
- Does the coverage match the expected scope?

Use `search_output` to find pagination markers and `view_lines` to inspect \
items from different sections. If the task requires 20 pages, look for \
evidence of 20 pages worth of data.

**Important:** Success criteria may state expected quantities based on what \
the page displayed during exploration (e.g., "~200 items based on page \
header"). These are estimates derived from the live page, not exact \
thresholds. Websites routinely show counts that differ from what is \
actually extractable — items get removed between exploration and execution, \
lazy-loaded counts include hidden or filtered items, server-side caps limit \
results. A result of 195 when 200 was expected is normal variance, not a \
failure. Judge by whether the extraction was thorough — not by whether it \
hit an exact number.

### Step 3 — Verify Field Completeness and Data Integrity

**NEVER reject because values appear truncated in the overview.** The \
overview is sampled and lines are cut off for display. This is a display \
limitation, not a data problem. If a value looks cut off, use `view_lines` \
to see the full raw line. If the value is complete there, there is no \
issue. Do not list display truncation as a failure reason — it is not one.

For each required field:

1. **Use `view_lines` to examine actual items** from the start, middle, \
and end of the output. Do not just search for empty strings — read the \
actual values. Ask: are they complete? Do they end naturally, or cut off \
mid-sentence?

2. **Search for empty and null values** with `search_output` — look for \
`"field": ""` and `"field": null` — but understand this is a minimum \
check, not a completeness check.

3. **Assess data quality**: Do values make sense for what they represent? \
Are numeric fields actually numbers in a reasonable range? Are text fields \
substantive, or just a few words or a fragment? Are identifier fields \
(URLs, IDs) well-formed?

### Step 4 — Assess Attempt History (Critical for Retries)

If previous attempts exist, this step is **mandatory before deciding**:

1. **Compare outputs across attempts.** Are item counts stable? Are the \
same fields present? If multiple different scripts all produce \
approximately the same count, the website has that many extractable items.

2. **Compare scripts across attempts.** Did the agent try meaningfully \
different approaches? If the agent rewrote its pagination logic or \
selector strategy and still got the same count, the count is correct — \
the website is the constraint.

3. **Identify the pattern:**
   - **Script bug:** Outputs vary wildly between attempts, errors occur, \
the agent keeps making the same coding mistake → reject with a specific fix.
   - **Website limitation:** Outputs are stable across different \
approaches, close to the target, no errors → the script is correct, \
the expectation was off. Approve if the data is genuinely useful.
   - **Diminishing returns:** The agent has tried multiple reasonable \
approaches, the output is consistent → approve if the data serves the \
user's purpose.

### Step 5 — Decide

**APPROVE when:**
- The output covers the requested scope (with reasonable tolerance for \
page-reported estimates)
- All required fields are present and complete — not truncated, not empty
- Values are well-formed and make sense for what they represent
- OR: output is stable across multiple attempts and represents the best \
achievable result from this website

**REJECT when:**
- Scope is significantly incomplete (e.g., 50 items when 200 were \
expected — not 195 when 200 were expected)
- Required fields are missing, empty, or truncated in a significant \
portion of items
- Script crashed or produced no useful output
- There is a clear, fixable bug and you can articulate a specific change \
that would produce a meaningfully better result

**Never reject when:**
- The shortfall is minor and has been stable across attempts
- You cannot describe a concrete fix that would improve the outcome
- Rejecting would produce the same result again

## Rejection Feedback

When rejecting, your `decide` feedback must be direct and actionable. \
The scraping agent reads this and uses it to fix the script.

Structure it as:

```
WHAT FAILED:
1. [Primary failure — specific, with evidence: line numbers, counts, \
examples]
2. [Secondary issues, if any]

WHAT TO FIX:
1. [Specific instruction for the scraping agent — what code change is \
needed]
2. [Specific instruction]
```

Be precise. "Only 48 items extracted but expected ~300 (20 pages × ~15 \
items/page) — script stops after page 3 despite more pages existing" is \
useful feedback. "Fix pagination" is not.

## Tools

- `search_output(query, context_lines)` — search stdout for a keyword. \
Use aggressively: verify item counts, find pagination evidence, check \
for empty or truncated fields.
- `view_lines(start_line, end_line)` — inspect specific lines. Use to \
examine actual item data from start, middle, and end of output.
- `decide(approved, reasoning, feedback)` — your final verdict. You MUST \
call this to finish.

You MUST call `decide`. A text response without a tool call will be \
prompted to decide immediately.\
"""


# ═══════════════════════════════════════════════════════════════
#  Output Sampling
# ═══════════════════════════════════════════════════════════════


def _sample_output(
    stdout: str,
    num_slices: int = 7,
    lines_per_slice: int = 15,
) -> str:
    """Create a sampled overview of potentially large output.

    For output ≤ ``num_slices * lines_per_slice`` lines, returns the
    full output with line numbers.  For larger output, returns evenly-
    spaced slices with gap markers between them.
    """
    lines = stdout.split("\n")
    total = len(lines)
    threshold = num_slices * lines_per_slice

    if total <= threshold:
        # Small output — show everything.
        numbered = [f"[L{i + 1}] {line}" for i, line in enumerate(lines)]
        return "\n".join(numbered)

    # Large output — sample slices.
    parts: list[str] = []
    step = max(1, (total - lines_per_slice) // (num_slices - 1))
    prev_end = -1

    for s in range(num_slices):
        start = s * step
        if start >= total:
            break
        end = min(start + lines_per_slice, total)

        # Gap marker.
        if prev_end >= 0 and start > prev_end:
            parts.append(f"--- lines {prev_end + 1}-{start} omitted ({start - prev_end} lines) ---")

        # Slice with line numbers.
        for i in range(start, end):
            parts.append(f"[L{i + 1}] {lines[i]}")

        prev_end = end

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
#  Tool Execution
# ═══════════════════════════════════════════════════════════════


def _execute_validator_tool(
    name: str,
    arguments: dict[str, Any],
    stdout_lines: list[str],
) -> str:
    """Execute a validator tool against the stored stdout lines."""

    if name == "search_output":
        query = arguments.get("query", "")
        context = int(arguments.get("context_lines", 3))
        if not query:
            return "Error: query is required."

        matches: list[str] = []
        match_count = 0
        total = len(stdout_lines)

        for i, line in enumerate(stdout_lines):
            if query.lower() in line.lower():
                match_count += 1
                if match_count > 50:
                    continue  # count but don't collect
                start = max(0, i - context)
                end = min(total, i + context + 1)
                if matches:
                    matches.append("---")
                for j in range(start, end):
                    marker = " >> " if j == i else "    "
                    matches.append(f"{marker}[L{j + 1}] {stdout_lines[j]}")

        if match_count == 0:
            return f"No matches found for '{query}'."
        header = f"Found {match_count} matches for '{query}'"
        if match_count > 50:
            header += " (showing first 50)"
        return header + ":\n\n" + "\n".join(matches)

    if name == "view_lines":
        start = int(arguments.get("start_line", 1))
        end = int(arguments.get("end_line", start + 50))
        total = len(stdout_lines)

        # Clamp and validate.
        start = max(1, start)
        end = min(total, end)
        if end - start > 200:
            end = start + 200

        if start > total:
            return f"Output only has {total} lines."

        result_lines = [f"[L{i + 1}] {stdout_lines[i]}" for i in range(start - 1, end)]
        return f"Lines {start}-{end} of {total}:\n\n" + "\n".join(result_lines)

    return f"Unknown tool: {name}"


# ═══════════════════════════════════════════════════════════════
#  Attempt History
# ═══════════════════════════════════════════════════════════════


def _build_attempt_history(
    history: list[AttemptRecord],
    current_attempt: int,
    max_attempts: int,
) -> str:
    """Format previous attempts for the validator's context."""
    parts = [
        f"## Previous Attempts (you are reviewing attempt {current_attempt} of {max_attempts})\n",
    ]

    for rec in history:
        # Include a truncated version of the script so the validator
        # can see what approaches were tried.
        script_preview = rec.script
        if len(script_preview) > 1500:
            script_preview = (
                script_preview[:750] + "\n# ... (truncated) ...\n" + script_preview[-750:]
            )

        sample_lines = rec.stdout_sample.split("\n")[:15]
        sample_text = "\n".join(sample_lines)
        parts.append(
            f"### Attempt {rec.attempt_number}\n"
            f"**Script used:**\n```python\n{script_preview}\n```\n"
            f"**Rejection feedback:** {rec.rejection_feedback}\n"
            f"**Exit code:** {rec.returncode}\n"
            f"**Output sample (first 15 lines):**\n"
            f"```\n{sample_text}\n```\n"
        )

    # Add context about what the attempt history means.
    parts.append(
        f"**This is attempt {current_attempt} of {max_attempts}.**\n\n"
        f"The scripts and outputs above are from real execution runs "
        f"against the live website. Use them to understand whether "
        f"shortfalls are caused by the script or by the website "
        f"itself.\n\n"
        f"For example, if three different pagination strategies all "
        f"yield ~199 items when 200 was expected, the site simply "
        f"serves fewer items than its header claims — no code change "
        f'will produce 200. Or if a field like "rating" is null on '
        f"~10% of items across every attempt, those listings "
        f"genuinely have no rating on the page — the script is "
        f"extracting correctly, the data just does not exist for "
        f'every item. Or if a "description" field always cuts off '
        f"at exactly 150 characters, the website itself truncates it "
        f"in the listing view and the full text only lives on the "
        f"detail page.\n\n"
        f"These are just examples — the real pattern to look for is "
        f"consistency. When different scripts converge on the same "
        f"result, the website is the constraint. When outputs vary "
        f"wildly or the same error keeps recurring, that is a real "
        f"script problem that a rejection can help fix.\n"
    )
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
#  User Message Builder
# ═══════════════════════════════════════════════════════════════


def _build_user_message(
    task: str,
    script: str,
    stdout: str,
    stderr: str,
    returncode: int,
    stdout_lines: list[str],
    attempt_history: list[AttemptRecord] | None,
    requirements: str | None = None,
    checkpoints_summary: str | None = None,
    attempt_number: int = 1,
    max_attempts: int = 10,
) -> str:
    """Build the initial user message for the validator."""
    total_lines = len(stdout_lines)
    total_bytes = len(stdout.encode("utf-8"))
    sampled = _sample_output(stdout)

    parts = [
        f"## Original Task\n\n{task}\n",
    ]

    if requirements:
        parts.append(f"## Success Criteria\n\n{requirements}\n")

    parts.extend(
        [
            f"## Generated Script (attempt {attempt_number} of "
            f"{max_attempts})\n\n```python\n{script}\n```\n",
            f"## Execution Result\n\n**Exit code:** {returncode}\n",
        ]
    )

    if stderr.strip():
        stderr_display = stderr.strip()[:2000]
        parts.append(f"**Stderr:**\n```\n{stderr_display}\n```\n")

    parts.append(f"## Output Overview ({total_lines} lines, {total_bytes:,} bytes)\n\n{sampled}\n")

    if checkpoints_summary:
        parts.append(f"\n{checkpoints_summary}\n")

    if attempt_history:
        parts.append(
            _build_attempt_history(
                attempt_history,
                attempt_number,
                max_attempts,
            )
        )

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
#  Validator Trace
# ═══════════════════════════════════════════════════════════════


class _ValidatorTrace:
    """Writes a markdown trace of the validator's conversation."""

    def __init__(self, trace_dir: Path | None) -> None:
        self._path: Path | None = None
        self._start_time = time.time()
        if trace_dir and trace_dir.exists():
            self._path = trace_dir / "validator_trace.md"
            self._path.write_text(
                f"# Validator Trace — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
                encoding="utf-8",
            )

    def _elapsed(self) -> str:
        return f"{time.time() - self._start_time:.1f}s"

    def log_user_message(self, content: str) -> None:
        self._append(f"## [{self._elapsed()}] User Message\n\n````\n{content[:5000]}\n````\n\n")

    def log_assistant(self, content_blocks: list[dict]) -> None:
        parts = [f"## [{self._elapsed()}] Assistant\n\n"]
        for block in content_blocks:
            if block.get("type") == "text":
                parts.append(f"{block['text']}\n\n")
            elif block.get("type") == "tool_use":
                name = block.get("name", "?")
                args = block.get("input", {})
                parts.append(
                    f"**Tool call: `{name}`**\n\n"
                    f"```json\n{json.dumps(args, indent=2, default=str)}\n"
                    f"```\n\n"
                )
        self._append("".join(parts))

    def log_tool_result(self, name: str, content: str) -> None:
        preview = content[:3000]
        if len(content) > 3000:
            preview += f"\n... ({len(content) - 3000} more chars)"
        self._append(f"### [{self._elapsed()}] Tool result: `{name}`\n\n````\n{preview}\n````\n\n")

    def log_decision(
        self,
        approved: bool,
        reasoning: str,
        feedback: str,
    ) -> None:
        status = "APPROVED" if approved else "REJECTED"
        self._append(
            f"## [{self._elapsed()}] Decision: **{status}**\n\n"
            f"**Reasoning:** {reasoning}\n\n"
            f"**Feedback:** {feedback or '(none)'}\n\n"
        )

    def _append(self, text: str) -> None:
        if self._path:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(text)


# ═══════════════════════════════════════════════════════════════
#  Main Validation Function
# ═══════════════════════════════════════════════════════════════


async def validate_output(
    *,
    task: str,
    script: str,
    stdout: str,
    stderr: str,
    returncode: int,
    llm_config: LLMConfig,
    attempt_history: list[AttemptRecord] | None = None,
    requirements: str | None = None,
    max_turns: int = 20,
    trace_dir: Path | None = None,
    checkpoints_summary: str | None = None,
    attempt_number: int = 1,
    max_attempts: int = 10,
) -> tuple[bool, str]:
    """Run the validation agent and return ``(approved, feedback)``.

    Drop-in replacement for ``console.ask_user_approval()``.

    Args:
        task: The original scraping task description.
        script: The generated Python script.
        stdout: Script's stdout output.
        stderr: Script's stderr output.
        returncode: Script's exit code.
        llm_config: LLM configuration for the validator.
        attempt_history: Previous attempts (for retry awareness).
        requirements: Concrete success criteria from the requirements
            agent (generated once from exploration history).
        max_turns: Maximum tool-call turns (default 20).
        trace_dir: Directory to write ``validator_trace.md`` into.
        attempt_number: Current attempt number (1-indexed).
        max_attempts: Total allowed attempts.

    Returns:
        ``(approved, feedback)`` — feedback is empty if approved.
    """
    stdout_lines = stdout.split("\n")
    trace = _ValidatorTrace(trace_dir)

    user_message = _build_user_message(
        task,
        script,
        stdout,
        stderr,
        returncode,
        stdout_lines,
        attempt_history,
        requirements,
        checkpoints_summary,
        attempt_number=attempt_number,
        max_attempts=max_attempts,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_message},
    ]
    trace.log_user_message(user_message)

    for turn in range(max_turns):
        response = await call_llm(
            llm_config,
            system=_VALIDATOR_SYSTEM_PROMPT,
            messages=messages,
            tools=VALIDATOR_TOOL_SCHEMAS,
        )

        content_blocks = response.content
        tool_uses = [b for b in content_blocks if b.type == "tool_use"]

        # Serialize the assistant message.
        serialized = []
        for block in content_blocks:
            if block.type == "text":
                serialized.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                serialized.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
        messages.append({"role": "assistant", "content": serialized})
        trace.log_assistant(serialized)

        if not tool_uses:
            # No tool calls — nudge to use decide.
            nudge = (
                "You must call the `decide` tool to finish. "
                "Please make your approval or rejection decision now."
            )
            messages.append({"role": "user", "content": nudge})
            trace.log_user_message(nudge)
            continue

        # Process tool calls.
        tool_results: list[dict[str, Any]] = []
        for block in tool_uses:
            if block.name == "decide":
                approved = block.input.get("approved", False)
                reasoning = block.input.get("reasoning", "")
                feedback = block.input.get("feedback", "")

                logger.info(
                    "Validator decided: %s — %s",
                    "APPROVED" if approved else "REJECTED",
                    reasoning[:200],
                )
                trace.log_decision(approved, reasoning, feedback)

                if approved:
                    return True, ""
                return False, feedback if feedback else reasoning

            # Execute search_output or view_lines.
            result = _execute_validator_tool(
                block.name,
                block.input,
                stdout_lines,
            )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
            )
            trace.log_tool_result(block.name, result)

        # Budget warning on penultimate turn.
        if turn == max_turns - 2 and tool_results:
            for tr in tool_results:
                tr["content"] = (
                    str(tr["content"]) + "\n\n⚠️ You have 1 turn remaining. "
                    "You MUST call the `decide` tool on your next turn."
                )

        # Add tool results to conversation.
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    # Exhausted max turns — force one final decision-only call.
    logger.warning(
        "Validator exhausted %d turns. Forcing final decision call.",
        max_turns,
    )
    force_msg = (
        "You have used all your investigation turns. You MUST call the "
        "`decide` tool NOW with your best judgment based on everything "
        "you have seen so far. Do not call any other tool."
    )
    messages.append({"role": "user", "content": force_msg})
    trace.log_user_message(force_msg)

    response = await call_llm(
        llm_config,
        system=_VALIDATOR_SYSTEM_PROMPT,
        messages=messages,
        tools=VALIDATOR_TOOL_SCHEMAS,
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "decide":
            approved = block.input.get("approved", False)
            reasoning = block.input.get("reasoning", "")
            feedback = block.input.get("feedback", "")
            trace.log_decision(approved, reasoning, feedback)
            if approved:
                return True, ""
            return False, feedback if feedback else reasoning

    # Absolute fallback — should never reach here.
    logger.error("Validator failed to decide even after forced call.")
    trace.log_decision(True, "Fallback: validator failed to decide", "")
    return True, ""
