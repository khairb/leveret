"""History compression summarizer for the scraping agent.

When the conversation context approaches a model-specific token threshold,
this module compresses older exploration history into a dense, sequential
summary that preserves all actionable findings.

The summarizer is a separate, focused LLM call (using Haiku for cost and
speed) that receives only the compressible message window — not the entire
conversation.  Its output is a turn-by-turn extraction of everything the
agent discovered, quoted verbatim where possible.

See ``docs/HISTORY_COMPRESSION_SPEC.md`` for the full design specification
and rationale behind each decision.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .llm import LLMConfig, call_llm

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  Summarizer configuration
# ═══════════════════════════════════════════════════════════════

_SUMMARIZER_CONFIG = LLMConfig(
    model="anthropic:claude-haiku-4-5",
    max_tokens=8192,
    temperature=0.0,
)

# Maximum tokens per summarizer call before splitting.
_SPLIT_THRESHOLD = 20_000
_THREE_WAY_SPLIT_THRESHOLD = 40_000

# ═══════════════════════════════════════════════════════════════
#  Summarizer system prompt
# ═══════════════════════════════════════════════════════════════

_SUMMARIZER_SYSTEM_PROMPT = """\
You are a context compression engine for a web scraping agent. The agent \
explores websites interactively — navigating pages, inspecting DOM \
elements, writing and testing code, dealing with popups, pagination, \
anti-bot measures, and more. Your job is to produce a dense, accurate, \
sequential summary of what the agent did and discovered.

Your summary will REPLACE the original messages in the agent's context. \
The agent will rely on your summary to write its final scraping script. \
If you omit a finding, the agent will not know about it. If you \
hallucinate a detail, the agent will write broken code. Accuracy is \
paramount.

## Output Format

Write a **turn-by-turn sequential summary**. For each turn that produced \
findings, write a `**Turn N**` entry describing what happened and what \
was discovered. The turn numbers must match the original conversation.

For turns with no new findings (e.g., the agent re-checked something \
already known), write a single line:
`**Turn N** — [brief action, no new findings]`

## What to Capture

You do not know in advance what the agent encountered. Scraping sessions \
vary enormously. Adapt to whatever each turn contains. Here is the full \
range of things you may need to capture — include ALL that apply:

### Page State & Visual Observations
- What the agent SAW on the page — section IDs (e.g. [nav-header], \
[results-grid]), page layout, what content appeared where
- How many items/cards/rows were visible and their general structure
- What changed after an action — new sections appeared, content \
updated, overlays dismissed, URL changed
- Viewport or responsive behavior — elements visible only at certain \
widths, mobile vs desktop layouts
- Loading states — spinners, skeleton screens, lazy-loaded content
- Error states shown on the page — 404 pages, "no results" messages, \
CAPTCHA challenges

### DOM Structure & Selectors
- CSS selectors, XPath expressions — quote EXACTLY as discovered
- `data-testid`, `aria-label`, `role`, `id`, `name` attributes — \
verbatim
- Full HTML tag structures with all attributes when the agent inspected \
them: `<div class="listing" data-id="123" role="listitem">`
- Whether class names are stable or hashed/generated (and which to use)
- Tag hierarchy and nesting relevant to selector construction
- Shadow DOM or iframe boundaries encountered

### Code & Execution
- Python code snippets the agent wrote that WORKED — quote verbatim
- JavaScript evaluated via `page.evaluate(...)` — quote verbatim
- Extraction loops, data processing logic, helper functions
- What code FAILED and the exact error message
- Timing: `await page.wait_for_selector(...)`, explicit delays, \
`wait_until` parameters that were needed

### Navigation & Interaction
- Click sequences — what was clicked, in what order, what happened after
- Form fills — what inputs, what values, what submit mechanism
- Scroll behavior — infinite scroll detection, scroll-to-load patterns
- Pagination — mechanism (URL offset, click next, load more, cursor), \
stop conditions, total page/item counts
- Dropdown/modal/tab interactions — open/select/close sequences
- Consent/cookie popups — how they were dismissed, what selectors used
- Authentication flows — login steps, session handling, redirects

### Data & Extraction
- Data structure observed — what fields exist, their format, edge cases
- Price/date/number formats and locale variations
- Optional vs required fields — which items have/lack certain data
- Data cleaning needed — stripping currency symbols, parsing dates, \
handling nulls
- Sample extracted values that illustrate the format
- JSON-LD, `<script>` embedded data, meta tags discovered

### Network & API
- XHR/fetch endpoints discovered — full URLs with parameters
- Request headers required (auth tokens, custom headers)
- Response JSON structure
- Rate limiting or anti-bot responses observed
- Whether API approach was chosen over DOM scraping (and why)

### Problems & Solutions
- What went wrong and the EXACT error or symptom
- What the agent tried to fix it
- What ultimately worked (or didn't)
- Workarounds for anti-bot, dynamic content, timing issues
- Elements that behave differently on fresh pages vs explored pages

### Decisions & Strategy
- Explicit decisions the agent made: "use class selectors because IDs \
are dynamic", "scrape via pagination not API because auth headers needed"
- Strategy for the final script — navigation order, extraction approach, \
error handling approach
- Things the agent noted for the final script specifically

## Critical Rules

1. **ONLY report information explicitly present in the messages.** \
NEVER infer, reconstruct, or complete partial information. If a \
selector is partially visible or truncated, quote what is visible and \
write "[partial]" or "[truncated]".

2. **Quote verbatim.** Selectors, tags, URLs, code snippets, error \
messages, attribute values — character for character. Do not "fix", \
"clean up", or normalize quoted content. If the agent wrote \
`div.price-amount`, write `div.price-amount` — not `.price-amount` \
and not `div .price-amount`.

3. **Do not synthesize across turns.** Each turn's entry stands alone. \
If turn 8 says "the selector from earlier doesn't work here", report \
what turn 8 says — do not go find the selector from the earlier turn \
and substitute it. Cross-turn connections are the agent's job.

4. **Include negative findings.** "This does NOT work" is as important \
as "this works". Failed approaches prevent the agent from repeating \
mistakes.

5. **Preserve section IDs exactly.** Section IDs like [nav-header] or \
[item-3-div-listing] are used by zoom_section. Write them exactly as \
they appear, in square brackets.

6. **When in doubt, include it.** A slightly longer summary that quotes \
an extra code block is better than a terse summary that omits a \
critical selector. Err on the side of completeness.

7. **Do not editorialize.** Do not add commentary like "this is a good \
approach" or "the agent should have done X". Report facts only.

8. **Skip mechanical noise.** Do not include "Executed in 150ms" \
boilerplate, tool invocation mechanics, or repeated unchanged page \
views. Focus on findings, not process.

9. **Report what the page showed.** When the agent viewed a page, \
note what was on it — sections, structure, key content. When the page \
changed after an action, note what changed. The visual/structural \
state of the page at each step is critical context for script writing.
"""

_SUMMARIZER_USER_TEMPLATE = """\
## Task Context

The agent is working on this task:
{task_description}

## Messages to Summarize (Turns {start_turn}-{end_turn})

Below are the conversation messages from the agent's exploration. \
Produce a sequential turn-by-turn summary capturing all actionable \
findings.

---

{messages_text}
"""

_SPLIT_FIRST_HALF_NOTE = """\

**Note:** This covers the first portion of the exploration. Another \
summarizer covers the continuation. Include all findings from these \
turns — do not defer anything to "later"."""

_SPLIT_SECOND_HALF_NOTE = """\

**Note:** This covers the second portion of the exploration. The first \
turn overlaps with a previous summarizer's coverage — include it for \
context continuity but avoid duplicating findings already covered for \
that turn if they are clearly redundant (same selector, same code, \
same observation). When in doubt, include rather than omit."""


# ═══════════════════════════════════════════════════════════════
#  Message serialization for the summarizer
# ═══════════════════════════════════════════════════════════════


def _serialize_messages_for_summarizer(
    messages: list[dict],
    start_turn: int,
) -> str:
    """Convert conversation messages into readable text for the summarizer.

    Produces a human-readable transcript that preserves:
    - Role labels (ASSISTANT / USER / TOOL RESULT)
    - Tool call names and arguments
    - Tool result content
    - Agent reasoning text

    Turn numbers are inferred from assistant/user message pairs.
    """
    parts: list[str] = []
    turn = start_turn
    msg_idx = 0

    while msg_idx < len(messages):
        msg = messages[msg_idx]
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if role == "assistant":
            parts.append(f"--- Turn {turn} ---")
            parts.append("")

            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "text" and block.get("text", "").strip():
                        parts.append(f"ASSISTANT (reasoning):")
                        parts.append(block["text"].strip())
                        parts.append("")
                    elif block.get("type") == "tool_use":
                        name = block.get("name", "?")
                        args = block.get("input", {})
                        # Show tool call concisely
                        if name == "python":
                            code = args.get("code", "")
                            parts.append(f"TOOL CALL: python")
                            parts.append(f"```python\n{code}\n```")
                        else:
                            parts.append(f"TOOL CALL: {name}({args})")
                        parts.append("")
            elif isinstance(content, str) and content.strip():
                parts.append(f"ASSISTANT:")
                parts.append(content.strip())
                parts.append("")

        elif role == "user":
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "tool_result":
                        tool_content = block.get("content", "")
                        is_error = block.get("is_error", False)
                        label = "TOOL ERROR" if is_error else "TOOL RESULT"
                        if isinstance(tool_content, str) and tool_content.strip():
                            # Truncate very large tool results for the summarizer
                            text = tool_content.strip()
                            if len(text) > 8000:
                                text = (
                                    text[:4000]
                                    + f"\n\n[... {len(tool_content) - 8000} chars omitted ...]\n\n"
                                    + text[-4000:]
                                )
                            parts.append(f"{label}:")
                            parts.append(text)
                            parts.append("")
                    elif block.get("type") == "text":
                        text = block.get("text", "").strip()
                        if text:
                            parts.append(f"USER: {text}")
                            parts.append("")
            elif isinstance(content, str) and content.strip():
                # Plain text user message (status messages, system notes)
                text = content.strip()
                # Skip turn status messages — pure noise for summarizer
                if text.startswith("[Turn ") and "remaining" in text:
                    msg_idx += 1
                    continue
                parts.append(f"USER: {text}")
                parts.append("")

            # Advance turn counter after processing the user response
            # (each turn = one assistant + one user message pair)
            if msg_idx > 0 and messages[msg_idx - 1].get("role") == "assistant":
                turn += 1

        msg_idx += 1

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
#  Token estimation
# ═══════════════════════════════════════════════════════════════


def estimate_message_tokens(messages: list[dict]) -> int:
    """Estimate token count from message content (~4 chars per token)."""
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block.get("content"), str):
                    total_chars += len(block["content"])
                if block.get("type") == "text":
                    total_chars += len(block.get("text", ""))
                if block.get("type") == "tool_use":
                    total_chars += len(str(block.get("input", {})))
    return total_chars // 4


# ═══════════════════════════════════════════════════════════════
#  Turn number extraction
# ═══════════════════════════════════════════════════════════════


def _infer_turn_range(
    messages: list[dict],
    fallback_start: int = 1,
) -> tuple[int, int]:
    """Infer the turn range from messages by counting assistant messages.

    Returns (start_turn, end_turn).
    """
    import re

    # Try to extract turn numbers from status messages like "[Turn 5/75..."
    turn_re = re.compile(r"\[Turn (\d+)/")
    found_turns: list[int] = []

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            m = turn_re.search(content)
            if m:
                found_turns.append(int(m.group(1)))

    if found_turns:
        return min(found_turns), max(found_turns)

    # Fallback: count assistant messages as turns
    assistant_count = sum(
        1 for msg in messages if msg.get("role") == "assistant"
    )
    return fallback_start, fallback_start + max(assistant_count - 1, 0)


# ═══════════════════════════════════════════════════════════════
#  Core summarizer call
# ═══════════════════════════════════════════════════════════════


async def _call_summarizer(
    messages: list[dict],
    task_description: str,
    start_turn: int,
    end_turn: int,
    *,
    split_note: str = "",
) -> str:
    """Run a single summarizer call and return the summary text."""
    messages_text = _serialize_messages_for_summarizer(messages, start_turn)

    user_prompt = _SUMMARIZER_USER_TEMPLATE.format(
        task_description=task_description,
        start_turn=start_turn,
        end_turn=end_turn,
        messages_text=messages_text,
    )
    if split_note:
        user_prompt += split_note

    response = await call_llm(
        _SUMMARIZER_CONFIG,
        system=_SUMMARIZER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    # Extract text from response
    text_parts = [
        block.text for block in response.content
        if block.type == "text" and block.text.strip()
    ]
    return "\n".join(text_parts).strip()


# ═══════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════


def _find_turn_midpoint(messages: list[dict]) -> int:
    """Find the message index closest to the midpoint of assistant turns.

    Returns the index of the user message that follows the middle
    assistant message — so the first half ends on a complete
    interaction (assistant + user pair).
    """
    assistant_indices = [
        i for i, msg in enumerate(messages)
        if msg.get("role") == "assistant"
    ]
    if len(assistant_indices) < 2:
        return len(messages) // 2

    mid_assistant = assistant_indices[len(assistant_indices) // 2]

    # Find the user message after this assistant message (completes the pair)
    for i in range(mid_assistant + 1, len(messages)):
        if messages[i].get("role") == "user":
            return i + 1  # exclusive end of first half

    return mid_assistant + 1


async def run_summarizer(
    messages: list[dict],
    task_description: str,
    fallback_start_turn: int = 1,
) -> str:
    """Compress a sequence of messages into a dense sequential summary.

    Automatically splits into parallel calls if the message window
    exceeds the reliability threshold for a single Haiku call.

    Args:
        messages: The compressible message window.
        task_description: The original task (for context).
        fallback_start_turn: Turn number to use if detection fails.

    Returns:
        The formatted summary text (without the header/marker —
        that is added by the caller).
    """
    start_turn, end_turn = _infer_turn_range(messages, fallback_start_turn)
    token_estimate = estimate_message_tokens(messages)

    logger.info(
        "Summarizer: %d messages, ~%d tokens estimated, turns %d-%d",
        len(messages), token_estimate, start_turn, end_turn,
    )

    if token_estimate < _SPLIT_THRESHOLD:
        # Single call — within Haiku's reliable extraction window
        return await _call_summarizer(
            messages, task_description, start_turn, end_turn,
        )

    if token_estimate < _THREE_WAY_SPLIT_THRESHOLD:
        # Two parallel calls with 1-turn overlap
        midpoint = _find_turn_midpoint(messages)

        # Overlap: include 1 interaction pair (2 messages) from the boundary
        overlap = min(2, midpoint)
        first_half = messages[:midpoint]
        second_half = messages[midpoint - overlap:]

        first_start, first_end = _infer_turn_range(first_half, start_turn)
        second_start, second_end = _infer_turn_range(second_half, first_end)

        logger.info(
            "Summarizer: splitting into 2 calls — "
            "first: %d msgs (turns %d-%d), second: %d msgs (turns %d-%d)",
            len(first_half), first_start, first_end,
            len(second_half), second_start, second_end,
        )

        summary_a, summary_b = await asyncio.gather(
            _call_summarizer(
                first_half, task_description,
                first_start, first_end,
                split_note=_SPLIT_FIRST_HALF_NOTE,
            ),
            _call_summarizer(
                second_half, task_description,
                second_start, second_end,
                split_note=_SPLIT_SECOND_HALF_NOTE,
            ),
        )
        return f"{summary_a}\n\n{summary_b}"

    # Three parallel calls with 1-turn overlaps (very rare)
    third = len(messages) // 3
    two_thirds = 2 * third

    overlap = min(2, third)
    part_1 = messages[:third]
    part_2 = messages[third - overlap:two_thirds]
    part_3 = messages[two_thirds - overlap:]

    t1_start, t1_end = _infer_turn_range(part_1, start_turn)
    t2_start, t2_end = _infer_turn_range(part_2, t1_end)
    t3_start, t3_end = _infer_turn_range(part_3, t2_end)

    logger.info(
        "Summarizer: splitting into 3 calls — %d/%d/%d msgs",
        len(part_1), len(part_2), len(part_3),
    )

    s1, s2, s3 = await asyncio.gather(
        _call_summarizer(
            part_1, task_description, t1_start, t1_end,
            split_note=_SPLIT_FIRST_HALF_NOTE,
        ),
        _call_summarizer(
            part_2, task_description, t2_start, t2_end,
            split_note=_SPLIT_SECOND_HALF_NOTE,
        ),
        _call_summarizer(
            part_3, task_description, t3_start, t3_end,
            split_note=_SPLIT_SECOND_HALF_NOTE,
        ),
    )
    return f"{s1}\n\n{s2}\n\n{s3}"
