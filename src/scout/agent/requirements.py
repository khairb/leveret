"""Requirements agent — extracts concrete success criteria from exploration history.

Runs a single LLM call that reads the scraping agent's full conversation
history and produces structured, checkable requirements for the validator.
These requirements capture context discovered during exploration (e.g., total
page count, available fields, navigation steps) that the validator would
otherwise not have access to.

Requirements are generated once, then revised with execution evidence if
stagnation is detected (multiple rejections for the same issue).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .llm import LLMConfig, call_llm

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  System Prompt
# ═══════════════════════════════════════════════════════════════

_REQUIREMENTS_SYSTEM_PROMPT = """\
You are a requirements analyst who deeply understands web scraping objectives. \
Your purpose is to translate the user's intent — and the facts the scraping \
agent discovered during exploration — into a clear, honest contract that a \
validation agent can use to determine whether the final script succeeded.

You hold one core belief: **a requirement is hard if and only if the output \
is not genuinely useful to the user without it.** Everything else is soft.

You will be given:
1. The original task — what the user wants to accomplish
2. The exploration history — what the scraping agent discovered about the \
target website

## Your Mindset Before Writing Anything

Ask yourself: **What is the user actually going to do with this data?** \
What result would genuinely satisfy them — and what would disappoint them?

If the user said "all pages," they need all the data — 3 out of 20 pages is \
not a partial success, it is a failure. If they asked for specific fields, \
a value that is cut off mid-sentence is not a success — it is missing data. \
The user's language is a binding specification, not a rough sketch.

**These are always hard requirements — never classify them as soft:**

- **Scope coverage**: Any quantifier in the task ("all", "every", "complete", \
"each") means exactly that. The script must attempt the full scope.
- **Field completeness**: A requested field must be fully extracted — not \
just present. A truncated value is missing data. An empty string is missing \
data.

**Soft requirements are genuinely optional extras** — things where the output \
is still 100% usable for its stated purpose even if they fail. The soft list \
should be short. If you are unsure whether something is soft, it is hard.

## Understanding Page-Reported Numbers

Websites routinely display counts ("200 results", "Page 1 of 20") that \
do not match what is actually extractable. A page header might say "200 \
listings" but only 196 are real — the other 4 were delisted, are \
region-locked, or exist only as server-side placeholders that never \
render in the DOM. Lazy-loaded counters count items the user might see \
after infinite scrolling, not what a single session will yield. \
Anti-bot systems silently drop results under automated access. And the \
page you explored 2 minutes ago is not the same page the script will \
hit — listings get booked, posts get deleted, results shift.

When you see a number from the page UI, note it as an informed estimate \
with its source — "~200 items (page header)" — so the validator \
understands where the expectation comes from and can judge whether the \
extraction was thorough, rather than treating it as an exact pass/fail \
threshold.

## Output Format

Respond with ONLY the structured requirements below. No preamble, no \
explanation outside the structure.

```
## Hard Requirements
[Failure on any one = rejection. Each requirement must be concrete and \
measurable.]

- <requirement> (evidence: <what you saw in the exploration>)
...

## Soft Requirements
[Truly optional. Output remains fully usable without these.]

- <requirement> (evidence: <what you saw in the exploration>)
...

## Expected Output Profile
- Total items: <estimated count with reasoning and source — e.g., \
"~300 items (20 pages × ~15 items/page, based on page counter seen \
during exploration — actual extractable count may vary)">
- Required fields per item: <explicit list — these are hard>
- Output format: <JSON array / JSON lines / CSV / etc.>
- Scope: <pages, categories, filters that must all be covered>
```

## Rules

1. **Ground every requirement in evidence.** Cite what you saw — a page \
counter ("Page 1 of 20"), a results indicator ("1,247 results"), a DOM \
section with specific fields, a filter menu. Never invent requirements not \
supported by exploration.

2. **Be concrete and measurable.** "Must extract from all 20 pages" not \
"must handle pagination." "Each item must have all requested fields — \
non-empty and untruncated" not "must extract fields."

3. **Interpret the user's language strictly for scope and fields.** \
"All pages" = every single page. "All items" = every item. "Complete \
information" = nothing truncated. Treat their words as a specification.

4. **Include navigation prerequisites as hard.** Consent banners, region \
selection, login, filter application — if the exploration showed these must \
happen before data is accessible, they are hard requirements.

5. **State expected quantities with source and reasoning.** \
"~300 items (20 pages × ~15 items/page, based on page counter)" gives \
the validator a concrete target. Vague quantities like "multiple pages" \
are worthless. But always note the source — a page-reported number is \
an estimate, not a guarantee.

6. **Keep the soft list minimal and honest.** Before adding a soft \
requirement, ask: "Would I still call this a success if this fails?" If \
there is any hesitation, put it in hard. Soft means truly optional.\
"""


# ═══════════════════════════════════════════════════════════════
#  Conversation Summarizer
# ═══════════════════════════════════════════════════════════════


def _summarize_conversation(messages: list[dict]) -> str:
    """Convert the exploration conversation history into a readable summary.

    Keeps text content and tool calls/results, but truncates very long
    outputs (e.g., full page views) to keep the requirements agent's
    context manageable.
    """
    parts: list[str] = []
    max_content_chars = 3000  # per message block

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if isinstance(content, str):
            # Plain text message.
            text = content
            if len(text) > max_content_chars:
                text = (
                    text[: max_content_chars // 2]
                    + "\n\n... (truncated) ...\n\n"
                    + text[-max_content_chars // 2 :]
                )
            parts.append(f"### {role.upper()} (message {i + 1})\n\n{text}\n")

        elif isinstance(content, list):
            # List of content blocks (tool_use, tool_result, text).
            block_parts: list[str] = []
            for block in content:
                btype = block.get("type", "")

                if btype == "text":
                    text = block.get("text", "")
                    if len(text) > max_content_chars:
                        text = (
                            text[: max_content_chars // 2]
                            + "\n\n... (truncated) ...\n\n"
                            + text[-max_content_chars // 2 :]
                        )
                    block_parts.append(text)

                elif btype == "tool_use":
                    name = block.get("name", "?")
                    args = block.get("input", {})
                    # Show code for python tool calls (important context).
                    if name == "python" and "code" in args:
                        code = args["code"]
                        if len(code) > 1500:
                            code = code[:750] + "\n# ... truncated ...\n" + code[-750:]
                        block_parts.append(f"**Tool call: `{name}`**\n```python\n{code}\n```")
                    else:
                        block_parts.append(f"**Tool call: `{name}`** — {args}")

                elif btype == "tool_result":
                    result_content = block.get("content", "")
                    if isinstance(result_content, str):
                        if len(result_content) > max_content_chars:
                            result_content = (
                                result_content[: max_content_chars // 2]
                                + "\n\n... (truncated) ...\n\n"
                                + result_content[-max_content_chars // 2 :]
                            )
                        is_error = block.get("is_error", False)
                        label = "Tool ERROR" if is_error else "Tool result"
                        block_parts.append(f"**{label}:**\n```\n{result_content}\n```")

            if block_parts:
                parts.append(
                    f"### {role.upper()} (message {i + 1})\n\n" + "\n\n".join(block_parts) + "\n"
                )

    return "\n---\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════
#  Main Function
# ═══════════════════════════════════════════════════════════════


async def generate_requirements(
    *,
    task: str,
    conversation_history: list[dict],
    llm_config: LLMConfig,
) -> str:
    """Generate concrete success criteria from the exploration history.

    Args:
        task: The original scraping task description.
        conversation_history: The full message list from the scraping
            agent's exploration (all user/assistant/tool messages).
        llm_config: LLM configuration for the requirements agent.

    Returns:
        Structured requirements string (hard requirements, soft
        requirements, expected output profile).
    """
    start = time.time()

    conversation_summary = _summarize_conversation(conversation_history)

    user_message = f"## Original Task\n\n{task}\n\n## Exploration History\n\n{conversation_summary}"

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_message},
    ]

    logger.info(
        "Generating requirements from %d conversation messages...", len(conversation_history)
    )

    response = await call_llm(
        llm_config,
        system=_REQUIREMENTS_SYSTEM_PROMPT,
        messages=messages,
    )

    # Extract text from response.
    requirements = ""
    for block in response.content:
        if block.type == "text":
            requirements += block.text

    duration = time.time() - start
    logger.info(
        "Requirements generated in %.1fs (%d chars)",
        duration,
        len(requirements),
    )

    return requirements.strip()


# ═══════════════════════════════════════════════════════════════
#  Revision System Prompt
# ═══════════════════════════════════════════════════════════════

_REVISION_SYSTEM_PROMPT = """\
You are a requirements analyst revising success criteria based on \
execution evidence.

You originally wrote requirements based on the scraping agent's \
exploration of the target website. Now you have evidence from actual \
script execution — what the scripts actually produced when they ran \
against the live site, across multiple independent attempts.

Your job is to look at this evidence honestly and ask: where did my \
original requirements describe what the user needs, and where did they \
describe what the page claimed to have? These are not always the same \
thing.

Structural requirements — the fields the user asked for, the output \
format, the pages that must be visited, navigation prerequisites — \
these reflect what the user genuinely needs. They do not change based \
on execution evidence. If the user asked for titles and prices, every \
item still needs titles and prices.

What can be revised is anything derived from what the page displayed \
rather than what the user asked for. Item counts, result totals, \
per-field completeness rates — these are claims the page made that \
execution can verify or contradict. Look at the evidence across \
attempts and use your judgment.

For example: your original requirements might say "~200 items based \
on page header." But three different scripts all extract 197-199. \
The page header was a UI label, not a guarantee — maybe a few items \
were delisted, region-locked, or exist only as server-side \
placeholders. The right revision is to adjust the expected count to \
~198, because that is what the website actually serves.

For example: you required every item to have a "rating" field. But \
across every attempt, ~10% of items come back with rating as null. \
Those listings genuinely have no rating on the website — the element \
is not in their DOM. Requiring 100% presence for data that does not \
exist on the page will cause infinite rejections. The revision should \
note that rating is present on most items and null where the listing \
has no rating.

For example: you required full description text, but across attempts \
the description consistently cuts off at exactly 150 characters. The \
website itself truncates it in the listing view — the full text only \
lives on the detail page. The script is capturing what the site \
provides at the listing level.

For example: you expected 300 items (20 pages × 15 items/page), but \
every attempt gets 293. The last page only has 8 items — that is just \
how the data divides across pages, not a bug.

These are just examples — real cases vary. The principle is: when \
the evidence shows a stable, consistent gap between your requirement \
and what the scripts produce, and different scripts converge on the \
same result, the requirement was based on an inaccurate assumption \
about the website. Revise it to match what execution reveals.

On the other hand, when the evidence shows wild variance between \
attempts (50 items, then 120, then 80), that is script instability, \
not a website constraint. And when the result is far below the target \
(50 items when 200 expected), that is a broken script. In these cases, \
keep the original requirement — the problem is fixable.

## Format

Keep the same Hard/Soft/Expected Output Profile format. Mark any \
adjusted requirements with "(revised: <reason>)" so the validator \
understands what changed and why.

Respond with the **complete revised requirements** — not just the \
changes. The validator will see only the revised version.\
"""


# ═══════════════════════════════════════════════════════════════
#  Revision Function
# ═══════════════════════════════════════════════════════════════


def _format_attempt_evidence(
    attempt_history: list[dict],
) -> str:
    """Format attempt history as evidence for the revision agent."""
    parts: list[str] = []
    for rec in attempt_history:
        attempt_num = rec.get("attempt_number", "?")
        feedback = rec.get("rejection_feedback", "")
        stdout_sample = rec.get("stdout_sample", "")
        returncode = rec.get("returncode", "?")

        # Truncate stdout sample for the revision context.
        sample_lines = stdout_sample.split("\n")[:20]
        sample_text = "\n".join(sample_lines)

        parts.append(
            f"### Attempt {attempt_num}\n"
            f"**Exit code:** {returncode}\n"
            f"**Rejection feedback:** {feedback}\n"
            f"**Output sample (first 20 lines):**\n"
            f"```\n{sample_text}\n```\n"
        )
    return "\n".join(parts)


async def revise_requirements(
    *,
    task: str,
    original_requirements: str,
    attempt_history: list[dict],
    llm_config: LLMConfig,
) -> str:
    """Revise requirements based on execution evidence.

    Called when stagnation is detected — multiple rejections for
    similar issues suggest that some requirements may reflect
    page-reported numbers rather than actual extractable data.

    Args:
        task: The original scraping task description.
        original_requirements: The requirements as originally generated.
        attempt_history: List of dicts with keys: attempt_number,
            rejection_feedback, stdout_sample, returncode.
        llm_config: LLM configuration for the requirements agent.

    Returns:
        Revised requirements string.
    """
    start = time.time()

    evidence = _format_attempt_evidence(attempt_history)

    user_message = (
        f"## Original Task\n\n{task}\n\n"
        f"## Your Original Requirements\n\n{original_requirements}\n\n"
        f"## Execution Evidence ({len(attempt_history)} attempts)\n\n"
        f"{evidence}"
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_message},
    ]

    logger.info(
        "Revising requirements based on %d attempts...",
        len(attempt_history),
    )

    response = await call_llm(
        llm_config,
        system=_REVISION_SYSTEM_PROMPT,
        messages=messages,
    )

    revised = ""
    for block in response.content:
        if block.type == "text":
            revised += block.text

    duration = time.time() - start
    logger.info(
        "Requirements revised in %.1fs (%d chars)",
        duration,
        len(revised),
    )

    return revised.strip()
