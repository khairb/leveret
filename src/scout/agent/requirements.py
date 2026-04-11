"""Requirements agent — extracts concrete success criteria from exploration history.

Runs a single LLM call that reads the scraping agent's full conversation
history and produces structured, checkable requirements for the validator.
These requirements capture context discovered during exploration (e.g., total
page count, available fields, navigation steps) that the validator would
otherwise not have access to.

Requirements are generated once and reused across retry attempts.
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
is not genuinely useful to the user without it.** Everything else is soft. \
When in doubt, make it hard.

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
"each") means exactly that. The full scope must be covered.
- **Field completeness**: A requested field must be fully extracted — not \
just present. A truncated value is missing data. An empty string is missing \
data.

**Soft requirements are genuinely optional extras** — things where the output \
is still 100% usable for its stated purpose even if they fail. The soft list \
should be short. If you are unsure whether something is soft, it is hard.

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
- Total items: <exact number or tight range, with reasoning — e.g., \
"~300 items (20 pages × ~15 items/page, seen in exploration)">
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
non-empty and untruncated" not "must extract fields." \
"Must produce ~300 items" not "must extract multiple items."

3. **Interpret the user's language strictly.** "All pages" = every single \
page. "All items" = every item. "Complete information" = nothing truncated. \
Treat their words as a specification.

4. **Include navigation prerequisites as hard.** Consent banners, region \
selection, login, filter application — if the exploration showed these must \
happen before data is accessible, they are hard requirements.

5. **State expected quantities precisely.** "~300 items (20 pages × ~15 \
items/page)" gives the validator a concrete target. Vague quantities like \
"multiple pages" are worthless to a validator.

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
                    text[:max_content_chars // 2]
                    + "\n\n... (truncated) ...\n\n"
                    + text[-max_content_chars // 2:]
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
                            text[:max_content_chars // 2]
                            + "\n\n... (truncated) ...\n\n"
                            + text[-max_content_chars // 2:]
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
                                result_content[:max_content_chars // 2]
                                + "\n\n... (truncated) ...\n\n"
                                + result_content[-max_content_chars // 2:]
                            )
                        is_error = block.get("is_error", False)
                        label = "Tool ERROR" if is_error else "Tool result"
                        block_parts.append(f"**{label}:**\n```\n{result_content}\n```")

            if block_parts:
                parts.append(
                    f"### {role.upper()} (message {i + 1})\n\n"
                    + "\n\n".join(block_parts) + "\n"
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

    user_message = (
        f"## Original Task\n\n{task}\n\n"
        f"## Exploration History\n\n{conversation_summary}"
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_message},
    ]

    logger.info("Generating requirements from %d conversation messages...", len(conversation_history))

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
        duration, len(requirements),
    )

    return requirements.strip()
