"""Exploration planner — decomposes a scraping task into an exploration checklist.

Runs a single LLM call that reads the user's task description and schema
fields, then produces a checklist of investigation tasks the scraping agent
must resolve before writing a script.

The planner never sees the website. It works purely from the task text and
the output schema. Its job is decomposition — breaking a short user sentence
into the distinct things that need to be figured out.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from ..schema.compiler import CompiledSchema
from ..schema.nodes import (
    FreestyleDictNode,
    ListNode,
    Node,
    ObjectNode,
    ScalarNode,
)
from .llm import LLMConfig, call_llm

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Type names for field descriptions
# ═══════════════════════════════════════════════════════════════

_TYPE_NAMES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "float",
    bool: "boolean",
}


# ═══════════════════════════════════════════════════════════════
#  Field extraction from schema nodes
# ═══════════════════════════════════════════════════════════════


def _extract_field_info(root: Node, prefix: str = "") -> list[str]:
    """Walk the schema node tree and return a flat list of field descriptions.

    Each entry looks like ``"company_name (string, required)"`` or
    ``"funding.amount (float, optional)"``.  Nested objects use
    dot-notation; list items use ``[]`` suffixes.
    """
    results: list[str] = []

    if isinstance(root, ObjectNode):
        for name, (child, optional) in root.fields.items():
            full_name = f"{prefix}{name}"
            presence = "optional" if optional else "required"

            if isinstance(child, ScalarNode):
                type_name = _TYPE_NAMES.get(child.type_, "value")
                results.append(f"{full_name} ({type_name}, {presence})")

            elif isinstance(child, ObjectNode):
                # Nested object — recurse with dot prefix.
                results.extend(
                    _extract_field_info(child, prefix=f"{full_name}.")
                )

            elif isinstance(child, ListNode):
                if isinstance(child.item, ObjectNode):
                    # List of objects — recurse into the item schema.
                    results.extend(
                        _extract_field_info(
                            child.item, prefix=f"{full_name}[]."
                        )
                    )
                elif isinstance(child.item, ScalarNode):
                    item_type = _TYPE_NAMES.get(child.item.type_, "value")
                    results.append(
                        f"{full_name} (list of {item_type}, {presence})"
                    )
                else:
                    results.append(f"{full_name} (list, {presence})")

            elif isinstance(child, FreestyleDictNode):
                results.append(f"{full_name} (object, {presence})")

    elif isinstance(root, ListNode):
        # Unwrap nested ListNodes (e.g. Items([{...}]) → ListNode(ListNode(ObjectNode))).
        inner = root.item
        while isinstance(inner, ListNode):
            inner = inner.item
        if isinstance(inner, ObjectNode):
            results.extend(_extract_field_info(inner, prefix=prefix))
        elif isinstance(inner, ScalarNode):
            type_name = _TYPE_NAMES.get(inner.type_, "value")
            results.append(f"{prefix}item ({type_name})")

    return results


# ═══════════════════════════════════════════════════════════════
#  System Prompt
# ═══════════════════════════════════════════════════════════════

_PLANNER_SYSTEM_PROMPT = """\
You are a planning analyst who decomposes web scraping tasks into \
exploration checklists. You have never seen the target website. You do \
not know what it looks like, how it is structured, what buttons or \
filters exist, or how navigation works. You work purely from two inputs: \
the user's task description and the list of fields they want to extract.

Your job is to produce a checklist of things that need to be figured out \
before a scraping script can be written. Each checklist item is an \
investigation task — something an exploring agent will need to understand \
by actually looking at the website.

## Your Mindset

You are a careful reader, not a creative interpreter. You treat the \
user's words as a specification.

If they said "W24 batch companies," your checklist says "W24 batch \
companies" — you do not generalize to "companies from any batch" or \
add "handle other batches too." If they said "the first page of \
results," you do not add "figure out how to get all pages." If they \
said "get books" without mentioning "all," you do not insert a \
completeness requirement. You mirror their language exactly.

You do not know the website. This means you never suggest mechanisms. \
You never say "click the filter," "use pagination," "scroll down," \
"change the URL," "navigate to the detail page," "use the search bar," \
or anything like that. Those are solutions, and you do not know which \
solutions apply to a page you have never seen. You describe what needs \
to be understood — the exploring agent will discover the mechanism.

## Common Mistakes You Avoid

- **Adding scope the user did not ask for.** If they did not say "all," \
you do not add "collect everything" or "not just the first visible \
set." If they asked for "the top 10," you do not add a step about \
getting the rest.

- **Suggesting mechanisms.** You do not know the page has filters, tabs, \
pagination, infinite scroll, search bars, or dropdowns. Do not mention \
any of these. Write what needs to be figured out, not how.

- **Interpreting vague words.** If the user says "popular repositories," \
you write "popular repositories" in the checklist. You do not decide \
what "popular" means or add criteria for popularity.

- **Over-decomposing.** "Get all remote Python developer jobs in Berlin" \
has constraints (remote, Python, Berlin) but they are part of one \
goal: getting to the right job listings. Do not split each constraint \
into its own investigation task unless they are clearly independent \
things to figure out.

- **Creating per-field checklist items.** Do not create a separate \
"Figure out how to extract X" item for each field. Field extraction \
is covered by the final item that lists all fields. The other \
checklist items are about access, scope, and structure — not \
individual fields. A checklist with 20 per-field items is noise.

- **Under-decomposing.** "Get all apartments under $2000 listed in the \
last month with full descriptions" has a price constraint, a time \
constraint, a completeness scope ("all"), and a depth question (full \
descriptions — where do they live?). Each of these is a separate \
thing to figure out.

- **Skipping fields.** Every field the user defined in their schema must \
appear explicitly in the checklist. Do not write "extract all the \
fields" as a single item — that lets the exploring agent skip fields \
it finds difficult. Spell out every field name in the final item.

## Your Process

Before writing the checklist, briefly: (1) restate what the user is \
asking for in your own words to make sure you understand the task, \
then (2) identify the distinct things that need to be figured out — \
constraints, scope, structure, and access questions. Keep this analysis \
short — a few sentences, not a wall of text. Then produce the checklist.

## Output Format

After your brief analysis, output the checklist inside a fenced code \
block with the language tag ``checklist``:

```checklist
[ ] Figure out how to ...
[ ] Figure out how to ...
[ ] Figure out how to extract each of these fields:
    - field_one
    - field_two
    - field_three
```

The checklist has two parts:
- First: items about access, scope, or structure (NOT per-field items).
- Last: one item listing every schema field that must be extracted. \
Do not omit any field from this list.\
"""


# ═══════════════════════════════════════════════════════════════
#  User message builder
# ═══════════════════════════════════════════════════════════════


def _build_user_message(task: str, field_descriptions: list[str]) -> str:
    """Build the user message for the planner LLM call."""
    fields_block = "\n".join(f"- {f}" for f in field_descriptions)
    return f"## Task\n\n{task}\n\n## Schema Fields\n\n{fields_block}"


# ═══════════════════════════════════════════════════════════════
#  Checklist extraction (robust, multi-tier fallback)
# ═══════════════════════════════════════════════════════════════


def _extract_checklist(response_text: str) -> str:
    """Extract the checklist from the planner's response.

    Uses a three-tier fallback strategy:
    1. Find a ````` ```checklist ``` ````` fenced block.
    2. Find a "Step 4" heading and take everything after it.
    3. Collect all lines starting with ``[ ]`` (plus ``- `` continuations).
    4. Last resort: return the full response.
    """
    # Tier 1: fenced ```checklist block.
    match = re.search(
        r"```checklist\s*\n(.*?)```",
        response_text,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()

    # Tier 2: Step 4 section — take everything after it.
    step4_match = re.search(
        r"\*\*Step\s*4[^*]*\*\*[^\n]*\n(.*)",
        response_text,
        re.DOTALL,
    )
    if step4_match:
        text = step4_match.group(1).strip()
        # Strip any markdown fencing that wraps the content.
        text = re.sub(r"^```\w*\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        return text.strip()

    # Tier 3: collect all [ ] lines and their sub-items.
    lines: list[str] = []
    in_field_block = False
    for line in response_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[ ]"):
            lines.append(stripped)
            # Track if this is the "fields" item so we capture sub-items.
            in_field_block = "field" in stripped.lower()
        elif in_field_block and stripped.startswith("- "):
            # Preserve indentation for sub-items.
            lines.append(f"    {stripped}")
        else:
            in_field_block = False

    if lines:
        return "\n".join(lines)

    # Last resort: return everything.  The planner output is advisory —
    # a messy checklist is better than nothing.
    logger.warning("Could not extract structured checklist from planner response")
    return response_text.strip()


# ═══════════════════════════════════════════════════════════════
#  Main Function
# ═══════════════════════════════════════════════════════════════


async def generate_exploration_plan(
    *,
    task: str,
    compiled_schema: CompiledSchema,
    llm_config: LLMConfig,
) -> str:
    """Generate an exploration checklist for a scraping task.

    Runs a single LLM call that decomposes the user's task and schema
    fields into a checklist of investigation tasks. The checklist is
    returned as a plain string, ready to be injected into the main
    agent's initial message.

    Args:
        task: The user's natural-language task description.
        compiled_schema: The compiled output schema (for field names).
        llm_config: LLM configuration for the planner call.

    Returns:
        The exploration checklist as a multi-line string.
    """
    start = time.time()

    field_descriptions = _extract_field_info(compiled_schema.root)
    user_message = _build_user_message(task, field_descriptions)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_message},
    ]

    logger.info(
        "Generating exploration plan for %d schema fields...",
        len(field_descriptions),
    )

    response = await call_llm(
        llm_config,
        system=_PLANNER_SYSTEM_PROMPT,
        messages=messages,
    )

    # Extract text from response blocks.
    full_text = ""
    for block in response.content:
        if block.type == "text":
            full_text += block.text

    checklist = _extract_checklist(full_text)

    duration = time.time() - start
    logger.info(
        "Exploration plan generated in %.1fs (%d chars)",
        duration,
        len(checklist),
    )

    return checklist
