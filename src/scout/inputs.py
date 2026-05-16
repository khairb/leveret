"""Dynamic inputs — parameterize scraping scripts across runs.

The ``Input`` class lets users attach metadata (description, explicit type)
to input values.  Bare Python values are also accepted — the type is
inferred via ``type(value)``.

Prompt builders in this module generate the agent context fragments that
teach the LLM to use ``inputs["key"]`` instead of hardcoding values.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .errors import ConfigError

# ═══════════════════════════════════════════════════════════════
#  Input class
# ═══════════════════════════════════════════════════════════════

_SUPPORTED_TYPES = (str, int, float, bool)
_TYPE_NAMES = {str: "str", int: "int", float: "float", bool: "bool"}
_NAME_TO_TYPE = {"str": str, "int": int, "float": float, "bool": bool}


class Input:
    """A dynamic input parameter for a scraping script.

    Use ``Input`` to attach metadata (description, explicit type) to
    values passed via ``scraper.run(inputs={...})``. For simple cases,
    bare Python values work too — the type is inferred automatically.

    Args:
        value: Example value used during script generation and for
            type inference. The generated script will use
            ``inputs["key"]`` instead of hardcoding this value.
        description: Human-readable description of what this input
            represents. Helps the AI generate better code.
        type_: Explicit Python type (``str``, ``int``, ``float``, or
            ``bool``). When omitted, the type is inferred from
            ``value``.

    Examples::

        Input("python developer")
        Input("Berlin", description="City to filter by")
        Input(50, type_=int, description="Max listings")
    """

    __slots__ = ("value", "type_", "description")

    def __init__(
        self,
        value: Any,
        *,
        description: str | None = None,
        type_: type | None = None,
    ) -> None:
        if type_ is not None:
            if type_ not in _SUPPORTED_TYPES:
                raise ConfigError(
                    f"Input type_ must be one of str, int, float, bool "
                    f"(got {type_.__name__})"
                )
            self.type_ = type_
        else:
            inferred = type(value)
            if inferred not in _SUPPORTED_TYPES:
                raise ConfigError(
                    f"Cannot infer input type from {inferred.__name__!r}. "
                    f"Supported types: str, int, float, bool."
                )
            self.type_ = inferred
        self.value = value
        self.description = description

    def __repr__(self) -> str:
        parts = [repr(self.value)]
        if self.description is not None:
            parts.append(f"description={self.description!r}")
        return f"Input({', '.join(parts)})"


# ═══════════════════════════════════════════════════════════════
#  Normalization
# ═══════════════════════════════════════════════════════════════

# Every input key must be a valid Python identifier (used as dict key
# in the prompt and accessed via ``inputs["key"]``).
_IDENT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def normalize_inputs(
    raw: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
    """Normalize user-provided inputs into example values and definitions.

    Returns ``(None, None)`` when *raw* is ``None`` or an empty dict.

    Otherwise returns ``(example_values, input_defs)`` where:

    - **example_values** is a plain ``{key: value}`` dict for REPL
      injection and function calls.
    - **input_defs** is a list of dicts with keys ``name``, ``type``,
      ``description``, and ``example`` — structured metadata for prompt
      building and validation.
    """
    if raw is None or len(raw) == 0:
        return None, None

    if not isinstance(raw, dict):
        raise ConfigError(
            f"inputs must be a dict mapping names to values "
            f"(got {type(raw).__name__})"
        )

    example_values: dict[str, Any] = {}
    input_defs: list[dict[str, Any]] = []

    for key, val in raw.items():
        # Validate key.
        if not isinstance(key, str) or not _IDENT_RE.match(key):
            raise ConfigError(
                f"Input key {key!r} is not a valid Python identifier. "
                f"Keys must match [a-zA-Z_][a-zA-Z0-9_]*."
            )

        # Unwrap Input instances.
        if isinstance(val, Input):
            example = val.value
            type_ = val.type_
            description = val.description
        else:
            example = val
            inferred = type(val)
            if inferred not in _SUPPORTED_TYPES:
                raise ConfigError(
                    f"Input {key!r} has unsupported type "
                    f"{inferred.__name__!r}. "
                    f"Supported: str, int, float, bool. "
                    f"Wrap in Input(value, type_=...) for explicit typing."
                )
            type_ = inferred
            description = None

        example_values[key] = example
        input_defs.append({
            "name": key,
            "type": type_,
            "description": description,
            "example": example,
        })

    return example_values, input_defs


# ═══════════════════════════════════════════════════════════════
#  Prompt builders
# ═══════════════════════════════════════════════════════════════

def _access_pattern_list(input_defs: list[dict[str, Any]]) -> str:
    """Build a human-readable list of ``inputs["key"]`` references.

    For ≤3 fields:  ``inputs["a"], inputs["b"] and inputs["c"]``
    For >3 fields:  ``inputs["a"], inputs["b"], inputs["c"] and the
    other inputs fields``
    """
    names = [d["name"] for d in input_defs]
    refs = [f'inputs["{n}"]' for n in names]

    if len(refs) == 1:
        return refs[0]
    if len(refs) <= 3:
        return ", ".join(refs[:-1]) + " and " + refs[-1]
    # 4+ fields — show first 3, summarize the rest.
    return (
        ", ".join(refs[:3])
        + " and the other inputs fields"
    )


def _example_repr(value: Any) -> str:
    """Format an example value for display in the prompt."""
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def build_inputs_section(input_defs: list[dict[str, Any]]) -> str:
    """Build the ``## Dynamic Inputs`` section for the system prompt."""
    lines = [
        "\n## Dynamic Inputs\n",
        "Your function receives an `inputs` dict with the following fields:\n",
    ]

    for d in input_defs:
        type_name = _TYPE_NAMES[d["type"]]
        desc = d["description"] or ""
        example = _example_repr(d["example"])
        if desc:
            lines.append(f'  - "{d["name"]}" ({type_name}): {desc}, e.g. {example}')
        else:
            lines.append(f'  - "{d["name"]}" ({type_name}): e.g. {example}')

    access = _access_pattern_list(input_defs)
    lines.append("")
    lines.append(
        f"The `inputs` variable is already available in your environment "
        f"— use {access} in your exploration and in your final function."
    )
    lines.append("")
    lines.append(
        f"Always read values from inputs ({access}) — never hardcode "
        f"these values. They change every run."
    )
    lines.append("")

    return "\n".join(lines)


def build_inputs_hint(input_defs: list[dict[str, Any]]) -> str:
    """Build the inputs hint for the initial user message."""
    access = _access_pattern_list(input_defs)
    return (
        f" The `inputs` dict is available with the user's parameters "
        f"— use {access} in your exploration and in your final function."
    )


def build_inputs_rule(input_defs: list[dict[str, Any]]) -> str:
    """Build Rule 11 for the system prompt rules section."""
    access = _access_pattern_list(input_defs)

    # Build anti-pattern examples from actual example values.
    anti_examples = []
    for d in input_defs[:2]:  # Cap at 2 for readability.
        anti_examples.append(
            f'{_example_repr(d["example"])} → inputs["{d["name"]}"]'
        )
    anti_str = ", ".join(anti_examples)

    return (
        f'\n11. **Use `inputs` for all user-provided values — never '
        f'hardcode them.** '
        f'The `inputs` variable contains values that change every run. '
        f'Always access them as {access} — never write the literal value '
        f'in your code. If you find yourself typing a literal like '
        f'{anti_str} — replace it with the corresponding '
        f'inputs["..."] access.'
    )


def build_inputs_tool_desc_addition() -> str:
    """Return the addition to the python tool description."""
    return " and an `inputs` dict"


def build_inputs_phase3_example(input_defs: list[dict[str, Any]]) -> str:
    """Build the Phase 3 code example with the 4-param signature."""
    # Build the inputs comment line with all field access patterns.
    field_refs = ", ".join(f'inputs["{d["name"]}"]' for d in input_defs)

    return f"""\
```python
async def scrape(page, start_url, inputs, checkpoint):
    # page: a NEW browser page — not your exploration session. No cookies,
    #        no dismissed popups, no navigation history.
    # start_url: the original URL. Navigate here or to a more direct
    #            URL if you discovered one during exploration (e.g. a
    #            search URL with query params already filled in).
    # inputs: dict with user-provided values — {field_refs}.
    #         Never hardcode these values.
    # checkpoint: await checkpoint("label", data_preview?) to record state
    #
    # Return value must match the output schema below.
    # Raise an exception if scraping fails.

    await page.goto(start_url)  # or a more direct URL you discovered
    ...
    return data
```"""


# The original Phase 3 example (no inputs).
_PHASE3_EXAMPLE_NO_INPUTS = """\
```python
async def scrape(page, start_url, checkpoint):
    # page: a NEW browser page — not your exploration session. No cookies,
    #        no dismissed popups, no navigation history.
    # start_url: the original URL. Navigate here or to a more direct
    #            URL if you discovered one during exploration (e.g. a
    #            search URL with query params already filled in).
    # checkpoint: await checkpoint("label", data_preview?) to record state
    #
    # Return value must match the output schema below.
    # Raise an exception if scraping fails.

    await page.goto(start_url)  # or a more direct URL you discovered
    ...
    return data
```"""


def build_inputs_fragments(
    input_defs: list[dict[str, Any]],
) -> dict[str, str]:
    """Build all prompt fragments for dynamic inputs.

    Returns a dict keyed by placeholder name, ready to be passed
    to ``build_system_prompt(inputs_fragments=...)``.
    """
    return {
        "inputs_tool_desc": build_inputs_tool_desc_addition(),
        "inputs_section": build_inputs_section(input_defs),
        "phase3_code_example": build_inputs_phase3_example(input_defs),
        "inputs_rule": build_inputs_rule(input_defs),
    }


# ═══════════════════════════════════════════════════════════════
#  Metadata helpers
# ═══════════════════════════════════════════════════════════════

def format_inputs_metadata(input_defs: list[dict[str, Any]]) -> str:
    """Format input definitions for the script metadata header.

    Produces: ``query (str), location (str) and max_results (int)``
    """
    parts = [
        f"{d['name']} ({_TYPE_NAMES[d['type']]})" for d in input_defs
    ]
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


_INPUTS_META_RE = re.compile(r"(\w+)\s*\((\w+)\)")


def parse_inputs_metadata(line: str) -> dict[str, type]:
    """Parse the ``inputs:`` metadata line back to ``{name: type}``."""
    result: dict[str, type] = {}
    for m in _INPUTS_META_RE.finditer(line):
        name = m.group(1)
        type_name = m.group(2)
        if type_name in _NAME_TO_TYPE:
            result[name] = _NAME_TO_TYPE[type_name]
    return result


def validate_inputs_against_metadata(
    inputs: dict[str, Any] | None,
    expected_meta: str | None,
) -> None:
    """Validate caller's inputs against the script's metadata.

    Raises ``ConfigError`` with a descriptive message on mismatch.
    """
    has_inputs = inputs is not None and len(inputs) > 0
    has_meta = expected_meta is not None and expected_meta.strip() != ""

    if has_meta and not has_inputs:
        raise ConfigError(
            f"This script expects inputs: {expected_meta}. "
            f"Pass them via scraper.run(inputs={{...}})."
        )

    if has_inputs and not has_meta:
        raise ConfigError(
            "This script was generated without dynamic inputs. "
            "To use inputs, regenerate the script: scraper.regenerate()"
        )

    if not has_inputs and not has_meta:
        return  # Both absent — fine.

    assert inputs is not None and expected_meta is not None
    expected = parse_inputs_metadata(expected_meta)

    # Check for missing keys.
    for key in expected:
        if key not in inputs:
            raise ConfigError(
                f"Input mismatch — this script expects inputs: "
                f"{expected_meta}, but \"{key}\" is missing."
            )

    # Check for unexpected keys.
    for key in inputs:
        if key not in expected:
            raise ConfigError(
                f"Input mismatch — this script expects inputs: "
                f"{expected_meta}, but received unexpected key "
                f"\"{key}\"."
            )

    # Check type compatibility.
    for key, expected_type in expected.items():
        actual_val = inputs[key]
        # Unwrap Input instances.
        if isinstance(actual_val, Input):
            actual_val = actual_val.value
        actual_type = type(actual_val)
        # Allow int where float is expected.
        if expected_type is float and actual_type is int:
            continue
        if actual_type is not expected_type:
            raise ConfigError(
                f"Input type mismatch — \"{key}\" expects "
                f"{_TYPE_NAMES.get(expected_type, expected_type.__name__)} "
                f"but got {actual_type.__name__} ({actual_val!r})."
            )
