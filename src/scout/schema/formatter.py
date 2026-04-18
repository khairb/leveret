"""Layer 5: Error formatter.

Takes the flat list of ``RawError``s from Layer 4 and produces the
final agent-friendly string. Handles grouping, sorting, capping, and
value display.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .validate import RawError


# Sort priority — lower number = shown first.
_KIND_ORDER = {"type": 0, "missing": 0, "null": 1, "structure": 2, "constraint": 3}


@dataclass(slots=True)
class ErrorGroup:
    """A group of similar errors across list items."""

    pattern: str
    """Path with list indices replaced by ``*``, e.g. ``"items[*].price"``."""

    message: str
    """The error message shared by all items in this group."""

    error_kind: str
    """Category for sorting (type, missing, null, structure, constraint)."""

    count: int
    """How many items have this error."""

    examples: list[tuple[str, Any]] = field(default_factory=list)
    """Up to 3 ``(concrete_path, value)`` examples."""


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def group_errors(errors: list[RawError]) -> list[ErrorGroup]:
    """Group errors by pattern path + message."""
    buckets: dict[tuple[str, str], ErrorGroup] = {}

    for err in errors:
        # Replace [0], [1], [42] with [*] for grouping
        pattern = re.sub(r"\[\d+\]", "[*]", err.path)

        key = (pattern, err.message)

        if key not in buckets:
            buckets[key] = ErrorGroup(
                pattern=pattern,
                message=err.message,
                error_kind=err.error_kind,
                count=0,
                examples=[],
            )

        group = buckets[key]
        group.count += 1
        if len(group.examples) < 3:
            group.examples.append((err.path, err.value))

    return list(buckets.values())


# ---------------------------------------------------------------------------
# Value display
# ---------------------------------------------------------------------------

def _display_value(value: Any) -> str:
    """Format a value for display in error messages."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        truncated = value[:50] + "..." if len(value) > 50 else value
        return f'"{truncated}"'
    if isinstance(value, list):
        return f"<list with {len(value)} items>"
    if isinstance(value, dict):
        return f"<object with {len(value)} keys>"
    return repr(value)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_errors(
    errors: list[RawError],
    total_items: int | None = None,
    node_tree: Any = None,
) -> str:
    """Format validation errors into the agent-facing message.

    Args:
        errors: Flat list of RawErrors from the validator.
        total_items: Total number of items in the top-level list
            (for "N of M" display).
        node_tree: The root Node tree, used to detect optional fields
            for the constraint-violation note. Optional.

    Returns:
        Complete formatted error message ready for the agent.
    """
    if not errors:
        return ""

    groups = group_errors(errors)
    groups.sort(key=lambda g: (_KIND_ORDER.get(g.error_kind, 9), -g.count))

    # Cap at 10 groups
    shown = groups[:10]
    remaining = len(groups) - 10 if len(groups) > 10 else 0

    type_word = "error" if len(groups) == 1 else "error types"
    header_extra = ", showing first 10" if remaining else ""
    lines = [f"Schema validation failed ({len(groups)} {type_word}{header_extra}):\n"]

    optional_paths = _collect_optional_paths(node_tree) if node_tree else set()

    for i, group in enumerate(shown, 1):
        # Path display: use concrete path if only 1 item affected
        path = group.examples[0][0] if group.count == 1 else group.pattern

        # Count context
        count_ctx = _count_context(group, total_items)

        # Top-level errors have empty path — capitalize the message directly
        if path:
            lines.append(f"  [{i}] {path} \u2014 {group.message}{count_ctx}")
        else:
            # Capitalize first letter for top-level messages
            msg = group.message[0].upper() + group.message[1:] if group.message else ""
            lines.append(f"  [{i}] {msg}{count_ctx}")

        # Examples (only if values are not None AND path is not empty)
        if group.examples and group.examples[0][1] is not None and path:
            example_parts = [
                f"{ex_path} = {_display_value(ex_value)}"
                for ex_path, ex_value in group.examples
            ]
            if len(example_parts) == 1:
                lines.append(f"      Examples: {example_parts[0]}")
            elif len(example_parts) == 2:
                lines.append(
                    f"      Examples: {example_parts[0]}, {example_parts[1]}"
                )
            else:
                # First two comma-joined, third on continuation line
                lines.append(
                    f"      Examples: {example_parts[0]},"
                )
                lines.append(f"                {example_parts[1]},")
                lines.append(f"                {example_parts[2]}")

        # Optional field note for constraint violations
        if (
            group.error_kind == "constraint"
            and _pattern_is_optional(group.pattern, optional_paths)
        ):
            bad_vals = list(dict.fromkeys(
                _display_value(val) for _, val in group.examples
                if val is not None
            ))[:3]  # deduplicate, preserve order, cap at 3
            if bad_vals:
                joined = " and ".join(bad_vals) if len(bad_vals) <= 2 else \
                    ", ".join(bad_vals[:-1]) + ", and " + bad_vals[-1]
                lines.append(
                    f"      Note: this field is optional \u2014 null is valid, "
                    f"but {joined} {'is' if len(bad_vals) == 1 else 'are'} not."
                )
            else:
                lines.append(
                    f"      Note: this field is optional \u2014 null is valid, "
                    f"but the above values are not."
                )

    if remaining:
        lines.append(f"\n  ... and {remaining} more error types not shown.")

    return "\n".join(lines)


def _count_context(group: ErrorGroup, total_items: int | None) -> str:
    """Build the count context string for an error group."""
    if "[*]" in group.pattern:
        # List item error — always show count.
        # "N of M items" — always plural because we're describing the set.
        if total_items is not None:
            return f" ({group.count} of {total_items} items)"
        s = "item" if group.count == 1 else "items"
        return f" ({group.count} {s})"
    if group.count == 1:
        # Non-list, single occurrence — omit
        return ""
    return f" ({group.count} occurrences)"


# ---------------------------------------------------------------------------
# Optional field detection
# ---------------------------------------------------------------------------

def _collect_optional_paths(node: Any, prefix: str = "") -> set[str]:
    """Collect all paths to optional fields in the tree, with [*] for lists."""
    from .nodes import ListNode, ObjectNode, ScalarNode

    paths: set[str] = set()

    if isinstance(node, ObjectNode):
        for name, (field_node, optional) in node.fields.items():
            field_path = f"{prefix}.{name}" if prefix else name
            if optional:
                paths.add(field_path)
            paths.update(_collect_optional_paths(field_node, field_path))

    elif isinstance(node, ListNode):
        list_path = f"{prefix}[*]" if prefix else "[*]"
        paths.update(_collect_optional_paths(node.item, list_path))

    return paths


def _pattern_is_optional(pattern: str, optional_paths: set[str]) -> bool:
    """Check if a grouped error pattern corresponds to an optional field."""
    return pattern in optional_paths
