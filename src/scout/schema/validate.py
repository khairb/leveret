"""Layer 4: Schema validator.

Recursively validates data against a Node tree, collecting errors with
hierarchical short-circuiting. No external libraries — just recursive
type checking and constraint checking.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from .nodes import FreestyleDictNode, ListNode, Node, ObjectNode, ScalarNode


@dataclass(slots=True)
class RawError:
    """A single validation error before grouping.

    Attributes:
        path: Dot-notation path like ``"items[0].price"`` or ``""``.
        message: Human-readable error like ``"expected float, got string"``.
        value: The actual value (for display in error examples).
        error_kind: Category for sorting — ``"type"``, ``"missing"``,
            ``"null"``, ``"constraint"``, ``"structure"``.
    """

    path: str
    message: str
    value: Any = field(default=None, repr=False)
    error_kind: str = ""


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------


def validate(data: Any, node: Node, path: str = "") -> list[RawError]:
    """Validate data against a Node tree.

    Returns a list of errors (empty = valid).
    """
    if isinstance(node, ListNode):
        return _validate_list(data, node, path)
    if isinstance(node, ObjectNode):
        return _validate_object(data, node, path)
    if isinstance(node, ScalarNode):
        return _validate_scalar(data, node, path)
    if isinstance(node, FreestyleDictNode):
        return _validate_freestyle_dict(data, node, path)
    return []  # pragma: no cover


# ---------------------------------------------------------------------------
# List validation
# ---------------------------------------------------------------------------


def _validate_list(data: Any, node: ListNode, path: str) -> list[RawError]:
    errors: list[RawError] = []

    # Tier 1: type check
    if not isinstance(data, list):
        if data is None:
            return [
                RawError(
                    path,
                    "expected a list, got null. Your function returned None "
                    "\u2014 make sure it has an explicit return statement.",
                    data,
                    "type",
                )
            ]
        return [RawError(path, f"expected a list, got {_type_name(data)}", data, "type")]

    # Tier 2: empty list with min constraint — short-circuit completely
    if len(data) == 0 and node.min is not None and node.min > 0:
        s = "item" if node.min == 1 else "items"
        return [RawError(path, f"list is empty, minimum is {node.min} {s}", None, "structure")]

    # Tier 2: count constraints (non-empty — report but continue)
    if node.min is not None and len(data) < node.min:
        errors.append(
            RawError(
                path,
                f"returned {len(data)} {'item' if len(data) == 1 else 'items'}, "
                f"minimum is {node.min}",
                None,
                "structure",
            )
        )
    if node.max is not None and len(data) > node.max:
        errors.append(
            RawError(
                path,
                f"returned {len(data)} items, maximum is {node.max}",
                None,
                "structure",
            )
        )

    # Tier 3+: validate each item (cap at first 50 for over-max lists)
    over_max = node.max is not None and len(data) > node.max
    validate_count = min(len(data), 50) if over_max else len(data)
    for i in range(validate_count):
        item_path = f"{path}[{i}]" if path else f"[{i}]"
        errors.extend(validate(data[i], node.item, item_path))

    return errors


# ---------------------------------------------------------------------------
# Object validation
# ---------------------------------------------------------------------------


def _validate_object(data: Any, node: ObjectNode, path: str) -> list[RawError]:
    errors: list[RawError] = []

    # Tier 1: type check
    if not isinstance(data, dict):
        return [RawError(path, f"expected an object, got {_type_name(data)}", data, "type")]

    for field_name, (field_node, optional) in node.fields.items():
        field_path = f"{path}.{field_name}" if path else field_name

        # Missing field
        if field_name not in data:
            if not optional:
                errors.append(RawError(field_path, "missing required field", None, "missing"))
            continue

        value = data[field_name]

        # Null value
        if value is None:
            if not optional:
                errors.append(
                    RawError(field_path, "field is null, but it is required", None, "null")
                )
            continue  # None for optional → valid, skip constraints

        # Field exists and is not null — validate recursively
        errors.extend(validate(value, field_node, field_path))

    # Extra fields are silently ignored — never an error
    return errors


# ---------------------------------------------------------------------------
# Scalar validation
# ---------------------------------------------------------------------------


def _validate_scalar(data: Any, node: ScalarNode, path: str) -> list[RawError]:
    errors: list[RawError] = []

    # Type check with lenient coercion
    if not _check_type(data, node.type_):
        return [
            RawError(
                path,
                f"expected {node.type_.__name__}, got {_type_name(data)}",
                data,
                "type",
            )
        ]

    # Type is correct — check all constraints independently
    if node.type_ == str:
        if node.enum is not None and data not in node.enum:
            vals = ", ".join(f'"{v}"' for v in node.enum)
            errors.append(
                RawError(
                    path,
                    f'value "{_truncate(data)}" is not one of the allowed values: {vals}',
                    data,
                    "constraint",
                )
            )
        if node.min_length is not None and len(data) < node.min_length:
            errors.append(
                RawError(
                    path,
                    f"string has {len(data)} characters, minimum is {node.min_length}",
                    data,
                    "constraint",
                )
            )
        if node.max_length is not None and len(data) > node.max_length:
            errors.append(
                RawError(
                    path,
                    f"string has {len(data)} characters, maximum is {node.max_length}",
                    data,
                    "constraint",
                )
            )
        if node.pattern is not None and not re.search(node.pattern, data):
            errors.append(
                RawError(
                    path,
                    f'string "{_truncate(data)}" does not match pattern {node.pattern}',
                    data,
                    "constraint",
                )
            )

    elif node.type_ in (int, float):
        # Coerce int → float for comparison on float fields
        val = float(data) if node.type_ == float and isinstance(data, int) else data
        too_low = node.min is not None and val < node.min
        too_high = node.max is not None and val > node.max
        if too_low or too_high:
            if node.min is not None and node.max is not None:
                # Both bounds defined — use "between" phrasing
                errors.append(
                    RawError(
                        path,
                        f"value is {val}, must be between {node.min} and {node.max}",
                        val,
                        "constraint",
                    )
                )
            elif too_low:
                errors.append(
                    RawError(
                        path,
                        f"value is {val}, must be >= {node.min}",
                        val,
                        "constraint",
                    )
                )
            else:
                errors.append(
                    RawError(
                        path,
                        f"value is {val}, must be <= {node.max}",
                        val,
                        "constraint",
                    )
                )

    return errors


# ---------------------------------------------------------------------------
# Freestyle dict validation
# ---------------------------------------------------------------------------


def _validate_freestyle_dict(data: Any, node: FreestyleDictNode, path: str) -> list[RawError]:
    if not isinstance(data, dict):
        return [RawError(path, f"expected an object, got {_type_name(data)}", data, "type")]
    return []


# ---------------------------------------------------------------------------
# Type checking
# ---------------------------------------------------------------------------


def _check_type(data: Any, expected: type) -> bool:
    """Check if data matches the expected type, with lenient coercion.

    Coercion rules:
    - int fields: accept int. Accept float if no fractional part (42.0 → ok).
    - float fields: accept float and int (widening).
    - bool fields: accept bool only. Reject 0, 1, "true", etc.
    - str fields: accept str only.
    """
    if expected == bool:
        return isinstance(data, bool)  # Must check before int — bool is subclass of int
    if expected == int:
        if isinstance(data, bool):
            return False
        if isinstance(data, int):
            return True
        if isinstance(data, float):
            # 42.0 → acceptable as int, but inf/nan are not
            if math.isfinite(data) and data == int(data):
                return True
        return False
    if expected == float:
        if isinstance(data, bool):
            return False
        return isinstance(data, (int, float))
    if expected == str:
        return isinstance(data, str)
    return False


def _type_name(data: Any) -> str:
    """Human-readable type name for error messages."""
    if data is None:
        return "null"
    if isinstance(data, bool):
        return "boolean"
    if isinstance(data, int):
        return "integer"
    if isinstance(data, float):
        return "float"
    if isinstance(data, str):
        return "string"
    if isinstance(data, list):
        return f"list with {len(data)} items"
    if isinstance(data, dict):
        return f"object with {len(data)} keys"
    return type(data).__name__


def _truncate(s: str, max_len: int = 50) -> str:
    """Truncate a string for display."""
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s
