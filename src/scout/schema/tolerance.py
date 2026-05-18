"""Tolerance system for list validation.

Tolerance controls what percentage of list items must pass validation
for the overall result to be accepted. Failed items stay in the data
as-is — the user handles post-processing.

This is runtime policy, not a schema property: the same schema can be
validated with different tolerance levels.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from .nodes import ListNode, Node, ObjectNode
from .validate import RawError


class Tolerance(Enum):
    """How strictly to enforce per-item validation in lists.

    Web scraping often returns slightly inconsistent data — a few
    items may be missing a field or have an unexpected value. Tolerance
    controls what percentage of list items must pass schema validation
    for the overall result to be accepted.

    Members:
        STRICT: 100% of items must pass validation.
        BALANCED: 80% of items must pass (default).
        LENIENT: 50% of items must pass.
        TOLERANT: Deprecated alias for LENIENT.
    """

    STRICT = "strict"
    BALANCED = "balanced"
    LENIENT = "lenient"
    TOLERANT = "tolerant"


# Minimum fraction of items that must pass validation.
TOLERANCE_THRESHOLDS: dict[Tolerance, float] = {
    Tolerance.STRICT: 1.0,
    Tolerance.BALANCED: 0.8,
    Tolerance.LENIENT: 0.5,
    Tolerance.TOLERANT: 0.5,
}


def apply_tolerance(
    errors: list[RawError],
    root: Node,
    data: Any,
    tolerance: Tolerance,
) -> list[RawError]:
    """Filter validation errors based on tolerance policy.

    Walks the schema tree to find all ``ListNode`` positions, then for
    each list checks whether enough items passed validation. If the
    pass rate meets the threshold, per-item errors for that list are
    removed. Structural errors (count violations) are never filtered.

    Processes nested lists deepest-first so inner tolerance is applied
    before outer.

    Args:
        errors: Raw errors from ``validate()``.
        root: The parsed Node tree.
        data: The data that was validated.
        tolerance: The tolerance level to apply.

    Returns:
        Filtered list of errors (may be empty if all tolerated).
    """
    if tolerance is Tolerance.STRICT or not errors:
        return errors

    threshold = TOLERANCE_THRESHOLDS[tolerance]

    # Collect all list positions in the schema (path → ListNode).
    list_positions = _find_list_positions(root, data, "")

    # Process deepest-first (longest paths first) so inner lists
    # are resolved before outer lists.
    list_positions.sort(key=lambda pos: -len(pos[0]))

    for list_path, list_len in list_positions:
        if list_len == 0:
            continue  # No items to tolerate

        # Partition errors and collect failed indices in one pass.
        other_errors, failed = _partition_and_count(errors, list_path)

        if not failed:
            continue  # No per-item errors for this list

        pass_rate = (list_len - len(failed)) / list_len

        if pass_rate >= threshold:
            # Tolerance met — remove per-item errors, keep structural
            errors = other_errors

    return errors


def _find_list_positions(
    node: Node,
    data: Any,
    path: str,
) -> list[tuple[str, int]]:
    """Find all ListNode positions in the tree with their item counts.

    Returns a list of ``(path, item_count)`` tuples.
    """
    positions: list[tuple[str, int]] = []

    if isinstance(node, ListNode):
        items = data if isinstance(data, list) else []
        positions.append((path, len(items)))
        # Only recurse into items if the item schema can contain lists.
        if isinstance(node.item, (ListNode, ObjectNode)):
            for i, item in enumerate(items):
                item_path = f"{path}[{i}]" if path else f"[{i}]"
                positions.extend(_find_list_positions(node.item, item, item_path))

    elif isinstance(node, ObjectNode):
        if isinstance(data, dict):
            for field_name, (field_node, _optional) in node.fields.items():
                field_path = f"{path}.{field_name}" if path else field_name
                field_data = data.get(field_name)
                positions.extend(_find_list_positions(field_node, field_data, field_path))

    return positions


def _partition_and_count(
    errors: list[RawError],
    list_path: str,
) -> tuple[list[RawError], set[int]]:
    """Split errors and collect failed indices in a single pass.

    Returns ``(other_errors, failed_indices)`` where:
    - ``other_errors`` are errors NOT belonging to items of this list
      (structural errors, errors from other paths).
    - ``failed_indices`` are the set of item indices that have errors.
    """
    if list_path:
        pattern = re.compile(re.escape(list_path) + r"\[(\d+)\]")
    else:
        pattern = re.compile(r"^\[(\d+)\]")

    other_errors: list[RawError] = []
    failed: set[int] = set()

    for err in errors:
        m = pattern.search(err.path)
        if m:
            failed.add(int(m.group(1)))
        else:
            other_errors.append(err)

    return other_errors, failed
