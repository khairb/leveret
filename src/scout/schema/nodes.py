"""Internal node types — the normalized intermediate representation.

The parser produces a tree of these nodes from the user's schema.
The prompt renderer and validator both consume this tree.
These types are internal — not exported to users.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


@dataclass(frozen=True, slots=True)
class ScalarNode:
    """A leaf node: str, int, float, or bool with optional constraints."""

    type_: type
    optional: bool = False
    min: float | None = None
    max: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    enum: list[str] | None = field(default=None, hash=False)


@dataclass(frozen=True, slots=True)
class ObjectNode:
    """A dict with known fields, each mapping to a child node.

    ``fields`` maps field_name → (node, optional). The optional flag
    is stored here (not on the child node) because it describes the
    field's presence in the parent object, not a property of the
    node type itself. For ScalarNode children, the optional flag
    is duplicated on both the field tuple and the ScalarNode for
    convenience — the parser ensures they agree.
    """

    fields: dict[str, tuple[Node, bool]]


@dataclass(frozen=True, slots=True)
class ListNode:
    """A list of items, each matching a child node."""

    item: Node
    min: int | None = None
    max: int | None = None


@dataclass(frozen=True, slots=True)
class FreestyleDictNode:
    """An unconstrained dict — any keys, any values."""


# Union of all node types.
Node = Union[ScalarNode, ObjectNode, ListNode, FreestyleDictNode]
