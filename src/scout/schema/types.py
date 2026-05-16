"""Layer 1: User-facing schema types.

These are the only two classes the user imports. They are pure data
containers — no logic, no validation. Validation of their parameters
happens in Layer 2 (parsing).
"""

from __future__ import annotations

import warnings
from typing import Any, Union


class Field:
    """A schema field with type and optional constraints.

    Use ``Field`` when you need more control than a bare type. The
    first positional argument is always the Python type (``str``,
    ``int``, ``float``, or ``bool``). All constraints are keyword-only.

    Args:
        type_: Base Python type — ``str``, ``int``, ``float``, or ``bool``.
        min: Minimum numeric value (``int``/``float`` fields only).
        max: Maximum numeric value (``int``/``float`` fields only).
        min_length: Minimum string length (``str`` fields only).
        max_length: Maximum string length (``str`` fields only).
        pattern: Regex pattern the string must match (``str`` only).
        enum: List of allowed string values (``str`` fields only).
        optional: When ``True``, the field may be ``None``.

    Examples::

        Field(str)                              # required string
        Field(str, min_length=10)               # min string length
        Field(float, min=0)                     # non-negative float
        Field(int, min=1, max=5)                # bounded integer
        Field(str, enum=["a", "b", "c"])        # allowed values
        Field(str, optional=True)               # nullable field
    """

    __slots__ = (
        "type_", "min", "max", "min_length", "max_length",
        "pattern", "enum", "optional",
    )

    def __init__(
        self,
        type_: type,
        *,
        min: float | None = None,
        max: float | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        pattern: str | None = None,
        enum: list[str] | None = None,
        optional: bool = False,
    ) -> None:
        self.type_ = type_
        self.min = min
        self.max = max
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = pattern
        self.enum = enum
        self.optional = optional

    def __repr__(self) -> str:
        parts = [self.type_.__name__]
        if self.min is not None:
            parts.append(f"min={self.min!r}")
        if self.max is not None:
            parts.append(f"max={self.max!r}")
        if self.min_length is not None:
            parts.append(f"min_length={self.min_length!r}")
        if self.max_length is not None:
            parts.append(f"max_length={self.max_length!r}")
        if self.pattern is not None:
            parts.append(f"pattern={self.pattern!r}")
        if self.enum is not None:
            parts.append(f"enum={self.enum!r}")
        if self.optional:
            parts.append("optional=True")
        return f"Field({', '.join(parts)})"


class Items:
    """A list schema with item type and optional count constraints.

    Use ``Items`` when the scraper should return a list of results.
    The first positional argument is the item schema — a base type,
    a ``Field()``, a dict (object schema), or another ``Items()``
    for nested lists.

    Args:
        item: Schema for each item in the list — a type, ``Field``,
            dict, or nested ``Items``.
        min_items: Minimum number of items the list must contain.
        max_items: Maximum number of items the list may contain.
        allow_empty: When ``True``, an empty list passes validation
            even without an explicit ``min_items=0``.

    Examples::

        Items(str, min_items=5)                       # at least 5 strings
        Items({"title": str}, min_items=20)           # at least 20 objects
        Items(str, min_items=5, max_items=50)         # bounded list
    """

    __slots__ = ("item", "min_items", "max_items", "allow_empty")

    def __init__(
        self,
        item: Any,
        *,
        min_items: int | None = None,
        max_items: int | None = None,
        allow_empty: bool = False,
    ) -> None:
        self.item = item
        self.min_items = min_items
        self.max_items = max_items
        self.allow_empty = allow_empty

    def __repr__(self) -> str:
        parts = [repr(self.item)]
        if self.min_items is not None:
            parts.append(f"min_items={self.min_items!r}")
        if self.max_items is not None:
            parts.append(f"max_items={self.max_items!r}")
        if self.allow_empty:
            parts.append("allow_empty=True")
        return f"Items({', '.join(parts)})"


class _DeprecatedList:
    """Wrapper that emits a deprecation warning when List() is called."""

    def __call__(self, *args: Any, **kwargs: Any) -> Items:
        warnings.warn(
            "List is deprecated — use Items instead. "
            "List will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        return Items(*args, **kwargs)

    def __instancecheck__(self, instance: Any) -> bool:
        return isinstance(instance, Items)


List = _DeprecatedList()


# Type alias for schema definitions accepted by the parser.
# This is a structural alias — isinstance() checks do not work with it.
SchemaType = Union[
    type,       # str, int, float, bool, dict
    Field,
    Items,
    dict,       # {"key": SchemaType, ...}
    list,       # [SchemaType] — always exactly one element
]
