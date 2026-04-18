"""Layer 1: User-facing schema types.

These are the only two classes the user imports. They are pure data
containers — no logic, no validation. Validation of their parameters
happens in Layer 2 (parsing).
"""

from __future__ import annotations

from typing import Any, Union


class Field:
    """A schema field with type and optional constraints.

    The first positional argument is always the type. All constraints
    are keyword-only.

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


class List:
    """A list schema with item type and optional length constraints.

    The first positional argument is the item schema — a base type,
    a ``Field()``, a dict (object schema), or another ``List()``.

    Examples::

        List(str, min=5)                        # at least 5 strings
        List({"title": str}, min=20)            # at least 20 objects
        List(str, min=5, max=50)                # bounded list
    """

    __slots__ = ("item", "min", "max")

    def __init__(
        self,
        item: Any,
        *,
        min: int | None = None,
        max: int | None = None,
    ) -> None:
        self.item = item
        self.min = min
        self.max = max

    def __repr__(self) -> str:
        parts = [repr(self.item)]
        if self.min is not None:
            parts.append(f"min={self.min!r}")
        if self.max is not None:
            parts.append(f"max={self.max!r}")
        return f"List({', '.join(parts)})"


# Type alias for schema definitions accepted by the parser.
# This is a structural alias — isinstance() checks do not work with it.
SchemaType = Union[
    type,       # str, int, float, bool, dict
    Field,
    List,
    dict,       # {"key": SchemaType, ...}
    list,       # [SchemaType] — always exactly one element
]
