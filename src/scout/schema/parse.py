"""Layer 2: Schema parser.

Transforms the user's raw Python schema (dicts, lists, bare types,
``Field()``, ``List()``) into a normalized Node tree. This is the only
place where schema construction errors are raised.
"""

from __future__ import annotations

from typing import Any

from ..errors import ScoutSchemaError
from .nodes import FreestyleDictNode, ListNode, Node, ObjectNode, ScalarNode
from .types import Field, Items


def parse_schema(schema: Any) -> Node:
    """Parse a user schema into a Node tree.

    This is the main entry point for schema parsing. It accepts any valid
    schema definition and returns the corresponding Node tree.

    Raises:
        ScoutSchemaError: If the schema definition is invalid.
    """
    # Order matters: isinstance(dict_instance, dict) is True but we also
    # need to handle the dict *type* itself. Check isinstance first
    # (for dict instances used as object schemas), then check identity
    # (for bare `dict` used as freestyle dict type).

    if isinstance(schema, dict):
        return _parse_object(schema)

    if isinstance(schema, list):
        if len(schema) != 1:
            raise ScoutSchemaError(
                f"List schema must contain exactly one element "
                f"(the item schema), got {len(schema)}."
            )
        return ListNode(item=parse_schema(schema[0]))

    if isinstance(schema, Items):
        _validate_list_constraints(schema)
        return ListNode(
            item=parse_schema(schema.item),
            min=schema.min,
            max=schema.max,
            allow_empty=schema.allow_empty,
        )

    if isinstance(schema, Field):
        _validate_field_constraints(schema)
        return _field_to_scalar(schema)

    if schema in (str, int, float, bool):
        return ScalarNode(type_=schema)

    if schema is dict:
        return FreestyleDictNode()

    # ── Common mistakes with specific guidance ──

    if isinstance(schema, str):
        raise ScoutSchemaError(
            f"Schema must be a type (like str), not a string value.\n\n"
            f"  You passed: schema={schema!r}\n"
            f"  Did you mean: schema=str\n"
            f"  Or for a dict: schema={{'field_name': str}}"
        )

    if isinstance(schema, (int, float)):
        raise ScoutSchemaError(
            f"Schema must be a type (like {type(schema).__name__}), "
            f"not a value.\n\n"
            f"  You passed: schema={schema!r}\n"
            f"  Did you mean: schema={type(schema).__name__}"
        )

    if schema is list:
        raise ScoutSchemaError(
            "Schema cannot be bare 'list'. Specify the item type.\n\n"
            "  Examples:\n"
            "    schema=[{'title': str}]          # list of objects\n"
            "    schema=Items({'title': str})      # same, with constraints\n"
            "    schema=[str]                      # list of strings"
        )

    raise ScoutSchemaError(
        f"Invalid schema: expected a type (str, int, float, bool, dict), "
        f"Field(), Items(), dict, or list.\n\n"
        f"  Got: {type(schema).__name__}"
        + (f" value {schema!r}" if not callable(schema) else "")
        + "\n\n"
        f"  Quick examples:\n"
        f"    schema={{'title': str, 'price': float}}  # single object\n"
        f"    schema=[{{'title': str}}]                 # list of objects\n"
        f"    schema=Items({{'title': str}}, min=10)    # list with constraints"
    )


def _parse_object(schema: dict[str, Any]) -> ObjectNode:
    """Parse a dict schema into an ObjectNode."""
    fields: dict[str, tuple[Node, bool]] = {}
    for key, value in schema.items():
        if not isinstance(key, str):
            raise ScoutSchemaError(
                f"Field names must be strings, got {type(key).__name__}"
            )
        node, optional = _parse_field_value(value, key)
        fields[key] = (node, optional)
    return ObjectNode(fields=fields)


def _parse_field_value(value: Any, field_name: str) -> tuple[Node, bool]:
    """Parse a single field value within an object schema.

    Returns:
        ``(node, optional)`` — the parsed node and whether the field is optional.
    """
    if isinstance(value, Field):
        _validate_field_constraints(value)
        return _field_to_scalar(value), value.optional

    # For all other value types, the field is required.
    # Delegate to parse_schema for recursive parsing, but wrap errors
    # with the field name for context.
    try:
        node = parse_schema(value)
    except ScoutSchemaError:
        # Check if this is a raw Python value (like 42) that the user
        # accidentally used instead of a type
        if not isinstance(value, (type, dict, list, Items)):
            raise ScoutSchemaError(
                f"Invalid schema value for field {field_name!r}: expected a type "
                f"(str, int, float, bool, dict), Field(), Items(), dict, or list, "
                f"got {type(value).__name__} value {value!r}"
            ) from None
        raise
    return node, False


def _field_to_scalar(field: Field) -> ScalarNode:
    """Convert a validated Field to a ScalarNode."""
    return ScalarNode(
        type_=field.type_,
        optional=field.optional,
        min=field.min,
        max=field.max,
        min_length=field.min_length,
        max_length=field.max_length,
        pattern=field.pattern,
        enum=field.enum,
    )


# ---------------------------------------------------------------------------
# Constraint validation
# ---------------------------------------------------------------------------

def _validate_field_constraints(field: Field) -> None:
    """Validate that a Field's constraints are compatible with its type.

    Raises:
        ScoutSchemaError: If any constraint is invalid for the field type.
    """
    t = field.type_

    # Type must be one of the allowed base types
    if t not in (str, int, float, bool):
        if t is list:
            raise ScoutSchemaError(
                "Invalid type 'list' for Field. "
                "Use List() for list schemas, or [type] for inline list syntax."
            )
        raise ScoutSchemaError(
            f"Invalid type {t.__name__!r} for Field. "
            f"Allowed types: str, int, float, bool."
        )

    # Type-specific constraint compatibility
    if t == str:
        if field.min is not None or field.max is not None:
            # Build a suggested fix showing the corrected Field() call
            suggestions = []
            if field.min is not None:
                suggestions.append(f"min_length={field.min!r}")
            if field.max is not None:
                suggestions.append(f"max_length={field.max!r}")
            fix = f"Field(str, {', '.join(suggestions)})"
            raise ScoutSchemaError(
                f"'min'/'max' are not valid for str — "
                f"use 'min_length'/'max_length'.\n\n"
                f"  Did you mean: {fix}"
            )
        _validate_enum_constraints(field)
        _validate_string_length_constraints(field)
    else:
        # int, float, bool
        if field.min_length is not None or field.max_length is not None:
            raise ScoutSchemaError(
                f"'min_length'/'max_length' are not valid for {t.__name__}. "
                f"Use 'min'/'max'."
            )
        if field.pattern is not None:
            raise ScoutSchemaError(
                f"'pattern' is not valid for {t.__name__!r}. "
                f"'pattern' can only be used with 'str'."
            )
        if field.enum is not None:
            raise ScoutSchemaError(
                f"'enum' is not valid for {t.__name__!r}. "
                f"'enum' can only be used with 'str'."
            )
        _validate_numeric_range_constraints(field)


def _validate_enum_constraints(field: Field) -> None:
    """Validate enum constraints on a str field."""
    if field.enum is None:
        return

    if not isinstance(field.enum, list):
        raise ScoutSchemaError(
            f"'enum' must be a list of strings, got {type(field.enum).__name__}"
        )
    if len(field.enum) == 0:
        raise ScoutSchemaError("'enum' must not be empty")
    for i, val in enumerate(field.enum):
        if not isinstance(val, str):
            raise ScoutSchemaError(
                f"'enum' values must be strings, got {type(val).__name__} "
                f"at index {i}"
            )

    # enum is redundant with pattern/min_length/max_length
    if field.pattern is not None:
        raise ScoutSchemaError(
            "'enum' cannot be combined with 'pattern' — "
            "the allowed values are already known"
        )
    if field.min_length is not None or field.max_length is not None:
        raise ScoutSchemaError(
            "'enum' cannot be combined with 'min_length'/'max_length' — "
            "the allowed values are already known"
        )


def _validate_string_length_constraints(field: Field) -> None:
    """Validate min_length/max_length on a str field."""
    if field.min_length is not None and field.min_length < 0:
        raise ScoutSchemaError(
            f"'min_length' must be non-negative, got {field.min_length}"
        )
    if field.max_length is not None and field.max_length < 0:
        raise ScoutSchemaError(
            f"'max_length' must be non-negative, got {field.max_length}"
        )
    if (
        field.min_length is not None
        and field.max_length is not None
        and field.min_length > field.max_length
    ):
        raise ScoutSchemaError(
            f"'min_length' ({field.min_length}) must be <= "
            f"'max_length' ({field.max_length})"
        )


def _validate_numeric_range_constraints(field: Field) -> None:
    """Validate min/max on an int or float field."""
    if (
        field.min is not None
        and field.max is not None
        and field.min > field.max
    ):
        raise ScoutSchemaError(
            f"'min' ({field.min}) must be <= 'max' ({field.max})"
        )


def _validate_list_constraints(lst: Items) -> None:
    """Validate that Items constraints are internally consistent.

    Raises:
        ScoutSchemaError: If constraints are invalid.
    """
    if lst.min is not None and lst.min < 0:
        raise ScoutSchemaError(
            f"'min' must be non-negative, got {lst.min}"
        )
    if lst.max is not None and lst.max < 0:
        raise ScoutSchemaError(
            f"'max' must be non-negative, got {lst.max}"
        )
    if (
        lst.min is not None
        and lst.max is not None
        and lst.min > lst.max
    ):
        raise ScoutSchemaError(
            f"'min' ({lst.min}) must be <= 'max' ({lst.max})"
        )
