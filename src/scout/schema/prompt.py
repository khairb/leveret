"""Layer 3: Prompt renderer.

Walks the Node tree and produces the ``## Output Schema`` prompt section
with Structure (Python-like skeleton) and Requirements (natural language
bullets). Both are generated from the same tree, guaranteeing consistency.
"""

from __future__ import annotations

from .nodes import FreestyleDictNode, ListNode, Node, ObjectNode, ScalarNode


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_schema_prompt(root: Node) -> str:
    """Render the complete ``## Output Schema`` section for the agent prompt."""
    structure = render_structure(root)
    requirements = render_requirements(root)
    has_optional = _tree_has_optional(root)

    parts = [
        "## Output Schema",
        "",
        "Your scrape function must return data matching this schema. The return",
        "value is validated \u2014 if it doesn't match, your function will be rejected",
        "and you will receive the specific validation errors.",
    ]

    if has_optional:
        parts.extend([
            "",
            "For optional fields, return `None` when the value is not available.",
            "Always include the key \u2014 do not omit it.",
        ])

    parts.extend([
        "",
        "### Structure",
        "",
        structure,
        "",
        "### Requirements",
        "",
        requirements,
    ])

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Structure renderer
# ---------------------------------------------------------------------------

def render_structure(node: Node, indent: int = 0) -> str:
    """Render the ``### Structure`` skeleton."""
    pad = "    " * indent

    if isinstance(node, ScalarNode):
        return "..."

    if isinstance(node, FreestyleDictNode):
        return "{...}"

    if isinstance(node, ObjectNode):
        lines = ["{"]

        # Compute alignment column for inline comments.
        # Only inline fields (scalars + freestyle dicts) get comments.
        inline_fields = [
            (name, field_node, optional)
            for name, (field_node, optional) in node.fields.items()
            if isinstance(field_node, (ScalarNode, FreestyleDictNode))
        ]
        if inline_fields:
            # Each inline line looks like: <pad>    "<name>": ...,
            # The value is "..." (3 chars) or "{...}" (5 chars)
            max_prefix_len = max(
                len(f'"{name}": {"..." if isinstance(fn, ScalarNode) else "{{...}}"},')
                for name, fn, _ in inline_fields
            )
        else:
            max_prefix_len = 0

        for name, (field_node, optional) in node.fields.items():
            value_str = render_structure(field_node, indent + 1)
            if isinstance(field_node, (ScalarNode, FreestyleDictNode)):
                # Inline scalars and freestyle dicts on one line, column-aligned
                prefix = f'"{name}": {value_str},'
                padding = max_prefix_len - len(prefix) + 1
                comment = _build_comment(field_node, optional)
                lines.append(f'{pad}    {prefix}{" " * padding}{comment}')
            else:
                # Nested list or object — multiline, no inline comment
                lines.append(f'{pad}    "{name}": {value_str},')
        lines.append(f"{pad}}}")
        return "\n".join(lines)

    if isinstance(node, ListNode):
        inner = render_structure(node.item, indent + 1)
        continuation = _list_constraint_comment(node)
        lines = [
            "[",
            f"{pad}    {inner},",
            f"{pad}    ...{continuation}",
            f"{pad}]",
        ]
        return "\n".join(lines)

    return "..."  # pragma: no cover


# ---------------------------------------------------------------------------
# Requirements renderer
# ---------------------------------------------------------------------------

def render_requirements(node: Node, depth: int = 0) -> str:
    """Render the ``### Requirements`` bullet list."""
    indent = "  " * depth

    if isinstance(node, ListNode):
        item_desc = _describe_item_type(node.item)
        constraint = _list_constraint_text(node)
        lines = [f"{indent}- Return a **{item_desc}**{constraint}."]

        if isinstance(node.item, ObjectNode):
            lines.append(f"{indent}- Each object must have the following fields:")
            lines.extend(_render_object_fields(node.item, depth + 1))
        return "\n".join(lines)

    if isinstance(node, ObjectNode):
        lines = [f"{indent}- Return an **object** with the following fields:"]
        lines.extend(_render_object_fields(node, depth + 1))
        return "\n".join(lines)

    if isinstance(node, ScalarNode):
        article = "an" if node.type_ == int else "a"
        type_name = _BOLD_TYPE_NAMES[node.type_]
        constraint = _scalar_constraint_text(node)
        req = f"Required. {constraint}" if constraint else "Required."
        return f"{indent}- Return {article} {type_name}. {req}"

    if isinstance(node, FreestyleDictNode):
        return (
            f"{indent}- Return a **freestyle object**. "
            f"Extract whatever key-value pairs the page provides."
        )

    return ""  # pragma: no cover


# ---------------------------------------------------------------------------
# Structure helpers
# ---------------------------------------------------------------------------

_TYPE_NAMES = {str: "str", int: "int", float: "float", bool: "bool"}


def _build_comment(node: Node, optional: bool) -> str:
    """Build the ``# type, required/optional, constraints`` comment."""
    if isinstance(node, FreestyleDictNode):
        return "# dict, freestyle"

    if not isinstance(node, ScalarNode):
        # Nested list/object — no inline comment
        return ""

    parts: list[str] = [_TYPE_NAMES[node.type_]]

    if optional:
        parts.append("optional")
    else:
        parts.append("required")

    # Constraints
    parts.extend(_scalar_constraint_parts(node))

    return "# " + ", ".join(parts)


def _scalar_constraint_parts(node: ScalarNode) -> list[str]:
    """Constraint fragments for Structure comments."""
    parts: list[str] = []

    if node.type_ == str:
        if node.enum is not None:
            vals = ", ".join(f'"{v}"' for v in node.enum)
            parts.append(f"one of: {vals}")
        if node.min_length is not None and node.max_length is not None:
            parts.append(f"{node.min_length} to {node.max_length} chars")
        elif node.min_length is not None:
            parts.append(f"min length: {node.min_length}")
        elif node.max_length is not None:
            parts.append(f"max length: {node.max_length}")
        if node.pattern is not None:
            parts.append(f"pattern: {node.pattern}")

    elif node.type_ in (int, float):
        if node.min is not None and node.max is not None:
            parts.append(f"{node.min} to {node.max}")
        elif node.min is not None:
            parts.append(f">= {node.min}")
        elif node.max is not None:
            parts.append(f"<= {node.max}")

    return parts


def _list_constraint_comment(node: ListNode) -> str:
    """Comment for the ``...`` continuation line in a list."""
    if node.min is not None and node.max is not None:
        s = "item" if node.min == 1 and node.max == 1 else "items"
        return f"                       # {node.min} to {node.max} {s}"
    if node.min is not None:
        s = "item" if node.min == 1 else "items"
        return f"                       # minimum {node.min} {s}"
    if node.max is not None:
        s = "item" if node.max == 1 else "items"
        return f"                       # maximum {node.max} {s}"
    return ""


# ---------------------------------------------------------------------------
# Requirements helpers
# ---------------------------------------------------------------------------

_PLURAL_TYPE_NAMES = {
    str: "strings", int: "integers", float: "floats", bool: "booleans",
}


def _describe_item_type(node: Node) -> str:
    """Describe the item type for the top-level Requirements line."""
    if isinstance(node, ObjectNode):
        return "list of objects"
    if isinstance(node, ScalarNode):
        return f"list of {_PLURAL_TYPE_NAMES[node.type_]}"
    if isinstance(node, FreestyleDictNode):
        return "list of objects"
    return "list"  # pragma: no cover


def _list_constraint_text(node: ListNode) -> str:
    """Constraint text for the Requirements list description."""
    if node.min is not None and node.max is not None:
        s = "item" if node.min == 1 and node.max == 1 else "items"
        return f" with **{node.min} to {node.max} {s}**"
    if node.min is not None:
        s = "item" if node.min == 1 else "items"
        return f" with at least **{node.min} {s}**"
    if node.max is not None:
        s = "item" if node.max == 1 else "items"
        return f" with at most **{node.max} {s}**"
    return ""


def _render_object_fields(node: ObjectNode, depth: int) -> list[str]:
    """Render bullet points for each field in an object."""
    indent = "  " * depth
    lines: list[str] = []

    for name, (field_node, optional) in node.fields.items():
        lines.append(_render_field_bullet(name, field_node, optional, indent))

        # Recurse into nested structures
        if isinstance(field_node, ListNode) and isinstance(field_node.item, ObjectNode):
            lines.append(f"{indent}  - Each object in `{name}` must have:")
            lines.extend(_render_object_fields(field_node.item, depth + 2))
        elif isinstance(field_node, ObjectNode):
            lines.extend(_render_object_fields(field_node, depth + 1))

    return lines


def _render_field_bullet(
    name: str, node: Node, optional: bool, indent: str
) -> str:
    """Render a single field bullet point."""
    if isinstance(node, FreestyleDictNode):
        req = "Required" if not optional else "Optional"
        return (
            f"{indent}- `{name}` \u2014 a **freestyle object**. {req}. "
            f"Extract whatever key-value pairs the page provides."
        )

    if isinstance(node, ListNode):
        item_desc = _describe_nested_list_type(node.item)
        constraint = _list_constraint_text(node)
        req = "Required" if not optional else "Optional"
        return f"{indent}- `{name}` \u2014 a **{item_desc}**{constraint}. {req}."

    if isinstance(node, ObjectNode):
        req = "Required" if not optional else "Optional"
        return f"{indent}- `{name}` \u2014 an **object**. {req}."

    if isinstance(node, ScalarNode):
        return _render_scalar_bullet(name, node, optional, indent)

    return f"{indent}- `{name}`"  # pragma: no cover


def _describe_nested_list_type(node: Node) -> str:
    """Describe a nested list's item type."""
    if isinstance(node, ObjectNode):
        return "list of objects"
    if isinstance(node, ScalarNode):
        return f"list of {_PLURAL_TYPE_NAMES[node.type_]}"
    return "list"


def _render_scalar_bullet(
    name: str, node: ScalarNode, optional: bool, indent: str
) -> str:
    """Render a scalar field bullet with type, required/optional, constraints."""
    type_article = "an" if node.type_ == int else "a"
    type_name = _BOLD_TYPE_NAMES[node.type_]

    if optional:
        req_part = "Optional \u2014 `None` if not available."
        constraint_text = _scalar_constraint_text(node)
        if constraint_text:
            req_part += f" If present, {_lowercase_first(constraint_text)}"
    else:
        req_part = "Required."
        constraint_text = _scalar_constraint_text(node)
        if constraint_text:
            req_part += f" {constraint_text}"

    return f"{indent}- `{name}` \u2014 {type_article} {type_name}. {req_part}"


_BOLD_TYPE_NAMES = {
    str: "**string**",
    int: "**integer**",
    float: "**float**",
    bool: "**boolean**",
}


def _scalar_constraint_text(node: ScalarNode) -> str:
    """Natural English constraint description for Requirements."""
    parts: list[str] = []

    if node.type_ == str:
        if node.enum is not None:
            vals = ", ".join(f'"{v}"' for v in node.enum)
            parts.append(f'Must be one of: {vals}.')
        if node.min_length is not None and node.max_length is not None:
            parts.append(
                f"Between {node.min_length} and {node.max_length} characters."
            )
        elif node.min_length is not None:
            char = "character" if node.min_length == 1 else "characters"
            parts.append(f"At least {node.min_length} {char}.")
        elif node.max_length is not None:
            char = "character" if node.max_length == 1 else "characters"
            parts.append(f"At most {node.max_length} {char}.")
        if node.pattern is not None:
            parts.append(f"Must match pattern `{node.pattern}`.")

    elif node.type_ in (int, float):
        if node.min is not None and node.max is not None:
            parts.append(f"Between {node.min} and {node.max}.")
        elif node.min is not None:
            parts.append(f"Must be >= {node.min}.")
        elif node.max is not None:
            parts.append(f"Must be <= {node.max}.")

    return " ".join(parts)


def _lowercase_first(s: str) -> str:
    """Lowercase the first character of a string."""
    if not s:
        return s
    return s[0].lower() + s[1:]


# ---------------------------------------------------------------------------
# Tree traversal helpers
# ---------------------------------------------------------------------------

def _tree_has_optional(node: Node) -> bool:
    """Check if any ScalarNode in the tree has optional=True."""
    if isinstance(node, ScalarNode):
        return node.optional

    if isinstance(node, ObjectNode):
        for field_node, optional in node.fields.values():
            if optional:
                return True
            if _tree_has_optional(field_node):
                return True
        return False

    if isinstance(node, ListNode):
        return _tree_has_optional(node.item)

    return False
