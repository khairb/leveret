"""Schema compiler — ties all layers together.

The ``compile_schema()`` function is the main entry point. Call it once
at ``Scraper()`` construction time. It parses the schema, renders the
prompt section, and returns a ``CompiledSchema`` that can validate data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .formatter import format_errors
from .nodes import Node
from .parse import parse_schema
from .prompt import render_schema_prompt
from .tolerance import Tolerance, apply_tolerance
from .validate import validate


@dataclass(slots=True)
class CompiledSchema:
    """A fully compiled schema ready for prompt injection and validation.

    Attributes:
        root: The parsed Node tree (for introspection).
        prompt: The rendered ``## Output Schema`` section for the agent prompt.
    """

    root: Node
    prompt: str

    def validate(
        self, data: Any, tolerance: Tolerance | None = None,
    ) -> tuple[bool, str]:
        """Validate data against this schema.

        Args:
            data: The data to validate.
            tolerance: Optional tolerance level for list validation.
                When set to ``BALANCED`` or ``TOLERANT``, per-item
                errors are filtered if enough items pass. When
                ``None`` or ``STRICT``, all errors are kept.

        Returns:
            ``(True, "")`` if the data matches the schema.
            ``(False, error_message)`` if validation fails — the error
            message is fully formatted and ready to be included in the
            rejection feedback to the agent.
        """
        errors = validate(data, self.root)
        if not errors:
            return True, ""

        if tolerance is not None:
            errors = apply_tolerance(errors, self.root, data, tolerance)
            if not errors:
                return True, ""

        total_items = len(data) if isinstance(data, list) else None
        return False, format_errors(errors, total_items, node_tree=self.root)


def compile_schema(schema: Any) -> CompiledSchema:
    """Compile a user schema into a CompiledSchema.

    This is the main entry point. Call once at ``Scraper()`` construction time.

    Args:
        schema: The user's schema — a dict, list, ``Field()``, ``List()``,
            or bare type.

    Returns:
        A ``CompiledSchema`` with the rendered prompt and validation capability.

    Raises:
        ScoutSchemaError: If the schema definition is invalid.
    """
    root = parse_schema(schema)
    prompt = render_schema_prompt(root)
    return CompiledSchema(root=root, prompt=prompt)
