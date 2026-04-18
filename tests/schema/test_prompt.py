"""Tests for Layer 3: Prompt renderer.

These tests validate not just that the code runs, but that the output
is clear, correct, and useful for an AI agent. Every test checks the
actual rendered text that an agent would read.
"""

import pytest

from scout.schema.compiler import compile_schema
from scout.schema.parse import parse_schema
from scout.schema.prompt import (
    render_requirements,
    render_schema_prompt,
    render_structure,
)
from scout.schema.types import Field, List


class TestStructureRendering:
    """The Structure section shows a Python-like skeleton."""

    def test_simple_object_list(self):
        root = parse_schema([{"title": str, "url": str, "points": int}])
        s = render_structure(root)
        assert '"title": ...,' in s
        assert '"url": ...,' in s
        assert '"points": ...,' in s
        assert "# str, required" in s
        assert "# int, required" in s

    def test_freestyle_dict_renders_as_braces(self):
        root = parse_schema({"specs": dict})
        s = render_structure(root)
        assert '"specs": {...},' in s
        assert "# dict, freestyle" in s

    def test_nested_list_shows_dots_continuation(self):
        root = parse_schema({"tags": [str]})
        s = render_structure(root)
        lines = s.split("\n")
        assert any("..." in l and "tags" not in l for l in lines)

    def test_list_constraint_on_continuation_line(self):
        root = parse_schema(List(str, min=10))
        s = render_structure(root)
        assert "# minimum 10 items" in s

    def test_list_constraint_both_bounds(self):
        root = parse_schema(List(str, min=5, max=50))
        s = render_structure(root)
        assert "# 5 to 50 items" in s

    def test_list_constraint_singular(self):
        root = parse_schema(List(str, min=1))
        s = render_structure(root)
        assert "# minimum 1 item" in s
        assert "items" not in s.split("# minimum 1 item")[0].split("\n")[-1]

    def test_no_constraint_comment_when_unconstrained(self):
        root = parse_schema([str])
        s = render_structure(root)
        lines = [l for l in s.split("\n") if "..." in l and "#" not in l]
        # The continuation ... line should have no comment
        assert len(lines) >= 1

    def test_comments_aligned_within_object(self):
        """All # comments in an object should start at the same column."""
        root = parse_schema({
            "x": str,
            "long_field_name": int,
            "y": bool,
        })
        s = render_structure(root)
        comment_positions = []
        for line in s.split("\n"):
            if "#" in line:
                comment_positions.append(line.index("#"))
        assert len(set(comment_positions)) == 1, (
            f"Comments not aligned: positions {comment_positions}"
        )

    def test_enum_in_structure_comment(self):
        root = parse_schema({"status": Field(str, enum=["a", "b", "c"])})
        s = render_structure(root)
        assert 'one of: "a", "b", "c"' in s


class TestRequirementsRendering:
    """The Requirements section produces natural language bullets."""

    def test_list_of_objects(self):
        root = parse_schema([{"title": str}])
        r = render_requirements(root)
        assert "**list of objects**" in r
        assert "`title`" in r
        assert "**string**" in r
        assert "Required." in r

    def test_object_schema(self):
        root = parse_schema({"name": str, "count": int})
        r = render_requirements(root)
        assert "**object**" in r
        assert "`name`" in r
        assert "`count`" in r

    def test_list_constraint_text(self):
        root = parse_schema(List({"x": str}, min=20))
        r = render_requirements(root)
        assert "at least **20 items**" in r

    def test_list_constraint_both_bounds(self):
        root = parse_schema(List(str, min=10, max=50))
        r = render_requirements(root)
        assert "**10 to 50 items**" in r

    def test_optional_field(self):
        root = parse_schema({"bio": Field(str, optional=True)})
        r = render_requirements(root)
        assert "Optional" in r
        assert "`None`" in r

    def test_optional_with_constraint(self):
        root = parse_schema({"r": Field(int, min=1, max=5, optional=True)})
        r = render_requirements(root)
        assert "Optional" in r
        assert "If present" in r
        assert "between 1 and 5" in r

    def test_string_min_length(self):
        root = parse_schema({"desc": Field(str, min_length=10)})
        r = render_requirements(root)
        assert "At least 10 characters" in r

    def test_string_max_length(self):
        root = parse_schema({"snippet": Field(str, max_length=500)})
        r = render_requirements(root)
        assert "At most 500 characters" in r

    def test_string_both_lengths(self):
        root = parse_schema({"text": Field(str, min_length=10, max_length=500)})
        r = render_requirements(root)
        assert "Between 10 and 500 characters" in r

    def test_pattern(self):
        root = parse_schema({"date": Field(str, pattern=r"\d{4}-\d{2}-\d{2}")})
        r = render_requirements(root)
        assert r"`\d{4}-\d{2}-\d{2}`" in r

    def test_numeric_min_only(self):
        root = parse_schema({"price": Field(float, min=0)})
        r = render_requirements(root)
        assert ">= 0" in r

    def test_numeric_both_bounds(self):
        root = parse_schema({"rating": Field(int, min=1, max=5)})
        r = render_requirements(root)
        assert "Between 1 and 5" in r

    def test_enum_in_requirements(self):
        root = parse_schema({"status": Field(str, enum=["active", "inactive"])})
        r = render_requirements(root)
        assert 'Must be one of: "active", "inactive"' in r

    def test_freestyle_dict(self):
        root = parse_schema({"specs": dict})
        r = render_requirements(root)
        assert "freestyle object" in r
        assert "Extract whatever" in r

    def test_nested_list_transition_phrase(self):
        root = parse_schema({"team": [{"name": str}]})
        r = render_requirements(root)
        assert "Each object in `team` must have:" in r

    def test_flat_list_of_strings(self):
        root = parse_schema(List(str, min=10))
        r = render_requirements(root)
        assert "**list of strings**" in r

    def test_singular_character(self):
        root = parse_schema({"t": Field(str, min_length=1)})
        r = render_requirements(root)
        assert "At least 1 character." in r
        # Should NOT say "characters" for 1
        assert "1 characters" not in r


class TestSchemaPromptAssembly:
    """The full ## Output Schema section."""

    def test_header_always_present(self):
        prompt = render_schema_prompt(parse_schema([{"x": str}]))
        assert "## Output Schema" in prompt
        assert "### Structure" in prompt
        assert "### Requirements" in prompt

    def test_optional_paragraph_present_when_optional_fields(self):
        prompt = render_schema_prompt(
            parse_schema({"bio": Field(str, optional=True)})
        )
        assert "For optional fields" in prompt
        assert "return `None`" in prompt

    def test_optional_paragraph_absent_when_no_optional_fields(self):
        prompt = render_schema_prompt(parse_schema({"name": str}))
        assert "For optional fields" not in prompt

    def test_optional_paragraph_detects_nested_optional(self):
        """Optional field buried in nested structure should trigger paragraph."""
        prompt = render_schema_prompt(
            parse_schema([{"items": [{"bio": Field(str, optional=True)}]}])
        )
        assert "For optional fields" in prompt

    def test_validation_warning_present(self):
        prompt = render_schema_prompt(parse_schema([{"x": str}]))
        assert "validated" in prompt
        assert "rejected" in prompt


class TestPromptAgentFriendliness:
    """Does the prompt actually help an AI agent write correct code?

    These tests verify the prompt from an agent's perspective:
    Can the agent understand what to return, what constraints to follow,
    and what will cause rejection?
    """

    def test_complex_schema_is_readable(self):
        """A complex real-world schema should produce a prompt that an
        agent can parse without ambiguity."""
        schema = List({
            "title": Field(str, min_length=1),
            "price": Field(float, min=0),
            "currency": Field(str, enum=["USD", "EUR", "GBP"]),
            "rating": Field(int, min=1, max=5, optional=True),
            "description": Field(str, min_length=20),
            "in_stock": bool,
            "specs": dict,
            "variants": [{
                "color": str,
                "size": Field(str, enum=["S", "M", "L", "XL"]),
                "price": Field(float, min=0),
            }],
        }, min=10)
        prompt = render_schema_prompt(parse_schema(schema))

        # Agent should see: what to return
        assert "list of objects" in prompt
        assert "at least **10" in prompt

        # Agent should see: every field name
        for field in ["title", "price", "currency", "rating", "description",
                       "in_stock", "specs", "variants", "color", "size"]:
            assert f"`{field}`" in prompt, f"Field {field!r} missing from Requirements"

        # Agent should see: enum values
        assert '"USD"' in prompt
        assert '"EUR"' in prompt

        # Agent should see: the optional field is optional
        assert "Optional" in prompt

        # Agent should see: freestyle dict guidance
        assert "Extract whatever" in prompt

    def test_structure_and_requirements_agree(self):
        """Both sections must describe the same fields."""
        schema = {"name": str, "age": int, "bio": Field(str, optional=True)}
        prompt = render_schema_prompt(parse_schema(schema))
        structure = prompt.split("### Structure")[1].split("### Requirements")[0]
        requirements = prompt.split("### Requirements")[1]

        for field in ["name", "age", "bio"]:
            assert f'"{field}"' in structure, f"{field} missing from Structure"
            assert f"`{field}`" in requirements, f"{field} missing from Requirements"
