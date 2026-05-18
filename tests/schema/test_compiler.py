"""Tests for the CompiledSchema and compile_schema integration.

These tests verify the full pipeline: user schema → compiled object
that can render prompts and validate data.
"""

import pytest

from scout.errors import ScoutSchemaError
from scout.schema.compiler import CompiledSchema, compile_schema
from scout.schema.types import Field, List


class TestCompileSchema:
    """compile_schema() is the main entry point."""

    def test_returns_compiled_schema(self):
        cs = compile_schema([{"x": str}])
        assert isinstance(cs, CompiledSchema)

    def test_has_prompt(self):
        cs = compile_schema([{"title": str}])
        assert "## Output Schema" in cs.prompt
        assert "title" in cs.prompt

    def test_has_root_node(self):
        from scout.schema.nodes import ListNode

        cs = compile_schema([{"x": str}])
        assert isinstance(cs.root, ListNode)

    def test_invalid_schema_raises(self):
        with pytest.raises(ScoutSchemaError):
            compile_schema(42)

    def test_invalid_field_raises(self):
        with pytest.raises(ScoutSchemaError):
            compile_schema(Field(str, min=5))  # min not valid for str


class TestCompiledSchemaValidate:
    """CompiledSchema.validate() returns (bool, str)."""

    def test_valid_data_returns_true(self):
        cs = compile_schema([{"name": str, "price": float}])
        valid, msg = cs.validate([{"name": "Widget", "price": 9.99}])
        assert valid is True
        assert msg == ""

    def test_invalid_data_returns_false_with_message(self):
        cs = compile_schema([{"name": str}])
        valid, msg = cs.validate("not a list")
        assert valid is False
        assert "Expected a list" in msg

    def test_none_returns_false_with_return_hint(self):
        cs = compile_schema([{"x": str}])
        valid, msg = cs.validate(None)
        assert valid is False
        assert "return statement" in msg

    def test_empty_list_with_min(self):
        cs = compile_schema(List(str, min_items=5))
        valid, msg = cs.validate([])
        assert valid is False
        assert "empty" in msg

    def test_constraint_violation(self):
        cs = compile_schema([{"price": Field(float, min=0)}])
        valid, msg = cs.validate([{"price": -1.0}])
        assert valid is False
        assert ">= 0" in msg or "between" in msg

    def test_extra_fields_accepted(self):
        cs = compile_schema([{"name": str}])
        valid, _ = cs.validate([{"name": "x", "extra": 42}])
        assert valid is True


class TestCompileSchemaPromptQuality:
    """The compiled prompt is agent-ready."""

    def test_prompt_has_structure_and_requirements(self):
        cs = compile_schema(List({"title": str, "price": float}, min_items=10))
        assert "### Structure" in cs.prompt
        assert "### Requirements" in cs.prompt
        assert "minimum 10" in cs.prompt

    def test_prompt_mentions_all_fields(self):
        cs = compile_schema(
            {
                "name": str,
                "age": int,
                "bio": Field(str, optional=True),
            }
        )
        for field in ["name", "age", "bio"]:
            assert field in cs.prompt

    def test_prompt_has_optional_paragraph_when_needed(self):
        cs = compile_schema({"x": Field(str, optional=True)})
        assert "For optional fields" in cs.prompt

    def test_prompt_no_optional_paragraph_when_not_needed(self):
        cs = compile_schema({"x": str})
        assert "For optional fields" not in cs.prompt


class TestRoundTrip:
    """Compile a schema, validate good and bad data, check error quality."""

    def test_ecommerce_schema(self):
        cs = compile_schema(
            List(
                {
                    "title": Field(str, min_length=1),
                    "price": Field(float, min=0),
                    "currency": Field(str, enum=["USD", "EUR", "GBP"]),
                    "in_stock": bool,
                },
                min_items=5,
            )
        )

        # Good data
        good = [
            {"title": f"P{i}", "price": float(i), "currency": "USD", "in_stock": True}
            for i in range(10)
        ]
        assert cs.validate(good) == (True, "")

        # Bad data: wrong types
        bad = [{"title": "", "price": "free", "currency": "Dollar", "in_stock": "yes"}]
        valid, msg = cs.validate(bad)
        assert not valid
        assert "expected float" in msg
        assert "not one of" in msg

    def test_job_listings_schema(self):
        cs = compile_schema(
            List(
                {
                    "title": Field(str, min_length=3),
                    "company": str,
                    "salary": {
                        "min": Field(int, min=0, optional=True),
                        "max": Field(int, min=0, optional=True),
                    },
                    "posted": Field(str, pattern=r"\d{4}-\d{2}-\d{2}"),
                },
                min_items=10,
            )
        )

        good = [
            {
                "title": "Engineer",
                "company": "Acme",
                "salary": {"min": 80000, "max": 120000},
                "posted": "2024-03-15",
            }
            for _ in range(15)
        ]
        assert cs.validate(good) == (True, "")

        # Salary as a number instead of object
        bad = [{"title": "Eng", "company": "X", "salary": 80000, "posted": "2024-03-15"}]
        valid, msg = cs.validate(bad)
        assert not valid
        assert "expected an object" in msg
