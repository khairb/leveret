"""Tests for the schema validation gate in the agent loop.

These tests verify the schema validation gate logic that sits between
the exit-code check and the LLM validator in the agent loop. Since the
full agent loop requires a browser and LLM, we test the gate logic in
isolation by simulating what the loop does: parse JSON, validate against
a compiled schema, and check the (valid, feedback) result.

This mirrors the exact code path in loop.py lines 351-373.
"""

import json

import pytest

from scout.schema.compiler import CompiledSchema, compile_schema
from scout.schema.types import Field, List


# ── Helpers ──────────────────────────────────────────────────────

def _simulate_schema_gate(
    compiled_schema: CompiledSchema,
    return_value_json: str | None,
) -> tuple[bool, str]:
    """Simulate the schema validation gate from loop.py.

    This replicates the exact logic in the loop:
    1. Parse return_value_json (or use None if absent)
    2. Call compiled_schema.validate(return_data)
    3. Return (schema_passed, feedback)
    """
    if return_value_json is not None:
        try:
            return_data = json.loads(return_value_json)
        except (ValueError, TypeError):
            return_data = None
    else:
        return_data = None

    valid, feedback = compiled_schema.validate(return_data)
    return valid, feedback


# ── Happy paths ──────────────────────────────────────────────────

class TestSchemaGatePass:
    """Data that matches the schema passes the gate."""

    def test_simple_list_of_objects(self):
        cs = compile_schema([{"name": str, "price": float}])
        data = [{"name": "Widget", "price": 9.99}]
        valid, msg = _simulate_schema_gate(cs, json.dumps(data))
        assert valid is True
        assert msg == ""

    def test_complex_schema(self):
        cs = compile_schema(List({
            "title": Field(str, min_length=1),
            "price": Field(float, min=0),
            "currency": Field(str, enum=["USD", "EUR"]),
        }, min_items=5))
        data = [
            {"title": f"P{i}", "price": float(i), "currency": "USD"}
            for i in range(10)
        ]
        valid, msg = _simulate_schema_gate(cs, json.dumps(data))
        assert valid is True

    def test_object_schema(self):
        cs = compile_schema({"name": str, "count": int})
        data = {"name": "test", "count": 42}
        valid, msg = _simulate_schema_gate(cs, json.dumps(data))
        assert valid is True

    def test_optional_field_with_none(self):
        cs = compile_schema([{"name": str, "bio": Field(str, optional=True)}])
        data = [{"name": "Alice", "bio": None}]
        valid, msg = _simulate_schema_gate(cs, json.dumps(data))
        assert valid is True


# ── Schema failures — these skip the LLM validator ───────────────

class TestSchemaGateFail:
    """Data that violates the schema is rejected immediately."""

    def test_wrong_top_level_type(self):
        cs = compile_schema([{"x": str}])
        valid, msg = _simulate_schema_gate(cs, json.dumps("not a list"))
        assert valid is False
        assert "Expected a list" in msg

    def test_none_return_no_json(self):
        """When return_value_json is None (no markers found),
        the gate validates None — catching missing return statements."""
        cs = compile_schema([{"x": str}])
        valid, msg = _simulate_schema_gate(cs, None)
        assert valid is False
        assert "return statement" in msg

    def test_null_json_return(self):
        """When the function explicitly returns None (JSON 'null')."""
        cs = compile_schema([{"x": str}])
        valid, msg = _simulate_schema_gate(cs, "null")
        assert valid is False
        assert "return statement" in msg

    def test_wrong_field_type(self):
        cs = compile_schema([{"price": float}])
        data = [{"price": "free"}]
        valid, msg = _simulate_schema_gate(cs, json.dumps(data))
        assert valid is False
        assert "expected float" in msg

    def test_missing_required_field(self):
        cs = compile_schema([{"name": str, "price": float}])
        data = [{"name": "Widget"}]
        valid, msg = _simulate_schema_gate(cs, json.dumps(data))
        assert valid is False
        assert "price" in msg

    def test_constraint_violation(self):
        cs = compile_schema([{"rating": Field(int, min=1, max=5)}])
        data = [{"rating": 0}]
        valid, msg = _simulate_schema_gate(cs, json.dumps(data))
        assert valid is False
        assert "between 1 and 5" in msg

    def test_enum_violation(self):
        cs = compile_schema([{"status": Field(str, enum=["active", "inactive"])}])
        data = [{"status": "unknown"}]
        valid, msg = _simulate_schema_gate(cs, json.dumps(data))
        assert valid is False
        assert "not one of" in msg

    def test_list_too_short(self):
        cs = compile_schema(List({"x": str}, min_items=5))
        data = [{"x": "a"}]
        valid, msg = _simulate_schema_gate(cs, json.dumps(data))
        assert valid is False

    def test_invalid_json_treated_as_none(self):
        """Malformed JSON in return_value_json → parsed as None."""
        cs = compile_schema([{"x": str}])
        valid, msg = _simulate_schema_gate(cs, "not valid json {{{")
        assert valid is False
        # None fails the top-level type check


# ── Error message quality (agent perspective) ─────────────────────

class TestSchemaGateErrorQuality:
    """The error messages from schema rejection should be clear enough
    for an AI agent to understand and fix the code."""

    def test_error_mentions_field_path(self):
        cs = compile_schema([{"details": {"price": float}}])
        data = [{"details": {"price": "not a number"}}]
        valid, msg = _simulate_schema_gate(cs, json.dumps(data))
        assert valid is False
        assert "price" in msg
        assert "float" in msg

    def test_error_gives_count_context(self):
        """When many items have the same error, the message shows count."""
        cs = compile_schema([{"price": float}])
        data = [{"price": "bad"} for _ in range(20)]
        valid, msg = _simulate_schema_gate(cs, json.dumps(data))
        assert valid is False
        assert "20" in msg  # count of affected items

    def test_multiple_error_types_all_shown(self):
        cs = compile_schema([{
            "name": str,
            "price": float,
            "count": int,
        }])
        data = [{"name": 42, "price": "free", "count": "ten"}]
        valid, msg = _simulate_schema_gate(cs, json.dumps(data))
        assert valid is False
        # Multiple error types should be reported
        assert "name" in msg or "price" in msg

    def test_feedback_is_self_contained(self):
        """The feedback string should be usable directly in a rejection
        message — no additional formatting needed."""
        cs = compile_schema([{"x": Field(int, min=0)}])
        data = [{"x": -5}]
        valid, msg = _simulate_schema_gate(cs, json.dumps(data))
        assert valid is False
        # Should have structure: header + numbered errors
        assert msg.strip()  # non-empty
        assert "\n" in msg  # multi-line


# ── Real-world agent mistake scenarios ────────────────────────────

class TestRealWorldAgentMistakes:
    """These test scenarios that actually happen when an LLM writes
    scraping functions — the most common extraction mistakes."""

    def test_prices_extracted_as_strings_with_dollar_signs(self):
        """Agent uses .text_content() and gets '$12.99' instead of 12.99."""
        cs = compile_schema(List({
            "title": str,
            "price": Field(float, min=0),
        }, min_items=10))
        data = [
            {"title": f"Product {i}", "price": f"${i * 10}.99"}
            for i in range(20)
        ]
        valid, msg = _simulate_schema_gate(cs, json.dumps(data))
        assert valid is False
        assert "expected float, got string" in msg
        assert "20 of 20" in msg  # all items affected

    def test_boolean_extracted_as_string(self):
        """Agent gets 'true'/'false' strings from DOM attributes."""
        cs = compile_schema([{"name": str, "active": bool}])
        data = [{"name": "Item", "active": "true"}]
        valid, msg = _simulate_schema_gate(cs, json.dumps(data))
        assert valid is False
        assert "expected bool" in msg

    def test_integer_extracted_as_float(self):
        """Agent gets 4.0 instead of 4 — should pass (spec: int accepts whole floats)."""
        cs = compile_schema([{"rating": int}])
        data = [{"rating": 4.0}]
        valid, msg = _simulate_schema_gate(cs, json.dumps(data))
        assert valid is True

    def test_integer_with_decimal_rejected(self):
        """Agent gets 4.5 for an int field — not a whole number."""
        cs = compile_schema([{"count": int}])
        data = [{"count": 4.5}]
        valid, msg = _simulate_schema_gate(cs, json.dumps(data))
        assert valid is False

    def test_missing_field_due_to_typo(self):
        """Agent extracted 'titel' instead of 'title'."""
        cs = compile_schema([{"title": str, "price": float}])
        data = [{"titel": "Oops", "price": 9.99}]
        valid, msg = _simulate_schema_gate(cs, json.dumps(data))
        assert valid is False
        assert "title" in msg  # mentions the missing field

    def test_enum_value_wrong_case(self):
        """Agent extracted 'usd' instead of 'USD'."""
        cs = compile_schema([{"currency": Field(str, enum=["USD", "EUR"])}])
        data = [{"currency": "usd"}]
        valid, msg = _simulate_schema_gate(cs, json.dumps(data))
        assert valid is False
        assert "not one of" in msg

    def test_nested_object_returned_as_flat(self):
        """Agent flattened a nested structure."""
        cs = compile_schema([{
            "name": str,
            "address": {"street": str, "city": str},
        }])
        data = [{"name": "Alice", "address": "123 Main St, NYC"}]
        valid, msg = _simulate_schema_gate(cs, json.dumps(data))
        assert valid is False
        assert "expected an object" in msg

    def test_list_returned_as_single_object(self):
        """Agent returned one object instead of a list of objects."""
        cs = compile_schema([{"title": str}])
        data = {"title": "Just one item"}
        valid, msg = _simulate_schema_gate(cs, json.dumps(data))
        assert valid is False
        assert "Expected a list" in msg

    def test_too_few_items_with_pagination_missed(self):
        """Agent only extracted first page (10 items) but min is 50."""
        cs = compile_schema(List({"title": str}, min_items=50))
        data = [{"title": f"P{i}"} for i in range(10)]
        valid, msg = _simulate_schema_gate(cs, json.dumps(data))
        assert valid is False
        assert "10" in msg and "50" in msg

    def test_mixed_errors_across_many_items(self):
        """Multiple error types across many items — tests grouping."""
        cs = compile_schema(List({
            "title": Field(str, min_length=1),
            "price": Field(float, min=0),
            "rating": Field(int, min=1, max=5),
        }, min_items=10))
        data = []
        for i in range(15):
            item = {"title": f"P{i}", "price": float(i), "rating": i % 6}
            if i % 3 == 0:
                item["price"] = f"${i}.99"  # string instead of float
            if i % 5 == 0:
                item["title"] = ""  # too short
            data.append(item)
        valid, msg = _simulate_schema_gate(cs, json.dumps(data))
        assert valid is False
        # Should have multiple error groups
        assert "[1]" in msg
        assert "[2]" in msg

    def test_deeply_nested_error_path_is_readable(self):
        """Deeply nested paths should be clear, not confusing."""
        cs = compile_schema([{
            "categories": [{
                "products": [{
                    "variants": [{"size": Field(str, enum=["S", "M", "L"])}],
                }],
            }],
        }])
        data = [{
            "categories": [{
                "products": [{
                    "variants": [{"size": "XXL"}],
                }],
            }],
        }]
        valid, msg = _simulate_schema_gate(cs, json.dumps(data))
        assert valid is False
        assert "variants" in msg
        assert "size" in msg
        assert "XXL" in msg

    def test_optional_field_with_wrong_type_not_null(self):
        """Optional field present with wrong type — not just null."""
        cs = compile_schema([{
            "name": str,
            "rating": Field(int, min=1, max=5, optional=True),
        }])
        data = [{"name": "Item", "rating": "five stars"}]
        valid, msg = _simulate_schema_gate(cs, json.dumps(data))
        assert valid is False
        assert "expected int" in msg

    def test_correct_complex_data_passes(self):
        """A fully correct complex dataset passes cleanly."""
        cs = compile_schema(List({
            "title": Field(str, min_length=1),
            "price": Field(float, min=0),
            "currency": Field(str, enum=["USD", "EUR", "GBP"]),
            "rating": Field(int, min=1, max=5, optional=True),
            "in_stock": bool,
            "tags": [str],
        }, min_items=10))
        data = [
            {
                "title": f"Product {i}",
                "price": float(i * 10 + 0.99),
                "currency": ["USD", "EUR", "GBP"][i % 3],
                "rating": (i % 5) + 1 if i % 2 == 0 else None,
                "in_stock": i % 3 != 0,
                "tags": [f"tag-{j}" for j in range(3)],
            }
            for i in range(25)
        ]
        valid, msg = _simulate_schema_gate(cs, json.dumps(data))
        assert valid is True
        assert msg == ""
