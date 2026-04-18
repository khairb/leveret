"""Tests for Layer 4: Validator and Layer 5: Error formatter.

These tests validate both the correctness of validation logic AND the
quality of error messages from the agent's perspective. An agent that
receives these messages should understand exactly what went wrong and
how to fix it.
"""

import math

import pytest

from scout.schema.compiler import compile_schema
from scout.schema.formatter import format_errors
from scout.schema.parse import parse_schema
from scout.schema.types import Field, List
from scout.schema.validate import RawError, _check_type, validate


# ---------------------------------------------------------------------------
# Type coercion — the boundary between "right type" and "wrong type"
# ---------------------------------------------------------------------------

class TestTypeCoercion:
    """Type checking with lenient coercion rules."""

    # int
    def test_int_accepts_int(self):
        assert _check_type(42, int) is True

    def test_int_accepts_whole_float(self):
        assert _check_type(42.0, int) is True

    def test_int_rejects_fractional_float(self):
        assert _check_type(42.5, int) is False

    def test_int_rejects_bool(self):
        assert _check_type(True, int) is False
        assert _check_type(False, int) is False

    def test_int_rejects_string(self):
        assert _check_type("42", int) is False

    def test_int_rejects_inf(self):
        assert _check_type(float("inf"), int) is False

    def test_int_rejects_nan(self):
        assert _check_type(float("nan"), int) is False

    # float
    def test_float_accepts_float(self):
        assert _check_type(42.5, float) is True

    def test_float_accepts_int(self):
        assert _check_type(42, float) is True

    def test_float_rejects_bool(self):
        assert _check_type(True, float) is False

    def test_float_rejects_string(self):
        assert _check_type("42.5", float) is False

    def test_float_accepts_inf(self):
        """inf is a valid float — constraint checks catch it if needed."""
        assert _check_type(float("inf"), float) is True

    def test_float_accepts_nan(self):
        """nan is a valid float — constraint checks catch it if needed."""
        assert _check_type(float("nan"), float) is True

    # bool
    def test_bool_accepts_true_false(self):
        assert _check_type(True, bool) is True
        assert _check_type(False, bool) is True

    def test_bool_rejects_int(self):
        assert _check_type(0, bool) is False
        assert _check_type(1, bool) is False

    def test_bool_rejects_string(self):
        assert _check_type("true", bool) is False

    # str
    def test_str_accepts_string(self):
        assert _check_type("hello", str) is True

    def test_str_rejects_number(self):
        assert _check_type(42, str) is False


# ---------------------------------------------------------------------------
# Tier 1: Top-level type check
# ---------------------------------------------------------------------------

class TestTier1:
    """Wrong top-level type → single error, immediate short-circuit."""

    def test_list_schema_gets_string(self):
        root = parse_schema([{"x": str}])
        errs = validate("not a list", root)
        assert len(errs) == 1
        assert "expected a list" in errs[0].message
        assert "string" in errs[0].message

    def test_list_schema_gets_dict(self):
        root = parse_schema([{"x": str}])
        errs = validate({"x": "hello"}, root)
        assert len(errs) == 1
        assert "expected a list" in errs[0].message

    def test_list_schema_gets_none(self):
        root = parse_schema([{"x": str}])
        errs = validate(None, root)
        assert len(errs) == 1
        assert "null" in errs[0].message
        assert "return statement" in errs[0].message

    def test_object_schema_gets_list(self):
        root = parse_schema({"x": str})
        errs = validate(["wrong"], root)
        assert len(errs) == 1
        assert "expected an object" in errs[0].message


# ---------------------------------------------------------------------------
# Tier 2: Container constraints
# ---------------------------------------------------------------------------

class TestTier2:
    """List count constraints with correct short-circuiting."""

    def test_empty_list_short_circuits(self):
        root = parse_schema(List({"x": str, "y": int}, min=20))
        errs = validate([], root)
        # Only 1 error — doesn't try to validate items
        assert len(errs) == 1
        assert "empty" in errs[0].message

    def test_under_count_continues_to_validate_items(self):
        root = parse_schema(List({"x": str}, min=10))
        errs = validate([{"x": 42}], root)  # 1 item (under 10), wrong type
        # Should have BOTH the count error AND the type error
        messages = [e.message for e in errs]
        assert any("returned 1 item" in m for m in messages)
        assert any("expected str" in m for m in messages)

    def test_over_count_caps_validation_at_50(self):
        root = parse_schema(List({"x": str}, max=10))
        data = [{"x": "ok"}] * 100
        errs = validate(data, root)
        # Should report over-count, but not validate all 100
        assert any("returned 100" in e.message for e in errs)

    def test_max_zero_works(self):
        """Edge case: max=0 should be handled correctly (not falsy)."""
        root = parse_schema(List(str, max=0))
        errs = validate(["a"], root)
        assert any("maximum is 0" in e.message for e in errs)


# ---------------------------------------------------------------------------
# Tier 3: Object field checks
# ---------------------------------------------------------------------------

class TestTier3:
    """Object field presence, null handling, extra fields."""

    def test_missing_required_field(self):
        root = parse_schema({"name": str, "age": int})
        errs = validate({"name": "Alice"}, root)
        assert len(errs) == 1
        assert errs[0].path == "age"
        assert "missing required field" in errs[0].message

    def test_null_required_field(self):
        root = parse_schema({"name": str})
        errs = validate({"name": None}, root)
        assert len(errs) == 1
        assert "null" in errs[0].message
        assert "required" in errs[0].message

    def test_null_optional_field_is_valid(self):
        root = parse_schema({"bio": Field(str, optional=True)})
        errs = validate({"bio": None}, root)
        assert len(errs) == 0

    def test_missing_optional_field_is_valid(self):
        root = parse_schema({"bio": Field(str, optional=True)})
        errs = validate({}, root)
        assert len(errs) == 0

    def test_extra_fields_silently_ignored(self):
        root = parse_schema({"name": str})
        errs = validate({"name": "Alice", "age": 30, "extra": True}, root)
        assert len(errs) == 0

    def test_empty_string_is_valid_for_required_str(self):
        root = parse_schema({"name": str})
        errs = validate({"name": ""}, root)
        assert len(errs) == 0

    def test_zero_is_valid_for_required_int(self):
        root = parse_schema({"count": int})
        errs = validate({"count": 0}, root)
        assert len(errs) == 0

    def test_false_is_valid_for_required_bool(self):
        root = parse_schema({"active": bool})
        errs = validate({"active": False}, root)
        assert len(errs) == 0


# ---------------------------------------------------------------------------
# Tier 4: Value constraints
# ---------------------------------------------------------------------------

class TestTier4:
    """Field-level constraint validation."""

    def test_int_below_min(self):
        root = parse_schema({"r": Field(int, min=1, max=5)})
        errs = validate({"r": 0}, root)
        assert len(errs) == 1
        assert "between 1 and 5" in errs[0].message

    def test_int_above_max(self):
        root = parse_schema({"r": Field(int, min=1, max=5)})
        errs = validate({"r": 10}, root)
        assert len(errs) == 1
        assert "between 1 and 5" in errs[0].message

    def test_int_min_only(self):
        root = parse_schema({"count": Field(int, min=0)})
        errs = validate({"count": -1}, root)
        assert ">= 0" in errs[0].message

    def test_int_max_only(self):
        root = parse_schema({"count": Field(int, max=100)})
        errs = validate({"count": 200}, root)
        assert "<= 100" in errs[0].message

    def test_string_too_short(self):
        root = parse_schema({"desc": Field(str, min_length=10)})
        errs = validate({"desc": "short"}, root)
        assert "5 characters" in errs[0].message
        assert "minimum is 10" in errs[0].message

    def test_string_too_long(self):
        root = parse_schema({"bio": Field(str, max_length=5)})
        errs = validate({"bio": "too long text"}, root)
        assert "maximum is 5" in errs[0].message

    def test_pattern_mismatch(self):
        root = parse_schema({"date": Field(str, pattern=r"\d{4}-\d{2}-\d{2}")})
        errs = validate({"date": "not-a-date"}, root)
        assert "does not match pattern" in errs[0].message
        assert "not-a-date" in errs[0].message

    def test_pattern_match(self):
        root = parse_schema({"date": Field(str, pattern=r"\d{4}-\d{2}-\d{2}")})
        errs = validate({"date": "2024-03-15"}, root)
        assert len(errs) == 0

    def test_enum_mismatch(self):
        root = parse_schema({"status": Field(str, enum=["active", "inactive"])})
        errs = validate({"status": "pending"}, root)
        assert "not one of" in errs[0].message
        assert '"active"' in errs[0].message

    def test_enum_case_sensitive(self):
        root = parse_schema({"s": Field(str, enum=["Active"])})
        errs = validate({"s": "active"}, root)
        assert len(errs) == 1

    def test_enum_match(self):
        root = parse_schema({"s": Field(str, enum=["a", "b"])})
        errs = validate({"s": "a"}, root)
        assert len(errs) == 0

    def test_optional_constraint_violation(self):
        """Optional field present but violating constraint."""
        root = parse_schema({"r": Field(int, min=1, max=5, optional=True)})
        errs = validate({"r": 0}, root)
        assert len(errs) == 1
        assert "between 1 and 5" in errs[0].message

    def test_optional_valid_value(self):
        root = parse_schema({"r": Field(int, min=1, max=5, optional=True)})
        errs = validate({"r": 3}, root)
        assert len(errs) == 0

    def test_freestyle_dict_accepts_any_dict(self):
        root = parse_schema({"specs": dict})
        errs = validate({"specs": {"a": 1, "b": "two"}}, root)
        assert len(errs) == 0

    def test_freestyle_dict_rejects_non_dict(self):
        root = parse_schema({"specs": dict})
        errs = validate({"specs": "not a dict"}, root)
        assert len(errs) == 1


# ---------------------------------------------------------------------------
# Deep nesting
# ---------------------------------------------------------------------------

class TestDeepNesting:
    """Errors in deeply nested structures have correct paths."""

    def test_nested_path(self):
        root = parse_schema([{"items": [{"x": int}]}])
        errs = validate([{"items": [{"x": "bad"}]}], root)
        assert errs[0].path == "[0].items[0].x"

    def test_four_levels_deep(self):
        root = parse_schema([{"a": [{"b": [{"c": int}]}]}])
        errs = validate([{"a": [{"b": [{"c": "bad"}]}]}], root)
        assert errs[0].path == "[0].a[0].b[0].c"


# ---------------------------------------------------------------------------
# Error formatter — agent-facing quality
# ---------------------------------------------------------------------------

class TestErrorFormatterGrouping:
    """Errors are grouped, sorted, and capped correctly."""

    def test_groups_same_error_across_items(self):
        root = parse_schema(List({"price": float}, min=1))
        data = [{"price": f"${i}"} for i in range(50)]
        errs = validate(data, root)
        output = format_errors(errs, total_items=50, node_tree=root)
        assert "50 of 50 items" in output
        # Should be 1 grouped error, not 50 individual errors
        assert output.count("[*].price") == 1

    def test_type_errors_before_constraint_errors(self):
        root = parse_schema(List({"x": Field(int, min=0)}, min=1))
        data = [{"x": "bad"}, {"x": -1}]
        errs = validate(data, root)
        output = format_errors(errs, total_items=2, node_tree=root)
        lines = output.split("\n")
        type_idx = next(i for i, l in enumerate(lines) if "expected int" in l)
        constraint_idx = next(i for i, l in enumerate(lines) if ">=" in l)
        assert type_idx < constraint_idx

    def test_cap_at_10_groups(self):
        errs = [RawError(f"field_{i}", f"error {i}", None, "type") for i in range(15)]
        output = format_errors(errs)
        assert "showing first 10" in output
        assert "5 more error types" in output

    def test_singular_error_header(self):
        errs = [RawError("x", "bad", None, "type")]
        output = format_errors(errs)
        assert "1 error)" in output
        assert "error type" not in output


class TestErrorFormatterMessages:
    """Error messages are clear and actionable for the agent."""

    def test_none_return_gives_hint(self):
        cs = compile_schema([{"x": str}])
        _, msg = cs.validate(None)
        assert "return statement" in msg

    def test_empty_path_formatted_cleanly(self):
        """Top-level errors don't show an empty path."""
        cs = compile_schema(List(str, min=10))
        _, msg = cs.validate([])
        # Should NOT have double space or empty path
        assert "  [1] List is empty" in msg

    def test_systematic_price_bug_is_obvious(self):
        """The most common real-world bug: prices as strings."""
        cs = compile_schema(List({"price": Field(float, min=0)}, min=1))
        data = [{"price": f"${i}.99"} for i in range(30)]
        _, msg = cs.validate(data)
        assert "30 of 30 items" in msg
        assert "$" in msg  # shows the actual values so agent sees the $ prefix

    def test_optional_field_note_with_concrete_values(self):
        cs = compile_schema(List({"r": Field(int, min=1, max=5, optional=True)}, min=1))
        data = [{"r": 0}, {"r": -1}]
        _, msg = cs.validate(data)
        assert "null is valid" in msg
        assert "0" in msg

    def test_optional_field_note_deduplicates_values(self):
        cs = compile_schema(List({"r": Field(int, min=1, max=5, optional=True)}, min=1))
        data = [{"r": 0}, {"r": 0}, {"r": 0}]
        _, msg = cs.validate(data)
        # Should say "but 0 is not", NOT "but 0, 0, and 0 are not"
        assert "0 is not" in msg
        assert "0, 0" not in msg

    def test_count_context_always_plural_with_total(self):
        """'1 of 5 items' not '1 of 5 item'."""
        cs = compile_schema(List({"x": int}, min=1))
        data = [{"x": "bad"}] + [{"x": 1}] * 4
        _, msg = cs.validate(data)
        assert "1 of 5 items" in msg


class TestEndToEndValidation:
    """Full pipeline: compile → validate → readable error message."""

    def test_complex_realistic_scenario(self):
        """Simulate a realistic agent output with multiple error types."""
        cs = compile_schema(List({
            "title": Field(str, min_length=1),
            "price": Field(float, min=0),
            "rating": Field(int, min=1, max=5, optional=True),
            "url": str,
        }, min=20))

        # Agent got 8 items (not enough), some have type errors
        data = [
            {"title": "Widget A", "price": 10.0, "rating": 3, "url": "/a"},
            {"title": "Widget B", "price": "$20", "rating": 3, "url": "/b"},
            {"title": "", "price": 30.0, "rating": 0, "url": "/c"},
            {"title": "Widget D", "price": 40.0, "url": "/d"},  # rating missing (ok)
            {"title": "Widget E", "price": -5.0, "rating": 3, "url": "/e"},
            {"title": "Widget F", "price": 60.0, "rating": None, "url": "/f"},
            {"title": "Widget G", "price": 70.0, "rating": 3, "url": "/g"},
            {"title": "Widget H", "price": 80.0, "rating": 3, "url": "/h"},
        ]

        valid, msg = cs.validate(data)
        assert not valid

        # Agent should see: under-count
        assert "8" in msg and "20" in msg

        # Agent should see: price type error
        assert "expected float" in msg and "$20" in msg

        # Agent should see: rating constraint with optional note
        assert "between 1 and 5" in msg

        # Agent should see: title min_length
        assert "minimum is 1" in msg

        # Agent should see: price constraint
        assert ">= 0" in msg or "between" in msg
