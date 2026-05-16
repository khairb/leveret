"""Tests for the tolerance system.

Tolerance controls what percentage of list items must pass validation.
These tests verify the filtering logic, threshold boundaries, and
interactions with structural constraints.
"""

import pytest

from scout.schema.compiler import compile_schema
from scout.schema.parse import parse_schema
from scout.schema.tolerance import (
    Tolerance,
    TOLERANCE_THRESHOLDS,
    apply_tolerance,
    _find_list_positions,
    _partition_and_count,
)
from scout.schema.types import Field, Items
from scout.schema.validate import RawError, validate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_items(good: int, bad: int) -> list[dict]:
    """Build a list with `good` valid items and `bad` invalid items.

    Schema: {"name": str, "price": float}
    Bad items have price as a string.
    """
    items = [{"name": f"Item {i}", "price": float(i)} for i in range(good)]
    items += [{"name": f"Bad {i}", "price": f"${i}"} for i in range(bad)]
    return items


def _validate_with_tolerance(
    schema, data, tolerance: Tolerance,
) -> tuple[bool, str]:
    """Compile schema and validate with tolerance."""
    cs = compile_schema(schema)
    return cs.validate(data, tolerance=tolerance)


# ---------------------------------------------------------------------------
# Strict — current behavior preserved
# ---------------------------------------------------------------------------

class TestStrict:
    """Strict tolerance: 100% of items must pass."""

    def test_all_valid_passes(self):
        valid, _ = _validate_with_tolerance(
            [{"name": str}],
            [{"name": "Alice"}, {"name": "Bob"}],
            Tolerance.STRICT,
        )
        assert valid

    def test_one_bad_item_fails(self):
        valid, msg = _validate_with_tolerance(
            [{"name": str}],
            [{"name": "Alice"}, {"name": 42}],
            Tolerance.STRICT,
        )
        assert not valid
        assert "expected str" in msg

    def test_strict_matches_no_tolerance(self):
        """Strict should produce identical results to no tolerance."""
        schema = [{"x": Field(int, min=0)}]
        data = [{"x": 1}, {"x": -1}, {"x": 2}]
        cs = compile_schema(schema)

        result_none = cs.validate(data)
        result_strict = cs.validate(data, tolerance=Tolerance.STRICT)

        assert result_none == result_strict


# ---------------------------------------------------------------------------
# Balanced — 80% threshold
# ---------------------------------------------------------------------------

class TestBalanced:
    """Balanced tolerance: 80% of items must pass."""

    def test_85_percent_passes(self):
        # 85 good, 15 bad → 85% pass rate ≥ 80%
        data = _make_items(good=85, bad=15)
        valid, _ = _validate_with_tolerance(
            [{"name": str, "price": float}],
            data,
            Tolerance.BALANCED,
        )
        assert valid

    def test_70_percent_fails(self):
        # 70 good, 30 bad → 70% pass rate < 80%
        data = _make_items(good=70, bad=30)
        valid, msg = _validate_with_tolerance(
            [{"name": str, "price": float}],
            data,
            Tolerance.BALANCED,
        )
        assert not valid
        assert "expected float" in msg

    def test_exactly_80_percent_passes(self):
        # 80 good, 20 bad → exactly 80% = threshold
        data = _make_items(good=80, bad=20)
        valid, _ = _validate_with_tolerance(
            [{"name": str, "price": float}],
            data,
            Tolerance.BALANCED,
        )
        assert valid

    def test_one_bad_in_100_passes(self):
        # 99 good, 1 bad → 99% ≥ 80%
        data = _make_items(good=99, bad=1)
        valid, _ = _validate_with_tolerance(
            [{"name": str, "price": float}],
            data,
            Tolerance.BALANCED,
        )
        assert valid

    def test_one_bad_in_five_passes(self):
        # 4 good, 1 bad → 80% ≥ 80%
        data = _make_items(good=4, bad=1)
        valid, _ = _validate_with_tolerance(
            [{"name": str, "price": float}],
            data,
            Tolerance.BALANCED,
        )
        assert valid

    def test_two_bad_in_five_fails(self):
        # 3 good, 2 bad → 60% < 80%
        data = _make_items(good=3, bad=2)
        valid, _ = _validate_with_tolerance(
            [{"name": str, "price": float}],
            data,
            Tolerance.BALANCED,
        )
        assert not valid


# ---------------------------------------------------------------------------
# Tolerant — 50% threshold
# ---------------------------------------------------------------------------

class TestTolerant:
    """Tolerant tolerance: 50% of items must pass."""

    def test_55_percent_passes(self):
        data = _make_items(good=55, bad=45)
        valid, _ = _validate_with_tolerance(
            [{"name": str, "price": float}],
            data,
            Tolerance.TOLERANT,
        )
        assert valid

    def test_40_percent_fails(self):
        data = _make_items(good=40, bad=60)
        valid, msg = _validate_with_tolerance(
            [{"name": str, "price": float}],
            data,
            Tolerance.TOLERANT,
        )
        assert not valid

    def test_exactly_50_percent_passes(self):
        data = _make_items(good=50, bad=50)
        valid, _ = _validate_with_tolerance(
            [{"name": str, "price": float}],
            data,
            Tolerance.TOLERANT,
        )
        assert valid


# ---------------------------------------------------------------------------
# Structural constraints — always enforced
# ---------------------------------------------------------------------------

class TestStructuralConstraints:
    """min/max on Items are never filtered by tolerance."""

    def test_min_still_enforced_with_tolerant(self):
        # Only 5 items, min is 20 — should fail regardless of tolerance
        data = _make_items(good=5, bad=0)
        valid, msg = _validate_with_tolerance(
            Items({"name": str, "price": float}, min_items=20),
            data,
            Tolerance.TOLERANT,
        )
        assert not valid
        assert "5" in msg and "20" in msg

    def test_empty_list_with_min_fails(self):
        valid, msg = _validate_with_tolerance(
            Items({"name": str}, min_items=1),
            [],
            Tolerance.TOLERANT,
        )
        assert not valid
        assert "empty" in msg

    def test_max_still_enforced_with_tolerant(self):
        data = _make_items(good=50, bad=0)
        valid, msg = _validate_with_tolerance(
            Items({"name": str, "price": float}, max_items=10),
            data,
            Tolerance.TOLERANT,
        )
        assert not valid
        assert "50" in msg


# ---------------------------------------------------------------------------
# Non-list schemas — tolerance has no effect
# ---------------------------------------------------------------------------

class TestNonListSchemas:
    """Tolerance only applies to lists."""

    def test_object_schema_unaffected(self):
        valid, _ = _validate_with_tolerance(
            {"name": str, "age": int},
            {"name": "Alice", "age": "thirty"},
            Tolerance.TOLERANT,
        )
        assert not valid

    def test_scalar_schema_unaffected(self):
        valid, _ = _validate_with_tolerance(
            Field(int, min=0),
            -1,
            Tolerance.TOLERANT,
        )
        assert not valid


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Boundary conditions and special cases."""

    def test_all_items_bad_fails_with_tolerant(self):
        data = _make_items(good=0, bad=10)
        valid, _ = _validate_with_tolerance(
            [{"name": str, "price": float}],
            data,
            Tolerance.TOLERANT,
        )
        assert not valid

    def test_single_good_item_passes(self):
        valid, _ = _validate_with_tolerance(
            [{"name": str}],
            [{"name": "Alice"}],
            Tolerance.BALANCED,
        )
        assert valid

    def test_single_bad_item_fails_balanced(self):
        # 0% pass rate < 80%
        valid, _ = _validate_with_tolerance(
            [{"name": str}],
            [{"name": 42}],
            Tolerance.BALANCED,
        )
        assert not valid

    def test_single_bad_item_fails_tolerant(self):
        # 0% pass rate < 50%
        valid, _ = _validate_with_tolerance(
            [{"name": str}],
            [{"name": 42}],
            Tolerance.TOLERANT,
        )
        assert not valid

    def test_no_errors_always_passes(self):
        data = _make_items(good=10, bad=0)
        valid, _ = _validate_with_tolerance(
            [{"name": str, "price": float}],
            data,
            Tolerance.STRICT,
        )
        assert valid

    def test_list_of_scalars(self):
        """Tolerance works for lists of scalars too."""
        # 8 valid ints, 2 strings → 80% pass rate
        data = [1, 2, 3, 4, 5, 6, 7, 8, "bad1", "bad2"]
        valid, _ = _validate_with_tolerance(
            [int],
            data,
            Tolerance.BALANCED,
        )
        assert valid

    def test_list_of_scalars_fails_below_threshold(self):
        # 3 valid, 7 bad → 30% < 80%
        data = [1, 2, 3, "a", "b", "c", "d", "e", "f", "g"]
        valid, _ = _validate_with_tolerance(
            [int],
            data,
            Tolerance.BALANCED,
        )
        assert not valid


# ---------------------------------------------------------------------------
# Nested lists
# ---------------------------------------------------------------------------

class TestNestedLists:
    """Tolerance applies at each list level independently."""

    def test_inner_list_tolerated(self):
        """Inner list errors tolerated, outer item passes."""
        schema = [{"tags": [str]}]
        # Item 0: 8 good tags, 2 bad → 80% pass rate for inner list
        data = [
            {"tags": ["a", "b", "c", "d", "e", "f", "g", "h", 1, 2]},
        ]
        valid, _ = _validate_with_tolerance(schema, data, Tolerance.BALANCED)
        assert valid

    def test_inner_list_not_tolerated(self):
        """Inner list too many errors → outer item fails."""
        schema = [{"tags": [str]}]
        # 2 good, 8 bad → 20% < 80%
        data = [
            {"tags": ["a", "b", 1, 2, 3, 4, 5, 6, 7, 8]},
        ]
        valid, _ = _validate_with_tolerance(schema, data, Tolerance.BALANCED)
        assert not valid

    def test_inner_list_min_violation_tolerated_at_outer_level(self):
        """Inner list min/max violation counts as one failed item in outer list."""
        schema = Items({"name": str, "reviews": Items({"text": str}, min_items=5)})
        # 9 products with enough reviews, 1 product with too few
        data = [
            {"name": f"Product {i}", "reviews": [{"text": f"r{j}"} for j in range(5)]}
            for i in range(9)
        ]
        data.append({"name": "Bad Product", "reviews": [{"text": "only one"}]})
        # 9/10 = 90% ≥ 80% → tolerated at outer level
        valid, _ = _validate_with_tolerance(schema, data, Tolerance.BALANCED)
        assert valid

    def test_inner_list_min_violation_fails_when_too_many(self):
        """Too many inner min violations → outer list fails tolerance."""
        schema = Items({"name": str, "reviews": Items({"text": str}, min_items=5)})
        # 5 good, 5 with too few reviews → 50% < 80%
        data = [
            {"name": f"Good {i}", "reviews": [{"text": f"r{j}"} for j in range(5)]}
            for i in range(5)
        ]
        data += [
            {"name": f"Bad {i}", "reviews": [{"text": "only one"}]}
            for i in range(5)
        ]
        valid, _ = _validate_with_tolerance(schema, data, Tolerance.BALANCED)
        assert not valid


# ---------------------------------------------------------------------------
# apply_tolerance internals
# ---------------------------------------------------------------------------

class TestApplyToleranceInternals:
    """Unit tests for internal helper functions."""

    def test_find_list_positions_top_level(self):
        root = parse_schema([{"x": str}])
        data = [{"x": "a"}, {"x": "b"}, {"x": "c"}]
        positions = _find_list_positions(root, data, "")
        assert ("", 3) in positions

    def test_find_list_positions_nested(self):
        root = parse_schema({"items": [{"x": str}]})
        data = {"items": [{"x": "a"}, {"x": "b"}]}
        positions = _find_list_positions(root, data, "")
        assert ("items", 2) in positions

    def test_find_list_positions_skips_scalar_items(self):
        """Should not recurse into items when item type is scalar."""
        root = parse_schema([int])
        data = list(range(500))
        positions = _find_list_positions(root, data, "")
        # Only one position (the list itself), not 500
        assert len(positions) == 1
        assert positions[0] == ("", 500)

    def test_partition_and_count_top_level(self):
        errors = [
            RawError("[0].name", "bad", None, "type"),
            RawError("[1].name", "bad", None, "type"),
            RawError("", "list too short", None, "structure"),
        ]
        other_errs, failed = _partition_and_count(errors, "")
        assert len(failed) == 2
        assert failed == {0, 1}
        assert len(other_errs) == 1

    def test_partition_and_count_nested_path(self):
        errors = [
            RawError("data[0].x", "bad", None, "type"),
            RawError("data[1].x", "bad", None, "type"),
            RawError("other", "unrelated", None, "type"),
        ]
        other_errs, failed = _partition_and_count(errors, "data")
        assert failed == {0, 1}
        assert len(other_errs) == 1

    def test_partition_and_count_multiple_errors_same_index(self):
        errors = [
            RawError("[0].name", "bad", None, "type"),
            RawError("[0].price", "bad", None, "type"),
            RawError("[3].name", "bad", None, "type"),
        ]
        other_errs, failed = _partition_and_count(errors, "")
        assert failed == {0, 3}
        assert len(other_errs) == 0


# ---------------------------------------------------------------------------
# CompiledSchema.validate() integration
# ---------------------------------------------------------------------------

class TestCompiledSchemaIntegration:
    """Test tolerance through the CompiledSchema.validate() API."""

    def test_none_tolerance_is_strict(self):
        """tolerance=None should behave like strict."""
        cs = compile_schema([{"x": int}])
        data = [{"x": 1}, {"x": "bad"}]
        result_none = cs.validate(data, tolerance=None)
        result_strict = cs.validate(data, tolerance=Tolerance.STRICT)
        assert result_none[0] == result_strict[0]  # both fail

    def test_balanced_passes_through_compiled(self):
        cs = compile_schema([{"x": int}])
        # 9 good, 1 bad → 90% ≥ 80%
        data = [{"x": i} for i in range(9)] + [{"x": "bad"}]
        valid, _ = cs.validate(data, tolerance=Tolerance.BALANCED)
        assert valid

    def test_multiple_error_types_counted_per_item(self):
        """An item with 2 errors counts as 1 failed item, not 2."""
        cs = compile_schema([{"name": Field(str, min_length=1), "age": int}])
        # 8 good items
        data = [{"name": f"Person {i}", "age": i + 20} for i in range(8)]
        # 2 bad items — each has TWO errors (wrong type for both fields)
        data += [{"name": "", "age": "old"}, {"name": "", "age": "young"}]
        valid, _ = cs.validate(data, tolerance=Tolerance.BALANCED)
        # 8/10 = 80% ≥ 80% → passes
        assert valid
