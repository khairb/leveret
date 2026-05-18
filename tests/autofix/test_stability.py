"""Tests for stability assessor (spec §8).

Tests every stability level with various fingerprint combinations.
Covers Level 1 (EXACT), Level 2 (SAME_KIND), Level 3 (SAME_CATEGORY),
and multi-category scenarios (MIXED, CHAOTIC).
"""

import pytest

from scout.autofix.fingerprint import compare_fingerprints
from scout.autofix.stability import assess_stability
from scout.autofix.types import (
    ComparisonLevel,
    ErrorCategory,
    Fingerprint,
    StabilityLevel,
)

# ── Helpers ──────────────────────────────────────────────────


def _fp(
    category: ErrorCategory,
    error_type: str | None = None,
    method: str | None = None,
    target: str | None = None,
) -> Fingerprint:
    """Shorthand for creating a Fingerprint."""
    return Fingerprint(
        category=category,
        error_type=error_type,
        method=method,
        target=target,
        message="",
    )


# ── STABLE: All match at Level 1 or 2 ───────────────────────


class TestStable:
    """§8: STABLE — all fingerprints match at Level 1 (EXACT) or Level 2 (SAME_KIND)."""

    def test_3x_exact_match_category_b(self):
        """3x identical AttributeError → STABLE."""
        fps = [
            _fp(ErrorCategory.B, "AttributeError", None, "text_content"),
            _fp(ErrorCategory.B, "AttributeError", None, "text_content"),
            _fp(ErrorCategory.B, "AttributeError", None, "text_content"),
        ]
        assert assess_stability(fps) == StabilityLevel.STABLE

    def test_3x_exact_match_category_d(self):
        """3x identical timeout on same selector → STABLE."""
        fps = [
            _fp(ErrorCategory.D, "TimeoutError", "Page.wait_for_selector", ".product-card"),
            _fp(ErrorCategory.D, "TimeoutError", "Page.wait_for_selector", ".product-card"),
            _fp(ErrorCategory.D, "TimeoutError", "Page.wait_for_selector", ".product-card"),
        ]
        assert assess_stability(fps) == StabilityLevel.STABLE

    def test_3x_exact_no_target(self):
        """3x same error_type + method, both targets None → STABLE (EXACT)."""
        fps = [
            _fp(ErrorCategory.D, "TimeoutError", "Page.click", None),
            _fp(ErrorCategory.D, "TimeoutError", "Page.click", None),
            _fp(ErrorCategory.D, "TimeoutError", "Page.click", None),
        ]
        assert assess_stability(fps) == StabilityLevel.STABLE

    def test_3x_same_kind_different_targets(self):
        """3x same timeout method, different selectors → STABLE (Level 2).

        Per spec §8: 3x TimeoutError from wait_for_selector on different
        selectors (.product-card, .price-tag, .author) → SAME_KIND → STABLE.
        """
        fps = [
            _fp(ErrorCategory.D, "TimeoutError", "Page.wait_for_selector", ".product-card"),
            _fp(ErrorCategory.D, "TimeoutError", "Page.wait_for_selector", ".price-tag"),
            _fp(ErrorCategory.D, "TimeoutError", "Page.wait_for_selector", ".author"),
        ]
        assert assess_stability(fps) == StabilityLevel.STABLE

    def test_3x_category_b_same_kind(self):
        """3x AttributeError on different attributes → STABLE (Level 2)."""
        fps = [
            _fp(ErrorCategory.B, "AttributeError", None, "text_content"),
            _fp(ErrorCategory.B, "AttributeError", None, "inner_html"),
            _fp(ErrorCategory.B, "AttributeError", None, "get_attribute"),
        ]
        assert assess_stability(fps) == StabilityLevel.STABLE

    def test_2_fingerprints_exact(self):
        """2 fingerprints (early exit scenario) — both exact → STABLE."""
        fps = [
            _fp(ErrorCategory.B, "KeyError", None, "price"),
            _fp(ErrorCategory.B, "KeyError", None, "price"),
        ]
        assert assess_stability(fps) == StabilityLevel.STABLE

    def test_2_fingerprints_same_kind(self):
        """2 fingerprints, same kind → STABLE."""
        fps = [
            _fp(ErrorCategory.B, "KeyError", None, "price"),
            _fp(ErrorCategory.B, "KeyError", None, "title"),
        ]
        assert assess_stability(fps) == StabilityLevel.STABLE

    def test_mixed_exact_and_same_kind(self):
        """2 exact + 1 same_kind → STABLE (min level is SAME_KIND)."""
        fps = [
            _fp(ErrorCategory.D, "TimeoutError", "Page.click", ".btn"),
            _fp(ErrorCategory.D, "TimeoutError", "Page.click", ".btn"),
            _fp(ErrorCategory.D, "TimeoutError", "Page.click", ".button"),
        ]
        assert assess_stability(fps) == StabilityLevel.STABLE

    def test_category_g_stable(self):
        """3x same schema validation failure → STABLE."""
        fps = [
            _fp(ErrorCategory.G, "SchemaValidationError", None, "item_count < 5"),
            _fp(ErrorCategory.G, "SchemaValidationError", None, "item_count < 5"),
            _fp(ErrorCategory.G, "SchemaValidationError", None, "item_count < 5"),
        ]
        assert assess_stability(fps) == StabilityLevel.STABLE

    def test_category_e_stable(self):
        """3x pointer intercept → STABLE."""
        fps = [
            _fp(ErrorCategory.E, "pointer_intercept", "Page.click", "<div>"),
            _fp(ErrorCategory.E, "pointer_intercept", "Page.click", "<div>"),
            _fp(ErrorCategory.E, "pointer_intercept", "Page.click", "<div>"),
        ]
        assert assess_stability(fps) == StabilityLevel.STABLE

    def test_category_c_stable(self):
        """3x same network error → STABLE."""
        fps = [
            _fp(ErrorCategory.C, "net::ERR_CONNECTION_REFUSED", "Page.goto", None),
            _fp(ErrorCategory.C, "net::ERR_CONNECTION_REFUSED", "Page.goto", None),
            _fp(ErrorCategory.C, "net::ERR_CONNECTION_REFUSED", "Page.goto", None),
        ]
        assert assess_stability(fps) == StabilityLevel.STABLE

    def test_category_f1_stable(self):
        """3x same browser crash → STABLE."""
        fps = [
            _fp(ErrorCategory.F1, "TargetClosedError", None, None),
            _fp(ErrorCategory.F1, "TargetClosedError", None, None),
            _fp(ErrorCategory.F1, "TargetClosedError", None, None),
        ]
        assert assess_stability(fps) == StabilityLevel.STABLE

    def test_one_target_none_one_present(self):
        """One target None, one present → SAME_KIND (degraded) → STABLE."""
        fps = [
            _fp(ErrorCategory.D, "TimeoutError", "Page.click", ".btn"),
            _fp(ErrorCategory.D, "TimeoutError", "Page.click", None),
            _fp(ErrorCategory.D, "TimeoutError", "Page.click", ".btn"),
        ]
        # (0,1) → SAME_KIND, (0,2) → EXACT, (1,2) → SAME_KIND → min = SAME_KIND
        assert assess_stability(fps) == StabilityLevel.STABLE


# ── CONSISTENT: All same category, Level 3 ──────────────────


class TestConsistent:
    """§8: CONSISTENT — all same category but different error types/methods."""

    def test_category_c_different_errors(self):
        """3x Category C, different net::ERR_* codes → CONSISTENT.

        Per spec §8: ERR_CONNECTION_REFUSED, ERR_CONNECTION_RESET,
        ERR_CONNECTION_TIMED_OUT — network is broken in varying ways.
        """
        fps = [
            _fp(ErrorCategory.C, "net::ERR_CONNECTION_REFUSED", "Page.goto", None),
            _fp(ErrorCategory.C, "net::ERR_CONNECTION_RESET", "Page.goto", None),
            _fp(ErrorCategory.C, "net::ERR_CONNECTION_TIMED_OUT", "Page.goto", None),
        ]
        assert assess_stability(fps) == StabilityLevel.CONSISTENT

    def test_category_b_different_types(self):
        """3x Category B, different exception types → CONSISTENT.

        Per spec §8: AttributeError, KeyError, TypeError — script crashes
        in different places.
        """
        fps = [
            _fp(ErrorCategory.B, "AttributeError", None, "text_content"),
            _fp(ErrorCategory.B, "KeyError", None, "price"),
            _fp(ErrorCategory.B, "TypeError", None, None),
        ]
        assert assess_stability(fps) == StabilityLevel.CONSISTENT

    def test_category_d_different_methods(self):
        """3x Category D, different methods → CONSISTENT."""
        fps = [
            _fp(ErrorCategory.D, "TimeoutError", "Page.wait_for_selector", ".card"),
            _fp(ErrorCategory.D, "TimeoutError", "Page.click", ".btn"),
            _fp(ErrorCategory.D, "TimeoutError", "Locator.wait_for", ".item"),
        ]
        assert assess_stability(fps) == StabilityLevel.CONSISTENT

    def test_category_e_different_subtypes(self):
        """3x Category E, different sub-types → CONSISTENT."""
        fps = [
            _fp(ErrorCategory.E, "pointer_intercept", "Page.click", None),
            _fp(ErrorCategory.E, "not_visible", "Page.click", None),
            _fp(ErrorCategory.E, "not_enabled", "Page.fill", None),
        ]
        assert assess_stability(fps) == StabilityLevel.CONSISTENT

    def test_2_fingerprints_consistent(self):
        """2 fingerprints, same category, different error_type → CONSISTENT."""
        fps = [
            _fp(ErrorCategory.B, "AttributeError", None, None),
            _fp(ErrorCategory.B, "TypeError", None, None),
        ]
        assert assess_stability(fps) == StabilityLevel.CONSISTENT

    def test_same_error_type_different_method(self):
        """Same error_type but different method → SAME_CATEGORY → CONSISTENT."""
        fps = [
            _fp(ErrorCategory.D, "TimeoutError", "Page.click", ".btn"),
            _fp(ErrorCategory.D, "TimeoutError", "Page.fill", ".input"),
            _fp(ErrorCategory.D, "TimeoutError", "Locator.wait_for", ".card"),
        ]
        assert assess_stability(fps) == StabilityLevel.CONSISTENT

    def test_category_g_different_targets_same_type(self):
        """3x Category G, same error_type, different targets → SAME_KIND → STABLE.

        Schema validation errors with different specific constraints but same type.
        """
        fps = [
            _fp(ErrorCategory.G, "SchemaValidationError", None, "item_count < 5"),
            _fp(ErrorCategory.G, "SchemaValidationError", None, "missing field: title"),
            _fp(ErrorCategory.G, "SchemaValidationError", None, "type mismatch: price"),
        ]
        # All have same error_type + method(None) → SAME_KIND → STABLE
        assert assess_stability(fps) == StabilityLevel.STABLE


# ── MIXED: Exactly 2 categories ─────────────────────────────


class TestMixed:
    """§8: MIXED — attempts span exactly 2 categories with clear majority."""

    def test_2x_d_1x_c(self):
        """2x timeout + 1x network → MIXED.

        Per spec §8: network blip on attempt 2 was noise; timeout is the
        real issue.
        """
        fps = [
            _fp(ErrorCategory.D, "TimeoutError", "Page.click", ".btn"),
            _fp(ErrorCategory.C, "net::ERR_CONNECTION_RESET", "Page.goto", None),
            _fp(ErrorCategory.D, "TimeoutError", "Page.click", ".btn"),
        ]
        assert assess_stability(fps) == StabilityLevel.MIXED

    def test_2x_b_1x_d(self):
        """2x crash + 1x timeout → MIXED."""
        fps = [
            _fp(ErrorCategory.B, "AttributeError", None, "text_content"),
            _fp(ErrorCategory.D, "TimeoutError", "Page.click", ".btn"),
            _fp(ErrorCategory.B, "AttributeError", None, "text_content"),
        ]
        assert assess_stability(fps) == StabilityLevel.MIXED

    def test_2x_b_1x_e(self):
        """2x crash + 1x page state → MIXED."""
        fps = [
            _fp(ErrorCategory.B, "AttributeError", None, None),
            _fp(ErrorCategory.B, "KeyError", None, None),
            _fp(ErrorCategory.E, "pointer_intercept", "Page.click", None),
        ]
        assert assess_stability(fps) == StabilityLevel.MIXED

    def test_2x_c_1x_f1(self):
        """2x network + 1x crash → MIXED."""
        fps = [
            _fp(ErrorCategory.C, "net::ERR_CONNECTION_REFUSED", None, None),
            _fp(ErrorCategory.C, "net::ERR_NAME_NOT_RESOLVED", None, None),
            _fp(ErrorCategory.F1, "TargetClosedError", None, None),
        ]
        assert assess_stability(fps) == StabilityLevel.MIXED


# ── CHAOTIC: 3 categories or 2 with no majority ─────────────


class TestChaotic:
    """§8: CHAOTIC — 3+ categories, or 2 with no clear majority."""

    def test_3_different_categories(self):
        """B + D + C → CHAOTIC.

        Per spec §8: something different goes wrong every time.
        """
        fps = [
            _fp(ErrorCategory.B, "AttributeError", None, None),
            _fp(ErrorCategory.D, "TimeoutError", "Page.click", None),
            _fp(ErrorCategory.C, "net::ERR_CONNECTION_REFUSED", None, None),
        ]
        assert assess_stability(fps) == StabilityLevel.CHAOTIC

    def test_3_different_categories_variant(self):
        """E + F1 + G → CHAOTIC."""
        fps = [
            _fp(ErrorCategory.E, "pointer_intercept", None, None),
            _fp(ErrorCategory.F1, "TargetClosedError", None, None),
            _fp(ErrorCategory.G, "SchemaValidationError", None, None),
        ]
        assert assess_stability(fps) == StabilityLevel.CHAOTIC

    def test_2_categories_equal_split_with_3(self):
        """If 3 fingerprints have 2 categories but counts are 2+1,
        that's MIXED (clear majority of 2), not CHAOTIC."""
        fps = [
            _fp(ErrorCategory.B, "AttributeError", None, None),
            _fp(ErrorCategory.B, "TypeError", None, None),
            _fp(ErrorCategory.D, "TimeoutError", None, None),
        ]
        # 2x B + 1x D → clear majority → MIXED
        assert assess_stability(fps) == StabilityLevel.MIXED

    def test_2_categories_no_majority_2_fps(self):
        """2 fingerprints, different categories → CHAOTIC (1+1, no majority)."""
        fps = [
            _fp(ErrorCategory.B, "AttributeError", None, None),
            _fp(ErrorCategory.D, "TimeoutError", None, None),
        ]
        assert assess_stability(fps) == StabilityLevel.CHAOTIC

    def test_4_fingerprints_2_categories_equal(self):
        """4 fingerprints, 2+2 split → CHAOTIC (no majority)."""
        fps = [
            _fp(ErrorCategory.B, "AttributeError", None, None),
            _fp(ErrorCategory.B, "TypeError", None, None),
            _fp(ErrorCategory.D, "TimeoutError", None, None),
            _fp(ErrorCategory.D, "TimeoutError", None, None),
        ]
        assert assess_stability(fps) == StabilityLevel.CHAOTIC

    def test_4_fingerprints_3_categories(self):
        """4 fingerprints, 3 categories → CHAOTIC."""
        fps = [
            _fp(ErrorCategory.B, "AttributeError", None, None),
            _fp(ErrorCategory.D, "TimeoutError", None, None),
            _fp(ErrorCategory.C, "net::ERR_REFUSED", None, None),
            _fp(ErrorCategory.B, "KeyError", None, None),
        ]
        assert assess_stability(fps) == StabilityLevel.CHAOTIC


# ── Edge cases ───────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases for stability assessment."""

    def test_fewer_than_2_fingerprints_raises(self):
        """Must have at least 2 fingerprints."""
        with pytest.raises(ValueError, match="at least 2"):
            assess_stability([_fp(ErrorCategory.B)])

    def test_empty_fingerprints_raises(self):
        """Empty list raises ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            assess_stability([])

    def test_5_fingerprints_all_same(self):
        """5 identical fingerprints → STABLE."""
        fp = _fp(ErrorCategory.B, "AttributeError", None, "text_content")
        assert assess_stability([fp] * 5) == StabilityLevel.STABLE

    def test_degraded_fingerprints_no_error_type(self):
        """Fingerprints with None error_type — same category → CONSISTENT."""
        fps = [
            _fp(ErrorCategory.B, None, None, None),
            _fp(ErrorCategory.B, "AttributeError", None, None),
            _fp(ErrorCategory.B, "TypeError", None, None),
        ]
        # (0,1): same cat, different error_type → SAME_CATEGORY
        assert assess_stability(fps) == StabilityLevel.CONSISTENT

    def test_all_none_fields(self):
        """Fingerprints with all None fields, same category → STABLE (EXACT)."""
        fps = [
            _fp(ErrorCategory.B, None, None, None),
            _fp(ErrorCategory.B, None, None, None),
            _fp(ErrorCategory.B, None, None, None),
        ]
        assert assess_stability(fps) == StabilityLevel.STABLE

    def test_comparison_level_consistency(self):
        """Verify that assess_stability agrees with compare_fingerprints."""
        fp1 = _fp(ErrorCategory.D, "TimeoutError", "Page.click", ".btn")
        fp2 = _fp(ErrorCategory.D, "TimeoutError", "Page.click", ".btn")
        fp3 = _fp(ErrorCategory.D, "TimeoutError", "Page.click", ".btn")

        assert compare_fingerprints(fp1, fp2) == ComparisonLevel.EXACT
        assert compare_fingerprints(fp1, fp3) == ComparisonLevel.EXACT
        assert compare_fingerprints(fp2, fp3) == ComparisonLevel.EXACT
        assert assess_stability([fp1, fp2, fp3]) == StabilityLevel.STABLE

    def test_one_pair_downgrades_to_same_kind(self):
        """If 2 pairs are EXACT but 1 is SAME_KIND, min is SAME_KIND → STABLE."""
        fps = [
            _fp(ErrorCategory.D, "TimeoutError", "Page.click", ".btn-a"),
            _fp(ErrorCategory.D, "TimeoutError", "Page.click", ".btn-a"),
            _fp(ErrorCategory.D, "TimeoutError", "Page.click", ".btn-b"),
        ]
        assert assess_stability(fps) == StabilityLevel.STABLE

    def test_one_pair_downgrades_to_same_category(self):
        """If any pair is SAME_CATEGORY, min is SAME_CATEGORY → CONSISTENT."""
        fps = [
            _fp(ErrorCategory.D, "TimeoutError", "Page.click", ".btn"),
            _fp(ErrorCategory.D, "TimeoutError", "Page.click", ".btn"),
            _fp(ErrorCategory.D, "TimeoutError", "Page.fill", ".input"),
        ]
        # (0,2) and (1,2): different method → SAME_CATEGORY
        assert assess_stability(fps) == StabilityLevel.CONSISTENT
