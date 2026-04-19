"""Stability assessor — determines how consistently errors repeat across attempts.

Compares fingerprints from 3 failed diagnostic attempts to classify the
failure pattern as STABLE, CONSISTENT, MIXED, or CHAOTIC. This feeds
into the decision engine to determine whether regeneration is warranted.

Stability levels (spec §8):
  STABLE:     All attempts match at Level 1 or 2 (exact/same-kind).
  CONSISTENT: All attempts share the same category (Level 3 match).
  MIXED:      Attempts span exactly 2 categories.
  CHAOTIC:    Attempts span 3+ categories, or 2 with no clear majority.

Spec reference: docs/specific/AUTO_FIX_ALGORITHM.md §8
"""

from __future__ import annotations

from collections import Counter

from scout.autofix.fingerprint import compare_fingerprints
from scout.autofix.types import (
    ComparisonLevel,
    Fingerprint,
    StabilityLevel,
)


def assess_stability(fingerprints: list[Fingerprint]) -> StabilityLevel:
    """Assess how consistently the same error repeats across attempts.

    Compares fingerprints pairwise and uses the minimum comparison level
    to determine stability.

    Args:
        fingerprints: Error fingerprints from all failed attempts
            (typically 3). Must have at least 2 entries.

    Returns:
        A ``StabilityLevel`` indicating pattern consistency.

    Raises:
        ValueError: If fewer than 2 fingerprints are provided.
    """
    if len(fingerprints) < 2:
        raise ValueError(
            f"Need at least 2 fingerprints for stability assessment, "
            f"got {len(fingerprints)}"
        )

    # Count distinct categories across all fingerprints.
    categories = Counter(fp.category for fp in fingerprints)
    num_categories = len(categories)

    # §8: 3+ categories → CHAOTIC
    if num_categories >= 3:
        return StabilityLevel.CHAOTIC

    # §8: 2 categories → MIXED or CHAOTIC (no clear majority)
    if num_categories == 2:
        # A "clear majority" means one category appears more than the other.
        # With 3 fingerprints: 2+1 = MIXED (clear majority of 2).
        # With 2 fingerprints: 1+1 = CHAOTIC (no majority).
        counts = categories.most_common()
        if counts[0][1] > counts[1][1]:
            return StabilityLevel.MIXED
        # Equal counts — no clear majority → CHAOTIC
        return StabilityLevel.CHAOTIC

    # All fingerprints share the same category. Now check comparison levels.
    # Compare all pairs and find the minimum comparison level.
    min_level = _min_comparison_level(fingerprints)

    # §8: All match at Level 1 (EXACT) or Level 2 (SAME_KIND) → STABLE
    if min_level in (ComparisonLevel.EXACT, ComparisonLevel.SAME_KIND):
        return StabilityLevel.STABLE

    # §8: Same category but Level 3 (SAME_CATEGORY) → CONSISTENT
    # This is the only remaining case when all categories match.
    return StabilityLevel.CONSISTENT


def _min_comparison_level(
    fingerprints: list[Fingerprint],
) -> ComparisonLevel:
    """Find the minimum comparison level across all pairs of fingerprints.

    Compares every pair (i, j) where i < j and returns the weakest match.
    With 3 fingerprints, this is 3 comparisons: (0,1), (0,2), (1,2).
    """
    # Order: EXACT > SAME_KIND > SAME_CATEGORY > NONE
    _LEVEL_ORDER = {
        ComparisonLevel.EXACT: 3,
        ComparisonLevel.SAME_KIND: 2,
        ComparisonLevel.SAME_CATEGORY: 1,
        ComparisonLevel.NONE: 0,
    }

    min_level = ComparisonLevel.EXACT
    min_rank = _LEVEL_ORDER[min_level]

    for i in range(len(fingerprints)):
        for j in range(i + 1, len(fingerprints)):
            level = compare_fingerprints(fingerprints[i], fingerprints[j])
            rank = _LEVEL_ORDER[level]
            if rank < min_rank:
                min_level = level
                min_rank = rank
                # Short-circuit: NONE means different categories,
                # but we already checked that all categories match.
                # SAME_CATEGORY is the lowest possible here.
                if min_level == ComparisonLevel.SAME_CATEGORY:
                    return min_level

    return min_level
