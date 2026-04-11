"""Repeated Pattern Detection for DOM Trees

Detects groups of structurally similar sibling elements in a parsed HTML
document.  This allows the sectioner to keep semantic units (product cards,
article listings, search results) intact instead of fragmenting them by
character count.

The algorithm is two-phase:

1. **Exact signature** — groups siblings that share the same tag, class set,
   and child-tag structure (with grandchild counts for depth-2 resolution).
   Catches ~90 % of real-world repeated elements rendered from the same
   component template.

2. **Structural fallback** — for elements not matched in phase 1, groups by
   tag + child-tag structure alone (ignoring classes), then validates with a
   class-overlap check.  Catches class variations, CSS-in-JS hash
   differences, and classless markup.

Detection is recursive: every parent in the DOM is checked, so nested
repetition (categories containing products) is found at every level.

Usage::

    from sectioning.repeated_patterns import detect_all_groups

    # root is an lxml HtmlElement, emap is the sectioner's _ElementMap
    annotations = detect_all_groups(root, emap)
    annotations.is_atomic(element)           # True if member of a group
    annotations.get_groups(parent_element)   # groups under this parent
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from lxml.html import HtmlElement


# ═══════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RepeatedGroup:
    """A set of structurally similar sibling elements under one parent."""

    parent: HtmlElement
    members: list[HtmlElement]
    signature: tuple                   # the shared canonical signature
    member_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.member_count = len(self.members)


@dataclass
class GroupAnnotations:
    """Lookup structure for repeated-group membership.

    Populated by :func:`detect_all_groups` and consumed by the sectioner.
    """

    # element id() → the RepeatedGroup this element belongs to
    _atomic: dict[int, RepeatedGroup] = field(default_factory=dict)
    # parent element id() → list of groups among its children
    _parent_groups: dict[int, list[RepeatedGroup]] = field(default_factory=dict)

    def is_atomic(self, el: HtmlElement) -> bool:
        """True if *el* is a member of a detected repeated group."""
        return id(el) in self._atomic

    def get_group(self, el: HtmlElement) -> Optional[RepeatedGroup]:
        """Return the group *el* belongs to, or None."""
        return self._atomic.get(id(el))

    def get_groups_for_parent(self, parent: HtmlElement) -> list[RepeatedGroup]:
        """Return all repeated groups among *parent*'s direct children."""
        return self._parent_groups.get(id(parent), [])

    def register_group(self, group: RepeatedGroup) -> None:
        """Add a group to the annotations."""
        parent_key = id(group.parent)
        if parent_key not in self._parent_groups:
            self._parent_groups[parent_key] = []
        self._parent_groups[parent_key].append(group)
        for member in group.members:
            self._atomic[id(member)] = group

    @property
    def total_groups(self) -> int:
        return sum(len(gs) for gs in self._parent_groups.values())

    @property
    def total_atomic_elements(self) -> int:
        return len(self._atomic)


# ═══════════════════════════════════════════════════════════════════════════
#  Signature Computation
# ═══════════════════════════════════════════════════════════════════════════

def _element_children(el: HtmlElement) -> list[HtmlElement]:
    """Direct element children (skipping comments, PIs, text nodes)."""
    return [c for c in el if isinstance(c.tag, str)]


def _child_shape(el: HtmlElement) -> tuple[tuple[str, int], ...]:
    """Depth-2 structural shape: direct children as (tag, grandchild_count).

    The grandchild count is a cheap depth-2 signal that distinguishes
    elements with the same direct children but different internal complexity.
    For example ``div>(div[3], div[3])`` vs ``div>(div[1], div[10])``.
    """
    result = []
    for child in _element_children(el):
        grandchild_count = sum(1 for gc in child if isinstance(gc.tag, str))
        result.append((child.tag, grandchild_count))
    return tuple(result)


def compute_exact_signature(el: HtmlElement) -> tuple:
    """Exact canonical signature: tag + classes + child shape.

    Two elements with the same exact signature are considered identical
    in structure.  This catches template-rendered components where the
    same React/Vue/server component produces identical DOM structure
    with identical CSS classes.
    """
    tag = el.tag if isinstance(el.tag, str) else ""
    classes = frozenset((el.get("class") or "").split())
    shape = _child_shape(el)
    return (tag, classes, shape)


def compute_shape_signature(el: HtmlElement) -> tuple:
    """Structural shape signature: tag + child tags only (no classes).

    Used as a fallback when exact signatures don't produce groups.
    Catches cases where classes vary between otherwise identical elements
    (e.g. ``product featured`` vs ``product standard``).
    """
    tag = el.tag if isinstance(el.tag, str) else ""
    child_tags = tuple(
        c.tag for c in _element_children(el)
    )
    return (tag, child_tags)


# ═══════════════════════════════════════════════════════════════════════════
#  Class Similarity Validation
# ═══════════════════════════════════════════════════════════════════════════

def _validate_class_similarity(
    members: list[HtmlElement],
    threshold: float = 0.3,
) -> bool:
    """Check that members have sufficient class overlap.

    Uses a three-tier approach to handle real-world class patterns:

    1. **Core-class check** — if any CSS classes appear in more than half
       the members, accept.  Handles WordPress where structural classes
       (``wp-block-post``, ``hentry``) are shared but per-item metadata
       classes (``category-*``, ``tag-*``, ``post-NNNN``) vary.

    2. **Large-group bypass** — if the group has 5+ members with
       identical shape but zero shared classes, accept.  This is the
       CSS-in-JS pattern (Emotion, styled-components) where each element
       gets a unique hash class.

    3. **Conservative Jaccard fallback** — for small groups (3-4) with
       no shared classes, apply the old intersection/union Jaccard check
       with a lowered threshold.  Guards against false positives like
       three layout containers (header, main, footer) that happen to
       share the same child-tag shape.
    """
    class_sets = [
        frozenset((m.get("class") or "").split())
        for m in members
    ]

    # All classless → structural match is sufficient.
    non_empty = [cs for cs in class_sets if cs]
    if not non_empty:
        return True

    # Fewer than half have classes → classes are incidental, trust shape.
    if len(non_empty) < len(class_sets) / 2:
        return True

    # ── Tier 1: Core-class check ──
    # Count how often each class appears across members.
    class_freq: Counter[str] = Counter()
    for cs in class_sets:
        for cls in cs:
            class_freq[cls] += 1

    majority = len(members) / 2
    has_core = any(count > majority for count in class_freq.values())

    if has_core:
        return True

    # ── Tier 2: Large-group bypass ──
    # 5+ siblings with identical shape is overwhelmingly likely to be
    # a repeated group, even with fully unique classes.
    if len(members) >= 5:
        return True

    # ── Tier 3: Conservative Jaccard for small groups (3-4) ──
    all_intersection = class_sets[0]
    all_union = class_sets[0]
    for cs in class_sets[1:]:
        all_intersection = all_intersection & cs
        all_union = all_union | cs

    if not all_union:
        return True

    return len(all_intersection) / len(all_union) >= threshold


# ═══════════════════════════════════════════════════════════════════════════
#  Group Detection (per parent)
# ═══════════════════════════════════════════════════════════════════════════

def find_repeated_groups(
    parent: HtmlElement,
    sizes: dict[int, int],
    *,
    min_group_size: int = 3,
) -> list[RepeatedGroup]:
    """Find groups of structurally similar children under *parent*.

    Only considers children with non-zero size (visible content).

    Args:
        parent: The parent element to examine.
        sizes: Precomputed element sizes (from sectioner's _compute_sizes).
        min_group_size: Minimum members to form a group (default 3).

    Returns:
        List of :class:`RepeatedGroup` objects.  An element appears in
        at most one group.
    """
    children = [
        c for c in _element_children(parent)
        if sizes.get(id(c), 0) > 0
    ]

    if len(children) < min_group_size:
        return []

    # Track which children have been assigned to a group
    grouped_ids: set[int] = set()
    groups: list[RepeatedGroup] = []

    # ── Phase 1: Exact signature grouping ──
    exact_buckets: dict[tuple, list[HtmlElement]] = {}
    for child in children:
        sig = compute_exact_signature(child)
        exact_buckets.setdefault(sig, []).append(child)

    for sig, members in exact_buckets.items():
        if len(members) >= min_group_size:
            group = RepeatedGroup(
                parent=parent,
                members=members,
                signature=sig,
            )
            groups.append(group)
            grouped_ids.update(id(m) for m in members)

    # ── Phase 2: Structural fallback for ungrouped elements ──
    ungrouped = [c for c in children if id(c) not in grouped_ids]
    if len(ungrouped) >= min_group_size:
        shape_buckets: dict[tuple, list[HtmlElement]] = {}
        for child in ungrouped:
            sig = compute_shape_signature(child)
            shape_buckets.setdefault(sig, []).append(child)

        for sig, members in shape_buckets.items():
            if len(members) >= min_group_size:
                if _validate_class_similarity(members):
                    group = RepeatedGroup(
                        parent=parent,
                        members=members,
                        signature=sig,
                    )
                    groups.append(group)
                    grouped_ids.update(id(m) for m in members)

    return groups


# ═══════════════════════════════════════════════════════════════════════════
#  Recursive Detection (whole DOM)
# ═══════════════════════════════════════════════════════════════════════════

def detect_all_groups(
    root: HtmlElement,
    sizes: dict[int, int],
    *,
    min_group_size: int = 3,
) -> GroupAnnotations:
    """Detect repeated groups at every level of the DOM tree.

    Walks every element and checks its direct children for repeated
    patterns.  Returns :class:`GroupAnnotations` that the sectioner
    uses to keep semantic units intact.

    Complexity: O(N) where N is total element count — each element is
    visited once as a potential parent, and its children are hashed
    once for signature computation.
    """
    annotations = GroupAnnotations()

    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        groups = find_repeated_groups(el, sizes, min_group_size=min_group_size)
        for group in groups:
            annotations.register_group(group)

    return annotations
