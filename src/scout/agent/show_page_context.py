"""Show-page context management.

Provides page similarity detection and state tracking for deciding
between Variant A (full analysis) and Variant B (page update) prompts.

Section reference extraction determines which sections the agent
referenced in its reasoning, using both direct section ID mentions
and interactive element attribute scoring.

Task 4 will extend this module with filtered output building.
"""

from __future__ import annotations

import re as _re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..page.converter import RenderedInteractiveElement

# Similarity threshold: pages with similarity >= this value relative to
# the last Variant A baseline are considered "same page" (Variant B).
SIMILARITY_THRESHOLD: float = 0.7

# Default neighbor radius for filtered output.
NEIGHBOR_RADIUS: int = 3


def page_similarity(current_text: str, previous_text: str) -> float:
    """Position-invariant similarity using word multisets (Jaccard on Counters).

    Splits both texts on whitespace, builds word frequency counters, and
    computes ``|intersection| / |union|`` where intersection and union
    are multiset operations (min / max per word).

    Properties:
        - Position-invariant: word order does not affect the score.
        - Fast: O(n) in total word count.
        - Content-based: compares actual words, not structural metadata.

    Returns:
        Float in [0.0, 1.0].  1.0 means identical word multisets;
        0.0 means no words in common.  Two empty strings return 1.0
        (both pages are equally "empty").
    """
    words_a = current_text.split()
    words_b = previous_text.split()

    counter_a = Counter(words_a)
    counter_b = Counter(words_b)

    shared = sum((counter_a & counter_b).values())
    total = sum((counter_a | counter_b).values())

    return shared / total if total > 0 else 1.0


class ShowPageState:
    """Tracks the last analyzed page text to decide prompt variant.

    The agent loop calls :meth:`should_force_full_analysis` on every
    ``show_page`` invocation.  If it returns ``True``, the orchestrator
    uses Variant A (full analysis) and afterward calls
    :meth:`mark_analyzed` with the raw page text.  If ``False``,
    Variant B (page update) is used and ``mark_analyzed`` is **not**
    called — see its docstring for the rationale.
    """

    def __init__(self) -> None:
        self.last_analyzed_text: str | None = None

    def should_force_full_analysis(self, current_text: str) -> bool:
        """Return ``True`` when the page needs a full Variant A analysis.

        Triggers on:
        - First ``show_page`` in the session (no baseline yet).
        - Page similarity to the last Variant A baseline is below
          :data:`SIMILARITY_THRESHOLD`.
        """
        if self.last_analyzed_text is None:
            return True
        similarity = page_similarity(current_text, self.last_analyzed_text)
        return similarity < SIMILARITY_THRESHOLD

    def mark_analyzed(self, raw_text: str) -> None:
        """Record *raw_text* as the new Variant A baseline.

        Called **only** after the agent completes a Variant A (full)
        analysis.  Variant B intentionally does **not** update the
        baseline so that drift detection works correctly: if the page
        changes incrementally across several Variant B turns, each
        subsequent similarity check still compares against the last
        full-analysis snapshot.  This prevents a sequence of small
        changes from silently accumulating into a large unanalyzed
        divergence.
        """
        self.last_analyzed_text = raw_text


# ---------------------------------------------------------------------------
# Section reference extraction — constants
# ---------------------------------------------------------------------------

# Weights reflect how strongly each attribute type identifies an element.
# Higher weight = more unique, more identifying.
ATTRIBUTE_WEIGHTS: dict[str, float] = {
    "data-testid": 1.0,   # Almost always unique on the page
    "aria-label": 0.8,    # Usually descriptive, fairly unique
    "id": 1.0,            # Unique by definition
    "name": 0.6,          # Moderately identifying
    "href": 0.7,          # Links are usually unique by destination
    "placeholder": 0.5,   # Somewhat identifying for inputs
    "type": 0.1,          # Too common ("button", "submit", "text")
    "role": 0.1,          # Too common ("button", "tab", "link")
}

# Minimum score for an element to be considered a match.
MATCH_THRESHOLD: float = 0.5

# Maximum sections to keep per matched element group.
MAX_SECTIONS_PER_ELEMENT: int = 3

# Attributes used to group structurally identical elements across sections.
STRUCTURAL_ATTRS: frozenset[str] = frozenset({
    "data-testid", "role", "type", "name",
})


# ---------------------------------------------------------------------------
# Section reference extraction — data structures
# ---------------------------------------------------------------------------

@dataclass
class ElementMatchResult:
    """Result of matching a single interactive element group to reasoning."""

    element: RenderedInteractiveElement
    score: float
    matched_attributes: list[dict]
    sections_with_same_element: list[str]
    sections_kept: list[str]
    was_capped: bool


# ---------------------------------------------------------------------------
# Section reference extraction — word boundary matching
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1024)
def _compile_boundary_pattern(value: str) -> _re.Pattern[str]:
    """Cache compiled regex patterns to avoid recompilation.

    A typical scoring pass checks ~200 elements x ~5 attributes = ~1000
    calls.  Without caching, each call compiles a fresh regex (parsing +
    NFA construction).
    """
    return _re.compile(r"(?<!\w)" + _re.escape(value) + r"(?!\w)")


def _word_boundary_match(value: str, text: str) -> bool:
    """Check if *value* appears in *text* as a distinct token.

    Uses word boundary detection: the character before and after the
    match (if any) must be non-alphanumeric.

    Examples::

        "search" in "I need to search for"  → True
        "search" in "the search button"     → True
        "search" in "researching the page"  → False
        "dir" in "the directory listing"    → False
        "dir" in "class=dir dir-ltr"        → True

    Caveat: hyphenated values can match inside longer hyphenated strings
    (e.g., ``"filter-button"`` matches in ``"my-filter-button-large"``)
    because hyphens are not ``\\w`` characters.  This is acceptable
    because:

    - Full attribute matches (``attr="value"``) are checked first and
      are unambiguous.
    - Value-only matches already carry a 0.6 weight penalty.
    - ``data-testid`` values are specific enough that partial hyphen
      matches are rare.
    """
    return bool(_compile_boundary_pattern(value).search(text))


# ---------------------------------------------------------------------------
# Section reference extraction — element scoring
# ---------------------------------------------------------------------------

def score_element(
    element: RenderedInteractiveElement,
    reasoning: str,
) -> tuple[float, list[dict]]:
    """Score how likely *reasoning* is referring to *element*.

    Returns ``(score, matched_attributes)`` where score is a float
    (higher = stronger match) and matched_attributes is a list of
    dicts describing each attribute that matched.
    """
    score = 0.0
    matched_attrs: list[dict] = []

    # Fast path: full tag match (agent copied the whole thing).
    if element.full_tag_str and len(element.full_tag_str) > 20:
        if element.full_tag_str in reasoning:
            return 2.0, [{"attr": "full-tag", "value": element.full_tag_str,
                          "match": "exact-tag", "weight": 2.0}]

    # Check each attribute.
    for attr_name, attr_value in element.attributes.items():
        if not attr_value or attr_name == "class":
            continue  # Classes handled separately below

        weight = ATTRIBUTE_WEIGHTS.get(attr_name, 0.3)

        # Check 1: Full attribute string — attr="value"
        full_attr = f'{attr_name}="{attr_value}"'
        if full_attr in reasoning:
            score += weight
            matched_attrs.append({
                "attr": attr_name, "value": attr_value,
                "match": "full", "weight": weight,
            })
            continue

        # Check 2: Value alone with word boundary check (>= 6 chars).
        if len(attr_value) >= 6 and _word_boundary_match(attr_value, reasoning):
            reduced_weight = weight * 0.6
            score += reduced_weight
            matched_attrs.append({
                "attr": attr_name, "value": attr_value,
                "match": "value-only", "weight": reduced_weight,
            })

    # Handle classes: fraction of ELIGIBLE classes mentioned.
    if element.classes:
        eligible_classes = [c for c in element.classes if len(c) >= 4]
        classes_found = [
            c for c in eligible_classes
            if _word_boundary_match(c, reasoning)
        ]
        if classes_found and eligible_classes:
            class_fraction = len(classes_found) / len(eligible_classes)
            class_weight = 0.4 * class_fraction
            score += class_weight
            matched_attrs.append({
                "attr": "class", "value": classes_found,
                "match": f"{len(classes_found)}/{len(element.classes)} classes",
                "weight": class_weight,
            })

    # Handle visible text with word boundary matching.
    if element.element_text and len(element.element_text) > 2:
        if _word_boundary_match(element.element_text, reasoning):
            score += 0.3
            matched_attrs.append({
                "attr": "text", "value": element.element_text,
                "match": "visible-text", "weight": 0.3,
            })

    return score, matched_attrs


def _element_signature(element: RenderedInteractiveElement) -> str:
    """Fingerprint that groups structurally identical elements.

    Uses tag name and structural attributes (data-testid, role, type,
    name) — NOT instance-varying attributes like href or aria-label.
    """
    parts = [element.tag]
    for attr in sorted(STRUCTURAL_ATTRS):
        if attr in element.attributes:
            parts.append(f"{attr}={element.attributes[attr]}")
    return "|".join(parts)


# ---------------------------------------------------------------------------
# Section reference extraction — matching pipeline
# ---------------------------------------------------------------------------

def match_elements_to_reasoning(
    reasoning: str,
    all_elements: list[tuple[str, RenderedInteractiveElement]],
) -> list[ElementMatchResult]:
    """Score all elements, threshold, group by signature, cap sections.

    Args:
        reasoning: The agent's analysis text.
        all_elements: List of ``(section_id, element)`` pairs.

    Returns:
        List of :class:`ElementMatchResult` for each matched group.
    """
    # Step 1: Score every element.
    scored: list[tuple[str, RenderedInteractiveElement, float, list[dict]]] = []
    for section_id, el in all_elements:
        el_score, matched_attrs = score_element(el, reasoning)
        if el_score >= MATCH_THRESHOLD:
            scored.append((section_id, el, el_score, matched_attrs))

    # Step 2: Group by structural signature.
    signature_groups: dict[str, list[tuple[str, RenderedInteractiveElement, float, list[dict]]]] = defaultdict(list)
    for section_id, el, el_score, matched_attrs in scored:
        sig = _element_signature(el)
        signature_groups[sig].append((section_id, el, el_score, matched_attrs))

    # Step 3: For each group, decide how many sections to keep.
    results: list[ElementMatchResult] = []
    for group in signature_groups.values():
        # Sort by score descending so group[0] is the best match.
        group.sort(key=lambda x: x[2], reverse=True)

        all_sections = [section_id for section_id, _, _, _ in group]
        occurrence_count = len(all_sections)

        if occurrence_count <= MAX_SECTIONS_PER_ELEMENT:
            sections_kept = all_sections
            was_capped = False
        else:
            sections_kept = all_sections[:MAX_SECTIONS_PER_ELEMENT]
            was_capped = True

        best_section_id, best_el, best_score, best_attrs = group[0]
        results.append(ElementMatchResult(
            element=best_el,
            score=best_score,
            matched_attributes=best_attrs,
            sections_with_same_element=all_sections,
            sections_kept=sections_kept,
            was_capped=was_capped,
        ))

    return results


def get_sections_by_id(
    reasoning: str,
    section_ids: list[str],
) -> set[str]:
    """Find section IDs directly mentioned in *reasoning*.

    Simple substring match.  Section IDs are long, descriptive strings
    (e.g., ``item-2-div-seite``, ``header-unterk-nfte``) that do not
    collide with natural language.
    """
    return {sid for sid in section_ids if sid in reasoning}


def get_referenced_sections(
    reasoning: str,
    sections: list[tuple[str, str, list[RenderedInteractiveElement]]],
) -> tuple[set[str], list[ElementMatchResult]]:
    """Determine which sections the agent referenced.

    Combines two mechanisms:

    1. **Direct section ID mention** — substring match of section IDs
       in the reasoning text.
    2. **Interactive element matching** — attribute scoring of elements
       in sections *not* already matched by ID.

    Args:
        reasoning: The agent's analysis text.
        sections: List of ``(section_id, content, elements)`` tuples.

    Returns:
        ``(referenced_ids, element_matches)`` tuple.
    """
    all_section_ids = [sid for sid, _, _ in sections]

    # Mechanism 1: Direct section ID mentions.
    referenced = get_sections_by_id(reasoning, all_section_ids)

    # Mechanism 2: Interactive element matching — only for unmatched sections.
    unmatched_elements: list[tuple[str, RenderedInteractiveElement]] = [
        (sid, el)
        for sid, _, elements in sections
        if sid not in referenced
        for el in elements
    ]
    element_matches = match_elements_to_reasoning(reasoning, unmatched_elements)

    # Add matched sections (respecting per-group cap).
    for match in element_matches:
        for section_id in match.sections_kept:
            referenced.add(section_id)

    return referenced, element_matches


# ---------------------------------------------------------------------------
# Filtered output builder
# ---------------------------------------------------------------------------

def build_filtered_output(
    sections: list[tuple[str, str]],
    referenced: set[str],
    neighbor_radius: int = NEIGHBOR_RADIUS,
) -> str:
    """Build filtered show_page output with neighbor-aware omission.

    Classifies each section into one of three tiers:

    - **Kept** — section ID is in *referenced*.  Full content is emitted
      wrapped in ``[section-id] ──\\n{content}\\n──``.
    - **Neighbor** — within *neighbor_radius* positions of a kept section.
      Emitted as ``[section-id — omitted]`` (ID only, no content).
    - **Distant** — everything else.  Consecutive distant sections are
      accumulated and flushed as ``[N sections omitted]``.

    Blocks are joined with ``\\n\\n`` for visual separation.

    Args:
        sections: ``(section_id, section_content)`` pairs in page order.
        referenced: Set of section IDs the agent referenced.
        neighbor_radius: How many positions around a kept section count
            as neighbors.  Defaults to :data:`NEIGHBOR_RADIUS`.

    Returns:
        The filtered output string.
    """
    kept_indices = {i for i, (sid, _) in enumerate(sections) if sid in referenced}

    neighbor_indices: set[int] = set()
    for ki in kept_indices:
        for offset in range(-neighbor_radius, neighbor_radius + 1):
            idx = ki + offset
            if 0 <= idx < len(sections) and idx not in kept_indices:
                neighbor_indices.add(idx)

    blocks: list[str] = []
    distant_count = 0

    for i, (section_id, section_content) in enumerate(sections):
        if i in kept_indices:
            if distant_count > 0:
                blocks.append(f"[{distant_count} sections omitted]")
                distant_count = 0
            blocks.append(f"[{section_id}] ──\n{section_content}\n──")

        elif i in neighbor_indices:
            if distant_count > 0:
                blocks.append(f"[{distant_count} sections omitted]")
                distant_count = 0
            blocks.append(f"[{section_id} — omitted]")

        else:
            distant_count += 1

    if distant_count > 0:
        blocks.append(f"[{distant_count} sections omitted]")

    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Observability — logging dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ElementMatch:
    """Log entry for a single interactive element match."""

    marker: str                      # e.g. 'aria-label="Weiter"'
    marker_type: str                 # "aria-label", "data-testid", etc.
    reasoning_context: str           # snippet around the marker
    matched_sections: list[str]      # ALL section IDs with this element
    is_ambiguous: bool               # True if > 1 section matched


@dataclass
class ShowPageAnalysisLog:
    """Structured log entry emitted after every show_page analysis cycle."""

    timestamp: float
    turn_number: int
    url: str

    # Similarity
    similarity_score: float          # 0.0–1.0
    variant_used: str                # "A" or "B"

    # Page content
    total_sections: int
    total_page_chars: int

    # Agent's analysis
    analysis_char_count: int

    # Section extraction results
    sections_mentioned_by_id: list[str]
    sections_matched_by_element: list[str]
    total_sections_kept: int
    total_sections_neighbor: int
    total_sections_distant: int

    # Filtered output
    filtered_page_chars: int
    compression_ratio: float         # filtered / original

    # Interactive element matching detail
    element_matches: list[ElementMatch]
