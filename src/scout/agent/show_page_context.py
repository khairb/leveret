"""Show-page context management.

Provides page similarity detection and state tracking for deciding
between Variant A (full analysis) and Variant B (page update) prompts.

Section reference extraction determines which sections the agent
referenced in its reasoning, using both direct section ID mentions
and interactive element attribute scoring.

Task 4 will extend this module with filtered output building.
"""

from __future__ import annotations

import math as _math
import re as _re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..page.converter import RenderedInteractiveElement

# Similarity threshold: pages with similarity >= this value relative to
# the last Variant A baseline are considered "same page" (Variant B).
SIMILARITY_THRESHOLD: float = 0.8

# Default neighbor radius for filtered output.
NEIGHBOR_RADIUS: int = 3

# ---------------------------------------------------------------------------
# Indirect reference detection — constants
# ---------------------------------------------------------------------------

# Budget cap: keep at most 20% of total sections, hard-capped at 15.
INDIRECT_MAX_RATIO: float = 0.20
INDIRECT_MAX_ABSOLUTE: int = 15

# Narrower neighbor radius for indirectly referenced sections.
INDIRECT_NEIGHBOR_RADIUS: int = 1

# Minimum token length when decomposing section IDs.
MIN_ID_TOKEN_LENGTH: int = 4

# Characters for start/end preview snippets in skeleton output.
SKELETON_PREVIEW_CHARS: int = 80


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
        # Variant B carry-forward: sections kept/indirect from last Variant A.
        self.last_variant_a_kept: set[str] = set()
        self.last_variant_a_indirect: set[str] = set()

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


@dataclass
class IndirectMatchResult:
    """Result of matching a section via ID token or content keyword overlap."""

    section_id: str
    score: float                 # IDF-weighted, 0.0–1.0 normalised
    matched_tokens: list[str]
    match_source: str            # "id_tokens" or "content"


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
# Indirect reference detection — token decomposition & IDF scoring
# ---------------------------------------------------------------------------

def _tokenize_section_id(section_id: str) -> list[str]:
    """Decompose a section ID into meaningful tokens.

    Splits on ``-`` and ``_``, filters tokens shorter than
    :data:`MIN_ID_TOKEN_LENGTH`, lowercases, and deduplicates
    (preserving first-occurrence order).  Numeric-only tokens
    are also dropped.

    Examples::

        "sidebar-about"         → ["sidebar", "about"]
        "div-hq-nav"            → []  (all < 4 chars)
        "item-1-article-anthropics-financial"
            → ["item", "article", "anthropics", "financial"]
        "item-this-week-this-month"
            → ["item", "this", "week", "month"]  (deduplicated)
    """
    parts = _re.split(r"[-_]", section_id.lower())
    seen: set[str] = set()
    tokens: list[str] = []
    for p in parts:
        if (
            len(p) >= MIN_ID_TOKEN_LENGTH
            and not p.isdigit()
            and p not in seen
        ):
            seen.add(p)
            tokens.append(p)
    return tokens


def _compute_token_idf(
    all_sections: list[tuple[str, str, str, int]],
) -> dict[str, float]:
    """Compute IDF for each token across all section IDs and content.

    ``idf(token) = log(total_sections / sections_containing_token)``

    Tokens appearing in every section get ``idf ≈ 0`` (worthless).
    Tokens unique to one section get ``idf = log(N)`` (highly identifying).

    Content tokens are drawn from the first 200 characters of each
    section — enough to capture headings without scoring entire
    paragraphs.
    """
    total = len(all_sections)
    if total == 0:
        return {}
    doc_count: Counter[str] = Counter()
    for sid, content, _role, _ic in all_sections:
        # Unique tokens from this section's ID + leading content.
        tokens = set(_tokenize_section_id(sid))
        for w in content[:200].split():
            w_lower = w.lower()
            if len(w_lower) >= MIN_ID_TOKEN_LENGTH and w_lower.isalpha():
                tokens.add(w_lower)
        for t in tokens:
            doc_count[t] += 1
    # Smoothed IDF: log(1 + total/count) — always > 0, so even
    # ubiquitous tokens contribute a small positive weight rather
    # than collapsing the score to zero.
    return {
        token: _math.log(1 + total / count)
        for token, count in doc_count.items()
    }


def score_section_indirect(
    section_id: str,
    content: str,
    reasoning_lower: str,
    idf: dict[str, float],
) -> tuple[float, list[str], str]:
    """Score a section against the AI's reasoning via IDF-weighted tokens.

    Two sources are scored independently — section ID tokens and leading
    content keywords — and the better-scoring source wins.

    Returns ``(score, matched_tokens, match_source)`` where *score* is
    normalised to [0.0, 1.0] and *match_source* is ``"id_tokens"`` or
    ``"content"``.
    """
    # --- ID tokens ---
    id_tokens = _tokenize_section_id(section_id)
    id_matched = [
        t for t in id_tokens if _word_boundary_match(t, reasoning_lower)
    ]
    id_total_idf = sum(idf.get(t, 1.0) for t in id_tokens)
    id_matched_idf = sum(idf.get(t, 1.0) for t in id_matched)
    id_score = id_matched_idf / id_total_idf if id_total_idf > 0 else 0.0

    # --- Content keywords (first 200 chars, up to 8 unique tokens) ---
    seen: set[str] = set()
    content_tokens: list[str] = []
    for w in content[:200].split():
        w_lower = w.lower()
        if (
            len(w_lower) >= MIN_ID_TOKEN_LENGTH
            and w_lower.isalpha()
            and w_lower not in seen
        ):
            seen.add(w_lower)
            content_tokens.append(w_lower)
            if len(content_tokens) >= 8:
                break

    ct_matched = [
        t for t in content_tokens if _word_boundary_match(t, reasoning_lower)
    ]
    ct_total_idf = sum(idf.get(t, 1.0) for t in content_tokens)
    ct_matched_idf = sum(idf.get(t, 1.0) for t in ct_matched)
    ct_score = ct_matched_idf / ct_total_idf if ct_total_idf > 0 else 0.0

    # Return whichever source scored higher.
    if id_score >= ct_score:
        return id_score, id_matched, "id_tokens"
    return ct_score, ct_matched, "content"


def get_indirect_references(
    reasoning: str,
    sections: list[tuple[str, str, str, int]],
    already_referenced: set[str],
) -> tuple[set[str], list[IndirectMatchResult]]:
    """Find sections indirectly referenced via token overlap with reasoning.

    Uses relative ranking (no fixed threshold) because section naming
    conventions vary wildly between websites:

    1. Score all sections not in *already_referenced*.
    2. Drop sections with ``score == 0`` (zero token overlap).
    3. Sort descending by score.
    4. Take top ``min(floor(total * INDIRECT_MAX_RATIO), INDIRECT_MAX_ABSOLUTE)``.

    Args:
        reasoning: The agent's full analysis text.
        sections: ``(section_id, content, role, interactive_count)`` tuples.
        already_referenced: Section IDs already matched by direct mechanisms.

    Returns:
        ``(indirect_ids, match_details)`` tuple.
    """
    if not reasoning.strip() or not sections:
        return set(), []

    idf = _compute_token_idf(sections)
    reasoning_lower = reasoning.lower()

    candidates: list[IndirectMatchResult] = []
    for sid, content, _role, _ic in sections:
        if sid in already_referenced:
            continue
        score, matched, source = score_section_indirect(
            sid, content, reasoning_lower, idf,
        )
        if score > 0 and matched:
            candidates.append(
                IndirectMatchResult(sid, score, matched, source)
            )

    candidates.sort(key=lambda c: c.score, reverse=True)
    budget = min(
        max(1, int(len(sections) * INDIRECT_MAX_RATIO)),
        INDIRECT_MAX_ABSOLUTE,
    )
    candidates = candidates[:budget]

    return {c.section_id for c in candidates}, candidates


# ---------------------------------------------------------------------------
# Section preview builder
# ---------------------------------------------------------------------------

def _truncate_at_word(text: str, max_chars: int) -> str:
    """Truncate *text* at a word boundary, appending ``...``."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > max_chars // 2:
        cut = cut[:last_space]
    return cut + "..."


def _truncate_at_word_reverse(text: str, max_chars: int) -> str:
    """Take the last *max_chars* of *text*, trimmed to a word boundary."""
    if len(text) <= max_chars:
        return text
    tail = text[-max_chars:]
    first_space = tail.find(" ")
    if 0 < first_space < max_chars // 2:
        tail = tail[first_space + 1:]
    return "..." + tail


def _interactive_label(element: RenderedInteractiveElement) -> str:
    """Compact identity label for an interactive element.

    Priority: visible text → aria-label → data-testid → href (truncated).
    Format: ``tag "label"`` — e.g. ``a "Issues 45"``, ``button "Star"``.
    """
    attrs = element.attributes
    label = (
        element.element_text
        or attrs.get("aria-label", "")
        or attrs.get("data-testid", "")
        or attrs.get("placeholder", "")
    )
    if not label:
        href = attrs.get("href", "")
        if href:
            label = href[:40] + ("..." if len(href) > 40 else "")
    if not label:
        return ""
    # Truncate long labels.
    if len(label) > 30:
        label = label[:27] + "..."
    return f'{element.tag} "{label}"'


def _build_section_preview(
    content: str,
    interactive_elements: list[RenderedInteractiveElement] | None = None,
    max_chars: int = SKELETON_PREVIEW_CHARS,
) -> str:
    """Build a preview line showing start + end of content and top interactive elements.

    If content is short (<= 2 * max_chars), it is shown in full.
    Otherwise: first ~80 chars + ``···`` + last ~80 chars.
    """
    clean = content.strip().replace("\n", " ")
    # Collapse runs of whitespace.
    clean = _re.sub(r" {2,}", " ", clean)
    char_count = len(content.strip())

    if len(clean) <= max_chars * 2:
        text_part = clean
    else:
        start = _truncate_at_word(clean, max_chars)
        end = _truncate_at_word_reverse(clean, max_chars)
        text_part = f"{start} ··· {end}"

    # Top 3 interactive elements.
    interactive_part = ""
    if interactive_elements:
        labels = []
        for el in interactive_elements[:3]:
            lbl = _interactive_label(el)
            if lbl:
                labels.append(lbl)
        if labels:
            interactive_part = f" | {', '.join(labels)}"

    return f'[preview: "{text_part}" | {char_count} chars{interactive_part}]'


# ---------------------------------------------------------------------------
# Section metadata — embedded block for skeleton/stub conversion
# ---------------------------------------------------------------------------

_SECTION_META_START = "__SECTION_META__"
_SECTION_META_END = "__SECTION_META_END__"


@dataclass
class SectionMeta:
    """Parsed entry from a ``__SECTION_META__`` block."""

    section_id: str
    char_count: int
    interactive_count: int
    role: str
    preview_start: str
    preview_end: str
    interactive_labels: list[str] = field(default_factory=list)


def build_section_meta(
    sections: list[tuple[str, str, str, int]],
    interactive_elements_map: dict[str, list[RenderedInteractiveElement]] | None = None,
) -> str:
    """Build a hidden metadata block for later skeleton/stub conversion.

    Each line encodes one section as pipe-delimited fields::

        section_id|char_count|interactive_count|role|preview_start|preview_end|i_labels

    The block is placed after the filtered output, between the content
    the agent reads and the ``__PAGE_VIEW_END__`` marker.
    """
    lines: list[str] = [_SECTION_META_START]
    for sid, content, role, i_count in sections:
        clean = content.strip().replace("\n", " ").replace("|", "\u00a6")
        clean = _re.sub(r" {2,}", " ", clean)
        if len(clean) > 100:
            start = clean[:100].rsplit(" ", 1)[0] if " " in clean[:100] else clean[:100]
        else:
            start = clean
        if len(clean) > 100:
            tail = clean[-100:]
            end = tail.split(" ", 1)[-1] if " " in tail else tail
        else:
            end = ""
        # Interactive element labels.
        i_labels = ""
        if interactive_elements_map and sid in interactive_elements_map:
            labels = []
            for el in interactive_elements_map[sid][:3]:
                lbl = _interactive_label(el)
                if lbl:
                    labels.append(lbl)
            i_labels = ";".join(labels)
        lines.append(
            f"{sid}|{len(content.strip())}|{i_count}|{role}"
            f"|{start}|{end}|{i_labels}"
        )
    lines.append(_SECTION_META_END)
    return "\n".join(lines)


def parse_section_meta(text: str) -> list[SectionMeta] | None:
    """Parse a ``__SECTION_META__`` block from page view text.

    Returns ``None`` if the block is not found (graceful fallback for
    old-format page views without embedded metadata).
    """
    start_idx = text.find(_SECTION_META_START)
    end_idx = text.find(_SECTION_META_END)
    if start_idx < 0 or end_idx < 0:
        return None

    block = text[start_idx + len(_SECTION_META_START):end_idx].strip()
    if not block:
        return []

    results: list[SectionMeta] = []
    for line in block.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 7:
            continue
        sid = parts[0]
        try:
            char_count = int(parts[1])
            i_count = int(parts[2])
        except ValueError:
            continue
        role = parts[3]
        preview_start = parts[4].replace("\u00a6", "|")
        preview_end = parts[5].replace("\u00a6", "|")
        i_labels_raw = parts[6]
        i_labels = [l for l in i_labels_raw.split(";") if l] if i_labels_raw else []
        results.append(SectionMeta(
            section_id=sid,
            char_count=char_count,
            interactive_count=i_count,
            role=role,
            preview_start=preview_start,
            preview_end=preview_end,
            interactive_labels=i_labels,
        ))
    return results


# ---------------------------------------------------------------------------
# Filtered output builder
# ---------------------------------------------------------------------------

def _section_header(
    section_id: str,
    semantic_role: str = "",
    interactive_count: int = 0,
) -> str:
    """Build a section header matching ``_format_page_view`` style.

    ``--- [section-id] role (N interactive) ---``
    """
    role = semantic_role or "content"
    return (
        f"--- [{section_id}] {role} "
        f"({interactive_count} interactive) ---"
    )


def build_filtered_output(
    sections: list[tuple[str, str]] | list[tuple[str, str, str, int]],
    referenced: set[str],
    neighbor_radius: int = NEIGHBOR_RADIUS,
    page_header: str | None = None,
    indirect_refs: set[str] | None = None,
    indirect_neighbor_radius: int = INDIRECT_NEIGHBOR_RADIUS,
) -> str:
    """Build filtered show_page output with neighbor-aware omission.

    Classifies each section into one of four tiers:

    - **Kept** — section ID is in *referenced*.  Full header + content.
      Neighbors within *neighbor_radius*.
    - **Indirect** — section ID is in *indirect_refs* (and not in
      *referenced*).  Full header + content.  Neighbors within
      *indirect_neighbor_radius* (narrower).
    - **Neighbor** — within radius of a kept or indirect section.
      Emitted as header only (no content).
    - **Distant** — everything else.  Consecutive distant sections are
      accumulated and flushed as ``[N sections omitted]``.

    When *indirect_refs* is ``None`` (the default), the function behaves
    identically to the original three-tier version for backward
    compatibility.

    Blocks are joined with ``\\n\\n`` for visual separation.

    Args:
        sections: Tuples of ``(id, content)`` or
            ``(id, content, semantic_role, interactive_count)``.
        referenced: Set of section IDs the agent referenced.
        neighbor_radius: How many positions around a kept section count
            as neighbors.  Defaults to :data:`NEIGHBOR_RADIUS`.
        page_header: Optional page header line (e.g.
            ``=== Page State #5 | https://... ===``).  When provided it
            is prepended so the URL is never lost after filtering.
        indirect_refs: Set of section IDs matched via indirect token
            overlap.  These get full content with a narrower neighbor
            radius.  ``None`` disables indirect matching entirely.
        indirect_neighbor_radius: Neighbor radius for indirect refs.
            Defaults to :data:`INDIRECT_NEIGHBOR_RADIUS`.

    Returns:
        The filtered output string.
    """
    # Normalise to 4-tuples: (id, content, role, interactive_count).
    normalised: list[tuple[str, str, str, int]] = []
    for entry in sections:
        if len(entry) == 4:
            normalised.append(entry)  # type: ignore[arg-type]
        else:
            normalised.append((entry[0], entry[1], "", 0))

    _indirect = indirect_refs or set()

    kept_indices = {
        i for i, (sid, *_) in enumerate(normalised) if sid in referenced
    }
    indirect_indices = {
        i for i, (sid, *_) in enumerate(normalised)
        if sid in _indirect and i not in kept_indices
    }

    # Build neighbor sets: radius 3 for kept, radius 1 for indirect.
    full_content_indices = kept_indices | indirect_indices
    neighbor_indices: set[int] = set()
    for ki in kept_indices:
        for offset in range(-neighbor_radius, neighbor_radius + 1):
            idx = ki + offset
            if 0 <= idx < len(normalised) and idx not in full_content_indices:
                neighbor_indices.add(idx)
    for ii in indirect_indices:
        for offset in range(-indirect_neighbor_radius,
                            indirect_neighbor_radius + 1):
            idx = ii + offset
            if 0 <= idx < len(normalised) and idx not in full_content_indices:
                neighbor_indices.add(idx)

    blocks: list[str] = []
    distant_count = 0

    for i, (section_id, section_content, role, i_count) in enumerate(normalised):
        if i in full_content_indices:
            if distant_count > 0:
                blocks.append(f"[{distant_count} sections omitted]")
                distant_count = 0
            header = _section_header(section_id, role, i_count)
            blocks.append(f"{header}\n{section_content.strip()}")

        elif i in neighbor_indices:
            if distant_count > 0:
                blocks.append(f"[{distant_count} sections omitted]")
                distant_count = 0
            header = _section_header(section_id, role, i_count)
            blocks.append(f"{header}\n[omitted]")

        else:
            distant_count += 1

    if distant_count > 0:
        blocks.append(f"[{distant_count} sections omitted]")

    body = "\n\n".join(blocks)
    if page_header:
        return page_header + "\n\n" + body
    return body


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

    # Indirect reference detection (progressive compaction)
    sections_matched_indirect: list[str] = field(default_factory=list)
    variant_b_carried_forward: list[str] = field(default_factory=list)
