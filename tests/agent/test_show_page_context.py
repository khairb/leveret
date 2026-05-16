"""Tests for show_page context management (Tasks 2–4).

Covers page_similarity(), ShowPageState, section reference extraction,
and filtered output building.
"""

from scout.agent.show_page_context import (
    ATTRIBUTE_WEIGHTS,
    MATCH_THRESHOLD,
    MAX_SECTIONS_PER_ELEMENT,
    NEIGHBOR_RADIUS,
    SIMILARITY_THRESHOLD,
    ElementMatchResult,
    ShowPageState,
    _word_boundary_match,
    build_filtered_output,
    get_referenced_sections,
    get_sections_by_id,
    match_elements_to_reasoning,
    page_similarity,
    score_element,
)
from scout.page.converter import RenderedInteractiveElement


# ---------------------------------------------------------------------------
# page_similarity
# ---------------------------------------------------------------------------

class TestPageSimilarity:

    def test_identical_texts(self):
        text = "the quick brown fox jumps over the lazy dog"
        assert page_similarity(text, text) == 1.0

    def test_empty_vs_nonempty(self):
        assert page_similarity("", "hello world") == 0.0
        assert page_similarity("hello world", "") == 0.0

    def test_both_empty(self):
        assert page_similarity("", "") == 1.0

    def test_one_word_changed_in_1000(self):
        words = [f"word{i}" for i in range(1000)]
        original = " ".join(words)
        words[500] = "CHANGED"
        modified = " ".join(words)
        sim = page_similarity(original, modified)
        assert sim > 0.99, f"Expected ~0.998, got {sim}"

    def test_half_words_replaced(self):
        """50 shared words, 50 unique to each → 50/150 ≈ 0.333 Jaccard."""
        words_a = [f"alpha{i}" for i in range(100)]
        words_b = [f"alpha{i}" for i in range(50)] + [f"beta{i}" for i in range(50)]
        sim = page_similarity(" ".join(words_a), " ".join(words_b))
        assert 0.30 < sim < 0.36, f"Expected ~0.333, got {sim}"

    def test_completely_different(self):
        a = "alpha bravo charlie delta echo"
        b = "foxtrot golf hotel india juliet"
        assert page_similarity(a, b) == 0.0

    def test_position_invariance(self):
        words = "the quick brown fox jumps over the lazy dog".split()
        original = " ".join(words)
        import random
        rng = random.Random(42)
        shuffled_words = words[:]
        rng.shuffle(shuffled_words)
        shuffled = " ".join(shuffled_words)
        assert page_similarity(original, shuffled) == 1.0

    def test_duplicate_words_counted(self):
        """Multiset semantics: 'a a a' vs 'a a b' should not be 1.0."""
        sim = page_similarity("a a a", "a a b")
        # intersection: {a:2}, union: {a:3, b:1} → 2/4 = 0.5
        assert sim == 0.5


# ---------------------------------------------------------------------------
# ShowPageState
# ---------------------------------------------------------------------------

class TestShowPageState:

    def test_first_call_forces_full_analysis(self):
        state = ShowPageState()
        assert state.should_force_full_analysis("any text") is True

    def test_identical_page_after_mark(self):
        state = ShowPageState()
        text = "hello world foo bar"
        state.mark_analyzed(text)
        assert state.should_force_full_analysis(text) is False

    def test_completely_different_page_after_mark(self):
        state = ShowPageState()
        state.mark_analyzed("alpha bravo charlie")
        assert state.should_force_full_analysis("delta echo foxtrot") is True

    def test_minor_change_below_threshold(self):
        """A small change should stay above SIMILARITY_THRESHOLD → Variant B."""
        words = [f"word{i}" for i in range(100)]
        baseline = " ".join(words)
        state = ShowPageState()
        state.mark_analyzed(baseline)

        # Change 5 out of 100 words → similarity ~0.95
        modified_words = words[:]
        for i in range(5):
            modified_words[i] = f"changed{i}"
        modified = " ".join(modified_words)
        assert state.should_force_full_analysis(modified) is False

    def test_major_change_above_threshold(self):
        """A large change should drop below SIMILARITY_THRESHOLD → Variant A."""
        words = [f"word{i}" for i in range(100)]
        baseline = " ".join(words)
        state = ShowPageState()
        state.mark_analyzed(baseline)

        # Change 50 out of 100 words → similarity ~0.5
        modified_words = words[:]
        for i in range(50):
            modified_words[i] = f"changed{i}"
        modified = " ".join(modified_words)
        assert state.should_force_full_analysis(modified) is True

    def test_variant_b_does_not_update_baseline(self):
        """Variant B skips mark_analyzed, so baseline stays at last Variant A."""
        words = [f"word{i}" for i in range(100)]
        baseline = " ".join(words)
        state = ShowPageState()
        state.mark_analyzed(baseline)

        # Simulate several Variant B turns with incremental drift.
        # Each step changes 5 words from the ORIGINAL baseline.
        # Since we never call mark_analyzed, the baseline stays constant.
        for step in range(1, 3):
            drifted = words[:]
            for i in range(step * 5):
                drifted[i] = f"drift{i}"
            text = " ".join(drifted)
            # At step=2 (10 words changed), similarity is 90/100 = 0.90 > 0.8
            result = state.should_force_full_analysis(text)
            assert result is False, f"Step {step}: expected Variant B"

        # Now drift enough to cross the threshold (>20 words changed)
        big_drift = words[:]
        for i in range(25):
            big_drift[i] = f"drift{i}"
        assert state.should_force_full_analysis(" ".join(big_drift)) is True

    def test_threshold_constant(self):
        assert SIMILARITY_THRESHOLD == 0.8


# ---------------------------------------------------------------------------
# Helper to build mock RenderedInteractiveElement
# ---------------------------------------------------------------------------

def _make_element(
    *,
    tag: str = "button",
    iid: int = 1,
    attributes: dict[str, str] | None = None,
    classes: list[str] | None = None,
    element_text: str | None = None,
    full_tag_str: str = "",
) -> RenderedInteractiveElement:
    return RenderedInteractiveElement(
        iid=iid,
        tag=tag,
        attributes=attributes or {},
        classes=classes or [],
        element_text=element_text,
        full_tag_str=full_tag_str,
    )


# ---------------------------------------------------------------------------
# _word_boundary_match
# ---------------------------------------------------------------------------

class TestWordBoundaryMatch:

    def test_match_surrounded_by_spaces(self):
        assert _word_boundary_match("search", "I need to search for") is True

    def test_match_at_start(self):
        assert _word_boundary_match("search", "search the page") is True

    def test_match_at_end(self):
        assert _word_boundary_match("search", "the search") is True

    def test_no_match_inside_word(self):
        assert _word_boundary_match("search", "researching the page") is False

    def test_no_match_dir_in_directory(self):
        assert _word_boundary_match("dir", "the directory listing") is False

    def test_match_with_non_word_boundary(self):
        assert _word_boundary_match("dir", "class=dir dir-ltr") is True

    def test_no_match_when_absent(self):
        assert _word_boundary_match("foobar", "no match here") is False


# ---------------------------------------------------------------------------
# score_element
# ---------------------------------------------------------------------------

class TestScoreElement:

    def test_full_tag_match_short_circuits(self):
        """Agent copies the whole tag → score 2.0, immediate return."""
        tag_str = '<button data-testid="category-bar-filter-button" type="button">'
        el = _make_element(
            attributes={"data-testid": "category-bar-filter-button", "type": "button"},
            full_tag_str=tag_str,
        )
        reasoning = f"I see {tag_str} in the header"
        score, attrs = score_element(el, reasoning)
        assert score == 2.0
        assert len(attrs) == 1
        assert attrs[0]["match"] == "exact-tag"

    def test_full_tag_too_short_no_shortcircuit(self):
        """Tags <= 20 chars skip the full-tag fast path."""
        tag_str = '<input type="text">'
        el = _make_element(
            tag="input",
            attributes={"type": "text"},
            full_tag_str=tag_str,
        )
        reasoning = f'I see {tag_str}'
        score, attrs = score_element(el, reasoning)
        # Should fall through to per-attribute scoring, not 2.0
        assert score < 2.0

    def test_single_strong_attribute_above_threshold(self):
        """data-testid full match → 1.0 (above MATCH_THRESHOLD 0.5)."""
        el = _make_element(
            attributes={"data-testid": "category-bar-filter-button"},
        )
        reasoning = 'the button with data-testid="category-bar-filter-button"'
        score, attrs = score_element(el, reasoning)
        assert score == 1.0
        assert attrs[0]["match"] == "full"

    def test_aria_label_full_match(self):
        el = _make_element(attributes={"aria-label": "Weiter"})
        reasoning = 'the button with aria-label="Weiter"'
        score, _ = score_element(el, reasoning)
        assert score == ATTRIBUTE_WEIGHTS["aria-label"]  # 0.8

    def test_value_only_match_with_reduced_weight(self):
        """Value >= 6 chars matched alone → weight * 0.6."""
        el = _make_element(
            attributes={"data-testid": "category-bar-filter-button"},
        )
        reasoning = "the category-bar-filter-button"
        score, attrs = score_element(el, reasoning)
        expected = ATTRIBUTE_WEIGHTS["data-testid"] * 0.6  # 1.0 * 0.6
        assert score == expected
        assert attrs[0]["match"] == "value-only"

    def test_short_value_not_matched_alone(self):
        """Value < 6 chars should not trigger value-only match."""
        el = _make_element(attributes={"type": "text"})
        reasoning = "I see a text input"
        score, _ = score_element(el, reasoning)
        # "text" is 4 chars < 6, so no value-only match
        assert score == 0.0

    def test_single_weak_attribute_below_threshold(self):
        """Visible text only → 0.3, below MATCH_THRESHOLD 0.5."""
        el = _make_element(element_text="Filter")
        reasoning = "I need to use the Filter"
        score, _ = score_element(el, reasoning)
        assert score == 0.3
        assert score < MATCH_THRESHOLD

    def test_multiple_weak_attributes_combine(self):
        """aria-label full + type full → 0.8 + 0.1 = 0.9."""
        el = _make_element(
            attributes={"aria-label": "Weiter", "type": "button"},
        )
        reasoning = 'button aria-label="Weiter" type="button"'
        score, _ = score_element(el, reasoning)
        assert score == 0.9

    def test_word_boundary_prevents_substring_match(self):
        """'search' should not match inside 'researching'."""
        el = _make_element(
            attributes={"data-testid": "search"},
            element_text="search",
        )
        reasoning = "researching the page"
        score, _ = score_element(el, reasoning)
        assert score == 0.0

    def test_class_fraction_eligible_denominator(self):
        """Short classes (< 4 chars) excluded from both matching and denominator."""
        el = _make_element(
            classes=["a", "b", "cc", "btn-primary", "btn-lg"],
        )
        # Only btn-primary found; eligible = [btn-primary, btn-lg] (2 classes)
        reasoning = "the btn-primary button"
        score, attrs = score_element(el, reasoning)
        # 1/2 eligible → 0.4 * 0.5 = 0.2
        class_match = [a for a in attrs if a["attr"] == "class"]
        assert len(class_match) == 1
        assert abs(class_match[0]["weight"] - 0.2) < 0.001

    def test_class_short_names_excluded(self):
        """All classes < 4 chars → no class matching at all."""
        el = _make_element(classes=["a", "b", "cc"])
        reasoning = "a b cc"
        score, attrs = score_element(el, reasoning)
        class_match = [a for a in attrs if a["attr"] == "class"]
        assert len(class_match) == 0

    def test_visible_text_too_short(self):
        """Text <= 2 chars should not be matched."""
        el = _make_element(element_text="OK")
        reasoning = "press OK"
        score, _ = score_element(el, reasoning)
        assert score == 0.0

    def test_empty_attributes_skipped(self):
        """Attributes with empty string values are skipped."""
        el = _make_element(attributes={"aria-label": "", "type": "button"})
        reasoning = 'type="button"'
        score, attrs = score_element(el, reasoning)
        # Only type should match (0.1), aria-label skipped
        assert score == 0.1


# ---------------------------------------------------------------------------
# match_elements_to_reasoning
# ---------------------------------------------------------------------------

class TestMatchElementsToReasoning:

    def test_signature_grouping(self):
        """Same structural signature across sections groups correctly."""
        el1 = _make_element(
            iid=1,
            attributes={"data-testid": "save-btn", "type": "button"},
            element_text="Save",
        )
        el2 = _make_element(
            iid=2,
            attributes={"data-testid": "save-btn", "type": "button"},
            element_text="Save",
        )
        elements = [("section-1", el1), ("section-2", el2)]
        reasoning = 'the button data-testid="save-btn"'
        results = match_elements_to_reasoning(reasoning, elements)
        assert len(results) == 1  # one group
        assert set(results[0].sections_with_same_element) == {"section-1", "section-2"}
        assert results[0].was_capped is False

    def test_capping_at_max_sections(self):
        """18 identical buttons capped to MAX_SECTIONS_PER_ELEMENT."""
        elements = []
        for i in range(18):
            el = _make_element(
                iid=i,
                attributes={"data-testid": "listing-card-save-button", "type": "button"},
            )
            elements.append((f"item-{i}-div", el))

        reasoning = 'data-testid="listing-card-save-button"'
        results = match_elements_to_reasoning(reasoning, elements)
        assert len(results) == 1
        assert len(results[0].sections_with_same_element) == 18
        assert len(results[0].sections_kept) == MAX_SECTIONS_PER_ELEMENT
        assert results[0].was_capped is True

    def test_below_threshold_filtered_out(self):
        """Elements scoring below MATCH_THRESHOLD are not included."""
        el = _make_element(element_text="Filter")
        elements = [("section-1", el)]
        reasoning = "I see the Filter"
        results = match_elements_to_reasoning(reasoning, elements)
        # score is 0.3 (visible text) < 0.5 threshold
        assert len(results) == 0

    def test_different_signatures_separate_groups(self):
        """Elements with different structural attrs form separate groups."""
        el_save = _make_element(
            iid=1,
            attributes={"data-testid": "save-btn", "type": "button"},
        )
        el_next = _make_element(
            iid=2,
            attributes={"aria-label": "Weiter", "type": "button"},
        )
        elements = [("section-1", el_save), ("section-2", el_next)]
        reasoning = 'data-testid="save-btn" and aria-label="Weiter"'
        results = match_elements_to_reasoning(reasoning, elements)
        # Two different signatures → two groups
        assert len(results) == 2


# ---------------------------------------------------------------------------
# get_sections_by_id
# ---------------------------------------------------------------------------

class TestGetSectionsById:

    def test_finds_mentioned_ids(self):
        reasoning = "zoom into [item-1-div-wohnung-in] and [header-unterk-nfte]"
        ids = ["item-1-div-wohnung-in", "header-unterk-nfte", "footer-links"]
        result = get_sections_by_id(reasoning, ids)
        assert result == {"item-1-div-wohnung-in", "header-unterk-nfte"}

    def test_no_ids_found(self):
        reasoning = "I see a page with listings"
        ids = ["item-1-div-wohnung-in", "header-unterk-nfte"]
        result = get_sections_by_id(reasoning, ids)
        assert result == set()

    def test_empty_reasoning(self):
        result = get_sections_by_id("", ["section-1"])
        assert result == set()

    def test_empty_section_ids(self):
        result = get_sections_by_id("some reasoning", [])
        assert result == set()


# ---------------------------------------------------------------------------
# get_referenced_sections
# ---------------------------------------------------------------------------

class TestGetReferencedSections:

    def test_mechanism_1_only(self):
        """Section IDs mentioned directly — no element matching needed."""
        sections = [
            ("header-nav", "header content", []),
            ("item-1-div", "listing content", []),
            ("footer-links", "footer content", []),
        ]
        reasoning = "I need item-1-div for extraction"
        referenced, matches = get_referenced_sections(reasoning, sections)
        assert referenced == {"item-1-div"}
        assert matches == []

    def test_mechanism_2_catches_unmentioned_section(self):
        """Element in unmentioned section → section gets added."""
        el = _make_element(
            attributes={"aria-label": "Weiter"},
        )
        sections = [
            ("header-nav", "header content", []),
            ("pagination-div", "page 1 of 3", [el]),
        ]
        reasoning = 'the button with aria-label="Weiter"'
        referenced, matches = get_referenced_sections(reasoning, sections)
        assert "pagination-div" in referenced
        assert len(matches) == 1

    def test_mechanism_2_skips_already_matched_sections(self):
        """Elements in already-matched sections are not scored."""
        el = _make_element(
            attributes={"aria-label": "Weiter"},
        )
        sections = [
            ("pagination-div", "page 1 of 3", [el]),
        ]
        # Section is matched by ID AND has an element
        reasoning = 'pagination-div has aria-label="Weiter"'
        referenced, matches = get_referenced_sections(reasoning, sections)
        assert "pagination-div" in referenced
        # Element matching should produce no results (section already matched)
        assert matches == []

    def test_combined_mechanisms(self):
        """Both mechanisms contribute sections."""
        el = _make_element(
            attributes={"data-testid": "next-page-button"},
        )
        sections = [
            ("header-nav", "header content", []),
            ("item-1-div", "listing content", []),
            ("pagination-div", "page 1 of 3", [el]),
        ]
        reasoning = 'I need item-1-div. The data-testid="next-page-button" for paging.'
        referenced, matches = get_referenced_sections(reasoning, sections)
        assert "item-1-div" in referenced      # via mechanism 1
        assert "pagination-div" in referenced   # via mechanism 2

    def test_constants_values(self):
        assert MATCH_THRESHOLD == 0.5
        assert MAX_SECTIONS_PER_ELEMENT == 3
        assert ATTRIBUTE_WEIGHTS["data-testid"] == 1.0
        assert ATTRIBUTE_WEIGHTS["aria-label"] == 0.8
        assert ATTRIBUTE_WEIGHTS["type"] == 0.1


# ---------------------------------------------------------------------------
# build_filtered_output
# ---------------------------------------------------------------------------

def _make_sections(n: int) -> list[tuple[str, str]]:
    """Helper: create N sections with predictable IDs and content."""
    return [(f"section-{i}", f"Content of section {i}") for i in range(n)]


class TestBuildFilteredOutput:

    def test_neighbor_radius_constant(self):
        assert NEIGHBOR_RADIUS == 3

    def test_all_sections_kept(self):
        """All sections referenced → full output, no omission lines."""
        sections = _make_sections(5)
        referenced = {sid for sid, _ in sections}
        result = build_filtered_output(sections, referenced)

        for sid, content in sections:
            assert f"[{sid}]" in result
            assert content in result
        assert "omitted" not in result

    def test_no_sections_kept(self):
        """No sections referenced → single omission line."""
        sections = _make_sections(7)
        result = build_filtered_output(sections, set())
        assert result == "[7 sections omitted]"

    def test_one_kept_in_middle(self):
        """One section kept in the middle → neighbors on both sides, distant at edges."""
        sections = _make_sections(12)  # indices 0..11
        referenced = {"section-6"}
        result = build_filtered_output(sections, referenced)

        # Distant before neighbors: sections 0,1,2 → [3 sections omitted]
        assert "[3 sections omitted]" in result

        # Neighbors before kept: sections 3,4,5 (header + [omitted])
        for i in (3, 4, 5):
            assert f"[section-{i}]" in result
            assert "[omitted]" in result

        # Kept section: section-6
        assert "[section-6]" in result
        assert "Content of section 6" in result

        # Neighbors after kept: sections 7,8,9
        for i in (7, 8, 9):
            assert f"[section-{i}]" in result

        # Distant after neighbors: sections 10,11 → [2 sections omitted]
        assert "[2 sections omitted]" in result

    def test_adjacent_kept_sections(self):
        """Adjacent kept sections → no double-counting of neighbors."""
        sections = _make_sections(10)  # indices 0..9
        referenced = {"section-4", "section-5"}
        result = build_filtered_output(sections, referenced)

        # Both kept sections present with full content
        assert "Content of section 4" in result
        assert "Content of section 5" in result

        # Neighbors: sections 1,2,3 (before) and 6,7,8 (after)
        for i in (1, 2, 3, 6, 7, 8):
            assert f"[section-{i}]" in result

        # Distant: section 0 before, section 9 after
        # Section 0 is distant (outside radius of section-4)
        assert "[1 sections omitted]" in result

    def test_kept_at_index_zero(self):
        """Kept at index 0 → no negative index neighbors."""
        sections = _make_sections(8)
        referenced = {"section-0"}
        result = build_filtered_output(sections, referenced)

        # Kept section
        assert "[section-0]" in result
        assert "Content of section 0" in result

        # Neighbors: 1, 2, 3
        for i in (1, 2, 3):
            assert f"[section-{i}]" in result

        # Distant: sections 4..7
        assert "[4 sections omitted]" in result

        # No negative indices should appear
        assert "section--" not in result

    def test_kept_at_last_index(self):
        """Kept at last index → no overflow neighbors."""
        sections = _make_sections(8)  # indices 0..7
        referenced = {"section-7"}
        result = build_filtered_output(sections, referenced)

        # Distant: sections 0..3
        assert "[4 sections omitted]" in result

        # Neighbors: 4, 5, 6
        for i in (4, 5, 6):
            assert f"[section-{i}]" in result

        # Kept section
        assert "[section-7]" in result
        assert "Content of section 7" in result

    def test_block_separator_is_double_newline(self):
        """Blocks are joined with \\n\\n."""
        sections = _make_sections(3)
        referenced = {"section-1"}
        result = build_filtered_output(sections, referenced)
        blocks = result.split("\n\n")
        # Should have: [section-0 — omitted], kept section-1, [section-2 — omitted]
        assert len(blocks) == 3

    def test_kept_section_format(self):
        """Kept section uses --- [id] role (N interactive) --- header."""
        sections = [("my-section", "Hello world")]
        referenced = {"my-section"}
        result = build_filtered_output(sections, referenced)
        assert result == "--- [my-section] content (0 interactive) ---\nHello world"

    def test_kept_section_format_with_metadata(self):
        """Kept section with role/count uses full header."""
        sections = [("nav-main", "Home About Contact", "navigation", 3)]
        referenced = {"nav-main"}
        result = build_filtered_output(sections, referenced)
        assert result == "--- [nav-main] navigation (3 interactive) ---\nHome About Contact"

    def test_empty_sections_list(self):
        """Empty sections list → empty output."""
        result = build_filtered_output([], set())
        assert result == ""

    def test_custom_neighbor_radius(self):
        """Custom neighbor_radius=1 narrows the neighbor window."""
        sections = _make_sections(8)
        referenced = {"section-4"}
        result = build_filtered_output(sections, referenced, neighbor_radius=1)

        # Distant before: sections 0,1,2
        assert "[3 sections omitted]" in result

        # Neighbors: only 3 and 5
        assert "[section-3]" in result
        assert "[section-5]" in result

        # Sections 1,2 should NOT be neighbors (they're in the distant block)
        assert "[section-1]" not in result
        assert "[section-2]" not in result

        # Distant after: sections 6,7
        assert "[2 sections omitted]" in result
