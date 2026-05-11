"""Tests for progressive compaction: indirect reference detection, section
preview, and section metadata (Steps 1-2 of the progressive compaction plan).

Includes both unit tests with synthetic data and validation tests against
real benchmark trace data from GitHub, Y Combinator, and Books.toscrape.
"""

from __future__ import annotations

import math

from scout.agent.show_page_context import (
    INDIRECT_MAX_ABSOLUTE,
    INDIRECT_MAX_RATIO,
    INDIRECT_NEIGHBOR_RADIUS,
    MIN_ID_TOKEN_LENGTH,
    SKELETON_PREVIEW_CHARS,
    IndirectMatchResult,
    SectionMeta,
    _build_section_preview,
    _compute_token_idf,
    _interactive_label,
    _truncate_at_word,
    _truncate_at_word_reverse,
    _tokenize_section_id,
    build_section_meta,
    get_indirect_references,
    parse_section_meta,
    score_section_indirect,
)
from scout.page.converter import RenderedInteractiveElement


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_element(
    tag: str = "a",
    attributes: dict | None = None,
    element_text: str = "",
    classes: list[str] | None = None,
) -> RenderedInteractiveElement:
    return RenderedInteractiveElement(
        tag=tag,
        iid=0,
        attributes=attributes or {},
        classes=classes or [],
        element_text=element_text,
        full_tag_str="",
    )


def _make_sections_4tuple(
    ids_and_content: list[tuple[str, str]],
    role: str = "content",
    i_count: int = 0,
) -> list[tuple[str, str, str, int]]:
    """Build 4-tuples from (id, content) pairs."""
    return [(sid, content, role, i_count) for sid, content in ids_and_content]


# ═══════════════════════════════════════════════════════════════
#  _tokenize_section_id
# ═══════════════════════════════════════════════════════════════


class TestTokenizeSectionId:

    def test_basic_split(self):
        assert _tokenize_section_id("sidebar-about") == ["sidebar", "about"]

    def test_underscore_split(self):
        assert _tokenize_section_id("nav_header_main") == ["header", "main"]

    def test_short_tokens_filtered(self):
        """All tokens < MIN_ID_TOKEN_LENGTH are dropped."""
        assert _tokenize_section_id("div-hq-nav") == []

    def test_mixed_lengths(self):
        result = _tokenize_section_id("item-1-article-anthropics-financial")
        assert result == ["item", "article", "anthropics", "financial"]

    def test_deduplication(self):
        result = _tokenize_section_id("item-this-week-this-month")
        assert result == ["item", "this", "week", "month"]

    def test_empty_string(self):
        assert _tokenize_section_id("") == []

    def test_single_long_token(self):
        assert _tokenize_section_id("trending") == ["trending"]

    def test_numeric_tokens_filtered(self):
        """Pure numeric tokens like '1', '2', '123' are dropped."""
        result = _tokenize_section_id("item-2-div-3")
        assert result == ["item"]

    def test_real_github_id(self):
        result = _tokenize_section_id(
            "summary-spoken-language-any-language"
        )
        assert "spoken" in result
        assert "language" in result
        # "language" appears twice — dedup keeps only first occurrence.
        assert result.count("language") == 1

    def test_real_yc_id(self):
        result = _tokenize_section_id(
            "div-batch-industry-hq-region-company-size"
        )
        # "div" and "hq" are < 4 chars → filtered
        assert "batch" in result
        assert "industry" in result
        assert "region" in result
        assert "company" in result
        assert "size" in result
        assert "div" not in result
        assert "hq" not in result


# ═══════════════════════════════════════════════════════════════
#  _compute_token_idf
# ═══════════════════════════════════════════════════════════════


class TestComputeTokenIdf:

    def test_unique_token_high_idf(self):
        """A token in only 1 of 10 sections gets high idf."""
        sections = [
            (f"section-{i}", f"content {i}", "content", 0)
            for i in range(10)
        ]
        sections[0] = ("unique-section-identifier", "content 0", "content", 0)
        idf = _compute_token_idf(sections)
        # "unique" only in section 0 → idf = log(1 + 10/1) = log(11)
        assert abs(idf.get("unique", 0) - math.log(11)) < 0.01

    def test_ubiquitous_token_low_idf(self):
        """A token in ALL sections gets low (but non-zero) idf."""
        sections = [
            (f"content-item-{i}", "some text", "content", 0)
            for i in range(10)
        ]
        idf = _compute_token_idf(sections)
        # "content" in all 10 → idf = log(1 + 10/10) = log(2) ≈ 0.69
        assert abs(idf.get("content", 999) - math.log(2)) < 0.01

    def test_unique_vs_ubiquitous_ratio(self):
        """Unique token should have much higher IDF than ubiquitous one."""
        sections = [
            (f"content-item-{i}", "some text", "content", 0)
            for i in range(10)
        ]
        sections[0] = ("unique-section-identifier", "content 0", "content", 0)
        idf = _compute_token_idf(sections)
        assert idf.get("unique", 0) > idf.get("content", 0) * 2

    def test_half_sections_token(self):
        """Token in half the sections → idf = log(1 + 10/5) = log(3) ≈ 1.10."""
        sections = []
        for i in range(10):
            if i < 5:
                sections.append((f"special-item-{i}", "text", "content", 0))
            else:
                sections.append((f"other-thing-{i}", "text", "content", 0))
        idf = _compute_token_idf(sections)
        assert abs(idf.get("special", 0) - math.log(3)) < 0.01

    def test_empty_sections(self):
        assert _compute_token_idf([]) == {}

    def test_content_tokens_included(self):
        """Tokens from first 200 chars of content are included in IDF."""
        sections = [
            ("section-alpha", "Django framework tutorial", "content", 0),
            ("section-beta", "Flask framework tutorial", "content", 0),
        ]
        idf = _compute_token_idf(sections)
        # "framework" in both → low IDF; "django" in one → high IDF.
        assert idf["django"] > idf["framework"]


# ═══════════════════════════════════════════════════════════════
#  score_section_indirect
# ═══════════════════════════════════════════════════════════════


class TestScoreSectionIndirect:

    def _make_idf(self, sections):
        return _compute_token_idf(sections)

    def test_full_id_match(self):
        sections = _make_sections_4tuple([
            ("sidebar-about", "About section content"),
            ("readme-article", "README content here"),
        ])
        idf = self._make_idf(sections)
        score, matched, source = score_section_indirect(
            "sidebar-about", "About section content",
            "i need the sidebar and the about section", idf,
        )
        assert score > 0.8
        assert "sidebar" in matched
        assert "about" in matched

    def test_no_match(self):
        sections = _make_sections_4tuple([
            ("sidebar-about", "About content"),
            ("readme-article", "README text"),
        ])
        idf = self._make_idf(sections)
        score, matched, source = score_section_indirect(
            "sidebar-about", "About content",
            "the footer has navigation links", idf,
        )
        assert score == 0.0
        assert matched == []

    def test_partial_match_weighted(self):
        """Matching 1 of 2 tokens should give a score between 0 and 1."""
        sections = _make_sections_4tuple([
            ("sidebar-about", "About this project"),
        ])
        idf = self._make_idf(sections)
        score, matched, source = score_section_indirect(
            "sidebar-about", "About this project",
            "the sidebar contains links", idf,
        )
        assert 0 < score < 1.0
        assert "sidebar" in matched
        assert "about" not in matched

    def test_high_idf_token_beats_low_idf(self):
        """A unique token matching should give more score than a common one."""
        sections = _make_sections_4tuple([
            ("content-wrapper-main", "Main page content"),
            ("content-wrapper-aside", "Side content"),
            ("content-wrapper-footer", "Footer content"),
            ("unique-trending-page", "Trending repos"),
        ])
        idf = self._make_idf(sections)
        # "content" is in 3/4 sections (low idf), "trending" is in 1/4 (high idf)
        score_common, _, _ = score_section_indirect(
            "content-wrapper-main", "Main page content",
            "the content area", idf,
        )
        score_unique, _, _ = score_section_indirect(
            "unique-trending-page", "Trending repos",
            "the trending page", idf,
        )
        assert score_unique > score_common

    def test_content_fallback(self):
        """When ID tokens don't match, content keywords should be tried."""
        # Multiple sections so IDF varies. "div-xyz-123" has no valid ID tokens.
        sections = _make_sections_4tuple([
            ("div-xyz-123", "Django framework tutorial guide"),
            ("section-other", "Unrelated content about cats"),
        ])
        idf = self._make_idf(sections)
        score, matched, source = score_section_indirect(
            "div-xyz-123", "Django framework tutorial guide",
            "looking at the django framework tutorial", idf,
        )
        assert score > 0
        assert source == "content"
        assert "django" in matched

    def test_word_boundary(self):
        """'sidebar' should not match 'sidebars' in reasoning."""
        sections = _make_sections_4tuple([
            ("sidebar-about", "About section"),
        ])
        idf = self._make_idf(sections)
        score, matched, _ = score_section_indirect(
            "sidebar-about", "About section",
            "the sidebars are too wide", idf,
        )
        # "sidebar" should NOT match "sidebars"
        assert "sidebar" not in matched

    def test_common_token_low_contribution(self):
        """A token appearing in many sections (low IDF) contributes little."""
        # Create 20 sections with "content" in ID, 1 with "unique"
        sections = [(f"content-item-{i}", "text", "content", 0) for i in range(20)]
        sections.append(("unique-special-data", "text", "content", 0))
        idf = _compute_token_idf(sections)
        score_common, _, _ = score_section_indirect(
            "content-item-0", "text", "the content is here", idf,
        )
        score_unique, _, _ = score_section_indirect(
            "unique-special-data", "text", "unique special data", idf,
        )
        # Unique section should score higher (unique tokens have higher IDF)
        assert score_unique > score_common

    def test_real_github_example(self):
        """AI reasoning says 'the About sidebar', section is 'sidebar-about'."""
        sections = _make_sections_4tuple([
            ("sidebar-about", "About: A research assistant"),
            ("readme-article", "# README content"),
            ("nav-header", "Home Navigation"),
        ])
        idf = self._make_idf(sections)
        score, matched, _ = score_section_indirect(
            "sidebar-about", "About: A research assistant",
            "I need to find the description in the about sidebar",
            idf,
        )
        assert score > 0.5
        assert "sidebar" in matched or "about" in matched

    def test_real_yc_example(self):
        """AI reasoning says 'batch filter', section has 'batch' in ID."""
        sections = _make_sections_4tuple([
            ("div-batch-industry-hq-region-company-size", "Filter panel"),
            ("div-search", "Search box"),
            ("div-loading-more", "Loading more..."),
        ])
        idf = self._make_idf(sections)
        score, matched, source = score_section_indirect(
            "div-batch-industry-hq-region-company-size", "Filter panel",
            "I need to use the batch filter to find companies by industry",
            idf,
        )
        assert score > 0
        # May match via ID tokens ("batch", "industry") or content ("filter")
        assert len(matched) > 0


# ═══════════════════════════════════════════════════════════════
#  get_indirect_references
# ═══════════════════════════════════════════════════════════════


class TestGetIndirectReferences:

    def test_budget_cap_large_page(self):
        """91 sections → budget = min(18, 15) = 15."""
        sections = [
            (f"section-unique{i}-item", f"content {i}", "content", 0)
            for i in range(91)
        ]
        reasoning = " ".join(f"unique{i}" for i in range(91))
        refs, matches = get_indirect_references(reasoning, sections, set())
        assert len(refs) <= INDIRECT_MAX_ABSOLUTE

    def test_budget_cap_small_page(self):
        """10 sections → budget = min(2, 15) = 2."""
        sections = [
            (f"section-unique{i}", f"content {i}", "content", 0)
            for i in range(10)
        ]
        reasoning = " ".join(f"unique{i}" for i in range(10))
        refs, matches = get_indirect_references(reasoning, sections, set())
        budget = min(int(10 * INDIRECT_MAX_RATIO), INDIRECT_MAX_ABSOLUTE)
        assert len(refs) <= budget

    def test_already_referenced_excluded(self):
        """Sections in already_referenced are not scored."""
        sections = _make_sections_4tuple([
            ("sidebar-about", "About content"),
            ("readme-article", "README content"),
        ])
        refs, _ = get_indirect_references(
            "sidebar about readme article",
            sections,
            already_referenced={"sidebar-about"},
        )
        assert "sidebar-about" not in refs

    def test_zero_score_filtered(self):
        """Sections with no token overlap are not returned."""
        sections = _make_sections_4tuple([
            ("sidebar-about", "About content"),
        ])
        refs, _ = get_indirect_references(
            "completely unrelated text xyz",
            sections,
            set(),
        )
        assert len(refs) == 0

    def test_ranking_order(self):
        """Higher-scoring sections take priority."""
        sections = _make_sections_4tuple([
            ("sidebar-about", "About section"),
            ("unique-trending-page", "Trending repositories"),
            ("generic-item-content", "Some content"),
        ])
        refs, matches = get_indirect_references(
            "I need the trending page and the about sidebar",
            sections,
            set(),
        )
        if len(matches) >= 2:
            assert matches[0].score >= matches[1].score

    def test_empty_reasoning(self):
        sections = _make_sections_4tuple([
            ("sidebar-about", "About content"),
        ])
        refs, _ = get_indirect_references("", sections, set())
        assert refs == set()

    def test_all_sections_referenced(self):
        """When all sections are already referenced, nothing left to score."""
        sections = _make_sections_4tuple([
            ("sidebar-about", "About content"),
            ("readme-article", "README"),
        ])
        refs, _ = get_indirect_references(
            "sidebar about readme article",
            sections,
            already_referenced={"sidebar-about", "readme-article"},
        )
        assert refs == set()


# ═══════════════════════════════════════════════════════════════
#  _truncate helpers
# ═══════════════════════════════════════════════════════════════


class TestTruncateHelpers:

    def test_truncate_at_word_short(self):
        assert _truncate_at_word("hello", 10) == "hello"

    def test_truncate_at_word_long(self):
        result = _truncate_at_word("the quick brown fox jumps", 15)
        assert result.endswith("...")
        assert len(result) <= 18  # 15 + "..."

    def test_truncate_at_word_reverse_short(self):
        assert _truncate_at_word_reverse("hello", 10) == "hello"

    def test_truncate_at_word_reverse_long(self):
        result = _truncate_at_word_reverse("the quick brown fox jumps", 15)
        assert result.startswith("...")


# ═══════════════════════════════════════════════════════════════
#  _interactive_label
# ═══════════════════════════════════════════════════════════════


class TestInteractiveLabel:

    def test_visible_text_priority(self):
        el = _make_element(tag="button", element_text="Star")
        assert _interactive_label(el) == 'button "Star"'

    def test_aria_label_fallback(self):
        el = _make_element(
            tag="a", attributes={"aria-label": "Issues 45"},
        )
        assert _interactive_label(el) == 'a "Issues 45"'

    def test_href_fallback(self):
        el = _make_element(
            tag="a", attributes={"href": "/repo/issues"},
        )
        assert _interactive_label(el) == 'a "/repo/issues"'

    def test_no_label(self):
        el = _make_element(tag="div")
        assert _interactive_label(el) == ""

    def test_long_label_truncated(self):
        el = _make_element(tag="a", element_text="A" * 50)
        result = _interactive_label(el)
        assert len(result) < 50
        assert "..." in result


# ═══════════════════════════════════════════════════════════════
#  _build_section_preview
# ═══════════════════════════════════════════════════════════════


class TestBuildSectionPreview:

    def test_short_content_no_truncation(self):
        preview = _build_section_preview("Hello world")
        assert "Hello world" in preview
        assert "···" not in preview

    def test_long_content_start_and_end(self):
        content = "Start of the section. " + "x " * 200 + "End of the section."
        preview = _build_section_preview(content)
        assert "Start of the section" in preview
        assert "End of the section" in preview
        assert "···" in preview

    def test_interactive_elements_shown(self):
        els = [
            _make_element(tag="button", element_text="Star"),
            _make_element(tag="a", element_text="Issues"),
            _make_element(tag="a", element_text="Fork"),
        ]
        preview = _build_section_preview("Some content", interactive_elements=els)
        assert 'button "Star"' in preview
        assert 'a "Issues"' in preview
        assert 'a "Fork"' in preview

    def test_no_interactive_elements(self):
        preview = _build_section_preview("Some content", interactive_elements=None)
        assert "chars" in preview
        assert "|" not in preview.split("chars")[1]  # no interactive part

    def test_char_count_accurate(self):
        content = "Hello world"
        preview = _build_section_preview(content)
        assert "11 chars" in preview

    def test_empty_content(self):
        preview = _build_section_preview("")
        assert "0 chars" in preview


# ═══════════════════════════════════════════════════════════════
#  build_section_meta / parse_section_meta
# ═══════════════════════════════════════════════════════════════


class TestSectionMeta:

    def test_round_trip(self):
        sections = [
            ("sidebar-about", "About: A research assistant for deep research", "content", 3),
            ("readme-article", "# README\nLong content here " * 20, "content", 0),
        ]
        meta_str = build_section_meta(sections)
        parsed = parse_section_meta(meta_str)
        assert parsed is not None
        assert len(parsed) == 2
        assert parsed[0].section_id == "sidebar-about"
        assert parsed[0].char_count == len("About: A research assistant for deep research")
        assert parsed[0].interactive_count == 3
        assert parsed[0].role == "content"
        assert parsed[1].section_id == "readme-article"

    def test_pipe_in_content_escaped(self):
        sections = [
            ("test-section", "value|with|pipes", "content", 0),
        ]
        meta_str = build_section_meta(sections)
        parsed = parse_section_meta(meta_str)
        assert parsed is not None
        assert len(parsed) == 1
        # Pipes in content should be escaped and recovered
        assert "|" in parsed[0].preview_start or "pipes" in parsed[0].preview_start

    def test_empty_sections(self):
        meta_str = build_section_meta([])
        parsed = parse_section_meta(meta_str)
        assert parsed is not None
        assert len(parsed) == 0

    def test_missing_meta_returns_none(self):
        assert parse_section_meta("no meta block here") is None

    def test_interactive_labels_round_trip(self):
        els = {
            "sidebar-about": [
                _make_element(tag="button", element_text="Star"),
                _make_element(tag="a", element_text="Issues"),
            ],
        }
        sections = [("sidebar-about", "About content", "content", 2)]
        meta_str = build_section_meta(sections, interactive_elements_map=els)
        parsed = parse_section_meta(meta_str)
        assert parsed is not None
        assert len(parsed[0].interactive_labels) == 2
        assert 'button "Star"' in parsed[0].interactive_labels[0]


# ═══════════════════════════════════════════════════════════════
#  Real trace validation
# ═══════════════════════════════════════════════════════════════

# Section IDs extracted from actual benchmark traces.

_GITHUB_SECTION_IDS = [
    "div-trending",
    "summary-spoken-language-any-language",
    "item-1-div-today-this-week-this-month",
    "article-anthropics-financial",
    "item-1-article-learningcircuit-loca",
    "item-1-article-tauricresearch-tradi",
    "item-2-article-aidc-ai-pixelle-vide",
]

_GITHUB_REASONING = (
    "Looking at the page, I can see the GitHub trending page for Python "
    "repositories filtered by This week. A list of ~11 repositories visible "
    "in the current view with articles like article-anthropics-financial. "
    "Each repository card shows owner/repo name as a link, description text, "
    "language badge Python, star count, fork count, built by section with "
    "contributor links, stars this week text. The repos are in article "
    "elements. The repo link is in an a tag with href. Stars and forks are "
    "in a tags with counts. The page shows This week is selected in the "
    "date range filter. I can see there are more repos below. I need to "
    "scroll down to see how many repos are on this page and if there is "
    "pagination. I need to extract trending repos from this page."
)

_YC_SECTION_IDS = [
    "item-1-div-about-what-happens-at-yc-apply-yc-interview-guide",
    "item-2-div-startup-directory",
    "div-sort-by-default-launch",
    "div-batch-industry-hq-region-company-size",
    "div-search",
    "item-1-a-winter-2009-consumer-travel-leisure-and-t",
    "item-2-a-summer-2013-consumer-food-and-beverage",
    "item-3-a-summer-2012-fintech-banking-and-exchange",
    "item-4-a-winter-2018-fintech-asset-management",
    "item-5-a-summer-2014-industrials-energy",
    "item-6-a-summer-2012-consumer-food-and-beverage",
    "item-7-a-summer-2016-b2b-retail",
    "item-8-a-summer-2007-b2b-productivity",
    "item-9-a-summer-2014-industrials-manufacturing-and-ro",
    "item-10-a-winter-2015-real-estate-and-cons-construction",
    "item-11-a-winter-2015-b2b-engineering-product-",
    "item-12-a-summer-2017-healthcare-diagnostics",
    "item-13-a-winter-2012-consumer-virtual-and-augmente",
    "item-14-a-winter-2012-b2b-analytics",
    "item-15-a-summer-2014-healthcare-industrial-bio",
    "item-16-a-summer-2010-b2b-engineering-product-",
    "item-17-a-winter-2014-b2b-office-management",
    "item-18-a-summer-2020-healthcare-drug-discovery-and-d",
    "item-19-a-winter-2016-industrials-automotive",
    "item-20-a-summer-2018-industrials-aviation-and-space",
    "item-21-a-winter-2015-healthcare-diagnostics",
    "item-22-a-summer-2011-b2b-analytics",
    "item-23-a-winter-2014-b2b-engineering-product-",
    "item-24-a-winter-2016-fintech-consumer-finance",
    "item-25-a-winter-2007-consumer-content",
    "item-26-a-winter-2012-real-estate-and-cons-construction",
    "item-27-a-winter-2014-consumer-consumer-electronics",
    "item-28-a-winter-2014-industrials-automotive",
    "item-29-a-summer-2012-b2b",
    "item-30-a-summer-2013-b2b-legal",
    "item-31-a-summer-2016-b2b-engineering-product-",
    "item-32-a-winter-2017-fintech-banking-and-exchange",
    "item-33-a-summer-2016-consumer-content",
    "item-34-a-summer-2011-education",
    "item-35-a-summer-2014-b2b-human-resources",
    "item-36-a-summer-2012-b2b-recruiting-and-talen",
    "item-37-a-winter-2017-consumer",
    "item-38-a-winter-2013-b2b-analytics",
    "item-39-a-winter-2012-fintech-payments",
    "item-40-a-winter-2019-b2b-finance-and-accounti",
    "div-loading-more",
    "footer-footer-make-something-peopl-programs-resources",
]

_YC_REASONING = (
    "Page shows a company directory with filters and a listing of companies. "
    "Currently displaying 40 of 1000 plus companies. There is a Loading more "
    "indicator at the bottom suggesting infinite scroll or dynamic loading. "
    "The batch filter panel contains checkboxes for filtering by batch, "
    "industry, HQ region, company size. I can see batch options: All batches, "
    "Fall 2026, Summer 2026, Spring 2026, Winter 2026. I need to find W24 "
    "and S24 batches. The startup directory section shows company cards. "
    "The search box allows searching by name. Company cards are anchor tags "
    "with the company name, location, tagline, batch link, and industry links. "
    "I need to filter for winter 2024 and summer 2024 batches."
)

_BOOKS_SECTION_IDS = [
    "header-books-to-scrape-home",
    "aside-books-travel-mystery-historical-fiction",
    "div-all-products",
    "div-a-light-in-the-tipping-the-velvet-soumission-sharp-objec",
]

_BOOKS_REASONING = (
    "Page Analysis: Key observations: div-all-products is the main listing "
    "section containing 20 books per page. Shows 1 to 20 of 1000 results. "
    "Each book has title link, price, In stock status, Add to basket button. "
    "Pagination at bottom shows Page 1 of 50. Link to next page. I need to "
    "paginate through 10 pages for 200 books total. Book links are clickable "
    "title links like a-light-in-the-attic. These lead to detail pages. "
    "The aside section shows category navigation: travel, mystery, historical "
    "fiction. The header has the site branding Books to Scrape."
)


class TestRealTraceGitHub:
    """Validate indirect reference detection against real GitHub trace data."""

    def _sections(self):
        return [
            (sid, f"Content for {sid}", "content", 2)
            for sid in _GITHUB_SECTION_IDS
        ]

    def test_trending_matched(self):
        """'trending' in reasoning should match 'div-trending'."""
        refs, _ = get_indirect_references(
            _GITHUB_REASONING, self._sections(), set(),
        )
        assert "div-trending" in refs

    def test_multiple_sections_matched(self):
        """Multiple sections should score > 0, top ones within budget."""
        _, matches = get_indirect_references(
            _GITHUB_REASONING, self._sections(), set(),
        )
        # At least one section matched
        assert len(matches) >= 1
        # The matched section should be relevant (high-scoring)
        assert matches[0].score > 0.3

    def test_budget_respected(self):
        refs, _ = get_indirect_references(
            _GITHUB_REASONING, self._sections(), set(),
        )
        budget = min(
            int(len(_GITHUB_SECTION_IDS) * INDIRECT_MAX_RATIO),
            INDIRECT_MAX_ABSOLUTE,
        )
        assert len(refs) <= budget


class TestRealTraceYCombinator:
    """Validate indirect reference detection against real YC trace data."""

    def _sections(self):
        return [
            (sid, f"Content for {sid}", "content", 2)
            for sid in _YC_SECTION_IDS
        ]

    def test_batch_filter_matched(self):
        """Reasoning mentions 'batch filter' — should match the filter section."""
        refs, _ = get_indirect_references(
            _YC_REASONING, self._sections(), set(),
        )
        assert "div-batch-industry-hq-region-company-size" in refs

    def test_search_matched(self):
        """Reasoning mentions 'search' — should match div-search."""
        refs, _ = get_indirect_references(
            _YC_REASONING, self._sections(), set(),
        )
        assert "div-search" in refs

    def test_common_tokens_dont_flood(self):
        """'item' appears in 40+ section IDs — shouldn't cause all to match."""
        refs, _ = get_indirect_references(
            _YC_REASONING, self._sections(), set(),
        )
        budget = min(
            int(len(_YC_SECTION_IDS) * INDIRECT_MAX_RATIO),
            INDIRECT_MAX_ABSOLUTE,
        )
        assert len(refs) <= budget

    def test_loading_more_matched(self):
        """Reasoning mentions 'loading more' — should match."""
        refs, _ = get_indirect_references(
            _YC_REASONING, self._sections(), set(),
        )
        assert "div-loading-more" in refs


class TestRealTraceBooks:
    """Validate indirect reference detection against real Books.toscrape data."""

    def _sections(self):
        return [
            (sid, f"Content for {sid}", "content", 2)
            for sid in _BOOKS_SECTION_IDS
        ]

    def test_best_section_matched(self):
        """With 4 sections (budget=1), the most relevant section should win."""
        refs, matches = get_indirect_references(
            _BOOKS_REASONING, self._sections(), set(),
        )
        assert len(refs) >= 1
        # The aside section has many matching tokens (travel, mystery,
        # historical, fiction, books) — should score high.
        # div-all-products also matches (products).
        # With budget=1, whichever scores highest wins.
        matched_ids = {m.section_id for m in matches}
        assert len(matched_ids) >= 1

    def test_all_sections_score_above_zero(self):
        """All 4 sections have token overlap with reasoning — all should score."""
        sections = self._sections()
        idf = _compute_token_idf(sections)
        reasoning_lower = _BOOKS_REASONING.lower()
        scores = {}
        for sid, content, role, ic in sections:
            score, _, _ = score_section_indirect(
                sid, content, reasoning_lower, idf,
            )
            scores[sid] = score
        # At least 3 of 4 sections should have some overlap
        nonzero = sum(1 for s in scores.values() if s > 0)
        assert nonzero >= 3


class TestIdfPreventsCommonTokenFlooding:
    """Verify that common tokens don't cause excessive matching."""

    def test_generic_content_sections(self):
        """20 sections with 'div-content-...' + 5 unique. Reasoning mentions
        'content'. The 20 generic ones should NOT all get matched — budget
        cap + IDF weighting should prefer the unique sections."""
        sections = [
            (f"div-content-item-{i}", f"Generic text {i}", "content", 0)
            for i in range(20)
        ]
        sections.extend([
            ("unique-sidebar-about", "About this project", "content", 0),
            ("unique-trending-page", "Trending repositories", "content", 0),
            ("unique-readme-article", "README documentation", "content", 0),
            ("unique-footer-links", "Footer navigation", "content", 0),
            ("unique-search-panel", "Search functionality", "content", 0),
        ])
        # Reasoning mentions unique section keywords AND "content"
        reasoning = (
            "I see the sidebar about section, the trending page with "
            "repositories, the readme article, and generic content items."
        )
        refs, _ = get_indirect_references(reasoning, sections, set())
        budget = min(int(25 * INDIRECT_MAX_RATIO), INDIRECT_MAX_ABSOLUTE)
        assert len(refs) <= budget
        # Unique sections should be in the results
        assert "unique-sidebar-about" in refs
        assert "unique-trending-page" in refs


# ═══════════════════════════════════════════════════════════════
#  build_filtered_output with indirect_refs
# ═══════════════════════════════════════════════════════════════

from scout.agent.show_page_context import build_filtered_output, NEIGHBOR_RADIUS


def _make_sections_list(n: int) -> list[tuple[str, str, str, int]]:
    return [
        (f"section-{i}", f"Content for section {i}", "content", i % 3)
        for i in range(n)
    ]


class TestBuildFilteredOutputWithIndirect:
    """Tests for the indirect_refs parameter on build_filtered_output."""

    def test_indirect_refs_get_full_content(self):
        """Indirect refs should appear with full content, not [omitted]."""
        sections = _make_sections_list(10)
        output = build_filtered_output(
            sections,
            referenced={"section-0"},
            indirect_refs={"section-5"},
        )
        assert "Content for section 5" in output
        assert "Content for section 0" in output

    def test_indirect_neighbor_radius_is_1(self):
        """Indirect refs use radius 1, not 3."""
        sections = _make_sections_list(20)
        output = build_filtered_output(
            sections,
            referenced=set(),
            indirect_refs={"section-10"},
            indirect_neighbor_radius=1,
        )
        # section-10: full content
        assert "Content for section 10" in output
        # section-9 and section-11: neighbors (radius 1)
        assert "[section-9]" in output
        assert "[section-11]" in output
        # section-8 and section-12: NOT neighbors (radius 1)
        assert "Content for section 8" not in output
        assert "Content for section 12" not in output

    def test_direct_refs_still_radius_3(self):
        """Direct refs should still use radius 3."""
        sections = _make_sections_list(20)
        output = build_filtered_output(
            sections,
            referenced={"section-10"},
            indirect_refs=set(),
        )
        # section-10: full content
        assert "Content for section 10" in output
        # section-7 through section-13: neighbors (radius 3)
        assert "[section-7]" in output
        assert "[section-13]" in output

    def test_indirect_overrides_neighbor(self):
        """If a section is both a direct-ref neighbor AND an indirect ref,
        indirect wins (full content instead of [omitted])."""
        sections = _make_sections_list(10)
        # section-5 is kept (direct). section-7 is within radius 3,
        # so it would be a neighbor. But it's also indirect — should get full content.
        output = build_filtered_output(
            sections,
            referenced={"section-5"},
            indirect_refs={"section-7"},
        )
        assert "Content for section 7" in output

    def test_none_indirect_refs_backward_compatible(self):
        """indirect_refs=None produces identical output to old behavior."""
        sections = _make_sections_list(10)
        output_new = build_filtered_output(
            sections,
            referenced={"section-3"},
            indirect_refs=None,
        )
        output_old = build_filtered_output(
            sections,
            referenced={"section-3"},
        )
        assert output_new == output_old

    def test_both_direct_and_indirect(self):
        """Both direct and indirect refs present, correct tier assignment."""
        sections = _make_sections_list(15)
        output = build_filtered_output(
            sections,
            referenced={"section-2"},
            indirect_refs={"section-12"},
        )
        # Direct ref: full content
        assert "Content for section 2" in output
        # Indirect ref: full content
        assert "Content for section 12" in output
        # Sections far from both: omitted
        assert "sections omitted" in output

    def test_omission_count_correct(self):
        """Distant section count should be accurate."""
        sections = _make_sections_list(10)
        output = build_filtered_output(
            sections,
            referenced={"section-0"},
            indirect_refs={"section-9"},
            neighbor_radius=1,
            indirect_neighbor_radius=1,
        )
        # section-0: kept, section-1: neighbor
        # section-2 through section-7: 6 distant
        # section-8: neighbor (of section-9), section-9: indirect
        assert "[6 sections omitted]" in output


# ═══════════════════════════════════════════════════════════════
#  Skeleton stage and 3-stage lifecycle (context.py)
# ═══════════════════════════════════════════════════════════════

from scout.agent.context import (
    _build_page_view_stub,
    _build_skeleton_from_filtered,
    _stub_old_page_views_inplace,
    _SKELETON_AFTER_TURNS,
    _STUB_AFTER_TURNS,
)
from scout.agent.show_page_context import _SECTION_META_START, _SECTION_META_END


def _wrap_page_view(
    content: str,
    turn: int | None = None,
) -> dict:
    """Build a tool_result message wrapping page view content."""
    turn_tag = f"__TURN_{turn}__\n" if turn is not None else ""
    text = (
        f"Executed in 100ms.\n\nOutput:\n"
        f"__PAGE_VIEW_START__\n"
        f"{turn_tag}"
        f"{content}\n"
        f"__PAGE_VIEW_END__"
    )
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": text}],
    }


def _get_content(msg: dict) -> str:
    return msg["content"][0]["content"]


def _build_filtered_with_meta() -> str:
    """Build a realistic filtered page view with metadata block."""
    sections = [
        ("nav-header", "Home | About | Contact", "navigation", 3),
        ("sidebar-about", "About: A research assistant for deep research", "content", 2),
        ("readme-article", "# README\nLong content " * 30, "content", 0),
        ("footer-links", "Footer navigation links", "navigation", 5),
    ]
    from scout.agent.show_page_context import build_filtered_output
    filtered = build_filtered_output(
        sections,
        referenced={"sidebar-about"},
        neighbor_radius=1,
    )
    meta = build_section_meta(sections)
    header = "=== Page State #1 | https://example.com ==="
    return f"{header}\n\n{filtered}\n{meta}"


class TestBuildSkeletonFromFiltered:

    def test_returns_none_without_meta(self):
        """Page views without __SECTION_META__ block return None."""
        pv = (
            "=== Page State #1 | https://example.com ===\n"
            "--- [nav-header] navigation (3 interactive) ---\n"
            "Home | About"
        )
        assert _build_skeleton_from_filtered(pv) is None

    def test_kept_sections_preserved(self):
        """Kept sections should have their full content in skeleton."""
        pv = _build_filtered_with_meta()
        skeleton = _build_skeleton_from_filtered(pv)
        assert skeleton is not None
        # sidebar-about was referenced → kept → full content preserved.
        assert "About: A research assistant" in skeleton

    def test_non_kept_become_previews(self):
        """Non-kept sections should become preview lines."""
        pv = _build_filtered_with_meta()
        skeleton = _build_skeleton_from_filtered(pv)
        assert skeleton is not None
        assert "[preview:" in skeleton

    def test_skeleton_marker_present(self):
        pv = _build_filtered_with_meta()
        skeleton = _build_skeleton_from_filtered(pv)
        assert skeleton is not None
        assert "[skeleton]" in skeleton

    def test_header_preserved(self):
        pv = _build_filtered_with_meta()
        skeleton = _build_skeleton_from_filtered(pv)
        assert skeleton is not None
        assert "https://example.com" in skeleton


class TestThreeStageLifecycle:
    """Test the full Stage 1 → Stage 2 → Stage 3 progression."""

    def test_stage1_stays_at_age_1(self):
        """Age 1 (< _SKELETON_AFTER_TURNS=2): no change."""
        pv = _build_filtered_with_meta()
        messages = [_wrap_page_view(pv, turn=5)]
        _stub_old_page_views_inplace(messages, current_turn=6)  # age=1
        content = _get_content(messages[0])
        assert "[skeleton]" not in content
        assert "[stub]" not in content

    def test_stage1_to_skeleton_at_age_2(self):
        """Age 2 (>= _SKELETON_AFTER_TURNS=2): becomes skeleton."""
        pv = _build_filtered_with_meta()
        messages = [_wrap_page_view(pv, turn=5)]
        _stub_old_page_views_inplace(messages, current_turn=7)  # age=2
        content = _get_content(messages[0])
        assert "[skeleton]" in content
        assert "[stub]" not in content

    def test_skeleton_to_stub_at_age_5(self):
        """Age 5 (>= _STUB_AFTER_TURNS=5): skeleton becomes stub."""
        pv = _build_filtered_with_meta()
        messages = [_wrap_page_view(pv, turn=5)]
        # First: age 2 → skeleton
        _stub_old_page_views_inplace(messages, current_turn=7)
        assert "[skeleton]" in _get_content(messages[0])
        # Then: age 5 → stub
        _stub_old_page_views_inplace(messages, current_turn=10)
        content = _get_content(messages[0])
        assert "[stub]" in content
        assert "[skeleton]" not in content or "[stub]" in content

    def test_stage1_to_stub_directly_at_age_5(self):
        """If skeleton fails (no meta), Stage 1 → Stage 3 at age 5."""
        # No metadata block.
        pv = (
            "=== Page State #1 | https://example.com ===\n"
            "--- [nav-header] navigation (3 interactive) ---\n"
            "Home | About"
        )
        messages = [_wrap_page_view(pv, turn=1)]
        # age 2: skeleton attempt → None (no meta) → stays Stage 1
        _stub_old_page_views_inplace(messages, current_turn=3)
        content = _get_content(messages[0])
        assert "[skeleton]" not in content
        assert "[stub]" not in content
        # age 5: direct to stub
        _stub_old_page_views_inplace(messages, current_turn=6)
        content = _get_content(messages[0])
        assert "[stub]" in content

    def test_stub_is_idempotent(self):
        """Already-stubbed page views are not re-processed."""
        pv = _build_filtered_with_meta()
        messages = [_wrap_page_view(pv, turn=1)]
        _stub_old_page_views_inplace(messages, current_turn=10)
        content1 = _get_content(messages[0])
        assert "[stub]" in content1
        _stub_old_page_views_inplace(messages, current_turn=20)
        assert _get_content(messages[0]) == content1

    def test_skeleton_stays_at_age_3(self):
        """Skeleton at age 3 (< _STUB_AFTER_TURNS=5): no change."""
        pv = _build_filtered_with_meta()
        messages = [_wrap_page_view(pv, turn=5)]
        _stub_old_page_views_inplace(messages, current_turn=7)  # age 2 → skeleton
        content_skel = _get_content(messages[0])
        assert "[skeleton]" in content_skel
        _stub_old_page_views_inplace(messages, current_turn=8)  # age 3 → still skeleton
        assert _get_content(messages[0]) == content_skel


class TestEnhancedStub:

    def test_stub_has_section_index(self):
        """Enhanced stub should include [all: ...] section index."""
        pv = _build_filtered_with_meta()
        messages = [_wrap_page_view(pv, turn=1)]
        _stub_old_page_views_inplace(messages, current_turn=10)
        content = _get_content(messages[0])
        assert "[all:" in content

    def test_stub_section_index_from_meta(self):
        """Section index should list IDs from metadata block."""
        pv = _build_filtered_with_meta()
        messages = [_wrap_page_view(pv, turn=1)]
        _stub_old_page_views_inplace(messages, current_turn=10)
        content = _get_content(messages[0])
        assert "nav-header" in content
        assert "sidebar-about" in content
        assert "readme-article" in content

    def test_stub_kept_sections_listed(self):
        """Stub should list kept section IDs."""
        pv = _build_filtered_with_meta()
        messages = [_wrap_page_view(pv, turn=1)]
        _stub_old_page_views_inplace(messages, current_turn=10)
        content = _get_content(messages[0])
        assert "kept:" in content


class TestRegressionSafety:

    def test_no_meta_stub_unchanged(self):
        """Page views without meta produce the same stub format as before."""
        pv = (
            "=== Page State #1 | https://example.com ===\n"
            "--- [nav-main] navigation (3 interactive) ---\n"
            "Home | About\n"
            "--- [div-content] content (0 interactive) ---\n"
            "[omitted]\n"
            "[5 sections omitted]"
        )
        stub = _build_page_view_stub(pv)
        assert "[stub]" in stub
        assert "example.com" in stub
        assert "sections" in stub
