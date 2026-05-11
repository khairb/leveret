"""Tests for the Selector Extractor — driven by real trace data.

Each test case uses an actual code block the AI wrote during scraping,
with the expected extraction results verified manually.
"""

from __future__ import annotations

import pytest

from scout.agent.selector_extractor import extract_selectors, ExtractionResult


# ═══════════════════════════════════════════════════════════════════════
#  Test Data — Real code blocks from traces
# ═══════════════════════════════════════════════════════════════════════

# ── Category: Simple page.click() with literal selector ──────────────

SIMPLE_CLICK = """\
await page.click('article.product_pod h3 a')
await page.wait_for_load_state("domcontentloaded")
await show_page(page)
"""

CLICK_ATTRIBUTE_SELECTOR = """\
await page.click('button[aria-label="Waschmaschine"]')
await page.wait_for_timeout(300)
"""

CLICK_ID_SELECTOR = """\
await page.click('#didomi-notice-agree-button')
await page.wait_for_timeout(1000)
"""

CLICK_HAS_TEXT = """\
await page.click('a:has-text("Unterkünfte anzeigen")')
await page.wait_for_load_state('domcontentloaded')
"""

CLICK_DATA_TESTID = """\
await page.click('button[data-testid="structured-search-input-search-button"]')
await page.wait_for_load_state("networkidle")
"""

# ── Category: page.fill() ────────────────────────────────────────────

SIMPLE_FILL = """\
await page.fill('#bigsearch-query-location-input', 'Berlin')
await page.wait_for_timeout(1000)
await show_page(page)
"""

# ── Category: page.locator().click() ─────────────────────────────────

LOCATOR_CLICK = """\
await page.locator('a._showMoreLess_18olp_259').first.click()
await page.wait_for_timeout(500)
await show_page(page)
"""

LOCATOR_CLICK_HAS_TEXT_KWARG = """\
await page.locator('div[role="button"]', has_text='Wann').click()
await page.wait_for_timeout(500)
"""

LOCATOR_FILTER_CLICK = """\
search_btn = page.locator('button[data-testid="structured-search-input-search-button"]')
await search_btn.click()
await page.wait_for_load_state('domcontentloaded')
"""

# ── Category: page.query_selector() + read ───────────────────────────

QS_READ = """\
more_link = await page.query_selector('a[href*="?p="]')
if more_link:
    href = await more_link.get_attribute('href')
    print(f"Found 'More' link: {href}")
else:
    print("No 'More' link found")
"""

QS_CLICK = """\
next_button = await page.query_selector('ul.pager li.next a')
if next_button:
    await next_button.click()
    await page.wait_for_load_state("domcontentloaded")
"""

# ── Category: page.evaluate() with querySelectorAll ──────────────────

EVALUATE_QSA_MAP = """\
company_links = await page.evaluate(\"\"\"
    () => Array.from(document.querySelectorAll('a[href^="/companies/"]'))
        .map(a => a.getAttribute('href'))
        .filter(href => href && href.startsWith('/companies/') && !href.includes('?'))
\"\"\", isolated_context=True)
print(f"Found {len(company_links)} company links")
"""

EVALUATE_QSA_COMPLEX = """\
test_data = await page.evaluate(\"\"\"
    () => {
        const stories = [];
        const rows = document.querySelectorAll('tr.athing.submission');
        for (let i = 0; i < Math.min(3, rows.length); i++) {
            const titleRow = rows[i];
            const metaRow = titleRow.nextElementSibling;
            const titleLink = titleRow.querySelector('span.titleline > a[href^="http"]');
            const title = titleLink?.innerText?.trim() || '';
            const url = titleLink?.href || '';
            const scoreText = metaRow?.querySelector('span.score')?.innerText || '';
            const author = metaRow?.querySelector('a.hnuser')?.innerText?.trim() || '';
            stories.push({ title, url, score: scoreText, author });
        }
        return stories;
    }
\"\"\", isolated_context=True)
"""

EVALUATE_QSA_AIRBNB = """\
listings = await page.evaluate(\"\"\"
    () => {
        const cards = document.querySelectorAll('[data-testid="card-container"]');
        return Array.from(cards).map(card => {
            const title = card.querySelector('[data-testid="listing-card-title"]')?.innerText?.trim() || '';
            const price = card.querySelector('span[aria-label*="€"]')?.innerText?.trim() || '';
            const link = card.querySelector('a[href*="/rooms/"]')?.getAttribute('href') || '';
            return { title, price, link };
        });
    }
\"\"\", isolated_context=True)
"""

# ── Category: page.evaluate() with single querySelector ──────────────

EVALUATE_QS_SINGLE = """\
title = await page.evaluate(\"\"\"
    () => document.querySelector('h1')?.innerText?.trim() || ''
\"\"\", isolated_context=True)
print(f"Title: {title}")
"""

EVALUATE_QS_DATA_ATTR = """\
info = await page.evaluate(\"\"\"
    () => {
        const root = document.querySelector('[data-page]');
        if (!root) return null;
        const raw = root.getAttribute('data-page');
        return { len: raw ? raw.length : 0 };
    }
\"\"\", isolated_context=True)
"""

# ── Category: page.evaluate() with scrollTo (no selectors) ──────────

EVALUATE_SCROLL_ONLY = """\
await page.evaluate("window.scrollTo(0, document.body.scrollHeight)", isolated_context=True)
await page.wait_for_timeout(1000)
await show_page(page)
"""

# ── Category: page.goto() ────────────────────────────────────────────

SIMPLE_GOTO = """\
await page.goto('https://www.ycombinator.com/companies/airbnb', wait_until='domcontentloaded', timeout=30000)
await show_page(page)
"""

# ── Category: page.go_back() ─────────────────────────────────────────

SIMPLE_GO_BACK = """\
await page.go_back()
await page.wait_for_load_state("domcontentloaded")
await show_page(page)
"""

# ── Category: get_by_text / get_by_role ──────────────────────────────

GET_BY_TEXT_CLICK = """\
await page.get_by_text('See all options').first.click()
await page.wait_for_load_state('domcontentloaded')
await show_page(page)
"""

GET_BY_ROLE_CLICK = """\
await page.get_by_role("link", name="2").click()
await page.wait_for_timeout(2000)
"""

# ── Category: keyboard.press ─────────────────────────────────────────

KEYBOARD_PRESS = """\
await page.keyboard.press('Escape')
await page.wait_for_timeout(1000)
await page.go_back()
await page.wait_for_timeout(2000)
await show_page(page)
"""

# ── Category: page.inner_text / page.get_attribute ───────────────────

INNER_TEXT_READ = """\
text = await page.inner_text('.selector')
attr = await page.get_attribute('a.link', 'href')
"""

# ── Category: Multi-click sequence (no loop) — filter chain ──────────

MULTI_CLICK_FILTERS = """\
await page.click('button[aria-label="Waschmaschine"]')
await page.wait_for_timeout(300)

await page.click('button[aria-label="TV"]')
await page.wait_for_timeout(300)

await page.click('button[id="filter-item-amenities-4"]')
await page.wait_for_timeout(300)

await page.click('a:has-text("Unterkünfte anzeigen")')
await page.wait_for_load_state('domcontentloaded')
"""

# ── Category: Multi-click — stepper (no page change) ─────────────────

MULTI_CLICK_STEPPER = """\
adults_btn = page.locator('button[data-testid="stepper-adults-increase-button"]')
children_btn = page.locator('button[data-testid="stepper-children-increase-button"]')
infants_btn = page.locator('button[data-testid="stepper-infants-increase-button"]')

await adults_btn.click()
await adults_btn.click()
await children_btn.click()
await children_btn.click()
await infants_btn.click()
"""

# ── Category: Complete search flow (multi-step, final navigation) ─────

SEARCH_FLOW = """\
await page.fill('#bigsearch-query-location-input', 'Berlin')
await page.wait_for_timeout(500)
await page.keyboard.press('Enter')
await page.wait_for_timeout(1000)

await page.locator('div[role="button"]', has_text='Wann').click()
await page.wait_for_timeout(500)

await page.click('button[aria-label*="20, Monday, April 2026"]')
await page.wait_for_timeout(300)
await page.click('button[aria-label*="19, Tuesday, May 2026"]')
await page.wait_for_timeout(300)

await page.locator('button[data-testid="structured-search-input-search-button"]').click()
await page.wait_for_load_state('domcontentloaded')
"""

# ── Category: Pagination loop (click in for-loop) ────────────────────

PAGINATION_LOOP = """\
all_data = []
for page_num in range(1, 5):
    data = await page.evaluate(\"\"\"
        () => Array.from(document.querySelectorAll('[data-testid="card-container"]'))
            .map(card => card.querySelector('a')?.href || '')
    \"\"\", isolated_context=True)
    all_data.extend(data)
    next_btn = page.locator('a[aria-label="Weiter"]')
    if await next_btn.count() > 0:
        await next_btn.click()
        await page.wait_for_load_state('domcontentloaded')
"""

# ── Category: Detail page loop (goto in for-loop) ────────────────────

DETAIL_LOOP = """\
results = []
for link in company_links[:5]:
    await page.goto(f'https://www.ycombinator.com{link}', wait_until='domcontentloaded')
    data = await page.evaluate(\"\"\"
        () => ({
            name: document.querySelector('h1')?.innerText?.trim() || '',
            desc: document.querySelector('p.f4')?.innerText?.trim() || ''
        })
    \"\"\", isolated_context=True)
    results.append(data)
    await page.wait_for_timeout(1000)
"""

# ── Category: goto + click (2-step) ──────────────────────────────────

GOTO_THEN_CLICK = """\
await page.goto('https://www.airbnb.de/s/Berlin/homes', wait_until='domcontentloaded')
await page.wait_for_timeout(2000)
await page.click('#didomi-notice-agree-button')
await page.wait_for_timeout(1000)
"""

# ── Category: evaluate with .click() inside JS ───────────────────────

EVALUATE_JS_CLICK = """\
await page.evaluate(\"\"\"
    () => {
        const btn = document.querySelector('button.close-overlay');
        if (btn) btn.click();
    }
\"\"\", isolated_context=True)
await page.wait_for_timeout(500)
"""

# ── Category: page.reload() ──────────────────────────────────────────

RELOAD_THEN_EXTRACT = """\
await page.reload()
await page.wait_for_selector('div[data-testid="search-results"]', timeout=10000)
data = await page.evaluate(\"\"\"
    () => document.querySelectorAll('[data-testid="card-container"]').length
\"\"\", isolated_context=True)
"""

# ── Category: show_page / zoom_section only (no interactions) ────────

SHOW_PAGE_ONLY = """\
await show_page(page)
"""

ZOOM_ONLY = """\
await zoom_section(page, "item-1-tr-qwen-ai", "item-1-tr-mfiguiere-4-hours-ago")
"""

# ── Category: Mixed — evaluate with querySelectorAll + scroll ────────

EVALUATE_QSA_PLUS_SCROLL = """\
await page.evaluate("window.scrollTo(0, document.body.scrollHeight)", isolated_context=True)
await page.wait_for_timeout(2000)

links = await page.evaluate(\"\"\"
    () => Array.from(document.querySelectorAll('a[href*="/rooms/"]'))
        .map(a => a.href)
        .filter((v, i, arr) => arr.indexOf(v) === i)
\"\"\", isolated_context=True)
print(f"Found {len(links)} unique listing links")
"""

# ── Category: Fallback chain inside evaluate ─────────────────────────

EVALUATE_FALLBACK_CHAIN = """\
desc = await page.evaluate(\"\"\"
    () => {
        const el = document.querySelector('[data-testid="readme-blob"]')
            || document.querySelector('article')
            || document.querySelector('div[class*="markdown-body"]');
        return el ? el.innerText.trim().slice(0, 500) : '';
    }
\"\"\", isolated_context=True)
"""

# ── Category: locator.evaluate_all ───────────────────────────────────

LOCATOR_EVALUATE_ALL = """\
labels = await page.locator('button[aria-label]').evaluate_all(
    'els => els.map(el => el.getAttribute("aria-label"))'
)
print(labels[:10])
"""

# ── Category: wait_for_selector ──────────────────────────────────────

WAIT_FOR_SELECTOR = """\
await page.wait_for_selector('tr.athing', state='visible', timeout=30_000)
"""

# ── Category: page.route (no selectors) ──────────────────────────────

ROUTE_BLOCK = """\
await page.route("**/*.{png,jpg,jpeg,gif,svg,css,woff,woff2}", lambda route: route.abort())
await page.goto('https://example.com', wait_until='domcontentloaded')
"""


# ═══════════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════════

class TestSimpleClicks:
    """Test extraction of simple page.click() calls."""

    def test_click_css_descendant(self):
        """page.click with CSS descendant selector → navigating."""
        results = extract_selectors(SIMPLE_CLICK)
        assert len(results) == 1
        r = results[0]
        assert r.selector == "article.product_pod h3 a"
        assert r.selector_type == "css"
        assert r.action_category == "navigating"
        assert r.action == "click"

    def test_click_aria_label(self):
        """page.click with aria-label attribute selector."""
        results = extract_selectors(CLICK_ATTRIBUTE_SELECTOR)
        assert len(results) == 1
        r = results[0]
        assert r.selector == 'button[aria-label="Waschmaschine"]'
        assert r.action_category == "navigating"

    def test_click_id_selector(self):
        """page.click with ID selector."""
        results = extract_selectors(CLICK_ID_SELECTOR)
        assert len(results) == 1
        assert results[0].selector == "#didomi-notice-agree-button"

    def test_click_has_text(self):
        """page.click with Playwright :has-text() pseudo."""
        results = extract_selectors(CLICK_HAS_TEXT)
        assert len(results) == 1
        r = results[0]
        assert r.selector == 'a:has-text("Unterkünfte anzeigen")'
        assert r.selector_type == "playwright"

    def test_click_data_testid(self):
        """page.click with data-testid attribute selector."""
        results = extract_selectors(CLICK_DATA_TESTID)
        assert len(results) == 1
        assert results[0].selector == 'button[data-testid="structured-search-input-search-button"]'


class TestFill:
    """Test extraction of page.fill() calls."""

    def test_simple_fill(self):
        """page.fill → mutating action (typing changes element state)."""
        results = extract_selectors(SIMPLE_FILL)
        assert len(results) == 1
        r = results[0]
        assert r.selector == "#bigsearch-query-location-input"
        assert r.action_category == "mutating"
        assert r.action == "fill"


class TestLocatorClick:
    """Test extraction of page.locator().click() chains."""

    def test_locator_first_click(self):
        """page.locator('...').first.click()."""
        results = extract_selectors(LOCATOR_CLICK)
        assert len(results) == 1
        r = results[0]
        assert r.selector == "a._showMoreLess_18olp_259"
        assert r.action_category == "navigating"

    def test_locator_has_text_kwarg(self):
        """page.locator('...', has_text='...').click()."""
        results = extract_selectors(LOCATOR_CLICK_HAS_TEXT_KWARG)
        assert len(results) == 1
        r = results[0]
        assert r.selector == 'div[role="button"]'
        assert r.action_category == "navigating"

    def test_locator_assigned_then_clicked(self):
        """Locator assigned to variable, then .click() called."""
        results = extract_selectors(LOCATOR_FILTER_CLICK)
        assert len(results) == 1
        r = results[0]
        assert r.selector == 'button[data-testid="structured-search-input-search-button"]'
        assert r.action_category == "navigating"


class TestQuerySelector:
    """Test extraction of page.query_selector() calls."""

    def test_qs_read(self):
        """query_selector + get_attribute → passive."""
        results = extract_selectors(QS_READ)
        assert len(results) >= 1
        # The direct query_selector call is extracted
        qs_results = [r for r in results if r.action == "query_selector"]
        assert len(qs_results) == 1
        assert qs_results[0].selector == 'a[href*="?p="]'
        assert qs_results[0].action_category == "passive"

    def test_qs_click(self):
        """query_selector followed by .click() → navigating."""
        results = extract_selectors(QS_CLICK)
        assert len(results) >= 1
        # The click on the variable should be detected
        click_results = [r for r in results if r.action == "click"]
        assert len(click_results) == 1
        assert click_results[0].selector == "ul.pager li.next a"
        assert click_results[0].action_category == "navigating"


class TestEvaluateQuerySelectorAll:
    """Test extraction of selectors inside page.evaluate() JS."""

    def test_evaluate_qsa_map(self):
        """querySelectorAll inside evaluate → CSS selectors extracted."""
        results = extract_selectors(EVALUATE_QSA_MAP)
        assert len(results) >= 1
        assert any(r.selector == 'a[href^="/companies/"]' for r in results)
        assert all(r.selector_type == "css" for r in results)
        assert all(r.action_category == "passive" for r in results)

    def test_evaluate_qsa_complex(self):
        """Complex evaluate with nested querySelector calls."""
        results = extract_selectors(EVALUATE_QSA_COMPLEX)
        selectors = {r.selector for r in results}
        assert "tr.athing.submission" in selectors
        assert 'span.titleline > a[href^="http"]' in selectors
        assert "span.score" in selectors
        assert "a.hnuser" in selectors
        assert all(r.action_category == "passive" for r in results)

    def test_evaluate_qsa_airbnb(self):
        """Airbnb card extraction with data-testid selectors."""
        results = extract_selectors(EVALUATE_QSA_AIRBNB)
        selectors = {r.selector for r in results}
        assert '[data-testid="card-container"]' in selectors
        assert '[data-testid="listing-card-title"]' in selectors
        assert 'a[href*="/rooms/"]' in selectors


class TestEvaluateQuerySelector:
    """Test extraction of single querySelector inside evaluate."""

    def test_evaluate_qs_single(self):
        """Single querySelector → one CSS selector."""
        results = extract_selectors(EVALUATE_QS_SINGLE)
        assert len(results) == 1
        assert results[0].selector == "h1"
        assert results[0].selector_type == "css"

    def test_evaluate_qs_data_attr(self):
        """querySelector with data attribute."""
        results = extract_selectors(EVALUATE_QS_DATA_ATTR)
        assert len(results) == 1
        assert results[0].selector == "[data-page]"


class TestEvaluateNoSelectors:
    """Test that evaluate blocks with no selectors return empty."""

    def test_scroll_only(self):
        """scrollTo has no DOM selector → empty results."""
        results = extract_selectors(EVALUATE_SCROLL_ONLY)
        assert len(results) == 0


class TestNavigation:
    """Test page-changing navigation actions."""

    def test_goto(self):
        """page.goto → page_changing, no DOM selector."""
        results = extract_selectors(SIMPLE_GOTO)
        # goto doesn't have a DOM selector to highlight
        assert len(results) == 0

    def test_go_back(self):
        """page.go_back → page_changing, no DOM selector."""
        results = extract_selectors(SIMPLE_GO_BACK)
        assert len(results) == 0

    def test_reload(self):
        """page.reload → page_changing, evaluate selectors still extracted."""
        results = extract_selectors(RELOAD_THEN_EXTRACT)
        # reload is page-changing; selectors after it are on new page
        # The evaluate selectors should be extracted but marked after boundary
        selectors = [r.selector for r in results]
        assert '[data-testid="card-container"]' in selectors


class TestGetByHelpers:
    """Test extraction of get_by_text / get_by_role."""

    def test_get_by_text(self):
        """page.get_by_text().click() → playwright selector."""
        results = extract_selectors(GET_BY_TEXT_CLICK)
        assert len(results) == 1
        r = results[0]
        assert r.selector == "See all options"
        assert r.selector_type == "playwright"
        assert r.action == "get_by_text"
        assert r.action_category == "navigating"

    def test_get_by_role(self):
        """page.get_by_role().click() → playwright selector."""
        results = extract_selectors(GET_BY_ROLE_CLICK)
        assert len(results) == 1
        r = results[0]
        assert r.selector_type == "playwright"
        assert r.action == "get_by_role"


class TestMultiInteraction:
    """Test multi-interaction code blocks."""

    def test_multi_click_filters(self):
        """Multiple filter clicks + final navigate."""
        results = extract_selectors(MULTI_CLICK_FILTERS)
        assert len(results) == 4
        # First 3 are filter clicks, last is navigation
        assert results[0].selector == 'button[aria-label="Waschmaschine"]'
        assert results[1].selector == 'button[aria-label="TV"]'
        assert results[2].selector == 'button[id="filter-item-amenities-4"]'
        assert results[3].selector == 'a:has-text("Unterkünfte anzeigen")'
        # All clicks are navigating
        assert all(r.action_category == "navigating" for r in results)

    def test_multi_click_stepper(self):
        """Stepper clicks — locator assigned then clicked multiple times."""
        results = extract_selectors(MULTI_CLICK_STEPPER)
        # Should extract the 3 unique selectors
        selectors = {r.selector for r in results}
        assert 'button[data-testid="stepper-adults-increase-button"]' in selectors
        assert 'button[data-testid="stepper-children-increase-button"]' in selectors
        assert 'button[data-testid="stepper-infants-increase-button"]' in selectors

    def test_search_flow_boundary(self):
        """Complete search flow — selectors before and after page-change."""
        results = extract_selectors(SEARCH_FLOW)
        assert len(results) >= 5
        # fill is passive
        fills = [r for r in results if r.action == "fill"]
        assert len(fills) >= 1
        assert fills[0].action_category == "mutating"


class TestLoopDetection:
    """Test that loops with page-changing actions are detected."""

    def test_pagination_loop(self):
        """Loop with click inside → should detect loop context."""
        results = extract_selectors(PAGINATION_LOOP)
        # The evaluate selectors are extractable, but the click
        # inside the loop makes it a multi-navigation block.
        # Depending on design: either skip all, or only extract
        # selectors before the loop.
        loop_clicks = [r for r in results if r.action == "click" and r.in_loop]
        assert len(loop_clicks) > 0

    def test_detail_page_loop(self):
        """Loop with goto inside → multi-navigation, detect loop."""
        results = extract_selectors(DETAIL_LOOP)
        loop_gotos = [r for r in results if r.in_loop]
        assert len(loop_gotos) > 0


class TestGotoThenClick:
    """Test goto + click combinations."""

    def test_goto_then_click(self):
        """goto navigates, then click on new page."""
        results = extract_selectors(GOTO_THEN_CLICK)
        # goto has no DOM selector; click selector is after page change
        click_results = [r for r in results if r.action == "click"]
        assert len(click_results) == 1
        assert click_results[0].selector == "#didomi-notice-agree-button"


class TestEvaluateJsClick:
    """Test detection of .click() hidden inside page.evaluate() JS."""

    def test_js_click_detected(self):
        """JS-level click inside evaluate → navigating."""
        results = extract_selectors(EVALUATE_JS_CLICK)
        assert len(results) == 1
        r = results[0]
        assert r.selector == "button.close-overlay"
        assert r.action_category == "navigating"
        assert r.selector_type == "css"


class TestNoInteraction:
    """Test that non-interaction code returns empty."""

    def test_show_page_only(self):
        results = extract_selectors(SHOW_PAGE_ONLY)
        assert len(results) == 0

    def test_zoom_only(self):
        results = extract_selectors(ZOOM_ONLY)
        assert len(results) == 0

    def test_route_block(self):
        """page.route has no DOM selector."""
        results = extract_selectors(ROUTE_BLOCK)
        # Only the goto, which has no DOM selector to highlight
        assert len(results) == 0


class TestMixedPatterns:
    """Test blocks with mixed patterns."""

    def test_scroll_then_evaluate(self):
        """Scroll (no selector) + evaluate with selectors."""
        results = extract_selectors(EVALUATE_QSA_PLUS_SCROLL)
        selectors = {r.selector for r in results}
        assert 'a[href*="/rooms/"]' in selectors
        # scrollTo should not produce a selector
        assert "window.scrollTo" not in selectors

    def test_fallback_chain(self):
        """Multiple fallback selectors inside evaluate."""
        results = extract_selectors(EVALUATE_FALLBACK_CHAIN)
        selectors = {r.selector for r in results}
        assert '[data-testid="readme-blob"]' in selectors
        assert "article" in selectors
        assert 'div[class*="markdown-body"]' in selectors

    def test_locator_evaluate_all(self):
        """locator.evaluate_all — selector from locator creation."""
        results = extract_selectors(LOCATOR_EVALUATE_ALL)
        assert len(results) >= 1
        assert results[0].selector == "button[aria-label]"
        assert results[0].action_category == "passive"


class TestWaitForSelector:
    """Test wait_for_selector extraction."""

    def test_wait_for_selector(self):
        """wait_for_selector → passive (waiting, not acting)."""
        results = extract_selectors(WAIT_FOR_SELECTOR)
        assert len(results) == 1
        r = results[0]
        assert r.selector == "tr.athing"
        assert r.action_category == "passive"
        assert r.action == "wait_for_selector"


class TestKeyboardPress:
    """Test keyboard.press detection."""

    def test_keyboard_escape_and_go_back(self):
        """Escape + go_back → keyboard is passive, go_back is page-changing."""
        results = extract_selectors(KEYBOARD_PRESS)
        # keyboard.press has no DOM selector to highlight
        # go_back has no DOM selector either
        assert len(results) == 0


class TestInnerTextGetAttribute:
    """Test page.inner_text / page.get_attribute."""

    def test_inner_text_and_get_attribute(self):
        """Direct page.inner_text and page.get_attribute → passive reads."""
        results = extract_selectors(INNER_TEXT_READ)
        assert len(results) == 2
        assert results[0].selector == ".selector"
        assert results[0].action_category == "passive"
        assert results[1].selector == "a.link"
        assert results[1].action_category == "passive"
