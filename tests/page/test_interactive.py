"""Tests for interactive_elements_detection.py

Runs against a real headless Chromium browser via Patchright to verify
that detection, stamping, and cleanup all work correctly.
"""

import pytest
import pytest_asyncio
from patchright.async_api import async_playwright

from scout.page.interactive import (
    InteractiveElement,
    cleanup_markers,
    detect_interactive_elements,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest_asyncio.fixture
async def page():
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(headless=True)
    p = await browser.new_page()
    yield p
    await p.close()
    await browser.close()
    await pw.stop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def by_tag(elements: list[InteractiveElement]) -> dict[str, list[InteractiveElement]]:
    """Group elements by tag name for easier assertions."""
    groups: dict[str, list[InteractiveElement]] = {}
    for el in elements:
        groups.setdefault(el.tag, []).append(el)
    return groups


def find(elements: list[InteractiveElement], **kwargs) -> InteractiveElement | None:
    """Find the first element matching all keyword filters."""
    for el in elements:
        match = True
        for k, v in kwargs.items():
            if getattr(el, k, None) != v:
                match = False
                break
        if match:
            return el
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Layer 1 — Native HTML interactive elements
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.asyncio
async def test_layer1_links(page):
    await page.set_content('<a href="/home">Home</a><a href="/about">About</a>')
    elems = await detect_interactive_elements(page)
    assert len(elems) == 2
    assert all(e.tag == "a" for e in elems)
    assert all("layer1_native" in e.detected_by for e in elems)
    assert elems[0].attributes.get("href") == "/home"
    assert elems[1].text == "About"


@pytest.mark.asyncio
async def test_layer1_buttons(page):
    await page.set_content('<button id="btn1">Click</button>')
    elems = await detect_interactive_elements(page)
    assert len(elems) == 1
    assert elems[0].tag == "button"
    assert elems[0].text == "Click"
    assert elems[0].selector == "#btn1"


@pytest.mark.asyncio
async def test_layer1_form_elements(page):
    await page.set_content("""
        <input type="text" name="q" placeholder="Search">
        <select name="sort"><option>Price</option></select>
        <textarea name="comment">Hello</textarea>
    """)
    elems = await detect_interactive_elements(page)
    tags = {e.tag for e in elems}
    assert "input" in tags
    assert "select" in tags
    assert "textarea" in tags


@pytest.mark.asyncio
async def test_layer1_contenteditable(page):
    await page.set_content('<div contenteditable="true" id="editor">Edit me</div>')
    elems = await detect_interactive_elements(page)
    assert any(e.attributes.get("contenteditable") == "true" for e in elems)


@pytest.mark.asyncio
async def test_layer1_tabindex(page):
    await page.set_content("""
        <div tabindex="0" id="focusable">Tab to me</div>
        <div tabindex="-1" id="not-focusable">Skip me</div>
    """)
    elems = await detect_interactive_elements(page)
    # tabindex="0" → interactive, tabindex="-1" → not interactive
    ids = [e.attributes.get("id") for e in elems]
    assert "focusable" in ids
    assert "not-focusable" not in ids


@pytest.mark.asyncio
async def test_layer1_details_summary(page):
    await page.set_content("""
        <details>
            <summary>More info</summary>
            <p>Hidden content here.</p>
        </details>
    """)
    elems = await detect_interactive_elements(page)
    tags = {e.tag for e in elems}
    assert "details" in tags or "summary" in tags


@pytest.mark.asyncio
async def test_layer1_label(page):
    await page.set_content('<label for="name">Name:</label><input id="name" type="text">')
    elems = await detect_interactive_elements(page)
    tags = {e.tag for e in elems}
    assert "label" in tags
    assert "input" in tags


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Layer 2 — ARIA roles and attributes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.asyncio
async def test_layer2_aria_role(page):
    await page.set_content('<div role="button" id="custom-btn">Press</div>')
    elems = await detect_interactive_elements(page)
    assert len(elems) == 1
    assert elems[0].attributes.get("role") == "button"
    assert "layer2_aria" in elems[0].detected_by


@pytest.mark.asyncio
async def test_layer2_aria_attrs(page):
    await page.set_content("""
        <div aria-expanded="false" id="expander">Menu</div>
        <div aria-haspopup="true" id="popup-trigger">Options</div>
    """)
    elems = await detect_interactive_elements(page)
    ids = {e.attributes.get("id") for e in elems}
    assert "expander" in ids
    assert "popup-trigger" in ids
    assert all("layer2_aria" in e.detected_by for e in elems)


@pytest.mark.asyncio
async def test_layer2_multiple_aria_roles(page):
    await page.set_content("""
        <div role="tab" id="tab1">Tab 1</div>
        <div role="menuitem" id="mi1">Item 1</div>
        <div role="slider" id="sl1" aria-valuenow="50">50%</div>
    """)
    elems = await detect_interactive_elements(page)
    ids = {e.attributes.get("id") for e in elems}
    assert {"tab1", "mi1", "sl1"} <= ids


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Layer 5 — cursor:pointer heuristic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.asyncio
async def test_layer5_cursor_pointer(page):
    await page.set_content("""
        <style>.clickable { cursor: pointer; }</style>
        <div class="clickable" id="click-div">Click me</div>
        <div id="normal-div">Normal text</div>
    """)
    elems = await detect_interactive_elements(page)
    ids = {e.attributes.get("id") for e in elems}
    assert "click-div" in ids
    assert "normal-div" not in ids
    click_el = find(elems, tag="div")
    assert click_el is not None
    assert "layer5_cursor" in click_el.detected_by


@pytest.mark.asyncio
async def test_layer5_does_not_duplicate_native(page):
    """A <button> already found by Layer 1 should NOT be re-detected by Layer 5
    even though buttons typically have cursor:pointer."""
    await page.set_content('<button id="btn">Go</button>')
    elems = await detect_interactive_elements(page)
    assert len(elems) == 1
    assert "layer1_native" in elems[0].detected_by


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Deduplication
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.asyncio
async def test_deduplication_across_layers(page):
    """A <button role="button"> is found by both Layer 1 and Layer 2.
    It should appear once, with both layers in detected_by."""
    await page.set_content('<button role="button" id="dual">Both</button>')
    elems = await detect_interactive_elements(page)
    assert len(elems) == 1
    assert "layer1_native" in elems[0].detected_by
    assert "layer2_aria" in elems[0].detected_by


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Redundant child cleanup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.asyncio
async def test_cursor_pointer_child_inside_link_removed(page):
    """A <div> with cursor:pointer inside an <a> should be suppressed."""
    await page.set_content("""
        <a href="/vote" id="vote-link">
            <div style="cursor:pointer" id="arrow-div">▲</div>
        </a>
    """)
    elems = await detect_interactive_elements(page)
    ids = {e.attributes.get("id") for e in elems}
    assert "vote-link" in ids
    assert "arrow-div" not in ids


@pytest.mark.asyncio
async def test_span_inside_link_removed(page):
    """A <span> inheriting cursor:pointer from parent <a> should be suppressed."""
    await page.set_content("""
        <a href="/pricing" id="pricing-link">
            <span id="pricing-text">Pricing</span>
        </a>
    """)
    elems = await detect_interactive_elements(page)
    ids = {e.attributes.get("id") for e in elems}
    assert "pricing-link" in ids
    assert "pricing-text" not in ids


@pytest.mark.asyncio
async def test_nested_div_span_inside_link_removed(page):
    """Deeply nested non-interactive children inside a link are suppressed."""
    await page.set_content("""
        <a href="#History" id="toc-link">
            <div id="inner-div">
                <span id="inner-span">History</span>
            </div>
        </a>
    """)
    elems = await detect_interactive_elements(page)
    ids = {e.attributes.get("id") for e in elems}
    assert "toc-link" in ids
    assert "inner-div" not in ids
    assert "inner-span" not in ids


@pytest.mark.asyncio
async def test_button_inside_link_kept(page):
    """A <button> inside an <a> IS independently interactive — keep it."""
    await page.set_content("""
        <a href="/card" id="card-link">
            <span>Product</span>
            <button id="add-btn">Add to Cart</button>
        </a>
    """)
    elems = await detect_interactive_elements(page)
    ids = {e.attributes.get("id") for e in elems}
    assert "card-link" in ids
    assert "add-btn" in ids


@pytest.mark.asyncio
async def test_input_inside_interactive_kept(page):
    """Form elements inside an interactive parent are independently actionable."""
    await page.set_content("""
        <div role="combobox" id="combo" aria-expanded="false">
            <input type="text" id="combo-input" placeholder="Search...">
        </div>
    """)
    elems = await detect_interactive_elements(page)
    ids = {e.attributes.get("id") for e in elems}
    assert "combo" in ids
    assert "combo-input" in ids


@pytest.mark.asyncio
async def test_role_button_inside_link_kept(page):
    """An element with role='button' inside a link is independently interactive."""
    await page.set_content("""
        <a href="/item" id="item-link">
            <span>Item</span>
            <div role="button" id="action-btn">Action</div>
        </a>
    """)
    elems = await detect_interactive_elements(page)
    ids = {e.attributes.get("id") for e in elems}
    assert "item-link" in ids
    assert "action-btn" in ids


@pytest.mark.asyncio
async def test_cleanup_removes_data_iid_from_dom(page):
    """Suppressed children should have their data-iid removed from the DOM."""
    await page.set_content("""
        <a href="/link" id="parent-link">
            <div style="cursor:pointer" id="child-div">Text</div>
        </a>
    """)
    await detect_interactive_elements(page)
    html = await page.content()
    # parent-link should have data-iid
    assert 'id="parent-link"' in html
    # child-div should NOT have data-iid (it was removed)
    # Find the child-div tag and check it has no data-iid
    import re
    child_match = re.search(r'<div[^>]*id="child-div"[^>]*>', html)
    assert child_match is not None
    assert 'data-iid' not in child_match.group(0)


@pytest.mark.asyncio
async def test_non_nested_siblings_both_kept(page):
    """Sibling interactive elements are NOT affected by cleanup."""
    await page.set_content("""
        <a href="/a" id="link-a">A</a>
        <a href="/b" id="link-b">B</a>
        <div style="cursor:pointer" id="click-div">Click</div>
    """)
    elems = await detect_interactive_elements(page)
    ids = {e.attributes.get("id") for e in elems}
    assert "link-a" in ids
    assert "link-b" in ids
    assert "click-div" in ids


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Disabled elements
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.asyncio
async def test_disabled_native_excluded(page):
    await page.set_content("""
        <button disabled>Can't click</button>
        <input type="text" disabled>
        <button>Can click</button>
    """)
    elems = await detect_interactive_elements(page)
    assert len(elems) == 1
    assert elems[0].text == "Can click"


@pytest.mark.asyncio
async def test_aria_disabled_excluded(page):
    await page.set_content('<div role="button" aria-disabled="true">Nope</div>')
    elems = await detect_interactive_elements(page)
    assert len(elems) == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Hidden elements
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.asyncio
async def test_hidden_display_none_stamped(page):
    await page.set_content("""
        <div id="visible">Visible</div>
        <div id="hidden-root" style="display:none">
            <p id="child">Hidden child</p>
        </div>
    """)
    await detect_interactive_elements(page)
    html = await page.content()

    # The root of the hidden subtree should be stamped
    assert 'data-hidden="true"' in html
    # The child should NOT be stamped (parent is already the hide-root)
    assert html.count('data-hidden=') == 1


@pytest.mark.asyncio
async def test_hidden_visibility_hidden_stamped(page):
    await page.set_content("""
        <div style="visibility:hidden" id="invis">
            <span>Can't see me</span>
        </div>
        <div id="vis">Can see me</div>
    """)
    await detect_interactive_elements(page)
    html = await page.content()
    assert 'id="invis"' in html and 'data-hidden="true"' in html


@pytest.mark.asyncio
async def test_hidden_interactive_not_detected(page):
    """An interactive element inside a hidden container should not be detected."""
    await page.set_content("""
        <div style="display:none">
            <button id="ghost">Ghost button</button>
        </div>
        <button id="real">Real button</button>
    """)
    elems = await detect_interactive_elements(page)
    ids = [e.attributes.get("id") for e in elems]
    assert "ghost" not in ids
    assert "real" in ids


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DOM stamping (data-iid)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.asyncio
async def test_dom_has_data_iid_after_detection(page):
    await page.set_content("""
        <a href="/a">A</a>
        <button id="b">B</button>
    """)
    elems = await detect_interactive_elements(page)
    html = await page.content()

    for el in elems:
        assert f'data-iid="{el.iid}"' in html


@pytest.mark.asyncio
async def test_iids_are_sequential(page):
    await page.set_content("""
        <a href="/1">1</a><a href="/2">2</a><a href="/3">3</a>
    """)
    elems = await detect_interactive_elements(page)
    iids = [e.iid for e in elems]
    assert iids == [1, 2, 3]


@pytest.mark.asyncio
async def test_previous_markers_cleared_on_rerun(page):
    """Running detection twice should produce clean, fresh markers."""
    await page.set_content('<button id="b">Go</button>')

    elems1 = await detect_interactive_elements(page)
    elems2 = await detect_interactive_elements(page)

    # Should get iid=1 both times (reset on each run)
    assert elems1[0].iid == 1
    assert elems2[0].iid == 1

    html = await page.content()
    assert html.count("data-iid=") == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Selector generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.asyncio
async def test_selector_prefers_id(page):
    await page.set_content('<button id="main-cta">Go</button>')
    elems = await detect_interactive_elements(page)
    assert elems[0].selector == "#main-cta"


@pytest.mark.asyncio
async def test_selector_uses_data_testid(page):
    await page.set_content('<button data-testid="submit-btn">Submit</button>')
    elems = await detect_interactive_elements(page)
    assert "data-testid" in elems[0].selector


@pytest.mark.asyncio
async def test_selector_uses_name(page):
    await page.set_content('<input type="text" name="email">')
    elems = await detect_interactive_elements(page)
    assert 'name=' in elems[0].selector or 'type=' in elems[0].selector


@pytest.mark.asyncio
async def test_selector_fallback_positional(page):
    """Elements with no distinctive attributes get a positional selector."""
    await page.set_content("""
        <div><button>A</button><button>B</button></div>
    """)
    elems = await detect_interactive_elements(page)
    # Both buttons exist; selectors should differ
    assert elems[0].selector != elems[1].selector


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Bounding box
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.asyncio
async def test_bounding_box_present(page):
    await page.set_content('<button style="width:200px;height:50px">Big</button>')
    elems = await detect_interactive_elements(page)
    bb = elems[0].bounding_box
    assert bb["width"] == 200
    assert bb["height"] == 50
    assert "x" in bb and "y" in bb


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  cleanup_markers()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_cleanup_removes_iid():
    html = '<a href="/" data-iid="1">Home</a>'
    assert cleanup_markers(html) == '<a href="/">Home</a>'


def test_cleanup_removes_hidden():
    html = '<div data-hidden="true"><p>gone</p></div>'
    assert cleanup_markers(html) == '<div><p>gone</p></div>'


def test_cleanup_removes_both():
    html = '<button data-iid="5" data-hidden="true">X</button>'
    clean = cleanup_markers(html)
    assert "data-iid" not in clean
    assert "data-hidden" not in clean
    assert "<button>X</button>" == clean


def test_cleanup_preserves_other_data_attrs():
    html = '<div data-testid="foo" data-iid="3">bar</div>'
    clean = cleanup_markers(html)
    assert 'data-testid="foo"' in clean
    assert "data-iid" not in clean


def test_cleanup_no_markers():
    html = '<a href="/">Home</a>'
    assert cleanup_markers(html) == html


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Edge cases
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.asyncio
async def test_empty_page(page):
    await page.set_content("<html><body></body></html>")
    elems = await detect_interactive_elements(page)
    assert elems == []


@pytest.mark.asyncio
async def test_text_only_page(page):
    await page.set_content("<p>Just some text, nothing interactive.</p>")
    elems = await detect_interactive_elements(page)
    assert elems == []


@pytest.mark.asyncio
async def test_nested_interactive_elements(page):
    """Links wrapping buttons (unusual but exists in the wild)."""
    await page.set_content("""
        <a href="/card" id="card-link">
            <span>Product Card</span>
            <button id="add-to-cart">Add to Cart</button>
        </a>
    """)
    elems = await detect_interactive_elements(page)
    tags = {e.tag for e in elems}
    assert "a" in tags
    assert "button" in tags


@pytest.mark.asyncio
async def test_zero_size_element_excluded(page):
    """Elements with zero dimensions are not interactive."""
    await page.set_content("""
        <a href="/hidden" style="display:inline-block;width:0;height:0;overflow:hidden">Hidden link</a>
        <a href="/visible">Visible link</a>
    """)
    elems = await detect_interactive_elements(page)
    assert len(elems) == 1
    assert elems[0].attributes.get("href") == "/visible"


@pytest.mark.asyncio
async def test_text_truncation(page):
    """Very long text content is truncated to 500 characters."""
    long_text = "A" * 1000
    await page.set_content(f'<button id="long">{long_text}</button>')
    elems = await detect_interactive_elements(page)
    assert len(elems[0].text) == 500


@pytest.mark.asyncio
async def test_special_chars_in_attributes(page):
    """Attributes with special characters should not break detection."""
    await page.set_content("""
        <a href="/search?q=hello&sort=asc" id="special-link">Search</a>
    """)
    elems = await detect_interactive_elements(page)
    assert len(elems) == 1
    assert "&" in elems[0].attributes.get("href", "")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Real-world messy HTML
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REAL_WORLD_HTML = """<!DOCTYPE html>
<html lang="en">
<head><title>E-Commerce</title></head>
<body>
  <header>
    <nav>
      <a href="/">Home</a>
      <a href="/products">Products</a>
      <a href="/about">About</a>
    </nav>
    <div class="search" style="cursor:pointer" id="search-icon">🔍</div>
    <form action="/search" method="get">
      <input type="text" name="q" placeholder="Search products...">
      <button type="submit">Go</button>
    </form>
  </header>

  <main>
    <!-- Dropdown with ARIA -->
    <div role="combobox" aria-expanded="false" id="category-picker">
      Category
      <div role="listbox" style="display:none" id="dropdown-list">
        <div role="option" id="opt-electronics">Electronics</div>
        <div role="option" id="opt-clothing">Clothing</div>
      </div>
    </div>

    <!-- Product cards with nested links and buttons -->
    <div class="product-grid">
      <div class="card" style="cursor:pointer" id="card-1">
        <a href="/product/1">
          <img src="phone.jpg" alt="Phone">
          <h3>Smartphone Pro</h3>
          <span class="price">$999</span>
        </a>
        <button id="cart-1" aria-label="Add Smartphone Pro to cart">Add to Cart</button>
      </div>
      <div class="card" style="cursor:pointer" id="card-2">
        <a href="/product/2">
          <img src="laptop.jpg" alt="Laptop">
          <h3>Business Laptop</h3>
          <span class="price">$1499</span>
        </a>
        <button id="cart-2" disabled>Out of Stock</button>
      </div>
    </div>

    <!-- Pagination -->
    <nav aria-label="Pagination">
      <span class="current" tabindex="0">1</span>
      <a href="/products?page=2">2</a>
      <a href="/products?page=3">3</a>
      <a href="/products?page=2" aria-label="Next page">Next →</a>
    </nav>

    <!-- Hidden section -->
    <div class="modal" style="display:none" id="login-modal">
      <form>
        <input type="email" name="email" placeholder="Email">
        <input type="password" name="pass" placeholder="Password">
        <button type="submit">Login</button>
      </form>
    </div>

    <!-- Cookie banner with custom roles -->
    <div role="dialog" aria-label="Cookie consent" id="cookie-banner">
      <p>We use cookies.</p>
      <button id="accept-cookies">Accept</button>
      <button id="reject-cookies">Reject</button>
      <a href="/privacy" id="privacy-link">Privacy Policy</a>
    </div>

    <!-- Tab interface -->
    <div role="tablist" id="tabs">
      <div role="tab" id="tab-desc" aria-selected="true" tabindex="0">Description</div>
      <div role="tab" id="tab-reviews" aria-selected="false" tabindex="0">Reviews</div>
    </div>
  </main>

  <footer>
    <a href="/terms">Terms</a>
    <a href="/contact">Contact</a>
  </footer>
</body>
</html>"""


@pytest.mark.asyncio
async def test_real_world_detects_all_interactive(page):
    await page.set_content(REAL_WORLD_HTML)
    elems = await detect_interactive_elements(page)

    ids = {e.attributes.get("id") for e in elems if "id" in e.attributes}
    tags = {e.tag for e in elems}

    # Navigation links detected
    nav_links = [e for e in elems if e.tag == "a" and e.attributes.get("href", "").startswith("/")]
    assert len(nav_links) >= 7  # home, products, about, product/1, product/2, page 2, page 3, next, terms, contact, privacy

    # Form elements detected
    assert "input" in tags
    assert "button" in tags

    # ARIA combobox detected
    assert "category-picker" in ids

    # cursor:pointer cards detected
    assert "card-1" in ids or "card-2" in ids

    # Tabs detected
    assert "tab-desc" in ids
    assert "tab-reviews" in ids

    # Cookie banner buttons detected
    assert "accept-cookies" in ids
    assert "reject-cookies" in ids


@pytest.mark.asyncio
async def test_real_world_disabled_excluded(page):
    await page.set_content(REAL_WORLD_HTML)
    elems = await detect_interactive_elements(page)
    ids = {e.attributes.get("id") for e in elems}
    # cart-2 is disabled → should not appear
    assert "cart-2" not in ids
    # cart-1 is enabled → should appear
    assert "cart-1" in ids


@pytest.mark.asyncio
async def test_real_world_hidden_modal_stamped(page):
    await page.set_content(REAL_WORLD_HTML)
    await detect_interactive_elements(page)
    html = await page.content()
    # The hidden modal should be stamped
    assert 'id="login-modal"' in html
    assert 'data-hidden="true"' in html


@pytest.mark.asyncio
async def test_real_world_hidden_interactive_not_detected(page):
    await page.set_content(REAL_WORLD_HTML)
    elems = await detect_interactive_elements(page)
    # Elements inside the hidden modal should NOT be detected
    texts = {e.text for e in elems}
    assert "Login" not in texts


@pytest.mark.asyncio
async def test_real_world_dom_stamping_consistent(page):
    """Every returned element's iid matches a data-iid in the DOM."""
    await page.set_content(REAL_WORLD_HTML)
    elems = await detect_interactive_elements(page)
    html = await page.content()

    for el in elems:
        assert f'data-iid="{el.iid}"' in html

    # cleanup should produce valid HTML without markers
    clean = cleanup_markers(html)
    assert "data-iid=" not in clean
    assert "data-hidden=" not in clean


@pytest.mark.asyncio
async def test_real_world_element_count_reasonable(page):
    """Sanity check: the real-world page should have a reasonable number of elements."""
    await page.set_content(REAL_WORLD_HTML)
    elems = await detect_interactive_elements(page)
    # At least: 3 nav links + search input + submit button + combobox +
    # 2 product links + cart-1 + pagination span + 3 pagination links +
    # 2 cookie buttons + privacy link + 2 tabs + 2 footer links + cards
    assert len(elems) >= 15
    # But not an absurd number (would indicate false positives)
    assert len(elems) < 50
