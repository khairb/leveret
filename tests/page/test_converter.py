"""Comprehensive tests for the HTML-to-Text converter.

All tests save their output to test_outputs/ for manual inspection.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from scout.page.converter import InteractiveElement, html_to_text

# ---------------------------------------------------------------------------
# Output directory setup
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent / "test_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def _save(name: str, result: str, *, html: str = "") -> str:
    """Save test output to a file and return the result unchanged."""
    path = OUTPUT_DIR / f"{name}.txt"
    content = result
    if html:
        content = (
            "=== INPUT HTML ===\n"
            + html.strip()
            + "\n\n=== OUTPUT ===\n"
            + result
        )
    path.write_text(content, encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# Helper to build InteractiveElement quickly
# ---------------------------------------------------------------------------

def ie(iid: int, tag: str, attrs: dict | None = None, text: str = "", selector: str = "") -> InteractiveElement:
    return InteractiveElement(iid=iid, tag=tag, attributes=attrs or {}, text=text, selector=selector)


# ===========================================================================
# Test 1: The exact example from the spec
# ===========================================================================

class TestSpecExample:
    HTML = """
    <div class="header">
      <nav>
        <a href="/" data-iid="1">Home</a>
        <a href="/products" data-iid="2">Products</a>
        <a href="/about" data-iid="3">About</a>
      </nav>
    </div>
    <div class="search-bar">
      <input type="text" name="q" placeholder="Search products..." data-iid="4">
      <button type="submit" data-iid="5">Search</button>
    </div>
    <div class="product-list">
      <div class="product-card">
        <a href="/product/123" data-iid="6">
          <img src="headphones.jpg" alt="Sony WH-1000XM5">
          <span class="title">Sony WH-1000XM5</span>
          <span class="price">$348.00</span>
          <span class="rating">4.5 stars</span>
        </a>
      </div>
      <div class="product-card">
        <a href="/product/456" data-iid="7">
          <img src="earbuds.jpg" alt="AirPods Pro">
          <span class="title">AirPods Pro</span>
          <span class="price">$249.00</span>
          <span class="rating">4.7 stars</span>
        </a>
      </div>
    </div>
    <div class="pagination">
      <span class="current">1</span>
      <a href="/products?page=2" data-iid="8">2</a>
      <a href="/products?page=3" data-iid="9">3</a>
      <a href="/products?page=2" data-iid="10">Next</a>
    </div>
    """

    ELEMENTS = [
        ie(1, "a", {"href": "/"}),
        ie(2, "a", {"href": "/products"}),
        ie(3, "a", {"href": "/about"}),
        ie(4, "input", {"type": "text", "name": "q", "placeholder": "Search products..."}),
        ie(5, "button", {"type": "submit"}),
        ie(6, "a", {"href": "/product/123"}),
        ie(7, "a", {"href": "/product/456"}),
        ie(8, "a", {"href": "/products?page=2"}),
        ie(9, "a", {"href": "/products?page=3"}),
        ie(10, "a", {"href": "/products?page=2"}),
    ]

    def test_spec_example(self):
        result = _save("01_spec_example", html_to_text(self.HTML, self.ELEMENTS), html=self.HTML)

        assert '<a href="/">Home</a>' in result
        assert '<a href="/products">Products</a>' in result
        assert '<a href="/about">About</a>' in result
        assert '<input name="q" placeholder="Search products..." type="text">' in result
        assert '<button type="submit">Search</button>' in result

        assert '<a href="/product/123">' in result
        assert "Sony WH-1000XM5" in result
        assert "$348.00" in result
        assert "4.5 stars" in result
        assert '<a href="/product/456">' in result
        assert "AirPods Pro" in result
        assert "$249.00" in result

        assert "1" in result
        assert '<a href="/products?page=2">2</a>' in result
        assert '<a href="/products?page=3">3</a>' in result
        assert '<a href="/products?page=2">Next</a>' in result

        assert "<div" not in result
        assert "<span" not in result
        assert "<nav" not in result
        assert "class=" not in result

    def test_spec_example_image_alt_text(self):
        result = html_to_text(self.HTML, self.ELEMENTS)
        assert "Sony WH-1000XM5" in result
        assert "AirPods Pro" in result


# ===========================================================================
# Test 2: Hidden elements
# ===========================================================================

class TestHiddenElements:
    def test_data_hidden_attribute(self):
        html = """
        <div>
            <p>Visible text</p>
            <p data-hidden="true">This should be hidden</p>
        </div>
        """
        result = _save("02_hidden_data_attr", html_to_text(html), html=html)
        assert "Visible text" in result
        assert "hidden" not in result.lower() or "hidden" not in result

    def test_aria_hidden(self):
        html = """
        <div>
            <p>Visible</p>
            <div aria-hidden="true">
                <p>Screen reader hidden content</p>
                <a href="/secret" data-iid="1">Hidden link</a>
            </div>
        </div>
        """
        result = _save("02_hidden_aria", html_to_text(html, [ie(1, "a", {"href": "/secret"})]), html=html)
        assert "Visible" in result
        assert "Screen reader" not in result
        assert "Hidden link" not in result
        assert "/secret" not in result

    def test_inline_display_none(self):
        html = """
        <div>
            <p>Shown</p>
            <p style="display: none;">Hidden by display</p>
            <p style="display:none">Also hidden</p>
        </div>
        """
        result = _save("02_hidden_display_none", html_to_text(html), html=html)
        assert "Shown" in result
        assert "Hidden by display" not in result
        assert "Also hidden" not in result

    def test_inline_visibility_hidden(self):
        html = """
        <div>
            <p>Shown</p>
            <p style="visibility: hidden;">Invisible</p>
        </div>
        """
        result = _save("02_hidden_visibility", html_to_text(html), html=html)
        assert "Shown" in result
        assert "Invisible" not in result

    def test_hidden_subtree_excluded_entirely(self):
        html = """
        <div>
            <p>Before</p>
            <div data-hidden="true">
                <div>
                    <p>Deep nested hidden</p>
                    <button data-iid="1">Hidden button</button>
                </div>
            </div>
            <p>After</p>
        </div>
        """
        result = _save("02_hidden_subtree", html_to_text(html, [ie(1, "button")]), html=html)
        assert "Before" in result
        assert "After" in result
        assert "Deep nested" not in result
        assert "Hidden button" not in result


# ===========================================================================
# Test 3: Nested interactive elements
# ===========================================================================

class TestNestedInteractive:
    def test_link_containing_button(self):
        html = """
        <div>
            <a href="/card" data-iid="1">
                <div class="card-body">
                    <h3>Product Title</h3>
                    <p>Some description</p>
                    <button data-iid="2" type="button">Add to Cart</button>
                </div>
            </a>
        </div>
        """
        result = _save("03_nested_link_button", html_to_text(html, [
            ie(1, "a", {"href": "/card"}),
            ie(2, "button", {"type": "button"}),
        ]), html=html)
        assert '<a href="/card">' in result
        assert "</a>" in result
        assert '<button type="button">Add to Cart</button>' in result
        assert "Product Title" in result
        assert "Some description" in result

    def test_deeply_nested_interactive(self):
        html = """
        <form action="/submit" data-iid="1">
            <a href="/link" data-iid="2">
                <span>Click here or</span>
                <button data-iid="3" type="submit">Submit</button>
            </a>
        </form>
        """
        result = _save("03_nested_three_levels", html_to_text(html, [
            ie(1, "form", {"action": "/submit"}),
            ie(2, "a", {"href": "/link"}),
            ie(3, "button", {"type": "submit"}),
        ]), html=html)
        assert '<form action="/submit">' in result
        assert "</form>" in result
        assert '<a href="/link">' in result
        assert "</a>" in result
        assert '<button type="submit">Submit</button>' in result
        assert "Click here or" in result


# ===========================================================================
# Test 4: Malformed HTML
# ===========================================================================

class TestMalformedHTML:
    def test_unclosed_tags(self):
        html = """
        <div>
            <p>Paragraph without closing
            <p>Another paragraph
            <a href="/link" data-iid="1">A link</a>
        </div>
        """
        result = _save("04_malformed_unclosed", html_to_text(html, [ie(1, "a", {"href": "/link"})]), html=html)
        assert "Paragraph without closing" in result
        assert "Another paragraph" in result
        assert '<a href="/link">A link</a>' in result

    def test_invalid_nesting(self):
        html = "<p>Start <div>Div inside p</div> end</p>"
        result = _save("04_malformed_nesting", html_to_text(html), html=html)
        assert "Start" in result
        assert "Div inside p" in result

    def test_bare_text(self):
        html = "Just some bare text without any tags"
        result = _save("04_malformed_bare_text", html_to_text(html), html=html)
        assert "Just some bare text without any tags" in result

    def test_mixed_encoding_entities(self):
        html = "<div><p>Caf&eacute; &amp; Cr&egrave;me &mdash; $19.99</p></div>"
        result = _save("04_malformed_entities", html_to_text(html), html=html)
        assert "Café" in result
        assert "&" in result
        assert "Crème" in result


# ===========================================================================
# Test 5: Image handling
# ===========================================================================

class TestImages:
    def test_image_with_alt(self):
        html = '<div><img src="photo.jpg" alt="A beautiful sunset"></div>'
        result = _save("05_img_with_alt", html_to_text(html), html=html)
        assert "A beautiful sunset" in result
        assert "<img" not in result

    def test_image_without_alt(self):
        html = '<div><img src="photo.jpg"><p>After image</p></div>'
        result = _save("05_img_no_alt", html_to_text(html), html=html)
        assert "photo.jpg" not in result
        assert "After image" in result

    def test_image_empty_alt(self):
        html = '<div><img src="photo.jpg" alt=""><p>Text</p></div>'
        result = html_to_text(html)
        assert "photo.jpg" not in result
        assert "Text" in result

    def test_image_inside_interactive_link(self):
        html = """
        <a href="/product" data-iid="1">
            <img src="product.jpg" alt="Product Photo">
            <span>Product Name</span>
        </a>
        """
        result = _save("05_img_in_link", html_to_text(html, [ie(1, "a", {"href": "/product"})]), html=html)
        assert "Product Photo" in result
        assert "Product Name" in result
        assert '<a href="/product">' in result


# ===========================================================================
# Test 6: Attribute filtering
# ===========================================================================

class TestAttributeFiltering:
    def test_class_and_style_stripped(self):
        html = """
        <a href="/link" class="btn btn-primary" style="color: red;"
           data-iid="1" id="main-link" data-testid="nav-link">
            Click me
        </a>
        """
        result = _save("06_attr_filtering", html_to_text(html, [ie(1, "a", {"href": "/link"})]), html=html)
        assert 'href="/link"' in result
        assert 'id="main-link"' in result
        assert 'data-testid="nav-link"' in result
        assert "class=" not in result
        assert "style=" not in result
        assert "Click me" in result

    def test_only_allowed_attributes_kept(self):
        html = """
        <button data-iid="1" type="submit" name="go"
                class="fancy" style="margin: 10px"
                data-analytics="click-event" onclick="doStuff()">
            Go
        </button>
        """
        result = _save("06_attr_allowed_only", html_to_text(html, [ie(1, "button", {"type": "submit", "name": "go"})]), html=html)
        assert 'type="submit"' in result
        assert 'name="go"' in result
        assert "class=" not in result
        assert "style=" not in result
        assert "data-analytics" not in result
        assert "onclick" not in result

    def test_attributes_sorted_alphabetically(self):
        html = '<input data-iid="1" type="text" name="query" placeholder="Search..." id="search-box">'
        result = _save("06_attr_sorted", html_to_text(html, [
            ie(1, "input", {"type": "text", "name": "query", "placeholder": "Search...", "id": "search-box"})
        ]), html=html)
        tag_match = result[result.index("<input"):result.index(">") + 1]
        attr_positions = {}
        for attr in ["id", "name", "placeholder", "type"]:
            pos = tag_match.find(attr)
            if pos != -1:
                attr_positions[attr] = pos
        attrs_in_order = sorted(attr_positions, key=lambda a: attr_positions[a])
        assert attrs_in_order == sorted(attrs_in_order), f"Attributes not sorted: {attrs_in_order}"


# ===========================================================================
# Test 7: Empty and minimal inputs
# ===========================================================================

class TestEdgeCases:
    def test_empty_string(self):
        assert html_to_text("") == ""

    def test_whitespace_only(self):
        assert html_to_text("   \n\t  ") == ""

    def test_none_interactive_elements(self):
        html = "<div><p>Hello world</p></div>"
        result = html_to_text(html, None)
        assert "Hello world" in result

    def test_empty_interactive_elements_list(self):
        html = "<div><p>Hello world</p></div>"
        result = html_to_text(html, [])
        assert "Hello world" in result

    def test_no_matching_iids(self):
        html = "<div><p>Just text</p></div>"
        result = html_to_text(html, [ie(99, "a", {"href": "/nowhere"})])
        assert "Just text" in result
        assert "<a" not in result

    def test_script_and_style_excluded(self):
        html = """
        <div>
            <style>.hidden { display: none; }</style>
            <script>alert('xss')</script>
            <p>Real content</p>
            <script type="application/json">{"data": true}</script>
        </div>
        """
        result = _save("07_excluded_script_style", html_to_text(html), html=html)
        assert "Real content" in result
        assert "alert" not in result
        assert "display" not in result
        assert "application/json" not in result

    def test_noscript_excluded(self):
        html = """
        <div>
            <p>Main content</p>
            <noscript>Please enable JavaScript</noscript>
        </div>
        """
        result = html_to_text(html)
        assert "Main content" in result
        assert "JavaScript" not in result


# ===========================================================================
# Test 8: Whitespace normalisation
# ===========================================================================

class TestWhitespace:
    def test_collapse_multiple_spaces(self):
        html = "<div><p>Too    many     spaces</p></div>"
        result = html_to_text(html)
        assert "Too many spaces" in result

    def test_no_excessive_blank_lines(self):
        html = """
        <div>
            <p>First</p>
            <p>Second</p>
            <p>Third</p>
        </div>
        """
        result = _save("08_whitespace_no_excess", html_to_text(html), html=html)
        assert "\n\n\n" not in result

    def test_block_elements_separated(self):
        html = """
        <div>
            <h1>Title</h1>
            <p>Paragraph</p>
        </div>
        """
        result = _save("08_whitespace_blocks", html_to_text(html), html=html)
        assert "Title" in result
        assert "Paragraph" in result
        lines = [l for l in result.split("\n") if l.strip()]
        assert len(lines) >= 2


# ===========================================================================
# Test 9: Determinism
# ===========================================================================

class TestDeterminism:
    def test_identical_output_on_repeated_calls(self):
        html = """
        <div>
            <nav>
                <a href="/a" data-iid="1">A</a>
                <a href="/b" data-iid="2">B</a>
                <a href="/c" data-iid="3">C</a>
            </nav>
            <main>
                <h1>Title</h1>
                <p>Content with <strong>bold</strong> and <em>italic</em></p>
                <button data-iid="4" type="button">Click</button>
            </main>
        </div>
        """
        elements = [
            ie(1, "a", {"href": "/a"}),
            ie(2, "a", {"href": "/b"}),
            ie(3, "a", {"href": "/c"}),
            ie(4, "button", {"type": "button"}),
        ]
        results = [html_to_text(html, elements) for _ in range(20)]
        _save("09_determinism", results[0], html=html)
        assert all(r == results[0] for r in results), "Output is not deterministic"


# ===========================================================================
# Test 10: Performance — large synthetic DOM
# ===========================================================================

class TestPerformance:
    def test_large_page_under_one_second(self):
        """Generate a page with 50,000+ nodes and verify conversion < 1s."""
        cards = []
        elements = []
        for i in range(5000):
            iid = i + 1
            cards.append(f"""
            <div class="card">
                <a href="/product/{iid}" data-iid="{iid}">
                    <img src="img{iid}.jpg" alt="Product {iid}">
                    <h3 class="title">Product {iid}</h3>
                    <span class="price">${iid}.99</span>
                    <span class="rating">{(iid % 5) + 1} stars</span>
                    <span class="desc">Description for product number {iid}</span>
                </a>
            </div>
            """)
            elements.append(ie(iid, "a", {"href": f"/product/{iid}"}))

        html = f"""
        <html><body>
        <div class="product-list">
            {''.join(cards)}
        </div>
        </body></html>
        """

        start = time.perf_counter()
        result = html_to_text(html, elements)
        elapsed = time.perf_counter() - start

        # Save first and last 2000 chars for inspection
        preview = result[:2000] + "\n\n... (truncated) ...\n\n" + result[-2000:]
        _save("10_performance_50k_nodes", f"Elapsed: {elapsed:.3f}s\nHTML size: {len(html):,} chars\nOutput size: {len(result):,} chars\n\n{preview}")

        assert elapsed < 1.0, f"Took {elapsed:.3f}s — too slow"
        assert "Product 1" in result
        assert "Product 5000" in result
        assert '<a href="/product/1">' in result
        assert '<a href="/product/5000">' in result


# ===========================================================================
# Test 11: Real-world-like HTML (complex structure)
# ===========================================================================

class TestRealWorldLike:
    def test_complex_ecommerce_page(self):
        html = """
        <html>
        <head><title>Shop</title></head>
        <body>
        <header>
            <div class="logo">ShopCo</div>
            <nav class="main-nav">
                <a href="/" data-iid="1">Home</a>
                <a href="/sale" data-iid="2">Sale</a>
                <a href="/categories" data-iid="3">Categories</a>
            </nav>
            <div class="search">
                <input type="search" name="q" placeholder="Search..." data-iid="4">
                <button type="submit" data-iid="5">Go</button>
            </div>
            <div class="user-actions">
                <a href="/cart" data-iid="6">Cart (3)</a>
                <a href="/account" data-iid="7">My Account</a>
            </div>
        </header>

        <aside class="filters">
            <h3>Filters</h3>
            <div class="filter-group">
                <label>Brand</label>
                <select name="brand" data-iid="8">
                    <option value="">All</option>
                    <option value="sony">Sony</option>
                    <option value="apple">Apple</option>
                </select>
            </div>
            <div class="filter-group">
                <label>Price Range</label>
                <input type="range" name="price" data-iid="9" value="500">
            </div>
            <button type="button" data-iid="10">Apply Filters</button>
        </aside>

        <main>
            <h1>Electronics — 42 results</h1>
            <div class="sort-bar">
                <span>Sort by:</span>
                <select name="sort" data-iid="11">
                    <option value="relevance">Relevance</option>
                    <option value="price-asc">Price: Low to High</option>
                </select>
            </div>

            <div class="product-grid">
                <div class="product" data-hidden="false">
                    <a href="/p/1" data-iid="12">
                        <img src="p1.jpg" alt="Wireless Headphones">
                        <div class="info">
                            <h4>Wireless Headphones</h4>
                            <div class="price">$99.99</div>
                            <div class="rating">★★★★☆ (128)</div>
                        </div>
                    </a>
                    <button data-iid="13" type="button" aria-label="Add to cart">Add to Cart</button>
                </div>
                <div class="product">
                    <a href="/p/2" data-iid="14">
                        <img src="p2.jpg" alt="Bluetooth Speaker">
                        <div class="info">
                            <h4>Bluetooth Speaker</h4>
                            <div class="price">$49.99</div>
                            <div class="rating">★★★★★ (256)</div>
                        </div>
                    </a>
                    <button data-iid="15" type="button" aria-label="Add to cart">Add to Cart</button>
                </div>
            </div>

            <!-- Hidden promotional popup -->
            <div class="popup-overlay" style="display: none;">
                <div class="popup">
                    <h2>Special Offer!</h2>
                    <p>Get 20% off with code SAVE20</p>
                    <button data-iid="99">Close</button>
                </div>
            </div>
        </main>

        <nav class="pagination">
            <span class="current">1</span>
            <a href="/electronics?page=2" data-iid="16">2</a>
            <a href="/electronics?page=3" data-iid="17">3</a>
            <a href="/electronics?page=2" data-iid="18">Next →</a>
        </nav>

        <footer>
            <p>© 2024 ShopCo. All rights reserved.</p>
            <a href="/privacy" data-iid="19">Privacy Policy</a>
            <a href="/terms" data-iid="20">Terms of Service</a>
        </footer>

        <script>console.log("tracking");</script>
        <style>.hidden{display:none}</style>
        </body>
        </html>
        """
        elements = [
            ie(1, "a", {"href": "/"}),
            ie(2, "a", {"href": "/sale"}),
            ie(3, "a", {"href": "/categories"}),
            ie(4, "input", {"type": "search", "name": "q", "placeholder": "Search..."}),
            ie(5, "button", {"type": "submit"}),
            ie(6, "a", {"href": "/cart"}),
            ie(7, "a", {"href": "/account"}),
            ie(8, "select", {"name": "brand"}),
            ie(9, "input", {"type": "range", "name": "price", "value": "500"}),
            ie(10, "button", {"type": "button"}),
            ie(11, "select", {"name": "sort"}),
            ie(12, "a", {"href": "/p/1"}),
            ie(13, "button", {"type": "button", "aria-label": "Add to cart"}),
            ie(14, "a", {"href": "/p/2"}),
            ie(15, "button", {"type": "button", "aria-label": "Add to cart"}),
            ie(16, "a", {"href": "/electronics?page=2"}),
            ie(17, "a", {"href": "/electronics?page=3"}),
            ie(18, "a", {"href": "/electronics?page=2"}),
            ie(19, "a", {"href": "/privacy"}),
            ie(20, "a", {"href": "/terms"}),
            ie(99, "button"),
        ]

        result = _save("11_ecommerce_full_page", html_to_text(html, elements), html=html)

        assert "ShopCo" in result
        assert "Electronics" in result
        assert "42 results" in result
        assert "Wireless Headphones" in result
        assert "$99.99" in result
        assert "Bluetooth Speaker" in result

        assert '<a href="/">Home</a>' in result
        assert '<a href="/cart">Cart (3)</a>' in result
        assert '<button type="submit">Go</button>' in result
        assert '<a href="/p/1">' in result
        assert '<a href="/privacy">Privacy Policy</a>' in result

        assert "Special Offer" not in result
        assert "SAVE20" not in result
        assert "tracking" not in result
        assert ".hidden" not in result
        assert "2024 ShopCo" in result

        assert "<div" not in result
        assert "<span" not in result
        assert "<header" not in result
        assert "<aside" not in result

    def test_select_contents_as_text(self):
        html = """
        <div>
            <select name="color" data-iid="1">
                <option value="red">Red</option>
                <option value="blue">Blue</option>
            </select>
        </div>
        """
        result = _save("11_select_options", html_to_text(html, [ie(1, "select", {"name": "color"})]), html=html)
        assert '<select name="color">' in result
        assert "Red" in result
        assert "Blue" in result
        assert "</select>" in result
        assert "<option" not in result

    def test_table_content(self):
        html = """
        <table>
            <thead><tr><th>Name</th><th>Price</th></tr></thead>
            <tbody>
                <tr><td>Item A</td><td>$10</td></tr>
                <tr><td>Item B</td><td>$20</td></tr>
            </tbody>
        </table>
        """
        result = _save("11_table", html_to_text(html), html=html)
        assert "Name" in result
        assert "Price" in result
        assert "Item A" in result
        assert "$10" in result
        assert "Item B" in result
        assert "$20" in result


# ===========================================================================
# Test 12: Attribute escaping
# ===========================================================================

class TestAttributeEscaping:
    def test_quotes_in_attributes(self):
        html = """<a href='/search?q="test"&page=1' data-iid="1">Search</a>"""
        result = _save("12_attr_escape_quotes", html_to_text(html, [ie(1, "a", {"href": '/search?q="test"&page=1'})]), html=html)
        assert "Search" in result
        assert "<a" in result
        assert "</a>" in result

    def test_ampersand_in_href(self):
        html = """<a href="/filter?color=red&amp;size=L" data-iid="1">Red Large</a>"""
        result = _save("12_attr_escape_amp", html_to_text(html, [ie(1, "a", {"href": "/filter?color=red&size=L"})]), html=html)
        assert "Red Large" in result
        assert '<a href=' in result


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
