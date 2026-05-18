"""HTML-to-Text Converter

Converts full-page HTML into a compact text representation where:
- All visible content is rendered as plain text
- Interactive elements (links, buttons, inputs, etc.) preserve their HTML tags
- Hidden elements, scripts, and styles are excluded

Interactive elements are identified by `data-iid` attributes injected into the HTML
by the browser-side detection system. Each element's metadata (tag, attributes, text)
is provided as a companion list of InteractiveElement descriptors.

Hidden elements are identified by `data-hidden="true"` attributes injected by the
browser-side detection, plus Python-side safety-net checks for inline styles and
aria-hidden.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lxml import html as lxml_html
from lxml.html import HtmlElement

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class InteractiveElement:
    """Descriptor for an interactive element detected by the browser-side system."""

    iid: int  # matches data-iid in the HTML
    tag: str  # e.g. "a", "button", "input"
    attributes: dict[str, str] = field(default_factory=dict)
    text: str = ""
    selector: str = ""  # CSS selector, kept for other system uses


@dataclass
class RenderedInteractiveElement:
    """An interactive element as it was rendered during the HTML-to-text walk.

    Captures the exact tag string the agent sees plus metadata needed by
    the show_page context scoring algorithm (Task 3).
    """

    iid: int  # matches data-iid
    tag: str  # "button", "input", "a", ...
    attributes: dict[str, str]  # all ALLOWED_ATTRIBUTES present on this element
    classes: list[str]  # class list (split), empty if no class attr
    element_text: str | None  # visible text content (trimmed)
    full_tag_str: str  # opening tag exactly as rendered


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Attributes we preserve on interactive element tags.
ALLOWED_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "href",
        "src",
        "type",
        "name",
        "value",
        "placeholder",
        "role",
        "aria-label",
        "aria-expanded",
        "aria-haspopup",
        "data-testid",
        "id",
        "action",
        "method",
    }
)

# Tags whose entire subtree is always excluded.
EXCLUDED_TAGS: frozenset[str] = frozenset(
    {
        "script",
        "style",
        "noscript",
        "template",
        # <head> contents are never visible page content
        "head",
        # SVG is presentational — emit aria-label if present, skip otherwise
        "svg",
        # <textarea> and <xmp> contain raw text that is not visible page content
        "textarea",
        "xmp",
    }
)

# Void (self-closing) HTML elements — no closing tag emitted.
VOID_ELEMENTS: frozenset[str] = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)

# Block-level elements — we insert line breaks around these.
BLOCK_ELEMENTS: frozenset[str] = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "body",
        "br",
        "caption",
        "col",
        "colgroup",
        "dd",
        "details",
        "dialog",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hgroup",
        "hr",
        "html",
        "legend",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "summary",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }
)

# Regex to detect obvious inline hiding via the style attribute.
_INLINE_HIDDEN_RE = re.compile(
    r"""
    display\s*:\s*none
    | visibility\s*:\s*hidden
    """,
    re.VERBOSE | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Hidden-element detection (Python safety net)
# ---------------------------------------------------------------------------


def _is_hidden(el: HtmlElement) -> bool:
    """Check if an element should be treated as hidden.

    Primary hiding is done browser-side (data-hidden="true"). This function
    provides a safety net for:
    - data-hidden markers from the browser detection system
    - aria-hidden="true"
    - Obvious inline style hiding (display:none, visibility:hidden)
    """
    # Browser-side marker (most reliable)
    if el.get("data-hidden") == "true":
        return True

    # ARIA semantic hiding
    if el.get("aria-hidden") == "true":
        return True

    # Inline style check
    style = el.get("style", "")
    if style and _INLINE_HIDDEN_RE.search(style):
        return True

    return False


# ---------------------------------------------------------------------------
# Interactive element tag rendering
# ---------------------------------------------------------------------------


def _merge_allowed_attributes(
    el: HtmlElement,
    meta: InteractiveElement | None,
) -> dict[str, str]:
    """Merge DOM and metadata attributes, keeping only ALLOWED_ATTRIBUTES.

    DOM attributes are authoritative; metadata fills in anything the
    parser may have normalised away.
    """
    attrs: dict[str, str] = {}
    for attr_name, attr_value in el.attrib.items():
        if attr_name in ALLOWED_ATTRIBUTES:
            attrs[attr_name] = attr_value
    if meta:
        for attr_name, attr_value in meta.attributes.items():
            if attr_name in ALLOWED_ATTRIBUTES and attr_name not in attrs:
                attrs[attr_name] = attr_value
    return attrs


def _build_opening_tag(el: HtmlElement, meta: InteractiveElement | None) -> str:
    """Build the opening HTML tag for an interactive element, keeping only
    allowed attributes."""
    tag = el.tag
    parts = [tag]

    attrs = _merge_allowed_attributes(el, meta)

    # Deterministic attribute order: sorted alphabetically.
    for attr_name in sorted(attrs):
        value = attrs[attr_name]
        # Escape quotes in attribute values.
        escaped = value.replace("&", "&amp;").replace('"', "&quot;")
        parts.append(f'{attr_name}="{escaped}"')

    return "<" + " ".join(parts) + ">"


def _build_closing_tag(tag: str) -> str:
    return f"</{tag}>"


# ---------------------------------------------------------------------------
# Core tree walker
# ---------------------------------------------------------------------------


class _TreeWalker:
    """Walks the lxml DOM tree depth-first, producing text output."""

    def __init__(self, interactive_map: dict[int, InteractiveElement]):
        self._interactive_map = interactive_map
        # Set of data-iid values for elements we've identified as interactive
        # (built during walk from data-iid attributes on DOM nodes).
        self._iid_nodes: set[int] = set()
        self._parts: list[str] = []
        self._text: str = ""
        self.rendered_elements: list[RenderedInteractiveElement] = []

    # ---- public interface ----

    def walk(self, root: HtmlElement) -> str:
        """Walk the tree and return the final text output."""
        self._walk_node(root)
        self._text = self._finalise()
        return self._text

    # ---- internal walk ----

    def _walk_node(self, el: HtmlElement) -> None:
        tag = el.tag if isinstance(el.tag, str) else ""

        # Skip non-element nodes (comments, processing instructions).
        # In lxml, el.tag is a callable (not str) for these nodes.
        if not tag:
            return

        # Skip excluded tags entirely.
        # For SVG, emit aria-label as text if present (like img alt).
        if tag in EXCLUDED_TAGS:
            if tag == "svg":
                label = el.get("aria-label", "").strip()
                if label:
                    self._emit_text(label)
            return

        # Skip hidden elements entirely.
        if _is_hidden(el):
            return

        # Handle <img> — emit alt text only.
        if tag == "img":
            alt = el.get("alt", "").strip()
            if alt:
                self._emit_text(alt)
            return

        # Handle <br> — emit a newline.
        if tag == "br":
            self._parts.append("\n")
            return

        # Check if this element is interactive (has data-iid).
        iid_str = el.get("data-iid")
        is_interactive = iid_str is not None
        iid: int | None = None
        meta: InteractiveElement | None = None

        if is_interactive:
            try:
                iid = int(iid_str)
                meta = self._interactive_map.get(iid)
            except (ValueError, TypeError):
                is_interactive = False

        is_block = tag in BLOCK_ELEMENTS

        # --- emit opening ---
        if is_block:
            self._emit_block_break()

        if is_interactive:
            opening = _build_opening_tag(el, meta)
            self._parts.append(opening)

            # Capture rendered interactive element for sidecar data.
            self.rendered_elements.append(
                RenderedInteractiveElement(
                    iid=iid,  # type: ignore[arg-type]
                    tag=tag,
                    attributes=_merge_allowed_attributes(el, meta),
                    classes=el.get("class", "").split(),
                    element_text=(el.text_content() or "").strip() or None,
                    full_tag_str=opening,
                )
            )

            if tag in VOID_ELEMENTS:
                # Self-closing interactive element (e.g. <input>). Done.
                if is_block:
                    self._emit_block_break()
                return
            # For block-level interactive elements with substantial content,
            # start inner content on next line.
            if is_block:
                self._parts.append("\n")

        # --- recurse into children ---
        # el.text is the text before the first child.
        if el.text:
            self._emit_text(el.text)

        for child in el:
            self._walk_node(child)
            # child.tail is text after this child but before the next sibling.
            if child.tail:
                self._emit_text(child.tail)

        # --- emit closing ---
        if is_interactive:
            if is_block:
                # Ensure closing tag on its own line for readability.
                self._ensure_newline()
            self._parts.append(_build_closing_tag(tag))

        if is_block:
            self._emit_block_break()

    # ---- text emission helpers ----

    def _emit_text(self, text: str) -> None:
        """Emit a piece of text, normalising internal whitespace."""
        # Collapse whitespace runs to single space.
        normalised = re.sub(r"[ \t]+", " ", text)
        # Preserve explicit newlines but collapse runs of them.
        normalised = re.sub(r"\n[ \t]*", "\n", normalised)
        if normalised and normalised != " ":
            self._parts.append(normalised)

    def _emit_block_break(self) -> None:
        """Emit a block-level break (blank line separator)."""
        self._parts.append("\n\n")

    def _ensure_newline(self) -> None:
        """Make sure current output ends with a newline."""
        if self._parts and not self._parts[-1].endswith("\n"):
            self._parts.append("\n")

    # ---- post-processing ----

    def _finalise(self) -> str:
        raw = "".join(self._parts)

        # Normalise whitespace:
        # 1. Strip trailing spaces on each line.
        raw = re.sub(r"[ \t]+$", "", raw, flags=re.MULTILINE)
        # 2. Collapse 3+ consecutive newlines to 2 (one blank line).
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        # 3. Strip leading/trailing whitespace.
        raw = raw.strip()

        return raw


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _run_walker(
    html_string: str,
    interactive_elements: list[InteractiveElement] | None = None,
) -> _TreeWalker | None:
    """Parse HTML and walk the tree, returning the walker (or None on failure)."""
    if not html_string or not html_string.strip():
        return None

    interactive_map: dict[int, InteractiveElement] = {}
    if interactive_elements:
        for ie in interactive_elements:
            interactive_map[ie.iid] = ie

    try:
        doc = lxml_html.fromstring(html_string)
    except Exception:
        return None

    if not isinstance(doc, HtmlElement):
        return None

    walker = _TreeWalker(interactive_map)
    walker.walk(doc)
    return walker


def html_to_text(
    html_string: str,
    interactive_elements: list[InteractiveElement] | None = None,
) -> str:
    """Convert an HTML string to a text representation.

    Args:
        html_string: Full page HTML (as returned by page.content()).
                     May contain data-iid and data-hidden attributes injected
                     by the browser-side detection system.
        interactive_elements: List of interactive element descriptors. Each
                              must have an ``iid`` that matches a ``data-iid``
                              attribute in the HTML. If None, no elements are
                              treated as interactive (pure text extraction).

    Returns:
        A text string where visible content is plain text and interactive
        elements are preserved with their HTML tags.
    """
    walker = _run_walker(html_string, interactive_elements)
    if walker is None:
        return ""
    return walker._text


def html_to_text_with_elements(
    html_string: str,
    interactive_elements: list[InteractiveElement] | None = None,
) -> tuple[str, list[RenderedInteractiveElement]]:
    """Convert HTML to text and return rendered interactive elements.

    Like :func:`html_to_text` but also returns the
    :class:`RenderedInteractiveElement` list captured during the walk.
    Used by the sectioner to populate per-section sidecar data.

    Returns:
        ``(text, rendered_elements)`` tuple.
    """
    walker = _run_walker(html_string, interactive_elements)
    if walker is None:
        return "", []
    return walker._text, walker.rendered_elements
