"""Sectioning Algorithm

Divides the text representation of a web page into sections. Each section
corresponds to one or more complete DOM elements and carries multiple
identification strategies (XPath, CSS selector, DOM index) for robust
re-location across page states.

The algorithm is **semantically aware**: it first detects repeated DOM
structures (product cards, article listings, search results) and keeps
them intact as atomic units.  Non-repeated content is split by character
count with intelligent grouping.

Three size thresholds control granularity:

- ``min_size`` — sections smaller than this are merged with neighbours.
- ``preferred_max`` — the algorithm actively tries to stay below this.
- ``hard_max`` — absolute ceiling; only exceeded for single leaf elements
  with no internal structure to split on.

Usage::

    from sectioning.sectioner import section_page
    sections = section_page(html_string, interactive_elements)
    for s in sections:
        print(s.id, s.semantic_role, s.text[:80])
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from lxml import html as lxml_html
from lxml.html import HtmlElement
from lxml.html import tostring as html_tostring

# ---------------------------------------------------------------------------
# Sibling imports (within the page package)
# ---------------------------------------------------------------------------
from .converter import (
    BLOCK_ELEMENTS,
    EXCLUDED_TAGS,
    VOID_ELEMENTS,
    InteractiveElement,
    RenderedInteractiveElement,
    html_to_text_with_elements,
)
from .patterns import (
    GroupAnnotations,
    detect_all_groups,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

# Semantic tags that are good section boundaries.
SEMANTIC_TAGS: frozenset[str] = frozenset(
    {
        "main",
        "nav",
        "header",
        "footer",
        "section",
        "article",
        "aside",
        "form",
        "table",
        "ul",
        "ol",
    }
)

# Tags we prefer to split at (semantic + div as a major container).
SPLIT_PREFERRED_TAGS: frozenset[str] = SEMANTIC_TAGS | {"div"}

# Inline elements — avoid splitting inside these.
INLINE_TAGS: frozenset[str] = frozenset(
    {
        "a",
        "abbr",
        "b",
        "bdi",
        "bdo",
        "br",
        "cite",
        "code",
        "data",
        "dfn",
        "em",
        "i",
        "kbd",
        "mark",
        "q",
        "rp",
        "rt",
        "ruby",
        "s",
        "samp",
        "small",
        "span",
        "strong",
        "sub",
        "sup",
        "time",
        "u",
        "var",
        "wbr",
        "label",
    }
)

# Regex for inline hidden detection (mirrors converter).
_INLINE_HIDDEN_RE = re.compile(
    r"display\s*:\s*none|visibility\s*:\s*hidden",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ElementReference:
    """A DOM element reference with multiple location strategies.

    Provides *xpath*, *css_selector*, *dom_index*, and key *attributes*
    so that downstream consumers can relocate the element using whichever
    strategy is most robust for their use case.
    """

    xpath: str  # lxml-generated XPath
    css_selector: str  # path-based CSS selector
    tag: str  # element tag name
    dom_index: int  # depth-first traversal index
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass
class Section:
    """A section of the page's text representation.

    **Primary identification** — use ``start_element`` and ``end_element``
    references to locate the section in the DOM.

    **Cross-state matching** — use ``content_hash`` (changes on any text
    edit) and ``interactive_hash`` (changes only when interactive element
    structure changes) for the diffing step.

    **Fallback positioning** — ``char_start`` and ``char_end`` give
    character offsets in the concatenated section text.  Prefer element
    references; use these only as a last-resort fallback.
    """

    id: str
    text: str

    # ── Element boundaries (primary location strategy) ──
    start_element: ElementReference
    end_element: ElementReference

    # ── Fingerprints for cross-state matching ──
    content_hash: str  # SHA-256 prefix (16 hex chars) of text
    interactive_hash: str  # SHA-256 prefix of interactive structure

    # ── Structural metadata ──
    depth: int  # nesting depth from content root
    is_interactive: bool  # contains ≥ 1 interactive element
    interactive_count: int  # number of interactive elements
    parent_tag: str  # tag of the DOM parent element
    semantic_role: str  # "navigation", "content", "form", …

    # ── Fallback character positions ──
    # Avoid relying on these — prefer element references.
    char_start: int
    char_end: int

    # ── Sidecar: rendered interactive elements ──
    rendered_interactive_elements: list[RenderedInteractiveElement] = field(
        default_factory=list,
        repr=False,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Hidden Element Detection
# ═══════════════════════════════════════════════════════════════════════════


def _is_hidden(el: HtmlElement) -> bool:
    """Return *True* if the element should be treated as hidden.

    Checks ``data-hidden``, ``aria-hidden``, and inline style hiding.
    """
    if el.get("data-hidden") == "true":
        return True
    if el.get("aria-hidden") == "true":
        return True
    style = el.get("style", "")
    if style and _INLINE_HIDDEN_RE.search(style):
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  Element Identity
# ═══════════════════════════════════════════════════════════════════════════
#
# lxml uses proxy objects for elements. When a proxy is garbage-collected,
# Python may reuse its memory address, causing ``id()`` collisions in
# caches.  To avoid this we:
#   1. Materialise *all* element proxies upfront (``list(doc.iter())``)
#      so they stay alive for the lifetime of the computation.
#   2. Use that stable list for all identity-based caching.
#
# The ``_ElementMap`` helper provides a thin wrapper around this pattern.


class _ElementMap:
    """Identity-safe per-element storage for lxml proxy objects.

    Call :meth:`prepare` once with the root element to materialise
    every proxy.  After that, ``id(el)`` is safe to use as a dict key
    for any element in the tree.
    """

    def __init__(self) -> None:
        self._all_elements: list[HtmlElement] = []
        self.sizes: dict[int, int] = {}
        self.dom_indices: dict[int, int] = {}
        self.annotations: GroupAnnotations = GroupAnnotations()

    def prepare(self, root: HtmlElement) -> None:
        """Materialise all proxy objects so ``id()`` stays stable."""
        self._all_elements = list(root.iter())

    # Convenience: iterate *element* children (skipping non-element nodes).
    @staticmethod
    def element_children(el: HtmlElement) -> list[HtmlElement]:
        """Return element children as a materialised list."""
        return [c for c in el if isinstance(c.tag, str)]


# ═══════════════════════════════════════════════════════════════════════════
#  Size Estimation
# ═══════════════════════════════════════════════════════════════════════════


def _compute_sizes(el: HtmlElement, emap: _ElementMap) -> int:
    """Recursively estimate HTML size for *el* and descendants.

    Measures the approximate serialised HTML character count of the
    element's subtree.  This is what the AI agent will see when it
    zooms into a section, so the size thresholds (``preferred_max``,
    ``hard_max``) correspond directly to the character budget the
    agent's context window has to accommodate.

    Excluded subtrees (``<script>``, ``<style>``, hidden elements)
    return 0 because they are stripped from the output.

    Results are stored in ``emap.sizes``.
    """
    el_key = id(el)
    if el_key in emap.sizes:
        return emap.sizes[el_key]

    tag = el.tag if isinstance(el.tag, str) else ""

    # Skip excluded / hidden subtrees — these are stripped from output.
    if tag in EXCLUDED_TAGS or _is_hidden(el):
        emap.sizes[el_key] = 0
        return 0

    # ── Opening tag: <tag attr1="val1" attr2="val2"> ──
    size = 1 + len(tag)  # "<" + tag
    for attr_name, attr_val in el.attrib.items():
        # space + attr + ="  + val + "
        size += 1 + len(attr_name) + 2 + len(attr_val) + 1
    size += 1  # ">"

    # ── Void elements: self-closing, no children or closing tag ──
    if tag in VOID_ELEMENTS:
        emap.sizes[el_key] = size
        return size

    # ── Direct text content ──
    if el.text:
        size += len(el.text)

    # ── Children + their tail text ──
    for child in el:
        if isinstance(child.tag, str):
            size += _compute_sizes(child, emap)
        # else: comments/PIs — skip
        if child.tail:
            size += len(child.tail)

    # ── Closing tag: </tag> ──
    size += 2 + len(tag) + 1  # "</" + tag + ">"

    emap.sizes[el_key] = size
    return size


# ═══════════════════════════════════════════════════════════════════════════
#  DOM Index Map
# ═══════════════════════════════════════════════════════════════════════════


def _build_dom_index_map(el: HtmlElement, emap: _ElementMap) -> None:
    """Assign depth-first indices to all elements via ``emap.dom_indices``."""
    counter = 0
    for node in el.iter():
        if isinstance(node.tag, str):
            emap.dom_indices[id(node)] = counter
            counter += 1


# ═══════════════════════════════════════════════════════════════════════════
#  Element Reference Builders
# ═══════════════════════════════════════════════════════════════════════════


def _build_css_selector(el: HtmlElement) -> str:
    """Build a CSS selector path that identifies *el*."""
    # Strategy 1: id attribute (most specific).
    el_id = el.get("id")
    if el_id and not re.search(r"\s", el_id):
        return f"#{el_id}"

    # Strategy 2: data-testid.
    testid = el.get("data-testid")
    if testid:
        return f'[data-testid="{testid}"]'

    # Strategy 3: tag path from root.
    parts: list[str] = []
    current: HtmlElement | None = el
    while current is not None and isinstance(current.tag, str):
        tag = current.tag
        if tag in ("html", "body"):
            parts.append(tag)
            break

        parent = current.getparent()
        if parent is None:
            parts.append(tag)
            break

        # Check uniqueness among same-tag siblings.
        siblings = [c for c in parent if isinstance(c.tag, str) and c.tag == tag]
        if len(siblings) == 1:
            cls = (current.get("class") or "").split()
            first_cls = cls[0] if cls else ""
            if first_cls and re.match(r"^[a-zA-Z_-][a-zA-Z0-9_-]*$", first_cls):
                parts.append(f"{tag}.{first_cls}")
            else:
                parts.append(tag)
        else:
            idx = list(siblings).index(current) + 1
            parts.append(f"{tag}:nth-of-type({idx})")

        current = parent

    parts.reverse()
    return " > ".join(parts)


def _make_element_ref(
    el: HtmlElement,
    tree,
    emap: _ElementMap,
) -> ElementReference:
    """Create an :class:`ElementReference` for a DOM element."""
    tag = el.tag if isinstance(el.tag, str) else "unknown"

    try:
        xpath = tree.getpath(el)
    except Exception:
        xpath = ""

    css_selector = _build_css_selector(el)
    dom_index = emap.dom_indices.get(id(el), -1)

    attrs: dict[str, str] = {}
    for attr in ("id", "class", "role", "name", "data-testid", "aria-label", "href"):
        val = el.get(attr)
        if val is not None:
            attrs[attr] = val

    return ElementReference(
        xpath=xpath,
        css_selector=css_selector,
        tag=tag,
        dom_index=dom_index,
        attributes=attrs,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Section ID Generation
# ═══════════════════════════════════════════════════════════════════════════


def _slugify(text: str, max_len: int = 20) -> str:
    """Convert *text* to a URL-friendly slug, truncated to *max_len*."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len]


def _generate_section_id(
    elements: list[HtmlElement],
    index: int,
    used_ids: set[str],
    *,
    group_index: int | None = None,
) -> str:
    """Generate a deterministic, content-derived section ID.

    For repeated-group members, prepends the group position (``item-1-``,
    ``item-2-``, …) so the agent can immediately see which sections form
    a repeated sequence.

    Tries headings → link texts → input placeholders → list item count →
    first words of text content.  Falls back to ``{tag}-section-{index}``.
    """
    tags = [el.tag for el in elements if isinstance(el.tag, str)]
    primary_tag = tags[0] if tags else "section"

    tokens: list[str] = []

    for el in elements:
        # 1. Headings (strongest signal).
        for level in range(1, 7):
            for h in el.iter(f"h{level}"):
                t = (h.text_content() or "").strip()
                if t:
                    tokens.append(t)

    if not tokens:
        for el in elements:
            # 2. Link texts (good for nav sections).
            for a in el.iter("a"):
                t = (a.text_content() or "").strip()
                if t and len(t) < 40:
                    tokens.append(t)
                if len(tokens) >= 4:
                    break
            if len(tokens) >= 4:
                break

    if not tokens:
        for el in elements:
            # 3. Input placeholders.
            for inp in el.iter("input"):
                p = (inp.get("placeholder") or "").strip()
                if p:
                    tokens.append(p)

    # 4. Item count for lists.
    if primary_tag in ("ul", "ol"):
        item_count = 0
        for el in elements:
            item_count += sum(1 for _ in el.iter("li"))
        if item_count:
            tokens.append(f"{item_count}-items")

    # 5. Fallback: first few words of text content.
    if not tokens:
        full_text = " ".join((el.text_content() or "").strip() for el in elements)
        words = full_text.split()[:5]
        tokens = [w for w in words if len(w) > 1]

    # 6. Last resort: parent element context (class, id).
    if not tokens:
        parent = elements[0].getparent()
        if parent is not None:
            pid = parent.get("id")
            if pid:
                tokens.append(pid)
            else:
                pcls = (parent.get("class") or "").split()
                if pcls:
                    tokens.append(pcls[0])

    # Build the slug.
    if tokens:
        slug_parts = [_slugify(t) for t in tokens[:4] if _slugify(t)]
        slug = "-".join(slug_parts) if slug_parts else ""
    else:
        slug = ""

    # Prepend group position for repeated-group members.
    if group_index is not None:
        if slug:
            base_id = f"item-{group_index}-{primary_tag}-{slug}"
        else:
            base_id = f"item-{group_index}-{primary_tag}-section-{index}"
    else:
        if slug:
            base_id = f"{primary_tag}-{slug}"
        else:
            base_id = f"{primary_tag}-section-{index}"

    # Truncate to a reasonable length.
    base_id = base_id[:60]

    # Ensure uniqueness.
    final_id = base_id
    counter = 2
    while final_id in used_ids:
        final_id = f"{base_id}-{counter}"
        counter += 1

    used_ids.add(final_id)
    return final_id


# ═══════════════════════════════════════════════════════════════════════════
#  Semantic Role Classification
# ═══════════════════════════════════════════════════════════════════════════
#
# STRICT policy: only assign a role when we are highly confident.
# A wrong label is worse than no label — it misleads the AI agent.
# Default is always "content".

# Tag → role mapping.  Only HTML5 semantic elements.
_STRICT_TAG_ROLES: dict[str, str] = {
    "nav": "navigation",
    "header": "header",
    "footer": "footer",
    "form": "form",
    "table": "table",
    "aside": "sidebar",
    "article": "article",
    "main": "main-content",
    "ul": "list",
    "ol": "list",
}

# ARIA roles we trust (WAI-ARIA landmark + widget roles).
# Anything not in this set is ignored — too many sites misuse ARIA.
_TRUSTED_ARIA_ROLES: frozenset[str] = frozenset(
    {
        "navigation",
        "banner",
        "contentinfo",
        "complementary",
        "main",
        "form",
        "search",
        "region",
        "alert",
        "alertdialog",
        "dialog",
        "tablist",
        "toolbar",
        "menu",
        "menubar",
        "listbox",
    }
)


def _classify_semantic_role(
    elements: list[HtmlElement],
    emap: _ElementMap | None = None,
) -> str:
    """Classify semantic role from HTML tags, ARIA roles, and group membership.

    Only assigns a role when there is an explicit, trustworthy signal.
    Returns ``"content"`` for anything uncertain.
    """
    # 1. Direct HTML tag match — highest confidence.
    for el in elements:
        tag = el.tag if isinstance(el.tag, str) else ""
        if tag in _STRICT_TAG_ROLES:
            return _STRICT_TAG_ROLES[tag]

    # 2. Explicit ARIA role attribute — only trusted roles.
    for el in elements:
        role = (el.get("role") or "").lower().strip()
        if role in _TRUSTED_ARIA_ROLES:
            return role

    # 3. Repeated group membership — structural signal.
    if emap is not None:
        for el in elements:
            if emap.annotations.is_atomic(el):
                return "repeated-item"

    # 4. Everything else is content.  No guessing.
    return "content"


# ═══════════════════════════════════════════════════════════════════════════
#  Section Text Extraction
# ═══════════════════════════════════════════════════════════════════════════


def _elements_to_html(elements: list[HtmlElement]) -> str:
    """Serialize element(s) to HTML for passing through the converter.

    Includes tail text *between* grouped siblings but not after the last
    element (that tail belongs to whatever follows this section).
    """
    if not elements:
        return ""

    parts: list[str] = []
    for i, el in enumerate(elements):
        include_tail = i < len(elements) - 1
        try:
            parts.append(html_tostring(el, encoding="unicode", with_tail=include_tail))
        except Exception:
            # Fallback: raw text content.
            parts.append(el.text_content() or "")

    return "".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
#  Interactive Element Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _count_interactive(elements: list[HtmlElement]) -> int:
    """Count elements with ``data-iid`` in the subtree(s)."""
    count = 0
    for el in elements:
        if el.get("data-iid") is not None:
            count += 1
        for desc in el.iterdescendants():
            if isinstance(desc.tag, str) and desc.get("data-iid") is not None:
                count += 1
    return count


def _interactive_signature(elements: list[HtmlElement]) -> str:
    """Build a deterministic string representing interactive structure.

    The signature includes every interactive element's tag and key
    attributes, sorted for stability.
    """
    parts: list[str] = []
    for el in elements:
        _collect_interactive_sig(el, parts)
    parts.sort()
    return "|".join(parts)


def _collect_interactive_sig(el: HtmlElement, out: list[str]) -> None:
    """Recursively collect interactive element signatures."""
    if not isinstance(el.tag, str):
        return
    iid = el.get("data-iid")
    if iid is not None:
        tag = el.tag
        href = el.get("href", "")
        name = el.get("name", "")
        role = el.get("role", "")
        out.append(f"{tag}:iid={iid}:href={href}:name={name}:role={role}")
    for child in el:
        _collect_interactive_sig(child, out)


# ═══════════════════════════════════════════════════════════════════════════
#  Sectioning Algorithm
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class _Candidate:
    """Internal: a candidate section produced during recursive sectioning."""

    elements: list[HtmlElement]
    size: int
    depth: int
    # Opaque identifier for the repeated group this candidate belongs to.
    # Candidates with the same non-None group_key may be batched together;
    # candidates with different group_keys (or None) are never merged.
    group_key: int | None = None


def _is_dropdown_menu(el: HtmlElement) -> bool:
    """True if *el* is a dropdown/menu container that should stay atomic."""
    role = (el.get("role") or "").lower()
    if role in ("menu", "listbox", "menubar"):
        return True
    tag = el.tag if isinstance(el.tag, str) else ""
    if tag == "select":
        return True
    # Check if children are menu items — if so, parent is a menu container.
    children = _ElementMap.element_children(el)
    if len(children) >= 3:
        menu_roles = {"menuitem", "menuitemradio", "menuitemcheckbox", "option"}
        child_roles = [(c.get("role") or "").lower() for c in children if isinstance(c.tag, str)]
        if child_roles and all(r in menu_roles for r in child_roles if r):
            # At least half the children have menu-item roles.
            menu_count = sum(1 for r in child_roles if r in menu_roles)
            if menu_count >= len(children) * 0.5:
                return True
    return False


def _should_recurse_despite_size(
    el: HtmlElement,
    emap: _ElementMap,
    min_size: int,
    preferred_max: int,
) -> bool:
    """True if a within-range element should still be recursed into.

    Generic containers (``div``, ``span``) that wrap multiple meaningful
    children are likely layout wrappers, not atomic content blocks.
    Semantic elements (``nav``, ``article``, ``form``, …) are kept whole.
    """
    tag = el.tag if isinstance(el.tag, str) else ""
    # Semantic elements are atomic — they ARE the section.
    if tag in SEMANTIC_TAGS:
        return False
    # Dropdown menus are atomic.
    if _is_dropdown_menu(el):
        return False
    # Generic containers with multiple visible block-level children.
    children = [c for c in _ElementMap.element_children(el) if emap.sizes.get(id(c), 0) > 0]
    block_children = [
        c for c in children if (c.tag if isinstance(c.tag, str) else "") in BLOCK_ELEMENTS
    ]
    # 3+ block children → definitely a wrapper.
    if len(block_children) >= 3:
        return True
    # 2 block children where the largest is substantial (> 40% of preferred_max)
    # → likely a two-column layout or header+content wrapper.
    if len(block_children) >= 2:
        largest = max(emap.sizes.get(id(c), 0) for c in block_children)
        if largest > preferred_max * 0.4:
            return True
    return False


# ── Children-with-groups sectioning ──────────────────────────────────────


def _section_children_with_groups(
    el: HtmlElement,
    emap: _ElementMap,
    min_size: int,
    preferred_max: int,
    hard_max: int,
    depth: int,
) -> list[_Candidate]:
    """Section children of *el* using detected repeated-group annotations.

    Group members are kept atomic (never split internally) as long as
    they fit within *hard_max*.  Non-group children are processed with
    normal recursive sectioning.  The final candidate list is grouped
    so that adjacent small candidates of the *same* group may be batched
    together, but elements from different groups or non-group elements
    are never merged.
    """
    children = [c for c in _ElementMap.element_children(el) if emap.sizes.get(id(c), 0) > 0]
    if not children:
        size = emap.sizes.get(id(el), 0)
        return [_Candidate(elements=[el], size=size, depth=depth)]

    # Build lookup: child id → group id (using parent id of the group's parent
    # + hash of group signature as a unique key).
    parent_groups = emap.annotations.get_groups_for_parent(el)
    atomic_to_gkey: dict[int, int] = {}
    for group in parent_groups:
        gkey = id(group)
        for member in group.members:
            atomic_to_gkey[id(member)] = gkey

    candidates: list[_Candidate] = []
    for child in children:
        child_size = emap.sizes.get(id(child), 0)
        if child_size == 0:
            continue

        gkey = atomic_to_gkey.get(id(child))
        if gkey is not None:
            # ── Group member: keep atomic ──
            if child_size <= hard_max:
                candidates.append(
                    _Candidate(
                        elements=[child],
                        size=child_size,
                        depth=depth,
                        group_key=gkey,
                    )
                )
            else:
                # Exceeds hard_max — recurse inside (sub-groups may exist)
                candidates.extend(
                    _section_element(child, emap, min_size, preferred_max, hard_max, depth)
                )
        else:
            # ── Non-group child: normal recursive sectioning ──
            candidates.extend(
                _section_element(child, emap, min_size, preferred_max, hard_max, depth)
            )

    if not candidates:
        size = emap.sizes.get(id(el), 0)
        if size > hard_max:
            return []
        return [_Candidate(elements=[el], size=size, depth=depth)]

    return _group_candidates(candidates, min_size, preferred_max)


# ── Core recursive sectioning ────────────────────────────────────────────


def _section_element(
    el: HtmlElement,
    emap: _ElementMap,
    min_size: int,
    preferred_max: int,
    hard_max: int,
    depth: int,
) -> list[_Candidate]:
    """Recursively decide how to section *el*.

    Returns a list of :class:`_Candidate` objects representing sections
    that should be created from this element and its subtree.
    """
    tag = el.tag if isinstance(el.tag, str) else ""
    size = emap.sizes.get(id(el), 0)

    # Skip empty / invisible elements.
    if size == 0:
        return []

    # Dropdown menus are atomic UI controls — keep small ones as a single
    # section, but skip oversized ones (e.g. 300-item language pickers)
    # since they are UI chrome, not scrapable content.
    if _is_dropdown_menu(el):
        if size <= preferred_max:
            return [_Candidate(elements=[el], size=size, depth=depth)]
        return []

    # ── Atomic group member: keep whole up to hard_max ──
    # This is the key integration point with repeated pattern detection.
    # If this element is part of a detected repeated group, keep it as one
    # unit.  The outermost matching level wins (top-down processing ensures
    # we never recurse into an element that fits within hard_max).
    if emap.annotations.is_atomic(el) and size <= hard_max:
        group = emap.annotations.get_group(el)
        return [
            _Candidate(
                elements=[el],
                size=size,
                depth=depth,
                group_key=id(group) if group else None,
            )
        ]

    # html/body are structural wrappers — always recurse, never section.
    is_structural_wrapper = tag in ("html", "body")

    # ── Within preferred range → usually this element is one section ──
    if min_size <= size <= preferred_max and not is_structural_wrapper:
        # Generic containers with many children should still be split.
        if not _should_recurse_despite_size(el, emap, min_size, preferred_max):
            return [_Candidate(elements=[el], size=size, depth=depth)]
        # Fall through to recursion below.

    # ── Too large / structural wrapper / splittable container → recurse ──
    if size >= min_size or is_structural_wrapper:
        # Check if this parent has detected repeated groups among children.
        parent_groups = emap.annotations.get_groups_for_parent(el)
        if parent_groups:
            return _section_children_with_groups(
                el,
                emap,
                min_size,
                preferred_max,
                hard_max,
                depth + 1,
            )

        children = [c for c in _ElementMap.element_children(el) if emap.sizes.get(id(c), 0) > 0]

        if not children:
            # All content is direct text on this element; can't split further.
            return [_Candidate(elements=[el], size=size, depth=depth)]

        # Uniform siblings (all same tag, ≥ 2) get consistent grouping.
        child_tags = {c.tag for c in children if isinstance(c.tag, str)}
        if len(child_tags) == 1 and len(children) >= 2:
            return _section_uniform_siblings(
                children,
                emap,
                min_size,
                preferred_max,
                hard_max,
                depth + 1,
            )

        # Mixed children — recurse each individually.
        child_candidates: list[_Candidate] = []
        for child in children:
            child_candidates.extend(
                _section_element(child, emap, min_size, preferred_max, hard_max, depth + 1)
            )

        if not child_candidates:
            # Recursion produced nothing — either all children were
            # explicitly skipped (e.g. oversized dropdowns) or empty.
            # For oversized elements, don't fall back to emitting the
            # parent since that reintroduces the skipped content.
            if size > hard_max:
                return []
            return [_Candidate(elements=[el], size=size, depth=depth)]

        # Group adjacent small candidates.
        return _group_candidates(child_candidates, min_size, preferred_max)

    # ── Too small → return as-is; parent handles grouping ──
    return [_Candidate(elements=[el], size=size, depth=depth)]


def _section_uniform_siblings(
    children: list[HtmlElement],
    emap: _ElementMap,
    min_size: int,
    preferred_max: int,
    hard_max: int,
    depth: int,
) -> list[_Candidate]:
    """Section a list of same-tag siblings with consistent grouping.

    Either every child is its own section (if median size ≥ *min_size*)
    or they are grouped into equal batches.  Never a mix.
    """
    candidates: list[_Candidate] = []
    for child in children:
        size = emap.sizes.get(id(child), 0)
        if size == 0:
            continue
        if size > hard_max:
            # Oversized child — recurse into it.
            candidates.extend(
                _section_element(child, emap, min_size, preferred_max, hard_max, depth)
            )
        else:
            candidates.append(_Candidate(elements=[child], size=size, depth=depth))

    if not candidates:
        return []

    # Decide strategy based on median size.
    sizes = sorted(c.size for c in candidates)
    median = sizes[len(sizes) // 2]

    if median >= min_size:
        # Each sibling is large enough — keep individually.
        return candidates

    # Siblings are generally small — group them uniformly.
    return _group_uniform(candidates, min_size)


# ═══════════════════════════════════════════════════════════════════════════
#  Candidate Grouping
# ═══════════════════════════════════════════════════════════════════════════


def _group_candidates(
    candidates: list[_Candidate],
    min_size: int,
    preferred_max: int,
) -> list[_Candidate]:
    """Group adjacent undersized candidates to meet *min_size*.

    Respects group boundaries: candidates from different repeated groups
    (or a group vs non-group) are never merged together.
    """
    if not candidates:
        return []

    result: list[_Candidate] = []
    pending: list[_Candidate] = []
    pending_size = 0
    pending_gkey: int | None = None  # group_key of elements in pending

    def flush() -> None:
        nonlocal pending, pending_size, pending_gkey
        if pending:
            result.extend(_flush_pending(pending, pending_size, min_size))
            pending = []
            pending_size = 0
            pending_gkey = None

    for candidate in candidates:
        # Check if this candidate can be grouped with pending ones.
        # Never merge across different group_key values.
        if pending and candidate.group_key != pending_gkey:
            flush()

        if candidate.size >= min_size:
            # Flush accumulated small candidates.
            flush()
            result.append(candidate)
        else:
            pending.append(candidate)
            pending_size += candidate.size
            pending_gkey = candidate.group_key
            # Once the group is large enough, emit it.
            if pending_size >= min_size:
                result.append(_merge_candidates(pending))
                pending = []
                pending_size = 0
                pending_gkey = None

    # Remaining small candidates.
    flush()

    return result


def _flush_pending(
    pending: list[_Candidate],
    pending_size: int,
    min_size: int,
) -> list[_Candidate]:
    """Flush a run of small candidates, handling repeating patterns."""
    if not pending:
        return []

    # Detect uniform repeating pattern (≥ 3 single-element candidates
    # of the same tag) and group them into equal batches.
    if len(pending) >= 3 and _is_uniform_pattern(pending):
        return _group_uniform(pending, min_size)

    # Default: merge everything into one candidate.
    return [_merge_candidates(pending)]


def _is_uniform_pattern(candidates: list[_Candidate]) -> bool:
    """True if all candidates are single-element and share the same tag."""
    if not all(len(c.elements) == 1 for c in candidates):
        return False
    tags = {c.elements[0].tag for c in candidates}
    return len(tags) == 1


def _group_uniform(
    candidates: list[_Candidate],
    min_size: int,
) -> list[_Candidate]:
    """Group uniform candidates into equal-sized batches.

    Tries to make each batch reach *min_size* while keeping batch sizes
    as equal as possible (e.g. 5 groups of 2, not 4+1).
    """
    if not candidates:
        return []

    total = len(candidates)
    avg_size = sum(c.size for c in candidates) / total if total else 1
    items_per_group = max(1, round(min_size / avg_size)) if avg_size > 0 else 1

    # Ensure we don't create a tiny leftover group.
    num_groups = max(1, round(total / items_per_group))
    items_per_group = max(1, total // num_groups)
    remainder = total % num_groups

    result: list[_Candidate] = []
    idx = 0
    for g in range(num_groups):
        # Distribute remainder evenly across the first groups.
        batch_size = items_per_group + (1 if g < remainder else 0)
        batch = candidates[idx : idx + batch_size]
        if batch:
            result.append(_merge_candidates(batch))
        idx += batch_size

    return result


def _merge_candidates(candidates: list[_Candidate]) -> _Candidate:
    """Merge multiple candidates into a single candidate."""
    elements: list[HtmlElement] = []
    total_size = 0
    min_depth = float("inf")
    group_key = candidates[0].group_key if candidates else None
    for c in candidates:
        elements.extend(c.elements)
        total_size += c.size
        min_depth = min(min_depth, c.depth)
    return _Candidate(
        elements=elements,
        size=total_size,
        depth=int(min_depth) if min_depth != float("inf") else 0,
        group_key=group_key,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════


def section_page(
    html_string: str,
    interactive_elements: list[InteractiveElement] | None = None,
    *,
    min_size: int = 500,
    preferred_max: int = 15000,
    hard_max: int = 30000,
) -> list[Section]:
    """Divide a page into sections with IDs, element references, and metadata.

    Each section corresponds to one or more complete DOM elements.
    Sections carry multiple location strategies (XPath, CSS selector,
    DOM index), content fingerprints for cross-state diffing, and
    semantic metadata.

    The algorithm detects repeated DOM structures (product cards, article
    listings, search results) and keeps them intact as atomic units.

    Args:
        html_string: Full page HTML (may include ``data-iid`` and
            ``data-hidden`` markers from browser-side detection).
        interactive_elements: Interactive element descriptors whose
            ``iid`` values match ``data-iid`` attributes in the HTML.
        min_size: Minimum section size in characters.  Sections smaller
            than this are merged with adjacent content.
        preferred_max: Target maximum section size.  The algorithm tries
            to stay below this threshold.
        hard_max: Absolute maximum section size.  Only exceeded for single
            leaf elements that cannot be split further.

    Returns:
        Ordered list of :class:`Section` dataclasses covering the page
        content.  Sections are contiguous and non-overlapping.
    """
    if not html_string or not html_string.strip():
        return []

    # ── Parse HTML ──
    try:
        doc = lxml_html.fromstring(html_string)
    except Exception:
        return []

    if not isinstance(doc, HtmlElement):
        return []

    tree = doc.getroottree()

    # ── Stabilise element proxies ──
    # lxml reuses proxy memory addresses after GC.  Materialising every
    # proxy into _ElementMap keeps them alive so id() stays unique.
    emap = _ElementMap()
    emap.prepare(doc)

    # ── Phase 1: compute sizes ──
    _compute_sizes(doc, emap)

    # ── Phase 2: build DOM index map ──
    _build_dom_index_map(doc, emap)

    # ── Phase 2.5: detect repeated groups ──
    body = doc.find(".//body")
    root = body if body is not None else doc
    emap.annotations = detect_all_groups(root, emap.sizes)

    # ── Phase 3: find content root ──
    # (root already found above for group detection)

    # ── Phase 4: run sectioning algorithm ──
    candidates = _section_element(
        root,
        emap,
        min_size,
        preferred_max,
        hard_max,
        depth=0,
    )

    # ── Phase 5: build Section objects ──
    sections: list[Section] = []
    used_ids: set[str] = set()
    char_offset = 0

    # Track group-member counters for sequential numbering in IDs.
    group_counters: dict[int, int] = {}

    for i, candidate in enumerate(candidates):
        # Generate text via the converter, also capturing rendered
        # interactive elements for the sidecar data (Task 1).
        section_html = _elements_to_html(candidate.elements)
        section_text, rendered_elements = html_to_text_with_elements(
            section_html,
            interactive_elements,
        )

        if not section_text.strip():
            continue

        # Determine group index for repeated-group members.
        group_index = None
        if candidate.group_key is not None:
            counter = group_counters.get(candidate.group_key, 0) + 1
            group_counters[candidate.group_key] = counter
            group_index = counter

        # ID.
        section_id = _generate_section_id(
            candidate.elements,
            i,
            used_ids,
            group_index=group_index,
        )

        # Element references.
        start_ref = _make_element_ref(candidate.elements[0], tree, emap)
        end_ref = _make_element_ref(candidate.elements[-1], tree, emap)

        # Fingerprints.
        content_hash = hashlib.sha256(section_text.encode()).hexdigest()[:16]
        sig = _interactive_signature(candidate.elements)
        interactive_hash = hashlib.sha256(sig.encode()).hexdigest()[:16]

        # Metadata.
        i_count = _count_interactive(candidate.elements)
        parent = candidate.elements[0].getparent()
        parent_tag = parent.tag if parent is not None and isinstance(parent.tag, str) else ""

        char_end = char_offset + len(section_text)

        sections.append(
            Section(
                id=section_id,
                text=section_text,
                start_element=start_ref,
                end_element=end_ref,
                content_hash=content_hash,
                interactive_hash=interactive_hash,
                depth=candidate.depth,
                is_interactive=i_count > 0,
                interactive_count=i_count,
                parent_tag=parent_tag,
                semantic_role=_classify_semantic_role(
                    candidate.elements,
                    emap,
                ),
                char_start=char_offset,
                char_end=char_end,
                rendered_interactive_elements=rendered_elements,
            )
        )

        # +2 for the \n\n separator between sections.
        char_offset = char_end + 2

    return sections
