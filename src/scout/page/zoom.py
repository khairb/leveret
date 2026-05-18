"""Zoom-in: return contextual HTML for one or more sections.

Given a stored HTML snapshot and section references (from the sectioner),
this module locates the corresponding elements and uses ``extract_sections``
to produce a pruned DOM that shows the section content in full with
structural context (ancestors preserved, everything else replaced by
placeholders).

**Bridging lxml and BeautifulSoup** — The sectioner uses lxml and produces
``ElementReference.dom_index`` values (depth-first traversal indices).
``extract_sections`` uses BeautifulSoup.

We parse the HTML once with ``BeautifulSoup(html, "lxml")`` — using lxml
as the BS4 parser backend.  Since both the sectioner and this module use
the same underlying lxml parser, they produce the same tree structure
(including auto-inserted ``<tbody>`` etc.).  We then build a flat
depth-first element list from the BS4 tree and look up elements directly
by ``dom_index``.

Performance: one parse per ``zoom_in`` call, O(1) element lookup.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag

from .extract import extract_sections
from .sectioner import Section

# The BS4 document wrapper that appears in prettified output.
_DOCUMENT_OPEN_RE = re.compile(r"<\[document\]>\n?")
_DOCUMENT_CLOSE_RE = re.compile(r"\n?</\[document\]>\n?")

# Markers injected by interactive-element detection — internal only,
# must never be shown to the agent (they are unstable across sessions).
_DATA_IID_RE = re.compile(r'\s+data-iid="[^"]*"')
_DATA_HIDDEN_RE = re.compile(r'\s+data-hidden="[^"]*"')


# ═══════════════════════════════════════════════════════════════════════════
#  Element Location
# ═══════════════════════════════════════════════════════════════════════════


def _build_bs4_dom_index(soup: BeautifulSoup) -> list[Tag]:
    """Build a flat list of BS4 Tags in depth-first order.

    Mirrors the lxml sectioner's ``_build_dom_index_map``: depth-first
    pre-order traversal, counting only real Tag elements.  The list
    index equals the element's ``dom_index``.

    We skip the ``[document]`` wrapper (BS4's root) and start from the
    first real HTML element, matching lxml's ``fromstring()`` which
    returns the root element directly.
    """
    elements: list[Tag] = []

    def walk(node: Tag) -> None:
        if isinstance(node, Tag) and node.name != "[document]":
            elements.append(node)
            for child in node.children:
                if isinstance(child, Tag):
                    walk(child)

    # Walk children of soup (skipping the [document] wrapper itself).
    for child in soup.children:
        if isinstance(child, Tag):
            walk(child)

    return elements


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════


def zoom_in(
    html: str,
    sections: list[Section],
    section_ids: list[str],
) -> str:
    """Return contextual HTML for one or more sections.

    The result is a pruned DOM tree: the requested sections are shown in
    full, ancestor elements provide structural context, and everything
    else is replaced by placeholders.

    Args:
        html: The full stamped HTML snapshot (stored in ``PageState``).
        sections: All sections for the current state.
        section_ids: One or more section IDs to zoom into.

    Returns:
        Prettified HTML string showing the requested sections with
        structural context.  Returns an error message (not an exception)
        if any section ID is not found or cannot be located.
    """
    if not section_ids:
        return "[zoom_in] No section IDs provided."

    # Build section lookup.
    section_map: dict[str, Section] = {s.id: s for s in sections}

    # Validate all IDs upfront.
    missing = [sid for sid in section_ids if sid not in section_map]
    if missing:
        available = ", ".join(sorted(section_map.keys()))
        return (
            f"[zoom_in] Section ID(s) not found: {', '.join(missing)}\n"
            f"Available sections: {available}"
        )

    # ── Parse with BS4 using lxml backend (same parser as sectioner) ──
    soup = BeautifulSoup(html, "lxml")

    # Build flat element list matching sectioner's dom_index assignment.
    dom_elements = _build_bs4_dom_index(soup)

    if not dom_elements:
        return "[zoom_in] Could not parse HTML — no elements found."

    # ── Locate start/end elements for each section ────────────────
    bs4_sections: list[tuple[Tag, Tag]] = []
    for sid in section_ids:
        section = section_map[sid]
        start_idx = section.start_element.dom_index
        end_idx = section.end_element.dom_index

        if start_idx >= len(dom_elements) or end_idx >= len(dom_elements):
            return (
                f"[zoom_in] DOM index out of range for section '{sid}' "
                f"(start={start_idx}, end={end_idx}, "
                f"total_elements={len(dom_elements)}). "
                f"The page may have changed since the last capture."
            )

        start_tag = dom_elements[start_idx]
        end_tag = dom_elements[end_idx]
        bs4_sections.append((start_tag, end_tag))

    # ── Run extract_sections ──────────────────────────────────────
    result = extract_sections(bs4_sections)

    # ── Prettify and clean ────────────────────────────────────────
    output = result.prettify()

    # Strip the BS4 [document] wrapper — it's not real HTML.
    output = _DOCUMENT_OPEN_RE.sub("", output)
    output = _DOCUMENT_CLOSE_RE.sub("", output)

    # Strip internal detection markers so the agent never sees them.
    output = _DATA_IID_RE.sub("", output)
    output = _DATA_HIDDEN_RE.sub("", output)

    return output.strip() + "\n"
