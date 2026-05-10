"""Page State Manager

The AI agent's single interface to the browser page.  After every
interaction the agent performs, the host environment calls :meth:`capture`
which runs the full detection → HTML → sectioning pipeline and stores the
result.  The agent then uses :meth:`get_page_view` to see a compact text
representation of the page organised by sections, and :meth:`zoom_in` to
inspect the raw HTML of any section it needs structural detail for.

Usage::

    from scout.page import PageStateManager

    manager = PageStateManager(page)         # page is a Patchright Page
    state = await manager.capture()          # after each interaction
    print(manager.get_page_view())           # agent reads this
    print(manager.zoom_in("nav-main"))       # agent drills in
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Sibling imports (within the page package)
# ---------------------------------------------------------------------------
from .converter import (
    InteractiveElement as ConverterElement,
    html_to_text,
)
from .interactive import (
    InteractiveElement as DetectionElement,
    detect_interactive_elements,
)
from .sanitize import format_html_conservative
from .sectioner import Section, section_page

from .zoom import zoom_in as _zoom_in


# ═══════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PageState:
    """An immutable snapshot of the page at a single point in time.

    Created by :meth:`PageStateManager.capture` — never instantiated
    directly by consumers.
    """

    index: int
    """Sequential state number (0, 1, 2, ...)."""

    url: str
    """Page URL at capture time."""

    timestamp: float
    """Unix timestamp of capture."""

    raw_html: str
    """Original HTML straight from the browser, before sanitization."""

    html: str
    """Sanitized HTML snapshot (markers baked in).  Stored for zoom-in."""

    full_text: str
    """Complete text representation of the page (all sections joined)."""

    sections: list[Section]
    """Ordered, non-overlapping sections produced by the sectioner."""

    section_map: dict[str, Section] = field(repr=False)
    """section.id → Section for O(1) lookup."""

    interactive_elements: list[DetectionElement] = field(repr=False)
    """Interactive elements detected on this page."""

    capture_timings: dict[str, float] = field(default_factory=dict, repr=False)
    """Per-stage timing breakdown (ms) from the capture pipeline."""


# ═══════════════════════════════════════════════════════════════════════════
#  Bridge: detection elements → converter elements
# ═══════════════════════════════════════════════════════════════════════════

def _bridge_elements(
    detection_elements: list[DetectionElement],
) -> list[ConverterElement]:
    """Convert detection InteractiveElements to the converter's format.

    The converter only needs iid, tag, attributes, text, selector.
    Detection adds bounding_box and detected_by which are not needed
    for text conversion.
    """
    return [
        ConverterElement(
            iid=el.iid,
            tag=el.tag,
            attributes=el.attributes,
            text=el.text,
            selector=el.selector,
        )
        for el in detection_elements
    ]


# ═══════════════════════════════════════════════════════════════════════════
#  Page View Formatting
# ═══════════════════════════════════════════════════════════════════════════

def _format_page_view(state: PageState) -> str:
    """Build the formatted text view the agent sees after each capture.

    Structure::

        === Page State #0 | https://example.com ===

        --- [nav-main] navigation (3 interactive) ---
        <a href="/">Home</a> <a href="/about">About</a> ...

        --- [product-list] content (12 interactive) ---
        ...
    """
    parts: list[str] = []
    parts.append(f"=== Page State #{state.index} | {state.url} ===")
    parts.append("")

    for section in state.sections:
        i_count = section.interactive_count
        i_label = "interactive" if i_count != 1 else "interactive"
        header = (
            f"--- [{section.id}] {section.semantic_role} "
            f"({i_count} {i_label}) ---"
        )
        parts.append(header)
        parts.append(section.text)
        parts.append("")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
#  PageStateManager
# ═══════════════════════════════════════════════════════════════════════════

class PageStateManager:
    """Central manager for all page-state operations.

    Created once at the start of a scraping session with a live Patchright
    page object.  After each AI interaction, call :meth:`capture` to
    snapshot the current page.

    Args:
        page: A Patchright ``Page`` object (async API).
        min_size: Minimum section size in characters.
        preferred_max: Target maximum section size.
        hard_max: Absolute maximum section size (safety ceiling).
    """

    def __init__(
        self,
        page,
        *,
        min_size: int = 500,
        preferred_max: int = 15000,
        hard_max: int = 30000,
    ) -> None:
        self._page = page
        self._states: list[PageState] = []
        self._min_size = min_size
        self._preferred_max = preferred_max
        self._hard_max = hard_max

    # ── Properties ────────────────────────────────────────────────────

    @property
    def page(self):
        """The live Patchright page this manager operates on."""
        return self._page

    @page.setter
    def page(self, new_page) -> None:
        """Replace the page (e.g. after script execution for debugging)."""
        self._page = new_page

    @property
    def current_state(self) -> PageState | None:
        """The most recently captured state, or ``None`` if no captures."""
        return self._states[-1] if self._states else None

    @property
    def state_count(self) -> int:
        """Number of states captured so far."""
        return len(self._states)

    # ── Core: capture ─────────────────────────────────────────────────

    async def capture(self) -> PageState:
        """Run the full pipeline and store a new page state.

        Pipeline:
            1. Detect interactive elements (stamps DOM with data-iid)
            2. Capture stamped HTML
            3. Sanitize HTML
            4. Bridge detection elements to converter format
            5. Convert HTML to text (full page)
            6. Section the page
            7. Build and store PageState

        Returns:
            The newly created :class:`PageState`.
        """
        timings: dict[str, float] = {}

        def _ms_since(t: float) -> float:
            return (time.monotonic() - t) * 1000

        # 1. Detect interactive elements.
        t = time.monotonic()
        detection_elements = await detect_interactive_elements(self._page)
        timings["detect"] = _ms_since(t)

        # 2. Get stamped HTML.
        t = time.monotonic()
        raw_html = await self._page.content()
        timings["get_html"] = _ms_since(t)

        # 3. Sanitize HTML — remove scripts, styles, event handlers,
        #    the demo overlay (if present),
        #    unstable CSS classes, framework attributes.  Preserves
        #    data-iid / data-hidden attributes stamped by detection.
        #    truncate_repeating=False because the sectioner handles
        #    repeated patterns itself.
        t = time.monotonic()
        html = format_html_conservative(raw_html, truncate_repeating=False)
        timings["sanitize"] = _ms_since(t)

        # 4. Bridge to converter format.
        converter_elements = _bridge_elements(detection_elements)

        # 5. Full-page text representation.
        t = time.monotonic()
        full_text = html_to_text(html, converter_elements)
        timings["html_to_text"] = _ms_since(t)

        # 6. Section the page.
        t = time.monotonic()
        sections = section_page(
            html,
            converter_elements,
            min_size=self._min_size,
            preferred_max=self._preferred_max,
            hard_max=self._hard_max,
        )
        timings["section"] = _ms_since(t)

        # 7. Build state.
        state = PageState(
            index=len(self._states),
            url=self._page.url,
            timestamp=time.time(),
            raw_html=raw_html,
            html=html,
            full_text=full_text,
            sections=sections,
            section_map={s.id: s for s in sections},
            interactive_elements=detection_elements,
            capture_timings=timings,
        )
        self._states.append(state)
        return state

    # ── Text view ─────────────────────────────────────────────────────

    def get_page_view(self, state_index: int | None = None) -> str:
        """Return the formatted text view the agent sees.

        Args:
            state_index: Which state to view (default: current).

        Returns:
            Formatted string with section headers and text content.
        """
        state = self._resolve_state(state_index)
        if state is None:
            return "[get_page_view] No page state captured yet."
        return _format_page_view(state)

    def get_text(self, state_index: int | None = None) -> str:
        """Return the full text representation (no section markers).

        Args:
            state_index: Which state to view (default: current).

        Returns:
            The complete text representation of the page.
        """
        state = self._resolve_state(state_index)
        if state is None:
            return "[get_text] No page state captured yet."
        return state.full_text

    # ── Zoom-in ───────────────────────────────────────────────────────

    def zoom_in(self, *section_ids: str) -> str:
        """Return contextual HTML for one or more sections.

        Shows the requested sections in full with structural context
        (ancestor elements preserved, everything else replaced by
        placeholders).  Useful for writing CSS selectors or XPaths.

        Args:
            *section_ids: One or more section IDs to zoom into.

        Returns:
            Prettified HTML string, or an error message if a section ID
            is not found.
        """
        state = self.current_state
        if state is None:
            return "[zoom_in] No page state captured yet."
        return _zoom_in(
            html=state.html,
            sections=state.sections,
            section_ids=list(section_ids),
        )

    # ── Listing ───────────────────────────────────────────────────────

    def list_sections(self, state_index: int | None = None) -> list[dict]:
        """Return compact section summaries.

        Each dict contains::

            {
                "id": "nav-main",
                "role": "navigation",
                "interactive_count": 4,
                "char_count": 312,
            }
        """
        state = self._resolve_state(state_index)
        if state is None:
            return []
        return [
            {
                "id": s.id,
                "role": s.semantic_role,
                "interactive_count": s.interactive_count,
                "char_count": len(s.text),
            }
            for s in state.sections
        ]

    def list_interactive_elements(
        self,
        state_index: int | None = None,
    ) -> list[dict]:
        """Return interactive element summaries.

        Each dict contains::

            {
                "iid": 1,
                "tag": "a",
                "text": "Home",
                "selector": "a[href='/']",
                "attributes": {"href": "/"},
            }
        """
        state = self._resolve_state(state_index)
        if state is None:
            return []
        return [
            {
                "iid": el.iid,
                "tag": el.tag,
                "text": el.text[:100] if el.text else "",
                "selector": el.selector,
                "attributes": el.attributes,
            }
            for el in state.interactive_elements
        ]

    # ── Internals ─────────────────────────────────────────────────────

    def _resolve_state(self, state_index: int | None) -> PageState | None:
        """Resolve a state index, defaulting to the current state."""
        if not self._states:
            return None
        if state_index is None:
            return self._states[-1]
        if 0 <= state_index < len(self._states):
            return self._states[state_index]
        return None
