"""Bridge helpers that wire PageStateManager into the agent runtime.

Provides:
    - ``create_post_exec_hook`` — no-op stub (keeps the hook extension point).
    - ``create_show_page_function`` — factory for the ``show_page(page)``
      global that the agent calls to capture and print the page view.
    - ``create_zoom_section_function`` — factory for the ``zoom_section(page, ...)``
      global that the agent calls to inspect sanitized HTML of page sections.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..page.manager import PageStateManager
    from ..page.converter import RenderedInteractiveElement
    from playwright.async_api import Page
    from ..runtime.environment import ExecutionResult


@dataclass
class ShowPageSectionData:
    """Per-section sidecar data for the context management pipeline."""

    section_id: str
    content: str  # section text as shown to agent
    interactive_elements: list[RenderedInteractiveElement] = field(
        default_factory=list,
    )


@dataclass
class ShowPageResult:
    """Structured result from a show_page() call.

    Carries both the text the agent sees (``text_output``) and the
    structured sidecar data the context manager needs for filtering.
    """

    text_output: str  # formatted page view (what the agent sees)
    raw_text: str  # plain text for similarity comparison
    sections: list[ShowPageSectionData] = field(default_factory=list)


def create_post_exec_hook(
    psm_ref: list[Any],
) -> callable:
    """Return a no-op post-exec hook.

    Page view capture is now on-demand via ``show_page(page)``.
    The hook stub is kept so the runtime's hook machinery stays wired
    for potential future use (logging, metrics, etc.).
    """

    async def hook(page: Page, result: ExecutionResult) -> None:
        return None

    return hook


def create_show_page_function(
    psm_ref: list[Any],
    result_ref: list[Any],
) -> callable:
    """Create the ``show_page(page)`` function injected into the agent's REPL.

    The agent calls ``await show_page(page)`` after page interactions to
    capture the current page state and print it as sectioned text.

    Args:
        psm_ref: A single-element list.  Set ``psm_ref[0]`` to the
            :class:`PageStateManager` instance once it's created.
        result_ref: A single-element list.  After each call, ``result_ref[0]``
            is set to the :class:`ShowPageResult` so the agent loop can
            access the structured sidecar without relying on the return value.

    Returns:
        An async callable matching ``async def show_page(page) -> None``.
    """

    async def show_page(page: Page) -> None:
        psm: PageStateManager | None = psm_ref[0]
        if psm is None:
            print("[show_page] Page state manager not initialized yet.")
            return None

        import time as _time
        t0 = _time.monotonic()

        await asyncio.sleep(2)
        t_sleep = _time.monotonic()

        state = await psm.capture()
        t_capture = _time.monotonic()

        page_view = psm.get_page_view()
        t_format = _time.monotonic()

        # Print timing breakdown so it appears in captured stdout.
        timings = state.capture_timings if hasattr(state, "capture_timings") else {}
        print("__SHOW_PAGE_TIMING__")
        print(f"  sleep:          {(t_sleep - t0) * 1000:7.0f}ms")
        if timings:
            for label, ms in timings.items():
                print(f"  {label + ':':16s}{ms:7.0f}ms")
        print(f"  capture (total):{(t_capture - t_sleep) * 1000:7.0f}ms")
        print(f"  format_view:    {(t_format - t_capture) * 1000:7.0f}ms")
        print(f"  show_page total:{(t_format - t0) * 1000:7.0f}ms")
        print("__SHOW_PAGE_TIMING_END__")

        print("__PAGE_VIEW_START__")
        print(page_view)
        print("__PAGE_VIEW_END__")

        # Store structured sidecar via shared ref for the agent loop.
        # Do NOT return it — returning non-None would cause REPL
        # double-print of the repr into captured stdout.
        section_data = [
            ShowPageSectionData(
                section_id=s.id,
                content=s.text,
                interactive_elements=s.rendered_interactive_elements,
            )
            for s in state.sections
        ]
        result_ref[0] = ShowPageResult(
            text_output=page_view,
            raw_text=state.full_text,
            sections=section_data,
        )

        return None  # Prevent REPL double-print via repr()

    return show_page


def create_zoom_section_function(psm_ref: list[Any]) -> callable:
    """Create the ``zoom_section(page, ...)`` function injected into the agent's REPL.

    The agent calls ``await zoom_section(page, "section-id")`` to see the
    sanitized HTML structure of a page section — the DOM tags, attributes,
    and stable CSS classes needed to write correct selectors.

    Accepts one or more section IDs as positional arguments.

    Args:
        psm_ref: A single-element list.  Set ``psm_ref[0]`` to the
            :class:`PageStateManager` instance once it's created.

    Returns:
        An async callable matching
        ``async def zoom_section(page, *section_ids) -> None``.
    """

    async def zoom_section(page: Page, *section_ids: str) -> None:
        psm: PageStateManager | None = psm_ref[0]
        if psm is None:
            print("[zoom_section] Page state manager not initialized yet.")
            return None

        if not section_ids:
            print("[zoom_section] No section IDs provided. "
                  "Pass one or more section IDs from the show_page output.")
            return None

        html = psm.zoom_in(*section_ids)
        ids_label = ", ".join(section_ids)
        print(f"__ZOOM_START__|{ids_label}|")
        print(html)
        print("__ZOOM_END__")
        return None  # Prevent REPL double-print via repr()

    return zoom_section
