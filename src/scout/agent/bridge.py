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
    from playwright.async_api import Page

    from ..page.converter import RenderedInteractiveElement
    from ..page.manager import PageStateManager
    from ..runtime.environment import ExecutionResult


@dataclass
class ShowPageSectionData:
    """Per-section sidecar data for the context management pipeline."""

    section_id: str
    content: str  # section text as shown to agent
    semantic_role: str = ""
    interactive_count: int = 0
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
    turn_ref: list[int] | None = None,
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
        turn_ref: A single-element list holding the current turn number.
            Used to embed a ``__TURN_N__`` tag in the output so that
            old page views can be stubbed based on age.

    Returns:
        An async callable matching ``async def show_page(page) -> None``.
    """

    async def show_page(page: Page) -> None:
        psm: PageStateManager | None = psm_ref[0]
        if psm is None:
            print("[show_page] Page state manager not initialized yet.")
            return None

        await asyncio.sleep(2)

        state = await psm.capture()
        page_view = psm.get_page_view()

        turn_tag = f"__TURN_{turn_ref[0]}__" if turn_ref else ""
        print("__PAGE_VIEW_START__")
        if turn_tag:
            print(turn_tag)
        print(page_view)
        print("__PAGE_VIEW_END__")

        # Store structured sidecar via shared ref for the agent loop.
        # Do NOT return it — returning non-None would cause REPL
        # double-print of the repr into captured stdout.
        section_data = [
            ShowPageSectionData(
                section_id=s.id,
                content=s.text,
                semantic_role=s.semantic_role,
                interactive_count=s.interactive_count,
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


def create_zoom_section_function(
    psm_ref: list[Any],
    turn_ref: list[int] | None = None,
) -> callable:
    """Create the ``zoom_section(page, ...)`` function injected into the agent's REPL.

    The agent calls ``await zoom_section(page, "section-id")`` to see the
    sanitized HTML structure of a page section — the DOM tags, attributes,
    and stable CSS classes needed to write correct selectors.

    Accepts one or more section IDs as positional arguments.

    Args:
        psm_ref: A single-element list.  Set ``psm_ref[0]`` to the
            :class:`PageStateManager` instance once it's created.
        turn_ref: A single-element list holding the current turn number.
            Used to embed a ``__TURN_N__`` tag in the output so that
            old zoom results can be stubbed based on age.

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
            print(
                "[zoom_section] No section IDs provided. "
                "Pass one or more section IDs from the show_page output."
            )
            return None

        html = psm.zoom_in(*section_ids)
        ids_label = ", ".join(section_ids)
        turn_tag = f"__TURN_{turn_ref[0]}__" if turn_ref else ""
        print(f"__ZOOM_START__|{ids_label}|")
        if turn_tag:
            print(turn_tag)
        print(html)
        print("__ZOOM_END__")
        return None  # Prevent REPL double-print via repr()

    return zoom_section
