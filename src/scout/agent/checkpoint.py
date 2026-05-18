"""Checkpoint observability for the scraping agent.

The engine wrapper embeds a checkpoint function that writes CP-*.json
files to a run directory.  After execution, the outer loop reads these
files and (on rejection) lets the agent expand any checkpoint to see
what the page looked like at that point.

Directory layout::

    {base_dir}/
      run_1/          # first execution
        CP-1.json
        CP-2.json
      run_2/          # second attempt (after rejection)
        CP-1.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .bridge import ShowPageResult


def read_checkpoints(run_dir: Path) -> list[dict]:
    """Read all checkpoint JSON files from *run_dir*, sorted by ID.

    Returns an empty list if the directory doesn't exist or has no
    checkpoint files.
    """
    if not run_dir.is_dir():
        return []

    files = sorted(
        run_dir.glob("CP-*.json"),
        key=lambda p: int(re.search(r"CP-(\d+)", p.name).group(1)),  # type: ignore[union-attr]
    )
    checkpoints: list[dict] = []
    for f in files:
        try:
            checkpoints.append(json.loads(f.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return checkpoints


def format_checkpoint_summary(checkpoints: list[dict]) -> str:
    """Format checkpoints into a compact summary for rejection messages.

    Example output::

        ## Checkpoints (3 captured)

        [CP-1 navigated] url=https://... | title="..." | elements=142 | 0.8s
        [CP-2 consent]   url=https://... | title="..." | elements=142 | 1.2s

        To inspect a checkpoint's full page state, call:
        expand_checkpoint("CP-1")
    """
    if not checkpoints:
        return ""

    lines = [f"## Checkpoints ({len(checkpoints)} captured)\n"]
    for cp in checkpoints:
        cp_id = cp.get("id", "?")
        label = cp.get("label", "?")
        url = cp.get("url", "")
        title = cp.get("title", "")
        elems = cp.get("element_count", 0)
        ts = cp.get("timestamp_s", 0)

        t = (title[:50] + "\u2026") if len(title) > 50 else title
        dp = ""
        if cp.get("data_preview"):
            dp = f" | data_preview={len(cp['data_preview'])} items"
        lines.append(f'[{cp_id} {label}] url={url} | title="{t}" | elements={elems}{dp} | {ts}s')

    lines.append('\nTo inspect any checkpoint\'s full page state, call: expand_checkpoint("CP-1")')
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  expand_checkpoint — injected into the agent's REPL
# ═══════════════════════════════════════════════════════════════


def _build_checkpoint_page_view(
    cp_id: str,
    data: dict,
) -> ShowPageResult:
    """Build a sectioned page view from a checkpoint's stored data.

    If the checkpoint has stored HTML, runs the sanitizer + sectioner to
    produce the same format as ``show_page``.  Falls back to a single
    section wrapping ``visible_text`` for old-format checkpoints.

    Returns a :class:`ShowPageResult` ready for the filtering pipeline.
    """
    from ..page.sanitize import format_html_conservative
    from ..page.sectioner import section_page
    from .bridge import ShowPageResult, ShowPageSectionData

    label = data.get("label", "?")
    ts = data.get("timestamp_s", 0)
    url = data.get("url", "")
    raw_html = data.get("html", "")

    header = f"=== Checkpoint {cp_id} ({label}) at {ts}s | {url} ==="

    if raw_html:
        sanitized = format_html_conservative(
            raw_html,
            truncate_repeating=False,
        )
        sections = section_page(sanitized, None)
    else:
        sections = []

    if sections:
        parts: list[str] = [header, ""]
        section_data: list[ShowPageSectionData] = []
        for s in sections:
            i_count = s.interactive_count
            h = f"--- [{s.id}] {s.semantic_role} ({i_count} interactive) ---"
            parts.append(h)
            parts.append(s.text)
            parts.append("")
            section_data.append(
                ShowPageSectionData(
                    section_id=s.id,
                    content=s.text,
                    semantic_role=s.semantic_role,
                    interactive_count=i_count,
                ),
            )
        text_output = "\n".join(parts).rstrip()
        raw_text = "\n".join(s.text for s in sections)
    else:
        # Fallback: wrap visible_text as a single section.
        visible_text = data.get("visible_text", "")
        fallback_id = f"checkpoint-{cp_id.lower()}"
        h = f"--- [{fallback_id}] content (0 interactive) ---"
        text_output = f"{header}\n\n{h}\n{visible_text}".rstrip()
        raw_text = visible_text
        section_data = [
            ShowPageSectionData(
                section_id=fallback_id,
                content=visible_text,
            ),
        ]

    # Append data_preview as a synthetic section if present.
    preview = data.get("data_preview")
    if preview is not None:
        try:
            preview_str = json.dumps(preview, indent=2, default=str)
        except (TypeError, ValueError):
            preview_str = str(preview)
        preview_id = "data-preview"
        h = f"--- [{preview_id}] data (0 interactive) ---"
        text_output += f"\n\n{h}\n{preview_str}"
        section_data.append(
            ShowPageSectionData(
                section_id=preview_id,
                content=preview_str,
            ),
        )

    return ShowPageResult(
        text_output=text_output,
        raw_text=raw_text,
        sections=section_data,
    )


def create_expand_checkpoint_function(
    run_dir_ref: list[Any],
    result_ref: list[Any],
    turn_ref: list[int] | None = None,
) -> callable:
    """Create the ``expand_checkpoint(...)`` function for the agent's REPL.

    Uses mutable refs (single-element lists) so the outer loop can point
    them at the latest state after each script execution.

    The output uses the same ``__PAGE_VIEW_START__`` / ``__PAGE_VIEW_END__``
    markers as ``show_page``, so the loop's analysis-then-filter pipeline
    handles it automatically.

    Args:
        run_dir_ref: ``[None]`` initially.  Set ``run_dir_ref[0]`` to the
            :class:`Path` of the current run's checkpoint directory.
        result_ref: Shared ``[None]`` ref — same one used by ``show_page``.
            After expansion, ``result_ref[0]`` is set to a
            :class:`ShowPageResult` for the filtering pipeline.
        turn_ref: A single-element list holding the current turn number.
            Used to embed a ``__TURN_N__`` tag in the output for
            turn-based stub collapse.

    Returns:
        A sync callable: ``def expand_checkpoint(*checkpoint_ids) -> None``.
    """

    def expand_checkpoint(*checkpoint_ids: str) -> None:
        run_dir: Path | None = run_dir_ref[0]
        if run_dir is None:
            print(
                "[expand_checkpoint] No checkpoints available — no final "
                "script has been executed yet.\n"
                "\n"
                "Checkpoints are captured during final script execution, "
                "not during interactive exploration. In your live Python "
                "environment you already have full observability via "
                "show_page(page) and zoom_section(page, ...). "
                "Checkpoints exist to give you that same visibility into "
                "the final script, which runs in a separate process where "
                "you cannot interact with the page.\n"
                "\n"
                "Once a final script runs and is rejected, you can call "
                'expand_checkpoint("CP-1") to inspect what the page '
                "looked like at each checkpoint the script recorded."
            )
            return None

        if not checkpoint_ids:
            print(
                "[expand_checkpoint] No checkpoint IDs provided. "
                'Pass one or more IDs, e.g. expand_checkpoint("CP-1")'
            )
            return None

        # Build a combined ShowPageResult across all requested checkpoints.
        all_sections: list = []
        all_text_parts: list[str] = []
        all_raw_parts: list[str] = []

        for cp_id in checkpoint_ids:
            path = run_dir / f"{cp_id}.json"
            if not path.exists():
                print(f"[expand_checkpoint] {cp_id} not found in {run_dir}")
                continue

            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                print(f"[expand_checkpoint] Error reading {cp_id}: {exc}")
                continue

            sp = _build_checkpoint_page_view(cp_id, data)
            all_sections.extend(sp.sections)
            all_text_parts.append(sp.text_output)
            all_raw_parts.append(sp.raw_text)

        if not all_text_parts:
            return None

        combined_text = "\n\n".join(all_text_parts)

        # Use the same markers as show_page so the loop picks it up.
        turn_tag = f"__TURN_{turn_ref[0]}__" if turn_ref else ""
        print("__PAGE_VIEW_START__")
        if turn_tag:
            print(turn_tag)
        print(combined_text)
        print("__PAGE_VIEW_END__")

        # Populate the shared sidecar ref for the filtering pipeline.
        from .bridge import ShowPageResult

        result_ref[0] = ShowPageResult(
            text_output=combined_text,
            raw_text="\n".join(all_raw_parts),
            sections=all_sections,
        )

        return None  # Prevent REPL double-print via repr()

    return expand_checkpoint


# ═══════════════════════════════════════════════════════════════
#  checkpoint guard — injected into the agent's REPL
# ═══════════════════════════════════════════════════════════════

_CHECKPOINT_GUARD_MESSAGE = (
    "[checkpoint] You are calling checkpoint() in your live interactive "
    "environment — this is not needed here.\n"
    "\n"
    "In this environment you already have full page observability:\n"
    "  • await show_page(page)  — see the full page as sectioned text\n"
    '  • await zoom_section(page, "section-id")  — inspect the DOM '
    "HTML of any section\n"
    "\n"
    "checkpoint() is for your final scrape function only. The function "
    "runs in a separate process where you cannot call show_page or "
    "zoom_section. Checkpoints give you that same visibility — each "
    "checkpoint captures the page state at a key moment so that if the "
    'function is rejected, you can call expand_checkpoint("CP-1") to '
    "see what happened.\n"
    "\n"
    "Use checkpoint as a parameter in your scrape function:\n"
    "  async def scrape(page, start_url, checkpoint):\n"
    '      await checkpoint("label")'
)


def create_checkpoint_guard() -> callable:
    """Create a ``checkpoint(...)`` guard injected into the agent's REPL.

    If the agent tries to call ``await checkpoint(page, ...)`` during
    interactive exploration, this prints an educational message explaining
    that checkpoints are for the final script only.

    Returns:
        An async callable matching
        ``async def checkpoint(page, label, **kwargs) -> None``.
    """

    async def checkpoint(*args: Any, **kwargs: Any) -> None:
        print(_CHECKPOINT_GUARD_MESSAGE)
        return None

    return checkpoint
