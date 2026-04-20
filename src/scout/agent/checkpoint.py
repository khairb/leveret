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
from typing import Any


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
        lines.append(
            f"[{cp_id} {label}] url={url} | "
            f'title="{t}" | elements={elems}{dp} | {ts}s'
        )

    lines.append(
        "\nTo inspect any checkpoint's full page state, call: "
        'expand_checkpoint("CP-1")'
    )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  expand_checkpoint — injected into the agent's REPL
# ═══════════════════════════════════════════════════════════════


def create_expand_checkpoint_function(
    run_dir_ref: list[Any],
) -> callable:
    """Create the ``expand_checkpoint(...)`` function for the agent's REPL.

    Uses a mutable ref (single-element list) so the outer loop can point
    it at the latest run directory after each script execution.

    Args:
        run_dir_ref: ``[None]`` initially.  Set ``run_dir_ref[0]`` to the
            :class:`Path` of the current run's checkpoint directory.

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

            label = data.get("label", "?")
            ts = data.get("timestamp_s", 0)
            url = data.get("url", "")
            title = data.get("title", "")
            elems = data.get("element_count", 0)
            text = data.get("visible_text", "")
            preview = data.get("data_preview")

            print(f"\n=== Checkpoint {cp_id} ({label}) at {ts}s ===")
            print(f"URL: {url}")
            print(f"Title: {title}")
            print(f"Elements: {elems}")

            if text:
                print(f"\nPage Text:\n{text}")
            else:
                print("\nPage Text: (empty)")

            if preview is not None:
                try:
                    preview_str = json.dumps(preview, indent=2, default=str)
                except (TypeError, ValueError):
                    preview_str = str(preview)
                print(f"\nData Preview:\n{preview_str}")
            else:
                print("\nData Preview: (none)")

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
    "  • await zoom_section(page, \"section-id\")  — inspect the DOM "
    "HTML of any section\n"
    "\n"
    "checkpoint() is for your final scrape function only. The function "
    "runs in a separate process where you cannot call show_page or "
    "zoom_section. Checkpoints give you that same visibility — each "
    "checkpoint captures the page state at a key moment so that if the "
    "function is rejected, you can call expand_checkpoint(\"CP-1\") to "
    "see what happened.\n"
    "\n"
    "Use checkpoint as a parameter in your scrape function:\n"
    "  async def scrape(page, start_url, checkpoint):\n"
    "      await checkpoint(\"label\")"
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
