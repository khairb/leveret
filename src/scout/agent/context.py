"""Conversation manager — message formatting, history, and context trimming.

Page-view context is managed by the show_page 4-phase pipeline
(see ``show_page_context.py``).  Old page views and zoom results are
stubbed after a fixed number of turns to maximise prompt-cache prefix
stability — see ``_STUB_AFTER_TURNS``.
"""

from __future__ import annotations

import re as _re
from dataclasses import dataclass, field
from typing import Any

# ── Markers ───────────────────────────────────────────────────
_PAGE_VIEW_START = "__PAGE_VIEW_START__"
_PAGE_VIEW_END = "__PAGE_VIEW_END__"
_ZOOM_START = "__ZOOM_START__"
_ZOOM_END = "__ZOOM_END__"

# ── Turn-based stub collapse ──────────────────────────────────
_STUB_AFTER_TURNS = 3
_TURN_TAG_RE = _re.compile(r"__TURN_(\d+)__")



@dataclass
class ConversationManager:
    """Manages the LLM conversation history.

    Stores messages in dict format and handles trimming when the
    conversation grows too long.
    """

    messages: list[dict] = field(default_factory=list)
    max_messages: int = 80
    skip_zoom_truncation: bool = False
    skip_page_view_truncation: bool = False

    # ── Public API ────────────────────────────────────────────────

    def get_messages(self) -> list[dict]:
        """Return messages for the API call.

        Trimming now happens in-place via ``_trim_if_needed()`` so that
        the message prefix stays stable between trim events — this
        maximizes prompt cache hit rate.
        """
        return list(self.messages)

    def _trim_if_needed(self) -> None:
        """Permanently drop old messages when the list exceeds the limit.

        Keeps the first message (initial task + page view) and the most
        recent messages, with a marker in between.  Because this modifies
        ``self.messages`` in place, the prefix stays stable until the
        *next* trim event.  This is critical for prompt cache stability:
        a sliding window in ``get_messages()`` would shift every message
        position each turn, breaking the cache prefix.
        """
        if len(self.messages) <= self.max_messages:
            return

        keep_recent = 60
        first = self.messages[0]
        recent = self.messages[-keep_recent:]

        # Ensure we don't start with an orphaned tool_result.
        while recent and _is_tool_result_message(recent[0]):
            recent = recent[1:]

        marker = {
            "role": "user",
            "content": (
                "[Earlier exploration steps omitted. "
                "The above shows your initial task and page view. "
                "Recent steps follow.]"
            ),
        }
        self.messages = [first, marker] + recent

    def add_user_message(self, content: str) -> None:
        """Append a plain-text user message."""
        self.messages.append({"role": "user", "content": content})
        self._trim_if_needed()

    def add_assistant_message(self, content: list[dict]) -> None:
        """Append the assistant's response content blocks.

        Args:
            content: The ``content`` list from the LLM response
                (may contain text and tool_use blocks).
        """
        self.messages.append({"role": "assistant", "content": content})
        self._trim_if_needed()

    def add_tool_results(
        self,
        results: list[dict],
        *,
        turn: int = 0,
    ) -> None:
        """Append tool results as a user message.

        Args:
            results: List of dicts, each with:
                ``{"type": "tool_result", "tool_use_id": ..., "content": ...}``
            turn: The current turn number.  Passed to the stub-collapse
                functions so they can determine message age.
        """
        self.messages.append({"role": "user", "content": results})
        # Stub old page views and zoom results whenever new tool
        # results arrive — this is the only entry point for new content.
        # Zoom truncation protects the just-added message so that all
        # zoom results from the current turn remain visible to the
        # agent before it gets a chance to respond.
        if not self.skip_page_view_truncation:
            _stub_old_page_views_inplace(self.messages, turn)
        if not self.skip_zoom_truncation:
            _stub_old_zoom_results_inplace(
                self.messages, turn, protect_last_msg=True,
            )
        self._trim_if_needed()

    def replace_last_show_page_result(self, filtered_content: str) -> None:
        """Replace the page view in the most recent show_page tool result.

        Walks backwards through messages to find the last tool_result
        containing ``__PAGE_VIEW_START__``, then replaces the content
        between the start/end markers with *filtered_content*.

        Preserves the ``__TURN_N__`` tag so that turn-based stub
        collapse can still determine the page view's age.
        """
        for i in range(len(self.messages) - 1, -1, -1):
            msg = self.messages[i]
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for j, block in enumerate(content):
                if (
                    block.get("type") == "tool_result"
                    and isinstance(block.get("content"), str)
                    and _PAGE_VIEW_START in block["content"]
                ):
                    text = block["content"]
                    start = text.find(_PAGE_VIEW_START)
                    end = text.find(_PAGE_VIEW_END)
                    if start >= 0 and end >= 0:
                        # Preserve __TURN_N__ tag from original content.
                        turn_match = _TURN_TAG_RE.search(
                            text[start:end],
                        )
                        turn_line = (
                            turn_match.group(0) + "\n"
                            if turn_match else ""
                        )
                        block["content"] = (
                            text[:start]
                            + _PAGE_VIEW_START + "\n"
                            + turn_line
                            + filtered_content + "\n"
                            + _PAGE_VIEW_END
                            + text[end + len(_PAGE_VIEW_END):]
                        )
                    return

    def remove_message(self, index: int) -> None:
        """Remove the message at *index*."""
        if 0 <= index < len(self.messages):
            del self.messages[index]


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════


def _is_tool_result_message(msg: dict) -> bool:
    """Return True if *msg* is a user message containing tool_result blocks."""
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(block.get("type") == "tool_result" for block in content)


# ═══════════════════════════════════════════════════════════════
#  Page-View Stub Collapse
# ═══════════════════════════════════════════════════════════════

# Regex to extract the header line:  === Page State #N | URL ===
_PAGE_HEADER_RE = _re.compile(
    r"(=== Page State #\d+\s*\|.*?===)"
)

# Regex to extract kept section IDs from filtered output.
# Matches lines like:  --- [section-id] role (N interactive) ---
# followed by actual content (not "[omitted]").
_SECTION_HEADER_RE = _re.compile(
    r"--- \[([^\]]+)\] .+? ---"
)


def _build_page_view_stub(page_view: str) -> str:
    """Build a compact stub from a page view's content between markers.

    Preserves the URL (in the header) and lists the section IDs that
    had full content (kept sections).  Returns the stub text to place
    between the ``__PAGE_VIEW_START__`` / ``__PAGE_VIEW_END__`` markers.
    """
    # Extract the header line (contains Page State # and URL).
    header_match = _PAGE_HEADER_RE.search(page_view)
    header = header_match.group(1) if header_match else "=== Page State ==="

    # Count total sections (every section header line).
    all_section_ids = _SECTION_HEADER_RE.findall(page_view)
    total_sections = len(all_section_ids)

    # Find kept sections — those followed by content, not "[omitted]".
    # Split on section headers and check what follows each one.
    kept: list[str] = []
    parts = _SECTION_HEADER_RE.split(page_view)
    # parts alternates: [before, id1, after1, id2, after2, ...]
    for i in range(1, len(parts), 2):
        section_id = parts[i]
        content_after = parts[i + 1] if i + 1 < len(parts) else ""
        # A kept section has real content, not just [omitted] or [N sections omitted].
        stripped = content_after.strip().split("\n")[0].strip()
        if stripped and stripped != "[omitted]" and not stripped.startswith("["):
            kept.append(section_id)

    # Also count sections reported as omitted in batch markers.
    omitted_match = _re.findall(r"\[(\d+) sections omitted\]", page_view)
    total_sections += sum(int(n) for n in omitted_match)

    # Build the stub.
    kept_str = ""
    if kept:
        display = kept[:6]
        suffix = ", ..." if len(kept) > 6 else ""
        kept_str = f", {len(kept)} kept: {', '.join(display)}{suffix}"

    return (
        f"{header} [stub]\n"
        f"[{total_sections} sections{kept_str}"
        f" — call show_page(page) for current state.]"
    )


def _stub_old_page_views_inplace(
    messages: list[dict],
    current_turn: int,
) -> None:
    """Stub old page views in-place based on turn age.

    Scans all tool_result blocks for ``__PAGE_VIEW_START__`` /
    ``__PAGE_VIEW_END__`` markers.  If the embedded ``__TURN_N__`` tag
    shows the page view is older than ``_STUB_AFTER_TURNS`` turns,
    it is replaced with a compact stub preserving the URL and section
    summary.  Page views without a turn tag are treated as old.

    Already-stubbed page views (containing ``[stub]``) are skipped.
    """
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                block.get("type") != "tool_result"
                or not isinstance(block.get("content"), str)
            ):
                continue
            text: str = block["content"]
            if _PAGE_VIEW_START not in text or _PAGE_VIEW_END not in text:
                continue
            # Already stubbed — skip.
            if "[stub]" in text:
                continue

            start_pos = text.find(_PAGE_VIEW_START)
            end_pos = text.find(_PAGE_VIEW_END)

            # Extract turn tag (only between markers).
            turn_match = _TURN_TAG_RE.search(text, start_pos, end_pos)
            if turn_match:
                created_turn = int(turn_match.group(1))
                if current_turn - created_turn < _STUB_AFTER_TURNS:
                    continue  # Too recent — keep it.
            # No tag or old enough → stub it.

            pv_start = start_pos + len(_PAGE_VIEW_START) + 1
            pv_end = end_pos
            if pv_end > 0 and text[pv_end - 1] == "\n":
                pv_end -= 1

            page_view = text[pv_start:pv_end]
            stub = _build_page_view_stub(page_view)

            block["content"] = (
                text[:start_pos]
                + _PAGE_VIEW_START + "\n"
                + stub + "\n"
                + _PAGE_VIEW_END
                + text[end_pos + len(_PAGE_VIEW_END):]
            )


# ═══════════════════════════════════════════════════════════════
#  Zoom-Result Stub Collapse
# ═══════════════════════════════════════════════════════════════

# Regex to parse the zoom start marker and extract section IDs.
# Format: __ZOOM_START__|section-id-1, section-id-2|
_ZOOM_START_RE = _re.compile(
    r"__ZOOM_START__\|([^|]*)\|"
)


def _stub_old_zoom_results_inplace(
    messages: list[dict],
    current_turn: int,
    *,
    protect_last_msg: bool = False,
) -> None:
    """Stub old zoom_section results in-place based on turn age.

    Scans all tool_result blocks for ``__ZOOM_START__`` /
    ``__ZOOM_END__`` markers.  If the embedded ``__TURN_N__`` tag
    shows the zoom is older than ``_STUB_AFTER_TURNS`` turns, it is
    replaced with a compact stub listing the zoomed section IDs.
    Zoom blocks without a turn tag are treated as old.

    A single tool_result can contain multiple zoom blocks (if the
    agent called ``zoom_section`` more than once in one code
    execution).  Each block is tracked independently.

    Already-stubbed zoom blocks (containing ``[stub]``) are skipped.

    Args:
        messages: The conversation message list (modified in-place).
        current_turn: The current turn number for age comparison.
        protect_last_msg: When ``True``, the last message in *messages*
            is excluded from scanning.  This prevents zoom results from
            the current turn from being stubbed before the agent has a
            chance to read them.
    """
    scan_end = len(messages) - 1 if protect_last_msg else len(messages)
    for i in range(scan_end):
        msg = messages[i]
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                block.get("type") != "tool_result"
                or not isinstance(block.get("content"), str)
            ):
                continue
            text: str = block["content"]
            if _ZOOM_START not in text or _ZOOM_END not in text:
                continue

            # Process all zoom blocks within this tool result.
            # Work backwards so character offsets stay valid.
            positions: list[tuple[int, int]] = []
            search_from = 0
            while True:
                start = text.find(_ZOOM_START, search_from)
                if start < 0:
                    break
                end = text.find(_ZOOM_END, start)
                if end < 0:
                    break
                positions.append((start, end))
                search_from = end + len(_ZOOM_END)

            for start_pos, end_pos in reversed(positions):
                zoom_block = text[start_pos:end_pos + len(_ZOOM_END)]

                # Already stubbed — skip.
                if "Zoom stub" in zoom_block:
                    continue

                # Check turn age.
                turn_match = _TURN_TAG_RE.search(zoom_block)
                if turn_match:
                    created_turn = int(turn_match.group(1))
                    if current_turn - created_turn < _STUB_AFTER_TURNS:
                        continue  # Too recent — keep it.

                # Extract section IDs from the start marker.
                m = _ZOOM_START_RE.search(text, start_pos)
                section_ids = m.group(1).strip() if m else "unknown"

                # Count lines in the zoom output for the stub.
                first_nl = text.find("\n", start_pos)
                zoom_content = (
                    text[first_nl + 1:end_pos] if first_nl >= 0 else ""
                )
                line_count = zoom_content.count("\n") + 1

                stub = (
                    f"[Zoom stub for sections: {section_ids} — "
                    f"{line_count} lines condensed. "
                    f"Call zoom_section(page, \"section-id\") "
                    f"to re-inspect.]"
                )

                block["content"] = (
                    text[:start_pos]
                    + f"{_ZOOM_START}|{section_ids}|\n"
                    + stub + "\n"
                    + _ZOOM_END
                    + text[end_pos + len(_ZOOM_END):]
                )
                # Re-read text after modification for next iteration.
                text = block["content"]
