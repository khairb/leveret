"""Conversation manager — message formatting, history, and context trimming.

Page-view context is managed by the show_page 4-phase pipeline
(see ``show_page_context.py``).  Old page views beyond the most recent
two are truncated in-place to keep history compact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── Page-view truncation constants ─────────────────────────────
_PAGE_VIEW_START = "__PAGE_VIEW_START__"
_PAGE_VIEW_END = "__PAGE_VIEW_END__"
_KEEP_RECENT_PAGE_VIEWS = 2
_KEEP_LINES = 10  # first N and last N lines to preserve



@dataclass
class ConversationManager:
    """Manages the LLM conversation history.

    Stores messages in dict format and handles trimming when the
    conversation grows too long.
    """

    messages: list[dict] = field(default_factory=list)
    max_messages: int = 80

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

    def add_tool_results(self, results: list[dict]) -> None:
        """Append tool results as a user message.

        Args:
            results: List of dicts, each with:
                ``{"type": "tool_result", "tool_use_id": ..., "content": ...}``
        """
        self.messages.append({"role": "user", "content": results})
        # Truncate old page views whenever new tool results arrive —
        # this is the only entry point for new page views.
        _truncate_old_page_views_inplace(self.messages)
        self._trim_if_needed()

    def replace_last_show_page_result(self, filtered_content: str) -> None:
        """Replace the page view in the most recent show_page tool result.

        Walks backwards through messages to find the last tool_result
        containing ``__PAGE_VIEW_START__``, then replaces the content
        between the start/end markers with *filtered_content*.
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
                        block["content"] = (
                            text[:start]
                            + _PAGE_VIEW_START + "\n"
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
#  Page-View Truncation
# ═══════════════════════════════════════════════════════════════


def _truncate_old_page_views_inplace(messages: list[dict]) -> None:
    """Truncate old page views in-place, keeping the last N intact.

    Scans all tool_result blocks for ``__PAGE_VIEW_START__`` /
    ``__PAGE_VIEW_END__`` markers.  The most recent
    ``_KEEP_RECENT_PAGE_VIEWS`` are left untouched.  Older ones are
    reduced to the first and last ``_KEEP_LINES`` lines with an
    explanatory gap marker.

    Modifies *messages* in-place so that the truncation is permanent
    and the message prefix stays stable for prompt cache hits.
    """
    # ── 1. Locate every page view ──────────────────────────────
    locations: list[tuple[int, int]] = []

    for i, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for j, block in enumerate(content):
            if (
                block.get("type") == "tool_result"
                and isinstance(block.get("content"), str)
                and _PAGE_VIEW_START in block["content"]
                and _PAGE_VIEW_END in block["content"]
            ):
                locations.append((i, j))

    # ── 2. Nothing to truncate? ────────────────────────────────
    if len(locations) <= _KEEP_RECENT_PAGE_VIEWS:
        return

    to_truncate = locations[:-_KEEP_RECENT_PAGE_VIEWS]

    # ── 3. Truncate old page views in-place ────────────────────
    for msg_idx, block_idx in to_truncate:
        block = messages[msg_idx]["content"][block_idx]
        text: str = block["content"]

        start_pos = text.find(_PAGE_VIEW_START)
        end_pos = text.find(_PAGE_VIEW_END)

        pv_start = start_pos + len(_PAGE_VIEW_START) + 1
        pv_end = end_pos
        if pv_end > 0 and text[pv_end - 1] == "\n":
            pv_end -= 1

        page_view = text[pv_start:pv_end]
        lines = page_view.split("\n")
        total = len(lines)

        if total > _KEEP_LINES * 2:
            first_lines = "\n".join(lines[:_KEEP_LINES])
            last_lines = "\n".join(lines[-_KEEP_LINES:])
            omitted = total - _KEEP_LINES * 2
            truncated_pv = (
                f"{first_lines}\n\n"
                f"[... {omitted} lines omitted — this is an older page view. "
                f"Call show_page(page) to see the current page state. ...]\n\n"
                f"{last_lines}"
            )
        else:
            truncated_pv = page_view

        block["content"] = (
            text[:start_pos]
            + _PAGE_VIEW_START + "\n"
            + truncated_pv + "\n"
            + _PAGE_VIEW_END
            + text[end_pos + len(_PAGE_VIEW_END):]
        )


