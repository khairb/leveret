"""Conversation manager — message formatting, history, and context trimming.

Includes automatic truncation of old ``show_page`` outputs to prevent
context window exhaustion.  Only the most recent page views are kept in
full; older ones are reduced to the first and last few lines with a
clear marker so the agent knows the content was trimmed.
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

    Stores messages in Anthropic's native format and handles trimming
    when the conversation grows too long.
    """

    messages: list[dict] = field(default_factory=list)
    max_messages: int = 80

    # ── Public API ────────────────────────────────────────────────

    def get_messages(self) -> list[dict]:
        """Return messages for the API call, trimmed if needed."""
        if len(self.messages) <= self.max_messages:
            msgs = list(self.messages)
        else:
            # Keep the first message (initial task + page view) and
            # the most recent messages, with a marker in between.
            keep_recent = 60
            first = self.messages[0]
            recent = self.messages[-keep_recent:]

            # Ensure we don't start with an orphaned tool_result.
            # A tool_result user message requires its matching
            # tool_use assistant message in the previous position.
            # Skip forward until we hit an assistant message (which
            # always starts a valid tool_use / tool_result pair).
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
            msgs = [first, marker] + recent

        return _truncate_old_page_views(msgs)

    def add_user_message(self, content: str) -> None:
        """Append a plain-text user message."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: list[dict]) -> None:
        """Append the assistant's response content blocks.

        Args:
            content: The ``content`` list from Anthropic's response
                (may contain text and tool_use blocks).
        """
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_results(self, results: list[dict]) -> None:
        """Append tool results as a user message.

        Args:
            results: List of dicts, each with:
                ``{"type": "tool_result", "tool_use_id": ..., "content": ...}``
        """
        self.messages.append({"role": "user", "content": results})


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


def _truncate_old_page_views(messages: list[dict]) -> list[dict]:
    """Keep only the last N full page views; truncate older ones.

    Scans all tool_result blocks for ``__PAGE_VIEW_START__`` /
    ``__PAGE_VIEW_END__`` markers.  The most recent
    ``_KEEP_RECENT_PAGE_VIEWS`` are left untouched.  Older ones are
    reduced to the first and last ``_KEEP_LINES`` lines with an
    explanatory gap marker.

    Only the messages that need modification are copied; the rest are
    returned by reference so the cache-control prefix stays stable.
    """
    # ── 1. Locate every page view ──────────────────────────────
    #    Each entry is (message_index, block_index_within_content).
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
        return messages

    to_truncate = locations[:-_KEEP_RECENT_PAGE_VIEWS]

    # ── 3. Truncate old page views ─────────────────────────────
    #    Copy only the messages/blocks we need to modify.
    msgs = list(messages)  # shallow copy of the list
    modified_msgs: set[int] = set()

    for msg_idx, block_idx in to_truncate:
        # Ensure we have a mutable copy of this message and its content.
        if msg_idx not in modified_msgs:
            msgs[msg_idx] = dict(msgs[msg_idx])
            msgs[msg_idx]["content"] = list(msgs[msg_idx]["content"])
            modified_msgs.add(msg_idx)

        block = dict(msgs[msg_idx]["content"][block_idx])
        text: str = block["content"]

        start_pos = text.find(_PAGE_VIEW_START)
        end_pos = text.find(_PAGE_VIEW_END)

        # Extract the page view between markers.
        pv_start = start_pos + len(_PAGE_VIEW_START) + 1  # skip marker + \n
        pv_end = end_pos  # up to (not including) end marker
        # Strip trailing newline before end marker if present.
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
            # Short enough — keep as-is.
            truncated_pv = page_view

        # Reconstruct the full tool result content with truncated page view.
        block["content"] = (
            text[:start_pos]
            + _PAGE_VIEW_START + "\n"
            + truncated_pv + "\n"
            + _PAGE_VIEW_END
            + text[end_pos + len(_PAGE_VIEW_END):]
        )
        msgs[msg_idx]["content"][block_idx] = block

    return msgs
