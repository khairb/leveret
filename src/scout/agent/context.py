"""Conversation manager — message formatting, history, and context trimming.

Page-view context is managed by the show_page 4-phase pipeline
(see ``show_page_context.py``).  Old page views and zoom results are
stubbed after a fixed number of turns to maximise prompt-cache prefix
stability — see ``_STUB_AFTER_TURNS``.

History compression (see ``summarizer.py``) replaces older exploration
messages with a dense sequential summary when the token count exceeds a
model-specific threshold.  Previous summaries and the first user message
are never re-compressed.
"""

from __future__ import annotations

import logging as _logging
import re as _re
from dataclasses import dataclass, field
from typing import Any

_logger = _logging.getLogger(__name__)

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

    # ── History compression ──────────────────────────────────────

    async def compress_history(
        self,
        task_description: str,
        model: str,
    ) -> dict[str, Any] | None:
        """Compress older exploration messages into a sequential summary.

        Identifies the compressible window (between the last summary and
        the 5th-last interaction), calls the summarizer, and replaces the
        window with a notice + summary block.

        Returns a dict with compression metadata (for logging), or
        ``None`` if compression was skipped.
        """
        from .summarizer import estimate_message_tokens, run_summarizer

        messages = self.messages

        # ── Identify boundaries ──────────────────────────────────
        # Protected: index 0 (first user message) and any previous
        # summary blocks.
        last_summary_idx = _find_last_summary_index(messages)

        # Compressible window starts after the last summary (or after
        # message 0 if no summaries exist).
        compress_start = (
            last_summary_idx + 1 if last_summary_idx >= 0 else 1
        )

        # Preserve the last 5 interactions (10 messages).
        # Walk backwards to find 5 complete assistant+user pairs.
        recent_start = _find_recent_window_start(messages, keep_pairs=5)

        # Nothing to compress if the window is too small.
        if recent_start <= compress_start:
            _logger.debug(
                "Compression skipped: recent window (%d) <= compress start (%d)",
                recent_start, compress_start,
            )
            return None

        compressible = messages[compress_start:recent_start]

        if len(compressible) < 4:
            _logger.debug(
                "Compression skipped: only %d compressible messages",
                len(compressible),
            )
            return None

        compressible_token_est = estimate_message_tokens(compressible)
        if compressible_token_est < 3000:
            _logger.debug(
                "Compression skipped: only ~%d tokens estimated",
                compressible_token_est,
            )
            return None

        # ── Protect the latest script + rejection pair ───────────
        # Only the most recent script/rejection pair is preserved
        # verbatim.  Older attempts are summarized — the summarizer
        # captures what was tried and why it failed, so the agent
        # retains that context without paying the full token cost.
        #
        # We walk backwards to find the latest rejection, then check
        # if the preceding message is the script submission.
        protected_indices: set[int] = set()
        for i in range(len(compressible) - 1, -1, -1):
            if _is_rejection_message(compressible[i]):
                protected_indices.add(i)
                if i > 0 and _is_script_submission(compressible[i - 1]):
                    protected_indices.add(i - 1)
                break  # Only the latest pair

        protected_msgs = [
            compressible[i] for i in sorted(protected_indices)
        ]
        summarizable = [
            msg for i, msg in enumerate(compressible)
            if i not in protected_indices
        ]

        if len(summarizable) < 4:
            _logger.debug(
                "Compression skipped: only %d summarizable messages "
                "(after protecting %d)",
                len(summarizable), len(protected_msgs),
            )
            return None

        # Recalculate token estimate on summarizable portion.
        summarizable_token_est = estimate_message_tokens(summarizable)
        if summarizable_token_est < 3000:
            return None

        # ── Infer turn range for the summary header ──────────────
        from .summarizer import _infer_turn_range

        start_turn, end_turn = _infer_turn_range(compressible)

        # ── Run the summarizer ───────────────────────────────────
        _logger.info(
            "Compressing turns %d-%d (%d messages, ~%d tokens)",
            start_turn, end_turn, len(summarizable), summarizable_token_est,
        )

        summary = await run_summarizer(
            summarizable,
            task_description,
            fallback_start_turn=start_turn,
        )

        # ── Replace compressible window with summary ��────────────
        notice = {
            "role": "user",
            "content": (
                f"[Context compressed — earlier exploration "
                f"(turns {start_turn}-{end_turn}) condensed below. "
                f"Your findings are preserved. Re-inspect any page "
                f'state with show_page(page) or '
                f'zoom_section(page, "id").]'
            ),
        }
        summary_msg = {
            "role": "user",
            "content": (
                f"[EXPLORATION SUMMARY — "
                f"Turns {start_turn}-{end_turn}]\n\n{summary}"
            ),
        }

        self.messages = (
            messages[:compress_start]
            + [notice, summary_msg]
            + protected_msgs
            + messages[recent_start:]
        )

        new_token_est = estimate_message_tokens(self.messages)

        meta = {
            "start_turn": start_turn,
            "end_turn": end_turn,
            "messages_compressed": len(compressible),
            "tokens_before_est": compressible_token_est,
            "tokens_after_est": new_token_est,
            "summary_length": len(summary),
            "was_split": summarizable_token_est >= 20_000,
        }

        _logger.info(
            "Compression complete: %d messages → summary "
            "(~%d→~%d tokens est.)",
            len(compressible), compressible_token_est, new_token_est,
        )

        return meta


# ═══════════════════════════════════════════════════════════════
#  Model threshold configuration
# ═══════════════════════════════════════════════════════════════

# ── Dynamic context window lookup ────────────────────────────
#
# We use a public model database JSON (2,600+ models across
# all providers) to look up context windows dynamically.  The
# JSON is fetched once from GitHub and cached locally — zero
# external package dependencies.
#
# If the fetch fails, we fall back to a bundled snapshot and
# ultimately to a conservative default.

_DEFAULT_CONTEXT_WINDOW = 128_000

_MODEL_DB_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/"
    "main/model_prices_and_context_window.json"
)

# Module-level cache — loaded once, reused for the lifetime of
# the process.
_model_db: dict | None = None
_model_db_loaded = False


def _load_model_db() -> dict:
    """Load the model database (fetch from GitHub, cache locally).

    Returns an empty dict if all sources fail — the caller falls
    back to ``_DEFAULT_CONTEXT_WINDOW``.
    """
    global _model_db, _model_db_loaded

    if _model_db_loaded:
        return _model_db or {}

    _model_db_loaded = True
    import json
    from pathlib import Path

    cache_path = Path(__file__).parent / "_model_db_cache.json"

    # 1. Try fetching from GitHub (with short timeout).
    try:
        import urllib.request
        req = urllib.request.Request(
            _MODEL_DB_URL,
            headers={"User-Agent": "scout-agent/1.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        if isinstance(data, dict) and len(data) > 100:
            _model_db = data
            # Update local cache for offline use.
            try:
                cache_path.write_text(
                    json.dumps(data), encoding="utf-8",
                )
            except OSError:
                pass
            _logger.debug(
                "Model DB loaded from GitHub (%d models)",
                len(data),
            )
            return data
    except Exception as exc:
        _logger.debug("GitHub fetch failed: %s", exc)

    # 2. Try local cache.
    try:
        if cache_path.exists():
            data = json.loads(
                cache_path.read_text(encoding="utf-8"),
            )
            if isinstance(data, dict) and len(data) > 100:
                _model_db = data
                _logger.debug(
                    "Model DB loaded from cache (%d models)",
                    len(data),
                )
                return data
    except Exception as exc:
        _logger.debug("Cache load failed: %s", exc)

    # 3. All sources failed.
    _logger.warning(
        "Could not load model database — using default "
        "context window (%d) for all models",
        _DEFAULT_CONTEXT_WINDOW,
    )
    _model_db = {}
    return {}


def _lookup_model_db(name: str) -> int | None:
    """Look up a model's max_input_tokens in the model database.

    Tries the bare name, then with provider prefixes, then with
    date suffixes stripped.  Returns ``None`` if not found.
    """
    db = _load_model_db()
    if not db:
        return None

    # Build candidate keys to try.
    candidates = [name]

    for prefix in ("anthropic/", "openai/", "google/", "mistral/",
                    "groq/", "cohere/", "bedrock/", "azure/"):
        candidates.append(prefix + name)

    # Strip date suffixes (e.g., -20250514).
    stripped = _re.sub(r"-\d{8}$", "", name)
    if stripped != name:
        candidates.append(stripped)
        for prefix in ("anthropic/", "openai/"):
            candidates.append(prefix + stripped)

    for candidate in candidates:
        entry = db.get(candidate)
        if entry and isinstance(entry, dict):
            window = entry.get("max_input_tokens")
            if window and isinstance(window, (int, float)) and window > 0:
                return int(window)

    return None


def _get_context_window(model: str) -> int:
    """Get the context window size for a model.

    Uses a public model database (fetched from GitHub, cached
    locally) for dynamic lookup across all providers.  Falls
    back to a conservative default.
    """
    # Strip our ``provider:model`` prefix.
    name = model.split(":", 1)[-1] if ":" in model else model

    window = _lookup_model_db(name)
    if window is not None:
        return window

    _logger.debug(
        "Could not determine context window for model '%s' — "
        "using default %d",
        model, _DEFAULT_CONTEXT_WINDOW,
    )
    return _DEFAULT_CONTEXT_WINDOW


# ── Trigger fraction (what % of the window triggers compression) ─
#
# Claude models are exceptionally good at long-context reasoning:
# - Sonnet 4 shows <5% accuracy degradation across its full 200K.
# - Opus at 256K scores ~93% on MRCR v2 benchmarks.
# - Claude Code itself triggers compaction at ~83.5% (167K / 200K).
#
# We trigger earlier than Claude Code because our agent performs
# code synthesis from scattered observations — a harder task than
# Q&A.  For non-Claude models we are more conservative since
# long-context quality varies.
#
# Per-model overrides for models we've tested and tuned.
# Models NOT in this dict get a fraction from ``_classify_model``.
_TRIGGER_OVERRIDES: dict[str, float] = {
    # Anthropic — strong long-context.
    "claude-haiku-4-5":    0.65,   # smallest Claude, trigger earlier
    "claude-sonnet-4":     0.75,
    "claude-sonnet-4-5":   0.75,
    "claude-sonnet-4-6":   0.75,
    "claude-opus-4":       0.80,   # best long-context model
    "claude-opus-4-5":     0.80,
    "claude-opus-4-6":     0.80,
    # OpenAI — observed degradation in traces.
    "gpt-4o":              0.40,
    "gpt-4o-mini":         0.20,   # degrades very early (~18K in traces)
    "gpt-4.1":             0.20,
    "gpt-4.1-mini":        0.10,   # small model, large window
    "gpt-4.1-nano":        0.06,   # smallest model, largest window
    "o1":                  0.50,
    "o3":                  0.50,
    "o3-mini":             0.30,
    "o4-mini":             0.30,
}


def _classify_model(model: str) -> float:
    """Classify an unknown model and return a trigger fraction.

    Uses name heuristics to determine model tier:
    - Claude models get higher fractions (better long-context).
    - Models with "mini", "nano", "small", "lite" get lower fractions.
    - Large/flagship models get moderate fractions.
    """
    name = model.lower()

    # Claude models — excellent long-context, use high fractions.
    if "claude" in name:
        if "haiku" in name:
            return 0.65
        if "opus" in name:
            return 0.80
        return 0.75  # sonnet or unknown claude variant

    # Small / mini / nano / lite models — degrade earlier.
    if any(tag in name for tag in ("mini", "nano", "small", "lite")):
        return 0.20

    # Reasoning models (o-series) — moderate.
    if name.startswith("o") and any(c.isdigit() for c in name[:3]):
        return 0.40

    # Default for unknown large models — conservative.
    return 0.40


def _get_trigger_fraction(model: str) -> float:
    """Get the compression trigger fraction for a model.

    Checks per-model overrides first, then classifies by name
    heuristic, with prefix matching for date-suffixed model names.
    """
    name = model.split(":", 1)[-1] if ":" in model else model

    # Exact match.
    if name in _TRIGGER_OVERRIDES:
        return _TRIGGER_OVERRIDES[name]

    # Prefix match (handles date-suffixed names like
    # "claude-sonnet-4-5-20250514").
    for key in sorted(_TRIGGER_OVERRIDES, key=len, reverse=True):
        if name.startswith(key):
            return _TRIGGER_OVERRIDES[key]

    # Classify by name heuristic.
    return _classify_model(name)


def get_compression_threshold(model: str) -> int:
    """Return the token threshold at which compression should trigger.

    Combines dynamic context window lookup with
    model-specific trigger fractions to compute the threshold.

    Args:
        model: Model name in ``provider:model`` or bare format.

    Returns:
        Token count threshold.
    """
    window = _get_context_window(model)
    fraction = _get_trigger_fraction(model)
    return int(window * fraction)


# ═══════════════════════════════════════════════════════════════
#  Compression helpers
# ═══════════════════════════════════════════════════════════════

# Marker prefix used to identify summary messages.
_SUMMARY_MARKER = "[EXPLORATION SUMMARY — "


def _is_summary_message(msg: dict) -> bool:
    """Return True if *msg* is a compression summary block."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content.startswith(_SUMMARY_MARKER)
    return False


def _find_last_summary_index(messages: list[dict]) -> int:
    """Return the index of the last summary message, or -1 if none."""
    for i in range(len(messages) - 1, -1, -1):
        if _is_summary_message(messages[i]):
            return i
    return -1


def _find_recent_window_start(
    messages: list[dict],
    keep_pairs: int = 5,
) -> int:
    """Find the start index that preserves the last *keep_pairs* interactions.

    An interaction is an assistant message + its subsequent user message.
    Walks backwards to find complete pairs and returns the index of the
    earliest message in the recent window.
    """
    pairs_found = 0
    i = len(messages) - 1
    boundary = len(messages)

    while i >= 0 and pairs_found < keep_pairs:
        # Find a user message (the observation half)
        if messages[i].get("role") == "user":
            # Look for the preceding assistant message (the action half)
            if i > 0 and messages[i - 1].get("role") == "assistant":
                pairs_found += 1
                boundary = i - 1
                i -= 2
                continue
        i -= 1

    return boundary


def _is_rejection_message(msg: dict) -> bool:
    """Return True if *msg* is a function rejection from the validator.

    Detects messages generated by ``_build_rejection_message()`` in the
    agent loop.  These are self-contained: they include the full
    rejected script (``**Your function:**``), validator feedback,
    execution output, stderr, exit code, and debugging instructions.
    """
    if msg.get("role") != "user":
        return False
    content = msg.get("content", "")
    if not isinstance(content, str):
        return False
    return (
        "## Function Rejected" in content
        or "VALIDATION RESULT" in content
        or "Script rejected" in content
    )


def _is_script_submission(msg: dict) -> bool:
    """Return True if *msg* is an assistant message submitting a scrape script.

    Detects assistant messages containing ``async def scrape(`` —
    these are the agent's script submissions, distinct from tool
    results that might echo script code.
    """
    if msg.get("role") != "assistant":
        return False
    content = msg.get("content", "")
    if isinstance(content, str):
        return "async def scrape(" in content
    if isinstance(content, list):
        return any(
            "async def scrape(" in block.get("text", "")
            for block in content
            if block.get("type") == "text"
        )
    return False


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
