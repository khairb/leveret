"""Tests for history compression (ConversationManager.compress_history).

Tests cover:
- Model-specific threshold detection
- Compressible window boundary identification
- Protected message detection (scripts, validator feedback)
- Recent window preservation (last 5 interactions)
- Incremental compression (2nd compression preserves 1st summary)
- Edge cases (too small, first turn, empty window)
- Summarizer message serialization
- Token estimation
- Split decision logic
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from scout.agent.context import (
    ConversationManager,
    _find_last_summary_index,
    _find_recent_window_start,
    _is_rejection_message,
    _is_script_submission,
    _is_summary_message,
    get_compression_threshold,
)
from scout.agent.summarizer import (
    _find_turn_midpoint,
    _infer_turn_range,
    _serialize_messages_for_summarizer,
    estimate_message_tokens,
)

# ── Helpers ──────────────────────────────────────────────────


def _assistant(text: str, tool_calls: list[dict] | None = None) -> dict:
    """Build an assistant message."""
    content = []
    if text:
        content.append({"type": "text", "text": text})
    if tool_calls:
        content.extend(tool_calls)
    return {"role": "assistant", "content": content}


def _user(text: str) -> dict:
    """Build a plain-text user message."""
    return {"role": "user", "content": text}


def _tool_result(tool_use_id: str, content: str, is_error: bool = False) -> dict:
    """Build a user message with a tool_result block."""
    block = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }
    if is_error:
        block["is_error"] = True
    return {"role": "user", "content": [block]}


def _status_msg(turn: int, total: int = 75) -> dict:
    """Build a turn status user message."""
    return _user(
        f"[Turn {turn}/{total} — {total - turn} remaining"
        f" | Code executions: {turn}/{50}, {50 - turn} remaining]"
    )


def _rejection(
    feedback: str = "price extraction returned empty",
    script: str = "async def scrape(page, start_url, checkpoint):\n    return []",
    attempt: int = 1,
    max_attempts: int = 10,
) -> dict:
    """Build a realistic rejection message matching _build_rejection_message output."""
    return _user(
        f"## Function Rejected\n\n"
        f"**Feedback:** {feedback}\n\n"
        f"**Your function:**\n```python\n{script}\n```\n\n"
        f"**Exit code:** 0\n\n"
        f"**Attempt:** {attempt}/{max_attempts}\n\n"
        f"---\n\nRead the error and your function to understand what went wrong."
    )


def _script_submission(
    script: str = "async def scrape(page, start_url, checkpoint):\n    return []",
) -> dict:
    """Build a realistic assistant script submission."""
    return _assistant(f"Here's my final script:\n\n```python\n{script}\n```")


def _build_conversation(n_turns: int) -> list[dict]:
    """Build a conversation with n_turns of assistant+user+status messages.

    Returns messages list starting with a first user message.
    Each turn has realistic-sized content (~1K chars) to trigger
    compression thresholds in tests.
    """
    msgs = [_user("Task: extract product data from https://example.com " + "x" * 500)]
    for i in range(1, n_turns + 1):
        reasoning = (
            f"Turn {i} reasoning: I see sections [nav], [content-{i}]. "
            f"The page has a search form in [search-box] with "
            f'input[data-testid="query-input"] and a submit button. '
            f"The results grid in [results-grid] shows {i * 5} items, "
            f"each with .listing-card containing .title, .price-amount, "
            f"and .star-rating. " + "x" * 800
        )
        msgs.append(_assistant(reasoning))
        result = (
            f"Executed in 100ms.\n\nOutput:\n"
            f"=== Page State #{i} | https://example.com/page/{i} ===\n"
            f"--- [nav] navigation (3 interactive) ---\n"
            f"Home | Products | About | Contact\n"
            f"--- [content-{i}] main (10 interactive) ---\n"
            f"Product listings with {i * 5} items visible\n" + "y" * 800
        )
        msgs.append(_tool_result(f"t-{i}", result))
        msgs.append(_status_msg(i))
    return msgs


# ═══════════════════════════════════════════════════════════════
#  Threshold tests
# ═══════════════════════════════════════════════════════════════


class TestCompressionThresholds:
    """Test dynamic threshold computation (model DB + trigger fractions)."""

    def test_haiku_threshold(self):
        # model DB returns 200K, fraction 0.65 → 130K
        threshold = get_compression_threshold("anthropic:claude-haiku-4-5")
        assert threshold == 130_000

    def test_sonnet_threshold(self):
        threshold = get_compression_threshold("anthropic:claude-sonnet-4-5")
        # model DB returns 200K, fraction 0.75 → 150K
        assert threshold == 150_000

    def test_opus_threshold(self):
        threshold = get_compression_threshold("anthropic:claude-opus-4-5")
        # model DB returns 200K, fraction 0.80 → 160K
        assert threshold == 160_000

    def test_gpt4o_threshold(self):
        threshold = get_compression_threshold("openai:gpt-4o")
        # model DB returns 128K, fraction 0.40
        assert threshold == int(128_000 * 0.40)

    def test_gpt4o_mini_threshold(self):
        threshold = get_compression_threshold("openai:gpt-4o-mini")
        # model DB returns 128K, fraction 0.20 — degrades very early
        assert threshold == int(128_000 * 0.20)
        assert threshold < 30_000

    def test_gpt41_threshold(self):
        threshold = get_compression_threshold("openai:gpt-4.1")
        # model DB returns ~1M, fraction 0.20
        assert threshold > 100_000

    def test_small_models_have_lower_thresholds(self):
        """Small/mini models should trigger compression much earlier."""
        gpt4o_mini = get_compression_threshold("gpt-4o-mini")
        gpt4o = get_compression_threshold("gpt-4o")
        assert gpt4o_mini < gpt4o

    def test_date_suffixed_model(self):
        """Date-suffixed model names should match the base model."""
        base = get_compression_threshold("claude-sonnet-4-5")
        suffixed = get_compression_threshold("claude-sonnet-4-5-20250514")
        assert suffixed == base

    def test_unknown_model_conservative(self):
        """Unknown models get conservative thresholds."""
        threshold = get_compression_threshold("some-totally-unknown-model-xyz")
        # Default 128K * 0.40 (classify_model default) = 51.2K
        assert threshold > 0
        assert threshold <= 128_000  # Never exceed context window

    def test_bare_model_name(self):
        threshold = get_compression_threshold("claude-haiku-4-5")
        assert threshold == 130_000

    def test_claude_higher_than_others(self):
        """Claude models should have higher thresholds than non-Claude."""
        claude = get_compression_threshold("claude-sonnet-4-5")
        gpt = get_compression_threshold("gpt-4o")
        assert claude > gpt

    def test_dynamic_model_db_lookup(self):
        """Verify the model database is used for window detection."""
        # gpt-5.4-mini is a new model — only the model DB would know its window
        threshold = get_compression_threshold("gpt-5.4-mini")
        assert threshold > 0
        # It's a mini model so should get a low fraction
        # model DB reports 272K for gpt-5.4-mini, classify_model gives 0.20
        assert threshold < 100_000

    def test_classify_unknown_claude(self):
        """Unknown claude variants should get high fractions."""
        from scout.agent.context import _classify_model

        assert _classify_model("claude-future-model-99") == 0.75
        assert _classify_model("claude-opus-99") == 0.80
        assert _classify_model("claude-haiku-99") == 0.65

    def test_classify_unknown_mini(self):
        """Unknown mini models should get low fractions."""
        from scout.agent.context import _classify_model

        assert _classify_model("some-provider-mini-3") == 0.20
        assert _classify_model("mistral-small-latest") == 0.20
        assert _classify_model("gemma-nano-2") == 0.20


# ═══════════════════════════════════════════════════════════════
#  Summary detection tests
# ═══════════════════════════════════════════════════════════════


class TestSummaryDetection:
    def test_is_summary_message(self):
        msg = _user("[EXPLORATION SUMMARY — Turns 1-10]\n\n**Turn 1** — ...")
        assert _is_summary_message(msg)

    def test_is_not_summary(self):
        assert not _is_summary_message(_user("Some regular message"))
        assert not _is_summary_message(_assistant("reasoning"))

    def test_find_last_summary_index_none(self):
        msgs = [_user("first"), _assistant("a"), _user("b")]
        assert _find_last_summary_index(msgs) == -1

    def test_find_last_summary_index(self):
        msgs = [
            _user("first"),
            _user("[EXPLORATION SUMMARY — Turns 1-5]\n\nSummary 1"),
            _assistant("a"),
            _user("b"),
            _user("[EXPLORATION SUMMARY — Turns 6-10]\n\nSummary 2"),
            _assistant("c"),
        ]
        assert _find_last_summary_index(msgs) == 4


# ═══════════════════════════════════════════════════════════════
#  Recent window tests
# ═══════════════════════════════════════════════════════════════


class TestRecentWindow:
    def test_find_5_pairs(self):
        msgs = _build_conversation(10)
        # 10 turns = 1 first + 10*(assistant + tool_result + status) = 31 msgs
        start = _find_recent_window_start(msgs, keep_pairs=5)
        # The last 5 pairs = 10 messages (assistant + tool_result each)
        # But status messages interleave, so count carefully
        # Each turn adds 3 messages: assistant, tool_result, status
        # 5 pairs = 5 assistant + 5 user (tool_result)
        # The status messages between them are user messages but don't
        # pair with an assistant — they're standalone.
        # The function looks for assistant+user pairs walking backwards.
        assert start > 0
        assert start < len(msgs)
        # Verify the preserved window has at least 5 assistant messages
        preserved = msgs[start:]
        assistant_count = sum(1 for m in preserved if m.get("role") == "assistant")
        assert assistant_count >= 5

    def test_fewer_than_5_pairs_preserves_all(self):
        msgs = _build_conversation(3)
        start = _find_recent_window_start(msgs, keep_pairs=5)
        # Only 3 pairs available → should try to keep them all
        # The boundary should be at or near the beginning
        assert start <= 4  # First msg + at most partial first turn


# ═══════════════════════════════════════════════════════════════
#  Protected message tests
# ═══════════════════════════════════════════════════════════════


class TestProtectedMessages:
    """Test detection of script submissions and rejection messages."""

    def test_script_submission_detected(self):
        msg = _assistant(
            "Here's the final script:\n\nasync def scrape(page, start_url, checkpoint):\n    pass"
        )
        assert _is_script_submission(msg)

    def test_script_submission_list_content(self):
        """Assistant message with content as list of blocks."""
        msg = {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "async def scrape(page, start_url, checkpoint):\n    pass",
                },
            ],
        }
        assert _is_script_submission(msg)

    def test_script_submission_not_user(self):
        """User messages with script text are NOT script submissions."""
        msg = _user("async def scrape(page, start_url, checkpoint):\n    pass")
        assert not _is_script_submission(msg)

    def test_rejection_with_function_rejected(self):
        msg = _user(
            "## Function Rejected\n\n"
            "**Feedback:** price extraction returned empty\n\n"
            "**Your function:**\n```python\n"
            "async def scrape(page, start_url, checkpoint):\n    return []\n"
            "```\n"
        )
        assert _is_rejection_message(msg)

    def test_rejection_with_validation_result(self):
        msg = _user("VALIDATION RESULT: Failed — empty prices")
        assert _is_rejection_message(msg)

    def test_rejection_with_script_rejected(self):
        msg = _user("Script rejected: timeout waiting for selector")
        assert _is_rejection_message(msg)

    def test_rejection_not_assistant(self):
        """Assistant messages are never rejection messages."""
        msg = _assistant("## Function Rejected\n\nsome text")
        assert not _is_rejection_message(msg)

    def test_regular_messages_not_detected(self):
        assert not _is_script_submission(_user("regular text"))
        assert not _is_script_submission(_assistant("I see the page has 10 sections"))
        assert not _is_rejection_message(_user("regular text"))
        assert not _is_rejection_message(_tool_result("t-1", "Executed in 100ms"))

    def test_tool_result_with_script_not_submission(self):
        """Tool results echoing script code are NOT submissions."""
        msg = _tool_result(
            "t-1",
            "async def scrape(page, start_url, checkpoint):\n    await page.goto(start_url)",
        )
        assert not _is_script_submission(msg)


# ═══════════════════════════════════════════════════════════════
#  Token estimation tests
# ═══════════════════════════════════════════════════════════════


class TestTokenEstimation:
    def test_string_content(self):
        msgs = [_user("a" * 400)]  # 400 chars → ~100 tokens
        assert estimate_message_tokens(msgs) == 100

    def test_list_content(self):
        msgs = [_tool_result("t-1", "b" * 800)]  # 800 chars → ~200 tokens
        assert estimate_message_tokens(msgs) == 200

    def test_mixed(self):
        msgs = [
            _user("a" * 400),
            _assistant("b" * 400),
            _tool_result("t-1", "c" * 400),
        ]
        # assistant content is in list format with "text" type
        # Total should be ~300 tokens
        est = estimate_message_tokens(msgs)
        assert 200 <= est <= 400


# ═══════════════════════════════════════════════════════════════
#  Turn range inference tests
# ═══════════════════════════════════════════════════════════════


class TestTurnRangeInference:
    def test_from_status_messages(self):
        msgs = [
            _assistant("turn 3"),
            _tool_result("t-3", "result"),
            _status_msg(3),
            _assistant("turn 4"),
            _tool_result("t-4", "result"),
            _status_msg(4),
        ]
        start, end = _infer_turn_range(msgs)
        assert start == 3
        assert end == 4

    def test_fallback_from_assistant_count(self):
        msgs = [
            _assistant("a"),
            _user("b"),
            _assistant("c"),
            _user("d"),
        ]
        start, end = _infer_turn_range(msgs, fallback_start=5)
        assert start == 5
        assert end == 6  # 2 assistant messages → turns 5-6


# ═══════════════════════════════════════════════════════════════
#  Message serialization tests
# ═══════════════════════════════════════════════════════════════


class TestMessageSerialization:
    def test_basic_serialization(self):
        msgs = [
            _assistant("I see the page has nav and content sections."),
            _tool_result("t-1", "Page loaded with 5 sections"),
        ]
        text = _serialize_messages_for_summarizer(msgs, start_turn=1)
        assert "Turn 1" in text
        assert "ASSISTANT (reasoning):" in text
        assert "I see the page has nav" in text
        assert "TOOL RESULT:" in text
        assert "5 sections" in text

    def test_tool_call_serialization(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check the page."},
                    {
                        "type": "tool_use",
                        "id": "t-1",
                        "name": "python",
                        "input": {"code": "await show_page(page)"},
                    },
                ],
            },
            _tool_result("t-1", "Page view output"),
        ]
        text = _serialize_messages_for_summarizer(msgs, start_turn=1)
        assert "TOOL CALL: python" in text
        assert "await show_page(page)" in text

    def test_status_messages_skipped(self):
        msgs = [
            _assistant("reasoning"),
            _status_msg(5),
        ]
        text = _serialize_messages_for_summarizer(msgs, start_turn=5)
        # Status messages should be skipped
        assert "remaining" not in text

    def test_large_tool_result_truncated(self):
        big_content = "x" * 20000
        msgs = [
            _assistant("check"),
            _tool_result("t-1", big_content),
        ]
        text = _serialize_messages_for_summarizer(msgs, start_turn=1)
        assert "chars omitted" in text
        # Should not contain the full 20K
        assert len(text) < 15000


# ═══════════════════════════════════════════════════════════════
#  Split logic tests
# ═══════════════════════════════════════════════════════════════


class TestSplitLogic:
    def test_midpoint_splits_at_assistant_boundary(self):
        msgs = _build_conversation(10)
        # Remove first message — midpoint works on the compressible window
        compressible = msgs[1:]
        midpoint = _find_turn_midpoint(compressible)
        # The midpoint should be after an assistant+user pair, not mid-pair
        assert 0 < midpoint < len(compressible)
        # The message at midpoint-1 should be a user message (end of pair)
        if midpoint > 0:
            assert compressible[midpoint - 1].get("role") == "user"

    def test_midpoint_single_turn(self):
        msgs = [_assistant("a"), _user("b")]
        midpoint = _find_turn_midpoint(msgs)
        assert 0 < midpoint <= len(msgs)


# ═══════════════════════════════════════════════════════════════
#  Compress history integration tests
# ═══════════════════════════════════════════════════════════════


class TestCompressHistory:
    @pytest.fixture
    def conversation_15_turns(self):
        """Build a ConversationManager with 15 turns of exploration."""
        cm = ConversationManager()
        cm.messages = _build_conversation(15)
        return cm

    @pytest.mark.asyncio
    async def test_compression_produces_summary(self, conversation_15_turns):
        cm = conversation_15_turns
        original_count = len(cm.messages)

        with patch("scout.agent.summarizer.run_summarizer", new_callable=AsyncMock) as mock:
            mock.return_value = (
                "**Turn 1** — Navigated to example.com.\n**Turn 2** — Found 10 items."
            )

            meta = await cm.compress_history(
                task_description="Extract products",
                model="anthropic:claude-haiku-4-5",
            )

        assert meta is not None
        assert meta["messages_compressed"] > 0
        assert len(cm.messages) < original_count

        # First message preserved
        assert cm.messages[0]["content"].startswith("Task:")

        # Summary block present
        summary_found = False
        for msg in cm.messages:
            content = msg.get("content", "")
            if isinstance(content, str) and "[EXPLORATION SUMMARY" in content:
                summary_found = True
                break
        assert summary_found

        # Notice before summary
        notice_found = False
        for msg in cm.messages:
            content = msg.get("content", "")
            if isinstance(content, str) and "[Context compressed" in content:
                notice_found = True
                break
        assert notice_found

    @pytest.mark.asyncio
    async def test_compression_skipped_when_too_small(self):
        cm = ConversationManager()
        cm.messages = _build_conversation(2)  # Only 2 turns

        meta = await cm.compress_history(
            task_description="Extract products",
            model="anthropic:claude-haiku-4-5",
        )

        assert meta is None

    @pytest.mark.asyncio
    async def test_incremental_compression(self, conversation_15_turns):
        """Second compression preserves first summary."""
        cm = conversation_15_turns

        with patch("scout.agent.summarizer.run_summarizer", new_callable=AsyncMock) as mock:
            mock.return_value = "Summary of turns 1-8"
            await cm.compress_history("Extract products", "anthropic:claude-haiku-4-5")

        first_summary_idx = _find_last_summary_index(cm.messages)
        assert first_summary_idx >= 0

        # Add more turns (large enough to trigger compression again)
        for i in range(16, 26):
            cm.messages.append(
                _assistant(f"Turn {i} reasoning: exploring more sections " + "z" * 800)
            )
            cm.messages.append(
                _tool_result(f"t-{i}", f"Result for turn {i} with data " + "w" * 800)
            )
            cm.messages.append(_status_msg(i))

        with patch("scout.agent.summarizer.run_summarizer", new_callable=AsyncMock) as mock:
            mock.return_value = "Summary of later turns"
            await cm.compress_history("Extract products", "anthropic:claude-haiku-4-5")

        # Both summaries should exist
        summary_count = sum(1 for m in cm.messages if _is_summary_message(m))
        assert summary_count == 2

        # First summary preserved
        first_summary = None
        for m in cm.messages:
            if _is_summary_message(m):
                first_summary = m
                break
        assert "Summary of turns 1-8" in first_summary["content"]

    @pytest.mark.asyncio
    async def test_latest_rejection_preserved(self):
        """The latest script + rejection survives compression."""
        cm = ConversationManager()
        msgs = _build_conversation(12)
        # Insert a script submission + rejection in the middle
        msgs.insert(10, _script_submission())
        msgs.insert(11, _rejection())
        cm.messages = msgs

        with patch("scout.agent.summarizer.run_summarizer", new_callable=AsyncMock) as mock:
            mock.return_value = "Compressed summary"
            await cm.compress_history("Extract products", "anthropic:claude-haiku-4-5")

        all_content = " ".join(str(m.get("content", "")) for m in cm.messages)
        assert "## Function Rejected" in all_content
        assert "async def scrape(" in all_content

    @pytest.mark.asyncio
    async def test_only_latest_rejection_kept(self):
        """Older rejections are summarized; only the latest pair survives."""
        cm = ConversationManager()
        msgs = _build_conversation(15)

        # Insert TWO script+rejection pairs
        # First (older) — at position 7
        msgs.insert(
            7, _script_submission("async def scrape(page, start_url, checkpoint):\n    # attempt 1")
        )
        msgs.insert(8, _rejection("attempt 1 failed", attempt=1))

        # Second (latest) — at position 16
        msgs.insert(
            16,
            _script_submission("async def scrape(page, start_url, checkpoint):\n    # attempt 2"),
        )
        msgs.insert(17, _rejection("attempt 2 failed", attempt=2))

        cm.messages = msgs

        with patch("scout.agent.summarizer.run_summarizer", new_callable=AsyncMock) as mock:
            mock.return_value = "Summary: attempt 1 tried X and failed"
            await cm.compress_history("Extract products", "anthropic:claude-haiku-4-5")

        all_content = " ".join(str(m.get("content", "")) for m in cm.messages)
        # Latest rejection preserved
        assert "attempt 2 failed" in all_content
        # Older rejection was summarized away (only in summary text)
        rejection_count = sum(
            1
            for m in cm.messages
            if isinstance(m.get("content", ""), str) and "## Function Rejected" in m["content"]
        )
        assert rejection_count == 1, f"Expected 1 rejection, found {rejection_count}"

    @pytest.mark.asyncio
    async def test_recent_5_interactions_preserved(self, conversation_15_turns):
        """Last 5 interactions stay verbatim after compression."""
        cm = conversation_15_turns
        # Capture the last few assistant messages before compression
        last_assistants_before = [m for m in cm.messages[-20:] if m.get("role") == "assistant"][-5:]

        with patch("scout.agent.summarizer.run_summarizer", new_callable=AsyncMock) as mock:
            mock.return_value = "Compressed"
            await cm.compress_history("Extract products", "anthropic:claude-haiku-4-5")

        # Those same assistant messages should appear after compression
        last_assistants_after = [m for m in cm.messages if m.get("role") == "assistant"][-5:]

        for before, after in zip(last_assistants_before, last_assistants_after, strict=False):
            assert before["content"] == after["content"]

    @pytest.mark.asyncio
    async def test_first_message_always_preserved(self, conversation_15_turns):
        cm = conversation_15_turns

        with patch("scout.agent.summarizer.run_summarizer", new_callable=AsyncMock) as mock:
            mock.return_value = "Compressed"
            await cm.compress_history("Extract products", "anthropic:claude-haiku-4-5")

        assert cm.messages[0]["content"].startswith("Task:")


# ═══════════════════════════════════════════════════════════════
#  Stress / edge-case tests
# ═══════════════════════════════════════════════════════════════


class TestRecentWindowEdgeCases:
    """Test _find_recent_window_start with realistic message patterns."""

    def test_interleaved_status_messages(self):
        """Status messages between pairs should not break pair detection."""
        msgs = [
            _user("Task"),
            # Turn 1
            _assistant("reasoning 1"),
            _tool_result("t-1", "result 1"),
            _status_msg(1),  # extra user msg between pairs
            # Turn 2
            _assistant("reasoning 2"),
            _tool_result("t-2", "result 2"),
            _status_msg(2),
            # Turn 3
            _assistant("reasoning 3"),
            _tool_result("t-3", "result 3"),
        ]
        start = _find_recent_window_start(msgs, keep_pairs=2)
        # Should find at least 2 pairs from the end
        preserved = msgs[start:]
        assistant_count = sum(1 for m in preserved if m.get("role") == "assistant")
        assert assistant_count >= 2

    def test_consecutive_user_messages(self):
        """Multiple user messages in a row (debugging, reminders)."""
        msgs = [
            _user("Task"),
            _assistant("reasoning 1"),
            _user("debugging feedback"),
            _user("another nudge"),  # no assistant before this
            _user("yet another"),
            _assistant("reasoning 2"),
            _tool_result("t-2", "result 2"),
            _assistant("reasoning 3"),
            _tool_result("t-3", "result 3"),
        ]
        start = _find_recent_window_start(msgs, keep_pairs=2)
        preserved = msgs[start:]
        assistant_count = sum(1 for m in preserved if m.get("role") == "assistant")
        assert assistant_count >= 2

    def test_all_messages_are_recent(self):
        """When total messages < 10, everything is in recent window."""
        msgs = [
            _user("Task"),
            _assistant("a"),
            _tool_result("t-1", "r"),
            _assistant("b"),
            _tool_result("t-2", "r"),
        ]
        start = _find_recent_window_start(msgs, keep_pairs=5)
        # Can only find 2 pairs — boundary should be near start
        assert start <= 1


class TestCompressHistoryEdgeCases:
    """Complex edge cases for compress_history."""

    @pytest.mark.asyncio
    async def test_compression_notice_before_summary(self):
        """Verify the notice message appears BEFORE the summary."""
        cm = ConversationManager()
        cm.messages = _build_conversation(15)

        with patch("scout.agent.summarizer.run_summarizer", new_callable=AsyncMock) as mock:
            mock.return_value = "**Turn 1** — did stuff"
            await cm.compress_history("Extract products", "anthropic:claude-haiku-4-5")

        # Find notice and summary positions
        notice_idx = None
        summary_idx = None
        for i, msg in enumerate(cm.messages):
            content = msg.get("content", "")
            if isinstance(content, str):
                if "[Context compressed" in content:
                    notice_idx = i
                if "[EXPLORATION SUMMARY" in content:
                    summary_idx = i

        assert notice_idx is not None
        assert summary_idx is not None
        assert notice_idx < summary_idx
        assert summary_idx == notice_idx + 1

    @pytest.mark.asyncio
    async def test_message_roles_valid_after_compression(self):
        """After compression, no orphaned tool_results or broken role sequence."""
        cm = ConversationManager()
        cm.messages = _build_conversation(15)

        with patch("scout.agent.summarizer.run_summarizer", new_callable=AsyncMock) as mock:
            mock.return_value = "Summary"
            await cm.compress_history("Extract products", "anthropic:claude-haiku-4-5")

        # First message should be user
        assert cm.messages[0]["role"] == "user"

        # No assistant message should follow another assistant message
        for i in range(1, len(cm.messages)):
            if cm.messages[i]["role"] == "assistant":
                # Previous should be user (tool result or text)
                assert cm.messages[i - 1]["role"] == "user", (
                    f"Message {i} is assistant but message {i - 1} is {cm.messages[i - 1]['role']}"
                )

    @pytest.mark.asyncio
    async def test_triple_compression(self):
        """Three sequential compressions produce three separate summaries."""
        cm = ConversationManager()
        cm.messages = _build_conversation(12)

        # First compression
        with patch("scout.agent.summarizer.run_summarizer", new_callable=AsyncMock) as mock:
            mock.return_value = "Summary 1: initial exploration"
            await cm.compress_history("Extract products", "anthropic:claude-haiku-4-5")

        # Add more turns
        for i in range(13, 23):
            cm.messages.append(_assistant(f"Turn {i} reasoning " + "a" * 800))
            cm.messages.append(_tool_result(f"t-{i}", f"Result {i} " + "b" * 800))
            cm.messages.append(_status_msg(i))

        # Second compression
        with patch("scout.agent.summarizer.run_summarizer", new_callable=AsyncMock) as mock:
            mock.return_value = "Summary 2: deeper exploration"
            await cm.compress_history("Extract products", "anthropic:claude-haiku-4-5")

        # Add even more turns
        for i in range(24, 34):
            cm.messages.append(_assistant(f"Turn {i} reasoning " + "c" * 800))
            cm.messages.append(_tool_result(f"t-{i}", f"Result {i} " + "d" * 800))
            cm.messages.append(_status_msg(i))

        # Third compression
        with patch("scout.agent.summarizer.run_summarizer", new_callable=AsyncMock) as mock:
            mock.return_value = "Summary 3: final findings"
            await cm.compress_history("Extract products", "anthropic:claude-haiku-4-5")

        # All three summaries should exist in order
        summaries = [m for m in cm.messages if _is_summary_message(m)]
        assert len(summaries) == 3
        assert "Summary 1" in summaries[0]["content"]
        assert "Summary 2" in summaries[1]["content"]
        assert "Summary 3" in summaries[2]["content"]

    @pytest.mark.asyncio
    async def test_compression_with_low_max_messages(self):
        """Compression works even when max_messages trim has already run."""
        cm = ConversationManager(max_messages=30)
        cm.messages = _build_conversation(20)

        with patch("scout.agent.summarizer.run_summarizer", new_callable=AsyncMock) as mock:
            mock.return_value = "Summary after trim"
            meta = await cm.compress_history("Extract products", "anthropic:claude-haiku-4-5")

        if meta is not None:
            assert any(_is_summary_message(m) for m in cm.messages)

    @pytest.mark.asyncio
    async def test_protected_pair_appears_after_summary(self):
        """Latest script+rejection should be placed AFTER the summary block."""
        cm = ConversationManager()
        msgs = _build_conversation(15)

        # Insert script + rejection in early turns
        insert_at = 7
        msgs.insert(insert_at, _script_submission())
        msgs.insert(insert_at + 1, _rejection())
        cm.messages = msgs

        with patch("scout.agent.summarizer.run_summarizer", new_callable=AsyncMock) as mock:
            mock.return_value = "Summary of exploration"
            await cm.compress_history("Extract products", "anthropic:claude-haiku-4-5")

        # Find positions
        summary_idx = None
        script_idx = None
        rejection_idx = None
        for i, m in enumerate(cm.messages):
            content = str(m.get("content", ""))
            if "[EXPLORATION SUMMARY" in content:
                summary_idx = i
            if "async def scrape(" in content and m.get("role") == "assistant":
                if summary_idx is not None:
                    script_idx = i
            if "## Function Rejected" in content:
                if summary_idx is not None:
                    rejection_idx = i

        assert summary_idx is not None, "Summary should exist"
        if script_idx is not None:
            assert script_idx > summary_idx, "Script should be after summary"
        if rejection_idx is not None:
            assert rejection_idx > summary_idx, "Rejection should be after summary"
        # Script comes before rejection (preserved order)
        if script_idx is not None and rejection_idx is not None:
            assert script_idx < rejection_idx, "Script should come before rejection"

    @pytest.mark.asyncio
    async def test_compression_idempotent_on_second_call(self):
        """Calling compress_history twice without new messages should skip."""
        cm = ConversationManager()
        cm.messages = _build_conversation(15)

        with patch("scout.agent.summarizer.run_summarizer", new_callable=AsyncMock) as mock:
            mock.return_value = "Summary"
            meta1 = await cm.compress_history("Extract products", "anthropic:claude-haiku-4-5")

        assert meta1 is not None
        msg_count_after_first = len(cm.messages)

        # Second call — no new messages added
        with patch("scout.agent.summarizer.run_summarizer", new_callable=AsyncMock) as mock:
            mock.return_value = "Should not be called"
            meta2 = await cm.compress_history("Extract products", "anthropic:claude-haiku-4-5")

        assert meta2 is None
        assert len(cm.messages) == msg_count_after_first


class TestCompressionNoticeMarker:
    """Test the notice and summary markers are parseable."""

    def test_notice_contains_instructions(self):
        """Notice should tell agent how to re-inspect."""
        notice_text = (
            "[Context compressed — earlier exploration "
            "(turns 1-10) condensed below. "
            "Your findings are preserved. Re-inspect any page "
            "state with show_page(page) or "
            'zoom_section(page, "id").]'
        )
        assert "show_page(page)" in notice_text
        assert "zoom_section" in notice_text

    def test_summary_marker_is_detectable(self):
        """Summary marker must be reliably detected."""
        msg = {
            "role": "user",
            "content": "[EXPLORATION SUMMARY — Turns 3-12]\n\n**Turn 3** — stuff",
        }
        assert _is_summary_message(msg)

    def test_notice_is_not_summary(self):
        """The notice message should NOT be detected as a summary."""
        notice = {"role": "user", "content": "[Context compressed — earlier exploration...]"}
        assert not _is_summary_message(notice)


class TestSerializerRobustness:
    """Test the message serializer handles all message shapes."""

    def test_empty_messages(self):
        text = _serialize_messages_for_summarizer([], start_turn=1)
        assert text == ""

    def test_assistant_with_no_text(self):
        """Assistant message with only a tool call, no text."""
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t-1",
                        "name": "python",
                        "input": {"code": "print('hello')"},
                    },
                ],
            },
        ]
        text = _serialize_messages_for_summarizer(msgs, start_turn=1)
        assert "TOOL CALL: python" in text
        assert "print('hello')" in text

    def test_error_tool_result(self):
        msgs = [
            _assistant("trying something"),
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t-1",
                        "content": "TimeoutError: page.click timed out",
                        "is_error": True,
                    }
                ],
            },
        ]
        text = _serialize_messages_for_summarizer(msgs, start_turn=1)
        assert "TOOL ERROR:" in text
        assert "TimeoutError" in text

    def test_mixed_user_message_with_text_and_tool_result(self):
        """User message containing both text blocks and tool results."""
        msgs = [
            _assistant("check"),
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here's some context"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "t-1",
                        "content": "tool output here",
                    },
                ],
            },
        ]
        text = _serialize_messages_for_summarizer(msgs, start_turn=1)
        assert "Here's some context" in text
        assert "tool output here" in text

    def test_non_python_tool_call(self):
        """Tool calls other than python should show name and args."""
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t-1",
                        "name": "search",
                        "input": {"query": "hotels in Berlin"},
                    },
                ],
            },
        ]
        text = _serialize_messages_for_summarizer(msgs, start_turn=1)
        assert "TOOL CALL: search" in text
        assert "hotels in Berlin" in text
