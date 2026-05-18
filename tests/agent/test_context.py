"""Tests for conversation context stub collapse (page views & zoom results)."""

from __future__ import annotations

from scout.agent.context import (
    ConversationManager,
    _build_page_view_stub,
    _stub_old_page_views_inplace,
    _stub_old_zoom_results_inplace,
)

# ── Helpers ──────────────────────────────────────────────────


def _make_page_view_result(
    page_state: int,
    url: str = "https://example.com",
    turn: int | None = None,
    sections: list[tuple[str, str, str]] | None = None,
    omitted: int = 0,
) -> dict:
    """Build a tool_result message containing a page view."""
    parts: list[str] = []
    turn_tag = f"__TURN_{turn}__\n" if turn is not None else ""
    parts.append(f"__PAGE_VIEW_START__\n{turn_tag}=== Page State #{page_state} | {url} ===")
    if sections:
        for sid, role, content in sections:
            parts.append(f"\n--- [{sid}] {role} (0 interactive) ---")
            parts.append(content)
    if omitted:
        parts.append(f"\n[{omitted} sections omitted]")
    parts.append("\n__PAGE_VIEW_END__")
    content = "".join(parts)
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": f"t-pv-{page_state}",
                "content": f"Executed in 100ms (step {page_state}).\n\nOutput:\n{content}",
            }
        ],
    }


def _make_zoom_result(
    section_ids: str,
    html: str = "<div>content</div>",
    turn: int | None = None,
) -> dict:
    """Build a tool_result message containing a zoom block."""
    turn_tag = f"__TURN_{turn}__\n" if turn is not None else ""
    content = (
        f"Executed in 50ms.\n\nOutput:\n"
        f"__ZOOM_START__|{section_ids}|\n"
        f"{turn_tag}{html}\n"
        f"__ZOOM_END__"
    )
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "t-zoom",
                "content": content,
            }
        ],
    }


def _make_assistant_msg(text: str = "I see the page.") -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
    }


def _get_tool_result_content(msg: dict) -> str:
    """Extract the content string from a tool_result message."""
    return msg["content"][0]["content"]


# ═══════════════════════════════════════════════════════════════
#  Page View Stub Tests
# ═══════════════════════════════════════════════════════════════


class TestPageViewStubCollapse:
    """Turn-based page view stub collapse."""

    def test_page_view_stubbed_after_5_turns(self):
        """A page view created at turn 1 should be stubbed at turn 6."""
        messages = [
            _make_page_view_result(
                1,
                turn=1,
                sections=[
                    ("nav-main", "navigation", "Home | About"),
                    ("div-content", "content", "<a href='/'>Link</a>"),
                ],
            ),
        ]
        # Turn 5: still within window.
        _stub_old_page_views_inplace(messages, current_turn=5)
        assert "[stub]" not in _get_tool_result_content(messages[0])

        # Turn 6: now old enough.
        _stub_old_page_views_inplace(messages, current_turn=6)
        content = _get_tool_result_content(messages[0])
        assert "[stub]" in content
        assert "Page State #1" in content
        assert "example.com" in content

    def test_page_view_preserved_within_5_turns(self):
        """Page views within the 5-turn window should stay intact."""
        messages = [
            _make_page_view_result(
                5,
                turn=10,
                sections=[
                    ("s1", "content", "Some text"),
                ],
            ),
        ]
        _stub_old_page_views_inplace(messages, current_turn=14)
        content = _get_tool_result_content(messages[0])
        assert "[stub]" not in content
        assert "Some text" in content

    def test_no_turn_tag_treated_as_old(self):
        """Page views without a __TURN_N__ tag should be stubbed."""
        messages = [
            _make_page_view_result(
                1,
                turn=None,
                sections=[
                    ("s1", "content", "Old page"),
                ],
            ),
        ]
        _stub_old_page_views_inplace(messages, current_turn=5)
        assert "[stub]" in _get_tool_result_content(messages[0])

    def test_already_stubbed_not_modified(self):
        """A page view that's already stubbed should not be touched."""
        messages = [
            _make_page_view_result(
                1,
                turn=1,
                sections=[
                    ("s1", "content", "Text"),
                ],
            ),
        ]
        # First stub.
        _stub_old_page_views_inplace(messages, current_turn=10)
        content_after_first = _get_tool_result_content(messages[0])
        assert "[stub]" in content_after_first

        # Second call — should be identical (idempotent).
        _stub_old_page_views_inplace(messages, current_turn=20)
        assert _get_tool_result_content(messages[0]) == content_after_first

    def test_stub_preserves_url(self):
        """The stub should include the full URL from the header."""
        url = "https://www.example-travel.com/s/Berlin?checkin=2026-04-20&adults=2"
        messages = [
            _make_page_view_result(
                9,
                url=url,
                turn=1,
                sections=[
                    ("s1", "content", "Data"),
                ],
            ),
        ]
        _stub_old_page_views_inplace(messages, current_turn=10)
        content = _get_tool_result_content(messages[0])
        assert url in content

    def test_stub_shows_kept_section_ids(self):
        """The stub should list section IDs that had real content."""
        messages = [
            _make_page_view_result(
                1,
                turn=1,
                sections=[
                    ("div-search", "content", "Search bar"),
                    ("div-results", "content", "<a href>Link</a>"),
                    ("nav-footer", "navigation", "[omitted]"),
                ],
                omitted=50,
            ),
        ]
        _stub_old_page_views_inplace(messages, current_turn=10)
        content = _get_tool_result_content(messages[0])
        assert "div-search" in content
        assert "div-results" in content
        # nav-footer had [omitted] as content, should not be in kept list.

    def test_stub_shows_total_section_count(self):
        """The stub should show total sections including omitted ones."""
        messages = [
            _make_page_view_result(
                1,
                turn=1,
                sections=[
                    ("s1", "content", "Text"),
                    ("s2", "content", "Text"),
                ],
                omitted=120,
            ),
        ]
        _stub_old_page_views_inplace(messages, current_turn=10)
        content = _get_tool_result_content(messages[0])
        assert "122 sections" in content

    def test_multiple_page_views_only_old_ones_stubbed(self):
        """Only page views older than 5 turns should be stubbed."""
        messages = [
            _make_page_view_result(
                1,
                turn=1,
                sections=[
                    ("s1", "content", "Old page"),
                ],
            ),
            _make_assistant_msg(),
            _make_page_view_result(
                2,
                turn=5,
                sections=[
                    ("s2", "content", "Recent page"),
                ],
            ),
        ]
        _stub_old_page_views_inplace(messages, current_turn=9)
        # Turn 1 page: age = 9-1 = 8 >= 5 → stubbed.
        assert "[stub]" in _get_tool_result_content(messages[0])
        # Turn 5 page: age = 9-5 = 4 < 5 → kept.
        assert "[stub]" not in _get_tool_result_content(messages[2])
        assert "Recent page" in _get_tool_result_content(messages[2])


# ═══════════════════════════════════════════════════════════════
#  Zoom Stub Tests
# ═══════════════════════════════════════════════════════════════


class TestZoomStubCollapse:
    """Turn-based zoom result stub collapse."""

    def test_zoom_stubbed_after_5_turns(self):
        """A zoom created at turn 2 should be stubbed at turn 7."""
        messages = [
            _make_zoom_result("div-search", turn=2, html="<div>big html</div>"),
        ]
        _stub_old_zoom_results_inplace(messages, current_turn=6)
        assert "Zoom stub" not in _get_tool_result_content(messages[0])

        _stub_old_zoom_results_inplace(messages, current_turn=7)
        content = _get_tool_result_content(messages[0])
        assert "Zoom stub" in content
        assert "div-search" in content

    def test_zoom_preserved_within_5_turns(self):
        """Zoom within window should keep HTML content."""
        messages = [
            _make_zoom_result("s1", turn=10, html="<div>data</div>"),
        ]
        _stub_old_zoom_results_inplace(messages, current_turn=14)
        content = _get_tool_result_content(messages[0])
        assert "<div>data</div>" in content

    def test_zoom_no_turn_tag_treated_as_old(self):
        messages = [
            _make_zoom_result("s1", turn=None, html="<div>old</div>"),
        ]
        _stub_old_zoom_results_inplace(messages, current_turn=5)
        assert "Zoom stub" in _get_tool_result_content(messages[0])

    def test_zoom_already_stubbed_not_modified(self):
        messages = [
            _make_zoom_result("s1", turn=1, html="<div>x</div>"),
        ]
        _stub_old_zoom_results_inplace(messages, current_turn=10)
        first = _get_tool_result_content(messages[0])
        _stub_old_zoom_results_inplace(messages, current_turn=20)
        assert _get_tool_result_content(messages[0]) == first

    def test_zoom_protect_last_msg(self):
        """With protect_last_msg, the last message should never be stubbed."""
        messages = [
            _make_zoom_result("s1", turn=1, html="<div>old</div>"),
        ]
        _stub_old_zoom_results_inplace(
            messages,
            current_turn=10,
            protect_last_msg=True,
        )
        # Should NOT be stubbed because it's the last message.
        assert "<div>old</div>" in _get_tool_result_content(messages[0])

    def test_zoom_preserves_section_ids_in_marker(self):
        """Section IDs should be preserved in the __ZOOM_START__ marker."""
        messages = [
            _make_zoom_result("id-a, id-b", turn=1, html="<div>x</div>"),
        ]
        _stub_old_zoom_results_inplace(messages, current_turn=10)
        content = _get_tool_result_content(messages[0])
        assert "__ZOOM_START__|id-a, id-b|" in content


# ═══════════════════════════════════════════════════════════════
#  ConversationManager Integration
# ═══════════════════════════════════════════════════════════════


class TestConversationManagerStubIntegration:
    """Test that add_tool_results triggers stub collapse with turn number."""

    def test_add_tool_results_stubs_old_page_view(self):
        cm = ConversationManager()
        # Add an old page view at turn 1.
        cm.add_tool_results(
            [_make_page_view_result(1, turn=1)["content"][0]],
            turn=1,
        )
        # Add assistant response.
        cm.add_assistant_message([{"type": "text", "text": "OK"}])
        # Add new tool results at turn 7 — should trigger stub of turn 1 page.
        cm.add_tool_results(
            [{"type": "tool_result", "tool_use_id": "t2", "content": "done"}],
            turn=7,
        )
        # The old page view (message 0) should be stubbed.
        content = _get_tool_result_content(cm.messages[0])
        assert "[stub]" in content

    def test_add_tool_results_preserves_recent(self):
        cm = ConversationManager()
        cm.add_tool_results(
            [_make_page_view_result(1, turn=3)["content"][0]],
            turn=3,
        )
        cm.add_assistant_message([{"type": "text", "text": "OK"}])
        # Turn 7: age = 7-3 = 4 < 5 → should NOT stub.
        cm.add_tool_results(
            [{"type": "tool_result", "tool_use_id": "t2", "content": "done"}],
            turn=7,
        )
        content = _get_tool_result_content(cm.messages[0])
        assert "[stub]" not in content


# ═══════════════════════════════════════════════════════════════
#  _build_page_view_stub Unit Tests
# ═══════════════════════════════════════════════════════════════


class TestBuildPageViewStub:
    """Test the stub builder function directly."""

    def test_basic_stub(self):
        page_view = (
            "=== Page State #3 | https://example.com/page ===\n\n"
            "--- [s1] content (2 interactive) ---\n"
            "Hello world\n\n"
            "--- [s2] navigation (1 interactive) ---\n"
            "[omitted]\n\n"
            "[50 sections omitted]"
        )
        stub = _build_page_view_stub(page_view)
        assert "[stub]" in stub
        assert "Page State #3" in stub
        assert "example.com/page" in stub
        assert "s1" in stub  # kept section (has content)
        assert "52 sections" in stub  # 2 headers + 50 omitted

    def test_no_sections(self):
        page_view = "=== Page State #1 | https://example-social.com ==="
        stub = _build_page_view_stub(page_view)
        assert "[stub]" in stub
        assert "0 sections" in stub


class TestTurnTagPreservation:
    """Ensure __TURN_N__ tag survives show_page filtering."""

    def test_replace_preserves_turn_tag(self):
        """replace_last_show_page_result must keep the turn tag."""
        cm = ConversationManager()
        cm.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": (
                            "Output:\n__PAGE_VIEW_START__\n"
                            "__TURN_5__\n"
                            "=== Page State #1 | https://example.com ===\n"
                            "Full page content here\n"
                            "__PAGE_VIEW_END__"
                        ),
                    }
                ],
            }
        )
        cm.replace_last_show_page_result("Filtered content")
        content = cm.messages[0]["content"][0]["content"]
        assert "__TURN_5__" in content
        assert "Filtered content" in content

    def test_filtered_page_not_stubbed_within_window(self):
        """A filtered page view should NOT be stubbed if within 5 turns."""
        cm = ConversationManager()
        # Add page view at turn 5.
        cm.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": (
                            "Output:\n__PAGE_VIEW_START__\n"
                            "__TURN_5__\n"
                            "=== Page State #1 | https://example.com ===\n"
                            "--- [s1] content (0 interactive) ---\n"
                            "Big content\n"
                            "__PAGE_VIEW_END__"
                        ),
                    }
                ],
            }
        )
        # Filter it (simulates show_page pipeline).
        cm.replace_last_show_page_result(
            "=== Page State #1 | https://example.com ===\n"
            "--- [s1] content (0 interactive) ---\n"
            "Filtered"
        )
        # Add new tool results at turn 9 (age = 4, within window).
        cm.add_tool_results(
            [{"type": "tool_result", "tool_use_id": "t2", "content": "ok"}],
            turn=9,
        )
        content = cm.messages[0]["content"][0]["content"]
        assert "[stub]" not in content
        assert "Filtered" in content

    def test_filtered_page_stubbed_after_window(self):
        """A filtered page view SHOULD be stubbed after 5 turns."""
        cm = ConversationManager()
        cm.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": (
                            "Output:\n__PAGE_VIEW_START__\n"
                            "__TURN_5__\n"
                            "=== Page State #1 | https://example.com ===\n"
                            "--- [s1] content (0 interactive) ---\n"
                            "Big content\n"
                            "__PAGE_VIEW_END__"
                        ),
                    }
                ],
            }
        )
        cm.replace_last_show_page_result(
            "=== Page State #1 | https://example.com ===\n"
            "--- [s1] content (0 interactive) ---\n"
            "Filtered"
        )
        # Turn 10: age = 10 - 5 = 5, should now stub.
        cm.add_tool_results(
            [{"type": "tool_result", "tool_use_id": "t2", "content": "ok"}],
            turn=10,
        )
        content = cm.messages[0]["content"][0]["content"]
        assert "[stub]" in content
