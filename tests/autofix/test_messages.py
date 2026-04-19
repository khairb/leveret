"""Tests for the error message formatter (Phase 4).

Tests verify that formatted messages match spec S12 templates:
  - Anti-bot blocked message
  - Server error blocked message
  - Conservative mode declined D/E message
  - Stability too noisy (MIXED/CHAOTIC) message
  - Regeneration triggered message
  - Required fields present in all messages
  - User action suggestions present
"""

from __future__ import annotations

import pytest

from scout.autofix.diagnosis import (
    format_diagnosis_message,
    _category_label,
    _stability_label,
    _page_summary,
)
from scout.autofix.types import (
    AttemptResult,
    AutoFixAction,
    AutoFixMode,
    ErrorCategory,
    PageSignals,
    PageVerificationResult,
    StabilityLevel,
)


# -- Helpers ---------------------------------------------------------------


def _make_attempts(
    n: int = 3,
    error: str = "AttributeError: 'NoneType' has no attribute 'text_content'",
) -> list[AttemptResult]:
    """Create n failed AttemptResult objects."""
    return [
        AttemptResult(success=False, error=error)
        for _ in range(n)
    ]


# -- Test: Anti-bot blocked message ----------------------------------------


class TestAntibotMessage:
    """S12: Anti-bot detected -> specific message template."""

    def test_antibot_message_contains_key_phrases(self):
        msg = format_diagnosis_message(
            action=AutoFixAction.RAISE,
            reason="Anti-bot detected in 3/3 attempts — regeneration blocked",
            category=ErrorCategory.B,
            stability=StabilityLevel.STABLE,
            page_results=[PageVerificationResult.ANTI_BOT] * 3,
            attempts=_make_attempts(),
            mode=AutoFixMode.BALANCED,
        )
        assert "Cached script failed" in msg
        assert "Anti-bot detected" in msg
        assert "not real content" in msg
        assert "anti-bot system" in msg
        assert "IP/proxy" in msg

    def test_antibot_message_no_regenerate_suggestion(self):
        """Anti-bot messages suggest IP/proxy, not regenerate=True."""
        msg = format_diagnosis_message(
            action=AutoFixAction.RAISE,
            reason="Anti-bot detected",
            category=ErrorCategory.B,
            stability=StabilityLevel.STABLE,
            page_results=[PageVerificationResult.ANTI_BOT] * 3,
            attempts=_make_attempts(),
            mode=AutoFixMode.AGGRESSIVE,
        )
        # Should not suggest regenerate=True for anti-bot (useless)
        assert "regenerate=True" not in msg


# -- Test: Server error blocked message ------------------------------------


class TestServerErrorMessage:
    """S12: Server error detected -> specific message template."""

    def test_server_error_message_contains_key_phrases(self):
        msg = format_diagnosis_message(
            action=AutoFixAction.RAISE,
            reason="Insufficient page verification",
            category=ErrorCategory.D,
            stability=StabilityLevel.STABLE,
            page_results=[
                PageVerificationResult.REAL_PAGE,
                PageVerificationResult.SERVER_ERROR,
                PageVerificationResult.REAL_PAGE,
            ],
            attempts=_make_attempts(
                error="Page.wait_for_selector: Timeout 30000ms exceeded.",
            ),
            mode=AutoFixMode.BALANCED,
        )
        assert "Cached script failed" in msg
        assert "Server error detected" in msg
        assert "server returned errors" in msg
        assert "regenerate=True" in msg

    def test_server_error_page_summary(self):
        msg = format_diagnosis_message(
            action=AutoFixAction.RAISE,
            reason="test",
            category=ErrorCategory.B,
            stability=StabilityLevel.STABLE,
            page_results=[
                PageVerificationResult.SERVER_ERROR,
                PageVerificationResult.SERVER_ERROR,
                PageVerificationResult.REAL_PAGE,
            ],
            attempts=_make_attempts(),
            mode=AutoFixMode.BALANCED,
        )
        assert "Server error detected (2/3 attempts)" in msg


# -- Test: Conservative mode declined D/E ---------------------------------


class TestConservativeDeclinedMessage:
    """S12: Conservative mode declines D/E -> specific message template."""

    def test_conservative_d_message(self):
        msg = format_diagnosis_message(
            action=AutoFixAction.RAISE,
            reason="conservative mode does not regenerate ambiguous categories",
            category=ErrorCategory.D,
            stability=StabilityLevel.STABLE,
            page_results=[PageVerificationResult.REAL_PAGE] * 3,
            attempts=_make_attempts(
                error="Page.wait_for_selector: Timeout 30000ms exceeded.",
            ),
            mode=AutoFixMode.CONSERVATIVE,
        )
        assert "conservative" in msg
        assert "selector timeout" in msg.lower()
        assert "regenerate=True" in msg
        assert 'auto_fix="balanced"' in msg or 'auto_fix="aggressive"' in msg

    def test_conservative_e_message(self):
        msg = format_diagnosis_message(
            action=AutoFixAction.RAISE,
            reason="conservative mode does not regenerate ambiguous categories",
            category=ErrorCategory.E,
            stability=StabilityLevel.STABLE,
            page_results=[PageVerificationResult.REAL_PAGE] * 3,
            attempts=_make_attempts(
                error="<div> intercepts pointer events",
            ),
            mode=AutoFixMode.CONSERVATIVE,
        )
        assert "conservative" in msg
        assert "page state error" in msg.lower()


# -- Test: Stability too noisy message ------------------------------------


class TestStabilityNoisyMessage:
    """S12: MIXED/CHAOTIC stability -> specific message template."""

    def test_mixed_stability_message(self):
        msg = format_diagnosis_message(
            action=AutoFixAction.RAISE,
            reason="Failure pattern is mixed",
            category=ErrorCategory.D,
            stability=StabilityLevel.MIXED,
            page_results=[PageVerificationResult.REAL_PAGE] * 3,
            attempts=_make_attempts(),
            mode=AutoFixMode.BALANCED,
        )
        assert "mixed pattern" in msg
        assert "inconsistent" in msg
        assert "environmental" in msg
        assert "regenerate=True" in msg

    def test_chaotic_stability_message(self):
        msg = format_diagnosis_message(
            action=AutoFixAction.RAISE,
            reason="Failure pattern is chaotic",
            category=ErrorCategory.B,
            stability=StabilityLevel.CHAOTIC,
            page_results=[PageVerificationResult.REAL_PAGE] * 3,
            attempts=_make_attempts(),
            mode=AutoFixMode.AGGRESSIVE,
        )
        assert "chaotic pattern" in msg
        assert "inconsistent" in msg


# -- Test: Regeneration triggered message ----------------------------------


class TestRegenerateMessage:
    """S12: Regeneration triggered -> specific message template."""

    def test_regenerate_message_contains_mode(self):
        msg = format_diagnosis_message(
            action=AutoFixAction.REGENERATE,
            reason="Script fault confirmed — stable failure pattern",
            category=ErrorCategory.B,
            stability=StabilityLevel.STABLE,
            page_results=[PageVerificationResult.REAL_PAGE] * 3,
            attempts=_make_attempts(),
            mode=AutoFixMode.BALANCED,
        )
        assert "regenerating" in msg.lower()
        assert "balanced" in msg

    def test_regenerate_message_contains_reason(self):
        msg = format_diagnosis_message(
            action=AutoFixAction.REGENERATE,
            reason="Script fault confirmed — stable failure pattern",
            category=ErrorCategory.B,
            stability=StabilityLevel.STABLE,
            page_results=[PageVerificationResult.REAL_PAGE] * 3,
            attempts=_make_attempts(),
            mode=AutoFixMode.BALANCED,
        )
        assert "stable failure" in msg

    def test_regenerate_no_regenerate_suggestion(self):
        """Regeneration messages don't need regenerate=True suggestions."""
        msg = format_diagnosis_message(
            action=AutoFixAction.REGENERATE,
            reason="Script fault",
            category=ErrorCategory.B,
            stability=StabilityLevel.STABLE,
            page_results=[PageVerificationResult.REAL_PAGE] * 3,
            attempts=_make_attempts(),
            mode=AutoFixMode.BALANCED,
        )
        assert "regenerate=True" not in msg


# -- Test: Required fields in all messages ---------------------------------


class TestRequiredFields:
    """All messages must contain: category, attempt count, page summary."""

    @pytest.mark.parametrize("action", [AutoFixAction.REGENERATE, AutoFixAction.RAISE])
    def test_category_label_present(self, action):
        msg = format_diagnosis_message(
            action=action,
            reason="test reason",
            category=ErrorCategory.B,
            stability=StabilityLevel.STABLE,
            page_results=[PageVerificationResult.REAL_PAGE] * 3,
            attempts=_make_attempts(),
            mode=AutoFixMode.BALANCED,
        )
        assert "Category:" in msg
        assert "Runtime crash" in msg

    @pytest.mark.parametrize("action", [AutoFixAction.REGENERATE, AutoFixAction.RAISE])
    def test_attempt_count_present(self, action):
        msg = format_diagnosis_message(
            action=action,
            reason="test",
            category=ErrorCategory.B,
            stability=StabilityLevel.STABLE,
            page_results=[PageVerificationResult.REAL_PAGE] * 3,
            attempts=_make_attempts(),
            mode=AutoFixMode.BALANCED,
        )
        assert "3/3 failed" in msg

    @pytest.mark.parametrize("action", [AutoFixAction.REGENERATE, AutoFixAction.RAISE])
    def test_page_summary_present(self, action):
        msg = format_diagnosis_message(
            action=action,
            reason="test",
            category=ErrorCategory.B,
            stability=StabilityLevel.STABLE,
            page_results=[PageVerificationResult.REAL_PAGE] * 3,
            attempts=_make_attempts(),
            mode=AutoFixMode.BALANCED,
        )
        assert "Page:" in msg
        assert "Verified real" in msg

    def test_error_summary_present(self):
        msg = format_diagnosis_message(
            action=AutoFixAction.RAISE,
            reason="test",
            category=ErrorCategory.B,
            stability=StabilityLevel.STABLE,
            page_results=[PageVerificationResult.REAL_PAGE] * 3,
            attempts=_make_attempts(
                error="AttributeError: 'NoneType' has no attribute 'text_content'",
            ),
            mode=AutoFixMode.BALANCED,
        )
        assert "Error:" in msg
        assert "text_content" in msg

    def test_stability_label_present_when_available(self):
        msg = format_diagnosis_message(
            action=AutoFixAction.RAISE,
            reason="test",
            category=ErrorCategory.B,
            stability=StabilityLevel.STABLE,
            page_results=[PageVerificationResult.REAL_PAGE] * 3,
            attempts=_make_attempts(),
            mode=AutoFixMode.BALANCED,
        )
        assert "stable pattern" in msg

    def test_no_stability_when_none(self):
        msg = format_diagnosis_message(
            action=AutoFixAction.RAISE,
            reason="test",
            category=ErrorCategory.A,
            stability=None,
            page_results=[],
            attempts=_make_attempts(1, error="SyntaxError: invalid syntax"),
            mode=AutoFixMode.BALANCED,
        )
        assert "pattern" not in msg.split("Attempts:")[1].split("\n")[0]


# -- Test: Helper functions ------------------------------------------------


class TestHelpers:
    """Tests for individual formatting helpers."""

    def test_category_labels(self):
        assert _category_label(ErrorCategory.A) == "Parse error"
        assert _category_label(ErrorCategory.B) == "Runtime crash"
        assert _category_label(ErrorCategory.C) == "Network/server failure"
        assert _category_label(ErrorCategory.D) == "Selector timeout"
        assert _category_label(ErrorCategory.E) == "Page state error"
        assert _category_label(ErrorCategory.F1) == "Browser/page crash"
        assert _category_label(ErrorCategory.F2) == "Subprocess timeout"
        assert _category_label(ErrorCategory.F3) == "Infrastructure failure"
        assert _category_label(ErrorCategory.G) == "Schema validation failure"

    def test_stability_labels(self):
        assert _stability_label(StabilityLevel.STABLE) == "stable pattern"
        assert _stability_label(StabilityLevel.CONSISTENT) == "consistent pattern"
        assert _stability_label(StabilityLevel.MIXED) == "mixed pattern"
        assert _stability_label(StabilityLevel.CHAOTIC) == "chaotic pattern"

    def test_page_summary_all_real(self):
        results = [PageVerificationResult.REAL_PAGE] * 3
        assert _page_summary(results) == "Verified real (3/3 attempts)"

    def test_page_summary_antibot(self):
        results = [
            PageVerificationResult.ANTI_BOT,
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.REAL_PAGE,
        ]
        assert _page_summary(results) == "Anti-bot detected (1/3 attempts)"

    def test_page_summary_server_error(self):
        results = [
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.SERVER_ERROR,
            PageVerificationResult.REAL_PAGE,
        ]
        assert _page_summary(results) == "Server error detected (1/3 attempts)"

    def test_page_summary_partial(self):
        results = [
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.REDIRECTED,
            PageVerificationResult.NO_RESPONSE,
        ]
        assert _page_summary(results) == "Partially verified (1/3 REAL_PAGE)"

    def test_page_summary_none_real(self):
        results = [
            PageVerificationResult.NO_RESPONSE,
            PageVerificationResult.NO_RESPONSE,
            PageVerificationResult.NO_RESPONSE,
        ]
        assert _page_summary(results) == "Not verified (0/3 REAL_PAGE)"

    def test_page_summary_empty(self):
        assert _page_summary([]) == ""


# -- Test: Edge cases in message formatting --------------------------------


class TestMessageEdgeCases:
    """Edge cases in message formatting."""

    def test_empty_error_string(self):
        """No error summary when error is empty."""
        msg = format_diagnosis_message(
            action=AutoFixAction.RAISE,
            reason="test",
            category=ErrorCategory.G,
            stability=StabilityLevel.STABLE,
            page_results=[PageVerificationResult.REAL_PAGE] * 3,
            attempts=_make_attempts(error=""),
            mode=AutoFixMode.BALANCED,
        )
        assert "Cached script failed" in msg
        # Should not have "Error:" line when error is empty
        assert "Error:" not in msg

    def test_long_error_truncated(self):
        """Error summary longer than 120 chars is truncated."""
        long_error = "A" * 200
        msg = format_diagnosis_message(
            action=AutoFixAction.RAISE,
            reason="test",
            category=ErrorCategory.B,
            stability=StabilityLevel.STABLE,
            page_results=[PageVerificationResult.REAL_PAGE] * 3,
            attempts=_make_attempts(error=long_error),
            mode=AutoFixMode.BALANCED,
        )
        assert "..." in msg
        # The truncated summary should be <= 120 chars
        for line in msg.splitlines():
            if "Error:" in line:
                error_text = line.split("Error:")[1].strip()
                assert len(error_text) <= 120

    def test_no_page_results(self):
        """No page summary when page_results is empty."""
        msg = format_diagnosis_message(
            action=AutoFixAction.REGENERATE,
            reason="Parse error, regenerating",
            category=ErrorCategory.A,
            stability=None,
            page_results=[],
            attempts=_make_attempts(1, error="SyntaxError: invalid syntax"),
            mode=AutoFixMode.BALANCED,
        )
        assert "Page:" not in msg

    def test_single_attempt(self):
        """Message works with just 1 attempt."""
        msg = format_diagnosis_message(
            action=AutoFixAction.REGENERATE,
            reason="Parse error",
            category=ErrorCategory.A,
            stability=None,
            page_results=[],
            attempts=_make_attempts(1, error="SyntaxError: oops"),
            mode=AutoFixMode.CONSERVATIVE,
        )
        assert "1/1 failed" in msg

    def test_generic_raise_with_reason(self):
        """Generic RAISE (no anti-bot, no server error, not conservative D/E,
        not noisy) shows the reason from the decision engine."""
        msg = format_diagnosis_message(
            action=AutoFixAction.RAISE,
            reason="Page not fully verified (2/3)",
            category=ErrorCategory.B,
            stability=StabilityLevel.STABLE,
            page_results=[
                PageVerificationResult.REAL_PAGE,
                PageVerificationResult.REAL_PAGE,
                PageVerificationResult.REDIRECTED,
            ],
            attempts=_make_attempts(),
            mode=AutoFixMode.CONSERVATIVE,
        )
        # Not anti-bot, not server error, not conservative D/E, not noisy
        # -> generic raise with reason
        assert "Page not fully verified" in msg
        assert "regenerate=True" in msg
