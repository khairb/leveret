"""Tests for the diagnosis loop (Phase 4).

Tests cover:
  - 3-attempt loop with all failures -> stability -> decision
  - Attempt success on retry -> return AttemptResult
  - Immediate exits: Category A (REGENERATE), F2/F3 (RAISE)
  - Early exits mid-loop: A/F3 appearing on attempt 2-3
  - Delays between attempts (mocked)
  - CHAOTIC stability -> RAISE
  - Page signals collected and passed to page verifier
  - Category C flow (Tier 3 -> RAISE)
  - execute_fn crash -> graceful handling
  - E eligibility check (eligible vs ineligible)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from scout.autofix.diagnosis import (
    diagnose,
    format_diagnosis_message,
    _check_e_eligibility,
    _category_label,
    _stability_label,
    _page_summary,
)
from scout.autofix.types import (
    AttemptResult,
    AutoFixAction,
    RegenerateMode,
    DiagnosisResult,
    ErrorCategory,
    Fingerprint,
    PageSignals,
    PageVerificationResult,
    StabilityLevel,
)


# -- Helpers ---------------------------------------------------------------


_SENTINEL = object()


def _make_failed_attempt(
    error: str = "AttributeError: 'NoneType' object has no attribute 'text'",
    exit_code: int | None = 1,
    schema_error: str | None = None,
    page_signals: PageSignals | None | object = _SENTINEL,
) -> AttemptResult:
    """Create a failed AttemptResult.

    Pass ``page_signals=None`` explicitly to simulate no page signals
    (e.g., page crash). Omit to get default real-page signals.
    """
    if page_signals is _SENTINEL:
        # Default: real page signals
        page_signals = PageSignals(
            http_status=200,
            page_url="https://example.com/products",
            content="<html><body><h1>Products</h1></body></html>",
        )
    return AttemptResult(
        success=False,
        error=error,
        exit_code=exit_code,
        schema_error=schema_error,
        page_signals=page_signals,
    )


def _make_success_attempt(data: object = None) -> AttemptResult:
    """Create a successful AttemptResult."""
    if data is None:
        data = [{"name": "Product 1", "price": 9.99}]
    return AttemptResult(success=True, data=data)


def _make_execute_fn(results: list[AttemptResult]) -> AsyncMock:
    """Create a mock execute_fn that returns results in sequence."""
    mock = AsyncMock(side_effect=results)
    return mock


TARGET_URL = "https://example.com/products"


# -- Patching asyncio.sleep to avoid real delays --

@pytest.fixture(autouse=True)
def _no_sleep():
    """Patch asyncio.sleep to avoid real delays in tests."""
    with patch("scout.autofix.diagnosis.asyncio.sleep", new_callable=AsyncMock):
        yield


# -- Test: Attempt success on retry ----------------------------------------


class TestRetrySuccess:
    """Tests where the script succeeds on attempt 2 or 3."""

    @pytest.mark.asyncio
    async def test_attempt_2_succeeds(self):
        """Attempt 1 fails (Cat B), attempt 2 succeeds -> return data."""
        results = [
            _make_failed_attempt(),
            _make_success_attempt(data=[{"name": "OK"}]),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.BALANCED)

        assert isinstance(result, AttemptResult)
        assert result.success is True
        assert result.data == [{"name": "OK"}]
        assert execute_fn.call_count == 2

    @pytest.mark.asyncio
    async def test_attempt_3_succeeds(self):
        """Attempt 1-2 fail, attempt 3 succeeds -> return data."""
        results = [
            _make_failed_attempt(),
            _make_failed_attempt(),
            _make_success_attempt(data={"title": "Done"}),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.EAGER)

        assert isinstance(result, AttemptResult)
        assert result.success is True
        assert result.data == {"title": "Done"}
        assert execute_fn.call_count == 3

    @pytest.mark.asyncio
    async def test_attempt_1_succeeds(self):
        """Attempt 1 succeeds immediately -> return data, no retries."""
        results = [_make_success_attempt(data="hello")]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.BALANCED)

        assert isinstance(result, AttemptResult)
        assert result.success is True
        assert result.data == "hello"
        assert execute_fn.call_count == 1


# -- Test: Immediate exits (no retries) ------------------------------------


class TestImmediateExits:
    """Tests for categories that skip retries entirely."""

    @pytest.mark.asyncio
    async def test_category_a_regenerate_immediately(self):
        """Category A -> REGENERATE immediately, no more attempts."""
        results = [
            _make_failed_attempt(error="SyntaxError: invalid syntax"),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.BALANCED)

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.REGENERATE
        assert result.category == ErrorCategory.A
        assert result.stability is None  # No stability (no retries)
        assert len(result.attempts) == 1
        assert execute_fn.call_count == 1
        assert "Parse error" in result.message

    @pytest.mark.asyncio
    async def test_category_a_cautious(self):
        """Category A -> REGENERATE even in cautious mode."""
        results = [
            _make_failed_attempt(error="IndentationError: expected an indented block"),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.CAUTIOUS)

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.REGENERATE
        assert result.category == ErrorCategory.A

    @pytest.mark.asyncio
    async def test_category_f2_raise_immediately(self):
        """Category F2 -> RAISE immediately, no retries."""
        results = [
            _make_failed_attempt(
                error="Function timed out after 600 seconds",
            ),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.EAGER)

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.RAISE
        assert result.category == ErrorCategory.F2
        assert result.stability is None
        assert len(result.attempts) == 1
        assert execute_fn.call_count == 1

    @pytest.mark.asyncio
    async def test_category_f3_raise_immediately(self):
        """Category F3 -> RAISE immediately, no retries."""
        results = [
            _make_failed_attempt(
                error="executable doesn't exist at /path/to/chrome",
            ),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.BALANCED)

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.RAISE
        assert result.category == ErrorCategory.F3
        assert len(result.attempts) == 1
        assert execute_fn.call_count == 1


# -- Test: Early exits mid-loop -------------------------------------------


class TestEarlyExitMidLoop:
    """Tests for A/F3 appearing on attempt 2-3."""

    @pytest.mark.asyncio
    async def test_attempt_2_category_a(self):
        """Attempt 1 is D, attempt 2 is A -> REGENERATE immediately."""
        results = [
            _make_failed_attempt(
                error="Page.wait_for_selector: Timeout 5000ms exceeded.",
            ),
            _make_failed_attempt(error="SyntaxError: invalid syntax"),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.BALANCED)

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.REGENERATE
        assert result.category == ErrorCategory.A
        assert len(result.attempts) == 2
        assert execute_fn.call_count == 2

    @pytest.mark.asyncio
    async def test_attempt_2_category_f3(self):
        """Attempt 1 is B, attempt 2 is F3 -> RAISE immediately."""
        results = [
            _make_failed_attempt(),
            _make_failed_attempt(
                error="executable doesn't exist at /path/to/chrome",
            ),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.BALANCED)

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.RAISE
        assert result.category == ErrorCategory.F3
        assert len(result.attempts) == 2
        assert execute_fn.call_count == 2

    @pytest.mark.asyncio
    async def test_attempt_3_category_a(self):
        """Attempt 1-2 are D, attempt 3 is A -> REGENERATE."""
        results = [
            _make_failed_attempt(
                error="Page.wait_for_selector: Timeout 5000ms exceeded.",
            ),
            _make_failed_attempt(
                error="Page.click: Timeout 5000ms exceeded.",
            ),
            _make_failed_attempt(error="SyntaxError: unexpected EOF"),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.BALANCED)

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.REGENERATE
        assert result.category == ErrorCategory.A
        assert len(result.attempts) == 3
        assert execute_fn.call_count == 3

    @pytest.mark.asyncio
    async def test_attempt_3_category_f3(self):
        """Attempt 1-2 are B, attempt 3 is F3 -> RAISE."""
        results = [
            _make_failed_attempt(),
            _make_failed_attempt(),
            _make_failed_attempt(
                error="No space left on device",
            ),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.BALANCED)

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.RAISE
        assert result.category == ErrorCategory.F3
        assert len(result.attempts) == 3


# -- Test: Full 3-attempt diagnosis -> decision ----------------------------


class TestFullDiagnosis:
    """Tests for the full 3-attempt flow leading to stability + decision."""

    @pytest.mark.asyncio
    async def test_3x_category_b_stable_real_page_balanced_regenerate(self):
        """3x Cat B STABLE + 3/3 REAL_PAGE + balanced -> REGENERATE."""
        error = "AttributeError: 'NoneType' object has no attribute 'text'"
        results = [
            _make_failed_attempt(error=error),
            _make_failed_attempt(error=error),
            _make_failed_attempt(error=error),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.BALANCED)

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.REGENERATE
        assert result.category == ErrorCategory.B
        assert result.stability == StabilityLevel.STABLE
        assert len(result.page_results) == 3
        assert all(r == PageVerificationResult.REAL_PAGE for r in result.page_results)
        assert len(result.fingerprints) == 3
        assert len(result.attempts) == 3

    @pytest.mark.asyncio
    async def test_3x_category_b_stable_real_page_cautious_regenerate(self):
        """3x Cat B STABLE + 3/3 REAL_PAGE + cautious -> REGENERATE."""
        error = "KeyError: 'price'"
        results = [
            _make_failed_attempt(error=error),
            _make_failed_attempt(error=error),
            _make_failed_attempt(error=error),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.CAUTIOUS)

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.REGENERATE
        assert result.category == ErrorCategory.B
        assert result.stability == StabilityLevel.STABLE

    @pytest.mark.asyncio
    async def test_3x_category_d_stable_real_page_balanced_regenerate(self):
        """3x Cat D STABLE + 3/3 REAL_PAGE + balanced -> REGENERATE."""
        error = "Page.wait_for_selector: Timeout 5000ms exceeded."
        results = [
            _make_failed_attempt(error=error),
            _make_failed_attempt(error=error),
            _make_failed_attempt(error=error),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.BALANCED)

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.REGENERATE
        assert result.category == ErrorCategory.D
        assert result.stability == StabilityLevel.STABLE

    @pytest.mark.asyncio
    async def test_3x_category_d_stable_real_page_cautious_raise(self):
        """3x Cat D STABLE + 3/3 REAL_PAGE + cautious -> RAISE.

        S9: Cautious never regenerates ambiguous categories (D/E).
        """
        error = "Page.wait_for_selector: Timeout 5000ms exceeded."
        results = [
            _make_failed_attempt(error=error),
            _make_failed_attempt(error=error),
            _make_failed_attempt(error=error),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.CAUTIOUS)

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.RAISE
        assert result.category == ErrorCategory.D

    @pytest.mark.asyncio
    async def test_3x_category_c_raise_all_modes(self):
        """3x Cat C (Tier 3) -> RAISE in all modes."""
        error = "Page.goto: net::ERR_CONNECTION_REFUSED at https://example.com"
        for mode in RegenerateMode:
            results = [
                _make_failed_attempt(error=error),
                _make_failed_attempt(error=error),
                _make_failed_attempt(error=error),
            ]
            execute_fn = _make_execute_fn(results)

            result = await diagnose(execute_fn, TARGET_URL, mode)

            assert isinstance(result, DiagnosisResult)
            assert result.action == AutoFixAction.RAISE, f"Failed for {mode}"
            assert result.category == ErrorCategory.C

    @pytest.mark.asyncio
    async def test_3x_category_f1_raise_all_modes(self):
        """3x Cat F1 (Tier 3) -> RAISE in all modes."""
        error = "Page crashed"
        for mode in RegenerateMode:
            results = [
                _make_failed_attempt(error=error, page_signals=PageSignals()),
                _make_failed_attempt(error=error, page_signals=PageSignals()),
                _make_failed_attempt(error=error, page_signals=PageSignals()),
            ]
            execute_fn = _make_execute_fn(results)

            result = await diagnose(execute_fn, TARGET_URL, mode)

            assert isinstance(result, DiagnosisResult)
            assert result.action == AutoFixAction.RAISE, f"Failed for {mode}"

    @pytest.mark.asyncio
    async def test_3x_category_g_stable_real_page_balanced_regenerate(self):
        """3x Cat G STABLE + 3/3 REAL_PAGE + balanced -> REGENERATE."""
        results = [
            _make_failed_attempt(
                error="",
                exit_code=0,
                schema_error="Expected at least 5 items, got 0",
            ),
            _make_failed_attempt(
                error="",
                exit_code=0,
                schema_error="Expected at least 5 items, got 0",
            ),
            _make_failed_attempt(
                error="",
                exit_code=0,
                schema_error="Expected at least 5 items, got 0",
            ),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.BALANCED)

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.REGENERATE
        assert result.category == ErrorCategory.G
        assert result.stability == StabilityLevel.STABLE


# -- Test: Stability patterns ----------------------------------------------


class TestStabilityPatterns:
    """Tests for different stability outcomes in the full pipeline."""

    @pytest.mark.asyncio
    async def test_chaotic_3_categories_raise(self):
        """3 different categories -> CHAOTIC -> RAISE."""
        results = [
            _make_failed_attempt(
                error="AttributeError: no attribute 'text'",
            ),
            _make_failed_attempt(
                error="Page.wait_for_selector: Timeout 5000ms exceeded.",
            ),
            _make_failed_attempt(
                error="Page.goto: net::ERR_CONNECTION_REFUSED at https://example.com",
            ),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.EAGER)

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.RAISE
        assert result.stability == StabilityLevel.CHAOTIC

    @pytest.mark.asyncio
    async def test_mixed_2_categories_raise(self):
        """2x Cat D + 1x Cat C -> MIXED -> RAISE."""
        results = [
            _make_failed_attempt(
                error="Page.wait_for_selector: Timeout 5000ms exceeded.",
            ),
            _make_failed_attempt(
                error="Page.wait_for_selector: Timeout 5000ms exceeded.",
            ),
            _make_failed_attempt(
                error="Page.goto: net::ERR_CONNECTION_REFUSED at https://example.com",
            ),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.EAGER)

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.RAISE
        assert result.stability == StabilityLevel.MIXED

    @pytest.mark.asyncio
    async def test_consistent_same_category_different_errors(self):
        """3x Cat B with different error types -> CONSISTENT."""
        results = [
            _make_failed_attempt(error="AttributeError: no attribute 'x'"),
            _make_failed_attempt(error="KeyError: 'price'"),
            _make_failed_attempt(error="TypeError: cannot unpack"),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.BALANCED)

        assert isinstance(result, DiagnosisResult)
        assert result.category == ErrorCategory.B
        # Same category but different error types -> CONSISTENT
        assert result.stability == StabilityLevel.CONSISTENT


# -- Test: Page verification in diagnosis ----------------------------------


class TestPageVerification:
    """Tests for page signal collection and verification within diagnosis."""

    @pytest.mark.asyncio
    async def test_antibot_blocks_regeneration(self):
        """Anti-bot detected -> RAISE even with stable pattern."""
        error = "AttributeError: 'NoneType' object has no attribute 'text'"
        antibot_signals = PageSignals(
            http_status=200,
            page_url="https://example.com/products",
            content=(
                '<html><head><title>Just a moment...</title></head>'
                '<body><form class="challenge-form">'
                '<input type="hidden" name="__cf_chl_f_tk" value="abc">'
                '</form></body></html>'
            ),
            headers={"cf-mitigated": "challenge"},
        )
        results = [
            _make_failed_attempt(error=error, page_signals=antibot_signals),
            _make_failed_attempt(error=error, page_signals=antibot_signals),
            _make_failed_attempt(error=error, page_signals=antibot_signals),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.EAGER)

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.RAISE
        assert all(
            r == PageVerificationResult.ANTI_BOT for r in result.page_results
        )

    @pytest.mark.asyncio
    async def test_server_error_blocks_cautious(self):
        """1/3 SERVER_ERROR blocks regeneration in cautious mode."""
        error = "AttributeError: 'NoneType' object has no attribute 'text'"
        real = PageSignals(
            http_status=200,
            page_url="https://example.com/products",
            content="<html><body>Real page</body></html>",
        )
        server_err = PageSignals(
            http_status=503,
            page_url="https://example.com/products",
            content="Service Unavailable",
        )
        results = [
            _make_failed_attempt(error=error, page_signals=real),
            _make_failed_attempt(error=error, page_signals=server_err),
            _make_failed_attempt(error=error, page_signals=real),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.CAUTIOUS)

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.RAISE

    @pytest.mark.asyncio
    async def test_no_page_signals_produces_no_response(self):
        """No page signals (page crash) -> NO_RESPONSE verification."""
        error = "Page crashed"
        results = [
            _make_failed_attempt(error=error, page_signals=None),
            _make_failed_attempt(error=error, page_signals=None),
            _make_failed_attempt(error=error, page_signals=None),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.BALANCED)

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.RAISE
        assert all(
            r == PageVerificationResult.NO_RESPONSE for r in result.page_results
        )


# -- Test: E eligibility --------------------------------------------------


class TestEEligibility:
    """Tests for Category E eligible vs ineligible sub-types."""

    @pytest.mark.asyncio
    async def test_e_eligible_stable_balanced_regenerate(self):
        """Category E (pointer intercept, eligible) + STABLE + 3/3 REAL
        + balanced -> REGENERATE.
        """
        error = (
            "patchright._impl._errors.TimeoutError: "
            "Page.click: Timeout 5000ms exceeded.\n"
            "Call log:\n"
            "  - waiting for locator('#target')\n"
            "  - <div></div> intercepts pointer events\n"
        )
        results = [
            _make_failed_attempt(error=error),
            _make_failed_attempt(error=error),
            _make_failed_attempt(error=error),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.BALANCED)

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.REGENERATE
        assert result.category == ErrorCategory.E
        assert result.e_eligible is True

    @pytest.mark.asyncio
    async def test_e_ineligible_context_destroyed_raise(self):
        """Category E (context destroyed, ineligible) -> RAISE in all modes."""
        error = "Execution context was destroyed, most likely because of a navigation."
        for mode in RegenerateMode:
            results = [
                _make_failed_attempt(error=error),
                _make_failed_attempt(error=error),
                _make_failed_attempt(error=error),
            ]
            execute_fn = _make_execute_fn(results)

            result = await diagnose(execute_fn, TARGET_URL, mode)

            assert isinstance(result, DiagnosisResult)
            assert result.action == AutoFixAction.RAISE, f"Failed for {mode}"
            assert result.e_eligible is False

    @pytest.mark.asyncio
    async def test_e_ineligible_frame_detached_raise(self):
        """Category E (frame detached, ineligible) -> RAISE in all modes."""
        error = "Frame was detached"
        results = [
            _make_failed_attempt(error=error),
            _make_failed_attempt(error=error),
            _make_failed_attempt(error=error),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.EAGER)

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.RAISE
        assert result.e_eligible is False

    def test_check_e_eligibility_eligible(self):
        """Non-E categories and eligible E patterns -> True."""
        attempts = [
            AttemptResult(
                success=False,
                error="<div> intercepts pointer events",
                category=ErrorCategory.E,
            ),
        ]
        assert _check_e_eligibility(attempts) is True

    def test_check_e_eligibility_ineligible(self):
        """Context destroyed -> False."""
        attempts = [
            AttemptResult(
                success=False,
                error="Execution context was destroyed",
                category=ErrorCategory.E,
            ),
        ]
        assert _check_e_eligibility(attempts) is False

    def test_check_e_eligibility_non_e_category(self):
        """Non-E category with E-ineligible text in error -> True.

        Only Category E attempts are checked for E-ineligibility.
        """
        attempts = [
            AttemptResult(
                success=False,
                error="Execution context was destroyed",
                category=ErrorCategory.B,
            ),
        ]
        assert _check_e_eligibility(attempts) is True


# -- Test: execute_fn crash protection -------------------------------------


class TestExecuteFnCrashProtection:
    """Tests for graceful handling of execute_fn exceptions."""

    @pytest.mark.asyncio
    async def test_execute_fn_raises_exception(self):
        """execute_fn raises -> wrapped as failed AttemptResult."""
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Unexpected internal error")
            return _make_failed_attempt()

        result = await diagnose(fn, TARGET_URL, RegenerateMode.BALANCED)

        # Should not crash — the exception is caught and wrapped
        assert isinstance(result, DiagnosisResult)
        assert result.action in (AutoFixAction.REGENERATE, AutoFixAction.RAISE)
        assert len(result.attempts) == 3

    @pytest.mark.asyncio
    async def test_all_executions_crash(self):
        """All 3 execute_fn calls raise -> 3 failed attempts, no crash."""
        async def always_crash():
            raise ValueError("Boom!")

        result = await diagnose(always_crash, TARGET_URL, RegenerateMode.BALANCED)

        assert isinstance(result, DiagnosisResult)
        assert len(result.attempts) == 3
        # Crashes become Category B (catch-all)
        for attempt in result.attempts:
            assert attempt.success is False
            assert "ValueError: Boom!" in attempt.error


# -- Test: Delays between attempts -----------------------------------------


class TestDelays:
    """Tests verifying delays between diagnostic attempts."""

    @pytest.mark.asyncio
    async def test_delays_called_between_retries(self):
        """asyncio.sleep called with 3-5s delay between retry attempts."""
        results = [
            _make_failed_attempt(),
            _make_failed_attempt(),
            _make_failed_attempt(),
        ]
        execute_fn = _make_execute_fn(results)

        with patch(
            "scout.autofix.diagnosis.asyncio.sleep",
            new_callable=AsyncMock,
        ) as mock_sleep, patch(
            "scout.autofix.diagnosis.random.uniform",
            return_value=4.0,
        ):
            await diagnose(execute_fn, TARGET_URL, RegenerateMode.BALANCED)

            # 2 delays: before attempt 2 and before attempt 3
            assert mock_sleep.call_count == 2
            for call in mock_sleep.call_args_list:
                assert call.args[0] == 4.0

    @pytest.mark.asyncio
    async def test_no_delay_for_immediate_exit(self):
        """No delay when Category A exits immediately."""
        results = [
            _make_failed_attempt(error="SyntaxError: invalid syntax"),
        ]
        execute_fn = _make_execute_fn(results)

        with patch(
            "scout.autofix.diagnosis.asyncio.sleep",
            new_callable=AsyncMock,
        ) as mock_sleep:
            await diagnose(execute_fn, TARGET_URL, RegenerateMode.BALANCED)

            mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_delay_range_is_3_to_5(self):
        """random.uniform called with (3.0, 5.0)."""
        results = [
            _make_failed_attempt(),
            _make_failed_attempt(),
            _make_failed_attempt(),
        ]
        execute_fn = _make_execute_fn(results)

        with patch(
            "scout.autofix.diagnosis.asyncio.sleep",
            new_callable=AsyncMock,
        ), patch(
            "scout.autofix.diagnosis.random.uniform",
            return_value=3.5,
        ) as mock_uniform:
            await diagnose(execute_fn, TARGET_URL, RegenerateMode.BALANCED)

            for call in mock_uniform.call_args_list:
                assert call.args == (3.0, 5.0)


# -- Test: DiagnosisResult fields -----------------------------------------


class TestDiagnosisResultFields:
    """Tests verifying DiagnosisResult is populated correctly."""

    @pytest.mark.asyncio
    async def test_fingerprints_collected(self):
        """All 3 fingerprints collected in DiagnosisResult."""
        error = "KeyError: 'missing_field'"
        results = [
            _make_failed_attempt(error=error),
            _make_failed_attempt(error=error),
            _make_failed_attempt(error=error),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.BALANCED)

        assert isinstance(result, DiagnosisResult)
        assert len(result.fingerprints) == 3
        for fp in result.fingerprints:
            assert fp.category == ErrorCategory.B

    @pytest.mark.asyncio
    async def test_page_results_collected(self):
        """All 3 page verification results collected."""
        error = "AttributeError: no attr"
        results = [
            _make_failed_attempt(error=error),
            _make_failed_attempt(error=error),
            _make_failed_attempt(error=error),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.BALANCED)

        assert isinstance(result, DiagnosisResult)
        assert len(result.page_results) == 3

    @pytest.mark.asyncio
    async def test_attempts_stored(self):
        """All attempt results stored in DiagnosisResult."""
        error = "IndexError: list index out of range"
        results = [
            _make_failed_attempt(error=error),
            _make_failed_attempt(error=error),
            _make_failed_attempt(error=error),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.BALANCED)

        assert isinstance(result, DiagnosisResult)
        assert len(result.attempts) == 3
        for attempt in result.attempts:
            assert attempt.category == ErrorCategory.B
            assert attempt.fingerprint is not None
            assert attempt.page_result is not None

    @pytest.mark.asyncio
    async def test_message_populated(self):
        """DiagnosisResult.message is non-empty."""
        error = "AttributeError: no attr"
        results = [
            _make_failed_attempt(error=error),
            _make_failed_attempt(error=error),
            _make_failed_attempt(error=error),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.BALANCED)

        assert isinstance(result, DiagnosisResult)
        assert result.message
        assert "Cached script failed" in result.message

    @pytest.mark.asyncio
    async def test_primary_category_from_first_attempt(self):
        """DiagnosisResult.category comes from the first attempt."""
        results = [
            _make_failed_attempt(
                error="AttributeError: no attr",
            ),
            _make_failed_attempt(
                error="KeyError: 'key'",
            ),
            _make_failed_attempt(
                error="TypeError: bad type",
            ),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.BALANCED)

        assert isinstance(result, DiagnosisResult)
        # All are Cat B (same primary category)
        assert result.category == ErrorCategory.B


# -- Test: Missing edge cases from deep review -----------------------------


class TestEdgeCases:
    """Edge cases found during senior-level review."""

    @pytest.mark.asyncio
    async def test_f2_mid_loop_not_early_exit(self):
        """F2 on attempt 2 does NOT trigger early exit.

        Spec S7: Early exit only for A and F3 mid-loop. F2 is not
        mentioned — it should continue to attempt 3.
        """
        results = [
            _make_failed_attempt(
                error="AttributeError: 'NoneType' has no attribute 'text'",
            ),
            _make_failed_attempt(
                error="Function timed out after 600 seconds",
            ),
            _make_failed_attempt(
                error="AttributeError: 'NoneType' has no attribute 'text'",
            ),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.BALANCED)

        assert isinstance(result, DiagnosisResult)
        # All 3 attempts should run (F2 doesn't trigger early exit)
        assert len(result.attempts) == 3
        assert execute_fn.call_count == 3
        # 2 categories (B + F2) -> MIXED -> RAISE
        assert result.stability == StabilityLevel.MIXED
        assert result.action == AutoFixAction.RAISE

    @pytest.mark.asyncio
    async def test_e_ineligible_navigating_frame_detached(self):
        """'Navigating frame was detached' is also E-ineligible.

        E_INELIGIBLE_PATTERNS has 3 entries. This tests the third one.
        """
        error = "Navigating frame was detached"
        results = [
            _make_failed_attempt(error=error),
            _make_failed_attempt(error=error),
            _make_failed_attempt(error=error),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.EAGER)

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.RAISE
        assert result.e_eligible is False

    @pytest.mark.asyncio
    async def test_redirected_page_blocks_cautious(self):
        """Page URL domain mismatch -> REDIRECTED -> blocks regeneration."""
        error = "AttributeError: 'NoneType' has no attribute 'text'"
        redirected_signals = PageSignals(
            http_status=200,
            page_url="https://login.otherdomain.com/sso",
            content="<html><body>Login page</body></html>",
        )
        results = [
            _make_failed_attempt(error=error, page_signals=redirected_signals),
            _make_failed_attempt(error=error, page_signals=redirected_signals),
            _make_failed_attempt(error=error, page_signals=redirected_signals),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.BALANCED)

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.RAISE
        assert all(
            r == PageVerificationResult.REDIRECTED for r in result.page_results
        )

    @pytest.mark.asyncio
    async def test_e_eligible_defaults_true_for_non_e(self):
        """e_eligible is True when category is not E."""
        error = "AttributeError: 'NoneType' has no attribute 'text'"
        results = [
            _make_failed_attempt(error=error),
            _make_failed_attempt(error=error),
            _make_failed_attempt(error=error),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.BALANCED)

        assert isinstance(result, DiagnosisResult)
        assert result.category == ErrorCategory.B
        assert result.e_eligible is True

    @pytest.mark.asyncio
    async def test_crashed_execute_fn_produces_no_response_page(self):
        """When execute_fn crashes, the wrapped attempt has no page_signals.

        The full chain: crash -> AttemptResult(page_signals=None)
        -> verify_page(None, ...) -> NO_RESPONSE.
        """
        async def crash_once_then_fail():
            crash_once_then_fail.calls += 1
            if crash_once_then_fail.calls == 1:
                raise ConnectionError("Lost connection to browser")
            return _make_failed_attempt()

        crash_once_then_fail.calls = 0

        result = await diagnose(
            crash_once_then_fail, TARGET_URL, RegenerateMode.BALANCED,
        )

        assert isinstance(result, DiagnosisResult)
        # First attempt's page_result should be NO_RESPONSE (crashed, no signals)
        first = result.attempts[0]
        assert first.page_signals is None
        assert first.page_result == PageVerificationResult.NO_RESPONSE
        # Remaining attempts have real page signals
        for attempt in result.attempts[1:]:
            assert attempt.page_result == PageVerificationResult.REAL_PAGE

    @pytest.mark.asyncio
    async def test_stability_none_with_only_1_fingerprint_immediate_exit(self):
        """Immediate exits produce stability=None (only 1 attempt)."""
        results = [
            _make_failed_attempt(error="SyntaxError: invalid syntax"),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.BALANCED)

        assert isinstance(result, DiagnosisResult)
        assert result.stability is None
        assert len(result.fingerprints) == 1

    @pytest.mark.asyncio
    async def test_early_exit_a_on_attempt_2_has_2_fingerprints(self):
        """Early exit with A on attempt 2: 2 attempts, 2 fingerprints."""
        results = [
            _make_failed_attempt(error="KeyError: 'missing'"),
            _make_failed_attempt(error="SyntaxError: unexpected EOF"),
        ]
        execute_fn = _make_execute_fn(results)

        result = await diagnose(execute_fn, TARGET_URL, RegenerateMode.BALANCED)

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.REGENERATE
        assert len(result.attempts) == 2
        assert len(result.fingerprints) == 2
        # Stability is NOT computed for early-exit paths
        assert result.stability is None
