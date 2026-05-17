"""Diagnosis loop — orchestrates 3-attempt diagnosis with signal collection.

Runs the cached script up to 3 times, collecting error fingerprints
and page-level signals on each failure. Assesses stability and page
verification, then routes to the mode-specific decision engine.

The diagnosis loop:
  1. Run attempt 1 — on success, return data immediately.
  2. Classify the error. If Category A -> REGENERATE. If F2/F3 -> RAISE.
  3. For retryable categories: run attempts 2-3 with 3-5s delays.
  4. On any success -> return data. On A/F3 mid-loop -> exit early.
  5. After 3 failures: assess stability, verify pages, route to decision.

Spec reference: docs/specific/AUTO_FIX_ALGORITHM.md S7
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable

from scout.autofix.classifier import classify_error
from scout.autofix.decision import decide
from scout.autofix.fingerprint import extract_fingerprint
from scout.autofix.page_verifier import verify_page
from scout.autofix.stability import assess_stability
from scout.autofix.types import (
    AttemptResult,
    AutoFixAction,
    RegenerateMode,
    DiagnosisResult,
    ErrorCategory,
    E_INELIGIBLE_PATTERNS,
    NO_RETRY_CATEGORIES,
    PageVerificationResult,
    StabilityLevel,
)

logger = logging.getLogger(__name__)

# S7: Delay between diagnostic attempts (3-5 seconds).
_MIN_DELAY_S = 3.0
_MAX_DELAY_S = 5.0

# S7: Maximum diagnostic attempts.
_MAX_ATTEMPTS = 3


# -- Public API -----------------------------------------------------------


async def diagnose(
    execute_fn: Callable[[], Awaitable[AttemptResult]],
    target_url: str,
    mode: RegenerateMode,
) -> AttemptResult | DiagnosisResult:
    """Orchestrate the 3-attempt diagnosis loop (spec S7).

    Runs the cached script up to 3 times, collecting error fingerprints
    and page-level signals on each failure. Assesses stability and page
    verification, then routes to the mode-specific decision engine.

    Args:
        execute_fn: Async callable that runs the script once and returns
            an ``AttemptResult``. The caller (Scraper) provides this adapter
            for either in-process or subprocess execution.
        target_url: The URL being scraped (for page verification domain check).
        mode: The user's auto-fix risk tolerance.

    Returns:
        ``AttemptResult`` if any attempt succeeded (caller should use the data).
        ``DiagnosisResult`` if all attempts failed (caller acts on the decision).
    """
    attempts: list[AttemptResult] = []

    # -- Attempt 1 --
    attempt = await _safe_execute(execute_fn)
    if attempt.success:
        return attempt

    _classify_and_verify(attempt, target_url)
    attempts.append(attempt)
    primary_category = attempt.category

    logger.debug(
        "Auto-fix: attempt 1 failed — %s", primary_category.value,
    )

    # -- Immediate exits (no retries needed) --

    # S7: Category A -> REGENERATE immediately (deterministic, no info gain)
    if primary_category == ErrorCategory.A:
        return _build_immediate_result(
            AutoFixAction.REGENERATE,
            primary_category,
            attempts,
            "Parse error (Category A) — code is structurally broken, "
            "regenerating immediately",
        )

    # S7: Category F2/F3 -> RAISE immediately (cost-prohibitive or impossible)
    if primary_category in NO_RETRY_CATEGORIES:
        return _build_immediate_result(
            AutoFixAction.RAISE,
            primary_category,
            attempts,
            _no_retry_reason(primary_category),
        )

    # -- Retry loop (attempts 2-3) --
    for attempt_num in range(2, _MAX_ATTEMPTS + 1):
        delay = random.uniform(_MIN_DELAY_S, _MAX_DELAY_S)
        logger.debug(
            "Auto-fix: waiting %.1fs before attempt %d", delay, attempt_num,
        )
        await asyncio.sleep(delay)

        attempt = await _safe_execute(execute_fn)
        if attempt.success:
            logger.debug(
                "Auto-fix: attempt %d succeeded", attempt_num,
            )
            return attempt

        _classify_and_verify(attempt, target_url)
        attempts.append(attempt)

        logger.debug(
            "Auto-fix: attempt %d failed — %s",
            attempt_num, attempt.category.value,
        )

        # S7: Early exit — Category A or F3 appearing mid-loop means
        # the environment changed (script corrupted / infra broke).
        if attempt.category == ErrorCategory.A:
            return _build_immediate_result(
                AutoFixAction.REGENERATE,
                ErrorCategory.A,
                attempts,
                "Parse error (Category A) detected on retry — "
                "script file may be corrupted, regenerating",
            )
        if attempt.category == ErrorCategory.F3:
            return _build_immediate_result(
                AutoFixAction.RAISE,
                ErrorCategory.F3,
                attempts,
                "Infrastructure failure (Category F3) detected on retry — "
                "no script can execute until the infrastructure is repaired",
            )

    # -- All attempts failed: assess and decide --
    fingerprints = [
        a.fingerprint for a in attempts if a.fingerprint is not None
    ]
    page_results = [
        a.page_result for a in attempts if a.page_result is not None
    ]

    # S8: Assess stability only from attempts that saw the real page.
    # Attempts with SERVER_ERROR, NO_RESPONSE, SOFT_BLOCK, etc. didn't
    # see real content — their errors reflect the environment, not the
    # script. Including them would mix two unrelated signals.
    real_fingerprints = [
        a.fingerprint for a in attempts
        if a.fingerprint is not None
        and a.page_result == PageVerificationResult.REAL_PAGE
    ]

    stability = (
        assess_stability(real_fingerprints)
        if len(real_fingerprints) >= 2
        else None
    )

    e_eligible = _check_e_eligibility(attempts)

    # S9: Route to the decision engine
    action, reason = decide(
        category=primary_category,
        stability=stability,
        page_results=page_results,
        mode=mode,
        e_eligible=e_eligible,
    )

    message = format_diagnosis_message(
        action=action,
        reason=reason,
        category=primary_category,
        stability=stability,
        page_results=page_results,
        attempts=attempts,
        mode=mode,
    )

    if action == AutoFixAction.REGENERATE:
        logger.info(
            "Auto-fix: cached script failed (%s — %d/%d attempts). %s",
            primary_category.value, len(attempts), len(attempts), reason,
        )
    else:
        logger.info("Auto-fix: not regenerating — %s", reason)

    return DiagnosisResult(
        action=action,
        category=primary_category,
        stability=stability,
        page_results=page_results,
        fingerprints=fingerprints,
        attempts=attempts,
        message=message,
        e_eligible=e_eligible,
    )


# -- Message formatting ---------------------------------------------------


def format_diagnosis_message(
    *,
    action: AutoFixAction,
    reason: str,
    category: ErrorCategory,
    stability: StabilityLevel | None,
    page_results: list[PageVerificationResult],
    attempts: list[AttemptResult],
    mode: RegenerateMode,
) -> str:
    """Format a user-facing diagnostic message (spec S12).

    Produces a multi-line message describing what happened, what
    auto-fix decided, and what the user can do.

    Args:
        action: The decided action (REGENERATE or RAISE).
        reason: Human-readable reason from the decision engine.
        category: The error category from classification.
        stability: Stability assessment (None for immediate decisions).
        page_results: Page verification results for each attempt.
        attempts: Full attempt results for detail extraction.
        mode: The user's auto-fix mode.

    Returns:
        Formatted diagnostic message string.
    """
    total = len(attempts)
    failed = sum(1 for a in attempts if not a.success)

    first_error = _extract_error_summary(attempts)
    cat_label = _category_label(category)
    stab_label = _stability_label(stability) if stability else None
    page_label = _page_summary(page_results)

    lines: list[str] = ["Cached script failed.", ""]

    # -- Diagnostic details --
    lines.append(f"  Category: {cat_label}")
    if first_error:
        lines.append(f"  Error:    {first_error}")

    if stab_label:
        lines.append(f"  Attempts: {failed}/{total} failed ({stab_label})")
    else:
        lines.append(f"  Attempts: {failed}/{total} failed")

    if page_label:
        lines.append(f"  Page:     {page_label}")

    lines.append("")

    # -- Decision explanation --
    if action == AutoFixAction.REGENERATE:
        lines.append(
            f"  Auto-fix is regenerating the script ({mode.value} mode)."
        )
        lines.append(f"  Reason: {reason}")
    else:
        # S12: Explain why auto-fix declined and what the user can do
        _append_raise_explanation(
            lines, reason, category, stability, page_results, mode,
        )

    return "\n".join(lines)


# -- Internal helpers ------------------------------------------------------


async def _safe_execute(
    execute_fn: Callable[[], Awaitable[AttemptResult]],
) -> AttemptResult:
    """Run execute_fn with crash protection.

    The diagnostic system must never crash (dev guide). If execute_fn
    raises an unexpected exception, wrap it as a failed AttemptResult.
    """
    try:
        return await execute_fn()
    except Exception as exc:
        logger.warning(
            "Auto-fix: execute_fn raised unexpected %s: %s",
            type(exc).__name__, exc,
        )
        return AttemptResult(
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def _classify_and_verify(
    attempt: AttemptResult,
    target_url: str,
) -> None:
    """Classify error, extract fingerprint, and verify page for an attempt.

    Mutates the attempt in-place (AttemptResult is mutable).
    """
    category = classify_error(
        stderr=attempt.error or "",
        exit_code=attempt.exit_code,
        schema_error=attempt.schema_error,
    )
    attempt.category = category

    attempt.fingerprint = extract_fingerprint(
        stderr=attempt.error or "",
        category=category,
        schema_error=attempt.schema_error,
    )

    attempt.page_result = verify_page(
        signals=attempt.page_signals,
        target_url=target_url,
    )


def _check_e_eligibility(attempts: list[AttemptResult]) -> bool:
    """Check if Category E errors are eligible for regeneration.

    S4/S9: ``Execution context was destroyed`` and ``Frame was detached``
    are never eligible for regeneration in any mode. Returns False if
    any Category E attempt contains these patterns.
    """
    for attempt in attempts:
        if attempt.category == ErrorCategory.E and attempt.error:
            for pattern in E_INELIGIBLE_PATTERNS:
                if pattern in attempt.error:
                    return False
    return True


def _build_immediate_result(
    action: AutoFixAction,
    category: ErrorCategory,
    attempts: list[AttemptResult],
    message: str,
) -> DiagnosisResult:
    """Build a DiagnosisResult for immediate decisions (no stability/page)."""
    fingerprints = [
        a.fingerprint for a in attempts if a.fingerprint is not None
    ]
    page_results = [
        a.page_result for a in attempts if a.page_result is not None
    ]

    return DiagnosisResult(
        action=action,
        category=category,
        stability=None,
        page_results=page_results,
        fingerprints=fingerprints,
        attempts=attempts,
        message=message,
    )


def _no_retry_reason(category: ErrorCategory) -> str:
    """Reason string for no-retry categories (F2, F3)."""
    reasons = {
        ErrorCategory.F2: (
            "Subprocess timeout (Category F2) — diagnostic cost "
            "prohibitive, not retrying"
        ),
        ErrorCategory.F3: (
            "Infrastructure failure (Category F3) — no script can "
            "execute until the infrastructure is repaired"
        ),
    }
    return reasons.get(
        category, f"Category {category.value} does not support retries",
    )


# -- Message formatting helpers --------------------------------------------


def _append_raise_explanation(
    lines: list[str],
    reason: str,
    category: ErrorCategory,
    stability: StabilityLevel | None,
    page_results: list[PageVerificationResult],
    mode: RegenerateMode,
) -> None:
    """Append RAISE-specific explanation and user actions to message lines."""
    has_antibot = any(
        r == PageVerificationResult.ANTI_BOT for r in page_results
    )
    has_soft_block = any(
        r == PageVerificationResult.SOFT_BLOCK for r in page_results
    )
    has_server_err = any(
        r == PageVerificationResult.SERVER_ERROR for r in page_results
    )
    is_noisy = stability in (StabilityLevel.MIXED, StabilityLevel.CHAOTIC)
    is_conservative_d_e = (
        mode == RegenerateMode.CAUTIOUS
        and category in (ErrorCategory.D, ErrorCategory.E)
    )

    if has_antibot:
        # S12: Anti-bot blocked
        lines.append(
            "  Auto-fix will not regenerate — the page served is not "
            "real content.",
        )
        lines.append(
            "  A new script would face the same anti-bot system.",
        )
        lines.append("")
        lines.append(
            "  Try again later, or use a different IP/proxy.",
        )
    elif has_soft_block:
        # S12: Non-content page blocked
        lines.append(
            "  Auto-fix will not regenerate — the page appears to be "
            "a non-content page (login, maintenance, or rate-limit), "
            "not real site content.",
        )
        lines.append(
            "  A new script would encounter the same page.",
        )
        lines.append("")
        lines.append(
            "  Check the URL manually, or try from a different "
            "network/session.",
        )
    elif has_server_err:
        # S12: Server error blocked
        lines.append(
            "  Auto-fix will not regenerate — the server returned "
            "errors during diagnosis.",
        )
        lines.append(
            "  The failure may be caused by server issues, "
            "not a broken script.",
        )
        lines.append("")
        lines.append(
            "  Try again later, or force regeneration: "
            "scraper.run(regenerate=True)",
        )
    elif is_noisy:
        # S12: Stability too noisy
        lines.append(
            "  The failure pattern is inconsistent — likely an "
            "environmental issue.",
        )
        lines.append("  Auto-fix will not regenerate.")
        lines.append("")
        lines.append(
            "  Try again later, or force regeneration: "
            "scraper.run(regenerate=True)",
        )
    elif is_conservative_d_e:
        # S12: Conservative mode declined D/E
        lines.append(
            f"  Mode: {mode.value} — auto-fix does not regenerate "
            f"{_category_label(category).lower()} errors.",
        )
        lines.append(
            "  To regenerate: scraper.run(regenerate=True)",
        )
        lines.append(
            '  To enable: auto_regenerate="balanced" or auto_regenerate="eager"',
        )
    else:
        # Generic RAISE explanation
        lines.append(f"  Auto-fix will not regenerate — {reason}")
        lines.append("")
        lines.append(
            "  Try again later, or force regeneration: "
            "scraper.run(regenerate=True)",
        )


def _extract_error_summary(attempts: list[AttemptResult]) -> str:
    """Extract a one-line error summary from the first failed attempt."""
    for attempt in attempts:
        if not attempt.success and attempt.error:
            # Get the last meaningful line (usually the exception line)
            lines = [
                line.strip()
                for line in attempt.error.strip().splitlines()
                if line.strip()
            ]
            if lines:
                last = lines[-1]
                if len(last) > 120:
                    return last[:117] + "..."
                return last
    return ""


def _category_label(category: ErrorCategory) -> str:
    """Human-readable label for an error category."""
    labels = {
        ErrorCategory.A: "Parse error",
        ErrorCategory.B: "Runtime crash",
        ErrorCategory.C: "Network/server failure",
        ErrorCategory.D: "Selector timeout",
        ErrorCategory.E: "Page state error",
        ErrorCategory.F1: "Browser/page crash",
        ErrorCategory.F2: "Subprocess timeout",
        ErrorCategory.F3: "Infrastructure failure",
        ErrorCategory.G: "Schema validation failure",
    }
    return labels.get(category, f"Unknown ({category.value})")


def _stability_label(stability: StabilityLevel) -> str:
    """Human-readable label for a stability level."""
    labels = {
        StabilityLevel.STABLE: "stable pattern",
        StabilityLevel.CONSISTENT: "consistent pattern",
        StabilityLevel.MIXED: "mixed pattern",
        StabilityLevel.CHAOTIC: "chaotic pattern",
    }
    return labels.get(stability, stability.value)


def _page_summary(
    page_results: list[PageVerificationResult],
) -> str:
    """Summarize page verification results for display."""
    if not page_results:
        return ""

    total = len(page_results)
    real = sum(
        1 for r in page_results if r == PageVerificationResult.REAL_PAGE
    )
    antibot = sum(
        1 for r in page_results if r == PageVerificationResult.ANTI_BOT
    )
    server_err = sum(
        1 for r in page_results if r == PageVerificationResult.SERVER_ERROR
    )

    soft_block = sum(
        1 for r in page_results if r == PageVerificationResult.SOFT_BLOCK
    )

    if antibot > 0:
        return f"Anti-bot detected ({antibot}/{total} attempts)"
    if soft_block > 0:
        return f"Non-content page detected ({soft_block}/{total} attempts)"
    if server_err > 0:
        return f"Server error detected ({server_err}/{total} attempts)"
    if real == total:
        return f"Verified real ({real}/{total} attempts)"
    if real > 0:
        return f"Partially verified ({real}/{total} REAL_PAGE)"
    return f"Not verified (0/{total} REAL_PAGE)"
