"""Shared types for the autofix system.

Enums, dataclasses, and type aliases used across all autofix modules:
classifier, fingerprint, antibot, page_verifier, stability, decision,
and diagnosis.

Spec reference: docs/specific/AUTO_FIX_ALGORITHM.md
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


# ── Error classification ──────────────────────────────────────


class ErrorCategory(enum.Enum):
    """Error category assigned by the classifier (spec S4).

    Categories are checked in priority order during classification:
    A -> F -> C -> E -> D -> B (output-stage) -> B (catch-all) -> G (external).

    Examples::

        >>> ErrorCategory.A   # SyntaxError, IndentationError, ImportError
        >>> ErrorCategory.F1  # Page crashed, browser closed
        >>> ErrorCategory.D   # Post-navigation timeout
    """

    A = "A"    # Code is structurally broken (parse errors)
    B = "B"    # Code crashed against page content (runtime errors)
    C = "C"    # Network/server failure (page never loaded)
    D = "D"    # Script's expectation wasn't met (post-nav timeout)
    E = "E"    # Page state prevented interaction
    F1 = "F1"  # Browser/page crash or process death
    F2 = "F2"  # Subprocess/execution timeout
    F3 = "F3"  # Infrastructure failure (browser not installed, disk full)
    G = "G"    # Output is wrong — schema validation failed, no error


# S9: Categories that never trigger regeneration (Tier 3).
TIER_3_CATEGORIES: frozenset[ErrorCategory] = frozenset({
    ErrorCategory.C,
    ErrorCategory.F1,
    ErrorCategory.F2,
    ErrorCategory.F3,
})

# S9: Categories that skip retries entirely.
NO_RETRY_CATEGORIES: frozenset[ErrorCategory] = frozenset({
    ErrorCategory.A,   # Deterministic — same code, same error
    ErrorCategory.F2,  # Diagnostic cost too high (full timeout x 3)
    ErrorCategory.F3,  # Infrastructure broken — no script can run
})

# S4/S9: Category E sub-types that are never eligible for regeneration.
E_INELIGIBLE_PATTERNS: frozenset[str] = frozenset({
    "Execution context was destroyed",
    "Frame was detached",
    "Navigating frame was detached",
})


# ── Error fingerprinting ─────────────────────────────────────


class ComparisonLevel(enum.Enum):
    """How closely two fingerprints match (spec S5).

    Levels are ordered from most to least specific:
    EXACT > SAME_KIND > SAME_CATEGORY > NONE.
    """

    EXACT = "exact"              # Same category + error_type + method + target
    SAME_KIND = "same_kind"      # Same category + error_type + method, different target
    SAME_CATEGORY = "same_category"  # Same category, different error_type or method
    NONE = "none"                # Different categories


@dataclass(frozen=True, slots=True)
class Fingerprint:
    """Structured representation of a single error for cross-attempt comparison.

    Extracted from raw stderr/stdout by ``extract_fingerprint()``.
    Compared via ``compare_fingerprints()`` to determine stability.

    Attributes:
        category: The error category (A-G).
        error_type: Exception class or error family.
            Examples: ``"SyntaxError"``, ``"TimeoutError"``,
            ``"net::ERR_CONNECTION_REFUSED"``.
        method: Browser automation method that failed, if applicable.
            Examples: ``"Page.wait_for_selector"``, ``"Page.goto"``.
            None for Python-only errors.
        target: The specific thing that failed.
            For element ops: the CSS selector (``".product-card"``).
            For Python errors: the attribute or key name.
            For network errors: the error code.
            None when not extractable.
        message: The full raw error string (preserved for debugging).
    """

    category: ErrorCategory
    error_type: str | None = None
    method: str | None = None
    target: str | None = None
    message: str = ""


# ── Stability assessment ──────────────────────────────────────


class StabilityLevel(enum.Enum):
    """How consistently the same error repeats across 3 attempts (spec S8).

    Determined by comparing fingerprints from all failed attempts.
    """

    STABLE = "stable"        # All 3 match at Level 1 or 2 (exact/same-kind)
    CONSISTENT = "consistent"  # All 3 same category, Level 3 match
    MIXED = "mixed"          # Exactly 2 categories
    CHAOTIC = "chaotic"      # 3 categories, or 2 with no clear majority


# ── Page verification ─────────────────────────────────────────


class PageVerificationResult(enum.Enum):
    """Whether the script ran against the real page (spec S6).

    Determined by HTTP status, URL domain match, and anti-bot detection.
    """

    REAL_PAGE = "real_page"      # HTTP 200 + same domain + no anti-bot + real content
    ANTI_BOT = "anti_bot"        # Anti-bot patterns detected
    SERVER_ERROR = "server_error"  # HTTP 5xx or 429
    REDIRECTED = "redirected"    # page.url domain differs from target
    NO_RESPONSE = "no_response"  # page.goto() failed, no Response object
    SOFT_BLOCK = "soft_block"    # HTTP 200 but not real content (login, maintenance, etc.)


@dataclass(frozen=True, slots=True)
class AntibotResult:
    """Result from anti-bot content/header detection (spec S6).

    Returned by ``detect_antibot()`` when anti-bot is detected.
    None is returned when no anti-bot is detected.

    Attributes:
        provider: Identified provider name, or None for generic/structural.
            Examples: ``"cloudflare"``, ``"akamai"``, ``"datadome"``.
        tier: Detection confidence tier (1=provider-specific, 2=generic, 3=structural).
        pattern_matched: Description of the pattern that triggered detection.
    """

    provider: str | None
    tier: int  # 1, 2, or 3
    pattern_matched: str


@dataclass(frozen=True, slots=True)
class SoftBlockResult:
    """Result from non-content page detection.

    Returned by ``detect_non_content()`` when the page is identified as
    a non-content page served with HTTP 200 (login wall, maintenance page,
    rate-limit page, etc.). None is returned when no non-content patterns
    are detected.

    Attributes:
        category: Detection category.
            One of: ``"login"``, ``"maintenance"``, ``"rate_limit"``,
            ``"geo_restriction"``, ``"suspended"``.
        pattern_matched: Description of the pattern that triggered detection.
    """

    category: str
    pattern_matched: str


@dataclass(frozen=True, slots=True)
class PageSignals:
    """Raw page-level signals collected after a script failure (spec S6/S7).

    Collected by the response listener and page inspection after each
    failed attempt. Fed into ``verify_page()`` to produce a
    ``PageVerificationResult``.

    Attributes:
        http_status: HTTP status code from the last document response.
            None if no response was captured.
        page_url: The final URL after all redirects (``page.url``).
            None if the page is closed/crashed.
        content: Page HTML content (``page.content()``).
            None if the page is closed/crashed or content retrieval failed.
        headers: Response headers from the last document response.
        cookies: Cookies from the response (list of dicts with name/value).
    """

    http_status: int | None = None
    page_url: str | None = None
    content: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    cookies: list[dict[str, str]] = field(default_factory=list)


# ── Auto-fix mode and action ─────────────────────────────────


class AutoFixMode(enum.Enum):
    """User-selected risk tolerance for automatic regeneration (spec S3).

    Controls evidence thresholds in the decision engine:
    - CONSERVATIVE: Only high-confidence categories (B, G). Never D/E.
    - BALANCED: High-confidence + ambiguous when page verified real.
    - AGGRESSIVE: Lower thresholds, tolerates some tainted attempts.
    """

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class AutoFixAction(enum.Enum):
    """Decision output from the decision engine (spec S9).

    Either regenerate the script or raise the error to the user.
    """

    REGENERATE = "regenerate"
    RAISE = "raise"


# ── Attempt and diagnosis results ─────────────────────────────


@dataclass(slots=True)
class AttemptResult:
    """Result of a single script execution attempt during diagnosis.

    Produced by the ``execute_fn`` adapter passed to ``diagnose()``.
    Captures everything needed for classification, fingerprinting,
    and page verification.

    Attributes:
        success: Whether the script produced valid output.
        data: The script's return value (only when success=True).
        error: Raw error string (stderr content) when success=False.
        exit_code: Subprocess exit code (None for in-process execution).
        schema_error: Schema validation error message (Category G).
            Set when the script returned data but schema rejected it.
        page_signals: Page-level signals collected after failure.
            None when success=True or when signals couldn't be collected.
        fingerprint: Populated by the diagnosis loop after classification.
        page_result: Populated by the diagnosis loop after page verification.
        category: Populated by the diagnosis loop after classification.
    """

    success: bool
    data: Any = None
    error: str | None = None
    exit_code: int | None = None
    schema_error: str | None = None
    page_signals: PageSignals | None = None

    # Populated during diagnosis (not by execute_fn)
    fingerprint: Fingerprint | None = None
    page_result: PageVerificationResult | None = None
    category: ErrorCategory | None = None


@dataclass(frozen=True, slots=True)
class DiagnosisResult:
    """Final output of the diagnosis loop (spec S7).

    Contains the decision (REGENERATE or RAISE), the evidence that
    led to it, and a formatted message for the user.

    Attributes:
        action: Whether to regenerate the script or raise an error.
        category: The error category from the first (or dominant) attempt.
        stability: Stability assessment across all failed attempts.
            None when retries were skipped (Category A, F2, F3).
        page_results: Page verification results for each attempt.
            Empty when page verification is not applicable.
        fingerprints: Error fingerprints for each failed attempt.
        attempts: Full attempt results (for post-regen validation).
        message: Formatted diagnostic message for the user (spec S12).
        e_eligible: Whether Category E errors are eligible for regeneration.
            Only meaningful when category is E. True by default.
    """

    action: AutoFixAction
    category: ErrorCategory
    stability: StabilityLevel | None = None
    page_results: list[PageVerificationResult] = field(default_factory=list)
    fingerprints: list[Fingerprint] = field(default_factory=list)
    attempts: list[AttemptResult] = field(default_factory=list)
    message: str = ""
    e_eligible: bool = True
