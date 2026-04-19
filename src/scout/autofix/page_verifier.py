"""Page verification — determines if the script ran against the real page.

Combines HTTP status, URL domain match, and anti-bot detection to produce
a ``PageVerificationResult``. Also provides ``check_page_gate()`` which
enforces the universal gate rule across all 3 diagnostic attempts.

Decision order for ``verify_page()`` (spec §6):
  1. No response data → NO_RESPONSE
  2. HTTP 5xx or 429 → SERVER_ERROR
  3. Domain mismatch → REDIRECTED
  4. Anti-bot detected → ANTI_BOT
  5. All checks pass → REAL_PAGE

Gate rules for ``check_page_gate()`` (spec §6):
  Conservative/Balanced: ALL 3 attempts must be REAL_PAGE.
  Aggressive: At least 2/3 REAL_PAGE, but any ANTI_BOT blocks all.

Spec reference: docs/specific/AUTO_FIX_ALGORITHM.md §6
"""

from __future__ import annotations

from typing import Sequence
from urllib.parse import urlparse

from scout.autofix.antibot import detect_antibot
from scout.autofix.types import (
    AutoFixMode,
    PageVerificationResult,
    PageSignals,
)


# ── Public API ───────────────────────────────────────────────


def verify_page(
    signals: PageSignals | None,
    target_url: str,
) -> PageVerificationResult:
    """Verify whether the script ran against the real page.

    Uses page-level signals (HTTP status, URL, content, headers, cookies)
    to determine what the script saw.

    Args:
        signals: Page signals collected after script failure.
            None when signals couldn't be collected (page crash, etc.).
        target_url: The URL the script was supposed to scrape.

    Returns:
        A ``PageVerificationResult`` indicating what the script saw.
    """
    # §6: No response data → NO_RESPONSE
    if signals is None:
        return PageVerificationResult.NO_RESPONSE

    if signals.http_status is None and signals.page_url is None:
        return PageVerificationResult.NO_RESPONSE

    # §6: HTTP 5xx or 429 → SERVER_ERROR
    if signals.http_status is not None:
        if signals.http_status >= 500 or signals.http_status == 429:
            return PageVerificationResult.SERVER_ERROR

    # §6: Domain mismatch → REDIRECTED
    if signals.page_url is not None and target_url:
        if not _domains_match(signals.page_url, target_url):
            return PageVerificationResult.REDIRECTED

    # §6: Anti-bot detected → ANTI_BOT
    antibot = detect_antibot(
        content=signals.content,
        headers=signals.headers,
        cookies=signals.cookies,
    )
    if antibot is not None:
        return PageVerificationResult.ANTI_BOT

    # All checks pass → REAL_PAGE
    return PageVerificationResult.REAL_PAGE


def check_page_gate(
    results: Sequence[PageVerificationResult],
    mode: AutoFixMode,
) -> tuple[bool, str]:
    """Check whether page verification results pass the universal gate.

    The gate determines whether regeneration can proceed based on
    how many attempts saw the real page.

    Args:
        results: Page verification results from all failed attempts
            (typically 3).
        mode: The user's auto-fix risk tolerance.

    Returns:
        Tuple of ``(passes_gate, reason_string)``.
        ``passes_gate`` is True if regeneration can proceed.
        ``reason_string`` explains why the gate passed or failed.
    """
    if not results:
        return False, "No page verification results available"

    real_count = sum(
        1 for r in results if r == PageVerificationResult.REAL_PAGE
    )
    has_antibot = any(
        r == PageVerificationResult.ANTI_BOT for r in results
    )
    total = len(results)

    # §6: ANTI_BOT blocks regeneration in ALL modes, even aggressive.
    if has_antibot:
        return False, (
            f"Anti-bot detected in {_count_of(results, PageVerificationResult.ANTI_BOT)}"
            f"/{total} attempts — regeneration blocked"
        )

    if mode in (AutoFixMode.CONSERVATIVE, AutoFixMode.BALANCED):
        # §6: Conservative/Balanced require ALL attempts to be REAL_PAGE.
        if real_count == total:
            return True, f"Page verified real ({real_count}/{total} attempts)"
        # Describe what tainted the results.
        taint = _describe_taint(results)
        return False, (
            f"Page not fully verified ({real_count}/{total} REAL_PAGE"
            f"{taint}) — regeneration blocked in {mode.value} mode"
        )

    # Aggressive mode: at least 2/3 REAL_PAGE (no ANTI_BOT, already checked).
    required = max(2, (total * 2 + 2) // 3)  # ceil(2/3 * total)
    if real_count >= required:
        return True, (
            f"Page verified real ({real_count}/{total} attempts, "
            "aggressive mode)"
        )
    taint = _describe_taint(results)
    return False, (
        f"Insufficient page verification ({real_count}/{total} REAL_PAGE, "
        f"need {required}{taint}) — regeneration blocked"
    )


# ── Internal helpers ─────────────────────────────────────────


def _domains_match(page_url: str, target_url: str) -> bool:
    """Check if two URLs have matching domains.

    Strips ``www.`` prefix and compares domains case-insensitively.
    Port differences are considered a mismatch. Path, query, and
    fragment are ignored.

    Handles edge cases:
    - ``data:`` and ``blob:`` URLs → always mismatch.
    - ``about:blank`` → always mismatch.
    - Missing scheme → treated as relative (mismatch).
    """
    try:
        page_parsed = urlparse(page_url)
        target_parsed = urlparse(target_url)
    except Exception:
        return False

    # data:, blob:, about: URLs are never a domain match.
    if page_parsed.scheme in ("data", "blob", "about"):
        return False

    page_host = _normalize_host(page_parsed.hostname)
    target_host = _normalize_host(target_parsed.hostname)

    if not page_host or not target_host:
        return False

    # Compare normalized hostnames.
    if page_host != target_host:
        return False

    # Port comparison: only check when at least one URL has an explicit port.
    # HTTP→HTTPS upgrades (80→443) with no explicit ports are normal browser
    # behavior, not redirects. Example: target http://example.com → page
    # https://example.com is fine (browser upgraded).
    page_port = page_parsed.port  # None if no explicit port
    target_port = target_parsed.port
    if page_port is not None or target_port is not None:
        # At least one URL has an explicit port — resolve and compare.
        resolved_page = page_port if page_port is not None else _default_port(page_parsed.scheme)
        resolved_target = target_port if target_port is not None else _default_port(target_parsed.scheme)
        if resolved_page != resolved_target:
            return False

    return True


def _normalize_host(hostname: str | None) -> str:
    """Normalize hostname: lowercase, strip www. prefix."""
    if hostname is None:
        return ""
    host = hostname.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _default_port(scheme: str) -> int | None:
    """Return default port for a scheme."""
    if scheme == "https":
        return 443
    if scheme == "http":
        return 80
    return None


def _count_of(
    results: Sequence[PageVerificationResult],
    value: PageVerificationResult,
) -> int:
    """Count occurrences of a value in results."""
    return sum(1 for r in results if r == value)


def _describe_taint(
    results: Sequence[PageVerificationResult],
) -> str:
    """Describe non-REAL_PAGE results for error messages."""
    taint_parts: list[str] = []
    for result_type in (
        PageVerificationResult.SERVER_ERROR,
        PageVerificationResult.REDIRECTED,
        PageVerificationResult.NO_RESPONSE,
        PageVerificationResult.ANTI_BOT,
    ):
        count = _count_of(results, result_type)
        if count > 0:
            taint_parts.append(f"{count}x {result_type.value}")
    if taint_parts:
        return ", " + ", ".join(taint_parts)
    return ""
