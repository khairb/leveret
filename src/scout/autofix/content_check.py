"""Non-content page detection — identifies HTTP 200 pages that aren't real content.

Detects login walls, maintenance pages, non-standard rate limiting,
geographic restrictions, and account suspensions. Uses size-gated
pattern matching: patterns are only checked when the response body
is under 10KB, preventing false positives on real content pages.

Runs AFTER anti-bot detection and BEFORE returning REAL_PAGE in the
page verification pipeline.

Design reference: docs/specific/AUTOFIX_BATTLE_TEST_FIXES.md §3
"""

from __future__ import annotations

import re

from scout.autofix.types import SoftBlockResult

# ── Size gate ──────────────────────────────────────────────────
#
# Only check patterns on pages under 10KB. Real content pages
# (product listings, articles) are typically 30-200KB. Non-content
# pages (login forms, maintenance notices) are typically 1-8KB.
# Same threshold as anti-bot Tier 2 detection.

_MAX_SIZE = 10_240  # 10KB


# ── Category 1: Login / Authentication ─────────────────────────

# Primary: password input field (very high confidence)
_LOGIN_PASSWORD_RE = re.compile(
    r"<input[^>]+type=[\"']password[\"']",
    re.IGNORECASE,
)

# Secondary: title contains login/signin + page has a form
_LOGIN_TITLE_RE = re.compile(
    r"<title>[^<]*(sign\s*in|log\s*in|login)[^<]*</title>",
    re.IGNORECASE,
)
_FORM_TAG_RE = re.compile(r"<form[\s>]", re.IGNORECASE)


# ── Category 2: Maintenance / Error Pages ──────────────────────

_MAINTENANCE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"under\s+maintenance", re.IGNORECASE), "under maintenance"),
    (re.compile(r"maintenance\s+mode", re.IGNORECASE), "maintenance mode"),
    (re.compile(r"we[''\u2019]ll\s+be\s+(right\s+)?back", re.IGNORECASE), "we'll be back"),
    (re.compile(r"we\s+will\s+be\s+(right\s+)?back", re.IGNORECASE), "we will be back"),
    (re.compile(r"something\s+went\s+wrong", re.IGNORECASE), "something went wrong"),
    (re.compile(r"temporarily\s+unavailable", re.IGNORECASE), "temporarily unavailable"),
    (
        re.compile(r"scheduled\s+(down\s*time|maintenance)", re.IGNORECASE),
        "scheduled downtime/maintenance",
    ),
    (re.compile(r"currently\s+unavailable", re.IGNORECASE), "currently unavailable"),
    (re.compile(r"be\s+right\s+back", re.IGNORECASE), "be right back"),
    (
        re.compile(
            r"experiencing\s+(technical\s+)?difficulties",
            re.IGNORECASE,
        ),
        "experiencing difficulties",
    ),
    (re.compile(r"under\s+construction", re.IGNORECASE), "under construction"),
]

# "coming soon" restricted to <title> or <h1> to avoid product teasers
_COMING_SOON_TITLE_RE = re.compile(
    r"<title>[^<]*coming\s+soon[^<]*</title>",
    re.IGNORECASE,
)
_COMING_SOON_H1_RE = re.compile(
    r"<h1[^>]*>[^<]*coming\s+soon[^<]*</h1>",
    re.IGNORECASE,
)


# ── Category 3: Non-Standard Rate Limiting ─────────────────────

_RATE_LIMIT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"too\s+many\s+requests", re.IGNORECASE), "too many requests"),
    (re.compile(r"rate\s+limit", re.IGNORECASE), "rate limit"),
    (
        re.compile(
            r"you[''\u2019]re\s+browsing\s+too\s+fast",
            re.IGNORECASE,
        ),
        "browsing too fast",
    ),
]

# "slow down" requires context words to avoid false positives
_SLOW_DOWN_RE = re.compile(
    r"slow\s+down",
    re.IGNORECASE,
)
_SLOW_DOWN_CONTEXT_RE = re.compile(
    r"request|try\s+again|wait",
    re.IGNORECASE,
)

# Auto-refresh meta tag + wait/try/moment context
_META_REFRESH_RE = re.compile(
    r'<meta\s+http-equiv=["\']refresh["\']',
    re.IGNORECASE,
)
_META_REFRESH_CONTEXT_RE = re.compile(
    r"wait|try|moment",
    re.IGNORECASE,
)


# ── Category 4: Geographic Restrictions ────────────────────────

_GEO_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"not\s+available\s+in\s+your\s+(country|region|area|location)",
            re.IGNORECASE,
        ),
        "not available in your region",
    ),
    (re.compile(r"geo(graphically)?\s*restrict", re.IGNORECASE), "geographically restricted"),
    (re.compile(r"region(al)?\s+restriction", re.IGNORECASE), "regional restriction"),
    (
        re.compile(
            r"blocked\s+in\s+your\s+(country|region)",
            re.IGNORECASE,
        ),
        "blocked in your region",
    ),
    (
        re.compile(
            r"not\s+accessible\s+from\s+your\s+location",
            re.IGNORECASE,
        ),
        "not accessible from your location",
    ),
]


# ── Category 5: Account Suspended / Banned ─────────────────────

_SUSPENDED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"account\s+(has\s+been\s+)?(suspended|banned|disabled|terminated)",
            re.IGNORECASE,
        ),
        "account suspended/banned",
    ),
    (
        re.compile(
            r"(ip|address)\s+(has\s+been\s+)?(blocked|banned)",
            re.IGNORECASE,
        ),
        "IP blocked/banned",
    ),
]


# ── Public API ───────────────────────────────────────────────


def detect_non_content(
    content: str | None,
) -> SoftBlockResult | None:
    """Detect whether a page is a non-content page served as HTTP 200.

    Checks for login walls, maintenance pages, rate-limit pages,
    geographic restrictions, and account suspensions using size-gated
    pattern matching (only pages < 10KB are checked).

    Args:
        content: Page HTML content (``page.content()``).
            None if content could not be retrieved.

    Returns:
        ``SoftBlockResult`` if a non-content page is detected,
        ``None`` if the page appears to be real content.
    """
    if content is None:
        return None

    if len(content) >= _MAX_SIZE:
        return None

    # Category 1: Login / Authentication
    result = _check_login(content)
    if result is not None:
        return result

    # Category 2: Maintenance / Error
    result = _check_maintenance(content)
    if result is not None:
        return result

    # Category 3: Rate Limiting
    result = _check_rate_limit(content)
    if result is not None:
        return result

    # Category 4: Geographic Restrictions
    result = _check_geo_restriction(content)
    if result is not None:
        return result

    # Category 5: Account Suspended
    result = _check_suspended(content)
    if result is not None:
        return result

    return None


# ── Internal checks ──────────────────────────────────────────


def _check_login(content: str) -> SoftBlockResult | None:
    """Check for login/authentication pages."""
    # Primary: password input field (very high confidence)
    if _LOGIN_PASSWORD_RE.search(content):
        return SoftBlockResult(
            category="login",
            pattern_matched="password input field",
        )

    # Secondary: title with sign in/log in + form element
    if _LOGIN_TITLE_RE.search(content) and _FORM_TAG_RE.search(content):
        return SoftBlockResult(
            category="login",
            pattern_matched="login page title with form",
        )

    return None


def _check_maintenance(content: str) -> SoftBlockResult | None:
    """Check for maintenance/error pages."""
    for pattern, description in _MAINTENANCE_PATTERNS:
        if pattern.search(content):
            return SoftBlockResult(
                category="maintenance",
                pattern_matched=description,
            )

    # "coming soon" restricted to title/h1
    if _COMING_SOON_TITLE_RE.search(content):
        return SoftBlockResult(
            category="maintenance",
            pattern_matched="coming soon (in title)",
        )
    if _COMING_SOON_H1_RE.search(content):
        return SoftBlockResult(
            category="maintenance",
            pattern_matched="coming soon (in h1)",
        )

    return None


def _check_rate_limit(content: str) -> SoftBlockResult | None:
    """Check for non-standard rate limiting pages."""
    for pattern, description in _RATE_LIMIT_PATTERNS:
        if pattern.search(content):
            return SoftBlockResult(
                category="rate_limit",
                pattern_matched=description,
            )

    # "slow down" with context
    if _SLOW_DOWN_RE.search(content) and _SLOW_DOWN_CONTEXT_RE.search(content):
        return SoftBlockResult(
            category="rate_limit",
            pattern_matched="slow down (with request/wait context)",
        )

    # Auto-refresh meta + wait/try/moment context
    if _META_REFRESH_RE.search(content) and _META_REFRESH_CONTEXT_RE.search(content):
        return SoftBlockResult(
            category="rate_limit",
            pattern_matched="auto-refresh with wait/try context",
        )

    return None


def _check_geo_restriction(content: str) -> SoftBlockResult | None:
    """Check for geographic restriction pages."""
    for pattern, description in _GEO_PATTERNS:
        if pattern.search(content):
            return SoftBlockResult(
                category="geo_restriction",
                pattern_matched=description,
            )
    return None


def _check_suspended(content: str) -> SoftBlockResult | None:
    """Check for account/IP suspension pages."""
    for pattern, description in _SUSPENDED_PATTERNS:
        if pattern.search(content):
            return SoftBlockResult(
                category="suspended",
                pattern_matched=description,
            )
    return None
