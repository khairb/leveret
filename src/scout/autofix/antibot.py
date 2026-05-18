"""Anti-bot content and header detection.

Detects whether a page response is from an anti-bot system rather than
the real website. Uses a tiered approach:

  Tier 1 — Provider-specific structural markers (high confidence)
  Tier 2 — Generic markers (only when response body < 10KB)
  Tier 3 — Structural integrity (suspiciously empty responses)

Stops at the first match. Returns ``AntibotResult`` if detected,
``None`` if the page appears clean.

Spec reference: docs/specific/AUTO_FIX_ALGORITHM.md §6
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from scout.autofix.types import AntibotResult

# ── Constants ────────────────────────────────────────────────

# Tier 2 size gate: only check generic markers on small pages.
# Large legitimate pages can incidentally contain words like "access denied".
_TIER_2_MAX_SIZE = 10_240  # 10KB

# Tier 3 thresholds for structural integrity checks.
_TIER_3_EMPTY_THRESHOLD = 100  # < 100 bytes = effectively empty
_TIER_3_SHELL_THRESHOLD = 500  # < 500 bytes = possible empty shell

# ── Tier 1: Provider-specific content patterns ───────────────
#
# Each provider has content regex patterns. Checked first because
# they are high-confidence and unlikely to false-positive on
# legitimate pages.

# -- Cloudflare --
_CF_CHALLENGE_FORM_RE = re.compile(
    r"challenge-form.*__cf_chl_f_tk=",
    re.DOTALL,
)
_CF_ORCHESTRATE_RE = re.compile(
    r"/cdn-cgi/challenge-platform/\S+orchestrate",
)
_CF_ERROR_CODE_RE = re.compile(
    r'<span\s+class="cf-error-code">\d{4}</span>',
)
_CF_JUST_A_MOMENT_RE = re.compile(
    r"<title>\s*Just\s+a\s+moment",
)

# -- Akamai --
_AKAMAI_PARDON_RE = re.compile(
    r"Pardon\s+Our\s+Interruption",
)
_AKAMAI_REFERENCE_RE = re.compile(
    r"Reference\s*#\s*[\d]+\.[0-9a-f]+\.\d+\.[0-9a-f]+",
)

# -- PerimeterX --
_PX_APP_ID_RE = re.compile(r"window\._pxAppId\s*=")
_PX_CAPTCHA_CDN_RE = re.compile(r"captcha\.px-cdn\.net")

# -- DataDome --
_DD_CAPTCHA_DELIVERY_RE = re.compile(r"captcha-delivery\.com")

# -- Imperva / Incapsula --
_IMPERVA_RESOURCE_RE = re.compile(r"_Incapsula_Resource")
_IMPERVA_INCIDENT_RE = re.compile(r"Incapsula\s+incident\s+ID")

# -- Kasada --
_KASADA_RE = re.compile(r"KPSDK\.scriptStart\s*=\s*KPSDK\.now\(\)")


# ── Tier 1: Header/cookie patterns ──────────────────────────
#
# Some providers are detectable via response headers or cookies
# alone, or headers/cookies strengthen a content match.
# Header names are compared case-insensitively.

# Provider → (header_name_lower, value_pattern_or_None)
# value_pattern is a regex if not None; None means any value matches.
_HEADER_PATTERNS: dict[str, list[tuple[str, re.Pattern[str] | None]]] = {
    "cloudflare": [
        ("cf-mitigated", re.compile(r"challenge", re.IGNORECASE)),
    ],
    "perimeterx": [
        ("x-px-authorization", None),
    ],
    "datadome": [
        ("x-dd-b", None),
    ],
    "imperva": [
        ("x-cdn", re.compile(r"Incapsula", re.IGNORECASE)),
        ("x-iinfo", None),
    ],
    "kasada": [
        ("x-kasada", None),
    ],
    "aws_waf": [
        ("x-amzn-waf-action", None),
    ],
}

# Provider → list of cookie names
_COOKIE_PATTERNS: dict[str, list[str]] = {
    "cloudflare": ["cf_clearance"],
    "akamai": ["_abck"],
    "perimeterx": ["_px3", "_pxhd"],
    "datadome": ["datadome"],
}


# ── Tier 2: Generic markers ─────────────────────────────────
#
# Only checked when body < 10KB to prevent false matches.

_GENERIC_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"Access\s+Denied", re.IGNORECASE), "Access Denied"),
    (re.compile(r"Checking\s+your\s+browser", re.IGNORECASE), "Checking your browser"),
    (re.compile(r'class=["\']g-recaptcha["\']'), "reCAPTCHA widget"),
    (re.compile(r'class=["\']h-captcha["\']'), "hCaptcha widget"),
    (
        re.compile(
            r"Access\s+to\s+This\s+Page\s+Has\s+Been\s+Blocked",
            re.IGNORECASE,
        ),
        "Access to This Page Has Been Blocked",
    ),
    (re.compile(r"blocked\s+by\s+security", re.IGNORECASE), "blocked by security"),
    (re.compile(r"Request\s+unsuccessful", re.IGNORECASE), "Request unsuccessful"),
]


# ── Tier 3: Structural integrity ────────────────────────────
#
# Content element tags that indicate real page content.
_CONTENT_TAGS_RE = re.compile(
    r"<(?:p|h[1-6]|article)[\s>]",
    re.IGNORECASE,
)

# Count visible text characters (strip HTML tags, collapse whitespace).
_TAG_STRIP_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")

# Script tags.
_SCRIPT_TAG_RE = re.compile(r"<script[\s>]", re.IGNORECASE)


# ── Public API ───────────────────────────────────────────────


def detect_antibot(
    content: str | None,
    headers: dict[str, str] | None = None,
    cookies: Sequence[dict[str, str]] | None = None,
) -> AntibotResult | None:
    """Detect whether a page response is from an anti-bot system.

    Uses a tiered approach (spec §6):
      1. Provider-specific patterns (content + headers/cookies)
      2. Generic markers (only when body < 10KB)
      3. Structural integrity (suspiciously empty/shell pages)

    Stops at the first match.

    Args:
        content: Page HTML content (``page.content()``).
            None if content could not be retrieved.
        headers: Response headers (keys may be any case).
        cookies: Response cookies (list of dicts with ``name`` key).

    Returns:
        ``AntibotResult`` if anti-bot is detected, ``None`` if clean.
    """
    if headers is None:
        headers = {}
    if cookies is None:
        cookies = []

    # Normalize header keys to lowercase for case-insensitive comparison.
    headers_lower = {k.lower(): v for k, v in headers.items()}

    # Build cookie name set for fast lookup.
    cookie_names = {c.get("name", "") for c in cookies}

    # ── Tier 1: Provider-specific ────────────────────────────

    # Check headers-only providers first (AWS WAF has no content pattern).
    result = _check_header_only_providers(headers_lower)
    if result is not None:
        return result

    # Check content + header/cookie providers.
    if content is not None:
        result = _check_tier1_content(content, headers_lower, cookie_names)
        if result is not None:
            return result

    # Check cookie-only signals (e.g., Akamai _abck without content match).
    # These are weaker signals on their own, but combined with the absence
    # of real content they're meaningful. We check them after content patterns
    # so content matches take priority.

    # ── Tier 2: Generic markers (only < 10KB) ────────────────

    if content is not None and len(content) < _TIER_2_MAX_SIZE:
        result = _check_tier2_generic(content)
        if result is not None:
            return result

    # ── Tier 3: Structural integrity ─────────────────────────

    if content is not None:
        result = _check_tier3_structural(content)
        if result is not None:
            return result

    return None


# ── Tier 1 internals ─────────────────────────────────────────


def _check_header_only_providers(
    headers_lower: dict[str, str],
) -> AntibotResult | None:
    """Check providers detectable by headers alone (e.g., AWS WAF)."""
    # AWS WAF: x-amzn-waf-action header
    if "x-amzn-waf-action" in headers_lower:
        return AntibotResult(
            provider="aws_waf",
            tier=1,
            pattern_matched="x-amzn-waf-action header",
        )
    return None


def _check_tier1_content(
    content: str,
    headers_lower: dict[str, str],
    cookie_names: set[str],
) -> AntibotResult | None:
    """Check Tier 1 provider-specific content patterns."""
    # -- Cloudflare --
    # §6: 4 content patterns + header + cookie
    if _CF_CHALLENGE_FORM_RE.search(content):
        return AntibotResult(
            provider="cloudflare",
            tier=1,
            pattern_matched="challenge-form with __cf_chl_f_tk",
        )
    if _CF_ORCHESTRATE_RE.search(content):
        return AntibotResult(
            provider="cloudflare",
            tier=1,
            pattern_matched="/cdn-cgi/challenge-platform/*/orchestrate",
        )
    if _CF_ERROR_CODE_RE.search(content):
        return AntibotResult(
            provider="cloudflare",
            tier=1,
            pattern_matched="cf-error-code span",
        )
    if _CF_JUST_A_MOMENT_RE.search(content):
        return AntibotResult(
            provider="cloudflare",
            tier=1,
            pattern_matched="<title>Just a moment",
        )
    # Cloudflare header: cf-mitigated: challenge
    if _check_header("cloudflare", headers_lower):
        return AntibotResult(
            provider="cloudflare",
            tier=1,
            pattern_matched="cf-mitigated: challenge header",
        )
    # Cloudflare cookie: cf_clearance
    if _check_cookies("cloudflare", cookie_names):
        # §6: cf_clearance alone is weaker — it persists after challenge
        # is solved. Only flag if there's also a small page or other signal.
        # For now, treat it as Tier 1 (spec lists it there).
        return AntibotResult(
            provider="cloudflare",
            tier=1,
            pattern_matched="cf_clearance cookie",
        )

    # -- Akamai --
    if _AKAMAI_PARDON_RE.search(content):
        return AntibotResult(
            provider="akamai",
            tier=1,
            pattern_matched="Pardon Our Interruption",
        )
    if _AKAMAI_REFERENCE_RE.search(content):
        return AntibotResult(
            provider="akamai",
            tier=1,
            pattern_matched="Akamai reference pattern",
        )
    # Akamai cookie: _abck
    if _check_cookies("akamai", cookie_names):
        return AntibotResult(
            provider="akamai",
            tier=1,
            pattern_matched="_abck cookie",
        )

    # -- PerimeterX --
    if _PX_APP_ID_RE.search(content):
        return AntibotResult(
            provider="perimeterx",
            tier=1,
            pattern_matched="window._pxAppId",
        )
    if _PX_CAPTCHA_CDN_RE.search(content):
        return AntibotResult(
            provider="perimeterx",
            tier=1,
            pattern_matched="captcha.px-cdn.net",
        )
    # PerimeterX header + cookies
    if _check_header("perimeterx", headers_lower):
        return AntibotResult(
            provider="perimeterx",
            tier=1,
            pattern_matched="x-px-authorization header",
        )
    if _check_cookies("perimeterx", cookie_names):
        return AntibotResult(
            provider="perimeterx",
            tier=1,
            pattern_matched="_px3/_pxhd cookie",
        )

    # -- DataDome --
    if _DD_CAPTCHA_DELIVERY_RE.search(content):
        return AntibotResult(
            provider="datadome",
            tier=1,
            pattern_matched="captcha-delivery.com",
        )
    # DataDome header/cookie
    if _check_header("datadome", headers_lower):
        return AntibotResult(
            provider="datadome",
            tier=1,
            pattern_matched="x-dd-b header",
        )
    if _check_cookies("datadome", cookie_names):
        return AntibotResult(
            provider="datadome",
            tier=1,
            pattern_matched="datadome cookie",
        )

    # -- Imperva / Incapsula --
    if _IMPERVA_RESOURCE_RE.search(content):
        return AntibotResult(
            provider="imperva",
            tier=1,
            pattern_matched="_Incapsula_Resource",
        )
    if _IMPERVA_INCIDENT_RE.search(content):
        return AntibotResult(
            provider="imperva",
            tier=1,
            pattern_matched="Incapsula incident ID",
        )
    # Imperva headers
    if _check_header("imperva", headers_lower):
        return AntibotResult(
            provider="imperva",
            tier=1,
            pattern_matched="x-cdn: Incapsula or x-iinfo header",
        )

    # -- Kasada --
    if _KASADA_RE.search(content):
        return AntibotResult(
            provider="kasada",
            tier=1,
            pattern_matched="KPSDK.scriptStart = KPSDK.now()",
        )
    if _check_header("kasada", headers_lower):
        return AntibotResult(
            provider="kasada",
            tier=1,
            pattern_matched="x-kasada header",
        )

    return None


def _check_header(
    provider: str,
    headers_lower: dict[str, str],
) -> bool:
    """Check if any header pattern matches for a provider."""
    patterns = _HEADER_PATTERNS.get(provider, [])
    for header_name, value_pattern in patterns:
        if header_name in headers_lower:
            if value_pattern is None:
                return True
            if value_pattern.search(headers_lower[header_name]):
                return True
    return False


def _check_cookies(
    provider: str,
    cookie_names: set[str],
) -> bool:
    """Check if any cookie name matches for a provider."""
    expected = _COOKIE_PATTERNS.get(provider, [])
    return any(name in cookie_names for name in expected)


# ── Tier 2 internals ─────────────────────────────────────────


def _check_tier2_generic(content: str) -> AntibotResult | None:
    """Check generic anti-bot markers (only for pages < 10KB)."""
    for pattern, description in _GENERIC_PATTERNS:
        if pattern.search(content):
            return AntibotResult(
                provider=None,
                tier=2,
                pattern_matched=description,
            )
    return None


# ── Tier 3 internals ─────────────────────────────────────────


def _check_tier3_structural(content: str) -> AntibotResult | None:
    """Check for structurally empty/shell pages.

    Three checks (spec §6, Tier 3):
    1. HTTP 200 but body < 100 bytes → effectively empty.
    2. Body < 500 bytes AND no content tags → empty shell.
    3. Scripts present, zero content elements, < 100 visible chars → script-only.
    """
    content_len = len(content)

    # Check 1: Effectively empty (< 100 bytes AND no content tags).
    # §6: "HTTP 200 but response body < 100 bytes" — truly empty responses.
    # If content tags are present, the page has real content even if short.
    if content_len < _TIER_3_EMPTY_THRESHOLD:
        if not _CONTENT_TAGS_RE.search(content):
            return AntibotResult(
                provider=None,
                tier=3,
                pattern_matched=f"effectively empty ({content_len} bytes)",
            )

    # Check 2: Empty shell (< 500 bytes, no content tags).
    if content_len < _TIER_3_SHELL_THRESHOLD:
        if not _CONTENT_TAGS_RE.search(content):
            return AntibotResult(
                provider=None,
                tier=3,
                pattern_matched=f"empty shell ({content_len} bytes, no content tags)",
            )

    # Check 3: Script-only shell — has scripts but no content elements
    # and very little visible text.
    if _SCRIPT_TAG_RE.search(content) and not _CONTENT_TAGS_RE.search(content):
        visible_text = _extract_visible_text(content)
        if len(visible_text) < 100:
            return AntibotResult(
                provider=None,
                tier=3,
                pattern_matched=(
                    f"script-only shell ({len(visible_text)} visible chars, no content tags)"
                ),
            )

    return None


def _extract_visible_text(html: str) -> str:
    """Extract visible text from HTML by stripping tags and collapsing whitespace."""
    # Remove script and style blocks entirely.
    no_scripts = re.sub(
        r"<(script|style)[^>]*>.*?</\1>",
        "",
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Strip remaining tags.
    text = _TAG_STRIP_RE.sub("", no_scripts)
    # Collapse whitespace.
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text
