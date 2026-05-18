"""Tests for the anti-bot detector.

Tests use real HTML fixtures from Phase 0 (tests/autofix/fixtures/antibot/)
and cover all 3 detection tiers:
  - Tier 1: Provider-specific patterns (Cloudflare, Akamai, PerimeterX,
            DataDome, Imperva, Kasada, AWS WAF)
  - Tier 2: Generic markers (Access Denied, reCAPTCHA, hCaptcha, etc.)
  - Tier 3: Structural integrity (empty shell, script-only shell)
  - Negative controls: large legitimate pages, clean pages
"""

from __future__ import annotations

from scout.autofix.antibot import detect_antibot
from tests.autofix.conftest import load_antibot_page

# ── Tier 1: Cloudflare ───────────────────────────────────────


class TestCloudflare:
    """Cloudflare detection — 4 content patterns + header + cookie."""

    def test_challenge_form(self):
        content, headers, cookies = load_antibot_page("cloudflare_challenge")
        result = detect_antibot(content)
        assert result is not None
        assert result.provider == "cloudflare"
        assert result.tier == 1

    def test_challenge_with_headers(self):
        content, headers, cookies = load_antibot_page("cloudflare_challenge")
        result = detect_antibot(content, headers, cookies)
        assert result is not None
        assert result.provider == "cloudflare"
        assert result.tier == 1

    def test_error_1020(self):
        content, _, _ = load_antibot_page("cloudflare_error_1020")
        result = detect_antibot(content)
        assert result is not None
        assert result.provider == "cloudflare"
        assert result.tier == 1
        assert "cf-error-code" in result.pattern_matched

    def test_just_a_moment_title(self):
        content = "<html><head><title>Just a moment...</title></head><body></body></html>"
        result = detect_antibot(content)
        assert result is not None
        assert result.provider == "cloudflare"

    def test_just_a_moment_with_whitespace(self):
        content = "<html><head><title>  Just  a  moment  </title></head><body></body></html>"
        result = detect_antibot(content)
        assert result is not None
        assert result.provider == "cloudflare"

    def test_orchestrate_script(self):
        content = '<script src="/cdn-cgi/challenge-platform/h/g/orchestrate/chl_page/v1"></script>'
        result = detect_antibot(content)
        assert result is not None
        assert result.provider == "cloudflare"
        assert "orchestrate" in result.pattern_matched

    def test_cf_mitigated_header_only(self):
        result = detect_antibot(
            "<html><body><p>Normal page</p></body></html>",
            headers={"cf-mitigated": "challenge"},
        )
        assert result is not None
        assert result.provider == "cloudflare"
        assert "cf-mitigated" in result.pattern_matched

    def test_cf_mitigated_header_case_insensitive(self):
        result = detect_antibot(
            "<html><body><p>Normal page</p></body></html>",
            headers={"CF-Mitigated": "Challenge"},
        )
        assert result is not None
        assert result.provider == "cloudflare"

    def test_cf_clearance_cookie(self):
        result = detect_antibot(
            "<html><body><p>Normal page</p></body></html>",
            cookies=[{"name": "cf_clearance", "value": "abc"}],
        )
        assert result is not None
        assert result.provider == "cloudflare"
        assert "cf_clearance" in result.pattern_matched


# ── Tier 1: Akamai ───────────────────────────────────────────


class TestAkamai:
    """Akamai detection — 2 content patterns + cookie."""

    def test_block_page(self):
        content, headers, cookies = load_antibot_page("akamai_block")
        result = detect_antibot(content, headers, cookies)
        assert result is not None
        assert result.provider == "akamai"
        assert result.tier == 1

    def test_pardon_pattern(self):
        content = "<html><body><h1>Pardon Our Interruption</h1></body></html>"
        result = detect_antibot(content)
        assert result is not None
        assert result.provider == "akamai"

    def test_reference_pattern(self):
        content = "<html><body>Reference # 18.abc123.1234567890.abcdef</body></html>"
        result = detect_antibot(content)
        assert result is not None
        assert result.provider == "akamai"

    def test_abck_cookie(self):
        result = detect_antibot(
            "<html><body><p>Normal page</p></body></html>",
            cookies=[{"name": "_abck", "value": "xyz"}],
        )
        assert result is not None
        assert result.provider == "akamai"
        assert "_abck" in result.pattern_matched


# ── Tier 1: PerimeterX ───────────────────────────────────────


class TestPerimeterX:
    """PerimeterX detection — 2 content patterns + cookies + header."""

    def test_challenge_page(self):
        content, headers, cookies = load_antibot_page("perimeterx_challenge")
        result = detect_antibot(content, headers, cookies)
        assert result is not None
        assert result.provider == "perimeterx"
        assert result.tier == 1

    def test_px_app_id(self):
        content = "<script>window._pxAppId = 'PX123';</script>"
        result = detect_antibot(content)
        assert result is not None
        assert result.provider == "perimeterx"

    def test_px_captcha_cdn(self):
        content = '<script src="https://captcha.px-cdn.net/PX123/captcha.js"></script>'
        result = detect_antibot(content)
        assert result is not None
        assert result.provider == "perimeterx"

    def test_px_authorization_header(self):
        result = detect_antibot(
            "<html><body><p>Normal page</p></body></html>",
            headers={"x-px-authorization": "1"},
        )
        assert result is not None
        assert result.provider == "perimeterx"

    def test_px_cookies(self):
        result = detect_antibot(
            "<html><body><p>Normal page</p></body></html>",
            cookies=[{"name": "_px3", "value": "abc"}],
        )
        assert result is not None
        assert result.provider == "perimeterx"

    def test_pxhd_cookie(self):
        result = detect_antibot(
            "<html><body><p>Normal page</p></body></html>",
            cookies=[{"name": "_pxhd", "value": "def"}],
        )
        assert result is not None
        assert result.provider == "perimeterx"


# ── Tier 1: DataDome ─────────────────────────────────────────


class TestDataDome:
    """DataDome detection — 1 content pattern + header + cookie."""

    def test_captcha_page(self):
        content, headers, cookies = load_antibot_page("datadome_captcha")
        result = detect_antibot(content, headers, cookies)
        assert result is not None
        assert result.provider == "datadome"
        assert result.tier == 1

    def test_captcha_delivery_domain(self):
        content = '<iframe src="https://captcha-delivery.com/captcha/check"></iframe>'
        result = detect_antibot(content)
        assert result is not None
        assert result.provider == "datadome"

    def test_x_dd_b_header(self):
        result = detect_antibot(
            "<html><body><p>Normal page</p></body></html>",
            headers={"x-dd-b": "1"},
        )
        assert result is not None
        assert result.provider == "datadome"

    def test_datadome_cookie(self):
        result = detect_antibot(
            "<html><body><p>Normal page</p></body></html>",
            cookies=[{"name": "datadome", "value": "abc123"}],
        )
        assert result is not None
        assert result.provider == "datadome"


# ── Tier 1: Imperva / Incapsula ──────────────────────────────


class TestImperva:
    """Imperva detection — 2 content patterns + 2 headers."""

    def test_block_page(self):
        content, headers, cookies = load_antibot_page("imperva_block")
        result = detect_antibot(content, headers, cookies)
        assert result is not None
        assert result.provider == "imperva"
        assert result.tier == 1

    def test_incapsula_resource(self):
        content = "<script>var _Incapsula_Resource = {};</script>"
        result = detect_antibot(content)
        assert result is not None
        assert result.provider == "imperva"

    def test_incapsula_incident_id(self):
        content = "<html><body>Incapsula incident ID: 12345678</body></html>"
        result = detect_antibot(content)
        assert result is not None
        assert result.provider == "imperva"

    def test_x_cdn_incapsula_header(self):
        result = detect_antibot(
            "<html><body><p>Normal page</p></body></html>",
            headers={"x-cdn": "Incapsula"},
        )
        assert result is not None
        assert result.provider == "imperva"

    def test_x_iinfo_header(self):
        result = detect_antibot(
            "<html><body><p>Normal page</p></body></html>",
            headers={"x-iinfo": "1-2-3"},
        )
        assert result is not None
        assert result.provider == "imperva"


# ── Tier 1: Kasada ───────────────────────────────────────────


class TestKasada:
    """Kasada detection — 1 content pattern + header."""

    def test_challenge_page(self):
        content, headers, cookies = load_antibot_page("kasada_challenge")
        result = detect_antibot(content, headers, cookies)
        assert result is not None
        assert result.provider == "kasada"
        assert result.tier == 1

    def test_kpsdk_script_start(self):
        content = "<script>KPSDK.scriptStart = KPSDK.now();</script>"
        result = detect_antibot(content)
        assert result is not None
        assert result.provider == "kasada"

    def test_x_kasada_header(self):
        result = detect_antibot(
            "<html><body><p>Normal page</p></body></html>",
            headers={"x-kasada": "1"},
        )
        assert result is not None
        assert result.provider == "kasada"


# ── Tier 1: AWS WAF ──────────────────────────────────────────


class TestAWSWAF:
    """AWS WAF detection — header only (no content pattern)."""

    def test_waf_action_header(self):
        result = detect_antibot(
            "<html><body><p>Normal page</p></body></html>",
            headers={"x-amzn-waf-action": "captcha"},
        )
        assert result is not None
        assert result.provider == "aws_waf"
        assert result.tier == 1

    def test_waf_header_from_fixture(self):
        """AWS WAF is header-only — load headers directly."""
        import json

        from tests.autofix.conftest import ANTIBOT_DIR

        headers_path = ANTIBOT_DIR / "aws_waf.headers.json"
        data = json.loads(headers_path.read_text(encoding="utf-8"))
        result = detect_antibot(
            "<html><body><h1>Normal</h1><p>Normal page content here.</p></body></html>",
            headers=data.get("headers", {}),
            cookies=data.get("cookies", []),
        )
        assert result is not None
        assert result.provider == "aws_waf"

    def test_waf_header_any_value(self):
        result = detect_antibot(
            "<html><body><p>Normal page</p></body></html>",
            headers={"x-amzn-waf-action": "block"},
        )
        assert result is not None
        assert result.provider == "aws_waf"


# ── Tier 2: Generic markers ─────────────────────────────────


class TestTier2Generic:
    """Generic anti-bot markers — only checked when body < 10KB."""

    def test_access_denied(self):
        content, _, _ = load_antibot_page("generic_access_denied")
        result = detect_antibot(content)
        assert result is not None
        assert result.tier == 2
        assert result.provider is None
        assert "Access Denied" in result.pattern_matched

    def test_recaptcha_widget(self):
        content, _, _ = load_antibot_page("generic_recaptcha")
        result = detect_antibot(content)
        assert result is not None
        assert result.tier == 2
        assert "reCAPTCHA" in result.pattern_matched

    def test_hcaptcha_widget(self):
        content, _, _ = load_antibot_page("generic_hcaptcha")
        result = detect_antibot(content)
        assert result is not None
        assert result.tier == 2
        assert "hCaptcha" in result.pattern_matched

    def test_checking_your_browser(self):
        content = "<html><body>Checking your browser before accessing the site.</body></html>"
        result = detect_antibot(content)
        assert result is not None
        assert result.tier == 2

    def test_blocked_by_security(self):
        content = "<html><body>This page has been blocked by security measures.</body></html>"
        result = detect_antibot(content)
        assert result is not None
        assert result.tier == 2

    def test_access_to_page_blocked(self):
        content = "<html><body>Access to This Page Has Been Blocked</body></html>"
        result = detect_antibot(content)
        assert result is not None
        assert result.tier == 2

    def test_request_unsuccessful(self):
        content = "<html><body>Request unsuccessful. Please try again.</body></html>"
        result = detect_antibot(content)
        assert result is not None
        assert result.tier == 2

    def test_large_page_does_not_trigger_tier2(self):
        """Pages > 10KB must NOT trigger Tier 2 (false positive prevention).

        The legitimate_large_page.html fixture contains "access denied" in
        article text but is > 10KB, so Tier 2 should not match.
        """
        content, _, _ = load_antibot_page("legitimate_large_page")
        assert len(content) > 10_240, "Fixture must be > 10KB for this test"
        result = detect_antibot(content)
        # Should be None — large legitimate page is not anti-bot.
        assert result is None

    def test_tier2_case_insensitive(self):
        content = "<html><body>ACCESS DENIED</body></html>"
        result = detect_antibot(content)
        assert result is not None
        assert result.tier == 2


# ── Tier 3: Structural integrity ─────────────────────────────


class TestTier3Structural:
    """Structural integrity checks for suspiciously empty pages."""

    def test_empty_shell(self):
        content, _, _ = load_antibot_page("empty_shell")
        result = detect_antibot(content)
        assert result is not None
        assert result.tier == 3
        assert "empty" in result.pattern_matched.lower()

    def test_script_only_shell(self):
        content, _, _ = load_antibot_page("script_only_shell")
        result = detect_antibot(content)
        assert result is not None
        assert result.tier == 3

    def test_very_short_body(self):
        content = "<html><body></body></html>"
        result = detect_antibot(content)
        assert result is not None
        assert result.tier == 3

    def test_effectively_empty(self):
        """< 100 bytes is always suspicious."""
        content = "<html></html>"
        result = detect_antibot(content)
        assert result is not None
        assert result.tier == 3
        assert "empty" in result.pattern_matched.lower()

    def test_shell_under_500_with_script_no_content(self):
        content = (
            "<!DOCTYPE html><html><head><title></title></head>"
            "<body><script>init();</script></body></html>"
        )
        assert len(content) < 500
        result = detect_antibot(content)
        assert result is not None
        assert result.tier == 3

    def test_shell_under_500_with_content_tag_passes(self):
        """Small page WITH content tags should pass (not anti-bot)."""
        content = (
            "<!DOCTYPE html><html><head><title>Hi</title></head>"
            "<body><p>This is real content with enough text.</p></body></html>"
        )
        assert len(content) < 500
        result = detect_antibot(content)
        assert result is None

    def test_small_page_under_100_with_content_tags_passes(self):
        """Even a tiny page with real content tags is not anti-bot."""
        content = "<html><body><p>Hi</p></body></html>"
        assert len(content) < 100
        result = detect_antibot(content)
        assert result is None

    def test_script_only_with_many_visible_chars_passes(self):
        """Script-only page with significant visible text (> 500 bytes) is not anti-bot."""
        # A page with scripts and 100+ visible chars. Must be > 500 bytes
        # to avoid Tier 3 Check 2 (empty shell < 500 bytes without content tags).
        text = "This is a very long text that contains enough characters. " * 10
        content = (
            f"<!DOCTYPE html><html><body><script>init();</script><div>{text}</div></body></html>"
        )
        assert len(content) >= 500
        result = detect_antibot(content)
        assert result is None


# ── Negative controls ────────────────────────────────────────


class TestNegativeControls:
    """Clean/legitimate pages must NOT trigger anti-bot detection."""

    def test_clean_real_page(self):
        content, _, _ = load_antibot_page("clean_real_page")
        result = detect_antibot(content)
        assert result is None

    def test_legitimate_large_page(self):
        content, _, _ = load_antibot_page("legitimate_large_page")
        result = detect_antibot(content)
        assert result is None

    def test_normal_page_no_headers(self):
        content = (
            "<html><body><h1>Products</h1>"
            "<p>Welcome to our store. We have many products available for purchase.</p>"
            "<p>Browse our catalog and find what you need today.</p>"
            "</body></html>"
        )
        result = detect_antibot(content)
        assert result is None

    def test_none_content(self):
        """None content (page crashed) should return None, not anti-bot."""
        result = detect_antibot(None)
        assert result is None

    def test_empty_string_content(self):
        """Empty string is < 100 bytes → Tier 3 effectively empty."""
        result = detect_antibot("")
        assert result is not None
        assert result.tier == 3

    def test_normal_page_with_unrelated_headers(self):
        result = detect_antibot(
            "<html><body><h1>Hello World</h1><p>This is a normal page with some content that is long enough to pass structural checks.</p></body></html>",
            headers={"content-type": "text/html", "server": "nginx"},
        )
        assert result is None

    def test_normal_page_with_unrelated_cookies(self):
        result = detect_antibot(
            "<html><body><h1>Hello World</h1><p>This is a normal page with some content that is long enough to pass structural checks.</p></body></html>",
            cookies=[
                {"name": "session_id", "value": "abc"},
                {"name": "theme", "value": "dark"},
            ],
        )
        assert result is None


# ── Edge cases ───────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_headers_dict(self):
        result = detect_antibot(
            "<html><body><h1>Hello World</h1><p>This is a normal page with enough content to pass structural integrity checks.</p></body></html>",
            headers={},
            cookies=[],
        )
        assert result is None

    def test_none_headers_and_cookies(self):
        result = detect_antibot(
            "<html><body><h1>Hello World</h1><p>This is a normal page with enough content to pass structural integrity checks.</p></body></html>",
            headers=None,
            cookies=None,
        )
        assert result is None

    def test_cookie_without_name_key(self):
        """Malformed cookie (no 'name' key) should not crash."""
        result = detect_antibot(
            "<html><body><h1>Hello World</h1><p>This is a normal page with enough content to pass structural integrity checks.</p></body></html>",
            cookies=[{"value": "abc"}],
        )
        assert result is None

    def test_tier1_beats_tier2(self):
        """When both Tier 1 and Tier 2 patterns match, Tier 1 wins."""
        content = (
            "<html><body>"
            "<script>var _Incapsula_Resource = {};</script>"
            "<h1>Request unsuccessful</h1>"
            "</body></html>"
        )
        result = detect_antibot(content)
        assert result is not None
        assert result.tier == 1
        assert result.provider == "imperva"

    def test_multiple_providers_first_wins(self):
        """When multiple Tier 1 providers match, first in priority wins."""
        content = (
            "<html><head><title>Just a moment...</title></head>"
            "<body>"
            '<script>window._pxAppId = "PX123";</script>'
            "</body></html>"
        )
        result = detect_antibot(content)
        assert result is not None
        # Cloudflare is checked before PerimeterX.
        assert result.provider == "cloudflare"

    def test_boundary_10kb(self):
        """Page exactly at 10KB boundary should still trigger Tier 2."""
        # 10239 bytes = just under 10KB
        padding = "x" * (10239 - len("<html><body>Access Denied</body></html>"))
        content = f"<html><body>Access Denied{padding}</body></html>"
        assert len(content) < 10_240
        result = detect_antibot(content)
        assert result is not None
        assert result.tier == 2

    def test_just_over_10kb_no_tier2(self):
        """Page just over 10KB should NOT trigger Tier 2."""
        base = "<html><body><p>Normal content</p> Access Denied "
        padding = "x" * (10_240 - len(base) - len("</body></html>") + 1)
        content = f"{base}{padding}</body></html>"
        assert len(content) >= 10_240
        result = detect_antibot(content)
        # Has a <p> tag, so not Tier 3 either.
        assert result is None
