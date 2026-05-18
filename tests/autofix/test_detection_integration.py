"""Integration tests for the full detection pipeline.

Runs every anti-bot fixture through the complete pipeline:
  detect_antibot() → verify_page() → check_page_gate()

Tests verify that:
  1. Every anti-bot fixture is correctly detected (no false negatives)
  2. Negative control fixtures are NOT detected (no false positives)
  3. verify_page() produces correct PageVerificationResult for each scenario
  4. check_page_gate() makes correct decisions for all mode × result combos
  5. Pattern variations and edge cases are handled correctly
  6. Priority ordering between tiers and between verify_page decision order
  7. Real-world formatting variations (whitespace, case, quoting)
"""

from __future__ import annotations

import json

import pytest

from scout.autofix.antibot import detect_antibot
from scout.autofix.page_verifier import check_page_gate, verify_page
from scout.autofix.types import (
    PageSignals,
    PageVerificationResult,
    RegenerateMode,
)
from tests.autofix.conftest import (
    ANTIBOT_DIR,
    load_antibot_page,
)

# ── Helper ───────────────────────────────────────────────────

# Realistic page content that passes all structural checks.
_REAL_CONTENT = (
    "<html><body><h1>Product Catalog</h1>"
    "<p>Welcome to our online store. Browse our wide selection of products "
    "available for purchase today. We ship worldwide.</p>"
    "<p>Contact our support team for any questions about your order.</p>"
    "</body></html>"
)


def _make_signals(
    http_status: int = 200,
    page_url: str = "https://example.com/products",
    content: str | None = _REAL_CONTENT,
    headers: dict[str, str] | None = None,
    cookies: list[dict[str, str]] | None = None,
) -> PageSignals:
    """Build a PageSignals with sensible defaults."""
    return PageSignals(
        http_status=http_status,
        page_url=page_url,
        content=content if content is not None else None,
        headers=headers or {},
        cookies=cookies or [],
    )


# ── Fixture-driven: every anti-bot fixture through detect_antibot ─


# Expected results: fixture name → (detected?, provider, tier)
_FIXTURE_EXPECTATIONS: dict[str, tuple[bool, str | None, int | None]] = {
    # Tier 1: Provider-specific — all should be detected
    "cloudflare_challenge": (True, "cloudflare", 1),
    "cloudflare_error_1020": (True, "cloudflare", 1),
    "akamai_block": (True, "akamai", 1),
    "perimeterx_challenge": (True, "perimeterx", 1),
    "datadome_captcha": (True, "datadome", 1),
    "imperva_block": (True, "imperva", 1),
    "kasada_challenge": (True, "kasada", 1),
    # Tier 2: Generic markers
    "generic_access_denied": (True, None, 2),
    "generic_recaptcha": (True, None, 2),
    "generic_hcaptcha": (True, None, 2),
    # Tier 3: Structural
    "empty_shell": (True, None, 3),
    "script_only_shell": (True, None, 3),
    # Negative controls — should NOT be detected
    "clean_real_page": (False, None, None),
    "legitimate_large_page": (False, None, None),
}


class TestFixtureDrivenDetection:
    """Run every anti-bot fixture through detect_antibot with full signals."""

    @pytest.mark.parametrize(
        "fixture_name",
        sorted(_FIXTURE_EXPECTATIONS.keys()),
    )
    def test_fixture_content_only(self, fixture_name: str):
        """Each fixture detected by content alone (no headers/cookies)."""
        expected_detected, expected_provider, expected_tier = _FIXTURE_EXPECTATIONS[fixture_name]

        # AWS WAF has no HTML file — skip content-only test for it.
        html_path = ANTIBOT_DIR / f"{fixture_name}.html"
        if not html_path.exists():
            pytest.skip(f"No HTML fixture for {fixture_name}")

        content, _, _ = load_antibot_page(fixture_name)
        result = detect_antibot(content)

        if expected_detected:
            assert result is not None, f"Expected {fixture_name} to be detected, but got None"
            if expected_provider is not None:
                assert result.provider == expected_provider, (
                    f"{fixture_name}: expected provider={expected_provider}, got {result.provider}"
                )
            if expected_tier is not None:
                assert result.tier == expected_tier, (
                    f"{fixture_name}: expected tier={expected_tier}, got {result.tier}"
                )
        else:
            assert result is None, f"Expected {fixture_name} to be clean, but got {result}"

    @pytest.mark.parametrize(
        "fixture_name",
        [name for name, (detected, _, _) in _FIXTURE_EXPECTATIONS.items() if detected],
    )
    def test_fixture_with_headers_and_cookies(self, fixture_name: str):
        """Anti-bot fixtures with companion headers/cookies are still detected."""
        html_path = ANTIBOT_DIR / f"{fixture_name}.html"
        if not html_path.exists():
            pytest.skip(f"No HTML fixture for {fixture_name}")

        content, headers, cookies = load_antibot_page(fixture_name)
        result = detect_antibot(content, headers, cookies)
        assert result is not None, f"Expected {fixture_name} to be detected with full signals"
        assert result.tier >= 1


# ── Fixture-driven: every anti-bot fixture through verify_page ─


class TestFixtureDrivenVerifyPage:
    """Run every anti-bot fixture through the full verify_page pipeline."""

    @pytest.mark.parametrize(
        "fixture_name",
        [name for name, (detected, _, _) in _FIXTURE_EXPECTATIONS.items() if detected],
    )
    def test_antibot_fixture_produces_anti_bot_result(self, fixture_name: str):
        """Anti-bot page + HTTP 200 + same domain → ANTI_BOT."""
        html_path = ANTIBOT_DIR / f"{fixture_name}.html"
        if not html_path.exists():
            pytest.skip(f"No HTML fixture for {fixture_name}")

        content, headers, cookies = load_antibot_page(fixture_name)
        signals = _make_signals(
            content=content,
            headers=headers,
            cookies=cookies,
        )
        result = verify_page(signals, "https://example.com/products")
        assert result == PageVerificationResult.ANTI_BOT, (
            f"Expected ANTI_BOT for {fixture_name}, got {result}"
        )

    @pytest.mark.parametrize(
        "fixture_name",
        [name for name, (detected, _, _) in _FIXTURE_EXPECTATIONS.items() if not detected],
    )
    def test_clean_fixture_produces_real_page(self, fixture_name: str):
        """Clean pages + HTTP 200 + same domain → REAL_PAGE."""
        content, headers, cookies = load_antibot_page(fixture_name)
        signals = _make_signals(
            content=content,
            headers=headers,
            cookies=cookies,
        )
        result = verify_page(signals, "https://example.com/products")
        assert result == PageVerificationResult.REAL_PAGE, (
            f"Expected REAL_PAGE for {fixture_name}, got {result}"
        )


# ── Headers-only providers through verify_page ───────────────


class TestHeaderOnlyDetectionPipeline:
    """Providers detectable by headers alone through the full pipeline."""

    def test_aws_waf_through_verify_page(self):
        """AWS WAF header → ANTI_BOT via verify_page."""
        headers_path = ANTIBOT_DIR / "aws_waf.headers.json"
        data = json.loads(headers_path.read_text(encoding="utf-8"))
        signals = _make_signals(
            headers=data.get("headers", {}),
            cookies=data.get("cookies", []),
        )
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.ANTI_BOT

    @pytest.mark.parametrize(
        "header_name,header_value,expected_provider",
        [
            ("x-amzn-waf-action", "captcha", "aws_waf"),
            ("x-amzn-waf-action", "block", "aws_waf"),
            ("x-amzn-waf-action", "challenge", "aws_waf"),
            ("cf-mitigated", "challenge", "cloudflare"),
            ("x-dd-b", "1", "datadome"),
            ("x-px-authorization", "3", "perimeterx"),
            ("x-kasada", "1", "kasada"),
            ("x-cdn", "Incapsula", "imperva"),
            ("x-iinfo", "7-123-456", "imperva"),
        ],
    )
    def test_header_only_signals_through_verify_page(
        self,
        header_name,
        header_value,
        expected_provider,
    ):
        """Individual anti-bot headers should produce ANTI_BOT."""
        signals = _make_signals(headers={header_name: header_value})
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.ANTI_BOT

    @pytest.mark.parametrize(
        "cookie_name,expected_provider",
        [
            ("cf_clearance", "cloudflare"),
            ("_abck", "akamai"),
            ("_px3", "perimeterx"),
            ("_pxhd", "perimeterx"),
            ("datadome", "datadome"),
        ],
    )
    def test_cookie_only_signals_through_verify_page(
        self,
        cookie_name,
        expected_provider,
    ):
        """Individual anti-bot cookies should produce ANTI_BOT."""
        signals = _make_signals(
            cookies=[{"name": cookie_name, "value": "test123"}],
        )
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.ANTI_BOT


# ── verify_page priority ordering integration tests ──────────


class TestVerifyPagePriorityIntegration:
    """Priority: NO_RESPONSE > SERVER_ERROR > REDIRECTED > ANTI_BOT > REAL_PAGE.

    Tests use real fixture content to verify ordering with realistic data.
    """

    def test_503_with_cloudflare_content_is_server_error(self):
        """HTTP 503 + Cloudflare challenge content → SERVER_ERROR (not ANTI_BOT)."""
        content, headers, cookies = load_antibot_page("cloudflare_challenge")
        signals = _make_signals(
            http_status=503,
            content=content,
            headers=headers,
            cookies=cookies,
        )
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.SERVER_ERROR

    def test_429_with_datadome_content_is_server_error(self):
        """HTTP 429 + DataDome content → SERVER_ERROR."""
        content, headers, cookies = load_antibot_page("datadome_captcha")
        signals = _make_signals(
            http_status=429,
            content=content,
            headers=headers,
            cookies=cookies,
        )
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.SERVER_ERROR

    def test_redirect_with_akamai_content_is_redirected(self):
        """Different domain + Akamai content → REDIRECTED (not ANTI_BOT)."""
        content, headers, cookies = load_antibot_page("akamai_block")
        signals = _make_signals(
            page_url="https://waf.provider.com/block",
            content=content,
            headers=headers,
            cookies=cookies,
        )
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.REDIRECTED

    def test_500_with_redirect_is_server_error(self):
        """HTTP 500 + different domain → SERVER_ERROR (checked before REDIRECTED)."""
        signals = _make_signals(
            http_status=500,
            page_url="https://error.cdn.com/500",
        )
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.SERVER_ERROR

    def test_no_status_no_url_with_antibot_content_is_no_response(self):
        """No status + no URL → NO_RESPONSE even with anti-bot content."""
        content, _, _ = load_antibot_page("cloudflare_challenge")
        signals = PageSignals(content=content)
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.NO_RESPONSE


# ── Pattern variation tests ──────────────────────────────────


class TestCloudflareVariations:
    """Cloudflare pattern variations that appear in the wild."""

    def test_cf_error_code_with_single_quotes(self):
        """Some Cloudflare pages use single-quoted class attributes."""
        content = "<html><body><span class='cf-error-code'>1020</span></body></html>"
        # The spec pattern uses double quotes: class="cf-error-code"
        # This tests whether single-quoted attributes are caught.
        result = detect_antibot(content)
        # The regex uses `"` — single quotes won't match Tier 1.
        # But this is a small page, so Tier 2 "Access" won't match.
        # And Tier 3 won't fire because it's under 500 with no scripts.
        # But it HAS no content tags either, so Tier 3 Check 2 catches it.
        # This is acceptable — in practice CF always uses double quotes.
        # The important thing is it doesn't slip through as REAL_PAGE.
        assert result is not None

    def test_challenge_form_with_newlines(self):
        """Challenge form pattern spanning multiple lines."""
        content = (
            "<html><body>\n"
            '<form id="challenge-form">\n'
            '  <input name="__cf_chl_f_tk=" value="abc">\n'
            "</form>\n"
            "</body></html>"
        )
        result = detect_antibot(content)
        assert result is not None
        assert result.provider == "cloudflare"

    def test_orchestrate_various_paths(self):
        """Different orchestrate URL path variations."""
        paths = [
            "/cdn-cgi/challenge-platform/h/g/orchestrate/chl_page/v1",
            "/cdn-cgi/challenge-platform/scripts/jsd/main.js?orchestrate",
            "/cdn-cgi/challenge-platform/h/b/orchestrate/chl_api/v1",
        ]
        for path in paths:
            content = f'<script src="{path}"></script>'
            result = detect_antibot(content)
            assert result is not None, f"Failed to detect orchestrate path: {path}"
            assert result.provider == "cloudflare"

    def test_cf_error_code_various_codes(self):
        """Various Cloudflare error codes (1006, 1015, 1020, 1106, etc.)."""
        for code in [1006, 1015, 1020, 1106, 1012]:
            content = f'<html><body><span class="cf-error-code">{code}</span></body></html>'
            result = detect_antibot(content)
            assert result is not None, f"Failed to detect CF error code {code}"
            assert result.provider == "cloudflare"

    def test_just_a_moment_various_formats(self):
        """Just a moment title with extra text."""
        titles = [
            "Just a moment...",
            "Just a moment",
            "Just a moment please",
            "  Just a moment...",
        ]
        for title in titles:
            content = f"<html><head><title>{title}</title></head><body></body></html>"
            result = detect_antibot(content)
            assert result is not None, f"Failed to detect title: {title}"
            assert result.provider == "cloudflare"

    def test_cf_mitigated_managed_value(self):
        """cf-mitigated: managed is NOT a challenge — should not match."""
        # The spec says the header value should be "challenge".
        # "managed" means Cloudflare is present but didn't challenge.
        result = detect_antibot(
            _REAL_CONTENT,
            headers={"cf-mitigated": "managed"},
        )
        # "managed" doesn't match the regex `challenge` → should not detect.
        assert result is None


class TestAkamaiVariations:
    """Akamai pattern variations."""

    def test_reference_various_formats(self):
        """Different Akamai reference ID formats."""
        refs = [
            "Reference # 18.abc123.1234567890.abcdef",
            "Reference #18.deadbeef.9876543210.cafebabe",
            "Reference # 7.a1b2c3.12345.ff00ff",
        ]
        for ref in refs:
            content = f"<html><body><p>{ref}</p></body></html>"
            result = detect_antibot(content)
            assert result is not None, f"Failed to detect Akamai ref: {ref}"
            assert result.provider == "akamai"

    def test_pardon_with_different_whitespace(self):
        """Pardon Our Interruption with varied whitespace."""
        variants = [
            "Pardon Our Interruption",
            "Pardon  Our  Interruption",
            "Pardon\nOur\nInterruption",
        ]
        for v in variants:
            content = f"<html><body><h1>{v}</h1></body></html>"
            result = detect_antibot(content)
            assert result is not None, f"Failed: '{v}'"
            assert result.provider == "akamai"


class TestPerimeterXVariations:
    """PerimeterX pattern variations."""

    def test_px_app_id_no_spaces(self):
        content = "<script>window._pxAppId='PX123';</script>"
        result = detect_antibot(content)
        assert result is not None
        assert result.provider == "perimeterx"

    def test_px_app_id_with_spaces(self):
        content = "<script>window._pxAppId = 'PX123';</script>"
        result = detect_antibot(content)
        assert result is not None
        assert result.provider == "perimeterx"

    def test_px_app_id_double_quotes(self):
        content = '<script>window._pxAppId="PXabcdef";</script>'
        result = detect_antibot(content)
        assert result is not None
        assert result.provider == "perimeterx"


class TestKasadaVariations:
    """Kasada pattern variations."""

    def test_kpsdk_with_spaces(self):
        content = "<script>KPSDK.scriptStart = KPSDK.now();</script>"
        result = detect_antibot(content)
        assert result is not None
        assert result.provider == "kasada"

    def test_kpsdk_no_spaces(self):
        content = "<script>KPSDK.scriptStart=KPSDK.now();</script>"
        result = detect_antibot(content)
        assert result is not None
        assert result.provider == "kasada"

    def test_kpsdk_extra_spaces(self):
        content = "<script>KPSDK.scriptStart  =  KPSDK.now();</script>"
        result = detect_antibot(content)
        assert result is not None
        assert result.provider == "kasada"


class TestImpervaVariations:
    """Imperva pattern variations."""

    def test_incapsula_incident_with_long_id(self):
        content = (
            "<html><body>Incapsula incident ID: 893426000021564786-226669634693259305</body></html>"
        )
        result = detect_antibot(content)
        assert result is not None
        assert result.provider == "imperva"

    def test_x_cdn_incapsula_case_insensitive(self):
        result = detect_antibot(
            _REAL_CONTENT,
            headers={"X-CDN": "incapsula"},
        )
        assert result is not None
        assert result.provider == "imperva"

    def test_x_cdn_not_incapsula_no_match(self):
        """x-cdn with a different CDN name should NOT match."""
        result = detect_antibot(
            _REAL_CONTENT,
            headers={"x-cdn": "Cloudfront"},
        )
        assert result is None


# ── Tier 2 variations ────────────────────────────────────────


class TestTier2Variations:
    """Tier 2 generic marker variations."""

    def test_access_denied_in_h1(self):
        content = "<html><body><h1>Access Denied</h1></body></html>"
        result = detect_antibot(content)
        assert result is not None
        assert result.tier == 2

    def test_access_denied_mixed_case(self):
        content = "<html><body>access denied</body></html>"
        result = detect_antibot(content)
        assert result is not None
        assert result.tier == 2

    def test_recaptcha_double_quotes(self):
        content = '<div class="g-recaptcha" data-sitekey="abc"></div>'
        result = detect_antibot(content)
        assert result is not None
        assert result.tier == 2

    def test_recaptcha_single_quotes(self):
        content = "<div class='g-recaptcha' data-sitekey='abc'></div>"
        result = detect_antibot(content)
        assert result is not None
        assert result.tier == 2

    def test_hcaptcha_double_quotes(self):
        content = '<div class="h-captcha" data-sitekey="abc"></div>'
        result = detect_antibot(content)
        assert result is not None
        assert result.tier == 2

    def test_hcaptcha_single_quotes(self):
        content = "<div class='h-captcha' data-sitekey='abc'></div>"
        result = detect_antibot(content)
        assert result is not None
        assert result.tier == 2

    def test_blocked_by_security_in_paragraph(self):
        content = (
            "<html><body><p>Your request has been blocked by security policy.</p></body></html>"
        )
        result = detect_antibot(content)
        assert result is not None
        assert result.tier == 2

    def test_checking_browser_variations(self):
        variants = [
            "Checking your browser",
            "Checking  your  browser",
            "CHECKING YOUR BROWSER",
            "checking your browser before accessing",
        ]
        for v in variants:
            content = f"<html><body>{v}</body></html>"
            result = detect_antibot(content)
            assert result is not None, f"Failed: '{v}'"
            assert result.tier == 2


# ── Tier 3 variations ────────────────────────────────────────


class TestTier3Variations:
    """Tier 3 structural integrity variations."""

    def test_empty_html(self):
        result = detect_antibot("<html></html>")
        assert result is not None
        assert result.tier == 3

    def test_empty_body(self):
        result = detect_antibot("<html><body></body></html>")
        assert result is not None
        assert result.tier == 3

    def test_div_only_spa_shell(self):
        content = '<html><body><div id="root"></div></body></html>'
        assert len(content) < 500
        result = detect_antibot(content)
        assert result is not None
        assert result.tier == 3

    def test_div_with_script_spa_shell(self):
        content = '<html><body><div id="app"></div><script src="/bundle.js"></script></body></html>'
        result = detect_antibot(content)
        assert result is not None
        assert result.tier == 3

    def test_page_with_only_noscript(self):
        """Page with <noscript> but no content tags."""
        content = '<html><body><noscript>Enable JavaScript</noscript><script src="/app.js"></script></body></html>'
        result = detect_antibot(content)
        assert result is not None
        assert result.tier == 3

    def test_page_with_p_tag_passes(self):
        """Even a small page with <p> passes Tier 3."""
        content = "<html><body><p>Hello world, this is real content.</p></body></html>"
        result = detect_antibot(content)
        assert result is None

    def test_page_with_h1_tag_passes(self):
        content = (
            "<html><body><h1>Welcome to our website with real content here.</h1></body></html>"
        )
        result = detect_antibot(content)
        assert result is None

    def test_page_with_article_tag_passes(self):
        content = "<html><body><article>This is a real article with actual content.</article></body></html>"
        result = detect_antibot(content)
        assert result is None

    def test_page_with_h2_through_h6_passes(self):
        for i in range(2, 7):
            content = f"<html><body><h{i}>Section title with content here.</h{i}></body></html>"
            result = detect_antibot(content)
            assert result is None, f"<h{i}> should be a content tag"


# ── Full pipeline: check_page_gate with anti-bot fixture data ─


class TestGateWithFixtureData:
    """Gate decisions using fixture-derived PageVerificationResults."""

    def test_3_real_pages_all_modes_pass(self):
        """3/3 REAL_PAGE → gate passes in all modes."""
        results = [PageVerificationResult.REAL_PAGE] * 3
        for mode in RegenerateMode:
            passes, _ = check_page_gate(results, mode)
            assert passes, f"Should pass in {mode.value}"

    def test_3_antibot_all_modes_block(self):
        """3/3 ANTI_BOT → gate blocks in all modes."""
        results = [PageVerificationResult.ANTI_BOT] * 3
        for mode in RegenerateMode:
            passes, reason = check_page_gate(results, mode)
            assert not passes, f"Should block in {mode.value}"
            assert "Anti-bot" in reason

    def test_2_real_1_antibot_all_modes_block(self):
        """2/3 REAL_PAGE + 1/3 ANTI_BOT → blocks even in aggressive."""
        results = [
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.ANTI_BOT,
        ]
        for mode in RegenerateMode:
            passes, reason = check_page_gate(results, mode)
            assert not passes, f"ANTI_BOT should block in {mode.value}"
            assert "Anti-bot" in reason

    def test_2_real_1_server_error_mode_behavior(self):
        """2/3 REAL_PAGE + 1/3 SERVER_ERROR → mode-dependent."""
        results = [
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.SERVER_ERROR,
        ]
        # Conservative: blocks (needs 3/3)
        passes, _ = check_page_gate(results, RegenerateMode.CAUTIOUS)
        assert not passes

        # Balanced: blocks (needs 3/3)
        passes, _ = check_page_gate(results, RegenerateMode.BALANCED)
        assert not passes

        # Aggressive: passes (2/3 sufficient, no ANTI_BOT)
        passes, _ = check_page_gate(results, RegenerateMode.EAGER)
        assert passes

    def test_mixed_taint_types(self):
        """Each tainted result type blocks conservative/balanced."""
        taint_types = [
            PageVerificationResult.SERVER_ERROR,
            PageVerificationResult.REDIRECTED,
            PageVerificationResult.NO_RESPONSE,
        ]
        for taint in taint_types:
            results = [
                PageVerificationResult.REAL_PAGE,
                PageVerificationResult.REAL_PAGE,
                taint,
            ]
            passes, _ = check_page_gate(results, RegenerateMode.BALANCED)
            assert not passes, f"{taint} should block balanced mode"

    def test_1_real_2_tainted_blocks_all(self):
        """1/3 REAL_PAGE → blocks all modes (even aggressive needs 2/3)."""
        results = [
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.SERVER_ERROR,
            PageVerificationResult.REDIRECTED,
        ]
        for mode in RegenerateMode:
            passes, _ = check_page_gate(results, mode)
            assert not passes, f"Should block in {mode.value}"

    def test_zero_real_blocks_all(self):
        """0/3 REAL_PAGE → blocks all modes."""
        results = [
            PageVerificationResult.SERVER_ERROR,
            PageVerificationResult.NO_RESPONSE,
            PageVerificationResult.REDIRECTED,
        ]
        for mode in RegenerateMode:
            passes, _ = check_page_gate(results, mode)
            assert not passes

    def test_single_attempt(self):
        """Gate with only 1 attempt."""
        passes, _ = check_page_gate(
            [PageVerificationResult.REAL_PAGE],
            RegenerateMode.BALANCED,
        )
        assert passes

        passes, _ = check_page_gate(
            [PageVerificationResult.ANTI_BOT],
            RegenerateMode.EAGER,
        )
        assert not passes

    def test_two_attempts(self):
        """Gate with only 2 attempts."""
        # 2/2 REAL_PAGE → passes all modes.
        passes, _ = check_page_gate(
            [PageVerificationResult.REAL_PAGE] * 2,
            RegenerateMode.BALANCED,
        )
        assert passes

        # 1/2 REAL_PAGE → fails all modes.
        # Aggressive needs max(2, ceil(2/3 * 2)) = 2, so 1/2 is not enough.
        results = [
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.SERVER_ERROR,
        ]
        passes, _ = check_page_gate(results, RegenerateMode.BALANCED)
        assert not passes
        passes, _ = check_page_gate(results, RegenerateMode.EAGER)
        assert not passes


# ── End-to-end: full scenario simulation ─────────────────────


class TestEndToEndScenarios:
    """Simulate real-world scenarios through the full pipeline."""

    def test_scenario_real_page_passes(self):
        """Normal page, HTTP 200, same domain → REAL_PAGE."""
        signals = _make_signals()
        result = verify_page(signals, "https://example.com/products")
        assert result == PageVerificationResult.REAL_PAGE

    def test_scenario_cloudflare_blocks(self):
        """Cloudflare challenge served instead of real page."""
        content, headers, cookies = load_antibot_page("cloudflare_challenge")
        signals = _make_signals(content=content, headers=headers, cookies=cookies)
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.ANTI_BOT

        # 3 attempts all ANTI_BOT → gate blocks all modes
        results = [result] * 3
        for mode in RegenerateMode:
            passes, _ = check_page_gate(results, mode)
            assert not passes

    def test_scenario_intermittent_cloudflare(self):
        """1/3 Cloudflare challenge, 2/3 real → blocks all modes."""
        real_signals = _make_signals()
        real_result = verify_page(real_signals, "https://example.com")

        cf_content, cf_headers, cf_cookies = load_antibot_page("cloudflare_challenge")
        cf_signals = _make_signals(content=cf_content, headers=cf_headers, cookies=cf_cookies)
        cf_result = verify_page(cf_signals, "https://example.com")

        results = [real_result, real_result, cf_result]
        for mode in RegenerateMode:
            passes, reason = check_page_gate(results, mode)
            assert not passes, f"Intermittent ANTI_BOT should block {mode.value}"
            assert "Anti-bot" in reason

    def test_scenario_server_down(self):
        """Server returns 503 → SERVER_ERROR → blocks."""
        signals = _make_signals(http_status=503)
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.SERVER_ERROR

        results = [result] * 3
        for mode in RegenerateMode:
            passes, _ = check_page_gate(results, mode)
            assert not passes

    def test_scenario_intermittent_server_error(self):
        """1/3 server error, 2/3 real → aggressive passes, others block."""
        results = [
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.SERVER_ERROR,
        ]
        passes, _ = check_page_gate(results, RegenerateMode.CAUTIOUS)
        assert not passes
        passes, _ = check_page_gate(results, RegenerateMode.BALANCED)
        assert not passes
        passes, _ = check_page_gate(results, RegenerateMode.EAGER)
        assert passes

    def test_scenario_login_redirect(self):
        """Redirected to login page → REDIRECTED."""
        signals = _make_signals(page_url="https://login.example-sso.com/auth")
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.REDIRECTED

    def test_scenario_page_crash(self):
        """Page crashed, no signals → NO_RESPONSE."""
        result = verify_page(None, "https://example.com")
        assert result == PageVerificationResult.NO_RESPONSE

    def test_scenario_spa_empty_shell(self):
        """SPA that didn't render → ANTI_BOT (Tier 3 structural)."""
        signals = _make_signals(
            content='<html><body><div id="root"></div><script src="/app.js"></script></body></html>',
        )
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.ANTI_BOT

    def test_scenario_real_page_with_captcha_cookie(self):
        """Real page served WITH a datadome cookie → ANTI_BOT.

        Even if the page content looks real, the anti-bot cookie
        signals that the anti-bot system is active.
        """
        signals = _make_signals(
            cookies=[{"name": "datadome", "value": "abc123"}],
        )
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.ANTI_BOT


# ── Robustness: malformed inputs ─────────────────────────────


class TestMalformedInputs:
    """Detector handles malformed/unusual inputs without crashing."""

    def test_binary_content(self):
        """Binary content (e.g., PDF mistakenly loaded) doesn't crash."""
        content = "\x00\x01\x02\x03" * 1000
        result = detect_antibot(content)
        # Should not crash; result doesn't matter as long as no exception.
        assert result is None or isinstance(result, type(result))

    def test_very_large_content(self):
        """1MB page doesn't cause performance issues."""
        content = "<html><body><p>" + "x" * 1_000_000 + "</p></body></html>"
        result = detect_antibot(content)
        assert result is None  # Has <p> tag, large, no anti-bot patterns.

    def test_unicode_content(self):
        """Unicode content (Chinese, Arabic, emoji) doesn't crash."""
        content = (
            "<html><body><h1>产品目录</h1><p>欢迎来到我们的商店。浏览我们的产品。</p></body></html>"
        )
        result = detect_antibot(content)
        assert result is None

    def test_html_entities(self):
        """HTML entities don't trigger false positives."""
        content = "<html><body><h1>Products &amp; Services</h1><p>Welcome to our store with content.</p></body></html>"
        result = detect_antibot(content)
        assert result is None

    def test_deeply_nested_html(self):
        """Deeply nested HTML doesn't cause regex issues."""
        nesting = "<div>" * 100 + "<p>content</p>" + "</div>" * 100
        content = f"<html><body>{nesting}</body></html>"
        result = detect_antibot(content)
        assert result is None

    def test_malformed_html(self):
        """Malformed HTML (unclosed tags) doesn't crash."""
        content = "<html><body><p>Unclosed paragraph<div>Nested wrong<h1>Title"
        result = detect_antibot(content)
        assert result is None  # Has <p> and <h1> tags → real content.

    def test_script_tag_with_antibot_pattern_in_string(self):
        """Anti-bot pattern inside a JS string literal on a real page."""
        # A real page might have JavaScript that mentions "Access Denied"
        # as a string literal. This should NOT trigger detection on large pages.
        js_code = 'var msg = "Access Denied"; console.log(msg);'
        content = (
            "<html><body>"
            "<h1>Real Page</h1>"
            f"<script>{js_code}</script>"
            "<p>This is a real page with some JavaScript. " + "x" * 10200 + "</p>"
            "</body></html>"
        )
        assert len(content) >= 10_240, "Content must be > 10KB for this test"
        result = detect_antibot(content)
        # Page is > 10KB → Tier 2 gate prevents false positive.
        assert result is None

    def test_empty_cookie_list(self):
        result = detect_antibot(_REAL_CONTENT, cookies=[])
        assert result is None

    def test_cookie_with_empty_name(self):
        result = detect_antibot(
            _REAL_CONTENT,
            cookies=[{"name": "", "value": "test"}],
        )
        assert result is None

    def test_headers_with_empty_values(self):
        result = detect_antibot(
            _REAL_CONTENT,
            headers={"x-custom": "", "content-type": ""},
        )
        assert result is None

    def test_verify_page_with_malformed_url(self):
        """Malformed URLs don't crash verify_page."""
        signals = _make_signals(page_url="not-a-valid-url")
        # Should not crash.
        result = verify_page(signals, "https://example.com")
        # page_url has no hostname → domain match fails → REDIRECTED.
        assert result == PageVerificationResult.REDIRECTED

    def test_verify_page_with_empty_page_url(self):
        signals = _make_signals(page_url="")
        # Empty URL has no hostname → treated as None in parsing.
        result = verify_page(signals, "https://example.com")
        assert result in (
            PageVerificationResult.REDIRECTED,
            PageVerificationResult.NO_RESPONSE,
        )
