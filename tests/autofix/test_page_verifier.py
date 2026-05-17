"""Tests for the page verifier.

Tests cover:
  - ``verify_page()``: all 5 PageVerificationResult outcomes
  - ``check_page_gate()``: universal gate rules for all 3 modes
  - Domain matching edge cases
  - Integration with anti-bot detector
"""

from __future__ import annotations

import pytest

from scout.autofix.page_verifier import verify_page, check_page_gate
from scout.autofix.types import (
    RegenerateMode,
    PageSignals,
    PageVerificationResult,
)

# Realistic page content that won't trigger anti-bot Tier 3 structural checks.
_REAL_CONTENT = (
    "<html><body><h1>Product Catalog</h1>"
    "<p>Welcome to our online store. Browse our wide selection of products.</p>"
    "<p>Free shipping on orders over $50. Contact support for help.</p>"
    "</body></html>"
)


# ── verify_page: NO_RESPONSE ────────────────────────────────


class TestVerifyPageNoResponse:
    """NO_RESPONSE when no signals are available."""

    def test_none_signals(self):
        result = verify_page(None, "https://example.com")
        assert result == PageVerificationResult.NO_RESPONSE

    def test_no_status_no_url(self):
        signals = PageSignals()
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.NO_RESPONSE

    def test_only_content_no_status_no_url(self):
        """Content alone without status or URL → NO_RESPONSE."""
        signals = PageSignals(content="<html></html>")
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.NO_RESPONSE


# ── verify_page: SERVER_ERROR ────────────────────────────────


class TestVerifyPageServerError:
    """SERVER_ERROR for HTTP 5xx and 429."""

    def test_500(self):
        signals = PageSignals(
            http_status=500,
            page_url="https://example.com",
        )
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.SERVER_ERROR

    def test_502(self):
        signals = PageSignals(http_status=502, page_url="https://example.com")
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.SERVER_ERROR

    def test_503(self):
        signals = PageSignals(http_status=503, page_url="https://example.com")
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.SERVER_ERROR

    def test_504(self):
        signals = PageSignals(http_status=504, page_url="https://example.com")
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.SERVER_ERROR

    def test_429_rate_limited(self):
        signals = PageSignals(http_status=429, page_url="https://example.com")
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.SERVER_ERROR

    def test_599(self):
        signals = PageSignals(http_status=599, page_url="https://example.com")
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.SERVER_ERROR

    def test_400_is_not_server_error(self):
        """Client errors (4xx except 429) are not SERVER_ERROR."""
        signals = PageSignals(
            http_status=400,
            page_url="https://example.com",
            content=_REAL_CONTENT,
        )
        result = verify_page(signals, "https://example.com")
        assert result != PageVerificationResult.SERVER_ERROR

    def test_403_is_not_server_error(self):
        signals = PageSignals(
            http_status=403,
            page_url="https://example.com",
            content=_REAL_CONTENT,
        )
        result = verify_page(signals, "https://example.com")
        assert result != PageVerificationResult.SERVER_ERROR

    def test_404_is_not_server_error(self):
        signals = PageSignals(
            http_status=404,
            page_url="https://example.com",
            content=_REAL_CONTENT,
        )
        result = verify_page(signals, "https://example.com")
        assert result != PageVerificationResult.SERVER_ERROR


# ── verify_page: REDIRECTED ─────────────────────────────────


class TestVerifyPageRedirected:
    """REDIRECTED when page.url domain differs from target URL domain."""

    def test_different_domain(self):
        signals = PageSignals(
            http_status=200,
            page_url="https://login.othersite.com/sso",
            content=_REAL_CONTENT,
        )
        result = verify_page(signals, "https://example.com/products")
        assert result == PageVerificationResult.REDIRECTED

    def test_subdomain_redirect(self):
        """Different subdomain (not www) is a redirect."""
        signals = PageSignals(
            http_status=200,
            page_url="https://auth.example.com/login",
            content=_REAL_CONTENT,
        )
        result = verify_page(signals, "https://example.com/products")
        assert result == PageVerificationResult.REDIRECTED

    def test_same_domain_different_path(self):
        """Same domain, different path is NOT a redirect."""
        signals = PageSignals(
            http_status=200,
            page_url="https://example.com/different-page",
            content=_REAL_CONTENT,
        )
        result = verify_page(signals, "https://example.com/products")
        assert result == PageVerificationResult.REAL_PAGE

    def test_www_normalization(self):
        """www.example.com and example.com should match."""
        signals = PageSignals(
            http_status=200,
            page_url="https://www.example.com/products",
            content=_REAL_CONTENT,
        )
        result = verify_page(signals, "https://example.com/products")
        assert result == PageVerificationResult.REAL_PAGE

    def test_www_normalization_reverse(self):
        """example.com → www.example.com should match."""
        signals = PageSignals(
            http_status=200,
            page_url="https://example.com/products",
            content=_REAL_CONTENT,
        )
        result = verify_page(signals, "https://www.example.com/products")
        assert result == PageVerificationResult.REAL_PAGE

    def test_data_url_is_redirected(self):
        signals = PageSignals(
            http_status=200,
            page_url="data:text/html,<h1>test</h1>",
        )
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.REDIRECTED

    def test_about_blank_is_redirected(self):
        signals = PageSignals(
            http_status=200,
            page_url="about:blank",
        )
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.REDIRECTED

    def test_blob_url_is_redirected(self):
        signals = PageSignals(
            http_status=200,
            page_url="blob:https://example.com/abc-123",
        )
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.REDIRECTED

    def test_different_port_is_redirected(self):
        signals = PageSignals(
            http_status=200,
            page_url="https://example.com:8443/products",
            content=_REAL_CONTENT,
        )
        result = verify_page(signals, "https://example.com/products")
        assert result == PageVerificationResult.REDIRECTED

    def test_http_to_https_same_domain(self):
        """HTTP→HTTPS on same domain is NOT a redirect (just protocol upgrade)."""
        signals = PageSignals(
            http_status=200,
            page_url="https://example.com/products",
            content=_REAL_CONTENT,
        )
        result = verify_page(signals, "http://example.com/products")
        assert result == PageVerificationResult.REAL_PAGE

    def test_case_insensitive_domain(self):
        signals = PageSignals(
            http_status=200,
            page_url="https://EXAMPLE.COM/products",
            content=_REAL_CONTENT,
        )
        result = verify_page(signals, "https://example.com/products")
        assert result == PageVerificationResult.REAL_PAGE


# ── verify_page: ANTI_BOT ───────────────────────────────────


class TestVerifyPageAntiBot:
    """ANTI_BOT when anti-bot patterns are detected."""

    def test_cloudflare_challenge_content(self):
        signals = PageSignals(
            http_status=200,
            page_url="https://example.com",
            content=(
                '<html><head><title>Just a moment...</title></head>'
                '<body><form id="challenge-form">'
                '<input name="__cf_chl_f_tk" value="abc">'
                '</form></body></html>'
            ),
        )
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.ANTI_BOT

    def test_antibot_header_only(self):
        signals = PageSignals(
            http_status=200,
            page_url="https://example.com",
            content=_REAL_CONTENT,
            headers={"x-amzn-waf-action": "captcha"},
        )
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.ANTI_BOT

    def test_antibot_empty_shell(self):
        """Empty shell page detected as ANTI_BOT via Tier 3."""
        signals = PageSignals(
            http_status=200,
            page_url="https://example.com",
            content="<html><body></body></html>",
        )
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.ANTI_BOT


# ── verify_page: REAL_PAGE ───────────────────────────────────


class TestVerifyPageRealPage:
    """REAL_PAGE when all checks pass."""

    def test_normal_page(self):
        signals = PageSignals(
            http_status=200,
            page_url="https://example.com/products",
            content=_REAL_CONTENT,
        )
        result = verify_page(signals, "https://example.com/products")
        assert result == PageVerificationResult.REAL_PAGE

    def test_with_query_params(self):
        signals = PageSignals(
            http_status=200,
            page_url="https://example.com/products?page=2&sort=name",
            content=_REAL_CONTENT,
        )
        result = verify_page(signals, "https://example.com/products")
        assert result == PageVerificationResult.REAL_PAGE

    def test_301_not_server_error(self):
        """HTTP 301 with same domain after redirect is REAL_PAGE."""
        signals = PageSignals(
            http_status=200,  # Final status after redirect
            page_url="https://example.com/new-products",
            content=_REAL_CONTENT,
        )
        result = verify_page(signals, "https://example.com/products")
        assert result == PageVerificationResult.REAL_PAGE


# ── verify_page: Priority order ─────────────────────────────


class TestVerifyPagePriority:
    """Priority: NO_RESPONSE > SERVER_ERROR > REDIRECTED > ANTI_BOT > REAL_PAGE."""

    def test_server_error_beats_antibot(self):
        """HTTP 503 with anti-bot content → SERVER_ERROR (checked first)."""
        signals = PageSignals(
            http_status=503,
            page_url="https://example.com",
            content='<html><head><title>Just a moment...</title></head></html>',
        )
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.SERVER_ERROR

    def test_server_error_beats_redirect(self):
        signals = PageSignals(
            http_status=500,
            page_url="https://other.com",
        )
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.SERVER_ERROR

    def test_redirect_beats_antibot(self):
        """Different domain with anti-bot content → REDIRECTED (checked first)."""
        signals = PageSignals(
            http_status=200,
            page_url="https://cloudflare.com/challenge",
            content='<html><head><title>Just a moment...</title></head></html>',
        )
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.REDIRECTED


# ── check_page_gate: Cautious/Balanced ───────────────────────


class TestPageGateCautiousBalanced:
    """Cautious and Balanced require 3/3 REAL_PAGE."""

    @pytest.mark.parametrize("mode", [RegenerateMode.CAUTIOUS, RegenerateMode.BALANCED])
    def test_all_real_passes(self, mode):
        results = [PageVerificationResult.REAL_PAGE] * 3
        passes, reason = check_page_gate(results, mode)
        assert passes is True

    @pytest.mark.parametrize("mode", [RegenerateMode.CAUTIOUS, RegenerateMode.BALANCED])
    def test_one_server_error_blocks(self, mode):
        results = [
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.SERVER_ERROR,
        ]
        passes, reason = check_page_gate(results, mode)
        assert passes is False

    @pytest.mark.parametrize("mode", [RegenerateMode.CAUTIOUS, RegenerateMode.BALANCED])
    def test_one_antibot_blocks(self, mode):
        results = [
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.ANTI_BOT,
        ]
        passes, reason = check_page_gate(results, mode)
        assert passes is False
        assert "Anti-bot" in reason

    @pytest.mark.parametrize("mode", [RegenerateMode.CAUTIOUS, RegenerateMode.BALANCED])
    def test_one_redirect_blocks(self, mode):
        results = [
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.REDIRECTED,
        ]
        passes, reason = check_page_gate(results, mode)
        assert passes is False

    @pytest.mark.parametrize("mode", [RegenerateMode.CAUTIOUS, RegenerateMode.BALANCED])
    def test_one_no_response_blocks(self, mode):
        results = [
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.NO_RESPONSE,
        ]
        passes, reason = check_page_gate(results, mode)
        assert passes is False

    @pytest.mark.parametrize("mode", [RegenerateMode.CAUTIOUS, RegenerateMode.BALANCED])
    def test_all_server_error_blocks(self, mode):
        results = [PageVerificationResult.SERVER_ERROR] * 3
        passes, reason = check_page_gate(results, mode)
        assert passes is False

    @pytest.mark.parametrize("mode", [RegenerateMode.CAUTIOUS, RegenerateMode.BALANCED])
    def test_empty_results_blocks(self, mode):
        passes, reason = check_page_gate([], mode)
        assert passes is False


# ── check_page_gate: Eager ───────────────────────────────────


class TestPageGateEager:
    """Eager: 2/3 REAL_PAGE, but ANTI_BOT blocks all."""

    def test_all_real_passes(self):
        results = [PageVerificationResult.REAL_PAGE] * 3
        passes, reason = check_page_gate(results, RegenerateMode.EAGER)
        assert passes is True

    def test_two_real_one_server_error_passes(self):
        """2/3 REAL_PAGE with 1 SERVER_ERROR → passes in eager."""
        results = [
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.SERVER_ERROR,
        ]
        passes, reason = check_page_gate(results, RegenerateMode.EAGER)
        assert passes is True

    def test_two_real_one_redirect_passes(self):
        results = [
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.REDIRECTED,
        ]
        passes, reason = check_page_gate(results, RegenerateMode.EAGER)
        assert passes is True

    def test_two_real_one_no_response_passes(self):
        results = [
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.NO_RESPONSE,
        ]
        passes, reason = check_page_gate(results, RegenerateMode.EAGER)
        assert passes is True

    def test_one_real_two_errors_blocks(self):
        """Only 1/3 REAL_PAGE → blocks even in eager."""
        results = [
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.SERVER_ERROR,
            PageVerificationResult.SERVER_ERROR,
        ]
        passes, reason = check_page_gate(results, RegenerateMode.EAGER)
        assert passes is False

    def test_antibot_blocks_even_with_two_real(self):
        """§6: ANTI_BOT blocks regeneration even in eager mode."""
        results = [
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.ANTI_BOT,
        ]
        passes, reason = check_page_gate(results, RegenerateMode.EAGER)
        assert passes is False
        assert "Anti-bot" in reason

    def test_all_antibot_blocks(self):
        results = [PageVerificationResult.ANTI_BOT] * 3
        passes, reason = check_page_gate(results, RegenerateMode.EAGER)
        assert passes is False

    def test_antibot_plus_server_error_blocks(self):
        results = [
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.ANTI_BOT,
            PageVerificationResult.SERVER_ERROR,
        ]
        passes, reason = check_page_gate(results, RegenerateMode.EAGER)
        assert passes is False

    def test_empty_results_blocks(self):
        passes, reason = check_page_gate([], RegenerateMode.EAGER)
        assert passes is False

    def test_zero_real_blocks(self):
        results = [
            PageVerificationResult.SERVER_ERROR,
            PageVerificationResult.REDIRECTED,
            PageVerificationResult.NO_RESPONSE,
        ]
        passes, reason = check_page_gate(results, RegenerateMode.EAGER)
        assert passes is False


# ── check_page_gate: Reason strings ─────────────────────────


class TestPageGateReasons:
    """Gate reason strings are informative."""

    def test_passes_reason_mentions_real(self):
        results = [PageVerificationResult.REAL_PAGE] * 3
        passes, reason = check_page_gate(results, RegenerateMode.BALANCED)
        assert "real" in reason.lower() or "verified" in reason.lower()

    def test_antibot_reason_mentions_antibot(self):
        results = [
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.ANTI_BOT,
        ]
        _, reason = check_page_gate(results, RegenerateMode.BALANCED)
        assert "Anti-bot" in reason

    def test_failure_reason_mentions_mode(self):
        results = [
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.SERVER_ERROR,
        ]
        _, reason = check_page_gate(results, RegenerateMode.CAUTIOUS)
        assert "cautious" in reason.lower()

    def test_eager_pass_reason_mentions_eager(self):
        results = [
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.SERVER_ERROR,
        ]
        passes, reason = check_page_gate(results, RegenerateMode.EAGER)
        assert passes is True
        assert "eager" in reason.lower()


# ── Domain matching edge cases ───────────────────────────────


class TestDomainMatching:
    """Edge cases for domain comparison in verify_page."""

    def test_trailing_slash_ignored(self):
        signals = PageSignals(
            http_status=200,
            page_url="https://example.com/",
            content=_REAL_CONTENT,
        )
        result = verify_page(signals, "https://example.com")
        assert result == PageVerificationResult.REAL_PAGE

    def test_fragment_ignored(self):
        signals = PageSignals(
            http_status=200,
            page_url="https://example.com/page#section",
            content=_REAL_CONTENT,
        )
        result = verify_page(signals, "https://example.com/page")
        assert result == PageVerificationResult.REAL_PAGE

    def test_ip_address_same(self):
        signals = PageSignals(
            http_status=200,
            page_url="http://192.168.1.1/page",
            content=_REAL_CONTENT,
        )
        result = verify_page(signals, "http://192.168.1.1/page")
        assert result == PageVerificationResult.REAL_PAGE

    def test_ip_address_different(self):
        signals = PageSignals(
            http_status=200,
            page_url="http://192.168.1.2/page",
            content=_REAL_CONTENT,
        )
        result = verify_page(signals, "http://192.168.1.1/page")
        assert result == PageVerificationResult.REDIRECTED

    def test_localhost_match(self):
        signals = PageSignals(
            http_status=200,
            page_url="http://localhost:8080/page",
            content=_REAL_CONTENT,
        )
        result = verify_page(signals, "http://localhost:8080/page")
        assert result == PageVerificationResult.REAL_PAGE

    def test_empty_target_url(self):
        """Empty target URL should not crash."""
        signals = PageSignals(
            http_status=200,
            page_url="https://example.com",
            content=_REAL_CONTENT,
        )
        result = verify_page(signals, "")
        # Empty target means no domain check → falls through to anti-bot/REAL_PAGE.
        assert result in (
            PageVerificationResult.REAL_PAGE,
            PageVerificationResult.REDIRECTED,
        )
