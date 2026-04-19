"""Battle tests: real-world scenarios for the auto-fix algorithm.

These tests model common real-world situations that scrapers encounter
in production. Each test describes what's happening in the real world,
simulates it with a programmable HTTP server, and verifies the
algorithm's decision.

Architecture:
  - BattleServer serves configurable HTTP response sequences
  - _fetch() makes real HTTP requests and extracts real page signals
  - Script behavior functions simulate what a cached script would do
  - make_execute_fn() combines fetch + behavior into a diagnose() callable
  - Each test programs the server, runs the full diagnosis pipeline,
    and checks the outcome

What's real:
  - HTTP responses (status, headers, body)
  - PageSignals (from actual HTTP responses)
  - Anti-bot detection (runs against real Cloudflare/Akamai HTML)
  - Page verification (real domain checks, real HTTP status checks)
  - Error classification (on realistic error strings)
  - Fingerprinting, stability assessment, decision engine

What's simulated:
  - The script's interaction with the page (behavior functions determine
    what error a script would produce against the fetched content)
"""

from __future__ import annotations

import asyncio
import http.client
from typing import Any
from urllib.parse import urlparse

import pytest
import pytest_asyncio

from scout.autofix.diagnosis import diagnose
from scout.autofix.types import (
    AttemptResult,
    AutoFixAction,
    AutoFixMode,
    DiagnosisResult,
    PageSignals,
    PageVerificationResult,
)
from tests.autofix.battle_server import (
    AB_TEST_VARIANT,
    BattleServer,
    CHANGED_LAYOUT,
    CLOUDFLARE_CHALLENGE,
    DATADOME_CHALLENGE,
    FEW_ITEMS_PAGE,
    LOGIN_WALL,
    OVERLAY_PAGE,
    PARTIAL_PRODUCTS,
    PRODUCTS_PAGE,
    RATE_LIMIT_SOFT,
    SERVER_503,
    SERVER_429,
    SERVER_ERROR_AS_200,
)


# ── Fixtures ─────────────────────────────────────────────────


@pytest_asyncio.fixture()
async def server():
    """Start a programmable HTTP server for the test."""
    async with BattleServer() as srv:
        yield srv


@pytest.fixture(autouse=True)
def _fast_delays(monkeypatch):
    """Patch diagnosis delays to near-zero for fast tests."""
    monkeypatch.setattr("scout.autofix.diagnosis._MIN_DELAY_S", 0.01)
    monkeypatch.setattr("scout.autofix.diagnosis._MAX_DELAY_S", 0.02)


# ── HTTP fetch ───────────────────────────────────────────────


async def _fetch(
    url: str,
) -> tuple[int, str, dict[str, str], list[dict[str, str]]]:
    """Fetch a URL and return (status, body, headers, cookies).

    Makes a real HTTP request to the battle server. Headers and cookies
    are split so PageSignals can be built accurately: cookies come from
    Set-Cookie headers, everything else goes into the headers dict.
    """

    def _do() -> tuple[int, str, dict[str, str], list[dict[str, str]]]:
        parsed = urlparse(url)
        conn = http.client.HTTPConnection(
            parsed.hostname, parsed.port, timeout=10,
        )
        try:
            conn.request("GET", parsed.path or "/")
            resp = conn.getresponse()
            status = resp.status
            body = resp.read().decode(errors="replace")

            headers_dict: dict[str, str] = {}
            cookies: list[dict[str, str]] = []
            for name, value in resp.getheaders():
                if name.lower() == "set-cookie":
                    parts = value.split(";")
                    if parts and "=" in parts[0]:
                        cname, cval = parts[0].strip().split("=", 1)
                        cookies.append({"name": cname, "value": cval})
                else:
                    headers_dict[name.lower()] = value

            return status, body, headers_dict, cookies
        finally:
            conn.close()

    return await asyncio.to_thread(_do)


def _build_signals(
    status: int,
    body: str,
    headers: dict[str, str],
    cookies: list[dict[str, str]],
    url: str,
) -> PageSignals:
    """Build PageSignals from raw HTTP response data."""
    return PageSignals(
        http_status=status,
        page_url=url,
        content=body,
        headers=headers,
        cookies=cookies,
    )


# ── Execute function builder ─────────────────────────────────


def make_execute_fn(
    url: str,
    behavior: Any,
) -> Any:
    """Create an execute_fn that fetches a URL and applies a behavior.

    Each call makes a real HTTP request, builds real PageSignals from
    the response, then passes them to the behavior function which
    determines the script outcome.
    """

    async def execute() -> AttemptResult:
        status, body, headers, cookies = await _fetch(url)
        signals = _build_signals(status, body, headers, cookies, url)
        return behavior(body, signals)

    return execute


# ── Script behaviors ─────────────────────────────────────────
#
# Each function models what a real cached script would do when
# executed against a page. Named after what the script does,
# not after error categories.


def script_waits_for(
    selector: str,
    success_data: Any = None,
) -> Any:
    """A script that does page.wait_for_selector(selector).

    Succeeds if the selector's class name appears in the HTML.
    Times out if not found — like a real Playwright wait.
    """
    class_name = selector.lstrip(".")

    def behavior(
        content: str, signals: PageSignals,
    ) -> AttemptResult:
        if f'class="{class_name}"' in content:
            return AttemptResult(
                success=True,
                data=success_data or [{"name": "Widget", "price": "9.99"}],
            )
        return AttemptResult(
            success=False,
            error=(
                "patchright._impl._errors.TimeoutError: "
                "Page.wait_for_selector: Timeout 5000ms exceeded.\n"
                "Call log:\n"
                f'  - waiting for selector "{selector}"'
            ),
            page_signals=signals,
        )

    return behavior


def script_queries(
    selector: str,
    success_data: Any = None,
) -> Any:
    """A script that does el = page.query_selector(selector); el.text_content().

    Succeeds if found. Crashes with AttributeError on None if not found —
    the classic "NoneType has no attribute" error.
    """
    class_name = selector.lstrip(".")

    def behavior(
        content: str, signals: PageSignals,
    ) -> AttemptResult:
        if f'class="{class_name}"' in content:
            return AttemptResult(
                success=True,
                data=success_data or [{"name": "Widget", "price": "9.99"}],
            )
        return AttemptResult(
            success=False,
            error=(
                "Traceback (most recent call last):\n"
                '  File "script.py", line 5, in scrape\n'
                "    name = await el.text_content()\n"
                "AttributeError: 'NoneType' object has no attribute "
                "'text_content'"
            ),
            page_signals=signals,
        )

    return behavior


def script_clicks_through_overlay() -> Any:
    """A script that tries to click an element blocked by an overlay.

    Always produces a pointer interception error — the overlay covers
    the target element on every attempt (fresh browser = no cookies =
    cookie consent shows every time).
    """

    def behavior(
        content: str, signals: PageSignals,
    ) -> AttemptResult:
        return AttemptResult(
            success=False,
            error=(
                "patchright._impl._errors.TimeoutError: "
                "Page.click: Timeout 5000ms exceeded.\n"
                "Call log:\n"
                "  2 \u00d7 waiting for element to be visible, "
                "enabled and stable\n"
                "    - element is visible, enabled and stable\n"
                "    - scrolling into view if needed\n"
                "    - done scrolling\n"
                '    - <div id="cookie-consent"></div> '
                "intercepts pointer events\n"
                "  - retrying click action"
            ),
            page_signals=signals,
        )

    return behavior


def script_extracts_items(
    data: Any,
    schema_error: str,
) -> Any:
    """A script that extracts data but schema validation rejects it.

    The script runs to completion, finds data, returns it. But the data
    doesn't satisfy the schema constraints (e.g., too few items).
    """

    def behavior(
        content: str, signals: PageSignals,
    ) -> AttemptResult:
        return AttemptResult(
            success=False,
            data=data,
            schema_error=schema_error,
            page_signals=signals,
        )

    return behavior


def script_has_syntax_error() -> Any:
    """A corrupted script file with a SyntaxError.

    The Python interpreter rejects the code before a single line runs.
    Page content is irrelevant — the script never executes.
    """

    def behavior(
        content: str, signals: PageSignals,
    ) -> AttemptResult:
        return AttemptResult(
            success=False,
            error=(
                "Traceback (most recent call last):\n"
                '  File "script.py", line 1\n'
                "    def scrape(page\n"
                "               ^\n"
                "SyntaxError: unexpected EOF while parsing"
            ),
            page_signals=signals,
        )

    return behavior


def script_hits_navigation_redirect() -> Any:
    """A script where the page navigates away during execution.

    JavaScript on the page redirects (auth flow, ad redirect, SPA routing).
    The execution context is destroyed. Happens on every attempt because
    it's a page behavior, not a transient condition.
    """

    def behavior(
        content: str, signals: PageSignals,
    ) -> AttemptResult:
        return AttemptResult(
            success=False,
            error=(
                "patchright._impl._errors.Error: "
                "Execution context was destroyed, most likely "
                "because of a navigation"
            ),
            page_signals=signals,
        )

    return behavior


def script_fails_then_works(
    n_failures: int,
    error_str: str,
    success_data: Any = None,
) -> Any:
    """A script that fails transiently, then succeeds.

    Models real-world situations where the first attempt fails (slow
    server, DNS cache miss, cold CDN) but subsequent attempts work.
    """
    counter = [0]

    def behavior(
        content: str, signals: PageSignals,
    ) -> AttemptResult:
        counter[0] += 1
        if counter[0] <= n_failures:
            return AttemptResult(
                success=False,
                error=error_str,
                page_signals=signals,
            )
        return AttemptResult(
            success=True,
            data=success_data or [{"name": "Widget", "price": "9.99"}],
        )

    return behavior


# ── Battle Tests ─────────────────────────────────────────────


class TestBattle:
    """Real-world scenarios that scrapers encounter in production.

    Each test describes a common situation, models it faithfully,
    and verifies the algorithm's decision. These are not designed
    to break or confirm the algorithm — they reflect what actually
    happens when you run scrapers against real websites.
    """

    # -- 1. The Website Redesign ------------------------------------

    @pytest.mark.asyncio
    async def test_website_redesign(self, server):
        """A company pushes a new frontend. All CSS class names changed.

        This is the #1 reason cached scripts break. The script's
        selectors (.product) no longer exist on the page. Every page
        load returns the redesigned layout with .item instead.

        The algorithm should regenerate — this is its core purpose.
        """
        server.program("/products", CHANGED_LAYOUT)
        url = server.url("/products")

        result = await diagnose(
            make_execute_fn(url, script_waits_for(".product")),
            url,
            AutoFixMode.BALANCED,
        )

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.REGENERATE

    # -- 2. The Server Outage ---------------------------------------

    @pytest.mark.asyncio
    async def test_server_outage(self, server):
        """The server is down. Returns 503 on every request.

        Very common — AWS outages, deployment failures, overloaded
        backends. The script can't find selectors on the error page,
        but this isn't the script's fault.

        The algorithm should NOT regenerate — a new script would hit
        the same broken server.
        """
        server.program("/products", SERVER_503)
        url = server.url("/products")

        result = await diagnose(
            make_execute_fn(url, script_waits_for(".product")),
            url,
            AutoFixMode.BALANCED,
        )

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.RAISE

    # -- 3. The Cloudflare Challenge --------------------------------

    @pytest.mark.asyncio
    async def test_cloudflare_challenge(self, server):
        """Site is behind Cloudflare. Every request gets the challenge page.

        Extremely common — Cloudflare protects millions of sites. The
        challenge page has completely different HTML. The script crashes
        trying to extract product data from "Just a moment..." HTML.

        The algorithm should NOT regenerate — a new script would face
        the same challenge.
        """
        server.program("/products", CLOUDFLARE_CHALLENGE)
        url = server.url("/products")

        result = await diagnose(
            make_execute_fn(url, script_queries(".product")),
            url,
            AutoFixMode.AGGRESSIVE,
        )

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.RAISE
        # Anti-bot should be detected in the page verification
        assert any(
            r == PageVerificationResult.ANTI_BOT
            for r in result.page_results
        )

    # -- 4. The Intermittent Server Error ---------------------------

    @pytest.mark.asyncio
    async def test_intermittent_server_error(self, server):
        """Server is mostly working but returns 503 on one request.

        Very common — rolling deploys, overloaded backends, transient
        health check failures. 2 out of 3 requests return the real page,
        1 returns a server error.

        In balanced mode, the algorithm should NOT regenerate — the
        evidence is incomplete (one attempt didn't see the real page).
        """
        server.program(
            "/products",
            CHANGED_LAYOUT,  # Request 1: real page
            SERVER_503,      # Request 2: server error
            CHANGED_LAYOUT,  # Request 3: real page
        )
        url = server.url("/products")

        result = await diagnose(
            make_execute_fn(url, script_waits_for(".product")),
            url,
            AutoFixMode.BALANCED,
        )

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.RAISE

    # -- 5. The Transient Network Blip ------------------------------

    @pytest.mark.asyncio
    async def test_transient_network_blip(self, server):
        """Network hiccup on first request. Second request works fine.

        Very common — WiFi hiccups, proxy rotation, DNS cache refresh,
        cold CDN edge. The script is fine, the network just had a
        momentary issue.

        The algorithm should return data from the successful retry.
        """
        server.program("/products", PRODUCTS_PAGE)
        url = server.url("/products")

        result = await diagnose(
            make_execute_fn(
                url,
                script_fails_then_works(
                    1,
                    "patchright._impl._errors.TimeoutError: "
                    "Page.wait_for_selector: Timeout 5000ms exceeded.\n"
                    "Call log:\n"
                    '  - waiting for selector ".product"',
                ),
            ),
            url,
            AutoFixMode.BALANCED,
        )

        # Should return successful AttemptResult, not DiagnosisResult
        assert isinstance(result, AttemptResult)
        assert result.success is True

    # -- 6. The Cookie Consent Overlay ------------------------------

    @pytest.mark.asyncio
    async def test_cookie_consent_overlay(self, server):
        """EU site shows full-screen cookie consent on every visit.

        Nearly universal on EU-facing sites since GDPR. The overlay
        covers the entire page. Every fresh browser session sees it
        because there are no saved cookies. The script's click target
        is blocked by the overlay.

        In balanced mode with stable evidence and verified real page,
        the algorithm should regenerate — a new script could handle
        the overlay (e.g., click "Accept" first).
        """
        server.program("/products", OVERLAY_PAGE)
        url = server.url("/products")

        result = await diagnose(
            make_execute_fn(url, script_clicks_through_overlay()),
            url,
            AutoFixMode.BALANCED,
        )

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.REGENERATE

    # -- 7. The Maintenance Page ------------------------------------

    @pytest.mark.asyncio
    async def test_maintenance_page(self, server):
        """Site is in maintenance mode. Serves "We'll be back" with HTTP 200.

        Common during off-peak hours, planned deployments, migrations.
        The page looks real to the algorithm — HTTP 200, same domain,
        no anti-bot markers. But it's a temporary error page, not the
        real site content.

        The algorithm detects this as a non-content page (SOFT_BLOCK)
        via the "something went wrong" pattern. It correctly refuses
        to regenerate — a new script would hit the same maintenance page.
        """
        server.program("/products", SERVER_ERROR_AS_200)
        url = server.url("/products")

        result = await diagnose(
            make_execute_fn(url, script_waits_for(".product")),
            url,
            AutoFixMode.BALANCED,
        )

        assert isinstance(result, DiagnosisResult)
        # Fix 1: Non-content detection catches "something went wrong"
        # as a maintenance page → SOFT_BLOCK → blocks regeneration.
        assert result.action == AutoFixAction.RAISE
        assert any(
            r == PageVerificationResult.SOFT_BLOCK
            for r in result.page_results
        )

    # -- 8. The Adaptive Anti-Bot -----------------------------------

    @pytest.mark.asyncio
    async def test_adaptive_anti_bot(self, server):
        """Anti-bot kicks in after 2 rapid requests.

        Common with sophisticated anti-bot (Cloudflare, DataDome).
        First 2 requests go through normally. Third request triggers
        rate-based challenge. The site adapted to the scraping pattern.

        The algorithm should NOT regenerate — even one anti-bot
        detection should block regeneration.
        """
        server.program(
            "/products",
            CHANGED_LAYOUT,        # Request 1: goes through
            CHANGED_LAYOUT,        # Request 2: goes through
            CLOUDFLARE_CHALLENGE,  # Request 3: challenged
        )
        url = server.url("/products")

        result = await diagnose(
            make_execute_fn(url, script_waits_for(".product")),
            url,
            AutoFixMode.BALANCED,
        )

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.RAISE

    # -- 9. The Schema Drift ----------------------------------------

    @pytest.mark.asyncio
    async def test_schema_drift(self, server):
        """Page used to have 10 products, now shows 2.

        Common — pagination changes, stock depletion, seasonal products.
        The script works perfectly — extracts all 2 items. But the
        schema says min=5. Schema validation fails.

        The algorithm should regenerate — the extraction logic may
        need updating for the new page structure.
        """
        server.program("/products", FEW_ITEMS_PAGE)
        url = server.url("/products")

        result = await diagnose(
            make_execute_fn(
                url,
                script_extracts_items(
                    data=[
                        {"name": "Widget A", "price": "9.99"},
                        {"name": "Widget B", "price": "19.99"},
                    ],
                    schema_error="Expected at least 5 items, got 2",
                ),
            ),
            url,
            AutoFixMode.BALANCED,
        )

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.REGENERATE

    # -- 10. The Corrupted Script File -------------------------------

    @pytest.mark.asyncio
    async def test_corrupted_script(self, server):
        """Script file got corrupted — has a SyntaxError.

        Uncommon but happens — bad disk writes, merge conflicts,
        accidental edits. The Python interpreter rejects the code
        before a single line runs.

        The algorithm should regenerate immediately with no retries —
        the error is deterministic.
        """
        server.program("/products", PRODUCTS_PAGE)
        url = server.url("/products")

        result = await diagnose(
            make_execute_fn(url, script_has_syntax_error()),
            url,
            AutoFixMode.BALANCED,
        )

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.REGENERATE
        # Should NOT have retried — SyntaxError is deterministic
        assert len(result.attempts) == 1

    # -- 11. The JavaScript Redirect --------------------------------

    @pytest.mark.asyncio
    async def test_javascript_redirect(self, server):
        """Page JavaScript navigates away during script execution.

        Common with aggressive ad networks, auth flows, SPA routing.
        "Execution context was destroyed" on every attempt — it's a
        page behavior, not a timing issue. A new script would face
        the same redirect.

        The algorithm should NOT regenerate — even in aggressive mode.
        """
        server.program("/products", PRODUCTS_PAGE)
        url = server.url("/products")

        result = await diagnose(
            make_execute_fn(url, script_hits_navigation_redirect()),
            url,
            AutoFixMode.AGGRESSIVE,
        )

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.RAISE

    # -- 12. The Rolling Deploy -------------------------------------

    @pytest.mark.asyncio
    async def test_rolling_deploy(self, server):
        """Site is mid-deploy. Requests hit different server versions.

        Common during blue-green deploys, canary releases, rolling
        restarts. Each attempt might see a different page version,
        producing different errors — the diagnosis is unstable.

        The algorithm should NOT regenerate — the evidence is
        inconsistent.
        """
        server.program(
            "/products",
            CHANGED_LAYOUT,   # Attempt 1: new layout (timeouts)
            PRODUCTS_PAGE,    # Attempt 2: old layout (succeeds!)
        )
        url = server.url("/products")

        result = await diagnose(
            make_execute_fn(url, script_waits_for(".product")),
            url,
            AutoFixMode.BALANCED,
        )

        # Attempt 1 fails, attempt 2 succeeds (old layout has .product)
        # → algorithm returns data from the successful attempt
        assert isinstance(result, AttemptResult)
        assert result.success is True

    # -- 13. The Network Noise with Real Failure --------------------

    @pytest.mark.asyncio
    async def test_network_noise_with_real_failure(self, server):
        """Script is broken AND the network has a blip.

        Common — network issues and script issues can co-occur.
        The script genuinely has stale selectors (schema failure on
        real pages), but attempt 2 hits a network error. Two real
        signals, one noise signal.

        The algorithm should regenerate — the script is clearly broken
        (2 real-page attempts both show schema failure). The network
        error on attempt 2 is correctly excluded from stability
        assessment (Fix 2: real-page-only stability).
        """
        server.program("/products", FEW_ITEMS_PAGE)
        url = server.url("/products")

        counter = [0]

        async def execute() -> AttemptResult:
            counter[0] += 1

            if counter[0] == 2:
                # Attempt 2: network failure — page never loaded
                return AttemptResult(
                    success=False,
                    error="net::ERR_CONNECTION_RESET",
                    page_signals=None,
                )

            # Attempts 1 and 3: real page, schema failure
            status, body, headers, cookies = await _fetch(url)
            signals = _build_signals(status, body, headers, cookies, url)
            return AttemptResult(
                success=False,
                data=[
                    {"name": "Widget A", "price": "9.99"},
                    {"name": "Widget B", "price": "19.99"},
                ],
                schema_error="Expected at least 5 items, got 2",
                page_signals=signals,
            )

        result = await diagnose(execute, url, AutoFixMode.BALANCED)

        assert isinstance(result, DiagnosisResult)
        # Fix 2: Network noise (attempt 2) is filtered out of stability.
        # Real fingerprints [G, G] → STABLE. Category G + STABLE +
        # 2/3 REAL_PAGE → balanced: REGENERATE.
        assert result.action == AutoFixAction.REGENERATE

    # -- 14. The Conservative User ----------------------------------

    @pytest.mark.asyncio
    async def test_conservative_user_with_clear_evidence(self, server):
        """User explicitly chose conservative mode. Evidence is strong.

        Common for cost-sensitive deployments — batch jobs where one
        false regeneration wastes money. The script has stale selectors
        (stable timeout, all pages verified real). But the error is a
        timeout — an ambiguous category.

        The algorithm should NOT regenerate — the user explicitly chose
        maximum caution for ambiguous errors. They'd rather call
        regenerate=True manually than risk a false positive.
        """
        server.program("/products", CHANGED_LAYOUT)
        url = server.url("/products")

        result = await diagnose(
            make_execute_fn(url, script_waits_for(".product")),
            url,
            AutoFixMode.CONSERVATIVE,
        )

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.RAISE

    # -- 15. The Aggressive User ------------------------------------

    @pytest.mark.asyncio
    async def test_aggressive_user_with_partial_evidence(self, server):
        """User chose aggressive mode. One request got a 503.

        Common for monitoring dashboards, real-time data pipelines.
        The user wants fast recovery even with imperfect evidence.
        The script crashes consistently on real pages. One 503 is
        just server noise.

        The algorithm should regenerate — the user accepted this
        trade-off by choosing aggressive mode.
        """
        server.program(
            "/products",
            SERVER_503,      # Request 1: server error
            CHANGED_LAYOUT,  # Request 2: real page
            CHANGED_LAYOUT,  # Request 3: real page
        )
        url = server.url("/products")

        result = await diagnose(
            make_execute_fn(url, script_queries(".product")),
            url,
            AutoFixMode.AGGRESSIVE,
        )

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.REGENERATE

    # =================================================================
    # EXPANDED SCENARIOS — pure real-world, no algorithm knowledge
    # =================================================================

    # -- 16. The Login Wall -----------------------------------------

    @pytest.mark.asyncio
    async def test_login_wall(self, server):
        """Site requires login. Fresh browser has no session cookie.

        Very common with paywalled sites, dashboards, admin panels.
        The scraper hits a login page every time. HTTP 200, same domain,
        no anti-bot. The login page has completely different HTML than
        the products page.

        The algorithm detects the password input field and correctly
        refuses to regenerate — a new script can't log in either.
        """
        server.program("/products", LOGIN_WALL)
        url = server.url("/products")

        result = await diagnose(
            make_execute_fn(url, script_waits_for(".product")),
            url,
            AutoFixMode.BALANCED,
        )

        assert isinstance(result, DiagnosisResult)
        # Fix 1: Non-content detection catches the password input field
        # on the login page → SOFT_BLOCK → blocks regeneration.
        assert result.action == AutoFixAction.RAISE
        assert any(
            r == PageVerificationResult.SOFT_BLOCK
            for r in result.page_results
        )

    # -- 17. The Rate Limiter That Doesn't Use 429 ------------------

    @pytest.mark.asyncio
    async def test_soft_rate_limit(self, server):
        """Site rate-limits with a friendly "slow down" page at HTTP 200.

        Common with sites that don't follow HTTP standards. Instead of
        429, they return 200 with a "please wait" message. No anti-bot
        markers.
        """
        server.program("/products", RATE_LIMIT_SOFT)
        url = server.url("/products")

        result = await diagnose(
            make_execute_fn(url, script_waits_for(".product")),
            url,
            AutoFixMode.BALANCED,
        )

        assert isinstance(result, DiagnosisResult)
        # Fix 1: Non-content detection catches "browsing too fast" and
        # the auto-refresh meta tag → SOFT_BLOCK → blocks regeneration.
        assert result.action == AutoFixAction.RAISE
        assert any(
            r == PageVerificationResult.SOFT_BLOCK
            for r in result.page_results
        )

    # -- 18. The A/B Test (old version served first) ----------------

    @pytest.mark.asyncio
    async def test_ab_test_old_version_first(self, server):
        """Site A/B tests a new layout. First request gets old version.

        Very common during gradual rollouts. The cached script works
        with the old version. But the new version is coming.
        """
        server.program("/products", PRODUCTS_PAGE)
        url = server.url("/products")

        result = await diagnose(
            make_execute_fn(url, script_waits_for(".product")),
            url,
            AutoFixMode.BALANCED,
        )

        # FINDING: Attempt 1 succeeds — returns data from old version.
        # The algorithm never sees the new version. The user gets data,
        # but it might be from a minority variant being phased out.
        # The algorithm cannot detect A/B tests.
        assert isinstance(result, AttemptResult)
        assert result.success is True

    # -- 19. The A/B Test (new version every time) ------------------

    @pytest.mark.asyncio
    async def test_ab_test_new_version_always(self, server):
        """A/B test where the scraper always gets the new variant.

        Maybe the A/B is cookie-based and fresh browsers always get
        variant B. The script expects .product but the page has
        .product-card.
        """
        server.program("/products", AB_TEST_VARIANT)
        url = server.url("/products")

        result = await diagnose(
            make_execute_fn(url, script_waits_for(".product")),
            url,
            AutoFixMode.BALANCED,
        )

        assert isinstance(result, DiagnosisResult)
        # This is actually correct — the script needs updating.
        assert result.action == AutoFixAction.REGENERATE

    # -- 20. The 429 Then Recovery ----------------------------------

    @pytest.mark.asyncio
    async def test_rate_limited_then_recovers(self, server):
        """Site rate-limits with 429, then allows requests again.

        Common with APIs and rate-limited sites. First request hits
        the rate limit, subsequent requests go through.
        """
        server.program(
            "/products",
            SERVER_429,
            PRODUCTS_PAGE,
        )
        url = server.url("/products")

        result = await diagnose(
            make_execute_fn(url, script_waits_for(".product")),
            url,
            AutoFixMode.BALANCED,
        )

        # Attempt 1 fails (429). Attempt 2 succeeds. Returns data.
        assert isinstance(result, AttemptResult)
        assert result.success is True

    # -- 21. The CDN Stale Cache ------------------------------------

    @pytest.mark.asyncio
    async def test_cdn_stale_then_fresh(self, server):
        """CDN serves stale cached page, then fresh page on retry.

        Common during deployments. CDN edge A has old version cached.
        Since attempt 1 succeeds, the algorithm returns the data.
        """
        server.program(
            "/products",
            PRODUCTS_PAGE,     # Stale CDN cache (script works)
            CHANGED_LAYOUT,    # Fresh version (script would fail)
        )
        url = server.url("/products")

        result = await diagnose(
            make_execute_fn(url, script_waits_for(".product")),
            url,
            AutoFixMode.BALANCED,
        )

        # FINDING: The algorithm returns potentially stale CDN data.
        # If the first attempt succeeds, it returns data regardless
        # of whether newer page versions exist.
        assert isinstance(result, AttemptResult)
        assert result.success is True

    # -- 22. A Different Anti-Bot Provider --------------------------

    @pytest.mark.asyncio
    async def test_datadome_challenge(self, server):
        """Site uses DataDome (not Cloudflare). Different provider.

        Tests that anti-bot detection works across providers.
        DataDome uses captcha-delivery.com iframes and x-dd-b headers.
        """
        server.program("/products", DATADOME_CHALLENGE)
        url = server.url("/products")

        result = await diagnose(
            make_execute_fn(url, script_queries(".product")),
            url,
            AutoFixMode.AGGRESSIVE,
        )

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.RAISE
        assert any(
            r == PageVerificationResult.ANTI_BOT
            for r in result.page_results
        )

    # -- 23. The Partial Page Load ----------------------------------

    @pytest.mark.asyncio
    async def test_partial_page_load(self, server):
        """Server sends partial HTML. Only some elements loaded.

        Common with overloaded servers or connection drops mid-transfer.
        The script finds some data but not enough to satisfy the schema.
        """
        server.program("/products", PARTIAL_PRODUCTS)
        url = server.url("/products")

        result = await diagnose(
            make_execute_fn(
                url,
                script_extracts_items(
                    data=[{"name": "Widget A", "price": "9.99"}],
                    schema_error="Expected at least 5 items, got 1",
                ),
            ),
            url,
            AutoFixMode.BALANCED,
        )

        assert isinstance(result, DiagnosisResult)
        # FINDING: Partial page loads returning HTTP 200 are
        # indistinguishable from pages that legitimately have few items.
        # The algorithm regenerates, but a new script would also
        # only find 1 item on the partial page.
        assert result.action == AutoFixAction.REGENERATE

    # -- 24. Total Chaos --------------------------------------------

    @pytest.mark.asyncio
    async def test_everything_goes_wrong(self, server):
        """Every attempt fails differently. Total chaos.

        Attempt 1: server error. Attempt 2: anti-bot. Attempt 3:
        server error again. Nothing is consistent.
        """
        server.program(
            "/products",
            SERVER_503,
            CLOUDFLARE_CHALLENGE,
            SERVER_503,
        )
        url = server.url("/products")

        result = await diagnose(
            make_execute_fn(url, script_queries(".product")),
            url,
            AutoFixMode.AGGRESSIVE,
        )

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.RAISE

    # -- 25. The Happy Path -----------------------------------------

    @pytest.mark.asyncio
    async def test_everything_works(self, server):
        """Page is fine, script is fine. Just works.

        Baseline — the algorithm must not interfere with a working
        scraper.
        """
        server.program("/products", PRODUCTS_PAGE)
        url = server.url("/products")

        result = await diagnose(
            make_execute_fn(url, script_waits_for(".product")),
            url,
            AutoFixMode.BALANCED,
        )

        assert isinstance(result, AttemptResult)
        assert result.success is True

    # -- 26. Server Recovers on Third Try ---------------------------

    @pytest.mark.asyncio
    async def test_server_recovers_on_third_attempt(self, server):
        """Server is down, then recovers on the third attempt.

        Common after deployments, restarts, or transient outages.
        """
        server.program(
            "/products",
            SERVER_503,
            SERVER_503,
            PRODUCTS_PAGE,
        )
        url = server.url("/products")

        result = await diagnose(
            make_execute_fn(url, script_waits_for(".product")),
            url,
            AutoFixMode.BALANCED,
        )

        assert isinstance(result, AttemptResult)
        assert result.success is True

    # -- 27. Script Crashes at Different Lines ----------------------

    @pytest.mark.asyncio
    async def test_script_crashes_at_different_lines(self, server):
        """Script crashes at different points on each attempt.

        Website redesigned. Script crashes on line 5 (AttributeError)
        attempt 1, line 12 (KeyError) attempt 2, line 5 again attempt 3.
        Same root cause (layout changed) but different crash points.
        """
        server.program("/products", CHANGED_LAYOUT)
        url = server.url("/products")

        counter = [0]

        async def execute() -> AttemptResult:
            counter[0] += 1
            status, body, headers, cookies = await _fetch(url)
            signals = _build_signals(status, body, headers, cookies, url)

            errors = [
                (
                    "Traceback (most recent call last):\n"
                    '  File "script.py", line 5, in scrape\n'
                    "    name = await el.text_content()\n"
                    "AttributeError: 'NoneType' object has no attribute "
                    "'text_content'"
                ),
                (
                    "Traceback (most recent call last):\n"
                    '  File "script.py", line 12, in scrape\n'
                    "    price = data['price']\n"
                    "KeyError: 'price'"
                ),
                (
                    "Traceback (most recent call last):\n"
                    '  File "script.py", line 5, in scrape\n'
                    "    name = await el.text_content()\n"
                    "AttributeError: 'NoneType' object has no attribute "
                    "'text_content'"
                ),
            ]
            idx = min(counter[0] - 1, len(errors) - 1)
            return AttemptResult(
                success=False,
                error=errors[idx],
                page_signals=signals,
            )

        result = await diagnose(execute, url, AutoFixMode.BALANCED)

        assert isinstance(result, DiagnosisResult)
        # Script crashes at multiple points — same root cause.
        # Algorithm should still regenerate.
        assert result.action == AutoFixAction.REGENERATE

    # -- 28. Aggressive Mode Still Blocked by Anti-Bot --------------

    @pytest.mark.asyncio
    async def test_aggressive_still_blocked_by_antibot(self, server):
        """Aggressive mode should not override anti-bot protection.

        2 real pages + 1 anti-bot. Aggressive tolerates server errors
        but anti-bot is different — a new script will be blocked too.
        """
        server.program(
            "/products",
            CHANGED_LAYOUT,
            CHANGED_LAYOUT,
            CLOUDFLARE_CHALLENGE,
        )
        url = server.url("/products")

        result = await diagnose(
            make_execute_fn(url, script_queries(".product")),
            url,
            AutoFixMode.AGGRESSIVE,
        )

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.RAISE

    # -- 29. Full Outage + Aggressive Mode --------------------------

    @pytest.mark.asyncio
    async def test_full_outage_aggressive_mode(self, server):
        """Complete server outage. Even aggressive shouldn't regenerate.

        No script can work against a server returning 503 on every
        request.
        """
        server.program("/products", SERVER_503)
        url = server.url("/products")

        result = await diagnose(
            make_execute_fn(url, script_queries(".product")),
            url,
            AutoFixMode.AGGRESSIVE,
        )

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.RAISE

    # -- 30. Conservative with Clear Crash --------------------------

    @pytest.mark.asyncio
    async def test_conservative_with_runtime_crash(self, server):
        """Conservative mode with a clear runtime crash (not a timeout).

        AttributeError is high-confidence — the element doesn't exist.
        Even conservative mode should regenerate for clear crashes
        on verified real pages.
        """
        server.program("/products", CHANGED_LAYOUT)
        url = server.url("/products")

        result = await diagnose(
            make_execute_fn(url, script_queries(".product")),
            url,
            AutoFixMode.CONSERVATIVE,
        )

        assert isinstance(result, DiagnosisResult)
        assert result.action == AutoFixAction.REGENERATE
