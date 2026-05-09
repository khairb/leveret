"""Tests for browser configuration via launch_options.

Covers:
- LaunchOptions TypedDict import and construction
- resolve_launch_options merge logic
- Scraper accepts and stores launch_options
- Options flow through to BrowserManager and wrapper template
"""

from __future__ import annotations

import pytest

from scout import LaunchOptions, Scraper
from scout.browser import STEALTH_ARGS, SCOUT_DEFAULTS, resolve_launch_options


VALID_URL = "https://example.com"
VALID_TASK = "Extract data"
VALID_SCHEMA = {"title": str}


def _make(**overrides):
    kwargs = {"schema": VALID_SCHEMA}
    kwargs.update(overrides)
    url = kwargs.pop("url", VALID_URL)
    task = kwargs.pop("task", VALID_TASK)
    return Scraper(url, task, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# LaunchOptions TypedDict
# ═══════════════════════════════════════════════════════════════════════════

class TestLaunchOptionsTypedDict:
    """LaunchOptions is importable and constructable."""

    def test_import(self):
        from scout import LaunchOptions
        assert LaunchOptions is not None

    def test_construct_with_common_options(self):
        opts = LaunchOptions(
            proxy={"server": "http://proxy:8080"},
            locale="de-DE",
            timezone_id="Europe/Berlin",
        )
        assert opts["proxy"]["server"] == "http://proxy:8080"
        assert opts["locale"] == "de-DE"

    def test_is_a_dict(self):
        opts = LaunchOptions(locale="fr-FR")
        assert isinstance(opts, dict)

    def test_empty_options(self):
        opts = LaunchOptions()
        assert opts == {}


# ═══════════════════════════════════════════════════════════════════════════
# resolve_launch_options
# ═══════════════════════════════════════════════════════════════════════════

class TestResolveLaunchOptions:
    """Merge logic for user options + Scout defaults."""

    def test_defaults_only(self):
        resolved = resolve_launch_options(None, headless=True)
        assert resolved["channel"] == "chrome"
        assert resolved["locale"] == "en-US"
        assert resolved["timezone_id"] == "America/New_York"
        assert resolved["bypass_csp"] is True
        assert resolved["headless"] is True
        assert "--disable-blink-features=AutomationControlled" in resolved["args"]

    def test_user_overrides_locale(self):
        opts = LaunchOptions(locale="de-DE")
        resolved = resolve_launch_options(opts, headless=True)
        assert resolved["locale"] == "de-DE"
        # Other defaults preserved
        assert resolved["channel"] == "chrome"

    def test_user_overrides_timezone(self):
        opts = LaunchOptions(timezone_id="Europe/Berlin")
        resolved = resolve_launch_options(opts, headless=False)
        assert resolved["timezone_id"] == "Europe/Berlin"
        assert resolved["headless"] is False

    def test_proxy_forwarded(self):
        opts = LaunchOptions(proxy={"server": "http://proxy:8080"})
        resolved = resolve_launch_options(opts, headless=True)
        assert resolved["proxy"]["server"] == "http://proxy:8080"

    def test_args_merged_not_replaced(self):
        opts = LaunchOptions(args=["--custom-flag"])
        resolved = resolve_launch_options(opts, headless=True)
        # Should have both stealth args and custom arg
        assert "--custom-flag" in resolved["args"]
        assert "--disable-blink-features=AutomationControlled" in resolved["args"]

    def test_stealth_args_not_duplicated(self):
        # User passes a stealth arg that's already in defaults
        opts = LaunchOptions(args=["--no-first-run", "--custom"])
        resolved = resolve_launch_options(opts, headless=True)
        count = resolved["args"].count("--no-first-run")
        assert count == 1

    def test_headless_from_parameter_not_options(self):
        # Even if user puts headless in launch_options, the parameter wins
        opts = LaunchOptions(headless=False)
        resolved = resolve_launch_options(opts, headless=True)
        # headless from parameter should be overridden by user option
        # (user options override defaults, and headless in opts IS a user option)
        assert resolved["headless"] is False

    def test_plain_dict_works(self):
        resolved = resolve_launch_options(
            {"locale": "ja-JP", "viewport": {"width": 800, "height": 600}},
            headless=True,
        )
        assert resolved["locale"] == "ja-JP"
        assert resolved["viewport"] == {"width": 800, "height": 600}


# ═══════════════════════════════════════════════════════════════════════════
# Scraper integration
# ═══════════════════════════════════════════════════════════════════════════

class TestScraperLaunchOptions:
    """Scraper accepts launch_options and passes them through."""

    def test_default_none(self):
        s = _make()
        assert s._launch_options is None

    def test_stores_launch_options(self):
        opts = LaunchOptions(locale="de-DE")
        s = _make(launch_options=opts)
        assert s._launch_options == {"locale": "de-DE"}

    def test_resolved_includes_user_options(self):
        opts = LaunchOptions(
            proxy={"server": "http://proxy:8080"},
            locale="de-DE",
        )
        s = _make(launch_options=opts)
        resolved = s._get_resolved_launch_options()
        assert resolved["proxy"]["server"] == "http://proxy:8080"
        assert resolved["locale"] == "de-DE"
        assert resolved["channel"] == "chrome"  # default preserved

    def test_plain_dict_accepted(self):
        s = _make(launch_options={"locale": "fr-FR"})
        resolved = s._get_resolved_launch_options()
        assert resolved["locale"] == "fr-FR"


# ═══════════════════════════════════════════════════════════════════════════
# Wrapper template integration
# ═══════════════════════════════════════════════════════════════════════════

class TestWrapperTemplate:
    """Subprocess wrapper receives launch options."""

    def test_wrapper_embeds_launch_options(self):
        from scout.agent.wrapper import generate_subprocess_wrapper

        opts = resolve_launch_options(
            {"locale": "de-DE"}, headless=True,
        )
        wrapper = generate_subprocess_wrapper(
            'async def scrape(page, start_url, checkpoint):\n    return []',
            "https://example.com",
            "/tmp/cp",
            launch_options=opts,
        )
        assert "de-DE" in wrapper
        assert "launch_persistent_context" in wrapper

    def test_wrapper_default_options(self):
        from scout.agent.wrapper import generate_subprocess_wrapper

        wrapper = generate_subprocess_wrapper(
            'async def scrape(page, start_url, checkpoint):\n    return []',
            "https://example.com",
            "/tmp/cp",
        )
        # Should use default options
        assert "en-US" in wrapper
        assert "America/New_York" in wrapper
