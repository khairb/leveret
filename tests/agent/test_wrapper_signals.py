"""Tests for subprocess wrapper page signal collection (Phase 5, Task 5.5).

Tests verify:
  - Generated wrapper code is valid Python (syntax check)
  - Basic mode (collect_page_signals=False) unchanged — no signal markers
  - Signal mode (collect_page_signals=True) includes response listener, signal
    collection, and PAGE_SIGNALS markers
  - parse_page_signals() extracts valid signal dicts
  - parse_page_signals() returns None for backward-compat cases
  - parse_page_signals() handles edge cases (malformed JSON, empty, etc.)
  - Signal markers don't interfere with return value markers
"""

from __future__ import annotations

import ast
import json

from scout.agent.wrapper import (
    PAGE_SIGNALS_END,
    PAGE_SIGNALS_START,
    RETURN_VALUE_END,
    RETURN_VALUE_START,
    generate_subprocess_wrapper,
    parse_page_signals,
    parse_return_value,
)

# -- Helpers ---------------------------------------------------------------

SIMPLE_SCRAPE = """\
async def scrape(page, url, checkpoint):
    return [{"title": "Test"}]
"""


def _gen(*, collect_page_signals: bool = False) -> str:
    """Generate a wrapper with the simple scrape function."""
    return generate_subprocess_wrapper(
        SIMPLE_SCRAPE,
        "https://example.com",
        "/tmp/cp",
        collect_page_signals=collect_page_signals,
    )


def _make_stdout_with_signals(signals_dict: dict) -> str:
    """Build a stdout string with page signals markers."""
    signals_json = json.dumps(signals_dict, ensure_ascii=False)
    return f"some checkpoint output\n{PAGE_SIGNALS_START}\n{signals_json}\n{PAGE_SIGNALS_END}\n"


def _make_stdout_with_return_value(data: object) -> str:
    """Build a stdout string with return value markers."""
    rv_json = json.dumps(data, ensure_ascii=False, indent=2)
    return f"checkpoint output\n{RETURN_VALUE_START}\n{rv_json}\n{RETURN_VALUE_END}\n"


# -- Test: Generated wrapper syntax ----------------------------------------


class TestGeneratedWrapperSyntax:
    """Both wrapper modes produce valid Python."""

    def test_basic_mode_valid_python(self):
        code = _gen(collect_page_signals=False)
        ast.parse(code)

    def test_signal_mode_valid_python(self):
        code = _gen(collect_page_signals=True)
        ast.parse(code)

    def test_signal_mode_with_complex_agent_code(self):
        """Signal mode works with multi-line agent code containing f-strings."""
        complex_code = """\
async def scrape(page, url, checkpoint):
    items = []
    for el in await page.query_selector_all(".item"):
        name = await el.text_content()
        items.append({"name": name, "url": f"{url}/detail"})
    await checkpoint("done", data_preview=items)
    return items
"""
        code = generate_subprocess_wrapper(
            complex_code,
            "https://example.com",
            "/tmp/cp",
            collect_page_signals=True,
        )
        ast.parse(code)


# -- Test: Basic mode backward compatibility -------------------------------


class TestBasicModeBackwardCompat:
    """collect_page_signals=False produces the original wrapper."""

    def test_no_signal_markers(self):
        code = _gen(collect_page_signals=False)
        assert PAGE_SIGNALS_START not in code
        assert PAGE_SIGNALS_END not in code

    def test_has_return_value_markers(self):
        code = _gen(collect_page_signals=False)
        assert RETURN_VALUE_START in code
        assert RETURN_VALUE_END in code

    def test_no_response_listener(self):
        code = _gen(collect_page_signals=False)
        assert "_doc_responses" not in code
        assert "_on_doc_response" not in code

    def test_no_page_content_collection(self):
        code = _gen(collect_page_signals=False)
        # Signal-collection page.content() uses asyncio.wait_for;
        # checkpoint's page.content() for HTML capture is expected.
        assert '_signals["content"]' not in code
        assert "context.cookies()" not in code


# -- Test: Signal mode content ---------------------------------------------


class TestSignalModeContent:
    """collect_page_signals=True adds signal collection infrastructure."""

    def test_has_signal_markers(self):
        code = _gen(collect_page_signals=True)
        assert PAGE_SIGNALS_START in code
        assert PAGE_SIGNALS_END in code

    def test_has_return_value_markers(self):
        """Success path still prints return value."""
        code = _gen(collect_page_signals=True)
        assert RETURN_VALUE_START in code
        assert RETURN_VALUE_END in code

    def test_has_response_listener(self):
        code = _gen(collect_page_signals=True)
        assert "_doc_responses" in code
        assert "_on_doc_response" in code
        assert 'page.on("response"' in code
        assert 'resource_type == "document"' in code

    def test_collects_page_url(self):
        code = _gen(collect_page_signals=True)
        assert "page.url" in code

    def test_collects_page_content_with_timeout(self):
        code = _gen(collect_page_signals=True)
        assert "page.content()" in code
        assert "timeout=5.0" in code
        assert "asyncio.wait_for" in code

    def test_collects_response_status(self):
        code = _gen(collect_page_signals=True)
        assert "_last_resp.status" in code

    def test_collects_response_headers(self):
        code = _gen(collect_page_signals=True)
        assert "_last_resp.headers" in code

    def test_collects_cookies(self):
        code = _gen(collect_page_signals=True)
        assert "context.cookies()" in code

    def test_seeds_goto_response(self):
        """Initial document response from page.goto() is captured."""
        code = _gen(collect_page_signals=True)
        assert "_goto_response" in code
        assert "_doc_responses.append(_goto_response)" in code

    def test_signals_in_try_except(self):
        """Every signal access is wrapped in try/except."""
        code = _gen(collect_page_signals=True)
        # The signal collection section is between _signals = {} and
        # the re-raise check
        signal_section_start = code.find("_signals = {}")
        signal_section_end = code.find("if _scrape_exc is not None:")
        signal_section = code[signal_section_start:signal_section_end]
        # At least 4 try blocks: page.url, page.content(), response, cookies
        assert signal_section.count("try:") >= 4
        assert signal_section.count("except Exception:") >= 4

    def test_scrape_wrapped_in_try_except(self):
        """scrape() call is wrapped in try/except with deferred re-raise."""
        code = _gen(collect_page_signals=True)
        assert "_scrape_exc = None" in code
        assert "except Exception as _exc:" in code
        assert "_scrape_exc = _exc" in code
        assert "if _scrape_exc is not None:" in code
        assert "raise _scrape_exc" in code


# -- Test: parse_page_signals ----------------------------------------------


class TestParsePageSignals:
    """parse_page_signals() extracts signal dicts from stdout."""

    def test_full_signals(self):
        signals = {
            "http_status": 200,
            "page_url": "https://example.com",
            "content": "<html><body>test</body></html>",
            "headers": {"content-type": "text/html", "server": "nginx"},
            "cookies": [{"name": "sid", "value": "abc123"}],
        }
        result = parse_page_signals(_make_stdout_with_signals(signals))
        assert result is not None
        assert result["http_status"] == 200
        assert result["page_url"] == "https://example.com"
        assert result["content"] == "<html><body>test</body></html>"
        assert result["headers"]["content-type"] == "text/html"
        assert result["cookies"][0]["name"] == "sid"

    def test_partial_signals(self):
        """Only some signals collected (e.g. page crashed mid-collection)."""
        signals = {"http_status": 503, "page_url": "https://example.com"}
        result = parse_page_signals(_make_stdout_with_signals(signals))
        assert result is not None
        assert result["http_status"] == 503
        assert "content" not in result

    def test_empty_signals(self):
        """All signal collection failed — empty dict."""
        result = parse_page_signals(_make_stdout_with_signals({}))
        assert result == {}

    def test_signals_with_large_content(self):
        """Large page content doesn't break parsing."""
        signals = {"content": "x" * 100_000, "http_status": 200}
        result = parse_page_signals(_make_stdout_with_signals(signals))
        assert result is not None
        assert len(result["content"]) == 100_000

    def test_signals_with_unicode(self):
        """Unicode content (e.g. non-Latin pages) is preserved."""
        signals = {
            "content": "<html>日本語テスト</html>",
            "page_url": "https://example.jp/テスト",
        }
        result = parse_page_signals(_make_stdout_with_signals(signals))
        assert "日本語" in result["content"]


# -- Test: parse_page_signals backward compatibility -----------------------


class TestParseSignalsBackwardCompat:
    """parse_page_signals returns None when no markers are present."""

    def test_no_markers(self):
        assert parse_page_signals("just some output") is None

    def test_empty_string(self):
        assert parse_page_signals("") is None

    def test_only_return_value_markers(self):
        stdout = _make_stdout_with_return_value([{"a": 1}])
        assert parse_page_signals(stdout) is None

    def test_only_start_marker(self):
        assert parse_page_signals(f"before {PAGE_SIGNALS_START} after") is None

    def test_only_end_marker(self):
        assert parse_page_signals(f"before {PAGE_SIGNALS_END} after") is None

    def test_reversed_markers(self):
        stdout = f"{PAGE_SIGNALS_END}\ndata\n{PAGE_SIGNALS_START}"
        assert parse_page_signals(stdout) is None


# -- Test: parse_page_signals error handling -------------------------------


class TestParseSignalsErrorHandling:
    """parse_page_signals handles malformed data gracefully."""

    def test_malformed_json(self):
        stdout = f"{PAGE_SIGNALS_START}\nnot valid json\n{PAGE_SIGNALS_END}"
        assert parse_page_signals(stdout) is None

    def test_empty_between_markers(self):
        stdout = f"{PAGE_SIGNALS_START}\n\n{PAGE_SIGNALS_END}"
        assert parse_page_signals(stdout) is None

    def test_json_array_not_dict(self):
        stdout = f"{PAGE_SIGNALS_START}\n[1, 2, 3]\n{PAGE_SIGNALS_END}"
        assert parse_page_signals(stdout) is None

    def test_json_string_not_dict(self):
        stdout = f'{PAGE_SIGNALS_START}\n"just a string"\n{PAGE_SIGNALS_END}'
        assert parse_page_signals(stdout) is None

    def test_json_number_not_dict(self):
        stdout = f"{PAGE_SIGNALS_START}\n42\n{PAGE_SIGNALS_END}"
        assert parse_page_signals(stdout) is None


# -- Test: Non-interference between marker types ---------------------------


class TestMarkerNonInterference:
    """Signal markers and return value markers don't interfere."""

    def test_return_value_parsing_ignores_signal_markers(self):
        """parse_return_value still works when signal markers are present."""
        stdout = f'output\n{PAGE_SIGNALS_START}\n{{"http_status": 200}}\n{PAGE_SIGNALS_END}\n'
        clean, rv = parse_return_value(stdout)
        # No return value markers → rv is None
        assert rv is None

    def test_signal_parsing_ignores_return_value_markers(self):
        """parse_page_signals returns None when only return value markers exist."""
        stdout = _make_stdout_with_return_value([1, 2, 3])
        assert parse_page_signals(stdout) is None

    def test_both_markers_in_stdout(self):
        """Hypothetical: both signal and return value in same stdout.

        This shouldn't happen (signals on failure, return value on success),
        but if it does, each parser extracts its own.
        """
        signals = {"http_status": 200}
        signals_json = json.dumps(signals)
        data = [{"x": 1}]
        data_json = json.dumps(data, indent=2)
        stdout = (
            f"{PAGE_SIGNALS_START}\n{signals_json}\n{PAGE_SIGNALS_END}\n"
            f"{RETURN_VALUE_START}\n{data_json}\n{RETURN_VALUE_END}\n"
        )
        # Each parser finds its own markers
        sig = parse_page_signals(stdout)
        assert sig is not None
        assert sig["http_status"] == 200

        _, rv = parse_return_value(stdout)
        assert rv is not None
        assert json.loads(rv) == [{"x": 1}]
