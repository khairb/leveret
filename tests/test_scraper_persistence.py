"""Tests for script persistence — save, load, metadata, config mismatch.

Track 2 of the Scraper class implementation. Tests cover:
- Metadata docstring generation and parsing
- Metadata escaping round-trips (triple quotes, backslashes, newlines)
- Script save with auto-directory creation
- Script load with full validation chain (syntax, function, async, signature)
- Config mismatch detection (domain, task)
- Filesystem error handling
- Real-world edge cases
"""

import asyncio
import logging
import textwrap
from pathlib import Path

import pytest

from scout.errors import ScoutConfigError, ScoutError, ScoutScriptLoadError
from scout.scraper import (
    _build_metadata_docstring,
    _check_domain_mismatch,
    _check_task_mismatch,
    _escape_metadata,
    _load_script,
    _parse_script_metadata,
    _save_script,
    _unescape_metadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_SCRAPE_CODE = textwrap.dedent("""\
    async def scrape(page, url, checkpoint):
        return [{"title": "Test", "price": 9.99}]
""")

URL = "https://example.com/products"
TASK = "Extract product prices"
MODEL = "claude-haiku-4-5"


def _write_script(path: Path, code: str, *, with_metadata: bool = True) -> None:
    """Write a script file, optionally with metadata docstring."""
    if with_metadata:
        content = (
            '"""\nScout Script\n\n'
            f"url:           {URL}\n"
            f"task:          {TASK}\n"
            f"generated:     2024-01-01T00:00:00Z\n"
            f"model:         {MODEL}\n"
            f"scout_version: 0.1.0\n"
            '"""\n\n'
        ) + code
    else:
        content = code
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# Metadata escaping
# ═══════════════════════════════════════════════════════════════════════════

class TestMetadataEscaping:

    def test_simple_string_unchanged(self):
        assert _escape_metadata("hello world") == "hello world"
        assert _unescape_metadata("hello world") == "hello world"

    def test_triple_quotes_escaped(self):
        val = 'has """ triple quotes'
        escaped = _escape_metadata(val)
        assert '"""' not in escaped
        assert _unescape_metadata(escaped) == val

    def test_backslash_escaped(self):
        val = "has\\backslash"
        escaped = _escape_metadata(val)
        assert _unescape_metadata(escaped) == val

    def test_newline_escaped(self):
        val = "line1\nline2"
        escaped = _escape_metadata(val)
        assert "\n" not in escaped
        assert _unescape_metadata(escaped) == val

    def test_all_special_chars_combined(self):
        val = 'all: """ and \\ and \n combined'
        escaped = _escape_metadata(val)
        restored = _unescape_metadata(escaped)
        assert restored == val

    def test_url_with_query_params(self):
        val = "https://example.com/?q=foo&bar=baz"
        assert _unescape_metadata(_escape_metadata(val)) == val

    def test_empty_string(self):
        assert _escape_metadata("") == ""
        assert _unescape_metadata("") == ""

    @pytest.mark.parametrize("val", [
        '"""',
        '""""""',
        'a"""b"""c',
        "\\\\\\",
        "\n\n\n",
        'mixed\n"""\n\\end',
        "unicode: äöü 日本語 🎯",
    ])
    def test_round_trip(self, val):
        assert _unescape_metadata(_escape_metadata(val)) == val


# ═══════════════════════════════════════════════════════════════════════════
# Metadata docstring building and parsing
# ═══════════════════════════════════════════════════════════════════════════

class TestMetadataDocstring:

    def test_build_contains_all_fields(self):
        doc = _build_metadata_docstring(URL, TASK, MODEL, "2024-01-01T00:00:00Z")
        assert "Scout Script" in doc
        assert f"url:           {URL}" in doc
        assert f"task:          {TASK}" in doc
        assert "generated:     2024-01-01T00:00:00Z" in doc
        assert f"model:         {MODEL}" in doc
        assert "scout_version:" in doc

    def test_build_is_valid_docstring(self):
        doc = _build_metadata_docstring(URL, TASK, MODEL, "2024-01-01T00:00:00Z")
        assert doc.startswith('"""')
        assert doc.endswith('"""\n')

    def test_parse_extracts_all_fields(self):
        doc = _build_metadata_docstring(URL, TASK, MODEL, "2024-01-01T00:00:00Z")
        meta = _parse_script_metadata(doc + "\nasync def scrape(): pass\n")
        assert meta["url"] == URL
        assert meta["task"] == TASK
        assert meta["generated"] == "2024-01-01T00:00:00Z"
        assert meta["model"] == MODEL
        assert "scout_version" in meta

    def test_parse_round_trip_with_special_chars(self):
        url = "https://example.com/?q=test&page=1"
        task = 'Extract "prices" from\nthe page'
        doc = _build_metadata_docstring(url, task, MODEL, "2024-01-01T00:00:00Z")
        full = doc + "\nasync def scrape(): pass\n"
        meta = _parse_script_metadata(full)
        assert meta["url"] == url
        assert meta["task"] == task

    def test_parse_no_docstring(self):
        meta = _parse_script_metadata("async def scrape(): pass")
        assert meta == {}

    def test_parse_empty_string(self):
        meta = _parse_script_metadata("")
        assert meta == {}


# ═══════════════════════════════════════════════════════════════════════════
# Script saving
# ═══════════════════════════════════════════════════════════════════════════

class TestSaveScript:

    def test_creates_file(self, tmp_path):
        path = tmp_path / "scraper.py"
        _save_script(VALID_SCRAPE_CODE, path, URL, TASK, MODEL)
        assert path.exists()

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "dir" / "scraper.py"
        _save_script(VALID_SCRAPE_CODE, path, URL, TASK, MODEL)
        assert path.exists()

    def test_file_has_metadata_docstring(self, tmp_path):
        path = tmp_path / "scraper.py"
        _save_script(VALID_SCRAPE_CODE, path, URL, TASK, MODEL)
        content = path.read_text()
        assert content.startswith('"""')
        assert "Scout Script" in content
        assert URL in content
        assert TASK in content

    def test_file_has_code_after_metadata(self, tmp_path):
        path = tmp_path / "scraper.py"
        _save_script(VALID_SCRAPE_CODE, path, URL, TASK, MODEL)
        content = path.read_text()
        assert "async def scrape(page, url, checkpoint):" in content

    def test_metadata_then_code_parseable(self, tmp_path):
        path = tmp_path / "scraper.py"
        _save_script(VALID_SCRAPE_CODE, path, URL, TASK, MODEL)
        content = path.read_text()
        # Should be valid Python
        import ast
        ast.parse(content)

    def test_overwrites_existing_file(self, tmp_path):
        path = tmp_path / "scraper.py"
        path.write_text("old content")
        _save_script(VALID_SCRAPE_CODE, path, URL, TASK, MODEL)
        content = path.read_text()
        assert "old content" not in content
        assert "Scout Script" in content

    def test_metadata_with_special_chars_in_url(self, tmp_path):
        path = tmp_path / "scraper.py"
        url = "https://example.com/?q=test&foo=bar"
        _save_script(VALID_SCRAPE_CODE, path, url, TASK, MODEL)
        content = path.read_text()
        meta = _parse_script_metadata(content)
        assert meta["url"] == url

    def test_metadata_with_special_chars_in_task(self, tmp_path):
        path = tmp_path / "scraper.py"
        task = 'Extract "all" prices\nfrom page'
        _save_script(VALID_SCRAPE_CODE, path, URL, task, MODEL)
        content = path.read_text()
        meta = _parse_script_metadata(content)
        assert meta["task"] == task

    def test_timestamp_is_utc(self, tmp_path):
        path = tmp_path / "scraper.py"
        _save_script(VALID_SCRAPE_CODE, path, URL, TASK, MODEL)
        content = path.read_text()
        meta = _parse_script_metadata(content)
        assert meta["generated"].endswith("Z")

    def test_file_parent_is_regular_file_error(self, tmp_path):
        """If parent path is a file, not a dir, ScoutError is raised."""
        blocker = tmp_path / "blocker"
        blocker.write_text("I am a file")
        path = blocker / "scraper.py"
        with pytest.raises(ScoutError, match="a file with that name already exists"):
            _save_script(VALID_SCRAPE_CODE, path, URL, TASK, MODEL)


# ═══════════════════════════════════════════════════════════════════════════
# Script loading — happy path
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadScriptHappy:

    def test_loads_valid_script(self, tmp_path):
        path = tmp_path / "scraper.py"
        _write_script(path, VALID_SCRAPE_CODE)
        fn, meta = _load_script(path)
        assert callable(fn)
        assert asyncio.iscoroutinefunction(fn)

    def test_returns_metadata(self, tmp_path):
        path = tmp_path / "scraper.py"
        _write_script(path, VALID_SCRAPE_CODE)
        fn, meta = _load_script(path)
        assert meta["url"] == URL
        assert meta["task"] == TASK

    def test_function_executes(self, tmp_path):
        path = tmp_path / "scraper.py"
        _write_script(path, VALID_SCRAPE_CODE)
        fn, _ = _load_script(path)
        result = asyncio.run(fn(None, "https://x.com", lambda *a, **k: None))
        assert isinstance(result, list)
        assert result[0]["title"] == "Test"

    def test_script_with_helpers(self, tmp_path):
        """Script can have helper functions alongside scrape."""
        code = textwrap.dedent("""\
            def _parse_price(text):
                return float(text.replace("$", ""))

            async def scrape(page, url, checkpoint):
                return [{"price": _parse_price("$9.99")}]
        """)
        path = tmp_path / "scraper.py"
        _write_script(path, code)
        fn, _ = _load_script(path)
        result = asyncio.run(fn(None, "url", lambda *a, **k: None))
        assert result[0]["price"] == 9.99

    def test_script_without_metadata(self, tmp_path):
        """Script without metadata docstring still loads."""
        path = tmp_path / "scraper.py"
        _write_script(path, VALID_SCRAPE_CODE, with_metadata=False)
        fn, meta = _load_script(path)
        assert callable(fn)
        assert meta == {}  # no metadata found

    def test_save_then_load_round_trip(self, tmp_path):
        """Full round-trip: save → load → execute."""
        path = tmp_path / "scraper.py"
        _save_script(VALID_SCRAPE_CODE, path, URL, TASK, MODEL)
        fn, meta = _load_script(path)
        assert meta["url"] == URL
        result = asyncio.run(fn(None, "url", lambda *a, **k: None))
        assert isinstance(result, list)


# ═══════════════════════════════════════════════════════════════════════════
# Script loading — error paths
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadScriptErrors:

    def test_empty_file(self, tmp_path):
        path = tmp_path / "scraper.py"
        path.write_text("")
        with pytest.raises(ScoutScriptLoadError, match="is empty"):
            _load_script(path)

    def test_whitespace_only_file(self, tmp_path):
        path = tmp_path / "scraper.py"
        path.write_text("   \n\n  ")
        with pytest.raises(ScoutScriptLoadError, match="is empty"):
            _load_script(path)

    def test_syntax_error(self, tmp_path):
        path = tmp_path / "scraper.py"
        path.write_text("def scrape(:\n  pass")
        with pytest.raises(ScoutScriptLoadError, match="syntax error"):
            _load_script(path)

    def test_no_scrape_function(self, tmp_path):
        path = tmp_path / "scraper.py"
        path.write_text("async def extract(page, url, checkpoint): pass\n")
        with pytest.raises(ScoutScriptLoadError, match='no function named "scrape"'):
            _load_script(path)

    def test_sync_scrape_function(self, tmp_path):
        path = tmp_path / "scraper.py"
        path.write_text("def scrape(page, url, checkpoint): pass\n")
        with pytest.raises(ScoutScriptLoadError, match="must be async"):
            _load_script(path)

    def test_wrong_signature_missing_param(self, tmp_path):
        path = tmp_path / "scraper.py"
        path.write_text("async def scrape(page, url): pass\n")
        with pytest.raises(ScoutScriptLoadError, match="wrong signature"):
            _load_script(path)

    def test_wrong_signature_extra_param(self, tmp_path):
        path = tmp_path / "scraper.py"
        path.write_text("async def scrape(page, url, checkpoint, extra): pass\n")
        with pytest.raises(ScoutScriptLoadError, match="wrong signature"):
            _load_script(path)

    def test_wrong_signature_wrong_names(self, tmp_path):
        path = tmp_path / "scraper.py"
        path.write_text("async def scrape(browser, link, cb): pass\n")
        with pytest.raises(ScoutScriptLoadError, match="wrong signature"):
            _load_script(path)

    def test_wrong_param_order(self, tmp_path):
        path = tmp_path / "scraper.py"
        path.write_text("async def scrape(url, page, checkpoint): pass\n")
        with pytest.raises(ScoutScriptLoadError, match="wrong signature"):
            _load_script(path)

    def test_error_message_includes_file_path(self, tmp_path):
        path = tmp_path / "my_scraper.py"
        path.write_text("x = 1\n")
        with pytest.raises(ScoutScriptLoadError, match="my_scraper.py"):
            _load_script(path)

    def test_error_message_suggests_regeneration(self, tmp_path):
        path = tmp_path / "scraper.py"
        path.write_text("x = 1\n")
        with pytest.raises(ScoutScriptLoadError, match="regenerate=True"):
            _load_script(path)

    def test_file_not_found(self, tmp_path):
        path = tmp_path / "nonexistent.py"
        with pytest.raises(ScoutScriptLoadError):
            _load_script(path)

    def test_exec_error_in_script(self, tmp_path):
        """Script that raises during module execution."""
        path = tmp_path / "scraper.py"
        path.write_text(
            "raise RuntimeError('boom at import time')\n"
            "async def scrape(page, url, checkpoint): pass\n"
        )
        with pytest.raises(ScoutScriptLoadError, match="failed to load"):
            _load_script(path)


# ═══════════════════════════════════════════════════════════════════════════
# Config mismatch detection
# ═══════════════════════════════════════════════════════════════════════════

class TestDomainMismatch:

    def test_same_domain_different_path_ok(self):
        # Should not raise
        _check_domain_mismatch(
            Path("./hn.py"),
            "https://example.com/products",
            "https://example.com/categories",
        )

    def test_different_domain_raises(self):
        with pytest.raises(ScoutConfigError, match="generated for a different site"):
            _check_domain_mismatch(
                Path("./hn.py"),
                "https://news.ycombinator.com",
                "https://reddit.com/r/python",
            )

    def test_error_message_contains_both_urls(self):
        with pytest.raises(ScoutConfigError) as exc_info:
            _check_domain_mismatch(
                Path("./hn.py"),
                "https://news.ycombinator.com",
                "https://reddit.com/r/python",
            )
        msg = str(exc_info.value)
        assert "news.ycombinator.com" in msg
        assert "reddit.com" in msg

    def test_error_message_has_actionable_fixes(self):
        with pytest.raises(ScoutConfigError) as exc_info:
            _check_domain_mismatch(
                Path("./hn.py"),
                "https://a.com",
                "https://b.com",
            )
        msg = str(exc_info.value)
        assert "regenerate=True" in msg
        assert "script=" in msg

    def test_www_normalization_ok(self):
        # www.example.com vs example.com — should NOT raise
        _check_domain_mismatch(
            Path("./s.py"),
            "https://www.example.com/a",
            "https://example.com/b",
        )

    def test_http_vs_https_ok(self):
        # Same host, different scheme — OK
        _check_domain_mismatch(
            Path("./s.py"),
            "http://example.com",
            "https://example.com",
        )

    def test_subdomain_mismatch_raises(self):
        with pytest.raises(ScoutConfigError):
            _check_domain_mismatch(
                Path("./s.py"),
                "https://m.example.com",
                "https://example.com",
            )

    def test_empty_script_url_skips_check(self):
        """If metadata has no URL, skip the check."""
        _check_domain_mismatch(Path("./s.py"), "", "https://example.com")

    def test_script_path_in_error_message(self):
        with pytest.raises(ScoutConfigError, match="scrapers/hn.py"):
            _check_domain_mismatch(
                Path("./scrapers/hn.py"),
                "https://a.com",
                "https://b.com",
            )


class TestTaskMismatch:

    def test_same_task_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="scout"):
            _check_task_mismatch("Extract prices", "Extract prices")
        assert "changed" not in caplog.text

    def test_different_task_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="scout"):
            _check_task_mismatch("Extract prices", "Extract reviews")
        assert "Task description has changed" in caplog.text
        assert "regenerate=True" in caplog.text

    def test_empty_script_task_no_warning(self, caplog):
        """If metadata has no task, skip the check."""
        with caplog.at_level(logging.WARNING, logger="scout"):
            _check_task_mismatch("", "Extract reviews")
        assert "changed" not in caplog.text


# ═══════════════════════════════════════════════════════════════════════════
# Real-world edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestRealWorldEdgeCases:

    def test_script_with_triple_quote_in_url(self, tmp_path):
        """URL with triple quotes round-trips through save/load."""
        path = tmp_path / "scraper.py"
        url = 'https://example.com/?q=test"""more'
        _save_script(VALID_SCRAPE_CODE, path, url, TASK, MODEL)
        content = path.read_text()
        # File should be valid Python (docstring not broken)
        import ast
        ast.parse(content)
        meta = _parse_script_metadata(content)
        assert meta["url"] == url

    def test_script_with_newlines_in_task(self, tmp_path):
        path = tmp_path / "scraper.py"
        task = "Extract:\n- prices\n- titles\n- descriptions"
        _save_script(VALID_SCRAPE_CODE, path, URL, task, MODEL)
        content = path.read_text()
        import ast
        ast.parse(content)
        meta = _parse_script_metadata(content)
        assert meta["task"] == task

    def test_script_with_backslashes_in_task(self, tmp_path):
        path = tmp_path / "scraper.py"
        task = "Extract \\prices\\ from C:\\Users\\data"
        _save_script(VALID_SCRAPE_CODE, path, URL, task, MODEL)
        content = path.read_text()
        import ast
        ast.parse(content)
        meta = _parse_script_metadata(content)
        assert meta["task"] == task

    def test_complex_real_world_script(self, tmp_path):
        """A realistic script with helpers, imports-like patterns, etc."""
        code = textwrap.dedent("""\
            import re

            def _clean_price(text):
                m = re.search(r"[\\d.]+", text)
                return float(m.group()) if m else 0.0

            async def scrape(page, url, checkpoint):
                items = []
                cards = await page.query_selector_all(".product-card")
                for card in cards:
                    title = await (await card.query_selector("h2")).inner_text()
                    price_text = await (await card.query_selector(".price")).inner_text()
                    items.append({
                        "title": title.strip(),
                        "price": _clean_price(price_text),
                    })
                    checkpoint("product", items[-1])
                return items
        """)
        path = tmp_path / "scraper.py"
        _save_script(code, path, URL, TASK, MODEL)
        fn, meta = _load_script(path)
        assert asyncio.iscoroutinefunction(fn)
        assert meta["url"] == URL

    def test_user_edited_script_loads(self, tmp_path):
        """User edited the script — added comments, changed logic."""
        path = tmp_path / "scraper.py"
        _write_script(path, textwrap.dedent("""\
            # User edited: added filtering
            async def scrape(page, url, checkpoint):
                # Only get products over $10
                items = [{"title": "Expensive", "price": 99.99}]
                return [i for i in items if i["price"] > 10]
        """))
        fn, _ = _load_script(path)
        result = asyncio.run(fn(None, "url", lambda *a, **k: None))
        assert len(result) == 1

    def test_multiple_save_load_cycles(self, tmp_path):
        """Overwrite and reload multiple times."""
        path = tmp_path / "scraper.py"
        for i in range(5):
            code = f'async def scrape(page, url, checkpoint): return [{{"n": {i}}}]\n'
            _save_script(code, path, URL, f"Task v{i}", MODEL)
            fn, meta = _load_script(path)
            result = asyncio.run(fn(None, "url", lambda *a, **k: None))
            assert result[0]["n"] == i
            assert meta["task"] == f"Task v{i}"

    def test_domain_mismatch_with_run_url_override(self):
        """The run(url=...) override URL is checked against script domain."""
        # Same domain, different path — OK
        _check_domain_mismatch(
            Path("./product.py"),
            "https://example.com/product/123",
            "https://example.com/product/456",
        )

        # Different domain — Error
        with pytest.raises(ScoutConfigError):
            _check_domain_mismatch(
                Path("./product.py"),
                "https://example.com/product/123",
                "https://other-store.com/product/456",
            )

    def test_spec_url_mismatch_table(self):
        """Verify all cases from the spec's URL mismatch table."""
        base = Path("./s.py")

        # Same domain, different path — OK
        _check_domain_mismatch(base, "https://example.com/product/123",
                                "https://example.com/product/456")
        _check_domain_mismatch(base, "https://example.com/products",
                                "https://example.com/categories")

        # Different domain — Error
        with pytest.raises(ScoutConfigError):
            _check_domain_mismatch(base, "https://news.ycombinator.com",
                                    "https://reddit.com")

        # Same host, different scheme — OK
        _check_domain_mismatch(base, "http://example.com",
                                "https://example.com")
