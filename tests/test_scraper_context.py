"""Tests for Track 4: Context manager, browser lifecycle, and export.

Covers:
- Sync context manager (__enter__/__exit__)
- Async context manager (__aenter__/__aexit__)
- Browser lifecycle (launch, reuse, cleanup)
- In-process execution path
- close() method
- __del__ warning
- export() method
- Edge cases: nesting, regenerate inside with, timeout
"""

import asyncio
import json
import threading
import warnings
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from scout.scraper import Scraper, ScraperResult, _build_pre_import_namespace
from scout.errors import (
    ScoutError,
    ScoutScriptRuntimeError,
    ScoutScriptTimeoutError,
    ScoutValidationError,
)


# ── Helpers ──────────────────────────────────────────────────────

def _make_scraper(tmp_path=None, **kwargs):
    """Create a Scraper with minimal valid params."""
    defaults = dict(
        url="https://example.com",
        task="Extract data",
        schema=[{"title": str}],
    )
    if tmp_path is not None:
        defaults["script"] = str(tmp_path / "scraper.py")
    defaults.update(kwargs)
    return Scraper(**defaults)


def _write_valid_script(path: Path, return_data=None):
    """Write a minimal valid scrape function to disk."""
    if return_data is None:
        return_data = [{"title": "Test"}]
    data_repr = json.dumps(return_data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '"""\nScout Script\n\n'
        'url:           https://example.com\n'
        'task:          Extract data\n'
        'generated:     2024-01-01T00:00:00.000000Z\n'
        'model:         test\n'
        'scout_version: test\n'
        '"""\n\n'
        f'async def scrape(page, url, checkpoint):\n'
        f'    return {data_repr}\n',
        encoding="utf-8",
    )


def _mock_browser_manager():
    """Create a mock BrowserManager with all expected methods."""
    mgr = AsyncMock()
    mgr.headless = True

    mock_page = AsyncMock()
    mock_page.goto = AsyncMock()
    mock_page.close = AsyncMock()

    mgr.start = AsyncMock()
    mgr.new_page = AsyncMock(return_value=mock_page)
    mgr.stop = AsyncMock()

    return mgr, mock_page


# ── Pre-import namespace ─────────────────────────────────────────

class TestPreImportNamespace:
    """Verify _build_pre_import_namespace provides all expected modules."""

    def test_contains_all_expected_modules(self):
        ns = _build_pre_import_namespace()
        expected = [
            "json", "re", "math", "os", "time",
            "asyncio", "tempfile", "shutil",
            "datetime", "urljoin", "urlparse",
        ]
        for name in expected:
            assert name in ns, f"Missing pre-import: {name}"

    def test_modules_are_callable_or_module(self):
        ns = _build_pre_import_namespace()
        # json, re, etc. are modules; datetime, urljoin, urlparse are callables
        assert callable(ns["urljoin"])
        assert callable(ns["urlparse"])
        assert callable(ns["datetime"])

    def test_json_is_the_real_module(self):
        import json
        ns = _build_pre_import_namespace()
        assert ns["json"] is json


# ── Sync context manager ─────────────────────────────────────────

class TestSyncContextManager:

    def test_enter_returns_self(self, tmp_path):
        s = _make_scraper(tmp_path)
        result = s.__enter__()
        assert result is s
        s.__exit__(None, None, None)

    def test_sets_context_managed_flag(self, tmp_path):
        s = _make_scraper(tmp_path)
        assert s._context_managed is False
        s.__enter__()
        assert s._context_managed is True
        s.__exit__(None, None, None)
        assert s._context_managed is False

    def test_creates_background_loop_and_thread(self, tmp_path):
        s = _make_scraper(tmp_path)
        s.__enter__()
        try:
            assert s._bg_loop is not None
            assert s._bg_thread is not None
            assert s._bg_thread.is_alive()
            assert s._bg_thread.daemon is True
            assert s._bg_thread.name == "scout-browser"
        finally:
            s.__exit__(None, None, None)

    def test_exit_cleans_up_loop_and_thread(self, tmp_path):
        s = _make_scraper(tmp_path)
        s.__enter__()
        thread = s._bg_thread
        s.__exit__(None, None, None)
        assert s._bg_loop is None
        assert s._bg_thread is None
        assert not thread.is_alive()

    def test_exit_does_not_suppress_exceptions(self, tmp_path):
        s = _make_scraper(tmp_path)
        result = s.__enter__()
        suppress = s.__exit__(ValueError, ValueError("test"), None)
        assert suppress is False

    def test_nested_context_manager_raises(self, tmp_path):
        s = _make_scraper(tmp_path)
        s.__enter__()
        try:
            with pytest.raises(ScoutError, match="already inside"):
                s.__enter__()
        finally:
            s.__exit__(None, None, None)

    def test_page_count_and_start_time_initialized(self, tmp_path):
        s = _make_scraper(tmp_path)
        s.__enter__()
        try:
            assert s._cm_page_count == 0
            assert s._cm_start_time > 0
        finally:
            s.__exit__(None, None, None)


# ── Async context manager ────────────────────────────────────────

class TestAsyncContextManager:

    @pytest.mark.asyncio
    async def test_aenter_returns_self(self, tmp_path):
        s = _make_scraper(tmp_path)
        result = await s.__aenter__()
        assert result is s
        await s.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_sets_context_managed_flag(self, tmp_path):
        s = _make_scraper(tmp_path)
        await s.__aenter__()
        assert s._context_managed is True
        await s.__aexit__(None, None, None)
        assert s._context_managed is False

    @pytest.mark.asyncio
    async def test_no_background_thread_for_async(self, tmp_path):
        s = _make_scraper(tmp_path)
        await s.__aenter__()
        try:
            # Async context manager doesn't need a background thread
            assert s._bg_loop is None
            assert s._bg_thread is None
        finally:
            await s.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_nested_async_raises(self, tmp_path):
        s = _make_scraper(tmp_path)
        await s.__aenter__()
        try:
            with pytest.raises(ScoutError, match="already inside"):
                await s.__aenter__()
        finally:
            await s.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_exit_closes_browser(self, tmp_path):
        s = _make_scraper(tmp_path)
        mock_mgr, _ = _mock_browser_manager()
        await s.__aenter__()
        s._browser_mgr = mock_mgr
        s._cm_page_count = 5
        await s.__aexit__(None, None, None)
        mock_mgr.stop.assert_awaited_once()
        assert s._browser_mgr is None


# ── Browser lifecycle ─────────────────────────────────────────────

class TestBrowserLifecycle:

    @pytest.mark.asyncio
    async def test_close_browser_idempotent(self, tmp_path):
        s = _make_scraper(tmp_path)
        # No browser — should not raise
        await s._close_browser()
        assert s._browser_mgr is None

    @pytest.mark.asyncio
    async def test_close_browser_stops_manager(self, tmp_path):
        s = _make_scraper(tmp_path)
        mock_mgr, _ = _mock_browser_manager()
        s._browser_mgr = mock_mgr
        s._cm_page_count = 3
        s._cm_start_time = 0.0
        await s._close_browser()
        mock_mgr.stop.assert_awaited_once()
        assert s._browser_mgr is None

    @pytest.mark.asyncio
    async def test_close_browser_logs_summary(self, tmp_path, caplog):
        import logging
        s = _make_scraper(tmp_path)
        mock_mgr, _ = _mock_browser_manager()
        s._browser_mgr = mock_mgr
        s._cm_page_count = 10
        s._cm_start_time = 0.0
        with caplog.at_level(logging.INFO, logger="scout"):
            await s._close_browser()
        assert any("scraped 10 pages" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_close_browser_no_log_if_zero_pages(self, tmp_path, caplog):
        import logging
        s = _make_scraper(tmp_path)
        mock_mgr, _ = _mock_browser_manager()
        s._browser_mgr = mock_mgr
        s._cm_page_count = 0
        s._cm_start_time = 0.0
        with caplog.at_level(logging.INFO, logger="scout"):
            await s._close_browser()
        assert not any("scraped" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_close_browser_swallows_exceptions(self, tmp_path):
        s = _make_scraper(tmp_path)
        mock_mgr, _ = _mock_browser_manager()
        mock_mgr.stop = AsyncMock(side_effect=RuntimeError("boom"))
        s._browser_mgr = mock_mgr
        s._cm_page_count = 1
        s._cm_start_time = 0.0
        # Should not raise
        await s._close_browser()
        assert s._browser_mgr is None


# ── In-process execution ─────────────────────────────────────────

class TestInProcessExecution:

    @pytest.mark.asyncio
    async def test_launches_browser_on_first_call(self, tmp_path):
        s = _make_scraper(tmp_path)
        _write_valid_script(s._script_path)

        mock_mgr, mock_page = _mock_browser_manager()

        with patch(
            "scout.runtime.environment.BrowserManager", return_value=mock_mgr
        ) as MockBM:
            # Set up scraper state as if context-managed
            s._context_managed = True
            s._cm_start_time = 0.0
            s._cm_page_count = 0

            # Load the function
            from scout.scraper import _load_script
            fn, _ = _load_script(s._script_path)
            s._cached_fn = fn

            rv_json = await s._run_in_process("https://example.com")

            MockBM.assert_called_once_with(headless=True)
            mock_mgr.start.assert_awaited_once()
            mock_mgr.new_page.assert_awaited_once()
            mock_page.goto.assert_awaited_once()
            mock_page.close.assert_awaited_once()

            data = json.loads(rv_json)
            assert data == [{"title": "Test"}]
            assert s._cm_page_count == 1

    @pytest.mark.asyncio
    async def test_reuses_browser_on_subsequent_calls(self, tmp_path):
        s = _make_scraper(tmp_path)
        _write_valid_script(s._script_path)

        mock_mgr, mock_page = _mock_browser_manager()
        s._browser_mgr = mock_mgr  # already launched
        s._context_managed = True
        s._cm_start_time = 0.0
        s._cm_page_count = 1

        from scout.scraper import _load_script
        fn, _ = _load_script(s._script_path)
        s._cached_fn = fn

        rv_json = await s._run_in_process("https://example.com")

        # Should NOT re-create the BrowserManager
        mock_mgr.start.assert_not_awaited()
        # But should create a new page
        mock_mgr.new_page.assert_awaited_once()
        assert s._cm_page_count == 2

    @pytest.mark.asyncio
    async def test_timeout_raises_script_timeout_error(self, tmp_path):
        s = _make_scraper(tmp_path, timeout=1)

        mock_mgr, mock_page = _mock_browser_manager()
        s._browser_mgr = mock_mgr
        s._context_managed = True
        s._cm_start_time = 0.0
        s._cm_page_count = 0

        # Create a function that hangs
        async def slow_scrape(page, url, checkpoint):
            await asyncio.sleep(100)

        s._cached_fn = slow_scrape

        with pytest.raises(ScoutScriptTimeoutError, match="1s timeout"):
            await s._run_in_process("https://example.com")

        # Page should still be closed
        mock_page.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_runtime_error_wrapped(self, tmp_path):
        s = _make_scraper(tmp_path)

        mock_mgr, mock_page = _mock_browser_manager()
        s._browser_mgr = mock_mgr
        s._context_managed = True
        s._cm_start_time = 0.0
        s._cm_page_count = 0

        async def crashing_scrape(page, url, checkpoint):
            raise ValueError("element not found")

        s._cached_fn = crashing_scrape

        with pytest.raises(ScoutScriptRuntimeError, match="element not found"):
            await s._run_in_process("https://example.com")

    @pytest.mark.asyncio
    async def test_non_serializable_uses_default_str(self, tmp_path):
        """default=str in json.dumps converts non-serializable to strings."""
        s = _make_scraper(tmp_path)

        mock_mgr, mock_page = _mock_browser_manager()
        s._browser_mgr = mock_mgr
        s._context_managed = True
        s._cm_start_time = 0.0
        s._cm_page_count = 0

        async def obj_return_scrape(page, url, checkpoint):
            return [{"title": object()}]

        s._cached_fn = obj_return_scrape

        # default=str converts objects to their repr, doesn't raise
        rv_json = await s._run_in_process("https://example.com")
        data = json.loads(rv_json)
        assert isinstance(data[0]["title"], str)

    @pytest.mark.asyncio
    async def test_page_closed_even_on_error(self, tmp_path):
        s = _make_scraper(tmp_path)

        mock_mgr, mock_page = _mock_browser_manager()
        s._browser_mgr = mock_mgr
        s._context_managed = True
        s._cm_start_time = 0.0
        s._cm_page_count = 0

        async def crashing_scrape(page, url, checkpoint):
            raise RuntimeError("crash")

        s._cached_fn = crashing_scrape

        with pytest.raises(ScoutScriptRuntimeError):
            await s._run_in_process("https://example.com")

        mock_page.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_headless_flag_passed_to_browser_manager(self, tmp_path):
        s = _make_scraper(tmp_path, headless=False)
        _write_valid_script(s._script_path)

        mock_mgr, mock_page = _mock_browser_manager()

        with patch(
            "scout.runtime.environment.BrowserManager", return_value=mock_mgr
        ) as MockBM:
            s._context_managed = True
            s._cm_start_time = 0.0
            s._cm_page_count = 0

            from scout.scraper import _load_script
            fn, _ = _load_script(s._script_path)
            s._cached_fn = fn

            await s._run_in_process("https://example.com")
            MockBM.assert_called_once_with(headless=False)


# ── Context-managed run (integration) ────────────────────────────

class TestContextManagedRun:
    """Test the full _run_cached branching for context-managed mode."""

    @pytest.mark.asyncio
    async def test_run_cached_uses_in_process_when_context_managed(
        self, tmp_path
    ):
        s = _make_scraper(tmp_path)
        _write_valid_script(s._script_path, [{"title": "Hello"}])

        mock_mgr, mock_page = _mock_browser_manager()

        with patch(
            "scout.runtime.environment.BrowserManager", return_value=mock_mgr
        ):
            s._context_managed = True
            s._cm_start_time = 0.0
            s._cm_page_count = 0

            result = await s._run_cached("https://example.com")

            assert result.data == [{"title": "Hello"}]
            assert result.cached is True
            mock_mgr.new_page.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_cached_uses_subprocess_when_not_context_managed(
        self, tmp_path
    ):
        s = _make_scraper(tmp_path)
        _write_valid_script(s._script_path)
        s._context_managed = False

        with patch.object(
            s, "_execute_function",
            new_callable=AsyncMock,
            return_value=(
                "", json.dumps([{"title": "Test"}]), "", 0,
            ),
        ):
            result = await s._run_cached("https://example.com")
            assert result.cached is True
            s._execute_function.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_first_run_logs_cached_script_message(
        self, tmp_path, caplog
    ):
        import logging
        s = _make_scraper(tmp_path)
        _write_valid_script(s._script_path)

        mock_mgr, mock_page = _mock_browser_manager()

        with patch(
            "scout.runtime.environment.BrowserManager", return_value=mock_mgr
        ):
            s._context_managed = True
            s._cm_start_time = 0.0
            s._cm_page_count = 0

            with caplog.at_level(logging.INFO, logger="scout"):
                await s._run_cached("https://example.com")

            messages = [r.getMessage() for r in caplog.records]
            assert any("Running cached script" in m for m in messages)
            assert any("Launching browser" in m for m in messages)

    @pytest.mark.asyncio
    async def test_subsequent_run_logs_scraping_url(
        self, tmp_path, caplog
    ):
        import logging
        s = _make_scraper(tmp_path)
        _write_valid_script(s._script_path)

        mock_mgr, mock_page = _mock_browser_manager()
        s._browser_mgr = mock_mgr  # already launched
        s._context_managed = True
        s._cm_start_time = 0.0
        s._cm_page_count = 1

        with caplog.at_level(logging.INFO, logger="scout"):
            await s._run_cached("https://example.com/page/2")

        messages = [r.getMessage() for r in caplog.records]
        assert any(
            "Scraping" in m and "example.com/page/2" in m
            for m in messages
        )


# ── Sync run() inside context manager ────────────────────────────

class TestSyncRunInContextManager:

    def test_run_dispatches_to_background_loop(self, tmp_path):
        s = _make_scraper(tmp_path)
        _write_valid_script(s._script_path)

        mock_mgr, mock_page = _mock_browser_manager()

        with patch(
            "scout.runtime.environment.BrowserManager", return_value=mock_mgr
        ):
            with s:
                result = s.run()

            assert result.data == [{"title": "Test"}]
            assert result.cached is True

    def test_multiple_runs_reuse_browser(self, tmp_path):
        s = _make_scraper(tmp_path)
        _write_valid_script(s._script_path)

        mock_mgr, mock_page = _mock_browser_manager()

        with patch(
            "scout.runtime.environment.BrowserManager", return_value=mock_mgr
        ):
            with s:
                r1 = s.run()
                r2 = s.run(url="https://example.com/page/2")

            # Browser started once
            mock_mgr.start.assert_awaited_once()
            # But two pages created
            assert mock_mgr.new_page.await_count == 2
            assert s._browser_mgr is None  # cleaned up

    def test_run_with_url_override(self, tmp_path):
        s = _make_scraper(tmp_path)
        _write_valid_script(s._script_path)

        mock_mgr, mock_page = _mock_browser_manager()

        with patch(
            "scout.runtime.environment.BrowserManager", return_value=mock_mgr
        ):
            with s:
                result = s.run(url="https://example.com/other")
                assert result.url == "https://example.com/other"


# ── close() method ───────────────────────────────────────────────

class TestCloseMethod:

    def test_close_without_context_manager_is_noop(self, tmp_path):
        s = _make_scraper(tmp_path)
        s.close()  # should not raise

    def test_close_after_context_manager(self, tmp_path):
        s = _make_scraper(tmp_path)
        s.__enter__()
        s.close()
        assert s._bg_loop is None
        assert s._bg_thread is None
        assert s._context_managed is False

    def test_close_idempotent(self, tmp_path):
        s = _make_scraper(tmp_path)
        s.__enter__()
        s.close()
        s.close()  # second call should not raise


# ── __del__ warning ──────────────────────────────────────────────

class TestDelWarning:

    def test_no_warning_when_no_browser(self, tmp_path):
        s = _make_scraper(tmp_path)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s.__del__()
            assert len(w) == 0

    def test_warning_when_browser_open(self, tmp_path):
        s = _make_scraper(tmp_path)
        mock_mgr, _ = _mock_browser_manager()
        s._browser_mgr = mock_mgr
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s.__del__()
            assert len(w) == 1
            assert issubclass(w[0].category, ResourceWarning)
            assert "garbage-collected" in str(w[0].message)

    def test_del_safe_on_partial_init(self):
        """__del__ should not crash even if __init__ failed partway."""
        s = object.__new__(Scraper)
        # Don't call __init__ — _browser_mgr not set
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s.__del__()
            assert len(w) == 0


# ── export() method ──────────────────────────────────────────────

class TestExport:

    def test_export_creates_standalone_file(self, tmp_path):
        s = _make_scraper(tmp_path)
        _write_valid_script(s._script_path)

        export_path = tmp_path / "debug" / "standalone.py"
        s.export(export_path)

        assert export_path.exists()
        content = export_path.read_text()
        assert "async def scrape" in content
        assert "asyncio.run" in content

    def test_export_file_is_valid_python(self, tmp_path):
        import ast
        s = _make_scraper(tmp_path)
        _write_valid_script(s._script_path)

        export_path = tmp_path / "standalone.py"
        s.export(export_path)

        content = export_path.read_text()
        ast.parse(content)  # should not raise

    def test_export_refuses_overwrite_by_default(self, tmp_path):
        s = _make_scraper(tmp_path)
        _write_valid_script(s._script_path)

        export_path = tmp_path / "standalone.py"
        export_path.write_text("existing", encoding="utf-8")

        with pytest.raises(ScoutError, match="already exists"):
            s.export(export_path)

    def test_export_allows_overwrite_when_flag_set(self, tmp_path):
        s = _make_scraper(tmp_path)
        _write_valid_script(s._script_path)

        export_path = tmp_path / "standalone.py"
        export_path.write_text("old content", encoding="utf-8")

        s.export(export_path, overwrite=True)

        content = export_path.read_text()
        assert "async def scrape" in content

    def test_export_requires_script_path(self):
        s = _make_scraper()  # no script=
        with pytest.raises(ScoutError, match="no script= path"):
            s.export("output.py")

    def test_export_requires_script_on_disk(self, tmp_path):
        s = _make_scraper(tmp_path)
        # Don't write the script file
        with pytest.raises(ScoutError, match="not found"):
            s.export(tmp_path / "output.py")

    def test_export_requires_py_extension(self, tmp_path):
        s = _make_scraper(tmp_path)
        _write_valid_script(s._script_path)
        with pytest.raises(ScoutError, match=".py file"):
            s.export(tmp_path / "output.txt")

    def test_export_creates_parent_directories(self, tmp_path):
        s = _make_scraper(tmp_path)
        _write_valid_script(s._script_path)

        deep_path = tmp_path / "a" / "b" / "c" / "standalone.py"
        s.export(deep_path)
        assert deep_path.exists()

    def test_export_logs_path(self, tmp_path, caplog):
        import logging
        s = _make_scraper(tmp_path)
        _write_valid_script(s._script_path)

        export_path = tmp_path / "standalone.py"
        with caplog.at_level(logging.INFO, logger="scout"):
            s.export(export_path)

        assert any("Exported" in r.getMessage() for r in caplog.records)


# ── In-memory caching without script= ────────────────────────────

class TestInMemoryCaching:
    """Verify that _run_generate caches function even without script=."""

    @pytest.mark.asyncio
    async def test_generation_caches_source(self, tmp_path):
        """After generation, _cached_source should be set."""
        s = _make_scraper()  # no script=

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.final_script = (
            "async def scrape(page, url, checkpoint):\n"
            "    return [{'title': 'Generated'}]\n"
        )
        mock_result.return_value = json.dumps([{"title": "Generated"}])
        mock_result.error = None

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        with (
            patch("scout.agent.loop.AgentLoop", return_value=mock_agent),
            patch("scout.agent.llm.LLMConfig"),
            patch.object(s, "_check_api_key"),
            patch.object(s, "_check_playwright"),
        ):
            import time
            result = await s._run_generate(
                "https://example.com", time.monotonic()
            )

        assert s._cached_source is not None
        assert "async def scrape" in s._cached_source
        assert s._cached_fn is not None
        assert callable(s._cached_fn)

    @pytest.mark.asyncio
    async def test_cached_fn_works_in_process(self, tmp_path):
        """Cached function from generation should be callable in-process."""
        s = _make_scraper()

        # Simulate what _run_generate does for scriptless caching
        source = (
            "async def scrape(page, url, checkpoint):\n"
            "    return [{'title': 'Cached'}]\n"
        )
        ns = {
            "__file__": "<scout-generated>",
            **_build_pre_import_namespace(),
        }
        exec(compile(source, "<scout-generated>", "exec"), ns)
        s._cached_fn = ns["scrape"]

        mock_mgr, mock_page = _mock_browser_manager()
        s._browser_mgr = mock_mgr
        s._context_managed = True
        s._cm_start_time = 0.0
        s._cm_page_count = 0

        rv_json = await s._run_in_process("https://example.com")
        data = json.loads(rv_json)
        assert data == [{"title": "Cached"}]


# ── Pre-imports in loaded functions ──────────────────────────────

class TestPreImportsInLoadedFunctions:
    """Agent-authored functions that use pre-imported modules
    should work when executed in-process."""

    @pytest.mark.asyncio
    async def test_function_using_json_module(self, tmp_path):
        """Function uses json.dumps — needs json in namespace."""
        script_path = tmp_path / "scraper.py"
        script_path.write_text(
            '"""\nScout Script\n\n'
            'url:           https://example.com\n'
            'task:          test\n'
            'generated:     2024-01-01T00:00:00.000000Z\n'
            'model:         test\n'
            'scout_version: test\n'
            '"""\n\n'
            'async def scrape(page, url, checkpoint):\n'
            '    data = json.dumps({"a": 1})\n'
            '    return [{"title": data}]\n',
            encoding="utf-8",
        )

        from scout.scraper import _load_script
        fn, _ = _load_script(script_path)

        mock_mgr, mock_page = _mock_browser_manager()

        s = Scraper(
            "https://example.com", "test", schema=[{"title": str}],
            script=str(script_path),
        )
        s._cached_fn = fn
        s._browser_mgr = mock_mgr
        s._context_managed = True
        s._cm_start_time = 0.0
        s._cm_page_count = 0

        rv_json = await s._run_in_process("https://example.com")
        data = json.loads(rv_json)
        assert data == [{"title": '{"a": 1}'}]

    @pytest.mark.asyncio
    async def test_function_using_re_module(self, tmp_path):
        """Function uses re.sub — needs re in namespace."""
        script_path = tmp_path / "scraper.py"
        script_path.write_text(
            '"""\nScout Script\n\n'
            'url:           https://example.com\n'
            'task:          test\n'
            'generated:     2024-01-01T00:00:00.000000Z\n'
            'model:         test\n'
            'scout_version: test\n'
            '"""\n\n'
            'async def scrape(page, url, checkpoint):\n'
            '    cleaned = re.sub(r"\\$", "", "$99")\n'
            '    return [{"title": cleaned}]\n',
            encoding="utf-8",
        )

        from scout.scraper import _load_script
        fn, _ = _load_script(script_path)

        mock_mgr, mock_page = _mock_browser_manager()

        s = Scraper(
            "https://example.com", "test", schema=[{"title": str}],
            script=str(script_path),
        )
        s._cached_fn = fn
        s._browser_mgr = mock_mgr
        s._context_managed = True
        s._cm_start_time = 0.0
        s._cm_page_count = 0

        rv_json = await s._run_in_process("https://example.com")
        data = json.loads(rv_json)
        assert data == [{"title": "99"}]


# ── Edge cases ───────────────────────────────────────────────────

class TestEdgeCases:

    def test_context_manager_without_any_run(self, tmp_path):
        """Enter and exit without calling run() — should not crash."""
        s = _make_scraper(tmp_path)
        with s:
            pass  # no run() calls

    @pytest.mark.asyncio
    async def test_async_context_manager_without_run(self, tmp_path):
        s = _make_scraper(tmp_path)
        async with s:
            pass

    @pytest.mark.asyncio
    async def test_regenerate_inside_context_manager(self, tmp_path):
        """regenerate=True inside context manager triggers generation path."""
        s = _make_scraper(tmp_path)
        _write_valid_script(s._script_path)

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.final_script = (
            "async def scrape(page, url, checkpoint):\n"
            "    return [{'title': 'Regenerated'}]\n"
        )
        mock_result.return_value = json.dumps([{"title": "Regenerated"}])

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        with patch("scout.agent.loop.AgentLoop", return_value=mock_agent), \
             patch("scout.agent.llm.LLMConfig"), \
             patch.object(s, "_check_api_key"), \
             patch.object(s, "_check_playwright"):
            async with s:
                result = await s.async_run(regenerate=True)
                assert result.cached is False

    @pytest.mark.asyncio
    async def test_validation_error_in_context_managed_run(self, tmp_path):
        """Schema validation failure should propagate correctly."""
        s = Scraper(
            "https://example.com", "test",
            schema=[{"title": str, "price": float}],
            script=str(tmp_path / "scraper.py"),
        )
        # Write a script that returns data missing the 'price' field
        _write_valid_script(
            s._script_path,
            [{"title": "Test"}],  # missing 'price'
        )

        mock_mgr, mock_page = _mock_browser_manager()

        with patch(
            "scout.runtime.environment.BrowserManager", return_value=mock_mgr
        ):
            s._context_managed = True
            s._cm_start_time = 0.0
            s._cm_page_count = 0

            with pytest.raises(ScoutValidationError):
                await s._run_cached("https://example.com")

    def test_exit_cleans_up_on_exception(self, tmp_path):
        """Browser is closed even when an exception propagates."""
        s = _make_scraper(tmp_path)
        _write_valid_script(s._script_path)

        mock_mgr, mock_page = _mock_browser_manager()

        with patch(
            "scout.runtime.environment.BrowserManager", return_value=mock_mgr
        ):
            try:
                with s:
                    s.run()
                    raise RuntimeError("user code error")
            except RuntimeError:
                pass

            # Browser should be cleaned up
            assert s._browser_mgr is None
            assert s._context_managed is False

    @pytest.mark.asyncio
    async def test_get_function_source_uses_cached_source(self):
        """_get_function_source falls back to _cached_source."""
        s = _make_scraper()  # no script=
        s._cached_source = "async def scrape(page, url, checkpoint): pass"
        assert s._get_function_source() == s._cached_source

    @pytest.mark.asyncio
    async def test_get_function_source_prefers_cached_source(self, tmp_path):
        """_cached_source takes precedence over disk."""
        s = _make_scraper(tmp_path)
        _write_valid_script(s._script_path)
        s._cached_source = "async def scrape(page, url, checkpoint): return 'cached'"
        assert s._get_function_source() == s._cached_source
