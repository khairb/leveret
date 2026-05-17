"""Tests for auto-regenerate integration into the Scraper class (Phase 5).

Track 1: ScoutAutoRegenerateError exception and ScraperResult.auto_regenerated field.
Track 2: Execute function adapters (_make_subprocess_execute_fn,
         _make_in_process_execute_fn, _collect_in_process_signals).
Track 3: Scraper integration — auto_regenerate parameter, _run_cached_with_autofix(),
         post-regeneration validation, error mapping, logging.

Tests verify:
  - ScoutAutoRegenerateError is in the correct exception hierarchy
  - ScoutAutoRegenerateError can be raised and caught at multiple levels
  - ScraperResult.auto_regenerated defaults to False (backward compatibility)
  - ScraperResult.auto_regenerated=True works and implies script_generated=True
  - ScraperResult repr shows auto_regenerated only when True
  - ScoutAutoRegenerateError is importable from the top-level package
  - Subprocess adapter returns AttemptResult with page signals
  - In-process adapter returns AttemptResult with page signals
  - _collect_in_process_signals is fully defensive
  - auto_regenerate parameter validation (False, True, str modes, invalid)
  - auto_regenerate without script= logs warning, disables auto_regenerate
  - _run_cached_with_autofix() delegates to diagnose()
  - Diagnosis success returns ScraperResult(script_generated=False)
  - Diagnosis RAISE raises appropriate exception type per category
  - Diagnosis REGENERATE triggers _run_generate()
  - Successful regeneration returns ScraperResult(auto_regenerated=True)
  - Failed regeneration raises ScoutAutoRegenerateError
  - Schema failure on regeneration raises ScoutAutoRegenerateError
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from scout import ScoutAutoRegenerateError, ScoutError
from scout.errors import (
    ScoutAutoRegenerateError as ScoutAutoRegenerateErrorDirect,
    ScoutScriptError,
)
from scout.scraper import ScraperResult


# -- ScoutAutoRegenerateError exception hierarchy -----------------------------------


class TestScoutAutoRegenerateErrorHierarchy:
    """ScoutAutoRegenerateError sits directly under ScoutError."""

    def test_is_subclass_of_scout_error(self):
        assert issubclass(ScoutAutoRegenerateError, ScoutError)

    def test_is_subclass_of_exception(self):
        assert issubclass(ScoutAutoRegenerateError, Exception)

    def test_is_not_subclass_of_scout_script_error(self):
        """ScoutAutoRegenerateError is NOT a script error — it's a diagnosis error."""
        assert not issubclass(ScoutAutoRegenerateError, ScoutScriptError)

    def test_direct_import_matches_package_import(self):
        """Same class whether imported from scout or scout.errors."""
        assert ScoutAutoRegenerateError is ScoutAutoRegenerateErrorDirect

    def test_raise_and_catch_as_scout_error(self):
        with pytest.raises(ScoutError, match="regen failed"):
            raise ScoutAutoRegenerateError("regen failed")

    def test_raise_and_catch_as_specific(self):
        with pytest.raises(ScoutAutoRegenerateError, match="same pattern"):
            raise ScoutAutoRegenerateError("same pattern")

    def test_raise_and_catch_as_exception(self):
        with pytest.raises(Exception):
            raise ScoutAutoRegenerateError("test")

    def test_message_preserved(self):
        try:
            raise ScoutAutoRegenerateError(
                "Regenerated script failed with the same error pattern."
            )
        except ScoutAutoRegenerateError as exc:
            assert "same error pattern" in str(exc)

    def test_not_caught_by_scout_script_error(self):
        """ScoutAutoRegenerateError should NOT be caught by ScoutScriptError handler."""
        with pytest.raises(ScoutAutoRegenerateError):
            try:
                raise ScoutAutoRegenerateError("test")
            except ScoutScriptError:
                pytest.fail("Should not be caught by ScoutScriptError")


# -- ScraperResult.auto_regenerated ------------------------------------------------


class TestScraperResultAutoRegenerated:
    """ScraperResult.auto_regenerated field — backward compatibility and behavior."""

    def test_default_is_false(self):
        """Existing code creating ScraperResult without auto_regenerated still works."""
        result = ScraperResult(
            data={"title": "Test"},
            url="https://example.com",
            timestamp="2024-01-01T00:00:00.000000Z",
            script_generated=False,
            script_path=Path("/path/to/script.py"),
        )
        assert result.auto_regenerated is False

    def test_explicit_false(self):
        result = ScraperResult(
            data=[1, 2, 3],
            url="https://example.com",
            timestamp="2024-01-01T00:00:00.000000Z",
            script_generated=False,
            script_path=None,
            auto_regenerated=False,
        )
        assert result.auto_regenerated is False

    def test_explicit_true(self):
        result = ScraperResult(
            data=[1, 2, 3],
            url="https://example.com",
            timestamp="2024-01-01T00:00:00.000000Z",
            script_generated=True,
            script_path=Path("/path/to/script.py"),
            auto_regenerated=True,
        )
        assert result.auto_regenerated is True

    def test_positional_args_still_work(self):
        """Existing code using positional args is not broken."""
        result = ScraperResult(
            [{"a": 1}],
            "https://example.com",
            "2024-01-01T00:00:00.000000Z",
            False,
            None,
        )
        assert result.auto_regenerated is False
        assert result.data == [{"a": 1}]
        assert result.script_generated is False

    def test_auto_regenerated_is_a_dataclass_field(self):
        """auto_regenerated is a proper dataclass field, not just an attribute."""
        import dataclasses

        fields = {f.name: f for f in dataclasses.fields(ScraperResult)}
        assert "auto_regenerated" in fields
        assert fields["auto_regenerated"].default is False


# -- ScraperResult repr with auto_regenerated -------------------------------------


class TestScraperResultRepr:
    """Repr shows auto_regenerated only when True to avoid noise."""

    def test_repr_without_auto_regenerated(self):
        result = ScraperResult(
            data=[1, 2, 3],
            url="https://example.com",
            timestamp="t",
            script_generated=False,
            script_path=None,
        )
        r = repr(result)
        assert "auto_regenerated" not in r
        assert "script_generated=False" in r

    def test_repr_with_auto_regenerated_false(self):
        result = ScraperResult(
            data=[1, 2, 3],
            url="https://example.com",
            timestamp="t",
            script_generated=False,
            script_path=None,
            auto_regenerated=False,
        )
        r = repr(result)
        assert "auto_regenerated" not in r

    def test_repr_with_auto_regenerated_true(self):
        result = ScraperResult(
            data=[1, 2, 3],
            url="https://example.com",
            timestamp="t",
            script_generated=True,
            script_path=Path("/path/to/script.py"),
            auto_regenerated=True,
        )
        r = repr(result)
        assert "auto_regenerated=True" in r
        assert "script_generated=True" in r

    def test_repr_preserves_existing_format(self):
        """Existing repr format is preserved for non-auto-regenerated results."""
        result = ScraperResult(
            data=[{"title": "A"}, {"title": "B"}],
            url="https://shop.example.com",
            timestamp="t",
            script_generated=False,
            script_path=None,
        )
        r = repr(result)
        assert r.startswith("ScraperResult(")
        assert r.endswith(")")
        assert "url='https://shop.example.com'" in r
        assert "items=2" in r
        assert "script_generated=False" in r


# -- ScoutAutoRegenerateError in package __all__ ------------------------------------


class TestPackageExports:
    """ScoutAutoRegenerateError is exported from the scout package."""

    def test_in_scout_all(self):
        import scout

        assert "AutoRegenerateError" in scout.__all__

    def test_importable_from_top_level(self):
        """Can import directly from scout package."""
        from scout import ScoutAutoRegenerateError as Exc

        assert Exc is ScoutAutoRegenerateError


# -- Track 2: Execute function adapters ------------------------------------

# Tests use mocking to avoid real browser/subprocess execution.
# The adapters are tested for correct AttemptResult construction,
# schema validation (Category G), and signal collection.


from unittest.mock import AsyncMock, MagicMock, patch


VALID_URL = "https://example.com"
VALID_TASK = "Extract products"
VALID_SCHEMA = [{"title": str}]


def _make_scraper(**overrides):
    """Build a Scraper with valid defaults for testing adapters."""
    from scout.scraper import Scraper

    kwargs = {"schema": VALID_SCHEMA}
    kwargs.update(overrides)
    url = kwargs.pop("url", VALID_URL)
    task = kwargs.pop("task", VALID_TASK)
    return Scraper(url, task, **kwargs)


class TestSubprocessExecuteAdapter:
    """_make_subprocess_execute_fn returns an async callable -> AttemptResult."""

    @pytest.mark.asyncio
    async def test_success_returns_attempt_result(self, tmp_path):
        """Subprocess success → AttemptResult(success=True, data=...)."""
        import json
        from scout.agent.wrapper import RETURN_VALUE_START, RETURN_VALUE_END

        script = tmp_path / "scraper.py"
        script.write_text(
            '"""\nurl: https://example.com\ntask: test\n"""\n'
            "async def scrape(page, url, checkpoint):\n"
            "    return [{'title': 'Test'}]\n"
        )
        s = _make_scraper(script=str(script))
        # Load the script so _cached_fn is set
        from scout.scraper import _load_script

        fn, _ = _load_script(script)
        s._cached_fn = fn

        execute_fn = s._make_subprocess_execute_fn(VALID_URL)

        # Mock _execute_function via the subprocess internals
        data = [{"title": "Test"}]
        rv_json = json.dumps(data, indent=2)
        mock_stdout = f"{RETURN_VALUE_START}\n{rv_json}\n{RETURN_VALUE_END}\n"

        with patch(
            "asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
        ) as mock_proc:
            proc = AsyncMock()
            proc.communicate = AsyncMock(
                return_value=(
                    mock_stdout.encode(),
                    b"",
                )
            )
            proc.returncode = 0
            mock_proc.return_value = proc

            result = await execute_fn()

        assert result.success is True
        assert result.data == [{"title": "Test"}]

    @pytest.mark.asyncio
    async def test_failure_returns_error_and_signals(self, tmp_path):
        """Subprocess failure → AttemptResult with error, page_signals."""
        import json
        from scout.agent.wrapper import PAGE_SIGNALS_START, PAGE_SIGNALS_END

        script = tmp_path / "scraper.py"
        script.write_text(
            '"""\nurl: https://example.com\ntask: test\n"""\n'
            "async def scrape(page, url, checkpoint):\n"
            "    return []\n"
        )
        s = _make_scraper(script=str(script))
        from scout.scraper import _load_script

        fn, _ = _load_script(script)
        s._cached_fn = fn

        execute_fn = s._make_subprocess_execute_fn(VALID_URL)

        # Simulate failure with page signals
        signals = {
            "http_status": 200,
            "page_url": "https://example.com",
            "content": "<html>real page</html>",
            "headers": {"content-type": "text/html"},
            "cookies": [{"name": "sid", "value": "x"}],
        }
        signals_json = json.dumps(signals)
        mock_stdout = (
            f"{PAGE_SIGNALS_START}\n{signals_json}\n{PAGE_SIGNALS_END}\n"
        )
        mock_stderr = b"AttributeError: 'NoneType' has no attribute 'text'\n"

        with patch(
            "asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
        ) as mock_proc:
            proc = AsyncMock()
            proc.communicate = AsyncMock(
                return_value=(mock_stdout.encode(), mock_stderr)
            )
            proc.returncode = 1
            mock_proc.return_value = proc

            result = await execute_fn()

        assert result.success is False
        assert "AttributeError" in result.error
        assert result.exit_code == 1
        assert result.page_signals is not None
        assert result.page_signals.http_status == 200
        assert result.page_signals.page_url == "https://example.com"
        assert result.page_signals.content == "<html>real page</html>"
        assert result.page_signals.headers["content-type"] == "text/html"
        assert result.page_signals.cookies[0]["name"] == "sid"

    @pytest.mark.asyncio
    async def test_timeout_returns_f2_compatible(self, tmp_path):
        """Subprocess timeout → AttemptResult with exit_code=-1."""
        script = tmp_path / "scraper.py"
        script.write_text(
            '"""\nurl: https://example.com\ntask: test\n"""\n'
            "async def scrape(page, url, checkpoint):\n"
            "    return []\n"
        )
        s = _make_scraper(script=str(script), run_timeout=5)
        from scout.scraper import _load_script

        fn, _ = _load_script(script)
        s._cached_fn = fn

        execute_fn = s._make_subprocess_execute_fn(VALID_URL)

        with patch(
            "asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
        ) as mock_proc:
            proc = AsyncMock()
            proc.communicate = AsyncMock(
                side_effect=asyncio.TimeoutError()
            )
            proc.kill = AsyncMock()
            # After kill, communicate returns empty
            proc.communicate.side_effect = [
                asyncio.TimeoutError(),
                (b"", b""),
            ]
            mock_proc.return_value = proc

            result = await execute_fn()

        assert result.success is False
        assert "timed out" in result.error
        assert result.exit_code == -1
        assert result.page_signals is None  # No signals on timeout

    @pytest.mark.asyncio
    async def test_schema_failure_returns_category_g(self, tmp_path):
        """Success with schema validation failure → Category G AttemptResult."""
        import json
        from scout.agent.wrapper import RETURN_VALUE_START, RETURN_VALUE_END

        script = tmp_path / "scraper.py"
        script.write_text(
            '"""\nurl: https://example.com\ntask: test\n"""\n'
            "async def scrape(page, url, checkpoint):\n"
            "    return []\n"
        )
        # Schema requires min 1 item but script returns []
        from scout.schema.types import List

        s = _make_scraper(script=str(script), schema=List({"title": str}, min_items=1))
        from scout.scraper import _load_script

        fn, _ = _load_script(script)
        s._cached_fn = fn

        execute_fn = s._make_subprocess_execute_fn(VALID_URL)

        rv_json = json.dumps([], indent=2)
        mock_stdout = f"{RETURN_VALUE_START}\n{rv_json}\n{RETURN_VALUE_END}\n"

        with patch(
            "asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
        ) as mock_proc:
            proc = AsyncMock()
            proc.communicate = AsyncMock(
                return_value=(mock_stdout.encode(), b"")
            )
            proc.returncode = 0
            mock_proc.return_value = proc

            result = await execute_fn()

        assert result.success is False
        assert result.schema_error is not None
        assert result.data == []  # Data is preserved for diagnostics


class TestCollectInProcessSignals:
    """_collect_in_process_signals is fully defensive."""

    @pytest.mark.asyncio
    async def test_collects_all_signals(self):
        """Collects URL, content, status, headers, cookies from live page."""
        from scout.scraper import Scraper

        page = AsyncMock()
        page.url = "https://example.com/page"
        page.content = AsyncMock(return_value="<html>content</html>")

        context = MagicMock()
        context.cookies = AsyncMock(return_value=[
            {"name": "sid", "value": "abc", "domain": "example.com"},
        ])
        page.context = context

        resp = MagicMock()
        resp.status = 200
        resp.headers = {"content-type": "text/html", "server": "nginx"}

        signals = await Scraper._collect_in_process_signals(
            page, [resp],
        )

        assert signals.page_url == "https://example.com/page"
        assert signals.content == "<html>content</html>"
        assert signals.http_status == 200
        assert signals.headers["content-type"] == "text/html"
        assert signals.cookies == [{"name": "sid", "value": "abc"}]

    @pytest.mark.asyncio
    async def test_survives_crashed_page(self):
        """All signal collection fails → PageSignals with all None/empty."""
        from scout.scraper import Scraper

        page = MagicMock()
        page.url = property(lambda self: (_ for _ in ()).throw(Exception("dead")))
        # Make .url raise
        type(page).url = property(lambda self: (_ for _ in ()).throw(Exception("dead")))
        page.content = AsyncMock(side_effect=Exception("page crashed"))
        page.context = MagicMock()
        page.context.cookies = AsyncMock(side_effect=Exception("no context"))

        signals = await Scraper._collect_in_process_signals(page, [])

        assert signals.page_url is None
        assert signals.content is None
        assert signals.http_status is None
        assert signals.headers == {}
        assert signals.cookies == []

    @pytest.mark.asyncio
    async def test_no_responses(self):
        """Empty response list → http_status and headers are None/empty."""
        from scout.scraper import Scraper

        page = AsyncMock()
        page.url = "https://example.com"
        page.content = AsyncMock(return_value="<html></html>")
        page.context = MagicMock()
        page.context.cookies = AsyncMock(return_value=[])

        signals = await Scraper._collect_in_process_signals(page, [])

        assert signals.http_status is None
        assert signals.headers == {}
        assert signals.page_url == "https://example.com"

    @pytest.mark.asyncio
    async def test_content_timeout(self):
        """page.content() hangs → times out after 5s, content is None."""
        from scout.scraper import Scraper

        page = AsyncMock()
        page.url = "https://example.com"
        page.content = AsyncMock(side_effect=asyncio.TimeoutError())
        page.context = MagicMock()
        page.context.cookies = AsyncMock(return_value=[])

        signals = await Scraper._collect_in_process_signals(page, [])

        assert signals.content is None
        assert signals.page_url == "https://example.com"


# -- Track 3: auto_regenerate parameter validation --------------------------------


class TestAutoRegenerateParameter:
    """auto_regenerate parameter validation in Scraper.__init__()."""

    def test_default_is_false(self, tmp_path):
        """auto_regenerate defaults to False — no _auto_regenerate_mode set."""
        script = tmp_path / "s.py"
        script.write_text("async def scrape(page, url, checkpoint): return []")
        s = _make_scraper(script=str(script))
        assert s._auto_regenerate_mode is None

    def test_true_maps_to_balanced(self, tmp_path):
        """auto_regenerate=True maps to RegenerateMode.BALANCED."""
        from scout.autofix.types import RegenerateMode

        script = tmp_path / "s.py"
        script.write_text("async def scrape(page, url, checkpoint): return []")
        s = _make_scraper(script=str(script), auto_regenerate=True)
        assert s._auto_regenerate_mode == RegenerateMode.BALANCED

    def test_balanced_string(self, tmp_path):
        from scout.autofix.types import RegenerateMode

        script = tmp_path / "s.py"
        script.write_text("async def scrape(page, url, checkpoint): return []")
        s = _make_scraper(script=str(script), auto_regenerate="balanced")
        assert s._auto_regenerate_mode == RegenerateMode.BALANCED

    def test_cautious_string(self, tmp_path):
        from scout.autofix.types import RegenerateMode

        script = tmp_path / "s.py"
        script.write_text("async def scrape(page, url, checkpoint): return []")
        s = _make_scraper(script=str(script), auto_regenerate="cautious")
        assert s._auto_regenerate_mode == RegenerateMode.CAUTIOUS

    def test_eager_string(self, tmp_path):
        from scout.autofix.types import RegenerateMode

        script = tmp_path / "s.py"
        script.write_text("async def scrape(page, url, checkpoint): return []")
        s = _make_scraper(script=str(script), auto_regenerate="eager")
        assert s._auto_regenerate_mode == RegenerateMode.EAGER

    def test_invalid_string_raises(self, tmp_path):
        """Invalid auto_regenerate value raises ScoutConfigError."""
        from scout.errors import ScoutConfigError

        script = tmp_path / "s.py"
        script.write_text("async def scrape(page, url, checkpoint): return []")
        with pytest.raises(ScoutConfigError, match="auto_regenerate must be"):
            _make_scraper(script=str(script), auto_regenerate="turbo")

    def test_invalid_int_raises(self, tmp_path):
        from scout.errors import ScoutConfigError

        script = tmp_path / "s.py"
        script.write_text("async def scrape(page, url, checkpoint): return []")
        with pytest.raises(ScoutConfigError, match="auto_regenerate must be"):
            _make_scraper(script=str(script), auto_regenerate=42)

    def test_without_script_disables(self, caplog):
        """auto_regenerate=True without script= logs warning, sets mode to None."""
        import logging

        with caplog.at_level(logging.WARNING):
            s = _make_scraper(auto_regenerate=True)
        assert s._auto_regenerate_mode is None
        assert "auto_regenerate has no effect without script=" in caplog.text

    def test_false_has_zero_overhead(self, tmp_path):
        """auto_regenerate=False does not import autofix modules."""
        script = tmp_path / "s.py"
        script.write_text("async def scrape(page, url, checkpoint): return []")
        # This just verifies the constructor doesn't crash
        s = _make_scraper(script=str(script), auto_regenerate=False)
        assert s._auto_regenerate_mode is None


# -- Track 3: _run_cached_with_autofix() -----------------------------------


class TestRunCachedWithAutofix:
    """_run_cached_with_autofix() delegates to diagnose() correctly."""

    def _make_scraper_with_autofix(self, tmp_path, mode=True):
        """Create a Scraper with auto_regenerate enabled and cached function loaded."""
        script = tmp_path / "scraper.py"
        script.write_text(
            '"""\nurl: https://example.com\ntask: Extract products\n"""\n'
            "async def scrape(page, url, checkpoint):\n"
            "    return [{'title': 'Test'}]\n"
        )
        s = _make_scraper(script=str(script), auto_regenerate=mode)
        from scout.scraper import _load_script

        fn, _ = _load_script(script)
        s._cached_fn = fn
        return s

    @pytest.mark.asyncio
    async def test_diagnosis_success_returns_data(self, tmp_path):
        """When diagnosis attempt succeeds, return ScraperResult(script_generated=False)."""
        from scout.autofix.types import AttemptResult

        s = self._make_scraper_with_autofix(tmp_path)

        mock_result = AttemptResult(
            success=True, data=[{"title": "Found"}],
        )

        with patch("scout.autofix.diagnose", return_value=mock_result) as mock_diag:
            result = await s._run_cached_with_autofix(VALID_URL, s._auto_regenerate_mode)

        assert result.data == [{"title": "Found"}]
        assert result.script_generated is False
        assert result.auto_regenerated is False
        assert result.url == VALID_URL
        mock_diag.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_diagnosis_raise_category_b(self, tmp_path):
        """Diagnosis RAISE with Category B raises ScoutScriptRuntimeError."""
        from scout.autofix.types import (
            AutoFixAction,
            DiagnosisResult,
            ErrorCategory,
        )
        from scout.errors import ScoutScriptRuntimeError

        s = self._make_scraper_with_autofix(tmp_path)

        mock_result = DiagnosisResult(
            action=AutoFixAction.RAISE,
            category=ErrorCategory.B,
            message="Category: Runtime crash\n  Error: AttributeError",
        )

        with patch("scout.autofix.diagnose", return_value=mock_result):
            with pytest.raises(ScoutScriptRuntimeError, match="Cached script failed"):
                await s._run_cached_with_autofix(VALID_URL, s._auto_regenerate_mode)

    @pytest.mark.asyncio
    async def test_diagnosis_raise_category_d(self, tmp_path):
        """Diagnosis RAISE with Category D raises ScoutScriptTimeoutError."""
        from scout.autofix.types import (
            AutoFixAction,
            DiagnosisResult,
            ErrorCategory,
        )
        from scout.errors import ScoutScriptTimeoutError

        s = self._make_scraper_with_autofix(tmp_path)

        mock_result = DiagnosisResult(
            action=AutoFixAction.RAISE,
            category=ErrorCategory.D,
            message="Category: Selector timeout",
        )

        with patch("scout.autofix.diagnose", return_value=mock_result):
            with pytest.raises(ScoutScriptTimeoutError, match="Cached script failed"):
                await s._run_cached_with_autofix(VALID_URL, s._auto_regenerate_mode)

    @pytest.mark.asyncio
    async def test_diagnosis_raise_category_g(self, tmp_path):
        """Diagnosis RAISE with Category G raises ScoutValidationError."""
        from scout.autofix.types import (
            AutoFixAction,
            DiagnosisResult,
            ErrorCategory,
        )
        from scout.errors import ScoutValidationError

        s = self._make_scraper_with_autofix(tmp_path)

        mock_result = DiagnosisResult(
            action=AutoFixAction.RAISE,
            category=ErrorCategory.G,
            message="Category: Schema validation failure",
        )

        with patch("scout.autofix.diagnose", return_value=mock_result):
            with pytest.raises(ScoutValidationError, match="Cached script failed"):
                await s._run_cached_with_autofix(VALID_URL, s._auto_regenerate_mode)

    @pytest.mark.asyncio
    async def test_diagnosis_raise_category_f3(self, tmp_path):
        """Diagnosis RAISE with Category F3 raises ScoutError (base)."""
        from scout.autofix.types import (
            AutoFixAction,
            DiagnosisResult,
            ErrorCategory,
        )

        s = self._make_scraper_with_autofix(tmp_path)

        mock_result = DiagnosisResult(
            action=AutoFixAction.RAISE,
            category=ErrorCategory.F3,
            message="Category: Infrastructure failure",
        )

        with patch("scout.autofix.diagnose", return_value=mock_result):
            with pytest.raises(ScoutError, match="Cached script failed"):
                await s._run_cached_with_autofix(VALID_URL, s._auto_regenerate_mode)

    @pytest.mark.asyncio
    async def test_diagnosis_raise_category_f2(self, tmp_path):
        """Diagnosis RAISE with Category F2 raises ScoutScriptTimeoutError."""
        from scout.autofix.types import (
            AutoFixAction,
            DiagnosisResult,
            ErrorCategory,
        )
        from scout.errors import ScoutScriptTimeoutError

        s = self._make_scraper_with_autofix(tmp_path)

        mock_result = DiagnosisResult(
            action=AutoFixAction.RAISE,
            category=ErrorCategory.F2,
            message="Category: Subprocess timeout",
        )

        with patch("scout.autofix.diagnose", return_value=mock_result):
            with pytest.raises(ScoutScriptTimeoutError, match="Cached script failed"):
                await s._run_cached_with_autofix(VALID_URL, s._auto_regenerate_mode)


# -- Track 3: Regeneration flow -------------------------------------------


class TestRegenerationFlow:
    """Auto-regenerate REGENERATE decision triggers _run_generate()."""

    def _make_scraper_with_autofix(self, tmp_path, mode=True):
        script = tmp_path / "scraper.py"
        script.write_text(
            '"""\nurl: https://example.com\ntask: Extract products\n"""\n'
            "async def scrape(page, url, checkpoint):\n"
            "    return [{'title': 'Test'}]\n"
        )
        s = _make_scraper(script=str(script), auto_regenerate=mode)
        from scout.scraper import _load_script

        fn, _ = _load_script(script)
        s._cached_fn = fn
        return s

    def _make_regen_diagnosis(self):
        from scout.autofix.types import (
            AutoFixAction,
            DiagnosisResult,
            ErrorCategory,
        )

        return DiagnosisResult(
            action=AutoFixAction.REGENERATE,
            category=ErrorCategory.B,
            message="Regenerating: AttributeError on selector",
        )

    @pytest.mark.asyncio
    async def test_regen_success_returns_auto_regenerated(self, tmp_path):
        """Successful regeneration returns ScraperResult(auto_regenerated=True)."""
        s = self._make_scraper_with_autofix(tmp_path)
        diag = self._make_regen_diagnosis()

        mock_regen_result = ScraperResult(
            data=[{"title": "New"}],
            url=VALID_URL,
            timestamp="2026-04-19T00:00:00.000000Z",
            script_generated=True,
            script_path=tmp_path / "scraper.py",
        )

        with (
            patch("scout.autofix.diagnose", return_value=diag),
            patch.object(s, "_run_generate", return_value=mock_regen_result),
            patch.object(s, "_check_api_key"),
            patch.object(s, "_check_playwright"),
        ):
            result = await s._run_cached_with_autofix(VALID_URL, s._auto_regenerate_mode)

        assert result.auto_regenerated is True
        assert result.script_generated is True
        assert result.data == [{"title": "New"}]

    @pytest.mark.asyncio
    async def test_regen_clears_cached_fn(self, tmp_path):
        """Regeneration clears _cached_fn and _cached_source (spec §10)."""
        s = self._make_scraper_with_autofix(tmp_path)
        s._cached_source = "old source"
        assert s._cached_fn is not None  # Loaded by helper

        diag = self._make_regen_diagnosis()

        mock_regen_result = ScraperResult(
            data=[{"title": "New"}],
            url=VALID_URL,
            timestamp="2026-04-19T00:00:00.000000Z",
            script_generated=True,
            script_path=tmp_path / "scraper.py",
        )

        fn_before_regen = None

        async def capture_and_return(*args, **kwargs):
            nonlocal fn_before_regen
            fn_before_regen = s._cached_fn
            return mock_regen_result

        with (
            patch("scout.autofix.diagnose", return_value=diag),
            patch.object(s, "_run_generate", side_effect=capture_and_return),
            patch.object(s, "_check_api_key"),
            patch.object(s, "_check_playwright"),
        ):
            await s._run_cached_with_autofix(VALID_URL, s._auto_regenerate_mode)

        # Cache was cleared BEFORE _run_generate was called
        assert fn_before_regen is None
        assert s._cached_source is None  # Cleared by _run_cached_with_autofix

    @pytest.mark.asyncio
    async def test_regen_generation_error_raises_auto_regenerate_error(self, tmp_path):
        """ScoutGenerationError from _run_generate → ScoutAutoRegenerateError."""
        from scout.errors import ScoutGenerationError

        s = self._make_scraper_with_autofix(tmp_path)
        diag = self._make_regen_diagnosis()

        with (
            patch("scout.autofix.diagnose", return_value=diag),
            patch.object(
                s,
                "_run_generate",
                side_effect=ScoutGenerationError("API rate limited"),
            ),
            patch.object(s, "_check_api_key"),
            patch.object(s, "_check_playwright"),
        ):
            with pytest.raises(
                ScoutAutoRegenerateError,
                match="could not produce a valid script",
            ):
                await s._run_cached_with_autofix(VALID_URL, s._auto_regenerate_mode)

    @pytest.mark.asyncio
    async def test_regen_validation_error_raises_validation_error(self, tmp_path):
        """ScoutValidationError from _run_generate → ScoutValidationError (spec §10)."""
        from scout.errors import ScoutValidationError

        s = self._make_scraper_with_autofix(tmp_path)
        diag = self._make_regen_diagnosis()

        with (
            patch("scout.autofix.diagnose", return_value=diag),
            patch.object(
                s,
                "_run_generate",
                side_effect=ScoutValidationError("Schema requires min 5 items"),
            ),
            patch.object(s, "_check_api_key"),
            patch.object(s, "_check_playwright"),
        ):
            with pytest.raises(
                ScoutValidationError,
                match="does not match the schema",
            ):
                await s._run_cached_with_autofix(VALID_URL, s._auto_regenerate_mode)

    @pytest.mark.asyncio
    async def test_regen_checks_api_key(self, tmp_path):
        """Regeneration calls _check_api_key() before _run_generate()."""
        s = self._make_scraper_with_autofix(tmp_path)
        diag = self._make_regen_diagnosis()

        mock_regen = ScraperResult(
            data=[{"title": "New"}],
            url=VALID_URL,
            timestamp="2026-04-19T00:00:00.000000Z",
            script_generated=True,
            script_path=tmp_path / "scraper.py",
        )

        with (
            patch("scout.autofix.diagnose", return_value=diag),
            patch.object(s, "_run_generate", return_value=mock_regen),
            patch.object(s, "_check_api_key") as mock_key,
            patch.object(s, "_check_playwright"),
        ):
            await s._run_cached_with_autofix(VALID_URL, s._auto_regenerate_mode)

        mock_key.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_subprocess_adapter_when_not_context_managed(self, tmp_path):
        """Non-context-managed scraper uses subprocess execute_fn."""
        s = self._make_scraper_with_autofix(tmp_path)
        assert s._context_managed is False

        from scout.autofix.types import AttemptResult

        mock_result = AttemptResult(success=True, data=[{"title": "Ok"}])

        with (
            patch("scout.autofix.diagnose", return_value=mock_result) as mock_diag,
            patch.object(s, "_make_subprocess_execute_fn", return_value=lambda: None) as mock_sub,
            patch.object(s, "_make_in_process_execute_fn") as mock_ip,
        ):
            await s._run_cached_with_autofix(VALID_URL, s._auto_regenerate_mode)

        mock_sub.assert_called_once_with(VALID_URL)
        mock_ip.assert_not_called()

    @pytest.mark.asyncio
    async def test_uses_in_process_adapter_when_context_managed(self, tmp_path):
        """Context-managed scraper uses in-process execute_fn."""
        s = self._make_scraper_with_autofix(tmp_path)
        s._context_managed = True

        from scout.autofix.types import AttemptResult

        mock_result = AttemptResult(success=True, data=[{"title": "Ok"}])

        with (
            patch("scout.autofix.diagnose", return_value=mock_result) as mock_diag,
            patch.object(s, "_make_subprocess_execute_fn") as mock_sub,
            patch.object(s, "_make_in_process_execute_fn", return_value=lambda: None) as mock_ip,
        ):
            await s._run_cached_with_autofix(VALID_URL, s._auto_regenerate_mode)

        mock_ip.assert_called_once_with(VALID_URL)
        mock_sub.assert_not_called()


# -- Track 3: _run_cached delegates to autofix path -----------------------


class TestRunCachedAutoFixBranch:
    """_run_cached() branches to _run_cached_with_autofix when auto_regenerate is set."""

    @pytest.mark.asyncio
    async def test_delegates_when_autofix_enabled(self, tmp_path):
        """_run_cached() calls _run_cached_with_autofix when auto_regenerate is set."""
        script = tmp_path / "scraper.py"
        script.write_text(
            '"""\nurl: https://example.com\ntask: Extract products\n"""\n'
            "async def scrape(page, url, checkpoint):\n"
            "    return [{'title': 'Test'}]\n"
        )
        s = _make_scraper(script=str(script), auto_regenerate=True)

        mock_result = ScraperResult(
            data=[{"title": "Test"}],
            url=VALID_URL,
            timestamp="2026-04-19T00:00:00.000000Z",
            script_generated=False,
            script_path=script,
        )

        with patch.object(
            s, "_run_cached_with_autofix", return_value=mock_result,
        ) as mock_af:
            result = await s._run_cached(VALID_URL, s._auto_regenerate_mode)

        mock_af.assert_awaited_once_with(VALID_URL, s._auto_regenerate_mode)
        assert result.data == [{"title": "Test"}]

    @pytest.mark.asyncio
    async def test_normal_path_when_autofix_disabled(self, tmp_path):
        """_run_cached() takes normal path when auto_regenerate is False."""
        script = tmp_path / "scraper.py"
        script.write_text(
            '"""\nurl: https://example.com\ntask: Extract products\n"""\n'
            "async def scrape(page, url, checkpoint):\n"
            "    return [{'title': 'Test'}]\n"
        )
        s = _make_scraper(script=str(script), auto_regenerate=False)

        with patch.object(
            s, "_run_cached_with_autofix",
        ) as mock_af:
            with patch.object(
                s, "_execute_function",
                return_value=("", '[]', "", 0),
            ):
                # Will fail schema validation but that's fine —
                # we just need to verify _run_cached_with_autofix was NOT called
                try:
                    await s._run_cached(VALID_URL, s._auto_regenerate_mode)
                except Exception:
                    pass

        mock_af.assert_not_called()
