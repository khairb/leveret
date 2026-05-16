"""Tests for Scraper.run() and async_run() — Track 3.

Tests cover:
- URL resolution and validation
- Prerequisite checks (API key, Playwright)
- Cached execution path (load + execute + validate)
- Generation path (AgentLoop integration)
- Error mapping (LLM API errors → ScoutGenerationError)
- Event loop detection (Jupyter/async environments)
- Console output (log messages)
- Schema validation of return values
"""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scout.errors import (
    ScoutConfigError,
    ScoutError,
    ScoutGenerationError,
    ScoutSchemaError,
    ScoutScriptLoadError,
    ScoutScriptRuntimeError,
    ScoutScriptTimeoutError,
    ScoutValidationError,
)
from scout.scraper import (
    Scraper,
    ScraperResult,
    _load_script,
    _save_script,
)


# ── Helpers ──────────────────────────────────────────────────────

def _make_scraper(
    url: str = "https://example.com/products",
    task: str = "Extract products",
    schema: Any = None,
    script: str | Path | None = None,
    **kwargs: Any,
) -> Scraper:
    """Build a Scraper with sensible defaults."""
    if schema is None:
        schema = [{"name": str, "price": float}]
    return Scraper(url, task, schema=schema, script=script, **kwargs)


def _write_valid_script(path: Path, url: str = "https://example.com/products") -> None:
    """Write a minimal valid scraping script to disk."""
    code = textwrap.dedent("""\
        async def scrape(page, url, checkpoint):
            return [{"name": "Widget", "price": 9.99}]
    """)
    _save_script(code, path, url, "Extract products", "claude-haiku-4-5")


def _mock_execute_result(
    stdout: str = "",
    return_value_json: str | None = None,
    stderr: str = "",
    returncode: int = 0,
) -> tuple[str, str | None, str, int]:
    """Build a mock _execute_function return value."""
    return (stdout, return_value_json, stderr, returncode)


# ── Positional argument guard ────────────────────────────────────

class TestPositionalArgGuard:

    def test_run_positional_url_raises(self):
        """run('https://...') gives a helpful error, not a TypeError."""
        s = _make_scraper()
        with pytest.raises(ScoutError, match="does not accept positional"):
            s.run("https://example.com/other")

    def test_run_positional_suggests_keyword(self):
        s = _make_scraper()
        with pytest.raises(ScoutError, match=r"url="):
            s.run("https://example.com/other")

    def test_async_run_positional_raises(self):
        s = _make_scraper()
        with pytest.raises(ScoutError, match="does not accept positional"):
            asyncio.run(s.async_run("https://example.com/other"))


# ── URL resolution ───────────────────────────────────────────────

class TestURLResolution:

    def test_no_override_uses_constructor_url(self):
        s = _make_scraper(url="https://example.com/products")
        assert s._resolve_url(None) == "https://example.com/products"

    def test_override_url_returned(self):
        s = _make_scraper()
        assert s._resolve_url("https://example.com/other") == "https://example.com/other"

    def test_override_empty_string_raises(self):
        s = _make_scraper()
        with pytest.raises(ScoutError, match="url must be a valid"):
            s._resolve_url("")

    def test_override_no_scheme_raises(self):
        s = _make_scraper()
        with pytest.raises(ScoutError, match="url must start with https://"):
            s._resolve_url("not-a-url")

    def test_override_no_hostname_raises(self):
        s = _make_scraper()
        with pytest.raises(ScoutError, match="url must include a hostname"):
            s._resolve_url("http://")

    def test_override_valid_https(self):
        s = _make_scraper()
        result = s._resolve_url("https://other.com/page")
        assert result == "https://other.com/page"


# ── Prerequisite checks ─────────────────────────────────────────

class TestPrerequisiteChecks:

    def test_api_key_from_constructor(self):
        s = _make_scraper(api_key="sk-ant-test")
        s._check_api_key()  # should not raise

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        s = _make_scraper()
        s._check_api_key()  # should not raise

    def test_api_key_missing_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        s = _make_scraper()
        with pytest.raises(ScoutConfigError, match="API key not found"):
            s._check_api_key()

    def test_api_key_error_is_actionable(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        s = _make_scraper()
        with pytest.raises(ScoutConfigError, match="ANTHROPIC_API_KEY"):
            s._check_api_key()

    def test_playwright_installed(self):
        s = _make_scraper()
        # patchright should be importable in the test env
        # If not installed, this test itself is expected to fail
        try:
            import patchright  # noqa: F401
            s._check_playwright()  # should not raise
        except ImportError:
            pytest.skip("patchright not installed")

    def test_playwright_not_installed(self):
        s = _make_scraper()
        with patch.dict("sys.modules", {"patchright": None}):
            with pytest.raises(ScoutConfigError, match="Patchright is not installed"):
                s._check_playwright()


# ── Event loop detection ─────────────────────────────────────────

class TestEventLoopDetection:

    def test_run_inside_event_loop_raises(self):
        """run() inside a running loop raises with Jupyter hint."""
        s = _make_scraper()

        async def _inner():
            with pytest.raises(ScoutError, match="running event loop"):
                s.run()

        asyncio.run(_inner())

    def test_run_error_suggests_async_run(self):
        s = _make_scraper()

        async def _inner():
            with pytest.raises(ScoutError, match="await scraper.async_run"):
                s.run()

        asyncio.run(_inner())


# ── Cached execution path ───────────────────────────────────────

class TestCachedExecution:

    @pytest.fixture
    def script_path(self, tmp_path):
        """Create a valid script file."""
        path = tmp_path / "scraper.py"
        _write_valid_script(path)
        return path

    def test_cached_script_loaded_and_executed(self, script_path):
        """A valid script on disk is loaded, executed, and returns data."""
        s = _make_scraper(script=str(script_path))

        mock_result = _mock_execute_result(
            return_value_json=json.dumps([{"name": "Widget", "price": 9.99}]),
        )

        with patch.object(s, "_execute_function", new_callable=AsyncMock, return_value=mock_result):
            result = asyncio.run(s.async_run())

        assert isinstance(result, ScraperResult)
        assert result.data == [{"name": "Widget", "price": 9.99}]
        assert result.cached is True
        assert result.url == "https://example.com/products"
        assert result.script_path == str(script_path)

    def test_cached_script_validates_return_value(self, script_path):
        """Schema validation is applied to cached script output."""
        s = _make_scraper(script=str(script_path))

        # Return data that doesn't match schema (price is string)
        mock_result = _mock_execute_result(
            return_value_json=json.dumps([{"name": "Widget", "price": "free"}]),
        )

        with patch.object(s, "_execute_function", new_callable=AsyncMock, return_value=mock_result):
            with pytest.raises(ScoutValidationError, match="does not match the schema"):
                asyncio.run(s.async_run())

    def test_cached_script_timeout(self, script_path):
        """Script timeout raises ScoutScriptTimeoutError."""
        s = _make_scraper(script=str(script_path))

        mock_result = _mock_execute_result(
            stderr="Function timed out after 600 seconds",
            returncode=-1,
        )

        with patch.object(s, "_execute_function", new_callable=AsyncMock, return_value=mock_result):
            with pytest.raises(ScoutScriptTimeoutError, match="timeout"):
                asyncio.run(s.async_run())

    def test_cached_script_runtime_error(self, script_path):
        """Script crash raises ScoutScriptRuntimeError."""
        s = _make_scraper(script=str(script_path))

        mock_result = _mock_execute_result(
            stderr="Traceback (most recent call last):\n  ...\nTimeoutError: Selector not found",
            returncode=1,
        )

        with patch.object(s, "_execute_function", new_callable=AsyncMock, return_value=mock_result):
            with pytest.raises(ScoutScriptRuntimeError, match="crashed during execution"):
                asyncio.run(s.async_run())

    def test_runtime_error_suggests_auto_fix(self, script_path):
        """Runtime error message suggests scraper.regenerate()."""
        s = _make_scraper(script=str(script_path))

        mock_result = _mock_execute_result(
            stderr="Some error", returncode=1,
        )

        with patch.object(s, "_execute_function", new_callable=AsyncMock, return_value=mock_result):
            with pytest.raises(ScoutScriptRuntimeError, match="scraper.regenerate"):
                asyncio.run(s.async_run())

    def test_domain_mismatch_on_cached_script(self, tmp_path):
        """Loading a cached script for a different domain raises."""
        path = tmp_path / "scraper.py"
        _write_valid_script(path, url="https://other-site.com/page")

        s = _make_scraper(
            url="https://example.com/products",
            script=str(path),
        )

        with pytest.raises(ScoutConfigError, match="different site"):
            asyncio.run(s.async_run())

    def test_task_mismatch_logs_warning(self, script_path, caplog):
        """Task mismatch logs a warning but continues."""
        s = Scraper(
            "https://example.com/products",
            "Different task description",
            schema=[{"name": str, "price": float}],
            script=str(script_path),
        )

        mock_result = _mock_execute_result(
            return_value_json=json.dumps([{"name": "W", "price": 1.0}]),
        )

        import logging
        with caplog.at_level(logging.WARNING, logger="scout"):
            with patch.object(s, "_execute_function", new_callable=AsyncMock, return_value=mock_result):
                asyncio.run(s.async_run())

        assert any("Task description has changed" in r.message for r in caplog.records)

    def test_in_memory_cache_reused(self, script_path):
        """Second run uses the in-memory cached function."""
        s = _make_scraper(script=str(script_path))

        mock_result = _mock_execute_result(
            return_value_json=json.dumps([{"name": "W", "price": 1.0}]),
        )

        with patch.object(s, "_execute_function", new_callable=AsyncMock, return_value=mock_result):
            asyncio.run(s.async_run())
            # Second run — should reuse _cached_fn
            assert s._cached_fn is not None
            asyncio.run(s.async_run())

    def test_auto_fix_always_clears_cache(self, script_path, monkeypatch):
        """auto_fix="always" discards cached function and regenerates."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        s = _make_scraper(script=str(script_path))

        # Pre-populate the cache
        mock_result = _mock_execute_result(
            return_value_json=json.dumps([{"name": "W", "price": 1.0}]),
        )
        with patch.object(s, "_execute_function", new_callable=AsyncMock, return_value=mock_result):
            asyncio.run(s.async_run())
        assert s._cached_fn is not None

        # auto_fix="always" should go to generation path (which we mock)
        with patch.object(s, "_run_generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = ScraperResult(
                data=[{"name": "New", "price": 2.0}],
                url="https://example.com/products",
                timestamp="2024-01-01T00:00:00.000000Z",
                cached=False,
                script_path=str(script_path),
            )
            with patch.object(s, "_check_api_key"):
                with patch.object(s, "_check_playwright"):
                    result = asyncio.run(s.async_run(auto_fix="always"))

        assert result.cached is False
        assert s._cached_fn is None  # was cleared before generation

    def test_url_override_with_cached_script(self, script_path):
        """url= override works with cached scripts."""
        s = _make_scraper(script=str(script_path))

        mock_result = _mock_execute_result(
            return_value_json=json.dumps([{"name": "W", "price": 1.0}]),
        )

        with patch.object(s, "_execute_function", new_callable=AsyncMock, return_value=mock_result) as mock_exec:
            result = asyncio.run(s.async_run(url="https://example.com/other"))

        assert result.url == "https://example.com/other"

    def test_url_override_different_domain_raises(self, script_path):
        """url= override with different domain raises on cached path."""
        s = _make_scraper(script=str(script_path))

        with pytest.raises(ScoutConfigError, match="different site"):
            asyncio.run(s.async_run(url="https://other-site.com/page"))


# ── Generation path ──────────────────────────────────────────────

class TestGeneration:

    def test_generation_called_when_no_script(self, monkeypatch, tmp_path):
        """When no script exists, generation runs."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        s = _make_scraper()

        with patch.object(s, "_run_generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = ScraperResult(
                data=[{"name": "W", "price": 1.0}],
                url="https://example.com/products",
                timestamp="2024-01-01T00:00:00.000000Z",
                cached=False,
                script_path=None,
            )
            with patch.object(s, "_check_playwright"):
                result = asyncio.run(s.async_run())

        assert result.cached is False
        mock_gen.assert_awaited_once()

    def test_generation_checks_api_key_first(self, monkeypatch):
        """API key is checked before Playwright."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        s = _make_scraper()

        with pytest.raises(ScoutError, match="API key not found"):
            asyncio.run(s.async_run())

    def test_generation_checks_playwright_second(self, monkeypatch):
        """Playwright is checked after API key."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        s = _make_scraper()

        with patch.dict("sys.modules", {"patchright": None}):
            with pytest.raises(ScoutError, match="Patchright is not installed"):
                asyncio.run(s.async_run())

    def test_generation_no_prereqs_for_cached(self, tmp_path):
        """Cached scripts don't check API key or Playwright."""
        path = tmp_path / "scraper.py"
        _write_valid_script(path)
        s = _make_scraper(script=str(path))

        mock_result = _mock_execute_result(
            return_value_json=json.dumps([{"name": "W", "price": 1.0}]),
        )

        # Remove API key from env
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(s, "_execute_function", new_callable=AsyncMock, return_value=mock_result):
                result = asyncio.run(s.async_run())

        assert result.cached is True

    def test_generation_failure_raises_generation_error(self, monkeypatch):
        """AgentLoop failure maps to ScoutGenerationError."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        s = _make_scraper()

        mock_agent_result = MagicMock()
        mock_agent_result.success = False
        mock_agent_result.error = "Agent ran out of budget"

        mock_loop_cls = MagicMock()
        mock_loop_instance = MagicMock()
        mock_loop_instance.run = AsyncMock(return_value=mock_agent_result)
        mock_loop_cls.return_value = mock_loop_instance

        with patch.object(s, "_check_playwright"):
            with patch("scout.scraper.AgentLoop", mock_loop_cls) if False else \
                 patch.object(s, "_run_generate", new_callable=AsyncMock) as mock_gen:
                mock_gen.side_effect = ScoutGenerationError("Agent ran out of budget")
                with pytest.raises(ScoutGenerationError, match="Agent ran out of budget"):
                    asyncio.run(s.async_run())

    def test_generation_saves_script_when_path_set(self, monkeypatch, tmp_path):
        """Successful generation saves to script= path."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        script_path = tmp_path / "scrapers" / "test.py"
        s = _make_scraper(script=str(script_path))

        with patch.object(s, "_check_playwright"):
            with patch.object(s, "_run_generate", new_callable=AsyncMock) as mock_gen:
                mock_gen.return_value = ScraperResult(
                    data=[{"name": "Widget", "price": 9.99}],
                    url="https://example.com/products",
                    timestamp="2024-01-01T00:00:00.000000Z",
                    cached=False,
                    script_path=str(script_path),
                )
                result = asyncio.run(s.async_run())

        assert result.cached is False
        mock_gen.assert_awaited_once()


# ── Return value validation ──────────────────────────────────────

class TestReturnValueValidation:

    def test_valid_data_passes(self):
        s = _make_scraper()
        data = s._validate_return_value(
            json.dumps([{"name": "W", "price": 1.0}])
        )
        assert data == [{"name": "W", "price": 1.0}]

    def test_none_json_fails(self):
        s = _make_scraper()
        with pytest.raises(ScoutValidationError):
            s._validate_return_value(None)

    def test_null_json_fails(self):
        s = _make_scraper()
        with pytest.raises(ScoutValidationError):
            s._validate_return_value("null")

    def test_wrong_type_fails(self):
        s = _make_scraper()
        with pytest.raises(ScoutValidationError, match="does not match"):
            s._validate_return_value(json.dumps("not a list"))

    def test_wrong_field_type_fails(self):
        s = _make_scraper()
        with pytest.raises(ScoutValidationError):
            s._validate_return_value(json.dumps([{"name": 42, "price": 1.0}]))

    def test_missing_field_fails(self):
        s = _make_scraper()
        with pytest.raises(ScoutValidationError):
            s._validate_return_value(json.dumps([{"name": "W"}]))

    def test_malformed_json_fails(self):
        s = _make_scraper()
        with pytest.raises(ScoutValidationError):
            s._validate_return_value("{{{not json")

    def test_error_message_suggests_regenerate(self):
        s = _make_scraper()
        with pytest.raises(ScoutValidationError, match="scraper.regenerate"):
            s._validate_return_value(json.dumps("wrong"))

    def test_complex_schema_validation(self):
        from scout.schema.types import Field, List
        s = _make_scraper(schema=List({
            "title": Field(str, min_length=1),
            "price": Field(float, min=0),
        }, min=2))
        data = [
            {"title": "A", "price": 1.0},
            {"title": "B", "price": 2.0},
        ]
        result = s._validate_return_value(json.dumps(data))
        assert result == data

    def test_complex_schema_too_few_items(self):
        from scout.schema.types import Field, List
        s = _make_scraper(schema=List({"title": str}, min=10))
        with pytest.raises(ScoutValidationError):
            s._validate_return_value(json.dumps([{"title": "Only one"}]))


# ── LLM error mapping ───────────────────────────────────────────

class TestErrorMapping:

    def _make_scraper_for_mapping(self):
        return _make_scraper()

    def test_rate_limit_error(self):
        s = self._make_scraper_for_mapping()
        from pydantic_ai.exceptions import ModelHTTPError
        exc = ModelHTTPError(status_code=429, model_name="test", body="Rate limited")
        result = s._map_generation_error(exc)
        assert isinstance(result, ScoutGenerationError)
        assert "rate limit" in str(result).lower()

    def test_auth_error(self):
        s = self._make_scraper_for_mapping()
        from pydantic_ai.exceptions import ModelHTTPError
        exc = ModelHTTPError(status_code=401, model_name="test", body="Unauthorized")
        result = s._map_generation_error(exc)
        assert isinstance(result, ScoutGenerationError)
        assert "API key" in str(result)

    def test_connection_error(self):
        s = self._make_scraper_for_mapping()
        exc = ConnectionError("Connection error.")
        result = s._map_generation_error(exc)
        assert isinstance(result, ScoutGenerationError)
        assert "network" in str(result).lower()

    def test_generic_exception(self):
        s = self._make_scraper_for_mapping()
        exc = RuntimeError("Something went wrong")
        result = s._map_generation_error(exc)
        assert isinstance(result, ScoutGenerationError)
        assert "Something went wrong" in str(result)


# ── Console output ───────────────────────────────────────────────

class TestConsoleOutput:

    def test_generation_first_run_logs(self, monkeypatch, caplog):
        """First generation logs appropriate messages."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        s = _make_scraper(script=str(Path("/tmp/nonexistent_scraper.py")))

        import logging
        with caplog.at_level(logging.INFO, logger="scout"):
            with patch.object(s, "_check_playwright"):
                with patch.object(s, "_run_generate", new_callable=AsyncMock) as mock_gen:
                    mock_gen.return_value = ScraperResult(
                        data=[{"name": "W", "price": 1.0}],
                        url="https://example.com/products",
                        timestamp="2024-01-01T00:00:00.000000Z",
                        cached=False,
                        script_path="/tmp/nonexistent_scraper.py",
                    )
                    asyncio.run(s.async_run())

        messages = [r.message for r in caplog.records]
        assert any("No cached script found" in m for m in messages)
        assert any("First run calls the AI model API" in m for m in messages)

    def test_cached_run_logs_single_line(self, tmp_path, caplog):
        """Cached run logs just one line."""
        path = tmp_path / "scraper.py"
        _write_valid_script(path)
        s = _make_scraper(script=str(path))

        mock_result = _mock_execute_result(
            return_value_json=json.dumps([{"name": "W", "price": 1.0}]),
        )

        import logging
        with caplog.at_level(logging.INFO, logger="scout"):
            with patch.object(s, "_execute_function", new_callable=AsyncMock, return_value=mock_result):
                asyncio.run(s.async_run())

        messages = [r.getMessage() for r in caplog.records]
        assert any("Running cached script" in m for m in messages)

    def test_regenerate_logs_regenerating(self, tmp_path, monkeypatch, caplog):
        """regenerate=True logs regeneration message."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        path = tmp_path / "scraper.py"
        _write_valid_script(path)
        s = _make_scraper(script=str(path))

        import logging
        with caplog.at_level(logging.INFO, logger="scout"):
            with patch.object(s, "_check_playwright"):
                with patch.object(s, "_run_generate", new_callable=AsyncMock) as mock_gen:
                    mock_gen.return_value = ScraperResult(
                        data=[{"name": "W", "price": 1.0}],
                        url="https://example.com/products",
                        timestamp="2024-01-01T00:00:00.000000Z",
                        cached=False,
                        script_path=str(path),
                    )
                    asyncio.run(s.async_run(auto_fix="always"))

        messages = [r.getMessage() for r in caplog.records]
        assert any("Regenerating" in m for m in messages)

    def test_no_script_param_logs_hint(self, monkeypatch, caplog):
        """No script= logs a caching hint."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        s = _make_scraper()  # no script=

        import logging
        with caplog.at_level(logging.INFO, logger="scout"):
            with patch.object(s, "_check_playwright"):
                with patch.object(s, "_run_generate", new_callable=AsyncMock) as mock_gen:
                    mock_gen.return_value = ScraperResult(
                        data=[{"name": "W", "price": 1.0}],
                        url="https://example.com/products",
                        timestamp="2024-01-01T00:00:00.000000Z",
                        cached=False,
                        script_path=None,
                    )
                    asyncio.run(s.async_run())

        messages = [r.getMessage() for r in caplog.records]
        assert any("Generating scraping function" in m for m in messages)


# ── ScraperResult ────────────────────────────────────────────────

class TestScraperResultFromRun:

    def test_result_fields_cached(self, tmp_path):
        """ScraperResult has correct fields for cached run."""
        path = tmp_path / "scraper.py"
        _write_valid_script(path)
        s = _make_scraper(script=str(path))

        mock_result = _mock_execute_result(
            return_value_json=json.dumps([{"name": "W", "price": 1.0}]),
        )

        with patch.object(s, "_execute_function", new_callable=AsyncMock, return_value=mock_result):
            result = asyncio.run(s.async_run())

        assert result.data == [{"name": "W", "price": 1.0}]
        assert result.url == "https://example.com/products"
        assert result.timestamp.endswith("Z")
        assert result.cached is True
        assert result.script_path == str(path)

    def test_result_timestamp_is_iso8601(self, tmp_path):
        """Timestamp is ISO 8601 UTC."""
        path = tmp_path / "scraper.py"
        _write_valid_script(path)
        s = _make_scraper(script=str(path))

        mock_result = _mock_execute_result(
            return_value_json=json.dumps([{"name": "W", "price": 1.0}]),
        )

        with patch.object(s, "_execute_function", new_callable=AsyncMock, return_value=mock_result):
            result = asyncio.run(s.async_run())

        from datetime import datetime, timezone
        # Should parse without error
        dt = datetime.fromisoformat(result.timestamp.replace("Z", "+00:00"))
        assert dt.tzinfo is not None


# ── run() wrapper ────────────────────────────────────────────────

class TestRunWrapper:

    def test_run_calls_async_run(self, tmp_path):
        """run() delegates to async_run()."""
        path = tmp_path / "scraper.py"
        _write_valid_script(path)
        s = _make_scraper(script=str(path))

        mock_result = _mock_execute_result(
            return_value_json=json.dumps([{"name": "W", "price": 1.0}]),
        )

        with patch.object(s, "_execute_function", new_callable=AsyncMock, return_value=mock_result):
            result = s.run()

        assert isinstance(result, ScraperResult)
        assert result.cached is True

    def test_run_passes_url_and_auto_fix(self, tmp_path, monkeypatch):
        """run() passes url= and auto_fix= to async_run()."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        path = tmp_path / "scraper.py"
        _write_valid_script(path)
        s = _make_scraper(script=str(path))

        with patch.object(s, "async_run", new_callable=AsyncMock) as mock_async:
            mock_async.return_value = ScraperResult(
                data=[{"name": "W", "price": 1.0}],
                url="https://example.com/other",
                timestamp="2024-01-01T00:00:00.000000Z",
                cached=False,
                script_path=str(path),
            )
            s.run(url="https://example.com/other", auto_fix="always")

        mock_async.assert_awaited_once_with(
            url="https://example.com/other", auto_fix="always", inputs=None,
        )


# ── Edge cases ───────────────────────────────────────────────────

class TestEdgeCases:

    def test_no_script_no_cache_goes_to_generation(self, monkeypatch):
        """No script= and no cache triggers generation."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        s = _make_scraper()

        with patch.object(s, "_check_playwright"):
            with patch.object(s, "_run_generate", new_callable=AsyncMock) as mock_gen:
                mock_gen.return_value = ScraperResult(
                    data=[{"name": "W", "price": 1.0}],
                    url="https://example.com/products",
                    timestamp="2024-01-01T00:00:00.000000Z",
                    cached=False,
                    script_path=None,
                )
                result = asyncio.run(s.async_run())

        assert result.cached is False

    def test_script_path_not_existing_goes_to_generation(self, monkeypatch, tmp_path):
        """script= pointing to nonexistent file triggers generation."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        s = _make_scraper(script=str(tmp_path / "does_not_exist.py"))

        with patch.object(s, "_check_playwright"):
            with patch.object(s, "_run_generate", new_callable=AsyncMock) as mock_gen:
                mock_gen.return_value = ScraperResult(
                    data=[{"name": "W", "price": 1.0}],
                    url="https://example.com/products",
                    timestamp="2024-01-01T00:00:00.000000Z",
                    cached=False,
                    script_path=str(tmp_path / "does_not_exist.py"),
                )
                result = asyncio.run(s.async_run())

        assert result.cached is False

    def test_empty_return_value_none_json(self, tmp_path):
        """Function that returns nothing (None) fails validation."""
        path = tmp_path / "scraper.py"
        _write_valid_script(path)
        s = _make_scraper(script=str(path))

        mock_result = _mock_execute_result(
            return_value_json="null",  # explicit None return
        )

        with patch.object(s, "_execute_function", new_callable=AsyncMock, return_value=mock_result):
            with pytest.raises(ScoutValidationError):
                asyncio.run(s.async_run())

    def test_no_return_markers(self, tmp_path):
        """Function that crashes (no markers) fails validation."""
        path = tmp_path / "scraper.py"
        _write_valid_script(path)
        s = _make_scraper(script=str(path))

        # returncode=0 but no return value — function didn't return properly
        mock_result = _mock_execute_result(
            stdout="Some output",
            return_value_json=None,
            returncode=0,
        )

        with patch.object(s, "_execute_function", new_callable=AsyncMock, return_value=mock_result):
            with pytest.raises(ScoutValidationError):
                asyncio.run(s.async_run())

    def test_multiple_sequential_runs(self, tmp_path):
        """Multiple run() calls work correctly."""
        path = tmp_path / "scraper.py"
        _write_valid_script(path)
        s = _make_scraper(script=str(path))

        for i in range(3):
            mock_result = _mock_execute_result(
                return_value_json=json.dumps([{"name": f"P{i}", "price": float(i)}]),
            )
            with patch.object(s, "_execute_function", new_callable=AsyncMock, return_value=mock_result):
                result = s.run()
            assert result.data == [{"name": f"P{i}", "price": float(i)}]


# ═══════════════════════════════════════════════════════════════════════════
# GenerationError.is_transient
# ═══════════════════════════════════════════════════════════════════════════

class TestGenerationErrorTransient:
    """GenerationError exposes is_transient for programmatic retry decisions."""

    def test_transient_on_rate_limit(self):
        from scout.errors import GenerationError
        e = GenerationError("rate limited", status_code=429)
        assert e.is_transient is True

    def test_transient_on_server_error(self):
        from scout.errors import GenerationError
        e = GenerationError("server error", status_code=500)
        assert e.is_transient is True
        e2 = GenerationError("server error", status_code=503)
        assert e2.is_transient is True

    def test_not_transient_on_auth_error(self):
        from scout.errors import GenerationError
        e = GenerationError("auth failed", status_code=401)
        assert e.is_transient is False

    def test_not_transient_without_status_code(self):
        from scout.errors import GenerationError
        e = GenerationError("agent failed")
        assert e.is_transient is False
        assert e.status_code is None

    def test_status_code_preserved(self):
        from scout.errors import GenerationError
        e = GenerationError("error", status_code=429)
        assert e.status_code == 429
        assert str(e) == "error"


# ═══════════════════════════════════════════════════════════════════════════
# Script backup on overwrite
# ═══════════════════════════════════════════════════════════════════════════

class TestScriptBackup:
    """_save_script creates a .bak backup before overwriting."""

    def test_backup_created_on_overwrite(self, tmp_path):
        from scout.scraper import _save_script
        path = tmp_path / "scraper.py"

        # First write
        _save_script(
            "async def scrape(page, start_url, checkpoint):\n    return []",
            path, "https://example.com", "task", "model",
        )
        original_content = path.read_text()

        # Second write (overwrite)
        _save_script(
            "async def scrape(page, start_url, checkpoint):\n    return [1]",
            path, "https://example.com", "task", "model",
        )

        bak = tmp_path / "scraper.py.bak"
        assert bak.exists()
        assert bak.read_text() == original_content

    def test_no_backup_on_first_write(self, tmp_path):
        from scout.scraper import _save_script
        path = tmp_path / "scraper.py"

        _save_script(
            "async def scrape(page, start_url, checkpoint):\n    return []",
            path, "https://example.com", "task", "model",
        )

        bak = tmp_path / "scraper.py.bak"
        assert not bak.exists()
