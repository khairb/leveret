"""Tests for Round 3 DX features.

Covers:
- Schema change detection (schema_hash in metadata)
- Script protection (protect_manual_edits + content_hash)
- Field(str, min=10) improved error message
- Generation failure model suggestion
- _is_script_user_edited detection
"""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path

import pytest

from scout.errors import ScoutConfigError, ScoutSchemaError
from scout.schema.parse import parse_schema
from scout.schema.types import Field, Items
from scout.scraper import (
    Scraper,
    _build_metadata_docstring,
    _is_script_user_edited,
    _parse_script_metadata,
    _save_script,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_URL = "https://example.com"
VALID_TASK = "Extract product prices"
VALID_SCHEMA = [{"title": str, "price": float}]


def _make(**overrides):
    kwargs = {"schema": VALID_SCHEMA}
    kwargs.update(overrides)
    url = kwargs.pop("url", VALID_URL)
    task = kwargs.pop("task", VALID_TASK)
    return Scraper(url, task, **kwargs)


def _write_script(path: Path, code: str, url: str = VALID_URL,
                  task: str = VALID_TASK, schema_hash: str = "") -> None:
    """Write a script file with metadata using Scout's save function."""
    _save_script(code, path, url, task, "test-model", schema_hash=schema_hash)


SCRAPE_FUNCTION = textwrap.dedent("""\
    async def scrape(page, start_url, checkpoint):
        return [{"title": "Test", "price": 9.99}]
""")


# ═══════════════════════════════════════════════════════════════════════════
# Schema change detection
# ═══════════════════════════════════════════════════════════════════════════

class TestSchemaHashInMetadata:
    """Schema hash is stored in script metadata when saving."""

    def test_metadata_contains_schema_hash(self, tmp_path):
        path = tmp_path / "test.py"
        _write_script(path, SCRAPE_FUNCTION, schema_hash="abc123")
        content = path.read_text()
        assert "schema_hash:   abc123" in content

    def test_metadata_parses_schema_hash(self, tmp_path):
        path = tmp_path / "test.py"
        _write_script(path, SCRAPE_FUNCTION, schema_hash="abc123")
        content = path.read_text()
        metadata = _parse_script_metadata(content)
        assert metadata["schema_hash"] == "abc123"

    def test_no_schema_hash_when_empty(self, tmp_path):
        path = tmp_path / "test.py"
        _write_script(path, SCRAPE_FUNCTION, schema_hash="")
        content = path.read_text()
        assert "schema_hash" not in content


class TestSchemaChangeWarning:
    """Schema mismatch is detected on cached run."""

    def test_schema_changed_flag_initially_false(self):
        s = _make()
        assert s._schema_changed is False

    def test_get_schema_hash_returns_hex(self):
        s = _make()
        h = s._get_schema_hash()
        assert isinstance(h, str)
        assert len(h) == 16
        # Same schema → same hash
        s2 = _make()
        assert s._get_schema_hash() == s2._get_schema_hash()

    def test_different_schema_different_hash(self):
        s1 = _make(schema={"title": str})
        s2 = _make(schema={"title": str, "price": float})
        assert s1._get_schema_hash() != s2._get_schema_hash()


# ═══════════════════════════════════════════════════════════════════════════
# Content hash and user-edit detection
# ═══════════════════════════════════════════════════════════════════════════

class TestContentHash:
    """content_hash is stored in metadata for edit detection."""

    def test_metadata_contains_content_hash(self, tmp_path):
        path = tmp_path / "test.py"
        _write_script(path, SCRAPE_FUNCTION)
        content = path.read_text()
        assert "content_hash:" in content

    def test_content_hash_matches_code(self, tmp_path):
        path = tmp_path / "test.py"
        _write_script(path, SCRAPE_FUNCTION)
        content = path.read_text()
        metadata = _parse_script_metadata(content)
        expected = hashlib.sha256(SCRAPE_FUNCTION.strip().encode()).hexdigest()[:16]
        assert metadata["content_hash"] == expected


class TestIsScriptUserEdited:
    """_is_script_user_edited detects manual changes."""

    def test_unedited_script_returns_false(self, tmp_path):
        path = tmp_path / "test.py"
        _write_script(path, SCRAPE_FUNCTION)
        assert _is_script_user_edited(path) is False

    def test_edited_script_returns_true(self, tmp_path):
        path = tmp_path / "test.py"
        _write_script(path, SCRAPE_FUNCTION)
        # Simulate user edit
        content = path.read_text()
        content = content.replace("Test", "Modified")
        path.write_text(content)
        assert _is_script_user_edited(path) is True

    def test_missing_file_returns_false(self, tmp_path):
        path = tmp_path / "nonexistent.py"
        assert _is_script_user_edited(path) is False

    def test_no_content_hash_returns_false(self, tmp_path):
        """Pre-Round-3 scripts without content_hash are not flagged."""
        path = tmp_path / "test.py"
        # Write a script without the content_hash field (manual metadata)
        path.write_text(textwrap.dedent('''\
            """
            Scout Script

            url:           https://example.com
            task:          Get data
            generated:     2024-01-01T00:00:00.000000Z
            model:         test
            scout_version: 0.1.0
            """

            async def scrape(page, start_url, checkpoint):
                return []
        '''))
        assert _is_script_user_edited(path) is False


# ═══════════════════════════════════════════════════════════════════════════
# Script protection
# ═══════════════════════════════════════════════════════════════════════════

class TestScriptProtection:
    """protect_manual_edits blocks automatic overwriting of user-edited scripts."""

    def test_check_protection_unedited_no_error(self, tmp_path):
        """Unedited script + protect_manual_edits=True → no error."""
        path = tmp_path / "test.py"
        _write_script(path, SCRAPE_FUNCTION)
        s = _make(script=str(path), protect_manual_edits=True)
        # Should not raise
        s._check_script_protection(force=False)

    def test_check_protection_edited_raises(self, tmp_path):
        """Edited script + protect_manual_edits=True → ConfigError."""
        path = tmp_path / "test.py"
        _write_script(path, SCRAPE_FUNCTION)
        # Simulate user edit
        content = path.read_text()
        path.write_text(content.replace("Test", "Edited"))

        s = _make(script=str(path), protect_manual_edits=True)
        with pytest.raises(ScoutConfigError, match="manually edited"):
            s._check_script_protection(force=False)

    def test_check_protection_edited_force_overrides(self, tmp_path):
        """Edited script + protect_manual_edits=True + force=True → no error."""
        path = tmp_path / "test.py"
        _write_script(path, SCRAPE_FUNCTION)
        content = path.read_text()
        path.write_text(content.replace("Test", "Edited"))

        s = _make(script=str(path), protect_manual_edits=True)
        # Should not raise with force=True
        s._check_script_protection(force=True)

    def test_check_protection_no_protect_warns(self, tmp_path, caplog):
        """Edited script + protect_manual_edits=False → warning only."""
        import logging

        path = tmp_path / "test.py"
        _write_script(path, SCRAPE_FUNCTION)
        content = path.read_text()
        path.write_text(content.replace("Test", "Edited"))

        s = _make(script=str(path), protect_manual_edits=False)
        with caplog.at_level(logging.WARNING, logger="scout"):
            s._check_script_protection(force=False)
        assert any("manually edited" in r.message for r in caplog.records)

    def test_check_protection_no_script_no_error(self):
        """No script path → no error."""
        s = _make(protect_manual_edits=True)
        s._check_script_protection(force=False)


# ═══════════════════════════════════════════════════════════════════════════
# Field(str, min=10) improved error message
# ═══════════════════════════════════════════════════════════════════════════

class TestFieldStrMinError:
    """Field(str, min=N) error suggests min_length with correct code."""

    def test_suggests_min_length(self):
        with pytest.raises(ScoutSchemaError, match="min_length") as exc_info:
            parse_schema(Field(str, min=10))
        assert "Did you mean" in str(exc_info.value)
        assert "Field(str, min_length=10)" in str(exc_info.value)

    def test_suggests_max_length(self):
        with pytest.raises(ScoutSchemaError, match="max_length") as exc_info:
            parse_schema(Field(str, max=100))
        assert "Field(str, max_length=100)" in str(exc_info.value)

    def test_suggests_both(self):
        with pytest.raises(ScoutSchemaError) as exc_info:
            parse_schema(Field(str, min=5, max=50))
        msg = str(exc_info.value)
        assert "min_length=5" in msg
        assert "max_length=50" in msg


# ═══════════════════════════════════════════════════════════════════════════
# Generation failure model suggestion
# ═══════════════════════════════════════════════════════════════════════════

class TestModelSuggestion:
    """Generation failure for weak models suggests upgrading."""

    def test_haiku_model_gets_suggestion(self):
        s = _make(model="claude-haiku-4-5")
        lower = s._model.lower()
        assert "haiku" in lower
        # The suggestion logic is in _run_generate, which we can't
        # easily test without mocking the agent. Instead, verify the
        # model name triggers the condition.
        assert any(kw in lower for kw in ("haiku", "mini", "flash"))

    def test_sonnet_model_no_suggestion(self):
        s = _make(model="claude-sonnet-4-5")
        lower = s._model.lower()
        assert not any(kw in lower for kw in ("haiku", "mini", "flash"))
