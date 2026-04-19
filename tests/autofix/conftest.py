"""Shared pytest fixtures for autofix tests.

Provides:
- Fixture loaders for error JSON and anti-bot HTML files
- ``local_server`` fixture (starts/stops the local test server)
- Parametrize helpers for fixture-driven tests
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.autofix.local_server import LocalTestServer

# ── Paths ───────────────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).parent / "fixtures"
ERRORS_DIR = FIXTURES_DIR / "errors"
ANTIBOT_DIR = FIXTURES_DIR / "antibot"
PAGES_DIR = FIXTURES_DIR / "pages"

# ── Error fixture loaders ───────────────────────────────────────


def load_fixture(name: str) -> dict[str, Any]:
    """Load a JSON error fixture by name (without .json extension).

    Args:
        name: Fixture name, e.g. ``"A_syntax_error"`` or ``"B_key_error"``.

    Returns:
        Parsed JSON dict with keys: name, url, stdout, stderr, returncode.
    """
    path = ERRORS_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def all_fixtures_for_category(category: str) -> list[str]:
    """Return all fixture names for a given category letter.

    Args:
        category: Single letter, e.g. ``"A"``, ``"B"``, ``"F1"``.

    Returns:
        Sorted list of fixture names (without .json).
    """
    prefix = f"{category}_"
    return sorted(
        f.stem
        for f in ERRORS_DIR.glob(f"{prefix}*.json")
    )


def all_error_fixtures() -> list[str]:
    """Return all error fixture names."""
    return sorted(f.stem for f in ERRORS_DIR.glob("*.json"))


# ── Anti-bot fixture loaders ────────────────────────────────────


def load_antibot_page(
    name: str,
) -> tuple[str, dict[str, str], list[dict[str, str]]]:
    """Load an anti-bot HTML fixture with its companion headers/cookies.

    Args:
        name: Fixture name without extension, e.g. ``"cloudflare_challenge"``.

    Returns:
        Tuple of ``(html_content, headers_dict, cookies_list)``.
        If no companion ``.headers.json`` file exists, headers and cookies
        are empty.
    """
    html_path = ANTIBOT_DIR / f"{name}.html"
    headers_path = ANTIBOT_DIR / f"{name}.headers.json"

    html_content = html_path.read_text(encoding="utf-8")

    headers: dict[str, str] = {}
    cookies: list[dict[str, str]] = []

    if headers_path.exists():
        data = json.loads(headers_path.read_text(encoding="utf-8"))
        headers = data.get("headers", {})
        cookies = data.get("cookies", [])

    return html_content, headers, cookies


def all_antibot_fixtures() -> list[str]:
    """Return all anti-bot fixture names (HTML files only)."""
    return sorted(f.stem for f in ANTIBOT_DIR.glob("*.html"))


# ── Pytest fixtures ─────────────────────────────────────────────


@pytest.fixture()
def fixture_dir() -> Path:
    """Path to the fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture()
def errors_dir() -> Path:
    """Path to the error fixtures directory."""
    return ERRORS_DIR


@pytest.fixture()
def antibot_dir() -> Path:
    """Path to the anti-bot fixtures directory."""
    return ANTIBOT_DIR


@pytest.fixture()
async def local_server():
    """Start and stop the local test server for the duration of a test.

    Yields a ``LocalTestServer`` instance with ``.url(path)`` method.
    """
    server = LocalTestServer()
    await server.start()
    yield server
    await server.stop()
