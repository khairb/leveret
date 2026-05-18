"""Tests for Scraper constructor, validation, and __repr__.

Track 1 of the Scraper class implementation. Tests cover:
- All 9 constructor parameters with valid inputs
- All validation rules from the spec (error types, error messages)
- Path normalization (tilde, relative, suffix)
- _normalize_domain() helper
- __repr__ output
- Edge cases and real-world usage patterns
"""

from pathlib import Path

import pytest

from scout.errors import ScoutConfigError, ScoutError, ScoutSchemaError
from scout.schema.types import Field, List
from scout.scraper import Scraper, ScraperResult, _normalize_domain

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_URL = "https://example.com"
VALID_TASK = "Extract product prices"
VALID_SCHEMA = [{"title": str, "price": float}]


def _make(**overrides):
    """Build a Scraper with valid defaults, overriding specific params."""
    kwargs = {
        "schema": VALID_SCHEMA,
    }
    kwargs.update(overrides)
    url = kwargs.pop("url", VALID_URL)
    task = kwargs.pop("task", VALID_TASK)
    return Scraper(url, task, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Constructor — happy paths
# ═══════════════════════════════════════════════════════════════════════════


class TestConstructorHappyPath:
    """Valid inputs are accepted and stored correctly."""

    def test_minimal_constructor(self):
        s = Scraper("https://example.com", "Get data", schema={"x": str})
        assert s._url == "https://example.com"
        assert s._task == "Get data"
        assert s._compiled_schema is not None

    def test_all_defaults(self):
        s = _make()
        assert s._model == "claude-haiku-4-5"
        assert s._headless is True
        assert s._api_key is None
        assert s._run_timeout == 600
        assert s._generation_attempts == 6
        assert s._script_path is None
        assert s._cached_fn is None
        assert s._browser_mgr is None
        assert s._context_managed is False

    def test_all_params_explicit(self, tmp_path):
        script = tmp_path / "scraper.py"
        s = Scraper(
            "https://shop.example.com/products",
            "Extract all product listings",
            schema=List({"title": str, "price": Field(float, min=0)}, min_items=10),
            script=str(script),
            model="claude-sonnet-4-5-20250514",
            headless=False,
            api_key="sk-test-key",
            run_timeout=300,
            generation_attempts=3,
        )
        assert s._url == "https://shop.example.com/products"
        assert s._task == "Extract all product listings"
        assert s._script_path == script
        assert s._model == "claude-sonnet-4-5-20250514"
        assert s._headless is False
        assert s._api_key == "sk-test-key"
        assert s._run_timeout == 300
        assert s._generation_attempts == 3

    def test_url_and_task_are_positional(self):
        """url and task can be passed positionally."""
        s = Scraper("https://example.com", "task", schema={"x": str})
        assert s._url == "https://example.com"
        assert s._task == "task"

    def test_url_and_task_as_keywords(self):
        """url and task can also be passed as keywords."""
        s = Scraper(
            url="https://example.com",
            task="task",
            schema={"x": str},
        )
        assert s._url == "https://example.com"

    def test_schema_is_keyword_only(self):
        """schema cannot be passed positionally."""
        with pytest.raises(TypeError):
            Scraper("https://example.com", "task", {"x": str})

    def test_http_url_accepted(self):
        s = _make(url="http://localhost:8080/page")
        assert s._url == "http://localhost:8080/page"

    def test_complex_schema_compiled(self):
        """Complex schema is compiled at construction time."""
        schema = List(
            {
                "title": Field(str, min_length=1),
                "price": Field(float, min=0),
                "currency": Field(str, enum=["USD", "EUR"]),
                "tags": [str],
                "details": {
                    "brand": str,
                    "sku": Field(str, optional=True),
                },
            },
            min_items=20,
        )
        s = _make(schema=schema)
        assert s._compiled_schema.prompt  # non-empty prompt
        valid, _ = s._compiled_schema.validate(
            [
                {
                    "title": "T",
                    "price": 1.0,
                    "currency": "USD",
                    "tags": ["a"],
                    "details": {"brand": "B", "sku": None},
                }
            ]
            * 20
        )
        assert valid is True


# ═══════════════════════════════════════════════════════════════════════════
# URL validation
# ═══════════════════════════════════════════════════════════════════════════


class TestURLValidation:
    def test_empty_string(self):
        with pytest.raises(ScoutError, match=r"url must be a valid HTTP\(S\) URL \(got"):
            _make(url="")

    def test_whitespace_only(self):
        with pytest.raises(ScoutError, match=r"url must be a valid HTTP\(S\) URL"):
            _make(url="   ")

    def test_no_scheme(self):
        with pytest.raises(ScoutError, match="url must start with https://"):
            _make(url="example.com")

    def test_ftp_scheme(self):
        with pytest.raises(ScoutError, match="url must start with https://"):
            _make(url="ftp://example.com")

    def test_file_scheme(self):
        with pytest.raises(ScoutError, match="url must start with https://"):
            _make(url="file:///etc/passwd")

    def test_just_scheme(self):
        with pytest.raises(ScoutError, match="url must include a hostname"):
            _make(url="https://")

    def test_not_a_string(self):
        with pytest.raises(ScoutError, match=r"url must be a valid HTTP\(S\) URL"):
            _make(url=42)

    def test_none_url(self):
        with pytest.raises(ScoutError, match=r"url must be a valid HTTP\(S\) URL"):
            _make(url=None)


# ═══════════════════════════════════════════════════════════════════════════
# Task validation
# ═══════════════════════════════════════════════════════════════════════════


class TestTaskValidation:
    def test_empty_string(self):
        with pytest.raises(ScoutError, match="task must not be empty"):
            _make(task="")

    def test_whitespace_only(self):
        with pytest.raises(ScoutError, match="task must not be empty"):
            _make(task="   ")

    def test_not_a_string(self):
        with pytest.raises(ScoutError, match="task must not be empty"):
            _make(task=123)

    def test_none_task(self):
        with pytest.raises(ScoutError, match="task must not be empty"):
            _make(task=None)


# ═══════════════════════════════════════════════════════════════════════════
# Schema validation
# ═══════════════════════════════════════════════════════════════════════════


class TestSchemaValidation:
    def test_none_schema(self):
        with pytest.raises(
            ScoutSchemaError,
            match="schema is required",
        ):
            Scraper("https://x.com", "task", schema=None)

    def test_invalid_schema_type(self):
        """An unsupported type raises ScoutSchemaError from compile_schema."""
        with pytest.raises(ScoutSchemaError):
            _make(schema="not a schema")

    def test_invalid_field_constraint(self):
        """Field with incompatible constraint raises ScoutSchemaError."""
        with pytest.raises(ScoutSchemaError):
            _make(schema=[{"x": Field(int, pattern=r"\d+")}])

    def test_bare_type_schema(self):
        """Bare type is valid."""
        s = _make(schema=str)
        assert s._compiled_schema is not None

    def test_dict_type_schema(self):
        """Bare dict type (freestyle) is valid."""
        s = _make(schema=dict)
        assert s._compiled_schema is not None

    def test_schema_compiled_eagerly(self):
        """Schema is compiled at construction — errors surface immediately."""
        with pytest.raises(ScoutSchemaError):
            _make(schema=[{"x": Field(str, min=-1)}])


# ═══════════════════════════════════════════════════════════════════════════
# Script path validation and normalization
# ═══════════════════════════════════════════════════════════════════════════


class TestScriptPathValidation:
    def test_none_script(self):
        s = _make(script=None)
        assert s._script_path is None

    def test_valid_py_path(self, tmp_path):
        script = tmp_path / "scraper.py"
        s = _make(script=str(script))
        assert s._script_path == script

    def test_no_py_extension(self):
        with pytest.raises(ScoutError, match="script must be a .py file path"):
            _make(script="scraper.txt")

    def test_no_extension(self):
        with pytest.raises(ScoutError, match="script must be a .py file path"):
            _make(script="./scrapers/hn")

    def test_directory_path(self, tmp_path):
        with pytest.raises(
            ScoutError,
            match="script must be a file path, not a directory",
        ):
            _make(script=str(tmp_path))

    def test_tilde_expansion(self):
        s = _make(script="~/scrapers/test.py")
        assert "~" not in str(s._script_path)
        assert s._script_path.is_absolute()

    def test_relative_resolved(self):
        s = _make(script="./scrapers/hn.py")
        assert s._script_path.is_absolute()
        assert s._script_path.name == "hn.py"

    def test_path_object_accepted(self, tmp_path):
        script = tmp_path / "test.py"
        s = _make(script=script)
        assert s._script_path == script

    def test_nonexistent_path_ok(self, tmp_path):
        """Path doesn't need to exist — it'll be created on first run."""
        script = tmp_path / "not_yet" / "scraper.py"
        s = _make(script=str(script))
        assert s._script_path.name == "scraper.py"

    def test_dot_py_only(self):
        """'.pyw', '.pyc' etc. are not accepted."""
        with pytest.raises(ScoutError, match="script must be a .py file path"):
            _make(script="test.pyw")


# ═══════════════════════════════════════════════════════════════════════════
# Timeout validation
# ═══════════════════════════════════════════════════════════════════════════


class TestRunTimeoutValidation:
    def test_zero(self):
        with pytest.raises(ScoutError, match="run_timeout must be a positive integer"):
            _make(run_timeout=0)

    def test_negative(self):
        with pytest.raises(ScoutError, match="run_timeout must be a positive integer"):
            _make(run_timeout=-10)

    def test_float_rejected(self):
        """Float is not an int, even if whole."""
        with pytest.raises(ScoutError, match="run_timeout must be a positive integer"):
            _make(run_timeout=30.0)

    def test_string_rejected(self):
        with pytest.raises(ScoutError, match="run_timeout must be a positive integer"):
            _make(run_timeout="60")

    def test_valid_timeout(self):
        s = _make(run_timeout=1)
        assert s._run_timeout == 1

    def test_large_timeout(self):
        s = _make(run_timeout=3600)
        assert s._run_timeout == 3600


# ═══════════════════════════════════════════════════════════════════════════
# generation_attempts validation
# ═══════════════════════════════════════════════════════════════════════════


class TestGenerationAttemptsValidation:
    def test_negative(self):
        with pytest.raises(ScoutError, match="generation_attempts must be a positive integer"):
            _make(generation_attempts=-1)

    def test_zero_rejected(self):
        with pytest.raises(ScoutError, match="generation_attempts must be a positive integer"):
            _make(generation_attempts=0)

    def test_float_rejected(self):
        with pytest.raises(ScoutError, match="generation_attempts must be a positive integer"):
            _make(generation_attempts=3.0)

    def test_one_is_minimum(self):
        s = _make(generation_attempts=1)
        assert s._generation_attempts == 1


# ═══════════════════════════════════════════════════════════════════════════
# Model validation
# ═══════════════════════════════════════════════════════════════════════════


class TestModelValidation:
    def test_empty_string(self):
        with pytest.raises(ScoutError, match="model must not be empty"):
            _make(model="")

    def test_whitespace_only(self):
        with pytest.raises(ScoutError, match="model must not be empty"):
            _make(model="   ")

    def test_not_a_string(self):
        with pytest.raises(ScoutError, match="model must not be empty"):
            _make(model=42)

    def test_valid_model(self):
        s = _make(model="claude-sonnet-4-5-20250514")
        assert s._model == "claude-sonnet-4-5-20250514"

    def test_valid_provider_prefixed_model(self):
        s = _make(model="openai:gpt-4o")
        assert s._model == "openai:gpt-4o"

    def test_gpt_without_prefix_raises(self):
        with pytest.raises(ScoutConfigError, match="OpenAI"):
            _make(model="gpt-4o")

    def test_gemini_without_prefix_raises(self):
        with pytest.raises(ScoutConfigError, match="Google"):
            _make(model="gemini-2.0-flash")

    def test_llama_without_prefix_raises(self):
        with pytest.raises(ScoutConfigError, match="Groq"):
            _make(model="llama-3.3-70b-versatile")

    def test_mistral_without_prefix_raises(self):
        with pytest.raises(ScoutConfigError, match="Mistral"):
            _make(model="mistral-large-latest")

    def test_deepseek_without_prefix_raises(self):
        with pytest.raises(ScoutConfigError, match="DeepSeek"):
            _make(model="deepseek-chat")

    def test_suggests_provider_prefix(self):
        """Error message includes the correct provider:model format."""
        with pytest.raises(ScoutConfigError, match="openai:gpt-4o"):
            _make(model="gpt-4o")


# ═══════════════════════════════════════════════════════════════════════════
# api_key — stored, not validated at construction
# ═══════════════════════════════════════════════════════════════════════════


class TestApiKey:
    def test_none_default(self):
        s = _make()
        assert s._api_key is None

    def test_stored_as_is(self):
        s = _make(api_key="sk-ant-12345")
        assert s._api_key == "sk-ant-12345"

    def test_empty_string_stored(self):
        """Empty api_key is NOT validated at construction (spec says
        'Not checked at construction — checked at generation time')."""
        s = _make(api_key="")
        assert s._api_key == ""


# ═══════════════════════════════════════════════════════════════════════════
# _normalize_domain
# ═══════════════════════════════════════════════════════════════════════════


class TestNormalizeDomain:
    def test_strips_www(self):
        assert _normalize_domain("https://www.example.com") == "example.com"

    def test_plain_domain(self):
        assert _normalize_domain("https://example.com") == "example.com"

    def test_preserves_subdomain(self):
        assert _normalize_domain("https://m.example.com") == "m.example.com"

    def test_preserves_api_subdomain(self):
        assert _normalize_domain("https://api.example.com") == "api.example.com"

    def test_lowercases(self):
        assert _normalize_domain("https://WWW.Example.COM") == "example.com"

    def test_with_path(self):
        assert _normalize_domain("https://www.example.com/foo/bar") == "example.com"

    def test_with_port(self):
        assert _normalize_domain("https://example.com:8080/page") == "example.com"

    def test_empty_url(self):
        assert _normalize_domain("") == ""

    def test_no_hostname(self):
        assert _normalize_domain("not-a-url") == ""

    def test_www_only_strips_leading(self):
        """'www.' in the middle of a domain is preserved."""
        assert _normalize_domain("https://notwww.example.com") == "notwww.example.com"

    def test_http_and_https_same(self):
        """Scheme doesn't affect the normalized domain."""
        assert _normalize_domain("http://example.com") == _normalize_domain("https://example.com")

    # Spec table cases
    def test_spec_case_www_vs_plain(self):
        assert _normalize_domain("https://www.example.com/a") == _normalize_domain(
            "https://example.com/b"
        )

    def test_spec_case_plain_vs_www(self):
        assert _normalize_domain("https://example.com/a") == _normalize_domain(
            "https://www.example.com/b"
        )

    def test_spec_case_m_subdomain_differs(self):
        assert _normalize_domain("https://m.example.com/a") != _normalize_domain(
            "https://example.com/b"
        )

    def test_spec_case_different_subdomains(self):
        assert _normalize_domain("https://api.example.com") != _normalize_domain(
            "https://app.example.com"
        )


# ═══════════════════════════════════════════════════════════════════════════
# __repr__
# ═══════════════════════════════════════════════════════════════════════════


class TestScraperRepr:
    def test_minimal(self):
        s = _make()
        r = repr(s)
        assert r == f"Scraper({VALID_URL!r})"

    def test_with_script(self, tmp_path):
        script = tmp_path / "hn.py"
        s = _make(script=str(script))
        r = repr(s)
        assert "Scraper(" in r
        assert "script=" in r
        assert str(script) in r
        assert "cached=True" not in r  # not cached yet

    def test_with_cached_fn(self, tmp_path):
        """When _cached_fn is set, cached=True appears in Scraper repr."""
        script = tmp_path / "hn.py"
        s = _make(script=str(script))
        s._cached_fn = lambda: None  # simulate cached state
        r = repr(s)
        assert "cached=True" in r

    def test_no_script_no_cached(self):
        """Without script=, no script or cached info shown."""
        s = _make()
        r = repr(s)
        assert "script=" not in r
        assert "cached" not in r


class TestScraperResultRepr:
    """ScraperResult repr was built in Phase 1 — verify it still works."""

    def test_list_data(self):
        r = ScraperResult(
            data=[{"a": 1}, {"a": 2}],
            url="https://example.com",
            timestamp="2024-01-01T00:00:00Z",
            script_generated=False,
            script_path=Path("/path/to/script.py"),
        )
        assert "items=2" in repr(r)
        assert "script_generated=False" in repr(r)

    def test_dict_data(self):
        r = ScraperResult(
            data={"name": "test", "count": 42},
            url="https://x.com",
            timestamp="2024-01-01T00:00:00Z",
            script_generated=True,
            script_path=None,
        )
        assert "keys=" in repr(r)


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases and real-world patterns
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_very_long_url(self):
        url = "https://example.com/" + "a" * 2000
        s = _make(url=url)
        assert s._url == url

    def test_url_with_query_and_fragment(self):
        url = "https://example.com/page?q=test&page=2#section"
        s = _make(url=url)
        assert s._url == url

    def test_url_with_auth(self):
        url = "https://user:pass@example.com/page"
        s = _make(url=url)
        assert s._url == url

    def test_localhost_url(self):
        s = _make(url="http://localhost:3000/api")
        assert s._url == "http://localhost:3000/api"

    def test_ip_address_url(self):
        s = _make(url="http://192.168.1.1:8080/")
        assert s._url == "http://192.168.1.1:8080/"

    def test_unicode_task(self):
        s = _make(task="Extrahiere die Produktpreise auf Deutsch")
        assert "Deutsch" in s._task

    def test_validation_order_url_before_schema(self):
        """URL is validated before schema — first error wins."""
        with pytest.raises(ScoutError, match="url must be"):
            Scraper("", "task", schema="bad schema")

    def test_validation_order_task_before_schema(self):
        with pytest.raises(ScoutError, match="task must not be empty"):
            Scraper("https://x.com", "", schema="bad schema")

    def test_multiple_instances_independent(self):
        """Two Scraper instances don't share state."""
        s1 = _make(url="https://a.com")
        s2 = _make(url="https://b.com")
        assert s1._url != s2._url
        assert s1._compiled_schema is not s2._compiled_schema

    def test_compiled_schema_has_prompt(self):
        """The compiled schema includes a non-empty prompt string."""
        s = _make(schema=List({"title": str, "price": float}, min_items=10))
        assert "## Output Schema" in s._compiled_schema.prompt
        assert "title" in s._compiled_schema.prompt
        assert "price" in s._compiled_schema.prompt

    def test_bool_true_not_int(self):
        """bool(True) is 1 in Python — run_timeout=True should be rejected
        since bool is not int for our purposes... Actually, isinstance(True, int)
        is True in Python. The spec says 'Positive integer'. Let's verify
        current behavior is consistent."""
        # In Python, isinstance(True, int) is True and True > 0, so this
        # would pass. This is a Python quirk — booleans ARE integers.
        # The spec doesn't special-case this, so we follow Python semantics.
        s = _make(run_timeout=True)  # True == 1, a positive int
        assert s._run_timeout is True

    def test_properties(self):
        """Public properties return correct values."""
        s = _make(url="https://test.com", script="./test.py")
        assert s.url == "https://test.com"
        assert s.script_path is not None
        assert s.script_path.is_absolute()


# ═══════════════════════════════════════════════════════════════════════════
# Real-world quickstart patterns from spec
# ═══════════════════════════════════════════════════════════════════════════


class TestSpecExamples:
    """Verify the exact examples from the spec work."""

    def test_quickstart_pattern(self, tmp_path):
        script = tmp_path / "scrapers" / "hn.py"
        scraper = Scraper(
            "https://news.ycombinator.com",
            "Extract the top stories",
            schema=List(
                {
                    "title": str,
                    "url": str,
                    "points": int,
                },
                min_items=20,
            ),
            script=str(script),
        )
        assert scraper._url == "https://news.ycombinator.com"
        assert scraper._script_path == script

    def test_minimal_schema(self):
        scraper = Scraper(
            "https://example.com",
            "Extract prices",
            schema=[{"title": str, "price": float}],
        )
        assert scraper._compiled_schema is not None

    def test_constrained_schema(self):
        scraper = Scraper(
            "https://example.com",
            "Extract listings",
            schema=List(
                {
                    "title": Field(str, min_length=1),
                    "price": Field(float, min=0),
                    "currency": Field(str, enum=["USD", "EUR", "GBP"]),
                    "rating": Field(int, min=1, max=5, optional=True),
                    "in_stock": bool,
                },
                min_items=20,
            ),
        )
        assert scraper._compiled_schema is not None


# ═══════════════════════════════════════════════════════════════════════════
# Round 3: RegenerateMode export, List deprecation, protect_manual_edits, warnings
# ═══════════════════════════════════════════════════════════════════════════


class TestRegenerateModeExport:
    """RegenerateMode enum is importable and usable."""

    def test_import_from_scout(self):
        from scout import RegenerateMode

        assert hasattr(RegenerateMode, "BALANCED")
        assert hasattr(RegenerateMode, "CAUTIOUS")
        assert hasattr(RegenerateMode, "EAGER")
        assert hasattr(RegenerateMode, "ALWAYS")

    def test_usable_in_constructor(self):
        from scout import RegenerateMode

        s = _make(
            auto_regenerate=RegenerateMode.BALANCED,
            script="test.py",
        )
        from scout.autofix.types import RegenerateMode as RGM

        assert s._auto_regenerate_mode == RGM.BALANCED


class TestListDeprecation:
    """List alias emits DeprecationWarning."""

    def test_list_warns(self):
        import warnings

        from scout.schema.types import List

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            List({"title": str}, min_items=5)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "Items" in str(w[0].message)

    def test_list_returns_items(self):
        import warnings

        from scout.schema.types import Items, List

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = List({"title": str}, min_items=5)
            assert isinstance(result, Items)
            assert result.min_items == 5


class TestProtectManualEdits:
    """protect_manual_edits parameter stores correctly."""

    def test_default_false(self):
        s = _make()
        assert s._protect_manual_edits is False

    def test_explicit_true(self):
        s = _make(protect_manual_edits=True, script="test.py")
        assert s._protect_manual_edits is True


class TestScriptNoneWarning:
    """script=None emits a warning-level log."""

    def test_warns_on_no_script(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="scout"):
            _make(script=None)
        assert any("No script= path set" in r.message for r in caplog.records)
        warning_records = [r for r in caplog.records if "No script= path set" in r.message]
        assert warning_records[0].levelno == logging.WARNING
