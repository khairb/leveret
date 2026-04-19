"""Tests for the error fingerprinter.

Uses real Phase 0 error fixtures for fixture-driven tests.
Also covers extraction edge cases and comparison logic.

Spec reference: docs/specific/AUTO_FIX_ALGORITHM.md S5
"""

from __future__ import annotations

import pytest

from scout.autofix.classifier import classify_error
from scout.autofix.fingerprint import compare_fingerprints, extract_fingerprint
from scout.autofix.types import ComparisonLevel, ErrorCategory, Fingerprint
from tests.autofix.conftest import all_error_fixtures, load_fixture


# ══════════════════════════════════════════════════════════════
#  Fixture-driven tests — fingerprints from real fixtures
# ══════════════════════════════════════════════════════════════


def _expected_category(name: str) -> ErrorCategory:
    for prefix in ("F1", "F2", "F3"):
        if name.startswith(prefix + "_"):
            return ErrorCategory[prefix]
    return ErrorCategory[name.split("_")[0]]


@pytest.mark.parametrize("fixture_name", all_error_fixtures())
def test_fingerprint_extraction_from_fixtures(fixture_name: str) -> None:
    """Every fixture should produce a fingerprint with category set."""
    fixture = load_fixture(fixture_name)
    category = _expected_category(fixture_name)
    fp = extract_fingerprint(
        fixture["stderr"], category, schema_error=fixture.get("schema_error"),
    )
    assert fp.category == category
    # Should preserve the raw message (or schema_error for G).
    # Exception: B_empty_output has empty stderr and no schema_error.
    if fixture_name != "B_empty_output":
        assert fp.message


# ══════════════════════════════════════════════════════════════
#  Category A fingerprints
# ══════════════════════════════════════════════════════════════


class TestFingerprintCategoryA:
    def test_syntax_error_type(self) -> None:
        stderr = "SyntaxError: invalid syntax\n"
        fp = extract_fingerprint(stderr, ErrorCategory.A)
        assert fp.error_type == "SyntaxError"
        assert fp.method is None

    def test_indentation_error_type(self) -> None:
        stderr = "IndentationError: expected an indented block\n"
        fp = extract_fingerprint(stderr, ErrorCategory.A)
        assert fp.error_type == "IndentationError"

    def test_module_not_found_with_target(self) -> None:
        stderr = "ModuleNotFoundError: No module named 'nonexistent_module'\n"
        fp = extract_fingerprint(stderr, ErrorCategory.A)
        assert fp.error_type == "ModuleNotFoundError"
        assert fp.target == "nonexistent_module"

    def test_import_error_with_module(self) -> None:
        stderr = "ModuleNotFoundError: No module named 'nonexistent_module_xyz'\n"
        fp = extract_fingerprint(stderr, ErrorCategory.A)
        assert fp.target == "nonexistent_module_xyz"


# ══════════════════════════════════════════════════════════════
#  Category B fingerprints
# ══════════════════════════════════════════════════════════════


class TestFingerprintCategoryB:
    def test_attribute_error_extracts_attribute(self) -> None:
        stderr = "AttributeError: 'NoneType' object has no attribute 'text_content'\n"
        fp = extract_fingerprint(stderr, ErrorCategory.B)
        assert fp.error_type == "AttributeError"
        assert fp.target == "text_content"

    def test_key_error_extracts_key(self) -> None:
        stderr = "KeyError: 'nonexistent_key'\n"
        fp = extract_fingerprint(stderr, ErrorCategory.B)
        assert fp.error_type == "KeyError"
        assert fp.target == "nonexistent_key"

    def test_index_error(self) -> None:
        stderr = "IndexError: list index out of range\n"
        fp = extract_fingerprint(stderr, ErrorCategory.B)
        assert fp.error_type == "IndexError"

    def test_type_error(self) -> None:
        stderr = "TypeError: int() argument must be a string\n"
        fp = extract_fingerprint(stderr, ErrorCategory.B)
        assert fp.error_type == "TypeError"

    def test_value_error(self) -> None:
        stderr = "ValueError: invalid literal for int()\n"
        fp = extract_fingerprint(stderr, ErrorCategory.B)
        assert fp.error_type == "ValueError"

    def test_js_type_error_via_evaluate(self) -> None:
        stderr = (
            "patchright._impl._errors.Error: Page.evaluate: TypeError: "
            "Cannot read properties of null (reading 'textContent')\n"
        )
        fp = extract_fingerprint(stderr, ErrorCategory.B)
        assert fp.error_type == "JS.TypeError"
        assert fp.method == "Page.evaluate"

    def test_js_reference_error(self) -> None:
        stderr = (
            "patchright._impl._errors.Error: Page.evaluate: ReferenceError: "
            "undefinedVariable is not defined\n"
        )
        fp = extract_fingerprint(stderr, ErrorCategory.B)
        assert fp.error_type == "JS.ReferenceError"
        assert fp.method == "Page.evaluate"

    def test_empty_output_no_info(self) -> None:
        fp = extract_fingerprint("", ErrorCategory.B)
        assert fp.error_type is None
        assert fp.method is None


# ══════════════════════════════════════════════════════════════
#  Category C fingerprints
# ══════════════════════════════════════════════════════════════


class TestFingerprintCategoryC:
    def test_net_err_code_and_url(self) -> None:
        stderr = "Page.goto: net::ERR_NAME_NOT_RESOLVED at https://example.com/\n"
        fp = extract_fingerprint(stderr, ErrorCategory.C)
        assert fp.error_type == "net::ERR_NAME_NOT_RESOLVED"
        assert fp.target == "https://example.com/"

    def test_connection_refused(self) -> None:
        stderr = (
            "patchright._impl._errors.Error: Page.goto: "
            "net::ERR_CONNECTION_REFUSED at http://localhost:9999/\n"
        )
        fp = extract_fingerprint(stderr, ErrorCategory.C)
        assert fp.error_type == "net::ERR_CONNECTION_REFUSED"
        assert fp.method == "Page.goto"
        assert fp.target == "http://localhost:9999/"

    def test_navigation_timeout(self) -> None:
        stderr = (
            "patchright._impl._errors.TimeoutError: Page.goto: "
            "Timeout 30000ms exceeded.\n"
        )
        fp = extract_fingerprint(stderr, ErrorCategory.C)
        assert fp.error_type == "TimeoutError"
        assert fp.method == "Page.goto"


# ══════════════════════════════════════════════════════════════
#  Category D fingerprints
# ══════════════════════════════════════════════════════════════


class TestFingerprintCategoryD:
    def test_wait_for_selector_with_target(self) -> None:
        stderr = (
            "Page.wait_for_selector: Timeout 5000ms exceeded.\n"
            "Call log:\n"
            '  - waiting for locator(".product-card")\n'
        )
        fp = extract_fingerprint(stderr, ErrorCategory.D)
        assert fp.error_type == "TimeoutError"
        assert fp.method == "Page.wait_for_selector"
        assert fp.target == ".product-card"

    def test_click_with_selector(self) -> None:
        stderr = (
            "Page.click: Timeout 5000ms exceeded.\n"
            "Call log:\n"
            '  - waiting for locator(".btn-submit")\n'
        )
        fp = extract_fingerprint(stderr, ErrorCategory.D)
        assert fp.method == "Page.click"
        assert fp.target == ".btn-submit"

    def test_locator_wait_for(self) -> None:
        stderr = (
            "Locator.wait_for: Timeout 5000ms exceeded.\n"
            "Call log:\n"
            '  - waiting for locator("#main")\n'
        )
        fp = extract_fingerprint(stderr, ErrorCategory.D)
        assert fp.method == "Locator.wait_for"
        assert fp.target == "#main"

    def test_wait_for_function_no_selector(self) -> None:
        stderr = "Page.wait_for_function: Timeout 5000ms exceeded.\n"
        fp = extract_fingerprint(stderr, ErrorCategory.D)
        assert fp.method == "Page.wait_for_function"
        assert fp.target is None  # No selector extractable


# ══════════════════════════════════════════════════════════════
#  Category E fingerprints
# ══════════════════════════════════════════════════════════════


class TestFingerprintCategoryE:
    def test_pointer_intercept(self) -> None:
        stderr = (
            "Page.click: Timeout 5000ms exceeded.\n"
            "Call log:\n"
            "  - <div></div> intercepts pointer events\n"
        )
        fp = extract_fingerprint(stderr, ErrorCategory.E)
        assert fp.error_type == "pointer_intercept"
        assert fp.target == "<div>"

    def test_not_visible(self) -> None:
        stderr = (
            "Page.click: Timeout 5000ms exceeded.\n"
            "Call log:\n"
            "  - element is not visible\n"
        )
        fp = extract_fingerprint(stderr, ErrorCategory.E)
        assert fp.error_type == "not_visible"

    def test_strict_mode(self) -> None:
        stderr = (
            "patchright._impl._errors.Error: Locator.click: Error: "
            "strict mode violation:  resolved to 3 elements:\n"
        )
        fp = extract_fingerprint(stderr, ErrorCategory.E)
        assert fp.error_type == "strict_mode"
        assert fp.target == "resolved_to_3"

    def test_context_destroyed(self) -> None:
        stderr = (
            "patchright._impl._errors.Error: Page.evaluate: "
            "Execution context was destroyed\n"
        )
        fp = extract_fingerprint(stderr, ErrorCategory.E)
        assert fp.error_type == "context_destroyed"
        assert fp.method == "Page.evaluate"

    def test_frame_detached(self) -> None:
        stderr = (
            "patchright._impl._errors.Error: Frame.evaluate: "
            "Frame was detached\n"
        )
        fp = extract_fingerprint(stderr, ErrorCategory.E)
        assert fp.error_type == "frame_detached"
        assert fp.method == "Frame.evaluate"

    def test_dialog_blocking(self) -> None:
        stderr = (
            'Page.evaluate: Cannot evaluate, page has an open JavaScript dialog: '
            'alert with message "test"\n'
        )
        fp = extract_fingerprint(stderr, ErrorCategory.E)
        assert fp.error_type == "dialog_blocking"

    def test_not_enabled(self) -> None:
        stderr = (
            "Locator.fill: Timeout 5000ms exceeded.\n"
            "Call log:\n"
            "  - element is not enabled\n"
        )
        fp = extract_fingerprint(stderr, ErrorCategory.E)
        assert fp.error_type == "not_enabled"


# ══════════════════════════════════════════════════════════════
#  Category F fingerprints
# ══════════════════════════════════════════════════════════════


class TestFingerprintCategoryF:
    def test_f1_target_closed(self) -> None:
        stderr = (
            "patchright._impl._errors.TargetClosedError: "
            "Page.evaluate: Target page, context or browser has been closed\n"
        )
        fp = extract_fingerprint(stderr, ErrorCategory.F1)
        assert fp.error_type == "TargetClosedError"
        assert fp.method == "Page.evaluate"

    def test_f1_page_crash(self) -> None:
        stderr = "Page crashed\n"
        fp = extract_fingerprint(stderr, ErrorCategory.F1)
        assert fp.error_type == "PageCrash"

    def test_f1_browser_closed(self) -> None:
        stderr = "Browser has been closed\n"
        fp = extract_fingerprint(stderr, ErrorCategory.F1)
        assert fp.error_type == "BrowserClosed"

    def test_f1_connection_lost(self) -> None:
        stderr = "Connection closed while reading from the driver\n"
        fp = extract_fingerprint(stderr, ErrorCategory.F1)
        assert fp.error_type == "ConnectionLost"

    def test_f1_socket(self) -> None:
        stderr = "Socket closed\n"
        fp = extract_fingerprint(stderr, ErrorCategory.F1)
        assert fp.error_type == "SocketError"

    def test_f2_with_timeout_value(self) -> None:
        stderr = "Function timed out after 10 seconds\n"
        fp = extract_fingerprint(stderr, ErrorCategory.F2)
        assert fp.error_type == "SubprocessTimeout"
        assert fp.target == "10s"

    def test_f3_browser_not_found(self) -> None:
        stderr = "Executable doesn't exist at /path/to/chrome\n"
        fp = extract_fingerprint(stderr, ErrorCategory.F3)
        assert fp.error_type == "BrowserNotFound"

    def test_f3_disk_full(self) -> None:
        stderr = "No space left on device\n"
        fp = extract_fingerprint(stderr, ErrorCategory.F3)
        assert fp.error_type == "DiskFull"

    def test_f3_too_many_files(self) -> None:
        stderr = "Too many open files\n"
        fp = extract_fingerprint(stderr, ErrorCategory.F3)
        assert fp.error_type == "TooManyFiles"

    def test_f3_no_sandbox(self) -> None:
        stderr = "No usable sandbox\n"
        fp = extract_fingerprint(stderr, ErrorCategory.F3)
        assert fp.error_type == "SandboxError"

    def test_f3_no_display(self) -> None:
        stderr = "Cannot open display\n"
        fp = extract_fingerprint(stderr, ErrorCategory.F3)
        assert fp.error_type == "NoDisplay"

    def test_f3_generic(self) -> None:
        stderr = "Event loop is closed! Is Playwright already stopped?\n"
        fp = extract_fingerprint(stderr, ErrorCategory.F3)
        assert fp.error_type == "InfrastructureError"


# ══════════════════════════════════════════════════════════════
#  Category G fingerprints
# ══════════════════════════════════════════════════════════════


class TestFingerprintCategoryG:
    def test_min_items_constraint(self) -> None:
        fp = extract_fingerprint("", ErrorCategory.G, "Expected at least 5 items, got 0")
        assert fp.error_type == "SchemaValidationError"
        assert fp.target == "item_count < 5"

    def test_missing_field(self) -> None:
        fp = extract_fingerprint("", ErrorCategory.G, "Missing required field: 'price'")
        assert fp.target == "missing field: price"

    def test_type_mismatch(self) -> None:
        fp = extract_fingerprint(
            "", ErrorCategory.G, "Expected float for field 'price', got str",
        )
        assert fp.target == "type mismatch: price"

    def test_unknown_schema_error(self) -> None:
        fp = extract_fingerprint("", ErrorCategory.G, "Some custom validation error")
        assert fp.error_type == "SchemaValidationError"
        assert fp.target is None


# ══════════════════════════════════════════════════════════════
#  Fingerprint comparison
# ══════════════════════════════════════════════════════════════


class TestCompareFingerprints:
    """Tests for compare_fingerprints() at all 4 levels."""

    def test_exact_match(self) -> None:
        fp1 = Fingerprint(
            ErrorCategory.D, "TimeoutError", "Page.click", ".card", "msg",
        )
        fp2 = Fingerprint(
            ErrorCategory.D, "TimeoutError", "Page.click", ".card", "msg2",
        )
        assert compare_fingerprints(fp1, fp2) == ComparisonLevel.EXACT

    def test_same_kind_different_targets(self) -> None:
        fp1 = Fingerprint(
            ErrorCategory.D, "TimeoutError", "Page.click", ".card", "",
        )
        fp2 = Fingerprint(
            ErrorCategory.D, "TimeoutError", "Page.click", ".price", "",
        )
        assert compare_fingerprints(fp1, fp2) == ComparisonLevel.SAME_KIND

    def test_same_category_different_method(self) -> None:
        fp1 = Fingerprint(
            ErrorCategory.D, "TimeoutError", "Page.click", ".card", "",
        )
        fp2 = Fingerprint(
            ErrorCategory.D, "TimeoutError", "Page.fill", ".card", "",
        )
        assert compare_fingerprints(fp1, fp2) == ComparisonLevel.SAME_CATEGORY

    def test_same_category_different_error_type(self) -> None:
        fp1 = Fingerprint(ErrorCategory.B, "AttributeError", None, None, "")
        fp2 = Fingerprint(ErrorCategory.B, "KeyError", None, None, "")
        assert compare_fingerprints(fp1, fp2) == ComparisonLevel.SAME_CATEGORY

    def test_different_categories(self) -> None:
        fp1 = Fingerprint(ErrorCategory.B, "AttributeError", None, None, "")
        fp2 = Fingerprint(ErrorCategory.D, "TimeoutError", None, None, "")
        assert compare_fingerprints(fp1, fp2) == ComparisonLevel.NONE

    def test_both_targets_none_is_exact(self) -> None:
        fp1 = Fingerprint(ErrorCategory.D, "TimeoutError", "Page.click", None, "")
        fp2 = Fingerprint(ErrorCategory.D, "TimeoutError", "Page.click", None, "")
        assert compare_fingerprints(fp1, fp2) == ComparisonLevel.EXACT

    def test_one_target_none_is_same_kind(self) -> None:
        """Degraded: one has target, other doesn't."""
        fp1 = Fingerprint(
            ErrorCategory.D, "TimeoutError", "Page.click", ".card", "",
        )
        fp2 = Fingerprint(
            ErrorCategory.D, "TimeoutError", "Page.click", None, "",
        )
        assert compare_fingerprints(fp1, fp2) == ComparisonLevel.SAME_KIND

    def test_both_none_error_type_and_method(self) -> None:
        """Minimal fingerprints — same category only."""
        fp1 = Fingerprint(ErrorCategory.B, None, None, None, "error1")
        fp2 = Fingerprint(ErrorCategory.B, None, None, None, "error2")
        # Both error_type and method are None, so they match → EXACT
        assert compare_fingerprints(fp1, fp2) == ComparisonLevel.EXACT

    def test_schema_fingerprints_exact(self) -> None:
        fp1 = Fingerprint(
            ErrorCategory.G, "SchemaValidationError", None, "item_count < 5", "",
        )
        fp2 = Fingerprint(
            ErrorCategory.G, "SchemaValidationError", None, "item_count < 5", "",
        )
        assert compare_fingerprints(fp1, fp2) == ComparisonLevel.EXACT

    def test_schema_fingerprints_same_kind(self) -> None:
        fp1 = Fingerprint(
            ErrorCategory.G, "SchemaValidationError", None, "item_count < 5", "",
        )
        fp2 = Fingerprint(
            ErrorCategory.G, "SchemaValidationError", None, "missing field: price", "",
        )
        assert compare_fingerprints(fp1, fp2) == ComparisonLevel.SAME_KIND

    def test_network_errors_same_kind(self) -> None:
        """Different net::ERR_* codes = SAME_CATEGORY (different error_type)."""
        fp1 = Fingerprint(
            ErrorCategory.C, "net::ERR_CONNECTION_REFUSED", "Page.goto", None, "",
        )
        fp2 = Fingerprint(
            ErrorCategory.C, "net::ERR_CONNECTION_RESET", "Page.goto", None, "",
        )
        assert compare_fingerprints(fp1, fp2) == ComparisonLevel.SAME_CATEGORY

    def test_identical_network_errors_exact(self) -> None:
        fp1 = Fingerprint(
            ErrorCategory.C, "net::ERR_CONNECTION_REFUSED", "Page.goto",
            "http://example.com/", "",
        )
        fp2 = Fingerprint(
            ErrorCategory.C, "net::ERR_CONNECTION_REFUSED", "Page.goto",
            "http://example.com/", "",
        )
        assert compare_fingerprints(fp1, fp2) == ComparisonLevel.EXACT


# ══════════════════════════════════════════════════════════════
#  Selector extraction
# ══════════════════════════════════════════════════════════════


class TestSelectorExtraction:
    """Verify selectors are reliably extracted from various error formats."""

    def test_locator_in_call_log(self) -> None:
        stderr = (
            "Page.click: Timeout 5000ms exceeded.\n"
            "Call log:\n"
            '  - waiting for locator(".product-card")\n'
        )
        fp = extract_fingerprint(stderr, ErrorCategory.D)
        assert fp.target == ".product-card"

    def test_locator_with_id_selector(self) -> None:
        stderr = (
            "Page.click: Timeout 5000ms exceeded.\n"
            "Call log:\n"
            '  - waiting for locator("#submit-btn")\n'
        )
        fp = extract_fingerprint(stderr, ErrorCategory.D)
        assert fp.target == "#submit-btn"

    def test_locator_with_complex_selector(self) -> None:
        stderr = (
            "Page.click: Timeout 5000ms exceeded.\n"
            "Call log:\n"
            '  - waiting for locator("div.card > span.price")\n'
        )
        fp = extract_fingerprint(stderr, ErrorCategory.D)
        assert fp.target == "div.card > span.price"

    def test_wait_for_selector_in_code(self) -> None:
        """Selector from the scrape function code in the traceback."""
        stderr = (
            '  File "/tmp/script.py", line 42, in scrape\n'
            "    await page.wait_for_selector('.product-list', timeout=5000)\n"
            "Page.wait_for_selector: Timeout 5000ms exceeded.\n"
        )
        fp = extract_fingerprint(stderr, ErrorCategory.D)
        assert fp.target == ".product-list"

    def test_click_in_code(self) -> None:
        stderr = (
            '  File "/tmp/script.py", line 42, in scrape\n'
            "    await page.click('.submit-btn', timeout=5000)\n"
            "Page.click: Timeout 5000ms exceeded.\n"
        )
        fp = extract_fingerprint(stderr, ErrorCategory.D)
        assert fp.target == ".submit-btn"

    def test_pointer_intercept_extracts_element(self) -> None:
        stderr = (
            "Page.click: Timeout 5000ms exceeded.\n"
            "Call log:\n"
            '  - <div class="overlay"></div> intercepts pointer events\n'
        )
        fp = extract_fingerprint(stderr, ErrorCategory.E)
        assert fp.target == "<div>"

    def test_no_selector_available(self) -> None:
        """When no selector can be extracted, target is None."""
        stderr = "Page.wait_for_function: Timeout 5000ms exceeded.\n"
        fp = extract_fingerprint(stderr, ErrorCategory.D)
        assert fp.target is None
