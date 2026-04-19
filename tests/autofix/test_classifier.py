"""Tests for the error classifier.

Uses real Phase 0 error fixtures for all fixture-driven tests.
Also covers edge cases, priority ordering, and synthetic patterns
not triggered during Phase 0 harvesting.

Spec reference: docs/specific/AUTO_FIX_ALGORITHM.md S4, S13
"""

from __future__ import annotations

import pytest

from scout.autofix.classifier import classify_error
from scout.autofix.types import ErrorCategory
from tests.autofix.conftest import all_error_fixtures, all_fixtures_for_category, load_fixture


# ══════════════════════════════════════════════════════════════
#  Fixture-driven tests — every Phase 0 fixture is classified
# ══════════════════════════════════════════════════════════════


def _expected_category(name: str) -> ErrorCategory:
    """Derive expected category from fixture name prefix."""
    for prefix in ("F1", "F2", "F3"):
        if name.startswith(prefix + "_"):
            return ErrorCategory[prefix]
    letter = name.split("_")[0]
    return ErrorCategory[letter]


@pytest.mark.parametrize("fixture_name", all_error_fixtures())
def test_all_fixtures_classified_correctly(fixture_name: str) -> None:
    """Every Phase 0 fixture must classify to its expected category."""
    fixture = load_fixture(fixture_name)
    expected = _expected_category(fixture_name)
    actual = classify_error(
        fixture["stderr"],
        exit_code=fixture["returncode"],
        schema_error=fixture.get("schema_error"),
        stdout=fixture.get("stdout", ""),
    )
    assert actual == expected, (
        f"Fixture {fixture_name}: expected {expected.value}, got {actual.value}"
    )


# ══════════════════════════════════════════════════════════════
#  Category A — Parse errors
# ══════════════════════════════════════════════════════════════


class TestCategoryA:
    """SyntaxError, IndentationError, TabError, ImportError, ModuleNotFoundError."""

    def test_syntax_error(self) -> None:
        stderr = 'SyntaxError: invalid syntax\n'
        assert classify_error(stderr) == ErrorCategory.A

    def test_indentation_error(self) -> None:
        stderr = 'IndentationError: expected an indented block\n'
        assert classify_error(stderr) == ErrorCategory.A

    def test_tab_error(self) -> None:
        stderr = 'TabError: inconsistent use of tabs and spaces in indentation\n'
        assert classify_error(stderr) == ErrorCategory.A

    def test_import_error(self) -> None:
        stderr = 'ImportError: cannot import name "foo" from "bar"\n'
        assert classify_error(stderr) == ErrorCategory.A

    def test_module_not_found_error(self) -> None:
        stderr = "ModuleNotFoundError: No module named 'nonexistent'\n"
        assert classify_error(stderr) == ErrorCategory.A

    def test_syntax_error_in_traceback(self) -> None:
        """SyntaxError with full traceback context."""
        stderr = (
            '  File "/tmp/script.py", line 42\n'
            '    data = [\n'
            '           ^\n'
            "SyntaxError: '[' was never closed\n"
        )
        assert classify_error(stderr) == ErrorCategory.A

    def test_syntax_error_beats_timeout_keyword(self) -> None:
        """Priority: A takes precedence even if 'Timeout' appears elsewhere."""
        stderr = (
            "# wait_for_selector: Timeout 5000ms exceeded\n"
            "SyntaxError: invalid syntax\n"
        )
        assert classify_error(stderr) == ErrorCategory.A


# ══════════════════════════════════════════════════════════════
#  Category B — Runtime crashes (catch-all)
# ══════════════════════════════════════════════════════════════


class TestCategoryB:
    """Python runtime errors, JS evaluate errors, output-stage errors."""

    def test_attribute_error(self) -> None:
        stderr = "AttributeError: 'NoneType' object has no attribute 'text_content'\n"
        assert classify_error(stderr) == ErrorCategory.B

    def test_type_error(self) -> None:
        stderr = "TypeError: int() argument must be a string\n"
        assert classify_error(stderr) == ErrorCategory.B

    def test_key_error(self) -> None:
        stderr = "KeyError: 'nonexistent_key'\n"
        assert classify_error(stderr) == ErrorCategory.B

    def test_index_error(self) -> None:
        stderr = "IndexError: list index out of range\n"
        assert classify_error(stderr) == ErrorCategory.B

    def test_value_error(self) -> None:
        stderr = "ValueError: invalid literal for int()\n"
        assert classify_error(stderr) == ErrorCategory.B

    def test_name_error(self) -> None:
        stderr = "NameError: name 'undefined_var' is not defined\n"
        assert classify_error(stderr) == ErrorCategory.B

    def test_unbound_local_error(self) -> None:
        stderr = "UnboundLocalError: local variable 'x' referenced before assignment\n"
        assert classify_error(stderr) == ErrorCategory.B

    def test_recursion_error(self) -> None:
        stderr = "RecursionError: maximum recursion depth exceeded\n"
        assert classify_error(stderr) == ErrorCategory.B

    def test_zero_division_error(self) -> None:
        stderr = "ZeroDivisionError: division by zero\n"
        assert classify_error(stderr) == ErrorCategory.B

    def test_overflow_error(self) -> None:
        stderr = "OverflowError: (34, 'Result too large')\n"
        assert classify_error(stderr) == ErrorCategory.B

    def test_stop_iteration(self) -> None:
        stderr = "StopIteration\n"
        # StopIteration doesn't end in Error, but the catch-all handles it.
        assert classify_error(stderr) == ErrorCategory.B

    def test_assertion_error(self) -> None:
        stderr = "AssertionError: expected True\n"
        assert classify_error(stderr) == ErrorCategory.B

    def test_file_not_found_error(self) -> None:
        stderr = "FileNotFoundError: [Errno 2] No such file or directory: 'data.json'\n"
        assert classify_error(stderr) == ErrorCategory.B

    def test_permission_error(self) -> None:
        stderr = "PermissionError: [Errno 13] Permission denied\n"
        assert classify_error(stderr) == ErrorCategory.B

    def test_connection_error_in_script(self) -> None:
        """Script makes direct network calls (urllib, requests)."""
        stderr = "ConnectionError: Connection refused\n"
        assert classify_error(stderr) == ErrorCategory.B

    def test_unicode_decode_error(self) -> None:
        stderr = "UnicodeDecodeError: 'utf-8' codec can't decode byte\n"
        assert classify_error(stderr) == ErrorCategory.B

    def test_json_decode_error(self) -> None:
        stderr = "json.decoder.JSONDecodeError: Expecting value: line 1 column 1\n"
        assert classify_error(stderr) == ErrorCategory.B

    def test_re_error(self) -> None:
        stderr = "re.error: bad character range a-Z\n"
        assert classify_error(stderr) == ErrorCategory.B

    def test_asyncio_cancelled_error(self) -> None:
        stderr = "asyncio.exceptions.CancelledError\n"
        assert classify_error(stderr) == ErrorCategory.B

    def test_js_type_error_via_evaluate(self) -> None:
        """JS TypeError through page.evaluate — Category B, not D."""
        stderr = (
            "patchright._impl._errors.Error: Page.evaluate: TypeError: "
            "Cannot read properties of null (reading 'textContent')\n"
        )
        assert classify_error(stderr) == ErrorCategory.B

    def test_js_reference_error_via_evaluate(self) -> None:
        stderr = (
            "patchright._impl._errors.Error: Page.evaluate: ReferenceError: "
            "undefinedVariable is not defined\n"
        )
        assert classify_error(stderr) == ErrorCategory.B

    def test_empty_output_no_stderr(self) -> None:
        """Script returned None — empty stderr, exit 0, no stdout."""
        assert classify_error("", exit_code=0, stdout="") == ErrorCategory.B

    def test_null_return_value(self) -> None:
        """Script returned null via return value markers."""
        stdout = "__SCOUT_RETURN_VALUE_START_a7b3__null__SCOUT_RETURN_VALUE_END_a7b3__"
        assert classify_error("", exit_code=0, stdout=stdout) == ErrorCategory.B

    def test_script_no_output_file(self) -> None:
        stderr = "Script did not produce output file\n"
        assert classify_error(stderr) == ErrorCategory.B

    def test_script_empty_output(self) -> None:
        stderr = "Script produced empty output file\n"
        assert classify_error(stderr) == ErrorCategory.B

    def test_script_invalid_json(self) -> None:
        stderr = "Script output is not valid JSON\n"
        assert classify_error(stderr) == ErrorCategory.B

    def test_api_misuse_strict_mode_is_E_not_B(self) -> None:
        """Strict mode violation is Category E, not B."""
        stderr = (
            "patchright._impl._errors.Error: Locator.click: Error: "
            "strict mode violation:  resolved to 3 elements:\n"
            '    1) <div class="item">1</div>\n'
        )
        assert classify_error(stderr) == ErrorCategory.E

    def test_api_misuse_locator_resolution_failure(self) -> None:
        """Locator that can't resolve — Category B (not timeout)."""
        stderr = (
            "patchright._impl._errors.Error: Page.click: "
            "Could not resolve .nonexistent to DOM Element\n"
        )
        assert classify_error(stderr) == ErrorCategory.B

    def test_catch_all_unknown_error(self) -> None:
        """Completely unknown error falls to B catch-all."""
        stderr = "SomeBizarreThirdPartyException: something weird happened\n"
        assert classify_error(stderr) == ErrorCategory.B

    def test_empty_stderr_with_nonzero_exit(self) -> None:
        """Empty stderr but non-zero exit — still Category B catch-all."""
        assert classify_error("", exit_code=1) == ErrorCategory.B


# ══════════════════════════════════════════════════════════════
#  Category C — Network/server failure
# ══════════════════════════════════════════════════════════════


class TestCategoryC:
    """net::ERR_*, navigation timeouts."""

    def test_dns_not_resolved(self) -> None:
        stderr = "Page.goto: net::ERR_NAME_NOT_RESOLVED at https://example.com/\n"
        assert classify_error(stderr) == ErrorCategory.C

    def test_connection_refused(self) -> None:
        stderr = "Page.goto: net::ERR_CONNECTION_REFUSED at http://localhost:9999/\n"
        assert classify_error(stderr) == ErrorCategory.C

    def test_connection_timed_out(self) -> None:
        stderr = "Page.goto: net::ERR_CONNECTION_TIMED_OUT at https://slow.example.com/\n"
        assert classify_error(stderr) == ErrorCategory.C

    def test_connection_reset(self) -> None:
        stderr = "Page.goto: net::ERR_CONNECTION_RESET at https://example.com/\n"
        assert classify_error(stderr) == ErrorCategory.C

    def test_ssl_protocol_error(self) -> None:
        stderr = "Page.goto: net::ERR_SSL_PROTOCOL_ERROR at https://example.com/\n"
        assert classify_error(stderr) == ErrorCategory.C

    def test_cert_date_invalid(self) -> None:
        stderr = "Page.goto: net::ERR_CERT_DATE_INVALID at https://expired.badssl.com/\n"
        assert classify_error(stderr) == ErrorCategory.C

    def test_too_many_redirects(self) -> None:
        stderr = "Page.goto: net::ERR_TOO_MANY_REDIRECTS at http://example.com/loop\n"
        assert classify_error(stderr) == ErrorCategory.C

    def test_empty_response(self) -> None:
        stderr = "Page.goto: net::ERR_EMPTY_RESPONSE at http://example.com/\n"
        assert classify_error(stderr) == ErrorCategory.C

    def test_internet_disconnected(self) -> None:
        stderr = "Page.goto: net::ERR_INTERNET_DISCONNECTED\n"
        assert classify_error(stderr) == ErrorCategory.C

    def test_aborted(self) -> None:
        stderr = "Page.goto: net::ERR_ABORTED at chrome://crash\n"
        assert classify_error(stderr) == ErrorCategory.C

    def test_blocked_by_client(self) -> None:
        stderr = "net::ERR_BLOCKED_BY_CLIENT\n"
        assert classify_error(stderr) == ErrorCategory.C

    def test_blocked_by_response(self) -> None:
        stderr = "net::ERR_BLOCKED_BY_RESPONSE\n"
        assert classify_error(stderr) == ErrorCategory.C

    def test_http2_protocol_error(self) -> None:
        stderr = "net::ERR_HTTP2_PROTOCOL_ERROR\n"
        assert classify_error(stderr) == ErrorCategory.C

    def test_failed(self) -> None:
        stderr = "net::ERR_FAILED\n"
        assert classify_error(stderr) == ErrorCategory.C

    def test_navigation_timeout_page_goto(self) -> None:
        stderr = (
            "patchright._impl._errors.TimeoutError: Page.goto: "
            "Timeout 30000ms exceeded.\n"
        )
        assert classify_error(stderr) == ErrorCategory.C

    def test_navigation_timeout_page_reload(self) -> None:
        stderr = "Page.reload: Timeout 30000ms exceeded.\n"
        assert classify_error(stderr) == ErrorCategory.C

    def test_navigation_timeout_page_go_back(self) -> None:
        stderr = "Page.go_back: Timeout 30000ms exceeded.\n"
        assert classify_error(stderr) == ErrorCategory.C

    def test_navigation_timeout_page_go_forward(self) -> None:
        stderr = "Page.go_forward: Timeout 30000ms exceeded.\n"
        assert classify_error(stderr) == ErrorCategory.C

    def test_navigation_timeout_frame_goto(self) -> None:
        stderr = "Frame.goto: Timeout 5000ms exceeded.\n"
        assert classify_error(stderr) == ErrorCategory.C

    def test_net_err_in_non_goto_context_still_C(self) -> None:
        """net::ERR_* is always C, even from non-navigation methods."""
        stderr = (
            "patchright._impl._errors.Error: Page.click: "
            "net::ERR_CONNECTION_RESET\n"
        )
        assert classify_error(stderr) == ErrorCategory.C

    def test_net_err_with_patchright_prefix(self) -> None:
        """Full patchright prefix with net::ERR_*."""
        stderr = (
            "patchright._impl._errors.Error: Page.goto: "
            "net::ERR_NAME_NOT_RESOLVED at https://example.com/\n"
            "Call log:\n"
            '  - navigating to "https://example.com/"\n'
        )
        assert classify_error(stderr) == ErrorCategory.C


# ══════════════════════════════════════════════════════════════
#  Category D — Post-navigation timeouts
# ══════════════════════════════════════════════════════════════


class TestCategoryD:
    """Element wait timeouts, evaluation timeouts, behavioral wait timeouts."""

    def test_wait_for_selector(self) -> None:
        stderr = "Page.wait_for_selector: Timeout 5000ms exceeded.\n"
        assert classify_error(stderr) == ErrorCategory.D

    def test_locator_wait_for(self) -> None:
        stderr = "Locator.wait_for: Timeout 5000ms exceeded.\n"
        assert classify_error(stderr) == ErrorCategory.D

    def test_click_timeout_no_call_log(self) -> None:
        """click timeout without E-specific call log = Category D."""
        stderr = (
            "patchright._impl._errors.TimeoutError: Page.click: "
            "Timeout 5000ms exceeded.\n"
            "Call log:\n"
            '  - waiting for locator(".nonexistent")\n'
        )
        assert classify_error(stderr) == ErrorCategory.D

    def test_fill_timeout(self) -> None:
        stderr = "Page.fill: Timeout 5000ms exceeded.\n"
        assert classify_error(stderr) == ErrorCategory.D

    def test_hover_timeout(self) -> None:
        stderr = "Page.hover: Timeout 5000ms exceeded.\n"
        assert classify_error(stderr) == ErrorCategory.D

    def test_dblclick_timeout(self) -> None:
        stderr = "Page.dblclick: Timeout 5000ms exceeded.\n"
        assert classify_error(stderr) == ErrorCategory.D

    def test_select_option_timeout(self) -> None:
        stderr = "Page.select_option: Timeout 5000ms exceeded.\n"
        assert classify_error(stderr) == ErrorCategory.D

    def test_check_timeout(self) -> None:
        stderr = "Page.check: Timeout 5000ms exceeded.\n"
        assert classify_error(stderr) == ErrorCategory.D

    def test_type_timeout(self) -> None:
        stderr = "Page.type: Timeout 5000ms exceeded.\n"
        assert classify_error(stderr) == ErrorCategory.D

    def test_press_timeout(self) -> None:
        stderr = "Page.press: Timeout 5000ms exceeded.\n"
        assert classify_error(stderr) == ErrorCategory.D

    def test_text_content_timeout(self) -> None:
        stderr = "Page.text_content: Timeout 5000ms exceeded.\n"
        assert classify_error(stderr) == ErrorCategory.D

    def test_inner_text_timeout(self) -> None:
        stderr = "Page.inner_text: Timeout 5000ms exceeded.\n"
        assert classify_error(stderr) == ErrorCategory.D

    def test_inner_html_timeout(self) -> None:
        stderr = "Page.inner_html: Timeout 5000ms exceeded.\n"
        assert classify_error(stderr) == ErrorCategory.D

    def test_get_attribute_timeout(self) -> None:
        stderr = "Page.get_attribute: Timeout 5000ms exceeded.\n"
        assert classify_error(stderr) == ErrorCategory.D

    def test_evaluate_timeout(self) -> None:
        stderr = "Page.evaluate: Timeout 30000ms exceeded.\n"
        assert classify_error(stderr) == ErrorCategory.D

    def test_wait_for_function_timeout(self) -> None:
        stderr = "Page.wait_for_function: Timeout 5000ms exceeded.\n"
        assert classify_error(stderr) == ErrorCategory.D

    def test_wait_for_response_timeout(self) -> None:
        stderr = "Page.wait_for_response: Timeout 5000ms exceeded.\n"
        assert classify_error(stderr) == ErrorCategory.D

    def test_expect_navigation_timeout(self) -> None:
        stderr = "Page.expect_navigation: Timeout 5000ms exceeded.\n"
        assert classify_error(stderr) == ErrorCategory.D

    def test_expect_popup_timeout(self) -> None:
        stderr = "Page.expect_popup: Timeout 5000ms exceeded.\n"
        assert classify_error(stderr) == ErrorCategory.D

    def test_expect_download_timeout(self) -> None:
        stderr = "Page.expect_download: Timeout 5000ms exceeded.\n"
        assert classify_error(stderr) == ErrorCategory.D

    def test_wait_for_load_state_post_nav(self) -> None:
        """wait_for_load_state after successful nav = Category D."""
        stderr = "Page.wait_for_load_state: Timeout 5000ms exceeded.\n"
        assert classify_error(stderr) == ErrorCategory.D

    def test_frame_click_timeout(self) -> None:
        """Frame-level methods produce D, not C."""
        stderr = "Frame.click: Timeout 5000ms exceeded.\n"
        assert classify_error(stderr) == ErrorCategory.D

    def test_element_handle_fill_timeout(self) -> None:
        stderr = "ElementHandle.fill: Timeout 5000ms exceeded.\n"
        assert classify_error(stderr) == ErrorCategory.D

    def test_locator_click_timeout_no_e_pattern(self) -> None:
        """Locator timeout without E-specific call log = D."""
        stderr = (
            "patchright._impl._errors.TimeoutError: "
            "Locator.click: Timeout 5000ms exceeded.\n"
            "Call log:\n"
            '  - waiting for locator(".absent")\n'
        )
        assert classify_error(stderr) == ErrorCategory.D


# ══════════════════════════════════════════════════════════════
#  Category E — Page state prevented interaction
# ══════════════════════════════════════════════════════════════


class TestCategoryE:
    """Call log patterns, non-timeout E errors, disambiguation from D."""

    def test_pointer_intercept_in_call_log(self) -> None:
        """Timeout with 'intercepts pointer events' in call log = E, not D."""
        stderr = (
            "patchright._impl._errors.TimeoutError: Page.click: "
            "Timeout 5000ms exceeded.\n"
            "Call log:\n"
            "  - waiting for locator(\"#target\")\n"
            "  - attempting click action\n"
            "    - element is visible, enabled and stable\n"
            "    - <div></div> intercepts pointer events\n"
        )
        assert classify_error(stderr) == ErrorCategory.E

    def test_not_visible_in_call_log(self) -> None:
        stderr = (
            "patchright._impl._errors.TimeoutError: Page.click: "
            "Timeout 5000ms exceeded.\n"
            "Call log:\n"
            "  - element is not visible\n"
        )
        assert classify_error(stderr) == ErrorCategory.E

    def test_not_enabled_in_call_log(self) -> None:
        stderr = (
            "Locator.fill: Timeout 5000ms exceeded.\n"
            "Call log:\n"
            "  - element is not enabled\n"
        )
        assert classify_error(stderr) == ErrorCategory.E

    def test_not_stable_in_call_log(self) -> None:
        stderr = (
            "Page.click: Timeout 5000ms exceeded.\n"
            "Call log:\n"
            "  - element is not stable\n"
        )
        assert classify_error(stderr) == ErrorCategory.E

    def test_outside_viewport_in_call_log(self) -> None:
        stderr = (
            "Page.click: Timeout 5000ms exceeded.\n"
            "Call log:\n"
            "  - Element is outside of the viewport\n"
        )
        assert classify_error(stderr) == ErrorCategory.E

    def test_detached_in_call_log(self) -> None:
        stderr = (
            "Page.click: Timeout 5000ms exceeded.\n"
            "Call log:\n"
            "  - Element is not attached to the DOM\n"
        )
        assert classify_error(stderr) == ErrorCategory.E

    def test_node_detached_in_call_log(self) -> None:
        stderr = (
            "Page.click: Timeout 5000ms exceeded.\n"
            "Call log:\n"
            "  - Node is detached\n"
        )
        assert classify_error(stderr) == ErrorCategory.E

    def test_checkbox_unchanged_in_call_log(self) -> None:
        stderr = (
            "Page.check: Timeout 5000ms exceeded.\n"
            "Call log:\n"
            "  - Clicking the checkbox did not change its state\n"
        )
        assert classify_error(stderr) == ErrorCategory.E

    def test_wrong_element_type_in_call_log(self) -> None:
        stderr = (
            "Page.fill: Timeout 5000ms exceeded.\n"
            "Call log:\n"
            "  - Element is not an <input>\n"
        )
        assert classify_error(stderr) == ErrorCategory.E

    def test_execution_context_destroyed(self) -> None:
        stderr = (
            "patchright._impl._errors.Error: Page.evaluate: "
            "Execution context was destroyed, most likely because of a navigation.\n"
        )
        assert classify_error(stderr) == ErrorCategory.E

    def test_frame_detached(self) -> None:
        stderr = (
            "patchright._impl._errors.Error: Frame.evaluate: "
            "Frame was detached\n"
        )
        assert classify_error(stderr) == ErrorCategory.E

    def test_navigating_frame_detached(self) -> None:
        stderr = "Navigating frame was detached\n"
        assert classify_error(stderr) == ErrorCategory.E

    def test_navigation_interrupted(self) -> None:
        stderr = "Navigation interrupted by another navigation\n"
        assert classify_error(stderr) == ErrorCategory.E

    def test_frame_attempting_navigation(self) -> None:
        stderr = "Frame is currently attempting a navigation\n"
        assert classify_error(stderr) == ErrorCategory.E

    def test_content_during_navigation(self) -> None:
        stderr = "Unable to retrieve content because the page is navigating\n"
        assert classify_error(stderr) == ErrorCategory.E

    def test_frame_not_available(self) -> None:
        stderr = "Frame for this navigation request is not available\n"
        assert classify_error(stderr) == ErrorCategory.E

    def test_strict_mode_violation(self) -> None:
        stderr = (
            "patchright._impl._errors.Error: Locator.click: Error: "
            "strict mode violation:  resolved to 3 elements:\n"
            '    1) <div class="item">1</div>\n'
            '    2) <div class="item">2</div>\n'
            '    3) <div class="item">3</div>\n'
        )
        assert classify_error(stderr) == ErrorCategory.E

    def test_dialog_blocking_patchright_format(self) -> None:
        stderr = (
            "patchright._impl._errors.Error: Page.evaluate: "
            'Cannot evaluate, page has an open JavaScript dialog: '
            'alert with message "blocking dialog"\n'
        )
        assert classify_error(stderr) == ErrorCategory.E

    def test_dialog_blocking_old_format(self) -> None:
        stderr = "Open JavaScript dialog prevents evaluation\n"
        assert classify_error(stderr) == ErrorCategory.E

    def test_e_takes_priority_over_d_for_timeout(self) -> None:
        """When timeout has E patterns in call log, it's E not D."""
        stderr = (
            "patchright._impl._errors.TimeoutError: Page.click: "
            "Timeout 5000ms exceeded.\n"
            "Call log:\n"
            "  2 x waiting for element to be visible, enabled and stable\n"
            "    - element is visible, enabled and stable\n"
            "    - scrolling into view if needed\n"
            "    - done scrolling\n"
            "    - <div></div> intercepts pointer events\n"
            "  - retrying click action\n"
        )
        assert classify_error(stderr) == ErrorCategory.E

    def test_fill_timeout_with_not_visible_is_E(self) -> None:
        """fill timeout with 'not visible' in call log = E."""
        stderr = (
            "Locator.fill: Timeout 5000ms exceeded.\n"
            "Call log:\n"
            "  2 x waiting for element to be visible, enabled and stable\n"
            "    - element is not visible\n"
        )
        assert classify_error(stderr) == ErrorCategory.E


# ══════════════════════════════════════════════════════════════
#  Category F — Process/browser death
# ══════════════════════════════════════════════════════════════


class TestCategoryF:
    """F1 (browser crash), F2 (subprocess timeout), F3 (infrastructure)."""

    # -- F1 --

    def test_page_crashed(self) -> None:
        stderr = "Page crashed\n"
        assert classify_error(stderr) == ErrorCategory.F1

    def test_target_crashed(self) -> None:
        stderr = "Target crashed\n"
        assert classify_error(stderr) == ErrorCategory.F1

    def test_target_closed(self) -> None:
        stderr = (
            "patchright._impl._errors.TargetClosedError: "
            "Page.evaluate: Target page, context or browser has been closed\n"
        )
        assert classify_error(stderr) == ErrorCategory.F1

    def test_browser_closed(self) -> None:
        stderr = "Browser has been closed\n"
        assert classify_error(stderr) == ErrorCategory.F1

    def test_browser_closed_alt(self) -> None:
        stderr = "Browser closed\n"
        assert classify_error(stderr) == ErrorCategory.F1

    def test_page_closed(self) -> None:
        stderr = "Page closed\n"
        assert classify_error(stderr) == ErrorCategory.F1

    def test_context_closed(self) -> None:
        stderr = "Context closed\n"
        assert classify_error(stderr) == ErrorCategory.F1

    def test_navigation_crashed(self) -> None:
        stderr = "Navigation failed because page crashed!\n"
        assert classify_error(stderr) == ErrorCategory.F1

    def test_connection_closed_driver(self) -> None:
        stderr = "Connection closed while reading from the driver\n"
        assert classify_error(stderr) == ErrorCategory.F1

    def test_playwright_connection_closed(self) -> None:
        stderr = "Playwright connection closed\n"
        assert classify_error(stderr) == ErrorCategory.F1

    def test_socket_closed(self) -> None:
        stderr = "Socket closed\n"
        assert classify_error(stderr) == ErrorCategory.F1

    def test_socket_error(self) -> None:
        stderr = "Socket error\n"
        assert classify_error(stderr) == ErrorCategory.F1

    def test_gc_collected(self) -> None:
        stderr = "The object has been collected to prevent unbounded heap growth.\n"
        assert classify_error(stderr) == ErrorCategory.F1

    def test_sigsegv_exit_code(self) -> None:
        assert classify_error("", exit_code=139) == ErrorCategory.F1

    def test_sigabrt_exit_code(self) -> None:
        assert classify_error("", exit_code=134) == ErrorCategory.F1

    def test_sigbus_exit_code(self) -> None:
        assert classify_error("", exit_code=135) == ErrorCategory.F1

    def test_negative_sigsegv(self) -> None:
        """asyncio subprocess reports -11 for SIGSEGV."""
        assert classify_error("", exit_code=-11) == ErrorCategory.F1

    def test_negative_sigabrt(self) -> None:
        assert classify_error("", exit_code=-6) == ErrorCategory.F1

    def test_negative_sigbus(self) -> None:
        assert classify_error("", exit_code=-7) == ErrorCategory.F1

    def test_f1_beats_timeout_in_stderr(self) -> None:
        """F1 takes priority over timeout patterns."""
        stderr = (
            "Page crashed\n"
            "Page.click: Timeout 5000ms exceeded.\n"
        )
        assert classify_error(stderr) == ErrorCategory.F1

    # -- F2 --

    def test_script_execution_timeout(self) -> None:
        stderr = "Script execution timed out after 600 seconds\n"
        assert classify_error(stderr) == ErrorCategory.F2

    def test_function_timeout(self) -> None:
        stderr = "Function timed out after 10 seconds\n"
        assert classify_error(stderr) == ErrorCategory.F2

    # -- F3 --

    def test_executable_doesnt_exist(self) -> None:
        stderr = (
            "patchright._impl._errors.Error: "
            "BrowserType.launch_persistent_context: "
            "Executable doesn't exist at /path/to/chrome\n"
        )
        assert classify_error(stderr) == ErrorCategory.F3

    def test_missing_dependencies(self) -> None:
        stderr = "Host system is missing dependencies to run browsers.\n"
        assert classify_error(stderr) == ErrorCategory.F3

    def test_no_display(self) -> None:
        stderr = "Cannot open display\n"
        assert classify_error(stderr) == ErrorCategory.F3

    def test_no_space(self) -> None:
        stderr = "No space left on device\n"
        assert classify_error(stderr) == ErrorCategory.F3

    def test_too_many_files(self) -> None:
        stderr = "Too many open files\n"
        assert classify_error(stderr) == ErrorCategory.F3

    def test_emfile(self) -> None:
        stderr = "OSError: [Errno 24] EMFILE\n"
        assert classify_error(stderr) == ErrorCategory.F3

    def test_no_sandbox(self) -> None:
        stderr = "No usable sandbox\n"
        assert classify_error(stderr) == ErrorCategory.F3

    def test_event_loop_closed(self) -> None:
        stderr = "Event loop is closed! Is Playwright already stopped?\n"
        assert classify_error(stderr) == ErrorCategory.F3

    def test_max_argument_depth(self) -> None:
        stderr = "Maximum argument depth exceeded\n"
        assert classify_error(stderr) == ErrorCategory.F3

    def test_dev_shm(self) -> None:
        stderr = "/dev/shm too small\n"
        assert classify_error(stderr) == ErrorCategory.F3

    def test_missing_chrome_libs(self) -> None:
        stderr = "error while loading shared libraries: libnss3.so\n"
        assert classify_error(stderr) == ErrorCategory.F3

    def test_mkdtemp_failure(self) -> None:
        stderr = "mkdtemp: No space left on device\n"
        assert classify_error(stderr) == ErrorCategory.F3

    def test_sigkill_exit_code(self) -> None:
        """Exit code 137 (SIGKILL/OOM) is F3, not F1."""
        assert classify_error("", exit_code=137) == ErrorCategory.F3

    def test_negative_sigkill(self) -> None:
        assert classify_error("", exit_code=-9) == ErrorCategory.F3

    def test_browser_launch_f3(self) -> None:
        """BrowserType.launch errors are F3 (infrastructure)."""
        stderr = (
            "patchright._impl._errors.Error: "
            "BrowserType.launch: Browser closed.\n"
            "Looks like Playwright was just installed.\n"
        )
        assert classify_error(stderr) == ErrorCategory.F3


# ══════════════════════════════════════════════════════════════
#  Category G — Schema validation failure
# ══════════════════════════════════════════════════════════════


class TestCategoryG:
    """Schema validation errors (detected externally)."""

    def test_schema_error_no_stderr(self) -> None:
        assert classify_error(
            "", exit_code=0, schema_error="Expected at least 5 items, got 0",
        ) == ErrorCategory.G

    def test_schema_error_with_empty_stderr(self) -> None:
        assert classify_error(
            "  \n", exit_code=0, schema_error="Missing field: price",
        ) == ErrorCategory.G

    def test_schema_error_none_exit(self) -> None:
        """In-process execution has no exit code."""
        assert classify_error(
            "", exit_code=None, schema_error="Wrong type for field 'price'",
        ) == ErrorCategory.G

    def test_crash_plus_schema_error_is_crash(self) -> None:
        """If the script crashed AND has schema_error, classify by crash."""
        stderr = "AttributeError: 'NoneType' object has no attribute 'x'\n"
        result = classify_error(
            stderr, exit_code=1, schema_error="Expected at least 5 items",
        )
        # The crash is the real error; schema_error shouldn't override it.
        assert result == ErrorCategory.B


# ══════════════════════════════════════════════════════════════
#  Priority ordering tests
# ══════════════════════════════════════════════════════════════


class TestPriorityOrdering:
    """Verify priority order when multiple patterns match."""

    def test_a_beats_everything(self) -> None:
        """Category A has highest priority."""
        stderr = (
            "net::ERR_CONNECTION_REFUSED\n"
            "Page crashed\n"
            "SyntaxError: invalid syntax\n"
        )
        assert classify_error(stderr) == ErrorCategory.A

    def test_f_beats_c(self) -> None:
        """F takes priority over C."""
        stderr = (
            "net::ERR_CONNECTION_REFUSED\n"
            "Page crashed\n"
        )
        assert classify_error(stderr) == ErrorCategory.F1

    def test_c_beats_d(self) -> None:
        """C (net::ERR_*) takes priority over D (timeout)."""
        stderr = (
            "Page.click: Timeout 5000ms exceeded.\n"
            "net::ERR_CONNECTION_RESET\n"
        )
        assert classify_error(stderr) == ErrorCategory.C

    def test_e_beats_d_via_call_log(self) -> None:
        """E patterns in call log take priority over D."""
        stderr = (
            "Page.hover: Timeout 5000ms exceeded.\n"
            "Call log:\n"
            "  - element is not visible\n"
        )
        assert classify_error(stderr) == ErrorCategory.E

    def test_f2_beats_f1(self) -> None:
        """F2 (subprocess timeout) takes priority over F1 patterns."""
        stderr = (
            "Function timed out after 600 seconds\n"
            "Page crashed\n"
        )
        assert classify_error(stderr) == ErrorCategory.F2


# ══════════════════════════════════════════════════════════════
#  Route handler wrapper stripping
# ══════════════════════════════════════════════════════════════


class TestRouteHandlerWrapper:
    """Errors wrapped by route callback handlers."""

    def test_strip_route_wrapper(self) -> None:
        """Route wrapper is stripped, underlying error is classified.

        Real Patchright format: the error line appears BEFORE the wrapper,
        then the wrapper text, then the call log.
        """
        stderr = (
            "patchright._impl._errors.TimeoutError: Page.click: "
            "Timeout 5000ms exceeded.\n"
            '"Page.click: Timeout 5000ms exceeded." '
            "while running route callback.\n"
            "Consider awaiting `page.unroute_all(behavior='ignoreErrors')` "
            "before the end of the test to ignore remaining routes in flight.\n"
            "Call log:\n"
            '  - waiting for locator(".absent")\n'
        )
        assert classify_error(stderr) == ErrorCategory.D

    def test_strip_route_wrapper_with_e_pattern(self) -> None:
        stderr = (
            "patchright._impl._errors.TimeoutError: Page.click: "
            "Timeout 5000ms exceeded.\n"
            '"Page.click: Timeout 5000ms exceeded." '
            "while running route callback.\n"
            "Consider awaiting `page.unroute_all(behavior='ignoreErrors')` "
            "before the end of the test to ignore remaining routes in flight.\n"
            "Call log:\n"
            "  - element is not visible\n"
        )
        assert classify_error(stderr) == ErrorCategory.E


# ══════════════════════════════════════════════════════════════
#  Edge cases
# ══════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Unusual inputs, empty strings, encoding artifacts."""

    def test_empty_stderr_exit_0_no_schema(self) -> None:
        """Empty stderr, exit 0, no schema = B (empty output)."""
        assert classify_error("", exit_code=0) == ErrorCategory.B

    def test_empty_stderr_exit_none_no_schema(self) -> None:
        """No stderr, no exit code, no schema = B catch-all."""
        assert classify_error("") == ErrorCategory.B

    def test_binary_garbage_in_stderr(self) -> None:
        """Binary data that doesn't match any pattern = B catch-all."""
        stderr = "\x00\xff\xfe binary garbage \x01\x02\x03"
        assert classify_error(stderr) == ErrorCategory.B

    def test_very_long_stderr(self) -> None:
        """Very long stderr should not cause performance issues."""
        stderr = "x" * 100_000 + "\nAttributeError: some error\n"
        assert classify_error(stderr) == ErrorCategory.B

    def test_multiline_traceback_with_timeout(self) -> None:
        """Full traceback with timeout — the timeout line matters."""
        stderr = (
            "Traceback (most recent call last):\n"
            '  File "/tmp/script.py", line 42, in scrape\n'
            "    await page.click('.btn', timeout=5000)\n"
            "  File \"patchright/...\", line 100, in click\n"
            "    ...\n"
            "patchright._impl._errors.TimeoutError: Page.click: "
            "Timeout 5000ms exceeded.\n"
            "Call log:\n"
            '  - waiting for locator(".btn")\n'
        )
        assert classify_error(stderr) == ErrorCategory.D

    def test_playwright_prefix_also_works(self) -> None:
        """playwright prefix (not just patchright) should also match."""
        stderr = (
            "playwright._impl._errors.TargetClosedError: "
            "Page.evaluate: Target page, context or browser has been closed\n"
        )
        assert classify_error(stderr) == ErrorCategory.F1


# ══════════════════════════════════════════════════════════════
#  Review-discovered edge cases (Phase 1 review)
# ══════════════════════════════════════════════════════════════


class TestReviewEdgeCases:
    """Edge cases discovered during Phase 1 code review."""

    def test_connection_timed_out_in_call_log_not_f3(self) -> None:
        """'Connection timed out' in call log should NOT trigger F3.

        The F3 pattern 'Connection timed out' is scoped to BrowserType
        launch context to avoid false positives.
        """
        stderr = (
            "Page.click: Timeout 5000ms exceeded.\n"
            "Call log:\n"
            "  Connection timed out while loading resource\n"
        )
        assert classify_error(stderr) == ErrorCategory.D

    def test_connection_timed_out_in_launch_context_is_f3(self) -> None:
        """'Connection timed out' in BrowserType.launch context IS F3."""
        stderr = (
            "patchright._impl._errors.Error: "
            "BrowserType.launch_persistent_context: Connection timed out\n"
        )
        assert classify_error(stderr) == ErrorCategory.F3

    def test_navigation_timeout_with_e_pattern_in_call_log_is_c(self) -> None:
        """Page.goto timeout with E pattern in call log is still C (nav wins)."""
        stderr = (
            "Page.goto: Timeout 30000ms exceeded.\n"
            "Call log:\n"
            "  - element is not visible\n"
        )
        # C is checked at Priority 3, before E at Priority 4
        assert classify_error(stderr) == ErrorCategory.C

    def test_e_call_log_pattern_outside_call_log_is_d(self) -> None:
        """E patterns outside the Call log: section should not trigger E."""
        stderr = (
            "patchright._impl._errors.TimeoutError: Page.click: "
            "Timeout 5000ms exceeded.\n"
            "  - some context: intercepts pointer events\n"
            # No "Call log:" section — E patterns should not match
        )
        # No Call log: header, so E call log check returns None
        assert classify_error(stderr) == ErrorCategory.D

    def test_browser_type_launch_with_browser_closed_is_f3(self) -> None:
        """BrowserType.launch with 'Browser closed' is F3, not F1."""
        stderr = (
            "patchright._impl._errors.Error: "
            "BrowserType.launch: Browser closed.\n"
            "Looks like Playwright was just installed.\n"
        )
        assert classify_error(stderr) == ErrorCategory.F3

    def test_schema_error_with_only_whitespace_stderr(self) -> None:
        """Schema error with whitespace-only stderr = Category G."""
        assert classify_error(
            "   \n  \t  \n", exit_code=0,
            schema_error="Expected at least 5 items, got 0",
        ) == ErrorCategory.G

    def test_f3_connection_timed_out_standalone_is_not_f3(self) -> None:
        """Standalone 'Connection timed out' without BrowserType is NOT F3."""
        stderr = "Some error: Connection timed out\n"
        # Should fall to B catch-all, not F3
        assert classify_error(stderr) == ErrorCategory.B
