"""Error fingerprinter — extracts structured fingerprints for cross-attempt comparison.

Parses raw error output into a ``Fingerprint`` dataclass with category,
error_type, method, target, and the full raw message. Fingerprints are
compared at three levels of specificity to assess stability across
diagnostic attempts.

Spec reference: docs/specific/AUTO_FIX_ALGORITHM.md S5
"""

from __future__ import annotations

import re

from scout.autofix.types import ComparisonLevel, ErrorCategory, Fingerprint

# ── Compiled regex patterns ───────────────────────────────────

# Extract the Patchright/Playwright method prefix from error messages.
# Matches patterns like "Page.click:", "Locator.wait_for:", "Frame.evaluate:"
# Also matches "ElementHandle.click:" etc.
_PW_METHOD_RE = re.compile(
    r"(?:Page|Locator|Frame|ElementHandle|BrowserContext|Browser|BrowserType)"
    r"\.\w+",
)

# Extract the full "{Method}: {ErrorMessage}" line from Patchright errors.
# e.g., "Page.evaluate: TypeError: Cannot read properties of null ..."
# e.g., "Page.click: Timeout 5000ms exceeded."
# e.g., "Page.goto: net::ERR_NAME_NOT_RESOLVED at https://..."
_PW_ERROR_LINE_RE = re.compile(
    r"(?:patchright|playwright)\._impl\._errors\.\w+:\s*"
    r"((?:Page|Locator|Frame|ElementHandle|BrowserContext|Browser|BrowserType)"
    r"\.\w+):\s*(.*)",
    re.DOTALL,
)

# Alternative: method + error without the patchright prefix.
# Some synthetic fixtures or route-wrapped errors may not have the prefix.
_METHOD_ERROR_LINE_RE = re.compile(
    r"((?:Page|Locator|Frame|ElementHandle)\.\w+):\s+"
    r"((?:Timeout \d+ms exceeded|net::ERR_\w+|.*?Error:.*?|.*?violation:.*?).*)",
    re.DOTALL,
)

# Extract Python exception type from the last line of a traceback.
# Matches "SyntaxError: ...", "AttributeError: ...", "KeyError: ..."
_PYTHON_EXCEPTION_RE = re.compile(
    r"^(\w+(?:\.\w+)*Error|KeyError|IndexError|StopIteration|"
    r"AssertionError|EOFError|RecursionError|OverflowError|"
    r"ZeroDivisionError|NameError|UnboundLocalError):\s*(.*)",
    re.MULTILINE,
)

# Broader Python exception catch — matches any "ExceptionName: message" at line start.
_PYTHON_EXCEPTION_BROAD_RE = re.compile(
    r"^([A-Z]\w*(?:Error|Exception|Warning|Interrupt)):\s*(.*)",
    re.MULTILINE,
)

# Extract net::ERR_* code.
_NET_ERR_RE = re.compile(r"(net::ERR_\w+)")

# Extract CSS selector from Playwright messages.
# Matches patterns like:
#   - locator(".product-card")
#   - locator("#target")
#   - waiting for locator(".item")
#   - query_selector(".nonexistent")
_SELECTOR_LOCATOR_RE = re.compile(
    r'locator\(["\']([^"\']+)["\']\)',
)

# Extract selector from wait_for_selector calls in code.
_SELECTOR_WAIT_RE = re.compile(
    r"wait_for_selector\(\s*['\"]([^'\"]+)['\"]",
)

# Extract selector from click/fill/etc calls in code.
_SELECTOR_ACTION_RE = re.compile(
    r"(?:click|fill|hover|dblclick|tap|check|uncheck|press|type|focus|blur|"
    r"select_option|set_input_files|dispatch_event|select_text|"
    r"scroll_into_view_if_needed|text_content|inner_text|inner_html|"
    r"get_attribute|input_value|query_selector|query_selector_all)"
    r"\(\s*['\"]([^'\"]+)['\"]",
)

# Extract attribute/key from Python errors.
# "AttributeError: 'NoneType' object has no attribute 'text_content'"
_ATTR_ERROR_RE = re.compile(
    r"has no attribute ['\"](\w+)['\"]",
)

# "KeyError: 'nonexistent_key'"
_KEY_ERROR_RE = re.compile(
    r"KeyError:\s*['\"]([^'\"]+)['\"]",
)

# Extract the timeout method from "Method: Timeout Nms exceeded"
# Uses \b[A-Z]\w+ to anchor the match and avoid catastrophic backtracking.
_TIMEOUT_METHOD_RE = re.compile(
    r"(\b[A-Z]\w+\.\w+):\s+Timeout\s+\d+ms\s+exceeded",
)

# Extract JS error type from evaluate errors.
# "Page.evaluate: TypeError: Cannot read properties of null ..."
# "Page.evaluate: ReferenceError: undefinedVariable is not defined"
_JS_ERROR_RE = re.compile(
    r"(?:TypeError|ReferenceError|SyntaxError|RangeError|URIError):\s*(.*)",
)

# Strict mode violation — extract element count.
_STRICT_MODE_RE = re.compile(
    r"strict mode violation:\s*\w*\s*resolved to (\d+) elements",
)

# Pointer intercept — extract intercepting element tag.
# Format: "<div></div> intercepts pointer events" or
#          "<div class="overlay"></div> intercepts pointer events"
_POINTER_INTERCEPT_RE = re.compile(
    r"<(\w+)[^>]*>(?:</\w+>)?\s*intercepts pointer events",
)

# Schema validation error target extraction.
# "Expected at least 5 items, got 0" -> "item_count < 5"
_SCHEMA_MIN_RE = re.compile(r"[Ee]xpected at least (\d+) items?, got (\d+)")
_SCHEMA_FIELD_RE = re.compile(r"[Mm]issing (?:required )?field:?\s*['\"]?(\w+)['\"]?")
_SCHEMA_TYPE_RE = re.compile(r"[Ee]xpected (\w+) for field ['\"]?(\w+)['\"]?, got (\w+)")

# F2 timeout value extraction.
_F2_TIMEOUT_RE = re.compile(r"timed out after (\d+) seconds")

# Navigation target (URL) from net::ERR_ errors.
_NAV_URL_RE = re.compile(r"net::ERR_\w+ at (\S+)")


# ── Public API ────────────────────────────────────────────────


def extract_fingerprint(
    stderr: str,
    category: ErrorCategory,
    schema_error: str | None = None,
) -> Fingerprint:
    """Extract a structured fingerprint from raw error output.

    The fingerprint captures the error type, method, and target for
    cross-attempt comparison. Extraction is best-effort — fields that
    cannot be parsed are set to None.

    Args:
        stderr: Raw stderr output from the subprocess.
        category: The error category (from ``classify_error()``).
        schema_error: Schema validation error message (for Category G).

    Returns:
        A Fingerprint with as many fields populated as possible.
    """
    if category == ErrorCategory.G:
        return _fingerprint_schema(schema_error or "")

    if category == ErrorCategory.A:
        return _fingerprint_parse_error(stderr, category)

    if category in {ErrorCategory.F1, ErrorCategory.F2, ErrorCategory.F3}:
        return _fingerprint_process_death(stderr, category)

    if category == ErrorCategory.C:
        return _fingerprint_network(stderr, category)

    if category == ErrorCategory.E:
        return _fingerprint_page_state(stderr, category)

    if category == ErrorCategory.D:
        return _fingerprint_timeout(stderr, category)

    # Category B (catch-all)
    return _fingerprint_runtime(stderr, category)


def compare_fingerprints(a: Fingerprint, b: Fingerprint) -> ComparisonLevel:
    """Compare two fingerprints at three levels of specificity (spec S5).

    Returns the highest matching level:
    - EXACT: Same category + error_type + method + target
    - SAME_KIND: Same category + error_type + method, different target
    - SAME_CATEGORY: Same category, different error_type or method
    - NONE: Different categories

    Args:
        a: First fingerprint.
        b: Second fingerprint.

    Returns:
        The comparison level between the two fingerprints.
    """
    if a.category != b.category:
        return ComparisonLevel.NONE

    # Same category — check deeper fields
    if a.error_type == b.error_type and a.method == b.method:
        # Both error_type and method match
        if a.target is not None and b.target is not None:
            if a.target == b.target:
                return ComparisonLevel.EXACT
            return ComparisonLevel.SAME_KIND
        # One or both targets are None — can't distinguish EXACT from SAME_KIND.
        # If both are None, it's effectively EXACT (no target to differ on).
        if a.target is None and b.target is None:
            return ComparisonLevel.EXACT
        # One has target, other doesn't — treat as SAME_KIND (degraded).
        return ComparisonLevel.SAME_KIND

    return ComparisonLevel.SAME_CATEGORY


# ── Internal extraction helpers ───────────────────────────────


def _fingerprint_parse_error(
    stderr: str, category: ErrorCategory,
) -> Fingerprint:
    """Extract fingerprint for Category A (parse errors)."""
    error_type = None
    target = None
    message = stderr

    # Find the exception line
    match = _PYTHON_EXCEPTION_RE.search(stderr)
    if not match:
        match = _PYTHON_EXCEPTION_BROAD_RE.search(stderr)

    if match:
        error_type = match.group(1)
        detail = match.group(2).strip()
        # For import errors, the target is the module name.
        mod_match = re.search(r"No module named ['\"]([^'\"]+)['\"]", detail)
        if mod_match:
            target = mod_match.group(1)

    return Fingerprint(
        category=category,
        error_type=error_type,
        method=None,  # No browser method for parse errors
        target=target,
        message=message,
    )


def _fingerprint_runtime(
    stderr: str, category: ErrorCategory,
) -> Fingerprint:
    """Extract fingerprint for Category B (runtime crashes)."""
    error_type = None
    method = None
    target = None

    # Try Patchright error line first (JS errors via evaluate, API misuse).
    pw_match = _PW_ERROR_LINE_RE.search(stderr)
    if pw_match:
        method = pw_match.group(1)
        error_detail = pw_match.group(2).strip()
        # Extract JS error type
        js_match = _JS_ERROR_RE.search(error_detail)
        if js_match:
            # e.g., "TypeError: Cannot read properties of null"
            for js_type in ("TypeError", "ReferenceError", "SyntaxError",
                            "RangeError", "URIError"):
                if js_type in error_detail:
                    error_type = f"JS.{js_type}"
                    break
        if not error_type:
            # API misuse — use the error message as error_type
            error_type = _extract_short_error_type(error_detail)
    else:
        # Python exception
        match = _PYTHON_EXCEPTION_RE.search(stderr)
        if not match:
            match = _PYTHON_EXCEPTION_BROAD_RE.search(stderr)
        if match:
            error_type = match.group(1)
            detail = match.group(2).strip()

            # Extract target based on error type
            if error_type == "AttributeError":
                attr_match = _ATTR_ERROR_RE.search(detail)
                if attr_match:
                    target = attr_match.group(1)
            elif error_type == "KeyError":
                key_match = _KEY_ERROR_RE.search(stderr)
                if key_match:
                    target = key_match.group(1)

    # Try to extract selector from the scrape function code in the traceback.
    if target is None:
        target = _extract_selector(stderr)

    return Fingerprint(
        category=category,
        error_type=error_type,
        method=method,
        target=target,
        message=stderr,
    )


def _fingerprint_network(
    stderr: str, category: ErrorCategory,
) -> Fingerprint:
    """Extract fingerprint for Category C (network/server failure)."""
    error_type = None
    method = None
    target = None

    # Extract net::ERR_* code
    net_match = _NET_ERR_RE.search(stderr)
    if net_match:
        error_type = net_match.group(1)
        # Extract URL target
        url_match = _NAV_URL_RE.search(stderr)
        if url_match:
            target = url_match.group(1)
    else:
        # Navigation timeout (Page.goto: Timeout ...)
        timeout_match = _TIMEOUT_METHOD_RE.search(stderr)
        if timeout_match:
            error_type = "TimeoutError"
            method = timeout_match.group(1)

    # Extract method if not yet found
    if method is None:
        pw_match = _PW_ERROR_LINE_RE.search(stderr)
        if pw_match:
            method = pw_match.group(1)

    return Fingerprint(
        category=category,
        error_type=error_type,
        method=method,
        target=target,
        message=stderr,
    )


def _fingerprint_timeout(
    stderr: str, category: ErrorCategory,
) -> Fingerprint:
    """Extract fingerprint for Category D (post-navigation timeouts)."""
    error_type = "TimeoutError"
    method = None
    target = None

    # Extract method from "Method: Timeout Nms exceeded"
    timeout_match = _TIMEOUT_METHOD_RE.search(stderr)
    if timeout_match:
        method = timeout_match.group(1)

    # Extract selector target from call log or code
    target = _extract_selector(stderr)

    return Fingerprint(
        category=category,
        error_type=error_type,
        method=method,
        target=target,
        message=stderr,
    )


def _fingerprint_page_state(
    stderr: str, category: ErrorCategory,
) -> Fingerprint:
    """Extract fingerprint for Category E (page state errors)."""
    error_type = None
    method = None
    target = None

    # Determine the specific E sub-type
    if re.search(r"intercepts pointer events", stderr):
        error_type = "pointer_intercept"
        # Extract the intercepting element
        intercept_match = _POINTER_INTERCEPT_RE.search(stderr)
        if intercept_match:
            target = f"<{intercept_match.group(1)}>"
    elif re.search(r"element is not visible", stderr):
        error_type = "not_visible"
    elif re.search(r"element is not enabled", stderr):
        error_type = "not_enabled"
    elif re.search(r"element is not stable", stderr):
        error_type = "not_stable"
    elif re.search(r"Element is outside of the viewport", stderr):
        error_type = "outside_viewport"
    elif re.search(r"Element is not attached|Node is detached", stderr):
        error_type = "detached"
    elif re.search(r"strict mode violation:", stderr):
        error_type = "strict_mode"
        sm_match = _STRICT_MODE_RE.search(stderr)
        if sm_match:
            target = f"resolved_to_{sm_match.group(1)}"
    elif re.search(r"Execution context was destroyed", stderr):
        error_type = "context_destroyed"
    elif re.search(r"Frame was detached|Navigating frame was detached", stderr):
        error_type = "frame_detached"
    elif re.search(r"Navigation interrupted", stderr):
        error_type = "nav_interrupted"
    elif re.search(r"Frame is currently attempting a navigation", stderr):
        error_type = "frame_navigating"
    elif re.search(r"Unable to retrieve content.*navigating", stderr):
        error_type = "content_during_nav"
    elif re.search(r"Frame for this navigation request", stderr):
        error_type = "frame_not_available"
    elif re.search(r"[Cc]annot evaluate.*JavaScript dialog|"
                    r"JavaScript dialog prevents evaluation", stderr):
        error_type = "dialog_blocking"
    elif re.search(r"Clicking the checkbox did not change", stderr):
        error_type = "checkbox_unchanged"
    elif re.search(r"Cannot set input files to detached", stderr):
        error_type = "input_files_detached"
    elif re.search(r"Element is not an <", stderr):
        error_type = "wrong_element_type"

    # Extract method
    pw_match = _PW_ERROR_LINE_RE.search(stderr)
    if pw_match:
        method = pw_match.group(1)
    elif _TIMEOUT_METHOD_RE.search(stderr):
        method = _TIMEOUT_METHOD_RE.search(stderr).group(1)

    # Extract selector if not already set
    if target is None:
        target = _extract_selector(stderr)

    return Fingerprint(
        category=category,
        error_type=error_type,
        method=method,
        target=target,
        message=stderr,
    )


def _fingerprint_process_death(
    stderr: str, category: ErrorCategory,
) -> Fingerprint:
    """Extract fingerprint for Category F (process/browser death)."""
    error_type = None
    method = None
    target = None

    if category == ErrorCategory.F2:
        error_type = "SubprocessTimeout"
        timeout_match = _F2_TIMEOUT_RE.search(stderr)
        if timeout_match:
            target = f"{timeout_match.group(1)}s"
    elif category == ErrorCategory.F3:
        # Extract the specific infrastructure error
        if re.search(r"[Ee]xecutable doesn'?t exist", stderr):
            error_type = "BrowserNotFound"
        elif re.search(r"No space left on device", stderr):
            error_type = "DiskFull"
        elif re.search(r"Too many open files|EMFILE", stderr):
            error_type = "TooManyFiles"
        elif re.search(r"No usable sandbox", stderr):
            error_type = "SandboxError"
        elif re.search(r"Cannot open display", stderr):
            error_type = "NoDisplay"
        else:
            error_type = "InfrastructureError"
    else:
        # F1: Browser/page crash
        if re.search(r"TargetClosedError", stderr):
            error_type = "TargetClosedError"
        elif re.search(r"Page crashed|Target crashed", stderr):
            error_type = "PageCrash"
        elif re.search(r"Browser.*closed", stderr):
            error_type = "BrowserClosed"
        elif re.search(r"Connection closed|Playwright connection", stderr):
            error_type = "ConnectionLost"
        elif re.search(r"Socket closed|Socket error", stderr):
            error_type = "SocketError"
        else:
            error_type = "ProcessDeath"

        # Extract method if present
        pw_match = _PW_ERROR_LINE_RE.search(stderr)
        if pw_match:
            method = pw_match.group(1)

    return Fingerprint(
        category=category,
        error_type=error_type,
        method=method,
        target=target,
        message=stderr,
    )


def _fingerprint_schema(schema_error: str) -> Fingerprint:
    """Extract fingerprint for Category G (schema validation failure)."""
    error_type = "SchemaValidationError"
    target = None

    # Try specific schema error patterns
    min_match = _SCHEMA_MIN_RE.search(schema_error)
    if min_match:
        target = f"item_count < {min_match.group(1)}"
    else:
        field_match = _SCHEMA_FIELD_RE.search(schema_error)
        if field_match:
            target = f"missing field: {field_match.group(1)}"
        else:
            type_match = _SCHEMA_TYPE_RE.search(schema_error)
            if type_match:
                target = f"type mismatch: {type_match.group(2)}"

    return Fingerprint(
        category=ErrorCategory.G,
        error_type=error_type,
        method=None,
        target=target,
        message=schema_error,
    )


# ── Shared extraction utilities ───────────────────────────────


def _extract_selector(stderr: str) -> str | None:
    """Extract a CSS/XPath selector from error output.

    Searches in order:
    1. Playwright locator() calls in call log: locator(".product-card")
    2. wait_for_selector() calls in script code
    3. click/fill/etc calls in script code
    """
    # 1. From call log locator() references
    match = _SELECTOR_LOCATOR_RE.search(stderr)
    if match:
        return match.group(1)

    # 2. From wait_for_selector() in script code
    match = _SELECTOR_WAIT_RE.search(stderr)
    if match:
        return match.group(1)

    # 3. From action methods in script code
    match = _SELECTOR_ACTION_RE.search(stderr)
    if match:
        return match.group(1)

    return None


def _extract_short_error_type(error_detail: str) -> str:
    """Extract a short error type label from a Patchright error detail.

    For API misuse errors like "strict mode violation: ...", returns
    the first few words as the error type.
    """
    # Take first line, truncate to 60 chars
    first_line = error_detail.split("\n", 1)[0].strip()
    if len(first_line) > 60:
        first_line = first_line[:57] + "..."
    return first_line
