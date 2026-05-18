"""Error classifier — maps raw error output to an ErrorCategory.

Classifies subprocess stderr, exit codes, and schema errors into one of
nine categories (A, B, C, D, E, F1, F2, F3, G) using priority-ordered
pattern matching.

Priority order (spec S4):
  1. Category A  — parse errors (SyntaxError, IndentationError, ImportError)
  2. Category F  — process/browser death (F1, F2, F3)
  3. Category C  — network/server failure (net::ERR_*, navigation timeouts)
  4. Category E  — page state prevented interaction (call log patterns, then non-timeout E)
  5. Category D  — post-navigation timeouts (general Timeout pattern)
  6. Category B  — output-stage errors, then catch-all

Category G is detected externally by the caller (schema validation) and
passed in via the ``schema_error`` parameter.

Spec reference: docs/specific/AUTO_FIX_ALGORITHM.md S4, S13
"""

from __future__ import annotations

import re

from scout.autofix.types import ErrorCategory

# ── Compiled regex patterns ───────────────────────────────────
#
# All patterns are compiled at module level to avoid re-compilation
# in the hot path (up to 3 calls per run() invocation).

# -- Route handler wrapper (strip before classification) --
_ROUTE_HANDLER_WRAPPER_RE = re.compile(
    r'"([^"]*?)"\s+while running route callback\.\s*'
    r"Consider awaiting .?page\.unroute_all\(behavior=.?ignoreErrors.?\).?\s*"
    r"before the end of the test to ignore remaining routes in flight\.",
    re.DOTALL,
)

# -- Category A: Parse errors --
# These appear as the LAST line of stderr (the exception line in a traceback).
# Match the exception class name at the start of a line.
_CAT_A_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^SyntaxError:", re.MULTILINE),
    re.compile(r"^IndentationError:", re.MULTILINE),
    re.compile(r"^TabError:", re.MULTILINE),
    re.compile(r"^ModuleNotFoundError:", re.MULTILINE),
    re.compile(r"^ImportError:", re.MULTILINE),
]

# -- Category F1: Browser/page crash --
_F1_MESSAGE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"Page crashed"),
    re.compile(r"Target crashed"),
    re.compile(r"Target page, context or browser has been closed"),
    re.compile(r"Browser has been closed"),
    re.compile(r"Browser closed"),
    re.compile(r"Page closed"),
    re.compile(r"Context closed"),
    re.compile(r"Navigation failed because page crashed"),
    re.compile(r"Connection closed while reading from the driver"),
    re.compile(r"Playwright connection closed"),
    re.compile(r"Socket closed"),
    re.compile(r"Socket error"),
    re.compile(r"The object has been collected to prevent unbounded heap growth"),
    # TargetClosedError exception class (patchright or playwright prefix)
    re.compile(r"(?:patchright|playwright)\._impl\._errors\.TargetClosedError"),
]

# Signal-based process death exit codes (spec S4, Category F1).
# 128+signal: SIGSEGV=139, SIGABRT=134, SIGBUS=135
# Negative return codes from asyncio subprocess: -11, -6, -7
_F1_EXIT_CODES: frozenset[int] = frozenset({139, 134, 135, -11, -6, -7})

# -- Category F2: Subprocess/execution timeout --
_F2_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"Script execution timed out after \d+ seconds"),
    re.compile(r"Function timed out after \d+ seconds"),
]

# -- Category F3: Infrastructure failure --
_F3_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"[Ee]xecutable doesn'?t exist"),
    re.compile(r"Host system is missing dependencies"),
    re.compile(r"Cannot open display"),
    re.compile(r"Failed to connect to the bus"),
    re.compile(r"No space left on device"),
    re.compile(r"Too many open files"),
    re.compile(r"\bEMFILE\b"),
    re.compile(r"No usable sandbox"),
    re.compile(r"Event loop is closed"),
    # §4 F3: Browser launch connection timed out. Scoped to BrowserType
    # context to avoid false positives on "Connection timed out" appearing
    # in call logs or other error contexts.
    re.compile(r"BrowserType\.\w+:.*Connection timed out"),
    re.compile(r"Maximum argument depth exceeded"),
    re.compile(r"/dev/shm"),
    re.compile(r"libnss3\.so|libatk-1\.0\.so|libxss\.so"),
    re.compile(r"Cannot create directory"),
    re.compile(r"mkdtemp"),
    re.compile(r"BrowserType\.launch|BrowserType\.launch_persistent_context"),
]

# Exit code 137 (SIGKILL/OOM) is F3, not F1.
_F3_EXIT_CODES: frozenset[int] = frozenset({137, -9})

# -- Category C: Network/server failure --
# Any net::ERR_* is Category C regardless of which method produced it.
_NET_ERR_RE = re.compile(r"net::ERR_\w+")

# Navigation methods whose timeouts are Category C (closed set).
# Patchright always uses PascalCase in error messages (Page.goto, not page.goto).
_NAVIGATION_METHODS: frozenset[str] = frozenset(
    {
        "Page.goto",
        "Page.reload",
        "Page.go_back",
        "Page.go_forward",
        "Frame.goto",
    }
)

# General timeout pattern: "{Method}: Timeout {N}ms exceeded"
# Captures the method prefix (e.g., "Page.click", "Locator.wait_for").
# Uses atomic-style grouping via possessive quantifier workaround to prevent
# catastrophic backtracking on large inputs (100KB+).
_TIMEOUT_RE = re.compile(
    r"(\b[A-Z]\w+\.\w+):\s+Timeout\s+\d+ms\s+exceeded",
)

# -- Category E: Page state errors (in call log) --
# These patterns appear in the "Call log:" section of timeout errors.
# They must be checked BEFORE Category D (spec S13, consequence #1).
_E_CALL_LOG_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"intercepts pointer events"),
    re.compile(r"element is not visible"),
    re.compile(r"element is not enabled"),
    re.compile(r"element is not stable"),
    re.compile(r"Element is outside of the viewport"),
    re.compile(r"Element is not attached to the DOM"),
    re.compile(r"Node is detached"),
    re.compile(r"Clicking the checkbox did not change its state"),
    re.compile(r"Cannot set input files to detached element"),
    re.compile(r"Element is not an <input>"),
    re.compile(r"Element is not a <select>"),
]

# Category E: Non-timeout errors (these are Error, not TimeoutError).
_E_NONTIMEOUT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"Execution context was destroyed"),
    re.compile(r"Frame was detached"),
    re.compile(r"Navigating frame was detached"),
    re.compile(r"Navigation interrupted by another navigation"),
    re.compile(r"Frame is currently attempting a navigation"),
    re.compile(r"Unable to retrieve content because the page is navigating"),
    re.compile(r"Frame for this navigation request is not available"),
    re.compile(r"strict mode violation:"),
    # Dialog blocking — both old and new formats.
    re.compile(r"Cannot evaluate.*page has an open JavaScript dialog"),
    re.compile(r"Open JavaScript dialog prevents evaluation"),
]

# -- Category B: Output-stage errors --
_B_OUTPUT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"Script did not produce output file"),
    re.compile(r"Script produced empty output file"),
    re.compile(r"Script output is not valid JSON"),
]


# ── Public API ────────────────────────────────────────────────


def classify_error(
    stderr: str,
    exit_code: int | None = None,
    schema_error: str | None = None,
    stdout: str = "",
) -> ErrorCategory:
    """Classify a script execution error into an ErrorCategory.

    Uses priority-ordered pattern matching on stderr, exit code, and
    schema validation results. Earlier categories take precedence.

    Args:
        stderr: Raw stderr output from the subprocess (decoded).
        exit_code: Process exit code (None for in-process execution).
            Signal-based codes (139, 134, 135) indicate F1 process death.
            Exit code 137 indicates F3 (OOM/SIGKILL).
        schema_error: Schema validation error message, if the script
            returned data that failed schema validation (Category G).
            This is checked LAST — schema errors imply the script ran
            successfully but produced wrong output.
        stdout: Raw stdout output (used for B_empty_output detection
            when stderr is empty and exit_code is 0).

    Returns:
        The most specific ErrorCategory matching the error.
        Falls through to Category B as catch-all for unrecognized errors.
    """
    # §4: Category G is detected externally. If present, it means the script
    # ran to completion and returned data. But we still check for other
    # categories first — a script that crashes AND has schema_error should
    # be classified by the crash, not the schema.
    # However, if there's no real stderr and exit_code is 0, schema_error wins.
    has_real_stderr = bool(stderr and stderr.strip())
    if schema_error and not has_real_stderr and (exit_code is None or exit_code == 0):
        return ErrorCategory.G

    # Strip route handler wrapper before classification (spec S4, S13 #7).
    cleaned = _strip_route_wrapper(stderr)

    # Priority 1: Category A — parse errors
    if _check_category_a(cleaned):
        return ErrorCategory.A

    # Priority 2: Category F — process/browser death
    f_result = _check_category_f(cleaned, exit_code)
    if f_result is not None:
        return f_result

    # Priority 3: Category C — network/server failure
    if _check_category_c(cleaned):
        return ErrorCategory.C

    # Priority 4: Category E — page state (check BEFORE D, spec S13 #1-2)
    if _check_category_e(cleaned):
        return ErrorCategory.E

    # Priority 5: Category D — post-navigation timeouts
    if _check_category_d(cleaned):
        return ErrorCategory.D

    # Priority 6: Category B output-stage errors
    if _check_category_b_output(cleaned, stdout, exit_code):
        return ErrorCategory.B

    # §4: Schema error with real stderr is unusual — the crash categories
    # above should have caught it. But if nothing matched AND schema_error
    # is set, the script may have produced partial output before crashing.
    # In this case, the crash (catch-all B) takes priority over schema.
    # Only return G if there's genuinely no other error signal.
    if schema_error and not has_real_stderr:
        return ErrorCategory.G

    # Priority 7: Category B catch-all
    # Any error that doesn't match above categories.
    return ErrorCategory.B


# ── Internal classification helpers ───────────────────────────


def _strip_route_wrapper(stderr: str) -> str:
    """Strip route handler error wrapper if present (spec S4, S13 #7).

    Route callbacks wrap the original error with:
      "{original_error}" while running route callback.
      Consider awaiting `page.unroute_all(behavior='ignoreErrors')`...

    The classifier should classify based on the original error.
    """
    return _ROUTE_HANDLER_WRAPPER_RE.sub("", stderr)


def _check_category_a(stderr: str) -> bool:
    """Check for Category A parse errors.

    Matches SyntaxError, IndentationError, TabError, ImportError,
    ModuleNotFoundError at the start of a line (the exception line
    in a Python traceback).
    """
    return any(p.search(stderr) for p in _CAT_A_PATTERNS)


def _check_category_f(
    stderr: str,
    exit_code: int | None,
) -> ErrorCategory | None:
    """Check for Category F process/browser death.

    Returns F1, F2, or F3 if matched, None otherwise.
    Checks in order: F2, F3, F1 (F2 and F3 are more specific).
    """
    # F2: Subprocess timeout (check first — timeout messages are specific)
    if any(p.search(stderr) for p in _F2_PATTERNS):
        return ErrorCategory.F2

    # F3: Infrastructure failure (exit code or message patterns)
    if exit_code is not None and exit_code in _F3_EXIT_CODES:
        return ErrorCategory.F3
    if any(p.search(stderr) for p in _F3_PATTERNS):
        return ErrorCategory.F3

    # F1: Browser/page crash (exit code or message patterns)
    if exit_code is not None and exit_code in _F1_EXIT_CODES:
        return ErrorCategory.F1
    if any(p.search(stderr) for p in _F1_MESSAGE_PATTERNS):
        return ErrorCategory.F1

    return None


def _check_category_c(stderr: str) -> bool:
    """Check for Category C network/server failure.

    Two sub-checks:
    1. Any net::ERR_* pattern (regardless of method).
    2. Navigation method timeouts (Page.goto, Page.reload, etc.).
    """
    # net::ERR_* always Category C
    if _NET_ERR_RE.search(stderr):
        return True

    # Navigation timeout: "{NavigationMethod}: Timeout Nms exceeded"
    match = _TIMEOUT_RE.search(stderr)
    if match:
        method = match.group(1)
        if method in _NAVIGATION_METHODS:
            return True

    return False


def _check_category_e(stderr: str) -> bool:
    """Check for Category E page state errors.

    Two sub-checks:
    1. Timeout errors with E-specific patterns in the call log section.
       These surface as TimeoutError but the call log reveals the real cause.
    2. Non-timeout E errors (context destroyed, frame detached, dialog, strict mode).

    Must be checked BEFORE Category D (spec S13 #1-2).
    """
    # Check non-timeout E patterns first (these are Error, not TimeoutError)
    if any(p.search(stderr) for p in _E_NONTIMEOUT_PATTERNS):
        return True

    # Check for timeout errors with E-specific call log patterns.
    # Only look at the call log section if there's a timeout pattern.
    if _TIMEOUT_RE.search(stderr):
        # Extract everything after "Call log:" (the call log section)
        call_log = _extract_call_log(stderr)
        if call_log and any(p.search(call_log) for p in _E_CALL_LOG_PATTERNS):
            return True

    return False


def _check_category_d(stderr: str) -> bool:
    """Check for Category D post-navigation timeouts.

    Matches any "{Method}: Timeout {N}ms exceeded" where the method
    is NOT a navigation method (those are Category C) and the call log
    does NOT contain Category E patterns (already checked).

    Uses a general pattern rather than enumerating methods (spec S13 #5).
    """
    match = _TIMEOUT_RE.search(stderr)
    if match:
        method = match.group(1)
        # Navigation methods are Category C (already checked), but
        # guard against re-ordering.
        if method not in _NAVIGATION_METHODS:
            return True
    return False


def _check_category_b_output(
    stderr: str,
    stdout: str,
    exit_code: int | None,
) -> bool:
    """Check for Category B output-stage errors.

    Catches:
    - "Script did not produce output file"
    - "Script produced empty output file"
    - "Script output is not valid JSON"
    - Script returned None/null (exit code 0, no stderr, no output markers)
    """
    if any(p.search(stderr) for p in _B_OUTPUT_PATTERNS):
        return True

    # None return: exit code 0, no stderr, but no valid output in stdout.
    # The subprocess wrapper writes output between markers. If the markers
    # are absent or contain "null", the script returned None.
    if exit_code is not None and exit_code == 0 and not stderr.strip():
        # Check if stdout has the return value markers with null/empty
        if not stdout.strip():
            return True
        if "null" in stdout and "__SCOUT_RETURN_VALUE_" in stdout:
            return True

    return False


def _extract_call_log(stderr: str) -> str | None:
    """Extract the 'Call log:' section from stderr.

    Playwright/Patchright appends a call log after error messages:
        patchright._impl._errors.TimeoutError: Page.click: Timeout 5000ms exceeded.
        Call log:
          - waiting for locator("#target")
          ...

    Returns the text after "Call log:" or None if not found.
    """
    # Find "Call log:" (case-sensitive, as Playwright uses this exact casing)
    idx = stderr.find("Call log:")
    if idx == -1:
        return None
    return stderr[idx:]
