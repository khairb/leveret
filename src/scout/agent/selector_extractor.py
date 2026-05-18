"""Selector Extractor — parses AI-written code to extract DOM selectors.

Three-stage pure-function pipeline:

  Stage 1: Python-level extraction (direct Playwright API calls)
  Stage 2: Evaluate JS extraction (selectors inside page.evaluate())
  Stage 3: Context analysis (loop detection, navigation boundaries)

Usage::

    from scout.agent.selector_extractor import extract_selectors

    results = extract_selectors(code_string)
    for r in results:
        print(r.selector, r.action_category, r.in_loop)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Literal

# ═══════════════════════════════════════════════════════════════════════
#  Output model
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ExtractionResult:
    """A single extracted DOM selector with metadata."""

    selector: str
    selector_type: Literal["css", "playwright"]
    action_category: Literal["navigating", "mutating", "passive"]
    action: str
    line: int
    in_loop: bool = False
    after_navigation: bool = False
    source: Literal["python", "evaluate_js"] = "python"


# ═══════════════════════════════════════════════════════════════════════
#  Method classification constants
# ═══════════════════════════════════════════════════════════════════════

# page.METHOD("selector") — navigating (could trigger page change)
_PAGE_NAVIGATING: set[str] = {"click", "dblclick", "tap"}

# page.METHOD("selector") — mutating (changes element state, page stays)
_PAGE_MUTATING: set[str] = {
    "fill",
    "type",
    "press",
    "check",
    "uncheck",
    "set_checked",
    "select_option",
    "focus",
    "set_input_files",
    "dispatch_event",
}

# page.METHOD("selector") — passive (read-only)
_PAGE_PASSIVE: set[str] = {
    "query_selector",
    "query_selector_all",
    "wait_for_selector",
    "inner_text",
    "inner_html",
    "text_content",
    "get_attribute",
    "input_value",
    "is_visible",
    "is_hidden",
    "is_enabled",
    "is_disabled",
    "is_checked",
    "is_editable",
    "eval_on_selector",
    "eval_on_selector_all",
}

# page.METHOD("selector") — deferred (creates locator, action from chain)
_PAGE_DEFERRED: set[str] = {"locator", "frame_locator"}

# All page methods that take a selector as first arg
_PAGE_SELECTOR_METHODS = _PAGE_NAVIGATING | _PAGE_MUTATING | _PAGE_PASSIVE | _PAGE_DEFERRED

# Locator/element terminal actions — navigating
_CHAIN_NAVIGATING: set[str] = {"click", "dblclick", "tap"}

# Locator/element terminal actions — mutating
_CHAIN_MUTATING: set[str] = {
    "fill",
    "type",
    "press",
    "press_sequentially",
    "check",
    "uncheck",
    "set_checked",
    "select_option",
    "set_input_files",
    "clear",
    "focus",
    "blur",
    "dispatch_event",
}

# Locator/element terminal actions — passive
_CHAIN_PASSIVE: set[str] = {
    "inner_text",
    "inner_html",
    "text_content",
    "get_attribute",
    "input_value",
    "is_visible",
    "is_hidden",
    "is_enabled",
    "is_disabled",
    "is_checked",
    "is_editable",
    "count",
    "all",
    "bounding_box",
    "wait_for",
    "evaluate",
    "evaluate_all",
    "evaluate_handle",
    "all_inner_texts",
    "all_text_contents",
    "element_handle",
    "element_handles",
    "scroll_into_view_if_needed",
    "highlight",
}

_ALL_CHAIN_TERMINALS = _CHAIN_NAVIGATING | _CHAIN_MUTATING | _CHAIN_PASSIVE

# Hard navigation markers (no DOM selector, boundary only)
_HARD_NAV: set[str] = {"goto", "go_back", "go_forward", "reload"}

# get_by_* helpers
_GET_BY_METHODS: set[str] = {
    "get_by_text",
    "get_by_role",
    "get_by_label",
    "get_by_placeholder",
    "get_by_alt_text",
    "get_by_title",
    "get_by_test_id",
}


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════


def _pos_to_line(code: str, pos: int) -> int:
    """Convert a character position to a 1-based line number."""
    return code[:pos].count("\n") + 1


def _classify_method(method: str) -> str:
    """Return action category for a page-level method."""
    if method in _PAGE_NAVIGATING:
        return "navigating"
    if method in _PAGE_MUTATING:
        return "mutating"
    return "passive"


def _classify_chain_terminal(method: str) -> str:
    """Return action category for a locator/element terminal method."""
    if method in _CHAIN_NAVIGATING:
        return "navigating"
    if method in _CHAIN_MUTATING:
        return "mutating"
    return "passive"


def _is_comment_line(code: str, pos: int) -> bool:
    """Check if the match at *pos* is on a comment line."""
    line_start = code.rfind("\n", 0, pos) + 1
    prefix = code[line_start:pos].lstrip()
    return prefix.startswith("#")


# ═══════════════════════════════════════════════════════════════════════
#  Stage 1: Python-level extraction
# ═══════════════════════════════════════════════════════════════════════

# Regex: page.METHOD('selector') or page.METHOD("selector")
_METHODS_ALT = "|".join(sorted(_PAGE_SELECTOR_METHODS, key=len, reverse=True))
_RE_PAGE_CALL = re.compile(
    rf"page\.({_METHODS_ALT})\(\s*(?:'([^']*)'|\"([^\"]*)\")",
)

# Regex: page.get_by_TEXT("arg") or page.get_by_ROLE("arg")
_GET_BY_ALT = "|".join(sorted(_GET_BY_METHODS, key=len, reverse=True))
_RE_GET_BY = re.compile(
    rf"(?:page|\w+)\.({_GET_BY_ALT})\(\s*(?:'([^']*)'|\"([^\"]*)\")",
)

# Regex: VAR = [await] page.(locator|query_selector|...)("sel")
_ASSIGN_METHODS = "|".join(
    sorted(
        _PAGE_DEFERRED | {"query_selector", "query_selector_all"} | _GET_BY_METHODS,
        key=len,
        reverse=True,
    )
)
_RE_ASSIGN = re.compile(
    rf"(\w+)\s*=\s*(?:await\s+)?(?:page|\w+)"
    rf"\.({_ASSIGN_METHODS})\(\s*(?:'([^']*)'|\"([^\"]*)\")",
)

# Regex: await VAR.ACTION(...)
_ALL_TERMINALS_ALT = "|".join(sorted(_ALL_CHAIN_TERMINALS, key=len, reverse=True))
_RE_VAR_ACTION = re.compile(
    rf"(?:await\s+)?(\w+)\.({_ALL_TERMINALS_ALT})\(",
)

# Regex: page.locator("sel")...TERMINAL() on same expression
_RE_LOCATOR_CHAIN = re.compile(
    r"page\.locator\(\s*(?:'([^']*)'|\"([^\"]*)\")"
    r"[^)\n]*\)"  # rest of locator() args
    r"((?:\.\w+(?:\([^)]*\))?)*)"  # chain of .method() or .property
)

# Regex: page.get_by_*("arg")...TERMINAL() on same expression
_RE_GETBY_CHAIN = re.compile(
    rf"(?:page|\w+)\.({_GET_BY_ALT})\(\s*(?:'([^']*)'|\"([^\"]*)\")"
    rf"[^)]*\)"
    rf"((?:\.\w+(?:\([^)]*\))?)*)"
)

# Regex: hard navigation markers
_HARD_NAV_ALT = "|".join(sorted(_HARD_NAV, key=len, reverse=True))
_RE_HARD_NAV = re.compile(
    rf"(?:await\s+)?page\.({_HARD_NAV_ALT})\(",
)

# Regex: detect terminal in a chain suffix
_RE_TERMINAL_IN_CHAIN = re.compile(rf"\.({_ALL_TERMINALS_ALT})\(")


def _extract_direct_page_calls(code: str) -> list[ExtractionResult]:
    """Extract selectors from direct page.METHOD('selector') calls."""
    results: list[ExtractionResult] = []
    for m in _RE_PAGE_CALL.finditer(code):
        if _is_comment_line(code, m.start()):
            continue
        method = m.group(1)
        selector = m.group(2) or m.group(3)
        if not selector:
            continue
        # Skip deferred (locator/frame_locator) — handled by chain/var tracking
        if method in _PAGE_DEFERRED:
            continue
        line = _pos_to_line(code, m.start())
        cat = _classify_method(method)
        sel_type: str = "playwright" if ":has-text(" in selector or ">> " in selector else "css"
        results.append(
            ExtractionResult(
                selector=selector,
                selector_type=sel_type,
                action_category=cat,
                action=method,
                line=line,
                source="python",
            )
        )
    return results


def _extract_locator_chains(code: str) -> list[ExtractionResult]:
    """Extract selectors from inline page.locator('sel')...action() chains."""
    results: list[ExtractionResult] = []

    # page.locator("sel")...terminal()
    for m in _RE_LOCATOR_CHAIN.finditer(code):
        if _is_comment_line(code, m.start()):
            continue
        selector = m.group(1) or m.group(2)
        if not selector:
            continue
        chain_suffix = m.group(3) or ""
        tm = _RE_TERMINAL_IN_CHAIN.search(chain_suffix)
        if not tm:
            continue  # deferred — handled by variable tracking
        terminal = tm.group(1)
        line = _pos_to_line(code, m.start())
        cat = _classify_chain_terminal(terminal)
        sel_type = "playwright" if ":has-text(" in selector or ">> " in selector else "css"
        results.append(
            ExtractionResult(
                selector=selector,
                selector_type=sel_type,
                action_category=cat,
                action=terminal,
                line=line,
                source="python",
            )
        )

    # page.get_by_*("arg")...terminal()
    for m in _RE_GETBY_CHAIN.finditer(code):
        if _is_comment_line(code, m.start()):
            continue
        method = m.group(1)
        selector = m.group(2) or m.group(3)
        if not selector:
            continue
        chain_suffix = m.group(4) or ""
        tm = _RE_TERMINAL_IN_CHAIN.search(chain_suffix)
        if not tm:
            continue  # deferred
        terminal = tm.group(1)
        line = _pos_to_line(code, m.start())
        cat = _classify_chain_terminal(terminal)
        results.append(
            ExtractionResult(
                selector=selector,
                selector_type="playwright",
                action_category=cat,
                action=method,
                line=line,
                source="python",
            )
        )

    return results


def _extract_variable_tracking(
    code: str,
) -> tuple[list[ExtractionResult], dict[str, tuple[str, str, str]]]:
    """Track variable assignments and resolve actions on those variables."""
    var_dict: dict[str, tuple[str, str, str]] = {}  # name -> (selector, sel_type, method)

    for m in _RE_ASSIGN.finditer(code):
        if _is_comment_line(code, m.start()):
            continue
        var_name = m.group(1)
        method = m.group(2)
        selector = m.group(3) or m.group(4)
        if not selector:
            continue
        sel_type = "playwright" if method in _GET_BY_METHODS else "css"
        if ":has-text(" in selector or ">> " in selector:
            sel_type = "playwright"
        var_dict[var_name] = (selector, sel_type, method)

    results: list[ExtractionResult] = []
    for m in _RE_VAR_ACTION.finditer(code):
        if _is_comment_line(code, m.start()):
            continue
        var_name = m.group(1)
        action = m.group(2)
        if var_name not in var_dict:
            continue
        selector, sel_type, _orig_method = var_dict[var_name]
        line = _pos_to_line(code, m.start())
        cat = _classify_chain_terminal(action)
        results.append(
            ExtractionResult(
                selector=selector,
                selector_type=sel_type,
                action_category=cat,
                action=action,
                line=line,
                source="python",
            )
        )

    return results, var_dict


def _find_navigation_markers(code: str) -> list[tuple[int, str]]:
    """Find hard navigation calls (goto, go_back, reload, etc.)."""
    markers: list[tuple[int, str]] = []
    for m in _RE_HARD_NAV.finditer(code):
        if _is_comment_line(code, m.start()):
            continue
        line = _pos_to_line(code, m.start())
        markers.append((line, m.group(1)))
    return markers


def _extract_python_level(
    code: str,
) -> tuple[list[ExtractionResult], list[tuple[int, str]], dict[str, tuple[str, str, str]]]:
    """Stage 1: Extract all Python-level selectors."""
    direct = _extract_direct_page_calls(code)
    chains = _extract_locator_chains(code)
    var_results, var_dict = _extract_variable_tracking(code)
    nav_markers = _find_navigation_markers(code)

    # Deduplicate: chain results may overlap with direct results
    # Use (selector, line) as dedup key
    seen: set[tuple[str, int]] = set()
    merged: list[ExtractionResult] = []
    for r in direct + chains + var_results:
        key = (r.selector, r.line)
        if key not in seen:
            seen.add(key)
            merged.append(r)

    return merged, nav_markers, var_dict


# ═══════════════════════════════════════════════════════════════════════
#  Stage 2: Evaluate JS extraction
# ═══════════════════════════════════════════════════════════════════════

# Regex: find .evaluate(, .evaluate_all(, .evaluate_handle(, wait_for_function(
_RE_EVALUATE_CALL = re.compile(
    r"\.(evaluate_all|evaluate_handle|evaluate)\(|"
    r"page\.wait_for_function\(",
)

# Regex: querySelector/querySelectorAll in JS (handles optional chaining ?.)
_RE_JS_QS = re.compile(
    r"\w+\??\.\bquerySelector(?:All)?\(\s*(?:'([^']*)'|\"([^\"]*)\")\s*\)",
)

# Regex: el.closest("sel"), el.matches("sel")
_RE_JS_CLOSEST_MATCHES = re.compile(
    r"\w+\.(?:closest|matches)\(\s*(?:'([^']*)'|\"([^\"]*)\")\s*\)",
)

# Regex: getElementById, getElementsByClassName, etc.
_RE_JS_GET_BY_ID = re.compile(
    r"document\.getElementById\(\s*(?:'([^']*)'|\"([^\"]*)\")\s*\)",
)
_RE_JS_GET_BY_CLASS = re.compile(
    r"document\.getElementsByClassName\(\s*(?:'([^']*)'|\"([^\"]*)\")\s*\)",
)
_RE_JS_GET_BY_TAG = re.compile(
    r"document\.getElementsByTagName\(\s*(?:'([^']*)'|\"([^\"]*)\")\s*\)",
)
_RE_JS_GET_BY_NAME = re.compile(
    r"document\.getElementsByName\(\s*(?:'([^']*)'|\"([^\"]*)\")\s*\)",
)

# Regex: JS-level .click() on a querySelector result (inline)
_RE_JS_QS_CLICK_INLINE = re.compile(
    r"querySelector\(\s*(?:'([^']*)'|\"([^\"]*)\")\s*\)(?:\??)\.click\(\)",
)

# Regex: JS variable tracking — assignment from querySelector then .click()
_RE_JS_VAR_QS = re.compile(
    r"(?:const|let|var)\s+(\w+)\s*=\s*\w+\??\.\bquerySelector\(\s*(?:'([^']*)'|\"([^\"]*)\")\s*\)",
)
_RE_JS_VAR_CLICK = re.compile(
    r"(\w+)\.click\(\)",
)

# Regex: JS-level navigation patterns
_RE_JS_NAV = re.compile(
    r"(?:window\.)?location(?:\.href)?\s*=|"
    r"location\.(?:assign|replace)\(|"
    r"\.submit\(\)",
)


def _extract_js_string(code: str, start_pos: int) -> str | None:
    """Extract the first string literal after start_pos (opening paren)."""
    i = start_pos
    length = len(code)
    # Skip whitespace
    while i < length and code[i] in " \t\n\r":
        i += 1
    if i >= length:
        return None
    # Check for f-string prefix
    if code[i] in "fF" and i + 1 < length and code[i + 1] in "\"'":
        return None
    # Skip r/b/u prefix
    if code[i] in "rRbBuU" and i + 1 < length and code[i + 1] in "\"'":
        i += 1
    # Triple-quoted
    for triple in ('"""', "'''"):
        if code[i : i + 3] == triple:
            end = code.find(triple, i + 3)
            if end != -1:
                return code[i + 3 : end]
            return None
    # Single/double quoted
    if code[i] in "\"'":
        quote = code[i]
        j = i + 1
        while j < length:
            if code[j] == "\\":
                j += 2
            elif code[j] == quote:
                return code[i + 1 : j]
            else:
                j += 1
    return None


def _extract_js_selectors(
    js_string: str,
    line: int,
) -> list[ExtractionResult]:
    """Extract CSS selectors from a JavaScript string."""
    results: list[ExtractionResult] = []
    seen: set[str] = set()

    # Detect JS-level clicks to mark selectors as navigating
    js_click_selectors: set[str] = set()
    # Inline: querySelector("sel").click()
    for m in _RE_JS_QS_CLICK_INLINE.finditer(js_string):
        sel = m.group(1) or m.group(2)
        if sel:
            js_click_selectors.add(sel)
    # Variable: const btn = querySelector("sel"); ... btn.click()
    js_var_sels: dict[str, str] = {}
    for m in _RE_JS_VAR_QS.finditer(js_string):
        var_name = m.group(1)
        sel = m.group(2) or m.group(3)
        if var_name and sel:
            js_var_sels[var_name] = sel
    for m in _RE_JS_VAR_CLICK.finditer(js_string):
        var_name = m.group(1)
        if var_name in js_var_sels:
            js_click_selectors.add(js_var_sels[var_name])

    bool(_RE_JS_NAV.search(js_string))

    # querySelector / querySelectorAll
    for m in _RE_JS_QS.finditer(js_string):
        sel = m.group(1) or m.group(2)
        if not sel or sel in seen:
            continue
        seen.add(sel)
        cat = "navigating" if sel in js_click_selectors else "passive"
        results.append(
            ExtractionResult(
                selector=sel,
                selector_type="css",
                action_category=cat,
                action="evaluate",
                line=line,
                source="evaluate_js",
            )
        )

    # el.closest("sel"), el.matches("sel")
    for m in _RE_JS_CLOSEST_MATCHES.finditer(js_string):
        sel = m.group(1) or m.group(2)
        if not sel or sel in seen:
            continue
        seen.add(sel)
        results.append(
            ExtractionResult(
                selector=sel,
                selector_type="css",
                action_category="passive",
                action="evaluate",
                line=line,
                source="evaluate_js",
            )
        )

    # getElementById → #id
    for m in _RE_JS_GET_BY_ID.finditer(js_string):
        val = m.group(1) or m.group(2)
        if not val:
            continue
        sel = f"#{val}"
        if sel in seen:
            continue
        seen.add(sel)
        results.append(
            ExtractionResult(
                selector=sel,
                selector_type="css",
                action_category="passive",
                action="evaluate",
                line=line,
                source="evaluate_js",
            )
        )

    # getElementsByClassName → .class
    for m in _RE_JS_GET_BY_CLASS.finditer(js_string):
        val = m.group(1) or m.group(2)
        if not val:
            continue
        sel = f".{val}"
        if sel in seen:
            continue
        seen.add(sel)
        results.append(
            ExtractionResult(
                selector=sel,
                selector_type="css",
                action_category="passive",
                action="evaluate",
                line=line,
                source="evaluate_js",
            )
        )

    # getElementsByTagName → tag
    for m in _RE_JS_GET_BY_TAG.finditer(js_string):
        val = m.group(1) or m.group(2)
        if not val:
            continue
        if val in seen:
            continue
        seen.add(val)
        results.append(
            ExtractionResult(
                selector=val,
                selector_type="css",
                action_category="passive",
                action="evaluate",
                line=line,
                source="evaluate_js",
            )
        )

    # getElementsByName → [name="..."]
    for m in _RE_JS_GET_BY_NAME.finditer(js_string):
        val = m.group(1) or m.group(2)
        if not val:
            continue
        sel = f'[name="{val}"]'
        if sel in seen:
            continue
        seen.add(sel)
        results.append(
            ExtractionResult(
                selector=sel,
                selector_type="css",
                action_category="passive",
                action="evaluate",
                line=line,
                source="evaluate_js",
            )
        )

    return results


def _extract_evaluate_js(code: str) -> list[ExtractionResult]:
    """Stage 2: Extract selectors from all page.evaluate() JS strings."""
    results: list[ExtractionResult] = []

    for m in _RE_EVALUATE_CALL.finditer(code):
        if _is_comment_line(code, m.start()):
            continue
        # Find the opening paren
        paren_pos = m.end()  # right after the '('
        js_string = _extract_js_string(code, paren_pos)
        if not js_string:
            continue
        line = _pos_to_line(code, m.start())
        results.extend(_extract_js_selectors(js_string, line))

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Stage 3: Context analysis
# ═══════════════════════════════════════════════════════════════════════

_RE_LOOP_HEADER = re.compile(r"^(\s*)(?:async\s+)?(?:for|while)\s+", re.MULTILINE)

_RE_NAV_ACTION = re.compile(
    r"\.(?:click|dblclick|tap)\(|"
    r"page\.(?:goto|go_back|go_forward|reload)\(",
)


def _detect_loops(code: str) -> list[tuple[int, int, bool]]:
    """Detect loops and whether they contain navigating actions.

    Returns list of (start_line, end_line, has_navigating_action).
    Lines are 1-based.
    """
    lines = code.split("\n")
    loops: list[tuple[int, int, bool]] = []

    for m in _RE_LOOP_HEADER.finditer(code):
        loop_indent = len(m.group(1))
        loop_start = _pos_to_line(code, m.start())

        # Find loop body extent via indentation
        loop_end = loop_start
        line_idx = loop_start  # 0-based index = loop_start (since loop_start is 1-based)
        for j in range(line_idx, len(lines)):
            body_line = lines[j]
            if body_line.strip() == "":
                continue
            body_indent = len(body_line) - len(body_line.lstrip())
            if body_indent <= loop_indent:
                break
            loop_end = j + 1

        # Check for navigating actions in loop body
        body_text = "\n".join(lines[loop_start:loop_end])  # lines after header
        has_nav = bool(_RE_NAV_ACTION.search(body_text))

        loops.append((loop_start, loop_end, has_nav))

    return loops


def _analyze_context(
    python_results: list[ExtractionResult],
    js_results: list[ExtractionResult],
    nav_markers: list[tuple[int, str]],
    code: str,
) -> list[ExtractionResult]:
    """Stage 3: Apply loop detection and navigation boundaries."""
    all_results = python_results + js_results
    all_results.sort(key=lambda r: r.line)

    if not all_results:
        return []

    loops = _detect_loops(code)
    nav_lines = sorted(line for line, _ in nav_markers)

    final: list[ExtractionResult] = []
    for r in all_results:
        in_loop = any(start <= r.line <= end and has_nav for start, end, has_nav in loops)
        after_nav = any(nav_line < r.line for nav_line in nav_lines)

        if in_loop != r.in_loop or after_nav != r.after_navigation:
            r = replace(r, in_loop=in_loop, after_navigation=after_nav)
        final.append(r)

    return final


# ═══════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════


def extract_selectors(code: str) -> list[ExtractionResult]:
    """Extract all DOM selectors from an AI-written Python code block.

    Analyzes both Python-level Playwright API calls and JavaScript
    selectors inside ``page.evaluate()`` calls.  Returns results
    annotated with action classification, loop detection, and
    navigation boundary analysis.

    Args:
        code: Raw Python code string from the AI's tool call.

    Returns:
        List of :class:`ExtractionResult`, sorted by line number.
        Empty list if no selectors found.
    """
    py_results, nav_markers, _var_dict = _extract_python_level(code)
    js_results = _extract_evaluate_js(code)
    return _analyze_context(py_results, js_results, nav_markers, code)
