"""Static timeout prediction for agent-submitted Python code.

Parses the code string with ``ast`` (no execution) and walks the tree to
identify Patchright/Playwright API calls, explicit waits, loops, and other
patterns that consume wall-clock time.  Returns a predicted timeout in
seconds, always in the range [BASELINE, MAX_TIMEOUT].

Usage::

    from scout.agent.timeout_predict import predict_timeout

    timeout = predict_timeout(code_string)
"""

from __future__ import annotations

import ast
from typing import Union

# ── Constants ────────────────────────────────────────────────────

BASELINE: float = 20.0
"""Minimum timeout — we never go below this."""

MAX_TIMEOUT: float = 300.0
"""Hard ceiling — absolute maximum allowed execution time."""

MARGIN: float = 1.2
"""Safety multiplier applied to the raw estimate."""

# Heuristic iteration counts when we can't resolve the real value.
_DEFAULT_FOR_ITERATIONS: int = 5
_DEFAULT_WHILE_ITERATIONS: int = 10
_MAX_LOOP_ITERATIONS: int = 50
_MAX_LOOP_CONTRIBUTION: float = 225.0

# ── Per-call cost table (seconds) ───────────────────────────────

# Keys are method/function names matched on ``ast.Attribute.attr`` or
# ``ast.Name.id``.  Values are either a float (fixed cost) or the
# string ``"dynamic"`` meaning we need to inspect arguments.

_FIXED_COSTS: dict[str, float] = {
    # Navigation — involves network round-trip + page rendering.
    "goto": 10.0,
    "reload": 10.0,
    "go_back": 8.0,
    "go_forward": 8.0,
    # Waits (defaults — overridden when a timeout kwarg is present)
    "wait_for_load_state": 8.0,
    "wait_for_url": 10.0,
    # Interaction — each waits for the selector, then performs action.
    # If the selector is wrong the per-op timeout (10 s) is hit.
    "click": 5.0,
    "dblclick": 5.0,
    "fill": 3.0,
    "press": 2.0,
    "select_option": 4.0,
    "hover": 3.0,
    "check": 3.0,
    "uncheck": 3.0,
    "set_checked": 3.0,
    # Extraction — waits for selector, then reads DOM.
    "content": 3.0,
    "evaluate": 3.0,
    "evaluate_all": 4.0,
    "inner_text": 5.0,
    "text_content": 5.0,
    "inner_html": 5.0,
    "all_inner_texts": 5.0,
    "all_text_contents": 5.0,
    "get_attribute": 5.0,
    "input_value": 5.0,
    "bounding_box": 3.0,
    "screenshot": 5.0,
    # Query — near-instant but keep a small budget.
    "count": 1.0,
    "is_visible": 1.0,
    "is_enabled": 1.0,
    "is_checked": 1.0,
    # Context-manager based waits
    "expect_navigation": 10.0,
    "expect_response": 10.0,
    "expect_request": 10.0,
    # Injected builtins
    "show_page": 15.0,
    "zoom_section": 5.0,
}

# These calls have a ``timeout`` keyword (in milliseconds) that we try
# to extract from the AST.  The value here is the *default* cost when
# the kwarg is absent.
_TIMEOUT_KWARG_COSTS: dict[str, float] = {
    "wait_for_selector": 5.0,
    "wait_for_function": 5.0,
    "wait_for": 5.0,
}

# These calls take an explicit duration as their first positional arg.
_DURATION_ARG_COSTS: dict[str, str] = {
    "wait_for_timeout": "ms",   # first arg in milliseconds
    "sleep": "s",               # asyncio.sleep — seconds
}

# Scroll helper with ``max_scrolls`` kwarg.
_SCROLL_HELPER = "scroll_to_bottom"
_SCROLL_DEFAULT_COST: float = 15.0
_SCROLL_PER_ITERATION: float = 1.0


# ── Helpers ──────────────────────────────────────────────────────

def _resolve_constant(node: ast.expr) -> float | int | None:
    """Try to resolve a node to a numeric constant.

    Handles: plain constants, unary minus, and simple underscore
    separators (``10_000`` is parsed by Python as ``10000``).
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _resolve_constant(node.operand)
        if inner is not None:
            return -inner
    return None


def _get_timeout_kwarg(call: ast.Call, default: float) -> float:
    """Extract ``timeout=<ms>`` from a call's keyword arguments.

    Returns the value converted to *seconds*, or *default* if not found
    or not a static constant.
    """
    for kw in call.keywords:
        if kw.arg == "timeout":
            val = _resolve_constant(kw.value)
            if val is not None and val > 0:
                return val / 1000.0  # ms → s
    return default


def _get_first_positional(call: ast.Call) -> float | int | None:
    """Return the first positional argument as a number, or ``None``."""
    if call.args:
        return _resolve_constant(call.args[0])
    return None


def _get_max_scrolls(call: ast.Call) -> float:
    """Extract ``max_scrolls=N`` from scroll_to_bottom call."""
    for kw in call.keywords:
        if kw.arg == "max_scrolls":
            val = _resolve_constant(kw.value)
            if val is not None and val > 0:
                return min(val, _MAX_LOOP_ITERATIONS) * _SCROLL_PER_ITERATION
    return _SCROLL_DEFAULT_COST


def _call_name(node: ast.Call) -> tuple[str | None, str]:
    """Return ``(object_name_or_None, method_name)`` for a Call node.

    Examples::

        page.goto(...)          → ("page", "goto")
        asyncio.sleep(...)      → ("asyncio", "sleep")
        show_page(page)         → (None, "show_page")
        loc.click()             → ("loc", "click")
        page.locator('.x').click() → (None, "click")  (chained)
    """
    func = node.func
    if isinstance(func, ast.Attribute):
        attr = func.attr
        if isinstance(func.value, ast.Name):
            return func.value.id, attr
        # Chained call like page.locator(...).click() — we still want "click".
        return None, attr
    if isinstance(func, ast.Name):
        return None, func.id
    return None, ""


# ── Scoring engine ───────────────────────────────────────────────

def _score_call(node: ast.Call) -> float:
    """Score a single call node."""
    obj, name = _call_name(node)

    # scroll_to_bottom helper
    if name == _SCROLL_HELPER:
        return _get_max_scrolls(node)

    # Duration-argument calls (wait_for_timeout, asyncio.sleep)
    if name in _DURATION_ARG_COSTS:
        unit = _DURATION_ARG_COSTS[name]
        val = _get_first_positional(node)
        if val is not None and val > 0:
            return val / 1000.0 if unit == "ms" else float(val)
        return 5.0  # can't resolve → conservative default

    # Timeout-kwarg calls (wait_for_selector, wait_for_function, wait_for)
    if name in _TIMEOUT_KWARG_COSTS:
        return _get_timeout_kwarg(node, _TIMEOUT_KWARG_COSTS[name])

    # Fixed-cost calls
    if name in _FIXED_COSTS:
        return _FIXED_COSTS[name]

    return 0.0


def _extract_range_bound(node: ast.For) -> int | None:
    """If the for-loop is ``for x in range(N)`` return N as an int."""
    iter_node = node.iter
    if not isinstance(iter_node, ast.Call):
        return None
    func = iter_node.func
    if not (isinstance(func, ast.Name) and func.id == "range"):
        return None
    # range(stop) or range(start, stop)
    args = iter_node.args
    if len(args) == 1:
        val = _resolve_constant(args[0])
        return int(val) if val is not None and val > 0 else None
    if len(args) >= 2:
        start = _resolve_constant(args[0])
        stop = _resolve_constant(args[1])
        if start is not None and stop is not None:
            return max(0, int(stop) - int(start))
    return None


def _score_body(stmts: list[ast.stmt]) -> float:
    """Score a list of statements (a block body)."""
    total = 0.0
    for stmt in stmts:
        total += _score_stmt(stmt)
    return total


def _score_stmt(node: ast.stmt) -> float:
    """Recursively score a single statement."""

    # ── Loops ────────────────────────────────────────────────
    if isinstance(node, ast.For):
        bound = _extract_range_bound(node)
        iterations = bound if bound is not None else _DEFAULT_FOR_ITERATIONS
        iterations = min(iterations, _MAX_LOOP_ITERATIONS)
        body_cost = _score_body(node.body)
        return min(iterations * body_cost, _MAX_LOOP_CONTRIBUTION)

    if isinstance(node, ast.While):
        body_cost = _score_body(node.body)
        return min(_DEFAULT_WHILE_ITERATIONS * body_cost, _MAX_LOOP_CONTRIBUTION)

    # ── Async for ────────────────────────────────────────────
    if isinstance(node, ast.AsyncFor):
        iterations = _DEFAULT_FOR_ITERATIONS
        body_cost = _score_body(node.body)
        return min(iterations * body_cost, _MAX_LOOP_CONTRIBUTION)

    # ── Async with (expect_navigation, expect_response) ──────
    if isinstance(node, ast.AsyncWith):
        cost = 0.0
        for item in node.items:
            ctx = item.context_expr
            if isinstance(ctx, ast.Call):
                cost += _score_call(ctx)
        cost += _score_body(node.body)
        return cost

    # ── With statement ───────────────────────────────────────
    if isinstance(node, ast.With):
        cost = 0.0
        for item in node.items:
            ctx = item.context_expr
            if isinstance(ctx, ast.Call):
                cost += _score_call(ctx)
        cost += _score_body(node.body)
        return cost

    # ── Try/except — score the try body ──────────────────────
    if isinstance(node, ast.Try):
        cost = _score_body(node.body)
        for handler in node.handlers:
            cost = max(cost, _score_body(handler.body))
        return cost

    # ── If/elif/else — take the max branch ───────────────────
    if isinstance(node, ast.If):
        branch_costs = [_score_body(node.body)]
        if node.orelse:
            branch_costs.append(_score_body(node.orelse))
        return max(branch_costs)

    # ── Function / class definitions — don't score the body
    #    (it only runs if called, and we score calls separately)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return 0.0

    # ── Expression statements (most agent code is this) ──────
    if isinstance(node, ast.Expr):
        return _score_expr(node.value)

    # ── Assignments: x = await page.evaluate(...) ────────────
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        val = node.value if isinstance(node, ast.Assign) else node.value
        if val is not None:
            return _score_expr(val)
        return 0.0

    # ── Augmented assignment: items += await ... ─────────────
    if isinstance(node, ast.AugAssign):
        return _score_expr(node.value)

    return 0.0


def _score_expr(node: ast.expr) -> float:
    """Score an expression node."""
    if isinstance(node, ast.Await):
        return _score_expr(node.value)

    if isinstance(node, ast.Call):
        # Score the call itself.
        cost = _score_call(node)
        # Also score any call arguments that are themselves awaited calls.
        # e.g. ``await page.fill(await get_selector(), "text")``
        for arg in node.args:
            cost += _score_expr(arg)
        for kw in node.keywords:
            cost += _score_expr(kw.value)
        return cost

    # Chained attribute calls: ``(await resp_info.value).json()``
    # is NamedExpr or nested Await — handle Await above covers it.

    # List/set/dict comprehensions with await inside are unusual but
    # possible. We skip them — agent code rarely uses comprehensions
    # with awaits.

    return 0.0


# ── Public API ───────────────────────────────────────────────────

def predict_timeout(code: str) -> float:
    """Predict a timeout (in seconds) for agent-submitted Python code.

    Returns a value in ``[BASELINE, MAX_TIMEOUT]``.  If the code cannot
    be parsed, returns ``BASELINE`` (the code will fail fast on a syntax
    error anyway).

    This function is purely static — it never executes the code.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return BASELINE

    raw = _score_body(tree.body)
    estimate = raw * MARGIN
    return max(BASELINE, min(estimate, MAX_TIMEOUT))
