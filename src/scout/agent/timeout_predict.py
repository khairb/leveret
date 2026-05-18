"""Static timeout prediction for agent-submitted Python code.

Parses the code string with ``ast`` (no execution) and walks the tree to
identify Patchright/Playwright API calls, explicit waits, loops, and other
patterns that consume wall-clock time.  Returns a predicted timeout in
seconds, always in the range [BASELINE, MAX_TIMEOUT].

The prediction uses **two independent signals** and takes the maximum:

1. **AST Scorer** — walks statements and sums per-call costs from a lookup
   table.  Accurate for simple sequential code but blind to function
   bodies (it skips them by design).

2. **Await Counter** — walks the *entire* tree (including function bodies)
   and counts ``await`` expressions, multiplying by enclosing loop
   iteration counts.  Catches patterns the scorer misses (user-defined
   function calls, comprehension awaits, exotic APIs not in the cost
   table).

Usage::

    from scout.agent.timeout_predict import predict_timeout

    timeout = predict_timeout(code_string)
"""

from __future__ import annotations

import ast

# ── Constants ────────────────────────────────────────────────────

BASELINE: float = 30.0
"""Minimum timeout — we never go below this."""

MAX_TIMEOUT: float = 3000.0
"""Hard ceiling — absolute maximum allowed execution time."""

MARGIN: float = 1.5
"""Safety multiplier applied to the raw estimate."""

# Heuristic iteration counts when we can't resolve the real value.
_DEFAULT_FOR_ITERATIONS: int = 10
_DEFAULT_WHILE_ITERATIONS: int = 10
_MAX_LOOP_ITERATIONS: int = 50
_MAX_LOOP_CONTRIBUTION: float = 225.0

# ── Per-call cost table (seconds) ───────────────────────────────

# Keys are method/function names matched on ``ast.Attribute.attr`` or
# ``ast.Name.id``.  Values are a float (fixed cost in seconds).

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
    # Element handle / query
    "query_selector": 3.0,
    "query_selector_all": 3.0,
    "evaluate_handle": 3.0,
    "element_handle": 3.0,
    # Locator collection
    "all": 2.0,
    # Scrolling / viewport
    "scroll_into_view_if_needed": 2.0,
    "wheel": 2.0,
    # Keyboard
    "type": 2.0,
    # Context-manager based waits
    "expect_navigation": 10.0,
    "expect_response": 10.0,
    "expect_request": 10.0,
    # Injected builtins
    "show_page": 25.0,
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
    "wait_for_timeout": "ms",  # first arg in milliseconds
    "sleep": "s",  # asyncio.sleep — seconds
}

# Scroll helper with ``max_scrolls`` kwarg.
_SCROLL_HELPER = "scroll_to_bottom"
_SCROLL_DEFAULT_COST: float = 15.0
_SCROLL_PER_ITERATION: float = 1.0

# ── Await counter constants ─────────────────────────────────────

_PER_AWAIT_COST: float = 6.0
"""Flat cost per ``await`` expression in the await-counter floor."""

_MAX_AWAIT_MULTIPLIER: int = 50
"""Cap on the effective loop multiplier for any single await."""


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


# ── AST Scoring engine ──────────────────────────────────────────


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
    #    NOTE: the await counter (below) DOES walk into these.
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

    return 0.0


# ── Await counter (safety floor) ────────────────────────────────


def _count_await_floor(tree: ast.Module) -> float:
    """Walk the *entire* AST and sum a flat cost per ``await``.

    Unlike the AST scorer, this visitor enters function/class bodies,
    so it catches awaits hidden inside user-defined functions that the
    scorer skips.  Each ``await`` is weighted by the product of its
    enclosing loop iteration counts.

    This produces a conservative floor that complements the detailed
    scorer: it is less accurate for known Playwright calls but never
    misses an ``await``, regardless of the method name.
    """
    total = 0.0

    def _visit(node: ast.AST, loop_mul: int = 1) -> None:
        nonlocal total
        effective = min(loop_mul, _MAX_AWAIT_MULTIPLIER)

        if isinstance(node, ast.Await):
            total += _PER_AWAIT_COST * effective
            # Recurse into the awaited value — it may contain nested
            # awaits like ``await page.fill(await get_sel(), "text")``
            _visit(node.value, loop_mul)
            return

        if isinstance(node, ast.For):
            bound = _extract_range_bound(node)
            iters = bound if bound is not None else _DEFAULT_FOR_ITERATIONS
            iters = min(iters, _MAX_LOOP_ITERATIONS)
            for child in ast.iter_child_nodes(node):
                _visit(child, loop_mul * iters)
            return

        if isinstance(node, ast.AsyncFor):
            for child in ast.iter_child_nodes(node):
                _visit(child, loop_mul * _DEFAULT_FOR_ITERATIONS)
            return

        if isinstance(node, ast.While):
            for child in ast.iter_child_nodes(node):
                _visit(child, loop_mul * _DEFAULT_WHILE_ITERATIONS)
            return

        # Comprehensions: each generator acts as a loop level.
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            comp_mul = loop_mul
            for _gen in node.generators:
                comp_mul *= _DEFAULT_FOR_ITERATIONS
            # Element expression(s) run inside all generators.
            if isinstance(node, ast.DictComp):
                _visit(node.key, comp_mul)
                _visit(node.value, comp_mul)
            else:
                _visit(node.elt, comp_mul)
            # Generator iterables/conditions run at the enclosing level.
            for gen in node.generators:
                _visit(gen.iter, loop_mul)
                for if_ in gen.ifs:
                    _visit(if_, loop_mul)
            return

        # Default: recurse into ALL children.
        # This deliberately enters FunctionDef / AsyncFunctionDef /
        # ClassDef bodies — the key difference from the AST scorer.
        for child in ast.iter_child_nodes(node):
            _visit(child, loop_mul)

    _visit(tree)
    return total


# ── Public API ───────────────────────────────────────────────────


def _build_context_code(
    code: str,
    tree: ast.Module,
    function_sources: dict[str, str],
) -> str | None:
    """If *code* calls functions from *function_sources*, return a
    combined string with the referenced definitions prepended.

    Returns ``None`` if no external functions are referenced.
    """
    # Collect all called names that are in function_sources.
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            _, name = _call_name(node)
            if name in function_sources:
                called.add(name)

    if not called:
        return None

    parts = [function_sources[name] for name in sorted(called)]
    parts.append(code)
    return "\n\n".join(parts)


def predict_timeout(
    code: str,
    function_sources: dict[str, str] | None = None,
) -> float:
    """Predict a timeout (in seconds) for agent-submitted Python code.

    Returns a value in ``[BASELINE, MAX_TIMEOUT]``.  If the code cannot
    be parsed, returns ``BASELINE`` (the code will fail fast on a syntax
    error anyway).

    Parameters
    ----------
    code:
        The Python code string to analyse.
    function_sources:
        Optional mapping of ``{name: source}`` for functions defined in
        previous REPL steps.  When provided, any calls to these functions
        will be scored as if the definitions were part of *code*.

    Uses two signals — a detailed AST scorer and a coarse await counter
    — and takes the maximum to avoid blind spots.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return BASELINE

    # If the code references user-defined functions from earlier steps,
    # build a combined tree so both scorers can see the function bodies.
    effective_tree = tree
    if function_sources:
        combined = _build_context_code(code, tree, function_sources)
        if combined is not None:
            try:
                effective_tree = ast.parse(combined)
            except SyntaxError:
                pass  # fall back to original tree

    ast_scored = _score_body(effective_tree.body)
    await_floor = _count_await_floor(effective_tree)
    raw = max(ast_scored, await_floor)
    estimate = raw * MARGIN
    return max(BASELINE, min(estimate, MAX_TIMEOUT))
