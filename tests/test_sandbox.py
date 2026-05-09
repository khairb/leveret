"""Comprehensive tests for the sandbox module.

Tests compilation, import whitelist, builtin restrictions, asyncio proxy,
print passthrough, and compatibility with real AI-generated scrapers.
"""

import io
import sys
import textwrap
from pathlib import Path

import pytest

from scout.runtime.sandbox import (
    ALLOWED_MODULES,
    SandboxError,
    PassthroughPrintCollector,
    _safe_asyncio,
    _safe_import,
    build_restricted_builtins,
    build_restricted_globals,
    build_safe_pre_imports,
    compile_restricted_agent_code,
)


# ═══════════════════════════════════════════════════════════════
#  A. Compile existing scrapers
# ═══════════════════════════════════════════════════════════════

SCRAPERS_DIR = Path(__file__).parent.parent / "scrapers"


@pytest.mark.parametrize(
    "scraper_path",
    sorted(SCRAPERS_DIR.glob("*.py")) if SCRAPERS_DIR.exists() else [],
    ids=lambda p: p.name,
)
def test_existing_scrapers_compile(scraper_path):
    """Every existing scraper must compile through the sandbox."""
    source = scraper_path.read_text(encoding="utf-8")
    # Should NOT raise
    code = compile_restricted_agent_code(source, filename=str(scraper_path))
    assert code is not None


# ═══════════════════════════════════════════════════════════════
#  B. Blocked patterns (must raise SandboxError or ImportError)
# ═══════════════════════════════════════════════════════════════

BLOCKED_COMPILE_PATTERNS = [
    # exec/eval blocked at compile time by RestrictedPython
    ("exec('print(1)')", "exec call"),
    ("eval('1+1')", "eval call"),
    ("from os import *", "star import"),
]

@pytest.mark.parametrize("code,desc", BLOCKED_COMPILE_PATTERNS, ids=lambda x: x if isinstance(x, str) and len(x) < 50 else "")
def test_blocked_compile_patterns(code, desc):
    """Patterns blocked at compile time."""
    with pytest.raises(SandboxError):
        compile_restricted_agent_code(code)


BLOCKED_IMPORT_MODULES = [
    "os", "subprocess", "sys", "shutil", "tempfile", "pathlib",
    "socket", "ctypes", "importlib", "pickle", "shelve", "marshal",
    "inspect", "webbrowser", "multiprocessing", "threading", "signal",
    "io", "gzip", "bz2", "lzma", "uuid",
    "urllib", "urllib.request", "http.server", "http.client",
    "code", "codeop", "ast", "runpy", "gc",
]

@pytest.mark.parametrize("module", BLOCKED_IMPORT_MODULES)
def test_blocked_imports(module):
    """Dangerous modules are blocked by the import whitelist."""
    with pytest.raises(ImportError):
        _safe_import(module)


def test_blocked_relative_import():
    """Relative imports are blocked."""
    with pytest.raises(ImportError, match="Relative"):
        _safe_import("foo", level=1)


def test_blocked_open_builtin():
    """open() is not in restricted builtins."""
    builtins = build_restricted_builtins()
    assert "open" not in builtins


def test_blocked_exec_eval_builtins():
    """exec/eval/compile are not in restricted builtins."""
    builtins = build_restricted_builtins()
    for name in ("exec", "eval", "compile", "breakpoint"):
        assert name not in builtins


# ═══════════════════════════════════════════════════════════════
#  B2. Asyncio safety
# ═══════════════════════════════════════════════════════════════

def test_asyncio_proxy_has_safe_functions():
    """Safe asyncio functions are available."""
    assert hasattr(_safe_asyncio, "sleep")
    assert hasattr(_safe_asyncio, "wait_for")
    assert hasattr(_safe_asyncio, "gather")
    assert hasattr(_safe_asyncio, "TimeoutError")
    assert hasattr(_safe_asyncio, "create_task")
    assert hasattr(_safe_asyncio, "iscoroutine")


def test_asyncio_proxy_blocks_dangerous_functions():
    """Dangerous asyncio functions are NOT on the proxy."""
    assert not hasattr(_safe_asyncio, "create_subprocess_exec")
    assert not hasattr(_safe_asyncio, "create_subprocess_shell")
    assert not hasattr(_safe_asyncio, "open_connection")
    assert not hasattr(_safe_asyncio, "start_server")
    assert not hasattr(_safe_asyncio, "open_unix_connection")
    assert not hasattr(_safe_asyncio, "start_unix_server")


def test_asyncio_import_returns_proxy():
    """import asyncio returns the safe proxy, not the real module."""
    result = _safe_import("asyncio")
    assert result is _safe_asyncio
    assert not hasattr(result, "create_subprocess_exec")


# ═══════════════════════════════════════════════════════════════
#  C. Allowed patterns (must compile successfully)
# ═══════════════════════════════════════════════════════════════

ALLOWED_PATTERNS = [
    # Basic async
    (
        "async def scrape(page, s, c):\n    await page.goto(s)",
        "async def/await",
    ),
    # F-strings
    (
        "x = f'hello {1+1}'",
        "f-string",
    ),
    # List/dict comprehension
    (
        "x = [i for i in range(10)]",
        "list comprehension",
    ),
    (
        "x = {k: v for k, v in {'a': 1}.items()}",
        "dict comprehension",
    ),
    # Nested async function
    (
        textwrap.dedent("""\
        async def scrape(page, s, c):
            async def inner():
                await page.click('a')
            await inner()
        """),
        "nested async function",
    ),
    # Try/except/raise
    (
        textwrap.dedent("""\
        try:
            x = 1
        except Exception as e:
            raise ValueError('fail')
        """),
        "try/except/raise",
    ),
    # For with unpacking
    (
        "for k, v in {'a': 1}.items():\n    pass",
        "for with unpacking",
    ),
    # Augmented assignment on name
    (
        "x = 0\nx += 1",
        "augmented assignment name",
    ),
    # Augmented assignment on item
    (
        "d = {}\nd['x'] = 0\nd['x'] += 1",
        "augmented assignment item",
    ),
    # Augmented assignment on attribute
    (
        textwrap.dedent("""\
        class Obj:
            count = 0
        o = Obj()
        o.count += 1
        """),
        "augmented assignment attribute",
    ),
    # Set operations
    (
        "s = set()\ns.add(1)",
        "set operations",
    ),
    # Underscore variables
    (
        "_items = []\n_count = 0\n_temp = 'hello'",
        "underscore variables",
    ),
    # Nonlocal
    (
        textwrap.dedent("""\
        def outer():
            x = 0
            def inner():
                nonlocal x
                x = 1
            inner()
        """),
        "nonlocal",
    ),
    # Import allowed modules
    (
        "import re\nimport json\nx = re.search(r'\\\\d+', '123')\ny = json.dumps({'a': 1})",
        "import re and json",
    ),
    # Page object methods (compile-time only, no execution)
    (
        textwrap.dedent("""\
        async def scrape(page, s, c):
            await page.goto(s)
            await page.click('button')
            items = await page.evaluate('() => []')
            loc = page.locator('div')
            count = await loc.count()
            return items
        """),
        "page object methods",
    ),
    # Lambda
    (
        "f = lambda x: x + 1\nresult = f(2)",
        "lambda",
    ),
    # Walrus operator
    (
        "if (n := 10) > 5:\n    pass",
        "walrus operator",
    ),
    # Star unpacking
    (
        "a, *rest = [1, 2, 3, 4, 5]",
        "star unpacking",
    ),
    # Async for
    (
        textwrap.dedent("""\
        async def scrape(page, s, c):
            async for item in page.locator('div').all():
                pass
        """),
        "async for",
    ),
    # Async with
    (
        textwrap.dedent("""\
        async def scrape(page, s, c):
            async with page.expect_navigation():
                await page.click('a')
        """),
        "async with",
    ),
    # Generator expression
    (
        "total = sum(x * 2 for x in range(10))",
        "generator expression",
    ),
    # Multiple assignment
    (
        "a = b = []",
        "multiple assignment",
    ),
    # Chained comparison
    (
        "x = 5\nresult = 0 < x < 10",
        "chained comparison",
    ),
    # Global statement
    (
        textwrap.dedent("""\
        _counter = 0
        def inc():
            global _counter
            _counter += 1
        """),
        "global statement",
    ),
    # Yield
    (
        textwrap.dedent("""\
        def gen():
            for i in range(10):
                yield i
        """),
        "yield",
    ),
    # Complex scraper pattern
    (
        textwrap.dedent("""\
        async def scrape(page, start_url, checkpoint):
            import re
            from collections import defaultdict

            rating_map = {"One": 1, "Two": 2, "Three": 3}
            all_items = []
            seen_urls = set()
            page_num = 1

            await page.goto(start_url, wait_until="domcontentloaded")
            await page.wait_for_selector("article", timeout=30000)
            await checkpoint("loaded")

            while True:
                items = await page.evaluate(
                    "() => Array.from(document.querySelectorAll('article')).map(a => a.innerText)"
                )
                for item in items:
                    m = re.search(r'\\\\d+\\\\.\\\\d+', item)
                    if m:
                        all_items.append({"text": item, "price": float(m.group())})

                next_btn = page.locator('a.next')
                if await next_btn.count() == 0:
                    break

                await next_btn.click()
                page_num += 1
                await checkpoint(f"page_{page_num}")

            return all_items
        """),
        "complex scraper pattern",
    ),
]


@pytest.mark.parametrize("code,desc", ALLOWED_PATTERNS, ids=lambda x: x if isinstance(x, str) and len(x) < 50 else "")
def test_allowed_patterns_compile(code, desc):
    """Pattern must compile through sandbox without errors."""
    result = compile_restricted_agent_code(code)
    assert result is not None


# ═══════════════════════════════════════════════════════════════
#  D. Import whitelist at runtime
# ═══════════════════════════════════════════════════════════════

ALLOWED_IMPORT_MODULES = [
    "json", "re", "math", "collections", "itertools", "functools",
    "datetime", "time", "calendar",
    "html", "html.parser", "html.entities",
    "base64", "hashlib", "csv", "string", "textwrap",
    "copy", "decimal", "random", "enum", "dataclasses",
    "typing", "zlib", "pprint", "contextlib", "abc",
    "statistics", "difflib", "unicodedata", "fractions",
    "binascii", "hmac", "operator",
    "urllib.parse",
    "collections.abc",  # submodule of allowed parent
]


@pytest.mark.parametrize("module", ALLOWED_IMPORT_MODULES)
def test_allowed_imports(module):
    """Whitelisted modules can be imported."""
    result = _safe_import(module)
    assert result is not None


def test_import_runtime_in_restricted_globals():
    """Code importing allowed modules runs in restricted globals."""
    code = compile_restricted_agent_code(
        "import collections\nx = collections.Counter([1, 1, 2, 3])"
    )
    ns = build_restricted_globals(build_safe_pre_imports())
    exec(code, ns)
    assert ns["x"].most_common(1) == [(1, 2)]


def test_import_blocked_runtime_in_restricted_globals():
    """Code importing blocked modules fails in restricted globals."""
    code = compile_restricted_agent_code("import os")
    ns = build_restricted_globals(build_safe_pre_imports())
    with pytest.raises(ImportError):
        exec(code, ns)


# ═══════════════════════════════════════════════════════════════
#  E. Print passthrough
# ═══════════════════════════════════════════════════════════════

def test_print_passthrough():
    """print() in restricted code goes to real stdout."""
    code = compile_restricted_agent_code('print("sandbox_test_output")')
    ns = build_restricted_globals(build_safe_pre_imports())

    # Capture stdout
    captured = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = captured
        exec(code, ns)
    finally:
        sys.stdout = old_stdout

    assert "sandbox_test_output" in captured.getvalue()


def test_print_collector_class():
    """PassthroughPrintCollector has required interface."""
    pc = PassthroughPrintCollector()
    assert hasattr(pc, "_call_print")
    assert callable(pc._call_print)
    assert pc() == ""


# ═══════════════════════════════════════════════════════════════
#  F. Builtins verification
# ═══════════════════════════════════════════════════════════════

def test_builtins_include_essentials():
    """Essential builtins are available."""
    builtins = build_restricted_builtins()
    essentials = [
        "dict", "list", "set", "frozenset",
        "enumerate", "filter", "map", "max", "min", "sum",
        "all", "any", "hasattr", "getattr",
        "type", "isinstance", "issubclass",
        "print", "sorted", "zip", "len", "range",
        "int", "float", "str", "bool", "bytes",
        "abs", "round", "repr", "ord", "chr",
        "iter", "next", "reversed",
        "super", "object", "property",
        "Exception", "ValueError", "TypeError", "KeyError",
        "IndexError", "RuntimeError", "StopIteration",
    ]
    for name in essentials:
        assert name in builtins, f"Missing essential builtin: {name}"


def test_builtins_exclude_dangerous():
    """Dangerous builtins are NOT available."""
    builtins = build_restricted_builtins()
    for name in ("exec", "eval", "compile", "open", "breakpoint",
                  "globals", "locals", "vars", "dir", "input"):
        assert name not in builtins, f"Dangerous builtin present: {name}"


# ═══════════════════════════════════════════════════════════════
#  G. End-to-end execution in restricted globals
# ═══════════════════════════════════════════════════════════════

def test_execute_simple_function():
    """A simple function compiles and executes in the sandbox."""
    source = textwrap.dedent("""\
    def compute():
        items = [{"name": "a", "price": 1.0}, {"name": "b", "price": 2.0}]
        total = sum(item["price"] for item in items)
        return total
    """)
    code = compile_restricted_agent_code(source)
    ns = build_restricted_globals(build_safe_pre_imports())
    exec(code, ns)
    assert ns["compute"]() == 3.0


def test_execute_with_allowed_imports():
    """Code with whitelisted imports executes correctly."""
    source = textwrap.dedent("""\
    import re
    import collections

    def extract():
        text = "price: $10.99, price: $20.50, price: $10.99"
        prices = re.findall(r'\\$([\\d.]+)', text)
        counts = collections.Counter(prices)
        return dict(counts)
    """)
    code = compile_restricted_agent_code(source)
    ns = build_restricted_globals(build_safe_pre_imports())
    exec(code, ns)
    result = ns["extract"]()
    assert result == {"10.99": 2, "20.50": 1}


def test_execute_with_pre_imported_modules():
    """Pre-imported modules (json, re, etc.) work."""
    source = textwrap.dedent("""\
    def process():
        data = json.dumps({"key": "value"})
        parsed = json.loads(data)
        match = re.search(r'key', data)
        return parsed["key"], match is not None
    """)
    code = compile_restricted_agent_code(source)
    ns = build_restricted_globals(build_safe_pre_imports())
    exec(code, ns)
    val, matched = ns["process"]()
    assert val == "value"
    assert matched is True


def test_execute_with_stringio():
    """Pre-imported StringIO works for csv parsing."""
    source = textwrap.dedent("""\
    import csv

    def parse_csv():
        data = "name,price\\nBook,10.99\\nPen,1.50"
        reader = csv.reader(StringIO(data))
        rows = list(reader)
        return rows
    """)
    code = compile_restricted_agent_code(source)
    ns = build_restricted_globals(build_safe_pre_imports())
    exec(code, ns)
    rows = ns["parse_csv"]()
    assert len(rows) == 3
    assert rows[1] == ["Book", "10.99"]


def test_execute_inplace_operations():
    """Augmented assignment works in sandbox."""
    source = textwrap.dedent("""\
    def compute():
        x = 10
        x += 5
        x -= 2
        x *= 3
        return x
    """)
    code = compile_restricted_agent_code(source)
    ns = build_restricted_globals(build_safe_pre_imports())
    exec(code, ns)
    assert ns["compute"]() == 39  # (10 + 5 - 2) * 3


# ═══════════════════════════════════════════════════════════════
#  H. Pre-imports safety
# ═══════════════════════════════════════════════════════════════

def test_safe_pre_imports_no_dangerous_modules():
    """Safe pre-imports do NOT include os, shutil, tempfile."""
    pre = build_safe_pre_imports()
    for name in ("os", "shutil", "tempfile", "subprocess", "sys"):
        assert name not in pre, f"Dangerous module in pre-imports: {name}"


def test_safe_pre_imports_asyncio_is_proxy():
    """asyncio in pre-imports is the safe proxy."""
    pre = build_safe_pre_imports()
    assert pre["asyncio"] is _safe_asyncio
    assert not hasattr(pre["asyncio"], "create_subprocess_exec")


def test_safe_pre_imports_has_stringio():
    """StringIO and BytesIO are pre-imported."""
    pre = build_safe_pre_imports()
    assert pre["StringIO"] is io.StringIO
    assert pre["BytesIO"] is io.BytesIO


# ═══════════════════════════════════════════════════════════════
#  I. Security: dunder traversal blocked
# ═══════════════════════════════════════════════════════════════

def test_dunder_traversal_blocked_at_compile():
    """Classic sandbox escape via __class__.__bases__ is blocked at compile time."""
    source = "x = ().__class__.__bases__"
    with pytest.raises(SandboxError, match="__class__"):
        compile_restricted_agent_code(source)


def test_dunder_globals_blocked_at_compile():
    """Access to __globals__ is blocked at compile time."""
    source = textwrap.dedent("""\
    def escape():
        def inner():
            pass
        return inner.__globals__
    """)
    with pytest.raises(SandboxError, match="__globals__"):
        compile_restricted_agent_code(source)


def test_dunder_via_getattr_blocked_at_runtime():
    """Even if code avoids direct dunder access, runtime guard catches it."""
    from scout.runtime.sandbox import _guarded_getattr
    with pytest.raises(AttributeError, match="not allowed"):
        _guarded_getattr((), "__class__")
    with pytest.raises(AttributeError, match="not allowed"):
        _guarded_getattr(object, "__subclasses__")


def test_allowed_dunders_via_builtins():
    """Normal operations work via builtins (len, str, etc.)."""
    source = textwrap.dedent("""\
    def test():
        items = [1, 2, 3]
        length = len(items)
        text = str(42)
        return length, text
    """)
    code = compile_restricted_agent_code(source)
    ns = build_restricted_globals(build_safe_pre_imports())
    exec(code, ns)
    length, text = ns["test"]()
    assert length == 3
    assert text == "42"


def test_module_write_blocked():
    """Cannot modify module attributes."""
    source = textwrap.dedent("""\
    import json
    json.custom_attr = "hacked"
    """)
    code = compile_restricted_agent_code(source)
    ns = build_restricted_globals(build_safe_pre_imports())
    with pytest.raises(AttributeError, match="Cannot modify module"):
        exec(code, ns)
