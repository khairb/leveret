"""Sandbox for AI-generated scraping code.

Validates and compiles agent code through RestrictedPython, then executes
it in a restricted namespace with a curated module whitelist and safe
builtins. Opt-in via ``sandbox=True`` on :class:`~scout.Scraper`.

Security layers:
    1. **AST validation** — RestrictedPython blocks ``exec()``, ``eval()``,
       ``from x import *``, and other dangerous constructs at compile time.
    2. **Import whitelist** — only audited-safe stdlib modules are importable.
    3. **Safe asyncio proxy** — exposes ``sleep``/``gather``/``wait_for`` but
       NOT ``create_subprocess_exec``/``open_connection``/``start_server``.
    4. **Restricted builtins** — no ``open``, ``exec``, ``eval``, ``compile``,
       ``breakpoint``, ``globals``, ``locals``, ``vars``, ``dir``.
"""

from __future__ import annotations

import asyncio
import io
import operator
import types
from datetime import datetime
from typing import Any
from urllib.parse import urljoin, urlparse

from RestrictedPython import compile_restricted_exec, safe_builtins
from RestrictedPython.Guards import (
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
)
from RestrictedPython.transformer import RestrictingNodeTransformer


# ═══════════════════════════════════════════════════════════════
#  Exceptions
# ═══════════════════════════════════════════════════════════════

class SandboxError(Exception):
    """Raised when agent code fails sandbox validation."""


# ═══════════════════════════════════════════════════════════════
#  Custom AST Transformer
# ═══════════════════════════════════════════════════════════════

class ScoutTransformer(RestrictingNodeTransformer):
    """RestrictedPython policy tuned for async browser automation code.

    Allows async constructs, nonlocal, underscore names, and augmented
    assignment on attributes/items — all needed by AI-generated scrapers.
    Keeps blocking exec/eval calls and star imports.
    """

    # ── Allow async constructs (blocked by default) ──

    def visit_AsyncFunctionDef(self, node):
        return self.node_contents_visit(node)

    def visit_Await(self, node):
        return self.node_contents_visit(node)

    def visit_AsyncFor(self, node):
        return self.node_contents_visit(node)

    def visit_AsyncWith(self, node):
        return self.node_contents_visit(node)

    # ── Allow nonlocal (blocked by default) ──

    def visit_Nonlocal(self, node):
        return node

    # ── Allow augmented assignment on attributes/items ──
    #    e.g. obj[key] += val, obj.attr += val

    def visit_AugAssign(self, node):
        return self.node_contents_visit(node)

    # ── Allow match/case (Python 3.10+) ──
    #    RestrictedPython's generic_visit blocks unknown node types.

    def visit_Match(self, node):
        return self.node_contents_visit(node)

    def visit_match_case(self, node):
        return self.node_contents_visit(node)

    def visit_MatchValue(self, node):
        return self.node_contents_visit(node)

    def visit_MatchSingleton(self, node):
        return self.node_contents_visit(node)

    def visit_MatchSequence(self, node):
        return self.node_contents_visit(node)

    def visit_MatchMapping(self, node):
        return self.node_contents_visit(node)

    def visit_MatchClass(self, node):
        return self.node_contents_visit(node)

    def visit_MatchStar(self, node):
        return self.node_contents_visit(node)

    def visit_MatchAs(self, node):
        return self.node_contents_visit(node)

    def visit_MatchOr(self, node):
        return self.node_contents_visit(node)

    # ── Allow underscore-prefixed variable names ──
    #    AI frequently generates _items, _count, _temp, etc.

    def check_name(self, node, name, allow_magic_methods=False):
        # Keep blocking RestrictedPython's reserved names
        if name in ("printed", "builtins"):
            self.error(
                node,
                f"'{name}' is a reserved name in RestrictedPython.",
            )
            return
        # Allow everything else including _prefixed names.
        # Dunder attribute access (__class__, __bases__) goes through
        # _getattr_ guard at runtime, not through check_name.


# ═══════════════════════════════════════════════════════════════
#  Print Passthrough
# ═══════════════════════════════════════════════════════════════

class PassthroughPrintCollector:
    """Drop-in replacement for RestrictedPython's PrintCollector.

    Forwards print() calls to real stdout instead of collecting them
    in memory. This is critical — Scout's subprocess markers and
    progress output rely on print() going to stdout.
    """

    def __init__(self, _getattr_=None):
        pass

    def _call_print(self, *args, **kwargs):
        print(*args, **kwargs)

    def __call__(self):
        return ""


# ═══════════════════════════════════════════════════════════════
#  Module Whitelist (security-audited)
# ═══════════════════════════════════════════════════════════════

ALLOWED_MODULES = frozenset({
    # Pure computation — no I/O
    "json", "re", "math", "itertools", "functools",
    "decimal", "fractions", "statistics",
    # Pure string/text manipulation
    "string", "textwrap", "unicodedata", "difflib",
    # HTML handling — pure parsing
    "html", "html.parser", "html.entities",
    # Date/time
    "datetime", "calendar", "time",
    # URL handling (NOT urllib.request — that does HTTP)
    "urllib.parse",
    # Encoding/hashing — bytes in/out
    "base64", "binascii", "hashlib", "hmac",
    # Data formats — works with file-like objects only
    "csv",
    # Type system — no runtime behavior
    "typing", "enum", "dataclasses",
    # Data structures
    "collections", "operator", "copy",
    # Compression — in-memory only (no file functions)
    "zlib",
    # Misc safe
    "random", "pprint", "contextlib", "abc",
})

# asyncio is handled specially via proxy (see _safe_asyncio below).
# uuid is denied: uuid1()/getnode() spawn subprocesses.
# io is denied: io.open()/io.FileIO() do file I/O.
# gzip/bz2/lzma denied: .open() does file I/O.


# ═══════════════════════════════════════════════════════════════
#  Safe asyncio Proxy
# ═══════════════════════════════════════════════════════════════

_safe_asyncio = types.SimpleNamespace(
    # Timing and scheduling
    sleep=asyncio.sleep,
    wait_for=asyncio.wait_for,
    gather=asyncio.gather,
    shield=asyncio.shield,
    wait=asyncio.wait,
    as_completed=asyncio.as_completed,
    create_task=asyncio.create_task,
    ensure_future=asyncio.ensure_future,
    # Exceptions
    TimeoutError=asyncio.TimeoutError,
    CancelledError=asyncio.CancelledError,
    # Synchronization primitives
    Queue=asyncio.Queue,
    Event=asyncio.Event,
    Lock=asyncio.Lock,
    Semaphore=asyncio.Semaphore,
    # Introspection
    iscoroutine=asyncio.iscoroutine,
    iscoroutinefunction=asyncio.iscoroutinefunction,
    # NOT included: create_subprocess_exec, create_subprocess_shell,
    #   open_connection, start_server, open_unix_connection,
    #   start_unix_server
)

_PROXY_MODULES: dict[str, Any] = {"asyncio": _safe_asyncio}

_real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__  # type: ignore[union-attr]


def _safe_import(
    name: str,
    globals: dict | None = None,
    locals: dict | None = None,
    fromlist: tuple = (),
    level: int = 0,
) -> Any:
    """Import function that enforces the module whitelist."""
    if level != 0:
        raise ImportError("Relative imports are not allowed in sandbox mode")

    top = name.split(".")[0]

    # Return proxy for modules with dangerous subsets
    if top in _PROXY_MODULES:
        return _PROXY_MODULES[top]

    # Check if exact module or any parent is in the whitelist
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        if ".".join(parts[:i]) in ALLOWED_MODULES:
            return _real_import(name, globals, locals, fromlist, level)

    raise ImportError(
        f"Module '{name}' is not available in sandbox mode. "
        f"Allowed: json, re, math, collections, itertools, "
        f"functools, datetime, html, base64, csv, string, copy, "
        f"decimal, random, zlib, and others. "
        f"See docs for the full list."
    )


# ═══════════════════════════════════════════════════════════════
#  Restricted Builtins
# ═══════════════════════════════════════════════════════════════

_EXTRA_BUILTINS: dict[str, Any] = {
    # Data structures
    "dict": dict, "list": list, "set": set, "frozenset": frozenset,
    # Iteration
    "enumerate": enumerate, "filter": filter, "map": map,
    "iter": iter, "next": next, "reversed": reversed,
    # Aggregation
    "max": max, "min": min, "sum": sum, "all": all, "any": any,
    # Introspection
    "hasattr": hasattr, "getattr": getattr,
    "type": type, "isinstance": isinstance, "issubclass": issubclass,
    # OOP
    "super": super, "object": object,
    "property": property, "staticmethod": staticmethod,
    "classmethod": classmethod,
    # Output and formatting
    "print": print, "sorted": sorted, "zip": zip,
    "bin": bin, "ascii": ascii,
    # Import (our whitelist version)
    "__import__": _safe_import,
}


def build_restricted_builtins() -> dict[str, Any]:
    """Build the restricted __builtins__ dict.

    Starts from RestrictedPython's safe_builtins (includes all exception
    types, basic type constructors, and safe functions) then adds commonly
    needed builtins while keeping exec/eval/open/etc. blocked.
    """
    builtins = dict(safe_builtins)
    builtins.update(_EXTRA_BUILTINS)
    return builtins


# ═══════════════════════════════════════════════════════════════
#  In-place Variable Operations
# ═══════════════════════════════════════════════════════════════

_INPLACE_OPS = {
    "+=": operator.iadd,
    "-=": operator.isub,
    "*=": operator.imul,
    "/=": operator.itruediv,
    "//=": operator.ifloordiv,
    "%=": operator.imod,
    "**=": operator.ipow,
    "&=": operator.iand,
    "|=": operator.ior,
    "^=": operator.ixor,
    ">>=": operator.irshift,
    "<<=": operator.ilshift,
}


def _inplacevar(op: str, x: Any, y: Any) -> Any:
    """Handle augmented assignment operations (x += y, etc.)."""
    return _INPLACE_OPS[op](x, y)


# ═══════════════════════════════════════════════════════════════
#  Restricted Globals (execution namespace)
# ═══════════════════════════════════════════════════════════════

def build_safe_pre_imports() -> dict[str, Any]:
    """Build the safe pre-import namespace.

    Excludes os, shutil, tempfile. Provides asyncio as a safe proxy.
    Provides StringIO/BytesIO directly (io module is denied).
    """
    import math

    return {
        "json": __import__("json"),
        "re": __import__("re"),
        "math": math,
        "time": __import__("time"),
        "asyncio": _safe_asyncio,
        "datetime": datetime,
        "urljoin": urljoin,
        "urlparse": urlparse,
        "StringIO": io.StringIO,
        "BytesIO": io.BytesIO,
    }


# Dunder attributes allowed for normal Python operations.
# These are needed for iteration, context managers, containers, etc.
_ALLOWED_DUNDERS = frozenset({
    "__init__", "__len__", "__str__", "__repr__", "__bool__",
    "__iter__", "__next__", "__aiter__", "__anext__",
    "__getitem__", "__setitem__", "__delitem__", "__contains__",
    "__enter__", "__exit__", "__aenter__", "__aexit__",
    "__call__", "__hash__", "__eq__", "__ne__",
    "__lt__", "__le__", "__gt__", "__ge__",
    "__add__", "__radd__", "__sub__", "__mul__", "__truediv__",
    "__floordiv__", "__mod__", "__pow__",
    "__and__", "__or__", "__xor__", "__invert__",
    "__neg__", "__pos__", "__abs__",
    "__int__", "__float__", "__complex__",
    "__index__", "__format__",
    "__name__", "__doc__", "__module__", "__qualname__",
    "__wrapped__", "__slots__",
})


def _guarded_getattr(obj: Any, name: str) -> Any:
    """Attribute access guard that blocks dunder traversal attacks.

    Allows normal attribute access (page.goto, item.price, etc.) and
    common dunder methods (__len__, __iter__, __str__, etc.) but blocks
    introspection dunders used in sandbox escapes (__class__, __bases__,
    __subclasses__, __globals__, __code__, __builtins__, etc.).
    """
    if name.startswith("__") and name.endswith("__"):
        if name not in _ALLOWED_DUNDERS:
            raise AttributeError(
                f"Access to '{name}' is not allowed in sandbox mode"
            )
    return getattr(obj, name)


def _guarded_write(obj: Any) -> Any:
    """Write guard that blocks attribute writes to modules.

    Allows writes to normal objects (dicts, lists, user objects) but
    prevents patching module attributes.
    """
    import types
    if isinstance(obj, types.ModuleType):
        raise AttributeError(
            "Cannot modify module attributes in sandbox mode"
        )
    return obj


def build_restricted_globals(
    pre_imports: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete restricted globals dict for sandboxed execution.

    Combines restricted builtins, permissive guards (for browser automation
    compatibility), and pre-imported safe modules.
    """
    if pre_imports is None:
        pre_imports = build_safe_pre_imports()

    return {
        "__builtins__": build_restricted_builtins(),
        "__name__": "__scout_sandbox__",
        "__metaclass__": type,
        # Guards — permissive for normal access, blocks dunder traversal
        "_getattr_": _guarded_getattr,
        "_getitem_": lambda obj, key: obj[key],
        "_getiter_": iter,
        "_write_": _guarded_write,
        "_apply_": lambda f, *a, **kw: f(*a, **kw),
        "_inplacevar_": _inplacevar,
        "_print_": PassthroughPrintCollector,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "_unpack_sequence_": guarded_unpack_sequence,
        # Pre-imported safe modules
        **pre_imports,
    }


# ═══════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════

def compile_restricted_agent_code(
    source: str,
    filename: str = "<agent>",
) -> Any:
    """Compile agent code through RestrictedPython with Scout's policy.

    Args:
        source: The agent-authored Python source code
            (typically ``async def scrape(page, start_url, checkpoint)``).
        filename: Filename for error messages.

    Returns:
        Compiled code object ready for ``exec()``.

    Raises:
        SandboxError: If the code fails validation (e.g. contains
            ``exec()``, ``eval()``, or ``from x import *``).
    """
    result = compile_restricted_exec(
        source,
        filename=filename,
        policy=ScoutTransformer,
    )

    if result.errors:
        errors_str = "\n".join(f"  - {e}" for e in result.errors)
        raise SandboxError(
            f"Agent code failed sandbox validation:\n{errors_str}"
        )

    return result.code
