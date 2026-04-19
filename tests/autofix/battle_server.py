"""Programmable HTTP server for battle-testing the auto-fix algorithm.

Serves configurable response sequences per URL path. Each request to a
path advances through its programmed responses; the last response repeats
forever. This allows tests to simulate real-world scenarios where a server
changes behavior between requests (site deploys, intermittent errors,
adaptive anti-bot).

Usage::

    async with BattleServer() as server:
        server.program("/products",
            PRODUCTS_PAGE,       # Request 1: normal page
            SERVER_503,          # Request 2: server error
            PRODUCTS_PAGE,       # Request 3+: normal again
        )
        url = server.url("/products")
        # ... run diagnosis against url ...
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from urllib.parse import unquote


# ── Response data ────────────────────────────────────────────


@dataclass
class Response:
    """A single HTTP response to serve."""

    status: int = 200
    body: str = ""
    headers: dict[str, str] = field(default_factory=dict)


# ── Pre-built response templates ─────────────────────────────
#
# Each template represents a real-world page scenario. The HTML
# is realistic enough for anti-bot detection, page verification,
# and content checks to produce correct results.


PRODUCTS_PAGE = Response(
    body=(
        "<html><head><title>Products - Example Store</title></head>\n"
        "<body>\n"
        "<h1>Our Products</h1>\n"
        "<p>Browse our collection of quality products.</p>\n"
        '<div class="products">\n'
        '  <div class="product"><span class="name">Widget A</span>'
        '<span class="price">9.99</span></div>\n'
        '  <div class="product"><span class="name">Widget B</span>'
        '<span class="price">19.99</span></div>\n'
        '  <div class="product"><span class="name">Widget C</span>'
        '<span class="price">29.99</span></div>\n'
        '  <div class="product"><span class="name">Widget D</span>'
        '<span class="price">39.99</span></div>\n'
        '  <div class="product"><span class="name">Widget E</span>'
        '<span class="price">49.99</span></div>\n'
        "</div>\n"
        "</body></html>"
    ),
)

CHANGED_LAYOUT = Response(
    body=(
        "<html><head><title>Products - Example Store</title></head>\n"
        "<body>\n"
        "<h1>Our Products</h1>\n"
        "<p>Browse our redesigned catalog.</p>\n"
        '<div class="catalog">\n'
        '  <div class="item"><span class="title">Widget A</span>'
        '<span class="cost">9.99</span></div>\n'
        '  <div class="item"><span class="title">Widget B</span>'
        '<span class="cost">19.99</span></div>\n'
        '  <div class="item"><span class="title">Widget C</span>'
        '<span class="cost">29.99</span></div>\n'
        '  <div class="item"><span class="title">Widget D</span>'
        '<span class="cost">39.99</span></div>\n'
        '  <div class="item"><span class="title">Widget E</span>'
        '<span class="cost">49.99</span></div>\n'
        "</div>\n"
        "</body></html>"
    ),
)

FEW_ITEMS_PAGE = Response(
    body=(
        "<html><head><title>Products</title></head>\n"
        "<body>\n"
        "<h1>Products</h1>\n"
        "<p>Our products page.</p>\n"
        '<div class="products">\n'
        '  <div class="product"><span class="name">Widget A</span>'
        '<span class="price">9.99</span></div>\n'
        '  <div class="product"><span class="name">Widget B</span>'
        '<span class="price">19.99</span></div>\n'
        "</div>\n"
        "</body></html>"
    ),
)

OVERLAY_PAGE = Response(
    body=(
        "<html><head><title>Products</title></head>\n"
        "<body>\n"
        '<div id="cookie-consent" style="position:fixed;top:0;left:0;'
        "width:100%;height:100%;z-index:9999;background:rgba(0,0,0,0.8)"
        '">\n'
        "  <p>We use cookies. Accept?</p>\n"
        "  <button>Accept All</button>\n"
        "</div>\n"
        '<div class="products">\n'
        '  <div class="product"><span class="name">Widget A</span>'
        '<span class="price">9.99</span></div>\n'
        "</div>\n"
        "</body></html>"
    ),
)

CLOUDFLARE_CHALLENGE = Response(
    body=(
        "<html>\n"
        "<head><title>Just a moment...</title></head>\n"
        "<body>\n"
        '<div class="main-wrapper">\n'
        '    <div class="challenge-form">\n'
        '        <input type="hidden" name="__cf_chl_f_tk" '
        'value="abc123">\n'
        "    </div>\n"
        "</div>\n"
        '<script src="/cdn-cgi/challenge-platform/scripts/jsd/'
        'main.js"></script>\n'
        "</body>\n"
        "</html>"
    ),
    headers={
        "cf-mitigated": "challenge",
        "Set-Cookie": "cf_clearance=abc123; Path=/",
    },
)

SERVER_503 = Response(
    status=503,
    body="<html><body><h1>Service Unavailable</h1></body></html>",
)

SERVER_429 = Response(
    status=429,
    body="<html><body><h1>Too Many Requests</h1></body></html>",
)

SERVER_ERROR_AS_200 = Response(
    body=(
        "<html><head><title>Error</title></head>\n"
        "<body>\n"
        "<h1>Oops! Something went wrong</h1>\n"
        "<p>We're sorry, but something unexpected happened on our end. "
        "Our team has been notified and is looking into it.</p>\n"
        "<p>Please try again later. If the problem persists, contact "
        "our support team at support@example.com.</p>\n"
        '<div class="error-details">\n'
        "  <p>Error reference: ERR-2024-0819-4521</p>\n"
        "  <p>Timestamp: 2024-08-19T14:32:01Z</p>\n"
        "</div>\n"
        "</body></html>"
    ),
)

LOGIN_WALL = Response(
    body=(
        "<html><head><title>Sign In - Example Store</title></head>\n"
        "<body>\n"
        "<h1>Sign in to continue</h1>\n"
        "<p>You need to be signed in to view this page.</p>\n"
        '<form action="/login" method="POST">\n'
        '  <input type="email" name="email" placeholder="Email">\n'
        '  <input type="password" name="password" placeholder="Password">\n'
        '  <button type="submit">Sign In</button>\n'
        "</form>\n"
        '<p>Don\'t have an account? <a href="/signup">Sign up</a></p>\n'
        "</body></html>"
    ),
)

PARTIAL_PRODUCTS = Response(
    body=(
        "<html><head><title>Products - Example Store</title></head>\n"
        "<body>\n"
        "<h1>Our Products</h1>\n"
        "<p>Browse our collection of quality products.</p>\n"
        '<div class="products">\n'
        '  <div class="product"><span class="name">Widget A</span>'
        '<span class="price">9.99</span></div>\n'
        "</div>\n"
        '<div class="loading-spinner">Loading more products...</div>\n'
        "</body></html>"
    ),
)

RATE_LIMIT_SOFT = Response(
    body=(
        "<html><head><title>Slow Down</title></head>\n"
        "<body>\n"
        "<h1>You're browsing too fast</h1>\n"
        "<p>Please wait a moment before continuing. We limit the number "
        "of requests to protect our servers.</p>\n"
        "<p>This page will automatically refresh in 30 seconds.</p>\n"
        '<meta http-equiv="refresh" content="30">\n'
        "</body></html>"
    ),
)

AB_TEST_VARIANT = Response(
    body=(
        "<html><head><title>Products - Example Store</title></head>\n"
        "<body>\n"
        "<h1>Our Products</h1>\n"
        "<p>Check out our new design!</p>\n"
        '<div class="product-grid">\n'
        '  <article class="product-card">'
        '<h2 class="product-title">Widget A</h2>'
        '<span class="product-price">$9.99</span></article>\n'
        '  <article class="product-card">'
        '<h2 class="product-title">Widget B</h2>'
        '<span class="product-price">$19.99</span></article>\n'
        '  <article class="product-card">'
        '<h2 class="product-title">Widget C</h2>'
        '<span class="product-price">$29.99</span></article>\n'
        "</div>\n"
        "</body></html>"
    ),
)

DATADOME_CHALLENGE = Response(
    body=(
        "<html>\n"
        "<head><title>Blocked</title></head>\n"
        "<body>\n"
        '<iframe src="https://geo.captcha-delivery.com/captcha/'
        'check"></iframe>\n'
        "</body>\n"
        "</html>"
    ),
    headers={
        "x-dd-b": "1",
        "Set-Cookie": "datadome=abc123; Path=/",
    },
)


# ── HTTP response builder ────────────────────────────────────


def _build_http_response(resp: Response) -> bytes:
    """Build raw HTTP/1.1 response bytes from a Response."""
    body_bytes = resp.body.encode("utf-8")
    status_text = {
        200: "OK",
        302: "Found",
        404: "Not Found",
        429: "Too Many Requests",
        503: "Service Unavailable",
    }.get(resp.status, "Unknown")

    headers = {
        "Content-Type": "text/html; charset=utf-8",
        "Content-Length": str(len(body_bytes)),
        "Connection": "close",
    }
    headers.update(resp.headers)

    lines = [f"HTTP/1.1 {resp.status} {status_text}"]
    for key, val in headers.items():
        lines.append(f"{key}: {val}")
    lines.append("")
    lines.append("")
    return "\r\n".join(lines).encode() + body_bytes


# ── Server implementation ────────────────────────────────────


class BattleServer:
    """HTTP server with per-request programmable responses.

    Each path can be programmed with a sequence of responses.
    The Nth request to a path returns the Nth response; extra
    requests repeat the last response forever.

    Usage::

        async with BattleServer() as server:
            server.program("/p", PRODUCTS_PAGE, CHANGED_LAYOUT)
            url = server.url("/p")
            # Request 1 → PRODUCTS_PAGE
            # Request 2+ → CHANGED_LAYOUT
    """

    def __init__(
        self, host: str = "127.0.0.1", port: int = 0,
    ) -> None:
        self._host = host
        self._port = port
        self._sequences: dict[str, list[Response]] = {}
        self._counters: dict[str, int] = defaultdict(int)
        self._server: asyncio.Server | None = None
        self._actual_port: int | None = None

    def program(self, path: str, *responses: Response) -> None:
        """Set response sequence for a path. Last response repeats.

        Args:
            path: URL path (e.g., ``"/products"``).
            responses: One or more ``Response`` objects to serve in order.
                The last response repeats for all subsequent requests.
        """
        if not path.startswith("/"):
            path = "/" + path
        self._sequences[path] = list(responses)
        self._counters[path] = 0

    def reset(self) -> None:
        """Reset all request counters (keeps programmed sequences)."""
        for key in self._counters:
            self._counters[key] = 0

    @property
    def port(self) -> int:
        """The port the server is listening on."""
        if self._actual_port is None:
            raise RuntimeError("Server not started")
        return self._actual_port

    def url(self, path: str) -> str:
        """Build a full URL for the given path."""
        if not path.startswith("/"):
            path = "/" + path
        return f"http://{self._host}:{self.port}{path}"

    async def start(self) -> None:
        """Start the server."""
        self._server = await asyncio.start_server(
            self._handle_client, self._host, self._port,
        )
        sockets = self._server.sockets
        if sockets:
            self._actual_port = sockets[0].getsockname()[1]
        else:
            raise RuntimeError("Server started but no sockets bound")

    async def stop(self) -> None:
        """Stop the server."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            self._actual_port = None

    async def __aenter__(self) -> BattleServer:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    # -- Internal --

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single HTTP request."""
        try:
            data = await asyncio.wait_for(reader.read(8192), timeout=5.0)
            if not data:
                return
            path = self._parse_path(data)
            resp = self._get_response(path)
            writer.write(_build_http_response(resp))
            await writer.drain()
        except (
            asyncio.CancelledError,
            ConnectionResetError,
            BrokenPipeError,
        ):
            pass
        except Exception:
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    def _get_response(self, path: str) -> Response:
        """Get the next response for a path, advancing the counter."""
        seq = self._sequences.get(path)
        if not seq:
            return Response(status=404, body="Not programmed")
        idx = self._counters[path]
        self._counters[path] += 1
        if idx >= len(seq):
            return seq[-1]  # Last response repeats
        return seq[idx]

    @staticmethod
    def _parse_path(data: bytes) -> str:
        """Extract the path from an HTTP request line."""
        try:
            first_line = data.split(b"\r\n", 1)[0].decode(errors="replace")
            parts = first_line.split(" ")
            if len(parts) >= 2:
                return unquote(parts[1])
        except Exception:
            pass
        return "/"
