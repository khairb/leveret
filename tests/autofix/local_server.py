"""Local HTTP test server for autofix error harvesting.

Pure-asyncio HTTP server (no external dependencies) that serves crafted
HTML pages, anti-bot simulations, error responses, and edge-case endpoints.
Used by the error harvesting harness and by pytest fixtures for integration
tests.

Endpoints
---------
- ``GET /page/<name>``     — crafted HTML pages (overlay, hidden, dialog, etc.)
- ``GET /slow``            — never responds (navigation timeout)
- ``GET /redirect-loop``   — infinite redirect loop
- ``GET /empty``           — accepts connection, closes immediately
- ``GET /503``             — returns HTTP 503
- ``GET /429``             — returns HTTP 429
- ``GET /cloudflare``      — Cloudflare challenge simulation
- ``GET /akamai``          — Akamai block simulation
- ``GET /datadome``        — DataDome captcha simulation
- ``GET /generic-block``   — generic "Access Denied" page (< 10 KB)
- ``GET /empty-shell``     — empty SPA shell (``<div id="root">``)
- ``GET /redirect-away``   — 302 redirect to a different domain
- ``GET /normal``          — clean, normal HTML page (positive control)
"""

from __future__ import annotations

import asyncio
from urllib.parse import unquote

# ──────────────────────────────────────────────────────────────
#  Crafted HTML pages (served at /page/<name>)
# ──────────────────────────────────────────────────────────────

HTML_PAGES: dict[str, str] = {
    "overlay.html": (
        "<html><body>"
        '<div style="position:fixed;top:0;left:0;width:100%;height:100%;'
        'z-index:9999;background:rgba(0,0,0,0.5)"></div>'
        '<button id="target">Click me</button>'
        "</body></html>"
    ),
    "hidden.html": (
        '<html><body><div style="display:none" id="hidden">Hidden content</div></body></html>'
    ),
    "strict.html": (
        "<html><body>"
        '<div class="item">1</div>'
        '<div class="item">2</div>'
        '<div class="item">3</div>'
        "</body></html>"
    ),
    "navigate_away.html": (
        "<html><body>"
        "<script>setTimeout(() => location.href = '/nonexistent', 100)</script>"
        "<p>Will navigate away</p>"
        "</body></html>"
    ),
    "dialog.html": ("<html><body><script>alert('blocking dialog')</script></body></html>"),
    "detach_frame.html": (
        "<html><body>"
        '<iframe id="f" src="about:blank"></iframe>'
        "<script>"
        "setTimeout(() => document.getElementById('f').remove(), 500)"
        "</script>"
        "</body></html>"
    ),
    "spa_shell.html": ('<html><body><div id="root"></div></body></html>'),
    "delayed_dialog.html": (
        "<html><body>"
        "<p>Content loaded</p>"
        "<script>"
        "setTimeout(() => alert('blocking dialog'), 500);"
        "</script>"
        "</body></html>"
    ),
    "normal.html": (
        "<html><head><title>Test Page</title></head><body>"
        "<h1>Hello World</h1>"
        "<p>This is a normal page with real content.</p>"
        '<div class="products">'
        '<div class="product"><span class="name">Widget</span>'
        '<span class="price">9.99</span></div>'
        '<div class="product"><span class="name">Gadget</span>'
        '<span class="price">19.99</span></div>'
        "</div>"
        "</body></html>"
    ),
}

# ──────────────────────────────────────────────────────────────
#  Anti-bot simulation HTML
# ──────────────────────────────────────────────────────────────

_CLOUDFLARE_HTML = """\
<html>
<head><title>Just a moment...</title></head>
<body>
<div class="main-wrapper">
    <div class="challenge-form">
        <input type="hidden" name="__cf_chl_f_tk" value="abc123">
    </div>
</div>
<script src="/cdn-cgi/challenge-platform/scripts/jsd/main.js"></script>
</body>
</html>"""

_AKAMAI_HTML = """\
<html>
<head><title>Access Denied</title></head>
<body>
<h1>Pardon Our Interruption</h1>
<p>Reference # 18.abc123.1234567890.abcdef</p>
</body>
</html>"""

_DATADOME_HTML = """\
<html>
<head><title>Blocked</title></head>
<body>
<iframe src="https://geo.captcha-delivery.com/captcha/check"></iframe>
</body>
</html>"""

_GENERIC_BLOCK_HTML = """\
<html>
<head><title>Blocked</title></head>
<body>
<h1>Access Denied</h1>
<p>Your request has been blocked by security.</p>
</body>
</html>"""

_EMPTY_SHELL_HTML = '<html><body><div id="root"></div><script src="/app.js"></script></body></html>'

_NORMAL_HTML = """\
<html>
<head><title>Test Page</title></head>
<body>
<h1>Hello World</h1>
<p>This is a normal page with real content for testing.</p>
</body>
</html>"""

# ──────────────────────────────────────────────────────────────
#  HTTP response helpers
# ──────────────────────────────────────────────────────────────


def _http_response(
    status: int,
    body: str,
    *,
    content_type: str = "text/html; charset=utf-8",
    extra_headers: dict[str, str] | None = None,
) -> bytes:
    """Build a minimal HTTP/1.1 response."""
    body_bytes = body.encode()
    headers = {
        "Content-Type": content_type,
        "Content-Length": str(len(body_bytes)),
        "Connection": "close",
    }
    if extra_headers:
        headers.update(extra_headers)

    status_text = {
        200: "OK",
        302: "Found",
        404: "Not Found",
        429: "Too Many Requests",
        503: "Service Unavailable",
    }.get(status, "Unknown")

    lines = [f"HTTP/1.1 {status} {status_text}"]
    for key, val in headers.items():
        lines.append(f"{key}: {val}")
    lines.append("")
    lines.append("")
    header_bytes = "\r\n".join(lines).encode()
    return header_bytes + body_bytes


def _redirect_response(location: str) -> bytes:
    """Build a 302 redirect response."""
    body = f"<html><body>Redirecting to {location}</body></html>"
    return _http_response(302, body, extra_headers={"Location": location})


# ──────────────────────────────────────────────────────────────
#  Request router
# ──────────────────────────────────────────────────────────────


def _parse_request_path(data: bytes) -> str:
    """Extract the path from the first line of an HTTP request."""
    try:
        first_line = data.split(b"\r\n", 1)[0].decode(errors="replace")
        # e.g. "GET /page/overlay.html HTTP/1.1"
        parts = first_line.split(" ")
        if len(parts) >= 2:
            return unquote(parts[1])
    except Exception:
        pass
    return "/"


async def _route(path: str, writer: asyncio.StreamWriter) -> None:
    """Route a request path to the appropriate handler."""

    # -- Crafted HTML pages --
    if path.startswith("/page/"):
        page_name = path[len("/page/") :]
        html = HTML_PAGES.get(page_name)
        if html is not None:
            writer.write(_http_response(200, html))
        else:
            writer.write(_http_response(404, f"Page '{page_name}' not found"))
        return

    # -- Slow response (never completes — triggers navigation timeout) --
    if path == "/slow":
        # Write nothing; keep connection open until client gives up.
        # The caller will eventually hit a timeout and close the socket.
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        return

    # -- Redirect loop --
    if path == "/redirect-loop":
        writer.write(_redirect_response("/redirect-loop"))
        return

    # -- Empty response (close immediately) --
    if path == "/empty":
        # Close without writing anything — triggers net::ERR_EMPTY_RESPONSE
        return

    # -- Server error --
    if path == "/503":
        writer.write(_http_response(503, "Service Unavailable"))
        return

    # -- Rate limited --
    if path == "/429":
        writer.write(_http_response(429, "Too Many Requests"))
        return

    # -- Cloudflare challenge simulation --
    if path == "/cloudflare":
        writer.write(
            _http_response(
                200,
                _CLOUDFLARE_HTML,
                extra_headers={
                    "cf-mitigated": "challenge",
                    "Set-Cookie": "cf_clearance=abc123; Path=/",
                },
            )
        )
        return

    # -- Akamai block simulation --
    if path == "/akamai":
        writer.write(
            _http_response(
                200,
                _AKAMAI_HTML,
                extra_headers={
                    "Set-Cookie": "_abck=abc123; Path=/",
                },
            )
        )
        return

    # -- DataDome captcha simulation --
    if path == "/datadome":
        writer.write(
            _http_response(
                200,
                _DATADOME_HTML,
                extra_headers={
                    "x-dd-b": "1",
                    "Set-Cookie": "datadome=abc123; Path=/",
                },
            )
        )
        return

    # -- Generic block page --
    if path == "/generic-block":
        writer.write(_http_response(200, _GENERIC_BLOCK_HTML))
        return

    # -- Empty SPA shell --
    if path == "/empty-shell":
        writer.write(_http_response(200, _EMPTY_SHELL_HTML))
        return

    # -- Redirect to different domain --
    if path == "/redirect-away":
        writer.write(_redirect_response("https://login.example.com/sso"))
        return

    # -- Normal page (positive control) --
    if path == "/normal":
        writer.write(_http_response(200, _NORMAL_HTML))
        return

    # -- Fallback --
    writer.write(_http_response(404, "Not found"))


# ──────────────────────────────────────────────────────────────
#  Server lifecycle
# ──────────────────────────────────────────────────────────────


async def _handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    """Handle a single HTTP request."""
    try:
        # Read enough to parse the first line (method + path).
        data = await asyncio.wait_for(reader.read(8192), timeout=5.0)
        if not data:
            return
        path = _parse_request_path(data)
        await _route(path, writer)
        await writer.drain()
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        pass
    except Exception:
        # Silently ignore unexpected errors in test server
        pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


class LocalTestServer:
    """Async context manager that runs a local HTTP server for testing.

    Usage::

        async with LocalTestServer() as server:
            url = server.url("/page/overlay.html")
            # ... run tests against url ...
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self._host = host
        self._port = port
        self._server: asyncio.Server | None = None
        self._actual_port: int | None = None

    @property
    def port(self) -> int:
        """The port the server is listening on (assigned after start)."""
        if self._actual_port is None:
            raise RuntimeError("Server not started")
        return self._actual_port

    @property
    def base_url(self) -> str:
        """Base URL including scheme, host, and port."""
        return f"http://{self._host}:{self.port}"

    def url(self, path: str) -> str:
        """Build a full URL for the given path."""
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{self.base_url}{path}"

    async def start(self) -> None:
        """Start the server. Use port=0 for an OS-assigned free port."""
        self._server = await asyncio.start_server(
            _handle_client,
            self._host,
            self._port,
        )
        # Retrieve the actual port (useful when port=0).
        sockets = self._server.sockets
        if sockets:
            self._actual_port = sockets[0].getsockname()[1]
        else:
            raise RuntimeError("Server started but no sockets bound")

    async def stop(self) -> None:
        """Stop the server and wait for cleanup."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            self._actual_port = None

    async def __aenter__(self) -> LocalTestServer:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()
