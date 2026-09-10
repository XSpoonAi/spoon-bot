from __future__ import annotations

import importlib.util
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "spoon_bot"
    / "skills"
    / "builtin"
    / "service_expose"
    / "scripts"
    / "service_expose.py"
)
SPEC = importlib.util.spec_from_file_location("service_expose_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
_verify_url = MODULE._verify_url


@contextmanager
def _preview_server(routes: dict[str, tuple[int, str]]) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
            status, body = routes.get(self.path, (404, "missing"))
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_verify_url_rejects_missing_critical_asset() -> None:
    routes = {
        "/": (200, '<html><body>Ready<script src="/missing.js"></script></body></html>'),
    }
    with _preview_server(routes) as url:
        result = _verify_url(url, expected_text="Ready", wait_seconds=0)

    assert result is not None
    assert result["ok"] is False
    assert result["assets_checked"] == 1
    assert result["asset_failures"][0]["url"].endswith("/missing.js")


def test_verify_url_checks_scripts_and_stylesheets() -> None:
    routes = {
        "/": (
            200,
            '<html><head><link rel="stylesheet" href="/app.css"></head>'
            '<body>Ready<script src="/app.js"></script></body></html>',
        ),
        "/app.css": (200, "body { color: black; }"),
        "/app.js": (200, "document.body.dataset.ready = 'true';"),
    }
    with _preview_server(routes) as url:
        result = _verify_url(url, expected_text="Ready", wait_seconds=0)

    assert result is not None
    assert result["ok"] is True
    assert result["assets_checked"] == 2
