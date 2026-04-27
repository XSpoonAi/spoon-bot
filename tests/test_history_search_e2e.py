"""End-to-end test for the conversation-history-search feature.

This script exercises the full stack *without* needing a real LLM:

1.  Boots the gateway FastAPI app in-process with a stub agent whose
    ``sessions`` attribute is a real :class:`SessionManager`.
2.  Pre-populates two sessions with:
      - user / assistant messages
      - ``role="tool"`` messages carrying ``tool_call_id`` + serialised
        tool output (the exact shape ``_persist_turn_tool_trace``
        produces during a real agent turn).
3.  Hits every public search surface and asserts the results match:
      - ``SessionManager.search_messages`` (store layer)
      - ``SearchHistoryTool.execute`` (builtin tool layer)
      - ``GET /v1/sessions/search`` and ``GET /v1/sessions/{key}/search``
        (REST layer)
      - WebSocket ``session.search`` method via
        :func:`TestClient.websocket_connect` (WS layer)

Usage::

    # Runs under pytest:
    pytest tests/test_history_search_e2e.py -v

    # …or as a standalone script:
    python tests/test_history_search_e2e.py

Exits non-zero on the first assertion failure so the user's terminal
shows the actual breakage.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Make sure we import *this* worktree, not any other spoon_bot on PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# Disable auth so TestClient requests pass without tokens.
os.environ.setdefault("GATEWAY_AUTH_REQUIRED", "false")
os.environ.setdefault("SPOON_BOT_LOG_LEVEL", "WARNING")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from spoon_bot.session.manager import SessionManager
from spoon_bot.session.store import FileSessionStore, SQLiteSessionStore, SearchHit
from spoon_bot.agent.tools.history_search import SearchHistoryTool
from spoon_bot.gateway.api.v1 import sessions as sessions_router_module
from spoon_bot.gateway.websocket.handler import websocket_endpoint
from spoon_bot.gateway import app as app_module


# ---------------------------------------------------------------------------
# Test harness (zero-dependency on pytest for stand-alone runs)
# ---------------------------------------------------------------------------


@dataclass
class _Result:
    passed: int = 0
    failed: int = 0
    failures: list[str] = field(default_factory=list)


_RESULT = _Result()


def _record(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        _RESULT.passed += 1
        print(f"  PASS  {name}")
    else:
        _RESULT.failed += 1
        _RESULT.failures.append(f"{name}: {detail}")
        print(f"  FAIL  {name} :: {detail}")


def _assert(name: str, ok: bool, detail: str = "") -> None:
    _record(name, bool(ok), detail)
    if not ok:
        # Make pytest surface the failure too.
        raise AssertionError(f"{name}: {detail}")


# ---------------------------------------------------------------------------
# Stub agent — only needs ``.sessions`` and ``._session``.
# ---------------------------------------------------------------------------


class _StubAgent:
    def __init__(self, sessions: SessionManager, active_session_key: str) -> None:
        self.sessions = sessions
        self.session_key = active_session_key
        self._session = sessions.get_or_create(active_session_key)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_manager(tmpdir: Path, *, backend: str) -> SessionManager:
    if backend == "file":
        store = FileSessionStore(tmpdir / "sessions_file")
    elif backend == "sqlite":
        store = SQLiteSessionStore(tmpdir / "sessions.db")
    else:
        raise ValueError(backend)
    return SessionManager(store=store)


def _seed_session(mgr: SessionManager, key: str, *, variant: str) -> None:
    """Seed a session with realistic tool-call traffic.

    ``variant=alpha`` – a research turn with a web_search call.
    ``variant=beta``  – a coding turn with a read_file call.
    """
    s = mgr.get_or_create(key)

    if variant == "alpha":
        s.add_message("user", "look up the current price of ETH on mainnet")
        # assistant issues a tool call (mirrors _capture_turn_tool_trace output)
        s.add_message(
            "assistant",
            "I'll look that up.",
            tool_calls=[
                {
                    "id": "call_eth_price_001",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": json.dumps({"query": "ETH price USD"}),
                    },
                }
            ],
        )
        s.add_message(
            "tool",
            "ETH/USD: 3421.55 (source: coingecko, ts=2025-11-04T10:22Z)",
            tool_call_id="call_eth_price_001",
            name="web_search",
        )
        s.add_message("assistant", "Ethereum is currently trading around $3,421.55.")
    elif variant == "beta":
        s.add_message("user", "show me the top of README.md please")
        s.add_message(
            "assistant",
            "Reading the file now.",
            tool_calls=[
                {
                    "id": "call_read_readme_42",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "README.md", "limit": 3}),
                    },
                }
            ],
        )
        s.add_message(
            "tool",
            "# spoon-bot\n\nAn agent runtime with pluggable channels and skills.",
            tool_call_id="call_read_readme_42",
            name="read_file",
        )
        s.add_message("assistant", "Here are the first lines of README.md: …")
    else:
        raise ValueError(variant)

    mgr.save(s)


# ---------------------------------------------------------------------------
# Phase 1 — SessionManager.search_messages (store-layer)
# ---------------------------------------------------------------------------


def phase_store_layer(mgr: SessionManager) -> None:
    print("\n[phase] store-layer SessionManager.search_messages")

    # Substring match across both sessions
    hits = mgr.search_messages("ETH", limit=10)
    _assert(
        "store.substring.any-session",
        any(h.session_key == "s-alpha" and h.role in ("assistant", "tool") for h in hits),
        f"no s-alpha hit in {[h.session_key + ':' + h.role for h in hits]}",
    )

    # tool_call_id lives in extras — include_extras=True must find it
    hits = mgr.search_messages("call_read_readme_42", include_extras=True, limit=10)
    _assert(
        "store.extras.tool_call_id",
        any(h.matched_in == "extras" and h.session_key == "s-beta" for h in hits),
        f"no extras-hit for tool_call_id in {hits!r}",
    )

    # include_extras=False must NOT match tool_call_id
    hits = mgr.search_messages("call_read_readme_42", include_extras=False, limit=10)
    _assert("store.extras.disabled", len(hits) == 0, f"expected 0 hits, got {len(hits)}")

    # Role filter
    hits = mgr.search_messages(
        "coingecko", roles=["tool"], limit=10, include_extras=False
    )
    _assert(
        "store.roles.tool-only",
        len(hits) == 1 and hits[0].role == "tool",
        f"expected 1 tool hit, got {[h.role for h in hits]}",
    )

    # Session scoping
    hits = mgr.search_messages("README", session_key="s-alpha", limit=10)
    _assert("store.scope.single-session", len(hits) == 0, f"s-alpha should not mention README, got {hits!r}")

    hits = mgr.search_messages("README", session_key="s-beta", limit=10)
    _assert("store.scope.correct-session", len(hits) >= 1, f"expected s-beta hit, got {hits!r}")

    # Regex
    hits = mgr.search_messages(r"\$\d+,\d{3}", regex=True, limit=10)
    _assert("store.regex", len(hits) >= 1, "regex search for $X,XXX should hit the eth reply")

    # Case sensitivity: "ETH" is in content; "eth" (lowercase) also present
    hits_ci = mgr.search_messages("eth", case_sensitive=False, limit=20)
    hits_cs = mgr.search_messages("ETH", case_sensitive=True, limit=20)
    _assert(
        "store.case.insensitive-superset",
        len(hits_ci) >= len(hits_cs),
        f"CI should find >= CS; got CI={len(hits_ci)} CS={len(hits_cs)}",
    )

    # Snippet sanity
    hits = mgr.search_messages("ETH/USD", limit=1)
    _assert(
        "store.snippet.non-empty",
        len(hits) == 1 and "ETH/USD" in hits[0].snippet,
        f"snippet was {hits[0].snippet!r}" if hits else "no hits",
    )

    # Dirty-flush semantics: write a message to cache then search without explicit save
    live = mgr.get_or_create("s-alpha")
    live.add_message("user", "UNIQUE_TOKEN_FLUSH_TEST_9271")
    # Do NOT call mgr.save(live) — rely on search_messages to flush.
    hits = mgr.search_messages("UNIQUE_TOKEN_FLUSH_TEST_9271", limit=5)
    _assert(
        "store.dirty-flush",
        any("UNIQUE_TOKEN_FLUSH_TEST_9271" in h.content for h in hits),
        "dirty cached message was not flushed before search",
    )


# ---------------------------------------------------------------------------
# Phase 2 — SearchHistoryTool (builtin tool used by the LLM)
# ---------------------------------------------------------------------------


def phase_builtin_tool(mgr: SessionManager) -> None:
    print("\n[phase] builtin SearchHistoryTool.execute()")

    tool = SearchHistoryTool(mgr, default_session_key="s-alpha")

    # scope=current should only search the default session
    out = asyncio.run(tool.execute(query="README"))
    payload = json.loads(out)
    _assert(
        "tool.scope.current-no-match",
        payload["total"] == 0,
        f"scope=current (s-alpha) should not find README; got {payload['total']} hits",
    )

    # scope=all finds README hit
    out = asyncio.run(tool.execute(query="README", scope="all"))
    payload = json.loads(out)
    _assert(
        "tool.scope.all",
        payload["total"] >= 1
        and any(h["session_key"] == "s-beta" for h in payload["hits"]),
        f"expected scope=all to return a s-beta README hit; got {payload}",
    )

    # tool_call_id propagates into hit.tool_call_id when matched_in=extras
    out = asyncio.run(
        tool.execute(query="call_eth_price_001", scope="all", include_extras=True)
    )
    payload = json.loads(out)
    matching = [h for h in payload["hits"] if h["tool_call_id"] == "call_eth_price_001"]
    _assert(
        "tool.tool_call_id.surfaced",
        len(matching) >= 1,
        f"tool_call_id not surfaced in hit; got {payload}",
    )

    # Assistant tool-call traces remain searchable by default because they
    # encode earlier tool arguments / call-site evidence.
    out = asyncio.run(tool.execute(query="ETH price USD", include_extras=True))
    payload = json.loads(out)
    _assert(
        "tool.default-keeps.assistant-tool-trace",
        payload["total"] >= 1
        and any(h["role"] == "assistant" and h["tool_calls"] for h in payload["hits"]),
        f"expected assistant tool-call trace, got {payload}",
    )

    # Plain assistant narrative replies are omitted by default to avoid
    # reviving stale plans after compaction.
    out = asyncio.run(tool.execute(query="currently trading around"))
    payload = json.loads(out)
    _assert(
        "tool.default-omits.assistant-reply",
        payload["total"] == 0
        and "roles=['assistant']" in payload.get("note", ""),
        f"plain assistant reply should be omitted by default; got {payload}",
    )

    out = asyncio.run(tool.execute(query="currently trading around", roles=["assistant"]))
    payload = json.loads(out)
    _assert(
        "tool.roles.assistant-explicit",
        payload["total"] >= 1
        and all(h["role"] == "assistant" for h in payload["hits"]),
        f"explicit assistant role filter should restore assistant replies; got {payload}",
    )

    # Roles filter, current scope
    out = asyncio.run(tool.execute(query="coingecko", roles=["tool"]))
    payload = json.loads(out)
    _assert(
        "tool.roles.filter",
        payload["total"] == 1 and payload["hits"][0]["role"] == "tool",
        f"expected single tool hit, got {payload}",
    )

    # Bad regex is reported, not crashed
    out = asyncio.run(tool.execute(query="[unclosed", regex=True))
    _assert(
        "tool.regex.error-message",
        out.startswith("Error:") and "regex" in out.lower(),
        f"bad regex should return Error; got {out!r}",
    )

    # Empty query rejected
    out = asyncio.run(tool.execute(query=""))
    _assert(
        "tool.empty-query",
        out.startswith("Error:"),
        f"empty query should error; got {out!r}",
    )


# ---------------------------------------------------------------------------
# Phase 3 — REST + WebSocket via TestClient
# ---------------------------------------------------------------------------


def _build_app(stub_agent: _StubAgent) -> FastAPI:
    """Build a minimal FastAPI app that wires our stub as the global agent."""
    from spoon_bot.gateway.config import GatewayConfig
    from spoon_bot.gateway.websocket.manager import ConnectionManager

    config = GatewayConfig()
    app_module._agent = stub_agent  # type: ignore[attr-defined]
    app_module._config = config  # type: ignore[attr-defined]
    app_module._auth_required = False  # type: ignore[attr-defined]
    app_module._connection_manager = ConnectionManager()  # type: ignore[attr-defined]

    app = FastAPI()
    app.include_router(
        sessions_router_module.router, prefix="/v1/sessions", tags=["sessions"]
    )
    app.add_api_websocket_route("/v1/ws", websocket_endpoint)
    return app


def phase_rest(client: TestClient) -> None:
    print("\n[phase] REST  GET /v1/sessions/search")

    # Cross-session search
    r = client.get("/v1/sessions/search", params={"q": "ETH"})
    _assert("rest.search.all.status", r.status_code == 200, f"status={r.status_code} body={r.text}")
    body = r.json()
    _assert(
        "rest.search.all.hit",
        body["success"] and body["data"]["total"] >= 1,
        f"body={body}",
    )

    # Per-session search: beta mentions README, alpha does not
    r = client.get("/v1/sessions/s-beta/search", params={"q": "README"})
    _assert("rest.search.beta.status", r.status_code == 200, r.text)
    body = r.json()
    _assert(
        "rest.search.beta.hit",
        body["data"]["total"] >= 1
        and all(h["session_key"] == "s-beta" for h in body["data"]["hits"]),
        f"body={body}",
    )

    r = client.get("/v1/sessions/s-alpha/search", params={"q": "README"})
    body = r.json()
    _assert(
        "rest.search.alpha.no-readme",
        r.status_code == 200 and body["data"]["total"] == 0,
        f"body={body}",
    )

    # Per-session search hits tool_call_id in extras
    r = client.get(
        "/v1/sessions/s-beta/search", params={"q": "call_read_readme_42"}
    )
    body = r.json()
    _assert(
        "rest.search.extras",
        r.status_code == 200
        and body["data"]["total"] >= 1
        and any(h["matched_in"] == "extras" for h in body["data"]["hits"]),
        f"body={body}",
    )

    # 404 for unknown session
    r = client.get("/v1/sessions/does-not-exist/search", params={"q": "anything"})
    _assert("rest.search.404", r.status_code == 404, f"status={r.status_code} body={r.text}")

    # Invalid regex returns 400
    r = client.get("/v1/sessions/search", params={"q": "[unclosed", "regex": "true"})
    _assert(
        "rest.search.invalid-regex",
        r.status_code == 400,
        f"status={r.status_code} body={r.text}",
    )

    # Ensure static /search did NOT get swallowed by /{session_key}
    r = client.get("/v1/sessions/search", params={"q": "ETH", "limit": 1})
    body = r.json()
    _assert(
        "rest.search.route-order",
        r.status_code == 200 and isinstance(body["data"]["hits"], list),
        f"/search was swallowed by /{{session_key}}: status={r.status_code}",
    )


def phase_websocket(client: TestClient) -> None:
    print("\n[phase] WS    session.search")

    with client.websocket_connect("/v1/ws") as ws:
        # First frame is the connection.established event — consume it.
        first = ws.receive_json()
        _assert(
            "ws.connection.established",
            first.get("event") == "connection.established",
            f"first frame: {first!r}",
        )

        def _call(params: dict[str, Any], *, req_id: str) -> dict[str, Any]:
            ws.send_json({"id": req_id, "method": "session.search", "params": params})
            while True:
                msg = ws.receive_json()
                if msg.get("id") == req_id:
                    return msg

        # Cross-session search, scope via session_key=None
        resp = _call({"q": "coingecko"}, req_id="req-1")
        _assert(
            "ws.search.basic",
            resp.get("error") is None
            and resp.get("result", {}).get("total", 0) >= 1,
            f"resp={resp}",
        )
        hits = resp["result"]["hits"]
        _assert(
            "ws.search.tool-hit",
            any(h["role"] == "tool" and "coingecko" in h["content"] for h in hits),
            f"hits={hits}",
        )

        # Scoped to a single session
        resp = _call(
            {"q": "README", "session_key": "s-beta"}, req_id="req-2"
        )
        _assert(
            "ws.search.scoped",
            resp.get("error") is None
            and resp["result"]["total"] >= 1
            and all(h["session_key"] == "s-beta" for h in resp["result"]["hits"]),
            f"resp={resp}",
        )

        # Role filter
        resp = _call(
            {"q": "ETH/USD", "roles": ["tool"]}, req_id="req-3"
        )
        _assert(
            "ws.search.roles",
            resp.get("error") is None
            and resp["result"]["total"] == 1
            and resp["result"]["hits"][0]["role"] == "tool",
            f"resp={resp}",
        )

        # Invalid regex → structured error field (not a disconnect)
        resp = _call({"q": "[unclosed", "regex": True}, req_id="req-4")
        result = resp.get("result") or {}
        err_field = result.get("error")
        _assert(
            "ws.search.invalid-regex",
            err_field in {"INVALID_REGEX", "INVALID_QUERY"},
            f"resp={resp}",
        )

        # Empty q → INVALID_QUERY
        resp = _call({"q": ""}, req_id="req-5")
        result = resp.get("result") or {}
        err_field = result.get("error")
        _assert(
            "ws.search.empty-q",
            err_field in {"INVALID_QUERY", "INVALID_REGEX"},
            f"resp={resp}",
        )

        # Extras matching through the WS layer
        resp = _call(
            {"q": "call_eth_price_001", "include_extras": True},
            req_id="req-6",
        )
        result = resp.get("result") or {}
        hits = result.get("hits") or []
        _assert(
            "ws.search.extras",
            result.get("error") is None
            and any(h.get("matched_in") == "extras" for h in hits),
            f"resp={resp}",
        )


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------


def _run_all_for_backend(tmpdir: Path, backend: str) -> None:
    print(f"\n================ backend: {backend} ================")
    mgr = _build_manager(tmpdir, backend=backend)
    _seed_session(mgr, "s-alpha", variant="alpha")
    _seed_session(mgr, "s-beta", variant="beta")

    phase_store_layer(mgr)
    phase_builtin_tool(mgr)

    stub = _StubAgent(mgr, active_session_key="s-alpha")
    app = _build_app(stub)
    with TestClient(app) as client:
        phase_rest(client)
        phase_websocket(client)


def test_history_search_full_stack(tmp_path: Path) -> None:
    """Pytest entrypoint – runs the full matrix."""
    for backend in ("file", "sqlite"):
        sub = tmp_path / backend
        sub.mkdir()
        _run_all_for_backend(sub, backend)

    if _RESULT.failed:
        raise AssertionError(
            f"{_RESULT.failed} check(s) failed: {_RESULT.failures}"
        )


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for backend in ("file", "sqlite"):
            sub = root / backend
            sub.mkdir()
            try:
                _run_all_for_backend(sub, backend)
            except AssertionError as exc:
                print(f"\n[FATAL] backend={backend} :: {exc}")
                break

    print("\n===================  SUMMARY  ===================")
    print(f"  passed: {_RESULT.passed}")
    print(f"  failed: {_RESULT.failed}")
    for f in _RESULT.failures:
        print(f"   - {f}")
    return 0 if _RESULT.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
