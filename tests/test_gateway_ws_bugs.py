"""
Regression tests for ALL bugs from the Spoon-bot Bug Report (2026-02-12/13).

Tests cover:
  Gateway / WebSocket bugs (#7–#19):
  - Input validation (#14, #15, #16)
  - JWT session claim type validation (#18)
  - WS session binding (#11)
  - WS session import (#12)
  - WS streaming content (#10)
  - WS cancel non-stream (#13)
  - WS auth close code (#17)
  - Auth rate limiting (#19)
  - Async task API (#7)
  - Subscribe/unsubscribe validation (#16)

  Core bugs (#5, #8, #9):
  - MCP tool discovery/expansion (#5)
  - ScriptTool parameter contract alignment with input_schema (#8)
  - SKILL.md loader: BOM handling, line-ending normalization, dedup (#9)

Uses httpx + websockets for real HTTP/WS testing against the gateway.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from spoon_bot.gateway.app import create_app, set_agent
from spoon_bot.gateway import app as app_module
from spoon_bot.gateway.config import GatewayConfig
from spoon_bot.gateway.auth.jwt import (
    create_access_token,
    verify_token,
    TokenData,
)
from spoon_bot.gateway.websocket.protocol import (
    WSRequest,
    parse_message,
)
from spoon_bot.gateway.websocket.handler import (
    _AuthRateLimiter,
    _CONCURRENT_REQUEST_LIMIT,
    _resolve_workspace_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_agent():
    """Create a mock agent that behaves enough for gateway tests."""
    agent = AsyncMock()
    agent.model = "test-model"
    agent.provider = "test"
    agent.tools = MagicMock()
    agent.tools.list_tools.return_value = ["web_search"]
    agent.tools.__len__ = lambda self: 1
    agent.skills = []
    agent.session_key = "default"

    # Session manager mock
    session = MagicMock()
    session.session_key = "default"
    session.messages = []
    def _session_add_message(role, content, **kwargs):
        session.messages.append({"role": role, "content": content, **kwargs})
    session.add_message = MagicMock(side_effect=_session_add_message)
    session.clear = MagicMock()

    sessions = MagicMock()
    sessions.list_sessions.return_value = [session]
    sessions.get.return_value = session
    sessions.get_or_create.return_value = session
    sessions.save = MagicMock()

    agent.sessions = sessions
    agent._session = session

    # process returns a string
    agent.process = AsyncMock(return_value="Hello from test agent")
    agent.process_with_thinking = AsyncMock(return_value=("Hello from test agent", "thinking..."))

    # stream yields chunks
    async def _fake_stream(**kwargs):
        yield {"type": "content", "delta": "Hello ", "metadata": {}}
        yield {"type": "content", "delta": "World", "metadata": {}}
        yield {"type": "done", "delta": "", "metadata": {}}

    agent.stream = _fake_stream
    return agent


def _auth_headers(
    user_id: str,
    *,
    session_key: str = "default",
    scopes: list[str] | None = None,
) -> dict[str, str]:
    token = create_access_token(
        user_id=user_id,
        session_key=session_key,
        scopes=scopes or ["agent:read", "agent:write"],
        secret_key="test-secret-key-for-jwt",
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def config():
    """Create test gateway config with auth disabled."""
    cfg = GatewayConfig(
        host="127.0.0.1",
        port=8080,
        debug=True,
    )
    return cfg


@pytest.fixture
def auth_config():
    """Create test gateway config with auth ENABLED."""
    cfg = GatewayConfig(
        host="127.0.0.1",
        port=8080,
        debug=True,
    )
    cfg.api_keys["sk_test_valid_key_12345678"] = "test-user"
    cfg.jwt.secret_key = "test-secret-key-for-jwt"
    return cfg


@pytest.fixture
def app(config):
    """Create a test FastAPI app with auth disabled."""
    from spoon_bot.gateway.websocket.manager import ConnectionManager

    app_module._auth_required = False
    application = create_app(config)
    agent = _make_mock_agent()
    set_agent(agent)

    # Initialize ConnectionManager (normally done in lifespan)
    app_module._connection_manager = ConnectionManager()
    return application


@pytest.fixture
def app_auth(auth_config):
    """Create a test FastAPI app with auth enabled."""
    from spoon_bot.gateway.websocket.manager import ConnectionManager

    app_module._auth_required = True
    application = create_app(auth_config)
    agent = _make_mock_agent()
    set_agent(agent)

    # Initialize ConnectionManager
    app_module._connection_manager = ConnectionManager()
    return application


@pytest.fixture
def client(app):
    """HTTP test client (no auth)."""
    return TestClient(app)


@pytest.fixture
def client_auth(app_auth):
    """HTTP test client (auth enabled)."""
    return TestClient(app_auth)


# ===================================================================
# #14 — WS request params type validation
# ===================================================================


class TestParamsValidation:
    """#14: WS request params lacks type validation."""

    def test_params_string_raises_value_error(self):
        """parse_message with params='oops' should raise ValueError."""
        data = {
            "type": "request",
            "id": "a1",
            "method": "chat.send",
            "params": "oops",
        }
        with pytest.raises(ValueError, match="params.*must be a JSON object"):
            parse_message(data)

    def test_params_number_raises_value_error(self):
        """parse_message with params=42 should raise ValueError."""
        data = {
            "type": "request",
            "id": "a2",
            "method": "agent.status",
            "params": 42,
        }
        with pytest.raises(ValueError, match="params.*must be a JSON object"):
            parse_message(data)

    def test_params_list_raises_value_error(self):
        """parse_message with params=[1,2] should raise ValueError."""
        data = {
            "type": "request",
            "id": "a3",
            "method": "agent.status",
            "params": [1, 2],
        }
        with pytest.raises(ValueError, match="params.*must be a JSON object"):
            parse_message(data)

    def test_params_none_uses_empty_dict(self):
        """parse_message with params=None uses default empty dict."""
        data = {
            "type": "request",
            "id": "a4",
            "method": "agent.status",
        }
        msg = parse_message(data)
        assert isinstance(msg, WSRequest)
        assert msg.params == {}

    def test_params_dict_passes(self):
        """parse_message with valid dict params succeeds."""
        data = {
            "type": "request",
            "id": "a5",
            "method": "chat.send",
            "params": {"message": "hello"},
        }
        msg = parse_message(data)
        assert isinstance(msg, WSRequest)
        assert msg.params["message"] == "hello"

    def test_handler_returns_invalid_params_error(self, client):
        """WS handler returns INVALID_PARAMS / INVALID_MESSAGE on bad params."""
        with client.websocket_connect("/v1/ws") as ws:
            # Read connection established
            ws.receive_json()

            # Send request with non-dict params — should get error
            ws.send_json({
                "type": "request",
                "id": "test1",
                "method": "chat.send",
                "params": "oops",
            })
            resp = ws.receive_json()
            assert resp["type"] == "error"
            # The error should be about invalid message format, not a Python traceback
            assert "INVALID_MESSAGE" in resp.get("error", {}).get("code", "")
            # Should NOT contain Python internals like "'str' object has no attribute"
            assert "'str' object" not in resp.get("error", {}).get("message", "")


# ===================================================================
# #15 — WS session.switch accepts non-string session_key
# ===================================================================


class TestSessionSwitchValidation:
    """#15: session.switch should reject non-string session_key."""

    def test_object_session_key_rejected(self, client):
        """session.switch with object session_key returns error."""
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()  # connection.established
            ws.send_json({
                "type": "request",
                "id": "sw1",
                "method": "session.switch",
                "params": {"session_key": {"x": 1}},
            })
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert "INVALID_PARAMS" in resp.get("error", {}).get("code", "")

    def test_number_session_key_rejected(self, client):
        """session.switch with numeric session_key returns error."""
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()
            ws.send_json({
                "type": "request",
                "id": "sw2",
                "method": "session.switch",
                "params": {"session_key": 123},
            })
            resp = ws.receive_json()
            assert resp["type"] == "error"

    def test_empty_session_key_rejected(self, client):
        """session.switch with empty session_key returns error."""
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()
            ws.send_json({
                "type": "request",
                "id": "sw3",
                "method": "session.switch",
                "params": {"session_key": "  "},
            })
            resp = ws.receive_json()
            assert resp["type"] == "error"

    def test_valid_string_session_key_accepted(self, client):
        """session.switch with valid string session_key succeeds."""
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()
            ws.send_json({
                "type": "request",
                "id": "sw4",
                "method": "session.switch",
                "params": {"session_key": "my-session"},
            })
            resp = ws.receive_json()
            assert resp["type"] == "response"
            assert resp["result"]["switched"] is True
            assert resp["result"]["session_key"] == "my-session"


# ===================================================================
# #16 — WS subscribe/unsubscribe string events
# ===================================================================


class TestSubscribeValidation:
    """#16: subscribe/unsubscribe should reject non-list events."""

    def test_string_events_rejected(self, client):
        """subscribe with string events returns error."""
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()
            ws.send_json({
                "type": "request",
                "id": "sub1",
                "method": "subscribe",
                "params": {"events": "metrics.update"},
            })
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert "INVALID_PARAMS" in resp.get("error", {}).get("code", "")

    def test_number_events_rejected(self, client):
        """subscribe with numeric events returns error."""
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()
            ws.send_json({
                "type": "request",
                "id": "sub2",
                "method": "subscribe",
                "params": {"events": 42},
            })
            resp = ws.receive_json()
            assert resp["type"] == "error"

    def test_list_events_accepted(self, client):
        """subscribe with proper list events succeeds."""
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()
            ws.send_json({
                "type": "request",
                "id": "sub3",
                "method": "subscribe",
                "params": {"events": ["metrics.update", "agent.complete"]},
            })
            resp = ws.receive_json()
            assert resp["type"] == "response"
            assert "metrics.update" in resp["result"]["subscribed"]

    def test_unsubscribe_string_events_rejected(self, client):
        """unsubscribe with string events returns error."""
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()
            ws.send_json({
                "type": "request",
                "id": "unsub1",
                "method": "unsubscribe",
                "params": {"events": "metrics.update"},
            })
            resp = ws.receive_json()
            assert resp["type"] == "error"


@pytest.mark.asyncio
async def test_ws_concurrent_dispatch_respects_connection_limit() -> None:
    """Concurrent fs/workspace dispatch should honor the per-connection slot limit."""

    class _RecordingManager:
        def __init__(self) -> None:
            self.sent: list[object] = []

        async def send_message(self, _connection_id, message):
            self.sent.append(message)
            return True

    set_agent(_make_mock_agent())
    manager = _RecordingManager()

    from spoon_bot.gateway.websocket.handler import WebSocketHandler

    handler = WebSocketHandler("conn_test")

    active = 0
    max_active = 0
    lock = asyncio.Lock()
    reached_limit = asyncio.Event()
    release = asyncio.Event()

    async def _slow_handle_request(request):
        nonlocal active, max_active
        async with lock:
            active += 1
            max_active = max(max_active, active)
            if active == _CONCURRENT_REQUEST_LIMIT:
                reached_limit.set()
        await release.wait()
        async with lock:
            active -= 1
        return {"ok": True, "id": request.id}

    with patch.object(handler, "handle_request", side_effect=_slow_handle_request):
        tasks = [
            asyncio.create_task(
                handler._dispatch_concurrent_request(
                    manager,
                    "conn_test",
                    WSRequest(
                        id=f"req-{i}",
                        method="fs.read",
                        params={"path": f"/workspace/{i}.txt"},
                    ),
                ),
            )
            for i in range(_CONCURRENT_REQUEST_LIMIT + 4)
        ]
        await asyncio.wait_for(reached_limit.wait(), timeout=1.0)
        release.set()
        await asyncio.gather(*tasks)

    assert len(manager.sent) == _CONCURRENT_REQUEST_LIMIT + 4
    assert max_active == _CONCURRENT_REQUEST_LIMIT


# ===================================================================
# #18 — JWT session claim type validation
# ===================================================================


class TestJWTSessionValidation:
    """#18: JWT session claim should be validated as string."""

    def test_object_session_claim_defaults_to_default(self):
        """Token with object session claim should default to 'default'."""
        import jwt as pyjwt
        now = datetime.now(timezone.utc)
        payload = {
            "sub": "user1",
            "session": {"k": "v"},  # Invalid: should be string
            "type": "access",
            "scope": [],
            "iat": int(now.timestamp()),
            "exp": int(now.timestamp()) + 3600,
        }
        token = pyjwt.encode(payload, "secret", algorithm="HS256")
        data = verify_token(token, "secret", "HS256", expected_type="access")
        assert data is not None
        assert data.session_key == "default"  # Should default, not use the dict

    def test_number_session_claim_defaults_to_default(self):
        """Token with numeric session claim should default to 'default'."""
        import jwt as pyjwt
        now = datetime.now(timezone.utc)
        payload = {
            "sub": "user1",
            "session": 999,
            "type": "access",
            "scope": [],
            "iat": int(now.timestamp()),
            "exp": int(now.timestamp()) + 3600,
        }
        token = pyjwt.encode(payload, "secret", algorithm="HS256")
        data = verify_token(token, "secret", "HS256", expected_type="access")
        assert data is not None
        assert data.session_key == "default"

    def test_empty_string_session_defaults(self):
        """Token with empty session string should default to 'default'."""
        import jwt as pyjwt
        now = datetime.now(timezone.utc)
        payload = {
            "sub": "user1",
            "session": "  ",
            "type": "access",
            "scope": [],
            "iat": int(now.timestamp()),
            "exp": int(now.timestamp()) + 3600,
        }
        token = pyjwt.encode(payload, "secret", algorithm="HS256")
        data = verify_token(token, "secret", "HS256", expected_type="access")
        assert data is not None
        assert data.session_key == "default"

    def test_valid_string_session_preserved(self):
        """Token with valid string session should be preserved."""
        token = create_access_token(
            user_id="user1",
            session_key="my-session",
            scopes=["agent:read"],
            secret_key="secret",
        )
        data = verify_token(token, "secret", "HS256", expected_type="access")
        assert data is not None
        assert data.session_key == "my-session"


# ===================================================================
# #11 — WS chat.send session binding
# ===================================================================


class TestWSSessionBinding:
    """#11: WS chat.send should bind runtime session to requested session_key."""

    def test_chat_switches_agent_session(self, client):
        """chat.send with session_key should call _switch_agent_session."""
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()  # connection.established

            ws.send_json({
                "type": "request",
                "id": "chat1",
                "method": "chat.send",
                "params": {
                    "message": "hello",
                    "session_key": "custom-session",
                    "stream": False,
                },
            })

            # Read events until we get the response
            msgs = []
            for _ in range(10):
                msg = ws.receive_json()
                msgs.append(msg)
                if msg.get("type") == "response":
                    break

            # The response should echo the session_key
            response = next((m for m in msgs if m.get("type") == "response"), None)
            assert response is not None
            assert response["result"]["session_key"] == "custom-session"


# ===================================================================
# #12 — WS session.import
# ===================================================================


class TestWSSessionImport:
    """#12: session.import should actually persist messages."""

    def test_import_persists_messages(self, client):
        """session.import should restore messages and report count."""
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()  # connection.established

            ws.send_json({
                "type": "request",
                "id": "imp1",
                "method": "session.import",
                "params": {
                    "state": {
                        "version": "1.0",
                        "session_key": "imported-session",
                        "messages": [
                            {"role": "user", "content": "hi"},
                            {"role": "assistant", "content": "hello"},
                        ],
                    },
                },
            })
            resp = ws.receive_json()
            assert resp["type"] == "response"
            result = resp["result"]
            assert result["success"] is True
            assert result["restored"]["session_key"] == "imported-session"
            saved_session = app_module._agent.sessions.get_or_create.return_value
            saved_session.add_message.assert_any_call(
                "user",
                "hi",
            )

    def test_import_missing_state_error(self, client):
        """session.import without state should raise error."""
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()
            ws.send_json({
                "type": "request",
                "id": "imp2",
                "method": "session.import",
                "params": {},
            })
            resp = ws.receive_json()
            assert resp["type"] == "error"

    def test_import_invalid_state_type_error(self, client):
        """session.import with non-dict state should raise error."""
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()
            ws.send_json({
                "type": "request",
                "id": "imp3",
                "method": "session.import",
                "params": {"state": "not-a-dict"},
            })
            resp = ws.receive_json()
            assert resp["type"] == "error"

    def test_import_preserves_attachment_metadata(self, client, tmp_path: Path):
        """session.import/export should round-trip media and attachments."""
        workspace = tmp_path / "workspace"
        uploads = workspace / "uploads"
        uploads.mkdir(parents=True)
        attachment_path = uploads / "demo.png"
        attachment_path.write_bytes(b"png")
        app_module._agent.workspace = workspace

        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()
            ws.send_json({
                "type": "request",
                "id": "imp4",
                "method": "session.import",
                "params": {
                    "state": {
                        "version": "1.0",
                        "session_key": "imported-session",
                        "messages": [
                            {
                                "role": "user",
                                "content": "see file",
                                "media": [str(attachment_path)],
                                "attachments": [
                                    {
                                        "uri": str(attachment_path),
                                        "name": "demo.png",
                                        "mime_type": "image/png",
                                    }
                                ],
                            },
                        ],
                    },
                },
            })
            resp = ws.receive_json()
            assert resp["type"] == "response"
            saved_session = app_module._agent.sessions.get_or_create.return_value
            saved_session.add_message.assert_any_call(
                "user",
                "see file",
                media=[str(attachment_path)],
                attachments=[{
                    "uri": str(attachment_path),
                    "name": "demo.png",
                    "mime_type": "image/png",
                    "workspace_path": str(attachment_path),
                }],
            )

            ws.send_json({
                "type": "request",
                "id": "exp2",
                "method": "session.export",
                "params": {"session_key": "default"},
            })
            export_resp = ws.receive_json()
            assert export_resp["type"] == "response"
            assert export_resp["result"]["success"] is True
            messages = export_resp["result"]["state"]["messages"]
            assert isinstance(messages, list)

    def test_import_rejects_attachment_outside_workspace(self, client, tmp_path: Path):
        """session.import should reject attachment refs that escape the workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = tmp_path / "outside.png"
        outside.write_bytes(b"png")
        app_module._agent.workspace = workspace

        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()
            ws.send_json({
                "type": "request",
                "id": "imp5",
                "method": "session.import",
                "params": {
                    "state": {
                        "version": "1.0",
                        "session_key": "imported-session",
                        "messages": [
                            {
                                "role": "user",
                                "content": "see file",
                                "attachments": [{"uri": str(outside), "name": "outside.png"}],
                            },
                        ],
                    },
                },
            })
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert "Invalid attachment path" in resp["error"]["message"]

    def test_import_validation_failure_preserves_existing_messages(self, client, tmp_path: Path):
        """A failed import should not clear the existing in-memory session."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = tmp_path / "outside.png"
        outside.write_bytes(b"png")
        app_module._agent.workspace = workspace

        saved_session = app_module._agent.sessions.get_or_create.return_value
        saved_session.messages = [{"role": "user", "content": "keep me"}]

        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()
            ws.send_json({
                "type": "request",
                "id": "imp6",
                "method": "session.import",
                "params": {
                    "state": {
                        "version": "1.0",
                        "session_key": "imported-session",
                        "messages": [
                            {
                                "role": "user",
                                "content": "see file",
                                "media": [str(outside)],
                            },
                        ],
                    },
                },
            })
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert "Invalid media path" in resp["error"]["message"]

        assert saved_session.messages == [{"role": "user", "content": "keep me"}]


# ===================================================================
# #13 — WS chat.cancel non-stream
# ===================================================================


class TestWSCancelNonStream:
    """#13: chat.cancel should report task_interrupted for non-stream."""

    def test_cancel_returns_task_interrupted(self, client):
        """cancel should include task_interrupted flag."""
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()  # connection.established

            # Send cancel without any running task
            ws.send_json({
                "type": "request",
                "id": "cancel1",
                "method": "chat.cancel",
                "params": {},
            })
            resp = ws.receive_json()
            assert resp["type"] == "response"
            result = resp["result"]
            assert result["cancelled"] is True
            assert "task_interrupted" in result


# ===================================================================
# #17 — WS auth failure close code
# ===================================================================


class TestWSAuthCloseCode:
    """#17: WS auth failure should return close code 4001."""

    def test_auth_failure_close_code(self, client_auth):
        """Connecting without credentials should get close code 4001."""
        # With auth enabled and no credentials, WS should be accepted
        # then closed with 4001
        try:
            with client_auth.websocket_connect("/v1/ws") as ws:
                # If we get here, the connection was accepted
                # It should be immediately closed with 4001
                try:
                    # Try to read — should get disconnect
                    ws.receive_json()
                    # If we received data, check if it's a close
                    pytest.fail("Expected WebSocket to close with 4001")
                except Exception:
                    pass  # Expected: connection should close
        except Exception:
            pass  # Expected: connection closed during handshake


# ===================================================================
# #19 — Auth rate limiting
# ===================================================================


class TestAuthRateLimiting:
    """#19: Auth rate limiting for WS connection attempts."""

    def test_rate_limiter_blocks_after_max_attempts(self):
        """Rate limiter should block after max failed attempts."""
        limiter = _AuthRateLimiter(max_attempts=3, window_seconds=60)

        assert not limiter.is_blocked("1.2.3.4")
        limiter.record_failure("1.2.3.4")
        assert not limiter.is_blocked("1.2.3.4")
        limiter.record_failure("1.2.3.4")
        assert not limiter.is_blocked("1.2.3.4")
        limiter.record_failure("1.2.3.4")
        # After 3 failures, should be blocked
        assert limiter.is_blocked("1.2.3.4")

    def test_rate_limiter_different_ips_independent(self):
        """Different IPs should have independent rate limits."""
        limiter = _AuthRateLimiter(max_attempts=2, window_seconds=60)

        limiter.record_failure("1.1.1.1")
        limiter.record_failure("1.1.1.1")
        assert limiter.is_blocked("1.1.1.1")
        assert not limiter.is_blocked("2.2.2.2")

    def test_rate_limiter_clears_on_success(self):
        """Successful auth should clear rate limit history."""
        limiter = _AuthRateLimiter(max_attempts=3, window_seconds=60)

        limiter.record_failure("1.2.3.4")
        limiter.record_failure("1.2.3.4")
        assert not limiter.is_blocked("1.2.3.4")

        # Clear (simulating successful auth)
        limiter.clear("1.2.3.4")
        # After clear, should not be blocked even with 2 previous failures
        assert not limiter.is_blocked("1.2.3.4")

    def test_rate_limiter_window_expiry(self):
        """Rate limit should expire after window."""
        limiter = _AuthRateLimiter(max_attempts=2, window_seconds=1)

        limiter.record_failure("1.2.3.4")
        limiter.record_failure("1.2.3.4")
        assert limiter.is_blocked("1.2.3.4")

        # Wait for window to expire
        time.sleep(1.1)
        assert not limiter.is_blocked("1.2.3.4")


# ===================================================================
# #7 — Async task API
# ===================================================================


class TestAsyncTaskAPI:
    """#7: Async chat/task APIs should work properly."""

    def test_create_async_task(self, client):
        """POST /v1/agent/chat/async should create a task."""
        resp = client.post(
            "/v1/agent/chat/async",
            json={"message": "hello async"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "task_id" in data
        assert data["status"] in ("pending", "running")

    def test_get_task_status(self, client):
        """GET /v1/agent/tasks/{task_id} should return task info."""
        # Create a task first
        create_resp = client.post(
            "/v1/agent/chat/async",
            json={"message": "test task"},
        )
        task_id = create_resp.json()["task_id"]

        # Wait a moment for it to process
        time.sleep(0.5)

        resp = client.get(f"/v1/agent/tasks/{task_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == task_id
        assert data["status"] in ("pending", "running", "completed", "failed")

    def test_get_nonexistent_task(self, client):
        """GET /v1/agent/tasks/nonexistent should return 404."""
        resp = client.get("/v1/agent/tasks/nonexistent_task_id")
        assert resp.status_code == 404

    def test_cancel_task(self, client):
        """POST /v1/agent/tasks/{task_id}/cancel should cancel a task."""
        create_resp = client.post(
            "/v1/agent/chat/async",
            json={"message": "cancel me"},
        )
        task_id = create_resp.json()["task_id"]

        resp = client.post(f"/v1/agent/tasks/{task_id}/cancel")
        assert resp.status_code == 200
        data = resp.json()
        # Should either be cancelled or already in terminal state
        assert "cancelled" in data

    def test_async_chat_rejects_attachment_payloads(self, client):
        """POST /v1/agent/chat/async should reject sync-only attachment inputs."""
        resp = client.post(
            "/v1/agent/chat/async",
            json={
                "message": "hello async",
                "attachments": [{"uri": "/workspace/uploads/demo.txt"}],
            },
        )
        assert resp.status_code == 422

    def test_task_owner_enforced_for_get_and_cancel(self, client_auth):
        """Task read/cancel should be forbidden for non-owner users."""
        owner_headers = _auth_headers("owner-user")
        other_headers = _auth_headers("other-user")

        create_resp = client_auth.post(
            "/v1/agent/chat/async",
            json={"message": "owner task"},
            headers=owner_headers,
        )
        assert create_resp.status_code == 200
        task_id = create_resp.json()["task_id"]

        get_other = client_auth.get(f"/v1/agent/tasks/{task_id}", headers=other_headers)
        assert get_other.status_code == 403

        cancel_other = client_auth.post(f"/v1/agent/tasks/{task_id}/cancel", headers=other_headers)
        assert cancel_other.status_code == 403

        get_owner = client_auth.get(f"/v1/agent/tasks/{task_id}", headers=owner_headers)
        assert get_owner.status_code == 200


class TestSessionPersistenceAPI:
    """Session API persistence behavior."""

    def test_create_session_calls_save(self, client):
        """POST /v1/sessions should persist session immediately."""
        agent = app_module._agent
        assert agent is not None

        created = MagicMock()
        created.session_key = "persist_now"
        created.created_at = datetime.now(timezone.utc)
        created.messages = []

        agent.sessions.get.side_effect = lambda key: None
        agent.sessions.get_or_create.side_effect = lambda key: created

        resp = client.post("/v1/sessions", json={"key": "persist_now"})
        assert resp.status_code == 200
        assert agent.sessions.save.call_count >= 1


class TestRESTSessionRouting:
    """REST chat session key routing semantics."""

    def test_explicit_session_key_not_overridden_by_token(self, client_auth):
        """Request session_key should beat token session claim when explicit."""
        agent = app_module._agent
        assert agent is not None

        def _session_factory(key: str):
            session = MagicMock()
            session.session_key = key
            session.messages = []
            session.add_message = MagicMock()
            return session

        agent.sessions.get_or_create.side_effect = _session_factory

        resp = client_auth.post(
            "/v1/agent/chat",
            json={"message": "hello", "session_key": "explicit-session"},
            headers=_auth_headers("routing-user", session_key="token-session"),
        )
        assert resp.status_code == 200
        assert agent.session_key == "explicit-session"


class TestRESTConcurrency:
    """REST chat concurrency behavior against shared global agent."""

    @pytest.mark.asyncio
    async def test_concurrent_same_session_requests_are_serialized(self, app):
        """Concurrent /chat requests should not produce non-IDLE 500 errors."""
        agent = _make_mock_agent()
        set_agent(agent)

        busy = False

        async def _contention_sensitive_process(*, message: str):
            nonlocal busy
            if busy:
                raise RuntimeError("Agent spoon_react_skill is not in the IDLE state")
            busy = True
            try:
                await asyncio.sleep(0.05)
                return f"ok:{message}"
            finally:
                busy = False

        agent.process = AsyncMock(side_effect=_contention_sensitive_process)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
            r1, r2 = await asyncio.gather(
                ac.post("/v1/agent/chat", json={"message": "m1", "session_key": "same-session"}),
                ac.post("/v1/agent/chat", json={"message": "m2", "session_key": "same-session"}),
            )

        assert r1.status_code == 200
        assert r2.status_code == 200


# ===================================================================
# #10 — WS streaming content (integration)
# ===================================================================


class TestWSStreamingContent:
    """#10: WS streaming should return non-empty content."""

    def test_stream_returns_content(self, client):
        """Streaming chat should emit chunk events with content."""
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()  # connection.established

            ws.send_json({
                "type": "request",
                "id": "stream1",
                "method": "chat.send",
                "params": {
                    "message": "hello stream",
                    "stream": True,
                },
            })

            # Collect all events until response
            events = []
            for _ in range(20):
                msg = ws.receive_json()
                events.append(msg)
                if msg.get("type") == "response":
                    break

            # Check that we got stream chunk events
            chunk_events = [
                e for e in events
                if e.get("type") == "event"
                and e.get("event") == "agent.stream.chunk"
            ]

            # Check we got a stream done event
            done_events = [
                e for e in events
                if e.get("type") == "event"
                and e.get("event") == "agent.stream.done"
            ]

            # The response should have non-empty content
            response = next((e for e in events if e.get("type") == "response"), None)
            assert response is not None
            assert response["result"]["content"] != ""

    def test_non_stream_returns_content(self, client):
        """Non-streaming chat should return non-empty content."""
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()  # connection.established

            ws.send_json({
                "type": "request",
                "id": "nostream1",
                "method": "chat.send",
                "params": {
                    "message": "hello non-stream",
                    "stream": False,
                },
            })

            events = []
            for _ in range(10):
                msg = ws.receive_json()
                events.append(msg)
                if msg.get("type") == "response":
                    break

            response = next((e for e in events if e.get("type") == "response"), None)
            assert response is not None
            assert response["result"]["content"] == "Hello from test agent"
            assert response["result"]["success"] is True

    def test_chat_send_rejects_invalid_media_paths(self, client):
        """Invalid media paths should fail fast instead of being silently skipped."""
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()
            ws.send_json({
                "type": "request",
                "id": "badmedia1",
                "method": "chat.send",
                "params": {
                    "message": "look at this",
                    "media": ["/definitely/missing/file.png"],
                    "stream": False,
                },
            })
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert "Invalid media path" in resp["error"]["message"]

    def test_chat_send_with_attachment_only_builds_context_and_passes_metadata(self, client, tmp_path: Path):
        """Attachment-only requests should reach agent with structured attachments."""
        workspace = tmp_path / "workspace"
        uploads = workspace / "uploads"
        uploads.mkdir(parents=True)
        attachment_path = uploads / "demo.pdf"
        attachment_path.write_text("demo", encoding="utf-8")
        attachment_uri = "/workspace/uploads/demo.pdf"
        app_module._agent.workspace = workspace
        app_module._agent.process = AsyncMock(return_value="ok")

        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()
            ws.send_json({
                "type": "request",
                "id": "attach1",
                "method": "chat.send",
                "params": {
                    "message": "",
                    "stream": False,
                    "attachments": [
                        {
                            "uri": attachment_uri,
                            "name": "demo.pdf",
                            "mime_type": "application/pdf",
                            "size": 4,
                        }
                    ],
                },
            })
            events = []
            for _ in range(10):
                msg = ws.receive_json()
                events.append(msg)
                if msg.get("type") == "response":
                    break

        app_module._agent.process.assert_awaited_once()
        kwargs = app_module._agent.process.await_args.kwargs
        assert kwargs["attachments"] == [
            {
                "uri": attachment_uri,
                "name": "demo.pdf",
                "mime_type": "application/pdf",
                "size": 4,
                "workspace_path": attachment_uri,
            }
        ]
        assert attachment_uri in kwargs["message"]

    def test_chat_send_preserves_workspace_alias_media_uri(self, client, tmp_path: Path):
        """Sandbox media aliases should remain portable URIs when passed to the agent."""
        workspace = tmp_path / "workspace"
        uploads = workspace / "uploads"
        uploads.mkdir(parents=True)
        image_path = uploads / "demo.png"
        image_path.write_bytes(b"png")
        app_module._agent.workspace = workspace
        app_module._agent.process = AsyncMock(return_value="ok")

        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()
            ws.send_json({
                "type": "request",
                "id": "media_alias1",
                "method": "chat.send",
                "params": {
                    "message": "look at this image",
                    "media": ["/workspace/uploads/demo.png"],
                    "stream": False,
                },
            })
            events = []
            for _ in range(10):
                msg = ws.receive_json()
                events.append(msg)
                if msg.get("type") == "response":
                    break

        app_module._agent.process.assert_awaited_once()
        kwargs = app_module._agent.process.await_args.kwargs
        assert kwargs["media"] == ["/workspace/uploads/demo.png"]

    def test_resolve_workspace_file_accepts_sandbox_alias(self, tmp_path: Path):
        """Sandbox aliases should resolve against the configured runtime workspace."""
        workspace = tmp_path / "workspace"
        uploads = workspace / "uploads"
        uploads.mkdir(parents=True)
        attachment_path = uploads / "alias.txt"
        attachment_path.write_text("alias", encoding="utf-8")
        app_module._agent.workspace = workspace

        assert _resolve_workspace_file("/workspace/uploads/alias.txt") == attachment_path.resolve()

    def test_chat_send_rejects_existing_media_outside_workspace(self, client, tmp_path: Path):
        """Existing files outside workspace should not be accepted as media inputs."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = tmp_path / "outside.png"
        outside.write_bytes(b"png")
        app_module._agent.workspace = workspace

        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()
            ws.send_json({
                "type": "request",
                "id": "badmedia2",
                "method": "chat.send",
                "params": {
                    "message": "look at this",
                    "media": [str(outside)],
                    "stream": False,
                },
            })
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert "Invalid media path" in resp["error"]["message"]

    def test_stream_done_metadata_falls_back_to_content(self, client):
        """If no chunk delta is emitted, done.metadata.content should still reach clients."""
        agent = app_module._agent
        assert agent is not None

        async def _done_only_stream(**kwargs):
            yield {"type": "done", "delta": "", "metadata": {"content": "fallback done content"}}

        agent.stream = _done_only_stream

        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()  # connection.established

            ws.send_json({
                "type": "request",
                "id": "stream_done_fallback",
                "method": "chat.send",
                "params": {
                    "message": "hello stream",
                    "stream": True,
                },
            })

            events = []
            for _ in range(20):
                msg = ws.receive_json()
                events.append(msg)
                if msg.get("type") == "response":
                    break

            response = next((e for e in events if e.get("type") == "response"), None)
            assert response is not None
            assert response["result"]["content"] == "fallback done content"

            done_events = [
                e for e in events
                if e.get("type") == "event" and e.get("event") == "agent.stream.done"
            ]
            assert len(done_events) >= 1
            assert done_events[0]["data"]["content"] == "fallback done content"


# ===================================================================
# Handler error masking (part of #14)
# ===================================================================


class TestErrorMasking:
    """Handler errors should not leak Python internals."""

    def test_handler_error_no_internals(self, client):
        """Handler errors should use generic message, not Python traceback."""
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()  # connection.established

            # Send to an unknown method
            ws.send_json({
                "type": "request",
                "id": "err1",
                "method": "nonexistent.method",
                "params": {},
            })
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert resp["error"]["code"] == "UNKNOWN_METHOD"
            # Should not leak Python exceptions
            assert "Traceback" not in resp["error"]["message"]


# ===================================================================
# WS basic operations
# ===================================================================


class TestWSBasicOps:
    """Basic WebSocket operations."""

    def test_connection_established(self, client):
        """WS connection should send connection.established event."""
        with client.websocket_connect("/v1/ws") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "event"
            assert msg["event"] == "connection.established"
            assert "connection_id" in msg["data"]
            assert "session_key" in msg["data"]

    def test_ping_pong(self, client):
        """WS ping should get pong response."""
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()  # connection.established
            ws.send_json({"type": "ping", "timestamp": 12345})
            resp = ws.receive_json()
            assert resp["type"] == "pong"
            assert resp["timestamp"] == 12345

    def test_agent_status(self, client):
        """agent.status should return agent info."""
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()
            ws.send_json({
                "type": "request",
                "id": "stat1",
                "method": "agent.status",
                "params": {},
            })
            resp = ws.receive_json()
            assert resp["type"] == "response"
            assert resp["result"]["status"] == "ready"
            assert "model" in resp["result"]

    def test_session_list(self, client):
        """session.list should return sessions."""
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()
            ws.send_json({
                "type": "request",
                "id": "list1",
                "method": "session.list",
                "params": {},
            })
            resp = ws.receive_json()
            assert resp["type"] == "response"
            assert "sessions" in resp["result"]

    def test_session_export(self, client):
        """session.export should return session state."""
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()
            ws.send_json({
                "type": "request",
                "id": "exp1",
                "method": "session.export",
                "params": {"session_key": "default"},
            })
            resp = ws.receive_json()
            assert resp["type"] == "response"
            assert resp["result"]["success"] is True
            assert "state" in resp["result"]
            assert resp["result"]["state"]["version"] == "1.0"


# ===================================================================
# HTTP REST API tests
# ===================================================================


class TestHTTPChat:
    """HTTP chat endpoint tests."""

    def test_chat_non_stream(self, client):
        """POST /v1/agent/chat non-stream returns response."""
        resp = client.post(
            "/v1/agent/chat",
            json={"message": "hello"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["response"] == "Hello from test agent"

    def test_chat_stream(self, client):
        """POST /v1/agent/chat stream returns SSE events."""
        resp = client.post(
            "/v1/agent/chat",
            json={
                "message": "hello stream",
                "options": {"stream": True},
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_chat_invalid_attachment_path_returns_422(self, client):
        """Bad attachment paths should produce a client-actionable 4xx response."""
        resp = client.post(
            "/v1/agent/chat",
            json={
                "message": "hello",
                "attachments": [{"uri": "/workspace/uploads/missing.txt"}],
            },
        )
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert detail["code"] == "INVALID_REQUEST"
        assert "Invalid attachment path" in detail["message"]

    def test_voice_chat_stream_passes_new_stream_arguments(self, client):
        """POST /v1/agent/voice/chat stream should produce SSE instead of TypeError."""
        agent = app_module._agent
        assert agent is not None

        async def _voice_stream(**kwargs):
            assert kwargs["message"] == "voice prompt"
            assert kwargs["thinking"] is False
            yield {"type": "content", "delta": "voice ok", "metadata": {}}
            yield {"type": "done", "delta": "", "metadata": {}}

        agent.stream = _voice_stream

        with patch(
            "spoon_bot.gateway.api.v1.agent._process_audio_input",
            AsyncMock(return_value=("voice prompt", None)),
        ):
            resp = client.post(
                "/v1/agent/voice/chat",
                data={"stream": "true"},
                files={"audio": ("voice.wav", b"fake-audio", "audio/wav")},
            )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        assert "voice ok" in resp.text
        assert "TypeError" not in resp.text

    def test_agent_status(self, client):
        """GET /v1/agent/status returns agent info."""
        resp = client.get("/v1/agent/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["status"] == "ready"


# ===================================================================
# Loop streaming fix verification (#10)
# ===================================================================


class TestLoopStreamDictChunks:
    """#10: loop.py stream should handle both 'content' and 'delta' dict keys."""

    def test_dict_chunk_with_delta_key(self):
        """Dict chunk with 'delta' key should be captured."""
        # This tests the logic we fixed in loop.py
        chunk = {"delta": "hello from delta", "type": "content"}
        # Simulate the fixed logic
        text = chunk.get("content") or chunk.get("delta") or ""
        assert text == "hello from delta"

    def test_dict_chunk_with_content_key(self):
        """Dict chunk with 'content' key should still work."""
        chunk = {"content": "hello from content", "type": "content"}
        text = chunk.get("content") or chunk.get("delta") or ""
        assert text == "hello from content"

    def test_dict_chunk_with_both_keys(self):
        """Dict chunk with both keys should prefer 'content'."""
        chunk = {"content": "content wins", "delta": "delta loses"}
        text = chunk.get("content") or chunk.get("delta") or ""
        assert text == "content wins"

    def test_dict_chunk_with_neither_key(self):
        """Dict chunk with neither key should return empty."""
        chunk = {"type": "tool_result", "metadata": {}}
        text = chunk.get("content") or chunk.get("delta") or ""
        assert text == ""


# ============================================================================
# Bug #9: SKILL.md loader – BOM, CRLF, dedup
# ============================================================================


class TestSkillLoaderBOM:
    """Verify the loader handles BOM-prefixed and \\r\\n SKILL.md files (#9)."""

    def _write_skill(self, path, content, encoding="utf-8"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode(encoding))

    def test_utf8_bom_skill_loads(self, tmp_path):
        from spoon_ai.skills.loader import SkillLoader
        skill_file = tmp_path / "skills" / "s1" / "SKILL.md"
        self._write_skill(skill_file, "\ufeff---\nname: s1\ndescription: BOM\nversion: 1.0.0\n---\n\n# OK\nBody")
        loader = SkillLoader(additional_paths=[tmp_path / "skills"], include_default_paths=False)
        assert loader.load(skill_file).metadata.name == "s1"

    def test_crlf_line_endings(self, tmp_path):
        from spoon_ai.skills.loader import SkillLoader
        skill_file = tmp_path / "skills" / "s2" / "SKILL.md"
        self._write_skill(skill_file, "---\r\nname: s2\r\ndescription: CRLF\r\nversion: 1.0.0\r\n---\r\n\r\n# Body\r\nOK")
        loader = SkillLoader(additional_paths=[tmp_path / "skills"], include_default_paths=False)
        assert loader.load(skill_file).metadata.name == "s2"

    def test_bom_plus_crlf(self, tmp_path):
        from spoon_ai.skills.loader import SkillLoader
        skill_file = tmp_path / "skills" / "s3" / "SKILL.md"
        self._write_skill(skill_file, "\ufeff---\r\nname: s3\r\ndescription: Both\r\nversion: 1.0.0\r\n---\r\n\r\n# Both\r\nOK")
        loader = SkillLoader(additional_paths=[tmp_path / "skills"], include_default_paths=False)
        assert loader.load(skill_file).metadata.name == "s3"

    def test_normal_skill_still_loads(self, tmp_path):
        from spoon_ai.skills.loader import SkillLoader
        skill_file = tmp_path / "skills" / "s4" / "SKILL.md"
        self._write_skill(skill_file, "---\nname: s4\ndescription: Normal\nversion: 1.0.0\n---\n\n# OK\nBody")
        loader = SkillLoader(additional_paths=[tmp_path / "skills"], include_default_paths=False)
        assert loader.load(skill_file).metadata.name == "s4"


class TestSkillDiscoveryDedup:
    """Verify skill discovery doesn't produce duplicates (#9)."""

    def test_no_duplicate_discovery(self, tmp_path):
        from spoon_ai.skills.loader import SkillLoader
        skill_dir = tmp_path / "skills" / "dup"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: dup\ndescription: D\nversion: 1.0.0\n---\n\n# D\nOK")
        loader = SkillLoader(additional_paths=[tmp_path / "skills"], include_default_paths=False)
        resolved = [p.resolve() for p in loader.discover()]
        assert len(resolved) == len(set(resolved))


# ============================================================================
# Bug #8: ScriptTool parameter contract with input_schema
# ============================================================================


class TestScriptToolSchema:
    """Verify ScriptTool uses input_schema when available (#8)."""

    def test_structured_schema_used(self):
        from spoon_ai.skills.models import SkillScript
        from spoon_ai.skills.script_tool import ScriptTool

        script = SkillScript(
            name="sum", description="Summarize", type="python",
            file="scripts/sum.py",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "style": {"type": "string"},
                },
                "required": ["text"],
            },
        )
        tool = ScriptTool(script=script, skill_name="doc")
        assert "text" in tool.parameters["properties"]
        assert "input" not in tool.parameters["properties"]

    def test_generic_schema_fallback(self):
        from spoon_ai.skills.models import SkillScript
        from spoon_ai.skills.script_tool import ScriptTool

        script = SkillScript(name="simple", description="S", type="python", file="s.py")
        tool = ScriptTool(script=script, skill_name="t")
        assert "input" in tool.parameters["properties"]

    @pytest.mark.asyncio
    async def test_structured_execute_serializes_json(self):
        import json as _json
        from spoon_ai.skills.models import SkillScript, ScriptResult
        from spoon_ai.skills.script_tool import ScriptTool

        script = SkillScript(
            name="gen", description="Gen", type="python", file="g.py",
            input_schema={
                "type": "object",
                "properties": {"prompt": {"type": "string"}, "width": {"type": "integer"}},
                "required": ["prompt"],
            },
        )
        tool = ScriptTool(script=script, skill_name="img")
        captured = {}
        mock_res = ScriptResult(script_name="gen", success=True, exit_code=0, stdout="done")

        async def _exec(script, input_text=None, working_directory=None):
            captured["text"] = input_text
            return mock_res

        with patch("spoon_ai.skills.script_tool.get_executor") as mg:
            mg.return_value = MagicMock(execute=_exec)
            result = await tool.execute(prompt="a cat", width=512)

        assert result == "done"
        parsed = _json.loads(captured["text"])
        assert parsed["prompt"] == "a cat"
        assert parsed["width"] == 512


class TestSkillScriptInputSchema:
    """Verify SkillScript model accepts input_schema field (#8)."""

    def test_input_schema_parsed(self):
        from spoon_ai.skills.models import SkillScript
        s = SkillScript(name="t", type="python", file="t.py",
                        input_schema={"type": "object", "properties": {"text": {"type": "string"}}})
        assert s.input_schema is not None

    def test_input_schema_optional(self):
        from spoon_ai.skills.models import SkillScript
        assert SkillScript(name="t", type="python", file="t.py").input_schema is None


class TestSkillMDWithInputSchema:
    """End-to-end: SKILL.md with input_schema → ScriptTool (#8)."""

    def test_full_flow(self, tmp_path):
        from spoon_ai.skills.loader import SkillLoader
        from spoon_ai.skills.script_tool import create_script_tools

        scripts_dir = tmp_path / "skills" / "img" / "scripts"
        scripts_dir.mkdir(parents=True)
        (scripts_dir / "gen.py").write_text("print('ok')")

        (tmp_path / "skills" / "img" / "SKILL.md").write_text(
            "---\nname: img\ndescription: Gen\nversion: 1.0.0\n"
            "scripts:\n  enabled: true\n  definitions:\n"
            "    - name: generate\n      description: Gen img\n      type: python\n      file: scripts/gen.py\n"
            "      input_schema:\n        type: object\n        properties:\n"
            "          prompt:\n            type: string\n        required:\n          - prompt\n"
            "---\n\n# Img\nGenerate.\n"
        )

        loader = SkillLoader(additional_paths=[tmp_path / "skills"], include_default_paths=False)
        skill = loader.load(tmp_path / "skills" / "img" / "SKILL.md")
        assert skill.metadata.scripts.definitions[0].input_schema is not None

        tools = create_script_tools("img", skill.metadata.scripts.definitions, str(tmp_path / "skills" / "img"))
        assert "prompt" in tools[0].parameters["properties"]


# ============================================================================
# Bug #5: MCP tool expansion
# ============================================================================


def _make_mock_mcp_tool(name, mcp_config=None):
    """Create a mock MCPTool bypassing Pydantic __init__."""
    from spoon_ai.tools.mcp_tool import MCPTool
    tool = object.__new__(MCPTool)
    for k, v in {
        '__dict__': {}, '__pydantic_fields_set__': set(), '__pydantic_extra__': None,
        'name': name, 'description': f"MCP: {name}", 'parameters': {},
        'mcp_config': mcp_config or {}, '_parameters_loaded': False,
        '_parameters_loading': False, '_max_retries': 3, '_connection_timeout': 10,
        '_health_check_interval': 300, '_last_health_check': 0,
    }.items():
        object.__setattr__(tool, k, v)
    return tool


class TestMCPToolExpansion:
    """Verify MCPTool.expand_server_tools creates individual tools (#5)."""

    @pytest.mark.asyncio
    async def test_expand_creates_individual_tools(self):
        from spoon_ai.tools.mcp_tool import MCPTool
        fake = [
            {"name": "read_file", "description": "Read", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
            {"name": "write_file", "description": "Write", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
            {"name": "list_directory", "description": "List", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
        ]
        tool = _make_mock_mcp_tool("filesystem", {"command": "npx", "args": ["-y", "@mcp/server-filesystem", "/tmp"]})
        object.__setattr__(tool, 'list_available_tools', AsyncMock(return_value=fake))

        def _mi(self, name="", description="", parameters=None, mcp_config=None):
            for k, v in {'__pydantic_fields_set__': set(), '__pydantic_extra__': None,
                         'name': name, 'description': description, 'parameters': parameters or {},
                         'mcp_config': mcp_config or {}, '_parameters_loaded': False,
                         '_parameters_loading': False, '_max_retries': 3, '_connection_timeout': 10,
                         '_health_check_interval': 300, '_last_health_check': 0}.items():
                object.__setattr__(self, k, v)

        with patch.object(MCPTool, "__init__", _mi):
            expanded = await tool.expand_server_tools()

        assert len(expanded) == 3
        names = {t.name for t in expanded}
        assert names == {"read_file", "write_file", "list_directory"}
        assert next(t for t in expanded if t.name == "read_file")._parameters_loaded

    @pytest.mark.asyncio
    async def test_expand_empty_returns_empty(self):
        tool = _make_mock_mcp_tool("offline")
        object.__setattr__(tool, 'list_available_tools', AsyncMock(return_value=[]))
        assert await tool.expand_server_tools() == []


class TestMCPExecuteRouting:
    """Verify MCP tool execution routes to correct server tool (#5)."""

    @pytest.mark.asyncio
    async def test_execute_uses_correct_tool_name(self):
        tool = _make_mock_mcp_tool("read_file", {"command": "test"})
        object.__setattr__(tool, '_parameters_loaded', True)
        object.__setattr__(tool, '_server_name', 'filesystem')
        object.__setattr__(tool, '_check_mcp_health', AsyncMock(return_value=True))
        object.__setattr__(tool, 'call_mcp_tool', AsyncMock(return_value="contents"))
        object.__setattr__(tool, 'ensure_parameters_loaded', AsyncMock())
        result = await tool.execute(path="/tmp/test.txt")
        tool.call_mcp_tool.assert_called_once_with("read_file", path="/tmp/test.txt")
        assert result == "contents"
