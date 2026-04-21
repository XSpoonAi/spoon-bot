"""Regression tests for the WS disconnect / history-warning noise fix.

These tests lock in the behaviour introduced by commit 2e45e6d and its
follow-ups:

1. ``ConnectionManager.send_message`` must short-circuit on a client-side
   WebSocket disconnect (no retry, no WARNING, connection auto-cleaned).
2. Genuine transient errors must still be retried with the pre-existing
   backoff behaviour.
3. A pre-closed ``client_state`` must be detected before the first send.
4. ``AgentLoop._warn_dropped_refs`` must dedupe per session and include the
   dropped paths the first time they are observed.
5. ``ConnectionManager._ping_loop`` must proactively evict connections that
   have been idle past the configured stale threshold (half-open detection).
6. Ping interval and stale threshold must be env-configurable with sane
   fallbacks for invalid values.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import WebSocketDisconnect
from loguru import logger
from starlette.websockets import WebSocketState

from spoon_bot.agent.loop import AgentLoop
from spoon_bot.gateway.websocket.manager import (
    ConnectionManager,
    _env_float,
    _is_disconnect_error,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _LoguruCapture:
    """Capture loguru records at DEBUG and above for assertions."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []
        self._sink_id: int | None = None

    def __enter__(self) -> "_LoguruCapture":
        def _sink(message: Any) -> None:
            record = message.record
            self.records.append((record["level"].name, record["message"]))

        self._sink_id = logger.add(_sink, level="DEBUG")
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._sink_id is not None:
            logger.remove(self._sink_id)

    def levels(self) -> list[str]:
        return [lvl for lvl, _ in self.records]

    def messages(self, level: str | None = None) -> list[str]:
        if level is None:
            return [m for _, m in self.records]
        return [m for lvl, m in self.records if lvl == level]


def _make_fake_ws(*, send_side_effect: Any, state: WebSocketState = WebSocketState.CONNECTED) -> MagicMock:
    ws = MagicMock()
    ws.accept = AsyncMock()
    ws.close = AsyncMock()
    ws.client_state = state
    ws.send_json = AsyncMock(side_effect=send_side_effect)
    return ws


# ---------------------------------------------------------------------------
# _is_disconnect_error
# ---------------------------------------------------------------------------


class TestIsDisconnectError:
    def test_websocket_disconnect_is_detected(self) -> None:
        assert _is_disconnect_error(WebSocketDisconnect(code=1005)) is True

    @pytest.mark.parametrize(
        "msg",
        [
            'Cannot call "send" once a close message has been sent.',
            "WebSocket is disconnected",
            "Unexpected ASGI message 'websocket.send', after sending 'websocket.close'.",
        ],
    )
    def test_starlette_runtime_errors_are_detected(self, msg: str) -> None:
        assert _is_disconnect_error(RuntimeError(msg)) is True

    def test_unrelated_runtime_error_is_not_detected(self) -> None:
        assert _is_disconnect_error(RuntimeError("something unrelated")) is False

    def test_generic_exception_is_not_detected(self) -> None:
        assert _is_disconnect_error(ValueError("boom")) is False

    def test_websockets_connection_closed_is_detected(self) -> None:
        # ``websockets`` is an optional transitive dep — skip if absent.
        websockets = pytest.importorskip("websockets.exceptions")
        exc = websockets.ConnectionClosedError(rcvd=None, sent=None)
        assert _is_disconnect_error(exc) is True


# ---------------------------------------------------------------------------
# ConnectionManager.send_message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSendMessageDisconnectHandling:
    async def test_disconnect_short_circuits_without_retry_warnings(self) -> None:
        manager = ConnectionManager()
        ws = _make_fake_ws(send_side_effect=WebSocketDisconnect(code=1005))
        conn_id = await manager.connect(ws, user_id="u", session_key="s")

        with _LoguruCapture() as cap:
            result = await manager.send_message(conn_id, {"type": "ping"})

        assert result is False
        # No WARNING-level retry spam for a genuine peer disconnect.
        assert not any(
            "failed (attempt" in msg for msg in cap.messages(level="WARNING")
        ), cap.records
        # Connection must be evicted so future sends fast-fail.
        assert manager.connection_count == 0
        assert ws.send_json.await_count == 1

    async def test_preclosed_client_state_skips_send(self) -> None:
        manager = ConnectionManager()
        ws = _make_fake_ws(
            send_side_effect=RuntimeError("should not be called"),
            state=WebSocketState.DISCONNECTED,
        )
        conn_id = await manager.connect(ws, user_id="u", session_key="s")
        # Manually patch state (accept() flips it); simulate a stale conn.
        ws.client_state = WebSocketState.DISCONNECTED

        result = await manager.send_message(conn_id, {"type": "ping"})

        assert result is False
        assert ws.send_json.await_count == 0
        assert manager.connection_count == 0

    async def test_starlette_close_runtime_error_is_treated_as_disconnect(self) -> None:
        manager = ConnectionManager()
        ws = _make_fake_ws(
            send_side_effect=RuntimeError(
                'Cannot call "send" once a close message has been sent.'
            )
        )
        conn_id = await manager.connect(ws, user_id="u", session_key="s")

        with _LoguruCapture() as cap:
            result = await manager.send_message(conn_id, {"type": "ping"})

        assert result is False
        assert not any("failed (attempt" in m for m in cap.messages(level="WARNING"))
        assert manager.connection_count == 0

    async def test_transient_errors_still_retry_and_escalate(self) -> None:
        manager = ConnectionManager()
        ws = _make_fake_ws(send_side_effect=OSError("transient"))
        conn_id = await manager.connect(ws, user_id="u", session_key="s")

        with _LoguruCapture() as cap:
            result = await manager.send_message(conn_id, {"type": "ping"})

        assert result is False
        # Legitimate failure path: 2 retry warnings + 1 terminal ERROR.
        retry_warns = [m for m in cap.messages(level="WARNING") if "failed (attempt" in m]
        error_msgs = cap.messages(level="ERROR")
        assert len(retry_warns) == 2, cap.records
        assert any("after 3 attempts" in m for m in error_msgs), cap.records
        assert ws.send_json.await_count == 3
        assert manager.connection_count == 0  # terminal failure disconnects too

    async def test_successful_send_does_not_log_warnings(self) -> None:
        manager = ConnectionManager()
        ws = _make_fake_ws(send_side_effect=None)
        conn_id = await manager.connect(ws, user_id="u", session_key="s")

        with _LoguruCapture() as cap:
            result = await manager.send_message(conn_id, {"type": "ping"})

        assert result is True
        assert not cap.messages(level="WARNING")
        assert not cap.messages(level="ERROR")
        assert manager.connection_count == 1


# ---------------------------------------------------------------------------
# AgentLoop._warn_dropped_refs
# ---------------------------------------------------------------------------


class _StubLoop:
    """Minimal stand-in that exposes just what ``_warn_dropped_refs`` needs."""

    def __init__(self, session_key: str = "s1") -> None:
        self.session_key = session_key
        self._warned_invalid_attachment_refs: dict[str, set[str]] = {}
        self._warned_invalid_media_refs: dict[str, set[str]] = {}

    warn = AgentLoop._warn_dropped_refs


class TestWarnDroppedRefsDedup:
    def test_first_call_emits_warning_with_paths(self) -> None:
        stub = _StubLoop(session_key="sess-a")
        dropped = ["/ws/a.png", "/ws/b.pdf", "/ws/c.txt"]

        with _LoguruCapture() as cap:
            _StubLoop.warn(stub, kind="attachment", dropped=dropped)

        warnings = cap.messages(level="WARNING")
        assert len(warnings) == 1
        assert "session=sess-a" in warnings[0]
        for ref in dropped:
            assert ref in warnings[0]

    def test_repeated_identical_drops_collapse_to_debug(self) -> None:
        stub = _StubLoop(session_key="sess-a")
        dropped = ["/ws/a.png", "/ws/b.pdf", "/ws/c.txt"]

        with _LoguruCapture() as cap:
            _StubLoop.warn(stub, kind="attachment", dropped=dropped)
            _StubLoop.warn(stub, kind="attachment", dropped=dropped)
            _StubLoop.warn(stub, kind="attachment", dropped=dropped)

        assert len(cap.messages(level="WARNING")) == 1
        assert len(cap.messages(level="DEBUG")) == 2

    def test_new_session_gets_a_fresh_warning(self) -> None:
        stub = _StubLoop(session_key="sess-a")
        dropped = ["/ws/a.png"]
        with _LoguruCapture() as cap:
            _StubLoop.warn(stub, kind="attachment", dropped=dropped)
            stub.session_key = "sess-b"
            _StubLoop.warn(stub, kind="attachment", dropped=dropped)

        assert len(cap.messages(level="WARNING")) == 2

    def test_new_ref_in_known_session_triggers_fresh_warning(self) -> None:
        stub = _StubLoop(session_key="sess-a")
        with _LoguruCapture() as cap:
            _StubLoop.warn(stub, kind="attachment", dropped=["/ws/a.png"])
            _StubLoop.warn(
                stub, kind="attachment", dropped=["/ws/a.png", "/ws/new.jpg"]
            )

        warnings = cap.messages(level="WARNING")
        assert len(warnings) == 2
        assert "/ws/new.jpg" in warnings[1]
        # First warning mentioned only the original ref.
        assert "/ws/new.jpg" not in warnings[0]

    def test_media_and_attachment_caches_are_independent(self) -> None:
        stub = _StubLoop(session_key="sess-a")
        dropped = ["/ws/a.png"]
        with _LoguruCapture() as cap:
            _StubLoop.warn(stub, kind="attachment", dropped=dropped)
            _StubLoop.warn(stub, kind="media", dropped=dropped)

        assert len(cap.messages(level="WARNING")) == 2

    def test_long_drop_list_is_truncated_in_the_warning(self) -> None:
        stub = _StubLoop(session_key="sess-a")
        dropped = [f"/ws/f{i}.bin" for i in range(12)]
        with _LoguruCapture() as cap:
            _StubLoop.warn(stub, kind="attachment", dropped=dropped)

        warnings = cap.messages(level="WARNING")
        assert len(warnings) == 1
        assert "+7 more" in warnings[0]

    def test_empty_drop_list_is_a_no_op(self) -> None:
        stub = _StubLoop(session_key="sess-a")
        with _LoguruCapture() as cap:
            _StubLoop.warn(stub, kind="attachment", dropped=[])
            _StubLoop.warn(stub, kind="attachment", dropped=[""])

        assert not cap.messages(level="WARNING")
        assert not cap.messages(level="DEBUG")


# ---------------------------------------------------------------------------
# Keep-alive / half-open detection
# ---------------------------------------------------------------------------


class TestEnvFloat:
    def test_returns_default_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SPOON_BOT_TEST_FLOAT", raising=False)
        assert _env_float("SPOON_BOT_TEST_FLOAT", default=7.5) == pytest.approx(7.5)

    def test_parses_valid_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SPOON_BOT_TEST_FLOAT", "12.25")
        assert _env_float("SPOON_BOT_TEST_FLOAT", default=0.0) == pytest.approx(12.25)

    def test_clamps_to_minimum(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SPOON_BOT_TEST_FLOAT", "0.01")
        assert _env_float("SPOON_BOT_TEST_FLOAT", default=10.0, minimum=1.0) == 1.0

    def test_falls_back_on_garbage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SPOON_BOT_TEST_FLOAT", "not-a-number")
        with _LoguruCapture() as cap:
            value = _env_float("SPOON_BOT_TEST_FLOAT", default=4.0)
        assert value == pytest.approx(4.0)
        assert any("Invalid value" in m for m in cap.messages(level="WARNING"))

    def test_falls_back_on_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SPOON_BOT_TEST_FLOAT", "   ")
        assert _env_float("SPOON_BOT_TEST_FLOAT", default=2.5) == pytest.approx(2.5)


class TestManagerKeepAliveConfig:
    def test_defaults_are_sub_proxy_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SPOON_BOT_WS_PING_INTERVAL_SECONDS", raising=False)
        monkeypatch.delenv("SPOON_BOT_WS_STALE_SECONDS", raising=False)
        mgr = ConnectionManager()
        # Default ping interval must stay well below typical 60s idle timeouts.
        assert mgr._ping_interval <= 30.0
        # Stale threshold is at least one full interval bigger than the ping
        # cadence so a single missed ping doesn't nuke the connection.
        assert mgr._stale_threshold > mgr._ping_interval

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SPOON_BOT_WS_PING_INTERVAL_SECONDS", "5")
        monkeypatch.setenv("SPOON_BOT_WS_STALE_SECONDS", "40")
        mgr = ConnectionManager()
        assert mgr._ping_interval == pytest.approx(5.0)
        assert mgr._stale_threshold == pytest.approx(40.0)

    def test_stale_threshold_must_exceed_interval(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SPOON_BOT_WS_PING_INTERVAL_SECONDS", "10")
        monkeypatch.setenv("SPOON_BOT_WS_STALE_SECONDS", "5")  # too low
        mgr = ConnectionManager()
        # Minimum clamp in _env_float keeps the threshold strictly larger
        # than the ping interval so one late tick can't evict a live peer.
        assert mgr._stale_threshold > mgr._ping_interval


class TestTouch:
    @pytest.mark.asyncio
    async def test_touch_refreshes_last_activity(self) -> None:
        manager = ConnectionManager()
        ws = _make_fake_ws(send_side_effect=None)
        conn_id = await manager.connect(ws, user_id="u", session_key="s")
        conn = manager.get_connection(conn_id)
        assert conn is not None

        # Artificially backdate last_activity to something clearly stale.
        conn.last_activity = datetime.utcnow() - timedelta(minutes=5)

        manager.touch(conn_id)

        refreshed = (datetime.utcnow() - conn.last_activity).total_seconds()
        assert refreshed < 1.0

    def test_touch_unknown_connection_is_a_no_op(self) -> None:
        manager = ConnectionManager()
        # Must not raise even if the connection id is unknown.
        manager.touch("does-not-exist")


@pytest.mark.asyncio
class TestPingLoopHalfOpenEviction:
    async def test_stale_connection_is_evicted_without_send(self) -> None:
        manager = ConnectionManager()
        manager._ping_interval = 0.01
        manager._stale_threshold = 0.05

        ws = _make_fake_ws(send_side_effect=None)
        conn_id = await manager.connect(ws, user_id="u", session_key="s")
        conn = manager.get_connection(conn_id)
        assert conn is not None

        # Simulate a connection that went quiet long before the loop ticks.
        conn.last_activity = datetime.utcnow() - timedelta(seconds=10)

        await manager.start()
        try:
            # Give the loop enough time to tick at least once.
            for _ in range(50):
                if manager.connection_count == 0:
                    break
                await asyncio.sleep(0.02)
        finally:
            await manager.stop()

        assert manager.connection_count == 0
        # Because we evicted before the send path, send_json must not have
        # been exercised by the ping loop.
        assert ws.send_json.await_count == 0

    async def test_fresh_connection_is_pinged_not_evicted(self) -> None:
        manager = ConnectionManager()
        manager._ping_interval = 0.01
        manager._stale_threshold = 1.0  # plenty of headroom

        ws = _make_fake_ws(send_side_effect=None)
        conn_id = await manager.connect(ws, user_id="u", session_key="s")

        await manager.start()
        try:
            # Wait for at least one ping to be dispatched.
            for _ in range(50):
                if ws.send_json.await_count > 0:
                    break
                await asyncio.sleep(0.02)
        finally:
            await manager.stop()

        assert manager.connection_count == 0  # stop() clears the table
        assert ws.send_json.await_count >= 1
        # Every ping payload must look like a ping frame.
        for call in ws.send_json.await_args_list:
            payload = call.args[0]
            assert payload.get("type") == "ping"

    async def test_ping_loop_cancels_cleanly_on_stop(self) -> None:
        manager = ConnectionManager()
        manager._ping_interval = 10.0  # long enough that we'd block if buggy
        await manager.start()
        # stop() must return promptly even with no connections.
        await asyncio.wait_for(manager.stop(), timeout=1.0)
        assert manager._ping_task is None or manager._ping_task.done()
