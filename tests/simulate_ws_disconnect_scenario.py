"""Reproduce the `send_message failed → connection closed` user scenario.

Simulates what happens when the sandbox proxy tears down a WebSocket while
the agent is streaming a long response back to the client. Runs two variants
of ``send_message`` — the **old** retry-on-every-exception implementation and
the **new** disconnect-aware implementation — against the same event trace so
you can compare wall-clock time, log noise, and final connection state.

Run with::

    python tests/simulate_ws_disconnect_scenario.py

Nothing here imports pytest; it's a self-contained reproducer meant to be
exercised by hand (or in CI smoke jobs) and to document the behaviour change.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

# Ensure the worktree root is on sys.path so we import the PR's spoon_bot,
# not whichever copy the shared venv was editable-installed against.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import WebSocketDisconnect
from loguru import logger
from starlette.websockets import WebSocketState

from spoon_bot.gateway.websocket.manager import Connection, ConnectionManager


# ---------------------------------------------------------------------------
# Fake WebSocket that drops after the Nth send — models sandbox idle timeout.
# ---------------------------------------------------------------------------


class FlakyWebSocket:
    """Accepts N sends, then behaves like a peer that already closed."""

    def __init__(self, *, sends_before_drop: int) -> None:
        self._sends_before_drop = sends_before_drop
        self.sends = 0
        self.client_state = WebSocketState.CONNECTED
        self.close_calls = 0

    async def accept(self) -> None:
        return None

    async def close(self) -> None:
        self.close_calls += 1
        self.client_state = WebSocketState.DISCONNECTED

    async def send_json(self, _payload: Any) -> None:
        self.sends += 1
        if self.sends > self._sends_before_drop:
            # Flip state BEFORE raising, mirroring what Starlette does once
            # the peer's close frame has been processed.
            self.client_state = WebSocketState.DISCONNECTED
            raise WebSocketDisconnect(code=1005)


# ---------------------------------------------------------------------------
# Minimal Connection / ConnectionManager pair so we can plug in either the
# OLD or NEW send_message implementation for side-by-side comparison.
# ---------------------------------------------------------------------------


class _OldManager:
    """Inline re-implementation of the pre-PR send_message for the A/B run."""

    def __init__(self) -> None:
        self._connections: dict[str, Connection] = {}

    async def connect(self, ws: FlakyWebSocket) -> str:
        await ws.accept()
        conn_id = str(uuid4())
        self._connections[conn_id] = Connection(
            id=conn_id,
            websocket=ws,
            user_id="sim-user",
            session_key="sim-session",
        )
        return conn_id

    async def disconnect(self, conn_id: str) -> None:
        conn = self._connections.pop(conn_id, None)
        if conn:
            try:
                await conn.websocket.close()
            except Exception:
                pass

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    async def send(self, conn_id: str, data: dict) -> bool:
        conn = self._connections.get(conn_id)
        if not conn:
            return False
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                async with conn.send_lock:
                    await conn.websocket.send_json(data)
                conn.update_activity()
                return True
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        f"send_message to {conn_id} failed (attempt "
                        f"{attempt + 1}/{max_retries + 1}): {e}"
                    )
                    await asyncio.sleep(0.1 * (attempt + 1))
                    if conn_id not in self._connections:
                        return False
                    continue
                logger.error(
                    f"Failed to send message to {conn_id} after "
                    f"{max_retries + 1} attempts: {e}"
                )
                await self.disconnect(conn_id)
                return False
        return False


class _NewManager:
    """Thin facade over the real, post-PR ``ConnectionManager.send_message``."""

    def __init__(self) -> None:
        self._real = ConnectionManager()

    async def connect(self, ws: FlakyWebSocket) -> str:
        return await self._real.connect(ws, user_id="sim-user", session_key="sim-session")

    async def disconnect(self, conn_id: str) -> None:
        await self._real.disconnect(conn_id)

    @property
    def connection_count(self) -> int:
        return self._real.connection_count

    async def send(self, conn_id: str, data: dict) -> bool:
        return await self._real.send_message(conn_id, data)


def _build_manager(variant: str):
    assert variant in {"old", "new"}
    return _OldManager() if variant == "old" else _NewManager()


# ---------------------------------------------------------------------------
# The scenario: agent streams 20 chunks; sandbox kills the WS after chunk #3.
# ---------------------------------------------------------------------------


async def run_scenario(variant: str) -> dict[str, Any]:
    manager = _build_manager(variant)
    ws = FlakyWebSocket(sends_before_drop=3)
    conn_id = await manager.connect(ws)

    log_records: list[tuple[str, str]] = []

    def _sink(msg: Any) -> None:
        rec = msg.record
        log_records.append((rec["level"].name, rec["message"]))

    sink_id = logger.add(_sink, level="DEBUG")
    try:
        start = time.monotonic()
        results: list[bool] = []
        for i in range(20):
            ok = await manager.send(
                conn_id,
                {"type": "stream.chunk", "i": i, "delta": f"chunk {i}"},
            )
            results.append(ok)
        elapsed = time.monotonic() - start
    finally:
        try:
            logger.remove(sink_id)
        except ValueError:
            # loguru may have re-initialised between scenarios; safe to ignore.
            pass

    warnings = [m for lvl, m in log_records if lvl == "WARNING"]
    errors = [m for lvl, m in log_records if lvl == "ERROR"]
    debugs = [m for lvl, m in log_records if lvl == "DEBUG"]

    return {
        "variant": variant,
        "wall_clock_seconds": round(elapsed, 3),
        "chunk_success": sum(1 for ok in results if ok),
        "chunk_fail": sum(1 for ok in results if not ok),
        "send_json_calls": ws.sends,
        "retry_warning_count": sum(1 for w in warnings if "failed (attempt" in w),
        "escalated_error_count": sum(1 for e in errors if "after 3 attempts" in e),
        "debug_count": len(debugs),
        "connection_still_registered": manager.connection_count,
        "close_calls": ws.close_calls,
        "sample_warning": warnings[0] if warnings else "",
        "sample_error": errors[0] if errors else "",
    }


def print_report(title: str, stats: dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    for key in [
        "wall_clock_seconds",
        "chunk_success",
        "chunk_fail",
        "send_json_calls",
        "retry_warning_count",
        "escalated_error_count",
        "debug_count",
        "connection_still_registered",
        "close_calls",
    ]:
        print(f"  {key:<32} {stats[key]}")
    if stats.get("sample_warning"):
        print(f"  sample_warning                  {stats['sample_warning']!r}")
    if stats.get("sample_error"):
        print(f"  sample_error                    {stats['sample_error']!r}")


async def main() -> int:
    logger.remove()  # start from a clean handler set
    old_stats = await run_scenario("old")
    new_stats = await run_scenario("new")

    # Reattach stderr sink so the final verdict is visible even without -s.
    logger.add(sys.stderr, level="INFO")

    print_report("OLD behaviour (pre-fix)", old_stats)
    print_report("NEW behaviour (this PR)", new_stats)

    print("\n=== delta ===")
    print(
        f"  retry WARNINGs cut by:           "
        f"{old_stats['retry_warning_count']} -> {new_stats['retry_warning_count']}"
    )
    print(
        f"  escalated ERRORs cut by:         "
        f"{old_stats['escalated_error_count']} -> {new_stats['escalated_error_count']}"
    )
    print(
        f"  wall-clock speedup on a dead WS: "
        f"{old_stats['wall_clock_seconds']}s -> {new_stats['wall_clock_seconds']}s"
    )
    print(
        f"  doomed send_json calls cut by:   "
        f"{old_stats['send_json_calls']} -> {new_stats['send_json_calls']}"
    )
    print(
        f"  dead conn self-cleanup?          "
        f"old leaves {old_stats['connection_still_registered']} registered, "
        f"new leaves {new_stats['connection_still_registered']}"
    )

    # Assert the fix actually changed the right things — gives a non-zero
    # exit code if someone reverts part of this PR by accident.
    assert new_stats["retry_warning_count"] == 0, "new path should emit zero retry WARNINGs"
    assert new_stats["escalated_error_count"] == 0, "new path should emit zero terminal ERRORs"
    assert new_stats["connection_still_registered"] == 0, "new path should self-clean dead conn"
    assert new_stats["send_json_calls"] <= old_stats["send_json_calls"], (
        "new path should not call send_json more than old"
    )
    assert new_stats["wall_clock_seconds"] <= old_stats["wall_clock_seconds"], (
        "new path should not be slower"
    )

    print("\nOK: scenario reproduces the fix (retry spam gone, dead conn cleaned, faster).")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
