"""Live WebSocket integration tests for P0-1/P0-2 gateway features.

This script connects to a running spoon-bot gateway server and verifies:
  1. Connection establishment and auth-free access
  2. trace_id propagation in all WS events (P0-1.5)
  3. timing payload in agent.complete event (P0-1.5)
  4. Cancellation propagation via agent.cancel (P0-2.5)
  5. trace_id uniqueness across different chat requests (P0-1.5)
  6. Non-streaming chat with trace_id + timing (P0-1.5)
  7. Streaming chat with trace_id in chunk events (P0-1.5)

Usage:
    python tests/ws_live_test.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import traceback
from uuid import uuid4

import websockets


WS_URL = "ws://127.0.0.1:8080/v1/ws"
TIMEOUT = 60  # seconds per test


class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error: str | None = None
        self.details: dict = {}

    def ok(self, **details):
        self.passed = True
        self.details = details

    def fail(self, msg: str, **details):
        self.passed = False
        self.error = msg
        self.details = details


def make_request(method: str, params: dict | None = None) -> dict:
    return {
        "type": "request",
        "id": f"req_{uuid4().hex[:8]}",
        "method": method,
        "params": params or {},
    }


async def recv_until(ws, predicate, *, timeout: float = TIMEOUT) -> list[dict]:
    """Receive messages until predicate returns True or timeout."""
    messages = []
    try:
        async with asyncio.timeout(timeout):
            while True:
                raw = await ws.recv()
                msg = json.loads(raw)
                messages.append(msg)
                if predicate(msg):
                    break
    except TimeoutError:
        pass
    return messages


async def recv_response(ws, request_id: str, *, timeout: float = TIMEOUT) -> tuple[dict | None, list[dict]]:
    """Receive events + final response for a given request ID."""
    events = []
    response = None
    try:
        async with asyncio.timeout(timeout):
            while True:
                raw = await ws.recv()
                msg = json.loads(raw)
                if msg.get("type") == "response" and msg.get("id") == request_id:
                    response = msg
                    break
                elif msg.get("type") == "error" and msg.get("id") == request_id:
                    response = msg
                    break
                else:
                    events.append(msg)
    except TimeoutError:
        pass
    return response, events


# ──────────────────────────────────────────────
# Test 1: Connection establishment
# ──────────────────────────────────────────────
async def test_connection(results: list[TestResult]):
    t = TestResult("1. Connection establishment")
    results.append(t)
    try:
        async with websockets.connect(WS_URL) as ws:
            # First message should be connection.established
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            msg = json.loads(raw)
            if msg.get("event") == "connection.established":
                conn_id = msg.get("data", {}).get("connection_id")
                t.ok(connection_id=conn_id)
                print(f"  [PASS] Connection established (conn_id={conn_id})")
            else:
                t.fail(f"Unexpected first message: {msg}")
                print(f"  [FAIL] Expected connection.established, got: {msg.get('event')}")
    except Exception as e:
        t.fail(str(e))
        print(f"  [FAIL] {e}")


# ──────────────────────────────────────────────
# Test 2: Non-streaming chat — trace_id + timing
# ──────────────────────────────────────────────
async def test_nonstream_trace_timing(results: list[TestResult]):
    t = TestResult("2. Non-streaming chat: trace_id + timing in response")
    results.append(t)
    try:
        async with websockets.connect(WS_URL) as ws:
            # Skip connection.established
            await asyncio.wait_for(ws.recv(), timeout=10)

            req = make_request("agent.chat", {
                "message": "Say 'hello' in exactly one word.",
                "stream": False,
            })
            await ws.send(json.dumps(req))

            response, events = await recv_response(ws, req["id"], timeout=TIMEOUT)

            if response is None:
                t.fail("No response received (timeout)")
                print(f"  [FAIL] No response received. Events: {len(events)}")
                return

            if response.get("type") == "error":
                t.fail(f"Error response: {response}")
                print(f"  [FAIL] Error: {response}")
                return

            result = response.get("result", {})
            trace_id = result.get("trace_id")
            timing = result.get("timing")

            # Check trace_id
            has_trace = trace_id is not None and trace_id.startswith("trc_")
            # Check timing
            has_timing = timing is not None and "total_elapsed_ms" in (timing or {})

            # Check events for trace_id
            thinking_events = [e for e in events if e.get("event") == "agent.thinking"]
            complete_events = [e for e in events if e.get("event") == "agent.complete"]

            thinking_has_trace = all(
                e.get("data", {}).get("trace_id", "").startswith("trc_")
                for e in thinking_events
            ) if thinking_events else True

            complete_has_trace = all(
                e.get("data", {}).get("trace_id", "").startswith("trc_")
                for e in complete_events
            ) if complete_events else True

            complete_has_timing = all(
                "timing" in e.get("data", {})
                for e in complete_events
            ) if complete_events else True

            if has_trace and has_timing and thinking_has_trace and complete_has_trace and complete_has_timing:
                t.ok(
                    trace_id=trace_id,
                    timing_elapsed_ms=timing.get("total_elapsed_ms"),
                    events_count=len(events),
                    thinking_events=len(thinking_events),
                    complete_events=len(complete_events),
                )
                print(f"  [PASS] trace_id={trace_id}, timing={timing.get('total_elapsed_ms')}ms, events={len(events)}")
            else:
                t.fail(
                    f"Missing fields: trace={has_trace}, timing={has_timing}, "
                    f"thinking_trace={thinking_has_trace}, complete_trace={complete_has_trace}, "
                    f"complete_timing={complete_has_timing}"
                )
                print(f"  [FAIL] trace={has_trace}, timing={has_timing}")
                print(f"         Result keys: {list(result.keys())}")
                if complete_events:
                    print(f"         Complete data keys: {list(complete_events[0].get('data', {}).keys())}")

    except Exception as e:
        t.fail(str(e))
        print(f"  [FAIL] {e}")
        traceback.print_exc()


# ──────────────────────────────────────────────
# Test 3: Streaming chat — trace_id in chunk events
# ──────────────────────────────────────────────
async def test_streaming_trace(results: list[TestResult]):
    t = TestResult("3. Streaming chat: trace_id in chunk events")
    results.append(t)
    try:
        async with websockets.connect(WS_URL) as ws:
            await asyncio.wait_for(ws.recv(), timeout=10)

            req = make_request("agent.chat", {
                "message": "Count from 1 to 3, one number per line.",
                "stream": True,
            })
            await ws.send(json.dumps(req))

            response, events = await recv_response(ws, req["id"], timeout=TIMEOUT)

            if response is None:
                t.fail("No response received (timeout)")
                print(f"  [FAIL] No response. Events collected: {len(events)}")
                return

            if response.get("type") == "error":
                t.fail(f"Error: {response}")
                print(f"  [FAIL] Error: {response}")
                return

            # Categorize events
            chunk_events = [e for e in events if e.get("event") == "agent.stream.chunk"]
            done_events = [e for e in events if e.get("event") == "agent.stream.done"]
            complete_events = [e for e in events if e.get("event") == "agent.complete"]
            thinking_events = [e for e in events if e.get("event") == "agent.thinking"]

            # Check trace_id in all event types
            all_traced = True
            untrace_count = 0
            for e in chunk_events + done_events + complete_events + thinking_events:
                tid = e.get("data", {}).get("trace_id")
                if not tid or not tid.startswith("trc_"):
                    all_traced = False
                    untrace_count += 1

            # Check timing in complete event
            complete_has_timing = all(
                "timing" in e.get("data", {})
                for e in complete_events
            ) if complete_events else False

            result = response.get("result", {})
            trace_id = result.get("trace_id")

            if all_traced and complete_has_timing and trace_id:
                t.ok(
                    trace_id=trace_id,
                    chunk_events=len(chunk_events),
                    done_events=len(done_events),
                    complete_events=len(complete_events),
                )
                print(
                    f"  [PASS] trace_id={trace_id}, chunks={len(chunk_events)}, "
                    f"done={len(done_events)}, complete={len(complete_events)}"
                )
            else:
                t.fail(
                    f"all_traced={all_traced} (untraced={untrace_count}), "
                    f"complete_timing={complete_has_timing}, trace_id={trace_id}"
                )
                print(f"  [FAIL] all_traced={all_traced}, complete_timing={complete_has_timing}")

    except Exception as e:
        t.fail(str(e))
        print(f"  [FAIL] {e}")
        traceback.print_exc()


# ──────────────────────────────────────────────
# Test 4: Cancellation propagation
# ──────────────────────────────────────────────
async def test_cancellation(results: list[TestResult]):
    t = TestResult("4. Cancellation propagation via agent.cancel")
    results.append(t)
    try:
        async with websockets.connect(WS_URL) as ws:
            await asyncio.wait_for(ws.recv(), timeout=10)

            # Start a streaming request that should take a while
            chat_req = make_request("agent.chat", {
                "message": "Write a detailed 500-word essay about the history of computing.",
                "stream": True,
            })
            await ws.send(json.dumps(chat_req))

            # Wait for a few chunk events
            collected = []
            got_chunks = False
            try:
                async with asyncio.timeout(15):
                    while True:
                        raw = await ws.recv()
                        msg = json.loads(raw)
                        collected.append(msg)
                        if msg.get("event") == "agent.stream.chunk":
                            got_chunks = True
                            break
            except TimeoutError:
                pass

            if not got_chunks:
                # Even without chunks, send cancel
                pass

            # Send cancel
            cancel_req = make_request("agent.cancel", {})
            await ws.send(json.dumps(cancel_req))

            # Collect remaining messages (should end quickly)
            cancel_response = None
            remaining = []
            try:
                async with asyncio.timeout(10):
                    while True:
                        raw = await ws.recv()
                        msg = json.loads(raw)
                        remaining.append(msg)
                        if msg.get("type") == "response" and msg.get("id") == cancel_req["id"]:
                            cancel_response = msg
                        # Check if we get the chat response too (it should come back)
                        if msg.get("type") == "response" and msg.get("id") == chat_req["id"]:
                            break
                        if msg.get("type") == "error" and msg.get("id") == chat_req["id"]:
                            break
            except TimeoutError:
                pass

            cancel_result = cancel_response.get("result", {}) if cancel_response else {}
            cancelled = cancel_result.get("cancelled", False)

            if cancelled or cancel_response:
                t.ok(
                    cancelled=cancelled,
                    chunks_before_cancel=len([e for e in collected if e.get("event") == "agent.stream.chunk"]),
                    remaining_events=len(remaining),
                )
                print(f"  [PASS] Cancel acknowledged. cancelled={cancelled}, events_before={len(collected)}")
            else:
                t.fail("Cancel not acknowledged")
                print(f"  [FAIL] No cancel response. collected={len(collected)}, remaining={len(remaining)}")

    except Exception as e:
        t.fail(str(e))
        print(f"  [FAIL] {e}")
        traceback.print_exc()


# ──────────────────────────────────────────────
# Test 5: Trace ID uniqueness across requests
# ──────────────────────────────────────────────
async def test_trace_uniqueness(results: list[TestResult]):
    t = TestResult("5. Trace ID uniqueness across requests")
    results.append(t)
    try:
        trace_ids = []

        async with websockets.connect(WS_URL) as ws:
            await asyncio.wait_for(ws.recv(), timeout=10)

            for i in range(3):
                req = make_request("agent.chat", {
                    "message": f"Reply with the number {i + 1}.",
                    "stream": False,
                })
                await ws.send(json.dumps(req))
                response, events = await recv_response(ws, req["id"], timeout=TIMEOUT)

                if response and response.get("type") == "response":
                    tid = response.get("result", {}).get("trace_id")
                    if tid:
                        trace_ids.append(tid)

            unique_count = len(set(trace_ids))
            all_unique = unique_count == len(trace_ids) and len(trace_ids) == 3
            all_prefixed = all(t.startswith("trc_") for t in trace_ids)

            if all_unique and all_prefixed:
                t.ok(trace_ids=trace_ids)
                print(f"  [PASS] 3 unique trace IDs: {trace_ids}")
            else:
                t.fail(f"Not unique or not properly prefixed. IDs: {trace_ids}")
                print(f"  [FAIL] trace_ids={trace_ids}, unique={unique_count}")

    except Exception as e:
        t.fail(str(e))
        print(f"  [FAIL] {e}")
        traceback.print_exc()


# ──────────────────────────────────────────────
# Test 6: Trace ID consistency within a single request
# ──────────────────────────────────────────────
async def test_trace_consistency(results: list[TestResult]):
    t = TestResult("6. Trace ID consistency within a single streaming request")
    results.append(t)
    try:
        async with websockets.connect(WS_URL) as ws:
            await asyncio.wait_for(ws.recv(), timeout=10)

            req = make_request("agent.chat", {
                "message": "List three colors.",
                "stream": True,
            })
            await ws.send(json.dumps(req))
            response, events = await recv_response(ws, req["id"], timeout=TIMEOUT)

            if response is None or response.get("type") == "error":
                t.fail(f"No valid response: {response}")
                print(f"  [FAIL] {response}")
                return

            # Collect all trace_ids from events
            event_trace_ids = set()
            for e in events:
                tid = e.get("data", {}).get("trace_id")
                if tid:
                    event_trace_ids.add(tid)

            result_trace_id = response.get("result", {}).get("trace_id")
            if result_trace_id:
                event_trace_ids.add(result_trace_id)

            if len(event_trace_ids) == 1:
                the_id = event_trace_ids.pop()
                t.ok(trace_id=the_id, events_checked=len(events))
                print(f"  [PASS] All events share trace_id={the_id} ({len(events)} events)")
            elif len(event_trace_ids) == 0:
                t.fail("No trace_ids found in any events")
                print(f"  [FAIL] No trace_ids found. Events: {len(events)}")
            else:
                t.fail(f"Inconsistent trace_ids: {event_trace_ids}")
                print(f"  [FAIL] Multiple trace_ids: {event_trace_ids}")

    except Exception as e:
        t.fail(str(e))
        print(f"  [FAIL] {e}")
        traceback.print_exc()


# ──────────────────────────────────────────────
# Test 7: Ping/pong
# ──────────────────────────────────────────────
async def test_ping_pong(results: list[TestResult]):
    t = TestResult("7. Ping/pong heartbeat")
    results.append(t)
    try:
        async with websockets.connect(WS_URL) as ws:
            await asyncio.wait_for(ws.recv(), timeout=10)

            ping_msg = {"type": "ping", "timestamp": "2026-02-11T00:00:00Z"}
            await ws.send(json.dumps(ping_msg))

            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            msg = json.loads(raw)

            if msg.get("type") == "pong":
                t.ok(timestamp=msg.get("timestamp"))
                print(f"  [PASS] Pong received: {msg}")
            else:
                t.fail(f"Expected pong, got: {msg}")
                print(f"  [FAIL] Expected pong, got: {msg}")

    except Exception as e:
        t.fail(str(e))
        print(f"  [FAIL] {e}")


# ──────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────
async def main():
    print("=" * 60)
    print("P0-1/P0-2 WebSocket Live Integration Tests")
    print(f"Server: {WS_URL}")
    print("=" * 60)
    print()

    results: list[TestResult] = []

    tests = [
        ("Connection", test_connection),
        ("Ping/Pong", test_ping_pong),
        ("Non-streaming trace+timing", test_nonstream_trace_timing),
        ("Streaming trace", test_streaming_trace),
        ("Trace uniqueness", test_trace_uniqueness),
        ("Trace consistency", test_trace_consistency),
        ("Cancellation", test_cancellation),
    ]

    for label, test_fn in tests:
        print(f"\n--- {label} ---")
        await test_fn(results)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        extra = ""
        if r.error:
            extra = f" — {r.error}"
        print(f"  [{status}] {r.name}{extra}")

    print(f"\n  Total: {passed}/{len(results)} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
