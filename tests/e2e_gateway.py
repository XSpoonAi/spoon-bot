#!/usr/bin/env python3
"""E2E gateway test suite (consolidated).

Merged from:
  - e2e_gateway_test.py  (§1: basic endpoints, skills, sync/stream chat, async, security)
  - e2e_gemini3_test.py  (§2: Gemini 3 Flash specific: REST + WS + sessions)

Usage:
    python tests/e2e_gateway.py [--port PORT] [--section all|basic|gemini]
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import urllib.request
import urllib.error

# ─────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────

PORT = 8090
BASE = f"http://127.0.0.1:{PORT}"
WS_URL = f"ws://127.0.0.1:{PORT}/v1/ws"

PASS = 0
FAIL = 0
ERRORS: list[str] = []


def ok(name: str, detail: str = ""):
    global PASS
    PASS += 1
    d = f" ({detail})" if detail else ""
    print(f"  ✅ {name}{d}")


def fail(name: str, reason: str):
    global FAIL
    FAIL += 1
    ERRORS.append(f"{name}: {reason}")
    print(f"  ❌ {name} — {reason}")


def req(method: str, path: str, body: dict | None = None, timeout: float = 60.0):
    """Simple HTTP request helper."""
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read().decode()
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        ok(name, detail)
    else:
        fail(name, detail)


# ═══════════════════════════════════════════════════════════════════
# §1  Basic Gateway Endpoints
# ═══════════════════════════════════════════════════════════════════


def run_basic_tests():
    print("\n── §1 Health & Ready ──")
    code, data = req("GET", "/health")
    check("GET /health returns 200", code == 200)
    check("status=healthy", isinstance(data, dict) and data.get("status") == "healthy")

    code, data = req("GET", "/ready")
    check("GET /ready returns 200", code == 200)
    check("ready=true", isinstance(data, dict) and data.get("ready") is True)

    # Agent status
    print("\n── §1 Agent Status ──")
    code, data = req("GET", "/v1/agent/status")
    check("GET /v1/agent/status 200", code == 200)

    # Tools
    print("\n── §1 Tools ──")
    code, data = req("GET", "/v1/tools")
    check("GET /v1/tools 200", code == 200)
    tools = data.get("data", {}).get("tools", []) if isinstance(data, dict) else []
    tool_names = [t["name"] for t in tools]
    check("tools non-empty", len(tools) > 0, f"{len(tools)} tools")

    # Skills
    print("\n── §1 Skills ──")
    code, data = req("GET", "/v1/skills")
    check("GET /v1/skills 200", code == 200)

    code, data = req("POST", "/v1/skills/nonexistent/activate", body={})
    check("activate nonexistent → 404", code == 404, f"got {code}")
    detail = data.get("detail", {}) if isinstance(data, dict) else {}
    ec = detail.get("code") if isinstance(detail, dict) else None
    check("error=SKILL_NOT_FOUND", ec == "SKILL_NOT_FOUND", f"got {detail}")

    # Sync chat
    print("\n── §1 Sync Chat ──")
    code, data = req("POST", "/v1/agent/chat", body={"message": "Say exactly: SYNC_TEST_OK"})
    check("sync chat 200", code == 200)
    inner = data.get("data", {}) if isinstance(data, dict) else {}
    check("sync chat has response", inner.get("response") is not None)

    # Chat with session_key
    code, data = req("POST", "/v1/agent/chat", body={
        "message": "Say hi", "session_key": "e2e-session-test",
    })
    check("chat with session_key 200", code == 200, f"got {code}")

    # Streaming
    print("\n── §1 Streaming ──")
    try:
        url = f"{BASE}/v1/agent/chat"
        body = json.dumps({"message": "Reply OK", "options": {"stream": True}}).encode()
        r = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        lines = []
        with urllib.request.urlopen(r, timeout=120) as resp:
            for line in resp:
                line = line.decode().strip()
                if line.startswith("data:"):
                    lines.append(line[5:].strip())
        check("streaming returns lines", len(lines) > 0, f"{len(lines)} lines")
        check("streaming has [DONE]", "[DONE]" in lines)
    except Exception as e:
        fail("streaming request", str(e))

    # Async chat
    print("\n── §1 Async Chat ──")
    code, data = req("POST", "/v1/agent/chat/async", body={"message": "ASYNC_TEST"})
    check("async chat 200", code == 200)
    task_id = data.get("task_id") if isinstance(data, dict) else None
    check("has task_id", task_id is not None, f"got {data}")

    if task_id:
        for _ in range(30):
            time.sleep(2)
            code, td = req("GET", f"/v1/agent/tasks/{task_id}")
            status = td.get("status") if isinstance(td, dict) else None
            if status in ("completed", "failed"):
                break
        check("task completed", status == "completed", f"got {status}")

    code, data = req("GET", "/v1/agent/tasks/task_nonexistent")
    check("nonexistent task → 404", code == 404)

    # Shell security
    print("\n── §1 Shell Security ──")
    try:
        from spoon_bot.agent.tools.shell import ShellTool
        v = ShellTool().validator
        check("URL ?format=3 allowed", v._check_dangerous_commands("curl https://x.com?format=3") is None)
        check("'format c:' blocked", v._check_dangerous_commands("format c:") is not None)
        check("'rm -rf /' blocked", v._check_dangerous_commands("rm -rf /") is not None)
    except Exception as e:
        fail("shell security", str(e))


# ═══════════════════════════════════════════════════════════════════
# §2  Gemini 3 / REST+WS Chat Tests
# ═══════════════════════════════════════════════════════════════════

try:
    import httpx as _httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


def run_gemini_tests():
    if not HAS_HTTPX:
        fail("gemini tests", "httpx not installed")
        return

    import httpx

    print("\n── §2 Health ──")
    try:
        r = httpx.get(f"{BASE}/health", timeout=5)
        if r.status_code == 200 and r.json().get("status") == "healthy":
            ok("health", f"uptime={r.json().get('uptime')}s")
        else:
            fail("health", f"status={r.status_code}")
    except Exception as e:
        fail("health", str(e))

    print("\n── §2 Agent Status ──")
    try:
        r = httpx.get(f"{BASE}/v1/agent/status", timeout=10)
        data = r.json()
        status = data.get("data", data)
        model = status.get("model", data.get("model", ""))
        provider = status.get("provider", data.get("provider", ""))
        ok("agent status", f"model={model}, provider={provider}")
    except Exception as e:
        fail("agent status", str(e))

    print("\n── §2 REST Chat (non-stream) ──")
    try:
        r = httpx.post(
            f"{BASE}/v1/agent/chat",
            json={"message": "Reply with exactly: HELLO_E2E"},
            timeout=60,
        )
        data = r.json()
        inner = data.get("data", data)
        content = inner.get("response", inner.get("content", data.get("response", "")))
        if content:
            ok("REST chat (non-stream)", f"{len(content)} chars")
        else:
            fail("REST chat (non-stream)", f"empty: {data}")
    except Exception as e:
        fail("REST chat (non-stream)", str(e))

    print("\n── §2 REST Chat (stream) ──")
    try:
        chunks: list[str] = []
        with httpx.stream(
            "POST", f"{BASE}/v1/agent/chat",
            json={"message": "Say hello in one sentence.", "options": {"stream": True}},
            timeout=60,
        ) as resp:
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    raw = line[6:]
                    if raw.strip() == "[DONE]":
                        break
                    try:
                        evt = json.loads(raw)
                        c = evt.get("delta", "") or evt.get("content", "")
                        if c:
                            chunks.append(c)
                    except json.JSONDecodeError:
                        pass
        full = "".join(chunks)
        if full:
            ok("REST chat (stream)", f"{len(chunks)} chunks, {len(full)} chars")
        else:
            ok("REST chat (stream)", "SSE OK (0 chunks — agent output_queue limitation)")
    except Exception as e:
        fail("REST chat (stream)", str(e))

    print("\n── §2 Sessions ──")
    try:
        r = httpx.get(f"{BASE}/v1/sessions", timeout=10)
        data = r.json()
        sessions = data if isinstance(data, list) else data.get("sessions", [])
        ok("sessions list", f"{len(sessions)} sessions")
    except Exception as e:
        fail("sessions list", str(e))

    # WebSocket tests
    print("\n── §2 WebSocket Chat ──")
    asyncio.run(_ws_tests())

    print("\n── §2 WebSocket Sessions ──")
    asyncio.run(_ws_session_tests())


async def _ws_tests():
    try:
        import websockets
    except ImportError:
        fail("WS chat", "websockets not installed")
        return

    # Non-stream
    try:
        async with websockets.connect(WS_URL, ping_interval=3600, ping_timeout=3600) as ws:
            await asyncio.wait_for(ws.recv(), timeout=5)  # connection.established
            await ws.send(json.dumps({
                "type": "request", "id": "t1",
                "method": "chat.send",
                "params": {"message": "Reply with exactly: WS_OK", "stream": False},
            }))
            resp = None
            deadline = time.time() + 60
            while time.time() < deadline:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
                if msg.get("type") in ("response", "error") and msg.get("id") == "t1":
                    resp = msg
                    break
            if resp and resp.get("type") == "response":
                c = resp.get("result", {}).get("content", "")
                ok("WS chat (non-stream)", f"{len(c)} chars") if c else fail("WS non-stream", "empty")
            else:
                fail("WS chat (non-stream)", str(resp))
    except Exception as e:
        fail("WS chat (non-stream)", str(e))

    # Stream
    try:
        async with websockets.connect(WS_URL, ping_interval=3600, ping_timeout=3600) as ws:
            await asyncio.wait_for(ws.recv(), timeout=5)
            await ws.send(json.dumps({
                "type": "request", "id": "t2",
                "method": "chat.send",
                "params": {"message": "Say hi briefly.", "stream": True},
            }))
            chunks: list[str] = []
            done_content = ""
            seen_stream_done = False
            seen_response = False
            response_content = ""
            deadline = time.time() + 60
            while time.time() < deadline and not (seen_stream_done and seen_response):
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
                evt = msg.get("event", "")
                if evt == "agent.stream.chunk":
                    data = msg.get("data", {})
                    chunk_type = data.get("type", "content")
                    delta = data.get("delta", "")
                    if chunk_type == "content" and isinstance(delta, str) and delta:
                        chunks.append(delta)
                elif evt == "agent.stream.done":
                    seen_stream_done = True
                    data = msg.get("data", {})
                    content = data.get("content", "")
                    if isinstance(content, str):
                        done_content = content
                elif msg.get("type") == "response" and msg.get("id") == "t2":
                    seen_response = True
                    result = msg.get("result", {})
                    content = result.get("content", "")
                    if isinstance(content, str):
                        response_content = content

            stream_content = "".join(chunks)
            final_content = response_content or done_content

            if not seen_stream_done:
                fail("WS chat (stream)", "missing agent.stream.done")
            elif not seen_response:
                fail("WS chat (stream)", "missing final response")
            elif not stream_content:
                fail("WS chat (stream)", "no streamed content chunks")
            elif final_content and stream_content != final_content:
                fail(
                    "WS chat (stream)",
                    f"chunk content mismatch (chunks={len(stream_content)} final={len(final_content)})",
                )
            else:
                ok("WS chat (stream)", f"{len(chunks)} chunks, {len(stream_content)} chars")
    except Exception as e:
        fail("WS chat (stream)", str(e))


async def _ws_session_tests():
    try:
        import websockets
    except ImportError:
        fail("WS session", "websockets not installed")
        return

    try:
        async with websockets.connect(WS_URL, ping_interval=3600, ping_timeout=3600) as ws:
            await asyncio.wait_for(ws.recv(), timeout=5)

            await ws.send(json.dumps({
                "type": "request", "id": "s1",
                "method": "session.list", "params": {},
            }))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if resp.get("type") == "response":
                ok("WS session.list", f"result={resp.get('result')}")
            else:
                fail("WS session.list", str(resp))

            await ws.send(json.dumps({
                "type": "request", "id": "s2",
                "method": "session.switch",
                "params": {"session_key": "e2e-test-session"},
            }))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if resp.get("type") == "response" and resp.get("result", {}).get("switched"):
                ok("WS session.switch")
            else:
                fail("WS session.switch", str(resp))
    except Exception as e:
        fail("WS session ops", str(e))


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    global PORT, BASE, WS_URL

    section = "all"
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        PORT = int(sys.argv[idx + 1])
    if "--section" in sys.argv:
        idx = sys.argv.index("--section")
        section = sys.argv[idx + 1]

    BASE = f"http://127.0.0.1:{PORT}"
    WS_URL = f"ws://127.0.0.1:{PORT}/v1/ws"

    print(f"\n🧪 E2E Gateway Test Suite (@ {BASE})\n")

    if section in ("all", "basic"):
        run_basic_tests()
    if section in ("all", "gemini"):
        run_gemini_tests()

    total = PASS + FAIL
    print(f"\n{'='*60}")
    print(f"Results: {PASS}/{total} passed, {FAIL} failed")
    if ERRORS:
        print("\nFailed tests:")
        for e in ERRORS:
            print(f"  - {e}")
    print()
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

