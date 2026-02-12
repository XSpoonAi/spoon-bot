#!/usr/bin/env python3
"""
End-to-end gateway test script for spoon-bot.

Tests all API endpoints against a running gateway instance.
Start the gateway first:
    python -m uvicorn spoon_bot.gateway.server:create_app --factory --host 127.0.0.1 --port 8080
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error

# Ensure spoon_bot is importable when running the test script directly
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

BASE = "http://127.0.0.1:9090"
PASS = 0
FAIL = 0
RESULTS = []


def req(method: str, path: str, body: dict | None = None, timeout: float = 60.0) -> tuple[int, dict | str]:
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


def sse_req(path: str, body: dict, timeout: float = 60.0) -> list[str]:
    """Send SSE request and collect data lines."""
    url = f"{BASE}{path}"
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    r = urllib.request.Request(url, data=data, headers=headers, method="POST")
    lines = []
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        for line in resp:
            line = line.decode().strip()
            if line.startswith("data:"):
                lines.append(line[5:].strip())
    return lines


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
    msg = f"  [{status}] {name}"
    if detail and not condition:
        msg += f" — {detail}"
    print(msg)
    RESULTS.append((name, condition))


def main():
    global PASS, FAIL

    print("=" * 64)
    print("  Spoon-bot Gateway E2E Test Suite")
    print("=" * 64)

    # -------------------------------------------------------------------
    # 1. Health & Ready
    # -------------------------------------------------------------------
    print("\n--- Health & Ready ---")

    code, data = req("GET", "/health")
    check("GET /health returns 200", code == 200)
    check("Health status is healthy", isinstance(data, dict) and data.get("status") == "healthy")

    code, data = req("GET", "/ready")
    check("GET /ready returns 200", code == 200)
    check("Ready flag is true", isinstance(data, dict) and data.get("ready") is True)

    # -------------------------------------------------------------------
    # 2. Agent Status
    # -------------------------------------------------------------------
    print("\n--- Agent Status ---")

    code, data = req("GET", "/v1/agent/status")
    check("GET /v1/agent/status returns 200", code == 200)
    check("Status data has tools_available", isinstance(data, dict) and data.get("data", {}).get("stats", {}).get("tools_available", 0) > 0)

    # -------------------------------------------------------------------
    # 3. Tools
    # -------------------------------------------------------------------
    print("\n--- Tools ---")

    code, data = req("GET", "/v1/tools")
    check("GET /v1/tools returns 200", code == 200)
    tools = data.get("data", {}).get("tools", []) if isinstance(data, dict) else []
    tool_names = [t["name"] for t in tools]
    check("Tools list is non-empty", len(tools) > 0, f"got {len(tools)}")
    check("shell tool registered", "shell" in tool_names)

    # -------------------------------------------------------------------
    # 4. Skills (Bug #2)
    # -------------------------------------------------------------------
    print("\n--- Skills (Bug #2) ---")

    code, data = req("GET", "/v1/skills")
    check("GET /v1/skills returns 200", code == 200)

    code, data = req("POST", "/v1/skills/nonexistent/activate", body={})
    check("Activate nonexistent skill returns 404", code == 404, f"got {code}")
    detail = data.get("detail", {}) if isinstance(data, dict) else {}
    error_code = detail.get("code") if isinstance(detail, dict) else None
    check("Error code is SKILL_NOT_FOUND", error_code == "SKILL_NOT_FOUND", f"got {detail}")

    code, data = req("POST", "/v1/skills/nonexistent/deactivate", body={})
    check("Deactivate nonexistent skill returns 404", code == 404)

    # -------------------------------------------------------------------
    # 5. Sync Chat (Bug #4 — session_key)
    # -------------------------------------------------------------------
    print("\n--- Sync Chat (Bug #4: session_key) ---")

    code, data = req("POST", "/v1/agent/chat", body={"message": "Say exactly: SYNC_TEST_OK"})
    check("Sync chat returns 200", code == 200)
    check("Sync chat has response", isinstance(data, dict) and data.get("data", {}).get("response") is not None)

    code, data = req("POST", "/v1/agent/chat", body={
        "message": "Say hi",
        "session_key": "e2e-session-test"
    })
    check("Chat with session_key returns 200", code == 200, f"got {code}")
    check("Chat with session_key has response",
          isinstance(data, dict) and data.get("data", {}).get("response") is not None)

    # -------------------------------------------------------------------
    # 6. Streaming Chat
    # -------------------------------------------------------------------
    print("\n--- Streaming Chat ---")

    try:
        lines = sse_req("/v1/agent/chat", body={
            "message": "Reply with exactly one word: OK",
            "options": {"stream": True},
        }, timeout=120)
        check("Streaming returns data lines", len(lines) > 0, f"got {len(lines)} lines")

        has_content = any('"type":"content"' in l or '"type": "content"' in l for l in lines if l != "[DONE]")
        has_done = "[DONE]" in lines
        check("Streaming has content chunk", has_content, f"lines: {lines[:3]}")
        check("Streaming has [DONE] sentinel", has_done)
    except Exception as e:
        check("Streaming request succeeded", False, str(e))

    # -------------------------------------------------------------------
    # 7. Async Chat (Bug #7)
    # -------------------------------------------------------------------
    print("\n--- Async Chat (Bug #7) ---")

    code, data = req("POST", "/v1/agent/chat/async", body={"message": "Say exactly: ASYNC_TEST"})
    check("Async chat returns 200", code == 200)
    task_id = data.get("task_id") if isinstance(data, dict) else None
    check("Async chat returns task_id", task_id is not None, f"got {data}")

    if task_id:
        # Poll until completed or timeout
        for _ in range(30):
            time.sleep(2)
            code, task_data = req("GET", f"/v1/agent/tasks/{task_id}")
            status = task_data.get("status") if isinstance(task_data, dict) else None
            if status in ("completed", "failed"):
                break

        check("Task status is completed", status == "completed", f"got {status}")
        check("Task has result", task_data.get("result") is not None if isinstance(task_data, dict) else False)

        # Cancel completed task
        code, cancel_data = req("POST", f"/v1/agent/tasks/{task_id}/cancel")
        check("Cancel completed task returns terminal message",
              isinstance(cancel_data, dict) and cancel_data.get("cancelled") is False)

    # Task not found
    code, data = req("GET", "/v1/agent/tasks/task_nonexistent")
    check("Get nonexistent task returns 404", code == 404)

    # -------------------------------------------------------------------
    # 8. ShellTool Security (Bug #6) — unit test via import
    # -------------------------------------------------------------------
    print("\n--- ShellTool Security (Bug #6) ---")

    try:
        from spoon_bot.agent.tools.shell import ShellTool
        tool = ShellTool()
        v = tool.validator

        r_url = v._check_dangerous_commands("curl https://api.example.com/v1/weather?format=3")
        r_format = v._check_dangerous_commands("format c:")
        r_rm = v._check_dangerous_commands("rm -rf /")
        r_ok = v._check_dangerous_commands("curl https://example.com")

        check("URL with ?format=3 is allowed", r_url is None, f"got {r_url}")
        check("'format c:' is blocked", r_format is not None)
        check("'rm -rf /' is blocked", r_rm is not None)
        check("Normal curl is allowed", r_ok is None)
    except Exception as e:
        check("ShellTool security test", False, str(e))

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print(f"\n{'=' * 64}")
    print(f"  Results: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
    print(f"{'=' * 64}")

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
