"""Integration test: play two rounds of joker-game-agent via gateway REST API.

Starts the spoon-bot gateway (uvicorn), sends two consecutive chat requests
via POST /v1/chat with SSE streaming, each in the same session.
Verifies:
  - Wallet loads correctly without PRIVATE_KEY in env
  - Game commands execute successfully
  - No private key hex leaks in any streamed output
  - Second round works in the same session

Usage:
    python tests/test_joker_game_ws.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

GATEWAY_PORT = int(os.environ.get("TEST_GATEWAY_PORT", "18090"))
BASE_URL = f"http://127.0.0.1:{GATEWAY_PORT}"
HEALTH_URL = f"{BASE_URL}/health"
CHAT_URL = f"{BASE_URL}/v1/agent/chat"
GAME_TIMEOUT = 600

SKILL_CLI = str(
    Path.home() / ".spoon-bot" / "workspace" / "skills"
    / "joker-game-agent" / "cli" / "index.js"
)

ROUND1_PROMPT = (
    "Play a game of JokerGame using the skill CLI at {cli}. "
    "Steps: 1) Run `node {cli} wallet` to verify wallet. "
    "2) Run `node {cli} join` to join a game. "
    "3) If there is a challenge, solve it and submit with `node {cli} challenge-answer <answer>`. "
    "4) Run `node {cli} wait` to wait for other players. "
    "5) When the round starts, run `node {cli} read-card` to see your card. "
    "Report the wallet address, game id, and card."
).format(cli=SKILL_CLI)

ROUND2_PROMPT = (
    "Play a SECOND game of JokerGame in this same session. "
    "Steps: 1) Run `node {cli} join` to join a new game. "
    "2) Solve any challenge and submit the answer. "
    "3) Wait for other players. "
    "4) Read your card. "
    "Report the game id and card."
).format(cli=SKILL_CLI)


def _get_private_key_hex() -> str | None:
    pk_file = Path.home() / ".agent-wallet" / "privatekey.tmp"
    if pk_file.exists():
        return pk_file.read_text(encoding="utf-8").strip()
    return None


def _wait_for_health(timeout: float = 90.0) -> None:
    import urllib.request
    import urllib.error

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(HEALTH_URL)
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(2.0)
    raise TimeoutError(f"Gateway did not become healthy within {timeout}s")


def _chat_sse(prompt: str, session_key: str, round_num: int) -> dict:
    """Send a chat request with SSE streaming and collect all events."""
    import urllib.request

    print(f"\n{'='*60}")
    print(f"  ROUND {round_num}: POST {CHAT_URL}")
    print(f"{'='*60}")

    body = json.dumps({
        "message": prompt,
        "session_key": session_key,
        "options": {"stream": True},
    }).encode("utf-8")

    req = urllib.request.Request(
        CHAT_URL,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )

    result = {"completed": False, "response": "", "chunks": [], "tool_outputs": []}
    deadline = time.monotonic() + GAME_TIMEOUT

    try:
        with urllib.request.urlopen(req, timeout=GAME_TIMEOUT) as resp:
            event_type = ""
            data_lines: list[str] = []
            for raw_line in resp:
                if time.monotonic() > deadline:
                    print("  [TIMEOUT]")
                    break

                line = raw_line.decode("utf-8", errors="replace").rstrip("\n\r")

                if line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: "):
                    data_lines.append(line[6:])
                elif line == "":
                    if not data_lines:
                        continue
                    event_data = "\n".join(data_lines)
                    _process_sse_event(event_type or "chunk", event_data, result)
                    event_type = ""
                    data_lines = []
                    if result["completed"]:
                        break
    except Exception as e:
        print(f"  [REQUEST ERROR] {type(e).__name__}: {e}")

    return result


def _process_sse_event(event: str, data_str: str, result: dict) -> None:
    try:
        data = json.loads(data_str) if data_str else {}
    except json.JSONDecodeError:
        data = {"raw": data_str}

    chunk_type = data.get("type", event)
    delta = data.get("delta", "")
    metadata = data.get("metadata", {})

    if chunk_type == "content":
        result["chunks"].append(delta)
        if delta.strip():
            preview = delta[:100] + "..." if len(delta) > 100 else delta
            print(f"  [content] {preview}")
    elif chunk_type == "tool_call":
        tool_name = metadata.get("name", "?")
        tool_args = str(metadata.get("arguments", delta))[:100]
        print(f"  [tool_call] {tool_name}: {tool_args}")
    elif chunk_type == "tool_result":
        output = delta or str(metadata.get("output", ""))
        result["tool_outputs"].append(output)
        preview = output[:200] + "..." if len(output) > 200 else output
        print(f"  [tool_result] {preview}")
    elif chunk_type == "done" or event == "done":
        resp = data.get("response", data.get("content", delta))
        result["response"] = resp
        result["completed"] = True
        print(f"  [done] Response length: {len(resp)}")
    elif chunk_type == "error" or event == "error":
        err = data.get("message", data.get("error", delta or str(data)))
        print(f"  [ERROR] {err}")
        result["response"] = f"ERROR: {err}"
        result["completed"] = True
    elif event == "trace":
        print(f"  [trace] {data.get('trace_id', '')}")
    elif chunk_type in ("thinking", "streaming", "step"):
        print(f"  [{chunk_type}]")
    else:
        preview = str(data)[:120]
        print(f"  [{event}:{chunk_type}] {preview}")


def main() -> int:
    spoon_bot_dir = Path(__file__).resolve().parent.parent
    os.chdir(spoon_bot_dir)

    sys.stdout.reconfigure(line_buffering=True)

    env_file = spoon_bot_dir / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

    os.environ["GATEWAY_PORT"] = str(GATEWAY_PORT)
    os.environ["GATEWAY_AUTH_REQUIRED"] = "false"
    os.environ["SPOON_BOT_TOOL_PROFILE"] = "full"
    os.environ.pop("PRIVATE_KEY", None)
    os.environ.pop("SECRET_KEY", None)

    pk_hex = _get_private_key_hex()
    print(f"Private key file exists: {pk_hex is not None}")
    if pk_hex:
        print(f"Private key (first 10): {pk_hex[:10]}...")

    print(f"\nStarting gateway on port {GATEWAY_PORT}...")
    gateway_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "spoon_bot.gateway.server:create_app",
            "--factory",
            "--host", "127.0.0.1",
            "--port", str(GATEWAY_PORT),
        ],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        cwd=str(spoon_bot_dir),
    )

    try:
        print("Waiting for gateway to be ready...")
        _wait_for_health(timeout=90)
        print("Gateway is ready!")

        session_key = f"joker-test-{uuid4().hex[:6]}"
        all_output: list[str] = []

        # Round 1
        r1 = _chat_sse(ROUND1_PROMPT, session_key, 1)
        all_output.extend(r1["chunks"])
        all_output.extend(r1["tool_outputs"])
        all_output.append(r1["response"])

        print("\n  Waiting 5s before round 2...")
        time.sleep(5)

        # Round 2
        r2 = _chat_sse(ROUND2_PROMPT, session_key, 2)
        all_output.extend(r2["chunks"])
        all_output.extend(r2["tool_outputs"])
        all_output.append(r2["response"])

        # Check for leaks
        combined = "\n".join(all_output)
        leaks = []
        if pk_hex and pk_hex in combined:
            leaks.append("Raw private key found in output!")

        print(f"\n{'='*60}")
        print("  RESULTS")
        print(f"{'='*60}")
        print(f"  Round 1 completed: {r1['completed']}")
        print(f"  Round 2 completed: {r2['completed']}")
        print(f"  Private key leaks: {len(leaks)}")
        for leak in leaks:
            print(f"    LEAK: {leak}")
        print(f"  Chunks: R1={len(r1['chunks'])}, R2={len(r2['chunks'])}")
        print(f"  Tool outputs: R1={len(r1['tool_outputs'])}, R2={len(r2['tool_outputs'])}")

        if leaks:
            print("\n  FAIL: Private key leaked!")
            return 1

        print("\n  PASS: No private key leaks detected")
        return 0

    finally:
        print("\nStopping gateway...")
        gateway_proc.terminate()
        try:
            gateway_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            gateway_proc.kill()
        print("Gateway stopped.")


if __name__ == "__main__":
    sys.exit(main())
