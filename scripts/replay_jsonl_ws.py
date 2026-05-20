from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import urllib.error
import urllib.request

import websockets
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a JSONL conversation into a live local spoon-bot gateway over WebSocket.",
    )
    parser.add_argument("--jsonl", help="Path to the source JSONL file.")
    parser.add_argument(
        "--session-key",
        help="Target session key. Defaults to the JSONL stem.",
    )
    parser.add_argument(
        "--prompt",
        help="Direct prompt to send over WebSocket. Skips JSONL replay history import.",
    )
    parser.add_argument(
        "--prompt-file",
        help="Path to a UTF-8 text file containing one prompt per block separated by a line with only ---.",
    )
    parser.add_argument(
        "--prompt-index",
        type=int,
        default=-1,
        help="Zero-based user-message index to replay. Defaults to the last user message.",
    )
    parser.add_argument(
        "--history-tail-messages",
        type=int,
        help="If set, keep only the last N messages before the replayed prompt.",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Gateway port. Defaults to an ephemeral free port.",
    )
    parser.add_argument(
        "--gateway-timeout",
        type=int,
        default=900,
        help="Seconds to wait for the streamed run to finish.",
    )
    parser.add_argument(
        "--ws-ping-interval",
        type=float,
        default=0,
        help="Protocol ping interval in seconds; 0 disables client pings for long agent turns.",
    )
    parser.add_argument(
        "--ws-ping-timeout",
        type=float,
        default=0,
        help="Protocol ping timeout in seconds; 0 disables client ping timeouts.",
    )
    parser.add_argument(
        "--print-events",
        action="store_true",
        help="Print every incoming WebSocket payload.",
    )
    return parser.parse_args()


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def find_config_path(repo_root: Path) -> Path | None:
    explicit = os.environ.get("SPOON_BOT_CONFIG")
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.exists() else None
    for candidate in (
        Path.home() / ".spoon-bot" / "config.yaml",
        repo_root / "config.yaml",
    ):
        if candidate.exists():
            return candidate
    return None


def load_base_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format in {config_path}")
    return data


def build_temp_config(
    repo_root: Path,
    workspace: Path,
    config_source_path: Path | None,
) -> Path:
    base_config = load_base_config(config_source_path or find_config_path(repo_root))
    agent = dict(base_config.get("agent") or {})
    agent["workspace"] = str(workspace)
    agent["tool_profile"] = "full"
    agent.setdefault("enable_skills", True)

    replay_config = {
        "agent": agent,
        "channels": {
            "telegram": {"enabled": False},
            "discord": {"enabled": False},
            "feishu": {"enabled": False},
            "cli": {"enabled": False},
        },
        "cron": {"enabled": False},
    }

    fd, temp_path = tempfile.mkstemp(prefix="spoon-replay-", suffix=".yaml")
    os.close(fd)
    path = Path(temp_path)
    path.write_text(yaml.safe_dump(replay_config, sort_keys=False), encoding="utf-8")
    return path


def parse_jsonl_messages(source: Path) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            messages.append(payload)
    return messages


def parse_prompt_file(source: Path) -> list[str]:
    raw = source.read_text(encoding="utf-8")
    prompts = [block.strip() for block in raw.split("\n---\n")]
    return [prompt for prompt in prompts if prompt]


def pick_target_prompt(
    messages: list[dict[str, Any]],
    prompt_index: int,
) -> tuple[str, list[dict[str, Any]], int]:
    user_positions = [idx for idx, message in enumerate(messages) if message.get("role") == "user"]
    if not user_positions:
        raise ValueError("No user messages found in JSONL.")

    resolved_index = prompt_index
    if resolved_index < 0:
        resolved_index = len(user_positions) + resolved_index
    if resolved_index < 0 or resolved_index >= len(user_positions):
        raise IndexError(
            f"prompt_index {prompt_index} is out of range for {len(user_positions)} user messages.",
        )

    message_index = user_positions[resolved_index]
    prompt_message = messages[message_index]
    prompt = str(prompt_message.get("content") or "").strip()
    if not prompt:
        raise ValueError("Selected user message has empty content.")
    return prompt, messages[:message_index], message_index


def backup_if_exists(path: Path) -> None:
    if not path.exists():
        return
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak.{timestamp}")
    shutil.copy2(path, backup)
    meta = path.with_suffix(".meta.json")
    if meta.exists():
        shutil.copy2(meta, meta.with_suffix(meta.suffix + f".bak.{timestamp}"))


def write_session_history(
    workspace: Path,
    session_key: str,
    history: list[dict[str, Any]],
) -> Path:
    sessions_dir = workspace / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_key)
    session_path = sessions_dir / f"{safe_key}.jsonl"
    meta_path = session_path.with_suffix(".meta.json")
    backup_if_exists(session_path)

    session_path.write_text(
        "".join(json.dumps(message, ensure_ascii=False) + "\n" for message in history),
        encoding="utf-8",
    )

    now = datetime.now(UTC).isoformat()
    meta = {
        "session_key": session_key,
        "created_at": now,
        "updated_at": now,
        "metadata": {"replayed_from_jsonl": True},
        "message_count": len(history),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return session_path


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_health(base_url: str, timeout_seconds: int = 90) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=3) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(1.5)
    raise TimeoutError(f"Gateway did not become healthy within {timeout_seconds}s")


async def stream_chat(
    *,
    ws_url: str,
    prompt: str,
    session_key: str,
    timeout_seconds: int,
    print_events: bool,
    ws_ping_interval: float,
    ws_ping_timeout: float,
) -> dict[str, Any]:
    request_id = f"replay_{int(time.time())}"
    events: list[dict[str, Any]] = []
    content_parts: list[str] = []
    tool_results: list[str] = []

    ping_interval = ws_ping_interval if ws_ping_interval > 0 else None
    ping_timeout = ws_ping_timeout if ws_ping_timeout > 0 else None
    async with websockets.connect(
        ws_url,
        ping_interval=ping_interval,
        ping_timeout=ping_timeout,
    ) as ws:
        established = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        events.append(established)

        await ws.send(json.dumps({
            "type": "request",
            "id": request_id,
            "method": "chat.send",
            "params": {
                "message": prompt,
                "session_key": session_key,
                "stream": True,
                "thinking": False,
            },
        }))

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=max(1.0, deadline - time.monotonic()))
            message = json.loads(raw)
            events.append(message)
            if print_events:
                print(json.dumps(message, ensure_ascii=False))

            if message.get("type") == "event" and message.get("event") == "agent.stream.chunk":
                data = message.get("data") or {}
                if data.get("type") == "content":
                    content_parts.append(str(data.get("delta") or ""))
                elif data.get("type") == "tool_result":
                    delta = str(data.get("delta") or "")
                    if delta:
                        tool_results.append(delta)
            if message.get("type") == "response" and message.get("id") == request_id:
                return {
                    "events": events,
                    "content": "".join(content_parts),
                    "tool_results": tool_results,
                    "response": message.get("result") or {},
                }

    raise TimeoutError(f"Timed out waiting for WebSocket replay response after {timeout_seconds}s")


def start_gateway(
    repo_root: Path,
    config_path: Path,
    port: int,
) -> tuple[subprocess.Popen[str], Path, Any]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["GATEWAY_AUTH_REQUIRED"] = "false"
    env["SPOON_BOT_CONFIG"] = str(config_path)
    env["GATEWAY_PORT"] = str(port)
    fd, log_name = tempfile.mkstemp(prefix="spoon-replay-gateway-", suffix=".log")
    os.close(fd)
    log_path = Path(log_name)
    log_file = log_path.open("w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "spoon_bot.gateway.server:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(repo_root),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process, log_path, log_file


def collect_process_output(log_path: Path, log_file: Any) -> str:
    try:
        log_file.flush()
    except Exception:
        pass
    try:
        log_file.close()
    except Exception:
        pass
    try:
        return log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    if not args.jsonl and not args.prompt and not args.prompt_file:
        raise ValueError("Provide one of --jsonl, --prompt, or --prompt-file.")
    if sum(bool(value) for value in (args.jsonl, args.prompt, args.prompt_file)) > 1:
        raise ValueError("Use only one of --jsonl, --prompt, or --prompt-file.")

    source_path: Path | None = None
    prompt_source = "direct"
    if args.jsonl:
        source_path = Path(args.jsonl).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"JSONL not found: {source_path}")
        prompt_source = "jsonl"
    elif args.prompt_file:
        source_path = Path(args.prompt_file).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {source_path}")
        prompt_source = "prompt_file"

    config_source_path = find_config_path(repo_root)
    if config_source_path is not None:
        load_dotenv(config_source_path.parent / ".env")
    load_dotenv(repo_root / ".env")

    workspace = Path.home() / ".spoon-bot" / "workspace"
    config_path = build_temp_config(repo_root, workspace, config_source_path)

    session_key = args.session_key or (source_path.stem if source_path else f"replay_{int(time.time())}")
    prompts: list[str] = []
    history: list[dict[str, Any]] = []
    message_index: int | None = None
    session_path: Path | None = None

    if args.jsonl and source_path is not None:
        messages = parse_jsonl_messages(source_path)
        prompt, history, message_index = pick_target_prompt(messages, args.prompt_index)
        if args.history_tail_messages is not None:
            if args.history_tail_messages < 0:
                raise ValueError("--history-tail-messages must be non-negative")
            history = history[-args.history_tail_messages :]
        session_path = write_session_history(workspace, session_key, history)
        prompts = [prompt]
    elif args.prompt_file and source_path is not None:
        prompts = parse_prompt_file(source_path)
        if not prompts:
            raise ValueError("Prompt file did not contain any prompt blocks.")
    elif args.prompt:
        prompt = str(args.prompt).strip()
        if not prompt:
            raise ValueError("--prompt must not be empty.")
        prompts = [prompt]

    port = args.port or find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    ws_url = f"ws://127.0.0.1:{port}/v1/ws"

    gateway, gateway_log_path, gateway_log_file = start_gateway(repo_root, config_path, port)
    gateway_output = ""
    try:
        wait_for_health(base_url)
        results = []
        for index, prompt in enumerate(prompts):
            result = asyncio.run(
                stream_chat(
                    ws_url=ws_url,
                    prompt=prompt,
                    session_key=session_key,
                    timeout_seconds=args.gateway_timeout,
                    print_events=args.print_events,
                    ws_ping_interval=args.ws_ping_interval,
                    ws_ping_timeout=args.ws_ping_timeout,
                )
            )
            results.append(
                {
                    "prompt_index": index,
                    "prompt": prompt,
                    "content": result["content"],
                    "tool_result_count": len(result["tool_results"]),
                    "response": result["response"],
                }
            )
        print(json.dumps({
            "source": str(source_path) if source_path else None,
            "prompt_source": prompt_source,
            "session_key": session_key,
            "replayed_user_message_index": message_index,
            "history_messages": len(history),
            "history_tail_messages": args.history_tail_messages,
            "session_path": str(session_path) if session_path else None,
            "results": results,
        }, ensure_ascii=False, indent=2))
        return 0
    finally:
        gateway.terminate()
        try:
            gateway.wait(timeout=15)
        except subprocess.TimeoutExpired:
            gateway.kill()
            gateway.wait(timeout=5)
        gateway_output = collect_process_output(gateway_log_path, gateway_log_file)
        if gateway_output.strip():
            print("\n=== gateway log tail ===")
            print(gateway_output[-8000:])
        try:
            gateway_log_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            config_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
