#!/usr/bin/env python3
"""
Auto-generate API documentation from spoon-bot gateway source code.

Parses:
  - FastAPI route decorators (GET/POST/DELETE endpoints)
  - Pydantic request/response models
  - WebSocket handler methods and protocol enums
  - Gateway configuration / environment variables
  - Docker entry-point modes

Usage:
    python scripts/generate_api_docs.py              # write to docs/api.md
    python scripts/generate_api_docs.py --check      # exit 1 if docs are stale
    python scripts/generate_api_docs.py -o out.md    # custom output path
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
GATEWAY = ROOT / "spoon_bot" / "gateway"

API_V1 = GATEWAY / "api" / "v1"
HEALTH_FILE = GATEWAY / "api" / "health.py"
MODELS_REQ = GATEWAY / "models" / "requests.py"
MODELS_RESP = GATEWAY / "models" / "responses.py"
WS_HANDLER = GATEWAY / "websocket" / "handler.py"
WS_PROTOCOL = GATEWAY / "websocket" / "protocol.py"
CONFIG_FILE = GATEWAY / "config.py"
SERVER_FILE = GATEWAY / "server.py"
ROUTER_FILE = API_V1 / "router.py"

DEFAULT_OUT = ROOT / "docs" / "api.md"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EndpointInfo:
    method: str          # GET / POST / DELETE / WS
    path: str            # e.g. "/health"
    function_name: str
    docstring: str = ""
    request_model: str = ""
    response_model: str = ""
    auth_required: bool = False
    tags: list[str] = field(default_factory=list)
    source_file: str = ""
    line_number: int = 0


@dataclass
class ModelInfo:
    name: str
    docstring: str = ""
    fields: list[FieldInfo] = field(default_factory=list)
    source_file: str = ""


@dataclass
class FieldInfo:
    name: str
    type_annotation: str
    default: str = ""
    description: str = ""


@dataclass
class WSMethodInfo:
    method: str        # e.g. "chat.send"
    aliases: list[str] = field(default_factory=list)
    handler: str = ""
    docstring: str = ""


@dataclass
class WSEventInfo:
    event: str
    category: str = ""


@dataclass
class EnvVarInfo:
    name: str
    default: str = ""
    description: str = ""


# ---------------------------------------------------------------------------
# AST Helpers
# ---------------------------------------------------------------------------

def _read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_file(path: Path) -> ast.Module:
    return ast.parse(_read_source(path), filename=str(path))


def _get_docstring(node: ast.AST) -> str:
    """Extract docstring from a function/class node."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        ds = ast.get_docstring(node)
        return ds.strip() if ds else ""
    return ""


def _unparse_annotation(node: ast.AST) -> str:
    """Best-effort annotation to string."""
    try:
        return ast.unparse(node)
    except Exception:
        return "Any"


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def extract_endpoints(file_path: Path, prefix: str = "") -> list[EndpointInfo]:
    """Extract FastAPI route endpoints from a router file."""
    endpoints: list[EndpointInfo] = []
    tree = _parse_file(file_path)

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        for deco in node.decorator_list:
            # Match router.get("/path"), router.post("/path"), etc.
            if not isinstance(deco, ast.Call):
                continue
            if not isinstance(deco.func, ast.Attribute):
                continue

            method_name = deco.func.attr.upper()
            if method_name not in ("GET", "POST", "PUT", "DELETE", "PATCH"):
                continue

            # First positional arg is the path
            path = ""
            if deco.args and isinstance(deco.args[0], ast.Constant):
                path = deco.args[0].value

            # Check for response_model keyword
            response_model = ""
            for kw in deco.keywords:
                if kw.arg == "response_model":
                    response_model = ast.unparse(kw.value)

            # Check function params for request model
            request_model = ""
            has_user_dep = False
            for arg in node.args.args:
                if arg.annotation:
                    ann = ast.unparse(arg.annotation)
                    if "Request" in ann:
                        request_model = ann
                    if "CurrentUser" in ann or "current_user" in arg.arg:
                        has_user_dep = True

            full_path = prefix + path

            endpoints.append(EndpointInfo(
                method=method_name,
                path=full_path,
                function_name=node.name,
                docstring=_get_docstring(node),
                request_model=request_model,
                response_model=response_model,
                auth_required=has_user_dep,
                source_file=str(file_path.relative_to(ROOT)),
                line_number=node.lineno,
            ))

    return endpoints


def extract_pydantic_models(file_path: Path) -> list[ModelInfo]:
    """Extract Pydantic BaseModel classes from a file."""
    models: list[ModelInfo] = []
    tree = _parse_file(file_path)

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        # Check if extends BaseModel
        is_pydantic = False
        for base in node.bases:
            base_str = ast.unparse(base)
            if "BaseModel" in base_str:
                is_pydantic = True
                break

        if not is_pydantic:
            continue

        fields: list[FieldInfo] = []
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                fname = item.target.id
                ftype = ast.unparse(item.annotation) if item.annotation else "Any"
                fdefault = ""
                if item.value is not None:
                    fdefault = ast.unparse(item.value)
                fields.append(FieldInfo(name=fname, type_annotation=ftype, default=fdefault))

        models.append(ModelInfo(
            name=node.name,
            docstring=_get_docstring(node),
            fields=fields,
            source_file=str(file_path.relative_to(ROOT)),
        ))

    return models


def extract_enum_members(file_path: Path, class_name: str) -> list[tuple[str, str]]:
    """Extract Enum members (name, value) from a specific Enum class."""
    tree = _parse_file(file_path)
    members: list[tuple[str, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name) and isinstance(item.value, ast.Constant):
                        members.append((target.id, str(item.value.value)))

    return members


def extract_ws_handlers(file_path: Path) -> list[WSMethodInfo]:
    """Extract WebSocket handler method mappings from handler.py."""
    source = _read_source(file_path)
    methods: list[WSMethodInfo] = []
    seen: set[str] = set()

    # Parse handler registrations: "method.name": self._handle_xxx
    pattern = re.compile(
        r'["\']([a-z_.]+)["\']\s*:\s*self\.(_handle_\w+)',
        re.MULTILINE,
    )

    for match in pattern.finditer(source):
        method_str = match.group(1)
        handler_name = match.group(2)

        if handler_name in seen:
            # Alias — find existing and add alias
            for m in methods:
                if m.handler == handler_name and method_str not in m.aliases:
                    m.aliases.append(method_str)
            continue

        seen.add(handler_name)

        # Try to extract docstring from handler function
        docstring = ""
        tree = _parse_file(file_path)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == handler_name:
                    docstring = _get_docstring(node)
                    break

        methods.append(WSMethodInfo(
            method=method_str,
            handler=handler_name,
            docstring=docstring,
        ))

    return methods


def extract_env_vars(file_path: Path) -> list[EnvVarInfo]:
    """Extract os.environ.get() calls from a file to discover env vars."""
    source = _read_source(file_path)
    env_vars: list[EnvVarInfo] = []
    seen: set[str] = set()

    pattern = re.compile(
        r'os\.environ\.get\(\s*["\'](\w+)["\']\s*(?:,\s*["\']([^"\']*)["\'])?\s*\)',
        re.MULTILINE,
    )

    for match in pattern.finditer(source):
        name = match.group(1)
        default = match.group(2) or ""

        if name in seen:
            continue
        seen.add(name)

        env_vars.append(EnvVarInfo(name=name, default=default))

    return env_vars


def extract_router_prefixes(file_path: Path) -> dict[str, str]:
    """Extract router.include_router prefix mappings."""
    source = _read_source(file_path)
    prefixes: dict[str, str] = {}

    # router.include_router(xxx_router, prefix="/xxx", ...)
    pattern = re.compile(
        r'include_router\(\s*(\w+)\s*,\s*prefix\s*=\s*["\']([^"\']+)["\']',
        re.MULTILINE,
    )

    for match in pattern.finditer(source):
        router_var = match.group(1)
        prefix = match.group(2)
        prefixes[router_var] = prefix

    return prefixes


# ---------------------------------------------------------------------------
# Markdown Generator
# ---------------------------------------------------------------------------

def _model_to_json_example(model: ModelInfo) -> str:
    """Generate a JSON-like example from a Pydantic model."""
    lines = ["{"]
    for i, f in enumerate(model.fields):
        comma = "," if i < len(model.fields) - 1 else ""
        example = _type_to_example(f.type_annotation, f.default, f.name)
        lines.append(f'  "{f.name}": {example}{comma}')
    lines.append("}")
    return "\n".join(lines)


def _type_to_example(type_str: str, default: str, name: str) -> str:
    """Produce a JSON example value for a type annotation."""
    if default and default != "None" and not default.startswith("Field(") and default != "field(default_factory=list)":
        # Literal defaults
        if default.startswith('"') or default.startswith("'"):
            return default.replace("'", '"')
        if default in ("True", "False"):
            return default.lower()
        return default

    lower = type_str.lower()
    if "str" in lower:
        return f'"..."'
    if "int" in lower:
        return "0"
    if "float" in lower:
        return "0.0"
    if "bool" in lower:
        return "false"
    if "list" in lower:
        return "[]"
    if "dict" in lower:
        return "{}"
    if "datetime" in lower:
        return '"2026-01-01T00:00:00.000000"'
    return "null"


def generate_markdown(out_path: Path) -> str:
    """Generate full API documentation markdown."""
    lines: list[str] = []

    def w(text: str = "") -> None:
        lines.append(text)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # =====================================================================
    # Header
    # =====================================================================
    w("# Spoon-Bot Gateway API (Frontend Integration)")
    w()
    w("This document describes the **Gateway API** exposed by spoon-bot. The frontend should connect to these endpoints for agent interactions, session management, tool/skill control, and real-time streaming.")
    w()
    w(f"> **Auto-generated** from source code on {now}.  ")
    w("> Regenerate with: `python scripts/generate_api_docs.py`")
    w()
    w("Base URL (local): `http://localhost:8080`  ")
    w("API Docs (Swagger): `http://localhost:8080/docs`  ")
    w("API Docs (ReDoc): `http://localhost:8080/redoc`")
    w()

    # =====================================================================
    # Table of Contents
    # =====================================================================
    w("---")
    w()
    w("## Table of Contents")
    w()
    toc = [
        "Authentication",
        "Health & Status",
        "Agent Chat (HTTP)",
        "Agent Chat (SSE Streaming)",
        "Agent Status",
        "Async Tasks",
        "Session Management",
        "Tool Management",
        "Skill Management",
        "WebSocket Real-Time Communication",
        "Request/Response Models",
        "Error Handling",
        "Environment Variables",
        "Docker Deployment",
    ]
    for i, t in enumerate(toc, 1):
        anchor = t.lower().replace(" ", "-").replace("/", "").replace("(", "").replace(")", "")
        w(f"{i}. [{t}](#{anchor})")
    w()

    # =====================================================================
    # 1) Authentication
    # =====================================================================
    w("---")
    w()
    w("## 1) Authentication")
    w()
    w("All `/v1/*` endpoints (except `/v1/auth/login`) require authentication. Two methods are supported:")
    w()
    w("| Method | Header / Param | Example |")
    w("|--------|----------------|---------|")
    w("| API Key | `X-API-Key` header | `X-API-Key: sk_live_abc123...` |")
    w("| JWT Bearer Token | `Authorization` header | `Authorization: Bearer eyJhbG...` |")
    w()
    w("> **Bypass Mode:** Set `GATEWAY_AUTH_REQUIRED=false` to disable authentication entirely.")
    w()
    w("### Permission Scopes")
    w()
    w("| Scope | Description |")
    w("|-------|-------------|")
    w("| `agent:read` | Read agent status, list tools/skills/sessions |")
    w("| `agent:write` | Send messages to agent, manage sessions |")
    w("| `admin` | Execute tools directly, all permissions |")
    w()

    # Auth endpoints
    auth_endpoints = extract_endpoints(API_V1 / "auth.py", prefix="/v1/auth")
    for ep in auth_endpoints:
        w(f"### `{ep.method} {ep.path}`")
        w()
        if ep.docstring:
            w(ep.docstring.split("\n")[0])
        w()
        if ep.request_model:
            w(f"**Request Body:** `{ep.request_model}`")
            w()
        if ep.response_model:
            w(f"**Response Model:** `{ep.response_model}`")
            w()
        w(f"*Source: `{ep.source_file}:{ep.line_number}`*")
        w()

    # =====================================================================
    # 2) Health
    # =====================================================================
    w("---")
    w()
    w("## 2) Health & Status")
    w()
    w("These endpoints do **not** require authentication.")
    w()

    health_endpoints = extract_endpoints(HEALTH_FILE)
    for ep in health_endpoints:
        w(f"### `{ep.method} {ep.path}`")
        w()
        if ep.docstring:
            w(ep.docstring.split("\n")[0])
        w()
        if ep.response_model:
            w(f"**Response Model:** `{ep.response_model}`")
        w()
        w(f"*Source: `{ep.source_file}:{ep.line_number}`*")
        w()

    # =====================================================================
    # 3-6) Agent endpoints
    # =====================================================================
    agent_endpoints = extract_endpoints(API_V1 / "agent.py", prefix="/v1/agent")

    chat_eps = [e for e in agent_endpoints if "/chat" in e.path and "async" not in e.path]
    async_eps = [e for e in agent_endpoints if "async" in e.path or "task" in e.path]
    status_eps = [e for e in agent_endpoints if "status" in e.path]

    # Section 3: Chat HTTP
    w("---")
    w()
    w("## 3) Agent Chat (HTTP)")
    w()
    for ep in chat_eps:
        w(f"### `{ep.method} {ep.path}`")
        w()
        if ep.docstring:
            for line in ep.docstring.split("\n"):
                w(line)
            w()
        w(f"**Auth Required:** {'Yes' if ep.auth_required else 'No'}")
        w()
        if ep.request_model:
            w(f"**Request Body:** `{ep.request_model}`")
            w()
        w("**Request Example:**")
        w("```json")
        w('{')
        w('  "message": "Help me write a Python script",')
        w('  "session_key": "default",')
        w('  "media": [],')
        w('  "options": {')
        w('    "max_iterations": 20,')
        w('    "stream": false,')
        w('    "thinking": false,')
        w('    "model": null')
        w('  }')
        w('}')
        w("```")
        w()
        w("**Request Fields:**")
        w()
        w("| Field | Type | Required | Default | Description |")
        w("|-------|------|----------|---------|-------------|")
        w("| `message` | string | Yes | - | User message (1-100,000 chars) |")
        w("| `session_key` | string | No | `\"default\"` | Session identifier (`^[a-zA-Z0-9_-]{1,64}$`) |")
        w("| `media` | string[] | No | `[]` | Media file paths (max 10) |")
        w("| `options.max_iterations` | int | No | `20` | Max tool call iterations (1-100) |")
        w("| `options.stream` | bool | No | `false` | Enable SSE streaming (see Section 4) |")
        w("| `options.thinking` | bool | No | `false` | Include extended thinking output |")
        w("| `options.model` | string | No | `null` | Override default LLM model |")
        w()
        w("**Response (non-streaming):** `APIResponse[ChatResponse]`")
        w("```json")
        w('{')
        w('  "success": true,')
        w('  "data": {')
        w('    "response": "...",')
        w('    "tool_calls": [],')
        w('    "usage": null,')
        w('    "thinking_content": null')
        w('  },')
        w('  "meta": {')
        w('    "request_id": "req_a1b2c3d4e5f6",')
        w('    "timestamp": "2026-02-09T07:00:00.000000",')
        w('    "duration_ms": 2500')
        w('  }')
        w('}')
        w("```")
        w()
        w(f"*Source: `{ep.source_file}:{ep.line_number}`*")
        w()

    # Section 4: SSE Streaming
    w("---")
    w()
    w("## 4) Agent Chat (SSE Streaming)")
    w()
    w("### `POST /v1/agent/chat` (with `stream: true`)")
    w()
    w("Same endpoint as non-streaming, but returns a Server-Sent Events (SSE) stream when `options.stream` is `true`.")
    w()
    w("**Request:**")
    w("```json")
    w('{ "message": "Explain quantum computing", "options": { "stream": true } }')
    w("```")
    w()
    w("**Response:** `Content-Type: text/event-stream`")
    w()
    w("```")
    w('data: {"type": "content", "delta": "Quantum ", "metadata": {}}')
    w()
    w('data: {"type": "thinking", "delta": "Let me think...", "metadata": {}}')
    w()
    w("data: [DONE]")
    w("```")
    w()
    w("**Chunk Types:**")
    w()
    w("| Type | Description |")
    w("|------|-------------|")
    w("| `content` | Response text chunk |")
    w("| `thinking` | Extended thinking content chunk |")
    w("| `tool_call` | Tool call notification |")
    w("| `error` | Error occurred during streaming |")
    w()
    w("**Response Headers:**")
    w("```")
    w("Content-Type: text/event-stream")
    w("Cache-Control: no-cache")
    w("Connection: keep-alive")
    w("X-Request-ID: req_a1b2c3d4e5f6")
    w("```")
    w()

    # Section 5: Status
    w("---")
    w()
    w("## 5) Agent Status")
    w()
    for ep in status_eps:
        w(f"### `{ep.method} {ep.path}`")
        w()
        if ep.docstring:
            w(ep.docstring.split("\n")[0])
        w()
        w(f"**Auth Required:** {'Yes' if ep.auth_required else 'No'}")
        w()
        if ep.response_model:
            w(f"**Response Model:** `{ep.response_model}`")
        w()
        w(f"*Source: `{ep.source_file}:{ep.line_number}`*")
        w()

    # Section 6: Async Tasks
    w("---")
    w()
    w("## 6) Async Tasks")
    w()
    w("> **Note:** Async task endpoints are currently placeholder implementations.")
    w()
    for ep in async_eps:
        w(f"### `{ep.method} {ep.path}`")
        w()
        if ep.docstring:
            w(ep.docstring.split("\n")[0])
        w()
        w(f"*Source: `{ep.source_file}:{ep.line_number}`*")
        w()

    # =====================================================================
    # 7) Sessions
    # =====================================================================
    w("---")
    w()
    w("## 7) Session Management")
    w()
    session_endpoints = extract_endpoints(API_V1 / "sessions.py", prefix="/v1/sessions")
    for ep in session_endpoints:
        w(f"### `{ep.method} {ep.path}`")
        w()
        if ep.docstring:
            w(ep.docstring.split("\n")[0])
        w()
        w(f"**Auth Required:** {'Yes' if ep.auth_required else 'No'}")
        w()
        if ep.request_model:
            w(f"**Request Body:** `{ep.request_model}`")
        w()
        w(f"*Source: `{ep.source_file}:{ep.line_number}`*")
        w()

    # =====================================================================
    # 8) Tools
    # =====================================================================
    w("---")
    w()
    w("## 8) Tool Management")
    w()
    tool_endpoints = extract_endpoints(API_V1 / "tools.py", prefix="/v1/tools")
    for ep in tool_endpoints:
        w(f"### `{ep.method} {ep.path}`")
        w()
        if ep.docstring:
            w(ep.docstring.split("\n")[0])
        w()
        w(f"**Auth Required:** {'Yes' if ep.auth_required else 'No'}")
        w()
        if ep.request_model:
            w(f"**Request Body:** `{ep.request_model}`")
        w()
        w(f"*Source: `{ep.source_file}:{ep.line_number}`*")
        w()

    # =====================================================================
    # 9) Skills
    # =====================================================================
    w("---")
    w()
    w("## 9) Skill Management")
    w()
    skill_endpoints = extract_endpoints(API_V1 / "skills.py", prefix="/v1/skills")
    for ep in skill_endpoints:
        w(f"### `{ep.method} {ep.path}`")
        w()
        if ep.docstring:
            w(ep.docstring.split("\n")[0])
        w()
        w(f"**Auth Required:** {'Yes' if ep.auth_required else 'No'}")
        w()
        if ep.request_model:
            w(f"**Request Body:** `{ep.request_model}`")
        w()
        w(f"*Source: `{ep.source_file}:{ep.line_number}`*")
        w()

    # =====================================================================
    # 10) WebSocket
    # =====================================================================
    w("---")
    w()
    w("## 10) WebSocket Real-Time Communication")
    w()
    w("### Connection")
    w()
    w("**Endpoint:** `ws://localhost:8080/v1/ws`")
    w()
    w("**Authentication via Query Parameters:**")
    w("```")
    w("ws://localhost:8080/v1/ws?token=<jwt_access_token>")
    w("ws://localhost:8080/v1/ws?api_key=<api_key>")
    w("```")
    w()
    w("> If `GATEWAY_AUTH_REQUIRED=false`, no authentication parameters are needed.")
    w()

    # Protocol
    w("### Message Protocol")
    w()
    w("| Type | Direction | Description |")
    w("|------|-----------|-------------|")

    msg_types = extract_enum_members(WS_PROTOCOL, "MessageType")
    for name, value in msg_types:
        direction = "Client -> Server" if value in ("request", "ping") else "Server -> Client"
        if value == "stream":
            direction = "Server -> Client"
        w(f"| `{value}` | {direction} | {name.replace('_', ' ').title()} |")
    w()

    # Client methods
    w("### Client Methods (Client -> Server)")
    w()
    ws_methods = extract_ws_handlers(WS_HANDLER)
    for m in ws_methods:
        aliases = ""
        if m.aliases:
            alias_strs = ", ".join(f"`{a}`" for a in m.aliases)
            aliases = f" (aliases: {alias_strs})"
        w(f"#### `{m.method}`{aliases}")
        w()
        if m.docstring:
            w(m.docstring.split("—")[0].strip() if "—" in m.docstring else m.docstring.split("\n")[0])
        w()

    # Server events
    w("### Server Events (Server -> Client)")
    w()
    w("| Event | Category | Description |")
    w("|-------|----------|-------------|")

    events = extract_enum_members(WS_PROTOCOL, "ServerEvent")
    for name, value in events:
        # Determine category from prefix
        if value.startswith("agent."):
            cat = "Agent"
        elif value.startswith("confirm."):
            cat = "Confirmation"
        elif value.startswith("resource.") or value.startswith("metrics."):
            cat = "Resource"
        elif value.startswith("connection."):
            cat = "Connection"
        else:
            cat = "Other"
        desc = name.replace("_", " ").title()
        w(f"| `{value}` | {cat} | {desc} |")
    w()

    # =====================================================================
    # 11) Models
    # =====================================================================
    w("---")
    w()
    w("## 11) Request/Response Models")
    w()

    # Request models
    w("### Request Models")
    w()
    w(f"*Source: `{MODELS_REQ.relative_to(ROOT)}`*")
    w()
    req_models = extract_pydantic_models(MODELS_REQ)
    for model in req_models:
        w(f"#### `{model.name}`")
        w()
        if model.docstring:
            w(model.docstring)
            w()
        if model.fields:
            w("| Field | Type | Default |")
            w("|-------|------|---------|")
            for f in model.fields:
                default = f.default if f.default else "*(required)*"
                w(f"| `{f.name}` | `{f.type_annotation}` | {default} |")
        w()

    # Response models
    w("### Response Models")
    w()
    w(f"*Source: `{MODELS_RESP.relative_to(ROOT)}`*")
    w()
    resp_models = extract_pydantic_models(MODELS_RESP)
    for model in resp_models:
        w(f"#### `{model.name}`")
        w()
        if model.docstring:
            w(model.docstring)
            w()
        if model.fields:
            w("| Field | Type | Default |")
            w("|-------|------|---------|")
            for f in model.fields:
                default = f.default if f.default else "*(required)*"
                w(f"| `{f.name}` | `{f.type_annotation}` | {default} |")
        w()

    # =====================================================================
    # 12) Error Handling
    # =====================================================================
    w("---")
    w()
    w("## 12) Error Handling")
    w()
    w("### Standard API Error Response")
    w()
    w("```json")
    w('{')
    w('  "detail": {')
    w('    "code": "ERROR_CODE",')
    w('    "message": "Human-readable description"')
    w('  }')
    w('}')
    w("```")
    w()
    w("### HTTP Status Codes")
    w()
    w("| Status | Description |")
    w("|--------|-------------|")
    w("| 200 | Success |")
    w("| 401 | Authentication required or invalid |")
    w("| 403 | Insufficient permissions |")
    w("| 404 | Resource not found |")
    w("| 409 | Conflict (e.g., session already exists) |")
    w("| 422 | Validation error |")
    w("| 500 | Internal server error |")
    w()
    w("### Common Error Codes")
    w()
    w("| Code | Description |")
    w("|------|-------------|")
    w("| `AUTH_REQUIRED` | Missing authentication credentials |")
    w("| `AUTH_INVALID` | Invalid credentials |")
    w("| `FORBIDDEN` | Insufficient scope for this operation |")
    w("| `NOT_FOUND` | Requested resource not found |")
    w("| `SESSION_EXISTS` | Session with this key already exists |")
    w("| `AGENT_ERROR` | Agent processing failure |")
    w("| `INTERNAL_ERROR` | Unexpected internal error |")
    w("| `ACTIVATION_FAILED` | Skill activation failed |")
    w()

    # =====================================================================
    # 13) Environment Variables
    # =====================================================================
    w("---")
    w()
    w("## 13) Environment Variables")
    w()

    # Collect env vars from config and server
    config_vars = extract_env_vars(CONFIG_FILE)
    server_vars = extract_env_vars(SERVER_FILE)

    # Deduplicate
    seen_names: set[str] = set()
    all_vars: list[EnvVarInfo] = []
    for v in server_vars + config_vars:
        if v.name not in seen_names:
            seen_names.add(v.name)
            all_vars.append(v)

    w("| Variable | Default | Source |")
    w("|----------|---------|--------|")
    for v in all_vars:
        default = f"`{v.default}`" if v.default else "*(none)*"
        source = "server.py" if v in server_vars else "config.py"
        w(f"| `{v.name}` | {default} | {source} |")
    w()

    w("### Default Models Per Provider")
    w()
    w("| Provider | Default Model |")
    w("|----------|---------------|")
    # Extract from server.py default_models dict
    server_src = _read_source(SERVER_FILE)
    model_pattern = re.compile(r'"(\w+)":\s*"([^"]+)"')
    in_defaults = False
    for line in server_src.split("\n"):
        if "default_models" in line:
            in_defaults = True
            continue
        if in_defaults:
            if "}" in line:
                break
            m = model_pattern.search(line)
            if m:
                w(f"| `{m.group(1)}` | `{m.group(2)}` |")
    w()

    # =====================================================================
    # 14) Docker
    # =====================================================================
    w("---")
    w()
    w("## 14) Docker Deployment")
    w()
    w("### Quick Start")
    w()
    w("```bash")
    w("# 1. Configure environment")
    w("cp .env.example .env")
    w("# Edit .env with your API keys")
    w()
    w("# 2. Build and start")
    w("docker compose up -d")
    w()
    w("# 3. Verify")
    w("curl http://localhost:8080/health")
    w("curl http://localhost:8080/ready")
    w("```")
    w()
    w("### Run Modes")
    w()
    w("| Mode | Description |")
    w("|------|-------------|")
    w("| `gateway` | HTTP/WebSocket API server (default) |")
    w("| `agent` | One-shot agent execution |")
    w("| `cli` | CLI mode |")
    w("| `onboard` | Initialize workspace |")
    w()
    w("Set mode via `SPOON_BOT_MODE` environment variable.")
    w()

    # =====================================================================
    # Endpoint Summary Table
    # =====================================================================
    w("---")
    w()
    w("## API Endpoint Summary")
    w()

    # Collect ALL endpoints
    all_endpoints: list[EndpointInfo] = []
    all_endpoints.extend(health_endpoints)
    all_endpoints.extend(auth_endpoints)
    all_endpoints.extend(agent_endpoints)
    all_endpoints.extend(session_endpoints)
    all_endpoints.extend(tool_endpoints)
    all_endpoints.extend(skill_endpoints)

    w("| Method | Path | Auth | Description |")
    w("|--------|------|------|-------------|")
    for ep in all_endpoints:
        auth = "Yes" if ep.auth_required else "No"
        desc = ep.docstring.split("\n")[0][:60] if ep.docstring else ep.function_name
        w(f"| `{ep.method}` | `{ep.path}` | {auth} | {desc} |")

    w(f"| `WS` | `/v1/ws` | Query param | Real-time bidirectional communication |")
    w()

    # =====================================================================
    # Footer
    # =====================================================================
    w("---")
    w()
    w("## Notes for Frontend")
    w()
    w("- Use **WebSocket** (`/v1/ws`) for real-time streaming UI with bidirectional communication.")
    w("- Use **HTTP SSE** (`POST /v1/agent/chat` with `stream: true`) for simpler streaming without WebSocket.")
    w("- Use **HTTP POST** (`POST /v1/agent/chat`) for simple request/response (no streaming).")
    w("- When `GATEWAY_AUTH_REQUIRED=false`, all endpoints work without authentication.")
    w("- Session keys must match pattern `^[a-zA-Z0-9_-]{1,64}$`.")
    w("- Messages are sanitized: control characters (except `\\n`, `\\r`, `\\t`) are stripped.")
    w("- OpenAPI docs are available at `/docs` (Swagger UI) and `/redoc` (ReDoc).")
    w()

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate spoon-bot API documentation from source code."
    )
    parser.add_argument(
        "-o", "--output",
        default=str(DEFAULT_OUT),
        help="Output file path (default: docs/api.md)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if existing docs are up-to-date (exit 1 if stale).",
    )
    args = parser.parse_args()

    out_path = Path(args.output)

    # Ensure required source files exist
    required_files = [
        HEALTH_FILE, MODELS_REQ, MODELS_RESP,
        WS_HANDLER, WS_PROTOCOL, CONFIG_FILE, SERVER_FILE,
    ]
    for api_file in ["auth.py", "agent.py", "sessions.py", "tools.py", "skills.py"]:
        required_files.append(API_V1 / api_file)

    for f in required_files:
        if not f.exists():
            print(f"ERROR: Required source file not found: {f}", file=sys.stderr)
            return 1

    # Generate
    content = generate_markdown(out_path)

    if args.check:
        if not out_path.exists():
            print(f"STALE: {out_path} does not exist.", file=sys.stderr)
            return 1

        existing = out_path.read_text(encoding="utf-8")
        # Strip the auto-generated timestamp line for comparison
        ts_re = re.compile(r"> \*\*Auto-generated\*\* from source code on .+\.")
        existing_clean = ts_re.sub("", existing).strip()
        content_clean = ts_re.sub("", content).strip()

        if existing_clean != content_clean:
            print(
                f"STALE: {out_path} is out of date.\n"
                f"Run: python scripts/generate_api_docs.py",
                file=sys.stderr,
            )
            return 1

        print(f"OK: {out_path} is up to date.")
        return 0

    # Write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    print(f"Generated: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
