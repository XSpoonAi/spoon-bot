# Spoon-Bot Gateway API (Frontend Integration)

This document describes the **Gateway API** exposed by spoon-bot. The frontend should connect to these endpoints for agent interactions, session management, tool/skill control, and real-time streaming.

> **Auto-generated** from source code on 2026-04-24 13:42 UTC.  
> Regenerate with: `python scripts/generate_api_docs.py`

Base URL (local): `http://localhost:8080`  
API Docs (Swagger): `http://localhost:8080/docs`  
API Docs (ReDoc): `http://localhost:8080/redoc`

---

## Table of Contents

1. [Authentication](#authentication)
2. [Health & Status](#health-&-status)
3. [Agent Chat (HTTP)](#agent-chat-http)
4. [Agent Chat (SSE Streaming)](#agent-chat-sse-streaming)
5. [Agent Status](#agent-status)
6. [Async Tasks](#async-tasks)
7. [Session Management](#session-management)
8. [Tool Management](#tool-management)
9. [Skill Management](#skill-management)
10. [WebSocket Real-Time Communication](#websocket-real-time-communication)
11. [Request/Response Models](#requestresponse-models)
12. [Error Handling](#error-handling)
13. [Environment Variables](#environment-variables)
14. [Docker Deployment](#docker-deployment)

---

## 1) Authentication

All `/v1/*` endpoints (except `/v1/auth/login`) require authentication. Two methods are supported:

| Method | Header / Param | Example |
|--------|----------------|---------|
| API Key | `X-API-Key` header | `X-API-Key: sk_live_abc123...` |
| JWT Bearer Token | `Authorization` header | `Authorization: Bearer eyJhbG...` |

> **Bypass Mode:** Set `GATEWAY_AUTH_REQUIRED=false` to disable authentication entirely.

### Permission Scopes

| Scope | Description |
|-------|-------------|
| `agent:read` | Read agent status, list tools/skills/sessions |
| `agent:write` | Send messages to agent, manage sessions |
| `admin` | Execute tools directly, all permissions |

### `POST /v1/auth/login`

Authenticate and get tokens.

**Request Body:** `LoginRequest`

**Response Model:** `TokenResponse`

*Source: `spoon_bot/gateway/api/v1/auth.py:22`*

### `POST /v1/auth/refresh`

Refresh access token using refresh token.

**Request Body:** `RefreshRequest`

**Response Model:** `TokenResponse`

*Source: `spoon_bot/gateway/api/v1/auth.py:68`*

### `POST /v1/auth/logout`

Logout and invalidate refresh token.

**Request Body:** `RefreshRequest`

*Source: `spoon_bot/gateway/api/v1/auth.py:103`*

### `GET /v1/auth/verify`

Verify current authentication.

*Source: `spoon_bot/gateway/api/v1/auth.py:114`*

---

## 2) Health & Status

These endpoints do **not** require authentication.

### `GET /health`

Health check endpoint.

**Response Model:** `HealthResponse`

*Source: `spoon_bot/gateway/api/health.py:19`*

### `GET /ready`

Readiness check endpoint.


*Source: `spoon_bot/gateway/api/health.py:77`*

### `GET /`

Root endpoint with API information.


*Source: `spoon_bot/gateway/api/health.py:109`*

---

## 3) Agent Chat (HTTP)

### `POST /v1/agent/chat`

Send a message to the agent and get a response.

When options.stream=true, returns Server-Sent Events (SSE).
Otherwise returns a standard JSON response.

**Auth Required:** Yes

**Request Body:** `ChatRequest`

**Request Example:**
```json
{
  "message": "Help me write a Python script",
  "session_key": "default",
  "media": [],
  "options": {
    "max_iterations": 20,
    "stream": false,
    "thinking": false,
    "model": null
  }
}
```

**Request Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `message` | string | Yes | - | User message (1-100,000 chars) |
| `session_key` | string | No | `"default"` | Session identifier (`^[a-zA-Z0-9_-]{1,64}$`) |
| `media` | string[] | No | `[]` | Media file paths (max 10) |
| `options.max_iterations` | int | No | `20` | Max tool call iterations (1-100) |
| `options.stream` | bool | No | `false` | Enable SSE streaming (see Section 4) |
| `options.thinking` | bool | No | `false` | Include extended thinking output |
| `options.model` | string | No | `null` | Override default LLM model |

**Response (non-streaming):** `APIResponse[ChatResponse]`
```json
{
  "success": true,
  "data": {
    "response": "...",
    "tool_calls": [],
    "usage": null,
    "thinking_content": null
  },
  "meta": {
    "request_id": "req_a1b2c3d4e5f6",
    "timestamp": "2026-02-09T07:00:00.000000",
    "duration_ms": 2500
  }
}
```

*Source: `spoon_bot/gateway/api/v1/agent.py:316`*

### `POST /v1/agent/voice/chat`

Send voice + optional text to the agent (multipart upload).

**Auth Required:** Yes

**Request Example:**
```json
{
  "message": "Help me write a Python script",
  "session_key": "default",
  "media": [],
  "options": {
    "max_iterations": 20,
    "stream": false,
    "thinking": false,
    "model": null
  }
}
```

**Request Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `message` | string | Yes | - | User message (1-100,000 chars) |
| `session_key` | string | No | `"default"` | Session identifier (`^[a-zA-Z0-9_-]{1,64}$`) |
| `media` | string[] | No | `[]` | Media file paths (max 10) |
| `options.max_iterations` | int | No | `20` | Max tool call iterations (1-100) |
| `options.stream` | bool | No | `false` | Enable SSE streaming (see Section 4) |
| `options.thinking` | bool | No | `false` | Include extended thinking output |
| `options.model` | string | No | `null` | Override default LLM model |

**Response (non-streaming):** `APIResponse[ChatResponse]`
```json
{
  "success": true,
  "data": {
    "response": "...",
    "tool_calls": [],
    "usage": null,
    "thinking_content": null
  },
  "meta": {
    "request_id": "req_a1b2c3d4e5f6",
    "timestamp": "2026-02-09T07:00:00.000000",
    "duration_ms": 2500
  }
}
```

*Source: `spoon_bot/gateway/api/v1/agent.py:774`*

---

## 4) Agent Chat (SSE Streaming)

### `POST /v1/agent/chat` (with `stream: true`)

Same endpoint as non-streaming, but returns a Server-Sent Events (SSE) stream when `options.stream` is `true`.

**Request:**
```json
{ "message": "Explain quantum computing", "options": { "stream": true } }
```

**Response:** `Content-Type: text/event-stream`

```
data: {"type": "content", "delta": "Quantum ", "metadata": {}}

data: {"type": "thinking", "delta": "Let me think...", "metadata": {}}

data: [DONE]
```

**Chunk Types:**

| Type | Description |
|------|-------------|
| `content` | Response text chunk |
| `thinking` | Extended thinking content chunk |
| `tool_call` | Tool call notification |
| `error` | Error occurred during streaming |

**Response Headers:**
```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
X-Request-ID: req_a1b2c3d4e5f6
```

---

## 5) Agent Status

### `GET /v1/agent/status`

Get agent status and statistics, including channel health.

**Auth Required:** Yes

**Response Model:** `APIResponse[StatusResponse]`

The response includes `runtime_metrics` for multi-session runtime visibility:

| Field | Description |
|-------|-------------|
| `active` | Number of live session runtimes |
| `running` | Number of runtimes currently processing a task |
| `idle` | Number of live runtimes without an active task |
| `created_total` | Total runtimes created since gateway startup |
| `closed_total` | Total runtimes closed since gateway startup |
| `idle_closed_total` | Runtimes closed by idle cleanup |
| `evicted_total` | Runtimes evicted by the active runtime limit |
| `explicit_closed_total` | Runtimes closed via REST or WebSocket `session.close` |
| `idle_seconds` | Configured idle threshold; `null` means disabled |
| `max_active` | Configured active runtime limit; `null` means disabled |

*Source: `spoon_bot/gateway/api/v1/agent.py:630`*

---

## 6) Async Tasks

> **Note:** Async task endpoints are currently placeholder implementations.

### `POST /v1/agent/chat/async`

Send a message asynchronously.

*Source: `spoon_bot/gateway/api/v1/agent.py:528`*

### `GET /v1/agent/tasks/{task_id}`

Get the status of an async task.

*Source: `spoon_bot/gateway/api/v1/agent.py:564`*

### `POST /v1/agent/tasks/{task_id}/cancel`

Cancel an async task.

*Source: `spoon_bot/gateway/api/v1/agent.py:599`*

---

## 7) Session Management

### `GET /v1/sessions`

List all sessions.

**Auth Required:** Yes


*Source: `spoon_bot/gateway/api/v1/sessions.py:92`*

### `POST /v1/sessions`

Create a new session.

**Auth Required:** Yes

**Request Body:** `SessionCreateRequest`

*Source: `spoon_bot/gateway/api/v1/sessions.py:123`*

### `GET /v1/sessions/search`

Search across *all* persisted sessions for the current agent.

**Auth Required:** Yes


*Source: `spoon_bot/gateway/api/v1/sessions.py:162`*

### `GET /v1/sessions/{session_key}`

Get session details.

**Auth Required:** Yes


*Source: `spoon_bot/gateway/api/v1/sessions.py:222`*

### `GET /v1/sessions/{session_key}/search`

Search persisted messages within a single session.

**Auth Required:** Yes


*Source: `spoon_bot/gateway/api/v1/sessions.py:254`*

### `DELETE /v1/sessions/{session_key}`

Delete a session.

**Auth Required:** Yes


*Source: `spoon_bot/gateway/api/v1/sessions.py:308`*

### `POST /v1/sessions/{session_key}/close`

Close a session's in-memory runtime while keeping persisted history intact.
Use this to release tools, MCP clients, skill state, and other runtime resources
for an inactive session without deleting its conversation history. The `default`
runtime and busy runtimes are not closable through this endpoint.

**Auth Required:** Yes

**Success Response:**

```json
{
  "closed": true,
  "session_key": "alpha"
}
```

**Other Responses:**

- `{"closed": false, "reason": "not_running"}` when the session has no live runtime.
- `409 SESSION_BUSY` when the runtime is currently processing a task.
- `{"closed": false, "reason": "not_closable"}` when the runtime is protected or could not be closed.

*Source: `spoon_bot/gateway/api/v1/sessions.py:318`*

### `POST /v1/sessions/{session_key}/clear`

Clear session history.

**Auth Required:** Yes


*Source: `spoon_bot/gateway/api/v1/sessions.py:319`*

---

## 8) Tool Management

### `GET /v1/tools`

List all available tools.

**Auth Required:** Yes


*Source: `spoon_bot/gateway/api/v1/tools.py:26`*

### `GET /v1/tools/{tool_name}/schema`

Get the schema for a specific tool.

**Auth Required:** Yes


*Source: `spoon_bot/gateway/api/v1/tools.py:53`*

### `POST /v1/tools/{tool_name}/execute`

Execute a tool directly.

**Auth Required:** Yes

**Request Body:** `ToolExecuteRequest`

*Source: `spoon_bot/gateway/api/v1/tools.py:84`*

---

## 9) Skill Management

### `GET /v1/skills`

List all available skills.

**Auth Required:** Yes


*Source: `spoon_bot/gateway/api/v1/skills.py:51`*

### `POST /v1/skills/{skill_name}/activate`

Activate a skill.

**Auth Required:** Yes

**Request Body:** `SkillActivateRequest`

*Source: `spoon_bot/gateway/api/v1/skills.py:86`*

### `POST /v1/skills/{skill_name}/deactivate`

Deactivate a skill.

**Auth Required:** Yes


*Source: `spoon_bot/gateway/api/v1/skills.py:144`*

---

## 10) WebSocket Real-Time Communication

### Connection

**Endpoint:** `ws://localhost:8080/v1/ws`

**Authentication via Query Parameters:**
```
ws://localhost:8080/v1/ws?token=<jwt_access_token>
ws://localhost:8080/v1/ws?api_key=<api_key>
```

> If `GATEWAY_AUTH_REQUIRED=false`, no authentication parameters are needed.

### Message Protocol

| Type | Direction | Description |
|------|-----------|-------------|
| `request` | Client -> Server | Request |
| `response` | Server -> Client | Response |
| `error` | Server -> Client | Error |
| `event` | Server -> Client | Event |
| `stream` | Server -> Client | Stream |
| `ping` | Client -> Server | Ping |
| `pong` | Server -> Client | Pong |

### Client Methods (Client -> Server)

#### `agent.chat`

Alias of `chat.send`. Send a chat request to the agent.

Params:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | Yes | User message text. Attachments may add extra context. |
| `session_key` | string | No | Target logical session. Defaults to the connection's current session, then `default`. |
| `stream` | bool | No | When `true`, emits `agent.stream.chunk` events followed by `agent.stream.done` and `agent.complete`. |
| `thinking` | bool | No | Request thinking/reasoning chunks when supported by the provider. |
| `reasoning_effort` | string | No | Provider-specific reasoning effort. |
| `attachments` | list | No | Workspace attachment references. |
| `media` | list | No | Workspace media paths or aliases. |

Concurrency semantics:

- The same `session_key` is serialized by that session runtime's lock.
- Different `session_key` values run on independent session runtimes and may stream concurrently, even on the same WebSocket connection.
- Clients should route stream events by `request_id` and `session_key`, not by connection alone.
- Sending a new chat for the same `session_key` supersedes that session's active chat on the connection.

The server returns a normal JSON-RPC-style response after the chat finishes. For streaming chats, the incremental content arrives first as events.

#### `chat.send`

Same params and behavior as `agent.chat`.

#### `agent.cancel`

Cancel an in-flight chat request.

Params:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `session_key` | string | No | Cancel the active task for this session. If omitted, cancels the connection's current chat task. |

Cancellation is session-scoped when `session_key` is provided. Cancelling one session does not cancel other sessions streaming on the same WebSocket connection.

#### `agent.status`

Return agent status, active session runtime information, and `runtime_metrics`.

#### `session.switch`

Switch the connection's default session.

Params: `{ "session_key": "alpha" }`

`session.switch` affects subsequent requests that omit `session_key`. Requests that include `session_key` are routed to that explicit session regardless of the connection default.

#### `session.close`

Close a session's in-memory runtime while preserving persisted history.
Params: `{ "session_key": "alpha" }`. If omitted, the current connection's
session is used. Busy runtimes return `{ "closed": false, "reason": "busy" }`.
Closing the connection's current runtime moves the connection back to the
registry default session.

#### `session.clear`

Handle session clear.

#### `subscribe`

Handle event subscription (#16)

#### `unsubscribe`

Handle event unsubscription (#16)

#### `workspace.tree`

Return workspace directory tree.

#### `audio.stream.start`

Handle audio.stream.start

#### `audio.stream.end`

Handle audio.stream.end

### Server Events (Server -> Client)

| Event | Category | Description |
|-------|----------|-------------|
| `agent.thinking` | Agent | Agent Thinking |
| `agent.step` | Agent | Agent Step |
| `agent.streaming` | Agent | Agent Streaming |
| `agent.stream.chunk` | Agent | Agent Stream Chunk |
| `agent.stream.done` | Agent | Agent Stream Done |
| `agent.tool_call` | Agent | Agent Tool Call |
| `agent.tool_result` | Agent | Agent Tool Result |
| `agent.complete` | Agent | Agent Complete |
| `agent.error` | Agent | Agent Error |
| `agent.cancelled` | Agent | Agent Cancelled |
| `agent.idle` | Agent | Agent Idle |
| `confirm.request` | Confirmation | Confirm Request |
| `confirm.timeout` | Confirmation | Confirm Timeout |
| `confirm.response` | Confirmation | Confirm Response |
| `metrics.update` | Resource | Metrics Update |
| `resource.token_limit` | Resource | Resource Token Limit |
| `resource.time_limit` | Resource | Resource Time Limit |
| `audio.stream.started` | Other | Audio Stream Started |
| `audio.stream.error` | Other | Audio Stream Error |
| `audio.stream.transcription` | Other | Audio Stream Transcription |
| `connection.established` | Connection | Connection Established |
| `connection.ready` | Connection | Connection Ready |
| `connection.error` | Connection | Connection Error |
| `sandbox.stdout` | Other | Sandbox Stdout |
| `sandbox.file.changed` | Other | Sandbox File Changed |
| `term.closed` | Other | Term Closed |

### Agent Event Payloads

Agent lifecycle and streaming events include routing metadata so clients can handle multiple sessions on one WebSocket connection.

#### `agent.thinking`

```json
{
  "task_id": "task_1234abcd",
  "request_id": "client-request-id",
  "session_key": "alpha",
  "status": "processing",
  "trace_id": "trc_1234abcd"
}
```

#### `agent.stream.chunk`

```json
{
  "task_id": "task_1234abcd",
  "request_id": "client-request-id",
  "session_key": "alpha",
  "type": "content",
  "delta": "partial text",
  "metadata": {
    "provider": "openrouter",
    "channel": "text"
  },
  "trace_id": "trc_1234abcd",
  "source": {
    "kind": "agent",
    "label": "primary"
  }
}
```

`type` may be `thinking`, `content`, `tool_call`, `tool_result`, `done`, or `error` depending on provider and runtime state.

#### `agent.stream.done`

```json
{
  "task_id": "task_1234abcd",
  "request_id": "client-request-id",
  "session_key": "alpha",
  "content": "final accumulated response",
  "trace_id": "trc_1234abcd",
  "timing": {
    "elapsed_ms": 1234
  },
  "source": {
    "kind": "agent",
    "label": "primary"
  }
}
```

#### `agent.complete`

```json
{
  "task_id": "task_1234abcd",
  "request_id": "client-request-id",
  "session_key": "alpha",
  "status": "done",
  "response": "final response",
  "trace_id": "trc_1234abcd",
  "timing": {
    "elapsed_ms": 1234
  },
  "source": {
    "kind": "agent",
    "label": "primary"
  }
}
```

`agent.error` and `agent.cancelled` use the same routing fields (`task_id`, `request_id`, `session_key`, `trace_id`) plus error or cancellation details.

Deployment note: if gateway logs show `agent.thinking` or `agent.stream.chunk` without `request_id` and `session_key`, verify that the running image/process contains the multi-session runtime changes. Missing fields usually indicate an old deployment rather than a log truncation issue.

---

## 11) Request/Response Models

### Request Models

*Source: `spoon_bot/gateway/models/requests.py`*

#### `ChatOptions`

Options for chat requests.

| Field | Type | Default |
|-------|------|---------|
| `max_iterations` | `int` | Field(default=20, ge=1, le=100) |
| `stream` | `bool` | False |
| `thinking` | `bool` | False |
| `reasoning_effort` | `str | None` | None |
| `model` | `str | None` | None |

#### `ChatRequest`

Chat request model.

| Field | Type | Default |
|-------|------|---------|
| `message` | `str` | Field(default='', max_length=100000) |
| `session_key` | `str` | Field(default='default', pattern='^[a-zA-Z0-9_-]{1,64}$') |
| `media` | `list[str]` | Field(default_factory=list, max_length=10) |
| `attachments` | `list[dict[str, Any]]` | Field(default_factory=list, max_length=20) |
| `options` | `ChatOptions | None` | None |
| `audio_data` | `str | None` | Field(default=None, description='Base64-encoded audio data or data URL') |
| `audio_format` | `str | None` | Field(default=None, description='Audio format: wav, mp3, ogg, webm, flac, m4a, aac') |
| `audio_mime_type` | `str | None` | Field(default=None, description='MIME type (e.g. audio/wav). Used for format detection.') |
| `audio_language` | `str | None` | Field(default=None, description='ISO 639-1 language hint for transcription (e.g. en, zh)') |

#### `AsyncChatRequest`

Async chat request model.

| Field | Type | Default |
|-------|------|---------|
| `message` | `str` | Field(..., min_length=1, max_length=100000) |
| `session_key` | `str` | Field(default='default', pattern='^[a-zA-Z0-9_-]{1,64}$') |

#### `LoginRequest`

Login request model.

| Field | Type | Default |
|-------|------|---------|
| `username` | `str | None` | None |
| `password` | `str | None` | None |
| `api_key` | `str | None` | None |

#### `RefreshRequest`

Token refresh request model.

| Field | Type | Default |
|-------|------|---------|
| `refresh_token` | `str` | Field(..., min_length=10) |

#### `SessionCreateRequest`

Session creation request model.

| Field | Type | Default |
|-------|------|---------|
| `key` | `str` | Field(..., pattern='^[a-zA-Z0-9_-]{1,64}$') |
| `config` | `dict[str, Any] | None` | None |

#### `ToolExecuteRequest`

Tool execution request model.

| Field | Type | Default |
|-------|------|---------|
| `arguments` | `dict[str, Any]` | Field(default_factory=dict) |

#### `SkillActivateRequest`

Skill activation request model.

| Field | Type | Default |
|-------|------|---------|
| `context` | `dict[str, Any] | None` | None |

#### `MemoryAddRequest`

Memory addition request model.

| Field | Type | Default |
|-------|------|---------|
| `content` | `str` | Field(..., min_length=1, max_length=10000) |
| `tags` | `list[str]` | Field(default_factory=list, max_length=10) |

#### `ConfigUpdateRequest`

Configuration update request model.

| Field | Type | Default |
|-------|------|---------|
| `model` | `str | None` | None |
| `max_iterations` | `int | None` | Field(default=None, ge=1, le=100) |
| `shell_timeout` | `int | None` | Field(default=None, ge=1, le=3600) |
| `max_output` | `int | None` | Field(default=None, ge=100, le=1000000) |

### Response Models

*Source: `spoon_bot/gateway/models/responses.py`*

#### `MetaInfo`

Response metadata.

| Field | Type | Default |
|-------|------|---------|
| `request_id` | `str` | *(required)* |
| `timestamp` | `datetime` | Field(default_factory=datetime.utcnow) |
| `duration_ms` | `int | None` | None |
| `trace_id` | `str | None` | None |
| `timing` | `dict[str, Any] | None` | None |

#### `ErrorDetail`

Error detail model.

| Field | Type | Default |
|-------|------|---------|
| `code` | `str` | *(required)* |
| `message` | `str` | *(required)* |
| `details` | `dict[str, Any] | None` | None |
| `help_url` | `str | None` | None |

#### `APIResponse`

Standard API response wrapper.

| Field | Type | Default |
|-------|------|---------|
| `success` | `bool` | True |
| `data` | `T | None` | None |
| `meta` | `MetaInfo` | *(required)* |

#### `ErrorResponse`

Error response model.

| Field | Type | Default |
|-------|------|---------|
| `success` | `bool` | False |
| `error` | `ErrorDetail` | *(required)* |
| `meta` | `MetaInfo` | *(required)* |

#### `TokenResponse`

Token response for authentication.

| Field | Type | Default |
|-------|------|---------|
| `access_token` | `str` | *(required)* |
| `refresh_token` | `str | None` | None |
| `token_type` | `str` | 'bearer' |
| `expires_in` | `int` | *(required)* |

#### `UsageInfo`

Token usage information.

| Field | Type | Default |
|-------|------|---------|
| `prompt_tokens` | `int` | 0 |
| `completion_tokens` | `int` | 0 |
| `total_tokens` | `int` | 0 |

#### `ToolCallInfo`

Tool call information.

| Field | Type | Default |
|-------|------|---------|
| `id` | `str` | *(required)* |
| `name` | `str` | *(required)* |
| `arguments` | `dict[str, Any]` | *(required)* |
| `result` | `str | None` | None |

#### `TranscriptionInfo`

Audio transcription information.

| Field | Type | Default |
|-------|------|---------|
| `text` | `str` | *(required)* |
| `language` | `str | None` | None |
| `duration_seconds` | `float | None` | None |
| `provider` | `str` | 'whisper' |

#### `ResponseSource`

Machine-readable metadata describing who produced a response.

| Field | Type | Default |
|-------|------|---------|
| `type` | `str` | 'agent' |
| `is_subagent` | `bool` | False |
| `subagent_id` | `str | None` | None |
| `subagent_name` | `str | None` | None |

#### `ChatResponse`

Chat response model.

| Field | Type | Default |
|-------|------|---------|
| `response` | `str` | *(required)* |
| `tool_calls` | `list[ToolCallInfo]` | Field(default_factory=list) |
| `usage` | `UsageInfo | None` | None |
| `thinking_content` | `str | None` | None |
| `transcription` | `TranscriptionInfo | None` | None |
| `source` | `ResponseSource | None` | None |

#### `StreamChunk`

Streaming response chunk model.

| Field | Type | Default |
|-------|------|---------|
| `type` | `str` | *(required)* |
| `delta` | `str` | '' |
| `metadata` | `dict[str, Any]` | Field(default_factory=dict) |
| `source` | `ResponseSource | None` | None |

#### `SessionInfo`

Session information.

| Field | Type | Default |
|-------|------|---------|
| `key` | `str` | *(required)* |
| `created_at` | `datetime` | *(required)* |
| `message_count` | `int` | 0 |
| `config` | `dict[str, Any]` | Field(default_factory=dict) |

#### `SessionResponse`

Session response model.

| Field | Type | Default |
|-------|------|---------|
| `session` | `SessionInfo` | *(required)* |

#### `SessionListResponse`

Session list response model.

| Field | Type | Default |
|-------|------|---------|
| `sessions` | `list[SessionInfo]` | *(required)* |

#### `SessionSearchHit`

A single match from a session history search.

| Field | Type | Default |
|-------|------|---------|
| `session_key` | `str` | *(required)* |
| `seq` | `int` | *(required)* |
| `role` | `str` | *(required)* |
| `content` | `str` | *(required)* |
| `timestamp` | `str | None` | None |
| `matched_in` | `str` | 'content' |
| `snippet` | `str` | '' |
| `extras` | `dict[str, Any]` | Field(default_factory=dict) |

#### `SessionSearchResponse`

Session search response model.

| Field | Type | Default |
|-------|------|---------|
| `query` | `str` | *(required)* |
| `total` | `int` | *(required)* |
| `limit` | `int` | *(required)* |
| `offset` | `int` | *(required)* |
| `hits` | `list[SessionSearchHit]` | Field(default_factory=list) |

#### `ToolInfo`

Tool information.

| Field | Type | Default |
|-------|------|---------|
| `name` | `str` | *(required)* |
| `description` | `str` | *(required)* |
| `parameters` | `dict[str, Any]` | *(required)* |

#### `ToolResponse`

Tool execution response.

| Field | Type | Default |
|-------|------|---------|
| `result` | `Any` | *(required)* |
| `success` | `bool` | True |

#### `ToolListResponse`

Tool list response.

| Field | Type | Default |
|-------|------|---------|
| `tools` | `list[ToolInfo]` | *(required)* |

#### `SkillInfo`

Skill information.

| Field | Type | Default |
|-------|------|---------|
| `name` | `str` | *(required)* |
| `description` | `str` | *(required)* |
| `active` | `bool` | False |
| `triggers` | `list[str]` | Field(default_factory=list) |

#### `SkillResponse`

Skill operation response.

| Field | Type | Default |
|-------|------|---------|
| `activated` | `bool | None` | None |
| `deactivated` | `bool | None` | None |
| `skill` | `SkillInfo | None` | None |

#### `SkillListResponse`

Skill list response.

| Field | Type | Default |
|-------|------|---------|
| `skills` | `list[SkillInfo]` | *(required)* |

#### `MemoryResult`

Memory search result.

| Field | Type | Default |
|-------|------|---------|
| `id` | `str` | *(required)* |
| `content` | `str` | *(required)* |
| `tags` | `list[str]` | Field(default_factory=list) |
| `score` | `float | None` | None |
| `created_at` | `datetime | None` | None |

#### `MemoryResponse`

Memory operation response.

| Field | Type | Default |
|-------|------|---------|
| `id` | `str | None` | None |
| `created` | `bool | None` | None |
| `context` | `str | None` | None |
| `results` | `list[MemoryResult] | None` | None |

#### `AgentStats`

Agent statistics.

| Field | Type | Default |
|-------|------|---------|
| `total_requests` | `int` | 0 |
| `active_sessions` | `int` | 0 |
| `tools_available` | `int` | 0 |
| `skills_loaded` | `int` | 0 |

#### `ChannelStatusInfo`

Status of a single channel.

| Field | Type | Default |
|-------|------|---------|
| `name` | `str` | *(required)* |
| `status` | `str` | *(required)* |
| `message` | `str | None` | None |

#### `ChannelsInfo`

Aggregate channels status.

| Field | Type | Default |
|-------|------|---------|
| `running` | `int` | 0 |
| `total` | `int` | 0 |
| `channels` | `list[ChannelStatusInfo]` | Field(default_factory=list) |

#### `StatusResponse`

Agent status response.

| Field | Type | Default |
|-------|------|---------|
| `status` | `str` | *(required)* |
| `current_task` | `str | None` | None |
| `uptime` | `int` | *(required)* |
| `stats` | `AgentStats` | *(required)* |
| `channels` | `ChannelsInfo | None` | None |

#### `HealthCheck`

Health check item.

| Field | Type | Default |
|-------|------|---------|
| `name` | `str` | *(required)* |
| `status` | `str` | *(required)* |
| `message` | `str | None` | None |

#### `HealthResponse`

Health check response.

| Field | Type | Default |
|-------|------|---------|
| `status` | `str` | *(required)* |
| `version` | `str` | *(required)* |
| `uptime` | `int` | *(required)* |
| `checks` | `list[HealthCheck]` | Field(default_factory=list) |

---

## 12) Error Handling

### Standard API Error Response

```json
{
  "detail": {
    "code": "ERROR_CODE",
    "message": "Human-readable description"
  }
}
```

### HTTP Status Codes

| Status | Description |
|--------|-------------|
| 200 | Success |
| 401 | Authentication required or invalid |
| 403 | Insufficient permissions |
| 404 | Resource not found |
| 409 | Conflict (e.g., session already exists) |
| 422 | Validation error |
| 500 | Internal server error |

### Common Error Codes

| Code | Description |
|------|-------------|
| `AUTH_REQUIRED` | Missing authentication credentials |
| `AUTH_INVALID` | Invalid credentials |
| `FORBIDDEN` | Insufficient scope for this operation |
| `NOT_FOUND` | Requested resource not found |
| `SESSION_EXISTS` | Session with this key already exists |
| `AGENT_ERROR` | Agent processing failure |
| `INTERNAL_ERROR` | Unexpected internal error |
| `ACTIVATION_FAILED` | Skill activation failed |

---

## 13) Environment Variables

| Variable | Default | Source |
|----------|---------|--------|
| `SESSION_STORE_BACKEND` | `file` | server.py |
| `SESSION_STORE_DSN` | *(none)* | server.py |
| `SESSION_STORE_DB_PATH` | *(none)* | server.py |
| `SPOON_BOT_YOLO_MODE` | *(none)* | server.py |
| `GATEWAY_API_KEY` | *(none)* | server.py |
| `GATEWAY_AUTH_REQUIRED` | `true` | server.py |
| `JWT_SECRET` | *(none)* | config.py |
| `GATEWAY_HOST` | `127.0.0.1` | config.py |
| `GATEWAY_PORT` | `8080` | config.py |
| `GATEWAY_DEBUG` | *(none)* | config.py |
| `JWT_ACCESS_EXPIRE_MINUTES` | `15` | config.py |
| `GATEWAY_TIMEOUT_REQUEST_MS` | `0` | config.py |
| `GATEWAY_TIMEOUT_TOOL_MS` | `3600000` | config.py |
| `GATEWAY_TIMEOUT_STREAM_MS` | `0` | config.py |
| `GATEWAY_AUDIO_ENABLED` | `true` | config.py |
| `GATEWAY_AUDIO_STT_PROVIDER` | `whisper` | config.py |
| `GATEWAY_AUDIO_STT_MODEL` | `whisper-1` | config.py |
| `GATEWAY_AUDIO_DEFAULT_LANGUAGE` | *(none)* | config.py |
| `GATEWAY_AUDIO_STREAMING` | `true` | config.py |
| `GATEWAY_AUDIO_NATIVE_PROVIDERS` | `openai,gemini` | config.py |
| `SPOON_BOT_SESSION_RUNTIME_IDLE_SECONDS` | `1800` | session_registry.py |
| `SPOON_BOT_SESSION_RUNTIME_MAX_ACTIVE` | `64` | session_registry.py |

### Default Models Per Provider

| Provider | Default Model |
|----------|---------------|

---

## 14) Docker Deployment

### Quick Start

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 2. Build and start
docker compose up -d

# 3. Verify
curl http://localhost:8080/health
curl http://localhost:8080/ready
```

### Run Modes

| Mode | Description |
|------|-------------|
| `gateway` | HTTP/WebSocket API server (default) |
| `agent` | One-shot agent execution |
| `cli` | CLI mode |
| `onboard` | Initialize workspace |

Set mode via `SPOON_BOT_MODE` environment variable.

---

## API Endpoint Summary

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | No | Health check endpoint. |
| `GET` | `/ready` | No | Readiness check endpoint. |
| `GET` | `/` | No | Root endpoint with API information. |
| `POST` | `/v1/auth/login` | No | Authenticate and get tokens. |
| `POST` | `/v1/auth/refresh` | No | Refresh access token using refresh token. |
| `POST` | `/v1/auth/logout` | No | Logout and invalidate refresh token. |
| `GET` | `/v1/auth/verify` | Yes | Verify current authentication. |
| `POST` | `/v1/agent/chat` | Yes | Send a message to the agent and get a response. |
| `POST` | `/v1/agent/chat/async` | Yes | Send a message asynchronously. |
| `GET` | `/v1/agent/tasks/{task_id}` | Yes | Get the status of an async task. |
| `POST` | `/v1/agent/tasks/{task_id}/cancel` | Yes | Cancel an async task. |
| `GET` | `/v1/agent/status` | Yes | Get agent status and statistics, including channel health. |
| `POST` | `/v1/agent/voice/transcribe` | Yes | Transcribe an audio file to text (STT only, no agent process |
| `POST` | `/v1/agent/voice/chat` | Yes | Send voice + optional text to the agent (multipart upload). |
| `GET` | `/v1/sessions` | Yes | List all sessions. |
| `POST` | `/v1/sessions` | Yes | Create a new session. |
| `GET` | `/v1/sessions/search` | Yes | Search across *all* persisted sessions for the current agent |
| `GET` | `/v1/sessions/{session_key}` | Yes | Get session details. |
| `GET` | `/v1/sessions/{session_key}/search` | Yes | Search persisted messages within a single session. |
| `DELETE` | `/v1/sessions/{session_key}` | Yes | Delete a session. |
| `POST` | `/v1/sessions/{session_key}/clear` | Yes | Clear session history. |
| `GET` | `/v1/tools` | Yes | List all available tools. |
| `GET` | `/v1/tools/{tool_name}/schema` | Yes | Get the schema for a specific tool. |
| `POST` | `/v1/tools/{tool_name}/execute` | Yes | Execute a tool directly. |
| `GET` | `/v1/skills` | Yes | List all available skills. |
| `POST` | `/v1/skills/{skill_name}/activate` | Yes | Activate a skill. |
| `POST` | `/v1/skills/{skill_name}/deactivate` | Yes | Deactivate a skill. |
| `WS` | `/v1/ws` | Query param | Real-time bidirectional communication |

---

## Notes for Frontend

- Use **WebSocket** (`/v1/ws`) for real-time streaming UI with bidirectional communication.
- Use **HTTP SSE** (`POST /v1/agent/chat` with `stream: true`) for simpler streaming without WebSocket.
- Use **HTTP POST** (`POST /v1/agent/chat`) for simple request/response (no streaming).
- When `GATEWAY_AUTH_REQUIRED=false`, all endpoints work without authentication.
- Session keys must match pattern `^[a-zA-Z0-9_-]{1,64}$`.
- Messages are sanitized: control characters (except `\n`, `\r`, `\t`) are stripped.
- OpenAPI docs are available at `/docs` (Swagger UI) and `/redoc` (ReDoc).
