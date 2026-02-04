# spoon-bot WebSocket & REST API Architecture

## Overview

This document describes the complete WebSocket and REST API architecture for spoon-bot, enabling remote agent control, real-time communication, and multi-client support.

## Design Principles

1. **Security First**: JWT-based authentication with token refresh
2. **Real-time**: WebSocket for streaming responses and events
3. **RESTful**: Standard REST API for CRUD operations
4. **Extensible**: Plugin architecture for custom endpoints
5. **Observable**: Built-in metrics and health checks

---

## 1. Authentication Architecture

### 1.1 Authentication Flow

```
┌─────────┐     1. Login      ┌──────────┐
│  Client │ ─────────────────→│  /auth   │
└─────────┘                   │  /login  │
     │                        └────┬─────┘
     │                             │
     │   2. JWT Tokens             │
     │ ←───────────────────────────┘
     │   (access_token + refresh_token)
     │
     │   3. API Request + Bearer Token
     │ ─────────────────────────────────→ /api/*
     │
     │   4. WebSocket + Token
     └─────────────────────────────────→ /ws?token=xxx
```

### 1.2 Token Structure

```python
# Access Token (15 min expiry)
{
    "sub": "user_id",           # User identifier
    "session": "session_key",   # Agent session key
    "iat": 1234567890,          # Issued at
    "exp": 1234568790,          # Expiration
    "type": "access",
    "scope": ["agent:read", "agent:write", "admin"]
}

# Refresh Token (7 day expiry)
{
    "sub": "user_id",
    "jti": "unique_token_id",   # For revocation
    "iat": 1234567890,
    "exp": 1235172690,
    "type": "refresh"
}
```

### 1.3 Authentication Endpoints

```
POST /auth/login
    Request:  { "username": "...", "password": "...", "api_key": "..." }
    Response: { "access_token": "...", "refresh_token": "...", "expires_in": 900 }

POST /auth/refresh
    Request:  { "refresh_token": "..." }
    Response: { "access_token": "...", "expires_in": 900 }

POST /auth/logout
    Request:  { "refresh_token": "..." }
    Response: { "success": true }

GET /auth/verify
    Header:   Authorization: Bearer <access_token>
    Response: { "valid": true, "user_id": "...", "expires_at": "..." }
```

### 1.4 API Key Authentication (Alternative)

```
Header: X-API-Key: sk_live_xxxxxxxxxxxxx

# Key format
sk_<env>_<32_char_random>
env: live | test | dev
```

---

## 2. REST API Architecture

### 2.1 Base URL Structure

```
Production:  https://api.spoon-bot.ai/v1
Development: http://localhost:8080/v1
```

### 2.2 Standard Response Format

```python
# Success Response
{
    "success": true,
    "data": { ... },
    "meta": {
        "request_id": "req_xxx",
        "timestamp": "2024-01-01T00:00:00Z",
        "duration_ms": 123
    }
}

# Error Response
{
    "success": false,
    "error": {
        "code": "INVALID_REQUEST",
        "message": "Human readable error message",
        "details": { ... },  # Optional
        "help_url": "https://docs.spoon-bot.ai/errors/INVALID_REQUEST"
    },
    "meta": {
        "request_id": "req_xxx",
        "timestamp": "2024-01-01T00:00:00Z"
    }
}
```

### 2.3 Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `AUTH_REQUIRED` | 401 | Authentication required |
| `AUTH_INVALID` | 401 | Invalid credentials or token |
| `AUTH_EXPIRED` | 401 | Token expired |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `VALIDATION_ERROR` | 400 | Request validation failed |
| `RATE_LIMITED` | 429 | Rate limit exceeded |
| `AGENT_BUSY` | 409 | Agent is processing another request |
| `AGENT_ERROR` | 500 | Agent execution error |
| `INTERNAL_ERROR` | 500 | Internal server error |

### 2.4 REST Endpoints

#### Sessions

```
# List sessions
GET /v1/sessions
    Response: { "sessions": [{ "key": "...", "created_at": "...", "message_count": 10 }] }

# Create session
POST /v1/sessions
    Request:  { "key": "my-session", "config": { "model": "claude-3-sonnet" } }
    Response: { "session": { "key": "...", "created_at": "..." } }

# Get session
GET /v1/sessions/{session_key}
    Response: { "session": { "key": "...", "messages": [...], "config": {...} } }

# Delete session
DELETE /v1/sessions/{session_key}
    Response: { "deleted": true }

# Clear session history
POST /v1/sessions/{session_key}/clear
    Response: { "cleared": true, "messages_removed": 42 }
```

#### Agent Control

```
# Send message (sync - waits for response)
POST /v1/agent/chat
    Request:  {
        "message": "Hello, help me with...",
        "session_key": "default",
        "media": ["base64_encoded_image"],
        "options": {
            "max_iterations": 10,
            "stream": false
        }
    }
    Response: {
        "response": "Here's how I can help...",
        "tool_calls": [...],
        "usage": { "prompt_tokens": 100, "completion_tokens": 50 }
    }

# Send message (async - returns task ID)
POST /v1/agent/chat/async
    Request:  { "message": "...", "session_key": "default" }
    Response: { "task_id": "task_xxx", "status": "pending" }

# Get task status
GET /v1/agent/tasks/{task_id}
    Response: {
        "task_id": "task_xxx",
        "status": "completed|processing|failed",
        "progress": 0.75,
        "result": { ... }
    }

# Cancel task
POST /v1/agent/tasks/{task_id}/cancel
    Response: { "cancelled": true }

# Get agent status
GET /v1/agent/status
    Response: {
        "status": "ready|busy|error",
        "current_task": "task_xxx",
        "uptime": 3600,
        "stats": {
            "total_requests": 100,
            "active_sessions": 5
        }
    }
```

#### Tools

```
# List available tools
GET /v1/tools
    Response: { "tools": [{ "name": "shell", "description": "...", "parameters": {...} }] }

# Execute tool directly (admin only)
POST /v1/tools/{tool_name}/execute
    Request:  { "arguments": { "command": "ls -la" } }
    Response: { "result": "...", "success": true }

# Get tool schema
GET /v1/tools/{tool_name}/schema
    Response: { "schema": { "type": "function", "function": {...} } }
```

#### Skills

```
# List skills
GET /v1/skills
    Response: { "skills": [{ "name": "research", "description": "...", "active": false }] }

# Activate skill
POST /v1/skills/{skill_name}/activate
    Request:  { "context": { "topic": "AI" } }
    Response: { "activated": true, "skill": {...} }

# Deactivate skill
POST /v1/skills/{skill_name}/deactivate
    Response: { "deactivated": true }
```

#### Memory

```
# Search memory
GET /v1/memory/search?query=xxx&limit=10
    Response: { "results": [...] }

# Add memory
POST /v1/memory
    Request:  { "content": "Important fact...", "tags": ["user-pref"] }
    Response: { "id": "mem_xxx", "created": true }

# Get memory context
GET /v1/memory/context
    Response: { "context": "Summarized memory context..." }
```

#### Configuration

```
# Get config
GET /v1/config
    Response: { "config": { "model": "...", "max_iterations": 20 } }

# Update config
PATCH /v1/config
    Request:  { "model": "claude-3-opus", "max_iterations": 30 }
    Response: { "config": {...}, "updated_fields": ["model", "max_iterations"] }
```

#### Health & Metrics

```
# Health check
GET /health
    Response: { "status": "healthy", "version": "1.0.0", "uptime": 3600 }

# Readiness check
GET /ready
    Response: { "ready": true, "checks": { "database": true, "llm": true } }

# Metrics (Prometheus format)
GET /metrics
    Response: # HELP spoon_bot_requests_total ...
```

---

## 3. WebSocket Architecture

### 3.1 Connection

```javascript
// Connect with token
const ws = new WebSocket('wss://api.spoon-bot.ai/v1/ws?token=<access_token>');

// Or with API key
const ws = new WebSocket('wss://api.spoon-bot.ai/v1/ws?api_key=sk_live_xxx');
```

### 3.2 Message Format (JSON-RPC 2.0 inspired)

```python
# Client → Server (Request)
{
    "id": "msg_123",           # Unique message ID
    "type": "request",
    "method": "agent.chat",    # Method name
    "params": {                # Method parameters
        "message": "Hello!",
        "session_key": "default"
    }
}

# Server → Client (Response)
{
    "id": "msg_123",           # Matches request ID
    "type": "response",
    "result": {
        "content": "Hi! How can I help?",
        "tool_calls": []
    }
}

# Server → Client (Error)
{
    "id": "msg_123",
    "type": "error",
    "error": {
        "code": "AGENT_ERROR",
        "message": "Failed to process request"
    }
}

# Server → Client (Event - no ID)
{
    "type": "event",
    "event": "agent.thinking",
    "data": {
        "status": "calling_tool",
        "tool": "shell",
        "iteration": 3
    }
}

# Server → Client (Stream chunk)
{
    "id": "msg_123",
    "type": "stream",
    "chunk": "Here's ",
    "done": false
}
```

### 3.3 WebSocket Methods

#### Agent Methods

```python
# Chat (streaming)
{
    "method": "agent.chat",
    "params": {
        "message": "...",
        "session_key": "default",
        "stream": true,          # Enable streaming
        "media": []
    }
}

# Cancel current operation
{
    "method": "agent.cancel",
    "params": {}
}

# Get status
{
    "method": "agent.status",
    "params": {}
}
```

#### Session Methods

```python
# Switch session
{
    "method": "session.switch",
    "params": { "session_key": "new-session" }
}

# List sessions
{
    "method": "session.list",
    "params": {}
}

# Clear history
{
    "method": "session.clear",
    "params": { "session_key": "default" }
}
```

#### Subscription Methods

```python
# Subscribe to events
{
    "method": "subscribe",
    "params": {
        "events": ["agent.thinking", "agent.tool_call", "agent.error"]
    }
}

# Unsubscribe
{
    "method": "unsubscribe",
    "params": {
        "events": ["agent.thinking"]
    }
}
```

### 3.4 Server Events

| Event | Description | Data |
|-------|-------------|------|
| `agent.thinking` | Agent is processing | `{ status, iteration }` |
| `agent.tool_call` | Tool being called | `{ tool, arguments }` |
| `agent.tool_result` | Tool execution result | `{ tool, result, success }` |
| `agent.streaming` | Streaming text chunk | `{ chunk, done }` |
| `agent.complete` | Processing complete | `{ result, usage }` |
| `agent.error` | Error occurred | `{ code, message }` |
| `session.updated` | Session state changed | `{ session_key, ... }` |
| `memory.updated` | Memory was updated | `{ action, content }` |
| `connection.ping` | Keep-alive ping | `{ timestamp }` |

### 3.5 Connection Management

```python
# Client sends ping every 30s
{ "type": "ping", "timestamp": 1234567890 }

# Server responds with pong
{ "type": "pong", "timestamp": 1234567890 }

# Server sends heartbeat event
{
    "type": "event",
    "event": "connection.heartbeat",
    "data": { "timestamp": 1234567890, "latency_ms": 5 }
}
```

---

## 4. Implementation Architecture

### 4.1 Module Structure

```
spoon_bot/
├── gateway/
│   ├── __init__.py
│   ├── app.py              # FastAPI application factory
│   ├── config.py           # Gateway configuration
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── jwt.py          # JWT token handling
│   │   ├── api_key.py      # API key validation
│   │   └── middleware.py   # Auth middleware
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── router.py   # Main router
│   │   │   ├── agent.py    # Agent endpoints
│   │   │   ├── sessions.py # Session endpoints
│   │   │   ├── tools.py    # Tool endpoints
│   │   │   ├── skills.py   # Skill endpoints
│   │   │   ├── memory.py   # Memory endpoints
│   │   │   └── config.py   # Config endpoints
│   │   └── health.py       # Health endpoints
│   ├── websocket/
│   │   ├── __init__.py
│   │   ├── handler.py      # WebSocket connection handler
│   │   ├── manager.py      # Connection manager
│   │   ├── protocol.py     # Message protocol
│   │   └── events.py       # Event types
│   ├── models/
│   │   ├── __init__.py
│   │   ├── requests.py     # Request models (Pydantic)
│   │   ├── responses.py    # Response models
│   │   └── errors.py       # Error models
│   └── middleware/
│       ├── __init__.py
│       ├── cors.py         # CORS handling
│       ├── rate_limit.py   # Rate limiting
│       └── logging.py      # Request logging
```

### 4.2 Core Classes

```python
# gateway/websocket/manager.py
class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self._connections: dict[str, WebSocket] = {}
        self._subscriptions: dict[str, set[str]] = {}

    async def connect(self, websocket: WebSocket, user_id: str) -> str:
        """Accept connection and return connection ID."""

    async def disconnect(self, connection_id: str) -> None:
        """Close and cleanup connection."""

    async def send_to_user(self, user_id: str, message: dict) -> None:
        """Send message to specific user."""

    async def broadcast_event(self, event: str, data: dict) -> None:
        """Broadcast event to subscribed connections."""

    async def subscribe(self, connection_id: str, events: list[str]) -> None:
        """Subscribe connection to events."""

# gateway/websocket/handler.py
class WebSocketHandler:
    """Handles WebSocket message processing."""

    def __init__(self, agent: AgentLoop, manager: ConnectionManager):
        self._agent = agent
        self._manager = manager
        self._handlers = {
            "agent.chat": self._handle_chat,
            "agent.cancel": self._handle_cancel,
            "session.switch": self._handle_session_switch,
            "subscribe": self._handle_subscribe,
        }

    async def handle_message(self, conn_id: str, message: dict) -> dict:
        """Route and handle incoming message."""

    async def _handle_chat(self, conn_id: str, params: dict) -> AsyncGenerator:
        """Handle chat with streaming support."""
```

### 4.3 WebSocket Channel Integration

```python
# channels/websocket_channel.py
class WebSocketChannel(BaseChannel):
    """WebSocket channel for gateway integration."""

    def __init__(self, connection_id: str, websocket: WebSocket):
        super().__init__(f"ws_{connection_id}")
        self._ws = websocket
        self._connection_id = connection_id

    async def start(self) -> None:
        """Start receiving messages."""

    async def stop(self) -> None:
        """Close WebSocket connection."""

    async def send(self, message: OutboundMessage) -> None:
        """Send message through WebSocket."""
        await self._ws.send_json({
            "type": "response",
            "id": message.reply_to,
            "result": {"content": message.content}
        })
```

---

## 5. Security Considerations

### 5.1 Rate Limiting

```python
# Per-user limits
RATE_LIMITS = {
    "chat": {"requests": 60, "window": "1m"},      # 60 req/min
    "tools": {"requests": 30, "window": "1m"},     # 30 req/min
    "auth": {"requests": 5, "window": "1m"},       # 5 req/min
    "websocket_messages": {"requests": 100, "window": "1m"}
}
```

### 5.2 Input Validation

```python
# Request validation using Pydantic
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=100000)
    session_key: str = Field(default="default", pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    media: list[str] = Field(default_factory=list, max_items=10)
    options: ChatOptions | None = None

    @validator("message")
    def sanitize_message(cls, v):
        # Remove control characters except newlines
        return "".join(c for c in v if c.isprintable() or c in "\n\r\t")
```

### 5.3 Token Security

1. Access tokens: Short-lived (15 min), stateless
2. Refresh tokens: Stored server-side, revocable
3. API keys: Hashed storage, prefix-based identification
4. WebSocket: Re-authenticate on reconnect

### 5.4 CORS Configuration

```python
CORS_CONFIG = {
    "allow_origins": ["https://app.spoon-bot.ai"],
    "allow_methods": ["GET", "POST", "PUT", "DELETE", "PATCH"],
    "allow_headers": ["Authorization", "X-API-Key", "Content-Type"],
    "allow_credentials": True,
    "max_age": 600
}
```

---

## 6. Usage Examples

### 6.1 REST API (Python)

```python
import httpx

class SpoonBotClient:
    def __init__(self, api_key: str, base_url: str = "https://api.spoon-bot.ai/v1"):
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={"X-API-Key": api_key}
        )

    async def chat(self, message: str, session_key: str = "default") -> str:
        response = await self.client.post("/agent/chat", json={
            "message": message,
            "session_key": session_key
        })
        response.raise_for_status()
        return response.json()["data"]["response"]

# Usage
client = SpoonBotClient("sk_live_xxx")
response = await client.chat("Hello!")
```

### 6.2 WebSocket (JavaScript)

```javascript
class SpoonBotWS {
    constructor(token) {
        this.ws = new WebSocket(`wss://api.spoon-bot.ai/v1/ws?token=${token}`);
        this.pending = new Map();
        this.messageId = 0;

        this.ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.id && this.pending.has(msg.id)) {
                this.pending.get(msg.id)(msg);
                this.pending.delete(msg.id);
            } else if (msg.type === 'event') {
                this.onEvent?.(msg.event, msg.data);
            }
        };
    }

    async chat(message, options = {}) {
        const id = `msg_${++this.messageId}`;
        return new Promise((resolve) => {
            this.pending.set(id, resolve);
            this.ws.send(JSON.stringify({
                id,
                type: 'request',
                method: 'agent.chat',
                params: { message, ...options }
            }));
        });
    }

    subscribe(events, callback) {
        this.onEvent = callback;
        this.ws.send(JSON.stringify({
            type: 'request',
            method: 'subscribe',
            params: { events }
        }));
    }
}

// Usage
const client = new SpoonBotWS(accessToken);
client.subscribe(['agent.thinking', 'agent.streaming'], (event, data) => {
    console.log(event, data);
});
const response = await client.chat("Hello!");
```

### 6.3 curl Examples

```bash
# Login
curl -X POST https://api.spoon-bot.ai/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"api_key": "sk_live_xxx"}'

# Chat
curl -X POST https://api.spoon-bot.ai/v1/agent/chat \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "session_key": "default"}'

# List tools
curl https://api.spoon-bot.ai/v1/tools \
  -H "X-API-Key: sk_live_xxx"
```

---

## 7. Deployment Architecture

```
                    ┌─────────────────┐
                    │   Load Balancer │
                    │  (nginx/traefik)│
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
        │  Gateway  │  │  Gateway  │  │  Gateway  │
        │ Instance 1│  │ Instance 2│  │ Instance 3│
        └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │  Redis Cluster  │
                    │ (sessions/cache)│
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
        │  AgentLoop│  │  AgentLoop│  │  AgentLoop│
        │  Worker 1 │  │  Worker 2 │  │  Worker 3 │
        └───────────┘  └───────────┘  └───────────┘
```

---

## 8. Implementation Plan

### Phase 1: Core REST API (Week 1-2)
- [ ] Authentication endpoints
- [ ] Agent chat endpoint (sync)
- [ ] Session management
- [ ] Health endpoints

### Phase 2: WebSocket Support (Week 2-3)
- [ ] Connection manager
- [ ] Message protocol
- [ ] Streaming chat
- [ ] Event subscriptions

### Phase 3: Advanced Features (Week 3-4)
- [ ] Async task queue
- [ ] Tool/skill endpoints
- [ ] Memory API
- [ ] Configuration API

### Phase 4: Production Readiness (Week 4-5)
- [ ] Rate limiting
- [ ] Metrics/logging
- [ ] Documentation
- [ ] Integration tests

---

## Appendix A: Status Codes

| HTTP Code | Meaning |
|-----------|---------|
| 200 | Success |
| 201 | Created |
| 204 | No Content |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 409 | Conflict |
| 429 | Rate Limited |
| 500 | Server Error |
| 503 | Service Unavailable |

## Appendix B: WebSocket Close Codes

| Code | Meaning |
|------|---------|
| 1000 | Normal closure |
| 1001 | Going away |
| 1008 | Policy violation |
| 4000 | Authentication required |
| 4001 | Authentication failed |
| 4002 | Token expired |
| 4003 | Rate limited |
| 4004 | Invalid message format |
