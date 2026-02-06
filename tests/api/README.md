# spoon-bot Gateway API Integration Tests

Comprehensive JavaScript integration tests for the spoon-bot Gateway REST API.

## Endpoints Covered

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root API info |
| `/health` | GET | Health check |
| `/ready` | GET | Readiness check |
| `/docs` | GET | OpenAPI documentation |
| `/v1/auth/login` | POST | Authenticate with API key |
| `/v1/auth/refresh` | POST | Refresh access token |
| `/v1/auth/logout` | POST | Invalidate refresh token |
| `/v1/auth/verify` | GET | Verify current token |
| `/v1/agent/chat` | POST | Send message to agent |
| `/v1/agent/chat/async` | POST | Async message (placeholder) |
| `/v1/agent/status` | GET | Agent status & stats |
| `/v1/agent/tasks/:id` | GET | Get async task status |
| `/v1/agent/tasks/:id/cancel` | POST | Cancel async task |
| `/v1/sessions` | GET | List sessions |
| `/v1/sessions` | POST | Create session |
| `/v1/sessions/:key` | GET | Get session details |
| `/v1/sessions/:key` | DELETE | Delete session |
| `/v1/sessions/:key/clear` | POST | Clear session history |
| `/v1/tools` | GET | List available tools |
| `/v1/tools/:name/schema` | GET | Get tool schema |
| `/v1/skills` | GET | List skills |
| `/v1/skills/:name/activate` | POST | Activate skill |
| `/v1/ws` | WS | WebSocket chat |

## Requirements

- Node.js 18+
- Docker with BuildKit
- Running spoon-bot container

## Quick Start

```bash
# Run full test suite (build, start container, test, cleanup)
./run-tests.sh

# Skip Docker build (use existing image)
./run-tests.sh --skip-build

# Keep container running after tests
./run-tests.sh --keep-running
```

## Manual Testing

```bash
# 1. Build and start container
cd /path/to/spoon-bot
DOCKER_BUILDKIT=1 sudo docker build -t spoon-bot:test .
sudo docker run -d --name spoon-bot-test -p 18080:8080 \
  -e GATEWAY_AUTH_REQUIRED=false \
  -e OPENAI_API_KEY=your-key \
  spoon-bot:test

# 2. Run tests
cd tests/api
npm install
SPOON_BOT_URL=http://localhost:18080 AUTH_REQUIRED=false npm test

# 3. Cleanup
sudo docker rm -f spoon-bot-test
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SPOON_BOT_URL` | `http://localhost:8080` | Gateway URL |
| `SPOON_BOT_API_KEY` | (empty) | API key for authenticated tests |
| `AUTH_REQUIRED` | `true` | Skip auth-protected tests if `false` |
| `GATEWAY_PORT` | `18080` | Port for test container |

## Test Categories

### 1. Health & Root (5 tests)
Basic health checks and API info endpoints.

### 2. Authentication (9 tests, 4 skipped without API key)
JWT token flow, API key validation, token refresh.

### 3. Agent Chat (5 tests)
Message handling, session keys, input validation.

### 4. Agent Status (2 tests)
Agent stats and current task info.

### 5. Sessions (8 tests)
Session CRUD operations, validation.

> ⚠️ **Known Bug**: `SessionManager` may lack `get()` and `list_sessions()` methods, causing 500 errors.

### 6. Tools (3 tests)
Tool listing and schema retrieval.

### 7. Skills (2 tests)
Skill listing and activation.

> ⚠️ **Known Bug**: Skills interface may differ from expected, causing 500 errors.

### 8. Async Tasks (3 tests)
Placeholder async chat endpoints.

### 9. WebSocket (1 test)
WebSocket connection and message exchange.

### 10. Edge Cases & Security (5 tests)
Large payloads, invalid JSON, SQL injection, XSS handling.

### 11. Response Format (2 tests)
API response structure validation.

## Known Issues

The tests document several bugs in the current gateway implementation:

1. **SessionManager.get() missing** - Causes 500 on session detail endpoints
2. **SessionManager.list_sessions() missing** - Causes 500 on session list
3. **Skills interface mismatch** - Skills activate may crash

These are logged with ⚠️ warnings in test output.

## Adding Tests

Tests use Jest with axios for HTTP and ws for WebSocket:

```javascript
test('new endpoint test', async () => {
  const headers = tokens ? authHeaders(tokens.accessToken) : {};
  const res = await api.get('/v1/new-endpoint', { headers });
  expect(res.status).toBe(200);
  expect(res.data).toHaveProperty('expected_field');
});
```
