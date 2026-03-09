# spoon-bot

Local-first AI agent with native OS tools, powered by spoon-core.

## Features

- **Multi-Provider LLM**: Supports Anthropic, OpenAI, DeepSeek, Gemini, OpenRouter (400+ models), and more
- **Agent-centric**: Autonomous execution with safety rails and dynamic tool loading
- **OS-native**: Built-in shell/filesystem tools as priority
- **Memory-first**: Four-layer memory system (file + semantic search via memsearch + short-term + checkpointer)
- **Session Persistence**: Pluggable backends — JSONL files (default), SQLite, or PostgreSQL
- **Web Search**: Built-in Tavily integration for real-time information retrieval
- **Self-managing**: Self-configuration, self-upgrade, memory management tools
- **Web3-enabled**: Blockchain operations via spoon-core and spoon-toolkits
- **Extensible**: MCP servers + Skills ecosystem with dynamic tool activation
- **Multi-channel**: Telegram bot integration with polling/webhook modes
- **Multi-mode**: Agent / Interactive / Gateway (REST + WebSocket)

## Requirements

- **Python 3.12+** (required by spoon-ai-sdk)

## Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/XSpoonAi/spoon-bot.git
cd spoon-bot

# Install with uv (creates .venv automatically)
uv sync

# Or with all extras
uv sync --all-extras
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/XSpoonAi/spoon-bot.git
cd spoon-bot

# Create virtual environment (Python 3.12+ required)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install
pip install -e .

# Or with all extras
pip install -e ".[all]"
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure your API keys:

```bash
cp .env.example .env
```

**At least one LLM API key is required:**

| Provider | Environment Variable | Get API Key |
|----------|---------------------|-------------|
| Anthropic | `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/) |
| OpenAI | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/) |
| DeepSeek | `DEEPSEEK_API_KEY` | [platform.deepseek.com](https://platform.deepseek.com/) |
| Google Gemini | `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com/) |
| OpenRouter | `OPENROUTER_API_KEY` | [openrouter.ai](https://openrouter.ai/) |
| Tavily (web search) | `TAVILY_API_KEY` | [tavily.com](https://tavily.com/) |

Example `.env` file:

```bash
# Use any ONE of these (or multiple for fallback)
ANTHROPIC_API_KEY=sk-ant-xxx
OPENAI_API_KEY=sk-xxx
DEEPSEEK_API_KEY=xxx
GEMINI_API_KEY=xxx
OPENROUTER_API_KEY=sk-or-xxx

# Optional: web search
TAVILY_API_KEY=tvly-xxx
```

### Channel Configuration (config.yaml)

To enable Telegram or other channel integrations, create a `config.yaml` in the project root:

```yaml
agent:
  provider: anthropic
  model: claude-sonnet-4-20250514
  # api_key: sk-xxx  # Or use environment variable

channels:
  telegram:
    enabled: true
    accounts:
      - name: my_bot
        token: ${TELEGRAM_BOT_TOKEN}   # Reference env var
        mode: polling                   # polling (default) or webhook
        allowed_users: [123456789]      # Optional: restrict to specific user IDs
        proxy_url: "http://127.0.0.1:7897"  # Optional: for restricted networks
        groups:
          enabled: false
          require_mention: true
```

**Telegram setup:**

1. Create a bot via [@BotFather](https://t.me/BotFather) and get the token
2. Set `TELEGRAM_BOT_TOKEN` in `.env` or directly in `config.yaml`
3. Install the Telegram dependency: `uv sync --extra telegram`

**Proxy configuration** (for networks that cannot access Telegram API directly):

The proxy is resolved in priority order: `config.yaml proxy_url` > `TELEGRAM_PROXY` env var > `HTTPS_PROXY` env var. Users with unrestricted network access do not need this.

## Quick Start

### Using Anthropic Claude (Default)

```bash
export ANTHROPIC_API_KEY=your-key

# Initialize workspace
spoon-bot onboard

# Run in interactive mode
spoon-bot agent

# Run in one-shot mode
spoon-bot agent -m "List files in the current directory"
```

### Using Other Providers

```bash
# OpenAI GPT-5.2
export OPENAI_API_KEY=your-key
spoon-bot agent --provider openai --model gpt-5.2

# DeepSeek V3.2
export DEEPSEEK_API_KEY=your-key
spoon-bot agent --provider deepseek --model deepseek-v3.2

# Google Gemini 3
export GEMINI_API_KEY=your-key
spoon-bot agent --provider gemini --model gemini-3-flash-preview

# OpenRouter (access 400+ models with one key)
export OPENROUTER_API_KEY=sk-or-xxx
spoon-bot agent --provider openrouter --model anthropic/claude-sonnet-4.5
```

## LLM Providers

spoon-bot supports multiple LLM providers through the spoon-ai-sdk. You can switch providers without code changes.

### Supported Providers

| Provider | Recommended Models | Context | Notes |
|----------|-------------------|---------|-------|
| **Anthropic** | `claude-opus-4.6`, `claude-sonnet-4.5`, `claude-sonnet-4`, `claude-haiku-4.5` | 200K–1M | Default provider |
| **OpenAI** | `gpt-5.2`, `gpt-5.2-codex`, `gpt-5`, `gpt-5-mini`, `o4-mini`, `o3` | 200K–400K | Full tool support |
| **DeepSeek** | `deepseek-v3.2`, `deepseek-chat-v3.1`, `deepseek-r1` | 64K–164K | Cost-effective |
| **Gemini** | `gemini-3-pro-preview`, `gemini-3-flash-preview`, `gemini-2.5-pro`, `gemini-2.5-flash` | 1M | Google AI |
| **Qwen** | `qwen3-max-thinking`, `qwen3-coder-next`, `qwen3-coder-plus` | 262K–1M | Via OpenRouter |
| **OpenRouter** | 400+ models via single API | varies | Multi-model gateway |

### Per-Provider Configuration

#### Anthropic (Default)

```bash
export ANTHROPIC_API_KEY=sk-ant-xxx
spoon-bot agent --model claude-sonnet-4.5
```

```python
agent = await create_agent(provider="anthropic", model="claude-sonnet-4.5")
```

#### OpenAI

```bash
export OPENAI_API_KEY=sk-xxx
spoon-bot agent --provider openai --model gpt-5.2
```

```python
agent = await create_agent(provider="openai", model="gpt-5.2")
```

#### DeepSeek

```bash
export DEEPSEEK_API_KEY=xxx
spoon-bot agent --provider deepseek --model deepseek-v3.2
```

```python
agent = await create_agent(provider="deepseek", model="deepseek-v3.2")
```

#### Google Gemini

```bash
export GEMINI_API_KEY=xxx
spoon-bot agent --provider gemini --model gemini-3-flash-preview
```

```python
agent = await create_agent(provider="gemini", model="gemini-3-flash-preview")
```

#### OpenRouter (Multi-Model Gateway)

```bash
export OPENROUTER_API_KEY=sk-or-xxx
spoon-bot agent --provider openrouter --model anthropic/claude-sonnet-4.5
```

```python
agent = await create_agent(
    provider="openrouter",
    model="anthropic/claude-sonnet-4.5"
)
```

### OpenRouter Configuration

OpenRouter provides access to 400+ models through a single API key. There are two ways to configure it:

**Option A — Native OpenRouter provider (recommended):**

```bash
export OPENROUTER_API_KEY=sk-or-your-key
```

```python
agent = await create_agent(
    provider="openrouter",
    model="anthropic/claude-sonnet-4.5"
)
```

**Option B — OpenAI-compatible mode:**

```bash
export OPENAI_API_KEY=sk-or-your-key
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

```python
agent = await create_agent(
    provider="openai",
    model="anthropic/claude-sonnet-4.5",
    base_url="https://openrouter.ai/api/v1"
)
```

> **Note:** Do not mix Option A and Option B. Pick one approach.

#### Popular OpenRouter Models

| Model ID | Provider | Context | Price (in/out per 1M tokens) |
|----------|----------|---------|------------------------------|
| `anthropic/claude-opus-4.6` | Anthropic | 1M | $5.00 / $25.00 |
| `anthropic/claude-sonnet-4.5` | Anthropic | 1M | $3.00 / $15.00 |
| `anthropic/claude-haiku-4.5` | Anthropic | 200K | $1.00 / $5.00 |
| `openai/gpt-5.2` | OpenAI | 400K | $1.75 / $14.00 |
| `openai/gpt-5.2-codex` | OpenAI | 400K | $1.75 / $14.00 |
| `openai/gpt-5-mini` | OpenAI | 400K | $0.25 / $2.00 |
| `openai/gpt-5-nano` | OpenAI | 400K | $0.05 / $0.40 |
| `openai/o4-mini` | OpenAI | 200K | $1.10 / $4.40 |
| `openai/o3` | OpenAI | 200K | $2.00 / $8.00 |
| `google/gemini-3-pro-preview` | Google | 1M | $2.00 / $12.00 |
| `google/gemini-3-flash-preview` | Google | 1M | $0.50 / $3.00 |
| `google/gemini-2.5-pro` | Google | 1M | $1.25 / $10.00 |
| `google/gemini-2.5-flash` | Google | 1M | $0.30 / $2.50 |
| `deepseek/deepseek-v3.2` | DeepSeek | 164K | $0.25 / $0.38 |
| `deepseek/deepseek-r1` | DeepSeek | 64K | $0.70 / $2.50 |
| `qwen/qwen3-max-thinking` | Qwen | 262K | $1.20 / $6.00 |
| `qwen/qwen3-coder-plus` | Qwen | 1M | — |
| `qwen/qwen3-coder-next` | Qwen | 262K | $0.07 / $0.30 |
| `moonshotai/kimi-k2.5` | Moonshot | 262K | $0.45 / $2.25 |
| `minimax/minimax-m2.5` | MiniMax | 205K | $0.30 / $1.20 |

### Fallback and Load Balancing

The underlying spoon-ai-sdk supports automatic fallback chains:

```python
from spoon_ai.llm import LLMManager, ConfigurationManager

config_manager = ConfigurationManager()
llm_manager = LLMManager(config_manager)

# Set fallback chain: if OpenAI fails, try Anthropic, then Gemini
llm_manager.set_fallback_chain(["openai", "anthropic", "gemini"])
```

## Session Persistence

spoon-bot supports pluggable session storage backends for conversation history. By default, sessions are saved as JSONL files. You can switch to SQLite or PostgreSQL for more robust persistence.

### Configuration

Set the backend via environment variables in your `.env` file:

**File-based (default, zero config):**

```bash
# No configuration needed — JSONL files in workspace/sessions/
SESSION_STORE_BACKEND=file
```

**SQLite (zero-dependency, single file):**

```bash
SESSION_STORE_BACKEND=sqlite
SESSION_STORE_DB_PATH=./workspace/sessions.db
```

**PostgreSQL (production-grade):**

```bash
SESSION_STORE_BACKEND=postgres
SESSION_STORE_DSN=postgresql://user:password@localhost:5432/spoonbot
```

> **Note:** PostgreSQL requires `psycopg2-binary`. Install with `pip install psycopg2-binary` or `uv add psycopg2-binary`.

### Programmatic Usage

```python
from spoon_bot.session import create_session_store, SQLiteSessionStore

# Factory-based (reads from config)
store = create_session_store(backend="sqlite", db_path="./sessions.db")

# Direct instantiation
store = SQLiteSessionStore("./sessions.db")
store.save_session(session)
loaded = store.load_session("my-session-key")
keys = store.list_session_keys()
```

### Gateway Integration

When running the gateway server, session persistence is configured via environment variables and automatically applied:

```bash
SESSION_STORE_BACKEND=sqlite SESSION_STORE_DB_PATH=./sessions.db spoon-bot gateway
```

## Semantic Memory (memsearch)

spoon-bot supports semantic memory search via [memsearch](https://github.com/zilliztech/memsearch), which provides hybrid search (dense vector + BM25 full-text) over the agent's Markdown memory files using Milvus Lite.

### Setup

Install the `memory` extra:

```bash
# With uv
uv sync --extra memory

# With pip
pip install -e ".[memory]"
```

> **Note:** Milvus Lite requires **Linux or macOS**. On Windows, use WSL.

### Environment Variables

Embedding credentials use `OPENAI_EMBEDDING_*` variables to avoid conflicts with standard `OPENAI_*` keys (which may be used for Whisper STT/TTS or the OpenAI LLM provider). If `OPENAI_EMBEDDING_*` variables are not set, the system falls back to the standard `OPENAI_*` variables.

```bash
# Embedding API (OpenAI-compatible endpoint)
OPENAI_EMBEDDING_API_KEY=your-embedding-api-key
OPENAI_EMBEDDING_BASE_URL=https://ai.gitee.com/v1   # or https://api.openai.com/v1
OPENAI_EMBEDDING_MODEL=Qwen3-Embedding-0.6B          # or text-embedding-3-small

# These are used as fallback if OPENAI_EMBEDDING_* are not set:
# OPENAI_API_KEY=sk-xxx
# OPENAI_BASE_URL=https://api.openai.com/v1
```

### Programmatic Usage

```python
from pathlib import Path
from spoon_bot.memory.semantic_store import SemanticMemoryStore

store = SemanticMemoryStore(
    workspace=Path("./workspace"),
    embedding_provider="openai",
    embedding_model="Qwen3-Embedding-0.6B",
    embedding_api_key="your-key",           # or set OPENAI_EMBEDDING_API_KEY env
    embedding_base_url="https://ai.gitee.com/v1",  # or set OPENAI_EMBEDDING_BASE_URL env
)

# Index existing memory files
await store.initialize()

# Semantic search
results = await store.async_search("Redis caching configuration")
for r in results:
    print(f"[{r['heading']}] score={r['score']:.3f}  {r['content'][:80]}")

# File-based operations still work as before
store.add_memory("API uses JWT RS256 tokens", category="Architecture Decisions")
store.add_daily_note("Deployed v2.1.0 to staging")
```

### AgentLoop Integration

Enable semantic memory in the agent loop via `MemSearchConfig`:

```python
from spoon_bot.agent.loop import create_agent
from spoon_bot.config import MemSearchConfig

agent = await create_agent(
    provider="openrouter",
    model="google/gemini-3-flash-preview",
    memsearch_config=MemSearchConfig(
        enabled=True,
        embedding_provider="openai",
        # Model, API key, and base URL are read from env if not set here:
        #   OPENAI_EMBEDDING_MODEL, OPENAI_EMBEDDING_API_KEY, OPENAI_EMBEDDING_BASE_URL
    ),
)

# Or pass as a dict:
agent = await create_agent(
    provider="openrouter",
    model="google/gemini-3-flash-preview",
    memsearch_config={
        "enabled": True,
        "embedding_provider": "openai",
        "embedding_model": "Qwen3-Embedding-0.6B",
    },
)
```

When enabled, the `memory` tool's `search` action automatically uses semantic search instead of basic text matching.

## Web Search

spoon-bot includes a built-in web search tool powered by Tavily. The agent autonomously decides when to search the web for real-time information.

### Setup

```bash
# Get a free API key at https://tavily.com (1,000 free credits/month)
export TAVILY_API_KEY=tvly-your-key
```

Once configured, the agent will automatically use `web_search` when it needs current information (e.g., prices, news, documentation).

## Dynamic Tool Loading

The agent autonomously manages its tool inventory. Instead of hardcoded topic-to-tool mappings, the agent:

1. Reads tool descriptions at startup
2. Decides which tools to activate based on the user's request
3. Can batch-activate multiple tools in a single call
4. Deactivates unused tools to save context space

Use the `activate_tool` tool with actions `activate` (single or batch) and `list` (show available tools).

## Context Window

spoon-bot automatically resolves the context window based on the selected model. For example, Claude Sonnet 4.5 gets 1M tokens, GPT-5.2 gets 400K, and DeepSeek V3.2 gets 164K. The default for unknown models is 128K.

You can override this via environment variable or code:

```bash
# Override to 256K
CONTEXT_WINDOW=256000 spoon-bot agent --model gpt-5.2
```

```python
agent = await create_agent(model="gpt-5.2", context_window=256_000)
```

The agent includes its context budget in the system prompt, allowing it to adjust verbosity accordingly.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       SPOON-BOT TOOLS                            │
├──────────────────────────────────────────────────────────────────┤
│  Native OS Tools (Always Available, Priority)                    │
│  ├── shell           Execute commands (60s timeout, 10KB limit)  │
│  ├── read_file       Read file contents                          │
│  ├── write_file      Write content to file                       │
│  ├── edit_file       Edit file by replacing text                 │
│  └── list_dir        List directory contents                     │
├──────────────────────────────────────────────────────────────────┤
│  Web & Search Tools                                              │
│  ├── web_search      Tavily-powered real-time web search         │
│  └── web_fetch       Fetch and parse web page content            │
├──────────────────────────────────────────────────────────────────┤
│  Self-Management Tools                                           │
│  ├── self_config     get/set/list agent configuration            │
│  ├── self_upgrade    check updates, install/update skills        │
│  ├── memory          remember, note, search, forget, checkpoint  │
│  └── activate_tool   dynamically load/unload tools on demand     │
├──────────────────────────────────────────────────────────────────┤
│  Extension Tools                                                 │
│  ├── MCP tools       Dynamic via configured MCP servers          │
│  ├── spoon-toolkit   40+ crypto/blockchain/social tools          │
│  └── Skill tools     ScriptTool from SKILL.md scripts            │
├──────────────────────────────────────────────────────────────────┤
│  Session Persistence                                             │
│  ├── File (JSONL)    Default, zero-dependency                    │
│  ├── SQLite          Local embedded database                     │
│  └── PostgreSQL      Production-grade remote DB                  │
└──────────────────────────────────────────────────────────────────┘
```

## Workspace Structure

Default workspace: `~/.spoon-bot/workspace/`

```
~/.spoon-bot/
├── workspace/
│   ├── AGENTS.md        # Agent instructions
│   ├── SOUL.md          # Personality
│   ├── memory/
│   │   ├── MEMORY.md    # Long-term facts
│   │   └── YYYY-MM-DD.md # Daily notes
│   └── skills/          # Custom skills
├── config.yaml          # Channel configuration (for service mode)
├── service.pid          # PID of background service process
├── gateway.log          # Background service log
└── config.json          # Configuration
```

## CLI Commands

```bash
# Agent mode (interactive REPL)
spoon-bot agent                        # Interactive mode (default provider)
spoon-bot agent --provider openai      # Use specific provider
spoon-bot agent --model gpt-5.2       # Use specific model
spoon-bot agent -m "message"           # One-shot mode

# Gateway mode (multi-channel server)
spoon-bot gateway                      # Start all configured channels
spoon-bot gateway --channels telegram  # Start Telegram channel only
spoon-bot gateway --no-cli             # Disable CLI input in gateway mode
spoon-bot gateway --config path/to/config.yaml  # Custom config file

# Background service management
spoon-bot service start                # Start gateway in the background
spoon-bot service stop                 # Stop the background service
spoon-bot service restart              # Restart the service
spoon-bot service status               # Show PID, auto-start, and log path
spoon-bot service logs                 # Show last 50 log lines
spoon-bot service logs -f              # Follow log output (tail -f)
spoon-bot service install              # Register auto-start at login
spoon-bot service uninstall            # Remove auto-start registration

# General
spoon-bot onboard                      # Initialize workspace
spoon-bot status                       # Show status
spoon-bot version                      # Show version
```

## Background Service

Run the gateway as a persistent background service — no Docker, no terminal window required. Once started, the bot keeps running and responds to your chat tools (Telegram, Discord, etc.) automatically.

### Quick Start

```bash
# 1. Configure your channels (if not already done)
cp config.example.yaml ~/.spoon-bot/config.yaml
# edit ~/.spoon-bot/config.yaml with your tokens

# 2. Start the service in the background
spoon-bot service start

# 3. Check it's running
spoon-bot service status

# 4. Follow logs
spoon-bot service logs -f
```

### Auto-Start at Login

Register the gateway to start automatically every time you log in:

```bash
spoon-bot service install    # register
spoon-bot service uninstall  # remove
```

| Platform | Method | Requires Admin? |
|----------|--------|----------------|
| Windows (7+) | Task Scheduler (`ONLOGON` trigger) | No |
| Linux | systemd user service (`~/.config/systemd/user/`) | No |
| macOS | launchd agent (`~/Library/LaunchAgents/`) | No |

### Service Files

All state is stored in `~/.spoon-bot/`:

```
~/.spoon-bot/
├── config.yaml      # Channel configuration
├── service.pid      # PID of the running process
└── gateway.log      # Rolling log (append-only)
```

### Commands Reference

| Command | Description |
|---------|-------------|
| `spoon-bot service start [-c config.yaml]` | Start gateway in background |
| `spoon-bot service stop` | Gracefully stop (SIGTERM → SIGKILL after 10 s) |
| `spoon-bot service restart` | Stop + start |
| `spoon-bot service status` | Show running state, PID, auto-start flag, log path |
| `spoon-bot service logs [-n N] [-f]` | Print last N lines; `-f` to stream |
| `spoon-bot service install [-c config.yaml]` | Register OS-level auto-start |
| `spoon-bot service uninstall` | Remove OS-level auto-start |

## Gateway API

spoon-bot includes a WebSocket and REST API gateway for remote agent control, with JWT authentication, rate limiting, and real-time streaming.

### Installation

```bash
# With uv
uv sync --extra gateway

# With pip
pip install -e ".[gateway]"
```

### Quick Start (CLI)

```bash
# Install gateway dependencies
uv sync --extra gateway --extra telegram

# Start gateway with Telegram channel
spoon-bot gateway --channels telegram
```

### Quick Start (Programmatic)

```python
import asyncio
import uvicorn
from spoon_bot.gateway import create_app, GatewayConfig
from spoon_bot.agent.loop import create_agent
from spoon_bot.gateway.app import set_agent

async def main():
    # Create agent with your preferred provider
    agent = await create_agent(provider="openai", model="gpt-5.2")

    # Create gateway app
    config = GatewayConfig(host="0.0.0.0", port=8080)
    app = create_app(config)

    # Set agent
    set_agent(agent)

    # Run server
    uvicorn.run(app, host=config.host, port=config.port)

asyncio.run(main())
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (includes channel status) |
| `/ready` | GET | Readiness check |
| `/v1/auth/login` | POST | Authenticate and get JWT tokens |
| `/v1/agent/chat` | POST | Send message to agent (sync or streaming) |
| `/v1/agent/chat/async` | POST | Async task-based chat |
| `/v1/agent/status` | GET | Agent and channel status |
| `/v1/sessions` | GET/POST | Manage sessions |
| `/v1/tools` | GET | List available tools |
| `/v1/skills` | GET/POST | List and manage skills |
| `/v1/ws` | WS | WebSocket for real-time communication |

### WebSocket Protocol

The WebSocket endpoint supports:
- `chat.send` — send messages (stream/non-stream)
- `chat.cancel` — cancel in-flight requests (stream and non-stream)
- `session.switch` / `session.list` / `session.export` / `session.import` — session management
- `subscribe` / `unsubscribe` — event subscriptions
- Input validation with structured error codes (`INVALID_PARAMS`, `AGENT_NOT_READY`, etc.)

### Authentication

```bash
# Enable auth
GATEWAY_AUTH_REQUIRED=true
GATEWAY_API_KEY=your-secret-key
JWT_SECRET=your-jwt-secret

# Rate limiting (per IP)
GATEWAY_AUTH_REQUESTS_PER_MINUTE=20
```

See [docs/API_DESIGN.md](docs/API_DESIGN.md) for full API documentation.

## MCP Integration

spoon-bot supports Model Context Protocol (MCP) servers for extended capabilities:

```python
agent = await create_agent(
    mcp_config={
        "github": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"]
        },
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
        }
    }
)
```

Supported MCP transport types:
- stdio (command-line servers)
- npx/uvx (package runners)
- SSE (Server-Sent Events)
- HTTP/WebSocket

## Skills System

Create custom skills by adding `SKILL.md` files to your workspace. Each skill file must start with YAML frontmatter (`---`):

```
~/.spoon-bot/workspace/skills/
├── weather/
│   └── SKILL.md
├── image_generate/
│   ├── SKILL.md
│   └── scripts/
│       └── generate.py
└── custom_skill/
    └── SKILL.md
```

Example `SKILL.md`:

```markdown
---
name: weather
description: Get weather forecasts
version: "1.0"
---

# Weather Skill

Fetch weather data for any city.
```

Skills are automatically discovered at startup. The agent autonomously decides which skills to activate based on the conversation context. Script-based skills support structured `input_schema` for type-safe parameter passing.

## Development

```bash
# Install dev dependencies with uv
uv sync --extra dev

# Or with pip
pip install -e ".[dev]"

# Core regression suite (default, recommended for daily development)
uv run python scripts/run_test_suite.py

# Core suite with extra pytest flags
uv run python scripts/run_test_suite.py -- --maxfail=1 -q

# Extended scenarios (e2e/live/capability + high-churn/platform-specific tests)
uv run python scripts/run_test_suite.py --suite extended

# Full suite (core + extended)
uv run python scripts/run_test_suite.py --suite all

# Lint
ruff check spoon_bot/
```

### Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `SPOON_BOT_DEFAULT_PROVIDER` | `anthropic` | Default LLM provider |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `DEEPSEEK_API_KEY` | — | DeepSeek API key |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `OPENROUTER_API_KEY` | — | OpenRouter API key |
| `TAVILY_API_KEY` | — | Tavily web search API key |
| `OPENAI_EMBEDDING_API_KEY` | — | Embedding provider API key (falls back to `OPENAI_API_KEY`) |
| `OPENAI_EMBEDDING_BASE_URL` | — | Embedding provider base URL (falls back to `OPENAI_BASE_URL`) |
| `OPENAI_EMBEDDING_MODEL` | — | Embedding model name (e.g. `Qwen3-Embedding-0.6B`) |
| `SESSION_STORE_BACKEND` | `file` | Session backend: `file`, `sqlite`, `postgres` |
| `SESSION_STORE_DB_PATH` | `./workspace/sessions.db` | SQLite database path |
| `SESSION_STORE_DSN` | — | PostgreSQL connection string |
| `CONTEXT_WINDOW` | auto | Context window override (tokens) |
| `GATEWAY_AUTH_REQUIRED` | `false` | Enable gateway authentication |
| `GATEWAY_API_KEY` | — | Gateway API key |
| `JWT_SECRET` | — | JWT signing secret |
| `GATEWAY_AUTH_REQUESTS_PER_MINUTE` | `20` | Auth rate limit per IP |

## License

MIT License
