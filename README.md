# spoon-bot

Local-first AI agent with native OS tools, powered by spoon-core.

## Features

- **Multi-Provider LLM**: Supports Anthropic, OpenAI, DeepSeek, Gemini, OpenRouter, and more
- **Agent-centric**: Autonomous execution with safety rails
- **OS-native**: Built-in shell/filesystem tools as priority
- **Memory-first**: Four-layer memory system (file + short-term + Mem0 + checkpointer)
- **Self-managing**: Self-configuration, self-upgrade, memory management tools
- **Web3-enabled**: Blockchain operations via spoon-core and spoon-toolkits
- **Extensible**: MCP servers + Skills ecosystem
- **Multi-mode**: Agent / Interactive / Gateway modes

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

Example `.env` file:

```bash
# Use any ONE of these (or multiple for fallback)
ANTHROPIC_API_KEY=sk-ant-xxx
OPENAI_API_KEY=sk-xxx
DEEPSEEK_API_KEY=xxx
GEMINI_API_KEY=xxx
OPENROUTER_API_KEY=sk-or-xxx
```

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
# OpenAI GPT
export OPENAI_API_KEY=your-key
spoon-bot agent --provider openai --model gpt-4o

# DeepSeek
export DEEPSEEK_API_KEY=your-key
spoon-bot agent --provider deepseek --model deepseek-chat

# Google Gemini
export GEMINI_API_KEY=your-key
spoon-bot agent --provider gemini --model gemini-2.0-flash
```

## LLM Providers

spoon-bot supports multiple LLM providers through the spoon-ai-sdk. You can switch providers without code changes.

### Supported Providers

| Provider | Models | Notes |
|----------|--------|-------|
| **Anthropic** | `claude-sonnet-4-20250514`, `claude-opus-4-20250514`, `claude-3-5-haiku-20241022` | Default provider |
| **OpenAI** | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `o1`, `o1-mini` | Full tool support |
| **DeepSeek** | `deepseek-chat`, `deepseek-coder` | Cost-effective |
| **Gemini** | `gemini-2.0-flash`, `gemini-2.0-pro`, `gemini-1.5-pro` | Google AI |
| **OpenRouter** | 200+ models | Multi-model gateway |

### Using Programmatically

```python
from spoon_bot.agent.loop import create_agent

# Anthropic Claude (default)
agent = await create_agent()

# OpenAI GPT
agent = await create_agent(provider="openai", model="gpt-4o")

# DeepSeek
agent = await create_agent(provider="deepseek", model="deepseek-chat")

# Google Gemini
agent = await create_agent(provider="gemini", model="gemini-2.0-flash")

# OpenRouter (access any model)
agent = await create_agent(
    provider="openai",  # Use OpenAI-compatible API
    model="anthropic/claude-sonnet-4",
    base_url="https://openrouter.ai/api/v1"
)
```

### OpenRouter Configuration

OpenRouter provides access to 200+ models through a single API. Set your OpenRouter API key as `OPENAI_API_KEY` when using the `openai` provider with a custom base URL:

```bash
export OPENAI_API_KEY=sk-or-your-openrouter-key
```

```python
agent = await create_agent(
    provider="openai",
    model="anthropic/claude-sonnet-4",  # Any OpenRouter model
    base_url="https://openrouter.ai/api/v1"
)
```

### Fallback and Load Balancing

The underlying spoon-ai-sdk supports automatic fallback chains:

```python
from spoon_ai.llm import LLMManager, ConfigurationManager

config_manager = ConfigurationManager()
llm_manager = LLMManager(config_manager)

# Set fallback chain: if OpenAI fails, try Anthropic, then Gemini
llm_manager.set_fallback_chain(["openai", "anthropic", "gemini"])
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      SPOON-BOT TOOLS                             │
├─────────────────────────────────────────────────────────────────┤
│  Native OS Tools (Always Available, Priority)                   │
│  ├── shell          Execute commands (60s timeout, 10KB limit)  │
│  ├── read_file      Read file contents                          │
│  ├── write_file     Write content to file                       │
│  ├── edit_file      Edit file by replacing text                 │
│  └── list_dir       List directory contents                     │
├─────────────────────────────────────────────────────────────────┤
│  Self-Management Tools                                          │
│  ├── self_config    get/set/list agent configuration            │
│  ├── self_upgrade   check updates, install/update skills        │
│  └── memory         remember, note, search, forget, checkpoint  │
├─────────────────────────────────────────────────────────────────┤
│  Extension Tools                                                │
│  ├── MCP tools      Dynamic via configured MCP servers          │
│  ├── spoon-toolkit  40+ crypto/blockchain/social tools          │
│  └── Skill tools    ScriptTool from SKILL.md scripts            │
└─────────────────────────────────────────────────────────────────┘
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
└── config.json          # Configuration
```

## CLI Commands

```bash
spoon-bot agent                        # Interactive mode (default provider)
spoon-bot agent --provider openai      # Use specific provider
spoon-bot agent --model gpt-4o         # Use specific model
spoon-bot agent -m "message"           # One-shot mode
spoon-bot onboard                      # Initialize workspace
spoon-bot status                       # Show status
spoon-bot version                      # Show version
```

## Gateway API

spoon-bot includes a WebSocket and REST API gateway for remote agent control.

### Installation

```bash
# With uv
uv sync --extra gateway

# With pip
pip install -e ".[gateway]"
```

### Quick Start

```python
import asyncio
import uvicorn
from spoon_bot.gateway import create_app, GatewayConfig
from spoon_bot.agent.loop import create_agent
from spoon_bot.gateway.app import set_agent

async def main():
    # Create agent with your preferred provider
    agent = await create_agent(provider="openai", model="gpt-4o")

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
| `/v1/auth/login` | POST | Authenticate and get tokens |
| `/v1/agent/chat` | POST | Send message to agent |
| `/v1/agent/status` | GET | Get agent status |
| `/v1/sessions` | GET/POST | Manage sessions |
| `/v1/tools` | GET | List available tools |
| `/v1/skills` | GET | List available skills |
| `/v1/ws` | WS | WebSocket for real-time communication |

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

Create custom skills by adding `SKILL.md` files to your workspace:

```
~/.spoon-bot/workspace/skills/
├── code_review/
│   └── SKILL.md
├── git_helper/
│   └── SKILL.md
└── custom_skill/
    └── SKILL.md
```

Skills are automatically discovered and can be invoked by the agent.

## Development

```bash
# Install dev dependencies with uv
uv sync --extra dev

# Or with pip
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check spoon_bot/
```

## License

MIT License
