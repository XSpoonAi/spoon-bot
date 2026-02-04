# spoon-bot

Local-first AI agent with native OS tools, powered by spoon-core.

## Features

- **Agent-centric**: Autonomous execution with safety rails
- **OS-native**: Built-in shell/filesystem tools as priority
- **Memory-first**: Four-layer memory system (file + short-term + Mem0 + checkpointer)
- **Self-managing**: Self-configuration, self-upgrade, memory management tools
- **Web3-enabled**: Blockchain operations via spoon-core and spoon-toolkits
- **Extensible**: MCP servers + Skills ecosystem
- **Multi-mode**: Agent / Interactive / Gateway modes
- **spoon-core Integration**: Optional deep integration with spoon-core for enhanced capabilities

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

## Quick Start

```bash
# Set your API key
export ANTHROPIC_API_KEY=your-key

# Initialize workspace
spoon-bot onboard

# Run in interactive mode
spoon-bot agent

# Run in one-shot mode
spoon-bot agent -m "List files in the current directory"
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
│  Self-Management Tools (Coming)                                 │
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

## Configuration

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
└── config.json          # Configuration (coming)
```

## Commands

```bash
spoon-bot agent              # Interactive mode
spoon-bot agent -m "msg"     # One-shot mode
spoon-bot onboard            # Initialize workspace
spoon-bot status             # Show status
spoon-bot version            # Show version
```

## Gateway API

spoon-bot includes a WebSocket and REST API gateway for remote agent control.

### Installation

```bash
pip install spoon-bot[gateway]
```

### Quick Start

```python
import asyncio
import uvicorn
from spoon_bot.gateway import create_app, GatewayConfig
from spoon_bot.agent.loop import create_agent
from spoon_bot.gateway.app import set_agent

async def main():
    # Create agent
    agent = await create_agent()

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

## spoon-core Integration

spoon-bot is powered by [spoon-core](https://github.com/XSpoonAi/spoon-core) (spoon-ai-sdk) for its core capabilities:

### LLM Providers
When spoon-core is installed, you get access to multiple LLM providers:
- Anthropic Claude
- OpenAI GPT
- DeepSeek
- Ollama (local)
- Google Gemini
- OpenRouter

```python
from spoon_bot.agent.loop import create_agent

# Use spoon-core providers (auto-detected)
agent = await create_agent(provider="deepseek", model="deepseek-chat")
```

### Skill System
spoon-core enables advanced skill features:
- LLM-powered intent matching
- Script execution within skills
- State persistence across sessions

### MCP Integration
spoon-core provides enhanced MCP support:
- Multiple transport types (stdio, npx, uvx, SSE, HTTP, WebSocket)
- Connection pooling and health checks
- Automatic retry and recovery

spoon-ai-sdk is included as a required dependency (v0.4.0+). It will be installed automatically when you run `uv sync` or `pip install -e .`.

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
