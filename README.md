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

## Installation

```bash
# Clone and install
cd spoon-bot
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

## spoon-core Integration

spoon-bot can optionally integrate with [spoon-core](https://github.com/XSpoonAi/core) for enhanced capabilities:

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

Install spoon-core for full features:
```bash
pip install spoon-ai
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check spoon_bot/
```

## License

MIT License
