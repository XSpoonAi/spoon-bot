# Task Plan: Spoon-Bot Local Agent Implementation

## Goal
Transform spoon-cli into a nanobot-style local agent product (spoon-bot) with native OS tools as priority, integrating spoon-core memory and spoon-toolkits ecosystem.

## Implementation Phases

### Phase 1: Core Agent Loop (Priority: LOCAL AGENT FIRST)
- [ ] 1.1 Create project structure `spoon-bot/spoon_bot/`
- [ ] 1.2 Port AgentLoop from nanobot (`agent/loop.py`)
- [ ] 1.3 Create Tool base class and ToolRegistry (`agent/tools/base.py`, `registry.py`)
- [ ] 1.4 Implement ShellTool with safety guards (`agent/tools/shell.py`)
- [ ] 1.5 Implement FilesystemTools: read, write, edit, list (`agent/tools/filesystem.py`)
- [ ] 1.6 Create ContextBuilder with bootstrap file support (`agent/context.py`)
- [ ] 1.7 Integrate spoon-core ChatBot for LLM calls
- [ ] 1.8 **VERIFY**: `python -m spoon_bot agent -m "list files in current directory"`

### Phase 2: Session & Memory (Critical)
- [ ] 2.1 Port SessionManager from nanobot (JSONL storage)
- [ ] 2.2 Implement FileBasedMemory (MEMORY.md + daily notes)
- [ ] 2.3 Integrate spoon-core ShortTermMemoryManager
- [ ] 2.4 Integrate spoon-core SpoonMem0 (optional semantic memory)
- [ ] 2.5 Create IntegratedMemorySystem (4-layer unified)
- [ ] 2.6 Inject memory context into system prompt
- [ ] 2.7 **VERIFY**: Conversation persistence across restarts

### Phase 3: CLI & Onboarding
- [ ] 3.1 Create Typer CLI (`onboard`, `agent`, `gateway`, `status`)
- [ ] 3.2 Implement `onboard` command (config + workspace creation)
- [ ] 3.3 Create default bootstrap templates (AGENTS.md, SOUL.md, TOOLS.md)
- [ ] 3.4 Create pyproject.toml with dependencies
- [ ] 3.5 **VERIFY**: `spoon-bot onboard` creates all files

### Phase 4: MCP & Skills Integration
- [ ] 4.1 Port MCP client from spoon-cli
- [ ] 4.2 Create MCPToolAdapter for tool registration
- [ ] 4.3 Implement SkillsLoader with SKILL.md support
- [ ] 4.4 Integrate spoon-core SkillManager for advanced triggers
- [ ] 4.5 Create default skills (coding, research)
- [ ] 4.6 **VERIFY**: MCP server tools load; skill activation works

### Phase 5: Gateway Mode
- [ ] 5.1 Port MessageBus from nanobot
- [ ] 5.2 Port BaseChannel interface
- [ ] 5.3 Implement CLI channel (interactive mode)
- [ ] 5.4 Port Telegram channel
- [ ] 5.5 Create ChannelManager for routing
- [ ] 5.6 **VERIFY**: Gateway responds to Telegram messages

### Phase 6: Web3 & Toolkit Integration
- [ ] 6.1 Create ToolkitAdapter for spoon-toolkits
- [ ] 6.2 Integrate crypto tools (price, kline, alerts)
- [ ] 6.3 Integrate blockchain tools (Chainbase, ThirdWeb)
- [ ] 6.4 Integrate security tools (GoPlusLabs)
- [ ] 6.5 Port Web3 tools from spoon-core (Turnkey, x402)
- [ ] 6.6 Implement wallet configuration via SecretVault
- [ ] 6.7 **VERIFY**: `agent -m "check ETH balance"`

### Phase 7: Self-Management Tools
- [ ] 7.1 Implement SelfConfigTool (get/set/list/reset)
- [ ] 7.2 Implement SelfUpgradeTool (check/upgrade/install skills)
- [ ] 7.3 Implement MemoryTool (remember/note/search/forget/checkpoint)
- [ ] 7.4 Add safety guards (user confirmation for sensitive ops)
- [ ] 7.5 **VERIFY**: `agent -m "update model to claude-opus-4"`

### Phase 8: Background Services
- [ ] 8.1 Implement SpawnTool for background tasks
- [ ] 8.2 Port cron service from nanobot
- [ ] 8.3 Port heartbeat service from nanobot
- [ ] 8.4 Create HEARTBEAT.md template
- [ ] 8.5 **VERIFY**: Heartbeat executes tasks periodically

## Key Questions
1. Where should spoon-bot be created? → `C:\Users\Ricky\Documents\Project\XSpoonAi\spoon-bot\`
2. Reuse spoon-cli config system? → Yes, port with modifications
3. Default LLM provider? → Anthropic Claude (configurable)

## Decisions Made
- **Local agent priority**: Native OS tools built-in, MCP as optional extension
- **4-layer memory**: File-based + ShortTerm + Mem0 + Checkpointer
- **Tool registry pattern**: Native → Toolkit → MCP (priority order)
- **Project location**: New `spoon-bot/` directory in XSpoonAi monorepo

## Dependencies
```
spoon-ai (core)        # ChatBot, Memory, Tools, Skills
spoon-toolkits         # 40+ crypto/blockchain/social tools
typer                  # CLI framework
aiofiles              # Async file operations
mem0ai (optional)     # Semantic memory
```

## Status
**Currently in Phase 0** - Plan approved, ready to begin implementation

---

## Quick Reference: File Mapping

| Source | Target | Notes |
|--------|--------|-------|
| nanobot/agent/loop.py | spoon_bot/agent/loop.py | Port core loop |
| nanobot/agent/tools/*.py | spoon_bot/agent/tools/*.py | Port native tools |
| nanobot/bus/*.py | spoon_bot/bus/*.py | Port message bus |
| spoon-cli/mcp/*.py | spoon_bot/mcp/*.py | Port MCP client |
| spoon-core memory | Direct import | Use as dependency |
| spoon-toolkits | Direct import | Use as dependency |
