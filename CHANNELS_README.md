# Multi-Platform Channels Guide

Spoon-bot supports multiple communication platforms through a unified channel system.

## Supported Platforms

- **Telegram** - Polling & Webhook modes, Groups, Media, 23 slash commands
- **Discord** - Gateway connection, Guilds, DMs
- **Feishu (Lark)** - WebSocket long-connection & Webhook, Enterprise apps
- **CLI** - Local terminal interface

## Installation

Install channel dependencies:

```bash
# Telegram only
uv pip install -e ".[telegram]"

# Discord only
uv pip install -e ".[discord]"

# Feishu only
uv pip install -e ".[feishu]"

# All channels
uv pip install -e ".[all-channels]"
```

## Quick Start (`.env` only, no config.yaml needed)

The simplest way to get started. Create a `.env` file in the project root:

```bash
# .env
OPENROUTER_API_KEY=sk-or-v1-...
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
```

Then run:

```bash
uv run spoon-bot gateway
```

Spoon-bot will auto-discover the Telegram channel from the environment variable and start in polling mode.

Supported env vars:

| Platform | Env Var | Purpose |
|----------|---------|---------|
| Telegram | `TELEGRAM_BOT_TOKEN` | Bot token (required) |
| Telegram | `TELEGRAM_USER_ID` | Restrict access to your user ID |
| Discord | `DISCORD_BOT_TOKEN` | Bot token (required) |
| Discord | `DISCORD_GUILD_ID` | Restrict to one server |
| Discord | `DISCORD_USER_ID` | Restrict access to your user ID |
| Feishu | `FEISHU_APP_ID` | App ID (required) |
| Feishu | `FEISHU_APP_SECRET` | App secret (required) |
| Feishu | `FEISHU_VERIFICATION_TOKEN` | Verification token |
| Feishu | `FEISHU_ENCRYPT_KEY` | Encryption key |

These env vars work in two scenarios:
- **Auto-discovery**: No config.yaml channels section needed; channels are created automatically
- **Fallback**: config.yaml defines the channel but omits credentials; env vars fill in the gaps

## Configuration

There are three ways to provide channel credentials, in order of priority:

1. **config.yaml** (explicit value) - highest priority
2. **config.yaml with `${VAR}`** - resolved from environment
3. **Environment variable fallback** - auto-detected from well-known env var names

### Option A: `.env` file only (simplest)

```bash
# .env
OPENROUTER_API_KEY=sk-or-v1-...

# Telegram
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_USER_ID=123456789

# Discord
DISCORD_BOT_TOKEN=your_discord_token
DISCORD_GUILD_ID=123456789012345678
DISCORD_USER_ID=123456789012345678

# Feishu
FEISHU_APP_ID=your_app_id
FEISHU_APP_SECRET=your_app_secret
```

No `config.yaml` channels section needed. Channels are auto-discovered.

### Option B: config.yaml with `${VAR}` references

```bash
cp config.example.yaml config.yaml
```

```yaml
# config.yaml
agent:
  model: "anthropic/claude-sonnet-4"
  provider: "openrouter"

channels:
  telegram:
    enabled: true
    accounts:
      - name: "main_bot"
        token: "${TELEGRAM_BOT_TOKEN}"
        mode: "polling"
        allowed_users: [123456789]

  discord:
    enabled: true
    accounts:
      - name: "dev_bot"
        token: "${DISCORD_BOT_TOKEN}"
        intents:
          - guilds
          - guild_messages
          - message_content
          - dm_messages
```

Then set tokens in `.env` or shell:

```bash
# .env
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
DISCORD_BOT_TOKEN=your_token
OPENROUTER_API_KEY=sk-or-v1-...
```

### Option C: config.yaml with `enabled: true` but no token field

If `config.yaml` enables a channel but omits credentials or access control, env vars fill in the gaps:

```yaml
channels:
  telegram:
    enabled: true
    accounts:
      - name: "main_bot"
        mode: "polling"
        # token not specified — falls back to TELEGRAM_BOT_TOKEN env var
        # allowed_users not specified — falls back to TELEGRAM_USER_ID env var
```

### Config File Resolution Order

When no explicit `--config` path is given, spoon-bot searches:

1. `SPOON_BOT_CONFIG` environment variable
2. `~/.spoon-bot/config.yaml`
3. `./config.yaml`

## Configuration Priority

### Agent config (model, provider, etc.)

```
CLI args  >  YAML agent: section  >  env vars  >  defaults.py
```

### Channel credentials (token, app_id, etc.)

```
YAML explicit value  >  YAML ${VAR} resolved  >  well-known env var fallback
```

### Access control (allowed_users, allowed_guilds)

```
YAML list  >  env var fallback (TELEGRAM_USER_ID / DISCORD_USER_ID / DISCORD_GUILD_ID)
```

If YAML has `allowed_users` set, env vars are ignored. If YAML omits it, the env var is used. If neither is set, the bot is open to all users.

## Usage

### Gateway Mode (Multi-platform)

```bash
# Start all configured/auto-discovered channels
uv run spoon-bot gateway

# Start specific channels only
uv run spoon-bot gateway --channels telegram,discord

# Override LLM provider/model at runtime
uv run spoon-bot gateway --provider openrouter --model anthropic/claude-sonnet-4

# Use custom config file
uv run spoon-bot gateway --config my-config.yaml

# Disable CLI channel
uv run spoon-bot gateway --no-cli
```

#### Gateway CLI Options

| Option | Description |
|--------|-------------|
| `--config`, `-c` | Path to YAML config file |
| `--channels` | Comma-separated channels to start |
| `--model` | LLM model name (overrides YAML) |
| `--provider` | LLM provider (overrides YAML) |
| `--api-key` | API key for the LLM provider |
| `--base-url` | Custom LLM API base URL |
| `--tool-profile` | Tool profile: `core`, `coding`, `research`, `full` |
| `-w`, `--workspace` | Workspace directory |
| `--cli` / `--no-cli` | Enable/disable CLI channel |

### CLI Mode (Single-user)

```bash
# Interactive REPL
uv run spoon-bot agent

# One-shot message
uv run spoon-bot agent -m "Hello"
```

### Docker / uvicorn (HTTP/WS API only)

This starts the REST + WebSocket API server without channel support:

```bash
uvicorn spoon_bot.gateway.server:create_app --factory --host 0.0.0.0 --port 8080
```

Or via Docker:

```bash
docker compose up -d
```

Both paths also load `.env` and read YAML agent config.

## Platform-Specific Setup

### Telegram Bot

1. **Create Bot**
   - Message [@BotFather](https://t.me/BotFather) on Telegram
   - Send `/newbot` and follow instructions
   - Copy the bot token

2. **Get Your User ID**
   - Message [@userinfobot](https://t.me/userinfobot)
   - Copy your user ID
   - Add to `allowed_users` in config (or omit for unrestricted access)

3. **Configure** (choose one)

   **Via `.env` (simplest):**
   ```bash
   TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
   ```

   **Via `config.yaml` (more control):**
   ```yaml
   channels:
     telegram:
       enabled: true
       accounts:
         - name: "main_bot"
           token: "${TELEGRAM_BOT_TOKEN}"
           mode: "polling"
           allowed_users: [your_user_id]
   ```

4. **Start & Test**
   ```bash
   uv run spoon-bot gateway
   ```
   Then message your bot on Telegram!

#### Telegram Bot Commands (23 total)

| Command | Description |
|---------|-------------|
| `/start` | Start the bot |
| `/help` | Show command list |
| `/commands` | Browse all commands (paginated) |
| `/whoami` | Show your user info |
| `/model` | Quick model switch |
| `/models` | Browse models by provider |
| `/think` | Set thinking level |
| `/verbose` | Toggle verbose mode |
| `/reasoning` | Toggle reasoning display |
| `/skill` | Browse available skills |
| `/tools` | List active tools |
| `/status` | Show bot status |
| `/history` | Show recent history |
| `/context` | Show context window usage |
| `/usage` | Show token usage stats |
| `/memory` | Show memory summary |
| `/note` | Save a note |
| `/remember` | Remember a fact |
| `/stop` | Stop current task |
| `/new` | Start a new session |
| `/compact` | Compact conversation context |
| `/clear` | Clear conversation history |
| `/cancel` | Cancel current operation |

### Discord Bot

1. **Create Application**
   - Go to [Discord Developer Portal](https://discord.com/developers/applications)
   - Click "New Application"
   - Go to "Bot" tab, click "Add Bot"
   - Copy bot token

2. **Enable Intents**
   - In Bot settings, enable:
     - Message Content Intent (required)
     - Server Members Intent (optional)
     - Presence Intent (optional)

3. **Invite Bot to Server**
   - Go to OAuth2 -> URL Generator
   - Select scopes: `bot`, `applications.commands`
   - Select permissions: `Send Messages`, `Read Messages`
   - Copy URL and open in browser to select your server

4. **Get IDs**
   - Enable Developer Mode in Discord (User Settings -> Advanced)
   - Right-click your server -> Copy ID (guild_id)
   - Right-click your user -> Copy ID (user_id)

5. **Configure** (choose one)

   **Via `.env` (simplest):**
   ```bash
   DISCORD_BOT_TOKEN=your_token
   ```

   **Via `config.yaml` (more control):**
   ```yaml
   channels:
     discord:
       enabled: true
       accounts:
         - name: "dev_bot"
           token: "${DISCORD_BOT_TOKEN}"
           intents:
             - guilds
             - guild_messages
             - message_content
             - dm_messages
           allowed_guilds: [your_guild_id]
           allowed_users: [your_user_id]
   ```

6. **Start**
   ```bash
   uv run spoon-bot gateway
   ```
   Mention the bot in your server: `@YourBot hello`

### Feishu Bot

Feishu supports two connection modes:

- **WebSocket long-connection** (recommended, `mode: ws`) - No public IP needed
- **Webhook** (`mode: webhook`) - Requires public HTTPS endpoint

1. **Create Enterprise App**
   - Go to [Feishu Open Platform](https://open.feishu.cn/)
   - Create an enterprise app
   - Get App ID and App Secret

2. **Configure Permissions**
   - Add permissions:
     - `im:message` - Receive messages
     - `im:message:send_as_bot` - Send messages

3. **Configure** (choose one)

   **Via `.env` (simplest, uses WebSocket mode):**
   ```bash
   FEISHU_APP_ID=your_app_id
   FEISHU_APP_SECRET=your_app_secret
   ```

   **Via `config.yaml` (webhook mode):**
   ```yaml
   channels:
     feishu:
       enabled: true
       accounts:
         - name: "enterprise_bot"
           app_id: "${FEISHU_APP_ID}"
           app_secret: "${FEISHU_APP_SECRET}"
           verification_token: "${FEISHU_VERIFICATION_TOKEN}"
           mode: "webhook"
           webhook_url: "https://your-domain.com/feishu/webhook"
   ```

4. **Start**
   ```bash
   uv run spoon-bot gateway
   ```

## Advanced Features

### Group Chat Support

**Telegram:**
```yaml
telegram:
  accounts:
    - name: "main_bot"
      groups:
        enabled: true
        require_mention: true  # Bot must be @mentioned
        allowed_chats:
          - "@group_username"
          - -1001234567890  # Group chat ID
```

**Discord:**
- Bot responds to mentions automatically
- Use `allowed_guilds` to limit servers

### Multi-Account Support

Run multiple bot accounts simultaneously:

```yaml
telegram:
  accounts:
    - name: "personal_bot"
      token: "${TELEGRAM_TOKEN_1}"
      allowed_users: [123456]

    - name: "work_bot"
      token: "${TELEGRAM_TOKEN_2}"
      allowed_users: [789012]
```

### Webhook Mode (Telegram, Production)

For better performance in production:

```yaml
telegram:
  accounts:
    - name: "main_bot"
      mode: "webhook"
      webhook_url: "https://your-domain.com/telegram/webhook"
      webhook_secret: "your_secret"
```

### Proxy Support (Telegram)

```yaml
telegram:
  accounts:
    - name: "main_bot"
      proxy_url: "http://127.0.0.1:7890"
```

## Startup Paths

| Entry Point | Agent Config | Channels | Use Case |
|-------------|-------------|----------|----------|
| `uv run spoon-bot gateway` | CLI > YAML > env > defaults | YAML + env auto-discover | Multi-channel bot server |
| `uv run spoon-bot agent` | CLI > defaults | None (interactive REPL) | Single-user CLI |
| `uvicorn server:create_app` | env > YAML > defaults | None (REST/WS API only) | Docker / API deployment |

All paths load `.env` automatically via `python-dotenv`.

## Troubleshooting

### "Module not found" error

Install channel dependencies:
```bash
uv pip install -e ".[telegram]"
# or for all channels:
uv pip install -e ".[all-channels]"
```

### Bot not responding

1. Check logs for errors (look for `ERROR` or `WARNING` lines)
2. Verify token is correct: `echo $TELEGRAM_BOT_TOKEN`
3. Check user permissions — if `allowed_users` is set, your ID must be listed
4. For Discord: ensure Message Content Intent is enabled in Developer Portal

### Token not being picked up from `.env`

- Ensure `.env` is in the project root (same directory you run `uv run` from)
- Check for typos in variable names (`TELEGRAM_BOT_TOKEN`, not `TG_TOKEN`)
- Shell-exported values override `.env` values

### Webhook not receiving events

- Ensure webhook URL is publicly accessible (HTTPS required)
- Check webhook is properly configured in platform admin panel
- Verify webhook secret matches configuration

## Examples

### Example 1: Minimal Telegram Bot (`.env` only)

```bash
# .env
OPENROUTER_API_KEY=sk-or-v1-...
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
```

```bash
uv run spoon-bot gateway
```

### Example 2: Discord Server Bot

```yaml
# config.yaml
channels:
  discord:
    enabled: true
    accounts:
      - name: "server"
        token: "${DISCORD_BOT_TOKEN}"
        intents: [guilds, guild_messages, message_content]
        allowed_guilds: [987654321]
```

```bash
uv run spoon-bot gateway --channels discord
```

### Example 3: Multi-Platform Setup

```yaml
# config.yaml
agent:
  model: "anthropic/claude-sonnet-4"
  provider: "openrouter"

channels:
  telegram:
    enabled: true
    accounts:
      - name: "tg_bot"
        mode: "polling"
        allowed_users: [123456789]

  discord:
    enabled: true
    accounts:
      - name: "dc_bot"
        intents: [guilds, guild_messages, message_content, dm_messages]

  feishu:
    enabled: true
    accounts:
      - name: "feishu_bot"
        mode: "ws"
```

```bash
# .env — all credentials and access control in one place
OPENROUTER_API_KEY=sk-or-v1-...

TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_USER_ID=123456789

DISCORD_BOT_TOKEN=your_discord_token
DISCORD_GUILD_ID=123456789012345678
DISCORD_USER_ID=123456789012345678

FEISHU_APP_ID=your_app_id
FEISHU_APP_SECRET=your_app_secret
```

```bash
# Start all platforms
uv run spoon-bot gateway

# Or selectively
uv run spoon-bot gateway --channels telegram,feishu
```

## Support

- [GitHub Issues](https://github.com/XSpoonAi/spoon-bot/issues)
- [Documentation](https://github.com/XSpoonAi/spoon-bot#readme)
