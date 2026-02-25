"""
Default configuration values for spoon-bot.

This module provides a single source of truth for default values used across
the codebase. All modules (agent/loop.py, core.py, config.py, etc.) should
import defaults from here to ensure consistency.

MIGRATION NOTE: If you change defaults here, they will affect:
- Agent initialization (AgentLoop, create_agent)
- SpoonBot configuration (SpoonBotConfig)
- CLI commands
- Example configuration files
"""

# -----------------------------------------------------------------------------
# LLM Provider Defaults
# -----------------------------------------------------------------------------

# Default model to use when none specified
DEFAULT_MODEL = "claude-sonnet-4-20250514"

# Default LLM provider
DEFAULT_PROVIDER = "anthropic"

# -----------------------------------------------------------------------------
# Agent Defaults
# -----------------------------------------------------------------------------

# Maximum tool call iterations before stopping
DEFAULT_MAX_ITERATIONS = 20

# Shell command timeout in seconds
DEFAULT_SHELL_TIMEOUT = 60

# Maximum output characters for shell commands
DEFAULT_MAX_OUTPUT = 10000

# Default session key
DEFAULT_SESSION_KEY = "default"

# -----------------------------------------------------------------------------
# Retry/Resilience Defaults
# -----------------------------------------------------------------------------

# Maximum retry attempts for channel operations
DEFAULT_RETRY_MAX_ATTEMPTS = 3

# Delay between retries in seconds
DEFAULT_RETRY_DELAY = 1.0

# Health check interval in seconds
DEFAULT_HEALTH_CHECK_INTERVAL = 60.0

# -----------------------------------------------------------------------------
# Channel Defaults
# -----------------------------------------------------------------------------

# Maximum media file size in MB for channels
DEFAULT_MEDIA_MAX_MB = 20

# -----------------------------------------------------------------------------
# Workspace Defaults
# -----------------------------------------------------------------------------

# Default workspace subdirectory under user home
DEFAULT_WORKSPACE_SUBDIR = ".spoon-bot/workspace"
