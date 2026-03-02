"""Constants for Telegram bot commands and InlineKeyboard system."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Callback data prefixes (must be unique, kept short for 64-byte limit)
# ---------------------------------------------------------------------------

CALLBACK_PREFIX = {
    "model": "m:",
    "model_page": "mp:",
    "think": "t:",
    "verbose": "v:",
    "skill": "s:",
    "skill_page": "sp:",
    "confirm_clear": "cc:",
    # Phase 2: new commands
    "commands_page": "cp:",
    "reasoning": "r:",
    "usage_mode": "u:",
    "model_provider": "pv:",
    "model_list": "pvl:",
    "confirm_new": "cn:",
}

# ---------------------------------------------------------------------------
# Available models for /model command (flat list for quick-pick keyboard)
# ---------------------------------------------------------------------------

AVAILABLE_MODELS: list[dict[str, str]] = [
    {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4"},
    {"id": "claude-opus-4-20250514", "name": "Claude Opus 4"},
    {"id": "claude-haiku-3-5-20241022", "name": "Claude Haiku 3.5"},
    {"id": "gpt-4o", "name": "GPT-4o"},
    {"id": "gpt-4o-mini", "name": "GPT-4o Mini"},
    {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash"},
    {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro"},
    {"id": "deepseek-chat", "name": "DeepSeek Chat"},
    {"id": "deepseek-reasoner", "name": "DeepSeek Reasoner"},
]

MODELS_PER_PAGE = 6

# ---------------------------------------------------------------------------
# Provider-grouped models for /models command (two-level keyboard)
# ---------------------------------------------------------------------------

MODEL_PROVIDERS: dict[str, list[dict[str, str]]] = {
    "Anthropic": [
        {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4"},
        {"id": "claude-opus-4-20250514", "name": "Claude Opus 4"},
        {"id": "claude-haiku-3-5-20241022", "name": "Claude Haiku 3.5"},
    ],
    "OpenAI": [
        {"id": "gpt-4o", "name": "GPT-4o"},
        {"id": "gpt-4o-mini", "name": "GPT-4o Mini"},
    ],
    "Google": [
        {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash"},
        {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro"},
    ],
    "DeepSeek": [
        {"id": "deepseek-chat", "name": "DeepSeek Chat"},
        {"id": "deepseek-reasoner", "name": "DeepSeek Reasoner"},
    ],
}

MODELS_PER_PROVIDER_PAGE = 8

# ---------------------------------------------------------------------------
# Think levels for /think command
# ---------------------------------------------------------------------------

THINK_LEVELS: dict[str, str] = {
    "off": "Disabled",
    "basic": "Basic",
    "extended": "Extended",
}

# ---------------------------------------------------------------------------
# Reasoning display levels for /reasoning command
# ---------------------------------------------------------------------------

REASONING_LEVELS: dict[str, str] = {
    "off": "Off",
    "on": "On",
    "stream": "Stream",
}

# ---------------------------------------------------------------------------
# BotFather command list (registered via bot.set_my_commands)
# ---------------------------------------------------------------------------

BOT_COMMANDS: list[tuple[str, str]] = [
    ("start", "Start the bot"),
    ("help", "Show command list"),
    ("commands", "Browse all commands"),
    ("whoami", "Show your user info"),
    ("model", "Quick model switch"),
    ("models", "Browse models by provider"),
    ("think", "Set thinking level"),
    ("verbose", "Toggle verbose mode"),
    ("reasoning", "Toggle reasoning display"),
    ("skill", "Browse available skills"),
    ("tools", "List active tools"),
    ("status", "Show bot status"),
    ("history", "Show recent history"),
    ("context", "Show context window usage"),
    ("usage", "Show token usage stats"),
    ("memory", "Show memory summary"),
    ("note", "Save a note"),
    ("remember", "Remember a fact"),
    ("stop", "Stop current task"),
    ("new", "Start a new session"),
    ("compact", "Compact conversation context"),
    ("clear", "Clear conversation history"),
    ("cancel", "Cancel current operation"),
]

# ---------------------------------------------------------------------------
# Default per-user state
# ---------------------------------------------------------------------------

DEFAULT_USER_STATE: dict[str, object] = {
    "think_level": "off",
    "verbose": False,
    "reasoning": "off",
}
