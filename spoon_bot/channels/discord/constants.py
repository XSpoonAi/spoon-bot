"""Constants for Discord channel."""

# Discord message character limit
MAX_MESSAGE_LENGTH = 2000

# Leave margin to avoid edge cases
SAFE_MESSAGE_LENGTH = 1950

# Default intents needed for basic bot operation.
# message_content is always force-enabled in DiscordChannel._build_intents().
DEFAULT_INTENTS = [
    "guilds",
    "guild_messages",
    "dm_messages",
    "message_content",
]
