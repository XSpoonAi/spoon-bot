"""Constants for Feishu/Lark channel."""

# Feishu message character limit (conservative safe limit)
MAX_MESSAGE_LENGTH = 4000
SAFE_MESSAGE_LENGTH = 3800

# Chat types
CHAT_TYPE_P2P = "p2p"      # Direct message (private chat)
CHAT_TYPE_GROUP = "group"  # Group chat

# Inbound message types
MSG_TYPE_TEXT = "text"
MSG_TYPE_POST = "post"      # Rich text
MSG_TYPE_IMAGE = "image"
MSG_TYPE_FILE = "file"
MSG_TYPE_AUDIO = "audio"
MSG_TYPE_VIDEO = "video"
MSG_TYPE_STICKER = "sticker"

SUPPORTED_MSG_TYPES = (
    MSG_TYPE_TEXT,
    MSG_TYPE_POST,
    MSG_TYPE_IMAGE,
    MSG_TYPE_FILE,
    MSG_TYPE_AUDIO,
    MSG_TYPE_VIDEO,
    MSG_TYPE_STICKER,
)

# Outbound message types
MSG_TYPE_INTERACTIVE = "interactive"  # Card (schema 2.0)

# File types for upload
FILE_TYPE_OPUS = "opus"
FILE_TYPE_MP4 = "mp4"
FILE_TYPE_PDF = "pdf"
FILE_TYPE_DOC = "doc"
FILE_TYPE_XLS = "xls"
FILE_TYPE_PPT = "ppt"
FILE_TYPE_STREAM = "stream"  # Generic binary

# Emoji types for reactions
EMOJI_THUMBSUP = "THUMBSUP"
EMOJI_THUMBSDOWN = "THUMBSDOWN"
EMOJI_HEART = "HEART"
EMOJI_ONIT = "ONIT"         # "On it" — used as typing indicator
EMOJI_EYES = "EYES"         # "Seen"
EMOJI_DONE = "Done"         # Checkmark done
EMOJI_THINKING = "THINK"    # Thinking face

# Render modes (how to format outbound messages)
RENDER_MODE_AUTO = "auto"   # Detect markdown; use card when needed
RENDER_MODE_CARD = "card"   # Always use interactive card
RENDER_MODE_RAW = "raw"     # Always use plain text

# Sender name cache
SENDER_NAME_TTL = 600  # 10 minutes
SENDER_NAME_CACHE_MAX = 1000  # Max cached entries before eviction

# Message deduplication (WS reconnect can replay messages)
MESSAGE_DEDUP_TTL = 1800  # 30 minutes (seconds)
MESSAGE_DEDUP_MAX = 5000  # Max tracked message IDs
