"""Card message building for Feishu/Lark channel.

Feishu Interactive Cards (Schema 2.0) support full Markdown rendering
including code blocks, tables, and links — unlike plain text messages.
"""

from __future__ import annotations

import json
import re


def should_use_card(text: str) -> bool:
    """Return True if the text contains Markdown that benefits from card rendering.

    Detects:
    - Fenced code blocks: ```...```
    - Markdown tables: rows that look like |...|...| with a separator row

    Args:
        text: The message text to inspect.

    Returns:
        True if the text should be rendered as an interactive card.
    """
    # Fenced code block
    if "```" in text:
        return True

    # Markdown table: at least one pipe-separated row AND a separator row (|---|)
    lines = text.splitlines()
    has_pipe_row = any("|" in line for line in lines)
    has_separator = any(re.match(r"^\s*\|[\s\-|:]+\|\s*$", line) for line in lines)
    if has_pipe_row and has_separator:
        return True

    return False


def build_markdown_card(text: str) -> str:
    """Build a Feishu Interactive Card (Schema 2.0) wrapping Markdown content.

    The card uses a single ``markdown`` element in the body so that Feishu
    renders code blocks, tables, bold/italic, and hyperlinks correctly.

    Args:
        text: Markdown content to embed.

    Returns:
        JSON string suitable for use as the ``content`` field when sending
        a message with ``msg_type="interactive"``.
    """
    card = {
        "schema": "2.0",
        "config": {
            "wide_screen_mode": True,
        },
        "body": {
            "elements": [
                {
                    "tag": "markdown",
                    "content": text,
                }
            ]
        },
    }
    return json.dumps(card, ensure_ascii=False)
