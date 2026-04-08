"""Outbound rendering helpers for Feishu/Lark messages."""

from __future__ import annotations

import json


def normalize_markdown_for_feishu(text: str) -> str:
    """Normalize markdown before embedding it into a Feishu native post."""
    normalized = str(text or "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.strip("\n")


def build_post_message(text: str) -> str:
    """Build a Feishu native ``post`` payload that renders markdown content."""
    message_text = normalize_markdown_for_feishu(text)
    payload = {
        "zh_cn": {
            "content": [
                [
                    {
                        "tag": "md",
                        "text": message_text,
                    }
                ]
            ]
        }
    }
    return json.dumps(payload, ensure_ascii=False)

