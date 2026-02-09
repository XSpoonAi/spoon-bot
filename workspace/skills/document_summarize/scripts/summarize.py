#!/usr/bin/env python3
"""Document summarize skill script.

Reads JSON from stdin, generates a structured summary,
writes JSON result to stdout.

Input:  {"text": "...", "style": "brief|detailed|bullet_points", "max_length": 500}
Output: {"success": true, "summary": "...", "key_points": [...], "word_count": N, "reading_time_minutes": N}
"""

from __future__ import annotations

import json
import math
import re
import sys


def extract_key_points(text: str, max_points: int = 10) -> list[str]:
    """Extract key points from text by finding significant sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out very short sentences
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    return sentences[:max_points]


def summarize_brief(text: str, max_length: int = 500) -> str:
    """Generate a brief summary by taking the first portion of text."""
    words = text.split()
    max_words = max(max_length // 5, 20)  # rough word count from char limit
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]) + "..."


def summarize_detailed(text: str, max_length: int = 2000) -> str:
    """Generate a more detailed summary."""
    words = text.split()
    max_words = max(max_length // 5, 50)
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]) + "..."


def summarize_bullet_points(text: str, max_points: int = 10) -> str:
    """Generate a bullet-point summary."""
    points = extract_key_points(text, max_points)
    if not points:
        return "No key points extracted."
    return "\n".join(f"• {p}" for p in points)


def main() -> None:
    """Main entry point: read JSON from stdin, summarize, write JSON to stdout."""
    try:
        raw = sys.stdin.read()
        params = json.loads(raw)
    except (json.JSONDecodeError, Exception) as e:
        json.dump({"success": False, "error": f"Invalid JSON input: {e}"}, sys.stdout)
        return

    text = params.get("text", "")
    style = params.get("style", "brief").lower().strip()
    max_length = params.get("max_length", 500)

    if not text or not text.strip():
        json.dump({"success": False, "error": "Missing or empty required field: text"}, sys.stdout)
        return

    if style not in {"brief", "detailed", "bullet_points"}:
        json.dump(
            {"success": False, "error": "Invalid style. Supported: brief, detailed, bullet_points"},
            sys.stdout,
        )
        return

    try:
        if style == "brief":
            summary = summarize_brief(text, max_length)
        elif style == "detailed":
            summary = summarize_detailed(text, max_length)
        else:  # bullet_points
            summary = summarize_bullet_points(text)

        key_points = extract_key_points(text)
        word_count = len(text.split())
        # Average reading speed: ~200 words per minute
        reading_time_minutes = max(1, math.ceil(word_count / 200))

        json.dump(
            {
                "success": True,
                "summary": summary,
                "key_points": key_points,
                "word_count": word_count,
                "reading_time_minutes": reading_time_minutes,
            },
            sys.stdout,
        )
    except Exception as exc:
        json.dump({"success": False, "error": f"Summarization failed: {exc}"}, sys.stdout)


if __name__ == "__main__":
    main()
