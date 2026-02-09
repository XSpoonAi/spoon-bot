#!/usr/bin/env python3
"""Image generate skill script.

Reads JSON from stdin, generates an image using Pollinations.ai,
writes JSON result to stdout.

Input:  {"prompt": "...", "width": 512, "height": 512, "save_path": "..."}
Output: {"success": true, "file_path": "...", "size_bytes": N, "dimensions": "WxH"}
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path


def main() -> None:
    """Main entry point: read JSON from stdin, generate image, write JSON to stdout."""
    try:
        raw = sys.stdin.read()
        params = json.loads(raw)
    except (json.JSONDecodeError, Exception) as e:
        json.dump({"success": False, "error": f"Invalid JSON input: {e}"}, sys.stdout)
        return

    prompt = params.get("prompt", "")
    width = params.get("width", 512)
    height = params.get("height", 512)
    save_path = params.get("save_path")

    if not prompt or not prompt.strip():
        json.dump({"success": False, "error": "Missing or empty required field: prompt"}, sys.stdout)
        return

    # Validate dimensions
    try:
        width = int(width)
        height = int(height)
        if width < 64 or height < 64 or width > 2048 or height > 2048:
            json.dump(
                {"success": False, "error": "Dimensions must be between 64 and 2048"},
                sys.stdout,
            )
            return
    except (ValueError, TypeError):
        json.dump({"success": False, "error": "Invalid width or height"}, sys.stdout)
        return

    # Determine output path
    if save_path:
        output = Path(save_path).expanduser().resolve()
    else:
        output_dir = Path("workspace/generated_images")
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in "-_ " else "" for c in prompt[:50]).strip()
        safe_name = safe_name.replace(" ", "_") or "image"
        output = output_dir / f"{safe_name}_{stamp}.png"

    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Build Pollinations.ai URL
        encoded_prompt = urllib.parse.quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}"

        # Download image
        request = urllib.request.Request(url, headers={"User-Agent": "SpoonBot/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response:
            image_data = response.read()

        # Save to file
        output.write_bytes(image_data)
        size_bytes = len(image_data)

        json.dump(
            {
                "success": True,
                "file_path": str(output),
                "size_bytes": size_bytes,
                "dimensions": f"{width}x{height}",
            },
            sys.stdout,
        )
    except urllib.error.URLError as e:
        json.dump({"success": False, "error": f"Network error: {e}"}, sys.stdout)
    except Exception as exc:
        json.dump({"success": False, "error": f"Image generation failed: {exc}"}, sys.stdout)


if __name__ == "__main__":
    main()
