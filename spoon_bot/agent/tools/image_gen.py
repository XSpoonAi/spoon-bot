"""Image generation tool using Pollinations.ai free API."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from spoon_bot.agent.tools.base import Tool


class ImageGenerateTool(Tool):
    """Tool to generate images from text prompts using free APIs."""

    def __init__(self, workspace: Path | str | None = None) -> None:
        """Initialize image generation tool.

        Args:
            workspace: Base workspace directory for generated files.
        """
        self._workspace = Path(workspace) if workspace else Path.cwd()

    @property
    def name(self) -> str:
        return "image_generate"

    @property
    def description(self) -> str:
        return (
            "Generate an image from a text prompt using Pollinations.ai "
            "and save it to the workspace. Supports size, model, and seed options."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Text prompt describing the image to generate",
                },
                "width": {
                    "type": "integer",
                    "description": "Image width in pixels (default: 512, max: 2048)",
                },
                "height": {
                    "type": "integer",
                    "description": "Image height in pixels (default: 512, max: 2048)",
                },
                "seed": {
                    "type": "integer",
                    "description": "Optional random seed for reproducibility",
                },
                "model": {
                    "type": "string",
                    "description": "Optional model name (e.g. flux, sana, turbo)",
                },
                "save_path": {
                    "type": "string",
                    "description": "Optional custom output file path",
                },
                "enhance": {
                    "type": "boolean",
                    "description": "Enable prompt enhancement (default: false)",
                },
            },
            "required": ["prompt"],
        }

    def _validate_dimensions(self, width: int, height: int) -> str | None:
        """Validate requested image dimensions."""
        if width <= 0 or height <= 0:
            return "Error: width and height must be positive integers"
        if width > 2048 or height > 2048:
            return "Error: width and height must be <= 2048"
        return None

    def _sanitize_prompt_words(self, prompt: str) -> str:
        """Convert prompt text into a filesystem-safe slug."""
        words = prompt.strip().split()[:5]
        joined = "_".join(words)
        cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "", joined)
        return cleaned or "image"

    def _build_default_output_path(self, prompt: str) -> Path:
        """Build default output path under workspace generated_images dir."""
        output_dir = self._workspace / "generated_images"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_part = self._sanitize_prompt_words(prompt)
        filename = f"img_{timestamp}_{prompt_part}.jpg"
        return output_dir / filename

    def _build_url(self, prompt: str) -> str:
        """Build Pollinations prompt URL with encoded prompt text."""
        encoded_prompt = quote(prompt.strip(), safe="")
        return f"https://image.pollinations.ai/prompt/{encoded_prompt}"

    async def execute(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        seed: int | None = None,
        model: str | None = None,
        save_path: str | None = None,
        enhance: bool = False,
        **kwargs: Any,
    ) -> str:
        """Generate image with Pollinations API and save locally."""
        prompt_clean = (prompt or "").strip()
        if not prompt_clean:
            return "Error: prompt cannot be empty"

        dimension_error = self._validate_dimensions(width, height)
        if dimension_error:
            return dimension_error

        output_path = Path(save_path) if save_path else self._build_default_output_path(prompt_clean)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        params: dict[str, Any] = {
            "width": width,
            "height": height,
            "nologo": "true",
            "enhance": "true" if enhance else "false",
        }
        if seed is not None:
            params["seed"] = seed
        if model:
            params["model"] = model

        url = self._build_url(prompt_clean)

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()

            output_path.write_bytes(response.content)

            return (
                "Image generated successfully\n"
                f"Path: {output_path}\n"
                f"Prompt: {prompt_clean}\n"
                f"Size: {width}x{height}\n"
                f"Model: {model or 'default'}\n"
                f"Seed: {seed if seed is not None else 'random'}"
            )

        except httpx.TimeoutException:
            return "Error: image generation timed out after 60 seconds"
        except httpx.HTTPStatusError as e:
            return f"Error: image API returned status {e.response.status_code}"
        except httpx.HTTPError as e:
            return f"Error: network error during image generation: {e}"
        except OSError as e:
            return f"Error: failed to save image: {e}"

