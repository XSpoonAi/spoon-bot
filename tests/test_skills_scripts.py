"""Tests for skill scripts (document_export, document_summarize, image_generate).

Each test pipes JSON to the script via stdin and checks JSON output on stdout.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
PYTHON = sys.executable

# Script paths
EXPORT_SCRIPT = PROJECT_ROOT / "workspace" / "skills" / "document_export" / "scripts" / "export.py"
SUMMARIZE_SCRIPT = PROJECT_ROOT / "workspace" / "skills" / "document_summarize" / "scripts" / "summarize.py"
GENERATE_SCRIPT = PROJECT_ROOT / "workspace" / "skills" / "image_generate" / "scripts" / "generate.py"


def run_script(script_path: Path, input_data: dict) -> dict:
    """Run a skill script with JSON input and return parsed JSON output."""
    result = subprocess.run(
        [PYTHON, str(script_path)],
        input=json.dumps(input_data),
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"
    return json.loads(result.stdout)


# ── Document Export Script ──


class TestDocumentExportScript:
    def test_export_markdown(self, tmp_path: Path) -> None:
        output = tmp_path / "test.md"
        result = run_script(EXPORT_SCRIPT, {
            "content": "# Hello World\nSome content here.",
            "format": "markdown",
            "output_path": str(output),
        })
        assert result["success"] is True
        assert result["file_path"] == str(output)
        assert "markdown" in result["message"].lower()
        assert output.exists()
        assert "Hello World" in output.read_text()

    def test_export_markdown_json_content(self, tmp_path: Path) -> None:
        output = tmp_path / "test.md"
        result = run_script(EXPORT_SCRIPT, {
            "content": json.dumps({"key": "value"}),
            "format": "markdown",
            "output_path": str(output),
        })
        assert result["success"] is True
        content = output.read_text()
        assert "key" in content

    def test_export_mermaid(self, tmp_path: Path) -> None:
        output = tmp_path / "mindmap.md"
        result = run_script(EXPORT_SCRIPT, {
            "content": "Point A\nPoint B\nPoint C",
            "format": "mermaid",
            "title": "Test Mindmap",
            "output_path": str(output),
        })
        assert result["success"] is True
        assert output.exists()
        content = output.read_text()
        assert "mindmap" in content
        assert "Test Mindmap" in content
        assert "Point A" in content

    def test_export_invalid_format(self) -> None:
        result = run_script(EXPORT_SCRIPT, {
            "content": "test",
            "format": "docx",
        })
        assert result["success"] is False
        assert "error" in result

    def test_export_missing_content(self) -> None:
        result = run_script(EXPORT_SCRIPT, {
            "format": "markdown",
        })
        assert result["success"] is False
        assert "error" in result

    def test_export_invalid_json(self) -> None:
        """Test with invalid JSON input."""
        proc = subprocess.run(
            [PYTHON, str(EXPORT_SCRIPT)],
            input="not valid json",
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(PROJECT_ROOT),
        )
        output = json.loads(proc.stdout)
        assert output["success"] is False
        assert "error" in output

    def test_export_default_output_path(self) -> None:
        """Test export with no output_path (uses default)."""
        result = run_script(EXPORT_SCRIPT, {
            "content": "Hello default path",
            "format": "markdown",
        })
        assert result["success"] is True
        assert result["file_path"]
        # Clean up
        Path(result["file_path"]).unlink(missing_ok=True)


# ── Document Summarize Script ──


class TestDocumentSummarizeScript:
    def test_summarize_brief(self) -> None:
        text = "This is a test document. It contains multiple sentences. " \
               "The purpose is to test summarization. We want to extract key points. " \
               "The summary should be concise and informative."
        result = run_script(SUMMARIZE_SCRIPT, {
            "text": text,
            "style": "brief",
        })
        assert result["success"] is True
        assert "summary" in result
        assert result["summary"]
        assert "key_points" in result
        assert isinstance(result["key_points"], list)
        assert "word_count" in result
        assert result["word_count"] > 0
        assert "reading_time_minutes" in result
        assert result["reading_time_minutes"] >= 1

    def test_summarize_detailed(self) -> None:
        text = "Detailed document content. " * 50
        result = run_script(SUMMARIZE_SCRIPT, {
            "text": text,
            "style": "detailed",
            "max_length": 2000,
        })
        assert result["success"] is True
        assert result["summary"]

    def test_summarize_bullet_points(self) -> None:
        text = "First important point about the topic. Second critical finding in the research. " \
               "Third conclusion from the analysis. Fourth recommendation for improvement."
        result = run_script(SUMMARIZE_SCRIPT, {
            "text": text,
            "style": "bullet_points",
        })
        assert result["success"] is True
        assert "•" in result["summary"]

    def test_summarize_empty_text(self) -> None:
        result = run_script(SUMMARIZE_SCRIPT, {
            "text": "",
            "style": "brief",
        })
        assert result["success"] is False
        assert "error" in result

    def test_summarize_invalid_style(self) -> None:
        result = run_script(SUMMARIZE_SCRIPT, {
            "text": "Some text",
            "style": "invalid_style",
        })
        assert result["success"] is False
        assert "error" in result

    def test_summarize_default_style(self) -> None:
        """Test that default style (brief) is used when not specified."""
        result = run_script(SUMMARIZE_SCRIPT, {
            "text": "A test document with enough content to summarize properly.",
        })
        assert result["success"] is True
        assert result["summary"]

    def test_summarize_invalid_json(self) -> None:
        proc = subprocess.run(
            [PYTHON, str(SUMMARIZE_SCRIPT)],
            input="{bad json",
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(PROJECT_ROOT),
        )
        output = json.loads(proc.stdout)
        assert output["success"] is False


# ── Image Generate Script ──


class TestImageGenerateScript:
    def test_generate_missing_prompt(self) -> None:
        result = run_script(GENERATE_SCRIPT, {
            "prompt": "",
        })
        assert result["success"] is False
        assert "error" in result

    def test_generate_invalid_dimensions(self) -> None:
        result = run_script(GENERATE_SCRIPT, {
            "prompt": "a cat",
            "width": 10,
            "height": 10,
        })
        assert result["success"] is False
        assert "Dimensions" in result["error"]

    def test_generate_invalid_dimensions_too_large(self) -> None:
        result = run_script(GENERATE_SCRIPT, {
            "prompt": "a cat",
            "width": 9999,
            "height": 512,
        })
        assert result["success"] is False

    def test_generate_invalid_json(self) -> None:
        proc = subprocess.run(
            [PYTHON, str(GENERATE_SCRIPT)],
            input="not json",
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(PROJECT_ROOT),
        )
        output = json.loads(proc.stdout)
        assert output["success"] is False

    @pytest.mark.skipif(
        not __import__("os").environ.get("TEST_NETWORK", ""),
        reason="Skipping network test (set TEST_NETWORK=1 to enable)",
    )
    def test_generate_image_network(self, tmp_path: Path) -> None:
        """Integration test that actually downloads from Pollinations.ai."""
        save_path = tmp_path / "test_image.png"
        result = run_script(GENERATE_SCRIPT, {
            "prompt": "a simple red circle",
            "width": 64,
            "height": 64,
            "save_path": str(save_path),
        })
        assert result["success"] is True
        assert save_path.exists()
        assert result["size_bytes"] > 0
        assert result["dimensions"] == "64x64"
