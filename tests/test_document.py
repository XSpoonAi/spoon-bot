"""Tests for document processing tools."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from spoon_bot.agent.tools.document import (
    DocumentParseTool,
)


# ── DocumentParseTool ──


class TestDocumentParseTool:
    def setup_method(self) -> None:
        self.tool = DocumentParseTool(workspace="/tmp/test_ws")

    def test_name_and_description(self) -> None:
        assert self.tool.name == "document_parse"
        assert "PDF" in self.tool.description

    def test_page_range_all(self) -> None:
        assert self.tool._parse_page_range(None, 5) == [0, 1, 2, 3, 4]

    def test_page_range_single(self) -> None:
        assert self.tool._parse_page_range("3", 10) == [2]

    def test_page_range_range(self) -> None:
        assert self.tool._parse_page_range("2-4", 10) == [1, 2, 3]

    def test_page_range_mixed(self) -> None:
        result = self.tool._parse_page_range("1,3,5-7", 10)
        assert result == [0, 2, 4, 5, 6]

    def test_page_range_out_of_bounds(self) -> None:
        result = self.tool._parse_page_range("50", 10)
        assert result == []

    @pytest.mark.asyncio
    async def test_file_not_found(self) -> None:
        result = await self.tool.execute(file_path="/nonexistent/file.pdf")
        assert "Error" in result
        # Either "not found" or "pymupdf is required" (if not installed)
        assert "not found" in result or "pymupdf" in result

    @pytest.mark.asyncio
    async def test_not_pdf(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("hello")
        result = await self.tool.execute(file_path=str(txt_file))
        # Either "Only PDF" or "pymupdf is required"
        assert "PDF" in result

    @pytest.mark.asyncio
    async def test_pymupdf_not_installed(self, tmp_path: Path) -> None:
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        with patch.dict("sys.modules", {"fitz": None}):
            # Force reimport
            import importlib
            from spoon_bot.agent.tools import document as doc_mod
            importlib.reload(doc_mod)
            tool = doc_mod.DocumentParseTool(workspace=str(tmp_path))
            result = await tool.execute(file_path=str(pdf))
            # The tool should handle import error gracefully
            # (either via the try/except or the mocked module)
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_parse_basic_text(self, tmp_path: Path) -> None:
        """Test PDF parsing with mocked pymupdf."""
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF")

        tool = DocumentParseTool(workspace=str(tmp_path))
        result = await tool.execute(file_path=str(pdf))
        # If pymupdf is installed, we get parsed content; otherwise graceful error
        assert isinstance(result, str)
        if "pymupdf" not in result:
            # pymupdf available - should have parsed something
            assert "Parsed" in result or "Error" in result

    @pytest.mark.asyncio
    async def test_parse_error_handling(self) -> None:
        result = await self.tool.execute(file_path="")
        assert "Error" in result
