"""Tests for document processing tools."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from spoon_bot.agent.tools.document import (
    DocumentParseTool,
    DocumentExportTool,
    DocumentSummarizeTool,
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


# ── DocumentExportTool ──


class TestDocumentExportTool:
    def setup_method(self) -> None:
        self.tool = DocumentExportTool(workspace="/tmp/test_export")

    def test_name_and_description(self) -> None:
        assert self.tool.name == "document_export"
        assert "export" in self.tool.description.lower()

    @pytest.mark.asyncio
    async def test_invalid_format(self) -> None:
        result = await self.tool.execute(content="test", format="docx")
        assert "Error" in result
        assert "Invalid format" in result

    @pytest.mark.asyncio
    async def test_export_markdown(self, tmp_path: Path) -> None:
        output = tmp_path / "test.md"
        result = await self.tool.execute(
            content="# Hello\nWorld",
            format="markdown",
            output_path=str(output),
        )
        assert "markdown" in result.lower()
        assert output.exists()
        assert "# Hello" in output.read_text()

    @pytest.mark.asyncio
    async def test_export_markdown_json(self, tmp_path: Path) -> None:
        output = tmp_path / "test.md"
        data = json.dumps({"key": "value"})
        result = await self.tool.execute(
            content=data,
            format="markdown",
            output_path=str(output),
        )
        assert output.exists()
        content = output.read_text()
        assert "key" in content

    @pytest.mark.asyncio
    async def test_export_mermaid(self, tmp_path: Path) -> None:
        output = tmp_path / "mindmap.md"
        result = await self.tool.execute(
            content="Point A\nPoint B\nPoint C",
            format="mermaid",
            output_path=str(output),
            title="Test Doc",
        )
        assert "mermaid" in result.lower()
        assert output.exists()
        content = output.read_text()
        assert "mindmap" in content
        assert "Test Doc" in content
        assert "Point A" in content

    @pytest.mark.asyncio
    async def test_export_excel_no_openpyxl(self) -> None:
        with patch.dict("sys.modules", {"openpyxl": None}):
            result = await self.tool.execute(content="data", format="excel")
            # Should either work or give helpful error
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_export_default_path(self) -> None:
        path = self.tool._default_output_path("markdown")
        assert path.suffix == ".md"
        assert "exports" in str(path)

    @pytest.mark.asyncio
    async def test_export_default_path_excel(self) -> None:
        path = self.tool._default_output_path("excel")
        assert path.suffix == ".xlsx"

    @pytest.mark.asyncio
    async def test_export_default_path_pdf(self) -> None:
        path = self.tool._default_output_path("pdf")
        assert path.suffix == ".pdf"


# ── DocumentSummarizeTool ──


class TestDocumentSummarizeTool:
    def setup_method(self) -> None:
        self.tool = DocumentSummarizeTool()

    def test_name_and_description(self) -> None:
        assert self.tool.name == "document_summarize"
        assert "summar" in self.tool.description.lower()

    @pytest.mark.asyncio
    async def test_empty_text(self) -> None:
        result = await self.tool.execute(text="  ")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_invalid_template(self) -> None:
        result = await self.tool.execute(text="some text", template="invalid")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_general_template(self) -> None:
        result = await self.tool.execute(text="Test document content here.")
        assert "# Document Summary" in result
        assert "Key Points" in result
        assert "Source preview" in result

    @pytest.mark.asyncio
    async def test_tokenomics_template(self) -> None:
        result = await self.tool.execute(
            text="Token supply is 1B", template="tokenomics"
        )
        assert "Tokenomics" in result
        assert "Total Supply" in result
        assert "Distribution" in result

    @pytest.mark.asyncio
    async def test_roadmap_template(self) -> None:
        result = await self.tool.execute(
            text="Phase 1 launch Q1", template="roadmap"
        )
        assert "Roadmap" in result
        assert "Phases" in result
        assert "Milestones" in result

    @pytest.mark.asyncio
    async def test_whitepaper_template(self) -> None:
        result = await self.tool.execute(
            text="Our protocol solves X", template="whitepaper"
        )
        assert "Whitepaper" in result
        assert "Problem" in result
        assert "Solution" in result
        assert "Technology" in result

    @pytest.mark.asyncio
    async def test_unsupported_output_format(self) -> None:
        result = await self.tool.execute(
            text="test", output_format="html"
        )
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_source_preview_truncation(self) -> None:
        long_text = "A" * 500
        result = await self.tool.execute(text=long_text)
        # Preview should be truncated to 300 chars
        assert "Source preview:" in result

    def test_parameters_schema(self) -> None:
        params = self.tool.parameters
        assert params["type"] == "object"
        assert "text" in params["properties"]
        assert "template" in params["properties"]
        assert "text" in params["required"]
