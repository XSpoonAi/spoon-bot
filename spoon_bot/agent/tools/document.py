"""Document processing tools for parsing content."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spoon_bot.agent.tools.base import Tool, ToolParameterSchema
from spoon_bot.agent.tools.execution_context import capture_tool_output


class DocumentParseTool(Tool):
    """Parse PDF documents to extract text, tables, and images."""

    def __init__(self, workspace: str | Path = "./workspace") -> None:
        self.workspace = Path(workspace).resolve()
        self.images_dir = self.workspace / "extracted_images"

    @property
    def name(self) -> str:
        """Tool name used in function calls."""
        return "document_parse"

    @property
    def description(self) -> str:
        """Tool description for agent selection."""
        return "Parse PDF documents and extract text, tables, and images with page controls."

    @property
    def parameters(self) -> ToolParameterSchema:
        """JSON schema for parse parameters."""
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to PDF file",
                },
                "extract_text": {
                    "type": "boolean",
                    "description": "Extract plain text content",
                    "default": True,
                },
                "extract_tables": {
                    "type": "boolean",
                    "description": "Extract detected tables",
                    "default": True,
                },
                "extract_images": {
                    "type": "boolean",
                    "description": "Extract embedded images",
                    "default": False,
                },
                "page_range": {
                    "type": "string",
                    "description": "Pages like '1-10' or '1,3,5-7'",
                },
            },
            "required": ["file_path"],
        }

    def _parse_page_range(self, page_range: str | None, page_count: int) -> list[int]:
        if not page_range:
            return list(range(page_count))

        pages: set[int] = set()
        parts = [part.strip() for part in page_range.split(",") if part.strip()]
        for part in parts:
            if "-" in part:
                bounds = [p.strip() for p in part.split("-", 1)]
                if len(bounds) != 2:
                    continue
                start = int(bounds[0])
                end = int(bounds[1])
                if start > end:
                    start, end = end, start
                for page_num in range(start, end + 1):
                    if 1 <= page_num <= page_count:
                        pages.add(page_num - 1)
            else:
                page_num = int(part)
                if 1 <= page_num <= page_count:
                    pages.add(page_num - 1)

        if not pages:
            return []
        return sorted(pages)

    async def execute(
        self,
        file_path: str,
        extract_text: bool = True,
        extract_tables: bool = True,
        extract_images: bool = False,
        page_range: str | None = None,
        **_: Any,
    ) -> str:
        """Parse PDF and return structured extraction output."""
        try:
            import fitz  # type: ignore
        except ImportError:
            return "Error: pymupdf is required for PDF parsing. Install with: pip install pymupdf"

        pdf_path = Path(file_path).expanduser().resolve()
        if not pdf_path.exists():
            return f"Error: File not found: {file_path}"
        if pdf_path.suffix.lower() != ".pdf":
            return "Error: Only PDF files are supported"

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            return f"Error: Failed to open PDF: {exc}"

        try:
            if getattr(doc, "needs_pass", False):
                return "Error: PDF is password-protected"

            page_count = len(doc)
            try:
                selected_pages = self._parse_page_range(page_range, page_count)
            except Exception:
                return "Error: Invalid page_range format. Use like '1-10' or '1,3,5-7'"

            if not selected_pages:
                return "Error: No valid pages selected"

            warnings: list[str] = []
            if page_count > 100:
                warnings.append("Warning: Document has more than 100 pages")

            text_blocks: list[str] = []
            table_blocks: list[str] = []
            image_paths: list[str] = []

            if extract_images:
                self.images_dir.mkdir(parents=True, exist_ok=True)

            for page_index in selected_pages:
                page = doc[page_index]
                page_label = page_index + 1

                if extract_text:
                    text = page.get_text("text") or ""
                    if text.strip():
                        text_blocks.append(f"## Page {page_label}\n{text.strip()}")

                if extract_tables:
                    try:
                        tables = page.find_tables()
                        table_items = getattr(tables, "tables", []) if tables else []
                        for idx, table in enumerate(table_items, start=1):
                            rows = table.extract() if hasattr(table, "extract") else []
                            table_blocks.append(
                                f"## Page {page_label} Table {idx}\n{json.dumps(rows, ensure_ascii=False)}"
                            )
                    except Exception:
                        pass

                if extract_images:
                    try:
                        images = page.get_images(full=True)
                        for idx, img in enumerate(images, start=1):
                            xref = img[0]
                            image_info = doc.extract_image(xref)
                            image_bytes = image_info.get("image")
                            image_ext = image_info.get("ext", "png")
                            image_name = f"page_{page_label}_img_{idx}.{image_ext}"
                            image_path = self.images_dir / image_name
                            if image_bytes:
                                image_path.write_bytes(image_bytes)
                                image_paths.append(str(image_path))
                    except Exception:
                        pass

            result_parts: list[str] = [
                f"Parsed: {pdf_path.name}",
                f"Pages processed: {len(selected_pages)}/{page_count}",
            ]
            result_parts.extend(warnings)

            full_text_result = None
            if extract_text:
                full_text_result = "\n\n".join(text_blocks)
                joined_text = full_text_result
                if len(joined_text) > 50_000:
                    joined_text = joined_text[:50_000] + "\n\n[Truncated at 50KB]"
                result_parts.append("\n# Text\n" + (joined_text or "No text extracted"))

            if extract_tables:
                joined_tables = "\n\n".join(table_blocks)
                result_parts.append("\n# Tables\n" + (joined_tables or "No tables found"))

            if extract_images:
                images_text = "\n".join(image_paths) if image_paths else "No images extracted"
                result_parts.append("\n# Images\n" + images_text)

            summary_result = "\n".join(result_parts)
            if full_text_result is None:
                capture_tool_output(summary_result, summary_result)
                return summary_result

            full_parts = [
                f"Parsed: {pdf_path.name}",
                f"Pages processed: {len(selected_pages)}/{page_count}",
            ]
            full_parts.extend(warnings)
            full_parts.append("\n# Text\n" + (full_text_result or "No text extracted"))
            if extract_tables:
                joined_tables = "\n\n".join(table_blocks)
                full_parts.append("\n# Tables\n" + (joined_tables or "No tables found"))
            if extract_images:
                images_text = "\n".join(image_paths) if image_paths else "No images extracted"
                full_parts.append("\n# Images\n" + images_text)
            full_result = "\n".join(full_parts)
            capture_tool_output(summary_result, full_result)
            return summary_result
        finally:
            doc.close()
