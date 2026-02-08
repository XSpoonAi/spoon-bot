---
name: document_export
description: Export text content to various formats (PDF, Excel, Markdown, Mermaid mindmap)
version: 1.0.0
author: XSpoon Team
tags: [document, export, pdf, excel, markdown, mermaid]
triggers:
  - type: keyword
    keywords: [export, convert, generate pdf, generate excel, generate markdown, mindmap, mermaid]
    priority: 80
  - type: pattern
    patterns:
      - "(?i)(export|convert|generate) .*(pdf|excel|markdown|mermaid|mindmap)"
      - "(?i)(create|make) .*(document|report|spreadsheet)"
    priority: 75
scripts:
  enabled: true
  definitions:
    - name: document_export
      description: Export content to PDF, Excel, Markdown, or Mermaid mindmap format
      type: python
      file: scripts/export.py
      timeout: 60
---

# Document Export Skill

Export text content to various document formats.

## Supported Formats
- **Markdown** — Plain markdown or formatted JSON
- **Mermaid** — Mindmap diagram in mermaid syntax
- **Excel** — Spreadsheet with optional multiple sheets (pipe-delimited rows)
- **PDF** — Simple text-based PDF document

## Usage
Provide content and a target format. Optionally specify a title and output path.

### Input (JSON via stdin)
```json
{
  "content": "Your text content here",
  "format": "markdown|pdf|excel|mermaid",
  "title": "Optional document title",
  "output_path": "Optional output file path"
}
```

### Output (JSON via stdout)
```json
{
  "success": true,
  "file_path": "/path/to/exported/file",
  "message": "Exported markdown: /path/to/file.md"
}
```
