---
name: document_summarize
description: Summarize text content with key point extraction and configurable styles
version: 1.0.0
author: XSpoon Team
tags: [document, summarize, summary, analysis]
triggers:
  - type: keyword
    keywords: [summarize, summary, key points, bullet points, abstract, digest, tldr]
    priority: 80
  - type: pattern
    patterns:
      - "(?i)(summarize|create summary|generate summary|extract key points)"
      - "(?i)(brief|digest|tldr|overview) .*(document|text|content)"
    priority: 75
scripts:
  enabled: true
  definitions:
    - name: document_summarize
      description: Generate summaries with key points, word counts, and reading time estimates
      type: python
      file: scripts/summarize.py
      timeout: 30
---

# Document Summarize Skill

Provide structured summarization templates for document analysis.

## Supported Templates
- **general** — Title, summary, key points, conclusions
- **tokenomics** — Token name, supply, distribution, vesting, utility
- **roadmap** — Phases, milestones, dates, deliverables
- **whitepaper** — Abstract, problem, solution, technology, team

## Usage

### Input (JSON via stdin)
```json
{
  "text": "Document text to summarize...",
  "style": "brief|detailed|bullet_points",
  "max_length": 500
}
```

### Output (JSON via stdout)
```json
{
  "success": true,
  "summary": "...",
  "key_points": ["point1", "point2"],
  "word_count": 150,
  "reading_time_minutes": 1
}
```
