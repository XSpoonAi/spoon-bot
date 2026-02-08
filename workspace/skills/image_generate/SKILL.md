---
name: image_generate
description: Generate images from text descriptions using Pollinations.ai (free, no API key needed)
version: 1.0.0
author: XSpoon Team
tags: [image, generate, ai, art, picture]
triggers:
  - type: keyword
    keywords: [generate image, create image, draw, picture, illustration, art, generate picture]
    priority: 85
  - type: pattern
    patterns:
      - "(?i)(generate|create|draw|make) .*(image|picture|illustration|art|photo)"
      - "(?i)(image|picture) .*(of|about|showing|depicting)"
    priority: 80
scripts:
  enabled: true
  definitions:
    - name: image_generate
      description: Generate an image from a text prompt using Pollinations.ai
      type: python
      file: scripts/generate.py
      timeout: 120
---

# Image Generate Skill

Generate images from text descriptions using Pollinations.ai (free, no API key needed).

## Usage

### Input (JSON via stdin)
```json
{
  "prompt": "A beautiful sunset over mountains",
  "width": 512,
  "height": 512,
  "save_path": "optional/path/to/save.png"
}
```

### Output (JSON via stdout)
```json
{
  "success": true,
  "file_path": "/path/to/generated/image.png",
  "size_bytes": 123456,
  "dimensions": "512x512"
}
```

## Notes
- Uses Pollinations.ai free API (no API key required)
- Default dimensions: 512x512
- Supports any text prompt for image generation
