---
name: git_helper
description: A skill for automating common Git operations with safety checks
version: 1.0.0
author: XSpoon Team
tags:
  - git
  - version-control
  - automation
triggers:
  - type: keyword
    keywords:
      - git commit
      - commit changes
      - push code
      - create branch
    priority: 80
  - type: pattern
    patterns:
      - "(?i)(commit|push|branch|merge|rebase)"
    priority: 75
composable: true
persist_state: false
---

# Git Helper Skill

A skill for automating common Git operations with safety checks.

## Instructions

When activated for Git operations:

1. **Pre-commit Checks**
   - Run linters if configured
   - Check for sensitive files
   - Verify test status

2. **Commit Message Format**
   - Use conventional commits format
   - Include scope when relevant
   - Add body for complex changes

3. **Branch Operations**
   - Verify branch naming conventions
   - Check for uncommitted changes
   - Confirm before destructive operations

4. **Push Safety**
   - Verify remote branch exists
   - Check for force push attempts
   - Confirm branch protection rules

## Examples

```bash
# Good commit message format
feat(auth): add OAuth2 support for GitHub login

- Add OAuth2 client configuration
- Implement token refresh mechanism
- Add user profile sync

Closes #123
```
