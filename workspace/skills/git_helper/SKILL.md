# Git Helper Skill

A skill for automating common Git operations with safety checks.

## Metadata

- **name**: git_helper
- **version**: 1.0.0
- **author**: XSpoon Team
- **tags**: git, version-control, automation

## Triggers

- Keywords: "git commit", "commit changes", "push code", "create branch"
- Pattern: `(commit|push|branch|merge|rebase)`

## Description

This skill helps with Git operations including:
- Creating well-formatted commit messages
- Branch management
- Safe push operations with checks
- Merge conflict resolution

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
