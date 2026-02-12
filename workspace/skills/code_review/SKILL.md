---
name: code_review
description: A skill for performing automated code reviews with security and best practices analysis
version: 1.0.0
author: XSpoon Team
tags:
  - code
  - review
  - security
  - quality
triggers:
  - type: keyword
    keywords:
      - review code
      - code review
      - check code
      - analyze code
    priority: 80
  - type: pattern
    patterns:
      - "(?i)review\\s+(this|the)?\\s*(code|file|changes?)"
    priority: 75
composable: true
persist_state: false
---

# Code Review Skill

A skill for performing comprehensive code reviews focusing on:
- Security vulnerabilities (injection, XSS, etc.)
- Code quality and maintainability
- Performance issues
- Best practices adherence

## Instructions

When activated, analyze the provided code or file for:

1. **Security Analysis**
   - Check for injection vulnerabilities
   - Look for hardcoded credentials
   - Identify unsafe operations

2. **Code Quality**
   - Check naming conventions
   - Evaluate function complexity
   - Review error handling

3. **Performance**
   - Identify potential bottlenecks
   - Check for unnecessary operations
   - Review resource management

4. **Best Practices**
   - Verify documentation
   - Check for type hints
   - Review test coverage

## Output Format

Provide a structured review with:
- Summary (1-2 sentences)
- Findings (categorized by severity)
- Recommendations (actionable items)
- Code snippets with suggested fixes
