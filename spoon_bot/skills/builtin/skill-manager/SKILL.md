---
name: skill-manager
description: Search, install, and remove Agent Skills from skills.sh or confirmed skill GitHub sources
when_to_use: Use only when the user explicitly asks to install, search, remove, or inspect an Agent Skill, or after reviewing a GitHub/local source and confirming it contains a valid skill SKILL.md. Do not use for arbitrary GitHub repositories or files.
version: 1.0.0
tags: [management, skills, marketplace]
triggers:
  - type: keyword
    keywords: [install skill, search skill, find skill, remove skill, uninstall skill, skill marketplace, skills.sh, search_skills, install_skill, remove_skill]
    priority: 90
composable: true
---

# Skill Manager

Install confirmed Agent Skills from GitHub URLs, search skills.sh, and manage installed skills.

Do not use this skill for arbitrary GitHub repositories, GitHub files, or local files. Review those with normal tools first. Only install after the user explicitly asks for a skill install or the source has been confirmed to contain a valid skill `SKILL.md`.

## Install a skill from GitHub

Use this flow only for confirmed Agent Skill sources.

1. Call `skill_marketplace(action='install_skill', url='<github_url>')` — downloads the skill files
2. Call `self_upgrade(action='reload_skills')` — loads the new skill into the agent
3. Follow the instructions from the installed skill's SKILL.md (included in the install result)

## Other actions

- `skill_marketplace(action='search_skills', query='...')` — search skills.sh
- `skill_marketplace(action='remove_skill', skill_name='...')` — remove an installed skill
- `skill_marketplace(action='skill_info', skill_name='...')` — show skill details
