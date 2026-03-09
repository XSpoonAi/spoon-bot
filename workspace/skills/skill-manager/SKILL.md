---
name: skill-manager
description: Search, install, and remove skills from the skills.sh marketplace and GitHub
version: 1.0.0
tags: [management, skills, marketplace]
triggers:
  - type: keyword
    keywords: [install skill, search skill, find skill, remove skill, uninstall skill, skill marketplace, skills.sh, search_skills, install_skill, remove_skill, github.com]
    priority: 90
composable: true
---

# Skill Manager

Install skills from GitHub URLs, search skills.sh, and manage installed skills.

## Install a skill from GitHub

1. Call `skill_marketplace(action='install_skill', url='<github_url>')` — downloads the skill files
2. Call `self_upgrade(action='reload_skills')` — loads the new skill into the agent
3. Follow the instructions from the installed skill's SKILL.md (included in the install result)

## Other actions

- `skill_marketplace(action='search_skills', query='...')` — search skills.sh
- `skill_marketplace(action='remove_skill', skill_name='...')` — remove an installed skill
- `skill_marketplace(action='skill_info', skill_name='...')` — show skill details
