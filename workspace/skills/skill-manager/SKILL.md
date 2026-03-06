---
name: skill-manager
description: Search, install, and remove skills from the skills.sh marketplace and GitHub
version: 1.0.0
tags: [management, skills, marketplace]
triggers:
  - type: keyword
    keywords: [install skill, search skill, find skill, remove skill, uninstall skill, skill marketplace, skills.sh, search_skills, install_skill, remove_skill]
    priority: 90
composable: true
---

# Skill Manager

Search the skills.sh directory, install skills from GitHub, and manage installed skills.

## Workflow

1. **Search** for skills using `skill_marketplace` tool with `action=search_skills`
2. **Install** a skill with `action=install_skill` and a GitHub URL or `owner/repo/skillId` shorthand
3. After installation, **activate** the `self_upgrade` tool and call `reload_skills` to load the new skill into the running agent
4. **Remove** a skill with `action=remove_skill`

## Important

After installing or removing a skill:
1. First activate the `self_upgrade` tool: `activate_tool(action='activate', tool_name='self_upgrade')`
2. Then reload skills: `self_upgrade(action='reload_skills')`
