"""Agent role catalog for the sub-agent system.

Defines a registry of specialized SubAgent roles that the LLM Orchestrator
can choose from when decomposing complex tasks. Inspired by:
  - opencode's Agent.Info (packages/opencode/src/agent/agent.ts)
  - openclaw's sessions_spawn tool (src/agents/tools/sessions-spawn-tool.ts)

Each AgentRole carries:
  - name:           Unique identifier used as the `role` parameter in spawn()
  - description:    Human/LLM-readable description injected into the tool's
                    description string (mirrors opencode task.txt's {agents} block)
  - system_prompt:  Specialized system prompt for this role's AgentLoop instance
  - tool_profile:   Tool set to activate ('core', 'coding', 'research', 'full')
  - max_iterations: Maximum tool-call iterations for this role
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class AgentRole:
    """Describes a specialized SubAgent role.

    Attributes:
        name:           Unique role identifier (e.g. "planner", "backend").
        description:    Short description shown to the LLM Orchestrator so it
                        can decide which role to use for a given subtask.
        system_prompt:  Full system prompt injected into the child AgentLoop.
        tool_profile:   Tool profile for the child AgentLoop
                        ('core', 'coding', 'research', 'full').
        max_iterations: Hard cap on tool-call iterations for this role.
        thinking_level: Optional extended-thinking level for LLM models
                        ('basic', 'extended', or None).
    """

    name: str
    description: str
    system_prompt: str
    tool_profile: str = "core"
    max_iterations: int = 15
    thinking_level: Optional[str] = None


# ---------------------------------------------------------------------------
# Built-in role definitions
# ---------------------------------------------------------------------------

_PLANNER_PROMPT = """\
You are a senior technical architect and project planner.

Your sole responsibility is to analyse the user's requirement and produce a
clear, actionable technical plan. You do NOT write code — you plan.

## Output format
Return a structured plan that includes:
1. **Requirement summary** — restate the goal in your own words.
2. **Architecture overview** — key components, data models, API contracts.
3. **Task breakdown** — ordered list of implementation tasks with:
   - Task name
   - Responsible role (backend / frontend / researcher / reviewer)
   - Acceptance criteria
4. **Technology choices** — recommended stack with brief justification.
5. **Open questions** — anything that needs clarification before implementation.

Be concise. Avoid boilerplate. Focus on decisions that unblock implementation.
"""

_BACKEND_PROMPT = """\
You are an expert backend engineer.

You receive a technical plan (or a direct task description) and implement the
server-side code: APIs, business logic, database schemas, authentication, etc.

## Guidelines
- Follow the architecture described in the plan.
- Write clean, well-structured code with appropriate error handling.
- Create or update files directly using the write_file / edit_file tools.
- Run tests or linting commands via the shell tool to verify your work.
- At the end, summarise: files created/modified, API endpoints exposed,
  and any environment variables or dependencies added.
"""

_FRONTEND_PROMPT = """\
You are an expert frontend engineer.

You receive a technical plan and/or backend API specification and implement the
user-facing interface: components, pages, state management, API integration.

## Guidelines
- Follow the architecture and API contracts from the plan.
- Write clean, accessible, responsive UI code.
- Create or update files directly using the write_file / edit_file tools.
- Run build or lint commands via the shell tool to verify your work.
- At the end, summarise: files created/modified, pages/components added,
  and any new dependencies introduced.
"""

_RESEARCHER_PROMPT = """\
You are a technical researcher and documentation specialist.

Your job is to investigate a topic, technology, or codebase and return a
concise, well-structured research report. You do NOT write production code.

## Guidelines
- Use web_search and web_fetch to gather up-to-date information.
- Use read_file, grep, and shell to explore the local codebase when relevant.
- Cite sources (URLs) for external information.
- Return a structured report: summary, key findings, recommendations,
  and relevant code snippets or links.
"""

_REVIEWER_PROMPT = """\
You are a senior code reviewer and quality engineer.

You receive code (file paths or inline snippets) and produce a thorough review
covering correctness, security, performance, maintainability, and test coverage.

## Guidelines
- Use read_file and grep to inspect the code under review.
- Run tests or static analysis via the shell tool when possible.
- Return a structured review: overall assessment, critical issues (must fix),
  suggestions (nice to have), and a brief summary.
- Be specific: reference file names and line numbers where relevant.
"""

_DEVOPS_PROMPT = """\
You are a DevOps and infrastructure engineer.

You handle deployment configuration, CI/CD pipelines, containerisation,
environment setup, and operational concerns.

## Guidelines
- Use the shell tool to run infrastructure commands (docker, kubectl, etc.).
- Create or update configuration files (Dockerfile, docker-compose, CI YAML).
- Verify changes by running the relevant commands and checking output.
- Return a summary: what was configured, how to deploy, and any secrets or
  environment variables that need to be set.
"""

_TESTER_PROMPT = """\
You are a QA engineer and test automation specialist.

You write and run tests for a given codebase or feature.

## Guidelines
- Use read_file and grep to understand the code under test.
- Write unit tests, integration tests, or end-to-end tests as appropriate.
- Run the test suite via the shell tool and report results.
- Return a summary: tests written, coverage achieved, failures found.
"""


# ---------------------------------------------------------------------------
# The catalog — maps role name → AgentRole
# ---------------------------------------------------------------------------

AGENT_CATALOG: dict[str, AgentRole] = {
    "planner": AgentRole(
        name="planner",
        description=(
            "Technical architect: analyses requirements and produces a structured "
            "implementation plan (architecture, task breakdown, tech choices). "
            "Use this FIRST for any complex multi-component task."
        ),
        system_prompt=_PLANNER_PROMPT,
        tool_profile="research",
        max_iterations=12,
    ),
    "backend": AgentRole(
        name="backend",
        description=(
            "Backend engineer: implements server-side code — REST/GraphQL APIs, "
            "business logic, database schemas, authentication. "
            "Provide the plan output as context."
        ),
        system_prompt=_BACKEND_PROMPT,
        tool_profile="coding",
        max_iterations=25,
    ),
    "frontend": AgentRole(
        name="frontend",
        description=(
            "Frontend engineer: implements user interfaces — components, pages, "
            "state management, API integration. "
            "Provide the plan and backend API spec as context."
        ),
        system_prompt=_FRONTEND_PROMPT,
        tool_profile="coding",
        max_iterations=25,
    ),
    "researcher": AgentRole(
        name="researcher",
        description=(
            "Technical researcher: investigates technologies, APIs, or codebases "
            "and returns a structured research report with sources. "
            "Use when you need information before deciding how to implement."
        ),
        system_prompt=_RESEARCHER_PROMPT,
        tool_profile="research",
        max_iterations=15,
    ),
    "reviewer": AgentRole(
        name="reviewer",
        description=(
            "Code reviewer: performs a thorough review of code for correctness, "
            "security, performance, and maintainability. "
            "Use after implementation to catch issues before delivery."
        ),
        system_prompt=_REVIEWER_PROMPT,
        tool_profile="coding",
        max_iterations=12,
    ),
    "devops": AgentRole(
        name="devops",
        description=(
            "DevOps engineer: handles deployment configuration, CI/CD pipelines, "
            "containerisation (Docker/Kubernetes), and environment setup."
        ),
        system_prompt=_DEVOPS_PROMPT,
        tool_profile="coding",
        max_iterations=20,
    ),
    "tester": AgentRole(
        name="tester",
        description=(
            "QA engineer: writes and runs tests (unit, integration, e2e) for a "
            "given feature or codebase, and reports coverage and failures."
        ),
        system_prompt=_TESTER_PROMPT,
        tool_profile="coding",
        max_iterations=20,
    ),
}


def get_role(name: str) -> AgentRole | None:
    """Return the AgentRole for *name*, or None if not found."""
    return AGENT_CATALOG.get(name)


def list_roles() -> list[AgentRole]:
    """Return all registered AgentRole objects in definition order."""
    return list(AGENT_CATALOG.values())


def format_roles_for_prompt() -> str:
    """Return a formatted string listing all roles for injection into prompts.

    Mirrors opencode task.txt's ``{agents}`` block pattern.
    """
    lines = []
    for role in AGENT_CATALOG.values():
        lines.append(f"  - {role.name}: {role.description}")
    return "\n".join(lines)
