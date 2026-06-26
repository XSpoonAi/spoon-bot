"""Shell execution tool with comprehensive security guards and rate limiting."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger

from spoon_bot.agent.tools.base import Tool
from spoon_bot.agent.tools.execution_context import (
    capture_tool_output,
    current_session_fact_check_blocker,
    explicit_unavailable_tool_request_blocker,
    get_observed_cli_commands,
    get_request_execution_hints,
    get_tool_owner,
    observed_cli_command_matches,
    suppress_redundant_shell_file_read,
    suppress_repeated_background_job_poll,
)
from spoon_bot.config import (
    DEFAULT_MAX_OUTPUT,
    DEFAULT_SHELL_BACKGROUND_HANDOFF_TIMEOUT,
    DEFAULT_SHELL_MAX_TIMEOUT,
    DEFAULT_SHELL_TIMEOUT,
)
from spoon_bot.utils.privacy import SCRUBBED_ENV_VARS, is_sensitive_env_var
from spoon_bot.utils.rate_limit import (
    RateLimitConfig,
    get_rate_limiter,
)


class ShellSecurityError(Exception):
    """Raised when a command fails security validation."""
    pass


def _scrub_env(env: dict[str, str]) -> dict[str, str]:
    """Remove secret env vars that the agent must never see in output.

    Skills needing the private key should read from the keystore or
    the on-disk file; leaking it through the subprocess environment
    lets any ``env``/``printenv`` expose it into the conversation.
    """
    for key in SCRUBBED_ENV_VARS:
        env.pop(key, None)
    for key in list(env.keys()):
        if is_sensitive_env_var(key):
            env.pop(key, None)
    return env


class CommandValidator:
    """
    Validates shell commands for security risks.

    Security features:
    - Blocklist of dangerous commands
    - Detection of shell injection patterns
    - Pattern matching for dangerous operations
    - Optional whitelist mode
    """

    # Dangerous command patterns that should be blocked
    DANGEROUS_COMMANDS = frozenset({
        # File system destruction
        "rm -rf /",
        "rm -rf /*",
        "rm -rf ~",
        "rm -rf ~/*",
        "rm -rf .",
        "rm -rf ./*",
        "rm -fr /",
        "rm -fr /*",
        # Disk operations
        "mkfs",
        "dd if=/dev/zero",
        "dd if=/dev/random",
        "format c:",
        "format d:",
        "format e:",
        # Fork bombs and resource exhaustion
        ":(){ :|:& };:",
        # System modification
        "chmod -R 777 /",
        "chown -R",
        # Network attacks
        "nc -l",  # netcat listener (potential backdoor)
        # Windows specific
        "del /f /s /q c:\\",
        "rd /s /q c:\\",
    })

    # Regex patterns for dangerous operations
    DANGEROUS_PATTERNS = [
        # Recursive deletion at root or home
        re.compile(r"rm\s+(-[rfRF]+\s+)*[/~]($|\s|/\*)", re.IGNORECASE),
        # Fork bomb patterns
        re.compile(r":\s*\(\s*\)\s*\{.*\}"),
        # Disk formatting
        re.compile(r"mkfs\.[a-z0-9]+\s+/dev/", re.IGNORECASE),
        re.compile(r"dd\s+.*if=/dev/(zero|random|urandom)", re.IGNORECASE),
        # Dangerous chmod/chown
        re.compile(r"chmod\s+(-[rR]+\s+)*[0-7]{3,4}\s+/$", re.IGNORECASE),
        re.compile(r"chown\s+(-[rR]+\s+)*\S+:\S*\s+/$", re.IGNORECASE),
        # Windows format commands
        re.compile(r"format\s+[a-z]:", re.IGNORECASE),
        re.compile(r"(del|rd)\s+/[sfq]+\s+[a-z]:\\", re.IGNORECASE),
    ]

    # Shell metacharacters that enable command injection
    INJECTION_PATTERNS = [
        # Command chaining
        re.compile(r";\s*\S"),  # command1; command2
        re.compile(r"\|\|"),    # command1 || command2
        re.compile(r"&&"),      # command1 && command2
        # Newline / carriage-return multi-command injection
        re.compile(r"[\r\n]+\s*\S"),
        # Command substitution (skipped when allow_substitution is True)
        re.compile(r"\$\("),    # $(command)
        re.compile(r"`[^`]+`"), # `command`
        # Process substitution
        re.compile(r"<\("),     # <(command)
        re.compile(r">\("),     # >(command)
        # Dangerous redirections (exclude fd-prefixed like 2>/dev/null)
        re.compile(r">\s*/etc/"),           # Write to /etc/
        re.compile(r"(?<!\d)>\s*/dev/"),    # Write to /dev/ (but not 2>/dev/null)
        re.compile(r">\s*~/.ssh/"),         # Write to SSH config
        re.compile(r">\s*~/.bashrc"),    # Modify bashrc
        re.compile(r">\s*~/.profile"),   # Modify profile
        re.compile(r">\s*/root/"),       # Write to root home
    ]

    # Patterns that represent command substitution ($(...), `...`).
    # Skipped when allow_substitution is True.
    _SUBSTITUTION_PATTERNS: frozenset[str] = frozenset({
        r"\$\(",
        r"`[^`]+`",
    })

    # Sensitive file paths that should not be modified
    SENSITIVE_PATHS = frozenset({
        "/etc/passwd",
        "/etc/shadow",
        "/etc/sudoers",
        "/etc/hosts",
        "~/.ssh/authorized_keys",
        "~/.ssh/id_rsa",
        "~/.bashrc",
        "~/.profile",
        "~/.zshrc",
        "/root/",
    })

    # Default allowed commands (whitelist mode)
    DEFAULT_WHITELIST = frozenset({
        # Filesystem
        "ls", "dir", "pwd", "cd", "echo", "cat", "head", "tail", "grep",
        "find", "which", "where", "whoami", "date", "cal", "uname",
        "cp", "mv", "mkdir", "touch", "ln", "rm", "wc", "sort", "uniq",
        "chmod", "chown", "stat", "file", "basename", "dirname", "realpath",
        "tee", "xargs", "sed", "awk", "tr", "cut", "paste", "diff",
        # Version control
        "git", "gh", "svn",
        # JavaScript / Node.js
        "node", "npm", "npx", "pnpm", "yarn", "bun", "deno", "tsx", "ts-node",
        # Python
        "python", "python3", "pip", "pip3", "uv", "pipx", "poetry", "pdm",
        "pytest", "mypy", "ruff", "black", "isort", "flake8",
        # Rust
        "cargo", "rustc", "rustup",
        # Go
        "go",
        # Java / JVM
        "java", "javac", "mvn", "gradle",
        # C / C++
        "make", "cmake", "gcc", "g++", "clang",
        # Blockchain / Web3
        "cast", "forge", "anvil", "foundryup", "solc", "hardhat",
        # Containers / Infra
        "docker", "docker-compose", "podman", "kubectl", "helm", "terraform",
        # Network / HTTP
        "curl", "wget", "ping", "traceroute", "dig", "nslookup", "jq",
        # System info
        "ps", "top", "htop", "df", "du", "free", "uptime", "env", "printenv",
        # Archive
        "tar", "zip", "unzip", "gzip", "gunzip", "7z",
        # Editors / pagers
        "code", "vim", "nano", "less", "more",
        # Remote
        "ssh", "scp", "rsync",
        # Misc scripting
        "bash", "sh", "zsh", "source", "export", "set",
    })

    def __init__(
        self,
        whitelist_mode: bool = False,
        custom_whitelist: set[str] | None = None,
        custom_blocklist: set[str] | None = None,
        allow_pipes: bool = True,
        allow_chaining: bool = False,
        allow_substitution: bool = False,
        strict_mode: bool = False,
    ):
        """
        Initialize command validator.

        Args:
            whitelist_mode: If True, only allow whitelisted commands.
            custom_whitelist: Additional commands to whitelist.
            custom_blocklist: Additional commands/patterns to block.
            allow_pipes: If True, allow pipe (|) in commands.
            allow_chaining: If True, allow command chaining (&&, ||, ;).
                This lets the agent compose multi-step shell operations
                while still blocking dangerous commands and substitution.
            allow_substitution: If True, allow command substitution ($(...) and `...`).
                Enables running commands with variable expansion and template literals.
            strict_mode: If True, block all potentially dangerous patterns.
        """
        self.whitelist_mode = whitelist_mode
        self.whitelist = self.DEFAULT_WHITELIST.copy()
        if custom_whitelist:
            self.whitelist = self.whitelist | custom_whitelist

        self.custom_blocklist = custom_blocklist or set()
        self.allow_pipes = allow_pipes
        self.allow_chaining = allow_chaining
        self.allow_substitution = allow_substitution
        self.strict_mode = strict_mode

    def _extract_base_command(self, command: str) -> str:
        """Extract the base command from a full command string."""
        # Handle simple pipes by getting first command
        if self.allow_pipes and "|" in command:
            command = command.split("|")[0].strip()

        # Split and get first token
        parts = command.strip().split()
        if not parts:
            return ""

        base = parts[0]
        # Handle paths like /usr/bin/ls -> ls
        if "/" in base or "\\" in base:
            base = os.path.basename(base)

        return base.lower()

    def _check_dangerous_commands(self, command: str) -> str | None:
        """Check if command matches any dangerous command patterns.

        Uses word-boundary aware matching so that tokens appearing inside
        URLs or quoted strings (e.g. ``curl ...?format=3``) are not
        incorrectly flagged.
        """
        cmd_lower = command.lower().strip()

        # Check dangerous commands — match only when the dangerous string
        # appears as the start of the command or after a shell separator,
        # not inside an unrelated substring like a URL query parameter.
        for dangerous in self.DANGEROUS_COMMANDS:
            d_lower = dangerous.lower()
            # Quick substring pre-check
            if d_lower not in cmd_lower:
                continue
            # Build a word-boundary regex so "format c:" matches but
            # "?format=3" does not.
            escaped = re.escape(d_lower)
            if re.search(rf"(?:^|[\s;|&]){escaped}", cmd_lower):
                return f"Blocked dangerous command: '{dangerous}'"

        # Check custom blocklist with same boundary logic
        for blocked in self.custom_blocklist:
            b_lower = blocked.lower()
            if b_lower not in cmd_lower:
                continue
            escaped = re.escape(b_lower)
            if re.search(rf"(?:^|[\s;|&]){escaped}", cmd_lower):
                return f"Blocked by custom blocklist: '{blocked}'"

        # Check regex patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.search(command):
                return f"Blocked dangerous pattern: {pattern.pattern}"

        return None

    # Patterns that represent command chaining (&&, ||, ;, newline).
    # Skipped when allow_chaining is True.
    _CHAINING_PATTERNS: frozenset[str] = frozenset({
        r";\s*\S",
        r"\|\|",
        r"&&",
        r"[\r\n]+\s*\S",
    })

    def _check_injection_patterns(self, command: str) -> str | None:
        """Check for shell injection patterns."""
        for pattern in self.INJECTION_PATTERNS:
            # Skip chaining patterns when allow_chaining is enabled
            if self.allow_chaining and pattern.pattern in self._CHAINING_PATTERNS:
                continue
            # Skip substitution patterns when allow_substitution is enabled
            if self.allow_substitution and pattern.pattern in self._SUBSTITUTION_PATTERNS:
                continue
            match = pattern.search(command)
            if match:
                return f"Potential command injection detected: '{match.group()}'"

        # Check for simple pipe if not allowed
        if not self.allow_pipes and "|" in command:
            return "Pipe operator (|) not allowed"

        return None

    def _check_sensitive_paths(self, command: str) -> str | None:
        """Check if command accesses sensitive paths."""
        for path in self.SENSITIVE_PATHS:
            if path in command:
                return f"Access to sensitive path blocked: '{path}'"
        return None

    def _check_whitelist(self, command: str) -> str | None:
        """Check if command is in whitelist (when whitelist_mode is enabled)."""
        if not self.whitelist_mode:
            return None

        base_cmd = self._extract_base_command(command)
        if not base_cmd:
            return "Empty command"

        if base_cmd not in self.whitelist:
            return f"Command '{base_cmd}' not in whitelist"

        return None

    def validate(self, command: str) -> tuple[bool, str | None]:
        """
        Validate a command for security risks.

        Args:
            command: The command to validate.

        Returns:
            Tuple of (is_valid, error_message).
            If is_valid is False, error_message contains the reason.
        """
        if not command or not command.strip():
            return False, "Empty command"

        # Check dangerous commands first
        error = self._check_dangerous_commands(command)
        if error:
            return False, error

        # Check injection patterns
        error = self._check_injection_patterns(command)
        if error:
            return False, error

        # Check sensitive paths in strict mode
        if self.strict_mode:
            error = self._check_sensitive_paths(command)
            if error:
                return False, error

        # Check whitelist if enabled
        error = self._check_whitelist(command)
        if error:
            return False, error

        return True, None

    def sanitize_for_display(self, command: str, max_length: int = 100) -> str:
        """Sanitize command for safe logging/display."""
        # Truncate long commands
        if len(command) > max_length:
            return command[:max_length] + "..."
        return command


@dataclass
class _BackgroundShellJob:
    job_id: str
    command: str
    cwd: str
    process: Any
    stdout_task: asyncio.Task[None]
    stderr_task: asyncio.Task[None]
    buffer_limit: int
    owner_key: str = "default"
    stdout_text: str = ""
    stderr_text: str = ""
    status: str = "running"
    returncode: int | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def append_stdout(self, text: str) -> None:
        self.stdout_text = _append_capped_text(self.stdout_text, text, self.buffer_limit)

    def append_stderr(self, text: str) -> None:
        self.stderr_text = _append_capped_text(self.stderr_text, text, self.buffer_limit)


@dataclass(frozen=True)
class _SkillCommandTemplate:
    """Documented command shape parsed from a skill's SKILL.md."""

    prefix_tokens: tuple[str, ...]
    fixed_tokens: tuple[str, ...]
    positional_labels: tuple[str, ...]
    max_positionals: int
    allowed_flags: frozenset[str]
    flags_with_values: frozenset[str]
    flag_value_labels: tuple[tuple[str, str], ...]
    display: str


def _append_capped_text(existing: str, addition: str, limit: int) -> str:
    combined = existing + addition
    if len(combined) <= limit:
        return combined
    return combined[-limit:]


_SHELL_BACKGROUND_JOBS: dict[str, _BackgroundShellJob] = {}
_SILENT_BACKGROUND_TERMINATE_GRACE_SECONDS = 300.0


@dataclass(frozen=True)
class _ShellFileReadInspection:
    """A simple shell command that prints a file or a line range from a file."""

    path: Path
    start_line: int | None = 1
    end_line: int | None = None
    tail_lines: int | None = None


class ShellTool(Tool):
    """
    Tool to execute shell commands with comprehensive safety guards.

    Security features:
    - Command validation against dangerous patterns
    - Blocklist of destructive commands
    - Optional whitelist mode for maximum security
    - Detection of shell injection attempts
    - Timeout to prevent hanging
    - Output truncation to prevent context explosion
    - Working directory isolation
    - Safe argument parsing with shlex
    """

    _READ_ONLY_COMMANDS = frozenset({
        "basename",
        "cat",
        "date",
        "diff",
        "dir",
        "du",
        "echo",
        "file",
        "find",
        "free",
        "grep",
        "head",
        "jq",
        "less",
        "ls",
        "more",
        "pwd",
        "realpath",
        "rg",
        "sed",
        "stat",
        "tail",
        "uname",
        "wc",
        "where",
        "which",
        "whoami",
    })
    _READ_ONLY_GIT_SUBCOMMANDS = frozenset({
        "branch",
        "diff",
        "log",
        "remote",
        "rev-parse",
        "show",
        "status",
    })
    _READ_ONLY_VERSION_FLAGS = frozenset({"-v", "--version", "version"})
    _READ_ONLY_SKILL_ACTIONS = frozenset({
        "-h",
        "--help",
        "balance",
        "balances",
        "context",
        "get",
        "help",
        "history",
        "list",
        "logs",
        "show",
        "snapshot",
        "state",
        "status",
        "summary",
        "version",
        "wallet",
    })
    _READ_ONLY_SKILL_GROUP_ACTIONS = frozenset({
        "context",
        "get",
        "history",
        "list",
        "logs",
        "show",
        "snapshot",
        "state",
        "status",
        "summary",
    })
    _CURL_READ_ONLY_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
    _CURL_BODY_OR_UPLOAD_OPTIONS = frozenset({
        "--data",
        "--data-ascii",
        "--data-binary",
        "--data-raw",
        "--data-urlencode",
        "--form",
        "--form-string",
        "--json",
        "--upload-file",
    })
    _CURL_OUTPUT_FILE_OPTIONS = frozenset({
        "--create-dirs",
        "--output",
        "--output-dir",
        "--remote-header-name",
        "--remote-name",
        "--remote-name-all",
    })
    _WGET_BODY_OR_UPLOAD_OPTIONS = frozenset({
        "--body-data",
        "--body-file",
        "--method",
        "--post-data",
        "--post-file",
    })
    _SKILL_ENTRYPOINT_WRAPPERS = frozenset({
        "bash",
        "bun",
        "deno",
        "node",
        "npx",
        "npm",
        "pnpm",
        "poetry",
        "python",
        "python3",
        "sh",
        "tsx",
        "uv",
        "uvx",
        "yarn",
    })
    _SHELL_FILE_READ_TRACKING_MAX_BYTES = 20 * 1024 * 1024
    _DEFAULT_WORKSPACE_SKILL_FOREGROUND_TIMEOUT = 60
    _DEFAULT_BACKGROUND_JOB_POLL_WAIT_SECONDS = 30.0

    def __init__(
        self,
        timeout: int = DEFAULT_SHELL_TIMEOUT,
        max_timeout: int = DEFAULT_SHELL_MAX_TIMEOUT,
        max_output: int = DEFAULT_MAX_OUTPUT,
        working_dir: str | None = None,
        whitelist_mode: bool = False,
        custom_whitelist: set[str] | None = None,
        custom_blocklist: set[str] | None = None,
        allow_pipes: bool = True,
        allow_chaining: bool = False,
        allow_substitution: bool = False,
        strict_mode: bool = False,
        use_shell: bool = True,
        rate_limit_config: RateLimitConfig | None = None,
        background_handoff_timeout: float | None = None,
        yolo_mode: bool | None = None,
    ):
        """
        Initialize shell tool.

        Args:
            timeout: Default foreground timeout in seconds (default 600).
                     Commands exceeding this are moved to background.
            max_timeout: Maximum allowed per-command timeout override (default 7200).
            max_output: Maximum output characters (default 10000).
            working_dir: Default working directory.
            whitelist_mode: If True, only allow whitelisted commands.
            custom_whitelist: Additional commands to whitelist.
            custom_blocklist: Additional commands/patterns to block.
            allow_pipes: If True, allow pipe (|) in commands.
            allow_chaining: If True, allow && ; || for multi-step commands.
            allow_substitution: If True, allow $() and backtick substitution.
            strict_mode: If True, block all potentially dangerous patterns.
            use_shell: If True, use shell execution; if False, use direct exec.
            rate_limit_config: Configuration for rate limiting shell commands.
            background_handoff_timeout: Extra seconds to watch a command after
                its foreground budget expires before handing it to background.
            yolo_mode: If True, do not block shell commands with policy
                guardrails; execute them and let the command output be the
                source of truth. If None, read SPOON_BOT_YOLO_MODE.
        """
        self.timeout = timeout
        self.max_timeout = max(timeout, max_timeout)
        self.max_output = max_output
        self.working_dir = working_dir
        self.use_shell = use_shell
        env_yolo_mode = os.environ.get("SPOON_BOT_YOLO_MODE", "").strip().casefold()
        self.yolo_mode = (
            env_yolo_mode in {"1", "true", "yes", "y", "on"}
            if yolo_mode is None
            else bool(yolo_mode)
        )
        self.background_handoff_timeout = self._resolve_background_handoff_timeout(
            background_handoff_timeout
        )

        # Initialize command validator
        self.validator = CommandValidator(
            whitelist_mode=whitelist_mode,
            custom_whitelist=custom_whitelist,
            custom_blocklist=custom_blocklist,
            allow_pipes=allow_pipes,
            allow_chaining=allow_chaining,
            allow_substitution=allow_substitution,
            strict_mode=strict_mode,
        )

        # Initialize rate limiter
        self._rate_limit_config = rate_limit_config or RateLimitConfig.for_shell()
        self._rate_limiter = get_rate_limiter("shell_tool", self._rate_limit_config)
        logger.debug(
            f"ShellTool initialized with rate limit: "
            f"{self._rate_limit_config.requests_per_second}/s, "
            f"{self._rate_limit_config.requests_per_minute}/min"
        )

    @property
    def name(self) -> str:
        return "shell"

    @property
    def description(self) -> str:
        mode = "whitelist" if self.validator.whitelist_mode else "blocklist"
        timeout_min = self.timeout // 60
        max_min = self.max_timeout // 60
        return (
            "Execute a shell command or manage a long-running background shell job. "
            f"Default foreground budget: {timeout_min}min ({self.timeout}s). "
            f"Timeout is optional; omit it unless the user or command contract "
            f"explicitly needs a custom foreground budget (max {max_min}min). "
            "If the user provides an exact replay, simulation, dry-run, or no-op "
            "command, execute it exactly as provided; do not remove protective "
            "wrappers such as echo/printf or dry-run/no-op flags. "
            "Commands exceeding the budget keep running in the background — "
            "use job_status/job_output to monitor them. terminate_job requires "
            "force=true for a running job and should be used only when the "
            "latest user request cancels the job or separate evidence proves "
            "it is unrecoverably stuck. Silent running jobs are not stuck "
            "evidence; after one status check, use job_output/log evidence "
            "rather than repeating job_status with no new signal. "
            "Do not append unmanaged shell background operators such as '&'; "
            "run long-lived commands in the foreground so this tool can manage them. "
            "Do not install skills by cloning directly into workspace/skills; "
            "use the skill management tool for skill packages. "
            "For installed workspace skill CLI and skill dependency setup "
            "commands, run the exact command without pipes, "
            "output filters, file redirection, shell stderr/stdout merge, "
            "or shell timeout wrappers; this tool already captures stdout "
            "and stderr, and hands silent long-running skill commands to a "
            "managed background job. "
            f"Security mode: {mode}. "
            "Actions: execute, list_jobs, job_status, job_output, terminate_job. "
            "Dangerous commands and injection patterns are blocked."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute (required for action=execute)",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory for the command",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        f"Optional foreground timeout override in seconds "
                        f"(max {self.max_timeout}). Usually omit this; the tool "
                        f"uses a {self.timeout}s default and applies a shorter "
                        "managed-background handoff budget for stateful "
                        "workspace skill CLI commands. "
                        "Only provide this when the newest user request or the "
                        "command's own contract requires a custom foreground budget."
                    ),
                    "minimum": 1,
                    "maximum": self.max_timeout,
                },
                "action": {
                    "type": "string",
                    "enum": ["execute", "list_jobs", "job_status", "job_output", "terminate_job"],
                    "description": "Tool action. Defaults to execute.",
                    "default": "execute",
                },
                "job_id": {
                    "type": "string",
                    "description": "Background shell job ID for status/output/terminate actions",
                },
                "tail_chars": {
                    "type": "integer",
                    "description": "How many trailing characters of output to return for background job inspection",
                    "default": 4000,
                    "minimum": 200,
                    "maximum": 50000,
                },
                "wait_seconds": {
                    "type": "number",
                    "description": (
                        "For running background jobs, seconds to wait for "
                        "completion or new output before returning status. "
                        "Defaults to a short managed wait so long-running "
                        "commands can make progress without ending the task early."
                    ),
                    "minimum": 0,
                    "maximum": 120,
                },
                "force": {
                    "type": "boolean",
                    "description": (
                        "For terminate_job only. Set true only when the latest "
                        "user request explicitly cancels/abandons the running "
                        "job, or when separate evidence proves it is unrecoverably stuck."
                    ),
                    "default": False,
                },
            },
        }

    def tool_invocation_dedup_key(self, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """Normalize shell invocations for request-local duplicate suppression."""
        if not isinstance(arguments, dict):
            return arguments
        action = str(arguments.get("action") or "execute").strip().lower()
        if action in {"list_jobs", "job_status", "job_output"}:
            return None
        if action == "execute":
            command = arguments.get("command")
            if isinstance(command, str) and command.strip():
                try:
                    cwd = self._resolve_working_dir(arguments.get("working_dir"))
                except Exception:
                    cwd = str(self.working_dir or os.getcwd())
                skill_cli_key = self._skill_cli_invocation_dedup_key(command, cwd)
                if skill_cli_key is not None:
                    return {
                        "action": "execute",
                        "skill_cli": skill_cli_key,
                    }
                normalized = dict(arguments)
                normalized["command"] = self._normalize_exact_command(command)
                return normalized
        return arguments

    def tool_invocation_series_key(self, arguments: dict[str, Any]) -> str | None:
        """Group repeated side-effecting skill CLI actions within one request."""
        if not isinstance(arguments, dict):
            return None
        action = str(arguments.get("action") or "execute").strip().lower()
        if action != "execute":
            return None
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return None
        try:
            cwd = self._resolve_working_dir(arguments.get("working_dir"))
        except Exception:
            cwd = str(self.working_dir or os.getcwd())
        skill_cli_key = self._skill_cli_invocation_dedup_key(command, cwd)
        invocations = (
            skill_cli_key.get("invocations")
            if isinstance(skill_cli_key, dict)
            else None
        )
        if not isinstance(invocations, list) or not invocations:
            return None

        series_parts: list[str] = []
        for invocation in invocations:
            if not isinstance(invocation, dict):
                continue
            args = [
                str(arg).strip()
                for arg in invocation.get("args", [])
                if str(arg).strip()
            ]
            if not args or self._skill_args_are_read_only(args):
                continue
            skill = str(invocation.get("skill") or "").strip().casefold()
            action_name = args[0].casefold()
            if not skill or not action_name:
                continue
            skill_md = Path(cwd).resolve() / "skills" / skill / "SKILL.md"
            if not skill_md.exists():
                workspace_root = Path(self.working_dir or cwd).resolve()
                fallback_skill_md = workspace_root / "skills" / skill / "SKILL.md"
                if fallback_skill_md.exists():
                    skill_md = fallback_skill_md
            templates = self._parse_skill_command_templates(skill_md)
            if self._skill_args_contain_literal_placeholder(args, templates):
                continue
            series_action = self._skill_cli_side_effect_series_action(args, templates)
            if not series_action:
                continue
            series_identity = self._skill_cli_side_effect_series_identity(args, templates)
            series_parts.append(f"skill_cli:{skill}:{series_action}:values={series_identity}")
        if not series_parts:
            return None
        return "|".join(series_parts)

    @classmethod
    def _skill_cli_side_effect_series_action(
        cls,
        args: list[str],
        templates: list[_SkillCommandTemplate],
    ) -> str:
        """Return a side-effect series label from a documented CLI shape."""
        if not args or not templates:
            return ""
        matching_templates = [
            template
            for template in templates
            if tuple(args[: len(template.fixed_tokens)]) == template.fixed_tokens
        ]
        if not matching_templates:
            return ""
        template = sorted(
            matching_templates,
            key=lambda item: (len(item.fixed_tokens), item.max_positionals),
            reverse=True,
        )[0]
        fixed = "/".join(token.casefold() for token in template.fixed_tokens)
        positionals = cls._positional_values_for_template(args, template)
        flags = cls._observed_skill_cli_flags(args, template)
        flags_part = ",".join(flags) if flags else "none"
        return (
            f"{fixed}:pos{len(positionals)}of{template.max_positionals}:"
            f"flags={flags_part}"
        )

    @classmethod
    def _skill_cli_side_effect_series_identity(
        cls,
        args: list[str],
        templates: list[_SkillCommandTemplate],
    ) -> str:
        """Return a compact value identity for a documented side-effect call.

        The broad action shape is useful for diagnostics, but different
        positional or flag values can represent distinct external state
        transitions in a skill workflow. Include the observed values so a retry
        against a new challenge/id does not get collapsed into a prior action.
        """
        if not args or not templates:
            return "none"
        matching_templates = [
            template
            for template in templates
            if tuple(args[: len(template.fixed_tokens)]) == template.fixed_tokens
        ]
        if not matching_templates:
            return "none"
        template = sorted(
            matching_templates,
            key=lambda item: (len(item.fixed_tokens), item.max_positionals),
            reverse=True,
        )[0]
        positionals = cls._positional_values_for_template(args, template)
        flag_values = cls._observed_skill_cli_flag_values(args, template)
        identity = repr((positionals, flag_values))
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]

    def format_duplicate_invocation_result(
        self,
        duplicate_result: str,
        arguments: dict[str, Any],
        dedup_arguments: Any,
    ) -> str:
        """Return duplicate results without adding shell-specific STOP messages."""
        if (
            not isinstance(arguments, dict)
            or str(arguments.get("action") or "execute").strip().lower() != "execute"
        ):
            return duplicate_result

        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return duplicate_result

        try:
            cwd = self._resolve_working_dir(arguments.get("working_dir"))
        except Exception:
            cwd = str(self.working_dir or os.getcwd())

        if not self._command_is_read_only_inspection(command, cwd, dedup_arguments):
            return duplicate_result

        return duplicate_result

    def runtime_invocation_category(self, kwargs: dict[str, Any]) -> str | None:
        """Classify shell calls for request-local progress guards."""
        action = str((kwargs or {}).get("action") or "execute").strip().casefold()
        if action != "execute":
            return "read_only" if action in {"job_status", "job_output"} else "stateful"

        command = str((kwargs or {}).get("command") or "").strip()
        if not command:
            return "read_only"
        try:
            cwd = self._resolve_working_dir((kwargs or {}).get("working_dir"))
            dedup_arguments = self.tool_invocation_dedup_key(kwargs)
            if self._command_is_skill_cli_runtime_state_query(
                command,
                cwd,
                dedup_arguments,
            ):
                return "stateful"
            if self._command_is_read_only_inspection(command, cwd, dedup_arguments):
                return "read_only"
        except Exception:
            if self.command_is_plain_read_only_inspection(command):
                return "read_only"
        return "stateful"

    @classmethod
    def _skill_args_are_help_or_metadata(cls, args: list[str]) -> bool:
        """Return True when a skill CLI call only asks for command metadata.

        Runtime state queries are execution evidence for the installed skill.
        They must not be treated the same as reading SKILL.md, directory
        listings, or CLI help text when the skill-inspection budget is
        exhausted.
        """
        if not args:
            return True
        normalized = [
            str(arg or "").strip().casefold()
            for arg in args
            if str(arg or "").strip()
        ]
        if not normalized:
            return True
        metadata_tokens = {"-h", "--help", "help", "-v", "--version", "version"}
        return all(arg in metadata_tokens for arg in normalized)

    def _command_is_skill_cli_runtime_state_query(
        self,
        command: str,
        cwd: str,
        dedup_arguments: Any = None,
    ) -> bool:
        """Return True for direct installed-skill CLI calls that query live state.

        This is intentionally skill-agnostic: it relies on the workspace skill
        entrypoint parser, not on game names or prompt wording. CLI help/version
        calls remain read-only inspection; other direct skill CLI calls are
        allowed to run so the model can follow SKILL.md using real tool output.
        """
        skill_cli_key = (
            dedup_arguments.get("skill_cli")
            if isinstance(dedup_arguments, dict)
            else None
        )
        if skill_cli_key is None:
            skill_cli_key = self._skill_cli_invocation_dedup_key(command, cwd)
        if not isinstance(skill_cli_key, dict):
            return False
        invocations = skill_cli_key.get("invocations")
        if not isinstance(invocations, list) or not invocations:
            return False
        for invocation in invocations:
            if not isinstance(invocation, dict):
                return False
            args = [
                str(arg)
                for arg in invocation.get("args", [])
                if str(arg).strip()
            ]
            if self._skill_args_are_help_or_metadata(args):
                return False
        return True

    @classmethod
    def _skill_args_are_read_only(cls, args: list[str]) -> bool:
        if not args:
            return False
        action = str(args[0] or "").strip().casefold()
        if action in cls._READ_ONLY_SKILL_ACTIONS:
            return True
        if (
            len(args) >= 2
            and str(args[1] or "").strip().casefold()
            in cls._READ_ONLY_SKILL_GROUP_ACTIONS
        ):
            return True
        return False

    @classmethod
    def _segment_is_read_only_inspection(cls, segment: list[str]) -> bool:
        cleaned = cls._segment_before_redirection_or_pipe(segment)
        if not cleaned:
            return False
        command_name = os.path.basename(
            str(cleaned[0] or "").strip().strip("'\"").replace("\\", "/")
        ).casefold()
        if command_name == "curl":
            return cls._curl_segment_is_read_only_inspection(cleaned)
        if command_name == "wget":
            return cls._wget_segment_is_read_only_inspection(cleaned)
        if command_name in cls._READ_ONLY_COMMANDS:
            return True
        if command_name == "git" and len(cleaned) >= 2:
            return str(cleaned[1] or "").strip().casefold() in cls._READ_ONLY_GIT_SUBCOMMANDS
        if (
            command_name in {"node", "npm", "npx", "pnpm", "python", "python3"}
            and len(cleaned) >= 2
        ):
            return str(cleaned[1] or "").strip().casefold() in cls._READ_ONLY_VERSION_FLAGS
        return False

    @classmethod
    def _curl_segment_is_read_only_inspection(cls, segment: list[str]) -> bool:
        """Return True when a curl command only retrieves response data."""
        method = "GET"
        tokens = [str(token or "").strip() for token in segment[1:] if str(token or "").strip()]
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token in {"-I", "--head"}:
                method = "HEAD"
                index += 1
                continue
            if token in {"-X", "--request"}:
                if index + 1 >= len(tokens):
                    return False
                method = tokens[index + 1].strip().upper()
                index += 2
                continue
            if token.startswith("--request="):
                method = token.split("=", 1)[1].strip().upper()
                index += 1
                continue
            if token.startswith("-X") and len(token) > 2:
                method = token[2:].strip().upper()
                index += 1
                continue
            if token in {"-d", "-F", "-T", "-o", "-O", "-J"}:
                return False
            if len(token) > 2 and token[:2] in {"-d", "-F", "-T", "-o"}:
                return False
            long_name = token.split("=", 1)[0]
            if long_name in cls._CURL_BODY_OR_UPLOAD_OPTIONS:
                return False
            if long_name in cls._CURL_OUTPUT_FILE_OPTIONS:
                return False
            index += 1
        return method in cls._CURL_READ_ONLY_METHODS

    @classmethod
    def _wget_segment_is_read_only_inspection(cls, segment: list[str]) -> bool:
        """Return True when wget is explicitly configured for stdout/probe output."""
        tokens = [str(token or "").strip() for token in segment[1:] if str(token or "").strip()]
        writes_to_stdout = False
        spider = False
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token == "--spider":
                spider = True
                index += 1
                continue
            if token in {"-O", "--output-document"}:
                if index + 1 >= len(tokens):
                    return False
                writes_to_stdout = tokens[index + 1] == "-"
                if not writes_to_stdout:
                    return False
                index += 2
                continue
            if token.startswith("-O") and len(token) > 2:
                writes_to_stdout = token[2:] == "-"
                if not writes_to_stdout:
                    return False
                index += 1
                continue
            if token.startswith("-") and not token.startswith("--") and "O" in token[1:]:
                output_flag_index = token.find("O", 1)
                output_value = token[output_flag_index + 1 :]
                if not output_value:
                    if index + 1 >= len(tokens):
                        return False
                    output_value = tokens[index + 1]
                    index += 1
                writes_to_stdout = output_value == "-"
                if not writes_to_stdout:
                    return False
                index += 1
                continue
            if token.startswith("--output-document="):
                writes_to_stdout = token.split("=", 1)[1] == "-"
                if not writes_to_stdout:
                    return False
                index += 1
                continue
            long_name = token.split("=", 1)[0]
            if long_name in cls._WGET_BODY_OR_UPLOAD_OPTIONS:
                return False
            index += 1
        return writes_to_stdout or spider

    def _command_is_read_only_inspection(
        self,
        command: str,
        cwd: str,
        dedup_arguments: Any = None,
    ) -> bool:
        skill_cli_key = (
            dedup_arguments.get("skill_cli")
            if isinstance(dedup_arguments, dict)
            else None
        )
        if skill_cli_key is None:
            skill_cli_key = self._skill_cli_invocation_dedup_key(command, cwd)
        if isinstance(skill_cli_key, dict):
            invocations = skill_cli_key.get("invocations")
            if isinstance(invocations, list) and invocations:
                return all(
                    isinstance(invocation, dict)
                    and self._skill_args_are_read_only(
                        [
                            str(arg)
                            for arg in invocation.get("args", [])
                            if str(arg).strip()
                        ]
                    )
                    for invocation in invocations
                )

        return self.command_is_plain_read_only_inspection(command)

    @classmethod
    def command_is_plain_read_only_inspection(cls, command: str) -> bool:
        """Return True for shell commands that only inspect local state."""
        try:
            command_tokens = shlex.split(str(command or ""))
        except ValueError:
            command_tokens = str(command or "").split()
        if not command_tokens:
            return False

        segments = [
            segment
            for segment in cls._split_shell_segments(command_tokens)
            if segment and segment[0].casefold() != "cd"
        ]
        return bool(segments) and all(
            cls._segment_is_read_only_inspection(segment)
            for segment in segments
        )

    def _shell_file_read_inspection(
        self,
        command: str,
        cwd: str,
    ) -> _ShellFileReadInspection | None:
        """Return the inspected file/range for a pure shell file read."""
        try:
            command_tokens = shlex.split(str(command or ""))
        except ValueError:
            return None
        if not command_tokens or "|" in command_tokens:
            return None

        current_dir = Path(cwd).resolve()
        inspections: list[_ShellFileReadInspection] = []
        for segment in self._split_shell_segments(command_tokens):
            if not segment:
                continue
            if segment[0].casefold() == "cd":
                current_dir = self._resolve_cd_segment(current_dir, segment)
                continue
            cleaned = self._segment_before_redirection_or_pipe(segment)
            if not cleaned:
                continue
            inspection = self._extract_shell_file_read_segment(cleaned, current_dir)
            if inspection is None:
                return None
            inspections.append(inspection)

        if len(inspections) != 1:
            return None
        return inspections[0]

    @classmethod
    def _extract_shell_file_read_segment(
        cls,
        segment: list[str],
        current_dir: Path,
    ) -> _ShellFileReadInspection | None:
        if not segment:
            return None
        command_name = os.path.basename(
            str(segment[0] or "").strip().strip("'\"").replace("\\", "/")
        ).casefold()
        if command_name == "cat":
            path = cls._single_shell_read_path(segment[1:], current_dir)
            if path is None:
                return None
            return _ShellFileReadInspection(path=path)
        if command_name == "head":
            return cls._head_tail_file_read_inspection(segment[1:], current_dir, tail=False)
        if command_name == "tail":
            return cls._head_tail_file_read_inspection(segment[1:], current_dir, tail=True)
        if command_name == "sed":
            return cls._sed_file_read_inspection(segment[1:], current_dir)
        return None

    @staticmethod
    def _resolve_shell_read_path(raw_path: str, current_dir: Path) -> Path | None:
        value = str(raw_path or "").strip()
        if not value or value == "-" or value.startswith("-"):
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = current_dir / path
        try:
            path = path.resolve()
        except OSError:
            path = path.absolute()
        try:
            if not path.is_file():
                return None
        except OSError:
            return None
        return path

    @classmethod
    def _single_shell_read_path(
        cls,
        args: list[str],
        current_dir: Path,
    ) -> Path | None:
        paths: list[Path] = []
        for arg in args:
            token = str(arg or "").strip()
            if not token:
                continue
            if token.startswith("-"):
                continue
            path = cls._resolve_shell_read_path(token, current_dir)
            if path is None:
                return None
            paths.append(path)
        if len(paths) != 1:
            return None
        return paths[0]

    @classmethod
    def _head_tail_file_read_inspection(
        cls,
        args: list[str],
        current_dir: Path,
        *,
        tail: bool,
    ) -> _ShellFileReadInspection | None:
        line_count = 10
        tail_from_line: int | None = None
        paths: list[Path] = []
        index = 0
        while index < len(args):
            token = str(args[index] or "").strip()
            lowered = token.casefold()
            if not token:
                index += 1
                continue
            if lowered in {"-f", "--follow"} or lowered.startswith("--follow="):
                return None
            if lowered in {"-c", "--bytes"} or lowered.startswith(("-c", "--bytes=")):
                return None
            if lowered in {"-n", "--lines"}:
                if index + 1 >= len(args):
                    return None
                raw_count = str(args[index + 1] or "").strip()
                if tail and raw_count.startswith("+") and raw_count[1:].isdigit():
                    tail_from_line = max(1, int(raw_count[1:]))
                elif raw_count.isdigit():
                    line_count = max(1, int(raw_count))
                else:
                    return None
                index += 2
                continue
            if lowered.startswith("--lines="):
                raw_count = token.split("=", 1)[1].strip()
                if tail and raw_count.startswith("+") and raw_count[1:].isdigit():
                    tail_from_line = max(1, int(raw_count[1:]))
                elif raw_count.isdigit():
                    line_count = max(1, int(raw_count))
                else:
                    return None
                index += 1
                continue
            if lowered.startswith("-n") and len(token) > 2:
                raw_count = token[2:].strip()
                if raw_count.isdigit():
                    line_count = max(1, int(raw_count))
                    index += 1
                    continue
                return None
            if re.fullmatch(r"-\d+", token):
                line_count = max(1, int(token[1:]))
                index += 1
                continue
            if token.startswith("-"):
                index += 1
                continue
            path = cls._resolve_shell_read_path(token, current_dir)
            if path is None:
                return None
            paths.append(path)
            index += 1

        if len(paths) != 1:
            return None
        if tail:
            if tail_from_line is not None:
                return _ShellFileReadInspection(path=paths[0], start_line=tail_from_line)
            return _ShellFileReadInspection(path=paths[0], tail_lines=line_count)
        return _ShellFileReadInspection(path=paths[0], start_line=1, end_line=line_count)

    @classmethod
    def _sed_file_read_inspection(
        cls,
        args: list[str],
        current_dir: Path,
    ) -> _ShellFileReadInspection | None:
        quiet = False
        script: str | None = None
        paths: list[Path] = []
        index = 0
        while index < len(args):
            token = str(args[index] or "").strip()
            lowered = token.casefold()
            if not token:
                index += 1
                continue
            if lowered in {"-n", "--quiet", "--silent"}:
                quiet = True
                index += 1
                continue
            if lowered in {"-e", "--expression"}:
                if index + 1 >= len(args):
                    return None
                script = str(args[index + 1] or "").strip()
                index += 2
                continue
            if lowered.startswith("-e") and len(token) > 2:
                script = token[2:].strip()
                index += 1
                continue
            if lowered.startswith("-i") or lowered in {"-f", "--file"}:
                return None
            if token.startswith("-"):
                return None
            if script is None:
                script = token
            else:
                path = cls._resolve_shell_read_path(token, current_dir)
                if path is None:
                    return None
                paths.append(path)
            index += 1

        if not quiet or not script or len(paths) != 1:
            return None
        match = re.fullmatch(r"\s*(\d+)\s*(?:,\s*(\d+|\$)\s*)?p\s*", script)
        if not match:
            return None
        start = max(1, int(match.group(1)))
        raw_end = match.group(2)
        end = None if raw_end in {None, "$"} else max(start, int(raw_end))
        return _ShellFileReadInspection(path=paths[0], start_line=start, end_line=end)

    def _shell_file_read_content_info(self, path: Path) -> tuple[int, str] | None:
        try:
            stat_result = path.stat()
        except OSError:
            return None
        if stat_result.st_size > self._SHELL_FILE_READ_TRACKING_MAX_BYTES:
            return None

        digest = hashlib.sha256()
        newline_count = 0
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                    newline_count += chunk.count(b"\n")
        except OSError:
            return None
        return max(1, newline_count + 1), digest.hexdigest()

    def _suppress_redundant_shell_file_read(
        self,
        command: str,
        cwd: str,
    ) -> str | None:
        inspection = self._shell_file_read_inspection(command, cwd)
        if inspection is None:
            return None
        content_info = self._shell_file_read_content_info(inspection.path)
        if content_info is None:
            return None

        total_lines, fingerprint = content_info
        if inspection.tail_lines is not None:
            start_line = max(1, total_lines - inspection.tail_lines + 1)
            end_line = total_lines
        else:
            start_line = max(1, int(inspection.start_line or 1))
            end_line = (
                total_lines
                if inspection.end_line is None
                else min(total_lines, max(start_line, int(inspection.end_line)))
            )
        limit = max(1, end_line - start_line + 1)
        return suppress_redundant_shell_file_read(
            str(inspection.path),
            offset=start_line,
            limit=limit,
            total_lines=total_lines,
            content_fingerprint=fingerprint,
        )

    def _skill_cli_invocation_dedup_key(
        self,
        command: str,
        cwd: str,
    ) -> dict[str, Any] | None:
        """Return a stable key for logically identical workspace skill CLI calls."""
        try:
            command_tokens = shlex.split(str(command or ""))
        except ValueError:
            command_tokens = str(command or "").split()
        if not command_tokens:
            return None

        workspace_root = Path(self.working_dir or cwd).resolve()
        current_dir = Path(cwd).resolve()
        invocations: list[dict[str, Any]] = []
        for segment in self._split_shell_segments(command_tokens):
            if not segment:
                continue
            if segment[0].casefold() == "cd":
                current_dir = self._resolve_cd_segment(current_dir, segment)
                continue
            cleaned_segment = self._segment_before_redirection_or_pipe(segment)
            invocation = self._extract_skill_cli_invocation_for_segment(
                cleaned_segment,
                current_dir,
                workspace_root,
            )
            if invocation is None:
                continue
            skill_name, args, prefix_tokens = invocation
            invocations.append({
                "skill": skill_name,
                "entrypoint": list(self._normalize_skill_cli_prefix_tokens(prefix_tokens)),
                "args": list(args),
            })

        if not invocations:
            return None
        return {"invocations": invocations}

    def _parse_command_args(self, command: str) -> list[str]:
        """
        Parse command into arguments for safe execution without shell.

        Args:
            command: The command string to parse.

        Returns:
            List of command arguments.
        """
        if sys.platform == "win32":
            # Windows: simple split (shlex doesn't work well with Windows paths)
            # Use a more careful approach
            parts = []
            current = ""
            in_quotes = False
            quote_char = None

            for char in command:
                if char in ('"', "'") and not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char and in_quotes:
                    in_quotes = False
                    quote_char = None
                elif char == " " and not in_quotes:
                    if current:
                        parts.append(current)
                        current = ""
                else:
                    current += char

            if current:
                parts.append(current)

            return parts
        else:
            # Unix: use shlex for proper parsing
            try:
                return shlex.split(command)
            except ValueError:
                # Fallback to simple split if shlex fails
                return command.split()

    def _quote_arg(self, arg: str) -> str:
        """
        Quote an argument for safe shell usage.

        Args:
            arg: The argument to quote.

        Returns:
            Safely quoted argument.
        """
        if sys.platform == "win32":
            # Windows quoting
            if " " in arg or any(c in arg for c in '&|<>^'):
                return f'"{arg}"'
            return arg
        else:
            # Unix: use shlex.quote
            return shlex.quote(arg)

    async def _communicate_with_timeout(
        self,
        process: asyncio.subprocess.Process,
    ) -> tuple[bytes, bytes, int]:
        """Communicate with process, killing it on timeout to avoid zombies."""
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout,
            )
            return stdout, stderr, process.returncode or 0
        except asyncio.TimeoutError:
            # Kill the process to prevent zombie processes
            try:
                process.kill()
            except ProcessLookupError:
                pass  # Process already exited
            await process.wait()
            raise

    @staticmethod
    def _find_bash() -> str | None:
        """Find a usable bash executable on the system."""
        import shutil

        candidates = (
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
        )
        if sys.platform != "win32":
            bash = shutil.which("bash")
            if bash:
                return bash
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        bash = shutil.which("bash")
        if bash:
            return bash
        if sys.platform == "win32":
            for candidate in (
                r"C:\Windows\System32\bash.exe",
                r"C:\Users\Ricky\AppData\Local\Microsoft\WindowsApps\bash.exe",
            ):
                if os.path.isfile(candidate):
                    return candidate
        return None

    @staticmethod
    def _convert_win_paths_to_posix(command: str) -> str:
        """Convert Windows-style absolute paths in a bash command to POSIX format.

        Converts ``C:\\path\\to\\file`` to ``/c/path/to/file`` so that Git Bash
        receives a valid POSIX path instead of interpreting backslashes as escape
        characters (which silently strips them, producing broken paths like
        ``C:UsersRicky...``).

        Only backslash-separated Windows paths are converted; forward-slash paths
        (``C:/path``) and URLs (``https://...``) are left untouched.
        """
        def _replace(match: re.Match) -> str:
            drive = match.group(1).lower()
            rest = match.group(2).replace("\\", "/")
            return f"/{drive}/{rest}" if rest else f"/{drive}"

        # Match drive-letter paths with at least one backslash: C:\path\...
        # The character class [^\s"'] stops at whitespace or quotes so we don't
        # consume tokens that belong to the next shell argument.
        return re.sub(r'([A-Za-z]):\\([^\s"\']*)', _replace, command)

    @staticmethod
    def _windows_home_to_bash(home_path: str) -> str:
        """Convert a Windows home path into the POSIX form expected by Git Bash."""
        normalized = home_path.replace("\\", "/")
        match = re.match(r"^([A-Za-z]):/(.*)$", normalized)
        if not match:
            return normalized
        drive, rest = match.groups()
        return f"/{drive.lower()}/{rest}" if rest else f"/{drive.lower()}"

    @staticmethod
    def _posix_drive_path_to_windows(path: str) -> str:
        """Convert Git-Bash-style /c/foo paths to Windows paths for cwd checks."""
        normalized = path.replace("\\", "/")
        match = re.match(r"^/([A-Za-z])(?:/(.*))?$", normalized)
        if not match:
            return path
        drive, rest = match.groups()
        if not rest:
            return f"{drive.upper()}:\\"
        rest_win = rest.replace("/", "\\")
        return f"{drive.upper()}:\\{rest_win}"

    def _resolve_working_dir(self, working_dir: str | None) -> str:
        """Resolve command cwd, treating relative paths as workspace-relative."""
        base_dir = str(self.working_dir or os.getcwd())
        raw_dir = str(working_dir).strip() if working_dir is not None else ""
        cwd = raw_dir or base_dir

        if sys.platform == "win32":
            cwd = self._posix_drive_path_to_windows(cwd)

        if not os.path.isabs(cwd):
            cwd = os.path.join(base_dir, cwd)

        return os.path.abspath(os.path.expanduser(cwd))

    def _prepend_workspace_env_for_skill_command(self, command: str, cwd: str) -> str:
        """Load workspace-local env only for installed skill CLI commands."""
        if ".env.local" in command:
            return command
        try:
            workspace_root = Path(self.working_dir or cwd).expanduser().resolve()
            env_file = workspace_root / ".env.local"
        except Exception:
            return command
        if not env_file.is_file():
            return command
        exports = self._workspace_env_exports(env_file)
        if not exports:
            return command
        return f"{exports}; {command}"

    @staticmethod
    def _command_references_agent_wallet_path(command: str) -> bool:
        normalized = str(command or "").replace("\\", "/")
        return "~/.agent-wallet" in normalized or "/.agent-wallet" in normalized

    @staticmethod
    def _wallet_compat_home(wallet_root: Path) -> Path:
        base = Path(
            os.environ.get("SPOON_BOT_WALLET_COMPAT_HOME_ROOT")
            or os.environ.get("TMPDIR")
            or "/tmp"
        ).expanduser()
        digest = hashlib.sha256(str(wallet_root).encode("utf-8")).hexdigest()[:16]
        return (base / "spoon-bot-wallet-homes" / digest).resolve(strict=False)

    def _align_wallet_home_for_command(
        self,
        env: dict[str, str],
        command: str,
        cwd: str,
    ) -> None:
        """Expose the active built-in wallet through ~/.agent-wallet when needed.

        Some installed CLIs intentionally avoid private-key environment variables
        and load a wallet from ``~/.agent-wallet``.  When spoon-bot is running with
        an isolated wallet root, give those CLIs an isolated HOME containing only
        a compatibility ``.agent-wallet`` link instead of changing the real user
        home or leaking a private key into the subprocess environment.
        """
        try:
            if not (
                self._command_invokes_workspace_skill(command, cwd)
                or self._command_references_agent_wallet_path(command)
            ):
                return
        except Exception:
            return

        raw_wallet_root = ""
        try:
            workspace_wallet = (
                Path(self.working_dir or cwd).expanduser().resolve() / ".agent-wallet"
            )
            if workspace_wallet.is_dir():
                raw_wallet_root = str(workspace_wallet)
        except OSError:
            raw_wallet_root = ""
        if not raw_wallet_root:
            raw_wallet_root = (
                env.get("AGENT_WALLET_DIR")
                or os.environ.get("AGENT_WALLET_DIR")
                or env.get("SPOON_BOT_WALLET_PATH")
                or os.environ.get("SPOON_BOT_WALLET_PATH")
            )
        if not raw_wallet_root:
            return

        try:
            wallet_root = Path(raw_wallet_root).expanduser().resolve()
            if not wallet_root.exists() and (
                env.get("SPOON_BOT_WALLET_PATH")
                or os.environ.get("SPOON_BOT_WALLET_PATH")
            ):
                wallet_root.mkdir(parents=True, exist_ok=True)
            if not wallet_root.is_dir():
                return
            compat_home = self._wallet_compat_home(wallet_root)
            compat_home.mkdir(parents=True, exist_ok=True)
            try:
                compat_home.chmod(0o700)
            except OSError:
                pass
            compat_wallet = compat_home / ".agent-wallet"
            if compat_wallet.is_symlink():
                compat_wallet.unlink()
            compat_wallet.mkdir(parents=True, exist_ok=True)

            for filename in ("keystore.json", "pw.txt", "privatekey.tmp"):
                source = wallet_root / filename
                target = compat_wallet / filename
                if not source.exists():
                    continue
                if target.exists() or target.is_symlink():
                    try:
                        if target.resolve() == source.resolve():
                            continue
                        target.unlink()
                    except OSError:
                        continue
                try:
                    target.symlink_to(source)
                except OSError:
                    try:
                        shutil.copy2(source, target)
                    except OSError:
                        pass

            state_file = wallet_root / "state.env"
            compat_state_file = compat_wallet / "state.env"
            replacements = {
                "APP_DIR": str(compat_wallet),
                "STATE_FILE": str(compat_state_file),
                "KEYSTORE_FILE": str(compat_wallet / "keystore.json"),
                "PASSWORD_FILE": str(compat_wallet / "pw.txt"),
            }
            state_lines: list[str] = []
            seen_state_keys: set[str] = set()
            if state_file.is_file():
                for raw_line in state_file.read_text(
                    encoding="utf-8",
                    errors="replace",
                ).splitlines():
                    key, sep, _value = raw_line.partition("=")
                    normalized_key = key.strip()
                    if sep and normalized_key in replacements:
                        value = replacements[normalized_key].replace("'", "'\"'\"'")
                        state_lines.append(f"{normalized_key}='{value}'")
                        seen_state_keys.add(normalized_key)
                    else:
                        state_lines.append(raw_line)
            for key, raw_value in replacements.items():
                if key in seen_state_keys:
                    continue
                value = raw_value.replace("'", "'\"'\"'")
                state_lines.append(f"{key}='{value}'")
            compat_state_file.write_text(
                "\n".join(state_lines).rstrip() + "\n",
                encoding="utf-8",
            )
            env["HOME"] = str(compat_home)
            env["SPOON_BOT_WALLET_PATH"] = str(compat_wallet)
            env["AGENT_WALLET_DIR"] = str(compat_wallet)
        except OSError:
            return

    def _align_wallet_home_for_skill_command(
        self,
        env: dict[str, str],
        command: str,
        cwd: str,
    ) -> None:
        """Backward-compatible wrapper for tests and older callers."""
        self._align_wallet_home_for_command(env, command, cwd)

    @staticmethod
    def _workspace_env_exports(env_file: Path) -> str:
        """Return shell exports for non-sensitive workspace env values."""
        try:
            lines = env_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return ""

        exports: list[str] = []
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                continue
            if key in SCRUBBED_ENV_VARS or is_sensitive_env_var(key):
                continue
            value = value.strip()
            if (
                len(value) >= 2
                and value[0] == value[-1]
                and value[0] in {"'", '"'}
            ):
                value = value[1:-1]
            exports.append(f"export {key}={shlex.quote(value)}")
        return "; ".join(exports)

    def _run_sync(
        self,
        command: str,
        cwd: str,
    ) -> tuple[bytes, bytes, int]:
        """Synchronous subprocess execution (called via run_in_executor)."""
        env = _scrub_env(os.environ.copy())
        self._align_wallet_home_for_command(env, command, cwd)
        userprofile = env.get("USERPROFILE", "").strip()
        if userprofile and env.get("HOME", "").strip() in {"", "/root"}:
            env["HOME"] = userprofile.replace("\\", "/")
        if sys.platform == "win32":
            # Keep HOME aligned with the real Windows user profile so tools that
            # resolve "~" (e.g. wallet loaders) do not drift to "/root".
            home_path = str(Path.home())
            env["USERPROFILE"] = home_path
            if len(home_path) >= 2 and home_path[1] == ":":
                env["HOMEDRIVE"] = home_path[:2]
                env["HOMEPATH"] = home_path[2:] or "\\"
            bash = self._find_bash()
            if bash:
                env["HOME"] = self._windows_home_to_bash(home_path)
                cwd = cwd.replace("\\", "/")
                # Convert any Windows-style paths inside the command so that
                # Git Bash receives valid POSIX paths (C:\foo -> /c/foo).
                if self._is_git_bash(bash):
                    command = self._normalize_windows_python_command(command)
                command = self._convert_win_paths_to_posix(command)
                result = subprocess.run(
                    [bash, "-c", command],
                    capture_output=True,
                    stdin=subprocess.DEVNULL,
                    cwd=cwd,
                    env=env,
                    timeout=self.timeout,
                )
            else:
                env["HOME"] = home_path.replace("\\", "/")
                command = self._normalize_windows_python_command(command)
                result = subprocess.run(
                    command,
                    capture_output=True,
                    stdin=subprocess.DEVNULL,
                    shell=True,
                    cwd=cwd,
                    env=env,
                    timeout=self.timeout,
                )
        elif self.use_shell:
            result = subprocess.run(
                command,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                shell=True,
                executable="/bin/sh",
                cwd=cwd,
                env=env,
                timeout=self.timeout,
            )
        else:
            args = self._parse_command_args(command)
            if not args:
                raise ValueError("Empty command after parsing")
            result = subprocess.run(
                args,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                cwd=cwd,
                env=env,
                timeout=self.timeout,
            )
        return result.stdout, result.stderr, result.returncode

    def _build_output_result(
        self,
        stdout_text: str,
        stderr_text: str,
        returncode: int | None,
        *,
        max_chars: int | None = None,
        truncate: bool = True,
    ) -> str:
        already_satisfied = self._format_idempotent_conflict_result(
            stdout_text,
            stderr_text,
            returncode,
        )
        if already_satisfied is not None:
            return already_satisfied

        output_parts = []

        if stdout_text:
            output_parts.append(stdout_text)

        if stderr_text and stderr_text.strip():
            output_parts.append(f"STDERR:\n{stderr_text}")

        if returncode not in (None, 0):
            output_parts.append(f"\nExit code: {returncode}")

        result = "\n".join(output_parts) if output_parts else "(no output)"
        result = self._mask_secrets(result)

        limit = max_chars or self.max_output
        if truncate and limit and len(result) > limit:
            truncated = len(result) - limit
            marker = f"\n... (truncated, {truncated} more chars) ...\n"
            available = limit - len(marker)
            if available <= 0:
                result = result[:limit]
            else:
                head_limit = max(1, available // 2)
                tail_limit = max(1, available - head_limit)
                result = (
                    result[:head_limit].rstrip()
                    + marker
                    + result[-tail_limit:].lstrip()
                )

        return result

    @staticmethod
    def _looks_like_pnpm_corepack_node_mismatch(output: str) -> bool:
        """Detect Corepack selecting a pnpm release unsupported by current Node."""
        lower = str(output or "").casefold()
        if "pnpm" not in lower:
            return False
        return (
            "corepack is about to download" in lower
            and "requires at least node.js" in lower
        ) or "no such built-in module: node:sqlite" in lower

    @staticmethod
    def _pnpm_compat_retry_command(command: str) -> str | None:
        """Return the same command with pnpm routed through a Node-compatible major."""
        original = str(command or "").strip()
        if not original:
            return None

        # Keep the user's cwd, redirection, and install flags intact.  The
        # replacement is generic for unpinned pnpm installs; if a project has a
        # packageManager field, package-manager resolution remains the source of
        # truth and this fallback is only offered after Corepack has failed.
        retry = re.sub(
            r"(?<![\w./-])pnpm(\.cmd)?\s+install\b",
            "npx --yes pnpm@10 install",
            original,
            count=1,
            flags=re.IGNORECASE,
        )
        if retry == original:
            retry = re.sub(
                r"(?<![\w./-])pnpm(\.cmd)?\b",
                "npx --yes pnpm@10",
                original,
                count=1,
                flags=re.IGNORECASE,
            )
        return retry if retry != original else None

    def _append_pnpm_corepack_recovery(self, command: str, result: str) -> str:
        """Add a concrete next command for pnpm/Corepack Node-version failures."""
        if not self._looks_like_pnpm_corepack_node_mismatch(result):
            return result
        retry = self._pnpm_compat_retry_command(command)
        if not retry:
            return (
                f"{result}\n\n"
                "RECOVERY: Corepack selected a pnpm release that is incompatible "
                "with the current Node.js runtime. Retry the same install with a "
                "Node-compatible pnpm major, for example `npx --yes pnpm@10 install`, "
                "unless the package declares a different packageManager contract."
            )
        return (
            f"{result}\n\n"
            "RECOVERY: Corepack selected a pnpm release that is incompatible "
            "with the current Node.js runtime. This is package-manager bootstrap "
            "failure, not a skill failure. Retry without changing the skill flow:\n"
            f"{retry}"
        )

    def _format_idempotent_conflict_result(
        self,
        stdout_text: str,
        stderr_text: str,
        returncode: int | None,
    ) -> str | None:
        """Convert remote "already applied" conflicts into a recoverable state.

        Many command-line workflows intentionally use non-zero exit codes for
        HTTP 409 conflict responses. When the remote message says the requested
        operation is already satisfied, that is state evidence rather than a
        task failure.
        """
        if returncode in (None, 0):
            return None
        combined = "\n".join(part for part in (stdout_text, stderr_text) if part)
        if not combined.strip():
            return None

        lowered = combined.casefold()
        has_conflict_status = bool(
            re.search(r"\b(?:http\s*)?409\b", lowered)
            or " conflict" in lowered
        )
        has_already_state = bool(
            re.search(
                r"\b(?:already|exists|existed|duplicate|claimed|registered|active)\b",
                lowered,
            )
        )
        if not has_conflict_status or not has_already_state:
            return None

        detail = self._extract_remote_state_detail(combined)
        message = "Remote operation already satisfied; continuing from current state."
        if detail:
            message += f"\nDetail: {detail}"
        return self._mask_secrets(message)

    @staticmethod
    def _extract_remote_state_detail(output: str) -> str:
        """Extract a concise human-readable remote state detail from command output."""
        text = str(output or "")
        for pattern in (
            r'"(?:error|message|detail)"\s*:\s*"([^"]+)"',
            r"'(?:error|message|detail)'\s*:\s*'([^']+)'",
        ):
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return " ".join(match.group(1).split())[:240]

        for raw_line in text.splitlines():
            line = " ".join(raw_line.strip().split())
            if not line:
                continue
            lowered = line.casefold()
            if "409" in lowered or "conflict" in lowered:
                line = re.sub(r"^.*?\b(?:409|conflict)\b[:\s-]*", "", line, flags=re.IGNORECASE)
            if re.search(r"\b(?:already|exists|existed|duplicate|claimed|registered|active)\b", line, re.IGNORECASE):
                return line[:240]
        return ""

    async def _consume_process_stream(
        self,
        stream: Any,
        append: Any,
    ) -> None:
        if stream is None:
            return
        while True:
            chunk = await asyncio.to_thread(self._read_process_stream_chunk, stream)
            if not chunk:
                return
            append(chunk.decode("utf-8", errors="replace"))

    @staticmethod
    def _read_process_stream_chunk(stream: Any, size: int = 4096) -> bytes:
        """Read currently available process output without waiting for EOF.

        ``Popen(..., stdout=PIPE)`` returns a buffered reader. ``read(size)``
        can wait until the buffer fills or the process exits, which hides short
        progress lines from managed background-job summaries. ``read1`` performs
        at most one raw read and returns available bytes promptly.
        """
        read1 = getattr(stream, "read1", None)
        if callable(read1):
            return read1(size)
        return stream.read(size)

    async def _create_process(
        self,
        command: str,
        cwd: str,
    ) -> subprocess.Popen[bytes]:
        env = _scrub_env(os.environ.copy())
        self._align_wallet_home_for_command(env, command, cwd)
        userprofile = env.get("USERPROFILE", "").strip()
        if userprofile and env.get("HOME", "").strip() in {"", "/root"}:
            env["HOME"] = userprofile.replace("\\", "/")

        if sys.platform == "win32":
            home_path = str(Path.home())
            env["USERPROFILE"] = home_path
            if len(home_path) >= 2 and home_path[1] == ":":
                env["HOMEDRIVE"] = home_path[:2]
                env["HOMEPATH"] = home_path[2:] or "\\"
            bash = self._find_bash()
            process_group_kwargs = self._process_group_kwargs()
            if bash:
                env["HOME"] = self._windows_home_to_bash(home_path)
                if self._is_git_bash(bash):
                    command = self._normalize_windows_python_command(command)
                command = self._convert_win_paths_to_posix(command)
                cwd = cwd.replace("\\", "/")
                return subprocess.Popen(
                    [bash, "-c", command],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,
                    cwd=cwd,
                    env=env,
                    **process_group_kwargs,
                )
            env["HOME"] = home_path.replace("\\", "/")
            command = self._normalize_windows_python_command(command)
            return subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                shell=True,
                cwd=cwd,
                env=env,
                **process_group_kwargs,
            )

        if self.use_shell:
            return subprocess.Popen(
                command,
                shell=True,
                executable="/bin/sh",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                cwd=cwd,
                env=env,
                **self._process_group_kwargs(),
            )

        args = self._parse_command_args(command)
        if not args:
            raise ValueError("Empty command after parsing")
        return subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            cwd=cwd,
            env=env,
            **self._process_group_kwargs(),
        )

    @staticmethod
    def _normalize_windows_python_command(command: str) -> str:
        """Prefer the installed Windows Python launcher spelling when needed.

        Windows often exposes ``python`` while ``python3`` points at the Store
        alias. Linux/macOS shells keep their native command unchanged.
        """
        text = str(command or "")
        if not text.strip():
            return text
        try:
            tokens = shlex.split(text, posix=False)
        except ValueError:
            tokens = text.split()
        if not tokens:
            return text

        first = tokens[0].strip().strip('"\'')
        if first.casefold() not in {"python3", "python3.exe"}:
            return text
        python_path = shutil.which("python")
        python3_path = shutil.which("python3")
        python3_is_store_alias = bool(
            python3_path and "\\windowsapps\\" in python3_path.casefold()
        )
        if not python_path or (python3_path and not python3_is_store_alias):
            return text
        return re.sub(
            r"^(\s*)(['\"]?)(python3(?:\.exe)?)(\2)(?=\s|$)",
            r"\1\2python\4",
            text,
            count=1,
            flags=re.IGNORECASE,
        )

    @staticmethod
    def _is_git_bash(path: str | None) -> bool:
        """Return True for Git-for-Windows bash paths."""
        normalized = str(path or "").replace("/", "\\").casefold()
        return "\\git\\bin\\bash.exe" in normalized or "\\git\\usr\\bin\\bash.exe" in normalized

    @staticmethod
    def _process_group_kwargs() -> dict[str, Any]:
        """Create subprocesses in an isolated group so cancellation can kill children."""
        if sys.platform == "win32":
            flag = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            return {"creationflags": flag} if flag else {}
        return {"start_new_session": True}

    @staticmethod
    def _get_process_returncode(process: Any) -> int | None:
        poll = getattr(process, "poll", None)
        if callable(poll):
            try:
                return poll()
            except Exception:
                pass
        return getattr(process, "returncode", None)

    @staticmethod
    def _resolve_background_handoff_timeout(value: float | None = None) -> float:
        if value is None:
            raw = os.getenv("SPOON_BOT_SHELL_BACKGROUND_HANDOFF_TIMEOUT")
            if raw not in (None, ""):
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    value = DEFAULT_SHELL_BACKGROUND_HANDOFF_TIMEOUT
            else:
                value = DEFAULT_SHELL_BACKGROUND_HANDOFF_TIMEOUT
        try:
            return max(0.0, min(float(value), 120.0))
        except (TypeError, ValueError):
            return DEFAULT_SHELL_BACKGROUND_HANDOFF_TIMEOUT

    def _background_job_poll_wait_seconds(self, value: float | None = None) -> float:
        if value is None:
            raw = os.getenv("SPOON_BOT_BACKGROUND_JOB_POLL_WAIT_SECONDS", "").strip()
            if raw:
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    value = self._DEFAULT_BACKGROUND_JOB_POLL_WAIT_SECONDS
            else:
                value = self._DEFAULT_BACKGROUND_JOB_POLL_WAIT_SECONDS
        try:
            return max(0.0, min(float(value), 120.0))
        except (TypeError, ValueError):
            return self._DEFAULT_BACKGROUND_JOB_POLL_WAIT_SECONDS

    async def _wait_for_process(self, process: Any) -> int | None:
        wait = getattr(process, "wait")
        if asyncio.iscoroutinefunction(wait):
            return await wait()
        return await asyncio.to_thread(wait)

    async def _wait_for_background_handoff(
        self,
        job: _BackgroundShellJob,
    ) -> bool:
        """Briefly absorb near-timeout completions before exposing a background job."""
        grace_seconds = float(getattr(self, "background_handoff_timeout", 0.0) or 0.0)
        if grace_seconds <= 0:
            await self._refresh_background_job(job)
            return self._is_terminal_status(job.status)

        try:
            await asyncio.wait_for(self._wait_for_process(job.process), timeout=grace_seconds)
        except asyncio.TimeoutError:
            pass
        await self._refresh_background_job(job)
        return self._is_terminal_status(job.status)

    async def _wait_for_background_job_signal(
        self,
        job: _BackgroundShellJob,
        progress_key: str,
        *,
        wait_seconds: float,
    ) -> bool:
        """Wait briefly for a running background job to finish or emit output."""
        if wait_seconds <= 0 or self._is_terminal_status(job.status):
            return False

        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            await asyncio.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
            await self._refresh_background_job(job)
            if self._is_terminal_status(job.status):
                return True
            if self._background_job_progress_key(job) != progress_key:
                return True
        await self._refresh_background_job(job)
        return self._background_job_progress_key(job) != progress_key

    def _build_completed_job_result(
        self,
        job: _BackgroundShellJob,
    ) -> tuple[str, str]:
        full_result = self._build_output_result(
            job.stdout_text,
            job.stderr_text,
            job.returncode,
            truncate=False,
        )
        result = self._build_output_result(job.stdout_text, job.stderr_text, job.returncode)
        return result, full_result

    @staticmethod
    def _completed_job_capture_metadata(
        job: _BackgroundShellJob,
        result: str,
    ) -> dict[str, Any]:
        status = job.status
        outcome = ""
        if str(result or "").startswith("Remote operation already satisfied;"):
            status = "completed"
            outcome = "already_satisfied"
        metadata: dict[str, Any] = {
            "status": status,
            "returncode": job.returncode,
        }
        if outcome:
            metadata["tool_outcome"] = outcome
        return metadata

    def _maybe_explain_workspace_skill_no_output(
        self,
        command: str,
        cwd: str,
        result: str,
        returncode: int | None,
    ) -> str:
        """Make empty skill CLI output a recoverable diagnostic, not success."""
        if returncode not in (None, 0):
            return result
        if str(result or "").strip() != "(no output)":
            return result
        try:
            invokes_workspace_skill = self._command_invokes_workspace_skill(command, cwd)
        except Exception:
            invokes_workspace_skill = False
        if not invokes_workspace_skill:
            return result
        return (
            "Skill CLI completed with no stdout/stderr. Do not treat this as "
            "success. Re-run the exact documented command in the foreground "
            "without pipes, output filters, file redirection, shell timeout "
            "wrappers, stderr/stdout merge, or other shell wrappers. If the "
            "exact command still returns no output, follow that skill's "
            "no-output/recovery procedure "
            "from SKILL.md instead of repeating the same command."
        )

    async def _terminate_process_tree(
        self,
        process: Any,
        *,
        grace_seconds: float = 5.0,
    ) -> None:
        """Terminate a subprocess and its children when a shell run is cancelled."""
        if self._get_process_returncode(process) is not None:
            return

        pid = getattr(process, "pid", None)
        if sys.platform == "win32" and pid:
            try:
                await asyncio.to_thread(
                    subprocess.run,
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=grace_seconds,
                )
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
        else:
            if pid and hasattr(os, "killpg"):
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                except Exception:
                    try:
                        process.terminate()
                    except Exception:
                        pass
            else:
                try:
                    process.terminate()
                except Exception:
                    pass

            try:
                await asyncio.wait_for(self._wait_for_process(process), timeout=grace_seconds)
                return
            except asyncio.TimeoutError:
                if pid and hasattr(os, "killpg"):
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                    except Exception:
                        try:
                            process.kill()
                        except Exception:
                            pass
                else:
                    try:
                        process.kill()
                    except Exception:
                        pass

        try:
            await asyncio.wait_for(self._wait_for_process(process), timeout=grace_seconds)
        except Exception:
            pass

    async def _terminate_background_job(
        self,
        job: _BackgroundShellJob,
        *,
        status: str,
    ) -> None:
        await self._terminate_process_tree(job.process)
        for stream_task in (job.stdout_task, job.stderr_task):
            if not stream_task.done():
                stream_task.cancel()
        await asyncio.gather(job.stdout_task, job.stderr_task, return_exceptions=True)
        await self._refresh_background_job(job)
        job.status = status
        job.finished_at = time.time()

    async def _start_background_job(
        self,
        command: str,
        cwd: str,
        owner_key: str | None = None,
    ) -> _BackgroundShellJob:
        owner = owner_key or get_tool_owner()
        process = await self._create_process(command, cwd)
        job = _BackgroundShellJob(
            job_id=f"sh_{uuid4().hex[:10]}",
            owner_key=owner,
            command=command,
            cwd=cwd,
            process=process,
            stdout_task=asyncio.create_task(asyncio.sleep(0)),
            stderr_task=asyncio.create_task(asyncio.sleep(0)),
            buffer_limit=max(self.max_output * 20, 200_000),
        )
        job.stdout_task = asyncio.create_task(
            self._consume_process_stream(process.stdout, job.append_stdout)
        )
        job.stderr_task = asyncio.create_task(
            self._consume_process_stream(process.stderr, job.append_stderr)
        )
        _SHELL_BACKGROUND_JOBS[job.job_id] = job
        self._prune_background_jobs(owner_key=owner)
        return job

    async def _refresh_background_job(self, job: _BackgroundShellJob) -> _BackgroundShellJob:
        returncode = self._get_process_returncode(job.process)
        if job.status == "running" and returncode is not None:
            job.returncode = returncode
            job.status = "completed" if job.returncode == 0 else "failed"
            job.finished_at = time.time()
        if returncode is not None:
            stream_tasks = [job.stdout_task, job.stderr_task]
            try:
                await asyncio.wait_for(
                    asyncio.gather(*stream_tasks, return_exceptions=True),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                # A command like `node server.js &` can let the shell exit while
                # a child process keeps stdout/stderr pipes open. The shell
                # command has already reached a terminal status, so do not let
                # orphaned pipe readers block the tool result forever.
                for task in stream_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*stream_tasks, return_exceptions=True)
        return job

    @staticmethod
    def _is_terminal_status(status: str) -> bool:
        return status in {"completed", "failed", "terminated", "cancelled"}

    def _prune_background_jobs(
        self,
        *,
        owner_key: str | None = None,
        keep_completed_per_owner: int = 20,
    ) -> None:
        completed_jobs: dict[str, list[_BackgroundShellJob]] = {}
        for job in _SHELL_BACKGROUND_JOBS.values():
            if not self._is_terminal_status(job.status):
                continue
            if owner_key is not None and job.owner_key != owner_key:
                continue
            completed_jobs.setdefault(job.owner_key, []).append(job)

        for owner, jobs in completed_jobs.items():
            jobs.sort(key=lambda j: j.finished_at or j.created_at, reverse=True)
            for stale_job in jobs[keep_completed_per_owner:]:
                _SHELL_BACKGROUND_JOBS.pop(stale_job.job_id, None)

    def _format_background_job_summary(
        self,
        job: _BackgroundShellJob,
        *,
        timeout_seconds: int,
        tail_chars: int = 4000,
    ) -> str:
        recent_output = self._build_output_result(
            job.stdout_text,
            job.stderr_text,
            job.returncode,
            max_chars=tail_chars,
        )
        elapsed = time.time() - job.created_at
        return (
            f"Foreground timeout ({timeout_seconds}s) exceeded — command moved to background.\n"
            f"job_id: {job.job_id}\n"
            f"command: {job.command}\n"
            f"cwd: {job.cwd}\n"
            f"status: running (elapsed {elapsed:.0f}s)\n"
            "Recent output tail:\n"
            f"{recent_output}\n\n"
            "NEXT STEPS — monitor this job:\n"
            f"  1. Check progress: action='job_status', job_id='{job.job_id}'\n"
            f"  2. Read full output: action='job_output', job_id='{job.job_id}'\n"
            "The command is still active. Do not rerun the same command while "
            "this job exists. Quiet output can be normal for network, build, "
            "or waiting operations. If a first status check is still running, "
            "inspect job_output or the command's log/output artifact next; do "
            "not repeatedly call job_status without new evidence. Use "
            "action='terminate_job' only with force=true when the latest user "
            "request cancels the job or separate evidence proves it is "
            "unrecoverably stuck."
        )

    def _workspace_skill_foreground_timeout(self) -> int:
        """Return foreground budget before a workspace skill CLI becomes managed background."""
        raw = os.getenv("SPOON_BOT_WORKSPACE_SKILL_FOREGROUND_TIMEOUT", "").strip()
        if raw:
            try:
                parsed = int(raw)
            except ValueError:
                parsed = int(self.max_timeout)
        else:
            parsed = self._DEFAULT_WORKSPACE_SKILL_FOREGROUND_TIMEOUT
        return max(1, min(int(self.max_timeout), parsed))

    def _command_invokes_stateful_workspace_skill(self, command: str, cwd: str) -> bool:
        """Return True for installed-skill CLI calls that can advance remote/workflow state."""
        skill_cli_key = self._skill_cli_invocation_dedup_key(command, cwd)
        if not isinstance(skill_cli_key, dict):
            return False
        invocations = skill_cli_key.get("invocations")
        if not isinstance(invocations, list) or not invocations:
            return False
        for invocation in invocations:
            if not isinstance(invocation, dict):
                continue
            args = [
                str(arg)
                for arg in invocation.get("args", [])
                if str(arg).strip()
            ]
            if args and not self._skill_args_are_read_only(args):
                return True
        return False

    def _workspace_skill_stateful_invocations(
        self,
        command: str,
        cwd: str,
    ) -> list[dict[str, Any]]:
        """Return state-changing workspace skill CLI invocations."""
        skill_cli_key = self._skill_cli_invocation_dedup_key(command, cwd)
        if not isinstance(skill_cli_key, dict):
            return []
        invocations = skill_cli_key.get("invocations")
        if not isinstance(invocations, list):
            return []

        stateful: list[dict[str, Any]] = []
        for invocation in invocations:
            if not isinstance(invocation, dict):
                continue
            args = [
                str(arg)
                for arg in invocation.get("args", [])
                if str(arg).strip()
            ]
            if args and not self._skill_args_are_read_only(args):
                stateful.append(invocation)
        return stateful

    @staticmethod
    def _normalized_shell_hint(command: str) -> str:
        try:
            return " ".join(shlex.split(str(command or ""))).casefold()
        except ValueError:
            return " ".join(str(command or "").split()).casefold()

    def _command_matches_exact_shell_hint(
        self,
        command: str,
        hints: dict[str, Any],
    ) -> bool:
        exact_commands = hints.get("exact_shell_commands")
        if not isinstance(exact_commands, list) or not exact_commands:
            return False
        normalized_command = self._normalized_shell_hint(command)
        return any(
            normalized_command == self._normalized_shell_hint(str(item or ""))
            for item in exact_commands
        )

    def _format_background_job_output(
        self,
        job: _BackgroundShellJob,
        body: str,
    ) -> str:
        return (
            f"job_id: {job.job_id}\n"
            f"status: {job.status}\n"
            f"returncode: {job.returncode if job.returncode is not None else 'running'}\n"
            "Output:\n"
            f"{body}"
        )

    async def _handle_background_action(
        self,
        action: str,
        job_id: str | None,
        tail_chars: int,
        *,
        force: bool = False,
        wait_seconds: float | None = None,
    ) -> str:
        owner_key = get_tool_owner()
        self._prune_background_jobs(owner_key=owner_key)

        if action == "list_jobs":
            owner_jobs = [
                job for job in _SHELL_BACKGROUND_JOBS.values()
                if job.owner_key == owner_key
            ]
            if not owner_jobs:
                return "No background shell jobs"
            lines = ["Background shell jobs:"]
            for job in owner_jobs:
                await self._refresh_background_job(job)
                lines.append(
                    f"[{job.job_id}] {job.status} cwd={job.cwd} command={self.validator.sanitize_for_display(job.command)}"
                )
            return "\n".join(lines)

        if not job_id:
            return "Error: 'job_id' is required for this action"

        job = _SHELL_BACKGROUND_JOBS.get(job_id)
        if not job or job.owner_key != owner_key:
            return (
                "BACKGROUND_JOB_UNAVAILABLE: managed shell job is not available "
                "in this runtime process. Background job ids are volatile and "
                "may disappear after a worker restart or session handoff. Do not "
                "retry this job id. Verify the current live state with the "
                "appropriate CLI, status command, log file, or workspace artifact "
                "before deciding whether the workflow is complete, blocked, or "
                "needs a safe continuation."
            )

        await self._refresh_background_job(job)
        poll_wait_seconds = self._background_job_poll_wait_seconds(wait_seconds)

        if action == "job_status":
            progress_key = self._background_job_progress_key(job)
            if job.status == "running":
                await self._wait_for_background_job_signal(
                    job,
                    progress_key,
                    wait_seconds=poll_wait_seconds,
                )
            else:
                repeated_poll = suppress_repeated_background_job_poll(
                    job.job_id,
                    progress_key,
                )
                if repeated_poll is not None:
                    return repeated_poll
            no_progress_suffix = ""
            if job.status == "running" and self._background_job_progress_key(job) == progress_key:
                no_progress_suffix = (
                    f"\nStill running after waiting {poll_wait_seconds:.0f}s "
                    "with no new output. This is not a completion signal. If "
                    "the user's task depends on this job finishing, continue "
                    "monitoring job_status/job_output instead of summarizing "
                    "the task as done."
                )
            return (
                f"job_id: {job.job_id}\n"
                f"status: {job.status}\n"
                f"cwd: {job.cwd}\n"
                f"command: {job.command}\n"
                f"returncode: {job.returncode if job.returncode is not None else 'running'}\n"
                "Recent output tail:\n"
                f"{self._build_output_result(job.stdout_text, job.stderr_text, job.returncode, max_chars=tail_chars)}\n"
                "Guidance: if status is running, inspect job_output or the "
                "command's log/output artifact next; do not repeat job_status "
                "without new evidence. Do not rerun the same command. Terminate "
                "only when the caller explicitly wants to abandon it or you have "
                "evidence that it is unrecoverably stuck."
                f"{no_progress_suffix}"
            )

        if action == "job_output":
            progress_key = self._background_job_progress_key(job)
            if job.status == "running":
                await self._wait_for_background_job_signal(
                    job,
                    progress_key,
                    wait_seconds=poll_wait_seconds,
                )
            else:
                repeated_poll = suppress_repeated_background_job_poll(
                    job.job_id,
                    progress_key,
                )
                if repeated_poll is not None:
                    return repeated_poll
            no_progress_suffix = ""
            if job.status == "running" and self._background_job_progress_key(job) == progress_key:
                no_progress_suffix = (
                    f"\n\nStill running after waiting {poll_wait_seconds:.0f}s "
                    "with no new output. This is not a completion signal. If "
                    "the user's task depends on this job finishing, continue "
                    "monitoring job_status/job_output instead of summarizing "
                    "the task as done."
                )
            full_result = self._build_output_result(
                job.stdout_text,
                job.stderr_text,
                job.returncode,
                max_chars=tail_chars,
                truncate=False,
            )
            result = self._build_output_result(
                job.stdout_text,
                job.stderr_text,
                job.returncode,
                max_chars=tail_chars,
            )
            full_result = self._format_background_job_output(job, full_result)
            result = self._format_background_job_output(job, result)
            full_result = f"{full_result}{no_progress_suffix}"
            result = f"{result}{no_progress_suffix}"
            capture_tool_output(
                result,
                full_result,
                metadata=self._completed_job_capture_metadata(job, result),
            )
            return result

        if action == "terminate_job":
            if self._get_process_returncode(job.process) is None:
                elapsed = time.time() - job.created_at
                if not force:
                    return (
                        f"Not terminated: background shell job {job.job_id} is "
                        f"still running after {elapsed:.0f}s. A running "
                        "network/build/chain command is not abandoned unless "
                        "the latest user request explicitly cancels it or "
                        "separate evidence proves it is unrecoverably stuck. "
                        "Keep polling job_status/job_output, or retry "
                        "terminate_job with force=true only for an explicit "
                        "cancel/abandon request or proven unrecoverable stuck "
                        "state."
                    )
                await self._terminate_background_job(job, status="terminated")
                prefix = f"Terminated background shell job {job.job_id}.\n"
            else:
                await self._refresh_background_job(job)
                prefix = f"Background shell job {job.job_id} already {job.status}.\n"
            full_body = self._build_output_result(
                job.stdout_text,
                job.stderr_text,
                job.returncode,
                max_chars=tail_chars,
                truncate=False,
            )
            summary_body = self._build_output_result(
                job.stdout_text,
                job.stderr_text,
                job.returncode,
                max_chars=tail_chars,
            )
            result = prefix + summary_body
            full_result = f"{prefix}{full_body}"
            capture_tool_output(
                result,
                full_result,
                metadata=self._completed_job_capture_metadata(job, result),
            )
            return result

        return f"Error: Unknown action '{action}'"

    @staticmethod
    def _background_job_progress_key(job: _BackgroundShellJob) -> str:
        """Build a compact progress signature for request-local poll guards."""
        stdout_len = len(job.stdout_text or "")
        stderr_len = len(job.stderr_text or "")
        return (
            f"status={job.status}|returncode={job.returncode}|"
            f"stdout={stdout_len}|stderr={stderr_len}"
        )

    async def execute(
        self,
        command: str | None = None,
        working_dir: str | None = None,
        timeout: int | None = None,
        action: str = "execute",
        job_id: str | None = None,
        tail_chars: int = 4000,
        wait_seconds: float | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Execute a shell command with security validation and rate limiting.

        Args:
            command: The command to execute.
            working_dir: Optional working directory.
            timeout: Per-command foreground timeout override in seconds.

        Returns:
            Command output or error message.
        """
        fact_check_blocker = current_session_fact_check_blocker()
        if fact_check_blocker is not None:
            capture_tool_output(fact_check_blocker, fact_check_blocker)
            return fact_check_blocker
        tool_boundary_blocker = explicit_unavailable_tool_request_blocker(self.name)
        if tool_boundary_blocker is not None:
            capture_tool_output(tool_boundary_blocker, tool_boundary_blocker)
            return tool_boundary_blocker

        if action != "execute":
            return await self._handle_background_action(
                action,
                job_id,
                tail_chars,
                force=bool(kwargs.get("force") or kwargs.get("force_terminate")),
                wait_seconds=wait_seconds,
            )

        if not command or not str(command).strip():
            return "Error: 'command' is required for action='execute'"

        enforce_guardrails = not bool(getattr(self, "yolo_mode", False))

        if enforce_guardrails:
            manual_tunnel_rejection = self._reject_manual_tunnel_when_tool_available(str(command))
            if manual_tunnel_rejection is not None:
                capture_tool_output(manual_tunnel_rejection, manual_tunnel_rejection)
                return manual_tunnel_rejection

            port_cleanup_rejection = self._reject_port_listener_cleanup_when_tool_available(str(command))
            if port_cleanup_rejection is not None:
                capture_tool_output(port_cleanup_rejection, port_cleanup_rejection)
                return port_cleanup_rejection

            if self._has_unmanaged_background_operator(str(command)):
                return (
                    "Rejected: unmanaged shell background operator '&' is not supported. "
                    "Run long-lived commands in the foreground with an appropriate timeout "
                    "so the shell tool can keep them as managed background jobs, or "
                    "activate and use a matching dynamic service/tool capability when one "
                    "is listed in the inactive catalog."
                )

        # Apply rate limiting
        if self._rate_limit_config.enabled:
            wait_time = await self._rate_limiter.wait_and_acquire()
            if wait_time > 0.1:
                logger.info(f"Shell command rate limited, waited {wait_time:.2f}s")

        cwd = self._resolve_working_dir(working_dir)

        # Verify working directory exists and is accessible
        if not os.path.isdir(cwd):
            return f"Error: Working directory not found: {cwd}"

        # Resolve effective timeout: per-command override capped by max_timeout.
        # Stateful workspace skill CLIs can be quiet while waiting on remote
        # workflows. Keep their foreground budget short so a silent command is
        # handed to a managed background job instead of leaving the UI pinned on
        # a single tool_call. Read-only skill inspections and dependency setup
        # keep the ordinary budget.
        effective_timeout = self.timeout
        if timeout is not None:
            effective_timeout = max(1, min(int(timeout), self.max_timeout))

        skill_clone_rejection = self._reject_workspace_skill_clone(command, cwd)
        if skill_clone_rejection is not None:
            capture_tool_output(skill_clone_rejection, skill_clone_rejection)
            return skill_clone_rejection

        skill_wrapper_rejection = self._reject_wrapped_workspace_skill_command(
            command,
            cwd,
        )
        if skill_wrapper_rejection is not None:
            capture_tool_output(skill_wrapper_rejection, skill_wrapper_rejection)
            return skill_wrapper_rejection

        invokes_workspace_skill = self._command_invokes_workspace_skill(command, cwd)
        invokes_workspace_skill_setup = self._command_invokes_workspace_skill_setup(
            command,
            cwd,
        )
        if (
            invokes_workspace_skill
            and not invokes_workspace_skill_setup
            and self._command_invokes_stateful_workspace_skill(command, cwd)
        ):
            effective_timeout = min(
                effective_timeout,
                self._workspace_skill_foreground_timeout(),
            )

        command = self._augment_skill_cli_labeled_values(command, cwd)

        skill_command_rejection = self._reject_undocumented_skill_cli_arguments(
            command,
            cwd,
        )
        if skill_command_rejection is not None:
            if self._request_explicit_labeled_values():
                logger.info(
                    "Rejected skill CLI invocation that differs from extracted "
                    f"SKILL.md command forms: {skill_command_rejection}"
                )
                capture_tool_output(skill_command_rejection, skill_command_rejection)
                return skill_command_rejection
            logger.info(
                "Skill CLI invocation differs from extracted SKILL.md command "
                f"forms; allowing script execution because no structured "
                f"user-labeled value is being forced into the CLI. Diagnostic: "
                f"{skill_command_rejection}"
            )

        # Validate command
        if enforce_guardrails:
            is_valid, error_msg = self.validator.validate(command)
            if not is_valid:
                safe_cmd = self.validator.sanitize_for_display(command)
                return f"Security Error: {error_msg}\nCommand: {safe_cmd}"

        redundant_shell_read = self._suppress_redundant_shell_file_read(command, cwd)
        if redundant_shell_read is not None:
            capture_tool_output(redundant_shell_read, redundant_shell_read)
            return redundant_shell_read

        execution_command = (
            self._prepend_workspace_env_for_skill_command(command, cwd)
            if invokes_workspace_skill
            else command
        )

        try:
            owner_key = get_tool_owner()
            job = await self._start_background_job(
                execution_command,
                cwd,
                owner_key=owner_key,
            )
            try:
                await asyncio.wait_for(self._wait_for_process(job.process), timeout=effective_timeout)
                await self._refresh_background_job(job)
                result, full_result = self._build_completed_job_result(job)
                result = self._maybe_explain_workspace_skill_no_output(
                    command,
                    cwd,
                    result,
                    job.returncode,
                )
                full_result = self._maybe_explain_workspace_skill_no_output(
                    command,
                    cwd,
                    full_result,
                    job.returncode,
                )
                result = self._append_pnpm_corepack_recovery(command, result)
                full_result = self._append_pnpm_corepack_recovery(command, full_result)
                result = self._append_workspace_skill_dependency_recovery(
                    command,
                    cwd,
                    result,
                )
                full_result = self._append_workspace_skill_dependency_recovery(
                    command,
                    cwd,
                    full_result,
                )
                result = self._maybe_stop_after_exact_command_failure(command, result)
                full_result = self._maybe_stop_after_exact_command_failure(command, full_result)
                capture_tool_output(
                    result,
                    full_result,
                    metadata=self._completed_job_capture_metadata(job, result),
                )
                _SHELL_BACKGROUND_JOBS.pop(job.job_id, None)
                self._prune_background_jobs(owner_key=owner_key)
                return result
            except asyncio.TimeoutError:
                if await self._wait_for_background_handoff(job):
                    result, full_result = self._build_completed_job_result(job)
                    result = self._maybe_explain_workspace_skill_no_output(
                        command,
                        cwd,
                        result,
                        job.returncode,
                    )
                    full_result = self._maybe_explain_workspace_skill_no_output(
                        command,
                        cwd,
                        full_result,
                        job.returncode,
                    )
                    result = self._append_pnpm_corepack_recovery(command, result)
                    full_result = self._append_pnpm_corepack_recovery(command, full_result)
                    result = self._append_workspace_skill_dependency_recovery(
                        command,
                        cwd,
                        result,
                    )
                    full_result = self._append_workspace_skill_dependency_recovery(
                        command,
                        cwd,
                        full_result,
                    )
                    result = self._maybe_stop_after_exact_command_failure(command, result)
                    full_result = self._maybe_stop_after_exact_command_failure(command, full_result)
                    capture_tool_output(
                        result,
                        full_result,
                        metadata=self._completed_job_capture_metadata(job, result),
                    )
                    _SHELL_BACKGROUND_JOBS.pop(job.job_id, None)
                    self._prune_background_jobs(owner_key=owner_key)
                    return result
                self._prune_background_jobs(owner_key=owner_key)
                return self._format_background_job_summary(
                    job,
                    timeout_seconds=effective_timeout,
                    tail_chars=tail_chars,
                )
            except asyncio.CancelledError:
                await self._terminate_background_job(job, status="cancelled")
                _SHELL_BACKGROUND_JOBS.pop(job.job_id, None)
                self._prune_background_jobs(owner_key=owner_key)
                raise
        except FileNotFoundError as e:
            return f"Error: Command or file not found: {e}"
        except PermissionError:
            return "Error: Permission denied for command or directory"
        except ValueError as e:
            return f"Error: Invalid command format: {e}"
        except Exception as e:
            logger.error(f"Shell execute error ({type(e).__name__}): {e!r}")
            return f"Error executing command: {type(e).__name__}: {e}"

    @staticmethod
    def _normalize_exact_command(command: str | None) -> str:
        return " ".join(str(command or "").strip().split())

    @staticmethod
    def _split_shell_segments(tokens: list[str]) -> list[list[str]]:
        """Split a shell token list into simple command segments."""
        segments: list[list[str]] = []
        current: list[str] = []
        separators = {"&&", "||", ";"}
        for token in tokens:
            if token in separators:
                if current:
                    segments.append(current)
                    current = []
                continue
            current.append(token)
        if current:
            segments.append(current)
        return segments

    @staticmethod
    def _has_unmanaged_background_operator(command: str) -> bool:
        try:
            tokens = shlex.split(str(command or ""))
        except ValueError:
            tokens = str(command or "").split()
        return any(token == "&" for token in tokens)

    @staticmethod
    def _clone_positionals(tokens: list[str]) -> list[str]:
        """Return git clone positional arguments from a git-clone segment."""
        if len(tokens) < 2 or [token.casefold() for token in tokens[:2]] != ["git", "clone"]:
            return []

        positionals: list[str] = []
        options_with_values = {
            "-b",
            "--branch",
            "--depth",
            "--filter",
            "--origin",
            "-o",
            "--config",
            "-c",
            "--reference",
            "--separate-git-dir",
            "--upload-pack",
            "--template",
            "--jobs",
            "--server-option",
        }
        skip_next = False
        for token in tokens[2:]:
            if skip_next:
                skip_next = False
                continue
            lowered = token.casefold()
            if lowered in options_with_values:
                skip_next = True
                continue
            if any(lowered.startswith(option + "=") for option in options_with_values):
                continue
            if token.startswith("-"):
                continue
            positionals.append(token)
        return positionals

    @staticmethod
    def _resolve_cd_segment(current_dir: Path, tokens: list[str]) -> Path:
        """Resolve a simple cd segment for clone-target analysis."""
        if not tokens or tokens[0].casefold() != "cd" or len(tokens) < 2:
            return current_dir
        target = Path(tokens[1]).expanduser()
        if not target.is_absolute():
            target = current_dir / target
        try:
            return target.resolve()
        except OSError:
            return target.absolute()

    def _reject_workspace_skill_clone(self, command: str, cwd: str) -> str | None:
        """Reject manual git clone installs into workspace/skills."""
        try:
            command_tokens = shlex.split(str(command or ""))
        except ValueError:
            command_tokens = str(command or "").split()
        if not command_tokens:
            return None

        workspace_root = Path(self.working_dir or cwd).resolve()
        skills_root = (workspace_root / "skills").resolve()
        current_dir = Path(cwd).resolve()

        for segment in self._split_shell_segments(command_tokens):
            if not segment:
                continue
            if segment[0].casefold() == "cd":
                current_dir = self._resolve_cd_segment(current_dir, segment)
                continue
            positionals = self._clone_positionals(segment)
            if not positionals:
                continue

            source = positionals[0]
            if len(positionals) >= 2:
                destination = positionals[1]
            else:
                source_tail = source.rstrip("/\\").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                destination = source_tail.removesuffix(".git") or "repo"

            destination_path = Path(destination)
            if not destination_path.is_absolute():
                destination_path = current_dir / destination_path
            destination_path = destination_path.resolve()

            try:
                destination_path.relative_to(skills_root)
            except ValueError:
                continue

            return (
                "Rejected: workspace skills must be installed through the skill "
                "management toolchain, not by cloning directly into workspace/skills. "
                "For a GitHub skill source, call "
                "skill_marketplace(action='install_skill', url='<source_url>'); "
                "the installer validates SKILL.md and installs the correct skill "
                "directory. For ordinary repository work, clone outside workspace/skills."
            )

        return None

    def _reject_wrapped_workspace_skill_command(
        self,
        command: str,
        cwd: str,
    ) -> str | None:
        """Reject wrappers that hide or prematurely cut off workspace skill CLIs."""
        try:
            command_tokens = shlex.split(str(command or ""))
        except ValueError:
            command_tokens = str(command or "").split()
        if not command_tokens:
            return None

        workspace_root = Path(self.working_dir or cwd).resolve()
        current_dir = Path(cwd).resolve()

        for segment in self._split_shell_segments(command_tokens):
            if not segment:
                continue
            if segment[0].casefold() == "cd":
                current_dir = self._resolve_cd_segment(current_dir, segment)
                continue

            timeout_wrapped = self._timeout_wrapped_segment(segment)
            if timeout_wrapped:
                cleaned_timeout = self._segment_before_redirection_or_pipe(timeout_wrapped)
                invocation = self._extract_skill_cli_invocation_for_segment(
                    cleaned_timeout,
                    current_dir,
                    workspace_root,
                )
                if invocation is not None:
                    skill_name, _args, _prefix_tokens = invocation
                    return self._format_workspace_skill_wrapper_rejection(
                        skill_name,
                        "a shell timeout wrapper",
                        cleaned_timeout,
                    )
                setup_skill_name = self._workspace_skill_setup_command_name(
                    cleaned_timeout,
                    current_dir,
                    workspace_root,
                )
                if setup_skill_name:
                    return self._format_workspace_skill_setup_wrapper_rejection(
                        setup_skill_name,
                        "a shell timeout wrapper",
                        cleaned_timeout,
                        current_dir,
                        workspace_root,
                    )

            has_pipe = any(str(token).strip() in {"|", "|&"} for token in segment)
            pipeline_part: list[str] = []
            for token in [*segment, "|"]:
                if str(token).strip() in {"|", "|&"}:
                    cleaned_part = self._segment_before_redirection_or_pipe(pipeline_part)
                    invocation = self._extract_skill_cli_invocation_for_segment(
                        cleaned_part,
                        current_dir,
                        workspace_root,
                    )
                    if invocation is not None:
                        skill_name, _args, _prefix_tokens = invocation
                        part_has_redirection = self._segment_has_disallowed_output_control(
                            pipeline_part
                        )
                        if has_pipe or part_has_redirection:
                            reason = (
                                "a shell pipe/output filter"
                                if has_pipe
                                else "shell output redirection"
                            )
                            return self._format_workspace_skill_wrapper_rejection(
                                skill_name,
                                reason,
                                cleaned_part,
                            )
                    setup_skill_name = self._workspace_skill_setup_command_name(
                        cleaned_part,
                        current_dir,
                        workspace_root,
                    )
                    if setup_skill_name:
                        part_has_redirection = self._segment_has_disallowed_output_control(
                            pipeline_part
                        )
                        if has_pipe or part_has_redirection:
                            reason = (
                                "a shell pipe/output filter"
                                if has_pipe
                                else "shell output redirection"
                            )
                            return self._format_workspace_skill_setup_wrapper_rejection(
                                setup_skill_name,
                                reason,
                                cleaned_part,
                                current_dir,
                                workspace_root,
                            )
                    pipeline_part = []
                    continue
                pipeline_part.append(token)

            cleaned_segment = self._segment_before_redirection_or_pipe(segment)
            invocation = self._extract_skill_cli_invocation_for_segment(
                cleaned_segment,
                current_dir,
                workspace_root,
            )
            if invocation is None:
                setup_skill_name = self._workspace_skill_setup_command_name(
                    cleaned_segment,
                    current_dir,
                    workspace_root,
                )
                if setup_skill_name:
                    if self._segment_has_disallowed_output_control(segment):
                        return self._format_workspace_skill_setup_wrapper_rejection(
                            setup_skill_name,
                            "shell output redirection",
                            cleaned_segment,
                            current_dir,
                            workspace_root,
                )
                continue
            skill_name, _args, _prefix_tokens = invocation
            if self._segment_has_disallowed_output_control(segment):
                return self._format_workspace_skill_wrapper_rejection(
                    skill_name,
                    "shell output redirection",
                    cleaned_segment,
                )

        return None

    @staticmethod
    def _uses_cloudflared_command(command: str) -> bool:
        try:
            tokens = shlex.split(str(command or ""))
        except ValueError:
            tokens = str(command or "").split()
        for segment in ShellTool._split_shell_segments(tokens):
            if not segment:
                continue
            command_name = segment[0].casefold().replace("\\", "/")
            if command_name == "cloudflared" or command_name.endswith("/cloudflared"):
                return True
            if command_name == "timeout" and len(segment) >= 3:
                wrapped = segment[2].casefold().replace("\\", "/")
                if wrapped == "cloudflared" or wrapped.endswith("/cloudflared"):
                    return True
        return False

    def _reject_manual_tunnel_when_tool_available(self, command: str) -> str | None:
        """Keep public-link workflows on the structured service_expose tool."""
        if not self._uses_cloudflared_command(command):
            return None
        return (
            "Rejected: cloudflared is owned by the service exposure toolchain. "
            "Use service_expose to start, expose, inspect, or read tunnel logs; "
            "if it is not in the callable tool list, activate the matching "
            "service exposure tool first. This avoids unmanaged tunnel jobs and "
            "keeps local/public URL verification in one structured path."
        )

    @staticmethod
    def _token_mentions_port_listener(token: str) -> bool:
        normalized = str(token or "").strip().strip("'\"").casefold()
        if not normalized:
            return False
        if normalized.endswith(("/tcp", "/udp")):
            port = normalized.rsplit("/", 1)[0]
            return port.isdigit() and 1 <= int(port) <= 65535
        if normalized.startswith(":"):
            port = normalized[1:]
            return port.isdigit() and 1 <= int(port) <= 65535
        for prefix in ("tcp:", "udp:"):
            if normalized.startswith(prefix):
                port = normalized[len(prefix) :]
                return port.isdigit() and 1 <= int(port) <= 65535
        return False

    @staticmethod
    def _uses_port_listener_kill_command(command: str) -> bool:
        try:
            tokens = shlex.split(str(command or ""))
        except ValueError:
            tokens = str(command or "").split()
        if not tokens:
            return False

        for segment in ShellTool._split_shell_segments(tokens):
            if not segment:
                continue
            command_name = Path(segment[0]).name.casefold()
            has_port_target = any(ShellTool._token_mentions_port_listener(token) for token in segment[1:])
            if command_name == "fuser" and has_port_target:
                for token in segment[1:]:
                    if token == "-k" or (token.startswith("-") and "k" in token[1:]):
                        return True

        lowered = [token.casefold() for token in tokens]
        has_pipeline_kill = "|" in tokens and "xargs" in lowered and any(
            Path(token).name.casefold() in {"kill", "pkill", "killall"}
            for token in lowered
        )
        has_port_probe = any(ShellTool._token_mentions_port_listener(token) for token in tokens)
        return bool(has_pipeline_kill and has_port_probe)

    def _reject_port_listener_cleanup_when_tool_available(self, command: str) -> str | None:
        """Avoid killing unrelated preview/front-end services to free a port."""
        if not self._uses_port_listener_kill_command(command):
            return None
        return (
            "Rejected: shell port-listener cleanup is unsafe for preview service "
            "management. Do not kill processes by port to start a generated app; "
            "use service_expose with replace=true for the same managed service, "
            "or pass port=0 and a command that uses $PORT or {port} so the "
            "service starts on a free port without disrupting other apps."
        )

    @staticmethod
    def _strip_template_token(token: str) -> str:
        return str(token or "").strip().strip("[]")

    @classmethod
    def _template_token_is_placeholder(cls, token: str) -> bool:
        raw = str(token or "").strip()
        stripped = cls._strip_template_token(raw)
        return (
            not stripped
            or (raw.startswith("[") and raw.endswith("]") and not stripped.startswith("-"))
            or stripped.startswith("<")
            or stripped.endswith(">")
            or "<" in stripped
            or ">" in stripped
        )

    @classmethod
    def _template_placeholder_label(cls, token: str) -> str:
        stripped = cls._strip_template_token(token)
        stripped = stripped.strip("<>").strip()
        normalized = re.sub(r"[^A-Za-z0-9_-]+", "", stripped)
        return normalized.casefold()

    @staticmethod
    def _skill_cli_assignment(content: str) -> str:
        for line in str(content or "").splitlines():
            stripped = line.strip()
            if not stripped[:3].casefold() == "cli":
                continue
            rest = stripped[3:].lstrip()
            if rest.startswith(":"):
                rest = rest[1:].lstrip()
            if not rest.startswith("="):
                continue
            return rest[1:].strip()
        return ""

    @staticmethod
    def _strip_markdown_command_prefix(line: str) -> str:
        stripped = str(line or "").strip()
        stripped = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+|>\s*)+", "", stripped).strip()
        return stripped.strip("`").strip()

    @classmethod
    def _normalize_skill_command_template_line(
        cls,
        line: str,
        cli_value: str,
    ) -> str:
        stripped = cls._strip_markdown_command_prefix(line)
        if not stripped:
            return ""

        lower = stripped.casefold()
        if lower.startswith("run "):
            if re.search(r"\bagain\b", stripped, re.IGNORECASE):
                return ""
            stripped = stripped[4:].strip()
            lower = stripped.casefold()

        if stripped.startswith("$CLI"):
            command_line = stripped
        elif cli_value and stripped.startswith(cli_value):
            command_line = stripped
        else:
            return ""

        expanded = command_line.replace("$CLI", cli_value or "$CLI", 1)
        parseable = expanded.replace("{", "<").replace("}", ">")
        try:
            tokens = shlex.split(parseable, comments=True)
        except ValueError:
            tokens = parseable.split()
        return " ".join(tokens)

    @classmethod
    def _iter_skill_command_template_lines(
        cls,
        content: str,
        cli_value: str,
    ) -> list[str]:
        lines = str(content or "").splitlines()
        command_section_lines: list[str] = []
        in_commands_section = False
        for raw_line in lines:
            stripped = raw_line.strip()
            if re.match(r"^#{1,6}\s+commands\b", stripped, re.IGNORECASE):
                in_commands_section = True
                continue
            if in_commands_section and re.match(r"^#{1,6}\s+\S", stripped):
                break
            if in_commands_section:
                normalized = cls._normalize_skill_command_template_line(
                    raw_line,
                    cli_value,
                )
                if normalized:
                    command_section_lines.append(normalized)

        if command_section_lines:
            combined_lines: list[str] = []
            seen_lines: set[str] = set()
            for normalized in command_section_lines:
                if normalized not in seen_lines:
                    seen_lines.add(normalized)
                    combined_lines.append(normalized)
            for raw_line in lines:
                stripped = cls._strip_markdown_command_prefix(raw_line)
                if not stripped.casefold().startswith("run "):
                    continue
                normalized = cls._normalize_skill_command_template_line(
                    raw_line,
                    cli_value,
                )
                if normalized and normalized not in seen_lines:
                    seen_lines.add(normalized)
                    combined_lines.append(normalized)
            return combined_lines

        fallback_lines: list[str] = []
        for raw_line in lines:
            normalized = cls._normalize_skill_command_template_line(
                raw_line,
                cli_value,
            )
            if normalized:
                fallback_lines.append(normalized)
        return fallback_lines

    @classmethod
    def _parse_skill_command_templates(
        cls,
        skill_md: Path,
    ) -> list[_SkillCommandTemplate]:
        try:
            content = skill_md.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []
        references_dir = skill_md.parent / "references"
        if references_dir.is_dir():
            reference_chunks: list[str] = []
            for reference_md in sorted(references_dir.glob("*.md"))[:12]:
                try:
                    reference_chunks.append(
                        reference_md.read_text(encoding="utf-8", errors="replace")
                    )
                except Exception:
                    continue
            if reference_chunks:
                content = content + "\n\n" + "\n\n".join(reference_chunks)

        templates: list[_SkillCommandTemplate] = []
        seen_templates: set[tuple[Any, ...]] = set()
        cli_value = cls._skill_cli_assignment(content)
        for expanded in cls._iter_skill_command_template_lines(content, cli_value):
            parseable = expanded.replace("{", "<").replace("}", ">")
            try:
                tokens = shlex.split(parseable, comments=True)
            except ValueError:
                tokens = parseable.split()
            if not tokens:
                continue

            prefix_tokens: tuple[str, ...] = ()
            tail = list(tokens)
            if cli_value:
                try:
                    cli_tokens = shlex.split(cli_value)
                except ValueError:
                    cli_tokens = cli_value.split()
                if len(tail) >= len(cli_tokens):
                    prefix_tokens = tuple(tail[: len(cli_tokens)])
                    tail = tail[len(cli_tokens):]
            elif tail and tail[0] == "$CLI":
                tail = tail[1:]

            fixed_tokens: list[str] = []
            positional_labels: list[str] = []
            allowed_flags: set[str] = set()
            flags_with_values: set[str] = set()
            flag_value_labels: dict[str, str] = {}
            max_positionals = 0
            saw_positional = False
            index = 0
            while index < len(tail):
                raw_token = tail[index]
                token = cls._strip_template_token(raw_token)
                if not token:
                    index += 1
                    continue
                if token.startswith("-"):
                    flag = token.split("=", 1)[0]
                    allowed_flags.add(flag)
                    next_token = (
                        cls._strip_template_token(tail[index + 1])
                        if index + 1 < len(tail)
                        else ""
                    )
                    if next_token and cls._template_token_is_placeholder(next_token):
                        flags_with_values.add(flag)
                        label = cls._template_placeholder_label(next_token)
                        if label:
                            flag_value_labels[flag] = label
                        index += 2
                    else:
                        index += 1
                    continue
                if cls._template_token_is_placeholder(raw_token):
                    max_positionals += 1
                    saw_positional = True
                    label = cls._template_placeholder_label(raw_token)
                    positional_labels.append(label or f"arg{max_positionals}")
                    index += 1
                    continue
                if saw_positional:
                    max_positionals += 1
                    positional_labels.append(f"arg{max_positionals}")
                    index += 1
                    continue
                fixed_tokens.append(token)
                index += 1

            if not fixed_tokens:
                continue
            key = tuple(fixed_tokens)
            fingerprint = (
                tuple(cls._normalize_skill_cli_prefix_tokens(prefix_tokens)),
                key,
                tuple(positional_labels),
                max_positionals,
                tuple(sorted(allowed_flags)),
                tuple(sorted(flags_with_values)),
                tuple(sorted(flag_value_labels.items())),
            )
            if fingerprint in seen_templates:
                continue
            seen_templates.add(fingerprint)
            templates.append(
                _SkillCommandTemplate(
                    prefix_tokens=prefix_tokens,
                    fixed_tokens=key,
                    positional_labels=tuple(positional_labels),
                    max_positionals=max_positionals,
                    allowed_flags=frozenset(allowed_flags),
                    flags_with_values=frozenset(flags_with_values),
                    flag_value_labels=tuple(sorted(flag_value_labels.items())),
                    display=expanded,
                )
            )

        return templates

    @staticmethod
    def _segment_before_redirection_or_pipe(tokens: list[str]) -> list[str]:
        cleaned: list[str] = []
        for token in tokens:
            if ShellTool._is_lossless_stdio_merge_token(token):
                continue
            if ShellTool._is_shell_redirection_or_pipe_token(token):
                break
            cleaned.append(token)
        return cleaned

    @staticmethod
    def _is_lossless_stdio_merge_token(token: str) -> bool:
        value = str(token or "").strip()
        return bool(re.fullmatch(r"(?:[12])?>&[12]", value))

    @staticmethod
    def _is_shell_redirection_or_pipe_token(token: str) -> bool:
        value = str(token or "").strip()
        return (
            value in {"|", "|&"}
            or value.startswith((">", "<", "1>", "2>", "&>"))
            or bool(re.fullmatch(r"\d?>&\d+", value))
        )

    @classmethod
    def _segment_has_disallowed_output_control(cls, tokens: list[str]) -> bool:
        return any(
            cls._is_shell_redirection_or_pipe_token(token)
            for token in tokens
        )

    @classmethod
    def _strip_lossless_stdio_merge_tokens(cls, tokens: list[str]) -> list[str]:
        return [
            token
            for token in tokens
            if not cls._is_lossless_stdio_merge_token(token)
        ]

    @staticmethod
    def _timeout_wrapped_segment(tokens: list[str]) -> list[str] | None:
        """Return the command inside a leading GNU timeout wrapper, if present."""
        if not tokens:
            return None
        command_name = os.path.basename(
            str(tokens[0] or "").strip().strip("'\"").replace("\\", "/")
        ).casefold()
        if command_name not in {"timeout", "gtimeout"}:
            return None

        duration_re = re.compile(r"^\d+(?:\.\d+)?(?:[smhd])?$", re.IGNORECASE)
        for index, token in enumerate(tokens[1:], start=1):
            value = str(token or "").strip()
            if duration_re.fullmatch(value):
                return tokens[index + 1 :]
        return None

    @classmethod
    def _format_workspace_skill_wrapper_rejection(
        cls,
        skill_name: str,
        reason: str,
        direct_segment: list[str],
    ) -> str:
        direct = shlex.join(direct_segment) if direct_segment else "the documented CLI command"
        return (
            "Rejected: workspace skill CLI commands must run exactly as "
            f"documented in skills/{skill_name}/SKILL.md. Detected {reason}. "
            "Run the CLI in the foreground without shell pipes, output filters, "
            "file redirection, stderr/stdout merge, timeout wrappers, or other "
            "shell wrappers; the shell tool already captures stdout and stderr. Retry the "
            f"documented command directly: {direct}"
        )

    @classmethod
    def _workspace_skill_setup_command_name(
        cls,
        segment: list[str],
        current_dir: Path,
        workspace_root: Path,
    ) -> str:
        """Return skill name when a segment is a dependency install inside a skill."""
        if len(segment) < 2:
            return ""
        command_name = os.path.basename(
            str(segment[0] or "").strip().strip("'\"").replace("\\", "/")
        ).casefold()
        if command_name not in {"bun", "npm", "pnpm", "yarn"}:
            return ""
        subcommand = str(segment[1] or "").strip().casefold()
        if subcommand not in {"ci", "install", "i"}:
            return ""

        try:
            relative = current_dir.resolve(strict=False).relative_to(
                workspace_root.resolve(strict=False)
            )
        except ValueError:
            return ""
        parts = relative.parts
        if len(parts) < 2 or parts[0] != "skills":
            return ""
        return parts[1]

    @classmethod
    def _format_workspace_skill_setup_wrapper_rejection(
        cls,
        skill_name: str,
        reason: str,
        direct_segment: list[str],
        current_dir: Path,
        workspace_root: Path,
    ) -> str:
        direct = shlex.join(direct_segment) if direct_segment else "the dependency install command"
        try:
            rel_dir = current_dir.resolve(strict=False).relative_to(
                workspace_root.resolve(strict=False)
            )
            direct = f"cd {shlex.quote(str(rel_dir))} && {direct}"
        except ValueError:
            pass
        return (
            "Rejected: workspace skill dependency setup commands must run "
            f"directly for skills/{skill_name}/SKILL.md. Detected {reason}. "
            "Run the setup command in the foreground without shell pipes, "
            "output filters, file redirection, stderr/stdout merge, timeout "
            "wrappers, or other shell wrappers; the shell tool already captures "
            "stdout and stderr. "
            f"Retry the setup command directly: {direct}"
        )

    @classmethod
    def _token_can_be_skill_entrypoint(cls, tokens: list[str], index: int) -> bool:
        if index <= 0:
            return True
        command_name = os.path.basename(
            str(tokens[0] or "").strip().strip("'\"").replace("\\", "/")
        ).casefold()
        return command_name in cls._SKILL_ENTRYPOINT_WRAPPERS

    @classmethod
    def _extract_skill_cli_invocation(
        cls,
        tokens: list[str],
    ) -> tuple[str, list[str], list[str]] | None:
        for index, token in enumerate(tokens):
            if not cls._token_can_be_skill_entrypoint(tokens, index):
                continue
            normalized = str(token or "").replace("\\", "/").strip("'\"")
            if normalized.startswith("skills/"):
                parts = normalized.split("/")
                if len(parts) >= 3:
                    return parts[1], tokens[index + 1 :], tokens[: index + 1]
            marker_index = normalized.find("/skills/")
            if marker_index >= 0:
                remainder = normalized[marker_index + 1 :]
                parts = remainder.split("/")
                if len(parts) >= 3 and parts[0] == "skills":
                    return parts[1], tokens[index + 1 :], tokens[: index + 1]
        return None

    @staticmethod
    def _workspace_skill_relative_path(
        candidate: Path,
        workspace_root: Path,
    ) -> tuple[str, str] | None:
        try:
            resolved = candidate.resolve(strict=False)
            relative = resolved.relative_to(workspace_root.resolve())
        except Exception:
            return None

        parts = relative.parts
        if len(parts) < 3 or parts[0] != "skills":
            return None
        skill_name = parts[1]
        return skill_name, "/".join(parts)

    @classmethod
    def _extract_skill_cli_invocation_for_segment(
        cls,
        tokens: list[str],
        current_dir: Path,
        workspace_root: Path,
    ) -> tuple[str, list[str], list[str]] | None:
        invocation = cls._extract_skill_cli_invocation(tokens)
        if invocation is not None:
            return invocation

        for index, token in enumerate(tokens):
            if not cls._token_can_be_skill_entrypoint(tokens, index):
                continue
            normalized = str(token or "").replace("\\", "/").strip("'\"")
            if (
                not normalized
                or normalized.startswith("-")
                or normalized in {"node", "python", "python3", "bash", "sh"}
            ):
                continue
            if "/" not in normalized and not re.search(
                r"\.(?:cjs|mjs|js|py|sh)$",
                normalized,
                re.IGNORECASE,
            ):
                continue

            skill_path = cls._workspace_skill_relative_path(
                current_dir / normalized,
                workspace_root,
            )
            if skill_path is None:
                continue
            skill_name, relative_path = skill_path
            return skill_name, tokens[index + 1 :], [*tokens[:index], relative_path]

        return None

    @staticmethod
    def _normalize_skill_cli_prefix_token(token: str) -> str:
        value = str(token or "").strip().strip("'\"").replace("\\", "/")
        marker_index = value.find("/skills/")
        if marker_index >= 0:
            value = value[marker_index + 1 :]
        while value.startswith("./"):
            value = value[2:]
        return value.casefold()

    @classmethod
    def _normalize_skill_cli_prefix_tokens(cls, tokens: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        return tuple(
            cls._normalize_skill_cli_prefix_token(token)
            for token in tokens
            if str(token or "").strip()
        )

    @staticmethod
    def _flag_name(token: str) -> str:
        return str(token or "").split("=", 1)[0]

    @staticmethod
    def _label_tokens(label: str) -> set[str]:
        tokens: set[str] = set()
        buffer: list[str] = []
        for char in str(label or "").casefold():
            if char.isalnum() or char == "_":
                buffer.append(char)
                continue
            if len(buffer) >= 2 and buffer[0].isalnum():
                tokens.add("".join(buffer))
            buffer = []
        if len(buffer) >= 2 and buffer[0].isalnum():
            tokens.add("".join(buffer))
        return tokens

    @classmethod
    def _label_accepts_request_value(
        cls,
        template_label: str,
        request_labels: set[str],
    ) -> bool:
        template_labels = cls._label_tokens(template_label)
        return bool(template_labels and request_labels and template_labels & request_labels)

    @classmethod
    def _skill_args_contain_literal_placeholder(
        cls,
        args: list[str],
        templates: list[_SkillCommandTemplate],
    ) -> bool:
        if not args or not templates:
            return False
        normalized_args = [str(arg or "").strip().casefold() for arg in args]
        for template in templates:
            if tuple(args[: len(template.fixed_tokens)]) != template.fixed_tokens:
                continue
            label_tokens: set[str] = set()
            for label in template.positional_labels:
                label_tokens.update(cls._label_tokens(label))
            if not label_tokens:
                continue
            for token in normalized_args[len(template.fixed_tokens) :]:
                if not token or token.startswith("-"):
                    continue
                if token in label_tokens:
                    return True
        return False

    @classmethod
    def _flag_alias_value_label(
        cls,
        flag: str,
        template: _SkillCommandTemplate,
    ) -> str:
        flag_tokens = cls._label_tokens(flag.lstrip("-"))
        if not flag_tokens:
            return ""
        for label in dict(template.flag_value_labels).values():
            if flag_tokens & cls._label_tokens(label):
                return label
        return ""

    def _request_explicit_labeled_values(self) -> dict[str, set[str]]:
        hints = get_request_execution_hints()
        if not hints:
            hints = getattr(self, "_request_execution_hints", {})
        values = hints.get("explicit_request_values") if isinstance(hints, dict) else None
        if not isinstance(values, list):
            return {}
        result: dict[str, set[str]] = {}
        for entry in values:
            if not isinstance(entry, dict):
                continue
            value = str(entry.get("value") or "").strip()
            labels = entry.get("labels")
            if not value or not isinstance(labels, list):
                continue
            label_set = {
                str(label).strip().casefold()
                for label in labels
                if isinstance(label, str) and label.strip()
            }
            if label_set:
                result[value] = label_set
        return result

    @staticmethod
    def _template_preferred_flag(flags: list[str]) -> str:
        """Choose the most explicit documented flag from equivalent candidates."""
        if not flags:
            return ""
        return sorted(
            flags,
            key=lambda item: (
                0 if str(item).startswith("--") else 1,
                -len(str(item)),
                str(item),
            ),
        )[0]

    @classmethod
    def _positional_values_for_template(cls, args: list[str], template: _SkillCommandTemplate) -> list[str]:
        remaining = args[len(template.fixed_tokens):]
        positionals: list[str] = []
        index = 0
        flag_value_labels = dict(template.flag_value_labels)
        while index < len(remaining):
            token = remaining[index]
            if not token:
                index += 1
                continue
            if token.startswith("-"):
                flag = cls._flag_name(token)
                if (flag in template.flags_with_values or flag in flag_value_labels) and "=" not in token:
                    index += 2
                else:
                    index += 1
                continue
            positionals.append(token)
            index += 1
        return positionals

    @classmethod
    def _observed_skill_cli_flags(
        cls,
        args: list[str],
        template: _SkillCommandTemplate,
    ) -> list[str]:
        remaining = args[len(template.fixed_tokens):]
        flags: set[str] = set()
        index = 0
        flag_value_labels = dict(template.flag_value_labels)
        while index < len(remaining):
            token = remaining[index]
            if not token:
                index += 1
                continue
            if not token.startswith("-"):
                index += 1
                continue
            flag = cls._flag_name(token).casefold()
            if flag:
                flags.add(flag)
            if (flag in template.flags_with_values or flag in flag_value_labels) and "=" not in token:
                index += 2
            else:
                index += 1
        return sorted(flags)

    @classmethod
    def _observed_skill_cli_flag_values(
        cls,
        args: list[str],
        template: _SkillCommandTemplate,
    ) -> list[tuple[str, str]]:
        remaining = args[len(template.fixed_tokens):]
        values: list[tuple[str, str]] = []
        index = 0
        flag_value_labels = dict(template.flag_value_labels)
        while index < len(remaining):
            token = remaining[index]
            if not token:
                index += 1
                continue
            if not token.startswith("-"):
                index += 1
                continue
            flag = cls._flag_name(token).casefold()
            if not flag:
                index += 1
                continue
            if "=" in token:
                values.append((flag, token.split("=", 1)[1]))
                index += 1
                continue
            if flag in template.flags_with_values or flag in flag_value_labels:
                next_value = remaining[index + 1] if index + 1 < len(remaining) else ""
                values.append((flag, next_value))
                index += 2
                continue
            values.append((flag, ""))
            index += 1
        return sorted(values)

    def _repair_skill_cli_alias_flags_from_request_values(
        self,
        command_tokens: list[str],
        prefix_tokens: list[str],
        args: list[str],
        templates: list[_SkillCommandTemplate],
        explicit_request_values: dict[str, set[str]],
    ) -> list[str] | None:
        """Rewrite unsupported value flags to documented flags or positionals.

        The repair is intentionally data-driven: it only fires when the value is
        a structured user-provided fact and SKILL.md documents exactly where a
        value with that label belongs.
        """
        if not explicit_request_values:
            return None

        prefix_len = len(prefix_tokens)
        for template in sorted(
            templates,
            key=lambda item: (len(item.fixed_tokens), item.max_positionals),
            reverse=True,
        ):
            if tuple(args[: len(template.fixed_tokens)]) != template.fixed_tokens:
                continue
            remaining = args[len(template.fixed_tokens):]
            offset = len(template.fixed_tokens)
            index = 0
            while index < len(remaining):
                token = remaining[index]
                if not token or not token.startswith("-"):
                    index += 1
                    continue
                observed_flag = self._flag_name(token)
                if observed_flag in template.allowed_flags:
                    if observed_flag in template.flags_with_values and "=" not in token:
                        index += 2
                    else:
                        index += 1
                    continue

                value = ""
                value_is_inline = False
                if "=" in token:
                    value = token.split("=", 1)[1]
                    value_is_inline = True
                elif index + 1 < len(remaining):
                    value = remaining[index + 1]
                request_labels = explicit_request_values.get(value)
                if not value or not request_labels:
                    index += 1
                    continue

                command_index = prefix_len + offset + index
                flag_value_labels = dict(template.flag_value_labels)
                documented_flags = [
                    flag
                    for flag, label in flag_value_labels.items()
                    if flag in template.flags_with_values
                    and self._label_accepts_request_value(label, request_labels)
                ]
                replacement_flag = self._template_preferred_flag(documented_flags)
                if replacement_flag:
                    updated = list(command_tokens)
                    updated[command_index] = (
                        f"{replacement_flag}={value}"
                        if value_is_inline
                        else replacement_flag
                    )
                    logger.info(
                        "Repaired unsupported skill CLI value flag "
                        f"{observed_flag!r} to documented flag "
                        f"{replacement_flag!r} from SKILL.md."
                    )
                    return updated

                positionals = self._positional_values_for_template(args, template)
                if len(positionals) < len(template.positional_labels):
                    label = template.positional_labels[len(positionals)]
                    if self._label_accepts_request_value(label, request_labels):
                        updated = list(command_tokens)
                        if value_is_inline:
                            updated[command_index] = value
                        else:
                            del updated[command_index]
                        logger.info(
                            "Repaired unsupported skill CLI value flag "
                            f"{observed_flag!r} to documented positional "
                            f"parameter {label!r} from SKILL.md."
                        )
                        return updated

                index += 1
        return None

    @classmethod
    def _extra_positional_arg_indices(
        cls,
        args: list[str],
        template: _SkillCommandTemplate,
    ) -> list[int]:
        remaining = args[len(template.fixed_tokens):]
        extras: list[int] = []
        positionals = 0
        index = 0
        flag_value_labels = dict(template.flag_value_labels)
        while index < len(remaining):
            token = remaining[index]
            if not token:
                index += 1
                continue
            if token.startswith("-"):
                flag = cls._flag_name(token)
                if (
                    flag in template.flags_with_values
                    or flag in flag_value_labels
                ) and "=" not in token:
                    index += 2
                else:
                    index += 1
                continue
            if positionals >= template.max_positionals:
                extras.append(len(template.fixed_tokens) + index)
            positionals += 1
            index += 1
        return extras

    def _repair_extra_skill_cli_positionals_from_request_values(
        self,
        command_tokens: list[str],
        prefix_tokens: list[str],
        args: list[str],
        templates: list[_SkillCommandTemplate],
        explicit_request_values: dict[str, set[str]],
    ) -> list[str] | None:
        """Drop user-provided values that SKILL.md does not document as CLI args."""
        if not explicit_request_values:
            return None

        prefix_len = len(prefix_tokens)
        for template in sorted(
            templates,
            key=lambda item: (len(item.fixed_tokens), item.max_positionals),
            reverse=True,
        ):
            if tuple(args[: len(template.fixed_tokens)]) != template.fixed_tokens:
                continue
            observed_flags = {
                self._flag_name(token)
                for token in args[len(template.fixed_tokens):]
                if str(token or "").startswith("-")
            }
            if any(
                flag not in template.allowed_flags
                and not self._flag_alias_value_label(flag, template)
                for flag in observed_flags
            ):
                continue
            extra_arg_indices = self._extra_positional_arg_indices(args, template)
            if not extra_arg_indices:
                continue
            extra_values = [args[index] for index in extra_arg_indices]
            if not all(value in explicit_request_values for value in extra_values):
                continue
            remove_indices = {prefix_len + index for index in extra_arg_indices}
            updated = [
                token
                for index, token in enumerate(command_tokens)
                if index not in remove_indices
            ]
            logger.info(
                "Dropped extra skill CLI positional request value(s) not "
                f"documented by skills command form: {template.display}."
            )
            return updated
        return None

    def _repair_unsupported_skill_cli_flags_from_request_values_without_slots(
        self,
        command_tokens: list[str],
        prefix_tokens: list[str],
        args: list[str],
        templates: list[_SkillCommandTemplate],
        explicit_request_values: dict[str, set[str]],
    ) -> list[str] | None:
        """Drop unsupported value flags when SKILL.md has no matching slot.

        The model sometimes turns labeled request facts into nearby CLI flags
        even when the active SKILL.md command form documents no parameter for
        those facts.  This repair is intentionally narrow: it only removes a
        flag when the attached value came from the structured user request and
        no documented flag or positional slot accepts that value's label.
        """
        if not explicit_request_values:
            return None

        prefix_len = len(prefix_tokens)
        for template in sorted(
            templates,
            key=lambda item: (len(item.fixed_tokens), item.max_positionals),
            reverse=True,
        ):
            if tuple(args[: len(template.fixed_tokens)]) != template.fixed_tokens:
                continue

            remaining = args[len(template.fixed_tokens) :]
            remove_indices: set[int] = set()
            offset = len(template.fixed_tokens)
            index = 0
            while index < len(remaining):
                token = remaining[index]
                if not token or not token.startswith("-"):
                    index += 1
                    continue

                flag = self._flag_name(token)
                if flag in template.allowed_flags:
                    if flag in template.flags_with_values and "=" not in token:
                        index += 2
                    else:
                        index += 1
                    continue

                value = ""
                value_index: int | None = None
                if "=" in token:
                    value = token.split("=", 1)[1]
                    value_index = None
                elif index + 1 < len(remaining):
                    value = remaining[index + 1]
                    value_index = prefix_len + offset + index + 1

                request_labels = explicit_request_values.get(value)
                if not value or not request_labels:
                    index += 1
                    continue

                flag_value_labels = dict(template.flag_value_labels)
                accepts_flag_value = any(
                    candidate_flag in template.flags_with_values
                    and self._label_accepts_request_value(label, request_labels)
                    for candidate_flag, label in flag_value_labels.items()
                )
                accepts_positional_value = any(
                    self._label_accepts_request_value(label, request_labels)
                    for label in template.positional_labels
                )
                if accepts_flag_value or accepts_positional_value:
                    index += 2 if value_index is not None else 1
                    continue

                remove_indices.add(prefix_len + offset + index)
                if value_index is not None:
                    remove_indices.add(value_index)
                index += 2 if value_index is not None else 1

            if not remove_indices:
                continue

            updated = [
                token
                for index, token in enumerate(command_tokens)
                if index not in remove_indices
            ]
            logger.info(
                "Dropped unsupported skill CLI request value flag(s) not "
                f"documented by skills command form: {template.display}."
            )
            return updated
        return None

    @classmethod
    def _validate_skill_cli_args_against_template(
        cls,
        skill_name: str,
        prefix_tokens: list[str],
        args: list[str],
        template: _SkillCommandTemplate,
        explicit_request_values: dict[str, set[str]],
    ) -> str | None:
        if template.prefix_tokens:
            observed_prefix = cls._normalize_skill_cli_prefix_tokens(prefix_tokens)
            documented_prefix = cls._normalize_skill_cli_prefix_tokens(
                template.prefix_tokens
            )
            if observed_prefix != documented_prefix:
                return (
                    "Rejected: this skill CLI entrypoint is not documented in "
                    f"skills/{skill_name}/SKILL.md. Use the documented "
                    f"entrypoint from this command form: {template.display}. "
                    "Do not invent nearby build output paths or alternate "
                    "wrappers inside the skill directory."
                )

        remaining = args[len(template.fixed_tokens) :]
        if remaining in (["--help"], ["-h"]):
            return None

        positionals = 0
        positional_tokens: list[str] = []
        index = 0
        flag_value_labels = dict(template.flag_value_labels)
        while index < len(remaining):
            token = remaining[index]
            if not token:
                index += 1
                continue
            if token.startswith("-"):
                flag = cls._flag_name(token)
                alias_label = ""
                if flag not in template.allowed_flags:
                    alias_label = cls._flag_alias_value_label(flag, template)
                if flag not in template.allowed_flags and not alias_label:
                    allowed = ", ".join(sorted(template.allowed_flags)) or "none"
                    return (
                        "Rejected: unsupported option "
                        f"{flag!r} for skill command "
                        f"{' '.join(template.fixed_tokens)!r}. "
                        f"Follow skills/{skill_name}/SKILL.md exactly; do not "
                        "invent CLI flags from user wording. "
                        f"Documented form: {template.display}. "
                        f"Allowed options: {allowed}."
                    )
                if (
                    (flag in template.flags_with_values or alias_label)
                    and "=" not in token
                ):
                    value = remaining[index + 1] if index + 1 < len(remaining) else ""
                    label = flag_value_labels.get(flag, "") or alias_label
                    request_labels = explicit_request_values.get(value)
                    if (
                        request_labels
                        and not cls._label_accepts_request_value(label, request_labels)
                    ):
                        return (
                            "Rejected: this argument is a labeled value from "
                            "the user request, but skill command "
                            f"{' '.join(template.fixed_tokens)!r} does not "
                            f"document {flag!r} with a matching parameter "
                            f"label. Documented form: {template.display}."
                        )
                    index += 2
                else:
                    index += 1
                continue
            label = (
                template.positional_labels[positionals]
                if positionals < len(template.positional_labels)
                else ""
            )
            positional_label_tokens: set[str] = set()
            for template_label in template.positional_labels:
                positional_label_tokens.update(cls._label_tokens(template_label))
            if token.casefold() in positional_label_tokens:
                return (
                    "Rejected: this skill CLI argument is a placeholder label "
                    f"passed literally: {token!r}. Follow skills/{skill_name}/SKILL.md "
                    "with a concrete value for that placeholder instead of the "
                    "placeholder name itself."
                )
            request_labels = explicit_request_values.get(token)
            if (
                request_labels
                and label
                and not cls._label_accepts_request_value(label, request_labels)
            ):
                return (
                    "Rejected: this argument is a labeled value from the "
                    "user request, but "
                    f"skill command {' '.join(template.fixed_tokens)!r} does "
                    "not document that positional with a matching parameter "
                    f"label. Documented form: {template.display}."
                )
            positional_tokens.append(token)
            positionals += 1
            index += 1

        if positionals > template.max_positionals:
            extra_positionals = positional_tokens[template.max_positionals :]
            if cls._explicit_request_values_allow_undocumented_positionals(
                template,
                extra_positionals,
                explicit_request_values,
            ):
                return None
            return (
                "Rejected: too many positional arguments for skill command "
                f"{' '.join(template.fixed_tokens)!r}. Follow "
                f"skills/{skill_name}/SKILL.md exactly; do not pass values from "
                "the user request unless the documented command form includes a "
                f"matching placeholder. Documented form: {template.display}."
            )
        return None

    @classmethod
    def _explicit_request_values_allow_undocumented_positionals(
        cls,
        template: _SkillCommandTemplate,
        extra_positionals: list[str],
        explicit_request_values: dict[str, set[str]],
    ) -> bool:
        """Undocumented extra request values are repaired before validation."""
        return False

    @classmethod
    def _validate_skill_cli_args_against_templates(
        cls,
        skill_name: str,
        prefix_tokens: list[str],
        args: list[str],
        templates: list[_SkillCommandTemplate],
        explicit_request_values: dict[str, set[str]],
    ) -> str | None:
        if not args or not templates:
            return None

        matching_templates = [
            template
            for template in templates
            if tuple(args[: len(template.fixed_tokens)]) == template.fixed_tokens
        ]
        if not matching_templates:
            documented = ", ".join(template.display for template in templates[:6])
            return (
                "Rejected: this skill CLI invocation is not documented in "
                f"skills/{skill_name}/SKILL.md. Use one of the documented command "
                f"forms instead. Documented forms: {documented}"
            )

        violations: list[str] = []
        for template in sorted(
            matching_templates,
            key=lambda item: (len(item.fixed_tokens), item.max_positionals),
            reverse=True,
        ):
            violation = cls._validate_skill_cli_args_against_template(
                skill_name,
                prefix_tokens,
                args,
                template,
                explicit_request_values,
            )
            if violation is None:
                return None
            violations.append(violation)
        return violations[0] if violations else None

    def _augment_skill_cli_labeled_values(self, command: str, cwd: str) -> str:
        """Fill documented skill CLI value flags from structured request facts."""
        explicit_request_values = self._request_explicit_labeled_values()
        if not explicit_request_values:
            return command

        try:
            command_tokens = shlex.split(str(command or ""))
        except ValueError:
            return command
        if not command_tokens:
            return command
        command_tokens = self._strip_lossless_stdio_merge_tokens(command_tokens)
        if not command_tokens:
            return command
        if any(token in {"||", ";", "|"} for token in command_tokens):
            return command
        if "&&" in command_tokens:
            segments = self._split_shell_segments(command_tokens)
            if not segments:
                return command
            current_dir = Path(cwd).resolve()
            rendered_segments: list[str] = []
            changed = False
            for segment in segments:
                if not segment:
                    continue
                if segment[0].casefold() == "cd":
                    current_dir = self._resolve_cd_segment(current_dir, segment)
                    rendered_segments.append(shlex.join(segment))
                    continue
                segment_command = shlex.join(segment)
                augmented_segment = self._augment_skill_cli_labeled_values(
                    segment_command,
                    str(current_dir),
                )
                if augmented_segment != segment_command:
                    changed = True
                rendered_segments.append(augmented_segment)
            return " && ".join(rendered_segments) if changed else command
        if self._segment_has_disallowed_output_control(command_tokens):
            return command

        workspace_root = Path(self.working_dir or cwd).resolve()
        invocation = self._extract_skill_cli_invocation_for_segment(
            command_tokens,
            Path(cwd).resolve(),
            workspace_root,
        )
        if invocation is None:
            return command
        skill_name, args, prefix_tokens = invocation
        if not args:
            return command

        skill_md = workspace_root / "skills" / skill_name / "SKILL.md"
        if not skill_md.exists():
            fallback_skill_md = Path(cwd).resolve() / "skills" / skill_name / "SKILL.md"
            if fallback_skill_md.exists():
                skill_md = fallback_skill_md
        templates = self._parse_skill_command_templates(skill_md)
        matching_templates = [
            template
            for template in templates
            if tuple(args[: len(template.fixed_tokens)]) == template.fixed_tokens
        ]
        if not matching_templates:
            return command

        current_violation = self._validate_skill_cli_args_against_templates(
            skill_name,
            list(prefix_tokens),
            args,
            matching_templates,
            explicit_request_values,
        )
        if current_violation:
            repaired_tokens = self._repair_skill_cli_alias_flags_from_request_values(
                command_tokens,
                list(prefix_tokens),
                args,
                matching_templates,
                explicit_request_values,
            )
            if repaired_tokens is not None:
                return shlex.join(repaired_tokens)
            repaired_tokens = (
                self._repair_unsupported_skill_cli_flags_from_request_values_without_slots(
                    command_tokens,
                    list(prefix_tokens),
                    args,
                    matching_templates,
                    explicit_request_values,
                )
            )
            if repaired_tokens is not None:
                return self._augment_skill_cli_labeled_values(
                    shlex.join(repaired_tokens),
                    cwd,
                )
            repaired_tokens = self._repair_extra_skill_cli_positionals_from_request_values(
                command_tokens,
                list(prefix_tokens),
                args,
                matching_templates,
                explicit_request_values,
            )
            if repaired_tokens is not None:
                return shlex.join(repaired_tokens)

        existing_flags = {
            self._flag_name(token)
            for token in args
            if str(token or "").startswith("-")
        }
        existing_values = set(args)
        for template in sorted(
            matching_templates,
            key=lambda item: (len(item.fixed_tokens), item.max_positionals),
            reverse=True,
        ):
            for flag, label in dict(template.flag_value_labels).items():
                if flag in existing_flags:
                    continue
                for value, request_labels in explicit_request_values.items():
                    if value in existing_values:
                        continue
                    if not self._label_accepts_request_value(label, request_labels):
                        continue
                    prefix_len = len(prefix_tokens)
                    insert_at = prefix_len + len(template.fixed_tokens)
                    updated = list(command_tokens)
                    updated[insert_at:insert_at] = [flag, value]
                    augmented = shlex.join(updated)
                    logger.info(
                        "Augmented skill CLI command with structured request "
                        f"value for skills/{skill_name}/SKILL.md parameter {flag}."
                    )
                    return augmented
            remaining = args[len(template.fixed_tokens) :]
            positional_values: list[str] = []
            index = 0
            flag_value_labels = dict(template.flag_value_labels)
            while index < len(remaining):
                token = remaining[index]
                if not token:
                    index += 1
                    continue
                if token.startswith("-"):
                    flag = self._flag_name(token)
                    if (
                        flag in template.flags_with_values
                        or flag in flag_value_labels
                    ) and "=" not in token:
                        index += 2
                    else:
                        index += 1
                    continue
                positional_values.append(token)
                index += 1

            if len(positional_values) >= len(template.positional_labels):
                continue
            label = template.positional_labels[len(positional_values)]
            for value, request_labels in explicit_request_values.items():
                if value in existing_values:
                    continue
                if not self._label_accepts_request_value(label, request_labels):
                    continue
                updated = [*command_tokens, value]
                augmented = shlex.join(updated)
                logger.info(
                    "Augmented skill CLI command with structured request "
                    f"value for skills/{skill_name}/SKILL.md positional "
                    f"parameter {label!r}."
                )
                return augmented
        return command

    def _reject_undocumented_skill_cli_arguments(
        self,
        command: str,
        cwd: str,
    ) -> str | None:
        """Reject invented flags/positionals for installed skill CLI commands."""
        try:
            command_tokens = shlex.split(str(command or ""))
        except ValueError:
            command_tokens = str(command or "").split()
        if not command_tokens:
            return None

        workspace_root = Path(self.working_dir or cwd).resolve()
        current_dir = Path(cwd).resolve()
        template_cache: dict[str, list[_SkillCommandTemplate]] = {}

        for segment in self._split_shell_segments(command_tokens):
            if not segment:
                continue
            if segment[0].casefold() == "cd":
                current_dir = self._resolve_cd_segment(current_dir, segment)
                continue
            cleaned_segment = self._segment_before_redirection_or_pipe(segment)
            invocation = self._extract_skill_cli_invocation_for_segment(
                cleaned_segment,
                current_dir,
                workspace_root,
            )
            if invocation is None:
                continue
            skill_name, args, prefix_tokens = invocation
            if not args:
                continue

            skill_md = workspace_root / "skills" / skill_name / "SKILL.md"
            if not skill_md.exists():
                fallback_skill_md = current_dir / "skills" / skill_name / "SKILL.md"
                if fallback_skill_md.exists():
                    skill_md = fallback_skill_md
            templates = template_cache.get(skill_name)
            if templates is None:
                templates = self._parse_skill_command_templates(skill_md)
                template_cache[skill_name] = templates
            if self._skill_cli_invocation_matches_observed_cli_command(
                skill_name,
                args,
                prefix_tokens,
                current_dir,
                workspace_root,
            ):
                continue
            violation = self._validate_skill_cli_args_against_templates(
                skill_name,
                prefix_tokens,
                args,
                templates,
                self._request_explicit_labeled_values(),
            )
            if violation:
                return violation
        return None

    def _skill_cli_invocation_matches_observed_cli_command(
        self,
        skill_name: str,
        args: list[str],
        prefix_tokens: list[str],
        current_dir: Path,
        workspace_root: Path,
    ) -> bool:
        """Return True when previous shell output emitted this exact CLI call."""
        if not skill_name or not args:
            return False

        direct_command = shlex.join([*prefix_tokens, *args])
        if observed_cli_command_matches(direct_command):
            return True

        target_skill = str(skill_name or "").strip().casefold()
        target_args = [str(arg) for arg in args]
        for observed in get_observed_cli_commands():
            try:
                observed_tokens = shlex.split(observed)
            except ValueError:
                observed_tokens = str(observed or "").split()
            if not observed_tokens:
                continue
            observed_dir = Path(current_dir).resolve()
            for segment in self._split_shell_segments(observed_tokens):
                if not segment:
                    continue
                if segment[0].casefold() == "cd":
                    observed_dir = self._resolve_cd_segment(observed_dir, segment)
                    continue
                cleaned_segment = self._segment_before_redirection_or_pipe(segment)
                invocation = self._extract_skill_cli_invocation_for_segment(
                    cleaned_segment,
                    observed_dir,
                    workspace_root,
                )
                if invocation is None:
                    continue
                observed_skill, observed_args, _observed_prefix = invocation
                if (
                    str(observed_skill or "").strip().casefold() == target_skill
                    and [str(arg) for arg in observed_args] == target_args
                ):
                    return True
        return False

    def _command_invokes_workspace_skill(self, command: str, cwd: str) -> bool:
        """Return True when a command runs a CLI documented under workspace/skills."""
        try:
            command_tokens = shlex.split(str(command or ""))
        except ValueError:
            command_tokens = str(command or "").split()
        if not command_tokens:
            return False

        workspace_root = Path(self.working_dir or cwd).resolve()
        current_dir = Path(cwd).resolve()
        for segment in self._split_shell_segments(command_tokens):
            if not segment:
                continue
            if segment[0].casefold() == "cd":
                current_dir = self._resolve_cd_segment(current_dir, segment)
                continue
            cleaned_segment = self._segment_before_redirection_or_pipe(segment)
            invocation = self._extract_skill_cli_invocation_for_segment(
                cleaned_segment,
                current_dir,
                workspace_root,
            )
            if invocation is not None:
                return True

            for token in cleaned_segment:
                normalized = str(token or "").strip().strip("'\"").replace("\\", "/")
                if normalized.startswith("./"):
                    candidate = (current_dir / normalized).resolve(strict=False)
                    if "/skills/" in str(candidate).replace("\\", "/"):
                        return True
        return False

    def _command_invokes_workspace_skill_setup(self, command: str, cwd: str) -> bool:
        """Return True when a command installs dependencies inside workspace/skills."""
        try:
            command_tokens = shlex.split(str(command or ""))
        except ValueError:
            command_tokens = str(command or "").split()
        if not command_tokens:
            return False

        workspace_root = Path(self.working_dir or cwd).resolve()
        current_dir = Path(cwd).resolve()
        for segment in self._split_shell_segments(command_tokens):
            if not segment:
                continue
            if segment[0].casefold() == "cd":
                current_dir = self._resolve_cd_segment(current_dir, segment)
                continue
            cleaned_segment = self._segment_before_redirection_or_pipe(segment)
            if self._workspace_skill_setup_command_name(
                cleaned_segment,
                current_dir,
                workspace_root,
            ):
                return True
        return False

    @staticmethod
    def _looks_like_missing_runtime_dependency(output: str) -> bool:
        """Return True for common runtime loader failures before CLI execution."""
        text = str(output or "").casefold()
        if not text:
            return False
        return any(
            marker in text
            for marker in (
                "cannot find module",
                "module_not_found",
                "module not found",
                "no module named",
                "cannot find package",
                "package not found",
                "cannot resolve package",
            )
        )

    def _append_workspace_skill_dependency_recovery(
        self,
        command: str,
        cwd: str,
        result: str,
    ) -> str:
        """Append a generic setup recovery hint for failed workspace skill CLIs."""
        text = str(result or "")
        if "workspace skill dependency recovery" in text.casefold():
            return text
        if not self._looks_like_missing_runtime_dependency(text):
            return text

        invocation_key = self._skill_cli_invocation_dedup_key(command, cwd)
        invocations = (
            invocation_key.get("invocations")
            if isinstance(invocation_key, dict)
            else None
        )
        if not invocations:
            return text
        skill_name = str((invocations[0] or {}).get("skill") or "").strip()
        if not skill_name:
            return text

        return (
            text.rstrip()
            + "\n\n[workspace skill dependency recovery]\n"
            f"The failed command invoked a CLI under skills/{skill_name}/ and "
            "the runtime reported a missing dependency before the CLI could run. "
            "Use the current local SKILL.md as the source of truth, then run the "
            "documented Setup/Prerequisites/Install dependency command directly "
            "in the foreground before retrying this CLI. If the skill was just "
            "installed or updated, dependency/setup evidence from before that "
            "filesystem change is stale."
        )

    def _maybe_stop_after_exact_command_failure(self, command: str, result: str) -> str:
        """Stop the tool loop after a user-specified exact command fails clearly."""
        hints = get_request_execution_hints()
        exact_commands = hints.get("exact_shell_commands") if isinstance(hints, dict) else None
        if not isinstance(exact_commands, list) or not exact_commands:
            return result

        normalized_command = self._normalize_exact_command(command)
        normalized_exact = {self._normalize_exact_command(item) for item in exact_commands}
        if normalized_command not in normalized_exact:
            return result

        text = str(result or "")
        lower_text = text.lower()
        if "stop_tool_loop" in lower_text:
            return text
        if (
            "exit code:" not in lower_text
            and not lower_text.startswith("error:")
            and "failed after" not in lower_text
            and "fetch failed" not in lower_text
        ):
            return text
        return (
            "STOP_TOOL_LOOP: Exact requested shell command failed. Report this blocker directly "
            "instead of switching to additional exploratory tools.\n"
            f"{text}"
        )


    @staticmethod
    def _mask_secrets(text: str) -> str:
        """Mask sensitive values (keys, tokens, passwords, etc.) in output."""
        from spoon_bot.utils.privacy import mask_secrets
        return mask_secrets(text)


class SafeShellTool(ShellTool):
    """
    A more restrictive shell tool with whitelist mode enabled by default.

    This tool only allows commands from a predefined whitelist,
    providing maximum security at the cost of flexibility.
    """

    def __init__(
        self,
        timeout: int = DEFAULT_SHELL_TIMEOUT,
        max_timeout: int = DEFAULT_SHELL_MAX_TIMEOUT,
        max_output: int = DEFAULT_MAX_OUTPUT,
        working_dir: str | None = None,
        custom_whitelist: set[str] | None = None,
    ):
        """
        Initialize safe shell tool with whitelist mode.

        Args:
            timeout: Default foreground timeout in seconds.
            max_timeout: Maximum per-command timeout override.
            max_output: Maximum output characters.
            working_dir: Default working directory.
            custom_whitelist: Additional commands to whitelist.
        """
        super().__init__(
            timeout=timeout,
            max_timeout=max_timeout,
            max_output=max_output,
            working_dir=working_dir,
            whitelist_mode=True,
            custom_whitelist=custom_whitelist,
            allow_pipes=True,
            strict_mode=True,
            use_shell=True,
        )

    @property
    def name(self) -> str:
        return "safe_shell"

    @property
    def description(self) -> str:
        timeout_min = self.timeout // 60
        return (
            "Execute a shell command from a whitelist of safe commands. "
            f"Foreground budget: {timeout_min}min ({self.timeout}s), "
            "then the command stays running in the background. "
            "Only predefined commands are allowed for security."
        )
