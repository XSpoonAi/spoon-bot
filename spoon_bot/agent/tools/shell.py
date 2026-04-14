"""Shell execution tool with comprehensive security guards and rate limiting."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger

from spoon_bot.agent.tools.base import Tool
from spoon_bot.agent.tools.execution_context import capture_tool_output, get_tool_owner
from spoon_bot.config import DEFAULT_MAX_OUTPUT, DEFAULT_SHELL_MAX_TIMEOUT, DEFAULT_SHELL_TIMEOUT
from spoon_bot.utils.rate_limit import (
    RateLimitConfig,
    get_rate_limiter,
)
from spoon_bot.utils.errors import RateLimitExceeded, ToolExecutionError


class ShellSecurityError(Exception):
    """Raised when a command fails security validation."""
    pass


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


def _append_capped_text(existing: str, addition: str, limit: int) -> str:
    combined = existing + addition
    if len(combined) <= limit:
        return combined
    return combined[-limit:]


_SHELL_BACKGROUND_JOBS: dict[str, _BackgroundShellJob] = {}


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
        """
        self.timeout = timeout
        self.max_timeout = max(timeout, max_timeout)
        self.max_output = max_output
        self.working_dir = working_dir
        self.use_shell = use_shell

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
            f"You may override with timeout (max {max_min}min). "
            "Commands exceeding the budget keep running in the background — "
            "use job_status to monitor and terminate_job to stop if stuck. "
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
                        f"Optional foreground timeout in seconds (max {self.max_timeout}). "
                        f"Defaults to {self.timeout}s. Use a higher value for known "
                        "long-running operations like builds or installations."
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
            },
        }

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

    def _run_sync(
        self,
        command: str,
        cwd: str,
    ) -> tuple[bytes, bytes, int]:
        """Synchronous subprocess execution (called via run_in_executor)."""
        env = os.environ.copy()
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
            result = result[:limit] + f"\n... (truncated, {truncated} more chars)"

        return result

    async def _consume_process_stream(
        self,
        stream: Any,
        append: Any,
    ) -> None:
        if stream is None:
            return
        while True:
            chunk = await asyncio.to_thread(stream.read, 4096)
            if not chunk:
                return
            append(chunk.decode("utf-8", errors="replace"))

    async def _create_process(
        self,
        command: str,
        cwd: str,
    ) -> subprocess.Popen[bytes]:
        env = os.environ.copy()
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
            if bash:
                env["HOME"] = self._windows_home_to_bash(home_path)
                command = self._convert_win_paths_to_posix(command)
                cwd = cwd.replace("\\", "/")
                return subprocess.Popen(
                    [bash, "-c", command],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,
                    cwd=cwd,
                    env=env,
                )
            env["HOME"] = home_path.replace("\\", "/")
            return subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                shell=True,
                cwd=cwd,
                env=env,
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
        )

    @staticmethod
    def _get_process_returncode(process: Any) -> int | None:
        poll = getattr(process, "poll", None)
        if callable(poll):
            try:
                return poll()
            except Exception:
                pass
        return getattr(process, "returncode", None)

    async def _wait_for_process(self, process: Any) -> int | None:
        wait = getattr(process, "wait")
        if asyncio.iscoroutinefunction(wait):
            return await wait()
        return await asyncio.to_thread(wait)

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
            await asyncio.gather(job.stdout_task, job.stderr_task, return_exceptions=True)
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
            "NEXT STEPS — you SHOULD monitor this job:\n"
            f"  1. Check progress: action='job_status', job_id='{job.job_id}'\n"
            f"  2. Read full output: action='job_output', job_id='{job.job_id}'\n"
            f"  3. Stop if stuck:   action='terminate_job', job_id='{job.job_id}'\n"
            "Decide whether the command is making progress or is hung. "
            "If no new output appears after two checks, terminate it."
        )

    async def _handle_background_action(
        self,
        action: str,
        job_id: str | None,
        tail_chars: int,
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
            return f"Error: Background shell job not found: {job_id}"

        await self._refresh_background_job(job)

        if action == "job_status":
            return (
                f"job_id: {job.job_id}\n"
                f"status: {job.status}\n"
                f"cwd: {job.cwd}\n"
                f"command: {job.command}\n"
                f"returncode: {job.returncode if job.returncode is not None else 'running'}\n"
                "Recent output tail:\n"
                f"{self._build_output_result(job.stdout_text, job.stderr_text, job.returncode, max_chars=tail_chars)}"
            )

        if action == "job_output":
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
            capture_tool_output(result, full_result)
            return result

        if action == "terminate_job":
            if self._get_process_returncode(job.process) is None:
                job.process.terminate()
                try:
                    await asyncio.wait_for(self._wait_for_process(job.process), timeout=5.0)
                except asyncio.TimeoutError:
                    job.process.kill()
                    await self._wait_for_process(job.process)
            await self._refresh_background_job(job)
            job.status = "terminated"
            job.finished_at = time.time()
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
            result = (
                f"Terminated background shell job {job.job_id}.\n"
                f"{summary_body}"
            )
            full_result = f"Terminated background shell job {job.job_id}.\n{full_body}"
            capture_tool_output(result, full_result)
            return result

        return f"Error: Unknown action '{action}'"

    async def execute(
        self,
        command: str | None = None,
        working_dir: str | None = None,
        timeout: int | None = None,
        action: str = "execute",
        job_id: str | None = None,
        tail_chars: int = 4000,
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
        if action != "execute":
            return await self._handle_background_action(action, job_id, tail_chars)

        if not command or not str(command).strip():
            return "Error: 'command' is required for action='execute'"

        # Resolve effective timeout: per-command override capped by max_timeout
        effective_timeout = self.timeout
        if timeout is not None:
            effective_timeout = max(1, min(int(timeout), self.max_timeout))

        # Apply rate limiting
        if self._rate_limit_config.enabled:
            wait_time = await self._rate_limiter.wait_and_acquire()
            if wait_time > 0.1:
                logger.info(f"Shell command rate limited, waited {wait_time:.2f}s")

        # Validate command
        is_valid, error_msg = self.validator.validate(command)
        if not is_valid:
            safe_cmd = self.validator.sanitize_for_display(command)
            return f"Security Error: {error_msg}\nCommand: {safe_cmd}"

        cwd = working_dir or self.working_dir or os.getcwd()

        # Verify working directory exists and is accessible
        if not os.path.isdir(cwd):
            return f"Error: Working directory not found: {cwd}"

        try:
            owner_key = get_tool_owner()
            job = await self._start_background_job(command, cwd, owner_key=owner_key)
            try:
                await asyncio.wait_for(self._wait_for_process(job.process), timeout=effective_timeout)
                await self._refresh_background_job(job)
                full_result = self._build_output_result(
                    job.stdout_text,
                    job.stderr_text,
                    job.returncode,
                    truncate=False,
                )
                result = self._build_output_result(job.stdout_text, job.stderr_text, job.returncode)
                capture_tool_output(result, full_result)
                _SHELL_BACKGROUND_JOBS.pop(job.job_id, None)
                self._prune_background_jobs(owner_key=owner_key)
                return result
            except asyncio.TimeoutError:
                await self._refresh_background_job(job)
                self._prune_background_jobs(owner_key=owner_key)
                return self._format_background_job_summary(
                    job,
                    timeout_seconds=effective_timeout,
                    tail_chars=tail_chars,
                )
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
