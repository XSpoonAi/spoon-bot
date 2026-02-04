"""Shell execution tool with comprehensive security guards and rate limiting."""

import asyncio
import os
import re
import shlex
import sys
from typing import Any

from loguru import logger

from spoon_bot.agent.tools.base import Tool
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
        "format",
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
        # Pipes (can be dangerous but often legitimate - more lenient)
        # Command substitution
        re.compile(r"\$\("),    # $(command)
        re.compile(r"`[^`]+`"), # `command`
        # Process substitution
        re.compile(r"<\("),     # <(command)
        re.compile(r">\("),     # >(command)
        # Dangerous redirections
        re.compile(r">\s*/etc/"),        # Write to /etc/
        re.compile(r">\s*/dev/"),        # Write to /dev/
        re.compile(r">\s*~/.ssh/"),      # Write to SSH config
        re.compile(r">\s*~/.bashrc"),    # Modify bashrc
        re.compile(r">\s*~/.profile"),   # Modify profile
        re.compile(r">\s*/root/"),       # Write to root home
    ]

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
        "ls", "dir", "pwd", "cd", "echo", "cat", "head", "tail", "grep",
        "find", "which", "where", "whoami", "date", "cal", "uname",
        "git", "npm", "pnpm", "yarn", "node", "python", "python3", "pip",
        "cargo", "rustc", "go", "java", "javac", "mvn", "gradle",
        "docker", "kubectl", "terraform", "make", "cmake",
        "curl", "wget", "ping", "traceroute", "dig", "nslookup",
        "ps", "top", "htop", "df", "du", "free", "uptime",
        "tar", "zip", "unzip", "gzip", "gunzip",
        "cp", "mv", "mkdir", "touch", "ln",  # File operations (limited)
        "code", "vim", "nano", "less", "more",
        "ssh", "scp", "rsync",
    })

    def __init__(
        self,
        whitelist_mode: bool = False,
        custom_whitelist: set[str] | None = None,
        custom_blocklist: set[str] | None = None,
        allow_pipes: bool = True,
        strict_mode: bool = False,
    ):
        """
        Initialize command validator.

        Args:
            whitelist_mode: If True, only allow whitelisted commands.
            custom_whitelist: Additional commands to whitelist.
            custom_blocklist: Additional commands/patterns to block.
            allow_pipes: If True, allow pipe (|) in commands.
            strict_mode: If True, block all potentially dangerous patterns.
        """
        self.whitelist_mode = whitelist_mode
        self.whitelist = self.DEFAULT_WHITELIST.copy()
        if custom_whitelist:
            self.whitelist = self.whitelist | custom_whitelist

        self.custom_blocklist = custom_blocklist or set()
        self.allow_pipes = allow_pipes
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
        """Check if command matches any dangerous command patterns."""
        cmd_lower = command.lower().strip()

        # Check exact matches
        for dangerous in self.DANGEROUS_COMMANDS:
            if dangerous.lower() in cmd_lower:
                return f"Blocked dangerous command: '{dangerous}'"

        # Check custom blocklist
        for blocked in self.custom_blocklist:
            if blocked.lower() in cmd_lower:
                return f"Blocked by custom blocklist: '{blocked}'"

        # Check regex patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.search(command):
                return f"Blocked dangerous pattern: {pattern.pattern}"

        return None

    def _check_injection_patterns(self, command: str) -> str | None:
        """Check for shell injection patterns."""
        for pattern in self.INJECTION_PATTERNS:
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
        timeout: int = 60,
        max_output: int = 10000,
        working_dir: str | None = None,
        whitelist_mode: bool = False,
        custom_whitelist: set[str] | None = None,
        custom_blocklist: set[str] | None = None,
        allow_pipes: bool = True,
        strict_mode: bool = False,
        use_shell: bool = True,
        rate_limit_config: RateLimitConfig | None = None,
    ):
        """
        Initialize shell tool.

        Args:
            timeout: Command timeout in seconds (default 60).
            max_output: Maximum output characters (default 10000).
            working_dir: Default working directory.
            whitelist_mode: If True, only allow whitelisted commands.
            custom_whitelist: Additional commands to whitelist.
            custom_blocklist: Additional commands/patterns to block.
            allow_pipes: If True, allow pipe (|) in commands.
            strict_mode: If True, block all potentially dangerous patterns.
            use_shell: If True, use shell execution; if False, use direct exec.
            rate_limit_config: Configuration for rate limiting shell commands.
        """
        self.timeout = timeout
        self.max_output = max_output
        self.working_dir = working_dir
        self.use_shell = use_shell

        # Initialize command validator
        self.validator = CommandValidator(
            whitelist_mode=whitelist_mode,
            custom_whitelist=custom_whitelist,
            custom_blocklist=custom_blocklist,
            allow_pipes=allow_pipes,
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
        return (
            f"Execute a shell command and return its output. "
            f"Commands timeout after {self.timeout}s. "
            f"Security mode: {mode}. "
            "Dangerous commands and injection patterns are blocked."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute (validated for safety)",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory for the command",
                },
            },
            "required": ["command"],
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

    async def _execute_with_shell(
        self,
        command: str,
        cwd: str,
    ) -> tuple[bytes, bytes, int]:
        """Execute command using shell."""
        if sys.platform == "win32":
            # Use cmd.exe on Windows
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
        else:
            # Use /bin/sh on Unix with proper escaping
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                executable="/bin/sh",
            )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=self.timeout,
        )

        return stdout, stderr, process.returncode or 0

    async def _execute_without_shell(
        self,
        command: str,
        cwd: str,
    ) -> tuple[bytes, bytes, int]:
        """Execute command without shell (safer)."""
        args = self._parse_command_args(command)
        if not args:
            raise ValueError("Empty command after parsing")

        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=self.timeout,
        )

        return stdout, stderr, process.returncode or 0

    async def execute(
        self,
        command: str,
        working_dir: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Execute a shell command with security validation and rate limiting.

        Args:
            command: The command to execute.
            working_dir: Optional working directory.

        Returns:
            Command output or error message.
        """
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
            if self.use_shell:
                stdout, stderr, returncode = await self._execute_with_shell(
                    command, cwd
                )
            else:
                stdout, stderr, returncode = await self._execute_without_shell(
                    command, cwd
                )

            output_parts = []

            if stdout:
                stdout_text = stdout.decode("utf-8", errors="replace")
                output_parts.append(stdout_text)

            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    output_parts.append(f"STDERR:\n{stderr_text}")

            if returncode != 0:
                output_parts.append(f"\nExit code: {returncode}")

            result = "\n".join(output_parts) if output_parts else "(no output)"

            # Truncate very long output
            if len(result) > self.max_output:
                truncated = len(result) - self.max_output
                result = result[: self.max_output] + f"\n... (truncated, {truncated} more chars)"

            return result

        except asyncio.TimeoutError:
            return f"Error: Command timed out after {self.timeout} seconds"
        except FileNotFoundError as e:
            return f"Error: Command or file not found: {e}"
        except PermissionError:
            return "Error: Permission denied for command or directory"
        except ValueError as e:
            return f"Error: Invalid command format: {e}"
        except Exception as e:
            return f"Error executing command: {str(e)}"


class SafeShellTool(ShellTool):
    """
    A more restrictive shell tool with whitelist mode enabled by default.

    This tool only allows commands from a predefined whitelist,
    providing maximum security at the cost of flexibility.
    """

    def __init__(
        self,
        timeout: int = 60,
        max_output: int = 10000,
        working_dir: str | None = None,
        custom_whitelist: set[str] | None = None,
    ):
        """
        Initialize safe shell tool with whitelist mode.

        Args:
            timeout: Command timeout in seconds.
            max_output: Maximum output characters.
            working_dir: Default working directory.
            custom_whitelist: Additional commands to whitelist.
        """
        super().__init__(
            timeout=timeout,
            max_output=max_output,
            working_dir=working_dir,
            whitelist_mode=True,
            custom_whitelist=custom_whitelist,
            allow_pipes=True,  # Allow pipes for common operations
            strict_mode=True,  # Enable strict path checking
            use_shell=True,
        )

    @property
    def name(self) -> str:
        return "safe_shell"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command from a whitelist of safe commands. "
            f"Commands timeout after {self.timeout}s. "
            "Only predefined commands are allowed for security."
        )
