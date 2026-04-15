"""Path validation utilities to prevent path traversal attacks.

This module provides security utilities to ensure filesystem operations
stay within allowed workspace boundaries and cannot access sensitive paths.
"""

import os
import platform
from pathlib import Path
from typing import NamedTuple


class PathValidationResult(NamedTuple):
    """Result of path validation."""
    valid: bool
    resolved_path: Path | None
    error: str | None


class PathValidator:
    """
    Validates file paths to prevent path traversal attacks.

    Ensures that:
    1. All paths resolve within the allowed workspace
    2. Symlinks don't escape the workspace
    3. Sensitive system paths are blocked
    4. Path traversal patterns (../) are safely handled
    """

    # Sensitive paths that should never be accessed (Unix)
    UNIX_BLOCKLIST = [
        "/etc/passwd",
        "/etc/shadow",
        "/etc/sudoers",
        "/etc/ssh",
        "/root",
        "/.ssh",
        "/id_rsa",
        "/id_ed25519",
        "/authorized_keys",
        "/known_hosts",
        "/.gnupg",
        "/.aws/credentials",
        "/.aws/config",
        "/.config/gcloud",
        "/.kube/config",
        "/.docker/config.json",
        "/private/etc",  # macOS
        "/.bash_history",
        "/.zsh_history",
        "/.netrc",
        "/.pgpass",
        "/.mysql_history",
        "/.env",
        "/secrets",
        "/credentials",
    ]

    # Sensitive paths for Windows
    WINDOWS_BLOCKLIST = [
        "\\windows\\system32\\config\\sam",
        "\\windows\\system32\\config\\system",
        "\\windows\\system32\\config\\security",
        "\\users\\administrator",
        "\\.ssh",
        "\\id_rsa",
        "\\id_ed25519",
        "\\authorized_keys",
        "\\.aws\\credentials",
        "\\.aws\\config",
        "\\.azure",
        "\\.kube\\config",
        "\\.docker\\config.json",
        "\\appdata\\roaming\\microsoft\\credentials",
        "\\appdata\\local\\microsoft\\credentials",
        "\\ntds.dit",
        "\\.env",
        "\\secrets",
        "\\credentials",
        "\\.netrc",
        "\\pgpass.conf",
    ]

    # Patterns in filenames that suggest sensitive content
    SENSITIVE_PATTERNS = [
        ".env",
        ".pem",
        ".key",
        ".p12",
        ".pfx",
        "credentials",
        "secrets",
        "password",
        "private_key",
        "privatekey",
        "secret_key",
        "api_key",
        "token",
        ".htpasswd",
    ]

    def __init__(
        self,
        workspace: Path | str | None = None,
        allow_outside_workspace: bool = False,
        strict_mode: bool = True,
        additional_read_paths: list[Path | str] | None = None,
    ):
        """
        Initialize the path validator.

        Args:
            workspace: The allowed workspace directory. If None, uses CWD.
            allow_outside_workspace: If True, allows access outside workspace
                                     (NOT recommended for production).
            strict_mode: If True, also checks for sensitive filename patterns.
            additional_read_paths: Extra directories allowed for **read** operations
                                   (e.g. the spoon-bot home dir so skill files are
                                   accessible even if they sit outside the workspace).
        """
        if workspace is None:
            self._workspace = Path.cwd().resolve()
        else:
            self._workspace = Path(workspace).expanduser().resolve()

        self._allow_outside = allow_outside_workspace
        self._strict_mode = strict_mode
        self._is_windows = platform.system() == "Windows"
        self._additional_read_paths: list[Path] = []
        if additional_read_paths:
            self._additional_read_paths = [
                Path(p).expanduser().resolve() for p in additional_read_paths
            ]

        # Build the blocklist based on platform
        self._blocklist = self._build_blocklist()

    @property
    def workspace(self) -> Path:
        """Get the workspace path."""
        return self._workspace

    def _build_blocklist(self) -> set[str]:
        """Build the set of blocked path patterns for the current platform."""
        blocklist = set()

        if self._is_windows:
            # Add Windows-specific patterns (case-insensitive)
            for pattern in self.WINDOWS_BLOCKLIST:
                blocklist.add(pattern.lower())
        else:
            # Add Unix patterns
            for pattern in self.UNIX_BLOCKLIST:
                blocklist.add(pattern.lower())

        return blocklist

    def _normalize_path_for_check(self, path: Path) -> str:
        """Normalize a path for blocklist checking."""
        path_str = str(path)
        if self._is_windows:
            # Windows: case-insensitive, normalize separators
            return path_str.lower().replace("/", "\\")
        else:
            return path_str.lower()

    def _is_blocked_path(self, resolved_path: Path) -> tuple[bool, str | None]:
        """
        Check if a path matches any blocked patterns.

        Args:
            resolved_path: The resolved absolute path to check.

        Returns:
            Tuple of (is_blocked, reason) where reason explains why it's blocked.
        """
        normalized = self._normalize_path_for_check(resolved_path)
        is_within_workspace = self._is_within_workspace(resolved_path)

        # Check against blocklist patterns
        # Only block patterns outside workspace (allows project-specific .env, etc.)
        for pattern in self._blocklist:
            if pattern in normalized:
                # Allow certain patterns within workspace (project-specific files)
                workspace_allowed_patterns = ['/.env', '\\.env', '/secrets', '\\secrets']
                if is_within_workspace and any(p in pattern for p in workspace_allowed_patterns):
                    continue  # Allow project-specific sensitive files within workspace
                return True, f"Access to sensitive path pattern '{pattern}' is blocked"

        # In strict mode, check filename patterns for files OUTSIDE workspace
        # (paths within additional_read_paths are treated as workspace-like)
        if (
            self._strict_mode
            and not is_within_workspace
            and not self._is_within_additional_read_paths(resolved_path)
        ):
            filename = resolved_path.name.lower()
            for pattern in self.SENSITIVE_PATTERNS:
                if pattern.lower() in filename:
                    return True, f"Access to files matching pattern '{pattern}' outside workspace is blocked"

        return False, None

    def _is_within_workspace(self, resolved_path: Path) -> bool:
        """
        Check if a resolved path is within the workspace.

        Args:
            resolved_path: The resolved absolute path to check.

        Returns:
            True if the path is within the workspace.
        """
        try:
            resolved_path.relative_to(self._workspace)
            return True
        except ValueError:
            return False

    def _is_within_additional_read_paths(self, resolved_path: Path) -> bool:
        """Check if *resolved_path* falls under any additional read path."""
        for allowed in self._additional_read_paths:
            try:
                resolved_path.relative_to(allowed)
                return True
            except ValueError:
                continue
        return False

    def _check_symlink_target(self, path: Path) -> tuple[bool, str | None]:
        """
        Check if a symlink's target escapes the workspace.

        Args:
            path: The path to check (may or may not be a symlink).

        Returns:
            Tuple of (is_safe, error_message).
        """
        if not path.is_symlink():
            return True, None

        try:
            # Get the target of the symlink
            target = path.resolve(strict=True)

            # Check if target is within workspace
            if not self._is_within_workspace(target):
                return False, f"Symlink target '{target}' escapes workspace boundary"

            # Recursively check if target is also a symlink
            if target.is_symlink():
                return self._check_symlink_target(target)

            return True, None

        except OSError as e:
            # Broken symlink or permission error
            return False, f"Cannot resolve symlink: {str(e)}"

    @staticmethod
    def _normalize_posix_on_windows(path_str: str) -> str:
        """Convert Git Bash POSIX paths (/c/Users/...) to Windows (C:\\Users\\...) on Windows."""
        if platform.system() != "Windows":
            return path_str
        import re
        m = re.match(r'^/([a-zA-Z])(/.*)', path_str)
        if m:
            return f"{m.group(1).upper()}:{m.group(2)}"
        return path_str

    def validate_read_path(self, path: str | Path) -> PathValidationResult:
        """
        Validate a path for read operations.

        Args:
            path: The path to validate.

        Returns:
            PathValidationResult with validation status.
        """
        try:
            path = self._normalize_posix_on_windows(str(path))
            raw_path = Path(path).expanduser()

            # Resolve relative paths against workspace first (not CWD)
            if not raw_path.is_absolute() and self._workspace:
                ws_candidate = (self._workspace / raw_path)
                if ws_candidate.exists():
                    raw_path = ws_candidate

            # For read operations, try to resolve strictly (file must exist)
            try:
                resolved = raw_path.resolve(strict=True)
            except FileNotFoundError:
                resolved = raw_path.resolve(strict=False)
                if (
                    not self._allow_outside
                    and not self._is_within_workspace(resolved)
                    and not self._is_within_additional_read_paths(resolved)
                ):
                    return PathValidationResult(
                        valid=False,
                        resolved_path=None,
                        error=f"Path '{path}' resolves outside workspace boundary",
                    )
                blocked, reason = self._is_blocked_path(resolved)
                if blocked:
                    return PathValidationResult(valid=False, resolved_path=None, error=reason)
                return PathValidationResult(valid=True, resolved_path=resolved, error=None)

            # Check workspace boundary (workspace + additional read paths)
            if (
                not self._allow_outside
                and not self._is_within_workspace(resolved)
                and not self._is_within_additional_read_paths(resolved)
            ):
                return PathValidationResult(
                    valid=False,
                    resolved_path=None,
                    error=f"Path '{path}' resolves outside workspace boundary. "
                          f"Workspace: {self._workspace}",
                )

            # Check blocklist
            blocked, reason = self._is_blocked_path(resolved)
            if blocked:
                return PathValidationResult(valid=False, resolved_path=None, error=reason)

            # Check symlinks
            symlink_safe, symlink_error = self._check_symlink_target(raw_path)
            if not symlink_safe:
                return PathValidationResult(
                    valid=False,
                    resolved_path=None,
                    error=symlink_error,
                )

            return PathValidationResult(valid=True, resolved_path=resolved, error=None)

        except Exception as e:
            return PathValidationResult(
                valid=False,
                resolved_path=None,
                error=f"Path validation error: {str(e)}",
            )

    def validate_write_path(self, path: str | Path) -> PathValidationResult:
        """
        Validate a path for write operations.

        Write operations have stricter validation:
        - The path must be within workspace (never allow outside)
        - Parent directory check for path creation

        Args:
            path: The path to validate.

        Returns:
            PathValidationResult with validation status.
        """
        try:
            path = self._normalize_posix_on_windows(str(path))
            raw_path = Path(path).expanduser()
            # Resolve relative paths against workspace first
            if not raw_path.is_absolute() and self._workspace:
                raw_path = self._workspace / raw_path
            resolved = raw_path.resolve(strict=False)

            # For write operations, ALWAYS enforce workspace boundary
            if not self._is_within_workspace(resolved):
                return PathValidationResult(
                    valid=False,
                    resolved_path=None,
                    error=f"Write path '{path}' resolves outside workspace boundary. "
                          f"Workspace: {self._workspace}",
                )

            # Check blocklist
            blocked, reason = self._is_blocked_path(resolved)
            if blocked:
                return PathValidationResult(valid=False, resolved_path=None, error=reason)

            # Check if parent exists and is within workspace
            parent = resolved.parent
            if parent.exists():
                if not self._is_within_workspace(parent.resolve()):
                    return PathValidationResult(
                        valid=False,
                        resolved_path=None,
                        error=f"Parent directory escapes workspace boundary",
                    )

                # Check if parent is a symlink escaping workspace
                if parent.is_symlink():
                    symlink_safe, symlink_error = self._check_symlink_target(parent)
                    if not symlink_safe:
                        return PathValidationResult(
                            valid=False,
                            resolved_path=None,
                            error=f"Parent directory symlink issue: {symlink_error}",
                        )

            return PathValidationResult(valid=True, resolved_path=resolved, error=None)

        except Exception as e:
            return PathValidationResult(
                valid=False,
                resolved_path=None,
                error=f"Path validation error: {str(e)}",
            )

    def validate_directory_path(self, path: str | Path) -> PathValidationResult:
        """
        Validate a path for directory listing operations.

        Args:
            path: The directory path to validate.

        Returns:
            PathValidationResult with validation status.
        """
        # Directory validation follows read path rules
        result = self.validate_read_path(path)

        if result.valid and result.resolved_path:
            # Additional check: must be a directory (if exists)
            if result.resolved_path.exists() and not result.resolved_path.is_dir():
                return PathValidationResult(
                    valid=False,
                    resolved_path=None,
                    error=f"Path '{path}' is not a directory",
                )

        return result


# Global default validator (can be overridden)
_default_validator: PathValidator | None = None


def get_default_validator() -> PathValidator:
    """Get the default path validator instance."""
    global _default_validator
    if _default_validator is None:
        _default_validator = PathValidator()
    return _default_validator


def set_default_validator(validator: PathValidator) -> None:
    """Set the default path validator instance."""
    global _default_validator
    _default_validator = validator


def validate_read_path(path: str | Path, workspace: Path | None = None) -> PathValidationResult:
    """
    Convenience function to validate a read path.

    Args:
        path: The path to validate.
        workspace: Optional workspace override.

    Returns:
        PathValidationResult with validation status.
    """
    if workspace is not None:
        validator = PathValidator(workspace=workspace)
    else:
        validator = get_default_validator()
    return validator.validate_read_path(path)


def validate_write_path(path: str | Path, workspace: Path | None = None) -> PathValidationResult:
    """
    Convenience function to validate a write path.

    Args:
        path: The path to validate.
        workspace: Optional workspace override.

    Returns:
        PathValidationResult with validation status.
    """
    if workspace is not None:
        validator = PathValidator(workspace=workspace)
    else:
        validator = get_default_validator()
    return validator.validate_write_path(path)


def validate_directory_path(path: str | Path, workspace: Path | None = None) -> PathValidationResult:
    """
    Convenience function to validate a directory path.

    Args:
        path: The directory path to validate.
        workspace: Optional workspace override.

    Returns:
        PathValidationResult with validation status.
    """
    if workspace is not None:
        validator = PathValidator(workspace=workspace)
    else:
        validator = get_default_validator()
    return validator.validate_directory_path(path)
