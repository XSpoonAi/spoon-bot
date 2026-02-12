"""Tests for shell command security validation."""

import pytest
from spoon_bot.agent.tools.shell import CommandValidator, ShellTool, SafeShellTool


class TestCommandValidator:
    """Tests for the CommandValidator class."""

    @pytest.fixture
    def validator(self):
        """Default validator in blocklist mode."""
        return CommandValidator()

    @pytest.fixture
    def whitelist_validator(self):
        """Validator in whitelist mode."""
        return CommandValidator(whitelist_mode=True)

    @pytest.fixture
    def strict_validator(self):
        """Validator in strict mode."""
        return CommandValidator(strict_mode=True)

    # Test dangerous command blocking
    class TestDangerousCommands:
        """Test blocking of dangerous commands."""

        @pytest.mark.parametrize("command", [
            "rm -rf /",
            "rm -rf /*",
            "rm -rf ~",
            "rm -rf ~/*",
            "rm -rf .",
            "rm -fr /",
            "sudo rm -rf /",
            "  rm -rf /  ",  # with whitespace
        ])
        def test_blocks_recursive_deletion(self, command):
            validator = CommandValidator()
            is_valid, error = validator.validate(command)
            assert not is_valid
            assert error is not None
            assert "dangerous" in error.lower() or "blocked" in error.lower()

        @pytest.mark.parametrize("command", [
            "mkfs.ext4 /dev/sda",
            "dd if=/dev/zero of=/dev/sda",
            "dd if=/dev/random of=/dev/sdb",
        ])
        def test_blocks_disk_operations(self, command):
            validator = CommandValidator()
            is_valid, error = validator.validate(command)
            assert not is_valid

        @pytest.mark.parametrize("command", [
            "format c:",
            "format d:",
            "del /f /s /q c:\\",
            "rd /s /q c:\\",
        ])
        def test_blocks_windows_destructive_commands(self, command):
            validator = CommandValidator()
            is_valid, error = validator.validate(command)
            assert not is_valid

        def test_blocks_fork_bomb(self):
            validator = CommandValidator()
            is_valid, error = validator.validate(":(){ :|:& };:")
            assert not is_valid

    # Test injection pattern detection
    class TestInjectionPatterns:
        """Test detection of shell injection patterns."""

        @pytest.mark.parametrize("command", [
            "ls; echo hello",  # Pure injection without dangerous command
            "echo hello; cat /etc/passwd",
            "pwd ;cat /etc/shadow",
        ])
        def test_blocks_semicolon_chaining(self, command):
            validator = CommandValidator()
            is_valid, error = validator.validate(command)
            assert not is_valid
            # Should be blocked by either injection detection or dangerous command
            assert "injection" in error.lower() or "blocked" in error.lower()

        @pytest.mark.parametrize("command", [
            "ls && echo hello",  # Pure injection without dangerous command
            "echo test && cat /etc/passwd",
        ])
        def test_blocks_and_chaining(self, command):
            validator = CommandValidator()
            is_valid, error = validator.validate(command)
            assert not is_valid
            # Should be blocked by either injection detection or dangerous command
            assert "injection" in error.lower() or "blocked" in error.lower()

        @pytest.mark.parametrize("command", [
            "ls || echo hello",  # Pure injection without dangerous command
            "false || cat /etc/passwd",
        ])
        def test_blocks_or_chaining(self, command):
            validator = CommandValidator()
            is_valid, error = validator.validate(command)
            assert not is_valid
            # Should be blocked by either injection detection or dangerous command
            assert "injection" in error.lower() or "blocked" in error.lower()

        @pytest.mark.parametrize("command", [
            "echo $(cat /etc/passwd)",
            "ls $(whoami)",
            "ping $(hostname)",
        ])
        def test_blocks_command_substitution(self, command):
            validator = CommandValidator()
            is_valid, error = validator.validate(command)
            assert not is_valid
            assert "injection" in error.lower()

        @pytest.mark.parametrize("command", [
            "echo `cat /etc/passwd`",
            "ls `whoami`",
        ])
        def test_blocks_backtick_substitution(self, command):
            validator = CommandValidator()
            is_valid, error = validator.validate(command)
            assert not is_valid
            assert "injection" in error.lower()

        @pytest.mark.parametrize("command", [
            "echo test > /etc/passwd",
            "cat file > /etc/shadow",
            "echo key > ~/.ssh/authorized_keys",
            "echo export > ~/.bashrc",
        ])
        def test_blocks_dangerous_redirections(self, command):
            validator = CommandValidator()
            is_valid, error = validator.validate(command)
            assert not is_valid
            assert "injection" in error.lower()

    # Test whitelist mode
    class TestWhitelistMode:
        """Test whitelist mode functionality."""

        def test_allows_whitelisted_commands(self):
            validator = CommandValidator(whitelist_mode=True)

            safe_commands = ["ls -la", "git status", "python --version", "npm install"]
            for cmd in safe_commands:
                is_valid, error = validator.validate(cmd)
                assert is_valid, f"Command '{cmd}' should be allowed: {error}"

        def test_blocks_non_whitelisted_commands(self):
            validator = CommandValidator(whitelist_mode=True)

            is_valid, error = validator.validate("evil_command --danger")
            assert not is_valid
            assert "whitelist" in error.lower()

        def test_custom_whitelist(self):
            validator = CommandValidator(
                whitelist_mode=True,
                custom_whitelist={"mycustomcmd"}
            )

            is_valid, error = validator.validate("mycustomcmd --arg1")
            assert is_valid

        def test_handles_full_paths(self):
            validator = CommandValidator(whitelist_mode=True)

            is_valid, error = validator.validate("/usr/bin/ls -la")
            assert is_valid

    # Test safe commands pass validation
    class TestSafeCommands:
        """Test that legitimate safe commands pass validation."""

        @pytest.mark.parametrize("command", [
            "ls -la",
            "pwd",
            "git status",
            "git log --oneline",
            "python --version",
            "node --version",
            "npm install",
            "cat README.md",
            "grep pattern file.txt",
            "echo 'hello world'",
            "mkdir new_folder",
            "cp file1.txt file2.txt",
            "mv old.txt new.txt",
        ])
        def test_allows_safe_commands(self, command):
            validator = CommandValidator()
            is_valid, error = validator.validate(command)
            assert is_valid, f"Command '{command}' should be allowed: {error}"

    # Test pipe handling
    class TestPipeHandling:
        """Test pipe operator handling."""

        def test_allows_pipes_by_default(self):
            validator = CommandValidator(allow_pipes=True)

            is_valid, error = validator.validate("ls -la | grep pattern")
            assert is_valid

        def test_blocks_pipes_when_disabled(self):
            validator = CommandValidator(allow_pipes=False)

            is_valid, error = validator.validate("ls -la | grep pattern")
            assert not is_valid
            assert "pipe" in error.lower()

    # Test strict mode
    class TestStrictMode:
        """Test strict mode functionality."""

        def test_blocks_sensitive_paths_in_strict_mode(self):
            validator = CommandValidator(strict_mode=True)

            sensitive_commands = [
                "cat /etc/passwd",
                "nano /etc/shadow",
                "vim ~/.ssh/id_rsa",
            ]
            for cmd in sensitive_commands:
                is_valid, error = validator.validate(cmd)
                assert not is_valid, f"Command '{cmd}' should be blocked in strict mode"

        def test_allows_sensitive_paths_in_normal_mode(self):
            validator = CommandValidator(strict_mode=False)

            # In normal mode, reading sensitive files is allowed
            # (dangerous modifications are still blocked by injection patterns)
            is_valid, error = validator.validate("cat /etc/hosts")
            assert is_valid

    # Test custom blocklist
    class TestCustomBlocklist:
        """Test custom blocklist functionality."""

        def test_blocks_custom_patterns(self):
            validator = CommandValidator(
                custom_blocklist={"dangerous_tool", "bad_command"}
            )

            is_valid, error = validator.validate("dangerous_tool --arg")
            assert not is_valid
            assert "blocklist" in error.lower()

    # Test edge cases
    class TestEdgeCases:
        """Test edge cases and boundary conditions."""

        def test_empty_command(self):
            validator = CommandValidator()
            is_valid, error = validator.validate("")
            assert not is_valid
            assert "empty" in error.lower()

        def test_whitespace_only_command(self):
            validator = CommandValidator()
            is_valid, error = validator.validate("   ")
            assert not is_valid

        def test_sanitize_for_display_truncates(self):
            validator = CommandValidator()
            long_cmd = "a" * 200
            sanitized = validator.sanitize_for_display(long_cmd, max_length=50)
            assert len(sanitized) <= 53  # 50 + "..."
            assert sanitized.endswith("...")


class TestShellTool:
    """Tests for ShellTool class."""

    @pytest.fixture
    def shell_tool(self):
        return ShellTool(timeout=5)

    @pytest.mark.asyncio
    async def test_blocks_dangerous_command(self, shell_tool):
        result = await shell_tool.execute("rm -rf /")
        assert "Security Error" in result
        assert "dangerous" in result.lower() or "blocked" in result.lower()

    @pytest.mark.asyncio
    async def test_blocks_injection_attempt(self, shell_tool):
        result = await shell_tool.execute("echo hello; cat /etc/passwd")
        assert "Security Error" in result
        assert "injection" in result.lower()

    @pytest.mark.asyncio
    async def test_allows_safe_command(self, shell_tool):
        result = await shell_tool.execute("echo 'test'")
        # Should execute without security error
        assert "Security Error" not in result

    @pytest.mark.asyncio
    async def test_reports_invalid_working_dir(self, shell_tool):
        result = await shell_tool.execute(
            "ls",
            working_dir="/nonexistent/directory/path"
        )
        assert "Error" in result
        assert "not found" in result.lower()


class TestSafeShellTool:
    """Tests for SafeShellTool class."""

    @pytest.fixture
    def safe_shell(self):
        return SafeShellTool(timeout=5)

    @pytest.mark.asyncio
    async def test_blocks_non_whitelisted_command(self, safe_shell):
        result = await safe_shell.execute("some_unknown_command")
        assert "Security Error" in result
        assert "whitelist" in result.lower()

    @pytest.mark.asyncio
    async def test_allows_whitelisted_command(self, safe_shell):
        result = await safe_shell.execute("echo 'hello'")
        # Should execute without security error
        assert "Security Error" not in result

    def test_has_correct_name(self, safe_shell):
        assert safe_shell.name == "safe_shell"

    def test_description_mentions_whitelist(self, safe_shell):
        assert "whitelist" in safe_shell.description.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
