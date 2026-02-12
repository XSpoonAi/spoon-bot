"""
Unit tests for spoon-bot tools.

These tests use mocks and do not require API keys to run.
Run with: pytest tests/test_tools.py -v
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure spoon_bot is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestShellTool:
    """Tests for the ShellTool."""

    @pytest.fixture
    def shell_tool(self):
        """Create a ShellTool instance for testing."""
        from spoon_bot.agent.tools.shell import ShellTool
        from spoon_bot.utils.rate_limit import RateLimitConfig

        return ShellTool(
            timeout=5,
            max_output=1000,
            rate_limit_config=RateLimitConfig.unlimited(),
        )

    @pytest.mark.asyncio
    async def test_simple_echo_command(self, shell_tool):
        """Test executing a simple echo command."""
        result = await shell_tool.execute("echo hello")
        assert "hello" in result.lower()

    @pytest.mark.asyncio
    async def test_dangerous_command_blocked(self, shell_tool):
        """Test that dangerous commands are blocked."""
        result = await shell_tool.execute("rm -rf /")
        assert "Security Error" in result or "blocked" in result.lower()

    @pytest.mark.asyncio
    async def test_command_injection_blocked(self, shell_tool):
        """Test that command injection is blocked."""
        # Command chaining should be blocked
        result = await shell_tool.execute("echo hello; rm -rf /")
        assert "Security Error" in result or "blocked" in result.lower() or "injection" in result.lower()

    @pytest.mark.asyncio
    async def test_nonexistent_directory(self, shell_tool):
        """Test handling of nonexistent working directory."""
        result = await shell_tool.execute("echo test", working_dir="/nonexistent/path")
        assert "Error" in result or "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_output_truncation(self):
        """Test that long output is truncated."""
        from spoon_bot.agent.tools.shell import ShellTool
        from spoon_bot.utils.rate_limit import RateLimitConfig

        tool = ShellTool(
            max_output=50,
            rate_limit_config=RateLimitConfig.unlimited(),
        )

        # Generate long output
        if sys.platform == "win32":
            result = await tool.execute("echo " + "x" * 100)
        else:
            result = await tool.execute("echo " + "x" * 100)

        # Should be truncated
        assert "truncated" in result.lower() or len(result) <= 100


class TestCommandValidator:
    """Tests for the CommandValidator."""

    @pytest.fixture
    def validator(self):
        """Create a CommandValidator instance."""
        from spoon_bot.agent.tools.shell import CommandValidator
        return CommandValidator()

    def test_valid_command(self, validator):
        """Test that valid commands pass validation."""
        is_valid, error = validator.validate("ls -la")
        assert is_valid is True
        assert error is None

    def test_empty_command(self, validator):
        """Test that empty commands are rejected."""
        is_valid, error = validator.validate("")
        assert is_valid is False
        assert error is not None

    def test_dangerous_rm_rf(self, validator):
        """Test that rm -rf / is blocked."""
        is_valid, error = validator.validate("rm -rf /")
        assert is_valid is False
        assert "dangerous" in error.lower() or "blocked" in error.lower()

    def test_fork_bomb(self, validator):
        """Test that fork bombs are blocked."""
        is_valid, error = validator.validate(":(){ :|:& };:")
        assert is_valid is False

    def test_whitelist_mode(self):
        """Test whitelist mode validation."""
        from spoon_bot.agent.tools.shell import CommandValidator

        validator = CommandValidator(whitelist_mode=True)

        # Whitelisted command should pass
        is_valid, _ = validator.validate("ls -la")
        assert is_valid is True

        # Non-whitelisted command should fail
        is_valid, error = validator.validate("custom_dangerous_cmd")
        assert is_valid is False
        assert "whitelist" in error.lower()


class TestFilesystemTools:
    """Tests for filesystem tools.

    Note: Filesystem tools have workspace security that requires paths to be
    within the configured workspace. Tests set the workspace to the temp directory.
    """

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def read_tool(self, temp_dir):
        """Create a ReadFileTool instance with temp_dir as workspace."""
        from spoon_bot.agent.tools.filesystem import ReadFileTool
        tool = ReadFileTool(workspace=temp_dir)
        return tool

    @pytest.fixture
    def write_tool(self, temp_dir):
        """Create a WriteFileTool instance with temp_dir as workspace."""
        from spoon_bot.agent.tools.filesystem import WriteFileTool
        tool = WriteFileTool(workspace=temp_dir)
        return tool

    @pytest.fixture
    def list_tool(self, temp_dir):
        """Create a ListDirTool instance with temp_dir as workspace."""
        from spoon_bot.agent.tools.filesystem import ListDirTool
        tool = ListDirTool(workspace=temp_dir)
        return tool

    @pytest.mark.asyncio
    async def test_read_existing_file(self, read_tool, temp_dir):
        """Test reading an existing file within workspace."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, World!")

        result = await read_tool.execute(path=str(test_file))
        assert "Hello, World!" in result

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, read_tool, temp_dir):
        """Test reading a nonexistent file."""
        result = await read_tool.execute(path=str(temp_dir / "nonexistent.txt"))
        assert "Error" in result or "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_write_file(self, write_tool, temp_dir):
        """Test writing to a file within workspace."""
        test_file = temp_dir / "output.txt"
        result = await write_tool.execute(
            path=str(test_file),
            content="Test content"
        )

        assert test_file.exists()
        assert test_file.read_text() == "Test content"

    @pytest.mark.asyncio
    async def test_list_directory(self, list_tool, temp_dir):
        """Test listing directory contents within workspace."""
        # Create some test files
        (temp_dir / "file1.txt").write_text("content1")
        (temp_dir / "file2.txt").write_text("content2")

        result = await list_tool.execute(path=str(temp_dir))

        assert "file1.txt" in result
        assert "file2.txt" in result

    @pytest.mark.asyncio
    async def test_list_nonexistent_directory(self, list_tool, temp_dir):
        """Test listing a nonexistent directory within workspace."""
        # Path must be within workspace but nonexistent
        result = await list_tool.execute(path=str(temp_dir / "nonexistent_dir"))
        assert "Error" in result or "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_read_outside_workspace(self):
        """Test that reading outside workspace is blocked."""
        from spoon_bot.agent.tools.filesystem import ReadFileTool

        # Create tool with restricted workspace
        with tempfile.TemporaryDirectory() as workspace:
            tool = ReadFileTool(workspace=workspace)

            # Try to read a file outside workspace
            result = await tool.execute(path="/etc/passwd")
            assert "Security Error" in result or "outside" in result.lower()


class TestToolRegistry:
    """Tests for the ToolRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a ToolRegistry instance with validation disabled for mocks."""
        from spoon_bot.agent.tools.registry import ToolRegistry
        # Disable parameter validation for mock tool tests
        return ToolRegistry(validate_params=False)

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool."""
        tool = MagicMock()
        tool.name = "test_tool"
        tool.description = "A test tool"
        tool.parameters = {"type": "object", "properties": {}}
        tool.execute = AsyncMock(return_value="test result")
        tool.validate_parameters = MagicMock(return_value=[])  # No validation errors
        tool.to_schema.return_value = {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {"type": "object", "properties": {}},
            }
        }
        return tool

    def test_register_tool(self, registry, mock_tool):
        """Test registering a tool."""
        registry.register(mock_tool)
        assert "test_tool" in registry
        assert len(registry) == 1

    def test_get_tool(self, registry, mock_tool):
        """Test getting a registered tool."""
        registry.register(mock_tool)
        retrieved = registry.get("test_tool")
        assert retrieved is mock_tool

    def test_get_nonexistent_tool(self, registry):
        """Test getting a tool that doesn't exist."""
        result = registry.get("nonexistent")
        assert result is None

    def test_unregister_tool(self, registry, mock_tool):
        """Test unregistering a tool."""
        registry.register(mock_tool)
        result = registry.unregister("test_tool")
        assert result is True
        assert "test_tool" not in registry

    def test_list_tools(self, registry, mock_tool):
        """Test listing registered tools."""
        registry.register(mock_tool)
        tools = registry.list_tools()
        assert "test_tool" in tools

    @pytest.mark.asyncio
    async def test_execute_tool(self, registry, mock_tool):
        """Test executing a registered tool."""
        registry.register(mock_tool)
        result = await registry.execute("test_tool", {"arg": "value"})
        assert result == "test result"
        mock_tool.execute.assert_called_once_with(arg="value")

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, registry):
        """Test executing an unknown tool."""
        result = await registry.execute("unknown_tool", {})
        assert "Error" in result or "Unknown" in result

    def test_get_definitions(self, registry, mock_tool):
        """Test getting tool definitions."""
        registry.register(mock_tool)
        definitions = registry.get_definitions()
        assert len(definitions) == 1
        assert definitions[0]["type"] == "function"


class TestRateLimiting:
    """Tests for rate limiting utilities."""

    @pytest.fixture
    def rate_config(self):
        """Create a rate limit config for testing."""
        from spoon_bot.utils.rate_limit import RateLimitConfig
        return RateLimitConfig(
            requests_per_second=10.0,
            requests_per_minute=60.0,
            burst_size=5,
            enabled=True,
        )

    @pytest.mark.asyncio
    async def test_token_bucket_acquire(self, rate_config):
        """Test token bucket limiter acquire."""
        from spoon_bot.utils.rate_limit import TokenBucketLimiter

        limiter = TokenBucketLimiter.from_config(rate_config)

        # Should be able to acquire immediately (within burst capacity)
        result = await limiter.acquire()
        assert result is True

    @pytest.mark.asyncio
    async def test_token_bucket_burst(self, rate_config):
        """Test token bucket burst behavior."""
        from spoon_bot.utils.rate_limit import TokenBucketLimiter

        limiter = TokenBucketLimiter.from_config(rate_config)

        # Should be able to acquire burst_size tokens immediately
        for _ in range(rate_config.burst_size):
            result = await limiter.acquire()
            assert result is True

    @pytest.mark.asyncio
    async def test_sliding_window_acquire(self, rate_config):
        """Test sliding window limiter acquire."""
        from spoon_bot.utils.rate_limit import SlidingWindowLimiter

        limiter = SlidingWindowLimiter.from_config(rate_config)

        # Should be able to acquire immediately
        result = await limiter.acquire()
        assert result is True

    def test_rate_config_presets(self):
        """Test rate limit config presets."""
        from spoon_bot.utils.rate_limit import RateLimitConfig

        llm_config = RateLimitConfig.for_llm_api()
        assert llm_config.enabled is True
        assert llm_config.requests_per_second <= 5  # Conservative limit

        shell_config = RateLimitConfig.for_shell()
        assert shell_config.enabled is True

        unlimited = RateLimitConfig.unlimited()
        assert unlimited.enabled is False

    @pytest.mark.asyncio
    async def test_rate_limiter_reset(self, rate_config):
        """Test rate limiter reset."""
        from spoon_bot.utils.rate_limit import TokenBucketLimiter

        limiter = TokenBucketLimiter.from_config(rate_config)

        # Exhaust some tokens
        for _ in range(3):
            await limiter.acquire()

        # Reset
        limiter.reset()

        # Should have full capacity again
        assert limiter.tokens == rate_config.burst_size


class TestErrorHandling:
    """Tests for error handling utilities."""

    def test_format_user_error_spoon_error(self):
        """Test formatting SpoonBotError."""
        from spoon_bot.utils.errors import SpoonBotError, format_user_error

        error = SpoonBotError(
            "Technical error",
            user_message="Something went wrong. Please try again."
        )
        result = format_user_error(error)
        assert result == "Something went wrong. Please try again."

    def test_format_user_error_config_error(self):
        """Test formatting ConfigurationError."""
        from spoon_bot.utils.errors import ConfigurationError, format_user_error

        error = ConfigurationError(
            "Missing key",
            user_message="API key not configured."
        )
        result = format_user_error(error, include_type=True)
        assert "Configuration Error" in result
        assert "API key not configured" in result

    def test_format_user_error_api_error(self):
        """Test formatting APIError."""
        from spoon_bot.utils.errors import APIError, format_user_error

        error = APIError(
            "HTTP 401",
            status_code=401,
            provider="Anthropic",
        )
        result = format_user_error(error)
        assert "authentication" in result.lower() or "api key" in result.lower()

    def test_format_user_error_rate_limit(self):
        """Test formatting RateLimitExceeded."""
        from spoon_bot.utils.errors import RateLimitExceeded, format_user_error

        error = RateLimitExceeded(
            resource="API",
            limit=60,
            window=60.0,
            retry_after=5.0,
        )
        result = format_user_error(error)
        assert "too many requests" in result.lower() or "wait" in result.lower()

    def test_format_user_error_generic(self):
        """Test formatting generic exceptions."""
        from spoon_bot.utils.errors import format_user_error

        error = ValueError("Invalid input")
        result = format_user_error(error)
        # Should return a user-friendly message, not the raw error
        assert "Invalid" in result or "value" in result.lower()

    def test_error_suggestions(self):
        """Test getting error suggestions."""
        from spoon_bot.utils.errors import get_error_suggestions

        # API key error
        error = ValueError("ANTHROPIC_API_KEY not set")
        suggestions = get_error_suggestions(error)
        assert len(suggestions) > 0
        assert any("key" in s.lower() for s in suggestions)

        # Connection error
        error = ConnectionError("Connection refused")
        suggestions = get_error_suggestions(error)
        assert any("connection" in s.lower() or "internet" in s.lower() for s in suggestions)


class TestToolBase:
    """Tests for the Tool base class."""

    def test_tool_to_schema(self):
        """Test converting a tool to OpenAI schema format."""
        from spoon_bot.agent.tools.base import Tool

        class TestTool(Tool):
            @property
            def name(self) -> str:
                return "test_tool"

            @property
            def description(self) -> str:
                return "A test tool for testing"

            @property
            def parameters(self) -> dict:
                return {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"},
                    },
                    "required": ["input"],
                }

            async def execute(self, **kwargs) -> str:
                return f"Result: {kwargs.get('input')}"

        tool = TestTool()
        schema = tool.to_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"
        assert schema["function"]["description"] == "A test tool for testing"
        assert "properties" in schema["function"]["parameters"]

    def test_tool_repr(self):
        """Test tool string representation."""
        from spoon_bot.agent.tools.base import Tool

        class TestTool(Tool):
            @property
            def name(self):
                return "my_tool"

            @property
            def description(self):
                return "desc"

            @property
            def parameters(self):
                return {}

            async def execute(self, **kwargs):
                return ""

        tool = TestTool()
        assert "my_tool" in repr(tool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
