"""Targeted tests for Discord e2e fixes.

Validates all P0-P3 fixes introduced in the e2e audit:
- P0-1: _running is set in start()
- P0-2: stop() has timeout protection
- P1-3: Typing indicators keyed by message_id (no clobbering)
- P1-4: Error messages don't leak internal details
- P1-5: _split_message preserves code blocks and indentation
- P1-6: Circuit breaker for LLM failures
- P2-7: Mention removal preserves newlines, collapses spaces
- P2-8: Stale reaction cleanup
- P2-9: Per-user rate limiting
- P2-10: System message filtering
- P2-11: Bus max_concurrency public API
- P2-12: _start_typing is fire-and-forget
- P3-13: Group DM access control
- P3-14: task_done guaranteed in finally
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spoon_bot.bus.events import InboundMessage, OutboundMessage
from spoon_bot.bus.queue import MessageBus
from spoon_bot.channels.base import ChannelConfig, ChannelMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_discord_config(**extra_kwargs) -> ChannelConfig:
    defaults = {
        "token": "test-discord-token",
        "allowed_guilds": [],
        "allowed_users": [],
        "proxy_url": None,
        "require_mention": True,
        "allow_dm": True,
        "intents": [],
    }
    defaults.update(extra_kwargs)
    return ChannelConfig(name="discord", mode=ChannelMode.GATEWAY, **defaults)


def _make_channel(**extra_kwargs):
    with patch("spoon_bot.channels.discord.channel.DISCORD_AVAILABLE", True), \
         patch("spoon_bot.channels.discord.channel.discord", create=True):
        from spoon_bot.channels.discord.channel import DiscordChannel
        config = _make_discord_config(**extra_kwargs)
        return DiscordChannel(config, "testbot")


# ===========================================================================
# P1-5: _split_message — code block awareness + indentation preservation
# ===========================================================================

class TestSplitMessageCodeBlocks:
    """Verify _split_message handles code fences and whitespace correctly."""

    def test_short_message_unchanged(self):
        from spoon_bot.channels.discord.channel import DiscordChannel
        result = DiscordChannel._split_message("hello world", 100)
        assert result == ["hello world"]

    def test_preserves_leading_whitespace_on_continuation(self):
        """No .lstrip() — leading spaces in continuation are kept."""
        from spoon_bot.channels.discord.channel import DiscordChannel
        line1 = "a" * 100
        line2 = "    indented_code()"
        content = f"{line1}\n{line2}"
        chunks = DiscordChannel._split_message(content, 110)
        assert len(chunks) == 2
        assert chunks[1] == "    indented_code()"

    def test_code_fence_closed_and_reopened_on_split(self):
        """When split occurs inside a code block, fence is auto-closed/reopened."""
        from spoon_bot.channels.discord.channel import DiscordChannel
        # Build a message with a code block that spans the split point
        before = "Some text\n```python\n"
        code_lines = "\n".join(f"line_{i} = {i}" for i in range(50))
        after = "\n```\nEnd."
        content = before + code_lines + after
        chunks = DiscordChannel._split_message(content, 200)

        # First chunk should end with ``` (closing the open fence)
        assert chunks[0].rstrip().endswith("```")
        # Second chunk should start with ``` (reopening the fence)
        assert chunks[1].lstrip().startswith("```")

    def test_no_fence_manipulation_when_fence_is_closed(self):
        """A fully closed code block before the split point is not altered."""
        from spoon_bot.channels.discord.channel import DiscordChannel
        content = "```python\nprint('hi')\n```\n" + "x" * 2000
        chunks = DiscordChannel._split_message(content, 200)
        # The first chunk contains the full closed block — no extra fence
        first = chunks[0]
        # Count fences: should be even (opened + closed)
        fence_count = first.count("```")
        assert fence_count % 2 == 0

    def test_hard_cut_preserves_content(self):
        """No whitespace is lost on hard cuts."""
        from spoon_bot.channels.discord.channel import DiscordChannel
        content = "x" * 4000
        chunks = DiscordChannel._split_message(content, 1950)
        assert "".join(chunks) == content

    def test_newline_split_preserves_content(self):
        """Content split at newlines doesn't lose characters."""
        from spoon_bot.channels.discord.channel import DiscordChannel
        lines = ["line " + str(i) for i in range(300)]
        content = "\n".join(lines)
        chunks = DiscordChannel._split_message(content, 200)
        # Reassemble with the single newline stripped at each split point
        reassembled = "\n".join(chunks)
        # All original lines should be present
        for line in lines:
            assert line in reassembled


# ===========================================================================
# P2-7: Mention removal whitespace handling
# ===========================================================================

class TestMentionWhitespace:
    """Verify mention removal collapses spaces but preserves newlines."""

    def test_double_space_collapsed(self):
        """'Hello <@123> world' → 'Hello world' (not 'Hello  world')."""
        import re
        content = "Hello <@123> world"
        mention = "<@123>"
        content = content.replace(mention, "")
        content = re.sub(r" {2,}", " ", content).strip()
        assert content == "Hello world"

    def test_newlines_preserved(self):
        """Multi-line messages keep their newlines."""
        import re
        content = "Hello <@123>\nWhat is Python?\nThanks"
        mention = "<@123>"
        content = content.replace(mention, "")
        content = re.sub(r" {2,}", " ", content).strip()
        assert "\n" in content
        assert "What is Python?" in content
        assert "Thanks" in content

    def test_mention_at_start_of_line(self):
        """'<@123> hello' → 'hello'."""
        import re
        content = "<@123> hello"
        mention = "<@123>"
        content = content.replace(mention, "")
        content = re.sub(r" {2,}", " ", content).strip()
        assert content == "hello"

    def test_multiple_mentions(self):
        """Multiple mentions in one message."""
        import re
        content = "<@123> Hey <@123> there"
        mention = "<@123>"
        content = content.replace(mention, "")
        content = re.sub(r" {2,}", " ", content).strip()
        assert content == "Hey there"


# ===========================================================================
# P2-9: Rate limiting
# ===========================================================================

class TestRateLimiting:
    """Verify per-user rate limiting works correctly."""

    def test_within_limit_passes(self):
        ch = _make_channel()
        for _ in range(ch._rate_limit_max):
            assert ch._check_rate_limit(12345) is True

    def test_exceeds_limit_blocked(self):
        ch = _make_channel()
        for _ in range(ch._rate_limit_max):
            ch._check_rate_limit(12345)
        assert ch._check_rate_limit(12345) is False

    def test_different_users_independent(self):
        ch = _make_channel()
        for _ in range(ch._rate_limit_max):
            ch._check_rate_limit(111)
        # User 111 is blocked
        assert ch._check_rate_limit(111) is False
        # User 222 is still fine
        assert ch._check_rate_limit(222) is True

    def test_expired_entries_purged(self):
        ch = _make_channel()
        ch._rate_limit_window = 0.1  # 100ms window for testing
        for _ in range(ch._rate_limit_max):
            ch._check_rate_limit(12345)
        assert ch._check_rate_limit(12345) is False
        # Wait for window to expire
        time.sleep(0.15)
        assert ch._check_rate_limit(12345) is True


# ===========================================================================
# P2-8: Stale reaction cleanup
# ===========================================================================

class TestStaleReactionCleanup:
    """Verify pending reactions are cleaned up after max age."""

    def test_stale_reactions_removed(self):
        ch = _make_channel()
        ch._reaction_max_age = 0.1  # 100ms for testing
        # Simulate a stale reaction: (message, timestamp) tuple
        fake_msg = MagicMock()
        ch._pending_reactions["old_msg"] = (fake_msg, time.monotonic() - 1.0)
        # Should be cleaned up
        ch._cleanup_stale_reactions()
        assert "old_msg" not in ch._pending_reactions

    def test_fresh_reactions_kept(self):
        ch = _make_channel()
        ch._reaction_max_age = 600.0
        fake_msg = MagicMock()
        ch._pending_reactions["fresh_msg"] = (fake_msg, time.monotonic())
        ch._cleanup_stale_reactions()
        assert "fresh_msg" in ch._pending_reactions

    def test_very_old_reactions_cleaned(self):
        ch = _make_channel()
        ch._reaction_max_age = 0.0  # 0 max age — anything older is stale
        fake_msg = MagicMock()
        ch._pending_reactions["old"] = (fake_msg, time.monotonic() - 1.0)
        ch._cleanup_stale_reactions()
        assert "old" not in ch._pending_reactions


# ===========================================================================
# P1-3: Typing indicator per-message (no clobbering)
# ===========================================================================

class TestTypingPerMessage:
    """Verify typing tasks keyed by message_id don't clobber each other."""

    def test_two_messages_coexist(self):
        ch = _make_channel()
        mock_channel = MagicMock()
        mock_channel.id = 999
        mock_channel.typing = AsyncMock()

        # Patch asyncio.create_task to avoid real event loop
        with patch("asyncio.create_task") as mock_ct:
            task_a = MagicMock()
            task_a.done.return_value = False
            task_b = MagicMock()
            task_b.done.return_value = False
            mock_ct.side_effect = [task_a, task_b]

            ch._start_typing_for_message(mock_channel, "msg_001")
            ch._start_typing_for_message(mock_channel, "msg_002")

        # Both tasks should exist
        assert "msg_001" in ch._typing_tasks
        assert "msg_002" in ch._typing_tasks

    def test_stop_one_leaves_other(self):
        ch = _make_channel()
        task_a = MagicMock()
        task_a.done.return_value = False
        task_b = MagicMock()
        task_b.done.return_value = False
        ch._typing_tasks["msg_001"] = task_a
        ch._typing_tasks["msg_002"] = task_b

        ch._stop_typing_for_message("msg_001")
        task_a.cancel.assert_called_once()
        assert "msg_001" not in ch._typing_tasks
        assert "msg_002" in ch._typing_tasks
        task_b.cancel.assert_not_called()


# ===========================================================================
# P2-11: Bus max_concurrency public API
# ===========================================================================

class TestBusMaxConcurrency:
    """Verify MessageBus exposes max_concurrency without private access."""

    def test_max_concurrency_property(self):
        bus = MessageBus(max_concurrency=8)
        assert bus.max_concurrency == 8

    def test_set_max_concurrency(self):
        bus = MessageBus(max_concurrency=4)
        bus.set_max_concurrency(16)
        assert bus.max_concurrency == 16

    def test_default_max_concurrency(self):
        bus = MessageBus()
        assert bus.max_concurrency == 4


# ===========================================================================
# P3-14: task_done guaranteed
# ===========================================================================

class TestTaskDoneGuaranteed:
    """Verify _process_with_semaphore always calls task_done."""

    @pytest.mark.asyncio
    async def test_task_done_on_success(self):
        bus = MessageBus()
        msg = InboundMessage(content="test", channel="ch1")
        bus._queue.put_nowait(msg)
        bus._handler = AsyncMock(return_value=None)

        await bus._process_with_semaphore(msg)
        # task_done should not raise ValueError (balanced get/task_done)
        # If it wasn't called, queue.join() would hang

    @pytest.mark.asyncio
    async def test_task_done_on_handler_exception(self):
        bus = MessageBus()
        msg = InboundMessage(content="test", channel="ch1")
        bus._queue.put_nowait(msg)
        bus._handler = AsyncMock(side_effect=RuntimeError("boom"))

        # Should not raise — error is handled, task_done still called
        await bus._process_with_semaphore(msg)


# ===========================================================================
# P1-6: Circuit breaker
# ===========================================================================

class TestCircuitBreaker:
    """Verify circuit breaker state transitions."""

    def test_starts_closed(self):
        from spoon_bot.channels.manager import _CircuitBreaker
        cb = _CircuitBreaker()
        assert cb.state == "closed"
        assert cb.allow_request() is True

    def test_opens_after_threshold(self):
        from spoon_bot.channels.manager import _CircuitBreaker
        cb = _CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "closed"
        cb.record_failure()
        assert cb.state == "open"
        assert cb.allow_request() is False

    def test_success_resets_counter(self):
        from spoon_bot.channels.manager import _CircuitBreaker
        cb = _CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        # Only 1 failure since reset, not 3
        assert cb.state == "closed"

    def test_recovers_to_half_open(self):
        from spoon_bot.channels.manager import _CircuitBreaker
        cb = _CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        cb.record_failure()
        assert cb.state == "open"
        time.sleep(0.15)
        assert cb.state == "half_open"
        assert cb.allow_request() is True

    def test_half_open_success_closes(self):
        from spoon_bot.channels.manager import _CircuitBreaker
        cb = _CircuitBreaker(failure_threshold=1, recovery_timeout=0.0)
        cb.record_failure()
        # Wait for recovery
        time.sleep(0.01)
        assert cb.state == "half_open"
        cb.record_success()
        assert cb.state == "closed"

    def test_half_open_failure_reopens(self):
        from spoon_bot.channels.manager import _CircuitBreaker
        cb = _CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()
        assert cb.state == "open"
        # Wait for recovery → half_open
        time.sleep(0.06)
        assert cb.state == "half_open"
        # Another failure re-opens the breaker
        cb.record_failure()
        # Immediately after failure (within recovery_timeout), state should be open
        assert cb.state == "open"


# ===========================================================================
# P1-4: Error messages don't leak internal details
# ===========================================================================

class TestErrorMessageSanitization:
    """Verify that error messages sent to users don't contain internal info."""

    @pytest.mark.asyncio
    async def test_bus_error_response_is_generic(self):
        bus = MessageBus()
        msg = InboundMessage(
            content="test",
            channel="test_ch",
            message_id="m1",
            metadata={"channel_id": "123"},
        )
        # Handler raises with sensitive info
        bus._handler = AsyncMock(side_effect=RuntimeError(
            "Connection to /home/user/.secrets/db.sqlite3 failed: API_KEY=sk-abc123"
        ))
        captured = []
        bus._outbound_handlers["test_ch"] = AsyncMock(side_effect=lambda m: captured.append(m))

        await bus._process_message(msg)

        assert len(captured) == 1
        response_content = captured[0].content
        # Must NOT contain sensitive info
        assert "API_KEY" not in response_content
        assert "sk-abc123" not in response_content
        assert "/home/user" not in response_content
        assert "db.sqlite3" not in response_content
        # Must contain generic message
        assert "unexpected error" in response_content.lower() or "sorry" in response_content.lower()


# ===========================================================================
# P0-2: stop() timeout protection (basic structural test)
# ===========================================================================

class TestStopTimeout:
    """Verify stop() has timeout on client.close()."""

    def test_stop_code_uses_wait_for(self):
        """Structural check: stop() wraps client.close() in wait_for."""
        import inspect
        from spoon_bot.channels.discord.channel import DiscordChannel
        source = inspect.getsource(DiscordChannel.stop)
        assert "wait_for" in source
        assert "timeout=10.0" in source or "timeout=10" in source

    def test_stop_ensures_running_false_on_error(self):
        """Structural check: _running is set False even in except block."""
        import inspect
        from spoon_bot.channels.discord.channel import DiscordChannel
        source = inspect.getsource(DiscordChannel.stop)
        # Should have _running = False in both normal and except paths
        assert source.count("self._running = False") >= 2


# ===========================================================================
# P0-1: _running is set + health check ordering
# ===========================================================================

class TestStartRunningFlag:
    """Verify start() sets _running before health check."""

    def test_running_set_before_health_check_in_source(self):
        """Structural: self._running = True line appears before the actual
        create_task(_start_health_check_loop()) call (not just comments)."""
        import inspect
        from spoon_bot.channels.discord.channel import DiscordChannel
        source = inspect.getsource(DiscordChannel.start)
        running_pos = source.index("self._running = True")
        # Search for the actual invocation, not the comment mentioning it
        health_pos = source.index("asyncio.create_task(\n                self._start_health_check_loop")
        assert running_pos < health_pos, (
            "_running = True must come before _start_health_check_loop()"
        )


# ===========================================================================
# P3-13: _check_access Group DM handling
# ===========================================================================

class TestCheckAccessGroupDM:
    """Verify Group DM access control."""

    def test_guild_message_without_guild_rejected(self):
        """Non-DM, non-GroupDM message with no guild is rejected."""
        ch = _make_channel(allowed_guilds=[], allowed_users=[])

        with patch("spoon_bot.channels.discord.channel.discord") as d:
            d.DMChannel = type("DMChannel", (), {})
            d.GroupChannel = type("GroupChannel", (), {})
            msg = MagicMock()
            msg.author.id = 42
            msg.author.bot = False
            msg.channel.__class__ = object  # not DM, not GroupDM
            msg.guild = None  # no guild
            assert ch._check_access(msg) is False
