"""Slash command handlers for Telegram bot."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from telegram import Update
from telegram.ext import CommandHandler

from spoon_bot.channels.telegram.constants import (
    BOT_COMMANDS,
    REASONING_LEVELS,
    THINK_LEVELS,
    normalize_think_level,
)
from spoon_bot.channels.telegram.keyboards import (
    build_commands_keyboard,
    build_confirm_keyboard,
    build_model_keyboard,
    build_provider_keyboard,
    build_reasoning_keyboard,
    build_skill_keyboard,
    build_think_keyboard,
    build_verbose_keyboard,
)

if TYPE_CHECKING:
    from spoon_bot.agent.loop import AgentLoop
    from spoon_bot.channels.telegram.channel import TelegramChannel


class CommandHandlers:
    """Manages all slash command handlers for the Telegram bot.

    Initialized with a reference to the parent TelegramChannel so handlers
    can access user state, agent, and send messages through the bus.
    """

    def __init__(self, channel: TelegramChannel) -> None:
        self._channel = channel
        self._agent: AgentLoop | None = None

    def set_agent(self, agent: AgentLoop) -> None:
        """Set agent reference for commands that need it."""
        self._agent = agent

    def register(self, app: Any) -> None:
        """Register all command handlers on the Application.

        Args:
            app: telegram.ext.Application instance.
        """
        commands = {
            "start": self.handle_start,
            "help": self.handle_help,
            "commands": self.handle_commands,
            "whoami": self.handle_whoami,
            "model": self.handle_model,
            "models": self.handle_models,
            "think": self.handle_think,
            "verbose": self.handle_verbose,
            "reasoning": self.handle_reasoning,
            "skill": self.handle_skill,
            "tools": self.handle_tools,
            "status": self.handle_status,
            "history": self.handle_history,
            "context": self.handle_context,
            "usage": self.handle_usage,
            "memory": self.handle_memory,
            "note": self.handle_note,
            "remember": self.handle_remember,
            "stop": self.handle_stop,
            "new": self.handle_new,
            "compact": self.handle_compact,
            "clear": self.handle_clear,
            "cancel": self.handle_cancel,
        }
        for name, handler in commands.items():
            app.add_handler(CommandHandler(name, handler))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check(self, update: Update) -> bool:
        """Delegate access check to channel."""
        return self._channel._check_access(update)

    def _get_state(self, update: Update) -> dict:
        """Get per-user state."""
        return self._channel.get_user_state(update.effective_user.id)

    async def _reply(self, update: Update, text: str, **kwargs: Any) -> None:
        """Reply with Markdown, falling back to plain text on parse error."""
        try:
            await update.message.reply_text(text, parse_mode="Markdown", **kwargs)
        except Exception:
            # Strip parse_mode and retry as plain text
            await update.message.reply_text(text, **kwargs)

    # ------------------------------------------------------------------
    # /start
    # ------------------------------------------------------------------

    async def handle_start(self, update: Update, context: Any) -> None:
        if not self._check(update):
            await update.message.reply_text("You are not authorized to use this bot.")
            return

        bot_name = self._channel._bot_username or "SpoonBot"
        await update.message.reply_text(
            f"Hello! I'm @{bot_name}, your AI assistant.\n\n"
            "Just send me a message and I'll help you with:\n"
            "• Running shell commands\n"
            "• Reading and writing files\n"
            "• Code analysis and generation\n"
            "• Research and information gathering\n\n"
            "Type /help for the full command list."
        )

    # ------------------------------------------------------------------
    # /help
    # ------------------------------------------------------------------

    async def handle_help(self, update: Update, context: Any) -> None:
        if not self._check(update):
            return

        lines = ["Available Commands\n"]
        for cmd, desc in BOT_COMMANDS:
            lines.append(f"/{cmd} - {desc}")
        lines.append("\nJust type any message to chat with the agent.")

        await update.message.reply_text("\n".join(lines))

    # ------------------------------------------------------------------
    # /model
    # ------------------------------------------------------------------

    async def handle_model(self, update: Update, context: Any) -> None:
        if not self._check(update):
            return

        try:
            current_model = self._agent.model if self._agent else "unknown"
            keyboard = build_model_keyboard(page=0, current_model=current_model)
            await self._reply(
                update,
                f"*Current model:* `{current_model}`\n\nSelect a model:",
                reply_markup=keyboard,
            )
        except Exception as e:
            logger.error(f"/model error: {e}")
            await update.message.reply_text(f"Error: {e}")

    # ------------------------------------------------------------------
    # /think
    # ------------------------------------------------------------------

    async def handle_think(self, update: Update, context: Any) -> None:
        if not self._check(update):
            return

        try:
            state = self._get_state(update)
            current = normalize_think_level(state.get("think_level", "off"))
            keyboard = build_think_keyboard(current)

            level_label = THINK_LEVELS.get(current, current)
            await self._reply(
                update,
                f"Think mode: {level_label}\n\nSelect reasoning intensity:",
                reply_markup=keyboard,
            )
        except Exception as e:
            logger.error(f"/think error: {e}")
            await update.message.reply_text(f"Error: {e}")

    # ------------------------------------------------------------------
    # /verbose
    # ------------------------------------------------------------------

    async def handle_verbose(self, update: Update, context: Any) -> None:
        if not self._check(update):
            return

        try:
            state = self._get_state(update)
            current = state.get("verbose", False)
            keyboard = build_verbose_keyboard(current)
            status = "ON" if current else "OFF"
            await self._reply(
                update,
                f"Verbose mode: {status}\n\n"
                "When enabled, thinking process is included in responses.",
                reply_markup=keyboard,
            )
        except Exception as e:
            logger.error(f"/verbose error: {e}")
            await update.message.reply_text(f"Error: {e}")

    # ------------------------------------------------------------------
    # /clear
    # ------------------------------------------------------------------

    async def handle_clear(self, update: Update, context: Any) -> None:
        if not self._check(update):
            return

        try:
            keyboard = build_confirm_keyboard("clear")
            await update.message.reply_text(
                "Are you sure you want to clear conversation history?",
                reply_markup=keyboard,
            )
        except Exception as e:
            logger.error(f"/clear error: {e}")
            await update.message.reply_text(f"Error: {e}")

    # ------------------------------------------------------------------
    # /history
    # ------------------------------------------------------------------

    async def handle_history(self, update: Update, context: Any) -> None:
        if not self._check(update):
            return

        if not self._agent:
            await update.message.reply_text("Agent not available.")
            return

        try:
            history = self._agent.get_history()
            if not history:
                await update.message.reply_text("No conversation history.")
                return

            recent = history[-10:]
            lines = ["Recent History\n"]
            for msg in recent:
                role = msg.get("role", "?")
                content = msg.get("content", "")
                if len(content) > 100:
                    content = content[:100] + "..."
                emoji = "You" if role == "user" else "Bot"
                lines.append(f"[{emoji}] {content}")

            await update.message.reply_text("\n\n".join(lines))
        except Exception as e:
            logger.error(f"/history error: {e}")
            await update.message.reply_text(f"Error: {e}")

    # ------------------------------------------------------------------
    # /status
    # ------------------------------------------------------------------

    async def handle_status(self, update: Update, context: Any) -> None:
        if not self._check(update):
            return

        if not self._agent:
            await update.message.reply_text("Agent not available.")
            return

        try:
            state = self._get_state(update)
            tools = self._agent.get_available_tools()
            active_count = sum(1 for t in tools if t.get("active"))
            total_count = len(tools)
            skill_list = self._agent.skills

            lines = [
                "*Bot Status*\n",
                f"*Model:* `{self._agent.model}`",
                f"*Provider:* `{self._agent.provider}`",
                f"*Tools:* {active_count}/{total_count} active",
                f"*Skills:* {len(skill_list)}",
                f"*Think:* {THINK_LEVELS.get(normalize_think_level(state.get('think_level', 'off')), 'off')}",
                f"*Verbose:* {'ON' if state.get('verbose') else 'OFF'}",
            ]

            await self._reply(update, "\n".join(lines))
        except Exception as e:
            logger.error(f"/status error: {e}")
            await update.message.reply_text(f"Error: {e}")

    # ------------------------------------------------------------------
    # /skill
    # ------------------------------------------------------------------

    async def handle_skill(self, update: Update, context: Any) -> None:
        if not self._check(update):
            return

        if not self._agent:
            await update.message.reply_text("Agent not available.")
            return

        try:
            skills = self._agent.skills
            if not skills:
                await update.message.reply_text("No skills available.")
                return

            keyboard = build_skill_keyboard(skills, page=0)
            await self._reply(
                update,
                f"Available Skills ({len(skills)} total)\n\nSelect a skill for details:",
                reply_markup=keyboard,
            )
        except Exception as e:
            logger.error(f"/skill error: {e}")
            await update.message.reply_text(f"Error: {e}")

    # ------------------------------------------------------------------
    # /tools
    # ------------------------------------------------------------------

    async def handle_tools(self, update: Update, context: Any) -> None:
        if not self._check(update):
            return

        if not self._agent:
            await update.message.reply_text("Agent not available.")
            return

        try:
            tools = self._agent.get_available_tools()
            if not tools:
                await update.message.reply_text("No tools registered.")
                return

            active = [t for t in tools if t.get("active")]
            inactive = [t for t in tools if not t.get("active")]

            lines = [f"Active Tools ({len(active)})\n"]
            for t in active:
                lines.append(f"  - {t['name']}")

            if inactive:
                lines.append(f"\nInactive Tools ({len(inactive)})")
                for t in inactive[:10]:
                    desc = t.get("description", "")[:60]
                    lines.append(f"  - {t['name']}: {desc}")
                if len(inactive) > 10:
                    lines.append(f"  ... and {len(inactive) - 10} more")

            await update.message.reply_text("\n".join(lines))
        except Exception as e:
            logger.error(f"/tools error: {e}")
            await update.message.reply_text(f"Error: {e}")

    # ------------------------------------------------------------------
    # /memory
    # ------------------------------------------------------------------

    async def handle_memory(self, update: Update, context: Any) -> None:
        if not self._check(update):
            return

        if not self._agent:
            await update.message.reply_text("Agent not available.")
            return

        try:
            context_text = self._agent.memory.get_memory_context()
            if not context_text:
                await update.message.reply_text("Memory is empty.")
                return

            # Truncate if needed
            if len(context_text) > 3000:
                context_text = context_text[:3000] + "\n\n... (truncated)"

            await update.message.reply_text(
                f"*Memory Summary*\n\n{context_text}",
                parse_mode="Markdown",
            )
        except Exception as e:
            logger.error(f"Memory command error: {e}")
            await update.message.reply_text(f"Error reading memory: {e}")

    # ------------------------------------------------------------------
    # /note <text>
    # ------------------------------------------------------------------

    async def handle_note(self, update: Update, context: Any) -> None:
        if not self._check(update):
            return

        if not self._agent:
            await update.message.reply_text("Agent not available.")
            return

        text = update.message.text.partition(" ")[2].strip()
        if not text:
            await update.message.reply_text("Usage: /note <text to save>")
            return

        try:
            self._agent.note(text)
            await update.message.reply_text("Note saved.")
        except Exception as e:
            await update.message.reply_text(f"Error saving note: {e}")

    # ------------------------------------------------------------------
    # /remember <text>
    # ------------------------------------------------------------------

    async def handle_remember(self, update: Update, context: Any) -> None:
        if not self._check(update):
            return

        if not self._agent:
            await update.message.reply_text("Agent not available.")
            return

        text = update.message.text.partition(" ")[2].strip()
        if not text:
            await update.message.reply_text("Usage: /remember <fact to remember>")
            return

        try:
            self._agent.remember(text)
            await update.message.reply_text("Remembered.")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    # ------------------------------------------------------------------
    # /cancel
    # ------------------------------------------------------------------

    async def handle_cancel(self, update: Update, context: Any) -> None:
        if not self._check(update):
            return

        await update.message.reply_text("Operation cancelled.")

    # ------------------------------------------------------------------
    # /whoami  — user info (no agent needed)
    # ------------------------------------------------------------------

    async def handle_whoami(self, update: Update, context: Any) -> None:
        if not self._check(update):
            return

        user = update.effective_user
        chat = update.effective_chat

        chat_type = chat.type if chat else "unknown"
        chat_name = ""
        if chat_type in ("group", "supergroup"):
            chat_name = f"\nGroup: {chat.title}"
        elif chat_type == "channel":
            chat_name = f"\nChannel: {chat.title}"

        lines = [
            "Your Info\n",
            f"User ID: {user.id}",
            f"Name: {user.full_name}",
        ]
        if user.username:
            lines.append(f"Username: @{user.username}")
        lines.append(f"Chat Type: {chat_type}")
        if chat_name:
            lines.append(chat_name.strip())

        await update.message.reply_text("\n".join(lines))

    # ------------------------------------------------------------------
    # /commands  — paginated command browser
    # ------------------------------------------------------------------

    async def handle_commands(self, update: Update, context: Any) -> None:
        if not self._check(update):
            return

        keyboard = build_commands_keyboard(BOT_COMMANDS, page=0)
        await update.message.reply_text(
            f"All Commands ({len(BOT_COMMANDS)} total)\n\nUse the arrows to browse:",
            reply_markup=keyboard,
        )

    # ------------------------------------------------------------------
    # /context  — context window usage
    # ------------------------------------------------------------------

    async def handle_context(self, update: Update, context: Any) -> None:
        if not self._check(update):
            return

        if not self._agent:
            await update.message.reply_text("Agent not available.")
            return

        try:
            history = self._agent.get_history()
            msg_count = len(history)

            # Rough token estimate: ~4 chars per token
            total_chars = sum(len(m.get("content", "")) for m in history)
            est_tokens = total_chars // 4

            lines = [
                "Context Window\n",
                f"Model: {self._agent.model}",
                f"Messages: {msg_count}",
                f"Est. tokens: ~{est_tokens:,}",
            ]
            await update.message.reply_text("\n".join(lines))
        except Exception as e:
            logger.error(f"/context error: {e}")
            await update.message.reply_text(f"Error: {e}")

    # ------------------------------------------------------------------
    # /reasoning  — reasoning display toggle
    # ------------------------------------------------------------------

    async def handle_reasoning(self, update: Update, context: Any) -> None:
        if not self._check(update):
            return

        try:
            state = self._get_state(update)
            current = state.get("reasoning", "off")
            keyboard = build_reasoning_keyboard(current)
            label = REASONING_LEVELS.get(current, current)
            await self._reply(
                update,
                f"Reasoning display: {label}\n\nSelect reasoning mode:",
                reply_markup=keyboard,
            )
        except Exception as e:
            logger.error(f"/reasoning error: {e}")
            await update.message.reply_text(f"Error: {e}")

    # ------------------------------------------------------------------
    # /usage  — token usage stats
    # ------------------------------------------------------------------

    async def handle_usage(self, update: Update, context: Any) -> None:
        if not self._check(update):
            return

        if not self._agent:
            await update.message.reply_text("Agent not available.")
            return

        try:
            usage = self._agent.get_usage()
            lines = [
                "Usage Stats\n",
                f"Session: {usage['session_key']}",
                f"Messages: {usage['messages']}",
                f"Model: {usage['model']}",
            ]
            await update.message.reply_text("\n".join(lines))
        except Exception as e:
            logger.error(f"/usage error: {e}")
            await update.message.reply_text(f"Error: {e}")

    # ------------------------------------------------------------------
    # /models  — provider-grouped model browser
    # ------------------------------------------------------------------

    async def handle_models(self, update: Update, context: Any) -> None:
        if not self._check(update):
            return

        try:
            current_model = self._agent.model if self._agent else None
            keyboard = build_provider_keyboard(current_model)
            await self._reply(
                update,
                "Select a provider to browse models:",
                reply_markup=keyboard,
            )
        except Exception as e:
            logger.error(f"/models error: {e}")
            await update.message.reply_text(f"Error: {e}")

    # ------------------------------------------------------------------
    # /stop  — stop current agent task
    # ------------------------------------------------------------------

    async def handle_stop(self, update: Update, context: Any) -> None:
        if not self._check(update):
            return

        if not self._agent:
            await update.message.reply_text("Agent not available.")
            return

        try:
            self._agent.stop_current_task()
            await update.message.reply_text(
                "Stop signal sent. The current task will be interrupted on next iteration."
            )
        except Exception as e:
            logger.error(f"/stop error: {e}")
            await update.message.reply_text(f"Error: {e}")

    # ------------------------------------------------------------------
    # /new  — start a new session (with confirmation)
    # ------------------------------------------------------------------

    async def handle_new(self, update: Update, context: Any) -> None:
        if not self._check(update):
            return

        keyboard = build_confirm_keyboard("new")
        await update.message.reply_text(
            "Start a new session? Current conversation will be preserved but cleared from context.",
            reply_markup=keyboard,
        )

    # ------------------------------------------------------------------
    # /compact  — compact session context
    # ------------------------------------------------------------------

    async def handle_compact(self, update: Update, context: Any) -> None:
        if not self._check(update):
            return

        if not self._agent:
            await update.message.reply_text("Agent not available.")
            return

        try:
            removed = self._agent.compact_session()
            if removed == 0:
                await update.message.reply_text(
                    "Session is already compact (4 or fewer messages)."
                )
            else:
                remaining = len(self._agent.get_history())
                await update.message.reply_text(
                    f"Session compacted: removed {removed} messages, {remaining} remain."
                )
        except Exception as e:
            logger.error(f"/compact error: {e}")
            await update.message.reply_text(f"Error: {e}")
