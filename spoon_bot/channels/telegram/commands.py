"""Slash command handlers for Telegram bot."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from telegram import Update
from telegram.ext import CommandHandler

from spoon_bot.channels.telegram.constants import BOT_COMMANDS, REASONING_LEVELS, THINK_LEVELS
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
            "subagents": self.handle_subagents,
            "agents": self.handle_subagents,
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
            current = state.get("think_level", "off")
            keyboard = build_think_keyboard(current)

            level_label = THINK_LEVELS.get(current, current)
            await self._reply(
                update,
                f"Think mode: {level_label}\n\nSelect thinking level:",
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
                f"*Think:* {THINK_LEVELS.get(state.get('think_level', 'off'), 'off')}",
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
    # /subagents [list|spawn <task>|cancel [id|all]]
    # /agents — alias for /subagents
    # ------------------------------------------------------------------

    async def handle_subagents(self, update: Update, context: Any) -> None:
        if not self._check(update):
            return

        if not self._agent:
            await self._reply(update, "Agent not available.")
            return

        manager = self._agent.subagent_manager
        if manager is None:
            await self._reply(update, "Sub-agent manager not available.")
            return

        args = context.args or []
        subcommand = args[0].lower() if args else "list"

        if subcommand in ("list", "ls"):
            await self._subagents_list(update, manager)
        elif subcommand == "spawn":
            rest = args[1:]
            await self._subagents_spawn(update, manager, rest)
        elif subcommand in ("cancel", "kill"):
            target = args[1] if len(args) > 1 else ""
            await self._subagents_cancel(update, manager, target)
        elif subcommand == "steer":
            agent_id = args[1] if len(args) > 1 else ""
            message = " ".join(args[2:]).strip()
            await self._subagents_steer(update, manager, agent_id, message)
        elif subcommand == "info":
            agent_id = args[1] if len(args) > 1 else ""
            await self._subagents_info(update, manager, agent_id)
        elif subcommand == "help":
            await self._reply(
                update,
                "*Sub-agent commands:*\n"
                "`/subagents` — list all sub-agents\n"
                "`/subagents spawn [--model M] [--thinking L] <task>` — spawn\n"
                "`/subagents cancel <id|all>` — cascade-cancel sub-agent(s)\n"
                "`/subagents kill <id|all>` — alias for cancel\n"
                "`/subagents steer <id> <new instructions>` — redirect running agent\n"
                "`/subagents info <id>` — detailed metadata\n"
                "`/agents` — alias for /subagents",
            )
        else:
            await self._reply(
                update,
                "*Usage:*\n"
                "`/subagents` — list all sub-agents\n"
                "`/subagents spawn <task>` — spawn a new sub-agent\n"
                "`/subagents cancel <id|all>` — cascade-cancel sub-agent(s)\n"
                "`/subagents steer <id> <message>` — redirect running agent\n"
                "`/subagents info <id>` — show details\n"
                "`/subagents help` — full command reference",
            )

    async def _subagents_list(self, update: Update, manager: Any) -> None:
        summary = manager.get_status_summary()
        total = summary.get("total", 0)
        if total == 0:
            await self._reply(update, "No sub-agents spawned.")
            return

        lines = [f"*Sub-agents ({total} total):*"]

        active = summary.get("active", [])
        if active:
            lines.append("\n*Active:*")
            for e in active:
                agent_id = e["agent_id"]
                label = e["label"]
                state = e["state"]
                elapsed = e.get("elapsed_seconds")
                model = e.get("model") or ""
                pending_desc = e.get("pending_descendants", 0)
                elapsed_str = f" — {elapsed}s" if elapsed is not None else ""
                model_str = f" \\[{model}]" if model else ""
                desc_str = (
                    f" *(+{pending_desc} descendants)*"
                    if pending_desc > 0 else ""
                )
                lines.append(
                    f"`[{agent_id}]` {label}: *{state}*{elapsed_str}{model_str}{desc_str}"
                )

        recent = summary.get("recent", [])
        if recent:
            lines.append("\n*Recent:*")
            for e in recent:
                agent_id = e["agent_id"]
                label = e["label"]
                state = e["state"]
                elapsed = e.get("elapsed_seconds")
                model = e.get("model") or ""
                elapsed_str = f" — {elapsed}s" if elapsed is not None else ""
                model_str = f" \\[{model}]" if model else ""
                lines.append(f"`[{agent_id}]` {label}: *{state}*{elapsed_str}{model_str}")

        await self._reply(update, "\n".join(lines))

    async def _subagents_spawn(
        self, update: Update, manager: Any, args: list
    ) -> None:
        """Parse optional --model and --thinking flags, then spawn."""
        from spoon_bot.subagent.models import SubagentConfig

        model: str | None = None
        thinking: str | None = None
        task_parts: list[str] = []

        i = 0
        while i < len(args):
            token = args[i]
            if token == "--model" and i + 1 < len(args):
                model = args[i + 1]
                i += 2
            elif token == "--thinking" and i + 1 < len(args):
                thinking = args[i + 1]
                i += 2
            elif token.startswith("--model="):
                model = token.split("=", 1)[1]
                i += 1
            elif token.startswith("--thinking="):
                thinking = token.split("=", 1)[1]
                i += 1
            else:
                task_parts.append(token)
                i += 1

        task = " ".join(task_parts).strip()
        if not task:
            await self._reply(
                update,
                "Usage: `/subagents spawn [--model MODEL] [--thinking LEVEL] <task>`",
            )
            return

        config = SubagentConfig()
        if model:
            config.model = model
        if thinking:
            config.thinking_level = thinking

        try:
            record = await manager.spawn(task=task, label=task[:60], config=config)
            model_str = f"\nModel: {record.model_name}" if record.model_name else ""
            thinking_str = f"\nThinking: {thinking}" if thinking else ""
            await self._reply(
                update,
                f"*Sub-agent spawned!*\n"
                f"ID: `{record.agent_id}`\n"
                f"Label: {record.label}"
                f"{model_str}{thinking_str}\n"
                f"Use /subagents to monitor progress.",
            )
        except ValueError as exc:
            await self._reply(update, f"Failed to spawn sub-agent: {exc}")
        except Exception as exc:
            logger.error(f"Unexpected error spawning sub-agent: {exc}")
            await self._reply(update, f"Error spawning sub-agent: {exc}")

    async def _subagents_cancel(self, update: Update, manager: Any, target: str) -> None:
        if not target:
            await self._reply(
                update,
                "Usage: `/subagents cancel <agent_id>` or `/subagents cancel all`",
            )
            return

        if target.lower() == "all":
            count = await manager.cancel_all()
            await self._reply(
                update,
                f"Cascade cancellation requested for {count} sub-agent(s).",
            )
        else:
            # Count descendants for informative message
            record = manager.registry.get(target)
            descendants = (
                manager.registry.get_descendants(target) if record else []
            )
            found = await manager.cancel(target, cascade=True)
            if found:
                desc_count = len(descendants)
                desc_str = f" and {desc_count} descendants" if desc_count else ""
                await self._reply(
                    update,
                    f"Cancellation requested for sub-agent `{target}`{desc_str}.",
                )
            else:
                await self._reply(
                    update,
                    f"Sub-agent `{target}` not found or already finished.",
                )

    async def _subagents_steer(
        self, update: Update, manager: Any, agent_id: str, message: str
    ) -> None:
        if not agent_id:
            await self._reply(
                update,
                "Usage: `/subagents steer <agent_id> <new instructions>`",
            )
            return
        if not message:
            await self._reply(
                update,
                "Usage: `/subagents steer <agent_id> <new instructions>`\n"
                "You must provide new instructions for the sub-agent.",
            )
            return

        result = await manager.steer(agent_id, message)
        status = result.get("status")
        msg = result.get("message", "")

        if status == "accepted":
            await self._reply(
                update,
                f"*Steer accepted!*\n"
                f"Sub-agent `{agent_id}` will be redirected.\n"
                f"{msg}",
            )
        elif status == "rate_limited":
            await self._reply(update, f"*Rate limited:* {msg}")
        elif status == "done":
            await self._reply(update, f"*Cannot steer:* {msg}")
        else:
            await self._reply(update, f"Sub-agent not found: `{agent_id}`.")

    async def _subagents_info(
        self, update: Update, manager: Any, agent_id: str
    ) -> None:
        if not agent_id:
            await self._reply(update, "Usage: `/subagents info <agent_id>`")
            return

        info = await manager.get_info(agent_id)
        if info is None:
            await self._reply(update, f"Sub-agent `{agent_id}` not found.")
            return

        lines = [f"*Sub-agent `{agent_id}` info:*"]
        lines.append(f"Label: {info['label']}")
        lines.append(f"State: *{info['state']}*")
        lines.append(f"Task: {info['task'][:120]}")
        lines.append(f"Depth: {info['depth']}")
        if info.get("model"):
            lines.append(f"Model: `{info['model']}`")
        lines.append(f"Tool profile: {info['tool_profile']}")
        if info.get("thinking_level"):
            lines.append(f"Thinking: {info['thinking_level']}")
        if info.get("elapsed_seconds") is not None:
            lines.append(f"Elapsed: {info['elapsed_seconds']}s")
        if info.get("pending_descendants", 0) > 0:
            lines.append(f"Pending descendants: {info['pending_descendants']}")
        if info.get("children"):
            lines.append(f"Children: {', '.join(f'`{c}`' for c in info['children'])}")
        if info.get("token_usage"):
            tu = info["token_usage"]
            lines.append(
                f"Tokens: {tu.get('total_tokens', 0)} "
                f"(in {tu.get('input_tokens', 0)} / out {tu.get('output_tokens', 0)})"
            )
        if info.get("result_preview"):
            lines.append(f"Result preview:\n_{info['result_preview']}_")
        if info.get("error"):
            lines.append(f"Error: {info['error']}")

        await self._reply(update, "\n".join(lines))

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
