"""CallbackQuery router for Telegram InlineKeyboard interactions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from telegram import Update

from spoon_bot.channels.telegram.constants import (
    AVAILABLE_MODELS,
    BOT_COMMANDS,
    CALLBACK_PREFIX,
    MODEL_PROVIDERS,
    REASONING_LEVELS,
    THINK_LEVELS,
)
from spoon_bot.channels.telegram.keyboards import (
    build_commands_keyboard,
    build_model_keyboard,
    build_provider_keyboard,
    build_provider_models_keyboard,
    build_reasoning_keyboard,
    build_skill_keyboard,
    build_think_keyboard,
    build_verbose_keyboard,
)

if TYPE_CHECKING:
    from spoon_bot.agent.loop import AgentLoop
    from spoon_bot.channels.telegram.channel import TelegramChannel


class CallbackRouter:
    """Routes CallbackQuery events to the appropriate handler.

    Parses the ``callback_data`` prefix and dispatches to a method that
    handles the specific interaction (model switch, pagination, etc.).
    """

    def __init__(self, channel: TelegramChannel) -> None:
        self._channel = channel
        self._agent: AgentLoop | None = None

    def set_agent(self, agent: AgentLoop) -> None:
        """Set agent reference."""
        self._agent = agent

    # ------------------------------------------------------------------
    # Main dispatcher
    # ------------------------------------------------------------------

    async def handle_callback(self, update: Update, context: Any) -> None:
        """Route a CallbackQuery to the correct handler."""
        query = update.callback_query
        if not query or not query.data:
            return

        await query.answer()  # acknowledge immediately

        data = query.data
        user_id = query.from_user.id

        # Dispatch by prefix — order matters: longer prefixes first
        if data.startswith(CALLBACK_PREFIX["model_list"]):
            await self._on_provider_model_page(query, data)
        elif data.startswith(CALLBACK_PREFIX["model_page"]):
            await self._on_model_page(query, data)
        elif data.startswith(CALLBACK_PREFIX["model_provider"]):
            await self._on_provider_select(query, data)
        elif data.startswith(CALLBACK_PREFIX["model"]):
            await self._on_model_select(query, data, user_id)
        elif data.startswith(CALLBACK_PREFIX["think"]):
            await self._on_think_select(query, data, user_id)
        elif data.startswith(CALLBACK_PREFIX["verbose"]):
            await self._on_verbose_toggle(query, data, user_id)
        elif data.startswith(CALLBACK_PREFIX["reasoning"]):
            await self._on_reasoning_select(query, data, user_id)
        elif data.startswith(CALLBACK_PREFIX["skill_page"]):
            await self._on_skill_page(query, data)
        elif data.startswith(CALLBACK_PREFIX["skill"]):
            await self._on_skill_select(query, data)
        elif data.startswith(CALLBACK_PREFIX["commands_page"]):
            await self._on_commands_page(query, data)
        elif data.startswith(CALLBACK_PREFIX["confirm_new"]):
            await self._on_confirm_new(query, data, user_id)
        elif data.startswith(CALLBACK_PREFIX["confirm_clear"]):
            await self._on_confirm_clear(query, data, user_id)
        elif data == "noop":
            pass
        else:
            logger.warning(f"Unknown callback data: {data}")

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    async def _on_model_select(self, query: Any, data: str, user_id: int) -> None:
        model_id = data[len(CALLBACK_PREFIX["model"]):]

        # Validate model exists
        valid = any(m["id"] == model_id for m in AVAILABLE_MODELS)
        if not valid:
            await query.edit_message_text(f"Unknown model: {model_id}")
            return

        if self._agent:
            self._agent.model = model_id
            logger.info(f"User {user_id} switched model to {model_id}")

        # Re-render keyboard with updated selection
        keyboard = build_model_keyboard(page=0, current_model=model_id)
        model_name = next(
            (m["name"] for m in AVAILABLE_MODELS if m["id"] == model_id),
            model_id,
        )
        await query.edit_message_text(
            f"*Model switched to:* `{model_name}`\n\nSelect a model:",
            reply_markup=keyboard,
            parse_mode="Markdown",
        )

    async def _on_model_page(self, query: Any, data: str) -> None:
        page = int(data[len(CALLBACK_PREFIX["model_page"]):])
        current_model = self._agent.model if self._agent else None
        keyboard = build_model_keyboard(page=page, current_model=current_model)
        await query.edit_message_reply_markup(reply_markup=keyboard)

    # ------------------------------------------------------------------
    # Think level
    # ------------------------------------------------------------------

    async def _on_think_select(self, query: Any, data: str, user_id: int) -> None:
        level = data[len(CALLBACK_PREFIX["think"]):]
        if level not in THINK_LEVELS:
            return

        self._channel.set_user_state(user_id, "think_level", level)
        logger.info(f"User {user_id} set think level to {level}")

        keyboard = build_think_keyboard(level)
        label = THINK_LEVELS[level]
        await query.edit_message_text(
            f"*Think mode:* {label}\n\nSelect thinking level:",
            reply_markup=keyboard,
            parse_mode="Markdown",
        )

    # ------------------------------------------------------------------
    # Verbose toggle
    # ------------------------------------------------------------------

    async def _on_verbose_toggle(self, query: Any, data: str, user_id: int) -> None:
        value = data[len(CALLBACK_PREFIX["verbose"]):] == "on"

        self._channel.set_user_state(user_id, "verbose", value)
        logger.info(f"User {user_id} set verbose to {value}")

        keyboard = build_verbose_keyboard(value)
        status = "ON" if value else "OFF"
        await query.edit_message_text(
            f"*Verbose mode:* {status}\n\n"
            "When enabled, thinking process is included in responses.",
            reply_markup=keyboard,
            parse_mode="Markdown",
        )

    # ------------------------------------------------------------------
    # Skill browsing
    # ------------------------------------------------------------------

    async def _on_skill_select(self, query: Any, data: str) -> None:
        skill_name = data[len(CALLBACK_PREFIX["skill"]):]

        # Try to get skill details from agent
        detail = f"*Skill:* `{skill_name}`"
        if self._agent and self._agent._skill_manager:
            try:
                all_skills = self._agent._skill_manager.list()
                if skill_name in all_skills:
                    detail += "\n\nThis skill is available and ready to use."
            except Exception:
                pass

        await query.message.reply_text(detail, parse_mode="Markdown")

    async def _on_skill_page(self, query: Any, data: str) -> None:
        page = int(data[len(CALLBACK_PREFIX["skill_page"]):])
        skills = self._agent.skills if self._agent else []
        keyboard = build_skill_keyboard(skills, page=page)
        await query.edit_message_reply_markup(reply_markup=keyboard)

    # ------------------------------------------------------------------
    # Clear confirmation
    # ------------------------------------------------------------------

    async def _on_confirm_clear(self, query: Any, data: str, user_id: int) -> None:
        choice = data[len(CALLBACK_PREFIX["confirm_clear"]):]

        if choice == "yes" and self._agent:
            self._agent.clear_history()
            logger.info(f"User {user_id} cleared conversation history")
            await query.edit_message_text("Conversation history cleared.")
        else:
            await query.edit_message_text("Clear cancelled.")

    # ------------------------------------------------------------------
    # Commands pagination
    # ------------------------------------------------------------------

    async def _on_commands_page(self, query: Any, data: str) -> None:
        page = int(data[len(CALLBACK_PREFIX["commands_page"]):])
        keyboard = build_commands_keyboard(BOT_COMMANDS, page=page)
        await query.edit_message_reply_markup(reply_markup=keyboard)

    # ------------------------------------------------------------------
    # Reasoning display toggle
    # ------------------------------------------------------------------

    async def _on_reasoning_select(self, query: Any, data: str, user_id: int) -> None:
        level = data[len(CALLBACK_PREFIX["reasoning"]):]
        if level not in REASONING_LEVELS:
            return

        self._channel.set_user_state(user_id, "reasoning", level)
        logger.info(f"User {user_id} set reasoning to {level}")

        keyboard = build_reasoning_keyboard(level)
        label = REASONING_LEVELS[level]
        await query.edit_message_text(
            f"Reasoning display: {label}\n\nSelect reasoning mode:",
            reply_markup=keyboard,
        )

    # ------------------------------------------------------------------
    # New session confirmation
    # ------------------------------------------------------------------

    async def _on_confirm_new(self, query: Any, data: str, user_id: int) -> None:
        choice = data[len(CALLBACK_PREFIX["confirm_new"]):]

        if choice == "yes" and self._agent:
            new_key = self._agent.new_session()
            logger.info(f"User {user_id} started new session: {new_key}")
            await query.edit_message_text(f"New session started.\nSession ID: {new_key}")
        else:
            await query.edit_message_text("New session cancelled.")

    # ------------------------------------------------------------------
    # Provider selection (first level of /models keyboard)
    # ------------------------------------------------------------------

    async def _on_provider_select(self, query: Any, data: str) -> None:
        provider = data[len(CALLBACK_PREFIX["model_provider"]):]

        if provider == "__back__":
            # Return to provider list
            current_model = self._agent.model if self._agent else None
            keyboard = build_provider_keyboard(current_model)
            await query.edit_message_text(
                "Select a provider to browse models:",
                reply_markup=keyboard,
            )
            return

        if provider not in MODEL_PROVIDERS:
            await query.edit_message_text(f"Unknown provider: {provider}")
            return

        current_model = self._agent.model if self._agent else None
        keyboard = build_provider_models_keyboard(provider, page=0, current_model=current_model)
        model_count = len(MODEL_PROVIDERS[provider])
        await query.edit_message_text(
            f"{provider} — {model_count} models\n\nSelect a model:",
            reply_markup=keyboard,
        )

    # ------------------------------------------------------------------
    # Provider model list pagination (second level of /models keyboard)
    # ------------------------------------------------------------------

    async def _on_provider_model_page(self, query: Any, data: str) -> None:
        payload = data[len(CALLBACK_PREFIX["model_list"]):]
        # payload format: "{provider}:{page}"
        if ":" not in payload:
            return
        provider, page_str = payload.rsplit(":", 1)
        try:
            page = int(page_str)
        except ValueError:
            return

        current_model = self._agent.model if self._agent else None
        keyboard = build_provider_models_keyboard(provider, page=page, current_model=current_model)
        await query.edit_message_reply_markup(reply_markup=keyboard)
