"""Telegram channel for bot integration."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from loguru import logger

from spoon_bot.channels.base import BaseChannel
from spoon_bot.bus.events import InboundMessage, OutboundMessage

if TYPE_CHECKING:
    pass


class TelegramChannel(BaseChannel):
    """
    Telegram bot channel.

    Integrates with Telegram Bot API for messaging.
    Requires python-telegram-bot library (optional dependency).
    """

    def __init__(
        self,
        token: str,
        name: str = "telegram",
        allowed_users: list[int] | None = None,
    ):
        """
        Initialize Telegram channel.

        Args:
            token: Telegram bot token.
            name: Channel name.
            allowed_users: List of allowed user IDs (None = allow all).
        """
        super().__init__(name)
        self.token = token
        self.allowed_users = allowed_users
        self._app = None
        self._running = False

    async def start(self) -> None:
        """Start the Telegram bot."""
        try:
            from telegram import Update
            from telegram.ext import (
                Application,
                CommandHandler,
                MessageHandler,
                filters,
            )
        except ImportError:
            logger.error(
                "python-telegram-bot not installed. "
                "Install with: pip install python-telegram-bot"
            )
            return

        # Build application
        self._app = Application.builder().token(self.token).build()

        # Add handlers
        self._app.add_handler(CommandHandler("start", self._handle_start))
        self._app.add_handler(CommandHandler("help", self._handle_help))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

        # Start polling
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()

        self._running = True
        logger.info("Telegram channel started")

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

        self._running = False
        logger.info("Telegram channel stopped")

    async def send(self, message: OutboundMessage) -> None:
        """
        Send message to Telegram.

        Args:
            message: Outbound message to send.
        """
        if not self._app:
            logger.error("Telegram app not initialized")
            return

        # Get chat_id from metadata (set by inbound handler)
        chat_id = message.metadata.get("chat_id")
        if not chat_id:
            logger.error("No chat_id in message metadata")
            return

        try:
            await self._app.bot.send_message(
                chat_id=chat_id,
                text=message.content,
                parse_mode="Markdown",
            )
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            # Try without markdown
            try:
                await self._app.bot.send_message(
                    chat_id=chat_id,
                    text=message.content,
                )
            except Exception as e2:
                logger.error(f"Failed to send plain message: {e2}")

    def _check_user_allowed(self, user_id: int) -> bool:
        """Check if user is allowed."""
        if self.allowed_users is None:
            return True
        return user_id in self.allowed_users

    async def _handle_start(self, update: Any, context: Any) -> None:
        """Handle /start command."""
        if not self._check_user_allowed(update.effective_user.id):
            await update.message.reply_text("You are not authorized to use this bot.")
            return

        await update.message.reply_text(
            "Hello! I'm spoon-bot, your local AI assistant.\n\n"
            "Just send me a message and I'll help you with:\n"
            "- Running shell commands\n"
            "- Reading and writing files\n"
            "- Code analysis and generation\n"
            "- Research and information gathering\n\n"
            "Type /help for more info."
        )

    async def _handle_help(self, update: Any, context: Any) -> None:
        """Handle /help command."""
        if not self._check_user_allowed(update.effective_user.id):
            return

        await update.message.reply_text(
            "**spoon-bot Commands**\n\n"
            "/start - Start the bot\n"
            "/help - Show this help\n\n"
            "Just type any message to chat with the agent.",
            parse_mode="Markdown",
        )

    async def _handle_message(self, update: Any, context: Any) -> None:
        """Handle incoming messages."""
        user = update.effective_user
        message = update.message

        if not self._check_user_allowed(user.id):
            await message.reply_text("You are not authorized to use this bot.")
            return

        # Create session key from chat_id
        chat_id = message.chat_id
        session_key = f"telegram_{chat_id}"

        # Create inbound message
        inbound = InboundMessage(
            content=message.text,
            channel=self.name,
            session_key=session_key,
            sender_id=str(user.id),
            sender_name=user.full_name,
            message_id=str(message.message_id),
            metadata={"chat_id": chat_id},
        )

        # Publish to bus
        await self.publish(inbound)
