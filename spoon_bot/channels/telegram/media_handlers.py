"""Extended media handlers for Telegram bot (location, sticker)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from telegram import Update

from spoon_bot.bus.events import InboundMessage

if TYPE_CHECKING:
    from spoon_bot.channels.telegram.channel import TelegramChannel


class MediaHandlers:
    """Handles additional media types not covered by the core channel.

    Currently supports:
    - Location messages
    - Sticker messages
    """

    def __init__(self, channel: TelegramChannel) -> None:
        self._channel = channel

    # ------------------------------------------------------------------
    # Location
    # ------------------------------------------------------------------

    async def handle_location(self, update: Update, context: Any) -> None:
        """Handle location messages."""
        if not self._channel._check_access(update):
            return

        message = update.message
        user = update.effective_user
        location = message.location

        inbound = InboundMessage(
            content=f"[Location: {location.latitude}, {location.longitude}]",
            channel=self._channel.full_name,
            session_key=f"telegram_{self._channel.account_id}_{message.chat_id}",
            sender_id=str(user.id),
            sender_name=user.full_name,
            message_id=str(message.message_id),
            metadata={
                "chat_id": message.chat_id,
                "chat_type": message.chat.type,
                "latitude": location.latitude,
                "longitude": location.longitude,
            },
        )

        await self._channel.publish(inbound)

    # ------------------------------------------------------------------
    # Sticker
    # ------------------------------------------------------------------

    async def handle_sticker(self, update: Update, context: Any) -> None:
        """Handle sticker messages."""
        if not self._channel._check_access(update):
            return

        message = update.message
        user = update.effective_user
        sticker = message.sticker

        emoji = sticker.emoji or "?"

        inbound = InboundMessage(
            content=f"[Sticker: {emoji}]",
            channel=self._channel.full_name,
            session_key=f"telegram_{self._channel.account_id}_{message.chat_id}",
            sender_id=str(user.id),
            sender_name=user.full_name,
            message_id=str(message.message_id),
            metadata={
                "chat_id": message.chat_id,
                "chat_type": message.chat.type,
                "sticker_emoji": emoji,
                "sticker_set": sticker.set_name,
            },
        )

        await self._channel.publish(inbound)
