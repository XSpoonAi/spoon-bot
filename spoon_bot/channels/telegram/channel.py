"""Enhanced Telegram channel with full features."""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from spoon_bot.bus.events import InboundMessage, OutboundMessage
from spoon_bot.channels.base import BaseChannel, ChannelConfig, ChannelMode, ChannelStatus

try:
    from telegram import Bot, Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning(
        "python-telegram-bot not installed. "
        "Install with: pip install python-telegram-bot[all]"
    )


class TelegramChannel(BaseChannel):
    """
    Enhanced Telegram bot channel.

    Features:
    - Polling and Webhook modes
    - Group chat support with mention filtering
    - Media handling (images, documents, audio, video)
    - Inline buttons support
    - User whitelist
    - Per-account configuration
    """

    def __init__(self, config: ChannelConfig, account_id: str | None = None):
        """
        Initialize Telegram channel.

        Args:
            config: Channel configuration with Telegram-specific settings
            account_id: Account identifier (bot name)
        """
        if not TELEGRAM_AVAILABLE:
            raise ImportError("python-telegram-bot is required for Telegram channel")

        super().__init__(config, account_id)

        # Telegram-specific config
        self.token = config.extra.get("token")
        if not self.token:
            raise ValueError("Telegram token is required")

        self.allowed_users = set(config.extra.get("allowed_users", []))
        self.groups_config = config.extra.get("groups", {})
        self.groups_enabled = self.groups_config.get("enabled", False)
        self.require_mention = self.groups_config.get("require_mention", True)
        self.allowed_chats = set(self.groups_config.get("allowed_chats", []))
        self.media_max_mb = config.extra.get("media_max_mb", 20)
        self.proxy_url = config.extra.get("proxy_url")  # e.g., "http://127.0.0.1:7890"

        # Application
        self._app: Application | None = None
        self._bot: Bot | None = None
        self._bot_username: str | None = None

    async def start(self) -> None:
        """Start the Telegram bot."""
        self._set_status(ChannelStatus.STARTING)

        try:
            # Build application with extended timeouts for proxy
            from telegram.request import HTTPXRequest

            # Create request with longer timeouts (important for unstable proxies)
            request = HTTPXRequest(
                connection_pool_size=8,
                connect_timeout=30.0,  # Increased from default 5.0
                read_timeout=30.0,      # Increased from default 5.0
                write_timeout=30.0,     # Increased from default 5.0
                pool_timeout=30.0       # Increased from default 1.0
            )

            builder = Application.builder().token(self.token).request(request)

            # Configure proxy if provided
            if self.proxy_url:
                logger.info(f"[{self.full_name}] Using proxy: {self.proxy_url}")
                builder = builder.proxy(self.proxy_url)

            self._app = builder.build()
            self._bot = self._app.bot

            # Get bot info
            bot_info = await self._bot.get_me()
            self._bot_username = bot_info.username
            logger.info(f"[{self.full_name}] Bot username: @{self._bot_username}")

            # Add handlers
            self._register_handlers()

            # Start application
            await self._app.initialize()
            await self._app.start()

            if self.config.mode == ChannelMode.WEBHOOK:
                # Webhook mode
                await self._setup_webhook()
            else:
                # Polling mode
                await self._app.updater.start_polling(drop_pending_updates=True)

            self._running = True
            self._set_status(ChannelStatus.RUNNING)

            # Start health check
            self._health_check_task = asyncio.create_task(self._start_health_check_loop())

            logger.info(
                f"[{self.full_name}] Started in {self.config.mode.value} mode"
            )

        except Exception as e:
            self._set_status(ChannelStatus.ERROR, e)
            raise

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        if not self._running:
            return

        try:
            # Stop health check
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass

            # Stop application
            if self._app:
                if self.config.mode == ChannelMode.WEBHOOK:
                    await self._remove_webhook()
                else:
                    await self._app.updater.stop()

                await self._app.stop()
                await self._app.shutdown()

            self._running = False
            self._set_status(ChannelStatus.STOPPED)

            logger.info(f"[{self.full_name}] Stopped")

        except Exception as e:
            logger.error(f"[{self.full_name}] Error during stop: {e}")
            self._set_status(ChannelStatus.ERROR, e)

    async def send(self, message: OutboundMessage) -> None:
        """
        Send message to Telegram.

        Args:
            message: Outbound message to send
        """
        if not self._bot:
            logger.error(f"[{self.full_name}] Bot not initialized")
            return

        chat_id = message.metadata.get("chat_id")
        if not chat_id:
            logger.error(f"[{self.full_name}] No chat_id in message metadata")
            return

        # Telegram message limit is 4096 characters
        MAX_LENGTH = 4000  # Leave some margin
        content = message.content

        # Split long messages
        if len(content) > MAX_LENGTH:
            logger.info(f"[{self.full_name}] Splitting long message ({len(content)} chars)")
            chunks = []
            while content:
                if len(content) <= MAX_LENGTH:
                    chunks.append(content)
                    break

                # Try to split at newline
                split_pos = content.rfind('\n', 0, MAX_LENGTH)
                if split_pos == -1:
                    # No newline found, split at space
                    split_pos = content.rfind(' ', 0, MAX_LENGTH)
                if split_pos == -1:
                    # No space found, force split
                    split_pos = MAX_LENGTH

                chunks.append(content[:split_pos])
                content = content[split_pos:].lstrip()

            # Send each chunk
            for i, chunk in enumerate(chunks):
                async def _send_chunk(text=chunk, first=(i==0)):
                    try:
                        await self._bot.send_message(
                            chat_id=chat_id,
                            text=text,
                            parse_mode="Markdown",
                            reply_to_message_id=message.reply_to if message.reply_to and first else None,
                        )
                    except Exception:
                        # Fallback to plain text
                        await self._bot.send_message(
                            chat_id=chat_id,
                            text=text,
                            reply_to_message_id=message.reply_to if message.reply_to and first else None,
                        )

                await self.send_with_retry(_send_chunk)
                # Small delay between chunks
                if i < len(chunks) - 1:
                    await asyncio.sleep(0.5)
        else:
            # Send normal message
            async def _send():
                try:
                    await self._bot.send_message(
                        chat_id=chat_id,
                        text=content,
                        parse_mode="Markdown",
                        reply_to_message_id=message.reply_to if message.reply_to else None,
                    )
                except Exception:
                    # Fallback to plain text
                    await self._bot.send_message(
                        chat_id=chat_id,
                        text=content,
                        reply_to_message_id=message.reply_to if message.reply_to else None,
                    )

            await self.send_with_retry(_send)

    def _register_handlers(self) -> None:
        """Register message handlers."""
        if not self._app:
            return

        # Command handlers
        self._app.add_handler(CommandHandler("start", self._handle_start))
        self._app.add_handler(CommandHandler("help", self._handle_help))

        # Message handlers
        self._app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._handle_text_message,
            )
        )

        # Media handlers
        self._app.add_handler(
            MessageHandler(filters.PHOTO, self._handle_photo)
        )
        self._app.add_handler(
            MessageHandler(filters.Document.ALL, self._handle_document)
        )
        self._app.add_handler(
            MessageHandler(filters.AUDIO, self._handle_audio)
        )
        self._app.add_handler(
            MessageHandler(filters.VIDEO, self._handle_video)
        )
        self._app.add_handler(
            MessageHandler(filters.VOICE, self._handle_voice)
        )

        logger.debug(f"[{self.full_name}] Handlers registered")

    async def _handle_start(self, update: Update, context: Any) -> None:
        """Handle /start command."""
        if not self._check_access(update):
            await update.message.reply_text("You are not authorized to use this bot.")
            return

        await update.message.reply_text(
            f"Hello! I'm {self._bot_username}, your AI assistant.\n\n"
            "Just send me a message and I'll help you with:\n"
            "• Running shell commands\n"
            "• Reading and writing files\n"
            "• Code analysis and generation\n"
            "• Research and information gathering\n\n"
            "Type /help for more info."
        )

    async def _handle_help(self, update: Update, context: Any) -> None:
        """Handle /help command."""
        if not self._check_access(update):
            return

        await update.message.reply_text(
            f"**{self._bot_username} Commands**\n\n"
            "/start - Start the bot\n"
            "/help - Show this help\n\n"
            "Just type any message to chat with the agent.",
            parse_mode="Markdown",
        )

    async def _handle_text_message(self, update: Update, context: Any) -> None:
        """Handle text messages."""
        if not self._check_access(update):
            await update.message.reply_text("You are not authorized to use this bot.")
            return

        user = update.effective_user
        message = update.message

        # Create session key
        chat_id = message.chat_id
        session_key = f"telegram_{self.account_id}_{chat_id}"

        # Create inbound message
        inbound = InboundMessage(
            content=message.text,
            channel=self.full_name,
            session_key=session_key,
            sender_id=str(user.id),
            sender_name=user.full_name,
            message_id=str(message.message_id),
            metadata={
                "chat_id": chat_id,
                "chat_type": message.chat.type,
                "username": user.username,
            },
        )

        # Publish to bus
        await self.publish(inbound)

    async def _handle_photo(self, update: Update, context: Any) -> None:
        """Handle photo messages."""
        if not self._check_access(update):
            return

        message = update.message
        user = update.effective_user

        # Get the largest photo
        photo = message.photo[-1]
        file = await photo.get_file()

        # Create inbound message with media
        inbound = InboundMessage(
            content=message.caption or "[Photo]",
            channel=self.full_name,
            session_key=f"telegram_{self.account_id}_{message.chat_id}",
            sender_id=str(user.id),
            sender_name=user.full_name,
            message_id=str(message.message_id),
            metadata={
                "chat_id": message.chat_id,
                "chat_type": message.chat.type,
            },
            media=[
                {
                    "type": "photo",
                    "file_id": photo.file_id,
                    "file_url": file.file_path,
                    "size": photo.file_size,
                }
            ],
        )

        await self.publish(inbound)

    async def _handle_document(self, update: Update, context: Any) -> None:
        """Handle document messages."""
        if not self._check_access(update):
            return

        message = update.message
        user = update.effective_user
        document = message.document

        # Check file size
        max_size = self.media_max_mb * 1024 * 1024
        if document.file_size > max_size:
            await message.reply_text(
                f"File too large. Maximum size: {self.media_max_mb}MB"
            )
            return

        file = await document.get_file()

        # Create inbound message
        inbound = InboundMessage(
            content=message.caption or f"[Document: {document.file_name}]",
            channel=self.full_name,
            session_key=f"telegram_{self.account_id}_{message.chat_id}",
            sender_id=str(user.id),
            sender_name=user.full_name,
            message_id=str(message.message_id),
            metadata={
                "chat_id": message.chat_id,
                "chat_type": message.chat.type,
            },
            media=[
                {
                    "type": "document",
                    "file_id": document.file_id,
                    "file_name": document.file_name,
                    "file_url": file.file_path,
                    "mime_type": document.mime_type,
                    "size": document.file_size,
                }
            ],
        )

        await self.publish(inbound)

    async def _handle_audio(self, update: Update, context: Any) -> None:
        """Handle audio messages."""
        if not self._check_access(update):
            return

        message = update.message
        user = update.effective_user
        audio = message.audio

        file = await audio.get_file()

        inbound = InboundMessage(
            content=message.caption or "[Audio]",
            channel=self.full_name,
            session_key=f"telegram_{self.account_id}_{message.chat_id}",
            sender_id=str(user.id),
            sender_name=user.full_name,
            message_id=str(message.message_id),
            metadata={"chat_id": message.chat_id},
            media=[
                {
                    "type": "audio",
                    "file_id": audio.file_id,
                    "file_url": file.file_path,
                    "duration": audio.duration,
                }
            ],
        )

        await self.publish(inbound)

    async def _handle_video(self, update: Update, context: Any) -> None:
        """Handle video messages."""
        # Similar to photo handler
        if not self._check_access(update):
            return

        message = update.message
        user = update.effective_user
        video = message.video

        max_size = self.media_max_mb * 1024 * 1024
        if video.file_size > max_size:
            await message.reply_text(f"Video too large. Maximum: {self.media_max_mb}MB")
            return

        file = await video.get_file()

        inbound = InboundMessage(
            content=message.caption or "[Video]",
            channel=self.full_name,
            session_key=f"telegram_{self.account_id}_{message.chat_id}",
            sender_id=str(user.id),
            sender_name=user.full_name,
            message_id=str(message.message_id),
            metadata={"chat_id": message.chat_id},
            media=[
                {
                    "type": "video",
                    "file_id": video.file_id,
                    "file_url": file.file_path,
                    "duration": video.duration,
                    "width": video.width,
                    "height": video.height,
                }
            ],
        )

        await self.publish(inbound)

    async def _handle_voice(self, update: Update, context: Any) -> None:
        """Handle voice messages."""
        if not self._check_access(update):
            return

        message = update.message
        user = update.effective_user
        voice = message.voice

        file = await voice.get_file()

        inbound = InboundMessage(
            content="[Voice message]",
            channel=self.full_name,
            session_key=f"telegram_{self.account_id}_{message.chat_id}",
            sender_id=str(user.id),
            sender_name=user.full_name,
            message_id=str(message.message_id),
            metadata={"chat_id": message.chat_id},
            media=[
                {
                    "type": "voice",
                    "file_id": voice.file_id,
                    "file_url": file.file_path,
                    "duration": voice.duration,
                }
            ],
        )

        await self.publish(inbound)

    def _check_access(self, update: Update) -> bool:
        """
        Check if user has access.

        Args:
            update: Telegram update

        Returns:
            True if user is allowed
        """
        user = update.effective_user
        message = update.message or update.edited_message

        if not message:
            return False

        # Check user whitelist
        if self.allowed_users and user.id not in self.allowed_users:
            logger.warning(
                f"[{self.full_name}] Unauthorized user: {user.id} ({user.full_name})"
            )
            return False

        # Check group access
        if message.chat.type in ["group", "supergroup"]:
            if not self.groups_enabled:
                logger.debug(f"[{self.full_name}] Groups not enabled")
                return False

            # Check chat whitelist
            if self.allowed_chats:
                chat_identifier = message.chat.username or message.chat.id
                if chat_identifier not in self.allowed_chats:
                    logger.debug(
                        f"[{self.full_name}] Chat not in whitelist: {chat_identifier}"
                    )
                    return False

            # Check mention requirement
            if self.require_mention:
                if not message.text:
                    return False

                # Check if bot is mentioned
                mentioned = (
                    f"@{self._bot_username}" in message.text
                    or message.text.startswith("/")
                )
                if not mentioned:
                    return False

        return True

    async def _setup_webhook(self) -> None:
        """Setup webhook (for webhook mode)."""
        if not self.config.webhook_path:
            raise ValueError("webhook_path is required for webhook mode")

        webhook_url = self.config.webhook_path
        await self._bot.set_webhook(url=webhook_url)
        logger.info(f"[{self.full_name}] Webhook set to: {webhook_url}")

    async def _remove_webhook(self) -> None:
        """Remove webhook."""
        await self._bot.delete_webhook()
        logger.info(f"[{self.full_name}] Webhook removed")

    async def handle_webhook(self, request: Any) -> dict[str, Any]:
        """
        Handle incoming webhook request.

        Args:
            request: HTTP request object with .json() method

        Returns:
            Response dictionary
        """
        try:
            # Parse update from request
            update_data = await request.json()
            update = Update.de_json(update_data, self._bot)

            # Process update
            await self._app.process_update(update)

            return {"ok": True}

        except Exception as e:
            logger.error(f"[{self.full_name}] Webhook error: {e}")
            return {"ok": False, "error": str(e)}
