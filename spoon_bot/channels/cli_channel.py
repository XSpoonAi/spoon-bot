"""CLI channel for interactive terminal sessions."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

from loguru import logger
from rich.console import Console
from rich.prompt import Prompt

from spoon_bot.channels.base import BaseChannel
from spoon_bot.bus.events import InboundMessage, OutboundMessage

if TYPE_CHECKING:
    pass


class CLIChannel(BaseChannel):
    """
    Command-line interface channel.

    Provides interactive terminal sessions with the agent.
    """

    def __init__(
        self,
        name: str = "cli",
        prompt: str = "You: ",
        session_key: str = "cli_session",
    ):
        """
        Initialize CLI channel.

        Args:
            name: Channel name.
            prompt: Input prompt string.
            session_key: Session identifier.
        """
        super().__init__(name)
        self.prompt = prompt
        self.session_key = session_key
        self._console = Console()
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the CLI channel."""
        self._running = True
        self._task = asyncio.create_task(self._input_loop())
        logger.info("CLI channel started")

    async def stop(self) -> None:
        """Stop the CLI channel."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("CLI channel stopped")

    async def send(self, message: OutboundMessage) -> None:
        """
        Send message to terminal.

        Args:
            message: Outbound message to display.
        """
        self._console.print()
        self._console.print(f"[bold cyan]Agent:[/bold cyan] {message.content}")
        self._console.print()

    async def _input_loop(self) -> None:
        """Main input loop for CLI."""
        self._console.print("[bold green]spoon-bot[/bold green] CLI ready. Type 'exit' to quit.")
        self._console.print()

        while self._running:
            try:
                # Use asyncio.to_thread for blocking input
                user_input = await asyncio.to_thread(
                    Prompt.ask,
                    self.prompt.strip(),
                )

                if not user_input:
                    continue

                # Handle exit commands
                if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
                    self._console.print("[dim]Goodbye![/dim]")
                    self._running = False
                    break

                # Handle clear command
                if user_input.lower() in ("clear", "/clear"):
                    self._console.clear()
                    continue

                # Create and publish message
                message = InboundMessage(
                    content=user_input,
                    channel=self.name,
                    session_key=self.session_key,
                )

                await self.publish(message)

            except (EOFError, KeyboardInterrupt):
                self._console.print("\n[dim]Interrupted.[/dim]")
                self._running = False
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"CLI input error: {e}")
