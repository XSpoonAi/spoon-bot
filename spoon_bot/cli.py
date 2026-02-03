"""Command-line interface for spoon-bot."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

app = typer.Typer(
    name="spoon-bot",
    help="Local-first AI agent with native OS tools",
    no_args_is_help=True,
)
console = Console()


def get_workspace() -> Path:
    """Get the default workspace path."""
    return Path.home() / ".spoon-bot" / "workspace"


@app.command()
def agent(
    message: Optional[str] = typer.Option(
        None, "-m", "--message",
        help="Message to send (one-shot mode)",
    ),
    model: Optional[str] = typer.Option(
        None, "--model",
        help="Model to use (e.g., claude-sonnet-4-20250514)",
    ),
    workspace: Optional[Path] = typer.Option(
        None, "-w", "--workspace",
        help="Workspace directory",
    ),
    max_iterations: int = typer.Option(
        20, "--max-iterations",
        help="Maximum tool call iterations",
    ),
):
    """
    Run the spoon-bot agent.

    If --message is provided, runs in one-shot mode.
    Otherwise, starts an interactive REPL.
    """
    asyncio.run(_run_agent(
        message=message,
        model=model,
        workspace=workspace,
        max_iterations=max_iterations,
    ))


async def _run_agent(
    message: Optional[str],
    model: Optional[str],
    workspace: Optional[Path],
    max_iterations: int,
) -> None:
    """Internal async agent runner."""
    from spoon_bot.agent.loop import create_agent

    workspace = workspace or get_workspace()

    console.print(Panel(
        "[bold blue]spoon-bot[/bold blue] - Local AI Agent",
        subtitle=f"Workspace: {workspace}",
    ))

    try:
        agent = await create_agent(
            model=model,
            workspace=workspace,
            max_iterations=max_iterations,
        )
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("\nPlease set your ANTHROPIC_API_KEY environment variable.")
        raise typer.Exit(1)

    if message:
        # One-shot mode
        console.print(f"\n[dim]You:[/dim] {message}\n")
        with console.status("[bold green]Thinking...[/bold green]"):
            response = await agent.process(message)
        console.print(Markdown(response))
    else:
        # Interactive REPL
        console.print("\nEntering interactive mode. Type 'exit' or 'quit' to exit.\n")

        while True:
            try:
                user_input = Prompt.ask("[bold cyan]You[/bold cyan]")

                if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
                    console.print("[dim]Goodbye![/dim]")
                    break

                if user_input.lower() in ("/clear", "/reset"):
                    agent.clear_history()
                    console.print("[dim]History cleared.[/dim]")
                    continue

                if user_input.lower() == "/help":
                    console.print("""
[bold]Commands:[/bold]
  /exit, /quit  - Exit the agent
  /clear, /reset - Clear conversation history
  /help         - Show this help
                    """)
                    continue

                if not user_input.strip():
                    continue

                with console.status("[bold green]Thinking...[/bold green]"):
                    response = await agent.process(user_input)

                console.print()
                console.print(Markdown(response))
                console.print()

            except KeyboardInterrupt:
                console.print("\n[dim]Interrupted. Type 'exit' to quit.[/dim]")
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")


@app.command()
def onboard():
    """
    Initialize spoon-bot configuration and workspace.

    Creates the default workspace directory and configuration files.
    """
    workspace = get_workspace()
    config_dir = Path.home() / ".spoon-bot"

    console.print(Panel(
        "[bold blue]spoon-bot onboarding[/bold blue]",
        subtitle="Setting up your local AI agent",
    ))

    # Create directories
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "memory").mkdir(exist_ok=True)
    (workspace / "skills").mkdir(exist_ok=True)

    console.print(f"[green]✓[/green] Created workspace: {workspace}")

    # Create AGENTS.md
    agents_file = workspace / "AGENTS.md"
    if not agents_file.exists():
        agents_file.write_text("""# Agent Instructions

- Always explain what you're about to do before using tools
- Ask for confirmation before destructive operations (rm, overwrite)
- Prefer reading files before editing them
- Be concise but thorough in explanations
""")
        console.print(f"[green]✓[/green] Created {agents_file.name}")

    # Create SOUL.md
    soul_file = workspace / "SOUL.md"
    if not soul_file.exists():
        soul_file.write_text("""# Personality

You are a helpful local assistant focused on software development and system tasks.
You communicate concisely and prefer action over lengthy explanations.
You're friendly but professional.
""")
        console.print(f"[green]✓[/green] Created {soul_file.name}")

    # Create MEMORY.md
    memory_file = workspace / "memory" / "MEMORY.md"
    if not memory_file.exists():
        memory_file.write_text("""# Long-term Memory

This file stores persistent facts and preferences.
The agent can add new memories here.

## User Preferences
(To be learned)

## Important Facts
(To be remembered)
""")
        console.print(f"[green]✓[/green] Created memory/MEMORY.md")

    console.print(f"""
[bold green]Setup complete![/bold green]

To start the agent:
  [cyan]spoon-bot agent[/cyan]              # Interactive mode
  [cyan]spoon-bot agent -m "message"[/cyan] # One-shot mode

Make sure to set your API key:
  [cyan]export ANTHROPIC_API_KEY=your-key[/cyan]
""")


@app.command()
def status():
    """Show spoon-bot status and configuration."""
    import os

    workspace = get_workspace()
    config_dir = Path.home() / ".spoon-bot"

    console.print(Panel("[bold blue]spoon-bot status[/bold blue]"))

    # Check workspace
    if workspace.exists():
        console.print(f"[green]✓[/green] Workspace: {workspace}")
    else:
        console.print(f"[yellow]![/yellow] Workspace not found: {workspace}")

    # Check API key
    if os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[green]✓[/green] ANTHROPIC_API_KEY is set")
    else:
        console.print("[yellow]![/yellow] ANTHROPIC_API_KEY not set")

    # Check bootstrap files
    for filename in ["AGENTS.md", "SOUL.md"]:
        file_path = workspace / filename
        if file_path.exists():
            console.print(f"[green]✓[/green] {filename} exists")
        else:
            console.print(f"[dim]-[/dim] {filename} not found")

    # Check memory
    memory_file = workspace / "memory" / "MEMORY.md"
    if memory_file.exists():
        console.print(f"[green]✓[/green] Memory file exists")
    else:
        console.print(f"[dim]-[/dim] Memory file not found")


@app.command()
def gateway(
    telegram_token: Optional[str] = typer.Option(
        None, "--telegram", envvar="TELEGRAM_BOT_TOKEN",
        help="Telegram bot token",
    ),
    cli_enabled: bool = typer.Option(
        True, "--cli/--no-cli",
        help="Enable CLI channel",
    ),
    model: Optional[str] = typer.Option(
        None, "--model",
        help="Model to use",
    ),
    workspace: Optional[Path] = typer.Option(
        None, "-w", "--workspace",
        help="Workspace directory",
    ),
):
    """
    Start spoon-bot in gateway mode (24/7 server).

    Gateway mode enables multi-channel communication with the agent.
    Use --telegram to enable Telegram bot integration.
    """
    asyncio.run(_run_gateway(
        telegram_token=telegram_token,
        cli_enabled=cli_enabled,
        model=model,
        workspace=workspace,
    ))


async def _run_gateway(
    telegram_token: Optional[str],
    cli_enabled: bool,
    model: Optional[str],
    workspace: Optional[Path],
) -> None:
    """Internal async gateway runner."""
    from spoon_bot.agent.loop import create_agent
    from spoon_bot.channels.manager import ChannelManager
    from spoon_bot.channels.cli_channel import CLIChannel

    workspace = workspace or get_workspace()

    console.print(Panel(
        "[bold blue]spoon-bot Gateway[/bold blue]",
        subtitle=f"Multi-channel mode | Workspace: {workspace}",
    ))

    # Create agent
    try:
        agent = await create_agent(
            model=model,
            workspace=workspace,
        )
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("\nPlease set your ANTHROPIC_API_KEY environment variable.")
        raise typer.Exit(1)

    # Create channel manager
    manager = ChannelManager()
    manager.set_agent(agent)

    # Add CLI channel if enabled
    if cli_enabled:
        cli_channel = CLIChannel()
        manager.add_channel(cli_channel)
        console.print("[green]✓[/green] CLI channel enabled")

    # Add Telegram channel if token provided
    if telegram_token:
        try:
            from spoon_bot.channels.telegram_channel import TelegramChannel
            telegram_channel = TelegramChannel(token=telegram_token)
            manager.add_channel(telegram_channel)
            console.print("[green]✓[/green] Telegram channel enabled")
        except ImportError:
            console.print(
                "[yellow]![/yellow] python-telegram-bot not installed. "
                "Install with: pip install python-telegram-bot"
            )

    if not manager.channel_names:
        console.print("[red]Error:[/red] No channels enabled")
        raise typer.Exit(1)

    console.print(f"\nActive channels: {', '.join(manager.channel_names)}")
    console.print("\n[dim]Press Ctrl+C to stop[/dim]\n")

    # Start gateway
    try:
        await manager.start()

        # Keep running until interrupted
        while manager.is_running:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        console.print("\n[dim]Shutting down...[/dim]")
    finally:
        await manager.stop()

    console.print("[green]Gateway stopped[/green]")


@app.command()
def version():
    """Show spoon-bot version."""
    from spoon_bot import __version__
    console.print(f"spoon-bot version {__version__}")


if __name__ == "__main__":
    app()
