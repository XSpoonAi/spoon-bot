"""Command-line interface for spoon-bot.

Provides user-friendly CLI with:
- Clear error messages (no stack traces)
- Progress indicators for long operations
- Helpful feedback and suggestions
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from spoon_bot.utils.errors import (
    SpoonBotError,
    ConfigurationError,
    APIError,
    format_user_error,
    get_error_suggestions,
)

app = typer.Typer(
    name="spoon-bot",
    help="Local-first AI agent with native OS tools",
    no_args_is_help=True,
)
console = Console()


def print_error(error: Exception, show_suggestions: bool = True) -> None:
    """Print a user-friendly error message with optional suggestions."""
    user_message = format_user_error(error, include_type=True)

    # Create error panel
    error_text = Text()
    error_text.append(user_message, style="red")

    console.print(Panel(
        error_text,
        title="[bold red]Error[/bold red]",
        border_style="red",
        padding=(0, 1),
    ))

    if show_suggestions:
        suggestions = get_error_suggestions(error)
        if suggestions:
            console.print("\n[bold yellow]Suggestions:[/bold yellow]")
            for suggestion in suggestions[:3]:  # Limit to top 3
                console.print(f"  [dim]-[/dim] {suggestion}")


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[green][bold]Success:[/bold][/green] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[blue][bold]Info:[/bold][/blue] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[yellow][bold]Warning:[/bold][/yellow] {message}")


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


class AgentProgressContext:
    """Context manager for showing agent processing progress."""

    def __init__(self, console: Console):
        self.console = console
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}[/bold green]"),
            transient=True,
            console=console,
        )
        self.task_id = None

    async def __aenter__(self):
        self.progress.start()
        self.task_id = self.progress.add_task("Thinking...", total=None)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.progress.stop()
        return False

    def update(self, description: str):
        """Update the progress description."""
        if self.task_id is not None:
            self.progress.update(self.task_id, description=description)


async def _run_agent(
    message: Optional[str],
    model: Optional[str],
    workspace: Optional[Path],
    max_iterations: int,
) -> None:
    """Internal async agent runner."""
    from dotenv import load_dotenv

    load_dotenv(override=False)

    from spoon_bot.agent.loop import create_agent

    workspace = workspace or get_workspace()

    # Show startup panel
    startup_info = Table.grid(padding=(0, 2))
    startup_info.add_column()
    startup_info.add_column()
    startup_info.add_row("[dim]Workspace:[/dim]", str(workspace))
    if model:
        startup_info.add_row("[dim]Model:[/dim]", model)
    startup_info.add_row("[dim]Max iterations:[/dim]", str(max_iterations))

    console.print(Panel(
        startup_info,
        title="[bold blue]spoon-bot[/bold blue] - Local AI Agent",
        border_style="blue",
    ))

    # Initialize agent with progress indicator
    try:
        with console.status("[bold blue]Initializing agent...[/bold blue]") as status:
            status.update("[bold blue]Loading LLM provider...[/bold blue]")
            agent = await create_agent(
                model=model,
                workspace=workspace,
                max_iterations=max_iterations,
            )
            status.update("[bold blue]Loading tools...[/bold blue]")

        # Show loaded tools count
        tool_count = len(agent.tools)
        skill_count = len(agent.skills)
        print_info(f"Loaded {tool_count} tools, {skill_count} skills")

    except ValueError as e:
        # Configuration error (likely missing API key)
        config_error = ConfigurationError(
            str(e),
            user_message="Unable to initialize the agent. Please check your configuration.",
        )
        print_error(config_error)
        console.print("\n[bold]Quick fix:[/bold]")
        console.print("  [cyan]export ANTHROPIC_API_KEY=your-api-key[/cyan]")
        console.print("  [cyan]spoon-bot agent[/cyan]")
        raise typer.Exit(1)
    except Exception as e:
        print_error(e)
        raise typer.Exit(1)

    if message:
        # One-shot mode
        console.print(f"\n[bold cyan]You:[/bold cyan] {message}\n")

        try:
            async with AgentProgressContext(console) as progress:
                progress.update("Processing your request...")
                response = await agent.process(message)

            console.print(Panel(
                Markdown(response),
                title="[bold green]Response[/bold green]",
                border_style="green",
                padding=(1, 2),
            ))
        except Exception as e:
            print_error(e)
            raise typer.Exit(1)
    else:
        # Interactive REPL
        console.print()
        console.print(Panel(
            "[dim]Type your message and press Enter. "
            "Use [bold]/help[/bold] for commands, [bold]/exit[/bold] to quit.[/dim]",
            title="[bold]Interactive Mode[/bold]",
            border_style="dim",
        ))
        console.print()

        while True:
            try:
                user_input = Prompt.ask("[bold cyan]You[/bold cyan]")

                if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
                    console.print("\n[dim]Goodbye! Thanks for using spoon-bot.[/dim]")
                    break

                if user_input.lower() in ("/clear", "/reset"):
                    agent.clear_history()
                    print_success("Conversation history cleared.")
                    continue

                if user_input.lower() == "/help":
                    help_table = Table(show_header=True, header_style="bold")
                    help_table.add_column("Command")
                    help_table.add_column("Description")
                    help_table.add_row("/help", "Show this help message")
                    help_table.add_row("/exit, /quit", "Exit the agent")
                    help_table.add_row("/clear, /reset", "Clear conversation history")
                    help_table.add_row("/status", "Show agent status")
                    help_table.add_row("/tools", "List available tools")
                    console.print(help_table)
                    continue

                if user_input.lower() == "/status":
                    status_table = Table.grid(padding=(0, 2))
                    status_table.add_row("[dim]Model:[/dim]", agent.model)
                    status_table.add_row("[dim]Tools:[/dim]", str(len(agent.tools)))
                    status_table.add_row("[dim]Skills:[/dim]", str(len(agent.skills)))
                    status_table.add_row("[dim]History:[/dim]", f"{len(agent.get_history())} messages")
                    console.print(Panel(status_table, title="[bold]Agent Status[/bold]"))
                    continue

                if user_input.lower() == "/tools":
                    tools_list = agent.tools.list_tools()
                    console.print(f"\n[bold]Available tools ({len(tools_list)}):[/bold]")
                    for i, tool in enumerate(tools_list, 1):
                        console.print(f"  [dim]{i}.[/dim] {tool}")
                    console.print()
                    continue

                if not user_input.strip():
                    continue

                # Process with progress indicator
                try:
                    async with AgentProgressContext(console) as progress:
                        progress.update("Thinking...")
                        response = await agent.process(user_input)

                    console.print()
                    console.print(Panel(
                        Markdown(response),
                        border_style="dim",
                        padding=(0, 1),
                    ))
                    console.print()

                except APIError as e:
                    print_error(e)
                except Exception as e:
                    print_error(e, show_suggestions=False)
                    console.print("[dim]Try rephrasing your request or use /help for commands.[/dim]")

            except KeyboardInterrupt:
                console.print("\n[dim]Interrupted. Type '/exit' to quit or continue typing.[/dim]")
            except EOFError:
                console.print("\n[dim]Goodbye![/dim]")
                break


@app.command()
def onboard():
    """
    Initialize spoon-bot configuration and workspace.

    Creates the default workspace directory and configuration files.
    Initializes a git repository for version control.
    """
    from spoon_bot.services.git import GitManager

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

    # Initialize git repository
    git_manager = GitManager(workspace)
    if git_manager.is_git_available():
        if git_manager.init():
            console.print(f"[green]✓[/green] Initialized git repository")
        else:
            console.print(f"[yellow]![/yellow] Failed to initialize git repository")
    else:
        console.print(f"[dim]-[/dim] Git not available (optional)")

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
    config: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to config file (YAML)",
    ),
    channels: Optional[str] = typer.Option(
        None, "--channels",
        help="Comma-separated list of channels to enable (e.g., telegram,discord)",
    ),
    cli_enabled: bool = typer.Option(
        True, "--cli/--no-cli",
        help="Enable CLI channel",
    ),
    model: Optional[str] = typer.Option(
        None, "--model",
        help="LLM model name (overrides YAML agent.model)",
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider",
        help="LLM provider (anthropic/openai/openrouter/…) (overrides YAML agent.provider)",
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key",
        help="API key for the LLM provider (overrides YAML / env var)",
    ),
    base_url: Optional[str] = typer.Option(
        None, "--base-url",
        help="Custom LLM API base URL (overrides YAML agent.base_url)",
    ),
    tool_profile: Optional[str] = typer.Option(
        None, "--tool-profile",
        help="Tool profile (core/coding/research/full) (overrides YAML agent.tool_profile)",
    ),
    workspace: Optional[Path] = typer.Option(
        None, "-w", "--workspace",
        help="Workspace directory (overrides YAML agent.workspace)",
    ),
):
    """
    Start spoon-bot in gateway mode (multi-channel server).

    Gateway mode enables multi-platform communication with the agent.
    Supports Telegram, Discord, Feishu, and more.

    Configuration priority: CLI args > YAML agent section > env vars > defaults.

    Examples:
      # Load all channels + agent config from config.yaml
      spoon-bot gateway

      # Override model at runtime
      spoon-bot gateway --provider openrouter --model anthropic/claude-sonnet-4

      # Load specific channels
      spoon-bot gateway --channels telegram,discord

      # Use custom config file
      spoon-bot gateway --config my-config.yaml
    """
    asyncio.run(_run_gateway(
        config=config,
        channels=channels.split(',') if channels else None,
        cli_enabled=cli_enabled,
        model=model,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        tool_profile=tool_profile,
        workspace=workspace,
    ))


def _resolve_provider_api_key(provider: Optional[str]) -> Optional[str]:
    """Read the API key from the standard env var for the given provider."""
    env_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "ollama": None,  # no key needed
    }
    var = env_map.get((provider or "").lower())
    return os.environ.get(var) if var else None


async def _run_gateway(
    config: Optional[Path],
    channels: Optional[list[str]],
    cli_enabled: bool,
    model: Optional[str],
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    tool_profile: Optional[str] = None,
    workspace: Optional[Path] = None,
) -> None:
    """Internal async gateway runner.

    Configuration priority: CLI args > YAML agent section > env vars > defaults.
    """
    from dotenv import load_dotenv

    load_dotenv(override=False)

    from spoon_bot.agent.loop import create_agent
    from spoon_bot.channels.manager import ChannelManager
    from spoon_bot.channels.config import load_agent_config

    # ------------------------------------------------------------------
    # 1. Load agent config from YAML
    # ------------------------------------------------------------------
    try:
        agent_cfg = load_agent_config(config)
    except FileNotFoundError:
        agent_cfg = {}
    except Exception as exc:
        print_warning(f"Could not read agent config from YAML: {exc}")
        agent_cfg = {}

    # ------------------------------------------------------------------
    # 2. Resolve effective values (CLI > YAML > env > defaults)
    # ------------------------------------------------------------------
    effective_provider = (
        provider
        or agent_cfg.get("provider")
        or os.getenv("SPOON_BOT_DEFAULT_PROVIDER")
        # defaults.py fallback happens inside create_agent()
    )
    effective_model = (
        model
        or agent_cfg.get("model")
        or os.getenv("SPOON_BOT_DEFAULT_MODEL")
    )
    effective_base_url = (
        base_url
        or agent_cfg.get("base_url")
        or os.getenv("SPOON_BOT_BASE_URL")
    )
    effective_api_key = (
        api_key
        or agent_cfg.get("api_key")
        or _resolve_provider_api_key(effective_provider)
    )
    effective_workspace = (
        workspace
        or (Path(agent_cfg["workspace"]) if agent_cfg.get("workspace") else None)
        or get_workspace()
    )
    effective_tool_profile = (
        tool_profile
        or agent_cfg.get("tool_profile")
    )
    effective_max_iterations = agent_cfg.get("max_iterations")
    effective_enable_skills = agent_cfg.get("enable_skills", True)

    # ------------------------------------------------------------------
    # 3. Show startup panel with effective values
    # ------------------------------------------------------------------
    startup_info = Table.grid(padding=(0, 2))
    startup_info.add_column()
    startup_info.add_column()
    startup_info.add_row("[dim]Mode:[/dim]", "Gateway (Multi-channel)")
    startup_info.add_row("[dim]Provider:[/dim]", effective_provider or "(default)")
    startup_info.add_row("[dim]Model:[/dim]", effective_model or "(default)")
    startup_info.add_row("[dim]Workspace:[/dim]", str(effective_workspace))
    if config:
        startup_info.add_row("[dim]Config:[/dim]", str(config))
    if effective_tool_profile:
        startup_info.add_row("[dim]Tool profile:[/dim]", effective_tool_profile)
    startup_info.add_row("[dim]CLI:[/dim]", "Enabled" if cli_enabled else "Disabled")

    console.print(Panel(
        startup_info,
        title="[bold blue]spoon-bot Gateway[/bold blue]",
        subtitle="Loading configuration...",
        border_style="blue",
    ))

    # ------------------------------------------------------------------
    # 4. Create agent with merged config
    # ------------------------------------------------------------------
    create_kwargs: dict = dict(
        model=effective_model,
        provider=effective_provider,
        api_key=effective_api_key,
        base_url=effective_base_url,
        workspace=effective_workspace,
        enable_skills=effective_enable_skills,
    )
    if effective_tool_profile is not None:
        create_kwargs["tool_profile"] = effective_tool_profile
    if effective_max_iterations is not None:
        create_kwargs["max_iterations"] = int(effective_max_iterations)

    try:
        with console.status("[bold blue]Initializing agent...[/bold blue]"):
            agent = await create_agent(**create_kwargs)
        print_success(f"Agent initialized: {agent.provider}/{agent.model}")
    except ValueError as e:
        print_error(ConfigurationError(
            str(e),
            user_message="Unable to initialize agent. Check your API keys.",
        ))
        console.print("\n[bold]Quick fix:[/bold]")
        console.print("  [cyan]export OPENROUTER_API_KEY=your-key[/cyan]")
        raise typer.Exit(1)
    except Exception as e:
        print_error(e)
        raise typer.Exit(1)

    # Create channel manager
    manager = ChannelManager()
    manager.set_agent(agent)

    # Load channels from config
    try:
        with console.status("[bold blue]Loading channels...[/bold blue]"):
            await manager.load_from_config(config)

        # Start specific channels if requested
        if channels:
            await manager.start_channels(channels)
        else:
            await manager.start_all()

        # Show loaded channels
        channels_table = Table()
        channels_table.add_column("Channel", style="cyan")
        channels_table.add_column("Account", style="green")
        channels_table.add_column("Status", style="yellow")

        health = await manager.health_check_all()
        for ch_name, ch_health in health["channels"].items():
            status = ch_health.get("status", "unknown")
            status_icon = "✓" if status == "running" else "✗"
            status_color = "green" if status == "running" else "red"

            # Parse channel name (format: type:account)
            parts = ch_name.split(":", 1)
            ch_type = parts[0]
            ch_account = parts[1] if len(parts) > 1 else "default"

            channels_table.add_row(
                ch_type,
                ch_account,
                f"[{status_color}]{status_icon}[/{status_color}] {status}"
            )

        console.print(Panel(
            channels_table,
            title=f"[bold]Active Channels ({health['running_channels']}/{health['total_channels']})[/bold]",
            border_style="green",
        ))

        if health['running_channels'] == 0:
            print_warning("No channels are running. Check your configuration.")
            raise typer.Exit(1)

    except FileNotFoundError:
        print_warning("No config file found. Using defaults.")
        print_warning("Create config with: cp config.example.yaml ~/.spoon-bot/config.yaml")
    except ImportError as e:
        error_msg = str(e)
        print_error(ConfigurationError(
            error_msg,
            user_message="Missing dependencies for configured channels.",
        ))

        # Parse missing dependencies from error message
        if "Missing dependencies for channels:" in error_msg:
            console.print("\n[bold]Install the missing dependencies:[/bold]")
            # Extract suggestion from error message if available
            if "Install with:" in error_msg:
                install_cmd = error_msg.split("Install with:")[1].strip()
                console.print(f"  [cyan]{install_cmd}[/cyan]")
        else:
            console.print("\n[bold]Install missing channels:[/bold]")
            console.print("  [cyan]uv pip install -e \".[telegram]\"[/cyan]  # Telegram")
            console.print("  [cyan]uv pip install -e \".[discord]\"[/cyan]   # Discord")
            console.print("  [cyan]uv pip install -e \".[all-channels]\"[/cyan]  # All")
        raise typer.Exit(1)
    except Exception as e:
        print_error(e)
        raise typer.Exit(1)

    console.print("\n[dim]Gateway running. Press Ctrl+C to stop.[/dim]\n")

    # Keep running
    try:
        while manager.is_running:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        console.print("\n[dim]Shutting down gracefully...[/dim]")
    except Exception as e:
        print_error(e)
    finally:
        with console.status("[bold blue]Stopping channels...[/bold blue]"):
            await manager.stop()

    print_success("Gateway stopped")


@app.command()
def version():
    """Show spoon-bot version."""
    from spoon_bot import __version__
    console.print(f"spoon-bot version {__version__}")


if __name__ == "__main__":
    app()
