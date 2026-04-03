"""Command-line interface for spoon-bot.

Provides user-friendly CLI with:
- Clear error messages (no stack traces)
- Progress indicators for long operations
- Helpful feedback and suggestions
"""

import asyncio
import ctypes
import os
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
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


def _restore_console_input_mode() -> None:
    """Restore the Windows console input handle to standard line-buffered mode.

    Rich, subprocess, or ANSI escape processing can leave the console input
    handle in an unexpected mode (e.g. raw or VT-input) which causes
    ``sys.stdin.readline()`` to hang.  Calling this before each prompt
    guarantees the handle is back to normal.  No-op on non-Windows.
    """
    if sys.platform != "win32":
        return
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        STD_INPUT_HANDLE = ctypes.c_ulong(-10 & 0xFFFFFFFF)
        handle = kernel32.GetStdHandle(STD_INPUT_HANDLE)
        if handle == -1:
            return
        ENABLE_PROCESSED_INPUT = 0x0001
        ENABLE_LINE_INPUT = 0x0002
        ENABLE_ECHO_INPUT = 0x0004
        default_mode = ENABLE_PROCESSED_INPUT | ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT
        kernel32.SetConsoleMode(handle, default_mode)
    except Exception:
        pass


async def _read_stdin_line() -> str | None:
    """Read one line from stdin with Ctrl+C support on Windows.

    Plain ``asyncio.to_thread(sys.stdin.readline)`` spawns a thread that
    cannot be interrupted by Ctrl+C on Windows (the thread blocks in a
    Win32 ``ReadConsoleW`` call that ignores Python signals).

    This helper uses a daemon thread + asyncio.Queue so the event loop
    can service ``KeyboardInterrupt`` between short poll intervals.
    Returns ``None`` on EOF.
    """
    if sys.platform != "win32":
        line = await asyncio.to_thread(sys.stdin.readline)
        return line if line else None

    import threading

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)

    def _reader() -> None:
        try:
            line = sys.stdin.readline()
            loop.call_soon_threadsafe(queue.put_nowait, line)
        except Exception:
            loop.call_soon_threadsafe(queue.put_nowait, "")

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    try:
        while True:
            try:
                line = await asyncio.wait_for(queue.get(), timeout=0.3)
                return line if line else None
            except asyncio.TimeoutError:
                continue
    except (KeyboardInterrupt, asyncio.CancelledError):
        raise KeyboardInterrupt


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
        help="Workspace directory",
    ),
    max_iterations: int = typer.Option(
        50, "--max-iterations",
        help="Maximum tool call iterations",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Show detailed step-by-step agent logs",
    ),
):
    """
    Run the spoon-bot agent.

    If --message is provided, runs in one-shot mode.
    Otherwise, starts an interactive REPL.

    Configuration priority: CLI args > YAML agent section > env vars.
    """
    _configure_logging(verbose)
    asyncio.run(_run_agent(
        message=message,
        model=model,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        tool_profile=tool_profile,
        workspace=workspace,
        max_iterations=max_iterations,
    ))


class AgentProgressContext:
    """Context manager for showing agent processing progress.

    Uses a simple stdout-based spinner instead of Rich's Progress/Live
    renderer.  Rich's Live rendering corrupts the Windows console input
    mode, causing ``sys.stdin.readline()`` to hang on subsequent calls
    and making Ctrl+C unresponsive.  Writing spinner frames directly via
    ``sys.stdout`` avoids all Rich lock / terminal-state issues.
    """

    _FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self, console: Console):
        self.console = console
        self._description = "Thinking..."
        self._task: asyncio.Task | None = None

    async def __aenter__(self):
        self._task = asyncio.create_task(self._spin())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Clear the spinner line
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()
        return False

    async def _spin(self):
        idx = 0
        try:
            while True:
                frame = self._FRAMES[idx % len(self._FRAMES)]
                sys.stdout.write(f"\r\033[K  {frame} {self._description}")
                sys.stdout.flush()
                idx += 1
                await asyncio.sleep(0.08)
        except asyncio.CancelledError:
            pass

    def update(self, description: str):
        self._description = description


import re as _re

_STEP_RE = _re.compile(r"Agent \S+ is running step (\d+)/(\d+)")
_TOOL_CALL_RE = _re.compile(r"Tool call: (\S+)\((.*)?\)", _re.DOTALL)
_TOOL_RESULT_RE = _re.compile(r"Tool (\S+) executed with result: (.+)", _re.DOTALL)
_THOUGHT_RE = _re.compile(r"💭 Agent reasoning: (.+)", _re.DOTALL)
_ANSI_RE = _re.compile(r"\x1b\[[0-9;]*m")
_OBSERVED_PREFIX = _re.compile(r"^Observed output of cmd \S+ execution:\s*")


def _tui_step_filter(record):
    """Only pass through agent step/tool/thought messages in TUI mode."""
    msg = record["message"]
    return bool(
        _STEP_RE.search(msg)
        or _THOUGHT_RE.search(msg)
        or _TOOL_CALL_RE.search(msg)
        or _TOOL_RESULT_RE.search(msg)
    )


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _truncate(text: str, max_len: int = 200) -> str:
    text = _strip_ansi(text)
    text = _OBSERVED_PREFIX.sub("", text)
    first_line = text.split("\n", 1)[0].strip()
    if len(first_line) > max_len:
        return first_line[: max_len - 3] + "..."
    return first_line


def _format_tool_args(tool_name: str, raw_args: str) -> str:
    """Extract a human-readable summary from raw JSON tool args."""
    import json as _json

    if not raw_args:
        return ""
    try:
        obj = _json.loads(raw_args)
    except (ValueError, TypeError):
        return f"[dim]{_truncate(raw_args, 100)}[/dim]"

    if not isinstance(obj, dict):
        return f"[dim]{_truncate(raw_args, 100)}[/dim]"

    if tool_name == "shell":
        cmd = obj.get("command", "")
        return f"\n      [dim]$ {cmd}[/dim]" if cmd else ""
    if tool_name == "skill_marketplace":
        action = obj.get("action", "")
        url = obj.get("url", obj.get("skill_name", ""))
        if url:
            return f"[dim]{action} → {url}[/dim]"
        return f"[dim]{action}[/dim]"
    if tool_name in ("read_file", "read_text_file"):
        return f"[dim]{obj.get('path', '')}[/dim]"
    if tool_name == "write_file":
        return f"[dim]→ {obj.get('path', obj.get('file_path', ''))}[/dim]"
    if tool_name == "edit_file":
        return f"[dim]{obj.get('path', '')}[/dim]"
    if tool_name in ("list_dir", "list_directory"):
        return f"[dim]{obj.get('path', '')}[/dim]"
    if tool_name == "grep":
        pat = obj.get("pattern", "")
        path = obj.get("path", "")
        return f"[dim]{pat} in {path}[/dim]"
    if tool_name == "self_upgrade":
        return f"[dim]{obj.get('action', '')}[/dim]"

    parts = []
    for k, v in list(obj.items())[:4]:
        sv = str(v)
        if len(sv) > 60:
            sv = sv[:57] + "..."
        parts.append(f"{k}={sv}")
    return "[dim]" + ", ".join(parts) + "[/dim]"


def _tui_step_sink(message):
    """Render agent step logs as detailed TUI lines."""
    text = message.record["message"]

    m = _STEP_RE.search(text)
    if m:
        cur, total = m.group(1), m.group(2)
        console.print(f"\n  [bold cyan]●[/bold cyan] [bold]Step {cur}/{total}[/bold]", highlight=False)
        return

    m = _THOUGHT_RE.search(text)
    if m:
        thought = m.group(1).strip()
        if thought:
            display = thought[:200] + "…" if len(thought) > 200 else thought
            console.print(f"    [dim]💭[/dim] [italic]{display}[/italic]", highlight=False)
        return

    m = _TOOL_CALL_RE.search(text)
    if m:
        name = m.group(1)
        args = (m.group(2) or "").strip()
        args_display = _format_tool_args(name, args)
        if args_display:
            console.print(f"    [dim]╭─[/dim] [yellow]{name}[/yellow] {args_display}", highlight=False)
        else:
            console.print(f"    [dim]╭─[/dim] [yellow]{name}[/yellow]", highlight=False)
        return

    m = _TOOL_RESULT_RE.search(text)
    if m:
        tool_name = m.group(1)
        raw = m.group(2).strip()
        raw = _strip_ansi(raw)
        raw = _OBSERVED_PREFIX.sub("", raw)
        if not raw.strip():
            return
        if raw.startswith("```diff") or raw.startswith("---"):
            console.print("    [dim]╰→[/dim] [green]edit applied[/green]", highlight=False)
            return

        lines = raw.split("\n")
        max_lines = 5 if tool_name == "shell" else 3
        display_lines = lines[:max_lines]
        display = "\n".join(ln.strip() for ln in display_lines)
        if len(display) > 500:
            display = display[:500] + "..."
        if len(lines) > max_lines:
            display += f"\n      ... ({len(lines) - max_lines} more lines)"

        if "Error" in display or "failed" in display.lower() or "Security Error" in display:
            style = "[red]"
            end_style = "[/red]"
        elif "SUCCESS" in display or "Successfully" in display:
            style = "[green]"
            end_style = "[/green]"
        else:
            style = ""
            end_style = ""

        if "\n" in display:
            console.print(f"    [dim]╰→[/dim] {style}{display}{end_style}", highlight=False)
        else:
            console.print(f"    [dim]╰→[/dim] {style}{display}{end_style}", highlight=False)
        return



_log_file_id: int | None = None
_logging_configured: bool = False


def _configure_logging(verbose: bool = False, workspace: "Path | None" = None) -> None:
    """Set loguru output level and optionally add a file sink for local logs.

    Called twice: once early (console only) and once after workspace is resolved
    (adds the file sink).  The second call only adds the file handler; it does
    NOT re-remove + re-add console handlers.
    """
    global _log_file_id, _logging_configured
    from loguru import logger as _logger

    if not _logging_configured:
        import sys as _sys
        _logger.remove()
        if verbose:
            _logger.add(_sys.stderr, level="DEBUG")
        else:
            _logger.add(
                _tui_step_sink,
                level="INFO",
                filter=_tui_step_filter,
                format="{message}",
            )
        _logging_configured = True

    # Add workspace log file (only once)
    if workspace and _log_file_id is None:
        log_dir = Path(workspace) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"agent_{ts}.log"

        def _log_file_format(record):
            from spoon_bot.utils.privacy import mask_secrets
            msg = record["message"]
            msg = mask_secrets(msg)
            if len(msg) > 8000:
                msg = msg[:8000] + f"\n... [truncated {len(msg) - 8000} chars]"
            msg = msg.replace("{", "{{").replace("}", "}}").replace("<", r"\<")
            ts_str = record["time"].strftime("%H:%M:%S.%f")[:12]
            lvl = record["level"].name
            return f"{ts_str} | {lvl:<7} | {msg}\n"

        _log_file_id = _logger.add(
            str(log_path),
            level="DEBUG",
            format=_log_file_format,
            rotation="10 MB",
            retention="7 days",
            encoding="utf-8",
            colorize=False,
        )
        console.print(f"  [dim]Logs → {log_path}[/dim]", highlight=False)


async def _run_agent(
    message: Optional[str],
    model: Optional[str],
    provider: Optional[str],
    api_key: Optional[str],
    base_url: Optional[str],
    tool_profile: Optional[str],
    workspace: Optional[Path],
    max_iterations: int,
) -> None:
    """Internal async agent runner.

    Configuration priority: CLI args > YAML > env vars.
    All YAML/env resolution is handled by load_agent_config().
    """
    from dotenv import load_dotenv
    from loguru import logger

    load_dotenv(override=False)

    from spoon_bot.agent.loop import create_agent
    from spoon_bot.channels.config import load_agent_config

    # ------------------------------------------------------------------
    # 1. Load agent config (YAML > env vars) — centralized resolution
    # ------------------------------------------------------------------
    try:
        agent_cfg = load_agent_config()
    except Exception:
        agent_cfg = {}

    # ------------------------------------------------------------------
    # 2. Overlay CLI args on top (CLI args take highest priority)
    # ------------------------------------------------------------------
    effective_provider = provider or agent_cfg.get("provider")
    effective_model = model or agent_cfg.get("model")
    effective_base_url = base_url or agent_cfg.get("base_url")
    effective_api_key = api_key or agent_cfg.get("api_key")
    effective_workspace = (
        workspace
        or (Path(agent_cfg["workspace"]) if agent_cfg.get("workspace") else None)
        or get_workspace()
    )
    effective_tool_profile = tool_profile or agent_cfg.get("tool_profile")
    effective_max_iterations = max_iterations or agent_cfg.get("max_iterations", 50)
    effective_enable_skills = agent_cfg.get("enable_skills", True)

    # Ensure workspace directory exists
    effective_workspace.mkdir(parents=True, exist_ok=True)

    # Set up workspace-local log file
    _configure_logging(verbose=False, workspace=effective_workspace)

    # ------------------------------------------------------------------
    # 3. Show startup header (opencode-style compact)
    # ------------------------------------------------------------------
    _model_display = effective_model or "(not set)"
    _provider_display = effective_provider or ""
    _ws_short = str(effective_workspace).replace(str(Path.home()), "~")
    console.print()
    console.rule("[bold blue]spoon-bot[/bold blue]", style="dim")
    if _provider_display:
        console.print(f"  [dim]{_provider_display} /[/dim] [bold]{_model_display}[/bold]", highlight=False)
    else:
        console.print(f"  [bold]{_model_display}[/bold]", highlight=False)
    console.print(f"  [dim]{_ws_short}[/dim]", highlight=False)

    # ------------------------------------------------------------------
    # 4. Create agent with merged config
    # ------------------------------------------------------------------
    create_kwargs: dict = dict(
        model=effective_model,
        provider=effective_provider,
        api_key=effective_api_key,
        base_url=effective_base_url,
        workspace=effective_workspace,
        max_iterations=effective_max_iterations,
        enable_skills=effective_enable_skills,
    )
    if agent_cfg.get("mcp_config") is not None:
        create_kwargs["mcp_config"] = agent_cfg["mcp_config"]
    if agent_cfg.get("shell_timeout") is not None:
        create_kwargs["shell_timeout"] = int(agent_cfg["shell_timeout"])
    if agent_cfg.get("max_output") is not None:
        create_kwargs["max_output"] = int(agent_cfg["max_output"])
    if agent_cfg.get("context_window") is not None:
        create_kwargs["context_window"] = int(agent_cfg["context_window"])
    if agent_cfg.get("enabled_tools") is not None:
        create_kwargs["enabled_tools"] = set(agent_cfg["enabled_tools"])
    if effective_tool_profile is not None:
        create_kwargs["tool_profile"] = effective_tool_profile
    if agent_cfg.get("auto_reload"):
        create_kwargs["auto_reload"] = True
        if agent_cfg.get("auto_reload_interval") is not None:
            create_kwargs["auto_reload_interval"] = float(agent_cfg["auto_reload_interval"])

    try:
        console.print("  [dim]Initializing...[/dim]", end="")
        agent = await create_agent(**create_kwargs)
        tool_count = len(agent.tools)
        skill_count = len(agent.skills)
        console.print(f"\r  [green]Ready[/green] — {tool_count} tools · {skill_count} skills")
        if os.environ.get("SPOON_BOT_WALLET_AUTO_CREATED") == "1":
            wallet_addr = os.environ.get("WALLET_ADDRESS", "").strip()
            wallet_note = f" ({wallet_addr})" if wallet_addr else ""
            console.print(f"  [yellow]Wallet auto-created[/yellow]{wallet_note}")
        console.rule(style="dim")

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

    # ------------------------------------------------------------------
    # Start channels in background (interactive REPL only).
    # Channels (Telegram, Discord, etc.) run via asyncio tasks, so the
    # event loop must not be blocked.  For one-shot mode channels are
    # skipped because the process exits immediately after the response.
    # ------------------------------------------------------------------
    manager = None
    if not message:
        try:
            from spoon_bot.bootstrap import init_channels

            manager = await init_channels(agent)
            count = manager.running_channels_count
            if count > 0:
                print_info(f"Channels running: {count}")
        except FileNotFoundError:
            pass  # No config file — channels are optional in agent mode
        except ImportError as e:
            print_warning(f"Some channel dependencies missing: {e}")
        except Exception as e:
            logger.debug(f"Could not start channels: {e}")

    try:
        if message:
            # One-shot mode — start with a clean session so stale history
            # from previous runs does not confuse the agent.
            agent.clear_history()
            console.print(f"\n  [bold cyan]>[/bold cyan] {message}\n")

            try:
                async with AgentProgressContext(console) as progress:
                    progress.update("Processing your request...")
                    response = await agent.process(message)

                console.print()
                console.rule(style="dim")
                console.print()
                console.print(Markdown(response), highlight=False)
                console.print()
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
                    _restore_console_input_mode()
                    sys.stdout.write("\033[1;36mYou\033[0m: ")
                    sys.stdout.flush()
                    line = await _read_stdin_line()
                    if line is None:
                        console.print("\n[dim]Goodbye![/dim]")
                        break
                    user_input = line.strip()

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
    finally:
        if manager:
            await manager.stop()
        # Clean up agent resources (MCP connections, skills, etc.)
        await agent.cleanup()


@app.command()
def onboard():
    """
    Initialize spoon-bot configuration and workspace.

    Creates the default workspace directory and configuration files.
    Initializes a git repository for version control.
    """
    from spoon_bot.services.git import GitManager
    from spoon_bot.skills.builtin import ensure_builtin_skills

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
    installed_skills = ensure_builtin_skills(workspace)

    console.print(f"[green]✓[/green] Created workspace: {workspace}")
    for skill_dir in installed_skills:
        console.print(f"[green]✓[/green] Installed built-in skill: {skill_dir.name}")

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
    cli_enabled: Optional[bool] = typer.Option(
        None, "--cli/--no-cli",
        help="Override CLI channel (default: use config)",
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


async def _run_gateway(
    config: Optional[Path],
    channels: Optional[list[str]],
    cli_enabled: Optional[bool],
    model: Optional[str],
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    tool_profile: Optional[str] = None,
    workspace: Optional[Path] = None,
) -> None:
    """Internal async gateway runner.

    Configuration priority: CLI args > YAML > env vars.
    All YAML/env resolution is handled by load_agent_config().
    """
    from dotenv import load_dotenv

    load_dotenv(override=False)

    from spoon_bot.agent.loop import create_agent
    from spoon_bot.channels.manager import ChannelManager
    from spoon_bot.channels.config import load_agent_config

    # ------------------------------------------------------------------
    # 1. Load agent config (YAML > env vars) — centralized resolution
    # ------------------------------------------------------------------
    try:
        agent_cfg = load_agent_config(config)
    except FileNotFoundError:
        agent_cfg = {}
    except Exception as exc:
        print_warning(f"Could not read agent config: {exc}")
        agent_cfg = {}

    # ------------------------------------------------------------------
    # 2. Overlay CLI args on top (CLI args take highest priority)
    # ------------------------------------------------------------------
    effective_provider = provider or agent_cfg.get("provider")
    effective_model = model or agent_cfg.get("model")
    effective_base_url = base_url or agent_cfg.get("base_url")
    effective_api_key = api_key or agent_cfg.get("api_key")
    effective_workspace = (
        workspace
        or (Path(agent_cfg["workspace"]) if agent_cfg.get("workspace") else None)
        or get_workspace()
    )
    effective_tool_profile = tool_profile or agent_cfg.get("tool_profile")
    effective_max_iterations = agent_cfg.get("max_iterations")
    effective_enable_skills = agent_cfg.get("enable_skills", True)

    # Set up workspace-local log file
    effective_workspace.mkdir(parents=True, exist_ok=True)
    _configure_logging(workspace=effective_workspace)

    # ------------------------------------------------------------------
    # 3. Show startup panel with effective values
    # ------------------------------------------------------------------
    startup_info = Table.grid(padding=(0, 2))
    startup_info.add_column()
    startup_info.add_column()
    startup_info.add_row("[dim]Mode:[/dim]", "Gateway (Multi-channel)")
    startup_info.add_row("[dim]Provider:[/dim]", effective_provider or "(not set)")
    startup_info.add_row("[dim]Model:[/dim]", effective_model or "(not set)")
    startup_info.add_row("[dim]Workspace:[/dim]", str(effective_workspace))
    if config:
        startup_info.add_row("[dim]Config:[/dim]", str(config))
    if effective_tool_profile:
        startup_info.add_row("[dim]Tool profile:[/dim]", effective_tool_profile)
    cli_label = {True: "Force enabled", False: "Force disabled", None: "Config default"}[cli_enabled]
    startup_info.add_row("[dim]CLI channel:[/dim]", cli_label)

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
    if agent_cfg.get("mcp_config") is not None:
        create_kwargs["mcp_config"] = agent_cfg["mcp_config"]
    if agent_cfg.get("shell_timeout") is not None:
        create_kwargs["shell_timeout"] = int(agent_cfg["shell_timeout"])
    if agent_cfg.get("max_output") is not None:
        create_kwargs["max_output"] = int(agent_cfg["max_output"])
    if agent_cfg.get("context_window") is not None:
        create_kwargs["context_window"] = int(agent_cfg["context_window"])
    if agent_cfg.get("enabled_tools") is not None:
        create_kwargs["enabled_tools"] = set(agent_cfg["enabled_tools"])
    if agent_cfg.get("session_store_backend") is not None:
        create_kwargs["session_store_backend"] = agent_cfg["session_store_backend"]
    if agent_cfg.get("session_store_dsn") is not None:
        create_kwargs["session_store_dsn"] = agent_cfg["session_store_dsn"]
    if agent_cfg.get("session_store_db_path") is not None:
        create_kwargs["session_store_db_path"] = agent_cfg["session_store_db_path"]
    if effective_tool_profile is not None:
        create_kwargs["tool_profile"] = effective_tool_profile
    if effective_max_iterations is not None:
        create_kwargs["max_iterations"] = int(effective_max_iterations)
    # Auto-reload is especially useful in long-running gateway mode
    if agent_cfg.get("auto_reload"):
        create_kwargs["auto_reload"] = True
        if agent_cfg.get("auto_reload_interval") is not None:
            create_kwargs["auto_reload_interval"] = float(agent_cfg["auto_reload_interval"])

    try:
        with console.status("[bold blue]Initializing agent...[/bold blue]"):
            agent = await create_agent(**create_kwargs)
        print_success(f"Agent initialized: {agent.provider}/{agent.model}")
        if os.environ.get("SPOON_BOT_WALLET_AUTO_CREATED") == "1":
            wallet_addr = os.environ.get("WALLET_ADDRESS", "").strip()
            wallet_note = f" ({wallet_addr})" if wallet_addr else ""
            print_info(f"Wallet auto-created{wallet_note}")
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

        # Enforce --cli/--no-cli override (None = use config as-is)
        if cli_enabled is False:
            manager.remove_channel("cli:default")
        elif cli_enabled is True and "cli:default" not in manager.channel_names:
            from spoon_bot.channels.cli_channel import CLIChannel
            manager.add_channel(CLIChannel())

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
                console.print(f"  {install_cmd}", highlight=False)
        else:
            console.print("\n[bold]Install missing channels:[/bold]")
            console.print("  uv pip install -e \".\\[telegram]\"   # Telegram", highlight=False)
            console.print("  uv pip install -e \".\\[discord]\"    # Discord", highlight=False)
            console.print("  uv pip install -e \".\\[all-channels]\"  # All", highlight=False)
        raise typer.Exit(1)
    except Exception as e:
        print_error(e)
        raise typer.Exit(1)

    console.print("\n[dim]Gateway running. Press Ctrl+C to stop.[/dim]\n")

    # Graceful shutdown: use signal handler so Ctrl+C cleanly stops the
    # gateway.  On Windows, loop.add_signal_handler is unsupported so we
    # use signal.signal + loop.call_soon_threadsafe to set an asyncio
    # Event from the signal-handler context.  Double Ctrl+C forces exit.
    import signal

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()
    _force_next = False

    def _on_signal(*_args: Any) -> None:
        nonlocal _force_next
        if _force_next:
            # Second Ctrl+C — force-kill immediately
            os._exit(1)
        _force_next = True
        # Schedule set() on the event loop (signal-handler safe)
        loop.call_soon_threadsafe(shutdown_event.set)

    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _on_signal)
    try:
        signal.signal(signal.SIGTERM, _on_signal)
    except (OSError, ValueError):
        pass  # SIGTERM may not exist on Windows

    # Keep running until shutdown is requested
    try:
        while manager.is_running and not shutdown_event.is_set():
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

    except (KeyboardInterrupt, asyncio.CancelledError):
        pass  # Caught by signal handler above
    except Exception as e:
        print_error(e)
    finally:
        console.print("\n[dim]Shutting down gracefully...[/dim]")
        # Signal agent to stop any in-progress work
        if hasattr(agent, '_agent') and agent._agent:
            if hasattr(agent._agent, '_shutdown_event'):
                agent._agent._shutdown_event.set()
        # Stop all channels, message bus, and clean up resources
        with console.status("[bold blue]Stopping channels...[/bold blue]"):
            await manager.stop()
        with console.status("[bold blue]Cleaning up agent...[/bold blue]"):
            await agent.cleanup()
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_sigint)

    print_success("Gateway stopped")


@app.command()
def version():
    """Show spoon-bot version."""
    from spoon_bot import __version__
    console.print(f"spoon-bot version {__version__}")


# ---------------------------------------------------------------------------
# Service commands
# ---------------------------------------------------------------------------

service_app = typer.Typer(
    name="service",
    help="Manage spoon-bot as a background service (no Docker required).",
    no_args_is_help=True,
)
app.add_typer(service_app)


@service_app.command("start")
def service_start(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to config.yaml (defaults to ~/.spoon-bot/config.yaml)",
    ),
) -> None:
    """Start the gateway in the background."""
    from spoon_bot.services.daemon import start

    with console.status("[bold blue]Starting service...[/bold blue]"):
        ok, msg = start(config)

    if ok:
        print_success(msg)
    else:
        print_error(Exception(msg))
        raise typer.Exit(1)


@service_app.command("stop")
def service_stop() -> None:
    """Stop the background service."""
    from spoon_bot.services.daemon import stop

    with console.status("[bold blue]Stopping service...[/bold blue]"):
        ok, msg = stop()

    if ok:
        print_success(msg)
    else:
        print_error(Exception(msg))
        raise typer.Exit(1)


@service_app.command("restart")
def service_restart(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to config.yaml",
    ),
) -> None:
    """Restart the background service."""
    from spoon_bot.services.daemon import restart

    with console.status("[bold blue]Restarting service...[/bold blue]"):
        ok, msg = restart(config)

    if ok:
        print_success(msg)
    else:
        print_error(Exception(msg))
        raise typer.Exit(1)


@service_app.command("status")
def service_status() -> None:
    """Show the service status."""
    from spoon_bot.services.daemon import get_status

    info = get_status()

    status_icon = "[green]●[/green]" if info["running"] else "[red]●[/red]"
    status_text = "[green]running[/green]" if info["running"] else "[red]stopped[/red]"

    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("Status:", f"{status_icon} {status_text}")
    if info["pid"]:
        table.add_row("PID:", str(info["pid"]))
    table.add_row("Auto-start:", "[green]installed[/green]" if info["auto_start"] else "[dim]not installed[/dim]")
    table.add_row("Log file:", info["log_file"])
    table.add_row("PID file:", info["pid_file"])

    console.print(Panel(table, title="[bold]spoon-bot Service[/bold]"))


@service_app.command("logs")
def service_logs(
    lines: int = typer.Option(50, "-n", "--lines", help="Number of lines to show"),
    follow: bool = typer.Option(False, "-f", "--follow", help="Stream new log output"),
) -> None:
    """View service logs."""
    from spoon_bot.services.daemon import tail_logs

    tail_logs(lines=lines, follow=follow)


@service_app.command("install")
def service_install(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to config.yaml to use at startup",
    ),
) -> None:
    """Install spoon-bot to auto-start at user login.

    \b
    Windows → Windows Task Scheduler (no admin required)
    Linux   → systemd user service (~/.config/systemd/user/)
    macOS   → launchd agent (~/Library/LaunchAgents/)
    """
    from spoon_bot.services.daemon import install_auto_start

    with console.status("[bold blue]Installing auto-start...[/bold blue]"):
        ok, msg = install_auto_start(config)

    if ok:
        print_success(msg)
        console.print("\n[dim]Run [bold]spoon-bot service start[/bold] to start immediately.[/dim]")
    else:
        print_error(Exception(msg))
        raise typer.Exit(1)


@service_app.command("uninstall")
def service_uninstall() -> None:
    """Remove auto-start registration."""
    from spoon_bot.services.daemon import uninstall_auto_start

    with console.status("[bold blue]Removing auto-start...[/bold blue]"):
        ok, msg = uninstall_auto_start()

    if ok:
        print_success(msg)
    else:
        print_error(Exception(msg))
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
