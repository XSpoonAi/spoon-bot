from io import StringIO
from types import SimpleNamespace

from rich.console import Console

import spoon_bot.cli as cli


def _render_tui(monkeypatch, *messages: str) -> str:
    test_console = Console(
        file=StringIO(),
        force_terminal=False,
        color_system=None,
        width=240,
        record=True,
    )
    monkeypatch.setattr(cli, "console", test_console)

    for message in messages:
        cli._tui_step_sink(SimpleNamespace(record={"message": message}))

    return test_console.export_text()


def test_tui_step_sink_displays_full_agent_reasoning(monkeypatch) -> None:
    thought = (
        "First line of reasoning\n"
        "Second line keeps [literal] brackets\n"
        + ("0123456789" * 30)
        + " END-OF-THINK"
    )

    rendered = _render_tui(monkeypatch, f"💭 Agent reasoning: {thought}")

    assert "First line of reasoning" in rendered
    assert "Second line keeps [literal] brackets" in rendered
    assert "END-OF-THINK" in rendered


def test_format_tool_args_preserves_full_shell_payload() -> None:
    raw_args = '{"command": "git status && git diff", "working_dir": "/tmp/project", "timeout": 120}'

    formatted = cli._format_tool_args("shell", raw_args)

    assert "$ git status && git diff" in formatted
    assert "working_dir: /tmp/project" in formatted
    assert "timeout: 120" in formatted


def test_format_tool_args_formats_write_file_path() -> None:
    raw_args = '{"path": "notes/[todo].md"}'

    formatted = cli._format_tool_args("write_file", raw_args)

    rendered = Console(file=StringIO(), force_terminal=False, color_system=None, record=True)
    rendered.print(formatted, highlight=False)

    assert "→ notes/[todo].md" in rendered.export_text()


def test_tui_step_sink_displays_full_shell_output(monkeypatch) -> None:
    raw_args = '{"command": "printf \'alpha\\nbeta\\ngamma\\ndelta\\nepsilon\\nzeta\'", "working_dir": "/tmp/project"}'
    shell_output = "alpha\nbeta\ngamma\ndelta\nepsilon\nzeta\n[result]"

    rendered = _render_tui(
        monkeypatch,
        f"Tool call: shell({raw_args})",
        f"Tool shell executed with result: {shell_output}",
    )

    assert "$ printf 'alpha" in rendered
    assert "zeta'" in rendered
    assert "working_dir: /tmp/project" in rendered
    assert "alpha" in rendered
    assert "zeta" in rendered
    assert "[result]" in rendered
    assert "more lines" not in rendered
