import argparse
import asyncio
import shutil
from pathlib import Path
from importlib.resources import files

from hola.cui.basic.chat import DEFAULT_HOST, DEFAULT_MODEL, run_chat
from hola.tools import get_tool, list_tools
from hola.tools.types import ToolDefinition


def get_skill_command(output_dir: str | None = None) -> None:
    """Output the hola agent skill."""
    skill_md = (files('hola') / 'SKILL.md').read_text()

    if output_dir:
        output_path = Path(output_dir) / 'hola'
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / 'SKILL.md').write_text(skill_md)
        print(f"Skill copied to {output_path}/SKILL.md")
    else:
        print(skill_md)

def opencode_skill_command() -> None:
    """Output the hola agent skill for OpenCode."""
    skill_md = (files('hola') / 'SKILL.md').read_text()

    output_dir = '.opencode/skills'
    output_path = Path(output_dir) / 'hola'
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'SKILL.md').write_text(skill_md)
    print(f"Skill copied to {output_path}/SKILL.md")
    print("Restart OpenCode to make it available to agents")

def cursor_skill_command() -> None:
    """Output the hola agent skill for Cursor."""
    skill_md = (files('hola') / 'SKILL.md').read_text()

    output_dir = '.cursor/skills'
    output_path = Path(output_dir) / 'hola'
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'SKILL.md').write_text(skill_md)
    print(f"Skill copied to {output_path}/SKILL.md")

def claude_skill_command() -> None:
    """Output the hola agent skill for Claude Code."""
    skill_md = (files('hola') / 'SKILL.md').read_text()

    output_dir = '.claude/skills'
    output_path = Path(output_dir) / 'hola'
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'SKILL.md').write_text(skill_md)
    print(f"Skill copied to {output_path}/SKILL.md")

def codex_skill_command() -> None:
    """Output the hola agent skill for Codex."""
    skill_md = (files('hola') / 'SKILL.md').read_text()

    output_dir = '.agents/skills'
    output_path = Path(output_dir) / 'hola'
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'SKILL.md').write_text(skill_md)
    print(f"Skill copied to {output_path}/SKILL.md")


def resolve_tools_arg(tools_arg: str | None) -> list[ToolDefinition] | None:
    """Resolve a tools argument string to ToolDefinitions.

    A value of '*' returns all registered tools. Otherwise split on
    commas, validating each name against the registry. If a name is
    invalid, an error prints valid tool names and returns `None`.
    """
    if not tools_arg:
        return None
    if tools_arg == '*':
        return list_tools()
    tool_names = [name.strip() for name in tools_arg.split(',') if name.strip()]
    tools: list[ToolDefinition] = []
    for name in tool_names:
        td = get_tool(name)
        if td is None:
            valid_tools = [t.name for t in list_tools()]
            valid_list = ', '.join(valid_tools)
            print(f"Error: unknown tool '{name}'. Valid tools: {valid_list}")
            return None
        tools.append(td)
    return tools or None


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='hola',
        description='Extensible Ollama Harness for Robust Local Agents'
    )

    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        title='commands',
        description='Subcommands for hola'
    )

    # Create skill subcommand parser (shared for all aliases)
    skill_parser = argparse.ArgumentParser(add_help=False)
    skill_parser.add_argument(
        '--output', '-o',
        help='Output directory (default: print to stdout)',
        default=None
    )

    # Create chat subcommand parser
    chat_parser = argparse.ArgumentParser(add_help=False)
    chat_parser.add_argument(
        '--host',
        help=(
            "Base URL of the OpenAI-compatible inference server "
            f"(default: {DEFAULT_HOST})"
        ),
        default=None,
    )
    chat_parser.add_argument(
        '--model',
        help=f"Model name to request (default: {DEFAULT_MODEL})",
        default=None,
    )
    chat_parser.add_argument(
        '--system',
        help="Optional system prompt",
        default=None,
    )
    chat_parser.add_argument(
        '--api-key',
        help="Optional API key for Authorization: Bearer auth",
        default=None,
    )
    chat_parser.add_argument(
        '--ollama',
        action='store_true',
        help='Connect with the OllamaClient'
    )
    chat_parser.add_argument(
        '--no-thinking',
        action='store_true',
        help='Disable thinking-mode display',
    )
    chat_parser.add_argument(
        '--tools',
        help=(
            "Tools to enable. '*' for all tools, or a comma-separated "
            "list of tool names to validate against the registry"
        ),
        default=None,
    )

    # Add subcommands
    subparsers.add_parser(
        'skill',
        parents=[skill_parser],
        help='skill [--output path] [-O path]: Output the hola agent skill '
            'to stdout or an output path',
        description=f'Output the hola agent skill'
    )

    subparsers.add_parser(
        'opencode',
        parents=[skill_parser],
        help='opencode: Install the hola agent skill compatibly with OpenCode',
        description=f'Output the hola agent skill for OpenCode'
    )

    subparsers.add_parser(
        'claude',
        parents=[skill_parser],
        help='claude: Install the hola agent skill compatibly with Claude Code',
        description=f'Output the hola agent skill for Claude Code'
    )

    subparsers.add_parser(
        'cursor',
        parents=[skill_parser],
        help='cursor: Install the hola agent skill compatibly with Cursor',
        description=f'Output the hola agent skill for Cursor'
    )

    subparsers.add_parser(
        'codex',
        parents=[skill_parser],
        help='codex: Install the hola agent skill compatibly with Codex',
        description=f'Output the hola agent skill for Codex'
    )

    subparsers.add_parser(
        'chat',
        parents=[chat_parser],
        help='chat: Run an asynchronous chat CUI against an '
             'OpenAI-compatible server',
        description=(
            'Run an asynchronous chat CUI against an OpenAI-compatible server'
        ),
    )

    args = parser.parse_args()

    if args.command == 'skill':
        get_skill_command(args.output)
    elif args.command == 'opencode':
        opencode_skill_command()
    elif args.command == 'cursor':
        cursor_skill_command()
    elif args.command == 'claude':
        claude_skill_command()
    elif args.command == 'codex':
        codex_skill_command()
    elif args.command == 'chat':
        host = args.host if args.host is not None else DEFAULT_HOST
        model = args.model if args.model is not None else DEFAULT_MODEL
        show_thinking = not args.no_thinking
        use_ollama = args.ollama
        tools = resolve_tools_arg(args.tools)
        if args.tools is not None and tools is None:
            return
        try:
            asyncio.run(
                run_chat(
                    host, model, args.system, args.api_key,
                    show_thinking, tools, use_ollama,
                )
            )
        except KeyboardInterrupt:
            pass
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
