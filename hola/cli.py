import argparse
import shutil
from pathlib import Path
from importlib.resources import files


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
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
