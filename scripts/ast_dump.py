"""
Dump Python AST node types and names for a given code string.
"""

import ast
import sys


def read_code_input() -> str:
    """Read code from stdin or sys.argv, whichever is supplied."""
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])

    if not sys.stdin.isatty():
        return sys.stdin.read()

    return ""


def print_node(node: ast.AST, depth: int = 0) -> None:
    """Print a single AST node with indentation and its name if available."""
    indent = "  " * depth
    node_type = type(node).__name__
    node_name = getattr(node, "name", None)
    node_id = getattr(node, "id", None)

    if node_name:
        print(f"{indent}{node_type} ({node_name})")
    elif node_id:
        print(f"{indent}{node_type} ({node_id})")
    else:
        print(f"{indent}{node_type}")


def walk_ast(tree: ast.AST, depth: int = 0) -> None:
    """Walk the AST tree and print each node with proper indentation."""
    print_node(tree, depth)

    for field_name, field_value in ast.iter_fields(tree):
        if isinstance(field_value, list):
            for item in field_value:
                if isinstance(item, ast.AST):
                    walk_ast(item, depth + 1)
        elif isinstance(field_value, ast.AST):
            walk_ast(field_value, depth + 1)


def main() -> None:
    """Parse code from stdin or sys.argv and dump its AST structure."""
    code = read_code_input()

    if not code:
        print("No code provided. Supply code via stdin or as argument.",
              file=sys.stderr)
        sys.exit(1)

    print("Code to parse:")
    print("-" * 60)
    print(code)
    print("-" * 60)
    print()

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print(f"Syntax error: {e}", file=sys.stderr)
        sys.exit(1)

    print("AST structure:")
    print("=" * 60)
    walk_ast(tree)


if __name__ == "__main__":
    main()