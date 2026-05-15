"""
Sandboxed Python code execution tool with permission system integration.
"""

from __future__ import annotations
from io import StringIO
from multiprocessing import Process, Queue
from typing import Any
from hola.errors import PermissionDeniedError, VerificationException
from hola.protocols import SessionProtocol
from hola.tools.core import tool
import ast
import builtins
import os
import sys


DEFAULT_ALLOWED_MODULES = [
    # stdlib
    'collections',
    'dataclasses',
    'datetime',
    'decimal',
    'fractions',
    'itertools',
    'json',
    'math',
    'random',
    're',
    'statistics',
    'struct',
    'time',
    # 3rd party libs
    'autodox',
    'flying_state_machines',
    'matplotlib',
    'numpy',
    'packify',
    'pandas',
]


AST_BLACKLIST = [
    ast.Del,
    ast.Delete,
    ast.Nonlocal,
    ast.Global,
]


METHOD_CALL_BLACKLIST = [
    '__getattribute__',
    '__getattr__',
    '__setattr__',
    '__delattr__',
]


ATTRIBUTE_BLACKLIST = [
    '__builtins__',
    '__globals__',
    '__class__',
    '__subclasses__',
    '__bases__',
    '__mro__',
    '__code__',
]


FUNCTION_BLACKLIST = [
    'exec',
    'eval',
    'compile',
    '__import__',
    '__getattr__',
    '__getattribute__',
    'open',
    'getattr',
    'setattr',
    'delattr',
    'exit',
    'quit',
]


def _verify_blacklist(code: str, allow_open: bool = False) -> None:
    """Verify `code` does not violate the AST blacklist. Raises
        `VerificationException` if blacklisted constructs are found.
        The `allow_open` flag permits `open` calls when `True`.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise VerificationException(f"Syntax error in code: {e}")

    for node in ast.walk(tree):
        if type(node) in AST_BLACKLIST:
            raise VerificationException(
                f"Blacklisted AST node: {type(node).__name__}"
            )

        if isinstance(node, ast.Name):
            func_name = node.id
            if func_name in FUNCTION_BLACKLIST:
                if not (allow_open and func_name == 'open'):
                    raise VerificationException(
                        f"Blacklisted function name: {func_name}"
                    )

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in FUNCTION_BLACKLIST:
                if not (allow_open and func_name == 'open'):
                    raise VerificationException(
                        f"Blacklisted function call: {func_name}"
                    )

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in METHOD_CALL_BLACKLIST:
                raise VerificationException(
                    f"Blacklisted method call: {node.func.attr}"
                )

        if isinstance(node, ast.Attribute):
            if node.attr in ATTRIBUTE_BLACKLIST:
                raise VerificationException(
                    f"Blacklisted attribute: {node.attr}"
                )


class _RestrictedImport:
    """Wraps `__import__` to enforce a module allowlist. Raises
        `ImportError` for modules not in the allowlist, including
        submodules.
    """

    def __init__(self, allowed_modules: list[str]) -> None:
        self.allowed_modules = allowed_modules
        self.original_import = builtins.__import__

    def __call__(
            self, name: str, globals: dict | None = None,
            locals: dict | None = None, fromlist: tuple = (),
            level: int = 0
        ) -> Any:
        if name in self.allowed_modules:
            return self.original_import(
                name, globals, locals, fromlist, level
            )

        for module in self.allowed_modules:
            if name.startswith(f"{module}."):
                return self.original_import(
                    name, globals, locals, fromlist, level
                )

        raise ImportError(
            f"Module '{name}' not in allowlist. Allowed modules: "
            f"{self.allowed_modules}"
        )


def _execute(
        code: str,
        allowed_modules: list[str],
        queue: Queue,
        restricted_open = None,
    ) -> None:
    """Execute `code` with restricted environment and write results to `queue`.
        Intended to be used in a subprocess, but this does not spawn a
        subprocess itself.
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()

    restricted_globals = {
        '__builtins__': {
            '__import__': _RestrictedImport(allowed_modules),
            '__build_class__': __build_class__,
            '__name__': '__main__',
            'print': print,
            'len': len,
            'range': range,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'sorted': sorted,
            'enumerate': enumerate,
            'zip': zip,
            'isinstance': isinstance,
            'type': type,
            'Exception': Exception,
            'BaseException': BaseException,
            'ValueError': ValueError,
            'TypeError': TypeError,
            'ZeroDivisionError': ZeroDivisionError,
            'NameError': NameError,
            'IndexError': IndexError,
            'KeyError': KeyError,
            'AttributeError': AttributeError,
            'RuntimeError': RuntimeError,
        },
    }
    if restricted_open:
        restricted_globals['__builtins__']['open'] = restricted_open

    try:
        exec(code, restricted_globals)
        stdout_val = sys.stdout.getvalue()
        stderr_val = sys.stderr.getvalue()
        queue.put({'success': True, 'stdout': stdout_val, 'stderr': stderr_val})
    except Exception as e:
        queue.put({'success': False, 'error': str(e)})
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


class _RestrictedOpenFunc:
    """Permission-aware wrapper for `open()` that checks session
        permissions before allowing file operations.
    """
    def __init__(self, sess: SessionProtocol) -> None:
        self.sess = sess
        self.original_open = builtins.open

    def __call__(
            self, file: str | os.PathLike, mode: str = 'r',
            *args: Any, **kwargs: Any
        ) -> Any:
        perm = self.sess.get_file_permission(file)

        required_read = 'r' in mode or '+' in mode
        required_write = 'w' in mode or 'a' in mode or '+' in mode or 'x' in mode

        if required_read and not perm.read:
            raise PermissionDeniedError(
                f"Read permission denied for {file}"
            )

        if required_write and not perm.write:
            raise PermissionDeniedError(
                f"Write permission denied for {file}"
            )

        return self.original_open(file, mode, *args, **kwargs)


def _execute_with_timeout(
        code: str, timeout: int, allowed_modules: list[str], restricted_open=None
    ) -> tuple[str, str]:
    """Execute `code` with timeout using multiprocessing and permission-aware
        file operations via `session`. Raises `TimeoutError` if execution
        exceeds `timeout` and `VerificationException` for errors. Returns stdout
        and stderr.
    """
    queue = Queue()
    process = Process(
        target=_execute,
        args=(code, allowed_modules, queue, restricted_open)
    )
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join(0.1)
        if process.is_alive():
            process.kill()
        raise TimeoutError(f"Code execution exceeded {timeout}s timeout")

    process.close()

    if queue.empty():
        raise VerificationException("No output from code execution")

    result = queue.get()
    output = []

    if result['success']:
        output.append(result['stdout'])
        output.append(result.get('stderr', ''))
        return tuple(output)
    else:
        raise VerificationException(f"Execution error: {result['error']}")


@tool(
    name="python",
    description="Execute Python code in a sandboxed environment. "
        "Supports imports from an allowed module list and "
        "permission-aware file operations when a session is provided."
)
def python(
        code: str, timeout: int = 5, session: SessionProtocol | None = None,
    ) -> tuple[str, str]:
    """Execute Python code with safety restrictions. Security mechanisms
        include AST-based blacklist validation, module allowlist for
        controlled imports, permission-aware file operations via session,
        and timeout enforcement to prevent infinite loops. Returns the
        stdout and stderr from code execution.
    """
    _verify_blacklist(code, allow_open=(session is not None))

    allowed_modules = DEFAULT_ALLOWED_MODULES
    restricted_open = None
    if session is not None:
        allowed_modules = session.get_allowed_modules() or DEFAULT_ALLOWED_MODULES
        restricted_open = _RestrictedOpenFunc(session)

    return _execute_with_timeout(code, timeout, allowed_modules, restricted_open)
