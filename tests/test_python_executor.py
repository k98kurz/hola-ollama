"""
Tests for python_executor module.
"""

import unittest
from pathlib import Path
import tempfile
import os

from hola.tools.python_executor import (
    _verify_blacklist,
    VerificationException,
    PermissionDeniedError,
    DEFAULT_ALLOWED_MODULES,
    python,
)


class TestVerifyBlacklist(unittest.TestCase):
    """Tests for AST blacklist validation."""

    def test_safe_code_passes(self):
        _verify_blacklist("x = 1 + 2")
        _verify_blacklist("print('hello')")
        _verify_blacklist("for i in range(10):\n    pass")
        _verify_blacklist("def foo():\n    return 42")

    def test_syntax_error(self):
        with self.assertRaises(VerificationException) as cm:
            _verify_blacklist("x = 1 +")
        self.assertIn("Syntax error", str(cm.exception))

    def test_del_keyword_blocked(self):
        with self.assertRaises(VerificationException) as cm:
            _verify_blacklist("del x")
        self.assertIn("Blacklisted AST node", str(cm.exception))

    def test_del_statement_blocked(self):
        with self.assertRaises(VerificationException) as cm:
            _verify_blacklist("del x, y")
        self.assertIn("Blacklisted AST node", str(cm.exception))

    def test_nonlocal_blocked(self):
        with self.assertRaises(VerificationException) as cm:
            _verify_blacklist("def foo():\n    nonlocal x")
        self.assertIn("Blacklisted AST node", str(cm.exception))

    def test_global_blocked(self):
        with self.assertRaises(VerificationException) as cm:
            _verify_blacklist("def foo():\n    global x")
        self.assertIn("Blacklisted AST node", str(cm.exception))

    def test_exec_blocked(self):
        with self.assertRaises(VerificationException) as cm:
            _verify_blacklist("exec('x = 1')")
        self.assertIn("Blacklisted function call", str(cm.exception))

    def test_eval_blocked(self):
        with self.assertRaises(VerificationException) as cm:
            _verify_blacklist("eval('1 + 1')")
        self.assertIn("Blacklisted function call", str(cm.exception))

    def test_compile_blocked(self):
        with self.assertRaises(VerificationException) as cm:
            _verify_blacklist("compile('x = 1', '<string>', 'exec')")
        self.assertIn("Blacklisted function call", str(cm.exception))

    def test_open_blocked(self):
        with self.assertRaises(VerificationException) as cm:
            _verify_blacklist("open('file.txt')")
        self.assertIn("Blacklisted function call", str(cm.exception))

    def test_getattr_blocked(self):
        with self.assertRaises(VerificationException) as cm:
            _verify_blacklist("getattr(obj, 'attr')")
        self.assertIn("Blacklisted function call", str(cm.exception))

    def test_setattr_blocked(self):
        with self.assertRaises(VerificationException) as cm:
            _verify_blacklist("setattr(obj, 'attr', 1)")
        self.assertIn("Blacklisted function call", str(cm.exception))

    def test_delattr_blocked(self):
        with self.assertRaises(VerificationException) as cm:
            _verify_blacklist("delattr(obj, 'attr')")
        self.assertIn("Blacklisted function call", str(cm.exception))

    def test_exit_blocked(self):
        with self.assertRaises(VerificationException) as cm:
            _verify_blacklist("exit()")
        self.assertIn("Blacklisted function call", str(cm.exception))

    def test_quit_blocked(self):
        with self.assertRaises(VerificationException) as cm:
            _verify_blacklist("quit()")
        self.assertIn("Blacklisted function call", str(cm.exception))


class TestModuleAllowlist(unittest.TestCase):
    """Tests for module allowlist enforcement."""

    def test_default_allowed_modules_defined(self):
        self.assertIsInstance(DEFAULT_ALLOWED_MODULES, list)
        self.assertGreater(len(DEFAULT_ALLOWED_MODULES), 0)
        self.assertIn('math', DEFAULT_ALLOWED_MODULES)
        self.assertIn('json', DEFAULT_ALLOWED_MODULES)

    def test_safe_module_import(self):
        result = python("import math\nprint(math.sqrt(16))", timeout=5)
        self.assertIn("4.0", result[0])

    def test_allowed_json_module(self):
        result = python(
            "import json\nprint(json.dumps({'key': 'value'}))",
            timeout=5
        )
        self.assertIn('"key"', result[0])
        self.assertIn('"value"', result[0])

    def test_allowed_re_module(self):
        result = python(
            "import re\nprint(re.sub(r'hello', 'world', 'hello there'))",
            timeout=5
        )
        self.assertIn("world there", result[0])

    def test_forbidden_module_import(self):
        with self.assertRaises(VerificationException) as cm:
            python("import os", timeout=5)
        self.assertIn("not in allowlist", str(cm.exception))

    def test_forbidden_submodule_import(self):
        with self.assertRaises(VerificationException) as cm:
            python("import os.path", timeout=5)
        self.assertIn("not in allowlist", str(cm.exception))


class TestTimeoutHandling(unittest.TestCase):
    """Tests for timeout enforcement."""

    def test_fast_code_completes(self):
        result = python("x = 1 + 1\nprint(x)", timeout=5)
        self.assertIn("2", result[0])

    def test_code_in_loop_times_out(self):
        with self.assertRaises(TimeoutError) as cm:
            python("while True:\n    pass", timeout=2)
        self.assertIn("timeout", str(cm.exception).lower())

    def test_long_operation_times_out(self):
        with self.assertRaises(TimeoutError) as cm:
            python("while True:\n    pass", timeout=2)
        self.assertIn("timeout", str(cm.exception).lower())

    def test_nested_loops_times_out(self):
        with self.assertRaises(TimeoutError):
            python(
                "for i in range(10000000):\n    for j in range(100):\n        pass",
                timeout=1
            )


class TestSuccessfulExecution(unittest.TestCase):
    """Tests for successful code execution scenarios."""

    def test_arithmetic_operations(self):
        result = python("print(2 + 2)", timeout=5)
        self.assertIn("4", result[0])

    def test_string_operations(self):
        result = python("print('hello' + ' world')", timeout=5)
        self.assertIn("hello world", result[0])

    def test_list_operations(self):
        result = python("print([1, 2, 3] * 2)", timeout=5)
        self.assertIn("[1, 2, 3, 1, 2, 3]", result[0])

    def test_dict_operations(self):
        result = python("print({'a': 1, 'b': 2}.get('a'))", timeout=5)
        self.assertIn("1", result[0])

    def test_function_definition(self):
        result = python(
            "def add(x, y):\n    return x + y\nprint(add(3, 4))",
            timeout=5
        )
        self.assertIn("7", result[0])

    def test_class_definition(self):
        result = python(
            "class Counter:\n    def __init__(self):\n        self.x = 0\n"
            "c = Counter()\nprint(c.x)",
            timeout=5
        )
        self.assertIn("0", result[0])

    def test_lambda_functions(self):
        result = python("f = lambda x: x * 2\nprint(f(5))", timeout=5)
        self.assertIn("10", result[0])

    def test_list_comprehensions(self):
        result = python("print([x*2 for x in range(5)])", timeout=5)
        self.assertIn("[0, 2, 4, 6, 8]", result[0])

    def test_exception_handling(self):
        result = python(
            "try:\n    1/0\nexcept ZeroDivisionError:\n    print('caught')",
            timeout=5
        )
        self.assertIn("caught", result[0])

    def test_print_multiple_lines(self):
        result = python("print('line1')\nprint('line2')", timeout=5)
        self.assertIn("line1", result[0])
        self.assertIn("line2", result[0])

    def test_empty_code(self):
        result = python("", timeout=5)
        self.assertEqual(result[0].strip(), "")


class TestExecutionErrors(unittest.TestCase):
    """Tests for various error conditions."""

    def test_runtime_error(self):
        with self.assertRaises(VerificationException) as cm:
            python("1/0", timeout=5)
        self.assertIn("Execution error", str(cm.exception))
        self.assertIn("division by zero", str(cm.exception))

    def test_name_error(self):
        with self.assertRaises(VerificationException) as cm:
            python("print(undefined_variable)", timeout=5)
        self.assertIn("Execution error", str(cm.exception))
        self.assertIn("undefined_variable", str(cm.exception))

    def test_type_error(self):
        with self.assertRaises(VerificationException) as cm:
            python("print(1 + 'a')", timeout=5)
        self.assertIn("Execution error", str(cm.exception))
        self.assertIn("unsupported operand type", str(cm.exception))

    def test_import_error_for_allowed_module(self):
        with self.assertRaises(VerificationException) as cm:
            python("import nonexistent_module", timeout=5)
        self.assertIn("Execution error", str(cm.exception))


class TestPermissionAwareFileOps(unittest.TestCase):
    """Tests for permission-aware file operations."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_file_read_without_permission_denied(self):
        test_file = self.temp_dir / "test.txt"
        test_file.write_text("hello")

        from hola.session.session import Session
        session = Session("test_session_python", portable=True)
        session.dbsession.cwd = "/some/other/path"

        with self.assertRaises(VerificationException) as cm:
            python(f"open('{test_file}', 'r')", timeout=5, session=session)
        self.assertIn("Execution error", str(cm.exception))

    def test_file_write_without_permission_denied(self):
        test_file = self.temp_dir / "test.txt"

        from hola.session.session import Session
        session = Session("test_session_python", portable=True)
        session.dbsession.cwd = str(self.temp_dir)

        with self.assertRaises(VerificationException) as cm:
            python(f"open('{test_file}', 'w')", timeout=5, session=session)
        self.assertIn("Execution error", str(cm.exception))

    def test_file_read_with_permission_succeeds(self):
        test_file = self.temp_dir / "test.txt"
        test_file.write_text("hello")

        from hola.session.session import Session
        from hola.permissions import Permission
        session = Session("test_session_python", portable=True)
        session.dbsession.cwd = str(self.temp_dir)
        session.set_permission(str(test_file), Permission(4))

        result = python(f"print(open('{test_file}', 'r').read())", timeout=5, session=session)
        self.assertIn("hello", result[0])

    def test_file_write_with_permission_succeeds(self):
        test_file = self.temp_dir / "test.txt"

        from hola.session.session import Session
        from hola.permissions import Permission
        session = Session("test_session_python", portable=True)
        session.dbsession.cwd = str(self.temp_dir)
        session.set_permission(str(test_file.parent), Permission(7))
        session.set_permission(str(test_file), Permission(7))

        python(f"open('{test_file}', 'w').write('world')", timeout=5, session=session)
        self.assertEqual(test_file.read_text(), "world")


if __name__ == "__main__":
    unittest.main()
