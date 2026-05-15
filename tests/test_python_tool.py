"""
Tests for python tool.
"""

from pathlib import Path
from hola.errors import VerificationException, PermissionDeniedError
from hola.permissions import Permission
from hola.session.session import Session
from hola.tools.python import _verify_blacklist, DEFAULT_ALLOWED_MODULES, python
import shutil
import tempfile
import unittest


class TestPythonTool(unittest.TestCase):
    """Comprehensive tests for the python execution tool."""
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_safe_code_passes_blacklist(self):
        _verify_blacklist("x = 1 + 2")
        _verify_blacklist("print('hello')")
        _verify_blacklist("for i in range(10):\n    pass")
        _verify_blacklist("def foo():\n    return 42")
        _verify_blacklist("class M: pass\nm = M()\nm.abc = 123")
        _verify_blacklist("s = str()\nprint(s.__str__())")
        _verify_blacklist("s = str()\nprint(s.__len__())")
        _verify_blacklist("s = str()\nprint(s.__repr__())")
        _verify_blacklist("obj = object()\nprint(obj.__dict__)")
        _verify_blacklist("def f(): pass\nprint(f.__name__)")
        _verify_blacklist("type('str') is str")

    def test_syntax_error(self):
        with self.assertRaises(VerificationException) as e:
            _verify_blacklist("x = 1 +")
        assert "Syntax error" in str(e.exception), e.exception

    def test_ast_nodes_blocked(self):
        test_cases = [
            "del x",
            "del x, y",
            "def foo():\n    nonlocal x",
            "def foo():\n    global x",
        ]
        for code in test_cases:
            with self.assertRaises(VerificationException) as e:
                _verify_blacklist(code)
            assert "Blacklisted AST node" in str(e.exception), e.exception

    def test_dangerous_code_blocked(self):
        test_cases = [
            ("exec('x = 1')", "exec"),
            ("eval('1 + 1')", "eval"),
            ("compile('x = 1', '<string>', 'exec')", "compile"),
            ("open('file.txt')", "open"),
            ("getattr(obj, 'attr')", "getattr"),
            ("setattr(obj, 'attr', 1)", "setattr"),
            ("delattr(obj, 'attr')", "delattr"),
            ("exit()", "exit"),
            ("quit()", "quit"),
            ("s=object(); s.__getattribute__('abc')", "__getattribute__"),
            ("s=object(); s.__getattr__('x')", "__getattr__"),
            ("s=object(); s.__setattr__('x', 1)", "__setattr__"),
            ("s=object(); s.__delattr__('x')", "__delattr__"),
            ("def f(): pass; f.__globals__", "__globals__"),
            ("object().__class__.__subclasses__()", "__subclasses__"),
            ("type().__bases__", "__bases__"),
            ("type().__mro__", "__mro__"),
            ("def f(): pass; f.__code__", "__code__"),
            ("import json\njson.__builtins__['open']('test.txt', 'w')", "__builtins__"),
            ("""
for cls in ().__class__.__bases__[0].__subclasses__():
    # Find a class defined in the 'os' module
    if cls.__name__ == '_wrap_close':
        # Access the 'os' module's dictionary via the function's globals
        os_globals = cls.__init__.__globals__

        # We now have full access to the 'os' module
        os_globals['system']('echo "Sandbox escaped via os.system!"')
        break""", "__subclasses__"),
        ]
        for code, func_name in test_cases:
            with self.assertRaises(VerificationException, msg=code) as e:
                _verify_blacklist(code)
            assert (
                "Blacklisted" in str(e.exception)
                or "not in allowlist" in str(e.exception)
            ), str(e.exception)
            assert func_name in str(e.exception), str(e.exception)

    def test_default_allowed_modules_defined(self):
        assert isinstance(DEFAULT_ALLOWED_MODULES, list), DEFAULT_ALLOWED_MODULES
        assert len(DEFAULT_ALLOWED_MODULES) > 0, DEFAULT_ALLOWED_MODULES
        assert 'math' in DEFAULT_ALLOWED_MODULES, DEFAULT_ALLOWED_MODULES
        assert 'json' in DEFAULT_ALLOWED_MODULES, DEFAULT_ALLOWED_MODULES

    def test_allowed_modules(self):
        test_cases = [
            ("import math\nprint(math.sqrt(16))", "4.0"),
            ("import json\nprint(json.dumps({'key': 'value'}))", '"key"'),
            ("import re\nprint(re.sub(r'hello', 'world', 'hello there'))",
             "world there"),
        ]
        for code, expected in test_cases:
            result = python(code, timeout=5)
            assert expected in result[0], (expected, result)

    def test_forbidden_modules(self):
        test_cases = [
            "import os",
            "import os.path",
        ]
        for code in test_cases:
            with self.assertRaises(VerificationException) as e:
                python(code, timeout=5)
            assert "not in allowlist" in str(e.exception), str(e.exception)

    def test_fast_code_completes(self):
        result = python("x = 1 + 1\nprint(x)", timeout=5)
        assert "2" in result[0], result

    def test_infinite_loop_times_out(self):
        with self.assertRaises(TimeoutError) as e:
            python("while True:\n    pass", timeout=2)
        assert "timeout" in str(e.exception).lower(), str(e.exception).lower()

    def test_nested_loops_times_out(self):
        with self.assertRaises(TimeoutError):
            python(
                "from time import sleep\nfor i in range(1000):\n    sleep(1)",
                timeout=1
            )

    def test_basic_operations(self):
        test_cases = [
            ("print(2 + 2)", "4"),
            ("print('hello' + ' world')", "hello world"),
            ("print([1, 2, 3] * 2)", "[1, 2, 3, 1, 2, 3]"),
            ("print({'a': 1, 'b': 2}.get('a'))", "1"),
        ]
        for code, expected in test_cases:
            result = python(code, timeout=5)
            assert expected in result[0], (expected, result)

    def test_functions_and_classes(self):
        test_cases = [
            ("def add(x, y):\n    return x + y\nprint(add(3, 4))", "7"),
            ("class Counter:\n    def __init__(self):\n        self.x = 0\n"
             "c = Counter()\nprint(c.x)", "0"),
            ("f = lambda x: x * 2\nprint(f(5))", "10"),
        ]
        for code, expected in test_cases:
            result = python(code, timeout=5)
            assert expected in result[0], (expected, result)

    def test_advanced_features(self):
        result = python("print([x*2 for x in range(5)])", timeout=5)
        assert "[0, 2, 4, 6, 8]" in result[0], result

        result = python(
            "try:\n    1/0\nexcept ZeroDivisionError:\n    print('caught')",
            timeout=5
        )
        assert "caught" in result[0], result

        result = python("print('line1')\nprint('line2')", timeout=5)
        assert "line1" in result[0], result
        assert "line2" in result[0], result

        result = python("", timeout=5)
        assert result[0].strip() == "", result

    def test_runtime_errors(self):
        test_cases = [
            ("1/0", "division by zero"),
            ("print(undefined_variable)", "undefined_variable"),
            ("print(1 + 'a')", "unsupported operand type"),
        ]
        for code, expected in test_cases:
            with self.assertRaises(VerificationException) as e:
                python(code, timeout=5)
            assert "Execution error" in str(e.exception), str(cm.expected)
            assert expected in str(e.exception), str(cm.expected)

    def test_import_errors(self):
        with self.assertRaises(VerificationException) as e:
            python("import nonexistent_module", timeout=5)
        assert "Execution error" in str(e.exception), str(cm.expected)

    def test_file_operations_without_permission_denied(self):
        session = Session("test_session_python", portable=True)
        session.dbsession.cwd = "/some/other/path"

        test_file = self.temp_dir / "test.txt"
        test_file.write_text("hello")

        with self.assertRaises(VerificationException) as e:
            python(f"open('{test_file}', 'r')", timeout=5, session=session)
        assert "Execution error" in str(e.exception), str(e.exception)

        test_file.unlink()
        with self.assertRaises(VerificationException) as e:
            python(f"open('{test_file}', 'w')", timeout=5, session=session)
        assert "Execution error" in str(e.exception), str(e.exception)

    def test_file_operations_with_permission_succeed(self):
        session = Session("test_session_python", portable=True)
        session.dbsession.cwd = str(self.temp_dir)

        test_file = self.temp_dir / "test.txt"
        test_file.write_text("hello")
        session.set_permission(str(test_file), Permission(4))

        result = python(
            f"print(open('{test_file}', 'r').read())", timeout=5,
            session=session
        )
        assert "hello" in result[0], result

        session.set_permission(str(test_file.parent), Permission(7))
        session.set_permission(str(test_file), Permission(7))
        python(f"open('{test_file}', 'w').write('world')", timeout=5,
                session=session)
        assert "world" == test_file.read_text(), test_file.read_text()


if __name__ == "__main__":
    unittest.main()
