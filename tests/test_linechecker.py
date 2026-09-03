"""
Tests for scripts/linechecker.py docstring-aware line length checking.
"""

import importlib.util
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

_scripts = Path(__file__).resolve().parent.parent / 'scripts'
_spec = importlib.util.spec_from_file_location(
    'linechecker', _scripts / 'linechecker.py'
)
linechecker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(linechecker)


class TestLineChecker(unittest.TestCase):
    """Bundle line length checking behavior tests."""
    def test_template_literal_with_standalone_closer(self):
        src = (
            'X = """prefix text\n'
            'some template content line that is 80 chars exactly aaaaa\n'
            '"""\n'
            'Y = """another string\n'
            'more template content that is seventy-eight chars long ok\n'
            '"""\n'
        )
        hard, soft, ws = linechecker.check_source(src)
        assert hard == 0, hard
        assert ws == 0, ws

    def test_multiline_module_docstring_long_lines_flagged(self):
        src = (
            '"""A module docstring whose opening line is fine.\n'
            'This second line is deliberately far longer than the seventy-two '
            'character docstring limit so it must be reported.\n'
            '"""\n'
        )
        hard, soft, ws = linechecker.check_source(src)
        assert hard == 1, hard
        assert ws == 0, ws

    def test_multiline_docstring_over_limit_flagged(self):
        src = (
            'def foo():\n'
            '    """Function docstring spanning several physical lines.\n'
            '    This body line is deliberately long enough to exceed the '
            'seventy-two character docstring limit and must be flagged.\n'
            '    """\n'
            '    pass\n'
        )
        hard, soft, ws = linechecker.check_source(src)
        assert hard == 1, hard
        assert ws == 0, ws

    def test_single_line_docstring_relaxed_limit(self):
        line = '    """' + 'A docstring padded to be seventy five chars long '
        line += 'y' * (75 - len(line)) + '"""'
        assert 72 < len(line) <= 80, len(line)
        hard, soft, ws = linechecker.check_source('def foo():\n' + line + '\n')
        assert hard == 0, hard
        assert ws == 0, ws

    def test_single_line_docstring_over_eighty_flagged(self):
        line = '    """' + 'B docstring padded to be eighty five chars long '
        line += 'z' * (85 - len(line)) + '"""'
        assert len(line) > 80, len(line)
        hard, soft, ws = linechecker.check_source('def foo():\n' + line + '\n')
        assert hard == 1, hard
        assert ws == 0, ws

    def test_multiline_docstring_line_not_relaxed(self):
        src = (
            'def foo():\n'
            '    """Opener.\n'
            '    This docstring line is deliberately seventy five chars long '
            'over the multiline docstring limit.\n'
            '    """\n'
            '    pass\n'
        )
        hard, soft, ws = linechecker.check_source(src)
        assert hard == 1, hard
        assert ws == 0, ws

    def test_code_lines_use_hardmax(self):
        code = "# " + "a" * 100 + "\n"
        hard, soft, ws = linechecker.check_source(code)
        assert hard == 1, hard
        assert ws == 0, ws

        ok = "x = 'short'\n"
        hard, soft, ws = linechecker.check_source(ok)
        assert hard == 0, hard
        assert ws == 0, ws

    def test_fstring_template_content_uses_code_limit(self):
        line = "f-string template content line " + "x" * 50
        assert 72 < len(line) <= 85, len(line)
        src = (
            'PREFIX = """Some prefix string content here.\n'
            '"""\n'
            'TEMPLATE = f"""{PREFIX}\n'
            f'{line}\n'
            '"""\n'
        )
        hard, soft, ws = linechecker.check_source(src)
        assert hard == 0, hard
        assert ws == 0, ws

    def test_trailing_whitespace_flagged(self):
        src = "x = 1   \n"
        hard, soft, ws = linechecker.check_source(src)
        assert hard == 0, hard
        assert ws == 1, ws

    def test_softmax_applies_to_non_docstring_lines_only(self):
        src = (
            'code_line = "short"\n'
            '"""A docstring line here is just fine and short.\n'
            '"""\n'
        )
        hard, soft, ws = linechecker.check_source(src, softmax=10)
        assert hard == 0, hard
        assert soft == 1, soft
        assert ws == 0, ws

    def test_syntax_error_source_does_not_crash(self):
        src = 'def broken(:\n    """doc\n'
        hard, soft, ws = linechecker.check_source(src)
        assert hard == 0, hard
        assert ws == 0, ws


class TestRecursiveLineChecker(unittest.TestCase):
    """Bundle directory traversal and aggregate behavior tests."""
    def setUp(self):
        self.root = Path('temp') / 'linechecker_test'
        if self.root.exists():
            shutil.rmtree(self.root)
        (self.root / 'sub' / 'venv').mkdir(parents=True)
        (self.root / 'sub' / 'excl').mkdir(parents=True)
        self.clean = self.root / 'a.py'
        self.clean.write_text("x = 'short'\n")
        (self.root / 'sub' / 'bad_doc.py').write_text(
            '"""A module docstring.\n'
            'This docstring line is deliberately longer than seventy-two '
            'characters in length.\n'
            '"""\n'
        )
        (self.root / 'sub' / 'long.py').write_text(
            '# ' + 'a' * 100 + '\n'
        )
        (self.root / 'sub' / 'venv' / 'skip.py').write_text(
            'y = 1\n'
        )
        (self.root / 'sub' / 'excl' / 'skip2.py').write_text(
            'z = 2\n'
        )
        (self.root / 'sub' / 'bad.py').write_bytes(b'\xff\xfe\x00bad')

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_collect_single_file(self):
        files = linechecker.collect_files(str(self.clean))
        assert files == [self.clean], files

    def test_collect_recursive_finds_nested(self):
        files = linechecker.collect_files(str(self.root), recursive=True)
        names = [f.relative_to(self.root).as_posix() for f in files]
        assert 'a.py' in names, names
        assert 'sub/bad_doc.py' in names, names
        assert 'sub/long.py' in names, names

    def test_collect_recursive_skips_ignored_and_excludes(self):
        files = linechecker.collect_files(
            str(self.root), recursive=True, excludes=['excl']
        )
        names = [f.relative_to(self.root).as_posix() for f in files]
        assert 'sub/venv/skip.py' not in names, names
        assert 'sub/excl/skip2.py' not in names, names

    def test_collect_top_level_only_without_recursive(self):
        files = linechecker.collect_files(str(self.root))
        names = [f.relative_to(self.root).as_posix() for f in files]
        assert names == ['a.py'], names

    def test_collect_missing_path_exits(self):
        with self.assertRaises(SystemExit):
            linechecker.collect_files('temp/nonexistent_dir_xyz')

    def test_run_aggregates_counts_and_skips_binary(self):
        hard, soft, ws, skipped = linechecker.run(
            str(self.root), recursive=True, excludes=['excl']
        )
        assert hard == 2, hard
        assert soft == 0, soft
        assert ws == 0, ws
        assert skipped == 1, skipped

    def test_cli_recursive_exit_codes(self):
        clean_root = Path('temp') / 'linechecker_clean'
        if clean_root.exists():
            shutil.rmtree(clean_root)
        clean_root.mkdir(parents=True)
        (clean_root / 'ok.py').write_text("x = 'fine'\n")
        try:
            r = subprocess.run(
                [sys.executable, str(_scripts / 'linechecker.py'),
                 '-r', str(self.root)],
                capture_output=True, text=True,
            )
            assert r.returncode == 1, r.returncode
            r = subprocess.run(
                [sys.executable, str(_scripts / 'linechecker.py'),
                 '-r', str(clean_root)],
                capture_output=True, text=True,
            )
            assert r.returncode == 0, r.returncode
        finally:
            shutil.rmtree(clean_root)

    def test_cli_unreadable_file_exits_nonzero(self):
        r = subprocess.run(
            [sys.executable, str(_scripts / 'linechecker.py'),
             str(self.root / 'sub' / 'bad.py')],
            capture_output=True, text=True,
        )
        assert r.returncode == 1, r.returncode
        assert 'could not be read' in r.stderr, r.stderr


if __name__ == '__main__':
    unittest.main()
