"""
Tests for bundled tools.
"""

import asyncio
import unittest
import shutil
from pathlib import Path
from hola.errors import ToolExecutionError
from hola.session.session import Session
from hola.session.storage import initialize
from hola.tools.core import execute_tool_call, list_tools
from hola.tools.types import ToolCall, ToolCallFunction
from hola.permissions import Permission


class TestSessionBase(unittest.TestCase):
    """Base class for bundled tool tests with proper session lifecycle."""
    def setUp(self):
        """Set up test fixtures."""
        initialize(True)
        session_id = self.__class__.__name__.lower()
        self.session = Session(session_id, portable=True)
        self.test_dir = Path.cwd() / ".hola"

    def permit(self, path: str | Path, mode: int = 7) -> None:
        """Grant `mode` permission on `path` to the session."""
        self.session.set_permission(str(path), Permission(mode))

    def tearDown(self):
        """Clean up after tests."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)


class TestMemorizeAndRecall(TestSessionBase):
    def test_memorize_and_recall_basic(self):
        async def test():
            call = ToolCall(
                function=ToolCallFunction(
                    name="memorize",
                    arguments={"key": "test_key", "value": "test_value"},
                ),
            )
            await execute_tool_call(call, self.session)

            recall_call = ToolCall(
                function=ToolCallFunction(
                    name="recall",
                    arguments={"key": "test_key"},
                ),
            )
            result = await execute_tool_call(recall_call, self.session)
            self.assertEqual(result, "test_value")

        asyncio.run(test())

    def test_memorize_overwrite(self):
        async def test():
            call1 = ToolCall(
                function=ToolCallFunction(
                    name="memorize",
                    arguments={"key": "test_key", "value": "old_value"},
                ),
            )
            await execute_tool_call(call1, self.session)

            call2 = ToolCall(
                function=ToolCallFunction(
                    name="memorize",
                    arguments={"key": "test_key", "value": "new_value"},
                ),
            )
            await execute_tool_call(call2, self.session)

            recall_call = ToolCall(
                function=ToolCallFunction(
                    name="recall",
                    arguments={"key": "test_key"},
                ),
            )
            result = await execute_tool_call(recall_call, self.session)
            self.assertEqual(result, "new_value")

        asyncio.run(test())

    def test_complex_data_structures(self):
        async def test():
            complex_data = {"nested": {"list": [1, 2, 3], "value": None}}

            call = ToolCall(
                function=ToolCallFunction(
                    name="memorize",
                    arguments={"key": "complex_key", "value": complex_data},
                ),
            )
            await execute_tool_call(call, self.session)

            recall_call = ToolCall(
                function=ToolCallFunction(
                    name="recall",
                    arguments={"key": "complex_key"},
                ),
            )
            result = await execute_tool_call(recall_call, self.session)
            self.assertEqual(result, complex_data)

        asyncio.run(test())

    def test_tools_registered_and_schema(self):
        tools = list_tools()
        tool_names = [t.name for t in tools]
        self.assertIn("memorize", tool_names)
        self.assertIn("recall", tool_names)

        memorize_tool = next((t for t in tools if t.name == "memorize"), None)
        self.assertIsNotNone(memorize_tool)
        params = memorize_tool.parameters.get("properties", {})
        self.assertNotIn("session", params)
        required = memorize_tool.parameters.get("required", [])
        self.assertNotIn("session", required)

        recall_tool = next((t for t in tools if t.name == "recall"), None)
        self.assertIsNotNone(recall_tool)
        params = recall_tool.parameters.get("properties", {})
        self.assertNotIn("session", params)
        required = recall_tool.parameters.get("required", [])
        self.assertNotIn("session", required)


class TestGrep(TestSessionBase):
    def test_grep_single_file(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            test_file = Path(self.session.dbsession.cwd) / "test.txt"
            test_file.write_text("hello world\nhello there\ngoodbye world")

            call = ToolCall(
                function=ToolCallFunction(
                    name="grep",
                    arguments={"pattern": "hello", "path": "test.txt"},
                ),
            )
            result = await execute_tool_call(call, self.session)
            self.assertEqual(len(result), 2)
            self.assertIn("hello world", result[0])
            self.assertIn("hello there", result[1])

            test_file.unlink()

        asyncio.run(test())

    def test_grep_line_numbers(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            test_file = Path(self.session.dbsession.cwd) / "test.txt"
            test_file.write_text("line one\nline two\nline three\nline four")

            call = ToolCall(
                function=ToolCallFunction(
                    name="grep",
                    arguments={"pattern": "line", "path": "test.txt", "n": True},
                ),
            )
            result = await execute_tool_call(call, self.session)
            self.assertEqual(len(result), 4)
            self.assertTrue(result[0].startswith("1:"))
            self.assertTrue(result[1].startswith("2:"))

            test_file.unlink()

        asyncio.run(test())

    def test_grep_recursive(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            testdir = Path(self.session.dbsession.cwd) / "testdir"
            testdir.mkdir()
            self.session.set_permission(str(testdir), Permission(7))

            subdir = testdir / "subdir"
            subdir.mkdir()
            self.session.set_permission(str(subdir), Permission(7))

            test_file1 = testdir / "test1.txt"
            test_file1.write_text("hello world")

            test_file2 = subdir / "test2.txt"
            test_file2.write_text("hello there")

            call = ToolCall(
                function=ToolCallFunction(
                    name="grep",
                    arguments={"pattern": "hello", "path": "testdir", "r": True},
                ),
            )
            result = await execute_tool_call(call, self.session)
            self.assertEqual(len(result), 2)

            test_file1.unlink()
            test_file2.unlink()
            subdir.rmdir()
            testdir.rmdir()

        asyncio.run(test())

    def test_grep_context_after(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            test_file = Path(self.session.dbsession.cwd) / "test.txt"
            test_file.write_text(
                "line one\nline two\nline three\nline four\nline five")

            call = ToolCall(
                function=ToolCallFunction(
                    name="grep",
                    arguments={"pattern": "line two", "path": "test.txt", "A": 2},
                ),
            )
            result = await execute_tool_call(call, self.session)
            self.assertIn("line two", result[0])
            self.assertEqual(len(result), 3)

            test_file.unlink()

        asyncio.run(test())

    def test_grep_context_before(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            test_file = Path(self.session.dbsession.cwd) / "test.txt"
            test_file.write_text(
                "line one\nline two\nline three\nline four\nline five")

            call = ToolCall(
                function=ToolCallFunction(
                    name="grep",
                    arguments={"pattern": "line four", "path": "test.txt", "B": 2},
                ),
            )
            result = await execute_tool_call(call, self.session)
            self.assertIn("line four", result[-1])
            self.assertEqual(len(result), 3)

            test_file.unlink()

        asyncio.run(test())

    def test_grep_data_string(self):
        async def test():
            call = ToolCall(
                function=ToolCallFunction(
                    name="grep",
                    arguments={
                        "pattern": "hello", "data": "hello world\nhello there"
                    },
                ),
            )
            result = await execute_tool_call(call, self.session)
            self.assertEqual(len(result), 2)

        asyncio.run(test())

    def test_grep_data_list(self):
        async def test():
            call = ToolCall(
                function=ToolCallFunction(
                    name="grep",
                    arguments={
                        "pattern": "hello",
                        "data": ["hello world", "hello there", "goodbye"],
                    },
                ),
            )
            result = await execute_tool_call(call, self.session)
            self.assertEqual(len(result), 2)

        asyncio.run(test())

    def test_grep_data_dict(self):
        async def test():
            call = ToolCall(
                function=ToolCallFunction(
                    name="grep",
                    arguments={
                        "pattern": "hello",
                        "data": {"key": "hello world", "other": "goodbye"},
                    },
                ),
            )
            result = await execute_tool_call(call, self.session)
            self.assertEqual(len(result), 1)

        asyncio.run(test())

    def test_grep_pattern_not_found(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            test_file = Path(self.session.dbsession.cwd) / "test.txt"
            test_file.write_text("hello world")

            call = ToolCall(
                function=ToolCallFunction(
                    name="grep",
                    arguments={"pattern": "xyz", "path": "test.txt"},
                ),
            )
            result = await execute_tool_call(call, self.session)
            self.assertEqual(len(result), 0)

            test_file.unlink()

        asyncio.run(test())

    def test_grep_permission_denied(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            test_file = Path(self.session.dbsession.cwd) / "test.txt"
            test_file.write_text("hello world")

            self.session.set_permission(str(test_file), Permission(0))

            call = ToolCall(
                function=ToolCallFunction(
                    name="grep",
                    arguments={"pattern": "hello", "path": "test.txt"},
                ),
            )

            with self.assertRaises(ToolExecutionError) as cm:
                await execute_tool_call(call, self.session)
            self.assertIn("permission denied", str(cm.exception).lower())

            test_file.unlink()

        asyncio.run(test())


class TestRead(TestSessionBase):
    def test_read_success(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            test_file = Path(self.session.dbsession.cwd) / "test.txt"
            test_file.write_text("Hello, World!")

            call = ToolCall(
                function=ToolCallFunction(
                    name="read",
                    arguments={"path": "test.txt"},
                ),
            )
            result = await execute_tool_call(call, self.session)
            self.assertEqual(result, "Hello, World!")
            test_file.unlink()

        asyncio.run(test())

    def test_read_failures(self):
        async def test():
            self.permit(self.session.dbsession.cwd, 4)

            test_file = Path(self.session.dbsession.cwd) / "test.txt"
            test_file.write_text("content")

            call_not_found = ToolCall(
                function=ToolCallFunction(
                    name="read",
                    arguments={"path": "nonexistent.txt"},
                ),
            )

            with self.assertRaises(ToolExecutionError) as cm:
                await execute_tool_call(call_not_found, self.session)
            self.assertIn("not found", str(cm.exception).lower())

            self.session.set_permission(str(test_file), Permission(0))

            call_no_perm = ToolCall(
                function=ToolCallFunction(
                    name="read",
                    arguments={"path": "test.txt"},
                ),
            )

            with self.assertRaises(ToolExecutionError) as cm:
                await execute_tool_call(call_no_perm, self.session)
            self.assertIn("permission denied", str(cm.exception).lower())

            test_file.unlink()

        asyncio.run(test())


class TestWrite(TestSessionBase):
    def test_write_success(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            test_file = Path(self.session.dbsession.cwd) / "test.txt"
            test_dir = Path(self.session.dbsession.cwd) / "subdir" / "nested"

            call = ToolCall(
                function=ToolCallFunction(
                    name="write",
                    arguments={"path": "test.txt", "content": "Hello, World!"},
                ),
            )
            await execute_tool_call(call, self.session)
            self.assertEqual(test_file.read_text(), "Hello, World!")

            call_nested = ToolCall(
                function=ToolCallFunction(
                    name="write",
                    arguments={
                        "path": "subdir/nested/test.txt",
                        "content": "Nested content",
                    },
                ),
            )
            await execute_tool_call(call_nested, self.session)
            self.assertTrue(test_dir.exists())
            self.assertEqual(
                (test_dir / "test.txt").read_text(), "Nested content"
            )

            test_file.unlink()
            (test_dir / "test.txt").unlink()
            test_dir.rmdir()
            test_dir.parent.rmdir()

        asyncio.run(test())

    def test_write_failures(self):
        async def test():
            call = ToolCall(
                function=ToolCallFunction(
                    name="write",
                    arguments={"path": "test.txt", "content": "content"},
                ),
            )

            with self.assertRaises(ToolExecutionError) as cm:
                await execute_tool_call(call, self.session)
            self.assertIn("permission denied", str(cm.exception).lower())

            self.permit(self.session.dbsession.cwd)

            test_file = Path(self.session.dbsession.cwd) / "test.txt"
            test_file.write_text("existing")

            self.session.set_permission(str(test_file), Permission(0))

            call_no_write = ToolCall(
                function=ToolCallFunction(
                    name="write",
                    arguments={"path": "test.txt", "content": "new"},
                ),
            )

            with self.assertRaises(ToolExecutionError) as cm:
                await execute_tool_call(call_no_write, self.session)
            self.assertIn("permission denied", str(cm.exception).lower())

            test_file.unlink()

        asyncio.run(test())

    def test_write_nested_permissions(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            dstdir = Path(self.session.dbsession.cwd) / "dstdir"
            dstdir.mkdir(exist_ok=True)
            self.session.set_permission(str(dstdir), Permission(0))

            call_denied = ToolCall(
                function=ToolCallFunction(
                    name="write",
                    arguments={
                        "path": "dstdir/subdir/file.txt",
                        "content": "content",
                    },
                ),
            )

            with self.assertRaises(ToolExecutionError) as cm:
                await execute_tool_call(call_denied, self.session)
            self.assertIn("permission denied", str(cm.exception).lower())

            self.session.set_permission(str(dstdir), Permission(7))

            await execute_tool_call(call_denied, self.session)
            self.assertTrue((dstdir / "subdir" / "file.txt").exists())

            shutil.rmtree(dstdir)

        asyncio.run(test())


class TestRm(TestSessionBase):
    def test_rm_success(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            test_file = Path(self.session.dbsession.cwd) / "test.txt"
            test_dir = Path(self.session.dbsession.cwd) / "testdir"
            test_dir.mkdir(exist_ok=True)
            (test_dir / "file.txt").write_text("content")
            test_file.write_text("content")

            call_file = ToolCall(
                function=ToolCallFunction(
                    name="rm",
                    arguments={"path": "test.txt"},
                ),
            )
            await execute_tool_call(call_file, self.session)
            self.assertFalse(test_file.exists())

            call_dir = ToolCall(
                function=ToolCallFunction(
                    name="rm",
                    arguments={"path": "testdir", "r": True},
                ),
            )
            await execute_tool_call(call_dir, self.session)
            self.assertFalse(test_dir.exists())

        asyncio.run(test())

    def test_rm_failures(self):
        async def test():
            test_dir = Path(self.session.dbsession.cwd) / "testdir"
            test_dir.mkdir(exist_ok=True)
            (test_dir / "file.txt").write_text("content")

            call = ToolCall(
                function=ToolCallFunction(
                    name="rm",
                    arguments={"path": "testdir"},
                ),
            )

            with self.assertRaises(ToolExecutionError) as cm:
                await execute_tool_call(call, self.session)
            self.assertIn("permission denied", str(cm.exception).lower())

            self.session.set_permission(str(test_dir), Permission(7))
            with self.assertRaises(ToolExecutionError) as cm:
                await execute_tool_call(call, self.session)
            self.assertIn("not empty", str(cm.exception).lower())

            call_not_found = ToolCall(
                function=ToolCallFunction(
                    name="rm",
                    arguments={"path": "nonexistent"},
                ),
            )

            with self.assertRaises(ToolExecutionError) as cm:
                await execute_tool_call(call_not_found, self.session)
            self.assertIn("not found", str(cm.exception).lower())

            shutil.rmtree(test_dir)

        asyncio.run(test())


class TestMv(TestSessionBase):
    def test_mv_success(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            test_file = Path(self.session.dbsession.cwd) / "test.txt"
            test_dir = Path(self.session.dbsession.cwd) / "testdir"
            test_dir.mkdir(exist_ok=True)

            test_file.write_text("content")

            call_move = ToolCall(
                function=ToolCallFunction(
                    name="mv",
                    arguments={"src": "test.txt", "dst": "testdir/test.txt"},
                ),
            )
            await execute_tool_call(call_move, self.session)
            self.assertFalse(test_file.exists())
            self.assertTrue((test_dir / "test.txt").exists())

            call_rename = ToolCall(
                function=ToolCallFunction(
                    name="mv",
                    arguments={
                        "src": "testdir/test.txt",
                        "dst": "testdir/renamed.txt",
                    },
                ),
            )
            await execute_tool_call(call_rename, self.session)
            self.assertFalse((test_dir / "test.txt").exists())
            self.assertTrue((test_dir / "renamed.txt").exists())

            (test_dir / "renamed.txt").unlink()
            test_dir.rmdir()

        asyncio.run(test())

    def test_mv_failures(self):
        async def test():
            call = ToolCall(
                function=ToolCallFunction(
                    name="mv",
                    arguments={"src": "test.txt", "dst": "new.txt"},
                ),
            )

            with self.assertRaises(ToolExecutionError) as cm:
                await execute_tool_call(call, self.session)
            self.assertIn("permission denied", str(cm.exception).lower())

            self.permit(self.session.dbsession.cwd)

            test_file = Path(self.session.dbsession.cwd) / "test.txt"
            test_file.write_text("content")

            dstdir = Path(self.session.dbsession.cwd) / "dstdir"
            dstdir.mkdir(exist_ok=True)
            self.session.set_permission(str(dstdir), Permission(0))

            call_dst_denied = ToolCall(
                function=ToolCallFunction(
                    name="mv",
                    arguments={"src": "test.txt", "dst": "dstdir/new.txt"},
                ),
            )

            with self.assertRaises(ToolExecutionError) as cm:
                await execute_tool_call(call_dst_denied, self.session)
            self.assertIn("permission denied", str(cm.exception).lower())

            call_not_found = ToolCall(
                function=ToolCallFunction(
                    name="mv",
                    arguments={"src": "nonexistent.txt", "dst": "new.txt"},
                ),
            )

            with self.assertRaises(ToolExecutionError) as cm:
                await execute_tool_call(call_not_found, self.session)
            self.assertIn("not found", str(cm.exception).lower())

            test_file.unlink()
            shutil.rmtree(dstdir)

        asyncio.run(test())


class TestCp(TestSessionBase):
    def test_cp_success(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            test_file = Path(self.session.dbsession.cwd) / "test.txt"
            test_dir = Path(self.session.dbsession.cwd) / "srcdir"
            test_dir.mkdir(exist_ok=True)
            (test_dir / "file.txt").write_text("content")
            dstdir = Path(self.session.dbsession.cwd) / "dstdir"
            dstdir.mkdir(exist_ok=True)

            call_file = ToolCall(
                function=ToolCallFunction(
                    name="cp",
                    arguments={"src": "test.txt", "dst": "new.txt"},
                ),
            )

            test_file.write_text("original")
            await execute_tool_call(call_file, self.session)
            self.assertTrue(test_file.exists())
            self.assertTrue((Path(self.session.dbsession.cwd) / "new.txt").exists())
            self.assertEqual(test_file.read_text(), "original")

            dst_copy_dir = dstdir / "srcdir"
            if dst_copy_dir.exists():
                shutil.rmtree(dst_copy_dir)

            call_dir = ToolCall(
                function=ToolCallFunction(
                    name="cp",
                    arguments={"src": "srcdir", "dst": "dstdir/srcdir", "r": True},
                ),
            )
            await execute_tool_call(call_dir, self.session)
            self.assertTrue((dstdir / "srcdir" / "file.txt").exists())

            test_file.unlink()
            (Path(self.session.dbsession.cwd) / "new.txt").unlink()
            shutil.rmtree(dstdir)
            shutil.rmtree(test_dir)

        asyncio.run(test())

    def test_cp_failures(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            test_dir = Path(self.session.dbsession.cwd) / "testdir"
            test_dir.mkdir(exist_ok=True)
            (test_dir / "file.txt").write_text("content")

            call = ToolCall(
                function=ToolCallFunction(
                    name="cp",
                    arguments={"src": "testdir", "dst": "newdir"},
                ),
            )

            with self.assertRaises(ToolExecutionError) as cm:
                await execute_tool_call(call, self.session)
            self.assertIn("requires -r", str(cm.exception).lower())

            self.session.set_permission(str(test_dir), Permission(0))

            call_src_denied = ToolCall(
                function=ToolCallFunction(
                    name="cp",
                    arguments={"src": "testdir/file.txt", "dst": "new.txt"},
                ),
            )

            with self.assertRaises(ToolExecutionError) as cm:
                await execute_tool_call(call_src_denied, self.session)
            self.assertIn("permission denied", str(cm.exception).lower())

            self.session.set_permission(str(test_dir), Permission(7))

            dstdir = Path(self.session.dbsession.cwd) / "dstdir"
            dstdir.mkdir(exist_ok=True)
            self.session.set_permission(str(dstdir), Permission(0))

            call_dst_denied = ToolCall(
                function=ToolCallFunction(
                    name="cp",
                    arguments={"src": "testdir/file.txt", "dst": "dstdir/new.txt"},
                ),
            )

            with self.assertRaises(ToolExecutionError) as cm:
                await execute_tool_call(call_dst_denied, self.session)
            self.assertIn("permission denied", str(cm.exception).lower())

            call_not_found = ToolCall(
                function=ToolCallFunction(
                    name="cp",
                    arguments={"src": "nonexistent.txt", "dst": "new.txt"},
                ),
            )

            with self.assertRaises(ToolExecutionError) as cm:
                await execute_tool_call(call_not_found, self.session)
            self.assertIn("not found", str(cm.exception).lower())

            shutil.rmtree(test_dir)
            shutil.rmtree(dstdir)

        asyncio.run(test())


class TestLs(TestSessionBase):
    def test_ls_basic(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            test_file = Path(self.session.dbsession.cwd) / "test.txt"
            test_dir = Path(self.session.dbsession.cwd) / "testdir"
            test_dir.mkdir(exist_ok=True)
            hidden_file = Path(self.session.dbsession.cwd) / ".hidden"

            test_file.write_text("content")
            hidden_file.write_text("hidden")

            call = ToolCall(
                function=ToolCallFunction(
                    name="ls",
                    arguments={"path": "."},
                ),
            )
            result = await execute_tool_call(call, self.session)
            self.assertIn("test.txt", result)
            self.assertIn("testdir", result)
            self.assertNotIn(".hidden", result)

            test_file.unlink()
            hidden_file.unlink()
            test_dir.rmdir()

        asyncio.run(test())

    def test_ls_flags(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            test_file = Path(self.session.dbsession.cwd) / "test.txt"
            test_file.write_text("content")

            call_l = ToolCall(
                function=ToolCallFunction(
                    name="ls",
                    arguments={"path": ".", "l": True},
                ),
            )
            result = await execute_tool_call(call_l, self.session)
            self.assertTrue(any("-rw" in line for line in result))
            self.assertTrue(any("test.txt" in line for line in result))

            call_a = ToolCall(
                function=ToolCallFunction(
                    name="ls",
                    arguments={"path": ".", "a": True},
                ),
            )
            result = await execute_tool_call(call_a, self.session)
            self.assertTrue(any("test.txt" in line for line in result))

            test_file.unlink()

        asyncio.run(test())

    def test_ls_recursive(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            nested_dir = Path(self.session.dbsession.cwd) / "dir1" / "dir2"
            nested_dir.mkdir(parents=True)
            (nested_dir / "file.txt").write_text("content")

            call = ToolCall(
                function=ToolCallFunction(
                    name="ls",
                    arguments={"path": ".", "R": True},
                ),
            )
            result = await execute_tool_call(call, self.session)
            self.assertIn("dir1:", result)
            self.assertIn("dir1/dir2:", result)
            self.assertIn("file.txt", result)

            shutil.rmtree(Path(self.session.dbsession.cwd) / "dir1")

        asyncio.run(test())

    def test_ls_permissions(self):
        async def test():
            test_dir = Path(self.session.dbsession.cwd) / "testdir"
            test_dir.mkdir(exist_ok=True)
            self.session.set_permission(str(test_dir), Permission(0))

            call = ToolCall(
                function=ToolCallFunction(
                    name="ls",
                    arguments={"path": "testdir"},
                ),
            )

            with self.assertRaises(ToolExecutionError) as cm:
                await execute_tool_call(call, self.session)
            self.assertIn("permission denied", str(cm.exception).lower())

            test_dir.rmdir()

        asyncio.run(test())


class TestEdit(TestSessionBase):
    def test_edit_first_occurrence(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            test_file = Path(self.session.dbsession.cwd) / "test.txt"
            test_file.write_text("hello world hello world")

            call = ToolCall(
                function=ToolCallFunction(
                    name="edit",
                    arguments={
                        "path": "test.txt",
                        "old_string": "hello",
                        "new_string": "hi",
                    },
                ),
            )
            await execute_tool_call(call, self.session)
            self.assertEqual(test_file.read_text(), "hi world hello world")

            test_file.unlink()

        asyncio.run(test())

    def test_edit_all_occurrences(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            test_file = Path(self.session.dbsession.cwd) / "test.txt"
            test_file.write_text("hello world hello world")

            call = ToolCall(
                function=ToolCallFunction(
                    name="edit",
                    arguments={
                        "path": "test.txt",
                        "old_string": "hello",
                        "new_string": "hi",
                        "count": 0,
                    },
                ),
            )
            await execute_tool_call(call, self.session)
            self.assertEqual(test_file.read_text(), "hi world hi world")

            test_file.unlink()

        asyncio.run(test())

    def test_edit_skip_occurrences(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            test_file = Path(self.session.dbsession.cwd) / "test.txt"
            test_file.write_text("hello world hello world hello")

            call = ToolCall(
                function=ToolCallFunction(
                    name="edit",
                    arguments={
                        "path": "test.txt",
                        "old_string": "hello",
                        "new_string": "hi",
                        "skip": 1,
                    },
                ),
            )
            await execute_tool_call(call, self.session)
            self.assertEqual(test_file.read_text(), "hello world hi world hello")

            test_file.unlink()

        asyncio.run(test())

    def test_edit_specific_count(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            test_file = Path(self.session.dbsession.cwd) / "test.txt"
            test_file.write_text("hello hello hello hello")

            call = ToolCall(
                function=ToolCallFunction(
                    name="edit",
                    arguments={
                        "path": "test.txt",
                        "old_string": "hello",
                        "new_string": "hi",
                        "count": 2,
                    },
                ),
            )
            await execute_tool_call(call, self.session)
            self.assertEqual(test_file.read_text(), "hi hi hello hello")

            test_file.unlink()

        asyncio.run(test())

    def test_edit_failures(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            test_file = Path(self.session.dbsession.cwd) / "test.txt"
            test_file.write_text("content")

            call_not_found = ToolCall(
                function=ToolCallFunction(
                    name="edit",
                    arguments={
                        "path": "test.txt",
                        "old_string": "notfound",
                        "new_string": "new",
                    },
                ),
            )

            with self.assertRaises(ToolExecutionError) as cm:
                await execute_tool_call(call_not_found, self.session)
            self.assertIn("not found", str(cm.exception).lower())

            self.session.set_permission(str(test_file), Permission(0))
            call_no_write = ToolCall(
                function=ToolCallFunction(
                    name="edit",
                    arguments={
                        "path": "test.txt",
                        "old_string": "content",
                        "new_string": "new",
                    },
                ),
            )

            with self.assertRaises(ToolExecutionError) as cm:
                await execute_tool_call(call_no_write, self.session)
            self.assertIn("permission denied", str(cm.exception).lower())

            self.session.set_permission(str(test_file), Permission(2))
            call_no_read = ToolCall(
                function=ToolCallFunction(
                    name="edit",
                    arguments={
                        "path": "test.txt",
                        "old_string": "content",
                        "new_string": "new",
                    },
                ),
            )

            with self.assertRaises(ToolExecutionError) as cm:
                await execute_tool_call(call_no_read, self.session)
            self.assertIn("permission denied", str(cm.exception).lower())

            test_file.unlink()

        asyncio.run(test())


class TestAppend(TestSessionBase):
    def test_append_existing(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            test_file = Path(self.session.dbsession.cwd) / "test.txt"
            test_file.write_text("hello ")

            call = ToolCall(
                function=ToolCallFunction(
                    name="append",
                    arguments={
                        "path": "test.txt",
                        "content": "world",
                    },
                ),
            )
            await execute_tool_call(call, self.session)
            self.assertEqual(test_file.read_text(), "hello world")

            test_file.unlink()

        asyncio.run(test())

    def test_append_new_file(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            test_file = Path(self.session.dbsession.cwd) / "test.txt"

            call = ToolCall(
                function=ToolCallFunction(
                    name="append",
                    arguments={
                        "path": "test.txt",
                        "content": "hello world",
                    },
                ),
            )
            await execute_tool_call(call, self.session)
            self.assertEqual(test_file.read_text(), "hello world")

            test_file.unlink()

        asyncio.run(test())

    def test_append_failures(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            test_file = Path(self.session.dbsession.cwd) / "test.txt"
            test_file.write_text("content")

            self.session.set_permission(str(test_file), Permission(0))

            call = ToolCall(
                function=ToolCallFunction(
                    name="append",
                    arguments={
                        "path": "test.txt",
                        "content": "more",
                    },
                ),
            )

            with self.assertRaises(ToolExecutionError) as cm:
                await execute_tool_call(call, self.session)
            self.assertIn("permission denied", str(cm.exception).lower())

            self.session.set_permission(str(test_file), Permission(7))

            subdir = Path(self.session.dbsession.cwd) / "subdir"
            subdir.mkdir()
            self.session.set_permission(str(subdir), Permission(0))

            call_denied = ToolCall(
                function=ToolCallFunction(
                    name="append",
                    arguments={
                        "path": "subdir/new.txt",
                        "content": "content",
                    },
                ),
            )

            with self.assertRaises(ToolExecutionError) as cm:
                await execute_tool_call(call_denied, self.session)
            self.assertIn("permission denied", str(cm.exception).lower())

            test_file.unlink()
            shutil.rmtree(subdir)

        asyncio.run(test())


class TestFind(TestSessionBase):
    def test_find_all(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            test_dir = Path(self.session.dbsession.cwd) / "testdir"
            test_dir.mkdir()
            (test_dir / "file1.txt").write_text("content")
            (test_dir / "file2.txt").write_text("content")

            call = ToolCall(
                function=ToolCallFunction(
                    name="find",
                    arguments={"path": "testdir"},
                ),
            )
            result = await execute_tool_call(call, self.session)
            self.assertIn("file1.txt", result)
            self.assertIn("file2.txt", result)

            shutil.rmtree(test_dir)

        asyncio.run(test())

    def test_find_name_pattern(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            test_dir = Path(self.session.dbsession.cwd) / "testdir"
            test_dir.mkdir()
            (test_dir / "file1.txt").write_text("content")
            (test_dir / "file2.md").write_text("content")
            (test_dir / "file3.txt").write_text("content")

            call = ToolCall(
                function=ToolCallFunction(
                    name="find",
                    arguments={
                        "path": "testdir",
                        "name": "*.txt",
                    },
                ),
            )
            result = await execute_tool_call(call, self.session)
            self.assertIn("file1.txt", result)
            self.assertIn("file3.txt", result)
            self.assertNotIn("file2.md", result)

            shutil.rmtree(test_dir)

        asyncio.run(test())

    def test_find_type_filter(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            test_dir = Path(self.session.dbsession.cwd) / "testdir"
            test_dir.mkdir()
            (test_dir / "file.txt").write_text("content")
            (test_dir / "subdir").mkdir()

            call_files = ToolCall(
                function=ToolCallFunction(
                    name="find",
                    arguments={
                        "path": "testdir",
                        "type": "f",
                    },
                ),
            )
            result = await execute_tool_call(call_files, self.session)
            self.assertIn("file.txt", result)
            self.assertNotIn("subdir", result)

            call_dirs = ToolCall(
                function=ToolCallFunction(
                    name="find",
                    arguments={
                        "path": "testdir",
                        "type": "d",
                    },
                ),
            )
            result = await execute_tool_call(call_dirs, self.session)
            self.assertIn("subdir", result)
            self.assertNotIn("file.txt", result)

            shutil.rmtree(test_dir)

        asyncio.run(test())

    def test_find_depth_filter(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            test_dir = Path(self.session.dbsession.cwd) / "testdir"
            test_dir.mkdir()
            (test_dir / "file1.txt").write_text("content")
            (test_dir / "subdir").mkdir()
            (test_dir / "subdir" / "file2.txt").write_text("content")

            call_mindepth = ToolCall(
                function=ToolCallFunction(
                    name="find",
                    arguments={
                        "path": "testdir",
                        "mindepth": 1,
                    },
                ),
            )
            result = await execute_tool_call(call_mindepth, self.session)
            self.assertIn("file1.txt", result)
            self.assertIn("subdir", result)
            self.assertIn("subdir/file2.txt", result)

            call_maxdepth = ToolCall(
                function=ToolCallFunction(
                    name="find",
                    arguments={
                        "path": "testdir",
                        "maxdepth": 1,
                    },
                ),
            )
            result = await execute_tool_call(call_maxdepth, self.session)
            self.assertIn("file1.txt", result)
            self.assertIn("subdir", result)
            self.assertNotIn("subdir/file2.txt", result)

            shutil.rmtree(test_dir)

        asyncio.run(test())

    def test_find_permissions(self):
        async def test():
            self.permit(self.session.dbsession.cwd)

            test_dir = Path(self.session.dbsession.cwd) / "testdir"
            test_dir.mkdir()
            (test_dir / "file.txt").write_text("content")

            self.session.set_permission(str(test_dir), Permission(0))

            call = ToolCall(
                function=ToolCallFunction(
                    name="find",
                    arguments={"path": "testdir"},
                ),
            )

            with self.assertRaises(ToolExecutionError) as cm:
                await execute_tool_call(call, self.session)
            self.assertIn("permission denied", str(cm.exception).lower())

            self.session.set_permission(str(test_dir), Permission(7))

            subdir = test_dir / "subdir"
            subdir.mkdir()
            (subdir / "file.txt").write_text("content")
            self.session.set_permission(str(subdir), Permission(0))

            call = ToolCall(
                function=ToolCallFunction(
                    name="find",
                    arguments={"path": "testdir"},
                ),
            )
            result = await execute_tool_call(call, self.session)
            self.assertIn("file.txt", result)
            self.assertNotIn("subdir", result)
            self.assertNotIn("subdir/file.txt", result)

            shutil.rmtree(test_dir)

        asyncio.run(test())


if __name__ == "__main__":
    unittest.main()
