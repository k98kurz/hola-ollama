"""
Tests for session management functionality.
"""

from pathlib import Path
from hola.session import (
    Session,
    WorkflowEvent,
    WorkflowFSM,
    WorkflowState,
    create_session,
    delete_session,
    get_session,
    list_sessions,
    session_exists,
)
from hola.session.exceptions import (
    SessionAlreadyExistsError,
    SessionNotFoundError,
    TempFileError,
    TempFileNotFoundError,
)
from hola.session.manager import _generate_session_id
from hola.session.storage import initialize
from hola.types import Message as OMessage, Role, ToolCall, ToolCallFunction
import os
import shutil
import time
import unittest
import packify


class TestSessionLifecycle(unittest.TestCase):
    """Test cases for session lifecycle management."""
    def setUp(self):
        """Set up test fixtures."""
        initialize(True)
        self.test_tag = f"test_{time.time_ns()}"
        self.test_dir = Path.cwd() / ".hola"

    def tearDown(self):
        """Clean up after tests."""
        hola_dir = Path.cwd() / ".hola"
        if hola_dir.exists():
            shutil.rmtree(hola_dir)

    def test_session_creation(self):
        """Test session creation with basic params, custom ID, and tag."""
        session = Session("test_session_1", portable=True)
        assert session.session_id == "test_session_1", session.session_id
        assert session.portable is True, session.portable
        assert session.dbsession is not None, session.dbsession
        assert session.dbsession.id == "test_session_1", session.dbsession.id
        session.delete()

        session_id = "my_custom_id_123"
        session = create_session(session_id=session_id, portable=True)
        assert session.session_id == session_id, session.session_id
        session.delete()

        session = create_session(tag=self.test_tag, portable=True)
        session_id = session.session_id
        assert session_id.startswith("20"), session_id
        assert self.test_tag[:8] in session_id, session_id
        retrieved_session = get_session(session_id, portable=True)
        assert retrieved_session.session_id == session_id, (
            retrieved_session.session_id)
        session.delete()

        session_id = "init_test"
        session = Session(session_id, portable=True)
        assert session.dbsession is not None, session.dbsession
        assert session.dbsession.id == session_id, session.dbsession.id
        assert session.sequence == 0, session.sequence
        assert session.dbsession.cwd == os.getcwd(), session.dbsession.cwd
        session.delete()

    def test_session_retrieval(self):
        """Test retrieving existing session and handling not found."""
        session = create_session(portable=True)
        session_id = session.session_id
        retrieved_session = get_session(session_id, portable=True)
        assert retrieved_session.session_id == session_id, (
            retrieved_session.session_id
        )
        session.delete()

        with self.assertRaises(SessionNotFoundError):
            get_session("nonexistent_session", portable=True)

        session = create_session(portable=True)
        session_id = session.session_id
        assert session_exists(session_id, portable=True) is True
        session.delete()
        assert session_exists(session_id, portable=True) is False

    def test_session_listing(self):
        """Test listing sessions when empty and multiple."""
        sessions = list_sessions(portable=True)
        for session_id in sessions:
            delete_session(session_id, portable=True)
        sessions = list_sessions(portable=True)
        assert len(sessions) == 0, len(sessions)

        ids = []
        for i in range(3):
            session_id = f"list_test_{i}"
            create_session(session_id=session_id, portable=True)
            ids.append(session_id)

        sessions = list_sessions(portable=True)
        for session_id in ids:
            assert session_id in sessions, session_id

        for session_id in ids:
            delete_session(session_id, portable=True)

    def test_session_deletion(self):
        """Test deleting session via method and manager, with cleanup."""
        session = Session("del_method_test", portable=True)
        session.delete()
        assert session.dbsession is None, session.dbsession
        assert not session_exists("del_method_test", portable=True), (
            "Session still exists"
        )

        session = create_session(portable=True)
        session_id = session.session_id
        delete_session(session_id, portable=True)
        assert not session_exists(session_id, portable=True), "Session still exists"

        sessions = list_sessions(portable=True)
        assert session_id not in sessions, "Session in list"

        session_id = "temp_dir_test"
        session = Session(session_id, portable=True)
        session.create_temp_file("test.txt", "content")
        temp_dir = Path.cwd() / ".hola" / "sessions" / session_id / "temp"
        assert temp_dir.exists(), temp_dir
        session.delete()
        assert not temp_dir.exists(), temp_dir

        session = Session("temp_files_test", portable=True)
        session.create_temp_file("file1.txt", "content1")
        session.create_temp_file("file2.bin", b"bytes")
        session.delete()
        sessions = list_sessions(portable=True)
        assert "temp_files_test" not in sessions, sessions

    def test_session_errors(self):
        """Test session already exists and not found errors."""
        session_id = "duplicate_id_test"
        session = create_session(session_id=session_id, portable=True)

        with self.assertRaises(SessionAlreadyExistsError):
            create_session(session_id=session_id, portable=True)

        session.delete()

        with self.assertRaises(SessionNotFoundError):
            delete_session("nonexistent_session", portable=True)

    def test_generate_session_id_format(self):
        """Test session ID format matches expected pattern."""
        session_id = _generate_session_id()
        parts = session_id.split("-")

        assert len(parts) == 5, parts
        assert len(parts[0]) == 4, parts[0]
        assert len(parts[1]) == 2, parts[1]
        assert len(parts[2]) == 2, parts[2]
        assert ":" in parts[3], parts[3]
        assert len(parts[4]) == 8, parts[4]


class TestTempFiles(unittest.TestCase):
    """Test cases for temporary file operations."""
    def setUp(self):
        """Set up test fixtures."""
        initialize(True)
        self.session = Session("temp_test", portable=True)

    def tearDown(self):
        """Clean up after tests."""
        hola_dir = Path.cwd() / ".hola"
        if hola_dir.exists():
            shutil.rmtree(hola_dir)

    def test_crud_operations(self):
        """Test create, read, list, delete operations with string/bytes content."""
        content_str = "Hello, World!"
        file_path = self.session.create_temp_file("test.txt", content_str)
        assert file_path.exists(), file_path
        assert file_path.read_text() == content_str, file_path.read_text()

        content_bytes = b"\x00\x01\x02\x03\x04\x05"
        file_path = self.session.create_temp_file("test.bin", content_bytes)
        assert file_path.exists(), file_path
        assert file_path.read_bytes() == content_bytes, file_path.read_bytes()

        content = "test content"
        self.session.create_temp_file("read_test.txt", content)
        read_content = self.session.read_temp_file("read_test.txt")
        assert read_content.decode() == content, read_content

        self.session.create_temp_file("file1.txt", "content1")
        self.session.create_temp_file("file2.txt", "content2")
        self.session.create_temp_file("file3.bin", b"bytes")
        files = self.session.list_temp_files()
        assert len(files) == 6, len(files)
        assert "file1.txt" in files, files
        assert "file2.txt" in files, files
        assert "file3.bin" in files, files

        file_path = self.session.create_temp_file("delete.txt", "content")
        self.session.delete_temp_file("delete.txt")
        assert not file_path.exists(), file_path

    def test_error_handling(self):
        """Test file not found errors."""
        with self.assertRaises(TempFileNotFoundError):
            self.session.read_temp_file("nonexistent.txt")

        with self.assertRaises(TempFileNotFoundError):
            self.session.delete_temp_file("nonexistent.txt")

    def test_overwrite_behavior(self):
        """Test overwriting existing file."""
        original = "original content"
        updated = "updated content"

        self.session.create_temp_file("overwrite.txt", original)
        file_path = self.session.create_temp_file("overwrite.txt", updated)

        assert file_path.read_text() == updated, file_path.read_text()

    def test_directory_structure(self):
        """Test lazy temp directory creation."""
        temp_dir = Path.cwd() / ".hola" / "sessions" / self.session.session_id
        self.session.create_temp_file("test.txt", "content")
        assert temp_dir.exists(), temp_dir

        self.session.create_temp_file("file.txt", "content")
        (temp_dir / "subdir").mkdir(parents=True, exist_ok=True)
        files = self.session.list_temp_files()
        assert len(files) == 2, len(files)
        assert "file.txt" in files, files
        assert "test.txt" in files, files

    def test_temp_file_exceptions_exported(self):
        """Test that TempFileNotFoundError and TempFileError are exported."""
        assert TempFileNotFoundError is not None, TempFileNotFoundError
        assert TempFileError is not None, TempFileError


class TestSessionDataStorage(unittest.TestCase):
    """Test cases for session data storage (messages, tool outputs,
        metadata, memory).
    """
    def setUp(self):
        """Set up test fixtures."""
        initialize(True)
        self.session = Session("data_test", portable=True)

    def tearDown(self):
        """Clean up after tests."""
        hola_dir = Path.cwd() / ".hola"
        if hola_dir.exists():
            shutil.rmtree(hola_dir)

    def test_messages_crud(self):
        """Test message add, get, list, delete operations."""
        msg = OMessage(role=Role.USER, content="Hello")
        self.session.add_message("granite4.1:3b", msg)

        history = self.session.get_history()
        assert len(history) == 1, len(history)
        assert history[0][0] == "granite4.1:3b", history[0][0]
        assert history[0][1].content == "Hello", history[0][1].content

        initial_seq = self.session.sequence
        for i in range(5):
            self.session.add_message(
                "granite4.1:3b", OMessage(role=Role.USER, content=str(i))
            )
        assert self.session.sequence == initial_seq + 10, self.session.sequence

        msgs = [
            OMessage(role=Role.USER, content="Hello"),
            OMessage(role=Role.ASSISTANT, content="Hi there"),
            OMessage(role=Role.USER, content="How are you?"),
        ]
        for msg in msgs:
            self.session.add_message("granite4.1:3b", msg)

        history = self.session.get_history()
        assert len(history) == 9, len(history)

    def test_messages_with_tool_calls(self):
        """Test storing and retrieving messages with tool calls."""
        call1 = ToolCall(
            function=ToolCallFunction(
                name="memorize",
                arguments={
                    "key": "file.txt",
                    "value": ToolCall(
                        function=ToolCallFunction(
                            name="read_file",
                            arguments={"path": "/tmp/file.txt"},
                        ),
                    )
                }
            )
        )
        call2 = ToolCall(
            function=ToolCallFunction(
                name="grep",
                arguments={"pattern": "test"},
            ),
        )

        msg = OMessage(
            role=Role.ASSISTANT,
            content="I'll help you with that",
            tool_calls=[call1, call2],
        )

        self.session.add_message("granite4.1:3b", msg)

        history = self.session.get_history()
        assert len(history) == 1, len(history)
        assert history[0][0] == "granite4.1:3b", history[0][0]
        retrieved_msg = history[0][1]
        assert retrieved_msg.content == "I'll help you with that", retrieved_msg.content
        assert retrieved_msg.tool_calls is not None, retrieved_msg.tool_calls
        assert len(retrieved_msg.tool_calls) == 2, len(retrieved_msg.tool_calls)
        assert retrieved_msg.tool_calls[0].function.name == "memorize", (
            retrieved_msg.tool_calls[0].function.name)
        assert retrieved_msg.tool_calls[0].function.arguments.get(
            "key"
        ) == "file.txt", retrieved_msg.tool_calls[0].function.arguments
        assert retrieved_msg.tool_calls[1].function.name == "grep", (
            retrieved_msg.tool_calls[1].function.name)
        assert retrieved_msg.tool_calls[1].function.arguments == {
            "pattern": "test"
        }, retrieved_msg.tool_calls[1].function.arguments

    def test_messages_filtering(self):
        """Test message history filtering by limit and model."""
        for i in range(20):
            self.session.add_message(
                "granite4.1:3b", OMessage(role=Role.USER, content=str(i))
            )

        history = self.session.get_history(limit=5)
        assert len(history) == 5, len(history)

        for i in range(10):
            model = "model_a" if i % 2 == 0 else "model_b"
            self.session.add_message(
                model, OMessage(role=Role.USER, content=str(i))
            )

        history_a = self.session.get_history(model="model_a")
        history_b = self.session.get_history(model="model_b")
        assert len(history_a) == 5, len(history_a)
        assert len(history_b) == 5, len(history_b)

        for i in range(5):
            self.session.add_message(
                "granite4.1:3b", OMessage(role=Role.USER, content=f"Msg {i}")
            )

        history = self.session.get_history()
        contents = [msg[1].content for msg in history]
        assert contents[-5:] == [
            "Msg 0", "Msg 1", "Msg 2", "Msg 3", "Msg 4"
        ], contents

    def test_tool_outputs_crud(self):
        """Test tool output add, get, list, delete operations."""
        call_id = "call_123"
        result = {"status": "success", "data": [1, 2, 3]}
        self.session.add_tool_output(call_id, result)

        retrieved = self.session.get_tool_output(call_id)
        assert retrieved == result, retrieved

        outputs = {
            "call_1": {"result": 1},
            "call_2": {"result": 2},
            "call_3": {"result": 3},
        }
        for call_id, result in outputs.items():
            self.session.add_tool_output(call_id, result)

        all_outputs = self.session.list_tool_outputs()
        assert len(all_outputs) == 4, len(all_outputs)

    def test_tool_outputs_filtering(self):
        """Test tool output filtering scenarios."""
        call_ids = ["call_a", "call_b", "call_c"]
        for i, call_id in enumerate(call_ids):
            self.session.add_tool_output(call_id, {"value": i})

        outputs = self.session.list_tool_outputs()
        assert len(outputs) == 3, len(outputs)
        for call_id in call_ids:
            assert call_id in outputs, call_id

        result = self.session.get_tool_output("nonexistent_call")
        assert result is None, result

        outputs = self.session.list_tool_outputs()
        assert len(outputs) == 3, len(outputs)

    def test_metadata_crud(self):
        """Test metadata set, get, list, delete operations."""
        self.session.set_metadata("key1", "value1")
        assert self.session.get_metadata("key1") == "value1", "Value not set"

        self.session.set_metadata("key", "original")
        self.session.set_metadata("key", "updated")
        assert self.session.get_metadata("key") == "updated", "Value not updated"

        data = {"key1": "val1", "key2": 123, "key3": [1, 2, 3]}
        for key, value in data.items():
            self.session.set_metadata(key, value)

        metadata = self.session.list_metadata()
        assert len(metadata) == 4, len(metadata)
        for key in data:
            assert key in metadata, key

        self.session.set_metadata("test_key", {"nested": "data"})
        value = self.session.get_metadata("test_key")
        assert value == {"nested": "data"}, value

        self.session.unset_metadata("test_key")
        assert self.session.get_metadata("test_key", "default") == "default"

    def test_metadata_filtering(self):
        """Test metadata listing and default values."""
        metadata = self.session.list_metadata()
        assert len(metadata) == 0, len(metadata)

        self.session.set_metadata("key1", "value1")
        self.session.set_metadata("key2", "value2")
        self.session.set_metadata("key3", "value3")

        metadata = self.session.list_metadata()
        assert len(metadata) == 3, len(metadata)

        value = self.session.get_metadata("nonexistent", default="default_value")
        assert value == "default_value", value

    def test_memory_crud(self):
        """Test memory memorize, recall, list, forget operations."""
        self.session.memorize("name", "Alice")
        assert self.session.recall("name") == "Alice", "Value not memorized"

        self.session.memorize("age", 25)
        self.session.memorize("age", 26)
        assert self.session.recall("age") == 26, "Value not updated"

        self.session.memorize("key", "value")
        value = self.session.recall("key")
        assert value == "value", value

        self.session.memorize("temp", "data")
        self.session.forget("temp")
        assert self.session.recall("temp") is None, "Key not forgotten"

        memories = {"name": "Bob", "city": "NYC", "age": 30}
        for key, value in memories.items():
            self.session.memorize(key, value)

        listed = self.session.list_memories()
        assert len(listed) == 4, len(listed)
        for key in memories:
            assert key in listed, key

    def test_memory_filtering(self):
        """Test memory listing for non-existent keys."""
        listed = self.session.list_memories()
        assert len(listed) == 0, len(listed)

        value = self.session.recall("nonexistent")
        assert value is None, value

        self.session.forget("nonexistent")

    def test_config_crud(self):
        """Test config get, set, unset operations and persistence."""
        config = self.session.config
        assert len(config) == 0, len(config)

        self.session.config = {"key1": "value1"}
        config = self.session.config
        assert config["key1"] == "value1", config["key1"]

        self.session.config = {"key2": 123, "key3": [1, 2, 3]}
        config = self.session.config
        assert config["key1"] == "value1", config["key1"]
        assert config["key2"] == 123, config["key2"]
        assert config["key3"] == [1, 2, 3], config["key3"]

        self.session.unset_config("key1")
        config = self.session.config
        assert "key1" not in config, "key1 still exists"
        assert config["key2"] == 123, config["key2"]

        self.session.unset_config("key2", "key3")
        config = self.session.config
        assert len(config) == 0, len(config)

        self.session.unset_config("nonexistent", "also_nonexistent")
        config = self.session.config
        assert len(config) == 0, len(config)

        session_id = self.session.session_id
        self.session.config = {"persisted": True, "value": "test"}
        reloaded = get_session(session_id, portable=True)
        config = reloaded.config
        assert config["persisted"] is True, config["persisted"]
        assert config["value"] == "test", config["value"]

    def test_clear_history(self):
        """Test clearing history and resetting sequence."""
        for i in range(5):
            self.session.add_message(
                "granite4.1:3b", OMessage(role=Role.USER, content=str(i))
            )

        self.session.clear_history()
        history = self.session.get_history()

        assert len(history) == 0, len(history)
        assert self.session.sequence == 0, self.session.sequence


class TestSessionFSMData(unittest.TestCase):
    """Test cases for FSM state management."""
    def setUp(self):
        """Set up test fixtures."""
        initialize(True)
        self.session = Session("fsm_test", portable=True)

    def tearDown(self):
        """Clean up after tests."""
        hola_dir = Path.cwd() / ".hola"
        if hola_dir.exists():
            shutil.rmtree(hola_dir)

    def test_fsm_initialization(self):
        """Test session initializes with default WorkflowFSM."""
        assert self.session.state is not None, self.session.state
        assert self.session.state.current == WorkflowState.INIT, (
            self.session.state.current
        )

    def test_fsm_storage_and_retrieval(self):
        """Test FSM can be stored and retrieved from session."""
        self.session.state.input(WorkflowEvent.START)
        self.session.state.input(WorkflowEvent.EXECUTE)

        assert self.session.state.current == WorkflowState.EXECUTING, (
            self.session.state.current
        )

        reloaded = get_session(self.session.session_id, portable=True)
        reloaded.load_fsm_data(WorkflowFSM, inject={'WorkflowState': WorkflowState})
        assert reloaded.state.current == WorkflowState.EXECUTING, (
            reloaded.state.current
        )

    def test_fsm_persistence_across_session_reloads(self):
        """Test FSM state autosaves correctly across session reloads."""
        session_id = self.session.session_id
        self.session.state.input(WorkflowEvent.START)
        self.session.state.input(WorkflowEvent.EXECUTE)
        self.session.state.input(WorkflowEvent.REVIEW)

        assert self.session.state.current == WorkflowState.REVIEWING, (
            self.session.state.current
        )

        reloaded = get_session(session_id, portable=True)
        reloaded.load_fsm_data(WorkflowFSM, inject={'WorkflowState': WorkflowState})
        assert reloaded.state.current == WorkflowState.REVIEWING, (
            reloaded.state.current
        )

        reloaded.state.input(WorkflowEvent.REJECT)
        reloaded = get_session(session_id, portable=True)
        reloaded.load_fsm_data(WorkflowFSM, inject={'WorkflowState': WorkflowState})
        assert reloaded.state.current == WorkflowState.PLANNING, (
            reloaded.state.current
        )

    def test_set_fsm_replaces_instance(self):
        """Test set_fsm replaces the current FSM instance."""
        new_fsm = WorkflowFSM()
        self.session.set_fsm(new_fsm)
        self.session.state.input(WorkflowEvent.START)
        assert self.session.state.current == WorkflowState.PLANNING, (
            self.session.state.current
        )

        reloaded = get_session(self.session.session_id, portable=True)
        reloaded.load_fsm_data(WorkflowFSM, inject={'WorkflowState': WorkflowState})
        assert reloaded.state.current == WorkflowState.PLANNING, (
            reloaded.state.current
        )

    def test_load_fsm_data_custom_class(self):
        """Test loading FSM with custom inject dependencies."""
        self.session.state.input(WorkflowEvent.START)

        new_session = Session("fsm_load_test", portable=True)
        new_session.state.input(WorkflowEvent.START)
        assert new_session.state.current == WorkflowState.PLANNING, (
            new_session.state.current
        )
        new_session.delete()

    def test_save_fsm_data_manual(self):
        """Test manual save of FSM data."""
        self.session.save_fsm_data()
        self.session.state.current = WorkflowState.EXECUTING

        reloaded = get_session(self.session.session_id, portable=True)
        reloaded.load_fsm_data(WorkflowFSM, inject={'WorkflowState': WorkflowState})
        assert reloaded.state.current == WorkflowState.INIT, reloaded.state.current

        self.session.save_fsm_data()
        reloaded = get_session(self.session.session_id, portable=True)
        reloaded.load_fsm_data(WorkflowFSM, inject={'WorkflowState': WorkflowState})
        assert reloaded.state.current == WorkflowState.EXECUTING, (
            reloaded.state.current
        )


class TestSessionIntegration(unittest.TestCase):
    """Test cases for integration scenarios."""
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path.cwd() / ".hola"

    def tearDown(self):
        """Clean up after tests."""
        hola_dir = Path.cwd() / ".hola"
        if hola_dir.exists():
            shutil.rmtree(hola_dir)

    def test_full_lifecycle_workflow(self):
        """Test complete session workflow with lifecycle, persistence,
            and mixed operations.
        """
        initialize(True)
        session = create_session(tag="lifecycle", portable=True)
        session_id = session.session_id
        retrieved_session = get_session(session_id, portable=True)

        retrieved_session.add_message(
            "granite4.1:3b", OMessage(role=Role.USER, content="Hello")
        )
        retrieved_session.memorize("name", "TestUser")
        retrieved_session.set_metadata("version", "1.0")

        history = retrieved_session.get_history()
        assert len(history) == 1, len(history)

        name = retrieved_session.recall("name")
        assert name == "TestUser", name

        session1 = get_session(session_id, portable=True)
        session1.memorize("key", "value")
        session1.set_metadata("meta", "data")

        session2 = get_session(session_id, portable=True)
        value = session2.recall("key")
        meta = session2.get_metadata("meta")

        assert value == "value", value
        assert meta == "data", meta

        retrieved_session.delete()
        assert not session_exists(session_id, portable=True), "Session exists"

        session = Session("mixed_test", portable=True)
        session.add_message(
            "granite4.1:3b", OMessage(role=Role.USER, content="Hello")
        )
        session.create_temp_file("data.txt", "temp data")

        history = session.get_history()
        files = session.list_temp_files()

        assert len(history) == 1, len(history)
        assert len(files) == 1, len(files)

    def test_session_isolation(self):
        """Test that multiple sessions don't interfere with each other."""
        initialize(True)
        session1 = Session("iso_test_1", portable=True)
        session2 = Session("iso_test_2", portable=True)

        session1.memorize("key", "value1")
        session2.memorize("key", "value2")

        val1 = session1.recall("key")
        val2 = session2.recall("key")

        assert val1 == "value1", val1
        assert val2 == "value2", val2


if __name__ == "__main__":
    unittest.main()
