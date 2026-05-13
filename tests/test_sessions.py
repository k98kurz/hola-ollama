"""
Tests for session management functionality.
"""

from pathlib import Path
import os
import shutil
import threading
import time
import unittest
import packify
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
from hola.ollama.types import Message as OMessage, Role



class TestSessionCreation(unittest.TestCase):
    """Test cases for session creation and lifecycle."""
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

    def test_create_session_basic(self):
        """Test session creation with default params."""
        session = Session("test_session_1", portable=True)

        assert session.session_id == "test_session_1", session.session_id
        assert session.portable is True, session.portable
        assert session.dbsession is not None, session.dbsession
        assert session.dbsession.id == "test_session_1", session.dbsession.id
        session.delete()

    def test_create_session_with_tag(self):
        """Test session creation with custom tag."""
        session = create_session(tag=self.test_tag, portable=True)
        session_id = session.session_id
        retrieved_session = get_session(session_id, portable=True)

        assert session_id.startswith("20"), session_id
        assert self.test_tag[:8] in session_id, session_id
        retrieved_session.delete()

    def test_create_session_with_custom_id(self):
        """Test session creation with specific ID."""
        session_id = "my_custom_id_123"
        session = create_session(
            session_id=session_id,
            portable=True,
        )
        retrieved = get_session(session_id, portable=True)

        assert retrieved.session_id == session_id, retrieved.session_id
        session.delete()

    def test_create_session_already_exists(self):
        """Test session creation with duplicate ID."""
        session_id = "duplicate_id_test"
        session = create_session(
            session_id=session_id,
            portable=True,
        )

        with self.assertRaises(SessionAlreadyExistsError):
            create_session(session_id=session_id, portable=True)

        session.delete()

    def test_session_initialization(self):
        """Test session dbsession and sequence initialization."""
        session_id = "init_test"
        session = Session(session_id, portable=True)

        assert session.dbsession is not None, session.dbsession
        assert session.dbsession.id == session_id, session.dbsession.id
        assert session.sequence == 0, session.sequence
        assert session.dbsession.cwd == os.getcwd(), session.dbsession.cwd
        session.delete()


class TestSessionManager(unittest.TestCase):
    """Test cases for session manager functions."""
    def setUp(self):
        """Set up test fixtures."""
        initialize(True)
        self.test_dir = Path.cwd() / ".hola"

    def tearDown(self):
        """Clean up after tests."""
        hola_dir = Path.cwd() / ".hola"
        if hola_dir.exists():
            shutil.rmtree(hola_dir)

    def test_get_session_existing(self):
        """Test retrieving existing session."""
        session = create_session(portable=True)
        session_id = session.session_id
        retrieved_session = get_session(session_id, portable=True)

        assert retrieved_session.session_id == session_id, (
            retrieved_session.session_id
        )
        retrieved_session.delete()

    def test_get_session_not_found(self):
        """Test getting non-existent session raises error."""
        with self.assertRaises(SessionNotFoundError):
            get_session("nonexistent_session", portable=True)

    def test_list_sessions_empty(self):
        """Test listing sessions when none exist."""
        sessions = list_sessions(portable=True)
        for session_id in sessions:
            delete_session(session_id, portable=True)
        sessions = list_sessions(portable=True)
        assert len(sessions) == 0, len(sessions)

    def test_list_sessions_multiple(self):
        """Test listing multiple sessions."""
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

    def test_delete_session_existing(self):
        """Test deleting existing session."""
        session = create_session(portable=True)
        session_id = session.session_id
        delete_session(session_id, portable=True)

        assert not session_exists(session_id, portable=True), (
            "Session still exists"
        )

    def test_delete_session_not_found(self):
        """Test deleting non-existent session raises error."""
        with self.assertRaises(SessionNotFoundError):
            delete_session("nonexistent_session", portable=True)

    def test_session_exists_true(self):
        """Test checking existence of existing session."""
        session = create_session(portable=True)
        session_id = session.session_id
        exists = session_exists(session_id, portable=True)

        assert exists is True, exists
        delete_session(session_id, portable=True)

    def test_session_exists_false(self):
        """Test checking existence of non-existent session."""
        exists = session_exists("nonexistent_session", portable=True)
        assert exists is False, exists

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


class TestTempFileCRUD(unittest.TestCase):
    """Test cases for temporary file CRUD operations."""
    def setUp(self):
        """Set up test fixtures."""
        initialize(True)
        self.session = Session("temp_test", portable=True)

    def tearDown(self):
        """Clean up after tests."""
        hola_dir = Path.cwd() / ".hola"
        if hola_dir.exists():
            shutil.rmtree(hola_dir)

    def test_create_temp_file_string(self):
        """Test creating temp file with string content."""
        content = "Hello, World!"
        file_path = self.session.create_temp_file("test.txt", content)

        assert file_path.exists(), file_path
        assert file_path.read_text() == content, file_path.read_text()

    def test_create_temp_file_bytes(self):
        """Test creating temp file with bytes content."""
        content = b"\x00\x01\x02\x03\x04\x05"
        file_path = self.session.create_temp_file("test.bin", content)

        assert file_path.exists(), file_path
        assert file_path.read_bytes() == content, file_path.read_bytes()

    def test_create_temp_file_overwrite(self):
        """Test overwriting existing temp file."""
        original = "original content"
        updated = "updated content"

        self.session.create_temp_file("overwrite.txt", original)
        file_path = self.session.create_temp_file("overwrite.txt", updated)

        assert file_path.read_text() == updated, file_path.read_text()

    def test_create_temp_file_creates_directory(self):
        """Test lazy temp directory creation."""
        temp_dir = self.session.portable and (
            Path.cwd() / ".hola" / "sessions" / self.session.session_id
        )
        temp_dir = Path.cwd() / ".hola" / "sessions" / self.session.session_id

        self.session.create_temp_file("test.txt", "content")

        assert temp_dir.exists(), temp_dir

    def test_read_temp_file_success(self):
        """Test reading existing temp file."""
        content = "test content"
        self.session.create_temp_file("read_test.txt", content)
        read_content = self.session.read_temp_file("read_test.txt")

        assert read_content.decode() == content, read_content

    def test_read_temp_file_not_found(self):
        """Test reading non-existent temp file raises error."""
        with self.assertRaises(TempFileNotFoundError):
            self.session.read_temp_file("nonexistent.txt")

    def test_delete_temp_file_success(self):
        """Test deleting existing temp file."""
        file_path = self.session.create_temp_file("delete.txt", "content")
        self.session.delete_temp_file("delete.txt")

        assert not file_path.exists(), file_path

    def test_delete_temp_file_not_found(self):
        """Test deleting non-existent temp file raises error."""
        with self.assertRaises(TempFileNotFoundError):
            self.session.delete_temp_file("nonexistent.txt")

    def test_list_temp_files_empty(self):
        """Test listing temp files when none exist."""
        files = self.session.list_temp_files()
        assert len(files) == 0, files

    def test_list_temp_files_multiple(self):
        """Test listing multiple temp files."""
        self.session.create_temp_file("file1.txt", "content1")
        self.session.create_temp_file("file2.txt", "content2")
        self.session.create_temp_file("file3.bin", b"bytes")

        files = self.session.list_temp_files()
        assert len(files) == 3, len(files)
        assert "file1.txt" in files, files
        assert "file2.txt" in files, files
        assert "file3.bin" in files, files

    def test_list_temp_files_after_deletion(self):
        """Test listing temp files after deletion."""
        self.session.create_temp_file("to_delete.txt", "content")
        self.session.create_temp_file("keep.txt", "content")

        self.session.delete_temp_file("to_delete.txt")
        files = self.session.list_temp_files()

        assert len(files) == 1, files
        assert "keep.txt" in files, files

    def test_list_temp_files_subdirectories(self):
        """Test that subdirectories are excluded from temp file list."""
        temp_dir = Path.cwd() / ".hola" / "sessions" / self.session.session_id
        self.session.create_temp_file("file.txt", "content")
        (temp_dir / "subdir").mkdir(parents=True, exist_ok=True)

        files = self.session.list_temp_files()

        assert len(files) == 1, len(files)
        assert "file.txt" in files, files


class TestTempFileErrorHandling(unittest.TestCase):
    """Test cases for temp file error handling."""
    def setUp(self):
        """Set up test fixtures."""
        initialize(True)
        self.session = Session("error_test", portable=True)

    def tearDown(self):
        """Clean up after tests."""
        hola_dir = Path.cwd() / ".hola"
        if hola_dir.exists():
            shutil.rmtree(hola_dir)

    def test_create_temp_file_session_error(self):
        """Test SessionError raised on OSError during file creation."""
        temp_dir = Path.cwd() / ".hola" / "sessions" / self.session.session_id

        original_mode = temp_dir.stat().st_mode if temp_dir.exists() else 0o755
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.chmod(0o000)

        try:
            with self.assertRaises(Exception):
                self.session.create_temp_file("test.txt", "content")
        finally:
            temp_dir.chmod(original_mode)

    def test_read_temp_file_session_error(self):
        """Test SessionError raised on OSError during file read."""
        file_path = self.session.create_temp_file("test.txt", "content")
        file_path.chmod(0o000)

        try:
            with self.assertRaises(Exception):
                self.session.read_temp_file("test.txt")
        finally:
            file_path.chmod(0o644)

    def test_delete_temp_file_session_error(self):
        """Test SessionError raised on OSError during file deletion."""
        file_path = self.session.create_temp_file("test.txt", "content")
        file_path.chmod(0o000)

        try:
            self.session.delete_temp_file("test.txt")
        except Exception:
            pass
        finally:
            if file_path.exists():
                file_path.chmod(0o644)

        assert not file_path.exists(), "File should be deleted"

    def test_temp_file_exceptions_exported(self):
        """Test that TempFileNotFoundError and TempFileError are exported."""
        assert TempFileNotFoundError is not None, TempFileNotFoundError
        assert TempFileError is not None, TempFileError


class TestSessionMessages(unittest.TestCase):
    """Test cases for session message operations."""
    def setUp(self):
        """Set up test fixtures."""
        initialize(True)
        self.session = Session("msg_test", portable=True)

    def tearDown(self):
        """Clean up after tests."""
        hola_dir = Path.cwd() / ".hola"
        if hola_dir.exists():
            shutil.rmtree(hola_dir)

    def test_add_message(self):
        """Test adding a single message."""
        msg = OMessage(role=Role.USER, content="Hello")
        self.session.add_message("granite4.1:3b", msg)

        history = self.session.get_history()
        assert len(history) == 1, len(history)
        assert history[0][0] == "granite4.1:3b", history[0][0]
        assert history[0][1].content == "Hello", history[0][1].content

    def test_add_multiple_messages(self):
        """Test adding multiple messages."""
        msgs = [
            OMessage(role=Role.USER, content="Hello"),
            OMessage(role=Role.ASSISTANT, content="Hi there"),
            OMessage(role=Role.USER, content="How are you?"),
        ]
        for msg in msgs:
            self.session.add_message("granite4.1:3b", msg)

        history = self.session.get_history()
        assert len(history) == 3, len(history)

    def test_add_message_sequence_increment(self):
        """Test that sequence number increments with each message."""
        initial_seq = self.session.sequence
        for i in range(5):
            self.session.add_message(
                "granite4.1:3b", OMessage(role=Role.USER, content=str(i))
            )

        assert self.session.sequence == initial_seq + 5, self.session.sequence

    def test_get_history_empty(self):
        """Test getting history from new session."""
        history = self.session.get_history()
        assert len(history) == 0, len(history)

    def test_get_history_multiple(self):
        """Test getting multiple messages from history."""
        for i in range(10):
            self.session.add_message(
                f"model_{i % 3}",
                OMessage(role=Role.USER, content=f"Message {i}"),
            )

        history = self.session.get_history(limit=10)
        assert len(history) == 10, len(history)

    def test_get_history_limit(self):
        """Test that limit parameter restricts message count."""
        for i in range(20):
            self.session.add_message(
                "granite4.1:3b", OMessage(role=Role.USER, content=str(i))
            )

        history = self.session.get_history(limit=5)
        assert len(history) == 5, len(history)

    def test_get_history_filtered_by_model(self):
        """Test filtering history by model."""
        for i in range(10):
            model = "model_a" if i % 2 == 0 else "model_b"
            self.session.add_message(model, OMessage(role=Role.USER, content=str(i)))

        history_a = self.session.get_history(model="model_a")
        history_b = self.session.get_history(model="model_b")

        assert len(history_a) == 5, len(history_a)
        assert len(history_b) == 5, len(history_b)

    def test_get_history_ordered(self):
        """Test that history is ordered by sequence."""
        for i in range(5):
            self.session.add_message(
                "granite4.1:3b", OMessage(role=Role.USER, content=f"Msg {i}")
            )

        history = self.session.get_history()
        contents = [msg[1].content for msg in history]

        assert contents == ["Msg 0", "Msg 1", "Msg 2", "Msg 3", "Msg 4"], contents

    def test_clear_history(self):
        """Test clearing message history."""
        for i in range(5):
            self.session.add_message(
                "granite4.1:3b", OMessage(role=Role.USER, content=str(i))
            )

        self.session.clear_history()
        history = self.session.get_history()

        assert len(history) == 0, len(history)

    def test_clear_history_resets_sequence(self):
        """Test that clearing history resets sequence."""
        for i in range(5):
            self.session.add_message(
                "granite4.1:3b", OMessage(role=Role.USER, content=str(i))
            )

        self.session.clear_history()
        assert self.session.sequence == 0, self.session.sequence


class TestSessionToolOutputs(unittest.TestCase):
    """Test cases for session tool output storage."""
    def setUp(self):
        """Set up test fixtures."""
        initialize(True)
        self.session = Session("tool_test", portable=True)

    def tearDown(self):
        """Clean up after tests."""
        hola_dir = Path.cwd() / ".hola"
        if hola_dir.exists():
            shutil.rmtree(hola_dir)

    def test_add_tool_output(self):
        """Test storing tool output."""
        call_id = "call_123"
        result = {"status": "success", "data": [1, 2, 3]}
        self.session.add_tool_output(call_id, result)

        retrieved = self.session.get_tool_output(call_id)
        assert retrieved == result, retrieved

    def test_add_multiple_tool_outputs(self):
        """Test storing multiple tool outputs."""
        outputs = {
            "call_1": {"result": 1},
            "call_2": {"result": 2},
            "call_3": {"result": 3},
        }
        for call_id, result in outputs.items():
            self.session.add_tool_output(call_id, result)

        all_outputs = self.session.list_tool_outputs()
        assert len(all_outputs) == 3, len(all_outputs)

    def test_get_tool_output_success(self):
        """Test retrieving stored tool output."""
        call_id = "call_456"
        result = {"output": "test data"}
        self.session.add_tool_output(call_id, result)

        retrieved = self.session.get_tool_output(call_id)
        assert retrieved == result, retrieved

    def test_get_tool_output_not_found(self):
        """Test retrieving non-existent tool output."""
        result = self.session.get_tool_output("nonexistent_call")
        assert result is None, result

    def test_list_tool_outputs(self):
        """Test listing all tool outputs."""
        call_ids = ["call_a", "call_b", "call_c"]
        for i, call_id in enumerate(call_ids):
            self.session.add_tool_output(call_id, {"value": i})

        outputs = self.session.list_tool_outputs()
        assert len(outputs) == 3, len(outputs)
        for call_id in call_ids:
            assert call_id in outputs, call_id

    def test_list_tool_outputs_empty(self):
        """Test listing tool outputs when none exist."""
        outputs = self.session.list_tool_outputs()
        assert len(outputs) == 0, len(outputs)


class TestSessionMetadata(unittest.TestCase):
    """Test cases for session metadata operations."""
    def setUp(self):
        """Set up test fixtures."""
        initialize(True)
        self.session = Session("meta_test", portable=True)

    def tearDown(self):
        """Clean up after tests."""
        hola_dir = Path.cwd() / ".hola"
        if hola_dir.exists():
            shutil.rmtree(hola_dir)

    def test_set_metadata(self):
        """Test setting single metadata key-value pair."""
        self.session.set_metadata("key1", "value1")
        assert self.session.get_metadata("key1") == "value1", "Value not set"

    def test_set_multiple_metadata(self):
        """Test setting multiple metadata keys."""
        self.session.set_metadata("key1", "value1")
        self.session.set_metadata("key2", "value2")
        self.session.set_metadata("key3", "value3")

        metadata = self.session.list_metadata()
        assert len(metadata) == 3, len(metadata)

    def test_update_metadata(self):
        """Test updating existing metadata key."""
        self.session.set_metadata("key", "original")
        self.session.set_metadata("key", "updated")

        assert self.session.get_metadata("key") == "updated", "Value not updated"

    def test_get_metadata_success(self):
        """Test retrieving metadata value."""
        self.session.set_metadata("test_key", {"nested": "data"})
        value = self.session.get_metadata("test_key")

        assert value == {"nested": "data"}, value

    def test_get_metadata_default(self):
        """Test getting default value for non-existent key."""
        value = self.session.get_metadata("nonexistent", default="default_value")
        assert value == "default_value", value

    def test_list_metadata(self):
        """Test listing all metadata."""
        data = {"key1": "val1", "key2": 123, "key3": [1, 2, 3]}
        for key, value in data.items():
            self.session.set_metadata(key, value)

        metadata = self.session.list_metadata()
        assert len(metadata) == 3, len(metadata)
        for key in data:
            assert key in metadata, key

    def test_list_metadata_empty(self):
        """Test listing metadata when none exist."""
        metadata = self.session.list_metadata()
        assert len(metadata) == 0, len(metadata)


class TestSessionMemory(unittest.TestCase):
    """Test cases for session memory operations."""
    def setUp(self):
        """Set up test fixtures."""
        initialize(True)
        self.session = Session("mem_test", portable=True)

    def tearDown(self):
        """Clean up after tests."""
        hola_dir = Path.cwd() / ".hola"
        if hola_dir.exists():
            shutil.rmtree(hola_dir)

    def test_memorize(self):
        """Test storing a key-value pair in memory."""
        self.session.memorize("name", "Alice")
        assert self.session.recall("name") == "Alice", "Value not memorized"

    def test_memorize_update(self):
        """Test updating an existing memory."""
        self.session.memorize("age", 25)
        self.session.memorize("age", 26)

        assert self.session.recall("age") == 26, "Value not updated"

    def test_recall_success(self):
        """Test retrieving memorized value."""
        self.session.memorize("key", "value")
        value = self.session.recall("key")

        assert value == "value", value

    def test_recall_not_found(self):
        """Test recalling non-existent memory key."""
        value = self.session.recall("nonexistent")
        assert value is None, value

    def test_forget(self):
        """Test deleting a memory key."""
        self.session.memorize("temp", "data")
        self.session.forget("temp")

        assert self.session.recall("temp") is None, "Key not forgotten"

    def test_forget_nonexistent(self):
        """Test forgetting non-existent memory key."""
        self.session.forget("nonexistent")

    def test_list_memory(self):
        """Test listing all memories."""
        memories = {"name": "Bob", "city": "NYC", "age": 30}
        for key, value in memories.items():
            self.session.memorize(key, value)

        listed = self.session.list_memories()
        assert len(listed) == 3, len(listed)
        for key in memories:
            assert key in listed, key

    def test_list_memory_empty(self):
        """Test listing memories when none exist."""
        listed = self.session.list_memories()
        assert len(listed) == 0, len(listed)


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

    def test_fsm_persistence_across_sessions(self):
        """Test FSM state persists correctly across session reload."""
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

        reloaded.state.input(WorkflowEvent.REJECTED)
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
        self.session.state.input(WorkflowEvent.START)
        self.session.state.input(WorkflowEvent.EXECUTE)

        assert self.session.state.current == WorkflowState.EXECUTING, (
            self.session.state.current
        )

        self.session.save_fsm_data()

        reloaded = get_session(self.session.session_id, portable=True)
        reloaded.load_fsm_data(WorkflowFSM, inject={'WorkflowState': WorkflowState})
        assert reloaded.state.current == WorkflowState.EXECUTING, (
            reloaded.state.current
        )


class TestSessionDeletion(unittest.TestCase):
    """Test cases for session deletion and cleanup."""
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path.cwd() / ".hola"

    def tearDown(self):
        """Clean up after tests."""
        hola_dir = Path.cwd() / ".hola"
        if hola_dir.exists():
            shutil.rmtree(hola_dir)

    def test_session_delete_method(self):
        """Test deleting session via session.delete() method."""
        session = Session("del_method_test", portable=True)
        session.delete()

        assert not session_exists("del_method_test", portable=True), (
            "Session still exists"
        )

    def test_session_delete_clears_dbsession(self):
        """Test that session.delete() sets dbsession to None."""
        session = Session("clear_test", portable=True)
        session.delete()

        assert session.dbsession is None, session.dbsession

    def test_delete_session_via_manager(self):
        """Test deleting session via manager function."""
        session = create_session(portable=True)
        session_id = session.session_id
        delete_session(session_id, portable=True)

        assert not session_exists(session_id, portable=True), "Session still exists"

    def test_delete_session_removes_from_db(self):
        """Test that session deletion removes from database."""
        session = create_session(portable=True)
        session_id = session.session_id
        delete_session(session_id, portable=True)

        sessions = list_sessions(portable=True)
        assert session_id not in sessions, "Session in list"

    def test_delete_session_cleans_temp_dir(self):
        """Test that session deletion cleans up temp directory."""
        session_id = "temp_dir_test"
        session = Session(session_id, portable=True)
        session.create_temp_file("test.txt", "content")

        temp_dir = Path.cwd() / ".hola" / "sessions" / session_id / "temp"
        assert temp_dir.exists(), temp_dir

        session.delete()
        assert not temp_dir.exists(), temp_dir

    def test_delete_session_with_temp_files(self):
        """Test that temp files are cleaned up on deletion."""
        session = Session("temp_files_test", portable=True)
        session.create_temp_file("file1.txt", "content1")
        session.create_temp_file("file2.bin", b"bytes")

        session.delete()

        sessions = list_sessions(portable=True)
        assert "temp_files_test" not in sessions, sessions

    def test_delete_session_nonexistent(self):
        """Test deleting non-existent session raises error."""
        with self.assertRaises(SessionNotFoundError):
            delete_session("nonexistent_session", portable=True)


class TestSessionThreadSafety(unittest.TestCase):
    """Test cases for concurrent session operations."""
    def setUp(self):
        """Set up test fixtures."""
        initialize(True)
        self.session = Session("thread_test", portable=True)

    def tearDown(self):
        """Clean up after tests."""
        hola_dir = Path.cwd() / ".hola"
        if hola_dir.exists():
            shutil.rmtree(hola_dir)

    def test_concurrent_temp_file_creation(self):
        """Test multiple threads creating temp files."""
        threads = []
        errors = []

        def create_file(index):
            try:
                self.session.create_temp_file(
                    f"file_{index}.txt", f"content_{index}"
                )
            except Exception as e:
                errors.append(e)

        for i in range(10):
            thread = threading.Thread(target=create_file, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0, errors
        files = self.session.list_temp_files()
        assert len(files) == 10, len(files)

    def test_concurrent_temp_file_reads(self):
        """Test multiple threads reading temp files."""
        self.session.create_temp_file("read_test.txt", "shared content")
        threads = []
        results = []

        def read_file():
            content = self.session.read_temp_file("read_test.txt")
            results.append(content)

        for _ in range(10):
            thread = threading.Thread(target=read_file)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(results) == 10, len(results)
        assert all(r == b"shared content" for r in results), results

    def test_concurrent_temp_file_deletions(self):
        """Test multiple threads deleting temp files."""
        for i in range(5):
            self.session.create_temp_file(f"del_{i}.txt", f"content_{i}")

        threads = []
        errors = []

        def delete_file(index):
            try:
                self.session.delete_temp_file(f"del_{index}.txt")
            except Exception as e:
                errors.append(e)

        for i in range(5):
            thread = threading.Thread(target=delete_file, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0, errors
        assert len(self.session.list_temp_files()) == 0, "Files not deleted"

    def test_concurrent_message_additions(self):
        """Test multiple threads adding messages."""
        threads = []

        def add_messages(thread_id):
            for i in range(10):
                msg = OMessage(role=Role.USER, content=f"Thread {thread_id}, Msg {i}")
                self.session.add_message("granite4.1:3b", msg)

        for i in range(5):
            thread = threading.Thread(target=add_messages, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        history = self.session.get_history(limit=100)
        assert len(history) == 50, len(history)

    def test_concurrent_memory_operations(self):
        """Test multiple threads using memory operations."""
        threads = []
        errors = []

        def memory_ops(thread_id):
            try:
                for i in range(5):
                    key = f"thread_{thread_id}_key_{i}"
                    value = f"value_{i}"
                    self.session.memorize(key, value)
                    retrieved = self.session.recall(key)
                    assert retrieved == value, retrieved
            except Exception as e:
                errors.append(e)

        for i in range(3):
            thread = threading.Thread(target=memory_ops, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0, errors
        memories = self.session.list_memories()
        assert len(memories) == 15, len(memories)

    def test_session_lock_prevents_race(self):
        """Test that session lock prevents race conditions."""
        counter = [0]
        errors = []

        def increment_counter():
            try:
                self.session.add_message(
                    "granite4.1:3b", OMessage(role=Role.USER, content="test")
                )
                counter[0] += 1
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(10):
            thread = threading.Thread(target=increment_counter)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0, errors
        assert counter[0] == 10, counter[0]
        assert self.session.sequence == 10, self.session.sequence


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

    def test_full_session_lifecycle(self):
        """Test complete session lifecycle: create, use, delete."""
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

        retrieved_session.delete()
        assert not session_exists(session_id, portable=True), "Session exists"

    def test_session_with_messages_and_temp_files(self):
        """Test session with mixed message and temp file operations."""
        initialize(True)
        session = Session("mixed_test", portable=True)

        session.add_message(
            "granite4.1:3b", OMessage(role=Role.USER, content="Hello")
        )
        session.create_temp_file("data.txt", "temp data")

        history = session.get_history()
        files = session.list_temp_files()

        assert len(history) == 1, len(history)
        assert len(files) == 1, len(files)

        session.delete()

    def test_session_persistence_across_instances(self):
        """Test data persists between session instances."""
        initialize(True)
        session = create_session(portable=True)
        session_id = session.session_id
        session1 = get_session(session_id, portable=True)

        session1.memorize("key", "value")
        session1.set_metadata("meta", "data")

        session2 = get_session(session_id, portable=True)
        value = session2.recall("key")
        meta = session2.get_metadata("meta")

        assert value == "value", value
        assert meta == "data", meta

        delete_session(session_id, portable=True)

    def test_multiple_sessions_isolated(self):
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

        session1.delete()
        session2.delete()

    def test_temp_file_path_verification(self):
        """Test that temp files are stored in correct location."""
        initialize(True)
        session = Session("path_test", portable=True)
        expected_dir = Path.cwd() / ".hola" / "sessions" / "path_test" / "temp"

        session.create_temp_file("test.txt", "content")

        assert expected_dir.exists(), expected_dir
        assert (expected_dir / "test.txt").exists(), "File not in expected location"

        session.delete()


if __name__ == "__main__":
    unittest.main()
