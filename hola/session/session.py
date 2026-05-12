"""
Session class for Hola session management.
"""

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from hola.ollama.client import OllamaClient
from hola.ollama.types import Message as OMessage, Role
from hola.session.exceptions import (
    SessionDeleteError,
    SessionError,
    TempFileNotFoundError,
)
from hola.session.models import (
    Message as DBMessage,
    Memory,
    Session as DBSession,
)
from hola.session.storage import _get_session_temp_dir, initialize
import packify
import os


@dataclass(slots=True)
class Session:
    """Agent session with state management and persistence."""
    session_id: str
    client: OllamaClient
    portable: bool
    lock: Lock = field(default_factory=Lock)
    dbsession: DBSession = field(default=None)
    sequence: int = field(default=0)

    def __post_init__(self) -> None:
        initialize(self.portable)

        self.dbsession = DBSession.find(self.session_id) or DBSession.insert({
            'id': self.session_id,
            'cwd': os.getcwd(),
        })
        last_m_list = self.dbsession.messages().query().order_by(
            'sequence', 'desc'
        ).select(['sequence']).take(1)
        if last_m_list:
            last_m = last_m_list[0]
            if last_m.sequence:
                self.sequence = last_m.sequence + 1

    def add_message(self, model: str, message: OMessage) -> None:
        """Add a message to conversation history."""
        with self.lock:
            self.sequence += 1
            m = DBMessage({
                'session_id': self.session_id,
                'sequence': self.sequence,
                'role': message.role.value,
                'model': model,
                'content': message.content,
            })
            m.details = {
                'tool_name': message.tool_name,
                'tool_calls': message.tool_calls,
            }
            m.save()

    def get_history(
            self, model: str | None = None, limit: int = 10,
        ) -> list[tuple[str, OMessage]]:
        """Get conversation history, optionally filtered by model."""
        with self.lock:
            sqb = self.dbsession.messages().query().order_by('sequence', 'desc')
            if model is not None:
                sqb.equal('model', model)

            msgs = [
                (m.model, OMessage(m.role.value, m.content))
                for m in sqb.take(limit)
            ]
            msgs.reverse()
            return msgs

    def clear_history(self) -> None:
        """Clear conversation history."""
        with self.lock:
            sqb = self.dbsession.messages().query()
            sqb.delete()
            self.sequence = 0

    def add_tool_output(self, call_id: str, result: object) -> None:
        """Set tool output by call ID. Stored as Message with Role.TOOL."""
        with self.lock:
            m = DBMessage({
                'session_id': self.session_id,
                'role': Role.TOOL.value,
                'content': '',
                'call_id': call_id,
            })
            m.details = {'result': result}
            m.save()

    def get_tool_output(self, call_id: str) -> object | None:
        """Get tool output by call ID."""
        with self.lock:
            sqb = self.dbsession.messages().query().equal('call_id', call_id)
            call = sqb.first()
            if call:
                return call.details.get('result')
            return None

    def list_tool_outputs(self) -> dict[str, object]:
        """List all tool outputs."""
        with self.lock:
            outputs = {}
            sqb = self.dbsession.messages().query().equal('role', Role.TOOL.value)
            for m in sqb.get():
                call_id = m.call_id
                result = m.details.get('result')
                outputs[call_id] = result

            return outputs

    def get_fsm_data(self) -> bytes:
        """Get the bytes serialized FSM data to be deserialized by the
            unpack method of the FSM subclass.
        """
        return self.dbsession.state

    def set_fsm_data(self, data: bytes) -> None:
        """Set the serialized FSM bytes returned by the pack method of
            the FSM subclass.
        """
        self.dbsession.state = data
        self.dbsession.save()

    def set_metadata(self, key: str, value: object) -> None:
        """Set metadata key-value pair."""
        with self.lock:
            metadata = self.dbsession.metadata
            metadata[key] = value
            self.dbsession.metadata = metadata
            self.dbsession.save()

    def get_metadata(
            self, key: str, default: object = None
        ) -> object:
        """Get metadata value by key."""
        with self.lock:
            return self.dbsession.metadata.get(key, default)

    def list_metadata(self) -> dict[str, object]:
        """List all metadata."""
        with self.lock:
            return self.dbsession.metadata

    def memorize(self, key: str, value: object) -> None:
        """Memorize a key-value pair."""
        with self.lock:
            memory = self.dbsession.memories().query().equal('name', key).first()

            if memory:
                memory.content = value
                memory.save()
            else:
                memory = Memory({
                    'id': Memory.generate_id(),
                    'session_id': self.session_id,
                    'name': key,
                })
                memory.content = value
                memory.save()

    def recall(self, key: str) -> object:
        """Recall a memorized value by key."""
        with self.lock:
            memory = self.dbsession.memories().query().equal('name', key).first()
            if memory:
                return memory.content
            return None

    def forget(self, key: str) -> None:
        """Forget a memorized key-value pair."""
        with self.lock:
            memory = self.dbsession.memories().query().equal('name', key).first()
            if memory:
                memory.delete()

    def list_memories(self) -> dict[str, object]:
        """List all memorized key-value pairs."""
        with self.lock:
            return {
                mem.name: mem.content
                for mem in self.dbsession.memories().query().get()
            }

    def create_temp_file(
            self, filename: str, content: bytes | str
        ) -> Path:
        """Create a temporary file in the session's temp directory.
            Returns the full path to the created file.
        """
        with self.lock:
            temp_dir = _get_session_temp_dir(self.session_id, self.portable)
            temp_dir.mkdir(parents=True, exist_ok=True)

            file_path = temp_dir / filename
            try:
                if isinstance(content, str):
                    file_path.write_text(content)
                else:
                    file_path.write_bytes(content)
            except OSError as e:
                raise SessionError(f"Failed to create temp file: {e}")

            return file_path

    def read_temp_file(self, filename: str) -> bytes:
        """Read a temporary file as bytes."""
        with self.lock:
            temp_dir = _get_session_temp_dir(self.session_id, self.portable)
            file_path = temp_dir / filename

            if not file_path.exists():
                raise TempFileNotFoundError(f"Temp file not found: {filename}")

            try:
                return file_path.read_bytes()
            except OSError as e:
                raise SessionError(f"Failed to read temp file: {e}")

    def delete_temp_file(self, filename: str) -> None:
        """Delete a temporary file."""
        with self.lock:
            temp_dir = _get_session_temp_dir(self.session_id, self.portable)
            file_path = temp_dir / filename

            if not file_path.exists():
                raise TempFileNotFoundError(f"Temp file not found: {filename}")

            try:
                file_path.unlink()
            except OSError as e:
                raise SessionError(f"Failed to delete temp file: {e}")

    def list_temp_files(self) -> list[str]:
        """List all temporary files in the session's temp directory."""
        with self.lock:
            temp_dir = _get_session_temp_dir(self.session_id, self.portable)

            if not temp_dir.exists():
                return []

            return [f.name for f in temp_dir.iterdir() if f.is_file()]

    def delete(self) -> None:
        """Delete session and its storage."""
        with self.lock:
            from hola.session.storage import delete_session
            delete_session(self.session_id, self.portable)
            self.dbsession = None
