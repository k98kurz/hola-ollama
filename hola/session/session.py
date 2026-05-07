"""
Session class for Hola session management.
"""

from threading import Lock
from hola.ollama.client import OllamaClient
from hola.ollama.types import Message
from hola.session.exceptions import (
    SessionDeleteError,
    SessionLoadError,
    SessionSaveError,
)
from hola.session.storage import (
    delete_session,
    load_session_state,
    save_session_state,
    session_exists,
)
from hola.session.types import SessionState


class Session:
    """Agent session with state management and persistence."""

    def __init__(
            self, session_id: str, client: OllamaClient,
            auto_load: bool = True, auto_save: bool = True,
            portable: bool = False
        ) -> None:
        """Initialize session."""
        self._session_id = session_id
        self._client = client
        self._auto_load = auto_load
        self._auto_save = auto_save
        self._portable = portable
        self._state = SessionState()
        self._lock = Lock()

        if auto_load:
            self.load()

    @property
    def session_id(self) -> str:
        """Get the session ID."""
        return self._session_id

    @property
    def client(self) -> OllamaClient:
        """Get the OllamaClient."""
        return self._client

    def add_message(self, model: str, message: Message) -> None:
        """Add a message to conversation history."""
        with self._lock:
            self._state.conversation_history.append((model, message))
            if self._auto_save:
                self._save()

    def get_history(
            self, model: str | None = None
        ) -> list[tuple[str, Message]]:
        """Get conversation history, optionally filtered by model."""
        with self._lock:
            if model is None:
                return list(self._state.conversation_history)
            return [
                (m, msg) for m, msg in self._state.conversation_history
                if m == model
            ]

    def clear_history(self, model: str | None = None) -> None:
        """Clear conversation history, optionally filtered by model."""
        with self._lock:
            if model is None:
                self._state.conversation_history.clear()
            else:
                self._state.conversation_history = [
                    (m, msg) for m, msg in self._state.conversation_history
                    if m != model
                ]
            if self._auto_save:
                self._save()

    def set_tool_output(self, call_id: str, result: object) -> None:
        """Set tool output by call ID."""
        with self._lock:
            self._state.tool_outputs[call_id] = result
            if self._auto_save:
                self._save()

    def get_tool_output(self, call_id: str) -> object:
        """Get tool output by call ID."""
        with self._lock:
            return self._state.tool_outputs.get(call_id)

    def list_tool_outputs(self) -> dict[str, object]:
        """List all tool outputs."""
        with self._lock:
            return dict(self._state.tool_outputs)

    def set_fsm_state(self, fsm_name: str, state: dict[str, object]) -> None:
        """Set FSM state by name."""
        with self._lock:
            self._state.fsm_state[fsm_name] = state
            if self._auto_save:
                self._save()

    def get_fsm_state(self, fsm_name: str) -> dict[str, object] | None:
        """Get FSM state by name."""
        with self._lock:
            return self._state.fsm_state.get(fsm_name)

    def clear_fsm_state(self, fsm_name: str) -> None:
        """Clear FSM state by name."""
        with self._lock:
            self._state.fsm_state.pop(fsm_name, None)
            if self._auto_save:
                self._save()

    def set_metadata(self, key: str, value: object) -> None:
        """Set metadata key-value pair."""
        with self._lock:
            self._state.metadata[key] = value
            if self._auto_save:
                self._save()

    def get_metadata(self, key: str, default: object = None) -> object:
        """Get metadata value by key."""
        with self._lock:
            return self._state.metadata.get(key, default)

    def list_metadata(self) -> dict[str, object]:
        """List all metadata."""
        with self._lock:
            return dict(self._state.metadata)

    def memorize(self, key: str, value: object) -> None:
        """Memorize a key-value pair."""
        with self._lock:
            self._state.memory[key] = value
            if self._auto_save:
                self._save()

    def recall(self, key: str) -> object:
        """Recall a memorized value by key."""
        with self._lock:
            return self._state.memory.get(key)

    def forget(self, key: str) -> None:
        """Forget a memorized key-value pair."""
        with self._lock:
            self._state.memory.pop(key, None)
            if self._auto_save:
                self._save()

    def list_memory(self) -> dict[str, object]:
        """List all memorized key-value pairs."""
        with self._lock:
            return dict(self._state.memory)

    def load(self) -> None:
        """Load session state from storage."""
        if not session_exists(self._session_id, self._portable):
            return

        loaded_state = load_session_state(self._session_id, self._portable)
        if loaded_state is not None:
            with self._lock:
                self._state = loaded_state

    def save(self) -> None:
        """Save session state to storage."""
        with self._lock:
            try:
                save_session_state(
                    self._session_id, self._state, self._portable
                )
            except SessionSaveError as e:
                raise SessionSaveError(
                    f"Failed to save session '{self._session_id}': {e}"
                ) from e

    def _save(self) -> None:
        """Internal save method without acquiring lock."""
        try:
            save_session_state(self._session_id, self._state, self._portable)
        except SessionSaveError as e:
            raise SessionSaveError(
                f"Failed to save session '{self._session_id}': {e}"
            ) from e

    def delete(self) -> None:
        """Delete session and its storage."""
        with self._lock:
            try:
                delete_session(self._session_id, self._portable)
                self._state = SessionState()
            except SessionDeleteError as e:
                raise SessionDeleteError(
                    f"Failed to delete session '{self._session_id}': {e}"
                ) from e
