"""
Session storage layer for Hola session management.
"""

from pathlib import Path
from secrets import token_hex
from crossconfig import get_config
from packify import pack, unpack
from hola.session.exceptions import (
    SessionDeleteError,
    SessionLoadError,
    SessionSaveError,
)
from hola.session.types import SessionState


def _get_config(portable: bool):
    """Get config instance based on portable mode."""
    app_name = ".hola" if portable else "hola"
    return get_config(app_name, portable=portable)


def _get_storage_path(session_id: str, portable: bool = False) -> Path:
    """Get the storage directory path for a session."""
    return _get_config(portable).path(["sessions", "storage", session_id])


def _get_state_file_path(session_id: str, portable: bool = False) -> Path:
    """Get the state file path for a session."""
    return _get_storage_path(session_id, portable) / "state.pac"


def _get_temp_state_file_path(session_id: str, portable: bool = False) -> Path:
    """Get a temporary state file path for atomic writes."""
    temp_name = f"state.{token_hex(8)}.tmp"
    return _get_storage_path(session_id, portable) / temp_name


def load_session_state(
        session_id: str, portable: bool = False
    ) -> SessionState | None:
    """Load session state from storage. Returns None if not found."""
    state_file = _get_state_file_path(session_id, portable)

    if not state_file.exists():
        return None

    try:
        packed_data = state_file.read_bytes()
        return unpack(packed_data, inject={"SessionState": SessionState})
    except Exception as e:
        raise SessionLoadError(
            f"Failed to load session state for '{session_id}': {e}"
        ) from e


def save_session_state(
        session_id: str, state: SessionState, portable: bool = False
    ) -> None:
    """Save session state to storage."""
    storage_path = _get_storage_path(session_id, portable)
    state_file = _get_state_file_path(session_id, portable)
    temp_file = _get_temp_state_file_path(session_id, portable)

    try:
        storage_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise SessionSaveError(
            f"Failed to create storage directory for '{session_id}': {e}"
        ) from e

    try:
        packed_data = pack(state)
        temp_file.write_bytes(packed_data)
        temp_file.replace(state_file)
    except Exception as e:
        if temp_file.exists():
            temp_file.unlink()
        raise SessionSaveError(
            f"Failed to save session state for '{session_id}': {e}"
        ) from e


def delete_session(session_id: str, portable: bool = False) -> None:
    """Delete session storage."""
    storage_path = _get_storage_path(session_id, portable)

    if not storage_path.exists():
        raise SessionDeleteError(
            f"Session storage for '{session_id}' does not exist"
        )

    try:
        for item in storage_path.iterdir():
            if item.is_file():
                item.unlink()
        storage_path.rmdir()
    except OSError as e:
        raise SessionDeleteError(
            f"Failed to delete session storage for '{session_id}': {e}"
        ) from e


def list_sessions(portable: bool = False) -> list[str]:
    """List all existing session IDs."""
    storage_root = _get_config(portable).path(["sessions", "storage"])

    if not storage_root.exists():
        return []

    try:
        return [d.name for d in storage_root.iterdir() if d.is_dir()]
    except OSError:
        return []


def session_exists(session_id: str, portable: bool = False) -> bool:
    """Check if session exists."""
    state_file = _get_state_file_path(session_id, portable)
    return state_file.exists()
