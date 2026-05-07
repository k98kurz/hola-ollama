"""
Session factory functions for Hola session management.
"""

from datetime import datetime
from secrets import token_hex
from hola.ollama.client import OllamaClient
from hola.session.exceptions import (
    SessionAlreadyExistsError,
    SessionNotFoundError,
)
from hola.session.session import Session
from hola.session.storage import (
    delete_session as delete_session_storage,
    list_sessions as storage_list_sessions,
    session_exists as storage_session_exists,
)


_default_client: OllamaClient | None = None


def _generate_session_id(tag: str | None = None) -> str:
    """Generate session ID in format:
        `{YYYY-MM-DD}-{HH:mm:ss}-{hex}-{tag?}`.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    hex_token = token_hex(4)

    if tag:
        sanitized_tag = tag.replace(" ", "_")[:8]
        return f"{timestamp}-{hex_token}-{sanitized_tag}"

    return f"{timestamp}-{hex_token}"


def _get_or_create_default_client() -> OllamaClient:
    """Get or create default shared OllamaClient."""
    global _default_client
    if _default_client is None:
        _default_client = OllamaClient()
    return _default_client


def create_session(
        tag: str | None = None, session_id: str | None = None,
        client: OllamaClient | None = None, auto_load: bool = True,
        auto_save: bool = True, portable: bool = False
    ) -> Session:
    """Create a new session. Raises `SessionAlreadyExistsError` if
        `session_id` conflicts with existing session.
    """
    if session_id:
        sid = session_id
    else:
        sid = _generate_session_id(tag)

    if storage_session_exists(sid, portable):
        raise SessionAlreadyExistsError(f"Session '{sid}' already exists")

    client = client or _get_or_create_default_client()
    session = Session(
        sid, client, auto_load=False,
        auto_save=auto_save, portable=portable
    )
    session.save()
    session._auto_load = auto_load
    return session


def get_session(
        session_id: str, client: OllamaClient | None = None,
        auto_load: bool = True, auto_save: bool = True,
        portable: bool = False
    ) -> Session:
    """Get existing session. Raises `SessionNotFoundError` if
        `session_id` does not exist.
    """
    if not storage_session_exists(session_id, portable):
        raise SessionNotFoundError(f"Session '{session_id}' not found")

    client = client or _get_or_create_default_client()
    return Session(
        session_id, client, auto_load=auto_load,
        auto_save=auto_save, portable=portable
    )


def list_sessions(portable: bool = False) -> list[str]:
    """List all existing session IDs in the specified mode. Returns
        sessions from portable mode (`CWD/.hola`) or non-portable mode
        (system config).
    """
    return storage_list_sessions(portable)


def delete_session(session_id: str, portable: bool = False) -> None:
    """Delete session storage from the specified mode. Raises
        `SessionNotFoundError` if session does not exist.
    """
    if not storage_session_exists(session_id, portable):
        raise SessionNotFoundError(f"Session '{session_id}' not found")

    delete_session_storage(session_id, portable)


def session_exists(session_id: str, portable: bool = False) -> bool:
    """Check if session exists in the specified mode. Returns `True` if
        session exists in portable mode (`CWD/.hola`) or non-portable
        mode (system config), `False` otherwise.
    """
    return storage_session_exists(session_id, portable)
