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
from hola.session.storage import initialize


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
        client: OllamaClient | None = None, portable: bool = False
    ) -> Session:
    """Create a new session. Raises `SessionAlreadyExistsError` if
        `session_id` conflicts with existing session.
    """
    if session_id:
        sid = session_id
    else:
        sid = _generate_session_id(tag)

    from hola.session.storage import session_exists
    if session_exists(sid, portable):
        raise SessionAlreadyExistsError(f"Session '{sid}' already exists")

    client = client or _get_or_create_default_client()
    return Session(sid, client, portable=portable)


def get_session(
        session_id: str, client: OllamaClient | None = None,
        portable: bool = False
    ) -> Session:
    """Get existing session. Raises `SessionNotFoundError` if
        `session_id` does not exist.
    """
    from hola.session.storage import session_exists
    if not session_exists(session_id, portable):
        raise SessionNotFoundError(f"Session '{session_id}' not found")

    client = client or _get_or_create_default_client()
    return Session(session_id, client, portable=portable)


def list_sessions(portable: bool = False) -> list[str]:
    """List all existing session IDs in the specified mode. Returns
        sessions from portable mode (`CWD/.hola`) or non-portable mode
        (system config).
    """
    from hola.session.storage import list_sessions as storage_list_sessions
    return storage_list_sessions(portable)


def delete_session(session_id: str, portable: bool = False) -> None:
    """Delete session storage from the specified mode. Raises
        `SessionNotFoundError` if session does not exist.
    """
    from hola.session.storage import (
        delete_session as storage_delete_session,
        session_exists
    )
    if not session_exists(session_id, portable):
        raise SessionNotFoundError(f"Session '{session_id}' not found")

    storage_delete_session(session_id, portable)


def session_exists(session_id: str, portable: bool = False) -> bool:
    """Check if session exists in the specified mode. Returns `True` if
        session exists in portable mode (`CWD/.hola`) or non-portable
        mode (system config), `False` otherwise.
    """
    from hola.session.storage import session_exists as storage_session_exists
    return storage_session_exists(session_id, portable)
