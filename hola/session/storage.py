"""
Session storage layer for Hola session management.
"""

from pathlib import Path
from crossconfig import get_config


_db_initialized: dict[bool, bool] = {False: False, True: False}


def _get_config(portable: bool):
    """Get config instance based on portable mode."""
    app_name = ".hola" if portable else "hola"
    return get_config(app_name, portable=portable)


def _get_db_path(portable: bool = False) -> Path:
    """Get database file path based on mode."""
    config = _get_config(portable)
    db_name = "sessions.db"
    return config.path(db_name)


def initialize(portable: bool = False) -> None:
    """Initialize database for session management. Call once per process.
        If True, use portable mode (CWD/.hola/sessions.db). If False,
        use system config (~/.config/hola/sessions.db).
    """
    if _db_initialized[portable]:
        return

    from hola.session.models import (
        set_connection_string,
        publish_migrations,
        automigrate
    )

    db_path = str(_get_db_path(portable))
    set_connection_string(db_path)

    migrations_dir = _get_config(portable).path("migrations")
    migrations_dir.mkdir(parents=True, exist_ok=True)

    publish_migrations(migrations_dir)
    automigrate(str(migrations_dir), db_path)

    _db_initialized[portable] = True


def delete_session(session_id: str, portable: bool = False) -> None:
    """Delete session from database."""
    initialize(portable)

    from hola.session.models import Session

    db_session = Session.find(session_id)
    if db_session is None:
        from hola.session.exceptions import SessionNotFoundError
        raise SessionNotFoundError(f"Session '{session_id}' not found")

    db_session.delete()


def list_sessions(portable: bool = False) -> list[str]:
    """List all session IDs."""
    initialize(portable)

    from hola.session.models import Session

    sessions = Session.query().get()
    return [s.id for s in sessions]


def session_exists(session_id: str, portable: bool = False) -> bool:
    """Check if session exists."""
    initialize(portable)

    from hola.session.models import Session

    return Session.find(session_id) is not None
