"""
Database models for session management using sqloquent ORM.
"""

from .Memory import Memory
from .Message import MType, Message
from .Session import Session
from pathlib import Path
from sqloquent import has_many, belongs_to
from typing import Callable
import sqloquent.tools


# set relations
Memory.session = belongs_to(Memory, Session, 'session_id')
Message.session = belongs_to(Message, Session, 'session_id')
Session.memories = has_many(Session, Memory, 'session_id')
Session.messages = has_many(Session, Message, 'session_id')


# helper functions
def set_connection_string(connection_string: str) -> None:
    """Sets the connection info for Hola session db models."""
    Memory.connection_info = connection_string
    Message.connection_info = connection_string
    Session.connection_info = connection_string

def publish_migrations(
        migration_folder_path: Path,
        migration_callback: Callable[[str, str], str] | None = None
    ) -> None:
    """Writes migration files for the models. If a migration callback is
        provided, it will be used to modify the migration file contents.
        The migration callback will be called with the model name and
        the migration file contents, and whatever it returns will be
        used as the migration file contents.
    """
    migrations = {
        'Memory': sqloquent.tools.make_migration_from_model(Memory),
        'Message': sqloquent.tools.make_migration_from_model(Message),
        'Session': sqloquent.tools.make_migration_from_model(Session),
    }
    for name, m in migrations.items():
        m2 = migration_callback(name, m) if migration_callback else m
        m = m2 if type(m2) is str and len(m2) > 0 else m
        with open(migration_folder_path / f'00_create_{name}.py', 'w') as f:
            f.write(m)

def automigrate(migration_folder_path: str, db_file_path: str) -> None:
    """Executes the sqloquent automigrate tool."""
    sqloquent.tools.automigrate(migration_folder_path, db_file_path)

# avoid exporting unrelated things
del Callable, Path
